// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Two-pass encoding planning and configuration layer.
//!
//! This module provides the configuration, statistics analysis, and planning
//! layer for two-pass encoding. It complements the lower-level
//! pipeline executor by adding:
//!
//! - **`TwoPassPlan`**: a complete plan describing how to execute a two-pass
//!   encode (pass 1 analysis settings, pass 2 encoding settings, temp file
//!   management).
//! - **`FirstPassAnalysis`**: richer statistics from the first pass including
//!   scene-level complexity histogram, motion vectors, and GOP recommendations.
//! - **`SecondPassStrategy`**: bitrate distribution strategy for the second pass
//!   derived from first-pass analysis.
//! - **`TwoPassPlanner`**: builder that creates a `TwoPassPlan` from input
//!   properties and quality targets.

use crate::formats::{ContainerFormat, VideoCodec};
use crate::pipeline::BitrateMode;
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

// ── Quality target ─────────────────────────────────────────────────────────

/// Quality target for two-pass encoding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QualityTarget {
    /// Target a specific average bitrate (bits/s).
    AverageBitrate(u64),
    /// Target a specific file size in bytes.
    FileSize(u64),
    /// Target a PSNR value (dB).
    Psnr(f64),
    /// Target a SSIM value (0.0 - 1.0).
    Ssim(f64),
    /// Target a VMAF score (0 - 100).
    Vmaf(f64),
}

impl QualityTarget {
    /// Estimate the target bitrate for a given duration.
    ///
    /// For bitrate targets this is trivial; for file-size targets it divides
    /// by duration; for perceptual metrics it uses empirical scaling.
    #[must_use]
    pub fn estimated_bitrate(&self, duration_seconds: f64) -> u64 {
        match *self {
            Self::AverageBitrate(bps) => bps,
            Self::FileSize(bytes) => {
                if duration_seconds > 0.0 {
                    ((bytes as f64 * 8.0) / duration_seconds) as u64
                } else {
                    4_000_000 // default 4 Mbps
                }
            }
            Self::Psnr(db) => {
                // Empirical: PSNR 40 dB ≈ 4 Mbps at 1080p30
                // Scale quadratically around that anchor
                let ratio = db / 40.0;
                (4_000_000.0 * ratio * ratio) as u64
            }
            Self::Ssim(ssim) => {
                // SSIM 0.95 ≈ 4 Mbps at 1080p30
                let ratio = ssim / 0.95;
                (4_000_000.0 * ratio * ratio) as u64
            }
            Self::Vmaf(vmaf) => {
                // VMAF 90 ≈ 4 Mbps at 1080p30
                let ratio = vmaf / 90.0;
                (4_000_000.0 * ratio * ratio) as u64
            }
        }
    }

    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        match *self {
            Self::AverageBitrate(bps) => format!("{:.1} Mbps average", bps as f64 / 1_000_000.0),
            Self::FileSize(bytes) => format!("{:.1} MB target", bytes as f64 / (1024.0 * 1024.0)),
            Self::Psnr(db) => format!("{db:.1} dB PSNR target"),
            Self::Ssim(ssim) => format!("{ssim:.4} SSIM target"),
            Self::Vmaf(vmaf) => format!("{vmaf:.1} VMAF target"),
        }
    }
}

// ── First pass analysis ────────────────────────────────────────────────────

/// Scene-level complexity information gathered during the first pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneInfo {
    /// Frame index where this scene starts.
    pub start_frame: u64,
    /// Frame index where this scene ends (exclusive).
    pub end_frame: u64,
    /// Average complexity for this scene (0.0 - 1.0).
    pub avg_complexity: f64,
    /// Peak complexity within this scene.
    pub peak_complexity: f64,
    /// Average temporal complexity (motion between frames).
    pub temporal_complexity: f64,
    /// Whether this scene is a scene change boundary.
    pub is_scene_change: bool,
}

impl SceneInfo {
    /// Number of frames in this scene.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Recommended quantizer offset relative to the global CRF.
    ///
    /// Negative values = more bits (complex scenes), positive = fewer bits.
    #[must_use]
    pub fn quantizer_offset(&self, global_avg_complexity: f64) -> f64 {
        if global_avg_complexity <= 0.0 {
            return 0.0;
        }
        // Log-scale offset: complex scenes get lower quantizer
        let ratio = self.avg_complexity / global_avg_complexity;
        if ratio <= 0.0 {
            return 2.0; // very simple, save bits
        }
        -4.0 * ratio.ln()
    }
}

/// Rich analysis from the first pass of a two-pass encode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstPassAnalysis {
    /// Total frames analyzed.
    pub total_frames: u64,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Global average spatial complexity (0.0 - 1.0).
    pub avg_spatial_complexity: f64,
    /// Global average temporal complexity (0.0 - 1.0).
    pub avg_temporal_complexity: f64,
    /// Peak spatial complexity.
    pub peak_spatial_complexity: f64,
    /// Peak temporal complexity.
    pub peak_temporal_complexity: f64,
    /// Per-scene breakdown.
    pub scenes: Vec<SceneInfo>,
    /// Number of detected scene changes.
    pub scene_change_count: u32,
    /// Recommended GOP (Group of Pictures) size in frames.
    pub recommended_gop_size: u32,
    /// Recommended minimum keyframe interval.
    pub recommended_min_keyint: u32,
    /// Complexity histogram (10 buckets, each counting frames in that range).
    pub complexity_histogram: [u64; 10],
    /// How long the analysis took.
    pub analysis_duration: Duration,
}

impl FirstPassAnalysis {
    /// Compute the complexity distribution variance.
    ///
    /// High variance means the content has very different complexity levels
    /// across scenes, which benefits most from two-pass encoding.
    #[must_use]
    pub fn complexity_variance(&self) -> f64 {
        if self.scenes.is_empty() {
            return 0.0;
        }
        let mean = self.avg_spatial_complexity;
        let sum_sq: f64 = self
            .scenes
            .iter()
            .map(|s| {
                let diff = s.avg_complexity - mean;
                diff * diff * s.frame_count() as f64
            })
            .sum();
        sum_sq / self.total_frames.max(1) as f64
    }

    /// Whether two-pass encoding provides significant benefit for this content.
    ///
    /// Two-pass is most beneficial when complexity variance is high (> 0.01),
    /// indicating very different scene complexities that benefit from
    /// differential bit allocation.
    #[must_use]
    pub fn two_pass_benefit(&self) -> TwoPassBenefit {
        let variance = self.complexity_variance();
        if variance > 0.05 {
            TwoPassBenefit::High
        } else if variance > 0.01 {
            TwoPassBenefit::Moderate
        } else {
            TwoPassBenefit::Low
        }
    }

    /// Estimate the total bits needed to achieve the given quality target.
    #[must_use]
    pub fn estimate_total_bits(&self, target: &QualityTarget) -> u64 {
        let bitrate = target.estimated_bitrate(self.duration_seconds);
        (bitrate as f64 * self.duration_seconds) as u64
    }
}

/// How much benefit two-pass encoding provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TwoPassBenefit {
    /// Significant benefit: high complexity variance across scenes.
    High,
    /// Moderate benefit: some variance, worth it for quality-sensitive work.
    Moderate,
    /// Low benefit: uniform content, CRF single-pass is nearly as good.
    Low,
}

impl TwoPassBenefit {
    /// Human-readable description.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::High => "High: content has widely varying complexity; two-pass will significantly improve quality distribution",
            Self::Moderate => "Moderate: some variation; two-pass provides measurable improvement",
            Self::Low => "Low: uniform content; single-pass CRF is nearly equivalent",
        }
    }
}

// ── Second pass strategy ───────────────────────────────────────────────────

/// Bitrate distribution strategy for the second pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecondPassStrategy {
    /// Global target bitrate (bits/s).
    pub target_bitrate: u64,
    /// Maximum bitrate (VBV ceiling).
    pub max_bitrate: u64,
    /// VBV buffer size in bits.
    pub buffer_size: u64,
    /// Per-scene bitrate allocations.
    pub scene_allocations: Vec<SceneBitrateAllocation>,
    /// Recommended CRF as a starting point.
    pub base_crf: u32,
    /// Whether to use lookahead (and recommended size).
    pub lookahead_frames: u32,
    /// Recommended reference frame count.
    pub ref_frames: u32,
    /// Recommended B-frame count.
    pub b_frames: u32,
}

/// Bitrate allocation for a single scene in the second pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneBitrateAllocation {
    /// Scene start frame.
    pub start_frame: u64,
    /// Scene end frame (exclusive).
    pub end_frame: u64,
    /// Allocated bitrate for this scene (bits/s).
    pub allocated_bitrate: u64,
    /// Quantizer offset relative to base CRF.
    pub quantizer_offset: f64,
    /// Whether to force a keyframe at scene start.
    pub force_keyframe: bool,
}

impl SecondPassStrategy {
    /// Build a strategy from first-pass analysis and a quality target.
    pub fn from_analysis(
        analysis: &FirstPassAnalysis,
        target: &QualityTarget,
        codec: VideoCodec,
    ) -> Result<Self> {
        let target_bitrate = target.estimated_bitrate(analysis.duration_seconds);
        if target_bitrate == 0 {
            return Err(ConversionError::InvalidInput(
                "Target bitrate resolved to zero".to_string(),
            ));
        }

        let max_bitrate = (target_bitrate as f64 * 1.5) as u64;
        let buffer_size = target_bitrate * 2;

        // Base CRF from codec defaults, adjusted by content complexity
        let base_crf = Self::compute_base_crf(codec, analysis);

        // Allocate bits per scene proportionally to complexity
        let global_avg = analysis.avg_spatial_complexity.max(0.001);
        let mut scene_allocations = Vec::with_capacity(analysis.scenes.len());

        for scene in &analysis.scenes {
            let ratio = scene.avg_complexity / global_avg;
            let allocated = (target_bitrate as f64 * ratio.clamp(0.3, 3.0)) as u64;
            let qp_offset = scene.quantizer_offset(global_avg);

            scene_allocations.push(SceneBitrateAllocation {
                start_frame: scene.start_frame,
                end_frame: scene.end_frame,
                allocated_bitrate: allocated,
                quantizer_offset: qp_offset,
                force_keyframe: scene.is_scene_change,
            });
        }

        // Lookahead: longer for AV1 (more complex coding decisions)
        let lookahead_frames = match codec {
            VideoCodec::Av1 => 48,
            VideoCodec::Vp9 => 32,
            VideoCodec::Vp8 => 16,
            VideoCodec::Theora => 8,
        };

        let ref_frames = match codec {
            VideoCodec::Av1 => 7,
            VideoCodec::Vp9 => 5,
            VideoCodec::Vp8 => 3,
            VideoCodec::Theora => 3,
        };

        let b_frames = match codec {
            VideoCodec::Av1 => 3,
            VideoCodec::Vp9 => 2,
            VideoCodec::Vp8 => 0,
            VideoCodec::Theora => 0,
        };

        Ok(Self {
            target_bitrate,
            max_bitrate,
            buffer_size,
            scene_allocations,
            base_crf,
            lookahead_frames,
            ref_frames,
            b_frames,
        })
    }

    fn compute_base_crf(codec: VideoCodec, analysis: &FirstPassAnalysis) -> u32 {
        let default = codec.default_quality();
        let (min_q, max_q) = codec.quality_range();

        // Adjust CRF based on content complexity:
        // More complex content needs lower CRF (more bits)
        let complexity_factor = analysis.avg_spatial_complexity;
        let offset = ((complexity_factor - 0.5) * (max_q as f64 - min_q as f64) * 0.2) as i32;

        let adjusted = (default as i32 - offset).clamp(min_q as i32, max_q as i32);
        adjusted as u32
    }

    /// Total allocated bits across all scenes.
    #[must_use]
    pub fn total_allocated_bits(&self) -> u64 {
        self.scene_allocations
            .iter()
            .map(|s| {
                let frames = s.end_frame.saturating_sub(s.start_frame);
                // Approximate: allocated_bitrate * frames / assumed_fps
                // This is an estimate since we don't store fps here
                s.allocated_bitrate / 30 * frames
            })
            .sum()
    }

    /// Number of forced keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.scene_allocations
            .iter()
            .filter(|s| s.force_keyframe)
            .count()
    }
}

// ── Two-pass plan ──────────────────────────────────────────────────────────

/// Complete plan for a two-pass encode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TwoPassPlan {
    /// Video codec to use.
    pub codec: VideoCodec,
    /// Container format.
    pub container: ContainerFormat,
    /// Quality target.
    pub quality_target: QualityTarget,
    /// First-pass configuration.
    pub first_pass: FirstPassConfig,
    /// Second-pass configuration (derived after first pass completes).
    pub second_pass: SecondPassConfig,
    /// Temporary directory for stats files.
    pub temp_dir: PathBuf,
    /// Estimated total encoding time multiplier (e.g. 2.5 = 2.5x slower than single-pass).
    pub estimated_time_factor: f64,
}

/// Configuration for the first (analysis) pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FirstPassConfig {
    /// Encoding speed for analysis pass (should be fast).
    pub speed: AnalysisSpeed,
    /// Whether to compute motion vectors during analysis.
    pub compute_motion_vectors: bool,
    /// Whether to detect scene changes.
    pub detect_scene_changes: bool,
    /// Analysis segment size in frames.
    pub segment_size: u32,
    /// Path for the stats output file.
    pub stats_file: PathBuf,
}

/// Configuration for the second (encoding) pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecondPassConfig {
    /// Bitrate mode for encoding.
    pub bitrate_mode: BitrateMode,
    /// Maximum bitrate (VBV).
    pub max_bitrate: u64,
    /// VBV buffer size.
    pub buffer_size: u64,
    /// Lookahead frames.
    pub lookahead: u32,
    /// Reference frames.
    pub ref_frames: u32,
    /// B-frames.
    pub b_frames: u32,
    /// Path to the stats file from pass 1.
    pub stats_file: PathBuf,
    /// Whether to apply per-scene quantizer offsets.
    pub use_scene_offsets: bool,
}

/// Speed preset for the analysis pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisSpeed {
    /// Fastest analysis, less accurate statistics.
    Fast,
    /// Balanced analysis speed and accuracy.
    Standard,
    /// Thorough analysis, slower but more accurate.
    Thorough,
}

impl AnalysisSpeed {
    /// Segment size in frames for this speed setting.
    #[must_use]
    pub const fn segment_size(self) -> u32 {
        match self {
            Self::Fast => 120,
            Self::Standard => 60,
            Self::Thorough => 30,
        }
    }
}

// ── Planner ────────────────────────────────────────────────────────────────

/// Builder for creating two-pass encoding plans.
#[derive(Debug, Clone)]
pub struct TwoPassPlanner {
    codec: VideoCodec,
    container: ContainerFormat,
    quality_target: QualityTarget,
    analysis_speed: AnalysisSpeed,
    compute_motion: bool,
    detect_scenes: bool,
    temp_dir: PathBuf,
}

impl TwoPassPlanner {
    /// Create a new planner with defaults.
    pub fn new(codec: VideoCodec, quality_target: QualityTarget) -> Self {
        Self {
            codec,
            container: ContainerFormat::Webm,
            quality_target,
            analysis_speed: AnalysisSpeed::Standard,
            compute_motion: true,
            detect_scenes: true,
            temp_dir: std::env::temp_dir().join("oximedia_two_pass"),
        }
    }

    /// Set the container format.
    #[must_use]
    pub const fn with_container(mut self, container: ContainerFormat) -> Self {
        self.container = container;
        self
    }

    /// Set the analysis speed.
    #[must_use]
    pub const fn with_analysis_speed(mut self, speed: AnalysisSpeed) -> Self {
        self.analysis_speed = speed;
        self
    }

    /// Set whether to compute motion vectors.
    #[must_use]
    pub const fn with_motion_vectors(mut self, compute: bool) -> Self {
        self.compute_motion = compute;
        self
    }

    /// Set whether to detect scene changes.
    #[must_use]
    pub const fn with_scene_detection(mut self, detect: bool) -> Self {
        self.detect_scenes = detect;
        self
    }

    /// Set the temporary directory.
    #[must_use]
    pub fn with_temp_dir(mut self, dir: PathBuf) -> Self {
        self.temp_dir = dir;
        self
    }

    /// Build the two-pass plan.
    pub fn build(&self, duration_seconds: f64) -> Result<TwoPassPlan> {
        if duration_seconds <= 0.0 {
            return Err(ConversionError::InvalidInput(
                "Duration must be positive for two-pass planning".to_string(),
            ));
        }

        let target_bitrate = self.quality_target.estimated_bitrate(duration_seconds);
        if target_bitrate == 0 {
            return Err(ConversionError::InvalidInput(
                "Quality target resolves to zero bitrate".to_string(),
            ));
        }

        let stats_file = self.temp_dir.join(format!(
            "pass1_stats_{}.dat",
            // Use a simple hash of codec + target for unique filename
            self.codec.name()
        ));

        let segment_size = self.analysis_speed.segment_size();

        let first_pass = FirstPassConfig {
            speed: self.analysis_speed,
            compute_motion_vectors: self.compute_motion,
            detect_scene_changes: self.detect_scenes,
            segment_size,
            stats_file: stats_file.clone(),
        };

        // Lookahead / ref frames depend on codec
        let (lookahead, ref_frames, b_frames) = match self.codec {
            VideoCodec::Av1 => (48, 7, 3),
            VideoCodec::Vp9 => (32, 5, 2),
            VideoCodec::Vp8 => (16, 3, 0),
            VideoCodec::Theora => (8, 3, 0),
        };

        let max_bitrate = (target_bitrate as f64 * 1.5) as u64;
        let buffer_size = target_bitrate * 2;

        let second_pass = SecondPassConfig {
            bitrate_mode: BitrateMode::Vbr(target_bitrate),
            max_bitrate,
            buffer_size,
            lookahead,
            ref_frames,
            b_frames,
            stats_file,
            use_scene_offsets: self.detect_scenes,
        };

        // Time factor: AV1 is slowest, VP8 fastest, and two-pass adds ~1.3x
        let codec_factor = match self.codec {
            VideoCodec::Av1 => 3.0,
            VideoCodec::Vp9 => 2.0,
            VideoCodec::Vp8 => 1.2,
            VideoCodec::Theora => 1.5,
        };
        let analysis_factor = match self.analysis_speed {
            AnalysisSpeed::Fast => 1.1,
            AnalysisSpeed::Standard => 1.3,
            AnalysisSpeed::Thorough => 1.5,
        };
        let estimated_time_factor = codec_factor * analysis_factor;

        Ok(TwoPassPlan {
            codec: self.codec,
            container: self.container,
            quality_target: self.quality_target,
            first_pass,
            second_pass,
            temp_dir: self.temp_dir.clone(),
            estimated_time_factor,
        })
    }

    /// Build a plan and immediately derive the second-pass strategy from
    /// the given first-pass analysis.
    pub fn build_with_analysis(
        &self,
        analysis: &FirstPassAnalysis,
    ) -> Result<(TwoPassPlan, SecondPassStrategy)> {
        let plan = self.build(analysis.duration_seconds)?;
        let strategy =
            SecondPassStrategy::from_analysis(analysis, &self.quality_target, self.codec)?;
        Ok((plan, strategy))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_analysis() -> FirstPassAnalysis {
        FirstPassAnalysis {
            total_frames: 9000,
            duration_seconds: 300.0,
            avg_spatial_complexity: 0.4,
            avg_temporal_complexity: 0.3,
            peak_spatial_complexity: 0.9,
            peak_temporal_complexity: 0.8,
            scenes: vec![
                SceneInfo {
                    start_frame: 0,
                    end_frame: 3000,
                    avg_complexity: 0.2,
                    peak_complexity: 0.5,
                    temporal_complexity: 0.1,
                    is_scene_change: true,
                },
                SceneInfo {
                    start_frame: 3000,
                    end_frame: 6000,
                    avg_complexity: 0.6,
                    peak_complexity: 0.9,
                    temporal_complexity: 0.5,
                    is_scene_change: true,
                },
                SceneInfo {
                    start_frame: 6000,
                    end_frame: 9000,
                    avg_complexity: 0.4,
                    peak_complexity: 0.7,
                    temporal_complexity: 0.3,
                    is_scene_change: true,
                },
            ],
            scene_change_count: 3,
            recommended_gop_size: 250,
            recommended_min_keyint: 25,
            complexity_histogram: [100, 200, 500, 1000, 2000, 2500, 1500, 700, 300, 200],
            analysis_duration: Duration::from_secs(15),
        }
    }

    #[test]
    fn test_quality_target_bitrate_estimation() {
        let target = QualityTarget::AverageBitrate(5_000_000);
        assert_eq!(target.estimated_bitrate(300.0), 5_000_000);

        let file_target = QualityTarget::FileSize(100 * 1024 * 1024); // 100 MB
        let bitrate = file_target.estimated_bitrate(300.0);
        // 100 MB * 8 bits / 300 seconds ≈ 2.8 Mbps
        assert!(bitrate > 2_000_000 && bitrate < 3_000_000);

        let psnr_target = QualityTarget::Psnr(40.0);
        assert!(psnr_target.estimated_bitrate(300.0) > 0);
    }

    #[test]
    fn test_quality_target_description() {
        assert!(QualityTarget::AverageBitrate(5_000_000)
            .description()
            .contains("Mbps"));
        assert!(QualityTarget::FileSize(100_000_000)
            .description()
            .contains("MB"));
        assert!(QualityTarget::Psnr(40.0).description().contains("PSNR"));
        assert!(QualityTarget::Ssim(0.95).description().contains("SSIM"));
        assert!(QualityTarget::Vmaf(90.0).description().contains("VMAF"));
    }

    #[test]
    fn test_scene_info_frame_count() {
        let scene = SceneInfo {
            start_frame: 100,
            end_frame: 500,
            avg_complexity: 0.5,
            peak_complexity: 0.8,
            temporal_complexity: 0.3,
            is_scene_change: true,
        };
        assert_eq!(scene.frame_count(), 400);
    }

    #[test]
    fn test_scene_quantizer_offset() {
        let simple_scene = SceneInfo {
            start_frame: 0,
            end_frame: 100,
            avg_complexity: 0.1,
            peak_complexity: 0.2,
            temporal_complexity: 0.05,
            is_scene_change: false,
        };
        let complex_scene = SceneInfo {
            start_frame: 100,
            end_frame: 200,
            avg_complexity: 0.8,
            peak_complexity: 0.95,
            temporal_complexity: 0.7,
            is_scene_change: true,
        };

        let global_avg = 0.4;
        // Simple scene should have positive offset (save bits)
        let simple_offset = simple_scene.quantizer_offset(global_avg);
        // Complex scene should have negative offset (spend bits)
        let complex_offset = complex_scene.quantizer_offset(global_avg);
        assert!(
            simple_offset > complex_offset,
            "Simple scenes should have higher QP offset than complex"
        );
    }

    #[test]
    fn test_first_pass_analysis_variance() {
        let analysis = make_analysis();
        let variance = analysis.complexity_variance();
        assert!(
            variance > 0.0,
            "Variance should be positive for varied content"
        );
    }

    #[test]
    fn test_two_pass_benefit() {
        let mut analysis = make_analysis();
        let benefit = analysis.two_pass_benefit();
        // With scenes at 0.2, 0.6, 0.4 complexity, there's meaningful variance
        assert_ne!(benefit, TwoPassBenefit::Low);

        // Uniform content: all scenes at same complexity
        for scene in &mut analysis.scenes {
            scene.avg_complexity = 0.4;
        }
        let benefit_uniform = analysis.two_pass_benefit();
        assert_eq!(benefit_uniform, TwoPassBenefit::Low);
    }

    #[test]
    fn test_second_pass_strategy_from_analysis() {
        let analysis = make_analysis();
        let target = QualityTarget::AverageBitrate(4_000_000);
        let strategy = SecondPassStrategy::from_analysis(&analysis, &target, VideoCodec::Vp9);
        assert!(strategy.is_ok());
        let s = strategy.expect("should succeed");

        assert_eq!(s.target_bitrate, 4_000_000);
        assert!(s.max_bitrate > s.target_bitrate);
        assert_eq!(s.scene_allocations.len(), 3);
        assert!(s.lookahead_frames > 0);

        // Complex scene (scene[1]) should get more bits
        let simple_alloc = s.scene_allocations[0].allocated_bitrate;
        let complex_alloc = s.scene_allocations[1].allocated_bitrate;
        assert!(
            complex_alloc > simple_alloc,
            "Complex scene should get more bits"
        );
    }

    #[test]
    fn test_second_pass_keyframe_count() {
        let analysis = make_analysis();
        let target = QualityTarget::AverageBitrate(4_000_000);
        let strategy = SecondPassStrategy::from_analysis(&analysis, &target, VideoCodec::Av1)
            .expect("should succeed");
        assert_eq!(strategy.keyframe_count(), 3);
    }

    #[test]
    fn test_two_pass_planner_build() {
        let planner =
            TwoPassPlanner::new(VideoCodec::Vp9, QualityTarget::AverageBitrate(4_000_000));
        let plan = planner.build(300.0);
        assert!(plan.is_ok());
        let p = plan.expect("should succeed");

        assert_eq!(p.codec, VideoCodec::Vp9);
        assert!(p.estimated_time_factor > 1.0);
        assert!(p.first_pass.detect_scene_changes);
        assert!(p.second_pass.use_scene_offsets);
    }

    #[test]
    fn test_planner_zero_duration_fails() {
        let planner =
            TwoPassPlanner::new(VideoCodec::Vp9, QualityTarget::AverageBitrate(4_000_000));
        assert!(planner.build(0.0).is_err());
        assert!(planner.build(-1.0).is_err());
    }

    #[test]
    fn test_planner_with_analysis() {
        let analysis = make_analysis();
        let planner =
            TwoPassPlanner::new(VideoCodec::Av1, QualityTarget::AverageBitrate(8_000_000))
                .with_container(ContainerFormat::Matroska)
                .with_analysis_speed(AnalysisSpeed::Thorough);

        let result = planner.build_with_analysis(&analysis);
        assert!(result.is_ok());
        let (plan, strategy) = result.expect("should succeed");

        assert_eq!(plan.codec, VideoCodec::Av1);
        assert_eq!(plan.container, ContainerFormat::Matroska);
        assert_eq!(strategy.target_bitrate, 8_000_000);
    }

    #[test]
    fn test_analysis_speed_segment_sizes() {
        assert!(AnalysisSpeed::Fast.segment_size() > AnalysisSpeed::Thorough.segment_size());
        assert!(AnalysisSpeed::Standard.segment_size() > AnalysisSpeed::Thorough.segment_size());
    }

    #[test]
    fn test_planner_customization() {
        let planner = TwoPassPlanner::new(VideoCodec::Vp8, QualityTarget::FileSize(50_000_000))
            .with_motion_vectors(false)
            .with_scene_detection(false)
            .with_analysis_speed(AnalysisSpeed::Fast);

        let plan = planner.build(60.0).expect("should succeed");
        assert!(!plan.first_pass.compute_motion_vectors);
        assert!(!plan.first_pass.detect_scene_changes);
        assert!(!plan.second_pass.use_scene_offsets);
        assert_eq!(plan.first_pass.speed, AnalysisSpeed::Fast);
    }

    #[test]
    fn test_av1_higher_time_factor() {
        let av1_plan =
            TwoPassPlanner::new(VideoCodec::Av1, QualityTarget::AverageBitrate(4_000_000))
                .build(300.0)
                .expect("should succeed");
        let vp8_plan =
            TwoPassPlanner::new(VideoCodec::Vp8, QualityTarget::AverageBitrate(4_000_000))
                .build(300.0)
                .expect("should succeed");

        assert!(
            av1_plan.estimated_time_factor > vp8_plan.estimated_time_factor,
            "AV1 should have higher time factor than VP8"
        );
    }

    #[test]
    fn test_file_size_target() {
        // 200 MB target for 5-minute video
        let target = QualityTarget::FileSize(200 * 1024 * 1024);
        let bitrate = target.estimated_bitrate(300.0);
        // Should be roughly 5.6 Mbps
        assert!(bitrate > 4_000_000);
        assert!(bitrate < 7_000_000);
    }

    #[test]
    fn test_two_pass_benefit_descriptions() {
        assert!(!TwoPassBenefit::High.description().is_empty());
        assert!(!TwoPassBenefit::Moderate.description().is_empty());
        assert!(!TwoPassBenefit::Low.description().is_empty());
    }

    #[test]
    fn test_estimate_total_bits() {
        let analysis = make_analysis();
        let target = QualityTarget::AverageBitrate(4_000_000);
        let total = analysis.estimate_total_bits(&target);
        // 4 Mbps * 300s = 1.2 billion bits
        assert_eq!(total, 1_200_000_000);
    }

    #[test]
    fn test_scene_quantizer_offset_zero_global() {
        let scene = SceneInfo {
            start_frame: 0,
            end_frame: 100,
            avg_complexity: 0.5,
            peak_complexity: 0.8,
            temporal_complexity: 0.3,
            is_scene_change: false,
        };
        // Zero global avg should return 0
        assert_eq!(scene.quantizer_offset(0.0), 0.0);
        assert_eq!(scene.quantizer_offset(-1.0), 0.0);
    }
}
