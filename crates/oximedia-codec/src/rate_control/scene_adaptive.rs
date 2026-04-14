//! Scene-adaptive bitrate allocation using content analysis.
//!
//! This module extends the lookahead-based rate control with scene detection
//! and content-type analysis to dynamically allocate bits across scene
//! boundaries and content types (action, talking heads, static, etc.).
//!
//! # Algorithm Overview
//!
//! 1. Classify each lookahead frame by content type using spatial/temporal metrics
//! 2. Detect scene cuts via SAD-based inter-frame difference
//! 3. Compute per-scene complexity budgets from content classification
//! 4. Derive per-frame bit targets adjusted for scene transitions
//!
//! # References
//!
//! - x264/x265 scene-adaptive rate control
//! - Reinhard et al., "Scene change detection and adaptive encoding"

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::error::{CodecError, CodecResult};

// ─────────────────────────────────────────────────────────────────────────────
// Content classification
// ─────────────────────────────────────────────────────────────────────────────

/// Coarse content-type classification used to drive bit allocation.
///
/// Each variant has a different "complexity multiplier" — the fraction of the
/// average bits-per-frame budget to award this frame type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneContentType {
    /// Fast motion (sports, action) — high motion complexity.
    HighMotion,
    /// Mixed scene with moderate motion (drama, news).
    MidMotion,
    /// Mostly-static content (talking heads, slides).
    StaticScene,
    /// Fade-in / fade-out / dissolve transition.
    Transition,
    /// Hard cut — first frame of a new scene.
    SceneCut,
}

impl SceneContentType {
    /// Bit-allocation multiplier relative to the average budget.
    ///
    /// Values >1.0 mean "allocate more bits"; <1.0 mean "save bits here".
    #[must_use]
    pub fn complexity_multiplier(self) -> f32 {
        match self {
            Self::HighMotion => 1.55,
            Self::MidMotion => 1.10,
            Self::StaticScene => 0.65,
            Self::Transition => 0.80,
            Self::SceneCut => 1.40, // Scene-cut I-frame boost
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::HighMotion => "high-motion",
            Self::MidMotion => "mid-motion",
            Self::StaticScene => "static",
            Self::Transition => "transition",
            Self::SceneCut => "scene-cut",
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Frame metrics used by the allocator
// ─────────────────────────────────────────────────────────────────────────────

/// Per-frame content metrics fed to the scene-adaptive allocator.
#[derive(Debug, Clone)]
pub struct FrameContentMetrics {
    /// Frame index in presentation order.
    pub frame_index: u64,
    /// Spatial complexity (0.0 = flat, 1.0 = maximally complex).
    pub spatial_complexity: f32,
    /// Temporal complexity (0.0 = no change, 1.0 = complete scene change).
    pub temporal_complexity: f32,
    /// Average inter-frame SAD normalised to [0, 1].
    pub normalised_sad: f32,
    /// True when a hard scene cut is suspected.
    pub is_scene_cut: bool,
}

impl FrameContentMetrics {
    /// Build metrics from raw pixel statistics.
    ///
    /// # Arguments
    ///
    /// * `frame_index`      – Presentation-order index
    /// * `spatial_var`      – Spatial variance of the luma plane (raw value)
    /// * `inter_frame_sad`  – Sum of absolute differences vs previous frame
    /// * `frame_pixels`     – Total luma pixels in the frame
    ///
    /// `spatial_var` and `inter_frame_sad` are normalised internally so that
    /// a value of `1.0` represents the worst-case / maximum-complexity signal.
    #[must_use]
    pub fn from_raw(
        frame_index: u64,
        spatial_var: f32,
        inter_frame_sad: f64,
        frame_pixels: u32,
    ) -> Self {
        // Normalise spatial variance: typical max ≈ 255² / 4 ≈ 16256.0
        let spatial_complexity = (spatial_var / 16256.0_f32).min(1.0).max(0.0);

        // Normalise inter-frame SAD: max possible = 255 * pixels
        let max_sad = 255.0_f64 * frame_pixels as f64;
        let normalised_sad = if max_sad > 0.0 {
            (inter_frame_sad / max_sad).min(1.0).max(0.0) as f32
        } else {
            0.0
        };

        // Hard scene cut: SAD-based threshold (>15% of pixels fully changed)
        let is_scene_cut = normalised_sad > 0.15;
        let temporal_complexity = normalised_sad;

        Self {
            frame_index,
            spatial_complexity,
            temporal_complexity,
            normalised_sad,
            is_scene_cut,
        }
    }

    /// Classify this frame's content type.
    #[must_use]
    pub fn classify(&self) -> SceneContentType {
        if self.is_scene_cut {
            return SceneContentType::SceneCut;
        }
        // Detect transition: moderate temporal change, lower spatial complexity
        if self.temporal_complexity > 0.06
            && self.temporal_complexity < 0.15
            && self.spatial_complexity < 0.3
        {
            return SceneContentType::Transition;
        }
        // Threshold for high-motion
        if self.temporal_complexity >= 0.15 {
            return SceneContentType::HighMotion;
        }
        // Threshold for mid-motion
        if self.temporal_complexity >= 0.04 {
            return SceneContentType::MidMotion;
        }
        SceneContentType::StaticScene
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A contiguous run of frames that share the same scene.
#[derive(Debug, Clone)]
pub struct Scene {
    /// Index of the first frame in this scene.
    pub start_frame: u64,
    /// Index of the last frame in this scene (inclusive).
    pub end_frame: u64,
    /// Dominant content classification for this scene.
    pub content_type: SceneContentType,
    /// Average spatial complexity across the scene.
    pub avg_spatial: f32,
    /// Average temporal complexity across the scene.
    pub avg_temporal: f32,
}

impl Scene {
    /// Number of frames in the scene.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame) + 1
    }

    /// Bit-allocation multiplier for this scene.
    #[must_use]
    pub fn bit_multiplier(&self) -> f32 {
        // Blend content-type multiplier with direct spatial complexity
        let ct_mult = self.content_type.complexity_multiplier();
        let spatial_boost = 1.0 + 0.3 * self.avg_spatial;
        0.6 * ct_mult + 0.4 * spatial_boost
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene-adaptive allocator
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the scene-adaptive allocator.
#[derive(Debug, Clone)]
pub struct SceneAdaptiveConfig {
    /// Target average bitrate in bits per second.
    pub target_bitrate: u64,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// SAD threshold for scene-cut detection (fraction of pixels that must
    /// differ; 0.0–1.0). Defaults to 0.15.
    pub scene_cut_threshold: f32,
    /// Minimum scene length in frames before another cut is allowed.
    /// Prevents very short "flash" scenes from dominating allocation.
    pub min_scene_frames: u32,
    /// Maximum bit-allocation ratio per frame (clamp against wild swings).
    /// Default: 4.0 × average.
    pub max_per_frame_ratio: f32,
    /// Minimum bit-allocation ratio per frame. Default: 0.1 × average.
    pub min_per_frame_ratio: f32,
}

impl Default for SceneAdaptiveConfig {
    fn default() -> Self {
        Self {
            target_bitrate: 4_000_000, // 4 Mbps
            frame_rate: 30.0,
            scene_cut_threshold: 0.15,
            min_scene_frames: 4,
            max_per_frame_ratio: 4.0,
            min_per_frame_ratio: 0.10,
        }
    }
}

impl SceneAdaptiveConfig {
    /// Average bits per frame at the target bitrate and frame rate.
    #[must_use]
    pub fn avg_bits_per_frame(&self) -> f64 {
        if self.frame_rate > 0.0 {
            self.target_bitrate as f64 / self.frame_rate
        } else {
            0.0
        }
    }
}

/// Per-frame bit target emitted by the allocator.
#[derive(Debug, Clone)]
pub struct FrameBitTarget {
    /// Frame index (presentation order).
    pub frame_index: u64,
    /// Recommended bit budget for this frame.
    pub target_bits: u64,
    /// Content classification.
    pub content_type: SceneContentType,
    /// Allocation multiplier applied (for diagnostics).
    pub multiplier: f32,
}

/// Scene-adaptive bitrate allocator.
///
/// Feed frame metrics in presentation order via [`Self::push_frame`]; call
/// [`Self::flush`] at end-of-stream to ensure the last scene is fully allocated.
/// Retrieve frame bit targets via [`Self::drain_targets`].
pub struct SceneAdaptiveAllocator {
    config: SceneAdaptiveConfig,
    /// Pending frames not yet assigned to a scene.
    pending: Vec<FrameContentMetrics>,
    /// Completed scenes.
    scenes: Vec<Scene>,
    /// Allocated frame targets ready for consumption.
    targets: Vec<FrameBitTarget>,
    /// Frames in the current open scene.
    current_scene_frames: Vec<FrameContentMetrics>,
    /// Frames since the last accepted scene cut.
    frames_since_cut: u32,
}

impl SceneAdaptiveAllocator {
    /// Create a new allocator with the given configuration.
    #[must_use]
    pub fn new(config: SceneAdaptiveConfig) -> Self {
        Self {
            config,
            pending: Vec::new(),
            scenes: Vec::new(),
            targets: Vec::new(),
            current_scene_frames: Vec::new(),
            frames_since_cut: 0,
        }
    }

    /// Push metrics for the next frame in presentation order.
    ///
    /// When a scene cut is detected (and `min_scene_frames` has elapsed),
    /// the current scene is closed and bit targets are emitted for it.
    pub fn push_frame(&mut self, metrics: FrameContentMetrics) -> CodecResult<()> {
        self.frames_since_cut += 1;

        let is_cut = metrics.is_scene_cut
            && self.frames_since_cut >= self.config.min_scene_frames
            && !self.current_scene_frames.is_empty();

        if is_cut {
            self.close_current_scene()?;
            self.frames_since_cut = 0;
        }

        self.current_scene_frames.push(metrics);
        Ok(())
    }

    /// Flush all remaining buffered frames and emit their bit targets.
    ///
    /// Must be called at end-of-stream.
    pub fn flush(&mut self) -> CodecResult<()> {
        if !self.current_scene_frames.is_empty() {
            self.close_current_scene()?;
        }
        Ok(())
    }

    /// Drain all available [`FrameBitTarget`] entries.
    pub fn drain_targets(&mut self) -> Vec<FrameBitTarget> {
        std::mem::take(&mut self.targets)
    }

    /// Return a reference to completed scenes (for diagnostics / tests).
    #[must_use]
    pub fn scenes(&self) -> &[Scene] {
        &self.scenes
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Close the current scene, build a [`Scene`] descriptor, and emit
    /// per-frame bit targets for every frame in that scene.
    fn close_current_scene(&mut self) -> CodecResult<()> {
        if self.current_scene_frames.is_empty() {
            return Ok(());
        }

        let frames = std::mem::take(&mut self.current_scene_frames);

        // Aggregate statistics
        let n = frames.len() as f32;
        let avg_spatial = frames.iter().map(|f| f.spatial_complexity).sum::<f32>() / n;
        let avg_temporal = frames.iter().map(|f| f.temporal_complexity).sum::<f32>() / n;

        // Dominant content type: most complex frame wins for scene-cuts; otherwise majority
        let content_type = dominant_content_type(&frames);

        let start_frame = frames
            .first()
            .ok_or_else(|| CodecError::InvalidData("empty scene".into()))?
            .frame_index;
        let end_frame = frames
            .last()
            .ok_or_else(|| CodecError::InvalidData("empty scene".into()))?
            .frame_index;

        let scene = Scene {
            start_frame,
            end_frame,
            content_type,
            avg_spatial,
            avg_temporal,
        };

        let scene_mult = scene.bit_multiplier();
        let avg_bits = self.config.avg_bits_per_frame();

        // Per-frame allocation: scale by individual frame complexity within
        // the scene, normalised so the scene total equals the budgeted total.
        let scene_total_budget = avg_bits * frames.len() as f64 * scene_mult as f64;

        // Compute unnormalised weights from per-frame spatial+temporal
        let weights: Vec<f32> = frames
            .iter()
            .map(|f| {
                let ct_mult = f.classify().complexity_multiplier();
                ct_mult * (1.0 + 0.5 * f.spatial_complexity + 0.5 * f.temporal_complexity)
            })
            .collect();
        let weight_sum: f32 = weights.iter().sum();
        let weight_sum = if weight_sum > 0.0 { weight_sum } else { 1.0 };

        for (frame_metrics, w) in frames.iter().zip(weights.iter()) {
            let raw_bits = scene_total_budget * (*w as f64 / weight_sum as f64);
            // Clamp to [min, max] ratios
            let min_bits = avg_bits * self.config.min_per_frame_ratio as f64;
            let max_bits = avg_bits * self.config.max_per_frame_ratio as f64;
            let target_bits = raw_bits.min(max_bits).max(min_bits) as u64;

            self.targets.push(FrameBitTarget {
                frame_index: frame_metrics.frame_index,
                target_bits,
                content_type: frame_metrics.classify(),
                multiplier: *w / (weight_sum / frames.len() as f32),
            });
        }

        self.scenes.push(scene);
        Ok(())
    }
}

/// Pick the dominant [`SceneContentType`] from a slice of frame metrics.
///
/// Scene-cut frames take priority; otherwise use the most-frequent type.
fn dominant_content_type(frames: &[FrameContentMetrics]) -> SceneContentType {
    // If any frame is a scene cut the whole scene is labelled as one
    if frames.iter().any(|f| f.is_scene_cut) {
        return SceneContentType::SceneCut;
    }
    let mut counts = [0u32; 5]; // indexed by variant discriminant
    for f in frames {
        let idx = match f.classify() {
            SceneContentType::HighMotion => 0,
            SceneContentType::MidMotion => 1,
            SceneContentType::StaticScene => 2,
            SceneContentType::Transition => 3,
            SceneContentType::SceneCut => 4,
        };
        counts[idx] += 1;
    }
    let max_idx = counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(2);
    match max_idx {
        0 => SceneContentType::HighMotion,
        1 => SceneContentType::MidMotion,
        3 => SceneContentType::Transition,
        4 => SceneContentType::SceneCut,
        _ => SceneContentType::StaticScene,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FrameContentMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_from_raw_static_frame() {
        let m = FrameContentMetrics::from_raw(0, 100.0, 0.001 * 1920.0 * 1080.0, 1920 * 1080);
        assert!(!m.is_scene_cut);
        assert!(m.spatial_complexity > 0.0 && m.spatial_complexity < 1.0);
        assert!(m.temporal_complexity < 0.15);
    }

    #[test]
    fn test_from_raw_scene_cut() {
        // 20% of pixels fully changed → should be flagged as scene cut
        let pixels = 1920u32 * 1080;
        let sad = 0.20 * 255.0 * (pixels as f64);
        let m = FrameContentMetrics::from_raw(5, 8000.0, sad, pixels);
        assert!(m.is_scene_cut);
        assert_eq!(m.classify(), SceneContentType::SceneCut);
    }

    #[test]
    fn test_classify_static() {
        let m = FrameContentMetrics {
            frame_index: 0,
            spatial_complexity: 0.1,
            temporal_complexity: 0.01,
            normalised_sad: 0.01,
            is_scene_cut: false,
        };
        assert_eq!(m.classify(), SceneContentType::StaticScene);
    }

    #[test]
    fn test_classify_high_motion() {
        let m = FrameContentMetrics {
            frame_index: 1,
            spatial_complexity: 0.5,
            temporal_complexity: 0.40,
            normalised_sad: 0.40,
            is_scene_cut: false,
        };
        assert_eq!(m.classify(), SceneContentType::HighMotion);
    }

    #[test]
    fn test_classify_transition() {
        let m = FrameContentMetrics {
            frame_index: 2,
            spatial_complexity: 0.20,
            temporal_complexity: 0.10,
            normalised_sad: 0.10,
            is_scene_cut: false,
        };
        assert_eq!(m.classify(), SceneContentType::Transition);
    }

    // ── SceneContentType ─────────────────────────────────────────────────────

    #[test]
    fn test_multipliers_ordering() {
        // HighMotion should allocate more bits than StaticScene
        assert!(
            SceneContentType::HighMotion.complexity_multiplier()
                > SceneContentType::StaticScene.complexity_multiplier()
        );
        // SceneCut should be ≥ MidMotion
        assert!(
            SceneContentType::SceneCut.complexity_multiplier()
                >= SceneContentType::MidMotion.complexity_multiplier()
        );
    }

    // ── SceneAdaptiveAllocator ───────────────────────────────────────────────

    fn make_metrics(frame_index: u64, temporal: f32, is_cut: bool) -> FrameContentMetrics {
        FrameContentMetrics {
            frame_index,
            spatial_complexity: 0.3,
            temporal_complexity: temporal,
            normalised_sad: temporal,
            is_scene_cut: is_cut,
        }
    }

    #[test]
    fn test_allocator_single_scene() {
        let cfg = SceneAdaptiveConfig {
            target_bitrate: 1_000_000,
            frame_rate: 10.0,
            ..Default::default()
        };
        let mut alloc = SceneAdaptiveAllocator::new(cfg);
        for i in 0..10u64 {
            alloc.push_frame(make_metrics(i, 0.05, false)).unwrap();
        }
        alloc.flush().unwrap();
        let targets = alloc.drain_targets();
        assert_eq!(targets.len(), 10, "all 10 frames should have targets");
        for t in &targets {
            assert!(t.target_bits > 0, "target_bits must be positive");
        }
    }

    #[test]
    fn test_allocator_two_scenes() {
        let cfg = SceneAdaptiveConfig {
            target_bitrate: 2_000_000,
            frame_rate: 25.0,
            min_scene_frames: 2,
            ..Default::default()
        };
        let mut alloc = SceneAdaptiveAllocator::new(cfg);
        // 5 frames of static content
        for i in 0..5u64 {
            alloc.push_frame(make_metrics(i, 0.01, false)).unwrap();
        }
        // Scene cut at frame 5
        alloc.push_frame(make_metrics(5, 0.50, true)).unwrap();
        // 4 more high-motion frames
        for i in 6..10u64 {
            alloc.push_frame(make_metrics(i, 0.35, false)).unwrap();
        }
        alloc.flush().unwrap();
        let targets = alloc.drain_targets();
        assert_eq!(targets.len(), 10);
        // Scene 1 is static → lower bits than scene 2 (high motion)
        let scene1_avg: f64 = targets[..5]
            .iter()
            .map(|t| t.target_bits as f64)
            .sum::<f64>()
            / 5.0;
        let scene2_avg: f64 = targets[5..]
            .iter()
            .map(|t| t.target_bits as f64)
            .sum::<f64>()
            / 5.0;
        assert!(
            scene2_avg > scene1_avg,
            "high-motion scene should get more bits: {} vs {}",
            scene2_avg,
            scene1_avg
        );
    }

    #[test]
    fn test_allocator_clamps_targets() {
        let cfg = SceneAdaptiveConfig {
            target_bitrate: 500_000,
            frame_rate: 30.0,
            max_per_frame_ratio: 3.0,
            min_per_frame_ratio: 0.2,
            ..Default::default()
        };
        let avg_bits = cfg.avg_bits_per_frame();
        let mut alloc = SceneAdaptiveAllocator::new(cfg.clone());
        for i in 0..30u64 {
            // extreme temporal complexity to try to blow past max
            alloc.push_frame(make_metrics(i, 0.99, false)).unwrap();
        }
        alloc.flush().unwrap();
        let targets = alloc.drain_targets();
        for t in &targets {
            let ratio = t.target_bits as f64 / avg_bits;
            assert!(
                ratio <= cfg.max_per_frame_ratio as f64 + 1e-6,
                "ratio {} exceeds max {}",
                ratio,
                cfg.max_per_frame_ratio
            );
            assert!(
                ratio >= cfg.min_per_frame_ratio as f64 - 1e-6,
                "ratio {} below min {}",
                ratio,
                cfg.min_per_frame_ratio
            );
        }
    }

    #[test]
    fn test_scene_descriptors() {
        let cfg = SceneAdaptiveConfig {
            min_scene_frames: 2,
            ..Default::default()
        };
        let mut alloc = SceneAdaptiveAllocator::new(cfg);
        for i in 0..4u64 {
            alloc.push_frame(make_metrics(i, 0.01, false)).unwrap();
        }
        alloc.push_frame(make_metrics(4, 0.50, true)).unwrap();
        for i in 5..8u64 {
            alloc.push_frame(make_metrics(i, 0.20, false)).unwrap();
        }
        alloc.flush().unwrap();
        let scenes = alloc.scenes().to_vec();
        assert_eq!(scenes.len(), 2, "should detect exactly 2 scenes");
        assert_eq!(scenes[0].start_frame, 0);
        assert_eq!(scenes[0].end_frame, 3);
        assert_eq!(scenes[1].start_frame, 4);
    }

    #[test]
    fn test_avg_bits_per_frame() {
        let cfg = SceneAdaptiveConfig {
            target_bitrate: 3_000_000,
            frame_rate: 30.0,
            ..Default::default()
        };
        let expected = 3_000_000.0 / 30.0;
        let got = cfg.avg_bits_per_frame();
        assert!(
            (got - expected).abs() < 1.0,
            "expected ~{expected}, got {got}"
        );
    }
}
