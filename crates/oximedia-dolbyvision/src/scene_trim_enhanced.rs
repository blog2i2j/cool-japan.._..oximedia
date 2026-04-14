//! Enhanced scene trim with L1-based scene change detection and per-scene L2 optimization.
//!
//! This module extends the basic scene trim functionality with:
//! - `SceneChangeDetector`: analyzes L1 metadata sequences for luminance discontinuities
//!   across multiple dimensions (min_pq, max_pq, avg_pq) with adaptive thresholds
//! - `TrimOptimizer`: recalculates per-scene L2 trim metadata for optimal display mapping
//!   based on scene luminance characteristics

use crate::{Level1Metadata, Level2Metadata};

// ── Scene change detection ───────────────────────────────────────────────────

/// Configuration for the enhanced scene change detector.
#[derive(Debug, Clone)]
pub struct SceneChangeDetectorConfig {
    /// Weight for avg_pq discontinuity (0.0-1.0). Default: 0.5.
    pub avg_pq_weight: f64,
    /// Weight for max_pq discontinuity (0.0-1.0). Default: 0.3.
    pub max_pq_weight: f64,
    /// Weight for min_pq discontinuity (0.0-1.0). Default: 0.2.
    pub min_pq_weight: f64,
    /// Combined weighted discontinuity threshold (0.0-1.0 of PQ range).
    /// Transitions exceeding this are classified as scene changes. Default: 0.06.
    pub discontinuity_threshold: f64,
    /// Minimum number of frames between scene boundaries. Default: 8.
    pub min_scene_length: u64,
    /// Enable adaptive thresholding based on running variance. Default: true.
    pub adaptive_threshold: bool,
    /// Smoothing window size for variance estimation. Default: 5.
    pub variance_window: usize,
    /// Multiplier applied to running standard deviation for adaptive threshold. Default: 2.5.
    pub adaptive_sigma_multiplier: f64,
}

impl Default for SceneChangeDetectorConfig {
    fn default() -> Self {
        Self {
            avg_pq_weight: 0.5,
            max_pq_weight: 0.3,
            min_pq_weight: 0.2,
            discontinuity_threshold: 0.06,
            min_scene_length: 8,
            adaptive_threshold: true,
            variance_window: 5,
            adaptive_sigma_multiplier: 2.5,
        }
    }
}

/// A single L1 metadata sample with frame index.
#[derive(Debug, Clone, Copy)]
pub struct L1Sample {
    /// Frame index (zero-based, monotonically increasing).
    pub frame_index: u64,
    /// L1 metadata for this frame.
    pub min_pq: u16,
    /// Maximum PQ value.
    pub max_pq: u16,
    /// Average PQ value.
    pub avg_pq: u16,
}

impl L1Sample {
    /// Create from a frame index and Level1Metadata.
    #[must_use]
    pub fn from_level1(frame_index: u64, l1: &Level1Metadata) -> Self {
        Self {
            frame_index,
            min_pq: l1.min_pq,
            max_pq: l1.max_pq,
            avg_pq: l1.avg_pq,
        }
    }

    /// Normalize PQ values to 0.0-1.0 range.
    fn normalized(&self) -> (f64, f64, f64) {
        (
            f64::from(self.min_pq) / 4095.0,
            f64::from(self.max_pq) / 4095.0,
            f64::from(self.avg_pq) / 4095.0,
        )
    }
}

/// Detected scene boundary.
#[derive(Debug, Clone)]
pub struct SceneBoundary {
    /// Frame index where the scene change occurs (first frame of new scene).
    pub frame_index: u64,
    /// Discontinuity magnitude (weighted, normalized 0.0-1.0).
    pub discontinuity: f64,
    /// Per-component discontinuity breakdown.
    pub components: DiscontinuityComponents,
}

/// Per-component discontinuity values for a scene boundary.
#[derive(Debug, Clone, Copy)]
pub struct DiscontinuityComponents {
    /// Absolute change in normalized avg_pq.
    pub avg_pq_delta: f64,
    /// Absolute change in normalized max_pq.
    pub max_pq_delta: f64,
    /// Absolute change in normalized min_pq.
    pub min_pq_delta: f64,
}

/// Result of scene change detection.
#[derive(Debug, Clone)]
pub struct SceneChangeDetectionResult {
    /// Detected scene boundaries.
    pub boundaries: Vec<SceneBoundary>,
    /// Total number of frames analyzed.
    pub total_frames: u64,
    /// Number of scenes detected (boundaries.len() + 1).
    pub num_scenes: usize,
}

/// Scene change detector that analyzes L1 metadata sequences.
///
/// Uses weighted multi-component luminance discontinuity analysis with
/// optional adaptive thresholding to identify scene boundaries.
#[derive(Debug)]
pub struct SceneChangeDetector {
    config: SceneChangeDetectorConfig,
}

impl SceneChangeDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: SceneChangeDetectorConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(SceneChangeDetectorConfig::default())
    }

    /// Detect scene changes from a sequence of L1 samples.
    ///
    /// The input samples must be sorted by frame_index. Returns all detected
    /// scene boundaries with discontinuity metrics.
    #[must_use]
    pub fn detect(&self, samples: &[L1Sample]) -> SceneChangeDetectionResult {
        let total_frames = samples.len() as u64;

        if samples.len() < 2 {
            return SceneChangeDetectionResult {
                boundaries: Vec::new(),
                total_frames,
                num_scenes: if samples.is_empty() { 0 } else { 1 },
            };
        }

        // Compute frame-to-frame discontinuities
        let discontinuities = self.compute_discontinuities(samples);

        // Compute adaptive threshold if enabled
        let threshold = if self.config.adaptive_threshold && discontinuities.len() >= self.config.variance_window {
            self.adaptive_threshold(&discontinuities)
        } else {
            self.config.discontinuity_threshold
        };

        // Find scene boundaries above threshold with min-scene-length suppression
        let mut boundaries = Vec::new();
        let mut last_boundary_frame: Option<u64> = None;

        for (i, &(disc, components)) in discontinuities.iter().enumerate() {
            let frame_index = samples[i + 1].frame_index;

            // Enforce minimum scene length
            if let Some(last) = last_boundary_frame {
                if frame_index.saturating_sub(last) < self.config.min_scene_length {
                    continue;
                }
            }

            if disc >= threshold {
                boundaries.push(SceneBoundary {
                    frame_index,
                    discontinuity: disc,
                    components,
                });
                last_boundary_frame = Some(frame_index);
            }
        }

        let num_scenes = boundaries.len() + 1;

        SceneChangeDetectionResult {
            boundaries,
            total_frames,
            num_scenes,
        }
    }

    /// Compute weighted discontinuities between consecutive frames.
    fn compute_discontinuities(&self, samples: &[L1Sample]) -> Vec<(f64, DiscontinuityComponents)> {
        let mut result = Vec::with_capacity(samples.len().saturating_sub(1));

        for pair in samples.windows(2) {
            let (min_a, max_a, avg_a) = pair[0].normalized();
            let (min_b, max_b, avg_b) = pair[1].normalized();

            let avg_delta = (avg_b - avg_a).abs();
            let max_delta = (max_b - max_a).abs();
            let min_delta = (min_b - min_a).abs();

            let weighted = avg_delta * self.config.avg_pq_weight
                + max_delta * self.config.max_pq_weight
                + min_delta * self.config.min_pq_weight;

            let components = DiscontinuityComponents {
                avg_pq_delta: avg_delta,
                max_pq_delta: max_delta,
                min_pq_delta: min_delta,
            };

            result.push((weighted, components));
        }

        result
    }

    /// Compute adaptive threshold from running variance of discontinuities.
    fn adaptive_threshold(&self, discontinuities: &[(f64, DiscontinuityComponents)]) -> f64 {
        let values: Vec<f64> = discontinuities.iter().map(|(d, _)| *d).collect();
        let n = values.len() as f64;

        if n < 2.0 {
            return self.config.discontinuity_threshold;
        }

        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        // Adaptive threshold = mean + sigma_multiplier * std_dev
        // But never below the base threshold
        let adaptive = mean + self.config.adaptive_sigma_multiplier * std_dev;
        adaptive.max(self.config.discontinuity_threshold)
    }
}

// ── Trim optimizer ───────────────────────────────────────────────────────────

/// Scene luminance statistics derived from L1 metadata.
#[derive(Debug, Clone)]
pub struct SceneLuminanceStats {
    /// Scene index.
    pub scene_index: usize,
    /// First frame of the scene.
    pub first_frame: u64,
    /// Last frame of the scene (inclusive).
    pub last_frame: u64,
    /// Average of min_pq values across the scene.
    pub mean_min_pq: f64,
    /// Average of max_pq values across the scene.
    pub mean_max_pq: f64,
    /// Average of avg_pq values across the scene.
    pub mean_avg_pq: f64,
    /// Peak max_pq in the scene.
    pub peak_max_pq: u16,
    /// Minimum min_pq in the scene.
    pub floor_min_pq: u16,
    /// Standard deviation of avg_pq.
    pub std_dev_avg_pq: f64,
    /// Dynamic range (peak_max_pq - floor_min_pq) in PQ codes.
    pub dynamic_range_pq: u16,
}

/// Configuration for the trim optimizer.
#[derive(Debug, Clone)]
pub struct TrimOptimizerConfig {
    /// Target display peak luminance in nits. Default: 1000.
    pub target_display_nits: u32,
    /// Source mastering display peak luminance in nits. Default: 4000.
    pub source_mastering_nits: u32,
    /// Base trim slope (neutral = 1.0). Default: 1.0.
    pub base_trim_slope: f64,
    /// Chroma weight adjustment range. Default: 0.15.
    pub chroma_weight_range: f64,
    /// Saturation gain boost for dark scenes. Default: 0.1.
    pub dark_scene_saturation_boost: f64,
}

impl Default for TrimOptimizerConfig {
    fn default() -> Self {
        Self {
            target_display_nits: 1000,
            source_mastering_nits: 4000,
            base_trim_slope: 1.0,
            chroma_weight_range: 0.15,
            dark_scene_saturation_boost: 0.1,
        }
    }
}

/// Optimized L2 trim metadata for a scene.
#[derive(Debug, Clone)]
pub struct OptimizedTrim {
    /// Scene index this trim applies to.
    pub scene_index: usize,
    /// Frame range (first, last inclusive).
    pub frame_range: (u64, u64),
    /// The optimized L2 metadata.
    pub level2: Level2Metadata,
    /// Explanation of the optimization decisions.
    pub optimization_notes: Vec<String>,
}

/// Per-scene L2 trim optimizer.
///
/// Analyzes scene luminance statistics from L1 metadata and generates
/// optimized L2 trim parameters for each scene, targeting a specific
/// display capability.
#[derive(Debug)]
pub struct TrimOptimizer {
    config: TrimOptimizerConfig,
}

impl TrimOptimizer {
    /// Create a new optimizer with the given configuration.
    #[must_use]
    pub fn new(config: TrimOptimizerConfig) -> Self {
        Self { config }
    }

    /// Create an optimizer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TrimOptimizerConfig::default())
    }

    /// Compute scene luminance statistics from L1 samples and detected boundaries.
    ///
    /// The `boundaries` list contains frame indices where scene changes occur.
    /// Samples must be sorted by frame_index.
    #[must_use]
    pub fn compute_scene_stats(
        &self,
        samples: &[L1Sample],
        boundaries: &[u64],
    ) -> Vec<SceneLuminanceStats> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Build scene ranges from boundaries
        let mut ranges: Vec<(u64, u64)> = Vec::new();
        let first_frame = samples[0].frame_index;
        let last_frame = samples.last().map_or(first_frame, |s| s.frame_index);

        if boundaries.is_empty() {
            ranges.push((first_frame, last_frame));
        } else {
            // First scene: from start to first boundary - 1
            if boundaries[0] > first_frame {
                ranges.push((first_frame, boundaries[0] - 1));
            }
            // Middle scenes
            for w in boundaries.windows(2) {
                ranges.push((w[0], w[1] - 1));
            }
            // Last scene
            if let Some(&last_boundary) = boundaries.last() {
                if last_boundary <= last_frame {
                    ranges.push((last_boundary, last_frame));
                }
            }
        }

        // Compute stats per scene
        ranges
            .iter()
            .enumerate()
            .map(|(scene_idx, &(start, end))| {
                let scene_samples: Vec<&L1Sample> = samples
                    .iter()
                    .filter(|s| s.frame_index >= start && s.frame_index <= end)
                    .collect();

                self.compute_stats_for_samples(scene_idx, start, end, &scene_samples)
            })
            .collect()
    }

    /// Compute stats for a set of samples within a scene.
    fn compute_stats_for_samples(
        &self,
        scene_index: usize,
        first_frame: u64,
        last_frame: u64,
        samples: &[&L1Sample],
    ) -> SceneLuminanceStats {
        if samples.is_empty() {
            return SceneLuminanceStats {
                scene_index,
                first_frame,
                last_frame,
                mean_min_pq: 0.0,
                mean_max_pq: 0.0,
                mean_avg_pq: 0.0,
                peak_max_pq: 0,
                floor_min_pq: 4095,
                std_dev_avg_pq: 0.0,
                dynamic_range_pq: 0,
            };
        }

        let n = samples.len() as f64;
        let mean_min: f64 = samples.iter().map(|s| f64::from(s.min_pq)).sum::<f64>() / n;
        let mean_max: f64 = samples.iter().map(|s| f64::from(s.max_pq)).sum::<f64>() / n;
        let mean_avg: f64 = samples.iter().map(|s| f64::from(s.avg_pq)).sum::<f64>() / n;

        let peak = samples.iter().map(|s| s.max_pq).max().unwrap_or(0);
        let floor = samples.iter().map(|s| s.min_pq).min().unwrap_or(4095);

        let variance = if n > 1.0 {
            samples
                .iter()
                .map(|s| (f64::from(s.avg_pq) - mean_avg).powi(2))
                .sum::<f64>()
                / (n - 1.0)
        } else {
            0.0
        };

        SceneLuminanceStats {
            scene_index,
            first_frame,
            last_frame,
            mean_min_pq: mean_min,
            mean_max_pq: mean_max,
            mean_avg_pq: mean_avg,
            peak_max_pq: peak,
            floor_min_pq: floor,
            std_dev_avg_pq: variance.sqrt(),
            dynamic_range_pq: peak.saturating_sub(floor),
        }
    }

    /// Optimize L2 trim metadata for each scene.
    ///
    /// Takes scene luminance statistics and produces optimized L2 trim parameters.
    #[must_use]
    pub fn optimize(&self, scene_stats: &[SceneLuminanceStats]) -> Vec<OptimizedTrim> {
        scene_stats.iter().map(|stats| self.optimize_scene(stats)).collect()
    }

    /// Optimize L2 trim for a single scene.
    fn optimize_scene(&self, stats: &SceneLuminanceStats) -> OptimizedTrim {
        let mut notes = Vec::new();

        let target_max_pq = nits_to_pq_f64(self.config.target_display_nits as f64);
        let source_max_pq = nits_to_pq_f64(self.config.source_mastering_nits as f64);
        let compression_ratio = target_max_pq / source_max_pq.max(1.0);

        // Slope: adjust based on scene dynamic range vs display capability
        let scene_range_ratio = f64::from(stats.dynamic_range_pq) / 4095.0;
        let slope = if scene_range_ratio > compression_ratio {
            // Scene exceeds display capability: compress more aggressively
            let adj = self.config.base_trim_slope * compression_ratio / scene_range_ratio.max(0.01);
            notes.push(format!(
                "High DR scene ({} PQ codes): slope compressed to {:.3}",
                stats.dynamic_range_pq, adj
            ));
            adj
        } else {
            // Scene fits within display: gentle expansion
            let adj = self.config.base_trim_slope
                * (1.0 + (compression_ratio - scene_range_ratio) * 0.2);
            notes.push(format!(
                "Low DR scene ({} PQ codes): slope expanded to {:.3}",
                stats.dynamic_range_pq, adj
            ));
            adj
        };

        // Offset: lift dark scenes slightly, push down bright scenes
        let avg_normalized = stats.mean_avg_pq / 4095.0;
        let mid_reference = 0.5;
        let offset = (mid_reference - avg_normalized) * 0.1;
        notes.push(format!(
            "Scene avg PQ {:.0}: offset {:.4}",
            stats.mean_avg_pq, offset
        ));

        // Power: increase contrast for flat scenes, reduce for contrasty scenes
        let power = if stats.std_dev_avg_pq < 50.0 {
            // Flat scene: boost contrast
            let p = 1.0 + (50.0 - stats.std_dev_avg_pq) / 200.0;
            notes.push(format!(
                "Low variance scene (std={:.1}): power boosted to {:.3}",
                stats.std_dev_avg_pq, p
            ));
            p
        } else if stats.std_dev_avg_pq > 200.0 {
            // High variance: reduce contrast to preserve detail
            let p = 1.0 - (stats.std_dev_avg_pq - 200.0).min(300.0) / 1000.0;
            notes.push(format!(
                "High variance scene (std={:.1}): power reduced to {:.3}",
                stats.std_dev_avg_pq, p
            ));
            p
        } else {
            1.0
        };

        // Chroma weight: boost for dark scenes, attenuate for very bright
        let chroma_weight = if avg_normalized < 0.3 {
            1.0 + self.config.chroma_weight_range * (1.0 - avg_normalized / 0.3)
        } else if avg_normalized > 0.8 {
            1.0 - self.config.chroma_weight_range * ((avg_normalized - 0.8) / 0.2)
        } else {
            1.0
        };

        // Saturation: boost in dark scenes to compensate for perceptual desaturation
        let saturation_gain = if avg_normalized < 0.25 {
            1.0 + self.config.dark_scene_saturation_boost
        } else {
            1.0
        };

        // Convert to fixed-point (scaled by 2^12)
        let scale = f64::from(1i32 << 12);

        let level2 = Level2Metadata {
            target_display_index: 0,
            trim_slope: (slope.clamp(0.1, 4.0) * scale) as i16,
            trim_offset: (offset.clamp(-0.5, 0.5) * scale) as i16,
            trim_power: (power.clamp(0.1, 4.0) * scale) as i16,
            trim_chroma_weight: (chroma_weight.clamp(0.1, 4.0) * scale) as i16,
            trim_saturation_gain: (saturation_gain.clamp(0.1, 4.0) * scale) as i16,
            ms_weight: (1.0 * scale) as i16,
            target_mid_contrast: (avg_normalized * 4095.0).clamp(0.0, 65535.0) as u16,
            clip_trim: 0,
            saturation_vector_field: Vec::new(),
            hue_vector_field: Vec::new(),
        };

        OptimizedTrim {
            scene_index: stats.scene_index,
            frame_range: (stats.first_frame, stats.last_frame),
            level2,
            optimization_notes: notes,
        }
    }
}

/// Convert nits to normalized PQ value (0.0-1.0 range).
fn nits_to_pq_f64(nits: f64) -> f64 {
    const M1: f64 = 0.159_301_758_113_479_8;
    const M2: f64 = 78.843_750;
    const C1: f64 = 0.835_937_5;
    const C2: f64 = 18.851_562_5;
    const C3: f64 = 18.6875;

    let y = nits / 10_000.0;
    let y_m1 = y.powf(M1);
    ((C1 + C2 * y_m1) / (1.0 + C3 * y_m1)).powf(M2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(pq_sequence: &[(u16, u16, u16)]) -> Vec<L1Sample> {
        pq_sequence
            .iter()
            .enumerate()
            .map(|(i, &(min, max, avg))| L1Sample {
                frame_index: i as u64,
                min_pq: min,
                max_pq: max,
                avg_pq: avg,
            })
            .collect()
    }

    #[test]
    fn test_detect_obvious_scene_change() {
        // Two distinct luminance levels
        let mut pqs = Vec::new();
        for _ in 0..20 {
            pqs.push((100, 2000, 1000)); // dark scene
        }
        for _ in 0..20 {
            pqs.push((500, 3800, 3000)); // bright scene
        }
        let samples = make_samples(&pqs);

        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&samples);

        assert!(
            !result.boundaries.is_empty(),
            "should detect at least one scene change"
        );
        assert_eq!(result.total_frames, 40);
        // The boundary should be near frame 20
        let boundary_frame = result.boundaries[0].frame_index;
        assert!(
            boundary_frame >= 18 && boundary_frame <= 22,
            "boundary at frame {boundary_frame}, expected near 20"
        );
    }

    #[test]
    fn test_no_scene_change_stable() {
        // Stable luminance throughout
        let pqs: Vec<(u16, u16, u16)> = (0..30).map(|_| (100, 2000, 1000)).collect();
        let samples = make_samples(&pqs);

        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&samples);

        assert!(
            result.boundaries.is_empty(),
            "stable scene should have no boundaries"
        );
        assert_eq!(result.num_scenes, 1);
    }

    #[test]
    fn test_multiple_scene_changes() {
        let mut pqs = Vec::new();
        // Scene 1: dark
        for _ in 0..15 {
            pqs.push((50, 1500, 800));
        }
        // Scene 2: bright
        for _ in 0..15 {
            pqs.push((400, 3900, 3200));
        }
        // Scene 3: medium
        for _ in 0..15 {
            pqs.push((200, 2500, 1500));
        }
        let samples = make_samples(&pqs);

        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&samples);

        assert!(
            result.boundaries.len() >= 2,
            "expected at least 2 boundaries, got {}",
            result.boundaries.len()
        );
        assert_eq!(result.num_scenes, result.boundaries.len() + 1);
    }

    #[test]
    fn test_min_scene_length_suppression() {
        // Test that a second scene change close to the first is suppressed.
        // First boundary is valid; a second boundary within min_scene_length is suppressed.
        let mut pqs = Vec::new();
        for _ in 0..15 {
            pqs.push((100, 2000, 1000)); // stable scene A
        }
        // Scene change at frame 15
        for _ in 0..3 {
            pqs.push((500, 3800, 3000)); // brief flash
        }
        // Another change at frame 18 (only 3 frames later) - should be suppressed
        for _ in 0..15 {
            pqs.push((100, 2000, 1000)); // back to scene A
        }
        let samples = make_samples(&pqs);

        let config = SceneChangeDetectorConfig {
            min_scene_length: 10,
            adaptive_threshold: false,
            ..Default::default()
        };
        let detector = SceneChangeDetector::new(config);
        let result = detector.detect(&samples);

        // Should detect the first boundary (at ~15) but suppress the one at ~18
        assert!(
            result.boundaries.len() <= 1,
            "second boundary within min_scene_length should be suppressed, got {} boundaries",
            result.boundaries.len()
        );
    }

    #[test]
    fn test_empty_samples() {
        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&[]);

        assert!(result.boundaries.is_empty());
        assert_eq!(result.total_frames, 0);
        assert_eq!(result.num_scenes, 0);
    }

    #[test]
    fn test_single_sample() {
        let samples = make_samples(&[(100, 2000, 1000)]);
        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&samples);

        assert!(result.boundaries.is_empty());
        assert_eq!(result.num_scenes, 1);
    }

    #[test]
    fn test_discontinuity_components() {
        let mut pqs = Vec::new();
        for _ in 0..10 {
            pqs.push((100, 2000, 1000));
        }
        for _ in 0..10 {
            pqs.push((500, 3800, 3000));
        }
        let samples = make_samples(&pqs);

        let detector = SceneChangeDetector::with_defaults();
        let result = detector.detect(&samples);

        assert!(!result.boundaries.is_empty());
        let b = &result.boundaries[0];
        assert!(b.components.avg_pq_delta > 0.0);
        assert!(b.components.max_pq_delta > 0.0);
        assert!(b.discontinuity > 0.0);
    }

    #[test]
    fn test_adaptive_threshold() {
        // With adaptive thresholding, gradual changes should not trigger
        let pqs: Vec<(u16, u16, u16)> = (0..50)
            .map(|i| {
                let base = 1000 + i * 20; // Gradual increase
                (100, base as u16, (base / 2) as u16)
            })
            .collect();
        let samples = make_samples(&pqs);

        let config = SceneChangeDetectorConfig {
            adaptive_threshold: true,
            ..Default::default()
        };
        let detector = SceneChangeDetector::new(config);
        let result = detector.detect(&samples);

        // Gradual changes should NOT trigger scene changes
        assert!(
            result.boundaries.is_empty(),
            "gradual changes should not trigger boundaries, got {}",
            result.boundaries.len()
        );
    }

    #[test]
    fn test_l1_sample_from_level1() {
        let l1 = Level1Metadata {
            min_pq: 62,
            max_pq: 3696,
            avg_pq: 2048,
        };
        let sample = L1Sample::from_level1(42, &l1);
        assert_eq!(sample.frame_index, 42);
        assert_eq!(sample.min_pq, 62);
        assert_eq!(sample.max_pq, 3696);
        assert_eq!(sample.avg_pq, 2048);
    }

    // ── TrimOptimizer tests ──────────────────────────────────────────────────

    #[test]
    fn test_optimize_single_scene() {
        let optimizer = TrimOptimizer::with_defaults();
        let stats = vec![SceneLuminanceStats {
            scene_index: 0,
            first_frame: 0,
            last_frame: 99,
            mean_min_pq: 100.0,
            mean_max_pq: 3000.0,
            mean_avg_pq: 1500.0,
            peak_max_pq: 3200,
            floor_min_pq: 80,
            std_dev_avg_pq: 100.0,
            dynamic_range_pq: 3120,
        }];

        let trims = optimizer.optimize(&stats);
        assert_eq!(trims.len(), 1);
        assert_eq!(trims[0].scene_index, 0);
        assert_eq!(trims[0].frame_range, (0, 99));
        // Slope should be reasonable
        let slope_f = f64::from(trims[0].level2.trim_slope) / f64::from(1i32 << 12);
        assert!(
            slope_f > 0.1 && slope_f < 4.0,
            "slope {slope_f} out of range"
        );
    }

    #[test]
    fn test_optimize_dark_scene_saturation_boost() {
        let optimizer = TrimOptimizer::with_defaults();
        let stats = vec![SceneLuminanceStats {
            scene_index: 0,
            first_frame: 0,
            last_frame: 49,
            mean_min_pq: 10.0,
            mean_max_pq: 800.0,
            mean_avg_pq: 400.0, // avg_normalized ~ 0.098 < 0.25
            peak_max_pq: 900,
            floor_min_pq: 5,
            std_dev_avg_pq: 30.0,
            dynamic_range_pq: 895,
        }];

        let trims = optimizer.optimize(&stats);
        let sat_gain_f = f64::from(trims[0].level2.trim_saturation_gain) / f64::from(1i32 << 12);
        assert!(
            sat_gain_f > 1.0,
            "dark scene should get saturation boost, got {sat_gain_f}"
        );
    }

    #[test]
    fn test_optimize_high_dr_scene_compressed() {
        let optimizer = TrimOptimizer::with_defaults();
        let stats = vec![SceneLuminanceStats {
            scene_index: 0,
            first_frame: 0,
            last_frame: 49,
            mean_min_pq: 10.0,
            mean_max_pq: 3900.0,
            mean_avg_pq: 2000.0,
            peak_max_pq: 4050,
            floor_min_pq: 5,
            std_dev_avg_pq: 100.0,
            dynamic_range_pq: 4045, // Very high DR
        }];

        let trims = optimizer.optimize(&stats);
        let notes = &trims[0].optimization_notes;
        assert!(
            notes.iter().any(|n| n.contains("High DR")),
            "should note high DR: {notes:?}"
        );
    }

    #[test]
    fn test_compute_scene_stats_single_scene() {
        let samples = make_samples(&[
            (100, 2000, 1000),
            (120, 2100, 1050),
            (90, 1900, 950),
        ]);
        let optimizer = TrimOptimizer::with_defaults();
        let stats = optimizer.compute_scene_stats(&samples, &[]);

        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].first_frame, 0);
        assert_eq!(stats[0].last_frame, 2);
        assert!(stats[0].mean_avg_pq > 900.0 && stats[0].mean_avg_pq < 1100.0);
    }

    #[test]
    fn test_compute_scene_stats_with_boundaries() {
        let samples = make_samples(&[
            (100, 2000, 1000),
            (110, 2050, 1020),
            (500, 3800, 3000),
            (510, 3850, 3050),
        ]);
        let optimizer = TrimOptimizer::with_defaults();
        let stats = optimizer.compute_scene_stats(&samples, &[2]);

        assert_eq!(stats.len(), 2);
        // First scene: frames 0-1
        assert_eq!(stats[0].first_frame, 0);
        assert_eq!(stats[0].last_frame, 1);
        // Second scene: frames 2-3
        assert_eq!(stats[1].first_frame, 2);
        assert_eq!(stats[1].last_frame, 3);
    }

    #[test]
    fn test_end_to_end_detect_and_optimize() {
        // End-to-end: detect scenes then optimize trims
        let mut pqs = Vec::new();
        for _ in 0..20 {
            pqs.push((100, 2000, 1000));
        }
        for _ in 0..20 {
            pqs.push((500, 3800, 3000));
        }
        let samples = make_samples(&pqs);

        let detector = SceneChangeDetector::with_defaults();
        let detection = detector.detect(&samples);

        let boundary_frames: Vec<u64> = detection.boundaries.iter().map(|b| b.frame_index).collect();

        let optimizer = TrimOptimizer::with_defaults();
        let stats = optimizer.compute_scene_stats(&samples, &boundary_frames);
        let trims = optimizer.optimize(&stats);

        assert!(
            trims.len() >= 2,
            "expected at least 2 optimized trims, got {}",
            trims.len()
        );

        // Different scenes should produce different trim slopes
        if trims.len() >= 2 {
            assert_ne!(
                trims[0].level2.trim_slope, trims[1].level2.trim_slope,
                "different scenes should have different slopes"
            );
        }
    }

    #[test]
    fn test_low_variance_scene_power_boost() {
        let optimizer = TrimOptimizer::with_defaults();
        let stats = vec![SceneLuminanceStats {
            scene_index: 0,
            first_frame: 0,
            last_frame: 49,
            mean_min_pq: 1000.0,
            mean_max_pq: 2000.0,
            mean_avg_pq: 1500.0,
            peak_max_pq: 2100,
            floor_min_pq: 900,
            std_dev_avg_pq: 10.0, // Very low variance
            dynamic_range_pq: 1200,
        }];

        let trims = optimizer.optimize(&stats);
        let power_f = f64::from(trims[0].level2.trim_power) / f64::from(1i32 << 12);
        assert!(
            power_f > 1.0,
            "low variance scene should get power boost, got {power_f}"
        );
    }
}
