//! Two-pass encoding management.
//!
//! Two-pass encoding runs the encoder twice: the first pass analyzes the
//! complexity of each frame, and the second pass uses that information to
//! optimally distribute bits across the timeline for a given target bitrate.

#![allow(dead_code)]

/// Configuration for a two-pass encode.
#[derive(Debug, Clone)]
pub struct TwoPassConfig {
    /// Target average bitrate in kilobits per second.
    pub target_bitrate_kbps: u32,
    /// Total input duration in milliseconds (used for bit-budget calculations).
    pub input_duration_ms: u64,
    /// Whether to perform a detailed per-frame complexity analysis in pass one.
    pub complexity_analysis: bool,
}

impl TwoPassConfig {
    /// Creates a new two-pass config with the given target bitrate and duration.
    #[must_use]
    pub fn new(target_bitrate_kbps: u32, input_duration_ms: u64) -> Self {
        Self {
            target_bitrate_kbps,
            input_duration_ms,
            complexity_analysis: true,
        }
    }

    /// Calculates the total bit budget for the encode.
    #[must_use]
    pub fn total_bits(&self) -> u64 {
        // bits = kbps * 1000 * seconds
        let seconds = self.input_duration_ms as f64 / 1000.0;
        (f64::from(self.target_bitrate_kbps) * 1000.0 * seconds) as u64
    }
}

/// Results from the first pass of a two-pass encode.
#[derive(Debug, Clone)]
pub struct PassOneResult {
    /// Per-frame complexity scores (0.0 = very simple, 1.0 = very complex).
    pub complexity_map: Vec<f64>,
    /// Mean complexity across all analyzed frames.
    pub avg_complexity: f64,
    /// Peak (maximum) complexity observed.
    pub peak_complexity: f64,
    /// How many milliseconds of content were analyzed.
    pub duration_analyzed_ms: u64,
}

impl PassOneResult {
    /// Creates a new pass-one result from a complexity map.
    fn from_complexities(complexities: Vec<f64>, duration_analyzed_ms: u64) -> Self {
        let n = complexities.len();
        let (avg, peak) = if n == 0 {
            (0.0, 0.0)
        } else {
            let sum: f64 = complexities.iter().sum();
            let peak = complexities
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            (sum / n as f64, peak)
        };
        Self {
            complexity_map: complexities,
            avg_complexity: avg,
            peak_complexity: peak,
            duration_analyzed_ms,
        }
    }

    /// Allocates a bit count for the frame at `frame_idx` from the total bit budget.
    ///
    /// Frames with higher complexity receive proportionally more bits.
    /// Falls back to an equal allocation if the sum of complexities is zero.
    #[must_use]
    pub fn allocate_bits(&self, frame_idx: usize, total_bits: u64) -> u64 {
        let n = self.complexity_map.len();
        if n == 0 || frame_idx >= n {
            return 0;
        }

        let sum: f64 = self.complexity_map.iter().sum();
        if sum <= 0.0 {
            // Uniform allocation
            return total_bits / n as u64;
        }

        let weight = self.complexity_map[frame_idx] / sum;
        (total_bits as f64 * weight) as u64
    }

    /// Returns `true` if the frame at `frame_idx` is in a complex region.
    ///
    /// A region is considered complex if its complexity score is above the
    /// average by more than one standard deviation.
    #[must_use]
    pub fn is_complex_region(&self, idx: usize) -> bool {
        let n = self.complexity_map.len();
        if n == 0 || idx >= n {
            return false;
        }

        if n == 1 {
            return self.complexity_map[0] > 0.5;
        }

        let mean = self.avg_complexity;
        let variance: f64 = self
            .complexity_map
            .iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_dev = variance.sqrt();
        self.complexity_map[idx] > mean + std_dev
    }

    /// Returns the fraction of frames classified as complex regions.
    #[must_use]
    pub fn complex_region_fraction(&self) -> f64 {
        let n = self.complexity_map.len();
        if n == 0 {
            return 0.0;
        }
        let count = (0..n).filter(|&i| self.is_complex_region(i)).count();
        count as f64 / n as f64
    }
}

/// A two-pass encoder that manages both encoding passes.
pub struct TwoPassEncoder {
    /// Configuration for this encoder.
    pub config: TwoPassConfig,
    /// Results from the first pass, once completed.
    pub pass_one_result: Option<PassOneResult>,
}

impl TwoPassEncoder {
    /// Creates a new two-pass encoder from the given config.
    #[must_use]
    pub fn new(config: TwoPassConfig) -> Self {
        Self {
            config,
            pass_one_result: None,
        }
    }

    /// Ingests the per-frame complexity data from pass one and stores the result.
    ///
    /// Returns a reference to the stored `PassOneResult`.
    pub fn analyze_pass_one(&mut self, complexities: Vec<f64>) -> &PassOneResult {
        let duration = self.config.input_duration_ms;
        self.pass_one_result
            .insert(PassOneResult::from_complexities(complexities, duration))
    }

    /// Returns the recommended bitrate in kbps for the frame at `frame_idx` in pass two.
    ///
    /// If pass one has not been run, falls back to the configured target bitrate.
    #[must_use]
    pub fn encode_bitrate_for_frame(&self, frame_idx: usize) -> u32 {
        let Some(pass_one) = &self.pass_one_result else {
            return self.config.target_bitrate_kbps;
        };

        let total_bits = self.config.total_bits();
        let n = pass_one.complexity_map.len();
        if n == 0 {
            return self.config.target_bitrate_kbps;
        }

        let frame_bits = pass_one.allocate_bits(frame_idx, total_bits);

        // Convert from total frame bits to kbps at a nominal 1-second window
        // (approximate: we treat the frame budget as a per-frame kbps target)
        let duration_s = self.config.input_duration_ms as f64 / 1000.0;
        if duration_s <= 0.0 {
            return self.config.target_bitrate_kbps;
        }
        let avg_bits_per_frame = total_bits as f64 / n as f64;
        let scale = if avg_bits_per_frame > 0.0 {
            frame_bits as f64 / avg_bits_per_frame
        } else {
            1.0
        };

        // Clamp to [10%, 500%] of target to avoid extreme values
        let scaled = (f64::from(self.config.target_bitrate_kbps) * scale).clamp(
            f64::from(self.config.target_bitrate_kbps) * 0.1,
            f64::from(self.config.target_bitrate_kbps) * 5.0,
        );
        scaled as u32
    }

    /// Returns whether pass one has been completed.
    #[must_use]
    pub fn pass_one_complete(&self) -> bool {
        self.pass_one_result.is_some()
    }

    /// Returns the collected statistics from pass one as a serializable report.
    #[must_use]
    pub fn statistics(&self) -> Option<TwoPassStatistics> {
        let pass_one = self.pass_one_result.as_ref()?;
        let total_bits = self.config.total_bits();

        Some(TwoPassStatistics::from_pass_one(
            pass_one,
            &self.config,
            total_bits,
        ))
    }
}

/// Comprehensive statistics collected from the two-pass encoding process.
///
/// This provides detailed information about the content complexity, bit
/// allocation, and quality metrics gathered during the first pass that
/// guides the second pass encoding.
#[derive(Debug, Clone)]
pub struct TwoPassStatistics {
    /// Total number of frames analyzed.
    pub frame_count: usize,
    /// Mean complexity score across all frames (0.0-1.0).
    pub mean_complexity: f64,
    /// Peak (maximum) complexity observed.
    pub peak_complexity: f64,
    /// Minimum complexity observed.
    pub min_complexity: f64,
    /// Standard deviation of complexity scores.
    pub complexity_std_dev: f64,
    /// Fraction of frames classified as complex regions (0.0-1.0).
    pub complex_region_fraction: f64,
    /// Total bit budget in bits.
    pub total_bit_budget: u64,
    /// Average bits per frame.
    pub avg_bits_per_frame: u64,
    /// Maximum bits allocated to any single frame.
    pub max_bits_per_frame: u64,
    /// Minimum bits allocated to any single frame.
    pub min_bits_per_frame: u64,
    /// Target bitrate in kbps.
    pub target_bitrate_kbps: u32,
    /// Content duration in milliseconds.
    pub duration_ms: u64,
    /// Complexity histogram (10 bins covering 0.0-1.0).
    pub complexity_histogram: [u32; 10],
    /// Scene change indices (frames where complexity jumps significantly).
    pub scene_change_indices: Vec<usize>,
}

impl TwoPassStatistics {
    /// Constructs statistics from pass one results.
    fn from_pass_one(pass_one: &PassOneResult, config: &TwoPassConfig, total_bits: u64) -> Self {
        let n = pass_one.complexity_map.len();
        let (min_c, max_c, std_dev) = if n == 0 {
            (0.0, 0.0, 0.0)
        } else {
            let min = pass_one
                .complexity_map
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max = pass_one.peak_complexity;
            let mean = pass_one.avg_complexity;
            let variance: f64 = pass_one
                .complexity_map
                .iter()
                .map(|&c| (c - mean).powi(2))
                .sum::<f64>()
                / n as f64;
            (min, max, variance.sqrt())
        };

        // Compute per-frame bit allocations
        let mut max_bits: u64 = 0;
        let mut min_bits: u64 = u64::MAX;
        for i in 0..n {
            let bits = pass_one.allocate_bits(i, total_bits);
            if bits > max_bits {
                max_bits = bits;
            }
            if bits < min_bits {
                min_bits = bits;
            }
        }
        if n == 0 {
            min_bits = 0;
        }

        let avg_bits = if n > 0 { total_bits / n as u64 } else { 0 };

        // Build complexity histogram (10 bins: [0.0-0.1), [0.1-0.2), ... [0.9-1.0])
        let mut histogram = [0u32; 10];
        for &c in &pass_one.complexity_map {
            let bin = ((c * 10.0).floor() as usize).min(9);
            histogram[bin] += 1;
        }

        // Detect scene changes: frames where complexity jumps by more than
        // 2x the standard deviation from the previous frame
        let scene_changes = Self::detect_scene_changes(&pass_one.complexity_map, std_dev);

        Self {
            frame_count: n,
            mean_complexity: pass_one.avg_complexity,
            peak_complexity: max_c,
            min_complexity: min_c,
            complexity_std_dev: std_dev,
            complex_region_fraction: pass_one.complex_region_fraction(),
            total_bit_budget: total_bits,
            avg_bits_per_frame: avg_bits,
            max_bits_per_frame: max_bits,
            min_bits_per_frame: min_bits,
            target_bitrate_kbps: config.target_bitrate_kbps,
            duration_ms: config.input_duration_ms,
            complexity_histogram: histogram,
            scene_change_indices: scene_changes,
        }
    }

    /// Detects scene changes by finding frames where complexity changes
    /// by more than 2 standard deviations from the previous frame.
    fn detect_scene_changes(complexities: &[f64], std_dev: f64) -> Vec<usize> {
        let threshold = 2.0 * std_dev;
        if threshold <= 0.0 || complexities.len() < 2 {
            return Vec::new();
        }

        let mut changes = Vec::new();
        for i in 1..complexities.len() {
            let delta = (complexities[i] - complexities[i - 1]).abs();
            if delta > threshold {
                changes.push(i);
            }
        }
        changes
    }

    /// Returns the compression ratio (mean complexity / peak complexity).
    ///
    /// Values close to 1.0 indicate uniform content; values close to 0.0
    /// indicate highly variable content that benefits most from two-pass.
    #[must_use]
    pub fn content_uniformity(&self) -> f64 {
        if self.peak_complexity <= 0.0 {
            return 1.0;
        }
        self.mean_complexity / self.peak_complexity
    }

    /// Returns the bit allocation ratio (max bits / avg bits).
    ///
    /// Higher values indicate more aggressive bit redistribution.
    #[must_use]
    pub fn bit_allocation_ratio(&self) -> f64 {
        if self.avg_bits_per_frame == 0 {
            return 1.0;
        }
        self.max_bits_per_frame as f64 / self.avg_bits_per_frame as f64
    }

    /// Returns `true` if the content would significantly benefit from
    /// two-pass encoding (high complexity variance).
    #[must_use]
    pub fn benefits_from_two_pass(&self) -> bool {
        // High variance relative to mean indicates benefit
        if self.mean_complexity <= 0.0 {
            return false;
        }
        let cv = self.complexity_std_dev / self.mean_complexity;
        cv > 0.3 // coefficient of variation > 30%
    }

    /// Returns the number of detected scene changes.
    #[must_use]
    pub fn scene_change_count(&self) -> usize {
        self.scene_change_indices.len()
    }

    /// Returns a human-readable summary of the statistics.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Frames: {} | Complexity: mean={:.3}, peak={:.3}, std={:.3} | \
             Bits: budget={}, avg/frame={}, max/frame={} | \
             Scene changes: {} | Two-pass benefit: {}",
            self.frame_count,
            self.mean_complexity,
            self.peak_complexity,
            self.complexity_std_dev,
            self.total_bit_budget,
            self.avg_bits_per_frame,
            self.max_bits_per_frame,
            self.scene_change_count(),
            if self.benefits_from_two_pass() {
                "high"
            } else {
                "low"
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_pass_config_total_bits() {
        let cfg = TwoPassConfig::new(5000, 10_000); // 5 Mbps for 10 seconds
        assert_eq!(cfg.total_bits(), 50_000_000);
    }

    #[test]
    fn test_two_pass_config_zero_duration() {
        let cfg = TwoPassConfig::new(5000, 0);
        assert_eq!(cfg.total_bits(), 0);
    }

    #[test]
    fn test_pass_one_result_avg_complexity() {
        let result = PassOneResult::from_complexities(vec![0.2, 0.4, 0.6, 0.8], 4000);
        assert!((result.avg_complexity - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_pass_one_result_peak_complexity() {
        let result = PassOneResult::from_complexities(vec![0.1, 0.9, 0.5], 3000);
        assert!((result.peak_complexity - 0.9).abs() < 1e-9);
    }

    #[test]
    fn test_pass_one_result_empty() {
        let result = PassOneResult::from_complexities(vec![], 0);
        assert_eq!(result.avg_complexity, 0.0);
        assert_eq!(result.peak_complexity, 0.0);
    }

    #[test]
    fn test_allocate_bits_proportional() {
        // Frame 1 has 3x the complexity of frame 0
        let result = PassOneResult::from_complexities(vec![0.25, 0.75], 2000);
        let total_bits = 10_000_000u64;
        let bits_0 = result.allocate_bits(0, total_bits);
        let bits_1 = result.allocate_bits(1, total_bits);
        assert_eq!(bits_0 + bits_1, total_bits);
        assert!(bits_1 > bits_0);
    }

    #[test]
    fn test_allocate_bits_out_of_range() {
        let result = PassOneResult::from_complexities(vec![0.5, 0.5], 2000);
        assert_eq!(result.allocate_bits(99, 1_000_000), 0);
    }

    #[test]
    fn test_is_complex_region_simple() {
        // All frames have the same complexity → nothing is complex
        let result = PassOneResult::from_complexities(vec![0.5, 0.5, 0.5, 0.5], 4000);
        assert!(!result.is_complex_region(0));
    }

    #[test]
    fn test_is_complex_region_clear_outlier() {
        // Frame 3 is a clear outlier
        let result = PassOneResult::from_complexities(vec![0.1, 0.1, 0.1, 0.9], 4000);
        assert!(result.is_complex_region(3));
        assert!(!result.is_complex_region(0));
    }

    #[test]
    fn test_two_pass_encoder_fallback_before_pass_one() {
        let cfg = TwoPassConfig::new(4000, 5000);
        let encoder = TwoPassEncoder::new(cfg);
        assert!(!encoder.pass_one_complete());
        // Should return target bitrate before pass one
        assert_eq!(encoder.encode_bitrate_for_frame(0), 4000);
    }

    #[test]
    fn test_two_pass_encoder_analyze_and_encode() {
        let cfg = TwoPassConfig::new(4000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.2, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]);
        assert!(encoder.pass_one_complete());
        // The complex frame should receive more bits → higher bitrate
        let complex_bitrate = encoder.encode_bitrate_for_frame(2);
        let simple_bitrate = encoder.encode_bitrate_for_frame(0);
        assert!(complex_bitrate > simple_bitrate);
    }

    #[test]
    fn test_complex_region_fraction() {
        let result = PassOneResult::from_complexities(vec![0.1, 0.1, 0.1, 0.9], 4000);
        let fraction = result.complex_region_fraction();
        assert!(fraction > 0.0);
        assert!(fraction <= 1.0);
    }

    #[test]
    fn test_is_complex_region_single_frame() {
        let result = PassOneResult::from_complexities(vec![0.8], 1000);
        assert!(result.is_complex_region(0));
        let result_low = PassOneResult::from_complexities(vec![0.2], 1000);
        assert!(!result_low.is_complex_region(0));
    }

    // ── TwoPassStatistics tests ─────────────────────────────────────────

    #[test]
    fn test_statistics_basic() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.2, 0.4, 0.6, 0.8]);
        let stats = encoder
            .statistics()
            .expect("should have stats after pass one");
        assert_eq!(stats.frame_count, 4);
        assert!((stats.mean_complexity - 0.5).abs() < 1e-9);
        assert!((stats.peak_complexity - 0.8).abs() < 1e-9);
        assert!((stats.min_complexity - 0.2).abs() < 1e-9);
        assert!(stats.complexity_std_dev > 0.0);
        assert_eq!(stats.target_bitrate_kbps, 5000);
        assert_eq!(stats.duration_ms, 10_000);
    }

    #[test]
    fn test_statistics_bit_budget() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.5, 0.5, 0.5, 0.5]);
        let stats = encoder.statistics().expect("should have stats");
        assert_eq!(stats.total_bit_budget, 50_000_000);
        assert_eq!(stats.avg_bits_per_frame, 12_500_000);
    }

    #[test]
    fn test_statistics_none_before_pass_one() {
        let cfg = TwoPassConfig::new(4000, 5000);
        let encoder = TwoPassEncoder::new(cfg);
        assert!(encoder.statistics().is_none());
    }

    #[test]
    fn test_statistics_content_uniformity_uniform() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.5, 0.5, 0.5, 0.5]);
        let stats = encoder.statistics().expect("should have stats");
        assert!((stats.content_uniformity() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_content_uniformity_variable() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.1, 0.1, 0.1, 0.9]);
        let stats = encoder.statistics().expect("should have stats");
        assert!(stats.content_uniformity() < 0.5);
    }

    #[test]
    fn test_statistics_bit_allocation_ratio_uniform() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.5, 0.5, 0.5, 0.5]);
        let stats = encoder.statistics().expect("should have stats");
        // Uniform content -> equal allocation -> ratio ~1.0
        assert!((stats.bit_allocation_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_statistics_bit_allocation_ratio_variable() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.1, 0.1, 0.1, 0.9]);
        let stats = encoder.statistics().expect("should have stats");
        // Complex frame should get more bits -> ratio > 1
        assert!(stats.bit_allocation_ratio() > 1.5);
    }

    #[test]
    fn test_statistics_benefits_from_two_pass_variable() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9]);
        let stats = encoder.statistics().expect("should have stats");
        assert!(stats.benefits_from_two_pass());
    }

    #[test]
    fn test_statistics_not_benefits_from_two_pass_uniform() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let stats = encoder.statistics().expect("should have stats");
        assert!(!stats.benefits_from_two_pass());
    }

    #[test]
    fn test_statistics_complexity_histogram() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        // 4 frames at 0.15, 1 at 0.85
        encoder.analyze_pass_one(vec![0.15, 0.15, 0.15, 0.15, 0.85]);
        let stats = encoder.statistics().expect("should have stats");
        // Bin 1 (0.1-0.2) should have 4 frames
        assert_eq!(stats.complexity_histogram[1], 4);
        // Bin 8 (0.8-0.9) should have 1 frame
        assert_eq!(stats.complexity_histogram[8], 1);
    }

    #[test]
    fn test_statistics_scene_changes() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        // Clear scene change at index 3 (0.1 -> 0.9)
        encoder.analyze_pass_one(vec![0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1]);
        let stats = encoder.statistics().expect("should have stats");
        assert!(stats.scene_change_count() > 0);
        assert!(stats.scene_change_indices.contains(&3));
    }

    #[test]
    fn test_statistics_no_scene_changes_uniform() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.5, 0.5, 0.5, 0.5]);
        let stats = encoder.statistics().expect("should have stats");
        assert_eq!(stats.scene_change_count(), 0);
    }

    #[test]
    fn test_statistics_summary_not_empty() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![0.2, 0.4, 0.6, 0.8]);
        let stats = encoder.statistics().expect("should have stats");
        let summary = stats.summary();
        assert!(!summary.is_empty());
        assert!(summary.contains("Frames: 4"));
        assert!(summary.contains("Scene changes:"));
    }

    #[test]
    fn test_statistics_empty_complexities() {
        let cfg = TwoPassConfig::new(5000, 10_000);
        let mut encoder = TwoPassEncoder::new(cfg);
        encoder.analyze_pass_one(vec![]);
        let stats = encoder.statistics().expect("should have stats");
        assert_eq!(stats.frame_count, 0);
        assert_eq!(stats.avg_bits_per_frame, 0);
        assert_eq!(stats.min_bits_per_frame, 0);
        assert!((stats.content_uniformity() - 1.0).abs() < 1e-9);
        assert!((stats.bit_allocation_ratio() - 1.0).abs() < 1e-9);
        assert!(!stats.benefits_from_two_pass());
    }
}
