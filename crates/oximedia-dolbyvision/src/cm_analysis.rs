//! Content mapping analysis for Dolby Vision
//!
//! Provides statistical analysis of PQ-encoded frame metadata to characterise
//! content and recommend optimal tone-mapping trim strategies.

use crate::scene_trim::TrimTarget;

/// Statistical summary of PQ values across a sequence of frames
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ContentMappingStats {
    /// Minimum PQ code value (0–4095) across all frames
    pub min_pq: u16,
    /// Maximum PQ code value (0–4095) across all frames
    pub max_pq: u16,
    /// Average PQ value (floating-point) across all frames
    pub avg_pq: f32,
    /// 10th percentile PQ value
    pub p10_pq: u16,
    /// 90th percentile PQ value
    pub p90_pq: u16,
    /// 99th percentile PQ value
    pub p99_pq: u16,
}

/// Histogram of PQ code values (0–4095)
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PqHistogram {
    /// 4096 buckets, one per PQ code value
    pub buckets: Vec<u32>,
    /// Total number of samples added
    pub total_pixels: u64,
}

impl PqHistogram {
    /// Create a new empty histogram
    #[must_use]
    pub fn new() -> Self {
        Self {
            buckets: vec![0u32; 4096],
            total_pixels: 0,
        }
    }

    /// Add a single PQ sample to the histogram
    pub fn add_sample(&mut self, pq: u16) {
        let idx = usize::from(pq.min(4095));
        self.buckets[idx] = self.buckets[idx].saturating_add(1);
        self.total_pixels = self.total_pixels.saturating_add(1);
    }

    /// Return the PQ value at the given percentile (0.0–100.0)
    #[must_use]
    pub fn percentile(&self, p: f32) -> u16 {
        if self.total_pixels == 0 {
            return 0;
        }
        let target = (f64::from(p) / 100.0 * self.total_pixels as f64).ceil() as u64;
        let mut cumulative: u64 = 0;
        for (i, &count) in self.buckets.iter().enumerate() {
            cumulative += u64::from(count);
            if cumulative >= target {
                return i as u16;
            }
        }
        4095
    }
}

impl Default for PqHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// A single frame's Dolby Vision metadata (defined locally for analysis)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct DvMetadataFrame {
    /// Maximum PQ value in this frame (0–4095)
    pub max_pq: u16,
    /// Average PQ value in this frame (0–4095)
    pub avg_pq: u16,
    /// Frame index
    pub frame_idx: u64,
}

impl DvMetadataFrame {
    /// Create a new `DvMetadataFrame`
    #[must_use]
    pub fn new(max_pq: u16, avg_pq: u16, frame_idx: u64) -> Self {
        Self {
            max_pq,
            avg_pq,
            frame_idx,
        }
    }
}

/// Analyzes a sequence of `DvMetadataFrame` values to produce statistics
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct CmAnalyzer;

impl CmAnalyzer {
    /// Analyze a slice of frames and return aggregate content mapping statistics
    #[must_use]
    pub fn analyze(frames: &[DvMetadataFrame]) -> ContentMappingStats {
        if frames.is_empty() {
            return ContentMappingStats {
                min_pq: 0,
                max_pq: 0,
                avg_pq: 0.0,
                p10_pq: 0,
                p90_pq: 0,
                p99_pq: 0,
            };
        }

        let mut histogram = PqHistogram::new();
        let mut min_pq = u16::MAX;
        let mut max_pq = u16::MIN;
        let mut sum: f64 = 0.0;

        for frame in frames {
            // Populate histogram with max_pq samples per frame
            histogram.add_sample(frame.max_pq);

            if frame.max_pq < min_pq {
                min_pq = frame.max_pq;
            }
            if frame.max_pq > max_pq {
                max_pq = frame.max_pq;
            }
            sum += f64::from(frame.avg_pq);
        }

        let avg_pq = (sum / frames.len() as f64) as f32;

        ContentMappingStats {
            min_pq,
            max_pq,
            avg_pq,
            p10_pq: histogram.percentile(10.0),
            p90_pq: histogram.percentile(90.0),
            p99_pq: histogram.percentile(99.0),
        }
    }
}

/// Content character classification based on statistical analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ContentCharacter {
    /// Very dark content (low average PQ)
    Dark,
    /// Night scene (moderate avg, low min)
    NightScene,
    /// Day exterior (high max, moderate avg)
    DayExterior,
    /// High contrast content
    HighContrast,
    /// Low contrast content
    LowContrast,
}

impl ContentCharacter {
    /// Classify content character from statistics
    #[must_use]
    pub fn from_stats(stats: &ContentMappingStats) -> Self {
        let range = stats.max_pq.saturating_sub(stats.min_pq);
        let avg = stats.avg_pq;

        if avg < 500.0 && stats.max_pq < 1500 {
            return Self::Dark;
        }
        if avg < 700.0 && stats.min_pq < 200 && stats.max_pq > 2500 {
            return Self::NightScene;
        }
        if stats.max_pq > 3500 && avg > 1500.0 {
            return Self::DayExterior;
        }
        if range > 3000 {
            return Self::HighContrast;
        }
        Self::LowContrast
    }
}

/// Recommends optimal trim strategies based on content character
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct OptimalTrimStrategy;

impl OptimalTrimStrategy {
    /// Recommend trim parameters for the given content character and target display luminance
    #[must_use]
    pub fn recommend(character: ContentCharacter, target_nits: f32) -> TrimTarget {
        // Start from the display's standard trim
        let mut base = TrimTarget::for_display(target_nits);

        // Apply content-character adjustments
        match character {
            ContentCharacter::Dark => {
                // Lift blacks slightly to avoid crushed shadows
                base.trim_offset += 0.03;
                base.trim_power *= 1.05;
            }
            ContentCharacter::NightScene => {
                // Protect shadow detail
                base.trim_slope *= 0.97;
                base.trim_offset += 0.01;
            }
            ContentCharacter::DayExterior => {
                // Aggressive highlight mapping for bright scenes
                base.trim_slope *= 1.03;
                base.trim_power *= 0.97;
            }
            ContentCharacter::HighContrast => {
                // Widen dynamic range reproduction
                base.trim_slope *= 1.02;
                base.target_mid_contrast *= 1.05;
            }
            ContentCharacter::LowContrast => {
                // Slightly boost contrast for SDR-like displays
                base.trim_slope *= 0.99;
                base.target_mid_contrast *= 0.98;
            }
        }

        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_histogram_add_and_percentile() {
        let mut hist = PqHistogram::new();
        for pq in 0u16..=100 {
            hist.add_sample(pq);
        }
        assert_eq!(hist.total_pixels, 101);
        // 50th percentile should be around 50
        let p50 = hist.percentile(50.0);
        assert!(p50 <= 52, "p50={p50}");
    }

    #[test]
    fn test_pq_histogram_empty() {
        let hist = PqHistogram::new();
        assert_eq!(hist.percentile(50.0), 0);
    }

    #[test]
    fn test_pq_histogram_clamps_at_4095() {
        let mut hist = PqHistogram::new();
        hist.add_sample(5000); // should clamp to 4095
        assert_eq!(hist.buckets[4095], 1);
    }

    #[test]
    fn test_cm_analyzer_empty() {
        let stats = CmAnalyzer::analyze(&[]);
        assert_eq!(stats.min_pq, 0);
        assert_eq!(stats.max_pq, 0);
        assert!((stats.avg_pq).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cm_analyzer_uniform_frames() {
        let frames: Vec<DvMetadataFrame> = (0..10)
            .map(|i| DvMetadataFrame::new(2000, 1000, i))
            .collect();
        let stats = CmAnalyzer::analyze(&frames);
        assert_eq!(stats.min_pq, 2000);
        assert_eq!(stats.max_pq, 2000);
        assert!((stats.avg_pq - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_cm_analyzer_varying_frames() {
        let frames = vec![
            DvMetadataFrame::new(1000, 500, 0),
            DvMetadataFrame::new(3000, 1500, 1),
            DvMetadataFrame::new(2000, 1000, 2),
        ];
        let stats = CmAnalyzer::analyze(&frames);
        assert_eq!(stats.min_pq, 1000);
        assert_eq!(stats.max_pq, 3000);
        assert!((stats.avg_pq - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_content_character_dark() {
        let stats = ContentMappingStats {
            min_pq: 0,
            max_pq: 1200,
            avg_pq: 400.0,
            p10_pq: 100,
            p90_pq: 1000,
            p99_pq: 1100,
        };
        assert_eq!(ContentCharacter::from_stats(&stats), ContentCharacter::Dark);
    }

    #[test]
    fn test_content_character_day_exterior() {
        let stats = ContentMappingStats {
            min_pq: 500,
            max_pq: 3800,
            avg_pq: 2000.0,
            p10_pq: 600,
            p90_pq: 3600,
            p99_pq: 3700,
        };
        assert_eq!(
            ContentCharacter::from_stats(&stats),
            ContentCharacter::DayExterior
        );
    }

    #[test]
    fn test_content_character_high_contrast() {
        let stats = ContentMappingStats {
            min_pq: 50,
            max_pq: 3800,
            avg_pq: 1200.0,
            p10_pq: 100,
            p90_pq: 3500,
            p99_pq: 3700,
        };
        assert_eq!(
            ContentCharacter::from_stats(&stats),
            ContentCharacter::HighContrast
        );
    }

    #[test]
    fn test_optimal_trim_dark_content() {
        let trim = OptimalTrimStrategy::recommend(ContentCharacter::Dark, 1000.0);
        // Dark content gets positive trim_offset adjustment
        assert!(trim.trim_offset > 0.0, "trim_offset={}", trim.trim_offset);
    }

    #[test]
    fn test_optimal_trim_day_exterior() {
        let base = TrimTarget::for_display(1000.0);
        let trim = OptimalTrimStrategy::recommend(ContentCharacter::DayExterior, 1000.0);
        // Day exterior should have higher trim_slope than base
        assert!(trim.trim_slope > base.trim_slope, "expected higher slope");
    }

    #[test]
    fn test_dv_metadata_frame_creation() {
        let frame = DvMetadataFrame::new(3000, 1500, 42);
        assert_eq!(frame.max_pq, 3000);
        assert_eq!(frame.avg_pq, 1500);
        assert_eq!(frame.frame_idx, 42);
    }
}

// ── Spec-required types ───────────────────────────────────────────────────────

/// Dolby Vision Content Mapping (CM) version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CmVersion {
    /// CM v2.9 (original, Level 2 trims only).
    Cm2_9,
    /// CM v4.0 (extended, supports Level 8 target-display trims).
    Cm4_0,
}

impl CmVersion {
    /// Returns the version identifier string.
    #[must_use]
    pub fn version_string(&self) -> &str {
        match self {
            Self::Cm2_9 => "2.9",
            Self::Cm4_0 => "4.0",
        }
    }

    /// Returns `true` if this version supports Level 8 target-display trims.
    #[must_use]
    pub fn supports_level8(&self) -> bool {
        matches!(self, Self::Cm4_0)
    }
}

/// Aggregate content mapping analysis for a sequence of Dolby Vision frames.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ContentMappingAnalysis {
    /// Number of analysed frames.
    pub frame_count: u64,
    /// Average maximum PQ per frame (0.0–4095.0).
    pub avg_max_pq: f32,
    /// Average minimum PQ per frame (0.0–4095.0).
    pub avg_min_pq: f32,
    /// Number of detected scenes.
    pub scene_count: u32,
    /// Weighted complexity of the trim metadata (arbitrary units).
    pub trim_complexity: f32,
}

impl ContentMappingAnalysis {
    /// Ratio of average max PQ to average min PQ; or 0.0 if min is zero.
    #[must_use]
    pub fn dynamic_range_ratio(&self) -> f32 {
        if self.avg_min_pq < f32::EPSILON {
            return 0.0;
        }
        self.avg_max_pq / self.avg_min_pq
    }

    /// Returns `true` if more than 80 % of the average PQ range lies above
    /// PQ = 2 000 (roughly > 203 nits), indicating HDR-heavy content.
    #[must_use]
    pub fn is_hdr_heavy(&self) -> bool {
        self.avg_max_pq > 2_000.0
    }
}

/// Statistical summary of a set of normalised PQ values (0.0–1.0).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct PqStatistics {
    /// Minimum value.
    pub min: f32,
    /// Maximum value.
    pub max: f32,
    /// Arithmetic mean.
    pub mean: f32,
    /// 95th-percentile value.
    pub percentile_95: f32,
}

impl PqStatistics {
    /// Compute statistics from a slice of PQ values.
    ///
    /// Uses a sort-based approach to determine the 95th percentile.
    /// Returns all-zero if `pq_values` is empty.
    #[must_use]
    pub fn compute(pq_values: &[f32]) -> Self {
        if pq_values.is_empty() {
            return Self {
                min: 0.0,
                max: 0.0,
                mean: 0.0,
                percentile_95: 0.0,
            };
        }

        let mut sorted = pq_values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = *sorted.last().unwrap_or(&sorted[0]);
        let mean = sorted.iter().sum::<f32>() / sorted.len() as f32;

        let idx_95 = ((sorted.len() - 1) as f32 * 0.95).round() as usize;
        let percentile_95 = sorted[idx_95.min(sorted.len() - 1)];

        Self {
            min,
            max,
            mean,
            percentile_95,
        }
    }
}

/// PQ ↔ nits conversion utilities (simplified ST.2084).
#[allow(dead_code)]
pub struct PqConverter;

impl PqConverter {
    /// Convert nits to a normalised PQ signal (0.0–1.0).
    ///
    /// Uses the simplified ST.2084 formula: `(nits / 10 000) ^ 0.1593`.
    #[must_use]
    pub fn nits_to_pq(nits: f32) -> f32 {
        let y = (nits / 10_000.0_f32).max(0.0);
        y.powf(0.159_3_f32).min(1.0)
    }

    /// Convert a normalised PQ signal (0.0–1.0) back to nits.
    ///
    /// Inverse of `nits_to_pq`: `pq ^ (1 / 0.1593) * 10 000`.
    #[must_use]
    pub fn pq_to_nits(pq: f32) -> f32 {
        let pq = pq.clamp(0.0, 1.0);
        pq.powf(1.0 / 0.159_3_f32) * 10_000.0_f32
    }
}

#[cfg(test)]
mod spec_tests {
    use super::*;

    #[test]
    fn test_cm_version_string_2_9() {
        assert_eq!(CmVersion::Cm2_9.version_string(), "2.9");
    }

    #[test]
    fn test_cm_version_string_4_0() {
        assert_eq!(CmVersion::Cm4_0.version_string(), "4.0");
    }

    #[test]
    fn test_cm_version_supports_level8_false() {
        assert!(!CmVersion::Cm2_9.supports_level8());
    }

    #[test]
    fn test_cm_version_supports_level8_true() {
        assert!(CmVersion::Cm4_0.supports_level8());
    }

    #[test]
    fn test_content_mapping_analysis_dynamic_range_ratio() {
        let a = ContentMappingAnalysis {
            frame_count: 100,
            avg_max_pq: 3000.0,
            avg_min_pq: 100.0,
            scene_count: 5,
            trim_complexity: 1.5,
        };
        assert!((a.dynamic_range_ratio() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_content_mapping_analysis_zero_min_pq() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 2000.0,
            avg_min_pq: 0.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert_eq!(a.dynamic_range_ratio(), 0.0);
    }

    #[test]
    fn test_content_mapping_analysis_is_hdr_heavy_true() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 3000.0,
            avg_min_pq: 50.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert!(a.is_hdr_heavy());
    }

    #[test]
    fn test_content_mapping_analysis_is_hdr_heavy_false() {
        let a = ContentMappingAnalysis {
            frame_count: 10,
            avg_max_pq: 1500.0,
            avg_min_pq: 50.0,
            scene_count: 1,
            trim_complexity: 1.0,
        };
        assert!(!a.is_hdr_heavy());
    }

    #[test]
    fn test_pq_statistics_empty() {
        let s = PqStatistics::compute(&[]);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
    }

    #[test]
    fn test_pq_statistics_single() {
        let s = PqStatistics::compute(&[0.5]);
        assert!((s.min - 0.5).abs() < 1e-6);
        assert!((s.max - 0.5).abs() < 1e-6);
        assert!((s.mean - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_pq_statistics_percentile_95() {
        let values: Vec<f32> = (0..=100).map(|i| i as f32 / 100.0).collect();
        let s = PqStatistics::compute(&values);
        assert!(
            s.percentile_95 >= 0.94 && s.percentile_95 <= 0.96,
            "p95={}",
            s.percentile_95
        );
    }

    #[test]
    fn test_pq_converter_nits_to_pq_zero() {
        assert_eq!(PqConverter::nits_to_pq(0.0), 0.0);
    }

    #[test]
    fn test_pq_converter_nits_to_pq_ten_thousand() {
        let pq = PqConverter::nits_to_pq(10_000.0);
        assert!((pq - 1.0).abs() < 0.01, "pq={pq}");
    }

    #[test]
    fn test_pq_converter_roundtrip() {
        let nits = 1000.0_f32;
        let pq = PqConverter::nits_to_pq(nits);
        let recovered = PqConverter::pq_to_nits(pq);
        assert!((recovered - nits).abs() < 1.0, "recovered={recovered}");
    }
}
