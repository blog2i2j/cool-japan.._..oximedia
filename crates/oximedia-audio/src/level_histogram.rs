//! Audio level histogram analyzer.
//!
//! This module provides statistical analysis of audio amplitude distribution,
//! measuring dynamic range, crest factor, headroom, and level distribution
//! across configurable dBFS bins.
//!
//! Professional audio workflows use histograms to understand:
//! - How much headroom is available before clipping
//! - The effective dynamic range of the material
//! - Whether the signal uses the full amplitude range efficiently
//! - Crest factor (peak-to-RMS ratio) indicating signal density
//!
//! # Example
//!
//! ```
//! use oximedia_audio::level_histogram::{LevelHistogram, LevelHistogramConfig};
//!
//! let config = LevelHistogramConfig {
//!     bin_count: 100,
//!     floor_db: -100.0,
//!     ceiling_db: 0.0,
//! };
//! let mut hist = LevelHistogram::new(config);
//!
//! // Feed audio samples
//! let samples: Vec<f32> = (0..4096).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
//! hist.process(&samples);
//!
//! let stats = hist.statistics();
//! assert!(stats.peak_db <= 0.0);
//! assert!(stats.crest_factor_db >= 0.0);
//! ```

#![allow(dead_code)]
#![forbid(unsafe_code)]

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the level histogram analyzer.
#[derive(Debug, Clone)]
pub struct LevelHistogramConfig {
    /// Number of histogram bins.
    pub bin_count: usize,
    /// Floor level in dBFS (samples below this are counted in the underflow bin).
    pub floor_db: f64,
    /// Ceiling level in dBFS (samples above this are counted in the overflow bin).
    pub ceiling_db: f64,
}

impl Default for LevelHistogramConfig {
    fn default() -> Self {
        Self {
            bin_count: 100,
            floor_db: -96.0,
            ceiling_db: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Statistics snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical summary of the level histogram.
#[derive(Debug, Clone)]
pub struct LevelStatistics {
    /// Peak sample level in dBFS.
    pub peak_db: f64,
    /// RMS level in dBFS.
    pub rms_db: f64,
    /// Crest factor (peak minus RMS) in dB — higher means more dynamic.
    pub crest_factor_db: f64,
    /// Dynamic range: difference between the 95th percentile and 5th percentile
    /// levels in dB.
    pub dynamic_range_db: f64,
    /// Headroom: distance from peak to 0 dBFS.
    pub headroom_db: f64,
    /// Total number of samples analysed.
    pub total_samples: u64,
    /// Number of samples at or above 0 dBFS (digital clipping).
    pub clipped_samples: u64,
    /// Number of digital-silence samples (absolute value < 1e-10).
    pub silence_samples: u64,
    /// Median level in dBFS.
    pub median_db: f64,
    /// Mean level in dBFS (computed from linear RMS, converted to dB).
    pub mean_db: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// LevelHistogram
// ─────────────────────────────────────────────────────────────────────────────

/// Histogram-based audio level analyzer.
///
/// Accumulates sample levels into dBFS bins and tracks running statistics.
pub struct LevelHistogram {
    config: LevelHistogramConfig,
    /// Bin counts.
    bins: Vec<u64>,
    /// Samples below floor.
    underflow: u64,
    /// Samples above ceiling.
    overflow: u64,
    /// Running sum-of-squares for RMS computation.
    sum_squares: f64,
    /// Peak absolute sample value (linear).
    peak_linear: f64,
    /// Total sample count.
    total: u64,
    /// Count of exactly-zero or near-zero samples.
    silence_count: u64,
    /// Count of clipped samples (|sample| >= 1.0).
    clip_count: u64,
    /// Width of each bin in dB.
    bin_width: f64,
}

impl LevelHistogram {
    /// Create a new level histogram analyzer.
    #[must_use]
    pub fn new(config: LevelHistogramConfig) -> Self {
        let bin_count = config.bin_count.max(1);
        let bin_width = (config.ceiling_db - config.floor_db) / bin_count as f64;
        Self {
            bins: vec![0_u64; bin_count],
            underflow: 0,
            overflow: 0,
            sum_squares: 0.0,
            peak_linear: 0.0,
            total: 0,
            silence_count: 0,
            clip_count: 0,
            bin_width,
            config: LevelHistogramConfig {
                bin_count,
                ..config
            },
        }
    }

    /// Process a block of f32 samples.
    pub fn process(&mut self, samples: &[f32]) {
        for &s in samples {
            self.push_sample(f64::from(s));
        }
    }

    /// Process a block of f64 samples.
    pub fn process_f64(&mut self, samples: &[f64]) {
        for &s in samples {
            self.push_sample(s);
        }
    }

    /// Push a single sample (linear amplitude).
    fn push_sample(&mut self, sample: f64) {
        let abs = sample.abs();
        self.total += 1;
        self.sum_squares += sample * sample;

        if abs > self.peak_linear {
            self.peak_linear = abs;
        }

        if abs < 1e-10 {
            self.silence_count += 1;
        }

        if abs >= 1.0 {
            self.clip_count += 1;
        }

        // Convert to dBFS
        let db = if abs < 1e-20 {
            self.config.floor_db - 1.0 // below floor
        } else {
            20.0 * abs.log10()
        };

        if db < self.config.floor_db {
            self.underflow += 1;
        } else if db >= self.config.ceiling_db {
            self.overflow += 1;
        } else {
            let idx = ((db - self.config.floor_db) / self.bin_width) as usize;
            let idx = idx.min(self.config.bin_count - 1);
            self.bins[idx] += 1;
        }
    }

    /// Get the histogram bin counts.
    #[must_use]
    pub fn bins(&self) -> &[u64] {
        &self.bins
    }

    /// Get the dBFS value for the centre of a given bin index.
    #[must_use]
    pub fn bin_center_db(&self, index: usize) -> f64 {
        self.config.floor_db + (index as f64 + 0.5) * self.bin_width
    }

    /// Number of samples below the floor.
    #[must_use]
    pub fn underflow_count(&self) -> u64 {
        self.underflow
    }

    /// Number of samples above the ceiling.
    #[must_use]
    pub fn overflow_count(&self) -> u64 {
        self.overflow
    }

    /// Total number of samples processed.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.total
    }

    /// Compute percentile level (0.0 .. 1.0) in dBFS.
    ///
    /// `p = 0.5` gives the median.  Returns the floor if no samples have been
    /// processed.
    #[must_use]
    pub fn percentile_db(&self, p: f64) -> f64 {
        let p = p.clamp(0.0, 1.0);
        if self.total == 0 {
            return self.config.floor_db;
        }

        let target = (p * self.total as f64).ceil() as u64;
        let mut cumulative = self.underflow;

        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return self.bin_center_db(i);
            }
        }

        // If we get here, it's in the overflow region.
        self.config.ceiling_db
    }

    /// Compute full statistics snapshot.
    #[must_use]
    pub fn statistics(&self) -> LevelStatistics {
        let peak_db = if self.peak_linear <= 0.0 {
            self.config.floor_db
        } else {
            20.0 * self.peak_linear.log10()
        };

        let rms_linear = if self.total == 0 {
            0.0
        } else {
            (self.sum_squares / self.total as f64).sqrt()
        };

        let rms_db = if rms_linear <= 0.0 {
            self.config.floor_db
        } else {
            20.0 * rms_linear.log10()
        };

        let crest_factor_db = (peak_db - rms_db).max(0.0);
        let headroom_db = (-peak_db).max(0.0);

        let p05 = self.percentile_db(0.05);
        let p95 = self.percentile_db(0.95);
        let dynamic_range_db = (p95 - p05).abs();

        let median_db = self.percentile_db(0.5);

        LevelStatistics {
            peak_db,
            rms_db,
            crest_factor_db,
            dynamic_range_db,
            headroom_db,
            total_samples: self.total,
            clipped_samples: self.clip_count,
            silence_samples: self.silence_count,
            median_db,
            mean_db: rms_db,
        }
    }

    /// Reset all accumulated data.
    pub fn reset(&mut self) {
        self.bins.fill(0);
        self.underflow = 0;
        self.overflow = 0;
        self.sum_squares = 0.0;
        self.peak_linear = 0.0;
        self.total = 0;
        self.silence_count = 0;
        self.clip_count = 0;
    }

    /// Get the normalised histogram (bins summing to 1.0).
    #[must_use]
    pub fn normalised_bins(&self) -> Vec<f64> {
        let total_in_bins: u64 = self.bins.iter().sum::<u64>() + self.underflow + self.overflow;
        if total_in_bins == 0 {
            return vec![0.0; self.config.bin_count];
        }
        let scale = 1.0 / total_in_bins as f64;
        self.bins.iter().map(|&c| c as f64 * scale).collect()
    }

    /// Find the bin index with the highest count (mode).
    #[must_use]
    pub fn mode_bin(&self) -> Option<usize> {
        self.bins
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .filter(|(_, &c)| c > 0)
            .map(|(i, _)| i)
    }

    /// Level in dBFS of the most common amplitude (mode).
    #[must_use]
    pub fn mode_db(&self) -> Option<f64> {
        self.mode_bin().map(|i| self.bin_center_db(i))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> LevelHistogramConfig {
        LevelHistogramConfig::default()
    }

    // 1. Empty histogram has zero counts.
    #[test]
    fn test_empty_histogram() {
        let hist = LevelHistogram::new(default_config());
        assert_eq!(hist.total_samples(), 0);
        assert_eq!(hist.underflow_count(), 0);
        assert_eq!(hist.overflow_count(), 0);
    }

    // 2. Processing increments total count.
    #[test]
    fn test_total_count() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.5_f32; 100]);
        assert_eq!(hist.total_samples(), 100);
    }

    // 3. Full-scale signal has peak near 0 dBFS.
    #[test]
    fn test_full_scale_peak() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[1.0_f32; 10]);
        let stats = hist.statistics();
        assert!((stats.peak_db - 0.0).abs() < 0.01, "peak = {}", stats.peak_db);
    }

    // 4. Half-scale signal has peak near -6 dBFS.
    #[test]
    fn test_half_scale_peak() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.5_f32; 100]);
        let stats = hist.statistics();
        let expected = 20.0 * (0.5_f64).log10(); // ~-6.02 dBFS
        assert!(
            (stats.peak_db - expected).abs() < 0.1,
            "peak = {} expected = {expected}",
            stats.peak_db
        );
    }

    // 5. Silence is counted correctly.
    #[test]
    fn test_silence_detection() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.0_f32; 50]);
        hist.process(&[0.5_f32; 50]);
        let stats = hist.statistics();
        assert_eq!(stats.silence_samples, 50);
    }

    // 6. Clipping detection.
    #[test]
    fn test_clip_detection() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[1.0_f32; 10]);
        hist.process(&[0.5_f32; 90]);
        let stats = hist.statistics();
        assert_eq!(stats.clipped_samples, 10);
    }

    // 7. Headroom calculation.
    #[test]
    fn test_headroom() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.5_f32; 100]);
        let stats = hist.statistics();
        let expected_headroom = -(20.0 * (0.5_f64).log10());
        assert!(
            (stats.headroom_db - expected_headroom).abs() < 0.1,
            "headroom = {} expected = {expected_headroom}",
            stats.headroom_db
        );
    }

    // 8. Crest factor of a constant signal is ~0 dB.
    #[test]
    fn test_crest_factor_constant() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.5_f32; 10000]);
        let stats = hist.statistics();
        // Constant signal: peak == RMS, so crest factor == 0
        assert!(
            stats.crest_factor_db < 0.1,
            "crest factor = {}",
            stats.crest_factor_db
        );
    }

    // 9. Crest factor of a sine wave is ~3 dB.
    #[test]
    fn test_crest_factor_sine() {
        let mut hist = LevelHistogram::new(default_config());
        let samples: Vec<f32> = (0..48000)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
            .collect();
        hist.process(&samples);
        let stats = hist.statistics();
        // Sine crest factor = 20*log10(1/0.7071) ≈ 3.01 dB
        assert!(
            (stats.crest_factor_db - 3.01).abs() < 0.1,
            "crest factor = {}",
            stats.crest_factor_db
        );
    }

    // 10. Percentile returns floor for empty histogram.
    #[test]
    fn test_percentile_empty() {
        let hist = LevelHistogram::new(default_config());
        let p50 = hist.percentile_db(0.5);
        assert!((p50 - (-96.0)).abs() < 1e-6);
    }

    // 11. Reset clears all data.
    #[test]
    fn test_reset() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.5_f32; 100]);
        hist.reset();
        assert_eq!(hist.total_samples(), 0);
        assert_eq!(hist.underflow_count(), 0);
        let stats = hist.statistics();
        assert_eq!(stats.total_samples, 0);
    }

    // 12. Mode bin is the bin with the highest count.
    #[test]
    fn test_mode_bin() {
        let mut hist = LevelHistogram::new(LevelHistogramConfig {
            bin_count: 10,
            floor_db: -60.0,
            ceiling_db: 0.0,
        });
        // Feed samples all at the same level → they should cluster in one bin.
        hist.process(&[0.1_f32; 1000]);
        let mode = hist.mode_bin();
        assert!(mode.is_some());
    }

    // 13. Normalised bins sum to approximately 1.0.
    #[test]
    fn test_normalised_bins_sum() {
        let mut hist = LevelHistogram::new(default_config());
        hist.process(&[0.3_f32; 500]);
        hist.process(&[0.7_f32; 500]);
        let normed = hist.normalised_bins();
        let total: f64 = normed.iter().sum::<f64>();
        // Total is bins only (excludes underflow/overflow), so may be < 1.0
        assert!(total <= 1.0 + 1e-10, "normed sum = {total}");
        assert!(total > 0.0, "normed sum should be positive");
    }

    // 14. bin_center_db returns correct center values.
    #[test]
    fn test_bin_center_db() {
        let config = LevelHistogramConfig {
            bin_count: 10,
            floor_db: -100.0,
            ceiling_db: 0.0,
        };
        let hist = LevelHistogram::new(config);
        // Bin 0 center: -100 + 0.5 * 10 = -95
        assert!((hist.bin_center_db(0) - (-95.0)).abs() < 1e-6);
        // Bin 9 center: -100 + 9.5 * 10 = -5
        assert!((hist.bin_center_db(9) - (-5.0)).abs() < 1e-6);
    }

    // 15. Dynamic range increases with wider signal.
    #[test]
    fn test_dynamic_range() {
        let mut hist = LevelHistogram::new(default_config());
        // Mix quiet and loud samples for a wide dynamic range.
        hist.process(&[0.01_f32; 2000]);
        hist.process(&[0.9_f32; 2000]);
        let stats = hist.statistics();
        assert!(
            stats.dynamic_range_db > 10.0,
            "dynamic range should be > 10 dB, got {}",
            stats.dynamic_range_db
        );
    }
}
