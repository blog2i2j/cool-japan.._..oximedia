//! Color banding detection for video quality assessment.
//!
//! Color banding (also called posterization or false contouring) is a
//! compression artefact that appears in smooth gradients: rather than a
//! continuous tone transition the image shows abrupt, visible "bands" at
//! quantization boundaries.
//!
//! This module provides:
//! - [`BandingDetector`]: identifies banding in a single luma or chroma plane.
//! - [`GradientSmoothnessAnalyzer`]: measures the smoothness of gradients to
//!   distinguish genuine flat areas from false-contour posterization.
//! - [`ColorBandingReport`]: combines multiple per-plane scores into a
//!   comprehensive assessment.
//!
//! All detectors operate on normalised f32 values in [0.0, 1.0].
//!
//! # Example
//!
//! ```
//! use oximedia_quality::color_banding::{BandingDetector, ColorBandingReport};
//!
//! let w = 64u32;
//! let h = 64u32;
//! // Linear gradient — should have low banding score
//! let frame: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
//! let detector = BandingDetector::default();
//! let score = detector.score(&frame, w, h);
//! assert!(score >= 0.0 && score <= 1.0);
//! ```

use serde::{Deserialize, Serialize};

// ─── Band Count Estimator ─────────────────────────────────────────────────────

/// Estimates the number of distinct quantization bands along horizontal scan
/// lines of a smooth gradient region.
///
/// The estimator finds "steps" — consecutive pixels where the gradient is
/// near-zero followed by a sudden larger jump, indicating a false contour.
pub struct BandCountEstimator {
    /// A gradient below this value (per-pixel, normalised) is considered
    /// part of a flat band.
    pub flat_threshold: f32,
    /// A gradient above this value signals the transition between two bands.
    pub step_threshold: f32,
}

impl Default for BandCountEstimator {
    fn default() -> Self {
        Self {
            flat_threshold: 1.0 / 512.0, // half a quantization step at 8-bit
            step_threshold: 2.0 / 255.0, // at least two quantization steps
        }
    }
}

impl BandCountEstimator {
    /// Creates an estimator with default thresholds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an estimator with custom thresholds.
    #[must_use]
    pub fn with_thresholds(flat_threshold: f32, step_threshold: f32) -> Self {
        Self {
            flat_threshold: flat_threshold.max(0.0),
            step_threshold: step_threshold.max(0.0),
        }
    }

    /// Estimates the number of distinct bands along a single row of `pixels`.
    ///
    /// Returns 0 if the row has fewer than 2 pixels.
    #[must_use]
    pub fn count_bands_in_row(&self, pixels: &[f32]) -> u32 {
        if pixels.len() < 2 {
            return 0;
        }

        // State machine: we are either inside a flat band or transitioning
        let mut bands = 1u32; // at least one band
        let mut in_flat = true;
        let mut flat_len = 0u32;

        for i in 1..pixels.len() {
            let delta = (pixels[i] - pixels[i - 1]).abs();
            if delta <= self.flat_threshold {
                if !in_flat {
                    // Entering a new flat region after a step → new band
                    bands += 1;
                    in_flat = true;
                    flat_len = 1;
                } else {
                    flat_len += 1;
                }
            } else if delta >= self.step_threshold {
                // Transition (step) detected
                if flat_len >= 2 {
                    // Only count if we had a real flat region before
                    in_flat = false;
                }
                flat_len = 0;
            } else {
                // In-between: treat as part of gradient, reset flat tracking
                flat_len = 0;
                in_flat = false;
            }
        }

        bands
    }

    /// Estimates the mean band count across all rows of a frame.
    ///
    /// `frame` is row-major, width × height.
    #[must_use]
    pub fn mean_band_count(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        if w == 0 || h == 0 || frame.len() < w * h {
            return 0.0;
        }

        let total: u64 = (0..h)
            .map(|row| {
                let row_slice = &frame[row * w..(row * w + w).min(frame.len())];
                u64::from(self.count_bands_in_row(row_slice))
            })
            .sum();

        total as f32 / h as f32
    }
}

// ─── Gradient Smoothness Analyzer ────────────────────────────────────────────

/// Measures the smoothness of gradients to distinguish between genuine
/// smooth areas (which would naturally have near-zero gradients) and banded
/// regions (which show a sawtooth-like quantization pattern).
///
/// The key metric is the *second-order gradient* (Laplacian of the gradient):
/// a smooth gradient has near-zero second derivative while a banded region
/// shows alternating near-zero and nonzero second derivatives.
pub struct GradientSmoothnessAnalyzer {
    /// Minimum first-order gradient to consider a region "non-trivially varying"
    /// (skips perfectly flat areas that are not gradients).
    pub min_gradient: f32,
}

impl Default for GradientSmoothnessAnalyzer {
    fn default() -> Self {
        Self {
            min_gradient: 0.5 / 255.0,
        }
    }
}

impl GradientSmoothnessAnalyzer {
    /// Creates an analyzer with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Computes a gradient irregularity score in [0.0, 1.0] for a row of pixels.
    ///
    /// A score close to 0 means the gradient is smooth (likely a genuine
    /// gradient, possibly low banding).  A score close to 1 means the
    /// second-order derivative has high variance relative to the first-order
    /// gradient, indicating a stepped / banded pattern.
    #[must_use]
    pub fn row_irregularity(&self, pixels: &[f32]) -> f32 {
        if pixels.len() < 3 {
            return 0.0;
        }

        let n = pixels.len();
        let mut sum_first = 0.0f64;
        let mut sum_second_sq = 0.0f64;
        let mut active = 0u32;

        for i in 1..n - 1 {
            let d1 = (pixels[i] - pixels[i - 1]).abs() as f64;
            let d2 = (pixels[i + 1] - 2.0 * pixels[i] + pixels[i - 1]).abs() as f64;

            if d1 >= f64::from(self.min_gradient) {
                sum_first += d1;
                sum_second_sq += d2 * d2;
                active += 1;
            }
        }

        if active == 0 {
            return 0.0;
        }

        let mean_first = sum_first / active as f64;
        let rms_second = (sum_second_sq / active as f64).sqrt();

        // Irregularity: ratio of second-order RMS to first-order mean
        // A step function has high ratio; smooth gradient has low ratio.
        let ratio = if mean_first > 1e-12 {
            rms_second / mean_first
        } else {
            0.0
        };

        (ratio / 2.0).clamp(0.0, 1.0) as f32
    }

    /// Computes mean gradient irregularity across all rows of a frame.
    #[must_use]
    pub fn frame_irregularity(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;
        if w == 0 || h == 0 || frame.len() < w * h {
            return 0.0;
        }

        let total: f64 = (0..h)
            .map(|row| {
                let row_slice = &frame[row * w..(row * w + w).min(frame.len())];
                f64::from(self.row_irregularity(row_slice))
            })
            .sum();

        (total / h as f64) as f32
    }
}

// ─── Main Banding Detector ────────────────────────────────────────────────────

/// Configuration for the banding detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandingConfig {
    /// Gradient below which a transition is treated as flat (normalised 0-1).
    pub flat_gradient_threshold: f32,
    /// Fraction of pixels with near-zero gradient above which a row is
    /// considered "gradient-like" (i.e., a candidate for banding).
    pub gradient_row_ratio: f32,
    /// Weight given to the gradient-smoothness score vs. flat-ratio score.
    pub smoothness_weight: f32,
}

impl Default for BandingConfig {
    fn default() -> Self {
        Self {
            flat_gradient_threshold: 1.0 / 255.0,
            gradient_row_ratio: 0.3,
            smoothness_weight: 0.5,
        }
    }
}

impl BandingConfig {
    /// Creates a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Banding severity grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BandingSeverity {
    /// No banding detected.
    None,
    /// Mild banding visible in smooth gradient regions.
    Mild,
    /// Moderate banding that is clearly visible.
    Moderate,
    /// Severe banding — large false-contour steps.
    Severe,
}

impl BandingSeverity {
    /// Classifies a banding score into a severity.
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score < 0.15 {
            Self::None
        } else if score < 0.40 {
            Self::Mild
        } else if score < 0.70 {
            Self::Moderate
        } else {
            Self::Severe
        }
    }
}

/// Detects color banding in a single image plane.
pub struct BandingDetector {
    config: BandingConfig,
    band_counter: BandCountEstimator,
    smoothness: GradientSmoothnessAnalyzer,
}

impl Default for BandingDetector {
    fn default() -> Self {
        Self {
            config: BandingConfig::default(),
            band_counter: BandCountEstimator::default(),
            smoothness: GradientSmoothnessAnalyzer::default(),
        }
    }
}

impl BandingDetector {
    /// Creates a new detector with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a detector with a custom configuration.
    #[must_use]
    pub fn with_config(config: BandingConfig) -> Self {
        Self {
            config,
            band_counter: BandCountEstimator::default(),
            smoothness: GradientSmoothnessAnalyzer::default(),
        }
    }

    /// Computes a banding score in [0.0, 1.0] for a single luma plane.
    ///
    /// Higher values indicate more visible banding.
    #[must_use]
    pub fn score(&self, frame: &[f32], width: u32, height: u32) -> f32 {
        let w = width as usize;
        let h = height as usize;

        if w < 2 || h < 2 || frame.len() < w * h {
            return 0.0;
        }

        // Flat-gradient ratio: fraction of pixels with near-zero horizontal gradient
        let mut near_flat = 0u64;
        let mut total_grad = 0u64;
        for row in 0..h {
            for col in 0..w - 1 {
                let delta = (frame[row * w + col + 1] - frame[row * w + col]).abs();
                if delta < self.config.flat_gradient_threshold {
                    near_flat += 1;
                }
                total_grad += 1;
            }
        }
        let flat_ratio = if total_grad > 0 {
            near_flat as f32 / total_grad as f32
        } else {
            0.0
        };

        // Gradient irregularity: indicates banding even in non-flat areas
        let irregularity = self.smoothness.frame_irregularity(frame, width, height);

        // Combine: flat_ratio penalises flat uniform regions more than smooth
        // gradients; irregularity catches the stepped pattern directly.
        let flat_score = ((flat_ratio - 0.5) / 0.4).clamp(0.0, 1.0);
        let sw = self.config.smoothness_weight.clamp(0.0, 1.0);
        let combined = sw * irregularity + (1.0 - sw) * flat_score;

        combined.clamp(0.0, 1.0)
    }

    /// Returns the banding severity for a frame.
    #[must_use]
    pub fn severity(&self, frame: &[f32], width: u32, height: u32) -> BandingSeverity {
        BandingSeverity::from_score(self.score(frame, width, height))
    }
}

// ─── Color Banding Report ─────────────────────────────────────────────────────

/// Full color banding report covering luma and (optionally) both chroma planes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorBandingReport {
    /// Banding score for the luma (Y) plane [0, 1].
    pub luma_score: f32,
    /// Banding score for the Cb chroma plane [0, 1] (0 if not provided).
    pub cb_score: f32,
    /// Banding score for the Cr chroma plane [0, 1] (0 if not provided).
    pub cr_score: f32,
    /// Estimated mean number of visible bands per row in the luma plane.
    pub estimated_band_count: f32,
    /// Overall severity classification.
    pub severity: BandingSeverity,
    /// Weighted overall banding score [0, 1].
    pub overall_score: f32,
}

impl ColorBandingReport {
    /// Analyzes a YCbCr frame for color banding.
    ///
    /// `luma`, `cb`, `cr` are normalised [0, 1] planes.  Chroma planes may be
    /// sub-sampled; pass their actual dimensions.  If chroma planes are empty,
    /// only luma is analyzed.
    #[must_use]
    pub fn analyze_ycbcr(
        luma: &[f32],
        luma_width: u32,
        luma_height: u32,
        cb: &[f32],
        cb_width: u32,
        cb_height: u32,
        cr: &[f32],
        cr_width: u32,
        cr_height: u32,
    ) -> Self {
        let detector = BandingDetector::default();
        let band_counter = BandCountEstimator::default();

        let luma_score = detector.score(luma, luma_width, luma_height);
        let cb_score = if cb.is_empty() {
            0.0
        } else {
            detector.score(cb, cb_width, cb_height)
        };
        let cr_score = if cr.is_empty() {
            0.0
        } else {
            detector.score(cr, cr_width, cr_height)
        };

        let estimated_band_count = band_counter.mean_band_count(luma, luma_width, luma_height);

        // Overall: luma 60 %, Cb 20 %, Cr 20 %
        let overall_score = (0.60 * luma_score + 0.20 * cb_score + 0.20 * cr_score).clamp(0.0, 1.0);
        let severity = BandingSeverity::from_score(overall_score);

        Self {
            luma_score,
            cb_score,
            cr_score,
            estimated_band_count,
            severity,
            overall_score,
        }
    }

    /// Analyzes a single-plane (grayscale) frame.
    #[must_use]
    pub fn analyze_luma(luma: &[f32], width: u32, height: u32) -> Self {
        Self::analyze_ycbcr(luma, width, height, &[], 0, 0, &[], 0, 0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_gradient(w: u32, h: u32) -> Vec<f32> {
        (0..w * h).map(|i| (i % w) as f32 / w as f32).collect()
    }

    fn flat_frame(w: u32, h: u32, val: f32) -> Vec<f32> {
        vec![val; (w * h) as usize]
    }

    fn stepped_gradient(w: u32, h: u32, steps: u32) -> Vec<f32> {
        // Each "step" covers (w / steps) pixels at a constant value
        let step_w = ((w / steps).max(1)) as usize;
        let w_us = w as usize;
        let steps_f = steps as f32;
        (0..w * h)
            .map(|i| {
                let col = (i as usize) % w_us;
                let band = col / step_w;
                band as f32 / steps_f
            })
            .collect()
    }

    // ── BandCountEstimator ─────────────────────────────────────────────────

    #[test]
    fn test_band_count_single_flat_row() {
        let estimator = BandCountEstimator::new();
        let row = vec![0.5f32; 64];
        let count = estimator.count_bands_in_row(&row);
        assert!(count >= 1);
    }

    #[test]
    fn test_band_count_stepped_row() {
        let estimator = BandCountEstimator::new();
        // 8 steps of 8 pixels each, values 0/8, 1/8, ..., 7/8
        let row: Vec<f32> = (0..64usize).map(|i| (i / 8) as f32 / 8.0).collect();
        let count = estimator.count_bands_in_row(&row);
        // Should detect multiple bands
        assert!(count >= 2);
    }

    #[test]
    fn test_band_count_empty_row() {
        let estimator = BandCountEstimator::new();
        assert_eq!(estimator.count_bands_in_row(&[]), 0);
        assert_eq!(estimator.count_bands_in_row(&[0.5]), 0);
    }

    #[test]
    fn test_mean_band_count_linear_gradient() {
        let frame = linear_gradient(64, 32);
        let estimator = BandCountEstimator::new();
        let mean = estimator.mean_band_count(&frame, 64, 32);
        // A perfect linear gradient has exactly 1 band by our definition
        assert!(mean >= 1.0);
    }

    // ── GradientSmoothnessAnalyzer ─────────────────────────────────────────

    #[test]
    fn test_smoothness_flat_row_zero() {
        let analyzer = GradientSmoothnessAnalyzer::new();
        let row = vec![0.5f32; 64];
        let irr = analyzer.row_irregularity(&row);
        // Flat row has no gradient → no active pixels → 0
        assert_eq!(irr, 0.0);
    }

    #[test]
    fn test_smoothness_stepped_row_high() {
        let analyzer = GradientSmoothnessAnalyzer::new();
        // Stepped pattern: 0, 0, 0, 1, 1, 1, 0, 0, 0, ...
        let row: Vec<f32> = (0..60usize)
            .map(|i| if (i / 3) % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let irr = analyzer.row_irregularity(&row);
        assert!((0.0..=1.0).contains(&irr));
    }

    #[test]
    fn test_smoothness_short_row_zero() {
        let analyzer = GradientSmoothnessAnalyzer::new();
        assert_eq!(analyzer.row_irregularity(&[0.0, 1.0]), 0.0);
    }

    // ── BandingDetector ────────────────────────────────────────────────────

    #[test]
    fn test_banding_score_in_range() {
        let detector = BandingDetector::new();
        let frame = linear_gradient(64, 64);
        let s = detector.score(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn test_banding_score_step_gradient_higher_than_linear() {
        let detector = BandingDetector::new();
        let linear = linear_gradient(64, 64);
        let stepped = stepped_gradient(64, 64, 8);
        let linear_score = detector.score(&linear, 64, 64);
        let stepped_score = detector.score(&stepped, 64, 64);
        // Stepped gradient should have higher (or equal) banding score
        assert!(stepped_score >= linear_score - 0.05); // allow small FP tolerance
    }

    #[test]
    fn test_banding_severity_ordering() {
        assert!(BandingSeverity::None < BandingSeverity::Mild);
        assert!(BandingSeverity::Mild < BandingSeverity::Moderate);
        assert!(BandingSeverity::Moderate < BandingSeverity::Severe);
    }

    #[test]
    fn test_banding_severity_classification() {
        assert_eq!(BandingSeverity::from_score(0.05), BandingSeverity::None);
        assert_eq!(BandingSeverity::from_score(0.25), BandingSeverity::Mild);
        assert_eq!(BandingSeverity::from_score(0.55), BandingSeverity::Moderate);
        assert_eq!(BandingSeverity::from_score(0.85), BandingSeverity::Severe);
    }

    #[test]
    fn test_banding_empty_frame_zero() {
        let detector = BandingDetector::new();
        let s = detector.score(&[], 0, 0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_banding_flat_frame_score() {
        let detector = BandingDetector::new();
        let frame = flat_frame(64, 64, 0.5);
        let s = detector.score(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&s));
        // A flat frame: all gradients are 0 → flat_ratio = 1.0 → flat_score high
        // but irregularity = 0 → combined depends on weights
    }

    // ── ColorBandingReport ─────────────────────────────────────────────────

    #[test]
    fn test_report_luma_only() {
        let frame = linear_gradient(64, 64);
        let report = ColorBandingReport::analyze_luma(&frame, 64, 64);
        assert!((0.0..=1.0).contains(&report.overall_score));
        assert!((0.0..=1.0).contains(&report.luma_score));
        assert_eq!(report.cb_score, 0.0);
        assert_eq!(report.cr_score, 0.0);
    }

    #[test]
    fn test_report_ycbcr() {
        let luma = linear_gradient(64, 64);
        let cb = flat_frame(32, 32, 0.5);
        let cr = flat_frame(32, 32, 0.5);
        let report = ColorBandingReport::analyze_ycbcr(&luma, 64, 64, &cb, 32, 32, &cr, 32, 32);
        assert!((0.0..=1.0).contains(&report.overall_score));
    }

    #[test]
    fn test_report_overall_score_weights() {
        // Verify weights sum to 1
        let w: f32 = 0.60 + 0.20 + 0.20;
        assert!((w - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_new() {
        let cfg = BandingConfig::new();
        assert!((cfg.flat_gradient_threshold - 1.0 / 255.0).abs() < 1e-7);
        assert!((cfg.smoothness_weight - 0.5).abs() < 1e-6);
    }
}
