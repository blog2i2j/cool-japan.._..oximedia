//! Confidence calibration for shot angle classification using histogram-based features.
//!
//! This module provides `ConfidenceCalibrator`, which post-processes the raw confidence
//! scores produced by `classify::AngleClassifier` using a histogram-based Platt-like
//! sigmoid calibration fit on empirical bucket statistics.
//!
//! # Algorithm Overview
//!
//! 1. **Feature extraction** — extract per-channel luminance histograms from the frame.
//! 2. **Histogram statistics** — compute mean, variance, skewness, and the histogram's
//!    low-frequency (bright sky / floor) mass to infer whether the frame contains strong
//!    vertical composition cues consistent with extreme angles.
//! 3. **Calibration curve** — a sigmoid with learnable slope `a` and bias `b` maps
//!    the raw score to a calibrated probability:
//!    ```text
//!    P_cal = 1 / (1 + exp(-(a * raw + b)))
//!    ```
//! 4. **Histogram similarity** — the calibrated score is additionally modulated by how
//!    well the extracted features agree with reference histograms stored per angle class.
//!
//! The calibrator can be updated incrementally with new (frame, ground-truth-angle)
//! observations so that it refines over time.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FrameBuffer, GrayImage};
use crate::types::CameraAngle;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of histogram bins per channel.
const HIST_BINS: usize = 32;

/// Number of supported angle classes.
const N_CLASSES: usize = 5;

/// Minimum samples required before calibration is considered reliable.
const MIN_RELIABLE_SAMPLES: u32 = 20;

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// A compact 1-D luminance histogram (32 bins, normalised).
#[derive(Debug, Clone)]
pub struct LuminanceHistogram {
    /// Normalised bin counts (sum ≈ 1.0).
    pub bins: [f32; HIST_BINS],
    /// Mean luminance (0–255).
    pub mean: f32,
    /// Variance of luminance.
    pub variance: f32,
    /// Skewness (3rd standardised moment).
    pub skewness: f32,
    /// Fraction of pixels in the brightest 25% of bins.
    pub bright_mass: f32,
    /// Fraction of pixels in the darkest 25% of bins.
    pub dark_mass: f32,
}

impl LuminanceHistogram {
    /// Compute the chi-squared distance to another histogram.
    ///
    /// Returns 0.0 for identical histograms and increases as they diverge.
    #[must_use]
    pub fn chi_squared_distance(&self, other: &Self) -> f32 {
        let mut dist = 0.0f32;
        for i in 0..HIST_BINS {
            let sum = self.bins[i] + other.bins[i];
            if sum > f32::EPSILON {
                let diff = self.bins[i] - other.bins[i];
                dist += (diff * diff) / sum;
            }
        }
        dist * 0.5
    }

    /// Compute the Bhattacharyya distance to another histogram.
    #[must_use]
    pub fn bhattacharyya_distance(&self, other: &Self) -> f32 {
        let mut coeff = 0.0f32;
        for i in 0..HIST_BINS {
            coeff += (self.bins[i] * other.bins[i]).sqrt();
        }
        // Guard against log(0)
        if coeff < f32::EPSILON {
            f32::INFINITY
        } else {
            -(coeff.ln())
        }
    }
}

/// Per-class calibration state.
#[derive(Debug, Clone)]
struct ClassCalibrationState {
    /// Platt sigmoid slope.
    slope: f32,
    /// Platt sigmoid bias.
    bias: f32,
    /// Number of positive samples seen.
    pos_count: u32,
    /// Number of negative samples seen.
    neg_count: u32,
    /// Running mean of raw scores for positive samples.
    pos_mean_score: f32,
    /// Running mean of raw scores for negative samples.
    neg_mean_score: f32,
    /// Reference histogram (mean of positive sample histograms).
    reference_histogram: Option<LuminanceHistogram>,
}

impl ClassCalibrationState {
    fn new() -> Self {
        Self {
            slope: 1.0,
            bias: 0.0,
            pos_count: 0,
            neg_count: 0,
            pos_mean_score: 0.5,
            neg_mean_score: 0.5,
            reference_histogram: None,
        }
    }

    /// Returns true if enough samples have been collected to trust calibration.
    fn is_reliable(&self) -> bool {
        self.pos_count + self.neg_count >= MIN_RELIABLE_SAMPLES
    }
}

/// Calibration record: a single training observation.
#[derive(Debug, Clone)]
pub struct CalibrationSample {
    /// Angle class label.
    pub angle: CameraAngle,
    /// Raw classifier confidence (0.0–1.0).
    pub raw_confidence: f32,
    /// Luminance histogram extracted from the corresponding frame.
    pub histogram: LuminanceHistogram,
}

/// Calibrated classification result.
#[derive(Debug, Clone)]
pub struct CalibratedAngle {
    /// The classified angle.
    pub angle: CameraAngle,
    /// Raw classifier confidence before calibration.
    pub raw_confidence: f32,
    /// Calibrated posterior probability.
    pub calibrated_confidence: f32,
    /// Histogram-similarity modulation factor (0.0–1.0).
    pub histogram_similarity: f32,
    /// Whether the calibration was based on reliable statistics.
    pub is_reliable: bool,
}

// ---------------------------------------------------------------------------
// Feature Extraction
// ---------------------------------------------------------------------------

/// Extract a grayscale luminance histogram from an RGB frame buffer.
///
/// Uses BT.601 coefficients to compute luminance.
pub fn extract_luminance_histogram(frame: &FrameBuffer) -> ShotResult<LuminanceHistogram> {
    let (h, w, c) = frame.dim();
    if c < 3 {
        return Err(ShotError::InvalidFrame(
            "Frame must have at least 3 channels for luminance extraction".to_string(),
        ));
    }

    let n_pixels = h * w;
    if n_pixels == 0 {
        return Err(ShotError::InvalidFrame("Empty frame".to_string()));
    }

    let mut bins = [0u32; HIST_BINS];
    let bin_width = 256.0f32 / HIST_BINS as f32;

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut luminances = Vec::with_capacity(n_pixels);

    for y in 0..h {
        for x in 0..w {
            let r = f32::from(frame.get(y, x, 0));
            let g = f32::from(frame.get(y, x, 1));
            let b = f32::from(frame.get(y, x, 2));
            let lum = r * 0.299 + g * 0.587 + b * 0.114;
            let idx = ((lum / bin_width) as usize).min(HIST_BINS - 1);
            bins[idx] += 1;
            sum += f64::from(lum);
            sum_sq += f64::from(lum) * f64::from(lum);
            luminances.push(lum);
        }
    }

    let mean = (sum / n_pixels as f64) as f32;
    let variance = ((sum_sq / n_pixels as f64) - (sum / n_pixels as f64).powi(2)) as f32;
    let variance = variance.max(0.0);

    // Skewness
    let std_dev = variance.sqrt();
    let skewness = if std_dev < f32::EPSILON {
        0.0
    } else {
        let sum_cubed: f64 = luminances
            .iter()
            .map(|&l| ((f64::from(l) - f64::from(mean)) / f64::from(std_dev)).powi(3))
            .sum();
        (sum_cubed / n_pixels as f64) as f32
    };

    // Normalise bins
    let n_float = n_pixels as f32;
    let mut norm_bins = [0.0f32; HIST_BINS];
    let bright_boundary = (HIST_BINS * 3) / 4;
    let dark_boundary = HIST_BINS / 4;
    let mut bright_mass = 0.0f32;
    let mut dark_mass = 0.0f32;

    for (i, &count) in bins.iter().enumerate() {
        let norm = count as f32 / n_float;
        norm_bins[i] = norm;
        if i >= bright_boundary {
            bright_mass += norm;
        }
        if i < dark_boundary {
            dark_mass += norm;
        }
    }

    Ok(LuminanceHistogram {
        bins: norm_bins,
        mean,
        variance,
        skewness,
        bright_mass,
        dark_mass,
    })
}

/// Extract a compact per-channel histogram feature vector from an RGB frame.
///
/// Returns a vector of statistics useful for calibration: per-channel means,
/// variances, and the overall luminance statistics.
pub fn extract_channel_statistics(frame: &FrameBuffer) -> ShotResult<ChannelStatistics> {
    let (h, w, c) = frame.dim();
    if c < 3 {
        return Err(ShotError::InvalidFrame(
            "Frame must have at least 3 channels".to_string(),
        ));
    }

    let n_pixels = (h * w) as f64;
    if n_pixels < 1.0 {
        return Err(ShotError::InvalidFrame("Empty frame".to_string()));
    }

    let mut sum_r = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sum_r2 = 0.0f64;
    let mut sum_g2 = 0.0f64;
    let mut sum_b2 = 0.0f64;

    for y in 0..h {
        for x in 0..w {
            let r = f64::from(frame.get(y, x, 0));
            let g = f64::from(frame.get(y, x, 1));
            let b = f64::from(frame.get(y, x, 2));
            sum_r += r;
            sum_g += g;
            sum_b += b;
            sum_r2 += r * r;
            sum_g2 += g * g;
            sum_b2 += b * b;
        }
    }

    let mean_r = (sum_r / n_pixels) as f32;
    let mean_g = (sum_g / n_pixels) as f32;
    let mean_b = (sum_b / n_pixels) as f32;
    let var_r = ((sum_r2 / n_pixels) - (sum_r / n_pixels).powi(2)).max(0.0) as f32;
    let var_g = ((sum_g2 / n_pixels) - (sum_g / n_pixels).powi(2)).max(0.0) as f32;
    let var_b = ((sum_b2 / n_pixels) - (sum_b / n_pixels).powi(2)).max(0.0) as f32;

    // Compute top-third vs bottom-third luminance difference (vertical gradient cue)
    let third_h = h / 3;
    let mut sum_top = 0.0f64;
    let mut sum_bottom = 0.0f64;
    let n_band = (third_h * w) as f64;

    for y in 0..third_h {
        for x in 0..w {
            let r = f64::from(frame.get(y, x, 0));
            let g = f64::from(frame.get(y, x, 1));
            let b = f64::from(frame.get(y, x, 2));
            sum_top += r * 0.299 + g * 0.587 + b * 0.114;
        }
    }
    for y in (h.saturating_sub(third_h))..h {
        for x in 0..w {
            let r = f64::from(frame.get(y, x, 0));
            let g = f64::from(frame.get(y, x, 1));
            let b = f64::from(frame.get(y, x, 2));
            sum_bottom += r * 0.299 + g * 0.587 + b * 0.114;
        }
    }

    let vertical_gradient = if n_band > 0.0 {
        ((sum_top - sum_bottom) / (n_band * 255.0)) as f32
    } else {
        0.0
    };

    Ok(ChannelStatistics {
        mean_r,
        mean_g,
        mean_b,
        var_r,
        var_g,
        var_b,
        vertical_gradient,
    })
}

/// Per-channel statistics for calibration feature extraction.
#[derive(Debug, Clone, Copy)]
pub struct ChannelStatistics {
    /// Mean red channel (0–255).
    pub mean_r: f32,
    /// Mean green channel (0–255).
    pub mean_g: f32,
    /// Mean blue channel (0–255).
    pub mean_b: f32,
    /// Variance of red channel.
    pub var_r: f32,
    /// Variance of green channel.
    pub var_g: f32,
    /// Variance of blue channel.
    pub var_b: f32,
    /// Normalised luminance difference between top and bottom thirds of frame.
    /// Positive = top is brighter (sky / high angle), negative = bottom brighter.
    pub vertical_gradient: f32,
}

// ---------------------------------------------------------------------------
// Calibrator
// ---------------------------------------------------------------------------

/// Histogram-based confidence calibrator for camera angle classification.
///
/// Maintains per-class sigmoid calibration parameters and reference histograms.
/// Can be updated incrementally with new training observations.
#[derive(Debug)]
pub struct ConfidenceCalibrator {
    /// Per-class state (indexed by `angle_to_index`).
    states: [ClassCalibrationState; N_CLASSES],
    /// Modulation weight: how strongly histogram similarity adjusts calibrated score.
    histogram_weight: f32,
}

impl Default for ConfidenceCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceCalibrator {
    /// Create a new calibrator with default (uncalibrated) parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            states: [
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
            ],
            histogram_weight: 0.2,
        }
    }

    /// Create a calibrator with a custom histogram modulation weight.
    ///
    /// `histogram_weight` must be in [0.0, 1.0]; 0.0 disables histogram modulation.
    pub fn with_histogram_weight(weight: f32) -> ShotResult<Self> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(ShotError::InvalidParameters(
                "histogram_weight must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(Self {
            states: [
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
                ClassCalibrationState::new(),
            ],
            histogram_weight: weight,
        })
    }

    /// Update calibration state with a new observation.
    ///
    /// `raw_confidence` is the score produced by the upstream classifier for class
    /// `angle`. The histogram is compared against the stored reference histogram for
    /// that class and the reference is updated as a running mean.
    pub fn update(&mut self, sample: &CalibrationSample) {
        let idx = angle_to_index(sample.angle);
        let state = &mut self.states[idx];

        // Update running mean scores
        state.pos_count += 1;
        let n = state.pos_count as f32;
        state.pos_mean_score += (sample.raw_confidence - state.pos_mean_score) / n;

        // Update reference histogram (online mean)
        match &mut state.reference_histogram {
            None => {
                state.reference_histogram = Some(sample.histogram.clone());
            }
            Some(ref_hist) => {
                for i in 0..HIST_BINS {
                    ref_hist.bins[i] += (sample.histogram.bins[i] - ref_hist.bins[i]) / n;
                }
                ref_hist.mean += (sample.histogram.mean - ref_hist.mean) / n;
                ref_hist.variance += (sample.histogram.variance - ref_hist.variance) / n;
                ref_hist.skewness += (sample.histogram.skewness - ref_hist.skewness) / n;
                ref_hist.bright_mass += (sample.histogram.bright_mass - ref_hist.bright_mass) / n;
                ref_hist.dark_mass += (sample.histogram.dark_mass - ref_hist.dark_mass) / n;
            }
        }

        // Refit sigmoid if we have enough data
        if state.is_reliable() {
            self.refit_sigmoid(idx);
        }
    }

    /// Record a *negative* observation: `raw_confidence` was assigned to class `angle`
    /// but the true label was different. This shifts the sigmoid bias.
    pub fn update_negative(&mut self, angle: CameraAngle, raw_confidence: f32) {
        let idx = angle_to_index(angle);
        let state = &mut self.states[idx];
        state.neg_count += 1;
        let n = state.neg_count as f32;
        state.neg_mean_score += (raw_confidence - state.neg_mean_score) / n;
        if state.is_reliable() {
            self.refit_sigmoid(idx);
        }
    }

    /// Calibrate a raw confidence score for the given angle class.
    ///
    /// Applies the sigmoid calibration and optionally modulates by histogram
    /// similarity if a reference histogram is available.
    pub fn calibrate(
        &self,
        angle: CameraAngle,
        raw_confidence: f32,
        frame: &FrameBuffer,
    ) -> ShotResult<CalibratedAngle> {
        let histogram = extract_luminance_histogram(frame)?;
        self.calibrate_with_histogram(angle, raw_confidence, &histogram)
    }

    /// Calibrate using a pre-computed histogram (avoids redundant extraction).
    pub fn calibrate_with_histogram(
        &self,
        angle: CameraAngle,
        raw_confidence: f32,
        histogram: &LuminanceHistogram,
    ) -> ShotResult<CalibratedAngle> {
        let idx = angle_to_index(angle);
        let state = &self.states[idx];

        // Sigmoid calibration
        let raw_clamped = raw_confidence.clamp(0.0, 1.0);
        let logit = state.slope * raw_clamped + state.bias;
        let calibrated = sigmoid(logit);

        // Histogram similarity modulation
        let (histogram_similarity, is_reliable) = match &state.reference_histogram {
            None => (1.0f32, false),
            Some(ref_hist) => {
                let chi2 = histogram.chi_squared_distance(ref_hist);
                // chi2 = 0 → similarity 1.0; chi2 large → similarity approaches 0
                let sim = (-chi2 * 2.0).exp();
                (sim, state.is_reliable())
            }
        };

        // Blend calibrated score with histogram similarity
        let modulated = calibrated * (1.0 - self.histogram_weight)
            + calibrated * histogram_similarity * self.histogram_weight;
        let final_confidence = modulated.clamp(0.0, 1.0);

        Ok(CalibratedAngle {
            angle,
            raw_confidence,
            calibrated_confidence: final_confidence,
            histogram_similarity,
            is_reliable,
        })
    }

    /// Return the current sigmoid parameters `(slope, bias)` for the given angle.
    #[must_use]
    pub fn sigmoid_params(&self, angle: CameraAngle) -> (f32, f32) {
        let state = &self.states[angle_to_index(angle)];
        (state.slope, state.bias)
    }

    /// Return the reference histogram for the given angle, if available.
    #[must_use]
    pub fn reference_histogram(&self, angle: CameraAngle) -> Option<&LuminanceHistogram> {
        self.states[angle_to_index(angle)]
            .reference_histogram
            .as_ref()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Refit the Platt sigmoid for class `idx` using stored mean scores.
    ///
    /// This uses the closed-form approximation from Platt (1999):
    /// slope = log(pos_mean / (1-pos_mean)) - log(neg_mean / (1-neg_mean)) / (pos_mean - neg_mean)
    /// But we use a simplified numerical version for stability.
    fn refit_sigmoid(&mut self, idx: usize) {
        let state = &self.states[idx];
        let pos = state.pos_mean_score.clamp(0.01, 0.99);
        let neg = state.neg_mean_score.clamp(0.01, 0.99);

        if (pos - neg).abs() < 1e-4 {
            return; // Not enough separation to refit
        }

        // Target logits for positive and negative means
        let logit_pos = (pos / (1.0 - pos)).ln();
        let logit_neg = (neg / (1.0 - neg)).ln();

        // Fit: logit = slope * raw + bias
        // Two-point solve: slope = (logit_pos - logit_neg) / (pos - neg)
        //                  bias  = logit_pos - slope * pos
        let slope = (logit_pos - logit_neg) / (pos - neg);
        let bias = logit_pos - slope * pos;

        let state_mut = &mut self.states[idx];
        state_mut.slope = slope.clamp(-10.0, 10.0);
        state_mut.bias = bias.clamp(-5.0, 5.0);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Map a `CameraAngle` to a 0-based array index.
#[inline]
fn angle_to_index(angle: CameraAngle) -> usize {
    match angle {
        CameraAngle::EyeLevel => 0,
        CameraAngle::High => 1,
        CameraAngle::Low => 2,
        CameraAngle::BirdsEye => 3,
        CameraAngle::Dutch | CameraAngle::Unknown => 4,
    }
}

/// Standard logistic sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Convert a `GrayImage` to a `LuminanceHistogram`.
pub fn histogram_from_gray(gray: &GrayImage) -> LuminanceHistogram {
    let (h, w) = gray.dim();
    let n_pixels = h * w;
    let bin_width = 256.0f32 / HIST_BINS as f32;
    let mut bins = [0u32; HIST_BINS];
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut values = Vec::with_capacity(n_pixels);

    for y in 0..h {
        for x in 0..w {
            let v = f32::from(gray.get(y, x));
            let idx = ((v / bin_width) as usize).min(HIST_BINS - 1);
            bins[idx] += 1;
            sum += f64::from(v);
            sum_sq += f64::from(v) * f64::from(v);
            values.push(v);
        }
    }

    let n = n_pixels.max(1) as f64;
    let mean = (sum / n) as f32;
    let variance = ((sum_sq / n) - (sum / n).powi(2)).max(0.0) as f32;
    let std_dev = variance.sqrt();

    let skewness = if std_dev < f32::EPSILON || values.is_empty() {
        0.0
    } else {
        let s: f64 = values
            .iter()
            .map(|&v| ((f64::from(v) - f64::from(mean)) / f64::from(std_dev)).powi(3))
            .sum();
        (s / n) as f32
    };

    let n_f = n_pixels.max(1) as f32;
    let mut norm_bins = [0.0f32; HIST_BINS];
    let bright_boundary = (HIST_BINS * 3) / 4;
    let dark_boundary = HIST_BINS / 4;
    let mut bright_mass = 0.0f32;
    let mut dark_mass = 0.0f32;

    for (i, &count) in bins.iter().enumerate() {
        let norm = count as f32 / n_f;
        norm_bins[i] = norm;
        if i >= bright_boundary {
            bright_mass += norm;
        }
        if i < dark_boundary {
            dark_mass += norm;
        }
    }

    LuminanceHistogram {
        bins: norm_bins,
        mean,
        variance,
        skewness,
        bright_mass,
        dark_mass,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(h: usize, w: usize, r: u8, g: u8, b: u8) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

    #[test]
    fn test_extract_luminance_histogram_uniform() {
        let frame = make_frame(20, 20, 128, 128, 128);
        let hist = extract_luminance_histogram(&frame).expect("histogram extraction failed");
        // All mass in one bin
        let non_zero: Vec<_> = hist.bins.iter().filter(|&&v| v > 0.0).collect();
        assert_eq!(non_zero.len(), 1);
        assert!((hist.bins.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_extract_luminance_histogram_mean() {
        // All channels = 0 → luminance = 0
        let frame = make_frame(10, 10, 0, 0, 0);
        let hist = extract_luminance_histogram(&frame).expect("histogram extraction failed");
        assert!(hist.mean < 1.0);
        assert!(hist.dark_mass > 0.9);
    }

    #[test]
    fn test_histogram_chi_squared_self_distance() {
        let frame = make_frame(20, 20, 200, 100, 50);
        let hist = extract_luminance_histogram(&frame).expect("ok");
        let dist = hist.chi_squared_distance(&hist.clone());
        assert!(dist.abs() < 1e-5);
    }

    #[test]
    fn test_histogram_bhattacharyya_self() {
        let frame = make_frame(20, 20, 200, 100, 50);
        let hist = extract_luminance_histogram(&frame).expect("ok");
        let dist = hist.bhattacharyya_distance(&hist.clone());
        // Self-distance should be 0 (or very close)
        assert!(dist.abs() < 1e-4, "Self Bhattacharyya distance = {dist}");
    }

    #[test]
    fn test_calibrator_default_sigmoid() {
        let cal = ConfidenceCalibrator::new();
        let frame = make_frame(10, 10, 100, 100, 100);
        let result = cal
            .calibrate(CameraAngle::EyeLevel, 0.7, &frame)
            .expect("calibrate ok");
        // Default slope=1, bias=0 → sigmoid(0.7) ≈ 0.668
        assert!(result.calibrated_confidence > 0.0);
        assert!(result.calibrated_confidence <= 1.0);
        assert!(!result.is_reliable);
    }

    #[test]
    fn test_calibrator_update_positive() {
        let mut cal = ConfidenceCalibrator::new();
        let frame = make_frame(20, 20, 180, 90, 60);
        for _ in 0..25 {
            let hist = extract_luminance_histogram(&frame).expect("ok");
            cal.update(&CalibrationSample {
                angle: CameraAngle::High,
                raw_confidence: 0.8,
                histogram: hist,
            });
        }
        let (slope, bias) = cal.sigmoid_params(CameraAngle::High);
        // After many positive-only samples pos_mean ≈ 0.8, neg_mean ≈ 0.5 (default)
        // check that the sigmoid parameters have been updated
        assert!(slope.is_finite());
        assert!(bias.is_finite());
    }

    #[test]
    fn test_calibrator_reliable_after_min_samples() {
        let mut cal = ConfidenceCalibrator::new();
        let frame = make_frame(20, 20, 60, 60, 200);
        let hist = extract_luminance_histogram(&frame).expect("ok");
        for i in 0..MIN_RELIABLE_SAMPLES {
            let raw = 0.5 + (i as f32 * 0.01);
            cal.update(&CalibrationSample {
                angle: CameraAngle::Low,
                raw_confidence: raw,
                histogram: hist.clone(),
            });
        }
        let result = cal.calibrate(CameraAngle::Low, 0.6, &frame).expect("ok");
        assert!(result.is_reliable);
    }

    #[test]
    fn test_calibrate_with_histogram_modulation() {
        let cal = ConfidenceCalibrator::new();
        let frame = make_frame(10, 10, 100, 150, 80);
        let hist = extract_luminance_histogram(&frame).expect("ok");
        let result = cal
            .calibrate_with_histogram(CameraAngle::Dutch, 0.5, &hist)
            .expect("ok");
        assert!(result.histogram_similarity >= 0.0);
        assert!(result.histogram_similarity <= 1.0);
    }

    #[test]
    fn test_histogram_weight_validation() {
        assert!(ConfidenceCalibrator::with_histogram_weight(0.5).is_ok());
        assert!(ConfidenceCalibrator::with_histogram_weight(1.5).is_err());
        assert!(ConfidenceCalibrator::with_histogram_weight(-0.1).is_err());
    }

    #[test]
    fn test_channel_statistics_extraction() {
        let frame = make_frame(20, 20, 255, 0, 0);
        let stats = extract_channel_statistics(&frame).expect("ok");
        assert!((stats.mean_r - 255.0).abs() < 1.0);
        assert!(stats.mean_g < 1.0);
        assert!(stats.mean_b < 1.0);
    }

    #[test]
    fn test_histogram_from_gray() {
        let mut gray = GrayImage::zeros(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                gray.set(y, x, 200);
            }
        }
        let hist = histogram_from_gray(&gray);
        assert!((hist.bins.iter().sum::<f32>() - 1.0).abs() < 0.01);
        assert!(hist.bright_mass > 0.9);
    }

    #[test]
    fn test_all_angle_classes_calibrate() {
        let cal = ConfidenceCalibrator::new();
        let frame = make_frame(10, 10, 120, 120, 120);
        let angles = [
            CameraAngle::EyeLevel,
            CameraAngle::High,
            CameraAngle::Low,
            CameraAngle::BirdsEye,
            CameraAngle::Dutch,
        ];
        for angle in angles {
            let result = cal.calibrate(angle, 0.6, &frame).expect("ok");
            assert!(result.calibrated_confidence >= 0.0);
            assert!(result.calibrated_confidence <= 1.0);
        }
    }
}
