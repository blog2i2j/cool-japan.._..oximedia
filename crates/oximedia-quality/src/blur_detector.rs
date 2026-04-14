//! Advanced blur and sharpness measurement.
//!
//! Provides three complementary sharpness estimators:
//!
//! | Method | What it measures | Best for |
//! |--------|-----------------|----------|
//! | Laplacian variance | High-frequency content | General-purpose sharpness |
//! | Tenengrad | Sobel gradient energy | Edge-focussed sharpness |
//! | BRISQUE-lite | Normalised contrast statistics | Perceptual sharpness |
//!
//! A combined score (weighted average of all three) is also available.
//!
//! # Score interpretation
//!
//! All methods return **higher scores for sharper images**.  A combined score
//! of 0 corresponds to a completely flat (uniform) frame.

use crate::{Frame, MetricType, QualityScore};
use oximedia_core::{OxiError, OxiResult};
use serde::{Deserialize, Serialize};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Minimum frame dimension (pixels) for meaningful sharpness analysis.
const MIN_DIM: usize = 8;

/// Sobel edge threshold below which pixels are ignored by the Tenengrad method.
const TENENGRAD_THRESHOLD: f64 = 10.0;

// ── Method selector ───────────────────────────────────────────────────────────

/// Sharpness measurement method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharpnessMethod {
    /// Variance of the Laplacian (high-frequency content).
    LaplacianVariance,
    /// Tenengrad — variance of Sobel gradient magnitudes above a threshold.
    Tenengrad,
    /// BRISQUE-lite — normalised local contrast statistics (perceptual).
    BrisqueLite,
    /// Weighted combination of all three methods.
    Combined,
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Detailed sharpness measurement result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SharpnessResult {
    /// Overall sharpness score (higher = sharper).
    pub score: f64,
    /// Laplacian variance sub-score.
    pub laplacian: f64,
    /// Tenengrad sub-score.
    pub tenengrad: f64,
    /// BRISQUE-lite sub-score.
    pub brisque_lite: f64,
    /// Human-readable sharpness grade.
    pub grade: SharpnessGrade,
}

/// Perceptual sharpness grade.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharpnessGrade {
    /// Score ≥ 1000 — very sharp, broadcast-quality.
    VerySharp,
    /// Score 300–999 — acceptably sharp.
    Sharp,
    /// Score 50–299 — slight blur; may be acceptable for streaming.
    SlightBlur,
    /// Score 10–49 — noticeable blur.
    Blurry,
    /// Score < 10 — severe blur or nearly uniform frame.
    VeryBlurry,
}

impl SharpnessGrade {
    /// Derives a grade from a combined score.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        if score >= 1000.0 {
            Self::VerySharp
        } else if score >= 300.0 {
            Self::Sharp
        } else if score >= 50.0 {
            Self::SlightBlur
        } else if score >= 10.0 {
            Self::Blurry
        } else {
            Self::VeryBlurry
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::VerySharp => "Very Sharp",
            Self::Sharp => "Sharp",
            Self::SlightBlur => "Slight Blur",
            Self::Blurry => "Blurry",
            Self::VeryBlurry => "Very Blurry",
        }
    }
}

// ── Detector ──────────────────────────────────────────────────────────────────

/// Advanced blur / sharpness detector.
///
/// Supports multiple measurement methods; defaults to `SharpnessMethod::Combined`.
pub struct AdvancedBlurDetector {
    method: SharpnessMethod,
    /// Weights applied to [laplacian, tenengrad, brisque_lite] for `Combined`.
    weights: [f64; 3],
    /// Sobel magnitude threshold for Tenengrad (skip low-contrast pixels).
    tenengrad_threshold: f64,
}

impl AdvancedBlurDetector {
    /// Creates a detector with the combined method and default weights.
    #[must_use]
    pub fn new() -> Self {
        Self {
            method: SharpnessMethod::Combined,
            weights: [0.4, 0.4, 0.2],
            tenengrad_threshold: TENENGRAD_THRESHOLD,
        }
    }

    /// Creates a detector that uses a specific method.
    #[must_use]
    pub fn with_method(method: SharpnessMethod) -> Self {
        Self {
            method,
            ..Self::new()
        }
    }

    /// Sets custom combination weights for `[laplacian, tenengrad, brisque_lite]`.
    ///
    /// Weights need not sum to 1; they are normalised internally.
    #[must_use]
    pub fn with_weights(mut self, weights: [f64; 3]) -> Self {
        self.weights = weights;
        self
    }

    /// Sets the Sobel magnitude threshold for the Tenengrad method.
    #[must_use]
    pub fn with_tenengrad_threshold(mut self, threshold: f64) -> Self {
        self.tenengrad_threshold = threshold;
        self
    }

    // ── Public API ─────────────────────────────────────────────────────────────

    /// Measures sharpness of a frame, returning a detailed result.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` if the frame is smaller than 8×8.
    pub fn measure(&self, frame: &Frame) -> OxiResult<SharpnessResult> {
        let w = frame.width;
        let h = frame.height;

        if w < MIN_DIM || h < MIN_DIM {
            return Err(OxiError::InvalidData(format!(
                "Frame ({w}×{h}) must be at least {MIN_DIM}×{MIN_DIM} for sharpness analysis"
            )));
        }

        let plane = &frame.planes[0];

        let laplacian = self.laplacian_variance(plane, w, h);
        let tenengrad = self.tenengrad_score(plane, w, h);
        let brisque_lite = self.brisque_lite_score(plane, w, h);

        let score = match self.method {
            SharpnessMethod::LaplacianVariance => laplacian,
            SharpnessMethod::Tenengrad => tenengrad,
            SharpnessMethod::BrisqueLite => brisque_lite,
            SharpnessMethod::Combined => self.combine(laplacian, tenengrad, brisque_lite),
        };

        let grade = SharpnessGrade::from_score(score);

        Ok(SharpnessResult {
            score,
            laplacian,
            tenengrad,
            brisque_lite,
            grade,
        })
    }

    /// Returns a `QualityScore` compatible with the quality API.
    ///
    /// # Errors
    ///
    /// Returns `OxiError::InvalidData` if the frame is too small.
    pub fn detect(&self, frame: &Frame) -> OxiResult<QualityScore> {
        let result = self.measure(frame)?;
        let mut score = QualityScore::new(MetricType::Blur, result.score);
        score.add_component("laplacian", result.laplacian);
        score.add_component("tenengrad", result.tenengrad);
        score.add_component("brisque_lite", result.brisque_lite);
        Ok(score)
    }

    // ── Private methods ────────────────────────────────────────────────────────

    /// Variance of the Laplacian response.
    fn laplacian_variance(&self, plane: &[u8], w: usize, h: usize) -> f64 {
        // 3×3 Laplacian kernel (4-connected)
        const KERNEL: [f64; 9] = [0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];

        let n = (w - 2) * (h - 2);
        if n == 0 {
            return 0.0;
        }

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut v = 0.0_f64;
                for dy in 0..3 {
                    for dx in 0..3 {
                        v +=
                            f64::from(plane[(y + dy - 1) * w + (x + dx - 1)]) * KERNEL[dy * 3 + dx];
                    }
                }
                sum += v;
                sum_sq += v * v;
            }
        }

        let n_f = n as f64;
        let mean = sum / n_f;
        (sum_sq / n_f) - mean * mean
    }

    /// Tenengrad: mean Sobel gradient magnitude squared, edge pixels only.
    fn tenengrad_score(&self, plane: &[u8], w: usize, h: usize) -> f64 {
        const SOBEL_X: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        const SOBEL_Y: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut count = 0_u64;
        let mut total = 0.0_f64;

        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let mut gx = 0.0_f64;
                let mut gy = 0.0_f64;

                for dy in 0..3 {
                    for dx in 0..3 {
                        let val = f64::from(plane[(y + dy - 1) * w + (x + dx - 1)]);
                        let ki = dy * 3 + dx;
                        gx += val * SOBEL_X[ki];
                        gy += val * SOBEL_Y[ki];
                    }
                }

                let mag = (gx * gx + gy * gy).sqrt();
                if mag > self.tenengrad_threshold {
                    total += mag * mag;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            total / count as f64
        }
    }

    /// BRISQUE-lite: normalised local contrast — simplified version of the
    /// Locally Mean Subtracted Contrast Normalised (MSCN) coefficient
    /// variance used in BRISQUE.
    ///
    /// We compute `MSCN(x,y) = (I(x,y) - mu) / (sigma + 1)` using a
    /// 7×7 local window, then return the variance of those coefficients.
    fn brisque_lite_score(&self, plane: &[u8], w: usize, h: usize) -> f64 {
        const HALF: usize = 3; // half window = 7/2
        if w < HALF * 2 + 1 || h < HALF * 2 + 1 {
            return 0.0;
        }

        let win = (2 * HALF + 1) as f64;
        let win_area = win * win;

        let mut mscn_sum = 0.0_f64;
        let mut mscn_sq_sum = 0.0_f64;
        let mut pixel_count = 0_u64;

        for y in HALF..h - HALF {
            for x in HALF..w - HALF {
                // Local mean
                let mut mu = 0.0_f64;
                for dy in 0..2 * HALF + 1 {
                    for dx in 0..2 * HALF + 1 {
                        mu += f64::from(plane[(y + dy - HALF) * w + (x + dx - HALF)]);
                    }
                }
                mu /= win_area;

                // Local standard deviation
                let mut sigma2 = 0.0_f64;
                for dy in 0..2 * HALF + 1 {
                    for dx in 0..2 * HALF + 1 {
                        let diff = f64::from(plane[(y + dy - HALF) * w + (x + dx - HALF)]) - mu;
                        sigma2 += diff * diff;
                    }
                }
                sigma2 /= win_area;
                let sigma = sigma2.sqrt();

                let mscn = (f64::from(plane[y * w + x]) - mu) / (sigma + 1.0);
                mscn_sum += mscn;
                mscn_sq_sum += mscn * mscn;
                pixel_count += 1;
            }
        }

        if pixel_count == 0 {
            return 0.0;
        }

        let n = pixel_count as f64;
        let mean = mscn_sum / n;
        // Variance of MSCN coefficients: higher variance → more texture → sharper.
        (mscn_sq_sum / n) - mean * mean
    }

    /// Weighted combination of the three sub-scores.
    fn combine(&self, laplacian: f64, tenengrad: f64, brisque_lite: f64) -> f64 {
        let [w0, w1, w2] = self.weights;
        let total_w = w0 + w1 + w2;
        if total_w < 1e-12 {
            return 0.0;
        }
        // Normalise each sub-score to a comparable range before combining.
        // Laplacian variance can be very large; we scale it down.
        let lap_norm = laplacian.sqrt();
        let ten_norm = tenengrad.sqrt();
        let bri_norm = brisque_lite * 100.0;

        (w0 * lap_norm + w1 * ten_norm + w2 * bri_norm) / total_w
    }
}

impl Default for AdvancedBlurDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    fn uniform_frame(width: usize, height: usize, value: u8) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        frame.planes[0].fill(value);
        frame
    }

    fn checkerboard_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        for y in 0..height {
            for x in 0..width {
                frame.planes[0][y * width + x] = if (x + y) % 2 == 0 { 0 } else { 255 };
            }
        }
        frame
    }

    fn gradient_frame(width: usize, height: usize) -> Frame {
        let mut frame =
            Frame::new(width, height, PixelFormat::Yuv420p).expect("frame creation should succeed");
        for y in 0..height {
            for x in 0..width {
                frame.planes[0][y * width + x] = ((x * 255) / width.max(1)) as u8;
            }
        }
        frame
    }

    #[test]
    fn test_uniform_frame_scores_zero() {
        let detector = AdvancedBlurDetector::new();
        let frame = uniform_frame(64, 64, 128);
        let result = detector.measure(&frame).expect("measure should succeed");
        // Flat frame: Laplacian response is 0 everywhere → score ~ 0.
        assert!(result.laplacian < 1e-6, "laplacian={}", result.laplacian);
    }

    #[test]
    fn test_checkerboard_is_sharper_than_gradient() {
        let detector = AdvancedBlurDetector::new();
        let sharp = checkerboard_frame(64, 64);
        let smooth = gradient_frame(64, 64);

        let sharp_result = detector.measure(&sharp).expect("measure should succeed");
        let smooth_result = detector.measure(&smooth).expect("measure should succeed");

        assert!(
            sharp_result.score > smooth_result.score,
            "sharp={} smooth={}",
            sharp_result.score,
            smooth_result.score
        );
    }

    #[test]
    fn test_laplacian_method_only() {
        let detector = AdvancedBlurDetector::with_method(SharpnessMethod::LaplacianVariance);
        let frame = checkerboard_frame(64, 64);
        let result = detector.measure(&frame).expect("measure should succeed");
        // Score for LaplacianVariance should equal the laplacian sub-score.
        assert!((result.score - result.laplacian).abs() < 1e-9);
    }

    #[test]
    fn test_tenengrad_method_only() {
        let detector = AdvancedBlurDetector::with_method(SharpnessMethod::Tenengrad);
        let frame = checkerboard_frame(64, 64);
        let result = detector.measure(&frame).expect("measure should succeed");
        assert!((result.score - result.tenengrad).abs() < 1e-9);
    }

    #[test]
    fn test_brisque_lite_method_only() {
        let detector = AdvancedBlurDetector::with_method(SharpnessMethod::BrisqueLite);
        let frame = checkerboard_frame(64, 64);
        let result = detector.measure(&frame).expect("measure should succeed");
        assert!((result.score - result.brisque_lite).abs() < 1e-9);
    }

    #[test]
    fn test_too_small_frame_returns_error() {
        let detector = AdvancedBlurDetector::new();
        let small = uniform_frame(4, 4, 128);
        assert!(
            detector.measure(&small).is_err(),
            "Expected error for 4×4 frame"
        );
    }

    #[test]
    fn test_detect_returns_quality_score_with_components() {
        let detector = AdvancedBlurDetector::new();
        let frame = checkerboard_frame(64, 64);
        let score = detector.detect(&frame).expect("detect should succeed");
        assert!(score.score >= 0.0);
        assert!(score.components.contains_key("laplacian"));
        assert!(score.components.contains_key("tenengrad"));
        assert!(score.components.contains_key("brisque_lite"));
    }

    #[test]
    fn test_sharpness_grade_thresholds() {
        assert_eq!(
            SharpnessGrade::from_score(2000.0),
            SharpnessGrade::VerySharp
        );
        assert_eq!(SharpnessGrade::from_score(500.0), SharpnessGrade::Sharp);
        assert_eq!(
            SharpnessGrade::from_score(100.0),
            SharpnessGrade::SlightBlur
        );
        assert_eq!(SharpnessGrade::from_score(20.0), SharpnessGrade::Blurry);
        assert_eq!(SharpnessGrade::from_score(1.0), SharpnessGrade::VeryBlurry);
    }

    #[test]
    fn test_grade_labels_non_empty() {
        for grade in &[
            SharpnessGrade::VerySharp,
            SharpnessGrade::Sharp,
            SharpnessGrade::SlightBlur,
            SharpnessGrade::Blurry,
            SharpnessGrade::VeryBlurry,
        ] {
            assert!(!grade.label().is_empty());
        }
    }

    #[test]
    fn test_custom_weights() {
        // Luma-only (Laplacian) — set tenengrad and brisque weights to zero.
        let detector = AdvancedBlurDetector::new().with_weights([1.0, 0.0, 0.0]);
        let frame = checkerboard_frame(64, 64);
        let result = detector.measure(&frame).expect("measure should succeed");
        // With weights [1,0,0] and normalisation (lap_norm = sqrt(laplacian)):
        let expected = result.laplacian.sqrt();
        assert!(
            (result.score - expected).abs() < 1e-9,
            "score={} expected={}",
            result.score,
            expected
        );
    }
}
