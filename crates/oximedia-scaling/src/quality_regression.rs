//! PSNR and SSIM quality metrics for regression testing scaling output.
//!
//! This module provides reference implementations of two widely-used
//! full-reference image quality metrics:
//!
//! * **PSNR** (Peak Signal-to-Noise Ratio) — measures pixel-level fidelity.
//! * **SSIM** (Structural Similarity Index) — measures perceptual similarity.
//!
//! Both metrics compare a *distorted* image against a *reference* image of
//! the same dimensions.  These are intended primarily for regression tests:
//! verify that scaling algorithm changes do not degrade quality below a
//! known threshold.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::quality_regression::{psnr, ssim};
//!
//! let reference = vec![128u8; 64];
//! let distorted = vec![130u8; 64];
//! let p = psnr(&reference, &distorted, 8, 8).expect("ok");
//! assert!(p > 30.0, "PSNR should be high for near-identical images");
//! let s = ssim(&reference, &distorted, 8, 8).expect("ok");
//! assert!(s > 0.9, "SSIM should be near 1.0");
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from quality metric computation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum QualityError {
    /// The two images have different dimensions or buffer sizes.
    #[error("dimension mismatch: reference {ref_len} vs distorted {dist_len}")]
    DimensionMismatch {
        /// Reference buffer length.
        ref_len: usize,
        /// Distorted buffer length.
        dist_len: usize,
    },
    /// Zero-area image (width or height is 0).
    #[error("zero dimension: {width}x{height}")]
    ZeroDimension {
        /// Image width.
        width: usize,
        /// Image height.
        height: usize,
    },
    /// Images are too small for SSIM windowed computation.
    #[error("image too small for SSIM window: need at least {min_size}x{min_size}")]
    TooSmallForSsim {
        /// Minimum required dimension.
        min_size: usize,
    },
}

// ---------------------------------------------------------------------------
// PSNR
// ---------------------------------------------------------------------------

/// Compute the Peak Signal-to-Noise Ratio between two 8-bit greyscale images.
///
/// Returns the PSNR in dB.  Identical images produce `f64::INFINITY`.
///
/// Formula: `PSNR = 10 · log₁₀(MAX² / MSE)` where `MAX = 255`.
pub fn psnr(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64, QualityError> {
    validate(reference, distorted, width, height)?;
    let n = width * height;
    let mse: f64 = reference
        .iter()
        .zip(distorted.iter())
        .take(n)
        .map(|(&r, &d)| {
            let diff = r as f64 - d as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;

    if mse < f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (255.0_f64 * 255.0 / mse).log10())
}

/// Compute PSNR on `f32` images normalised to `[0, 1]`.
///
/// `MAX = 1.0`.
pub fn psnr_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
) -> Result<f64, QualityError> {
    validate_f32(reference, distorted, width, height)?;
    let n = width * height;
    let mse: f64 = reference
        .iter()
        .zip(distorted.iter())
        .take(n)
        .map(|(&r, &d)| {
            let diff = r as f64 - d as f64;
            diff * diff
        })
        .sum::<f64>()
        / n as f64;

    if mse < f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (1.0 / mse).log10())
}

// ---------------------------------------------------------------------------
// SSIM
// ---------------------------------------------------------------------------

/// SSIM stabilisation constants (for 8-bit images, `L = 255`).
const K1: f64 = 0.01;
const K2: f64 = 0.03;
const L: f64 = 255.0;
const C1: f64 = (K1 * L) * (K1 * L);
const C2: f64 = (K2 * L) * (K2 * L);

/// SSIM window radius.
const SSIM_WINDOW_RADIUS: usize = 3;
/// SSIM window diameter.
const SSIM_WINDOW_SIZE: usize = SSIM_WINDOW_RADIUS * 2 + 1;

/// Compute the mean SSIM (Structural Similarity Index) between two 8-bit
/// greyscale images using a sliding window of 7×7 with uniform weighting.
///
/// Returns a value in `[−1, 1]` (typically `[0, 1]` for natural images).
/// Identical images return `1.0`.
pub fn ssim(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<f64, QualityError> {
    validate(reference, distorted, width, height)?;
    if width < SSIM_WINDOW_SIZE || height < SSIM_WINDOW_SIZE {
        return Err(QualityError::TooSmallForSsim {
            min_size: SSIM_WINDOW_SIZE,
        });
    }

    let mut sum = 0.0f64;
    let mut count = 0u64;

    let y_end = height - SSIM_WINDOW_RADIUS;
    let x_end = width - SSIM_WINDOW_RADIUS;

    for wy in SSIM_WINDOW_RADIUS..y_end {
        for wx in SSIM_WINDOW_RADIUS..x_end {
            let (mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy) =
                window_stats(reference, distorted, width, wx, wy);

            let num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2);
            let den = (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x2 + sigma_y2 + C2);
            sum += num / den;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(1.0);
    }
    Ok(sum / count as f64)
}

/// Compute local window statistics for SSIM.
fn window_stats(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    cx: usize,
    cy: usize,
) -> (f64, f64, f64, f64, f64) {
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    let mut sum_xy = 0.0f64;
    let n = (SSIM_WINDOW_SIZE * SSIM_WINDOW_SIZE) as f64;

    for dy in 0..SSIM_WINDOW_SIZE {
        let row = (cy + dy).saturating_sub(SSIM_WINDOW_RADIUS);
        for dx in 0..SSIM_WINDOW_SIZE {
            let col = (cx + dx).saturating_sub(SSIM_WINDOW_RADIUS);
            let x = reference[row * width + col] as f64;
            let y = distorted[row * width + col] as f64;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_xy += x * y;
        }
    }

    let mu_x = sum_x / n;
    let mu_y = sum_y / n;
    let sigma_x2 = (sum_x2 / n) - mu_x * mu_x;
    let sigma_y2 = (sum_y2 / n) - mu_y * mu_y;
    let sigma_xy = (sum_xy / n) - mu_x * mu_y;

    (mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy)
}

// ---------------------------------------------------------------------------
// Round-trip quality check
// ---------------------------------------------------------------------------

/// Perform a scale-down then scale-up round-trip and measure PSNR degradation.
///
/// Returns the PSNR between the original and the round-tripped image.
/// This is useful for regression testing: ensuring a given scaling algorithm
/// produces a minimum quality after a round-trip.
///
/// The scaling is done with nearest-neighbor for simplicity (this module
/// focuses on the metric, not the scaler).
pub fn roundtrip_psnr(
    image: &[u8],
    width: usize,
    height: usize,
    half_w: usize,
    half_h: usize,
) -> Result<f64, QualityError> {
    if width == 0 || height == 0 {
        return Err(QualityError::ZeroDimension { width, height });
    }
    if half_w == 0 || half_h == 0 {
        return Err(QualityError::ZeroDimension {
            width: half_w,
            height: half_h,
        });
    }
    let n = width * height;
    if image.len() < n {
        return Err(QualityError::DimensionMismatch {
            ref_len: n,
            dist_len: image.len(),
        });
    }

    // Downscale
    let mut small = vec![0u8; half_w * half_h];
    for dy in 0..half_h {
        let sy = (dy * height / half_h).min(height - 1);
        for dx in 0..half_w {
            let sx = (dx * width / half_w).min(width - 1);
            small[dy * half_w + dx] = image[sy * width + sx];
        }
    }

    // Upscale back
    let mut restored = vec![0u8; n];
    for dy in 0..height {
        let sy = (dy * half_h / height).min(half_h - 1);
        for dx in 0..width {
            let sx = (dx * half_w / width).min(half_w - 1);
            restored[dy * width + dx] = small[sy * half_w + sx];
        }
    }

    psnr(image, &restored, width, height)
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
) -> Result<(), QualityError> {
    if width == 0 || height == 0 {
        return Err(QualityError::ZeroDimension { width, height });
    }
    let n = width * height;
    if reference.len() < n || distorted.len() < n {
        return Err(QualityError::DimensionMismatch {
            ref_len: reference.len(),
            dist_len: distorted.len(),
        });
    }
    Ok(())
}

fn validate_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
) -> Result<(), QualityError> {
    if width == 0 || height == 0 {
        return Err(QualityError::ZeroDimension { width, height });
    }
    let n = width * height;
    if reference.len() < n || distorted.len() < n {
        return Err(QualityError::DimensionMismatch {
            ref_len: reference.len(),
            dist_len: distorted.len(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical_images() {
        let img = vec![128u8; 8 * 8];
        let p = psnr(&img, &img, 8, 8).expect("ok");
        assert!(
            p.is_infinite(),
            "identical images should give infinite PSNR"
        );
    }

    #[test]
    fn test_psnr_slight_difference() {
        let reference = vec![100u8; 16 * 16];
        let distorted = vec![102u8; 16 * 16];
        let p = psnr(&reference, &distorted, 16, 16).expect("ok");
        assert!(
            p > 40.0,
            "very similar images should have PSNR > 40 dB, got {p}"
        );
    }

    #[test]
    fn test_psnr_large_difference() {
        let reference = vec![0u8; 8 * 8];
        let distorted = vec![255u8; 8 * 8];
        let p = psnr(&reference, &distorted, 8, 8).expect("ok");
        // MSE = 255^2 = 65025, PSNR = 10·log10(255²/65025) = 10·log10(1) = 0
        assert!(
            (p - 0.0).abs() < 1e-6,
            "max difference should give 0 dB PSNR, got {p}"
        );
    }

    #[test]
    fn test_psnr_zero_dimension_error() {
        let result = psnr(&[1], &[1], 0, 1);
        assert!(matches!(result, Err(QualityError::ZeroDimension { .. })));
    }

    #[test]
    fn test_psnr_dimension_mismatch_error() {
        let result = psnr(&[1, 2], &[1], 2, 2);
        assert!(matches!(
            result,
            Err(QualityError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_psnr_f32_identical() {
        let img = vec![0.5f32; 4 * 4];
        let p = psnr_f32(&img, &img, 4, 4).expect("ok");
        assert!(p.is_infinite());
    }

    #[test]
    fn test_psnr_f32_slight_diff() {
        let reference = vec![0.5f32; 8 * 8];
        let distorted = vec![0.51f32; 8 * 8];
        let p = psnr_f32(&reference, &distorted, 8, 8).expect("ok");
        assert!(
            p > 30.0,
            "near-identical f32 images should have high PSNR, got {p}"
        );
    }

    #[test]
    fn test_ssim_identical() {
        let img = vec![128u8; 16 * 16];
        let s = ssim(&img, &img, 16, 16).expect("ok");
        assert!(
            (s - 1.0).abs() < 1e-6,
            "identical images should have SSIM = 1.0, got {s}"
        );
    }

    #[test]
    fn test_ssim_similar() {
        let reference = vec![100u8; 16 * 16];
        let distorted = vec![102u8; 16 * 16];
        let s = ssim(&reference, &distorted, 16, 16).expect("ok");
        assert!(
            s > 0.95,
            "very similar images should have SSIM > 0.95, got {s}"
        );
    }

    #[test]
    fn test_ssim_too_small_error() {
        let img = vec![128u8; 4 * 4];
        let result = ssim(&img, &img, 4, 4);
        assert!(matches!(result, Err(QualityError::TooSmallForSsim { .. })));
    }

    #[test]
    fn test_ssim_dissimilar() {
        let reference = vec![0u8; 16 * 16];
        let distorted = vec![255u8; 16 * 16];
        let s = ssim(&reference, &distorted, 16, 16).expect("ok");
        assert!(
            s < 0.1,
            "maximally different images should have low SSIM, got {s}"
        );
    }

    #[test]
    fn test_roundtrip_psnr_basic() {
        // A uniform image should survive a round-trip with infinite PSNR.
        let img = vec![100u8; 16 * 16];
        let p = roundtrip_psnr(&img, 16, 16, 8, 8).expect("ok");
        assert!(
            p.is_infinite(),
            "uniform image round-trip should be lossless, got {p}"
        );
    }

    #[test]
    fn test_roundtrip_psnr_gradient() {
        // A gradient will lose quality after round-trip.
        let img: Vec<u8> = (0..16 * 16).map(|i| (i % 256) as u8).collect();
        let p = roundtrip_psnr(&img, 16, 16, 8, 8).expect("ok");
        assert!(
            p > 5.0 && p < 100.0,
            "gradient round-trip PSNR should be moderate, got {p}"
        );
    }

    #[test]
    fn test_roundtrip_psnr_zero_dim() {
        let result = roundtrip_psnr(&[1], 0, 1, 1, 1);
        assert!(matches!(result, Err(QualityError::ZeroDimension { .. })));
    }

    #[test]
    fn test_error_display_messages() {
        let e1 = QualityError::DimensionMismatch {
            ref_len: 10,
            dist_len: 5,
        };
        assert!(e1.to_string().contains("10"));
        assert!(e1.to_string().contains("5"));

        let e2 = QualityError::ZeroDimension {
            width: 0,
            height: 10,
        };
        assert!(e2.to_string().contains("0x10"));

        let e3 = QualityError::TooSmallForSsim { min_size: 7 };
        assert!(e3.to_string().contains("7"));
    }

    #[test]
    fn test_psnr_symmetry() {
        let a = vec![100u8; 8 * 8];
        let b = vec![110u8; 8 * 8];
        let p1 = psnr(&a, &b, 8, 8).expect("ok");
        let p2 = psnr(&b, &a, 8, 8).expect("ok");
        assert!(
            (p1 - p2).abs() < 1e-10,
            "PSNR should be symmetric: {p1} vs {p2}"
        );
    }
}
