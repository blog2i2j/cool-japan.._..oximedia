//! Sharpness metric computation via Laplacian variance.
//!
//! The Laplacian operator is a second-order derivative filter that highlights
//! regions of rapid intensity change (edges).  The variance of the Laplacian
//! response is a robust no-reference sharpness metric: a high variance
//! indicates a sharp image with well-defined edges; a low variance indicates
//! a blurry image.
//!
//! # Reference
//!
//! Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy: a
//! comparative study", ICPR 2000 — the "variance of Laplacian" criterion.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::sharpness::laplacian_variance;
//!
//! // 4×4 greyscale image (stride = 4 bytes)
//! let img: Vec<u8> = (0u8..16).collect();
//! let score = laplacian_variance(&img, 4, 4);
//! assert!(score >= 0.0);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute the variance of the Laplacian response of a greyscale image.
///
/// # Parameters
///
/// * `img`  — flat row-major greyscale pixel buffer (`w * h` bytes).
/// * `w`    — image width in pixels.
/// * `h`    — image height in pixels.
///
/// # Returns
///
/// A non-negative float representing image sharpness.  Higher values mean
/// sharper images.  Returns `0.0` when the image is too small (< 3×3).
///
/// # Notes
///
/// Interior pixels are convolved with the discrete 3×3 Laplacian kernel:
///
/// ```text
///  0  1  0
///  1 -4  1
///  0  1  0
/// ```
///
/// Border pixels are skipped (zero-padding is avoided to prevent spurious
/// edge responses).
pub fn laplacian_variance(img: &[u8], w: u32, h: u32) -> f32 {
    let w = w as usize;
    let h = h as usize;

    // Need at least a 3×3 image to compute any interior Laplacian responses.
    if w < 3 || h < 3 || img.len() < w * h {
        return 0.0;
    }

    let n = (w - 2) * (h - 2); // number of interior pixels
    if n == 0 {
        return 0.0;
    }

    let mut sum: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;

    for row in 1..(h - 1) {
        for col in 1..(w - 1) {
            let center = img[row * w + col] as f64;
            let top = img[(row - 1) * w + col] as f64;
            let bottom = img[(row + 1) * w + col] as f64;
            let left = img[row * w + (col - 1)] as f64;
            let right = img[row * w + (col + 1)] as f64;

            // 4-connected Laplacian: top + bottom + left + right − 4·center
            let lap = top + bottom + left + right - 4.0 * center;

            sum += lap;
            sum_sq += lap * lap;
        }
    }

    let mean = sum / n as f64;
    let variance = (sum_sq / n as f64) - mean * mean;

    // Clamp to handle floating-point rounding producing tiny negatives.
    variance.max(0.0) as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All-black image → all Laplacian responses are zero → variance = 0.
    #[test]
    fn test_flat_image_zero_variance() {
        let img = vec![128u8; 16 * 16];
        let v = laplacian_variance(&img, 16, 16);
        assert!(v.abs() < 1e-5, "flat image variance should be ~0, got {v}");
    }

    /// A single sharp edge (step function) should yield a high variance.
    #[test]
    fn test_edge_image_nonzero_variance() {
        let w = 8usize;
        let h = 8usize;
        let mut img = vec![0u8; w * h];
        // Left half black, right half white
        for row in 0..h {
            for col in (w / 2)..w {
                img[row * w + col] = 255;
            }
        }
        let v = laplacian_variance(&img, w as u32, h as u32);
        assert!(v > 0.0, "step-function edge should give positive variance");
    }

    /// Images smaller than 3×3 should return 0.
    #[test]
    fn test_too_small_returns_zero() {
        assert_eq!(laplacian_variance(&[1, 2, 3, 4], 2, 2), 0.0);
        assert_eq!(laplacian_variance(&[], 0, 0), 0.0);
        assert_eq!(laplacian_variance(&[100u8; 9], 3, 3).is_nan(), false);
    }

    /// A checker-board pattern should have higher sharpness than a smooth ramp.
    #[test]
    fn test_checker_sharper_than_ramp() {
        let w = 8u32;
        let h = 8u32;

        // Checkerboard
        let checker: Vec<u8> = (0..(w * h) as usize)
            .map(|i| {
                let row = i / w as usize;
                let col = i % w as usize;
                if (row + col) % 2 == 0 {
                    0
                } else {
                    255
                }
            })
            .collect();

        // Horizontal ramp
        let ramp: Vec<u8> = (0..(w * h) as usize)
            .map(|i| {
                let col = i % w as usize;
                ((col as f32 / (w - 1) as f32) * 255.0) as u8
            })
            .collect();

        let v_checker = laplacian_variance(&checker, w, h);
        let v_ramp = laplacian_variance(&ramp, w, h);

        assert!(
            v_checker > v_ramp,
            "checker ({v_checker}) should be sharper than ramp ({v_ramp})"
        );
    }

    /// Verify that the buffer-length guard works.
    #[test]
    fn test_short_buffer_returns_zero() {
        // Buffer is shorter than w*h → guard triggers
        let short = vec![0u8; 5];
        assert_eq!(laplacian_variance(&short, 4, 4), 0.0);
    }
}
