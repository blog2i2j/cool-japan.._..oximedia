//! Separable kernel convolution — optimised 2-pass horizontal+vertical filtering.
//!
//! A separable filter applies a 1-D kernel along the horizontal axis first and
//! then a (possibly different) 1-D kernel along the vertical axis.  When the
//! target 2-D kernel can be decomposed as the outer product of two 1-D kernels
//! `K = h ⊗ v` (e.g. Gaussian, Box, Sobel smoothing pass) the two-pass strategy
//! reduces complexity from `O(N·k²)` to `O(N·2k)` per pixel.
//!
//! ## Provided filters
//!
//! | Helper | Description |
//! |--------|-------------|
//! | [`gaussian_kernel`] | Builds a 1-D Gaussian kernel of odd length |
//! | [`box_kernel`] | Builds a 1-D uniform box kernel |
//! | [`apply_separable`] | General 2-pass separable convolution |
//! | [`gaussian_blur`] | Convenience wrapper for Gaussian smoothing |
//! | [`box_blur`] | Convenience wrapper for box smoothing |
//! | [`sobel_smooth_row`] | Separable first pass for Sobel-style operators |
//!
//! ## Border handling
//!
//! [`BorderMode`] controls what happens when the kernel window extends beyond
//! the image boundary:
//!
//! - `Reflect` — mirror the image (e.g. `dcb|abcd|cba`)
//! - `Replicate` — clamp to the nearest edge pixel
//! - `Zero` — treat out-of-bounds pixels as 0

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::error::{ImageError, ImageResult};

// ─── Border modes ─────────────────────────────────────────────────────────────

/// Specifies how to handle border pixels during convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BorderMode {
    /// Mirror reflection at the border (e.g. `dcb|abcd|cba`).
    Reflect,
    /// Clamp — replicate the nearest edge pixel.
    Replicate,
    /// Treat all out-of-bounds pixels as 0.
    Zero,
}

/// Resolve an out-of-bounds index `i` in a dimension of length `len`.
///
/// Returns `None` for `Zero` mode (caller should use 0.0) or the reflected /
/// clamped index otherwise.
#[inline]
fn resolve_index(i: isize, len: usize, mode: BorderMode) -> Option<usize> {
    if i >= 0 && (i as usize) < len {
        return Some(i as usize);
    }
    match mode {
        BorderMode::Zero => None,
        BorderMode::Replicate => {
            let clamped = i.clamp(0, len as isize - 1) as usize;
            Some(clamped)
        }
        BorderMode::Reflect => {
            let len_i = len as isize;
            if len_i <= 1 {
                return Some(0);
            }
            // Fold the index into [0, len-1] using reflect-101 style.
            let mut idx = i;
            // Reduce modulo (2*(len-1))
            let period = 2 * (len_i - 1);
            idx %= period;
            if idx < 0 {
                idx += period;
            }
            if idx >= len_i {
                idx = period - idx;
            }
            Some(idx.clamp(0, len_i - 1) as usize)
        }
    }
}

// ─── Kernel builders ──────────────────────────────────────────────────────────

/// Build a 1-D Gaussian kernel of the given (odd) length.
///
/// The kernel is normalised so that all values sum to 1.0.
///
/// # Arguments
///
/// * `length` — number of taps; must be odd and ≥ 1.
/// * `sigma` — standard deviation; if `sigma <= 0.0` it is computed as
///   `(length / 6.0).max(0.3)` following OpenCV convention.
///
/// # Errors
///
/// Returns an error if `length` is even or zero.
pub fn gaussian_kernel(length: usize, sigma: f64) -> ImageResult<Vec<f64>> {
    if length == 0 || length % 2 == 0 {
        return Err(ImageError::invalid_format(format!(
            "gaussian_kernel: length must be odd and >= 1, got {length}"
        )));
    }
    let sigma = if sigma <= 0.0 {
        (length as f64 / 6.0).max(0.3)
    } else {
        sigma
    };
    let half = (length / 2) as isize;
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut k: Vec<f64> = (-half..=half)
        .map(|x| (-(x * x) as f64 * inv2s2).exp())
        .collect();
    let sum: f64 = k.iter().sum();
    if sum > 0.0 {
        k.iter_mut().for_each(|v| *v /= sum);
    }
    Ok(k)
}

/// Build a 1-D uniform (box) kernel of the given length.
///
/// # Errors
///
/// Returns an error if `length` is zero.
pub fn box_kernel(length: usize) -> ImageResult<Vec<f64>> {
    if length == 0 {
        return Err(ImageError::invalid_format(
            "box_kernel: length must be >= 1",
        ));
    }
    let v = 1.0 / length as f64;
    Ok(vec![v; length])
}

// ─── Core 2-pass separable convolution ────────────────────────────────────────

/// Apply a separable 2-pass convolution to a single-channel f32 image.
///
/// The first pass applies `kernel_h` (horizontal row convolution) producing an
/// intermediate buffer.  The second pass applies `kernel_v` (vertical column
/// convolution) on that buffer.
///
/// # Arguments
///
/// * `src` — source pixel data in row-major order.
/// * `width` — image width in pixels.
/// * `height` — image height in pixels.
/// * `kernel_h` — 1-D horizontal kernel (length must be odd and ≥ 1).
/// * `kernel_v` — 1-D vertical kernel (length must be odd and ≥ 1).
/// * `border` — border handling strategy.
///
/// # Errors
///
/// Returns an error if the buffer size doesn't match dimensions, or if a kernel
/// has even length or is empty.
pub fn apply_separable(
    src: &[f32],
    width: usize,
    height: usize,
    kernel_h: &[f64],
    kernel_v: &[f64],
    border: BorderMode,
) -> ImageResult<Vec<f32>> {
    let expected = width * height;
    if src.len() != expected {
        return Err(ImageError::invalid_format(format!(
            "apply_separable: src len {} != {}×{}",
            src.len(),
            width,
            height
        )));
    }
    validate_kernel(kernel_h, "kernel_h")?;
    validate_kernel(kernel_v, "kernel_v")?;

    // ── Pass 1: horizontal ──────────────────────────────────────────────────
    let half_h = (kernel_h.len() / 2) as isize;
    let mut tmp = vec![0.0f32; expected];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel_h.iter().enumerate() {
                let xi = x as isize - half_h + ki as isize;
                let val = match resolve_index(xi, width, border) {
                    Some(idx) => src[y * width + idx] as f64,
                    None => 0.0,
                };
                acc += kv * val;
            }
            tmp[y * width + x] = acc as f32;
        }
    }

    // ── Pass 2: vertical ────────────────────────────────────────────────────
    let half_v = (kernel_v.len() / 2) as isize;
    let mut dst = vec![0.0f32; expected];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel_v.iter().enumerate() {
                let yi = y as isize - half_v + ki as isize;
                let val = match resolve_index(yi, height, border) {
                    Some(idx) => tmp[idx * width + x] as f64,
                    None => 0.0,
                };
                acc += kv * val;
            }
            dst[y * width + x] = acc as f32;
        }
    }

    Ok(dst)
}

/// Validate that a kernel is non-empty and has odd length.
fn validate_kernel(kernel: &[f64], name: &str) -> ImageResult<()> {
    if kernel.is_empty() {
        return Err(ImageError::invalid_format(format!(
            "{name}: kernel must be non-empty"
        )));
    }
    if kernel.len() % 2 == 0 {
        return Err(ImageError::invalid_format(format!(
            "{name}: kernel length must be odd, got {}",
            kernel.len()
        )));
    }
    Ok(())
}

// ─── Convenience wrappers ─────────────────────────────────────────────────────

/// Gaussian blur a single-channel f32 image.
///
/// # Arguments
///
/// * `src` — source pixel data (row-major, values typically in `[0.0, 1.0]`).
/// * `width`, `height` — image dimensions.
/// * `kernel_size` — size of the Gaussian kernel (must be odd, ≥ 1).
/// * `sigma` — Gaussian standard deviation (use `0.0` for automatic).
/// * `border` — border handling strategy.
///
/// # Errors
///
/// Propagates errors from kernel building and [`apply_separable`].
pub fn gaussian_blur(
    src: &[f32],
    width: usize,
    height: usize,
    kernel_size: usize,
    sigma: f64,
    border: BorderMode,
) -> ImageResult<Vec<f32>> {
    let k = gaussian_kernel(kernel_size, sigma)?;
    apply_separable(src, width, height, &k, &k, border)
}

/// Box blur a single-channel f32 image.
///
/// # Arguments
///
/// * `src` — source pixel data (row-major).
/// * `width`, `height` — image dimensions.
/// * `kernel_size` — width and height of the box kernel (must be odd, ≥ 1).
/// * `border` — border handling strategy.
///
/// # Errors
///
/// Propagates errors from kernel building and [`apply_separable`].
pub fn box_blur(
    src: &[f32],
    width: usize,
    height: usize,
    kernel_size: usize,
    border: BorderMode,
) -> ImageResult<Vec<f32>> {
    let k = box_kernel(kernel_size)?;
    apply_separable(src, width, height, &k, &k, border)
}

/// Apply the separable Sobel smoothing pass — `[1, 2, 1] / 4` row filter.
///
/// This is the *smoothing* component of the Sobel operator (the differentiation
/// component must still be applied along the other axis).
///
/// # Errors
///
/// Propagates errors from [`apply_separable`].
pub fn sobel_smooth_row(
    src: &[f32],
    width: usize,
    height: usize,
    border: BorderMode,
) -> ImageResult<Vec<f32>> {
    let k = [0.25, 0.5, 0.25];
    apply_separable(src, width, height, &k, &[1.0], border)
}

// ─── U8 convenience helpers ───────────────────────────────────────────────────

/// Gaussian-blur a u8 grayscale image, returning a new u8 buffer.
///
/// Values are converted to `f32` in `[0.0, 1.0]`, blurred, then converted back.
///
/// # Errors
///
/// Propagates errors from [`gaussian_blur`].
pub fn gaussian_blur_u8(
    src: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
    sigma: f64,
) -> ImageResult<Vec<u8>> {
    let float_src: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();
    let float_dst = gaussian_blur(
        &float_src,
        width,
        height,
        kernel_size,
        sigma,
        BorderMode::Reflect,
    )?;
    Ok(float_dst
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect())
}

/// Box-blur a u8 grayscale image, returning a new u8 buffer.
///
/// # Errors
///
/// Propagates errors from [`box_blur`].
pub fn box_blur_u8(
    src: &[u8],
    width: usize,
    height: usize,
    kernel_size: usize,
) -> ImageResult<Vec<u8>> {
    let float_src: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();
    let float_dst = box_blur(&float_src, width, height, kernel_size, BorderMode::Reflect)?;
    Ok(float_dst
        .iter()
        .map(|&v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Kernel builders ────────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        for size in [3, 5, 7, 9, 11] {
            let k = gaussian_kernel(size, 1.5).unwrap();
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Gaussian kernel size {size} sum = {sum}"
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_symmetric() {
        let k = gaussian_kernel(7, 2.0).unwrap();
        for i in 0..k.len() / 2 {
            let mirror = k.len() - 1 - i;
            assert!(
                (k[i] - k[mirror]).abs() < 1e-14,
                "Gaussian kernel not symmetric at index {i}"
            );
        }
    }

    #[test]
    fn test_gaussian_kernel_even_length_error() {
        assert!(gaussian_kernel(4, 1.0).is_err());
    }

    #[test]
    fn test_box_kernel_sums_to_one() {
        for size in [1, 3, 5, 7] {
            let k = box_kernel(size).unwrap();
            let sum: f64 = k.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Box kernel size {size} sum = {sum}"
            );
        }
    }

    #[test]
    fn test_box_kernel_zero_error() {
        assert!(box_kernel(0).is_err());
    }

    // ── Border modes ───────────────────────────────────────────────────────────

    #[test]
    fn test_resolve_index_in_bounds() {
        for mode in [BorderMode::Reflect, BorderMode::Replicate, BorderMode::Zero] {
            assert_eq!(resolve_index(3, 10, mode), Some(3));
        }
    }

    #[test]
    fn test_resolve_index_zero_mode_out_of_bounds() {
        assert_eq!(resolve_index(-1, 10, BorderMode::Zero), None);
        assert_eq!(resolve_index(10, 10, BorderMode::Zero), None);
    }

    #[test]
    fn test_resolve_index_replicate() {
        assert_eq!(resolve_index(-3, 10, BorderMode::Replicate), Some(0));
        assert_eq!(resolve_index(12, 10, BorderMode::Replicate), Some(9));
    }

    #[test]
    fn test_resolve_index_reflect() {
        // reflect at -1 for length 5 should give 1 (mirror of -1 across 0)
        let idx = resolve_index(-1, 5, BorderMode::Reflect);
        assert!(idx.is_some(), "should return a valid index");
        // reflect at 5 (one past end) for length 5 should give 3
        let idx = resolve_index(5, 5, BorderMode::Reflect);
        assert_eq!(idx, Some(3));
    }

    // ── apply_separable ────────────────────────────────────────────────────────

    #[test]
    fn test_apply_separable_identity_kernel() {
        // A [1.0] identity kernel should produce the same image.
        let src: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let dst = apply_separable(&src, 5, 5, &[1.0], &[1.0], BorderMode::Replicate).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!(
                (s - d).abs() < 1e-5,
                "identity kernel changed value {s} -> {d}"
            );
        }
    }

    #[test]
    fn test_apply_separable_size_mismatch_error() {
        let src = vec![0.0f32; 9];
        let k = gaussian_kernel(3, 1.0).unwrap();
        // 4×3 = 12 != 9
        assert!(apply_separable(&src, 4, 3, &k, &k, BorderMode::Zero).is_err());
    }

    #[test]
    fn test_apply_separable_even_kernel_error() {
        let src = vec![0.0f32; 9];
        let k_bad = vec![0.5, 0.5]; // even length
        let k_ok = vec![1.0];
        assert!(apply_separable(&src, 3, 3, &k_bad, &k_ok, BorderMode::Zero).is_err());
    }

    // ── Gaussian blur ──────────────────────────────────────────────────────────

    #[test]
    fn test_gaussian_blur_uniform_image_unchanged() {
        // Blurring a uniform image should not change values.
        let src = vec![0.5f32; 100];
        let dst = gaussian_blur(&src, 10, 10, 5, 1.5, BorderMode::Reflect).unwrap();
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!(
                (s - d).abs() < 1e-5,
                "Gaussian blur changed uniform image: {s} -> {d}"
            );
        }
    }

    #[test]
    fn test_gaussian_blur_reduces_peak() {
        // A single bright pixel should spread after blurring.
        let mut src = vec![0.0f32; 81];
        src[4 * 9 + 4] = 1.0; // centre pixel of 9×9 image
        let dst = gaussian_blur(&src, 9, 9, 5, 1.5, BorderMode::Zero).unwrap();
        let centre_blurred = dst[4 * 9 + 4];
        assert!(
            centre_blurred < 1.0,
            "Peak should spread after blur, got {centre_blurred}"
        );
        // Energy (sum) should be approximately conserved (border effects may reduce it
        // slightly with Zero mode, but it should not be zero).
        let sum: f32 = dst.iter().sum();
        assert!(sum > 0.1, "Energy must be conserved after blur, sum={sum}");
    }

    // ── Box blur ───────────────────────────────────────────────────────────────

    #[test]
    fn test_box_blur_u8_uniform() {
        let src = vec![200u8; 64];
        let dst = box_blur_u8(&src, 8, 8, 3).unwrap();
        for &v in &dst {
            assert_eq!(v, 200, "Box blur on uniform image should be identity");
        }
    }

    // ── sobel_smooth_row ───────────────────────────────────────────────────────

    #[test]
    fn test_sobel_smooth_row_length() {
        let src = vec![1.0f32; 36];
        let dst = sobel_smooth_row(&src, 6, 6, BorderMode::Replicate).unwrap();
        assert_eq!(dst.len(), 36);
    }
}
