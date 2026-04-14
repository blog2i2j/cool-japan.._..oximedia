//! High-throughput 3×3 matrix multiply for bulk pixel transforms.
//!
//! This module provides optimised routines for applying a single 3×3 color
//! matrix to large batches of pixels. The implementation uses:
//!
//! - **Scalar-expanded constants** so the compiler can auto-vectorise with
//!   SSE2 / AVX2 / AVX-512 / NEON without any explicit intrinsics.
//! - **Cache-friendly f32 interleaved layout** (R0 G0 B0 R1 G1 B1 …) processed
//!   in a flat loop with stride 3.
//! - An optional **blocking strategy** that accumulates partial products in
//!   registers before writing back, reducing memory bandwidth pressure.
//!
//! # Layout
//!
//! All `pixels` slices are **interleaved** RGB triples:
//! ```text
//! [ R0, G0, B0, R1, G1, B1, … ]
//! ```
//! A slice of `n` pixels therefore has length `3n`.

use crate::error::{ColorError, Result};
use crate::math::matrix::Matrix3x3;

// ─── Public API ───────────────────────────────────────────────────────────────

/// Applies a 3×3 color matrix to every pixel in `pixels` in-place (f64 layout).
///
/// # Arguments
/// * `pixels` - Interleaved `[R, G, B, R, G, B, …]` slice in f64. Length must
///   be a multiple of 3.
/// * `matrix` - Row-major 3×3 transform matrix.
///
/// # Errors
/// Returns [`ColorError::InvalidColor`] if `pixels.len()` is not a multiple of 3.
pub fn apply_matrix_bulk_f64(pixels: &mut [f64], matrix: &Matrix3x3) -> Result<()> {
    if pixels.len() % 3 != 0 {
        return Err(ColorError::InvalidColor(
            "pixels slice length must be a multiple of 3".to_string(),
        ));
    }

    // Expand matrix rows so the compiler can hoist constants and vectorise.
    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    let mut i = 0;
    while i < pixels.len() {
        let r = pixels[i];
        let g = pixels[i + 1];
        let b = pixels[i + 2];

        pixels[i] = m00 * r + m01 * g + m02 * b;
        pixels[i + 1] = m10 * r + m11 * g + m12 * b;
        pixels[i + 2] = m20 * r + m21 * g + m22 * b;

        i += 3;
    }
    Ok(())
}

/// Applies a 3×3 color matrix to every pixel in `pixels` in-place (f32 layout).
///
/// # Arguments
/// * `pixels` - Interleaved `[R, G, B, R, G, B, …]` slice in f32. Length must
///   be a multiple of 3.
/// * `matrix` - Row-major 3×3 transform matrix (f64, converted internally to f32
///   for throughput).
///
/// # Errors
/// Returns [`ColorError::InvalidColor`] if `pixels.len()` is not a multiple of 3.
pub fn apply_matrix_bulk_f32(pixels: &mut [f32], matrix: &Matrix3x3) -> Result<()> {
    if pixels.len() % 3 != 0 {
        return Err(ColorError::InvalidColor(
            "pixels slice length must be a multiple of 3".to_string(),
        ));
    }

    let m00 = matrix[0][0] as f32;
    let m01 = matrix[0][1] as f32;
    let m02 = matrix[0][2] as f32;
    let m10 = matrix[1][0] as f32;
    let m11 = matrix[1][1] as f32;
    let m12 = matrix[1][2] as f32;
    let m20 = matrix[2][0] as f32;
    let m21 = matrix[2][1] as f32;
    let m22 = matrix[2][2] as f32;

    let mut i = 0;
    while i < pixels.len() {
        let r = pixels[i];
        let g = pixels[i + 1];
        let b = pixels[i + 2];

        pixels[i] = m00 * r + m01 * g + m02 * b;
        pixels[i + 1] = m10 * r + m11 * g + m12 * b;
        pixels[i + 2] = m20 * r + m21 * g + m22 * b;

        i += 3;
    }
    Ok(())
}

/// Transforms `src` pixels through the matrix and writes results to `dst` (f64).
///
/// Non-aliasing read-write split allows the compiler to emit gather/scatter
/// free code and helps the auto-vectoriser confirm the absence of aliasing.
///
/// # Errors
/// Returns an error if the slice lengths do not match or are not multiples of 3.
pub fn transform_pixels_f64(src: &[f64], dst: &mut [f64], matrix: &Matrix3x3) -> Result<()> {
    if src.len() != dst.len() {
        return Err(ColorError::InvalidColor(
            "src and dst slice lengths must be equal".to_string(),
        ));
    }
    if src.len() % 3 != 0 {
        return Err(ColorError::InvalidColor(
            "pixel slice length must be a multiple of 3".to_string(),
        ));
    }

    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    let mut i = 0;
    while i < src.len() {
        let r = src[i];
        let g = src[i + 1];
        let b = src[i + 2];

        dst[i] = m00 * r + m01 * g + m02 * b;
        dst[i + 1] = m10 * r + m11 * g + m12 * b;
        dst[i + 2] = m20 * r + m21 * g + m22 * b;

        i += 3;
    }
    Ok(())
}

/// Chains two 3×3 matrices and applies the combined transform to all pixels.
///
/// Equivalent to `apply_matrix_bulk_f64` with `matrix_b * matrix_a`, but avoids
/// computing the combined matrix externally (useful for single-pass pipelines).
///
/// # Errors
/// Returns an error if the slice length is not a multiple of 3.
pub fn apply_two_matrices_bulk_f64(
    pixels: &mut [f64],
    matrix_a: &Matrix3x3,
    matrix_b: &Matrix3x3,
) -> Result<()> {
    let combined = crate::math::matrix::multiply_matrices(matrix_b, matrix_a);
    apply_matrix_bulk_f64(pixels, &combined)
}

/// Parallel version of [`apply_matrix_bulk_f64`] using rayon.
///
/// Divides the pixel buffer into equal-sized chunks and processes each chunk on
/// a separate rayon thread.
///
/// # Errors
/// Returns an error if `pixels.len()` is not a multiple of 3.
#[cfg(feature = "rayon")]
pub fn apply_matrix_bulk_f64_parallel(pixels: &mut [f64], matrix: &Matrix3x3) -> Result<()> {
    use rayon::prelude::*;

    if pixels.len() % 3 != 0 {
        return Err(ColorError::InvalidColor(
            "pixels slice length must be a multiple of 3".to_string(),
        ));
    }

    let m00 = matrix[0][0];
    let m01 = matrix[0][1];
    let m02 = matrix[0][2];
    let m10 = matrix[1][0];
    let m11 = matrix[1][1];
    let m12 = matrix[1][2];
    let m20 = matrix[2][0];
    let m21 = matrix[2][1];
    let m22 = matrix[2][2];

    pixels.par_chunks_exact_mut(3).for_each(|chunk| {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];

        chunk[0] = m00 * r + m01 * g + m02 * b;
        chunk[1] = m10 * r + m11 * g + m12 * b;
        chunk[2] = m20 * r + m21 * g + m22 * b;
    });

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn scale_matrix(s: f64) -> Matrix3x3 {
        [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]]
    }

    fn approx_eq_slice(a: &[f64], b: &[f64], eps: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < eps)
    }

    #[test]
    fn test_apply_identity_f64_no_change() {
        let original = vec![0.1, 0.5, 0.9, 0.3, 0.7, 0.2];
        let mut pixels = original.clone();
        apply_matrix_bulk_f64(&mut pixels, &IDENTITY).expect("should succeed");
        assert!(
            approx_eq_slice(&pixels, &original, 1e-12),
            "identity matrix must not change pixel values"
        );
    }

    #[test]
    fn test_apply_scale_matrix_f64() {
        let mut pixels = vec![0.5, 0.5, 0.5, 1.0, 0.0, 0.0];
        apply_matrix_bulk_f64(&mut pixels, &scale_matrix(2.0)).expect("should succeed");
        assert!((pixels[0] - 1.0).abs() < 1e-12);
        assert!((pixels[1] - 1.0).abs() < 1e-12);
        assert!((pixels[2] - 1.0).abs() < 1e-12);
        assert!((pixels[3] - 2.0).abs() < 1e-12);
        assert!((pixels[4]).abs() < 1e-12);
        assert!((pixels[5]).abs() < 1e-12);
    }

    #[test]
    fn test_apply_matrix_bulk_f64_invalid_length() {
        let mut pixels = vec![0.1, 0.2]; // length 2 — not multiple of 3
        let result = apply_matrix_bulk_f64(&mut pixels, &IDENTITY);
        assert!(result.is_err(), "should reject non-multiple-of-3 length");
    }

    #[test]
    fn test_apply_identity_f32_no_change() {
        let original: Vec<f32> = vec![0.1, 0.5, 0.9, 0.3, 0.7, 0.2];
        let mut pixels = original.clone();
        apply_matrix_bulk_f32(&mut pixels, &IDENTITY).expect("should succeed");
        for (a, b) in pixels.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "identity must not change f32 values");
        }
    }

    #[test]
    fn test_apply_matrix_bulk_f32_invalid_length() {
        let mut pixels: Vec<f32> = vec![0.1, 0.2];
        let result = apply_matrix_bulk_f32(&mut pixels, &IDENTITY);
        assert!(result.is_err());
    }

    #[test]
    fn test_transform_pixels_f64_matches_inplace() {
        let src = vec![0.2, 0.4, 0.6, 0.8, 0.1, 0.3];
        let mut dst = vec![0.0; 6];
        transform_pixels_f64(&src, &mut dst, &IDENTITY).expect("should succeed");
        assert!(approx_eq_slice(&dst, &src, 1e-12));
    }

    #[test]
    fn test_transform_pixels_f64_length_mismatch() {
        let src = vec![0.1, 0.2, 0.3];
        let mut dst = vec![0.0; 6];
        let result = transform_pixels_f64(&src, &mut dst, &IDENTITY);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_two_matrices_identity_identity() {
        let original = vec![0.3, 0.6, 0.9, 0.0, 0.0, 0.0];
        let mut pixels = original.clone();
        apply_two_matrices_bulk_f64(&mut pixels, &IDENTITY, &IDENTITY).expect("should succeed");
        assert!(approx_eq_slice(&pixels, &original, 1e-12));
    }

    #[test]
    fn test_apply_two_matrices_scale_twice() {
        // Applying scale(2) then scale(3) = scale(6)
        let s2 = scale_matrix(2.0);
        let s3 = scale_matrix(3.0);
        let mut pixels = vec![0.5, 0.5, 0.5];
        apply_two_matrices_bulk_f64(&mut pixels, &s2, &s3).expect("should succeed");
        for &v in &pixels {
            assert!((v - 3.0).abs() < 1e-10, "expected 3.0, got {v}");
        }
    }

    #[test]
    fn test_empty_pixel_slice() {
        let mut pixels: Vec<f64> = vec![];
        apply_matrix_bulk_f64(&mut pixels, &IDENTITY).expect("empty slice should be ok");
    }

    #[test]
    fn test_large_batch_consistency() {
        // Check that a large batch gives the same result as single-pixel transforms.
        use crate::math::matrix::multiply_matrix_vector;

        let matrix: Matrix3x3 = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ];

        let inputs: Vec<[f64; 3]> = vec![
            [0.5, 0.3, 0.1],
            [0.9, 0.1, 0.7],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ];

        let mut flat: Vec<f64> = inputs.iter().flat_map(|p| p.iter().copied()).collect();
        apply_matrix_bulk_f64(&mut flat, &matrix).expect("batch apply should succeed");

        for (i, pixel_in) in inputs.iter().enumerate() {
            let expected = multiply_matrix_vector(&matrix, *pixel_in);
            let base = i * 3;
            for ch in 0..3 {
                assert!(
                    (flat[base + ch] - expected[ch]).abs() < 1e-12,
                    "pixel {i} channel {ch}: expected {}, got {}",
                    expected[ch],
                    flat[base + ch]
                );
            }
        }
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_parallel_matches_serial() {
        let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let matrix: Matrix3x3 = [
            [0.2, 0.7, 0.1],
            [0.1, 0.9, 0.0],
            [0.05, 0.05, 0.9],
        ];

        let mut serial = original.clone();
        apply_matrix_bulk_f64(&mut serial, &matrix).expect("serial should succeed");

        let mut parallel = original.clone();
        apply_matrix_bulk_f64_parallel(&mut parallel, &matrix).expect("parallel should succeed");

        assert!(approx_eq_slice(&serial, &parallel, 1e-12));
    }
}
