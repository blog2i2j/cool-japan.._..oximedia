//! Batch 3×3 matrix colour transforms on flat interleaved pixel buffers.
//!
//! This module complements [`super::simd_matrix`] (which operates on
//! `&mut [[f32; 3]]` slices) by providing functions that work directly on
//! **flat `f32` slices** in RGB-packed or RGBA-interleaved layout.  Rayon is
//! used for data-parallel processing of large buffers when the `rayon` feature
//! is enabled.
//!
//! # Buffer layouts
//!
//! | Function | Layout | Alpha |
//! |---|---|---|
//! | [`apply_matrix_rgb`] | `[R, G, B, R, G, B, …]` | N/A |
//! | [`apply_matrix_rgba`] | `[R, G, B, A, R, G, B, A, …]` | unchanged |
//!
//! # Matrix convention
//!
//! All matrices are **row-major**: `matrix[row][col]`.  To transform a pixel
//! `[r, g, b]`:
//!
//! ```text
//! r' = m[0][0]*r + m[0][1]*g + m[0][2]*b
//! g' = m[1][0]*r + m[1][1]*g + m[1][2]*b
//! b' = m[2][0]*r + m[2][1]*g + m[2][2]*b
//! ```

#[cfg(feature = "rayon")]
use rayon::prelude::*;

// ── Well-known colour matrices ────────────────────────────────────────────────

/// Pre-defined well-known 3×3 colour-conversion matrices.
pub mod matrices {
    /// sRGB primaries → XYZ D65 (IEC 61966-2-1).
    pub const SRGB_TO_XYZ_D65: [[f32; 3]; 3] = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ];

    /// XYZ D65 → sRGB (inverse of [`SRGB_TO_XYZ_D65`]).
    pub const XYZ_D65_TO_SRGB: [[f32; 3]; 3] = [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ];

    /// sRGB D65 → Rec.2020 D65.
    pub const SRGB_TO_REC2020: [[f32; 3]; 3] = [
        [0.6274040, 0.3292820, 0.0433136],
        [0.0690970, 0.9195400, 0.0113630],
        [0.0163916, 0.0880132, 0.8955952],
    ];
}

// ── Scalar helpers ────────────────────────────────────────────────────────────

/// Apply a 3×3 matrix to a single `(r, g, b)` triple.
#[inline(always)]
fn transform_rgb_scalar(m: &[[f32; 3]; 3], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    )
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Apply a 3×3 colour matrix to every RGB pixel in a **flat packed-RGB** buffer.
///
/// `pixels` must have a length that is a multiple of 3 (`[R, G, B, R, G, B, …]`).
///
/// # Panics
///
/// Panics if `pixels.len()` is not a multiple of 3.
pub fn apply_matrix_rgb(pixels: &mut [f32], matrix: &[[f32; 3]; 3]) {
    assert!(
        pixels.len() % 3 == 0,
        "RGB buffer length must be a multiple of 3"
    );

    #[cfg(feature = "rayon")]
    {
        pixels
            .par_chunks_exact_mut(3)
            .for_each(|chunk| {
                let (r2, g2, b2) = transform_rgb_scalar(matrix, chunk[0], chunk[1], chunk[2]);
                chunk[0] = r2;
                chunk[1] = g2;
                chunk[2] = b2;
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for chunk in pixels.chunks_exact_mut(3) {
            let (r2, g2, b2) = transform_rgb_scalar(matrix, chunk[0], chunk[1], chunk[2]);
            chunk[0] = r2;
            chunk[1] = g2;
            chunk[2] = b2;
        }
    }
}

/// Apply a 3×3 colour matrix to every RGB pixel in an **RGBA interleaved** buffer.
///
/// `pixels` must have a length that is a multiple of 4 (`[R, G, B, A, …]`).
/// Alpha values are passed through unchanged.
///
/// # Panics
///
/// Panics if `pixels.len()` is not a multiple of 4.
pub fn apply_matrix_rgba(pixels: &mut [f32], matrix: &[[f32; 3]; 3]) {
    assert!(
        pixels.len() % 4 == 0,
        "RGBA buffer length must be a multiple of 4"
    );

    #[cfg(feature = "rayon")]
    {
        pixels
            .par_chunks_exact_mut(4)
            .for_each(|chunk| {
                let (r2, g2, b2) = transform_rgb_scalar(matrix, chunk[0], chunk[1], chunk[2]);
                chunk[0] = r2;
                chunk[1] = g2;
                chunk[2] = b2;
                // chunk[3] (alpha) is intentionally left unchanged
            });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for chunk in pixels.chunks_exact_mut(4) {
            let (r2, g2, b2) = transform_rgb_scalar(matrix, chunk[0], chunk[1], chunk[2]);
            chunk[0] = r2;
            chunk[1] = g2;
            chunk[2] = b2;
        }
    }
}

/// Compose two 3×3 matrices: returns `a × b` (row-major matrix multiplication).
///
/// The result can be used to concatenate two sequential colour transforms into
/// a single pass, avoiding repeated per-pixel matrix loads.
#[must_use]
pub fn compose_matrices(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0_f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] =
                a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

/// Invert a 3×3 matrix via cofactor expansion.
///
/// Returns `None` if the matrix is singular (|det| < 1e-10).
#[must_use]
pub fn invert_matrix_3x3(m: &[[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-10 {
        return None;
    }
    let inv_det = 1.0 / det;

    Some([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// ── BatchMatrixTransform ──────────────────────────────────────────────────────

/// A reusable 3×3 colour matrix transform that can be applied to batches of
/// `[f32; 3]` colour triples without per-call matrix copies.
///
/// # Example
///
/// ```
/// use oximedia_colormgmt::transforms::batch_matrix::BatchMatrixTransform;
///
/// let identity = BatchMatrixTransform::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
/// let colors = vec![[0.5_f32, 0.3, 0.2], [0.1, 0.9, 0.4]];
/// let out = identity.apply_batch(&colors);
/// assert!((out[0][0] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct BatchMatrixTransform {
    /// The 3×3 colour matrix (row-major).
    pub matrix: [[f32; 3]; 3],
}

impl BatchMatrixTransform {
    /// Create a new transform from a 3×3 row-major matrix.
    #[must_use]
    pub fn new(matrix: [[f32; 3]; 3]) -> Self {
        Self { matrix }
    }

    /// Apply the stored matrix to every colour triple in `colors`.
    ///
    /// Returns a `Vec<[f32; 3]>` with the same length as `colors`.
    #[must_use]
    pub fn apply_batch(&self, colors: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let m = &self.matrix;
        colors
            .iter()
            .map(|&[r, g, b]| {
                [
                    m[0][0] * r + m[0][1] * g + m[0][2] * b,
                    m[1][0] * r + m[1][1] * g + m[1][2] * b,
                    m[2][0] * r + m[2][1] * g + m[2][2] * b,
                ]
            })
            .collect()
    }

    /// Compose `self` with another transform, returning `self × other`.
    ///
    /// The result applies `self` first, then `other`.
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self::new(compose_matrices(&self.matrix, &other.matrix))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    /// Helper: check two f32 values are within tolerance.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_apply_matrix_rgb_identity() {
        let mut pixels: Vec<f32> = vec![0.5, 0.3, 0.2, 0.1, 0.9, 0.4];
        let original = pixels.clone();
        apply_matrix_rgb(&mut pixels, &IDENTITY);
        for (a, b) in pixels.iter().zip(original.iter()) {
            assert!(approx_eq(*a, *b, 1e-6), "identity should not change values");
        }
    }

    #[test]
    fn test_apply_matrix_rgba_identity_alpha_unchanged() {
        let mut pixels: Vec<f32> = vec![
            0.5, 0.3, 0.2, 0.75, // pixel 0
            0.1, 0.9, 0.4, 0.5,  // pixel 1
        ];
        apply_matrix_rgba(&mut pixels, &IDENTITY);
        // Alpha channels must be unchanged
        assert!(approx_eq(pixels[3], 0.75, 1e-6), "alpha[0] unchanged");
        assert!(approx_eq(pixels[7], 0.5, 1e-6), "alpha[1] unchanged");
        // RGB values unchanged (identity)
        assert!(approx_eq(pixels[0], 0.5, 1e-6));
        assert!(approx_eq(pixels[1], 0.3, 1e-6));
        assert!(approx_eq(pixels[2], 0.2, 1e-6));
    }

    #[test]
    fn test_compose_matrices_identity() {
        // I × I = I
        let result = compose_matrices(&IDENTITY, &IDENTITY);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(result[row][col], expected, 1e-6),
                    "I×I[{row}][{col}] should be {expected}, got {}",
                    result[row][col]
                );
            }
        }
    }

    #[test]
    fn test_compose_matrices_multiplication() {
        let a = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let b = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]];
        // a × b should be close to I (scaling by 2,3,4 then 0.5,0.5,0.5)
        let result = compose_matrices(&a, &b);
        assert!(approx_eq(result[0][0], 1.0, 1e-5));
        assert!(approx_eq(result[1][1], 1.5, 1e-5)); // 3 * 0.5
        assert!(approx_eq(result[2][2], 2.0, 1e-5)); // 4 * 0.5
    }

    #[test]
    fn test_invert_matrix_identity() {
        let inv = invert_matrix_3x3(&IDENTITY).expect("identity is invertible");
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(inv[row][col], expected, 1e-5),
                    "inv(I)[{row}][{col}] should be {expected}, got {}",
                    inv[row][col]
                );
            }
        }
    }

    #[test]
    fn test_invert_matrix_then_compose_is_identity() {
        let m = matrices::SRGB_TO_XYZ_D65;
        let inv = invert_matrix_3x3(&m).expect("matrix should be invertible");
        let product = compose_matrices(&m, &inv);
        for row in 0..3 {
            for col in 0..3 {
                let expected = if row == col { 1.0_f32 } else { 0.0_f32 };
                assert!(
                    approx_eq(product[row][col], expected, 1e-4),
                    "A×inv(A)[{row}][{col}] should be ~{expected}, got {}",
                    product[row][col]
                );
            }
        }
    }

    #[test]
    fn test_invert_singular_matrix_returns_none() {
        // All-zeros matrix has determinant 0
        let singular = [[0.0_f32; 3]; 3];
        assert!(invert_matrix_3x3(&singular).is_none());
    }

    #[test]
    fn test_srgb_white_to_xyz() {
        // sRGB white (1, 1, 1) → XYZ D65 should be approximately (0.95, 1.0, 1.09)
        let mut pixels: Vec<f32> = vec![1.0, 1.0, 1.0];
        apply_matrix_rgb(&mut pixels, &matrices::SRGB_TO_XYZ_D65);
        assert!(approx_eq(pixels[0], 0.9504559, 0.01), "X ~0.950: {}", pixels[0]);
        assert!(approx_eq(pixels[1], 1.0000000, 0.01), "Y ~1.000: {}", pixels[1]);
        assert!(approx_eq(pixels[2], 1.0890577, 0.01), "Z ~1.089: {}", pixels[2]);
    }

    #[test]
    fn test_apply_matrix_rgba_rgb_transform_applied() {
        // Apply a 2x scaling matrix to RGB channels; alpha must be untouched
        let scale2 = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let mut pixels: Vec<f32> = vec![0.1, 0.2, 0.3, 0.9];
        apply_matrix_rgba(&mut pixels, &scale2);
        assert!(approx_eq(pixels[0], 0.2, 1e-6));
        assert!(approx_eq(pixels[1], 0.4, 1e-6));
        assert!(approx_eq(pixels[2], 0.6, 1e-6));
        assert!(approx_eq(pixels[3], 0.9, 1e-6), "alpha must be unchanged");
    }
}
