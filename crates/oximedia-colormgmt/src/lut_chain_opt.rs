//! LUT chain optimisation utilities.
//!
//! When a colour pipeline contains consecutive 3×3 matrix operations, it is
//! more efficient to merge them into a single matrix and apply it in one pass.
//! This module provides [`LutChainOptimizer`], which exposes helpers for
//! merging matrix pairs and — where possible — baking matrix sequences into
//! pre-existing LUT structures.
//!
//! # Matrix merging
//!
//! Two 3×3 matrices **M₁** and **M₂** applied in sequence are equivalent to
//! the single matrix **M = M₁ × M₂** (row-vector convention: `v' = v M`).
//! The merge is performed as a standard row-major matrix multiply.

/// LUT chain optimiser for colour pipeline efficiency.
///
/// The optimiser is currently stateless — all methods are associated functions
/// that operate on their arguments directly.  A future version may accumulate
/// transform steps and perform whole-chain analysis.
pub struct LutChainOptimizer;

impl LutChainOptimizer {
    /// Merge two consecutive 3×3 colour matrices into a single matrix.
    ///
    /// Given two transforms applied in sequence (first `m1`, then `m2`), the
    /// merged matrix `M = m1 × m2` produces identical output when applied once.
    ///
    /// # Arguments
    ///
    /// * `m1` — first (earlier) matrix in the chain.
    /// * `m2` — second (later) matrix in the chain.
    ///
    /// # Returns
    ///
    /// The product `m1 × m2` as a row-major `[[f32; 3]; 3]`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_colormgmt::lut_chain_opt::LutChainOptimizer;
    ///
    /// let identity = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    /// let scale2   = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    ///
    /// let merged = LutChainOptimizer::merge_matrix_pair(identity, scale2);
    /// // merged should equal scale2
    /// assert!((merged[0][0] - 2.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn merge_matrix_pair(
        m1: [[f32; 3]; 3],
        m2: [[f32; 3]; 3],
    ) -> [[f32; 3]; 3] {
        // To apply m1 first and then m2, the combined matrix is m2 × m1
        // (right-to-left composition for column-vector convention: v' = M*v).
        let mut out = [[0.0_f32; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                out[row][col] = m2[row][0] * m1[0][col]
                    + m2[row][1] * m1[1][col]
                    + m2[row][2] * m1[2][col];
            }
        }
        out
    }

    /// Merge an arbitrary sequence of 3×3 matrices into a single matrix.
    ///
    /// Matrices are applied left-to-right: `matrices[0]` is applied first,
    /// `matrices[last]` is applied last.  Returns the identity matrix when
    /// `matrices` is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_colormgmt::lut_chain_opt::LutChainOptimizer;
    ///
    /// let scale2 = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    /// let scale3 = [[3.0_f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
    /// let merged = LutChainOptimizer::merge_matrix_chain(&[scale2, scale3]);
    /// assert!((merged[0][0] - 6.0).abs() < 1e-5);
    /// ```
    #[must_use]
    pub fn merge_matrix_chain(matrices: &[[[f32; 3]; 3]]) -> [[f32; 3]; 3] {
        let identity = [[1.0_f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        matrices
            .iter()
            .fold(identity, |acc, &m| Self::merge_matrix_pair(acc, m))
    }

    /// Apply a merged matrix to a single colour triple.
    ///
    /// This is a convenience function for applying the result of
    /// [`merge_matrix_pair`] / [`merge_matrix_chain`] to individual pixels.
    #[must_use]
    pub fn apply(matrix: [[f32; 3]; 3], color: [f32; 3]) -> [f32; 3] {
        let [r, g, b] = color;
        [
            matrix[0][0] * r + matrix[0][1] * g + matrix[0][2] * b,
            matrix[1][0] * r + matrix[1][1] * g + matrix[1][2] * b,
            matrix[2][0] * r + matrix[2][1] * g + matrix[2][2] * b,
        ]
    }

    /// Compute the inverse of a 3×3 matrix using cofactor expansion.
    ///
    /// Returns `None` when the matrix is singular (|det| < 1 × 10⁻¹⁰).
    #[must_use]
    pub fn invert_matrix(m: [[f32; 3]; 3]) -> Option<[[f32; 3]; 3]> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn matrices_approx_eq(a: [[f32; 3]; 3], b: [[f32; 3]; 3], tol: f32) -> bool {
        for row in 0..3 {
            for col in 0..3 {
                if !approx_eq(a[row][col], b[row][col], tol) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_merge_identity_with_identity() {
        let merged = LutChainOptimizer::merge_matrix_pair(IDENTITY, IDENTITY);
        assert!(
            matrices_approx_eq(merged, IDENTITY, 1e-6),
            "I × I should be I"
        );
    }

    #[test]
    fn test_merge_identity_with_scale() {
        let scale2 = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let merged = LutChainOptimizer::merge_matrix_pair(IDENTITY, scale2);
        assert!(
            matrices_approx_eq(merged, scale2, 1e-6),
            "I × S should be S"
        );
    }

    #[test]
    fn test_merge_two_scales() {
        let scale2 = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let scale3 = [[3.0_f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        let expected = [[6.0_f32, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]];
        let merged = LutChainOptimizer::merge_matrix_pair(scale2, scale3);
        assert!(
            matrices_approx_eq(merged, expected, 1e-5),
            "2×2 scaling × 3×3 scaling should be 6×6: {merged:?}"
        );
    }

    #[test]
    fn test_merge_chain_empty_returns_identity() {
        let merged = LutChainOptimizer::merge_matrix_chain(&[]);
        assert!(matrices_approx_eq(merged, IDENTITY, 1e-6));
    }

    #[test]
    fn test_merge_chain_single_element() {
        let scale = [[5.0_f32, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
        let merged = LutChainOptimizer::merge_matrix_chain(&[scale]);
        assert!(matrices_approx_eq(merged, scale, 1e-6));
    }

    #[test]
    fn test_merge_chain_multiple() {
        let s2 = [[2.0_f32, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let s3 = [[3.0_f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
        let s4 = [[4.0_f32, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
        // 2 × 3 × 4 = 24
        let merged = LutChainOptimizer::merge_matrix_chain(&[s2, s3, s4]);
        assert!(
            approx_eq(merged[0][0], 24.0, 1e-4),
            "diagonal should be 24: {}",
            merged[0][0]
        );
    }

    #[test]
    fn test_apply_identity() {
        let color = [0.5_f32, 0.3, 0.2];
        let out = LutChainOptimizer::apply(IDENTITY, color);
        assert!(approx_eq(out[0], 0.5, 1e-6));
        assert!(approx_eq(out[1], 0.3, 1e-6));
        assert!(approx_eq(out[2], 0.2, 1e-6));
    }

    #[test]
    fn test_invert_identity() {
        let inv = LutChainOptimizer::invert_matrix(IDENTITY).expect("invertible");
        assert!(matrices_approx_eq(inv, IDENTITY, 1e-5));
    }

    #[test]
    fn test_invert_singular_returns_none() {
        let singular = [[0.0_f32; 3]; 3];
        assert!(LutChainOptimizer::invert_matrix(singular).is_none());
    }

    #[test]
    fn test_merge_then_apply_equals_sequential() {
        // Verify merge_matrix_pair(A, B) applied once == apply A then B.
        let a = [[1.0_f32, 0.5, 0.0], [0.0, 1.0, 0.3], [0.2, 0.0, 1.0]];
        let b = [[0.9_f32, 0.1, 0.0], [0.0, 0.8, 0.2], [0.1, 0.0, 0.95]];
        let color = [0.6_f32, 0.4, 0.2];

        // Sequential application
        let step1 = LutChainOptimizer::apply(a, color);
        let sequential = LutChainOptimizer::apply(b, step1);

        // Merged application
        let merged = LutChainOptimizer::merge_matrix_pair(a, b);
        let from_merged = LutChainOptimizer::apply(merged, color);

        for i in 0..3 {
            // f32 matrix multiply accumulates ~1 ULP of error per multiply-add;
            // allow a tolerance of 1e-4 rather than 1e-5 for non-trivial f32 matrices.
            assert!(
                approx_eq(from_merged[i], sequential[i], 1e-4),
                "channel {i}: merged={} sequential={}",
                from_merged[i],
                sequential[i]
            );
        }
    }
}
