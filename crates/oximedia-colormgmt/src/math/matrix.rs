//! 3x3 matrix operations for color transformations.

use crate::error::{ColorError, Result};

/// 3x3 matrix type for color transformations.
pub type Matrix3x3 = [[f64; 3]; 3];

/// 3x3 matrix with named operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix3 {
    /// Matrix elements in row-major order.
    pub m: Matrix3x3,
}

impl Matrix3 {
    /// Creates a new 3x3 matrix.
    #[must_use]
    pub const fn new(m: Matrix3x3) -> Self {
        Self { m }
    }

    /// Creates an identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Multiplies this matrix by a vector.
    #[must_use]
    pub fn multiply_vector(&self, v: [f64; 3]) -> [f64; 3] {
        multiply_matrix_vector(&self.m, v)
    }

    /// Multiplies this matrix by another matrix.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        Self {
            m: multiply_matrices(&self.m, &other.m),
        }
    }

    /// Inverts this matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is singular (determinant is zero).
    pub fn invert(&self) -> Result<Self> {
        Ok(Self {
            m: invert_matrix_3x3(&self.m)?,
        })
    }

    /// Transposes this matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        Self {
            m: transpose_matrix(&self.m),
        }
    }

    /// Calculates the determinant of this matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        matrix_determinant(&self.m)
    }
}

impl Default for Matrix3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// Multiplies a 3x3 matrix by a 3-element vector.
#[must_use]
pub fn multiply_matrix_vector(m: &Matrix3x3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiplies two 3x3 matrices.
#[must_use]
pub fn multiply_matrices(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    result
}

/// Inverts a 3x3 matrix.
///
/// # Errors
///
/// Returns an error if the matrix is singular (determinant is zero).
pub fn invert_matrix_3x3(m: &Matrix3x3) -> Result<Matrix3x3> {
    let det = matrix_determinant(m);

    if det.abs() < 1e-10 {
        return Err(ColorError::Matrix("Matrix is singular".to_string()));
    }

    let inv_det = 1.0 / det;

    let mut inv = [[0.0; 3]; 3];

    // Calculate cofactor matrix and transpose (adjugate)
    inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;

    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;

    inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;

    Ok(inv)
}

/// Calculates the determinant of a 3x3 matrix.
#[must_use]
pub fn matrix_determinant(m: &Matrix3x3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Transposes a 3x3 matrix.
#[must_use]
pub fn transpose_matrix(m: &Matrix3x3) -> Matrix3x3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let id = Matrix3::identity();
        let v = [1.0, 2.0, 3.0];
        let result = id.multiply_vector(v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let m = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let v = [1.0, 2.0, 3.0];
        let result = multiply_matrix_vector(&m, v);
        assert_eq!(result, [2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = Matrix3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let b = Matrix3::identity();
        let result = a.multiply(&b);
        assert_eq!(result.m, a.m);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let det = matrix_determinant(&m);
        assert!((det - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_invert() {
        let m = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        let inv = invert_matrix_3x3(&m).expect("matrix inversion should succeed");
        let expected = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0 / 3.0]];

        for i in 0..3 {
            for j in 0..3 {
                assert!((inv[i][j] - expected[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_matrix_transpose() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let t = transpose_matrix(&m);
        assert_eq!(t[0][1], m[1][0]);
        assert_eq!(t[1][0], m[0][1]);
    }

    #[test]
    fn test_singular_matrix_invert() {
        let m = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]];
        assert!(invert_matrix_3x3(&m).is_err());
    }
}
