//! Chromatic adaptation transforms for white point conversion.

use crate::error::Result;
use crate::math::matrix::{
    invert_matrix_3x3, multiply_matrices, multiply_matrix_vector, Matrix3x3,
};

/// Chromatic adaptation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaticAdaptation {
    /// Bradford chromatic adaptation (most common, industry standard).
    Bradford,
    /// Von Kries chromatic adaptation.
    VonKries,
    /// XYZ scaling (simple but less accurate).
    XyzScaling,
}

impl ChromaticAdaptation {
    /// Gets the transformation matrix for this adaptation method.
    #[must_use]
    pub fn matrix(&self) -> Matrix3x3 {
        match self {
            Self::Bradford => BRADFORD_MATRIX,
            Self::VonKries => VON_KRIES_MATRIX,
            Self::XyzScaling => [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }
}

/// Bradford chromatic adaptation matrix.
const BRADFORD_MATRIX: Matrix3x3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Von Kries chromatic adaptation matrix.
const VON_KRIES_MATRIX: Matrix3x3 = [
    [0.4002, 0.7076, -0.0808],
    [-0.2263, 1.1653, 0.0457],
    [0.0000, 0.0000, 0.9182],
];

/// Computes Bradford chromatic adaptation transform from source to destination white point.
///
/// # Arguments
///
/// * `src_white` - Source white point in XYZ
/// * `dst_white` - Destination white point in XYZ
///
/// # Returns
///
/// 3x3 transformation matrix for adapting from source to destination white point.
///
/// # Errors
///
/// Returns an error if the matrix inversion fails.
#[allow(dead_code)]
pub fn bradford_transform(src_white: [f64; 3], dst_white: [f64; 3]) -> Result<Matrix3x3> {
    chromatic_adaptation_transform(src_white, dst_white, &BRADFORD_MATRIX)
}

/// Computes Von Kries chromatic adaptation transform from source to destination white point.
///
/// # Arguments
///
/// * `src_white` - Source white point in XYZ
/// * `dst_white` - Destination white point in XYZ
///
/// # Returns
///
/// 3x3 transformation matrix for adapting from source to destination white point.
///
/// # Errors
///
/// Returns an error if the matrix inversion fails.
#[allow(dead_code)]
pub fn von_kries_transform(src_white: [f64; 3], dst_white: [f64; 3]) -> Result<Matrix3x3> {
    chromatic_adaptation_transform(src_white, dst_white, &VON_KRIES_MATRIX)
}

/// Generic chromatic adaptation transform.
///
/// # Arguments
///
/// * `src_white` - Source white point in XYZ
/// * `dst_white` - Destination white point in XYZ
/// * `adaptation_matrix` - The adaptation cone response matrix (e.g., Bradford, Von Kries)
///
/// # Returns
///
/// 3x3 transformation matrix for adapting from source to destination white point.
///
/// # Errors
///
/// Returns an error if the matrix inversion fails.
pub fn chromatic_adaptation_transform(
    src_white: [f64; 3],
    dst_white: [f64; 3],
    adaptation_matrix: &Matrix3x3,
) -> Result<Matrix3x3> {
    // Convert white points to cone response space
    let src_cone = multiply_matrix_vector(adaptation_matrix, src_white);
    let dst_cone = multiply_matrix_vector(adaptation_matrix, dst_white);

    // Create scaling matrix in cone response space
    let scale = [
        [dst_cone[0] / src_cone[0], 0.0, 0.0],
        [0.0, dst_cone[1] / src_cone[1], 0.0],
        [0.0, 0.0, dst_cone[2] / src_cone[2]],
    ];

    // Compute: M^-1 * Scale * M
    let inv_adaptation = invert_matrix_3x3(adaptation_matrix)?;
    let temp = multiply_matrices(&scale, adaptation_matrix);
    Ok(multiply_matrices(&inv_adaptation, &temp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bradford_matrix() {
        let method = ChromaticAdaptation::Bradford;
        let m = method.matrix();
        assert_eq!(m, BRADFORD_MATRIX);
    }

    #[test]
    fn test_von_kries_matrix() {
        let method = ChromaticAdaptation::VonKries;
        let m = method.matrix();
        assert_eq!(m, VON_KRIES_MATRIX);
    }

    #[test]
    fn test_chromatic_adaptation_identity() {
        // D65 white point in XYZ
        let d65 = [0.95047, 1.0, 1.08883];

        // Adapting to the same white point should give identity-like result
        let transform = bradford_transform(d65, d65).expect("Bradford transform should succeed");

        // Apply transform to D65, should get D65 back
        let result = multiply_matrix_vector(&transform, d65);

        assert!((result[0] - d65[0]).abs() < 1e-6);
        assert!((result[1] - d65[1]).abs() < 1e-6);
        assert!((result[2] - d65[2]).abs() < 1e-6);
    }

    #[test]
    fn test_d65_to_d50_adaptation() {
        let d65 = [0.95047, 1.0, 1.08883];
        let d50 = [0.96422, 1.0, 0.82521];

        let transform = bradford_transform(d65, d50).expect("Bradford transform should succeed");
        let result = multiply_matrix_vector(&transform, d65);

        // Result should be close to D50
        assert!((result[0] - d50[0]).abs() < 1e-3);
        assert!((result[1] - d50[1]).abs() < 1e-3);
        assert!((result[2] - d50[2]).abs() < 1e-3);
    }
}
