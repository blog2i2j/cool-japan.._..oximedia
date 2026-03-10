//! Chromatic adaptation transforms.
//!
//! This module provides chromatic adaptation transforms for converting colors
//! between different illuminants.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Illuminant, Matrix3x3, Rgb, Xyz};

/// Chromatic adaptation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaticAdaptationMethod {
    /// Bradford chromatic adaptation transform.
    Bradford,
    /// Von Kries chromatic adaptation.
    VonKries,
    /// CAT02 (CIECAM02) chromatic adaptation.
    Cat02,
    /// XYZ scaling (simple).
    XyzScaling,
}

/// Chromatic adaptation processor.
pub struct ChromaticAdaptation {
    method: ChromaticAdaptationMethod,
    source_illuminant: Illuminant,
    target_illuminant: Illuminant,
    transform_matrix: Matrix3x3,
}

impl ChromaticAdaptation {
    /// Create a new chromatic adaptation transform.
    ///
    /// # Arguments
    ///
    /// * `method` - Adaptation method
    /// * `source_illuminant` - Source white point
    /// * `target_illuminant` - Target white point
    ///
    /// # Errors
    ///
    /// Returns an error if the transform cannot be computed.
    pub fn new(
        method: ChromaticAdaptationMethod,
        source_illuminant: Illuminant,
        target_illuminant: Illuminant,
    ) -> CalibrationResult<Self> {
        let transform_matrix =
            Self::compute_transform_matrix(method, source_illuminant, target_illuminant)?;

        Ok(Self {
            method,
            source_illuminant,
            target_illuminant,
            transform_matrix,
        })
    }

    /// Compute the chromatic adaptation transform matrix.
    fn compute_transform_matrix(
        method: ChromaticAdaptationMethod,
        source: Illuminant,
        target: Illuminant,
    ) -> CalibrationResult<Matrix3x3> {
        let source_xyz = source.xyz();
        let target_xyz = target.xyz();

        match method {
            ChromaticAdaptationMethod::Bradford => {
                Self::bradford_transform(&source_xyz, &target_xyz)
            }
            ChromaticAdaptationMethod::VonKries => {
                Self::von_kries_transform(&source_xyz, &target_xyz)
            }
            ChromaticAdaptationMethod::Cat02 => Self::cat02_transform(&source_xyz, &target_xyz),
            ChromaticAdaptationMethod::XyzScaling => {
                Self::xyz_scaling_transform(&source_xyz, &target_xyz)
            }
        }
    }

    /// Bradford chromatic adaptation transform.
    fn bradford_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // Bradford matrix
        let bradford = [
            [0.895_1, 0.266_4, -0.161_4],
            [-0.750_2, 1.713_5, 0.036_7],
            [0.038_9, -0.068_5, 1.029_6],
        ];

        // Inverse Bradford matrix
        let bradford_inv = [
            [0.986_993, -0.147_054, 0.159_963],
            [0.432_305, 0.518_360, 0.049_291],
            [-0.008_529, 0.040_043, 0.968_487],
        ];

        // Transform source and target white points to Bradford space
        let source_lms = Self::apply_matrix(&bradford, source_xyz);
        let target_lms = Self::apply_matrix(&bradford, target_xyz);

        if source_lms[0] < 1e-10 || source_lms[1] < 1e-10 || source_lms[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        // Compute scaling matrix
        let scale = [
            [target_lms[0] / source_lms[0], 0.0, 0.0],
            [0.0, target_lms[1] / source_lms[1], 0.0],
            [0.0, 0.0, target_lms[2] / source_lms[2]],
        ];

        // Compute final transform: bradford_inv * scale * bradford
        let temp = Self::multiply_matrices(&scale, &bradford);
        Ok(Self::multiply_matrices(&bradford_inv, &temp))
    }

    /// Von Kries chromatic adaptation.
    fn von_kries_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // Von Kries matrix (Hunt-Pointer-Estevez)
        let von_kries = [
            [0.400_24, 0.707_6, -0.080_8],
            [-0.226_3, 1.165_3, 0.045_7],
            [0.0, 0.0, 0.918_2],
        ];

        let von_kries_inv = [
            [1.859_936, -1.129_382, 0.219_897],
            [0.361_191, 0.638_812, -0.000_006],
            [0.0, 0.0, 1.089_064],
        ];

        let source_lms = Self::apply_matrix(&von_kries, source_xyz);
        let target_lms = Self::apply_matrix(&von_kries, target_xyz);

        if source_lms[0] < 1e-10 || source_lms[1] < 1e-10 || source_lms[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        let scale = [
            [target_lms[0] / source_lms[0], 0.0, 0.0],
            [0.0, target_lms[1] / source_lms[1], 0.0],
            [0.0, 0.0, target_lms[2] / source_lms[2]],
        ];

        let temp = Self::multiply_matrices(&scale, &von_kries);
        Ok(Self::multiply_matrices(&von_kries_inv, &temp))
    }

    /// CAT02 chromatic adaptation (CIECAM02).
    fn cat02_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        // CAT02 matrix
        let cat02 = [
            [0.732_8, 0.429_6, -0.162_4],
            [-0.703_6, 1.697_5, 0.006_1],
            [0.003_0, 0.013_6, 0.983_4],
        ];

        let cat02_inv = [
            [1.096_124, -0.278_869, 0.182_745],
            [0.454_369, 0.473_533, 0.072_098],
            [-0.009_628, -0.005_698, 1.015_326],
        ];

        let source_rgb = Self::apply_matrix(&cat02, source_xyz);
        let target_rgb = Self::apply_matrix(&cat02, target_xyz);

        if source_rgb[0] < 1e-10 || source_rgb[1] < 1e-10 || source_rgb[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        let scale = [
            [target_rgb[0] / source_rgb[0], 0.0, 0.0],
            [0.0, target_rgb[1] / source_rgb[1], 0.0],
            [0.0, 0.0, target_rgb[2] / source_rgb[2]],
        ];

        let temp = Self::multiply_matrices(&scale, &cat02);
        Ok(Self::multiply_matrices(&cat02_inv, &temp))
    }

    /// Simple XYZ scaling.
    fn xyz_scaling_transform(source_xyz: &Xyz, target_xyz: &Xyz) -> CalibrationResult<Matrix3x3> {
        if source_xyz[0] < 1e-10 || source_xyz[1] < 1e-10 || source_xyz[2] < 1e-10 {
            return Err(CalibrationError::ChromaticAdaptationFailed(
                "Invalid source illuminant".to_string(),
            ));
        }

        Ok([
            [target_xyz[0] / source_xyz[0], 0.0, 0.0],
            [0.0, target_xyz[1] / source_xyz[1], 0.0],
            [0.0, 0.0, target_xyz[2] / source_xyz[2]],
        ])
    }

    /// Apply a 3x3 matrix to a color.
    fn apply_matrix(matrix: &Matrix3x3, color: &[f64; 3]) -> [f64; 3] {
        [
            matrix[0][0] * color[0] + matrix[0][1] * color[1] + matrix[0][2] * color[2],
            matrix[1][0] * color[0] + matrix[1][1] * color[1] + matrix[1][2] * color[2],
            matrix[2][0] * color[0] + matrix[2][1] * color[1] + matrix[2][2] * color[2],
        ]
    }

    /// Multiply two 3x3 matrices.
    fn multiply_matrices(a: &Matrix3x3, b: &Matrix3x3) -> Matrix3x3 {
        let mut result = [[0.0; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        result
    }

    /// Adapt an XYZ color from source to target illuminant.
    #[must_use]
    pub fn adapt_xyz(&self, xyz: &Xyz) -> Xyz {
        Self::apply_matrix(&self.transform_matrix, xyz)
    }

    /// Adapt an RGB color (assumes RGB is in XYZ space).
    #[must_use]
    pub fn adapt_rgb(&self, rgb: &Rgb) -> Rgb {
        self.adapt_xyz(rgb)
    }

    /// Get the source illuminant.
    #[must_use]
    pub fn source_illuminant(&self) -> Illuminant {
        self.source_illuminant
    }

    /// Get the target illuminant.
    #[must_use]
    pub fn target_illuminant(&self) -> Illuminant {
        self.target_illuminant
    }

    /// Get the adaptation method.
    #[must_use]
    pub fn method(&self) -> ChromaticAdaptationMethod {
        self.method
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chromatic_adaptation_new() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());

        let ca = result.expect("expected successful result");
        assert_eq!(ca.source_illuminant(), Illuminant::D50);
        assert_eq!(ca.target_illuminant(), Illuminant::D65);
        assert_eq!(ca.method(), ChromaticAdaptationMethod::Bradford);
    }

    #[test]
    fn test_bradford_same_illuminant() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D65,
            Illuminant::D65,
        );

        assert!(result.is_ok());

        let ca = result.expect("expected successful result");
        let xyz = [0.5, 0.5, 0.5];
        let adapted = ca.adapt_xyz(&xyz);

        // Same illuminant should result in near-identity transform
        assert!((adapted[0] - xyz[0]).abs() < 0.01);
        assert!((adapted[1] - xyz[1]).abs() < 0.01);
        assert!((adapted[2] - xyz[2]).abs() < 0.01);
    }

    #[test]
    fn test_von_kries_adaptation() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::VonKries,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_cat02_adaptation() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Cat02,
            Illuminant::A,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_xyz_scaling() {
        let result = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::XyzScaling,
            Illuminant::D50,
            Illuminant::D65,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_adapt_xyz() {
        let ca = ChromaticAdaptation::new(
            ChromaticAdaptationMethod::Bradford,
            Illuminant::D50,
            Illuminant::D65,
        )
        .expect("unexpected None/Err");

        let xyz = [0.5, 0.5, 0.5];
        let adapted = ca.adapt_xyz(&xyz);

        // Values should be different after adaptation
        assert!(adapted[0] > 0.0 && adapted[0] <= 1.0);
        assert!(adapted[1] > 0.0 && adapted[1] <= 1.0);
        assert!(adapted[2] > 0.0 && adapted[2] <= 1.0);
    }

    #[test]
    fn test_multiply_matrices() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let scale = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let result = ChromaticAdaptation::multiply_matrices(&identity, &scale);

        assert!((result[0][0] - 2.0).abs() < 1e-10);
        assert!((result[1][1] - 2.0).abs() < 1e-10);
        assert!((result[2][2] - 2.0).abs() < 1e-10);
    }
}
