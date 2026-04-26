//! Camera calibration workflow.
//!
//! This module provides tools for calibrating cameras using `ColorChecker` targets.

use crate::camera::{ColorChecker, ColorCheckerType};
use crate::error::{CalibrationError, CalibrationResult};
use crate::icc::IccProfile;
use crate::{Illuminant, Matrix3x3, Rgb};
use oximedia_lut::{Lut3d, LutSize};
use serde::{Deserialize, Serialize};

/// Camera calibration configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// `ColorChecker` type to use for calibration.
    pub checker_type: ColorCheckerType,
    /// Target illuminant for the calibration.
    pub illuminant: Illuminant,
    /// Whether to include neutral axis calibration.
    pub calibrate_neutral_axis: bool,
    /// Whether to generate a 3D LUT in addition to the profile.
    pub generate_lut: bool,
    /// LUT size if generating a LUT.
    pub lut_size: usize,
    /// Minimum detection confidence (0.0-1.0).
    pub min_confidence: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            checker_type: ColorCheckerType::Classic24,
            illuminant: Illuminant::D65,
            calibrate_neutral_axis: true,
            generate_lut: true,
            lut_size: 33,
            min_confidence: 0.85,
        }
    }
}

/// Camera calibration result.
#[derive(Clone, Debug)]
pub struct CalibrationOutput {
    /// Detected `ColorChecker`.
    pub colorchecker: ColorChecker,
    /// Generated color matrix (3x3).
    pub color_matrix: Matrix3x3,
    /// Generated ICC profile (if requested).
    pub icc_profile: Option<IccProfile>,
    /// Generated 3D LUT (if requested).
    pub lut: Option<Lut3d>,
    /// Average color error (Delta E).
    pub average_error: f64,
    /// Maximum color error (Delta E).
    pub max_error: f64,
}

/// Camera calibrator.
#[derive(Clone, Debug)]
pub struct CameraCalibrator {
    config: CalibrationConfig,
}

impl CameraCalibrator {
    /// Create a new camera calibrator with the given configuration.
    #[must_use]
    pub fn new(config: CalibrationConfig) -> Self {
        Self { config }
    }

    /// Create a camera calibrator with default configuration.
    #[must_use]
    pub fn default_calibrator() -> Self {
        Self::new(CalibrationConfig::default())
    }

    /// Calibrate a camera from an image containing a `ColorChecker`.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if calibration fails.
    pub fn calibrate_from_image(
        &self,
        _image_data: &[u8],
        _width: usize,
        _height: usize,
    ) -> CalibrationResult<CalibrationOutput> {
        // Detect ColorChecker
        let colorchecker = ColorChecker::detect_in_image(_image_data, self.config.checker_type)?;

        // Verify detection confidence
        if colorchecker.confidence < self.config.min_confidence {
            return Err(CalibrationError::ColorCheckerNotFound(format!(
                "Detection confidence {} below minimum {}",
                colorchecker.confidence, self.config.min_confidence
            )));
        }

        // Generate color matrix
        let color_matrix = self.compute_color_matrix(&colorchecker)?;

        // Generate ICC profile if requested
        let icc_profile = None; // Placeholder

        // Generate 3D LUT if requested
        let lut = if self.config.generate_lut {
            Some(self.generate_calibration_lut(&colorchecker, &color_matrix)?)
        } else {
            None
        };

        // Calculate errors
        let average_error = colorchecker.calculate_average_error();
        let max_error = self.calculate_max_error(&colorchecker);

        Ok(CalibrationOutput {
            colorchecker,
            color_matrix,
            icc_profile,
            lut,
            average_error,
            max_error,
        })
    }

    /// Compute the color matrix from `ColorChecker` measurements.
    fn compute_color_matrix(&self, colorchecker: &ColorChecker) -> CalibrationResult<Matrix3x3> {
        // This is a simplified implementation
        // A real implementation would use least-squares optimization to find
        // the best matrix that transforms measured colors to reference colors

        if colorchecker.patches.is_empty() {
            return Err(CalibrationError::InsufficientData(
                "No patches available for matrix computation".to_string(),
            ));
        }

        // For now, return an identity matrix
        // In a real implementation, this would be computed from the patches
        Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Generate a 3D calibration LUT from the `ColorChecker` measurements.
    ///
    /// For each grid point the algorithm:
    ///   1. Applies the provided 3×3 colour matrix.
    ///   2. If the `ColorChecker` has patches, computes an IDW correction
    ///      (same technique as `LutGenerator::from_colorchecker`) and blends
    ///      70 % matrix output + 30 % patch-corrected output.
    ///   3. Clamps the result to [0, 1] and stores it in the LUT.
    fn generate_calibration_lut(
        &self,
        colorchecker: &ColorChecker,
        color_matrix: &Matrix3x3,
    ) -> CalibrationResult<Lut3d> {
        let lut_size = LutSize::from(self.config.lut_size);
        let n = lut_size.as_usize();
        let mut lut = Lut3d::new(lut_size);
        let has_patches = !colorchecker.patches.is_empty();

        for ri in 0..n {
            for gi in 0..n {
                for bi in 0..n {
                    // Normalise grid indices to [0, 1].
                    let r = ri as f64 / (n - 1) as f64;
                    let g = gi as f64 / (n - 1) as f64;
                    let b = bi as f64 / (n - 1) as f64;

                    // Apply the 3×3 colour matrix.
                    let matrix_out = self.apply_matrix(color_matrix, &[r, g, b]);

                    let final_out = if has_patches {
                        // IDW patch correction — same IDW formula as
                        // `LutGenerator::from_colorchecker`.
                        let mut weight_sum = 0.0_f64;
                        let mut correction = [0.0_f64; 3];

                        for patch in &colorchecker.patches {
                            let dr = r - patch.measured_rgb[0];
                            let dg = g - patch.measured_rgb[1];
                            let db = b - patch.measured_rgb[2];
                            let dist_sq = dr * dr + dg * dg + db * db;
                            let weight = 1.0 / (dist_sq + 1e-10);

                            correction[0] +=
                                weight * (patch.reference_rgb[0] - patch.measured_rgb[0]);
                            correction[1] +=
                                weight * (patch.reference_rgb[1] - patch.measured_rgb[1]);
                            correction[2] +=
                                weight * (patch.reference_rgb[2] - patch.measured_rgb[2]);
                            weight_sum += weight;
                        }

                        let patch_out = [
                            r + correction[0] / weight_sum,
                            g + correction[1] / weight_sum,
                            b + correction[2] / weight_sum,
                        ];

                        // Blend: 70 % matrix result + 30 % patch correction.
                        [
                            (0.7 * matrix_out[0] + 0.3 * patch_out[0]).clamp(0.0, 1.0),
                            (0.7 * matrix_out[1] + 0.3 * patch_out[1]).clamp(0.0, 1.0),
                            (0.7 * matrix_out[2] + 0.3 * patch_out[2]).clamp(0.0, 1.0),
                        ]
                    } else {
                        // No patches: pure matrix output.
                        [
                            matrix_out[0].clamp(0.0, 1.0),
                            matrix_out[1].clamp(0.0, 1.0),
                            matrix_out[2].clamp(0.0, 1.0),
                        ]
                    };

                    lut.set(ri, gi, bi, final_out);
                }
            }
        }

        Ok(lut)
    }

    /// Calculate the maximum color error from the `ColorChecker`.
    fn calculate_max_error(&self, colorchecker: &ColorChecker) -> f64 {
        colorchecker
            .patches
            .iter()
            .map(|patch| self.calculate_patch_error(&patch.measured_rgb, &patch.reference_rgb))
            .fold(0.0_f64, f64::max)
    }

    /// Calculate color error for a single patch.
    fn calculate_patch_error(&self, measured: &Rgb, reference: &Rgb) -> f64 {
        // Simplified Euclidean distance in RGB space
        let dr = measured[0] - reference[0];
        let dg = measured[1] - reference[1];
        let db = measured[2] - reference[2];

        (dr * dr + dg * dg + db * db).sqrt() * 100.0
    }

    /// Apply the calibration to an image.
    ///
    /// # Arguments
    ///
    /// * `calibration` - Calibration output to apply
    /// * `image_data` - Raw image data (RGB format)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if application fails.
    pub fn apply_calibration(
        &self,
        calibration: &CalibrationOutput,
        image_data: &[u8],
        _width: usize,
        _height: usize,
    ) -> CalibrationResult<Vec<u8>> {
        // Apply the color matrix to each pixel
        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            // Apply color matrix
            let rgb = [r, g, b];
            let corrected = self.apply_matrix(&calibration.color_matrix, &rgb);

            // Convert back to u8
            output.push((corrected[0] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((corrected[1] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((corrected[2] * 255.0).clamp(0.0, 255.0) as u8);
        }

        Ok(output)
    }

    /// Apply a 3x3 color matrix to an RGB color.
    fn apply_matrix(&self, matrix: &Matrix3x3, rgb: &Rgb) -> Rgb {
        [
            matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
            matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
            matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
        ]
    }

    /// Verify calibration accuracy.
    ///
    /// # Arguments
    ///
    /// * `calibration` - Calibration to verify
    /// * `max_average_error` - Maximum acceptable average Delta E
    /// * `max_single_error` - Maximum acceptable single patch Delta E
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails.
    pub fn verify_calibration(
        &self,
        calibration: &CalibrationOutput,
        max_average_error: f64,
        max_single_error: f64,
    ) -> CalibrationResult<()> {
        if calibration.average_error > max_average_error {
            return Err(CalibrationError::VerificationFailed(format!(
                "Average error {} exceeds maximum {}",
                calibration.average_error, max_average_error
            )));
        }

        if calibration.max_error > max_single_error {
            return Err(CalibrationError::VerificationFailed(format!(
                "Maximum error {} exceeds limit {}",
                calibration.max_error, max_single_error
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_config_default() {
        let config = CalibrationConfig::default();
        assert_eq!(config.checker_type, ColorCheckerType::Classic24);
        assert_eq!(config.illuminant, Illuminant::D65);
        assert!(config.calibrate_neutral_axis);
        assert!(config.generate_lut);
        assert_eq!(config.lut_size, 33);
        assert!((config.min_confidence - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_camera_calibrator_new() {
        let config = CalibrationConfig::default();
        let calibrator = CameraCalibrator::new(config.clone());
        assert_eq!(calibrator.config.checker_type, config.checker_type);
    }

    #[test]
    fn test_camera_calibrator_default() {
        let calibrator = CameraCalibrator::default_calibrator();
        assert_eq!(calibrator.config.checker_type, ColorCheckerType::Classic24);
    }

    #[test]
    fn test_apply_matrix_identity() {
        let calibrator = CameraCalibrator::default_calibrator();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rgb = [0.5, 0.6, 0.7];
        let result = calibrator.apply_matrix(&identity, &rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_patch_error() {
        let calibrator = CameraCalibrator::default_calibrator();
        let measured = [0.5, 0.5, 0.5];
        let reference = [0.5, 0.5, 0.5];
        let error = calibrator.calculate_patch_error(&measured, &reference);
        assert!(error < 1e-10);
    }

    // ------------------------------------------------------------------
    // generate_calibration_lut tests
    // ------------------------------------------------------------------

    #[test]
    fn test_generate_calibration_lut_identity_matrix() {
        use crate::camera::colorchecker::ColorChecker;
        use oximedia_lut::LutInterpolation;

        let calibrator = CameraCalibrator::default_calibrator();
        let identity: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // classic24 has measured == reference, so IDW correction is zero.
        // With an identity matrix the output must equal the input.
        let checker = ColorChecker {
            checker_type: crate::camera::ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        let result = calibrator.generate_calibration_lut(&checker, &identity);
        assert!(result.is_ok(), "expected Ok for identity matrix");

        let lut = result.expect("lut Ok");

        // Sample the neutral-gray point at 0.5, 0.5, 0.5.
        let gray = [0.5, 0.5, 0.5];
        let out = lut.apply(&gray, LutInterpolation::Tetrahedral);

        // Because measured == reference in classic24, IDW correction is 0.
        // Both matrix and patch paths yield 0.5 → blend is still 0.5.
        for (ch, &v) in out.iter().enumerate() {
            assert!(
                (v - 0.5).abs() < 1e-4,
                "channel {ch}: expected ~0.5, got {v}"
            );
        }
    }

    #[test]
    fn test_generate_calibration_lut_empty_patches() {
        let calibrator = CameraCalibrator::default_calibrator();
        let identity: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let checker = ColorChecker {
            checker_type: crate::camera::ColorCheckerType::Classic24,
            patches: vec![],
            bounding_box: None,
            confidence: 1.0,
        };

        let result = calibrator.generate_calibration_lut(&checker, &identity);
        assert!(
            result.is_ok(),
            "expected Ok for empty patches with identity matrix"
        );
    }

    #[test]
    fn test_generate_calibration_lut_custom_config() {
        use crate::camera::colorchecker::ColorChecker;

        let config = CalibrationConfig {
            lut_size: 17,
            generate_lut: true,
            ..CalibrationConfig::default()
        };
        let calibrator = CameraCalibrator::new(config);
        let identity: Matrix3x3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let checker = ColorChecker {
            checker_type: crate::camera::ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        let result = calibrator.generate_calibration_lut(&checker, &identity);
        assert!(result.is_ok());
        let lut = result.expect("lut Ok");
        assert_eq!(lut.size(), 17);
    }
}
