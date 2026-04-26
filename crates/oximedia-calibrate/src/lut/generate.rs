//! Calibration LUT generation from measurements.
//!
//! This module provides tools for generating calibration LUTs from color measurements.

use crate::camera::ColorChecker;
use crate::error::{CalibrationError, CalibrationResult};
use crate::{Matrix3x3, Rgb};
use oximedia_lut::{Lut1d, Lut3d, LutSize};

/// LUT generator for calibration.
pub struct LutGenerator;

impl LutGenerator {
    /// Generate a 1D LUT from gamma measurements.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Target gamma value
    /// * `size` - LUT size (number of entries)
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn gamma_lut(gamma: f64, size: usize) -> CalibrationResult<Lut1d> {
        if size == 0 {
            return Err(CalibrationError::LutGenerationFailed(
                "LUT size must be greater than 0".to_string(),
            ));
        }

        if gamma <= 0.0 {
            return Err(CalibrationError::LutGenerationFailed(
                "Gamma must be positive".to_string(),
            ));
        }

        let mut lut = Lut1d::new(size);

        for i in 0..size {
            let input = i as f64 / (size - 1) as f64;
            let output = input.powf(1.0 / gamma);
            lut.set_r(i, output);
            lut.set_g(i, output);
            lut.set_b(i, output);
        }

        Ok(lut)
    }

    /// Generate a 3D calibration LUT from `ColorChecker` measurements.
    ///
    /// Uses inverse-distance-weighted (IDW) interpolation to compute a per-grid-point
    /// correction from the measured patches. Each grid point's output is the input
    /// plus a weighted average of `(reference_rgb - measured_rgb)` corrections,
    /// where the weight for each patch is `1 / (distance² + ε)`.
    ///
    /// If no patches are present, returns an identity LUT (output == input).
    ///
    /// # Arguments
    ///
    /// * `colorchecker` - `ColorChecker` with measured and reference colors
    /// * `lut_size` - Size of the 3D LUT
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn from_colorchecker(
        colorchecker: &ColorChecker,
        lut_size: LutSize,
    ) -> CalibrationResult<Lut3d> {
        // No patches: return an identity LUT so output == input everywhere.
        if colorchecker.patches.is_empty() {
            return Ok(Lut3d::identity(lut_size));
        }

        let n = lut_size.as_usize();
        let mut lut = Lut3d::new(lut_size);

        for ri in 0..n {
            for gi in 0..n {
                for bi in 0..n {
                    // Normalize grid indices to [0, 1].
                    let r = ri as f64 / (n - 1) as f64;
                    let g = gi as f64 / (n - 1) as f64;
                    let b = bi as f64 / (n - 1) as f64;

                    // IDW: accumulate weighted corrections from every patch.
                    // weight_k = 1 / (dist_k² + ε), correction = ref - measured.
                    let mut weight_sum = 0.0_f64;
                    let mut correction = [0.0_f64; 3];

                    for patch in &colorchecker.patches {
                        let dr = r - patch.measured_rgb[0];
                        let dg = g - patch.measured_rgb[1];
                        let db = b - patch.measured_rgb[2];
                        let dist_sq = dr * dr + dg * dg + db * db;
                        let weight = 1.0 / (dist_sq + 1e-10);

                        correction[0] += weight * (patch.reference_rgb[0] - patch.measured_rgb[0]);
                        correction[1] += weight * (patch.reference_rgb[1] - patch.measured_rgb[1]);
                        correction[2] += weight * (patch.reference_rgb[2] - patch.measured_rgb[2]);
                        weight_sum += weight;
                    }

                    // Normalise by total weight and clamp to valid range.
                    let out_r = (r + correction[0] / weight_sum).clamp(0.0, 1.0);
                    let out_g = (g + correction[1] / weight_sum).clamp(0.0, 1.0);
                    let out_b = (b + correction[2] / weight_sum).clamp(0.0, 1.0);

                    lut.set(ri, gi, bi, [out_r, out_g, out_b]);
                }
            }
        }

        Ok(lut)
    }

    /// Generate a 3D LUT from a color transformation matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - 3x3 color transformation matrix
    /// * `lut_size` - Size of the 3D LUT
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn from_matrix(matrix: &Matrix3x3, lut_size: LutSize) -> CalibrationResult<Lut3d> {
        let size = lut_size.as_usize();
        let mut lut = Lut3d::new(lut_size);

        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rgb = [
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    ];

                    let transformed = Self::apply_matrix(matrix, &rgb);
                    lut.set(r, g, b, transformed);
                }
            }
        }

        Ok(lut)
    }

    /// Apply a 3x3 matrix to an RGB color.
    fn apply_matrix(matrix: &Matrix3x3, rgb: &Rgb) -> Rgb {
        [
            matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
            matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
            matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
        ]
    }

    /// Generate a neutral-axis correction LUT.
    ///
    /// This LUT corrects the neutral (grayscale) axis to ensure pure whites and blacks.
    ///
    /// # Arguments
    ///
    /// * `lut_size` - Size of the 3D LUT
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn neutral_axis_lut(lut_size: LutSize) -> CalibrationResult<Lut3d> {
        let size = lut_size.as_usize();
        let mut lut = Lut3d::new(lut_size);

        for b in 0..size {
            for g in 0..size {
                for r in 0..size {
                    let rgb = [
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    ];

                    // If RGB values are close to each other (neutral), keep them neutral
                    let avg = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                    let max_diff = (rgb[0] - avg)
                        .abs()
                        .max((rgb[1] - avg).abs())
                        .max((rgb[2] - avg).abs());

                    let corrected = if max_diff < 0.05 {
                        // Force to neutral
                        [avg, avg, avg]
                    } else {
                        // Keep as-is
                        rgb
                    };

                    lut.set(r, g, b, corrected);
                }
            }
        }

        Ok(lut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gamma_lut() {
        let result = LutGenerator::gamma_lut(2.2, 256);
        assert!(result.is_ok());

        let lut = result.expect("expected successful result");
        assert_eq!(lut.size(), 256);
    }

    #[test]
    fn test_gamma_lut_zero_size() {
        let result = LutGenerator::gamma_lut(2.2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gamma_lut_zero_gamma() {
        let result = LutGenerator::gamma_lut(0.0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_gamma_lut_negative_gamma() {
        let result = LutGenerator::gamma_lut(-1.0, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_matrix_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = LutGenerator::from_matrix(&identity, LutSize::Size17);
        assert!(result.is_ok());

        let lut = result.expect("expected successful result");
        assert_eq!(lut.size(), LutSize::Size17.as_usize());
    }

    #[test]
    fn test_neutral_axis_lut() {
        let result = LutGenerator::neutral_axis_lut(LutSize::Size17);
        assert!(result.is_ok());

        let lut = result.expect("expected successful result");
        assert_eq!(lut.size(), LutSize::Size17.as_usize());
    }

    #[test]
    fn test_apply_matrix() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let rgb = [0.5, 0.6, 0.7];
        let result = LutGenerator::apply_matrix(&identity, &rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_apply_matrix_scale() {
        let scale = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];

        let rgb = [0.5, 0.6, 0.7];
        let result = LutGenerator::apply_matrix(&scale, &rgb);

        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 1.2).abs() < 1e-10);
        assert!((result[2] - 1.4).abs() < 1e-10);
    }

    // ------------------------------------------------------------------
    // from_colorchecker tests
    // ------------------------------------------------------------------

    #[test]
    fn test_from_colorchecker_empty_patches() {
        use crate::camera::{ColorChecker, ColorCheckerType};

        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: vec![],
            bounding_box: None,
            confidence: 1.0,
        };

        let result = LutGenerator::from_colorchecker(&checker, LutSize::Size17);
        assert!(result.is_ok(), "expected Ok for empty patches");

        let lut = result.expect("lut Ok");
        // Identity LUT: midpoint of the grid should map to itself.
        let mid = 8; // index 8 of 17 → 8/16 = 0.5
        let val = lut.get(mid, mid, mid);
        assert!(
            (val[0] - 0.5).abs() < 1e-6,
            "identity R mismatch: {}",
            val[0]
        );
        assert!(
            (val[1] - 0.5).abs() < 1e-6,
            "identity G mismatch: {}",
            val[1]
        );
        assert!(
            (val[2] - 0.5).abs() < 1e-6,
            "identity B mismatch: {}",
            val[2]
        );
    }

    #[test]
    fn test_from_colorchecker_two_patches() {
        use crate::camera::{ColorChecker, ColorCheckerType, PatchColor};

        let patches = vec![
            PatchColor {
                index: 0,
                measured_rgb: [0.2, 0.2, 0.2],
                reference_rgb: [0.25, 0.25, 0.25],
                reference_lab: [0.0, 0.0, 0.0],
                reference_xyz: [0.0, 0.0, 0.0],
                name: "Patch A".to_string(),
            },
            PatchColor {
                index: 1,
                measured_rgb: [0.8, 0.8, 0.8],
                reference_rgb: [0.75, 0.75, 0.75],
                reference_lab: [0.0, 0.0, 0.0],
                reference_xyz: [0.0, 0.0, 0.0],
                name: "Patch B".to_string(),
            },
        ];

        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches,
            bounding_box: None,
            confidence: 1.0,
        };

        let result = LutGenerator::from_colorchecker(&checker, LutSize::Size17);
        assert!(result.is_ok(), "expected Ok with two patches");

        let lut = result.expect("lut Ok");
        assert_eq!(lut.size(), LutSize::Size17.as_usize());
    }
}
