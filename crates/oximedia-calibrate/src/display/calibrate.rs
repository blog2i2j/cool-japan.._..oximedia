//! Display calibration workflow.
//!
//! This module provides tools for calibrating displays (monitors).

use crate::display::{GammaCurve, UniformityReport};
use crate::error::{CalibrationError, CalibrationResult};
use crate::{Illuminant, Matrix3x3, Rgb};
use serde::{Deserialize, Serialize};

/// Display calibration configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Target white point illuminant.
    pub target_white_point: Illuminant,
    /// Target gamma value.
    pub target_gamma: f64,
    /// Target luminance in cd/m².
    pub target_luminance: f64,
    /// Whether to measure uniformity.
    pub measure_uniformity: bool,
    /// Number of measurement points for uniformity (grid size).
    pub uniformity_grid_size: usize,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            target_white_point: Illuminant::D65,
            target_gamma: 2.2,
            target_luminance: 120.0,
            measure_uniformity: true,
            uniformity_grid_size: 9,
        }
    }
}

/// Display calibration result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplayCalibration {
    /// Display manufacturer.
    pub manufacturer: String,
    /// Display model.
    pub model: String,
    /// Measured gamma curve.
    pub gamma_curve: GammaCurve,
    /// Color primaries matrix (RGB to XYZ).
    pub primaries_matrix: Matrix3x3,
    /// White point XYZ.
    pub white_point: [f64; 3],
    /// Maximum luminance in cd/m².
    pub max_luminance: f64,
    /// Black luminance in cd/m².
    pub black_luminance: f64,
    /// Contrast ratio.
    pub contrast_ratio: f64,
    /// Uniformity report (if measured).
    pub uniformity: Option<UniformityReport>,
}

/// Display calibrator.
#[derive(Clone, Debug)]
pub struct DisplayCalibrator {
    config: DisplayConfig,
}

impl DisplayCalibrator {
    /// Create a new display calibrator with the given configuration.
    #[must_use]
    pub fn new(config: DisplayConfig) -> Self {
        Self { config }
    }

    /// Create a display calibrator with default configuration.
    #[must_use]
    pub fn default_calibrator() -> Self {
        Self::new(DisplayConfig::default())
    }

    /// Calibrate a display using measurement data.
    ///
    /// # Arguments
    ///
    /// * `manufacturer` - Display manufacturer
    /// * `model` - Display model
    /// * `measurements` - Measurement data (RGB input, XYZ output)
    ///
    /// # Errors
    ///
    /// Returns an error if calibration fails.
    pub fn calibrate_from_measurements(
        &self,
        manufacturer: String,
        model: String,
        _measurements: &[(Rgb, [f64; 3])],
    ) -> CalibrationResult<DisplayCalibration> {
        // This is a placeholder implementation
        // In a real implementation, this would:
        // 1. Analyze the measurements to determine the gamma curve
        // 2. Compute the color primaries matrix
        // 3. Measure white point and luminance levels
        // 4. Generate the calibration profile

        let gamma_curve = GammaCurve::new(self.config.target_gamma);
        let identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        Ok(DisplayCalibration {
            manufacturer,
            model,
            gamma_curve,
            primaries_matrix: identity_matrix,
            white_point: self.config.target_white_point.xyz(),
            max_luminance: self.config.target_luminance,
            black_luminance: 0.1,
            contrast_ratio: 1200.0,
            uniformity: None,
        })
    }

    /// Measure display gamma curve.
    ///
    /// # Arguments
    ///
    /// * `measurements` - Pairs of (input level, measured luminance)
    ///
    /// # Errors
    ///
    /// Returns an error if measurement fails.
    pub fn measure_gamma(&self, measurements: &[(f64, f64)]) -> CalibrationResult<GammaCurve> {
        if measurements.is_empty() {
            return Err(CalibrationError::InsufficientData(
                "No gamma measurements provided".to_string(),
            ));
        }

        // Fit a gamma curve to the measurements
        let gamma = self.fit_gamma_curve(measurements)?;

        Ok(GammaCurve::new(gamma))
    }

    /// Fit a gamma curve to measurements using least-squares.
    fn fit_gamma_curve(&self, measurements: &[(f64, f64)]) -> CalibrationResult<f64> {
        // This is a simplified implementation
        // A real implementation would use proper curve fitting

        if measurements.is_empty() {
            return Err(CalibrationError::InsufficientData(
                "No measurements for gamma curve fitting".to_string(),
            ));
        }

        // Return target gamma as placeholder
        Ok(self.config.target_gamma)
    }

    /// Measure display color primaries.
    ///
    /// # Arguments
    ///
    /// * `red_xyz` - XYZ values for full red (R=1, G=0, B=0)
    /// * `green_xyz` - XYZ values for full green (R=0, G=1, B=0)
    /// * `blue_xyz` - XYZ values for full blue (R=0, G=0, B=1)
    ///
    /// # Returns
    ///
    /// The RGB to XYZ conversion matrix.
    #[must_use]
    pub fn compute_primaries_matrix(
        &self,
        red_xyz: [f64; 3],
        green_xyz: [f64; 3],
        blue_xyz: [f64; 3],
    ) -> Matrix3x3 {
        // The primaries matrix is simply the three primary XYZ values as columns
        [
            [red_xyz[0], green_xyz[0], blue_xyz[0]],
            [red_xyz[1], green_xyz[1], blue_xyz[1]],
            [red_xyz[2], green_xyz[2], blue_xyz[2]],
        ]
    }

    /// Compute contrast ratio from max and black luminance.
    #[must_use]
    pub fn compute_contrast_ratio(&self, max_luminance: f64, black_luminance: f64) -> f64 {
        if black_luminance <= 0.0 {
            return f64::INFINITY;
        }
        max_luminance / black_luminance
    }

    /// Validate a display calibration.
    ///
    /// # Arguments
    ///
    /// * `calibration` - Calibration to validate
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails.
    pub fn validate_calibration(&self, calibration: &DisplayCalibration) -> CalibrationResult<()> {
        // Check gamma is in reasonable range
        let gamma = calibration.gamma_curve.gamma;
        if !(1.0..=3.0).contains(&gamma) {
            return Err(CalibrationError::DisplayCalibrationFailed(format!(
                "Gamma {gamma} outside reasonable range [1.0, 3.0]"
            )));
        }

        // Check luminance values
        if calibration.max_luminance <= calibration.black_luminance {
            return Err(CalibrationError::DisplayCalibrationFailed(
                "Maximum luminance must exceed black luminance".to_string(),
            ));
        }

        // Check contrast ratio
        if calibration.contrast_ratio < 100.0 {
            return Err(CalibrationError::DisplayCalibrationFailed(format!(
                "Contrast ratio {} is too low (minimum 100:1)",
                calibration.contrast_ratio
            )));
        }

        Ok(())
    }

    /// Apply display calibration to an RGB value.
    #[must_use]
    pub fn apply_calibration(&self, calibration: &DisplayCalibration, rgb: &Rgb) -> Rgb {
        // Apply gamma correction

        // Apply primaries matrix (if needed)
        calibration.gamma_curve.apply(rgb)
    }

    /// Serialize calibration to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn save_calibration(&self, calibration: &DisplayCalibration) -> CalibrationResult<String> {
        serde_json::to_string_pretty(calibration).map_err(|e| {
            CalibrationError::DisplayCalibrationFailed(format!(
                "Failed to serialize calibration: {e}"
            ))
        })
    }

    /// Deserialize calibration from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn load_calibration(&self, json: &str) -> CalibrationResult<DisplayCalibration> {
        serde_json::from_str(json).map_err(|e| {
            CalibrationError::DisplayCalibrationFailed(format!(
                "Failed to deserialize calibration: {e}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_config_default() {
        let config = DisplayConfig::default();
        assert_eq!(config.target_white_point, Illuminant::D65);
        assert!((config.target_gamma - 2.2).abs() < 1e-10);
        assert!((config.target_luminance - 120.0).abs() < 1e-10);
        assert!(config.measure_uniformity);
        assert_eq!(config.uniformity_grid_size, 9);
    }

    #[test]
    fn test_display_calibrator_new() {
        let config = DisplayConfig::default();
        let calibrator = DisplayCalibrator::new(config.clone());
        assert!((calibrator.config.target_gamma - config.target_gamma).abs() < 1e-10);
    }

    #[test]
    fn test_display_calibrator_default() {
        let calibrator = DisplayCalibrator::default_calibrator();
        assert!((calibrator.config.target_gamma - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_compute_primaries_matrix() {
        let calibrator = DisplayCalibrator::default_calibrator();
        let red_xyz = [0.64, 0.33, 0.03];
        let green_xyz = [0.30, 0.60, 0.10];
        let blue_xyz = [0.15, 0.06, 0.79];

        let matrix = calibrator.compute_primaries_matrix(red_xyz, green_xyz, blue_xyz);

        assert!((matrix[0][0] - 0.64).abs() < 1e-10);
        assert!((matrix[0][1] - 0.30).abs() < 1e-10);
        assert!((matrix[0][2] - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_compute_contrast_ratio() {
        let calibrator = DisplayCalibrator::default_calibrator();
        let ratio = calibrator.compute_contrast_ratio(120.0, 0.1);
        assert!((ratio - 1200.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_contrast_ratio_infinite() {
        let calibrator = DisplayCalibrator::default_calibrator();
        let ratio = calibrator.compute_contrast_ratio(120.0, 0.0);
        assert!(ratio.is_infinite());
    }

    #[test]
    fn test_measure_gamma_empty() {
        let calibrator = DisplayCalibrator::default_calibrator();
        let result = calibrator.measure_gamma(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_calibration_json_roundtrip() {
        let calibrator = DisplayCalibrator::default_calibrator();
        let calibration = DisplayCalibration {
            manufacturer: "Test".to_string(),
            model: "Monitor".to_string(),
            gamma_curve: GammaCurve::new(2.2),
            primaries_matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            white_point: Illuminant::D65.xyz(),
            max_luminance: 120.0,
            black_luminance: 0.1,
            contrast_ratio: 1200.0,
            uniformity: None,
        };

        let json = calibrator
            .save_calibration(&calibration)
            .expect("save should succeed");
        let restored = calibrator
            .load_calibration(&json)
            .expect("load should succeed");

        assert_eq!(restored.manufacturer, calibration.manufacturer);
        assert_eq!(restored.model, calibration.model);
    }
}
