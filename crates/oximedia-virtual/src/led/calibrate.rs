//! LED wall calibration
//!
//! Provides calibration workflows for LED wall geometry, brightness,
//! and uniformity correction.

use super::LedWall;
use crate::math::{Point3, Vector3};
use crate::{Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// Calibration pattern type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationPattern {
    /// Solid color
    Solid,
    /// Checkerboard pattern
    Checkerboard,
    /// Gradient pattern
    Gradient,
    /// Grid pattern
    Grid,
}

/// LED calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedCalibrationData {
    /// Brightness calibration per panel
    pub brightness: Vec<f32>,
    /// Color correction per panel (RGB multipliers)
    pub color_correction: Vec<[f32; 3]>,
    /// Geometric correction (position offsets)
    pub geometric_correction: Vec<Vector3<f64>>,
    /// Uniformity map
    pub uniformity: Vec<Vec<f32>>,
}

impl LedCalibrationData {
    /// Create new calibration data
    #[must_use]
    pub fn new(num_panels: usize) -> Self {
        Self {
            brightness: vec![1.0; num_panels],
            color_correction: vec![[1.0, 1.0, 1.0]; num_panels],
            geometric_correction: vec![Vector3::zeros(); num_panels],
            uniformity: vec![Vec::new(); num_panels],
        }
    }

    /// Apply calibration to pixel value
    #[must_use]
    pub fn apply_to_pixel(&self, panel_idx: usize, rgb: [u8; 3]) -> [u8; 3] {
        if panel_idx >= self.brightness.len() {
            return rgb;
        }

        let brightness = self.brightness[panel_idx];
        let color = self.color_correction[panel_idx];

        let r = (f32::from(rgb[0]) * brightness * color[0]).min(255.0) as u8;
        let g = (f32::from(rgb[1]) * brightness * color[1]).min(255.0) as u8;
        let b = (f32::from(rgb[2]) * brightness * color[2]).min(255.0) as u8;

        [r, g, b]
    }
}

/// LED wall calibrator
pub struct LedCalibrator {
    calibration_data: Option<LedCalibrationData>,
}

impl LedCalibrator {
    /// Create new LED calibrator
    #[must_use]
    pub fn new() -> Self {
        Self {
            calibration_data: None,
        }
    }

    /// Start calibration process
    pub fn start_calibration(&mut self, wall: &LedWall) -> Result<()> {
        self.calibration_data = Some(LedCalibrationData::new(wall.panels.len()));
        Ok(())
    }

    /// Calibrate brightness
    pub fn calibrate_brightness(&mut self, wall: &LedWall, measurements: &[f32]) -> Result<()> {
        let cal_data = self.calibration_data.as_mut().ok_or_else(|| {
            VirtualProductionError::Calibration("Calibration not started".to_string())
        })?;

        if measurements.len() != wall.panels.len() {
            return Err(VirtualProductionError::Calibration(format!(
                "Expected {} measurements, got {}",
                wall.panels.len(),
                measurements.len()
            )));
        }

        // Find maximum brightness
        let max_brightness = measurements.iter().copied().fold(0.0f32, f32::max);

        // Compute correction factors
        for (i, &measurement) in measurements.iter().enumerate() {
            if measurement > 0.0 {
                cal_data.brightness[i] = max_brightness / measurement;
            }
        }

        Ok(())
    }

    /// Calibrate color
    pub fn calibrate_color(&mut self, wall: &LedWall, measurements: &[[f32; 3]]) -> Result<()> {
        let cal_data = self.calibration_data.as_mut().ok_or_else(|| {
            VirtualProductionError::Calibration("Calibration not started".to_string())
        })?;

        if measurements.len() != wall.panels.len() {
            return Err(VirtualProductionError::Calibration(format!(
                "Expected {} measurements, got {}",
                wall.panels.len(),
                measurements.len()
            )));
        }

        // Compute color correction for each panel
        for (i, measurement) in measurements.iter().enumerate() {
            // Find max channel to normalize
            let max_channel = measurement.iter().copied().fold(0.0f32, f32::max);

            if max_channel > 0.0 {
                cal_data.color_correction[i] = [
                    max_channel / measurement[0].max(1e-6),
                    max_channel / measurement[1].max(1e-6),
                    max_channel / measurement[2].max(1e-6),
                ];
            }
        }

        Ok(())
    }

    /// Calibrate geometry
    pub fn calibrate_geometry(
        &mut self,
        wall: &LedWall,
        measured_positions: &[Point3<f64>],
    ) -> Result<()> {
        let cal_data = self.calibration_data.as_mut().ok_or_else(|| {
            VirtualProductionError::Calibration("Calibration not started".to_string())
        })?;

        if measured_positions.len() != wall.panels.len() {
            return Err(VirtualProductionError::Calibration(format!(
                "Expected {} positions, got {}",
                wall.panels.len(),
                measured_positions.len()
            )));
        }

        // Compute position offsets
        for (i, measured) in measured_positions.iter().enumerate() {
            let expected = wall.panels[i].position;
            cal_data.geometric_correction[i] = measured - &expected;
        }

        Ok(())
    }

    /// Generate calibration pattern
    #[must_use]
    pub fn generate_pattern(
        &self,
        pattern_type: CalibrationPattern,
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        let mut pixels = vec![0u8; width * height * 3];

        match pattern_type {
            CalibrationPattern::Solid => {
                // Fill with white
                pixels.fill(255);
            }
            CalibrationPattern::Checkerboard => {
                let square_size = 64;
                for y in 0..height {
                    for x in 0..width {
                        let checker = ((x / square_size) + (y / square_size)) % 2;
                        let value = if checker == 0 { 255 } else { 0 };
                        let idx = (y * width + x) * 3;
                        pixels[idx] = value;
                        pixels[idx + 1] = value;
                        pixels[idx + 2] = value;
                    }
                }
            }
            CalibrationPattern::Gradient => {
                for y in 0..height {
                    for x in 0..width {
                        let value = ((x as f32 / width as f32) * 255.0) as u8;
                        let idx = (y * width + x) * 3;
                        pixels[idx] = value;
                        pixels[idx + 1] = value;
                        pixels[idx + 2] = value;
                    }
                }
            }
            CalibrationPattern::Grid => {
                let grid_size = 32;
                for y in 0..height {
                    for x in 0..width {
                        let on_grid = (x % grid_size == 0) || (y % grid_size == 0);
                        let value = if on_grid { 255 } else { 0 };
                        let idx = (y * width + x) * 3;
                        pixels[idx] = value;
                        pixels[idx + 1] = value;
                        pixels[idx + 2] = value;
                    }
                }
            }
        }

        pixels
    }

    /// Get calibration data
    #[must_use]
    pub fn calibration_data(&self) -> Option<&LedCalibrationData> {
        self.calibration_data.as_ref()
    }

    /// Finalize calibration
    pub fn finalize(&mut self) -> Result<LedCalibrationData> {
        self.calibration_data.take().ok_or_else(|| {
            VirtualProductionError::Calibration("No calibration data available".to_string())
        })
    }
}

impl Default for LedCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_data() {
        let cal = LedCalibrationData::new(4);
        assert_eq!(cal.brightness.len(), 4);
        assert_eq!(cal.color_correction.len(), 4);
    }

    #[test]
    fn test_calibration_data_apply() {
        let mut cal = LedCalibrationData::new(1);
        cal.brightness[0] = 0.5;
        cal.color_correction[0] = [1.0, 0.5, 0.25];

        let result = cal.apply_to_pixel(0, [255, 255, 255]);
        assert_eq!(result[0], 127); // 255 * 0.5 * 1.0
        assert_eq!(result[1], 63); // 255 * 0.5 * 0.5
        assert_eq!(result[2], 31); // 255 * 0.5 * 0.25
    }

    #[test]
    fn test_led_calibrator() {
        let calibrator = LedCalibrator::new();
        assert!(calibrator.calibration_data().is_none());
    }

    #[test]
    fn test_calibration_workflow() {
        let mut calibrator = LedCalibrator::new();
        let wall = LedWall::new("Test".to_string());

        let result = calibrator.start_calibration(&wall);
        assert!(result.is_ok());
        assert!(calibrator.calibration_data().is_some());
    }

    #[test]
    fn test_generate_pattern() {
        let calibrator = LedCalibrator::new();
        let pattern = calibrator.generate_pattern(CalibrationPattern::Solid, 100, 100);
        assert_eq!(pattern.len(), 100 * 100 * 3);
        assert_eq!(pattern[0], 255);
    }
}
