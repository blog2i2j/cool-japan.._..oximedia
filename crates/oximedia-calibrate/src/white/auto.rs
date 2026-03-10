//! Automatic white balance algorithms.
//!
//! This module provides automatic white balance algorithms for scene analysis.

use crate::error::{CalibrationError, CalibrationResult};
use crate::Rgb;
use serde::{Deserialize, Serialize};

/// White balance method.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhiteBalanceMethod {
    /// Gray world algorithm (assumes average color is neutral gray).
    GrayWorld,
    /// White patch algorithm (assumes brightest area is white).
    WhitePatch,
    /// Combination of gray world and white patch.
    Combined,
    /// Machine learning-based white balance.
    MachineLearning,
}

/// Automatic white balance result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhiteBalanceCorrection {
    /// Red multiplier.
    pub red_multiplier: f64,
    /// Green multiplier (usually 1.0).
    pub green_multiplier: f64,
    /// Blue multiplier.
    pub blue_multiplier: f64,
    /// Method used.
    pub method: WhiteBalanceMethod,
    /// Confidence in the result (0.0-1.0).
    pub confidence: f64,
}

impl WhiteBalanceCorrection {
    /// Create a new white balance correction.
    #[must_use]
    pub fn new(
        red_multiplier: f64,
        green_multiplier: f64,
        blue_multiplier: f64,
        method: WhiteBalanceMethod,
        confidence: f64,
    ) -> Self {
        Self {
            red_multiplier,
            green_multiplier,
            blue_multiplier,
            method,
            confidence,
        }
    }

    /// Apply the white balance correction to an RGB color.
    #[must_use]
    pub fn apply(&self, rgb: &Rgb) -> Rgb {
        [
            (rgb[0] * self.red_multiplier).clamp(0.0, 1.0),
            (rgb[1] * self.green_multiplier).clamp(0.0, 1.0),
            (rgb[2] * self.blue_multiplier).clamp(0.0, 1.0),
        ]
    }

    /// Apply the white balance correction to an entire image.
    #[must_use]
    pub fn apply_to_image(&self, image_data: &[u8]) -> Vec<u8> {
        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let corrected = self.apply(&[r, g, b]);

            output.push((corrected[0] * 255.0).round() as u8);
            output.push((corrected[1] * 255.0).round() as u8);
            output.push((corrected[2] * 255.0).round() as u8);
        }

        output
    }
}

/// Automatic white balance processor.
pub struct AutoWhiteBalance;

impl AutoWhiteBalance {
    /// Compute automatic white balance using gray world algorithm.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    ///
    /// # Errors
    ///
    /// Returns an error if white balance computation fails.
    pub fn gray_world(image_data: &[u8]) -> CalibrationResult<WhiteBalanceCorrection> {
        if image_data.is_empty() || image_data.len() % 3 != 0 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Invalid image data".to_string(),
            ));
        }

        // Calculate average RGB values
        let mut r_sum = 0.0;
        let mut g_sum = 0.0;
        let mut b_sum = 0.0;
        let pixel_count = image_data.len() / 3;

        for chunk in image_data.chunks_exact(3) {
            r_sum += f64::from(chunk[0]);
            g_sum += f64::from(chunk[1]);
            b_sum += f64::from(chunk[2]);
        }

        let r_avg = r_sum / pixel_count as f64;
        let g_avg = g_sum / pixel_count as f64;
        let b_avg = b_sum / pixel_count as f64;

        // Calculate gray (average of all channels)
        let gray = (r_avg + g_avg + b_avg) / 3.0;

        if gray < 1e-10 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Image is too dark for white balance".to_string(),
            ));
        }

        // Calculate multipliers
        let r_mult = gray / r_avg;
        let g_mult = gray / g_avg;
        let b_mult = gray / b_avg;

        Ok(WhiteBalanceCorrection::new(
            r_mult,
            g_mult,
            b_mult,
            WhiteBalanceMethod::GrayWorld,
            0.85,
        ))
    }

    /// Compute automatic white balance using white patch algorithm.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    /// * `percentile` - Percentile to use for white point (e.g., 99.0 for top 1%)
    ///
    /// # Errors
    ///
    /// Returns an error if white balance computation fails.
    pub fn white_patch(
        image_data: &[u8],
        percentile: f64,
    ) -> CalibrationResult<WhiteBalanceCorrection> {
        if image_data.is_empty() || image_data.len() % 3 != 0 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Invalid image data".to_string(),
            ));
        }

        // Collect all channel values
        let mut r_values: Vec<u8> = Vec::with_capacity(image_data.len() / 3);
        let mut g_values: Vec<u8> = Vec::with_capacity(image_data.len() / 3);
        let mut b_values: Vec<u8> = Vec::with_capacity(image_data.len() / 3);

        for chunk in image_data.chunks_exact(3) {
            r_values.push(chunk[0]);
            g_values.push(chunk[1]);
            b_values.push(chunk[2]);
        }

        // Sort to find percentile
        r_values.sort_unstable();
        g_values.sort_unstable();
        b_values.sort_unstable();

        let index = ((percentile / 100.0) * r_values.len() as f64) as usize;
        let index = index.min(r_values.len() - 1);

        let r_white = f64::from(r_values[index]);
        let g_white = f64::from(g_values[index]);
        let b_white = f64::from(b_values[index]);

        if r_white < 1e-10 || g_white < 1e-10 || b_white < 1e-10 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Cannot find valid white point".to_string(),
            ));
        }

        // Use the maximum as reference
        let max_white = r_white.max(g_white).max(b_white);

        Ok(WhiteBalanceCorrection::new(
            max_white / r_white,
            max_white / g_white,
            max_white / b_white,
            WhiteBalanceMethod::WhitePatch,
            0.80,
        ))
    }

    /// Compute automatic white balance using combined method.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    ///
    /// # Errors
    ///
    /// Returns an error if white balance computation fails.
    pub fn combined(image_data: &[u8]) -> CalibrationResult<WhiteBalanceCorrection> {
        let gray_world = Self::gray_world(image_data)?;
        let white_patch = Self::white_patch(image_data, 99.0)?;

        // Average the two methods
        let r_mult = (gray_world.red_multiplier + white_patch.red_multiplier) / 2.0;
        let g_mult = (gray_world.green_multiplier + white_patch.green_multiplier) / 2.0;
        let b_mult = (gray_world.blue_multiplier + white_patch.blue_multiplier) / 2.0;

        Ok(WhiteBalanceCorrection::new(
            r_mult,
            g_mult,
            b_mult,
            WhiteBalanceMethod::Combined,
            0.90,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_balance_correction_new() {
        let correction =
            WhiteBalanceCorrection::new(1.2, 1.0, 0.8, WhiteBalanceMethod::GrayWorld, 0.85);

        assert!((correction.red_multiplier - 1.2).abs() < 1e-10);
        assert!((correction.green_multiplier - 1.0).abs() < 1e-10);
        assert!((correction.blue_multiplier - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_white_balance_correction_apply() {
        let correction =
            WhiteBalanceCorrection::new(1.2, 1.0, 0.8, WhiteBalanceMethod::GrayWorld, 0.85);

        let rgb = [0.5, 0.5, 0.5];
        let corrected = correction.apply(&rgb);

        assert!((corrected[0] - 0.6).abs() < 1e-10);
        assert!((corrected[1] - 0.5).abs() < 1e-10);
        assert!((corrected[2] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_gray_world_empty_image() {
        let result = AutoWhiteBalance::gray_world(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gray_world_invalid_size() {
        let result = AutoWhiteBalance::gray_world(&[128, 128]); // Not multiple of 3
        assert!(result.is_err());
    }

    #[test]
    fn test_gray_world_neutral_image() {
        // Gray image should result in multipliers close to 1.0
        let image = vec![128; 300]; // 100 gray pixels
        let result = AutoWhiteBalance::gray_world(&image);

        assert!(result.is_ok());
        let correction = result.expect("expected successful result");

        // All multipliers should be close to 1.0 for neutral gray
        assert!((correction.red_multiplier - 1.0).abs() < 0.1);
        assert!((correction.green_multiplier - 1.0).abs() < 0.1);
        assert!((correction.blue_multiplier - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_white_patch_empty_image() {
        let result = AutoWhiteBalance::white_patch(&[], 99.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_white_balance_correction_apply_to_image() {
        let correction =
            WhiteBalanceCorrection::new(1.0, 1.0, 1.0, WhiteBalanceMethod::GrayWorld, 0.85);

        let image = vec![128, 128, 128, 255, 0, 0];
        let output = correction.apply_to_image(&image);

        assert_eq!(output.len(), image.len());
    }
}
