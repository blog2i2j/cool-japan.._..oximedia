//! Custom white balance from reference patches.
//!
//! This module provides tools for custom white balance from user-selected reference areas.

use crate::error::{CalibrationError, CalibrationResult};
use crate::Rgb;

/// Custom white balance from a reference patch.
pub struct CustomWhiteBalance;

impl CustomWhiteBalance {
    /// Compute white balance from a reference gray/white patch.
    ///
    /// # Arguments
    ///
    /// * `patch_rgb` - Average RGB color of the reference patch
    ///
    /// # Errors
    ///
    /// Returns an error if the patch color is invalid.
    pub fn from_gray_patch(patch_rgb: &Rgb) -> CalibrationResult<[f64; 3]> {
        // Check that patch is not too dark
        let luminance = (patch_rgb[0] + patch_rgb[1] + patch_rgb[2]) / 3.0;

        if luminance < 0.1 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Reference patch is too dark".to_string(),
            ));
        }

        // Use the maximum channel as reference
        let max_channel = patch_rgb[0].max(patch_rgb[1]).max(patch_rgb[2]);

        if max_channel < 1e-10 {
            return Err(CalibrationError::WhiteBalanceFailed(
                "Reference patch is black".to_string(),
            ));
        }

        Ok([
            max_channel / patch_rgb[0],
            max_channel / patch_rgb[1],
            max_channel / patch_rgb[2],
        ])
    }

    /// Compute white balance from a manually specified white point.
    ///
    /// # Arguments
    ///
    /// * `white_point` - Desired white point RGB values
    #[must_use]
    pub fn from_white_point(white_point: &Rgb) -> [f64; 3] {
        let max_channel = white_point[0].max(white_point[1]).max(white_point[2]);

        if max_channel < 1e-10 {
            return [1.0, 1.0, 1.0];
        }

        [
            max_channel / white_point[0],
            max_channel / white_point[1],
            max_channel / white_point[2],
        ]
    }

    /// Apply custom white balance to an RGB color.
    #[must_use]
    pub fn apply(multipliers: &[f64; 3], rgb: &Rgb) -> Rgb {
        [
            (rgb[0] * multipliers[0]).clamp(0.0, 1.0),
            (rgb[1] * multipliers[1]).clamp(0.0, 1.0),
            (rgb[2] * multipliers[2]).clamp(0.0, 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_gray_patch_neutral() {
        let patch = [0.5, 0.5, 0.5];
        let result = CustomWhiteBalance::from_gray_patch(&patch);

        assert!(result.is_ok());
        let mult = result.expect("expected successful result");

        assert!((mult[0] - 1.0).abs() < 1e-10);
        assert!((mult[1] - 1.0).abs() < 1e-10);
        assert!((mult[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_gray_patch_warm() {
        let patch = [0.6, 0.5, 0.4];
        let result = CustomWhiteBalance::from_gray_patch(&patch);

        assert!(result.is_ok());
        let mult = result.expect("expected successful result");

        // Should boost blue and green relative to red
        assert!(mult[0] < mult[2]);
    }

    #[test]
    fn test_from_gray_patch_too_dark() {
        let patch = [0.05, 0.05, 0.05];
        let result = CustomWhiteBalance::from_gray_patch(&patch);

        assert!(result.is_err());
    }

    #[test]
    fn test_from_white_point() {
        let white_point = [1.0, 1.0, 1.0];
        let mult = CustomWhiteBalance::from_white_point(&white_point);

        assert!((mult[0] - 1.0).abs() < 1e-10);
        assert!((mult[1] - 1.0).abs() < 1e-10);
        assert!((mult[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_apply() {
        let multipliers = [1.2, 1.0, 0.8];
        let rgb = [0.5, 0.5, 0.5];
        let result = CustomWhiteBalance::apply(&multipliers, &rgb);

        assert!((result[0] - 0.6).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.4).abs() < 1e-10);
    }
}
