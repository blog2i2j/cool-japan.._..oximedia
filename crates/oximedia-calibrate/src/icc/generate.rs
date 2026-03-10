//! ICC profile generation.
//!
//! This module provides tools for generating ICC color profiles.

use crate::camera::CameraProfile;
use crate::error::CalibrationResult;
use crate::icc::IccProfile;
use crate::{Illuminant, Matrix3x3};

/// ICC profile generator.
pub struct IccProfileGenerator;

impl IccProfileGenerator {
    /// Generate an ICC profile from a camera profile.
    ///
    /// # Arguments
    ///
    /// * `camera_profile` - Camera profile to convert to ICC
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn from_camera_profile(camera_profile: &CameraProfile) -> CalibrationResult<IccProfile> {
        // Validate the camera profile
        camera_profile.validate()?;

        // Create ICC profile
        let profile = IccProfile::new(
            format!(
                "{} {} Camera Profile",
                camera_profile.manufacturer, camera_profile.model
            ),
            camera_profile.forward_matrix,
            camera_profile.illuminant,
        );

        Ok(profile)
    }

    /// Generate an ICC display profile.
    ///
    /// # Arguments
    ///
    /// * `manufacturer` - Display manufacturer
    /// * `model` - Display model
    /// * `primaries_matrix` - RGB to XYZ conversion matrix
    /// * `white_point` - Target white point
    /// * `gamma` - Display gamma value
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn display_profile(
        manufacturer: String,
        model: String,
        primaries_matrix: Matrix3x3,
        white_point: Illuminant,
        _gamma: f64,
    ) -> CalibrationResult<IccProfile> {
        let profile = IccProfile::new(
            format!("{manufacturer} {model} Display Profile"),
            primaries_matrix,
            white_point,
        );

        Ok(profile)
    }

    /// Generate a basic sRGB ICC profile.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn srgb_profile() -> CalibrationResult<IccProfile> {
        // sRGB primaries matrix (Rec.709)
        let srgb_matrix = [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ];

        Ok(IccProfile::new(
            "sRGB".to_string(),
            srgb_matrix,
            Illuminant::D65,
        ))
    }

    /// Generate a basic Adobe RGB ICC profile.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn adobe_rgb_profile() -> CalibrationResult<IccProfile> {
        // Adobe RGB primaries matrix
        let adobe_matrix = [
            [0.5767, 0.1856, 0.1882],
            [0.2974, 0.6273, 0.0753],
            [0.0270, 0.0707, 0.9911],
        ];

        Ok(IccProfile::new(
            "Adobe RGB".to_string(),
            adobe_matrix,
            Illuminant::D65,
        ))
    }

    /// Generate a basic `ProPhoto` RGB ICC profile.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn prophoto_rgb_profile() -> CalibrationResult<IccProfile> {
        // `ProPhoto` RGB primaries matrix
        let prophoto_matrix = [
            [0.7976, 0.1352, 0.0313],
            [0.2880, 0.7118, 0.0001],
            [0.0000, 0.0000, 0.8252],
        ];

        Ok(IccProfile::new(
            "ProPhoto RGB".to_string(),
            prophoto_matrix,
            Illuminant::D50,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_camera_profile() {
        let camera_profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        let result = IccProfileGenerator::from_camera_profile(&camera_profile);
        assert!(result.is_ok());

        let icc = result.expect("expected successful result");
        assert!(icc.description.contains("Test"));
        assert!(icc.description.contains("Camera"));
    }

    #[test]
    fn test_display_profile() {
        let matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let result = IccProfileGenerator::display_profile(
            "Test".to_string(),
            "Monitor".to_string(),
            matrix,
            Illuminant::D65,
            2.2,
        );

        assert!(result.is_ok());

        let icc = result.expect("expected successful result");
        assert!(icc.description.contains("Test"));
        assert!(icc.description.contains("Monitor"));
    }

    #[test]
    fn test_srgb_profile() {
        let result = IccProfileGenerator::srgb_profile();
        assert!(result.is_ok());

        let icc = result.expect("expected successful result");
        assert_eq!(icc.description, "sRGB");
        assert_eq!(icc.white_point, Illuminant::D65);
    }

    #[test]
    fn test_adobe_rgb_profile() {
        let result = IccProfileGenerator::adobe_rgb_profile();
        assert!(result.is_ok());

        let icc = result.expect("expected successful result");
        assert_eq!(icc.description, "Adobe RGB");
        assert_eq!(icc.white_point, Illuminant::D65);
    }

    #[test]
    fn test_prophoto_rgb_profile() {
        let result = IccProfileGenerator::prophoto_rgb_profile();
        assert!(result.is_ok());

        let icc = result.expect("expected successful result");
        assert_eq!(icc.description, "ProPhoto RGB");
        assert_eq!(icc.white_point, Illuminant::D50);
    }
}
