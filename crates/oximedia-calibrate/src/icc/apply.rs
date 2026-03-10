//! ICC profile application.
//!
//! This module provides tools for applying ICC color profiles to images.

use crate::error::{CalibrationError, CalibrationResult};
use crate::icc::IccProfile;
use crate::Rgb;

/// ICC profile applicator.
pub struct IccProfileApplicator {
    profile: IccProfile,
}

impl IccProfileApplicator {
    /// Create a new ICC profile applicator.
    #[must_use]
    pub fn new(profile: IccProfile) -> Self {
        Self { profile }
    }

    /// Apply the ICC profile to an RGB color.
    #[must_use]
    pub fn apply_to_color(&self, rgb: &Rgb) -> Rgb {
        // Convert RGB to XYZ using the profile

        // Convert back to RGB (for now, just return XYZ as RGB)
        // In a real implementation, this would convert to the target color space
        self.profile.rgb_to_xyz(rgb)
    }

    /// Apply the ICC profile to an entire image.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if application fails.
    pub fn apply_to_image(
        &self,
        image_data: &[u8],
        _width: usize,
        _height: usize,
    ) -> CalibrationResult<Vec<u8>> {
        if image_data.len() % 3 != 0 {
            return Err(CalibrationError::InvalidImageDimensions(
                "Image data length must be a multiple of 3 for RGB".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(image_data.len());

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let transformed = self.apply_to_color(&[r, g, b]);

            output.push((transformed[0] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((transformed[1] * 255.0).clamp(0.0, 255.0) as u8);
            output.push((transformed[2] * 255.0).clamp(0.0, 255.0) as u8);
        }

        Ok(output)
    }

    /// Convert between two ICC profiles.
    ///
    /// # Arguments
    ///
    /// * `source_profile` - Source color space profile
    /// * `target_profile` - Target color space profile
    /// * `rgb` - RGB color in source color space
    ///
    /// # Returns
    ///
    /// RGB color in target color space.
    #[must_use]
    pub fn convert_between_profiles(
        source_profile: &IccProfile,
        target_profile: &IccProfile,
        rgb: &Rgb,
    ) -> Rgb {
        // Convert source RGB to XYZ
        let xyz = source_profile.rgb_to_xyz(rgb);

        // Convert XYZ to target RGB
        target_profile.xyz_to_rgb(&xyz)
    }

    /// Get the underlying ICC profile.
    #[must_use]
    pub fn profile(&self) -> &IccProfile {
        &self.profile
    }

    /// Verify that the profile is valid for application.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile is invalid.
    pub fn verify(&self) -> CalibrationResult<()> {
        self.profile.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Illuminant;

    #[test]
    fn test_icc_profile_applicator_new() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let applicator = IccProfileApplicator::new(profile);
        assert_eq!(applicator.profile().description, "Test Profile");
    }

    #[test]
    fn test_icc_profile_applicator_apply_to_color() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let applicator = IccProfileApplicator::new(profile);
        let rgb = [0.5, 0.6, 0.7];
        let result = applicator.apply_to_color(&rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_icc_profile_applicator_apply_to_image() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let applicator = IccProfileApplicator::new(profile);
        let image = vec![128, 128, 128, 255, 0, 0];
        let result = applicator.apply_to_image(&image, 2, 1);

        assert!(result.is_ok());
        let output = result.expect("expected successful result");
        assert_eq!(output.len(), image.len());
    }

    #[test]
    fn test_icc_profile_applicator_invalid_image() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let applicator = IccProfileApplicator::new(profile);
        let image = vec![128, 128]; // Invalid: not a multiple of 3
        let result = applicator.apply_to_image(&image, 1, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_convert_between_profiles() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile1 = IccProfile::new("Profile 1".to_string(), identity, Illuminant::D65);

        let profile2 = IccProfile::new("Profile 2".to_string(), identity, Illuminant::D65);

        let rgb = [0.5, 0.6, 0.7];
        let result = IccProfileApplicator::convert_between_profiles(&profile1, &profile2, &rgb);

        // With identity matrices, should be unchanged
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.6).abs() < 1e-10);
        assert!((result[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_icc_profile_applicator_verify() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let profile = IccProfile::new("Test Profile".to_string(), identity, Illuminant::D65);

        let applicator = IccProfileApplicator::new(profile);
        assert!(applicator.verify().is_ok());
    }
}
