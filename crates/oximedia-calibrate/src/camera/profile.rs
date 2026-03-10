//! Camera profile generation and management.
//!
//! This module provides tools for generating and managing camera color profiles.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Illuminant, Matrix3x3};
use serde::{Deserialize, Serialize};

/// Camera profile quality level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileQuality {
    /// Draft quality (fast, lower accuracy).
    Draft,
    /// Standard quality (balanced speed and accuracy).
    Standard,
    /// High quality (slower, higher accuracy).
    High,
    /// Maximum quality (slowest, maximum accuracy).
    Maximum,
}

impl ProfileQuality {
    /// Get the number of optimization iterations for this quality level.
    #[must_use]
    pub const fn optimization_iterations(&self) -> usize {
        match self {
            Self::Draft => 10,
            Self::Standard => 50,
            Self::High => 200,
            Self::Maximum => 1000,
        }
    }

    /// Get the convergence threshold for this quality level.
    #[must_use]
    pub const fn convergence_threshold(&self) -> f64 {
        match self {
            Self::Draft => 1e-3,
            Self::Standard => 1e-4,
            Self::High => 1e-5,
            Self::Maximum => 1e-6,
        }
    }
}

/// Camera color profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraProfile {
    /// Camera manufacturer.
    pub manufacturer: String,
    /// Camera model.
    pub model: String,
    /// Profile name.
    pub profile_name: String,
    /// Target illuminant.
    pub illuminant: Illuminant,
    /// Forward color matrix (camera RGB to XYZ).
    pub forward_matrix: Matrix3x3,
    /// Inverse color matrix (XYZ to camera RGB).
    pub inverse_matrix: Matrix3x3,
    /// White balance multipliers (R, G, B).
    pub white_balance: [f64; 3],
    /// Black level (0.0-1.0).
    pub black_level: f64,
    /// White level (0.0-1.0).
    pub white_level: f64,
    /// Profile quality used for generation.
    pub quality: ProfileQuality,
    /// Average color error (Delta E).
    pub average_error: f64,
    /// Maximum color error (Delta E).
    pub max_error: f64,
}

impl CameraProfile {
    /// Create a new camera profile.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manufacturer: String,
        model: String,
        profile_name: String,
        illuminant: Illuminant,
        forward_matrix: Matrix3x3,
        inverse_matrix: Matrix3x3,
        quality: ProfileQuality,
    ) -> Self {
        Self {
            manufacturer,
            model,
            profile_name,
            illuminant,
            forward_matrix,
            inverse_matrix,
            white_balance: [1.0, 1.0, 1.0],
            black_level: 0.0,
            white_level: 1.0,
            quality,
            average_error: 0.0,
            max_error: 0.0,
        }
    }

    /// Create a default identity profile.
    #[must_use]
    pub fn identity(manufacturer: String, model: String) -> Self {
        let identity_matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        Self::new(
            manufacturer,
            model,
            "Identity".to_string(),
            Illuminant::D65,
            identity_matrix,
            identity_matrix,
            ProfileQuality::Standard,
        )
    }

    /// Apply the profile to convert camera RGB to XYZ.
    #[must_use]
    pub fn camera_to_xyz(&self, rgb: &[f64; 3]) -> [f64; 3] {
        self.apply_matrix(&self.forward_matrix, rgb)
    }

    /// Apply the profile to convert XYZ to camera RGB.
    #[must_use]
    pub fn xyz_to_camera(&self, xyz: &[f64; 3]) -> [f64; 3] {
        self.apply_matrix(&self.inverse_matrix, xyz)
    }

    /// Apply a 3x3 matrix to a color value.
    fn apply_matrix(&self, matrix: &Matrix3x3, color: &[f64; 3]) -> [f64; 3] {
        [
            matrix[0][0] * color[0] + matrix[0][1] * color[1] + matrix[0][2] * color[2],
            matrix[1][0] * color[0] + matrix[1][1] * color[1] + matrix[1][2] * color[2],
            matrix[2][0] * color[0] + matrix[2][1] * color[1] + matrix[2][2] * color[2],
        ]
    }

    /// Set white balance multipliers.
    pub fn set_white_balance(&mut self, multipliers: [f64; 3]) {
        self.white_balance = multipliers;
    }

    /// Get white balance multipliers.
    #[must_use]
    pub fn get_white_balance(&self) -> [f64; 3] {
        self.white_balance
    }

    /// Set black and white levels.
    pub fn set_levels(&mut self, black_level: f64, white_level: f64) {
        self.black_level = black_level;
        self.white_level = white_level;
    }

    /// Apply black and white level correction to a linear RGB value.
    #[must_use]
    pub fn apply_levels(&self, rgb: &[f64; 3]) -> [f64; 3] {
        let range = self.white_level - self.black_level;
        if range <= 0.0 {
            return *rgb;
        }

        [
            ((rgb[0] - self.black_level) / range).clamp(0.0, 1.0),
            ((rgb[1] - self.black_level) / range).clamp(0.0, 1.0),
            ((rgb[2] - self.black_level) / range).clamp(0.0, 1.0),
        ]
    }

    /// Set color error statistics.
    pub fn set_error_stats(&mut self, average_error: f64, max_error: f64) {
        self.average_error = average_error;
        self.max_error = max_error;
    }

    /// Validate the profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile is invalid.
    pub fn validate(&self) -> CalibrationResult<()> {
        // Check that matrices are not all zeros
        let forward_sum: f64 = self.forward_matrix.iter().flatten().sum();
        let inverse_sum: f64 = self.inverse_matrix.iter().flatten().sum();

        if forward_sum.abs() < 1e-10 {
            return Err(CalibrationError::ProfileGenerationFailed(
                "Forward matrix is zero".to_string(),
            ));
        }

        if inverse_sum.abs() < 1e-10 {
            return Err(CalibrationError::ProfileGenerationFailed(
                "Inverse matrix is zero".to_string(),
            ));
        }

        // Check that white balance multipliers are positive
        if self.white_balance.iter().any(|&x| x <= 0.0) {
            return Err(CalibrationError::ProfileGenerationFailed(
                "White balance multipliers must be positive".to_string(),
            ));
        }

        // Check that black level is less than white level
        if self.black_level >= self.white_level {
            return Err(CalibrationError::ProfileGenerationFailed(
                "Black level must be less than white level".to_string(),
            ));
        }

        Ok(())
    }

    /// Serialize the profile to JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> CalibrationResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            CalibrationError::ProfileGenerationFailed(format!("Failed to serialize profile: {e}"))
        })
    }

    /// Deserialize a profile from JSON.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> CalibrationResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            CalibrationError::ProfileGenerationFailed(format!("Failed to deserialize profile: {e}"))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_quality_iterations() {
        assert_eq!(ProfileQuality::Draft.optimization_iterations(), 10);
        assert_eq!(ProfileQuality::Standard.optimization_iterations(), 50);
        assert_eq!(ProfileQuality::High.optimization_iterations(), 200);
        assert_eq!(ProfileQuality::Maximum.optimization_iterations(), 1000);
    }

    #[test]
    fn test_profile_quality_threshold() {
        assert!((ProfileQuality::Draft.convergence_threshold() - 1e-3).abs() < 1e-10);
        assert!((ProfileQuality::Standard.convergence_threshold() - 1e-4).abs() < 1e-10);
        assert!((ProfileQuality::High.convergence_threshold() - 1e-5).abs() < 1e-10);
        assert!((ProfileQuality::Maximum.convergence_threshold() - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_camera_profile_identity() {
        let profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        assert_eq!(profile.manufacturer, "Test");
        assert_eq!(profile.model, "Camera");
        assert_eq!(profile.illuminant, Illuminant::D65);

        let rgb = [0.5, 0.6, 0.7];
        let xyz = profile.camera_to_xyz(&rgb);
        assert!((xyz[0] - 0.5).abs() < 1e-10);
        assert!((xyz[1] - 0.6).abs() < 1e-10);
        assert!((xyz[2] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_camera_profile_white_balance() {
        let mut profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        let wb = [1.2, 1.0, 0.8];
        profile.set_white_balance(wb);
        assert_eq!(profile.get_white_balance(), wb);
    }

    #[test]
    fn test_camera_profile_levels() {
        let mut profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        profile.set_levels(0.1, 0.9);
        assert!((profile.black_level - 0.1).abs() < 1e-10);
        assert!((profile.white_level - 0.9).abs() < 1e-10);

        let rgb = [0.5, 0.5, 0.5];
        let corrected = profile.apply_levels(&rgb);
        assert!((corrected[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_camera_profile_validate() {
        let profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        assert!(profile.validate().is_ok());
    }

    #[test]
    fn test_camera_profile_validate_invalid() {
        let mut profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        profile.forward_matrix = [[0.0; 3]; 3];
        assert!(profile.validate().is_err());
    }

    #[test]
    fn test_camera_profile_json_roundtrip() {
        let profile = CameraProfile::identity("Test".to_string(), "Camera".to_string());

        let json = profile
            .to_json()
            .expect("JSON serialization should succeed");
        let restored = CameraProfile::from_json(&json).expect("unexpected None/Err");

        assert_eq!(restored.manufacturer, profile.manufacturer);
        assert_eq!(restored.model, profile.model);
        assert_eq!(restored.illuminant, profile.illuminant);
    }
}
