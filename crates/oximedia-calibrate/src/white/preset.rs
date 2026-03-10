//! White balance presets for common lighting conditions.
//!
//! This module provides predefined white balance settings for standard illuminants.

use crate::error::CalibrationResult;
use crate::Rgb;

/// White balance preset for common lighting conditions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WhiteBalancePreset {
    /// Daylight (5500K-6500K).
    Daylight,
    /// Cloudy/Overcast (6500K-8000K).
    Cloudy,
    /// Shade (7000K-8000K).
    Shade,
    /// Tungsten/Incandescent (2800K-3200K).
    Tungsten,
    /// Fluorescent (4000K-5000K).
    Fluorescent,
    /// Flash (5500K).
    Flash,
    /// Custom/Manual.
    Custom(f64, f64, f64),
}

impl WhiteBalancePreset {
    /// Get the RGB multipliers for this preset.
    #[must_use]
    pub fn multipliers(&self) -> [f64; 3] {
        match self {
            Self::Daylight => [1.0, 1.0, 1.0],
            Self::Cloudy => [1.0, 1.0, 1.15],
            Self::Shade => [1.0, 1.0, 1.25],
            Self::Tungsten => [1.0, 0.85, 0.65],
            Self::Fluorescent => [1.0, 0.95, 1.05],
            Self::Flash => [1.0, 1.0, 1.0],
            Self::Custom(r, g, b) => [*r, *g, *b],
        }
    }

    /// Get the approximate color temperature in Kelvin.
    #[must_use]
    pub fn color_temperature(&self) -> u32 {
        match self {
            Self::Daylight => 6000,
            Self::Cloudy => 7000,
            Self::Shade => 7500,
            Self::Tungsten => 3000,
            Self::Fluorescent => 4500,
            Self::Flash => 5500,
            Self::Custom(_, _, _) => 5500,
        }
    }

    /// Apply this white balance preset to an RGB color.
    #[must_use]
    pub fn apply(&self, rgb: &Rgb) -> Rgb {
        let mult = self.multipliers();
        [
            (rgb[0] * mult[0]).clamp(0.0, 1.0),
            (rgb[1] * mult[1]).clamp(0.0, 1.0),
            (rgb[2] * mult[2]).clamp(0.0, 1.0),
        ]
    }

    /// Apply this white balance preset to an image.
    ///
    /// # Errors
    ///
    /// Returns an error if image dimensions are invalid.
    pub fn apply_to_image(&self, image_data: &[u8]) -> CalibrationResult<Vec<u8>> {
        let mut output = Vec::with_capacity(image_data.len());
        let mult = self.multipliers();

        for chunk in image_data.chunks_exact(3) {
            let r = f64::from(chunk[0]) / 255.0;
            let g = f64::from(chunk[1]) / 255.0;
            let b = f64::from(chunk[2]) / 255.0;

            let corrected = [
                (r * mult[0]).clamp(0.0, 1.0),
                (g * mult[1]).clamp(0.0, 1.0),
                (b * mult[2]).clamp(0.0, 1.0),
            ];

            output.push((corrected[0] * 255.0).round() as u8);
            output.push((corrected[1] * 255.0).round() as u8);
            output.push((corrected[2] * 255.0).round() as u8);
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_daylight_multipliers() {
        let preset = WhiteBalancePreset::Daylight;
        let mult = preset.multipliers();

        assert!((mult[0] - 1.0).abs() < 1e-10);
        assert!((mult[1] - 1.0).abs() < 1e-10);
        assert!((mult[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tungsten_multipliers() {
        let preset = WhiteBalancePreset::Tungsten;
        let mult = preset.multipliers();

        assert!((mult[0] - 1.0).abs() < 1e-10);
        assert!((mult[1] - 0.85).abs() < 1e-10);
        assert!((mult[2] - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_custom_multipliers() {
        let preset = WhiteBalancePreset::Custom(1.2, 1.0, 0.8);
        let mult = preset.multipliers();

        assert!((mult[0] - 1.2).abs() < 1e-10);
        assert!((mult[1] - 1.0).abs() < 1e-10);
        assert!((mult[2] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_color_temperatures() {
        assert_eq!(WhiteBalancePreset::Daylight.color_temperature(), 6000);
        assert_eq!(WhiteBalancePreset::Tungsten.color_temperature(), 3000);
        assert_eq!(WhiteBalancePreset::Fluorescent.color_temperature(), 4500);
    }

    #[test]
    fn test_apply() {
        let preset = WhiteBalancePreset::Daylight;
        let rgb = [0.5, 0.5, 0.5];
        let result = preset.apply(&rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_apply_tungsten() {
        let preset = WhiteBalancePreset::Tungsten;
        let rgb = [0.5, 0.5, 0.5];
        let result = preset.apply(&rgb);

        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 0.425).abs() < 1e-10);
        assert!((result[2] - 0.325).abs() < 1e-10);
    }

    #[test]
    fn test_apply_to_image() {
        let preset = WhiteBalancePreset::Daylight;
        let image = vec![128, 128, 128, 255, 0, 0];
        let result = preset.apply_to_image(&image);

        assert!(result.is_ok());
        let output = result.expect("expected successful result");
        assert_eq!(output.len(), image.len());
    }
}
