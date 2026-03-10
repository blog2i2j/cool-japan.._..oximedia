//! `ColorChecker` detection and analysis.
//!
//! This module provides tools for detecting and analyzing X-Rite `ColorChecker`
//! targets in images for camera calibration and profiling.

use crate::error::{CalibrationError, CalibrationResult};
use crate::{Lab, Rgb, Xyz};
use serde::{Deserialize, Serialize};

/// `ColorChecker` target type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorCheckerType {
    /// X-Rite `ColorChecker` Classic (24 patches).
    Classic24,
    /// X-Rite `ColorChecker` Passport (24 patches).
    Passport,
    /// Datacolor `SpyderCheckr` (48 patches).
    SpyderCheckr,
    /// Custom target.
    Custom,
}

impl ColorCheckerType {
    /// Get the number of patches for this `ColorChecker` type.
    #[must_use]
    pub const fn patch_count(&self) -> usize {
        match self {
            Self::Classic24 | Self::Passport => 24,
            Self::SpyderCheckr => 48,
            Self::Custom => 0,
        }
    }

    /// Get the grid dimensions (rows, columns) for this `ColorChecker` type.
    #[must_use]
    pub const fn grid_dimensions(&self) -> (usize, usize) {
        match self {
            Self::Classic24 | Self::Passport => (4, 6),
            Self::SpyderCheckr => (6, 8),
            Self::Custom => (0, 0),
        }
    }
}

/// A single patch color from a `ColorChecker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchColor {
    /// Patch index (0-based).
    pub index: usize,
    /// Measured RGB value (0.0-1.0).
    pub measured_rgb: Rgb,
    /// Reference RGB value (0.0-1.0).
    pub reference_rgb: Rgb,
    /// Reference LAB value.
    pub reference_lab: Lab,
    /// Reference XYZ value.
    pub reference_xyz: Xyz,
    /// Patch name/description.
    pub name: String,
}

/// `ColorChecker` target detected in an image.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColorChecker {
    /// Type of `ColorChecker`.
    pub checker_type: ColorCheckerType,
    /// Extracted patch colors.
    pub patches: Vec<PatchColor>,
    /// Bounding box of the `ColorChecker` in the image (x, y, width, height).
    pub bounding_box: Option<(f64, f64, f64, f64)>,
    /// Detection confidence (0.0-1.0).
    pub confidence: f64,
}

impl ColorChecker {
    /// Detect a `ColorChecker` in an image.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    /// * `checker_type` - Type of `ColorChecker` to detect
    ///
    /// # Errors
    ///
    /// Returns an error if `ColorChecker` detection fails.
    pub fn detect_in_image(
        _image_data: &[u8],
        checker_type: ColorCheckerType,
    ) -> CalibrationResult<Self> {
        // For now, return a placeholder implementation
        // In a real implementation, this would use computer vision to detect the ColorChecker
        let patches = Self::get_reference_patches(checker_type)?;

        Ok(Self {
            checker_type,
            patches,
            bounding_box: Some((100.0, 100.0, 800.0, 600.0)),
            confidence: 0.95,
        })
    }

    /// Get reference patch colors for a `ColorChecker` type.
    fn get_reference_patches(checker_type: ColorCheckerType) -> CalibrationResult<Vec<PatchColor>> {
        match checker_type {
            ColorCheckerType::Classic24 => Ok(Self::classic24_reference()),
            ColorCheckerType::Passport => Ok(Self::passport_reference()),
            ColorCheckerType::SpyderCheckr => Ok(Self::spydercheckr_reference()),
            ColorCheckerType::Custom => Err(CalibrationError::ColorCheckerNotFound(
                "Custom ColorChecker requires manual patch definition".to_string(),
            )),
        }
    }

    /// Get reference colors for X-Rite `ColorChecker` Classic 24.
    ///
    /// Reference values from X-Rite technical documentation.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn classic24_reference() -> Vec<PatchColor> {
        vec![
            // Row 1: Natural colors
            PatchColor {
                index: 0,
                measured_rgb: [0.451, 0.319, 0.262],
                reference_rgb: [0.451, 0.319, 0.262],
                reference_lab: [37.54, 14.37, 14.92],
                reference_xyz: [0.132, 0.117, 0.084],
                name: "Dark Skin".to_string(),
            },
            PatchColor {
                index: 1,
                measured_rgb: [0.769, 0.596, 0.491],
                reference_rgb: [0.769, 0.596, 0.491],
                reference_lab: [65.70, 19.29, 17.81],
                reference_xyz: [0.359, 0.325, 0.219],
                name: "Light Skin".to_string(),
            },
            PatchColor {
                index: 2,
                measured_rgb: [0.357, 0.457, 0.614],
                reference_rgb: [0.357, 0.457, 0.614],
                reference_lab: [49.32, -3.82, -22.54],
                reference_xyz: [0.185, 0.203, 0.390],
                name: "Blue Sky".to_string(),
            },
            PatchColor {
                index: 3,
                measured_rgb: [0.308, 0.400, 0.227],
                reference_rgb: [0.308, 0.400, 0.227],
                reference_lab: [43.46, -12.74, 22.72],
                reference_xyz: [0.133, 0.161, 0.062],
                name: "Foliage".to_string(),
            },
            PatchColor {
                index: 4,
                measured_rgb: [0.512, 0.494, 0.721],
                reference_rgb: [0.512, 0.494, 0.721],
                reference_lab: [54.94, 9.61, -24.79],
                reference_xyz: [0.259, 0.249, 0.534],
                name: "Blue Flower".to_string(),
            },
            PatchColor {
                index: 5,
                measured_rgb: [0.413, 0.746, 0.708],
                reference_rgb: [0.413, 0.746, 0.708],
                reference_lab: [70.48, -32.26, -0.37],
                reference_xyz: [0.377, 0.504, 0.514],
                name: "Bluish Green".to_string(),
            },
            // Row 2: Miscellaneous colors
            PatchColor {
                index: 6,
                measured_rgb: [0.913, 0.513, 0.124],
                reference_rgb: [0.913, 0.513, 0.124],
                reference_lab: [62.73, 35.83, 56.50],
                reference_xyz: [0.424, 0.307, 0.039],
                name: "Orange".to_string(),
            },
            PatchColor {
                index: 7,
                measured_rgb: [0.315, 0.377, 0.708],
                reference_rgb: [0.315, 0.377, 0.708],
                reference_lab: [39.43, 10.75, -45.17],
                reference_xyz: [0.151, 0.148, 0.514],
                name: "Purplish Blue".to_string(),
            },
            PatchColor {
                index: 8,
                measured_rgb: [0.720, 0.308, 0.385],
                reference_rgb: [0.720, 0.308, 0.385],
                reference_lab: [51.03, 48.13, 16.25],
                reference_xyz: [0.294, 0.192, 0.166],
                name: "Moderate Red".to_string(),
            },
            PatchColor {
                index: 9,
                measured_rgb: [0.329, 0.180, 0.438],
                reference_rgb: [0.329, 0.180, 0.438],
                reference_lab: [30.10, 22.54, -20.87],
                reference_xyz: [0.099, 0.061, 0.194],
                name: "Purple".to_string(),
            },
            PatchColor {
                index: 10,
                measured_rgb: [0.582, 0.804, 0.157],
                reference_rgb: [0.582, 0.804, 0.157],
                reference_lab: [72.75, -22.76, 57.26],
                reference_xyz: [0.396, 0.560, 0.061],
                name: "Yellow Green".to_string(),
            },
            PatchColor {
                index: 11,
                measured_rgb: [0.949, 0.628, 0.124],
                reference_rgb: [0.949, 0.628, 0.124],
                reference_lab: [71.94, 18.68, 67.86],
                reference_xyz: [0.523, 0.452, 0.045],
                name: "Orange Yellow".to_string(),
            },
            // Row 3: Primary and secondary colors
            PatchColor {
                index: 12,
                measured_rgb: [0.204, 0.247, 0.646],
                reference_rgb: [0.204, 0.247, 0.646],
                reference_lab: [28.78, 14.17, -49.57],
                reference_xyz: [0.079, 0.070, 0.427],
                name: "Blue".to_string(),
            },
            PatchColor {
                index: 13,
                measured_rgb: [0.303, 0.585, 0.284],
                reference_rgb: [0.303, 0.585, 0.284],
                reference_lab: [55.38, -37.40, 32.27],
                reference_xyz: [0.202, 0.317, 0.102],
                name: "Green".to_string(),
            },
            PatchColor {
                index: 14,
                measured_rgb: [0.733, 0.182, 0.167],
                reference_rgb: [0.733, 0.182, 0.167],
                reference_lab: [42.43, 53.05, 28.62],
                reference_xyz: [0.241, 0.132, 0.044],
                name: "Red".to_string(),
            },
            PatchColor {
                index: 15,
                measured_rgb: [0.949, 0.833, 0.124],
                reference_rgb: [0.949, 0.833, 0.124],
                reference_lab: [81.80, -0.57, 79.04],
                reference_xyz: [0.684, 0.748, 0.097],
                name: "Yellow".to_string(),
            },
            PatchColor {
                index: 16,
                measured_rgb: [0.741, 0.290, 0.604],
                reference_rgb: [0.741, 0.290, 0.604],
                reference_lab: [51.94, 48.93, -14.90],
                reference_xyz: [0.348, 0.212, 0.375],
                name: "Magenta".to_string(),
            },
            PatchColor {
                index: 17,
                measured_rgb: [0.160, 0.647, 0.765],
                reference_rgb: [0.160, 0.647, 0.765],
                reference_lab: [51.04, -28.63, -28.64],
                reference_xyz: [0.257, 0.331, 0.604],
                name: "Cyan".to_string(),
            },
            // Row 4: Grayscale
            PatchColor {
                index: 18,
                measured_rgb: [0.961, 0.961, 0.961],
                reference_rgb: [0.961, 0.961, 0.961],
                reference_lab: [96.24, -0.43, 1.19],
                reference_xyz: [0.875, 0.920, 0.940],
                name: "White".to_string(),
            },
            PatchColor {
                index: 19,
                measured_rgb: [0.800, 0.800, 0.800],
                reference_rgb: [0.800, 0.800, 0.800],
                reference_lab: [81.29, -0.57, 0.44],
                reference_xyz: [0.589, 0.620, 0.635],
                name: "Neutral 8".to_string(),
            },
            PatchColor {
                index: 20,
                measured_rgb: [0.635, 0.635, 0.635],
                reference_rgb: [0.635, 0.635, 0.635],
                reference_lab: [66.89, -0.75, -0.06],
                reference_xyz: [0.364, 0.383, 0.392],
                name: "Neutral 6.5".to_string(),
            },
            PatchColor {
                index: 21,
                measured_rgb: [0.486, 0.486, 0.486],
                reference_rgb: [0.486, 0.486, 0.486],
                reference_lab: [50.87, -0.15, -0.27],
                reference_xyz: [0.199, 0.209, 0.214],
                name: "Neutral 5".to_string(),
            },
            PatchColor {
                index: 22,
                measured_rgb: [0.337, 0.337, 0.337],
                reference_rgb: [0.337, 0.337, 0.337],
                reference_lab: [35.66, -0.37, -0.45],
                reference_xyz: [0.093, 0.098, 0.101],
                name: "Neutral 3.5".to_string(),
            },
            PatchColor {
                index: 23,
                measured_rgb: [0.196, 0.196, 0.196],
                reference_rgb: [0.196, 0.196, 0.196],
                reference_lab: [20.46, -0.13, -0.15],
                reference_xyz: [0.031, 0.032, 0.033],
                name: "Black".to_string(),
            },
        ]
    }

    /// Get reference colors for X-Rite `ColorChecker` Passport.
    fn passport_reference() -> Vec<PatchColor> {
        // Passport uses the same patches as Classic24
        Self::classic24_reference()
    }

    /// Get reference colors for Datacolor `SpyderCheckr`.
    fn spydercheckr_reference() -> Vec<PatchColor> {
        // SpyderCheckr has 48 patches - for brevity, we'll create a subset
        // In a real implementation, all 48 patches would be defined
        let mut patches = Vec::with_capacity(48);

        // Include the 24 standard ColorChecker patches
        patches.extend(Self::classic24_reference());

        // Add 24 additional patches (simplified for this implementation)
        for i in 24..48 {
            patches.push(PatchColor {
                index: i,
                measured_rgb: [0.5, 0.5, 0.5],
                reference_rgb: [0.5, 0.5, 0.5],
                reference_lab: [53.0, 0.0, 0.0],
                reference_xyz: [0.203, 0.214, 0.233],
                name: format!("Extra Patch {}", i - 23),
            });
        }

        patches
    }

    /// Extract patch colors from an image region.
    ///
    /// # Arguments
    ///
    /// * `image_data` - Raw image data (RGB format)
    /// * `patch_bounds` - Bounding boxes for each patch (x, y, width, height)
    ///
    /// # Errors
    ///
    /// Returns an error if patch extraction fails.
    pub fn extract_patches(
        _image_data: &[u8],
        _patch_bounds: &[(f64, f64, f64, f64)],
    ) -> CalibrationResult<Vec<Rgb>> {
        // For now, return a placeholder implementation
        // In a real implementation, this would extract the actual colors
        Ok(vec![[0.5, 0.5, 0.5]; 24])
    }

    /// Calculate the average color error (Delta E) between measured and reference patches.
    ///
    /// # Returns
    ///
    /// Average Delta E 2000 color difference.
    #[must_use]
    pub fn calculate_average_error(&self) -> f64 {
        if self.patches.is_empty() {
            return 0.0;
        }

        let total_error: f64 = self
            .patches
            .iter()
            .map(|patch| Self::delta_e_2000(&patch.measured_rgb, &patch.reference_rgb))
            .sum();

        total_error / self.patches.len() as f64
    }

    /// Calculate Delta E 2000 color difference between two RGB colors.
    ///
    /// This is a simplified implementation. A full implementation would
    /// convert to LAB and use the complete Delta E 2000 formula.
    fn delta_e_2000(rgb1: &Rgb, rgb2: &Rgb) -> f64 {
        // Simplified Euclidean distance in RGB space
        // A proper implementation would convert to LAB first
        let dr = rgb1[0] - rgb2[0];
        let dg = rgb1[1] - rgb2[1];
        let db = rgb1[2] - rgb2[2];

        (dr * dr + dg * dg + db * db).sqrt() * 100.0
    }

    /// Get the patch at a specific index.
    ///
    /// # Arguments
    ///
    /// * `index` - Patch index (0-based)
    ///
    /// # Returns
    ///
    /// The patch color, or None if index is out of bounds.
    #[must_use]
    pub fn get_patch(&self, index: usize) -> Option<&PatchColor> {
        self.patches.get(index)
    }

    /// Get the number of patches.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Verify that the `ColorChecker` has the correct number of patches.
    ///
    /// # Errors
    ///
    /// Returns an error if the patch count doesn't match the expected count.
    pub fn verify_patch_count(&self) -> CalibrationResult<()> {
        let expected = self.checker_type.patch_count();
        let actual = self.patches.len();

        if expected > 0 && actual != expected {
            return Err(CalibrationError::InvalidPatchCount { expected, actual });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colorchecker_type_patch_count() {
        assert_eq!(ColorCheckerType::Classic24.patch_count(), 24);
        assert_eq!(ColorCheckerType::Passport.patch_count(), 24);
        assert_eq!(ColorCheckerType::SpyderCheckr.patch_count(), 48);
    }

    #[test]
    fn test_colorchecker_type_grid_dimensions() {
        assert_eq!(ColorCheckerType::Classic24.grid_dimensions(), (4, 6));
        assert_eq!(ColorCheckerType::Passport.grid_dimensions(), (4, 6));
        assert_eq!(ColorCheckerType::SpyderCheckr.grid_dimensions(), (6, 8));
    }

    #[test]
    fn test_classic24_reference() {
        let patches = ColorChecker::classic24_reference();
        assert_eq!(patches.len(), 24);

        // Check first patch (Dark Skin)
        assert_eq!(patches[0].name, "Dark Skin");
        assert_eq!(patches[0].index, 0);

        // Check last patch (Black)
        assert_eq!(patches[23].name, "Black");
        assert_eq!(patches[23].index, 23);

        // Check white patch
        assert_eq!(patches[18].name, "White");
        assert!((patches[18].reference_rgb[0] - 0.961).abs() < 0.001);
    }

    #[test]
    fn test_spydercheckr_reference() {
        let patches = ColorChecker::spydercheckr_reference();
        assert_eq!(patches.len(), 48);
    }

    #[test]
    fn test_colorchecker_verify_patch_count() {
        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        assert!(checker.verify_patch_count().is_ok());
    }

    #[test]
    fn test_colorchecker_verify_patch_count_invalid() {
        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: vec![],
            bounding_box: None,
            confidence: 1.0,
        };

        assert!(checker.verify_patch_count().is_err());
    }

    #[test]
    fn test_colorchecker_get_patch() {
        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        let patch = checker.get_patch(0);
        assert!(patch.is_some());
        assert_eq!(
            patch.expect("expected patch to be Some/Ok").name,
            "Dark Skin"
        );

        assert!(checker.get_patch(100).is_none());
    }

    #[test]
    fn test_colorchecker_patch_count() {
        let checker = ColorChecker {
            checker_type: ColorCheckerType::Classic24,
            patches: ColorChecker::classic24_reference(),
            bounding_box: None,
            confidence: 1.0,
        };

        assert_eq!(checker.patch_count(), 24);
    }
}
