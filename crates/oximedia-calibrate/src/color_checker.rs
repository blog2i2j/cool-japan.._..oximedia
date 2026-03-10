#![allow(dead_code)]
//! Color checker target definitions and patch reference data.
//!
//! This module provides reference color values for standard color checker targets
//! such as the X-Rite `ColorChecker` Classic (24 patches), `ColorChecker` SG, and
//! custom targets. It supports multiple color spaces for reference data.

/// Type of color checker target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckerType {
    /// X-Rite `ColorChecker` Classic (24 patches, 4x6 grid).
    Classic24,
    /// X-Rite `ColorChecker` SG (140 patches).
    Sg140,
    /// X-Rite `ColorChecker` Passport (compact 24 patches).
    Passport,
    /// Datacolor `SpyderCheckr` (48 patches).
    SpyderCheckr48,
    /// User-defined custom target.
    Custom,
}

/// Color space for reference values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceColorSpace {
    /// sRGB color space.
    Srgb,
    /// CIE L*a*b* (D50 illuminant).
    LabD50,
    /// CIE L*a*b* (D65 illuminant).
    LabD65,
    /// CIE XYZ tristimulus.
    Xyz,
}

/// A single patch on a color checker.
#[derive(Debug, Clone)]
pub struct CheckerPatch {
    /// Patch name or identifier.
    pub name: String,
    /// Row index on the target grid.
    pub row: usize,
    /// Column index on the target grid.
    pub col: usize,
    /// Reference L*a*b* values (D50).
    pub lab_d50: [f64; 3],
    /// Reference sRGB values (0-255 range).
    pub srgb: [u8; 3],
    /// Whether this patch is a neutral/gray.
    pub is_neutral: bool,
}

impl CheckerPatch {
    /// Create a new checker patch.
    #[must_use]
    pub fn new(
        name: &str,
        row: usize,
        col: usize,
        lab_d50: [f64; 3],
        srgb: [u8; 3],
        is_neutral: bool,
    ) -> Self {
        Self {
            name: name.to_string(),
            row,
            col,
            lab_d50,
            srgb,
            is_neutral,
        }
    }

    /// Get the lightness (L*) value.
    #[must_use]
    pub fn lightness(&self) -> f64 {
        self.lab_d50[0]
    }

    /// Get the chroma (C*) value.
    #[must_use]
    pub fn chroma(&self) -> f64 {
        (self.lab_d50[1].powi(2) + self.lab_d50[2].powi(2)).sqrt()
    }

    /// Get the hue angle in degrees.
    #[must_use]
    pub fn hue_angle(&self) -> f64 {
        let h = self.lab_d50[2].atan2(self.lab_d50[1]).to_degrees();
        if h < 0.0 {
            h + 360.0
        } else {
            h
        }
    }
}

/// A complete color checker target definition.
#[derive(Debug, Clone)]
pub struct ColorCheckerTarget {
    /// Target type.
    pub checker_type: CheckerType,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// All patches on the target.
    pub patches: Vec<CheckerPatch>,
}

impl ColorCheckerTarget {
    /// Create the standard X-Rite `ColorChecker` Classic 24-patch target.
    #[must_use]
    pub fn classic_24() -> Self {
        let patches = vec![
            CheckerPatch::new(
                "Dark Skin",
                0,
                0,
                [37.986, 13.555, 14.059],
                [115, 82, 68],
                false,
            ),
            CheckerPatch::new(
                "Light Skin",
                0,
                1,
                [65.711, 18.130, 17.810],
                [194, 150, 130],
                false,
            ),
            CheckerPatch::new(
                "Blue Sky",
                0,
                2,
                [49.927, -4.880, -21.925],
                [98, 122, 157],
                false,
            ),
            CheckerPatch::new(
                "Foliage",
                0,
                3,
                [43.139, -13.095, 21.905],
                [87, 108, 67],
                false,
            ),
            CheckerPatch::new(
                "Blue Flower",
                0,
                4,
                [55.112, 8.844, -25.399],
                [133, 128, 177],
                false,
            ),
            CheckerPatch::new(
                "Bluish Green",
                0,
                5,
                [70.719, -33.397, -0.199],
                [103, 189, 170],
                false,
            ),
            CheckerPatch::new(
                "Orange",
                1,
                0,
                [62.661, 36.067, 57.096],
                [214, 126, 44],
                false,
            ),
            CheckerPatch::new(
                "Purplish Blue",
                1,
                1,
                [40.020, 10.410, -45.964],
                [80, 91, 166],
                false,
            ),
            CheckerPatch::new(
                "Moderate Red",
                1,
                2,
                [51.124, 48.239, 16.248],
                [193, 90, 99],
                false,
            ),
            CheckerPatch::new(
                "Purple",
                1,
                3,
                [30.325, 22.976, -21.587],
                [94, 60, 108],
                false,
            ),
            CheckerPatch::new(
                "Yellow Green",
                1,
                4,
                [72.532, -23.709, 57.255],
                [157, 188, 64],
                false,
            ),
            CheckerPatch::new(
                "Orange Yellow",
                1,
                5,
                [71.941, 19.363, 67.857],
                [224, 163, 46],
                false,
            ),
            CheckerPatch::new(
                "Blue",
                2,
                0,
                [28.778, 14.179, -50.297],
                [56, 61, 150],
                false,
            ),
            CheckerPatch::new(
                "Green",
                2,
                1,
                [55.261, -38.342, 31.370],
                [70, 148, 73],
                false,
            ),
            CheckerPatch::new("Red", 2, 2, [42.101, 53.378, 28.190], [175, 54, 60], false),
            CheckerPatch::new(
                "Yellow",
                2,
                3,
                [81.733, 4.039, 79.819],
                [231, 199, 31],
                false,
            ),
            CheckerPatch::new(
                "Magenta",
                2,
                4,
                [51.935, 49.986, -14.574],
                [187, 86, 149],
                false,
            ),
            CheckerPatch::new(
                "Cyan",
                2,
                5,
                [51.038, -28.631, -28.638],
                [8, 133, 161],
                false,
            ),
            CheckerPatch::new(
                "White",
                3,
                0,
                [96.539, -0.425, 1.186],
                [243, 243, 242],
                true,
            ),
            CheckerPatch::new(
                "Neutral 8",
                3,
                1,
                [81.257, -0.638, -0.335],
                [200, 200, 200],
                true,
            ),
            CheckerPatch::new(
                "Neutral 6.5",
                3,
                2,
                [66.766, -0.734, -0.504],
                [160, 160, 160],
                true,
            ),
            CheckerPatch::new(
                "Neutral 5",
                3,
                3,
                [50.867, -0.153, -0.270],
                [122, 122, 121],
                true,
            ),
            CheckerPatch::new(
                "Neutral 3.5",
                3,
                4,
                [35.656, -0.421, -1.231],
                [85, 85, 85],
                true,
            ),
            CheckerPatch::new("Black", 3, 5, [20.461, -0.079, -0.973], [52, 52, 52], true),
        ];

        Self {
            checker_type: CheckerType::Classic24,
            rows: 4,
            cols: 6,
            patches,
        }
    }

    /// Create a custom target with user-supplied patches.
    #[must_use]
    pub fn custom(rows: usize, cols: usize, patches: Vec<CheckerPatch>) -> Self {
        Self {
            checker_type: CheckerType::Custom,
            rows,
            cols,
            patches,
        }
    }

    /// Get the total number of patches.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Get all neutral/gray patches.
    #[must_use]
    pub fn neutral_patches(&self) -> Vec<&CheckerPatch> {
        self.patches.iter().filter(|p| p.is_neutral).collect()
    }

    /// Get all chromatic (non-neutral) patches.
    #[must_use]
    pub fn chromatic_patches(&self) -> Vec<&CheckerPatch> {
        self.patches.iter().filter(|p| !p.is_neutral).collect()
    }

    /// Find a patch by name.
    #[must_use]
    pub fn find_patch(&self, name: &str) -> Option<&CheckerPatch> {
        self.patches.iter().find(|p| p.name == name)
    }

    /// Find a patch by grid position.
    #[must_use]
    pub fn patch_at(&self, row: usize, col: usize) -> Option<&CheckerPatch> {
        self.patches.iter().find(|p| p.row == row && p.col == col)
    }

    /// Get the average lightness of neutral patches (useful for exposure check).
    #[must_use]
    pub fn neutral_avg_lightness(&self) -> f64 {
        let neutrals = self.neutral_patches();
        if neutrals.is_empty() {
            return 0.0;
        }
        neutrals.iter().map(|p| p.lightness()).sum::<f64>() / neutrals.len() as f64
    }

    /// Get the lightness range of neutral patches (min, max).
    #[must_use]
    pub fn neutral_lightness_range(&self) -> (f64, f64) {
        let neutrals = self.neutral_patches();
        if neutrals.is_empty() {
            return (0.0, 0.0);
        }
        let min = neutrals
            .iter()
            .map(|p| p.lightness())
            .fold(f64::MAX, f64::min);
        let max = neutrals
            .iter()
            .map(|p| p.lightness())
            .fold(f64::MIN, f64::max);
        (min, max)
    }
}

/// Compare measured patch values against the reference target.
#[derive(Debug)]
pub struct PatchComparison {
    /// Patch name.
    pub name: String,
    /// Reference Lab values.
    pub reference_lab: [f64; 3],
    /// Measured Lab values.
    pub measured_lab: [f64; 3],
    /// Delta E (CIE76 approximation).
    pub delta_e: f64,
}

/// Compute CIE76 Delta E between two Lab colors.
#[must_use]
pub fn delta_e_cie76(lab1: &[f64; 3], lab2: &[f64; 3]) -> f64 {
    ((lab1[0] - lab2[0]).powi(2) + (lab1[1] - lab2[1]).powi(2) + (lab1[2] - lab2[2]).powi(2)).sqrt()
}

/// Compare measured Lab values against a reference target.
#[must_use]
pub fn compare_patches(target: &ColorCheckerTarget, measured: &[[f64; 3]]) -> Vec<PatchComparison> {
    target
        .patches
        .iter()
        .zip(measured.iter())
        .map(|(patch, meas)| {
            let de = delta_e_cie76(&patch.lab_d50, meas);
            PatchComparison {
                name: patch.name.clone(),
                reference_lab: patch.lab_d50,
                measured_lab: *meas,
                delta_e: de,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classic24_patch_count() {
        let target = ColorCheckerTarget::classic_24();
        assert_eq!(target.patch_count(), 24);
    }

    #[test]
    fn test_classic24_neutral_count() {
        let target = ColorCheckerTarget::classic_24();
        assert_eq!(target.neutral_patches().len(), 6);
    }

    #[test]
    fn test_classic24_chromatic_count() {
        let target = ColorCheckerTarget::classic_24();
        assert_eq!(target.chromatic_patches().len(), 18);
    }

    #[test]
    fn test_find_patch_by_name() {
        let target = ColorCheckerTarget::classic_24();
        let white = target.find_patch("White").expect("unexpected None/Err");
        assert!(white.is_neutral);
        assert!(white.lightness() > 90.0);
    }

    #[test]
    fn test_find_patch_missing() {
        let target = ColorCheckerTarget::classic_24();
        assert!(target.find_patch("NonExistent").is_none());
    }

    #[test]
    fn test_patch_at_position() {
        let target = ColorCheckerTarget::classic_24();
        let patch = target.patch_at(0, 0).expect("unexpected None/Err");
        assert_eq!(patch.name, "Dark Skin");
    }

    #[test]
    fn test_patch_chroma() {
        let patch = CheckerPatch::new("Test", 0, 0, [50.0, 30.0, 40.0], [128, 128, 128], false);
        let expected = (30.0_f64.powi(2) + 40.0_f64.powi(2)).sqrt();
        assert!((patch.chroma() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_patch_hue_angle() {
        let patch = CheckerPatch::new("Test", 0, 0, [50.0, 0.0, 50.0], [128, 128, 128], false);
        assert!((patch.hue_angle() - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_neutral_avg_lightness() {
        let target = ColorCheckerTarget::classic_24();
        let avg = target.neutral_avg_lightness();
        assert!(avg > 20.0);
        assert!(avg < 100.0);
    }

    #[test]
    fn test_neutral_lightness_range() {
        let target = ColorCheckerTarget::classic_24();
        let (min, max) = target.neutral_lightness_range();
        assert!(min < max);
        assert!(min > 10.0);
        assert!(max > 90.0);
    }

    #[test]
    fn test_delta_e_identical() {
        let lab = [50.0, 10.0, -20.0];
        assert!((delta_e_cie76(&lab, &lab) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_delta_e_known() {
        let lab1 = [50.0, 0.0, 0.0];
        let lab2 = [53.0, 4.0, 0.0];
        let expected = (9.0_f64 + 16.0).sqrt();
        assert!((delta_e_cie76(&lab1, &lab2) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compare_patches() {
        let target = ColorCheckerTarget::classic_24();
        let measured: Vec<[f64; 3]> = target.patches.iter().map(|p| p.lab_d50).collect();
        let comparisons = compare_patches(&target, &measured);
        assert_eq!(comparisons.len(), 24);
        for c in &comparisons {
            assert!(c.delta_e < 1e-10); // Identical values
        }
    }

    #[test]
    fn test_custom_target() {
        let patches = vec![
            CheckerPatch::new("A", 0, 0, [50.0, 0.0, 0.0], [128, 128, 128], true),
            CheckerPatch::new("B", 0, 1, [50.0, 20.0, 20.0], [200, 100, 100], false),
        ];
        let target = ColorCheckerTarget::custom(1, 2, patches);
        assert_eq!(target.patch_count(), 2);
        assert_eq!(target.checker_type, CheckerType::Custom);
    }
}
