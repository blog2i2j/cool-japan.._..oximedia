//! HSL qualifiers for secondary color correction.

use super::types::RgbColor;

/// HSL range qualifier for secondary color correction.
#[derive(Clone, Debug)]
pub struct HslQualifier {
    /// Hue range (min, max) in degrees (0-360).
    pub hue_range: (f64, f64),
    /// Saturation range (min, max) (0.0-1.0).
    pub sat_range: (f64, f64),
    /// Luminance range (min, max) (0.0-1.0).
    pub lum_range: (f64, f64),
    /// Soft edge feathering amount (0.0-1.0).
    pub feather: f64,
    /// Enable hue qualification.
    pub qualify_hue: bool,
    /// Enable saturation qualification.
    pub qualify_sat: bool,
    /// Enable luminance qualification.
    pub qualify_lum: bool,
    /// Invert the mask.
    pub invert: bool,
    /// Blur radius for mask softening.
    pub blur_radius: f64,
    /// Denoise strength for mask cleanup.
    pub denoise: f64,
    /// Matte black point (0.0-1.0).
    pub black_point: f64,
    /// Matte white point (0.0-1.0).
    pub white_point: f64,
}

impl Default for HslQualifier {
    fn default() -> Self {
        Self {
            hue_range: (0.0, 360.0),
            sat_range: (0.0, 1.0),
            lum_range: (0.0, 1.0),
            feather: 0.1,
            qualify_hue: false,
            qualify_sat: false,
            qualify_lum: false,
            invert: false,
            blur_radius: 0.0,
            denoise: 0.0,
            black_point: 0.0,
            white_point: 1.0,
        }
    }
}

impl HslQualifier {
    /// Create a new HSL qualifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set hue range.
    #[must_use]
    pub fn with_hue_range(mut self, min: f64, max: f64) -> Self {
        self.hue_range = (min, max);
        self.qualify_hue = true;
        self
    }

    /// Set saturation range.
    #[must_use]
    pub fn with_sat_range(mut self, min: f64, max: f64) -> Self {
        self.sat_range = (min, max);
        self.qualify_sat = true;
        self
    }

    /// Set luminance range.
    #[must_use]
    pub fn with_lum_range(mut self, min: f64, max: f64) -> Self {
        self.lum_range = (min, max);
        self.qualify_lum = true;
        self
    }

    /// Set feathering amount.
    #[must_use]
    pub const fn with_feather(mut self, feather: f64) -> Self {
        self.feather = feather;
        self
    }

    /// Set invert flag.
    #[must_use]
    pub const fn with_invert(mut self, invert: bool) -> Self {
        self.invert = invert;
        self
    }

    /// Set blur radius.
    #[must_use]
    pub const fn with_blur(mut self, radius: f64) -> Self {
        self.blur_radius = radius;
        self
    }

    /// Set denoise strength.
    #[must_use]
    pub const fn with_denoise(mut self, strength: f64) -> Self {
        self.denoise = strength;
        self
    }

    /// Set matte black and white points.
    #[must_use]
    pub const fn with_matte_range(mut self, black: f64, white: f64) -> Self {
        self.black_point = black;
        self.white_point = white;
        self
    }

    /// Calculate the mask value for a color (0.0 = no effect, 1.0 = full effect).
    #[must_use]
    pub fn calculate_mask(&self, color: RgbColor) -> f64 {
        let hsl = color.to_hsl();

        let mut mask = 1.0;

        // Check hue range
        if self.qualify_hue {
            let hue_mask = self.calculate_hue_mask(hsl.h);
            mask *= hue_mask;
        }

        // Check saturation range
        if self.qualify_sat {
            let sat_mask = self.calculate_range_mask(hsl.s, self.sat_range.0, self.sat_range.1);
            mask *= sat_mask;
        }

        // Check luminance range
        if self.qualify_lum {
            let lum_mask = self.calculate_range_mask(hsl.l, self.lum_range.0, self.lum_range.1);
            mask *= lum_mask;
        }

        // Apply matte refinement (levels adjustment)
        mask = self.refine_matte(mask);

        // Invert if requested
        if self.invert {
            mask = 1.0 - mask;
        }

        mask
    }

    /// Refine the matte using black and white point adjustments.
    fn refine_matte(&self, mask: f64) -> f64 {
        if (self.black_point - 0.0).abs() < f64::EPSILON
            && (self.white_point - 1.0).abs() < f64::EPSILON
        {
            return mask;
        }

        // Levels adjustment
        if mask <= self.black_point {
            0.0
        } else if mask >= self.white_point {
            1.0
        } else {
            // Linear remap from [black, white] to [0, 1]
            (mask - self.black_point) / (self.white_point - self.black_point)
        }
    }

    /// Apply denoise to mask (simple threshold-based).
    #[allow(dead_code)]
    fn denoise_mask(&self, mask: f64) -> f64 {
        if self.denoise <= 0.0 {
            return mask;
        }

        // Simple threshold denoising
        let threshold = self.denoise * 0.1;
        if mask < threshold {
            0.0
        } else if mask > 1.0 - threshold {
            1.0
        } else {
            // Smooth transition
            let t = (mask - threshold) / (1.0 - 2.0 * threshold);
            t.clamp(0.0, 1.0)
        }
    }

    /// Calculate hue mask with wrapping support.
    fn calculate_hue_mask(&self, hue: f64) -> f64 {
        let (min, max) = self.hue_range;

        if min <= max {
            // Normal range
            self.calculate_range_mask(hue, min, max)
        } else {
            // Wrapped range (e.g., 350-10 degrees)
            let mask1 = self.calculate_range_mask(hue, min, 360.0);
            let mask2 = self.calculate_range_mask(hue, 0.0, max);
            mask1.max(mask2)
        }
    }

    /// Calculate mask for a linear range with feathering.
    fn calculate_range_mask(&self, value: f64, min: f64, max: f64) -> f64 {
        if value < min {
            // Below range
            if self.feather > 0.0 {
                let dist = min - value;
                let fade = (1.0 - (dist / self.feather)).clamp(0.0, 1.0);
                fade * fade // Smooth falloff
            } else {
                0.0
            }
        } else if value > max {
            // Above range
            if self.feather > 0.0 {
                let dist = value - max;
                let fade = (1.0 - (dist / self.feather)).clamp(0.0, 1.0);
                fade * fade // Smooth falloff
            } else {
                0.0
            }
        } else {
            // Within range
            1.0
        }
    }

    /// Apply a correction to a color using this qualifier.
    #[must_use]
    pub fn apply<F>(&self, color: RgbColor, correction: F) -> RgbColor
    where
        F: Fn(RgbColor) -> RgbColor,
    {
        let mask = self.calculate_mask(color);

        if mask <= 0.0 {
            return color;
        }

        let corrected = correction(color);

        if mask >= 1.0 {
            corrected
        } else {
            // Blend based on mask
            color.lerp(&corrected, mask)
        }
    }
}

/// Multiple HSL qualifiers that can be combined.
#[derive(Clone, Debug)]
pub struct MultiQualifier {
    /// Individual qualifiers.
    qualifiers: Vec<HslQualifier>,
    /// Combination mode.
    mode: QualifierCombineMode,
}

/// Mode for combining multiple qualifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualifierCombineMode {
    /// Union (OR) - pixel matches if it matches any qualifier.
    Union,
    /// Intersection (AND) - pixel matches only if it matches all qualifiers.
    Intersection,
    /// Difference - first minus others.
    Difference,
}

impl MultiQualifier {
    /// Create a new multi-qualifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            qualifiers: Vec::new(),
            mode: QualifierCombineMode::Union,
        }
    }

    /// Add a qualifier.
    pub fn add_qualifier(&mut self, qualifier: HslQualifier) {
        self.qualifiers.push(qualifier);
    }

    /// Set combine mode.
    pub fn set_mode(&mut self, mode: QualifierCombineMode) {
        self.mode = mode;
    }

    /// Calculate combined mask.
    #[must_use]
    pub fn calculate_mask(&self, color: RgbColor) -> f64 {
        if self.qualifiers.is_empty() {
            return 1.0;
        }

        let masks: Vec<f64> = self
            .qualifiers
            .iter()
            .map(|q| q.calculate_mask(color))
            .collect();

        match self.mode {
            QualifierCombineMode::Union => masks.iter().fold(0.0_f64, |acc, &m| acc.max(m)),
            QualifierCombineMode::Intersection => masks.iter().fold(1.0_f64, |acc, &m| acc.min(m)),
            QualifierCombineMode::Difference => {
                if masks.is_empty() {
                    return 0.0;
                }
                let first = masks[0];
                let others_max = masks.iter().skip(1).fold(0.0_f64, |acc, &m| acc.max(m));
                (first - others_max).max(0.0)
            }
        }
    }
}

impl Default for MultiQualifier {
    fn default() -> Self {
        Self::new()
    }
}
