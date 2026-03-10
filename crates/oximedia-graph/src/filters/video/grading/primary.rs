//! Primary color correction: Lift/Gamma/Gain and ASC CDL.

use super::types::RgbColor;
use super::utility::apply_saturation;

/// Lift/Gamma/Gain color correction parameters.
///
/// These controls provide primary color correction:
/// - Lift: Adjusts the black level (shadows)
/// - Gamma: Adjusts the midtones
/// - Gain: Adjusts the white level (highlights)
#[derive(Clone, Debug, PartialEq)]
pub struct LiftGammaGain {
    /// Lift (shadows) adjustment per channel.
    pub lift: RgbColor,
    /// Gamma (midtones) adjustment per channel.
    pub gamma: RgbColor,
    /// Gain (highlights) adjustment per channel.
    pub gain: RgbColor,
    /// Master lift (affects all channels).
    pub master_lift: f64,
    /// Master gamma (affects all channels).
    pub master_gamma: f64,
    /// Master gain (affects all channels).
    pub master_gain: f64,
    /// Contrast (affects the curve slope).
    pub contrast: f64,
    /// Pivot point for contrast adjustment (0.0-1.0).
    pub pivot: f64,
    /// Enable printer lights mode (legacy film grading).
    pub printer_lights: bool,
}

impl Default for LiftGammaGain {
    fn default() -> Self {
        Self {
            lift: RgbColor::gray(0.0),
            gamma: RgbColor::gray(1.0),
            gain: RgbColor::gray(1.0),
            master_lift: 0.0,
            master_gamma: 1.0,
            master_gain: 1.0,
            contrast: 1.0,
            pivot: 0.5,
            printer_lights: false,
        }
    }
}

impl LiftGammaGain {
    /// Create new lift/gamma/gain with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set lift values (shadows).
    #[must_use]
    pub fn with_lift(mut self, r: f64, g: f64, b: f64) -> Self {
        self.lift = RgbColor::new(r, g, b);
        self
    }

    /// Set gamma values (midtones).
    #[must_use]
    pub fn with_gamma(mut self, r: f64, g: f64, b: f64) -> Self {
        self.gamma = RgbColor::new(r, g, b);
        self
    }

    /// Set gain values (highlights).
    #[must_use]
    pub fn with_gain(mut self, r: f64, g: f64, b: f64) -> Self {
        self.gain = RgbColor::new(r, g, b);
        self
    }

    /// Set master lift.
    #[must_use]
    pub const fn with_master_lift(mut self, lift: f64) -> Self {
        self.master_lift = lift;
        self
    }

    /// Set master gamma.
    #[must_use]
    pub const fn with_master_gamma(mut self, gamma: f64) -> Self {
        self.master_gamma = gamma;
        self
    }

    /// Set master gain.
    #[must_use]
    pub const fn with_master_gain(mut self, gain: f64) -> Self {
        self.master_gain = gain;
        self
    }

    /// Set contrast.
    #[must_use]
    pub const fn with_contrast(mut self, contrast: f64) -> Self {
        self.contrast = contrast;
        self
    }

    /// Set pivot point for contrast.
    #[must_use]
    pub const fn with_pivot(mut self, pivot: f64) -> Self {
        self.pivot = pivot;
        self
    }

    /// Enable printer lights mode.
    #[must_use]
    pub const fn with_printer_lights(mut self, enabled: bool) -> Self {
        self.printer_lights = enabled;
        self
    }

    /// Apply lift/gamma/gain to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        if self.printer_lights {
            return self.apply_printer_lights(color);
        }

        let r = self.apply_channel(color.r, self.lift.r, self.gamma.r, self.gain.r);
        let g = self.apply_channel(color.g, self.lift.g, self.gamma.g, self.gain.g);
        let b = self.apply_channel(color.b, self.lift.b, self.gamma.b, self.gain.b);

        let mut result = RgbColor::new(r, g, b);

        // Apply master controls
        result = self.apply_master(result);

        // Apply contrast around pivot
        result = self.apply_contrast(result);

        result
    }

    /// Apply printer lights mode (legacy film grading).
    /// In this mode, values are interpreted as light points (1 point = 0.025 log exposure).
    fn apply_printer_lights(&self, color: RgbColor) -> RgbColor {
        let points_to_linear = |points: f64| -> f64 {
            if points.abs() < f64::EPSILON {
                1.0
            } else {
                2_f64.powf(points * 0.025 / 0.3) // ~1 point = 1/12 stop
            }
        };

        let r = color.r * points_to_linear(-self.lift.r) * points_to_linear(-self.gain.r);
        let g = color.g * points_to_linear(-self.lift.g) * points_to_linear(-self.gain.g);
        let b = color.b * points_to_linear(-self.lift.b) * points_to_linear(-self.gain.b);

        RgbColor::new(r, g, b)
    }

    /// Apply contrast around pivot point.
    fn apply_contrast(&self, color: RgbColor) -> RgbColor {
        if (self.contrast - 1.0).abs() < f64::EPSILON {
            return color;
        }

        // Apply contrast around pivot
        let apply_to_channel = |value: f64| -> f64 {
            let centered = value - self.pivot;
            let contrasted = centered * self.contrast;
            (contrasted + self.pivot).clamp(0.0, 1.0)
        };

        RgbColor::new(
            apply_to_channel(color.r),
            apply_to_channel(color.g),
            apply_to_channel(color.b),
        )
    }

    /// Apply lift/gamma/gain to a single channel.
    fn apply_channel(&self, value: f64, lift: f64, gamma: f64, gain: f64) -> f64 {
        // Lift: adds to the signal (affects shadows most)
        let lifted = value + lift;

        // Gamma: power function (affects midtones)
        let gamma_corrected = if lifted > 0.0 && gamma > 0.0 {
            lifted.powf(1.0 / gamma)
        } else {
            lifted
        };

        // Gain: multiplies the signal (affects highlights most)
        gamma_corrected * gain
    }

    /// Apply master controls.
    fn apply_master(&self, color: RgbColor) -> RgbColor {
        let r = color.r + self.master_lift;
        let g = color.g + self.master_lift;
        let b = color.b + self.master_lift;

        let r = if r > 0.0 && self.master_gamma > 0.0 {
            r.powf(1.0 / self.master_gamma)
        } else {
            r
        };
        let g = if g > 0.0 && self.master_gamma > 0.0 {
            g.powf(1.0 / self.master_gamma)
        } else {
            g
        };
        let b = if b > 0.0 && self.master_gamma > 0.0 {
            b.powf(1.0 / self.master_gamma)
        } else {
            b
        };

        RgbColor::new(
            r * self.master_gain,
            g * self.master_gain,
            b * self.master_gain,
        )
    }
}

/// ASC CDL (Color Decision List) parameters.
///
/// Standard color correction format used in professional workflows.
/// Formula: out = (in * slope + offset)^power
#[derive(Clone, Debug, PartialEq)]
pub struct AscCdl {
    /// Slope (similar to gain).
    pub slope: RgbColor,
    /// Offset (similar to lift).
    pub offset: RgbColor,
    /// Power (similar to gamma).
    pub power: RgbColor,
    /// Saturation adjustment.
    pub saturation: f64,
}

impl Default for AscCdl {
    fn default() -> Self {
        Self {
            slope: RgbColor::gray(1.0),
            offset: RgbColor::gray(0.0),
            power: RgbColor::gray(1.0),
            saturation: 1.0,
        }
    }
}

impl AscCdl {
    /// Create new ASC CDL with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply ASC CDL to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        // Apply slope and offset
        let r = color.r * self.slope.r + self.offset.r;
        let g = color.g * self.slope.g + self.offset.g;
        let b = color.b * self.slope.b + self.offset.b;

        // Clamp to positive values before power
        let r = r.max(0.0);
        let g = g.max(0.0);
        let b = b.max(0.0);

        // Apply power
        let r = r.powf(self.power.r);
        let g = g.powf(self.power.g);
        let b = b.powf(self.power.b);

        let mut result = RgbColor::new(r, g, b);

        // Apply saturation
        if (self.saturation - 1.0).abs() > f64::EPSILON {
            result = apply_saturation(result, self.saturation);
        }

        result
    }
}
