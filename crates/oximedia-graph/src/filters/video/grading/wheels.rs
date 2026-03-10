//! Color wheels: shadow/midtone/highlight wheels, log/offset/power, temperature/tint.

use super::types::RgbColor;
use super::utility::apply_saturation;

/// Color wheel parameters for a specific tonal region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorWheel {
    /// Hue shift in degrees (-180 to 180).
    pub hue: f64,
    /// Saturation adjustment (0.0 = grayscale, 1.0 = no change, 2.0 = double).
    pub saturation: f64,
    /// Luminance adjustment (-1.0 to 1.0).
    pub luminance: f64,
}

impl Default for ColorWheel {
    fn default() -> Self {
        Self {
            hue: 0.0,
            saturation: 1.0,
            luminance: 0.0,
        }
    }
}

impl ColorWheel {
    /// Create a new color wheel.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set hue shift.
    #[must_use]
    pub const fn with_hue(mut self, hue: f64) -> Self {
        self.hue = hue;
        self
    }

    /// Set saturation.
    #[must_use]
    pub const fn with_saturation(mut self, saturation: f64) -> Self {
        self.saturation = saturation;
        self
    }

    /// Set luminance.
    #[must_use]
    pub const fn with_luminance(mut self, luminance: f64) -> Self {
        self.luminance = luminance;
        self
    }

    /// Apply color wheel to a color with given weight.
    #[must_use]
    pub fn apply(&self, color: RgbColor, weight: f64) -> RgbColor {
        if weight <= 0.0 {
            return color;
        }

        let mut hsl = color.to_hsl();

        // Apply hue shift
        hsl.h = (hsl.h + self.hue * weight).rem_euclid(360.0);

        // Apply saturation
        hsl.s = (hsl.s * (1.0 + (self.saturation - 1.0) * weight)).clamp(0.0, 1.0);

        // Apply luminance
        hsl.l = (hsl.l + self.luminance * weight).clamp(0.0, 1.0);

        hsl.to_rgb()
    }
}

/// Shadow/Midtone/Highlight color wheels.
#[derive(Clone, Debug)]
pub struct ColorWheels {
    /// Shadow color wheel adjustments
    pub shadows: ColorWheel,
    /// Midtone color wheel adjustments
    pub midtones: ColorWheel,
    /// Highlight color wheel adjustments
    pub highlights: ColorWheel,
    /// Shadow/highlight split point (0.0 - 1.0).
    pub shadow_max: f64,
    /// Midtone/highlight split point (0.0 - 1.0).
    pub highlight_min: f64,
    /// Enable offset mode (affects entire tonal range equally).
    pub offset_mode: bool,
    /// Global saturation multiplier.
    pub global_saturation: f64,
}

impl Default for ColorWheels {
    fn default() -> Self {
        Self {
            shadows: ColorWheel::default(),
            midtones: ColorWheel::default(),
            highlights: ColorWheel::default(),
            shadow_max: 0.33,
            highlight_min: 0.67,
            offset_mode: false,
            global_saturation: 1.0,
        }
    }
}

impl ColorWheels {
    /// Create new color wheels.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply color wheels to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let mut result = color;

        if self.offset_mode {
            // Offset mode: apply all wheels equally (no luminance-based weighting)
            result = self.shadows.apply(result, 1.0);
            result = self.midtones.apply(result, 1.0);
            result = self.highlights.apply(result, 1.0);
        } else {
            // Normal mode: weight by luminance
            let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

            // Calculate weights for each region
            let shadow_weight = self.calculate_shadow_weight(luma);
            let midtone_weight = self.calculate_midtone_weight(luma);
            let highlight_weight = self.calculate_highlight_weight(luma);

            // Apply each wheel with its weight
            result = self.shadows.apply(result, shadow_weight);
            result = self.midtones.apply(result, midtone_weight);
            result = self.highlights.apply(result, highlight_weight);
        }

        // Apply global saturation
        if (self.global_saturation - 1.0).abs() > f64::EPSILON {
            result = apply_saturation(result, self.global_saturation);
        }

        result
    }

    /// Set offset mode.
    #[must_use]
    pub const fn with_offset_mode(mut self, enabled: bool) -> Self {
        self.offset_mode = enabled;
        self
    }

    /// Set global saturation.
    #[must_use]
    pub const fn with_global_saturation(mut self, saturation: f64) -> Self {
        self.global_saturation = saturation;
        self
    }

    /// Set shadow/midtone/highlight split points.
    #[must_use]
    pub const fn with_split_points(mut self, shadow_max: f64, highlight_min: f64) -> Self {
        self.shadow_max = shadow_max;
        self.highlight_min = highlight_min;
        self
    }

    /// Calculate shadow weight based on luminance.
    fn calculate_shadow_weight(&self, luma: f64) -> f64 {
        if luma <= self.shadow_max {
            1.0
        } else if luma >= self.highlight_min {
            0.0
        } else {
            // Smooth transition
            let t = (luma - self.shadow_max) / (self.highlight_min - self.shadow_max);
            (1.0 - t).max(0.0)
        }
    }

    /// Calculate midtone weight based on luminance.
    fn calculate_midtone_weight(&self, luma: f64) -> f64 {
        if luma <= self.shadow_max || luma >= self.highlight_min {
            0.0
        } else {
            // Peak at midpoint
            let mid = (self.shadow_max + self.highlight_min) / 2.0;
            let range = self.highlight_min - self.shadow_max;
            1.0 - ((luma - mid).abs() / (range / 2.0))
        }
    }

    /// Calculate highlight weight based on luminance.
    fn calculate_highlight_weight(&self, luma: f64) -> f64 {
        if luma >= self.highlight_min {
            1.0
        } else if luma <= self.shadow_max {
            0.0
        } else {
            // Smooth transition
            let t = (luma - self.shadow_max) / (self.highlight_min - self.shadow_max);
            t.max(0.0)
        }
    }
}

/// Log/Offset/Power color correction.
#[derive(Clone, Debug, PartialEq)]
pub struct LogOffsetPower {
    /// Logarithmic adjustment
    pub log: RgbColor,
    /// Linear offset adjustment
    pub offset: RgbColor,
    /// Power/gamma adjustment
    pub power: RgbColor,
}

impl Default for LogOffsetPower {
    fn default() -> Self {
        Self {
            log: RgbColor::gray(0.0),
            offset: RgbColor::gray(0.0),
            power: RgbColor::gray(1.0),
        }
    }
}

impl LogOffsetPower {
    /// Create new log/offset/power.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply log/offset/power to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let r = self.apply_channel(color.r, self.log.r, self.offset.r, self.power.r);
        let g = self.apply_channel(color.g, self.log.g, self.offset.g, self.power.g);
        let b = self.apply_channel(color.b, self.log.b, self.offset.b, self.power.b);

        RgbColor::new(r, g, b)
    }

    /// Apply to a single channel.
    fn apply_channel(&self, value: f64, log: f64, offset: f64, power: f64) -> f64 {
        // Log adjustment (affects overall brightness)
        let logged = if value > 0.0 {
            value * 2_f64.powf(log)
        } else {
            value
        };

        // Offset (adds to signal)
        let offsetted = logged + offset;

        // Power (gamma-like adjustment)
        if offsetted > 0.0 && power > 0.0 {
            offsetted.powf(power)
        } else {
            offsetted
        }
    }
}

/// Temperature and tint adjustment.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemperatureTint {
    /// Temperature adjustment in Kelvin offset (-10000 to 10000).
    pub temperature: f64,
    /// Tint adjustment (magenta/green) (-1.0 to 1.0).
    pub tint: f64,
}

impl Default for TemperatureTint {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
        }
    }
}

impl TemperatureTint {
    /// Create new temperature/tint adjustment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply temperature and tint to a color.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let mut result = color;

        // Apply temperature (blue/orange shift)
        if self.temperature.abs() > f64::EPSILON {
            let temp_factor = self.temperature / 10000.0;
            result.r += temp_factor;
            result.b -= temp_factor;
        }

        // Apply tint (magenta/green shift)
        if self.tint.abs() > f64::EPSILON {
            result.g += self.tint * 0.5;
            result.r -= self.tint * 0.25;
            result.b -= self.tint * 0.25;
        }

        result
    }
}
