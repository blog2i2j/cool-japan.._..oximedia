//! Core color types for the grading module.

use super::utility::{hsl_to_rgb, rgb_to_hsl};

/// RGB color representation for color grading operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
}

impl RgbColor {
    /// Create a new RGB color.
    #[must_use]
    pub const fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Create a grayscale color.
    #[must_use]
    pub const fn gray(v: f64) -> Self {
        Self { r: v, g: v, b: v }
    }

    /// Clamp color values to [0, 1] range.
    #[must_use]
    pub fn clamp(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
        }
    }

    /// Convert to HSL color space.
    #[must_use]
    pub fn to_hsl(self) -> HslColor {
        rgb_to_hsl(self.r, self.g, self.b)
    }

    /// Linear interpolation between two colors.
    #[must_use]
    pub fn lerp(self, other: &Self, t: f64) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }
}

/// HSL (Hue, Saturation, Luminance) color representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HslColor {
    /// Hue (0.0 - 360.0)
    pub h: f64,
    /// Saturation (0.0 - 1.0)
    pub s: f64,
    /// Luminance (0.0 - 1.0)
    pub l: f64,
}

impl HslColor {
    /// Create a new HSL color.
    #[must_use]
    pub const fn new(h: f64, s: f64, l: f64) -> Self {
        Self { h, s, l }
    }

    /// Convert to RGB color space.
    #[must_use]
    pub fn to_rgb(self) -> RgbColor {
        hsl_to_rgb(self.h, self.s, self.l)
    }
}

/// Color channel for histogram and scope operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChannel {
    Red,
    Green,
    Blue,
    Luma,
}
