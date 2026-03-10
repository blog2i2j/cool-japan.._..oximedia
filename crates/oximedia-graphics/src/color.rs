//! Color management and broadcast-safe color utilities

use crate::error::{GraphicsError, Result};
use serde::{Deserialize, Serialize};

/// RGBA color with 8-bit components
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
    /// Alpha component (0-255)
    pub a: u8,
}

impl Color {
    /// Create a new color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::new(r, g, b, 255)
    }

    /// Create from hex string (e.g., "#FF0000" or "FF0000")
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 && hex.len() != 8 {
            return Err(GraphicsError::InvalidColor(format!(
                "Invalid hex color: {hex}"
            )));
        }

        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| GraphicsError::InvalidColor(hex.to_string()))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| GraphicsError::InvalidColor(hex.to_string()))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| GraphicsError::InvalidColor(hex.to_string()))?;
        let a = if hex.len() == 8 {
            u8::from_str_radix(&hex[6..8], 16)
                .map_err(|_| GraphicsError::InvalidColor(hex.to_string()))?
        } else {
            255
        };

        Ok(Self::new(r, g, b, a))
    }

    /// Convert to hex string
    #[must_use]
    pub fn to_hex(&self) -> String {
        if self.a == 255 {
            format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
        }
    }

    /// Convert to normalized float values (0.0-1.0)
    #[must_use]
    pub fn to_float(&self) -> [f32; 4] {
        [
            f32::from(self.r) / 255.0,
            f32::from(self.g) / 255.0,
            f32::from(self.b) / 255.0,
            f32::from(self.a) / 255.0,
        ]
    }

    /// Create from normalized float values
    #[must_use]
    pub fn from_float(values: [f32; 4]) -> Self {
        Self::new(
            (values[0].clamp(0.0, 1.0) * 255.0) as u8,
            (values[1].clamp(0.0, 1.0) * 255.0) as u8,
            (values[2].clamp(0.0, 1.0) * 255.0) as u8,
            (values[3].clamp(0.0, 1.0) * 255.0) as u8,
        )
    }

    /// Apply broadcast-safe color limiting (legal range)
    #[must_use]
    pub fn broadcast_safe(&self) -> Self {
        const MIN_LUMA: u8 = 16;
        const MAX_LUMA: u8 = 235;
        const MIN_CHROMA: u8 = 16;
        const MAX_CHROMA: u8 = 240;

        // Convert to YCbCr
        let (y, cb, cr) = self.to_ycbcr();

        // Clamp to legal range
        let y = y.clamp(MIN_LUMA, MAX_LUMA);
        let cb = cb.clamp(MIN_CHROMA, MAX_CHROMA);
        let cr = cr.clamp(MIN_CHROMA, MAX_CHROMA);

        // Convert back to RGB
        Self::from_ycbcr(y, cb, cr, self.a)
    }

    /// Convert RGB to YCbCr (BT.709)
    #[must_use]
    pub fn to_ycbcr(&self) -> (u8, u8, u8) {
        let r = f32::from(self.r);
        let g = f32::from(self.g);
        let b = f32::from(self.b);

        let y = 16.0 + (0.2126 * r + 0.7152 * g + 0.0722 * b);
        let cb = 128.0 + (-0.1146 * r - 0.3854 * g + 0.5000 * b);
        let cr = 128.0 + (0.5000 * r - 0.4542 * g - 0.0458 * b);

        (
            y.clamp(0.0, 255.0) as u8,
            cb.clamp(0.0, 255.0) as u8,
            cr.clamp(0.0, 255.0) as u8,
        )
    }

    /// Convert YCbCr to RGB (BT.709)
    #[must_use]
    pub fn from_ycbcr(y: u8, cb: u8, cr: u8, a: u8) -> Self {
        let y = f32::from(y) - 16.0;
        let cb = f32::from(cb) - 128.0;
        let cr = f32::from(cr) - 128.0;

        let r = y + 1.5748 * cr;
        let g = y - 0.1873 * cb - 0.4681 * cr;
        let b = y + 1.8556 * cb;

        Self::new(
            r.clamp(0.0, 255.0) as u8,
            g.clamp(0.0, 255.0) as u8,
            b.clamp(0.0, 255.0) as u8,
            a,
        )
    }

    /// Premultiply alpha
    #[must_use]
    pub fn premultiply(&self) -> Self {
        let alpha = f32::from(self.a) / 255.0;
        Self::new(
            (f32::from(self.r) * alpha) as u8,
            (f32::from(self.g) * alpha) as u8,
            (f32::from(self.b) * alpha) as u8,
            self.a,
        )
    }

    /// Blend with another color (over operation)
    #[must_use]
    pub fn blend(&self, bg: &Color) -> Self {
        let alpha = f32::from(self.a) / 255.0;
        let inv_alpha = 1.0 - alpha;

        Self::new(
            (f32::from(self.r) * alpha + f32::from(bg.r) * inv_alpha) as u8,
            (f32::from(self.g) * alpha + f32::from(bg.g) * inv_alpha) as u8,
            (f32::from(self.b) * alpha + f32::from(bg.b) * inv_alpha) as u8,
            ((f32::from(self.a) * alpha + f32::from(bg.a) * inv_alpha) * 255.0) as u8,
        )
    }

    /// Common color constants
    pub const TRANSPARENT: Self = Self::new(0, 0, 0, 0);
    /// Black color
    pub const BLACK: Self = Self::rgb(0, 0, 0);
    /// White color
    pub const WHITE: Self = Self::rgb(255, 255, 255);
    /// Red color
    pub const RED: Self = Self::rgb(255, 0, 0);
    /// Green color
    pub const GREEN: Self = Self::rgb(0, 255, 0);
    /// Blue color
    pub const BLUE: Self = Self::rgb(0, 0, 255);
    /// Yellow color
    pub const YELLOW: Self = Self::rgb(255, 255, 0);
    /// Cyan color
    pub const CYAN: Self = Self::rgb(0, 255, 255);
    /// Magenta color
    pub const MAGENTA: Self = Self::rgb(255, 0, 255);
}

impl Default for Color {
    fn default() -> Self {
        Self::BLACK
    }
}

/// Gradient fill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Gradient {
    /// Linear gradient
    Linear {
        /// Start point (x, y)
        start: (f32, f32),
        /// End point (x, y)
        end: (f32, f32),
        /// Color stops (position 0.0-1.0, color)
        stops: Vec<(f32, Color)>,
    },
    /// Radial gradient
    Radial {
        /// Center point (x, y)
        center: (f32, f32),
        /// Radius
        radius: f32,
        /// Color stops (position 0.0-1.0, color)
        stops: Vec<(f32, Color)>,
    },
}

impl Gradient {
    /// Create a linear gradient
    #[must_use]
    pub fn linear(start: (f32, f32), end: (f32, f32), stops: Vec<(f32, Color)>) -> Self {
        Self::Linear { start, end, stops }
    }

    /// Create a radial gradient
    #[must_use]
    pub fn radial(center: (f32, f32), radius: f32, stops: Vec<(f32, Color)>) -> Self {
        Self::Radial {
            center,
            radius,
            stops,
        }
    }

    /// Sample color at position
    #[must_use]
    pub fn sample(&self, x: f32, y: f32) -> Color {
        match self {
            Self::Linear { start, end, stops } => {
                let dx = end.0 - start.0;
                let dy = end.1 - start.1;
                let len_sq = dx * dx + dy * dy;

                if len_sq == 0.0 {
                    return stops.first().map_or(Color::BLACK, |(_, c)| *c);
                }

                let t = ((x - start.0) * dx + (y - start.1) * dy) / len_sq;
                self.sample_stops(t.clamp(0.0, 1.0), stops)
            }
            Self::Radial {
                center,
                radius,
                stops,
            } => {
                let dx = x - center.0;
                let dy = y - center.1;
                let dist = (dx * dx + dy * dy).sqrt();
                let t = (dist / radius).clamp(0.0, 1.0);
                self.sample_stops(t, stops)
            }
        }
    }

    fn sample_stops(&self, t: f32, stops: &[(f32, Color)]) -> Color {
        if stops.is_empty() {
            return Color::BLACK;
        }

        if stops.len() == 1 {
            return stops[0].1;
        }

        // Find the two stops to interpolate between
        let mut prev = &stops[0];
        for stop in stops.iter().skip(1) {
            if t <= stop.0 {
                let t_local = (t - prev.0) / (stop.0 - prev.0);
                return self.lerp_color(&prev.1, &stop.1, t_local);
            }
            prev = stop;
        }

        stops.last().map_or(Color::BLACK, |(_, c)| *c)
    }

    fn lerp_color(&self, a: &Color, b: &Color, t: f32) -> Color {
        Color::new(
            (f32::from(a.r) + (f32::from(b.r) - f32::from(a.r)) * t) as u8,
            (f32::from(a.g) + (f32::from(b.g) - f32::from(a.g)) * t) as u8,
            (f32::from(a.b) + (f32::from(b.b) - f32::from(a.b)) * t) as u8,
            (f32::from(a.a) + (f32::from(b.a) - f32::from(a.a)) * t) as u8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_creation() {
        let c = Color::new(255, 128, 64, 32);
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 128);
        assert_eq!(c.b, 64);
        assert_eq!(c.a, 32);
    }

    #[test]
    fn test_color_rgb() {
        let c = Color::rgb(255, 128, 64);
        assert_eq!(c.a, 255);
    }

    #[test]
    fn test_color_from_hex() {
        let c = Color::from_hex("#FF8040").expect("c should be valid");
        assert_eq!(c.r, 255);
        assert_eq!(c.g, 128);
        assert_eq!(c.b, 64);
        assert_eq!(c.a, 255);

        let c = Color::from_hex("FF804080").expect("c should be valid");
        assert_eq!(c.a, 128);
    }

    #[test]
    fn test_color_to_hex() {
        let c = Color::rgb(255, 128, 64);
        assert_eq!(c.to_hex(), "#FF8040");

        let c = Color::new(255, 128, 64, 128);
        assert_eq!(c.to_hex(), "#FF804080");
    }

    #[test]
    fn test_color_float_conversion() {
        let c = Color::rgb(255, 128, 0);
        let f = c.to_float();
        assert!((f[0] - 1.0).abs() < 0.01);
        assert!((f[1] - 0.502).abs() < 0.01);
        assert!((f[2] - 0.0).abs() < 0.01);

        let c2 = Color::from_float(f);
        assert_eq!(c, c2);
    }

    #[test]
    fn test_broadcast_safe() {
        let c = Color::rgb(255, 255, 255);
        let safe = c.broadcast_safe();
        // Should clamp to legal range
        assert!(safe.r <= 235);
    }

    #[test]
    fn test_ycbcr_conversion() {
        let c = Color::rgb(128, 128, 128);
        let (y, cb, cr) = c.to_ycbcr();
        let c2 = Color::from_ycbcr(y, cb, cr, 255);
        // Allow some rounding error
        assert!((c.r as i16 - c2.r as i16).abs() <= 2);
        assert!((c.g as i16 - c2.g as i16).abs() <= 2);
        assert!((c.b as i16 - c2.b as i16).abs() <= 2);
    }

    #[test]
    fn test_color_blend() {
        let fg = Color::new(255, 0, 0, 128);
        let bg = Color::rgb(0, 0, 255);
        let blended = fg.blend(&bg);
        // Should be somewhere between red and blue
        assert!(blended.r > 0);
        assert!(blended.b > 0);
    }

    #[test]
    fn test_gradient_linear() {
        let gradient = Gradient::linear(
            (0.0, 0.0),
            (100.0, 0.0),
            vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
        );

        let c1 = gradient.sample(0.0, 0.0);
        let c2 = gradient.sample(50.0, 0.0);
        let c3 = gradient.sample(100.0, 0.0);

        assert_eq!(c1, Color::BLACK);
        assert_eq!(c3, Color::WHITE);
        assert!(c2.r > 0 && c2.r < 255);
    }

    #[test]
    fn test_gradient_radial() {
        let gradient = Gradient::radial(
            (50.0, 50.0),
            50.0,
            vec![(0.0, Color::WHITE), (1.0, Color::BLACK)],
        );

        let c1 = gradient.sample(50.0, 50.0);
        let c2 = gradient.sample(100.0, 50.0);

        assert_eq!(c1, Color::WHITE);
        assert_eq!(c2, Color::BLACK);
    }

    #[test]
    fn test_color_constants() {
        assert_eq!(Color::BLACK, Color::rgb(0, 0, 0));
        assert_eq!(Color::WHITE, Color::rgb(255, 255, 255));
        assert_eq!(Color::RED, Color::rgb(255, 0, 0));
        assert_eq!(Color::GREEN, Color::rgb(0, 255, 0));
        assert_eq!(Color::BLUE, Color::rgb(0, 0, 255));
    }
}
