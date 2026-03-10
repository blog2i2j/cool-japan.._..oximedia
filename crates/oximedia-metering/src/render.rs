//! Meter rendering and visualization.
//!
//! Provides data structures and utilities for rendering meters visually.

use crate::{MeteringError, MeteringResult};

/// Color in RGB format.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red component (0-255).
    pub r: u8,
    /// Green component (0-255).
    pub g: u8,
    /// Blue component (0-255).
    pub b: u8,
}

impl Color {
    /// Create a new color.
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create a color from hex string (e.g., "#FF0000" for red).
    pub fn from_hex(hex: &str) -> MeteringResult<Self> {
        let hex = hex.trim_start_matches('#');

        if hex.len() != 6 {
            return Err(MeteringError::InvalidConfig(
                "Hex color must be 6 characters".to_string(),
            ));
        }

        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| MeteringError::InvalidConfig("Invalid hex color".to_string()))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| MeteringError::InvalidConfig("Invalid hex color".to_string()))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| MeteringError::InvalidConfig("Invalid hex color".to_string()))?;

        Ok(Self { r, g, b })
    }

    /// Interpolate between two colors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other color
    /// * `t` - Interpolation factor (0.0 to 1.0)
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: (f64::from(self.r) + (f64::from(other.r) - f64::from(self.r)) * t) as u8,
            g: (f64::from(self.g) + (f64::from(other.g) - f64::from(self.g)) * t) as u8,
            b: (f64::from(self.b) + (f64::from(other.b) - f64::from(self.b)) * t) as u8,
        }
    }
}

/// Common meter colors.
pub mod colors {
    use super::Color;

    /// Green (safe zone).
    pub const GREEN: Color = Color::new(0, 255, 0);
    /// Yellow (warning zone).
    pub const YELLOW: Color = Color::new(255, 255, 0);
    /// Red (danger zone).
    pub const RED: Color = Color::new(255, 0, 0);
    /// Dark green (lower range).
    pub const DARK_GREEN: Color = Color::new(0, 128, 0);
    /// Orange (intermediate warning).
    pub const ORANGE: Color = Color::new(255, 165, 0);
    /// Black (background).
    pub const BLACK: Color = Color::new(0, 0, 0);
    /// White (foreground/text).
    pub const WHITE: Color = Color::new(255, 255, 255);
    /// Dark gray (scale markings).
    pub const DARK_GRAY: Color = Color::new(64, 64, 64);
    /// Light gray (grid).
    pub const LIGHT_GRAY: Color = Color::new(192, 192, 192);
}

/// Meter orientation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Orientation {
    /// Horizontal meter (left to right).
    Horizontal,
    /// Vertical meter (bottom to top).
    Vertical,
}

/// Meter scale type.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScaleType {
    /// Linear scale.
    Linear,
    /// Logarithmic scale (dB).
    Logarithmic,
}

/// Color gradient for meter display.
#[derive(Clone, Debug)]
pub struct ColorGradient {
    stops: Vec<(f64, Color)>,
}

impl ColorGradient {
    /// Create a new color gradient.
    ///
    /// # Arguments
    ///
    /// * `stops` - List of (position, color) tuples where position is 0.0 to 1.0
    pub fn new(stops: Vec<(f64, Color)>) -> Self {
        Self { stops }
    }

    /// Create a standard traffic light gradient (green -> yellow -> red).
    pub fn traffic_light() -> Self {
        Self::new(vec![
            (0.0, colors::DARK_GREEN),
            (0.6, colors::GREEN),
            (0.8, colors::YELLOW),
            (0.95, colors::ORANGE),
            (1.0, colors::RED),
        ])
    }

    /// Create a standard PPM gradient.
    pub fn ppm() -> Self {
        Self::new(vec![
            (0.0, colors::DARK_GREEN),
            (0.7, colors::GREEN),
            (0.9, colors::YELLOW),
            (1.0, colors::RED),
        ])
    }

    /// Get the color at a specific position.
    ///
    /// # Arguments
    ///
    /// * `position` - Position in gradient (0.0 to 1.0)
    pub fn color_at(&self, position: f64) -> Color {
        let position = position.clamp(0.0, 1.0);

        // Find the two stops to interpolate between
        for i in 0..self.stops.len() - 1 {
            let (pos1, color1) = self.stops[i];
            let (pos2, color2) = self.stops[i + 1];

            if position >= pos1 && position <= pos2 {
                let range = pos2 - pos1;
                let t = if range > 0.0 {
                    (position - pos1) / range
                } else {
                    0.0
                };
                return color1.lerp(&color2, t);
            }
        }

        // Return last color if position is beyond all stops
        self.stops.last().map_or(colors::BLACK, |(_, c)| *c)
    }
}

/// Bar meter renderer configuration.
#[derive(Clone, Debug)]
pub struct BarMeterConfig {
    /// Meter orientation.
    pub orientation: Orientation,
    /// Meter width in pixels.
    pub width: usize,
    /// Meter height in pixels.
    pub height: usize,
    /// Minimum value (e.g., -60.0 dBFS).
    pub min_value: f64,
    /// Maximum value (e.g., 0.0 dBFS).
    pub max_value: f64,
    /// Scale type.
    pub scale_type: ScaleType,
    /// Color gradient.
    pub gradient: ColorGradient,
    /// Show peak hold indicator.
    pub show_peak_hold: bool,
    /// Show scale markings.
    pub show_scale: bool,
}

impl Default for BarMeterConfig {
    fn default() -> Self {
        Self {
            orientation: Orientation::Vertical,
            width: 30,
            height: 200,
            min_value: -60.0,
            max_value: 0.0,
            scale_type: ScaleType::Logarithmic,
            gradient: ColorGradient::traffic_light(),
            show_peak_hold: true,
            show_scale: true,
        }
    }
}

/// Bar meter render data.
#[derive(Clone, Debug)]
pub struct BarMeterData {
    /// Current level (0.0 to 1.0 normalized).
    pub level: f64,
    /// Peak hold level (0.0 to 1.0 normalized).
    pub peak_hold: f64,
    /// Whether the meter is clipping.
    pub is_clipping: bool,
}

impl BarMeterData {
    /// Create bar meter data from dBFS values.
    ///
    /// # Arguments
    ///
    /// * `level_dbfs` - Current level in dBFS
    /// * `peak_hold_dbfs` - Peak hold level in dBFS
    /// * `min_dbfs` - Minimum dBFS for normalization
    /// * `max_dbfs` - Maximum dBFS for normalization
    pub fn from_dbfs(level_dbfs: f64, peak_hold_dbfs: f64, min_dbfs: f64, max_dbfs: f64) -> Self {
        let normalize = |db: f64| {
            if db.is_infinite() && db.is_sign_negative() {
                0.0
            } else {
                ((db - min_dbfs) / (max_dbfs - min_dbfs)).clamp(0.0, 1.0)
            }
        };

        Self {
            level: normalize(level_dbfs),
            peak_hold: normalize(peak_hold_dbfs),
            is_clipping: level_dbfs >= max_dbfs,
        }
    }
}

/// Scale marking on a meter.
#[derive(Clone, Debug)]
pub struct ScaleMark {
    /// Position (0.0 to 1.0).
    pub position: f64,
    /// Label text.
    pub label: String,
    /// Whether this is a major marking.
    pub is_major: bool,
}

/// Generate scale markings for a dBFS meter.
pub fn generate_db_scale(min_db: f64, max_db: f64) -> Vec<ScaleMark> {
    let mut marks = Vec::new();
    let range = max_db - min_db;

    // Major markings every 10 dB
    let mut db = (min_db / 10.0).ceil() * 10.0;
    while db <= max_db {
        let position = (db - min_db) / range;
        marks.push(ScaleMark {
            position,
            label: format!("{db:.0}"),
            is_major: true,
        });
        db += 10.0;
    }

    // Minor markings every 5 dB
    let mut db = (min_db / 5.0).ceil() * 5.0;
    while db <= max_db {
        let position = (db - min_db) / range;
        // Skip if this is already a major marking
        if !marks.iter().any(|m| (m.position - position).abs() < 0.01) {
            marks.push(ScaleMark {
                position,
                label: String::new(),
                is_major: false,
            });
        }
        db += 5.0;
    }

    marks
}

/// Circular meter configuration for radial displays.
#[derive(Clone, Debug)]
pub struct CircularMeterConfig {
    /// Center X coordinate.
    pub center_x: usize,
    /// Center Y coordinate.
    pub center_y: usize,
    /// Radius in pixels.
    pub radius: usize,
    /// Start angle in degrees (0 = right, 90 = top).
    pub start_angle: f64,
    /// End angle in degrees.
    pub end_angle: f64,
    /// Color gradient.
    pub gradient: ColorGradient,
}

impl Default for CircularMeterConfig {
    fn default() -> Self {
        Self {
            center_x: 100,
            center_y: 100,
            radius: 80,
            start_angle: 135.0, // Lower left
            end_angle: 45.0,    // Lower right
            gradient: ColorGradient::traffic_light(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_creation() {
        let color = Color::new(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
    }

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#FF8040").expect("color should be valid");
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
    }

    #[test]
    fn test_color_lerp() {
        let c1 = Color::new(0, 0, 0);
        let c2 = Color::new(255, 255, 255);
        let mid = c1.lerp(&c2, 0.5);

        assert!(mid.r > 120 && mid.r < 135);
        assert!(mid.g > 120 && mid.g < 135);
        assert!(mid.b > 120 && mid.b < 135);
    }

    #[test]
    fn test_gradient() {
        let gradient = ColorGradient::traffic_light();

        let color_low = gradient.color_at(0.0);
        let color_high = gradient.color_at(1.0);

        // Low should be greenish, high should be reddish
        assert!(color_low.g > color_low.r);
        assert!(color_high.r > color_high.g);
    }

    #[test]
    fn test_bar_meter_data_from_dbfs() {
        let data = BarMeterData::from_dbfs(-10.0, -5.0, -60.0, 0.0);

        assert!(data.level > 0.8); // -10 dB is high on -60 to 0 scale
        assert!(data.peak_hold > 0.9); // -5 dB is very high
    }

    #[test]
    fn test_bar_meter_data_clipping() {
        let data = BarMeterData::from_dbfs(0.5, 0.5, -60.0, 0.0);

        assert!(data.is_clipping);
    }

    #[test]
    fn test_generate_db_scale() {
        let marks = generate_db_scale(-60.0, 0.0);

        assert!(!marks.is_empty());

        // Should have markings at 0, -10, -20, etc.
        let has_zero = marks.iter().any(|m| m.label == "0");
        let has_minus_10 = marks.iter().any(|m| m.label == "-10");

        assert!(has_zero);
        assert!(has_minus_10);
    }

    #[test]
    fn test_default_configs() {
        let bar_config = BarMeterConfig::default();
        assert_eq!(bar_config.min_value, -60.0);
        assert_eq!(bar_config.max_value, 0.0);

        let circular_config = CircularMeterConfig::default();
        assert_eq!(circular_config.radius, 80);
    }
}
