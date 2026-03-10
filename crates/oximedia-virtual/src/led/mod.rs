//! LED wall rendering and calibration subsystem
//!
//! Provides real-time LED wall content rendering with perspective correction,
//! color calibration, and viewing frustum calculation.

pub mod calibrate;
pub mod calibration;
pub mod color;
pub mod frustum;
pub mod perspective;
pub mod processor;
pub mod render;

use crate::math::Point3;
use serde::{Deserialize, Serialize};

/// LED wall panel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedPanel {
    /// Panel position in world space (meters)
    pub position: Point3<f64>,
    /// Panel width in meters
    pub width: f64,
    /// Panel height in meters
    pub height: f64,
    /// Resolution (width x height in pixels)
    pub resolution: (usize, usize),
    /// Pixel pitch in millimeters
    pub pixel_pitch: f64,
}

impl LedPanel {
    /// Create new LED panel
    #[must_use]
    pub fn new(
        position: Point3<f64>,
        width: f64,
        height: f64,
        resolution: (usize, usize),
        pixel_pitch: f64,
    ) -> Self {
        Self {
            position,
            width,
            height,
            resolution,
            pixel_pitch,
        }
    }

    /// Get total pixel count
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.resolution.0 * self.resolution.1
    }

    /// Get aspect ratio
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        self.width / self.height
    }
}

/// LED wall configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedWall {
    /// Wall panels
    pub panels: Vec<LedPanel>,
    /// Wall name/identifier
    pub name: String,
}

impl LedWall {
    /// Create new LED wall
    #[must_use]
    pub fn new(name: String) -> Self {
        Self {
            panels: Vec::new(),
            name,
        }
    }

    /// Add panel to wall
    pub fn add_panel(&mut self, panel: LedPanel) {
        self.panels.push(panel);
    }

    /// Get total resolution
    #[must_use]
    pub fn total_resolution(&self) -> (usize, usize) {
        let mut total_width = 0;
        let mut max_height = 0;

        for panel in &self.panels {
            total_width += panel.resolution.0;
            max_height = max_height.max(panel.resolution.1);
        }

        (total_width, max_height)
    }

    /// Get total pixel count
    #[must_use]
    pub fn total_pixel_count(&self) -> usize {
        self.panels.iter().map(LedPanel::pixel_count).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_led_panel() {
        let panel = LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5);

        assert_eq!(panel.pixel_count(), 1920 * 1080);
        assert!((panel.aspect_ratio() - 5.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_led_wall() {
        let mut wall = LedWall::new("Main Wall".to_string());
        wall.add_panel(LedPanel::new(Point3::origin(), 5.0, 3.0, (1920, 1080), 2.5));

        assert_eq!(wall.panels.len(), 1);
        assert_eq!(wall.total_resolution(), (1920, 1080));
    }
}
