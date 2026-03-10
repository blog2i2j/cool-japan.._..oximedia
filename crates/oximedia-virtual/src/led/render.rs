//! LED wall content rendering
//!
//! Provides real-time rendering of content for LED walls with
//! perspective correction and multi-panel support.

use super::{perspective::PerspectiveCorrection, LedPanel, LedWall};
use crate::math::{Matrix4, Point3, Vector3};
use crate::{tracking::CameraPose, Result, VirtualProductionError};
use serde::{Deserialize, Serialize};

/// LED renderer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedRendererConfig {
    /// Target frame rate
    pub target_fps: f64,
    /// Enable perspective correction
    pub perspective_correction: bool,
    /// Enable color correction
    pub color_correction: bool,
    /// Render quality (0.0 - 1.0)
    pub quality: f32,
    /// Enable motion blur
    pub motion_blur: bool,
}

impl Default for LedRendererConfig {
    fn default() -> Self {
        Self {
            target_fps: 60.0,
            perspective_correction: true,
            color_correction: true,
            quality: 1.0,
            motion_blur: false,
        }
    }
}

/// Render output for LED wall
#[derive(Debug, Clone)]
pub struct RenderOutput {
    /// Rendered pixel data (RGB)
    pub pixels: Vec<u8>,
    /// Output width
    pub width: usize,
    /// Output height
    pub height: usize,
    /// Frame number
    pub frame_number: u64,
    /// Timestamp in nanoseconds
    pub timestamp_ns: u64,
}

impl RenderOutput {
    /// Create new render output
    #[must_use]
    pub fn new(width: usize, height: usize, frame_number: u64, timestamp_ns: u64) -> Self {
        Self {
            pixels: vec![0; width * height * 3],
            width,
            height,
            frame_number,
            timestamp_ns,
        }
    }

    /// Get pixel at position
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.pixels[idx], self.pixels[idx + 1], self.pixels[idx + 2]])
    }

    /// Set pixel at position
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = (y * self.width + x) * 3;
        self.pixels[idx] = rgb[0];
        self.pixels[idx + 1] = rgb[1];
        self.pixels[idx + 2] = rgb[2];
    }
}

/// LED wall renderer
pub struct LedRenderer {
    config: LedRendererConfig,
    led_wall: Option<LedWall>,
    perspective: PerspectiveCorrection,
    frame_number: u64,
}

impl LedRenderer {
    /// Create new LED renderer
    pub fn new(config: LedRendererConfig) -> Result<Self> {
        let perspective = PerspectiveCorrection::new()?;

        Ok(Self {
            config,
            led_wall: None,
            perspective,
            frame_number: 0,
        })
    }

    /// Set LED wall configuration
    pub fn set_led_wall(&mut self, wall: LedWall) {
        self.led_wall = Some(wall);
    }

    /// Render frame for LED wall
    pub fn render(
        &mut self,
        camera_pose: &CameraPose,
        source_frame: &[u8],
        source_width: usize,
        source_height: usize,
        timestamp_ns: u64,
    ) -> Result<RenderOutput> {
        let (output_width, output_height) = self
            .led_wall
            .as_ref()
            .ok_or_else(|| VirtualProductionError::LedWall("No LED wall configured".to_string()))?
            .total_resolution();

        // Collect panel data to avoid borrow conflict with render_panel
        let panels: Vec<_> = self
            .led_wall
            .as_ref()
            .expect("invariant: led_wall checked via ok_or_else just above")
            .panels
            .clone();

        let mut output =
            RenderOutput::new(output_width, output_height, self.frame_number, timestamp_ns);

        // Render each panel
        for panel in &panels {
            self.render_panel(
                panel,
                camera_pose,
                source_frame,
                source_width,
                source_height,
                &mut output,
            )?;
        }

        self.frame_number += 1;

        Ok(output)
    }

    /// Render content for a single panel
    #[allow(clippy::too_many_arguments)]
    fn render_panel(
        &mut self,
        panel: &LedPanel,
        camera_pose: &CameraPose,
        source_frame: &[u8],
        source_width: usize,
        source_height: usize,
        output: &mut RenderOutput,
    ) -> Result<()> {
        let (panel_width, panel_height) = panel.resolution;

        // Apply perspective correction if enabled
        let transform = if self.config.perspective_correction {
            self.perspective.compute_transform(camera_pose, panel)?
        } else {
            Matrix4::identity()
        };

        // Render each pixel
        for y in 0..panel_height {
            for x in 0..panel_width {
                // Compute world position of this pixel
                let pixel_x = (x as f64 / panel_width as f64) * panel.width;
                let pixel_y = (y as f64 / panel_height as f64) * panel.height;

                let world_pos = panel.position + Vector3::new(pixel_x, pixel_y, 0.0);

                // Apply perspective transform
                let transformed = transform * world_pos.to_homogeneous();
                let screen_pos = Point3::from_homogeneous(transformed).unwrap_or(world_pos);

                // Map to source frame coordinates
                let src_x = ((screen_pos.x / panel.width) * source_width as f64) as usize;
                let src_y = ((screen_pos.y / panel.height) * source_height as f64) as usize;

                // Sample source pixel
                if src_x < source_width && src_y < source_height {
                    let src_idx = (src_y * source_width + src_x) * 3;
                    if src_idx + 2 < source_frame.len() {
                        let rgb = [
                            source_frame[src_idx],
                            source_frame[src_idx + 1],
                            source_frame[src_idx + 2],
                        ];
                        output.set_pixel(x, y, rgb);
                    }
                }
            }
        }

        Ok(())
    }

    /// Get current frame number
    #[must_use]
    pub fn frame_number(&self) -> u64 {
        self.frame_number
    }

    /// Reset frame counter
    pub fn reset_frame_counter(&mut self) {
        self.frame_number = 0;
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &LedRendererConfig {
        &self.config
    }

    /// Get LED wall
    #[must_use]
    pub fn led_wall(&self) -> Option<&LedWall> {
        self.led_wall.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_output() {
        let output = RenderOutput::new(1920, 1080, 0, 0);
        assert_eq!(output.width, 1920);
        assert_eq!(output.height, 1080);
        assert_eq!(output.pixels.len(), 1920 * 1080 * 3);
    }

    #[test]
    fn test_render_output_pixel() {
        let mut output = RenderOutput::new(100, 100, 0, 0);
        output.set_pixel(50, 50, [255, 128, 64]);

        let pixel = output.get_pixel(50, 50);
        assert_eq!(pixel, Some([255, 128, 64]));
    }

    #[test]
    fn test_led_renderer_creation() {
        let config = LedRendererConfig::default();
        let renderer = LedRenderer::new(config);
        assert!(renderer.is_ok());
    }

    #[test]
    fn test_led_renderer_frame_counter() {
        let config = LedRendererConfig::default();
        let mut renderer = LedRenderer::new(config).expect("should succeed in test");

        assert_eq!(renderer.frame_number(), 0);
        renderer.reset_frame_counter();
        assert_eq!(renderer.frame_number(), 0);
    }

    #[test]
    fn test_led_renderer_set_wall() {
        let config = LedRendererConfig::default();
        let mut renderer = LedRenderer::new(config).expect("should succeed in test");

        let wall = LedWall::new("Test Wall".to_string());
        renderer.set_led_wall(wall);

        assert!(renderer.led_wall().is_some());
    }
}
