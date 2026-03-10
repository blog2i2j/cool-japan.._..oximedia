//! Video overlay integration

use crate::error::{GraphicsError, Result};
use crate::primitives::{Point, Rect};
use crate::render::RenderTarget;

/// Video overlay compositor
pub struct OverlayCompositor {
    /// Width of video frame
    pub width: u32,
    /// Height of video frame
    pub height: u32,
}

impl OverlayCompositor {
    /// Create a new overlay compositor
    pub fn new(width: u32, height: u32) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(GraphicsError::InvalidDimensions(width, height));
        }

        Ok(Self { width, height })
    }

    /// Composite graphics onto video frame
    pub fn composite(
        &self,
        video_frame: &mut [u8],
        graphics: &RenderTarget,
        position: Point,
        opacity: f32,
    ) -> Result<()> {
        if video_frame.len() != (self.width * self.height * 4) as usize {
            return Err(GraphicsError::InvalidParameter(
                "Video frame size mismatch".to_string(),
            ));
        }

        let alpha = opacity.clamp(0.0, 1.0);

        for y in 0..graphics.height.min(self.height) {
            for x in 0..graphics.width.min(self.width) {
                let gx = x;
                let gy = y;
                let vx = (x as f32 + position.x) as u32;
                let vy = (y as f32 + position.y) as u32;

                if vx >= self.width || vy >= self.height {
                    continue;
                }

                let g_idx = ((gy * graphics.width + gx) * 4) as usize;
                let v_idx = ((vy * self.width + vx) * 4) as usize;

                // Get graphics pixel
                let gr = graphics.data[g_idx];
                let gg = graphics.data[g_idx + 1];
                let gb = graphics.data[g_idx + 2];
                let ga = (f32::from(graphics.data[g_idx + 3]) * alpha) as u8;

                // Get video pixel
                let vr = video_frame[v_idx];
                let vg = video_frame[v_idx + 1];
                let vb = video_frame[v_idx + 2];

                // Alpha blend
                let ga_f = f32::from(ga) / 255.0;
                let inv_ga = 1.0 - ga_f;

                video_frame[v_idx] = (f32::from(gr) * ga_f + f32::from(vr) * inv_ga) as u8;
                video_frame[v_idx + 1] = (f32::from(gg) * ga_f + f32::from(vg) * inv_ga) as u8;
                video_frame[v_idx + 2] = (f32::from(gb) * ga_f + f32::from(vb) * inv_ga) as u8;
                video_frame[v_idx + 3] = 255; // Keep video alpha
            }
        }

        Ok(())
    }

    /// Create alpha matte from graphics
    pub fn create_matte(&self, graphics: &RenderTarget) -> Result<Vec<u8>> {
        let mut matte = vec![0u8; (self.width * self.height) as usize];

        for y in 0..graphics.height.min(self.height) {
            for x in 0..graphics.width.min(self.width) {
                let g_idx = ((y * graphics.width + x) * 4) as usize;
                let m_idx = (y * self.width + x) as usize;

                matte[m_idx] = graphics.data[g_idx + 3]; // Alpha channel
            }
        }

        Ok(matte)
    }

    /// Apply chroma key (green screen)
    pub fn chroma_key(&self, frame: &mut [u8], key_color: [u8; 3], threshold: u8) -> Result<()> {
        if frame.len() != (self.width * self.height * 4) as usize {
            return Err(GraphicsError::InvalidParameter(
                "Frame size mismatch".to_string(),
            ));
        }

        for pixel in frame.chunks_exact_mut(4) {
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // Calculate color distance
            let dr = (i32::from(r) - i32::from(key_color[0])).unsigned_abs() as u8;
            let dg = (i32::from(g) - i32::from(key_color[1])).unsigned_abs() as u8;
            let db = (i32::from(b) - i32::from(key_color[2])).unsigned_abs() as u8;

            let distance = dr.max(dg).max(db);

            if distance < threshold {
                // Make transparent
                pixel[3] = 0;
            }
        }

        Ok(())
    }

    /// Apply luma key (brightness-based keying)
    pub fn luma_key(&self, frame: &mut [u8], min_luma: u8, max_luma: u8) -> Result<()> {
        if frame.len() != (self.width * self.height * 4) as usize {
            return Err(GraphicsError::InvalidParameter(
                "Frame size mismatch".to_string(),
            ));
        }

        for pixel in frame.chunks_exact_mut(4) {
            let r = pixel[0];
            let g = pixel[1];
            let b = pixel[2];

            // Calculate luma (ITU-R BT.709)
            let luma =
                (0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b)) as u8;

            if luma >= min_luma && luma <= max_luma {
                // Make transparent
                pixel[3] = 0;
            }
        }

        Ok(())
    }
}

/// Safe area guide for broadcast graphics
#[derive(Debug, Clone, Copy)]
pub struct SafeArea {
    /// Title safe area (90% of frame)
    pub title_safe: Rect,
    /// Action safe area (93% of frame)
    pub action_safe: Rect,
}

impl SafeArea {
    /// Calculate safe areas for given dimensions
    #[must_use]
    pub fn calculate(width: u32, height: u32) -> Self {
        let w = width as f32;
        let h = height as f32;

        // Title safe: 90% of frame
        let title_margin_x = w * 0.05;
        let title_margin_y = h * 0.05;
        let title_safe = Rect::new(title_margin_x, title_margin_y, w * 0.9, h * 0.9);

        // Action safe: 93% of frame
        let action_margin_x = w * 0.035;
        let action_margin_y = h * 0.035;
        let action_safe = Rect::new(action_margin_x, action_margin_y, w * 0.93, h * 0.93);

        Self {
            title_safe,
            action_safe,
        }
    }

    /// Check if point is in title safe area
    #[must_use]
    pub fn is_title_safe(&self, point: Point) -> bool {
        self.title_safe.contains(point)
    }

    /// Check if point is in action safe area
    #[must_use]
    pub fn is_action_safe(&self, point: Point) -> bool {
        self.action_safe.contains(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overlay_compositor_creation() {
        let comp = OverlayCompositor::new(1920, 1080).expect("comp should be valid");
        assert_eq!(comp.width, 1920);
        assert_eq!(comp.height, 1080);
    }

    #[test]
    fn test_overlay_compositor_invalid() {
        assert!(OverlayCompositor::new(0, 0).is_err());
    }

    #[test]
    fn test_composite() {
        let comp = OverlayCompositor::new(100, 100).expect("comp should be valid");
        let mut video = vec![0u8; 100 * 100 * 4];
        let graphics = RenderTarget::new(100, 100).expect("graphics should be valid");

        let result = comp.composite(&mut video, &graphics, Point::new(0.0, 0.0), 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_matte() {
        let comp = OverlayCompositor::new(100, 100).expect("comp should be valid");
        let graphics = RenderTarget::new(100, 100).expect("graphics should be valid");

        let matte = comp.create_matte(&graphics).expect("matte should be valid");
        assert_eq!(matte.len(), 100 * 100);
    }

    #[test]
    fn test_chroma_key() {
        let comp = OverlayCompositor::new(100, 100).expect("comp should be valid");
        let mut frame = vec![0u8; 100 * 100 * 4];

        // Set all pixels to green
        for pixel in frame.chunks_exact_mut(4) {
            pixel[0] = 0;
            pixel[1] = 255;
            pixel[2] = 0;
            pixel[3] = 255;
        }

        let result = comp.chroma_key(&mut frame, [0, 255, 0], 50);
        assert!(result.is_ok());

        // Check that pixels are now transparent
        assert_eq!(frame[3], 0);
    }

    #[test]
    fn test_luma_key() {
        let comp = OverlayCompositor::new(100, 100).expect("comp should be valid");
        let mut frame = vec![0u8; 100 * 100 * 4];

        // Set all pixels to white
        for pixel in frame.chunks_exact_mut(4) {
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
            pixel[3] = 255;
        }

        let result = comp.luma_key(&mut frame, 200, 255);
        assert!(result.is_ok());

        // Check that pixels are now transparent
        assert_eq!(frame[3], 0);
    }

    #[test]
    fn test_safe_area() {
        let safe_area = SafeArea::calculate(1920, 1080);

        // Check title safe
        assert!(safe_area.is_title_safe(Point::new(960.0, 540.0))); // Center
        assert!(!safe_area.is_title_safe(Point::new(10.0, 10.0))); // Corner

        // Check action safe
        assert!(safe_area.is_action_safe(Point::new(960.0, 540.0))); // Center
    }

    #[test]
    fn test_safe_area_dimensions() {
        let safe_area = SafeArea::calculate(1920, 1080);

        // Title safe should be 90% of frame
        assert!((safe_area.title_safe.width - 1920.0 * 0.9).abs() < 1.0);
        assert!((safe_area.title_safe.height - 1080.0 * 0.9).abs() < 1.0);

        // Action safe should be 93% of frame
        assert!((safe_area.action_safe.width - 1920.0 * 0.93).abs() < 1.0);
        assert!((safe_area.action_safe.height - 1080.0 * 0.93).abs() < 1.0);
    }
}
