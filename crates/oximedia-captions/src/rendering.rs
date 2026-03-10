//! Caption rendering and burn-in

use crate::error::Result;
use crate::types::{Caption, CaptionTrack, Timestamp};

/// Caption renderer for burning captions into video frames
pub struct CaptionRenderer {
    /// Video width
    width: u32,
    /// Video height
    height: u32,
    /// Safe title area margins (left, top, right, bottom)
    safe_area: (u32, u32, u32, u32),
}

impl CaptionRenderer {
    /// Create a new caption renderer
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        // Default safe title area: 10% margin on all sides
        let margin_h = width / 10;
        let margin_v = height / 10;

        Self {
            width,
            height,
            safe_area: (margin_h, margin_v, margin_h, margin_v),
        }
    }

    /// Set custom safe title area
    pub fn set_safe_area(&mut self, left: u32, top: u32, right: u32, bottom: u32) {
        self.safe_area = (left, top, right, bottom);
    }

    /// Get captions to display at a given timestamp
    #[must_use]
    pub fn get_active_captions<'a>(
        &self,
        track: &'a CaptionTrack,
        timestamp: Timestamp,
    ) -> Vec<&'a Caption> {
        track
            .captions
            .iter()
            .filter(|c| c.start <= timestamp && c.end > timestamp)
            .collect()
    }

    /// Calculate position for a caption
    #[must_use]
    pub fn calculate_position(&self, caption: &Caption) -> (u32, u32) {
        let (margin_left, margin_top, margin_right, margin_bottom) = self.safe_area;

        let x = match caption.style.alignment {
            crate::types::Alignment::Left => margin_left,
            crate::types::Alignment::Center => self.width / 2,
            crate::types::Alignment::Right => self.width - margin_right,
            crate::types::Alignment::Justified => margin_left,
        };

        let y = match caption.position.vertical {
            crate::types::VerticalPosition::Top => margin_top,
            crate::types::VerticalPosition::Middle => self.height / 2,
            crate::types::VerticalPosition::Bottom => self.height - margin_bottom,
            crate::types::VerticalPosition::Custom(percent) => {
                (self.height as f32 * (f32::from(percent) / 100.0)) as u32
            }
        };

        (x, y)
    }

    /// Get the safe title area bounding box
    #[must_use]
    pub fn safe_area_bounds(&self) -> (u32, u32, u32, u32) {
        let (left, top, right, bottom) = self.safe_area;
        (left, top, self.width - right, self.height - bottom)
    }
}

/// Preview generator for captions
pub struct PreviewGenerator {
    renderer: CaptionRenderer,
}

impl PreviewGenerator {
    /// Create a new preview generator
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            renderer: CaptionRenderer::new(width, height),
        }
    }

    /// Generate a preview frame with captions
    pub fn generate_preview(
        &self,
        track: &CaptionTrack,
        timestamp: Timestamp,
    ) -> Result<PreviewFrame> {
        let active_captions = self.renderer.get_active_captions(track, timestamp);

        let mut frame = PreviewFrame {
            width: self.renderer.width,
            height: self.renderer.height,
            captions: Vec::new(),
        };

        for caption in active_captions {
            let (x, y) = self.renderer.calculate_position(caption);
            frame.captions.push(RenderedCaption {
                text: caption.text.clone(),
                position: (x, y),
                style: caption.style.clone(),
            });
        }

        Ok(frame)
    }
}

/// Preview frame with rendered captions
#[derive(Debug, Clone)]
pub struct PreviewFrame {
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Rendered captions
    pub captions: Vec<RenderedCaption>,
}

/// Rendered caption data
#[derive(Debug, Clone)]
pub struct RenderedCaption {
    /// Caption text
    pub text: String,
    /// Position (x, y)
    pub position: (u32, u32),
    /// Style
    pub style: crate::types::CaptionStyle,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Language;

    #[test]
    fn test_renderer_creation() {
        let renderer = CaptionRenderer::new(1920, 1080);
        assert_eq!(renderer.width, 1920);
        assert_eq!(renderer.height, 1080);
    }

    #[test]
    fn test_active_captions() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let renderer = CaptionRenderer::new(1920, 1080);
        let active = renderer.get_active_captions(&track, Timestamp::from_secs(2));
        assert_eq!(active.len(), 1);

        let active = renderer.get_active_captions(&track, Timestamp::from_secs(5));
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_position_calculation() {
        let renderer = CaptionRenderer::new(1920, 1080);
        let caption = Caption::new(
            Timestamp::from_secs(1),
            Timestamp::from_secs(3),
            "Test".to_string(),
        );

        let (x, y) = renderer.calculate_position(&caption);
        assert_eq!(x, 960); // Center horizontally
        assert!(y > 0); // Bottom vertically with margin
    }
}
