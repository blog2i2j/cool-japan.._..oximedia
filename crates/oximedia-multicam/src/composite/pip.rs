//! Picture-in-picture composition.

use super::{Compositor, Layout};
use crate::{AngleId, Result};

/// PIP corner position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipPosition {
    /// Top-left corner
    TopLeft,
    /// Top-right corner
    TopRight,
    /// Bottom-left corner
    BottomLeft,
    /// Bottom-right corner
    BottomRight,
    /// Custom position (x, y in pixels)
    Custom(u32, u32),
}

/// Picture-in-picture compositor
#[derive(Debug)]
pub struct PictureInPicture {
    /// Output dimensions
    dimensions: (u32, u32),
    /// Current layout
    layout: Layout,
    /// PIP position
    pip_position: PipPosition,
    /// PIP size as fraction of main (0.0 to 1.0)
    pip_scale: f32,
    /// PIP padding from edges (pixels)
    pip_padding: u32,
    /// Border width (pixels)
    border_width: u32,
    /// Border color (RGBA)
    border_color: [u8; 4],
}

impl PictureInPicture {
    /// Create a new PIP compositor
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            dimensions: (width, height),
            layout: Layout::Single { angle: 0 },
            pip_position: PipPosition::BottomRight,
            pip_scale: 0.25,
            pip_padding: 20,
            border_width: 2,
            border_color: [255, 255, 255, 255],
        }
    }

    /// Set PIP position
    pub fn set_position(&mut self, position: PipPosition) {
        self.pip_position = position;
    }

    /// Get PIP position
    #[must_use]
    pub fn position(&self) -> PipPosition {
        self.pip_position
    }

    /// Set PIP scale (0.0 to 1.0)
    pub fn set_scale(&mut self, scale: f32) {
        self.pip_scale = scale.clamp(0.1, 0.5);
    }

    /// Get PIP scale
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.pip_scale
    }

    /// Set PIP padding
    pub fn set_padding(&mut self, padding: u32) {
        self.pip_padding = padding;
    }

    /// Get PIP padding
    #[must_use]
    pub fn padding(&self) -> u32 {
        self.pip_padding
    }

    /// Set border width
    pub fn set_border_width(&mut self, width: u32) {
        self.border_width = width;
    }

    /// Set border color
    pub fn set_border_color(&mut self, color: [u8; 4]) {
        self.border_color = color;
    }

    /// Calculate PIP dimensions
    #[must_use]
    pub fn calculate_pip_dimensions(&self) -> (u32, u32) {
        let (width, height) = self.dimensions;
        let pip_width = (width as f32 * self.pip_scale) as u32;
        let pip_height = (height as f32 * self.pip_scale) as u32;
        (pip_width, pip_height)
    }

    /// Calculate PIP position in pixels
    #[must_use]
    pub fn calculate_pip_position(&self) -> (u32, u32) {
        let (width, height) = self.dimensions;
        let (pip_width, pip_height) = self.calculate_pip_dimensions();
        let padding = self.pip_padding;

        match self.pip_position {
            PipPosition::TopLeft => (padding, padding),
            PipPosition::TopRight => (width - pip_width - padding, padding),
            PipPosition::BottomLeft => (padding, height - pip_height - padding),
            PipPosition::BottomRight => {
                (width - pip_width - padding, height - pip_height - padding)
            }
            PipPosition::Custom(x, y) => (x, y),
        }
    }

    /// Get main video region
    #[must_use]
    pub fn main_region(&self) -> (u32, u32, u32, u32) {
        let (width, height) = self.dimensions;
        (0, 0, width, height)
    }

    /// Get PIP video region
    #[must_use]
    pub fn pip_region(&self) -> (u32, u32, u32, u32) {
        let (x, y) = self.calculate_pip_position();
        let (w, h) = self.calculate_pip_dimensions();
        (x, y, w, h)
    }

    /// Swap main and inset
    pub fn swap_angles(&mut self) {
        if let Layout::PictureInPicture { main, inset } = self.layout {
            self.layout = Layout::PictureInPicture {
                main: inset,
                inset: main,
            };
        }
    }

    /// Set main angle
    pub fn set_main_angle(&mut self, angle: AngleId) {
        if let Layout::PictureInPicture { inset, .. } = self.layout {
            self.layout = Layout::PictureInPicture { main: angle, inset };
        }
    }

    /// Set inset angle
    pub fn set_inset_angle(&mut self, angle: AngleId) {
        if let Layout::PictureInPicture { main, .. } = self.layout {
            self.layout = Layout::PictureInPicture { main, inset: angle };
        }
    }

    /// Get main angle
    #[must_use]
    pub fn main_angle(&self) -> Option<AngleId> {
        if let Layout::PictureInPicture { main, .. } = self.layout {
            Some(main)
        } else {
            None
        }
    }

    /// Get inset angle
    #[must_use]
    pub fn inset_angle(&self) -> Option<AngleId> {
        if let Layout::PictureInPicture { inset, .. } = self.layout {
            Some(inset)
        } else {
            None
        }
    }
}

impl Compositor for PictureInPicture {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.dimensions = (width, height);
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn set_layout(&mut self, layout: Layout) -> Result<()> {
        match layout {
            Layout::PictureInPicture { .. } | Layout::Single { .. } => {
                self.layout = layout;
                Ok(())
            }
            _ => Err(crate::MultiCamError::LayoutError(
                "PIP compositor only supports PictureInPicture and Single layouts".to_string(),
            )),
        }
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// PIP animation
#[derive(Debug)]
pub struct PipAnimation {
    /// Start position
    start_position: PipPosition,
    /// End position
    end_position: PipPosition,
    /// Duration (frames)
    duration: u32,
    /// Current frame
    current_frame: u32,
}

impl PipAnimation {
    /// Create a new PIP animation
    #[must_use]
    pub fn new(start: PipPosition, end: PipPosition, duration: u32) -> Self {
        Self {
            start_position: start,
            end_position: end,
            duration,
            current_frame: 0,
        }
    }

    /// Update animation
    pub fn update(&mut self) {
        if self.current_frame < self.duration {
            self.current_frame += 1;
        }
    }

    /// Get current position
    #[must_use]
    pub fn current_position(
        &self,
        width: u32,
        height: u32,
        pip_size: (u32, u32),
        padding: u32,
    ) -> (u32, u32) {
        if self.duration == 0 {
            return Self::position_to_pixels(self.end_position, width, height, pip_size, padding);
        }

        let progress = self.current_frame as f32 / self.duration as f32;
        let start = Self::position_to_pixels(self.start_position, width, height, pip_size, padding);
        let end = Self::position_to_pixels(self.end_position, width, height, pip_size, padding);

        let x = start.0 as f32 + (end.0 as f32 - start.0 as f32) * progress;
        let y = start.1 as f32 + (end.1 as f32 - start.1 as f32) * progress;

        (x as u32, y as u32)
    }

    /// Convert position enum to pixels
    fn position_to_pixels(
        pos: PipPosition,
        width: u32,
        height: u32,
        pip_size: (u32, u32),
        padding: u32,
    ) -> (u32, u32) {
        match pos {
            PipPosition::TopLeft => (padding, padding),
            PipPosition::TopRight => (width - pip_size.0 - padding, padding),
            PipPosition::BottomLeft => (padding, height - pip_size.1 - padding),
            PipPosition::BottomRight => {
                (width - pip_size.0 - padding, height - pip_size.1 - padding)
            }
            PipPosition::Custom(x, y) => (x, y),
        }
    }

    /// Check if animation is complete
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_frame >= self.duration
    }

    /// Reset animation
    pub fn reset(&mut self) {
        self.current_frame = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pip_creation() {
        let pip = PictureInPicture::new(1920, 1080);
        assert_eq!(pip.dimensions(), (1920, 1080));
        assert_eq!(pip.scale(), 0.25);
    }

    #[test]
    fn test_pip_dimensions() {
        let pip = PictureInPicture::new(1920, 1080);
        let (w, h) = pip.calculate_pip_dimensions();
        assert_eq!(w, 480); // 1920 * 0.25
        assert_eq!(h, 270); // 1080 * 0.25
    }

    #[test]
    fn test_pip_position() {
        let mut pip = PictureInPicture::new(1920, 1080);
        pip.set_position(PipPosition::TopLeft);

        let (x, y) = pip.calculate_pip_position();
        assert_eq!(x, 20); // padding
        assert_eq!(y, 20);
    }

    #[test]
    fn test_swap_angles() {
        let mut pip = PictureInPicture::new(1920, 1080);
        pip.set_layout(Layout::PictureInPicture { main: 0, inset: 1 })
            .expect("multicam test operation should succeed");

        pip.swap_angles();
        assert_eq!(pip.main_angle(), Some(1));
        assert_eq!(pip.inset_angle(), Some(0));
    }

    #[test]
    fn test_pip_regions() {
        let pip = PictureInPicture::new(1920, 1080);
        let main = pip.main_region();
        assert_eq!(main, (0, 0, 1920, 1080));

        let pip_rect = pip.pip_region();
        assert!(pip_rect.2 < 1920); // Width smaller than main
        assert!(pip_rect.3 < 1080); // Height smaller than main
    }

    #[test]
    fn test_pip_animation() {
        let mut anim = PipAnimation::new(PipPosition::TopLeft, PipPosition::BottomRight, 10);
        assert!(!anim.is_complete());

        for _ in 0..10 {
            anim.update();
        }
        assert!(anim.is_complete());
    }

    #[test]
    fn test_animation_position() {
        let anim = PipAnimation::new(PipPosition::TopLeft, PipPosition::BottomRight, 10);
        let pos = anim.current_position(1920, 1080, (480, 270), 20);
        assert_eq!(pos.0, 20); // At start
    }
}
