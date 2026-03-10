//! Split-screen composition.

use super::{Compositor, Layout};
use crate::{AngleId, Result};

/// Split orientation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitOrientation {
    /// Horizontal split (top/bottom)
    Horizontal,
    /// Vertical split (left/right)
    Vertical,
}

/// Split-screen compositor
#[derive(Debug)]
pub struct SplitScreen {
    /// Output dimensions
    dimensions: (u32, u32),
    /// Current layout
    layout: Layout,
    /// Split orientation
    orientation: SplitOrientation,
    /// Split ratio (0.0 to 1.0, position of divider)
    split_ratio: f32,
    /// Gap width (pixels)
    gap_width: u32,
    /// Gap color (RGBA)
    gap_color: [u8; 4],
}

impl SplitScreen {
    /// Create a new split-screen compositor
    #[must_use]
    pub fn new(width: u32, height: u32, orientation: SplitOrientation) -> Self {
        Self {
            dimensions: (width, height),
            layout: Layout::Single { angle: 0 },
            orientation,
            split_ratio: 0.5,
            gap_width: 0,
            gap_color: [0, 0, 0, 255],
        }
    }

    /// Set orientation
    pub fn set_orientation(&mut self, orientation: SplitOrientation) {
        self.orientation = orientation;
    }

    /// Get orientation
    #[must_use]
    pub fn orientation(&self) -> SplitOrientation {
        self.orientation
    }

    /// Set split ratio
    pub fn set_split_ratio(&mut self, ratio: f32) {
        self.split_ratio = ratio.clamp(0.1, 0.9);
    }

    /// Get split ratio
    #[must_use]
    pub fn split_ratio(&self) -> f32 {
        self.split_ratio
    }

    /// Set gap width
    pub fn set_gap_width(&mut self, width: u32) {
        self.gap_width = width;
    }

    /// Set gap color
    pub fn set_gap_color(&mut self, color: [u8; 4]) {
        self.gap_color = color;
    }

    /// Calculate split regions
    #[must_use]
    pub fn calculate_regions(&self) -> ((u32, u32, u32, u32), (u32, u32, u32, u32)) {
        let (width, height) = self.dimensions;
        let gap = self.gap_width;

        match self.orientation {
            SplitOrientation::Vertical => {
                let split_x = (width as f32 * self.split_ratio) as u32;
                let left = (0, 0, split_x.saturating_sub(gap / 2), height);
                let right = (split_x + gap / 2, 0, width - split_x - gap / 2, height);
                (left, right)
            }
            SplitOrientation::Horizontal => {
                let split_y = (height as f32 * self.split_ratio) as u32;
                let top = (0, 0, width, split_y.saturating_sub(gap / 2));
                let bottom = (0, split_y + gap / 2, width, height - split_y - gap / 2);
                (top, bottom)
            }
        }
    }

    /// Get first angle
    #[must_use]
    pub fn first_angle(&self) -> Option<AngleId> {
        match self.layout {
            Layout::SplitScreen { left, .. } => Some(left),
            _ => None,
        }
    }

    /// Get second angle
    #[must_use]
    pub fn second_angle(&self) -> Option<AngleId> {
        match self.layout {
            Layout::SplitScreen { right, .. } => Some(right),
            _ => None,
        }
    }

    /// Set first angle
    pub fn set_first_angle(&mut self, angle: AngleId) {
        if let Layout::SplitScreen { right, .. } = self.layout {
            self.layout = Layout::SplitScreen { left: angle, right };
        }
    }

    /// Set second angle
    pub fn set_second_angle(&mut self, angle: AngleId) {
        if let Layout::SplitScreen { left, .. } = self.layout {
            self.layout = Layout::SplitScreen { left, right: angle };
        }
    }

    /// Swap angles
    pub fn swap_angles(&mut self) {
        if let Layout::SplitScreen { left, right } = self.layout {
            self.layout = Layout::SplitScreen {
                left: right,
                right: left,
            };
        }
    }

    /// Create multi-way split
    #[must_use]
    pub fn create_multi_split(
        angle_count: usize,
        width: u32,
        height: u32,
        orientation: SplitOrientation,
    ) -> Vec<(u32, u32, u32, u32)> {
        let mut regions = Vec::new();

        match orientation {
            SplitOrientation::Vertical => {
                let region_width = width / angle_count as u32;
                for i in 0..angle_count {
                    let x = i as u32 * region_width;
                    regions.push((x, 0, region_width, height));
                }
            }
            SplitOrientation::Horizontal => {
                let region_height = height / angle_count as u32;
                for i in 0..angle_count {
                    let y = i as u32 * region_height;
                    regions.push((0, y, width, region_height));
                }
            }
        }

        regions
    }
}

impl Compositor for SplitScreen {
    fn set_dimensions(&mut self, width: u32, height: u32) {
        self.dimensions = (width, height);
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn set_layout(&mut self, layout: Layout) -> Result<()> {
        match layout {
            Layout::SplitScreen { .. } | Layout::Single { .. } => {
                self.layout = layout;
                Ok(())
            }
            _ => Err(crate::MultiCamError::LayoutError(
                "SplitScreen compositor only supports SplitScreen and Single layouts".to_string(),
            )),
        }
    }

    fn layout(&self) -> &Layout {
        &self.layout
    }
}

/// Three-way split compositor
#[derive(Debug)]
pub struct ThreeWaySplit {
    /// Output dimensions
    dimensions: (u32, u32),
    /// Split mode
    mode: ThreeWayMode,
}

/// Three-way split mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeWayMode {
    /// Horizontal (3 equal columns)
    Horizontal,
    /// Vertical (3 equal rows)
    Vertical,
    /// One large + two small (main on left, 2 stacked on right)
    MainLeftTwoRight,
    /// One large + two small (main on top, 2 side by side below)
    MainTopTwoBottom,
}

impl ThreeWaySplit {
    /// Create a new three-way split compositor
    #[must_use]
    pub fn new(width: u32, height: u32, mode: ThreeWayMode) -> Self {
        Self {
            dimensions: (width, height),
            mode,
        }
    }

    /// Calculate regions for three angles
    #[must_use]
    pub fn calculate_regions(&self) -> [(u32, u32, u32, u32); 3] {
        let (width, height) = self.dimensions;

        match self.mode {
            ThreeWayMode::Horizontal => {
                let w = width / 3;
                [(0, 0, w, height), (w, 0, w, height), (w * 2, 0, w, height)]
            }
            ThreeWayMode::Vertical => {
                let h = height / 3;
                [(0, 0, width, h), (0, h, width, h), (0, h * 2, width, h)]
            }
            ThreeWayMode::MainLeftTwoRight => {
                let main_w = (width * 2) / 3;
                let side_w = width / 3;
                let side_h = height / 2;
                [
                    (0, 0, main_w, height),
                    (main_w, 0, side_w, side_h),
                    (main_w, side_h, side_w, side_h),
                ]
            }
            ThreeWayMode::MainTopTwoBottom => {
                let main_h = (height * 2) / 3;
                let bottom_h = height / 3;
                let bottom_w = width / 2;
                [
                    (0, 0, width, main_h),
                    (0, main_h, bottom_w, bottom_h),
                    (bottom_w, main_h, bottom_w, bottom_h),
                ]
            }
        }
    }

    /// Set split mode
    pub fn set_mode(&mut self, mode: ThreeWayMode) {
        self.mode = mode;
    }

    /// Get split mode
    #[must_use]
    pub fn mode(&self) -> ThreeWayMode {
        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_creation() {
        let split = SplitScreen::new(1920, 1080, SplitOrientation::Vertical);
        assert_eq!(split.dimensions(), (1920, 1080));
        assert_eq!(split.orientation(), SplitOrientation::Vertical);
    }

    #[test]
    fn test_split_ratio() {
        let mut split = SplitScreen::new(1920, 1080, SplitOrientation::Vertical);
        split.set_split_ratio(0.6);
        assert_eq!(split.split_ratio(), 0.6);
    }

    #[test]
    fn test_vertical_split_regions() {
        let split = SplitScreen::new(1920, 1080, SplitOrientation::Vertical);
        let (left, right) = split.calculate_regions();

        assert_eq!(left.0, 0); // x
        assert_eq!(left.1, 0); // y
        assert!(left.2 > 0); // width
        assert_eq!(left.3, 1080); // height

        assert!(right.0 > 0); // x
        assert_eq!(right.1, 0); // y
        assert!(right.2 > 0); // width
        assert_eq!(right.3, 1080); // height
    }

    #[test]
    fn test_horizontal_split_regions() {
        let split = SplitScreen::new(1920, 1080, SplitOrientation::Horizontal);
        let (top, bottom) = split.calculate_regions();

        assert_eq!(top.0, 0); // x
        assert_eq!(top.1, 0); // y
        assert_eq!(top.2, 1920); // width
        assert!(top.3 > 0); // height

        assert_eq!(bottom.0, 0); // x
        assert!(bottom.1 > 0); // y
        assert_eq!(bottom.2, 1920); // width
        assert!(bottom.3 > 0); // height
    }

    #[test]
    fn test_swap_angles() {
        let mut split = SplitScreen::new(1920, 1080, SplitOrientation::Vertical);
        split
            .set_layout(Layout::SplitScreen { left: 0, right: 1 })
            .expect("multicam test operation should succeed");

        split.swap_angles();
        assert_eq!(split.first_angle(), Some(1));
        assert_eq!(split.second_angle(), Some(0));
    }

    #[test]
    fn test_multi_split() {
        let regions = SplitScreen::create_multi_split(4, 1920, 1080, SplitOrientation::Vertical);
        assert_eq!(regions.len(), 4);
        assert_eq!(regions[0].2, 480); // Each region is 1920/4 = 480 wide
    }

    #[test]
    fn test_three_way_horizontal() {
        let split = ThreeWaySplit::new(1920, 1080, ThreeWayMode::Horizontal);
        let regions = split.calculate_regions();
        assert_eq!(regions.len(), 3);
        assert_eq!(regions[0].2, 640); // Each region is 1920/3 = 640 wide
    }

    #[test]
    fn test_three_way_main_left() {
        let split = ThreeWaySplit::new(1920, 1080, ThreeWayMode::MainLeftTwoRight);
        let regions = split.calculate_regions();

        // Main region should be larger
        assert!(regions[0].2 > regions[1].2);
        assert!(regions[0].3 > regions[1].3);
    }
}
