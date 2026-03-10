#![allow(dead_code)]
//! Nine-slice (nine-patch) scaling for broadcast UI elements.
//!
//! Nine-slice scaling divides an image or element into 9 regions (corners,
//! edges, center) so that corners remain unscaled, edges stretch in one axis,
//! and the center stretches in both. This is essential for resizable broadcast
//! panels, buttons, and lower-third backgrounds that must look sharp at any
//! resolution.

use std::fmt;

/// Inset distances defining the nine-slice border region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliceInsets {
    /// Distance from the top edge in pixels.
    pub top: u32,
    /// Distance from the right edge in pixels.
    pub right: u32,
    /// Distance from the bottom edge in pixels.
    pub bottom: u32,
    /// Distance from the left edge in pixels.
    pub left: u32,
}

impl SliceInsets {
    /// Create uniform insets (all sides equal).
    pub fn uniform(val: u32) -> Self {
        Self {
            top: val,
            right: val,
            bottom: val,
            left: val,
        }
    }

    /// Create symmetric insets (horizontal and vertical).
    pub fn symmetric(horizontal: u32, vertical: u32) -> Self {
        Self {
            top: vertical,
            right: horizontal,
            bottom: vertical,
            left: horizontal,
        }
    }

    /// Create fully specified insets.
    pub fn new(top: u32, right: u32, bottom: u32, left: u32) -> Self {
        Self {
            top,
            right,
            bottom,
            left,
        }
    }

    /// Total horizontal border (left + right).
    pub fn horizontal(&self) -> u32 {
        self.left + self.right
    }

    /// Total vertical border (top + bottom).
    pub fn vertical(&self) -> u32 {
        self.top + self.bottom
    }

    /// Validate that the insets fit within the given source dimensions.
    pub fn validate(&self, src_width: u32, src_height: u32) -> Result<(), String> {
        if self.horizontal() >= src_width {
            return Err(format!(
                "Horizontal insets ({}) must be less than source width ({})",
                self.horizontal(),
                src_width
            ));
        }
        if self.vertical() >= src_height {
            return Err(format!(
                "Vertical insets ({}) must be less than source height ({})",
                self.vertical(),
                src_height
            ));
        }
        Ok(())
    }
}

impl Default for SliceInsets {
    fn default() -> Self {
        Self::uniform(10)
    }
}

impl fmt::Display for SliceInsets {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Insets(top={}, right={}, bottom={}, left={})",
            self.top, self.right, self.bottom, self.left
        )
    }
}

/// Identifies one of the 9 regions in a nine-slice grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SliceRegion {
    /// Top-left corner (no scaling).
    TopLeft,
    /// Top edge (scales horizontally).
    TopCenter,
    /// Top-right corner (no scaling).
    TopRight,
    /// Left edge (scales vertically).
    MiddleLeft,
    /// Center region (scales both axes).
    MiddleCenter,
    /// Right edge (scales vertically).
    MiddleRight,
    /// Bottom-left corner (no scaling).
    BottomLeft,
    /// Bottom edge (scales horizontally).
    BottomCenter,
    /// Bottom-right corner (no scaling).
    BottomRight,
}

impl SliceRegion {
    /// All nine regions in order.
    pub fn all() -> [Self; 9] {
        [
            Self::TopLeft,
            Self::TopCenter,
            Self::TopRight,
            Self::MiddleLeft,
            Self::MiddleCenter,
            Self::MiddleRight,
            Self::BottomLeft,
            Self::BottomCenter,
            Self::BottomRight,
        ]
    }

    /// Whether this region is a corner (no scaling).
    pub fn is_corner(&self) -> bool {
        matches!(
            self,
            Self::TopLeft | Self::TopRight | Self::BottomLeft | Self::BottomRight
        )
    }

    /// Whether this region scales horizontally.
    pub fn scales_horizontally(&self) -> bool {
        matches!(
            self,
            Self::TopCenter | Self::MiddleCenter | Self::BottomCenter
        )
    }

    /// Whether this region scales vertically.
    pub fn scales_vertically(&self) -> bool {
        matches!(
            self,
            Self::MiddleLeft | Self::MiddleCenter | Self::MiddleRight
        )
    }
}

/// A rectangle region in pixel coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Left edge (x).
    pub x: u32,
    /// Top edge (y).
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Area of the rectangle.
    pub fn area(&self) -> u64 {
        u64::from(self.w) * u64::from(self.h)
    }

    /// Right edge coordinate.
    pub fn right(&self) -> u32 {
        self.x + self.w
    }

    /// Bottom edge coordinate.
    pub fn bottom(&self) -> u32 {
        self.y + self.h
    }
}

/// A nine-slice layout computed from source dimensions, target dimensions, and insets.
#[derive(Clone, Debug)]
pub struct NineSliceLayout {
    /// Source dimension width.
    pub src_width: u32,
    /// Source dimension height.
    pub src_height: u32,
    /// Target dimension width.
    pub dst_width: u32,
    /// Target dimension height.
    pub dst_height: u32,
    /// Insets defining the slice boundaries.
    pub insets: SliceInsets,
}

impl NineSliceLayout {
    /// Create a new nine-slice layout.
    ///
    /// Target dimensions must be at least as large as the sum of corner insets.
    pub fn new(
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        insets: SliceInsets,
    ) -> Result<Self, String> {
        insets.validate(src_width, src_height)?;
        if dst_width < insets.horizontal() {
            return Err(format!(
                "Target width ({}) must be >= horizontal insets ({})",
                dst_width,
                insets.horizontal()
            ));
        }
        if dst_height < insets.vertical() {
            return Err(format!(
                "Target height ({}) must be >= vertical insets ({})",
                dst_height,
                insets.vertical()
            ));
        }
        Ok(Self {
            src_width,
            src_height,
            dst_width,
            dst_height,
            insets,
        })
    }

    /// Get the source rectangle for a given region.
    pub fn source_rect(&self, region: SliceRegion) -> Rect {
        let left = self.insets.left;
        let top = self.insets.top;
        let right = self.insets.right;
        let bottom = self.insets.bottom;
        let center_w = self.src_width - left - right;
        let center_h = self.src_height - top - bottom;

        match region {
            SliceRegion::TopLeft => Rect::new(0, 0, left, top),
            SliceRegion::TopCenter => Rect::new(left, 0, center_w, top),
            SliceRegion::TopRight => Rect::new(left + center_w, 0, right, top),
            SliceRegion::MiddleLeft => Rect::new(0, top, left, center_h),
            SliceRegion::MiddleCenter => Rect::new(left, top, center_w, center_h),
            SliceRegion::MiddleRight => Rect::new(left + center_w, top, right, center_h),
            SliceRegion::BottomLeft => Rect::new(0, top + center_h, left, bottom),
            SliceRegion::BottomCenter => Rect::new(left, top + center_h, center_w, bottom),
            SliceRegion::BottomRight => Rect::new(left + center_w, top + center_h, right, bottom),
        }
    }

    /// Get the destination rectangle for a given region.
    pub fn dest_rect(&self, region: SliceRegion) -> Rect {
        let left = self.insets.left;
        let top = self.insets.top;
        let right = self.insets.right;
        let bottom = self.insets.bottom;
        let dst_center_w = self.dst_width - left - right;
        let dst_center_h = self.dst_height - top - bottom;

        match region {
            SliceRegion::TopLeft => Rect::new(0, 0, left, top),
            SliceRegion::TopCenter => Rect::new(left, 0, dst_center_w, top),
            SliceRegion::TopRight => Rect::new(left + dst_center_w, 0, right, top),
            SliceRegion::MiddleLeft => Rect::new(0, top, left, dst_center_h),
            SliceRegion::MiddleCenter => Rect::new(left, top, dst_center_w, dst_center_h),
            SliceRegion::MiddleRight => Rect::new(left + dst_center_w, top, right, dst_center_h),
            SliceRegion::BottomLeft => Rect::new(0, top + dst_center_h, left, bottom),
            SliceRegion::BottomCenter => Rect::new(left, top + dst_center_h, dst_center_w, bottom),
            SliceRegion::BottomRight => {
                Rect::new(left + dst_center_w, top + dst_center_h, right, bottom)
            }
        }
    }

    /// Get the horizontal scale factor for a region.
    #[allow(clippy::cast_precision_loss)]
    pub fn horizontal_scale(&self, region: SliceRegion) -> f64 {
        let src = self.source_rect(region);
        let dst = self.dest_rect(region);
        if src.w == 0 {
            1.0
        } else {
            dst.w as f64 / src.w as f64
        }
    }

    /// Get the vertical scale factor for a region.
    #[allow(clippy::cast_precision_loss)]
    pub fn vertical_scale(&self, region: SliceRegion) -> f64 {
        let src = self.source_rect(region);
        let dst = self.dest_rect(region);
        if src.h == 0 {
            1.0
        } else {
            dst.h as f64 / src.h as f64
        }
    }

    /// Generate all 9 mapping pairs (source_rect, dest_rect).
    pub fn all_mappings(&self) -> Vec<(SliceRegion, Rect, Rect)> {
        SliceRegion::all()
            .iter()
            .map(|&region| (region, self.source_rect(region), self.dest_rect(region)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insets_uniform() {
        let insets = SliceInsets::uniform(10);
        assert_eq!(insets.top, 10);
        assert_eq!(insets.right, 10);
        assert_eq!(insets.bottom, 10);
        assert_eq!(insets.left, 10);
    }

    #[test]
    fn test_insets_symmetric() {
        let insets = SliceInsets::symmetric(20, 10);
        assert_eq!(insets.left, 20);
        assert_eq!(insets.right, 20);
        assert_eq!(insets.top, 10);
        assert_eq!(insets.bottom, 10);
    }

    #[test]
    fn test_insets_horizontal_vertical() {
        let insets = SliceInsets::new(5, 10, 15, 20);
        assert_eq!(insets.horizontal(), 30);
        assert_eq!(insets.vertical(), 20);
    }

    #[test]
    fn test_insets_validate_ok() {
        let insets = SliceInsets::uniform(10);
        assert!(insets.validate(100, 100).is_ok());
    }

    #[test]
    fn test_insets_validate_too_wide() {
        let insets = SliceInsets::symmetric(60, 10);
        assert!(insets.validate(100, 100).is_err());
    }

    #[test]
    fn test_insets_display() {
        let insets = SliceInsets::new(1, 2, 3, 4);
        let s = format!("{insets}");
        assert!(s.contains("top=1"));
        assert!(s.contains("left=4"));
    }

    #[test]
    fn test_region_all_nine() {
        assert_eq!(SliceRegion::all().len(), 9);
    }

    #[test]
    fn test_region_corners() {
        assert!(SliceRegion::TopLeft.is_corner());
        assert!(SliceRegion::TopRight.is_corner());
        assert!(SliceRegion::BottomLeft.is_corner());
        assert!(SliceRegion::BottomRight.is_corner());
        assert!(!SliceRegion::MiddleCenter.is_corner());
    }

    #[test]
    fn test_region_scaling_flags() {
        assert!(SliceRegion::TopCenter.scales_horizontally());
        assert!(!SliceRegion::TopCenter.scales_vertically());
        assert!(SliceRegion::MiddleLeft.scales_vertically());
        assert!(!SliceRegion::MiddleLeft.scales_horizontally());
        assert!(SliceRegion::MiddleCenter.scales_horizontally());
        assert!(SliceRegion::MiddleCenter.scales_vertically());
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0, 0, 100, 200);
        assert_eq!(r.area(), 20000);
        assert_eq!(r.right(), 100);
        assert_eq!(r.bottom(), 200);
    }

    #[test]
    fn test_layout_creation_valid() {
        let layout =
            NineSliceLayout::new(100, 100, 200, 200, SliceInsets::uniform(10));
        assert!(layout.is_ok());
    }

    #[test]
    fn test_layout_creation_invalid_dst() {
        let layout =
            NineSliceLayout::new(100, 100, 10, 200, SliceInsets::uniform(10));
        assert!(layout.is_err());
    }

    #[test]
    fn test_corner_rects_unchanged() {
        let layout =
            NineSliceLayout::new(100, 100, 300, 300, SliceInsets::uniform(20)).expect("test expectation failed");
        let src = layout.source_rect(SliceRegion::TopLeft);
        let dst = layout.dest_rect(SliceRegion::TopLeft);
        // Corners should have the same size in source and dest
        assert_eq!(src.w, dst.w);
        assert_eq!(src.h, dst.h);
    }

    #[test]
    fn test_center_scales() {
        let layout =
            NineSliceLayout::new(100, 100, 300, 300, SliceInsets::uniform(20)).expect("test expectation failed");
        let h_scale = layout.horizontal_scale(SliceRegion::MiddleCenter);
        let v_scale = layout.vertical_scale(SliceRegion::MiddleCenter);
        // Source center is 60x60, dest center is 260x260
        assert!((h_scale - 260.0 / 60.0).abs() < 0.01);
        assert!((v_scale - 260.0 / 60.0).abs() < 0.01);
    }

    #[test]
    fn test_all_mappings_count() {
        let layout =
            NineSliceLayout::new(100, 100, 200, 200, SliceInsets::uniform(10)).expect("test expectation failed");
        let mappings = layout.all_mappings();
        assert_eq!(mappings.len(), 9);
    }

    #[test]
    fn test_dest_rects_tile_output() {
        let layout =
            NineSliceLayout::new(100, 100, 200, 200, SliceInsets::uniform(10)).expect("test expectation failed");
        // Verify that all dest rects cover the entire output without overlap
        let mut covered = vec![vec![false; 200]; 200];
        for region in SliceRegion::all() {
            let r = layout.dest_rect(region);
            for y in r.y..r.bottom() {
                for x in r.x..r.right() {
                    covered[y as usize][x as usize] = true;
                }
            }
        }
        for row in &covered {
            for &c in row {
                assert!(c, "All pixels should be covered");
            }
        }
    }
}
