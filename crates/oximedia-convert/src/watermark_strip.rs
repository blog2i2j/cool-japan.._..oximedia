#![allow(dead_code)]
//! Watermark and overlay stripping utilities for conversion.
//!
//! This module provides detection and removal of common overlays applied during
//! pre-conversion review, including:
//! - Static text overlays (timecode burn-in detection)
//! - Semi-transparent watermark detection
//! - Logo region detection for inpainting preparation
//! - Safe-area and action-safe margin computation

use std::fmt;

/// A rectangular region within a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    /// Left edge (pixels from left).
    pub x: u32,
    /// Top edge (pixels from top).
    pub y: u32,
    /// Width of the region.
    pub width: u32,
    /// Height of the region.
    pub height: u32,
}

impl Region {
    /// Create a new region.
    #[must_use]
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Area of the region in pixels.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Check if a point (px, py) is inside this region.
    #[must_use]
    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }

    /// Check if two regions overlap.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }

    /// Compute the intersection of two regions, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        if x1 < x2 && y1 < y2 {
            Some(Self {
                x: x1,
                y: y1,
                width: x2 - x1,
                height: y2 - y1,
            })
        } else {
            None
        }
    }

    /// Compute the bounding box union of two regions.
    #[must_use]
    pub fn union_bbox(&self, other: &Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let x2 = (self.x + self.width).max(other.x + other.width);
        let y2 = (self.y + self.height).max(other.y + other.height);
        Self {
            x,
            y,
            width: x2 - x,
            height: y2 - y,
        }
    }
}

impl fmt::Display for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Region(x={}, y={}, {}x{})",
            self.x, self.y, self.width, self.height
        )
    }
}

/// Type of detected overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayKind {
    /// Timecode burn-in text.
    TimecodeBurnIn,
    /// Semi-transparent watermark.
    Watermark,
    /// Logo overlay.
    Logo,
    /// Bug or DOG (digitally originated graphic).
    Bug,
    /// Unknown static overlay.
    Unknown,
}

/// A detected overlay in a frame.
#[derive(Debug, Clone)]
pub struct DetectedOverlay {
    /// Region of the overlay.
    pub region: Region,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
    /// Type of overlay.
    pub kind: OverlayKind,
}

/// Configuration for overlay detection.
#[derive(Debug, Clone)]
pub struct DetectionConfig {
    /// Minimum confidence threshold to report an overlay.
    pub min_confidence: f64,
    /// Number of frames to analyze for static overlay detection.
    pub analysis_frames: usize,
    /// Variance threshold below which a pixel is considered static.
    pub static_variance_threshold: f64,
    /// Minimum region area (pixels) to consider as an overlay.
    pub min_region_area: u64,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.5,
            analysis_frames: 30,
            static_variance_threshold: 0.02,
            min_region_area: 100,
        }
    }
}

/// Analyze a set of frames to detect static overlays.
///
/// Each frame is a flat array of grayscale `f32` values in `[0, 1]`,
/// with dimensions `width x height` in row-major order.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn detect_static_overlays(
    frames: &[Vec<f32>],
    width: u32,
    height: u32,
    config: &DetectionConfig,
) -> Vec<DetectedOverlay> {
    if frames.is_empty() || width == 0 || height == 0 {
        return Vec::new();
    }
    let num_pixels = (width * height) as usize;
    let n = frames.len() as f64;

    // Compute per-pixel variance across frames
    let mut mean = vec![0.0_f64; num_pixels];
    for frame in frames {
        for (i, &val) in frame.iter().enumerate().take(num_pixels) {
            mean[i] += f64::from(val);
        }
    }
    for m in &mut mean {
        *m /= n;
    }

    let mut variance = vec![0.0_f64; num_pixels];
    for frame in frames {
        for (i, &val) in frame.iter().enumerate().take(num_pixels) {
            let diff = f64::from(val) - mean[i];
            variance[i] += diff * diff;
        }
    }
    for v in &mut variance {
        *v /= n;
    }

    // Mark static pixels
    let static_mask: Vec<bool> = variance
        .iter()
        .map(|&v| v < config.static_variance_threshold)
        .collect();

    // Find connected static regions using simple bounding box
    // (simplified: scan for contiguous static rows at edges)
    let mut overlays = Vec::new();

    // Check top region (timecode burn-in is often at top or bottom)
    let top_region = find_static_band(&static_mask, width, height, BandLocation::Top, config);
    if let Some(region) = top_region {
        if region.area() >= config.min_region_area {
            overlays.push(DetectedOverlay {
                region,
                confidence: 0.7,
                kind: OverlayKind::TimecodeBurnIn,
            });
        }
    }

    // Check bottom region
    let bot_region = find_static_band(&static_mask, width, height, BandLocation::Bottom, config);
    if let Some(region) = bot_region {
        if region.area() >= config.min_region_area {
            overlays.push(DetectedOverlay {
                region,
                confidence: 0.7,
                kind: OverlayKind::TimecodeBurnIn,
            });
        }
    }

    // Check corners for logos/bugs
    for corner in &[
        CornerLocation::TopRight,
        CornerLocation::TopLeft,
        CornerLocation::BottomRight,
        CornerLocation::BottomLeft,
    ] {
        let corner_region = find_static_corner(&static_mask, width, height, *corner, config);
        if let Some(region) = corner_region {
            if region.area() >= config.min_region_area {
                overlays.push(DetectedOverlay {
                    region,
                    confidence: 0.6,
                    kind: OverlayKind::Bug,
                });
            }
        }
    }

    overlays
        .into_iter()
        .filter(|o| o.confidence >= config.min_confidence)
        .collect()
}

/// Location of a band search.
#[derive(Debug, Clone, Copy)]
enum BandLocation {
    /// Top of the frame.
    Top,
    /// Bottom of the frame.
    Bottom,
}

/// Corner location.
#[derive(Debug, Clone, Copy)]
enum CornerLocation {
    /// Top-left corner.
    TopLeft,
    /// Top-right corner.
    TopRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-right corner.
    BottomRight,
}

/// Find a static horizontal band at the top or bottom.
#[allow(clippy::cast_precision_loss)]
fn find_static_band(
    mask: &[bool],
    width: u32,
    height: u32,
    location: BandLocation,
    _config: &DetectionConfig,
) -> Option<Region> {
    let max_band_height = (height / 8).max(1);
    let w = width as usize;

    let row_range: Box<dyn Iterator<Item = u32>> = match location {
        BandLocation::Top => Box::new(0..max_band_height),
        BandLocation::Bottom => Box::new((height - max_band_height)..height),
    };

    let mut band_end = 0_u32;
    let mut found = false;
    for row in row_range {
        let start = row as usize * w;
        let end = start + w;
        if end > mask.len() {
            break;
        }
        let static_frac = mask[start..end].iter().filter(|&&v| v).count() as f64 / w as f64;
        if static_frac > 0.8 {
            band_end = row + 1;
            found = true;
        } else if found {
            break;
        }
    }

    if !found {
        return None;
    }

    match location {
        BandLocation::Top => Some(Region::new(0, 0, width, band_end)),
        BandLocation::Bottom => {
            let start = height.saturating_sub(max_band_height);
            Some(Region::new(0, start, width, band_end.saturating_sub(start)))
        }
    }
}

/// Find a static region in a corner.
#[allow(clippy::cast_precision_loss)]
fn find_static_corner(
    mask: &[bool],
    width: u32,
    height: u32,
    corner: CornerLocation,
    _config: &DetectionConfig,
) -> Option<Region> {
    let check_w = (width / 6).max(1);
    let check_h = (height / 6).max(1);
    let w = width as usize;

    let (start_x, start_y) = match corner {
        CornerLocation::TopLeft => (0_u32, 0_u32),
        CornerLocation::TopRight => (width - check_w, 0),
        CornerLocation::BottomLeft => (0, height - check_h),
        CornerLocation::BottomRight => (width - check_w, height - check_h),
    };

    let mut static_count = 0_u64;
    let total = u64::from(check_w) * u64::from(check_h);

    for row in start_y..(start_y + check_h) {
        for col in start_x..(start_x + check_w) {
            let idx = row as usize * w + col as usize;
            if idx < mask.len() && mask[idx] {
                static_count += 1;
            }
        }
    }

    let frac = static_count as f64 / total.max(1) as f64;
    if frac > 0.8 {
        Some(Region::new(start_x, start_y, check_w, check_h))
    } else {
        None
    }
}

/// Compute safe-area margins (title-safe and action-safe) for broadcast.
///
/// Returns `(title_safe, action_safe)` regions.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn compute_safe_areas(width: u32, height: u32) -> (Region, Region) {
    // Title safe: 10% inset on each side
    let title_inset_x = (f64::from(width) * 0.1).round() as u32;
    let title_inset_y = (f64::from(height) * 0.1).round() as u32;
    let title_safe = Region::new(
        title_inset_x,
        title_inset_y,
        width.saturating_sub(2 * title_inset_x),
        height.saturating_sub(2 * title_inset_y),
    );

    // Action safe: 5% inset on each side
    let action_inset_x = (f64::from(width) * 0.05).round() as u32;
    let action_inset_y = (f64::from(height) * 0.05).round() as u32;
    let action_safe = Region::new(
        action_inset_x,
        action_inset_y,
        width.saturating_sub(2 * action_inset_x),
        height.saturating_sub(2 * action_inset_y),
    );

    (title_safe, action_safe)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_area() {
        let r = Region::new(0, 0, 100, 50);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_region_contains() {
        let r = Region::new(10, 20, 100, 50);
        assert!(r.contains(10, 20));
        assert!(r.contains(50, 40));
        assert!(!r.contains(9, 20));
        assert!(!r.contains(110, 70));
    }

    #[test]
    fn test_region_overlaps() {
        let a = Region::new(0, 0, 100, 100);
        let b = Region::new(50, 50, 100, 100);
        assert!(a.overlaps(&b));
        let c = Region::new(200, 200, 10, 10);
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_region_intersection() {
        let a = Region::new(0, 0, 100, 100);
        let b = Region::new(50, 50, 100, 100);
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter.x, 50);
        assert_eq!(inter.y, 50);
        assert_eq!(inter.width, 50);
        assert_eq!(inter.height, 50);
    }

    #[test]
    fn test_region_no_intersection() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(20, 20, 10, 10);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_region_union_bbox() {
        let a = Region::new(10, 10, 50, 50);
        let b = Region::new(100, 100, 20, 20);
        let u = a.union_bbox(&b);
        assert_eq!(u.x, 10);
        assert_eq!(u.y, 10);
        assert_eq!(u.width, 110);
        assert_eq!(u.height, 110);
    }

    #[test]
    fn test_region_display() {
        let r = Region::new(5, 10, 200, 100);
        assert_eq!(format!("{r}"), "Region(x=5, y=10, 200x100)");
    }

    #[test]
    fn test_detect_static_overlays_empty() {
        let result = detect_static_overlays(&[], 0, 0, &DetectionConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_detect_static_overlays_all_static() {
        // All frames identical => everything is static
        let frame = vec![0.5_f32; 640 * 480];
        let frames = vec![frame.clone(), frame.clone(), frame];
        let config = DetectionConfig {
            min_confidence: 0.5,
            analysis_frames: 3,
            static_variance_threshold: 0.02,
            min_region_area: 10,
        };
        let overlays = detect_static_overlays(&frames, 640, 480, &config);
        // Should detect static bands at top and bottom, and corners
        assert!(!overlays.is_empty());
    }

    #[test]
    fn test_compute_safe_areas() {
        let (title, action) = compute_safe_areas(1920, 1080);
        assert!(title.x > 0);
        assert!(title.y > 0);
        assert!(title.width < 1920);
        assert!(title.height < 1080);
        assert!(action.width > title.width);
        assert!(action.height > title.height);
    }

    #[test]
    fn test_safe_area_ratios() {
        let (title, action) = compute_safe_areas(1920, 1080);
        // Title safe: ~80% of width (10% each side)
        assert!(title.width > 1500 && title.width < 1600);
        // Action safe: ~90% of width (5% each side)
        assert!(action.width > 1700 && action.width < 1850);
    }

    #[test]
    fn test_detection_config_defaults() {
        let config = DetectionConfig::default();
        assert!((config.min_confidence - 0.5).abs() < 1e-10);
        assert_eq!(config.analysis_frames, 30);
    }

    #[test]
    fn test_overlay_kind_debug() {
        let kind = OverlayKind::Watermark;
        assert_eq!(format!("{kind:?}"), "Watermark");
    }
}
