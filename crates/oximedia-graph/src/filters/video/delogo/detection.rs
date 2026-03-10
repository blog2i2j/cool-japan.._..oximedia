//! Logo detection and analysis utilities for delogo filter.

#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use super::{color, Rectangle};

/// Analyze a region for logo characteristics.
#[derive(Debug, Clone)]
pub struct LogoCharacteristics {
    /// Average brightness.
    pub brightness: f32,
    /// Contrast level.
    pub contrast: f32,
    /// Edge density.
    pub edge_density: f32,
    /// Color variance.
    pub variance: f32,
    /// Opacity estimate (0.0 = transparent, 1.0 = opaque).
    pub opacity: f32,
}

impl LogoCharacteristics {
    /// Analyze a region.
    #[must_use]
    pub fn analyze(data: &[u8], region: &Rectangle, width: u32, height: u32) -> Self {
        let mut values = Vec::new();

        for y in region.y..region.bottom().min(height) {
            for x in region.x..region.right().min(width) {
                let idx = (y * width + x) as usize;
                if let Some(&val) = data.get(idx) {
                    values.push(val);
                }
            }
        }

        let brightness = color::mean(&values);
        let variance = color::std_dev(&values, brightness);
        let contrast = compute_contrast(&values);
        let edge_density = estimate_edge_density(data, region, width, height);
        let opacity = estimate_opacity(&values, brightness);

        Self {
            brightness,
            contrast,
            edge_density,
            variance,
            opacity,
        }
    }

    /// Check if the region likely contains a logo.
    #[must_use]
    pub fn is_likely_logo(&self) -> bool {
        // Logos typically have high contrast and edge density
        self.edge_density > 0.3 && self.contrast > 30.0
    }

    /// Check if the logo is semi-transparent.
    #[must_use]
    pub fn is_semitransparent(&self) -> bool {
        self.opacity < 0.8
    }
}

/// Compute contrast using difference between min and max.
fn compute_contrast(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }

    let min = *data.iter().min().unwrap_or(&0);
    let max = *data.iter().max().unwrap_or(&255);

    (max - min) as f32
}

/// Estimate edge density in a region.
fn estimate_edge_density(data: &[u8], region: &Rectangle, width: u32, height: u32) -> f32 {
    let mut edge_count = 0;
    let mut total = 0;

    for y in (region.y + 1)..(region.bottom() - 1).min(height - 1) {
        for x in (region.x + 1)..(region.right() - 1).min(width - 1) {
            let idx = (y * width + x) as usize;
            let val = data.get(idx).copied().unwrap_or(128);

            // Simple edge detection using neighbors
            let mut gradient = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }

                    let nx = (x as i32 + dx) as u32;
                    let ny = (y as i32 + dy) as u32;
                    let nidx = (ny * width + nx) as usize;
                    let nval = data.get(nidx).copied().unwrap_or(128);

                    gradient += (val as f32 - nval as f32).abs();
                }
            }

            if gradient > 200.0 {
                edge_count += 1;
            }
            total += 1;
        }
    }

    if total > 0 {
        edge_count as f32 / total as f32
    } else {
        0.0
    }
}

/// Estimate opacity based on pixel values.
fn estimate_opacity(data: &[u8], _mean: f32) -> f32 {
    if data.is_empty() {
        return 1.0;
    }

    // Count pixels close to extremes (fully opaque tends to be more extreme)
    let extreme_count = data.iter().filter(|&&v| !(50..=205).contains(&v)).count();

    let opacity = extreme_count as f32 / data.len() as f32;
    opacity.clamp(0.3, 1.0)
}

/// Non-maximum suppression for detected regions.
pub fn non_maximum_suppression(regions: &[Rectangle], iou_threshold: f32) -> Vec<Rectangle> {
    if regions.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<_> = regions.to_vec();
    sorted.sort_by_key(|r| r.area());
    sorted.reverse();

    let mut keep = Vec::new();

    for region in sorted {
        let mut should_keep = true;

        for kept in &keep {
            let iou = compute_iou(&region, kept);
            if iou > iou_threshold {
                should_keep = false;
                break;
            }
        }

        if should_keep {
            keep.push(region);
        }
    }

    keep
}

/// Compute intersection over union for two rectangles.
fn compute_iou(a: &Rectangle, b: &Rectangle) -> f32 {
    let intersect_x1 = a.x.max(b.x);
    let intersect_y1 = a.y.max(b.y);
    let intersect_x2 = a.right().min(b.right());
    let intersect_y2 = a.bottom().min(b.bottom());

    if intersect_x2 <= intersect_x1 || intersect_y2 <= intersect_y1 {
        return 0.0;
    }

    let intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1);
    let union_area = a.area() + b.area() - intersect_area;

    if union_area > 0 {
        intersect_area as f32 / union_area as f32
    } else {
        0.0
    }
}
