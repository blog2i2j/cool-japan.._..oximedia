#![allow(dead_code)]
//! Region-of-interest adaptive denoising.
//!
//! Applies different denoising strengths to different image regions. Useful
//! for preserving detail in faces or text areas while aggressively denoising
//! flat backgrounds.

/// A rectangular region of interest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    /// Left column (inclusive).
    pub x: u32,
    /// Top row (inclusive).
    pub y: u32,
    /// Region width in pixels.
    pub width: u32,
    /// Region height in pixels.
    pub height: u32,
}

impl Region {
    /// Create a new region.
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Right edge (exclusive).
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// Bottom edge (exclusive).
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// Area in pixels.
    pub fn area(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Check whether a point is inside this region.
    pub fn contains(&self, px: u32, py: u32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    /// Intersection with another region. Returns `None` if disjoint.
    pub fn intersect(&self, other: &Region) -> Option<Region> {
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x0 < x1 && y0 < y1 {
            Some(Region::new(x0, y0, x1 - x0, y1 - y0))
        } else {
            None
        }
    }
}

/// Priority for region processing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegionPriority {
    /// Low priority region (background).
    Low,
    /// Normal priority.
    Normal,
    /// High priority region (faces, text).
    High,
    /// Critical region (must preserve full detail).
    Critical,
}

impl Default for RegionPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Denoising parameters for a specific region.
#[derive(Clone, Debug)]
pub struct RegionParams {
    /// The region bounds.
    pub region: Region,
    /// Denoising strength override for this region (0.0 = none, 1.0 = max).
    pub strength: f32,
    /// Priority level.
    pub priority: RegionPriority,
    /// Label for identification.
    pub label: String,
}

impl RegionParams {
    /// Create parameters for a region.
    pub fn new(region: Region, strength: f32) -> Self {
        Self {
            region,
            strength: strength.clamp(0.0, 1.0),
            priority: RegionPriority::Normal,
            label: String::new(),
        }
    }

    /// Set the priority.
    pub fn with_priority(mut self, priority: RegionPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Set a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }
}

/// Configuration for region-based denoising.
#[derive(Clone, Debug)]
pub struct RegionDenoiseConfig {
    /// Default strength for areas not covered by any region.
    pub default_strength: f32,
    /// Frame width.
    pub frame_width: u32,
    /// Frame height.
    pub frame_height: u32,
    /// Feather radius for blending region boundaries (in pixels).
    pub feather_radius: u32,
    /// Regions with their parameters.
    pub regions: Vec<RegionParams>,
}

impl RegionDenoiseConfig {
    /// Create a new configuration.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            default_strength: 0.5,
            frame_width: width,
            frame_height: height,
            feather_radius: 8,
            regions: Vec::new(),
        }
    }

    /// Add a region.
    pub fn add_region(&mut self, params: RegionParams) {
        self.regions.push(params);
    }

    /// Number of defined regions.
    pub fn region_count(&self) -> usize {
        self.regions.len()
    }

    /// Sort regions by priority (highest first).
    pub fn sort_by_priority(&mut self) {
        self.regions.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
}

/// A per-pixel strength map for region-based denoising.
pub struct StrengthMap {
    /// Width of the map.
    pub width: u32,
    /// Height of the map.
    pub height: u32,
    /// Strength values in row-major order.
    data: Vec<f32>,
}

impl StrengthMap {
    /// Create a uniform strength map.
    #[allow(clippy::cast_precision_loss)]
    pub fn uniform(width: u32, height: u32, strength: f32) -> Self {
        let len = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data: vec![strength; len],
        }
    }

    /// Build a strength map from a region config.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_config(config: &RegionDenoiseConfig) -> Self {
        let w = config.frame_width;
        let h = config.frame_height;
        let len = (w as usize) * (h as usize);
        let mut data = vec![config.default_strength; len];

        // Sort regions by priority to let highest-priority paint last
        let mut sorted_regions: Vec<&RegionParams> = config.regions.iter().collect();
        sorted_regions.sort_by(|a, b| a.priority.cmp(&b.priority));

        for rp in &sorted_regions {
            let r = &rp.region;
            let x_end = r.right().min(w);
            let y_end = r.bottom().min(h);
            for py in r.y..y_end {
                for px in r.x..x_end {
                    let feather = Self::feather_weight(px, py, r, config.feather_radius);
                    let idx = (py as usize) * (w as usize) + (px as usize);
                    // Blend between default and region strength
                    data[idx] = data[idx] * (1.0 - feather) + rp.strength * feather;
                }
            }
        }

        Self {
            width: w,
            height: h,
            data,
        }
    }

    /// Get strength at a pixel coordinate.
    #[allow(clippy::cast_precision_loss)]
    pub fn get(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        self.data[(y as usize) * (self.width as usize) + (x as usize)]
    }

    /// Average strength across the map.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_strength(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().map(|&v| f64::from(v)).sum();
        (sum / self.data.len() as f64) as f32
    }

    /// Compute feather weight for a pixel relative to a region boundary.
    #[allow(clippy::cast_precision_loss)]
    fn feather_weight(px: u32, py: u32, region: &Region, radius: u32) -> f32 {
        if radius == 0 {
            return 1.0;
        }
        let dx_left = px.saturating_sub(region.x);
        let dx_right = region.right().saturating_sub(px + 1);
        let dy_top = py.saturating_sub(region.y);
        let dy_bottom = region.bottom().saturating_sub(py + 1);
        let min_dist = dx_left.min(dx_right).min(dy_top).min(dy_bottom);
        if min_dist >= radius {
            1.0
        } else {
            min_dist as f32 / radius as f32
        }
    }
}

/// Apply per-pixel strength to a flat luma buffer.
///
/// Each pixel is blended: `output = original * (1 - strength) + denoised * strength`.
#[allow(clippy::cast_precision_loss)]
pub fn apply_strength_map(original: &[f32], denoised: &[f32], map: &StrengthMap) -> Vec<f32> {
    original
        .iter()
        .zip(denoised.iter())
        .enumerate()
        .map(|(i, (&o, &d))| {
            let x = (i % map.width as usize) as u32;
            let y = (i / map.width as usize) as u32;
            let s = map.get(x, y);
            o * (1.0 - s) + d * s
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_new() {
        let r = Region::new(10, 20, 100, 50);
        assert_eq!(r.x, 10);
        assert_eq!(r.y, 20);
        assert_eq!(r.width, 100);
        assert_eq!(r.height, 50);
    }

    #[test]
    fn test_region_edges() {
        let r = Region::new(10, 20, 100, 50);
        assert_eq!(r.right(), 110);
        assert_eq!(r.bottom(), 70);
        assert_eq!(r.area(), 5000);
    }

    #[test]
    fn test_region_contains() {
        let r = Region::new(10, 10, 20, 20);
        assert!(r.contains(10, 10));
        assert!(r.contains(29, 29));
        assert!(!r.contains(30, 10));
        assert!(!r.contains(10, 30));
        assert!(!r.contains(9, 10));
    }

    #[test]
    fn test_region_intersect() {
        let a = Region::new(0, 0, 20, 20);
        let b = Region::new(10, 10, 20, 20);
        let c = a.intersect(&b).expect("c should be valid");
        assert_eq!(c, Region::new(10, 10, 10, 10));
    }

    #[test]
    fn test_region_intersect_disjoint() {
        let a = Region::new(0, 0, 10, 10);
        let b = Region::new(20, 20, 10, 10);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_region_priority_ordering() {
        assert!(RegionPriority::Low < RegionPriority::Normal);
        assert!(RegionPriority::Normal < RegionPriority::High);
        assert!(RegionPriority::High < RegionPriority::Critical);
    }

    #[test]
    fn test_region_params_builder() {
        let rp = RegionParams::new(Region::new(0, 0, 50, 50), 0.3)
            .with_priority(RegionPriority::High)
            .with_label("face");
        assert!((rp.strength - 0.3).abs() < f32::EPSILON);
        assert_eq!(rp.priority, RegionPriority::High);
        assert_eq!(rp.label, "face");
    }

    #[test]
    fn test_region_params_strength_clamp() {
        let rp = RegionParams::new(Region::new(0, 0, 10, 10), 2.0);
        assert!((rp.strength - 1.0).abs() < f32::EPSILON);
        let rp2 = RegionParams::new(Region::new(0, 0, 10, 10), -1.0);
        assert!((rp2.strength - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_add_region() {
        let mut cfg = RegionDenoiseConfig::new(100, 100);
        assert_eq!(cfg.region_count(), 0);
        cfg.add_region(RegionParams::new(Region::new(0, 0, 50, 50), 0.2));
        assert_eq!(cfg.region_count(), 1);
    }

    #[test]
    fn test_strength_map_uniform() {
        let map = StrengthMap::uniform(10, 10, 0.5);
        assert!((map.get(0, 0) - 0.5).abs() < f32::EPSILON);
        assert!((map.get(9, 9) - 0.5).abs() < f32::EPSILON);
        assert!((map.average_strength() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_strength_map_out_of_bounds() {
        let map = StrengthMap::uniform(10, 10, 0.7);
        assert!((map.get(10, 10) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_apply_strength_map() {
        let map = StrengthMap::uniform(4, 1, 0.5);
        let orig = vec![1.0, 1.0, 1.0, 1.0];
        let denoised = vec![0.0, 0.0, 0.0, 0.0];
        let result = apply_strength_map(&orig, &denoised, &map);
        for v in &result {
            assert!((*v - 0.5).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_strength_map_from_config() {
        let mut cfg = RegionDenoiseConfig::new(20, 20);
        cfg.default_strength = 0.8;
        cfg.feather_radius = 0;
        cfg.add_region(RegionParams::new(Region::new(5, 5, 10, 10), 0.1));
        let map = StrengthMap::from_config(&cfg);
        // Inside region (center) should be close to 0.1
        assert!((map.get(10, 10) - 0.1).abs() < f32::EPSILON);
        // Outside region should be default
        assert!((map.get(0, 0) - 0.8).abs() < f32::EPSILON);
    }
}
