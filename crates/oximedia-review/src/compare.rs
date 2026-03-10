//! Side-by-side and A/B visual comparison for media review.

use std::collections::HashMap;

/// Comparison layout mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareLayout {
    /// Two images side by side (left/right split).
    SideBySide,
    /// Image A on top of Image B (top/bottom split).
    TopBottom,
    /// Wipe transition between A and B.
    Wipe {
        /// Position of the wipe (0.0 - 1.0).
        position: f32,
        /// Angle of the wipe.
        angle: WipeAngle,
    },
    /// Onion-skin overlay with alpha blend.
    Overlay {
        /// Alpha blend factor (0.0 = all A, 1.0 = all B).
        alpha: f32,
    },
    /// Difference blend (absolute pixel difference).
    Difference,
    /// Split view with interactive cursor divider.
    InteractiveSplit {
        /// Horizontal split position (0.0 - 1.0).
        x_position: f32,
    },
}

/// Angle for a wipe transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WipeAngle {
    /// Horizontal wipe (left-right).
    Horizontal,
    /// Vertical wipe (top-bottom).
    Vertical,
    /// Diagonal wipe at 45 degrees.
    Diagonal45,
    /// Diagonal wipe at 135 degrees.
    Diagonal135,
}

/// An image version for comparison.
#[derive(Debug, Clone)]
pub struct CompareVersion {
    /// Unique identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// RGBA frame data.
    pub frame_data: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Additional metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl CompareVersion {
    /// Create a new compare version with the given dimensions.
    #[must_use]
    pub fn new(id: impl Into<String>, label: impl Into<String>, width: u32, height: u32) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            frame_data: Vec::new(),
            width,
            height,
            metadata: HashMap::new(),
        }
    }

    /// Attach RGBA frame data to this version.
    #[must_use]
    pub fn with_frame_data(mut self, data: Vec<u8>) -> Self {
        self.frame_data = data;
        self
    }

    /// Add a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: &str, val: &str) -> Self {
        self.metadata.insert(key.to_string(), val.to_string());
        self
    }

    /// Total number of pixels (width * height).
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Result of a comparison operation.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// Layout used for this comparison.
    pub layout: CompareLayout,
    /// RGBA composite output.
    pub output_data: Vec<u8>,
    /// Width of composite output.
    pub width: u32,
    /// Height of composite output.
    pub height: u32,
    /// Optional pixel-difference statistics.
    pub diff_stats: Option<DiffStats>,
}

/// Per-pixel difference statistics between two versions.
#[derive(Debug, Clone)]
pub struct DiffStats {
    /// Mean absolute error across all channels and pixels.
    pub mean_absolute_error: f64,
    /// Maximum single-channel difference value.
    pub max_difference: u8,
    /// Fraction of pixels that differ (0.0 - 1.0).
    pub changed_pixel_ratio: f64,
    /// True when the two inputs are byte-identical.
    pub identical: bool,
}

impl DiffStats {
    /// Compute difference statistics between two RGBA byte slices.
    ///
    /// Both slices must have the same length.
    #[must_use]
    pub fn compute(a: &[u8], b: &[u8]) -> Self {
        if a.is_empty() || a.len() != b.len() {
            return Self {
                mean_absolute_error: 0.0,
                max_difference: 0,
                changed_pixel_ratio: 0.0,
                identical: a == b,
            };
        }

        let mut total_diff: u64 = 0;
        let mut max_diff: u8 = 0;
        let mut changed_pixels: u64 = 0;
        let pixel_count = a.len() / 4;

        for chunk in a.chunks(4).zip(b.chunks(4)) {
            let (pa, pb) = chunk;
            let mut pixel_changed = false;
            for i in 0..4 {
                let d = pa[i].abs_diff(pb[i]);
                total_diff += u64::from(d);
                if d > max_diff {
                    max_diff = d;
                }
                if d > 0 {
                    pixel_changed = true;
                }
            }
            if pixel_changed {
                changed_pixels += 1;
            }
        }

        let total_samples = a.len() as f64;
        let mae = total_diff as f64 / total_samples;
        let changed_ratio = if pixel_count > 0 {
            changed_pixels as f64 / pixel_count as f64
        } else {
            0.0
        };

        Self {
            mean_absolute_error: mae,
            max_difference: max_diff,
            changed_pixel_ratio: changed_ratio,
            identical: max_diff == 0,
        }
    }
}

/// Visual comparator for media review.
pub struct MediaComparator {
    versions: Vec<CompareVersion>,
    layout: CompareLayout,
}

impl MediaComparator {
    /// Create a new comparator with default side-by-side layout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
            layout: CompareLayout::SideBySide,
        }
    }

    /// Set the comparison layout.
    #[must_use]
    pub fn with_layout(mut self, layout: CompareLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Add a version to the comparator.
    pub fn add_version(&mut self, version: CompareVersion) -> &mut Self {
        self.versions.push(version);
        self
    }

    /// Number of versions currently registered.
    #[must_use]
    pub fn version_count(&self) -> usize {
        self.versions.len()
    }

    /// Look up a version by its ID.
    #[must_use]
    pub fn get_version(&self, id: &str) -> Option<&CompareVersion> {
        self.versions.iter().find(|v| v.id == id)
    }

    /// Remove a version by its ID.  Returns `true` if a version was removed.
    pub fn remove_version(&mut self, id: &str) -> bool {
        let before = self.versions.len();
        self.versions.retain(|v| v.id != id);
        self.versions.len() < before
    }

    /// Compare version A against version B using the configured layout.
    ///
    /// # Errors
    ///
    /// Returns an error string if either version ID is not found.
    pub fn compare(&self, id_a: &str, id_b: &str) -> Result<CompareResult, String> {
        let a = self
            .get_version(id_a)
            .ok_or_else(|| format!("version '{id_a}' not found"))?;
        let b = self
            .get_version(id_b)
            .ok_or_else(|| format!("version '{id_b}' not found"))?;

        let out_w = a.width.max(b.width);
        let out_h = a.height.max(b.height);

        let (output_data, diff_stats) = match self.layout {
            CompareLayout::SideBySide => {
                let data = compose_side_by_side(a, b, out_w, out_h);
                (data, None)
            }
            CompareLayout::TopBottom => {
                let data = compose_top_bottom(a, b, out_w, out_h);
                (data, None)
            }
            CompareLayout::Wipe { position, angle } => {
                let data = compose_wipe(a, b, position, angle, out_w, out_h);
                (data, None)
            }
            CompareLayout::Overlay { alpha } => {
                let data = compose_overlay(a, b, alpha, out_w, out_h);
                (data, None)
            }
            CompareLayout::Difference => {
                let data = compose_difference(a, b, out_w, out_h);
                let stats = DiffStats::compute(&a.frame_data, &b.frame_data);
                (data, Some(stats))
            }
            CompareLayout::InteractiveSplit { x_position } => {
                let data = compose_wipe(a, b, x_position, WipeAngle::Horizontal, out_w, out_h);
                (data, None)
            }
        };

        Ok(CompareResult {
            layout: self.layout,
            output_data,
            width: out_w,
            height: out_h,
            diff_stats,
        })
    }

    /// Compute pixel difference statistics between two versions.
    #[must_use]
    pub fn diff(&self, id_a: &str, id_b: &str) -> Option<DiffStats> {
        let a = self.get_version(id_a)?;
        let b = self.get_version(id_b)?;
        Some(DiffStats::compute(&a.frame_data, &b.frame_data))
    }
}

impl Default for MediaComparator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Composition helpers
// ---------------------------------------------------------------------------

/// Sample a pixel from a version at (x, y), returning [0,0,0,255] for OOB.
fn sample_pixel(v: &CompareVersion, x: u32, y: u32) -> [u8; 4] {
    if x >= v.width || y >= v.height || v.frame_data.is_empty() {
        return [0, 0, 0, 255];
    }
    let idx = ((y * v.width + x) as usize) * 4;
    if idx + 3 >= v.frame_data.len() {
        return [0, 0, 0, 255];
    }
    [
        v.frame_data[idx],
        v.frame_data[idx + 1],
        v.frame_data[idx + 2],
        v.frame_data[idx + 3],
    ]
}

/// Side-by-side composition: left half from A, right half from B.
#[allow(dead_code)]
fn compose_side_by_side(
    a: &CompareVersion,
    b: &CompareVersion,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let mid = width / 2;
    for y in 0..height {
        for x in 0..width {
            let pixel = if x < mid {
                sample_pixel(a, x, y)
            } else {
                sample_pixel(b, x - mid, y)
            };
            let idx = ((y * width + x) as usize) * 4;
            out[idx..idx + 4].copy_from_slice(&pixel);
        }
    }
    out
}

/// Top-bottom composition: top half from A, bottom half from B.
fn compose_top_bottom(a: &CompareVersion, b: &CompareVersion, width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    let mid = height / 2;
    for y in 0..height {
        for x in 0..width {
            let pixel = if y < mid {
                sample_pixel(a, x, y)
            } else {
                sample_pixel(b, x, y - mid)
            };
            let idx = ((y * width + x) as usize) * 4;
            out[idx..idx + 4].copy_from_slice(&pixel);
        }
    }
    out
}

/// Wipe composition: pixels on the A-side of the wipe boundary come from A.
#[allow(dead_code)]
fn compose_wipe(
    a: &CompareVersion,
    b: &CompareVersion,
    position: f32,
    angle: WipeAngle,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let nx = x as f32 / width as f32;
            let ny = y as f32 / height as f32;
            let use_a = match angle {
                WipeAngle::Horizontal => nx < position,
                WipeAngle::Vertical => ny < position,
                WipeAngle::Diagonal45 => (nx + ny) / 2.0 < position,
                WipeAngle::Diagonal135 => (nx + (1.0 - ny)) / 2.0 < position,
            };
            let pixel = if use_a {
                sample_pixel(a, x, y)
            } else {
                sample_pixel(b, x, y)
            };
            let idx = ((y * width + x) as usize) * 4;
            out[idx..idx + 4].copy_from_slice(&pixel);
        }
    }
    out
}

/// Overlay composition: `output = A * (1-alpha) + B * alpha` per channel.
#[allow(dead_code)]
fn compose_overlay(
    a: &CompareVersion,
    b: &CompareVersion,
    alpha: f32,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let alpha = alpha.clamp(0.0, 1.0);
    let inv = 1.0 - alpha;
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let pa = sample_pixel(a, x, y);
            let pb = sample_pixel(b, x, y);
            let idx = ((y * width + x) as usize) * 4;
            for i in 0..4 {
                out[idx + i] = (pa[i] as f32 * inv + pb[i] as f32 * alpha).round() as u8;
            }
        }
    }
    out
}

/// Difference composition: `output = |A - B|` per channel.
#[allow(dead_code)]
fn compose_difference(a: &CompareVersion, b: &CompareVersion, width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let pa = sample_pixel(a, x, y);
            let pb = sample_pixel(b, x, y);
            let idx = ((y * width + x) as usize) * 4;
            for i in 0..4 {
                out[idx + i] = pa[i].abs_diff(pb[i]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Comparison filters
// ---------------------------------------------------------------------------

/// Filter mode for comparison output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareFilter {
    /// No filtering (identity).
    None,
    /// Amplify differences by a gain factor.
    DifferenceAmplify {
        /// Gain applied to each channel difference (1.0 = normal).
        gain: f32,
    },
    /// Threshold: pixels whose absolute difference exceeds the threshold
    /// are white, otherwise black.
    Threshold {
        /// Per-channel threshold (0-255).
        threshold: u8,
    },
    /// Heatmap: maps the per-pixel mean absolute error to a colour gradient
    /// (blue = no change, red = large change).
    Heatmap,
    /// Channel isolate: show difference in a single colour channel only.
    ChannelIsolate {
        /// 0=R, 1=G, 2=B.
        channel: u8,
    },
}

/// Apply a comparison filter to two RGBA images of the same dimensions.
///
/// Returns a new RGBA buffer of size `width * height * 4`.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::cast_sign_loss)]
pub fn apply_compare_filter(
    a: &CompareVersion,
    b: &CompareVersion,
    filter: CompareFilter,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let n = (width * height * 4) as usize;
    let mut out = vec![0u8; n];

    match filter {
        CompareFilter::None => {
            // Just copy version A
            for y in 0..height {
                for x in 0..width {
                    let pa = sample_pixel(a, x, y);
                    let idx = ((y * width + x) as usize) * 4;
                    out[idx..idx + 4].copy_from_slice(&pa);
                }
            }
        }
        CompareFilter::DifferenceAmplify { gain } => {
            let g = gain.clamp(0.0, 255.0);
            for y in 0..height {
                for x in 0..width {
                    let pa = sample_pixel(a, x, y);
                    let pb = sample_pixel(b, x, y);
                    let idx = ((y * width + x) as usize) * 4;
                    for i in 0..3 {
                        let diff = pa[i].abs_diff(pb[i]) as f32 * g;
                        out[idx + i] = (diff.round() as u32).min(255) as u8;
                    }
                    out[idx + 3] = 255;
                }
            }
        }
        CompareFilter::Threshold { threshold } => {
            for y in 0..height {
                for x in 0..width {
                    let pa = sample_pixel(a, x, y);
                    let pb = sample_pixel(b, x, y);
                    let idx = ((y * width + x) as usize) * 4;
                    let changed = (0..3).any(|i| pa[i].abs_diff(pb[i]) > threshold);
                    let val = if changed { 255u8 } else { 0u8 };
                    out[idx] = val;
                    out[idx + 1] = val;
                    out[idx + 2] = val;
                    out[idx + 3] = 255;
                }
            }
        }
        CompareFilter::Heatmap => {
            for y in 0..height {
                for x in 0..width {
                    let pa = sample_pixel(a, x, y);
                    let pb = sample_pixel(b, x, y);
                    let idx = ((y * width + x) as usize) * 4;
                    // Mean absolute error across RGB
                    let mae: f32 = (0..3).map(|i| pa[i].abs_diff(pb[i]) as f32).sum::<f32>() / 3.0;
                    let t = (mae / 255.0).clamp(0.0, 1.0);
                    // Blue (cold) -> Red (hot) gradient
                    out[idx] = (t * 255.0).round() as u8; // R
                    out[idx + 1] = 0; // G
                    out[idx + 2] = ((1.0 - t) * 255.0).round() as u8; // B
                    out[idx + 3] = 255;
                }
            }
        }
        CompareFilter::ChannelIsolate { channel } => {
            let ch = (channel as usize).min(2);
            for y in 0..height {
                for x in 0..width {
                    let pa = sample_pixel(a, x, y);
                    let pb = sample_pixel(b, x, y);
                    let idx = ((y * width + x) as usize) * 4;
                    let diff = pa[ch].abs_diff(pb[ch]);
                    out[idx] = 0;
                    out[idx + 1] = 0;
                    out[idx + 2] = 0;
                    out[idx + ch] = diff;
                    out[idx + 3] = 255;
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(width: u32, height: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (width * height * 4) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..(width * height) {
            v.extend_from_slice(&[r, g, b, a]);
        }
        v
    }

    #[test]
    fn test_compare_version_new() {
        let v = CompareVersion::new("v1", "Version 1", 1920, 1080);
        assert_eq!(v.id, "v1");
        assert_eq!(v.label, "Version 1");
        assert_eq!(v.width, 1920);
        assert_eq!(v.height, 1080);
        assert!(v.frame_data.is_empty());
        assert!(v.metadata.is_empty());
    }

    #[test]
    fn test_diff_stats_identical() {
        let data = solid_rgba(4, 4, 128, 64, 32, 255);
        let stats = DiffStats::compute(&data, &data);
        assert!(stats.identical);
        assert_eq!(stats.mean_absolute_error, 0.0);
        assert_eq!(stats.max_difference, 0);
        assert_eq!(stats.changed_pixel_ratio, 0.0);
    }

    #[test]
    fn test_diff_stats_different() {
        // A = all zeros, B = all 10 → MAE = 10.0
        let a = vec![0u8; 16]; // 4 pixels × 4 channels
        let b = vec![10u8; 16];
        let stats = DiffStats::compute(&a, &b);
        assert!(!stats.identical);
        assert!((stats.mean_absolute_error - 10.0).abs() < 1e-9);
        assert_eq!(stats.max_difference, 10);
        assert!((stats.changed_pixel_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compose_side_by_side_size() {
        let a = CompareVersion::new("a", "A", 64, 32)
            .with_frame_data(solid_rgba(64, 32, 255, 0, 0, 255));
        let b = CompareVersion::new("b", "B", 64, 32)
            .with_frame_data(solid_rgba(64, 32, 0, 255, 0, 255));
        let out = compose_side_by_side(&a, &b, 64, 32);
        assert_eq!(out.len(), (64 * 32 * 4) as usize);
    }

    #[test]
    fn test_compose_overlay_alpha_zero() {
        let w = 4u32;
        let h = 4u32;
        let a =
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 100, 0, 0, 255));
        let b =
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 200, 0, 0, 255));
        let out = compose_overlay(&a, &b, 0.0, w, h);
        // alpha=0.0 → entirely A
        assert_eq!(out[0], 100);
    }

    #[test]
    fn test_compose_overlay_alpha_one() {
        let w = 4u32;
        let h = 4u32;
        let a =
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 100, 0, 0, 255));
        let b =
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 200, 0, 0, 255));
        let out = compose_overlay(&a, &b, 1.0, w, h);
        // alpha=1.0 → entirely B
        assert_eq!(out[0], 200);
    }

    #[test]
    fn test_compose_difference_identical() {
        let w = 4u32;
        let h = 4u32;
        let data = solid_rgba(w, h, 128, 64, 32, 255);
        let v = CompareVersion::new("v", "V", w, h).with_frame_data(data);
        let out = compose_difference(&v, &v, w, h);
        assert!(out.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_comparator_add_remove() {
        let mut cmp = MediaComparator::new();
        assert_eq!(cmp.version_count(), 0);
        cmp.add_version(CompareVersion::new("a", "A", 10, 10));
        cmp.add_version(CompareVersion::new("b", "B", 10, 10));
        assert_eq!(cmp.version_count(), 2);
        assert!(cmp.remove_version("a"));
        assert_eq!(cmp.version_count(), 1);
        assert!(!cmp.remove_version("nonexistent"));
    }

    #[test]
    fn test_comparator_compare_side_by_side() {
        let mut cmp = MediaComparator::new();
        let w = 8u32;
        let h = 8u32;
        cmp.add_version(
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 255, 0, 0, 255)),
        );
        cmp.add_version(
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 0, 255, 0, 255)),
        );
        let result = cmp.compare("a", "b").expect("compare should succeed");
        assert_eq!(result.width, w);
        assert_eq!(result.height, h);
        assert_eq!(result.output_data.len(), (w * h * 4) as usize);
    }

    #[test]
    fn test_compare_filter_threshold() {
        let w = 4u32;
        let h = 4u32;
        let a =
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 100, 0, 0, 255));
        let b =
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 200, 0, 0, 255));
        let out = apply_compare_filter(&a, &b, CompareFilter::Threshold { threshold: 50 }, w, h);
        // Difference is 100 which exceeds threshold of 50, so all pixels white
        assert_eq!(out[0], 255);
        assert_eq!(out[1], 255);
    }

    #[test]
    fn test_compare_filter_heatmap() {
        let w = 4u32;
        let h = 4u32;
        let a = CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 0, 0, 0, 255));
        let b = CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 0, 0, 0, 255));
        let out = apply_compare_filter(&a, &b, CompareFilter::Heatmap, w, h);
        // Identical: R=0 (cold), B=255
        assert_eq!(out[0], 0); // R
        assert_eq!(out[2], 255); // B
    }

    #[test]
    fn test_compare_filter_amplify() {
        let w = 4u32;
        let h = 4u32;
        let a =
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 100, 0, 0, 255));
        let b =
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 110, 0, 0, 255));
        let out = apply_compare_filter(
            &a,
            &b,
            CompareFilter::DifferenceAmplify { gain: 10.0 },
            w,
            h,
        );
        // Difference is 10 * gain 10 = 100
        assert_eq!(out[0], 100);
    }

    #[test]
    fn test_wipe_angle_types() {
        let w = 8u32;
        let h = 8u32;
        let a =
            CompareVersion::new("a", "A", w, h).with_frame_data(solid_rgba(w, h, 255, 0, 0, 255));
        let b =
            CompareVersion::new("b", "B", w, h).with_frame_data(solid_rgba(w, h, 0, 255, 0, 255));
        for angle in [
            WipeAngle::Horizontal,
            WipeAngle::Vertical,
            WipeAngle::Diagonal45,
            WipeAngle::Diagonal135,
        ] {
            let out = compose_wipe(&a, &b, 0.5, angle, w, h);
            assert_eq!(out.len(), (w * h * 4) as usize);
        }
    }
}
