//! Instance segmentation: connected-component labeling with per-object binary masks.
//!
//! This module provides a pure-Rust, CPU-based instance segmentation pipeline that
//! does not depend on any external neural network runtime.  The algorithm is:
//!
//! 1. **Threshold** the input image to separate foreground from background.
//! 2. **Label connected components** using a union-find (disjoint-set) data structure
//!    over the thresholded foreground pixels (8-connectivity).
//! 3. **Filter** components whose pixel count falls below `min_area_pixels`, or
//!    discard the smallest components until at most `max_instances` remain.
//! 4. **Emit** one [`SegmentMask`] per surviving component, with a full-image binary
//!    mask and an axis-aligned bounding box.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cv::instance_segmentation::{InstanceSegmenter, SegmentationConfig};
//!
//! let config = SegmentationConfig::default();
//! let segmenter = InstanceSegmenter::new(config);
//!
//! // 8x8 white square on a black background (RGBA, 4 bytes/px)
//! let mut image = vec![0u8; 16 * 16 * 4];
//! for row in 4..12usize {
//!     for col in 4..12usize {
//!         let idx = (row * 16 + col) * 4;
//!         image[idx] = 255;
//!         image[idx + 1] = 255;
//!         image[idx + 2] = 255;
//!         image[idx + 3] = 255;
//!     }
//! }
//! let masks = segmenter.segment(&image, 16, 16);
//! assert!(!masks.is_empty());
//! ```

use crate::error::{CvError, CvResult};

// ── Error ─────────────────────────────────────────────────────────────────────

/// Errors that can arise during instance segmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentationError {
    /// The image dimensions (width × height) are too small to segment.
    ImageTooSmall,
    /// The supplied width / height do not match the pixel buffer size.
    InvalidDimensions,
}

impl std::fmt::Display for SegmentationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ImageTooSmall => write!(f, "image is too small for segmentation"),
            Self::InvalidDimensions => write!(f, "pixel buffer size does not match width × height"),
        }
    }
}

impl std::error::Error for SegmentationError {}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration parameters for [`InstanceSegmenter`].
#[derive(Debug, Clone)]
pub struct SegmentationConfig {
    /// Minimum number of pixels a connected component must have to be returned.
    pub min_area_pixels: u32,
    /// Hard cap on the number of instances returned.
    ///
    /// When more components survive the area filter, only the largest
    /// `max_instances` are kept.
    pub max_instances: u32,
    /// Luminance threshold in `[0, 255]` for foreground/background separation.
    ///
    /// Pixels with intensity **above** this value are treated as foreground.
    pub threshold: u8,
}

impl Default for SegmentationConfig {
    fn default() -> Self {
        Self {
            min_area_pixels: 50,
            max_instances: 32,
            threshold: 127,
        }
    }
}

impl SegmentationConfig {
    /// Validate the configuration, returning an error on invalid values.
    pub fn validate(&self) -> CvResult<()> {
        if self.max_instances == 0 {
            return Err(CvError::invalid_parameter(
                "max_instances",
                "must be at least 1",
            ));
        }
        Ok(())
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

/// A single detected object instance with its binary pixel mask.
#[derive(Debug, Clone)]
pub struct SegmentMask {
    /// Unique instance identifier (1-based, assigned by component order).
    pub object_id: u32,
    /// Semantic class identifier.  This simple segmenter always sets `class_id = 1`
    /// (generic "foreground object"); callers can re-assign classes downstream.
    pub class_id: u32,
    /// Detection confidence in `[0.0, 1.0]`.
    ///
    /// This segmenter derives confidence from the mean foreground intensity
    /// normalised to `[0, 1]`.
    pub confidence: f32,
    /// Binary foreground mask, one byte per pixel, same row-major layout as the
    /// source image.  A value of `255` means foreground; `0` means background.
    ///
    /// Length == `width * height` of the **source** image (not the bounding box).
    pub mask: Vec<u8>,
    /// Axis-aligned bounding box `(x_min, y_min, x_max, y_max)` in pixel coords.
    pub bounding_box: (u32, u32, u32, u32),
}

impl SegmentMask {
    /// Area of the bounding box in pixels.
    #[must_use]
    pub fn bbox_area(&self) -> u64 {
        let (x0, y0, x1, y1) = self.bounding_box;
        let w = x1.saturating_sub(x0) as u64 + 1;
        let h = y1.saturating_sub(y0) as u64 + 1;
        w * h
    }

    /// Number of foreground pixels in the mask.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.mask.iter().filter(|&&b| b > 0).count()
    }
}

// ── Union-Find (Disjoint-Set) ─────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            // Path compression (halving)
            let grandparent = self.parent[self.parent[x]];
            self.parent[x] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] = self.rank[ra].saturating_add(1);
            }
        }
    }
}

// ── Segmenter ─────────────────────────────────────────────────────────────────

/// Instance segmenter based on connected-component labeling.
///
/// See the [module documentation](self) for an algorithm description and usage
/// example.
#[derive(Debug, Clone)]
pub struct InstanceSegmenter {
    config: SegmentationConfig,
}

impl InstanceSegmenter {
    /// Create a new segmenter with the given configuration.
    #[must_use]
    pub fn new(config: SegmentationConfig) -> Self {
        Self { config }
    }

    /// Create a segmenter with default configuration.
    #[must_use]
    pub fn default() -> Self {
        Self::new(SegmentationConfig::default())
    }

    /// Convert a raw pixel buffer to a grayscale luminance map.
    ///
    /// Accepts 1 (grayscale), 3 (RGB) or 4 (RGBA) bytes per pixel.
    /// Returns `None` if the bytes-per-pixel ratio is unrecognised.
    fn to_gray(image: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
        let n = (width as usize) * (height as usize);
        if image.is_empty() || n == 0 {
            return Some(vec![]);
        }
        let bpp = image.len() / n;
        match bpp {
            1 => Some(image.to_vec()),
            3 => Some(
                image
                    .chunks_exact(3)
                    .map(|c| {
                        let r = u32::from(c[0]);
                        let g = u32::from(c[1]);
                        let b = u32::from(c[2]);
                        // BT.601 luma approximation (integer arithmetic)
                        ((r * 299 + g * 587 + b * 114) / 1000) as u8
                    })
                    .collect(),
            ),
            4 => Some(
                image
                    .chunks_exact(4)
                    .map(|c| {
                        let r = u32::from(c[0]);
                        let g = u32::from(c[1]);
                        let b = u32::from(c[2]);
                        ((r * 299 + g * 587 + b * 114) / 1000) as u8
                    })
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Validate image dimensions against the pixel buffer.
    fn validate(image: &[u8], width: u32, height: u32) -> Result<(), SegmentationError> {
        if width == 0 || height == 0 {
            return Err(SegmentationError::ImageTooSmall);
        }
        let n = (width as usize) * (height as usize);
        if image.len() < n {
            return Err(SegmentationError::InvalidDimensions);
        }
        Ok(())
    }

    /// Run instance segmentation on `image`.
    ///
    /// `image` may be grayscale (1 byte/px), RGB (3 bytes/px), or RGBA (4
    /// bytes/px).  The returned masks are sorted by descending pixel count.
    ///
    /// Returns an empty `Vec` if no foreground components are found.
    ///
    /// Panics are not possible; errors are expressed via the return type.
    #[allow(clippy::cast_precision_loss)]
    pub fn segment(&self, image: &[u8], width: u32, height: u32) -> Vec<SegmentMask> {
        if Self::validate(image, width, height).is_err() {
            return Vec::new();
        }

        let gray = match Self::to_gray(image, width, height) {
            Some(g) => g,
            None => return Vec::new(),
        };

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // ── Step 1: threshold ─────────────────────────────────────────────────
        let foreground: Vec<bool> = gray.iter().map(|&p| p > self.config.threshold).collect();

        // ── Step 2: connected-component labeling (union-find, 8-connectivity) ─
        let mut uf = UnionFind::new(n);
        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !foreground[idx] {
                    continue;
                }
                // Check 8 neighbours with lower raster index to avoid
                // double-connecting
                let neighbours: &[(i64, i64)] = &[(-1, -1), (0, -1), (1, -1), (-1, 0)];
                for &(dx, dy) in neighbours {
                    let nx = x as i64 + dx;
                    let ny = y as i64 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                        continue;
                    }
                    let nidx = ny as usize * w + nx as usize;
                    if foreground[nidx] {
                        uf.union(idx, nidx);
                    }
                }
            }
        }

        // ── Step 3: gather component statistics ───────────────────────────────
        use std::collections::HashMap;

        // root → (pixel_count, sum_intensity, x_min, y_min, x_max, y_max)
        let mut components: HashMap<usize, (u32, u64, u32, u32, u32, u32)> = HashMap::new();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !foreground[idx] {
                    continue;
                }
                let root = uf.find(idx);
                let intensity = u64::from(gray[idx]);
                let entry = components
                    .entry(root)
                    .or_insert((0, 0, x as u32, y as u32, x as u32, y as u32));
                entry.0 += 1;
                entry.1 += intensity;
                if (x as u32) < entry.2 {
                    entry.2 = x as u32;
                }
                if (y as u32) < entry.3 {
                    entry.3 = y as u32;
                }
                if (x as u32) > entry.4 {
                    entry.4 = x as u32;
                }
                if (y as u32) > entry.5 {
                    entry.5 = y as u32;
                }
            }
        }

        // ── Step 4: filter by min_area ────────────────────────────────────────
        let min_area = self.config.min_area_pixels;
        let mut surviving: Vec<(usize, u32, u64, u32, u32, u32, u32)> = components
            .into_iter()
            .filter(|(_, (count, ..))| *count >= min_area)
            .map(|(root, (count, sum, x0, y0, x1, y1))| (root, count, sum, x0, y0, x1, y1))
            .collect();

        // Sort by pixel count descending for consistent ordering
        surviving.sort_by(|a, b| b.1.cmp(&a.1));

        // Enforce max_instances cap
        surviving.truncate(self.config.max_instances as usize);

        if surviving.is_empty() {
            return Vec::new();
        }

        // Build a set of accepted roots for O(1) lookup
        let accepted: std::collections::HashSet<usize> =
            surviving.iter().map(|(root, ..)| *root).collect();

        // ── Step 5: build per-instance full-image masks ───────────────────────
        // Map root → instance index in `surviving`
        let root_to_idx: HashMap<usize, usize> = surviving
            .iter()
            .enumerate()
            .map(|(i, (root, ..))| (*root, i))
            .collect();

        // Allocate masks
        let num_instances = surviving.len();
        let mut masks_data: Vec<Vec<u8>> = vec![vec![0u8; n]; num_instances];

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if !foreground[idx] {
                    continue;
                }
                let root = uf.find(idx);
                if !accepted.contains(&root) {
                    continue;
                }
                if let Some(&inst_idx) = root_to_idx.get(&root) {
                    masks_data[inst_idx][idx] = 255;
                }
            }
        }

        // ── Step 6: assemble SegmentMask structs ──────────────────────────────
        surviving
            .into_iter()
            .zip(masks_data.into_iter())
            .enumerate()
            .map(|(i, ((_, count, sum, x0, y0, x1, y1), mask))| {
                let confidence = if count > 0 {
                    (sum as f32 / (count as f32 * 255.0)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                SegmentMask {
                    object_id: (i as u32) + 1,
                    class_id: 1,
                    confidence,
                    mask,
                    bounding_box: (x0, y0, x1, y1),
                }
            })
            .collect()
    }

    /// Typed variant that validates dimensions and returns an error on failure.
    pub fn segment_checked(
        &self,
        image: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<SegmentMask>, SegmentationError> {
        Self::validate(image, width, height)?;
        Ok(self.segment(image, width, height))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ─────────────────────────────────────────────────────────────────

    /// Build a grayscale image of size `w x h` filled with `bg`, then paint a
    /// rectangle `[rx0, rx1) x [ry0, ry1)` with `fg`.
    fn make_image_with_rect(
        w: u32,
        h: u32,
        bg: u8,
        rx0: u32,
        ry0: u32,
        rx1: u32,
        ry1: u32,
        fg: u8,
    ) -> Vec<u8> {
        let mut img = vec![bg; (w * h) as usize];
        for y in ry0..ry1 {
            for x in rx0..rx1 {
                img[(y * w + x) as usize] = fg;
            }
        }
        img
    }

    fn default_segmenter() -> InstanceSegmenter {
        InstanceSegmenter::new(SegmentationConfig {
            min_area_pixels: 4,
            max_instances: 32,
            threshold: 127,
        })
    }

    // Tests ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_image_returns_empty() {
        let seg = default_segmenter();
        let result = seg.segment(&[], 0, 0);
        assert!(result.is_empty(), "expected empty for zero-size image");
    }

    #[test]
    fn test_all_black_returns_empty() {
        let seg = default_segmenter();
        let img = vec![0u8; 20 * 20];
        let result = seg.segment(&img, 20, 20);
        assert!(result.is_empty(), "no foreground pixels → no masks");
    }

    #[test]
    fn test_single_object_detected() {
        let seg = default_segmenter();
        // 20x20 black image with a 8x8 white square in the centre
        let img = make_image_with_rect(20, 20, 0, 6, 6, 14, 14, 255);
        let masks = seg.segment(&img, 20, 20);
        assert_eq!(masks.len(), 1, "exactly one object expected");
        let m = &masks[0];
        assert_eq!(m.object_id, 1);
        assert_eq!(m.class_id, 1);
        assert!(m.confidence > 0.0);
        // mask must have 20*20 bytes
        assert_eq!(m.mask.len(), 20 * 20);
        // foreground pixel count should be 8*8 = 64
        assert_eq!(m.pixel_count(), 64);
    }

    #[test]
    fn test_single_object_bounding_box() {
        let seg = default_segmenter();
        let img = make_image_with_rect(20, 20, 0, 5, 3, 10, 8, 255);
        let masks = seg.segment(&img, 20, 20);
        assert_eq!(masks.len(), 1);
        let (x0, y0, x1, y1) = masks[0].bounding_box;
        assert_eq!(x0, 5);
        assert_eq!(y0, 3);
        assert_eq!(x1, 9); // inclusive last pixel
        assert_eq!(y1, 7);
    }

    #[test]
    fn test_two_separated_objects() {
        let seg = default_segmenter();
        // Two 4x4 squares that don't touch
        let mut img = vec![0u8; 30 * 30];
        // Square A: columns 2-5, rows 2-5
        for y in 2..6usize {
            for x in 2..6usize {
                img[y * 30 + x] = 255;
            }
        }
        // Square B: columns 20-23, rows 20-23
        for y in 20..24usize {
            for x in 20..24usize {
                img[y * 30 + x] = 255;
            }
        }
        let masks = seg.segment(&img, 30, 30);
        assert_eq!(masks.len(), 2, "two objects expected");
        // object IDs must be 1 and 2
        let ids: Vec<u32> = masks.iter().map(|m| m.object_id).collect();
        assert!(ids.contains(&1) && ids.contains(&2));
    }

    #[test]
    fn test_max_instances_cap() {
        let seg = InstanceSegmenter::new(SegmentationConfig {
            min_area_pixels: 1,
            max_instances: 1,
            threshold: 127,
        });
        let mut img = vec![0u8; 30 * 30];
        // Two separated single pixels won't pass area=1 min_area but we use >=1
        for y in 0..4usize {
            for x in 0..4usize {
                img[y * 30 + x] = 255;
            }
        }
        for y in 20..24usize {
            for x in 20..24usize {
                img[y * 30 + x] = 255;
            }
        }
        let masks = seg.segment(&img, 30, 30);
        assert!(masks.len() <= 1, "max_instances cap not respected");
    }

    #[test]
    fn test_min_area_filter_removes_small_blobs() {
        let seg = InstanceSegmenter::new(SegmentationConfig {
            min_area_pixels: 100,
            max_instances: 32,
            threshold: 127,
        });
        // Small 4x4 blob — 16 pixels, below the 100 threshold
        let img = make_image_with_rect(20, 20, 0, 2, 2, 6, 6, 255);
        let masks = seg.segment(&img, 20, 20);
        assert!(masks.is_empty(), "blob should be filtered by min_area");
    }

    #[test]
    fn test_rgb_image_supported() {
        let seg = default_segmenter();
        // 10x10 RGB image, all white
        let img = vec![255u8; 10 * 10 * 3];
        let masks = seg.segment(&img, 10, 10);
        assert_eq!(masks.len(), 1);
    }

    #[test]
    fn test_rgba_image_supported() {
        let seg = default_segmenter();
        let img = vec![255u8; 10 * 10 * 4];
        let masks = seg.segment(&img, 10, 10);
        assert_eq!(masks.len(), 1);
    }

    #[test]
    fn test_checked_invalid_dimensions() {
        let seg = default_segmenter();
        let result = seg.segment_checked(&[], 0, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SegmentationError::ImageTooSmall);
    }

    #[test]
    fn test_checked_buffer_too_small() {
        let seg = default_segmenter();
        // buffer has only 10 bytes but 20x20 = 400 pixels expected
        let result = seg.segment_checked(&[0u8; 10], 20, 20);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SegmentationError::InvalidDimensions);
    }

    #[test]
    fn test_config_validate_max_instances_zero() {
        let cfg = SegmentationConfig {
            min_area_pixels: 10,
            max_instances: 0,
            threshold: 127,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_pixel_count_matches_segment_count() {
        let seg = default_segmenter();
        // 6x6 foreground square = 36 pixels
        let img = make_image_with_rect(20, 20, 0, 2, 2, 8, 8, 255);
        let masks = seg.segment(&img, 20, 20);
        assert!(!masks.is_empty());
        assert_eq!(masks[0].pixel_count(), 36);
    }

    #[test]
    fn test_mask_length_equals_image_size() {
        let seg = default_segmenter();
        let img = make_image_with_rect(15, 15, 0, 3, 3, 10, 10, 200);
        let masks = seg.segment(&img, 15, 15);
        if let Some(m) = masks.first() {
            assert_eq!(m.mask.len(), 15 * 15);
        }
    }
}
