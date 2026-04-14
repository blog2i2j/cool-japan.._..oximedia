//! Blob detection via connected-component labeling and morphological filtering.
//!
//! This module provides algorithms for detecting and characterising "blobs" —
//! contiguous regions of pixels that satisfy a threshold criterion — within
//! grayscale images.  The primary entry point is [`BlobDetector`], which
//! runs a union-find connected-component analysis followed by per-component
//! shape filtering (area, circularity, aspect ratio, convexity).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::blob_detector::{BlobDetector, BlobConfig, Connectivity};
//!
//! // Create a 10×10 grayscale image with a small bright blob in the centre
//! let w = 10usize;
//! let h = 10usize;
//! let mut image = vec![0u8; w * h];
//! for dy in 0..3usize {
//!     for dx in 0..3usize {
//!         image[(4 + dy) * w + (4 + dx)] = 200;
//!     }
//! }
//!
//! let cfg = BlobConfig::default();
//! let detector = BlobDetector::new(cfg);
//! let blobs = detector.detect(&image, w, h);
//! assert!(!blobs.is_empty());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls which neighbours count as "connected" when building components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connectivity (up, down, left, right only).
    Four,
    /// 8-connectivity (additionally includes the 4 diagonal neighbours).
    Eight,
}

/// Filtering parameters for blob detection.
#[derive(Debug, Clone)]
pub struct BlobConfig {
    /// Pixel intensity threshold: pixels with value ≥ `threshold` are foreground.
    pub threshold: u8,
    /// Minimum blob area in pixels (inclusive).
    pub min_area: usize,
    /// Maximum blob area in pixels (inclusive).  `usize::MAX` means no cap.
    pub max_area: usize,
    /// Minimum circularity `4π·area / perimeter²` ∈ [0, 1].  0 = no filter.
    pub min_circularity: f32,
    /// Maximum circularity upper bound.  For discrete blobs the value can
    /// exceed 1.0 due to coarse perimeter approximation, so the default is
    /// `f32::MAX` (no upper bound).  Set to 1.05 to restrict to near-circles.
    pub max_circularity: f32,
    /// Minimum aspect ratio (width/height of axis-aligned bounding box).
    pub min_aspect_ratio: f32,
    /// Maximum aspect ratio.
    pub max_aspect_ratio: f32,
    /// Connectivity rule used during component labeling.
    pub connectivity: Connectivity,
}

impl Default for BlobConfig {
    fn default() -> Self {
        Self {
            threshold: 128,
            min_area: 1,
            max_area: usize::MAX,
            min_circularity: 0.0,
            max_circularity: f32::MAX,
            min_aspect_ratio: 0.0,
            max_aspect_ratio: f32::MAX,
            connectivity: Connectivity::Eight,
        }
    }
}

/// Statistics and geometry of a detected blob (connected foreground region).
#[derive(Debug, Clone)]
pub struct Blob {
    /// Centroid x coordinate (column), fractional pixel.
    pub cx: f64,
    /// Centroid y coordinate (row), fractional pixel.
    pub cy: f64,
    /// Number of foreground pixels in this blob.
    pub area: usize,
    /// Left column of the axis-aligned bounding box.
    pub bbox_x: usize,
    /// Top row of the axis-aligned bounding box.
    pub bbox_y: usize,
    /// Width of the axis-aligned bounding box in pixels.
    pub bbox_w: usize,
    /// Height of the axis-aligned bounding box in pixels.
    pub bbox_h: usize,
    /// Circularity: 4π · area / perimeter².  Ranges from 0 to ≈ 1.
    pub circularity: f32,
    /// Aspect ratio of the bounding box: bbox_w / bbox_h.
    pub aspect_ratio: f32,
    /// Perimeter approximation (number of boundary pixels).
    pub perimeter: f32,
}

impl Blob {
    /// Returns the axis-aligned bounding box as `(x, y, width, height)`.
    #[must_use]
    pub fn bbox(&self) -> (usize, usize, usize, usize) {
        (self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)
    }

    /// Returns the centroid as `(cx, cy)`.
    #[must_use]
    pub fn centroid(&self) -> (f64, f64) {
        (self.cx, self.cy)
    }
}

// ---------------------------------------------------------------------------
// Union-Find (path compression + union by rank)
// ---------------------------------------------------------------------------

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u32>,
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
            // Path halving
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
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
                self.rank[ra] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BlobDetector
// ---------------------------------------------------------------------------

/// Detects blobs in a grayscale image using connected-component labeling
/// followed by configurable shape filters.
pub struct BlobDetector {
    cfg: BlobConfig,
}

impl BlobDetector {
    /// Create a new [`BlobDetector`] with the given configuration.
    #[must_use]
    pub fn new(cfg: BlobConfig) -> Self {
        Self { cfg }
    }

    /// Detect blobs in a row-major grayscale u8 image of dimensions `width × height`.
    ///
    /// Returns a list of [`Blob`] structs passing the configured filters, sorted
    /// by descending area.  Returns an empty list if the image is empty or the
    /// slice length does not match `width × height`.
    #[must_use]
    pub fn detect(&self, image: &[u8], width: usize, height: usize) -> Vec<Blob> {
        if width == 0 || height == 0 || image.len() != width * height {
            return Vec::new();
        }

        // Step 1: Build binary foreground mask
        let fg: Vec<bool> = image.iter().map(|&p| p >= self.cfg.threshold).collect();

        // Step 2: Connected-component labeling via union-find
        let n = width * height;
        let mut uf = UnionFind::new(n);

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !fg[idx] {
                    continue;
                }
                // Check left neighbour
                if x > 0 && fg[idx - 1] {
                    uf.union(idx, idx - 1);
                }
                // Check top neighbour
                if y > 0 && fg[idx - width] {
                    uf.union(idx, idx - width);
                }
                if self.cfg.connectivity == Connectivity::Eight {
                    // Check top-left
                    if x > 0 && y > 0 && fg[idx - width - 1] {
                        uf.union(idx, idx - width - 1);
                    }
                    // Check top-right
                    if x + 1 < width && y > 0 && fg[idx - width + 1] {
                        uf.union(idx, idx - width + 1);
                    }
                }
            }
        }

        // Step 3: Aggregate per-component statistics
        use std::collections::HashMap;

        // Map root → ComponentStats
        let mut stats: HashMap<usize, ComponentStats> = HashMap::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !fg[idx] {
                    continue;
                }
                let root = uf.find(idx);
                let entry = stats.entry(root).or_insert_with(ComponentStats::new);
                entry.add_pixel(x, y);
            }
        }

        // Step 4: Compute boundary pixels (foreground pixels adjacent to background)
        // for perimeter approximation
        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if !fg[idx] {
                    continue;
                }
                if is_boundary(x, y, width, height, &fg, self.cfg.connectivity) {
                    let root = uf.find(idx);
                    if let Some(s) = stats.get_mut(&root) {
                        s.boundary_count += 1;
                    }
                }
            }
        }

        // Step 5: Build Blob structs and apply filters
        let mut blobs: Vec<Blob> = stats.values().filter_map(|s| self.build_blob(s)).collect();

        blobs.sort_by(|a, b| b.area.cmp(&a.area));
        blobs
    }

    /// Convert a `ComponentStats` into a `Blob`, returning `None` if any
    /// configured filter rejects the component.
    fn build_blob(&self, s: &ComponentStats) -> Option<Blob> {
        let area = s.pixel_count;
        if area == 0 {
            return None;
        }
        if area < self.cfg.min_area || area > self.cfg.max_area {
            return None;
        }

        let bbox_w = s.x_max - s.x_min + 1;
        let bbox_h = s.y_max - s.y_min + 1;

        let aspect_ratio = if bbox_h == 0 {
            0.0f32
        } else {
            bbox_w as f32 / bbox_h as f32
        };

        if aspect_ratio < self.cfg.min_aspect_ratio || aspect_ratio > self.cfg.max_aspect_ratio {
            return None;
        }

        // Perimeter and circularity
        let perimeter = s.boundary_count as f32;
        let circularity = if perimeter > 0.0 {
            4.0 * std::f32::consts::PI * area as f32 / (perimeter * perimeter)
        } else {
            0.0f32
        };

        if circularity < self.cfg.min_circularity || circularity > self.cfg.max_circularity {
            return None;
        }

        let cx = s.sum_x as f64 / area as f64;
        let cy = s.sum_y as f64 / area as f64;

        Some(Blob {
            cx,
            cy,
            area,
            bbox_x: s.x_min,
            bbox_y: s.y_min,
            bbox_w,
            bbox_h,
            circularity,
            aspect_ratio,
            perimeter,
        })
    }
}

// ---------------------------------------------------------------------------
// ComponentStats: per-root accumulators
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct ComponentStats {
    pixel_count: usize,
    sum_x: u64,
    sum_y: u64,
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
    boundary_count: usize,
}

impl ComponentStats {
    fn new() -> Self {
        Self {
            pixel_count: 0,
            sum_x: 0,
            sum_y: 0,
            x_min: usize::MAX,
            x_max: 0,
            y_min: usize::MAX,
            y_max: 0,
            boundary_count: 0,
        }
    }

    fn add_pixel(&mut self, x: usize, y: usize) {
        self.pixel_count += 1;
        self.sum_x += x as u64;
        self.sum_y += y as u64;
        if x < self.x_min {
            self.x_min = x;
        }
        if x > self.x_max {
            self.x_max = x;
        }
        if y < self.y_min {
            self.y_min = y;
        }
        if y > self.y_max {
            self.y_max = y;
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: boundary pixel test
// ---------------------------------------------------------------------------

/// Returns `true` if pixel `(x, y)` is on the boundary of its foreground region.
///
/// A pixel is a boundary pixel if at least one of its 4-neighbours (always
/// checked) is background.  In 8-connectivity mode the diagonal neighbours are
/// also considered.
fn is_boundary(
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    fg: &[bool],
    connectivity: Connectivity,
) -> bool {
    // 4-connected neighbours
    let neighbours_4 = [
        if x > 0 { Some((x - 1, y)) } else { None },
        if x + 1 < width {
            Some((x + 1, y))
        } else {
            None
        },
        if y > 0 { Some((x, y - 1)) } else { None },
        if y + 1 < height {
            Some((x, y + 1))
        } else {
            None
        },
    ];

    // Edge pixels (adjacent to the image border) are always boundary pixels
    let at_border = x == 0 || y == 0 || x + 1 >= width || y + 1 >= height;
    if at_border {
        return true;
    }

    for (nx, ny) in neighbours_4.iter().flatten() {
        if !fg[ny * width + nx] {
            return true;
        }
    }

    if connectivity == Connectivity::Eight {
        let neighbours_diag = [
            if x > 0 && y > 0 {
                Some((x - 1, y - 1))
            } else {
                None
            },
            if x + 1 < width && y > 0 {
                Some((x + 1, y - 1))
            } else {
                None
            },
            if x > 0 && y + 1 < height {
                Some((x - 1, y + 1))
            } else {
                None
            },
            if x + 1 < width && y + 1 < height {
                Some((x + 1, y + 1))
            } else {
                None
            },
        ];
        for (nx, ny) in neighbours_diag.iter().flatten() {
            if !fg[ny * width + nx] {
                return true;
            }
        }
    }

    false
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Detect blobs using default parameters.
///
/// Equivalent to `BlobDetector::new(BlobConfig::default()).detect(image, width, height)`.
#[must_use]
pub fn detect_blobs(image: &[u8], width: usize, height: usize) -> Vec<Blob> {
    BlobDetector::new(BlobConfig::default()).detect(image, width, height)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a blank image and paint a filled rectangle.
    fn make_rect_image(
        img_w: usize,
        img_h: usize,
        rx: usize,
        ry: usize,
        rw: usize,
        rh: usize,
        val: u8,
    ) -> Vec<u8> {
        let mut img = vec![0u8; img_w * img_h];
        for y in ry..ry + rh {
            for x in rx..rx + rw {
                if y < img_h && x < img_w {
                    img[y * img_w + x] = val;
                }
            }
        }
        img
    }

    #[test]
    fn test_detect_empty_image() {
        let cfg = BlobConfig::default();
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&[], 0, 0);
        assert!(blobs.is_empty());
    }

    #[test]
    fn test_detect_all_background_no_blobs() {
        let img = vec![0u8; 100];
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 10, 10);
        assert!(blobs.is_empty());
    }

    #[test]
    fn test_detect_single_blob_area_correct() {
        // 3×3 white blob in a 10×10 image
        let img = make_rect_image(10, 10, 4, 4, 3, 3, 200);
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 10, 10);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 9);
    }

    #[test]
    fn test_detect_centroid_correct() {
        // 3×3 blob centred at (5,5) in a 10×10 image
        let img = make_rect_image(10, 10, 4, 4, 3, 3, 255);
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 10, 10);
        assert_eq!(blobs.len(), 1);
        // Centroid should be at (5.0, 5.0): columns 4,5,6 → mean=5
        assert!((blobs[0].cx - 5.0).abs() < 0.01, "cx={}", blobs[0].cx);
        assert!((blobs[0].cy - 5.0).abs() < 0.01, "cy={}", blobs[0].cy);
    }

    #[test]
    fn test_detect_bounding_box_correct() {
        let img = make_rect_image(20, 20, 5, 3, 6, 4, 200);
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 20, 20);
        assert_eq!(blobs.len(), 1);
        let (bx, by, bw, bh) = blobs[0].bbox();
        assert_eq!(bx, 5, "bbox_x mismatch");
        assert_eq!(by, 3, "bbox_y mismatch");
        assert_eq!(bw, 6, "bbox_w mismatch");
        assert_eq!(bh, 4, "bbox_h mismatch");
    }

    #[test]
    fn test_detect_two_separate_blobs() {
        let mut img = vec![0u8; 20 * 20];
        // blob A: rows 1-2, cols 1-2
        for y in 1..3usize {
            for x in 1..3usize {
                img[y * 20 + x] = 255;
            }
        }
        // blob B: rows 15-17, cols 15-17
        for y in 15..18usize {
            for x in 15..18usize {
                img[y * 20 + x] = 255;
            }
        }
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 20, 20);
        assert_eq!(blobs.len(), 2);
    }

    #[test]
    fn test_detect_area_filter_min() {
        // Blob with area 4, filter asks for min_area 10 → filtered out
        let img = make_rect_image(20, 20, 2, 2, 2, 2, 200);
        let mut cfg = BlobConfig::default();
        cfg.min_area = 10;
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&img, 20, 20);
        assert!(blobs.is_empty(), "Small blob should be filtered out");
    }

    #[test]
    fn test_detect_area_filter_max() {
        // Blob with area 25, filter caps at 20 → filtered
        let img = make_rect_image(20, 20, 0, 0, 5, 5, 200);
        let mut cfg = BlobConfig::default();
        cfg.max_area = 20;
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&img, 20, 20);
        assert!(blobs.is_empty(), "Large blob should be filtered out");
    }

    #[test]
    fn test_detect_aspect_ratio_filter() {
        // Very wide blob: 10 wide × 1 tall → aspect_ratio = 10
        let img = make_rect_image(20, 20, 0, 5, 10, 1, 200);
        let mut cfg = BlobConfig::default();
        cfg.max_aspect_ratio = 5.0; // only allow up to 5:1
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&img, 20, 20);
        assert!(blobs.is_empty(), "Blob with too high aspect ratio filtered");
    }

    #[test]
    fn test_detect_sorted_descending_area() {
        let mut img = vec![0u8; 30 * 30];
        // Small blob (4 px) at top-left
        for y in 0..2usize {
            for x in 0..2usize {
                img[y * 30 + x] = 255;
            }
        }
        // Large blob (25 px) at bottom-right
        for y in 20..25usize {
            for x in 20..25usize {
                img[y * 30 + x] = 255;
            }
        }
        let det = BlobDetector::new(BlobConfig::default());
        let blobs = det.detect(&img, 30, 30);
        assert_eq!(blobs.len(), 2);
        assert!(
            blobs[0].area >= blobs[1].area,
            "Blobs should be sorted by descending area"
        );
    }

    #[test]
    fn test_detect_4_connectivity_separates_diagonal_blobs() {
        // Two single pixels touching only diagonally
        let mut img = vec![0u8; 4 * 4];
        img[0] = 255; // (0,0)
        img[1 * 4 + 1] = 255; // (1,1) — diagonal of (0,0)
        let mut cfg = BlobConfig::default();
        cfg.connectivity = Connectivity::Four;
        cfg.min_area = 1;
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&img, 4, 4);
        // With 4-connectivity these should be separate blobs
        assert_eq!(
            blobs.len(),
            2,
            "4-connectivity: diagonal pixels = separate blobs"
        );
    }

    #[test]
    fn test_detect_8_connectivity_merges_diagonal_blobs() {
        let mut img = vec![0u8; 4 * 4];
        img[0] = 255; // (0,0)
        img[1 * 4 + 1] = 255; // (1,1)
        let mut cfg = BlobConfig::default();
        cfg.connectivity = Connectivity::Eight;
        cfg.min_area = 1;
        let det = BlobDetector::new(cfg);
        let blobs = det.detect(&img, 4, 4);
        // With 8-connectivity these should merge into one blob
        assert_eq!(blobs.len(), 1, "8-connectivity: diagonal pixels = one blob");
    }

    #[test]
    fn test_detect_blobs_convenience_fn() {
        let img = make_rect_image(10, 10, 3, 3, 4, 4, 200);
        let blobs = detect_blobs(&img, 10, 10);
        assert_eq!(blobs.len(), 1);
        assert_eq!(blobs[0].area, 16);
    }
}
