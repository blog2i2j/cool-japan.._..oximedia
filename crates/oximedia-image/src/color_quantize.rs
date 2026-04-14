//! Color quantization algorithms for palette generation and image color reduction.
//
// Implements three complementary quantization strategies:
//
// - **Median-cut** — splits the color space recursively at the median of the
//   longest axis; fast and produces perceptually balanced palettes.
// - **Octree** — builds a tree in 8-bit RGB space and prunes leaf clusters;
//   memory-efficient and handles large images well.
// - **K-means** — iterative centroid refinement for minimum quantization error.
//
// After palette generation every method maps each input pixel to its nearest
// palette entry (Euclidean distance in RGB) and optionally applies
// Floyd-Steinberg error diffusion dithering.

/// An RGB color with f64 components in [0.0, 1.0].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbColor {
    /// Red channel.
    pub r: f64,
    /// Green channel.
    pub g: f64,
    /// Blue channel.
    pub b: f64,
}

impl RgbColor {
    /// Create a new color from f64 components clamped to [0, 1].
    #[must_use]
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Create from u8 components.
    #[must_use]
    pub fn from_u8(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f64 / 255.0,
            g: g as f64 / 255.0,
            b: b as f64 / 255.0,
        }
    }

    /// Convert to u8 triple.
    #[must_use]
    pub fn to_u8(self) -> (u8, u8, u8) {
        (
            (self.r * 255.0).round() as u8,
            (self.g * 255.0).round() as u8,
            (self.b * 255.0).round() as u8,
        )
    }

    /// Squared Euclidean distance to another color.
    #[must_use]
    pub fn sq_dist(self, other: Self) -> f64 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        dr * dr + dg * dg + db * db
    }

    /// Euclidean distance to another color.
    #[must_use]
    pub fn dist(self, other: Self) -> f64 {
        self.sq_dist(other).sqrt()
    }

    /// Component-wise addition (unclamped).
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    /// Scalar multiplication (unclamped).
    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        Self {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }
}

impl Default for RgbColor {
    fn default() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Find the nearest palette entry index for a given color.
#[must_use]
pub fn nearest_color(color: RgbColor, palette: &[RgbColor]) -> usize {
    palette
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, color.sq_dist(p)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Map every pixel in `pixels` to its nearest palette entry, returning a
/// vector of palette indices (one per pixel).
#[must_use]
pub fn map_to_palette(pixels: &[RgbColor], palette: &[RgbColor]) -> Vec<usize> {
    pixels.iter().map(|&c| nearest_color(c, palette)).collect()
}

/// Reconstruct a pixel list by replacing each pixel with its palette entry.
#[must_use]
pub fn quantize_pixels(pixels: &[RgbColor], palette: &[RgbColor]) -> Vec<RgbColor> {
    pixels
        .iter()
        .map(|&c| {
            let idx = nearest_color(c, palette);
            palette[idx]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Floyd-Steinberg dithering
// ---------------------------------------------------------------------------

/// Apply Floyd-Steinberg error diffusion dithering to a pixel grid.
///
/// `pixels` is row-major with `width` columns.  The function modifies a copy
/// of the pixel buffer and maps each dithered pixel to the nearest palette
/// entry, returning the resulting palette-index image.
#[must_use]
pub fn floyd_steinberg_dither(
    pixels: &[RgbColor],
    width: usize,
    height: usize,
    palette: &[RgbColor],
) -> Vec<usize> {
    let mut buf: Vec<RgbColor> = pixels.to_vec();
    let mut indices = vec![0usize; width * height];

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let old = buf[idx];
            let nearest_idx = nearest_color(old, palette);
            let new_color = palette[nearest_idx];
            indices[idx] = nearest_idx;

            let err = RgbColor {
                r: old.r - new_color.r,
                g: old.g - new_color.g,
                b: old.b - new_color.b,
            };

            // Distribute error: right, below-left, below, below-right
            let diffuse = |buf: &mut Vec<RgbColor>, tx: usize, ty: usize, frac: f64| {
                if tx < width && ty < height {
                    let ti = ty * width + tx;
                    buf[ti].r = (buf[ti].r + err.r * frac).clamp(0.0, 1.0);
                    buf[ti].g = (buf[ti].g + err.g * frac).clamp(0.0, 1.0);
                    buf[ti].b = (buf[ti].b + err.b * frac).clamp(0.0, 1.0);
                }
            };

            if x + 1 < width {
                diffuse(&mut buf, x + 1, y, 7.0 / 16.0);
            }
            if y + 1 < height {
                if x > 0 {
                    diffuse(&mut buf, x - 1, y + 1, 3.0 / 16.0);
                }
                diffuse(&mut buf, x, y + 1, 5.0 / 16.0);
                if x + 1 < width {
                    diffuse(&mut buf, x + 1, y + 1, 1.0 / 16.0);
                }
            }
        }
    }

    indices
}

// ---------------------------------------------------------------------------
// Median-cut quantization
// ---------------------------------------------------------------------------

/// Perform median-cut color quantization and return a palette of `num_colors`
/// entries.
///
/// The algorithm recursively partitions the set of colors into buckets by
/// splitting along the channel with the greatest range, at the median value.
/// The centroid of each final bucket becomes a palette entry.
#[must_use]
pub fn median_cut(pixels: &[RgbColor], num_colors: usize) -> Vec<RgbColor> {
    if pixels.is_empty() || num_colors == 0 {
        return Vec::new();
    }
    let clamped = num_colors.min(pixels.len());
    let mut buckets: Vec<Vec<RgbColor>> = vec![pixels.to_vec()];

    while buckets.len() < clamped {
        // Pick the bucket with the largest range
        let Some(split_idx) = find_largest_range_bucket(&buckets) else {
            break;
        };
        let bucket = buckets.remove(split_idx);
        let (a, b) = split_bucket(bucket);
        buckets.push(a);
        buckets.push(b);
    }

    buckets.iter().map(|b| bucket_centroid(b)).collect()
}

/// Find the index of the bucket with the largest color-space range.
fn find_largest_range_bucket(buckets: &[Vec<RgbColor>]) -> Option<usize> {
    buckets
        .iter()
        .enumerate()
        .map(|(i, b)| (i, bucket_range(b)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

/// Compute the maximum channel range of a bucket.
fn bucket_range(bucket: &[RgbColor]) -> f64 {
    if bucket.is_empty() {
        return 0.0;
    }
    let (rmin, rmax, gmin, gmax, bmin, bmax) = channel_extremes(bucket);
    let rrange = rmax - rmin;
    let grange = gmax - gmin;
    let brange = bmax - bmin;
    rrange.max(grange).max(brange)
}

/// Compute per-channel min/max for a bucket.
fn channel_extremes(bucket: &[RgbColor]) -> (f64, f64, f64, f64, f64, f64) {
    let mut rmin = f64::INFINITY;
    let mut rmax = f64::NEG_INFINITY;
    let mut gmin = f64::INFINITY;
    let mut gmax = f64::NEG_INFINITY;
    let mut bmin = f64::INFINITY;
    let mut bmax = f64::NEG_INFINITY;
    for c in bucket {
        rmin = rmin.min(c.r);
        rmax = rmax.max(c.r);
        gmin = gmin.min(c.g);
        gmax = gmax.max(c.g);
        bmin = bmin.min(c.b);
        bmax = bmax.max(c.b);
    }
    (rmin, rmax, gmin, gmax, bmin, bmax)
}

/// Split a bucket along the widest channel at the median value.
fn split_bucket(mut bucket: Vec<RgbColor>) -> (Vec<RgbColor>, Vec<RgbColor>) {
    let (rmin, rmax, gmin, gmax, bmin, bmax) = channel_extremes(&bucket);
    let rrange = rmax - rmin;
    let grange = gmax - gmin;
    let brange = bmax - bmin;

    if rrange >= grange && rrange >= brange {
        bucket.sort_by(|a, b| a.r.partial_cmp(&b.r).unwrap_or(std::cmp::Ordering::Equal));
    } else if grange >= brange {
        bucket.sort_by(|a, b| a.g.partial_cmp(&b.g).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        bucket.sort_by(|a, b| a.b.partial_cmp(&b.b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mid = bucket.len() / 2;
    let right = bucket.split_off(mid);
    (bucket, right)
}

/// Compute the centroid (mean color) of a bucket.
fn bucket_centroid(bucket: &[RgbColor]) -> RgbColor {
    if bucket.is_empty() {
        return RgbColor::default();
    }
    let n = bucket.len() as f64;
    let sum = bucket
        .iter()
        .fold(RgbColor::default(), |acc, &c| acc.add(c));
    RgbColor {
        r: (sum.r / n).clamp(0.0, 1.0),
        g: (sum.g / n).clamp(0.0, 1.0),
        b: (sum.b / n).clamp(0.0, 1.0),
    }
}

// ---------------------------------------------------------------------------
// Octree quantization
// ---------------------------------------------------------------------------

/// Maximum octree depth (levels 0..=7).
const MAX_DEPTH: usize = 7;

/// One node of the octree.
struct OctreeNode {
    /// Sum of red values inserted into this node.
    r_sum: u64,
    /// Sum of green values inserted into this node.
    g_sum: u64,
    /// Sum of blue values inserted into this node.
    b_sum: u64,
    /// Number of pixels accumulated.
    count: u64,
    /// Children (one per octant).
    children: [Option<Box<OctreeNode>>; 8],
    /// True if this is a leaf node.
    is_leaf: bool,
}

impl OctreeNode {
    fn new() -> Self {
        Self {
            r_sum: 0,
            g_sum: 0,
            b_sum: 0,
            count: 0,
            children: Default::default(),
            is_leaf: false,
        }
    }

    /// Insert an 8-bit RGB triple at the given tree depth.
    fn insert(&mut self, r: u8, g: u8, b: u8, depth: usize) {
        if depth >= MAX_DEPTH {
            self.r_sum += r as u64;
            self.g_sum += g as u64;
            self.b_sum += b as u64;
            self.count += 1;
            self.is_leaf = true;
            return;
        }

        let shift = 7 - depth;
        let index = (((r >> shift) & 1) as usize) << 2
            | (((g >> shift) & 1) as usize) << 1
            | (((b >> shift) & 1) as usize);

        if self.children[index].is_none() {
            self.children[index] = Some(Box::new(OctreeNode::new()));
        }

        if let Some(child) = &mut self.children[index] {
            child.insert(r, g, b, depth + 1);
        }
    }

    /// Count the number of leaf nodes in this subtree.
    fn leaf_count(&self) -> usize {
        if self.is_leaf {
            return 1;
        }
        self.children
            .iter()
            .filter_map(|c| c.as_ref())
            .map(|c| c.leaf_count())
            .sum()
    }

    /// Merge all children of this node into itself, making it a leaf.
    fn merge(&mut self) {
        for child_opt in &mut self.children {
            if let Some(child) = child_opt.take() {
                self.r_sum += child.r_sum;
                self.g_sum += child.g_sum;
                self.b_sum += child.b_sum;
                self.count += child.count;
            }
        }
        self.is_leaf = true;
    }

    /// Collect all leaf centroids into `palette`.
    fn collect_palette(&self, palette: &mut Vec<RgbColor>) {
        if self.is_leaf && self.count > 0 {
            let n = self.count as f64;
            palette.push(RgbColor::new(
                self.r_sum as f64 / n / 255.0,
                self.g_sum as f64 / n / 255.0,
                self.b_sum as f64 / n / 255.0,
            ));
            return;
        }
        for child in self.children.iter().filter_map(|c| c.as_ref()) {
            child.collect_palette(palette);
        }
    }

    /// Find a node at the deepest interior level that has multiple children
    /// and reduce it.  Returns true if a merge was performed.
    fn reduce_deepest(&mut self, target_depth: usize, current_depth: usize) -> bool {
        if current_depth + 1 == target_depth {
            // Merge any children that are not leaves themselves
            let has_reducible = self
                .children
                .iter()
                .filter_map(|c| c.as_ref())
                .any(|c| !c.is_leaf || c.count > 0);
            if has_reducible {
                self.merge();
                return true;
            }
            return false;
        }
        for child in self.children.iter_mut().filter_map(|c| c.as_mut()) {
            if child.reduce_deepest(target_depth, current_depth + 1) {
                return true;
            }
        }
        false
    }
}

/// Perform octree color quantization and return at most `num_colors` palette
/// entries.
#[must_use]
pub fn octree_quantize(pixels: &[RgbColor], num_colors: usize) -> Vec<RgbColor> {
    if pixels.is_empty() || num_colors == 0 {
        return Vec::new();
    }

    let mut root = OctreeNode::new();
    for &p in pixels {
        let (r, g, b) = p.to_u8();
        root.insert(r, g, b, 0);
    }

    // Reduce from the deepest level upward until leaf count <= num_colors
    for target_depth in (1..=MAX_DEPTH).rev() {
        if root.leaf_count() <= num_colors {
            break;
        }
        // Keep merging at this depth until no more reductions are possible
        // or we've gone below the target
        loop {
            if root.leaf_count() <= num_colors {
                break;
            }
            if !root.reduce_deepest(target_depth, 0) {
                break;
            }
        }
    }

    let mut palette = Vec::new();
    root.collect_palette(&mut palette);
    palette
}

// ---------------------------------------------------------------------------
// K-means quantization
// ---------------------------------------------------------------------------

/// Perform k-means color quantization.
///
/// `num_colors` centroids are initialised by choosing evenly-spaced samples
/// from the input (a simple but deterministic seed strategy).  The algorithm
/// iterates until assignments stop changing or `max_iterations` is reached.
#[must_use]
pub fn kmeans_quantize(
    pixels: &[RgbColor],
    num_colors: usize,
    max_iterations: usize,
) -> Vec<RgbColor> {
    if pixels.is_empty() || num_colors == 0 {
        return Vec::new();
    }
    let k = num_colors.min(pixels.len());

    // Initialise centroids by uniform sampling (deterministic)
    let step = pixels.len() / k;
    let mut centroids: Vec<RgbColor> = (0..k).map(|i| pixels[i * step]).collect();

    let mut assignments = vec![0usize; pixels.len()];

    for _ in 0..max_iterations {
        // Assignment step
        let mut changed = false;
        for (i, &p) in pixels.iter().enumerate() {
            let best = nearest_color(p, &centroids);
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step
        let mut sums = vec![RgbColor::default(); k];
        let mut counts = vec![0u64; k];
        for (i, &p) in pixels.iter().enumerate() {
            let ci = assignments[i];
            sums[ci] = sums[ci].add(p);
            counts[ci] += 1;
        }

        for j in 0..k {
            if counts[j] > 0 {
                let n = counts[j] as f64;
                centroids[j] = RgbColor::new(sums[j].r / n, sums[j].g / n, sums[j].b / n);
            }
        }
    }

    centroids
}

// ---------------------------------------------------------------------------
// High-level API
// ---------------------------------------------------------------------------

/// Quantization algorithm to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuantizeAlgorithm {
    /// Median-cut (fast, good perceptual quality).
    MedianCut,
    /// Octree (memory-efficient, good for large images).
    Octree,
    /// K-means (slowest, minimises quantization error).
    KMeans,
}

/// Configuration for color quantization.
#[derive(Clone, Debug)]
pub struct QuantizeConfig {
    /// Number of colors in the output palette.
    pub num_colors: usize,
    /// Algorithm to use.
    pub algorithm: QuantizeAlgorithm,
    /// Whether to apply Floyd-Steinberg dithering after quantization.
    pub dither: bool,
    /// Maximum iterations for k-means (ignored for other algorithms).
    pub max_iterations: usize,
}

impl Default for QuantizeConfig {
    fn default() -> Self {
        Self {
            num_colors: 256,
            algorithm: QuantizeAlgorithm::MedianCut,
            dither: false,
            max_iterations: 50,
        }
    }
}

impl QuantizeConfig {
    /// Create a new config.
    #[must_use]
    pub fn new(num_colors: usize, algorithm: QuantizeAlgorithm) -> Self {
        Self {
            num_colors,
            algorithm,
            ..Default::default()
        }
    }

    /// Enable dithering.
    #[must_use]
    pub fn with_dither(mut self, dither: bool) -> Self {
        self.dither = dither;
        self
    }

    /// Set k-means iteration limit.
    #[must_use]
    pub fn with_max_iterations(mut self, iters: usize) -> Self {
        self.max_iterations = iters;
        self
    }
}

/// Result of a color quantization operation.
pub struct QuantizeResult {
    /// The generated palette.
    pub palette: Vec<RgbColor>,
    /// Per-pixel palette indices (row-major).
    pub indices: Vec<usize>,
}

/// Quantize a pixel buffer to a reduced palette.
///
/// `pixels` is a row-major buffer of `width * height` colors.
/// Returns [`QuantizeResult`] containing the palette and index map.
#[must_use]
pub fn quantize(
    pixels: &[RgbColor],
    width: usize,
    height: usize,
    config: &QuantizeConfig,
) -> QuantizeResult {
    let palette = match config.algorithm {
        QuantizeAlgorithm::MedianCut => median_cut(pixels, config.num_colors),
        QuantizeAlgorithm::Octree => octree_quantize(pixels, config.num_colors),
        QuantizeAlgorithm::KMeans => {
            kmeans_quantize(pixels, config.num_colors, config.max_iterations)
        }
    };

    let indices = if config.dither {
        floyd_steinberg_dither(pixels, width, height, &palette)
    } else {
        map_to_palette(pixels, &palette)
    };

    QuantizeResult { palette, indices }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_pixels(n: usize) -> Vec<RgbColor> {
        (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                RgbColor::new(t, 1.0 - t, 0.5)
            })
            .collect()
    }

    #[test]
    fn test_rgb_color_new_clamps() {
        let c = RgbColor::new(-0.1, 1.5, 0.5);
        assert!((c.r - 0.0).abs() < f64::EPSILON);
        assert!((c.g - 1.0).abs() < f64::EPSILON);
        assert!((c.b - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rgb_color_round_trip_u8() {
        let c = RgbColor::from_u8(128, 64, 255);
        let (r, g, b) = c.to_u8();
        assert_eq!(r, 128);
        assert_eq!(g, 64);
        assert_eq!(b, 255);
    }

    #[test]
    fn test_nearest_color() {
        let palette = vec![
            RgbColor::new(0.0, 0.0, 0.0),
            RgbColor::new(1.0, 0.0, 0.0),
            RgbColor::new(0.0, 1.0, 0.0),
        ];
        let query = RgbColor::new(0.9, 0.1, 0.0);
        assert_eq!(nearest_color(query, &palette), 1);
    }

    #[test]
    fn test_median_cut_palette_size() {
        let pixels = gradient_pixels(200);
        let palette = median_cut(&pixels, 8);
        assert!(palette.len() <= 8);
        assert!(!palette.is_empty());
    }

    #[test]
    fn test_octree_palette_size() {
        let pixels = gradient_pixels(200);
        let palette = octree_quantize(&pixels, 16);
        assert!(palette.len() <= 16);
        assert!(!palette.is_empty());
    }

    #[test]
    fn test_kmeans_palette_size() {
        let pixels = gradient_pixels(100);
        let palette = kmeans_quantize(&pixels, 4, 20);
        assert_eq!(palette.len(), 4);
    }

    #[test]
    fn test_map_to_palette_coverage() {
        let pixels = gradient_pixels(64);
        let palette = median_cut(&pixels, 8);
        let indices = map_to_palette(&pixels, &palette);
        assert_eq!(indices.len(), 64);
        // All indices must be valid palette positions
        for &idx in &indices {
            assert!(idx < palette.len());
        }
    }

    #[test]
    fn test_floyd_steinberg_dither_indices() {
        let pixels: Vec<RgbColor> = vec![
            RgbColor::new(0.0, 0.0, 0.0),
            RgbColor::new(1.0, 1.0, 1.0),
            RgbColor::new(0.5, 0.5, 0.5),
            RgbColor::new(0.2, 0.8, 0.4),
        ];
        let palette = vec![RgbColor::new(0.0, 0.0, 0.0), RgbColor::new(1.0, 1.0, 1.0)];
        let indices = floyd_steinberg_dither(&pixels, 2, 2, &palette);
        assert_eq!(indices.len(), 4);
        for &idx in &indices {
            assert!(idx < palette.len());
        }
    }

    #[test]
    fn test_quantize_high_level_median_cut() {
        let pixels = gradient_pixels(128);
        let config = QuantizeConfig::new(16, QuantizeAlgorithm::MedianCut);
        let result = quantize(&pixels, 128, 1, &config);
        assert!(!result.palette.is_empty());
        assert_eq!(result.indices.len(), 128);
    }

    #[test]
    fn test_quantize_with_dither() {
        let pixels = gradient_pixels(100);
        let config = QuantizeConfig::new(8, QuantizeAlgorithm::MedianCut).with_dither(true);
        let result = quantize(&pixels, 100, 1, &config);
        assert_eq!(result.indices.len(), 100);
        for &idx in &result.indices {
            assert!(idx < result.palette.len());
        }
    }

    #[test]
    fn test_quantize_empty_input() {
        let config = QuantizeConfig::default();
        let result = quantize(&[], 0, 0, &config);
        assert!(result.palette.is_empty());
        assert!(result.indices.is_empty());
    }

    #[test]
    fn test_quantize_kmeans() {
        let pixels = gradient_pixels(80);
        let config = QuantizeConfig::new(4, QuantizeAlgorithm::KMeans).with_max_iterations(10);
        let result = quantize(&pixels, 80, 1, &config);
        assert_eq!(result.palette.len(), 4);
        assert_eq!(result.indices.len(), 80);
    }
}
