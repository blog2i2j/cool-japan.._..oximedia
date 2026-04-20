#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
//! Colour palette extraction and manipulation.
//!
//! Provides three palette-extraction algorithms:
//! - **MedianCut** — recursive median-cut in RGB colour space
//! - **KMeans(max_iter)** — k-means++ initialisation with iterative refinement
//! - **Octree** — octree colour quantisation reducing to N colours
//!
//! After extraction the palette can be used to quantise image data, find
//! the nearest colour, sort entries, or evaluate visual diversity.

use std::collections::HashMap;

// ── Types ─────────────────────────────────────────────────────────────────

/// A single entry in a colour palette, with a relative weight (frequency).
#[derive(Debug, Clone, PartialEq)]
pub struct PaletteColor {
    /// Red channel (0..255).
    pub r: u8,
    /// Green channel (0..255).
    pub g: u8,
    /// Blue channel (0..255).
    pub b: u8,
    /// Relative frequency of this colour in the source image (0..1).
    pub weight: f32,
}

impl PaletteColor {
    /// Squared Euclidean distance to another colour in RGB space.
    #[must_use]
    pub fn distance_sq_to(&self, r: u8, g: u8, b: u8) -> u32 {
        let dr = (self.r as i32) - (r as i32);
        let dg = (self.g as i32) - (g as i32);
        let db = (self.b as i32) - (b as i32);
        (dr * dr + dg * dg + db * db) as u32
    }

    /// Perceived luminance (BT.601).
    #[must_use]
    pub fn luminance(&self) -> f32 {
        0.299 * (self.r as f32 / 255.0)
            + 0.587 * (self.g as f32 / 255.0)
            + 0.114 * (self.b as f32 / 255.0)
    }
}

/// Algorithm used to extract the colour palette.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaletteAlgorithm {
    /// Recursive median-cut in RGB space.
    MedianCut,
    /// K-means clustering with k-means++ seeding, given max iterations.
    KMeans(u32),
    /// Octree colour quantisation.
    Octree,
}

/// A colour palette extracted from image data.
#[derive(Debug, Clone)]
pub struct ColorPalette {
    /// Palette entries, each with an RGB value and relative weight.
    pub colors: Vec<PaletteColor>,
    /// Algorithm that was used to build this palette.
    pub algorithm: PaletteAlgorithm,
}

// ── Public API ────────────────────────────────────────────────────────────

impl ColorPalette {
    /// Extract a palette of at most `n_colors` entries from a flat RGB pixel
    /// buffer (`pixels.len()` must be a multiple of 3: R,G,B,R,G,B,…).
    ///
    /// Returns an empty palette if `pixels` is empty or `n_colors == 0`.
    #[must_use]
    pub fn extract(pixels: &[u8], n_colors: usize, algorithm: PaletteAlgorithm) -> Self {
        if pixels.is_empty() || n_colors == 0 || pixels.len() < 3 {
            return Self {
                colors: Vec::new(),
                algorithm,
            };
        }

        let rgb_pixels = collect_rgb(pixels);

        let colors = match &algorithm {
            PaletteAlgorithm::MedianCut => median_cut_extract(&rgb_pixels, n_colors),
            PaletteAlgorithm::KMeans(max_iter) => kmeans_extract(&rgb_pixels, n_colors, *max_iter),
            PaletteAlgorithm::Octree => octree_extract(&rgb_pixels, n_colors),
        };

        Self { colors, algorithm }
    }

    /// Return the palette entry closest (by Euclidean RGB distance) to the
    /// given colour. Returns `None` only when the palette is empty.
    #[must_use]
    pub fn nearest_color(&self, r: u8, g: u8, b: u8) -> Option<&PaletteColor> {
        self.colors
            .iter()
            .min_by_key(|pc| pc.distance_sq_to(r, g, b))
    }

    /// Replace every pixel in a flat RGB buffer with the nearest palette colour.
    #[must_use]
    pub fn quantize_image(&self, pixels: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(pixels.len());
        for chunk in pixels.chunks_exact(3) {
            if let Some(pc) = self.nearest_color(chunk[0], chunk[1], chunk[2]) {
                out.push(pc.r);
                out.push(pc.g);
                out.push(pc.b);
            } else {
                out.push(chunk[0]);
                out.push(chunk[1]);
                out.push(chunk[2]);
            }
        }
        out
    }

    /// Return the palette entry with the highest weight (dominant colour).
    /// Returns `None` when the palette is empty.
    #[must_use]
    pub fn dominant_color(&self) -> Option<&PaletteColor> {
        self.colors.iter().max_by(|a, b| {
            a.weight
                .partial_cmp(&b.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Sort palette entries by weight descending (most dominant first).
    pub fn sort_by_weight(&mut self) {
        self.colors.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Sort palette entries by perceived luminance ascending (dark first).
    pub fn sort_by_luminance(&mut self) {
        self.colors.sort_by(|a, b| {
            a.luminance()
                .partial_cmp(&b.luminance())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Average pairwise Euclidean RGB distance normalised to [0, 1] where
    /// `1.0` corresponds to the maximum possible RGB distance (≈ 441.67).
    ///
    /// Returns `0.0` for palettes with fewer than 2 entries.
    #[must_use]
    pub fn diversity_score(&self) -> f32 {
        let n = self.colors.len();
        if n < 2 {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        let mut count = 0u32;
        for i in 0..n {
            for j in (i + 1)..n {
                let dr = (self.colors[i].r as f32) - (self.colors[j].r as f32);
                let dg = (self.colors[i].g as f32) - (self.colors[j].g as f32);
                let db = (self.colors[i].b as f32) - (self.colors[j].b as f32);
                sum += (dr * dr + dg * dg + db * db).sqrt();
                count += 1;
            }
        }
        if count == 0 {
            return 0.0;
        }
        (sum / count as f32) / 441.673
    }
}

// ── Helper: parse pixel bytes → (r,g,b) triples ──────────────────────────

fn collect_rgb(pixels: &[u8]) -> Vec<(u8, u8, u8)> {
    pixels.chunks_exact(3).map(|c| (c[0], c[1], c[2])).collect()
}

// ── Weighted frequency map ────────────────────────────────────────────────

/// Returns a frequency-weighted list of unique colours.
fn color_frequencies(pixels: &[(u8, u8, u8)]) -> Vec<((u8, u8, u8), usize)> {
    let mut map: HashMap<(u8, u8, u8), usize> = HashMap::new();
    for &px in pixels {
        *map.entry(px).or_insert(0) += 1;
    }
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    v
}

fn weight_from_count(count: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        count as f32 / total as f32
    }
}

// ── Median-cut ────────────────────────────────────────────────────────────

struct ColorBox {
    colors: Vec<(u8, u8, u8)>,
}

impl ColorBox {
    fn new(colors: Vec<(u8, u8, u8)>) -> Self {
        Self { colors }
    }

    fn widest_channel(&self) -> u8 {
        let (mut rmin, mut rmax) = (u8::MAX, u8::MIN);
        let (mut gmin, mut gmax) = (u8::MAX, u8::MIN);
        let (mut bmin, mut bmax) = (u8::MAX, u8::MIN);
        for &(r, g, b) in &self.colors {
            rmin = rmin.min(r);
            rmax = rmax.max(r);
            gmin = gmin.min(g);
            gmax = gmax.max(g);
            bmin = bmin.min(b);
            bmax = bmax.max(b);
        }
        let rr = rmax - rmin;
        let gr = gmax - gmin;
        let br = bmax - bmin;
        if rr >= gr && rr >= br {
            0
        } else if gr >= br {
            1
        } else {
            2
        }
    }

    fn average(&self) -> (u8, u8, u8) {
        if self.colors.is_empty() {
            return (0, 0, 0);
        }
        let n = self.colors.len() as u64;
        let mut rs = 0u64;
        let mut gs = 0u64;
        let mut bs = 0u64;
        for &(r, g, b) in &self.colors {
            rs += r as u64;
            gs += g as u64;
            bs += b as u64;
        }
        ((rs / n) as u8, (gs / n) as u8, (bs / n) as u8)
    }

    fn split(mut self) -> (Self, Self) {
        let ch = self.widest_channel();
        match ch {
            0 => self.colors.sort_by_key(|c| c.0),
            1 => self.colors.sort_by_key(|c| c.1),
            _ => self.colors.sort_by_key(|c| c.2),
        }
        let mid = self.colors.len() / 2;
        let right = self.colors.split_off(mid);
        (Self::new(self.colors), Self::new(right))
    }
}

fn median_cut_extract(pixels: &[(u8, u8, u8)], n_colors: usize) -> Vec<PaletteColor> {
    let freq = color_frequencies(pixels);
    let total = pixels.len();
    if freq.is_empty() || n_colors == 0 {
        return Vec::new();
    }
    if n_colors >= freq.len() {
        return freq
            .into_iter()
            .map(|((r, g, b), cnt)| PaletteColor {
                r,
                g,
                b,
                weight: weight_from_count(cnt, total),
            })
            .collect();
    }

    // Build initial box from all pixel instances (with repetition for frequency)
    let expanded: Vec<(u8, u8, u8)> = pixels.to_vec();
    let mut boxes = vec![ColorBox::new(expanded)];

    while boxes.len() < n_colors {
        let idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .max_by_key(|(_, b)| b.colors.len())
            .map(|(i, _)| i);
        let Some(idx) = idx else { break };
        let biggest = boxes.remove(idx);
        let (a, b) = biggest.split();
        if !a.colors.is_empty() {
            boxes.push(a);
        }
        if !b.colors.is_empty() {
            boxes.push(b);
        }
    }

    let box_total: usize = boxes.iter().map(|b| b.colors.len()).sum();
    boxes
        .iter()
        .map(|bx| {
            let (r, g, b) = bx.average();
            PaletteColor {
                r,
                g,
                b,
                weight: weight_from_count(bx.colors.len(), box_total.max(1)),
            }
        })
        .collect()
}

// ── K-means ───────────────────────────────────────────────────────────────

/// Squared distance between two RGB triples (integer arithmetic).
fn dist_sq(a: (u8, u8, u8), b: (f32, f32, f32)) -> f32 {
    let dr = (a.0 as f32) - b.0;
    let dg = (a.1 as f32) - b.1;
    let db = (a.2 as f32) - b.2;
    dr * dr + dg * dg + db * db
}

/// K-means++ seeding: choose initial centroids with probability proportional
/// to squared distance from the nearest already-chosen centroid.
fn kmeans_pp_init(pixels: &[(u8, u8, u8)], k: usize) -> Vec<(f32, f32, f32)> {
    if pixels.is_empty() || k == 0 {
        return Vec::new();
    }
    // Seed with a deterministic "random" first centroid (middle element)
    let first = pixels[pixels.len() / 2];
    let mut centroids: Vec<(f32, f32, f32)> =
        vec![(first.0 as f32, first.1 as f32, first.2 as f32)];

    while centroids.len() < k {
        // Compute D² for each pixel
        let dists: Vec<f32> = pixels
            .iter()
            .map(|&p| {
                centroids
                    .iter()
                    .map(|&c| dist_sq(p, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();
        let sum: f32 = dists.iter().sum();
        if sum < 1e-10 {
            // All remaining pixels are identical to existing centroids
            break;
        }
        // Pick the pixel with the highest D² as next centroid (deterministic)
        let best_idx = dists
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let p = pixels[best_idx];
        centroids.push((p.0 as f32, p.1 as f32, p.2 as f32));
    }
    centroids
}

fn kmeans_extract(pixels: &[(u8, u8, u8)], n_colors: usize, max_iter: u32) -> Vec<PaletteColor> {
    if pixels.is_empty() || n_colors == 0 {
        return Vec::new();
    }
    let k = n_colors.min(pixels.len());
    let mut centroids = kmeans_pp_init(pixels, k);
    let total = pixels.len();

    for _ in 0..max_iter {
        // Assignment
        let assignments: Vec<usize> = pixels
            .iter()
            .map(|&p| {
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        dist_sq(p, **a)
                            .partial_cmp(&dist_sq(p, **b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect();

        // Update centroids
        let mut sums = vec![(0.0_f32, 0.0_f32, 0.0_f32); k];
        let mut counts = vec![0usize; k];
        for (&p, &a) in pixels.iter().zip(assignments.iter()) {
            sums[a].0 += p.0 as f32;
            sums[a].1 += p.1 as f32;
            sums[a].2 += p.2 as f32;
            counts[a] += 1;
        }

        let mut changed = false;
        for i in 0..k {
            if counts[i] > 0 {
                let new_c = (
                    sums[i].0 / counts[i] as f32,
                    sums[i].1 / counts[i] as f32,
                    sums[i].2 / counts[i] as f32,
                );
                if (new_c.0 - centroids[i].0).abs() > 0.5
                    || (new_c.1 - centroids[i].1).abs() > 0.5
                    || (new_c.2 - centroids[i].2).abs() > 0.5
                {
                    changed = true;
                }
                centroids[i] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    // Compute final weights by counting assignments
    let assignments: Vec<usize> = pixels
        .iter()
        .map(|&p| {
            centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    dist_sq(p, **a)
                        .partial_cmp(&dist_sq(p, **b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect();

    let mut counts = vec![0usize; k];
    for &a in &assignments {
        counts[a] += 1;
    }

    centroids
        .into_iter()
        .zip(counts.iter())
        .map(|(c, &cnt)| PaletteColor {
            r: c.0.round().clamp(0.0, 255.0) as u8,
            g: c.1.round().clamp(0.0, 255.0) as u8,
            b: c.2.round().clamp(0.0, 255.0) as u8,
            weight: weight_from_count(cnt, total),
        })
        .collect()
}

// ── Octree quantisation ───────────────────────────────────────────────────

/// A single octree node.
struct OctreeNode {
    r_sum: u64,
    g_sum: u64,
    b_sum: u64,
    count: u32,
    children: [Option<Box<OctreeNode>>; 8],
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
}

struct Octree {
    root: Box<OctreeNode>,
    leaf_count: usize,
    max_depth: usize,
}

impl Octree {
    fn new(max_depth: usize) -> Self {
        Self {
            root: Box::new(OctreeNode::new()),
            leaf_count: 0,
            max_depth,
        }
    }

    fn insert(&mut self, r: u8, g: u8, b: u8) {
        let added = Self::insert_node(&mut self.root, r, g, b, 0, self.max_depth);
        self.leaf_count += added;
    }

    fn insert_node(
        node: &mut OctreeNode,
        r: u8,
        g: u8,
        b: u8,
        depth: usize,
        max_depth: usize,
    ) -> usize {
        if depth >= max_depth {
            node.r_sum += r as u64;
            node.g_sum += g as u64;
            node.b_sum += b as u64;
            node.count += 1;
            if !node.is_leaf {
                node.is_leaf = true;
                return 1;
            }
            return 0;
        }
        let shift = 7 - depth;
        let idx = (((r >> shift) & 1) << 2) | (((g >> shift) & 1) << 1) | ((b >> shift) & 1);
        let idx = idx as usize;
        if node.children[idx].is_none() {
            node.children[idx] = Some(Box::new(OctreeNode::new()));
        }
        if let Some(child) = node.children[idx].as_mut() {
            Self::insert_node(child, r, g, b, depth + 1, max_depth)
        } else {
            0
        }
    }

    /// Reduce the tree by merging leaf siblings until leaf count ≤ n_colors.
    fn reduce_to(&mut self, n_colors: usize) {
        while self.leaf_count > n_colors {
            let reduced = Self::reduce_deepest(&mut self.root, self.max_depth);
            if reduced == 0 {
                break;
            }
            // A merge turns multiple leaves into 1 leaf; net change = -(reduced - 1)
            if self.leaf_count >= reduced {
                self.leaf_count -= reduced - 1;
            } else {
                self.leaf_count = 1;
            }
        }
    }

    /// Merge the first node at the deepest level that has all-leaf children.
    /// Returns the number of children that were merged (0 if nothing merged).
    fn reduce_deepest(node: &mut OctreeNode, max_depth: usize) -> usize {
        // Try children first (deeper levels preferred)
        for i in 0..8 {
            if let Some(child) = node.children[i].as_mut() {
                if !child.is_leaf {
                    let reduced = Self::reduce_deepest(child, max_depth);
                    if reduced > 0 {
                        return reduced;
                    }
                }
            }
        }
        // Count leaf children
        let leaf_children: Vec<usize> = (0..8)
            .filter(|&i| node.children[i].as_ref().map_or(false, |c| c.is_leaf))
            .collect();

        if leaf_children.is_empty() {
            return 0;
        }

        // Merge all leaf children into this node
        let count = leaf_children.len();
        for &i in &leaf_children {
            if let Some(child) = node.children[i].take() {
                node.r_sum += child.r_sum;
                node.g_sum += child.g_sum;
                node.b_sum += child.b_sum;
                node.count += child.count;
            }
        }
        node.is_leaf = true;
        count
    }

    /// Collect all leaf colours.
    fn leaves(&self) -> Vec<(u8, u8, u8, u32)> {
        let mut out = Vec::new();
        Self::collect_leaves(&self.root, &mut out);
        out
    }

    fn collect_leaves(node: &OctreeNode, out: &mut Vec<(u8, u8, u8, u32)>) {
        if node.is_leaf && node.count > 0 {
            let r = (node.r_sum / node.count as u64).min(255) as u8;
            let g = (node.g_sum / node.count as u64).min(255) as u8;
            let b = (node.b_sum / node.count as u64).min(255) as u8;
            out.push((r, g, b, node.count));
            return;
        }
        for child in node.children.iter().flatten() {
            Self::collect_leaves(child, out);
        }
    }
}

fn octree_extract(pixels: &[(u8, u8, u8)], n_colors: usize) -> Vec<PaletteColor> {
    if pixels.is_empty() || n_colors == 0 {
        return Vec::new();
    }
    // Depth 6 gives up to 8^6 = 262144 potential leaves, sufficient for 8-bit
    let mut tree = Octree::new(6);
    for &(r, g, b) in pixels {
        tree.insert(r, g, b);
    }
    tree.reduce_to(n_colors);

    let leaves = tree.leaves();
    let total: u32 = leaves.iter().map(|l| l.3).sum();
    leaves
        .into_iter()
        .map(|(r, g, b, cnt)| PaletteColor {
            r,
            g,
            b,
            weight: weight_from_count(cnt as usize, total as usize),
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat RGB pixel buffer from a slice of (r,g,b) triples.
    #[allow(dead_code)]
    fn pixels_from(colors: &[(u8, u8, u8)]) -> Vec<u8> {
        let mut v = Vec::with_capacity(colors.len() * 3);
        for &(r, g, b) in colors {
            v.extend_from_slice(&[r, g, b]);
        }
        v
    }

    /// Uniform red + blue image, 50 pixels each.
    fn red_blue_pixels() -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..50 {
            v.extend_from_slice(&[200, 50, 50]);
        }
        for _ in 0..50 {
            v.extend_from_slice(&[50, 50, 200]);
        }
        v
    }

    // ── MedianCut ─────────────────────────────────────────────────────────

    #[test]
    fn test_median_cut_empty() {
        let p = ColorPalette::extract(&[], 4, PaletteAlgorithm::MedianCut);
        assert!(p.colors.is_empty());
    }

    #[test]
    fn test_median_cut_two_clusters() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::MedianCut);
        assert!(!p.colors.is_empty());
        assert!(p.colors.len() <= 2);
    }

    #[test]
    fn test_median_cut_returns_palette_algorithm() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 4, PaletteAlgorithm::MedianCut);
        assert_eq!(p.algorithm, PaletteAlgorithm::MedianCut);
    }

    #[test]
    fn test_median_cut_weight_sums_approx_one() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 4, PaletteAlgorithm::MedianCut);
        let total: f32 = p.colors.iter().map(|c| c.weight).sum();
        assert!((total - 1.0).abs() < 0.01, "weights sum={total}");
    }

    #[test]
    fn test_median_cut_single_color() {
        // 10 repetitions of (100,150,200)
        let single: Vec<u8> = (0..10).flat_map(|_| [100u8, 150, 200]).collect();
        let p = ColorPalette::extract(&single, 4, PaletteAlgorithm::MedianCut);
        assert_eq!(p.colors.len(), 1);
    }

    // ── KMeans ────────────────────────────────────────────────────────────

    #[test]
    fn test_kmeans_two_clusters() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::KMeans(100));
        assert_eq!(p.colors.len(), 2);
    }

    #[test]
    fn test_kmeans_returns_correct_algorithm() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::KMeans(50));
        assert_eq!(p.algorithm, PaletteAlgorithm::KMeans(50));
    }

    #[test]
    fn test_kmeans_weight_sums_approx_one() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::KMeans(100));
        let total: f32 = p.colors.iter().map(|c| c.weight).sum();
        assert!((total - 1.0).abs() < 0.01, "weights sum={total}");
    }

    #[test]
    fn test_kmeans_empty_input() {
        let p = ColorPalette::extract(&[], 4, PaletteAlgorithm::KMeans(100));
        assert!(p.colors.is_empty());
    }

    // ── Octree ────────────────────────────────────────────────────────────

    #[test]
    fn test_octree_two_clusters() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::Octree);
        assert!(!p.colors.is_empty());
        assert!(p.colors.len() <= 4, "octree len={}", p.colors.len());
    }

    #[test]
    fn test_octree_returns_correct_algorithm() {
        let pixels = red_blue_pixels();
        let p = ColorPalette::extract(&pixels, 2, PaletteAlgorithm::Octree);
        assert_eq!(p.algorithm, PaletteAlgorithm::Octree);
    }

    #[test]
    fn test_octree_empty_input() {
        let p = ColorPalette::extract(&[], 4, PaletteAlgorithm::Octree);
        assert!(p.colors.is_empty());
    }

    // ── nearest_color ─────────────────────────────────────────────────────

    #[test]
    fn test_nearest_color_black_white() {
        let p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 0,
                    weight: 0.5,
                },
                PaletteColor {
                    r: 255,
                    g: 255,
                    b: 255,
                    weight: 0.5,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        let near = p.nearest_color(240, 240, 240).expect("should find nearest");
        assert_eq!((near.r, near.g, near.b), (255, 255, 255));
    }

    #[test]
    fn test_nearest_color_empty_palette() {
        let p = ColorPalette {
            colors: Vec::new(),
            algorithm: PaletteAlgorithm::MedianCut,
        };
        assert!(p.nearest_color(100, 100, 100).is_none());
    }

    // ── quantize_image ────────────────────────────────────────────────────

    #[test]
    fn test_quantize_image_replaces_pixels() {
        let p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 0,
                    weight: 0.5,
                },
                PaletteColor {
                    r: 255,
                    g: 255,
                    b: 255,
                    weight: 0.5,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        let input = vec![10u8, 10, 10, 245, 245, 245];
        let out = p.quantize_image(&input);
        assert_eq!(out[0..3], [0, 0, 0]);
        assert_eq!(out[3..6], [255, 255, 255]);
    }

    // ── dominant_color ────────────────────────────────────────────────────

    #[test]
    fn test_dominant_color() {
        let p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 255,
                    g: 0,
                    b: 0,
                    weight: 0.7,
                },
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 255,
                    weight: 0.3,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        let dom = p.dominant_color().expect("should have dominant");
        assert_eq!((dom.r, dom.g, dom.b), (255, 0, 0));
    }

    // ── sort_by_weight / sort_by_luminance ────────────────────────────────

    #[test]
    fn test_sort_by_weight() {
        let mut p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 0,
                    weight: 0.2,
                },
                PaletteColor {
                    r: 128,
                    g: 128,
                    b: 128,
                    weight: 0.5,
                },
                PaletteColor {
                    r: 255,
                    g: 255,
                    b: 255,
                    weight: 0.3,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        p.sort_by_weight();
        assert!(p.colors[0].weight >= p.colors[1].weight);
        assert!(p.colors[1].weight >= p.colors[2].weight);
    }

    #[test]
    fn test_sort_by_luminance() {
        let mut p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 255,
                    g: 255,
                    b: 255,
                    weight: 0.33,
                },
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 0,
                    weight: 0.33,
                },
                PaletteColor {
                    r: 128,
                    g: 128,
                    b: 128,
                    weight: 0.34,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        p.sort_by_luminance();
        assert!(p.colors[0].luminance() <= p.colors[1].luminance());
        assert!(p.colors[1].luminance() <= p.colors[2].luminance());
    }

    // ── diversity_score ───────────────────────────────────────────────────

    #[test]
    fn test_diversity_score_identical_colors() {
        let p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 100,
                    g: 100,
                    b: 100,
                    weight: 0.5,
                },
                PaletteColor {
                    r: 100,
                    g: 100,
                    b: 100,
                    weight: 0.5,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        assert!(p.diversity_score() < 0.01);
    }

    #[test]
    fn test_diversity_score_black_white() {
        let p = ColorPalette {
            colors: vec![
                PaletteColor {
                    r: 0,
                    g: 0,
                    b: 0,
                    weight: 0.5,
                },
                PaletteColor {
                    r: 255,
                    g: 255,
                    b: 255,
                    weight: 0.5,
                },
            ],
            algorithm: PaletteAlgorithm::MedianCut,
        };
        let score = p.diversity_score();
        assert!((score - 1.0).abs() < 0.01, "score={score}");
    }

    #[test]
    fn test_diversity_score_empty() {
        let p = ColorPalette {
            colors: Vec::new(),
            algorithm: PaletteAlgorithm::MedianCut,
        };
        assert_eq!(p.diversity_score(), 0.0);
    }
}
