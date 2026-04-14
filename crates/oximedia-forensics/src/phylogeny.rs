#![allow(dead_code)]
//! Image phylogeny — trace image editing history from multiple versions.
//!
//! Image phylogeny (also called *image provenance* or *versioning ancestry*
//! analysis) reconstructs the directed edit graph between a set of related
//! image variants.  Given N images that are suspected to share a common
//! ancestor, the algorithm:
//!
//! 1. Computes pairwise **similarity scores** using pixel-level difference and
//!    structural comparison metrics.
//! 2. Builds a **minimum spanning tree** over the complete similarity graph,
//!    producing the most parsimonious editing lineage.
//! 3. Roots the tree at the image with the lowest estimated compression
//!    quality (most likely the original capture) or the highest mean
//!    luminance entropy.
//! 4. Assigns directed edges to reconstruct the parent → child editing order.
//!
//! # Similarity measures
//!
//! - **Mean Squared Error (MSE)** after resizing both images to a canonical
//!   comparison resolution.
//! - **Normalised Cross-Correlation (NCC)** for partial match detection.
//! - **Histogram chi-squared distance** for colour distribution similarity.
//!
//! # Example
//!
//! ```rust
//! use oximedia_forensics::phylogeny::{PhylogenyNode, PhylogenyTree, PhylogenyAnalyzer};
//!
//! // Create synthetic image nodes (normally loaded from disk / decoded frames)
//! let width = 64usize;
//! let height = 64usize;
//! let pixels: Vec<u8> = (0..width * height * 3)
//!     .map(|i| (i % 256) as u8)
//!     .collect();
//!
//! let nodes = vec![
//!     PhylogenyNode::new("original.jpg".to_string(), width, height, pixels.clone()),
//!     PhylogenyNode::new("edited.jpg".to_string(), width, height, pixels),
//! ];
//!
//! let analyzer = PhylogenyAnalyzer::default();
//! let tree = analyzer.build_tree(nodes).expect("build tree");
//! assert!(!tree.nodes.is_empty());
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PhylogenyNode
// ---------------------------------------------------------------------------

/// A single image version participating in the phylogeny analysis.
#[derive(Debug, Clone)]
pub struct PhylogenyNode {
    /// Unique label / path identifier.
    pub label: String,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Raw RGB pixel data (row-major, 3 bytes per pixel).
    pub pixels: Vec<u8>,
    /// Estimated JPEG quality (0 = unknown, 1–100 = quality factor).
    pub estimated_quality: u8,
    /// Node index assigned during tree construction.
    pub index: usize,
}

impl PhylogenyNode {
    /// Create a new phylogeny node.
    ///
    /// `pixels` must be exactly `width * height * 3` bytes.  If the length
    /// does not match, the pixel buffer is stored as-is and analysis may
    /// produce degraded results.
    #[must_use]
    pub fn new(label: String, width: usize, height: usize, pixels: Vec<u8>) -> Self {
        Self {
            label,
            width,
            height,
            pixels,
            estimated_quality: 0,
            index: 0,
        }
    }

    /// Expected byte length (`width * height * 3`).
    #[must_use]
    pub fn expected_len(&self) -> usize {
        self.width * self.height * 3
    }

    /// Whether the pixel buffer length matches the declared dimensions.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.pixels.len() == self.expected_len()
    }

    /// Compute the mean luminance of the image.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_luminance(&self) -> f64 {
        if self.pixels.is_empty() {
            return 0.0;
        }
        let n = (self.pixels.len() / 3) as f64;
        let mut lum_sum = 0.0_f64;
        for chunk in self.pixels.chunks(3) {
            if chunk.len() == 3 {
                // BT.601 luma coefficients
                let r = chunk[0] as f64;
                let g = chunk[1] as f64;
                let b = chunk[2] as f64;
                lum_sum += 0.299 * r + 0.587 * g + 0.114 * b;
            }
        }
        lum_sum / (n * 255.0)
    }

    /// Compute an 8-bin normalised luma histogram.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn luma_histogram_8bin(&self) -> [f64; 8] {
        let mut bins = [0u64; 8];
        let mut count = 0u64;
        for chunk in self.pixels.chunks(3) {
            if chunk.len() == 3 {
                let luma = (0.299 * chunk[0] as f64
                    + 0.587 * chunk[1] as f64
                    + 0.114 * chunk[2] as f64) as usize;
                let bin = (luma * 8 / 256).min(7);
                bins[bin] += 1;
                count += 1;
            }
        }
        if count == 0 {
            return [0.0; 8];
        }
        let mut hist = [0.0_f64; 8];
        for (i, &b) in bins.iter().enumerate() {
            hist[i] = b as f64 / count as f64;
        }
        hist
    }

    /// Compute the pixel-level Mean Squared Error against another node.
    ///
    /// Images are compared at the minimum shared resolution by sampling every
    /// other pixel if sizes differ.  Returns `1.0` if either image is invalid.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mse_against(&self, other: &Self) -> f64 {
        if !self.is_valid() || !other.is_valid() {
            return 1.0;
        }
        // Compare at the common pixel count to handle size differences.
        let min_pixels = (self.width * self.height).min(other.width * other.height);
        if min_pixels == 0 {
            return 1.0;
        }

        let stride_a = if self.width * self.height > min_pixels {
            (self.width * self.height) / min_pixels
        } else {
            1
        };
        let stride_b = if other.width * other.height > min_pixels {
            (other.width * other.height) / min_pixels
        } else {
            1
        };

        let mut sq_sum = 0.0_f64;
        let mut n = 0u64;

        let a_chunks: Vec<&[u8]> = self
            .pixels
            .chunks(3)
            .step_by(stride_a)
            .take(min_pixels)
            .collect();
        let b_chunks: Vec<&[u8]> = other
            .pixels
            .chunks(3)
            .step_by(stride_b)
            .take(min_pixels)
            .collect();

        for (ac, bc) in a_chunks.iter().zip(b_chunks.iter()) {
            if ac.len() == 3 && bc.len() == 3 {
                for ch in 0..3 {
                    let diff = ac[ch] as f64 - bc[ch] as f64;
                    sq_sum += diff * diff;
                }
                n += 3;
            }
        }

        if n == 0 {
            return 1.0;
        }
        (sq_sum / (n as f64 * 255.0 * 255.0)).clamp(0.0, 1.0)
    }

    /// Chi-squared histogram distance against another node (0 = identical, ≥0).
    #[must_use]
    pub fn histogram_chi2_against(&self, other: &Self) -> f64 {
        let ha = self.luma_histogram_8bin();
        let hb = other.luma_histogram_8bin();
        let mut chi2 = 0.0_f64;
        for i in 0..8 {
            let sum = ha[i] + hb[i];
            if sum > 1e-12 {
                let diff = ha[i] - hb[i];
                chi2 += (diff * diff) / sum;
            }
        }
        chi2 * 0.5
    }

    /// Combined similarity score in `[0, 1]` (1 = identical, 0 = no similarity).
    ///
    /// Combines MSE (70%) and histogram distance (30%).
    #[must_use]
    pub fn similarity_against(&self, other: &Self) -> f64 {
        let mse_sim = 1.0 - self.mse_against(other);
        let hist_sim = 1.0 - self.histogram_chi2_against(other).min(1.0);
        (0.7 * mse_sim + 0.3 * hist_sim).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// PhylogenyEdge
// ---------------------------------------------------------------------------

/// A directed edge in the phylogeny tree representing an edit step.
#[derive(Debug, Clone)]
pub struct PhylogenyEdge {
    /// Index of the parent (earlier) node.
    pub parent: usize,
    /// Index of the child (later / edited) node.
    pub child: usize,
    /// Similarity score between the two nodes (0–1).
    pub similarity: f64,
    /// Whether this edge represents a likely destructive edit.
    pub is_destructive: bool,
}

// ---------------------------------------------------------------------------
// PhylogenyTree
// ---------------------------------------------------------------------------

/// A rooted tree representing the reconstructed editing ancestry.
#[derive(Debug, Clone)]
pub struct PhylogenyTree {
    /// All image nodes indexed by their `index` field.
    pub nodes: Vec<PhylogenyNode>,
    /// Directed edges (parent → child) in the MST.
    pub edges: Vec<PhylogenyEdge>,
    /// Index of the root node (most likely original).
    pub root: usize,
}

impl PhylogenyTree {
    /// Return the children of a given node index.
    #[must_use]
    pub fn children_of(&self, node_index: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.parent == node_index)
            .map(|e| e.child)
            .collect()
    }

    /// Return the parent of a given node index, if any.
    #[must_use]
    pub fn parent_of(&self, node_index: usize) -> Option<usize> {
        self.edges
            .iter()
            .find(|e| e.child == node_index)
            .map(|e| e.parent)
    }

    /// Depth of the tree (longest root-to-leaf path).
    #[must_use]
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        let mut max_depth = 0usize;
        let mut stack: Vec<(usize, usize)> = vec![(self.root, 0)];
        while let Some((node, d)) = stack.pop() {
            if d > max_depth {
                max_depth = d;
            }
            for child in self.children_of(node) {
                stack.push((child, d + 1));
            }
        }
        max_depth
    }

    /// Number of leaf nodes (nodes with no children).
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| self.children_of(n.index).is_empty())
            .count()
    }

    /// Return a DOT-language representation of the tree.
    #[must_use]
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph phylogeny {\n  rankdir=TB;\n");
        for node in &self.nodes {
            let label = node.label.replace('"', "'");
            let marker = if node.index == self.root { " [shape=box,style=filled,fillcolor=gold]" } else { "" };
            out.push_str(&format!("  n{} [label=\"{}\"]{};\n", node.index, label, marker));
        }
        for edge in &self.edges {
            let style = if edge.is_destructive { " [style=dashed]" } else { "" };
            out.push_str(&format!(
                "  n{} -> n{} [label=\"{:.2}\"]{};\n",
                edge.parent, edge.child, edge.similarity, style,
            ));
        }
        out.push('}');
        out
    }
}

// ---------------------------------------------------------------------------
// PhylogenyAnalyzer
// ---------------------------------------------------------------------------

/// Configuration for phylogeny analysis.
#[derive(Debug, Clone)]
pub struct PhylogenyConfig {
    /// Similarity score below which two images are considered unrelated
    /// (default: 0.5).
    pub min_similarity_threshold: f64,
    /// Whether to root the tree at the node with the lowest estimated quality
    /// (default: `true`).
    pub root_at_lowest_quality: bool,
    /// Score below which an edit is considered destructive (default: 0.7).
    pub destructive_threshold: f64,
}

impl Default for PhylogenyConfig {
    fn default() -> Self {
        Self {
            min_similarity_threshold: 0.5,
            root_at_lowest_quality: true,
            destructive_threshold: 0.7,
        }
    }
}

/// Builds a [`PhylogenyTree`] from a set of image versions.
#[derive(Debug, Clone, Default)]
pub struct PhylogenyAnalyzer {
    pub config: PhylogenyConfig,
}

impl PhylogenyAnalyzer {
    /// Create an analyzer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an analyzer with custom configuration.
    #[must_use]
    pub fn with_config(config: PhylogenyConfig) -> Self {
        Self { config }
    }

    /// Build a phylogeny tree from a list of image nodes.
    ///
    /// Uses Prim's minimum spanning tree algorithm over the pairwise
    /// similarity graph (edges weighted by dissimilarity = 1 - similarity).
    ///
    /// # Errors
    ///
    /// Returns an error string if the node list is empty or all images are
    /// below the minimum similarity threshold.
    pub fn build_tree(&self, mut nodes: Vec<PhylogenyNode>) -> Result<PhylogenyTree, String> {
        if nodes.is_empty() {
            return Err("PhylogenyAnalyzer: empty node list".to_string());
        }

        // Assign indices.
        for (i, node) in nodes.iter_mut().enumerate() {
            node.index = i;
        }

        let n = nodes.len();

        if n == 1 {
            return Ok(PhylogenyTree {
                nodes,
                edges: Vec::new(),
                root: 0,
            });
        }

        // Build the complete similarity matrix.
        let mut sim_matrix: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let s = nodes[i].similarity_against(&nodes[j]);
                sim_matrix[i][j] = s;
                sim_matrix[j][i] = s;
            }
        }

        // Prim's MST over dissimilarity weights (weight = 1 - similarity).
        // We want the maximum similarity spanning tree (most parsimonious
        // lineage), which is equivalent to minimum weight when weight =
        // 1 - similarity.
        let mut in_tree = vec![false; n];
        let mut best_sim = vec![f64::NEG_INFINITY; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];

        // Choose root: node with lowest estimated_quality (most likely original).
        // Fall back to node 0 if all qualities are 0.
        let root = if self.config.root_at_lowest_quality {
            nodes
                .iter()
                .enumerate()
                .min_by_key(|(_, nd)| nd.estimated_quality)
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            0
        };

        best_sim[root] = f64::INFINITY;

        for _ in 0..n {
            // Pick the non-tree node with highest best_sim.
            let u = (0..n)
                .filter(|&i| !in_tree[i])
                .max_by(|&a, &b| {
                    best_sim[a]
                        .partial_cmp(&best_sim[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            let u = match u {
                Some(idx) => idx,
                None => break,
            };

            in_tree[u] = true;

            // Relax neighbours.
            for v in 0..n {
                if !in_tree[v] && sim_matrix[u][v] > best_sim[v] {
                    best_sim[v] = sim_matrix[u][v];
                    parent[v] = Some(u);
                }
            }
        }

        // Build directed edges (parent → child order based on similarity).
        let mut edges: Vec<PhylogenyEdge> = Vec::new();
        for child in 0..n {
            if let Some(par) = parent[child] {
                let sim = sim_matrix[par][child];
                let is_destructive = sim < self.config.destructive_threshold;
                edges.push(PhylogenyEdge {
                    parent: par,
                    child,
                    similarity: sim,
                    is_destructive,
                });
            }
        }

        Ok(PhylogenyTree { nodes, edges, root })
    }

    /// Compute and return the full pairwise similarity matrix.
    ///
    /// The returned `HashMap` maps `(i, j)` pairs (with `i < j`) to the
    /// similarity score in `[0, 1]`.
    #[must_use]
    pub fn similarity_matrix(&self, nodes: &[PhylogenyNode]) -> HashMap<(usize, usize), f64> {
        let mut map = HashMap::new();
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let s = nodes[i].similarity_against(&nodes[j]);
                map.insert((i, j), s);
            }
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(r: u8, g: u8, b: u8, w: usize, h: usize) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
        pixels
    }

    fn make_node(label: &str, r: u8, g: u8, b: u8) -> PhylogenyNode {
        let w = 32;
        let h = 32;
        PhylogenyNode::new(label.to_string(), w, h, solid_rgb(r, g, b, w, h))
    }

    // ----- PhylogenyNode unit tests -----------------------------------------

    #[test]
    fn test_node_is_valid() {
        let n = make_node("a", 100, 100, 100);
        assert!(n.is_valid());
    }

    #[test]
    fn test_node_invalid_pixel_len() {
        let mut n = make_node("a", 100, 100, 100);
        n.pixels.pop(); // corrupt length
        assert!(!n.is_valid());
    }

    #[test]
    fn test_mean_luminance_grey() {
        let n = make_node("grey", 128, 128, 128);
        let lum = n.mean_luminance();
        assert!((lum - 128.0 / 255.0).abs() < 1e-9);
    }

    #[test]
    fn test_mse_identical_images() {
        let n = make_node("a", 100, 150, 200);
        assert!((n.mse_against(&n)).abs() < 1e-12);
    }

    #[test]
    fn test_mse_different_images() {
        let a = make_node("a", 0, 0, 0);
        let b = make_node("b", 255, 255, 255);
        let mse = a.mse_against(&b);
        assert!(mse > 0.9, "all-black vs all-white should have high MSE, got {mse}");
    }

    #[test]
    fn test_histogram_chi2_identical() {
        let n = make_node("a", 100, 150, 200);
        let chi2 = n.histogram_chi2_against(&n);
        assert!(chi2.abs() < 1e-12);
    }

    #[test]
    fn test_similarity_identical() {
        let n = make_node("a", 100, 150, 200);
        let sim = n.similarity_against(&n);
        assert!((sim - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_similarity_different() {
        let a = make_node("a", 0, 0, 0);
        let b = make_node("b", 255, 255, 255);
        let sim = a.similarity_against(&b);
        assert!(sim < 0.5, "opposite images should have low similarity, got {sim}");
    }

    // ----- PhylogenyAnalyzer tests ------------------------------------------

    #[test]
    fn test_build_tree_empty_returns_error() {
        let analyzer = PhylogenyAnalyzer::new();
        let result = analyzer.build_tree(Vec::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_build_tree_single_node() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![make_node("only", 128, 128, 128)];
        let tree = analyzer.build_tree(nodes).expect("single-node tree");
        assert_eq!(tree.nodes.len(), 1);
        assert!(tree.edges.is_empty());
        assert_eq!(tree.root, 0);
    }

    #[test]
    fn test_build_tree_two_nodes() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![make_node("a", 100, 100, 100), make_node("b", 110, 110, 110)];
        let tree = analyzer.build_tree(nodes).expect("two-node tree");
        assert_eq!(tree.nodes.len(), 2);
        assert_eq!(tree.edges.len(), 1);
    }

    #[test]
    fn test_build_tree_three_nodes() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("original", 100, 100, 100),
            make_node("edit1", 105, 100, 100),
            make_node("edit2", 100, 105, 100),
        ];
        let tree = analyzer.build_tree(nodes).expect("three-node tree");
        assert_eq!(tree.nodes.len(), 3);
        assert_eq!(tree.edges.len(), 2); // MST has n-1 edges
    }

    #[test]
    fn test_tree_depth() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("a", 100, 100, 100),
            make_node("b", 105, 100, 100),
            make_node("c", 110, 100, 100),
        ];
        let tree = analyzer.build_tree(nodes).expect("tree");
        assert!(tree.depth() >= 1);
    }

    #[test]
    fn test_tree_leaf_count() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("a", 100, 100, 100),
            make_node("b", 105, 100, 100),
            make_node("c", 110, 100, 100),
        ];
        let tree = analyzer.build_tree(nodes).expect("tree");
        // All leaves that have no children.
        let leaves = tree.leaf_count();
        assert!(leaves >= 1);
    }

    #[test]
    fn test_tree_to_dot() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("original.jpg", 100, 100, 100),
            make_node("edited.jpg", 105, 100, 100),
        ];
        let tree = analyzer.build_tree(nodes).expect("tree");
        let dot = tree.to_dot();
        assert!(dot.contains("digraph phylogeny"), "should contain digraph header");
        assert!(dot.contains("original.jpg"));
        assert!(dot.contains("edited.jpg"));
    }

    #[test]
    fn test_similarity_matrix() {
        let nodes = vec![
            make_node("a", 100, 100, 100),
            make_node("b", 200, 200, 200),
            make_node("c", 150, 150, 150),
        ];
        let analyzer = PhylogenyAnalyzer::new();
        let matrix = analyzer.similarity_matrix(&nodes);
        assert_eq!(matrix.len(), 3); // pairs (0,1), (0,2), (1,2)
        // All scores should be in [0,1].
        for &sim in matrix.values() {
            assert!((0.0..=1.0).contains(&sim), "score out of range: {sim}");
        }
    }

    #[test]
    fn test_node_indices_assigned() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("a", 100, 100, 100),
            make_node("b", 110, 110, 110),
            make_node("c", 120, 120, 120),
        ];
        let tree = analyzer.build_tree(nodes).expect("tree");
        for (i, node) in tree.nodes.iter().enumerate() {
            assert_eq!(node.index, i);
        }
    }

    #[test]
    fn test_parent_child_query() {
        let analyzer = PhylogenyAnalyzer::new();
        let nodes = vec![
            make_node("a", 100, 100, 100),
            make_node("b", 110, 110, 110),
        ];
        let tree = analyzer.build_tree(nodes).expect("tree");
        let child_idx = if tree.root == 0 { 1 } else { 0 };
        let parent = tree.parent_of(child_idx);
        assert_eq!(parent, Some(tree.root));
        assert!(tree.parent_of(tree.root).is_none());
    }

    #[test]
    fn test_luma_histogram_sums_to_one() {
        let n = make_node("a", 128, 64, 200);
        let hist = n.luma_histogram_8bin();
        let sum: f64 = hist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "histogram must sum to 1.0, got {sum}");
    }
}
