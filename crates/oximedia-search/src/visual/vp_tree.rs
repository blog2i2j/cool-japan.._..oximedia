//! VP-tree (vantage-point tree) specialised for f32 vector similarity search.
//!
//! This module implements a VP-tree where each item is a dense f32 feature
//! vector identified by a `usize` ID. Unlike the generic `vp_tree` module at
//! the crate root (which is parameterised over any `T`), this module provides
//! a concrete, allocation-efficient structure with pre-computed Euclidean
//! distances and a richer public API (KNN + radius search).
//!
//! # Algorithm overview
//!
//! A VP-tree partitions a metric space by repeatedly choosing a *vantage
//! point*, computing distances from every remaining point to it, and splitting
//! at the **median** distance: points within the median radius go into the
//! *inner* sub-tree; points beyond go into the *outer* sub-tree. This yields
//! O(log n) average-case search with correct triangle-inequality pruning.
//!
//! # Example
//!
//! ```
//! use oximedia_search::visual::vp_tree::{FloatVpTree, euclidean_dist};
//!
//! let points: Vec<(usize, Vec<f32>)> = vec![
//!     (0, vec![0.0, 0.0]),
//!     (1, vec![1.0, 0.0]),
//!     (2, vec![0.0, 1.0]),
//!     (3, vec![5.0, 5.0]),
//! ];
//!
//! let tree = FloatVpTree::build(points);
//! let knn = tree.search_knn(&[0.1, 0.1], 2);
//! assert_eq!(knn.len(), 2);
//! assert_eq!(knn[0].0, 0); // closest to origin
//! ```

#![allow(dead_code)]

use std::collections::BinaryHeap;

// ─────────────────────────────────────────────────────────────────────────────
// Distance function
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Euclidean (L2) distance between two f32 slices.
///
/// Both slices must have the same length; if they differ, the shorter length
/// is used (excess dimensions are ignored).
///
/// # Example
///
/// ```
/// use oximedia_search::visual::vp_tree::euclidean_dist;
///
/// let a = [3.0_f32, 4.0];
/// let b = [0.0_f32, 0.0];
/// assert!((euclidean_dist(&a, &b) - 5.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn euclidean_dist(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let sum_sq: f32 = (0..len).map(|i| {
        let diff = a[i] - b[i];
        diff * diff
    }).sum();
    sum_sq.sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal tree node
// ─────────────────────────────────────────────────────────────────────────────

/// A single node in the VP-tree.
///
/// Each internal node holds a vantage-point (index into the flat `points`
/// store), the median partition threshold, and left/right subtrees.
/// Leaf nodes are represented by having `left` and `right` both `None` and
/// `threshold` set to `0.0`.
pub struct VpNode {
    /// ID of the vantage point.
    pub point_id: usize,
    /// Index into the flat point store.
    store_idx: usize,
    /// Median distance threshold: points with `dist ≤ threshold` go left.
    pub threshold: f32,
    /// Inner subtree (distance ≤ threshold from vantage).
    pub left: Option<Box<VpNode>>,
    /// Outer subtree (distance > threshold from vantage).
    pub right: Option<Box<VpNode>>,
}

impl VpNode {
    fn leaf(point_id: usize, store_idx: usize) -> Self {
        Self {
            point_id,
            store_idx,
            threshold: 0.0,
            left: None,
            right: None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Max-heap entry for KNN search
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
struct KnnEntry {
    dist: f32,
    id: usize,
}

impl Eq for KnnEntry {}

impl PartialOrd for KnnEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KnnEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Largest distance first (max-heap so we can evict the farthest).
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FloatVpTree
// ─────────────────────────────────────────────────────────────────────────────

/// A VP-tree storing `(id, Vec<f32>)` point pairs, enabling sub-linear KNN
/// and radius searches using Euclidean distance.
pub struct FloatVpTree {
    /// Flat storage: index corresponds to internal store index.
    points: Vec<(usize, Vec<f32>)>,
    /// Root of the VP-tree (or `None` for an empty tree).
    root: Option<Box<VpNode>>,
}

impl FloatVpTree {
    /// Build a VP-tree from `(id, vector)` pairs.
    ///
    /// The vantage point at each level is chosen as the *first* item in the
    /// current partition (a simple deterministic strategy). Distances are
    /// computed, sorted, and the median is used as the split threshold.
    ///
    /// # Complexity
    ///
    /// O(n log n) construction; O(log n) average-case query.
    #[must_use]
    pub fn build(points: Vec<(usize, Vec<f32>)>) -> Self {
        if points.is_empty() {
            return Self {
                points: Vec::new(),
                root: None,
            };
        }

        // Build flat store; `indices` references into this store.
        let mut indices: Vec<usize> = (0..points.len()).collect();
        let root = Box::new(Self::build_subtree(&points, &mut indices));

        Self {
            points,
            root: Some(root),
        }
    }

    /// Return `true` if the tree contains no points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return the number of points in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Find the `k` nearest neighbours to `query`, returning `(id, distance)`
    /// pairs sorted by distance ascending.
    ///
    /// Returns fewer than `k` results when the tree has fewer than `k` points.
    #[must_use]
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }
        let effective_k = k.min(self.points.len());
        let mut heap: BinaryHeap<KnnEntry> = BinaryHeap::with_capacity(effective_k + 1);
        let mut tau = f32::INFINITY;

        if let Some(ref root) = self.root {
            Self::knn_search(root, &self.points, query, effective_k, &mut heap, &mut tau);
        }

        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|e| (e.id, e.dist))
            .collect();
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    /// Find all points within Euclidean `radius` of `query`.
    ///
    /// Returns `(id, distance)` pairs sorted by distance ascending.
    #[must_use]
    pub fn search_radius(&self, query: &[f32], radius: f32) -> Vec<(usize, f32)> {
        if self.is_empty() {
            return Vec::new();
        }
        let mut results: Vec<(usize, f32)> = Vec::new();
        if let Some(ref root) = self.root {
            Self::radius_search(root, &self.points, query, radius, &mut results);
        }
        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results
    }

    // ── Internal construction ─────────────────────────────────────────────

    fn build_subtree(
        all_points: &[(usize, Vec<f32>)],
        indices: &mut [usize],
    ) -> VpNode {
        match indices.len() {
            0 => unreachable!("build_subtree called with empty slice"),
            1 => {
                let store_idx = indices[0];
                let (id, _) = &all_points[store_idx];
                VpNode::leaf(*id, store_idx)
            }
            _ => {
                // Use first index as vantage point.
                let vp_store = indices[0];
                let (vp_id, vp_vec) = &all_points[vp_store];
                let vp_id = *vp_id;
                let rest = &mut indices[1..];
                let n = rest.len();

                // Compute distances from vantage to every other point.
                let mut dists: Vec<(usize, f32)> = rest
                    .iter()
                    .map(|&si| {
                        let (_, v) = &all_points[si];
                        (si, euclidean_dist(vp_vec, v))
                    })
                    .collect();

                // Sort by distance to find the median threshold.
                dists.sort_by(|a, b| a.1.total_cmp(&b.1));
                let mid = n / 2;
                let threshold = dists.get(mid.saturating_sub(1))
                    .map(|e| e.1)
                    .unwrap_or(0.0);

                let mut inner_indices: Vec<usize> =
                    dists[..mid].iter().map(|e| e.0).collect();
                let mut outer_indices: Vec<usize> =
                    dists[mid..].iter().map(|e| e.0).collect();

                let left = if inner_indices.is_empty() {
                    None
                } else {
                    Some(Box::new(Self::build_subtree(all_points, &mut inner_indices)))
                };
                let right = if outer_indices.is_empty() {
                    None
                } else {
                    Some(Box::new(Self::build_subtree(all_points, &mut outer_indices)))
                };

                VpNode {
                    point_id: vp_id,
                    store_idx: vp_store,
                    threshold,
                    left,
                    right,
                }
            }
        }
    }

    // ── Internal KNN search ───────────────────────────────────────────────

    fn knn_search(
        node: &VpNode,
        all_points: &[(usize, Vec<f32>)],
        query: &[f32],
        k: usize,
        heap: &mut BinaryHeap<KnnEntry>,
        tau: &mut f32,
    ) {
        let (_, vp_vec) = &all_points[node.store_idx];
        let d = euclidean_dist(query, vp_vec);

        // Try to add the vantage point itself to the heap.
        Self::push_candidate(heap, k, node.point_id, d, tau);

        // Choose which subtree to explore first (closer first).
        if d <= node.threshold {
            // Query is inside the ball — visit inner first.
            if let Some(ref inner) = node.left {
                if d - *tau <= node.threshold {
                    Self::knn_search(inner, all_points, query, k, heap, tau);
                }
            }
            if let Some(ref outer) = node.right {
                if d + *tau > node.threshold {
                    Self::knn_search(outer, all_points, query, k, heap, tau);
                }
            }
        } else {
            // Query is outside the ball — visit outer first.
            if let Some(ref outer) = node.right {
                if d + *tau > node.threshold {
                    Self::knn_search(outer, all_points, query, k, heap, tau);
                }
            }
            if let Some(ref inner) = node.left {
                if d - *tau <= node.threshold {
                    Self::knn_search(inner, all_points, query, k, heap, tau);
                }
            }
        }
    }

    fn push_candidate(
        heap: &mut BinaryHeap<KnnEntry>,
        k: usize,
        id: usize,
        dist: f32,
        tau: &mut f32,
    ) {
        if heap.len() < k {
            heap.push(KnnEntry { dist, id });
            if heap.len() == k {
                *tau = heap.peek().map(|e| e.dist).unwrap_or(f32::INFINITY);
            }
        } else if dist < *tau {
            heap.pop();
            heap.push(KnnEntry { dist, id });
            *tau = heap.peek().map(|e| e.dist).unwrap_or(f32::INFINITY);
        }
    }

    // ── Internal radius search ─────────────────────────────────────────────

    fn radius_search(
        node: &VpNode,
        all_points: &[(usize, Vec<f32>)],
        query: &[f32],
        radius: f32,
        results: &mut Vec<(usize, f32)>,
    ) {
        let (_, vp_vec) = &all_points[node.store_idx];
        let d = euclidean_dist(query, vp_vec);

        if d <= radius {
            results.push((node.point_id, d));
        }

        // Inner ball covers [0, threshold] from vantage.
        // Intersects query ball if d - radius <= threshold.
        if let Some(ref inner) = node.left {
            if d - radius <= node.threshold {
                Self::radius_search(inner, all_points, query, radius, results);
            }
        }

        // Outer shell covers (threshold, ∞).
        // Intersects query ball if d + radius > threshold.
        if let Some(ref outer) = node.right {
            if d + radius > node.threshold {
                Self::radius_search(outer, all_points, query, radius, results);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(raw: &[(usize, &[f32])]) -> Vec<(usize, Vec<f32>)> {
        raw.iter()
            .map(|(id, v)| (*id, v.to_vec()))
            .collect()
    }

    fn brute_knn(points: &[(usize, Vec<f32>)], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = points
            .iter()
            .map(|(id, v)| (*id, euclidean_dist(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists.truncate(k);
        dists
    }

    // ── euclidean_dist ────────────────────────────────────────────────────

    #[test]
    fn test_euclidean_dist_zero() {
        let a = [1.0_f32, 2.0, 3.0];
        assert!((euclidean_dist(&a, &a)).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_dist_known() {
        let a = [3.0_f32, 4.0];
        let b = [0.0_f32, 0.0];
        assert!((euclidean_dist(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_euclidean_dist_unit_vector() {
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        // Distance between two unit vectors at 90 degrees = sqrt(2)
        let d = euclidean_dist(&a, &b);
        assert!((d - std::f32::consts::SQRT_2).abs() < 1e-5, "got {}", d);
    }

    // ── Empty tree ────────────────────────────────────────────────────────

    #[test]
    fn test_empty_tree() {
        let tree = FloatVpTree::build(vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.search_knn(&[0.0], 5).is_empty());
        assert!(tree.search_radius(&[0.0], 100.0).is_empty());
    }

    #[test]
    fn test_k_zero_returns_empty() {
        let tree = FloatVpTree::build(pts(&[(0, &[1.0, 2.0])]));
        assert!(tree.search_knn(&[1.0, 2.0], 0).is_empty());
    }

    // ── Single point ──────────────────────────────────────────────────────

    #[test]
    fn test_single_point_knn() {
        let tree = FloatVpTree::build(pts(&[(42, &[3.0, 4.0])]));
        let knn = tree.search_knn(&[3.0, 4.0], 1);
        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].0, 42);
        assert!(knn[0].1.abs() < 1e-5);
    }

    #[test]
    fn test_single_point_radius_hit() {
        let tree = FloatVpTree::build(pts(&[(7, &[0.0, 0.0])]));
        let res = tree.search_radius(&[1.0, 0.0], 2.0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, 7);
    }

    #[test]
    fn test_single_point_radius_miss() {
        let tree = FloatVpTree::build(pts(&[(7, &[10.0, 10.0])]));
        let res = tree.search_radius(&[0.0, 0.0], 1.0);
        assert!(res.is_empty());
    }

    // ── Multi-point correctness ───────────────────────────────────────────

    #[test]
    fn test_knn_correctness_2d() {
        let raw: Vec<(usize, Vec<f32>)> = (0..20)
            .map(|i| (i, vec![i as f32, 0.0]))
            .collect();
        let tree = FloatVpTree::build(raw.clone());
        let query = [7.5_f32, 0.0];
        let k = 3;

        let vp_res = tree.search_knn(&query, k);
        let bf_res = brute_knn(&raw, &query, k);

        assert_eq!(vp_res.len(), k);
        for (vp, bf) in vp_res.iter().zip(bf_res.iter()) {
            assert!(
                (vp.1 - bf.1).abs() < 1e-4,
                "vp dist {} != bf dist {}",
                vp.1,
                bf.1
            );
        }
    }

    #[test]
    fn test_knn_exceeds_tree_size() {
        let raw = pts(&[(0, &[0.0]), (1, &[1.0]), (2, &[2.0])]);
        let tree = FloatVpTree::build(raw);
        let res = tree.search_knn(&[1.0], 100);
        assert_eq!(res.len(), 3, "should clamp to tree size");
    }

    #[test]
    fn test_knn_sorted_ascending() {
        let raw: Vec<(usize, Vec<f32>)> = (0..10)
            .map(|i| (i, vec![i as f32 * 2.0]))
            .collect();
        let tree = FloatVpTree::build(raw);
        let query = [5.0_f32];
        let res = tree.search_knn(&query, 5);
        for i in 1..res.len() {
            assert!(res[i - 1].1 <= res[i].1, "not sorted at index {}", i);
        }
    }

    #[test]
    fn test_radius_search_all_within() {
        let raw = pts(&[
            (0, &[0.0, 0.0]),
            (1, &[1.0, 0.0]),
            (2, &[0.0, 1.0]),
            (3, &[-1.0, 0.0]),
        ]);
        let tree = FloatVpTree::build(raw);
        // All points are at distance 0 or 1 from origin — radius 1.5 should capture all.
        let res = tree.search_radius(&[0.0, 0.0], 1.5);
        assert_eq!(res.len(), 4);
    }

    #[test]
    fn test_radius_search_none_within() {
        let raw = pts(&[
            (0, &[10.0, 10.0]),
            (1, &[20.0, 20.0]),
        ]);
        let tree = FloatVpTree::build(raw);
        let res = tree.search_radius(&[0.0, 0.0], 1.0);
        assert!(res.is_empty());
    }

    #[test]
    fn test_radius_search_sorted_ascending() {
        let raw: Vec<(usize, Vec<f32>)> = (0..8)
            .map(|i| (i, vec![i as f32]))
            .collect();
        let tree = FloatVpTree::build(raw);
        let res = tree.search_radius(&[3.5], 4.0);
        for i in 1..res.len() {
            assert!(res[i - 1].1 <= res[i].1, "not sorted at index {}", i);
        }
    }

    #[test]
    fn test_len_and_is_empty() {
        let tree = FloatVpTree::build(pts(&[
            (0, &[1.0]),
            (1, &[2.0]),
            (2, &[3.0]),
        ]));
        assert_eq!(tree.len(), 3);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_knn_brute_force_agreement_high_dim() {
        // 4-D points.
        let raw: Vec<(usize, Vec<f32>)> = (0u32..32)
            .map(|i| {
                let f = i as f32;
                (i as usize, vec![f.sin(), f.cos(), f * 0.1, f * f * 0.01])
            })
            .collect();

        let tree = FloatVpTree::build(raw.clone());
        let query = [0.5_f32, 0.5, 0.3, 0.1];
        let k = 5;

        let vp_res = tree.search_knn(&query, k);
        let bf_res = brute_knn(&raw, &query, k);

        assert_eq!(vp_res.len(), k);
        for (vp, bf) in vp_res.iter().zip(bf_res.iter()) {
            assert!(
                (vp.1 - bf.1).abs() < 1e-3,
                "high-dim mismatch: vp={} bf={}",
                vp.1,
                bf.1
            );
        }
    }
}
