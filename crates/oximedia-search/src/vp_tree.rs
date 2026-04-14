//! VP-tree (vantage-point tree) for sub-linear k-nearest-neighbour search.
//!
//! A VP-tree is a metric-space data structure that partitions items by their
//! distance to a selected *vantage point*. Each internal node stores a vantage
//! point `vp` and a threshold radius `mu`; items within `mu` go into the left
//! (inner) sub-tree, items beyond `mu` go into the right (outer) sub-tree.
//! KNN queries prune entire sub-trees that cannot improve the current candidate
//! list, yielding O(log n) average-case performance.
//!
//! # Example
//!
//! ```
//! use oximedia_search::vp_tree::VpTree;
//!
//! // Hamming distance on u64 "perceptual hash" values
//! let dist = |a: &u64, b: &u64| (a ^ b).count_ones() as f32;
//! let items: Vec<u64> = vec![0b0000, 0b0001, 0b0011, 0b0111, 0b1111];
//! let tree = VpTree::build(items, |a, b| (a ^ b).count_ones() as f32);
//!
//! let neighbours = tree.knn(&0b0000u64, 2, &dist);
//! assert_eq!(neighbours.len(), 2);
//! assert_eq!(*neighbours[0].1, 0b0000u64); // exact match first
//! ```

#![allow(dead_code)]

use std::collections::BinaryHeap;

// ─────────────────────────────────────────────────────────────────────────────
// Internal node representation
// ─────────────────────────────────────────────────────────────────────────────

/// One node in the VP-tree.
#[derive(Debug)]
enum VpNode<T> {
    /// Leaf node holding one item.
    Leaf { item: T },
    /// Internal node that splits by distance to the vantage point.
    Internal {
        /// The vantage-point item for this node.
        vantage: T,
        /// Median distance used as partition threshold.
        mu: f32,
        /// Items whose distance to `vantage` is ≤ `mu` (inner ball).
        inner: Box<VpNode<T>>,
        /// Items whose distance to `vantage` is > `mu` (outer shell).
        outer: Box<VpNode<T>>,
    },
    /// Empty subtree (used when a partition produces no items on one side).
    Empty,
}

// ─────────────────────────────────────────────────────────────────────────────
// Max-heap entry for the KNN priority queue
// ─────────────────────────────────────────────────────────────────────────────

/// Wrapper that reverses ordering so `BinaryHeap` acts as a max-heap on
/// distance, letting us efficiently evict the *farthest* candidate.
#[derive(Debug, PartialEq)]
struct HeapEntry {
    dist: f32,
    index: usize,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Largest distance first (max-heap).
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.index.cmp(&other.index))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// A VP-tree providing sub-linear k-nearest-neighbour and range searches.
///
/// The tree is immutable after construction. Rebuild with [`VpTree::build`]
/// to incorporate new items.
///
/// # Type parameters
///
/// * `T` — item type stored in the tree.
pub struct VpTree<T> {
    /// Flat storage of all items (the tree nodes hold indices into this vec).
    items: Vec<T>,
    /// Root node of the tree.
    root: VpNode<usize>,
    /// Total number of items.
    len: usize,
}

impl<T: Clone> VpTree<T> {
    /// Build a VP-tree from `items` using the supplied `dist_fn` metric.
    ///
    /// The distance function must be a proper metric (non-negative,
    /// symmetric, satisfying the triangle inequality) for the pruning to be
    /// correct, though the tree will still build for non-metric functions.
    ///
    /// # Panics
    ///
    /// Does not panic; returns an empty tree when `items` is empty.
    pub fn build<F>(items: Vec<T>, dist_fn: F) -> Self
    where
        F: Fn(&T, &T) -> f32,
    {
        let len = items.len();
        if len == 0 {
            return Self {
                items: Vec::new(),
                root: VpNode::Empty,
                len: 0,
            };
        }

        // Build index array, then recursively construct the tree.
        let mut indices: Vec<usize> = (0..len).collect();
        let root = Self::build_node(&items, &mut indices, &dist_fn);

        Self { items, root, len }
    }

    /// Return the number of items in the tree.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if the tree contains no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Find the `k` nearest neighbours to `query`.
    ///
    /// Returns a vector of `(distance, &item)` pairs sorted by distance
    /// ascending. If `k` exceeds the number of items, all items are returned.
    pub fn knn<F>(&self, query: &T, k: usize, dist_fn: &F) -> Vec<(f32, &T)>
    where
        F: Fn(&T, &T) -> f32,
    {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        let effective_k = k.min(self.len);
        // Max-heap of (distance, index) — tracks the k closest seen so far.
        // The heap's top is always the *farthest* of the current candidates.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(effective_k + 1);
        // Current search radius τ: prune sub-trees farther than τ.
        let mut tau = f32::INFINITY;

        Self::search_knn(
            &self.root,
            &self.items,
            query,
            effective_k,
            dist_fn,
            &mut heap,
            &mut tau,
        );

        // Collect, sort ascending by distance.
        let mut result: Vec<(f32, &T)> = heap
            .into_iter()
            .map(|e| (e.dist, &self.items[e.index]))
            .collect();
        result.sort_by(|a, b| a.0.total_cmp(&b.0));
        result
    }

    /// Find all items within `radius` of `query` (inclusive).
    ///
    /// Returns `(distance, &item)` pairs sorted by distance ascending.
    pub fn range_search<F>(&self, query: &T, radius: f32, dist_fn: &F) -> Vec<(f32, &T)>
    where
        F: Fn(&T, &T) -> f32,
    {
        if self.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<(f32, usize)> = Vec::new();
        Self::search_range(
            &self.root,
            &self.items,
            query,
            radius,
            dist_fn,
            &mut results,
        );

        let mut output: Vec<(f32, &T)> = results
            .into_iter()
            .map(|(d, idx)| (d, &self.items[idx]))
            .collect();
        output.sort_by(|a, b| a.0.total_cmp(&b.0));
        output
    }

    // ── Internal construction ─────────────────────────────────────────────

    fn build_node<F>(all_items: &[T], indices: &mut [usize], dist_fn: &F) -> VpNode<usize>
    where
        F: Fn(&T, &T) -> f32,
    {
        match indices.len() {
            0 => VpNode::Empty,
            1 => VpNode::Leaf { item: indices[0] },
            _ => {
                // Choose vantage point: pick the first index (could be made
                // smarter, but this is correct and deterministic).
                let vp_idx = indices[0];
                let rest = &mut indices[1..];
                let n = rest.len();

                // Compute distances from the vantage point to all other items.
                let dists: Vec<f32> = rest
                    .iter()
                    .map(|&i| dist_fn(&all_items[vp_idx], &all_items[i]))
                    .collect();

                // Find the median distance (partition threshold μ).
                let mid = n / 2;
                // Use a partial sort to find the median without full sort.
                let mu = Self::nth_element_value(&mut dists.clone(), mid);

                // Partition `rest` into inner (d ≤ μ) and outer (d > μ) halves.
                // We sort by distance to get a stable partition.
                let mut indexed_dists: Vec<(usize, f32)> = rest
                    .iter()
                    .zip(dists.iter())
                    .map(|(&i, &d)| (i, d))
                    .collect();
                indexed_dists.sort_by(|a, b| a.1.total_cmp(&b.1));

                let mut inner_indices: Vec<usize> =
                    indexed_dists[..mid].iter().map(|x| x.0).collect();
                let mut outer_indices: Vec<usize> =
                    indexed_dists[mid..].iter().map(|x| x.0).collect();

                let inner_node = Self::build_node(all_items, &mut inner_indices, dist_fn);
                let outer_node = Self::build_node(all_items, &mut outer_indices, dist_fn);

                VpNode::Internal {
                    vantage: vp_idx,
                    mu,
                    inner: Box::new(inner_node),
                    outer: Box::new(outer_node),
                }
            }
        }
    }

    /// Partial selection: returns the value of the element that would be at
    /// position `k` in sorted order (0-indexed), without fully sorting.
    fn nth_element_value(arr: &mut Vec<f32>, k: usize) -> f32 {
        if arr.is_empty() {
            return 0.0;
        }
        let k = k.min(arr.len() - 1);
        arr.sort_by(|a, b| a.total_cmp(b));
        arr[k]
    }

    // ── Internal search ───────────────────────────────────────────────────

    fn search_knn<F>(
        node: &VpNode<usize>,
        all_items: &[T],
        query: &T,
        k: usize,
        dist_fn: &F,
        heap: &mut BinaryHeap<HeapEntry>,
        tau: &mut f32,
    ) where
        F: Fn(&T, &T) -> f32,
    {
        match node {
            VpNode::Empty => {}

            VpNode::Leaf { item } => {
                let d = dist_fn(query, &all_items[*item]);
                Self::push_candidate(heap, k, *item, d, tau);
            }

            VpNode::Internal {
                vantage,
                mu,
                inner,
                outer,
            } => {
                let d = dist_fn(query, &all_items[*vantage]);
                Self::push_candidate(heap, k, *vantage, d, tau);

                // Determine which subtree to explore first (closer first).
                if d <= *mu {
                    // Query is inside the ball — search inner first.
                    if d - *tau <= *mu {
                        Self::search_knn(inner, all_items, query, k, dist_fn, heap, tau);
                    }
                    if d + *tau > *mu {
                        Self::search_knn(outer, all_items, query, k, dist_fn, heap, tau);
                    }
                } else {
                    // Query is outside the ball — search outer first.
                    if d + *tau > *mu {
                        Self::search_knn(outer, all_items, query, k, dist_fn, heap, tau);
                    }
                    if d - *tau <= *mu {
                        Self::search_knn(inner, all_items, query, k, dist_fn, heap, tau);
                    }
                }
            }
        }
    }

    fn search_range<F>(
        node: &VpNode<usize>,
        all_items: &[T],
        query: &T,
        radius: f32,
        dist_fn: &F,
        results: &mut Vec<(f32, usize)>,
    ) where
        F: Fn(&T, &T) -> f32,
    {
        match node {
            VpNode::Empty => {}

            VpNode::Leaf { item } => {
                let d = dist_fn(query, &all_items[*item]);
                if d <= radius {
                    results.push((d, *item));
                }
            }

            VpNode::Internal {
                vantage,
                mu,
                inner,
                outer,
            } => {
                let d = dist_fn(query, &all_items[*vantage]);
                if d <= radius {
                    results.push((d, *vantage));
                }

                // Prune sub-trees that cannot intersect the query ball.
                //
                // Inner ball covers distances [0, μ] from vantage.
                // Query ball covers distances [d - radius, d + radius] from vantage.
                // They intersect iff d - radius ≤ μ.
                if d - radius <= *mu {
                    Self::search_range(inner, all_items, query, radius, dist_fn, results);
                }
                // Outer shell covers distances (μ, ∞) from vantage.
                // They intersect iff d + radius > μ.
                if d + radius > *mu {
                    Self::search_range(outer, all_items, query, radius, dist_fn, results);
                }
            }
        }
    }

    /// Push a candidate into the KNN heap, evicting the farthest if needed.
    fn push_candidate(
        heap: &mut BinaryHeap<HeapEntry>,
        k: usize,
        index: usize,
        dist: f32,
        tau: &mut f32,
    ) {
        if heap.len() < k {
            heap.push(HeapEntry { dist, index });
            if heap.len() == k {
                *tau = heap.peek().map(|e| e.dist).unwrap_or(f32::INFINITY);
            }
        } else if dist < *tau {
            heap.pop();
            heap.push(HeapEntry { dist, index });
            *tau = heap.peek().map(|e| e.dist).unwrap_or(f32::INFINITY);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience: VpTree with a stored distance function
// ─────────────────────────────────────────────────────────────────────────────

/// A [`VpTree`] that stores its distance function, allowing method-style KNN
/// calls without passing the metric on every query.
pub struct VpTreeOwned<T, F>
where
    F: Fn(&T, &T) -> f32,
{
    tree: VpTree<T>,
    dist_fn: F,
}

impl<T: Clone, F: Fn(&T, &T) -> f32> VpTreeOwned<T, F> {
    /// Build an owned VP-tree from `items` and `dist_fn`.
    pub fn build(items: Vec<T>, dist_fn: F) -> Self {
        let tree = VpTree::build(items, &dist_fn);
        Self { tree, dist_fn }
    }

    /// K-nearest-neighbour search.
    pub fn knn(&self, query: &T, k: usize) -> Vec<(f32, &T)> {
        self.tree.knn(query, k, &self.dist_fn)
    }

    /// Range search.
    pub fn range_search(&self, query: &T, radius: f32) -> Vec<(f32, &T)> {
        self.tree.range_search(query, radius, &self.dist_fn)
    }

    /// Number of items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Whether the tree is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hamming distance between two u64 values (counts differing bits).
    fn hamming(a: &u64, b: &u64) -> f32 {
        (a ^ b).count_ones() as f32
    }

    /// Brute-force KNN for correctness verification.
    fn brute_knn(items: &[u64], query: u64, k: usize) -> Vec<(f32, u64)> {
        let mut dists: Vec<(f32, u64)> = items.iter().map(|&x| (hamming(&query, &x), x)).collect();
        dists.sort_by(|a, b| a.0.total_cmp(&b.0));
        dists.truncate(k);
        dists
    }

    #[test]
    fn test_empty_tree() {
        let tree: VpTreeOwned<u64, _> = VpTreeOwned::build(vec![], hamming);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.knn(&0u64, 5).is_empty());
        assert!(tree.range_search(&0u64, 3.0).is_empty());
    }

    #[test]
    fn test_single_item() {
        let tree = VpTreeOwned::build(vec![0b1010_u64], hamming);
        assert_eq!(tree.len(), 1);

        let knn = tree.knn(&0b1010_u64, 1);
        assert_eq!(knn.len(), 1);
        assert_eq!(*knn[0].1, 0b1010_u64);
        assert!((knn[0].0).abs() < 1e-6);
    }

    #[test]
    fn test_knn_basic() {
        // Items differ by 1-4 bits from 0
        let items: Vec<u64> = vec![
            0b0000_0000, // dist 0 from 0
            0b0000_0001, // dist 1
            0b0000_0011, // dist 2
            0b0000_0111, // dist 3
            0b0000_1111, // dist 4
        ];
        let tree = VpTreeOwned::build(items.clone(), hamming);
        let query = 0u64;

        let knn = tree.knn(&query, 3);
        assert_eq!(knn.len(), 3);

        // Check sorted ascending by distance
        for i in 1..knn.len() {
            assert!(knn[i - 1].0 <= knn[i].0, "KNN results not sorted ascending");
        }

        // Closest should be exact match
        assert!((knn[0].0).abs() < 1e-6);
        assert_eq!(*knn[0].1, 0u64);

        // Verify against brute force
        let bf = brute_knn(&items, query, 3);
        for (tree_res, bf_res) in knn.iter().zip(bf.iter()) {
            assert!(
                (tree_res.0 - bf_res.0).abs() < 1e-6,
                "VP-tree distance {} != brute-force distance {}",
                tree_res.0,
                bf_res.0
            );
        }
    }

    #[test]
    fn test_knn_exceeds_len() {
        let items: Vec<u64> = vec![0b00, 0b01, 0b11];
        let tree = VpTreeOwned::build(items.clone(), hamming);
        let knn = tree.knn(&0u64, 100);
        assert_eq!(knn.len(), 3, "Should return all items when k > len");
    }

    #[test]
    fn test_range_search_exact_match() {
        let items: Vec<u64> = vec![
            0b0000_0000, // dist 0
            0b0000_0001, // dist 1
            0b0000_0011, // dist 2
            0b0000_0111, // dist 3
        ];
        let tree = VpTreeOwned::build(items, hamming);
        let within_1 = tree.range_search(&0u64, 1.0);
        // Should include distances 0 and 1
        assert_eq!(within_1.len(), 2, "Expected 2 items within radius 1");
        for (d, _) in &within_1 {
            assert!(*d <= 1.0, "Found item with distance {} > radius 1", d);
        }
    }

    #[test]
    fn test_range_search_zero_radius() {
        let items: Vec<u64> = vec![0b0000, 0b0001, 0b0011];
        let tree = VpTreeOwned::build(items, hamming);
        let exact = tree.range_search(&0b0000u64, 0.0);
        assert_eq!(exact.len(), 1);
        assert_eq!(*exact[0].1, 0b0000u64);
    }

    #[test]
    fn test_range_search_all_items() {
        let items: Vec<u64> = vec![0b0000, 0b1111, 0b0101, 0b1010];
        let tree = VpTreeOwned::build(items, hamming);
        // All items have at most 4 bits set, so radius 64 captures all
        let all = tree.range_search(&0u64, 64.0);
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_range_search_sorted_ascending() {
        let items: Vec<u64> = vec![0b0000, 0b0001, 0b0011, 0b0111, 0b1111];
        let tree = VpTreeOwned::build(items, hamming);
        let results = tree.range_search(&0u64, 5.0);
        for i in 1..results.len() {
            assert!(
                results[i - 1].0 <= results[i].0,
                "Range search results not sorted ascending at index {}",
                i
            );
        }
    }

    #[test]
    fn test_knn_correctness_large() {
        // Generate 64 items with varying numbers of set bits.
        let items: Vec<u64> = (0u64..64)
            .map(|i| {
                // Item i has i bits set starting from LSB.
                if i == 0 {
                    0u64
                } else if i >= 64 {
                    u64::MAX
                } else {
                    u64::MAX.wrapping_shr(64u32 - i as u32)
                }
            })
            .collect();

        let tree = VpTreeOwned::build(items.clone(), hamming);
        let query = 0b0000_0000_1111_1111u64; // 8 bits set

        let k = 5;
        let tree_results = tree.knn(&query, k);
        let bf_results = brute_knn(&items, query, k);

        assert_eq!(tree_results.len(), k);
        // Distances should match brute force
        for (tr, bf) in tree_results.iter().zip(bf_results.iter()) {
            assert!(
                (tr.0 - bf.0).abs() < 1e-6,
                "Mismatch: vp-tree dist={}, brute-force dist={}",
                tr.0,
                bf.0
            );
        }
    }

    #[test]
    fn test_knn_perceptual_hash_vectors() {
        // Simulate perceptual hash comparison: 8-byte (64-bit) hashes
        let hashes: Vec<u64> = vec![
            0x0000_0000_0000_0000, // all black
            0xFFFF_FFFF_FFFF_FFFF, // all white
            0x00FF_00FF_00FF_00FF, // alternating bytes
            0x5555_5555_5555_5555, // alternating bits (0101...)
            0xAAAA_AAAA_AAAA_AAAA, // alternating bits (1010...)
            0x0F0F_0F0F_0F0F_0F0F, // nibble alternating
            0xF0F0_F0F0_F0F0_F0F0, // nibble alternating inverse
            0x0000_0000_FFFF_FFFF, // half-half
        ];

        let tree = VpTreeOwned::build(hashes.clone(), hamming);

        // Query: near the "all black" hash — should find 0x0000...0000 first
        let query = 0x0000_0000_0000_0001u64; // 1 bit different from all-black

        let knn = tree.knn(&query, 3);
        assert_eq!(knn.len(), 3);
        // The closest item should be the all-black hash (1 bit away) or the
        // query itself (if query were in the tree). Since query is not in the
        // tree, the closest should be all-black.
        assert_eq!(*knn[0].1, 0x0000_0000_0000_0000u64);
        assert!((knn[0].0 - 1.0).abs() < 1e-6);

        // Verify sorted ascending
        for i in 1..knn.len() {
            assert!(knn[i - 1].0 <= knn[i].0);
        }
    }

    #[test]
    fn test_knn_k_zero() {
        let items: Vec<u64> = vec![0, 1, 2, 3];
        let tree = VpTreeOwned::build(items, hamming);
        let result = tree.knn(&0u64, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_range_search_no_results() {
        // Items all have many bits set; query is 0 with tiny radius
        let items: Vec<u64> = vec![0xFFFF_FFFF_FFFF_FFFFu64; 5];
        let tree = VpTreeOwned::build(items, hamming);
        let result = tree.range_search(&0u64, 10.0);
        // All items are 64 bits away from 0, which is > 10
        assert!(result.is_empty());
    }

    #[test]
    fn test_two_items() {
        let items: Vec<u64> = vec![0b00, 0b11];
        let tree = VpTreeOwned::build(items, hamming);
        let knn = tree.knn(&0b01u64, 2);
        assert_eq!(knn.len(), 2);
        // Both items are 1 bit away from 0b01
        assert!((knn[0].0 - 1.0).abs() < 1e-6);
        assert!((knn[1].0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_tree_len() {
        let items: Vec<u64> = (0..42).collect();
        let tree = VpTreeOwned::build(items, hamming);
        assert_eq!(tree.len(), 42);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_brute_force_agreement_random_like() {
        // Pseudo-random hashes generated from a simple LCG
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let mut items = Vec::with_capacity(128);
        for _ in 0..128 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            items.push(state);
        }

        let tree = VpTreeOwned::build(items.clone(), hamming);
        let query = items[0] ^ 0b1111; // slightly modified version of first item

        let k = 10;
        let tree_res = tree.knn(&query, k);
        let bf_res = brute_knn(&items, query, k);

        assert_eq!(tree_res.len(), k);

        // The top-k distances from VP-tree and brute force must match
        for (tr, bf) in tree_res.iter().zip(bf_res.iter()) {
            assert!(
                (tr.0 - bf.0).abs() < 1e-6,
                "VP-tree knn mismatch at top-{}: vp={} bf={}",
                k,
                tr.0,
                bf.0
            );
        }
    }
}
