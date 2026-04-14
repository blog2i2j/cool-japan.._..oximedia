//! VP-tree–backed visual index for sub-linear k-nearest-neighbour search.
//!
//! [`VpIndexedVisual`] is a drop-in upgrade over the naïve linear-scan
//! [`super::index::VisualIndex`].  After an initial batch of documents has been
//! indexed, calling [`VpIndexedVisual::rebuild_tree`] constructs a
//! [`super::vp_tree::FloatVpTree`] from the stored feature vectors.  Subsequent
//! similarity queries are served in O(log n) average time instead of O(n).
//!
//! # Design
//!
//! * **Mutable phase**: documents are appended to a flat `Vec`.  The VP-tree is
//!   *not* rebuilt on every insert (VP-trees are immutable after construction).
//! * **Query phase**: call [`VpIndexedVisual::rebuild_tree`] once to freeze the
//!   index, then call [`VpIndexedVisual::search_knn`] or
//!   [`VpIndexedVisual::search_radius`] in O(log n) time.
//! * **Dirty flag**: after any insertion the tree is marked dirty and falls back
//!   to a brute-force linear scan.  After [`rebuild_tree`] the fast path is
//!   active again.
//!
//! # Example
//!
//! ```rust
//! use oximedia_search::visual::vp_indexed::{VpIndexedVisual, VpSearchResult};
//! use uuid::Uuid;
//!
//! let mut index = VpIndexedVisual::new();
//!
//! let id1 = Uuid::new_v4();
//! let id2 = Uuid::new_v4();
//! index.add_document(id1, vec![0.0, 0.0, 0.0]);
//! index.add_document(id2, vec![1.0, 0.0, 0.0]);
//!
//! // Rebuild the VP-tree before querying.
//! index.rebuild_tree();
//!
//! let results = index.search_knn(&[0.1, 0.0, 0.0], 2);
//! assert_eq!(results.len(), 2);
//! assert_eq!(results[0].asset_id, id1); // closest to [0,0,0]
//! ```

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

use super::vp_tree::{euclidean_dist, FloatVpTree};

// ─────────────────────────────────────────────────────────────────────────────
// Result type
// ─────────────────────────────────────────────────────────────────────────────

/// One result returned from a VP-tree visual similarity search.
#[derive(Debug, Clone)]
pub struct VpSearchResult {
    /// Asset UUID.
    pub asset_id: Uuid,
    /// Euclidean distance to the query vector (lower is more similar).
    pub distance: f32,
    /// Similarity score in [0, 1] derived from distance: `1 / (1 + dist)`.
    pub score: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// VpIndexedVisual
// ─────────────────────────────────────────────────────────────────────────────

/// A visual feature index backed by a VP-tree for sub-linear KNN search.
///
/// # Threading
///
/// `VpIndexedVisual` is not `Sync` by default because it contains mutable
/// internal state.  Wrap in `Arc<Mutex<…>>` when sharing across threads.
pub struct VpIndexedVisual {
    /// Flat storage: store_idx → (uuid, feature_vector).
    entries: Vec<(Uuid, Vec<f32>)>,
    /// Mapping from the flat store index (usize) to UUID for VP-tree results.
    id_by_store_idx: HashMap<usize, Uuid>,
    /// The VP-tree; `None` when the index is dirty (needs rebuild).
    tree: Option<FloatVpTree>,
    /// `true` when documents have been added since the last [`rebuild_tree`].
    dirty: bool,
}

impl VpIndexedVisual {
    /// Create an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            id_by_store_idx: HashMap::new(),
            tree: None,
            dirty: false,
        }
    }

    /// Create an empty index with pre-allocated capacity for `n` documents.
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            entries: Vec::with_capacity(n),
            id_by_store_idx: HashMap::with_capacity(n),
            tree: None,
            dirty: false,
        }
    }

    /// Return the number of indexed documents.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no documents are indexed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return `true` if the VP-tree needs to be rebuilt before a fast query.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Add a document with its feature vector.
    ///
    /// The VP-tree is **not** rebuilt automatically; call [`Self::rebuild_tree`]
    /// when you have finished adding documents.
    pub fn add_document(&mut self, asset_id: Uuid, features: Vec<f32>) {
        let store_idx = self.entries.len();
        self.id_by_store_idx.insert(store_idx, asset_id);
        self.entries.push((asset_id, features));
        self.dirty = true;
        self.tree = None;
    }

    /// Remove a document by UUID.
    ///
    /// This is an O(n) operation that rebuilds the flat store in-place and
    /// marks the tree dirty.
    pub fn remove_document(&mut self, asset_id: Uuid) {
        let before = self.entries.len();
        self.entries.retain(|(id, _)| *id != asset_id);
        if self.entries.len() < before {
            // Reindex the id→store_idx mapping.
            self.id_by_store_idx.clear();
            for (si, (uid, _)) in self.entries.iter().enumerate() {
                self.id_by_store_idx.insert(si, *uid);
            }
            self.dirty = true;
            self.tree = None;
        }
    }

    /// Build (or rebuild) the VP-tree from the current set of documents.
    ///
    /// After this call, [`Self::search_knn`] and [`Self::search_radius`] use
    /// the O(log n) VP-tree path.  If the index is empty, the tree is cleared.
    pub fn rebuild_tree(&mut self) {
        if self.entries.is_empty() {
            self.tree = None;
            self.dirty = false;
            return;
        }

        // Build `(store_idx, Vec<f32>)` pairs as expected by FloatVpTree.
        let points: Vec<(usize, Vec<f32>)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(si, (_, v))| (si, v.clone()))
            .collect();

        self.tree = Some(FloatVpTree::build(points));
        self.dirty = false;
    }

    /// Find the `k` nearest neighbours to `query`.
    ///
    /// Uses the VP-tree when clean; falls back to a brute-force linear scan
    /// when dirty (after insertions since the last [`rebuild_tree`]).
    ///
    /// Returns results sorted by distance ascending (most similar first).
    #[must_use]
    pub fn search_knn(&self, query: &[f32], k: usize) -> Vec<VpSearchResult> {
        if self.is_empty() || k == 0 {
            return Vec::new();
        }

        if let Some(ref tree) = self.tree {
            // Fast VP-tree path.
            tree.search_knn(query, k)
                .into_iter()
                .filter_map(|(store_idx, dist)| {
                    self.id_by_store_idx
                        .get(&store_idx)
                        .copied()
                        .map(|uuid| VpSearchResult {
                            asset_id: uuid,
                            distance: dist,
                            score: 1.0 / (1.0 + dist),
                        })
                })
                .collect()
        } else {
            // Fallback: brute-force linear scan.
            self.linear_knn(query, k)
        }
    }

    /// Find all documents within Euclidean `radius` of `query`.
    ///
    /// Uses the VP-tree when clean; falls back to linear scan when dirty.
    ///
    /// Returns results sorted by distance ascending.
    #[must_use]
    pub fn search_radius(&self, query: &[f32], radius: f32) -> Vec<VpSearchResult> {
        if self.is_empty() {
            return Vec::new();
        }

        if let Some(ref tree) = self.tree {
            // Fast VP-tree path.
            tree.search_radius(query, radius)
                .into_iter()
                .filter_map(|(store_idx, dist)| {
                    self.id_by_store_idx
                        .get(&store_idx)
                        .copied()
                        .map(|uuid| VpSearchResult {
                            asset_id: uuid,
                            distance: dist,
                            score: 1.0 / (1.0 + dist),
                        })
                })
                .collect()
        } else {
            // Fallback: brute-force linear scan.
            self.linear_radius(query, radius)
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Brute-force linear KNN scan (used when tree is dirty).
    fn linear_knn(&self, query: &[f32], k: usize) -> Vec<VpSearchResult> {
        let mut scored: Vec<VpSearchResult> = self
            .entries
            .iter()
            .map(|(uuid, vec)| {
                let dist = euclidean_dist(query, vec);
                VpSearchResult {
                    asset_id: *uuid,
                    distance: dist,
                    score: 1.0 / (1.0 + dist),
                }
            })
            .collect();
        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        scored.truncate(k);
        scored
    }

    /// Brute-force linear radius scan (used when tree is dirty).
    fn linear_radius(&self, query: &[f32], radius: f32) -> Vec<VpSearchResult> {
        let mut scored: Vec<VpSearchResult> = self
            .entries
            .iter()
            .filter_map(|(uuid, vec)| {
                let dist = euclidean_dist(query, vec);
                if dist <= radius {
                    Some(VpSearchResult {
                        asset_id: *uuid,
                        distance: dist,
                        score: 1.0 / (1.0 + dist),
                    })
                } else {
                    None
                }
            })
            .collect();
        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        scored
    }
}

impl Default for VpIndexedVisual {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uuid(n: u8) -> Uuid {
        Uuid::from_bytes([
            n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ])
    }

    fn build_index(points: &[(u8, &[f32])]) -> VpIndexedVisual {
        let mut idx = VpIndexedVisual::new();
        for (n, v) in points {
            idx.add_document(make_uuid(*n), v.to_vec());
        }
        idx.rebuild_tree();
        idx
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_index_returns_nothing() {
        let idx = VpIndexedVisual::new();
        assert!(idx.is_empty());
        assert!(idx.search_knn(&[0.0], 5).is_empty());
        assert!(idx.search_radius(&[0.0], 100.0).is_empty());
    }

    #[test]
    fn test_add_marks_dirty() {
        let mut idx = VpIndexedVisual::new();
        assert!(!idx.is_dirty());
        idx.add_document(Uuid::new_v4(), vec![1.0]);
        assert!(idx.is_dirty());
    }

    #[test]
    fn test_rebuild_clears_dirty() {
        let mut idx = VpIndexedVisual::new();
        idx.add_document(Uuid::new_v4(), vec![1.0]);
        idx.rebuild_tree();
        assert!(!idx.is_dirty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut idx = VpIndexedVisual::with_capacity(4);
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        idx.add_document(Uuid::new_v4(), vec![0.0]);
        idx.add_document(Uuid::new_v4(), vec![1.0]);
        assert_eq!(idx.len(), 2);
        assert!(!idx.is_empty());
    }

    // ── KNN accuracy ────────────────────────────────────────────────────────

    #[test]
    fn test_knn_exact_match_first() {
        let id0 = make_uuid(0);
        let id1 = make_uuid(1);
        let id2 = make_uuid(2);
        let mut idx = VpIndexedVisual::new();
        idx.add_document(id0, vec![0.0, 0.0]);
        idx.add_document(id1, vec![1.0, 0.0]);
        idx.add_document(id2, vec![5.0, 5.0]);
        idx.rebuild_tree();

        let res = idx.search_knn(&[0.0, 0.0], 3);
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].asset_id, id0);
        assert!(res[0].distance.abs() < 1e-5, "exact match should have dist ≈ 0");
    }

    #[test]
    fn test_knn_sorted_ascending() {
        let idx = build_index(&[
            (0, &[0.0]),
            (1, &[2.0]),
            (2, &[10.0]),
            (3, &[3.0]),
            (4, &[1.0]),
        ]);
        let res = idx.search_knn(&[0.0], 5);
        for i in 1..res.len() {
            assert!(
                res[i - 1].distance <= res[i].distance,
                "Not sorted at index {}: {} > {}",
                i,
                res[i - 1].distance,
                res[i].distance
            );
        }
    }

    #[test]
    fn test_knn_k_exceeds_size_returns_all() {
        let idx = build_index(&[(0, &[0.0]), (1, &[1.0]), (2, &[2.0])]);
        let res = idx.search_knn(&[0.5], 100);
        assert_eq!(res.len(), 3);
    }

    #[test]
    fn test_knn_k_zero_returns_empty() {
        let idx = build_index(&[(0, &[0.0])]);
        assert!(idx.search_knn(&[0.0], 0).is_empty());
    }

    #[test]
    fn test_knn_score_is_similarity() {
        let idx = build_index(&[(0, &[0.0, 0.0]), (1, &[1.0, 0.0])]);
        let res = idx.search_knn(&[0.0, 0.0], 2);
        // score = 1 / (1 + dist)
        for r in &res {
            let expected = 1.0 / (1.0 + r.distance);
            assert!(
                (r.score - expected).abs() < 1e-5,
                "score mismatch: got {} expected {}",
                r.score,
                expected
            );
        }
    }

    // ── Radius search ────────────────────────────────────────────────────────

    #[test]
    fn test_radius_search_captures_nearby() {
        let idx = build_index(&[
            (0, &[0.0, 0.0]),
            (1, &[0.5, 0.0]),
            (2, &[10.0, 0.0]),
        ]);
        let res = idx.search_radius(&[0.0, 0.0], 1.0);
        assert_eq!(res.len(), 2, "only points within radius 1.0 should be returned");
        assert!(res.iter().all(|r| r.distance <= 1.0));
    }

    #[test]
    fn test_radius_search_zero_radius_exact_only() {
        let id = make_uuid(42);
        let mut idx = VpIndexedVisual::new();
        idx.add_document(id, vec![3.0, 4.0]);
        idx.add_document(Uuid::new_v4(), vec![3.1, 4.0]);
        idx.rebuild_tree();

        let res = idx.search_radius(&[3.0, 4.0], 0.0);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].asset_id, id);
    }

    // ── Fallback linear scan (dirty tree) ────────────────────────────────────

    #[test]
    fn test_dirty_fallback_knn_agrees_with_tree() {
        // Build tree
        let mut idx = build_index(&[
            (0, &[0.0, 0.0]),
            (1, &[1.0, 0.0]),
            (2, &[5.0, 5.0]),
        ]);

        // Add a new point → tree becomes dirty
        idx.add_document(make_uuid(3), vec![0.1, 0.0]);

        // Fallback linear KNN
        let res_dirty = idx.search_knn(&[0.0, 0.0], 2);

        // Rebuild and query again
        idx.rebuild_tree();
        let res_clean = idx.search_knn(&[0.0, 0.0], 2);

        assert_eq!(res_dirty.len(), res_clean.len());
        for (d, c) in res_dirty.iter().zip(res_clean.iter()) {
            assert!(
                (d.distance - c.distance).abs() < 1e-4,
                "dirty/clean distance mismatch: {} vs {}",
                d.distance,
                c.distance
            );
        }
    }

    // ── Remove document ──────────────────────────────────────────────────────

    #[test]
    fn test_remove_document_decrements_len() {
        let id = make_uuid(99);
        let mut idx = VpIndexedVisual::new();
        idx.add_document(id, vec![1.0, 2.0]);
        idx.add_document(Uuid::new_v4(), vec![3.0, 4.0]);
        idx.rebuild_tree();

        assert_eq!(idx.len(), 2);
        idx.remove_document(id);
        assert_eq!(idx.len(), 1);
        assert!(idx.is_dirty());
    }

    #[test]
    fn test_remove_document_excludes_from_results() {
        let id = make_uuid(77);
        let mut idx = VpIndexedVisual::new();
        idx.add_document(id, vec![0.0]);
        idx.add_document(make_uuid(78), vec![100.0]);
        idx.rebuild_tree();
        idx.remove_document(id);

        // Dirty linear scan — the removed document should not appear.
        let res = idx.search_knn(&[0.0], 5);
        assert!(res.iter().all(|r| r.asset_id != id));
    }

    // ── Default impl ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_is_empty() {
        let idx = VpIndexedVisual::default();
        assert!(idx.is_empty());
        assert!(!idx.is_dirty());
    }
}
