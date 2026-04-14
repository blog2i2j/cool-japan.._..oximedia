//! User segment-based recommendations — simplified k-means API.
//!
//! This module exposes a lightweight [`UserSegmenter`] whose interface centres
//! on raw `(user_id: u64, embedding: Vec<f32>)` pairs, making it easy to plug
//! in pre-computed user embeddings without a full [`UserProfile`].
//!
//! For the rich profile-aware segmentation see [`super::segment`].

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// UserSegment
// ---------------------------------------------------------------------------

/// A named user segment with key-value features and cosine similarity.
///
/// This lightweight profile-style type provides the `new + add_feature +
/// similarity_to` API used by recommendation pipelines that work with
/// feature-map–based segment definitions rather than raw embedding clusters.
#[derive(Debug, Clone)]
pub struct UserSegment {
    /// Stable string identifier for the segment (e.g. "segment_0").
    pub id: String,
    /// Human-readable label.
    pub name: String,
    /// User IDs that belong to this segment.
    pub users: Vec<u64>,
    /// Average embedding vector (centroid) over all members.
    pub avg_embedding: Vec<f32>,
    /// Numeric feature map (e.g. genre affinity, recency score).
    pub features: HashMap<String, f32>,
}

impl UserSegment {
    /// Creates a new, empty segment identified by `id`.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            users: Vec::new(),
            avg_embedding: Vec::new(),
            features: HashMap::new(),
        }
    }

    /// Inserts or updates a named feature value.
    pub fn add_feature(&mut self, key: impl Into<String>, value: f32) {
        self.features.insert(key.into(), value);
    }

    /// Cosine similarity of this segment's feature vector against `other`.
    ///
    /// Returns `0.0` when either segment has no features or the dot product
    /// norm is zero.
    #[must_use]
    pub fn similarity_to(&self, other: &UserSegment) -> f32 {
        let dot: f32 = self
            .features
            .iter()
            .filter_map(|(k, &v)| other.features.get(k).map(|&ov| v * ov))
            .sum();

        let norm_self: f32 = self.features.values().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_other: f32 = other.features.values().map(|&v| v * v).sum::<f32>().sqrt();

        if norm_self < 1e-9 || norm_other < 1e-9 {
            return 0.0;
        }

        dot / (norm_self * norm_other)
    }
}

// ---------------------------------------------------------------------------
// UserSegmenter
// ---------------------------------------------------------------------------

/// K-means–based user segmenter operating on raw f32 embedding vectors.
pub struct UserSegmenter {
    /// Number of clusters to form.
    k_segments: usize,
}

impl UserSegmenter {
    /// Create a new segmenter that will partition users into `k_segments`
    /// clusters.
    #[must_use]
    pub fn new(k_segments: usize) -> Self {
        Self {
            k_segments: k_segments.max(1),
        }
    }

    /// Cluster `users` (each a `(user_id, embedding)` pair) using k-means
    /// with 10 fixed iterations.
    ///
    /// Centroids are seeded with the first `k` users; thereafter the standard
    /// assign-then-update loop is executed for 10 iterations.  If there are
    /// fewer users than segments all users land in distinct single-member
    /// segments.
    #[must_use]
    pub fn cluster_users(&self, users: &[(u64, Vec<f32>)]) -> Vec<UserSegment> {
        if users.is_empty() {
            return Vec::new();
        }

        let k = self.k_segments.min(users.len());
        let dim = users[0].1.len();
        const ITERATIONS: usize = 10;

        // Seed centroids with the first k embeddings.
        let mut centroids: Vec<Vec<f32>> = users.iter().take(k).map(|(_, e)| e.clone()).collect();

        let mut assignments: Vec<usize> = vec![0; users.len()];

        for _ in 0..ITERATIONS {
            // Assignment step.
            for (idx, (_, emb)) in users.iter().enumerate() {
                let mut best_c = 0;
                let mut best_d = f32::INFINITY;
                for (c, centroid) in centroids.iter().enumerate() {
                    let d = euclidean_distance(emb, centroid);
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                assignments[idx] = best_c;
            }

            // Update step — recompute each centroid as the mean of members.
            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0u32; k];

            for (idx, (_, emb)) in users.iter().enumerate() {
                let c = assignments[idx];
                for (j, &v) in emb.iter().enumerate() {
                    sums[c][j] += v;
                }
                counts[c] += 1;
            }

            for c in 0..k {
                if counts[c] > 0 {
                    for j in 0..dim {
                        centroids[c][j] = sums[c][j] / counts[c] as f32;
                    }
                }
            }
        }

        // Build UserSegment structs.
        let mut member_map: HashMap<usize, Vec<u64>> = HashMap::new();
        for (idx, &c) in assignments.iter().enumerate() {
            member_map
                .entry(c)
                .or_default()
                .push(users[idx].0);
        }

        (0..k)
            .map(|c| {
                let members = member_map.remove(&c).unwrap_or_default();
                UserSegment {
                    id: format!("segment_{c}"),
                    name: format!("Segment {c}"),
                    users: members,
                    avg_embedding: centroids[c].clone(),
                    features: HashMap::new(),
                }
            })
            .collect()
    }

    /// Find the index of the segment whose centroid is nearest to
    /// `user_embedding` (Euclidean distance).
    ///
    /// Returns 0 when `segments` is empty (safe fallback).
    #[must_use]
    pub fn find_segment(user_embedding: &[f32], segments: &[UserSegment]) -> usize {
        if segments.is_empty() {
            return 0;
        }
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (i, seg) in segments.iter().enumerate() {
            let d = euclidean_distance(user_embedding, &seg.avg_embedding);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best
    }

    /// Return the first `n` items from `content_pool` as recommendations for
    /// `segment`.
    ///
    /// This is an intentional stub: callers that need more sophisticated
    /// filtering (e.g. category matching against segment affinities) should
    /// supply a pre-filtered `content_pool`.
    #[must_use]
    pub fn segment_recommendations(
        _segment: &UserSegment,
        content_pool: &[u64],
        n: usize,
    ) -> Vec<u64> {
        content_pool.iter().copied().take(n).collect()
    }
}

// ---------------------------------------------------------------------------
// Internal geometry helpers
// ---------------------------------------------------------------------------

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_users(specs: &[(u64, &[f32])]) -> Vec<(u64, Vec<f32>)> {
        specs
            .iter()
            .map(|(id, emb)| (*id, emb.to_vec()))
            .collect()
    }

    #[test]
    fn test_cluster_users_empty() {
        let segmenter = UserSegmenter::new(3);
        let result = segmenter.cluster_users(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_cluster_users_fewer_than_k() {
        let segmenter = UserSegmenter::new(5);
        let users = make_users(&[(1, &[1.0, 0.0]), (2, &[0.0, 1.0])]);
        let segments = segmenter.cluster_users(&users);
        // k is clamped to users.len() = 2
        assert_eq!(segments.len(), 2);
    }

    #[test]
    fn test_cluster_users_two_clear_clusters() {
        let segmenter = UserSegmenter::new(2);
        // Cluster A: near (0, 0), Cluster B: near (10, 10)
        let users = make_users(&[
            (1, &[0.1, 0.1]),
            (2, &[0.2, 0.0]),
            (3, &[9.9, 10.0]),
            (4, &[10.1, 9.8]),
        ]);
        let segments = segmenter.cluster_users(&users);
        assert_eq!(segments.len(), 2);
        // Every segment should have at least 1 member.
        for seg in &segments {
            assert!(!seg.users.is_empty());
        }
        // Total members should equal number of input users.
        let total: usize = segments.iter().map(|s| s.users.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_cluster_segment_ids_are_unique() {
        let segmenter = UserSegmenter::new(3);
        let users = make_users(&[
            (1, &[1.0]),
            (2, &[2.0]),
            (3, &[3.0]),
            (4, &[4.0]),
            (5, &[5.0]),
        ]);
        let segments = segmenter.cluster_users(&users);
        let ids: Vec<&str> = segments.iter().map(|s| s.id.as_str()).collect();
        let unique: std::collections::HashSet<&str> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_find_segment_nearest() {
        let segs = vec![
            UserSegment {
                id: "segment_0".into(),
                name: "A".into(),
                users: vec![1],
                avg_embedding: vec![0.0, 0.0],
                features: HashMap::new(),
            },
            UserSegment {
                id: "segment_1".into(),
                name: "B".into(),
                users: vec![2],
                avg_embedding: vec![10.0, 10.0],
                features: HashMap::new(),
            },
        ];
        // Embedding near segment 0.
        assert_eq!(UserSegmenter::find_segment(&[0.5, 0.5], &segs), 0);
        // Embedding near segment 1.
        assert_eq!(UserSegmenter::find_segment(&[9.5, 9.5], &segs), 1);
    }

    #[test]
    fn test_find_segment_empty_returns_zero() {
        assert_eq!(UserSegmenter::find_segment(&[1.0, 2.0], &[]), 0);
    }

    #[test]
    fn test_segment_recommendations_returns_n() {
        let seg = UserSegment {
            id: "segment_0".into(),
            name: "Test".into(),
            users: vec![1, 2],
            avg_embedding: vec![0.5],
            features: HashMap::new(),
        };
        let pool: Vec<u64> = (100..110).collect();
        let recs = UserSegmenter::segment_recommendations(&seg, &pool, 5);
        assert_eq!(recs.len(), 5);
        assert_eq!(&recs, &pool[..5]);
    }

    #[test]
    fn test_segment_recommendations_fewer_than_n() {
        let seg = UserSegment {
            id: "segment_0".into(),
            name: "Test".into(),
            users: vec![1],
            avg_embedding: vec![0.0],
            features: HashMap::new(),
        };
        let pool = vec![1u64, 2];
        let recs = UserSegmenter::segment_recommendations(&seg, &pool, 10);
        assert_eq!(recs.len(), 2);
    }

    #[test]
    fn test_segment_recommendations_empty_pool() {
        let seg = UserSegment {
            id: "segment_0".into(),
            name: "Empty".into(),
            users: vec![],
            avg_embedding: vec![],
            features: HashMap::new(),
        };
        let recs = UserSegmenter::segment_recommendations(&seg, &[], 5);
        assert!(recs.is_empty());
    }

    #[test]
    fn test_avg_embedding_is_within_bounds() {
        let segmenter = UserSegmenter::new(2);
        let users = make_users(&[
            (1, &[0.0, 0.0]),
            (2, &[2.0, 2.0]),
            (3, &[4.0, 4.0]),
            (4, &[6.0, 6.0]),
        ]);
        let segments = segmenter.cluster_users(&users);
        for seg in &segments {
            for &v in &seg.avg_embedding {
                assert!(v.is_finite(), "centroid coordinate must be finite");
            }
        }
    }

    // ── UserSegment::new + add_feature + similarity_to ──────────────────────

    #[test]
    fn test_user_segment_new_is_empty() {
        let seg = UserSegment::new("seg_a");
        assert_eq!(seg.id, "seg_a");
        assert!(seg.features.is_empty());
        assert!(seg.users.is_empty());
    }

    #[test]
    fn test_user_segment_add_feature() {
        let mut seg = UserSegment::new("seg_b");
        seg.add_feature("action", 0.9);
        seg.add_feature("comedy", 0.4);
        assert_eq!(seg.features.len(), 2);
        assert!((seg.features["action"] - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_user_segment_similarity_identical() {
        let mut seg = UserSegment::new("a");
        seg.add_feature("drama", 1.0);
        seg.add_feature("comedy", 0.5);
        // Cosine similarity with itself should be 1.0
        let s = seg.similarity_to(&seg.clone());
        assert!((s - 1.0).abs() < 1e-5, "self-similarity should be 1.0, got {s}");
    }

    #[test]
    fn test_user_segment_similarity_orthogonal() {
        let mut seg_a = UserSegment::new("a");
        seg_a.add_feature("action", 1.0);
        let mut seg_b = UserSegment::new("b");
        seg_b.add_feature("romance", 1.0);
        // No shared features → cosine similarity = 0
        let s = seg_a.similarity_to(&seg_b);
        assert!(s.abs() < 1e-5, "orthogonal segments: expected 0, got {s}");
    }

    #[test]
    fn test_user_segment_similarity_partial_overlap() {
        let mut seg_a = UserSegment::new("a");
        seg_a.add_feature("action", 1.0);
        seg_a.add_feature("sci-fi", 1.0);
        let mut seg_b = UserSegment::new("b");
        seg_b.add_feature("action", 1.0);
        seg_b.add_feature("romance", 1.0);
        let s = seg_a.similarity_to(&seg_b);
        // dot=1, |a|=√2, |b|=√2 → cosine = 1/2
        assert!((s - 0.5).abs() < 1e-5, "expected 0.5, got {s}");
    }

    #[test]
    fn test_user_segment_similarity_empty_features() {
        let seg_a = UserSegment::new("a");
        let seg_b = UserSegment::new("b");
        assert!((seg_a.similarity_to(&seg_b)).abs() < f32::EPSILON);
    }
}
