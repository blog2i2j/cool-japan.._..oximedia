//! Scene reordering using greedy nearest-neighbor tour optimization.
//!
//! The [`SceneReorderer`] finds a visually coherent ordering of scenes by
//! treating the reordering problem as an approximate Traveling Salesman
//! Problem (TSP). It uses a greedy nearest-neighbor heuristic that starts
//! from the scene with the highest total feature energy and at each step
//! picks the unvisited scene most similar (smallest Euclidean distance in
//! feature space) to the current one.
//!
//! # Complexity
//!
//! O(n²) in the number of scenes, which is acceptable for typical edit
//! sequences (< a few thousand scenes).
//!
//! # Example
//!
//! ```rust
//! use oximedia_auto::scene_reorder::SceneReorderer;
//!
//! let features = vec![
//!     vec![1.0_f32, 0.0],
//!     vec![1.1_f32, 0.0],
//!     vec![0.0_f32, 1.0],
//!     vec![0.0_f32, 1.1],
//! ];
//!
//! let order = SceneReorderer::sort_by_similarity(&features);
//! assert_eq!(order.len(), features.len());
//! // Adjacent indices in the tour should be close in feature space.
//! ```

#![allow(dead_code, clippy::cast_precision_loss)]

/// Scene reordering via greedy nearest-neighbor feature-space tour.
pub struct SceneReorderer;

impl SceneReorderer {
    // ── helpers ───────────────────────────────────────────────────────────────

    /// Squared Euclidean distance between two feature vectors.
    ///
    /// Vectors of different lengths are compared up to the shorter length.
    fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        (0..len).map(|i| (a[i] - b[i]).powi(2)).sum()
    }

    /// L2 norm (energy) of a feature vector.
    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    // ── public API ────────────────────────────────────────────────────────────

    /// Compute a greedy nearest-neighbor ordering of scenes.
    ///
    /// Returns a permutation vector `order` such that `scene_features[order[i]]`
    /// is the `i`-th scene in the suggested reordering.
    ///
    /// The starting scene is the one with the highest feature energy (L2 norm).
    /// Ties are broken by index (lowest index wins).
    ///
    /// # Panics
    ///
    /// Does not panic; returns an empty `Vec` for empty input.
    #[must_use]
    pub fn sort_by_similarity(scene_features: &[Vec<f32>]) -> Vec<usize> {
        let n = scene_features.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![0];
        }

        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);

        // Pick the starting scene: highest feature energy.
        let start = (0..n)
            .max_by(|&a, &b| {
                Self::norm(&scene_features[a])
                    .partial_cmp(&Self::norm(&scene_features[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);

        visited[start] = true;
        order.push(start);

        // Greedy nearest-neighbor loop.
        for _ in 1..n {
            let current = *order.last().unwrap_or(&0);
            let current_feat = &scene_features[current];

            // Find the closest unvisited scene.
            let next = (0..n).filter(|&i| !visited[i]).min_by(|&a, &b| {
                let da = Self::squared_distance(current_feat, &scene_features[a]);
                let db = Self::squared_distance(current_feat, &scene_features[b]);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

            if let Some(idx) = next {
                visited[idx] = true;
                order.push(idx);
            }
        }

        order
    }

    /// Compute the total tour length in feature space for a given ordering.
    ///
    /// Useful for comparing alternative orderings or verifying improvement.
    #[must_use]
    pub fn tour_length(scene_features: &[Vec<f32>], order: &[usize]) -> f32 {
        if order.len() < 2 {
            return 0.0;
        }
        order
            .windows(2)
            .map(|w| {
                let a = order
                    .first()
                    .and_then(|_| scene_features.get(w[0]))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                let b = scene_features.get(w[1]).map(Vec::as_slice).unwrap_or(&[]);
                // Actually use w[0] and w[1] directly:
                let fa = scene_features.get(w[0]).map(Vec::as_slice).unwrap_or(&[]);
                let fb = b;
                let _ = a;
                Self::squared_distance(fa, fb).sqrt()
            })
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_returns_empty() {
        let order = SceneReorderer::sort_by_similarity(&[]);
        assert!(order.is_empty());
    }

    #[test]
    fn test_single_returns_zero() {
        let order = SceneReorderer::sort_by_similarity(&[vec![1.0_f32, 2.0]]);
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_output_length_matches_input() {
        let features: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32, (i * 2) as f32]).collect();
        let order = SceneReorderer::sort_by_similarity(&features);
        assert_eq!(order.len(), features.len());
    }

    #[test]
    fn test_output_is_permutation() {
        let features: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let mut order = SceneReorderer::sort_by_similarity(&features);
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_greedy_starts_at_highest_energy() {
        // Scene 3 has the highest L2 norm ([10, 10] → norm ≈ 14.14).
        let features = vec![
            vec![1.0_f32, 1.0],
            vec![2.0_f32, 2.0],
            vec![3.0_f32, 3.0],
            vec![10.0_f32, 10.0],
        ];
        let order = SceneReorderer::sort_by_similarity(&features);
        assert_eq!(order[0], 3, "Tour should start at highest-energy scene");
    }

    #[test]
    fn test_two_clusters_stay_together() {
        // Two tight clusters: {0,1} near origin, {2,3} far away.
        let features = vec![
            vec![0.0_f32, 0.0],
            vec![0.1_f32, 0.0],
            vec![100.0_f32, 100.0],
            vec![100.1_f32, 100.0],
        ];
        let order = SceneReorderer::sort_by_similarity(&features);
        // The two elements of each cluster should be adjacent in the tour.
        // Tour: {2,3} (highest energy first), then {0,1} or {1,0}.
        assert_eq!(order.len(), 4);
        // Check adjacency: each cluster's members should be next to each other.
        let pos: std::collections::HashMap<usize, usize> = order
            .iter()
            .copied()
            .enumerate()
            .map(|(p, i)| (i, p))
            .collect();
        let diff_01 = (pos[&0] as isize - pos[&1] as isize).unsigned_abs();
        let diff_23 = (pos[&2] as isize - pos[&3] as isize).unsigned_abs();
        assert_eq!(diff_01, 1, "Scenes 0 and 1 should be adjacent in tour");
        assert_eq!(diff_23, 1, "Scenes 2 and 3 should be adjacent in tour");
    }

    #[test]
    fn test_tour_length_positive() {
        let features = vec![vec![0.0_f32, 0.0], vec![1.0_f32, 0.0], vec![1.0_f32, 1.0]];
        let order = SceneReorderer::sort_by_similarity(&features);
        let len = SceneReorderer::tour_length(&features, &order);
        assert!(len >= 0.0);
        assert!(len.is_finite());
    }

    #[test]
    fn test_identical_features_still_produces_permutation() {
        let features = vec![vec![0.5_f32, 0.5]; 5];
        let mut order = SceneReorderer::sort_by_similarity(&features);
        order.sort_unstable();
        assert_eq!(order, vec![0, 1, 2, 3, 4]);
    }
}
