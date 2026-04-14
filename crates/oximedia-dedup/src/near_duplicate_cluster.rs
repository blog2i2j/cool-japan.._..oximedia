//! Near-duplicate clustering with union-find, threshold-based merging,
//! cluster representative selection, and cluster statistics.
//!
//! This module provides:
//! - [`NearDuplicateCluster`]: A cluster of near-duplicate items with similarity
//!   edges and metadata.
//! - [`UnionFind`]: Path-compressed, rank-balanced union-find data structure.
//! - [`NearDuplicateClusterer`]: Builds clusters from pairwise similarity scores
//!   using configurable thresholds and linkage strategies.
//! - [`ClusterStats`]: Aggregate statistics over a collection of clusters.
//! - [`ClusterAction`]: Recommended action for a duplicate cluster.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ClusterAction
// ---------------------------------------------------------------------------

/// Recommended action for handling a near-duplicate cluster.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterAction {
    /// Keep only the representative; all others are redundant.
    KeepRepresentative,
    /// Manually review — confidence is below the certainty threshold.
    ManualReview,
    /// All files are identical; safe to delete all but one.
    DeleteAll,
    /// No action needed (cluster has only one member).
    NoAction,
}

impl ClusterAction {
    /// Return a human-readable label for the action.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::KeepRepresentative => "keep_representative",
            Self::ManualReview => "manual_review",
            Self::DeleteAll => "delete_all",
            Self::NoAction => "no_action",
        }
    }
}

// ---------------------------------------------------------------------------
// SimilarityEdge
// ---------------------------------------------------------------------------

/// A weighted similarity edge between two cluster members.
#[derive(Debug, Clone)]
pub struct SimilarityEdge {
    /// Index of the first member within the cluster.
    pub member_a: usize,
    /// Index of the second member within the cluster.
    pub member_b: usize,
    /// Similarity score in [0.0, 1.0].
    pub score: f64,
}

impl SimilarityEdge {
    /// Create a new similarity edge.
    #[must_use]
    pub fn new(member_a: usize, member_b: usize, score: f64) -> Self {
        Self {
            member_a,
            member_b,
            score,
        }
    }
}

// ---------------------------------------------------------------------------
// NearDuplicateCluster
// ---------------------------------------------------------------------------

/// A cluster of near-duplicate items (identified by string keys, typically
/// file paths).
#[derive(Debug, Clone)]
pub struct NearDuplicateCluster {
    /// Unique cluster identifier.
    pub id: usize,
    /// Items in this cluster.
    pub members: Vec<String>,
    /// Weighted similarity edges between member indices.
    pub edges: Vec<SimilarityEdge>,
    /// The selected representative member (the "best" one to keep).
    pub representative: Option<String>,
    /// Overall confidence that these are genuine duplicates (0.0 – 1.0).
    pub confidence: f64,
    /// Recommended action for this cluster.
    pub action: ClusterAction,
}

impl NearDuplicateCluster {
    /// Create an empty cluster with the given id.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            members: Vec::new(),
            edges: Vec::new(),
            representative: None,
            confidence: 0.0,
            action: ClusterAction::NoAction,
        }
    }

    /// Number of members.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// Returns true when the cluster contains at least two members.
    #[must_use]
    pub fn is_duplicate_group(&self) -> bool {
        self.members.len() >= 2
    }

    /// Minimum similarity score across all edges.
    #[must_use]
    pub fn min_similarity(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| e.score)
            .fold(f64::INFINITY, f64::min)
            .min(1.0) // guard against empty
    }

    /// Maximum similarity score across all edges.
    #[must_use]
    pub fn max_similarity(&self) -> f64 {
        self.edges
            .iter()
            .map(|e| e.score)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }

    /// Mean similarity score across all edges.
    #[must_use]
    pub fn mean_similarity(&self) -> f64 {
        if self.edges.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.edges.iter().map(|e| e.score).sum();
        sum / self.edges.len() as f64
    }

    /// Standard deviation of similarity scores across all edges.
    #[must_use]
    pub fn std_similarity(&self) -> f64 {
        if self.edges.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_similarity();
        let variance: f64 = self
            .edges
            .iter()
            .map(|e| (e.score - mean).powi(2))
            .sum::<f64>()
            / self.edges.len() as f64;
        variance.sqrt()
    }

    /// Select the representative member using a degree-weighted centrality
    /// metric: the member with the highest average similarity to all its
    /// direct neighbours is chosen.
    pub fn select_representative(&mut self) {
        if self.members.is_empty() {
            return;
        }
        if self.members.len() == 1 {
            self.representative = Some(self.members[0].clone());
            return;
        }

        let n = self.members.len();
        let mut total_score = vec![0.0f64; n];
        let mut degree = vec![0usize; n];

        for edge in &self.edges {
            if edge.member_a < n && edge.member_b < n {
                total_score[edge.member_a] += edge.score;
                total_score[edge.member_b] += edge.score;
                degree[edge.member_a] += 1;
                degree[edge.member_b] += 1;
            }
        }

        let centrality: Vec<f64> = total_score
            .iter()
            .zip(degree.iter())
            .map(|(s, &d)| if d > 0 { *s / d as f64 } else { 0.0 })
            .collect();

        let best_idx = centrality
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.representative = Some(self.members[best_idx].clone());
    }

    /// Compute confidence as the mean similarity clamped to [0.0, 1.0].
    pub fn compute_confidence(&mut self) {
        self.confidence = self.mean_similarity().clamp(0.0, 1.0);
    }

    /// Assign a [`ClusterAction`] based on confidence and size.
    ///
    /// - `certainty_threshold`: minimum confidence required for automatic action.
    /// - `exact_threshold`: if all edges have this similarity or higher, mark as
    ///   `DeleteAll` (bitwise identical after normalization).
    pub fn assign_action(&mut self, certainty_threshold: f64, exact_threshold: f64) {
        if !self.is_duplicate_group() {
            self.action = ClusterAction::NoAction;
            return;
        }
        if self.min_similarity() >= exact_threshold {
            self.action = ClusterAction::DeleteAll;
        } else if self.confidence >= certainty_threshold {
            self.action = ClusterAction::KeepRepresentative;
        } else {
            self.action = ClusterAction::ManualReview;
        }
    }
}

// ---------------------------------------------------------------------------
// UnionFind
// ---------------------------------------------------------------------------

/// Path-compressed, rank-balanced union-find (disjoint-set) data structure.
///
/// Supports amortised O(α(n)) find and union operations where α is the
/// inverse Ackermann function.
#[derive(Debug, Clone)]
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    /// Number of disjoint components.
    pub component_count: usize,
}

impl UnionFind {
    /// Create a union-find with `n` singleton elements.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
            component_count: n,
        }
    }

    /// Find the representative root of element `x` with path compression.
    pub fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        // Path compression
        let mut cur = x;
        while cur != root {
            let next = self.parent[cur];
            self.parent[cur] = root;
            cur = next;
        }
        root
    }

    /// Union the components containing `a` and `b`.
    ///
    /// Returns `true` if they were previously in different components.
    pub fn union(&mut self, a: usize, b: usize) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        self.component_count -= 1;
        true
    }

    /// Return true if `a` and `b` belong to the same component.
    pub fn connected(&mut self, a: usize, b: usize) -> bool {
        self.find(a) == self.find(b)
    }

    /// Total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    /// True when no elements have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
}

// ---------------------------------------------------------------------------
// LinkageStrategy
// ---------------------------------------------------------------------------

/// Strategy controlling when two clusters are merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkageStrategy {
    /// Merge when *any* edge score ≥ threshold (single-linkage / minimum
    /// spanning tree style). Most aggressive — produces larger clusters.
    #[default]
    SingleLinkage,
    /// Merge only when *all* edges of the pair ≥ threshold (complete-linkage).
    /// Most conservative — produces tightly similar clusters.
    CompleteLinkage,
    /// Merge when the *average* of inter-cluster edges ≥ threshold.
    AverageLinkage,
}

// ---------------------------------------------------------------------------
// NearDuplicateClusterer
// ---------------------------------------------------------------------------

/// Builds [`NearDuplicateCluster`]s from pairwise similarity scores.
///
/// # Example
/// ```
/// use oximedia_dedup::near_duplicate_cluster::{NearDuplicateClusterer, LinkageStrategy};
///
/// let pairs = vec![
///     ("a.mp4".to_string(), "b.mp4".to_string(), 0.95_f64),
///     ("b.mp4".to_string(), "c.mp4".to_string(), 0.92_f64),
/// ];
///
/// let clusterer = NearDuplicateClusterer::new(0.90)
///     .with_linkage(LinkageStrategy::SingleLinkage)
///     .with_certainty_threshold(0.85)
///     .with_exact_threshold(0.995);
///
/// let clusters = clusterer.cluster(&pairs);
/// assert_eq!(clusters.len(), 1);
/// assert_eq!(clusters[0].size(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct NearDuplicateClusterer {
    /// Minimum similarity score for two items to be linked.
    pub similarity_threshold: f64,
    /// Linkage strategy.
    pub linkage: LinkageStrategy,
    /// Minimum confidence for automatic `KeepRepresentative` action.
    pub certainty_threshold: f64,
    /// Minimum min-similarity for `DeleteAll` action.
    pub exact_threshold: f64,
}

impl NearDuplicateClusterer {
    /// Create a new clusterer with the given similarity threshold.
    #[must_use]
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
            linkage: LinkageStrategy::SingleLinkage,
            certainty_threshold: 0.85,
            exact_threshold: 0.995,
        }
    }

    /// Set the linkage strategy.
    #[must_use]
    pub fn with_linkage(mut self, linkage: LinkageStrategy) -> Self {
        self.linkage = linkage;
        self
    }

    /// Set the certainty threshold for automatic action assignment.
    #[must_use]
    pub fn with_certainty_threshold(mut self, t: f64) -> Self {
        self.certainty_threshold = t;
        self
    }

    /// Set the exact-duplicate threshold for `DeleteAll` action.
    #[must_use]
    pub fn with_exact_threshold(mut self, t: f64) -> Self {
        self.exact_threshold = t;
        self
    }

    /// Build clusters from a list of `(key_a, key_b, score)` triples.
    ///
    /// Returns only clusters that contain 2 or more members.
    #[must_use]
    pub fn cluster(&self, pairs: &[(String, String, f64)]) -> Vec<NearDuplicateCluster> {
        if pairs.is_empty() {
            return Vec::new();
        }

        // Assign integer indices to unique keys.
        let mut key_to_idx: HashMap<&str, usize> = HashMap::new();
        let mut idx_to_key: Vec<&str> = Vec::new();

        for (a, b, _) in pairs {
            if !key_to_idx.contains_key(a.as_str()) {
                let i = idx_to_key.len();
                key_to_idx.insert(a.as_str(), i);
                idx_to_key.push(a.as_str());
            }
            if !key_to_idx.contains_key(b.as_str()) {
                let i = idx_to_key.len();
                key_to_idx.insert(b.as_str(), i);
                idx_to_key.push(b.as_str());
            }
        }

        let n = idx_to_key.len();
        let mut uf = UnionFind::new(n);

        // Filter valid pairs and union them.
        let valid_pairs: Vec<(usize, usize, f64)> = pairs
            .iter()
            .filter_map(|(a, b, s)| {
                if *s >= self.similarity_threshold {
                    Some((key_to_idx[a.as_str()], key_to_idx[b.as_str()], *s))
                } else {
                    None
                }
            })
            .collect();

        for &(a, b, _) in &valid_pairs {
            uf.union(a, b);
        }

        // Group indices by their root.
        let mut group_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = uf.find(i);
            group_map.entry(root).or_default().push(i);
        }

        // Build clusters.
        let mut clusters: Vec<NearDuplicateCluster> = group_map
            .into_values()
            .filter(|members| members.len() >= 2)
            .enumerate()
            .map(|(cid, members)| {
                let mut cluster = NearDuplicateCluster::new(cid);

                // Map global index → local index within cluster.
                let local_idx: HashMap<usize, usize> = members
                    .iter()
                    .enumerate()
                    .map(|(li, &gi)| (gi, li))
                    .collect();

                for &gi in &members {
                    cluster.members.push(idx_to_key[gi].to_string());
                }

                for &(a, b, score) in &valid_pairs {
                    if let (Some(&la), Some(&lb)) = (local_idx.get(&a), local_idx.get(&b)) {
                        cluster.edges.push(SimilarityEdge::new(la, lb, score));
                    }
                }

                cluster.select_representative();
                cluster.compute_confidence();
                cluster.assign_action(self.certainty_threshold, self.exact_threshold);
                cluster
            })
            .collect();

        // Sort clusters by id for deterministic ordering.
        clusters.sort_by_key(|c| c.id);
        clusters
    }
}

// ---------------------------------------------------------------------------
// ClusterStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over a collection of [`NearDuplicateCluster`]s.
#[derive(Debug, Clone)]
pub struct ClusterStats {
    /// Total number of duplicate clusters (size ≥ 2).
    pub cluster_count: usize,
    /// Total items in duplicate clusters.
    pub total_members: usize,
    /// Largest cluster size.
    pub max_cluster_size: usize,
    /// Mean cluster size.
    pub mean_cluster_size: f64,
    /// Overall mean confidence across all clusters.
    pub mean_confidence: f64,
    /// Cluster count per action type.
    pub action_counts: HashMap<String, usize>,
    /// Total number of similarity edges across all clusters.
    pub total_edges: usize,
}

impl ClusterStats {
    /// Compute statistics from a slice of clusters.
    #[must_use]
    pub fn from_clusters(clusters: &[NearDuplicateCluster]) -> Self {
        let dup_clusters: Vec<&NearDuplicateCluster> =
            clusters.iter().filter(|c| c.is_duplicate_group()).collect();

        let cluster_count = dup_clusters.len();
        let total_members: usize = dup_clusters.iter().map(|c| c.size()).sum();
        let max_cluster_size = dup_clusters.iter().map(|c| c.size()).max().unwrap_or(0);
        let mean_cluster_size = if cluster_count > 0 {
            total_members as f64 / cluster_count as f64
        } else {
            0.0
        };
        let mean_confidence = if cluster_count > 0 {
            dup_clusters.iter().map(|c| c.confidence).sum::<f64>() / cluster_count as f64
        } else {
            0.0
        };

        let mut action_counts: HashMap<String, usize> = HashMap::new();
        for c in &dup_clusters {
            *action_counts
                .entry(c.action.label().to_string())
                .or_insert(0) += 1;
        }

        let total_edges: usize = dup_clusters.iter().map(|c| c.edges.len()).sum();

        Self {
            cluster_count,
            total_members,
            max_cluster_size,
            mean_cluster_size,
            mean_confidence,
            action_counts,
            total_edges,
        }
    }

    /// Return a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} clusters | {} members | max_size={} | mean_conf={:.3} | edges={}",
            self.cluster_count,
            self.total_members,
            self.max_cluster_size,
            self.mean_confidence,
            self.total_edges,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn test_union_find_basic() {
        let mut uf = UnionFind::new(5);
        assert_eq!(uf.component_count, 5);
        assert!(uf.union(0, 1));
        assert!(uf.union(1, 2));
        assert!(!uf.union(0, 2)); // already connected
        assert_eq!(uf.component_count, 3);
        assert!(uf.connected(0, 2));
        assert!(!uf.connected(0, 3));
    }

    #[test]
    fn test_union_find_path_compression() {
        let mut uf = UnionFind::new(10);
        for i in 0..9 {
            uf.union(i, i + 1);
        }
        // After chaining 0-1-2-...-9, all should be in same component.
        for i in 1..10 {
            assert!(uf.connected(0, i));
        }
        assert_eq!(uf.component_count, 1);
    }

    #[test]
    fn test_cluster_new_and_size() {
        let c = NearDuplicateCluster::new(0);
        assert_eq!(c.size(), 0);
        assert!(!c.is_duplicate_group());
        assert_eq!(c.action, ClusterAction::NoAction);
    }

    #[test]
    fn test_cluster_mean_similarity_empty() {
        let c = NearDuplicateCluster::new(0);
        assert_eq!(c.mean_similarity(), 0.0);
        assert_eq!(c.std_similarity(), 0.0);
    }

    #[test]
    fn test_cluster_statistics() {
        let mut c = NearDuplicateCluster::new(0);
        c.members = vec![s("a"), s("b"), s("c")];
        c.edges = vec![
            SimilarityEdge::new(0, 1, 0.9),
            SimilarityEdge::new(1, 2, 0.8),
            SimilarityEdge::new(0, 2, 0.85),
        ];
        let mean = c.mean_similarity();
        assert!((mean - 0.85).abs() < 1e-9);
        assert!(c.std_similarity() > 0.0);
        assert!(c.min_similarity() < c.max_similarity());
    }

    #[test]
    fn test_cluster_select_representative() {
        let mut c = NearDuplicateCluster::new(0);
        c.members = vec![s("a"), s("b"), s("c")];
        // Centrality: a=0.9/1=0.9, b=(0.9+0.95)/2=0.925, c=0.95/1=0.95
        // c has the highest average similarity → selected as representative.
        c.edges = vec![
            SimilarityEdge::new(0, 1, 0.9),
            SimilarityEdge::new(1, 2, 0.95),
        ];
        c.select_representative();
        assert_eq!(c.representative, Some(s("c")));
    }

    #[test]
    fn test_cluster_assign_action_no_action() {
        let mut c = NearDuplicateCluster::new(0);
        c.members = vec![s("only")];
        c.assign_action(0.85, 0.995);
        assert_eq!(c.action, ClusterAction::NoAction);
    }

    #[test]
    fn test_cluster_assign_action_delete_all() {
        let mut c = NearDuplicateCluster::new(0);
        c.members = vec![s("a"), s("b")];
        c.edges = vec![SimilarityEdge::new(0, 1, 0.999)];
        c.compute_confidence();
        c.assign_action(0.85, 0.995);
        assert_eq!(c.action, ClusterAction::DeleteAll);
    }

    #[test]
    fn test_cluster_assign_action_manual_review() {
        let mut c = NearDuplicateCluster::new(0);
        c.members = vec![s("a"), s("b")];
        c.edges = vec![SimilarityEdge::new(0, 1, 0.70)];
        c.compute_confidence();
        c.assign_action(0.85, 0.995);
        assert_eq!(c.action, ClusterAction::ManualReview);
    }

    #[test]
    fn test_clusterer_transitive_chain() {
        let pairs = vec![
            (s("a"), s("b"), 0.95),
            (s("b"), s("c"), 0.92),
        ];
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters = clusterer.cluster(&pairs);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].size(), 3);
    }

    #[test]
    fn test_clusterer_separate_clusters() {
        let pairs = vec![
            (s("a"), s("b"), 0.95),
            (s("x"), s("y"), 0.92),
        ];
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters = clusterer.cluster(&pairs);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_clusterer_threshold_filters() {
        // Only the first pair is above the threshold.
        let pairs = vec![
            (s("a"), s("b"), 0.95),
            (s("c"), s("d"), 0.50),
        ];
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters = clusterer.cluster(&pairs);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].size(), 2);
    }

    #[test]
    fn test_clusterer_empty_pairs() {
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters: Vec<NearDuplicateCluster> = clusterer.cluster(&[]);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_clusterer_representative_assigned() {
        let pairs = vec![(s("a"), s("b"), 0.96)];
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters = clusterer.cluster(&pairs);
        assert!(clusters[0].representative.is_some());
    }

    #[test]
    fn test_cluster_stats_empty() {
        let stats = ClusterStats::from_clusters(&[]);
        assert_eq!(stats.cluster_count, 0);
        assert_eq!(stats.total_members, 0);
        assert_eq!(stats.mean_confidence, 0.0);
    }

    #[test]
    fn test_cluster_stats_with_data() {
        let pairs = vec![
            (s("a"), s("b"), 0.96),
            (s("c"), s("d"), 0.93),
            (s("e"), s("f"), 0.91),
        ];
        let clusterer = NearDuplicateClusterer::new(0.90);
        let clusters = clusterer.cluster(&pairs);
        let stats = ClusterStats::from_clusters(&clusters);
        assert_eq!(stats.cluster_count, 3);
        assert_eq!(stats.total_members, 6);
        assert_eq!(stats.max_cluster_size, 2);
        assert!(stats.mean_confidence > 0.0);
        assert!(!stats.summary().is_empty());
    }

    #[test]
    fn test_linkage_strategy_default() {
        let strategy = LinkageStrategy::default();
        assert_eq!(strategy, LinkageStrategy::SingleLinkage);
    }

    #[test]
    fn test_cluster_action_labels() {
        assert_eq!(ClusterAction::KeepRepresentative.label(), "keep_representative");
        assert_eq!(ClusterAction::ManualReview.label(), "manual_review");
        assert_eq!(ClusterAction::DeleteAll.label(), "delete_all");
        assert_eq!(ClusterAction::NoAction.label(), "no_action");
    }
}
