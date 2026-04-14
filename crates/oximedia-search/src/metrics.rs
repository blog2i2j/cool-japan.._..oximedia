//! Information-retrieval evaluation metrics for search quality measurement.
//!
//! This module provides the standard precision/recall/MAP/nDCG suite used to
//! quantify how well a retrieval system ranks relevant documents.
//!
//! # Definitions
//!
//! | Metric | Formula |
//! |--------|---------|
//! | Precision@k | `|relevant ∩ retrieved[..k]| / k` |
//! | Recall@k | `|relevant ∩ retrieved[..k]| / |relevant|` |
//! | Average Precision (AP) | `Σ P@i · rel(i) / |relevant|` |
//! | nDCG@k | `DCG@k / IDCG@k` |
//!
//! where DCG@k = `Σ rel(i) / log₂(i + 2)` for `i ∈ 0..k`.
//!
//! # Example
//!
//! ```
//! use std::collections::{HashSet, HashMap};
//! use oximedia_search::metrics::{
//!     compute_precision_at_k, compute_recall_at_k, compute_ndcg, compute_map,
//! };
//!
//! let retrieved = vec![1usize, 3, 5, 7, 9];
//! let relevant: HashSet<usize> = [1, 3, 7].iter().copied().collect();
//!
//! assert!((compute_precision_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-5);
//! assert!((compute_recall_at_k(&retrieved, &relevant, 5) - 1.0).abs() < 1e-5);
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate struct
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate IR evaluation metrics computed for one query.
#[derive(Debug, Clone)]
pub struct SearchMetrics {
    /// Precision at each rank position `1..=k`.
    pub precision_at_k: Vec<f32>,
    /// Recall at each rank position `1..=k`.
    pub recall_at_k: Vec<f32>,
    /// Mean Average Precision across all relevant breakpoints.
    pub mean_avg_precision: f32,
    /// Normalised Discounted Cumulative Gain at position `k`.
    pub ndcg: f32,
}

impl SearchMetrics {
    /// Compute all metrics from a result list and a set of relevant IDs.
    ///
    /// `k` controls how many top results are evaluated.  Relevance scores
    /// default to `1.0` for binary relevance.
    ///
    /// # Arguments
    ///
    /// * `retrieved` — ordered list of document IDs returned by the system.
    /// * `relevant` — set of truly relevant document IDs (ground truth).
    /// * `k` — evaluation depth.
    #[must_use]
    pub fn compute(
        retrieved: &[usize],
        relevant: &HashSet<usize>,
        k: usize,
    ) -> Self {
        // Build binary relevance map (score = 1.0 for each relevant doc).
        let binary_scores: HashMap<usize, f32> =
            relevant.iter().map(|&id| (id, 1.0)).collect();

        let p_at_k: Vec<f32> = (1..=k.min(retrieved.len().max(1)))
            .map(|i| compute_precision_at_k(retrieved, relevant, i))
            .collect();
        let r_at_k: Vec<f32> = (1..=k.min(retrieved.len().max(1)))
            .map(|i| compute_recall_at_k(retrieved, relevant, i))
            .collect();
        let map = compute_map(retrieved, relevant, k);
        let ndcg = compute_ndcg(retrieved, &binary_scores, k);

        Self {
            precision_at_k: p_at_k,
            recall_at_k: r_at_k,
            mean_avg_precision: map,
            ndcg,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual metric functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Precision@k: the fraction of the top-`k` retrieved documents that
/// are relevant.
///
/// Returns `0.0` when `k == 0` or `relevant` is empty.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use oximedia_search::metrics::compute_precision_at_k;
///
/// let retrieved = vec![1usize, 2, 3, 4, 5];
/// let relevant: HashSet<usize> = [1, 3, 5].iter().copied().collect();
///
/// assert!((compute_precision_at_k(&retrieved, &relevant, 3) - 2.0 / 3.0).abs() < 1e-5);
/// assert!((compute_precision_at_k(&retrieved, &relevant, 5) - 3.0 / 5.0).abs() < 1e-5);
/// ```
#[must_use]
pub fn compute_precision_at_k(
    retrieved: &[usize],
    relevant: &HashSet<usize>,
    k: usize,
) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|id| relevant.contains(*id)).count();
    hits as f32 / k as f32
}

/// Compute Recall@k: the fraction of all relevant documents that appear in
/// the top-`k` retrieved list.
///
/// Returns `0.0` when `k == 0` or `relevant` is empty.
///
/// # Example
///
/// ```
/// use std::collections::HashSet;
/// use oximedia_search::metrics::compute_recall_at_k;
///
/// let retrieved = vec![1usize, 2, 3, 4, 5];
/// let relevant: HashSet<usize> = [1, 3, 10].iter().copied().collect();
///
/// // 2 of 3 relevant docs appear in top-5
/// let r = compute_recall_at_k(&retrieved, &relevant, 5);
/// assert!((r - 2.0 / 3.0).abs() < 1e-5, "got {}", r);
/// ```
#[must_use]
pub fn compute_recall_at_k(
    retrieved: &[usize],
    relevant: &HashSet<usize>,
    k: usize,
) -> f32 {
    if k == 0 || relevant.is_empty() {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let hits = top_k.filter(|id| relevant.contains(*id)).count();
    hits as f32 / relevant.len() as f32
}

/// Compute Average Precision (AP) for a single query.
///
/// AP is the mean of Precision@i for each position `i` where the retrieved
/// document at rank `i` is relevant.
///
/// ```text
/// AP = (1 / |R|) × Σ P@i × rel(i)
/// ```
///
/// where `|R|` is the number of relevant documents and `rel(i)` is `1` if
/// the document at rank `i` is relevant.
///
/// Returns `0.0` when `relevant` is empty.
#[must_use]
pub fn compute_average_precision(
    retrieved: &[usize],
    relevant: &HashSet<usize>,
    k: usize,
) -> f32 {
    if relevant.is_empty() {
        return 0.0;
    }
    let top_k = retrieved.iter().take(k);
    let mut running_hits: usize = 0;
    let mut sum_precision: f32 = 0.0;

    for (i, id) in top_k.enumerate() {
        if relevant.contains(id) {
            running_hits += 1;
            // Precision at rank (i+1)
            sum_precision += running_hits as f32 / (i + 1) as f32;
        }
    }

    sum_precision / relevant.len() as f32
}

/// Compute Mean Average Precision (MAP) for a single query.
///
/// For a single-query scenario MAP equals AP.  This function is provided as
/// a named alias for clarity; multi-query MAP should be computed by averaging
/// `compute_average_precision` over all queries.
#[must_use]
pub fn compute_map(
    retrieved: &[usize],
    relevant: &HashSet<usize>,
    k: usize,
) -> f32 {
    compute_average_precision(retrieved, relevant, k)
}

/// Compute normalised Discounted Cumulative Gain (nDCG) at depth `k`.
///
/// DCG uses a logarithmic discount:
///
/// ```text
/// DCG@k = Σ_{i=0}^{k-1}  rel(retrieved[i]) / log₂(i + 2)
/// ```
///
/// nDCG is `DCG@k / IDCG@k` where IDCG is the DCG of an ideal ranking
/// (scores sorted descending).
///
/// Returns `0.0` when `k == 0` or `relevance_scores` is empty.
/// Returns `1.0` when the ideal and actual rankings are identical.
///
/// # Arguments
///
/// * `retrieved` — ordered list of document IDs from the system.
/// * `relevance_scores` — map from document ID to a non-negative relevance
///   grade.  Documents absent from the map are treated as having relevance `0`.
/// * `k` — evaluation depth.
///
/// # Example
///
/// ```
/// use std::collections::HashMap;
/// use oximedia_search::metrics::compute_ndcg;
///
/// // Perfect ranking: most relevant document first.
/// let retrieved = vec![1usize, 2, 3];
/// let scores: HashMap<usize, f32> = [(1, 3.0), (2, 2.0), (3, 1.0)].iter().copied().collect();
/// let ndcg = compute_ndcg(&retrieved, &scores, 3);
/// assert!((ndcg - 1.0).abs() < 1e-5, "perfect ranking should yield nDCG=1.0, got {}", ndcg);
/// ```
#[must_use]
pub fn compute_ndcg(
    retrieved: &[usize],
    relevance_scores: &HashMap<usize, f32>,
    k: usize,
) -> f32 {
    if k == 0 || relevance_scores.is_empty() {
        return 0.0;
    }

    let dcg = discounted_cumulative_gain(retrieved, relevance_scores, k);

    // Build ideal ranking from relevance scores, sorted descending.
    let mut ideal_scores: Vec<f32> = relevance_scores.values().copied().collect();
    ideal_scores.sort_by(|a, b| b.total_cmp(a));
    // Convert to index-based "retrieved" list for IDCG computation.
    // We use synthetic IDs 0..n for the ideal sequence.
    let ideal_ids: Vec<usize> = (0..ideal_scores.len()).collect();
    let ideal_score_map: HashMap<usize, f32> = ideal_ids
        .iter()
        .copied()
        .zip(ideal_scores.iter().copied())
        .collect();
    let idcg = discounted_cumulative_gain(&ideal_ids, &ideal_score_map, k);

    if idcg < f32::EPSILON {
        return 0.0;
    }

    (dcg / idcg).min(1.0)
}

/// Compute raw DCG@k for the `retrieved` sequence.
fn discounted_cumulative_gain(
    retrieved: &[usize],
    relevance_scores: &HashMap<usize, f32>,
    k: usize,
) -> f32 {
    retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, id)| {
            let rel = relevance_scores.get(id).copied().unwrap_or(0.0);
            // Discount: 1 / log₂(i + 2)
            rel / (i as f32 + 2.0).log2()
        })
        .sum()
}

/// Compute the F1 score from precision and recall.
///
/// F1 = 2 × P × R / (P + R).  Returns `0.0` when both are zero.
#[must_use]
pub fn compute_f1(precision: f32, recall: f32) -> f32 {
    let denom = precision + recall;
    if denom < f32::EPSILON {
        return 0.0;
    }
    2.0 * precision * recall / denom
}

/// Compute R-Precision: Precision at rank `|R|` where `|R|` is the number of
/// relevant documents.
///
/// Useful when different queries have different numbers of relevant documents.
#[must_use]
pub fn compute_r_precision(retrieved: &[usize], relevant: &HashSet<usize>) -> f32 {
    let r = relevant.len();
    if r == 0 {
        return 0.0;
    }
    compute_precision_at_k(retrieved, relevant, r)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[usize]) -> HashSet<usize> {
        ids.iter().copied().collect()
    }

    fn scores(pairs: &[(usize, f32)]) -> HashMap<usize, f32> {
        pairs.iter().copied().collect()
    }

    // ── Precision@k ───────────────────────────────────────────────────────

    #[test]
    fn test_precision_k0_returns_zero() {
        let retrieved = vec![1usize, 2, 3];
        let relevant = set(&[1, 2]);
        assert_eq!(compute_precision_at_k(&retrieved, &relevant, 0), 0.0);
    }

    #[test]
    fn test_precision_perfect_top3() {
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 2, 3]);
        let p = compute_precision_at_k(&retrieved, &relevant, 3);
        assert!((p - 1.0).abs() < 1e-6, "got {}", p);
    }

    #[test]
    fn test_precision_no_relevant_in_top_k() {
        let retrieved = vec![10usize, 20, 30];
        let relevant = set(&[1, 2, 3]);
        let p = compute_precision_at_k(&retrieved, &relevant, 3);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_precision_partial() {
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 3, 5]);
        // top-3: [1,2,3] → 2 hits / 3
        let p = compute_precision_at_k(&retrieved, &relevant, 3);
        assert!((p - 2.0 / 3.0).abs() < 1e-6, "got {}", p);
    }

    // ── Recall@k ──────────────────────────────────────────────────────────

    #[test]
    fn test_recall_k0_returns_zero() {
        let retrieved = vec![1usize];
        let relevant = set(&[1]);
        assert_eq!(compute_recall_at_k(&retrieved, &relevant, 0), 0.0);
    }

    #[test]
    fn test_recall_empty_relevant_returns_zero() {
        let retrieved = vec![1usize, 2];
        let relevant = set(&[]);
        assert_eq!(compute_recall_at_k(&retrieved, &relevant, 2), 0.0);
    }

    #[test]
    fn test_recall_perfect() {
        let retrieved = vec![1usize, 2, 3, 99, 100];
        let relevant = set(&[1, 2, 3]);
        let r = compute_recall_at_k(&retrieved, &relevant, 5);
        assert!((r - 1.0).abs() < 1e-6, "got {}", r);
    }

    #[test]
    fn test_recall_partial() {
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 3, 10]);
        // top-5: hits = {1,3} → 2/3
        let r = compute_recall_at_k(&retrieved, &relevant, 5);
        assert!((r - 2.0 / 3.0).abs() < 1e-6, "got {}", r);
    }

    // ── Average Precision / MAP ───────────────────────────────────────────

    #[test]
    fn test_ap_all_relevant_at_top() {
        // Perfect ranking: all relevant docs at top
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 2, 3]);
        let ap = compute_average_precision(&retrieved, &relevant, 5);
        // P@1=1, P@2=1, P@3=1 → AP = 3/3 = 1.0
        assert!((ap - 1.0).abs() < 1e-5, "got {}", ap);
    }

    #[test]
    fn test_ap_empty_relevant() {
        let retrieved = vec![1usize, 2, 3];
        let relevant = set(&[]);
        assert_eq!(compute_average_precision(&retrieved, &relevant, 3), 0.0);
    }

    #[test]
    fn test_ap_interleaved() {
        // retrieved=[1,2,3,4,5], relevant={1,4}
        // P@1=1/1, P@4=2/4 → AP = (1.0 + 0.5) / 2 = 0.75
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 4]);
        let ap = compute_average_precision(&retrieved, &relevant, 5);
        assert!((ap - 0.75).abs() < 1e-5, "got {}", ap);
    }

    #[test]
    fn test_map_is_ap_single_query() {
        let retrieved = vec![1usize, 2, 3];
        let relevant = set(&[2, 3]);
        let ap = compute_average_precision(&retrieved, &relevant, 3);
        let map = compute_map(&retrieved, &relevant, 3);
        assert!((ap - map).abs() < 1e-6);
    }

    // ── nDCG ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ndcg_perfect_ranking() {
        let retrieved = vec![1usize, 2, 3];
        let rel = scores(&[(1, 3.0), (2, 2.0), (3, 1.0)]);
        let ndcg = compute_ndcg(&retrieved, &rel, 3);
        assert!((ndcg - 1.0).abs() < 1e-5, "got {}", ndcg);
    }

    #[test]
    fn test_ndcg_k0_returns_zero() {
        let retrieved = vec![1usize, 2];
        let rel = scores(&[(1, 1.0)]);
        assert_eq!(compute_ndcg(&retrieved, &rel, 0), 0.0);
    }

    #[test]
    fn test_ndcg_empty_scores_returns_zero() {
        let retrieved = vec![1usize, 2, 3];
        let rel: HashMap<usize, f32> = HashMap::new();
        assert_eq!(compute_ndcg(&retrieved, &rel, 3), 0.0);
    }

    #[test]
    fn test_ndcg_reversed_ranking_lower_than_perfect() {
        // Reversed order (least relevant first) should give lower nDCG.
        let perfect = vec![1usize, 2, 3];
        let reversed = vec![3usize, 2, 1];
        let rel = scores(&[(1, 3.0), (2, 2.0), (3, 1.0)]);

        let ndcg_perfect = compute_ndcg(&perfect, &rel, 3);
        let ndcg_reversed = compute_ndcg(&reversed, &rel, 3);

        assert!(
            ndcg_perfect > ndcg_reversed,
            "perfect={} reversed={}",
            ndcg_perfect,
            ndcg_reversed
        );
    }

    #[test]
    fn test_ndcg_binary_relevance() {
        // Binary relevance: scores are either 0 or 1.
        let retrieved = vec![1usize, 99, 2, 98, 3];
        let rel = scores(&[(1, 1.0), (2, 1.0), (3, 1.0)]);
        let ndcg = compute_ndcg(&retrieved, &rel, 5);
        assert!(ndcg > 0.0 && ndcg <= 1.0, "ndcg={}", ndcg);
    }

    // ── F1 / R-Precision ──────────────────────────────────────────────────

    #[test]
    fn test_f1_both_zero() {
        assert_eq!(compute_f1(0.0, 0.0), 0.0);
    }

    #[test]
    fn test_f1_perfect() {
        assert!((compute_f1(1.0, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_balanced() {
        let p = 0.6_f32;
        let r = 0.4_f32;
        let expected = 2.0 * p * r / (p + r);
        assert!((compute_f1(p, r) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_r_precision_empty_relevant() {
        let retrieved = vec![1usize, 2];
        assert_eq!(compute_r_precision(&retrieved, &set(&[])), 0.0);
    }

    #[test]
    fn test_r_precision_perfect() {
        let retrieved = vec![1usize, 2, 3, 4];
        let relevant = set(&[1, 2, 3]);
        let rp = compute_r_precision(&retrieved, &relevant);
        assert!((rp - 1.0).abs() < 1e-6, "got {}", rp);
    }

    // ── SearchMetrics struct ──────────────────────────────────────────────

    #[test]
    fn test_search_metrics_compute() {
        let retrieved = vec![1usize, 2, 3, 4, 5];
        let relevant = set(&[1, 3, 5]);
        let m = SearchMetrics::compute(&retrieved, &relevant, 5);

        // precision_at_k has 5 entries
        assert_eq!(m.precision_at_k.len(), 5);
        // recall_at_k has 5 entries
        assert_eq!(m.recall_at_k.len(), 5);
        // MAP is between 0 and 1
        assert!(m.mean_avg_precision >= 0.0 && m.mean_avg_precision <= 1.0);
        // nDCG is between 0 and 1
        assert!(m.ndcg >= 0.0 && m.ndcg <= 1.0);
    }

    #[test]
    fn test_search_metrics_perfect_top_k() {
        let retrieved = vec![1usize, 2, 3];
        let relevant = set(&[1, 2, 3]);
        let m = SearchMetrics::compute(&retrieved, &relevant, 3);

        // All precision@k values should be 1.0
        for p in &m.precision_at_k {
            assert!((*p - 1.0).abs() < 1e-5, "precision={}", p);
        }
        assert!((m.mean_avg_precision - 1.0).abs() < 1e-5);
    }
}
