//! Information Retrieval evaluation metrics for benchmarking search quality.
//!
//! This module implements standard IR evaluation metrics used to measure search
//! system quality against a ground-truth relevance judgement set (a *qrel*).
//!
//! # Metrics implemented
//!
//! | Metric | Description |
//! |---|---|
//! | Precision\@k | Fraction of top-k results that are relevant |
//! | Recall\@k | Fraction of relevant documents found in top-k |
//! | F1\@k | Harmonic mean of Precision\@k and Recall\@k |
//! | Average Precision (AP) | Area under the precision-recall curve for one query |
//! | Mean Average Precision (MAP) | AP averaged over a query set |
//! | NDCG\@k | Normalised Discounted Cumulative Gain |
//! | R-Precision | Precision at the rank equal to the number of relevant docs |
//! | Reciprocal Rank (RR) | Rank of the first relevant result (for MRR) |
//! | Mean Reciprocal Rank (MRR) | RR averaged over a query set |
//!
//! # Usage
//!
//! ```rust
//! use oximedia_search::ir_evaluation::{RelevanceJudgements, evaluate_query};
//!
//! let mut qrels = RelevanceJudgements::new();
//! qrels.add("q1", "doc-a", 2); // highly relevant
//! qrels.add("q1", "doc-b", 1); // partially relevant
//! qrels.add("q1", "doc-c", 0); // not relevant
//!
//! let ranked = vec!["doc-b", "doc-a", "doc-c"];
//! let metrics = evaluate_query(&qrels, "q1", &ranked, 3);
//! assert!(metrics.precision_at_k > 0.0);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Relevance judgements (qrels)
// ─────────────────────────────────────────────────────────────────────────────

/// Grade of relevance used in multi-graded NDCG evaluation.
pub type RelevanceGrade = u32;

/// A set of relevance judgements (query → document → grade).
///
/// Grade semantics follow the TREC convention:
/// - 0 = not relevant
/// - 1 = partially relevant
/// - 2 = relevant
/// - 3+ = highly relevant
///
/// Binary relevance (for Precision/Recall/AP) is defined as grade ≥ 1.
#[derive(Debug, Clone, Default)]
pub struct RelevanceJudgements {
    /// Outer key: query id.  Inner key: document id.  Value: grade.
    data: HashMap<String, HashMap<String, RelevanceGrade>>,
}

impl RelevanceJudgements {
    /// Create an empty judgement set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update the relevance grade for `(query_id, doc_id)`.
    pub fn add(&mut self, query_id: &str, doc_id: &str, grade: RelevanceGrade) {
        self.data
            .entry(query_id.to_string())
            .or_default()
            .insert(doc_id.to_string(), grade);
    }

    /// Return the grade of `doc_id` for `query_id`, defaulting to 0 (not
    /// relevant) for unjudged documents.
    #[must_use]
    pub fn grade(&self, query_id: &str, doc_id: &str) -> RelevanceGrade {
        self.data
            .get(query_id)
            .and_then(|docs| docs.get(doc_id))
            .copied()
            .unwrap_or(0)
    }

    /// Whether `doc_id` is considered binary-relevant for `query_id`.
    #[must_use]
    pub fn is_relevant(&self, query_id: &str, doc_id: &str) -> bool {
        self.grade(query_id, doc_id) >= 1
    }

    /// Return the total number of relevant documents for `query_id`.
    #[must_use]
    pub fn relevant_count(&self, query_id: &str) -> usize {
        self.data
            .get(query_id)
            .map(|docs| docs.values().filter(|&&g| g >= 1).count())
            .unwrap_or(0)
    }

    /// Return all query IDs in this judgement set.
    #[must_use]
    pub fn query_ids(&self) -> Vec<&str> {
        self.data.keys().map(String::as_str).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-query metric bundle
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluation metrics for a single query.
#[derive(Debug, Clone)]
pub struct QueryMetrics {
    /// Query identifier.
    pub query_id: String,
    /// Precision at the rank cut-off k.
    pub precision_at_k: f64,
    /// Recall at the rank cut-off k.
    pub recall_at_k: f64,
    /// F1 at the rank cut-off k (harmonic mean of P\@k and R\@k).
    pub f1_at_k: f64,
    /// Average Precision over the full ranked list (truncated at k).
    pub average_precision: f64,
    /// Normalised Discounted Cumulative Gain at rank cut-off k.
    pub ndcg_at_k: f64,
    /// R-Precision (precision at rank = number of relevant docs, up to k).
    pub r_precision: f64,
    /// Reciprocal rank of the first relevant result (0.0 if none found in k).
    pub reciprocal_rank: f64,
    /// Number of relevant documents in the judged set.
    pub relevant_total: usize,
    /// Rank cut-off applied.
    pub k: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate metrics computed over a set of queries.
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    /// Mean Average Precision.
    pub map: f64,
    /// Mean Reciprocal Rank.
    pub mrr: f64,
    /// Mean Precision\@k.
    pub mean_precision_at_k: f64,
    /// Mean Recall\@k.
    pub mean_recall_at_k: f64,
    /// Mean F1\@k.
    pub mean_f1_at_k: f64,
    /// Mean NDCG\@k.
    pub mean_ndcg_at_k: f64,
    /// Number of queries evaluated.
    pub query_count: usize,
}

impl AggregateMetrics {
    /// Compute aggregate metrics from a slice of per-query metrics.
    #[must_use]
    pub fn from_query_metrics(per_query: &[QueryMetrics]) -> Self {
        let n = per_query.len();
        if n == 0 {
            return Self {
                map: 0.0,
                mrr: 0.0,
                mean_precision_at_k: 0.0,
                mean_recall_at_k: 0.0,
                mean_f1_at_k: 0.0,
                mean_ndcg_at_k: 0.0,
                query_count: 0,
            };
        }
        let n_f = n as f64;
        let map = per_query.iter().map(|m| m.average_precision).sum::<f64>() / n_f;
        let mrr = per_query.iter().map(|m| m.reciprocal_rank).sum::<f64>() / n_f;
        let mean_precision_at_k =
            per_query.iter().map(|m| m.precision_at_k).sum::<f64>() / n_f;
        let mean_recall_at_k = per_query.iter().map(|m| m.recall_at_k).sum::<f64>() / n_f;
        let mean_f1_at_k = per_query.iter().map(|m| m.f1_at_k).sum::<f64>() / n_f;
        let mean_ndcg_at_k = per_query.iter().map(|m| m.ndcg_at_k).sum::<f64>() / n_f;

        Self {
            map,
            mrr,
            mean_precision_at_k,
            mean_recall_at_k,
            mean_f1_at_k,
            mean_ndcg_at_k,
            query_count: n,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute evaluation metrics for a single query.
///
/// - `qrels`: ground-truth relevance judgements.
/// - `query_id`: the query being evaluated.
/// - `ranked`: ordered list of document IDs returned by the system (rank 1 =
///   first element).
/// - `k`: rank cut-off; results beyond position `k` are ignored.
///
/// # Returns
///
/// A [`QueryMetrics`] struct with all computed values.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn evaluate_query(
    qrels: &RelevanceJudgements,
    query_id: &str,
    ranked: &[&str],
    k: usize,
) -> QueryMetrics {
    let relevant_total = qrels.relevant_count(query_id);
    let cutoff = k.min(ranked.len());
    let top_k = &ranked[..cutoff];

    // Precision@k / Recall@k
    let relevant_in_k = top_k
        .iter()
        .filter(|&&doc| qrels.is_relevant(query_id, doc))
        .count();

    let precision_at_k = if cutoff == 0 {
        0.0
    } else {
        relevant_in_k as f64 / cutoff as f64
    };

    let recall_at_k = if relevant_total == 0 {
        0.0
    } else {
        relevant_in_k as f64 / relevant_total as f64
    };

    let f1_at_k = if precision_at_k + recall_at_k == 0.0 {
        0.0
    } else {
        2.0 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
    };

    // Average Precision
    let average_precision = compute_average_precision(qrels, query_id, top_k, relevant_total);

    // NDCG@k
    let ndcg_at_k = compute_ndcg(qrels, query_id, top_k, k);

    // R-Precision: precision at rank R where R = min(relevant_total, k)
    let r = relevant_total.min(cutoff);
    let r_precision = if r == 0 {
        0.0
    } else {
        let relevant_in_r = ranked[..r]
            .iter()
            .filter(|&&doc| qrels.is_relevant(query_id, doc))
            .count();
        relevant_in_r as f64 / r as f64
    };

    // Reciprocal Rank: 1 / rank of first relevant result.
    let reciprocal_rank = top_k
        .iter()
        .enumerate()
        .find_map(|(i, &doc)| {
            if qrels.is_relevant(query_id, doc) {
                Some(1.0 / (i + 1) as f64)
            } else {
                None
            }
        })
        .unwrap_or(0.0);

    QueryMetrics {
        query_id: query_id.to_string(),
        precision_at_k,
        recall_at_k,
        f1_at_k,
        average_precision,
        ndcg_at_k,
        r_precision,
        reciprocal_rank,
        relevant_total,
        k,
    }
}

/// Evaluate a full set of queries and return aggregate metrics.
///
/// - `qrels`: relevance judgements.
/// - `ranked_lists`: map from query_id to the ranked list of document IDs.
/// - `k`: rank cut-off for all metrics.
#[must_use]
pub fn evaluate_all(
    qrels: &RelevanceJudgements,
    ranked_lists: &HashMap<String, Vec<String>>,
    k: usize,
) -> (Vec<QueryMetrics>, AggregateMetrics) {
    let mut per_query: Vec<QueryMetrics> = ranked_lists
        .iter()
        .map(|(qid, docs)| {
            let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            evaluate_query(qrels, qid, &doc_refs, k)
        })
        .collect();

    // Stable order for reproducibility.
    per_query.sort_by(|a, b| a.query_id.cmp(&b.query_id));

    let aggregate = AggregateMetrics::from_query_metrics(&per_query);
    (per_query, aggregate)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Average Precision for `top_k` results.
#[allow(clippy::cast_precision_loss)]
fn compute_average_precision(
    qrels: &RelevanceJudgements,
    query_id: &str,
    top_k: &[&str],
    relevant_total: usize,
) -> f64 {
    if relevant_total == 0 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    let mut hits = 0usize;
    for (i, &doc) in top_k.iter().enumerate() {
        if qrels.is_relevant(query_id, doc) {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64;
        }
    }
    sum / relevant_total as f64
}

/// Discounted Cumulative Gain using the standard `(2^grade - 1) / log2(rank+1)` formula.
#[allow(clippy::cast_precision_loss)]
fn compute_dcg(qrels: &RelevanceJudgements, query_id: &str, top_k: &[&str]) -> f64 {
    top_k
        .iter()
        .enumerate()
        .map(|(i, &doc)| {
            let grade = qrels.grade(query_id, doc) as f64;
            let gain = (2.0_f64.powf(grade) - 1.0) / (i as f64 + 2.0).log2();
            gain
        })
        .sum()
}

/// Compute NDCG\@k by dividing DCG by the ideal DCG (iDCG).
#[allow(clippy::cast_precision_loss)]
fn compute_ndcg(
    qrels: &RelevanceJudgements,
    query_id: &str,
    top_k: &[&str],
    k: usize,
) -> f64 {
    let dcg = compute_dcg(qrels, query_id, top_k);
    // Build ideal ranking: sort all judged docs for this query by grade desc.
    if let Some(docs_map) = qrels.data.get(query_id) {
        let mut ideal: Vec<RelevanceGrade> = docs_map.values().copied().collect();
        ideal.sort_unstable_by(|a, b| b.cmp(a));
        let ideal_top: Vec<&str> = std::iter::repeat("__ideal__").take(ideal.len().min(k)).collect();
        // We need to compute iDCG directly from the grade vector, not from doc
        // lookups, because the ideal docs are synthetic.
        let idcg: f64 = ideal
            .iter()
            .take(ideal_top.len())
            .enumerate()
            .map(|(i, &grade)| {
                let gain = (2.0_f64.powf(grade as f64) - 1.0) / (i as f64 + 2.0).log2();
                gain
            })
            .sum();
        if idcg == 0.0 {
            0.0
        } else {
            (dcg / idcg).min(1.0)
        }
    } else {
        0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small judgement set and a corresponding "perfect" ranking.
    fn build_qrels() -> RelevanceJudgements {
        let mut qrels = RelevanceJudgements::new();
        // query "q1": 3 relevant docs out of 5
        qrels.add("q1", "doc-a", 2);
        qrels.add("q1", "doc-b", 1);
        qrels.add("q1", "doc-c", 0);
        qrels.add("q1", "doc-d", 2);
        qrels.add("q1", "doc-e", 0);
        qrels
    }

    #[test]
    fn test_perfect_precision_at_3() {
        let qrels = build_qrels();
        // All top-3 are relevant.
        let ranked = vec!["doc-a", "doc-b", "doc-d", "doc-c", "doc-e"];
        let m = evaluate_query(&qrels, "q1", &ranked, 3);
        assert!(
            (m.precision_at_k - 1.0).abs() < 1e-9,
            "precision@3 should be 1.0 got {}",
            m.precision_at_k
        );
    }

    #[test]
    fn test_zero_precision_at_k_no_relevant() {
        let qrels = build_qrels();
        // Only non-relevant docs in top-3.
        let ranked = vec!["doc-c", "doc-e", "doc-c"];
        let m = evaluate_query(&qrels, "q1", &ranked, 3);
        assert!(
            m.precision_at_k < 1e-9,
            "precision@3 should be 0 got {}",
            m.precision_at_k
        );
    }

    #[test]
    fn test_recall_at_k_partial() {
        let qrels = build_qrels();
        // Only doc-a relevant, so 1 out of 3 relevant found.
        let ranked = vec!["doc-a", "doc-c", "doc-e"];
        let m = evaluate_query(&qrels, "q1", &ranked, 3);
        let expected_recall = 1.0 / 3.0; // 1 relevant found / 3 total relevant
        assert!(
            (m.recall_at_k - expected_recall).abs() < 1e-9,
            "recall@3 expected {expected_recall} got {}",
            m.recall_at_k
        );
    }

    #[test]
    fn test_f1_zero_when_no_relevant() {
        let qrels = build_qrels();
        let ranked = vec!["doc-c"];
        let m = evaluate_query(&qrels, "q1", &ranked, 1);
        assert!(m.f1_at_k < 1e-9, "f1 should be 0 got {}", m.f1_at_k);
    }

    #[test]
    fn test_average_precision_perfect_ranking() {
        let mut qrels = RelevanceJudgements::new();
        qrels.add("q2", "r1", 1);
        qrels.add("q2", "r2", 1);
        qrels.add("q2", "n1", 0);
        // Perfect ranking: relevant docs first.
        let ranked = vec!["r1", "r2", "n1"];
        let m = evaluate_query(&qrels, "q2", &ranked, 3);
        // AP = (1/1 + 2/2) / 2 = 1.0
        assert!(
            (m.average_precision - 1.0).abs() < 1e-9,
            "AP should be 1.0 got {}",
            m.average_precision
        );
    }

    #[test]
    fn test_reciprocal_rank_first_relevant_at_rank3() {
        let qrels = build_qrels();
        let ranked = vec!["doc-c", "doc-e", "doc-a"];
        let m = evaluate_query(&qrels, "q1", &ranked, 3);
        let expected_rr = 1.0 / 3.0;
        assert!(
            (m.reciprocal_rank - expected_rr).abs() < 1e-9,
            "RR expected {expected_rr} got {}",
            m.reciprocal_rank
        );
    }

    #[test]
    fn test_reciprocal_rank_zero_when_no_relevant_in_k() {
        let qrels = build_qrels();
        let ranked = vec!["doc-c", "doc-e"];
        let m = evaluate_query(&qrels, "q1", &ranked, 2);
        assert!(
            m.reciprocal_rank < 1e-9,
            "RR should be 0 when no relevant in k got {}",
            m.reciprocal_rank
        );
    }

    #[test]
    fn test_ndcg_perfect_ranking_equals_one() {
        let mut qrels = RelevanceJudgements::new();
        qrels.add("q3", "d1", 2);
        qrels.add("q3", "d2", 1);
        qrels.add("q3", "d3", 0);
        let ranked = vec!["d1", "d2", "d3"];
        let m = evaluate_query(&qrels, "q3", &ranked, 3);
        assert!(
            (m.ndcg_at_k - 1.0).abs() < 1e-9,
            "NDCG@3 with perfect ranking should be 1.0 got {}",
            m.ndcg_at_k
        );
    }

    #[test]
    fn test_ndcg_reversed_ranking_less_than_one() {
        let mut qrels = RelevanceJudgements::new();
        qrels.add("q4", "d1", 2);
        qrels.add("q4", "d2", 0);
        qrels.add("q4", "d3", 0);
        // Worst ranking: irrelevant first.
        let ranked = vec!["d2", "d3", "d1"];
        let m = evaluate_query(&qrels, "q4", &ranked, 3);
        assert!(
            m.ndcg_at_k < 1.0,
            "NDCG should be < 1 for imperfect ranking got {}",
            m.ndcg_at_k
        );
        assert!(m.ndcg_at_k >= 0.0, "NDCG should be >= 0 got {}", m.ndcg_at_k);
    }

    #[test]
    fn test_map_and_mrr_over_multiple_queries() {
        let mut qrels = RelevanceJudgements::new();
        qrels.add("qa", "r1", 1);
        qrels.add("qb", "r2", 1);

        let mut ranked_lists: HashMap<String, Vec<String>> = HashMap::new();
        ranked_lists.insert("qa".to_string(), vec!["r1".to_string(), "n1".to_string()]);
        ranked_lists.insert("qb".to_string(), vec!["n1".to_string(), "r2".to_string()]);

        let (per_query, agg) = evaluate_all(&qrels, &ranked_lists, 2);
        assert_eq!(per_query.len(), 2);
        // qa: AP=1.0, RR=1.0;  qb: AP=0.5, RR=0.5
        assert!(
            (agg.map - 0.75).abs() < 1e-9,
            "MAP expected 0.75 got {}",
            agg.map
        );
        assert!(
            (agg.mrr - 0.75).abs() < 1e-9,
            "MRR expected 0.75 got {}",
            agg.mrr
        );
    }

    #[test]
    fn test_aggregate_metrics_empty_input() {
        let agg = AggregateMetrics::from_query_metrics(&[]);
        assert_eq!(agg.query_count, 0);
        assert!(agg.map.abs() < 1e-9);
    }

    #[test]
    fn test_r_precision() {
        let mut qrels = RelevanceJudgements::new();
        // 2 relevant docs
        qrels.add("q5", "a", 1);
        qrels.add("q5", "b", 1);
        qrels.add("q5", "c", 0);
        // R=2, first 2 results: one relevant + one non-relevant → R-prec = 0.5
        let ranked = vec!["a", "c", "b"];
        let m = evaluate_query(&qrels, "q5", &ranked, 3);
        assert!(
            (m.r_precision - 0.5).abs() < 1e-9,
            "R-precision expected 0.5 got {}",
            m.r_precision
        );
    }

    #[test]
    fn test_relevant_count() {
        let qrels = build_qrels();
        assert_eq!(qrels.relevant_count("q1"), 3); // doc-a(2), doc-b(1), doc-d(2)
        assert_eq!(qrels.relevant_count("nonexistent"), 0);
    }

    #[test]
    fn test_grade_defaults_to_zero() {
        let qrels = build_qrels();
        assert_eq!(qrels.grade("q1", "unknown-doc"), 0);
    }
}
