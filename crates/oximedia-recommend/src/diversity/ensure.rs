//! Diversity enforcement for recommendations.
//!
//! Provides two complementary approaches to recommendation diversity:
//!
//! 1. **Category-capping** (`DiversityEnforcer`) — greedily builds a result
//!    set while limiting the number of items per category.
//!
//! 2. **Maximal Marginal Relevance** (`MaximumMarginalRelevance` /
//!    `MmrReranker`) — the iterative MMR algorithm (Carbonell & Goldstein 1998)
//!    that selects items by trading off relevance and novelty:
//!
//! ```text
//! MMR_i = λ · relevance(i) − (1 − λ) · max_{j ∈ S} sim(i, j)
//! ```
//!
//! where `S` is the set of already-selected items and `sim` is cosine
//! similarity over binary category feature vectors.

use crate::error::RecommendResult;
use crate::{DiversitySettings, Recommendation};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Category-capping diversity enforcer
// ---------------------------------------------------------------------------

/// Diversity enforcer — greedily caps the number of items per category.
pub struct DiversityEnforcer {
    /// Maximum items per category
    max_per_category: usize,
}

impl DiversityEnforcer {
    /// Create a new diversity enforcer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_per_category: 3,
        }
    }

    /// Create with a custom per-category cap.
    #[must_use]
    pub fn with_max_per_category(max: usize) -> Self {
        Self {
            max_per_category: max,
        }
    }

    /// Enforce diversity on a list of recommendations.
    ///
    /// Items are processed in score order; any item whose categories would
    /// push a category count above `max_per_category` is dropped.
    ///
    /// When `settings.include_serendipity` is `true`, the enforced list is
    /// additionally re-ranked using MMR with
    /// `lambda = 1 - settings.serendipity_weight` (a higher serendipity_weight
    /// means more diversity, i.e., lower λ).
    ///
    /// # Errors
    ///
    /// Returns an error if enforcement fails.
    pub fn enforce_diversity(
        &self,
        recommendations: Vec<Recommendation>,
        settings: &DiversitySettings,
    ) -> RecommendResult<Vec<Recommendation>> {
        if !settings.enabled {
            return Ok(recommendations);
        }

        let mut diverse_recommendations: Vec<Recommendation> = Vec::new();
        let mut category_counts: HashMap<String, usize> = HashMap::new();

        for rec in recommendations {
            let categories = &rec.metadata.categories;

            let can_add = categories.iter().all(|category| {
                *category_counts.get(category).unwrap_or(&0) < self.max_per_category
            });

            if can_add {
                for category in categories {
                    *category_counts.entry(category.clone()).or_insert(0) += 1;
                }
                diverse_recommendations.push(rec);
            }
        }

        // Optionally apply MMR reranking for serendipity
        let mut result = if settings.include_serendipity && diverse_recommendations.len() > 1 {
            // Higher serendipity_weight → lower λ → more diversity
            let lambda = 1.0 - settings.serendipity_weight.clamp(0.0, 1.0);
            let reranker = MmrReranker::new(lambda);
            reranker.rerank(diverse_recommendations)
        } else {
            diverse_recommendations
        };

        // Assign contiguous 1-indexed ranks after reranking
        for (idx, rec) in result.iter_mut().enumerate() {
            rec.rank = idx + 1;
        }

        Ok(result)
    }

    /// Calculate a category-diversity score for a list (∈ [0, 1]).
    ///
    /// Returns the fraction of unique categories over total category
    /// assignments.  Higher is more diverse.
    #[must_use]
    pub fn calculate_diversity_score(recommendations: &[Recommendation]) -> f32 {
        if recommendations.is_empty() {
            return 0.0;
        }

        let mut all_categories: HashSet<String> = HashSet::new();
        let mut total_categories = 0usize;

        for rec in recommendations {
            for category in &rec.metadata.categories {
                all_categories.insert(category.clone());
                total_categories += 1;
            }
        }

        if total_categories == 0 {
            return 0.0;
        }

        all_categories.len() as f32 / total_categories as f32
    }
}

impl Default for DiversityEnforcer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MMR scoring primitive
// ---------------------------------------------------------------------------

/// Maximum Marginal Relevance (MMR) score calculator.
///
/// Computes: `MMR = λ · relevance − (1 − λ) · max_similarity`
pub struct MaximumMarginalRelevance {
    /// λ ∈ [0, 1] — weight on relevance vs. diversity.
    /// λ = 1 → purely relevance-ranked; λ = 0 → purely novel items.
    lambda: f32,
}

impl MaximumMarginalRelevance {
    /// Create with a custom λ.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Compute the MMR score.
    #[must_use]
    pub fn calculate_score(&self, relevance: f32, max_similarity: f32) -> f32 {
        self.lambda * relevance - (1.0 - self.lambda) * max_similarity
    }

    /// Return the λ value.
    #[must_use]
    pub fn lambda(&self) -> f32 {
        self.lambda
    }
}

impl Default for MaximumMarginalRelevance {
    fn default() -> Self {
        Self::new(0.7) // Favour relevance slightly over diversity
    }
}

// ---------------------------------------------------------------------------
// Category feature vectors (internal helpers)
// ---------------------------------------------------------------------------

/// Build a binary category feature vector for a recommendation.
fn category_vector(rec: &Recommendation, vocab: &[String]) -> Vec<f32> {
    let cat_set: HashSet<&String> = rec.metadata.categories.iter().collect();
    vocab
        .iter()
        .map(|cat| if cat_set.contains(cat) { 1.0 } else { 0.0 })
        .collect()
}

/// Cosine similarity between two equal-length f32 vectors.
/// Returns 0.0 if either vector is the zero vector.
fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Full MMR reranker
// ---------------------------------------------------------------------------

/// Iterative MMR reranking algorithm over a list of `Recommendation`s.
///
/// Greedily selects the next item maximising:
///
/// ```text
/// MMR_i = λ · score(i) − (1 − λ) · max_{j ∈ selected} cosine_sim(features(i), features(j))
/// ```
///
/// where `features` is a binary category vector over the global category
/// vocabulary.
pub struct MmrReranker {
    /// λ trade-off parameter.
    lambda: f32,
}

impl MmrReranker {
    /// Create a new reranker.
    #[must_use]
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
        }
    }

    /// Rerank `candidates` using MMR, returning the full list reordered for diversity.
    ///
    /// The candidates should be sorted by descending relevance score before
    /// calling this method (the first pick is always the highest-relevance item).
    #[must_use]
    pub fn rerank(&self, candidates: Vec<Recommendation>) -> Vec<Recommendation> {
        if candidates.len() <= 1 {
            return candidates;
        }

        // Build global category vocabulary (sorted for determinism)
        let vocab: Vec<String> = {
            let mut seen: HashSet<String> = HashSet::new();
            for rec in &candidates {
                for cat in &rec.metadata.categories {
                    seen.insert(cat.clone());
                }
            }
            let mut v: Vec<String> = seen.into_iter().collect();
            v.sort();
            v
        };

        // Pre-compute category feature vectors
        let feature_vecs: Vec<Vec<f32>> = candidates
            .iter()
            .map(|rec| category_vector(rec, &vocab))
            .collect();

        let mmr = MaximumMarginalRelevance::new(self.lambda);
        let n = candidates.len();
        let mut remaining: Vec<usize> = (0..n).collect();
        let mut selected_order: Vec<usize> = Vec::with_capacity(n);
        // Indices of already-selected items (for similarity lookup)
        let mut selected_indices: Vec<usize> = Vec::with_capacity(n);

        while !remaining.is_empty() {
            let chosen_pos = if selected_indices.is_empty() {
                // First pick: the highest-relevance item (remaining[0] assuming sorted input)
                0
            } else {
                let mut best_score = f32::NEG_INFINITY;
                let mut best_pos = 0usize;
                for (pos, &cand_idx) in remaining.iter().enumerate() {
                    let relevance = candidates[cand_idx].score;
                    // Maximum cosine similarity to any already-selected item
                    let max_sim = selected_indices
                        .iter()
                        .map(|&sel_idx| {
                            cosine_sim_f32(&feature_vecs[cand_idx], &feature_vecs[sel_idx])
                        })
                        .fold(f32::NEG_INFINITY, f32::max);
                    let max_sim = max_sim.max(0.0);
                    let score = mmr.calculate_score(relevance, max_sim);
                    if score > best_score {
                        best_score = score;
                        best_pos = pos;
                    }
                }
                best_pos
            };

            let chosen_idx = remaining.remove(chosen_pos);
            selected_indices.push(chosen_idx);
            selected_order.push(chosen_idx);
        }

        // Reconstruct the reranked list preserving ownership.
        // Each idx in selected_order is unique (derived from remaining.remove), so
        // every take() returns Some.  Filter-map makes the None branch unreachable
        // without panicking.
        let mut boxed: Vec<Option<Recommendation>> = candidates.into_iter().map(Some).collect();
        selected_order
            .into_iter()
            .filter_map(|idx| boxed[idx].take())
            .collect()
    }
}

impl Default for MmrReranker {
    fn default() -> Self {
        Self::new(0.7)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContentMetadata, Recommendation, RecommendationReason};
    use uuid::Uuid;

    fn make_rec(score: f32, categories: Vec<&str>) -> Recommendation {
        Recommendation {
            content_id: Uuid::new_v4(),
            score,
            rank: 1,
            reasons: vec![RecommendationReason::Popular { view_count: 100 }],
            metadata: ContentMetadata {
                title: format!("Item {score}"),
                description: None,
                categories: categories.into_iter().map(String::from).collect(),
                duration_ms: None,
                thumbnail_url: None,
                created_at: 0,
                avg_rating: None,
                view_count: 0,
            },
            explanation: None,
        }
    }

    // ---- DiversityEnforcer ----

    #[test]
    fn test_diversity_enforcer_creation() {
        let enforcer = DiversityEnforcer::new();
        assert_eq!(enforcer.max_per_category, 3);
    }

    #[test]
    fn test_diversity_enforcer_with_custom_cap() {
        let enforcer = DiversityEnforcer::with_max_per_category(5);
        assert_eq!(enforcer.max_per_category, 5);
    }

    #[test]
    fn test_enforce_diversity_disabled_passes_all() {
        let enforcer = DiversityEnforcer::new();
        let items = vec![
            make_rec(0.9, vec!["action"]),
            make_rec(0.8, vec!["action"]),
            make_rec(0.7, vec!["action"]),
            make_rec(0.6, vec!["action"]),
        ];
        let settings = DiversitySettings {
            enabled: false,
            ..Default::default()
        };
        let result = enforcer.enforce_diversity(items, &settings).expect("ok");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_enforce_diversity_caps_category() {
        let enforcer = DiversityEnforcer::with_max_per_category(2);
        let items = vec![
            make_rec(0.9, vec!["action"]),
            make_rec(0.8, vec!["action"]),
            make_rec(0.7, vec!["action"]), // should be dropped (3rd action)
            make_rec(0.6, vec!["drama"]),
        ];
        let settings = DiversitySettings {
            enabled: true,
            include_serendipity: false,
            ..Default::default()
        };
        let result = enforcer.enforce_diversity(items, &settings).expect("ok");
        assert_eq!(result.len(), 3);
        for (i, rec) in result.iter().enumerate() {
            assert_eq!(rec.rank, i + 1);
        }
    }

    #[test]
    fn test_enforce_diversity_assigns_contiguous_ranks() {
        let enforcer = DiversityEnforcer::new();
        let items = vec![
            make_rec(0.9, vec!["a"]),
            make_rec(0.8, vec!["b"]),
            make_rec(0.7, vec!["c"]),
        ];
        let settings = DiversitySettings {
            enabled: true,
            include_serendipity: false,
            ..Default::default()
        };
        let result = enforcer.enforce_diversity(items, &settings).expect("ok");
        for (i, rec) in result.iter().enumerate() {
            assert_eq!(rec.rank, i + 1);
        }
    }

    #[test]
    fn test_calculate_diversity_score_empty() {
        assert_eq!(DiversityEnforcer::calculate_diversity_score(&[]), 0.0);
    }

    #[test]
    fn test_calculate_diversity_score_all_same_category() {
        let items = vec![make_rec(0.9, vec!["action"]), make_rec(0.8, vec!["action"])];
        // 1 unique / 2 total = 0.5
        let score = DiversityEnforcer::calculate_diversity_score(&items);
        assert!((score - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_calculate_diversity_score_all_unique() {
        let items = vec![make_rec(0.9, vec!["action"]), make_rec(0.8, vec!["drama"])];
        // 2 unique / 2 total = 1.0
        let score = DiversityEnforcer::calculate_diversity_score(&items);
        assert!((score - 1.0).abs() < f32::EPSILON);
    }

    // ---- MaximumMarginalRelevance ----

    #[test]
    fn test_mmr_calculate_score() {
        let mmr = MaximumMarginalRelevance::new(0.7);
        let score = mmr.calculate_score(0.9, 0.5);
        // 0.7 * 0.9 - 0.3 * 0.5 = 0.63 - 0.15 = 0.48
        assert!((score - 0.48).abs() < 1e-5);
    }

    #[test]
    fn test_mmr_score_positive_when_relevant_and_novel() {
        let mmr = MaximumMarginalRelevance::new(0.7);
        let score = mmr.calculate_score(0.9, 0.0);
        assert!(score > 0.0);
    }

    #[test]
    fn test_mmr_lambda_clamped() {
        let mmr1 = MaximumMarginalRelevance::new(-0.5);
        assert!((mmr1.lambda() - 0.0).abs() < f32::EPSILON);
        let mmr2 = MaximumMarginalRelevance::new(1.5);
        assert!((mmr2.lambda() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mmr_default_lambda() {
        let mmr = MaximumMarginalRelevance::default();
        assert!((mmr.lambda() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mmr_pure_relevance() {
        let mmr = MaximumMarginalRelevance::new(1.0);
        // λ=1: score = relevance (similarity ignored)
        let s = mmr.calculate_score(0.8, 0.99);
        assert!((s - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_mmr_pure_diversity() {
        let mmr = MaximumMarginalRelevance::new(0.0);
        // λ=0: score = -max_similarity
        let s = mmr.calculate_score(0.8, 0.6);
        assert!((s - (-0.6)).abs() < 1e-5);
    }

    // ---- cosine_sim_f32 ----

    #[test]
    fn test_cosine_sim_identical() {
        let v = vec![1.0f32, 1.0, 0.0];
        assert!((cosine_sim_f32(&v, &v) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!(cosine_sim_f32(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_sim_zero_vector() {
        let a = vec![0.0f32, 0.0];
        let b = vec![1.0f32, 1.0];
        assert!(cosine_sim_f32(&a, &b).abs() < 1e-5);
    }

    // ---- MmrReranker ----

    #[test]
    fn test_mmr_reranker_empty() {
        let reranker = MmrReranker::new(0.7);
        assert!(reranker.rerank(vec![]).is_empty());
    }

    #[test]
    fn test_mmr_reranker_single_item() {
        let reranker = MmrReranker::new(0.7);
        let result = reranker.rerank(vec![make_rec(0.9, vec!["action"])]);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_mmr_reranker_first_item_is_highest_relevance() {
        // With all distinct categories the first pick is always highest-scored
        let reranker = MmrReranker::new(0.7);
        let items = vec![
            make_rec(0.9, vec!["action"]),
            make_rec(0.7, vec!["drama"]),
            make_rec(0.5, vec!["comedy"]),
        ];
        let result = reranker.rerank(items);
        assert!((result[0].score - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mmr_reranker_promotes_diverse_items() {
        // Two action items + one drama.
        // With λ = 0.4 (heavy diversity), drama should beat the 2nd action item.
        //
        // After picking action(0.9), for the remaining:
        //   action(0.8): 0.4*0.8 − 0.6*1.0 = 0.32 − 0.60 = −0.28
        //   drama(0.6):  0.4*0.6 − 0.6*0.0 = 0.24 − 0.00 =  0.24
        // → drama wins.
        let reranker = MmrReranker::new(0.4);
        let items = vec![
            make_rec(0.9, vec!["action"]),
            make_rec(0.8, vec!["action"]),
            make_rec(0.6, vec!["drama"]),
        ];
        let result = reranker.rerank(items);
        assert_eq!(result.len(), 3);
        // First pick must be highest relevance (action 0.9)
        assert!((result[0].score - 0.9).abs() < f32::EPSILON);
        // Second pick should be drama (score 0.6)
        assert!((result[1].score - 0.6).abs() < f32::EPSILON);
    }

    #[test]
    fn test_mmr_reranker_preserves_all_items() {
        let reranker = MmrReranker::new(0.7);
        let items: Vec<Recommendation> = (0..10)
            .map(|i| make_rec(1.0 - i as f32 * 0.05, vec!["cat"]))
            .collect();
        let result = reranker.rerank(items);
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_mmr_reranker_no_categories_no_panic() {
        // Items with empty category lists → zero vectors → cosine_sim = 0
        let reranker = MmrReranker::new(0.7);
        let result = reranker.rerank(vec![make_rec(0.9, vec![]), make_rec(0.7, vec![])]);
        assert_eq!(result.len(), 2);
    }

    // ---- Integration: enforce_diversity with serendipity ----

    #[test]
    fn test_enforce_diversity_with_serendipity_uses_mmr() {
        let enforcer = DiversityEnforcer::new();
        let items = vec![
            make_rec(0.9, vec!["action"]),
            make_rec(0.8, vec!["action"]),
            make_rec(0.6, vec!["drama"]),
        ];
        let settings = DiversitySettings {
            enabled: true,
            include_serendipity: true,
            serendipity_weight: 0.9, // λ = 1 − 0.9 = 0.1 → heavy diversity
            category_diversity: 0.5,
        };
        let result = enforcer.enforce_diversity(items, &settings).expect("ok");
        assert_eq!(result.len(), 3);
        // Ranks must be 1-indexed and contiguous
        for (i, rec) in result.iter().enumerate() {
            assert_eq!(rec.rank, i + 1);
        }
        // With serendipity_weight=0.9 (λ=0.1) drama should beat 2nd action
        assert!((result[1].score - 0.6).abs() < f32::EPSILON);
    }
}
