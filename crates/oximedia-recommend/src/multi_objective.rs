//! Multi-objective optimization for recommendation ranking.
//!
//! Balances three competing objectives when ranking recommendation lists:
//!
//! 1. **Engagement** – how likely the user is to interact (click, watch, rate).
//! 2. **Diversity** – how varied the recommended set is across categories.
//! 3. **Freshness** – how recently the recommended content was published.
//!
//! The module provides:
//! - A weighted scalarisation approach (simple, fast, interpretable).
//! - A Pareto-front approximation that identifies non-dominated solutions.
//! - A marginal-diversity re-ranker (greedy submodular maximisation) to
//!   increase catalogue coverage without sacrificing relevance.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Objective scores
// ---------------------------------------------------------------------------

/// Per-item scores along each recommendation objective.
#[derive(Debug, Clone)]
pub struct ObjectiveScores {
    /// Item identifier.
    pub item_id: String,
    /// Engagement signal in [0, 1].
    pub engagement: f64,
    /// Diversity contribution in [0, 1] (lower if already represented in the list).
    pub diversity: f64,
    /// Content freshness in [0, 1] (1.0 = brand new, 0.0 = very old).
    pub freshness: f64,
}

impl ObjectiveScores {
    /// Create a new objective scores entry.
    #[must_use]
    pub fn new(
        item_id: impl Into<String>,
        engagement: f64,
        diversity: f64,
        freshness: f64,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            engagement: engagement.clamp(0.0, 1.0),
            diversity: diversity.clamp(0.0, 1.0),
            freshness: freshness.clamp(0.0, 1.0),
        }
    }

    /// Return a clamped copy with all objectives in [0, 1].
    #[must_use]
    pub fn clamped(&self) -> Self {
        Self {
            item_id: self.item_id.clone(),
            engagement: self.engagement.clamp(0.0, 1.0),
            diversity: self.diversity.clamp(0.0, 1.0),
            freshness: self.freshness.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Objective weights
// ---------------------------------------------------------------------------

/// Weights for the three objectives.
#[derive(Debug, Clone)]
pub struct ObjectiveWeights {
    /// Weight for engagement (must be ≥ 0).
    pub engagement: f64,
    /// Weight for diversity (must be ≥ 0).
    pub diversity: f64,
    /// Weight for freshness (must be ≥ 0).
    pub freshness: f64,
}

impl ObjectiveWeights {
    /// Create and normalise weights so they sum to 1.
    ///
    /// # Panics
    ///
    /// Does not panic; if all weights are 0 the default (equal weight) is used.
    #[must_use]
    pub fn new(engagement: f64, diversity: f64, freshness: f64) -> Self {
        let sum = engagement + diversity + freshness;
        if sum < f64::EPSILON {
            // All zero → equal weight
            return Self {
                engagement: 1.0 / 3.0,
                diversity: 1.0 / 3.0,
                freshness: 1.0 / 3.0,
            };
        }
        Self {
            engagement: engagement / sum,
            diversity: diversity / sum,
            freshness: freshness / sum,
        }
    }

    /// Weighted scalar score for a set of objective scores.
    #[must_use]
    pub fn scalar(&self, scores: &ObjectiveScores) -> f64 {
        self.engagement * scores.engagement
            + self.diversity * scores.diversity
            + self.freshness * scores.freshness
    }
}

impl Default for ObjectiveWeights {
    fn default() -> Self {
        // Equal-weight balanced default
        Self::new(1.0, 1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Pareto dominance
// ---------------------------------------------------------------------------

/// Returns `true` if `a` Pareto-dominates `b`.
///
/// `a` dominates `b` iff it is ≥ `b` on all objectives and strictly > on at least one.
#[must_use]
pub fn pareto_dominates(a: &ObjectiveScores, b: &ObjectiveScores) -> bool {
    let ge =
        a.engagement >= b.engagement && a.diversity >= b.diversity && a.freshness >= b.freshness;
    let gt = a.engagement > b.engagement || a.diversity > b.diversity || a.freshness > b.freshness;
    ge && gt
}

/// Extract the Pareto frontier from a set of items.
///
/// Returns items that are not dominated by any other item in the set.
#[must_use]
pub fn pareto_frontier(items: &[ObjectiveScores]) -> Vec<&ObjectiveScores> {
    items
        .iter()
        .filter(|candidate| !items.iter().any(|other| pareto_dominates(other, candidate)))
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-objective optimiser
// ---------------------------------------------------------------------------

/// An item with its computed category label (for diversity tracking).
#[derive(Debug, Clone)]
pub struct CategorisedItem {
    /// Item identifier.
    pub item_id: String,
    /// Categories this item belongs to.
    pub categories: Vec<String>,
    /// Objective scores.
    pub scores: ObjectiveScores,
}

impl CategorisedItem {
    /// Create a new categorised item.
    #[must_use]
    pub fn new(
        item_id: impl Into<String>,
        categories: Vec<String>,
        engagement: f64,
        freshness: f64,
    ) -> Self {
        let id: String = item_id.into();
        let scores = ObjectiveScores::new(id.clone(), engagement, 1.0, freshness);
        Self {
            item_id: id,
            categories,
            scores,
        }
    }
}

/// Result of multi-objective ranking.
#[derive(Debug, Clone)]
pub struct RankedItem {
    /// Item identifier.
    pub item_id: String,
    /// Final combined score.
    pub score: f64,
    /// Component scores used to compute the final score.
    pub components: ObjectiveScores,
    /// Rank in the output list (1-indexed).
    pub rank: usize,
}

/// Multi-objective optimiser for recommendation lists.
///
/// Combines weighted scalarisation with a greedy diversity-promoting
/// re-ranker to produce a balanced recommendation list.
#[derive(Debug, Clone)]
pub struct MultiObjectiveOptimiser {
    /// Weights for the three objectives.
    weights: ObjectiveWeights,
    /// Minimum diversity gain required to prefer a diverse item over a
    /// marginally less engaging one (0.0 = pure diversity, 1.0 = never swap).
    diversity_threshold: f64,
}

impl MultiObjectiveOptimiser {
    /// Create a new optimiser.
    #[must_use]
    pub fn new(weights: ObjectiveWeights, diversity_threshold: f64) -> Self {
        Self {
            weights,
            diversity_threshold: diversity_threshold.clamp(0.0, 1.0),
        }
    }

    /// Rank items using weighted scalarisation only (fast, no re-ranking).
    #[must_use]
    pub fn rank_weighted(&self, items: &[ObjectiveScores]) -> Vec<RankedItem> {
        let mut scored: Vec<(f64, &ObjectiveScores)> =
            items.iter().map(|s| (self.weights.scalar(s), s)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .enumerate()
            .map(|(idx, (score, s))| RankedItem {
                item_id: s.item_id.clone(),
                score,
                components: s.clone(),
                rank: idx + 1,
            })
            .collect()
    }

    /// Rank items using greedy marginal-diversity re-ranking.
    ///
    /// After selecting the top item, each subsequent pick is the unselected
    /// item that maximises:  `(1 − λ) * weighted_score + λ * marginal_diversity`
    /// where λ = `1 − diversity_threshold` and marginal diversity is the
    /// fraction of the item's categories not yet covered.
    #[must_use]
    pub fn rank_diverse(&self, items: &[CategorisedItem], limit: usize) -> Vec<RankedItem> {
        if items.is_empty() {
            return Vec::new();
        }

        let lambda = 1.0 - self.diversity_threshold; // diversity pressure
        let mut covered_categories: HashMap<String, u32> = HashMap::new();
        let mut selected: Vec<usize> = Vec::with_capacity(limit);
        let mut remaining: Vec<usize> = (0..items.len()).collect();

        while selected.len() < limit && !remaining.is_empty() {
            let mut best_idx_in_remaining = 0;
            let mut best_score = f64::NEG_INFINITY;

            for (pos, &item_idx) in remaining.iter().enumerate() {
                let item = &items[item_idx];
                let ws = self.weights.scalar(&item.scores);

                // Marginal diversity: fraction of new categories
                let total_cats = item.categories.len();
                let new_cats = item
                    .categories
                    .iter()
                    .filter(|c| !covered_categories.contains_key(*c))
                    .count();
                let marginal_div = if total_cats > 0 {
                    new_cats as f64 / total_cats as f64
                } else {
                    0.0
                };

                let combined = (1.0 - lambda) * ws + lambda * marginal_div;
                if combined > best_score {
                    best_score = combined;
                    best_idx_in_remaining = pos;
                }
            }

            let chosen_item_idx = remaining.remove(best_idx_in_remaining);
            selected.push(chosen_item_idx);

            // Update covered categories
            for cat in &items[chosen_item_idx].categories {
                *covered_categories.entry(cat.clone()).or_insert(0) += 1;
            }
        }

        selected
            .into_iter()
            .enumerate()
            .map(|(rank_idx, item_idx)| {
                let item = &items[item_idx];
                RankedItem {
                    item_id: item.item_id.clone(),
                    score: self.weights.scalar(&item.scores),
                    components: item.scores.clone(),
                    rank: rank_idx + 1,
                }
            })
            .collect()
    }

    /// Return the Pareto frontier of objective scores.
    #[must_use]
    pub fn pareto_frontier<'a>(&self, items: &'a [ObjectiveScores]) -> Vec<&'a ObjectiveScores> {
        pareto_frontier(items)
    }
}

impl Default for MultiObjectiveOptimiser {
    fn default() -> Self {
        Self::new(ObjectiveWeights::default(), 0.5)
    }
}

// ---------------------------------------------------------------------------
// Freshness utilities
// ---------------------------------------------------------------------------

/// Compute a freshness score from a publication age in days.
///
/// Uses an exponential decay with the given half-life in days.
#[must_use]
pub fn freshness_score(age_days: f64, half_life_days: f64) -> f64 {
    if half_life_days <= 0.0 {
        return 0.0;
    }
    let decay = (-(age_days / half_life_days) * std::f64::consts::LN_2).exp();
    decay.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores(id: &str, e: f64, d: f64, f: f64) -> ObjectiveScores {
        ObjectiveScores::new(id, e, d, f)
    }

    // ---- ObjectiveWeights ----

    #[test]
    fn test_weights_normalise_to_one() {
        let w = ObjectiveWeights::new(2.0, 1.0, 1.0);
        let sum = w.engagement + w.diversity + w.freshness;
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weights_all_zero_gives_equal() {
        let w = ObjectiveWeights::new(0.0, 0.0, 0.0);
        assert!((w.engagement - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_score() {
        let w = ObjectiveWeights::new(1.0, 0.0, 0.0); // pure engagement
        let s = make_scores("a", 0.8, 0.5, 0.3);
        assert!((w.scalar(&s) - 0.8).abs() < 1e-10);
    }

    // ---- Pareto dominance ----

    #[test]
    fn test_pareto_dominates_true() {
        let a = make_scores("a", 0.9, 0.9, 0.9);
        let b = make_scores("b", 0.5, 0.5, 0.5);
        assert!(pareto_dominates(&a, &b));
        assert!(!pareto_dominates(&b, &a));
    }

    #[test]
    fn test_pareto_dominates_false_for_incomparable() {
        let a = make_scores("a", 0.9, 0.1, 0.5);
        let b = make_scores("b", 0.1, 0.9, 0.5);
        assert!(!pareto_dominates(&a, &b));
        assert!(!pareto_dominates(&b, &a));
    }

    #[test]
    fn test_pareto_frontier_returns_non_dominated() {
        let items = vec![
            make_scores("a", 0.9, 0.9, 0.9), // dominates all below
            make_scores("b", 0.5, 0.5, 0.5),
            make_scores("c", 0.1, 0.1, 0.1),
        ];
        let frontier = pareto_frontier(&items);
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].item_id, "a");
    }

    #[test]
    fn test_pareto_frontier_incomparable_both_on_frontier() {
        let items = vec![
            make_scores("a", 1.0, 0.0, 0.5),
            make_scores("b", 0.0, 1.0, 0.5),
        ];
        let frontier = pareto_frontier(&items);
        assert_eq!(frontier.len(), 2);
    }

    // ---- MultiObjectiveOptimiser ----

    #[test]
    fn test_rank_weighted_ordering() {
        let opt = MultiObjectiveOptimiser::default();
        let items = vec![
            make_scores("low", 0.1, 0.1, 0.1),
            make_scores("high", 0.9, 0.9, 0.9),
            make_scores("mid", 0.5, 0.5, 0.5),
        ];
        let ranked = opt.rank_weighted(&items);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].item_id, "high");
        assert_eq!(ranked[2].item_id, "low");
    }

    #[test]
    fn test_rank_weighted_assigns_correct_ranks() {
        let opt = MultiObjectiveOptimiser::default();
        let items = vec![
            make_scores("a", 0.8, 0.8, 0.8),
            make_scores("b", 0.2, 0.2, 0.2),
        ];
        let ranked = opt.rank_weighted(&items);
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[1].rank, 2);
    }

    #[test]
    fn test_rank_diverse_limits_output() {
        let opt = MultiObjectiveOptimiser::new(ObjectiveWeights::default(), 0.5);
        let items: Vec<CategorisedItem> = (0..10)
            .map(|i| CategorisedItem::new(format!("item{i}"), vec!["cat".to_string()], 0.5, 0.5))
            .collect();
        let ranked = opt.rank_diverse(&items, 3);
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn test_rank_diverse_promotes_new_categories() {
        // Two categories: "action" and "drama"
        // High-engagement items: all action; mid-engagement: drama
        let opt = MultiObjectiveOptimiser::new(ObjectiveWeights::new(0.5, 0.5, 0.0), 0.0);
        let mut items = vec![
            CategorisedItem::new("a1", vec!["action".to_string()], 0.9, 0.5),
            CategorisedItem::new("a2", vec!["action".to_string()], 0.85, 0.5),
            CategorisedItem::new("d1", vec!["drama".to_string()], 0.5, 0.5),
        ];
        // Update diversity scores based on category
        for item in &mut items {
            item.scores.diversity = if item.categories.contains(&"drama".to_string()) {
                1.0
            } else {
                0.3
            };
        }
        let ranked = opt.rank_diverse(&items, 3);
        // Should include the drama item to cover diversity
        let has_drama = ranked.iter().any(|r| r.item_id == "d1");
        assert!(has_drama);
    }

    #[test]
    fn test_rank_diverse_empty_items() {
        let opt = MultiObjectiveOptimiser::default();
        let result = opt.rank_diverse(&[], 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rank_diverse_rank_field() {
        let opt = MultiObjectiveOptimiser::default();
        let items = vec![
            CategorisedItem::new("x", vec![], 0.8, 0.8),
            CategorisedItem::new("y", vec![], 0.5, 0.5),
        ];
        let ranked = opt.rank_diverse(&items, 2);
        assert_eq!(ranked[0].rank, 1);
        assert_eq!(ranked[1].rank, 2);
    }

    // ---- freshness_score ----

    #[test]
    fn test_freshness_brand_new() {
        assert!((freshness_score(0.0, 7.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_freshness_half_life() {
        let s = freshness_score(7.0, 7.0);
        assert!((s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_freshness_zero_half_life() {
        assert!((freshness_score(1.0, 0.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_freshness_decays_monotonically() {
        let s1 = freshness_score(1.0, 7.0);
        let s2 = freshness_score(7.0, 7.0);
        let s3 = freshness_score(30.0, 7.0);
        assert!(s1 > s2);
        assert!(s2 > s3);
    }

    #[test]
    fn test_objective_scores_clamped() {
        let s = ObjectiveScores::new("x", 1.5, -0.5, 2.0);
        let c = s.clamped();
        assert!((c.engagement - 1.0).abs() < f64::EPSILON);
        assert!(c.diversity.abs() < f64::EPSILON);
        assert!((c.freshness - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pareto_frontier_via_optimiser() {
        let opt = MultiObjectiveOptimiser::default();
        let items = vec![
            make_scores("a", 1.0, 0.0, 0.5),
            make_scores("b", 0.0, 1.0, 0.5),
            make_scores("c", 0.0, 0.0, 0.0),
        ];
        let frontier = opt.pareto_frontier(&items);
        assert_eq!(frontier.len(), 2);
        let ids: Vec<&str> = frontier.iter().map(|f| f.item_id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
    }
}
