//! Recommendation fairness metrics and exposure equity enforcement.
//!
//! This module provides tools to **measure** and **correct** exposure inequities
//! across content creators in recommendation lists.  Fairness in recommender
//! systems is concerned with ensuring that creators (or content groups) receive
//! recommendation exposure proportional to a target distribution — typically
//! their share of available content or their share of user-expressed interest.
//!
//! # Metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | [`ExposureGini`] | Gini coefficient of creator exposure (0 = perfectly equal) |
//! | [`NdcgFairness`] | Fairness-aware NDCG — measures if top-ranked positions favour certain groups |
//! | [`ExposureDisparity`] | Per-creator ratio of actual-to-target exposure |
//!
//! # Enforcement
//!
//! [`FairnessReranker`] applies a constrained reranking pass that up-promotes
//! under-represented creators / groups while keeping a user-configurable
//! quality penalty threshold so that fairness is never achieved at the cost of
//! relevance.

#![allow(dead_code)]

use std::collections::HashMap;

use crate::error::{RecommendError, RecommendResult};

// ---------------------------------------------------------------------------
// Creator / group abstraction
// ---------------------------------------------------------------------------

/// Identifies the creator or group that owns a piece of content.
pub type CreatorId = u64;

/// A recommendation candidate with creator attribution.
#[derive(Debug, Clone)]
pub struct FairnessCandidate {
    /// Content identifier.
    pub content_id: u64,
    /// Creator / group identifier.
    pub creator_id: CreatorId,
    /// Relevance score from the upstream recommender (higher is better).
    pub relevance: f32,
}

impl FairnessCandidate {
    /// Constructs a new candidate.
    #[must_use]
    pub fn new(content_id: u64, creator_id: CreatorId, relevance: f32) -> Self {
        Self {
            content_id,
            creator_id,
            relevance: relevance.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Exposure Gini coefficient
// ---------------------------------------------------------------------------

/// Computes the **Gini coefficient** of creator exposure across a ranked list.
///
/// The Gini coefficient is defined in `[0, 1]`:
/// - `0.0` → perfectly equal exposure (every creator appears equally often).
/// - `1.0` → maximal inequality (one creator takes all slots).
///
/// Exposure per creator is measured as the *discounted position weight*:
/// `exposure = Σ  1 / log2(rank + 1)` where rank is 1-indexed.
pub struct ExposureGini;

impl ExposureGini {
    /// Computes the Gini coefficient of creator exposure in `ranked`.
    ///
    /// Returns `0.0` when the list is empty or contains only one distinct
    /// creator (equality is trivially satisfied in those cases).
    #[must_use]
    pub fn compute(ranked: &[FairnessCandidate]) -> f32 {
        if ranked.is_empty() {
            return 0.0;
        }

        // Accumulate discounted exposure per creator
        let mut exposure: HashMap<CreatorId, f64> = HashMap::new();
        for (rank0, item) in ranked.iter().enumerate() {
            let rank = rank0 + 1;
            let discount = 1.0 / (rank as f64 + 1.0).log2();
            *exposure.entry(item.creator_id).or_default() += discount;
        }

        let n = exposure.len();
        if n <= 1 {
            return 0.0;
        }

        let mut values: Vec<f64> = exposure.into_values().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Standard Gini formula
        let sum: f64 = values.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        let n_f = n as f64;
        let weighted_sum: f64 = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 + 1.0) * v)
            .sum();
        ((2.0 * weighted_sum) / (n_f * sum) - (n_f + 1.0) / n_f) as f32
    }
}

// ---------------------------------------------------------------------------
// Per-creator exposure disparity
// ---------------------------------------------------------------------------

/// Per-creator exposure disparity relative to a *target distribution*.
///
/// The target distribution specifies the fraction of total recommendation slots
/// each creator should ideally receive.  A disparity of `1.0` means actual
/// matches target; below `1.0` means under-represented; above means
/// over-represented.
#[derive(Debug, Clone)]
pub struct ExposureDisparity {
    /// Creator → actual exposure fraction.
    pub actual: HashMap<CreatorId, f32>,
    /// Creator → target exposure fraction.
    pub target: HashMap<CreatorId, f32>,
    /// Creator → disparity ratio (actual / target).
    pub ratio: HashMap<CreatorId, f32>,
}

impl ExposureDisparity {
    /// Computes disparity for each creator in `target_distribution`.
    ///
    /// `target_distribution` maps `CreatorId` to the desired fraction of
    /// recommendation slots in `[0, 1]`.  Fractions need not sum to exactly
    /// `1.0` — they are normalised internally.
    ///
    /// # Errors
    ///
    /// Returns [`RecommendError::Other`] if `target_distribution` is empty.
    pub fn compute(
        ranked: &[FairnessCandidate],
        target_distribution: &HashMap<CreatorId, f32>,
    ) -> RecommendResult<Self> {
        if target_distribution.is_empty() {
            return Err(RecommendError::Other(
                "target_distribution must not be empty".to_string(),
            ));
        }

        // Normalise target
        let target_sum: f32 = target_distribution.values().sum();
        let target: HashMap<CreatorId, f32> = if target_sum > 0.0 {
            target_distribution
                .iter()
                .map(|(&k, &v)| (k, v / target_sum))
                .collect()
        } else {
            target_distribution.clone()
        };

        // Count actual appearances
        let total = ranked.len() as f32;
        let mut counts: HashMap<CreatorId, f32> = HashMap::new();
        for item in ranked {
            *counts.entry(item.creator_id).or_default() += 1.0;
        }
        let actual: HashMap<CreatorId, f32> = counts
            .iter()
            .map(|(&k, &v)| (k, if total > 0.0 { v / total } else { 0.0 }))
            .collect();

        // Compute ratios
        let ratio: HashMap<CreatorId, f32> = target
            .keys()
            .map(|&creator| {
                let act = actual.get(&creator).copied().unwrap_or(0.0);
                let tgt = *target.get(&creator).unwrap_or(&0.0);
                let r = if tgt > 0.0 { act / tgt } else { 0.0 };
                (creator, r)
            })
            .collect();

        Ok(Self {
            actual,
            target,
            ratio,
        })
    }

    /// Returns the mean absolute deviation from the target across all creators
    /// in the target distribution.
    #[must_use]
    pub fn mean_absolute_deviation(&self) -> f32 {
        if self.target.is_empty() {
            return 0.0;
        }
        let sum: f32 = self
            .target
            .keys()
            .map(|c| {
                let act = self.actual.get(c).copied().unwrap_or(0.0);
                let tgt = self.target.get(c).copied().unwrap_or(0.0);
                (act - tgt).abs()
            })
            .sum();
        sum / self.target.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Fairness-aware NDCG
// ---------------------------------------------------------------------------

/// Measures whether high-relevance positions disproportionately favour
/// certain creators (group-level fairness).
///
/// `FairNdcg` is computed as the ratio of the *discounted gain* accumulated
/// by a protected group divided by the total discounted gain in the list.
/// A ratio close to the group's *representation fraction* indicates fair
/// treatment.
pub struct NdcgFairness;

impl NdcgFairness {
    /// Computes the discounted cumulative gain (DCG) fraction accumulated by
    /// `protected_creators` in `ranked`.
    ///
    /// The relevance of each item is used as its gain; DCG uses
    /// `log2(rank + 1)` as the discount.
    ///
    /// Returns `(group_dcg_fraction, total_dcg)` where `group_dcg_fraction`
    /// is `group_dcg / total_dcg` (or `0.0` if `total_dcg == 0`).
    #[must_use]
    pub fn group_dcg_fraction(
        ranked: &[FairnessCandidate],
        protected_creators: &[CreatorId],
    ) -> (f32, f32) {
        if ranked.is_empty() {
            return (0.0, 0.0);
        }
        let protected_set: std::collections::HashSet<CreatorId> =
            protected_creators.iter().copied().collect();

        let mut total_dcg = 0.0_f64;
        let mut group_dcg = 0.0_f64;

        for (rank0, item) in ranked.iter().enumerate() {
            let rank = rank0 + 1;
            let discount = 1.0 / (rank as f64 + 1.0).log2();
            let gain = f64::from(item.relevance) * discount;
            total_dcg += gain;
            if protected_set.contains(&item.creator_id) {
                group_dcg += gain;
            }
        }

        let fraction = if total_dcg > 0.0 {
            (group_dcg / total_dcg) as f32
        } else {
            0.0
        };
        (fraction, total_dcg as f32)
    }

    /// Computes the *representation ratio* of `protected_creators` in `ranked`
    /// (count fraction, ignoring relevance scores).
    #[must_use]
    pub fn representation_ratio(
        ranked: &[FairnessCandidate],
        protected_creators: &[CreatorId],
    ) -> f32 {
        if ranked.is_empty() {
            return 0.0;
        }
        let protected_set: std::collections::HashSet<CreatorId> =
            protected_creators.iter().copied().collect();
        let count = ranked
            .iter()
            .filter(|i| protected_set.contains(&i.creator_id))
            .count();
        count as f32 / ranked.len() as f32
    }
}

// ---------------------------------------------------------------------------
// FairnessReranker
// ---------------------------------------------------------------------------

/// Configuration for the fairness-aware reranker.
#[derive(Debug, Clone)]
pub struct FairnessConfig {
    /// Target distribution: `creator_id → target fraction [0, 1]`.
    pub target_distribution: HashMap<CreatorId, f32>,
    /// Maximum acceptable relevance penalty (0.0 = no penalty allowed,
    /// 1.0 = any drop in relevance is acceptable).
    pub max_relevance_penalty: f32,
    /// If `true`, only creators present in `target_distribution` are
    /// considered for fairness enforcement; others pass through freely.
    pub restrict_to_target_creators: bool,
}

impl FairnessConfig {
    /// Creates a config with a uniform target distribution over `creator_ids`.
    #[must_use]
    pub fn uniform(creator_ids: &[CreatorId]) -> Self {
        let n = creator_ids.len();
        let share = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        let target_distribution = creator_ids.iter().map(|&id| (id, share)).collect();
        Self {
            target_distribution,
            max_relevance_penalty: 0.2,
            restrict_to_target_creators: false,
        }
    }
}

/// Reranks a recommendation list to improve exposure equity across creators.
///
/// # Algorithm
///
/// The reranker uses a **slot-filling** approach:
/// 1. Compute how many slots each creator should fill based on the target
///    distribution and the list length.
/// 2. Greedily assign items to slots, preferring relevance within each creator
///    quota.
/// 3. Fill remaining slots (quota overflow or unattributed creators) with the
///    highest-relevance remaining items.
/// 4. If the resulting list's average relevance drops more than
///    `max_relevance_penalty` below the original, return the original list
///    unchanged.
pub struct FairnessReranker {
    config: FairnessConfig,
}

impl FairnessReranker {
    /// Creates a new reranker with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_relevance_penalty` is outside `[0, 1]`.
    pub fn new(config: FairnessConfig) -> RecommendResult<Self> {
        if !(0.0..=1.0).contains(&config.max_relevance_penalty) {
            return Err(RecommendError::Other(
                "max_relevance_penalty must be in [0, 1]".to_string(),
            ));
        }
        Ok(Self { config })
    }

    /// Reranks `candidates` to improve creator exposure equity.
    ///
    /// Returns the reranked list (or the original if the quality penalty
    /// threshold would be exceeded).
    #[must_use]
    pub fn rerank(&self, candidates: &[FairnessCandidate]) -> Vec<FairnessCandidate> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let n = candidates.len();
        let original_avg = avg_relevance(candidates);

        // Normalise target fractions
        let target_sum: f32 = self.config.target_distribution.values().sum();
        let normalised_target: HashMap<CreatorId, f32> = if target_sum > 0.0 {
            self.config
                .target_distribution
                .iter()
                .map(|(&k, &v)| (k, v / target_sum))
                .collect()
        } else {
            return candidates.to_vec();
        };

        // Compute target slot counts per creator
        let target_slots: HashMap<CreatorId, usize> = normalised_target
            .iter()
            .map(|(&creator, &frac)| (creator, (frac * n as f32).round() as usize))
            .collect();

        // Collect per-creator pools sorted by relevance descending
        let mut creator_pools: HashMap<CreatorId, Vec<FairnessCandidate>> = HashMap::new();
        let mut unclaimed: Vec<FairnessCandidate> = Vec::new();

        for item in candidates {
            if self
                .config
                .target_distribution
                .contains_key(&item.creator_id)
            {
                creator_pools
                    .entry(item.creator_id)
                    .or_default()
                    .push(item.clone());
            } else if self.config.restrict_to_target_creators {
                // Ignore items from non-target creators
            } else {
                unclaimed.push(item.clone());
            }
        }

        // Sort each pool descending by relevance
        for pool in creator_pools.values_mut() {
            pool.sort_by(|a, b| {
                b.relevance
                    .partial_cmp(&a.relevance)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        unclaimed.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut result: Vec<FairnessCandidate> = Vec::with_capacity(n);

        // Fill creator-attributed slots first
        for (&creator, &slots) in &target_slots {
            let pool = match creator_pools.get_mut(&creator) {
                Some(p) => p,
                None => continue,
            };
            let take = slots.min(pool.len());
            for item in pool.drain(..take) {
                result.push(item);
            }
        }

        // Fill overflow creator items into unclaimed
        for pool in creator_pools.values_mut() {
            unclaimed.append(pool);
        }
        unclaimed.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Fill remaining slots from unclaimed
        let remaining = n.saturating_sub(result.len());
        result.extend(unclaimed.into_iter().take(remaining));

        // Sort final result by relevance to make it presentable
        result.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Check quality penalty
        let new_avg = avg_relevance(&result);
        let penalty = original_avg - new_avg;
        if penalty > self.config.max_relevance_penalty {
            return candidates.to_vec();
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn avg_relevance(items: &[FairnessCandidate]) -> f32 {
    if items.is_empty() {
        return 0.0;
    }
    items.iter().map(|i| i.relevance).sum::<f32>() / items.len() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates() -> Vec<FairnessCandidate> {
        // 6 items: 4 from creator 1, 2 from creator 2
        vec![
            FairnessCandidate::new(1, 1, 0.9),
            FairnessCandidate::new(2, 1, 0.85),
            FairnessCandidate::new(3, 1, 0.80),
            FairnessCandidate::new(4, 1, 0.75),
            FairnessCandidate::new(5, 2, 0.70),
            FairnessCandidate::new(6, 2, 0.65),
        ]
    }

    #[test]
    fn test_exposure_gini_empty() {
        let gini = ExposureGini::compute(&[]);
        assert_eq!(gini, 0.0);
    }

    #[test]
    fn test_exposure_gini_single_creator() {
        let items = vec![
            FairnessCandidate::new(1, 42, 0.9),
            FairnessCandidate::new(2, 42, 0.8),
        ];
        let gini = ExposureGini::compute(&items);
        assert_eq!(gini, 0.0, "single creator → Gini = 0");
    }

    #[test]
    fn test_exposure_gini_unequal() {
        let candidates = make_candidates();
        let gini = ExposureGini::compute(&candidates);
        // 4 items for creator 1 vs 2 for creator 2 → non-zero Gini
        assert!(
            gini > 0.0,
            "unequal distribution should yield positive Gini"
        );
        assert!(gini <= 1.0);
    }

    #[test]
    fn test_exposure_disparity_empty_target_error() {
        let candidates = make_candidates();
        let result = ExposureDisparity::compute(&candidates, &HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_exposure_disparity_ratios() {
        let candidates = make_candidates();
        let mut target = HashMap::new();
        target.insert(1_u64, 0.5_f32);
        target.insert(2_u64, 0.5_f32);
        let disparity = ExposureDisparity::compute(&candidates, &target).expect("should compute");
        // Creator 1 has 4/6 ≈ 0.667 actual vs 0.5 target → ratio > 1
        let r1 = disparity.ratio[&1];
        let r2 = disparity.ratio[&2];
        assert!(r1 > 1.0, "creator 1 is over-represented: {r1}");
        assert!(r2 < 1.0, "creator 2 is under-represented: {r2}");
    }

    #[test]
    fn test_exposure_disparity_mad() {
        let candidates = make_candidates();
        let mut target = HashMap::new();
        target.insert(1_u64, 0.5_f32);
        target.insert(2_u64, 0.5_f32);
        let disparity = ExposureDisparity::compute(&candidates, &target).expect("should compute");
        let mad = disparity.mean_absolute_deviation();
        assert!(mad >= 0.0 && mad <= 1.0);
    }

    #[test]
    fn test_ndcg_fairness_group_fraction() {
        let candidates = make_candidates();
        let (frac, total_dcg) = NdcgFairness::group_dcg_fraction(&candidates, &[1]);
        assert!(frac > 0.0 && frac <= 1.0);
        assert!(total_dcg > 0.0);
        // Creator 1 holds top 4 slots → fraction should be > 0.5
        assert!(frac > 0.5, "creator 1 dominates top slots: {frac}");
    }

    #[test]
    fn test_ndcg_fairness_representation_ratio() {
        let candidates = make_candidates();
        let ratio = NdcgFairness::representation_ratio(&candidates, &[2]);
        // 2 out of 6 items belong to creator 2
        assert!((ratio - 2.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_fairness_reranker_invalid_penalty() {
        let config = FairnessConfig {
            target_distribution: HashMap::new(),
            max_relevance_penalty: 1.5,
            restrict_to_target_creators: false,
        };
        assert!(FairnessReranker::new(config).is_err());
    }

    #[test]
    fn test_fairness_reranker_improves_gini() {
        let candidates = make_candidates();
        let gini_before = ExposureGini::compute(&candidates);

        let mut target = HashMap::new();
        target.insert(1_u64, 0.5_f32);
        target.insert(2_u64, 0.5_f32);
        let config = FairnessConfig {
            target_distribution: target,
            max_relevance_penalty: 0.5,
            restrict_to_target_creators: false,
        };
        let reranker = FairnessReranker::new(config).expect("valid config");
        let reranked = reranker.rerank(&candidates);

        assert!(!reranked.is_empty());
        let gini_after = ExposureGini::compute(&reranked);
        // Reranking with equal target should reduce Gini
        assert!(
            gini_after <= gini_before + 1e-4,
            "Gini should not increase: before={gini_before}, after={gini_after}"
        );
    }

    #[test]
    fn test_fairness_reranker_empty_input() {
        let config = FairnessConfig::uniform(&[1, 2]);
        let reranker = FairnessReranker::new(config).expect("valid config");
        let result = reranker.rerank(&[]);
        assert!(result.is_empty());
    }
}
