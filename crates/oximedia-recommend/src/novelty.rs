//! Novelty and familiarity scoring for content recommendations.
//!
//! [`NoveltyScorer`] measures how "new" a candidate item is relative to a
//! user's interaction history.  A candidate the user has never seen scores
//! `1.0` (maximally novel); one they have already interacted with scores `0.0`
//! (no novelty / maximally familiar).
//!
//! The scorer also exposes a continuous *familiarity* score that reflects how
//! many times an item appears in a user's history relative to the total number
//! of interactions, allowing downstream ranking to trade off novelty against
//! comfort.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// NoveltyScorer
// ---------------------------------------------------------------------------

/// Scores candidate items by their novelty relative to a known interaction set.
///
/// # Example
///
/// ```
/// use oximedia_recommend::novelty::NoveltyScorer;
///
/// let known = &[1_u64, 2, 3, 2, 3, 3];
/// assert_eq!(NoveltyScorer::score(known, 99), 1.0);
/// assert_eq!(NoveltyScorer::score(known, 1), 0.0);
/// ```
pub struct NoveltyScorer;

impl NoveltyScorer {
    /// Creates a new [`NoveltyScorer`] (stateless; provided for ergonomic
    /// object construction if preferred over the static methods).
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Binary novelty score.
    ///
    /// Returns `1.0` when `candidate_id` is **not** in `known_ids`, and `0.0`
    /// when it is.  Duplicates in `known_ids` are ignored; only set membership
    /// matters.
    ///
    /// # Arguments
    ///
    /// * `known_ids` – content IDs the user has already interacted with.
    /// * `candidate_id` – item to evaluate.
    #[must_use]
    pub fn score(known_ids: &[u64], candidate_id: u64) -> f32 {
        let known: HashSet<u64> = known_ids.iter().copied().collect();
        if known.contains(&candidate_id) {
            0.0
        } else {
            1.0
        }
    }

    /// Soft novelty score in `[0, 1]`.
    ///
    /// Instead of hard binary membership, the score is based on how often
    /// `candidate_id` appears in `known_ids` relative to the length of the
    /// list:
    ///
    /// `soft_novelty = 1.0 − (count(candidate) / total_interactions)`
    ///
    /// * `1.0` → item never seen.
    /// * `0.0` → all interactions in history are this item (maximally familiar).
    ///
    /// When `known_ids` is empty, returns `1.0`.
    #[must_use]
    pub fn soft_score(known_ids: &[u64], candidate_id: u64) -> f32 {
        if known_ids.is_empty() {
            return 1.0;
        }

        let count = known_ids.iter().filter(|&&id| id == candidate_id).count();
        if count == 0 {
            return 1.0;
        }

        let familiarity = count as f32 / known_ids.len() as f32;
        1.0 - familiarity
    }

    /// Compute novelty scores for a slice of candidates at once.
    ///
    /// Returns a `Vec<f32>` aligned with `candidates`.  Uses binary scoring
    /// (same as [`score`](Self::score)).
    #[must_use]
    pub fn score_batch(known_ids: &[u64], candidates: &[u64]) -> Vec<f32> {
        let known: HashSet<u64> = known_ids.iter().copied().collect();
        candidates
            .iter()
            .map(|&cid| if known.contains(&cid) { 0.0 } else { 1.0 })
            .collect()
    }

    /// Returns the familiarity frequency map: item_id → proportion of
    /// interactions in `known_ids`.
    ///
    /// Useful for visualising which items dominate a user's history.
    #[must_use]
    pub fn familiarity_map(known_ids: &[u64]) -> HashMap<u64, f32> {
        if known_ids.is_empty() {
            return HashMap::new();
        }

        let mut counts: HashMap<u64, u32> = HashMap::new();
        for &id in known_ids {
            *counts.entry(id).or_insert(0) += 1;
        }

        let total = known_ids.len() as f32;
        counts
            .into_iter()
            .map(|(id, count)| (id, count as f32 / total))
            .collect()
    }
}

impl Default for NoveltyScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_unknown_item_is_one() {
        assert_eq!(NoveltyScorer::score(&[1, 2, 3], 99), 1.0);
    }

    #[test]
    fn test_score_known_item_is_zero() {
        assert_eq!(NoveltyScorer::score(&[1, 2, 3], 2), 0.0);
    }

    #[test]
    fn test_score_empty_known_list() {
        assert_eq!(NoveltyScorer::score(&[], 5), 1.0);
    }

    #[test]
    fn test_score_with_duplicates_in_known() {
        // Duplicates should not affect binary score
        assert_eq!(NoveltyScorer::score(&[1, 1, 1], 1), 0.0);
        assert_eq!(NoveltyScorer::score(&[1, 1, 1], 2), 1.0);
    }

    #[test]
    fn test_soft_score_never_seen() {
        assert!((NoveltyScorer::soft_score(&[1, 2, 3], 99) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soft_score_seen_once_out_of_four() {
        // count=1, total=4 → familiarity=0.25 → soft_novelty=0.75
        let s = NoveltyScorer::soft_score(&[1, 2, 3, 1], 2);
        let expected = 1.0 - 1.0 / 4.0;
        assert!((s - expected).abs() < 1e-6, "expected {expected}, got {s}");
    }

    #[test]
    fn test_soft_score_all_same_item() {
        // Every interaction is item 5 → familiarity=1.0 → soft_novelty=0.0
        assert!((NoveltyScorer::soft_score(&[5, 5, 5], 5)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soft_score_empty_known() {
        assert_eq!(NoveltyScorer::soft_score(&[], 7), 1.0);
    }

    #[test]
    fn test_score_batch_empty_known() {
        let scores = NoveltyScorer::score_batch(&[], &[1, 2, 3]);
        assert!(scores.iter().all(|&s| (s - 1.0).abs() < f32::EPSILON));
    }

    #[test]
    fn test_score_batch_mixed() {
        let scores = NoveltyScorer::score_batch(&[10, 20], &[10, 30, 20, 40]);
        assert!((scores[0]).abs() < f32::EPSILON, "known item 10 → 0");
        assert!((scores[1] - 1.0).abs() < f32::EPSILON, "new item 30 → 1");
        assert!((scores[2]).abs() < f32::EPSILON, "known item 20 → 0");
        assert!((scores[3] - 1.0).abs() < f32::EPSILON, "new item 40 → 1");
    }

    #[test]
    fn test_familiarity_map_proportions_sum_to_one() {
        let map = NoveltyScorer::familiarity_map(&[1, 2, 1, 3]);
        let total: f32 = map.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "proportions should sum to 1, got {total}"
        );
    }

    #[test]
    fn test_familiarity_map_empty() {
        assert!(NoveltyScorer::familiarity_map(&[]).is_empty());
    }

    #[test]
    fn test_familiarity_map_single_item() {
        let map = NoveltyScorer::familiarity_map(&[7, 7, 7]);
        assert!((map[&7] - 1.0).abs() < f32::EPSILON);
    }
}
