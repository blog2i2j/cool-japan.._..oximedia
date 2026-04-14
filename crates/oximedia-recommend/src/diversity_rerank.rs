//! Diversity-aware re-ranking via Maximal Marginal Relevance (MMR).
//!
//! MMR was introduced by Carbonell & Goldstein (1998) to strike a balance between
//! relevance and novelty.  At each step the algorithm greedily selects the candidate
//! that maximises:
//!
//! ```text
//! MMR(d) = λ · relevance(d) − (1 − λ) · max_{s ∈ S} sim(d, s)
//! ```
//!
//! where `S` is the set of already-selected items.
//!
//! The item-to-item similarity used here is a blend of:
//! - Jaccard coefficient over genre sets.
//! - A fixed bonus when two items share the same creator.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration controlling the MMR trade-off.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityConfig {
    /// Trade-off parameter λ ∈ [0, 1].
    ///
    /// `1.0` = pure relevance ordering (no diversity penalty).
    /// `0.0` = maximise diversity only (ignore relevance scores).
    pub lambda: f32,
    /// Additional similarity penalty when two items share a genre (additive to
    /// Jaccard before clamping to [0, 1]).
    pub genre_penalty: f32,
    /// Fixed similarity bonus applied when two items share the same creator ID.
    pub creator_penalty: f32,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            genre_penalty: 0.1,
            creator_penalty: 0.3,
        }
    }
}

/// A candidate item entering the re-ranking pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedItem {
    /// Unique item identifier.
    pub id: String,
    /// Relevance score produced by the upstream ranker (higher = more relevant).
    pub relevance_score: f32,
    /// Genre tags for similarity computation.
    pub genres: Vec<String>,
    /// Creator / channel identifier (used for creator diversity).
    pub creator_id: String,
}

/// Diversity statistics computed over a re-ranked result list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityStats {
    /// Shannon entropy of the genre distribution (nats).  Higher = more diverse.
    pub genre_entropy: f32,
    /// Fraction of distinct creator IDs relative to result length.
    /// Ranges from 1/n (all same creator) to 1.0 (all different).
    pub creator_diversity: f32,
    /// Average relevance score of the re-ranked list divided by the average
    /// relevance of the input list (measures how much relevance was sacrificed).
    pub relevance_retention: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Jaccard similarity between two genre slices.
fn jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&str> = a.iter().map(String::as_str).collect();
    let set_b: HashSet<&str> = b.iter().map(String::as_str).collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Composite similarity between two `RankedItem`s as used in MMR.
///
/// `sim = clamp(jaccard_genres + same_creator_bonus, 0, 1)`
fn item_similarity(a: &RankedItem, b: &RankedItem, config: &DiversityConfig) -> f32 {
    let j = jaccard(&a.genres, &b.genres);
    let creator_bonus = if a.creator_id == b.creator_id {
        config.creator_penalty
    } else {
        0.0
    };
    // genre_penalty scales the raw jaccard contribution.
    let raw = j * (1.0 + config.genre_penalty) + creator_bonus;
    raw.clamp(0.0, 1.0)
}

// ──────────────────────────────────────────────────────────────────────────────
// DiversityReranker
// ──────────────────────────────────────────────────────────────────────────────

/// Re-ranks a list of candidate items using Maximal Marginal Relevance.
#[derive(Debug, Default)]
pub struct DiversityReranker;

impl DiversityReranker {
    /// Create a new `DiversityReranker`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Re-rank `items` according to `config` and return up to `n` results.
    ///
    /// The algorithm is O(n² · |items|) which is acceptable for typical
    /// recommendation list sizes (≤ 1000 items).
    ///
    /// When `n == 0` or `items` is empty the returned list is empty.
    #[must_use]
    pub fn rerank(
        &self,
        items: &[RankedItem],
        config: &DiversityConfig,
        n: usize,
    ) -> Vec<RankedItem> {
        if n == 0 || items.is_empty() {
            return Vec::new();
        }

        // Normalise relevance scores to [0, 1] so they are comparable to similarity.
        let max_rel = items
            .iter()
            .map(|i| i.relevance_score)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_rel = items
            .iter()
            .map(|i| i.relevance_score)
            .fold(f32::INFINITY, f32::min);
        let rel_range = (max_rel - min_rel).max(f32::EPSILON);

        let norm_rel: Vec<f32> = items
            .iter()
            .map(|i| (i.relevance_score - min_rel) / rel_range)
            .collect();

        let limit = n.min(items.len());
        let mut selected: Vec<usize> = Vec::with_capacity(limit);
        let mut remaining: Vec<usize> = (0..items.len()).collect();

        // Seed with the highest relevance item.
        let seed_idx = norm_rel
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        selected.push(seed_idx);
        remaining.retain(|&i| i != seed_idx);

        while selected.len() < limit && !remaining.is_empty() {
            // Cache max similarity of each remaining item to the selected set.
            let mut best_mmr = f32::NEG_INFINITY;
            let mut best_pos = 0usize;

            for (pos, &cand_idx) in remaining.iter().enumerate() {
                let rel_term = config.lambda * norm_rel[cand_idx];

                // Maximum similarity to any already-selected item.
                let max_sim = selected
                    .iter()
                    .map(|&sel_idx| item_similarity(&items[cand_idx], &items[sel_idx], config))
                    .fold(f32::NEG_INFINITY, f32::max)
                    .max(0.0); // at least 0 if no selected items

                let mmr = rel_term - (1.0 - config.lambda) * max_sim;
                if mmr > best_mmr {
                    best_mmr = mmr;
                    best_pos = pos;
                }
            }

            let chosen = remaining.remove(best_pos);
            selected.push(chosen);
        }

        selected.iter().map(|&i| items[i].clone()).collect()
    }

    /// Compute diversity statistics for a re-ranked list.
    ///
    /// `original_items` is the unmodified input list and is used to compute
    /// `relevance_retention`.
    ///
    /// Returns `None` if `reranked` is empty.
    #[must_use]
    pub fn compute_stats(
        &self,
        reranked: &[RankedItem],
        original_items: &[RankedItem],
    ) -> Option<DiversityStats> {
        if reranked.is_empty() {
            return None;
        }

        // Genre entropy.
        let mut genre_counts: HashMap<&str, u32> = HashMap::new();
        let mut total_genres = 0u32;
        for item in reranked {
            for g in &item.genres {
                *genre_counts.entry(g.as_str()).or_insert(0) += 1;
                total_genres += 1;
            }
        }
        let genre_entropy = if total_genres == 0 {
            0.0
        } else {
            let n = total_genres as f32;
            genre_counts
                .values()
                .map(|&c| {
                    let p = c as f32 / n;
                    if p > 0.0 {
                        -p * p.ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        // Creator diversity.
        let distinct_creators: HashSet<&str> =
            reranked.iter().map(|i| i.creator_id.as_str()).collect();
        let creator_diversity = distinct_creators.len() as f32 / reranked.len() as f32;

        // Relevance retention.
        let avg_orig = if original_items.is_empty() {
            0.0
        } else {
            original_items
                .iter()
                .map(|i| i.relevance_score)
                .sum::<f32>()
                / original_items.len() as f32
        };
        let avg_reranked =
            reranked.iter().map(|i| i.relevance_score).sum::<f32>() / reranked.len() as f32;
        let relevance_retention = if avg_orig.abs() < f32::EPSILON {
            1.0
        } else {
            (avg_reranked / avg_orig).clamp(0.0, 2.0)
        };

        Some(DiversityStats {
            genre_entropy,
            creator_diversity,
            relevance_retention,
        })
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, rel: f32, genres: &[&str], creator: &str) -> RankedItem {
        RankedItem {
            id: id.to_string(),
            relevance_score: rel,
            genres: genres.iter().map(|s| s.to_string()).collect(),
            creator_id: creator.to_string(),
        }
    }

    fn items_mixed() -> Vec<RankedItem> {
        vec![
            item("a", 1.0, &["action", "thriller"], "c1"),
            item("b", 0.9, &["action", "thriller"], "c1"),
            item("c", 0.8, &["comedy"], "c2"),
            item("d", 0.7, &["drama", "romance"], "c3"),
            item("e", 0.6, &["sci-fi"], "c4"),
            item("f", 0.5, &["comedy", "romance"], "c2"),
        ]
    }

    // 1. lambda=1.0 → sorted by relevance (pure relevance, no diversity).
    #[test]
    fn test_lambda_one_pure_relevance_order() {
        let items = items_mixed();
        let config = DiversityConfig {
            lambda: 1.0,
            ..Default::default()
        };
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 6);
        // Should be in descending relevance order: a, b, c, d, e, f.
        let ids: Vec<&str> = result.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "c", "d", "e", "f"]);
    }

    // 2. lambda=0.0 → maximise diversity, avoid same-creator + same-genre clusters.
    #[test]
    fn test_lambda_zero_maximises_diversity() {
        let items = items_mixed();
        let config = DiversityConfig {
            lambda: 0.0,
            genre_penalty: 0.2,
            creator_penalty: 0.5,
        };
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 4);
        assert_eq!(result.len(), 4);
        // The first item is always the highest relevance seed (item "a").
        assert_eq!(result[0].id, "a");
        // Second item should NOT be "b" (same creator, same genres → high similarity).
        assert_ne!(
            result[1].id, "b",
            "b shares creator+genre with a and should be penalised"
        );
    }

    // 3. Mixed genres selected when lambda=0.5.
    #[test]
    fn test_mixed_genres_selected() {
        let items = items_mixed();
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 4);
        let genre_sets: Vec<&Vec<String>> = result.iter().map(|i| &i.genres).collect();
        // There should be at least 2 distinct top-level genres across results.
        let all_genres: HashSet<&str> = genre_sets
            .iter()
            .flat_map(|g| g.iter().map(String::as_str))
            .collect();
        assert!(
            all_genres.len() >= 2,
            "Expected genre diversity; got genres: {all_genres:?}"
        );
    }

    // 4. n=0 returns empty list.
    #[test]
    fn test_n_zero_returns_empty() {
        let items = items_mixed();
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        assert!(reranker.rerank(&items, &config, 0).is_empty());
    }

    // 5. n > items.len() clamps to items.len().
    #[test]
    fn test_n_exceeds_items_len() {
        let items = items_mixed();
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 100);
        assert_eq!(result.len(), items.len());
    }

    // 6. Empty input returns empty list.
    #[test]
    fn test_empty_items_returns_empty() {
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        assert!(reranker.rerank(&[], &config, 5).is_empty());
    }

    // 7. Diversity stats: creator_diversity is 1.0 when all creators are distinct.
    #[test]
    fn test_stats_all_distinct_creators() {
        let items = vec![
            item("a", 1.0, &["action"], "c1"),
            item("b", 0.8, &["comedy"], "c2"),
            item("c", 0.6, &["drama"], "c3"),
        ];
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 3);
        let stats = reranker
            .compute_stats(&result, &items)
            .expect("stats should be Some");
        assert!(
            (stats.creator_diversity - 1.0).abs() < 0.01,
            "creator_diversity = {}",
            stats.creator_diversity
        );
    }

    // 8. Genre entropy is positive when multiple genres are present.
    #[test]
    fn test_stats_genre_entropy_positive() {
        let items = items_mixed();
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 4);
        let stats = reranker
            .compute_stats(&result, &items)
            .expect("stats should be Some");
        assert!(
            stats.genre_entropy > 0.0,
            "genre_entropy should be positive for diverse results, got {}",
            stats.genre_entropy
        );
    }

    // 9. compute_stats returns None for empty list.
    #[test]
    fn test_stats_empty_returns_none() {
        let reranker = DiversityReranker::new();
        assert!(reranker.compute_stats(&[], &[]).is_none());
    }

    // 10. Single item list works without panics.
    #[test]
    fn test_single_item_list() {
        let items = vec![item("only", 0.9, &["drama"], "c1")];
        let config = DiversityConfig::default();
        let reranker = DiversityReranker::new();
        let result = reranker.rerank(&items, &config, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "only");
    }
}
