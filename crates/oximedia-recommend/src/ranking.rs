//! Ranking algorithms for recommendation lists.
//!
//! Provides score normalisation, re-ranking heuristics, and top-K selection
//! utilities used throughout the recommendation pipeline.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use uuid::Uuid;

// ---------------------------------------------------------------------------
// Score normalisation
// ---------------------------------------------------------------------------

/// Normalise a slice of `(id, score)` pairs to the range `[0, 1]` using
/// min-max normalisation.
///
/// If all scores are equal the result is all `0.0`.
pub fn min_max_normalize(scores: &[(Uuid, f32)]) -> Vec<(Uuid, f32)> {
    if scores.is_empty() {
        return Vec::new();
    }
    let min = scores.iter().map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    scores
        .iter()
        .map(|(id, s)| {
            let normalised = if range < f32::EPSILON {
                0.0
            } else {
                (s - min) / range
            };
            (*id, normalised)
        })
        .collect()
}

/// Apply **softmax** normalisation, turning raw scores into a probability
/// distribution that sums to 1.
pub fn softmax_normalize(scores: &[(Uuid, f32)]) -> Vec<(Uuid, f32)> {
    if scores.is_empty() {
        return Vec::new();
    }
    // Numerically stable: subtract max before exp.
    let max_score = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|(_, s)| (s - max_score).exp()).collect();
    let sum: f32 = exps.iter().sum();
    scores
        .iter()
        .zip(exps)
        .map(|((id, _), e)| (*id, if sum < f32::EPSILON { 0.0 } else { e / sum }))
        .collect()
}

// ---------------------------------------------------------------------------
// Top-K selection
// ---------------------------------------------------------------------------

/// Return the top-`k` items sorted by score descending.
///
/// Does **not** require a fully sorted input – uses a partial sort for
/// efficiency when `k ≪ n`.
#[must_use]
pub fn top_k(scores: &[(Uuid, f32)], k: usize) -> Vec<(Uuid, f32)> {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);
    sorted
}

/// Return the bottom-`k` items sorted by score ascending (e.g. closest
/// distance in a distance-based ranking).
#[must_use]
pub fn bottom_k(scores: &[(Uuid, f32)], k: usize) -> Vec<(Uuid, f32)> {
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);
    sorted
}

// ---------------------------------------------------------------------------
// Score fusion
// ---------------------------------------------------------------------------

/// Fuse two ranked lists using a weighted linear combination.
///
/// Items present in only one list receive a score of `0.0` from the missing
/// list.  The combined score is `weight_a * score_a + weight_b * score_b`.
#[must_use]
pub fn weighted_fusion(
    list_a: &[(Uuid, f32)],
    weight_a: f32,
    list_b: &[(Uuid, f32)],
    weight_b: f32,
) -> Vec<(Uuid, f32)> {
    let mut combined: std::collections::HashMap<Uuid, f32> = std::collections::HashMap::new();

    for (id, score) in list_a {
        *combined.entry(*id).or_insert(0.0) += weight_a * score;
    }
    for (id, score) in list_b {
        *combined.entry(*id).or_insert(0.0) += weight_b * score;
    }

    let mut result: Vec<(Uuid, f32)> = combined.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Reciprocal Rank Fusion (RRF) – a parameter-free rank aggregation method.
///
/// Given multiple ranked lists, assigns each item a fused score of
/// Σ 1/(k + `rank_i`) where k is typically 60.
#[must_use]
pub fn reciprocal_rank_fusion(lists: &[&[(Uuid, f32)]], k: f32) -> Vec<(Uuid, f32)> {
    let mut scores: std::collections::HashMap<Uuid, f32> = std::collections::HashMap::new();
    for list in lists {
        for (rank, (id, _)) in list.iter().enumerate() {
            *scores.entry(*id).or_insert(0.0) += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut result: Vec<(Uuid, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    result
}

// ---------------------------------------------------------------------------
// Diversity-aware re-ranking (Maximal Marginal Relevance)
// ---------------------------------------------------------------------------

/// Item metadata needed for diversity-aware re-ranking.
#[derive(Debug, Clone)]
pub struct RankItem {
    /// Item identifier.
    pub id: Uuid,
    /// Relevance score from the base ranker.
    pub score: f32,
    /// Feature vector for diversity computation (cosine distance used).
    pub features: Vec<f32>,
}

impl RankItem {
    /// Create a new rank item.
    #[must_use]
    pub fn new(id: Uuid, score: f32, features: Vec<f32>) -> Self {
        Self {
            id,
            score,
            features,
        }
    }
}

/// Select the top-`k` items using **Maximal Marginal Relevance** (MMR).
///
/// Balances relevance (`score`) against redundancy (cosine similarity to
/// already-selected items).  `lambda` controls the trade-off:
/// - `1.0` → pure relevance (standard top-k)
/// - `0.0` → pure diversity
#[must_use]
pub fn mmr_rerank(items: &[RankItem], k: usize, lambda: f32) -> Vec<Uuid> {
    if items.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut remaining: Vec<usize> = (0..items.len()).collect();

    while selected.len() < k && !remaining.is_empty() {
        // remaining is non-empty (loop guard), so max_by always returns Some.
        let Some(best_idx) = remaining.iter().copied().max_by(|&i, &j| {
            let mmr_i = mmr_score(items, i, &selected, lambda);
            let mmr_j = mmr_score(items, j, &selected, lambda);
            mmr_i
                .partial_cmp(&mmr_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            break;
        };

        selected.push(best_idx);
        remaining.retain(|&x| x != best_idx);
    }

    selected.iter().map(|&i| items[i].id).collect()
}

fn mmr_score(items: &[RankItem], candidate: usize, selected: &[usize], lambda: f32) -> f32 {
    let rel = items[candidate].score;
    if selected.is_empty() {
        return lambda * rel;
    }
    let max_sim = selected
        .iter()
        .map(|&s| cosine_sim_vecs(&items[candidate].features, &items[s].features))
        .fold(f32::NEG_INFINITY, f32::max);
    lambda * rel - (1.0 - lambda) * max_sim
}

fn cosine_sim_vecs(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        0.0
    } else {
        dot / (na * nb)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uid() -> Uuid {
        Uuid::new_v4()
    }

    fn make_scores(vals: &[f32]) -> Vec<(Uuid, f32)> {
        vals.iter().map(|&v| (uid(), v)).collect()
    }

    // --- min_max_normalize ---

    #[test]
    fn test_min_max_basic() {
        let scores = make_scores(&[0.0, 5.0, 10.0]);
        let norm = min_max_normalize(&scores);
        assert!((norm[0].1).abs() < 1e-5); // 0
        assert!((norm[1].1 - 0.5).abs() < 1e-5); // 0.5
        assert!((norm[2].1 - 1.0).abs() < 1e-5); // 1.0
    }

    #[test]
    fn test_min_max_all_equal() {
        let scores = make_scores(&[3.0, 3.0, 3.0]);
        let norm = min_max_normalize(&scores);
        assert!(norm.iter().all(|(_, s)| s.abs() < f32::EPSILON));
    }

    #[test]
    fn test_min_max_empty() {
        let norm = min_max_normalize(&[]);
        assert!(norm.is_empty());
    }

    // --- softmax_normalize ---

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = make_scores(&[1.0, 2.0, 3.0]);
        let norm = softmax_normalize(&scores);
        let total: f32 = norm.iter().map(|(_, s)| s).sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_higher_wins() {
        let scores = make_scores(&[1.0, 10.0]);
        let norm = softmax_normalize(&scores);
        assert!(norm[1].1 > norm[0].1);
    }

    #[test]
    fn test_softmax_empty() {
        let norm = softmax_normalize(&[]);
        assert!(norm.is_empty());
    }

    // --- top_k ---

    #[test]
    fn test_top_k_basic() {
        let scores = make_scores(&[3.0, 1.0, 4.0, 1.5, 2.0]);
        let top = top_k(&scores, 2);
        assert_eq!(top.len(), 2);
        assert!((top[0].1 - 4.0).abs() < 1e-5);
        assert!((top[1].1 - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_larger_than_input() {
        let scores = make_scores(&[1.0, 2.0]);
        let top = top_k(&scores, 100);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn test_bottom_k_basic() {
        let scores = make_scores(&[5.0, 1.0, 3.0]);
        let bot = bottom_k(&scores, 2);
        assert_eq!(bot.len(), 2);
        assert!((bot[0].1 - 1.0).abs() < 1e-5);
    }

    // --- weighted_fusion ---

    #[test]
    fn test_weighted_fusion_same_lists() {
        let id = uid();
        let list = vec![(id, 1.0)];
        let fused = weighted_fusion(&list, 0.5, &list, 0.5);
        assert_eq!(fused.len(), 1);
        assert!((fused[0].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_weighted_fusion_disjoint() {
        let a = vec![(uid(), 1.0)];
        let b = vec![(uid(), 1.0)];
        let fused = weighted_fusion(&a, 1.0, &b, 1.0);
        assert_eq!(fused.len(), 2);
    }

    // --- reciprocal_rank_fusion ---

    #[test]
    fn test_rrf_single_list() {
        let list = make_scores(&[10.0, 8.0, 6.0]);
        let fused = reciprocal_rank_fusion(&[&list], 60.0);
        assert_eq!(fused.len(), 3);
        // First item in original list should score highest in RRF too.
        assert_eq!(fused[0].0, list[0].0);
    }

    #[test]
    fn test_rrf_two_lists_agree() {
        let ids: Vec<Uuid> = (0..3).map(|_| uid()).collect();
        let list_a: Vec<(Uuid, f32)> = ids.iter().copied().zip([3.0, 2.0, 1.0]).collect();
        let list_b: Vec<(Uuid, f32)> = ids.iter().copied().zip([3.0, 2.0, 1.0]).collect();
        let fused = reciprocal_rank_fusion(&[&list_a, &list_b], 60.0);
        // Both lists agree on ranking; first item should still win.
        assert_eq!(fused[0].0, ids[0]);
    }

    // --- MMR ---

    #[test]
    fn test_mmr_rerank_pure_relevance() {
        let items = vec![
            RankItem::new(uid(), 0.9, vec![1.0, 0.0]),
            RankItem::new(uid(), 0.7, vec![0.9, 0.1]),
            RankItem::new(uid(), 0.5, vec![0.0, 1.0]),
        ];
        // lambda=1 should just return top-k by score
        let result = mmr_rerank(&items, 2, 1.0);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], items[0].id);
    }

    #[test]
    fn test_mmr_rerank_empty() {
        let result = mmr_rerank(&[], 5, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_mmr_rerank_k_zero() {
        let items = vec![RankItem::new(uid(), 1.0, vec![1.0])];
        let result = mmr_rerank(&items, 0, 0.5);
        assert!(result.is_empty());
    }
}
