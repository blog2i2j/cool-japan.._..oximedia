//! User segment-based recommendations.
//!
//! Clusters users into behavioural segments and generates recommendations
//! for cold and warm users alike by leveraging aggregated segment-level
//! preferences rather than only individual interaction histories.
//!
//! # Algorithm
//!
//! 1. **Feature extraction** — each user is described by a feature vector
//!    derived from their interaction statistics (genre distribution, activity
//!    level, recency, session length).
//! 2. **K-means clustering** — users are assigned to the nearest segment
//!    centroid using Euclidean distance; centroids are updated via online
//!    mini-batch updates.
//! 3. **Segment preference aggregation** — for each segment, item popularity
//!    is tracked as an exponentially decaying sum of interaction ratings.
//! 4. **Recommendation** — a user's recommendations are drawn from their
//!    assigned segment's top-K items, blended with any available personal
//!    signal.
//!
//! # Design
//!
//! - No `unsafe`, no external RNG, no `unwrap()`.
//! - LCG-based pseudo-randomness for centroid seeding.
//! - All public types implement `Debug`.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// LCG helper
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

#[inline]
fn lcg_f64(state: u64) -> f64 {
    let s = lcg_next(state);
    (s >> 11) as f64 / (1u64 << 53) as f64
}

// ─────────────────────────────────────────────────────────────────────────────
// User feature vector
// ─────────────────────────────────────────────────────────────────────────────

/// Fixed-dimension feature vector describing a user's behaviour.
///
/// The features are:
/// - \[0\] activity_level  : interactions per day (normalised, 0–1)
/// - \[1\] recency         : days since last interaction (normalised, 0–1, inverted)
/// - \[2\] avg_rating      : average implicit rating (0–1)
/// - \[3\] session_length  : average session length in items (normalised)
/// - [4..4+G] genre_dist : genre affinity distribution (G = `GENRE_DIM`)
pub const FEATURE_DIM: usize = 4 + GENRE_DIM;
/// Number of genre dimensions in the user feature vector.
pub const GENRE_DIM: usize = 8;

/// User feature vector for clustering.
#[derive(Debug, Clone)]
pub struct UserFeatureVector {
    /// Feature values (length = [`FEATURE_DIM`]).
    pub values: [f64; FEATURE_DIM],
}

impl UserFeatureVector {
    /// Create a zero feature vector.
    #[must_use]
    pub fn zeros() -> Self {
        Self {
            values: [0.0; FEATURE_DIM],
        }
    }

    /// Euclidean distance to another feature vector.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// L2-normalise in-place; no-op for zero vectors.
    pub fn normalise(&mut self) {
        let norm: f64 = self.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }
}

impl Default for UserFeatureVector {
    fn default() -> Self {
        Self::zeros()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment
// ─────────────────────────────────────────────────────────────────────────────

/// A single user segment with a centroid and aggregated item preferences.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Segment identifier (0-based index).
    pub id: usize,
    /// Centroid in feature space.
    pub centroid: UserFeatureVector,
    /// Aggregated item scores: item_id → score.
    pub item_scores: HashMap<String, f64>,
    /// Number of users currently assigned to this segment.
    pub user_count: usize,
}

impl Segment {
    /// Create a new segment with the given centroid.
    #[must_use]
    pub fn new(id: usize, centroid: UserFeatureVector) -> Self {
        Self {
            id,
            centroid,
            item_scores: HashMap::new(),
            user_count: 0,
        }
    }

    /// Add an item interaction score with exponential decay weight `alpha`.
    ///
    /// The score is updated as: `score = alpha * old + (1 - alpha) * rating`.
    pub fn update_item_score(&mut self, item_id: &str, rating: f64, alpha: f64) {
        let alpha = alpha.clamp(0.0, 1.0);
        let entry = self.item_scores.entry(item_id.to_string()).or_insert(0.0);
        *entry = alpha * (*entry) + (1.0 - alpha) * rating;
    }

    /// Return top-K items sorted by score descending.
    #[must_use]
    pub fn top_items(&self, k: usize) -> Vec<(String, f64)> {
        let mut items: Vec<(String, f64)> = self
            .item_scores
            .iter()
            .map(|(id, &score)| (id.clone(), score))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        items.truncate(k);
        items
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment cluster model
// ─────────────────────────────────────────────────────────────────────────────

/// K-means based user segment model.
///
/// Maintains `k` segments, assigns users to segments, and aggregates
/// item preferences within each segment.
#[derive(Debug)]
pub struct SegmentModel {
    /// Segments (length = k).
    segments: Vec<Segment>,
    /// User → segment assignment.
    user_assignments: HashMap<String, usize>,
    /// Exponential decay factor for item score updates.
    decay_alpha: f64,
    /// Learning rate for centroid online update.
    centroid_lr: f64,
    /// Total assignment updates performed.
    update_count: u64,
}

impl SegmentModel {
    /// Create a new segment model with `k` segments.
    ///
    /// Centroids are initialised with LCG-derived pseudo-random values.
    #[must_use]
    pub fn new(k: usize, decay_alpha: f64, centroid_lr: f64) -> Self {
        let mut segments = Vec::with_capacity(k);
        let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;
        for i in 0..k {
            let mut centroid = UserFeatureVector::zeros();
            for v in &mut centroid.values {
                state = lcg_next(state);
                *v = lcg_f64(state);
            }
            centroid.normalise();
            segments.push(Segment::new(i, centroid));
        }
        Self {
            segments,
            user_assignments: HashMap::new(),
            decay_alpha: decay_alpha.clamp(0.0, 1.0),
            centroid_lr: centroid_lr.clamp(0.0, 1.0),
            update_count: 0,
        }
    }

    /// Return the number of segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Assign a user to the nearest segment based on their feature vector.
    ///
    /// The centroid of the assigned segment is updated online:
    /// `centroid = (1 - lr) * centroid + lr * user_vector`.
    pub fn assign(&mut self, user_id: impl Into<String>, features: &UserFeatureVector) {
        // Find nearest centroid
        let best = self
            .segments
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.centroid
                    .distance(features)
                    .partial_cmp(&b.centroid.distance(features))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let user_id_str = user_id.into();
        // Decrement old segment count if reassigning
        if let Some(&old) = self.user_assignments.get(&user_id_str) {
            if old != best {
                if let Some(seg) = self.segments.get_mut(old) {
                    seg.user_count = seg.user_count.saturating_sub(1);
                }
            }
        }

        // Online centroid update
        let lr = self.centroid_lr;
        if let Some(seg) = self.segments.get_mut(best) {
            for (c, f) in seg.centroid.values.iter_mut().zip(features.values.iter()) {
                *c = (1.0 - lr) * (*c) + lr * f;
            }
            seg.user_count += 1;
        }

        self.user_assignments.insert(user_id_str, best);
        self.update_count += 1;
    }

    /// Record a user-item interaction in the user's assigned segment.
    pub fn record_interaction(&mut self, user_id: &str, item_id: &str, rating: f64) {
        let Some(&seg_idx) = self.user_assignments.get(user_id) else {
            return;
        };
        let alpha = self.decay_alpha;
        if let Some(seg) = self.segments.get_mut(seg_idx) {
            seg.update_item_score(item_id, rating, alpha);
        }
    }

    /// Get the segment assigned to a user.
    ///
    /// Returns `None` if the user has never been assigned.
    #[must_use]
    pub fn user_segment(&self, user_id: &str) -> Option<usize> {
        self.user_assignments.get(user_id).copied()
    }

    /// Get recommendations for a user from their segment's top items.
    ///
    /// If the user is unassigned, returns recommendations from the largest segment.
    #[must_use]
    pub fn recommend(&self, user_id: &str, limit: usize) -> Vec<SegmentRecommendation> {
        let seg_idx = self
            .user_assignments
            .get(user_id)
            .copied()
            .unwrap_or_else(|| {
                // Use the largest segment as fallback
                self.segments
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, s)| s.user_count)
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        let Some(seg) = self.segments.get(seg_idx) else {
            return Vec::new();
        };

        seg.top_items(limit)
            .into_iter()
            .enumerate()
            .map(|(rank, (item_id, score))| SegmentRecommendation {
                item_id,
                score,
                segment_id: seg_idx,
                rank,
            })
            .collect()
    }

    /// Return total update count.
    #[must_use]
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Return a reference to a specific segment.
    #[must_use]
    pub fn segment(&self, idx: usize) -> Option<&Segment> {
        self.segments.get(idx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recommendation output
// ─────────────────────────────────────────────────────────────────────────────

/// A segment-based recommendation result.
#[derive(Debug, Clone)]
pub struct SegmentRecommendation {
    /// Recommended item identifier.
    pub item_id: String,
    /// Aggregated segment score for this item.
    pub score: f64,
    /// Segment that produced this recommendation.
    pub segment_id: usize,
    /// Rank in the recommendation list (0-based).
    pub rank: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Blended recommender (personal signal + segment signal)
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the blended segment recommender.
#[derive(Debug, Clone)]
pub struct SegmentRecommenderConfig {
    /// Weight for segment-level signal (0–1).
    pub segment_weight: f64,
    /// Weight for personal signal (0–1); should sum to ≤ 1 with segment_weight.
    pub personal_weight: f64,
    /// Exponential decay alpha for segment item scores.
    pub decay_alpha: f64,
    /// Centroid learning rate.
    pub centroid_lr: f64,
    /// Number of segments (k).
    pub k: usize,
}

impl Default for SegmentRecommenderConfig {
    fn default() -> Self {
        Self {
            segment_weight: 0.6,
            personal_weight: 0.4,
            decay_alpha: 0.9,
            centroid_lr: 0.05,
            k: 5,
        }
    }
}

/// High-level blended segment recommender.
///
/// Combines segment-level aggregated scores with per-user personal scores.
#[derive(Debug)]
pub struct SegmentRecommender {
    /// Internal segment model.
    model: SegmentModel,
    /// Per-user personal item scores.
    personal_scores: HashMap<String, HashMap<String, f64>>,
    /// Configuration.
    config: SegmentRecommenderConfig,
}

impl SegmentRecommender {
    /// Create a new recommender.
    #[must_use]
    pub fn new(config: SegmentRecommenderConfig) -> Self {
        let model = SegmentModel::new(config.k, config.decay_alpha, config.centroid_lr);
        Self {
            model,
            personal_scores: HashMap::new(),
            config,
        }
    }

    /// Register or update a user's feature vector, assigning them to a segment.
    pub fn update_user_features(&mut self, user_id: &str, features: &UserFeatureVector) {
        self.model.assign(user_id, features);
    }

    /// Record a user-item interaction (updates both personal and segment scores).
    pub fn record_interaction(&mut self, user_id: &str, item_id: &str, rating: f64) {
        // Personal score: simple EMA
        let alpha = self.config.decay_alpha;
        let personal = self.personal_scores.entry(user_id.to_string()).or_default();
        let entry = personal.entry(item_id.to_string()).or_insert(0.0);
        *entry = alpha * (*entry) + (1.0 - alpha) * rating;

        // Segment score
        self.model.record_interaction(user_id, item_id, rating);
    }

    /// Generate blended recommendations for a user.
    ///
    /// Returns up to `limit` items sorted by blended score descending.
    #[must_use]
    pub fn recommend(&self, user_id: &str, limit: usize) -> Vec<BlendedRecommendation> {
        // Collect segment items
        let seg_recs = self.model.recommend(user_id, limit * 2);

        // Collect personal items
        let personal_empty = HashMap::new();
        let personal = self.personal_scores.get(user_id).unwrap_or(&personal_empty);

        // Merge
        let mut scores: HashMap<String, (f64, usize)> = HashMap::new(); // item_id → (blended_score, segment_id)
        let sw = self.config.segment_weight;
        let pw = self.config.personal_weight;

        for rec in &seg_recs {
            let personal_score = personal.get(&rec.item_id).copied().unwrap_or(0.0);
            let blended = sw * rec.score + pw * personal_score;
            scores.insert(rec.item_id.clone(), (blended, rec.segment_id));
        }

        // Also include personal items not in segment top-K
        for (item_id, &pscore) in personal {
            scores
                .entry(item_id.clone())
                .or_insert((pw * pscore, usize::MAX));
        }

        let mut results: Vec<BlendedRecommendation> = scores
            .into_iter()
            .map(|(item_id, (score, segment_id))| BlendedRecommendation {
                item_id,
                score,
                segment_id: if segment_id == usize::MAX {
                    None
                } else {
                    Some(segment_id)
                },
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Return a reference to the underlying segment model.
    #[must_use]
    pub fn model(&self) -> &SegmentModel {
        &self.model
    }
}

/// A blended recommendation combining segment and personal signals.
#[derive(Debug, Clone)]
pub struct BlendedRecommendation {
    /// Item identifier.
    pub item_id: String,
    /// Blended score.
    pub score: f64,
    /// Segment that contributed the segment signal (`None` if personal-only).
    pub segment_id: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(values: &[f64]) -> UserFeatureVector {
        let mut fv = UserFeatureVector::zeros();
        for (i, &v) in values.iter().enumerate().take(FEATURE_DIM) {
            fv.values[i] = v;
        }
        fv
    }

    // ─── UserFeatureVector ──────────────────────────────────────────────────

    #[test]
    fn test_feature_vector_distance_zero_self() {
        let fv = make_features(&[1.0, 0.5, 0.2, 0.8]);
        assert!((fv.distance(&fv)).abs() < 1e-10);
    }

    #[test]
    fn test_feature_vector_normalise() {
        let mut fv = make_features(&[3.0, 4.0]);
        fv.normalise();
        let norm: f64 = fv.values.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10, "norm after normalise = {norm}");
    }

    #[test]
    fn test_feature_vector_normalise_zero_no_panic() {
        let mut fv = UserFeatureVector::zeros();
        fv.normalise(); // should not panic or divide by zero
        assert!(fv.values.iter().all(|&v| v == 0.0));
    }

    // ─── Segment ────────────────────────────────────────────────────────────

    #[test]
    fn test_segment_item_score_update() {
        let mut seg = Segment::new(0, UserFeatureVector::zeros());
        seg.update_item_score("item1", 1.0, 0.0); // alpha=0 → pure replace
        let score = *seg.item_scores.get("item1").expect("item1 should exist");
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segment_top_items_sorted() {
        let mut seg = Segment::new(0, UserFeatureVector::zeros());
        seg.update_item_score("a", 0.3, 0.0);
        seg.update_item_score("b", 0.9, 0.0);
        seg.update_item_score("c", 0.6, 0.0);
        let top = seg.top_items(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].0, "b");
        assert_eq!(top[1].0, "c");
    }

    #[test]
    fn test_segment_top_items_limit() {
        let mut seg = Segment::new(0, UserFeatureVector::zeros());
        for i in 0..10 {
            seg.update_item_score(&format!("item{i}"), i as f64 * 0.1, 0.0);
        }
        let top = seg.top_items(3);
        assert_eq!(top.len(), 3);
    }

    // ─── SegmentModel ───────────────────────────────────────────────────────

    #[test]
    fn test_segment_model_creation() {
        let model = SegmentModel::new(4, 0.9, 0.05);
        assert_eq!(model.segment_count(), 4);
        assert_eq!(model.update_count(), 0);
    }

    #[test]
    fn test_segment_model_assign_user() {
        let mut model = SegmentModel::new(3, 0.9, 0.1);
        let fv = make_features(&[0.8, 0.2, 0.5]);
        model.assign("user1", &fv);
        let seg = model.user_segment("user1");
        assert!(seg.is_some());
        assert!(seg.expect("should have segment") < 3);
        assert_eq!(model.update_count(), 1);
    }

    #[test]
    fn test_segment_model_record_interaction() {
        let mut model = SegmentModel::new(2, 0.5, 0.1);
        let fv = make_features(&[0.5; FEATURE_DIM]);
        model.assign("user1", &fv);
        model.record_interaction("user1", "item_x", 0.8);
        let seg_idx = model.user_segment("user1").expect("should have segment");
        let seg = model.segment(seg_idx).expect("segment should exist");
        assert!(seg.item_scores.contains_key("item_x"));
    }

    #[test]
    fn test_segment_model_recommend_unassigned_user() {
        let mut model = SegmentModel::new(3, 0.9, 0.1);
        // Add one user so largest segment has count > 0
        let fv = make_features(&[1.0; FEATURE_DIM]);
        model.assign("user1", &fv);
        model.record_interaction("user1", "item1", 0.9);
        // Recommend for unknown user — falls back to largest segment
        let recs = model.recommend("unknown_user", 5);
        assert!(!recs.is_empty(), "fallback should return segment items");
    }

    #[test]
    fn test_segment_model_recommend_limit() {
        let mut model = SegmentModel::new(2, 0.9, 0.1);
        let fv = make_features(&[0.5; FEATURE_DIM]);
        model.assign("user1", &fv);
        for i in 0..20 {
            model.record_interaction("user1", &format!("item{i}"), 0.5);
        }
        let recs = model.recommend("user1", 5);
        assert!(recs.len() <= 5);
    }

    // ─── SegmentRecommender ─────────────────────────────────────────────────

    #[test]
    fn test_segment_recommender_default_config() {
        let config = SegmentRecommenderConfig::default();
        assert_eq!(config.k, 5);
        assert!((config.segment_weight - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_segment_recommender_full_flow() {
        let config = SegmentRecommenderConfig {
            k: 3,
            ..Default::default()
        };
        let mut rec = SegmentRecommender::new(config);

        // Register users
        let fv1 = make_features(&[0.9, 0.1, 0.8, 0.3]);
        let fv2 = make_features(&[0.1, 0.9, 0.2, 0.7]);
        rec.update_user_features("u1", &fv1);
        rec.update_user_features("u2", &fv2);

        // Record interactions
        rec.record_interaction("u1", "item_a", 0.9);
        rec.record_interaction("u1", "item_b", 0.7);
        rec.record_interaction("u2", "item_c", 0.8);

        let recs_u1 = rec.recommend("u1", 5);
        assert!(!recs_u1.is_empty(), "u1 should get recommendations");
        // Sorted descending
        for w in recs_u1.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn test_segment_recommender_personal_only_user() {
        // User with personal interactions but never assigned
        let config = SegmentRecommenderConfig {
            k: 2,
            personal_weight: 1.0,
            segment_weight: 0.0,
            ..Default::default()
        };
        let mut rec = SegmentRecommender::new(config);
        rec.record_interaction("unassigned", "item_x", 0.95);
        let recs = rec.recommend("unassigned", 3);
        // Should still include personal item even without segment assignment
        assert!(!recs.is_empty());
        assert_eq!(recs[0].item_id, "item_x");
    }

    #[test]
    fn test_blended_recommendation_segment_id_none_for_personal_only() {
        let config = SegmentRecommenderConfig {
            k: 2,
            segment_weight: 0.0,
            personal_weight: 1.0,
            ..Default::default()
        };
        let mut rec = SegmentRecommender::new(config);
        // No assignment, only personal interaction
        rec.record_interaction("u99", "solo_item", 0.8);
        let recs = rec.recommend("u99", 5);
        let solo = recs.iter().find(|r| r.item_id == "solo_item");
        assert!(solo.is_some());
        assert!(solo.expect("should exist").segment_id.is_none());
    }
}
