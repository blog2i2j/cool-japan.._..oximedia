//! Content recommendation scoring based on engagement similarity.
//!
//! This module computes content-to-content similarity scores using engagement
//! feature vectors.  The features characterise *how* viewers interact with a
//! piece of content (watch-time ratio, completion rate, rewatch fraction, etc.)
//! and are derived from the existing [`ContentEngagementScore`] type.
//!
//! ## Similarity metrics
//! Three complementary similarity functions are provided:
//!
//! | Metric | Best for |
//! |--------|----------|
//! | Cosine | Direction-based comparison (ignores magnitude) |
//! | Euclidean | Absolute-distance comparison |
//! | Pearson | Correlation of feature deviations from the mean |
//!
//! ## Recommendation workflow
//! 1. Collect [`ContentEngagementScore`] values for all candidate items.
//! 2. Optionally normalise the feature vectors via [`FeatureNormaliser`].
//! 3. Call [`recommend_similar`] to retrieve the top-N most similar items.
//!
//! [`ContentEngagementScore`]: crate::engagement::ContentEngagementScore

use std::collections::HashMap;

use crate::engagement::{ContentEngagementScore, EngagementComponents};
use crate::error::AnalyticsError;

// ─── Feature vector ───────────────────────────────────────────────────────────

/// Five-dimensional engagement feature vector derived from
/// [`EngagementComponents`].
///
/// Each dimension is nominally in the range `[0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct EngagementFeatures {
    /// Content identifier this feature vector belongs to.
    pub content_id: String,
    /// Ratio of average watch time to content duration (0.0–1.0).
    pub watch_time_ratio: f64,
    /// Fraction of viewers who reached ≥ 95 % completion (0.0–1.0).
    pub completion_rate: f64,
    /// Fraction of viewers who rewatched any segment (0.0–1.0).
    pub rewatch_rate: f64,
    /// Social engagement placeholder (0.0–1.0).
    pub social_score: f64,
    /// Normalised forward-seek penalty (0.0 = no penalty, 1.0 = very high).
    pub seek_penalty: f64,
}

impl EngagementFeatures {
    /// Build features from a fully-computed [`ContentEngagementScore`].
    pub fn from_score(score: &ContentEngagementScore) -> Self {
        let c = &score.components;
        Self {
            content_id: score.content_id.clone(),
            watch_time_ratio: c.watch_time_score as f64,
            completion_rate: c.completion_score as f64,
            rewatch_rate: c.rewatch_score as f64,
            social_score: c.social_score as f64,
            seek_penalty: c.seek_forward_penalty as f64,
        }
    }

    /// Build features from raw component values.
    pub fn from_components(
        content_id: impl Into<String>,
        components: &EngagementComponents,
    ) -> Self {
        Self {
            content_id: content_id.into(),
            watch_time_ratio: components.watch_time_score as f64,
            completion_rate: components.completion_score as f64,
            rewatch_rate: components.rewatch_score as f64,
            social_score: components.social_score as f64,
            seek_penalty: components.seek_forward_penalty as f64,
        }
    }

    /// Return the feature values as a fixed-length array `[watch_time_ratio,
    /// completion_rate, rewatch_rate, social_score, seek_penalty]`.
    pub fn as_array(&self) -> [f64; 5] {
        [
            self.watch_time_ratio,
            self.completion_rate,
            self.rewatch_rate,
            self.social_score,
            self.seek_penalty,
        ]
    }

    /// L2 (Euclidean) norm of the feature vector.
    pub fn l2_norm(&self) -> f64 {
        self.as_array().iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

// ─── Similarity metrics ───────────────────────────────────────────────────────

/// Available similarity / distance metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimilarityMetric {
    /// Cosine similarity ∈ [−1, 1]; higher = more similar.
    #[default]
    Cosine,
    /// Negative Euclidean distance ∈ (−∞, 0]; higher = more similar.
    NegativeEuclidean,
    /// Pearson correlation coefficient ∈ [−1, 1]; higher = more similar.
    Pearson,
}

/// Cosine similarity between two feature arrays: dot(a, b) / (|a| × |b|).
///
/// Returns `0.0` when either vector is the zero vector.
fn cosine_similarity(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < f64::EPSILON || norm_b < f64::EPSILON {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Negative Euclidean distance: −‖a − b‖₂.
///
/// Higher (less negative) values indicate more similar vectors.
fn negative_euclidean(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let dist: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt();
    -dist
}

/// Pearson correlation coefficient between two equal-length arrays.
///
/// Returns `0.0` when either array has zero variance.
fn pearson_correlation(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < f64::EPSILON {
        0.0
    } else {
        cov / denom
    }
}

/// Compute the similarity between two feature vectors using the chosen metric.
pub fn similarity(a: &EngagementFeatures, b: &EngagementFeatures, metric: SimilarityMetric) -> f64 {
    let va = a.as_array();
    let vb = b.as_array();
    match metric {
        SimilarityMetric::Cosine => cosine_similarity(&va, &vb),
        SimilarityMetric::NegativeEuclidean => negative_euclidean(&va, &vb),
        SimilarityMetric::Pearson => pearson_correlation(&va, &vb),
    }
}

// ─── Feature normaliser ───────────────────────────────────────────────────────

/// Min-max normaliser for a collection of engagement feature vectors.
///
/// After fitting on a corpus, call [`FeatureNormaliser::transform`] to map each
/// dimension to `[0.0, 1.0]`.  If a dimension has zero range, all values are
/// mapped to `0.5`.
#[derive(Debug, Clone)]
pub struct FeatureNormaliser {
    min: [f64; 5],
    max: [f64; 5],
}

impl FeatureNormaliser {
    /// Fit the normaliser on a slice of feature vectors.
    ///
    /// Returns an error if `features` is empty.
    pub fn fit(features: &[EngagementFeatures]) -> Result<Self, AnalyticsError> {
        if features.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "cannot fit normaliser on empty feature set".to_string(),
            ));
        }

        let mut min = [f64::MAX; 5];
        let mut max = [f64::MIN; 5];

        for f in features {
            let arr = f.as_array();
            for (i, &v) in arr.iter().enumerate() {
                if v < min[i] {
                    min[i] = v;
                }
                if v > max[i] {
                    max[i] = v;
                }
            }
        }

        Ok(Self { min, max })
    }

    /// Transform a feature vector using the fitted min/max.
    pub fn transform(&self, features: &EngagementFeatures) -> EngagementFeatures {
        let arr = features.as_array();
        let mut normalised = [0.0f64; 5];
        for (i, &v) in arr.iter().enumerate() {
            let range = self.max[i] - self.min[i];
            normalised[i] = if range < f64::EPSILON {
                0.5
            } else {
                (v - self.min[i]) / range
            };
        }
        EngagementFeatures {
            content_id: features.content_id.clone(),
            watch_time_ratio: normalised[0],
            completion_rate: normalised[1],
            rewatch_rate: normalised[2],
            social_score: normalised[3],
            seek_penalty: normalised[4],
        }
    }
}

// ─── Recommendation result ────────────────────────────────────────────────────

/// A single recommendation result: a candidate content item and its similarity
/// to the query item.
#[derive(Debug, Clone, PartialEq)]
pub struct RecommendationResult {
    /// Content identifier of the recommended item.
    pub content_id: String,
    /// Similarity score to the query item (metric-dependent range).
    pub similarity_score: f64,
}

// ─── Recommender ──────────────────────────────────────────────────────────────

/// Content recommender based on engagement similarity.
///
/// Build the recommender once from the full catalogue; then call
/// [`ContentRecommender::recommend`] repeatedly for individual queries.
pub struct ContentRecommender {
    /// Indexed feature vectors: content_id → features.
    index: HashMap<String, EngagementFeatures>,
    metric: SimilarityMetric,
}

impl ContentRecommender {
    /// Create a new recommender backed by the given feature vectors.
    ///
    /// Returns an error if `features` is empty.
    pub fn new(
        features: impl IntoIterator<Item = EngagementFeatures>,
        metric: SimilarityMetric,
    ) -> Result<Self, AnalyticsError> {
        let index: HashMap<String, EngagementFeatures> = features
            .into_iter()
            .map(|f| (f.content_id.clone(), f))
            .collect();

        if index.is_empty() {
            return Err(AnalyticsError::InsufficientData(
                "recommender index is empty".to_string(),
            ));
        }

        Ok(Self { index, metric })
    }

    /// Recommend up to `top_n` items most similar to `query_content_id`.
    ///
    /// The query item itself is excluded from the results.
    ///
    /// # Errors
    /// - [`AnalyticsError::InvalidInput`] — the query content ID is not in the
    ///   index.
    pub fn recommend(
        &self,
        query_content_id: &str,
        top_n: usize,
    ) -> Result<Vec<RecommendationResult>, AnalyticsError> {
        let query = self.index.get(query_content_id).ok_or_else(|| {
            AnalyticsError::InvalidInput(format!(
                "content '{}' not found in recommender index",
                query_content_id
            ))
        })?;

        let mut scored: Vec<RecommendationResult> = self
            .index
            .values()
            .filter(|f| f.content_id != query_content_id)
            .map(|candidate| {
                let score = similarity(query, candidate, self.metric);
                RecommendationResult {
                    content_id: candidate.content_id.clone(),
                    similarity_score: score,
                }
            })
            .collect();

        // Sort by similarity descending (NaN last).
        scored.sort_by(|a, b| {
            b.similarity_score
                .partial_cmp(&a.similarity_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_n);

        Ok(scored)
    }

    /// Return the number of items in the index.
    pub fn index_size(&self) -> usize {
        self.index.len()
    }
}

/// Convenience function: compute top-N content recommendations for a query
/// item directly from [`ContentEngagementScore`] slices without constructing a
/// [`ContentRecommender`] explicitly.
///
/// # Errors
/// See [`ContentRecommender::new`] and [`ContentRecommender::recommend`].
pub fn recommend_similar(
    query_content_id: &str,
    scores: &[ContentEngagementScore],
    top_n: usize,
    metric: SimilarityMetric,
) -> Result<Vec<RecommendationResult>, AnalyticsError> {
    let features: Vec<EngagementFeatures> =
        scores.iter().map(EngagementFeatures::from_score).collect();

    let recommender = ContentRecommender::new(features, metric)?;
    recommender.recommend(query_content_id, top_n)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_score(
        id: &str,
        watch: f32,
        completion: f32,
        rewatch: f32,
        social: f32,
        seek: f32,
    ) -> ContentEngagementScore {
        ContentEngagementScore {
            content_id: id.to_string(),
            score: (watch + completion + rewatch + social - seek) / 4.0,
            components: EngagementComponents {
                watch_time_score: watch,
                completion_score: completion,
                rewatch_score: rewatch,
                social_score: social,
                seek_forward_penalty: seek,
            },
        }
    }

    #[test]
    fn cosine_identical_vectors_returns_one() {
        let a = EngagementFeatures {
            content_id: "a".to_string(),
            watch_time_ratio: 0.8,
            completion_rate: 0.7,
            rewatch_rate: 0.2,
            social_score: 0.5,
            seek_penalty: 0.1,
        };
        let score = similarity(&a, &a, SimilarityMetric::Cosine);
        assert!((score - 1.0).abs() < 1e-9, "cosine of identical={score}");
    }

    #[test]
    fn euclidean_same_point_returns_zero() {
        let a = EngagementFeatures {
            content_id: "a".to_string(),
            watch_time_ratio: 0.5,
            completion_rate: 0.5,
            rewatch_rate: 0.5,
            social_score: 0.5,
            seek_penalty: 0.5,
        };
        let score = similarity(&a, &a, SimilarityMetric::NegativeEuclidean);
        assert!(score.abs() < 1e-9, "negative_euclidean same point={score}");
    }

    #[test]
    fn recommender_excludes_query_item() {
        let scores: Vec<ContentEngagementScore> = vec![
            make_score("a", 0.9, 0.9, 0.3, 0.5, 0.1),
            make_score("b", 0.8, 0.8, 0.2, 0.5, 0.1),
            make_score("c", 0.1, 0.1, 0.0, 0.5, 0.9),
        ];
        let recs = recommend_similar("a", &scores, 5, SimilarityMetric::Cosine)
            .expect("recommendations should succeed");
        assert!(
            recs.iter().all(|r| r.content_id != "a"),
            "query item must not appear in results"
        );
    }

    #[test]
    fn similar_content_ranks_higher_than_dissimilar() {
        let scores = vec![
            make_score("query", 0.9, 0.9, 0.3, 0.5, 0.05),
            make_score("similar", 0.85, 0.88, 0.28, 0.5, 0.06),
            make_score("dissimilar", 0.1, 0.05, 0.0, 0.5, 0.95),
        ];
        let recs = recommend_similar("query", &scores, 2, SimilarityMetric::Cosine).expect("recs");
        assert_eq!(
            recs[0].content_id, "similar",
            "most similar should rank first"
        );
    }

    #[test]
    fn recommender_returns_at_most_top_n() {
        let scores: Vec<_> = (0..10u32)
            .map(|i| make_score(&format!("c{i}"), 0.5 + i as f32 * 0.02, 0.5, 0.1, 0.5, 0.05))
            .collect();
        let recs = recommend_similar("c0", &scores, 3, SimilarityMetric::Cosine).expect("recs");
        assert!(recs.len() <= 3, "should return at most 3 results");
    }

    #[test]
    fn recommender_error_on_unknown_query_id() {
        let scores = vec![make_score("known", 0.5, 0.5, 0.2, 0.5, 0.1)];
        let result = recommend_similar("unknown_id", &scores, 3, SimilarityMetric::Cosine);
        assert!(result.is_err(), "should error on unknown query ID");
    }

    #[test]
    fn feature_normaliser_maps_to_unit_interval() {
        let f1 = EngagementFeatures {
            content_id: "f1".to_string(),
            watch_time_ratio: 0.2,
            completion_rate: 0.3,
            rewatch_rate: 0.1,
            social_score: 0.4,
            seek_penalty: 0.5,
        };
        let f2 = EngagementFeatures {
            content_id: "f2".to_string(),
            watch_time_ratio: 0.8,
            completion_rate: 0.9,
            rewatch_rate: 0.7,
            social_score: 0.6,
            seek_penalty: 0.2,
        };
        let normaliser =
            FeatureNormaliser::fit(&[f1.clone(), f2.clone()]).expect("fit should succeed");
        let n1 = normaliser.transform(&f1);
        let n2 = normaliser.transform(&f2);
        // After min-max normalisation, min → 0.0, max → 1.0.
        assert!((n1.watch_time_ratio).abs() < 1e-9);
        assert!((n2.watch_time_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_identical_vectors_returns_one_or_nan_handled() {
        let a = EngagementFeatures {
            content_id: "a".to_string(),
            watch_time_ratio: 0.7,
            completion_rate: 0.6,
            rewatch_rate: 0.3,
            social_score: 0.5,
            seek_penalty: 0.1,
        };
        let b = EngagementFeatures {
            content_id: "b".to_string(),
            watch_time_ratio: 0.8,
            completion_rate: 0.7,
            rewatch_rate: 0.4,
            social_score: 0.6,
            seek_penalty: 0.2,
        };
        // a and b are linearly scaled versions of each other; Pearson should be
        // close to 1.0.
        let score = similarity(&a, &b, SimilarityMetric::Pearson);
        assert!(score > 0.9, "pearson score={score}");
    }

    #[test]
    fn recommender_index_size_matches_input() {
        let scores: Vec<_> = (0..5u32)
            .map(|i| make_score(&format!("c{i}"), 0.5, 0.5, 0.1, 0.5, 0.1))
            .collect();
        let features: Vec<EngagementFeatures> =
            scores.iter().map(EngagementFeatures::from_score).collect();
        let recommender =
            ContentRecommender::new(features, SimilarityMetric::Cosine).expect("recommender");
        assert_eq!(recommender.index_size(), 5);
    }
}
