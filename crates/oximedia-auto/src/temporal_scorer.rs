//! Temporal context scoring: score scenes relative to their neighbours.
//!
//! Standard scene scorers assign importance independently to each scene.
//! This module augments those scores with **temporal context** — whether a
//! scene is more or less important than its immediate neighbours and how it
//! fits into the overall interest trajectory of the edit.
//!
//! # Techniques
//!
//! - **Relative contrast**: boost scenes that are significantly more important
//!   than their neighbours; suppress scenes that are buried in uniformly
//!   high-scoring segments.
//! - **Local trend detection**: identify rising, falling or plateau sub-segments
//!   so editors can preserve narrative momentum.
//! - **Narrative-position weighting**: apply a curve (e.g. story-arc) that
//!   favours scenes at narratively important positions (acts, climaxes).
//! - **Outlier suppression**: dampen isolated spikes to avoid choppy edits.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::temporal_scorer::{TemporalScorer, TemporalConfig};
//! use oximedia_auto::scoring::{ScoredScene, ContentType, Sentiment, ImportanceScore};
//! use oximedia_core::Timestamp;
//!
//! let config = TemporalConfig::default();
//! let scorer = TemporalScorer::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::{ContentType, ImportanceScore, ScoredScene, Sentiment};
use oximedia_core::Timestamp;
use std::collections::HashMap;

// ─── Temporal Configuration ───────────────────────────────────────────────────

/// Weights controlling how temporal context influences the final score.
#[derive(Debug, Clone)]
pub struct TemporalWeights {
    /// Weight for the original (base) scene score.
    pub base: f64,
    /// Weight for the relative contrast component.
    pub contrast: f64,
    /// Weight for the local trend component.
    pub trend: f64,
    /// Weight for the narrative-position component.
    pub narrative_position: f64,
}

impl Default for TemporalWeights {
    fn default() -> Self {
        Self {
            base: 0.50,
            contrast: 0.25,
            trend: 0.15,
            narrative_position: 0.10,
        }
    }
}

impl TemporalWeights {
    /// Return the sum of all weights.
    #[must_use]
    pub fn total(&self) -> f64 {
        self.base + self.contrast + self.trend + self.narrative_position
    }

    /// Validate that all weights are non-negative and sum to approximately 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if any weight is negative or the total deviates from
    /// 1.0 by more than 0.01.
    pub fn validate(&self) -> AutoResult<()> {
        if self.base < 0.0
            || self.contrast < 0.0
            || self.trend < 0.0
            || self.narrative_position < 0.0
        {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "temporal weights must be non-negative".to_string(),
            });
        }
        let total = self.total();
        if (total - 1.0).abs() > 0.01 {
            return Err(AutoError::InvalidParameter {
                name: "weights".to_string(),
                value: format!("temporal weights must sum to 1.0 (got {total:.3})"),
            });
        }
        Ok(())
    }
}

/// Narrative position curve applied to scenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrativePositionCurve {
    /// All positions weighted equally.
    Flat,
    /// Three-act structure: rising importance at act boundaries.
    ThreeAct,
    /// Hero's journey: dip after refusal, peak at crisis and return.
    HeroJourney,
    /// Simple climax: weight increases monotonically towards the end.
    LinearClimax,
    /// Reverse climax: weight decreases monotonically (cold-open style).
    ReverseClimax,
}

impl NarrativePositionCurve {
    /// Evaluate the narrative-position weight at normalised position `t` ∈ [0, 1].
    #[must_use]
    pub fn weight_at(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Flat => 0.5,
            Self::ThreeAct => {
                // Act boundaries at ~0.25 and ~0.75; peaks at those points.
                let act1 = gaussian(t, 0.25, 0.08);
                let act2 = gaussian(t, 0.75, 0.08);
                let midpoint = gaussian(t, 0.50, 0.10) * 0.6;
                (act1 + act2 + midpoint).clamp(0.0, 1.0)
            }
            Self::HeroJourney => {
                // Approximate: ordinary world (low), call/refusal, road of trials (mid),
                // innermost cave (dip), ordeal/reward (peak), return (moderate).
                let call = gaussian(t, 0.20, 0.06) * 0.5;
                let ordeal = gaussian(t, 0.65, 0.08);
                let reward = gaussian(t, 0.80, 0.07) * 0.8;
                (call + ordeal + reward).clamp(0.0, 1.0)
            }
            Self::LinearClimax => t,
            Self::ReverseClimax => 1.0 - t,
        }
    }
}

fn gaussian(t: f64, mean: f64, std: f64) -> f64 {
    let exponent = -0.5 * ((t - mean) / std).powi(2);
    exponent.exp()
}

/// Local trend classification for a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalTrend {
    /// Scores are rising in this neighbourhood.
    Rising,
    /// Scores are falling in this neighbourhood.
    Falling,
    /// Scores are approximately flat.
    Plateau,
}

/// Configuration for the temporal scorer.
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Number of neighbours on each side to consider for context.
    pub context_window: usize,
    /// Weights for combining score components.
    pub weights: TemporalWeights,
    /// Narrative position curve.
    pub narrative_curve: NarrativePositionCurve,
    /// Contrast sensitivity: how strongly contrast vs neighbours boosts/damps
    /// score.  Higher values → more aggressive boosting.
    pub contrast_sensitivity: f64,
    /// Trend bonus applied when a scene continues a rising segment [0, 1].
    pub rising_trend_bonus: f64,
    /// Trend penalty applied when a scene is the sole spike in a valley [0, 1].
    pub spike_penalty: f64,
    /// Smoothing window for outlier detection.
    pub outlier_window: usize,
    /// Z-score threshold above which a scene is considered an isolated spike.
    pub spike_z_threshold: f64,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            context_window: 3,
            weights: TemporalWeights::default(),
            narrative_curve: NarrativePositionCurve::ThreeAct,
            contrast_sensitivity: 0.5,
            rising_trend_bonus: 0.1,
            spike_penalty: 0.15,
            outlier_window: 5,
            spike_z_threshold: 2.0,
        }
    }
}

impl TemporalConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if weights are invalid or sensitivity is out of range.
    pub fn validate(&self) -> AutoResult<()> {
        self.weights.validate()?;
        if self.context_window == 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "context_window must be >= 1".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.contrast_sensitivity) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "contrast_sensitivity must be in [0.0, 1.0]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.rising_trend_bonus) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "rising_trend_bonus must be in [0.0, 1.0]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.spike_penalty) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "spike_penalty must be in [0.0, 1.0]".to_string(),
            });
        }
        if self.outlier_window == 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "outlier_window must be >= 1".to_string(),
            });
        }
        if self.spike_z_threshold < 0.0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "spike_z_threshold must be non-negative".to_string(),
            });
        }
        Ok(())
    }
}

// ─── Temporally Scored Scene ──────────────────────────────────────────────────

/// A scene annotated with temporal-context scoring metadata.
#[derive(Debug, Clone)]
pub struct TemporalScoredScene {
    /// The original scored scene.
    pub scene: ScoredScene,
    /// Final temporally-adjusted importance score [0, 1].
    pub temporal_score: ImportanceScore,
    /// The raw contrast component [−1, 1].
    pub contrast_component: f64,
    /// The local trend at this scene.
    pub local_trend: LocalTrend,
    /// The narrative-position weight applied.
    pub narrative_weight: f64,
    /// Whether this scene was classified as an isolated spike.
    pub is_spike: bool,
    /// Metadata (e.g. "context_window", "rank").
    pub metadata: HashMap<String, String>,
}

impl TemporalScoredScene {
    fn new(
        scene: ScoredScene,
        temporal_score: ImportanceScore,
        contrast_component: f64,
        local_trend: LocalTrend,
        narrative_weight: f64,
        is_spike: bool,
    ) -> Self {
        Self {
            scene,
            temporal_score,
            contrast_component,
            local_trend,
            narrative_weight,
            is_spike,
            metadata: HashMap::new(),
        }
    }
}

// ─── Temporal Scorer ─────────────────────────────────────────────────────────

/// Scores scenes using temporal context from surrounding neighbours.
pub struct TemporalScorer {
    config: TemporalConfig,
}

impl TemporalScorer {
    /// Create a new temporal scorer with the given configuration.
    #[must_use]
    pub fn new(config: TemporalConfig) -> Self {
        Self { config }
    }

    /// Create a temporal scorer with default configuration.
    #[must_use]
    pub fn default_scorer() -> Self {
        Self::new(TemporalConfig::default())
    }

    /// Score a sequence of scenes using temporal context.
    ///
    /// The input order must be chronological.  The scorer computes contextual
    /// adjustments for each scene and returns a parallel vector of
    /// `TemporalScoredScene` values.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn score(&self, scenes: &[ScoredScene]) -> AutoResult<Vec<TemporalScoredScene>> {
        self.config.validate()?;
        if scenes.is_empty() {
            return Ok(Vec::new());
        }

        let base_scores: Vec<f64> = scenes.iter().map(|s| s.adjusted_score()).collect();
        let n = scenes.len();

        // Pre-compute local statistics for outlier detection.
        let spike_flags = self.detect_spikes(&base_scores);

        let mut result = Vec::with_capacity(n);

        for (i, scene) in scenes.iter().enumerate() {
            let t = i as f64 / (n - 1).max(1) as f64;
            let base = base_scores[i];

            // Contrast component: compare to local window average.
            let contrast = self.contrast_component(i, &base_scores);

            // Trend.
            let trend = self.local_trend(i, &base_scores);
            let trend_adjustment = match trend {
                LocalTrend::Rising => self.config.rising_trend_bonus,
                LocalTrend::Falling => 0.0,
                LocalTrend::Plateau => 0.0,
            };

            // Narrative position.
            let narrative_w = self.config.narrative_curve.weight_at(t);

            // Spike suppression.
            let spike_adj = if spike_flags[i] {
                -self.config.spike_penalty
            } else {
                0.0
            };

            // Weighted combination.
            let w = &self.config.weights;
            let score = (w.base * base
                + w.contrast * (0.5 + contrast * self.config.contrast_sensitivity)
                + w.trend * trend_adjustment
                + w.narrative_position * narrative_w
                + spike_adj)
                .clamp(0.0, 1.0);

            result.push(TemporalScoredScene::new(
                scene.clone(),
                score,
                contrast,
                trend,
                narrative_w,
                spike_flags[i],
            ));
        }

        Ok(result)
    }

    /// Return only the top-N scenes by temporal score.
    ///
    /// # Errors
    ///
    /// Returns an error if scoring fails.
    pub fn top_n(&self, scenes: &[ScoredScene], n: usize) -> AutoResult<Vec<TemporalScoredScene>> {
        let mut scored = self.score(scenes)?;
        scored.sort_by(|a, b| {
            b.temporal_score
                .partial_cmp(&a.temporal_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(n);
        Ok(scored)
    }

    /// Compute the contrast component for scene `i`.
    ///
    /// Returns a value in [−1, 1]; positive means the scene is above its local
    /// average, negative means below.
    fn contrast_component(&self, i: usize, scores: &[f64]) -> f64 {
        let n = scores.len();
        let lo = i.saturating_sub(self.config.context_window);
        let hi = (i + self.config.context_window + 1).min(n);

        // Compute local window average excluding scene i.
        let window_scores: Vec<f64> = (lo..hi).filter(|j| *j != i).map(|j| scores[j]).collect();

        if window_scores.is_empty() {
            return 0.0;
        }

        let local_avg = window_scores.iter().sum::<f64>() / window_scores.len() as f64;
        let score = scores[i];

        // Normalise to [−1, 1].
        (score - local_avg).clamp(-1.0, 1.0)
    }

    /// Determine the local trend at scene `i`.
    fn local_trend(&self, i: usize, scores: &[f64]) -> LocalTrend {
        let lo = i.saturating_sub(self.config.context_window);
        let hi = (i + self.config.context_window + 1).min(scores.len());

        if hi <= lo + 1 {
            return LocalTrend::Plateau;
        }

        let window = &scores[lo..hi];
        // Compute slope via simple linear regression over the window.
        let n = window.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = window.iter().sum::<f64>() / n;

        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;

        for (j, y) in window.iter().enumerate() {
            let x = j as f64 - x_mean;
            numerator += x * (y - y_mean);
            denominator += x * x;
        }

        if denominator.abs() < 1e-10 {
            return LocalTrend::Plateau;
        }

        let slope = numerator / denominator;
        let threshold = 0.02; // Minimum slope to classify as rising/falling.
        if slope > threshold {
            LocalTrend::Rising
        } else if slope < -threshold {
            LocalTrend::Falling
        } else {
            LocalTrend::Plateau
        }
    }

    /// Detect isolated spikes using a Z-score over a sliding window.
    fn detect_spikes(&self, scores: &[f64]) -> Vec<bool> {
        let n = scores.len();
        let window = self.config.outlier_window;
        let z_threshold = self.config.spike_z_threshold;

        (0..n)
            .map(|i| {
                let lo = i.saturating_sub(window / 2);
                let hi = (i + window / 2 + 1).min(n);
                let neighbours: Vec<f64> =
                    (lo..hi).filter(|j| *j != i).map(|j| scores[j]).collect();
                if neighbours.len() < 2 {
                    return false;
                }
                let mean = neighbours.iter().sum::<f64>() / neighbours.len() as f64;
                let var = neighbours.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                    / neighbours.len() as f64;
                let std = var.sqrt();
                // When neighbours are near-uniform (std ≈ 0), detect spike by
                // absolute deviation: if scene score exceeds mean by more than
                // half the z-threshold * 0.1 (≈ 20% for default z=2), it is a
                // spike.
                if std < 1e-10 {
                    let abs_deviation = (scores[i] - mean).abs();
                    return abs_deviation > z_threshold * 0.1;
                }
                let z = (scores[i] - mean) / std;
                z > z_threshold
            })
            .collect()
    }
}

// ─── Segment Analysis ─────────────────────────────────────────────────────────

/// Contiguous segment sharing the same local trend.
#[derive(Debug, Clone)]
pub struct TrendSegment {
    /// Start index into the original scene list.
    pub start_idx: usize,
    /// End index (inclusive) into the original scene list.
    pub end_idx: usize,
    /// The trend characterising this segment.
    pub trend: LocalTrend,
    /// Mean temporal score over the segment.
    pub mean_score: f64,
}

impl TrendSegment {
    /// Number of scenes in this segment.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end_idx - self.start_idx + 1
    }

    /// Whether the segment is a single scene.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start_idx == self.end_idx
    }
}

/// Find contiguous trend segments in a sequence of temporally-scored scenes.
#[must_use]
pub fn find_trend_segments(scenes: &[TemporalScoredScene]) -> Vec<TrendSegment> {
    if scenes.is_empty() {
        return Vec::new();
    }

    let mut segments: Vec<TrendSegment> = Vec::new();
    let mut seg_start = 0;
    let mut current_trend = scenes[0].local_trend;
    let mut score_acc = scenes[0].temporal_score;

    for (i, ts) in scenes.iter().enumerate().skip(1) {
        if ts.local_trend != current_trend {
            let count = i - seg_start;
            segments.push(TrendSegment {
                start_idx: seg_start,
                end_idx: i - 1,
                trend: current_trend,
                mean_score: score_acc / count as f64,
            });
            seg_start = i;
            current_trend = ts.local_trend;
            score_acc = ts.temporal_score;
        } else {
            score_acc += ts.temporal_score;
        }
    }

    // Final segment.
    let count = scenes.len() - seg_start;
    segments.push(TrendSegment {
        start_idx: seg_start,
        end_idx: scenes.len() - 1,
        trend: current_trend,
        mean_score: score_acc / count as f64,
    });

    segments
}

// ─── Helper constructors ──────────────────────────────────────────────────────

/// Build a simple `ScoredScene` with the given score for testing purposes.
#[must_use]
pub fn make_scene(start_ms: i64, end_ms: i64, score: f64) -> ScoredScene {
    ScoredScene::new(
        Timestamp::new(start_ms, oximedia_core::Rational::new(1, 1000)),
        Timestamp::new(end_ms, oximedia_core::Rational::new(1, 1000)),
        score,
        ContentType::Unknown,
        Sentiment::Neutral,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_scenes(n: usize, score: f64) -> Vec<ScoredScene> {
        (0..n)
            .map(|i| make_scene((i as i64) * 1000, (i as i64 + 1) * 1000, score))
            .collect()
    }

    #[test]
    fn test_default_config_validates() {
        let config = TemporalConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_weights_must_sum_to_one() {
        let mut weights = TemporalWeights::default();
        weights.base = 0.99;
        assert!(weights.validate().is_err());
    }

    #[test]
    fn test_empty_scenes_returns_empty() {
        let scorer = TemporalScorer::default_scorer();
        let result = scorer.score(&[]).expect("should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_uniform_scores_no_spikes() {
        let scenes = uniform_scenes(10, 0.5);
        let scorer = TemporalScorer::default_scorer();
        let result = scorer.score(&scenes).expect("should succeed");
        for ts in &result {
            assert!(!ts.is_spike, "uniform sequence should have no spikes");
        }
    }

    #[test]
    fn test_spike_detection() {
        let mut scenes = uniform_scenes(10, 0.2);
        // Insert a spike at index 5.
        scenes[5] = make_scene(5000, 6000, 0.95);
        let scorer = TemporalScorer::default_scorer();
        let result = scorer.score(&scenes).expect("should succeed");
        // At least the spike scene should be flagged (or nearby scenes).
        let spike_count = result.iter().filter(|ts| ts.is_spike).count();
        assert!(spike_count >= 1, "spike should be detected");
    }

    #[test]
    fn test_temporal_scores_in_range() {
        let scenes: Vec<ScoredScene> = (0..15)
            .map(|i| {
                let score = (i as f64 / 14.0).clamp(0.0, 1.0);
                make_scene((i as i64) * 1000, (i as i64 + 1) * 1000, score)
            })
            .collect();
        let scorer = TemporalScorer::default_scorer();
        let result = scorer.score(&scenes).expect("should succeed");
        for ts in &result {
            assert!(
                (0.0..=1.0).contains(&ts.temporal_score),
                "temporal_score {} out of range",
                ts.temporal_score
            );
        }
    }

    #[test]
    fn test_rising_sequence_classified_as_rising() {
        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| {
                let score = i as f64 / 9.0;
                make_scene((i as i64) * 1000, (i as i64 + 1) * 1000, score)
            })
            .collect();
        let scorer = TemporalScorer::default_scorer();
        let result = scorer.score(&scenes).expect("should succeed");
        // Interior scenes (not the very first few or last few) should be Rising.
        let mid = &result[5];
        assert_eq!(
            mid.local_trend,
            LocalTrend::Rising,
            "mid scene of rising sequence should be Rising"
        );
    }

    #[test]
    fn test_top_n_returns_correct_count() {
        let scenes = uniform_scenes(20, 0.5);
        let scorer = TemporalScorer::default_scorer();
        let top5 = scorer.top_n(&scenes, 5).expect("should succeed");
        assert!(top5.len() <= 5);
    }

    #[test]
    fn test_top_n_sorted_descending() {
        let scenes: Vec<ScoredScene> = (0..10)
            .map(|i| {
                let s = (i as f64) / 9.0;
                make_scene((i as i64) * 1000, (i as i64 + 1) * 1000, s)
            })
            .collect();
        let scorer = TemporalScorer::default_scorer();
        let top5 = scorer.top_n(&scenes, 5).expect("should succeed");
        for pair in top5.windows(2) {
            assert!(
                pair[0].temporal_score >= pair[1].temporal_score,
                "top_n should be sorted descending"
            );
        }
    }

    #[test]
    fn test_narrative_position_three_act() {
        let curve = NarrativePositionCurve::ThreeAct;
        let w_start = curve.weight_at(0.0);
        let w_act1 = curve.weight_at(0.25);
        let w_mid = curve.weight_at(0.50);
        let w_act2 = curve.weight_at(0.75);
        // Act boundaries should have higher weight than start.
        assert!(w_act1 > w_start || w_mid > w_start);
        assert!(w_act2 > w_start);
    }

    #[test]
    fn test_find_trend_segments_count() {
        let scenes: Vec<ScoredScene> = (0..8)
            .map(|i| {
                let s = if i < 4 {
                    i as f64 / 3.0
                } else {
                    (7 - i) as f64 / 3.0
                };
                make_scene((i as i64) * 1000, (i as i64 + 1) * 1000, s)
            })
            .collect();
        let scorer = TemporalScorer::default_scorer();
        let temporal = scorer.score(&scenes).expect("should succeed");
        let segments = find_trend_segments(&temporal);
        assert!(!segments.is_empty(), "should find at least one segment");
    }

    #[test]
    fn test_contrast_component_above_average_is_positive() {
        let scorer = TemporalScorer::default_scorer();
        let scores = vec![0.1, 0.1, 0.9, 0.1, 0.1];
        let contrast = scorer.contrast_component(2, &scores);
        assert!(contrast > 0.0, "peak scene should have positive contrast");
    }

    #[test]
    fn test_contrast_component_below_average_is_negative() {
        let scorer = TemporalScorer::default_scorer();
        let scores = vec![0.9, 0.9, 0.1, 0.9, 0.9];
        let contrast = scorer.contrast_component(2, &scores);
        assert!(contrast < 0.0, "valley scene should have negative contrast");
    }

    #[test]
    fn test_invalid_context_window_zero() {
        let mut config = TemporalConfig::default();
        config.context_window = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_contrast_sensitivity() {
        let mut config = TemporalConfig::default();
        config.contrast_sensitivity = 1.5;
        assert!(config.validate().is_err());
    }
}
