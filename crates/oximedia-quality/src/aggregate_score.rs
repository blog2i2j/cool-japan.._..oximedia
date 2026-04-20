//! Aggregate quality scoring.
//!
//! Combines multiple individual quality metrics into a single composite score
//! with configurable weights, confidence intervals, and rating categories.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single metric contribution to the aggregate score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricContribution {
    /// Metric name (e.g. "PSNR", "SSIM", "VMAF")
    pub name: String,
    /// Raw metric score (in the metric's native units/range)
    pub raw_score: f64,
    /// Normalised score mapped to [0, 100]
    pub normalised_score: f64,
    /// Weight of this metric in the composite (0.0–1.0)
    pub weight: f64,
    /// Confidence in this metric's measurement (0.0–1.0)
    pub confidence: f64,
}

impl MetricContribution {
    /// Creates a new metric contribution.
    ///
    /// `normalised_score` should already be mapped to [0, 100].
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        raw_score: f64,
        normalised_score: f64,
        weight: f64,
    ) -> Self {
        Self {
            name: name.into(),
            raw_score,
            normalised_score: normalised_score.clamp(0.0, 100.0),
            weight: weight.clamp(0.0, 1.0),
            confidence: 1.0,
        }
    }

    /// Sets the confidence level for this contribution.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Returns the weighted contribution to the aggregate.
    #[must_use]
    pub fn weighted_value(&self) -> f64 {
        self.normalised_score * self.weight * self.confidence
    }
}

/// Weighting strategy for the aggregate score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightingStrategy {
    /// All metrics have equal weight (normalised to sum to 1.0)
    Equal,
    /// Weights are provided explicitly via `MetricContribution::weight`
    Explicit,
    /// Weight metrics inversely by their variance (higher variance = lower weight)
    InverseVariance,
}

/// Aggregation method used to combine normalised metric scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Weighted arithmetic mean
    WeightedMean,
    /// Weighted geometric mean
    WeightedGeometricMean,
    /// Weighted harmonic mean
    WeightedHarmonicMean,
    /// Minimum of all normalised scores (pessimistic)
    Minimum,
    /// Percentile of the distribution of normalised scores
    Percentile(u8),
}

/// Quality rating categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityRating {
    /// Excellent – imperceptible degradation
    Excellent,
    /// Good – perceptible but not annoying
    Good,
    /// Fair – slightly annoying
    Fair,
    /// Poor – annoying
    Poor,
    /// Bad – very annoying
    Bad,
}

impl QualityRating {
    /// Maps a score in [0, 100] to a quality rating using standard MOS thresholds.
    #[must_use]
    pub fn from_score(score: f64) -> Self {
        match score as u32 {
            80..=100 => Self::Excellent,
            60..=79 => Self::Good,
            40..=59 => Self::Fair,
            20..=39 => Self::Poor,
            _ => Self::Bad,
        }
    }

    /// Returns the score range for this rating as `(min, max)`.
    #[must_use]
    pub const fn score_range(self) -> (f64, f64) {
        match self {
            Self::Excellent => (80.0, 100.0),
            Self::Good => (60.0, 80.0),
            Self::Fair => (40.0, 60.0),
            Self::Poor => (20.0, 40.0),
            Self::Bad => (0.0, 20.0),
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Excellent => "Excellent",
            Self::Good => "Good",
            Self::Fair => "Fair",
            Self::Poor => "Poor",
            Self::Bad => "Bad",
        }
    }
}

/// A bootstrap confidence interval for the aggregate score.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound of the interval
    pub lower: f64,
    /// Point estimate (aggregate score)
    pub estimate: f64,
    /// Upper bound of the interval
    pub upper: f64,
    /// Confidence level (e.g. 0.95 for 95% CI)
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Creates a new confidence interval.
    #[must_use]
    pub fn new(lower: f64, estimate: f64, upper: f64, confidence_level: f64) -> Self {
        Self {
            lower: lower.clamp(0.0, 100.0),
            estimate: estimate.clamp(0.0, 100.0),
            upper: upper.clamp(0.0, 100.0),
            confidence_level,
        }
    }

    /// Returns the half-width (margin of error) of the interval.
    #[must_use]
    pub fn margin_of_error(&self) -> f64 {
        (self.upper - self.lower) / 2.0
    }

    /// Returns true if the interval is narrow (high confidence).
    #[must_use]
    pub fn is_narrow(&self, threshold: f64) -> bool {
        self.margin_of_error() <= threshold
    }
}

/// The complete aggregate quality score result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateScore {
    /// Final aggregate score in [0, 100]
    pub score: f64,
    /// Quality rating derived from the score
    pub rating: QualityRating,
    /// Confidence interval for the score
    pub confidence_interval: Option<ConfidenceInterval>,
    /// Individual metric contributions used to compute the score
    pub contributions: Vec<MetricContribution>,
    /// Weighting strategy applied
    pub weighting_strategy: WeightingStrategy,
    /// Aggregation method applied
    pub aggregation_method: AggregationMethod,
    /// Any warnings generated during aggregation
    pub warnings: Vec<String>,
}

impl AggregateScore {
    /// Returns the per-metric normalised scores as a map.
    #[must_use]
    pub fn score_map(&self) -> HashMap<String, f64> {
        self.contributions
            .iter()
            .map(|c| (c.name.clone(), c.normalised_score))
            .collect()
    }

    /// Returns true if the score meets the given minimum threshold.
    #[must_use]
    pub fn meets_threshold(&self, min_score: f64) -> bool {
        self.score >= min_score
    }
}

/// Builder for computing an aggregate quality score.
#[derive(Debug, Clone)]
pub struct AggregateScoreBuilder {
    contributions: Vec<MetricContribution>,
    weighting_strategy: WeightingStrategy,
    aggregation_method: AggregationMethod,
    confidence_level: Option<f64>,
}

impl Default for AggregateScoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AggregateScoreBuilder {
    /// Creates a new builder with equal weights and weighted mean aggregation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            contributions: Vec::new(),
            weighting_strategy: WeightingStrategy::Equal,
            aggregation_method: AggregationMethod::WeightedMean,
            confidence_level: None,
        }
    }

    /// Adds a metric contribution.
    #[must_use]
    pub fn add_metric(mut self, contribution: MetricContribution) -> Self {
        self.contributions.push(contribution);
        self
    }

    /// Sets the weighting strategy.
    #[must_use]
    pub fn weighting_strategy(mut self, strategy: WeightingStrategy) -> Self {
        self.weighting_strategy = strategy;
        self
    }

    /// Sets the aggregation method.
    #[must_use]
    pub fn aggregation_method(mut self, method: AggregationMethod) -> Self {
        self.aggregation_method = method;
        self
    }

    /// Sets the desired confidence level for the confidence interval.
    #[must_use]
    pub fn confidence_level(mut self, level: f64) -> Self {
        self.confidence_level = Some(level);
        self
    }

    /// Builds the aggregate score.
    ///
    /// Returns `None` if no contributions have been added.
    #[must_use]
    pub fn build(mut self) -> Option<AggregateScore> {
        if self.contributions.is_empty() {
            return None;
        }

        let mut warnings = Vec::new();

        // Normalise weights if using Equal strategy
        if self.weighting_strategy == WeightingStrategy::Equal {
            let n = self.contributions.len() as f64;
            for c in &mut self.contributions {
                c.weight = 1.0 / n;
            }
        }

        // Validate weight sum
        let weight_sum: f64 = self.contributions.iter().map(|c| c.weight).sum();
        if (weight_sum - 1.0).abs() > 0.05 {
            warnings.push(format!(
                "Weights sum to {weight_sum:.3}, expected 1.0; results may be inaccurate"
            ));
        }

        let score = match self.aggregation_method {
            AggregationMethod::WeightedMean => {
                let sum: f64 = self
                    .contributions
                    .iter()
                    .map(|c| c.normalised_score * c.weight * c.confidence)
                    .sum();
                let weight_sum_eff: f64 = self
                    .contributions
                    .iter()
                    .map(|c| c.weight * c.confidence)
                    .sum();
                if weight_sum_eff <= 0.0 {
                    0.0
                } else {
                    sum / weight_sum_eff
                }
            }
            AggregationMethod::WeightedGeometricMean => {
                let log_sum: f64 = self
                    .contributions
                    .iter()
                    .filter(|c| c.normalised_score > 0.0)
                    .map(|c| c.weight * c.normalised_score.ln())
                    .sum();
                let w_sum: f64 = self
                    .contributions
                    .iter()
                    .filter(|c| c.normalised_score > 0.0)
                    .map(|c| c.weight)
                    .sum();
                if w_sum > 0.0 {
                    (log_sum / w_sum).exp()
                } else {
                    0.0
                }
            }
            AggregationMethod::WeightedHarmonicMean => {
                let inv_sum: f64 = self
                    .contributions
                    .iter()
                    .filter(|c| c.normalised_score > 0.0)
                    .map(|c| c.weight / c.normalised_score)
                    .sum();
                let w_sum: f64 = self
                    .contributions
                    .iter()
                    .filter(|c| c.normalised_score > 0.0)
                    .map(|c| c.weight)
                    .sum();
                if inv_sum > 0.0 {
                    w_sum / inv_sum
                } else {
                    0.0
                }
            }
            AggregationMethod::Minimum => self
                .contributions
                .iter()
                .map(|c| c.normalised_score)
                .fold(f64::INFINITY, f64::min),
            AggregationMethod::Percentile(p) => {
                let mut scores: Vec<f64> = self
                    .contributions
                    .iter()
                    .map(|c| c.normalised_score)
                    .collect();
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((f64::from(p) / 100.0) * scores.len() as f64) as usize;
                scores[idx.min(scores.len() - 1)]
            }
        };

        let score = score.clamp(0.0, 100.0);
        let rating = QualityRating::from_score(score);

        // Compute a simple CI based on standard deviation of contributions
        let confidence_interval = self.confidence_level.map(|level| {
            let scores: Vec<f64> = self
                .contributions
                .iter()
                .map(|c| c.normalised_score)
                .collect();
            let mean = score;
            let variance =
                scores.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();
            // z-score approximation: 1.96 for 95%, 1.645 for 90%
            let z = if level >= 0.95 { 1.96 } else { 1.645 };
            let margin = z * std_dev / (scores.len() as f64).sqrt();
            ConfidenceInterval::new(mean - margin, mean, mean + margin, level)
        });

        Some(AggregateScore {
            score,
            rating,
            confidence_interval,
            contributions: self.contributions,
            weighting_strategy: self.weighting_strategy,
            aggregation_method: self.aggregation_method,
            warnings,
        })
    }
}

/// Normalises a PSNR value (dB) to a [0, 100] score.
///
/// Typical range: < 20 dB = bad, 40+ dB = excellent.
#[must_use]
pub fn normalise_psnr(psnr_db: f64) -> f64 {
    // Linear map: 0 dB → 0, 50 dB → 100
    (psnr_db / 50.0 * 100.0).clamp(0.0, 100.0)
}

/// Normalises an SSIM value [0, 1] to a [0, 100] score.
#[must_use]
pub fn normalise_ssim(ssim: f64) -> f64 {
    (ssim * 100.0).clamp(0.0, 100.0)
}

/// Normalises a VMAF score [0, 100] — already in range.
#[must_use]
pub fn normalise_vmaf(vmaf: f64) -> f64 {
    vmaf.clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_rating_from_score() {
        assert_eq!(QualityRating::from_score(95.0), QualityRating::Excellent);
        assert_eq!(QualityRating::from_score(70.0), QualityRating::Good);
        assert_eq!(QualityRating::from_score(50.0), QualityRating::Fair);
        assert_eq!(QualityRating::from_score(30.0), QualityRating::Poor);
        assert_eq!(QualityRating::from_score(10.0), QualityRating::Bad);
    }

    #[test]
    fn test_quality_rating_score_range() {
        let (lo, hi) = QualityRating::Excellent.score_range();
        assert!(lo < hi);
        assert_eq!(hi, 100.0);
    }

    #[test]
    fn test_quality_rating_label_nonempty() {
        for r in [
            QualityRating::Excellent,
            QualityRating::Good,
            QualityRating::Fair,
            QualityRating::Poor,
            QualityRating::Bad,
        ] {
            assert!(!r.label().is_empty());
        }
    }

    #[test]
    fn test_confidence_interval_margin() {
        let ci = ConfidenceInterval::new(80.0, 85.0, 90.0, 0.95);
        assert!((ci.margin_of_error() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_confidence_interval_is_narrow() {
        let ci = ConfidenceInterval::new(83.0, 85.0, 87.0, 0.95);
        assert!(ci.is_narrow(5.0));
        assert!(!ci.is_narrow(1.0));
    }

    #[test]
    fn test_normalise_psnr() {
        assert!((normalise_psnr(0.0) - 0.0).abs() < 1e-9);
        assert!((normalise_psnr(25.0) - 50.0).abs() < 1e-9);
        assert!((normalise_psnr(50.0) - 100.0).abs() < 1e-9);
        assert_eq!(normalise_psnr(100.0), 100.0); // clamped
    }

    #[test]
    fn test_normalise_ssim() {
        assert!((normalise_ssim(1.0) - 100.0).abs() < 1e-9);
        assert!((normalise_ssim(0.5) - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_normalise_vmaf() {
        assert!((normalise_vmaf(75.0) - 75.0).abs() < 1e-9);
        assert_eq!(normalise_vmaf(110.0), 100.0);
    }

    #[test]
    fn test_metric_contribution_weighted_value() {
        let c = MetricContribution::new("SSIM", 0.95, 95.0, 0.5);
        assert!((c.weighted_value() - 47.5).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_score_builder_equal_weights() {
        let result = AggregateScoreBuilder::new()
            .add_metric(MetricContribution::new("PSNR", 40.0, 80.0, 1.0))
            .add_metric(MetricContribution::new("SSIM", 0.9, 90.0, 1.0))
            .build()
            .expect("should succeed in test");
        // Equal weights → mean of 80 and 90 = 85
        assert!((result.score - 85.0).abs() < 1e-6);
        assert_eq!(result.rating, QualityRating::Excellent);
    }

    #[test]
    fn test_aggregate_score_builder_empty_returns_none() {
        let result = AggregateScoreBuilder::new().build();
        assert!(result.is_none());
    }

    #[test]
    fn test_aggregate_score_builder_minimum() {
        let result = AggregateScoreBuilder::new()
            .aggregation_method(AggregationMethod::Minimum)
            .weighting_strategy(WeightingStrategy::Explicit)
            .add_metric(MetricContribution::new("A", 0.0, 90.0, 0.5))
            .add_metric(MetricContribution::new("B", 0.0, 40.0, 0.5))
            .build()
            .expect("should succeed in test");
        assert!((result.score - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_aggregate_score_meets_threshold() {
        let result = AggregateScoreBuilder::new()
            .add_metric(MetricContribution::new("VMAF", 85.0, 85.0, 1.0))
            .build()
            .expect("should succeed in test");
        assert!(result.meets_threshold(80.0));
        assert!(!result.meets_threshold(90.0));
    }

    #[test]
    fn test_aggregate_score_score_map() {
        let result = AggregateScoreBuilder::new()
            .add_metric(MetricContribution::new("PSNR", 40.0, 80.0, 1.0))
            .add_metric(MetricContribution::new("SSIM", 0.9, 90.0, 1.0))
            .build()
            .expect("should succeed in test");
        let map = result.score_map();
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("PSNR"));
        assert!(map.contains_key("SSIM"));
    }

    #[test]
    fn test_aggregate_score_with_confidence_interval() {
        let result = AggregateScoreBuilder::new()
            .confidence_level(0.95)
            .add_metric(MetricContribution::new("A", 0.0, 80.0, 1.0))
            .add_metric(MetricContribution::new("B", 0.0, 90.0, 1.0))
            .add_metric(MetricContribution::new("C", 0.0, 85.0, 1.0))
            .build()
            .expect("should succeed in test");
        let ci = result.confidence_interval.expect("should succeed in test");
        assert_eq!(ci.confidence_level, 0.95);
        assert!(ci.lower <= ci.estimate);
        assert!(ci.estimate <= ci.upper);
    }

    #[test]
    fn test_aggregate_score_weighted_geometric_mean() {
        let result = AggregateScoreBuilder::new()
            .aggregation_method(AggregationMethod::WeightedGeometricMean)
            .weighting_strategy(WeightingStrategy::Explicit)
            .add_metric(MetricContribution::new("A", 0.0, 100.0, 0.5))
            .add_metric(MetricContribution::new("B", 0.0, 100.0, 0.5))
            .build()
            .expect("should succeed in test");
        assert!((result.score - 100.0).abs() < 1e-6);
    }
}
