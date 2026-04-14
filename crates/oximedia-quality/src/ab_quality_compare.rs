//! A/B quality comparison between two or more encode variants.
//!
//! Provides tooling to rank multiple encodes by perceptual quality,
//! compute quality-per-bit efficiency, and identify the Pareto-optimal
//! frontier (variants that are not dominated on both quality and bitrate).

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── types ────────────────────────────────────────────────────────────────────

/// Composite quality score bundle for one encode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbQualityScore {
    /// Peak Signal-to-Noise Ratio in dB.
    pub psnr_db: f32,
    /// Structural Similarity Index [0, 1].
    pub ssim: f32,
    /// Composite perceptual score in [0, 1], derived from PSNR and SSIM.
    pub composite: f32,
}

impl AbQualityScore {
    /// Creates a new `AbQualityScore`.
    ///
    /// The `composite` field is computed as a weighted average:
    /// `0.4 * psnr_norm + 0.6 * ssim`, where `psnr_norm` is PSNR normalised
    /// to a 0–1 range assuming 20–50 dB covers practical quality.
    #[must_use]
    pub fn new(psnr_db: f32, ssim: f32) -> Self {
        let psnr_norm = ((psnr_db - 20.0) / 30.0).clamp(0.0, 1.0);
        let composite = (0.4 * psnr_norm + 0.6 * ssim).clamp(0.0, 1.0);
        Self {
            psnr_db,
            ssim,
            composite,
        }
    }
}

/// A single encode variant with associated quality metrics and bitrate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeVariant {
    /// Human-readable label (e.g. `"h264-crf18"`, `"av1-q32"`).
    pub label: String,
    /// Target/measured bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Codec identifier string (e.g. `"h264"`, `"av1"`, `"vp9"`).
    pub codec: String,
    /// Aggregate quality metrics for this variant.
    pub metrics: AbQualityScore,
}

impl EncodeVariant {
    /// Creates a new `EncodeVariant`.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        bitrate_kbps: u32,
        codec: impl Into<String>,
        psnr_db: f32,
        ssim: f32,
    ) -> Self {
        Self {
            label: label.into(),
            bitrate_kbps,
            codec: codec.into(),
            metrics: AbQualityScore::new(psnr_db, ssim),
        }
    }

    /// Quality-per-kilobit efficiency ratio.
    ///
    /// Returns the composite quality score divided by bitrate, giving a measure
    /// of how much quality is achieved per unit of bandwidth.
    ///
    /// Returns `0.0` if `bitrate_kbps` is zero.
    #[must_use]
    pub fn efficiency(&self) -> f32 {
        if self.bitrate_kbps == 0 {
            return 0.0;
        }
        self.metrics.composite / self.bitrate_kbps as f32
    }
}

// ─── A/B comparison ───────────────────────────────────────────────────────────

/// Errors that can occur during A/B comparison.
#[derive(Debug, Clone, PartialEq)]
pub enum AbCompareError {
    /// The variant collection is empty.
    NoVariants,
    /// A variant with the given label was not found.
    VariantNotFound(String),
}

impl fmt::Display for AbCompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoVariants => write!(f, "comparison contains no variants"),
            Self::VariantNotFound(label) => write!(f, "variant '{label}' not found"),
        }
    }
}

impl std::error::Error for AbCompareError {}

/// Head-to-head comparison between exactly two encode variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbComparison {
    /// Variant A.
    pub variant_a: EncodeVariant,
    /// Variant B.
    pub variant_b: EncodeVariant,
}

impl AbComparison {
    /// Creates a new A/B comparison.
    #[must_use]
    pub fn new(variant_a: EncodeVariant, variant_b: EncodeVariant) -> Self {
        Self {
            variant_a,
            variant_b,
        }
    }

    /// Returns the label of the variant with higher composite quality.
    ///
    /// If both have identical composite scores, returns variant A's label.
    #[must_use]
    pub fn winner(&self) -> &str {
        if self.variant_b.metrics.composite > self.variant_a.metrics.composite {
            &self.variant_b.label
        } else {
            &self.variant_a.label
        }
    }

    /// Returns the absolute SSIM difference between A and B (always non-negative).
    #[must_use]
    pub fn quality_delta(&self) -> f32 {
        (self.variant_a.metrics.ssim - self.variant_b.metrics.ssim).abs()
    }

    /// Returns the quality-per-bitrate efficiency ratio for variant A.
    ///
    /// Returns `0.0` when bitrate is zero.
    #[must_use]
    pub fn efficiency_a(&self) -> f32 {
        self.variant_a.efficiency()
    }

    /// Returns the quality-per-bitrate efficiency ratio for variant B.
    ///
    /// Returns `0.0` when bitrate is zero.
    #[must_use]
    pub fn efficiency_b(&self) -> f32 {
        self.variant_b.efficiency()
    }

    /// Returns the ratio of variant A efficiency to variant B efficiency.
    ///
    /// Values > 1.0 mean A is more efficient; < 1.0 mean B is more efficient.
    /// Returns `1.0` when both efficiencies are zero.
    #[must_use]
    pub fn bitrate_efficiency_ratio(&self) -> f32 {
        let ea = self.efficiency_a();
        let eb = self.efficiency_b();
        if eb.abs() < 1e-9 {
            if ea.abs() < 1e-9 {
                1.0
            } else {
                f32::INFINITY
            }
        } else {
            ea / eb
        }
    }
}

// ─── multi-variant suite ──────────────────────────────────────────────────────

/// A collection of encode variants for batch comparison.
///
/// Supports finding the overall best variant and computing the
/// Pareto-optimal frontier (variants not dominated by any other on
/// both composite quality *and* bitrate simultaneously).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbTestSuite {
    variants: Vec<EncodeVariant>,
}

impl AbTestSuite {
    /// Creates a new, empty test suite.
    #[must_use]
    pub fn new() -> Self {
        Self {
            variants: Vec::new(),
        }
    }

    /// Adds an encode variant to the suite.
    pub fn add_variant(&mut self, variant: EncodeVariant) {
        self.variants.push(variant);
    }

    /// Returns a slice of all variants.
    #[must_use]
    pub fn variants(&self) -> &[EncodeVariant] {
        &self.variants
    }

    /// Returns the variant with the highest composite quality score.
    ///
    /// When quality is equal, the variant with the lower bitrate is preferred.
    ///
    /// # Errors
    /// Returns [`AbCompareError::NoVariants`] if the suite is empty.
    pub fn best_overall(&self) -> Result<&EncodeVariant, AbCompareError> {
        self.variants
            .iter()
            .max_by(|a, b| {
                a.metrics
                    .composite
                    .partial_cmp(&b.metrics.composite)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        // Prefer lower bitrate when quality is equal.
                        b.bitrate_kbps.cmp(&a.bitrate_kbps)
                    })
            })
            .ok_or(AbCompareError::NoVariants)
    }

    /// Returns the Pareto-optimal frontier.
    ///
    /// A variant is Pareto-optimal if no other variant in the suite achieves
    /// *both* higher (or equal) composite quality *and* lower (or equal)
    /// bitrate simultaneously while being strictly better on at least one.
    ///
    /// # Errors
    /// Returns [`AbCompareError::NoVariants`] if the suite is empty.
    pub fn pareto_frontier(&self) -> Result<Vec<&EncodeVariant>, AbCompareError> {
        if self.variants.is_empty() {
            return Err(AbCompareError::NoVariants);
        }

        let frontier: Vec<&EncodeVariant> = self
            .variants
            .iter()
            .filter(|candidate| {
                // Keep candidate if no other variant dominates it.
                !self.variants.iter().any(|other| {
                    // `other` dominates `candidate` if:
                    //   other.quality >= candidate.quality (weakly better quality)
                    //   other.bitrate <= candidate.bitrate (weakly better bitrate)
                    //   strictly better on at least one dimension.
                    let better_quality =
                        other.metrics.composite > candidate.metrics.composite + f32::EPSILON;
                    let equal_quality = (other.metrics.composite - candidate.metrics.composite)
                        .abs()
                        <= f32::EPSILON;
                    let lower_bitrate = other.bitrate_kbps < candidate.bitrate_kbps;
                    let equal_bitrate = other.bitrate_kbps == candidate.bitrate_kbps;

                    // Domination requires weakly better on both AND strictly better on one.
                    let weakly_better_quality = better_quality || equal_quality;
                    let weakly_better_bitrate = lower_bitrate || equal_bitrate;
                    let strictly_better = better_quality || lower_bitrate;

                    // Avoid self-domination.
                    let is_different = other.label != candidate.label
                        || other.bitrate_kbps != candidate.bitrate_kbps;

                    weakly_better_quality
                        && weakly_better_bitrate
                        && strictly_better
                        && is_different
                })
            })
            .collect();

        Ok(frontier)
    }

    /// Computes a head-to-head comparison between two named variants.
    ///
    /// # Errors
    /// Returns [`AbCompareError::VariantNotFound`] if either label is absent,
    /// or [`AbCompareError::NoVariants`] if the suite is empty.
    pub fn compare(&self, label_a: &str, label_b: &str) -> Result<AbComparison, AbCompareError> {
        if self.variants.is_empty() {
            return Err(AbCompareError::NoVariants);
        }
        let a = self
            .variants
            .iter()
            .find(|v| v.label == label_a)
            .ok_or_else(|| AbCompareError::VariantNotFound(label_a.to_string()))?
            .clone();
        let b = self
            .variants
            .iter()
            .find(|v| v.label == label_b)
            .ok_or_else(|| AbCompareError::VariantNotFound(label_b.to_string()))?
            .clone();
        Ok(AbComparison::new(a, b))
    }
}

impl Default for AbTestSuite {
    fn default() -> Self {
        Self::new()
    }
}

// ─── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_variant(label: &str, kbps: u32, psnr: f32, ssim: f32) -> EncodeVariant {
        EncodeVariant::new(label, kbps, "h264", psnr, ssim)
    }

    // ── AbQualityScore ────────────────────────────────────────────────────────

    #[test]
    fn composite_score_high_quality() {
        let score = AbQualityScore::new(50.0, 0.98);
        // psnr_norm = 1.0, composite = 0.4 * 1.0 + 0.6 * 0.98 = 0.988
        assert!(score.composite > 0.98, "composite: {}", score.composite);
    }

    #[test]
    fn composite_score_low_quality() {
        let score = AbQualityScore::new(20.0, 0.50);
        // psnr_norm = 0.0, composite = 0.6 * 0.50 = 0.30
        assert!(
            (score.composite - 0.30).abs() < 0.01,
            "composite: {}",
            score.composite
        );
    }

    // ── AbComparison ─────────────────────────────────────────────────────────

    #[test]
    fn winner_is_higher_quality() {
        let a = make_variant("low", 1000, 28.0, 0.75);
        let b = make_variant("high", 2000, 40.0, 0.95);
        let cmp = AbComparison::new(a, b);
        assert_eq!(cmp.winner(), "high");
    }

    #[test]
    fn tie_returns_variant_a() {
        let a = make_variant("alpha", 1000, 35.0, 0.90);
        let b = make_variant("beta", 1500, 35.0, 0.90);
        let cmp = AbComparison::new(a, b);
        assert_eq!(cmp.winner(), "alpha");
    }

    #[test]
    fn quality_delta_is_absolute() {
        let a = make_variant("a", 1000, 30.0, 0.80);
        let b = make_variant("b", 2000, 40.0, 0.92);
        let cmp = AbComparison::new(a, b);
        let delta = cmp.quality_delta();
        assert!((delta - 0.12).abs() < 0.001, "delta: {}", delta);
    }

    #[test]
    fn efficiency_ratio_greater_than_one_when_a_better() {
        // A: composite ≈ 0.60, bitrate 500 → efficiency = 0.60/500 = 0.0012
        // B: composite ≈ 0.60, bitrate 2000 → efficiency = 0.60/2000 = 0.0003
        let a = make_variant("efficient", 500, 35.0, 0.90);
        let b = make_variant("wasteful", 2000, 35.0, 0.90);
        let cmp = AbComparison::new(a, b);
        assert!(
            cmp.bitrate_efficiency_ratio() > 1.0,
            "ratio: {}",
            cmp.bitrate_efficiency_ratio()
        );
    }

    // ── AbTestSuite ───────────────────────────────────────────────────────────

    #[test]
    fn best_overall_returns_highest_quality() {
        let mut suite = AbTestSuite::new();
        suite.add_variant(make_variant("low", 500, 28.0, 0.70));
        suite.add_variant(make_variant("mid", 1000, 35.0, 0.85));
        suite.add_variant(make_variant("high", 2000, 45.0, 0.96));

        let best = suite.best_overall().expect("should have best");
        assert_eq!(best.label, "high");
    }

    #[test]
    fn empty_suite_returns_no_variants_error() {
        let suite = AbTestSuite::new();
        assert!(matches!(
            suite.best_overall(),
            Err(AbCompareError::NoVariants)
        ));
        assert!(matches!(
            suite.pareto_frontier(),
            Err(AbCompareError::NoVariants)
        ));
    }

    #[test]
    fn pareto_frontier_excludes_dominated() {
        let mut suite = AbTestSuite::new();
        // "dominated" is strictly worse quality AND higher bitrate than "pareto".
        suite.add_variant(make_variant("pareto", 1000, 40.0, 0.95));
        suite.add_variant(make_variant("dominated", 2000, 35.0, 0.85));
        suite.add_variant(make_variant("efficient", 500, 38.0, 0.92));

        let frontier = suite.pareto_frontier().expect("should return frontier");
        let labels: Vec<&str> = frontier.iter().map(|v| v.label.as_str()).collect();

        // "dominated" should not be on the frontier: "pareto" beats it on both.
        assert!(
            !labels.contains(&"dominated"),
            "dominated should not be on frontier: {labels:?}"
        );
        // "pareto" and "efficient" should be on the frontier.
        assert!(
            labels.contains(&"pareto"),
            "pareto should be on frontier: {labels:?}"
        );
    }

    #[test]
    fn pareto_frontier_single_variant_is_on_frontier() {
        let mut suite = AbTestSuite::new();
        suite.add_variant(make_variant("only", 1000, 40.0, 0.90));
        let frontier = suite.pareto_frontier().expect("should return frontier");
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].label, "only");
    }

    #[test]
    fn compare_named_variants() {
        let mut suite = AbTestSuite::new();
        suite.add_variant(make_variant("a", 500, 30.0, 0.80));
        suite.add_variant(make_variant("b", 1000, 42.0, 0.95));

        let cmp = suite.compare("a", "b").expect("should succeed");
        assert_eq!(cmp.winner(), "b");
        assert!(cmp.quality_delta() > 0.10);
    }

    #[test]
    fn compare_missing_label_returns_error() {
        let mut suite = AbTestSuite::new();
        suite.add_variant(make_variant("a", 500, 30.0, 0.80));

        let result = suite.compare("a", "nonexistent");
        assert!(matches!(
            result,
            Err(AbCompareError::VariantNotFound(ref s)) if s == "nonexistent"
        ));
    }
}
