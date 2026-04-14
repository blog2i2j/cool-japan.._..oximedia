//! Temporal quality stability analysis.
//!
//! [`QualityStabilityAnalyzer`] accumulates per-frame quality scores and
//! quantifies how stable the quality signal is over time.  A stability score
//! of `1.0` means the quality is perfectly constant; lower values indicate
//! higher frame-to-frame variability.
//!
//! # Stability Definition
//!
//! ```text
//! stability = 1.0 - (std_dev / mean)       if mean > 0
//!           = 1.0                           if all values are identical
//!           = 0.0                           if mean ≈ 0 and std_dev > 0
//! ```
//!
//! The result is clamped to `[0.0, 1.0]` so it can be interpreted directly as
//! a probability-like "quality consistency" metric.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// QualityStabilityAnalyzer
// ---------------------------------------------------------------------------

/// Accumulates quality scores and derives a temporal stability metric.
///
/// # Example
///
/// ```
/// use oximedia_quality::temporal_stable::QualityStabilityAnalyzer;
///
/// let mut analyzer = QualityStabilityAnalyzer::new();
/// for _ in 0..10 { analyzer.add(0.85); }
/// assert!((analyzer.stability() - 1.0).abs() < 1e-6); // constant → perfectly stable
/// ```
pub struct QualityStabilityAnalyzer {
    scores: Vec<f64>,
}

impl QualityStabilityAnalyzer {
    /// Creates a new, empty analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self { scores: Vec::new() }
    }

    /// Creates an analyzer pre-seeded with an expected capacity (avoids
    /// reallocation in tight loops).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            scores: Vec::with_capacity(capacity),
        }
    }

    /// Appends a quality score.
    ///
    /// Scores should be in a consistent range (e.g. 0–1 for SSIM or 0–100 for
    /// VMAF); the stability metric is scale-invariant when expressed as
    /// `std_dev / mean` so mixing ranges within a single instance should be
    /// avoided.
    pub fn add(&mut self, score: f64) {
        self.scores.push(score);
    }

    /// Returns the temporal stability in `[0.0, 1.0]`.
    ///
    /// * `1.0` — all accumulated scores are identical (perfect stability).
    /// * `0.0` — maximum variability (coefficient of variation ≥ 1).
    ///
    /// Returns `1.0` when fewer than two scores have been added (degenerate
    /// case: a single sample has zero variance and is trivially stable).
    #[must_use]
    pub fn stability(&self) -> f64 {
        let n = self.scores.len();
        if n < 2 {
            return 1.0;
        }

        let mean = self.mean();
        if mean.abs() < 1e-12 {
            // Mean is effectively zero; if all values are also zero the signal
            // is stable.  Otherwise it is degenerate.
            let std_dev = self.std_dev(mean);
            return if std_dev < 1e-12 { 1.0 } else { 0.0 };
        }

        let std_dev = self.std_dev(mean);
        let cv = std_dev / mean; // coefficient of variation
        (1.0 - cv).clamp(0.0, 1.0)
    }

    /// Returns the arithmetic mean of all accumulated scores, or `0.0` when
    /// the list is empty.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        self.scores.iter().sum::<f64>() / self.scores.len() as f64
    }

    /// Returns the standard deviation (population std dev) of the scores, or
    /// `0.0` when fewer than two scores are available.
    #[must_use]
    pub fn std_dev_value(&self) -> f64 {
        if self.scores.len() < 2 {
            return 0.0;
        }
        self.std_dev(self.mean())
    }

    /// Minimum score, or `0.0` when empty.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.scores
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Maximum score, or `0.0` when empty.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.scores
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Total number of scores accumulated.
    #[must_use]
    pub fn count(&self) -> usize {
        self.scores.len()
    }

    /// Clears all accumulated scores, resetting the analyzer.
    pub fn reset(&mut self) {
        self.scores.clear();
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn std_dev(&self, mean: f64) -> f64 {
        let variance = self
            .scores
            .iter()
            .map(|&s| {
                let d = s - mean;
                d * d
            })
            .sum::<f64>()
            / self.scores.len() as f64;
        variance.sqrt()
    }
}

impl Default for QualityStabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export from `realtime_quality` for convenient access alongside the
/// stability analyser.
pub use crate::realtime_quality::RealtimeQualityGate;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── QualityStabilityAnalyzer ─────────────────────────────────────────────

    #[test]
    fn test_empty_stability_is_one() {
        let a = QualityStabilityAnalyzer::new();
        assert!((a.stability() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_single_score_stability_is_one() {
        let mut a = QualityStabilityAnalyzer::new();
        a.add(0.75);
        assert!((a.stability() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constant_scores_stability_is_one() {
        let mut a = QualityStabilityAnalyzer::new();
        for _ in 0..20 {
            a.add(0.85);
        }
        assert!(
            (a.stability() - 1.0).abs() < 1e-9,
            "constant scores should yield stability 1.0, got {}",
            a.stability()
        );
    }

    #[test]
    fn test_high_variance_lower_stability() {
        let mut a = QualityStabilityAnalyzer::new();
        // Alternating 0.1 and 0.9 → high variance
        for _ in 0..10 {
            a.add(0.1);
            a.add(0.9);
        }
        let s = a.stability();
        assert!(
            s < 0.5,
            "high variance should yield stability < 0.5, got {s}"
        );
    }

    #[test]
    fn test_low_variance_high_stability() {
        let mut a = QualityStabilityAnalyzer::new();
        for i in 0..100 {
            a.add(0.80 + (i as f64 % 3.0) * 0.005); // scores in [0.80, 0.81]
        }
        let s = a.stability();
        assert!(
            s > 0.90,
            "low variance should yield stability > 0.90, got {s}"
        );
    }

    #[test]
    fn test_stability_clamped_to_zero_minimum() {
        let mut a = QualityStabilityAnalyzer::new();
        // Extreme: alternating 0 and 100
        a.add(0.0);
        a.add(100.0);
        let s = a.stability();
        assert!(s >= 0.0 && s <= 1.0, "stability out of range: {s}");
    }

    #[test]
    fn test_mean_is_correct() {
        let mut a = QualityStabilityAnalyzer::new();
        a.add(0.6);
        a.add(0.8);
        a.add(1.0);
        assert!((a.mean() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let mut a = QualityStabilityAnalyzer::new();
        a.add(0.3);
        a.add(0.7);
        a.add(0.5);
        assert!((a.min() - 0.3).abs() < f64::EPSILON);
        assert!((a.max() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reset_clears_scores() {
        let mut a = QualityStabilityAnalyzer::new();
        a.add(0.5);
        a.add(0.6);
        assert_eq!(a.count(), 2);
        a.reset();
        assert_eq!(a.count(), 0);
        assert!((a.stability() - 1.0).abs() < f64::EPSILON);
    }

    // ── RealtimeQualityGate (re-exported from realtime_quality) ────────────

    #[test]
    fn test_gate_passes_above_threshold() {
        let gate = RealtimeQualityGate::new(0.80_f32);
        assert!(gate.check(0.85_f32));
        assert!(gate.check(0.80_f32));
    }

    #[test]
    fn test_gate_rejects_below_threshold() {
        let gate = RealtimeQualityGate::new(0.80_f32);
        assert!(!gate.check(0.79_f32));
        assert!(!gate.check(0.0_f32));
    }

    #[test]
    fn test_gate_boundary_exactly_at_threshold() {
        let gate = RealtimeQualityGate::new(0.75_f32);
        assert!(gate.check(0.75_f32));
    }

    #[test]
    fn test_gate_threshold_accessor() {
        let gate = RealtimeQualityGate::new(0.65_f32);
        assert!((gate.threshold() - 0.65_f32).abs() < f32::EPSILON);
    }
}
