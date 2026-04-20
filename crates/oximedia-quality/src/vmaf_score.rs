//! VMAF score tracking and statistical aggregation.
//!
//! Provides [`VmafScore`] for holding per-frame VMAF results, [`VmafModel`]
//! for identifying which model variant produced the scores, and [`VmafTracker`]
//! for accumulating scores across a video sequence and computing statistics.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

/// VMAF model variant used to compute a score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VmafModel {
    /// Default VMAF model (v0.6.1 / `vmaf_v0.6.1.json`).
    Default,
    /// VMAF model tuned for phone-screen content.
    Phone,
    /// VMAF model targeting 4K / UHD display.
    FourK,
    /// VMAF NEG (No Enhancement Gain) variant.
    Neg,
    /// Custom or unknown model.
    Custom,
}

impl VmafModel {
    /// Returns a display name for the model.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Default => "vmaf_v0.6.1",
            Self::Phone => "vmaf_v0.6.1_phone",
            Self::FourK => "vmaf_4k_v0.6.1",
            Self::Neg => "vmaf_neg_v0.6.1",
            Self::Custom => "custom",
        }
    }

    /// Returns the typical score range maximum for this model.
    #[must_use]
    pub fn max_score(&self) -> f64 {
        match self {
            Self::Phone => 110.0,
            _ => 100.0,
        }
    }
}

/// VMAF score for a single video frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VmafScore {
    /// Frame index (0-based).
    pub frame_index: usize,
    /// VMAF composite score.
    pub score: f64,
    /// VMAF model used to compute this score.
    pub model: VmafModel,
    /// Optional ADMA (detail) sub-feature score.
    pub adm2: Option<f64>,
    /// Optional motion sub-feature score.
    pub motion2: Option<f64>,
    /// Optional VIF (Visual Information Fidelity) sub-feature score.
    pub vif_scale: Option<f64>,
}

impl VmafScore {
    /// Creates a new `VmafScore` for the given frame with only the composite
    /// score populated.
    #[must_use]
    pub fn new(frame_index: usize, score: f64, model: VmafModel) -> Self {
        Self {
            frame_index,
            score,
            model,
            adm2: None,
            motion2: None,
            vif_scale: None,
        }
    }

    /// Returns `true` if the score meets the commonly used broadcast threshold
    /// of 93.0 for the default model.
    #[must_use]
    pub fn is_broadcast_quality(&self) -> bool {
        self.score >= 93.0
    }

    /// Clamps the score to the model's valid range [0, `max_score`].
    #[must_use]
    pub fn clamped(&self) -> f64 {
        self.score.clamp(0.0, self.model.max_score())
    }
}

/// Tracks VMAF scores across a video sequence and provides aggregate statistics.
#[derive(Debug, Default)]
pub struct VmafTracker {
    scores: Vec<VmafScore>,
}

impl VmafTracker {
    /// Creates a new, empty `VmafTracker`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a per-frame score to the tracker.
    pub fn push(&mut self, score: VmafScore) {
        self.scores.push(score);
    }

    /// Returns all recorded scores.
    #[must_use]
    pub fn scores(&self) -> &[VmafScore] {
        &self.scores
    }

    /// Returns the number of frames tracked.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.scores.len()
    }

    /// Returns the mean VMAF score across all frames.
    ///
    /// Returns `None` if no scores have been recorded.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }
        let sum: f64 = self.scores.iter().map(|s| s.score).sum();
        Some(sum / self.scores.len() as f64)
    }

    /// Returns the minimum VMAF score across all frames.
    #[must_use]
    pub fn min_score(&self) -> Option<f64> {
        self.scores.iter().map(|s| s.score).reduce(f64::min)
    }

    /// Returns the maximum VMAF score across all frames.
    #[must_use]
    pub fn max_score(&self) -> Option<f64> {
        self.scores.iter().map(|s| s.score).reduce(f64::max)
    }

    /// Returns the p-th percentile VMAF score (0 ≤ p ≤ 100).
    ///
    /// Uses the nearest-rank method.  Returns `None` if no scores are present.
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }
        let p_clamped = p.clamp(0.0, 100.0);
        let mut sorted: Vec<f64> = self.scores.iter().map(|s| s.score).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p_clamped / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        Some(sorted[idx.min(sorted.len() - 1)])
    }

    /// Returns the harmonic mean of VMAF scores (emphasises low-scoring frames).
    ///
    /// Returns `None` if no scores are present or if any score is ≤ 0.
    #[must_use]
    pub fn harmonic_mean(&self) -> Option<f64> {
        if self.scores.is_empty() {
            return None;
        }
        let sum_recip: f64 = self.scores.iter().map(|s| 1.0 / s.score.max(1e-10)).sum();
        Some(self.scores.len() as f64 / sum_recip)
    }

    /// Returns the standard deviation of VMAF scores.
    #[must_use]
    pub fn std_dev(&self) -> Option<f64> {
        let mean = self.mean()?;
        let variance = self
            .scores
            .iter()
            .map(|s| (s.score - mean).powi(2))
            .sum::<f64>()
            / self.scores.len() as f64;
        Some(variance.sqrt())
    }

    /// Returns the frame with the lowest VMAF score.
    #[must_use]
    pub fn worst_frame(&self) -> Option<&VmafScore> {
        self.scores.iter().min_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the fraction of frames that are below `threshold`.
    #[must_use]
    pub fn fraction_below(&self, threshold: f64) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let count = self.scores.iter().filter(|s| s.score < threshold).count();
        count as f64 / self.scores.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tracker(scores: &[f64]) -> VmafTracker {
        let mut t = VmafTracker::new();
        for (i, &s) in scores.iter().enumerate() {
            t.push(VmafScore::new(i, s, VmafModel::Default));
        }
        t
    }

    #[test]
    fn test_empty_tracker() {
        let t = VmafTracker::new();
        assert_eq!(t.frame_count(), 0);
        assert!(t.mean().is_none());
        assert!(t.percentile(50.0).is_none());
    }

    #[test]
    fn test_mean() {
        let t = make_tracker(&[80.0, 90.0, 70.0]);
        assert!((t.mean().expect("should succeed in test") - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_min_max() {
        let t = make_tracker(&[80.0, 95.0, 60.0]);
        assert_eq!(t.min_score().expect("should succeed in test"), 60.0);
        assert_eq!(t.max_score().expect("should succeed in test"), 95.0);
    }

    #[test]
    fn test_percentile_50() {
        let t = make_tracker(&[70.0, 80.0, 90.0]);
        let p50 = t.percentile(50.0).expect("should succeed in test");
        assert_eq!(p50, 80.0);
    }

    #[test]
    fn test_percentile_0_is_min() {
        let t = make_tracker(&[70.0, 80.0, 90.0]);
        assert_eq!(t.percentile(0.0).expect("should succeed in test"), 70.0);
    }

    #[test]
    fn test_percentile_100_is_max() {
        let t = make_tracker(&[70.0, 80.0, 90.0]);
        assert_eq!(t.percentile(100.0).expect("should succeed in test"), 90.0);
    }

    #[test]
    fn test_harmonic_mean_lower_than_arithmetic() {
        let t = make_tracker(&[70.0, 80.0, 90.0]);
        let h = t.harmonic_mean().expect("should succeed in test");
        let a = t.mean().expect("should succeed in test");
        assert!(h <= a);
    }

    #[test]
    fn test_std_dev_uniform() {
        let t = make_tracker(&[80.0, 80.0, 80.0]);
        assert!(t.std_dev().expect("should succeed in test").abs() < 1e-9);
    }

    #[test]
    fn test_worst_frame() {
        let t = make_tracker(&[80.0, 60.0, 90.0]);
        let worst = t.worst_frame().expect("should succeed in test");
        assert_eq!(worst.score, 60.0);
        assert_eq!(worst.frame_index, 1);
    }

    #[test]
    fn test_fraction_below() {
        let t = make_tracker(&[70.0, 80.0, 90.0, 60.0]);
        let f = t.fraction_below(75.0);
        assert!((f - 0.5).abs() < 1e-9); // 70 and 60 are below 75
    }

    #[test]
    fn test_fraction_below_empty() {
        let t = VmafTracker::new();
        assert_eq!(t.fraction_below(80.0), 0.0);
    }

    #[test]
    fn test_vmaf_model_name() {
        assert_eq!(VmafModel::Default.name(), "vmaf_v0.6.1");
        assert_eq!(VmafModel::FourK.name(), "vmaf_4k_v0.6.1");
    }

    #[test]
    fn test_vmaf_model_max_score_phone() {
        assert_eq!(VmafModel::Phone.max_score(), 110.0);
    }

    #[test]
    fn test_vmaf_score_clamped() {
        let s = VmafScore::new(0, 105.0, VmafModel::Default);
        assert_eq!(s.clamped(), 100.0);
    }

    #[test]
    fn test_broadcast_quality() {
        let good = VmafScore::new(0, 94.0, VmafModel::Default);
        let bad = VmafScore::new(1, 85.0, VmafModel::Default);
        assert!(good.is_broadcast_quality());
        assert!(!bad.is_broadcast_quality());
    }

    #[test]
    fn test_frame_count() {
        let t = make_tracker(&[80.0, 90.0]);
        assert_eq!(t.frame_count(), 2);
    }
}
