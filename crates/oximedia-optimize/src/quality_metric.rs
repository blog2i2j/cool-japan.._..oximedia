//! Quality metric tracking for `oximedia-optimize`.
//!
//! Provides named metric types (PSNR, SSIM, VMAF, …), individual measurements
//! with score normalization, and a tracker that accumulates measurements over
//! time for summary statistics.

#![allow(dead_code)]

/// The type of perceptual/objective quality metric being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Peak Signal-to-Noise Ratio (higher is better, dB scale, ~20–50 dB).
    Psnr,
    /// Structural Similarity Index (higher is better, 0–1).
    Ssim,
    /// Video Multi-Method Assessment Fusion (higher is better, 0–100).
    Vmaf,
    /// Mean Absolute Error (lower is better).
    Mae,
    /// Mean Squared Error (lower is better).
    Mse,
    /// Visual Information Fidelity (higher is better).
    Vif,
    /// Temporal SSIM across adjacent frames (higher is better, 0–1).
    TemporalSsim,
}

impl MetricType {
    /// Returns `true` when a higher score indicates better quality.
    #[must_use]
    pub fn higher_is_better(&self) -> bool {
        match self {
            Self::Psnr | Self::Ssim | Self::Vmaf | Self::Vif | Self::TemporalSsim => true,
            Self::Mae | Self::Mse => false,
        }
    }

    /// Returns the typical maximum score for normalization purposes.
    ///
    /// For unbounded metrics (PSNR, MAE, MSE) a practical ceiling is returned.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn typical_max(&self) -> f64 {
        match self {
            Self::Psnr => 60.0,
            Self::Ssim | Self::TemporalSsim => 1.0,
            Self::Vmaf => 100.0,
            Self::Mae => 255.0,
            Self::Mse => 65025.0, // 255^2
            Self::Vif => 1.0,
        }
    }

    /// Returns a short display name.
    #[must_use]
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Psnr => "PSNR",
            Self::Ssim => "SSIM",
            Self::Vmaf => "VMAF",
            Self::Mae => "MAE",
            Self::Mse => "MSE",
            Self::Vif => "VIF",
            Self::TemporalSsim => "T-SSIM",
        }
    }
}

/// A single quality measurement for one frame or segment.
#[derive(Debug, Clone)]
pub struct QualityMeasurement {
    /// Metric type.
    pub metric: MetricType,
    /// Raw score value.
    pub score: f64,
    /// Optional frame index this measurement applies to.
    pub frame_index: Option<usize>,
}

impl QualityMeasurement {
    /// Creates a new quality measurement.
    #[must_use]
    pub fn new(metric: MetricType, score: f64) -> Self {
        Self {
            metric,
            score,
            frame_index: None,
        }
    }

    /// Creates a quality measurement tagged with a frame index.
    #[must_use]
    pub fn with_frame(metric: MetricType, score: f64, frame_index: usize) -> Self {
        Self {
            metric,
            score,
            frame_index: Some(frame_index),
        }
    }

    /// Returns the score normalized to [0, 1] relative to `MetricType::typical_max`.
    ///
    /// For "lower is better" metrics the complement is returned so that 1.0
    /// always represents perfect quality.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn normalized_score(&self) -> f64 {
        let max = self.metric.typical_max();
        let raw = (self.score / max).clamp(0.0, 1.0);
        if self.metric.higher_is_better() {
            raw
        } else {
            1.0 - raw
        }
    }

    /// Returns `true` when the normalized score meets or exceeds `threshold`.
    #[must_use]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.normalized_score() >= threshold
    }
}

/// Accumulates [`QualityMeasurement`] values and provides summary statistics.
#[derive(Debug, Default)]
pub struct QualityMetricTracker {
    measurements: Vec<QualityMeasurement>,
}

impl QualityMetricTracker {
    /// Creates an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new measurement.
    pub fn record(&mut self, measurement: QualityMeasurement) {
        self.measurements.push(measurement);
    }

    /// Returns a reference to all recorded measurements.
    #[must_use]
    pub fn all(&self) -> &[QualityMeasurement] {
        &self.measurements
    }

    /// Returns the number of recorded measurements.
    #[must_use]
    pub fn count(&self) -> usize {
        self.measurements.len()
    }

    /// Returns the measurement with the highest normalized score, or `None`
    /// if the tracker is empty.
    #[must_use]
    pub fn best(&self) -> Option<&QualityMeasurement> {
        self.measurements.iter().max_by(|a, b| {
            a.normalized_score()
                .partial_cmp(&b.normalized_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the measurement with the lowest normalized score, or `None`
    /// if the tracker is empty.
    #[must_use]
    pub fn worst(&self) -> Option<&QualityMeasurement> {
        self.measurements.iter().min_by(|a, b| {
            a.normalized_score()
                .partial_cmp(&b.normalized_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the arithmetic mean of normalized scores, or `None` if empty.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn avg(&self) -> Option<f64> {
        if self.measurements.is_empty() {
            return None;
        }
        let sum: f64 = self.measurements.iter().map(|m| m.normalized_score()).sum();
        Some(sum / self.measurements.len() as f64)
    }

    /// Returns all measurements for a specific metric type.
    #[must_use]
    pub fn for_metric(&self, metric: MetricType) -> Vec<&QualityMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.metric == metric)
            .collect()
    }

    /// Clears all recorded measurements.
    pub fn reset(&mut self) {
        self.measurements.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_type_higher_is_better_psnr() {
        assert!(MetricType::Psnr.higher_is_better());
    }

    #[test]
    fn test_metric_type_higher_is_better_mse_false() {
        assert!(!MetricType::Mse.higher_is_better());
    }

    #[test]
    fn test_metric_type_typical_max_vmaf() {
        assert_eq!(MetricType::Vmaf.typical_max(), 100.0);
    }

    #[test]
    fn test_metric_type_display_name_psnr() {
        assert_eq!(MetricType::Psnr.display_name(), "PSNR");
    }

    #[test]
    fn test_quality_measurement_normalized_score_psnr_perfect() {
        let m = QualityMeasurement::new(MetricType::Psnr, 60.0);
        assert!((m.normalized_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_quality_measurement_normalized_score_mse_zero_is_perfect() {
        let m = QualityMeasurement::new(MetricType::Mse, 0.0);
        assert!((m.normalized_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_quality_measurement_normalized_score_clamped() {
        let m = QualityMeasurement::new(MetricType::Vmaf, 150.0); // above max
        assert!((m.normalized_score() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_quality_measurement_with_frame_stores_index() {
        let m = QualityMeasurement::with_frame(MetricType::Ssim, 0.95, 42);
        assert_eq!(m.frame_index, Some(42));
    }

    #[test]
    fn test_quality_measurement_meets_threshold_true() {
        let m = QualityMeasurement::new(MetricType::Vmaf, 80.0);
        assert!(m.meets_threshold(0.8));
    }

    #[test]
    fn test_quality_measurement_meets_threshold_false() {
        let m = QualityMeasurement::new(MetricType::Vmaf, 50.0);
        assert!(!m.meets_threshold(0.9));
    }

    #[test]
    fn test_tracker_empty_on_new() {
        let t = QualityMetricTracker::new();
        assert_eq!(t.count(), 0);
        assert!(t.best().is_none());
        assert!(t.worst().is_none());
        assert!(t.avg().is_none());
    }

    #[test]
    fn test_tracker_record_and_count() {
        let mut t = QualityMetricTracker::new();
        t.record(QualityMeasurement::new(MetricType::Psnr, 40.0));
        t.record(QualityMeasurement::new(MetricType::Psnr, 45.0));
        assert_eq!(t.count(), 2);
    }

    #[test]
    fn test_tracker_best_highest_score() {
        let mut t = QualityMetricTracker::new();
        t.record(QualityMeasurement::new(MetricType::Vmaf, 60.0));
        t.record(QualityMeasurement::new(MetricType::Vmaf, 90.0));
        let best = t.best().expect("best value should exist");
        assert!((best.score - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_worst_lowest_score() {
        let mut t = QualityMetricTracker::new();
        t.record(QualityMeasurement::new(MetricType::Ssim, 0.7));
        t.record(QualityMeasurement::new(MetricType::Ssim, 0.95));
        let worst = t.worst().expect("worst value should exist");
        assert!((worst.score - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_avg_correct() {
        let mut t = QualityMetricTracker::new();
        // Both VMAF 60 and 80, each normalized to 0.6 and 0.8
        t.record(QualityMeasurement::new(MetricType::Vmaf, 60.0));
        t.record(QualityMeasurement::new(MetricType::Vmaf, 80.0));
        let avg = t.avg().expect("average should be computable");
        assert!((avg - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_tracker_reset_clears_measurements() {
        let mut t = QualityMetricTracker::new();
        t.record(QualityMeasurement::new(MetricType::Psnr, 40.0));
        t.reset();
        assert!(t.measurements.is_empty());
    }

    fn is_empty(t: &QualityMetricTracker) -> bool {
        t.count() == 0
    }

    // Helper to access is_empty via count
    #[test]
    fn test_tracker_for_metric_filters_correctly() {
        let mut t = QualityMetricTracker::new();
        t.record(QualityMeasurement::new(MetricType::Psnr, 40.0));
        t.record(QualityMeasurement::new(MetricType::Ssim, 0.9));
        let psnr_only = t.for_metric(MetricType::Psnr);
        assert_eq!(psnr_only.len(), 1);
        assert_eq!(psnr_only[0].metric, MetricType::Psnr);
    }
}
