#![allow(dead_code)]
//! Regression detection for benchmark results.
//!
//! Compares current benchmark results against historical baselines to detect
//! performance regressions exceeding configurable thresholds.

use std::collections::HashMap;

/// A threshold definition for regression detection.
#[derive(Debug, Clone)]
pub struct RegressionThreshold {
    /// Name of the metric this threshold applies to.
    pub metric: String,
    /// Maximum allowed relative degradation (0.0 = 0%, 0.1 = 10%).
    pub max_degradation: f64,
}

impl RegressionThreshold {
    /// Create a new regression threshold.
    pub fn new(metric: impl Into<String>, max_degradation: f64) -> Self {
        Self {
            metric: metric.into(),
            max_degradation: max_degradation.clamp(0.0, 1.0),
        }
    }

    /// Returns how much the given ratio exceeds this threshold, or `None` if it doesn't.
    ///
    /// `ratio` is `(baseline - current) / baseline` for metrics where higher is better.
    pub fn exceeded_by(&self, ratio: f64) -> Option<f64> {
        if ratio > self.max_degradation {
            Some(ratio - self.max_degradation)
        } else {
            None
        }
    }
}

/// A detected performance regression for a single metric.
#[derive(Debug, Clone)]
pub struct BenchRegression {
    /// Name of the metric that regressed.
    pub metric: String,
    /// Baseline (reference) value.
    pub baseline_value: f64,
    /// Current (measured) value.
    pub current_value: f64,
    /// Relative degradation (0.0–1.0+).
    pub degradation_ratio: f64,
    /// The threshold that was violated.
    pub threshold: f64,
}

impl BenchRegression {
    /// Returns `true` if the degradation ratio exceeds the threshold.
    pub fn is_regression(&self) -> bool {
        self.degradation_ratio > self.threshold
    }

    /// Human-readable description of the regression.
    pub fn description(&self) -> String {
        format!(
            "{}: baseline={:.4}, current={:.4}, degradation={:.1}% (threshold={:.1}%)",
            self.metric,
            self.baseline_value,
            self.current_value,
            self.degradation_ratio * 100.0,
            self.threshold * 100.0,
        )
    }
}

/// Accumulated result record used by the detector.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Metric name.
    pub metric: String,
    /// Measured value (higher = better for fps-type metrics; lower = better for latency).
    pub value: f64,
    /// If `true`, higher values are better (default). If `false`, lower is better.
    pub higher_is_better: bool,
}

impl BenchResult {
    /// Create a result where higher values indicate better performance.
    pub fn higher_is_better(metric: impl Into<String>, value: f64) -> Self {
        Self {
            metric: metric.into(),
            value,
            higher_is_better: true,
        }
    }

    /// Create a result where lower values indicate better performance (e.g. latency).
    pub fn lower_is_better(metric: impl Into<String>, value: f64) -> Self {
        Self {
            metric: metric.into(),
            value,
            higher_is_better: false,
        }
    }
}

/// Detects performance regressions by comparing new results against stored baselines.
#[derive(Debug, Default)]
pub struct RegressionDetector {
    baselines: HashMap<String, BenchResult>,
    thresholds: Vec<RegressionThreshold>,
    history: Vec<BenchResult>,
}

impl RegressionDetector {
    /// Create a new detector with no baselines or thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a threshold for a named metric.
    pub fn add_threshold(&mut self, threshold: RegressionThreshold) {
        self.thresholds.push(threshold);
    }

    /// Record a baseline value for a metric.
    pub fn set_baseline(&mut self, result: BenchResult) {
        self.baselines.insert(result.metric.clone(), result);
    }

    /// Record a new measurement result.
    pub fn add_result(&mut self, result: BenchResult) {
        self.history.push(result);
    }

    /// Detect regressions for all recorded results against baselines.
    pub fn detect_regressions(&self) -> Vec<BenchRegression> {
        let mut regressions = Vec::new();

        for result in &self.history {
            let Some(baseline) = self.baselines.get(&result.metric) else {
                continue;
            };

            // Find applicable threshold.
            let threshold_val = self
                .thresholds
                .iter()
                .find(|t| t.metric == result.metric)
                .map(|t| t.max_degradation)
                .unwrap_or(0.05); // default 5%

            // Calculate degradation ratio.
            let degradation = if result.higher_is_better {
                if baseline.value == 0.0 {
                    0.0
                } else {
                    (baseline.value - result.value) / baseline.value
                }
            } else {
                // lower is better: regression when current > baseline
                if baseline.value == 0.0 {
                    0.0
                } else {
                    (result.value - baseline.value) / baseline.value
                }
            };

            if degradation > threshold_val {
                regressions.push(BenchRegression {
                    metric: result.metric.clone(),
                    baseline_value: baseline.value,
                    current_value: result.value,
                    degradation_ratio: degradation,
                    threshold: threshold_val,
                });
            }
        }

        regressions
    }

    /// Return the single worst regression (highest excess over threshold), if any.
    pub fn worst_regression(&self) -> Option<BenchRegression> {
        self.detect_regressions().into_iter().max_by(|a, b| {
            (a.degradation_ratio - a.threshold)
                .partial_cmp(&(b.degradation_ratio - b.threshold))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Clear all recorded results (but keep baselines and thresholds).
    pub fn clear_results(&mut self) {
        self.history.clear();
    }

    /// Number of stored baselines.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_not_exceeded() {
        let t = RegressionThreshold::new("fps", 0.10);
        assert!(t.exceeded_by(0.05).is_none());
    }

    #[test]
    fn test_threshold_exceeded() {
        let t = RegressionThreshold::new("fps", 0.10);
        let excess = t.exceeded_by(0.20);
        assert!(excess.is_some());
        let excess = excess.expect("excess should be valid");
        assert!((excess - 0.10).abs() < 1e-9);
    }

    #[test]
    fn test_threshold_clamp() {
        let t = RegressionThreshold::new("fps", 2.0);
        assert_eq!(t.max_degradation, 1.0);
        let t2 = RegressionThreshold::new("fps", -0.5);
        assert_eq!(t2.max_degradation, 0.0);
    }

    #[test]
    fn test_bench_regression_is_regression_true() {
        let r = BenchRegression {
            metric: "fps".to_string(),
            baseline_value: 100.0,
            current_value: 70.0,
            degradation_ratio: 0.30,
            threshold: 0.10,
        };
        assert!(r.is_regression());
    }

    #[test]
    fn test_bench_regression_is_regression_false() {
        let r = BenchRegression {
            metric: "fps".to_string(),
            baseline_value: 100.0,
            current_value: 98.0,
            degradation_ratio: 0.02,
            threshold: 0.10,
        };
        assert!(!r.is_regression());
    }

    #[test]
    fn test_bench_regression_description() {
        let r = BenchRegression {
            metric: "encoding_fps".to_string(),
            baseline_value: 60.0,
            current_value: 42.0,
            degradation_ratio: 0.30,
            threshold: 0.05,
        };
        let desc = r.description();
        assert!(desc.contains("encoding_fps"));
        assert!(desc.contains("30.0%"));
    }

    #[test]
    fn test_detector_no_regressions_when_no_baselines() {
        let mut det = RegressionDetector::new();
        det.add_result(BenchResult::higher_is_better("fps", 100.0));
        assert!(det.detect_regressions().is_empty());
    }

    #[test]
    fn test_detector_no_regression_within_threshold() {
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::higher_is_better("fps", 100.0));
        det.add_threshold(RegressionThreshold::new("fps", 0.10));
        det.add_result(BenchResult::higher_is_better("fps", 95.0)); // 5% drop, within 10%
        assert!(det.detect_regressions().is_empty());
    }

    #[test]
    fn test_detector_detects_regression() {
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::higher_is_better("fps", 100.0));
        det.add_threshold(RegressionThreshold::new("fps", 0.10));
        det.add_result(BenchResult::higher_is_better("fps", 80.0)); // 20% drop
        let regs = det.detect_regressions();
        assert_eq!(regs.len(), 1);
        assert_eq!(regs[0].metric, "fps");
    }

    #[test]
    fn test_detector_lower_is_better_regression() {
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::lower_is_better("latency_ms", 10.0));
        det.add_threshold(RegressionThreshold::new("latency_ms", 0.10));
        det.add_result(BenchResult::lower_is_better("latency_ms", 15.0)); // 50% worse
        let regs = det.detect_regressions();
        assert_eq!(regs.len(), 1);
    }

    #[test]
    fn test_detector_worst_regression() {
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::higher_is_better("fps", 100.0));
        det.set_baseline(BenchResult::higher_is_better("throughput", 200.0));
        det.add_threshold(RegressionThreshold::new("fps", 0.05));
        det.add_threshold(RegressionThreshold::new("throughput", 0.05));
        det.add_result(BenchResult::higher_is_better("fps", 80.0)); // 20% drop
        det.add_result(BenchResult::higher_is_better("throughput", 160.0)); // 20% drop
        let worst = det.worst_regression();
        assert!(worst.is_some());
    }

    #[test]
    fn test_detector_worst_regression_none_when_none() {
        let det = RegressionDetector::new();
        assert!(det.worst_regression().is_none());
    }

    #[test]
    fn test_detector_clear_results() {
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::higher_is_better("fps", 100.0));
        det.add_result(BenchResult::higher_is_better("fps", 50.0));
        assert_eq!(det.history.len(), 1);
        det.clear_results();
        assert!(det.history.is_empty());
        // baselines should remain
        assert_eq!(det.baseline_count(), 1);
    }

    #[test]
    fn test_detector_default_threshold() {
        // No explicit threshold registered → defaults to 5%
        let mut det = RegressionDetector::new();
        det.set_baseline(BenchResult::higher_is_better("fps", 100.0));
        det.add_result(BenchResult::higher_is_better("fps", 93.0)); // 7% drop, over 5%
        let regs = det.detect_regressions();
        assert_eq!(regs.len(), 1);
    }
}
