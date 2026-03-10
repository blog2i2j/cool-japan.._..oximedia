//! Baseline comparison and regression detection for `OxiMedia` benchmarks.
//!
//! Provides a simple store of baseline results and a comparison mechanism
//! that detects performance regressions against those baselines.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// A single recorded baseline measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    /// Benchmark name (e.g. "encode_av1_1080p").
    pub name: String,
    /// Metric name (e.g. "encode_fps", "psnr_db").
    pub metric: String,
    /// Recorded value for the metric.
    pub value: f64,
    /// Unit for the value (e.g. "fps", "dB", "ms").
    pub unit: String,
    /// Unix timestamp (seconds) when this baseline was recorded.
    pub recorded_at: u64,
}

impl BaselineResult {
    /// Create a new baseline result.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
        unit: impl Into<String>,
        recorded_at: u64,
    ) -> Self {
        Self {
            name: name.into(),
            metric: metric.into(),
            value,
            unit: unit.into(),
            recorded_at,
        }
    }
}

/// Severity classification of a detected regression.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    /// No regression detected.
    None,
    /// Minor regression (within 5–10% of threshold).
    Minor,
    /// Moderate regression (10–25%).
    Moderate,
    /// Severe regression (>25%).
    Severe,
}

/// A comparison between a baseline measurement and the current value.
#[derive(Debug, Clone)]
pub struct RegressionCheck {
    /// The baseline being compared against.
    pub baseline: BaselineResult,
    /// The current observed value.
    pub current: f64,
    /// Allowable percentage change before regression is declared (e.g. 5.0 = 5%).
    pub threshold_pct: f64,
}

impl RegressionCheck {
    /// Create a new regression check.
    #[must_use]
    pub fn new(baseline: BaselineResult, current: f64, threshold_pct: f64) -> Self {
        Self {
            baseline,
            current,
            threshold_pct,
        }
    }

    /// Returns `true` when the current value represents a regression beyond the threshold.
    ///
    /// A *regression* is defined as a decrease in value that exceeds `threshold_pct` percent
    /// of the baseline (for metrics where higher is better such as FPS, PSNR, SSIM).
    /// For latency-style metrics the caller should invert the value before calling this.
    #[must_use]
    pub fn is_regression(&self) -> bool {
        if self.baseline.value == 0.0 {
            return false;
        }
        let change_pct = self.improvement_pct();
        change_pct < -self.threshold_pct
    }

    /// Signed improvement percentage relative to the baseline.
    ///
    /// Positive means the current value is better (higher) than baseline.
    /// Negative means the current value is worse (lower).
    #[must_use]
    pub fn improvement_pct(&self) -> f64 {
        if self.baseline.value == 0.0 {
            return 0.0;
        }
        ((self.current - self.baseline.value) / self.baseline.value) * 100.0
    }

    /// Classify the severity of the regression.
    ///
    /// If `is_regression()` is `false`, returns `RegressionSeverity::None`.
    #[must_use]
    pub fn severity(&self) -> RegressionSeverity {
        if !self.is_regression() {
            return RegressionSeverity::None;
        }
        let drop_pct = self.improvement_pct().abs();
        if drop_pct > 25.0 {
            RegressionSeverity::Severe
        } else if drop_pct > 10.0 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        }
    }
}

/// Persistent store of baseline measurements.
#[derive(Debug, Clone, Default)]
pub struct BaselineStore {
    /// All recorded baselines.
    pub baselines: Vec<BaselineResult>,
}

impl BaselineStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new baseline measurement.
    pub fn record(&mut self, baseline: BaselineResult) {
        self.baselines.push(baseline);
    }

    /// Check the current value against the stored baseline for the given name and metric.
    ///
    /// Returns `None` if no baseline exists for that combination.
    #[must_use]
    pub fn check_regression(
        &self,
        name: &str,
        metric: &str,
        current: f64,
        threshold: f64,
    ) -> Option<RegressionCheck> {
        // Use the most-recently recorded baseline for this name+metric pair.
        let baseline = self
            .baselines
            .iter()
            .filter(|b| b.name == name && b.metric == metric)
            .max_by_key(|b| b.recorded_at)?;
        Some(RegressionCheck::new(baseline.clone(), current, threshold))
    }

    /// Update (overwrite) the baseline for the given name and metric with a new value.
    ///
    /// If no existing baseline matches, a new one is added.
    pub fn update(&mut self, name: &str, metric: &str, value: f64, now: u64) {
        // If an entry exists, update it in place; otherwise append.
        if let Some(existing) = self
            .baselines
            .iter_mut()
            .rfind(|b| b.name == name && b.metric == metric)
        {
            existing.value = value;
            existing.recorded_at = now;
        } else {
            self.baselines.push(BaselineResult {
                name: name.to_string(),
                metric: metric.to_string(),
                value,
                unit: String::new(),
                recorded_at: now,
            });
        }
    }

    /// Return all baselines for the given benchmark name.
    #[must_use]
    pub fn get_all(&self, name: &str) -> Vec<&BaselineResult> {
        self.baselines.iter().filter(|b| b.name == name).collect()
    }

    /// Return the most recent baseline for the given name and metric.
    #[must_use]
    pub fn latest(&self, name: &str, metric: &str) -> Option<&BaselineResult> {
        self.baselines
            .iter()
            .filter(|b| b.name == name && b.metric == metric)
            .max_by_key(|b| b.recorded_at)
    }

    /// Return the total number of stored baselines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.baselines.len()
    }

    /// Return `true` if the store contains no baselines.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.baselines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline(name: &str, metric: &str, value: f64, ts: u64) -> BaselineResult {
        BaselineResult::new(name, metric, value, "fps", ts)
    }

    #[test]
    fn test_baseline_result_construction() {
        let b = baseline("encode_av1", "fps", 30.0, 1_700_000_000);
        assert_eq!(b.name, "encode_av1");
        assert_eq!(b.metric, "fps");
        assert_eq!(b.value, 30.0);
        assert_eq!(b.recorded_at, 1_700_000_000);
    }

    #[test]
    fn test_no_regression_within_threshold() {
        let b = baseline("enc", "fps", 30.0, 0);
        let check = RegressionCheck::new(b, 28.5, 5.0); // -5% exactly
                                                        // 5% drop equals threshold, should NOT be a regression (strictly less than)
        assert!(!check.is_regression());
    }

    #[test]
    fn test_regression_beyond_threshold() {
        let b = baseline("enc", "fps", 30.0, 0);
        let check = RegressionCheck::new(b, 27.0, 5.0); // -10% drop
        assert!(check.is_regression());
    }

    #[test]
    fn test_improvement_pct_positive() {
        let b = baseline("enc", "fps", 30.0, 0);
        let check = RegressionCheck::new(b, 33.0, 5.0);
        assert!((check.improvement_pct() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_improvement_pct_negative() {
        let b = baseline("enc", "fps", 30.0, 0);
        let check = RegressionCheck::new(b, 27.0, 5.0);
        assert!((check.improvement_pct() - (-10.0)).abs() < 1e-9);
    }

    #[test]
    fn test_severity_none_when_no_regression() {
        let b = baseline("enc", "fps", 30.0, 0);
        let check = RegressionCheck::new(b, 30.0, 5.0);
        assert_eq!(check.severity(), RegressionSeverity::None);
    }

    #[test]
    fn test_severity_minor() {
        let b = baseline("enc", "fps", 100.0, 0);
        // 8% drop > 5% threshold -> minor (not >10%)
        let check = RegressionCheck::new(b, 92.0, 5.0);
        assert_eq!(check.severity(), RegressionSeverity::Minor);
    }

    #[test]
    fn test_severity_moderate() {
        let b = baseline("enc", "fps", 100.0, 0);
        // 20% drop -> moderate
        let check = RegressionCheck::new(b, 80.0, 5.0);
        assert_eq!(check.severity(), RegressionSeverity::Moderate);
    }

    #[test]
    fn test_severity_severe() {
        let b = baseline("enc", "fps", 100.0, 0);
        // 30% drop -> severe
        let check = RegressionCheck::new(b, 70.0, 5.0);
        assert_eq!(check.severity(), RegressionSeverity::Severe);
    }

    #[test]
    fn test_store_record_and_check() {
        let mut store = BaselineStore::new();
        store.record(baseline("enc", "fps", 30.0, 1000));
        let check = store
            .check_regression("enc", "fps", 27.0, 5.0)
            .expect("check should be valid");
        assert!(check.is_regression());
    }

    #[test]
    fn test_store_check_no_baseline_returns_none() {
        let store = BaselineStore::new();
        assert!(store.check_regression("enc", "fps", 30.0, 5.0).is_none());
    }

    #[test]
    fn test_store_update_existing() {
        let mut store = BaselineStore::new();
        store.record(baseline("enc", "fps", 30.0, 1000));
        store.update("enc", "fps", 35.0, 2000);
        let latest = store.latest("enc", "fps").expect("latest should be valid");
        assert_eq!(latest.value, 35.0);
        assert_eq!(latest.recorded_at, 2000);
    }

    #[test]
    fn test_store_update_new_entry() {
        let mut store = BaselineStore::new();
        store.update("new_bench", "latency", 5.0, 1000);
        assert_eq!(store.len(), 1);
        let b = store
            .latest("new_bench", "latency")
            .expect("b should be valid");
        assert_eq!(b.value, 5.0);
    }

    #[test]
    fn test_store_uses_most_recent_baseline_for_check() {
        let mut store = BaselineStore::new();
        store.record(baseline("enc", "fps", 30.0, 1000));
        store.record(baseline("enc", "fps", 40.0, 2000)); // newer
        let check = store
            .check_regression("enc", "fps", 38.0, 5.0)
            .expect("check should be valid");
        // Should compare against 40.0 (latest)
        assert!((check.baseline.value - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_store_is_empty() {
        let store = BaselineStore::new();
        assert!(store.is_empty());
    }

    #[test]
    fn test_get_all_filters_by_name() {
        let mut store = BaselineStore::new();
        store.record(baseline("enc_a", "fps", 30.0, 1));
        store.record(baseline("enc_b", "fps", 25.0, 2));
        store.record(baseline("enc_a", "psnr", 38.0, 3));
        let results = store.get_all("enc_a");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_regression_check_zero_baseline() {
        let b = BaselineResult::new("enc", "fps", 0.0, "fps", 0);
        let check = RegressionCheck::new(b, 0.0, 5.0);
        assert!(!check.is_regression());
        assert_eq!(check.improvement_pct(), 0.0);
    }
}
