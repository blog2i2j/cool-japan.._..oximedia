//! Performance regression detection.

use crate::benchmark::runner::BenchmarkResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Regression information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionInfo {
    /// Benchmark name.
    pub name: String,

    /// Baseline time.
    pub baseline: Duration,

    /// Current time.
    pub current: Duration,

    /// Regression percentage.
    pub regression_percent: f64,

    /// Number of standard deviations from baseline.
    pub std_deviations: f64,

    /// Whether this is statistically significant.
    pub is_significant: bool,
}

/// Regression detector.
#[derive(Debug)]
pub struct RegressionDetector {
    baselines: HashMap<String, BenchmarkResult>,
    threshold_percent: f64,
    std_dev_threshold: f64,
}

impl RegressionDetector {
    /// Create a new regression detector.
    pub fn new(threshold_percent: f64, std_dev_threshold: f64) -> Self {
        Self {
            baselines: HashMap::new(),
            threshold_percent,
            std_dev_threshold,
        }
    }

    /// Set baseline for a benchmark.
    pub fn set_baseline(&mut self, name: String, result: BenchmarkResult) {
        self.baselines.insert(name, result);
    }

    /// Detect regression compared to baseline.
    pub fn detect(&self, current: &BenchmarkResult) -> Option<RegressionInfo> {
        let baseline = self.baselines.get(&current.name)?;

        let baseline_secs = baseline.mean.as_secs_f64();
        let current_secs = current.mean.as_secs_f64();

        let regression_percent = ((current_secs - baseline_secs) / baseline_secs) * 100.0;

        if regression_percent < self.threshold_percent {
            return None; // No regression
        }

        let std_dev_secs = baseline.std_dev.as_secs_f64();
        let std_deviations = if std_dev_secs > 0.0 {
            (current_secs - baseline_secs) / std_dev_secs
        } else {
            0.0
        };

        let is_significant = std_deviations.abs() >= self.std_dev_threshold;

        Some(RegressionInfo {
            name: current.name.clone(),
            baseline: baseline.mean,
            current: current.mean,
            regression_percent,
            std_deviations,
            is_significant,
        })
    }

    /// Detect all regressions in a set of results.
    pub fn detect_all(&self, results: &[BenchmarkResult]) -> Vec<RegressionInfo> {
        results
            .iter()
            .filter_map(|result| self.detect(result))
            .collect()
    }

    /// Get baseline count.
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }

    /// Generate a report.
    pub fn report(&self, regressions: &[RegressionInfo]) -> String {
        let mut report = String::new();

        if regressions.is_empty() {
            report.push_str("No performance regressions detected.\n");
        } else {
            report.push_str(&format!(
                "Performance Regressions Detected: {}\n\n",
                regressions.len()
            ));

            for regression in regressions {
                let significance = if regression.is_significant {
                    "SIGNIFICANT"
                } else {
                    "MINOR"
                };

                report.push_str(&format!("[{}] {}\n", significance, regression.name));
                report.push_str(&format!("  Baseline: {:?}\n", regression.baseline));
                report.push_str(&format!("  Current:  {:?}\n", regression.current));
                report.push_str(&format!(
                    "  Regression: {:.2}%\n",
                    regression.regression_percent
                ));
                report.push_str(&format!(
                    "  Std Deviations: {:.2}\n\n",
                    regression.std_deviations
                ));
            }
        }

        report
    }
}

/// Configuration for advanced regression detection algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Basic percentage threshold (same as `threshold_percent`).
    pub threshold_percent: f64,
    /// Standard-deviation threshold (same as `std_dev_threshold`).
    pub std_dev_threshold: f64,
    /// Whether to use the Mann-Whitney U test for significance.
    pub use_mann_whitney: bool,
    /// Whether to use CUSUM change-point detection.
    pub use_cusum: bool,
    /// CUSUM drift parameter (typically 0.5 * expected shift / std_dev).
    pub cusum_drift: f64,
    /// CUSUM decision interval (h parameter, typically 4-5).
    pub cusum_h: f64,
    /// Minimum sample count for statistical tests.
    pub min_samples: usize,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            threshold_percent: 5.0,
            std_dev_threshold: 2.0,
            use_mann_whitney: true,
            use_cusum: true,
            cusum_drift: 0.5,
            cusum_h: 4.5,
            min_samples: 10,
        }
    }
}

/// Extended regression information with algorithm details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedRegressionInfo {
    /// Basic regression info.
    pub info: RegressionInfo,
    /// Mann-Whitney U statistic (if computed).
    pub mann_whitney_u: Option<f64>,
    /// Mann-Whitney p-value approximation (if computed).
    pub mann_whitney_p: Option<f64>,
    /// CUSUM change-point index (if detected).
    pub cusum_change_point: Option<usize>,
    /// CUSUM maximum cumulative sum at change-point.
    pub cusum_max_s: Option<f64>,
    /// Overall confidence that this is a true regression (0.0-1.0).
    pub confidence: f64,
}

impl RegressionDetector {
    /// Detect regressions using advanced statistical algorithms.
    ///
    /// If `baseline_samples` and `current_samples` are provided (raw timing
    /// data from repeated runs), the detector can apply:
    /// - Mann-Whitney U test for distribution shift significance
    /// - CUSUM change-point detection on the combined time series
    pub fn detect_advanced(
        &self,
        current: &BenchmarkResult,
        baseline_samples: &[f64],
        current_samples: &[f64],
        config: &RegressionConfig,
    ) -> Option<ExtendedRegressionInfo> {
        let basic = self.detect(current)?;

        let mut ext = ExtendedRegressionInfo {
            info: basic.clone(),
            mann_whitney_u: None,
            mann_whitney_p: None,
            cusum_change_point: None,
            cusum_max_s: None,
            confidence: 0.0,
        };

        let mut confidence_votes = 0_u32;
        let mut total_votes = 1_u32; // The basic threshold test already passed

        // Mann-Whitney U test
        if config.use_mann_whitney
            && baseline_samples.len() >= config.min_samples
            && current_samples.len() >= config.min_samples
        {
            let (u_stat, p_approx) = Self::mann_whitney_u(baseline_samples, current_samples);
            ext.mann_whitney_u = Some(u_stat);
            ext.mann_whitney_p = Some(p_approx);

            total_votes += 1;
            if p_approx < 0.05 {
                confidence_votes += 1;
            }
        }

        // CUSUM change-point detection
        if config.use_cusum && baseline_samples.len() >= config.min_samples {
            // Combine baseline + current into a single series
            let mut combined = baseline_samples.to_vec();
            combined.extend_from_slice(current_samples);

            if let Some((cp_idx, max_s)) =
                Self::cusum_detect(&combined, config.cusum_drift, config.cusum_h)
            {
                ext.cusum_change_point = Some(cp_idx);
                ext.cusum_max_s = Some(max_s);

                total_votes += 1;
                // Change-point near the boundary = strong evidence
                let boundary = baseline_samples.len();
                let distance = if cp_idx > boundary {
                    cp_idx - boundary
                } else {
                    boundary - cp_idx
                };
                if distance <= baseline_samples.len() / 4 {
                    confidence_votes += 1;
                }
            }
        }

        // Basic threshold test counts as a vote
        confidence_votes += 1;
        ext.confidence = f64::from(confidence_votes) / f64::from(total_votes);

        // Only report if we have reasonable confidence
        if ext.confidence >= 0.5 || basic.is_significant {
            Some(ext)
        } else {
            None
        }
    }

    /// Mann-Whitney U test for two independent samples.
    ///
    /// Returns (U statistic, approximate two-tailed p-value).
    /// Uses the normal approximation for the p-value.
    fn mann_whitney_u(sample_a: &[f64], sample_b: &[f64]) -> (f64, f64) {
        let n1 = sample_a.len() as f64;
        let n2 = sample_b.len() as f64;

        // Count how many times an element from B exceeds an element from A
        let mut u: f64 = 0.0;
        for &a in sample_a {
            for &b in sample_b {
                if b > a {
                    u += 1.0;
                } else if (b - a).abs() < f64::EPSILON {
                    u += 0.5;
                }
            }
        }

        // Normal approximation
        let mu_u = n1 * n2 / 2.0;
        let sigma_u = ((n1 * n2 * (n1 + n2 + 1.0)) / 12.0).sqrt();

        let z = if sigma_u > 0.0 {
            (u - mu_u) / sigma_u
        } else {
            0.0
        };

        // Approximate two-tailed p-value using the error function approximation
        let p = 2.0 * (1.0 - Self::normal_cdf(z.abs()));

        (u, p)
    }

    /// Approximate normal CDF using the Abramowitz-Stegun formula.
    fn normal_cdf(x: f64) -> f64 {
        if x < -8.0 {
            return 0.0;
        }
        if x > 8.0 {
            return 1.0;
        }

        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989422804014327; // 1/sqrt(2*pi)
        let p = d * (-x * x / 2.0).exp();

        let poly = t
            * (0.319381530
                + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

        if x >= 0.0 {
            1.0 - p * poly
        } else {
            p * poly
        }
    }

    /// CUSUM (Cumulative Sum) change-point detection.
    ///
    /// Detects an upward shift in the time series.
    /// Returns the index of the detected change-point and the maximum
    /// cumulative sum if a change is detected.
    fn cusum_detect(series: &[f64], drift: f64, h: f64) -> Option<(usize, f64)> {
        if series.len() < 4 {
            return None;
        }

        // Estimate mean and std_dev from the first half (assumed baseline)
        let half = series.len() / 2;
        let mean: f64 = series[..half].iter().sum::<f64>() / half as f64;
        let variance: f64 = series[..half]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / half as f64;
        let std_dev = variance.sqrt();

        if std_dev < f64::EPSILON {
            return None;
        }

        // Run CUSUM
        let mut s_pos = 0.0_f64;
        let mut max_s = 0.0_f64;
        let mut change_point: Option<usize> = None;

        for (i, &x) in series.iter().enumerate() {
            let z = (x - mean) / std_dev;
            s_pos = (s_pos + z - drift).max(0.0);

            if s_pos > max_s {
                max_s = s_pos;
            }

            if s_pos > h && change_point.is_none() {
                change_point = Some(i);
            }
        }

        change_point.map(|cp| (cp, max_s))
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(5.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(name: &str, mean_ms: u64, std_dev_ms: u64) -> BenchmarkResult {
        BenchmarkResult {
            name: name.to_string(),
            iterations: 100,
            mean: Duration::from_millis(mean_ms),
            median: Duration::from_millis(mean_ms),
            std_dev: Duration::from_millis(std_dev_ms),
            min: Duration::from_millis(mean_ms),
            max: Duration::from_millis(mean_ms),
            throughput: 1000.0 / mean_ms as f64,
        }
    }

    #[test]
    fn test_regression_detector() {
        let mut detector = RegressionDetector::new(5.0, 2.0);
        let baseline = make_result("test", 100, 5);
        detector.set_baseline("test".to_string(), baseline);

        let current = make_result("test", 120, 5);
        let regression = detector.detect(&current);

        assert!(regression.is_some());
        let reg = regression.expect("should succeed in test");
        assert!((reg.regression_percent - 20.0).abs() < 0.0001);
    }

    #[test]
    fn test_no_regression() {
        let mut detector = RegressionDetector::new(10.0, 2.0);
        let baseline = make_result("test", 100, 5);
        detector.set_baseline("test".to_string(), baseline);

        let current = make_result("test", 105, 5);
        let regression = detector.detect(&current);

        assert!(regression.is_none());
    }

    #[test]
    fn test_detect_advanced_significant() {
        let mut detector = RegressionDetector::new(5.0, 2.0);
        let baseline = make_result("test", 100, 5);
        detector.set_baseline("test".to_string(), baseline);

        let current = make_result("test", 130, 5);
        let config = RegressionConfig::default();

        // Create sample data representing baseline and current runs
        let baseline_samples: Vec<f64> = (0..20).map(|_| 0.100).collect();
        let current_samples: Vec<f64> = (0..20).map(|_| 0.130).collect();

        let ext = detector.detect_advanced(&current, &baseline_samples, &current_samples, &config);
        assert!(ext.is_some());
        let ext = ext.expect("should find regression");
        assert!(ext.confidence > 0.0);
        assert!(ext.mann_whitney_u.is_some());
    }

    #[test]
    fn test_cusum_detects_shift() {
        // Create a series with a clear shift: add small noise to baseline
        // so std_dev is non-zero, then shift by several std_devs
        let mut series: Vec<f64> = Vec::new();
        for i in 0..30 {
            // Baseline near 100 with +-1 noise
            series.push(100.0 + (i as f64 * 0.3).sin());
        }
        for i in 0..30 {
            // Shifted to ~110
            series.push(110.0 + (i as f64 * 0.3).sin());
        }

        // Use a lower decision interval for this magnitude of shift
        let result = RegressionDetector::cusum_detect(&series, 0.5, 3.0);
        assert!(result.is_some());
        let (cp, max_s) = result.expect("should detect change point");
        // Change-point should be detected somewhere in the shift region
        assert!(cp >= 20 && cp <= 45, "change point was {cp}");
        assert!(max_s > 3.0);
    }

    #[test]
    fn test_mann_whitney_identical_samples() {
        let a: Vec<f64> = (0..20).map(|_| 100.0).collect();
        let (_, p) = RegressionDetector::mann_whitney_u(&a, &a);
        // Identical samples should have p-value >= 0.05
        assert!(p >= 0.05 || (p - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_significance() {
        let mut detector = RegressionDetector::new(5.0, 2.0);
        let baseline = make_result("test", 100, 5);
        detector.set_baseline("test".to_string(), baseline);

        let current = make_result("test", 120, 5);
        let regression = detector.detect(&current).expect("should succeed in test");

        assert!(regression.is_significant);
    }
}
