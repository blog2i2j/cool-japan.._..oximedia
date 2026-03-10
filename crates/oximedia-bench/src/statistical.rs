//! Statistical analysis for benchmark results.
//!
//! Provides descriptive statistics, outlier detection, and stability assessment
//! for raw benchmark sample data.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Descriptive statistics computed from a set of benchmark samples.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchStats {
    /// Minimum observed value.
    pub min: f64,
    /// Maximum observed value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl BenchStats {
    /// Returns the range (`max - min`).
    #[must_use]
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Coefficient of Variation: `std_dev / mean`.
    ///
    /// Returns `f64::INFINITY` when `mean` is zero.
    #[must_use]
    pub fn cv(&self) -> f64 {
        if self.mean == 0.0 {
            f64::INFINITY
        } else {
            self.std_dev / self.mean
        }
    }

    /// Returns `true` when the CV is at or below `max_cv`, indicating stable results.
    #[must_use]
    pub fn is_stable(&self, max_cv: f64) -> bool {
        self.cv() <= max_cv
    }
}

/// Compute statistics from a non-empty slice of samples.
///
/// Returns `None` when `samples` is empty.
#[must_use]
pub fn compute_stats(samples: &[f64]) -> Option<BenchStats> {
    if samples.is_empty() {
        return None;
    }

    let n = samples.len();

    // Sort a copy for percentile computation.
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[n - 1];

    let mean = samples.iter().sum::<f64>() / n as f64;

    let median = percentile_sorted(&sorted, 0.50);
    let p95 = percentile_sorted(&sorted, 0.95);
    let p99 = percentile_sorted(&sorted, 0.99);

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    Some(BenchStats {
        min,
        max,
        mean,
        median,
        std_dev,
        p95,
        p99,
    })
}

/// Compute a percentile from a **sorted** slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let frac = index - lower as f64;
    sorted[lower] + frac * (sorted[upper] - sorted[lower])
}

/// Outlier detector using the IQR (interquartile range) method.
#[derive(Debug, Clone)]
pub struct OutlierDetector {
    /// Multiplier applied to the IQR to set the fence distance.
    /// The standard Tukey fence uses 1.5; a wider fence uses 3.0.
    pub iqr_multiplier: f64,
}

impl Default for OutlierDetector {
    fn default() -> Self {
        Self {
            iqr_multiplier: 1.5,
        }
    }
}

impl OutlierDetector {
    /// Create an outlier detector with a custom IQR multiplier.
    #[must_use]
    pub fn new(iqr_multiplier: f64) -> Self {
        Self { iqr_multiplier }
    }

    /// Find the indices of outlier samples using the IQR method.
    ///
    /// A value is an outlier when it falls outside
    /// `[Q1 - k*IQR, Q3 + k*IQR]` where `k = iqr_multiplier`.
    ///
    /// Returns an empty `Vec` when `samples` has fewer than 4 elements.
    #[must_use]
    pub fn find_outliers(&self, samples: &[f64]) -> Vec<usize> {
        if samples.len() < 4 {
            return Vec::new();
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = percentile_sorted(&sorted, 0.25);
        let q3 = percentile_sorted(&sorted, 0.75);
        let iqr = q3 - q1;

        let lower_fence = q1 - self.iqr_multiplier * iqr;
        let upper_fence = q3 + self.iqr_multiplier * iqr;

        samples
            .iter()
            .enumerate()
            .filter(|(_, &v)| v < lower_fence || v > upper_fence)
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- compute_stats tests -----

    #[test]
    fn test_compute_stats_empty_returns_none() {
        assert!(compute_stats(&[]).is_none());
    }

    #[test]
    fn test_compute_stats_single_element() {
        let stats = compute_stats(&[42.0]).expect("stats should be valid");
        assert_eq!(stats.min, 42.0);
        assert_eq!(stats.max, 42.0);
        assert_eq!(stats.mean, 42.0);
        assert_eq!(stats.std_dev, 0.0);
    }

    #[test]
    fn test_compute_stats_mean() {
        let stats = compute_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]).expect("stats should be valid");
        assert!((stats.mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_min_max() {
        let stats = compute_stats(&[5.0, 1.0, 3.0]).expect("stats should be valid");
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
    }

    #[test]
    fn test_compute_stats_range() {
        let stats = compute_stats(&[10.0, 20.0]).expect("stats should be valid");
        assert_eq!(stats.range(), 10.0);
    }

    #[test]
    fn test_compute_stats_std_dev_uniform() {
        // All same value → std_dev should be 0
        let stats = compute_stats(&[7.0, 7.0, 7.0, 7.0]).expect("stats should be valid");
        assert!(stats.std_dev < 1e-10);
    }

    #[test]
    fn test_compute_stats_median_odd() {
        let stats = compute_stats(&[1.0, 3.0, 5.0]).expect("stats should be valid");
        assert!((stats.median - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_median_even() {
        let stats = compute_stats(&[1.0, 2.0, 3.0, 4.0]).expect("stats should be valid");
        assert!((stats.median - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_bench_stats_cv() {
        let stats = compute_stats(&[10.0, 10.0, 10.0]).expect("stats should be valid");
        assert_eq!(stats.cv(), 0.0);
    }

    #[test]
    fn test_bench_stats_cv_zero_mean() {
        let s = BenchStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            p95: 0.0,
            p99: 0.0,
        };
        assert_eq!(s.cv(), f64::INFINITY);
    }

    #[test]
    fn test_bench_stats_is_stable() {
        let stats = compute_stats(&[100.0, 101.0, 99.0, 100.0]).expect("stats should be valid");
        assert!(stats.is_stable(0.1)); // CV should be very small
    }

    #[test]
    fn test_bench_stats_not_stable() {
        let stats = compute_stats(&[1.0, 100.0, 50.0]).expect("stats should be valid");
        assert!(!stats.is_stable(0.05));
    }

    // ----- OutlierDetector tests -----

    #[test]
    fn test_outlier_detector_default() {
        let det = OutlierDetector::default();
        assert!((det.iqr_multiplier - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_outliers_none_expected() {
        let det = OutlierDetector::default();
        let samples = vec![10.0, 11.0, 12.0, 11.5, 10.5];
        let outliers = det.find_outliers(&samples);
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_find_outliers_detects_high() {
        let det = OutlierDetector::default();
        let samples = vec![10.0, 11.0, 10.5, 11.5, 10.2, 10.8, 200.0];
        let outliers = det.find_outliers(&samples);
        assert!(outliers.contains(&6)); // index of 200.0
    }

    #[test]
    fn test_find_outliers_too_few_samples() {
        let det = OutlierDetector::default();
        let samples = vec![1.0, 2.0, 3.0];
        let outliers = det.find_outliers(&samples);
        assert!(outliers.is_empty());
    }
}
