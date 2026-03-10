//! Regression detector and benchmark history store.
//!
//! This module implements:
//! - [`RegressionDetector`]: z-score-based regression detection
//! - [`BenchmarkHistory`]: rolling in-memory history store with baseline computation
//! - Internal statistical helpers: mean, sample std-dev, z-score, confidence interval

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::types::{
    BenchmarkBaseline, BenchmarkRecord, ConfidenceRegressionAnalysis, DetectorConfig,
    RegressionAnalysis, RegressionKind, Severity, TrendAnalysis,
};

// ─────────────────────────────────────────────────────────────────────────────
// Internal statistical helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Arithmetic mean of `values`. Returns `0.0` for an empty slice.
pub(super) fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation (divides by n-1). Returns `0.0` for ≤1 element.
pub(super) fn sample_std_dev(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

/// Z-score: `(value - mean) / std_dev`.
/// Returns `0.0` if `std_dev` is effectively zero (avoids division by zero).
pub(super) fn z_score(value: f64, mean_val: f64, std_dev: f64) -> f64 {
    if std_dev < f64::EPSILON {
        return 0.0;
    }
    (value - mean_val) / std_dev
}

/// Critical z-value for two-sided normal confidence intervals.
pub(super) fn z_critical(confidence: f64) -> f64 {
    match confidence {
        c if (c - 0.99).abs() < 0.001 => 2.576,
        c if (c - 0.95).abs() < 0.001 => 1.960,
        c if (c - 0.90).abs() < 0.001 => 1.645,
        _ => 1.960, // default to 95 %
    }
}

/// Ordinary least-squares linear regression: returns `(slope, intercept, r²)`.
pub(super) fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, ys.first().copied().unwrap_or(0.0), 0.0);
    }

    let mean_x = mean(xs);
    let mean_y = mean(ys);

    let ss_xx: f64 = xs.iter().map(|x| (x - mean_x).powi(2)).sum();
    let ss_xy: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let ss_yy: f64 = ys.iter().map(|y| (y - mean_y).powi(2)).sum();

    if ss_xx.abs() < f64::EPSILON {
        return (0.0, mean_y, 0.0);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    // R² = (SS_xy)² / (SS_xx * SS_yy)
    let r_squared = if ss_yy.abs() < f64::EPSILON {
        1.0 // all y values identical — perfect horizontal fit
    } else {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    };

    (slope, intercept, r_squared.clamp(0.0, 1.0))
}

// ─────────────────────────────────────────────────────────────────────────────
// RegressionDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Detects performance regressions by comparing a current result to a historical baseline.
///
/// Uses z-score analysis to filter out noise: a change is only flagged if
/// `|z| > threshold` **and** the absolute percentage change exceeds
/// `min_regression_percent`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegressionDetector {
    /// Configuration for detection thresholds.
    pub config: DetectorConfig,
}

impl RegressionDetector {
    /// Create a detector with custom configuration.
    #[must_use]
    pub fn with_config(config: DetectorConfig) -> Self {
        Self { config }
    }

    /// Compare `current` against the distribution described by `history`.
    ///
    /// Returns a [`RegressionAnalysis`] with per-metric regression kinds and
    /// the baseline statistics used for the comparison.
    #[must_use]
    pub fn detect(
        &self,
        current: &BenchmarkRecord,
        history: &[BenchmarkRecord],
    ) -> RegressionAnalysis {
        // Build baseline from all historical records for this benchmark name.
        let relevant: Vec<&BenchmarkRecord> =
            history.iter().filter(|r| r.name == current.name).collect();

        if relevant.is_empty() {
            // No history — treat as stable (first run).
            let baseline = BenchmarkBaseline {
                mean_fps: current.throughput_fps,
                std_fps: 0.0,
                mean_latency_ms: current.latency_ms,
                std_latency_ms: 0.0,
                mean_quality: current.quality_score,
                std_quality: 0.0,
                sample_count: 0,
            };
            return RegressionAnalysis {
                name: current.name.clone(),
                fps_kind: RegressionKind::Stable,
                latency_kind: RegressionKind::Stable,
                quality_kind: RegressionKind::Stable,
                fps_z_score: 0.0,
                latency_z_score: 0.0,
                baseline,
                has_regression: false,
            };
        }

        let fps_values: Vec<f64> = relevant.iter().map(|r| r.throughput_fps).collect();
        let lat_values: Vec<f64> = relevant.iter().map(|r| r.latency_ms).collect();
        let qual_values: Vec<f64> = relevant.iter().map(|r| r.quality_score).collect();

        let mean_fps = mean(&fps_values);
        let std_fps = sample_std_dev(&fps_values);
        let mean_lat = mean(&lat_values);
        let std_lat = sample_std_dev(&lat_values);
        let mean_qual = mean(&qual_values);
        let std_qual = sample_std_dev(&qual_values);

        let baseline = BenchmarkBaseline {
            mean_fps,
            std_fps,
            mean_latency_ms: mean_lat,
            std_latency_ms: std_lat,
            mean_quality: mean_qual,
            std_quality: std_qual,
            sample_count: relevant.len(),
        };

        // FPS: higher is better — regression when current < mean.
        let fps_z = z_score(current.throughput_fps, mean_fps, std_fps);
        let fps_kind = self.classify_fps_regression(current.throughput_fps, mean_fps, fps_z);

        // Latency: lower is better — regression when current > mean.
        let lat_z = z_score(current.latency_ms, mean_lat, std_lat);
        let latency_kind = self.classify_latency_regression(current.latency_ms, mean_lat, lat_z);

        // Quality score: higher is better.
        let qual_z = z_score(current.quality_score, mean_qual, std_qual);
        let quality_kind = self.classify_fps_regression(current.quality_score, mean_qual, qual_z);

        let has_regression = fps_kind.is_regression()
            || latency_kind.is_regression()
            || quality_kind.is_regression();

        RegressionAnalysis {
            name: current.name.clone(),
            fps_kind,
            latency_kind,
            quality_kind,
            fps_z_score: fps_z,
            latency_z_score: lat_z,
            baseline,
            has_regression,
        }
    }

    /// Classify a "higher-is-better" metric (FPS, quality score).
    ///
    /// When the baseline has zero variance (all samples identical) a change
    /// beyond `min_regression_percent` is flagged deterministically, since
    /// there is no noise to filter out.
    fn classify_fps_regression(&self, current: f64, baseline_mean: f64, z: f64) -> RegressionKind {
        if baseline_mean.abs() < f64::EPSILON {
            return RegressionKind::Stable;
        }
        let percent_change = (baseline_mean - current) / baseline_mean * 100.0;
        let zero_std = z == 0.0 && (current - baseline_mean).abs() > f64::EPSILON;

        let is_regression = percent_change > self.config.min_regression_percent
            && (zero_std || z < -self.config.z_score_threshold);
        let is_improvement = percent_change < -self.config.min_regression_percent
            && (zero_std || z > self.config.z_score_threshold);

        if is_regression {
            RegressionKind::Regression {
                percent: percent_change,
                severity: Severity::from_percent(percent_change),
            }
        } else if is_improvement {
            RegressionKind::Improvement {
                percent: percent_change.abs(),
            }
        } else {
            RegressionKind::Stable
        }
    }

    /// Classify a "lower-is-better" metric (latency).
    ///
    /// When the baseline has zero variance (all samples identical) a change
    /// beyond `min_regression_percent` is flagged deterministically.
    fn classify_latency_regression(
        &self,
        current: f64,
        baseline_mean: f64,
        z: f64,
    ) -> RegressionKind {
        if baseline_mean.abs() < f64::EPSILON {
            return RegressionKind::Stable;
        }
        // For latency, an *increase* is a regression.
        let percent_change = (current - baseline_mean) / baseline_mean * 100.0;
        let zero_std = z == 0.0 && (current - baseline_mean).abs() > f64::EPSILON;

        let is_regression = percent_change > self.config.min_regression_percent
            && (zero_std || z > self.config.z_score_threshold);
        let is_improvement = percent_change < -self.config.min_regression_percent
            && (zero_std || z < -self.config.z_score_threshold);

        if is_regression {
            RegressionKind::Regression {
                percent: percent_change,
                severity: Severity::from_percent(percent_change),
            }
        } else if is_improvement {
            RegressionKind::Improvement {
                percent: percent_change.abs(),
            }
        } else {
            RegressionKind::Stable
        }
    }

    /// Compute a two-sided confidence interval for `samples`.
    ///
    /// `confidence` should be one of `0.90`, `0.95`, or `0.99`.
    /// The interval is `(mean - margin, mean + margin)` using a normal
    /// approximation (appropriate for n ≥ 30; reasonable for smaller samples).
    #[must_use]
    pub fn confidence_interval(samples: &[f64], confidence: f64) -> (f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        if samples.len() == 1 {
            return (samples[0], samples[0]);
        }

        let m = mean(samples);
        let s = sample_std_dev(samples);
        let n = samples.len() as f64;

        let z = z_critical(confidence);
        let margin = z * s / n.sqrt();
        (m - margin, m + margin)
    }

    /// Like [`RegressionDetector::detect`] but additionally computes
    /// confidence intervals for each metric and reports whether the
    /// observed value falls outside those intervals.
    ///
    /// The confidence level is taken from [`DetectorConfig::confidence_level`].
    #[must_use]
    pub fn detect_with_confidence(
        &self,
        current: &BenchmarkRecord,
        history: &[BenchmarkRecord],
    ) -> ConfidenceRegressionAnalysis {
        let core = self.detect(current, history);

        let relevant: Vec<&BenchmarkRecord> =
            history.iter().filter(|r| r.name == current.name).collect();

        let conf = self.config.confidence_level;

        if relevant.is_empty() {
            return ConfidenceRegressionAnalysis {
                fps_ci: (current.throughput_fps, current.throughput_fps),
                latency_ci: (current.latency_ms, current.latency_ms),
                quality_ci: (current.quality_score, current.quality_score),
                fps_outside_ci: false,
                latency_outside_ci: false,
                quality_outside_ci: false,
                confidence_level: conf,
                core,
            };
        }

        let fps_vals: Vec<f64> = relevant.iter().map(|r| r.throughput_fps).collect();
        let lat_vals: Vec<f64> = relevant.iter().map(|r| r.latency_ms).collect();
        let qual_vals: Vec<f64> = relevant.iter().map(|r| r.quality_score).collect();

        let fps_ci = Self::confidence_interval(&fps_vals, conf);
        let latency_ci = Self::confidence_interval(&lat_vals, conf);
        let quality_ci = Self::confidence_interval(&qual_vals, conf);

        let fps_outside_ci = current.throughput_fps < fps_ci.0 || current.throughput_fps > fps_ci.1;
        let latency_outside_ci =
            current.latency_ms < latency_ci.0 || current.latency_ms > latency_ci.1;
        let quality_outside_ci =
            current.quality_score < quality_ci.0 || current.quality_score > quality_ci.1;

        ConfidenceRegressionAnalysis {
            core,
            fps_ci,
            latency_ci,
            quality_ci,
            fps_outside_ci,
            quality_outside_ci,
            latency_outside_ci,
            confidence_level: conf,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// History store
// ─────────────────────────────────────────────────────────────────────────────

/// Persistent in-memory store of benchmark records with rolling-window eviction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkHistory {
    /// All stored records (sorted by insertion order / timestamp).
    pub records: Vec<BenchmarkRecord>,
    /// Maximum number of records to retain (oldest are dropped first).
    pub max_records: usize,
}

impl BenchmarkHistory {
    /// Create a new history store that retains at most `max_records` entries.
    #[must_use]
    pub fn new(max_records: usize) -> Self {
        Self {
            records: Vec::new(),
            max_records: max_records.max(1),
        }
    }

    /// Add a new record, evicting the oldest if the capacity is exceeded.
    pub fn add(&mut self, record: BenchmarkRecord) {
        self.records.push(record);
        if self.records.len() > self.max_records {
            let excess = self.records.len() - self.max_records;
            self.records.drain(0..excess);
        }
    }

    /// Iterate over all records that match the given benchmark name.
    pub fn records_for<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a BenchmarkRecord> {
        self.records.iter().filter(move |r| r.name == name)
    }

    /// Compute a [`BenchmarkBaseline`] from the most recent `window` records for `name`.
    ///
    /// Returns `None` if there are no records for `name`.
    #[must_use]
    pub fn baseline(&self, name: &str, window: usize) -> Option<BenchmarkBaseline> {
        let mut matching: Vec<&BenchmarkRecord> = self.records_for(name).collect();
        if matching.is_empty() {
            return None;
        }

        // Use only the most recent `window` records.
        let take = window.min(matching.len());
        let start = matching.len() - take;
        matching = matching[start..].to_vec();

        let fps_vals: Vec<f64> = matching.iter().map(|r| r.throughput_fps).collect();
        let lat_vals: Vec<f64> = matching.iter().map(|r| r.latency_ms).collect();
        let qual_vals: Vec<f64> = matching.iter().map(|r| r.quality_score).collect();

        Some(BenchmarkBaseline {
            mean_fps: mean(&fps_vals),
            std_fps: sample_std_dev(&fps_vals),
            mean_latency_ms: mean(&lat_vals),
            std_latency_ms: sample_std_dev(&lat_vals),
            mean_quality: mean(&qual_vals),
            std_quality: sample_std_dev(&qual_vals),
            sample_count: matching.len(),
        })
    }

    /// Compute a [`TrendAnalysis`] for `name` using all available records.
    ///
    /// Returns `None` if fewer than two records exist (cannot fit a line).
    #[must_use]
    pub fn trend(&self, name: &str) -> Option<TrendAnalysis> {
        let matching: Vec<&BenchmarkRecord> = self.records_for(name).collect();
        if matching.len() < 2 {
            return None;
        }

        // Use sequential run index (0, 1, 2, …) as the x-axis so that
        // unevenly-spaced timestamps don't distort the slope.
        let xs: Vec<f64> = (0..matching.len()).map(|i| i as f64).collect();
        let ys: Vec<f64> = matching.iter().map(|r| r.throughput_fps).collect();

        let (slope, intercept, r_squared) = linear_regression(&xs, &ys);

        let n = matching.len() as f64;
        let projected = slope * n + intercept;

        Some(TrendAnalysis {
            slope_fps_per_run: slope,
            is_trending_down: slope < 0.0,
            projected_fps_next: projected,
            r_squared,
            sample_count: matching.len(),
        })
    }

    /// Serialize the history to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error string if serialization fails.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }

    /// Deserialize a [`BenchmarkHistory`] from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error string if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }

    /// Return the total number of stored records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Return `true` if the history contains no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Return all distinct benchmark names present in the history.
    #[must_use]
    pub fn benchmark_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .records
            .iter()
            .map(|r| r.name.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        names.sort();
        names
    }
}
