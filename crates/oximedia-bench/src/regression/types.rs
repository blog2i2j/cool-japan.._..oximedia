//! Core data types for the regression tracking subsystem.
//!
//! This module contains all plain data structures used across the regression
//! submodules: records, baselines, analysis results, severity classifications,
//! and configuration structs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Core record
// ─────────────────────────────────────────────────────────────────────────────

/// A single stored benchmark result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    /// Benchmark identifier (e.g. `"encode_av1_1080p"`).
    pub name: String,

    /// Unix timestamp (seconds since epoch) of when this run was recorded.
    pub timestamp: u64,

    /// Processing throughput in frames per second.
    pub throughput_fps: f64,

    /// Per-frame latency in milliseconds.
    pub latency_ms: f64,

    /// Peak memory usage in bytes during the run.
    pub memory_bytes: u64,

    /// Quality score (e.g. PSNR in dB, or VMAF 0–100).
    pub quality_score: f64,

    /// Arbitrary key-value annotations (codec version, host, git SHA, …).
    pub metadata: HashMap<String, String>,
}

impl BenchmarkRecord {
    /// Construct a minimal record for quick testing.
    #[must_use]
    pub fn simple(name: impl Into<String>, timestamp: u64, fps: f64, latency_ms: f64) -> Self {
        Self {
            name: name.into(),
            timestamp,
            throughput_fps: fps,
            latency_ms,
            memory_bytes: 0,
            quality_score: 0.0,
            metadata: HashMap::new(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Severity / kind enums
// ─────────────────────────────────────────────────────────────────────────────

/// Regression severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    /// 5–15 % degradation — worth watching.
    Minor,
    /// 15–30 % degradation — should be investigated.
    Moderate,
    /// 30–50 % degradation — likely a bug or configuration issue.
    Major,
    /// > 50 % degradation — critical production risk.
    Critical,
}

impl Severity {
    /// Classify a percent change into a severity level.
    ///
    /// `percent` is a **positive** value representing how much performance
    /// dropped compared to the baseline (e.g. `20.0` → 20 % worse).
    #[must_use]
    pub fn from_percent(percent: f64) -> Self {
        match percent {
            p if p >= 50.0 => Severity::Critical,
            p if p >= 30.0 => Severity::Major,
            p if p >= 15.0 => Severity::Moderate,
            _ => Severity::Minor,
        }
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Severity::Minor => "minor",
            Severity::Moderate => "moderate",
            Severity::Major => "major",
            Severity::Critical => "critical",
        }
    }
}

/// The outcome of comparing a single metric to its historical baseline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionKind {
    /// Performance improved by `percent` %.
    Improvement {
        /// Percentage improvement (positive value).
        percent: f64,
    },
    /// Performance degraded beyond the noise threshold.
    Regression {
        /// Percentage degradation (positive value).
        percent: f64,
        /// How severe this regression is.
        severity: Severity,
    },
    /// Change is within the statistical noise threshold — no action needed.
    Stable,
}

impl RegressionKind {
    /// Return `true` if this is a regression of any severity.
    #[must_use]
    pub fn is_regression(&self) -> bool {
        matches!(self, RegressionKind::Regression { .. })
    }

    /// Return `true` if this is an improvement.
    #[must_use]
    pub fn is_improvement(&self) -> bool {
        matches!(self, RegressionKind::Improvement { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Baseline
// ─────────────────────────────────────────────────────────────────────────────

/// Statistical baseline computed from recent historical records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkBaseline {
    /// Mean throughput in FPS across the baseline window.
    pub mean_fps: f64,
    /// Sample standard deviation of throughput.
    pub std_fps: f64,
    /// Mean latency in milliseconds.
    pub mean_latency_ms: f64,
    /// Sample standard deviation of latency.
    pub std_latency_ms: f64,
    /// Mean quality score.
    pub mean_quality: f64,
    /// Sample standard deviation of quality score.
    pub std_quality: f64,
    /// Number of samples used.
    pub sample_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Trend analysis struct
// ─────────────────────────────────────────────────────────────────────────────

/// Trend analysis computed via ordinary-least-squares linear regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Change in FPS per sequential run (slope of the regression line).
    pub slope_fps_per_run: f64,
    /// Whether throughput is trending downward over recent runs.
    pub is_trending_down: bool,
    /// Projected FPS for the *next* run based on the current trend.
    pub projected_fps_next: f64,
    /// R² coefficient of determination (0–1, higher = better linear fit).
    pub r_squared: f64,
    /// Number of data points used in the regression.
    pub sample_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Full analysis result
// ─────────────────────────────────────────────────────────────────────────────

/// Complete regression analysis for a single benchmark run compared to its baseline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    /// Benchmark name.
    pub name: String,
    /// Regression status for throughput (FPS).
    pub fps_kind: RegressionKind,
    /// Regression status for latency (lower is better, so inversion is applied).
    pub latency_kind: RegressionKind,
    /// Regression status for quality score (higher is better).
    pub quality_kind: RegressionKind,
    /// Z-score of the current FPS relative to the baseline distribution.
    pub fps_z_score: f64,
    /// Z-score of the current latency relative to the baseline distribution.
    pub latency_z_score: f64,
    /// Baseline statistics used for this analysis.
    pub baseline: BenchmarkBaseline,
    /// Whether at least one metric shows a statistically significant regression.
    pub has_regression: bool,
}

impl RegressionAnalysis {
    /// Return the worst severity across all regressing metrics, if any.
    #[must_use]
    pub fn worst_severity(&self) -> Option<Severity> {
        let kinds = [&self.fps_kind, &self.latency_kind, &self.quality_kind];
        let mut worst: Option<Severity> = None;
        for kind in &kinds {
            if let RegressionKind::Regression { severity, .. } = kind {
                worst = Some(match worst {
                    None => *severity,
                    Some(current) => Self::worse_severity(current, *severity),
                });
            }
        }
        worst
    }

    fn worse_severity(a: Severity, b: Severity) -> Severity {
        let rank = |s: Severity| match s {
            Severity::Minor => 0,
            Severity::Moderate => 1,
            Severity::Major => 2,
            Severity::Critical => 3,
        };
        if rank(a) >= rank(b) {
            a
        } else {
            b
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Detector configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration knobs for [`crate::regression::RegressionDetector`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Z-score threshold above which a change is flagged as a regression.
    /// The default value (2.0) corresponds to roughly 95 % confidence.
    pub z_score_threshold: f64,

    /// Minimum percentage change required before triggering a regression,
    /// acting as an absolute noise floor below the z-score test.
    pub min_regression_percent: f64,

    /// Confidence level for the reported confidence intervals (0.90 / 0.95 / 0.99).
    pub confidence_level: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            z_score_threshold: 2.0,
            min_regression_percent: 5.0,
            confidence_level: 0.95,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Confidence analysis result
// ─────────────────────────────────────────────────────────────────────────────

/// A regression analysis result that also carries confidence intervals for
/// each metric, enabling callers to reason about statistical uncertainty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceRegressionAnalysis {
    /// Core regression analysis (same as [`RegressionAnalysis`]).
    pub core: RegressionAnalysis,
    /// 95 % confidence interval for the FPS baseline mean: (lower, upper).
    pub fps_ci: (f64, f64),
    /// 95 % confidence interval for the latency baseline mean: (lower, upper).
    pub latency_ci: (f64, f64),
    /// 95 % confidence interval for the quality baseline mean: (lower, upper).
    pub quality_ci: (f64, f64),
    /// Whether the current FPS value lies *outside* the FPS confidence interval.
    pub fps_outside_ci: bool,
    /// Whether the current latency value lies *outside* the latency confidence interval.
    pub latency_outside_ci: bool,
    /// Whether the current quality value lies *outside* the quality confidence interval.
    pub quality_outside_ci: bool,
    /// Confidence level used (e.g. 0.95).
    pub confidence_level: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Outlier result types
// ─────────────────────────────────────────────────────────────────────────────

/// Outlier detection result for a single value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierInfo {
    /// The index of this value in the original slice.
    pub index: usize,
    /// The outlier value.
    pub value: f64,
    /// The lower fence used (Q1 - 1.5*IQR for standard; Q1 - 3*IQR for extreme).
    pub lower_fence: f64,
    /// The upper fence used.
    pub upper_fence: f64,
    /// Whether this is a mild outlier (outside 1.5*IQR) vs extreme (outside 3*IQR).
    pub is_extreme: bool,
}

/// Result of IQR outlier detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IqrOutlierResult {
    /// Q1 (25th percentile).
    pub q1: f64,
    /// Median (50th percentile).
    pub median: f64,
    /// Q3 (75th percentile).
    pub q3: f64,
    /// Inter-quartile range: Q3 - Q1.
    pub iqr: f64,
    /// Lower mild fence: Q1 - 1.5 * IQR.
    pub lower_mild_fence: f64,
    /// Upper mild fence: Q3 + 1.5 * IQR.
    pub upper_mild_fence: f64,
    /// Lower extreme fence: Q1 - 3.0 * IQR.
    pub lower_extreme_fence: f64,
    /// Upper extreme fence: Q3 + 3.0 * IQR.
    pub upper_extreme_fence: f64,
    /// All detected outliers.
    pub outliers: Vec<OutlierInfo>,
    /// Cleaned data — the input with outliers removed.
    pub cleaned: Vec<f64>,
}

impl IqrOutlierResult {
    /// Returns the number of detected outliers.
    #[must_use]
    pub fn outlier_count(&self) -> usize {
        self.outliers.len()
    }

    /// Returns `true` if any outliers were detected.
    #[must_use]
    pub fn has_outliers(&self) -> bool {
        !self.outliers.is_empty()
    }
}
