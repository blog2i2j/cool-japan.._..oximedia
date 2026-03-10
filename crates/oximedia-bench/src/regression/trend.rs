//! Mann-Kendall monotonic trend test and advanced trend analysis.
//!
//! This module implements:
//! - [`MannKendallTrend`]: trend direction enum
//! - [`MannKendallResult`]: result of the non-parametric Mann-Kendall test
//! - [`TrendAnalyzer`]: high-level API for running trend tests on benchmark records
//! - Internal helpers: `mann_kendall_inner`, `standard_normal_cdf`

use serde::{Deserialize, Serialize};

use super::types::BenchmarkRecord;

// ─────────────────────────────────────────────────────────────────────────────
// Mann-Kendall types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of the Mann-Kendall monotonic trend test.
///
/// The Mann-Kendall test is a non-parametric test for monotonic trends in a
/// time series.  It does not assume normality and is robust against outliers.
///
/// Reference: Mann (1945), Kendall (1975).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannKendallResult {
    /// Kendall's S statistic (sum of sign differences).
    pub s_statistic: f64,
    /// Variance of S (accounts for ties).
    pub var_s: f64,
    /// Standardised Z score.
    pub z_score: f64,
    /// Approximate two-sided p-value (normal approximation, valid for n ≥ 8).
    pub p_value: f64,
    /// Detected trend direction.
    pub trend: MannKendallTrend,
    /// Whether the trend is statistically significant at α = 0.05.
    pub significant: bool,
    /// Number of data points used.
    pub n: usize,
    /// Sen's slope estimator (median of pair-wise slopes).
    pub sens_slope: f64,
}

/// Trend direction from the Mann-Kendall test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MannKendallTrend {
    /// Statistically significant increasing trend.
    Increasing,
    /// Statistically significant decreasing trend.
    Decreasing,
    /// No statistically significant monotonic trend.
    NoTrend,
}

// ─────────────────────────────────────────────────────────────────────────────
// TrendAnalyzer
// ─────────────────────────────────────────────────────────────────────────────

/// Advanced trend analysis on time series of benchmark records.
#[derive(Debug, Clone, Default)]
pub struct TrendAnalyzer;

impl TrendAnalyzer {
    /// Creates a new `TrendAnalyzer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Performs the Mann-Kendall monotonic trend test on the FPS values of
    /// the records for the given benchmark name.
    ///
    /// Returns `None` if fewer than 4 records are available (test is
    /// unreliable with very small samples).
    #[must_use]
    pub fn mann_kendall_test(
        &self,
        name: &str,
        history: &[BenchmarkRecord],
    ) -> Option<MannKendallResult> {
        let values: Vec<f64> = history
            .iter()
            .filter(|r| r.name == name)
            .map(|r| r.throughput_fps)
            .collect();

        if values.len() < 4 {
            return None;
        }

        Some(mann_kendall_inner(&values))
    }

    /// Performs Mann-Kendall on an arbitrary slice of f64 values.
    ///
    /// Returns `None` if fewer than 4 values.
    #[must_use]
    pub fn mann_kendall_values(&self, values: &[f64]) -> Option<MannKendallResult> {
        if values.len() < 4 {
            return None;
        }
        Some(mann_kendall_inner(values))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Core Mann-Kendall computation.
fn mann_kendall_inner(values: &[f64]) -> MannKendallResult {
    let n = values.len();

    // Compute S statistic: sum of sign(values[j] - values[i]) for all i < j.
    let mut s = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = values[j] - values[i];
            if diff > 0.0 {
                s += 1.0;
            } else if diff < 0.0 {
                s -= 1.0;
            }
            // diff == 0 contributes 0 (tie)
        }
    }

    // Compute variance of S accounting for ties.
    // Var(S) = [n(n-1)(2n+5) - Σ_g t_g(t_g-1)(2t_g+5)] / 18
    // where t_g is the count of the g-th tied group.
    let n_f = n as f64;
    let base_var = n_f * (n_f - 1.0) * (2.0 * n_f + 5.0) / 18.0;

    // Count ties
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut tie_correction = 0.0_f64;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && (sorted[j] - sorted[i]).abs() < f64::EPSILON {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction += t * (t - 1.0) * (2.0 * t + 5.0) / 18.0;
        }
        i = j;
    }

    let var_s = (base_var - tie_correction).max(0.0);

    // Compute Z score with continuity correction
    let z = if s > 0.0 {
        (s - 1.0) / var_s.sqrt().max(f64::EPSILON)
    } else if s < 0.0 {
        (s + 1.0) / var_s.sqrt().max(f64::EPSILON)
    } else {
        0.0
    };

    // Two-sided p-value using normal approximation (Φ is the standard normal CDF)
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));

    // Significance at α = 0.05 (|z| > 1.96)
    let significant = z.abs() > 1.96;

    // Trend direction
    let trend = if significant {
        if s > 0.0 {
            MannKendallTrend::Increasing
        } else {
            MannKendallTrend::Decreasing
        }
    } else {
        MannKendallTrend::NoTrend
    };

    // Sen's slope estimator: median of all pair-wise slopes (x_j - x_i)/(j - i)
    let mut slopes: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            slopes.push((values[j] - values[i]) / (j - i) as f64);
        }
    }
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sens_slope = if slopes.is_empty() {
        0.0
    } else {
        let mid = slopes.len() / 2;
        if slopes.len() % 2 == 0 {
            (slopes[mid - 1] + slopes[mid]) / 2.0
        } else {
            slopes[mid]
        }
    };

    MannKendallResult {
        s_statistic: s,
        var_s,
        z_score: z,
        p_value,
        trend,
        significant,
        n,
        sens_slope,
    }
}

/// Standard normal CDF approximation using the rational approximation by
/// Abramowitz and Stegun §26.2.17 (maximum error 7.5e-8).
fn standard_normal_cdf(x: f64) -> f64 {
    if x < 0.0 {
        return 1.0 - standard_normal_cdf(-x);
    }
    let t = 1.0 / (1.0 + 0.2316419 * x);
    let poly = t
        * (0.319_381_53
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    1.0 - ((-x * x / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()) * poly
}
