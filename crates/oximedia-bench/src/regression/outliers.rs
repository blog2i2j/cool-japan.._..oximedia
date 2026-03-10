//! IQR-based outlier detection for benchmark data.
//!
//! This module implements:
//! - [`OutlierDetector`]: detects statistical outliers using the Tukey IQR fences method
//! - `percentile`: linear-interpolation percentile on a pre-sorted slice (private helper)

use super::types::{BenchmarkRecord, IqrOutlierResult, OutlierInfo};

// ─────────────────────────────────────────────────────────────────────────────
// OutlierDetector
// ─────────────────────────────────────────────────────────────────────────────

/// Detects statistical outliers in benchmark data.
#[derive(Debug, Clone, Default)]
pub struct OutlierDetector;

impl OutlierDetector {
    /// Creates a new `OutlierDetector`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detects outliers in `values` using the IQR (Tukey fences) method.
    ///
    /// A value `x` is considered a *mild* outlier if:
    ///   x < Q1 − 1.5·IQR  or  x > Q3 + 1.5·IQR
    ///
    /// A value is an *extreme* outlier if:
    ///   x < Q1 − 3·IQR    or  x > Q3 + 3·IQR
    ///
    /// Returns `None` if `values` has fewer than 4 elements (quartiles are
    /// unreliable with very small samples).
    #[must_use]
    pub fn iqr_method(&self, values: &[f64]) -> Option<IqrOutlierResult> {
        if values.len() < 4 {
            return None;
        }

        let mut sorted: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_vals: Vec<f64> = sorted.iter().map(|(_, v)| *v).collect();

        let q1 = percentile(&sorted_vals, 0.25);
        let median = percentile(&sorted_vals, 0.50);
        let q3 = percentile(&sorted_vals, 0.75);
        let iqr = q3 - q1;

        let lower_mild = q1 - 1.5 * iqr;
        let upper_mild = q3 + 1.5 * iqr;
        let lower_extreme = q1 - 3.0 * iqr;
        let upper_extreme = q3 + 3.0 * iqr;

        let mut outliers: Vec<OutlierInfo> = Vec::new();
        let mut cleaned: Vec<f64> = Vec::new();

        for &(original_idx, val) in &sorted {
            let is_mild_outlier = val < lower_mild || val > upper_mild;
            let is_extreme_outlier = val < lower_extreme || val > upper_extreme;

            if is_mild_outlier {
                let (lower_fence, upper_fence) = if is_extreme_outlier {
                    (lower_extreme, upper_extreme)
                } else {
                    (lower_mild, upper_mild)
                };
                outliers.push(OutlierInfo {
                    index: original_idx,
                    value: val,
                    lower_fence,
                    upper_fence,
                    is_extreme: is_extreme_outlier,
                });
            } else {
                cleaned.push(val);
            }
        }

        // Restore cleaned data to original index order
        let mut cleaned_indexed: Vec<(usize, f64)> = cleaned
            .iter()
            .copied()
            .zip(
                sorted
                    .iter()
                    .filter(|(orig_i, v)| {
                        let is_mild = *v < lower_mild || *v > upper_mild;
                        let _ = orig_i;
                        !is_mild
                    })
                    .map(|(i, _)| *i),
            )
            .map(|(v, i)| (i, v))
            .collect();
        cleaned_indexed.sort_by_key(|(i, _)| *i);
        let cleaned: Vec<f64> = cleaned_indexed.iter().map(|(_, v)| *v).collect();

        Some(IqrOutlierResult {
            q1,
            median,
            q3,
            iqr,
            lower_mild_fence: lower_mild,
            upper_mild_fence: upper_mild,
            lower_extreme_fence: lower_extreme,
            upper_extreme_fence: upper_extreme,
            outliers,
            cleaned,
        })
    }

    /// Detects outlier benchmark *records* for a given benchmark name using
    /// the IQR method on throughput (FPS) values.
    ///
    /// Returns `None` if fewer than 4 records are found.
    #[must_use]
    pub fn detect_fps_outliers<'a>(
        &self,
        name: &str,
        history: &'a [BenchmarkRecord],
    ) -> Option<(IqrOutlierResult, Vec<&'a BenchmarkRecord>)> {
        let records: Vec<&BenchmarkRecord> = history.iter().filter(|r| r.name == name).collect();

        if records.len() < 4 {
            return None;
        }

        let fps: Vec<f64> = records.iter().map(|r| r.throughput_fps).collect();
        let iqr_result = self.iqr_method(&fps)?;

        // Map outlier indices back to the original records
        let outlier_records: Vec<&BenchmarkRecord> = iqr_result
            .outliers
            .iter()
            .map(|o| records[o.index])
            .collect();

        Some((iqr_result, outlier_records))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the p-th percentile of an already-sorted slice using linear interpolation.
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let h = p * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = h - lo as f64;

    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}
