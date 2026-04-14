//! Exact percentile computation over pre-sorted data.
//!
//! For large-scale approximate quantiles see the [`crate::quantile`] module
//! which implements t-digest.  This module provides exact percentiles useful
//! when the full dataset already fits in memory (e.g. quality-gate checks,
//! off-line benchmark reports).

use crate::error::AnalyticsError;

// ─── Percentiles ─────────────────────────────────────────────────────────────

/// Exact percentile calculator backed by a sorted data slice.
///
/// The nearest-rank method is used: `p(n)` returns the value at index
/// `⌈(n/100) × len⌉ − 1` (1-based rank clamped to valid range).
#[derive(Debug, Clone)]
pub struct Percentiles {
    /// Sorted copy of the input data.
    sorted: Vec<f64>,
}

impl Percentiles {
    /// Constructs a `Percentiles` from a **pre-sorted** slice.
    ///
    /// If `data` is not sorted the returned percentile values may be
    /// incorrect; consider using [`Percentiles::from_unsorted`] to
    /// sort automatically.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `data` is empty.
    pub fn from_sorted(data: &[f64]) -> Result<Self, AnalyticsError> {
        if data.is_empty() {
            return Err(AnalyticsError::InvalidInput(
                "data slice must not be empty".into(),
            ));
        }
        Ok(Self {
            sorted: data.to_vec(),
        })
    }

    /// Constructs a `Percentiles` by sorting a copy of `data`.
    ///
    /// NaN values are sorted to the end via `total_cmp`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `data` is empty.
    pub fn from_unsorted(data: &[f64]) -> Result<Self, AnalyticsError> {
        if data.is_empty() {
            return Err(AnalyticsError::InvalidInput(
                "data slice must not be empty".into(),
            ));
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));
        Ok(Self { sorted })
    }

    /// Returns the `n`-th percentile value using the nearest-rank method.
    ///
    /// `n` must be in `[1, 100]`.  Passing `0` is clamped to the minimum
    /// value; passing `100` returns the maximum value.
    ///
    /// # Arguments
    ///
    /// * `n` – percentile in the range `[1, 100]` (e.g. `50` for median,
    ///   `95` for p95, `99` for p99).
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `n > 100`.
    pub fn p(&self, n: u8) -> Result<f64, AnalyticsError> {
        if n > 100 {
            return Err(AnalyticsError::InvalidInput(
                "percentile n must be in [1, 100]".into(),
            ));
        }
        let len = self.sorted.len();
        // Nearest-rank: index = ceil(n/100 * len) - 1, clamped to [0, len-1].
        let rank = if n == 0 {
            0usize
        } else {
            let raw = (n as f64 / 100.0 * len as f64).ceil() as usize;
            raw.saturating_sub(1).min(len - 1)
        };
        Ok(self.sorted[rank])
    }

    /// Convenience: returns the median (p50).
    #[must_use]
    pub fn median(&self) -> f64 {
        // p(50) on non-empty data never returns Err.
        self.p(50).unwrap_or(f64::NAN)
    }

    /// Returns the number of data points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Returns `true` when the underlying dataset is empty.
    ///
    /// Because construction requires non-empty data this always returns
    /// `false` for successfully constructed instances.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}

/// Computes multiple percentiles in a single pass over sorted data.
///
/// Returns a `Vec<f64>` of the same length as `ns` with the value for each
/// requested percentile.
///
/// # Errors
///
/// Propagates any error from [`Percentiles::from_sorted`] or [`Percentiles::p`].
pub fn compute_percentiles(data: &[f64], ns: &[u8]) -> Result<Vec<f64>, AnalyticsError> {
    let p = Percentiles::from_sorted(data)?;
    ns.iter().map(|&n| p.p(n)).collect::<Result<Vec<_>, _>>()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(v: &[f64]) -> Percentiles {
        Percentiles::from_sorted(v).expect("non-empty")
    }

    #[test]
    fn empty_slice_errors() {
        assert!(Percentiles::from_sorted(&[]).is_err());
        assert!(Percentiles::from_unsorted(&[]).is_err());
    }

    #[test]
    fn single_element() {
        let p = sorted(&[42.0]);
        assert_eq!(p.p(1).unwrap(), 42.0);
        assert_eq!(p.p(50).unwrap(), 42.0);
        assert_eq!(p.p(100).unwrap(), 42.0);
    }

    #[test]
    fn p100_is_max() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let p = sorted(&data);
        assert_eq!(p.p(100).unwrap(), 10.0);
    }

    #[test]
    fn p50_median_five_elements() {
        // [1, 2, 3, 4, 5] → rank = ceil(0.5*5)=3 → index 2 → 3
        let p = sorted(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(p.p(50).unwrap(), 3.0);
        assert_eq!(p.median(), 3.0);
    }

    #[test]
    fn p95_ten_elements() {
        // [1..10] → rank = ceil(0.95*10)=10 → index 9 → 10
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let p = sorted(&data);
        assert_eq!(p.p(95).unwrap(), 10.0);
    }

    #[test]
    fn from_unsorted_produces_correct_result() {
        let data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let p = Percentiles::from_unsorted(&data).expect("non-empty");
        assert_eq!(p.p(100).unwrap(), 5.0);
        assert_eq!(p.p(1).unwrap(), 1.0);
    }

    #[test]
    fn n_above_100_errors() {
        let p = sorted(&[1.0, 2.0, 3.0]);
        assert!(p.p(101).is_err());
    }

    #[test]
    fn compute_percentiles_batch() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let result = compute_percentiles(&data, &[25, 50, 75, 99]).expect("ok");
        assert_eq!(result.len(), 4);
        // p99 of 1..=100 → rank=ceil(99)=99 → index 98 → 99.0
        assert_eq!(result[3], 99.0);
    }

    #[test]
    fn len_and_is_empty() {
        let p = sorted(&[1.0, 2.0, 3.0]);
        assert_eq!(p.len(), 3);
        assert!(!p.is_empty());
    }
}
