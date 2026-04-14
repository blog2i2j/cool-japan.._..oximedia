//! Bounded loudness history with linear-regression trend.
//!
//! [`LoudnessHistory`] is a lightweight, capacity-bounded ring buffer of LUFS
//! readings that also provides a [`trend`](LoudnessHistory::trend) estimate
//! computed via ordinary least-squares linear regression over the stored
//! samples.  A positive slope means loudness is increasing over time; a
//! negative slope means it is decreasing.
//!
//! This module is a lighter-weight companion to
//! [`crate::loudness_history::LoudnessHistory`], which tracks full
//! time-stamped measurements with true-peak and LRA data.  Use this module
//! when only the LUFS trend is needed.
//!
//! # Example
//!
//! ```
//! use oximedia_normalize::history::LoudnessHistory;
//!
//! let mut h = LoudnessHistory::new(10);
//! h.add(-23.0);
//! h.add(-22.0);
//! h.add(-21.0);
//!
//! // Loudness is rising → positive trend
//! let t = h.trend();
//! assert!(t > 0.0, "trend should be positive, got {t}");
//! ```

use std::collections::VecDeque;

/// Capacity-bounded loudness history with OLS trend estimation.
///
/// Stores at most `capacity` LUFS readings (oldest are evicted when full).
/// [`trend`](Self::trend) returns the slope of the best-fit line through the
/// stored samples as LUFS per sample index.
///
/// A capacity of 0 is treated as 1 to avoid division-by-zero edge cases.
#[derive(Debug, Clone)]
pub struct LoudnessHistory {
    capacity: usize,
    buf: VecDeque<f32>,
}

impl LoudnessHistory {
    /// Create a new history that retains at most `capacity` samples.
    ///
    /// If `capacity` is 0 it is silently promoted to 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            buf: VecDeque::with_capacity(capacity),
        }
    }

    /// Append a LUFS reading.  Evicts the oldest sample when full.
    pub fn add(&mut self, lufs: f32) {
        if self.buf.len() >= self.capacity {
            self.buf.pop_front();
        }
        self.buf.push_back(lufs);
    }

    /// Number of samples currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` when no samples have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Slope of the OLS linear regression (LUFS per sample index).
    ///
    /// Returns `0.0` when fewer than two samples are available — there is no
    /// trend to estimate with a single point.
    ///
    /// The regression uses x = 0, 1, …, n−1 as the independent variable and
    /// the stored LUFS values as y.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_normalize::history::LoudnessHistory;
    ///
    /// let mut h = LoudnessHistory::new(100);
    /// for i in 0..5 {
    ///     h.add(-23.0 + i as f32);  // strictly increasing
    /// }
    /// let slope = h.trend();
    /// assert!((slope - 1.0).abs() < 0.001, "slope={slope}");
    /// ```
    #[must_use]
    pub fn trend(&self) -> f32 {
        let n = self.buf.len();
        if n < 2 {
            return 0.0;
        }
        let nf = n as f32;
        // mean of x = 0..n-1 is (n-1)/2
        let mean_x = (nf - 1.0) / 2.0;
        let mean_y: f32 = self.buf.iter().sum::<f32>() / nf;

        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for (i, &y) in self.buf.iter().enumerate() {
            let dx = i as f32 - mean_x;
            num += dx * (y - mean_y);
            den += dx * dx;
        }

        if den == 0.0 {
            0.0
        } else {
            num / den
        }
    }

    /// Return the most recently added value, or `None` if empty.
    #[must_use]
    pub fn latest(&self) -> Option<f32> {
        self.buf.back().copied()
    }

    /// Arithmetic mean of all stored values, or `None` if empty.
    #[must_use]
    pub fn mean(&self) -> Option<f32> {
        if self.buf.is_empty() {
            return None;
        }
        Some(self.buf.iter().sum::<f32>() / self.buf.len() as f32)
    }

    /// Clear all stored samples.
    pub fn clear(&mut self) {
        self.buf.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let h = LoudnessHistory::new(10);
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_add_and_len() {
        let mut h = LoudnessHistory::new(5);
        h.add(-23.0);
        h.add(-22.0);
        assert_eq!(h.len(), 2);
    }

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut h = LoudnessHistory::new(3);
        h.add(-20.0);
        h.add(-21.0);
        h.add(-22.0);
        h.add(-23.0); // -20.0 should be evicted
        assert_eq!(h.len(), 3);
        assert_eq!(h.latest(), Some(-23.0));
    }

    #[test]
    fn test_trend_zero_with_single_sample() {
        let mut h = LoudnessHistory::new(10);
        h.add(-23.0);
        assert_eq!(h.trend(), 0.0);
    }

    #[test]
    fn test_trend_zero_with_no_samples() {
        let h = LoudnessHistory::new(10);
        assert_eq!(h.trend(), 0.0);
    }

    #[test]
    fn test_trend_positive_increasing() {
        let mut h = LoudnessHistory::new(100);
        for i in 0..5 {
            h.add(-23.0 + i as f32);
        }
        let slope = h.trend();
        assert!((slope - 1.0).abs() < 0.001, "slope={slope}");
    }

    #[test]
    fn test_trend_negative_decreasing() {
        let mut h = LoudnessHistory::new(100);
        for i in 0..5 {
            h.add(-20.0 - i as f32);
        }
        let slope = h.trend();
        assert!((slope - (-1.0)).abs() < 0.001, "slope={slope}");
    }

    #[test]
    fn test_trend_flat_is_zero() {
        let mut h = LoudnessHistory::new(100);
        for _ in 0..5 {
            h.add(-23.0);
        }
        assert!(h.trend().abs() < 1e-6);
    }

    #[test]
    fn test_mean() {
        let mut h = LoudnessHistory::new(10);
        h.add(-20.0);
        h.add(-24.0);
        let m = h.mean().expect("mean should exist");
        assert!((m - (-22.0)).abs() < 1e-5, "mean={m}");
    }

    #[test]
    fn test_clear() {
        let mut h = LoudnessHistory::new(10);
        h.add(-23.0);
        h.clear();
        assert!(h.is_empty());
    }

    #[test]
    fn test_zero_capacity_promoted_to_one() {
        let mut h = LoudnessHistory::new(0);
        h.add(-23.0);
        assert_eq!(h.len(), 1);
    }
}
