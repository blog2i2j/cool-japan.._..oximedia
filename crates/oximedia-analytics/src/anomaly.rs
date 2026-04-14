//! Z-score anomaly detection over a rolling window.
//!
//! Maintains a fixed-size sliding window of recent values, computes the
//! running mean and standard deviation, and flags new observations whose
//! z-score exceeds a configurable threshold.
//!
//! For production-grade streaming anomaly detection the `crate::realtime`
//! module exposes `AnomalyDetector`, which is backed by a
//! similar algorithm.  This module provides a clean standalone API that is
//! easier to embed in pipelines that do not require the full realtime
//! aggregation stack.

use crate::error::AnalyticsError;
use std::collections::VecDeque;

// ─── ZScoreDetector ──────────────────────────────────────────────────────────

/// Rolling z-score anomaly detector.
///
/// Each call to [`ZScoreDetector::update`] appends the new value to a
/// fixed-length window.  Once the window has at least 2 observations the
/// z-score is computed:
///
/// ```text
/// z = (value − mean) / stddev
/// ```
///
/// When `|z|` exceeds the threshold the method returns `Some(z)`, otherwise
/// `None`.
#[derive(Debug, Clone)]
pub struct ZScoreDetector {
    /// Maximum number of observations kept in the sliding window.
    window_size: usize,
    /// Configurable z-score threshold (default 3.0).
    threshold: f64,
    /// Circular buffer of recent observations.
    window: VecDeque<f64>,
    /// Running sum (kept for O(1) mean update when removing old values).
    sum: f64,
    /// Running sum-of-squares (kept for O(1) variance update).
    sum_sq: f64,
}

impl ZScoreDetector {
    /// Creates a new detector with the given sliding-window size and the
    /// default threshold of **3.0**.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `window` is less than 2.
    pub fn new(window: usize) -> Result<Self, AnalyticsError> {
        if window < 2 {
            return Err(AnalyticsError::InvalidInput(
                "window size must be ≥ 2".into(),
            ));
        }
        Ok(Self {
            window_size: window,
            threshold: 3.0,
            window: VecDeque::with_capacity(window + 1),
            sum: 0.0,
            sum_sq: 0.0,
        })
    }

    /// Creates a new detector with a custom z-score `threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`AnalyticsError::InvalidInput`] when `window < 2` or
    /// `threshold ≤ 0`.
    pub fn with_threshold(window: usize, threshold: f64) -> Result<Self, AnalyticsError> {
        if window < 2 {
            return Err(AnalyticsError::InvalidInput(
                "window size must be ≥ 2".into(),
            ));
        }
        if threshold <= 0.0 {
            return Err(AnalyticsError::InvalidInput("threshold must be > 0".into()));
        }
        Ok(Self {
            window_size: window,
            threshold,
            window: VecDeque::with_capacity(window + 1),
            sum: 0.0,
            sum_sq: 0.0,
        })
    }

    /// Feeds a new `value` into the detector and returns the **signed z-score**
    /// when it exceeds the threshold, or `None` otherwise.
    ///
    /// # Behaviour
    ///
    /// - The first observation (and any observation before the window has ≥ 2
    ///   points) always returns `None` — there is not enough history to compute
    ///   a meaningful standard deviation.
    /// - Once the window is full the oldest value is evicted before the new
    ///   one is added.
    /// - If the running standard deviation rounds to zero (constant window),
    ///   `None` is returned to avoid division by zero.
    #[must_use]
    pub fn update(&mut self, value: f64) -> Option<f64> {
        // Evict oldest value when at capacity.
        if self.window.len() == self.window_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }

        // Insert new value.
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;

        let n = self.window.len();
        if n < 2 {
            return None;
        }

        // Population mean and standard deviation over the current window.
        let mean = self.sum / n as f64;
        let variance = (self.sum_sq / n as f64) - (mean * mean);
        // Clamp to avoid negative variance from floating-point cancellation.
        let std_dev = variance.max(0.0).sqrt();

        if std_dev < f64::EPSILON {
            return None;
        }

        let z = (value - mean) / std_dev;
        if z.abs() > self.threshold {
            Some(z)
        } else {
            None
        }
    }

    /// Returns the current rolling mean of the window.
    ///
    /// Returns `None` when the window is empty.
    #[must_use]
    pub fn mean(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.sum / self.window.len() as f64)
        }
    }

    /// Returns the current number of observations in the window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.window.len()
    }

    /// Resets the detector, clearing all accumulated state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_less_than_two_errors() {
        assert!(ZScoreDetector::new(0).is_err());
        assert!(ZScoreDetector::new(1).is_err());
    }

    #[test]
    fn valid_threshold_required() {
        assert!(ZScoreDetector::with_threshold(5, 0.0).is_err());
        assert!(ZScoreDetector::with_threshold(5, -1.0).is_err());
    }

    #[test]
    fn first_observation_none() {
        let mut d = ZScoreDetector::new(5).expect("valid");
        assert!(d.update(1.0).is_none());
    }

    #[test]
    fn constant_window_no_anomaly() {
        let mut d = ZScoreDetector::new(5).expect("valid");
        for _ in 0..5 {
            assert!(d.update(42.0).is_none());
        }
    }

    #[test]
    fn outlier_detected() {
        let mut d = ZScoreDetector::new(20).expect("valid");
        // Fill with near-constant signal.
        for i in 0..19 {
            let _ = d.update(10.0 + (i % 2) as f64 * 0.01);
        }
        // Feed an extreme outlier.
        let result = d.update(1000.0);
        assert!(result.is_some(), "extreme outlier should be detected");
        assert!(result.unwrap() > 0.0, "positive z-score");
    }

    #[test]
    fn negative_outlier_detected() {
        let mut d = ZScoreDetector::new(20).expect("valid");
        for _ in 0..19 {
            let _ = d.update(100.0);
        }
        let result = d.update(-1000.0);
        assert!(result.is_some());
        assert!(result.unwrap() < 0.0, "negative z-score");
    }

    #[test]
    fn normal_values_not_flagged() {
        let mut d = ZScoreDetector::new(10).expect("valid");
        // Slightly varying but well within 3σ.
        let values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.0, 1.03, 0.97];
        for &v in &values {
            let r = d.update(v);
            // None or small z — assert no anomaly.
            assert!(r.is_none(), "normal value {v} should not be flagged");
        }
    }

    #[test]
    fn reset_clears_state() {
        let mut d = ZScoreDetector::new(5).expect("valid");
        for _ in 0..5 {
            let _ = d.update(1.0);
        }
        d.reset();
        assert_eq!(d.window_len(), 0);
        assert!(d.mean().is_none());
        assert!(d.update(999.0).is_none()); // only 1 point — not enough history
    }

    #[test]
    fn mean_tracks_window() {
        let mut d = ZScoreDetector::new(4).expect("valid");
        d.update(2.0);
        d.update(4.0);
        let mean = d.mean().expect("some mean");
        assert!((mean - 3.0).abs() < 1e-9);
    }
}
