//! ETA smoothing — rolling-average / exponential-decay ETA for progress reports.
//!
//! Raw ETA estimates computed from instantaneous encoding speed are notoriously
//! unstable: a short burst of fast frames can make the ETA jump from 10 minutes
//! to 2 minutes and back.  This module provides two complementary smoothers:
//!
//! - [`RollingEtaSmoother`] — maintains a fixed-size ring-buffer of past ETA
//!   samples and returns their arithmetic mean.  Simple and auditable.
//!
//! - [`ExponentialEtaSmoother`] — applies an exponentially weighted moving
//!   average (EWMA) with a configurable smoothing factor `α`.  Reacts faster
//!   than a rolling window when genuine progress accelerates, while still
//!   damping noise.
//!
//! Both implement the [`EtaSmoother`] trait so they can be used interchangeably
//! in a [`SmoothedProgressTracker`].
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//! use oximedia_transcode::eta_smoother::{RollingEtaSmoother, EtaSmoother};
//!
//! let mut smoother = RollingEtaSmoother::new(8);
//! smoother.push(Duration::from_secs(120));
//! smoother.push(Duration::from_secs(100));
//! smoother.push(Duration::from_secs(110));
//! let smoothed = smoother.smoothed_eta().unwrap();
//! assert!(smoothed.as_secs() > 0);
//! ```

use std::collections::VecDeque;
use std::time::Duration;

// ──────────────────────────────────────────────────────────────────────────────
// Trait
// ──────────────────────────────────────────────────────────────────────────────

/// Common interface for ETA smoothers.
pub trait EtaSmoother {
    /// Push a new raw ETA sample.
    fn push(&mut self, raw_eta: Duration);

    /// Returns the current smoothed ETA, or `None` if not enough data has been
    /// collected yet.
    fn smoothed_eta(&self) -> Option<Duration>;

    /// Reset all internal state (e.g. when a new encoding pass starts).
    fn reset(&mut self);

    /// Number of samples currently held.
    fn sample_count(&self) -> usize;
}

// ──────────────────────────────────────────────────────────────────────────────
// Rolling (simple moving average)
// ──────────────────────────────────────────────────────────────────────────────

/// An ETA smoother based on a simple moving average over a ring buffer.
///
/// The most recent `window_size` ETA samples are averaged.  Older samples are
/// evicted automatically once the buffer is full.
pub struct RollingEtaSmoother {
    window: VecDeque<Duration>,
    window_size: usize,
    min_samples: usize,
}

impl RollingEtaSmoother {
    /// Create a new rolling smoother with the given window size.
    ///
    /// `window_size` must be ≥ 1; values < 1 are clamped to 1.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        let window_size = window_size.max(1);
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            min_samples: 1,
        }
    }

    /// Set the minimum number of samples required before a smoothed ETA is
    /// returned.  Defaults to 1.  Clamped to `window_size`.
    #[must_use]
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples.min(self.window_size).max(1);
        self
    }

    /// Current window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl EtaSmoother for RollingEtaSmoother {
    fn push(&mut self, raw_eta: Duration) {
        if self.window.len() == self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(raw_eta);
    }

    fn smoothed_eta(&self) -> Option<Duration> {
        if self.window.len() < self.min_samples {
            return None;
        }
        let total_nanos: u128 = self
            .window
            .iter()
            .map(|d| d.as_nanos())
            .sum();
        let avg_nanos = total_nanos / self.window.len() as u128;
        Some(Duration::from_nanos(avg_nanos as u64))
    }

    fn reset(&mut self) {
        self.window.clear();
    }

    fn sample_count(&self) -> usize {
        self.window.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Exponential (EWMA)
// ──────────────────────────────────────────────────────────────────────────────

/// An ETA smoother based on an exponentially weighted moving average (EWMA).
///
/// The smoothed value is updated as:
///
/// ```text
/// smoothed = α × raw_eta + (1 − α) × smoothed_prev
/// ```
///
/// where `α ∈ (0, 1]`.  A higher `α` reacts faster to changes;
/// a lower `α` provides more stable (slower-reacting) estimates.
pub struct ExponentialEtaSmoother {
    /// Smoothing factor α (0 < α ≤ 1).
    alpha: f64,
    /// Current smoothed value in nanoseconds, or `None` if no samples yet.
    smoothed_nanos: Option<f64>,
    /// Number of samples received.
    count: usize,
}

impl ExponentialEtaSmoother {
    /// Create a new EWMA smoother.
    ///
    /// `alpha` — smoothing factor in the range (0.0, 1.0].  Common choices:
    /// - `0.1` — very smooth, slow to react
    /// - `0.3` — balanced default
    /// - `0.7` — fast reaction, moderate smoothing
    ///
    /// Values outside `(0, 1]` are clamped to the nearest valid bound.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(1e-6, 1.0);
        Self { alpha, smoothed_nanos: None, count: 0 }
    }

    /// Returns the smoothing factor.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns the current smoothed ETA as a raw `f64` in seconds, or `None`.
    #[must_use]
    pub fn smoothed_seconds(&self) -> Option<f64> {
        self.smoothed_nanos
            .map(|n| n / 1_000_000_000.0)
    }
}

impl EtaSmoother for ExponentialEtaSmoother {
    fn push(&mut self, raw_eta: Duration) {
        let raw_nanos = raw_eta.as_nanos() as f64;
        self.smoothed_nanos = Some(match self.smoothed_nanos {
            None => raw_nanos,
            Some(prev) => self.alpha * raw_nanos + (1.0 - self.alpha) * prev,
        });
        self.count += 1;
    }

    fn smoothed_eta(&self) -> Option<Duration> {
        self.smoothed_nanos
            .map(|n| Duration::from_nanos(n as u64))
    }

    fn reset(&mut self) {
        self.smoothed_nanos = None;
        self.count = 0;
    }

    fn sample_count(&self) -> usize {
        self.count
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Hybrid smoother (rolling + EWMA in series)
// ──────────────────────────────────────────────────────────────────────────────

/// A two-stage smoother: first a rolling average, then an EWMA over the rolling
/// average output.  This combines the outlier-rejection of a window average with
/// the temporal continuity of EWMA.
pub struct HybridEtaSmoother {
    rolling: RollingEtaSmoother,
    ewma: ExponentialEtaSmoother,
}

impl HybridEtaSmoother {
    /// Create a hybrid smoother.
    ///
    /// - `window_size` — size of the inner rolling window.
    /// - `alpha` — EWMA smoothing factor for the second stage.
    #[must_use]
    pub fn new(window_size: usize, alpha: f64) -> Self {
        Self {
            rolling: RollingEtaSmoother::new(window_size),
            ewma: ExponentialEtaSmoother::new(alpha),
        }
    }
}

impl EtaSmoother for HybridEtaSmoother {
    fn push(&mut self, raw_eta: Duration) {
        self.rolling.push(raw_eta);
        if let Some(rolling_out) = self.rolling.smoothed_eta() {
            self.ewma.push(rolling_out);
        }
    }

    fn smoothed_eta(&self) -> Option<Duration> {
        self.ewma.smoothed_eta()
    }

    fn reset(&mut self) {
        self.rolling.reset();
        self.ewma.reset();
    }

    fn sample_count(&self) -> usize {
        self.ewma.sample_count()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SmoothedProgressTracker
// ──────────────────────────────────────────────────────────────────────────────

/// Wrapper that integrates any [`EtaSmoother`] into a simple progress tracker.
///
/// The caller reports the current frame and total frames; the tracker computes
/// a raw ETA from the elapsed time and feeds it into the smoother.
pub struct SmoothedProgressTracker<S: EtaSmoother> {
    smoother: S,
    total_frames: u64,
    current_frame: u64,
    start_nanos: Option<u64>,
}

impl<S: EtaSmoother> SmoothedProgressTracker<S> {
    /// Create a new tracker wrapping `smoother` for a job with `total_frames`.
    #[must_use]
    pub fn new(smoother: S, total_frames: u64) -> Self {
        Self {
            smoother,
            total_frames,
            current_frame: 0,
            start_nanos: None,
        }
    }

    /// Report progress as elapsed nanoseconds and current frame.
    ///
    /// Returns the smoothed ETA, or `None` if there is not enough history yet.
    pub fn update(&mut self, elapsed_nanos: u64, current_frame: u64) -> Option<Duration> {
        self.current_frame = current_frame;
        if self.start_nanos.is_none() {
            self.start_nanos = Some(elapsed_nanos.saturating_sub(1));
        }

        if current_frame == 0 || elapsed_nanos == 0 {
            return None;
        }

        let remaining = self.total_frames.saturating_sub(current_frame);
        // raw ETA = (elapsed / frames_done) × frames_remaining
        let nanos_per_frame = elapsed_nanos as f64 / current_frame as f64;
        let raw_eta_nanos = (nanos_per_frame * remaining as f64) as u64;
        let raw_eta = Duration::from_nanos(raw_eta_nanos);

        self.smoother.push(raw_eta);
        self.smoother.smoothed_eta()
    }

    /// Progress percentage (0.0–100.0).
    #[must_use]
    pub fn percent(&self) -> f64 {
        if self.total_frames == 0 {
            return 100.0;
        }
        (self.current_frame as f64 / self.total_frames as f64 * 100.0).min(100.0)
    }

    /// Reset the tracker for a new pass.
    pub fn reset(&mut self) {
        self.smoother.reset();
        self.current_frame = 0;
        self.start_nanos = None;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RollingEtaSmoother ─────────────────────────────────────────────────────

    #[test]
    fn test_rolling_returns_none_before_min_samples() {
        let mut s = RollingEtaSmoother::new(4).with_min_samples(3);
        s.push(Duration::from_secs(10));
        s.push(Duration::from_secs(20));
        assert!(s.smoothed_eta().is_none(), "should require 3 samples");
        s.push(Duration::from_secs(30));
        assert!(s.smoothed_eta().is_some());
    }

    #[test]
    fn test_rolling_average_correct() {
        let mut s = RollingEtaSmoother::new(3);
        s.push(Duration::from_secs(10));
        s.push(Duration::from_secs(20));
        s.push(Duration::from_secs(30));
        let eta = s.smoothed_eta().expect("should have data");
        assert_eq!(eta.as_secs(), 20, "average of 10, 20, 30 = 20");
    }

    #[test]
    fn test_rolling_evicts_oldest_sample() {
        let mut s = RollingEtaSmoother::new(3);
        s.push(Duration::from_secs(100)); // will be evicted
        s.push(Duration::from_secs(10));
        s.push(Duration::from_secs(20));
        s.push(Duration::from_secs(30)); // evicts 100
        let eta = s.smoothed_eta().expect("data present");
        assert_eq!(eta.as_secs(), 20, "average of 10, 20, 30 = 20");
    }

    #[test]
    fn test_rolling_reset_clears_samples() {
        let mut s = RollingEtaSmoother::new(4);
        s.push(Duration::from_secs(60));
        s.push(Duration::from_secs(60));
        assert_eq!(s.sample_count(), 2);
        s.reset();
        assert_eq!(s.sample_count(), 0);
        assert!(s.smoothed_eta().is_none());
    }

    #[test]
    fn test_rolling_window_size_clamped_to_1() {
        let s = RollingEtaSmoother::new(0);
        assert_eq!(s.window_size(), 1);
    }

    #[test]
    fn test_rolling_single_sample_returns_itself() {
        let mut s = RollingEtaSmoother::new(5);
        s.push(Duration::from_secs(42));
        let eta = s.smoothed_eta().expect("one sample is enough");
        assert_eq!(eta.as_secs(), 42);
    }

    // ── ExponentialEtaSmoother ─────────────────────────────────────────────────

    #[test]
    fn test_ewma_no_data_returns_none() {
        let s = ExponentialEtaSmoother::new(0.3);
        assert!(s.smoothed_eta().is_none());
    }

    #[test]
    fn test_ewma_first_sample_equals_raw() {
        let mut s = ExponentialEtaSmoother::new(0.3);
        s.push(Duration::from_secs(60));
        let eta = s.smoothed_eta().expect("one sample");
        assert_eq!(eta.as_secs(), 60);
    }

    #[test]
    fn test_ewma_converges_toward_new_value() {
        let mut s = ExponentialEtaSmoother::new(0.5);
        s.push(Duration::from_secs(100));
        // Push a smaller value; smoothed should move toward it.
        s.push(Duration::from_secs(50));
        let eta = s.smoothed_eta().expect("has data");
        assert!(eta.as_secs() < 100, "should move below 100 s");
        assert!(eta.as_secs() > 50, "should not yet reach 50 s");
    }

    #[test]
    fn test_ewma_alpha_clamped() {
        let s = ExponentialEtaSmoother::new(0.0); // should clamp to tiny ε
        assert!(s.alpha() > 0.0);
        let s2 = ExponentialEtaSmoother::new(2.0); // should clamp to 1.0
        assert!((s2.alpha() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ewma_sample_count_increments() {
        let mut s = ExponentialEtaSmoother::new(0.3);
        s.push(Duration::from_secs(10));
        s.push(Duration::from_secs(20));
        assert_eq!(s.sample_count(), 2);
    }

    #[test]
    fn test_ewma_reset() {
        let mut s = ExponentialEtaSmoother::new(0.3);
        s.push(Duration::from_secs(30));
        s.reset();
        assert!(s.smoothed_eta().is_none());
        assert_eq!(s.sample_count(), 0);
    }

    // ── HybridEtaSmoother ──────────────────────────────────────────────────────

    #[test]
    fn test_hybrid_returns_smoothed_value() {
        let mut h = HybridEtaSmoother::new(3, 0.5);
        for _ in 0..5 {
            h.push(Duration::from_secs(60));
        }
        let eta = h.smoothed_eta().expect("has data");
        // All inputs = 60 s, so result should be very close to 60 s.
        assert!((eta.as_secs_f64() - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_hybrid_reset_propagates() {
        let mut h = HybridEtaSmoother::new(3, 0.5);
        h.push(Duration::from_secs(30));
        h.reset();
        assert_eq!(h.sample_count(), 0);
    }

    // ── SmoothedProgressTracker ────────────────────────────────────────────────

    #[test]
    fn test_progress_tracker_zero_frame_returns_none() {
        let smoother = RollingEtaSmoother::new(4);
        let mut tracker = SmoothedProgressTracker::new(smoother, 1000);
        let eta = tracker.update(1_000_000, 0);
        assert!(eta.is_none());
    }

    #[test]
    fn test_progress_tracker_eta_decreases_as_work_progresses() {
        let smoother = RollingEtaSmoother::new(4);
        let mut tracker = SmoothedProgressTracker::new(smoother, 1000);
        let eta1 = tracker.update(1_000_000_000, 100); // 1 s elapsed, 100/1000 done
        let eta2 = tracker.update(2_000_000_000, 500); // 2 s elapsed, 500/1000 done
        // eta2 should reflect faster progress → smaller ETA
        if let (Some(e1), Some(e2)) = (eta1, eta2) {
            assert!(e2 < e1, "ETA should decrease as more work is done");
        }
    }

    #[test]
    fn test_progress_tracker_percent() {
        let smoother = ExponentialEtaSmoother::new(0.3);
        let mut tracker = SmoothedProgressTracker::new(smoother, 200);
        tracker.update(500_000_000, 50);
        let pct = tracker.percent();
        assert!((pct - 25.0).abs() < 0.1, "50/200 = 25%, got {pct}");
    }

    #[test]
    fn test_progress_tracker_reset() {
        let smoother = RollingEtaSmoother::new(4);
        let mut tracker = SmoothedProgressTracker::new(smoother, 500);
        tracker.update(1_000_000_000, 100);
        tracker.reset();
        assert!((tracker.percent() - 0.0).abs() < 1e-9);
    }
}
