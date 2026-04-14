//! Online running statistics for bitrate estimation optimisation.
//!
//! This module provides zero-allocation, single-pass statistical accumulators
//! that can be used instead of buffering the entire sample set.  The primary
//! use-case is per-frame bitrate analysis inside the `bitrate_estimator`
//! pipeline: rather than storing every frame's QP/bits and computing
//! statistics at the end of a full pass, callers update an accumulator for
//! each encoded frame and query the current statistics at any time.
//!
//! # Algorithms
//!
//! | Type | Algorithm | Notes |
//! |------|-----------|-------|
//! | [`RunningStats`] | Welford online algorithm | Numerically stable mean + variance |
//! | [`Ewma`] | Exponential weighted moving average | Configurable α |
//! | [`RollingWindow`] | Fixed-size circular buffer | Exact sliding-window mean / stddev |
//! | [`PercentileEstimator`] | P² algorithm (Jain & Chlamtac 1985) | O(1) memory per quantile |
//!
//! # Example
//!
//! ```rust
//! use oximedia_transcode::running_stats::{RunningStats, Ewma};
//!
//! let mut stats = RunningStats::new();
//! for bits in [4_000u64, 5_000, 6_000, 4_500, 5_500] {
//!     stats.push(bits as f64);
//! }
//! assert!((stats.mean() - 5_000.0).abs() < 1.0);
//!
//! let mut ewma = Ewma::new(0.1);
//! ewma.update(100.0);
//! ewma.update(200.0);
//! assert!(ewma.value() > 100.0);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::VecDeque;

// ─── RunningStats (Welford) ───────────────────────────────────────────────────

/// Numerically-stable online mean and variance using Welford's algorithm.
///
/// Each call to [`push`](RunningStats::push) updates the running mean and the
/// running sum-of-squared-deviations in O(1) time with no stored samples.
/// The result is numerically more stable than the naive two-pass formula when
/// values span several orders of magnitude.
///
/// # References
///
/// Welford, B. P. (1962).  *Note on a method for calculating corrected sums
/// of squares and products*.  Technometrics, 4(3), 419–420.
#[derive(Debug, Clone, Default)]
pub struct RunningStats {
    /// Number of samples pushed so far.
    count: u64,
    /// Current running mean (μ_n).
    mean: f64,
    /// Running sum of squared deviations from the mean (M_n = Σ(xᵢ − μ_n)²).
    m2: f64,
    /// Minimum value seen.
    min: f64,
    /// Maximum value seen.
    max: f64,
}

impl RunningStats {
    /// Creates an empty accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Pushes a new sample and updates all running statistics.
    ///
    /// Silently ignores `NaN` values to avoid poisoning the accumulator.
    pub fn push(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Returns the number of samples pushed.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns `true` if no samples have been pushed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns the current running mean, or `0.0` if empty.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mean
        }
    }

    /// Returns the sample variance (divides by n − 1), or `0.0` for < 2 samples.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Returns the population variance (divides by n), or `0.0` if empty.
    #[must_use]
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Returns the sample standard deviation, or `0.0` for < 2 samples.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Returns the population standard deviation, or `0.0` if empty.
    #[must_use]
    pub fn population_stddev(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// Returns the minimum value seen, or `f64::INFINITY` if empty.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Returns the maximum value seen, or `f64::NEG_INFINITY` if empty.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Returns the range (max − min), or `0.0` if fewer than two samples.
    #[must_use]
    pub fn range(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.max - self.min
        }
    }

    /// Returns the coefficient of variation (stddev / |mean|) as a fraction.
    ///
    /// Returns `f64::NAN` when the mean is zero.
    #[must_use]
    pub fn cv(&self) -> f64 {
        let m = self.mean();
        if m == 0.0 {
            f64::NAN
        } else {
            self.stddev() / m.abs()
        }
    }

    /// Merges another `RunningStats` into `self` using Chan's parallel algorithm.
    ///
    /// After merging, `self` contains the combined statistics as though all
    /// samples from both accumulators had been pushed into a single one.
    ///
    /// # References
    ///
    /// Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). *Updating Formulae
    /// and a Pairwise Algorithm for Computing Sample Variances*.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }
        let combined = self.count + other.count;
        let delta = other.mean - self.mean;
        let new_mean =
            (self.mean * self.count as f64 + other.mean * other.count as f64) / combined as f64;
        let new_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64) / combined as f64;

        self.count = combined;
        self.mean = new_mean;
        self.m2 = new_m2;
        if other.min < self.min {
            self.min = other.min;
        }
        if other.max > self.max {
            self.max = other.max;
        }
    }

    /// Resets the accumulator to its initial empty state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

// ─── EWMA ────────────────────────────────────────────────────────────────────

/// Exponentially weighted moving average (EWMA).
///
/// `α ∈ (0, 1]` is the smoothing factor.  Higher α gives more weight to
/// recent observations.  The first call to [`update`](Ewma::update) sets the
/// initial value to the first sample (no warm-up bias).
///
/// # Usage
///
/// ```rust
/// use oximedia_transcode::running_stats::Ewma;
///
/// let mut ewma = Ewma::new(0.2);
/// for v in [100.0_f64, 110.0, 90.0, 105.0] {
///     ewma.update(v);
/// }
/// // value is close to the true mean but smoothed
/// assert!((ewma.value() - 100.0).abs() < 20.0);
/// ```
#[derive(Debug, Clone)]
pub struct Ewma {
    alpha: f64,
    value: Option<f64>,
    count: u64,
    /// Running variance using EWMA-based Welford variant.
    variance: f64,
}

impl Ewma {
    /// Creates an EWMA with smoothing factor `alpha`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds when `alpha` is outside `(0.0, 1.0]`.
    #[must_use]
    pub fn new(alpha: f64) -> Self {
        debug_assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        let alpha = alpha.clamp(1e-9, 1.0);
        Self {
            alpha,
            value: None,
            count: 0,
            variance: 0.0,
        }
    }

    /// Updates the EWMA with a new observation.
    ///
    /// Silently ignores `NaN` values.
    pub fn update(&mut self, sample: f64) {
        if sample.is_nan() {
            return;
        }
        self.count += 1;
        match self.value {
            None => {
                self.value = Some(sample);
                self.variance = 0.0;
            }
            Some(prev) => {
                let new_val = self.alpha * sample + (1.0 - self.alpha) * prev;
                let diff = sample - prev;
                self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
                self.value = Some(new_val);
            }
        }
    }

    /// Returns the current EWMA value, or `None` before any update.
    #[must_use]
    pub fn value_opt(&self) -> Option<f64> {
        self.value
    }

    /// Returns the current EWMA value, or `0.0` if not yet initialised.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.value.unwrap_or(0.0)
    }

    /// Returns the EWMA-based variance estimate.
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Returns the EWMA-based standard-deviation estimate.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Returns the number of observations processed.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the smoothing factor α.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Resets the EWMA to its initial state (no observations).
    pub fn reset(&mut self) {
        self.value = None;
        self.count = 0;
        self.variance = 0.0;
    }
}

// ─── RollingWindow ───────────────────────────────────────────────────────────

/// Exact sliding-window mean and variance over the last `capacity` samples.
///
/// Uses a circular buffer (VecDeque) plus Welford incremental updates on
/// the window boundary (remove oldest, add newest) so each push is O(1)
/// in both time and space.
#[derive(Debug, Clone)]
pub struct RollingWindow {
    capacity: usize,
    buffer: VecDeque<f64>,
    /// Welford running stats over the current window (not the whole history).
    sum: f64,
    sum_sq: f64,
}

impl RollingWindow {
    /// Creates a rolling window with the given `capacity` (must be ≥ 1).
    ///
    /// # Panics
    ///
    /// Panics when `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 1, "RollingWindow capacity must be ≥ 1");
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Pushes a new sample into the window, evicting the oldest if full.
    pub fn push(&mut self, value: f64) {
        if self.buffer.len() == self.capacity {
            // Evict oldest.
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.buffer.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    /// Returns the current number of samples in the window (≤ capacity).
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the window contains no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns `true` when the window has been filled to capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }

    /// Returns the window capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the mean of the current window, or `0.0` if empty.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.buffer.is_empty() {
            0.0
        } else {
            self.sum / self.buffer.len() as f64
        }
    }

    /// Returns the population variance of the current window.
    ///
    /// Uses the computational formula: σ² = E[X²] − (E[X])².
    #[must_use]
    pub fn variance(&self) -> f64 {
        let n = self.buffer.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        let mean = self.sum / n;
        let mean_sq = self.sum_sq / n;
        // Clamp to avoid negative variance from floating-point cancellation.
        (mean_sq - mean * mean).max(0.0)
    }

    /// Returns the population standard deviation of the current window.
    #[must_use]
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Returns the minimum value in the current window, or `f64::INFINITY` if empty.
    #[must_use]
    pub fn min(&self) -> f64 {
        self.buffer
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Returns the maximum value in the current window, or `f64::NEG_INFINITY` if empty.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.buffer
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Returns a snapshot of all current window samples, oldest first.
    #[must_use]
    pub fn samples(&self) -> Vec<f64> {
        self.buffer.iter().copied().collect()
    }
}

// ─── PercentileEstimator (P² algorithm) ──────────────────────────────────────

/// P² algorithm for online quantile estimation with O(1) memory per quantile.
///
/// Estimates a specific percentile `p` (0 < p < 1) without storing
/// individual samples.  After at least 5 observations the estimate is
/// maintained via the P² marker update rule.
///
/// Accuracy is typically within a few percent for stationary distributions
/// with thousands of samples.
///
/// # References
///
/// Jain, R., & Chlamtac, I. (1985). *The P² algorithm for dynamic calculation
/// of quantiles and histograms without storing observations.*
/// Communications of the ACM, 28(10), 1076–1085.
#[derive(Debug, Clone)]
pub struct PercentileEstimator {
    /// Target quantile in (0, 1).
    p: f64,
    /// Five marker heights q[0..5].
    q: [f64; 5],
    /// Five desired marker positions (non-integer).
    dn: [f64; 5],
    /// Five actual integer marker positions n[0..5].
    n: [i64; 5],
    /// Total number of observations.
    count: u64,
    /// Bootstrap buffer (first 5 samples before algorithm begins).
    bootstrap: Vec<f64>,
}

impl PercentileEstimator {
    /// Creates an estimator for the given quantile `p` (e.g. 0.95 for p95).
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `p` is not in (0, 1).
    #[must_use]
    pub fn new(p: f64) -> Self {
        debug_assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
        let p = p.clamp(1e-9, 1.0 - 1e-9);
        Self {
            p,
            q: [0.0; 5],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            n: [1, 2, 3, 4, 5],
            count: 0,
            bootstrap: Vec::with_capacity(5),
        }
    }

    /// Updates the estimator with a new observation.
    pub fn update(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.count += 1;

        // Bootstrap phase: collect first 5 samples.
        if self.bootstrap.len() < 5 {
            self.bootstrap.push(x);
            if self.bootstrap.len() == 5 {
                // Sort and initialise markers.
                self.bootstrap.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                for i in 0..5 {
                    self.q[i] = self.bootstrap[i];
                }
                self.n = [1, 2, 3, 4, 5];
            }
            return;
        }

        // P² update phase.
        // 1. Find cell k where x falls.
        let k = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x < self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        // 2. Increment positions n[k+1..5].
        for i in (k + 1)..5 {
            self.n[i] += 1;
        }

        // 3. Update desired positions.
        let obs_count = (self.count - 5) as f64 + 1.0; // 1-indexed after bootstrap
        self.dn = [
            0.0,
            self.p / 2.0 * obs_count,
            self.p * obs_count,
            (1.0 + self.p) / 2.0 * obs_count,
            obs_count,
        ];

        // 4. Adjust marker heights.
        for i in 1..=3 {
            let d = self.dn[i] - self.n[i] as f64;
            if (d >= 1.0 && (self.n[i + 1] - self.n[i]) > 1)
                || (d <= -1.0 && (self.n[i - 1] - self.n[i]) < -1)
            {
                let sign = if d > 0.0 { 1 } else { -1 };
                let q_new = self.parabolic(i, sign as f64);
                if q_new > self.q[i - 1] && q_new < self.q[i + 1] {
                    self.q[i] = q_new;
                } else {
                    self.q[i] = self.linear(i, sign as f64);
                }
                self.n[i] += sign;
            }
        }
    }

    /// Parabolic interpolation (P² formula).
    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qm = self.q[i - 1];
        let qp = self.q[i + 1];
        let ni = self.n[i] as f64;
        let nm = self.n[i - 1] as f64;
        let np = self.n[i + 1] as f64;
        qi + d / (np - nm)
            * ((ni - nm + d) * (qp - qi) / (np - ni)
                + (np - ni - d) * (qi - qm) / (ni - nm))
    }

    /// Linear interpolation fallback.
    fn linear(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let idx = if d > 0.0 { i + 1 } else { i - 1 };
        let qother = self.q[idx];
        let ni = self.n[i] as f64;
        let nother = self.n[idx] as f64;
        qi + d * (qother - qi) / (nother - ni)
    }

    /// Returns the current percentile estimate, or `None` before 5 observations.
    #[must_use]
    pub fn estimate(&self) -> Option<f64> {
        if self.count < 5 {
            None
        } else {
            Some(self.q[2])
        }
    }

    /// Returns the number of observations processed.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the target quantile.
    #[must_use]
    pub fn quantile(&self) -> f64 {
        self.p
    }
}

// ─── BitrateRunningAnalyzer ───────────────────────────────────────────────────

/// A composite analyzer that tracks per-frame bitrate statistics across a
/// single encode pass using all available online estimators.
///
/// This is the primary integration point between `running_stats` and the
/// `bitrate_estimator` module.  Feed it the bit count for each encoded frame;
/// query aggregate statistics at any time without buffering frame data.
///
/// # Example
///
/// ```rust
/// use oximedia_transcode::running_stats::BitrateRunningAnalyzer;
///
/// let mut analyzer = BitrateRunningAnalyzer::new(30.0, 60);
/// for bits in [40_000u64, 50_000, 60_000, 45_000, 55_000] {
///     analyzer.push_frame(bits);
/// }
/// let summary = analyzer.summary();
/// assert!(summary.mean_bps > 0.0);
/// assert!(summary.peak_bps >= summary.mean_bps);
/// ```
#[derive(Debug)]
pub struct BitrateRunningAnalyzer {
    /// Frame rate (fps) used to convert per-frame bits to bps.
    fps: f64,
    /// Welford stats over all frames (bits per frame).
    global: RunningStats,
    /// EWMA for bitrate trend.
    trend: Ewma,
    /// Rolling window for recent-peak detection.
    window: RollingWindow,
    /// P95 estimator over per-frame bits.
    p95: PercentileEstimator,
    /// Total bits accumulated.
    total_bits: u64,
    /// Total frames pushed.
    frame_count: u64,
}

impl BitrateRunningAnalyzer {
    /// Creates a new analyzer.
    ///
    /// * `fps` – The frame rate of the encoded stream.
    /// * `window_frames` – Size of the rolling window for recent-peak analysis.
    #[must_use]
    pub fn new(fps: f64, window_frames: usize) -> Self {
        let window_frames = window_frames.max(1);
        Self {
            fps,
            global: RunningStats::new(),
            trend: Ewma::new(0.1),
            window: RollingWindow::new(window_frames),
            p95: PercentileEstimator::new(0.95),
            total_bits: 0,
            frame_count: 0,
        }
    }

    /// Feeds the bit count for one encoded frame into all estimators.
    pub fn push_frame(&mut self, bits_per_frame: u64) {
        let bits_f = bits_per_frame as f64;
        self.global.push(bits_f);
        self.trend.update(bits_f);
        self.window.push(bits_f);
        self.p95.update(bits_f);
        self.total_bits += bits_per_frame;
        self.frame_count += 1;
    }

    /// Returns a point-in-time [`BitrateSummary`] from all running estimators.
    #[must_use]
    pub fn summary(&self) -> BitrateSummary {
        let scale = self.fps; // bits/frame → bits/s
        BitrateSummary {
            frame_count: self.frame_count,
            total_bits: self.total_bits,
            mean_bps: self.global.mean() * scale,
            stddev_bps: self.global.stddev() * scale,
            peak_bps: self.global.max() * scale,
            min_bps: if self.global.is_empty() {
                0.0
            } else {
                self.global.min() * scale
            },
            trend_bps: self.trend.value() * scale,
            window_mean_bps: self.window.mean() * scale,
            window_stddev_bps: self.window.stddev() * scale,
            p95_bps: self.p95.estimate().map(|v| v * scale),
            cv: self.global.cv(),
        }
    }

    /// Resets all accumulators.
    pub fn reset(&mut self) {
        self.global.reset();
        self.trend.reset();
        self.window = RollingWindow::new(self.window.capacity());
        self.p95 = PercentileEstimator::new(0.95);
        self.total_bits = 0;
        self.frame_count = 0;
    }
}

/// Point-in-time bitrate statistics produced by [`BitrateRunningAnalyzer`].
#[derive(Debug, Clone)]
pub struct BitrateSummary {
    /// Number of frames analysed so far.
    pub frame_count: u64,
    /// Total bits counted across all frames.
    pub total_bits: u64,
    /// Mean bits-per-second (over all frames).
    pub mean_bps: f64,
    /// Standard deviation of bits-per-second.
    pub stddev_bps: f64,
    /// Maximum per-frame bitrate seen.
    pub peak_bps: f64,
    /// Minimum per-frame bitrate seen.
    pub min_bps: f64,
    /// EWMA-smoothed bitrate trend (recent emphasis).
    pub trend_bps: f64,
    /// Mean bitrate in the recent rolling window.
    pub window_mean_bps: f64,
    /// Stddev of bitrate in the recent rolling window.
    pub window_stddev_bps: f64,
    /// P95 per-frame bitrate estimate (None before 5 frames).
    pub p95_bps: Option<f64>,
    /// Coefficient of variation (stddev / mean); low = stable bitrate.
    pub cv: f64,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RunningStats ─────────────────────────────────────────────────────────

    #[test]
    fn test_running_stats_empty() {
        let stats = RunningStats::new();
        assert!(stats.is_empty());
        assert_eq!(stats.count(), 0);
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.stddev(), 0.0);
    }

    #[test]
    fn test_running_stats_single_sample() {
        let mut stats = RunningStats::new();
        stats.push(42.0);
        assert_eq!(stats.count(), 1);
        assert!((stats.mean() - 42.0).abs() < 1e-10);
        // Sample variance undefined with 1 sample.
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.min(), 42.0);
        assert_eq!(stats.max(), 42.0);
    }

    #[test]
    fn test_running_stats_known_values() {
        let mut stats = RunningStats::new();
        for v in [2.0_f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.push(v);
        }
        // Mean = 5.0, population std = 2.0
        assert!((stats.mean() - 5.0).abs() < 1e-10, "mean {}", stats.mean());
        assert!((stats.population_stddev() - 2.0).abs() < 1e-10);
        assert_eq!(stats.min(), 2.0);
        assert_eq!(stats.max(), 9.0);
        assert_eq!(stats.range(), 7.0);
    }

    #[test]
    fn test_running_stats_nan_ignored() {
        let mut stats = RunningStats::new();
        stats.push(10.0);
        stats.push(f64::NAN);
        stats.push(20.0);
        assert_eq!(stats.count(), 2);
        assert!((stats.mean() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_stats_merge() {
        let mut a = RunningStats::new();
        let mut b = RunningStats::new();
        for v in [1.0_f64, 2.0, 3.0] {
            a.push(v);
        }
        for v in [4.0_f64, 5.0, 6.0] {
            b.push(v);
        }
        a.merge(&b);
        assert_eq!(a.count(), 6);
        assert!((a.mean() - 3.5).abs() < 1e-10, "merged mean {}", a.mean());
        assert_eq!(a.min(), 1.0);
        assert_eq!(a.max(), 6.0);
    }

    #[test]
    fn test_running_stats_merge_empty_rhs() {
        let mut a = RunningStats::new();
        a.push(5.0);
        let empty = RunningStats::new();
        a.merge(&empty);
        assert_eq!(a.count(), 1);
        assert!((a.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_stats_reset() {
        let mut stats = RunningStats::new();
        stats.push(100.0);
        stats.reset();
        assert!(stats.is_empty());
        assert_eq!(stats.mean(), 0.0);
    }

    #[test]
    fn test_running_stats_cv() {
        let mut stats = RunningStats::new();
        // All identical → stddev = 0 → cv = 0.
        for _ in 0..5 {
            stats.push(10.0);
        }
        let cv = stats.cv();
        assert!(cv.is_finite() && cv < 1e-10, "cv {cv}");
    }

    // ── EWMA ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_ewma_initial_value() {
        let mut ewma = Ewma::new(0.5);
        assert!(ewma.value_opt().is_none());
        ewma.update(100.0);
        assert!((ewma.value() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewma_convergence() {
        let mut ewma = Ewma::new(0.3);
        for _ in 0..200 {
            ewma.update(50.0);
        }
        // Should converge to 50.0 after enough updates.
        assert!((ewma.value() - 50.0).abs() < 0.1, "value {}", ewma.value());
    }

    #[test]
    fn test_ewma_nan_ignored() {
        let mut ewma = Ewma::new(0.5);
        ewma.update(10.0);
        let before = ewma.value();
        ewma.update(f64::NAN);
        assert!((ewma.value() - before).abs() < 1e-12);
        assert_eq!(ewma.count(), 1);
    }

    #[test]
    fn test_ewma_reset() {
        let mut ewma = Ewma::new(0.2);
        ewma.update(42.0);
        ewma.reset();
        assert!(ewma.value_opt().is_none());
        assert_eq!(ewma.count(), 0);
    }

    // ── RollingWindow ─────────────────────────────────────────────────────────

    #[test]
    fn test_rolling_window_basic() {
        let mut w = RollingWindow::new(3);
        assert!(w.is_empty());
        w.push(1.0);
        w.push(2.0);
        w.push(3.0);
        assert!(w.is_full());
        assert!((w.mean() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rolling_window_eviction() {
        let mut w = RollingWindow::new(3);
        w.push(10.0);
        w.push(20.0);
        w.push(30.0);
        // Evict 10.0, add 40.0 → window = [20, 30, 40]
        w.push(40.0);
        assert_eq!(w.len(), 3);
        assert!((w.mean() - 30.0).abs() < 1e-10, "mean {}", w.mean());
    }

    #[test]
    fn test_rolling_window_variance() {
        let mut w = RollingWindow::new(4);
        for v in [2.0_f64, 4.0, 4.0, 4.0] {
            w.push(v);
        }
        // population var = 0.75
        assert!((w.variance() - 0.75).abs() < 1e-10, "var {}", w.variance());
    }

    #[test]
    fn test_rolling_window_min_max() {
        let mut w = RollingWindow::new(5);
        for v in [5.0_f64, 3.0, 8.0, 1.0, 6.0] {
            w.push(v);
        }
        assert_eq!(w.min(), 1.0);
        assert_eq!(w.max(), 8.0);
    }

    // ── PercentileEstimator ───────────────────────────────────────────────────

    #[test]
    fn test_percentile_estimator_bootstrap() {
        let mut est = PercentileEstimator::new(0.5);
        for v in 1..=4 {
            est.update(v as f64);
            // Fewer than 5 samples → no estimate yet.
            assert!(est.estimate().is_none());
        }
        est.update(5.0);
        assert!(est.estimate().is_some());
    }

    #[test]
    fn test_percentile_estimator_median_uniform() {
        let mut est = PercentileEstimator::new(0.5);
        for v in 1..=1000 {
            est.update(v as f64);
        }
        // Median of 1..=1000 is ~500.5.
        let estimated = est.estimate().expect("should have estimate");
        assert!(
            (estimated - 500.5).abs() < 30.0,
            "median estimate {estimated}"
        );
    }

    #[test]
    fn test_percentile_estimator_p95() {
        let mut est = PercentileEstimator::new(0.95);
        // Insert 0..=99; p95 ≈ 94.05 (0-indexed).
        for v in 0..=99 {
            est.update(v as f64);
        }
        let estimated = est.estimate().expect("should have estimate");
        // Allow ±10 tolerance for small samples.
        assert!(
            (estimated - 94.05).abs() < 15.0,
            "p95 estimate {estimated}"
        );
    }

    // ── BitrateRunningAnalyzer ────────────────────────────────────────────────

    #[test]
    fn test_bitrate_analyzer_basic() {
        let mut analyzer = BitrateRunningAnalyzer::new(30.0, 30);
        for bits in [40_000u64, 50_000, 60_000, 45_000, 55_000] {
            analyzer.push_frame(bits);
        }
        let s = analyzer.summary();
        assert_eq!(s.frame_count, 5);
        assert!(s.mean_bps > 0.0);
        assert!(s.peak_bps >= s.mean_bps);
        assert!(s.min_bps <= s.mean_bps);
        assert_eq!(s.total_bits, 250_000);
    }

    #[test]
    fn test_bitrate_analyzer_reset() {
        let mut analyzer = BitrateRunningAnalyzer::new(25.0, 10);
        analyzer.push_frame(100_000);
        analyzer.reset();
        let s = analyzer.summary();
        assert_eq!(s.frame_count, 0);
        assert_eq!(s.total_bits, 0);
    }

    #[test]
    fn test_bitrate_analyzer_trend_smoother_than_peak() {
        let mut analyzer = BitrateRunningAnalyzer::new(30.0, 10);
        // Alternate between 10 000 and 200 000 bits/frame.
        for i in 0..100 {
            analyzer.push_frame(if i % 2 == 0 { 10_000 } else { 200_000 });
        }
        let s = analyzer.summary();
        // Trend (EWMA α=0.1) should be less volatile than peak.
        assert!(s.peak_bps > s.trend_bps);
    }
}
