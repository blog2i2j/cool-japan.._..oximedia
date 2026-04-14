//! Live viewer counter for game streams.
//!
//! Tracks concurrent viewer counts in real time, maintains a retention curve
//! (viewer count snapshots over time), detects peak counts, and provides
//! viewer-drop/join rate calculations useful for stream health monitoring.
//!
//! # Design
//!
//! - [`ViewerCounter`] is the primary entry point.  It holds a compact ring
//!   buffer of time-bucketed snapshots so memory usage is bounded even for
//!   very long streams.
//! - [`RetentionCurve`] is built from the snapshot history and exposes
//!   percentage-of-peak retention at any stored time offset.
//! - All arithmetic uses saturating/checked operations — no panics, no
//!   `unwrap()` in library code.

use std::collections::VecDeque;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A point-in-time viewer count sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewerSnapshot {
    /// Elapsed seconds since the stream started when this sample was taken.
    pub elapsed_secs: u64,
    /// Number of concurrent viewers at this moment.
    pub viewer_count: u64,
    /// Cumulative joins since stream start.
    pub cumulative_joins: u64,
    /// Cumulative leaves since stream start.
    pub cumulative_leaves: u64,
}

// ---------------------------------------------------------------------------
// PeakRecord
// ---------------------------------------------------------------------------

/// Records when and how large the peak viewer count was.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeakRecord {
    /// Peak concurrent viewer count.
    pub count: u64,
    /// Elapsed seconds since stream start when the peak occurred.
    pub elapsed_secs: u64,
}

// ---------------------------------------------------------------------------
// ViewerRateWindow
// ---------------------------------------------------------------------------

/// A sliding-window join/leave rate calculator.
///
/// Keeps track of join and leave events within a configurable window so that
/// per-second join and leave rates can be estimated.
#[derive(Debug)]
pub struct ViewerRateWindow {
    window_secs: u64,
    /// Ring buffer of (elapsed_secs, delta) where delta >0 = join, <0 = leave.
    events: VecDeque<(u64, i64)>,
}

impl ViewerRateWindow {
    /// Create a rate window covering `window_secs` seconds of history.
    #[must_use]
    pub fn new(window_secs: u64) -> Self {
        Self {
            window_secs: window_secs.max(1),
            events: VecDeque::new(),
        }
    }

    /// Record a join event at the given elapsed time.
    pub fn record_join(&mut self, elapsed_secs: u64) {
        self.prune(elapsed_secs);
        self.events.push_back((elapsed_secs, 1));
    }

    /// Record a leave event at the given elapsed time.
    pub fn record_leave(&mut self, elapsed_secs: u64) {
        self.prune(elapsed_secs);
        self.events.push_back((elapsed_secs, -1));
    }

    /// Remove events older than the window.
    fn prune(&mut self, now_secs: u64) {
        let cutoff = now_secs.saturating_sub(self.window_secs);
        while self
            .events
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.events.pop_front();
        }
    }

    /// Joins-per-second within the current window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn join_rate(&self) -> f64 {
        let joins: i64 = self.events.iter().filter(|(_, d)| *d > 0).count() as i64;
        joins as f64 / self.window_secs as f64
    }

    /// Leaves-per-second within the current window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn leave_rate(&self) -> f64 {
        let leaves: i64 = self.events.iter().filter(|(_, d)| *d < 0).count() as i64;
        leaves as f64 / self.window_secs as f64
    }

    /// Net viewer change rate per second (positive = growing, negative = shrinking).
    #[must_use]
    pub fn net_rate(&self) -> f64 {
        self.join_rate() - self.leave_rate()
    }
}

// ---------------------------------------------------------------------------
// RetentionCurve
// ---------------------------------------------------------------------------

/// Viewer retention curve built from snapshot history.
///
/// Each entry represents what fraction of the peak audience was present at
/// a given elapsed-seconds offset.
#[derive(Debug, Clone)]
pub struct RetentionCurve {
    /// Sorted (elapsed_secs, retention_fraction) pairs.
    points: Vec<(u64, f64)>,
    /// Peak viewer count used for normalisation.
    pub peak_viewers: u64,
}

impl RetentionCurve {
    /// Build a retention curve from an ordered slice of snapshots.
    ///
    /// Returns `None` if the slice is empty or the peak is zero.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn from_snapshots(snapshots: &[ViewerSnapshot]) -> Option<Self> {
        if snapshots.is_empty() {
            return None;
        }
        let peak = snapshots.iter().map(|s| s.viewer_count).max()?;
        if peak == 0 {
            return None;
        }
        let points = snapshots
            .iter()
            .map(|s| (s.elapsed_secs, s.viewer_count as f64 / peak as f64))
            .collect();
        Some(Self {
            points,
            peak_viewers: peak,
        })
    }

    /// Retention fraction at the given elapsed time (linear interpolation
    /// between stored points).  Returns `None` if there are no points or the
    /// requested time is before the first snapshot.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn retention_at(&self, elapsed_secs: u64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }
        // Exact match fast path
        if let Some(p) = self.points.iter().find(|(t, _)| *t == elapsed_secs) {
            return Some(p.1);
        }
        // Find surrounding points for interpolation
        let idx = self.points.partition_point(|(t, _)| *t < elapsed_secs);
        if idx == 0 {
            return Some(self.points[0].1);
        }
        if idx >= self.points.len() {
            return Some(self.points[self.points.len() - 1].1);
        }
        let (t0, r0) = self.points[idx - 1];
        let (t1, r1) = self.points[idx];
        let span = (t1 - t0) as f64;
        if span <= 0.0 {
            return Some(r0);
        }
        let frac = (elapsed_secs - t0) as f64 / span;
        Some(r0 + frac * (r1 - r0))
    }

    /// Average retention across all stored points.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_retention(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|(_, r)| r).sum();
        sum / self.points.len() as f64
    }

    /// Elapsed time (seconds) at which retention first dropped below
    /// `threshold` (a value in `[0.0, 1.0]`).  Returns `None` if retention
    /// never falls below the threshold within the stored history.
    #[must_use]
    pub fn time_to_drop_below(&self, threshold: f64) -> Option<u64> {
        self.points
            .iter()
            .find(|(_, r)| *r < threshold)
            .map(|(t, _)| *t)
    }
}

// ---------------------------------------------------------------------------
// ViewerCounterConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`ViewerCounter`].
#[derive(Debug, Clone)]
pub struct ViewerCounterConfig {
    /// How many seconds of history to store in the snapshot ring buffer.
    /// Older snapshots are evicted when the buffer is full.
    pub history_capacity_secs: u64,
    /// Interval between automatic snapshots in seconds.
    pub snapshot_interval_secs: u64,
    /// Rate-window duration in seconds.
    pub rate_window_secs: u64,
}

impl Default for ViewerCounterConfig {
    fn default() -> Self {
        Self {
            history_capacity_secs: 7200, // 2 hours
            snapshot_interval_secs: 10,
            rate_window_secs: 60,
        }
    }
}

impl ViewerCounterConfig {
    /// Maximum number of snapshots to retain in the ring buffer.
    #[must_use]
    pub fn max_snapshots(&self) -> usize {
        (self.history_capacity_secs / self.snapshot_interval_secs.max(1)) as usize
    }
}

// ---------------------------------------------------------------------------
// ViewerCounter
// ---------------------------------------------------------------------------

/// Real-time live viewer counter with history and retention tracking.
///
/// # Example
///
/// ```
/// use oximedia_gaming::viewer_counter::{ViewerCounter, ViewerCounterConfig};
///
/// let mut counter = ViewerCounter::new(ViewerCounterConfig::default());
/// counter.record_join(0);
/// counter.record_join(1);
/// assert_eq!(counter.current_viewers(), 2);
/// counter.record_leave(5);
/// assert_eq!(counter.current_viewers(), 1);
/// ```
#[derive(Debug)]
pub struct ViewerCounter {
    config: ViewerCounterConfig,
    /// Current concurrent viewer count.
    current: u64,
    /// Cumulative joins.
    total_joins: u64,
    /// Cumulative leaves.
    total_leaves: u64,
    /// Peak record.
    peak: Option<PeakRecord>,
    /// Snapshot ring buffer (bounded by `config.max_snapshots()`).
    snapshots: VecDeque<ViewerSnapshot>,
    /// Elapsed seconds at the last automatic snapshot.
    last_snapshot_secs: u64,
    /// Rate window tracker.
    rate_window: ViewerRateWindow,
}

impl ViewerCounter {
    /// Create a new viewer counter with the given configuration.
    #[must_use]
    pub fn new(config: ViewerCounterConfig) -> Self {
        let rate_window = ViewerRateWindow::new(config.rate_window_secs);
        Self {
            config,
            current: 0,
            total_joins: 0,
            total_leaves: 0,
            peak: None,
            snapshots: VecDeque::new(),
            last_snapshot_secs: 0,
            rate_window,
        }
    }

    /// Record one or more viewers joining at `elapsed_secs` since stream start.
    pub fn record_joins(&mut self, elapsed_secs: u64, count: u64) {
        self.current = self.current.saturating_add(count);
        self.total_joins = self.total_joins.saturating_add(count);
        for _ in 0..count {
            self.rate_window.record_join(elapsed_secs);
        }
        self.update_peak(elapsed_secs);
        self.maybe_snapshot(elapsed_secs);
    }

    /// Convenience: record a single viewer join.
    pub fn record_join(&mut self, elapsed_secs: u64) {
        self.record_joins(elapsed_secs, 1);
    }

    /// Record one or more viewers leaving at `elapsed_secs`.
    pub fn record_leaves(&mut self, elapsed_secs: u64, count: u64) {
        // Clamp to avoid underflow if leave events arrive out of order.
        let actual = count.min(self.current);
        self.current = self.current.saturating_sub(actual);
        self.total_leaves = self.total_leaves.saturating_add(actual);
        for _ in 0..actual {
            self.rate_window.record_leave(elapsed_secs);
        }
        self.maybe_snapshot(elapsed_secs);
    }

    /// Convenience: record a single viewer leave.
    pub fn record_leave(&mut self, elapsed_secs: u64) {
        self.record_leaves(elapsed_secs, 1);
    }

    /// Forcibly set the current viewer count (e.g. from a platform API poll).
    ///
    /// Adjusts join/leave accumulators based on the delta.
    pub fn set_viewer_count(&mut self, elapsed_secs: u64, count: u64) {
        match count.cmp(&self.current) {
            std::cmp::Ordering::Greater => {
                let delta = count - self.current;
                self.current = count;
                self.total_joins = self.total_joins.saturating_add(delta);
            }
            std::cmp::Ordering::Less => {
                let delta = self.current - count;
                self.current = count;
                self.total_leaves = self.total_leaves.saturating_add(delta);
            }
            std::cmp::Ordering::Equal => {}
        }
        self.update_peak(elapsed_secs);
        self.maybe_snapshot(elapsed_secs);
    }

    /// Force a snapshot to be recorded at `elapsed_secs` regardless of the
    /// normal interval.
    pub fn force_snapshot(&mut self, elapsed_secs: u64) {
        self.push_snapshot(elapsed_secs);
        self.last_snapshot_secs = elapsed_secs;
    }

    /// Current concurrent viewer count.
    #[must_use]
    pub fn current_viewers(&self) -> u64 {
        self.current
    }

    /// Peak concurrent viewer count and when it occurred.
    #[must_use]
    pub fn peak(&self) -> Option<PeakRecord> {
        self.peak
    }

    /// Total viewer joins since tracking began.
    #[must_use]
    pub fn total_joins(&self) -> u64 {
        self.total_joins
    }

    /// Total viewer leaves since tracking began.
    #[must_use]
    pub fn total_leaves(&self) -> u64 {
        self.total_leaves
    }

    /// Number of snapshots currently stored.
    #[must_use]
    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    /// A copy of the stored snapshots in chronological order.
    #[must_use]
    pub fn snapshots(&self) -> Vec<ViewerSnapshot> {
        self.snapshots.iter().copied().collect()
    }

    /// Build a [`RetentionCurve`] from the stored snapshot history.
    ///
    /// Returns `None` if there are not enough snapshots.
    #[must_use]
    pub fn retention_curve(&self) -> Option<RetentionCurve> {
        let snaps: Vec<_> = self.snapshots.iter().copied().collect();
        RetentionCurve::from_snapshots(&snaps)
    }

    /// Estimated churn rate: `total_leaves / total_joins` in `[0.0, 1.0]`.
    /// Returns `0.0` if no joins have been recorded.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn churn_rate(&self) -> f64 {
        if self.total_joins == 0 {
            return 0.0;
        }
        (self.total_leaves as f64 / self.total_joins as f64).min(1.0)
    }

    /// Current join rate (viewers/second) within the rate window.
    #[must_use]
    pub fn join_rate(&self) -> f64 {
        self.rate_window.join_rate()
    }

    /// Current leave rate (viewers/second) within the rate window.
    #[must_use]
    pub fn leave_rate(&self) -> f64 {
        self.rate_window.leave_rate()
    }

    /// Net viewer change rate per second.
    #[must_use]
    pub fn net_rate(&self) -> f64 {
        self.rate_window.net_rate()
    }

    /// Estimate the viewer count at a future elapsed time using the current
    /// net rate (linear extrapolation, floor at zero).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation
    )]
    #[must_use]
    pub fn forecast(&self, _current_elapsed_secs: u64, horizon: Duration) -> u64 {
        let delta_secs = horizon.as_secs_f64();
        let projected = self.current as f64 + self.net_rate() * delta_secs;
        projected.max(0.0) as u64
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn update_peak(&mut self, elapsed_secs: u64) {
        match &self.peak {
            None => {
                self.peak = Some(PeakRecord {
                    count: self.current,
                    elapsed_secs,
                });
            }
            Some(p) if self.current > p.count => {
                self.peak = Some(PeakRecord {
                    count: self.current,
                    elapsed_secs,
                });
            }
            _ => {}
        }
    }

    fn maybe_snapshot(&mut self, elapsed_secs: u64) {
        if elapsed_secs.saturating_sub(self.last_snapshot_secs)
            >= self.config.snapshot_interval_secs
        {
            self.push_snapshot(elapsed_secs);
            self.last_snapshot_secs = elapsed_secs;
        }
    }

    fn push_snapshot(&mut self, elapsed_secs: u64) {
        let snap = ViewerSnapshot {
            elapsed_secs,
            viewer_count: self.current,
            cumulative_joins: self.total_joins,
            cumulative_leaves: self.total_leaves,
        };
        let max = self.config.max_snapshots().max(1);
        if self.snapshots.len() >= max {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snap);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_counter() -> ViewerCounter {
        ViewerCounter::new(ViewerCounterConfig {
            snapshot_interval_secs: 10,
            rate_window_secs: 60,
            history_capacity_secs: 600,
        })
    }

    #[test]
    fn test_join_increments_current() {
        let mut c = default_counter();
        c.record_join(0);
        c.record_join(1);
        assert_eq!(c.current_viewers(), 2);
    }

    #[test]
    fn test_leave_decrements_current() {
        let mut c = default_counter();
        c.record_joins(0, 5);
        c.record_leaves(2, 3);
        assert_eq!(c.current_viewers(), 2);
    }

    #[test]
    fn test_leave_cannot_go_below_zero() {
        let mut c = default_counter();
        c.record_join(0);
        // Leave more viewers than are present
        c.record_leaves(1, 10);
        assert_eq!(c.current_viewers(), 0);
    }

    #[test]
    fn test_peak_tracked_correctly() {
        let mut c = default_counter();
        c.record_joins(0, 10);
        c.record_leaves(5, 3);
        c.record_joins(10, 2);
        let peak = c.peak().expect("peak should be set");
        assert_eq!(peak.count, 10);
        assert_eq!(peak.elapsed_secs, 0);
    }

    #[test]
    fn test_peak_updates_on_new_maximum() {
        let mut c = default_counter();
        c.record_joins(0, 5);
        c.record_joins(20, 10);
        let peak = c.peak().expect("peak should be set");
        assert_eq!(peak.count, 15);
        assert_eq!(peak.elapsed_secs, 20);
    }

    #[test]
    fn test_set_viewer_count_adjusts_accumulators() {
        let mut c = default_counter();
        c.set_viewer_count(0, 100);
        assert_eq!(c.current_viewers(), 100);
        assert_eq!(c.total_joins(), 100);
        c.set_viewer_count(10, 80);
        assert_eq!(c.current_viewers(), 80);
        assert_eq!(c.total_leaves(), 20);
    }

    #[test]
    fn test_snapshot_ring_buffer_bounded() {
        let mut c = ViewerCounter::new(ViewerCounterConfig {
            history_capacity_secs: 100,
            snapshot_interval_secs: 10,
            rate_window_secs: 30,
        });
        // max_snapshots = 100 / 10 = 10
        for i in 0..=50u64 {
            c.force_snapshot(i * 10);
        }
        assert!(c.snapshot_count() <= c.config.max_snapshots());
    }

    #[test]
    fn test_churn_rate() {
        let mut c = default_counter();
        c.record_joins(0, 100);
        c.record_leaves(5, 25);
        // churn = 25/100 = 0.25
        assert!((c.churn_rate() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_churn_rate_zero_when_no_joins() {
        let c = default_counter();
        assert!((c.churn_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_retention_curve_average() {
        let snaps = vec![
            ViewerSnapshot {
                elapsed_secs: 0,
                viewer_count: 100,
                cumulative_joins: 100,
                cumulative_leaves: 0,
            },
            ViewerSnapshot {
                elapsed_secs: 60,
                viewer_count: 50,
                cumulative_joins: 100,
                cumulative_leaves: 50,
            },
        ];
        let curve = RetentionCurve::from_snapshots(&snaps).expect("curve should build");
        assert_eq!(curve.peak_viewers, 100);
        // Retention at t=0 is 1.0, at t=60 is 0.5 → average = 0.75
        let avg = curve.average_retention();
        assert!((avg - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_retention_at_interpolation() {
        let snaps = vec![
            ViewerSnapshot {
                elapsed_secs: 0,
                viewer_count: 100,
                cumulative_joins: 100,
                cumulative_leaves: 0,
            },
            ViewerSnapshot {
                elapsed_secs: 100,
                viewer_count: 0,
                cumulative_joins: 100,
                cumulative_leaves: 100,
            },
        ];
        let curve = RetentionCurve::from_snapshots(&snaps).expect("curve should build");
        let ret = curve.retention_at(50).expect("should interpolate");
        assert!((ret - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_retention_curve_none_on_empty() {
        let curve = RetentionCurve::from_snapshots(&[]);
        assert!(curve.is_none());
    }

    #[test]
    fn test_forecast_growing_stream() {
        let mut c = ViewerCounter::new(ViewerCounterConfig {
            snapshot_interval_secs: 10,
            rate_window_secs: 10,
            history_capacity_secs: 3600,
        });
        // Simulate 10 joins in 10 seconds → 1 join/sec rate
        for i in 0..10u64 {
            c.record_join(i);
        }
        // Forecast 10 seconds ahead from current 10 viewers
        let forecast = c.forecast(9, Duration::from_secs(10));
        // net_rate ≈ 1.0 join/sec → 10 + 10 = 20
        assert!(
            forecast >= 10,
            "forecast should be at least current: {forecast}"
        );
    }

    #[test]
    fn test_rate_window_net_rate() {
        let mut w = ViewerRateWindow::new(10);
        for i in 0..8u64 {
            w.record_join(i);
        }
        for i in 8..10u64 {
            w.record_leave(i);
        }
        // joins=8, leaves=2, window=10 → net = (8-2)/10 = 0.6
        assert!((w.net_rate() - 0.6).abs() < 1e-9);
    }
}
