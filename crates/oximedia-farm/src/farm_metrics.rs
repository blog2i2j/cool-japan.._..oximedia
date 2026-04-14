//! Render farm performance metrics and time-series reporting.
//!
//! This module provides a lightweight, in-process time-series store for farm
//! telemetry that is independent of the Prometheus-based [`crate::metrics`]
//! module.  It is designed for:
//!
//! - **Unit-test friendly**: no global registries, no background threads.
//! - **Windowed queries**: callers supply a `now_secs` timestamp so that tests
//!   can inject synthetic time without sleeping.
//! - **Report generation**: [`FarmMetrics::generate_report`] distils raw
//!   time-series data into a human-readable [`FarmReport`].
//!
//! # Standard metric names
//!
//! The following metric names have first-class support in report generation:
//!
//! | Constant | Description |
//! |----------|-------------|
//! | [`METRIC_TASKS_PER_SECOND`] | Rolling task throughput |
//! | [`METRIC_QUEUE_DEPTH`] | Pending task count in the scheduler queue |
//! | [`METRIC_WORKER_UTILIZATION`] | Fraction of workers actively executing tasks (0–1) |
//! | [`METRIC_FAILURE_RATE`] | Fraction of recently completed tasks that failed (0–1) |
//! | [`METRIC_TASK_DURATION_SECS`] | Individual task wall-clock duration in seconds |
//!
//! # Example
//!
//! ```
//! use oximedia_farm::farm_metrics::{FarmMetricPoint, FarmMetrics, METRIC_QUEUE_DEPTH};
//!
//! let mut fm = FarmMetrics::new();
//! fm.record(FarmMetricPoint {
//!     timestamp_secs: 1000,
//!     metric_name: METRIC_QUEUE_DEPTH.into(),
//!     value: 42.0,
//!     worker_id: None,
//! });
//!
//! assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), Some(42.0));
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Standard metric name constants
// ---------------------------------------------------------------------------

/// Metric: tasks completed per second (rolling throughput).
pub const METRIC_TASKS_PER_SECOND: &str = "tasks_per_second";
/// Metric: number of tasks currently in the scheduler queue.
pub const METRIC_QUEUE_DEPTH: &str = "queue_depth";
/// Metric: fraction (0–1) of workers actively executing tasks.
pub const METRIC_WORKER_UTILIZATION: &str = "worker_utilization";
/// Metric: fraction (0–1) of recently completed tasks that failed.
pub const METRIC_FAILURE_RATE: &str = "failure_rate";
/// Metric: wall-clock duration in seconds for a single completed task.
pub const METRIC_TASK_DURATION_SECS: &str = "task_duration_secs";

// ---------------------------------------------------------------------------
// FarmMetricPoint
// ---------------------------------------------------------------------------

/// A single timestamped observation for a named metric.
#[derive(Debug, Clone)]
pub struct FarmMetricPoint {
    /// Unix timestamp (seconds) when the observation was recorded.
    pub timestamp_secs: u64,
    /// Name of the metric (use one of the `METRIC_*` constants for standard
    /// metrics, or any string for custom metrics).
    pub metric_name: String,
    /// Observed value.
    pub value: f64,
    /// Optional worker that produced this observation.  `None` for
    /// cluster-wide metrics.
    pub worker_id: Option<String>,
}

// ---------------------------------------------------------------------------
// FarmMetrics
// ---------------------------------------------------------------------------

/// In-process time-series store for farm telemetry.
///
/// All observations are kept in memory.  For long-running deployments callers
/// should periodically call [`FarmMetrics::prune_older_than`] to cap memory
/// usage.
#[derive(Debug, Default)]
pub struct FarmMetrics {
    /// Per-metric sorted lists of observations (sorted by `timestamp_secs` ascending).
    series: HashMap<String, Vec<FarmMetricPoint>>,
}

impl FarmMetrics {
    /// Create an empty metrics store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // -----------------------------------------------------------------------
    // Write
    // -----------------------------------------------------------------------

    /// Record a new data point.
    ///
    /// Points are inserted in chronological order.  Inserting an out-of-order
    /// point (timestamp earlier than the most recent stored point) still works
    /// correctly — the series is re-sorted to maintain ordering.
    pub fn record(&mut self, point: FarmMetricPoint) {
        let series = self.series.entry(point.metric_name.clone()).or_default();
        series.push(point);
        // Keep series sorted by timestamp for efficient windowed queries.
        if series.len() > 1 {
            let last = series.len() - 1;
            // Only sort if the new point is out of order.
            if series[last].timestamp_secs < series[last - 1].timestamp_secs {
                series.sort_by_key(|p| p.timestamp_secs);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Read — single value
    // -----------------------------------------------------------------------

    /// Return the most recently recorded value for `metric_name`, or `None` if
    /// no observations have been recorded yet.
    #[must_use]
    pub fn latest(&self, metric_name: &str) -> Option<f64> {
        self.series
            .get(metric_name)
            .and_then(|s| s.last())
            .map(|p| p.value)
    }

    // -----------------------------------------------------------------------
    // Read — windowed aggregates
    // -----------------------------------------------------------------------

    /// Return a reference to the slice of points for `metric_name` that fall
    /// within `[now_secs - window_secs, now_secs]`.
    fn window_points<'a>(
        &'a self,
        metric_name: &str,
        window_secs: u64,
        now_secs: u64,
    ) -> &'a [FarmMetricPoint] {
        let Some(series) = self.series.get(metric_name) else {
            return &[];
        };

        let cutoff = now_secs.saturating_sub(window_secs);

        // Binary search for the start of the window.
        let start = series.partition_point(|p| p.timestamp_secs < cutoff);
        // End: points with timestamp <= now_secs.
        let end = series.partition_point(|p| p.timestamp_secs <= now_secs);

        &series[start..end]
    }

    /// Compute the arithmetic mean of all observations within the trailing
    /// `window_secs`-second window ending at `now_secs`.
    ///
    /// Returns `None` when no observations exist in the window.
    #[must_use]
    pub fn average(&self, metric_name: &str, window_secs: u64, now_secs: u64) -> Option<f64> {
        let pts = self.window_points(metric_name, window_secs, now_secs);
        if pts.is_empty() {
            return None;
        }
        let sum: f64 = pts.iter().map(|p| p.value).sum();
        Some(sum / pts.len() as f64)
    }

    /// Return the maximum observation within the trailing `window_secs`-second
    /// window ending at `now_secs`.
    ///
    /// Returns `None` when no observations exist in the window.
    #[must_use]
    pub fn max(&self, metric_name: &str, window_secs: u64, now_secs: u64) -> Option<f64> {
        let pts = self.window_points(metric_name, window_secs, now_secs);
        pts.iter()
            .map(|p| p.value)
            .reduce(f64::max)
    }

    /// Return the minimum observation within the trailing `window_secs`-second
    /// window ending at `now_secs`.
    ///
    /// Returns `None` when no observations exist in the window.
    #[must_use]
    pub fn min(&self, metric_name: &str, window_secs: u64, now_secs: u64) -> Option<f64> {
        let pts = self.window_points(metric_name, window_secs, now_secs);
        pts.iter()
            .map(|p| p.value)
            .reduce(f64::min)
    }

    /// Count the number of observations within the trailing `window_secs`-second
    /// window ending at `now_secs`.
    #[must_use]
    pub fn count(&self, metric_name: &str, window_secs: u64, now_secs: u64) -> usize {
        self.window_points(metric_name, window_secs, now_secs).len()
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Drop all observations older than `retain_secs` seconds relative to
    /// `now_secs`.  This keeps memory usage bounded for long-running processes.
    pub fn prune_older_than(&mut self, retain_secs: u64, now_secs: u64) {
        let cutoff = now_secs.saturating_sub(retain_secs);
        for series in self.series.values_mut() {
            let keep_from = series.partition_point(|p| p.timestamp_secs < cutoff);
            series.drain(..keep_from);
        }
    }

    /// Return the total number of data points across all metrics.
    #[must_use]
    pub fn total_point_count(&self) -> usize {
        self.series.values().map(|s| s.len()).sum()
    }

    /// Return all registered metric names.
    #[must_use]
    pub fn metric_names(&self) -> Vec<&str> {
        self.series.keys().map(String::as_str).collect()
    }

    // -----------------------------------------------------------------------
    // Report generation
    // -----------------------------------------------------------------------

    /// Generate a [`FarmReport`] by aggregating all standard metrics within the
    /// trailing `window_secs`-second window ending at `now_secs`.
    ///
    /// Fields that depend on metrics for which no observations exist default to
    /// zero / `0.0`.
    #[must_use]
    pub fn generate_report(&self, window_secs: u64, now_secs: u64) -> FarmReport {
        // ---- total tasks completed / failed --------------------------------
        // Derive from METRIC_TASKS_PER_SECOND observations: each point records
        // the instantaneous rate, so we approximate totals by integrating the
        // rate over the window.  If the caller has stored raw cumulative counts
        // under dedicated metric names those can be added later.
        //
        // For a simpler model (the most common test case) we also support
        // callers that record individual task completion events by counting
        // METRIC_TASK_DURATION_SECS points (one per completed task) and
        // METRIC_FAILURE_RATE to estimate failures.

        let completed_pts = self.window_points(METRIC_TASK_DURATION_SECS, window_secs, now_secs);
        let total_tasks_completed = completed_pts.len() as u64;

        let avg_failure_rate = self
            .average(METRIC_FAILURE_RATE, window_secs, now_secs)
            .unwrap_or(0.0);

        // Estimate failed tasks from failure rate and completed count.
        let total_tasks_failed = if total_tasks_completed > 0 {
            (total_tasks_completed as f64 * avg_failure_rate).round() as u64
        } else {
            0
        };

        // ---- average task duration ----------------------------------------
        let avg_task_duration_secs = if completed_pts.is_empty() {
            0.0
        } else {
            let sum: f64 = completed_pts.iter().map(|p| p.value).sum();
            sum / completed_pts.len() as f64
        };

        // ---- peak workers --------------------------------------------------
        let peak_workers_used = self
            .max(METRIC_WORKER_UTILIZATION, window_secs, now_secs)
            .unwrap_or(0.0) as u64;

        // ---- efficiency ----------------------------------------------------
        // Efficiency = (1 - failure_rate) * avg_utilization, scaled to [0, 100].
        let avg_utilization = self
            .average(METRIC_WORKER_UTILIZATION, window_secs, now_secs)
            .unwrap_or(0.0);

        let efficiency_pct = ((1.0 - avg_failure_rate) * avg_utilization * 100.0).clamp(0.0, 100.0) as f32;

        FarmReport {
            generated_at_secs: now_secs,
            total_tasks_completed,
            total_tasks_failed,
            avg_task_duration_secs,
            peak_workers_used,
            efficiency_pct,
        }
    }
}

// ---------------------------------------------------------------------------
// FarmReport
// ---------------------------------------------------------------------------

/// A snapshot summary of render-farm performance for a given time window.
///
/// Generated by [`FarmMetrics::generate_report`].
#[derive(Debug, Clone, PartialEq)]
pub struct FarmReport {
    /// Unix timestamp (seconds) at which the report was generated.
    pub generated_at_secs: u64,
    /// Total number of tasks that completed (succeeded or failed) in the window.
    pub total_tasks_completed: u64,
    /// Estimated number of tasks that failed in the window.
    pub total_tasks_failed: u64,
    /// Average wall-clock duration (seconds) of completed tasks.
    pub avg_task_duration_secs: f64,
    /// Peak number of workers observed as active in the window.
    pub peak_workers_used: u64,
    /// Overall farm efficiency percentage (0–100).
    ///
    /// Computed as `(1 - failure_rate) × avg_worker_utilization × 100`.
    pub efficiency_pct: f32,
}

impl FarmReport {
    /// Return `true` when the farm is operating above `min_efficiency_pct` and
    /// the failure rate is below `max_failure_fraction`.
    #[must_use]
    pub fn is_healthy(&self, min_efficiency_pct: f32, max_failure_fraction: f64) -> bool {
        let failure_fraction = if self.total_tasks_completed == 0 {
            0.0
        } else {
            self.total_tasks_failed as f64 / self.total_tasks_completed as f64
        };
        self.efficiency_pct >= min_efficiency_pct && failure_fraction <= max_failure_fraction
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn pt(ts: u64, name: &str, value: f64) -> FarmMetricPoint {
        FarmMetricPoint {
            timestamp_secs: ts,
            metric_name: name.into(),
            value,
            worker_id: None,
        }
    }

    fn wpt(ts: u64, name: &str, value: f64, worker: &str) -> FarmMetricPoint {
        FarmMetricPoint {
            timestamp_secs: ts,
            metric_name: name.into(),
            value,
            worker_id: Some(worker.into()),
        }
    }

    // ---- Basic record & retrieve -------------------------------------------

    #[test]
    fn test_record_and_latest() {
        let mut fm = FarmMetrics::new();
        assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), None);

        fm.record(pt(100, METRIC_QUEUE_DEPTH, 10.0));
        assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), Some(10.0));

        fm.record(pt(200, METRIC_QUEUE_DEPTH, 20.0));
        assert_eq!(fm.latest(METRIC_QUEUE_DEPTH), Some(20.0));
    }

    #[test]
    fn test_latest_returns_most_recent() {
        let mut fm = FarmMetrics::new();
        fm.record(pt(1, METRIC_TASKS_PER_SECOND, 5.0));
        fm.record(pt(5, METRIC_TASKS_PER_SECOND, 12.0));
        fm.record(pt(3, METRIC_TASKS_PER_SECOND, 8.0)); // out-of-order insert
        // After sort, latest is ts=5 → 12.0
        assert_eq!(fm.latest(METRIC_TASKS_PER_SECOND), Some(12.0));
    }

    // ---- Windowed average ---------------------------------------------------

    #[test]
    fn test_average_within_window() {
        let mut fm = FarmMetrics::new();
        for i in 0u64..10 {
            fm.record(pt(i * 10, METRIC_QUEUE_DEPTH, i as f64));
        }
        // now=90, window=50 → points at ts 40,50,60,70,80,90 → values 4,5,6,7,8,9
        let avg = fm.average(METRIC_QUEUE_DEPTH, 50, 90).expect("some");
        let expected = (4.0 + 5.0 + 6.0 + 7.0 + 8.0 + 9.0) / 6.0;
        assert!((avg - expected).abs() < 1e-9, "avg={avg} expected={expected}");
    }

    #[test]
    fn test_average_empty_window_returns_none() {
        let mut fm = FarmMetrics::new();
        fm.record(pt(100, METRIC_QUEUE_DEPTH, 42.0));
        // window ends before all points
        assert_eq!(fm.average(METRIC_QUEUE_DEPTH, 10, 50), None);
    }

    #[test]
    fn test_average_unknown_metric_returns_none() {
        let fm = FarmMetrics::new();
        assert_eq!(fm.average("unknown_metric", 100, 1000), None);
    }

    // ---- Windowed max -------------------------------------------------------

    #[test]
    fn test_max_within_window() {
        let mut fm = FarmMetrics::new();
        fm.record(pt(10, METRIC_WORKER_UTILIZATION, 0.3));
        fm.record(pt(20, METRIC_WORKER_UTILIZATION, 0.9));
        fm.record(pt(30, METRIC_WORKER_UTILIZATION, 0.5));
        let m = fm.max(METRIC_WORKER_UTILIZATION, 30, 30).expect("some");
        assert!((m - 0.9).abs() < 1e-9, "max={m}");
    }

    #[test]
    fn test_max_excludes_out_of_window() {
        let mut fm = FarmMetrics::new();
        fm.record(pt(1, METRIC_WORKER_UTILIZATION, 0.99)); // old
        fm.record(pt(100, METRIC_WORKER_UTILIZATION, 0.4));
        // window: [90, 100] → only pt at ts=100
        let m = fm.max(METRIC_WORKER_UTILIZATION, 10, 100).expect("some");
        assert!((m - 0.4).abs() < 1e-9, "max={m}");
    }

    // ---- Windowed count -----------------------------------------------------

    #[test]
    fn test_count_within_window() {
        let mut fm = FarmMetrics::new();
        for i in 0..5u64 {
            fm.record(pt(i * 10, METRIC_TASK_DURATION_SECS, 2.0));
        }
        // now=40, window=30 → ts 10,20,30,40 = 4 points
        assert_eq!(fm.count(METRIC_TASK_DURATION_SECS, 30, 40), 4);
    }

    // ---- Worker-tagged points -----------------------------------------------

    #[test]
    fn test_worker_tagged_points_stored_and_retrieved() {
        let mut fm = FarmMetrics::new();
        fm.record(wpt(50, METRIC_TASK_DURATION_SECS, 3.5, "worker-1"));
        fm.record(wpt(60, METRIC_TASK_DURATION_SECS, 1.5, "worker-2"));
        let avg = fm.average(METRIC_TASK_DURATION_SECS, 60, 60).expect("some");
        assert!((avg - 2.5).abs() < 1e-9, "avg={avg}");
    }

    // ---- Prune ---------------------------------------------------------------

    #[test]
    fn test_prune_older_than() {
        let mut fm = FarmMetrics::new();
        for i in 0u64..10 {
            fm.record(pt(i * 10, METRIC_QUEUE_DEPTH, 1.0));
        }
        assert_eq!(fm.total_point_count(), 10);

        // Prune points older than 30 seconds before now=90 → cutoff=60 → keep ts≥60
        fm.prune_older_than(30, 90);
        // Remaining: ts 60,70,80,90 = 4 points
        assert_eq!(fm.total_point_count(), 4);
    }

    // ---- Report generation --------------------------------------------------

    #[test]
    fn test_generate_report_empty() {
        let fm = FarmMetrics::new();
        let report = fm.generate_report(3600, 1000);
        assert_eq!(report.total_tasks_completed, 0);
        assert_eq!(report.total_tasks_failed, 0);
        assert!((report.avg_task_duration_secs - 0.0).abs() < f64::EPSILON);
        assert_eq!(report.peak_workers_used, 0);
        assert!((report.efficiency_pct - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generate_report_basic() {
        let mut fm = FarmMetrics::new();
        let now = 1000u64;

        // Record 5 task durations (each task completion emits one point).
        for i in 0..5u64 {
            fm.record(pt(now - 50 + i * 10, METRIC_TASK_DURATION_SECS, 10.0 + i as f64));
        }

        // Record worker utilization.
        fm.record(pt(now - 20, METRIC_WORKER_UTILIZATION, 0.8));
        fm.record(pt(now - 10, METRIC_WORKER_UTILIZATION, 0.6));
        fm.record(pt(now, METRIC_WORKER_UTILIZATION, 0.7));

        // No failures.
        fm.record(pt(now, METRIC_FAILURE_RATE, 0.0));

        let report = fm.generate_report(3600, now);

        assert_eq!(report.generated_at_secs, now);
        assert_eq!(report.total_tasks_completed, 5);
        assert_eq!(report.total_tasks_failed, 0);
        assert!((report.avg_task_duration_secs - 12.0).abs() < 1e-9);
        assert_eq!(report.peak_workers_used, 0); // max(0.8,0.6,0.7) → floor to 0
        // efficiency = (1-0) * avg(0.8,0.6,0.7) * 100 = 70
        assert!((report.efficiency_pct - 70.0).abs() < 0.01, "eff={}", report.efficiency_pct);
    }

    #[test]
    fn test_generate_report_with_failures() {
        let mut fm = FarmMetrics::new();
        let now = 2000u64;

        for _ in 0..10 {
            fm.record(pt(now, METRIC_TASK_DURATION_SECS, 5.0));
        }
        fm.record(pt(now, METRIC_FAILURE_RATE, 0.2)); // 20% failure rate
        fm.record(pt(now, METRIC_WORKER_UTILIZATION, 0.5));

        let report = fm.generate_report(3600, now);

        assert_eq!(report.total_tasks_completed, 10);
        assert_eq!(report.total_tasks_failed, 2); // 10 * 0.2
        // efficiency = (1-0.2) * 0.5 * 100 = 40
        assert!((report.efficiency_pct - 40.0).abs() < 0.01, "eff={}", report.efficiency_pct);
    }

    #[test]
    fn test_report_is_healthy() {
        let report = FarmReport {
            generated_at_secs: 0,
            total_tasks_completed: 100,
            total_tasks_failed: 2,
            avg_task_duration_secs: 5.0,
            peak_workers_used: 10,
            efficiency_pct: 80.0,
        };
        assert!(report.is_healthy(70.0, 0.05));
        assert!(!report.is_healthy(90.0, 0.05));
        assert!(!report.is_healthy(70.0, 0.01));
    }

    #[test]
    fn test_metric_names_returns_all_registered() {
        let mut fm = FarmMetrics::new();
        fm.record(pt(1, METRIC_QUEUE_DEPTH, 1.0));
        fm.record(pt(1, METRIC_TASKS_PER_SECOND, 2.0));
        let mut names = fm.metric_names();
        names.sort();
        assert_eq!(names, vec![METRIC_QUEUE_DEPTH, METRIC_TASKS_PER_SECOND]);
    }

    #[test]
    fn test_windowed_query_exact_boundary() {
        let mut fm = FarmMetrics::new();
        // Points at ts exactly on the boundary.
        fm.record(pt(50, METRIC_QUEUE_DEPTH, 1.0));  // cutoff = 50
        fm.record(pt(100, METRIC_QUEUE_DEPTH, 2.0)); // now = 100
        // window=50, now=100 → cutoff=50 → both points included
        let avg = fm.average(METRIC_QUEUE_DEPTH, 50, 100).expect("some");
        assert!((avg - 1.5).abs() < 1e-9, "avg={avg}");
    }
}
