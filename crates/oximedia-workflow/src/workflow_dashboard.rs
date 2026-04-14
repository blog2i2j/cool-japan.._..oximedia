//! Workflow dashboard data provider.
//!
//! Aggregates metrics from the workflow orchestration engine into a structured
//! snapshot suitable for rendering in a web UI or reporting tool. The dashboard
//! provider collects per-workflow and global statistics, task-level health
//! metrics, queue depths, error histograms, and throughput windows — all
//! without assuming any particular frontend technology.

use crate::task::TaskState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// WorkflowStatus summary
// ---------------------------------------------------------------------------

/// Aggregate status counts for a single workflow instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStatusSummary {
    /// Workflow identifier (string to remain portable).
    pub workflow_id: String,
    /// Human-readable name.
    pub name: String,
    /// Current high-level state label (e.g. "running", "completed", "failed").
    pub state: String,
    /// Total tasks in this workflow.
    pub total_tasks: usize,
    /// Tasks that have completed successfully.
    pub completed_tasks: usize,
    /// Tasks currently executing.
    pub running_tasks: usize,
    /// Tasks waiting for dependencies.
    pub pending_tasks: usize,
    /// Tasks that failed.
    pub failed_tasks: usize,
    /// Tasks that were skipped.
    pub skipped_tasks: usize,
    /// Progress percentage `[0.0, 100.0]`.
    pub progress_pct: f64,
    /// Unix timestamp (seconds) when the workflow started, if known.
    pub started_at_secs: Option<u64>,
    /// Elapsed wall-clock seconds since start, if known.
    pub elapsed_secs: Option<u64>,
    /// Estimated seconds remaining, if available.
    pub eta_secs: Option<u64>,
}

impl WorkflowStatusSummary {
    /// Compute progress percentage from task counts.
    ///
    /// Considers completed + failed + skipped as "done".
    #[must_use]
    pub fn compute_progress(done: usize, total: usize) -> f64 {
        if total == 0 {
            return 100.0;
        }
        ((done as f64) / (total as f64) * 100.0).min(100.0)
    }
}

// ---------------------------------------------------------------------------
// TaskHealthEntry
// ---------------------------------------------------------------------------

/// Per-task health record for dashboard display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskHealthEntry {
    /// Task name.
    pub task_name: String,
    /// Current state.
    pub state: String,
    /// Number of retry attempts so far.
    pub retry_count: u32,
    /// Last error message, if any.
    pub last_error: Option<String>,
    /// Average duration in seconds over recent runs.
    pub avg_duration_secs: f64,
    /// Whether the task is considered healthy (no errors, low retry count).
    pub healthy: bool,
}

impl TaskHealthEntry {
    /// Construct a health entry from raw data.
    #[must_use]
    pub fn new(
        task_name: impl Into<String>,
        state: TaskState,
        retry_count: u32,
        last_error: Option<String>,
        avg_duration_secs: f64,
    ) -> Self {
        let healthy = !matches!(state, TaskState::Failed) && retry_count < 3;
        Self {
            task_name: task_name.into(),
            state: format!("{state:?}"),
            retry_count,
            last_error,
            avg_duration_secs,
            healthy,
        }
    }
}

// ---------------------------------------------------------------------------
// ErrorHistogram
// ---------------------------------------------------------------------------

/// Bucketed error counts grouped by error category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorHistogram {
    /// Map from error category string to count.
    pub buckets: HashMap<String, u64>,
    /// Total errors across all categories.
    pub total: u64,
}

impl ErrorHistogram {
    /// Create an empty histogram.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the counter for `category`.
    pub fn record(&mut self, category: impl Into<String>) {
        let count = self.buckets.entry(category.into()).or_insert(0);
        *count += 1;
        self.total += 1;
    }

    /// Return the most common error category, if any.
    #[must_use]
    pub fn top_category(&self) -> Option<(&str, u64)> {
        self.buckets
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, &v)| (k.as_str(), v))
    }

    /// Return sorted entries from most to least frequent.
    #[must_use]
    pub fn sorted_entries(&self) -> Vec<(&str, u64)> {
        let mut entries: Vec<(&str, u64)> =
            self.buckets.iter().map(|(k, &v)| (k.as_str(), v)).collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries
    }
}

// ---------------------------------------------------------------------------
// ThroughputWindow
// ---------------------------------------------------------------------------

/// Rolling-window throughput statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputWindow {
    /// Window duration in seconds.
    pub window_secs: u64,
    /// Tasks completed within the window.
    pub completed_in_window: u64,
    /// Tasks failed within the window.
    pub failed_in_window: u64,
    /// Tasks per second (completed / window_secs).
    pub tasks_per_second: f64,
    /// Success rate within the window `[0.0, 1.0]`.
    pub success_rate: f64,
}

impl ThroughputWindow {
    /// Compute a throughput window from raw counts.
    #[must_use]
    pub fn compute(completed: u64, failed: u64, window_secs: u64) -> Self {
        let tasks_per_second = if window_secs > 0 {
            completed as f64 / window_secs as f64
        } else {
            0.0
        };
        let total = completed + failed;
        let success_rate = if total > 0 {
            completed as f64 / total as f64
        } else {
            1.0
        };
        Self {
            window_secs,
            completed_in_window: completed,
            failed_in_window: failed,
            tasks_per_second,
            success_rate,
        }
    }
}

// ---------------------------------------------------------------------------
// QueueDepthSnapshot
// ---------------------------------------------------------------------------

/// Current depth of each priority tier in the task queue.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueueDepthSnapshot {
    /// Critical priority tasks.
    pub critical: usize,
    /// High priority tasks.
    pub high: usize,
    /// Normal priority tasks.
    pub normal: usize,
    /// Low priority tasks.
    pub low: usize,
}

impl QueueDepthSnapshot {
    /// Total tasks across all priority tiers.
    #[must_use]
    pub fn total(&self) -> usize {
        self.critical + self.high + self.normal + self.low
    }
}

// ---------------------------------------------------------------------------
// DashboardSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of the full dashboard state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSnapshot {
    /// Snapshot timestamp (seconds since Unix epoch).
    pub timestamp_secs: u64,
    /// Overall counts.
    pub global: GlobalCounters,
    /// Per-workflow status summaries.
    pub workflows: Vec<WorkflowStatusSummary>,
    /// Task health entries.
    pub task_health: Vec<TaskHealthEntry>,
    /// Error distribution.
    pub error_histogram: ErrorHistogram,
    /// Short-window throughput (last 60 seconds).
    pub throughput_1m: ThroughputWindow,
    /// Medium-window throughput (last 300 seconds).
    pub throughput_5m: ThroughputWindow,
    /// Current queue depth.
    pub queue_depth: QueueDepthSnapshot,
    /// Custom metric key-values for extension.
    pub custom_metrics: HashMap<String, serde_json::Value>,
}

/// Global workflow engine counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalCounters {
    /// Total workflows submitted since start.
    pub total_workflows: u64,
    /// Workflows currently running.
    pub running_workflows: u64,
    /// Workflows completed successfully (all time).
    pub completed_workflows: u64,
    /// Workflows that failed (all time).
    pub failed_workflows: u64,
    /// Total tasks executed (all time).
    pub total_tasks_executed: u64,
    /// Tasks currently running.
    pub running_tasks: u64,
}

impl GlobalCounters {
    /// Overall workflow success rate `[0.0, 1.0]`.
    #[must_use]
    pub fn workflow_success_rate(&self) -> f64 {
        let finished = self.completed_workflows + self.failed_workflows;
        if finished == 0 {
            return 1.0;
        }
        self.completed_workflows as f64 / finished as f64
    }
}

// ---------------------------------------------------------------------------
// DashboardDataProvider
// ---------------------------------------------------------------------------

/// Aggregates metrics from workflow engine state into `DashboardSnapshot`s.
///
/// The provider holds an internal event log and counters that are updated
/// as workflows and tasks progress. Call [`DashboardDataProvider::snapshot`]
/// to obtain a consistent read of all metrics at a given timestamp.
#[derive(Debug, Default)]
pub struct DashboardDataProvider {
    /// Global counters.
    counters: GlobalCounters,
    /// Per-workflow status records (workflow_id → summary).
    workflow_records: HashMap<String, WorkflowStatusSummary>,
    /// Per-task health records (task_name → entry).
    task_health: HashMap<String, TaskHealthEntry>,
    /// Error histogram.
    error_histogram: ErrorHistogram,
    /// Completed event timestamps (unix secs) for throughput windows.
    completed_events: Vec<u64>,
    /// Failed event timestamps (unix secs) for throughput windows.
    failed_events: Vec<u64>,
    /// Current queue depth.
    queue_depth: QueueDepthSnapshot,
    /// Custom metrics.
    custom_metrics: HashMap<String, serde_json::Value>,
    /// Maximum event history to retain.
    max_event_history: usize,
}

impl DashboardDataProvider {
    /// Create a new dashboard data provider with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_event_history: 10_000,
            ..Default::default()
        }
    }

    /// Register a workflow starting.
    pub fn on_workflow_started(
        &mut self,
        workflow_id: impl Into<String>,
        name: impl Into<String>,
        total_tasks: usize,
        started_at_secs: u64,
    ) {
        self.counters.total_workflows += 1;
        self.counters.running_workflows += 1;

        let id = workflow_id.into();
        let summary = WorkflowStatusSummary {
            workflow_id: id.clone(),
            name: name.into(),
            state: "running".to_string(),
            total_tasks,
            completed_tasks: 0,
            running_tasks: 0,
            pending_tasks: total_tasks,
            failed_tasks: 0,
            skipped_tasks: 0,
            progress_pct: 0.0,
            started_at_secs: Some(started_at_secs),
            elapsed_secs: None,
            eta_secs: None,
        };
        self.workflow_records.insert(id, summary);
    }

    /// Record a workflow completing (successfully or otherwise).
    pub fn on_workflow_completed(&mut self, workflow_id: &str, success: bool, now_secs: u64) {
        if self.counters.running_workflows > 0 {
            self.counters.running_workflows -= 1;
        }
        if success {
            self.counters.completed_workflows += 1;
        } else {
            self.counters.failed_workflows += 1;
        }

        if let Some(record) = self.workflow_records.get_mut(workflow_id) {
            record.state = if success { "completed" } else { "failed" }.to_string();
            record.progress_pct = 100.0;
            if let Some(start) = record.started_at_secs {
                record.elapsed_secs = Some(now_secs.saturating_sub(start));
            }
        }
    }

    /// Record a task completing.
    pub fn on_task_completed(
        &mut self,
        workflow_id: &str,
        task_name: &str,
        success: bool,
        duration_secs: f64,
        now_secs: u64,
    ) {
        self.counters.total_tasks_executed += 1;
        if self.counters.running_tasks > 0 {
            self.counters.running_tasks -= 1;
        }

        if success {
            self.completed_events.push(now_secs);
        } else {
            self.failed_events.push(now_secs);
            self.error_histogram.record(task_name);
        }
        self.trim_event_history(now_secs);

        // Update workflow record
        if let Some(record) = self.workflow_records.get_mut(workflow_id) {
            if success {
                record.completed_tasks += 1;
            } else {
                record.failed_tasks += 1;
            }
            let done = record.completed_tasks + record.failed_tasks + record.skipped_tasks;
            record.progress_pct = WorkflowStatusSummary::compute_progress(done, record.total_tasks);

            // Compute ETA from elapsed and progress
            if let Some(start) = record.started_at_secs {
                let elapsed = now_secs.saturating_sub(start);
                record.elapsed_secs = Some(elapsed);
                if record.progress_pct > 0.0 && record.progress_pct < 100.0 {
                    let total_estimated = (elapsed as f64 / (record.progress_pct / 100.0)) as u64;
                    record.eta_secs = Some(total_estimated.saturating_sub(elapsed));
                }
            }
        }

        // Update task health
        let entry = self
            .task_health
            .entry(task_name.to_string())
            .or_insert_with(|| TaskHealthEntry {
                task_name: task_name.to_string(),
                state: "pending".to_string(),
                retry_count: 0,
                last_error: None,
                avg_duration_secs: 0.0,
                healthy: true,
            });
        let state = if success {
            TaskState::Completed
        } else {
            TaskState::Failed
        };
        let new_entry = TaskHealthEntry::new(
            task_name,
            state,
            entry.retry_count,
            entry.last_error.clone(),
            // Running average: (prev * n + new) / (n+1) — use simple EWMA with α=0.3
            entry.avg_duration_secs * 0.7 + duration_secs * 0.3,
        );
        self.task_health.insert(task_name.to_string(), new_entry);
    }

    /// Record a task failure with an error message.
    pub fn on_task_failed(&mut self, task_name: &str, error: impl Into<String>, retry_count: u32) {
        let error_msg = error.into();
        self.error_histogram.record(task_name);
        let entry = self
            .task_health
            .entry(task_name.to_string())
            .or_insert_with(|| TaskHealthEntry {
                task_name: task_name.to_string(),
                state: "failed".to_string(),
                retry_count: 0,
                last_error: None,
                avg_duration_secs: 0.0,
                healthy: false,
            });
        entry.state = "failed".to_string();
        entry.last_error = Some(error_msg);
        entry.retry_count = retry_count;
        entry.healthy = false;
    }

    /// Update the queue depth snapshot.
    pub fn update_queue_depth(&mut self, snapshot: QueueDepthSnapshot) {
        self.queue_depth = snapshot;
    }

    /// Set a custom metric value.
    pub fn set_custom_metric(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.custom_metrics.insert(key.into(), value);
    }

    /// Build a `DashboardSnapshot` at the given timestamp.
    #[must_use]
    pub fn snapshot(&self, now_secs: u64) -> DashboardSnapshot {
        let throughput_1m = self.throughput_window(now_secs, 60);
        let throughput_5m = self.throughput_window(now_secs, 300);

        DashboardSnapshot {
            timestamp_secs: now_secs,
            global: self.counters.clone(),
            workflows: self.workflow_records.values().cloned().collect(),
            task_health: self.task_health.values().cloned().collect(),
            error_histogram: self.error_histogram.clone(),
            throughput_1m,
            throughput_5m,
            queue_depth: self.queue_depth.clone(),
            custom_metrics: self.custom_metrics.clone(),
        }
    }

    /// Compute throughput window metrics for the given window in seconds.
    fn throughput_window(&self, now_secs: u64, window_secs: u64) -> ThroughputWindow {
        let cutoff = now_secs.saturating_sub(window_secs);
        let completed = self
            .completed_events
            .iter()
            .filter(|&&t| t >= cutoff)
            .count() as u64;
        let failed = self.failed_events.iter().filter(|&&t| t >= cutoff).count() as u64;
        ThroughputWindow::compute(completed, failed, window_secs)
    }

    /// Trim event history older than 10 minutes to bound memory usage.
    fn trim_event_history(&mut self, now_secs: u64) {
        let cutoff = now_secs.saturating_sub(600);
        self.completed_events.retain(|&t| t >= cutoff);
        self.failed_events.retain(|&t| t >= cutoff);

        // Also enforce max_event_history
        let max = self.max_event_history;
        if self.completed_events.len() > max {
            let excess = self.completed_events.len() - max;
            self.completed_events.drain(0..excess);
        }
        if self.failed_events.len() > max {
            let excess = self.failed_events.len() - max;
            self.failed_events.drain(0..excess);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_progress_zero_total() {
        assert_eq!(WorkflowStatusSummary::compute_progress(0, 0), 100.0);
    }

    #[test]
    fn test_compute_progress_half() {
        let p = WorkflowStatusSummary::compute_progress(5, 10);
        assert!((p - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_progress_full() {
        let p = WorkflowStatusSummary::compute_progress(10, 10);
        assert!((p - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_error_histogram_record_and_total() {
        let mut h = ErrorHistogram::new();
        h.record("timeout");
        h.record("timeout");
        h.record("io_error");
        assert_eq!(h.total, 3);
        assert_eq!(h.buckets["timeout"], 2);
        assert_eq!(h.buckets["io_error"], 1);
    }

    #[test]
    fn test_error_histogram_top_category() {
        let mut h = ErrorHistogram::new();
        h.record("network");
        h.record("network");
        h.record("disk");
        let top = h.top_category();
        assert!(top.is_some());
        assert_eq!(top.expect("should have entry").0, "network");
    }

    #[test]
    fn test_error_histogram_sorted_entries() {
        let mut h = ErrorHistogram::new();
        h.record("a");
        h.record("b");
        h.record("b");
        h.record("c");
        h.record("c");
        h.record("c");
        let entries = h.sorted_entries();
        assert_eq!(entries[0].0, "c");
        assert_eq!(entries[0].1, 3);
        assert_eq!(entries[1].0, "b");
    }

    #[test]
    fn test_throughput_window_compute() {
        let tw = ThroughputWindow::compute(60, 5, 60);
        assert!((tw.tasks_per_second - 1.0).abs() < 0.001);
        let expected_rate = 60.0 / 65.0;
        assert!((tw.success_rate - expected_rate).abs() < 0.001);
    }

    #[test]
    fn test_throughput_window_zero_window() {
        let tw = ThroughputWindow::compute(10, 0, 0);
        assert_eq!(tw.tasks_per_second, 0.0);
    }

    #[test]
    fn test_throughput_window_all_success() {
        let tw = ThroughputWindow::compute(10, 0, 60);
        assert!((tw.success_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_queue_depth_total() {
        let q = QueueDepthSnapshot {
            critical: 1,
            high: 2,
            normal: 3,
            low: 4,
        };
        assert_eq!(q.total(), 10);
    }

    #[test]
    fn test_global_counters_success_rate() {
        let mut g = GlobalCounters::default();
        g.completed_workflows = 8;
        g.failed_workflows = 2;
        assert!((g.workflow_success_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_global_counters_success_rate_empty() {
        let g = GlobalCounters::default();
        assert!((g.workflow_success_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_dashboard_provider_workflow_lifecycle() {
        let mut provider = DashboardDataProvider::new();
        provider.on_workflow_started("wf-1", "Test Workflow", 4, 1000);

        {
            let snap = provider.snapshot(1010);
            assert_eq!(snap.global.total_workflows, 1);
            assert_eq!(snap.global.running_workflows, 1);
            assert_eq!(snap.workflows.len(), 1);
        }

        provider.on_task_completed("wf-1", "encode", true, 2.0, 1010);
        provider.on_task_completed("wf-1", "qc", true, 1.0, 1015);
        provider.on_workflow_completed("wf-1", true, 1020);

        let snap = provider.snapshot(1020);
        assert_eq!(snap.global.completed_workflows, 1);
        assert_eq!(snap.global.running_workflows, 0);
        let wf = snap
            .workflows
            .iter()
            .find(|w| w.workflow_id == "wf-1")
            .expect("workflow in snapshot");
        assert_eq!(wf.state, "completed");
    }

    #[test]
    fn test_dashboard_provider_task_failure() {
        let mut provider = DashboardDataProvider::new();
        provider.on_workflow_started("wf-2", "Failing WF", 2, 2000);
        provider.on_task_failed("transcode", "codec error", 1);

        let snap = provider.snapshot(2010);
        let health = snap
            .task_health
            .iter()
            .find(|h| h.task_name == "transcode")
            .expect("health entry");
        assert!(!health.healthy);
        assert_eq!(health.retry_count, 1);
        assert!(health.last_error.is_some());
    }

    #[test]
    fn test_dashboard_provider_throughput_window() {
        let mut provider = DashboardDataProvider::new();
        provider.on_workflow_started("wf-3", "Throughput Test", 5, 1000);
        // Complete tasks at t=1050 (within 60-second window at t=1100)
        for _ in 0..3 {
            provider.on_task_completed("wf-3", "task", true, 1.0, 1050);
        }
        provider.on_task_completed("wf-3", "task2", false, 1.0, 1050);

        let snap = provider.snapshot(1100);
        // All events are within 60s window
        assert_eq!(snap.throughput_1m.completed_in_window, 3);
        assert_eq!(snap.throughput_1m.failed_in_window, 1);
    }

    #[test]
    fn test_dashboard_provider_custom_metrics() {
        let mut provider = DashboardDataProvider::new();
        provider.set_custom_metric("gpu_utilization", serde_json::json!(0.85));

        let snap = provider.snapshot(5000);
        assert_eq!(
            snap.custom_metrics.get("gpu_utilization"),
            Some(&serde_json::json!(0.85))
        );
    }

    #[test]
    fn test_queue_depth_update() {
        let mut provider = DashboardDataProvider::new();
        provider.update_queue_depth(QueueDepthSnapshot {
            critical: 2,
            high: 5,
            normal: 10,
            low: 1,
        });

        let snap = provider.snapshot(1000);
        assert_eq!(snap.queue_depth.total(), 18);
        assert_eq!(snap.queue_depth.critical, 2);
    }

    #[test]
    fn test_task_health_entry_healthy() {
        let entry = TaskHealthEntry::new("encode", TaskState::Completed, 0, None, 2.5);
        assert!(entry.healthy);
    }

    #[test]
    fn test_task_health_entry_unhealthy_failed() {
        let entry = TaskHealthEntry::new(
            "encode",
            TaskState::Failed,
            0,
            Some("timeout".to_string()),
            2.5,
        );
        assert!(!entry.healthy);
    }

    #[test]
    fn test_task_health_entry_unhealthy_many_retries() {
        let entry = TaskHealthEntry::new("encode", TaskState::Running, 5, None, 2.5);
        assert!(!entry.healthy);
    }

    #[test]
    fn test_dashboard_snapshot_serialization() {
        let mut provider = DashboardDataProvider::new();
        provider.on_workflow_started("wf-s", "Serialize Test", 2, 9000);
        let snap = provider.snapshot(9010);
        let json = serde_json::to_string(&snap).expect("serialize");
        let _decoded: DashboardSnapshot = serde_json::from_str(&json).expect("deserialize");
    }
}
