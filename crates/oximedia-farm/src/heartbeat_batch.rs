//! Batch heartbeat processing for the farm coordinator.
//!
//! Collects individual worker heartbeat reports and flushes them in bulk,
//! reducing per-heartbeat database overhead.  A batch is flushed when either:
//!
//! - The number of pending reports reaches `max_batch_size`, **or**
//! - The elapsed time since the last flush exceeds `flush_interval`.
//!
//! # Example
//!
//! ```rust
//! use std::time::Duration;
//! use oximedia_farm::heartbeat_batch::{HeartbeatBatch, HeartbeatReport};
//!
//! let mut batch = HeartbeatBatch::new(32, Duration::from_secs(1));
//! let report = HeartbeatReport::new("worker-1");
//! let auto_flushed = batch.add(report);
//! // auto_flushed is true only when max_batch_size was hit.
//! ```

use std::time::{Duration, Instant};

/// A snapshot of a worker's current status sent to the coordinator.
#[derive(Debug, Clone)]
pub struct HeartbeatReport {
    /// Unique worker identifier.
    pub worker_id: String,
    /// CPU utilisation in `[0.0, 1.0]`.
    pub cpu_usage: f64,
    /// Memory utilisation in `[0.0, 1.0]`.
    pub memory_usage: f64,
    /// Number of tasks currently running on the worker.
    pub active_tasks: u32,
    /// Timestamp when the report was generated.
    pub timestamp: Instant,
}

impl HeartbeatReport {
    /// Create a minimal heartbeat report for a given worker.
    #[must_use]
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            active_tasks: 0,
            timestamp: Instant::now(),
        }
    }

    /// Create a heartbeat report with full details.
    #[must_use]
    pub fn with_details(
        worker_id: impl Into<String>,
        cpu_usage: f64,
        memory_usage: f64,
        active_tasks: u32,
    ) -> Self {
        Self {
            worker_id: worker_id.into(),
            cpu_usage,
            memory_usage,
            active_tasks,
            timestamp: Instant::now(),
        }
    }
}

/// Accumulates [`HeartbeatReport`]s and flushes them in bulk.
///
/// Thread-safety is the caller's responsibility; wrap in a `Mutex` or
/// equivalent when used from multiple threads.
pub struct HeartbeatBatch {
    /// Pending reports waiting to be flushed.
    pending: Vec<HeartbeatReport>,
    /// Maximum number of reports before an automatic flush is triggered.
    max_batch_size: usize,
    /// Minimum time between flushes.
    flush_interval: Duration,
    /// When the last flush occurred (or when the batch was created).
    last_flush: Instant,
    /// Total number of reports that have been flushed (across all flushes).
    total_flushed: u64,
    /// Total number of auto-flushes triggered by batch-size overflow.
    size_triggered_flushes: u64,
    /// Total number of timer-triggered flushes.
    interval_triggered_flushes: u64,
}

impl HeartbeatBatch {
    /// Create a new batch processor.
    ///
    /// # Panics
    ///
    /// Panics if `max_batch_size == 0`.
    #[must_use]
    pub fn new(max_batch_size: usize, flush_interval: Duration) -> Self {
        assert!(max_batch_size > 0, "max_batch_size must be > 0");
        Self {
            pending: Vec::with_capacity(max_batch_size),
            max_batch_size,
            flush_interval,
            last_flush: Instant::now(),
            total_flushed: 0,
            size_triggered_flushes: 0,
            interval_triggered_flushes: 0,
        }
    }

    /// Add a [`HeartbeatReport`] to the batch.
    ///
    /// Returns `true` if the batch was **automatically flushed** because
    /// `max_batch_size` was reached.  The caller must process the flushed
    /// reports (available via [`Self::flush`] before calling `add` again for
    /// a fresh accumulation, or use the return value for side effects).
    ///
    /// Note: if the batch size limit is reached, the report is added first
    /// and then the batch is flushed atomically.
    pub fn add(&mut self, report: HeartbeatReport) -> bool {
        self.pending.push(report);
        if self.pending.len() >= self.max_batch_size {
            self.size_triggered_flushes += 1;
            true
        } else {
            false
        }
    }

    /// Return `true` when the flush interval has elapsed since the last flush.
    #[must_use]
    pub fn should_flush(&self) -> bool {
        self.last_flush.elapsed() >= self.flush_interval
    }

    /// Drain all pending reports and reset the flush timer.
    ///
    /// Returns the drained reports.  The internal buffer is cleared and the
    /// last-flush timestamp is updated.
    pub fn flush(&mut self) -> Vec<HeartbeatReport> {
        let batch = std::mem::take(&mut self.pending);
        self.total_flushed += batch.len() as u64;
        self.last_flush = Instant::now();
        batch
    }

    /// Drain pending reports only if the flush interval has elapsed.
    ///
    /// Returns `Some(reports)` when a flush occurred, `None` otherwise.
    pub fn flush_if_ready(&mut self) -> Option<Vec<HeartbeatReport>> {
        if self.should_flush() {
            self.interval_triggered_flushes += 1;
            Some(self.flush())
        } else {
            None
        }
    }

    /// Number of reports currently pending in the batch.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Maximum batch size before an automatic flush is triggered.
    #[must_use]
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// The configured flush interval.
    #[must_use]
    pub fn flush_interval(&self) -> Duration {
        self.flush_interval
    }

    /// Total number of reports that have ever been flushed.
    #[must_use]
    pub fn total_flushed(&self) -> u64 {
        self.total_flushed
    }

    /// Number of flushes triggered by the batch-size limit.
    #[must_use]
    pub fn size_triggered_flushes(&self) -> u64 {
        self.size_triggered_flushes
    }

    /// Number of flushes triggered by the interval timer.
    #[must_use]
    pub fn interval_triggered_flushes(&self) -> u64 {
        self.interval_triggered_flushes
    }

    /// Return `true` if there are no pending reports.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── HeartbeatReport ──────────────────────────────────────────────────────

    #[test]
    fn test_report_new_defaults() {
        let r = HeartbeatReport::new("w-1");
        assert_eq!(r.worker_id, "w-1");
        assert!((r.cpu_usage - 0.0).abs() < f64::EPSILON);
        assert_eq!(r.active_tasks, 0);
    }

    #[test]
    fn test_report_with_details() {
        let r = HeartbeatReport::with_details("w-2", 0.75, 0.5, 3);
        assert_eq!(r.worker_id, "w-2");
        assert!((r.cpu_usage - 0.75).abs() < f64::EPSILON);
        assert!((r.memory_usage - 0.5).abs() < f64::EPSILON);
        assert_eq!(r.active_tasks, 3);
    }

    // ── HeartbeatBatch construction ──────────────────────────────────────────

    #[test]
    fn test_batch_initial_state() {
        let batch = HeartbeatBatch::new(10, Duration::from_secs(1));
        assert_eq!(batch.pending_count(), 0);
        assert_eq!(batch.max_batch_size(), 10);
        assert!(batch.is_empty());
        assert_eq!(batch.total_flushed(), 0);
    }

    // ── add / auto-flush ─────────────────────────────────────────────────────

    #[test]
    fn test_add_single_report_no_auto_flush() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        let triggered = batch.add(HeartbeatReport::new("w-1"));
        assert!(!triggered, "should not auto-flush with 1 of 10 capacity");
        assert_eq!(batch.pending_count(), 1);
    }

    #[test]
    fn test_auto_flush_on_max_batch_size() {
        let mut batch = HeartbeatBatch::new(5, Duration::from_secs(60));
        for i in 0..4 {
            let triggered = batch.add(HeartbeatReport::new(format!("w-{i}")));
            assert!(!triggered, "should not flush before limit");
        }
        let triggered = batch.add(HeartbeatReport::new("w-4"));
        assert!(triggered, "5th add should trigger auto-flush flag");
        // Pending has 5 reports (the caller is expected to flush now)
        assert_eq!(batch.pending_count(), 5);
    }

    #[test]
    fn test_flush_drains_all_pending() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        for i in 0..7 {
            batch.add(HeartbeatReport::new(format!("worker-{i}")));
        }
        let drained = batch.flush();
        assert_eq!(drained.len(), 7);
        assert_eq!(batch.pending_count(), 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_flush_updates_total_flushed() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        for i in 0..6 {
            batch.add(HeartbeatReport::new(format!("w-{i}")));
        }
        let _ = batch.flush();
        assert_eq!(batch.total_flushed(), 6);
        batch.add(HeartbeatReport::new("w-extra"));
        let _ = batch.flush();
        assert_eq!(batch.total_flushed(), 7);
    }

    #[test]
    fn test_flush_empty_batch_returns_empty_vec() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        let drained = batch.flush();
        assert!(drained.is_empty());
    }

    // ── should_flush / timer-based flushing ──────────────────────────────────

    #[test]
    fn test_should_flush_false_immediately_after_create() {
        let batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        assert!(!batch.should_flush(), "interval has not elapsed yet");
    }

    #[test]
    fn test_should_flush_true_after_interval() {
        let batch = HeartbeatBatch::new(10, Duration::from_nanos(1));
        // 1 ns interval → should be expired almost immediately
        std::thread::sleep(Duration::from_millis(10));
        assert!(batch.should_flush(), "interval should have elapsed");
    }

    #[test]
    fn test_flush_if_ready_returns_none_before_interval() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        batch.add(HeartbeatReport::new("w-1"));
        let result = batch.flush_if_ready();
        assert!(result.is_none(), "should not flush before interval");
        assert_eq!(batch.pending_count(), 1, "report should still be pending");
    }

    #[test]
    fn test_flush_if_ready_flushes_after_interval() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_nanos(1));
        batch.add(HeartbeatReport::new("w-1"));
        batch.add(HeartbeatReport::new("w-2"));
        std::thread::sleep(Duration::from_millis(10));
        let result = batch.flush_if_ready();
        assert!(result.is_some(), "should flush after interval");
        let reports = result.expect("Some");
        assert_eq!(reports.len(), 2);
        assert_eq!(batch.interval_triggered_flushes(), 1);
    }

    // ── 100 heartbeats from 10 workers ───────────────────────────────────────

    #[test]
    fn test_100_heartbeats_from_10_workers() {
        let max_batch = 20;
        let mut batch = HeartbeatBatch::new(max_batch, Duration::from_secs(60));
        let mut auto_flush_count = 0usize;
        let mut all_reports: Vec<HeartbeatReport> = Vec::new();

        for i in 0..100 {
            let worker_id = format!("worker-{}", i % 10);
            let triggered = batch.add(HeartbeatReport::new(worker_id));
            if triggered {
                auto_flush_count += 1;
                all_reports.extend(batch.flush());
            }
        }
        // Flush remainder
        all_reports.extend(batch.flush());

        assert_eq!(all_reports.len(), 100, "all 100 reports should be flushed");
        // With 100 reports and max_batch=20, we expect exactly 5 auto-flushes
        // (the 20th, 40th, 60th, 80th, 100th add trigger the flag)
        assert_eq!(auto_flush_count, 5, "expected 5 size-triggered flushes");
        assert_eq!(batch.size_triggered_flushes(), 5);
    }

    #[test]
    fn test_flush_preserves_worker_id_order() {
        let mut batch = HeartbeatBatch::new(10, Duration::from_secs(60));
        let ids = ["alpha", "beta", "gamma"];
        for &id in &ids {
            batch.add(HeartbeatReport::new(id));
        }
        let flushed = batch.flush();
        let actual_ids: Vec<&str> = flushed.iter().map(|r| r.worker_id.as_str()).collect();
        assert_eq!(actual_ids, ids, "insertion order must be preserved");
    }

    #[test]
    fn test_size_triggered_flushes_counter() {
        let mut batch = HeartbeatBatch::new(3, Duration::from_secs(60));
        // First group of 3 → triggers
        batch.add(HeartbeatReport::new("a"));
        batch.add(HeartbeatReport::new("b"));
        let t1 = batch.add(HeartbeatReport::new("c")); // triggers
        assert!(t1);
        let _ = batch.flush();
        // Second group of 3 → triggers
        batch.add(HeartbeatReport::new("d"));
        batch.add(HeartbeatReport::new("e"));
        let t2 = batch.add(HeartbeatReport::new("f")); // triggers
        assert!(t2);
        let _ = batch.flush();
        assert_eq!(batch.size_triggered_flushes(), 2);
    }
}
