//! Worker-level metrics collection

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Worker metrics collector
pub struct WorkerMetrics {
    tasks_completed: AtomicU64,
    tasks_failed: AtomicU64,
    total_processing_time_ms: AtomicU64,
    start_time: Instant,
}

impl WorkerMetrics {
    /// Create a new metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            tasks_completed: AtomicU64::new(0),
            tasks_failed: AtomicU64::new(0),
            total_processing_time_ms: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Increment completed tasks counter
    pub fn increment_completed(&self) {
        self.tasks_completed.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment failed tasks counter
    pub fn increment_failed(&self) {
        self.tasks_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Add processing time
    pub fn add_processing_time(&self, duration: Duration) {
        self.total_processing_time_ms
            .fetch_add(duration.as_millis() as u64, Ordering::Relaxed);
    }

    /// Get tasks completed count
    pub fn tasks_completed(&self) -> u64 {
        self.tasks_completed.load(Ordering::Relaxed)
    }

    /// Get tasks failed count
    pub fn tasks_failed(&self) -> u64 {
        self.tasks_failed.load(Ordering::Relaxed)
    }

    /// Get total processing time
    pub fn total_processing_time(&self) -> Duration {
        Duration::from_millis(self.total_processing_time_ms.load(Ordering::Relaxed))
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get average processing time per task
    pub fn average_processing_time(&self) -> Option<Duration> {
        let completed = self.tasks_completed();
        if completed == 0 {
            return None;
        }

        let total_ms = self.total_processing_time_ms.load(Ordering::Relaxed);
        Some(Duration::from_millis(total_ms / completed))
    }

    /// Get tasks per hour
    pub fn tasks_per_hour(&self) -> f64 {
        let uptime_hours = self.uptime().as_secs() as f64 / 3600.0;
        if uptime_hours == 0.0 {
            return 0.0;
        }

        self.tasks_completed() as f64 / uptime_hours
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        let completed = self.tasks_completed();
        let failed = self.tasks_failed();
        let total = completed + failed;

        if total == 0 {
            return 0.0;
        }

        completed as f64 / total as f64
    }

    /// Reset all metrics
    pub fn reset(&self) {
        self.tasks_completed.store(0, Ordering::Relaxed);
        self.tasks_failed.store(0, Ordering::Relaxed);
        self.total_processing_time_ms.store(0, Ordering::Relaxed);
    }

    /// Get metrics snapshot
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            tasks_completed: self.tasks_completed(),
            tasks_failed: self.tasks_failed(),
            total_processing_time: self.total_processing_time(),
            uptime: self.uptime(),
            average_processing_time: self.average_processing_time(),
            tasks_per_hour: self.tasks_per_hour(),
            success_rate: self.success_rate(),
        }
    }
}

impl Default for WorkerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of worker metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub tasks_completed: u64,
    pub tasks_failed: u64,
    pub total_processing_time: Duration,
    pub uptime: Duration,
    pub average_processing_time: Option<Duration>,
    pub tasks_per_hour: f64,
    pub success_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = WorkerMetrics::new();
        assert_eq!(metrics.tasks_completed(), 0);
        assert_eq!(metrics.tasks_failed(), 0);
    }

    #[test]
    fn test_increment_completed() {
        let metrics = WorkerMetrics::new();
        metrics.increment_completed();
        metrics.increment_completed();
        assert_eq!(metrics.tasks_completed(), 2);
    }

    #[test]
    fn test_increment_failed() {
        let metrics = WorkerMetrics::new();
        metrics.increment_failed();
        assert_eq!(metrics.tasks_failed(), 1);
    }

    #[test]
    fn test_processing_time() {
        let metrics = WorkerMetrics::new();
        metrics.add_processing_time(Duration::from_secs(10));
        metrics.add_processing_time(Duration::from_secs(20));

        let total = metrics.total_processing_time();
        assert_eq!(total.as_secs(), 30);
    }

    #[test]
    fn test_average_processing_time() {
        let metrics = WorkerMetrics::new();

        // No tasks completed
        assert!(metrics.average_processing_time().is_none());

        // Add some tasks
        metrics.increment_completed();
        metrics.add_processing_time(Duration::from_secs(10));

        metrics.increment_completed();
        metrics.add_processing_time(Duration::from_secs(20));

        let avg = metrics.average_processing_time().unwrap();
        assert_eq!(avg.as_secs(), 15);
    }

    #[test]
    fn test_success_rate() {
        let metrics = WorkerMetrics::new();

        // No tasks
        assert_eq!(metrics.success_rate(), 0.0);

        // All successful
        metrics.increment_completed();
        metrics.increment_completed();
        assert_eq!(metrics.success_rate(), 1.0);

        // One failure
        metrics.increment_failed();
        assert!((metrics.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_snapshot() {
        let metrics = WorkerMetrics::new();
        metrics.increment_completed();
        metrics.increment_failed();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.tasks_completed, 1);
        assert_eq!(snapshot.tasks_failed, 1);
    }

    #[test]
    fn test_reset() {
        let metrics = WorkerMetrics::new();
        metrics.increment_completed();
        metrics.increment_failed();
        metrics.add_processing_time(Duration::from_secs(10));

        metrics.reset();

        assert_eq!(metrics.tasks_completed(), 0);
        assert_eq!(metrics.tasks_failed(), 0);
        assert_eq!(metrics.total_processing_time().as_secs(), 0);
    }
}
