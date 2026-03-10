#![allow(dead_code)]
//! Performance statistics and counters for the acceleration layer.
//!
//! Tracks kernel execution times, memory transfer throughput,
//! task completion rates, and other metrics for profiling and
//! monitoring GPU/CPU workloads.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A single recorded timing measurement for a kernel or operation.
#[derive(Debug, Clone, Copy)]
pub struct TimingSample {
    /// Duration of the operation.
    pub duration: Duration,
    /// Timestamp when the operation started.
    pub started_at: Instant,
}

impl TimingSample {
    /// Creates a new timing sample.
    #[must_use]
    pub fn new(duration: Duration, started_at: Instant) -> Self {
        Self {
            duration,
            started_at,
        }
    }

    /// Returns the duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Returns the duration in microseconds.
    #[must_use]
    pub fn duration_us(&self) -> f64 {
        self.duration.as_secs_f64() * 1_000_000.0
    }
}

/// Rolling statistics tracker for a named operation.
#[derive(Debug, Clone)]
pub struct OperationStats {
    /// Operation name.
    pub name: String,
    /// Total number of invocations.
    pub invocation_count: u64,
    /// Total cumulative duration.
    pub total_duration: Duration,
    /// Minimum observed duration.
    pub min_duration: Option<Duration>,
    /// Maximum observed duration.
    pub max_duration: Option<Duration>,
    /// Recent timing samples (bounded).
    recent_samples: Vec<TimingSample>,
    /// Maximum number of recent samples to keep.
    max_samples: usize,
}

impl OperationStats {
    /// Creates a new operation statistics tracker.
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            invocation_count: 0,
            total_duration: Duration::ZERO,
            min_duration: None,
            max_duration: None,
            recent_samples: Vec::new(),
            max_samples: 100,
        }
    }

    /// Creates a new tracker with a custom sample buffer size.
    #[must_use]
    pub fn with_max_samples(name: &str, max_samples: usize) -> Self {
        Self {
            max_samples: max_samples.max(1),
            ..Self::new(name)
        }
    }

    /// Records a completed operation.
    pub fn record(&mut self, duration: Duration, started_at: Instant) {
        self.invocation_count += 1;
        self.total_duration += duration;

        self.min_duration = Some(match self.min_duration {
            Some(min) => min.min(duration),
            None => duration,
        });
        self.max_duration = Some(match self.max_duration {
            Some(max) => max.max(duration),
            None => duration,
        });

        if self.recent_samples.len() >= self.max_samples {
            self.recent_samples.remove(0);
        }
        self.recent_samples
            .push(TimingSample::new(duration, started_at));
    }

    /// Returns the average duration, or `None` if no samples recorded.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_duration(&self) -> Option<Duration> {
        if self.invocation_count == 0 {
            return None;
        }
        Some(self.total_duration / self.invocation_count as u32)
    }

    /// Returns the average duration in milliseconds, or `None` if no samples.
    #[must_use]
    pub fn average_ms(&self) -> Option<f64> {
        self.average_duration().map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the minimum duration in milliseconds, or `None`.
    #[must_use]
    pub fn min_ms(&self) -> Option<f64> {
        self.min_duration.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the maximum duration in milliseconds, or `None`.
    #[must_use]
    pub fn max_ms(&self) -> Option<f64> {
        self.max_duration.map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Returns the number of recent samples.
    #[must_use]
    pub fn recent_count(&self) -> usize {
        self.recent_samples.len()
    }

    /// Returns the median of the recent samples, or `None` if empty.
    #[must_use]
    pub fn recent_median_ms(&self) -> Option<f64> {
        if self.recent_samples.is_empty() {
            return None;
        }
        let mut durations: Vec<f64> = self
            .recent_samples
            .iter()
            .map(TimingSample::duration_ms)
            .collect();
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = durations.len() / 2;
        if durations.len() % 2 == 0 && durations.len() >= 2 {
            Some((durations[mid - 1] + durations[mid]) / 2.0)
        } else {
            Some(durations[mid])
        }
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.invocation_count = 0;
        self.total_duration = Duration::ZERO;
        self.min_duration = None;
        self.max_duration = None;
        self.recent_samples.clear();
    }
}

/// Memory transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    /// Host (CPU) to device (GPU).
    HostToDevice,
    /// Device (GPU) to host (CPU).
    DeviceToHost,
    /// Device to device (intra-GPU or multi-GPU).
    DeviceToDevice,
}

impl TransferDirection {
    /// Returns a short label for the direction.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::HostToDevice => "H2D",
            Self::DeviceToHost => "D2H",
            Self::DeviceToDevice => "D2D",
        }
    }
}

/// Tracks memory transfer statistics.
#[derive(Debug, Clone)]
pub struct TransferStats {
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Total bytes transferred.
    pub total_bytes: u64,
    /// Total number of transfers.
    pub transfer_count: u64,
    /// Total time spent on transfers.
    pub total_duration: Duration,
}

impl TransferStats {
    /// Creates a new transfer statistics tracker.
    #[must_use]
    pub fn new(direction: TransferDirection) -> Self {
        Self {
            direction,
            total_bytes: 0,
            transfer_count: 0,
            total_duration: Duration::ZERO,
        }
    }

    /// Records a completed transfer.
    pub fn record(&mut self, bytes: u64, duration: Duration) {
        self.total_bytes += bytes;
        self.transfer_count += 1;
        self.total_duration += duration;
    }

    /// Returns the average throughput in bytes per second, or `None`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn throughput_bps(&self) -> Option<f64> {
        let secs = self.total_duration.as_secs_f64();
        if secs <= 0.0 {
            return None;
        }
        Some(self.total_bytes as f64 / secs)
    }

    /// Returns the throughput in megabytes per second, or `None`.
    #[must_use]
    pub fn throughput_mbps(&self) -> Option<f64> {
        self.throughput_bps().map(|bps| bps / (1024.0 * 1024.0))
    }

    /// Returns the average transfer size in bytes, or `None`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn average_transfer_size(&self) -> Option<f64> {
        if self.transfer_count == 0 {
            return None;
        }
        Some(self.total_bytes as f64 / self.transfer_count as f64)
    }

    /// Resets all transfer statistics.
    pub fn reset(&mut self) {
        self.total_bytes = 0;
        self.transfer_count = 0;
        self.total_duration = Duration::ZERO;
    }
}

/// Aggregated acceleration statistics across all operations.
#[derive(Debug)]
pub struct AccelStatistics {
    /// Per-operation timing statistics.
    operations: HashMap<String, OperationStats>,
    /// Per-direction transfer statistics.
    transfers: HashMap<TransferDirection, TransferStats>,
    /// Total tasks submitted.
    pub tasks_submitted: u64,
    /// Total tasks completed successfully.
    pub tasks_completed: u64,
    /// Total tasks that failed.
    pub tasks_failed: u64,
    /// Timestamp when stats collection started.
    pub started_at: Instant,
}

impl AccelStatistics {
    /// Creates a new statistics collector.
    #[must_use]
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            transfers: HashMap::new(),
            tasks_submitted: 0,
            tasks_completed: 0,
            tasks_failed: 0,
            started_at: Instant::now(),
        }
    }

    /// Records an operation timing.
    pub fn record_operation(&mut self, name: &str, duration: Duration, started_at: Instant) {
        self.operations
            .entry(name.to_string())
            .or_insert_with(|| OperationStats::new(name))
            .record(duration, started_at);
    }

    /// Records a memory transfer.
    pub fn record_transfer(
        &mut self,
        direction: TransferDirection,
        bytes: u64,
        duration: Duration,
    ) {
        self.transfers
            .entry(direction)
            .or_insert_with(|| TransferStats::new(direction))
            .record(bytes, duration);
    }

    /// Increments the submitted task counter.
    pub fn record_task_submitted(&mut self) {
        self.tasks_submitted += 1;
    }

    /// Increments the completed task counter.
    pub fn record_task_completed(&mut self) {
        self.tasks_completed += 1;
    }

    /// Increments the failed task counter.
    pub fn record_task_failed(&mut self) {
        self.tasks_failed += 1;
    }

    /// Returns how long statistics have been collected.
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Returns the task success rate (0.0 to 1.0), or `None` if no tasks.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> Option<f64> {
        let total = self.tasks_completed + self.tasks_failed;
        if total == 0 {
            return None;
        }
        Some(self.tasks_completed as f64 / total as f64)
    }

    /// Returns statistics for a named operation, if available.
    #[must_use]
    pub fn get_operation(&self, name: &str) -> Option<&OperationStats> {
        self.operations.get(name)
    }

    /// Returns transfer statistics for a direction, if available.
    #[must_use]
    pub fn get_transfer(&self, direction: TransferDirection) -> Option<&TransferStats> {
        self.transfers.get(&direction)
    }

    /// Returns the names of all tracked operations.
    #[must_use]
    pub fn operation_names(&self) -> Vec<&str> {
        self.operations.keys().map(String::as_str).collect()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        self.operations.clear();
        self.transfers.clear();
        self.tasks_submitted = 0;
        self.tasks_completed = 0;
        self.tasks_failed = 0;
        self.started_at = Instant::now();
    }
}

impl Default for AccelStatistics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_sample() {
        let now = Instant::now();
        let sample = TimingSample::new(Duration::from_millis(50), now);
        assert!((sample.duration_ms() - 50.0).abs() < 0.01);
        assert!((sample.duration_us() - 50000.0).abs() < 10.0);
    }

    #[test]
    fn test_operation_stats_empty() {
        let stats = OperationStats::new("test");
        assert_eq!(stats.invocation_count, 0);
        assert!(stats.average_duration().is_none());
        assert!(stats.min_ms().is_none());
        assert!(stats.max_ms().is_none());
        assert!(stats.recent_median_ms().is_none());
    }

    #[test]
    fn test_operation_stats_record() {
        let mut stats = OperationStats::new("scale");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(20), now);
        stats.record(Duration::from_millis(30), now);
        assert_eq!(stats.invocation_count, 3);
        let avg = stats.average_ms().expect("avg should be valid");
        assert!((avg - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_operation_stats_min_max() {
        let mut stats = OperationStats::new("conv");
        let now = Instant::now();
        stats.record(Duration::from_millis(5), now);
        stats.record(Duration::from_millis(15), now);
        stats.record(Duration::from_millis(10), now);
        assert!((stats.min_ms().expect("min_ms should succeed") - 5.0).abs() < 0.01);
        assert!((stats.max_ms().expect("max_ms should succeed") - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_median_odd() {
        let mut stats = OperationStats::new("test");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(30), now);
        stats.record(Duration::from_millis(20), now);
        // Sorted: 10, 20, 30 -> median = 20
        let median = stats.recent_median_ms().expect("median should be valid");
        assert!((median - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_median_even() {
        let mut stats = OperationStats::new("test");
        let now = Instant::now();
        stats.record(Duration::from_millis(10), now);
        stats.record(Duration::from_millis(20), now);
        // Sorted: 10, 20 -> median = 15
        let median = stats.recent_median_ms().expect("median should be valid");
        assert!((median - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_operation_stats_reset() {
        let mut stats = OperationStats::new("op");
        let now = Instant::now();
        stats.record(Duration::from_millis(1), now);
        stats.reset();
        assert_eq!(stats.invocation_count, 0);
        assert!(stats.average_duration().is_none());
    }

    #[test]
    fn test_operation_stats_sample_eviction() {
        let mut stats = OperationStats::with_max_samples("op", 3);
        let now = Instant::now();
        for i in 0..5 {
            stats.record(Duration::from_millis(i * 10), now);
        }
        assert_eq!(stats.recent_count(), 3);
    }

    #[test]
    fn test_transfer_direction_label() {
        assert_eq!(TransferDirection::HostToDevice.label(), "H2D");
        assert_eq!(TransferDirection::DeviceToHost.label(), "D2H");
        assert_eq!(TransferDirection::DeviceToDevice.label(), "D2D");
    }

    #[test]
    fn test_transfer_stats_record() {
        let mut stats = TransferStats::new(TransferDirection::HostToDevice);
        stats.record(1024, Duration::from_millis(1));
        stats.record(2048, Duration::from_millis(2));
        assert_eq!(stats.total_bytes, 3072);
        assert_eq!(stats.transfer_count, 2);
    }

    #[test]
    fn test_transfer_stats_throughput() {
        let mut stats = TransferStats::new(TransferDirection::DeviceToHost);
        stats.record(1_000_000, Duration::from_secs(1));
        let bps = stats.throughput_bps().expect("bps should be valid");
        assert!((bps - 1_000_000.0).abs() < 1.0);
        let mbps = stats.throughput_mbps().expect("mbps should be valid");
        assert!(mbps > 0.0);
    }

    #[test]
    fn test_transfer_stats_empty_throughput() {
        let stats = TransferStats::new(TransferDirection::HostToDevice);
        assert!(stats.throughput_bps().is_none());
        assert!(stats.average_transfer_size().is_none());
    }

    #[test]
    fn test_transfer_stats_reset() {
        let mut stats = TransferStats::new(TransferDirection::HostToDevice);
        stats.record(100, Duration::from_millis(1));
        stats.reset();
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.transfer_count, 0);
    }

    #[test]
    fn test_accel_statistics_record_operations() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("scale", Duration::from_millis(5), now);
        s.record_operation("scale", Duration::from_millis(10), now);
        let op = s.get_operation("scale").expect("op should be valid");
        assert_eq!(op.invocation_count, 2);
    }

    #[test]
    fn test_accel_statistics_tasks() {
        let mut s = AccelStatistics::new();
        s.record_task_submitted();
        s.record_task_submitted();
        s.record_task_completed();
        s.record_task_failed();
        assert_eq!(s.tasks_submitted, 2);
        assert_eq!(s.tasks_completed, 1);
        assert_eq!(s.tasks_failed, 1);
        assert!((s.success_rate().expect("success_rate should succeed") - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_accel_statistics_success_rate_none() {
        let s = AccelStatistics::new();
        assert!(s.success_rate().is_none());
    }

    #[test]
    fn test_accel_statistics_reset() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("op", Duration::from_millis(1), now);
        s.record_task_submitted();
        s.reset();
        assert!(s.operation_names().is_empty());
        assert_eq!(s.tasks_submitted, 0);
    }

    #[test]
    fn test_accel_statistics_operation_names() {
        let mut s = AccelStatistics::new();
        let now = Instant::now();
        s.record_operation("alpha", Duration::from_millis(1), now);
        s.record_operation("beta", Duration::from_millis(1), now);
        let names = s.operation_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
    }
}
