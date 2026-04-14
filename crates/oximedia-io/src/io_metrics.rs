//! I/O operation metrics: latency, throughput, error rate, and queue depth.
//!
//! Provides [`IoMetrics`] for collecting real-time statistics about I/O
//! operations in the media pipeline.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Six latency buckets covering sub-microsecond to >10 ms.
#[derive(Debug, Clone, Default)]
pub struct LatencyBucket {
    /// Operations completing in < 1 µs.
    pub sub_us: u64,
    /// Operations completing in 1–10 µs.
    pub us_1_10: u64,
    /// Operations completing in 10–100 µs.
    pub us_10_100: u64,
    /// Operations completing in 100 µs – 1 ms.
    pub us_100_1000: u64,
    /// Operations completing in 1–10 ms.
    pub ms_1_10: u64,
    /// Operations completing in >= 10 ms.
    pub ms_10_plus: u64,
}

impl LatencyBucket {
    /// Record a latency observation.
    pub fn observe(&mut self, d: Duration) {
        let nanos = d.as_nanos();
        if nanos < 1_000 {
            self.sub_us = self.sub_us.saturating_add(1);
        } else if nanos < 10_000 {
            self.us_1_10 = self.us_1_10.saturating_add(1);
        } else if nanos < 100_000 {
            self.us_10_100 = self.us_10_100.saturating_add(1);
        } else if nanos < 1_000_000 {
            self.us_100_1000 = self.us_100_1000.saturating_add(1);
        } else if nanos < 10_000_000 {
            self.ms_1_10 = self.ms_1_10.saturating_add(1);
        } else {
            self.ms_10_plus = self.ms_10_plus.saturating_add(1);
        }
    }

    /// Total observations recorded.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.sub_us
            .saturating_add(self.us_1_10)
            .saturating_add(self.us_10_100)
            .saturating_add(self.us_100_1000)
            .saturating_add(self.ms_1_10)
            .saturating_add(self.ms_10_plus)
    }
}

/// Comprehensive I/O operation metrics collector.
#[derive(Debug, Clone, Default)]
pub struct IoMetrics {
    /// Total bytes successfully read.
    pub read_bytes: u64,
    /// Total number of read operations.
    pub read_ops: u64,
    /// Number of failed read operations.
    pub read_errors: u64,
    /// Latency distribution for read operations.
    pub read_latency: LatencyBucket,
    /// Total bytes successfully written.
    pub write_bytes: u64,
    /// Total number of write operations.
    pub write_ops: u64,
    /// Number of failed write operations.
    pub write_errors: u64,
    /// Latency distribution for write operations.
    pub write_latency: LatencyBucket,
    /// Total number of seek operations.
    pub seek_ops: u64,
    /// Cumulative absolute seek distance in bytes.
    pub seek_distance: u64,
    /// Number of failed seek operations.
    pub seek_errors: u64,
    /// Timestamp of the first recorded operation.
    pub first_op_ns: Option<u64>,
}

impl IoMetrics {
    /// Create a new, zeroed metrics instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a read operation.
    pub fn record_read(&mut self, bytes: u64, success: bool) {
        self.touch();
        self.read_ops = self.read_ops.saturating_add(1);
        if success {
            self.read_bytes = self.read_bytes.saturating_add(bytes);
        } else {
            self.read_errors = self.read_errors.saturating_add(1);
        }
    }

    /// Record a read operation with latency.
    pub fn record_read_timed(&mut self, bytes: u64, success: bool, latency: Duration) {
        self.record_read(bytes, success);
        self.read_latency.observe(latency);
    }

    /// Record a write operation.
    pub fn record_write(&mut self, bytes: u64, success: bool) {
        self.touch();
        self.write_ops = self.write_ops.saturating_add(1);
        if success {
            self.write_bytes = self.write_bytes.saturating_add(bytes);
        } else {
            self.write_errors = self.write_errors.saturating_add(1);
        }
    }

    /// Record a write operation with latency.
    pub fn record_write_timed(&mut self, bytes: u64, success: bool, latency: Duration) {
        self.record_write(bytes, success);
        self.write_latency.observe(latency);
    }

    /// Record a seek operation.
    pub fn record_seek(&mut self, distance: u64, success: bool) {
        self.touch();
        self.seek_ops = self.seek_ops.saturating_add(1);
        if success {
            self.seek_distance = self.seek_distance.saturating_add(distance);
        } else {
            self.seek_errors = self.seek_errors.saturating_add(1);
        }
    }

    /// Return the total number of errors across all operation types.
    #[must_use]
    pub fn total_errors(&self) -> u64 {
        self.read_errors
            .saturating_add(self.write_errors)
            .saturating_add(self.seek_errors)
    }

    /// Return the total bytes transferred (read + written).
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.read_bytes.saturating_add(self.write_bytes)
    }

    /// Merge another `IoMetrics` instance into this one.
    pub fn merge(&mut self, other: &IoMetrics) {
        self.read_bytes = self.read_bytes.saturating_add(other.read_bytes);
        self.read_ops = self.read_ops.saturating_add(other.read_ops);
        self.read_errors = self.read_errors.saturating_add(other.read_errors);
        self.write_bytes = self.write_bytes.saturating_add(other.write_bytes);
        self.write_ops = self.write_ops.saturating_add(other.write_ops);
        self.write_errors = self.write_errors.saturating_add(other.write_errors);
        self.seek_ops = self.seek_ops.saturating_add(other.seek_ops);
        self.seek_distance = self.seek_distance.saturating_add(other.seek_distance);
        self.seek_errors = self.seek_errors.saturating_add(other.seek_errors);
        if self.first_op_ns.is_none() {
            self.first_op_ns = other.first_op_ns;
        }
    }

    fn touch(&mut self) {
        if self.first_op_ns.is_none() {
            let epoch_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            self.first_op_ns = Some(epoch_ns);
        }
    }
}

/// A simple RAII timer for measuring I/O operation latency.
pub struct IoTimer {
    start: Instant,
}

impl IoTimer {
    /// Start a new timer.
    #[must_use]
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    /// Return the elapsed time since construction.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_read() {
        let mut m = IoMetrics::new();
        m.record_read(1024, true);
        assert_eq!(m.read_bytes, 1024);
        assert_eq!(m.read_ops, 1);
        assert_eq!(m.read_errors, 0);
    }

    #[test]
    fn test_record_read_failure() {
        let mut m = IoMetrics::new();
        m.record_read(0, false);
        assert_eq!(m.read_bytes, 0);
        assert_eq!(m.read_errors, 1);
    }

    #[test]
    fn test_total_bytes() {
        let mut m = IoMetrics::new();
        m.record_read(100, true);
        m.record_write(200, true);
        assert_eq!(m.total_bytes(), 300);
    }

    #[test]
    fn test_total_errors() {
        let mut m = IoMetrics::new();
        m.record_read(0, false);
        m.record_write(0, false);
        m.record_seek(0, false);
        assert_eq!(m.total_errors(), 3);
    }

    #[test]
    fn test_merge() {
        let mut a = IoMetrics::new();
        let mut b = IoMetrics::new();
        a.record_read(100, true);
        b.record_read(200, true);
        a.merge(&b);
        assert_eq!(a.read_bytes, 300);
    }

    #[test]
    fn test_latency_bucket_observe() {
        let mut bucket = LatencyBucket::default();
        bucket.observe(Duration::from_nanos(500));
        bucket.observe(Duration::from_nanos(5_000));
        assert_eq!(bucket.sub_us, 1);
        assert_eq!(bucket.us_1_10, 1);
        assert_eq!(bucket.total(), 2);
    }

    #[test]
    fn test_io_timer() {
        let timer = IoTimer::start();
        std::thread::sleep(Duration::from_millis(1));
        let elapsed = timer.elapsed();
        assert!(elapsed >= Duration::from_millis(1));
    }
}
