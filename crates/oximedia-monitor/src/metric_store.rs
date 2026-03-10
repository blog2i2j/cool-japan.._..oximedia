//! In-memory metric store with type-tagged samples and pruning support.
#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// The kind of metric and its canonical unit label.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// CPU utilisation percentage (0–100).
    CpuPercent,
    /// Memory used in bytes.
    MemoryBytes,
    /// Disk I/O in bytes per second.
    DiskIoBytesPerSec,
    /// Network throughput in bits per second.
    NetworkBitsPerSec,
    /// Encoding frames per second.
    EncodeFps,
    /// Arbitrary floating-point gauge with a user-defined unit label.
    Custom(String),
}

impl MetricType {
    /// Return the unit string for this metric type.
    #[must_use]
    pub fn unit(&self) -> &str {
        match self {
            Self::CpuPercent => "%",
            Self::MemoryBytes => "bytes",
            Self::DiskIoBytesPerSec => "bytes/s",
            Self::NetworkBitsPerSec => "bits/s",
            Self::EncodeFps => "fps",
            Self::Custom(u) => u.as_str(),
        }
    }
}

/// A single recorded metric sample.
#[derive(Clone, Debug)]
pub struct MetricSample {
    /// The numeric value.
    pub value: f64,
    /// Wall-clock instant at which this sample was recorded.
    pub recorded_at: Instant,
}

impl MetricSample {
    /// Create a new sample with the current time.
    #[must_use]
    pub fn now(value: f64) -> Self {
        Self {
            value,
            recorded_at: Instant::now(),
        }
    }

    /// Age of this sample in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn age_ms(&self) -> f64 {
        self.recorded_at.elapsed().as_secs_f64() * 1_000.0
    }

    /// `true` if this sample is older than `max_age`.
    #[must_use]
    pub fn is_expired(&self, max_age: Duration) -> bool {
        self.recorded_at.elapsed() > max_age
    }
}

/// A bounded, per-metric-type time-series store.
#[derive(Debug)]
pub struct MetricStore {
    data: HashMap<MetricType, Vec<MetricSample>>,
    /// Maximum number of samples kept per metric type.
    capacity: usize,
}

impl MetricStore {
    /// Create a new store with the given per-metric capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::new(),
            capacity: capacity.max(1),
        }
    }

    /// Record a value for a metric type, evicting the oldest entry if necessary.
    pub fn record(&mut self, metric: MetricType, value: f64) {
        let samples = self.data.entry(metric).or_default();
        if samples.len() >= self.capacity {
            samples.remove(0);
        }
        samples.push(MetricSample::now(value));
    }

    /// Return the most-recently recorded sample for the given type, or `None`.
    #[must_use]
    pub fn latest(&self, metric: &MetricType) -> Option<&MetricSample> {
        self.data.get(metric)?.last()
    }

    /// Return the full history slice for the given type (oldest first).
    pub fn history(&self, metric: &MetricType) -> &[MetricSample] {
        self.data.get(metric).map_or(&[], Vec::as_slice)
    }

    /// Remove all samples older than `max_age` from every metric series.
    pub fn prune_old(&mut self, max_age: Duration) {
        for samples in self.data.values_mut() {
            samples.retain(|s| !s.is_expired(max_age));
        }
    }

    /// Total number of samples stored across all metric types.
    pub fn total_sample_count(&self) -> usize {
        self.data.values().map(Vec::len).sum()
    }

    /// Number of distinct metric types tracked.
    #[must_use]
    pub fn metric_count(&self) -> usize {
        self.data.len()
    }

    /// Clear all stored samples.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── MetricType ───────────────────────────────────────────────────────────

    #[test]
    fn metric_type_cpu_unit() {
        assert_eq!(MetricType::CpuPercent.unit(), "%");
    }

    #[test]
    fn metric_type_memory_unit() {
        assert_eq!(MetricType::MemoryBytes.unit(), "bytes");
    }

    #[test]
    fn metric_type_disk_io_unit() {
        assert_eq!(MetricType::DiskIoBytesPerSec.unit(), "bytes/s");
    }

    #[test]
    fn metric_type_network_unit() {
        assert_eq!(MetricType::NetworkBitsPerSec.unit(), "bits/s");
    }

    #[test]
    fn metric_type_encode_fps_unit() {
        assert_eq!(MetricType::EncodeFps.unit(), "fps");
    }

    #[test]
    fn metric_type_custom_unit() {
        let t = MetricType::Custom("req/s".to_string());
        assert_eq!(t.unit(), "req/s");
    }

    // ── MetricSample ─────────────────────────────────────────────────────────

    #[test]
    fn metric_sample_age_ms_non_negative() {
        let s = MetricSample::now(42.0);
        assert!(s.age_ms() >= 0.0);
    }

    #[test]
    fn metric_sample_not_expired_immediately() {
        let s = MetricSample::now(1.0);
        assert!(!s.is_expired(Duration::from_secs(60)));
    }

    // ── MetricStore ──────────────────────────────────────────────────────────

    #[test]
    fn store_starts_empty() {
        let s = MetricStore::new(10);
        assert_eq!(s.total_sample_count(), 0);
        assert_eq!(s.metric_count(), 0);
    }

    #[test]
    fn store_record_and_latest() {
        let mut s = MetricStore::new(5);
        s.record(MetricType::CpuPercent, 45.0);
        let latest = s
            .latest(&MetricType::CpuPercent)
            .expect("latest should succeed");
        assert!((latest.value - 45.0).abs() < 1e-9);
    }

    #[test]
    fn store_history_returns_all_samples() {
        let mut s = MetricStore::new(10);
        s.record(MetricType::EncodeFps, 30.0);
        s.record(MetricType::EncodeFps, 31.0);
        s.record(MetricType::EncodeFps, 29.0);
        assert_eq!(s.history(&MetricType::EncodeFps).len(), 3);
    }

    #[test]
    fn store_evicts_oldest_when_at_capacity() {
        let mut s = MetricStore::new(2);
        s.record(MetricType::MemoryBytes, 100.0);
        s.record(MetricType::MemoryBytes, 200.0);
        s.record(MetricType::MemoryBytes, 300.0); // evicts 100.0
        let h = s.history(&MetricType::MemoryBytes);
        assert_eq!(h.len(), 2);
        assert!((h[0].value - 200.0).abs() < 1e-9);
    }

    #[test]
    fn store_latest_none_for_unrecorded_metric() {
        let s = MetricStore::new(5);
        assert!(s.latest(&MetricType::CpuPercent).is_none());
    }

    #[test]
    fn store_prune_old_removes_expired() {
        let mut s = MetricStore::new(10);
        s.record(MetricType::CpuPercent, 50.0);
        // Prune with a zero-duration max age — everything is expired
        s.prune_old(Duration::ZERO);
        assert_eq!(s.history(&MetricType::CpuPercent).len(), 0);
    }

    #[test]
    fn store_total_sample_count_across_types() {
        let mut s = MetricStore::new(10);
        s.record(MetricType::CpuPercent, 1.0);
        s.record(MetricType::MemoryBytes, 2.0);
        s.record(MetricType::MemoryBytes, 3.0);
        assert_eq!(s.total_sample_count(), 3);
    }

    #[test]
    fn store_clear_removes_all() {
        let mut s = MetricStore::new(10);
        s.record(MetricType::EncodeFps, 60.0);
        s.clear();
        assert_eq!(s.total_sample_count(), 0);
    }

    #[test]
    fn store_metric_count_tracks_distinct_types() {
        let mut s = MetricStore::new(5);
        s.record(MetricType::CpuPercent, 1.0);
        s.record(MetricType::MemoryBytes, 2.0);
        assert_eq!(s.metric_count(), 2);
    }
}
