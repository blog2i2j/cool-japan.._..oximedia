#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
//! Storage metrics collection and aggregation.
//!
//! Provides counters, gauges, and histogram-style metrics for monitoring
//! storage operations (uploads, downloads, deletes) including byte counts,
//! operation latencies, and error rates.

use std::collections::HashMap;
use std::time::Duration;

/// A monotonically increasing counter.
#[derive(Debug, Clone)]
pub struct Counter {
    /// Counter name.
    pub name: String,
    /// Current value.
    value: u64,
}

impl Counter {
    /// Create a counter starting at zero.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: 0,
        }
    }

    /// Increment by one.
    pub fn inc(&mut self) {
        self.value += 1;
    }

    /// Increment by an arbitrary amount.
    pub fn inc_by(&mut self, n: u64) {
        self.value += n;
    }

    /// Current counter value.
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Reset to zero (useful in test scenarios).
    pub fn reset(&mut self) {
        self.value = 0;
    }
}

/// A gauge that can go up or down.
#[derive(Debug, Clone)]
pub struct Gauge {
    /// Gauge name.
    pub name: String,
    /// Current value.
    value: f64,
}

impl Gauge {
    /// Create a gauge at zero.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: 0.0,
        }
    }

    /// Set the gauge to a specific value.
    pub fn set(&mut self, v: f64) {
        self.value = v;
    }

    /// Increment by a delta.
    pub fn add(&mut self, delta: f64) {
        self.value += delta;
    }

    /// Current gauge value.
    pub fn value(&self) -> f64 {
        self.value
    }
}

/// A simple histogram that records values and computes basic statistics.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Histogram name.
    pub name: String,
    /// All recorded values.
    values: Vec<f64>,
}

impl Histogram {
    /// Create an empty histogram.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
        }
    }

    /// Record a value.
    pub fn observe(&mut self, v: f64) {
        self.values.push(v);
    }

    /// Number of observations.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Sum of all observations.
    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    /// Mean of all observations (returns 0.0 if empty).
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.sum() / self.values.len() as f64
    }

    /// Minimum observed value.
    pub fn min(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::min)
    }

    /// Maximum observed value.
    pub fn max(&self) -> Option<f64> {
        self.values.iter().copied().reduce(f64::max)
    }

    /// Approximate percentile (0.0 to 1.0). Returns `None` if empty.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        if self.values.is_empty() || !(0.0..=1.0).contains(&p) {
            return None;
        }
        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Reset all observations.
    pub fn reset(&mut self) {
        self.values.clear();
    }
}

/// Kind of storage operation being tracked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationKind {
    /// Object upload.
    Upload,
    /// Object download.
    Download,
    /// Object deletion.
    Delete,
    /// Object listing.
    List,
    /// Metadata fetch.
    HeadObject,
    /// Object copy.
    Copy,
}

impl std::fmt::Display for OperationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Upload => write!(f, "upload"),
            Self::Download => write!(f, "download"),
            Self::Delete => write!(f, "delete"),
            Self::List => write!(f, "list"),
            Self::HeadObject => write!(f, "head"),
            Self::Copy => write!(f, "copy"),
        }
    }
}

/// Aggregated metrics for all storage operations.
#[derive(Debug)]
pub struct StorageMetrics {
    /// Operation counters keyed by kind.
    op_counts: HashMap<OperationKind, Counter>,
    /// Error counters keyed by kind.
    error_counts: HashMap<OperationKind, Counter>,
    /// Bytes transferred gauges keyed by kind.
    bytes_transferred: HashMap<OperationKind, Counter>,
    /// Latency histograms keyed by kind (values in milliseconds).
    latencies: HashMap<OperationKind, Histogram>,
}

impl StorageMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            op_counts: HashMap::new(),
            error_counts: HashMap::new(),
            bytes_transferred: HashMap::new(),
            latencies: HashMap::new(),
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self, kind: OperationKind, bytes: u64, latency: Duration) {
        self.op_counts
            .entry(kind)
            .or_insert_with(|| Counter::new(format!("{kind}_ops")))
            .inc();
        self.bytes_transferred
            .entry(kind)
            .or_insert_with(|| Counter::new(format!("{kind}_bytes")))
            .inc_by(bytes);
        let ms = latency.as_secs_f64() * 1000.0;
        self.latencies
            .entry(kind)
            .or_insert_with(|| Histogram::new(format!("{kind}_latency_ms")))
            .observe(ms);
    }

    /// Record a failed operation.
    pub fn record_error(&mut self, kind: OperationKind) {
        self.error_counts
            .entry(kind)
            .or_insert_with(|| Counter::new(format!("{kind}_errors")))
            .inc();
    }

    /// Total operations for a given kind.
    pub fn total_ops(&self, kind: OperationKind) -> u64 {
        self.op_counts.get(&kind).map_or(0, Counter::value)
    }

    /// Total errors for a given kind.
    pub fn total_errors(&self, kind: OperationKind) -> u64 {
        self.error_counts.get(&kind).map_or(0, Counter::value)
    }

    /// Total bytes transferred for a given kind.
    pub fn total_bytes(&self, kind: OperationKind) -> u64 {
        self.bytes_transferred.get(&kind).map_or(0, Counter::value)
    }

    /// Mean latency in milliseconds for a given kind.
    pub fn mean_latency_ms(&self, kind: OperationKind) -> f64 {
        self.latencies.get(&kind).map_or(0.0, Histogram::mean)
    }

    /// P99 latency in milliseconds for a given kind.
    pub fn p99_latency_ms(&self, kind: OperationKind) -> Option<f64> {
        self.latencies.get(&kind).and_then(|h| h.percentile(0.99))
    }

    /// Error rate (errors / total ops) for a given kind. Returns 0.0 if no ops.
    pub fn error_rate(&self, kind: OperationKind) -> f64 {
        let ops = self.total_ops(kind) + self.total_errors(kind);
        if ops == 0 {
            return 0.0;
        }
        self.total_errors(kind) as f64 / ops as f64
    }

    /// Generate a summary report.
    pub fn summary(&self) -> MetricsSummary {
        let mut total_ops = 0u64;
        let mut total_errors = 0u64;
        let mut total_bytes = 0u64;
        for c in self.op_counts.values() {
            total_ops += c.value();
        }
        for c in self.error_counts.values() {
            total_errors += c.value();
        }
        for c in self.bytes_transferred.values() {
            total_bytes += c.value();
        }
        MetricsSummary {
            total_ops,
            total_errors,
            total_bytes,
        }
    }
}

impl Default for StorageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary snapshot of all metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Total successful operations across all kinds.
    pub total_ops: u64,
    /// Total errors across all kinds.
    pub total_errors: u64,
    /// Total bytes transferred across all kinds.
    pub total_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Counter ────────────────────────────────────────────────────────────

    #[test]
    fn test_counter_new() {
        let c = Counter::new("test");
        assert_eq!(c.value(), 0);
        assert_eq!(c.name, "test");
    }

    #[test]
    fn test_counter_inc() {
        let mut c = Counter::new("ops");
        c.inc();
        c.inc();
        assert_eq!(c.value(), 2);
    }

    #[test]
    fn test_counter_inc_by() {
        let mut c = Counter::new("bytes");
        c.inc_by(1024);
        c.inc_by(2048);
        assert_eq!(c.value(), 3072);
    }

    #[test]
    fn test_counter_reset() {
        let mut c = Counter::new("x");
        c.inc_by(100);
        c.reset();
        assert_eq!(c.value(), 0);
    }

    // ── Gauge ──────────────────────────────────────────────────────────────

    #[test]
    fn test_gauge_set_and_get() {
        let mut g = Gauge::new("connections");
        g.set(42.0);
        assert!((g.value() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gauge_add() {
        let mut g = Gauge::new("inflight");
        g.add(10.0);
        g.add(-3.0);
        assert!((g.value() - 7.0).abs() < f64::EPSILON);
    }

    // ── Histogram ──────────────────────────────────────────────────────────

    #[test]
    fn test_histogram_empty() {
        let h = Histogram::new("lat");
        assert_eq!(h.count(), 0);
        assert!((h.mean() - 0.0).abs() < f64::EPSILON);
        assert!(h.min().is_none());
    }

    #[test]
    fn test_histogram_observe() {
        let mut h = Histogram::new("lat");
        h.observe(10.0);
        h.observe(20.0);
        h.observe(30.0);
        assert_eq!(h.count(), 3);
        assert!((h.sum() - 60.0).abs() < f64::EPSILON);
        assert!((h.mean() - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_min_max() {
        let mut h = Histogram::new("lat");
        h.observe(5.0);
        h.observe(100.0);
        h.observe(50.0);
        assert!((h.min().expect("min should exist") - 5.0).abs() < f64::EPSILON);
        assert!((h.max().expect("max should exist") - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_percentile() {
        let mut h = Histogram::new("lat");
        for i in 1..=100 {
            h.observe(i as f64);
        }
        let p50 = h.percentile(0.5).expect("percentile should succeed");
        assert!((p50 - 50.0).abs() < 2.0); // approximate
        let p99 = h.percentile(0.99).expect("percentile should succeed");
        assert!(p99 >= 98.0);
    }

    #[test]
    fn test_histogram_percentile_invalid() {
        let h = Histogram::new("lat");
        assert!(h.percentile(0.5).is_none());
    }

    // ── OperationKind ──────────────────────────────────────────────────────

    #[test]
    fn test_operation_kind_display() {
        assert_eq!(OperationKind::Upload.to_string(), "upload");
        assert_eq!(OperationKind::Download.to_string(), "download");
        assert_eq!(OperationKind::Delete.to_string(), "delete");
    }

    // ── StorageMetrics ─────────────────────────────────────────────────────

    #[test]
    fn test_metrics_record_success() {
        let mut m = StorageMetrics::new();
        m.record_success(OperationKind::Upload, 1024, Duration::from_millis(50));
        assert_eq!(m.total_ops(OperationKind::Upload), 1);
        assert_eq!(m.total_bytes(OperationKind::Upload), 1024);
    }

    #[test]
    fn test_metrics_record_error() {
        let mut m = StorageMetrics::new();
        m.record_error(OperationKind::Download);
        m.record_error(OperationKind::Download);
        assert_eq!(m.total_errors(OperationKind::Download), 2);
    }

    #[test]
    fn test_metrics_error_rate() {
        let mut m = StorageMetrics::new();
        m.record_success(OperationKind::Upload, 0, Duration::from_millis(10));
        m.record_success(OperationKind::Upload, 0, Duration::from_millis(10));
        m.record_error(OperationKind::Upload);
        // 2 successes + 1 error = 1/3 error rate
        let rate = m.error_rate(OperationKind::Upload);
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_metrics_error_rate_no_ops() {
        let m = StorageMetrics::new();
        assert!((m.error_rate(OperationKind::Delete) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_mean_latency() {
        let mut m = StorageMetrics::new();
        m.record_success(OperationKind::List, 0, Duration::from_millis(100));
        m.record_success(OperationKind::List, 0, Duration::from_millis(200));
        let mean = m.mean_latency_ms(OperationKind::List);
        assert!((mean - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_metrics_summary() {
        let mut m = StorageMetrics::new();
        m.record_success(OperationKind::Upload, 100, Duration::from_millis(10));
        m.record_success(OperationKind::Download, 200, Duration::from_millis(20));
        m.record_error(OperationKind::Delete);
        let s = m.summary();
        assert_eq!(s.total_ops, 2);
        assert_eq!(s.total_errors, 1);
        assert_eq!(s.total_bytes, 300);
    }
}
