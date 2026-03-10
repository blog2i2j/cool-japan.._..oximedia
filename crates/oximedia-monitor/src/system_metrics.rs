//! System-level metrics collection for CPU, memory, disk, and network.
//!
//! Provides typed metric samples with criticality thresholds and a
//! ring-buffer history collector.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

/// The kind of system metric being measured.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SystemMetric {
    /// CPU utilisation as a percentage (0–100).
    CpuPercent,
    /// Memory utilisation as a percentage (0–100).
    MemPercent,
    /// Disk utilisation as a percentage (0–100).
    DiskPercent,
    /// Network throughput in Mbps.
    NetworkMbps,
}

impl SystemMetric {
    /// Returns the SI unit string for the metric.
    #[must_use]
    pub fn unit(&self) -> &'static str {
        match self {
            Self::CpuPercent | Self::MemPercent | Self::DiskPercent => "%",
            Self::NetworkMbps => "Mbps",
        }
    }

    /// Default critical threshold above which the metric is considered critical.
    #[must_use]
    pub fn default_critical_threshold(&self) -> f64 {
        match self {
            Self::CpuPercent => 95.0,
            Self::MemPercent => 90.0,
            Self::DiskPercent => 85.0,
            Self::NetworkMbps => 900.0,
        }
    }

    /// Short display name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuPercent => "cpu_percent",
            Self::MemPercent => "mem_percent",
            Self::DiskPercent => "disk_percent",
            Self::NetworkMbps => "network_mbps",
        }
    }
}

/// A single timestamped measurement of a system metric.
#[derive(Debug, Clone)]
pub struct SystemMetricSample {
    /// Which metric this sample represents.
    pub metric: SystemMetric,
    /// Measured value.
    pub value: f64,
    /// Unix timestamp in seconds when the sample was taken.
    pub timestamp_secs: u64,
    /// Custom critical threshold override. If `None` the metric's default is used.
    pub critical_threshold: Option<f64>,
}

impl SystemMetricSample {
    /// Create a new sample with the current monotonic-clock seconds.
    #[must_use]
    pub fn new(metric: SystemMetric, value: f64, timestamp_secs: u64) -> Self {
        Self {
            metric,
            value,
            timestamp_secs,
            critical_threshold: None,
        }
    }

    /// Create a sample with a custom critical threshold.
    #[must_use]
    pub fn with_threshold(
        metric: SystemMetric,
        value: f64,
        timestamp_secs: u64,
        threshold: f64,
    ) -> Self {
        Self {
            metric,
            value,
            timestamp_secs,
            critical_threshold: Some(threshold),
        }
    }

    /// Returns `true` when the value exceeds the critical threshold.
    #[must_use]
    pub fn is_critical(&self) -> bool {
        let threshold = self
            .critical_threshold
            .unwrap_or_else(|| self.metric.default_critical_threshold());
        self.value >= threshold
    }

    /// The effective critical threshold for this sample.
    #[must_use]
    pub fn effective_threshold(&self) -> f64 {
        self.critical_threshold
            .unwrap_or_else(|| self.metric.default_critical_threshold())
    }
}

/// Collects a rolling history of system metric samples.
#[derive(Debug)]
pub struct SystemMetricsCollector {
    /// Maximum number of samples retained per metric kind.
    capacity: usize,
    cpu: VecDeque<SystemMetricSample>,
    mem: VecDeque<SystemMetricSample>,
    disk: VecDeque<SystemMetricSample>,
    net: VecDeque<SystemMetricSample>,
}

impl SystemMetricsCollector {
    /// Create a new collector retaining up to `capacity` samples per metric.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cpu: VecDeque::with_capacity(capacity),
            mem: VecDeque::with_capacity(capacity),
            disk: VecDeque::with_capacity(capacity),
            net: VecDeque::with_capacity(capacity),
        }
    }

    /// Push a sample into the appropriate history buffer.
    pub fn record(&mut self, sample: SystemMetricSample) {
        let capacity = self.capacity;
        let buf = self.buf_for_mut(&sample.metric);
        if buf.len() >= capacity {
            buf.pop_front();
        }
        buf.push_back(sample);
    }

    /// Returns the most-recently recorded sample for the given metric, if any.
    #[must_use]
    pub fn current(&self, metric: &SystemMetric) -> Option<&SystemMetricSample> {
        self.buf_for(metric).back()
    }

    /// Returns up to `n` most-recent samples for the given metric (oldest first).
    #[must_use]
    pub fn history(&self, metric: &SystemMetric, n: usize) -> Vec<&SystemMetricSample> {
        let buf = self.buf_for(metric);
        let skip = buf.len().saturating_sub(n);
        buf.iter().skip(skip).collect()
    }

    /// Average value over the full stored history for a metric.
    ///
    /// Returns `None` if no samples have been recorded.
    #[must_use]
    pub fn average(&self, metric: &SystemMetric) -> Option<f64> {
        let buf = self.buf_for(metric);
        if buf.is_empty() {
            return None;
        }
        let sum: f64 = buf.iter().map(|s| s.value).sum();
        Some(sum / buf.len() as f64)
    }

    /// Returns `true` if the most recent sample for a metric is critical.
    #[must_use]
    pub fn is_critical(&self, metric: &SystemMetric) -> bool {
        self.current(metric)
            .is_some_and(SystemMetricSample::is_critical)
    }

    /// Total number of samples stored across all metrics.
    #[must_use]
    pub fn total_samples(&self) -> usize {
        self.cpu.len() + self.mem.len() + self.disk.len() + self.net.len()
    }

    fn buf_for(&self, metric: &SystemMetric) -> &VecDeque<SystemMetricSample> {
        match metric {
            SystemMetric::CpuPercent => &self.cpu,
            SystemMetric::MemPercent => &self.mem,
            SystemMetric::DiskPercent => &self.disk,
            SystemMetric::NetworkMbps => &self.net,
        }
    }

    fn buf_for_mut(&mut self, metric: &SystemMetric) -> &mut VecDeque<SystemMetricSample> {
        match metric {
            SystemMetric::CpuPercent => &mut self.cpu,
            SystemMetric::MemPercent => &mut self.mem,
            SystemMetric::DiskPercent => &mut self.disk,
            SystemMetric::NetworkMbps => &mut self.net,
        }
    }
}

impl Default for SystemMetricsCollector {
    fn default() -> Self {
        Self::new(60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(metric: SystemMetric, value: f64, ts: u64) -> SystemMetricSample {
        SystemMetricSample::new(metric, value, ts)
    }

    #[test]
    fn test_system_metric_unit() {
        assert_eq!(SystemMetric::CpuPercent.unit(), "%");
        assert_eq!(SystemMetric::MemPercent.unit(), "%");
        assert_eq!(SystemMetric::DiskPercent.unit(), "%");
        assert_eq!(SystemMetric::NetworkMbps.unit(), "Mbps");
    }

    #[test]
    fn test_system_metric_name() {
        assert_eq!(SystemMetric::CpuPercent.name(), "cpu_percent");
        assert_eq!(SystemMetric::MemPercent.name(), "mem_percent");
        assert_eq!(SystemMetric::DiskPercent.name(), "disk_percent");
        assert_eq!(SystemMetric::NetworkMbps.name(), "network_mbps");
    }

    #[test]
    fn test_system_metric_default_thresholds() {
        assert!(
            (SystemMetric::CpuPercent.default_critical_threshold() - 95.0).abs() < f64::EPSILON
        );
        assert!(
            (SystemMetric::MemPercent.default_critical_threshold() - 90.0).abs() < f64::EPSILON
        );
        assert!(
            (SystemMetric::DiskPercent.default_critical_threshold() - 85.0).abs() < f64::EPSILON
        );
        assert!(
            (SystemMetric::NetworkMbps.default_critical_threshold() - 900.0).abs() < f64::EPSILON
        );
    }

    #[test]
    fn test_sample_is_critical_above_threshold() {
        let s = sample(SystemMetric::CpuPercent, 96.0, 1000);
        assert!(s.is_critical());
    }

    #[test]
    fn test_sample_is_not_critical_below_threshold() {
        let s = sample(SystemMetric::CpuPercent, 50.0, 1000);
        assert!(!s.is_critical());
    }

    #[test]
    fn test_sample_custom_threshold_critical() {
        let s = SystemMetricSample::with_threshold(
            SystemMetric::NetworkMbps,
            500.0,
            0,
            400.0, // custom threshold — 500 > 400
        );
        assert!(s.is_critical());
        assert!((s.effective_threshold() - 400.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sample_at_exact_threshold_is_critical() {
        let s = sample(SystemMetric::DiskPercent, 85.0, 0);
        assert!(s.is_critical());
    }

    #[test]
    fn test_collector_record_and_current() {
        let mut c = SystemMetricsCollector::new(10);
        c.record(sample(SystemMetric::CpuPercent, 42.0, 1));
        let cur = c
            .current(&SystemMetric::CpuPercent)
            .expect("current should succeed");
        assert!((cur.value - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_current_none_when_empty() {
        let c = SystemMetricsCollector::new(10);
        assert!(c.current(&SystemMetric::MemPercent).is_none());
    }

    #[test]
    fn test_collector_evicts_oldest_when_full() {
        let mut c = SystemMetricsCollector::new(3);
        for i in 0..5u64 {
            c.record(sample(SystemMetric::DiskPercent, i as f64 * 10.0, i));
        }
        // Should keep last 3: 20.0, 30.0, 40.0
        let hist = c.history(&SystemMetric::DiskPercent, 3);
        assert_eq!(hist.len(), 3);
        assert!((hist[0].value - 20.0).abs() < f64::EPSILON);
        assert!((hist[2].value - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_average() {
        let mut c = SystemMetricsCollector::new(10);
        c.record(sample(SystemMetric::NetworkMbps, 100.0, 1));
        c.record(sample(SystemMetric::NetworkMbps, 200.0, 2));
        c.record(sample(SystemMetric::NetworkMbps, 300.0, 3));
        let avg = c
            .average(&SystemMetric::NetworkMbps)
            .expect("average should succeed");
        assert!((avg - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_collector_average_none_when_empty() {
        let c = SystemMetricsCollector::new(10);
        assert!(c.average(&SystemMetric::CpuPercent).is_none());
    }

    #[test]
    fn test_collector_is_critical() {
        let mut c = SystemMetricsCollector::new(10);
        c.record(sample(SystemMetric::MemPercent, 91.0, 1));
        assert!(c.is_critical(&SystemMetric::MemPercent));
    }

    #[test]
    fn test_collector_is_not_critical_when_empty() {
        let c = SystemMetricsCollector::new(10);
        assert!(!c.is_critical(&SystemMetric::CpuPercent));
    }

    #[test]
    fn test_collector_total_samples() {
        let mut c = SystemMetricsCollector::new(10);
        c.record(sample(SystemMetric::CpuPercent, 10.0, 1));
        c.record(sample(SystemMetric::MemPercent, 20.0, 2));
        c.record(sample(SystemMetric::DiskPercent, 30.0, 3));
        assert_eq!(c.total_samples(), 3);
    }
}
