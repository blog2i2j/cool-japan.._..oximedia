//! Cloud infrastructure monitoring and health tracking.

#![allow(dead_code)]

use std::collections::HashMap;

/// Observable cloud infrastructure metric.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CloudMetric {
    /// Round-trip latency in milliseconds.
    Latency,
    /// Service availability percentage.
    Availability,
    /// Data throughput in Mbps.
    Throughput,
    /// Request error rate percentage.
    ErrorRate,
}

impl CloudMetric {
    /// SI-style unit string for this metric.
    pub fn unit(&self) -> &'static str {
        match self {
            CloudMetric::Latency => "ms",
            CloudMetric::Availability => "%",
            CloudMetric::Throughput => "Mbps",
            CloudMetric::ErrorRate => "%",
        }
    }

    /// Upper threshold beyond which a sample is considered unhealthy.
    /// (For Availability and Throughput, lower than threshold is unhealthy —
    /// callers should interpret via `CloudMetricSample::is_healthy`.)
    pub fn healthy_threshold(&self) -> f64 {
        match self {
            CloudMetric::Latency => 200.0,     // ms — above this is bad
            CloudMetric::Availability => 99.9, // % — below this is bad
            CloudMetric::Throughput => 100.0,  // Mbps — below this is bad
            CloudMetric::ErrorRate => 1.0,     // % — above this is bad
        }
    }

    /// Returns `true` when *higher* values are good (Availability, Throughput).
    pub fn higher_is_better(&self) -> bool {
        matches!(self, CloudMetric::Availability | CloudMetric::Throughput)
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            CloudMetric::Latency => "Latency",
            CloudMetric::Availability => "Availability",
            CloudMetric::Throughput => "Throughput",
            CloudMetric::ErrorRate => "Error Rate",
        }
    }
}

/// A single timestamped metric observation.
#[derive(Debug, Clone)]
pub struct CloudMetricSample {
    /// The metric being measured.
    pub metric: CloudMetric,
    /// Observed value.
    pub value: f64,
    /// Unix timestamp (seconds) when this sample was taken.
    pub timestamp: i64,
    /// Optional region or endpoint tag.
    pub region: Option<String>,
}

impl CloudMetricSample {
    /// Create a new sample at the given timestamp.
    pub fn new(metric: CloudMetric, value: f64, timestamp: i64) -> Self {
        Self {
            metric,
            value,
            timestamp,
            region: None,
        }
    }

    /// Attach a region label.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Returns `true` when the sample is within healthy bounds for its metric.
    pub fn is_healthy(&self) -> bool {
        let threshold = self.metric.healthy_threshold();
        if self.metric.higher_is_better() {
            self.value >= threshold
        } else {
            self.value <= threshold
        }
    }
}

/// Collects `CloudMetricSample` observations and provides health summaries.
#[derive(Debug, Default)]
pub struct CloudMonitor {
    samples: Vec<CloudMetricSample>,
}

impl CloudMonitor {
    /// Create an empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new metric sample.
    pub fn record(&mut self, sample: CloudMetricSample) {
        self.samples.push(sample);
    }

    /// Produce a health summary across all recorded samples.
    pub fn health_summary(&self) -> HealthSummary {
        let total = self.samples.len();
        let healthy = self.samples.iter().filter(|s| s.is_healthy()).count();

        let mut per_metric: HashMap<String, (usize, usize)> = HashMap::new();
        for s in &self.samples {
            let entry = per_metric
                .entry(s.metric.display_name().to_string())
                .or_insert((0, 0));
            entry.0 += 1;
            if s.is_healthy() {
                entry.1 += 1;
            }
        }

        HealthSummary {
            total,
            healthy,
            per_metric,
        }
    }

    /// Return the most recent sample for a given metric, if any.
    pub fn latest(&self, metric: &CloudMetric) -> Option<&CloudMetricSample> {
        self.samples
            .iter()
            .filter(|s| &s.metric == metric)
            .max_by_key(|s| s.timestamp)
    }

    /// Average value across all samples for a given metric.
    #[allow(clippy::cast_precision_loss)]
    pub fn average(&self, metric: &CloudMetric) -> Option<f64> {
        let relevant: Vec<f64> = self
            .samples
            .iter()
            .filter(|s| &s.metric == metric)
            .map(|s| s.value)
            .collect();
        if relevant.is_empty() {
            None
        } else {
            Some(relevant.iter().sum::<f64>() / relevant.len() as f64)
        }
    }

    /// Total number of recorded samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Summary of monitored health state.
#[derive(Debug)]
pub struct HealthSummary {
    /// Total samples considered.
    pub total: usize,
    /// Number of healthy samples.
    pub healthy: usize,
    /// Per-metric (total, healthy) counts keyed by display name.
    pub per_metric: HashMap<String, (usize, usize)>,
}

impl HealthSummary {
    /// Overall health ratio in [0.0, 1.0]. Returns 1.0 when no samples exist.
    #[allow(clippy::cast_precision_loss)]
    pub fn health_ratio(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.healthy as f64 / self.total as f64
        }
    }

    /// Returns `true` when all recorded samples were healthy.
    pub fn is_all_healthy(&self) -> bool {
        self.healthy == self.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_unit() {
        assert_eq!(CloudMetric::Latency.unit(), "ms");
    }

    #[test]
    fn test_availability_unit() {
        assert_eq!(CloudMetric::Availability.unit(), "%");
    }

    #[test]
    fn test_throughput_unit() {
        assert_eq!(CloudMetric::Throughput.unit(), "Mbps");
    }

    #[test]
    fn test_error_rate_unit() {
        assert_eq!(CloudMetric::ErrorRate.unit(), "%");
    }

    #[test]
    fn test_higher_is_better_availability() {
        assert!(CloudMetric::Availability.higher_is_better());
    }

    #[test]
    fn test_higher_is_not_better_latency() {
        assert!(!CloudMetric::Latency.higher_is_better());
    }

    #[test]
    fn test_sample_healthy_low_latency() {
        let s = CloudMetricSample::new(CloudMetric::Latency, 50.0, 1000);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_sample_unhealthy_high_latency() {
        let s = CloudMetricSample::new(CloudMetric::Latency, 500.0, 1000);
        assert!(!s.is_healthy());
    }

    #[test]
    fn test_sample_healthy_high_availability() {
        let s = CloudMetricSample::new(CloudMetric::Availability, 99.95, 1000);
        assert!(s.is_healthy());
    }

    #[test]
    fn test_sample_unhealthy_low_availability() {
        let s = CloudMetricSample::new(CloudMetric::Availability, 98.0, 1000);
        assert!(!s.is_healthy());
    }

    #[test]
    fn test_sample_with_region() {
        let s =
            CloudMetricSample::new(CloudMetric::Throughput, 200.0, 1000).with_region("us-east-1");
        assert_eq!(s.region.as_deref(), Some("us-east-1"));
    }

    #[test]
    fn test_monitor_record_and_count() {
        let mut mon = CloudMonitor::new();
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 100.0, 1000));
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 150.0, 2000));
        assert_eq!(mon.sample_count(), 2);
    }

    #[test]
    fn test_monitor_latest() {
        let mut mon = CloudMonitor::new();
        mon.record(CloudMetricSample::new(CloudMetric::ErrorRate, 0.5, 1000));
        mon.record(CloudMetricSample::new(CloudMetric::ErrorRate, 0.8, 2000));
        let latest = mon
            .latest(&CloudMetric::ErrorRate)
            .expect("latest should be valid");
        assert!((latest.value - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_monitor_average() {
        let mut mon = CloudMonitor::new();
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 100.0, 1000));
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 200.0, 2000));
        let avg = mon
            .average(&CloudMetric::Latency)
            .expect("avg should be valid");
        assert!((avg - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_summary_all_healthy() {
        let mut mon = CloudMonitor::new();
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 50.0, 1000));
        mon.record(CloudMetricSample::new(
            CloudMetric::Availability,
            99.99,
            1000,
        ));
        let summary = mon.health_summary();
        assert!(summary.is_all_healthy());
        assert!((summary.health_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_summary_partial() {
        let mut mon = CloudMonitor::new();
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 50.0, 1000)); // healthy
        mon.record(CloudMetricSample::new(CloudMetric::Latency, 999.0, 2000)); // unhealthy
        let summary = mon.health_summary();
        assert!(!summary.is_all_healthy());
        assert!((summary.health_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_summary_empty_is_fully_healthy() {
        let mon = CloudMonitor::new();
        let summary = mon.health_summary();
        assert!((summary.health_ratio() - 1.0).abs() < f64::EPSILON);
    }
}
