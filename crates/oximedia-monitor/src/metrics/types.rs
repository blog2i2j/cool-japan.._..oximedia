//! Metric type definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Metric name.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetricName(String);

impl MetricName {
    /// Create a new metric name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is invalid (empty or contains invalid characters).
    pub fn new(name: impl Into<String>) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() {
            return Err("Metric name cannot be empty".to_string());
        }

        // Validate metric name (alphanumeric, underscore, dot)
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            return Err(format!("Invalid metric name: {name}"));
        }

        Ok(Self(name))
    }

    /// Get the metric name as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for MetricName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metric labels (key-value pairs).
pub type MetricLabels = HashMap<String, String>;

/// Metric kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricKind {
    /// Counter (monotonically increasing value).
    Counter,
    /// Gauge (value that can go up or down).
    Gauge,
    /// Histogram (distribution of values).
    Histogram,
    /// Summary (percentiles over time).
    Summary,
}

/// A metric with name, kind, and labels.
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name.
    pub name: MetricName,
    /// Metric kind.
    pub kind: MetricKind,
    /// Metric labels.
    pub labels: MetricLabels,
    /// Help text.
    pub help: String,
}

impl Metric {
    /// Create a new metric.
    #[must_use]
    pub fn new(name: MetricName, kind: MetricKind, help: impl Into<String>) -> Self {
        Self {
            name,
            kind,
            labels: HashMap::new(),
            help: help.into(),
        }
    }

    /// Add a label to the metric.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    /// Add multiple labels to the metric.
    #[must_use]
    pub fn with_labels(mut self, labels: MetricLabels) -> Self {
        self.labels.extend(labels);
        self
    }
}

/// Counter metric (monotonically increasing).
#[derive(Debug, Clone)]
pub struct Counter {
    value: Arc<AtomicU64>,
}

impl Counter {
    /// Create a new counter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Increment the counter by 1.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a specific amount.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to zero.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// Gauge metric (value that can go up or down).
#[derive(Debug, Clone)]
pub struct Gauge {
    value: Arc<AtomicU64>,
}

impl Gauge {
    /// Create a new gauge.
    #[must_use]
    pub fn new() -> Self {
        Self {
            value: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: f64) {
        self.value.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Increment the gauge by 1.
    pub fn inc(&self) {
        self.add(1.0);
    }

    /// Decrement the gauge by 1.
    pub fn dec(&self) {
        self.sub(1.0);
    }

    /// Add to the gauge.
    pub fn add(&self, delta: f64) {
        loop {
            let current = self.value.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new_value = (current_f64 + delta).to_bits();
            if self
                .value
                .compare_exchange(current, new_value, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Subtract from the gauge.
    pub fn sub(&self, delta: f64) {
        self.add(-delta);
    }

    /// Get the current value.
    #[must_use]
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }
}

impl Default for Gauge {
    fn default() -> Self {
        Self::new()
    }
}

/// Histogram metric (distribution of values).
#[derive(Debug, Clone)]
pub struct Histogram {
    buckets: Vec<f64>,
    counts: Vec<Arc<AtomicU64>>,
    sum: Arc<AtomicU64>,
    count: Arc<AtomicU64>,
}

impl Histogram {
    /// Create a new histogram with the given buckets.
    #[must_use]
    pub fn new(buckets: Vec<f64>) -> Self {
        let mut sorted_buckets = buckets;
        sorted_buckets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let counts = sorted_buckets
            .iter()
            .map(|_| Arc::new(AtomicU64::new(0)))
            .collect();

        Self {
            buckets: sorted_buckets,
            counts,
            sum: Arc::new(AtomicU64::new(0)),
            count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Create a histogram with default buckets.
    #[must_use]
    pub fn default_buckets() -> Self {
        Self::new(vec![
            0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
        ])
    }

    /// Observe a value.
    pub fn observe(&self, value: f64) {
        // Update sum using CAS loop for correct float accumulation
        loop {
            let current = self.sum.load(Ordering::Relaxed);
            let new_sum = f64::from_bits(current) + value;
            if self
                .sum
                .compare_exchange(
                    current,
                    new_sum.to_bits(),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update bucket counts
        for (i, &bucket) in self.buckets.iter().enumerate() {
            if value <= bucket {
                self.counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get bucket counts.
    #[must_use]
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        self.buckets
            .iter()
            .zip(self.counts.iter())
            .map(|(&bucket, count)| (bucket, count.load(Ordering::Relaxed)))
            .collect()
    }

    /// Get the sum of all observed values.
    #[must_use]
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get the count of observations.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the average value.
    #[must_use]
    pub fn avg(&self) -> f64 {
        let count = self.count();
        if count == 0 {
            0.0
        } else {
            self.sum() / count as f64
        }
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::default_buckets()
    }
}

/// Summary metric (percentiles).
#[derive(Debug, Clone)]
pub struct Summary {
    values: Arc<parking_lot::RwLock<Vec<f64>>>,
    max_values: usize,
}

impl Summary {
    /// Create a new summary with a maximum number of values to keep.
    #[must_use]
    pub fn new(max_values: usize) -> Self {
        Self {
            values: Arc::new(parking_lot::RwLock::new(Vec::with_capacity(max_values))),
            max_values,
        }
    }

    /// Observe a value.
    pub fn observe(&self, value: f64) {
        let mut values = self.values.write();
        values.push(value);

        // Keep only the most recent values
        if values.len() > self.max_values {
            let drain_count = values.len() - self.max_values;
            values.drain(0..drain_count);
        }
    }

    /// Calculate a percentile (0.0 to 1.0).
    #[must_use]
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let values = self.values.read();
        if values.is_empty() {
            return None;
        }

        let mut sorted: Vec<f64> = values.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
        Some(sorted[idx])
    }

    /// Get the count of observations.
    #[must_use]
    pub fn count(&self) -> usize {
        self.values.read().len()
    }

    /// Get the sum of all values.
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.values.read().iter().sum()
    }

    /// Get the average value.
    #[must_use]
    pub fn avg(&self) -> f64 {
        let values = self.values.read();
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    /// Get minimum value.
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.values
            .read()
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }

    /// Get maximum value.
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.values
            .read()
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
    }
}

impl Default for Summary {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_name() {
        let name = MetricName::new("cpu.usage").expect("failed to create");
        assert_eq!(name.as_str(), "cpu.usage");
    }

    #[test]
    fn test_invalid_metric_name() {
        assert!(MetricName::new("").is_err());
        assert!(MetricName::new("cpu usage").is_err());
        assert!(MetricName::new("cpu:usage").is_err());
    }

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.add(5);
        assert_eq!(counter.get(), 6);

        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0.0);

        gauge.set(42.5);
        assert_eq!(gauge.get(), 42.5);

        gauge.inc();
        assert_eq!(gauge.get(), 43.5);

        gauge.dec();
        assert_eq!(gauge.get(), 42.5);

        gauge.add(10.0);
        assert_eq!(gauge.get(), 52.5);

        gauge.sub(2.5);
        assert_eq!(gauge.get(), 50.0);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new(vec![1.0, 5.0, 10.0]);

        hist.observe(0.5);
        hist.observe(2.0);
        hist.observe(7.0);
        hist.observe(15.0);

        assert_eq!(hist.count(), 4);
        assert_eq!(hist.sum(), 24.5);
        assert_eq!(hist.avg(), 6.125);

        let buckets = hist.buckets();
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0], (1.0, 1)); // values <= 1.0
        assert_eq!(buckets[1], (5.0, 2)); // values <= 5.0
        assert_eq!(buckets[2], (10.0, 3)); // values <= 10.0
    }

    #[test]
    fn test_summary() {
        let summary = Summary::new(100);

        for i in 1..=100 {
            summary.observe(i as f64);
        }

        assert_eq!(summary.count(), 100);
        assert_eq!(summary.min(), Some(1.0));
        assert_eq!(summary.max(), Some(100.0));
        assert_eq!(summary.avg(), 50.5);

        let p50 = summary.percentile(0.5).expect("percentile should succeed");
        assert!((p50 - 50.0).abs() < 1.0);

        let p95 = summary.percentile(0.95).expect("percentile should succeed");
        assert!((p95 - 95.0).abs() < 1.0);

        let p99 = summary.percentile(0.99).expect("percentile should succeed");
        assert!((p99 - 99.0).abs() < 1.0);
    }

    #[test]
    fn test_summary_max_values() {
        let summary = Summary::new(10);

        for i in 1..=20 {
            summary.observe(i as f64);
        }

        // Should only keep the last 10 values
        assert_eq!(summary.count(), 10);
        assert_eq!(summary.min(), Some(11.0));
        assert_eq!(summary.max(), Some(20.0));
    }

    #[test]
    fn test_metric_builder() {
        let metric = Metric::new(
            MetricName::new("request.duration").expect("failed to create"),
            MetricKind::Histogram,
            "Request duration in seconds",
        )
        .with_label("method", "GET")
        .with_label("path", "/api/metrics");

        assert_eq!(metric.name.as_str(), "request.duration");
        assert_eq!(metric.kind, MetricKind::Histogram);
        assert_eq!(metric.labels.get("method"), Some(&"GET".to_string()));
        assert_eq!(metric.labels.get("path"), Some(&"/api/metrics".to_string()));
    }
}
