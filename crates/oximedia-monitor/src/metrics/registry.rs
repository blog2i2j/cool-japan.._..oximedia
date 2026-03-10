//! Metric registry for storing and retrieving metrics.

use crate::error::{MonitorError, MonitorResult};
use crate::metrics::types::{
    Counter, Gauge, Histogram, Metric, MetricKind, MetricLabels, MetricName, Summary,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Metric value types.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Counter value.
    Counter(Counter),
    /// Gauge value.
    Gauge(Gauge),
    /// Histogram value.
    Histogram(Histogram),
    /// Summary value.
    Summary(Summary),
}

/// Metric registry.
#[derive(Clone)]
pub struct MetricRegistry {
    metrics: Arc<DashMap<String, (Metric, MetricValue)>>,
}

impl MetricRegistry {
    /// Create a new metric registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(DashMap::new()),
        }
    }

    /// Register a counter metric.
    ///
    /// # Errors
    ///
    /// Returns an error if a metric with the same name already exists.
    pub fn register_counter(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
    ) -> MonitorResult<Counter> {
        let key = Self::make_key(&name, &labels);
        let metric = Metric::new(name, MetricKind::Counter, help).with_labels(labels);
        let counter = Counter::new();

        if self
            .metrics
            .insert(key.clone(), (metric, MetricValue::Counter(counter.clone())))
            .is_some()
        {
            return Err(MonitorError::InvalidMetricName(format!(
                "Metric already exists: {key}"
            )));
        }

        Ok(counter)
    }

    /// Register a gauge metric.
    ///
    /// # Errors
    ///
    /// Returns an error if a metric with the same name already exists.
    pub fn register_gauge(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
    ) -> MonitorResult<Gauge> {
        let key = Self::make_key(&name, &labels);
        let metric = Metric::new(name, MetricKind::Gauge, help).with_labels(labels);
        let gauge = Gauge::new();

        if self
            .metrics
            .insert(key.clone(), (metric, MetricValue::Gauge(gauge.clone())))
            .is_some()
        {
            return Err(MonitorError::InvalidMetricName(format!(
                "Metric already exists: {key}"
            )));
        }

        Ok(gauge)
    }

    /// Register a histogram metric.
    ///
    /// # Errors
    ///
    /// Returns an error if a metric with the same name already exists.
    pub fn register_histogram(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
        buckets: Option<Vec<f64>>,
    ) -> MonitorResult<Histogram> {
        let key = Self::make_key(&name, &labels);
        let metric = Metric::new(name, MetricKind::Histogram, help).with_labels(labels);
        let histogram = if let Some(buckets) = buckets {
            Histogram::new(buckets)
        } else {
            Histogram::default_buckets()
        };

        if self
            .metrics
            .insert(
                key.clone(),
                (metric, MetricValue::Histogram(histogram.clone())),
            )
            .is_some()
        {
            return Err(MonitorError::InvalidMetricName(format!(
                "Metric already exists: {key}"
            )));
        }

        Ok(histogram)
    }

    /// Register a summary metric.
    ///
    /// # Errors
    ///
    /// Returns an error if a metric with the same name already exists.
    pub fn register_summary(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
        max_values: Option<usize>,
    ) -> MonitorResult<Summary> {
        let key = Self::make_key(&name, &labels);
        let metric = Metric::new(name, MetricKind::Summary, help).with_labels(labels);
        let summary = if let Some(max_values) = max_values {
            Summary::new(max_values)
        } else {
            Summary::default()
        };

        if self
            .metrics
            .insert(key.clone(), (metric, MetricValue::Summary(summary.clone())))
            .is_some()
        {
            return Err(MonitorError::InvalidMetricName(format!(
                "Metric already exists: {key}"
            )));
        }

        Ok(summary)
    }

    /// Get or create a counter metric.
    pub fn counter(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
    ) -> MonitorResult<Counter> {
        let key = Self::make_key(&name, &labels);

        if let Some(entry) = self.metrics.get(&key) {
            match &entry.1 {
                MetricValue::Counter(counter) => Ok(counter.clone()),
                _ => Err(MonitorError::InvalidMetricName(format!(
                    "Metric {key} exists but is not a counter"
                ))),
            }
        } else {
            self.register_counter(name, help, labels)
        }
    }

    /// Get or create a gauge metric.
    pub fn gauge(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
    ) -> MonitorResult<Gauge> {
        let key = Self::make_key(&name, &labels);

        if let Some(entry) = self.metrics.get(&key) {
            match &entry.1 {
                MetricValue::Gauge(gauge) => Ok(gauge.clone()),
                _ => Err(MonitorError::InvalidMetricName(format!(
                    "Metric {key} exists but is not a gauge"
                ))),
            }
        } else {
            self.register_gauge(name, help, labels)
        }
    }

    /// Get or create a histogram metric.
    pub fn histogram(
        &self,
        name: MetricName,
        help: impl Into<String>,
        labels: MetricLabels,
        buckets: Option<Vec<f64>>,
    ) -> MonitorResult<Histogram> {
        let key = Self::make_key(&name, &labels);

        if let Some(entry) = self.metrics.get(&key) {
            match &entry.1 {
                MetricValue::Histogram(histogram) => Ok(histogram.clone()),
                _ => Err(MonitorError::InvalidMetricName(format!(
                    "Metric {key} exists but is not a histogram"
                ))),
            }
        } else {
            self.register_histogram(name, help, labels, buckets)
        }
    }

    /// Get all registered metrics.
    #[must_use]
    pub fn metrics(&self) -> Vec<(Metric, MetricValue)> {
        self.metrics
            .iter()
            .map(|entry| (entry.value().0.clone(), entry.value().1.clone()))
            .collect()
    }

    /// Get metrics by name.
    #[must_use]
    pub fn metrics_by_name(&self, name: &str) -> Vec<(Metric, MetricValue)> {
        self.metrics
            .iter()
            .filter(|entry| entry.value().0.name.as_str() == name)
            .map(|entry| (entry.value().0.clone(), entry.value().1.clone()))
            .collect()
    }

    /// Clear all metrics.
    pub fn clear(&self) {
        self.metrics.clear();
    }

    /// Get the number of registered metrics.
    #[must_use]
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    fn make_key(name: &MetricName, labels: &MetricLabels) -> String {
        if labels.is_empty() {
            name.to_string()
        } else {
            let mut label_pairs: Vec<_> = labels.iter().collect();
            label_pairs.sort_by_key(|(k, _)| *k);
            let labels_str = label_pairs
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",");
            format!("{name}{{{labels_str}}}")
        }
    }
}

impl Default for MetricRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Snapshot of metric values for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSnapshot {
    /// Metric name.
    pub name: String,
    /// Metric kind.
    pub kind: String,
    /// Metric labels.
    pub labels: MetricLabels,
    /// Metric value.
    pub value: MetricValueSnapshot,
}

/// Snapshot of metric values.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MetricValueSnapshot {
    /// Counter snapshot.
    Counter {
        /// Counter value.
        value: u64,
    },
    /// Gauge snapshot.
    Gauge {
        /// Gauge value.
        value: f64,
    },
    /// Histogram snapshot.
    Histogram {
        /// Sum of all observed values.
        sum: f64,
        /// Count of observations.
        count: u64,
        /// Histogram buckets (upper bound, count).
        buckets: Vec<(f64, u64)>,
    },
    /// Summary snapshot.
    Summary {
        /// Count of observations.
        count: usize,
        /// Sum of all observed values.
        sum: f64,
        /// Minimum value.
        min: Option<f64>,
        /// Maximum value.
        max: Option<f64>,
        /// 50th percentile.
        p50: Option<f64>,
        /// 95th percentile.
        p95: Option<f64>,
        /// 99th percentile.
        p99: Option<f64>,
    },
}

impl MetricRegistry {
    /// Create a snapshot of all metrics for serialization.
    #[must_use]
    pub fn snapshot(&self) -> Vec<MetricSnapshot> {
        self.metrics
            .iter()
            .map(|entry| {
                let (metric, value) = entry.value();
                let value_snapshot = match value {
                    MetricValue::Counter(counter) => MetricValueSnapshot::Counter {
                        value: counter.get(),
                    },
                    MetricValue::Gauge(gauge) => MetricValueSnapshot::Gauge { value: gauge.get() },
                    MetricValue::Histogram(histogram) => MetricValueSnapshot::Histogram {
                        sum: histogram.sum(),
                        count: histogram.count(),
                        buckets: histogram.buckets(),
                    },
                    MetricValue::Summary(summary) => MetricValueSnapshot::Summary {
                        count: summary.count(),
                        sum: summary.sum(),
                        min: summary.min(),
                        max: summary.max(),
                        p50: summary.percentile(0.5),
                        p95: summary.percentile(0.95),
                        p99: summary.percentile(0.99),
                    },
                };

                MetricSnapshot {
                    name: metric.name.to_string(),
                    kind: format!("{:?}", metric.kind),
                    labels: metric.labels.clone(),
                    value: value_snapshot,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_counter() {
        let registry = MetricRegistry::new();
        let counter = registry
            .register_counter(
                MetricName::new("test_counter").expect("failed to create"),
                "Test counter",
                MetricLabels::new(),
            )
            .expect("operation should succeed");

        counter.inc();
        counter.add(5);

        assert_eq!(counter.get(), 6);

        // Should fail to register duplicate
        assert!(registry
            .register_counter(
                MetricName::new("test_counter").expect("failed to create"),
                "Test counter",
                MetricLabels::new(),
            )
            .is_err());
    }

    #[test]
    fn test_registry_gauge() {
        let registry = MetricRegistry::new();
        let gauge = registry
            .register_gauge(
                MetricName::new("test_gauge").expect("failed to create"),
                "Test gauge",
                MetricLabels::new(),
            )
            .expect("operation should succeed");

        gauge.set(42.5);
        assert_eq!(gauge.get(), 42.5);
    }

    #[test]
    fn test_registry_histogram() {
        let registry = MetricRegistry::new();
        let hist = registry
            .register_histogram(
                MetricName::new("test_histogram").expect("failed to create"),
                "Test histogram",
                MetricLabels::new(),
                None,
            )
            .expect("operation should succeed");

        hist.observe(1.5);
        hist.observe(2.5);

        assert_eq!(hist.count(), 2);
        assert_eq!(hist.sum(), 4.0);
    }

    #[test]
    fn test_registry_labels() {
        let registry = MetricRegistry::new();

        let mut labels1 = MetricLabels::new();
        labels1.insert("method".to_string(), "GET".to_string());

        let mut labels2 = MetricLabels::new();
        labels2.insert("method".to_string(), "POST".to_string());

        let counter1 = registry
            .register_counter(
                MetricName::new("http_requests").expect("failed to create"),
                "HTTP requests",
                labels1.clone(),
            )
            .expect("operation should succeed");

        let counter2 = registry
            .register_counter(
                MetricName::new("http_requests").expect("failed to create"),
                "HTTP requests",
                labels2.clone(),
            )
            .expect("operation should succeed");

        counter1.inc();
        counter2.add(2);

        assert_eq!(counter1.get(), 1);
        assert_eq!(counter2.get(), 2);
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_get_or_create_counter() {
        let registry = MetricRegistry::new();

        let counter1 = registry
            .counter(
                MetricName::new("test").expect("failed to create"),
                "Test",
                MetricLabels::new(),
            )
            .expect("operation should succeed");

        counter1.inc();

        let counter2 = registry
            .counter(
                MetricName::new("test").expect("failed to create"),
                "Test",
                MetricLabels::new(),
            )
            .expect("operation should succeed");

        // Should be the same counter
        assert_eq!(counter2.get(), 1);
    }

    #[test]
    fn test_metrics_by_name() {
        let registry = MetricRegistry::new();

        let mut labels1 = MetricLabels::new();
        labels1.insert("status".to_string(), "200".to_string());

        let mut labels2 = MetricLabels::new();
        labels2.insert("status".to_string(), "404".to_string());

        registry
            .register_counter(
                MetricName::new("requests").expect("failed to create"),
                "Requests",
                labels1,
            )
            .expect("operation should succeed");

        registry
            .register_counter(
                MetricName::new("requests").expect("failed to create"),
                "Requests",
                labels2,
            )
            .expect("operation should succeed");

        let metrics = registry.metrics_by_name("requests");
        assert_eq!(metrics.len(), 2);
    }

    #[test]
    fn test_snapshot() {
        let registry = MetricRegistry::new();

        let counter = registry
            .register_counter(
                MetricName::new("test_counter").expect("failed to create"),
                "Test",
                MetricLabels::new(),
            )
            .expect("operation should succeed");

        counter.add(42);

        let snapshot = registry.snapshot();
        assert_eq!(snapshot.len(), 1);

        match &snapshot[0].value {
            MetricValueSnapshot::Counter { value } => assert_eq!(*value, 42),
            _ => panic!("Expected counter"),
        }
    }
}
