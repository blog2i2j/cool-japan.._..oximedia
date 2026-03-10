//! Distributed metrics collection and aggregation.
//!
//! Provides time-series metric recording, windowed queries, statistical
//! aggregation, and alert evaluation for the distributed encoding cluster.

#![allow(dead_code)]

use std::collections::HashMap;

/// A single metric measurement.
#[derive(Debug, Clone)]
pub struct MetricPoint {
    /// Name of the metric (e.g., "`cpu_usage`", "`frames_encoded`").
    pub name: String,
    /// Numeric value.
    pub value: f64,
    /// Key-value tags for dimensionality (e.g., `worker_id`, codec).
    pub tags: HashMap<String, String>,
    /// Unix epoch timestamp in milliseconds.
    pub timestamp_ms: u64,
}

impl MetricPoint {
    /// Create a new metric point.
    pub fn new(name: impl Into<String>, value: f64, timestamp_ms: u64) -> Self {
        Self {
            name: name.into(),
            value,
            tags: HashMap::new(),
            timestamp_ms,
        }
    }

    /// Create a metric point with tags.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

/// Aggregation function to apply over a set of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationFn {
    /// Sum of all values.
    Sum,
    /// Arithmetic mean.
    Mean,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// 50th percentile (median).
    P50,
    /// 95th percentile.
    P95,
    /// 99th percentile.
    P99,
}

/// Collects and aggregates distributed metrics.
#[derive(Debug, Default)]
pub struct MetricAggregator {
    /// All recorded metric points, grouped by metric name.
    points: HashMap<String, Vec<MetricPoint>>,
}

impl MetricAggregator {
    /// Create a new aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }

    /// Record a metric point.
    pub fn record(&mut self, point: MetricPoint) {
        self.points
            .entry(point.name.clone())
            .or_default()
            .push(point);
    }

    /// Return all values for a metric within the given time window ending at
    /// the most recent recorded timestamp (or the largest timestamp seen).
    ///
    /// `window_ms` is the duration in milliseconds to look back.  Points are
    /// included when `now_max - window_ms <= timestamp_ms <= now_max`.
    #[must_use]
    pub fn query(&self, name: &str, window_ms: u64) -> Vec<f64> {
        let pts = match self.points.get(name) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let now_max = pts.iter().map(|p| p.timestamp_ms).max().unwrap_or(0);
        let start = now_max.saturating_sub(window_ms);

        pts.iter()
            .filter(|p| p.timestamp_ms >= start)
            .map(|p| p.value)
            .collect()
    }

    /// Compute an aggregation over values in a time window.
    ///
    /// Returns `None` if there are no values in the window.
    #[must_use]
    pub fn aggregate(&self, name: &str, window_ms: u64, agg: AggregationFn) -> Option<f64> {
        let mut values = self.query(name, window_ms);
        if values.is_empty() {
            return None;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let result = match agg {
            AggregationFn::Sum => values.iter().sum(),
            AggregationFn::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AggregationFn::Min => *values
                .first()
                .expect("values non-empty: is_empty check returned above"),
            AggregationFn::Max => *values
                .last()
                .expect("values non-empty: is_empty check returned above"),
            AggregationFn::P50 => percentile(&values, 50.0),
            AggregationFn::P95 => percentile(&values, 95.0),
            AggregationFn::P99 => percentile(&values, 99.0),
        };

        Some(result)
    }
}

/// Compute the p-th percentile of a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// A snapshot of cluster-wide metrics.
#[derive(Debug, Clone)]
pub struct ClusterMetrics {
    /// Number of worker nodes.
    pub worker_count: u32,
    /// Total tasks ever submitted.
    pub total_tasks: u64,
    /// Tasks currently running.
    pub running_tasks: u32,
    /// Total failed tasks.
    pub failed_tasks: u64,
    /// Average throughput (tasks/sec or frames/sec depending on context).
    pub avg_throughput: f64,
}

impl ClusterMetrics {
    /// Compute cluster metrics from aggregated data.
    ///
    /// Reads the following metric names:
    /// - "`worker_count`" (latest value)
    /// - "`total_tasks`" (sum over window)
    /// - "`running_tasks`" (latest value)
    /// - "`failed_tasks`" (sum over window)
    /// - "throughput" (mean over window)
    #[must_use]
    pub fn compute(aggregator: &MetricAggregator, now_ms: u64) -> Self {
        let window_ms = now_ms; // look at all recorded data

        let worker_count = aggregator
            .aggregate("worker_count", window_ms, AggregationFn::Max)
            .unwrap_or(0.0) as u32;

        let total_tasks = aggregator
            .aggregate("total_tasks", window_ms, AggregationFn::Sum)
            .unwrap_or(0.0) as u64;

        let running_tasks = aggregator
            .aggregate("running_tasks", window_ms, AggregationFn::Max)
            .unwrap_or(0.0) as u32;

        let failed_tasks = aggregator
            .aggregate("failed_tasks", window_ms, AggregationFn::Sum)
            .unwrap_or(0.0) as u64;

        let avg_throughput = aggregator
            .aggregate("throughput", window_ms, AggregationFn::Mean)
            .unwrap_or(0.0);

        Self {
            worker_count,
            total_tasks,
            running_tasks,
            failed_tasks,
            avg_throughput,
        }
    }
}

/// Comparison operator for alert rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// Alert fires when metric is above the threshold.
    Above,
    /// Alert fires when metric is below the threshold.
    Below,
    /// Alert fires when metric equals the threshold (within floating-point epsilon).
    Equal,
}

/// A rule that triggers an alert based on a metric condition.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Name of the metric to watch.
    pub metric_name: String,
    /// Threshold value.
    pub threshold: f64,
    /// How to compare the aggregated value against the threshold.
    pub comparison: Comparison,
    /// Time window (ms) over which to aggregate the metric.
    pub window_ms: u64,
}

impl AlertRule {
    /// Create a new alert rule.
    pub fn new(
        metric_name: impl Into<String>,
        threshold: f64,
        comparison: Comparison,
        window_ms: u64,
    ) -> Self {
        Self {
            metric_name: metric_name.into(),
            threshold,
            comparison,
            window_ms,
        }
    }
}

/// A fired alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// The rule that triggered this alert.
    pub rule: AlertRule,
    /// The aggregated value that triggered the alert.
    pub current_value: f64,
    /// When the alert was triggered (Unix epoch ms).
    pub triggered_at_ms: u64,
}

/// Evaluates alert rules against recorded metrics.
pub struct AlertEvaluator;

impl AlertEvaluator {
    /// Check all rules and return any that are currently firing.
    #[must_use]
    pub fn check(rules: &[AlertRule], aggregator: &MetricAggregator, now_ms: u64) -> Vec<Alert> {
        let mut alerts = Vec::new();

        for rule in rules {
            // Use Mean as default aggregation for alerts
            if let Some(value) =
                aggregator.aggregate(&rule.metric_name, rule.window_ms, AggregationFn::Mean)
            {
                let fires = match rule.comparison {
                    Comparison::Above => value > rule.threshold,
                    Comparison::Below => value < rule.threshold,
                    Comparison::Equal => (value - rule.threshold).abs() < f64::EPSILON,
                };

                if fires {
                    alerts.push(Alert {
                        rule: rule.clone(),
                        current_value: value,
                        triggered_at_ms: now_ms,
                    });
                }
            }
        }

        alerts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(ms: u64) -> u64 {
        ms
    }

    fn record_values(agg: &mut MetricAggregator, name: &str, values: &[(f64, u64)]) {
        for &(v, t) in values {
            agg.record(MetricPoint::new(name, v, ts(t)));
        }
    }

    #[test]
    fn test_record_and_query() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "cpu", &[(0.5, 1000), (0.6, 2000), (0.7, 3000)]);
        let vals = agg.query("cpu", 5000);
        assert_eq!(vals.len(), 3);
    }

    #[test]
    fn test_query_windowed() {
        let mut agg = MetricAggregator::new();
        // Points at t=100, t=500, t=1000; max=1000, window=600 → include t>=400
        record_values(&mut agg, "fps", &[(10.0, 100), (20.0, 500), (30.0, 1000)]);
        let vals = agg.query("fps", 600);
        // Should include t=500 and t=1000
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_aggregate_sum() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "bytes", &[(100.0, 1), (200.0, 2), (300.0, 3)]);
        let sum = agg.aggregate("bytes", 100, AggregationFn::Sum);
        assert!((sum.expect("metric computation should succeed") - 600.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_mean() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "lat", &[(10.0, 1), (20.0, 2), (30.0, 3)]);
        let mean = agg.aggregate("lat", 100, AggregationFn::Mean);
        assert!((mean.expect("metric computation should succeed") - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_min_max() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "val", &[(5.0, 1), (1.0, 2), (9.0, 3)]);
        assert!(
            (agg.aggregate("val", 100, AggregationFn::Min)
                .expect("aggregation should succeed")
                - 1.0)
                .abs()
                < 1e-9
        );
        assert!(
            (agg.aggregate("val", 100, AggregationFn::Max)
                .expect("aggregation should succeed")
                - 9.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn test_aggregate_p50() {
        let mut agg = MetricAggregator::new();
        // sorted: 1,2,3,4,5 → p50 = 3
        record_values(
            &mut agg,
            "m",
            &[(3.0, 1), (1.0, 2), (5.0, 3), (2.0, 4), (4.0, 5)],
        );
        let p50 = agg.aggregate("m", 100, AggregationFn::P50);
        assert!((p50.expect("metric computation should succeed") - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_aggregate_empty() {
        let agg = MetricAggregator::new();
        assert!(agg
            .aggregate("nonexistent", 1000, AggregationFn::Mean)
            .is_none());
    }

    #[test]
    fn test_cluster_metrics_compute() {
        let mut agg = MetricAggregator::new();
        agg.record(MetricPoint::new("worker_count", 4.0, 1000));
        agg.record(MetricPoint::new("total_tasks", 100.0, 1000));
        agg.record(MetricPoint::new("running_tasks", 12.0, 1000));
        agg.record(MetricPoint::new("failed_tasks", 3.0, 1000));
        agg.record(MetricPoint::new("throughput", 25.0, 1000));

        let metrics = ClusterMetrics::compute(&agg, 2000);
        assert_eq!(metrics.worker_count, 4);
        assert_eq!(metrics.running_tasks, 12);
        assert!((metrics.avg_throughput - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_alert_above_fires() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "cpu", &[(0.95, 1000)]);
        let rules = vec![AlertRule::new("cpu", 0.8, Comparison::Above, 5000)];
        let alerts = AlertEvaluator::check(&rules, &agg, 1000);
        assert_eq!(alerts.len(), 1);
        assert!((alerts[0].current_value - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_alert_below_fires() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "workers", &[(2.0, 1000)]);
        let rules = vec![AlertRule::new("workers", 5.0, Comparison::Below, 5000)];
        let alerts = AlertEvaluator::check(&rules, &agg, 1000);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_alert_does_not_fire_when_ok() {
        let mut agg = MetricAggregator::new();
        record_values(&mut agg, "cpu", &[(0.5, 1000)]);
        let rules = vec![AlertRule::new("cpu", 0.8, Comparison::Above, 5000)];
        let alerts = AlertEvaluator::check(&rules, &agg, 1000);
        assert!(alerts.is_empty());
    }
}
