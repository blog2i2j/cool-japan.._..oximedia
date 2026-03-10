#![allow(dead_code)]

//! Metric export in multiple formats (Prometheus, JSON, CSV, `StatsD`).
//!
//! Provides serializers that take snapshot data and emit wire-format output
//! suitable for ingestion by popular monitoring back-ends.

use std::collections::BTreeMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Metric model
// ---------------------------------------------------------------------------

/// Type of metric value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Monotonically increasing counter.
    Counter,
    /// Instantaneous value.
    Gauge,
    /// Distribution of values.
    Histogram,
    /// Textual information (exported as label only).
    Info,
}

impl fmt::Display for MetricType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Counter => write!(f, "counter"),
            Self::Gauge => write!(f, "gauge"),
            Self::Histogram => write!(f, "histogram"),
            Self::Info => write!(f, "info"),
        }
    }
}

/// A labelled metric sample.
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Metric name (e.g. `cpu_usage_percent`).
    pub name: String,
    /// Help text / description.
    pub help: String,
    /// Metric type.
    pub metric_type: MetricType,
    /// Label key-value pairs.
    pub labels: BTreeMap<String, String>,
    /// Numeric value.
    pub value: f64,
    /// Timestamp (Unix epoch millis). `None` means omit.
    pub timestamp_ms: Option<u64>,
}

impl MetricSample {
    /// Create a new sample with the given name and value.
    pub fn new(name: impl Into<String>, value: f64, metric_type: MetricType) -> Self {
        Self {
            name: name.into(),
            help: String::new(),
            metric_type,
            labels: BTreeMap::new(),
            value,
            timestamp_ms: None,
        }
    }

    /// Attach a help description.
    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = help.into();
        self
    }

    /// Add a label.
    pub fn with_label(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.labels.insert(key.into(), val.into());
        self
    }

    /// Stamp with the current time.
    #[must_use]
    pub fn with_current_timestamp(mut self) -> Self {
        if let Ok(d) = SystemTime::now().duration_since(UNIX_EPOCH) {
            self.timestamp_ms = Some(d.as_millis() as u64);
        }
        self
    }

    /// Stamp with an explicit unix millis value.
    #[must_use]
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp_ms = Some(ts);
        self
    }
}

/// A batch of metric samples ready for export.
#[derive(Debug, Clone, Default)]
pub struct MetricBatch {
    /// All samples.
    pub samples: Vec<MetricSample>,
}

impl MetricBatch {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    /// Add a sample.
    pub fn push(&mut self, sample: MetricSample) {
        self.samples.push(sample);
    }

    /// Number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Exporters
// ---------------------------------------------------------------------------

/// Render a batch in Prometheus text exposition format.
#[must_use]
pub fn export_prometheus(batch: &MetricBatch) -> String {
    let mut out = String::new();
    let mut seen_help: std::collections::HashSet<String> = std::collections::HashSet::new();

    for s in &batch.samples {
        if !seen_help.contains(&s.name) {
            if !s.help.is_empty() {
                out.push_str(&format!("# HELP {} {}\n", s.name, s.help));
            }
            out.push_str(&format!("# TYPE {} {}\n", s.name, s.metric_type));
            seen_help.insert(s.name.clone());
        }

        out.push_str(&s.name);
        if !s.labels.is_empty() {
            out.push('{');
            let pairs: Vec<String> = s
                .labels
                .iter()
                .map(|(k, v)| format!("{k}=\"{v}\""))
                .collect();
            out.push_str(&pairs.join(","));
            out.push('}');
        }
        out.push(' ');
        out.push_str(&format_value(s.value));
        if let Some(ts) = s.timestamp_ms {
            out.push(' ');
            out.push_str(&ts.to_string());
        }
        out.push('\n');
    }
    out
}

/// Render a batch as a JSON array.
#[must_use]
pub fn export_json(batch: &MetricBatch) -> String {
    let mut entries: Vec<String> = Vec::new();
    for s in &batch.samples {
        let labels_json: Vec<String> = s
            .labels
            .iter()
            .map(|(k, v)| format!("\"{k}\":\"{v}\""))
            .collect();
        let ts = s
            .timestamp_ms
            .map_or_else(|| "null".to_string(), |t| t.to_string());
        entries.push(format!(
            "{{\"name\":\"{}\",\"type\":\"{}\",\"value\":{},\"labels\":{{{}}},\"timestamp\":{}}}",
            s.name,
            s.metric_type,
            format_value(s.value),
            labels_json.join(","),
            ts
        ));
    }
    format!("[{}]", entries.join(","))
}

/// Render a batch as CSV text.
pub fn export_csv(batch: &MetricBatch) -> String {
    let mut out = String::from("name,type,value,labels,timestamp\n");
    for s in &batch.samples {
        let labels_str: Vec<String> = s.labels.iter().map(|(k, v)| format!("{k}={v}")).collect();
        let ts = s.timestamp_ms.map_or_else(String::new, |t| t.to_string());
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            s.name,
            s.metric_type,
            format_value(s.value),
            labels_str.join(";"),
            ts
        ));
    }
    out
}

/// Render a batch in `StatsD` line protocol.
#[must_use]
pub fn export_statsd(batch: &MetricBatch) -> String {
    let mut out = String::new();
    for s in &batch.samples {
        let suffix = match s.metric_type {
            MetricType::Counter => "c",
            MetricType::Gauge => "g",
            MetricType::Histogram => "ms",
            MetricType::Info => "g",
        };
        out.push_str(&format!(
            "{}:{}|{}\n",
            s.name,
            format_value(s.value),
            suffix
        ));
    }
    out
}

/// Format f64 to string; whole numbers get no decimal.
fn format_value(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{v:.0}")
    } else {
        format!("{v}")
    }
}

/// Aggregate helper: compute a summary (count, sum, min, max, mean) from values.
#[derive(Debug, Clone, Copy)]
pub struct MetricSummary {
    /// Number of values.
    pub count: usize,
    /// Sum of values.
    pub sum: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
}

impl MetricSummary {
    /// Compute a summary from a slice of f64 values.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn compute(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let count = values.len();
        let mut sum = 0.0_f64;
        let mut mn = f64::MAX;
        let mut mx = f64::MIN;
        for &v in values {
            sum += v;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        Some(Self {
            count,
            sum,
            min: mn,
            max: mx,
            mean: sum / count as f64,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_batch() -> MetricBatch {
        let mut batch = MetricBatch::new();
        batch.push(
            MetricSample::new("cpu_usage", 72.5, MetricType::Gauge)
                .with_help("CPU usage percent")
                .with_label("host", "server1")
                .with_timestamp(1_700_000_000_000),
        );
        batch.push(
            MetricSample::new("requests_total", 1234.0, MetricType::Counter)
                .with_help("Total HTTP requests")
                .with_label("method", "GET"),
        );
        batch
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::Counter.to_string(), "counter");
        assert_eq!(MetricType::Gauge.to_string(), "gauge");
        assert_eq!(MetricType::Histogram.to_string(), "histogram");
        assert_eq!(MetricType::Info.to_string(), "info");
    }

    #[test]
    fn test_metric_sample_builder() {
        let s = MetricSample::new("test", 42.0, MetricType::Gauge)
            .with_help("A test metric")
            .with_label("env", "prod");
        assert_eq!(s.name, "test");
        assert!((s.value - 42.0).abs() < 1e-9);
        assert_eq!(s.labels.get("env").expect("failed to get value"), "prod");
    }

    #[test]
    fn test_metric_batch_push() {
        let mut batch = MetricBatch::new();
        assert!(batch.is_empty());
        batch.push(MetricSample::new("x", 1.0, MetricType::Counter));
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_prometheus_export_contains_help() {
        let batch = sample_batch();
        let prom = export_prometheus(&batch);
        assert!(prom.contains("# HELP cpu_usage CPU usage percent"));
        assert!(prom.contains("# TYPE cpu_usage gauge"));
    }

    #[test]
    fn test_prometheus_export_labels() {
        let batch = sample_batch();
        let prom = export_prometheus(&batch);
        assert!(prom.contains("cpu_usage{host=\"server1\"} 72.5 1700000000000"));
    }

    #[test]
    fn test_prometheus_counter_no_timestamp() {
        let batch = sample_batch();
        let prom = export_prometheus(&batch);
        // counter line should NOT have timestamp (we didn't attach one)
        assert!(prom.contains("requests_total{method=\"GET\"} 1234\n"));
    }

    #[test]
    fn test_json_export() {
        let batch = sample_batch();
        let json = export_json(&batch);
        assert!(json.starts_with('['));
        assert!(json.ends_with(']'));
        assert!(json.contains("\"name\":\"cpu_usage\""));
    }

    #[test]
    fn test_csv_export_header() {
        let batch = sample_batch();
        let csv = export_csv(&batch);
        assert!(csv.starts_with("name,type,value,labels,timestamp\n"));
    }

    #[test]
    fn test_csv_export_data() {
        let batch = sample_batch();
        let csv = export_csv(&batch);
        assert!(csv.contains("cpu_usage,gauge,72.5,host=server1,1700000000000"));
    }

    #[test]
    fn test_statsd_export() {
        let batch = sample_batch();
        let sd = export_statsd(&batch);
        assert!(sd.contains("cpu_usage:72.5|g"));
        assert!(sd.contains("requests_total:1234|c"));
    }

    #[test]
    fn test_metric_summary_compute() {
        let vals = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let s = MetricSummary::compute(&vals).expect("operation should succeed");
        assert_eq!(s.count, 5);
        assert!((s.sum - 150.0).abs() < 1e-9);
        assert!((s.min - 10.0).abs() < 1e-9);
        assert!((s.max - 50.0).abs() < 1e-9);
        assert!((s.mean - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_summary_empty() {
        assert!(MetricSummary::compute(&[]).is_none());
    }

    #[test]
    fn test_format_value_integer() {
        assert_eq!(format_value(42.0), "42");
    }

    #[test]
    fn test_format_value_fractional() {
        assert_eq!(format_value(3.14), "3.14");
    }

    #[test]
    fn test_sample_with_current_timestamp() {
        let s = MetricSample::new("t", 1.0, MetricType::Counter).with_current_timestamp();
        assert!(s.timestamp_ms.is_some());
    }
}
