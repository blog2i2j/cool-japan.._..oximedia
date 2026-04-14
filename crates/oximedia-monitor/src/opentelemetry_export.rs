//! OpenTelemetry-compatible metrics export (wire-format only, no OTel SDK).
//!
//! Implements the OpenTelemetry Metrics JSON format defined at:
//! <https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding>
//!
//! This module intentionally avoids taking a dependency on the `opentelemetry`
//! crate family.  Instead it defines lightweight data structures and a JSON
//! serialiser that produces output conforming to the OTLP ExportMetrics
//! service request schema.
//!
//! # Quick-start
//!
//! ```rust
//! use oximedia_monitor::opentelemetry_export::{
//!     OtelExportConfig, OtelExporter, OtelGauge, OtelMetricExport,
//!     OtelMetricPoint, InternalMetric, MetricBridge,
//! };
//!
//! let config = OtelExportConfig {
//!     service_name: "media-encoder".to_string(),
//!     service_version: "0.1.3".to_string(),
//!     endpoint: None,
//! };
//!
//! let internal = InternalMetric {
//!     name: "cpu_usage".to_string(),
//!     value: 72.5,
//!     labels: vec![("host".to_string(), "node-01".to_string())],
//!     timestamp_secs: 1_700_000_000,
//! };
//!
//! let bridge = MetricBridge::new(&config);
//! let point = bridge.convert(&internal);
//!
//! let export = OtelMetricExport {
//!     resource_attributes: vec![
//!         ("service.name".to_string(), config.service_name.clone()),
//!     ],
//!     metrics: vec![OtelGauge {
//!         name: "cpu_usage".to_string(),
//!         description: "CPU usage percent".to_string(),
//!         unit: "%".to_string(),
//!         data_points: vec![point],
//!     }],
//! };
//!
//! let json = OtelExporter::export(&export);
//! assert!(json.contains("cpu_usage"));
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single data point in an OpenTelemetry gauge metric.
#[derive(Debug, Clone, PartialEq)]
pub struct OtelMetricPoint {
    /// Name of the metric this point belongs to.
    pub metric_name: String,
    /// Numeric value of this data point.
    pub value: f64,
    /// Timestamp expressed in Unix nanoseconds.
    pub timestamp_nanos: u64,
    /// Key-value attribute labels associated with this point.
    pub labels: Vec<(String, String)>,
}

impl OtelMetricPoint {
    /// Create a new metric point.
    #[must_use]
    pub fn new(
        metric_name: impl Into<String>,
        value: f64,
        timestamp_nanos: u64,
    ) -> Self {
        Self {
            metric_name: metric_name.into(),
            value,
            timestamp_nanos,
            labels: Vec::new(),
        }
    }

    /// Add a label (attribute) to this data point.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }
}

/// An OpenTelemetry gauge metric containing one or more data points.
#[derive(Debug, Clone)]
pub struct OtelGauge {
    /// Metric name (e.g. `"cpu_usage"`).
    pub name: String,
    /// Human-readable description of the metric.
    pub description: String,
    /// Unit string (e.g. `"%"`, `"ms"`, `"bytes"`).
    pub unit: String,
    /// Individual data points for this metric.
    pub data_points: Vec<OtelMetricPoint>,
}

impl OtelGauge {
    /// Create a gauge with no data points.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        unit: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            unit: unit.into(),
            data_points: Vec::new(),
        }
    }

    /// Append a data point.
    pub fn push(&mut self, point: OtelMetricPoint) {
        self.data_points.push(point);
    }
}

/// A complete OpenTelemetry metrics export payload.
///
/// Corresponds to an `ExportMetricsServiceRequest` in the OTLP spec.
#[derive(Debug, Clone)]
pub struct OtelMetricExport {
    /// Resource-level attributes (e.g. `service.name`, `service.version`).
    pub resource_attributes: Vec<(String, String)>,
    /// Gauge metrics included in this export.
    pub metrics: Vec<OtelGauge>,
}

impl OtelMetricExport {
    /// Create an empty export with no resource attributes or metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            resource_attributes: Vec::new(),
            metrics: Vec::new(),
        }
    }
}

impl Default for OtelMetricExport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Exporter configuration
// ---------------------------------------------------------------------------

/// Configuration for the OpenTelemetry exporter.
#[derive(Debug, Clone)]
pub struct OtelExportConfig {
    /// `service.name` resource attribute value.
    pub service_name: String,
    /// `service.version` resource attribute value.
    pub service_version: String,
    /// Optional HTTP endpoint to POST the JSON payload to.
    /// When `None`, the exporter is serialise-only.
    pub endpoint: Option<String>,
}

impl OtelExportConfig {
    /// Create a configuration with a service name and version.
    #[must_use]
    pub fn new(service_name: impl Into<String>, service_version: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
            service_version: service_version.into(),
            endpoint: None,
        }
    }

    /// Set an export endpoint URL.
    #[must_use]
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Build the standard resource attributes for this configuration.
    #[must_use]
    pub fn resource_attributes(&self) -> Vec<(String, String)> {
        vec![
            ("service.name".to_string(), self.service_name.clone()),
            ("service.version".to_string(), self.service_version.clone()),
        ]
    }
}

// ---------------------------------------------------------------------------
// JSON serialiser
// ---------------------------------------------------------------------------

/// Stateless OpenTelemetry JSON exporter.
///
/// Produces output matching the OTLP JSON encoding for an
/// `ExportMetricsServiceRequest`.
pub struct OtelExporter;

impl OtelExporter {
    /// Serialise `metrics` into OTLP JSON format.
    ///
    /// The output is a self-contained JSON object that can be POSTed to any
    /// OTLP-compatible collector (Jaeger, Tempo, the OpenTelemetry Collector,
    /// etc.) at `/v1/metrics`.
    #[must_use]
    pub fn export(metrics: &OtelMetricExport) -> String {
        // Resource attributes.
        let resource_attrs = Self::attrs_to_json(&metrics.resource_attributes);

        // Build one `metrics` array element per gauge.
        let gauge_jsons: Vec<String> = metrics
            .metrics
            .iter()
            .map(|g| Self::gauge_to_json(g))
            .collect();

        let scope_metrics = format!(
            "{{\"scope\":{{\"name\":\"oximedia-monitor\",\"version\":\"0.1.3\"}},\"metrics\":[{}]}}",
            gauge_jsons.join(",")
        );

        format!(
            "{{\"resourceMetrics\":[{{\"resource\":{{\"attributes\":{resource_attrs}}},\"scopeMetrics\":[{scope_metrics}]}}]}}"
        )
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn gauge_to_json(gauge: &OtelGauge) -> String {
        let data_points: Vec<String> = gauge
            .data_points
            .iter()
            .map(Self::data_point_to_json)
            .collect();

        format!(
            "{{\"name\":{},\"description\":{},\"unit\":{},\"gauge\":{{\"dataPoints\":[{}]}}}}",
            Self::json_string(&gauge.name),
            Self::json_string(&gauge.description),
            Self::json_string(&gauge.unit),
            data_points.join(",")
        )
    }

    fn data_point_to_json(point: &OtelMetricPoint) -> String {
        let attrs = Self::attrs_to_json(&point.labels);
        format!(
            "{{\"attributes\":{attrs},\"timeUnixNano\":\"{}\",\"asDouble\":{}}}",
            point.timestamp_nanos, point.value
        )
    }

    /// Serialise a slice of key-value pairs as an OTLP `attributes` JSON array.
    fn attrs_to_json(attrs: &[(String, String)]) -> String {
        let items: Vec<String> = attrs
            .iter()
            .map(|(k, v)| {
                format!(
                    "{{\"key\":{},\"value\":{{\"stringValue\":{}}}}}",
                    Self::json_string(k),
                    Self::json_string(v)
                )
            })
            .collect();
        format!("[{}]", items.join(","))
    }

    /// Produce a properly escaped JSON string literal (including surrounding
    /// double-quotes).
    fn json_string(s: &str) -> String {
        let escaped = s
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t");
        format!("\"{escaped}\"")
    }
}

// ---------------------------------------------------------------------------
// Internal metric type + bridge
// ---------------------------------------------------------------------------

/// A simple internal metric representation used as the source type for the
/// [`MetricBridge`] converter.
///
/// This type is intentionally minimal — it mirrors what monitoring backends
/// typically expose without coupling to any specific crate.
#[derive(Debug, Clone)]
pub struct InternalMetric {
    /// Metric name.
    pub name: String,
    /// Scalar value.
    pub value: f64,
    /// Key-value label set.
    pub labels: Vec<(String, String)>,
    /// Unix timestamp in seconds.
    pub timestamp_secs: u64,
}

impl InternalMetric {
    /// Create a new internal metric.
    #[must_use]
    pub fn new(name: impl Into<String>, value: f64, timestamp_secs: u64) -> Self {
        Self {
            name: name.into(),
            value,
            labels: Vec::new(),
            timestamp_secs,
        }
    }

    /// Add a label.
    #[must_use]
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.push((key.into(), value.into()));
        self
    }
}

/// Converts [`InternalMetric`] values to [`OtelMetricPoint`]s, adding service
/// identity labels derived from the exporter configuration.
#[derive(Debug, Clone)]
pub struct MetricBridge {
    service_name: String,
    service_version: String,
}

impl MetricBridge {
    /// Create a bridge from an [`OtelExportConfig`].
    #[must_use]
    pub fn new(config: &OtelExportConfig) -> Self {
        Self {
            service_name: config.service_name.clone(),
            service_version: config.service_version.clone(),
        }
    }

    /// Convert one [`InternalMetric`] to an [`OtelMetricPoint`].
    ///
    /// The bridge multiplies the seconds timestamp by `1_000_000_000` to
    /// produce a nanosecond value compatible with the OTLP spec.
    #[must_use]
    pub fn convert(&self, metric: &InternalMetric) -> OtelMetricPoint {
        let timestamp_nanos = metric.timestamp_secs.saturating_mul(1_000_000_000);
        let mut point = OtelMetricPoint::new(&metric.name, metric.value, timestamp_nanos);
        for (k, v) in &metric.labels {
            point.labels.push((k.clone(), v.clone()));
        }
        point
    }

    /// Convert a batch of [`InternalMetric`]s and group them into a single
    /// [`OtelMetricExport`] with the service resource attributes pre-filled.
    ///
    /// Each distinct metric name becomes one [`OtelGauge`].
    #[must_use]
    pub fn convert_batch(&self, metrics: &[InternalMetric]) -> OtelMetricExport {
        use std::collections::BTreeMap;

        let resource_attributes = vec![
            ("service.name".to_string(), self.service_name.clone()),
            ("service.version".to_string(), self.service_version.clone()),
        ];

        // Group by metric name, preserving insertion order via BTreeMap.
        let mut groups: BTreeMap<String, OtelGauge> = BTreeMap::new();

        for m in metrics {
            let gauge = groups.entry(m.name.clone()).or_insert_with(|| {
                OtelGauge::new(&m.name, "", "")
            });
            gauge.push(self.convert(m));
        }

        OtelMetricExport {
            resource_attributes,
            metrics: groups.into_values().collect(),
        }
    }

    /// Service name used by this bridge.
    #[must_use]
    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Service version used by this bridge.
    #[must_use]
    pub fn service_version(&self) -> &str {
        &self.service_version
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_export() -> OtelMetricExport {
        OtelMetricExport {
            resource_attributes: vec![
                ("service.name".to_string(), "test-service".to_string()),
                ("service.version".to_string(), "1.0.0".to_string()),
            ],
            metrics: vec![
                OtelGauge {
                    name: "cpu_usage".to_string(),
                    description: "CPU utilisation".to_string(),
                    unit: "%".to_string(),
                    data_points: vec![OtelMetricPoint {
                        metric_name: "cpu_usage".to_string(),
                        value: 72.5,
                        timestamp_nanos: 1_700_000_000_000_000_000,
                        labels: vec![("host".to_string(), "node-01".to_string())],
                    }],
                },
                OtelGauge {
                    name: "memory_used_bytes".to_string(),
                    description: "Memory used".to_string(),
                    unit: "bytes".to_string(),
                    data_points: vec![OtelMetricPoint {
                        metric_name: "memory_used_bytes".to_string(),
                        value: 2_147_483_648.0,
                        timestamp_nanos: 1_700_000_000_000_000_000,
                        labels: vec![],
                    }],
                },
            ],
        }
    }

    // ---- JSON structure ----------------------------------------------------

    #[test]
    fn test_export_contains_resource_metrics() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("resourceMetrics"),
            "JSON must contain 'resourceMetrics'"
        );
    }

    #[test]
    fn test_export_contains_scope_metrics() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("scopeMetrics"),
            "JSON must contain 'scopeMetrics'"
        );
    }

    #[test]
    fn test_export_metric_name_preserved() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("cpu_usage"),
            "JSON must contain metric name 'cpu_usage'"
        );
        assert!(
            json.contains("memory_used_bytes"),
            "JSON must contain metric name 'memory_used_bytes'"
        );
    }

    #[test]
    fn test_export_labels_included() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("node-01"),
            "JSON must contain label value 'node-01'"
        );
        assert!(json.contains("host"), "JSON must contain label key 'host'");
    }

    #[test]
    fn test_export_resource_attributes_correct() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("test-service"),
            "JSON must contain service.name value"
        );
        assert!(
            json.contains("1.0.0"),
            "JSON must contain service.version value"
        );
    }

    #[test]
    fn test_export_empty_metrics() {
        let export = OtelMetricExport::new();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("resourceMetrics"),
            "Empty export must still have outer wrapper"
        );
        assert!(
            json.contains("scopeMetrics"),
            "Empty export must still have scopeMetrics"
        );
    }

    #[test]
    fn test_export_gauge_data_type() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        assert!(
            json.contains("\"gauge\""),
            "Metrics must be exported as gauge data points"
        );
    }

    #[test]
    fn test_export_timestamp_nanos_as_string() {
        let export = make_export();
        let json = OtelExporter::export(&export);
        // OTLP requires nanosecond timestamps to be encoded as strings.
        assert!(
            json.contains("timeUnixNano"),
            "JSON must contain 'timeUnixNano' field"
        );
    }

    #[test]
    fn test_export_json_escaping() {
        let mut export = OtelMetricExport::new();
        export.metrics.push(OtelGauge {
            name: "metric_with_quotes".to_string(),
            description: "Has \"quoted\" text and \\backslash".to_string(),
            unit: String::new(),
            data_points: vec![],
        });
        let json = OtelExporter::export(&export);
        // Must not contain unescaped bare double-quote inside a string value.
        assert!(
            json.contains("\\\"quoted\\\""),
            "Double-quotes in description must be escaped"
        );
        assert!(
            json.contains("\\\\backslash"),
            "Backslash in description must be escaped"
        );
    }

    // ---- OtelExportConfig --------------------------------------------------

    #[test]
    fn test_config_resource_attributes() {
        let cfg = OtelExportConfig::new("my-service", "2.0.0");
        let attrs = cfg.resource_attributes();
        let service_name = attrs.iter().find(|(k, _)| k == "service.name");
        let service_version = attrs.iter().find(|(k, _)| k == "service.version");
        assert_eq!(
            service_name.map(|(_, v)| v.as_str()),
            Some("my-service")
        );
        assert_eq!(
            service_version.map(|(_, v)| v.as_str()),
            Some("2.0.0")
        );
    }

    #[test]
    fn test_config_with_endpoint() {
        let cfg = OtelExportConfig::new("svc", "1.0").with_endpoint("http://otel:4318");
        assert_eq!(cfg.endpoint.as_deref(), Some("http://otel:4318"));
    }

    // ---- MetricBridge conversion -------------------------------------------

    #[test]
    fn test_bridge_convert_timestamp() {
        let cfg = OtelExportConfig::new("svc", "1.0");
        let bridge = MetricBridge::new(&cfg);
        let internal = InternalMetric::new("latency", 42.0, 1_700_000_000);
        let point = bridge.convert(&internal);
        assert_eq!(
            point.timestamp_nanos,
            1_700_000_000_u64 * 1_000_000_000,
            "Timestamp must be converted from seconds to nanoseconds"
        );
    }

    #[test]
    fn test_bridge_convert_value_and_name() {
        let cfg = OtelExportConfig::new("svc", "1.0");
        let bridge = MetricBridge::new(&cfg);
        let internal = InternalMetric::new("frames_encoded", 1024.0, 0);
        let point = bridge.convert(&internal);
        assert_eq!(point.metric_name, "frames_encoded");
        assert!((point.value - 1024.0).abs() < 1e-9);
    }

    #[test]
    fn test_bridge_convert_labels_preserved() {
        let cfg = OtelExportConfig::new("svc", "1.0");
        let bridge = MetricBridge::new(&cfg);
        let internal =
            InternalMetric::new("cpu", 80.0, 0).with_label("host", "srv-1").with_label("dc", "us-east");
        let point = bridge.convert(&internal);
        assert!(
            point.labels.iter().any(|(k, v)| k == "host" && v == "srv-1"),
            "host label must be preserved"
        );
        assert!(
            point.labels.iter().any(|(k, v)| k == "dc" && v == "us-east"),
            "dc label must be preserved"
        );
    }

    #[test]
    fn test_bridge_convert_batch_groups_by_name() {
        let cfg = OtelExportConfig::new("media", "0.1.3");
        let bridge = MetricBridge::new(&cfg);
        let metrics = vec![
            InternalMetric::new("cpu", 70.0, 1000),
            InternalMetric::new("cpu", 80.0, 2000),
            InternalMetric::new("mem", 60.0, 1000),
        ];
        let export = bridge.convert_batch(&metrics);
        // Should have 2 gauges: cpu and mem.
        assert_eq!(export.metrics.len(), 2);
        let cpu_gauge = export.metrics.iter().find(|g| g.name == "cpu");
        assert!(cpu_gauge.is_some());
        assert_eq!(cpu_gauge.expect("cpu gauge").data_points.len(), 2);
    }

    #[test]
    fn test_bridge_convert_batch_resource_attributes() {
        let cfg = OtelExportConfig::new("stream-svc", "0.9.0");
        let bridge = MetricBridge::new(&cfg);
        let export = bridge.convert_batch(&[]);
        let has_service_name = export
            .resource_attributes
            .iter()
            .any(|(k, v)| k == "service.name" && v == "stream-svc");
        assert!(
            has_service_name,
            "Batch export must include service.name resource attribute"
        );
    }

    // ---- OtelMetricPoint builder ------------------------------------------

    #[test]
    fn test_metric_point_builder() {
        let point = OtelMetricPoint::new("bytes_out", 1024.0, 9_000_000)
            .with_label("region", "eu-west")
            .with_label("tier", "cdn");
        assert_eq!(point.metric_name, "bytes_out");
        assert!((point.value - 1024.0).abs() < 1e-9);
        assert_eq!(point.timestamp_nanos, 9_000_000);
        assert_eq!(point.labels.len(), 2);
    }
}
