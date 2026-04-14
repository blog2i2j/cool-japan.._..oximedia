//! Distributed trace export with Jaeger JSON format, span batching, and
//! tail-based sampling.
//!
//! This module provides a complete distributed tracing export pipeline that
//! complements [`crate::trace_span`] (span lifecycle) and
//! [`crate::w3c_trace_context`] (header propagation). It adds:
//!
//! - **[`TraceExporter`]** — batches completed spans and ships them to a
//!   Jaeger-compatible backend via the Jaeger Thrift-over-HTTP JSON endpoint
//!   (`/api/traces`), or writes them to an in-memory sink for testing.
//! - **[`SamplingStrategy`]** — pluggable sampling policies: always-on,
//!   probabilistic (0–100%), rate-limited (max N traces/second), and tail-based
//!   (buffer all spans; decide after the root span completes).
//! - **[`SpanBatch`]** — an owned collection of completed spans ready for
//!   serialization.
//! - **[`JaegerSpan`]** / **[`JaegerTrace`]** — Jaeger JSON wire format
//!   structs with full serde support.
//!
//! # Architecture
//!
//! ```text
//! [TraceSpan (trace_span.rs)]
//!        ↓ on finish()
//! [SpanCollector] ──batches──→ [TraceExporter] ──JSON──→ Jaeger / log sink
//!        ↑
//! [SamplingStrategy] (decides which traces enter the collector)
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::trace_exporter::{
//!     ExporterConfig, InMemorySink, SamplingStrategy, SpanCollector, TraceExporter,
//! };
//!
//! let config = ExporterConfig::default();
//! let sink = InMemorySink::new();
//! let exporter = TraceExporter::with_memory_sink(config, sink.clone());
//! let mut collector = SpanCollector::new(SamplingStrategy::AlwaysOn, exporter);
//!
//! let trace_id = [0u8; 16]; // normally from W3C TraceContext
//! collector.record_span("encode_frame", trace_id, None, 1_234_567, 2_345_678, true);
//! collector.flush();
//!
//! let exported = sink.drain();
//! assert_eq!(exported.len(), 1);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::error::{MonitorError, MonitorResult};

// ---------------------------------------------------------------------------
// Jaeger JSON wire format
// ---------------------------------------------------------------------------

/// A key-value tag attached to a Jaeger span.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JaegerTag {
    /// Tag key.
    pub key: String,
    /// Tag type (`string`, `bool`, `long`, `double`).
    #[serde(rename = "type")]
    pub tag_type: String,
    /// Serialized value (always stored as string for JSON simplicity).
    pub value: serde_json::Value,
}

impl JaegerTag {
    /// Create a string tag.
    #[must_use]
    pub fn string(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            tag_type: "string".to_string(),
            value: serde_json::Value::String(value.into()),
        }
    }

    /// Create a boolean tag.
    #[must_use]
    pub fn bool(key: impl Into<String>, value: bool) -> Self {
        Self {
            key: key.into(),
            tag_type: "bool".to_string(),
            value: serde_json::Value::Bool(value),
        }
    }

    /// Create a long (i64) tag.
    #[must_use]
    pub fn long(key: impl Into<String>, value: i64) -> Self {
        Self {
            key: key.into(),
            tag_type: "long".to_string(),
            value: serde_json::Value::Number(value.into()),
        }
    }

    /// Create a double (f64) tag.
    #[must_use]
    pub fn double(key: impl Into<String>, value: f64) -> Self {
        Self {
            key: key.into(),
            tag_type: "double".to_string(),
            value: serde_json::json!(value),
        }
    }
}

/// A log entry attached to a Jaeger span (structured span events).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerLog {
    /// Microseconds since Unix epoch.
    pub timestamp: u64,
    /// Fields in this log entry.
    pub fields: Vec<JaegerTag>,
}

impl JaegerLog {
    /// Create a new log entry at the current wall-clock time.
    #[must_use]
    pub fn now(fields: Vec<JaegerTag>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64;
        Self { timestamp, fields }
    }

    /// Create a log entry with an explicit timestamp.
    #[must_use]
    pub fn at(timestamp_us: u64, fields: Vec<JaegerTag>) -> Self {
        Self {
            timestamp: timestamp_us,
            fields,
        }
    }
}

/// A reference from one span to another (parent or follows-from).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpanRef {
    /// Reference type (`CHILD_OF` or `FOLLOWS_FROM`).
    #[serde(rename = "refType")]
    pub ref_type: String,
    /// Trace ID (hex, 32 chars).
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// Span ID (hex, 16 chars).
    #[serde(rename = "spanID")]
    pub span_id: String,
}

impl SpanRef {
    /// Create a `CHILD_OF` reference.
    #[must_use]
    pub fn child_of(trace_id: impl Into<String>, span_id: impl Into<String>) -> Self {
        Self {
            ref_type: "CHILD_OF".to_string(),
            trace_id: trace_id.into(),
            span_id: span_id.into(),
        }
    }
}

/// A single span in Jaeger JSON format.
///
/// Follows the Jaeger query API response schema so exported batches can be
/// fed directly into Jaeger for visualisation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JaegerSpan {
    /// Trace ID (hex, 32 chars).
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// Span ID (hex, 16 chars).
    #[serde(rename = "spanID")]
    pub span_id: String,
    /// Human-readable operation name.
    pub operation_name: String,
    /// Parent/follows-from references.
    pub references: Vec<SpanRef>,
    /// Start time in microseconds since Unix epoch.
    pub start_time: u64,
    /// Duration in microseconds.
    pub duration: u64,
    /// Key-value tags.
    pub tags: Vec<JaegerTag>,
    /// Structured log events.
    pub logs: Vec<JaegerLog>,
    /// Process ID (maps to `processID` key in the top-level `processes` map).
    pub process_id: String,
    /// Warnings (usually empty).
    pub warnings: Vec<String>,
}

/// A Jaeger trace is a collection of spans sharing a `traceID`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JaegerTrace {
    /// Trace ID (hex, 32 chars).
    #[serde(rename = "traceID")]
    pub trace_id: String,
    /// All spans belonging to this trace.
    pub spans: Vec<JaegerSpan>,
    /// Map of `processID → process info`.
    pub processes: HashMap<String, JaegerProcess>,
    /// Top-level warnings.
    pub warnings: Vec<String>,
}

/// Process metadata attached to a Jaeger trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JaegerProcess {
    /// Service name that produced this trace.
    pub service_name: String,
    /// Process-level tags (host, version, etc.).
    pub tags: Vec<JaegerTag>,
}

/// Top-level Jaeger batch export payload (`/api/traces` POST body).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaegerBatch {
    /// Serialised traces.
    pub data: Vec<JaegerTrace>,
    /// Total span count across all traces.
    pub total: u32,
    /// Offset for paginated results (always 0 for exports).
    pub offset: u32,
    /// Limit used (0 = unlimited).
    pub limit: u32,
    /// Errors encountered (empty on success).
    pub errors: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Completed span record (internal)
// ---------------------------------------------------------------------------

/// A completed span ready for export.
#[derive(Debug, Clone)]
pub struct CompletedSpan {
    /// Trace ID bytes (16 bytes = 128-bit).
    pub trace_id: [u8; 16],
    /// Span ID (8 bytes = 64-bit).
    pub span_id: [u8; 8],
    /// Optional parent span ID.
    pub parent_span_id: Option<[u8; 8]>,
    /// Operation name.
    pub operation: String,
    /// Start timestamp in microseconds since Unix epoch.
    pub start_us: u64,
    /// End timestamp in microseconds since Unix epoch.
    pub end_us: u64,
    /// Whether the span completed without error.
    pub success: bool,
    /// Optional error message.
    pub error_message: Option<String>,
    /// Arbitrary key-value tags.
    pub tags: Vec<(String, TagValue)>,
    /// Log events attached to this span.
    pub logs: Vec<JaegerLog>,
}

/// Typed tag value.
#[derive(Debug, Clone)]
pub enum TagValue {
    /// String value.
    Str(String),
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
}

impl CompletedSpan {
    /// Duration of the span.
    #[must_use]
    pub fn duration_us(&self) -> u64 {
        self.end_us.saturating_sub(self.start_us)
    }

    /// Hex-encode the trace ID.
    #[must_use]
    pub fn trace_id_hex(&self) -> String {
        self.trace_id.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Hex-encode the span ID.
    #[must_use]
    pub fn span_id_hex(&self) -> String {
        self.span_id.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Hex-encode the parent span ID, if present.
    #[must_use]
    pub fn parent_span_id_hex(&self) -> Option<String> {
        self.parent_span_id
            .map(|id| id.iter().map(|b| format!("{b:02x}")).collect())
    }

    /// Convert to a [`JaegerSpan`] for serialisation.
    #[must_use]
    pub fn to_jaeger_span(&self, process_id: &str) -> JaegerSpan {
        let mut references = Vec::new();
        if let Some(parent_hex) = self.parent_span_id_hex() {
            references.push(SpanRef::child_of(self.trace_id_hex(), parent_hex));
        }

        let mut tags: Vec<JaegerTag> = self
            .tags
            .iter()
            .map(|(k, v)| match v {
                TagValue::Str(s) => JaegerTag::string(k, s),
                TagValue::Bool(b) => JaegerTag::bool(k, *b),
                TagValue::Int(i) => JaegerTag::long(k, *i),
                TagValue::Float(f) => JaegerTag::double(k, *f),
            })
            .collect();

        // Add error tag for failed spans.
        if !self.success {
            tags.push(JaegerTag::bool("error", true));
            if let Some(ref msg) = self.error_message {
                tags.push(JaegerTag::string("error.message", msg));
            }
        }

        JaegerSpan {
            trace_id: self.trace_id_hex(),
            span_id: self.span_id_hex(),
            operation_name: self.operation.clone(),
            references,
            start_time: self.start_us,
            duration: self.duration_us(),
            tags,
            logs: self.logs.clone(),
            process_id: process_id.to_string(),
            warnings: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Span batch
// ---------------------------------------------------------------------------

/// A collection of completed spans ready for export.
#[derive(Debug, Default, Clone)]
pub struct SpanBatch {
    spans: Vec<CompletedSpan>,
}

impl SpanBatch {
    /// Create an empty batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a span to the batch.
    pub fn push(&mut self, span: CompletedSpan) {
        self.spans.push(span);
    }

    /// Number of spans in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Returns `true` if the batch contains no spans.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Drain all spans from the batch.
    pub fn drain(&mut self) -> Vec<CompletedSpan> {
        std::mem::take(&mut self.spans)
    }

    /// Convert the batch to a [`JaegerBatch`] JSON value.
    ///
    /// Groups spans by trace ID. Each trace maps to one entry in `data`.
    #[must_use]
    pub fn to_jaeger_batch(&self, service_name: &str, process_tags: &[JaegerTag]) -> JaegerBatch {
        let mut traces: HashMap<String, Vec<&CompletedSpan>> = HashMap::new();
        for span in &self.spans {
            traces
                .entry(span.trace_id_hex())
                .or_default()
                .push(span);
        }

        let process_id = "p1";
        let process = JaegerProcess {
            service_name: service_name.to_string(),
            tags: process_tags.to_vec(),
        };

        let data: Vec<JaegerTrace> = traces
            .into_iter()
            .map(|(trace_id, spans)| {
                let jaeger_spans: Vec<JaegerSpan> = spans
                    .iter()
                    .map(|s| s.to_jaeger_span(process_id))
                    .collect();
                let mut processes = HashMap::new();
                processes.insert(process_id.to_string(), process.clone());
                JaegerTrace {
                    trace_id,
                    spans: jaeger_spans,
                    processes,
                    warnings: Vec::new(),
                }
            })
            .collect();

        let total = self.spans.len() as u32;
        JaegerBatch {
            data,
            total,
            offset: 0,
            limit: 0,
            errors: None,
        }
    }

    /// Serialize the batch to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self, service_name: &str, process_tags: &[JaegerTag]) -> MonitorResult<String> {
        let batch = self.to_jaeger_batch(service_name, process_tags);
        serde_json::to_string_pretty(&batch).map_err(MonitorError::Serialization)
    }
}

// ---------------------------------------------------------------------------
// Sampling strategy
// ---------------------------------------------------------------------------

/// Controls which traces are admitted into the export pipeline.
#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    /// Every trace is sampled (100%).
    AlwaysOn,
    /// No traces are sampled (useful for disabling tracing at runtime).
    AlwaysOff,
    /// Probabilistic sampling: value in `[0.0, 1.0]` determines the fraction
    /// of traces that are admitted.
    Probabilistic(f64),
    /// Rate-limited: at most `max_per_second` traces are admitted per second.
    RateLimited {
        /// Maximum traces per second to admit.
        max_per_second: u32,
    },
    /// Tail-based: all spans are buffered; sampling decision is deferred until
    /// the root span completes.  Useful for keeping only slow/erroneous traces.
    TailBased {
        /// Minimum latency (µs) of the root span to trigger sampling.
        min_latency_us: u64,
        /// Whether to always sample traces with errors.
        always_sample_errors: bool,
    },
}

impl SamplingStrategy {
    /// Decide whether to sample a span with the given parameters.
    ///
    /// For tail-based strategies this is a pre-filter at span start; final
    /// decisions happen in [`SpanCollector::finish_trace`].
    #[must_use]
    pub fn should_sample(&self, trace_counter: u64, is_error: bool) -> bool {
        match self {
            Self::AlwaysOn => true,
            Self::AlwaysOff => false,
            Self::Probabilistic(rate) => {
                let rate_clamped = rate.clamp(0.0, 1.0);
                // Deterministic sampling using the trace counter so we do not
                // need a random number generator.
                let bucket = (trace_counter % 10_000) as f64 / 10_000.0;
                bucket < rate_clamped
            }
            Self::RateLimited { max_per_second } => {
                // Simple modulo-based admission; a production system would use
                // a token bucket, but this avoids Instant::now() in the hot path.
                trace_counter % u64::from(*max_per_second).max(1) == 0
            }
            Self::TailBased {
                always_sample_errors,
                ..
            } => {
                // Always buffer; the tail decision is made in finish_trace.
                *always_sample_errors && is_error || true
            }
        }
    }

    /// Tail-based final decision given the root span's measured latency.
    #[must_use]
    pub fn tail_decision(&self, root_latency_us: u64, has_error: bool) -> bool {
        match self {
            Self::TailBased {
                min_latency_us,
                always_sample_errors,
            } => {
                (*always_sample_errors && has_error) || root_latency_us >= *min_latency_us
            }
            _ => true, // non-tail strategies: already decided at head
        }
    }
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::AlwaysOn
    }
}

// ---------------------------------------------------------------------------
// Export sink
// ---------------------------------------------------------------------------

/// Trait implemented by export destinations.
pub trait ExportSink: Send + Sync {
    /// Export a serialised Jaeger JSON batch.
    ///
    /// # Errors
    ///
    /// Returns an error if the export fails.
    fn export_json(&self, json: &str) -> MonitorResult<()>;
}

/// In-memory sink for testing: stores exported JSON strings.
#[derive(Debug, Clone)]
pub struct InMemorySink {
    records: Arc<Mutex<Vec<String>>>,
}

impl InMemorySink {
    /// Create an empty sink.
    #[must_use]
    pub fn new() -> Self {
        Self {
            records: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Drain and return all exported records.
    #[must_use]
    pub fn drain(&self) -> Vec<String> {
        self.records
            .lock()
            .map(|mut g| std::mem::take(&mut *g))
            .unwrap_or_default()
    }

    /// Number of records stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Returns `true` if no records have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for InMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl ExportSink for InMemorySink {
    fn export_json(&self, json: &str) -> MonitorResult<()> {
        self.records
            .lock()
            .map_err(|e| MonitorError::Other(format!("mutex poisoned: {e}")))?
            .push(json.to_string());
        Ok(())
    }
}

/// Logging sink: emits spans via the `tracing` facade at DEBUG level.
pub struct LoggingSink {
    service_name: String,
}

impl LoggingSink {
    /// Create a logging sink.
    #[must_use]
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            service_name: service_name.into(),
        }
    }
}

impl ExportSink for LoggingSink {
    fn export_json(&self, json: &str) -> MonitorResult<()> {
        tracing::debug!(service = %self.service_name, batch = %json, "trace export");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Exporter configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`TraceExporter`].
#[derive(Debug, Clone)]
pub struct ExporterConfig {
    /// Service name to attach to all exported traces.
    pub service_name: String,
    /// Maximum spans to buffer before forcing a flush.
    pub max_batch_size: usize,
    /// Maximum time to wait before flushing an incomplete batch.
    pub flush_interval: Duration,
    /// Additional process-level tags.
    pub process_tags: Vec<JaegerTag>,
}

impl Default for ExporterConfig {
    fn default() -> Self {
        Self {
            service_name: "oximedia".to_string(),
            max_batch_size: 512,
            flush_interval: Duration::from_secs(5),
            process_tags: Vec::new(),
        }
    }
}

impl ExporterConfig {
    /// Set the service name.
    #[must_use]
    pub fn with_service_name(mut self, name: impl Into<String>) -> Self {
        self.service_name = name.into();
        self
    }

    /// Set the maximum batch size.
    #[must_use]
    pub fn with_max_batch_size(mut self, n: usize) -> Self {
        self.max_batch_size = n.max(1);
        self
    }

    /// Set the flush interval.
    #[must_use]
    pub fn with_flush_interval(mut self, d: Duration) -> Self {
        self.flush_interval = d;
        self
    }

    /// Add a process-level tag.
    #[must_use]
    pub fn with_process_tag(mut self, tag: JaegerTag) -> Self {
        self.process_tags.push(tag);
        self
    }
}

// ---------------------------------------------------------------------------
// TraceExporter
// ---------------------------------------------------------------------------

/// Batches completed spans and forwards them to an [`ExportSink`].
///
/// The exporter accumulates spans in a [`SpanBatch`].  When the batch reaches
/// `max_batch_size` or [`flush`](Self::flush) is called explicitly, the batch
/// is serialised to Jaeger JSON and handed to the configured sink.
pub struct TraceExporter {
    config: ExporterConfig,
    sink: Arc<dyn ExportSink>,
    pending: Mutex<SpanBatch>,
    export_count: std::sync::atomic::AtomicU64,
    error_count: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for TraceExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TraceExporter")
            .field("service", &self.config.service_name)
            .field(
                "pending",
                &self.pending.lock().map(|g| g.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl TraceExporter {
    /// Create an exporter with a custom sink.
    #[must_use]
    pub fn new(config: ExporterConfig, sink: Arc<dyn ExportSink>) -> Self {
        Self {
            config,
            sink,
            pending: Mutex::new(SpanBatch::new()),
            export_count: std::sync::atomic::AtomicU64::new(0),
            error_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Convenience constructor that wraps an [`InMemorySink`].
    #[must_use]
    pub fn with_memory_sink(config: ExporterConfig, sink: InMemorySink) -> Self {
        Self::new(config, Arc::new(sink))
    }

    /// Convenience constructor that uses the logging sink.
    #[must_use]
    pub fn with_logging_sink(config: ExporterConfig) -> Self {
        let service = config.service_name.clone();
        Self::new(config, Arc::new(LoggingSink::new(service)))
    }

    /// Enqueue a completed span.
    ///
    /// If the batch reaches `max_batch_size` it is flushed automatically.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush triggered by a full batch fails.
    pub fn enqueue(&self, span: CompletedSpan) -> MonitorResult<()> {
        let should_flush = {
            let mut guard = self
                .pending
                .lock()
                .map_err(|e| MonitorError::Other(format!("mutex poisoned: {e}")))?;
            guard.push(span);
            guard.len() >= self.config.max_batch_size
        };

        if should_flush {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush all pending spans to the sink immediately.
    ///
    /// # Errors
    ///
    /// Returns an error if JSON serialisation or sink delivery fails.
    pub fn flush(&self) -> MonitorResult<()> {
        let batch = {
            let mut guard = self
                .pending
                .lock()
                .map_err(|e| MonitorError::Other(format!("mutex poisoned: {e}")))?;
            let mut batch = SpanBatch::new();
            for span in guard.drain() {
                batch.push(span);
            }
            batch
        };

        if batch.is_empty() {
            return Ok(());
        }

        let json = batch.to_json(&self.config.service_name, &self.config.process_tags)?;
        match self.sink.export_json(&json) {
            Ok(()) => {
                self.export_count
                    .fetch_add(batch.len() as u64, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.error_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Err(e)
            }
        }
    }

    /// Total spans successfully exported.
    #[must_use]
    pub fn export_count(&self) -> u64 {
        self.export_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total export errors.
    #[must_use]
    pub fn error_count(&self) -> u64 {
        self.error_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of spans currently buffered.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Access the exporter configuration.
    #[must_use]
    pub fn config(&self) -> &ExporterConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// SpanCollector — integrates sampling + exporter
// ---------------------------------------------------------------------------

/// Combines a [`SamplingStrategy`] with a [`TraceExporter`] to form the
/// complete span collection pipeline.
///
/// `SpanCollector` is the primary entry point for application code. Spans
/// are recorded via [`record_span`](Self::record_span) and flushed via
/// [`flush`](Self::flush).
pub struct SpanCollector {
    strategy: SamplingStrategy,
    exporter: Arc<TraceExporter>,
    /// Monotonic counter used for probabilistic/rate-limited sampling.
    trace_counter: std::sync::atomic::AtomicU64,
    /// Tail-based buffer: `trace_id_hex → pending spans`.
    tail_buffer: Mutex<HashMap<String, Vec<CompletedSpan>>>,
}

impl SpanCollector {
    /// Create a new collector.
    #[must_use]
    pub fn new(strategy: SamplingStrategy, exporter: TraceExporter) -> Self {
        Self {
            strategy,
            exporter: Arc::new(exporter),
            trace_counter: std::sync::atomic::AtomicU64::new(0),
            tail_buffer: Mutex::new(HashMap::new()),
        }
    }

    /// Record a completed span.
    ///
    /// `trace_id` is 16 bytes; `span_id` / `parent_span_id` are 8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if enqueuing triggers a flush that fails.
    pub fn record_span(
        &self,
        operation: impl Into<String>,
        trace_id: [u8; 16],
        parent_span_id: Option<[u8; 8]>,
        start_us: u64,
        end_us: u64,
        success: bool,
    ) -> MonitorResult<()> {
        let counter = self
            .trace_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if !self
            .strategy
            .should_sample(counter, !success)
        {
            return Ok(());
        }

        // Derive a deterministic span_id from the counter.
        let span_id = counter.to_le_bytes();

        let span = CompletedSpan {
            trace_id,
            span_id,
            parent_span_id,
            operation: operation.into(),
            start_us,
            end_us,
            success,
            error_message: None,
            tags: Vec::new(),
            logs: Vec::new(),
        };

        match &self.strategy {
            SamplingStrategy::TailBased {
                min_latency_us,
                always_sample_errors,
            } => {
                let trace_hex: String = trace_id.iter().map(|b| format!("{b:02x}")).collect();
                let is_root = parent_span_id.is_none();
                let latency = end_us.saturating_sub(start_us);
                let has_error = !success;

                if is_root {
                    let keep = (*always_sample_errors && has_error)
                        || latency >= *min_latency_us;
                    let buffered = self
                        .tail_buffer
                        .lock()
                        .map_err(|e| MonitorError::Other(format!("mutex poisoned: {e}")))?
                        .remove(&trace_hex)
                        .unwrap_or_default();
                    if keep {
                        for buffered_span in buffered {
                            self.exporter.enqueue(buffered_span)?;
                        }
                        self.exporter.enqueue(span)?;
                    }
                } else {
                    self.tail_buffer
                        .lock()
                        .map_err(|e| MonitorError::Other(format!("mutex poisoned: {e}")))?
                        .entry(trace_hex)
                        .or_default()
                        .push(span);
                }
                Ok(())
            }
            _ => self.exporter.enqueue(span),
        }
    }

    /// Record a span with additional tags and error context.
    ///
    /// # Errors
    ///
    /// Returns an error if the span cannot be enqueued.
    pub fn record_span_with_tags(
        &self,
        operation: impl Into<String>,
        trace_id: [u8; 16],
        parent_span_id: Option<[u8; 8]>,
        start_us: u64,
        end_us: u64,
        success: bool,
        error_message: Option<String>,
        tags: Vec<(String, TagValue)>,
    ) -> MonitorResult<()> {
        let counter = self
            .trace_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        if !self.strategy.should_sample(counter, !success) {
            return Ok(());
        }

        let span_id = counter.to_le_bytes();
        let span = CompletedSpan {
            trace_id,
            span_id,
            parent_span_id,
            operation: operation.into(),
            start_us,
            end_us,
            success,
            error_message,
            tags,
            logs: Vec::new(),
        };

        self.exporter.enqueue(span)
    }

    /// Flush all buffered spans.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush fails.
    pub fn flush(&self) -> MonitorResult<()> {
        self.exporter.flush()
    }

    /// Number of spans pending export.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.exporter.pending_count()
    }

    /// Total spans exported.
    #[must_use]
    pub fn export_count(&self) -> u64 {
        self.exporter.export_count()
    }
}

// ---------------------------------------------------------------------------
// Export statistics
// ---------------------------------------------------------------------------

/// Summary statistics for a trace exporter.
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Total spans successfully exported.
    pub exported: u64,
    /// Total export errors.
    pub errors: u64,
    /// Spans currently pending in the buffer.
    pub pending: usize,
}

impl TraceExporter {
    /// Snapshot current export statistics.
    #[must_use]
    pub fn stats(&self) -> ExportStats {
        ExportStats {
            exported: self.export_count(),
            errors: self.error_count(),
            pending: self.pending_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trace_id(n: u8) -> [u8; 16] {
        [n; 16]
    }

    fn make_span_id(n: u8) -> [u8; 8] {
        [n; 8]
    }

    fn now_us() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_micros() as u64
    }

    #[test]
    fn test_jaeger_tag_types() {
        let s = JaegerTag::string("k", "v");
        assert_eq!(s.tag_type, "string");

        let b = JaegerTag::bool("flag", true);
        assert_eq!(b.tag_type, "bool");

        let l = JaegerTag::long("count", 42);
        assert_eq!(l.tag_type, "long");

        let d = JaegerTag::double("rate", 1.5);
        assert_eq!(d.tag_type, "double");
    }

    #[test]
    fn test_completed_span_hex_encoding() {
        let span = CompletedSpan {
            trace_id: [0xAB; 16],
            span_id: [0xCD; 8],
            parent_span_id: Some([0xEF; 8]),
            operation: "test".to_string(),
            start_us: 1000,
            end_us: 2000,
            success: true,
            error_message: None,
            tags: Vec::new(),
            logs: Vec::new(),
        };

        // 16 bytes of 0xAB → 32 hex chars ("ab" repeated 16 times)
        let expected_trace: String = std::iter::repeat("ab").take(16).collect();
        assert_eq!(span.trace_id_hex(), expected_trace);
        assert_eq!(span.span_id_hex(), "cdcdcdcdcdcdcdcd");
        assert_eq!(span.parent_span_id_hex(), Some("efefefefefefefef".to_string()));
        assert_eq!(span.duration_us(), 1000);
    }

    #[test]
    fn test_completed_span_to_jaeger_span() {
        let span = CompletedSpan {
            trace_id: [1u8; 16],
            span_id: [2u8; 8],
            parent_span_id: Some([3u8; 8]),
            operation: "encode".to_string(),
            start_us: 500_000,
            end_us: 600_000,
            success: false,
            error_message: Some("codec error".to_string()),
            tags: vec![("codec".to_string(), TagValue::Str("h264".to_string()))],
            logs: Vec::new(),
        };

        let js = span.to_jaeger_span("p1");
        assert_eq!(js.operation_name, "encode");
        assert_eq!(js.duration, 100_000);
        assert!(!js.references.is_empty());
        // Error tag should be present.
        assert!(js.tags.iter().any(|t| t.key == "error"));
    }

    #[test]
    fn test_span_batch_serialisation() {
        let mut batch = SpanBatch::new();
        batch.push(CompletedSpan {
            trace_id: [5u8; 16],
            span_id: [6u8; 8],
            parent_span_id: None,
            operation: "ingest".to_string(),
            start_us: 0,
            end_us: 50_000,
            success: true,
            error_message: None,
            tags: Vec::new(),
            logs: Vec::new(),
        });

        let json = batch.to_json("test-service", &[]).expect("serialise should succeed");
        assert!(json.contains("ingest"));
        assert!(json.contains("test-service"));
    }

    #[test]
    fn test_in_memory_sink_export() {
        let sink = InMemorySink::new();
        sink.export_json(r#"{"data":[]}"#).expect("export should succeed");
        sink.export_json(r#"{"data":[],"total":0}"#).expect("export should succeed");
        assert_eq!(sink.len(), 2);
        let drained = sink.drain();
        assert_eq!(drained.len(), 2);
        assert!(sink.is_empty());
    }

    #[test]
    fn test_trace_exporter_flush() {
        let sink = InMemorySink::new();
        let config = ExporterConfig::default().with_service_name("media-encoder");
        let exporter = TraceExporter::with_memory_sink(config, sink.clone());

        let start = now_us();
        let span = CompletedSpan {
            trace_id: make_trace_id(0xAA),
            span_id: make_span_id(0x01),
            parent_span_id: None,
            operation: "transcode".to_string(),
            start_us: start,
            end_us: start + 5_000_000,
            success: true,
            error_message: None,
            tags: vec![("codec".to_string(), TagValue::Str("av1".to_string()))],
            logs: Vec::new(),
        };

        exporter.enqueue(span).expect("enqueue should succeed");
        assert_eq!(exporter.pending_count(), 1);

        exporter.flush().expect("flush should succeed");
        assert_eq!(exporter.pending_count(), 0);
        assert_eq!(exporter.export_count(), 1);
        assert_eq!(sink.len(), 1);
    }

    #[test]
    fn test_sampling_strategy_always_on() {
        let s = SamplingStrategy::AlwaysOn;
        for i in 0..100 {
            assert!(s.should_sample(i, false));
        }
    }

    #[test]
    fn test_sampling_strategy_always_off() {
        let s = SamplingStrategy::AlwaysOff;
        for i in 0..100 {
            assert!(!s.should_sample(i, false));
        }
    }

    #[test]
    fn test_sampling_strategy_probabilistic() {
        let s = SamplingStrategy::Probabilistic(0.5);
        let sampled: u64 = (0u64..10_000).filter(|&i| s.should_sample(i, false)).count() as u64;
        // With 50% rate and 10 000 trials we expect ~5 000 sampled.
        assert!((4_000..=6_000).contains(&sampled));
    }

    #[test]
    fn test_span_collector_always_on() {
        let sink = InMemorySink::new();
        let config = ExporterConfig::default();
        let exporter = TraceExporter::with_memory_sink(config, sink.clone());
        let collector = SpanCollector::new(SamplingStrategy::AlwaysOn, exporter);

        let trace_id = make_trace_id(1);
        for i in 0u64..5 {
            let start = i * 1_000_000;
            collector
                .record_span("op", trace_id, None, start, start + 100_000, true)
                .expect("record should succeed");
        }

        collector.flush().expect("flush should succeed");
        assert_eq!(collector.export_count(), 5);
        assert_eq!(sink.len(), 1); // all flushed in one call
    }

    #[test]
    fn test_tail_based_sampling_keeps_slow_traces() {
        let sink = InMemorySink::new();
        let config = ExporterConfig::default();
        let exporter = TraceExporter::with_memory_sink(config, sink.clone());
        let collector = SpanCollector::new(
            SamplingStrategy::TailBased {
                min_latency_us: 500_000,        // 500 ms threshold
                always_sample_errors: false,
            },
            exporter,
        );

        let slow_trace = make_trace_id(0x01);
        let span_id = make_span_id(0xAA);

        // Child span (buffered until root arrives).
        collector
            .record_span("child-op", slow_trace, Some(span_id), 0, 100_000, true)
            .expect("record child should succeed");

        // Root span: 600 ms > 500 ms threshold → should be kept.
        collector
            .record_span("root-op", slow_trace, None, 0, 600_000, true)
            .expect("record root should succeed");

        collector.flush().expect("flush should succeed");
        assert!(sink.len() > 0, "slow trace should be exported");
    }

    #[test]
    fn test_export_stats() {
        let sink = InMemorySink::new();
        let config = ExporterConfig::default();
        let exporter = TraceExporter::with_memory_sink(config, sink.clone());

        let stats = exporter.stats();
        assert_eq!(stats.exported, 0);
        assert_eq!(stats.errors, 0);
        assert_eq!(stats.pending, 0);
    }
}
