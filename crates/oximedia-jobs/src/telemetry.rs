//! Job telemetry: execution traces, span events, and metrics emission.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Span events ───────────────────────────────────────────────────────────────

/// The kind of event recorded in a span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpanEventKind {
    /// Job entered the queue.
    Queued,
    /// Job picked up by a worker.
    Started,
    /// A progress checkpoint.
    Progress,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
    /// Custom user-defined event.
    Custom(String),
}

impl std::fmt::Display for SpanEventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "Queued"),
            Self::Started => write!(f, "Started"),
            Self::Progress => write!(f, "Progress"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Custom(s) => write!(f, "Custom({s})"),
        }
    }
}

/// A single timestamped event within a span.
#[derive(Clone, Debug)]
pub struct SpanEvent {
    /// Kind of event.
    pub kind: SpanEventKind,
    /// Elapsed time since span start.
    pub elapsed: Duration,
    /// Optional human-readable message.
    pub message: Option<String>,
    /// Key-value attributes attached to the event.
    pub attributes: HashMap<String, String>,
}

impl SpanEvent {
    /// Create a minimal span event.
    #[must_use]
    pub fn new(kind: SpanEventKind, elapsed: Duration) -> Self {
        Self {
            kind,
            elapsed,
            message: None,
            attributes: HashMap::new(),
        }
    }

    /// Attach a message.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Attach a key-value attribute.
    #[must_use]
    pub fn with_attr(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.attributes.insert(k.into(), v.into());
        self
    }
}

// ── Span (execution trace for one job) ───────────────────────────────────────

/// An execution trace for a single job run.
pub struct JobSpan {
    /// Job identifier.
    pub job_id: String,
    /// Worker that executed the job.
    pub worker_id: Option<String>,
    /// Monotonic start time.
    start: Instant,
    /// Recorded events, in order.
    pub events: Vec<SpanEvent>,
    /// Whether the span has been closed.
    pub closed: bool,
}

impl JobSpan {
    /// Start a new span for `job_id`.
    #[must_use]
    pub fn start(job_id: impl Into<String>) -> Self {
        let mut span = Self {
            job_id: job_id.into(),
            worker_id: None,
            start: Instant::now(),
            events: Vec::new(),
            closed: false,
        };
        span.record(SpanEventKind::Queued, None);
        span
    }

    /// Set the worker ID.
    pub fn set_worker(&mut self, worker_id: impl Into<String>) {
        self.worker_id = Some(worker_id.into());
    }

    /// Record an event.
    pub fn record(&mut self, kind: SpanEventKind, message: Option<&str>) {
        let elapsed = self.start.elapsed();
        let mut evt = SpanEvent::new(kind, elapsed);
        if let Some(m) = message {
            evt.message = Some(m.to_string());
        }
        self.events.push(evt);
    }

    /// Record a progress checkpoint.
    pub fn record_progress(&mut self, percent: u8) {
        let elapsed = self.start.elapsed();
        let evt = SpanEvent::new(SpanEventKind::Progress, elapsed)
            .with_attr("percent", percent.to_string());
        self.events.push(evt);
    }

    /// Close the span with a `Completed` event.
    pub fn complete(&mut self) {
        self.record(SpanEventKind::Completed, None);
        self.closed = true;
    }

    /// Close the span with a `Failed` event.
    pub fn fail(&mut self, reason: &str) {
        self.record(SpanEventKind::Failed, Some(reason));
        self.closed = true;
    }

    /// Total duration from span start to now (or to close event).
    #[must_use]
    pub fn duration(&self) -> Duration {
        if let Some(last) = self.events.last() {
            last.elapsed
        } else {
            self.start.elapsed()
        }
    }

    /// Returns all events of the given kind.
    #[must_use]
    pub fn events_of_kind(&self, kind: &SpanEventKind) -> Vec<&SpanEvent> {
        self.events.iter().filter(|e| &e.kind == kind).collect()
    }
}

// ── Metrics counters ──────────────────────────────────────────────────────────

/// Simple atomic-style in-memory metrics counters.
#[derive(Clone, Debug, Default)]
pub struct TelemetryMetrics {
    /// Map from metric name to value.
    counters: HashMap<String, i64>,
    /// Histograms: name → list of samples.
    histograms: HashMap<String, Vec<f64>>,
}

impl TelemetryMetrics {
    /// Create empty metrics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment a counter by `delta`.
    pub fn increment(&mut self, name: &str, delta: i64) {
        *self.counters.entry(name.to_string()).or_insert(0) += delta;
    }

    /// Get the current value of a counter.
    #[must_use]
    pub fn get_counter(&self, name: &str) -> i64 {
        self.counters.get(name).copied().unwrap_or(0)
    }

    /// Record a histogram sample.
    pub fn record_sample(&mut self, name: &str, value: f64) {
        self.histograms
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Compute the mean of a histogram, returning `None` if empty.
    #[must_use]
    pub fn mean(&self, name: &str) -> Option<f64> {
        let samples = self.histograms.get(name)?;
        if samples.is_empty() {
            return None;
        }
        let sum: f64 = samples.iter().sum();
        Some(sum / samples.len() as f64)
    }

    /// Compute the p95 of a histogram, returning `None` if empty.
    #[must_use]
    pub fn p95(&self, name: &str) -> Option<f64> {
        let mut samples = self.histograms.get(name)?.clone();
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (samples.len() as f64 * 0.95) as usize;
        let idx = idx.min(samples.len() - 1);
        Some(samples[idx])
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.counters.clear();
        self.histograms.clear();
    }
}

// ── Telemetry collector ───────────────────────────────────────────────────────

/// Collects and stores job spans and metrics.
#[derive(Default)]
pub struct TelemetryCollector {
    /// Completed spans, keyed by job ID.
    spans: Vec<JobSpan>,
    /// Shared metrics counters.
    pub metrics: TelemetryMetrics,
}

impl TelemetryCollector {
    /// Create a new collector.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a closed span.
    pub fn submit_span(&mut self, span: JobSpan) {
        let duration_secs = span.duration().as_secs_f64();
        let succeeded = span.events_of_kind(&SpanEventKind::Completed).len() > 0;

        self.metrics.increment("jobs.total", 1);
        if succeeded {
            self.metrics.increment("jobs.succeeded", 1);
        } else {
            self.metrics.increment("jobs.failed", 1);
        }
        self.metrics
            .record_sample("job.duration_secs", duration_secs);
        self.spans.push(span);
    }

    /// Returns all stored spans.
    #[must_use]
    pub fn spans(&self) -> &[JobSpan] {
        &self.spans
    }

    /// Returns spans for a given job ID.
    #[must_use]
    pub fn spans_for_job(&self, job_id: &str) -> Vec<&JobSpan> {
        self.spans.iter().filter(|s| s.job_id == job_id).collect()
    }

    /// Total number of spans submitted.
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_event_kind_display() {
        assert_eq!(SpanEventKind::Completed.to_string(), "Completed");
        assert_eq!(
            SpanEventKind::Custom("ping".to_string()).to_string(),
            "Custom(ping)"
        );
    }

    #[test]
    fn test_span_event_with_message() {
        let evt = SpanEvent::new(SpanEventKind::Failed, Duration::from_millis(100))
            .with_message("timeout");
        assert_eq!(evt.message.expect("test expectation failed"), "timeout");
    }

    #[test]
    fn test_span_event_with_attr() {
        let evt = SpanEvent::new(SpanEventKind::Progress, Duration::ZERO).with_attr("pct", "50");
        assert_eq!(evt.attributes.get("pct").expect("get should succeed"), "50");
    }

    #[test]
    fn test_job_span_starts_with_queued_event() {
        let span = JobSpan::start("job-1");
        assert_eq!(span.events.len(), 1);
        assert_eq!(span.events[0].kind, SpanEventKind::Queued);
    }

    #[test]
    fn test_job_span_complete() {
        let mut span = JobSpan::start("job-1");
        span.record(SpanEventKind::Started, None);
        span.complete();
        assert!(span.closed);
        assert!(!span.events_of_kind(&SpanEventKind::Completed).is_empty());
    }

    #[test]
    fn test_job_span_fail() {
        let mut span = JobSpan::start("job-1");
        span.fail("disk full");
        assert!(span.closed);
        let failed = span.events_of_kind(&SpanEventKind::Failed);
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].message.as_deref(), Some("disk full"));
    }

    #[test]
    fn test_job_span_progress() {
        let mut span = JobSpan::start("job-1");
        span.record_progress(50);
        let prog = span.events_of_kind(&SpanEventKind::Progress);
        assert_eq!(prog.len(), 1);
        assert_eq!(
            prog[0]
                .attributes
                .get("percent")
                .expect("get should succeed"),
            "50"
        );
    }

    #[test]
    fn test_job_span_duration_positive() {
        let mut span = JobSpan::start("job-1");
        span.complete();
        assert!(span.duration() >= Duration::ZERO);
    }

    #[test]
    fn test_job_span_set_worker() {
        let mut span = JobSpan::start("job-1");
        span.set_worker("worker-A");
        assert_eq!(span.worker_id.expect("test expectation failed"), "worker-A");
    }

    #[test]
    fn test_telemetry_metrics_counter() {
        let mut m = TelemetryMetrics::new();
        m.increment("jobs", 1);
        m.increment("jobs", 2);
        assert_eq!(m.get_counter("jobs"), 3);
    }

    #[test]
    fn test_telemetry_metrics_missing_counter() {
        let m = TelemetryMetrics::new();
        assert_eq!(m.get_counter("nonexistent"), 0);
    }

    #[test]
    fn test_telemetry_metrics_mean() {
        let mut m = TelemetryMetrics::new();
        m.record_sample("lat", 1.0);
        m.record_sample("lat", 3.0);
        assert!((m.mean("lat").expect("mean should succeed") - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_telemetry_metrics_p95() {
        let mut m = TelemetryMetrics::new();
        for i in 1..=100 {
            m.record_sample("dur", i as f64);
        }
        let p95 = m.p95("dur").expect("p95 should be valid");
        assert!(p95 >= 95.0);
    }

    #[test]
    fn test_telemetry_metrics_reset() {
        let mut m = TelemetryMetrics::new();
        m.increment("x", 5);
        m.reset();
        assert_eq!(m.get_counter("x"), 0);
    }

    #[test]
    fn test_collector_submit_span_succeeded() {
        let mut col = TelemetryCollector::new();
        let mut span = JobSpan::start("job-1");
        span.complete();
        col.submit_span(span);
        assert_eq!(col.span_count(), 1);
        assert_eq!(col.metrics.get_counter("jobs.succeeded"), 1);
        assert_eq!(col.metrics.get_counter("jobs.failed"), 0);
    }

    #[test]
    fn test_collector_submit_span_failed() {
        let mut col = TelemetryCollector::new();
        let mut span = JobSpan::start("job-2");
        span.fail("oom");
        col.submit_span(span);
        assert_eq!(col.metrics.get_counter("jobs.failed"), 1);
    }

    #[test]
    fn test_collector_spans_for_job() {
        let mut col = TelemetryCollector::new();
        let mut s1 = JobSpan::start("job-A");
        s1.complete();
        let mut s2 = JobSpan::start("job-B");
        s2.complete();
        col.submit_span(s1);
        col.submit_span(s2);
        assert_eq!(col.spans_for_job("job-A").len(), 1);
        assert_eq!(col.spans_for_job("job-B").len(), 1);
        assert_eq!(col.spans_for_job("job-C").len(), 0);
    }
}
