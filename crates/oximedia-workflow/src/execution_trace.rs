#![allow(dead_code)]
//! Execution tracing and span tracking for workflow debugging.
//!
//! Records detailed timing and status information for every step in a
//! workflow execution, enabling post-mortem analysis, performance
//! profiling, and audit trails.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Unique identifier for a trace span.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpanId(String);

impl SpanId {
    /// Create a new span ID from a string.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The outcome of a traced span.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanStatus {
    /// Span completed successfully.
    Ok,
    /// Span completed with a warning.
    Warning(String),
    /// Span failed with an error.
    Error(String),
    /// Span was cancelled before completion.
    Cancelled,
    /// Span is still running.
    InProgress,
}

/// A key-value attribute attached to a span.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanAttribute {
    /// Attribute key.
    pub key: String,
    /// Attribute value.
    pub value: String,
}

impl SpanAttribute {
    /// Create a new attribute.
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// A single span in an execution trace.
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// Unique span identifier.
    pub id: SpanId,
    /// Optional parent span (for nesting).
    pub parent: Option<SpanId>,
    /// Human-readable name of the operation.
    pub name: String,
    /// When the span started.
    pub start_time: SystemTime,
    /// When the span ended (if finished).
    pub end_time: Option<SystemTime>,
    /// Duration (calculated when ended).
    pub duration: Option<Duration>,
    /// Current status of the span.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes.
    pub attributes: Vec<SpanAttribute>,
}

impl TraceSpan {
    /// Create a new span that starts now.
    pub fn start(id: SpanId, name: impl Into<String>) -> Self {
        Self {
            id,
            parent: None,
            name: name.into(),
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            status: SpanStatus::InProgress,
            attributes: Vec::new(),
        }
    }

    /// Create a new child span.
    pub fn start_child(id: SpanId, parent: &SpanId, name: impl Into<String>) -> Self {
        Self {
            id,
            parent: Some(parent.clone()),
            name: name.into(),
            start_time: SystemTime::now(),
            end_time: None,
            duration: None,
            status: SpanStatus::InProgress,
            attributes: Vec::new(),
        }
    }

    /// End the span with the given status.
    pub fn finish(&mut self, status: SpanStatus) {
        let now = SystemTime::now();
        self.end_time = Some(now);
        self.duration = now.duration_since(self.start_time).ok();
        self.status = status;
    }

    /// Add an attribute to the span.
    pub fn add_attribute(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.attributes.push(SpanAttribute::new(key, value));
    }

    /// Check if this span is still running.
    pub fn is_running(&self) -> bool {
        self.status == SpanStatus::InProgress
    }

    /// Check if this span has a parent.
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Get the attribute value for a given key.
    pub fn get_attribute(&self, key: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|a| a.key == key)
            .map(|a| a.value.as_str())
    }
}

/// An execution trace that collects spans for a single workflow run.
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Unique trace identifier (typically the workflow run ID).
    pub trace_id: String,
    /// All spans in this trace, indexed by span ID.
    spans: HashMap<String, TraceSpan>,
    /// Order in which spans were created.
    span_order: Vec<String>,
}

impl ExecutionTrace {
    /// Create a new execution trace.
    pub fn new(trace_id: impl Into<String>) -> Self {
        Self {
            trace_id: trace_id.into(),
            spans: HashMap::new(),
            span_order: Vec::new(),
        }
    }

    /// Start a new root span.
    ///
    /// # Panics
    ///
    /// This method will not panic; if the internal map entry is somehow
    /// missing after insertion the returned reference is obtained via
    /// `Entry::or_insert_with`, guaranteeing it exists.
    pub fn start_span(&mut self, id: SpanId, name: impl Into<String>) -> &mut TraceSpan {
        let span = TraceSpan::start(id.clone(), name);
        let key = id.0.clone();
        self.span_order.push(key.clone());
        self.spans.entry(key).or_insert(span)
    }

    /// Start a child span under a parent.
    ///
    /// # Panics
    ///
    /// Same guarantee as [`start_span`](Self::start_span).
    pub fn start_child_span(
        &mut self,
        id: SpanId,
        parent: &SpanId,
        name: impl Into<String>,
    ) -> &mut TraceSpan {
        let span = TraceSpan::start_child(id.clone(), parent, name);
        let key = id.0.clone();
        self.span_order.push(key.clone());
        self.spans.entry(key).or_insert(span)
    }

    /// Finish a span with the given status.
    pub fn finish_span(&mut self, id: &SpanId, status: SpanStatus) {
        if let Some(span) = self.spans.get_mut(&id.0) {
            span.finish(status);
        }
    }

    /// Get a span by ID.
    pub fn get_span(&self, id: &SpanId) -> Option<&TraceSpan> {
        self.spans.get(&id.0)
    }

    /// Get all spans in creation order.
    pub fn spans_ordered(&self) -> Vec<&TraceSpan> {
        self.span_order
            .iter()
            .filter_map(|k| self.spans.get(k))
            .collect()
    }

    /// Get only root spans (those without a parent).
    pub fn root_spans(&self) -> Vec<&TraceSpan> {
        self.spans_ordered()
            .into_iter()
            .filter(|s| s.is_root())
            .collect()
    }

    /// Get all children of a given span.
    pub fn children_of(&self, parent: &SpanId) -> Vec<&TraceSpan> {
        self.spans
            .values()
            .filter(|s| s.parent.as_ref() == Some(parent))
            .collect()
    }

    /// Return how many spans are in this trace.
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    /// Return how many spans are still running.
    pub fn running_count(&self) -> usize {
        self.spans.values().filter(|s| s.is_running()).count()
    }

    /// Return how many spans failed.
    pub fn error_count(&self) -> usize {
        self.spans
            .values()
            .filter(|s| matches!(s.status, SpanStatus::Error(_)))
            .count()
    }

    /// Compute total trace duration (from earliest start to latest end).
    pub fn total_duration(&self) -> Option<Duration> {
        let earliest = self.spans.values().map(|s| s.start_time).min()?;
        let latest = self.spans.values().filter_map(|s| s.end_time).max()?;
        latest.duration_since(earliest).ok()
    }

    /// Generate a simple summary of the trace.
    pub fn summary(&self) -> TraceSummary {
        TraceSummary {
            trace_id: self.trace_id.clone(),
            total_spans: self.span_count(),
            running: self.running_count(),
            errors: self.error_count(),
            total_duration: self.total_duration(),
        }
    }
}

/// A summary of an execution trace.
#[derive(Debug, Clone)]
pub struct TraceSummary {
    /// Trace identifier.
    pub trace_id: String,
    /// Total number of spans.
    pub total_spans: usize,
    /// Number of spans still running.
    pub running: usize,
    /// Number of spans that errored.
    pub errors: usize,
    /// Total wall-clock duration of the trace.
    pub total_duration: Option<Duration>,
}

impl TraceSummary {
    /// Check if all spans have completed (none running).
    pub fn is_complete(&self) -> bool {
        self.running == 0
    }

    /// Check if the trace had any errors.
    pub fn has_errors(&self) -> bool {
        self.errors > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_id_display() {
        let id = SpanId::new("span-1");
        assert_eq!(id.to_string(), "span-1");
        assert_eq!(id.as_str(), "span-1");
    }

    #[test]
    fn test_span_start_and_finish() {
        let mut span = TraceSpan::start(SpanId::new("s1"), "test operation");
        assert!(span.is_running());
        assert!(span.is_root());
        assert!(span.end_time.is_none());

        span.finish(SpanStatus::Ok);
        assert!(!span.is_running());
        assert!(span.end_time.is_some());
        assert!(span.duration.is_some());
    }

    #[test]
    fn test_child_span() {
        let parent_id = SpanId::new("parent");
        let span = TraceSpan::start_child(SpanId::new("child"), &parent_id, "child op");
        assert!(!span.is_root());
        assert_eq!(
            span.parent
                .as_ref()
                .expect("should succeed in test")
                .as_str(),
            "parent"
        );
    }

    #[test]
    fn test_span_attributes() {
        let mut span = TraceSpan::start(SpanId::new("s1"), "op");
        span.add_attribute("task_type", "transcode");
        span.add_attribute("codec", "h264");

        assert_eq!(span.get_attribute("task_type"), Some("transcode"));
        assert_eq!(span.get_attribute("codec"), Some("h264"));
        assert_eq!(span.get_attribute("nonexistent"), None);
    }

    #[test]
    fn test_execution_trace_basic() {
        let mut trace = ExecutionTrace::new("trace-1");
        trace.start_span(SpanId::new("s1"), "step 1");
        trace.start_span(SpanId::new("s2"), "step 2");

        assert_eq!(trace.span_count(), 2);
        assert_eq!(trace.running_count(), 2);
        assert_eq!(trace.error_count(), 0);
    }

    #[test]
    fn test_execution_trace_finish() {
        let mut trace = ExecutionTrace::new("trace-2");
        trace.start_span(SpanId::new("s1"), "step 1");
        trace.finish_span(&SpanId::new("s1"), SpanStatus::Ok);

        assert_eq!(trace.running_count(), 0);
        let span = trace
            .get_span(&SpanId::new("s1"))
            .expect("should succeed in test");
        assert_eq!(span.status, SpanStatus::Ok);
    }

    #[test]
    fn test_execution_trace_errors() {
        let mut trace = ExecutionTrace::new("trace-3");
        trace.start_span(SpanId::new("s1"), "good");
        trace.start_span(SpanId::new("s2"), "bad");
        trace.finish_span(&SpanId::new("s1"), SpanStatus::Ok);
        trace.finish_span(&SpanId::new("s2"), SpanStatus::Error("timeout".into()));

        assert_eq!(trace.error_count(), 1);
    }

    #[test]
    fn test_root_spans() {
        let mut trace = ExecutionTrace::new("trace-4");
        let root_id = SpanId::new("root");
        trace.start_span(root_id.clone(), "root op");
        trace.start_child_span(SpanId::new("child"), &root_id, "child op");

        let roots = trace.root_spans();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].name, "root op");
    }

    #[test]
    fn test_children_of() {
        let mut trace = ExecutionTrace::new("trace-5");
        let parent = SpanId::new("p1");
        trace.start_span(parent.clone(), "parent");
        trace.start_child_span(SpanId::new("c1"), &parent, "child 1");
        trace.start_child_span(SpanId::new("c2"), &parent, "child 2");

        let children = trace.children_of(&parent);
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_spans_ordered() {
        let mut trace = ExecutionTrace::new("trace-6");
        trace.start_span(SpanId::new("first"), "first");
        trace.start_span(SpanId::new("second"), "second");
        trace.start_span(SpanId::new("third"), "third");

        let ordered = trace.spans_ordered();
        assert_eq!(ordered[0].name, "first");
        assert_eq!(ordered[1].name, "second");
        assert_eq!(ordered[2].name, "third");
    }

    #[test]
    fn test_trace_summary() {
        let mut trace = ExecutionTrace::new("trace-7");
        trace.start_span(SpanId::new("s1"), "a");
        trace.start_span(SpanId::new("s2"), "b");
        trace.finish_span(&SpanId::new("s1"), SpanStatus::Ok);
        trace.finish_span(&SpanId::new("s2"), SpanStatus::Error("fail".into()));

        let summary = trace.summary();
        assert_eq!(summary.total_spans, 2);
        assert_eq!(summary.running, 0);
        assert_eq!(summary.errors, 1);
        assert!(summary.is_complete());
        assert!(summary.has_errors());
    }

    #[test]
    fn test_total_duration() {
        let mut trace = ExecutionTrace::new("trace-8");
        trace.start_span(SpanId::new("s1"), "op");
        // Small sleep so duration is measurable.
        std::thread::sleep(Duration::from_millis(5));
        trace.finish_span(&SpanId::new("s1"), SpanStatus::Ok);

        let dur = trace.total_duration();
        assert!(dur.is_some());
        assert!(dur.expect("should succeed in test") >= Duration::from_millis(1));
    }

    #[test]
    fn test_span_attribute_struct() {
        let attr = SpanAttribute::new("key", "value");
        assert_eq!(attr.key, "key");
        assert_eq!(attr.value, "value");
    }
}
