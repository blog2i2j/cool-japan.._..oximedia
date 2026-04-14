//! Chrome Trace Event Format JSON export.
//!
//! Produces JSON loadable in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev/).
//! Supports the following event phases:
//!
//! | Phase | Code | Description                          |
//! |-------|------|--------------------------------------|
//! | B/E   | `B`/`E` | Duration begin / end (paired)    |
//! | X     | `X`  | Complete event (begin + dur)         |
//! | I     | `i`  | Instant / point-in-time event        |
//!
//! Each event carries `pid`, `tid`, `ts` (microseconds), `dur` (for X),
//! `name`, `cat`, and optional `args` key-value map.
//!
//! This module extends `crate::chrome_trace` with a higher-level API that
//! integrates directly with `crate::span::SpanTracker` and provides
//! instant-event support plus convenience constructors.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::chrome_tracing::{TraceEvent, EventPhase, export_json};
//!
//! let events = vec![
//!     TraceEvent::begin("render", "video", 0.0, 1, 1),
//!     TraceEvent::end("render", "video", 500.0, 1, 1),
//!     TraceEvent::instant("frame_drop", "video", 250.0, 1, 1),
//! ];
//! let json = export_json(&events).expect("serialisation");
//! assert!(json.contains("traceEvents"));
//! ```

#![allow(dead_code)]

use crate::span::SpanTracker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EventPhase
// ---------------------------------------------------------------------------

/// Chrome Trace Event phase identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventPhase {
    /// Duration begin (`B`).
    Begin,
    /// Duration end (`E`).
    End,
    /// Complete event (`X`).
    Complete,
    /// Instant event (`i`).
    Instant,
}

impl EventPhase {
    /// Returns the single-character identifier used in Chrome Tracing JSON.
    #[must_use]
    pub fn as_char(&self) -> &'static str {
        match self {
            Self::Begin => "B",
            Self::End => "E",
            Self::Complete => "X",
            Self::Instant => "i",
        }
    }

    /// Parses a phase character string back into an `EventPhase`.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "B" => Some(Self::Begin),
            "E" => Some(Self::End),
            "X" => Some(Self::Complete),
            "i" => Some(Self::Instant),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TraceEvent
// ---------------------------------------------------------------------------

/// A Chrome Trace Event with all required and optional fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Event name.
    pub name: String,
    /// Comma-separated categories.
    pub cat: String,
    /// Phase character.
    pub ph: String,
    /// Process ID.
    pub pid: u64,
    /// Thread ID.
    pub tid: u64,
    /// Timestamp in microseconds from trace start.
    pub ts: f64,
    /// Duration in microseconds (only meaningful for `X` phase).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dur: Option<f64>,
    /// Optional arguments / metadata.
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub args: HashMap<String, String>,
}

impl TraceEvent {
    /// Creates a Begin (`B`) event.
    #[must_use]
    pub fn begin(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            name: name.into(),
            cat: cat.into(),
            ph: EventPhase::Begin.as_char().to_owned(),
            pid,
            tid,
            ts: ts_us,
            dur: None,
            args: HashMap::new(),
        }
    }

    /// Creates an End (`E`) event.
    #[must_use]
    pub fn end(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            name: name.into(),
            cat: cat.into(),
            ph: EventPhase::End.as_char().to_owned(),
            pid,
            tid,
            ts: ts_us,
            dur: None,
            args: HashMap::new(),
        }
    }

    /// Creates a Complete (`X`) event.
    #[must_use]
    pub fn complete(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        dur_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            name: name.into(),
            cat: cat.into(),
            ph: EventPhase::Complete.as_char().to_owned(),
            pid,
            tid,
            ts: ts_us,
            dur: Some(dur_us),
            args: HashMap::new(),
        }
    }

    /// Creates an Instant (`i`) event.
    #[must_use]
    pub fn instant(
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) -> Self {
        Self {
            name: name.into(),
            cat: cat.into(),
            ph: EventPhase::Instant.as_char().to_owned(),
            pid,
            tid,
            ts: ts_us,
            dur: None,
            args: HashMap::new(),
        }
    }

    /// Builder-style method to add an argument.
    #[must_use]
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.args.insert(key.into(), value.into());
        self
    }

    /// Returns the `EventPhase` parsed from the `ph` field.
    #[must_use]
    pub fn phase(&self) -> Option<EventPhase> {
        EventPhase::from_str(&self.ph)
    }

    /// Returns `true` if this is a complete (`X`) event.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.ph == "X"
    }

    /// Returns `true` if this is an instant (`i`) event.
    #[must_use]
    pub fn is_instant(&self) -> bool {
        self.ph == "i"
    }
}

// ---------------------------------------------------------------------------
// ChromeTracingExporter
// ---------------------------------------------------------------------------

/// High-level exporter that collects `TraceEvent`s and produces JSON.
#[derive(Debug, Clone, Default)]
pub struct ChromeTracingExporter {
    events: Vec<TraceEvent>,
}

impl ChromeTracingExporter {
    /// Creates an empty exporter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a pre-built event.
    pub fn push(&mut self, event: TraceEvent) {
        self.events.push(event);
    }

    /// Adds a Begin event.
    pub fn add_begin(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(TraceEvent::begin(name, cat, ts_us, pid, tid));
    }

    /// Adds an End event.
    pub fn add_end(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(TraceEvent::end(name, cat, ts_us, pid, tid));
    }

    /// Adds a Complete event.
    pub fn add_complete(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        dur_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(TraceEvent::complete(name, cat, ts_us, dur_us, pid, tid));
    }

    /// Adds an Instant event.
    pub fn add_instant(
        &mut self,
        name: impl Into<String>,
        cat: impl Into<String>,
        ts_us: f64,
        pid: u64,
        tid: u64,
    ) {
        self.events
            .push(TraceEvent::instant(name, cat, ts_us, pid, tid));
    }

    /// Number of stored events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns the stored events.
    #[must_use]
    pub fn events(&self) -> &[TraceEvent] {
        &self.events
    }

    /// Clears all stored events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Exports all closed spans from a `SpanTracker` as Complete (`X`) events.
    ///
    /// Open spans are silently skipped.
    pub fn export_from_tracker(&mut self, tracker: &SpanTracker, pid: u64, tid: u64) {
        let mut spans = tracker.all_spans();
        spans.sort_by_key(|s| s.start_ns);

        for span in spans {
            if let Some(end_ns) = span.end_ns {
                let ts_us = span.start_ns as f64 / 1_000.0;
                let dur_us = end_ns.saturating_sub(span.start_ns) as f64 / 1_000.0;
                let mut event = TraceEvent::complete(&span.name, "span", ts_us, dur_us, pid, tid);
                event
                    .args
                    .insert("span_id".to_owned(), span.id.value().to_string());
                if let Some(parent) = span.parent_id {
                    event
                        .args
                        .insert("parent_id".to_owned(), parent.value().to_string());
                }
                self.events.push(event);
            }
        }
    }

    /// Serialises all events to Chrome Tracing JSON.
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        export_json(&self.events)
    }
}

// ---------------------------------------------------------------------------
// Free function: export_json
// ---------------------------------------------------------------------------

/// Serialises a slice of `TraceEvent`s to Chrome Tracing JSON format.
///
/// The output is `{"traceEvents": [...]}`.
pub fn export_json(events: &[TraceEvent]) -> Result<String, serde_json::Error> {
    #[derive(Serialize)]
    struct Wrapper<'a> {
        #[serde(rename = "traceEvents")]
        trace_events: &'a [TraceEvent],
    }
    serde_json::to_string_pretty(&Wrapper {
        trace_events: events,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::span::SpanTracker;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_empty_export_is_valid_json() {
        let json = export_json(&[]).expect("must succeed");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid json");
        assert!(v["traceEvents"].is_array());
        assert_eq!(v["traceEvents"].as_array().expect("array").len(), 0);
    }

    #[test]
    fn test_begin_end_phases() {
        let events = vec![
            TraceEvent::begin("f", "c", 0.0, 1, 1),
            TraceEvent::end("f", "c", 100.0, 1, 1),
        ];
        let json = export_json(&events).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        assert_eq!(v["traceEvents"][0]["ph"], "B");
        assert_eq!(v["traceEvents"][1]["ph"], "E");
    }

    #[test]
    fn test_complete_event_has_dur() {
        let events = vec![TraceEvent::complete("dec", "codec", 10.0, 50.0, 2, 3)];
        let json = export_json(&events).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        let ev = &v["traceEvents"][0];
        assert_eq!(ev["ph"], "X");
        assert!((ev["dur"].as_f64().expect("dur") - 50.0).abs() < f64::EPSILON);
        assert_eq!(ev["pid"], 2);
        assert_eq!(ev["tid"], 3);
    }

    #[test]
    fn test_instant_event() {
        let events = vec![TraceEvent::instant("marker", "ui", 42.0, 1, 1)];
        let json = export_json(&events).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        assert_eq!(v["traceEvents"][0]["ph"], "i");
        assert_eq!(v["traceEvents"][0]["name"], "marker");
    }

    #[test]
    fn test_event_with_args() {
        let ev = TraceEvent::begin("fn", "cat", 0.0, 1, 1)
            .with_arg("frame", "42")
            .with_arg("quality", "high");
        let json = export_json(&[ev]).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        assert_eq!(v["traceEvents"][0]["args"]["frame"], "42");
        assert_eq!(v["traceEvents"][0]["args"]["quality"], "high");
    }

    #[test]
    fn test_event_phase_roundtrip() {
        assert_eq!(EventPhase::from_str("B"), Some(EventPhase::Begin));
        assert_eq!(EventPhase::from_str("E"), Some(EventPhase::End));
        assert_eq!(EventPhase::from_str("X"), Some(EventPhase::Complete));
        assert_eq!(EventPhase::from_str("i"), Some(EventPhase::Instant));
        assert_eq!(EventPhase::from_str("Z"), None);
    }

    #[test]
    fn test_trace_event_helpers() {
        let b = TraceEvent::begin("f", "c", 0.0, 1, 1);
        assert!(!b.is_complete());
        assert!(!b.is_instant());
        assert_eq!(b.phase(), Some(EventPhase::Begin));

        let x = TraceEvent::complete("f", "c", 0.0, 10.0, 1, 1);
        assert!(x.is_complete());
        assert!(!x.is_instant());

        let i = TraceEvent::instant("f", "c", 0.0, 1, 1);
        assert!(i.is_instant());
    }

    #[test]
    fn test_exporter_from_tracker() {
        let tracker = SpanTracker::new();
        {
            let _g = tracker.enter("encode");
            thread::sleep(Duration::from_millis(2));
        }
        let mut exporter = ChromeTracingExporter::new();
        exporter.export_from_tracker(&tracker, 1, 1);
        assert_eq!(exporter.event_count(), 1);
        let json = exporter.export_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        assert_eq!(v["traceEvents"][0]["ph"], "X");
        assert_eq!(v["traceEvents"][0]["name"], "encode");
        assert!(v["traceEvents"][0]["args"]["span_id"].is_string());
    }

    #[test]
    fn test_exporter_add_instant() {
        let mut e = ChromeTracingExporter::new();
        e.add_instant("gc_pause", "runtime", 100.0, 1, 1);
        assert_eq!(e.event_count(), 1);
        assert!(e.events()[0].is_instant());
    }

    #[test]
    fn test_exporter_clear() {
        let mut e = ChromeTracingExporter::new();
        e.add_begin("a", "c", 0.0, 1, 1);
        e.add_end("a", "c", 1.0, 1, 1);
        assert_eq!(e.event_count(), 2);
        e.clear();
        assert_eq!(e.event_count(), 0);
    }

    #[test]
    fn test_name_cat_pid_tid_preserved() {
        let events = vec![TraceEvent::begin("my_fn", "render", 0.0, 42, 99)];
        let json = export_json(&events).expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        let ev = &v["traceEvents"][0];
        assert_eq!(ev["name"], "my_fn");
        assert_eq!(ev["cat"], "render");
        assert_eq!(ev["pid"], 42);
        assert_eq!(ev["tid"], 99);
    }

    #[test]
    fn test_nested_tracker_export() {
        let tracker = SpanTracker::new();
        {
            let _outer = tracker.enter("outer");
            {
                let _inner = tracker.enter("inner");
                thread::sleep(Duration::from_millis(1));
            }
        }
        let mut exporter = ChromeTracingExporter::new();
        exporter.export_from_tracker(&tracker, 1, 1);
        assert_eq!(exporter.event_count(), 2);
        let json = exporter.export_json().expect("ok");
        let v: serde_json::Value = serde_json::from_str(&json).expect("valid");
        let names: Vec<&str> = v["traceEvents"]
            .as_array()
            .expect("array")
            .iter()
            .map(|e| e["name"].as_str().expect("name"))
            .collect();
        assert!(names.contains(&"outer"));
        assert!(names.contains(&"inner"));
    }
}
