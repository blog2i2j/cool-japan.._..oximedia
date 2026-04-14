//! Serializable profiling session snapshot.
//!
//! A `ProfilingSession` captures a point-in-time snapshot of all key profiling
//! metrics: custom metrics, event counts, hotspot summaries, and general
//! session metadata.  Snapshots are fully serializable to/from JSON, enabling
//! persistence, diffing between runs, and CI regression tracking.
//!
//! # Snapshot lifecycle
//! ```
//! use oximedia_profiler::session::{ProfilingSession, SessionMetric};
//!
//! let mut session = ProfilingSession::new("encode-benchmark");
//! session.record_metric("frames", SessionMetric::Count(1000));
//! session.record_metric("fps", SessionMetric::Rate(59.94));
//! session.finalise();
//!
//! let json = session.to_json().unwrap();
//! let restored = ProfilingSession::from_json(&json).unwrap();
//! assert_eq!(restored.name(), "encode-benchmark");
//! ```

use crate::{ProfilerError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// SessionMetric — individual metric value
// ---------------------------------------------------------------------------

/// A single metric value recorded in a profiling session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum SessionMetric {
    /// Integer counter (e.g. frames encoded, bytes processed).
    Count(u64),
    /// Floating-point rate (e.g. fps, MB/s).
    Rate(f64),
    /// Duration (nanosecond precision serialised as µs for readability).
    #[serde(with = "duration_micros")]
    Duration(Duration),
    /// Percentage (0.0–100.0).
    Percentage(f64),
    /// Free-form string annotation.
    Text(String),
}

mod duration_micros {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S: Serializer>(dur: &Duration, s: S) -> std::result::Result<S::Ok, S::Error> {
        dur.as_micros().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<Duration, D::Error> {
        let micros = u128::deserialize(d)?;
        Ok(Duration::from_micros(micros as u64))
    }
}

impl std::fmt::Display for SessionMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n}"),
            Self::Rate(r) => write!(f, "{r:.3}"),
            Self::Duration(d) => write!(f, "{:.3}ms", d.as_secs_f64() * 1_000.0),
            Self::Percentage(p) => write!(f, "{p:.2}%"),
            Self::Text(t) => write!(f, "{t}"),
        }
    }
}

// ---------------------------------------------------------------------------
// HotspotRecord — lightweight hotspot summary for a snapshot
// ---------------------------------------------------------------------------

/// Lightweight hotspot entry stored in a session snapshot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HotspotRecord {
    /// Function or stage name.
    pub name: String,
    /// Percentage of total profiling time (0.0–100.0).
    pub time_pct: f64,
    /// Hit count (samples or call count).
    pub hits: u64,
}

impl HotspotRecord {
    /// Create a new hotspot record.
    pub fn new(name: impl Into<String>, time_pct: f64, hits: u64) -> Self {
        Self {
            name: name.into(),
            time_pct,
            hits,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a profiling session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SessionStatus {
    /// Session has been created but not started.
    New,
    /// Session is actively collecting data.
    Running,
    /// Session has been finalised (snapshot is complete).
    Finalised,
}

// ---------------------------------------------------------------------------
// ProfilingSession — the snapshot
// ---------------------------------------------------------------------------

/// A complete, serialisable snapshot of a profiling session.
///
/// Records metrics, hotspots, event counts, and session metadata. Supports
/// JSON round-trip via [`to_json`](Self::to_json) / [`from_json`](Self::from_json)
/// and file I/O via [`save`](Self::save) / [`load`](Self::load).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    /// Session identifier / label.
    name: String,
    /// Wall-clock timestamp at creation (RFC 3339, filled lazily).
    created_at: String,
    /// Wall-clock timestamp when [`finalise`](Self::finalise) was called.
    finalised_at: Option<String>,
    /// Total duration of the profiling run (µs precision).
    #[serde(with = "duration_micros")]
    duration: Duration,
    /// Lifecycle status.
    status: SessionStatus,
    /// Named metrics collected during the session.
    metrics: HashMap<String, SessionMetric>,
    /// Top hotspots (time-ordered descending).
    hotspots: Vec<HotspotRecord>,
    /// Total number of profiling events recorded.
    event_count: u64,
    /// Arbitrary key-value annotations.
    annotations: HashMap<String, String>,
}

impl ProfilingSession {
    /// Create a new session with the given label.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            created_at: simple_timestamp(),
            finalised_at: None,
            duration: Duration::ZERO,
            status: SessionStatus::New,
            metrics: HashMap::new(),
            hotspots: Vec::new(),
            event_count: 0,
            annotations: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    /// Transition to `Running`.
    ///
    /// Returns an error if the session is not in `New` state.
    pub fn start(&mut self) -> Result<()> {
        if self.status != SessionStatus::New {
            return Err(ProfilerError::InvalidConfig(
                "session must be New to start".to_string(),
            ));
        }
        self.status = SessionStatus::Running;
        Ok(())
    }

    /// Transition to `Finalised`, recording `duration` and locking the snapshot.
    ///
    /// Calling `finalise` multiple times is idempotent.
    pub fn finalise_with_duration(&mut self, duration: Duration) {
        self.duration = duration;
        self.status = SessionStatus::Finalised;
        self.finalised_at = Some(simple_timestamp());
    }

    /// Finalise the session without setting a specific duration.
    pub fn finalise(&mut self) {
        self.finalise_with_duration(self.duration);
    }

    // -----------------------------------------------------------------------
    // Data recording
    // -----------------------------------------------------------------------

    /// Record or overwrite a named metric.
    pub fn record_metric(&mut self, name: impl Into<String>, metric: SessionMetric) {
        self.metrics.insert(name.into(), metric);
    }

    /// Increment the event counter by `n`.
    pub fn add_events(&mut self, n: u64) {
        self.event_count += n;
    }

    /// Add a hotspot record.  Hotspots are sorted by `time_pct` descending
    /// after each insertion.
    pub fn add_hotspot(&mut self, hotspot: HotspotRecord) {
        self.hotspots.push(hotspot);
        self.hotspots.sort_by(|a, b| {
            b.time_pct
                .partial_cmp(&a.time_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Attach a free-form annotation (key-value pair).
    pub fn annotate(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.annotations.insert(key.into(), value.into());
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Session name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Session status.
    pub fn status(&self) -> SessionStatus {
        self.status
    }

    /// Total profiling duration.
    pub fn duration(&self) -> Duration {
        self.duration
    }

    /// Get a named metric, if present.
    pub fn metric(&self, name: &str) -> Option<&SessionMetric> {
        self.metrics.get(name)
    }

    /// All metrics.
    pub fn metrics(&self) -> &HashMap<String, SessionMetric> {
        &self.metrics
    }

    /// Hotspot list (sorted by `time_pct` descending).
    pub fn hotspots(&self) -> &[HotspotRecord] {
        &self.hotspots
    }

    /// Total event count.
    pub fn event_count(&self) -> u64 {
        self.event_count
    }

    /// Retrieve an annotation.
    pub fn get_annotation(&self, key: &str) -> Option<&str> {
        self.annotations.get(key).map(String::as_str)
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Serialise the session to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(ProfilerError::Serialization)
    }

    /// Deserialise a session from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(ProfilerError::Serialization)
    }

    /// Write the session snapshot to a JSON file at `path`.
    ///
    /// Uses an atomic rename to avoid partially-written files.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = self.to_json()?;
        let mut tmp = path.to_path_buf();
        {
            let mut name = path
                .file_name()
                .ok_or_else(|| ProfilerError::Other("path has no filename".to_string()))?
                .to_os_string();
            name.push(".tmp");
            tmp.set_file_name(name);
        }
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a session snapshot from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
    }

    // -----------------------------------------------------------------------
    // Diff
    // -----------------------------------------------------------------------

    /// Compute the delta between `self` (baseline) and `other` (current) for
    /// all metrics that appear in both snapshots.
    ///
    /// Returns a list of `(name, baseline_display, current_display, delta_pct)` tuples.
    /// `delta_pct` is positive when `other` is higher, negative when lower.
    /// Non-numeric metrics are included with `delta_pct = 0.0`.
    pub fn diff(&self, other: &ProfilingSession) -> Vec<SessionDelta> {
        let mut deltas = Vec::new();
        for (name, baseline) in &self.metrics {
            if let Some(current) = other.metrics.get(name) {
                let delta_pct = numeric_delta_pct(baseline, current);
                deltas.push(SessionDelta {
                    name: name.clone(),
                    baseline: baseline.to_string(),
                    current: current.to_string(),
                    delta_pct,
                });
            }
        }
        // Sort by absolute delta descending so the most-changed metrics come first.
        deltas.sort_by(|a, b| {
            b.delta_pct
                .abs()
                .partial_cmp(&a.delta_pct.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        deltas
    }
}

/// Compute the percentage change from `baseline` to `current` for numeric
/// metric variants.  Returns `0.0` for text metrics.
fn numeric_delta_pct(baseline: &SessionMetric, current: &SessionMetric) -> f64 {
    let (b, c) = match (baseline, current) {
        (SessionMetric::Count(b), SessionMetric::Count(c)) => {
            (*b as f64, *c as f64)
        }
        (SessionMetric::Rate(b), SessionMetric::Rate(c)) => (*b, *c),
        (SessionMetric::Duration(b), SessionMetric::Duration(c)) => {
            (b.as_secs_f64(), c.as_secs_f64())
        }
        (SessionMetric::Percentage(b), SessionMetric::Percentage(c)) => (*b, *c),
        _ => return 0.0,
    };
    if b.abs() < f64::EPSILON {
        if c.abs() < f64::EPSILON {
            0.0
        } else {
            100.0
        }
    } else {
        (c - b) / b * 100.0
    }
}

/// A single metric delta between two sessions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionDelta {
    /// Metric name.
    pub name: String,
    /// Baseline metric formatted as a display string.
    pub baseline: String,
    /// Current metric formatted as a display string.
    pub current: String,
    /// Percentage change (positive = higher in `current`).
    pub delta_pct: f64,
}

impl SessionDelta {
    /// Returns `true` if the metric regressed (higher is worse for duration/rate,
    /// meaning the caller must decide; this method checks raw sign).
    pub fn is_positive(&self) -> bool {
        self.delta_pct > 0.0
    }

    /// Returns `true` if the absolute delta exceeds `threshold` percent.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.delta_pct.abs() >= threshold
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn simple_timestamp() -> String {
    // Minimal RFC-3339-like timestamp without external deps.
    // In production code one would pull in `time` or `chrono`.
    "1970-01-01T00:00:00Z".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(name: &str) -> ProfilingSession {
        let mut s = ProfilingSession::new(name);
        s.record_metric("frames", SessionMetric::Count(500));
        s.record_metric("fps", SessionMetric::Rate(29.97));
        s.record_metric(
            "encode_time",
            SessionMetric::Duration(Duration::from_millis(200)),
        );
        s.record_metric("cpu_usage", SessionMetric::Percentage(75.0));
        s.record_metric("label", SessionMetric::Text("test".to_string()));
        s.add_events(1024);
        s.add_hotspot(HotspotRecord::new("decode", 40.0, 800));
        s.add_hotspot(HotspotRecord::new("encode", 60.0, 1200));
        s.annotate("host", "ci-runner");
        s.finalise_with_duration(Duration::from_millis(500));
        s
    }

    #[test]
    fn test_new_session_status() {
        let s = ProfilingSession::new("test");
        assert_eq!(s.status(), SessionStatus::New);
        assert_eq!(s.name(), "test");
    }

    #[test]
    fn test_start_transitions_to_running() {
        let mut s = ProfilingSession::new("test");
        s.start().expect("should succeed in test");
        assert_eq!(s.status(), SessionStatus::Running);
    }

    #[test]
    fn test_start_twice_is_error() {
        let mut s = ProfilingSession::new("test");
        s.start().expect("should succeed in test");
        assert!(s.start().is_err());
    }

    #[test]
    fn test_finalise_sets_status() {
        let mut s = ProfilingSession::new("test");
        s.finalise_with_duration(Duration::from_secs(1));
        assert_eq!(s.status(), SessionStatus::Finalised);
        assert_eq!(s.duration(), Duration::from_secs(1));
    }

    #[test]
    fn test_record_metric_and_retrieve() {
        let mut s = ProfilingSession::new("m");
        s.record_metric("x", SessionMetric::Count(42));
        let m = s.metric("x").expect("should succeed in test");
        assert_eq!(*m, SessionMetric::Count(42));
        assert!(s.metric("missing").is_none());
    }

    #[test]
    fn test_hotspots_sorted_descending() {
        let mut s = ProfilingSession::new("h");
        s.add_hotspot(HotspotRecord::new("a", 30.0, 10));
        s.add_hotspot(HotspotRecord::new("b", 60.0, 20));
        s.add_hotspot(HotspotRecord::new("c", 10.0, 5));
        let hs = s.hotspots();
        assert_eq!(hs[0].name, "b");
        assert_eq!(hs[1].name, "a");
        assert_eq!(hs[2].name, "c");
    }

    #[test]
    fn test_event_count_accumulation() {
        let mut s = ProfilingSession::new("ev");
        s.add_events(100);
        s.add_events(50);
        assert_eq!(s.event_count(), 150);
    }

    #[test]
    fn test_annotation() {
        let mut s = ProfilingSession::new("ann");
        s.annotate("branch", "main");
        assert_eq!(s.get_annotation("branch"), Some("main"));
        assert!(s.get_annotation("missing").is_none());
    }

    #[test]
    fn test_json_round_trip() {
        let session = make_session("round-trip");
        let json = session.to_json().expect("should succeed in test");
        let restored = ProfilingSession::from_json(&json).expect("should succeed in test");
        assert_eq!(restored.name(), session.name());
        assert_eq!(restored.event_count(), session.event_count());
        assert_eq!(restored.hotspots().len(), session.hotspots().len());
        assert_eq!(restored.metrics().len(), session.metrics().len());
    }

    #[test]
    fn test_json_roundtrip_duration_metric() {
        let mut s = ProfilingSession::new("dur");
        s.record_metric(
            "t",
            SessionMetric::Duration(Duration::from_millis(42)),
        );
        let json = s.to_json().expect("should succeed in test");
        let back = ProfilingSession::from_json(&json).expect("should succeed in test");
        assert_eq!(back.metric("t"), Some(&SessionMetric::Duration(Duration::from_millis(42))));
    }

    #[test]
    fn test_save_and_load() {
        let session = make_session("save-load");
        let tmp = std::env::temp_dir().join("oximedia_profiler_session_test.json");
        session.save(&tmp).expect("should succeed in test");
        let loaded = ProfilingSession::load(&tmp).expect("should succeed in test");
        assert_eq!(loaded.name(), session.name());
        assert_eq!(loaded.event_count(), session.event_count());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_diff_basic() {
        let base = make_session("base");
        let mut current = make_session("current");
        // Double the frame count → +100%
        current.record_metric("frames", SessionMetric::Count(1000));
        // Halve fps → -50%
        current.record_metric("fps", SessionMetric::Rate(14.985));

        let deltas = base.diff(&current);
        let frames_delta = deltas
            .iter()
            .find(|d| d.name == "frames")
            .expect("should succeed in test");
        assert!((frames_delta.delta_pct - 100.0).abs() < 1.0);

        let fps_delta = deltas
            .iter()
            .find(|d| d.name == "fps")
            .expect("should succeed in test");
        assert!(fps_delta.delta_pct < 0.0);
    }

    #[test]
    fn test_diff_no_change() {
        let base = make_session("base");
        let same = make_session("same");
        let deltas = base.diff(&same);
        for d in &deltas {
            if d.name != "label" {
                assert!(d.delta_pct.abs() < 1e-6, "unexpected delta for {}", d.name);
            }
        }
    }

    #[test]
    fn test_session_delta_is_significant() {
        let d = SessionDelta {
            name: "fps".to_string(),
            baseline: "60".to_string(),
            current: "54".to_string(),
            delta_pct: -10.0,
        };
        assert!(d.is_significant(5.0));
        assert!(!d.is_significant(15.0));
    }

    #[test]
    fn test_session_metric_display() {
        assert_eq!(SessionMetric::Count(42).to_string(), "42");
        assert!(SessionMetric::Rate(59.94).to_string().starts_with("59.940"));
        assert!(SessionMetric::Percentage(75.5)
            .to_string()
            .contains("75.50"));
        assert_eq!(
            SessionMetric::Text("hi".to_string()).to_string(),
            "hi"
        );
    }
}
