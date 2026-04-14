//! Pipeline execution profiler using millisecond-resolution timestamps.
//!
//! This module provides a complementary profiling API to [`crate::profiler`].
//! Where the existing profiler uses `std::time::Duration` and `Instant`, the
//! types here represent timing as plain `u64` milliseconds.  This makes them
//! straightforward to serialise, compare across runs, and reconstruct from
//! logged data.
//!
//! # Design
//!
//! * [`NodeTiming`] — timing for a single node within one pipeline execution.
//! * [`PipelineExecutionProfile`] — a complete run containing many
//!   [`NodeTiming`] records, plus run-level metadata.
//! * [`PipelineProfiler`] — a bounded ring-buffer of
//!   [`PipelineExecutionProfile`] values with statistical helpers.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::pipeline_profiler::{NodeTiming, PipelineExecutionProfile, PipelineProfiler};
//!
//! let mut run = PipelineExecutionProfile::new("run-1", 1000);
//! let t = NodeTiming::new("scale", 1000).finish(1025, 4_096_000);
//! run.add_node_timing(t);
//! run.finish(1025);
//!
//! let mut profiler = PipelineProfiler::new(16);
//! profiler.record(run);
//!
//! assert_eq!(profiler.profile_count(), 1);
//! let latest = profiler.latest().expect("has latest");
//! assert_eq!(latest.total_duration_ms(), 25);
//! ```

// ── NodeTiming ────────────────────────────────────────────────────────────────

/// Timing data for a single pipeline node within one execution run.
#[derive(Debug, Clone)]
pub struct NodeTiming {
    /// Identifier of the node (matches the node's string ID in the graph).
    pub node_id: String,
    /// Wall-clock start time of this node's execution (milliseconds).
    pub start_ms: u64,
    /// Wall-clock end time of this node's execution (milliseconds).
    pub end_ms: u64,
    /// Number of input bytes consumed by this node during execution.
    pub input_bytes: u64,
    /// Number of output bytes produced by this node during execution.
    pub output_bytes: u64,
}

impl NodeTiming {
    /// Create a new `NodeTiming` for `node_id`, started at `start_ms`.
    ///
    /// `end_ms` and byte counts default to zero until [`finish`](Self::finish)
    /// is called.
    pub fn new(node_id: impl Into<String>, start_ms: u64) -> Self {
        Self {
            node_id: node_id.into(),
            start_ms,
            end_ms: 0,
            input_bytes: 0,
            output_bytes: 0,
        }
    }

    /// Finish the timing record with the given end time and output byte count.
    ///
    /// Returns `self` to allow a builder-style call chain.
    pub fn finish(mut self, end_ms: u64, output_bytes: u64) -> Self {
        self.end_ms = end_ms;
        self.output_bytes = output_bytes;
        self
    }

    /// Wall-clock duration of this node's execution in milliseconds.
    ///
    /// Returns `0` if `end_ms < start_ms` (which should never happen in
    /// correct usage).
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Throughput in megabytes per second for this node.
    ///
    /// Computed as `output_bytes / duration_ms * 1000 / 1024 / 1024`.
    /// Returns `0.0` when duration is zero.
    pub fn throughput_mbps(&self) -> f32 {
        let dur = self.duration_ms();
        if dur == 0 {
            return 0.0;
        }
        // output_bytes per ms → per second → convert to MiB/s.
        (self.output_bytes as f32 / dur as f32) * 1000.0 / (1024.0 * 1024.0)
    }
}

// ── PipelineExecutionProfile ──────────────────────────────────────────────────

/// A complete profile for one pipeline execution run.
///
/// Collects per-node [`NodeTiming`] records and provides analysis helpers
/// including slowest-node detection and a simplified critical-path estimate.
#[derive(Debug, Clone)]
pub struct PipelineExecutionProfile {
    /// Unique identifier for this run (e.g. a UUID or counter).
    pub run_id: String,
    /// Millisecond timestamp when the run started.
    pub start_ms: u64,
    /// Millisecond timestamp when the run finished (0 until [`finish`](Self::finish) is called).
    pub end_ms: u64,
    /// Per-node timing records collected during this run.
    pub node_timings: Vec<NodeTiming>,
}

impl PipelineExecutionProfile {
    /// Create a new profile for a run starting at `start_ms`.
    pub fn new(run_id: impl Into<String>, start_ms: u64) -> Self {
        Self {
            run_id: run_id.into(),
            start_ms,
            end_ms: 0,
            node_timings: Vec::new(),
        }
    }

    /// Append a [`NodeTiming`] record to this profile.
    pub fn add_node_timing(&mut self, timing: NodeTiming) {
        self.node_timings.push(timing);
    }

    /// Mark the run as finished at `end_ms`.
    pub fn finish(&mut self, end_ms: u64) {
        self.end_ms = end_ms;
    }

    /// Total wall-clock duration of the whole run in milliseconds.
    pub fn total_duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Return a reference to the [`NodeTiming`] with the longest duration.
    ///
    /// Returns `None` if no node timings have been recorded.
    pub fn slowest_node(&self) -> Option<&NodeTiming> {
        self.node_timings
            .iter()
            .max_by_key(|t| t.duration_ms())
    }

    /// Return a reference to the [`NodeTiming`] for `node_id`, if present.
    pub fn node_timing(&self, node_id: &str) -> Option<&NodeTiming> {
        self.node_timings.iter().find(|t| t.node_id == node_id)
    }

    /// Simplified critical-path estimate: node timings sorted by duration
    /// descending (longest first).
    ///
    /// In a real implementation this would traverse the DAG and accumulate
    /// path weights; here we use descending duration order as a conservative
    /// approximation suitable for single-threaded pipelines.
    pub fn critical_path(&self) -> Vec<&NodeTiming> {
        let mut sorted: Vec<&NodeTiming> = self.node_timings.iter().collect();
        sorted.sort_by(|a, b| b.duration_ms().cmp(&a.duration_ms()));
        sorted
    }

    /// Serialise this profile to a hand-written JSON string.
    ///
    /// The output is a JSON object with keys `run_id`, `start_ms`, `end_ms`,
    /// and `nodes` (an array of node objects).  No external crate is required.
    pub fn to_json(&self) -> String {
        let mut buf = String::with_capacity(128 + self.node_timings.len() * 80);
        buf.push('{');
        buf.push_str(&format!("\"run_id\":\"{}\",", json_escape(&self.run_id)));
        buf.push_str(&format!("\"start_ms\":{},", self.start_ms));
        buf.push_str(&format!("\"end_ms\":{},", self.end_ms));
        buf.push_str("\"nodes\":[");
        for (i, t) in self.node_timings.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push('{');
            buf.push_str(&format!("\"node_id\":\"{}\",", json_escape(&t.node_id)));
            buf.push_str(&format!("\"start_ms\":{},", t.start_ms));
            buf.push_str(&format!("\"end_ms\":{},", t.end_ms));
            buf.push_str(&format!("\"input_bytes\":{},", t.input_bytes));
            buf.push_str(&format!("\"output_bytes\":{}", t.output_bytes));
            buf.push('}');
        }
        buf.push_str("]}");
        buf
    }
}

/// Escape a string for safe embedding in a JSON string value.
fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ── PipelineProfiler ──────────────────────────────────────────────────────────

/// A bounded store of [`PipelineExecutionProfile`] records.
///
/// When the number of stored profiles reaches `max_profiles`, the oldest
/// profile is evicted before the new one is inserted (FIFO eviction).
pub struct PipelineProfiler {
    profiles: Vec<PipelineExecutionProfile>,
    max_profiles: usize,
}

impl PipelineProfiler {
    /// Create a new profiler that retains at most `max_profiles` profiles.
    ///
    /// `max_profiles` must be at least 1; if 0 is supplied it is silently
    /// treated as 1.
    pub fn new(max_profiles: usize) -> Self {
        let cap = max_profiles.max(1);
        Self {
            profiles: Vec::with_capacity(cap),
            max_profiles: cap,
        }
    }

    /// Record a new [`PipelineExecutionProfile`].
    ///
    /// Evicts the oldest profile when the store is at capacity.
    pub fn record(&mut self, profile: PipelineExecutionProfile) {
        if self.profiles.len() >= self.max_profiles {
            self.profiles.remove(0);
        }
        self.profiles.push(profile);
    }

    /// Return a reference to the most recently recorded profile.
    pub fn latest(&self) -> Option<&PipelineExecutionProfile> {
        self.profiles.last()
    }

    /// Arithmetic mean of `total_duration_ms` across all recorded profiles.
    ///
    /// Returns `None` if no profiles have been recorded.
    pub fn mean_duration_ms(&self) -> Option<f64> {
        if self.profiles.is_empty() {
            return None;
        }
        let sum: u64 = self.profiles.iter().map(|p| p.total_duration_ms()).sum();
        Some(sum as f64 / self.profiles.len() as f64)
    }

    /// 95th-percentile of `total_duration_ms` across all recorded profiles.
    ///
    /// Returns `None` if no profiles have been recorded.
    pub fn p95_duration_ms(&self) -> Option<f64> {
        if self.profiles.is_empty() {
            return None;
        }
        let mut durations: Vec<u64> =
            self.profiles.iter().map(|p| p.total_duration_ms()).collect();
        durations.sort_unstable();
        let idx = ((95 * (durations.len().saturating_sub(1))) + 50) / 100;
        let clamped = idx.min(durations.len().saturating_sub(1));
        Some(durations[clamped] as f64)
    }

    /// Number of profiles currently stored.
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(run_id: &str, start_ms: u64, end_ms: u64) -> PipelineExecutionProfile {
        let mut p = PipelineExecutionProfile::new(run_id, start_ms);
        p.finish(end_ms);
        p
    }

    #[test]
    fn node_timing_duration_ms() {
        let t = NodeTiming::new("scale", 1000).finish(1025, 0);
        assert_eq!(t.duration_ms(), 25);
    }

    #[test]
    fn node_timing_duration_ms_zero_when_end_before_start() {
        let t = NodeTiming::new("node", 500).finish(400, 0);
        assert_eq!(t.duration_ms(), 0, "saturating_sub should clamp to 0");
    }

    #[test]
    fn node_timing_throughput_mbps() {
        // 1 048 576 bytes (1 MiB) processed in 1000 ms → 1.0 MiB/s = 1.0 Mbps
        let t = NodeTiming::new("enc", 0).finish(1000, 1_048_576);
        let mbps = t.throughput_mbps();
        assert!(
            (mbps - 1.0).abs() < 0.001,
            "expected ~1.0 MiB/s, got {mbps}"
        );
    }

    #[test]
    fn node_timing_throughput_zero_duration() {
        let t = NodeTiming::new("enc", 0).finish(0, 1_000_000);
        assert_eq!(t.throughput_mbps(), 0.0);
    }

    #[test]
    fn profile_slowest_node() {
        let mut profile = PipelineExecutionProfile::new("r1", 0);
        profile.add_node_timing(NodeTiming::new("fast", 0).finish(5, 0));
        profile.add_node_timing(NodeTiming::new("slow", 0).finish(50, 0));
        profile.add_node_timing(NodeTiming::new("medium", 0).finish(20, 0));
        profile.finish(50);

        let slowest = profile.slowest_node().expect("should have slowest");
        assert_eq!(slowest.node_id, "slow");
    }

    #[test]
    fn profile_critical_path_sorted_desc() {
        let mut profile = PipelineExecutionProfile::new("r1", 0);
        profile.add_node_timing(NodeTiming::new("a", 0).finish(10, 0));
        profile.add_node_timing(NodeTiming::new("b", 0).finish(40, 0));
        profile.add_node_timing(NodeTiming::new("c", 0).finish(25, 0));
        profile.finish(40);

        let path = profile.critical_path();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].node_id, "b", "longest first");
        assert_eq!(path[1].node_id, "c");
        assert_eq!(path[2].node_id, "a");
    }

    #[test]
    fn profile_to_json_contains_run_id() {
        let mut profile = PipelineExecutionProfile::new("my-run-42", 100);
        profile.add_node_timing(NodeTiming::new("decoder", 100).finish(120, 2048));
        profile.finish(150);

        let json = profile.to_json();
        assert!(json.contains("my-run-42"), "JSON should contain run_id");
        assert!(json.contains("decoder"), "JSON should contain node id");
        assert!(json.contains("\"start_ms\":100"), "JSON should have start_ms");
    }

    #[test]
    fn profiler_record_and_latest() {
        let mut profiler = PipelineProfiler::new(10);
        profiler.record(make_profile("run-1", 0, 100));
        profiler.record(make_profile("run-2", 100, 250));

        assert_eq!(profiler.profile_count(), 2);
        let latest = profiler.latest().expect("should have latest");
        assert_eq!(latest.run_id, "run-2");
    }

    #[test]
    fn profiler_eviction_at_max() {
        let mut profiler = PipelineProfiler::new(3);
        profiler.record(make_profile("run-1", 0, 10));
        profiler.record(make_profile("run-2", 10, 20));
        profiler.record(make_profile("run-3", 20, 30));
        // At capacity — next insert should evict run-1.
        profiler.record(make_profile("run-4", 30, 40));

        assert_eq!(profiler.profile_count(), 3);
        // Oldest (run-1) should be gone.
        let ids: Vec<&str> = profiler
            .profiles
            .iter()
            .map(|p| p.run_id.as_str())
            .collect();
        assert!(!ids.contains(&"run-1"), "run-1 should have been evicted");
        assert_eq!(profiler.latest().map(|p| p.run_id.as_str()), Some("run-4"));
    }

    #[test]
    fn profiler_mean_duration_ms() {
        let mut profiler = PipelineProfiler::new(10);
        // Durations: 10, 20, 30 → mean = 20
        profiler.record(make_profile("r1", 0, 10));
        profiler.record(make_profile("r2", 0, 20));
        profiler.record(make_profile("r3", 0, 30));

        let mean = profiler.mean_duration_ms().expect("should have mean");
        assert!(
            (mean - 20.0).abs() < f64::EPSILON,
            "mean should be 20.0, got {mean}"
        );
    }

    #[test]
    fn profiler_mean_returns_none_when_empty() {
        let profiler = PipelineProfiler::new(10);
        assert!(profiler.mean_duration_ms().is_none());
        assert!(profiler.p95_duration_ms().is_none());
    }
}
