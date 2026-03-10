//! Graph execution profiling.
//!
//! Provides per-node timing statistics and port throughput metrics.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// NodeProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Per-node profiling statistics.
#[derive(Debug, Clone)]
pub struct NodeProfile {
    /// Unique node identifier.
    pub node_id: String,
    /// Average execution duration in microseconds.
    pub avg_duration_us: u64,
    /// Maximum observed execution duration in microseconds.
    pub max_duration_us: u64,
    /// Total number of times the node was executed.
    pub call_count: u64,
    /// Cumulative execution time in microseconds.
    pub total_us: u64,
}

impl NodeProfile {
    /// Create a new empty profile for a node.
    #[must_use]
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            avg_duration_us: 0,
            max_duration_us: 0,
            call_count: 0,
            total_us: 0,
        }
    }

    /// Simplified standard deviation: `(max - avg) / 2`.
    #[must_use]
    pub fn std_dev_us(&self) -> u64 {
        self.max_duration_us.saturating_sub(self.avg_duration_us) / 2
    }

    /// Record a new sample.
    fn record(&mut self, duration_us: u64) {
        self.total_us += duration_us;
        self.call_count += 1;
        self.avg_duration_us = self.total_us / self.call_count;
        if duration_us > self.max_duration_us {
            self.max_duration_us = duration_us;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphProfiler
// ─────────────────────────────────────────────────────────────────────────────

/// Collects per-node timing samples during graph execution.
#[derive(Debug, Default)]
pub struct GraphProfiler {
    profiles: HashMap<String, NodeProfile>,
}

impl GraphProfiler {
    /// Create a new, empty profiler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
        }
    }

    /// Record an execution duration for `node_id`.
    pub fn record(&mut self, node_id: &str, duration_us: u64) {
        let profile = self
            .profiles
            .entry(node_id.to_string())
            .or_insert_with(|| NodeProfile::new(node_id));
        profile.record(duration_us);
    }

    /// Retrieve the profile for a specific node.
    #[must_use]
    pub fn profile(&self, node_id: &str) -> Option<&NodeProfile> {
        self.profiles.get(node_id)
    }

    /// Return the `n` hottest (highest total execution time) node profiles,
    /// sorted descending by `total_us`.
    #[must_use]
    pub fn hottest_nodes(&self, n: usize) -> Vec<&NodeProfile> {
        let mut profiles: Vec<&NodeProfile> = self.profiles.values().collect();
        profiles.sort_by(|a, b| b.total_us.cmp(&a.total_us));
        profiles.truncate(n);
        profiles
    }

    /// Return all profiles as an unsorted slice.
    #[must_use]
    pub fn all_profiles(&self) -> Vec<&NodeProfile> {
        self.profiles.values().collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PortThroughput
// ─────────────────────────────────────────────────────────────────────────────

/// Throughput statistics for a single port.
#[derive(Debug, Clone)]
pub struct PortThroughput {
    /// Port identifier (e.g. `"node_a:output_0"`).
    pub port_id: String,
    /// Total bytes transferred through this port.
    pub bytes_transferred: u64,
    /// Total number of frames transferred.
    pub frames_transferred: u64,
    /// Average frame size in bytes.
    pub avg_frame_size: u64,
}

impl PortThroughput {
    /// Create a new port throughput record.
    #[must_use]
    pub fn new(
        port_id: impl Into<String>,
        bytes_transferred: u64,
        frames_transferred: u64,
    ) -> Self {
        let avg_frame_size = if frames_transferred > 0 {
            bytes_transferred / frames_transferred
        } else {
            0
        };
        Self {
            port_id: port_id.into(),
            bytes_transferred,
            frames_transferred,
            avg_frame_size,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GraphProfilingReport
// ─────────────────────────────────────────────────────────────────────────────

/// A complete profiling report for a graph execution run.
#[derive(Debug, Clone)]
pub struct GraphProfilingReport {
    /// Per-node profiles sorted by total execution time (descending).
    pub node_profiles: Vec<NodeProfile>,
    /// Per-port throughput records.
    pub port_throughputs: Vec<PortThroughput>,
    /// Wall-clock duration of the entire execution in microseconds.
    pub total_duration_us: u64,
    /// Estimated CPU efficiency: `sum(node_total_us) / total_duration_us * 100`.
    pub cpu_efficiency_pct: f32,
}

impl GraphProfilingReport {
    /// Generate a profiling report from a [`GraphProfiler`].
    ///
    /// `port_throughputs` and `total_duration_us` are passed externally as the
    /// profiler itself does not measure wall time or port traffic.
    #[must_use]
    pub fn generate(profiler: &GraphProfiler) -> Self {
        let mut node_profiles: Vec<NodeProfile> = profiler.profiles.values().cloned().collect();
        node_profiles.sort_by(|a, b| b.total_us.cmp(&a.total_us));

        let total_node_us: u64 = node_profiles.iter().map(|p| p.total_us).sum();
        let total_duration_us = total_node_us; // wall time = sum of node times (sequential baseline)

        let cpu_efficiency_pct = if total_duration_us > 0 {
            (total_node_us as f64 / total_duration_us as f64 * 100.0) as f32
        } else {
            100.0
        };

        Self {
            node_profiles,
            port_throughputs: vec![],
            total_duration_us,
            cpu_efficiency_pct,
        }
    }

    /// Generate a full report with explicit port throughputs and wall-clock
    /// duration.
    #[must_use]
    pub fn generate_full(
        profiler: &GraphProfiler,
        port_throughputs: Vec<PortThroughput>,
        total_duration_us: u64,
    ) -> Self {
        let mut report = Self::generate(profiler);
        report.port_throughputs = port_throughputs;

        let total_node_us: u64 = report.node_profiles.iter().map(|p| p.total_us).sum();
        report.total_duration_us = total_duration_us;
        report.cpu_efficiency_pct = if total_duration_us > 0 {
            (total_node_us as f64 / total_duration_us as f64 * 100.0).min(100.0) as f32
        } else {
            100.0
        };

        report
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── NodeProfile ───────────────────────────────────────────────────────────

    #[test]
    fn test_node_profile_initial_state() {
        let p = NodeProfile::new("node_a");
        assert_eq!(p.node_id, "node_a");
        assert_eq!(p.call_count, 0);
        assert_eq!(p.total_us, 0);
        assert_eq!(p.avg_duration_us, 0);
        assert_eq!(p.max_duration_us, 0);
    }

    #[test]
    fn test_node_profile_record_updates_stats() {
        let mut p = NodeProfile::new("n");
        p.record(100);
        p.record(200);
        assert_eq!(p.call_count, 2);
        assert_eq!(p.total_us, 300);
        assert_eq!(p.avg_duration_us, 150);
        assert_eq!(p.max_duration_us, 200);
    }

    #[test]
    fn test_node_profile_std_dev() {
        let mut p = NodeProfile::new("n");
        p.record(100);
        p.record(300); // avg=200, max=300 → std_dev=(300-200)/2=50
        assert_eq!(p.std_dev_us(), 50);
    }

    #[test]
    fn test_node_profile_std_dev_zero_when_equal() {
        let mut p = NodeProfile::new("n");
        p.record(100);
        assert_eq!(p.std_dev_us(), 0); // avg == max when single sample
    }

    // ── GraphProfiler ─────────────────────────────────────────────────────────

    #[test]
    fn test_profiler_record_creates_profile() {
        let mut profiler = GraphProfiler::new();
        profiler.record("node_x", 500);
        assert!(profiler.profile("node_x").is_some());
        assert_eq!(
            profiler
                .profile("node_x")
                .expect("profile should succeed")
                .call_count,
            1
        );
    }

    #[test]
    fn test_profiler_missing_node_returns_none() {
        let profiler = GraphProfiler::new();
        assert!(profiler.profile("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_multiple_records() {
        let mut profiler = GraphProfiler::new();
        for us in [100, 200, 300] {
            profiler.record("n", us);
        }
        let p = profiler.profile("n").expect("profile should succeed");
        assert_eq!(p.call_count, 3);
        assert_eq!(p.total_us, 600);
    }

    #[test]
    fn test_profiler_hottest_nodes_sorted() {
        let mut profiler = GraphProfiler::new();
        profiler.record("slow", 1000);
        profiler.record("fast", 10);
        profiler.record("medium", 500);
        let hot = profiler.hottest_nodes(2);
        assert_eq!(hot[0].node_id, "slow");
        assert_eq!(hot[1].node_id, "medium");
    }

    #[test]
    fn test_profiler_hottest_n_clamped() {
        let mut profiler = GraphProfiler::new();
        profiler.record("a", 100);
        // Requesting more than available should return all.
        let hot = profiler.hottest_nodes(10);
        assert_eq!(hot.len(), 1);
    }

    // ── PortThroughput ────────────────────────────────────────────────────────

    #[test]
    fn test_port_throughput_avg_frame_size() {
        let pt = PortThroughput::new("port_0", 1024, 4);
        assert_eq!(pt.avg_frame_size, 256);
    }

    #[test]
    fn test_port_throughput_zero_frames() {
        let pt = PortThroughput::new("port_0", 0, 0);
        assert_eq!(pt.avg_frame_size, 0);
    }

    // ── GraphProfilingReport ──────────────────────────────────────────────────

    #[test]
    fn test_report_generate_empty_profiler() {
        let profiler = GraphProfiler::new();
        let report = GraphProfilingReport::generate(&profiler);
        assert!(report.node_profiles.is_empty());
        assert_eq!(report.total_duration_us, 0);
        assert!((report.cpu_efficiency_pct - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_report_generate_sorted_profiles() {
        let mut profiler = GraphProfiler::new();
        profiler.record("cheap", 50);
        profiler.record("expensive", 5000);
        let report = GraphProfilingReport::generate(&profiler);
        assert_eq!(report.node_profiles[0].node_id, "expensive");
    }

    #[test]
    fn test_report_generate_full() {
        let mut profiler = GraphProfiler::new();
        profiler.record("n", 1000);
        let pt = PortThroughput::new("p0", 4096, 8);
        let report = GraphProfilingReport::generate_full(&profiler, vec![pt], 2000);
        assert_eq!(report.port_throughputs.len(), 1);
        assert_eq!(report.total_duration_us, 2000);
        // cpu efficiency = 1000/2000 * 100 = 50 %
        assert!((report.cpu_efficiency_pct - 50.0).abs() < 1.0);
    }
}
