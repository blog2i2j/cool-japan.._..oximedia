//! Distributed profiling aggregation across multiple nodes.
//!
//! In multi-node encoding clusters or distributed media pipelines, each node
//! records its own profiling spans locally.  This module provides
//! `DistributedProfileAggregator` to merge profiles from multiple nodes,
//! compensating for clock skew via NTP-style offset estimation.
//!
//! # Clock skew compensation
//!
//! Each `NodeProfile` carries a `clock_offset_ns` (signed, in nanoseconds)
//! representing the estimated difference between the node's clock and a
//! reference clock:
//!
//! ```text
//! t_reference = t_node + clock_offset_ns
//! ```
//!
//! The offset can be computed externally (e.g. NTP, PTP) and passed in.
//! The aggregator also provides a helper `estimate_offset_ntp` that uses four
//! timestamps from a simple NTP-style exchange to compute the offset.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::distributed_profile::{
//!     NodeProfile, NodeSpan, DistributedProfileAggregator,
//! };
//!
//! let mut agg = DistributedProfileAggregator::new();
//!
//! let profile_a = NodeProfile::new("node-a", 0, vec![
//!     NodeSpan::new("decode", 1_000, 5_000),
//! ]);
//! let profile_b = NodeProfile::new("node-b", 500, vec![
//!     NodeSpan::new("encode", 2_000, 6_000),
//! ]);
//!
//! agg.add_profile(profile_a);
//! agg.add_profile(profile_b);
//!
//! let result = agg.aggregate();
//! assert_eq!(result.total_spans(), 2);
//! ```

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// NodeSpan
// ---------------------------------------------------------------------------

/// A single profiling span recorded on one node.
///
/// Timestamps are in **nanoseconds** on the local node clock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSpan {
    /// Human-readable span name.
    pub name: String,
    /// Start time in nanoseconds (node-local clock).
    pub start_ns: u64,
    /// End time in nanoseconds (node-local clock).
    pub end_ns: u64,
    /// Optional parent span name for hierarchy reconstruction.
    pub parent: Option<String>,
    /// Optional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl NodeSpan {
    /// Creates a new span with the given name and time range.
    #[must_use]
    pub fn new(name: impl Into<String>, start_ns: u64, end_ns: u64) -> Self {
        Self {
            name: name.into(),
            start_ns,
            end_ns,
            parent: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder: sets the parent span name.
    #[must_use]
    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    /// Builder: adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns the duration in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        self.end_ns.saturating_sub(self.start_ns)
    }
}

// ---------------------------------------------------------------------------
// NodeProfile
// ---------------------------------------------------------------------------

/// Profiling data from a single node in a distributed system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeProfile {
    /// Unique identifier for this node.
    pub node_id: String,
    /// Clock offset in nanoseconds (signed).
    ///
    /// `t_reference = t_node + clock_offset_ns`
    ///
    /// A positive value means the node clock is behind the reference.
    pub clock_offset_ns: i64,
    /// Spans recorded on this node.
    pub spans: Vec<NodeSpan>,
}

impl NodeProfile {
    /// Creates a new node profile.
    #[must_use]
    pub fn new(
        node_id: impl Into<String>,
        clock_offset_ns: i64,
        spans: Vec<NodeSpan>,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            clock_offset_ns,
            spans,
        }
    }

    /// Returns the total number of spans.
    #[must_use]
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    /// Returns the total wall-clock duration across all spans (local time).
    #[must_use]
    pub fn total_duration_ns(&self) -> u64 {
        self.spans.iter().map(|s| s.duration_ns()).sum()
    }

    /// Returns the earliest span start time (local time), or `None` if empty.
    #[must_use]
    pub fn earliest_start_ns(&self) -> Option<u64> {
        self.spans.iter().map(|s| s.start_ns).min()
    }

    /// Returns the latest span end time (local time), or `None` if empty.
    #[must_use]
    pub fn latest_end_ns(&self) -> Option<u64> {
        self.spans.iter().map(|s| s.end_ns).max()
    }
}

// ---------------------------------------------------------------------------
// AdjustedSpan — span with reference-clock timestamps
// ---------------------------------------------------------------------------

/// A span whose timestamps have been adjusted to the reference clock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustedSpan {
    /// Source node id.
    pub node_id: String,
    /// Span name.
    pub name: String,
    /// Start time on reference clock (nanoseconds).
    pub start_ns: i64,
    /// End time on reference clock (nanoseconds).
    pub end_ns: i64,
    /// Parent span name (if any).
    pub parent: Option<String>,
    /// Metadata from the original span.
    pub metadata: HashMap<String, String>,
}

impl AdjustedSpan {
    /// Duration in nanoseconds (always non-negative).
    #[must_use]
    pub fn duration_ns(&self) -> u64 {
        (self.end_ns.saturating_sub(self.start_ns)) as u64
    }
}

// ---------------------------------------------------------------------------
// AggregatedProfile
// ---------------------------------------------------------------------------

/// The result of merging profiles from multiple nodes.
///
/// Contains a unified timeline of `AdjustedSpan`s sorted by reference-clock
/// start time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProfile {
    /// All spans from all nodes, sorted by reference-clock start time.
    pub timeline: Vec<AdjustedSpan>,
    /// Per-node summary statistics.
    pub node_summaries: HashMap<String, NodeSummary>,
}

impl AggregatedProfile {
    /// Total number of spans across all nodes.
    #[must_use]
    pub fn total_spans(&self) -> usize {
        self.timeline.len()
    }

    /// Number of distinct nodes contributing to this profile.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_summaries.len()
    }

    /// Returns all spans originating from the given node.
    #[must_use]
    pub fn spans_for_node(&self, node_id: &str) -> Vec<&AdjustedSpan> {
        self.timeline
            .iter()
            .filter(|s| s.node_id == node_id)
            .collect()
    }

    /// Returns the overall time range `(min_start, max_end)` in reference
    /// clock nanoseconds, or `None` if the timeline is empty.
    #[must_use]
    pub fn time_range(&self) -> Option<(i64, i64)> {
        let min_start = self.timeline.iter().map(|s| s.start_ns).min()?;
        let max_end = self.timeline.iter().map(|s| s.end_ns).max()?;
        Some((min_start, max_end))
    }

    /// Generates a human-readable summary of the aggregated profile.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Distributed Profile Summary ===\n");
        out.push_str(&format!("Nodes: {}\n", self.node_count()));
        out.push_str(&format!("Total spans: {}\n", self.total_spans()));

        if let Some((min_s, max_e)) = self.time_range() {
            let range_us = (max_e - min_s) as f64 / 1_000.0;
            out.push_str(&format!("Time range: {:.1} us\n", range_us));
        }

        out.push('\n');
        for (node_id, summary) in &self.node_summaries {
            out.push_str(&format!(
                "  {}: {} spans, offset={}ns, total_dur={}ns\n",
                node_id, summary.span_count, summary.clock_offset_ns, summary.total_duration_ns,
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// NodeSummary
// ---------------------------------------------------------------------------

/// Per-node statistics in the aggregated result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeSummary {
    /// Number of spans from this node.
    pub span_count: usize,
    /// Clock offset applied during aggregation.
    pub clock_offset_ns: i64,
    /// Sum of all span durations (local time).
    pub total_duration_ns: u64,
}

// ---------------------------------------------------------------------------
// NTP offset estimation
// ---------------------------------------------------------------------------

/// Estimates clock offset using four timestamps from an NTP-style exchange.
///
/// The four timestamps model a round-trip measurement:
///
/// ```text
///   t1 = client send time (reference clock)
///   t2 = server receive time (node clock)
///   t3 = server send time (node clock)
///   t4 = client receive time (reference clock)
/// ```
///
/// The estimated offset is: `offset = ((t2 - t1) + (t3 - t4)) / 2`
///
/// This value should be stored as `clock_offset_ns` in `NodeProfile` so that
/// `t_reference = t_node + offset`.
#[must_use]
pub fn estimate_offset_ntp(t1: i64, t2: i64, t3: i64, t4: i64) -> i64 {
    // offset = ((t2 - t1) + (t3 - t4)) / 2
    let d1 = t2 - t1;
    let d2 = t3 - t4;
    (d1 + d2) / 2
}

/// Estimates the round-trip delay from an NTP exchange.
///
/// ```text
/// delay = (t4 - t1) - (t3 - t2)
/// ```
#[must_use]
pub fn estimate_rtt_ntp(t1: i64, t2: i64, t3: i64, t4: i64) -> i64 {
    (t4 - t1) - (t3 - t2)
}

// ---------------------------------------------------------------------------
// DistributedProfileAggregator
// ---------------------------------------------------------------------------

/// Merges profiling data from multiple nodes into a unified timeline.
///
/// Clock skew is compensated using the `clock_offset_ns` stored in each
/// `NodeProfile`.
#[derive(Debug, Clone, Default)]
pub struct DistributedProfileAggregator {
    profiles: Vec<NodeProfile>,
}

impl DistributedProfileAggregator {
    /// Creates a new, empty aggregator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node profile to the aggregator.
    pub fn add_profile(&mut self, profile: NodeProfile) {
        self.profiles.push(profile);
    }

    /// Returns the number of profiles currently held.
    #[must_use]
    pub fn profile_count(&self) -> usize {
        self.profiles.len()
    }

    /// Clears all stored profiles.
    pub fn clear(&mut self) {
        self.profiles.clear();
    }

    /// Aggregates all profiles into a unified `AggregatedProfile`.
    ///
    /// Each span's timestamps are adjusted by the node's `clock_offset_ns`:
    /// `t_ref = t_node + offset`.  The resulting timeline is sorted by
    /// reference-clock start time.
    #[must_use]
    pub fn aggregate(&self) -> AggregatedProfile {
        let mut timeline = Vec::new();
        let mut node_summaries = HashMap::new();

        for profile in &self.profiles {
            let offset = profile.clock_offset_ns;

            let summary = NodeSummary {
                span_count: profile.span_count(),
                clock_offset_ns: offset,
                total_duration_ns: profile.total_duration_ns(),
            };
            node_summaries.insert(profile.node_id.clone(), summary);

            for span in &profile.spans {
                let adjusted = AdjustedSpan {
                    node_id: profile.node_id.clone(),
                    name: span.name.clone(),
                    start_ns: span.start_ns as i64 + offset,
                    end_ns: span.end_ns as i64 + offset,
                    parent: span.parent.clone(),
                    metadata: span.metadata.clone(),
                };
                timeline.push(adjusted);
            }
        }

        // Sort by reference-clock start time.
        timeline.sort_by_key(|s| s.start_ns);

        AggregatedProfile {
            timeline,
            node_summaries,
        }
    }

    /// Aggregates and returns only spans that overlap the given reference-clock
    /// time window `[start_ns, end_ns)`.
    #[must_use]
    pub fn aggregate_window(&self, start_ns: i64, end_ns: i64) -> AggregatedProfile {
        let full = self.aggregate();
        let filtered: Vec<AdjustedSpan> = full
            .timeline
            .into_iter()
            .filter(|s| s.end_ns > start_ns && s.start_ns < end_ns)
            .collect();

        AggregatedProfile {
            timeline: filtered,
            node_summaries: full.node_summaries,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profiles() -> (NodeProfile, NodeProfile) {
        let a = NodeProfile::new(
            "node-a",
            0,
            vec![
                NodeSpan::new("decode", 1_000, 5_000),
                NodeSpan::new("filter", 5_000, 8_000),
            ],
        );
        let b = NodeProfile::new(
            "node-b",
            500, // node-b is 500ns behind reference
            vec![
                NodeSpan::new("encode", 2_000, 7_000),
                NodeSpan::new("mux", 7_000, 9_000),
            ],
        );
        (a, b)
    }

    #[test]
    fn test_aggregate_basic() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        let result = agg.aggregate();
        assert_eq!(result.total_spans(), 4);
        assert_eq!(result.node_count(), 2);
    }

    #[test]
    fn test_clock_offset_applied() {
        let profile = NodeProfile::new(
            "n1",
            1_000, // offset +1000ns
            vec![NodeSpan::new("work", 100, 200)],
        );
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(profile);
        let result = agg.aggregate();
        // t_ref = 100 + 1000 = 1100
        assert_eq!(result.timeline[0].start_ns, 1_100);
        assert_eq!(result.timeline[0].end_ns, 1_200);
    }

    #[test]
    fn test_negative_offset() {
        let profile = NodeProfile::new(
            "n1",
            -500,
            vec![NodeSpan::new("work", 1_000, 2_000)],
        );
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(profile);
        let result = agg.aggregate();
        assert_eq!(result.timeline[0].start_ns, 500);
        assert_eq!(result.timeline[0].end_ns, 1_500);
    }

    #[test]
    fn test_timeline_sorted_by_start() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        let result = agg.aggregate();
        for window in result.timeline.windows(2) {
            assert!(window[0].start_ns <= window[1].start_ns);
        }
    }

    #[test]
    fn test_spans_for_node() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        let result = agg.aggregate();
        let a_spans = result.spans_for_node("node-a");
        assert_eq!(a_spans.len(), 2);
        for s in &a_spans {
            assert_eq!(s.node_id, "node-a");
        }
    }

    #[test]
    fn test_time_range() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        let result = agg.aggregate();
        let (min_s, max_e) = result.time_range().expect("non-empty");
        // node-a: decode starts at 1000+0=1000
        // node-b: mux ends at 9000+500=9500
        assert_eq!(min_s, 1_000);
        assert_eq!(max_e, 9_500);
    }

    #[test]
    fn test_ntp_offset_estimation() {
        // Symmetric exchange: offset should be 0 when clocks are synchronized
        let offset = estimate_offset_ntp(100, 100, 200, 200);
        assert_eq!(offset, 0);

        // Node clock is 50ns ahead: t2 = t1 + 50, t3 = t4 + 50
        // offset = ((150-100) + (250-200)) / 2 = (50 + 50) / 2 = 50
        let offset = estimate_offset_ntp(100, 150, 250, 200);
        assert_eq!(offset, 50);
    }

    #[test]
    fn test_ntp_rtt_estimation() {
        // rtt = (t4-t1) - (t3-t2)
        let rtt = estimate_rtt_ntp(100, 110, 120, 200);
        // (200-100) - (120-110) = 100 - 10 = 90
        assert_eq!(rtt, 90);
    }

    #[test]
    fn test_empty_aggregator() {
        let agg = DistributedProfileAggregator::new();
        let result = agg.aggregate();
        assert_eq!(result.total_spans(), 0);
        assert_eq!(result.node_count(), 0);
        assert!(result.time_range().is_none());
    }

    #[test]
    fn test_node_summary_stats() {
        let (a, _) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        let result = agg.aggregate();
        let summary = result.node_summaries.get("node-a").expect("must exist");
        assert_eq!(summary.span_count, 2);
        assert_eq!(summary.clock_offset_ns, 0);
        // decode: 4000ns, filter: 3000ns = 7000ns total
        assert_eq!(summary.total_duration_ns, 7_000);
    }

    #[test]
    fn test_aggregate_window() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        // Window [0, 3000) should capture decode (1000-5000 overlaps) and encode (2500-7500 overlaps)
        let result = agg.aggregate_window(0, 3_000);
        // decode: start=1000 end=5000 -> overlaps [0,3000)
        // filter: start=5000 end=8000 -> does not overlap
        // encode: start=2500 end=7500 -> overlaps
        // mux: start=7500 end=9500 -> does not overlap
        assert!(result.total_spans() >= 2);
    }

    #[test]
    fn test_node_span_with_metadata() {
        let span = NodeSpan::new("work", 100, 200)
            .with_parent("parent_op")
            .with_metadata("codec", "vp9");
        assert_eq!(span.parent.as_deref(), Some("parent_op"));
        assert_eq!(span.metadata.get("codec").map(|s| s.as_str()), Some("vp9"));
        assert_eq!(span.duration_ns(), 100);
    }

    #[test]
    fn test_aggregator_clear() {
        let (a, _) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        assert_eq!(agg.profile_count(), 1);
        agg.clear();
        assert_eq!(agg.profile_count(), 0);
    }

    #[test]
    fn test_summary_output() {
        let (a, b) = sample_profiles();
        let mut agg = DistributedProfileAggregator::new();
        agg.add_profile(a);
        agg.add_profile(b);
        let result = agg.aggregate();
        let summary = result.summary();
        assert!(summary.contains("Distributed Profile Summary"));
        assert!(summary.contains("node-a"));
        assert!(summary.contains("node-b"));
        assert!(summary.contains("Nodes: 2"));
        assert!(summary.contains("Total spans: 4"));
    }

    #[test]
    fn test_adjusted_span_duration() {
        let span = AdjustedSpan {
            node_id: "n".to_owned(),
            name: "x".to_owned(),
            start_ns: 100,
            end_ns: 350,
            parent: None,
            metadata: HashMap::new(),
        };
        assert_eq!(span.duration_ns(), 250);
    }
}
