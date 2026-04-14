//! Pipeline replay: re-run a saved pipeline configuration.
//!
//! [`PipelineReplayer`] wraps a [`PipelineGraph`] snapshot and re-executes the
//! pipeline's topological ordering, collecting per-node timing and status
//! information without performing real media I/O.  This is useful for:
//!
//! - Debugging: replay a problematic pipeline run with verbose logging.
//! - Testing: verify execution order and dependency resolution offline.
//! - Documentation: generate a human-readable execution trace.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::builder::PipelineBuilder;
//! use oximedia_pipeline::replay::PipelineReplayer;
//! use oximedia_pipeline::node::{SourceConfig, SinkConfig, StreamSpec, FrameFormat, StreamKind};
//!
//! let graph = PipelineBuilder::new()
//!     .source("src", SourceConfig::File("in.mkv".into()))
//!     .scale(1280, 720)
//!     .sink("sink", SinkConfig::File("out.mkv".into()))
//!     .build()
//!     .expect("valid graph");
//!
//! let replayer = PipelineReplayer::new(&graph);
//! let result = replayer.replay();
//! assert!(result.is_ok(), "replay should succeed on a valid graph");
//! ```

use std::time::{Duration, Instant};

use crate::execution_plan::ExecutionPlanner;
use crate::graph::PipelineGraph;
use crate::PipelineError;

// ---------------------------------------------------------------------------
// NodeReplayRecord
// ---------------------------------------------------------------------------

/// Outcome of replaying a single pipeline node.
#[derive(Debug, Clone)]
pub struct NodeReplayRecord {
    /// Node name.
    pub name: String,
    /// Execution order index (0-based topological position).
    pub order: usize,
    /// Simulated execution duration.
    pub duration: Duration,
    /// Whether the node was executed successfully.
    pub success: bool,
    /// Optional status message (e.g., skip reason for conditional nodes).
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// PipelineResult
// ---------------------------------------------------------------------------

/// Aggregated result of a complete pipeline replay run.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Per-node replay records in execution order.
    pub records: Vec<NodeReplayRecord>,
    /// Total replay wall-clock duration.
    pub total_duration: Duration,
    /// Whether all nodes completed successfully.
    pub success: bool,
    /// Number of nodes that were executed.
    pub nodes_executed: usize,
    /// Number of nodes that were skipped (conditional branches not taken).
    pub nodes_skipped: usize,
}

impl PipelineResult {
    /// Return a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Replay: {} nodes executed, {} skipped, total {:.2}ms — {}",
            self.nodes_executed,
            self.nodes_skipped,
            self.total_duration.as_secs_f64() * 1_000.0,
            if self.success { "SUCCESS" } else { "FAILED" }
        )
    }

    /// Return the record for the slowest node, if any.
    #[must_use]
    pub fn slowest_node(&self) -> Option<&NodeReplayRecord> {
        self.records.iter().max_by_key(|r| r.duration)
    }
}

// ---------------------------------------------------------------------------
// PipelineReplayer
// ---------------------------------------------------------------------------

/// Replays a [`PipelineGraph`] by simulating node execution in topological order.
///
/// The replayer uses [`ExecutionPlanner`] to derive the execution order, then
/// visits each node in that order.  No real media I/O is performed; each node
/// contributes a synthetic sub-millisecond duration to the timing report.
pub struct PipelineReplayer<'a> {
    graph: &'a PipelineGraph,
    /// Override: use a fixed simulated duration per node (for deterministic tests).
    simulated_node_duration: Option<Duration>,
}

impl<'a> PipelineReplayer<'a> {
    /// Create a new replayer for the given graph.
    #[must_use]
    pub fn new(graph: &'a PipelineGraph) -> Self {
        Self {
            graph,
            simulated_node_duration: None,
        }
    }

    /// Override the per-node simulated duration (useful for deterministic tests).
    #[must_use]
    pub fn with_simulated_duration(mut self, d: Duration) -> Self {
        self.simulated_node_duration = Some(d);
        self
    }

    /// Replay the pipeline and return a [`PipelineResult`].
    ///
    /// # Errors
    ///
    /// Returns a [`PipelineError`] if the graph contains a cycle or a
    /// referenced node cannot be resolved.
    pub fn replay(&self) -> Result<PipelineResult, PipelineError> {
        let plan = ExecutionPlanner::plan(self.graph)?;

        let overall_start = Instant::now();
        let mut records: Vec<NodeReplayRecord> = Vec::with_capacity(self.graph.nodes.len());
        let mut nodes_skipped = 0usize;

        for (order, stage) in plan.stages.iter().enumerate() {
            for &node_id in &stage.nodes {
                let node_spec = self.graph.nodes.get(&node_id).ok_or_else(|| {
                    PipelineError::NodeNotFound(node_id.to_string())
                })?;

                let node_start = Instant::now();
                let sim_dur = self.simulated_node_duration.unwrap_or(Duration::from_micros(50));
                // Simulate work via a no-op busy loop tracked by Instant.
                let _ = sim_dur; // In a real replay we'd call node.execute()
                let elapsed = node_start.elapsed();

                records.push(NodeReplayRecord {
                    name: node_spec.name.clone(),
                    order,
                    duration: elapsed.max(sim_dur),
                    success: true,
                    message: None,
                });
            }
        }

        // Also count any nodes not covered by the plan (isolated nodes).
        for (_id, spec) in &self.graph.nodes {
            if records.iter().all(|r| r.name != spec.name) {
                nodes_skipped += 1;
                records.push(NodeReplayRecord {
                    name: spec.name.clone(),
                    order: usize::MAX,
                    duration: Duration::ZERO,
                    success: true,
                    message: Some("skipped (not reached by planner)".to_string()),
                });
            }
        }

        let total_duration = overall_start.elapsed();
        let nodes_executed = records.iter().filter(|r| r.order != usize::MAX).count();

        Ok(PipelineResult {
            success: records.iter().all(|r| r.success),
            records,
            total_duration,
            nodes_executed,
            nodes_skipped,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::PipelineBuilder;
    use crate::node::SourceConfig;

    #[test]
    fn test_replay_simple_pipeline() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("in.mkv".into()))
            .scale(1280, 720)
            .sink("sink", crate::node::SinkConfig::File("out.mkv".into()))
            .build()
            .expect("valid graph");

        let replayer = PipelineReplayer::new(&graph)
            .with_simulated_duration(Duration::from_micros(10));
        let result = replayer.replay().expect("replay should succeed");
        assert!(result.success, "all nodes should succeed");
        assert!(result.nodes_executed >= 3, "at least src, scale, sink executed");
    }

    #[test]
    fn test_replay_result_summary() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("in.mkv".into()))
            .sink("sink", crate::node::SinkConfig::File("out.mkv".into()))
            .build()
            .expect("valid graph");

        let replayer = PipelineReplayer::new(&graph);
        let result = replayer.replay().expect("replay should succeed");
        let summary = result.summary();
        assert!(summary.contains("SUCCESS"), "summary should mention SUCCESS");
        assert!(summary.contains("Replay"), "summary should start with Replay");
    }

    #[test]
    fn test_replay_slowest_node() {
        let graph = PipelineBuilder::new()
            .source("src", SourceConfig::File("in.mkv".into()))
            .scale(1280, 720)
            .sink("sink", crate::node::SinkConfig::File("out.mkv".into()))
            .build()
            .expect("valid graph");

        let replayer = PipelineReplayer::new(&graph)
            .with_simulated_duration(Duration::from_micros(100));
        let result = replayer.replay().expect("replay should succeed");
        // slowest_node should return Some since records are non-empty.
        assert!(result.slowest_node().is_some());
    }

    #[test]
    fn test_replay_empty_graph() {
        let graph = PipelineGraph::new();
        let replayer = PipelineReplayer::new(&graph);
        let result = replayer.replay().expect("replay of empty graph should succeed");
        assert_eq!(result.nodes_executed, 0);
        assert!(result.success);
    }
}
