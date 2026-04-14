//! Pipeline debug tooling — frame tap points, debug sink nodes, and frame
//! capture at arbitrary stages of a pipeline graph.
//!
//! [`PipelineDebugger`] attaches named *tap points* to edges in a
//! [`PipelineGraph`].  Each tap injects a lightweight pass-through node whose
//! metadata is recorded so that callers can inspect what streams pass through
//! that location.  [`DebugSink`] provides a no-op sink that collects
//! [`DebugFrame`] descriptors without requiring actual decoded frame data.
//!
//! # Design goals
//!
//! * **Non-destructive**: the original graph topology is preserved; taps only
//!   add pass-through filter nodes.
//! * **Zero-copy**: only frame metadata (format, dimensions, timestamps) is
//!   stored — not the actual pixel data.
//! * **No unsafe / no unwrap** in library code.
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::graph::PipelineGraph;
//! use oximedia_pipeline::node::{NodeSpec, SourceConfig, SinkConfig, StreamSpec, FrameFormat};
//! use oximedia_pipeline::pipeline_debugger::{PipelineDebugger, TapConfig};
//!
//! let mut g = PipelineGraph::new();
//! let vs = StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25);
//! let src = NodeSpec::source("src", SourceConfig::File("in.mkv".into()), vs.clone());
//! let sink = NodeSpec::sink("sink", SinkConfig::Null, vs);
//! let s = g.add_node(src);
//! let sk = g.add_node(sink);
//! g.connect(s, "default", sk, "default").expect("connect ok");
//!
//! let mut dbg = PipelineDebugger::new();
//! let (patched, report) = dbg.attach_taps(g, &[
//!     TapConfig { edge_from_node_name: "src".into(), edge_from_pad: "default".into(), tap_name: "after_src".into() },
//! ]).expect("attach ok");
//!
//! // Tap node injected between src and sink
//! assert!(patched.node_count() >= 3);
//! assert_eq!(report.taps_inserted, 1);
//! ```

use std::collections::HashMap;

use crate::graph::{Edge, PipelineGraph};
use crate::node::{FilterConfig, NodeId, NodeSpec, NodeType, StreamSpec};
use crate::PipelineError;

// ── DebugFrame ────────────────────────────────────────────────────────────────

/// Lightweight metadata descriptor for a frame that passed through a tap point.
///
/// No actual pixel or sample data is stored — only attributes that are useful
/// for diagnostic purposes.
#[derive(Debug, Clone, PartialEq)]
pub struct DebugFrame {
    /// Name of the tap point this frame was observed at.
    pub tap_name: String,
    /// Sequential index of this frame at this tap (0-based).
    pub frame_index: u64,
    /// Presentation timestamp in the stream's native time base.
    pub pts: i64,
    /// Width in pixels (video only; `0` for audio).
    pub width: u32,
    /// Height in pixels (video only; `0` for audio).
    pub height: u32,
    /// Textual description of the frame format.
    pub format_name: String,
    /// Whether this frame was flagged as a keyframe.
    pub is_keyframe: bool,
    /// Estimated size in bytes (may be approximate).
    pub estimated_size_bytes: u64,
}

impl DebugFrame {
    /// Build a `DebugFrame` from a [`StreamSpec`] and positional information.
    pub fn from_stream_spec(
        tap_name: impl Into<String>,
        spec: &StreamSpec,
        frame_index: u64,
        pts: i64,
        is_keyframe: bool,
    ) -> Self {
        let estimated_size = match (spec.width, spec.height) {
            (Some(w), Some(h)) => {
                let bytes_per_px = spec.format.bytes_per_element() as u64;
                w as u64 * h as u64 * bytes_per_px
            }
            _ => {
                // Audio: rough estimate of 1024 samples × channels × bytes/sample
                let ch = spec.channels.unwrap_or(2) as u64;
                let bps = spec.format.bytes_per_element() as u64;
                1024 * ch * bps
            }
        };

        Self {
            tap_name: tap_name.into(),
            frame_index,
            pts,
            width: spec.width.unwrap_or(0),
            height: spec.height.unwrap_or(0),
            format_name: format!("{:?}", spec.format),
            is_keyframe,
            estimated_size_bytes: estimated_size,
        }
    }
}

// ── DebugSink ─────────────────────────────────────────────────────────────────

/// An in-memory accumulator of [`DebugFrame`] records.
///
/// Call [`DebugSink::record`] to ingest a new frame descriptor, and
/// [`DebugSink::frames`] to inspect them later.
#[derive(Debug, Clone, Default)]
pub struct DebugSink {
    frames: Vec<DebugFrame>,
    max_capacity: Option<usize>,
    overflow_count: u64,
}

impl DebugSink {
    /// Create an unbounded `DebugSink`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a `DebugSink` that retains at most `max_capacity` frames.
    /// Frames beyond the limit increment the overflow counter but are not
    /// stored.
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            max_capacity: Some(max_capacity),
            ..Default::default()
        }
    }

    /// Record a `DebugFrame`.  Returns `true` when the frame was stored,
    /// `false` when it was discarded due to capacity limits.
    pub fn record(&mut self, frame: DebugFrame) -> bool {
        if let Some(cap) = self.max_capacity {
            if self.frames.len() >= cap {
                self.overflow_count += 1;
                return false;
            }
        }
        self.frames.push(frame);
        true
    }

    /// Return a slice of all recorded frames.
    pub fn frames(&self) -> &[DebugFrame] {
        &self.frames
    }

    /// Number of frames recorded (not counting overflows).
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Number of frames that were dropped due to capacity overflow.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count
    }

    /// Reset the sink, clearing all stored frames and the overflow counter.
    pub fn clear(&mut self) {
        self.frames.clear();
        self.overflow_count = 0;
    }

    /// Drain and return all stored frames, leaving the sink empty.
    pub fn drain(&mut self) -> Vec<DebugFrame> {
        self.overflow_count = 0;
        std::mem::take(&mut self.frames)
    }
}

// ── TapConfig ─────────────────────────────────────────────────────────────────

/// Describes where to attach a tap point.
#[derive(Debug, Clone)]
pub struct TapConfig {
    /// The `name` of the node whose outgoing edge should be tapped.
    pub edge_from_node_name: String,
    /// The output pad name on that node.
    pub edge_from_pad: String,
    /// A human-readable label for the tap node.
    pub tap_name: String,
}

// ── TapReport ─────────────────────────────────────────────────────────────────

/// Summary produced by [`PipelineDebugger::attach_taps`].
#[derive(Debug, Clone, Default)]
pub struct TapReport {
    /// Number of tap nodes successfully inserted.
    pub taps_inserted: usize,
    /// Names of taps that could not be inserted (e.g. edge not found).
    pub failed_taps: Vec<String>,
    /// Map from tap name → injected [`NodeId`].
    pub tap_node_ids: HashMap<String, NodeId>,
}

impl TapReport {
    /// Returns `true` when all requested taps were inserted.
    pub fn all_inserted(&self) -> bool {
        self.failed_taps.is_empty()
    }
}

// ── TapNodeInfo ───────────────────────────────────────────────────────────────

/// Metadata about an active tap node in a debugged graph.
#[derive(Debug, Clone)]
pub struct TapNodeInfo {
    /// User-assigned tap name.
    pub tap_name: String,
    /// The node id of the injected pass-through node.
    pub node_id: NodeId,
    /// The stream spec the tap observes.
    pub stream_spec: StreamSpec,
    /// Name of the upstream node.
    pub upstream_node_name: String,
    /// Pad on the upstream node.
    pub upstream_pad: String,
}

// ── DebugGraphInfo ────────────────────────────────────────────────────────────

/// Describes the debug instrumentation applied to a graph.
#[derive(Debug, Clone, Default)]
pub struct DebugGraphInfo {
    /// All active tap nodes (keyed by tap name).
    pub taps: HashMap<String, TapNodeInfo>,
}

impl DebugGraphInfo {
    /// Returns `true` when there are no tap nodes registered.
    pub fn is_empty(&self) -> bool {
        self.taps.is_empty()
    }

    /// Look up a tap by its user-assigned name.
    pub fn tap(&self, name: &str) -> Option<&TapNodeInfo> {
        self.taps.get(name)
    }
}

// ── PipelineDebugger ──────────────────────────────────────────────────────────

/// Instruments a [`PipelineGraph`] with named tap points and debug sinks.
///
/// The debugger injects lightweight pass-through nodes at requested edges,
/// enabling downstream tooling to capture frame metadata at arbitrary stages.
#[derive(Debug, Default)]
pub struct PipelineDebugger {
    /// Accumulated debug info about the graphs processed so far.
    pub graph_info: DebugGraphInfo,
}

impl PipelineDebugger {
    /// Create a new `PipelineDebugger`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach tap points to `graph` according to the given `configs`.
    ///
    /// For each `TapConfig`, the debugger:
    /// 1. Finds the edge(s) that originate from the named node+pad.
    /// 2. Inserts a pass-through (`Null` type) node labelled with the tap name.
    /// 3. Rewires the edge through the new tap node.
    ///
    /// Returns the modified graph and a [`TapReport`].
    pub fn attach_taps(
        &mut self,
        graph: PipelineGraph,
        configs: &[TapConfig],
    ) -> Result<(PipelineGraph, TapReport), PipelineError> {
        let mut result = PipelineGraph::new();
        // Copy existing nodes.
        for (_, spec) in &graph.nodes {
            result.nodes.insert(spec.id, spec.clone());
        }
        result.edges = graph.edges.clone();

        let mut report = TapReport::default();

        for config in configs {
            // Find the upstream node by name.
            let upstream_id = result
                .nodes
                .iter()
                .find(|(_, s)| s.name == config.edge_from_node_name)
                .map(|(id, _)| *id);

            let upstream_id = match upstream_id {
                Some(id) => id,
                None => {
                    report.failed_taps.push(config.tap_name.clone());
                    continue;
                }
            };

            // Find the stream spec on the specified output pad.
            let stream_spec = result
                .nodes
                .get(&upstream_id)
                .and_then(|s| {
                    s.output_pads
                        .iter()
                        .find(|(n, _)| n == &config.edge_from_pad)
                        .map(|(_, sp)| sp.clone())
                });

            let stream_spec = match stream_spec {
                Some(sp) => sp,
                None => {
                    report.failed_taps.push(config.tap_name.clone());
                    continue;
                }
            };

            // Build the tap node (a pass-through Null node with a single
            // input and a single output, both carrying the same stream spec).
            let tap_node = NodeSpec::new(
                config.tap_name.clone(),
                NodeType::Null,
                vec![("default".to_string(), stream_spec.clone())],
                vec![("default".to_string(), stream_spec.clone())],
            );
            let tap_id = tap_node.id;
            result.nodes.insert(tap_id, tap_node);

            // Find all edges from upstream_id:from_pad and rewire them through
            // the tap.
            let mut edges_to_rewire: Vec<usize> = Vec::new();
            for (idx, edge) in result.edges.iter().enumerate() {
                if edge.from_node == upstream_id && edge.from_pad == config.edge_from_pad {
                    edges_to_rewire.push(idx);
                }
            }

            if edges_to_rewire.is_empty() {
                // No outgoing edges on that pad — the tap node is added but
                // not wired to anything downstream.
                // Still add an edge from upstream → tap.
                result.edges.push(Edge {
                    from_node: upstream_id,
                    from_pad: config.edge_from_pad.clone(),
                    to_node: tap_id,
                    to_pad: "default".to_string(),
                });
            } else {
                // Rewire: upstream → tap (insert) and tap → each downstream.
                let rewired_targets: Vec<(NodeId, String)> = edges_to_rewire
                    .iter()
                    .map(|&idx| {
                        let e = &result.edges[idx];
                        (e.to_node, e.to_pad.clone())
                    })
                    .collect();

                // Remove the original edges (high-index first to preserve indices).
                let mut sorted_indices = edges_to_rewire.clone();
                sorted_indices.sort_unstable_by(|a, b| b.cmp(a));
                for idx in sorted_indices {
                    result.edges.remove(idx);
                }

                // Add upstream → tap.
                result.edges.push(Edge {
                    from_node: upstream_id,
                    from_pad: config.edge_from_pad.clone(),
                    to_node: tap_id,
                    to_pad: "default".to_string(),
                });

                // Add tap → each original downstream.
                for (dst_node, dst_pad) in rewired_targets {
                    result.edges.push(Edge {
                        from_node: tap_id,
                        from_pad: "default".to_string(),
                        to_node: dst_node,
                        to_pad: dst_pad,
                    });
                }
            }

            let upstream_name = graph
                .nodes
                .get(&upstream_id)
                .map(|s| s.name.clone())
                .unwrap_or_default();

            let info = TapNodeInfo {
                tap_name: config.tap_name.clone(),
                node_id: tap_id,
                stream_spec,
                upstream_node_name: upstream_name,
                upstream_pad: config.edge_from_pad.clone(),
            };
            self.graph_info.taps.insert(config.tap_name.clone(), info);
            report.tap_node_ids.insert(config.tap_name.clone(), tap_id);
            report.taps_inserted += 1;
        }

        Ok((result, report))
    }

    /// Remove all tap nodes from `graph` that are registered in this debugger,
    /// restoring the original wiring.
    ///
    /// Returns the cleaned graph.
    pub fn remove_taps(&self, graph: PipelineGraph) -> PipelineGraph {
        let tap_ids: std::collections::HashSet<NodeId> = self
            .graph_info
            .taps
            .values()
            .map(|t| t.node_id)
            .collect();

        if tap_ids.is_empty() {
            return graph;
        }

        let mut result = PipelineGraph::new();
        // Copy non-tap nodes.
        for (id, spec) in &graph.nodes {
            if !tap_ids.contains(id) {
                result.nodes.insert(*id, spec.clone());
            }
        }

        // Reconstruct edges, bypassing tap nodes.
        // For each edge whose `to_node` is a tap, follow through to find the
        // tap's downstream edge and merge into a direct edge.
        let edge_map: HashMap<(NodeId, String), Vec<(NodeId, String)>> = {
            let mut m: HashMap<(NodeId, String), Vec<(NodeId, String)>> = HashMap::new();
            for e in &graph.edges {
                m.entry((e.from_node, e.from_pad.clone()))
                    .or_default()
                    .push((e.to_node, e.to_pad.clone()));
            }
            m
        };

        for edge in &graph.edges {
            // If this edge's `from_node` is a tap, skip it (it will be
            // replaced by the bypass logic below).
            if tap_ids.contains(&edge.from_node) {
                continue;
            }
            // If this edge leads to a tap, follow the tap's outgoing edges and
            // add bypass edges.
            if tap_ids.contains(&edge.to_node) {
                let tap_out_key = (edge.to_node, "default".to_string());
                if let Some(downstreams) = edge_map.get(&tap_out_key) {
                    for (dst_node, dst_pad) in downstreams {
                        if !tap_ids.contains(dst_node) {
                            result.edges.push(Edge {
                                from_node: edge.from_node,
                                from_pad: edge.from_pad.clone(),
                                to_node: *dst_node,
                                to_pad: dst_pad.clone(),
                            });
                        }
                    }
                }
            } else {
                result.edges.push(edge.clone());
            }
        }

        result
    }

    /// Generate a textual debug report of all active tap points.
    pub fn tap_summary(&self) -> String {
        if self.graph_info.is_empty() {
            return "No tap points registered.".to_string();
        }
        let mut lines = Vec::new();
        let mut names: Vec<&str> = self.graph_info.taps.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        for name in names {
            if let Some(info) = self.graph_info.taps.get(name) {
                lines.push(format!(
                    "tap '{}' at {}.{} (node_id={})",
                    info.tap_name, info.upstream_node_name, info.upstream_pad, info.node_id,
                ));
            }
        }
        lines.join("\n")
    }
}

/// Build a debug pass-through filter node that carries an identity `Format`
/// filter (no conversion) so that the tap is recognised as a filter in the
/// graph for validation purposes.
///
/// This is an alternative to `NodeType::Null` when the validator requires all
/// non-source/sink nodes to be filters.
pub fn make_debug_filter_node(tap_name: impl Into<String>, spec: StreamSpec) -> NodeSpec {
    NodeSpec::filter(
        tap_name,
        FilterConfig::Format(spec.format),
        spec.clone(),
        spec,
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{FrameFormat, SinkConfig, SourceConfig, StreamSpec};

    fn vs() -> StreamSpec {
        StreamSpec::video(FrameFormat::Yuv420p, 1920, 1080, 25)
    }

    fn simple_graph() -> (PipelineGraph, NodeId, NodeId) {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mkv".into()), vs());
        let sink = NodeSpec::sink("sink", SinkConfig::Null, vs());
        let s = g.add_node(src);
        let sk = g.add_node(sink);
        g.connect(s, "default", sk, "default").expect("connect ok");
        (g, s, sk)
    }

    // 1. attach_taps injects a tap node
    #[test]
    fn attach_tap_inserts_node() {
        let (g, _, _) = simple_graph();
        let mut dbg = PipelineDebugger::new();
        let tap_cfg = TapConfig {
            edge_from_node_name: "src".into(),
            edge_from_pad: "default".into(),
            tap_name: "after_src".into(),
        };
        let (patched, report) = dbg.attach_taps(g, &[tap_cfg]).expect("attach ok");
        assert_eq!(report.taps_inserted, 1);
        assert!(report.all_inserted());
        // src + tap + sink = 3
        assert_eq!(patched.node_count(), 3);
    }

    // 2. Tap on unknown node records a failure
    #[test]
    fn attach_tap_unknown_node_fails_gracefully() {
        let (g, _, _) = simple_graph();
        let mut dbg = PipelineDebugger::new();
        let tap_cfg = TapConfig {
            edge_from_node_name: "nonexistent".into(),
            edge_from_pad: "default".into(),
            tap_name: "ghost_tap".into(),
        };
        let (_, report) = dbg.attach_taps(g, &[tap_cfg]).expect("no error");
        assert_eq!(report.taps_inserted, 0);
        assert!(!report.all_inserted());
        assert!(report.failed_taps.contains(&"ghost_tap".to_string()));
    }

    // 3. Multiple taps inserted at once
    #[test]
    fn multiple_taps_inserted() {
        let mut g = PipelineGraph::new();
        let src = NodeSpec::source("src", SourceConfig::File("in.mkv".into()), vs());
        let filt = NodeSpec::filter(
            "scale",
            crate::node::FilterConfig::Scale { width: 640, height: 360 },
            vs(),
            StreamSpec::video(FrameFormat::Yuv420p, 640, 360, 25),
        );
        let sink = NodeSpec::sink(
            "sink",
            SinkConfig::Null,
            StreamSpec::video(FrameFormat::Yuv420p, 640, 360, 25),
        );
        let s = g.add_node(src);
        let f = g.add_node(filt);
        let sk = g.add_node(sink);
        g.connect(s, "default", f, "default").expect("ok");
        g.connect(f, "default", sk, "default").expect("ok");

        let mut dbg = PipelineDebugger::new();
        let (patched, report) = dbg
            .attach_taps(
                g,
                &[
                    TapConfig {
                        edge_from_node_name: "src".into(),
                        edge_from_pad: "default".into(),
                        tap_name: "tap_src".into(),
                    },
                    TapConfig {
                        edge_from_node_name: "scale".into(),
                        edge_from_pad: "default".into(),
                        tap_name: "tap_post_scale".into(),
                    },
                ],
            )
            .expect("ok");
        assert_eq!(report.taps_inserted, 2);
        // src + tap_src + scale + tap_post_scale + sink = 5
        assert_eq!(patched.node_count(), 5);
    }

    // 4. remove_taps restores original graph structure
    #[test]
    fn remove_taps_restores_graph() {
        let (g, _, _) = simple_graph();
        let original_node_count = g.node_count();
        let mut dbg = PipelineDebugger::new();
        let tap_cfg = TapConfig {
            edge_from_node_name: "src".into(),
            edge_from_pad: "default".into(),
            tap_name: "t1".into(),
        };
        let (patched, _) = dbg.attach_taps(g, &[tap_cfg]).expect("ok");
        let restored = dbg.remove_taps(patched);
        assert_eq!(restored.node_count(), original_node_count);
        assert_eq!(restored.edges.len(), 1);
    }

    // 5. DebugSink basic recording
    #[test]
    fn debug_sink_records_frames() {
        let mut sink = DebugSink::new();
        let frame = DebugFrame::from_stream_spec("tap1", &vs(), 0, 0, true);
        assert!(sink.record(frame));
        assert_eq!(sink.frame_count(), 1);
    }

    // 6. DebugSink capacity overflow
    #[test]
    fn debug_sink_capacity_overflow() {
        let mut sink = DebugSink::with_capacity(2);
        for i in 0..5 {
            let frame = DebugFrame::from_stream_spec("t", &vs(), i, i as i64, false);
            sink.record(frame);
        }
        assert_eq!(sink.frame_count(), 2);
        assert_eq!(sink.overflow_count(), 3);
    }

    // 7. DebugSink drain clears the sink
    #[test]
    fn debug_sink_drain() {
        let mut sink = DebugSink::new();
        for i in 0..3 {
            let frame = DebugFrame::from_stream_spec("t", &vs(), i, i as i64, false);
            sink.record(frame);
        }
        let drained = sink.drain();
        assert_eq!(drained.len(), 3);
        assert_eq!(sink.frame_count(), 0);
        assert_eq!(sink.overflow_count(), 0);
    }

    // 8. tap_summary output
    #[test]
    fn tap_summary_contains_tap_name() {
        let (g, _, _) = simple_graph();
        let mut dbg = PipelineDebugger::new();
        let tap_cfg = TapConfig {
            edge_from_node_name: "src".into(),
            edge_from_pad: "default".into(),
            tap_name: "my_tap".into(),
        };
        dbg.attach_taps(g, &[tap_cfg]).expect("ok");
        let summary = dbg.tap_summary();
        assert!(summary.contains("my_tap"), "summary: {summary}");
    }

    // 9. DebugFrame estimated_size calculation for video
    #[test]
    fn debug_frame_size_estimate_video() {
        let frame = DebugFrame::from_stream_spec("t", &vs(), 0, 0, false);
        // yuv420p: 1 byte/element, so ~1920*1080 = 2_073_600
        assert_eq!(frame.estimated_size_bytes, 1920 * 1080 * 1);
        assert_eq!(frame.width, 1920);
        assert_eq!(frame.height, 1080);
    }

    // 10. make_debug_filter_node produces a valid filter
    #[test]
    fn make_debug_filter_node_is_filter() {
        let node = make_debug_filter_node("debug_node", vs());
        assert_eq!(node.name, "debug_node");
        assert!(matches!(node.node_type, NodeType::Filter(_)));
        assert_eq!(node.input_pads.len(), 1);
        assert_eq!(node.output_pads.len(), 1);
    }
}
