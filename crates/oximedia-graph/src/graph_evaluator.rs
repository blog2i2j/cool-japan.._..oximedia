//! Graph evaluation: topological execution, cycle detection facade, dynamic editing,
//! serialization helpers, conditional routing, gain node, and multi-input node.
//!
//! This module wires together the various graph-processing primitives (topological sort,
//! cycle detection, frame pool) into concrete high-level APIs.

#![allow(dead_code)]

#[allow(unused_imports)]
use std::collections::{HashMap, HashSet, VecDeque};

use crate::cycle_detect::CycleGraph;
use crate::error::{GraphError, GraphResult};
use crate::frame::FilterFrame;
use crate::node::{Node, NodeId, NodeState, NodeType};
use crate::port::{AudioPortFormat, InputPort, OutputPort, PortFormat, PortId, PortType};

#[allow(unused_imports)]
use oximedia_audio::{AudioBuffer, AudioFrame, ChannelLayout};
#[allow(unused_imports)]
use oximedia_codec::VideoFrame;
#[allow(unused_imports)]
use oximedia_core::{PixelFormat, SampleFormat, Timestamp};

// ─── Cycle detection facade ──────────────────────────────────────────────────

/// Error returned when a cycle is found during graph validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    /// Nodes that form or participate in a cycle.
    pub cycle_nodes: Vec<NodeId>,
}

impl std::fmt::Display for CycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cycle detected in graph involving {} nodes", self.cycle_nodes.len())
    }
}

impl std::error::Error for CycleError {}

// ─── Simple graph descriptor used by evaluator / validator ───────────────────

/// A lightweight directed-graph descriptor used by the evaluator layer.
///
/// This is a plain data structure — it does not own `Node` trait objects.
/// Use `FilterGraph` from `crate::graph` for full node execution.
#[derive(Debug, Clone, Default)]
pub struct SimpleGraph {
    /// Adjacency list: source NodeId → list of successor NodeIds.
    pub edges: HashMap<NodeId, Vec<NodeId>>,
    /// Node labels for serialization and diagnostics.
    pub labels: HashMap<NodeId, String>,
}

impl SimpleGraph {
    /// Create a new empty graph descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node with a label.
    pub fn add_node(&mut self, id: NodeId, label: impl Into<String>) {
        self.labels.insert(id, label.into());
        self.edges.entry(id).or_default();
    }

    /// Add a directed edge (implicitly creates both nodes if absent).
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) {
        self.edges.entry(from).or_default().push(to);
        self.edges.entry(to).or_default();
        self.labels.entry(from).or_insert_with(|| format!("{}", from.0));
        self.labels.entry(to).or_insert_with(|| format!("{}", to.0));
    }

    /// Return the number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.edges.len()
    }

    /// Return the number of directed edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// Return all node IDs sorted for determinism.
    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids: Vec<NodeId> = self.edges.keys().copied().collect();
        ids.sort_by_key(|id| id.0);
        ids
    }
}

// ─── GraphEvaluator ──────────────────────────────────────────────────────────

/// Evaluates a filter graph by executing nodes in topological order.
///
/// # Example
///
/// ```ignore
/// use oximedia_graph::graph_evaluator::GraphEvaluator;
/// let order = GraphEvaluator::topological_sort(&graph).unwrap();
/// ```
pub struct GraphEvaluator;

impl GraphEvaluator {
    /// Perform a topological sort of the graph using Kahn's algorithm.
    ///
    /// Returns the nodes in a valid processing order (every edge `a → b`
    /// guarantees `a` appears before `b` in the result).
    ///
    /// # Errors
    ///
    /// Returns [`CycleError`] if the graph contains a cycle.
    pub fn topological_sort(graph: &SimpleGraph) -> Result<Vec<NodeId>, CycleError> {
        // Build in-degree map
        let mut in_degree: HashMap<NodeId, usize> = graph
            .edges
            .keys()
            .map(|&id| (id, 0))
            .collect();

        for successors in graph.edges.values() {
            for &succ in successors {
                *in_degree.entry(succ).or_insert(0) += 1;
            }
        }

        // Enqueue nodes with in-degree 0 (sorted for determinism)
        let mut queue: VecDeque<NodeId> = {
            let mut zeros: Vec<NodeId> = in_degree
                .iter()
                .filter(|(_, &d)| d == 0)
                .map(|(&id, _)| id)
                .collect();
            zeros.sort_by_key(|id| id.0);
            zeros.into_iter().collect()
        };

        let mut order = Vec::with_capacity(graph.node_count());

        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut successors: Vec<NodeId> = graph
                .edges
                .get(&node)
                .cloned()
                .unwrap_or_default();
            successors.sort_by_key(|id| id.0);
            for succ in successors {
                if let Some(d) = in_degree.get_mut(&succ) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(succ);
                    }
                }
            }
        }

        if order.len() != graph.node_count() {
            let cycle_nodes: Vec<NodeId> = in_degree
                .iter()
                .filter(|(_, &d)| d > 0)
                .map(|(&id, _)| id)
                .collect();
            return Err(CycleError { cycle_nodes });
        }

        Ok(order)
    }

    /// Process a frame through all nodes in topological order, applying
    /// each node's transform in sequence.
    ///
    /// `processors` maps `NodeId` to a boxed transform function
    /// `fn(Option<FilterFrame>) -> Option<FilterFrame>`.
    pub fn evaluate(
        order: &[NodeId],
        processors: &mut HashMap<NodeId, Box<dyn FnMut(Option<FilterFrame>) -> Option<FilterFrame>>>,
        input: FilterFrame,
    ) -> Option<FilterFrame> {
        let mut current = Some(input);
        for &id in order {
            if let Some(proc) = processors.get_mut(&id) {
                current = proc(current);
            }
        }
        current
    }
}

// ─── GraphValidator ───────────────────────────────────────────────────────────

/// Validates a filter graph for structural correctness.
pub struct GraphValidator;

impl GraphValidator {
    /// Return `true` if the graph has at least one cycle.
    ///
    /// Uses DFS with gray/black node colouring (identical to
    /// `CycleGraph::has_cycle` but operating on the `SimpleGraph` type).
    #[must_use]
    pub fn has_cycle(graph: &SimpleGraph) -> bool {
        let mut cg = CycleGraph::new();
        for (&from, successors) in &graph.edges {
            for &to in successors {
                cg.add_edge(
                    crate::cycle_detect::CycleNodeId(from.0 as usize),
                    crate::cycle_detect::CycleNodeId(to.0 as usize),
                );
            }
            // Ensure isolated nodes are registered
            if successors.is_empty() {
                cg.add_node(crate::cycle_detect::CycleNodeId(from.0 as usize));
            }
        }
        cg.has_cycle()
    }

    /// Return `true` if the graph is a valid DAG (no cycles, at least one node).
    #[must_use]
    pub fn is_valid_dag(graph: &SimpleGraph) -> bool {
        !graph.edges.is_empty() && !Self::has_cycle(graph)
    }

    /// Collect all nodes with in-degree 0 (potential source nodes).
    #[must_use]
    pub fn source_nodes(graph: &SimpleGraph) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = graph.edges.keys().map(|&k| (k, 0)).collect();
        for succs in graph.edges.values() {
            for &s in succs {
                *in_degree.entry(s).or_insert(0) += 1;
            }
        }
        let mut sources: Vec<NodeId> = in_degree
            .into_iter()
            .filter(|(_, d)| *d == 0)
            .map(|(id, _)| id)
            .collect();
        sources.sort_by_key(|id| id.0);
        sources
    }

    /// Collect all nodes with out-degree 0 (potential sink nodes).
    #[must_use]
    pub fn sink_nodes(graph: &SimpleGraph) -> Vec<NodeId> {
        let mut sinks: Vec<NodeId> = graph
            .edges
            .iter()
            .filter(|(_, succs)| succs.is_empty())
            .map(|(&id, _)| id)
            .collect();
        sinks.sort_by_key(|id| id.0);
        sinks
    }
}

// ─── EdgeId ───────────────────────────────────────────────────────────────────

/// Identifies a directed edge by its source and destination node IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeId {
    /// Source node.
    pub from: NodeId,
    /// Destination node.
    pub to: NodeId,
}

impl EdgeId {
    /// Create a new edge ID.
    #[must_use]
    pub fn new(from: NodeId, to: NodeId) -> Self {
        Self { from, to }
    }
}

// ─── GraphEditor ─────────────────────────────────────────────────────────────

/// Supports dynamic modification of a graph without full rebuild.
pub struct GraphEditor;

impl GraphEditor {
    /// Insert `new_node_id` on an existing edge `edge`, splitting it into two.
    ///
    /// Before: `edge.from → edge.to`
    /// After:  `edge.from → new_node_id → edge.to`
    ///
    /// The edge label of the new segments inherits the label of `new_node_id`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the edge does not exist in the graph.
    pub fn insert_node_between(
        graph: &mut SimpleGraph,
        new_node_id: NodeId,
        new_node_label: impl Into<String>,
        edge: EdgeId,
    ) -> GraphResult<()> {
        // Verify edge exists
        let successors = graph
            .edges
            .get(&edge.from)
            .ok_or(GraphError::NodeNotFound(edge.from))?;
        if !successors.contains(&edge.to) {
            return Err(GraphError::ConfigurationError(format!(
                "Edge {:?} → {:?} does not exist",
                edge.from, edge.to
            )));
        }

        // Remove old edge from.to
        if let Some(succs) = graph.edges.get_mut(&edge.from) {
            succs.retain(|&id| id != edge.to);
        }

        // Insert new node
        let label = new_node_label.into();
        graph.labels.insert(new_node_id, label);
        graph.edges.entry(new_node_id).or_default();

        // Add edges: from → new → to
        graph.edges.get_mut(&edge.from).unwrap_or(&mut vec![]).push(new_node_id);
        // Use entry to avoid double borrow
        graph.edges.entry(edge.from).or_default().push(new_node_id);
        // Add new_node → to
        graph.edges.entry(new_node_id).or_default().push(edge.to);

        Ok(())
    }
}

// ─── GraphSerializer ─────────────────────────────────────────────────────────

/// Serializes a filter graph to various text formats.
pub struct GraphSerializer;

impl GraphSerializer {
    /// Produce a valid Graphviz DOT representation of the graph.
    ///
    /// ```text
    /// digraph G {
    ///     0 [label="source"];
    ///     1 [label="filter"];
    ///     2 [label="sink"];
    ///     0 -> 1;
    ///     1 -> 2;
    /// }
    /// ```
    #[must_use]
    pub fn to_dot(graph: &SimpleGraph) -> String {
        let mut out = String::from("digraph G {\n");

        let mut node_ids = graph.node_ids();
        node_ids.sort_by_key(|id| id.0);

        for id in &node_ids {
            let label = graph
                .labels
                .get(id)
                .cloned()
                .unwrap_or_else(|| format!("{}", id.0));
            // Escape quotes in labels
            let escaped = label.replace('"', "\\\"");
            out.push_str(&format!("    {} [label=\"{}\"];\n", id.0, escaped));
        }

        for id in &node_ids {
            if let Some(succs) = graph.edges.get(id) {
                let mut sorted = succs.clone();
                sorted.sort_by_key(|s| s.0);
                for succ in sorted {
                    out.push_str(&format!("    {} -> {};\n", id.0, succ.0));
                }
            }
        }

        out.push_str("}\n");
        out
    }

    /// Produce a simple adjacency-list text representation.
    #[must_use]
    pub fn to_adjacency_list(graph: &SimpleGraph) -> String {
        let mut out = String::new();
        let mut ids = graph.node_ids();
        ids.sort_by_key(|id| id.0);
        for id in ids {
            let label = graph.labels.get(&id).cloned().unwrap_or_else(|| format!("{}", id.0));
            let succs = graph.edges.get(&id).cloned().unwrap_or_default();
            let succ_str: Vec<String> = succs.iter().map(|s| format!("{}", s.0)).collect();
            out.push_str(&format!("{} ({}): {}\n", id.0, label, succ_str.join(", ")));
        }
        out
    }
}

// ─── GraphStatsSnapshot ──────────────────────────────────────────────────────

/// A snapshot of graph structural statistics.
#[derive(Debug, Clone)]
pub struct GraphStatsSnapshot {
    /// Number of nodes.
    pub node_count: usize,
    /// Number of directed edges.
    pub edge_count: usize,
    /// Maximum depth (longest path from any source).
    pub max_depth: usize,
}

/// Measures structural statistics of a graph.
pub struct GraphStats;

impl GraphStats {
    /// Measure node count, edge count, and max depth.
    ///
    /// Max depth is computed via BFS from all source nodes (in-degree 0).
    #[must_use]
    pub fn measure(graph: &SimpleGraph) -> GraphStatsSnapshot {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        let max_depth = Self::compute_max_depth(graph);
        GraphStatsSnapshot { node_count, edge_count, max_depth }
    }

    fn compute_max_depth(graph: &SimpleGraph) -> usize {
        if graph.edges.is_empty() {
            return 0;
        }

        // Build in-degree map
        let mut in_degree: HashMap<NodeId, usize> = graph.edges.keys().map(|&k| (k, 0)).collect();
        for succs in graph.edges.values() {
            for &s in succs {
                *in_degree.entry(s).or_insert(0) += 1;
            }
        }

        // BFS depth propagation
        let mut depth: HashMap<NodeId, usize> = HashMap::new();
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        for &id in &queue {
            depth.insert(id, 0);
        }

        let mut max_d = 0;
        // Use topological BFS
        let mut local_in_degree = in_degree.clone();
        while let Some(node) = queue.pop_front() {
            let cur = *depth.get(&node).unwrap_or(&0);
            if let Some(succs) = graph.edges.get(&node) {
                for &succ in succs {
                    let new_d = cur + 1;
                    let entry = depth.entry(succ).or_insert(0);
                    if new_d > *entry {
                        *entry = new_d;
                        if new_d > max_d {
                            max_d = new_d;
                        }
                    }
                    if let Some(d) = local_in_degree.get_mut(&succ) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        max_d
    }
}

// ─── MultiInputNode ───────────────────────────────────────────────────────────

/// A node with a configurable number of inputs and outputs.
///
/// Used for fan-in topologies where multiple upstream outputs are combined.
pub struct MultiInputNode {
    id: NodeId,
    name: String,
    state: NodeState,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl MultiInputNode {
    /// Create a new multi-input node.
    ///
    /// All ports are typed as `Any` (video or audio).
    #[must_use]
    pub fn new(inputs: u32, outputs: u32, name: impl Into<String>) -> Self {
        let name = name.into();
        let input_ports = (0..inputs)
            .map(|i| InputPort::new(PortId(i), &format!("input_{i}"), PortType::Video))
            .collect();
        let output_ports = (0..outputs)
            .map(|i| OutputPort::new(PortId(i), &format!("output_{i}"), PortType::Video))
            .collect();
        Self {
            id: NodeId(0),
            name,
            state: NodeState::Idle,
            inputs: input_ports,
            outputs: output_ports,
        }
    }

    /// Set the node ID.
    #[must_use]
    pub fn with_id(mut self, id: NodeId) -> Self {
        self.id = id;
        self
    }

    /// Return the number of input ports.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Return the number of output ports.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
}

impl Node for MultiInputNode {
    fn id(&self) -> NodeId { self.id }
    fn name(&self) -> &str { &self.name }
    fn node_type(&self) -> NodeType { NodeType::Filter }
    fn state(&self) -> NodeState { self.state }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] { &self.inputs }
    fn outputs(&self) -> &[OutputPort] { &self.outputs }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        // Pass-through: forward the first available input unchanged
        Ok(input)
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.state = NodeState::Idle;
        Ok(())
    }
}

/// Connect multiple source nodes to a single target node's inputs.
///
/// Returns an error if `sources.len()` exceeds the target's input port count.
pub fn connect_fan_in(
    graph: &mut SimpleGraph,
    sources: &[NodeId],
    target: NodeId,
) -> GraphResult<()> {
    for &src in sources {
        if !graph.edges.contains_key(&src) {
            return Err(GraphError::NodeNotFound(src));
        }
        graph.edges.entry(src).or_default().push(target);
    }
    graph.edges.entry(target).or_default();
    Ok(())
}

// ─── ConditionalRouter ────────────────────────────────────────────────────────

/// A filter node that routes frames to one of two output ports based on a
/// predicate function applied to each incoming frame.
///
/// - Output port 0: condition is `true`
/// - Output port 1: condition is `false`
pub struct ConditionalRouter {
    id: NodeId,
    name: String,
    state: NodeState,
    condition: Box<dyn Fn(&FilterFrame) -> bool + Send + Sync>,
    /// Most recently routed output (0 = true branch, 1 = false branch).
    last_route: Option<u32>,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl ConditionalRouter {
    /// Create a new conditional router with the given predicate.
    ///
    /// The predicate receives a shared reference to the frame and returns
    /// `true` to route to output 0 or `false` to route to output 1.
    pub fn new(condition: Box<dyn Fn(&FilterFrame) -> bool + Send + Sync>) -> Self {
        Self {
            id: NodeId(0),
            name: "conditional_router".to_string(),
            state: NodeState::Idle,
            condition,
            last_route: None,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Video)],
            outputs: vec![
                OutputPort::new(PortId(0), "true_output", PortType::Video),
                OutputPort::new(PortId(1), "false_output", PortType::Video),
            ],
        }
    }

    /// Set the node ID.
    #[must_use]
    pub fn with_id(mut self, id: NodeId) -> Self {
        self.id = id;
        self
    }

    /// Return which output port the last frame was routed to.
    #[must_use]
    pub fn last_route(&self) -> Option<u32> {
        self.last_route
    }

    /// Evaluate the condition and return the output port index (0 or 1).
    #[must_use]
    pub fn route(&self, frame: &FilterFrame) -> u32 {
        if (self.condition)(frame) { 0 } else { 1 }
    }
}

impl Node for ConditionalRouter {
    fn id(&self) -> NodeId { self.id }
    fn name(&self) -> &str { &self.name }
    fn node_type(&self) -> NodeType { NodeType::Filter }
    fn state(&self) -> NodeState { self.state }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] { &self.inputs }
    fn outputs(&self) -> &[OutputPort] { &self.outputs }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(frame) => {
                self.last_route = Some(self.route(&frame));
                Ok(Some(frame))
            }
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.last_route = None;
        self.state = NodeState::Idle;
        Ok(())
    }
}

// ─── GainNode ─────────────────────────────────────────────────────────────────

/// An audio gain node that scales sample amplitude by a dB value.
///
/// `gain_db` of 0 dB = unity gain; +6 dB ≈ 2×; −6 dB ≈ 0.5×.
pub struct GainNode {
    id: NodeId,
    name: String,
    state: NodeState,
    /// Gain in dB.
    gain_db: f32,
    /// Precomputed linear multiplier: `10^(gain_db / 20)`.
    linear_gain: f32,
    inputs: Vec<InputPort>,
    outputs: Vec<OutputPort>,
}

impl GainNode {
    /// Create a new gain node with the specified dB gain.
    #[must_use]
    pub fn new(gain_db: f32) -> Self {
        let linear_gain = 10.0_f32.powf(gain_db / 20.0);
        let audio_fmt = PortFormat::Audio(AudioPortFormat::any());
        Self {
            id: NodeId(0),
            name: "gain".to_string(),
            state: NodeState::Idle,
            gain_db,
            linear_gain,
            inputs: vec![InputPort::new(PortId(0), "input", PortType::Audio)
                .with_format(audio_fmt.clone())],
            outputs: vec![OutputPort::new(PortId(0), "output", PortType::Audio)
                .with_format(audio_fmt)],
        }
    }

    /// Set the node ID.
    #[must_use]
    pub fn with_id(mut self, id: NodeId) -> Self {
        self.id = id;
        self
    }

    /// Return the gain in dB.
    #[must_use]
    pub fn gain_db(&self) -> f32 {
        self.gain_db
    }

    /// Return the linear gain multiplier.
    #[must_use]
    pub fn linear_gain(&self) -> f32 {
        self.linear_gain
    }

    /// Apply gain to an audio frame, returning a new frame.
    #[must_use]
    pub fn process_frame(&self, frame: &AudioFrame) -> AudioFrame {
        let channels = frame.channels.count();
        let sample_count = frame.sample_count();

        if sample_count == 0 || channels == 0 {
            return frame.clone();
        }

        match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                let bytes_per_sample = frame.format.bytes_per_sample();
                if bytes_per_sample == 0 {
                    return frame.clone();
                }

                let total_samples = data.len() / bytes_per_sample;
                let mut out = Vec::with_capacity(data.len());

                for i in 0..total_samples {
                    let offset = i * bytes_per_sample;
                    if offset + bytes_per_sample > data.len() {
                        break;
                    }
                    let scaled = Self::scale_sample(
                        &data[offset..offset + bytes_per_sample],
                        frame.format,
                        self.linear_gain,
                    );
                    out.extend_from_slice(&scaled);
                }

                let mut output_frame = AudioFrame::new(frame.format, frame.sample_rate, frame.channels.clone());
                output_frame.samples = AudioBuffer::Interleaved(bytes::Bytes::from(out));
                output_frame
            }
            AudioBuffer::Planar(planes) => {
                let bytes_per_sample = frame.format.bytes_per_sample();
                if bytes_per_sample == 0 {
                    return frame.clone();
                }

                let scaled_planes: Vec<bytes::Bytes> = planes
                    .iter()
                    .map(|plane| {
                        let total = plane.len() / bytes_per_sample;
                        let mut out = Vec::with_capacity(plane.len());
                        for i in 0..total {
                            let offset = i * bytes_per_sample;
                            if offset + bytes_per_sample > plane.len() {
                                break;
                            }
                            let scaled = Self::scale_sample(
                                &plane[offset..offset + bytes_per_sample],
                                frame.format,
                                self.linear_gain,
                            );
                            out.extend_from_slice(&scaled);
                        }
                        bytes::Bytes::from(out)
                    })
                    .collect();

                let mut output_frame = AudioFrame::new(frame.format, frame.sample_rate, frame.channels.clone());
                output_frame.samples = AudioBuffer::Planar(scaled_planes);
                output_frame
            }
        }
    }

    /// Scale a single sample byte slice by `linear_gain`.
    fn scale_sample(bytes: &[u8], format: SampleFormat, gain: f32) -> Vec<u8> {
        match format {
            SampleFormat::F32 => {
                if bytes.len() < 4 {
                    return bytes.to_vec();
                }
                let v = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let scaled = (v * gain).clamp(-1.0, 1.0);
                scaled.to_le_bytes().to_vec()
            }
            SampleFormat::S16 => {
                if bytes.len() < 2 {
                    return bytes.to_vec();
                }
                let v = i16::from_le_bytes([bytes[0], bytes[1]]);
                let scaled = ((v as f32) * gain).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                scaled.to_le_bytes().to_vec()
            }
            SampleFormat::S32 => {
                if bytes.len() < 4 {
                    return bytes.to_vec();
                }
                let v = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let scaled = ((v as f64) * gain as f64).clamp(i32::MIN as f64, i32::MAX as f64) as i32;
                scaled.to_le_bytes().to_vec()
            }
            SampleFormat::U8 => {
                if bytes.is_empty() {
                    return bytes.to_vec();
                }
                // Map [0,255] → [-1,1], scale, map back
                let v = (bytes[0] as f32 - 128.0) / 128.0;
                let scaled = (v * gain).clamp(-1.0, 1.0);
                let out = ((scaled * 128.0) + 128.0).clamp(0.0, 255.0) as u8;
                vec![out]
            }
            _ => bytes.to_vec(),
        }
    }
}

impl Node for GainNode {
    fn id(&self) -> NodeId { self.id }
    fn name(&self) -> &str { &self.name }
    fn node_type(&self) -> NodeType { NodeType::Filter }
    fn state(&self) -> NodeState { self.state }

    fn set_state(&mut self, state: NodeState) -> GraphResult<()> {
        self.state = state;
        Ok(())
    }

    fn inputs(&self) -> &[InputPort] { &self.inputs }
    fn outputs(&self) -> &[OutputPort] { &self.outputs }

    fn process(&mut self, input: Option<FilterFrame>) -> GraphResult<Option<FilterFrame>> {
        match input {
            Some(FilterFrame::Audio(audio_frame)) => {
                let output = self.process_frame(&audio_frame);
                Ok(Some(FilterFrame::Audio(output)))
            }
            Some(other) => Ok(Some(other)), // pass video frames through unchanged
            None => Ok(None),
        }
    }

    fn reset(&mut self) -> GraphResult<()> {
        self.state = NodeState::Idle;
        Ok(())
    }
}

// ─── CropNode ────────────────────────────────────────────────────────────────

/// A simple crop-region descriptor for use with the evaluator layer.
///
/// Note: the full `CropFilter` in `filters::video::crop` provides a complete
/// `Node` implementation. This struct is a thin wrapper for the evaluator API.
#[derive(Debug, Clone, Copy)]
pub struct CropRegion {
    /// Left pixel offset.
    pub x: u32,
    /// Top pixel offset.
    pub y: u32,
    /// Output width.
    pub w: u32,
    /// Output height.
    pub h: u32,
}

impl CropRegion {
    /// Create a new crop region.
    #[must_use]
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Crop a raw RGBA/RGB pixel buffer (packed, row-major).
    ///
    /// `src_w` is the original image width in pixels.
    /// `channels` is bytes per pixel (e.g. 4 for RGBA).
    #[must_use]
    pub fn crop_buffer(&self, data: &[u8], src_w: u32, channels: u32) -> Vec<u8> {
        let stride = (src_w * channels) as usize;
        let row_bytes = (self.w * channels) as usize;
        let mut out = Vec::with_capacity(row_bytes * self.h as usize);

        for row in self.y..self.y + self.h {
            let row_start = row as usize * stride + (self.x * channels) as usize;
            let row_end = row_start + row_bytes;
            if row_end <= data.len() {
                out.extend_from_slice(&data[row_start..row_end]);
            }
        }

        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chain() -> SimpleGraph {
        let mut g = SimpleGraph::new();
        g.add_node(NodeId(0), "A");
        g.add_node(NodeId(1), "B");
        g.add_node(NodeId(2), "C");
        g.add_edge(NodeId(0), NodeId(1));
        g.add_edge(NodeId(1), NodeId(2));
        g
    }

    // ── GraphEvaluator ──

    #[test]
    fn test_topological_sort_linear_chain() {
        let g = make_chain();
        let order = GraphEvaluator::topological_sort(&g).expect("sort should succeed");
        assert_eq!(order, vec![NodeId(0), NodeId(1), NodeId(2)]);
    }

    #[test]
    fn test_topological_sort_cycle_returns_error() {
        let mut g = SimpleGraph::new();
        g.add_edge(NodeId(0), NodeId(0)); // self-loop
        let result = GraphEvaluator::topological_sort(&g);
        assert!(result.is_err());
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut g = SimpleGraph::new();
        g.add_edge(NodeId(0), NodeId(1));
        g.add_edge(NodeId(0), NodeId(2));
        g.add_edge(NodeId(1), NodeId(3));
        g.add_edge(NodeId(2), NodeId(3));
        let order = GraphEvaluator::topological_sort(&g).expect("sort should succeed");
        assert_eq!(order[0], NodeId(0));
        assert_eq!(*order.last().expect("last"), NodeId(3));
    }

    // ── GraphValidator ──

    #[test]
    fn test_has_cycle_self_loop() {
        let mut g = SimpleGraph::new();
        g.add_edge(NodeId(5), NodeId(5));
        assert!(GraphValidator::has_cycle(&g));
    }

    #[test]
    fn test_has_cycle_dag() {
        let g = make_chain();
        assert!(!GraphValidator::has_cycle(&g));
    }

    #[test]
    fn test_has_cycle_3_node_cycle() {
        let mut g = SimpleGraph::new();
        g.add_edge(NodeId(0), NodeId(1));
        g.add_edge(NodeId(1), NodeId(2));
        g.add_edge(NodeId(2), NodeId(0));
        assert!(GraphValidator::has_cycle(&g));
    }

    // ── GraphSerializer ──

    #[test]
    fn test_to_dot_contains_digraph() {
        let g = make_chain();
        let dot = GraphSerializer::to_dot(&g);
        assert!(dot.contains("digraph"), "DOT output must contain 'digraph'");
    }

    #[test]
    fn test_to_dot_contains_nodes() {
        let g = make_chain();
        let dot = GraphSerializer::to_dot(&g);
        assert!(dot.contains("label=\"A\""));
        assert!(dot.contains("label=\"B\""));
        assert!(dot.contains("label=\"C\""));
    }

    #[test]
    fn test_to_dot_contains_edges() {
        let g = make_chain();
        let dot = GraphSerializer::to_dot(&g);
        assert!(dot.contains("->"));
    }

    // ── GraphStats ──

    #[test]
    fn test_stats_measure_chain() {
        let g = make_chain();
        let snap = GraphStats::measure(&g);
        assert_eq!(snap.node_count, 3);
        assert_eq!(snap.edge_count, 2);
        assert_eq!(snap.max_depth, 2);
    }

    #[test]
    fn test_stats_measure_empty() {
        let g = SimpleGraph::new();
        let snap = GraphStats::measure(&g);
        assert_eq!(snap.node_count, 0);
        assert_eq!(snap.edge_count, 0);
        assert_eq!(snap.max_depth, 0);
    }

    // ── MultiInputNode ──

    #[test]
    fn test_multi_input_node_creation() {
        let n = MultiInputNode::new(3, 1, "fan_in");
        assert_eq!(n.input_count(), 3);
        assert_eq!(n.output_count(), 1);
        assert_eq!(n.name(), "fan_in");
    }

    #[test]
    fn test_connect_fan_in() {
        let mut g = SimpleGraph::new();
        g.add_node(NodeId(0), "src_a");
        g.add_node(NodeId(1), "src_b");
        g.add_node(NodeId(2), "mixer");
        let result = connect_fan_in(&mut g, &[NodeId(0), NodeId(1)], NodeId(2));
        assert!(result.is_ok());
        assert!(g.edges[&NodeId(0)].contains(&NodeId(2)));
        assert!(g.edges[&NodeId(1)].contains(&NodeId(2)));
    }

    // ── ConditionalRouter ──

    #[test]
    fn test_conditional_router_true_branch() {
        let router = ConditionalRouter::new(Box::new(|frame| frame.is_video()));
        let video = VideoFrame::new(PixelFormat::Yuv420p, 100, 100);
        let frame = FilterFrame::Video(video);
        assert_eq!(router.route(&frame), 0); // true → port 0
    }

    #[test]
    fn test_conditional_router_false_branch() {
        let router = ConditionalRouter::new(Box::new(|frame| frame.is_video()));
        let audio = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Stereo);
        let frame = FilterFrame::Audio(audio);
        assert_eq!(router.route(&frame), 1); // false → port 1
    }

    // ── GainNode ──

    #[test]
    fn test_gain_node_unity_gain() {
        let g = GainNode::new(0.0);
        assert!((g.linear_gain() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_gain_node_6db() {
        let g = GainNode::new(6.0206);
        // 10^(6.0206/20) ≈ 2.0
        assert!((g.linear_gain() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_gain_node_process_f32_sample() {
        use bytes::BytesMut;

        let gain = GainNode::new(6.0206); // ≈ 2×
        let mut frame = AudioFrame::new(SampleFormat::F32, 48000, ChannelLayout::Mono);
        let mut buf = BytesMut::new();
        buf.extend_from_slice(&0.25f32.to_le_bytes());
        frame.samples = AudioBuffer::Interleaved(buf.freeze());

        let out = gain.process_frame(&frame);
        if let AudioBuffer::Interleaved(data) = &out.samples {
            let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            // 0.25 × 2 = 0.5
            assert!((v - 0.5).abs() < 0.05, "expected ~0.5, got {v}");
        }
    }

    // ── CropRegion ──

    #[test]
    fn test_crop_region_extracts_subregion() {
        // 4×4 RGBA image, each pixel = (row*10, col*10, 0, 255)
        let mut data = vec![0u8; 4 * 4 * 4];
        for row in 0..4u32 {
            for col in 0..4u32 {
                let idx = ((row * 4 + col) * 4) as usize;
                data[idx] = (row * 10) as u8;
                data[idx + 1] = (col * 10) as u8;
                data[idx + 2] = 0;
                data[idx + 3] = 255;
            }
        }

        let crop = CropRegion::new(1, 1, 2, 2);
        let result = crop.crop_buffer(&data, 4, 4);
        assert_eq!(result.len(), 2 * 2 * 4);
        // First pixel of cropped region: row=1, col=1 → (10, 10, 0, 255)
        assert_eq!(result[0], 10);
        assert_eq!(result[1], 10);
    }

    // ── EdgeId / GraphEditor ──

    #[test]
    fn test_graph_editor_insert_node_between() {
        let mut g = SimpleGraph::new();
        g.add_node(NodeId(0), "A");
        g.add_node(NodeId(2), "C");
        g.add_edge(NodeId(0), NodeId(2));

        let edge = EdgeId::new(NodeId(0), NodeId(2));
        // Insert B between A and C
        let result = GraphEditor::insert_node_between(&mut g, NodeId(1), "B", edge);
        assert!(result.is_ok(), "insert_node_between should succeed: {result:?}");

        // A should now have B as successor (not C directly)
        assert!(g.edges[&NodeId(0)].contains(&NodeId(1)) || !g.edges[&NodeId(0)].contains(&NodeId(2)));
        // B should have C as successor
        assert!(g.edges[&NodeId(1)].contains(&NodeId(2)));
    }

    #[test]
    fn test_graph_editor_nonexistent_edge_returns_error() {
        let mut g = SimpleGraph::new();
        g.add_node(NodeId(0), "A");
        g.add_node(NodeId(1), "B");
        // No edge 0→1 yet
        let edge = EdgeId::new(NodeId(0), NodeId(1));
        let result = GraphEditor::insert_node_between(&mut g, NodeId(2), "C", edge);
        assert!(result.is_err());
    }
}
