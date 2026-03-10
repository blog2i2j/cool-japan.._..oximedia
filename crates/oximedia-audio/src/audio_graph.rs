//! Node-based audio signal routing graph.
//!
//! This module provides a directed-acyclic-graph (DAG) approach to audio
//! processing where individual [`AudioNode`] instances are connected together
//! and the [`AudioGraph`] evaluates them in topological order.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::audio_graph::{AudioGraph, AudioNode, NodeKind, ConnectionWeight};
//!
//! let mut graph = AudioGraph::new(48_000.0);
//! let src = graph.add_node(AudioNode::new("source", NodeKind::Source));
//! let gain = graph.add_node(AudioNode::new("gain", NodeKind::Gain(0.5)));
//! let sink = graph.add_node(AudioNode::new("sink", NodeKind::Sink));
//!
//! graph.connect(src, gain, ConnectionWeight::Unity);
//! graph.connect(gain, sink, ConnectionWeight::Unity);
//!
//! let topo = graph.topological_order();
//! assert_eq!(topo.len(), 3);
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Unique identifier for a node inside an [`AudioGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// The kind of processing a node performs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeKind {
    /// Produces audio (e.g. oscillator, file reader).
    Source,
    /// Consumes audio (e.g. output device).
    Sink,
    /// Applies gain to the signal.
    Gain(f32),
    /// Mixes multiple inputs together.
    Mixer,
    /// Passes audio through unchanged.
    Passthrough,
    /// Inverts the phase of the signal.
    Invert,
}

/// Weight applied to a connection between two nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConnectionWeight {
    /// No attenuation (1.0).
    Unity,
    /// Fixed gain multiplier.
    Fixed(f32),
    /// Muted connection (0.0).
    Muted,
}

impl ConnectionWeight {
    /// Return the numeric multiplier.
    pub fn value(&self) -> f32 {
        match self {
            Self::Unity => 1.0,
            Self::Fixed(g) => *g,
            Self::Muted => 0.0,
        }
    }
}

/// A single node in the audio graph.
#[derive(Debug, Clone)]
pub struct AudioNode {
    /// Human-readable label.
    pub label: String,
    /// The kind of processing.
    pub kind: NodeKind,
    /// Whether the node is currently bypassed.
    pub bypassed: bool,
}

impl AudioNode {
    /// Create a new node with the given label and kind.
    pub fn new(label: &str, kind: NodeKind) -> Self {
        Self {
            label: label.to_string(),
            kind,
            bypassed: false,
        }
    }

    /// Set the bypass flag.
    pub fn set_bypass(&mut self, bypass: bool) {
        self.bypassed = bypass;
    }

    /// Process a single sample according to this node's kind.
    pub fn process_sample(&self, input: f32) -> f32 {
        if self.bypassed {
            return input;
        }
        match self.kind {
            NodeKind::Gain(g) => input * g,
            NodeKind::Invert => -input,
            NodeKind::Source | NodeKind::Sink | NodeKind::Mixer | NodeKind::Passthrough => input,
        }
    }
}

/// A directed edge between two nodes.
#[derive(Debug, Clone)]
struct Edge {
    from: NodeId,
    to: NodeId,
    weight: ConnectionWeight,
}

/// A directed acyclic graph of audio processing nodes.
///
/// Nodes are evaluated in topological order so that every node's inputs have
/// been computed before its own processing runs.
#[derive(Debug)]
pub struct AudioGraph {
    nodes: Vec<AudioNode>,
    edges: Vec<Edge>,
    sample_rate: f32,
}

impl AudioGraph {
    /// Create an empty audio graph with the given sample rate.
    pub fn new(sample_rate: f32) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            sample_rate,
        }
    }

    /// Return the sample rate.
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Add a node and return its [`NodeId`].
    pub fn add_node(&mut self, node: AudioNode) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Connect the output of `from` to the input of `to` with a given weight.
    pub fn connect(&mut self, from: NodeId, to: NodeId, weight: ConnectionWeight) {
        self.edges.push(Edge { from, to, weight });
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get a reference to a node by id.
    pub fn node(&self, id: NodeId) -> Option<&AudioNode> {
        self.nodes.get(id.0)
    }

    /// Get a mutable reference to a node by id.
    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut AudioNode> {
        self.nodes.get_mut(id.0)
    }

    /// Compute in-degree for each node.
    fn in_degrees(&self) -> Vec<usize> {
        let mut deg = vec![0usize; self.nodes.len()];
        for e in &self.edges {
            deg[e.to.0] += 1;
        }
        deg
    }

    /// Return a topological ordering of node indices (Kahn's algorithm).
    ///
    /// Returns an empty vec if the graph contains a cycle.
    pub fn topological_order(&self) -> Vec<NodeId> {
        let n = self.nodes.len();
        let mut in_deg = self.in_degrees();
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, d) in in_deg.iter().enumerate() {
            if *d == 0 {
                queue.push_back(i);
            }
        }
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(NodeId(u));
            for e in &self.edges {
                if e.from.0 == u {
                    in_deg[e.to.0] -= 1;
                    if in_deg[e.to.0] == 0 {
                        queue.push_back(e.to.0);
                    }
                }
            }
        }
        if order.len() == n {
            order
        } else {
            Vec::new() // cycle detected
        }
    }

    /// Process a single sample through the entire graph.
    ///
    /// Each node receives the sum of its weighted inputs.
    /// Returns the output of the last node in topological order, or 0.0
    /// if the graph is empty or has a cycle.
    pub fn process_sample(&self, input: f32) -> f32 {
        let order = self.topological_order();
        if order.is_empty() {
            return 0.0;
        }
        let mut values = vec![0.0_f32; self.nodes.len()];
        // seed source nodes with input
        for id in &order {
            if self.nodes[id.0].kind == NodeKind::Source {
                values[id.0] = input;
            }
        }
        for id in &order {
            // accumulate weighted inputs from predecessors
            let mut acc = values[id.0];
            for e in &self.edges {
                if e.to == *id {
                    acc += values[e.from.0] * e.weight.value();
                }
            }
            values[id.0] = self.nodes[id.0].process_sample(acc);
        }
        // return last node value
        order.last().map_or(0.0, |id| values[id.0])
    }

    /// Remove all edges from the graph.
    pub fn clear_edges(&mut self) {
        self.edges.clear();
    }

    /// Return all successor ids for a given node.
    pub fn successors(&self, id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.from == id)
            .map(|e| e.to)
            .collect()
    }

    /// Return all predecessor ids for a given node.
    pub fn predecessors(&self, id: NodeId) -> Vec<NodeId> {
        self.edges
            .iter()
            .filter(|e| e.to == id)
            .map(|e| e.from)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = AudioNode::new("test", NodeKind::Gain(0.5));
        assert_eq!(node.label, "test");
        assert!(!node.bypassed);
    }

    #[test]
    fn test_node_process_gain() {
        let node = AudioNode::new("g", NodeKind::Gain(0.5));
        let out = node.process_sample(1.0);
        assert!((out - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_node_process_invert() {
        let node = AudioNode::new("inv", NodeKind::Invert);
        assert!((node.process_sample(0.7) - (-0.7)).abs() < 1e-6);
    }

    #[test]
    fn test_node_bypass() {
        let mut node = AudioNode::new("g", NodeKind::Gain(0.5));
        node.set_bypass(true);
        assert!((node.process_sample(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_connection_weight_values() {
        assert!((ConnectionWeight::Unity.value() - 1.0).abs() < 1e-6);
        assert!((ConnectionWeight::Fixed(0.3).value() - 0.3).abs() < 1e-6);
        assert!((ConnectionWeight::Muted.value()).abs() < 1e-6);
    }

    #[test]
    fn test_graph_add_nodes() {
        let mut g = AudioGraph::new(44100.0);
        let a = g.add_node(AudioNode::new("a", NodeKind::Source));
        let b = g.add_node(AudioNode::new("b", NodeKind::Sink));
        assert_eq!(g.node_count(), 2);
        assert_eq!(a, NodeId(0));
        assert_eq!(b, NodeId(1));
    }

    #[test]
    fn test_graph_connect() {
        let mut g = AudioGraph::new(48000.0);
        let a = g.add_node(AudioNode::new("a", NodeKind::Source));
        let b = g.add_node(AudioNode::new("b", NodeKind::Sink));
        g.connect(a, b, ConnectionWeight::Unity);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_topological_order_linear() {
        let mut g = AudioGraph::new(48000.0);
        let a = g.add_node(AudioNode::new("src", NodeKind::Source));
        let b = g.add_node(AudioNode::new("gain", NodeKind::Gain(0.5)));
        let c = g.add_node(AudioNode::new("sink", NodeKind::Sink));
        g.connect(a, b, ConnectionWeight::Unity);
        g.connect(b, c, ConnectionWeight::Unity);
        let order = g.topological_order();
        assert_eq!(order.len(), 3);
        assert_eq!(order[0], a);
        assert_eq!(order[1], b);
        assert_eq!(order[2], c);
    }

    #[test]
    fn test_process_sample_linear_chain() {
        let mut g = AudioGraph::new(48000.0);
        let src = g.add_node(AudioNode::new("src", NodeKind::Source));
        let gain = g.add_node(AudioNode::new("gain", NodeKind::Gain(0.5)));
        let sink = g.add_node(AudioNode::new("sink", NodeKind::Sink));
        g.connect(src, gain, ConnectionWeight::Unity);
        g.connect(gain, sink, ConnectionWeight::Unity);
        let out = g.process_sample(2.0);
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_process_sample_with_muted_edge() {
        let mut g = AudioGraph::new(48000.0);
        let src = g.add_node(AudioNode::new("src", NodeKind::Source));
        let sink = g.add_node(AudioNode::new("sink", NodeKind::Sink));
        g.connect(src, sink, ConnectionWeight::Muted);
        let out = g.process_sample(1.0);
        assert!(out.abs() < 1e-6);
    }

    #[test]
    fn test_successors_and_predecessors() {
        let mut g = AudioGraph::new(48000.0);
        let a = g.add_node(AudioNode::new("a", NodeKind::Source));
        let b = g.add_node(AudioNode::new("b", NodeKind::Passthrough));
        let c = g.add_node(AudioNode::new("c", NodeKind::Sink));
        g.connect(a, b, ConnectionWeight::Unity);
        g.connect(b, c, ConnectionWeight::Unity);
        assert_eq!(g.successors(a), vec![b]);
        assert_eq!(g.predecessors(c), vec![b]);
    }

    #[test]
    fn test_clear_edges() {
        let mut g = AudioGraph::new(48000.0);
        let a = g.add_node(AudioNode::new("a", NodeKind::Source));
        let b = g.add_node(AudioNode::new("b", NodeKind::Sink));
        g.connect(a, b, ConnectionWeight::Unity);
        assert_eq!(g.edge_count(), 1);
        g.clear_edges();
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_sample_rate() {
        let g = AudioGraph::new(96000.0);
        assert!((g.sample_rate() - 96000.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_lookup() {
        let mut g = AudioGraph::new(48000.0);
        let id = g.add_node(AudioNode::new("lookup", NodeKind::Mixer));
        let node = g.node(id).expect("should succeed");
        assert_eq!(node.label, "lookup");
    }

    #[test]
    fn test_empty_graph_returns_zero() {
        let g = AudioGraph::new(48000.0);
        assert!((g.process_sample(5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fixed_weight_connection() {
        let mut g = AudioGraph::new(48000.0);
        let src = g.add_node(AudioNode::new("src", NodeKind::Source));
        let sink = g.add_node(AudioNode::new("sink", NodeKind::Sink));
        g.connect(src, sink, ConnectionWeight::Fixed(0.25));
        let out = g.process_sample(4.0);
        assert!((out - 1.0).abs() < 1e-6);
    }
}
