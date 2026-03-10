#![allow(dead_code)]
//! Audio processing graph builder for Python bindings.
//!
//! Provides a directed acyclic graph (DAG) of audio processing nodes
//! that Python callers can construct, validate, and query before execution.

use std::collections::HashMap;

/// Unique identifier for a node in the audio graph.
pub type NodeId = u32;

/// Audio channel layout descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelLayout {
    /// Single channel.
    Mono,
    /// Two-channel stereo.
    Stereo,
    /// 5.1 surround.
    Surround51,
    /// 7.1 surround.
    Surround71,
}

impl ChannelLayout {
    /// Number of channels in this layout.
    pub fn channel_count(self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }
}

/// Audio sample format used at node boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioSampleFmt {
    /// Signed 16-bit integer.
    S16,
    /// Signed 32-bit integer.
    S32,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
}

impl AudioSampleFmt {
    /// Bytes per sample for this format.
    pub fn bytes_per_sample(self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Kind of processing a graph node performs.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// Audio source (file reader, generator, etc.).
    Source {
        /// Label for the source.
        label: String,
    },
    /// Gain adjustment node.
    Gain {
        /// Linear gain multiplier.
        value: f64,
    },
    /// Equalization filter.
    Eq {
        /// Center frequency in Hz.
        frequency_hz: f64,
        /// Gain in dB.
        gain_db: f64,
        /// Q factor.
        q: f64,
    },
    /// Mix/merge multiple inputs.
    Mixer {
        /// Number of expected inputs.
        input_count: u32,
    },
    /// Output / sink node.
    Sink {
        /// Label for the sink.
        label: String,
    },
    /// Sample rate conversion.
    Resample {
        /// Target sample rate.
        target_rate: u32,
    },
    /// Channel layout conversion.
    ChannelMap {
        /// Target channel layout.
        target: ChannelLayout,
    },
}

/// A single node in the audio processing graph.
#[derive(Debug, Clone)]
pub struct AudioNode {
    /// Unique node identifier.
    pub id: NodeId,
    /// Processing kind.
    pub kind: NodeKind,
    /// Sample format at this node's output.
    pub sample_format: AudioSampleFmt,
    /// Sample rate at this node's output.
    pub sample_rate: u32,
    /// Channel layout at this node's output.
    pub channels: ChannelLayout,
}

/// A directed edge between two nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioEdge {
    /// Source node identifier.
    pub from: NodeId,
    /// Destination node identifier.
    pub to: NodeId,
}

/// Validation result for the audio graph.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphValidation {
    /// Graph is valid.
    Ok,
    /// Graph has issues.
    Errors(Vec<String>),
}

/// An audio processing graph.
#[derive(Debug, Default)]
pub struct AudioGraph {
    /// All nodes keyed by ID.
    nodes: HashMap<NodeId, AudioNode>,
    /// All edges.
    edges: Vec<AudioEdge>,
    /// Next available node ID.
    next_id: NodeId,
}

impl AudioGraph {
    /// Create an empty audio graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph and return its ID.
    pub fn add_node(
        &mut self,
        kind: NodeKind,
        sample_format: AudioSampleFmt,
        sample_rate: u32,
        channels: ChannelLayout,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            AudioNode {
                id,
                kind,
                sample_format,
                sample_rate,
                channels,
            },
        );
        id
    }

    /// Connect two nodes with a directed edge.
    pub fn connect(&mut self, from: NodeId, to: NodeId) -> Result<(), AudioGraphError> {
        if !self.nodes.contains_key(&from) {
            return Err(AudioGraphError::NodeNotFound(from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(AudioGraphError::NodeNotFound(to));
        }
        if from == to {
            return Err(AudioGraphError::SelfLoop(from));
        }
        let edge = AudioEdge { from, to };
        if self.edges.contains(&edge) {
            return Err(AudioGraphError::DuplicateEdge(from, to));
        }
        self.edges.push(edge);
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Retrieve a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&AudioNode> {
        self.nodes.get(&id)
    }

    /// Remove a node and all its edges.
    pub fn remove_node(&mut self, id: NodeId) -> Result<(), AudioGraphError> {
        if self.nodes.remove(&id).is_none() {
            return Err(AudioGraphError::NodeNotFound(id));
        }
        self.edges.retain(|e| e.from != id && e.to != id);
        Ok(())
    }

    /// List IDs of all source nodes (no incoming edges).
    pub fn source_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .keys()
            .copied()
            .filter(|id| !self.edges.iter().any(|e| e.to == *id))
            .collect()
    }

    /// List IDs of all sink nodes (no outgoing edges).
    pub fn sink_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .keys()
            .copied()
            .filter(|id| !self.edges.iter().any(|e| e.from == *id))
            .collect()
    }

    /// Validate the graph for common issues.
    pub fn validate(&self) -> GraphValidation {
        let mut errors = Vec::new();
        if self.nodes.is_empty() {
            errors.push("graph has no nodes".into());
        }
        if self.source_nodes().is_empty() && !self.nodes.is_empty() {
            errors.push("graph has no source nodes (possible cycle)".into());
        }
        // Check for disconnected nodes (no edges at all)
        for &id in self.nodes.keys() {
            let has_edge = self.edges.iter().any(|e| e.from == id || e.to == id);
            if !has_edge && self.nodes.len() > 1 {
                errors.push(format!("node {id} is disconnected"));
            }
        }
        if errors.is_empty() {
            GraphValidation::Ok
        } else {
            GraphValidation::Errors(errors)
        }
    }

    /// Compute a simple topological ordering of node IDs.
    pub fn topological_order(&self) -> Result<Vec<NodeId>, AudioGraphError> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
        }
        for edge in &self.edges {
            *in_degree.entry(edge.to).or_insert(0) += 1;
        }
        let mut queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort();
        let mut order = Vec::new();
        while let Some(node) = queue.pop() {
            order.push(node);
            for edge in &self.edges {
                if edge.from == node {
                    if let Some(d) = in_degree.get_mut(&edge.to) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push(edge.to);
                            queue.sort();
                        }
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(AudioGraphError::CycleDetected);
        }
        Ok(order)
    }
}

/// Errors that can occur during audio graph operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AudioGraphError {
    /// Node ID not found.
    NodeNotFound(NodeId),
    /// Self-loop detected.
    SelfLoop(NodeId),
    /// Duplicate edge.
    DuplicateEdge(NodeId, NodeId),
    /// Cycle detected during topological sort.
    CycleDetected,
}

impl std::fmt::Display for AudioGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeNotFound(id) => write!(f, "node {id} not found"),
            Self::SelfLoop(id) => write!(f, "self-loop on node {id}"),
            Self::DuplicateEdge(a, b) => write!(f, "duplicate edge {a} -> {b}"),
            Self::CycleDetected => write!(f, "cycle detected in graph"),
        }
    }
}

impl std::error::Error for AudioGraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_layout_count() {
        assert_eq!(ChannelLayout::Mono.channel_count(), 1);
        assert_eq!(ChannelLayout::Stereo.channel_count(), 2);
        assert_eq!(ChannelLayout::Surround51.channel_count(), 6);
        assert_eq!(ChannelLayout::Surround71.channel_count(), 8);
    }

    #[test]
    fn test_sample_fmt_bytes() {
        assert_eq!(AudioSampleFmt::S16.bytes_per_sample(), 2);
        assert_eq!(AudioSampleFmt::S32.bytes_per_sample(), 4);
        assert_eq!(AudioSampleFmt::F32.bytes_per_sample(), 4);
        assert_eq!(AudioSampleFmt::F64.bytes_per_sample(), 8);
    }

    #[test]
    fn test_graph_add_nodes() {
        let mut g = AudioGraph::new();
        let src = g.add_node(
            NodeKind::Source { label: "mic".into() },
            AudioSampleFmt::F32,
            48000,
            ChannelLayout::Mono,
        );
        let sink = g.add_node(
            NodeKind::Sink { label: "out".into() },
            AudioSampleFmt::F32,
            48000,
            ChannelLayout::Mono,
        );
        assert_eq!(g.node_count(), 2);
        assert_ne!(src, sink);
    }

    #[test]
    fn test_graph_connect() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "a".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Stereo);
        let b = g.add_node(NodeKind::Sink { label: "b".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Stereo);
        assert!(g.connect(a, b).is_ok());
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_graph_self_loop() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "a".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let err = g.connect(a, a).unwrap_err();
        assert!(matches!(err, AudioGraphError::SelfLoop(_)));
    }

    #[test]
    fn test_graph_duplicate_edge() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "a".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let b = g.add_node(NodeKind::Sink { label: "b".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        g.connect(a, b).expect("connect should succeed");
        let err = g.connect(a, b).unwrap_err();
        assert!(matches!(err, AudioGraphError::DuplicateEdge(_, _)));
    }

    #[test]
    fn test_graph_remove_node() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "a".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let b = g.add_node(NodeKind::Gain { value: 0.5 }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        g.connect(a, b).expect("connect should succeed");
        g.remove_node(b).expect("remove_node should succeed");
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_graph_source_sink_nodes() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "in".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let b = g.add_node(NodeKind::Gain { value: 1.0 }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let c = g.add_node(NodeKind::Sink { label: "out".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        g.connect(a, b).expect("connect should succeed");
        g.connect(b, c).expect("connect should succeed");
        assert_eq!(g.source_nodes(), vec![a]);
        assert_eq!(g.sink_nodes(), vec![c]);
    }

    #[test]
    fn test_graph_topological_order() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "in".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let b = g.add_node(NodeKind::Gain { value: 1.0 }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let c = g.add_node(NodeKind::Sink { label: "out".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        g.connect(a, b).expect("connect should succeed");
        g.connect(b, c).expect("connect should succeed");
        let order = g.topological_order().expect("order should be valid");
        assert_eq!(order, vec![a, b, c]);
    }

    #[test]
    fn test_graph_validate_empty() {
        let g = AudioGraph::new();
        let v = g.validate();
        assert!(matches!(v, GraphValidation::Errors(_)));
    }

    #[test]
    fn test_graph_validate_ok() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "in".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let b = g.add_node(NodeKind::Sink { label: "out".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        g.connect(a, b).expect("connect should succeed");
        assert_eq!(g.validate(), GraphValidation::Ok);
    }

    #[test]
    fn test_graph_connect_nonexistent() {
        let mut g = AudioGraph::new();
        let a = g.add_node(NodeKind::Source { label: "a".into() }, AudioSampleFmt::F32, 44100, ChannelLayout::Mono);
        let err = g.connect(a, 999).unwrap_err();
        assert!(matches!(err, AudioGraphError::NodeNotFound(999)));
    }

    #[test]
    fn test_audio_graph_error_display() {
        assert!(AudioGraphError::CycleDetected.to_string().contains("cycle"));
        assert!(AudioGraphError::NodeNotFound(7).to_string().contains("7"));
    }
}
