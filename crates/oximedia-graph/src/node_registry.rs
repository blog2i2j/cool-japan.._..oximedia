//! Node registry for the filter graph pipeline.
//!
//! Provides a typed catalogue of all nodes registered in a graph, supporting
//! lookup by kind and counting of enabled nodes.

#![allow(dead_code)]

use std::collections::HashMap;

/// Classification of a graph node by its role in the pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// Produces frames (decoder, file reader, live capture).
    Source,
    /// Transforms frames (scaler, colour converter, denoiser).
    Filter,
    /// Consumes frames (encoder, display, file writer).
    Sink,
    /// Splits or merges multiple streams.
    Mux,
}

impl NodeKind {
    /// Returns `true` if this node kind originates data in the graph.
    pub fn is_source(&self) -> bool {
        *self == NodeKind::Source
    }

    /// Returns `true` if this node kind terminates a pipeline branch.
    pub fn is_sink(&self) -> bool {
        *self == NodeKind::Sink
    }

    /// Returns a short human-readable label for the kind.
    pub fn label(&self) -> &'static str {
        match self {
            NodeKind::Source => "source",
            NodeKind::Filter => "filter",
            NodeKind::Sink => "sink",
            NodeKind::Mux => "mux",
        }
    }
}

/// Lightweight descriptor for a node registered in the `NodeRegistry`.
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Numeric identifier for this node (unique within a registry).
    id: u64,
    /// Human-readable name.
    name: String,
    /// Classification of this node.
    kind: NodeKind,
    /// Whether this node is currently enabled (participates in processing).
    enabled: bool,
}

impl GraphNode {
    /// Creates a new enabled `GraphNode`.
    pub fn new(id: u64, name: impl Into<String>, kind: NodeKind) -> Self {
        Self {
            id,
            name: name.into(),
            kind,
            enabled: true,
        }
    }

    /// Returns the node ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Returns the node name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the node kind.
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }

    /// Returns `true` if the node is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enables or disables the node.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// A catalogue of `GraphNode` entries keyed by their numeric ID.
#[derive(Debug, Clone, Default)]
pub struct NodeRegistry {
    nodes: HashMap<u64, GraphNode>,
    next_id: u64,
}

impl NodeRegistry {
    /// Creates an empty `NodeRegistry`.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    /// Adds a new node with the given name and kind, assigning a fresh ID.
    /// Returns the assigned ID.
    pub fn add(&mut self, name: impl Into<String>, kind: NodeKind) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, GraphNode::new(id, name, kind));
        id
    }

    /// Looks up a node by ID.
    pub fn get(&self, id: u64) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Looks up a node by ID mutably.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    /// Returns all nodes whose kind matches `kind`.
    pub fn find_by_kind(&self, kind: &NodeKind) -> Vec<&GraphNode> {
        self.nodes.values().filter(|n| n.kind() == kind).collect()
    }

    /// Returns the number of currently enabled nodes.
    pub fn enabled_count(&self) -> usize {
        self.nodes.values().filter(|n| n.is_enabled()).count()
    }

    /// Returns the total number of registered nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if no nodes have been registered.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Removes a node by ID. Returns the removed node if present.
    pub fn remove(&mut self, id: u64) -> Option<GraphNode> {
        self.nodes.remove(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_kind_is_source() {
        assert!(NodeKind::Source.is_source());
        assert!(!NodeKind::Sink.is_source());
    }

    #[test]
    fn test_node_kind_is_sink() {
        assert!(NodeKind::Sink.is_sink());
        assert!(!NodeKind::Filter.is_sink());
    }

    #[test]
    fn test_node_kind_label() {
        assert_eq!(NodeKind::Source.label(), "source");
        assert_eq!(NodeKind::Filter.label(), "filter");
        assert_eq!(NodeKind::Sink.label(), "sink");
        assert_eq!(NodeKind::Mux.label(), "mux");
    }

    #[test]
    fn test_graph_node_enabled_by_default() {
        let n = GraphNode::new(0, "node0", NodeKind::Source);
        assert!(n.is_enabled());
    }

    #[test]
    fn test_graph_node_set_enabled() {
        let mut n = GraphNode::new(0, "node0", NodeKind::Filter);
        n.set_enabled(false);
        assert!(!n.is_enabled());
    }

    #[test]
    fn test_registry_add_returns_incremental_ids() {
        let mut reg = NodeRegistry::new();
        let id0 = reg.add("a", NodeKind::Source);
        let id1 = reg.add("b", NodeKind::Sink);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_registry_get() {
        let mut reg = NodeRegistry::new();
        let id = reg.add("scaler", NodeKind::Filter);
        let node = reg.get(id).expect("get should succeed");
        assert_eq!(node.name(), "scaler");
        assert_eq!(node.kind(), &NodeKind::Filter);
    }

    #[test]
    fn test_registry_find_by_kind() {
        let mut reg = NodeRegistry::new();
        reg.add("src1", NodeKind::Source);
        reg.add("src2", NodeKind::Source);
        reg.add("sink1", NodeKind::Sink);
        let sources = reg.find_by_kind(&NodeKind::Source);
        assert_eq!(sources.len(), 2);
    }

    #[test]
    fn test_registry_enabled_count() {
        let mut reg = NodeRegistry::new();
        let id0 = reg.add("a", NodeKind::Source);
        let id1 = reg.add("b", NodeKind::Filter);
        reg.get_mut(id1)
            .expect("get_mut should succeed")
            .set_enabled(false);
        assert_eq!(reg.enabled_count(), 1);
        let _ = id0;
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = NodeRegistry::new();
        let id = reg.add("x", NodeKind::Mux);
        assert!(reg.remove(id).is_some());
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_len() {
        let mut reg = NodeRegistry::new();
        reg.add("a", NodeKind::Source);
        reg.add("b", NodeKind::Sink);
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_find_by_kind_empty() {
        let reg = NodeRegistry::new();
        assert!(reg.find_by_kind(&NodeKind::Mux).is_empty());
    }

    #[test]
    fn test_registry_default() {
        let reg: NodeRegistry = Default::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_node_id_accessor() {
        let n = GraphNode::new(42, "node42", NodeKind::Filter);
        assert_eq!(n.id(), 42);
    }
}
