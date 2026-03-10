//! Pipeline-centric graph for media processing.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// Type of a node in a pipeline graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineNodeType {
    /// Produces frames but has no upstream inputs (e.g., a decoder).
    Source,
    /// Transforms frames (e.g., scaler, colour corrector).
    Transform,
    /// Merges multiple streams into one (e.g., overlay, mix).
    Merge,
    /// Splits one stream into multiple (e.g., tee).
    Split,
    /// Consumes frames and produces no outputs (e.g., encoder, display sink).
    Sink,
}

impl PipelineNodeType {
    /// Returns `true` for node types that terminate the pipeline (no outputs).
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Sink)
    }

    /// Returns `true` for node types that originate data (no required inputs).
    #[must_use]
    pub fn is_source(&self) -> bool {
        matches!(self, Self::Source)
    }
}

/// Lightweight handle identifying a node inside a [`PipelineGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineNodeId(pub u32);

/// Metadata stored per node.
#[derive(Debug, Clone)]
struct NodeMeta {
    node_type: PipelineNodeType,
    label: String,
}

/// A directed acyclic graph of pipeline nodes.
pub struct PipelineGraph {
    nodes: HashMap<PipelineNodeId, NodeMeta>,
    /// Adjacency list: `edges[a]` = set of nodes that `a` feeds into.
    edges: HashMap<PipelineNodeId, HashSet<PipelineNodeId>>,
    next_id: u32,
}

impl PipelineGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a new node and return its id.
    pub fn add_node(
        &mut self,
        node_type: PipelineNodeType,
        label: impl Into<String>,
    ) -> PipelineNodeId {
        let id = PipelineNodeId(self.next_id);
        self.next_id += 1;
        self.nodes.insert(
            id,
            NodeMeta {
                node_type,
                label: label.into(),
            },
        );
        self.edges.entry(id).or_default();
        id
    }

    /// Connect `from` → `to`.
    ///
    /// Returns an error string if either node does not exist.
    pub fn connect(&mut self, from: PipelineNodeId, to: PipelineNodeId) -> Result<(), String> {
        if !self.nodes.contains_key(&from) {
            return Err(format!("Source node {:?} not found", from));
        }
        if !self.nodes.contains_key(&to) {
            return Err(format!("Destination node {:?} not found", to));
        }
        self.edges.entry(from).or_default().insert(to);
        // Ensure destination also has an entry in the adjacency list.
        self.edges.entry(to).or_default();
        Ok(())
    }

    /// Returns `true` if the graph contains no directed cycles (is a valid DAG).
    #[must_use]
    pub fn is_valid_dag(&self) -> bool {
        // Kahn's algorithm
        let mut in_degree: HashMap<PipelineNodeId, usize> =
            self.nodes.keys().map(|&k| (k, 0)).collect();
        for neighbours in self.edges.values() {
            for &n in neighbours {
                *in_degree.entry(n).or_insert(0) += 1;
            }
        }
        let mut queue: VecDeque<PipelineNodeId> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&k, _)| k)
            .collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(neighbours) = self.edges.get(&node) {
                for &n in neighbours {
                    let deg = in_degree.entry(n).or_insert(0);
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(n);
                    }
                }
            }
        }
        visited == self.nodes.len()
    }

    /// Returns the ids of all source nodes (nodes with no incoming edges).
    #[must_use]
    pub fn sources(&self) -> Vec<PipelineNodeId> {
        let mut has_incoming: HashSet<PipelineNodeId> = HashSet::new();
        for neighbours in self.edges.values() {
            for &n in neighbours {
                has_incoming.insert(n);
            }
        }
        self.nodes
            .keys()
            .copied()
            .filter(|id| !has_incoming.contains(id))
            .collect()
    }

    /// Returns the ids of all sink nodes (nodes with no outgoing edges).
    #[must_use]
    pub fn sinks(&self) -> Vec<PipelineNodeId> {
        self.edges
            .iter()
            .filter(|(id, neighbours)| self.nodes.contains_key(id) && neighbours.is_empty())
            .map(|(&id, _)| id)
            .collect()
    }

    /// Total number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of directed edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|s| s.len()).sum()
    }

    /// Look up the type of a node.
    #[must_use]
    pub fn node_type(&self, id: PipelineNodeId) -> Option<PipelineNodeType> {
        self.nodes.get(&id).map(|m| m.node_type)
    }

    /// Look up the label of a node.
    #[must_use]
    pub fn node_label(&self, id: PipelineNodeId) -> Option<&str> {
        self.nodes.get(&id).map(|m| m.label.as_str())
    }
}

impl Default for PipelineGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> (PipelineGraph, PipelineNodeId, PipelineNodeId) {
        let mut g = PipelineGraph::new();
        let src = g.add_node(PipelineNodeType::Source, "src");
        let sink = g.add_node(PipelineNodeType::Sink, "sink");
        g.connect(src, sink).expect("connect should succeed");
        (g, src, sink)
    }

    // --- PipelineNodeType tests ---

    #[test]
    fn test_sink_is_terminal() {
        assert!(PipelineNodeType::Sink.is_terminal());
    }

    #[test]
    fn test_transform_not_terminal() {
        assert!(!PipelineNodeType::Transform.is_terminal());
    }

    #[test]
    fn test_source_is_source() {
        assert!(PipelineNodeType::Source.is_source());
    }

    #[test]
    fn test_merge_not_source() {
        assert!(!PipelineNodeType::Merge.is_source());
    }

    // --- PipelineGraph tests ---

    #[test]
    fn test_add_node_increments_count() {
        let mut g = PipelineGraph::new();
        g.add_node(PipelineNodeType::Source, "s");
        g.add_node(PipelineNodeType::Sink, "k");
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_connect_increments_edge_count() {
        let (g, _, _) = simple_graph();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_connect_nonexistent_node_returns_error() {
        let mut g = PipelineGraph::new();
        let src = g.add_node(PipelineNodeType::Source, "s");
        let ghost = PipelineNodeId(999);
        assert!(g.connect(src, ghost).is_err());
    }

    #[test]
    fn test_is_valid_dag_linear() {
        let (g, _, _) = simple_graph();
        assert!(g.is_valid_dag());
    }

    #[test]
    fn test_is_valid_dag_cycle_detected() {
        let mut g = PipelineGraph::new();
        let a = g.add_node(PipelineNodeType::Source, "a");
        let b = g.add_node(PipelineNodeType::Transform, "b");
        g.connect(a, b).expect("connect should succeed");
        g.connect(b, a).expect("connect should succeed"); // create cycle
        assert!(!g.is_valid_dag());
    }

    #[test]
    fn test_sources_returns_nodes_without_incoming() {
        let (g, src, _) = simple_graph();
        let sources = g.sources();
        assert_eq!(sources.len(), 1);
        assert!(sources.contains(&src));
    }

    #[test]
    fn test_sinks_returns_nodes_without_outgoing() {
        let (g, _, sink) = simple_graph();
        let sinks = g.sinks();
        assert_eq!(sinks.len(), 1);
        assert!(sinks.contains(&sink));
    }

    #[test]
    fn test_node_type_query() {
        let (g, src, _) = simple_graph();
        assert_eq!(g.node_type(src), Some(PipelineNodeType::Source));
    }

    #[test]
    fn test_node_label_query() {
        let (g, src, _) = simple_graph();
        assert_eq!(g.node_label(src), Some("src"));
    }

    #[test]
    fn test_diamond_graph_is_valid_dag() {
        let mut g = PipelineGraph::new();
        let a = g.add_node(PipelineNodeType::Source, "a");
        let b = g.add_node(PipelineNodeType::Transform, "b");
        let c = g.add_node(PipelineNodeType::Transform, "c");
        let d = g.add_node(PipelineNodeType::Sink, "d");
        g.connect(a, b).expect("connect should succeed");
        g.connect(a, c).expect("connect should succeed");
        g.connect(b, d).expect("connect should succeed");
        g.connect(c, d).expect("connect should succeed");
        assert!(g.is_valid_dag());
    }

    #[test]
    fn test_multiple_sources_and_sinks() {
        let mut g = PipelineGraph::new();
        let s1 = g.add_node(PipelineNodeType::Source, "s1");
        let s2 = g.add_node(PipelineNodeType::Source, "s2");
        let m = g.add_node(PipelineNodeType::Merge, "m");
        let k1 = g.add_node(PipelineNodeType::Sink, "k1");
        let k2 = g.add_node(PipelineNodeType::Sink, "k2");
        g.connect(s1, m).expect("connect should succeed");
        g.connect(s2, m).expect("connect should succeed");
        g.connect(m, k1).expect("connect should succeed");
        g.connect(m, k2).expect("connect should succeed");
        assert_eq!(g.sources().len(), 2);
        assert_eq!(g.sinks().len(), 2);
    }

    #[test]
    fn test_empty_graph_is_valid_dag() {
        let g = PipelineGraph::new();
        assert!(g.is_valid_dag());
    }
}
