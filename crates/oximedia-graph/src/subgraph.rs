//! Subgraph extraction from a larger pipeline/filter graph.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// Describes the boundary of a subgraph: which external nodes feed into it
/// (inputs) and which internal nodes feed out of it (outputs).
#[derive(Debug, Clone, Default)]
pub struct SubgraphBoundary {
    /// Indices (in the parent graph) of the external nodes that drive
    /// subgraph inputs.  A node is an input-boundary node when it resides
    /// *outside* the subgraph but has an edge pointing *into* a subgraph node.
    pub input_nodes: Vec<usize>,
    /// Indices (in the parent graph) of the internal nodes whose outputs
    /// leave the subgraph.  A node is an output-boundary node when it resides
    /// *inside* the subgraph but has an edge pointing to a node *outside*.
    pub output_nodes: Vec<usize>,
}

impl SubgraphBoundary {
    /// Number of external inputs.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_nodes.len()
    }

    /// Number of output ports leaving the subgraph.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.output_nodes.len()
    }

    /// Returns `true` when both input and output lists are non-empty.
    #[must_use]
    pub fn is_connected(&self) -> bool {
        !self.input_nodes.is_empty() && !self.output_nodes.is_empty()
    }
}

/// A self-contained subgraph composed of a subset of nodes and the edges that
/// connect them internally.
#[derive(Debug, Clone)]
pub struct Subgraph {
    /// Node indices from the parent graph that are part of this subgraph.
    nodes: HashSet<usize>,
    /// Internal edges `(from, to)` where both ends are within `nodes`.
    internal_edges: Vec<(usize, usize)>,
    /// Human-readable label.
    label: String,
}

impl Subgraph {
    /// Create an empty subgraph with a label.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            nodes: HashSet::new(),
            internal_edges: Vec::new(),
            label: label.into(),
        }
    }

    /// Add a node (by its parent-graph index) to the subgraph.
    pub fn add_node(&mut self, node_idx: usize) {
        self.nodes.insert(node_idx);
    }

    /// Add an internal edge.  Both endpoints must already be in the subgraph;
    /// returns an error string otherwise.
    pub fn add_internal_edge(&mut self, from: usize, to: usize) -> Result<(), String> {
        if !self.nodes.contains(&from) {
            return Err(format!("Node {from} is not part of this subgraph"));
        }
        if !self.nodes.contains(&to) {
            return Err(format!("Node {to} is not part of this subgraph"));
        }
        self.internal_edges.push((from, to));
        Ok(())
    }

    /// Number of nodes in the subgraph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of internal edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.internal_edges.len()
    }

    /// Returns `true` if `node_idx` is part of this subgraph.
    #[must_use]
    pub fn contains_node(&self, node_idx: usize) -> bool {
        self.nodes.contains(&node_idx)
    }

    /// Label of the subgraph.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Slice of all internal edges.
    #[must_use]
    pub fn internal_edges(&self) -> &[(usize, usize)] {
        &self.internal_edges
    }

    /// Compute the boundary of this subgraph given the full parent-graph edge
    /// list `all_edges`.
    ///
    /// An input boundary node is a node *outside* the subgraph that has an
    /// edge pointing *into* a subgraph node.
    ///
    /// An output boundary node is a node *inside* the subgraph that has an
    /// edge pointing *outside* the subgraph.
    #[must_use]
    pub fn boundary(&self, all_edges: &[(usize, usize)]) -> SubgraphBoundary {
        let mut inputs: HashSet<usize> = HashSet::new();
        let mut outputs: HashSet<usize> = HashSet::new();

        for &(from, to) in all_edges {
            if !self.nodes.contains(&from) && self.nodes.contains(&to) {
                inputs.insert(from);
            }
            if self.nodes.contains(&from) && !self.nodes.contains(&to) {
                outputs.insert(from);
            }
        }

        SubgraphBoundary {
            input_nodes: inputs.into_iter().collect(),
            output_nodes: outputs.into_iter().collect(),
        }
    }
}

/// Extracts a [`Subgraph`] from a parent graph by collecting all nodes
/// reachable from a set of seed nodes via forward traversal.
pub struct SubgraphExtractor {
    /// All directed edges in the parent graph.
    all_edges: Vec<(usize, usize)>,
    /// Total node count in the parent graph (for bounds checking).
    node_count: usize,
}

impl SubgraphExtractor {
    /// Create an extractor for a parent graph described by `edges`.
    #[must_use]
    pub fn new(node_count: usize, edges: Vec<(usize, usize)>) -> Self {
        Self {
            all_edges: edges,
            node_count,
        }
    }

    /// Extract the subgraph consisting of `seed` nodes and all nodes
    /// reachable from them by following directed edges.
    ///
    /// Returns an error if any seed index is out of range.
    pub fn extract(&self, seeds: &[usize], label: impl Into<String>) -> Result<Subgraph, String> {
        for &s in seeds {
            if s >= self.node_count {
                return Err(format!(
                    "Seed node {s} is out of range (node_count={})",
                    self.node_count
                ));
            }
        }

        // Build adjacency for forward traversal
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(f, t) in &self.all_edges {
            adj.entry(f).or_default().push(t);
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<usize> = seeds.to_vec();

        while let Some(node) = stack.pop() {
            if visited.insert(node) {
                if let Some(neighbours) = adj.get(&node) {
                    for &nb in neighbours {
                        if !visited.contains(&nb) {
                            stack.push(nb);
                        }
                    }
                }
            }
        }

        let mut sg = Subgraph::new(label);
        for n in &visited {
            sg.add_node(*n);
        }
        // Add internal edges
        for &(f, t) in &self.all_edges {
            if visited.contains(&f) && visited.contains(&t) {
                sg.add_internal_edge(f, t).ok();
            }
        }
        Ok(sg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_extractor() -> SubgraphExtractor {
        // 5-node graph: 0→1→3, 0→2→3, 3→4
        SubgraphExtractor::new(5, vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)])
    }

    // --- SubgraphBoundary tests ---

    #[test]
    fn test_boundary_input_count() {
        let b = SubgraphBoundary {
            input_nodes: vec![0, 1],
            output_nodes: vec![5],
        };
        assert_eq!(b.input_count(), 2);
    }

    #[test]
    fn test_boundary_output_count() {
        let b = SubgraphBoundary {
            input_nodes: vec![0],
            output_nodes: vec![5, 6],
        };
        assert_eq!(b.output_count(), 2);
    }

    #[test]
    fn test_boundary_is_connected() {
        let b = SubgraphBoundary {
            input_nodes: vec![0],
            output_nodes: vec![3],
        };
        assert!(b.is_connected());
    }

    #[test]
    fn test_boundary_not_connected_no_inputs() {
        let b = SubgraphBoundary {
            input_nodes: vec![],
            output_nodes: vec![3],
        };
        assert!(!b.is_connected());
    }

    // --- Subgraph tests ---

    #[test]
    fn test_add_node_increments_count() {
        let mut sg = Subgraph::new("test");
        sg.add_node(0);
        sg.add_node(1);
        assert_eq!(sg.node_count(), 2);
    }

    #[test]
    fn test_add_internal_edge_valid() {
        let mut sg = Subgraph::new("test");
        sg.add_node(0);
        sg.add_node(1);
        assert!(sg.add_internal_edge(0, 1).is_ok());
        assert_eq!(sg.edge_count(), 1);
    }

    #[test]
    fn test_add_internal_edge_missing_node_returns_error() {
        let mut sg = Subgraph::new("test");
        sg.add_node(0);
        assert!(sg.add_internal_edge(0, 99).is_err());
    }

    #[test]
    fn test_contains_node_true() {
        let mut sg = Subgraph::new("test");
        sg.add_node(5);
        assert!(sg.contains_node(5));
    }

    #[test]
    fn test_contains_node_false() {
        let sg = Subgraph::new("test");
        assert!(!sg.contains_node(0));
    }

    #[test]
    fn test_label_stored() {
        let sg = Subgraph::new("my_sub");
        assert_eq!(sg.label(), "my_sub");
    }

    #[test]
    fn test_boundary_detects_input_output_nodes() {
        let mut sg = Subgraph::new("inner");
        // subgraph contains nodes 1, 2, 3
        sg.add_node(1);
        sg.add_node(2);
        sg.add_node(3);
        let all_edges = vec![(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)];
        let b = sg.boundary(&all_edges);
        // Node 0 feeds in → input boundary
        assert!(b.input_nodes.contains(&0));
        // Node 3 feeds out to 4 → output boundary
        assert!(b.output_nodes.contains(&3));
    }

    // --- SubgraphExtractor tests ---

    #[test]
    fn test_extract_from_root_contains_all_nodes() {
        let ext = sample_extractor();
        let sg = ext.extract(&[0], "full").expect("extract should succeed");
        assert_eq!(sg.node_count(), 5);
    }

    #[test]
    fn test_extract_from_midpoint() {
        let ext = sample_extractor();
        let sg = ext.extract(&[3], "tail").expect("extract should succeed");
        assert!(sg.contains_node(3));
        assert!(sg.contains_node(4));
        assert!(!sg.contains_node(0));
    }

    #[test]
    fn test_extract_includes_internal_edges() {
        let ext = sample_extractor();
        let sg = ext.extract(&[0], "full").expect("extract should succeed");
        assert!(sg.edge_count() > 0);
    }

    #[test]
    fn test_extract_out_of_range_seed_returns_error() {
        let ext = sample_extractor();
        assert!(ext.extract(&[99], "bad").is_err());
    }

    #[test]
    fn test_extract_empty_seeds() {
        let ext = sample_extractor();
        let sg = ext.extract(&[], "empty").expect("extract should succeed");
        assert_eq!(sg.node_count(), 0);
    }
}
