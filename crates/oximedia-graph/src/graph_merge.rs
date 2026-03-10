#![allow(dead_code)]
//! Graph merging utilities for combining multiple filter graphs.
//!
//! This module supports merging two directed graphs into one, with
//! configurable strategies for handling node ID conflicts, edge
//! deduplication, and cross-graph linking.

use std::collections::{HashMap, HashSet};

/// A node identifier in a merge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MergeNodeId(
    /// Inner identifier value.
    pub usize,
);

impl std::fmt::Display for MergeNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MergeNode({})", self.0)
    }
}

/// A labeled node in a merge graph.
#[derive(Debug, Clone)]
pub struct MergeNode {
    /// Unique identifier.
    pub id: MergeNodeId,
    /// Human-readable label.
    pub label: String,
    /// Metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl MergeNode {
    /// Create a new merge node.
    pub fn new(id: MergeNodeId, label: &str) -> Self {
        Self {
            id,
            label: label.to_string(),
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// A directed edge in a merge graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MergeEdge {
    /// Source node.
    pub from: MergeNodeId,
    /// Destination node.
    pub to: MergeNodeId,
    /// Optional label for the edge.
    pub label: Option<String>,
    /// Weight of the edge (for priority resolution).
    pub weight: u32,
}

impl MergeEdge {
    /// Create a new merge edge.
    pub fn new(from: MergeNodeId, to: MergeNodeId) -> Self {
        Self {
            from,
            to,
            label: None,
            weight: 1,
        }
    }

    /// Create a new merge edge with a label.
    pub fn with_label(from: MergeNodeId, to: MergeNodeId, label: &str) -> Self {
        Self {
            from,
            to,
            label: Some(label.to_string()),
            weight: 1,
        }
    }

    /// Set the weight of the edge.
    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
}

/// Strategy for resolving node ID conflicts during a merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictStrategy {
    /// Keep nodes from the first graph, skip duplicates from the second.
    KeepFirst,
    /// Keep nodes from the second graph, overwriting the first.
    KeepSecond,
    /// Remap IDs of the second graph to avoid conflicts.
    Remap,
    /// Fail the merge if any conflict is detected.
    Fail,
}

#[allow(clippy::derivable_impls)]
impl Default for ConflictStrategy {
    fn default() -> Self {
        Self::Remap
    }
}

/// Configuration for the merge operation.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// How to handle node ID conflicts.
    pub conflict_strategy: ConflictStrategy,
    /// Whether to deduplicate edges after merging.
    pub deduplicate_edges: bool,
    /// Optional cross-links to add between the two graphs.
    pub cross_links: Vec<(MergeNodeId, MergeNodeId)>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            conflict_strategy: ConflictStrategy::Remap,
            deduplicate_edges: true,
            cross_links: Vec::new(),
        }
    }
}

/// Error types for merge operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeError {
    /// Node ID conflict with Fail strategy.
    Conflict(
        /// The conflicting node ID.
        MergeNodeId,
    ),
    /// A referenced node does not exist.
    NodeNotFound(
        /// The missing node.
        MergeNodeId,
    ),
    /// Both graphs are empty.
    EmptyGraphs,
}

impl std::fmt::Display for MergeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conflict(id) => write!(f, "Node conflict: {id}"),
            Self::NodeNotFound(id) => write!(f, "Node not found: {id}"),
            Self::EmptyGraphs => write!(f, "Both graphs are empty"),
        }
    }
}

/// A directed graph supporting merge operations.
pub struct MergeGraph {
    /// Nodes indexed by their ID.
    nodes: HashMap<MergeNodeId, MergeNode>,
    /// Directed edges.
    edges: Vec<MergeEdge>,
}

impl MergeGraph {
    /// Create a new empty merge graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: MergeNode) {
        self.nodes.insert(node.id, node);
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: MergeEdge) {
        self.edges.push(edge);
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if a node exists in the graph.
    pub fn has_node(&self, id: MergeNodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Get a reference to a node by ID.
    pub fn get_node(&self, id: MergeNodeId) -> Option<&MergeNode> {
        self.nodes.get(&id)
    }

    /// Return all node IDs, sorted.
    pub fn node_ids(&self) -> Vec<MergeNodeId> {
        let mut ids: Vec<MergeNodeId> = self.nodes.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Return all edges.
    pub fn edges(&self) -> &[MergeEdge] {
        &self.edges
    }

    /// Return the next available ID (max + 1).
    pub fn next_id(&self) -> MergeNodeId {
        let max = self.nodes.keys().map(|k| k.0).max().unwrap_or(0);
        MergeNodeId(max + 1)
    }

    /// Merge another graph into this one using the given configuration.
    pub fn merge(
        &mut self,
        other: &MergeGraph,
        config: &MergeConfig,
    ) -> Result<MergeMapping, MergeError> {
        let mut mapping = MergeMapping::new();

        match config.conflict_strategy {
            ConflictStrategy::Remap => {
                let mut next_id = self.next_id().0;
                for (&old_id, node) in &other.nodes {
                    if self.nodes.contains_key(&old_id) {
                        let new_id = MergeNodeId(next_id);
                        next_id += 1;
                        mapping.add(old_id, new_id);
                        let mut new_node = node.clone();
                        new_node.id = new_id;
                        self.nodes.insert(new_id, new_node);
                    } else {
                        mapping.add(old_id, old_id);
                        self.nodes.insert(old_id, node.clone());
                    }
                }
            }
            ConflictStrategy::KeepFirst => {
                for (&old_id, node) in &other.nodes {
                    self.nodes.entry(old_id).or_insert_with(|| node.clone());
                    mapping.add(old_id, old_id);
                }
            }
            ConflictStrategy::KeepSecond => {
                for (&old_id, node) in &other.nodes {
                    self.nodes.insert(old_id, node.clone());
                    mapping.add(old_id, old_id);
                }
            }
            ConflictStrategy::Fail => {
                for &old_id in other.nodes.keys() {
                    if self.nodes.contains_key(&old_id) {
                        return Err(MergeError::Conflict(old_id));
                    }
                }
                for (&old_id, node) in &other.nodes {
                    self.nodes.insert(old_id, node.clone());
                    mapping.add(old_id, old_id);
                }
            }
        }

        // Remap and add edges from the other graph
        for edge in &other.edges {
            let new_from = mapping.resolve(edge.from);
            let new_to = mapping.resolve(edge.to);
            let new_edge = MergeEdge {
                from: new_from,
                to: new_to,
                label: edge.label.clone(),
                weight: edge.weight,
            };
            self.edges.push(new_edge);
        }

        // Add cross-links
        for &(from, to) in &config.cross_links {
            self.edges.push(MergeEdge::new(from, to));
        }

        // Deduplicate edges if requested
        if config.deduplicate_edges {
            self.deduplicate_edges();
        }

        Ok(mapping)
    }

    /// Remove duplicate edges (same from/to pair).
    fn deduplicate_edges(&mut self) {
        let mut seen: HashSet<(MergeNodeId, MergeNodeId)> = HashSet::new();
        self.edges.retain(|e| seen.insert((e.from, e.to)));
    }

    /// Return the set of edges as (from, to) pairs.
    pub fn edge_pairs(&self) -> HashSet<(MergeNodeId, MergeNodeId)> {
        self.edges.iter().map(|e| (e.from, e.to)).collect()
    }

    /// Return the adjacency list representation.
    pub fn adjacency_list(&self) -> HashMap<MergeNodeId, Vec<MergeNodeId>> {
        let mut adj: HashMap<MergeNodeId, Vec<MergeNodeId>> = HashMap::new();
        for id in self.nodes.keys() {
            adj.entry(*id).or_default();
        }
        for edge in &self.edges {
            adj.entry(edge.from).or_default().push(edge.to);
        }
        for list in adj.values_mut() {
            list.sort();
        }
        adj
    }
}

impl Default for MergeGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks ID remapping from a merge operation.
#[derive(Debug, Clone)]
pub struct MergeMapping {
    /// Map from old ID to new ID.
    map: HashMap<MergeNodeId, MergeNodeId>,
}

impl MergeMapping {
    /// Create a new empty mapping.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    /// Add a mapping entry.
    pub fn add(&mut self, old_id: MergeNodeId, new_id: MergeNodeId) {
        self.map.insert(old_id, new_id);
    }

    /// Resolve an old ID to its new ID. Returns the old ID if no mapping exists.
    pub fn resolve(&self, old_id: MergeNodeId) -> MergeNodeId {
        self.map.get(&old_id).copied().unwrap_or(old_id)
    }

    /// Return the number of remapped entries.
    pub fn remap_count(&self) -> usize {
        self.map.iter().filter(|(k, v)| k != v).count()
    }

    /// Check if any IDs were actually remapped.
    pub fn has_remaps(&self) -> bool {
        self.remap_count() > 0
    }

    /// Return all mappings.
    pub fn mappings(&self) -> &HashMap<MergeNodeId, MergeNodeId> {
        &self.map
    }
}

impl Default for MergeMapping {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mn(id: usize) -> MergeNodeId {
        MergeNodeId(id)
    }

    fn make_node(id: usize, label: &str) -> MergeNode {
        MergeNode::new(mn(id), label)
    }

    #[test]
    fn test_empty_graph() {
        let graph = MergeGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_node_and_edge() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        graph.add_node(make_node(1, "B"));
        graph.add_edge(MergeEdge::new(mn(0), mn(1)));
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_merge_no_conflict() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));
        g1.add_node(make_node(1, "B"));
        g1.add_edge(MergeEdge::new(mn(0), mn(1)));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(2, "C"));
        g2.add_node(make_node(3, "D"));
        g2.add_edge(MergeEdge::new(mn(2), mn(3)));

        let config = MergeConfig::default();
        let mapping = g1.merge(&g2, &config).expect("merge should succeed");
        assert_eq!(g1.node_count(), 4);
        assert_eq!(g1.edge_count(), 2);
        assert!(!mapping.has_remaps());
    }

    #[test]
    fn test_merge_with_remap() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "B")); // Conflict!

        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::Remap,
            ..Default::default()
        };
        let mapping = g1.merge(&g2, &config).expect("merge should succeed");
        assert_eq!(g1.node_count(), 2);
        assert!(mapping.has_remaps());
    }

    #[test]
    fn test_merge_keep_first() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "B"));

        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::KeepFirst,
            ..Default::default()
        };
        g1.merge(&g2, &config).expect("merge should succeed");
        assert_eq!(g1.node_count(), 1);
        assert_eq!(
            g1.get_node(mn(0)).expect("value should be valid").label,
            "A"
        );
    }

    #[test]
    fn test_merge_keep_second() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "B"));

        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::KeepSecond,
            ..Default::default()
        };
        g1.merge(&g2, &config).expect("merge should succeed");
        assert_eq!(g1.node_count(), 1);
        assert_eq!(
            g1.get_node(mn(0)).expect("value should be valid").label,
            "B"
        );
    }

    #[test]
    fn test_merge_fail_on_conflict() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(0, "B"));

        let config = MergeConfig {
            conflict_strategy: ConflictStrategy::Fail,
            ..Default::default()
        };
        let result = g1.merge(&g2, &config);
        assert!(matches!(result, Err(MergeError::Conflict(_))));
    }

    #[test]
    fn test_merge_with_cross_links() {
        let mut g1 = MergeGraph::new();
        g1.add_node(make_node(0, "A"));

        let mut g2 = MergeGraph::new();
        g2.add_node(make_node(1, "B"));

        let config = MergeConfig {
            cross_links: vec![(mn(0), mn(1))],
            ..Default::default()
        };
        g1.merge(&g2, &config).expect("merge should succeed");
        assert_eq!(g1.edge_count(), 1);
    }

    #[test]
    fn test_edge_deduplication() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        graph.add_node(make_node(1, "B"));
        graph.add_edge(MergeEdge::new(mn(0), mn(1)));
        graph.add_edge(MergeEdge::new(mn(0), mn(1)));
        graph.deduplicate_edges();
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_edge_pairs() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        graph.add_node(make_node(1, "B"));
        graph.add_edge(MergeEdge::new(mn(0), mn(1)));
        let pairs = graph.edge_pairs();
        assert!(pairs.contains(&(mn(0), mn(1))));
    }

    #[test]
    fn test_adjacency_list() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        graph.add_node(make_node(1, "B"));
        graph.add_node(make_node(2, "C"));
        graph.add_edge(MergeEdge::new(mn(0), mn(1)));
        graph.add_edge(MergeEdge::new(mn(0), mn(2)));
        let adj = graph.adjacency_list();
        assert_eq!(adj[&mn(0)], vec![mn(1), mn(2)]);
    }

    #[test]
    fn test_node_metadata() {
        let node = make_node(0, "Test").with_metadata("type", "source");
        assert_eq!(node.metadata.get("type"), Some(&"source".to_string()));
    }

    #[test]
    fn test_edge_with_label() {
        let edge = MergeEdge::with_label(mn(0), mn(1), "data_flow");
        assert_eq!(edge.label, Some("data_flow".to_string()));
    }

    #[test]
    fn test_edge_weight() {
        let edge = MergeEdge::new(mn(0), mn(1)).with_weight(5);
        assert_eq!(edge.weight, 5);
    }

    #[test]
    fn test_merge_mapping_resolve() {
        let mut mapping = MergeMapping::new();
        mapping.add(mn(0), mn(10));
        assert_eq!(mapping.resolve(mn(0)), mn(10));
        assert_eq!(mapping.resolve(mn(5)), mn(5)); // unmapped
    }

    #[test]
    fn test_merge_error_display() {
        let err = MergeError::Conflict(mn(5));
        assert!(format!("{err}").contains("5"));
        let err2 = MergeError::EmptyGraphs;
        assert_eq!(format!("{err2}"), "Both graphs are empty");
    }

    #[test]
    fn test_next_id() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        graph.add_node(make_node(5, "B"));
        assert_eq!(graph.next_id(), mn(6));
    }

    #[test]
    fn test_has_node() {
        let mut graph = MergeGraph::new();
        graph.add_node(make_node(0, "A"));
        assert!(graph.has_node(mn(0)));
        assert!(!graph.has_node(mn(1)));
    }
}
