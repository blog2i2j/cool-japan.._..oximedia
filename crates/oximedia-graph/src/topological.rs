#![allow(dead_code)]
//! Topological sorting for directed acyclic graphs.
//!
//! This module provides Kahn's algorithm and DFS-based topological sort
//! implementations for ordering graph nodes such that every directed edge
//! goes from an earlier node to a later node in the ordering.

use std::collections::{HashMap, HashSet, VecDeque};

/// A node identifier in the topological graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TopoNodeId(
    /// Inner identifier value.
    pub usize,
);

impl std::fmt::Display for TopoNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Error types for topological sort operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopoError {
    /// The graph contains a cycle, making topological sort impossible.
    CycleDetected(
        /// Nodes involved in the cycle.
        Vec<TopoNodeId>,
    ),
    /// A referenced node does not exist in the graph.
    NodeNotFound(
        /// The missing node.
        TopoNodeId,
    ),
    /// The graph is empty.
    EmptyGraph,
}

impl std::fmt::Display for TopoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CycleDetected(nodes) => {
                write!(f, "Cycle detected involving {} nodes", nodes.len())
            }
            Self::NodeNotFound(id) => write!(f, "Node {id} not found"),
            Self::EmptyGraph => write!(f, "Graph is empty"),
        }
    }
}

/// Directed graph structure for topological sorting.
pub struct TopoGraph {
    /// Adjacency list: node -> set of successor nodes.
    adjacency: HashMap<TopoNodeId, HashSet<TopoNodeId>>,
    /// Reverse adjacency: node -> set of predecessor nodes.
    reverse: HashMap<TopoNodeId, HashSet<TopoNodeId>>,
}

impl TopoGraph {
    /// Create a new empty topological graph.
    pub fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, id: TopoNodeId) {
        self.adjacency.entry(id).or_default();
        self.reverse.entry(id).or_default();
    }

    /// Add a directed edge from `from` to `to`.
    pub fn add_edge(&mut self, from: TopoNodeId, to: TopoNodeId) {
        self.add_node(from);
        self.add_node(to);
        self.adjacency.entry(from).or_default().insert(to);
        self.reverse.entry(to).or_default().insert(from);
    }

    /// Return the number of nodes.
    pub fn node_count(&self) -> usize {
        self.adjacency.len()
    }

    /// Return the number of edges.
    pub fn edge_count(&self) -> usize {
        self.adjacency.values().map(|s| s.len()).sum()
    }

    /// Return the in-degree of a node.
    pub fn in_degree(&self, id: TopoNodeId) -> usize {
        self.reverse.get(&id).map_or(0, |s| s.len())
    }

    /// Return the out-degree of a node.
    pub fn out_degree(&self, id: TopoNodeId) -> usize {
        self.adjacency.get(&id).map_or(0, |s| s.len())
    }

    /// Return all nodes with in-degree zero (source nodes).
    pub fn sources(&self) -> Vec<TopoNodeId> {
        let mut sources: Vec<TopoNodeId> = self
            .adjacency
            .keys()
            .filter(|id| self.in_degree(**id) == 0)
            .copied()
            .collect();
        sources.sort();
        sources
    }

    /// Return all nodes with out-degree zero (sink nodes).
    pub fn sinks(&self) -> Vec<TopoNodeId> {
        let mut sinks: Vec<TopoNodeId> = self
            .adjacency
            .keys()
            .filter(|id| self.out_degree(**id) == 0)
            .copied()
            .collect();
        sinks.sort();
        sinks
    }

    /// Perform topological sort using Kahn's algorithm (BFS-based).
    ///
    /// Returns nodes in topological order or an error if a cycle exists.
    pub fn sort_kahn(&self) -> Result<Vec<TopoNodeId>, TopoError> {
        if self.adjacency.is_empty() {
            return Err(TopoError::EmptyGraph);
        }

        let mut in_degrees: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in self.adjacency.keys() {
            in_degrees.insert(node, self.in_degree(node));
        }

        let mut queue: VecDeque<TopoNodeId> = in_degrees
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort queue for deterministic output
        let mut sorted_start: Vec<TopoNodeId> = queue.drain(..).collect();
        sorted_start.sort();
        queue.extend(sorted_start);

        let mut result = Vec::with_capacity(self.adjacency.len());

        while let Some(node) = queue.pop_front() {
            result.push(node);
            if let Some(successors) = self.adjacency.get(&node) {
                let mut sorted_succ: Vec<TopoNodeId> = successors.iter().copied().collect();
                sorted_succ.sort();
                for succ in sorted_succ {
                    if let Some(deg) = in_degrees.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if result.len() != self.adjacency.len() {
            let remaining: Vec<TopoNodeId> = self
                .adjacency
                .keys()
                .filter(|id| !result.contains(id))
                .copied()
                .collect();
            return Err(TopoError::CycleDetected(remaining));
        }

        Ok(result)
    }

    /// Perform topological sort using DFS-based algorithm.
    ///
    /// Returns nodes in topological order or an error if a cycle exists.
    pub fn sort_dfs(&self) -> Result<Vec<TopoNodeId>, TopoError> {
        if self.adjacency.is_empty() {
            return Err(TopoError::EmptyGraph);
        }

        let mut visited: HashSet<TopoNodeId> = HashSet::new();
        let mut in_stack: HashSet<TopoNodeId> = HashSet::new();
        let mut result: Vec<TopoNodeId> = Vec::new();

        let mut nodes: Vec<TopoNodeId> = self.adjacency.keys().copied().collect();
        nodes.sort();

        for node in &nodes {
            if !visited.contains(node)
                && !Self::dfs_visit(
                    *node,
                    &self.adjacency,
                    &mut visited,
                    &mut in_stack,
                    &mut result,
                )
            {
                let cycle_nodes: Vec<TopoNodeId> = in_stack.into_iter().collect();
                return Err(TopoError::CycleDetected(cycle_nodes));
            }
        }

        result.reverse();
        Ok(result)
    }

    /// DFS visit helper. Returns false if a cycle is detected.
    fn dfs_visit(
        node: TopoNodeId,
        adjacency: &HashMap<TopoNodeId, HashSet<TopoNodeId>>,
        visited: &mut HashSet<TopoNodeId>,
        in_stack: &mut HashSet<TopoNodeId>,
        result: &mut Vec<TopoNodeId>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        if let Some(successors) = adjacency.get(&node) {
            let mut sorted_succ: Vec<TopoNodeId> = successors.iter().copied().collect();
            sorted_succ.sort();
            for succ in sorted_succ {
                if in_stack.contains(&succ) {
                    return false;
                }
                if !visited.contains(&succ)
                    && !Self::dfs_visit(succ, adjacency, visited, in_stack, result)
                {
                    return false;
                }
            }
        }

        in_stack.remove(&node);
        result.push(node);
        true
    }

    /// Check if the graph is a DAG (has no cycles).
    pub fn is_dag(&self) -> bool {
        self.sort_kahn().is_ok()
    }

    /// Return the longest path length in the DAG.
    pub fn longest_path(&self) -> Result<usize, TopoError> {
        let order = self.sort_kahn()?;
        let mut dist: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in &order {
            dist.insert(node, 0);
        }

        for &node in &order {
            let node_dist = dist[&node];
            if let Some(successors) = self.adjacency.get(&node) {
                for &succ in successors {
                    let entry = dist.entry(succ).or_insert(0);
                    if node_dist + 1 > *entry {
                        *entry = node_dist + 1;
                    }
                }
            }
        }

        Ok(dist.values().copied().max().unwrap_or(0))
    }

    /// Return the depth (longest path from any source) for each node.
    pub fn node_depths(&self) -> Result<HashMap<TopoNodeId, usize>, TopoError> {
        let order = self.sort_kahn()?;
        let mut depths: HashMap<TopoNodeId, usize> = HashMap::new();
        for &node in &order {
            depths.insert(node, 0);
        }

        for &node in &order {
            let node_depth = depths[&node];
            if let Some(successors) = self.adjacency.get(&node) {
                for &succ in successors {
                    let entry = depths.entry(succ).or_insert(0);
                    if node_depth + 1 > *entry {
                        *entry = node_depth + 1;
                    }
                }
            }
        }

        Ok(depths)
    }

    /// Check if node `a` can reach node `b` (transitively).
    pub fn can_reach(&self, a: TopoNodeId, b: TopoNodeId) -> bool {
        let mut visited: HashSet<TopoNodeId> = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(a);

        while let Some(current) = queue.pop_front() {
            if current == b {
                return true;
            }
            if visited.insert(current) {
                if let Some(successors) = self.adjacency.get(&current) {
                    for &succ in successors {
                        queue.push_back(succ);
                    }
                }
            }
        }

        false
    }
}

impl Default for TopoGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(id: usize) -> TopoNodeId {
        TopoNodeId(id)
    }

    #[test]
    fn test_empty_graph() {
        let graph = TopoGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert!(matches!(graph.sort_kahn(), Err(TopoError::EmptyGraph)));
    }

    #[test]
    fn test_single_node() {
        let mut graph = TopoGraph::new();
        graph.add_node(n(0));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order, vec![n(0)]);
    }

    #[test]
    fn test_linear_chain() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(3));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order, vec![n(0), n(1), n(2), n(3)]);
    }

    #[test]
    fn test_diamond_graph() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let order = graph.sort_kahn().expect("sort_kahn should succeed");
        assert_eq!(order[0], n(0));
        assert_eq!(order[3], n(3));
    }

    #[test]
    fn test_cycle_detection_kahn() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(0));
        let result = graph.sort_kahn();
        assert!(matches!(result, Err(TopoError::CycleDetected(_))));
    }

    #[test]
    fn test_cycle_detection_dfs() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(0));
        let result = graph.sort_dfs();
        assert!(matches!(result, Err(TopoError::CycleDetected(_))));
    }

    #[test]
    fn test_dfs_sort_matches_kahn() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let kahn = graph.sort_kahn().expect("sort_kahn should succeed");
        let dfs = graph.sort_dfs().expect("sort_dfs should succeed");
        // Both should have 0 first and 3 last
        assert_eq!(kahn[0], n(0));
        assert_eq!(dfs[0], n(0));
        assert_eq!(*kahn.last().expect("last should succeed"), n(3));
        assert_eq!(*dfs.last().expect("last should succeed"), n(3));
    }

    #[test]
    fn test_sources_and_sinks() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(2), n(3));
        graph.add_edge(n(2), n(4));
        assert_eq!(graph.sources(), vec![n(0), n(1)]);
        assert_eq!(graph.sinks(), vec![n(3), n(4)]);
    }

    #[test]
    fn test_in_out_degree() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(2));
        assert_eq!(graph.out_degree(n(0)), 2);
        assert_eq!(graph.in_degree(n(2)), 2);
        assert_eq!(graph.in_degree(n(0)), 0);
    }

    #[test]
    fn test_is_dag() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        assert!(graph.is_dag());

        graph.add_edge(n(2), n(0));
        assert!(!graph.is_dag());
    }

    #[test]
    fn test_longest_path() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(0), n(2));
        assert_eq!(
            graph.longest_path().expect("longest_path should succeed"),
            2
        );
    }

    #[test]
    fn test_node_depths() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(0), n(2));
        graph.add_edge(n(1), n(3));
        graph.add_edge(n(2), n(3));
        let depths = graph.node_depths().expect("node_depths should succeed");
        assert_eq!(depths[&n(0)], 0);
        assert_eq!(depths[&n(3)], 2);
    }

    #[test]
    fn test_can_reach() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        assert!(graph.can_reach(n(0), n(2)));
        assert!(!graph.can_reach(n(2), n(0)));
    }

    #[test]
    fn test_topo_error_display() {
        let err = TopoError::EmptyGraph;
        assert_eq!(format!("{err}"), "Graph is empty");
        let err2 = TopoError::NodeNotFound(n(5));
        assert!(format!("{err2}").contains("5"));
    }

    #[test]
    fn test_edge_count() {
        let mut graph = TopoGraph::new();
        graph.add_edge(n(0), n(1));
        graph.add_edge(n(1), n(2));
        graph.add_edge(n(0), n(2));
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_node_id_display() {
        let id = TopoNodeId(42);
        assert_eq!(format!("{id}"), "Node(42)");
    }
}
