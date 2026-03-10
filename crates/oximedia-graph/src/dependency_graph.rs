#![allow(dead_code)]
//! Dependency analysis for graph nodes.
//!
//! This module provides tools for analyzing dependencies between nodes
//! in a filter graph, computing execution order, detecting critical paths,
//! and identifying parallelizable groups.

use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for a dependency node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepNodeId(pub u64);

impl std::fmt::Display for DepNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "node_{}", self.0)
    }
}

/// A node in the dependency graph.
#[derive(Debug, Clone)]
pub struct DepNode {
    /// Unique identifier.
    pub id: DepNodeId,
    /// Human-readable label.
    pub label: String,
    /// Estimated execution cost (arbitrary units).
    pub cost: f64,
}

impl DepNode {
    /// Create a new dependency node.
    pub fn new(id: u64, label: &str, cost: f64) -> Self {
        Self {
            id: DepNodeId(id),
            label: label.to_string(),
            cost,
        }
    }
}

/// A directed dependency graph for analyzing execution order.
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// All nodes, keyed by their ID.
    nodes: HashMap<DepNodeId, DepNode>,
    /// Adjacency list: node -> set of dependents (successors).
    forward_edges: HashMap<DepNodeId, HashSet<DepNodeId>>,
    /// Reverse adjacency: node -> set of dependencies (predecessors).
    reverse_edges: HashMap<DepNodeId, HashSet<DepNodeId>>,
}

impl DependencyGraph {
    /// Create an empty dependency graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: DepNode) {
        let id = node.id;
        self.nodes.insert(id, node);
        self.forward_edges.entry(id).or_default();
        self.reverse_edges.entry(id).or_default();
    }

    /// Add a dependency edge: `from` must complete before `to`.
    ///
    /// Returns `true` if the edge was newly added.
    pub fn add_edge(&mut self, from: DepNodeId, to: DepNodeId) -> bool {
        self.forward_edges.entry(from).or_default().insert(to);
        self.reverse_edges.entry(to).or_default().insert(from)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.forward_edges.values().map(|s| s.len()).sum()
    }

    /// Get the direct dependencies (predecessors) of a node.
    pub fn dependencies_of(&self, id: DepNodeId) -> Vec<DepNodeId> {
        self.reverse_edges
            .get(&id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Get the direct dependents (successors) of a node.
    pub fn dependents_of(&self, id: DepNodeId) -> Vec<DepNodeId> {
        self.forward_edges
            .get(&id)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Find root nodes (no dependencies).
    pub fn roots(&self) -> Vec<DepNodeId> {
        self.nodes
            .keys()
            .filter(|id| self.reverse_edges.get(id).map_or(true, HashSet::is_empty))
            .copied()
            .collect()
    }

    /// Find leaf nodes (no dependents).
    pub fn leaves(&self) -> Vec<DepNodeId> {
        self.nodes
            .keys()
            .filter(|id| self.forward_edges.get(id).map_or(true, HashSet::is_empty))
            .copied()
            .collect()
    }

    /// Compute topological ordering using Kahn's algorithm.
    ///
    /// Returns `None` if the graph has a cycle.
    pub fn topological_order(&self) -> Option<Vec<DepNodeId>> {
        let mut in_degree: HashMap<DepNodeId, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_degree.insert(*id, self.reverse_edges.get(id).map_or(0, HashSet::len));
        }

        let mut queue: VecDeque<DepNodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(id, _)| *id)
            .collect();

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(successors) = self.forward_edges.get(&node) {
                for &succ in successors {
                    if let Some(deg) = in_degree.get_mut(&succ) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(succ);
                        }
                    }
                }
            }
        }

        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None
        }
    }

    /// Compute all transitive dependencies of a node.
    pub fn transitive_dependencies(&self, id: DepNodeId) -> HashSet<DepNodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![id];
        while let Some(current) = stack.pop() {
            if let Some(deps) = self.reverse_edges.get(&current) {
                for &dep in deps {
                    if visited.insert(dep) {
                        stack.push(dep);
                    }
                }
            }
        }
        visited
    }

    /// Compute the depth of each node (longest path from a root).
    pub fn compute_depths(&self) -> HashMap<DepNodeId, u32> {
        let mut depths: HashMap<DepNodeId, u32> = HashMap::new();
        if let Some(order) = self.topological_order() {
            for &node in &order {
                let max_pred_depth = self
                    .reverse_edges
                    .get(&node)
                    .map(|preds| {
                        preds
                            .iter()
                            .filter_map(|p| depths.get(p))
                            .max()
                            .copied()
                            .unwrap_or(0)
                    })
                    .unwrap_or(0);
                let depth = if self
                    .reverse_edges
                    .get(&node)
                    .map_or(true, HashSet::is_empty)
                {
                    0
                } else {
                    max_pred_depth + 1
                };
                depths.insert(node, depth);
            }
        }
        depths
    }

    /// Group nodes into parallelizable levels.
    ///
    /// Nodes at the same level have no inter-dependencies and can run concurrently.
    pub fn parallel_levels(&self) -> Vec<Vec<DepNodeId>> {
        let depths = self.compute_depths();
        if depths.is_empty() {
            return Vec::new();
        }
        let max_depth = depths.values().copied().max().unwrap_or(0);
        let mut levels = vec![Vec::new(); (max_depth + 1) as usize];
        for (id, depth) in &depths {
            levels[*depth as usize].push(*id);
        }
        levels
    }

    /// Compute the critical path (longest weighted path through the graph).
    #[allow(clippy::cast_precision_loss)]
    pub fn critical_path(&self) -> (Vec<DepNodeId>, f64) {
        let order = match self.topological_order() {
            Some(o) => o,
            None => return (Vec::new(), 0.0),
        };

        let mut dist: HashMap<DepNodeId, f64> = HashMap::new();
        let mut prev: HashMap<DepNodeId, DepNodeId> = HashMap::new();

        for &node in &order {
            let node_cost = self.nodes.get(&node).map_or(0.0, |n| n.cost);
            let max_pred = self.reverse_edges.get(&node).and_then(|preds| {
                preds
                    .iter()
                    .filter_map(|p| dist.get(p).map(|d| (*p, *d)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });

            let total = if let Some((pred_id, pred_dist)) = max_pred {
                prev.insert(node, pred_id);
                pred_dist + node_cost
            } else {
                node_cost
            };
            dist.insert(node, total);
        }

        // Find the node with maximum distance
        let end_node = dist
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| *id);

        let end_node = match end_node {
            Some(n) => n,
            None => return (Vec::new(), 0.0),
        };

        let total_cost = dist[&end_node];

        // Trace back the path
        let mut path = vec![end_node];
        let mut current = end_node;
        while let Some(&pred) = prev.get(&current) {
            path.push(pred);
            current = pred;
        }
        path.reverse();

        (path, total_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_linear_graph() -> DependencyGraph {
        let mut g = DependencyGraph::new();
        g.add_node(DepNode::new(0, "A", 1.0));
        g.add_node(DepNode::new(1, "B", 2.0));
        g.add_node(DepNode::new(2, "C", 3.0));
        g.add_edge(DepNodeId(0), DepNodeId(1));
        g.add_edge(DepNodeId(1), DepNodeId(2));
        g
    }

    fn make_diamond_graph() -> DependencyGraph {
        // A -> B, A -> C, B -> D, C -> D
        let mut g = DependencyGraph::new();
        g.add_node(DepNode::new(0, "A", 1.0));
        g.add_node(DepNode::new(1, "B", 2.0));
        g.add_node(DepNode::new(2, "C", 4.0));
        g.add_node(DepNode::new(3, "D", 1.0));
        g.add_edge(DepNodeId(0), DepNodeId(1));
        g.add_edge(DepNodeId(0), DepNodeId(2));
        g.add_edge(DepNodeId(1), DepNodeId(3));
        g.add_edge(DepNodeId(2), DepNodeId(3));
        g
    }

    #[test]
    fn test_add_node_and_edge() {
        let g = make_linear_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_roots_and_leaves() {
        let g = make_linear_graph();
        let roots = g.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0], DepNodeId(0));
        let leaves = g.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0], DepNodeId(2));
    }

    #[test]
    fn test_dependencies_of() {
        let g = make_linear_graph();
        let deps = g.dependencies_of(DepNodeId(2));
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], DepNodeId(1));
    }

    #[test]
    fn test_dependents_of() {
        let g = make_linear_graph();
        let deps = g.dependents_of(DepNodeId(0));
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], DepNodeId(1));
    }

    #[test]
    fn test_topological_order() {
        let g = make_linear_graph();
        let order = g
            .topological_order()
            .expect("topological_order should succeed");
        assert_eq!(order.len(), 3);
        // A before B, B before C
        let pos_a = order
            .iter()
            .position(|&x| x == DepNodeId(0))
            .expect("iter should succeed");
        let pos_b = order
            .iter()
            .position(|&x| x == DepNodeId(1))
            .expect("iter should succeed");
        let pos_c = order
            .iter()
            .position(|&x| x == DepNodeId(2))
            .expect("iter should succeed");
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_topological_order_diamond() {
        let g = make_diamond_graph();
        let order = g
            .topological_order()
            .expect("topological_order should succeed");
        assert_eq!(order.len(), 4);
        let pos_a = order
            .iter()
            .position(|&x| x == DepNodeId(0))
            .expect("iter should succeed");
        let pos_d = order
            .iter()
            .position(|&x| x == DepNodeId(3))
            .expect("iter should succeed");
        assert!(pos_a < pos_d);
    }

    #[test]
    fn test_transitive_dependencies() {
        let g = make_linear_graph();
        let trans = g.transitive_dependencies(DepNodeId(2));
        assert!(trans.contains(&DepNodeId(0)));
        assert!(trans.contains(&DepNodeId(1)));
        assert_eq!(trans.len(), 2);
    }

    #[test]
    fn test_compute_depths() {
        let g = make_linear_graph();
        let depths = g.compute_depths();
        assert_eq!(depths[&DepNodeId(0)], 0);
        assert_eq!(depths[&DepNodeId(1)], 1);
        assert_eq!(depths[&DepNodeId(2)], 2);
    }

    #[test]
    fn test_parallel_levels_diamond() {
        let g = make_diamond_graph();
        let levels = g.parallel_levels();
        assert_eq!(levels.len(), 3);
        // Level 0: A, Level 1: B and C, Level 2: D
        assert_eq!(levels[0].len(), 1);
        assert_eq!(levels[1].len(), 2);
        assert_eq!(levels[2].len(), 1);
    }

    #[test]
    fn test_critical_path_linear() {
        let g = make_linear_graph();
        let (path, cost) = g.critical_path();
        assert_eq!(path.len(), 3);
        assert!((cost - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_critical_path_diamond() {
        let g = make_diamond_graph();
        let (path, cost) = g.critical_path();
        // Critical path: A(1) -> C(4) -> D(1) = 6.0
        assert!((cost - 6.0).abs() < f64::EPSILON);
        assert!(path.contains(&DepNodeId(0)));
        assert!(path.contains(&DepNodeId(3)));
    }

    #[test]
    fn test_empty_graph() {
        let g = DependencyGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.roots().is_empty());
        assert!(g.leaves().is_empty());
        let (path, cost) = g.critical_path();
        assert!(path.is_empty());
        assert!((cost - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dep_node_id_display() {
        let id = DepNodeId(42);
        assert_eq!(format!("{id}"), "node_42");
    }
}
