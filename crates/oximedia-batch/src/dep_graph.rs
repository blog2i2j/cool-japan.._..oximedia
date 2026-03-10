//! Simple dependency graph for batch jobs using numeric IDs.
//!
//! Provides a directed acyclic graph (DAG) with Kahn's algorithm for
//! topological sorting and a `ReadyQueue` that surfaces tasks whose
//! dependencies have all been completed.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// A directed graph where an edge `(from, to)` means *"to depends on from"*.
#[derive(Debug, Default, Clone)]
pub struct DepGraph {
    /// All registered node IDs.
    pub nodes: Vec<u64>,
    /// Directed edges: `(from, to)` — "to depends on from".
    pub edges: Vec<(u64, u64)>,
}

impl DepGraph {
    /// Create an empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a node. No-op if the node already exists.
    pub fn add_node(&mut self, id: u64) {
        if !self.nodes.contains(&id) {
            self.nodes.push(id);
        }
    }

    /// Add a dependency edge: `to` depends on `from`.
    ///
    /// Automatically registers both nodes if missing.
    pub fn add_dependency(&mut self, from: u64, to: u64) {
        self.add_node(from);
        self.add_node(to);
        let edge = (from, to);
        if !self.edges.contains(&edge) {
            self.edges.push(edge);
        }
    }

    /// Return all nodes that `id` depends on (its direct predecessors).
    #[must_use]
    pub fn dependencies_of(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter_map(|&(from, to)| if to == id { Some(from) } else { None })
            .collect()
    }

    /// Return all nodes that depend on `id` (its direct successors).
    #[must_use]
    pub fn dependents_of(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter_map(|&(from, to)| if from == id { Some(to) } else { None })
            .collect()
    }
}

/// Perform a topological sort using Kahn's algorithm.
///
/// # Errors
///
/// Returns `Err(String)` if the graph contains a cycle.
pub fn topological_sort(graph: &DepGraph) -> Result<Vec<u64>, String> {
    // Build in-degree map
    let mut in_degree: HashMap<u64, usize> = graph.nodes.iter().map(|&n| (n, 0)).collect();
    for &(_, to) in &graph.edges {
        *in_degree.entry(to).or_insert(0) += 1;
    }

    // Start with nodes that have no dependencies
    let mut queue: VecDeque<u64> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&n, _)| n)
        .collect();

    // Sort for deterministic output
    let mut queue_vec: Vec<u64> = queue.drain(..).collect();
    queue_vec.sort_unstable();
    queue.extend(queue_vec);

    let mut order = Vec::with_capacity(graph.nodes.len());

    while let Some(node) = queue.pop_front() {
        order.push(node);
        let mut successors: Vec<u64> = graph
            .edges
            .iter()
            .filter_map(|&(from, to)| if from == node { Some(to) } else { None })
            .collect();
        successors.sort_unstable();
        for succ in successors {
            let deg = in_degree.entry(succ).or_insert(0);
            *deg -= 1;
            if *deg == 0 {
                queue.push_back(succ);
            }
        }
    }

    if order.len() == graph.nodes.len() {
        Ok(order)
    } else {
        Err("Cycle detected in dependency graph".to_string())
    }
}

/// Return `true` if the graph contains at least one cycle.
#[must_use]
pub fn has_cycle(graph: &DepGraph) -> bool {
    topological_sort(graph).is_err()
}

/// A queue of tasks that automatically surfaces those whose dependencies are met.
#[derive(Debug)]
pub struct ReadyQueue {
    /// Underlying dependency graph.
    pub graph: DepGraph,
    /// IDs of tasks that have been marked complete.
    pub completed: Vec<u64>,
}

impl ReadyQueue {
    /// Create a new ready queue from a dependency graph.
    #[must_use]
    pub fn new(graph: DepGraph) -> Self {
        Self {
            graph,
            completed: Vec::new(),
        }
    }

    /// Mark a task as completed.
    pub fn mark_complete(&mut self, id: u64) {
        if !self.completed.contains(&id) {
            self.completed.push(id);
        }
    }

    /// Return all tasks that are ready to run:
    /// every dependency is in `completed` and the task itself is not yet completed.
    #[must_use]
    pub fn ready_tasks(&self) -> Vec<u64> {
        let completed_set: HashSet<u64> = self.completed.iter().copied().collect();
        let mut ready: Vec<u64> = self
            .graph
            .nodes
            .iter()
            .filter(|&&id| {
                if completed_set.contains(&id) {
                    return false;
                }
                let deps = self.graph.dependencies_of(id);
                deps.iter().all(|dep| completed_set.contains(dep))
            })
            .copied()
            .collect();
        ready.sort_unstable();
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_graph() -> DepGraph {
        // 1 -> 2 -> 3  (3 depends on 2, 2 depends on 1)
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        g
    }

    #[test]
    fn test_add_node_no_duplicate() {
        let mut g = DepGraph::new();
        g.add_node(1);
        g.add_node(1);
        assert_eq!(g.nodes.len(), 1);
    }

    #[test]
    fn test_add_dependency_registers_nodes() {
        let mut g = DepGraph::new();
        g.add_dependency(10, 20);
        assert!(g.nodes.contains(&10));
        assert!(g.nodes.contains(&20));
    }

    #[test]
    fn test_dependencies_of() {
        let g = linear_graph();
        assert_eq!(g.dependencies_of(2), vec![1]);
        assert_eq!(g.dependencies_of(3), vec![2]);
        assert!(g.dependencies_of(1).is_empty());
    }

    #[test]
    fn test_dependents_of() {
        let g = linear_graph();
        assert_eq!(g.dependents_of(1), vec![2]);
        assert_eq!(g.dependents_of(2), vec![3]);
        assert!(g.dependents_of(3).is_empty());
    }

    #[test]
    fn test_topological_sort_linear() {
        let g = linear_graph();
        let order = topological_sort(&g).expect("operation should succeed");
        assert_eq!(order, vec![1, 2, 3]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        // 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 4);
        g.add_dependency(3, 4);
        let order = topological_sort(&g).expect("operation should succeed");
        assert_eq!(order[0], 1);
        assert_eq!(*order.last().expect("should have last element"), 4);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_cycle_returns_err() {
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        g.add_dependency(3, 1); // cycle
        assert!(topological_sort(&g).is_err());
    }

    #[test]
    fn test_has_cycle_true() {
        let mut g = DepGraph::new();
        g.add_dependency(5, 6);
        g.add_dependency(6, 5);
        assert!(has_cycle(&g));
    }

    #[test]
    fn test_has_cycle_false() {
        let g = linear_graph();
        assert!(!has_cycle(&g));
    }

    #[test]
    fn test_ready_queue_initial_no_deps() {
        let mut g = DepGraph::new();
        g.add_node(1);
        g.add_node(2);
        g.add_dependency(1, 2);
        let rq = ReadyQueue::new(g);
        assert_eq!(rq.ready_tasks(), vec![1]);
    }

    #[test]
    fn test_ready_queue_after_complete() {
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        let mut rq = ReadyQueue::new(g);
        assert_eq!(rq.ready_tasks(), vec![1]);
        rq.mark_complete(1);
        assert_eq!(rq.ready_tasks(), vec![2]);
        rq.mark_complete(2);
        assert_eq!(rq.ready_tasks(), vec![3]);
        rq.mark_complete(3);
        assert!(rq.ready_tasks().is_empty());
    }

    #[test]
    fn test_ready_queue_mark_complete_idempotent() {
        let mut g = DepGraph::new();
        g.add_node(7);
        let mut rq = ReadyQueue::new(g);
        rq.mark_complete(7);
        rq.mark_complete(7);
        assert_eq!(rq.completed.len(), 1);
    }
}
