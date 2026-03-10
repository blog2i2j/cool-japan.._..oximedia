//! Job dependency graph with topological sort and cycle detection.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the dependency graph representing a single job.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobNode {
    /// Unique job identifier.
    pub job_id: u64,
    /// Human-readable job name.
    pub name: String,
}

impl JobNode {
    /// Create a new `JobNode`.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>) -> Self {
        Self {
            job_id: id,
            name: name.into(),
        }
    }
}

/// Directed dependency graph.
///
/// An edge `(from, to)` means *to depends on from*, i.e. `from` must complete before `to` runs.
#[derive(Clone, Debug, Default)]
pub struct DepGraph {
    /// Registered job nodes.
    pub nodes: Vec<JobNode>,
    /// Directed edges: `(from_id, to_id)`.
    pub edges: Vec<(u64, u64)>,
}

impl DepGraph {
    /// Create an empty `DepGraph`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a job node.  Duplicate ids are allowed but not recommended.
    pub fn add_node(&mut self, node: JobNode) {
        self.nodes.push(node);
    }

    /// Add a dependency edge: `to` depends on `from`.
    pub fn add_dependency(&mut self, from: u64, to: u64) {
        self.edges.push((from, to));
    }

    /// Return the ids of all jobs that `id` *directly* depends on.
    #[must_use]
    pub fn dependencies_of(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|&&(_, to)| to == id)
            .map(|&(from, _)| from)
            .collect()
    }

    /// Return the ids of all jobs that *directly* depend on `id`.
    #[must_use]
    pub fn dependents_of(&self, id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|&&(from, _)| from == id)
            .map(|&(_, to)| to)
            .collect()
    }

    /// Return the ids of all *root* nodes — nodes that have no dependencies.
    #[must_use]
    pub fn roots(&self) -> Vec<u64> {
        let has_incoming: HashSet<u64> = self.edges.iter().map(|&(_, to)| to).collect();
        self.nodes
            .iter()
            .filter(|n| !has_incoming.contains(&n.job_id))
            .map(|n| n.job_id)
            .collect()
    }

    /// Return a topologically sorted list of job ids using Kahn's algorithm.
    ///
    /// Returns all node ids in an order where dependencies come before dependents.
    /// If the graph contains a cycle, only the nodes that can be reached from roots
    /// are returned (the remaining form the cycle — use `has_cycle` to check).
    #[must_use]
    pub fn topological_sort(&self) -> Vec<u64> {
        // Build in-degree map and adjacency list.
        let mut in_degree: HashMap<u64, usize> = self.nodes.iter().map(|n| (n.job_id, 0)).collect();
        let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();

        for &(from, to) in &self.edges {
            *in_degree.entry(to).or_insert(0) += 1;
            adj.entry(from).or_default().push(to);
        }

        let mut queue: VecDeque<u64> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Stable ordering: sort the initial queue for determinism.
        let mut queue_vec: Vec<u64> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut result = Vec::new();

        while let Some(id) = queue.pop_front() {
            result.push(id);
            if let Some(neighbours) = adj.get(&id) {
                let mut sorted = neighbours.clone();
                sorted.sort_unstable();
                for next in sorted {
                    let deg = in_degree.entry(next).or_insert(0);
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(next);
                    }
                }
            }
        }
        result
    }

    /// Returns `true` if the graph contains at least one cycle.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        self.topological_sort().len() != self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_linear_graph() -> DepGraph {
        // 1 -> 2 -> 3
        let mut g = DepGraph::new();
        g.add_node(JobNode::new(1, "A"));
        g.add_node(JobNode::new(2, "B"));
        g.add_node(JobNode::new(3, "C"));
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        g
    }

    // ---------- JobNode tests ----------

    #[test]
    fn test_job_node_new() {
        let n = JobNode::new(42, "encode");
        assert_eq!(n.job_id, 42);
        assert_eq!(n.name, "encode");
    }

    #[test]
    fn test_job_node_string_name() {
        let n = JobNode::new(1, String::from("transcode"));
        assert_eq!(n.name, "transcode");
    }

    // ---------- DepGraph add / query tests ----------

    #[test]
    fn test_add_node_and_edge() {
        let g = build_linear_graph();
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.edges.len(), 2);
    }

    #[test]
    fn test_dependencies_of() {
        let g = build_linear_graph();
        let deps = g.dependencies_of(2);
        assert_eq!(deps, vec![1]);
    }

    #[test]
    fn test_dependencies_of_root() {
        let g = build_linear_graph();
        assert!(g.dependencies_of(1).is_empty());
    }

    #[test]
    fn test_dependents_of() {
        let g = build_linear_graph();
        let deps = g.dependents_of(1);
        assert_eq!(deps, vec![2]);
    }

    #[test]
    fn test_dependents_of_leaf() {
        let g = build_linear_graph();
        assert!(g.dependents_of(3).is_empty());
    }

    #[test]
    fn test_roots_linear() {
        let g = build_linear_graph();
        let roots = g.roots();
        assert_eq!(roots, vec![1]);
    }

    #[test]
    fn test_roots_multiple() {
        let mut g = DepGraph::new();
        g.add_node(JobNode::new(1, "A"));
        g.add_node(JobNode::new(2, "B"));
        g.add_node(JobNode::new(3, "C"));
        g.add_dependency(1, 3);
        g.add_dependency(2, 3);
        let mut roots = g.roots();
        roots.sort();
        assert_eq!(roots, vec![1, 2]);
    }

    // ---------- topological_sort tests ----------

    #[test]
    fn test_topological_sort_linear() {
        let g = build_linear_graph();
        assert_eq!(g.topological_sort(), vec![1, 2, 3]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        // 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        let mut g = DepGraph::new();
        for i in 1..=4 {
            g.add_node(JobNode::new(i, format!("job-{i}")));
        }
        g.add_dependency(1, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 4);
        g.add_dependency(3, 4);
        let order = g.topological_sort();
        assert_eq!(order[0], 1);
        assert_eq!(*order.last().expect("last should succeed"), 4);
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_topological_sort_independent() {
        let mut g = DepGraph::new();
        g.add_node(JobNode::new(10, "X"));
        g.add_node(JobNode::new(20, "Y"));
        let order = g.topological_sort();
        assert_eq!(order.len(), 2);
    }

    // ---------- has_cycle tests ----------

    #[test]
    fn test_has_cycle_false() {
        let g = build_linear_graph();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_has_cycle_true() {
        let mut g = DepGraph::new();
        g.add_node(JobNode::new(1, "A"));
        g.add_node(JobNode::new(2, "B"));
        g.add_dependency(1, 2);
        g.add_dependency(2, 1); // cycle
        assert!(g.has_cycle());
    }

    #[test]
    fn test_empty_graph_no_cycle() {
        let g = DepGraph::new();
        assert!(!g.has_cycle());
    }
}
