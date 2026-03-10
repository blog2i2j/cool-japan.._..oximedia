//! Job dependency graph with topological sort and cycle detection.
//!
//! This module provides a directed acyclic graph (DAG) for managing job
//! dependencies, including topological ordering and cycle detection.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{BatchError, Result};
use crate::types::JobId;

/// Represents a directed acyclic graph of job dependencies.
///
/// Jobs are nodes; an edge from A to B means "A must complete before B".
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Adjacency list: job -> set of jobs that depend on it (successors).
    successors: HashMap<String, HashSet<String>>,
    /// Reverse adjacency: job -> set of jobs it depends on (predecessors).
    predecessors: HashMap<String, HashSet<String>>,
}

impl DependencyGraph {
    /// Create a new, empty dependency graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a job node (idempotent).
    pub fn add_job(&mut self, job_id: &JobId) {
        let key = job_id.as_str().to_string();
        self.successors.entry(key.clone()).or_default();
        self.predecessors.entry(key).or_default();
    }

    /// Add a dependency edge: `from` must complete before `to`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::DependencyError`] if both nodes are the same.
    pub fn add_dependency(&mut self, from: &JobId, to: &JobId) -> Result<()> {
        if from == to {
            return Err(BatchError::DependencyError(format!(
                "Self-dependency not allowed: {from}"
            )));
        }
        let f = from.as_str().to_string();
        let t = to.as_str().to_string();
        self.successors
            .entry(f.clone())
            .or_default()
            .insert(t.clone());
        self.successors.entry(t.clone()).or_default();
        self.predecessors.entry(t).or_default().insert(f.clone());
        self.predecessors.entry(f).or_default();
        Ok(())
    }

    /// Remove a job and all edges connected to it.
    pub fn remove_job(&mut self, job_id: &JobId) {
        let key = job_id.as_str();
        if let Some(succs) = self.successors.remove(key) {
            for s in &succs {
                if let Some(preds) = self.predecessors.get_mut(s) {
                    preds.remove(key);
                }
            }
        }
        if let Some(preds) = self.predecessors.remove(key) {
            for p in &preds {
                if let Some(succs) = self.successors.get_mut(p) {
                    succs.remove(key);
                }
            }
        }
    }

    /// Detect whether the graph contains a cycle.
    ///
    /// Uses DFS coloring (white / gray / black).
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut color: HashMap<&str, u8> = HashMap::new(); // 0=white,1=gray,2=black
        for node in self.successors.keys() {
            if color.get(node.as_str()).copied().unwrap_or(0) == 0
                && self.dfs_cycle(node.as_str(), &mut color)
            {
                return true;
            }
        }
        false
    }

    fn dfs_cycle<'a>(&'a self, node: &'a str, color: &mut HashMap<&'a str, u8>) -> bool {
        color.insert(node, 1); // gray
        if let Some(succs) = self.successors.get(node) {
            for s in succs {
                let c = color.get(s.as_str()).copied().unwrap_or(0);
                if c == 1 {
                    return true; // back edge
                }
                if c == 0 && self.dfs_cycle(s.as_str(), color) {
                    return true;
                }
            }
        }
        color.insert(node, 2); // black
        false
    }

    /// Return a topological ordering of all jobs.
    ///
    /// Uses Kahn's algorithm (BFS-based).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::DependencyError`] if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for node in self.successors.keys() {
            in_degree.entry(node.as_str()).or_insert(0);
        }
        for (node, preds) in &self.predecessors {
            *in_degree.entry(node.as_str()).or_insert(0) = preds.len();
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(succs) = self.successors.get(node) {
                for s in succs {
                    let deg = in_degree.entry(s.as_str()).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push_back(s.as_str());
                    }
                }
            }
        }

        if order.len() != self.successors.len() {
            return Err(BatchError::DependencyError(
                "Cycle detected in dependency graph".to_string(),
            ));
        }

        Ok(order)
    }

    /// Return direct predecessors (dependencies) of a job.
    #[must_use]
    pub fn predecessors_of(&self, job_id: &JobId) -> HashSet<String> {
        self.predecessors
            .get(job_id.as_str())
            .cloned()
            .unwrap_or_default()
    }

    /// Return direct successors (dependents) of a job.
    #[must_use]
    pub fn successors_of(&self, job_id: &JobId) -> HashSet<String> {
        self.successors
            .get(job_id.as_str())
            .cloned()
            .unwrap_or_default()
    }

    /// Return the total number of jobs (nodes) in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.successors.len()
    }

    /// Return the total number of dependency edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.successors.values().map(HashSet::len).sum()
    }

    /// Check whether a job is in the graph.
    #[must_use]
    pub fn contains(&self, job_id: &JobId) -> bool {
        self.successors.contains_key(job_id.as_str())
    }

    /// Return jobs with no predecessors (ready to run immediately).
    #[must_use]
    pub fn roots(&self) -> Vec<String> {
        self.predecessors
            .iter()
            .filter(|(_, preds)| preds.is_empty())
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Return jobs with no successors (leaf nodes).
    #[must_use]
    pub fn leaves(&self) -> Vec<String> {
        self.successors
            .iter()
            .filter(|(_, succs)| succs.is_empty())
            .map(|(k, _)| k.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> JobId {
        JobId::from(s)
    }

    #[test]
    fn test_new_graph_is_empty() {
        let g = DependencyGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_job() {
        let mut g = DependencyGraph::new();
        g.add_job(&id("a"));
        assert_eq!(g.node_count(), 1);
        assert!(g.contains(&id("a")));
    }

    #[test]
    fn test_add_dependency() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        assert_eq!(g.edge_count(), 1);
        assert!(g.successors_of(&id("a")).contains("b"));
        assert!(g.predecessors_of(&id("b")).contains("a"));
    }

    #[test]
    fn test_self_dependency_rejected() {
        let mut g = DependencyGraph::new();
        let err = g.add_dependency(&id("a"), &id("a"));
        assert!(err.is_err());
    }

    #[test]
    fn test_no_cycle_simple_chain() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        g.add_dependency(&id("b"), &id("c"))
            .expect("failed to add dependency");
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        g.add_dependency(&id("b"), &id("c"))
            .expect("failed to add dependency");
        g.add_dependency(&id("c"), &id("a"))
            .expect("failed to add dependency");
        assert!(g.has_cycle());
    }

    #[test]
    fn test_topological_order_linear() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        g.add_dependency(&id("b"), &id("c"))
            .expect("failed to add dependency");
        let order = g
            .topological_order()
            .expect("topological order should succeed");
        let ai = order
            .iter()
            .position(|x| x == "a")
            .expect("element not found in order");
        let bi = order
            .iter()
            .position(|x| x == "b")
            .expect("element not found in order");
        let ci = order
            .iter()
            .position(|x| x == "c")
            .expect("element not found in order");
        assert!(ai < bi && bi < ci);
    }

    #[test]
    fn test_topological_order_cycle_error() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("x"), &id("y"))
            .expect("failed to add dependency");
        g.add_dependency(&id("y"), &id("x"))
            .expect("failed to add dependency");
        assert!(g.topological_order().is_err());
    }

    #[test]
    fn test_roots() {
        let mut g = DependencyGraph::new();
        g.add_job(&id("root"));
        g.add_dependency(&id("root"), &id("child"))
            .expect("failed to add dependency");
        let roots = g.roots();
        assert!(roots.contains(&"root".to_string()));
        assert!(!roots.contains(&"child".to_string()));
    }

    #[test]
    fn test_leaves() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        let leaves = g.leaves();
        assert!(leaves.contains(&"b".to_string()));
        assert!(!leaves.contains(&"a".to_string()));
    }

    #[test]
    fn test_remove_job() {
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        g.remove_job(&id("a"));
        assert!(!g.contains(&id("a")));
        assert!(g.predecessors_of(&id("b")).is_empty());
    }

    #[test]
    fn test_diamond_dag() {
        // a -> b, a -> c, b -> d, c -> d
        let mut g = DependencyGraph::new();
        g.add_dependency(&id("a"), &id("b"))
            .expect("failed to add dependency");
        g.add_dependency(&id("a"), &id("c"))
            .expect("failed to add dependency");
        g.add_dependency(&id("b"), &id("d"))
            .expect("failed to add dependency");
        g.add_dependency(&id("c"), &id("d"))
            .expect("failed to add dependency");
        assert!(!g.has_cycle());
        let order = g
            .topological_order()
            .expect("topological order should succeed");
        let ai = order
            .iter()
            .position(|x| x == "a")
            .expect("element not found in order");
        let di = order
            .iter()
            .position(|x| x == "d")
            .expect("element not found in order");
        assert!(ai < di);
    }

    #[test]
    fn test_parallel_jobs_no_deps() {
        let mut g = DependencyGraph::new();
        g.add_job(&id("x"));
        g.add_job(&id("y"));
        g.add_job(&id("z"));
        assert!(!g.has_cycle());
        let order = g
            .topological_order()
            .expect("topological order should succeed");
        assert_eq!(order.len(), 3);
    }
}
