// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Directed Acyclic Graph (DAG) for distributed job dependencies.
//!
//! Provides cycle detection, topological ordering, and unreachable node warnings
//! for job dependency graphs in distributed encoding pipelines.

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors that can arise in DAG operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DagError {
    /// The graph contains a cycle (not a valid DAG).
    #[error("cycle detected involving node '{0}'")]
    CycleDetected(String),

    /// A referenced node does not exist in the graph.
    #[error("node '{0}' not found in graph")]
    NodeNotFound(String),

    /// An edge would create a self-loop.
    #[error("self-loop on node '{0}' is not allowed")]
    SelfLoop(String),
}

/// A directed acyclic graph for expressing job dependencies.
///
/// Nodes are identified by arbitrary string keys.  An edge `A → B` means
/// **A must complete before B can start**.
#[derive(Debug, Default, Clone)]
pub struct JobDag {
    /// Adjacency list: node → list of successor nodes it must precede.
    edges: HashMap<String, Vec<String>>,
}

impl JobDag {
    /// Create an empty DAG.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the DAG.  Adding an already-existing node is a no-op.
    pub fn add_node(&mut self, id: impl Into<String>) {
        self.edges.entry(id.into()).or_default();
    }

    /// Add a directed dependency edge: `from` must complete before `to`.
    ///
    /// Returns [`DagError::NodeNotFound`] if either endpoint is missing,
    /// [`DagError::SelfLoop`] if `from == to`, and [`DagError::CycleDetected`]
    /// if adding the edge would introduce a cycle.
    pub fn add_edge(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
    ) -> Result<(), DagError> {
        let from = from.into();
        let to = to.into();

        if from == to {
            return Err(DagError::SelfLoop(from));
        }
        if !self.edges.contains_key(&from) {
            return Err(DagError::NodeNotFound(from));
        }
        if !self.edges.contains_key(&to) {
            return Err(DagError::NodeNotFound(to));
        }

        // Tentatively add the edge, then check for cycles.
        self.edges.entry(from.clone()).or_default().push(to.clone());

        if self.has_cycle() {
            // Roll back.
            let succs = self.edges.get_mut(&from).expect("just inserted");
            succs.retain(|s| s != &to);
            return Err(DagError::CycleDetected(from));
        }

        Ok(())
    }

    /// Return `true` if the graph contains at least one cycle.
    pub fn has_cycle(&self) -> bool {
        let mut visited: HashSet<&str> = HashSet::new();
        let mut rec_stack: HashSet<&str> = HashSet::new();

        for node in self.edges.keys() {
            if !visited.contains(node.as_str()) {
                if self.dfs_cycle(node, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn dfs_cycle<'a>(
        &'a self,
        node: &'a str,
        visited: &mut HashSet<&'a str>,
        rec_stack: &mut HashSet<&'a str>,
    ) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        if let Some(succs) = self.edges.get(node) {
            for succ in succs {
                if !visited.contains(succ.as_str()) {
                    if self.dfs_cycle(succ, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(succ.as_str()) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Compute a topological ordering of all nodes (Kahn's algorithm).
    ///
    /// Returns [`DagError::CycleDetected`] if the graph is not acyclic.
    pub fn topological_order(&self) -> Result<Vec<String>, DagError> {
        // Build in-degree counts.
        let mut in_degree: HashMap<&str, usize> =
            self.edges.keys().map(|k| (k.as_str(), 0)).collect();
        for succs in self.edges.values() {
            for s in succs {
                *in_degree.entry(s.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter_map(|(&n, &d)| if d == 0 { Some(n) } else { None })
            .collect();

        // Ensure deterministic output.
        let mut queue_vec: Vec<&str> = queue.drain(..).collect();
        queue_vec.sort_unstable();
        queue.extend(queue_vec);

        let mut order: Vec<String> = Vec::with_capacity(self.edges.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.to_owned());
            if let Some(succs) = self.edges.get(node) {
                let mut next: Vec<&str> = Vec::new();
                for s in succs {
                    let deg = in_degree.entry(s.as_str()).or_insert(0);
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        next.push(s.as_str());
                    }
                }
                next.sort_unstable();
                queue.extend(next);
            }
        }

        if order.len() != self.edges.len() {
            return Err(DagError::CycleDetected("(unknown)".to_owned()));
        }

        Ok(order)
    }

    /// Return the set of nodes that are unreachable from any root (in-degree 0) node.
    ///
    /// In a valid DAG this set will be empty unless the graph has isolated
    /// strongly-connected components (i.e. cycles).  Callers can use this to
    /// emit warnings about dangling dependencies.
    pub fn unreachable_nodes(&self) -> Vec<String> {
        // Roots: nodes with in-degree 0.
        let mut in_degree: HashMap<&str, usize> =
            self.edges.keys().map(|k| (k.as_str(), 0)).collect();
        for succs in self.edges.values() {
            for s in succs {
                *in_degree.entry(s.as_str()).or_insert(0) += 1;
            }
        }

        let roots: Vec<&str> = in_degree
            .iter()
            .filter_map(|(&n, &d)| if d == 0 { Some(n) } else { None })
            .collect();

        // BFS from all roots.
        let mut reachable: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = roots.into_iter().collect();
        while let Some(n) = queue.pop_front() {
            if reachable.insert(n) {
                if let Some(succs) = self.edges.get(n) {
                    for s in succs {
                        queue.push_back(s.as_str());
                    }
                }
            }
        }

        let mut unreachable: Vec<String> = self
            .edges
            .keys()
            .filter(|k| !reachable.contains(k.as_str()))
            .cloned()
            .collect();
        unreachable.sort();
        unreachable
    }

    /// Return all direct predecessors of the given node.
    pub fn predecessors(&self, node: &str) -> Result<Vec<String>, DagError> {
        if !self.edges.contains_key(node) {
            return Err(DagError::NodeNotFound(node.to_owned()));
        }
        let mut preds: Vec<String> = self
            .edges
            .iter()
            .filter_map(|(k, succs)| {
                if succs.iter().any(|s| s == node) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();
        preds.sort();
        Ok(preds)
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.edges.len()
    }

    /// Number of directed edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_dag() -> JobDag {
        let mut g = JobDag::new();
        g.add_node("a");
        g.add_node("b");
        g.add_node("c");
        g.add_edge("a", "b").unwrap();
        g.add_edge("b", "c").unwrap();
        g
    }

    #[test]
    fn test_add_node_and_edge() {
        let g = linear_dag();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_topological_order_linear() {
        let g = linear_dag();
        let order = g.topological_order().unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_no_cycle_in_linear_dag() {
        assert!(!linear_dag().has_cycle());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = JobDag::new();
        g.add_node("x");
        g.add_node("y");
        g.add_node("z");
        g.add_edge("x", "y").unwrap();
        g.add_edge("y", "z").unwrap();
        // z → x would create a cycle
        let err = g.add_edge("z", "x").unwrap_err();
        assert!(matches!(err, DagError::CycleDetected(_)));
        // Graph should still be acyclic after rollback
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_self_loop_rejected() {
        let mut g = JobDag::new();
        g.add_node("a");
        assert!(matches!(
            g.add_edge("a", "a").unwrap_err(),
            DagError::SelfLoop(_)
        ));
    }

    #[test]
    fn test_missing_node_rejected() {
        let mut g = JobDag::new();
        g.add_node("a");
        assert!(matches!(
            g.add_edge("a", "ghost").unwrap_err(),
            DagError::NodeNotFound(_)
        ));
        assert!(matches!(
            g.add_edge("ghost", "a").unwrap_err(),
            DagError::NodeNotFound(_)
        ));
    }

    #[test]
    fn test_predecessors() {
        let mut g = linear_dag();
        g.add_node("d");
        g.add_edge("a", "d").unwrap();
        let preds = g.predecessors("b").unwrap();
        assert_eq!(preds, vec!["a"]);
    }

    #[test]
    fn test_unreachable_nodes_empty_for_dag() {
        assert!(linear_dag().unreachable_nodes().is_empty());
    }

    #[test]
    fn test_topological_order_diamond() {
        // a → b, a → c, b → d, c → d
        let mut g = JobDag::new();
        for n in ["a", "b", "c", "d"] {
            g.add_node(n);
        }
        g.add_edge("a", "b").unwrap();
        g.add_edge("a", "c").unwrap();
        g.add_edge("b", "d").unwrap();
        g.add_edge("c", "d").unwrap();
        let order = g.topological_order().unwrap();
        // 'a' must come first, 'd' last
        assert_eq!(order[0], "a");
        assert_eq!(order[3], "d");
    }

    #[test]
    fn test_empty_dag_topological_order() {
        let g = JobDag::new();
        assert!(g.topological_order().unwrap().is_empty());
    }
}
