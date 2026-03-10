// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job dependency management for oximedia-jobs.
//!
//! Provides a directed acyclic graph (DAG) for expressing inter-job
//! dependencies and computing execution order via topological sort.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// JobDependency
// ---------------------------------------------------------------------------

/// Describes a single job and its declared dependencies.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct JobDependency {
    /// Unique job identifier.
    pub job_id: u64,
    /// IDs of jobs that must complete before this job can run.
    pub depends_on: Vec<u64>,
    /// Scheduling priority hint (higher = more important).
    pub priority: i32,
}

impl JobDependency {
    /// Create a new dependency record.
    #[must_use]
    pub fn new(job_id: u64, depends_on: Vec<u64>, priority: i32) -> Self {
        Self {
            job_id,
            depends_on,
            priority,
        }
    }
}

// ---------------------------------------------------------------------------
// DependencyGraph
// ---------------------------------------------------------------------------

/// Directed acyclic graph (DAG) representing job execution dependencies.
///
/// - `deps[A]` = list of jobs that `A` depends on (A waits for them).
/// - `reverse[B]` = list of jobs that depend on `B` (they wait for B).
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Forward edges: job → its prerequisites.
    deps: HashMap<u64, Vec<u64>>,
    /// Reverse edges: job → jobs that depend on it.
    reverse: HashMap<u64, Vec<u64>>,
}

impl DependencyGraph {
    /// Create an empty dependency graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            deps: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Register a job in the graph (with no dependencies initially).
    pub fn add_job(&mut self, id: u64) {
        self.deps.entry(id).or_default();
        self.reverse.entry(id).or_default();
    }

    /// Add a dependency: `job_id` depends on `dep_id`.
    ///
    /// Both jobs are registered if they are not already present.
    /// Returns an error if adding the edge would introduce a cycle.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the edge creates a dependency cycle.
    pub fn add_dependency(&mut self, job_id: u64, dep_id: u64) -> Result<(), String> {
        if job_id == dep_id {
            return Err(format!("Job {job_id} cannot depend on itself"));
        }

        // Ensure both nodes exist
        self.add_job(job_id);
        self.add_job(dep_id);

        // Check that dep_id does not already (transitively) depend on job_id,
        // which would create a cycle.
        if self.transitive_deps(dep_id).contains(&job_id) {
            return Err(format!(
                "Adding dependency {job_id} → {dep_id} would create a cycle"
            ));
        }

        // Avoid duplicate edges
        let fwd = self.deps.entry(job_id).or_default();
        if !fwd.contains(&dep_id) {
            fwd.push(dep_id);
        }

        let rev = self.reverse.entry(dep_id).or_default();
        if !rev.contains(&job_id) {
            rev.push(job_id);
        }

        Ok(())
    }

    /// Return all jobs whose prerequisites are fully satisfied.
    ///
    /// A job is "ready" when every job it depends on is in `completed`.
    #[must_use]
    pub fn ready_jobs(&self, completed: &HashSet<u64>) -> Vec<u64> {
        self.deps
            .iter()
            .filter(|(job_id, prereqs)| {
                !completed.contains(job_id) && prereqs.iter().all(|p| completed.contains(p))
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Compute a topological ordering of all registered jobs (Kahn's algorithm).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn topological_order(&self) -> Result<Vec<u64>, String> {
        // In-degree map
        let mut in_degree: HashMap<u64, usize> = self.deps.keys().map(|&k| (k, 0)).collect();
        for (job_id, prereqs) in &self.deps {
            in_degree.entry(*job_id).or_insert(0);
            for &pre in prereqs {
                *in_degree.entry(pre).or_insert(0) += 0; // ensure present
                                                         // Count how many depend on `pre`
                let _ = pre; // suppress lint
            }
        }
        // Recount properly: in_degree[X] = number of nodes that X depends on
        let mut in_deg: HashMap<u64, usize> = self.deps.keys().map(|&k| (k, 0)).collect();
        for (_job, prereqs) in &self.deps {
            for &pre in prereqs {
                in_deg.entry(pre).or_insert(0); // ensure present but don't increment here
            }
        }
        // in_deg[job] = number of prerequisites for job
        for (&job, prereqs) in &self.deps {
            in_deg.insert(job, prereqs.len());
        }

        let mut queue: VecDeque<u64> = in_deg
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::with_capacity(self.deps.len());

        while let Some(node) = queue.pop_front() {
            order.push(node);
            if let Some(dependents) = self.reverse.get(&node) {
                for &dep in dependents {
                    // SAFETY: every node in `reverse` was also inserted into `in_deg`
                    // during the initialisation loop above, so this entry always exists.
                    if let Some(deg) = in_deg.get_mut(&dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        if order.len() != self.deps.len() {
            return Err("Dependency graph contains a cycle".to_string());
        }

        Ok(order)
    }

    /// Returns `true` if the graph has at least one cycle.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        self.topological_order().is_err()
    }

    /// Return all transitive dependencies of `job_id` (BFS over `deps`).
    #[must_use]
    pub fn transitive_deps(&self, job_id: u64) -> Vec<u64> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(direct) = self.deps.get(&job_id) {
            for &d in direct {
                if visited.insert(d) {
                    queue.push_back(d);
                }
            }
        }

        while let Some(node) = queue.pop_front() {
            if let Some(prereqs) = self.deps.get(&node) {
                for &p in prereqs {
                    if visited.insert(p) {
                        queue.push_back(p);
                    }
                }
            }
        }

        visited.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph_linear() -> DependencyGraph {
        // 1 → 2 → 3 (1 must finish before 2, 2 before 3)
        let mut g = DependencyGraph::new();
        g.add_job(1);
        g.add_job(2);
        g.add_job(3);
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed");
        g.add_dependency(3, 2)
            .expect("add_dependency should succeed");
        g
    }

    #[test]
    fn test_add_job() {
        let mut g = DependencyGraph::new();
        g.add_job(42);
        assert!(g.deps.contains_key(&42));
        assert!(g.reverse.contains_key(&42));
    }

    #[test]
    fn test_add_dependency_basic() {
        let mut g = DependencyGraph::new();
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed");
        assert!(g.deps[&2].contains(&1));
        assert!(g.reverse[&1].contains(&2));
    }

    #[test]
    fn test_add_dependency_self_loop_rejected() {
        let mut g = DependencyGraph::new();
        let res = g.add_dependency(5, 5);
        assert!(res.is_err());
    }

    #[test]
    fn test_add_dependency_cycle_rejected() {
        let mut g = DependencyGraph::new();
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed");
        g.add_dependency(3, 2)
            .expect("add_dependency should succeed");
        // 1 → 3 would create 3→2→1→3 cycle
        let res = g.add_dependency(1, 3);
        assert!(res.is_err());
    }

    #[test]
    fn test_add_dependency_no_duplicate_edges() {
        let mut g = DependencyGraph::new();
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed");
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed"); // duplicate
        assert_eq!(g.deps[&2].len(), 1);
    }

    #[test]
    fn test_ready_jobs_empty_completed() {
        let g = make_graph_linear();
        let completed = HashSet::new();
        let ready = g.ready_jobs(&completed);
        // Only job 1 has no prerequisites
        assert_eq!(ready, vec![1]);
    }

    #[test]
    fn test_ready_jobs_after_first_complete() {
        let g = make_graph_linear();
        let mut completed = HashSet::new();
        completed.insert(1u64);
        let ready = g.ready_jobs(&completed);
        assert_eq!(ready, vec![2]);
    }

    #[test]
    fn test_ready_jobs_all_complete() {
        let g = make_graph_linear();
        let completed: HashSet<u64> = [1, 2, 3].iter().cloned().collect();
        let ready = g.ready_jobs(&completed);
        assert!(ready.is_empty());
    }

    #[test]
    fn test_topological_order_linear() {
        let g = make_graph_linear();
        let order = g.topological_order().expect("order should be valid");
        assert_eq!(order.len(), 3);
        let pos: HashMap<u64, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&1] < pos[&2]);
        assert!(pos[&2] < pos[&3]);
    }

    #[test]
    fn test_topological_order_independent_jobs() {
        let mut g = DependencyGraph::new();
        g.add_job(10);
        g.add_job(20);
        g.add_job(30);
        let order = g.topological_order().expect("order should be valid");
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_has_cycle_false_for_dag() {
        let g = make_graph_linear();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_transitive_deps() {
        let g = make_graph_linear();
        let td = g.transitive_deps(3);
        // 3 depends (transitively) on 1 and 2
        let td_set: HashSet<u64> = td.into_iter().collect();
        assert!(td_set.contains(&1));
        assert!(td_set.contains(&2));
        assert!(!td_set.contains(&3));
    }

    #[test]
    fn test_transitive_deps_no_deps() {
        let g = make_graph_linear();
        let td = g.transitive_deps(1);
        assert!(td.is_empty());
    }

    #[test]
    fn test_diamond_dependency() {
        //   1
        //  / \
        // 2   3
        //  \ /
        //   4
        let mut g = DependencyGraph::new();
        g.add_dependency(2, 1)
            .expect("add_dependency should succeed");
        g.add_dependency(3, 1)
            .expect("add_dependency should succeed");
        g.add_dependency(4, 2)
            .expect("add_dependency should succeed");
        g.add_dependency(4, 3)
            .expect("add_dependency should succeed");

        assert!(!g.has_cycle());
        let order = g.topological_order().expect("order should be valid");
        let pos: HashMap<u64, usize> = order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&1] < pos[&2]);
        assert!(pos[&1] < pos[&3]);
        assert!(pos[&2] < pos[&4]);
        assert!(pos[&3] < pos[&4]);
    }
}
