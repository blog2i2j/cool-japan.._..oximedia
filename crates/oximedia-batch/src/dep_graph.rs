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

    /// Number of direct successors (dependents) of `node`.
    ///
    /// This is the *fan-out* of the node in the DAG.
    #[must_use]
    pub fn fan_out(&self, node: u64) -> usize {
        self.edges.iter().filter(|&&(from, _)| from == node).count()
    }

    /// Number of direct predecessors (dependencies) of `node`.
    ///
    /// This is the *fan-in* of the node in the DAG.
    #[must_use]
    pub fn fan_in(&self, node: u64) -> usize {
        self.edges.iter().filter(|&&(_, to)| to == node).count()
    }

    /// Return the critical path through the DAG — the longest path measured by
    /// node count (i.e., the path with the most nodes from any source to any
    /// sink).
    ///
    /// Uses dynamic programming over a topological ordering.  Returns an empty
    /// `Vec` if the graph is empty or contains a cycle.
    #[must_use]
    pub fn critical_path(&self) -> Vec<u64> {
        let order = match topological_sort(self) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };

        if order.is_empty() {
            return Vec::new();
        }

        // dp[node] = (longest_path_length_ending_here, predecessor_on_that_path)
        let mut dp: std::collections::HashMap<u64, (usize, Option<u64>)> =
            order.iter().map(|&n| (n, (1, None))).collect();

        for &node in &order {
            // Inspect all edges that *leave* this node.
            let successors = self.dependents_of(node);
            for succ in successors {
                let new_len = dp[&node].0 + 1;
                let entry = dp.entry(succ).or_insert((1, None));
                if new_len > entry.0 {
                    *entry = (new_len, Some(node));
                }
            }
        }

        // Find the sink node with the longest incoming path.
        let best_end = dp.iter().max_by_key(|(_, &(len, _))| len).map(|(&n, _)| n);

        let Some(mut cur) = best_end else {
            return Vec::new();
        };

        // Reconstruct path by following predecessor links backwards.
        let mut path = Vec::new();
        loop {
            path.push(cur);
            match dp[&cur].1 {
                Some(prev) => cur = prev,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Enumerate **all** simple paths from `from` to `to` using depth-first
    /// search.
    ///
    /// Returns at most 1 000 paths to prevent combinatorial explosion.
    /// Returns an empty `Vec` when no path exists or either node is unknown.
    #[must_use]
    pub fn all_paths(&self, from: u64, to: u64) -> Vec<Vec<u64>> {
        const MAX_PATHS: usize = 1_000;

        if !self.nodes.contains(&from) || !self.nodes.contains(&to) {
            return Vec::new();
        }

        let mut results: Vec<Vec<u64>> = Vec::new();
        let mut current_path: Vec<u64> = vec![from];
        let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();
        visited.insert(from);

        Self::dfs_all_paths(
            self,
            from,
            to,
            &mut current_path,
            &mut visited,
            &mut results,
            MAX_PATHS,
        );

        results
    }

    fn dfs_all_paths(
        graph: &DepGraph,
        current: u64,
        target: u64,
        path: &mut Vec<u64>,
        visited: &mut std::collections::HashSet<u64>,
        results: &mut Vec<Vec<u64>>,
        max_paths: usize,
    ) {
        if current == target {
            results.push(path.clone());
            return;
        }
        if results.len() >= max_paths {
            return;
        }
        for &next in &graph
            .edges
            .iter()
            .filter_map(|&(f, t)| if f == current { Some(t) } else { None })
            .collect::<Vec<_>>()
        {
            if !visited.contains(&next) {
                visited.insert(next);
                path.push(next);
                Self::dfs_all_paths(graph, next, target, path, visited, results, max_paths);
                path.pop();
                visited.remove(&next);
            }
            if results.len() >= max_paths {
                return;
            }
        }
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

    // ── fan_in / fan_out ──────────────────────────────────────────────────────

    #[test]
    fn test_fan_out_root_node() {
        let g = linear_graph(); // 1 -> 2 -> 3
        assert_eq!(g.fan_out(1), 1);
        assert_eq!(g.fan_out(2), 1);
        assert_eq!(g.fan_out(3), 0);
    }

    #[test]
    fn test_fan_in_leaf_node() {
        let g = linear_graph();
        assert_eq!(g.fan_in(1), 0);
        assert_eq!(g.fan_in(2), 1);
        assert_eq!(g.fan_in(3), 1);
    }

    #[test]
    fn test_fan_out_diamond_center() {
        // 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 4);
        g.add_dependency(3, 4);
        assert_eq!(g.fan_out(1), 2);
        assert_eq!(g.fan_in(4), 2);
    }

    // ── critical_path ─────────────────────────────────────────────────────────

    #[test]
    fn test_critical_path_linear() {
        let g = linear_graph();
        let cp = g.critical_path();
        assert_eq!(cp, vec![1, 2, 3]);
    }

    #[test]
    fn test_critical_path_selects_longest() {
        // 1 -> 2 -> 3 -> 5
        // 1 -> 4 -> 5
        // critical path should be 1, 2, 3, 5
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        g.add_dependency(3, 5);
        g.add_dependency(1, 4);
        g.add_dependency(4, 5);
        let cp = g.critical_path();
        assert_eq!(cp.len(), 4);
        assert_eq!(cp[0], 1);
        assert_eq!(*cp.last().expect("should have last"), 5);
    }

    #[test]
    fn test_critical_path_empty_graph() {
        let g = DepGraph::new();
        assert!(g.critical_path().is_empty());
    }

    #[test]
    fn test_critical_path_single_node() {
        let mut g = DepGraph::new();
        g.add_node(42);
        let cp = g.critical_path();
        assert_eq!(cp, vec![42]);
    }

    // ── all_paths ─────────────────────────────────────────────────────────────

    #[test]
    fn test_all_paths_linear() {
        let g = linear_graph();
        let paths = g.all_paths(1, 3);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![1, 2, 3]);
    }

    #[test]
    fn test_all_paths_diamond() {
        // 1 -> 2 -> 4
        // 1 -> 3 -> 4
        let mut g = DepGraph::new();
        g.add_dependency(1, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 4);
        g.add_dependency(3, 4);
        let paths = g.all_paths(1, 4);
        assert_eq!(paths.len(), 2);
        // Both paths start at 1 and end at 4.
        assert!(paths
            .iter()
            .all(|p| p[0] == 1 && *p.last().expect("should have last") == 4));
    }

    #[test]
    fn test_all_paths_no_path() {
        let g = linear_graph();
        // 3 -> 1 does not exist.
        let paths = g.all_paths(3, 1);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_all_paths_unknown_node() {
        let g = linear_graph();
        let paths = g.all_paths(99, 1);
        assert!(paths.is_empty());
    }
}

// =============================================================================
// ConditionalDependencyGraph — string-keyed jobs with typed DependencyCondition
// =============================================================================

/// Condition that controls when a downstream job becomes ready to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyCondition {
    /// Only proceed when the predecessor job succeeded.
    OnSuccess,
    /// Only proceed when the predecessor job failed.
    OnFailure,
    /// Proceed regardless of whether the predecessor succeeded or failed.
    OnCompletion,
    /// Proceed when at least `min_success_count` predecessors with `OnSuccess`
    /// semantics have succeeded.
    Threshold {
        /// Minimum number of successful predecessors required.
        min_success_count: u32,
    },
}

/// A typed dependency edge between two string-keyed jobs.
#[derive(Debug, Clone)]
pub struct JobDependency {
    /// The job that must complete before the downstream job can run.
    pub job_id: String,
    /// The condition that determines when the downstream job becomes ready.
    pub condition: DependencyCondition,
}

/// Current execution status of a job inside [`ConditionalDependencyGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Job has not started yet.
    Pending,
    /// Job is currently executing.
    Running,
    /// Job finished successfully.
    Succeeded,
    /// Job finished with an error.
    Failed,
}

/// A dependency graph with string job IDs and typed [`DependencyCondition`]
/// edges. Supports cycle detection (DFS) on [`add_dependency`].
///
/// [`add_dependency`]: ConditionalDependencyGraph::add_dependency
#[derive(Debug, Default)]
pub struct ConditionalDependencyGraph {
    /// Registered job IDs in insertion order.
    jobs: Vec<String>,
    /// All dependency edges. An entry `(upstream, downstream, condition)` means
    /// `downstream` depends on `upstream` under `condition`.
    edges: Vec<(String, String, DependencyCondition)>,
    /// Current status per job.
    statuses: std::collections::HashMap<String, JobStatus>,
}

impl ConditionalDependencyGraph {
    /// Create a new, empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a job in the graph. No-op if the job is already registered.
    pub fn add_job(&mut self, job_id: impl Into<String>) {
        let id = job_id.into();
        if !self.jobs.contains(&id) {
            self.statuses.insert(id.clone(), JobStatus::Pending);
            self.jobs.push(id);
        }
    }

    /// Add a conditional dependency edge: `to` depends on `from` under `condition`.
    ///
    /// Both jobs are auto-registered if they are not yet in the graph.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if:
    /// - `from == to` (self-dependency)
    /// - Adding this edge would create a cycle (DFS check)
    pub fn add_dependency(
        &mut self,
        from: impl Into<String>,
        to: impl Into<String>,
        condition: DependencyCondition,
    ) -> Result<(), String> {
        let from_id = from.into();
        let to_id = to.into();

        if from_id == to_id {
            return Err(format!("Self-dependency not allowed: {from_id}"));
        }

        self.add_job(from_id.clone());
        self.add_job(to_id.clone());

        // Temporarily add the edge then check for cycles.
        self.edges.push((from_id.clone(), to_id.clone(), condition));

        if self.has_cycle() {
            self.edges.pop();
            return Err(format!(
                "Cycle detected: adding edge {from_id} -> {to_id} would create a cycle"
            ));
        }

        Ok(())
    }

    /// Return the current status of a job, or `None` if the job is not registered.
    #[must_use]
    pub fn job_status(&self, job_id: &str) -> Option<&JobStatus> {
        self.statuses.get(job_id)
    }

    /// Mark a job as running.
    pub fn start_job(&mut self, job_id: &str) {
        if let Some(s) = self.statuses.get_mut(job_id) {
            *s = JobStatus::Running;
        }
    }

    /// Mark a job as succeeded.
    pub fn complete_job(&mut self, job_id: &str) {
        if let Some(s) = self.statuses.get_mut(job_id) {
            *s = JobStatus::Succeeded;
        }
    }

    /// Mark a job as failed.
    pub fn fail_job(&mut self, job_id: &str) {
        if let Some(s) = self.statuses.get_mut(job_id) {
            *s = JobStatus::Failed;
        }
    }

    /// Return the IDs of jobs that are ready to execute now.
    ///
    /// A job is *ready* when:
    /// - Its status is [`JobStatus::Pending`], **and**
    /// - Every incoming edge has its condition satisfied given current statuses.
    ///
    /// Condition satisfaction rules:
    /// - [`DependencyCondition::OnSuccess`]: predecessor is [`JobStatus::Succeeded`]
    /// - [`DependencyCondition::OnFailure`]: predecessor is [`JobStatus::Failed`]
    /// - [`DependencyCondition::OnCompletion`]: predecessor is Succeeded **or** Failed
    /// - [`DependencyCondition::Threshold`]: the count of predecessors (across all
    ///   incoming edges on this job) with `OnSuccess` condition that are currently
    ///   Succeeded ≥ `min_success_count`
    #[must_use]
    pub fn get_ready_jobs(&self) -> Vec<String> {
        let mut ready = Vec::new();

        'outer: for job_id in &self.jobs {
            // Only Pending jobs can become ready.
            if self.statuses.get(job_id.as_str()) != Some(&JobStatus::Pending) {
                continue;
            }

            let incoming: Vec<&(String, String, DependencyCondition)> = self
                .edges
                .iter()
                .filter(|(_, to, _)| to == job_id)
                .collect();

            // Jobs with no incoming edges are immediately ready.
            if incoming.is_empty() {
                ready.push(job_id.clone());
                continue;
            }

            // Handle Threshold condition first: collect all threshold edges.
            // If there is at least one Threshold edge we apply threshold logic;
            // non-threshold edges still apply their own checks.
            let mut threshold_required: Option<u32> = None;
            let mut success_count: u32 = 0;

            for (from, _, cond) in &incoming {
                match cond {
                    DependencyCondition::Threshold { min_success_count } => {
                        // Take the maximum threshold if multiple exist.
                        threshold_required = Some(
                            threshold_required
                                .map(|v: u32| v.max(*min_success_count))
                                .unwrap_or(*min_success_count),
                        );
                        if self.statuses.get(from.as_str()) == Some(&JobStatus::Succeeded) {
                            success_count += 1;
                        }
                    }
                    DependencyCondition::OnSuccess => {
                        let status = self.statuses.get(from.as_str());
                        match status {
                            Some(JobStatus::Succeeded) => {}            // condition met
                            Some(JobStatus::Failed) => continue 'outer, // permanently unsatisfiable
                            _ => continue 'outer,                       // predecessor not finished
                        }
                    }
                    DependencyCondition::OnFailure => {
                        let status = self.statuses.get(from.as_str());
                        match status {
                            Some(JobStatus::Failed) => {}                  // condition met
                            Some(JobStatus::Succeeded) => continue 'outer, // permanently unsatisfiable
                            _ => continue 'outer, // predecessor not finished
                        }
                    }
                    DependencyCondition::OnCompletion => {
                        let status = self.statuses.get(from.as_str());
                        match status {
                            Some(JobStatus::Succeeded) | Some(JobStatus::Failed) => {} // met
                            _ => continue 'outer, // not finished yet
                        }
                    }
                }
            }

            // Check threshold if applicable.
            if let Some(required) = threshold_required {
                if success_count < required {
                    continue 'outer;
                }
            }

            ready.push(job_id.clone());
        }

        ready
    }

    /// Return `true` if the graph contains at least one cycle (DFS coloring).
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        // Build adjacency map from edges.
        let mut adj: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
        for id in &self.jobs {
            adj.entry(id.as_str()).or_default();
        }
        for (from, to, _) in &self.edges {
            adj.entry(from.as_str()).or_default().push(to.as_str());
        }

        // 0 = white, 1 = gray, 2 = black
        let mut color: std::collections::HashMap<&str, u8> = std::collections::HashMap::new();

        for start in self.jobs.iter().map(String::as_str) {
            if color.get(start).copied().unwrap_or(0) == 0
                && Self::dfs_cycle(start, &adj, &mut color)
            {
                return true;
            }
        }
        false
    }

    fn dfs_cycle<'a>(
        node: &'a str,
        adj: &std::collections::HashMap<&'a str, Vec<&'a str>>,
        color: &mut std::collections::HashMap<&'a str, u8>,
    ) -> bool {
        color.insert(node, 1); // gray
        if let Some(neighbors) = adj.get(node) {
            for &next in neighbors {
                let c = color.get(next).copied().unwrap_or(0);
                if c == 1 {
                    return true; // back edge → cycle
                }
                if c == 0 && Self::dfs_cycle(next, adj, color) {
                    return true;
                }
            }
        }
        color.insert(node, 2); // black
        false
    }

    /// Return the number of registered jobs.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Return the number of dependency edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// =============================================================================
// ConditionalDependencyGraph tests
// =============================================================================

#[cfg(test)]
mod conditional_tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn linear_cdag() -> ConditionalDependencyGraph {
        // a --OnSuccess--> b --OnSuccess--> c
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnSuccess)
            .expect("add dep a->b");
        g.add_dependency("b", "c", DependencyCondition::OnSuccess)
            .expect("add dep b->c");
        g
    }

    // ── basic graph construction ──────────────────────────────────────────────

    #[test]
    fn test_new_graph_is_empty() {
        let g = ConditionalDependencyGraph::new();
        assert_eq!(g.job_count(), 0);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_job_idempotent() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("alpha");
        g.add_job("alpha");
        assert_eq!(g.job_count(), 1);
    }

    #[test]
    fn test_add_dependency_registers_jobs() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("x", "y", DependencyCondition::OnSuccess)
            .expect("add dep");
        assert_eq!(g.job_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_self_dependency_rejected() {
        let mut g = ConditionalDependencyGraph::new();
        let result = g.add_dependency("a", "a", DependencyCondition::OnSuccess);
        assert!(result.is_err());
        let msg = result.expect_err("expected error");
        assert!(msg.contains("Self-dependency"), "got: {msg}");
    }

    // ── job status transitions ────────────────────────────────────────────────

    #[test]
    fn test_initial_status_is_pending() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("job1");
        assert_eq!(g.job_status("job1"), Some(&JobStatus::Pending));
    }

    #[test]
    fn test_start_job() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("job1");
        g.start_job("job1");
        assert_eq!(g.job_status("job1"), Some(&JobStatus::Running));
    }

    #[test]
    fn test_complete_job() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("job1");
        g.complete_job("job1");
        assert_eq!(g.job_status("job1"), Some(&JobStatus::Succeeded));
    }

    #[test]
    fn test_fail_job() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("job1");
        g.fail_job("job1");
        assert_eq!(g.job_status("job1"), Some(&JobStatus::Failed));
    }

    // ── get_ready_jobs — linear chain ─────────────────────────────────────────

    #[test]
    fn test_linear_chain_initial_ready() {
        let g = linear_cdag();
        // Only "a" has no predecessors
        let ready = g.get_ready_jobs();
        assert_eq!(ready, vec!["a"]);
    }

    #[test]
    fn test_linear_chain_step_through() {
        let mut g = linear_cdag();
        // Initially only "a"
        assert_eq!(g.get_ready_jobs(), vec!["a"]);

        g.complete_job("a");
        let ready = g.get_ready_jobs();
        assert_eq!(ready, vec!["b"]);

        g.complete_job("b");
        let ready = g.get_ready_jobs();
        assert_eq!(ready, vec!["c"]);

        g.complete_job("c");
        assert!(g.get_ready_jobs().is_empty());
    }

    #[test]
    fn test_linear_chain_failure_blocks_on_success_dep() {
        let mut g = linear_cdag();
        // "a" fails → "b" has OnSuccess dep on "a" → "b" is never ready
        g.fail_job("a");
        assert!(g.get_ready_jobs().is_empty());
    }

    // ── OnFailure condition ───────────────────────────────────────────────────

    #[test]
    fn test_on_failure_condition() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnFailure)
            .expect("add dep");

        // "a" not finished → "b" not ready
        assert!(g.get_ready_jobs().is_empty() || g.get_ready_jobs() == vec!["a"]);

        g.fail_job("a");
        let ready = g.get_ready_jobs();
        assert!(
            ready.contains(&"b".to_string()),
            "b should be ready after a fails"
        );
    }

    #[test]
    fn test_on_failure_condition_not_met_on_success() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnFailure)
            .expect("add dep");
        g.complete_job("a"); // succeeded → OnFailure condition not met
        let ready = g.get_ready_jobs();
        assert!(!ready.contains(&"b".to_string()));
    }

    // ── OnCompletion condition ────────────────────────────────────────────────

    #[test]
    fn test_on_completion_triggers_on_success() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnCompletion)
            .expect("add dep");
        g.complete_job("a");
        let ready = g.get_ready_jobs();
        assert!(ready.contains(&"b".to_string()));
    }

    #[test]
    fn test_on_completion_triggers_on_failure() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnCompletion)
            .expect("add dep");
        g.fail_job("a");
        let ready = g.get_ready_jobs();
        assert!(ready.contains(&"b".to_string()));
    }

    #[test]
    fn test_on_completion_not_ready_while_running() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnCompletion)
            .expect("add dep");
        g.start_job("a");
        let ready = g.get_ready_jobs();
        assert!(!ready.contains(&"b".to_string()));
    }

    // ── Diamond pattern ───────────────────────────────────────────────────────

    #[test]
    fn test_diamond_fan_out_fan_in() {
        // root --OnSuccess--> left
        // root --OnSuccess--> right
        // left --OnSuccess--> join
        // right --OnSuccess--> join
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("root", "left", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("root", "right", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("left", "join", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("right", "join", DependencyCondition::OnSuccess)
            .expect("add dep");

        // Only root is ready at start
        let ready = g.get_ready_jobs();
        assert_eq!(ready, vec!["root"]);

        g.complete_job("root");
        let mut ready = g.get_ready_jobs();
        ready.sort();
        assert_eq!(ready, vec!["left", "right"]);

        g.complete_job("left");
        // right still needed for join
        assert!(!g.get_ready_jobs().contains(&"join".to_string()));

        g.complete_job("right");
        let ready = g.get_ready_jobs();
        assert!(ready.contains(&"join".to_string()));
    }

    // ── Threshold condition ───────────────────────────────────────────────────

    #[test]
    fn test_threshold_met() {
        // Two predecessors a, b both with Threshold{1} on "join"
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency(
            "a",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 1,
            },
        )
        .expect("add dep");
        g.add_dependency(
            "b",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 1,
            },
        )
        .expect("add dep");

        // Neither done yet
        assert!(!g.get_ready_jobs().contains(&"join".to_string()));

        g.complete_job("a");
        // 1 success >= threshold 1 → join ready
        assert!(g.get_ready_jobs().contains(&"join".to_string()));
    }

    #[test]
    fn test_threshold_not_met() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency(
            "a",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 2,
            },
        )
        .expect("add dep");
        g.add_dependency(
            "b",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 2,
            },
        )
        .expect("add dep");

        g.complete_job("a"); // only 1 success, need 2
        assert!(!g.get_ready_jobs().contains(&"join".to_string()));

        g.complete_job("b");
        assert!(g.get_ready_jobs().contains(&"join".to_string()));
    }

    #[test]
    fn test_threshold_with_failure_still_counts_successes() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency(
            "a",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 1,
            },
        )
        .expect("add dep");
        g.add_dependency(
            "b",
            "join",
            DependencyCondition::Threshold {
                min_success_count: 1,
            },
        )
        .expect("add dep");

        // b fails but a succeeds → 1 success >= threshold 1
        g.fail_job("b");
        g.complete_job("a");
        assert!(g.get_ready_jobs().contains(&"join".to_string()));
    }

    // ── Cycle detection ───────────────────────────────────────────────────────

    #[test]
    fn test_cycle_detection_self_loop() {
        let mut g = ConditionalDependencyGraph::new();
        let err = g.add_dependency("a", "a", DependencyCondition::OnSuccess);
        assert!(err.is_err());
    }

    #[test]
    fn test_cycle_detection_two_node() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnSuccess)
            .expect("add dep a->b");
        let err = g.add_dependency("b", "a", DependencyCondition::OnSuccess);
        assert!(err.is_err());
        let msg = err.expect_err("expected error");
        assert!(msg.contains("Cycle"), "got: {msg}");
    }

    #[test]
    fn test_cycle_detection_three_node_cycle() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "b", DependencyCondition::OnSuccess)
            .expect("add dep a->b");
        g.add_dependency("b", "c", DependencyCondition::OnSuccess)
            .expect("add dep b->c");
        let err = g.add_dependency("c", "a", DependencyCondition::OnSuccess);
        assert!(err.is_err());
    }

    #[test]
    fn test_no_cycle_acyclic_graph() {
        let g = linear_cdag();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_no_cycle_diamond() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("root", "a", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("root", "b", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("a", "join", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("b", "join", DependencyCondition::OnSuccess)
            .expect("add dep");
        assert!(!g.has_cycle());
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_graph_returns_no_ready_jobs() {
        let g = ConditionalDependencyGraph::new();
        assert!(g.get_ready_jobs().is_empty());
    }

    #[test]
    fn test_single_node_immediately_ready() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_job("solo");
        assert_eq!(g.get_ready_jobs(), vec!["solo"]);
    }

    #[test]
    fn test_fan_out_all_branches_ready_after_root() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("root", "a", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("root", "b", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("root", "c", DependencyCondition::OnSuccess)
            .expect("add dep");

        g.complete_job("root");
        let mut ready = g.get_ready_jobs();
        ready.sort();
        assert_eq!(ready, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_fan_in_all_deps_must_succeed() {
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("a", "join", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("b", "join", DependencyCondition::OnSuccess)
            .expect("add dep");
        g.add_dependency("c", "join", DependencyCondition::OnSuccess)
            .expect("add dep");

        g.complete_job("a");
        g.complete_job("b");
        assert!(!g.get_ready_jobs().contains(&"join".to_string()));

        g.complete_job("c");
        assert!(g.get_ready_jobs().contains(&"join".to_string()));
    }

    #[test]
    fn test_mixed_conditions_in_same_graph() {
        // "cleanup" depends on "work" via OnCompletion
        // "notify-fail" depends on "work" via OnFailure
        let mut g = ConditionalDependencyGraph::new();
        g.add_dependency("work", "cleanup", DependencyCondition::OnCompletion)
            .expect("add dep");
        g.add_dependency("work", "notify-fail", DependencyCondition::OnFailure)
            .expect("add dep");

        // work fails
        g.fail_job("work");
        let ready = g.get_ready_jobs();
        assert!(
            ready.contains(&"cleanup".to_string()),
            "cleanup should be ready"
        );
        assert!(
            ready.contains(&"notify-fail".to_string()),
            "notify-fail should be ready"
        );
    }
}
