//! Job dependency graph for the render farm.
//!
//! Provides a directed acyclic graph (DAG) of job dependencies, topological
//! sort, cycle detection, and ready-job filtering.

/// A directed dependency graph where an edge `(A, B)` means "A depends on B".
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct DependencyGraph {
    /// Registered job IDs (vertices).
    pub nodes: Vec<u64>,
    /// Dependency edges: `(job_id, depends_on)`.
    pub edges: Vec<(u64, u64)>,
}

impl DependencyGraph {
    /// Create an empty dependency graph.
    #[allow(dead_code)]
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Register a job in the graph.
    /// Does nothing if the job is already present.
    #[allow(dead_code)]
    pub fn add_job(&mut self, job_id: u64) {
        if !self.nodes.contains(&job_id) {
            self.nodes.push(job_id);
        }
    }

    /// Add a dependency: `job_id` depends on `depends_on`.
    ///
    /// Both jobs must already be registered. Returns an error if either is
    /// unknown or if adding the edge would create a cycle.
    #[allow(dead_code)]
    pub fn add_dependency(&mut self, job_id: u64, depends_on: u64) -> Result<(), String> {
        if !self.nodes.contains(&job_id) {
            return Err(format!("Unknown job: {job_id}"));
        }
        if !self.nodes.contains(&depends_on) {
            return Err(format!("Unknown dependency target: {depends_on}"));
        }
        // Temporarily add edge and check for cycle
        self.edges.push((job_id, depends_on));
        if self.has_cycle() {
            self.edges.pop();
            return Err(format!(
                "Adding dependency {job_id} -> {depends_on} would create a cycle"
            ));
        }
        Ok(())
    }

    /// Return jobs that are ready to run: registered, not in `completed`,
    /// and all their dependencies are in `completed`.
    #[allow(dead_code)]
    #[must_use]
    pub fn ready_jobs(&self, completed: &[u64]) -> Vec<u64> {
        self.nodes
            .iter()
            .copied()
            .filter(|&job_id| {
                // Not already completed
                if completed.contains(&job_id) {
                    return false;
                }
                // All dependencies must be completed
                self.dependencies_of(job_id)
                    .iter()
                    .all(|dep| completed.contains(dep))
            })
            .collect()
    }

    /// Return a topological ordering of all jobs.
    /// Returns an error if the graph contains a cycle.
    #[allow(dead_code)]
    pub fn topological_sort(&self) -> Result<Vec<u64>, String> {
        if self.has_cycle() {
            return Err("Cycle detected in dependency graph".to_string());
        }

        let mut result: Vec<u64> = Vec::new();
        let mut visited: Vec<u64> = Vec::new();
        let mut temp: Vec<u64> = Vec::new();

        for &node in &self.nodes {
            if !visited.contains(&node) {
                self.dfs_visit(node, &mut visited, &mut temp, &mut result)?;
            }
        }

        Ok(result)
    }

    fn dfs_visit(
        &self,
        node: u64,
        visited: &mut Vec<u64>,
        temp: &mut Vec<u64>,
        result: &mut Vec<u64>,
    ) -> Result<(), String> {
        if temp.contains(&node) {
            return Err(format!("Cycle involving node {node}"));
        }
        if visited.contains(&node) {
            return Ok(());
        }
        temp.push(node);
        // Visit nodes that `node` depends on first
        for dep in self.dependencies_of(node) {
            self.dfs_visit(dep, visited, temp, result)?;
        }
        temp.retain(|&x| x != node);
        visited.push(node);
        result.push(node);
        Ok(())
    }

    /// Return `true` if the graph contains a directed cycle.
    #[allow(dead_code)]
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        let mut visited: Vec<u64> = Vec::new();
        let mut in_stack: Vec<u64> = Vec::new();
        for &node in &self.nodes {
            if !visited.contains(&node) && self.dfs_cycle(node, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(&self, node: u64, visited: &mut Vec<u64>, in_stack: &mut Vec<u64>) -> bool {
        visited.push(node);
        in_stack.push(node);

        for dep in self.dependencies_of(node) {
            if !visited.contains(&dep) {
                if self.dfs_cycle(dep, visited, in_stack) {
                    return true;
                }
            } else if in_stack.contains(&dep) {
                return true;
            }
        }

        in_stack.retain(|&x| x != node);
        false
    }

    /// Return all jobs that `job_id` directly depends on.
    #[allow(dead_code)]
    #[must_use]
    pub fn dependencies_of(&self, job_id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|(j, _)| *j == job_id)
            .map(|(_, dep)| *dep)
            .collect()
    }

    /// Return all jobs that directly depend on `job_id`.
    #[allow(dead_code)]
    #[must_use]
    pub fn dependents_of(&self, job_id: u64) -> Vec<u64> {
        self.edges
            .iter()
            .filter(|(_, dep)| *dep == job_id)
            .map(|(j, _)| *j)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple linear chain: 1 -> 2 -> 3 (3 must run first).
    fn linear_graph() -> DependencyGraph {
        let mut g = DependencyGraph::new();
        g.add_job(1);
        g.add_job(2);
        g.add_job(3);
        g.add_dependency(1, 2).unwrap(); // 1 depends on 2
        g.add_dependency(2, 3).unwrap(); // 2 depends on 3
        g
    }

    #[test]
    fn test_add_job() {
        let mut g = DependencyGraph::new();
        g.add_job(10);
        g.add_job(20);
        assert_eq!(g.nodes.len(), 2);
        // Adding same ID again is idempotent
        g.add_job(10);
        assert_eq!(g.nodes.len(), 2);
    }

    #[test]
    fn test_add_dependency_unknown_job() {
        let mut g = DependencyGraph::new();
        g.add_job(1);
        let err = g.add_dependency(1, 99);
        assert!(err.is_err());
    }

    #[test]
    fn test_dependencies_of() {
        let g = linear_graph();
        let deps = g.dependencies_of(1);
        assert_eq!(deps, vec![2]);
        let deps3 = g.dependencies_of(3);
        assert!(deps3.is_empty());
    }

    #[test]
    fn test_dependents_of() {
        let g = linear_graph();
        let dependents = g.dependents_of(2);
        assert_eq!(dependents, vec![1]);
    }

    #[test]
    fn test_no_cycle() {
        let g = linear_graph();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = DependencyGraph::new();
        g.add_job(1);
        g.add_job(2);
        g.add_dependency(1, 2).unwrap();
        // Attempt to create a cycle: 2 depends on 1
        let result = g.add_dependency(2, 1);
        assert!(result.is_err());
        // The graph must not have a cycle (the edge was rolled back)
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_self_loop_detection() {
        let mut g = DependencyGraph::new();
        g.add_job(1);
        let result = g.add_dependency(1, 1);
        assert!(result.is_err());
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_topological_sort_linear() {
        let g = linear_graph();
        let order = g.topological_sort().unwrap();
        // 3 must come before 2, and 2 before 1
        let pos: std::collections::HashMap<u64, usize> =
            order.iter().enumerate().map(|(i, &j)| (j, i)).collect();
        assert!(pos[&3] < pos[&2]);
        assert!(pos[&2] < pos[&1]);
    }

    #[test]
    fn test_topological_sort_empty() {
        let g = DependencyGraph::new();
        let order = g.topological_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_ready_jobs_none_completed() {
        let g = linear_graph();
        // Only job 3 (no deps) should be ready
        let ready = g.ready_jobs(&[]);
        assert_eq!(ready, vec![3]);
    }

    #[test]
    fn test_ready_jobs_after_completion() {
        let g = linear_graph();
        // After 3 completes, job 2 becomes ready
        let ready = g.ready_jobs(&[3]);
        assert!(ready.contains(&2));
        assert!(!ready.contains(&3));
    }

    #[test]
    fn test_ready_jobs_all_completed() {
        let g = linear_graph();
        let ready = g.ready_jobs(&[1, 2, 3]);
        assert!(ready.is_empty());
    }

    #[test]
    fn test_diamond_dependency() {
        // Jobs: 1 depends on 2 and 3; both 2 and 3 depend on 4
        let mut g = DependencyGraph::new();
        for id in [1, 2, 3, 4] {
            g.add_job(id);
        }
        g.add_dependency(1, 2).unwrap();
        g.add_dependency(1, 3).unwrap();
        g.add_dependency(2, 4).unwrap();
        g.add_dependency(3, 4).unwrap();
        assert!(!g.has_cycle());
        let order = g.topological_sort().unwrap();
        let pos: std::collections::HashMap<u64, usize> =
            order.iter().enumerate().map(|(i, &j)| (j, i)).collect();
        assert!(pos[&4] < pos[&2]);
        assert!(pos[&4] < pos[&3]);
        assert!(pos[&2] < pos[&1]);
        assert!(pos[&3] < pos[&1]);
    }
}
