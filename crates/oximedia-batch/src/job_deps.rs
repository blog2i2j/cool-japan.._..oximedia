//! Job dependency chaining with topological sort and cycle detection.
//!
//! [`JobDependencyManager`] tracks dependencies between jobs using string-based
//! job IDs, provides topological ordering for execution, detects circular
//! dependencies at submission time, and supports fan-out / fan-in patterns.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{BatchError, Result};
use crate::types::JobId;

// ---------------------------------------------------------------------------
// DependencyStatus
// ---------------------------------------------------------------------------

/// Tracks the execution state of a job within the dependency graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyStatus {
    /// The job has unresolved dependencies and cannot run yet.
    Pending,
    /// All dependencies have been satisfied; the job is eligible to execute.
    Ready,
    /// The job has finished successfully.
    Completed,
    /// The job has failed (downstream dependents may also fail).
    Failed,
}

impl std::fmt::Display for DependencyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Ready => write!(f, "Ready"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

// ---------------------------------------------------------------------------
// JobDependencyManager
// ---------------------------------------------------------------------------

/// Manages dependencies between jobs, enforcing a DAG structure.
///
/// Internally keeps an adjacency list (successors / predecessors) plus a
/// per-job [`DependencyStatus`].
#[derive(Debug, Default)]
pub struct JobDependencyManager {
    /// Adjacency: job -> set of direct successors (jobs that depend on it).
    successors: HashMap<String, HashSet<String>>,
    /// Reverse adjacency: job -> set of direct predecessors (its dependencies).
    predecessors: HashMap<String, HashSet<String>>,
    /// Per-job status.
    status: HashMap<String, DependencyStatus>,
}

impl JobDependencyManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // -- Registration -------------------------------------------------------

    /// Register a job node.  If it has no predecessors it starts as `Ready`;
    /// otherwise it starts as `Pending`.
    ///
    /// This is idempotent — re-registering an existing job is a no-op.
    pub fn register_job(&mut self, job_id: &JobId) {
        let key = job_id.as_str().to_string();
        if self.status.contains_key(&key) {
            return;
        }
        self.successors.entry(key.clone()).or_default();
        self.predecessors.entry(key.clone()).or_default();
        self.status.insert(key, DependencyStatus::Ready);
    }

    /// Add a dependency: `job_id` depends on `depends_on`.
    ///
    /// Both jobs are automatically registered if not already present.
    ///
    /// # Errors
    ///
    /// - [`BatchError::DependencyError`] if the edge would create a cycle.
    /// - [`BatchError::DependencyError`] if `job_id == depends_on`.
    pub fn add_dependency(&mut self, job_id: &JobId, depends_on: &JobId) -> Result<()> {
        if job_id == depends_on {
            return Err(BatchError::DependencyError(format!(
                "Self-dependency not allowed: {}",
                job_id.as_str()
            )));
        }

        // Ensure both nodes exist.
        self.register_job(depends_on);
        self.register_job(job_id);

        let from = depends_on.as_str().to_string();
        let to = job_id.as_str().to_string();

        // Check for cycles *before* inserting the edge.
        if self.would_create_cycle(&from, &to) {
            return Err(BatchError::DependencyError(format!(
                "Adding dependency {} -> {} would create a cycle",
                from, to
            )));
        }

        self.successors.entry(from.clone()).or_default().insert(to.clone());
        self.predecessors.entry(to.clone()).or_default().insert(from);

        // Update status: `to` now has at least one unfinished predecessor.
        self.recompute_status(&to);

        Ok(())
    }

    // -- Status management --------------------------------------------------

    /// Current status of a job.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if the job is not registered.
    pub fn status(&self, job_id: &JobId) -> Result<DependencyStatus> {
        self.status
            .get(job_id.as_str())
            .copied()
            .ok_or_else(|| BatchError::JobNotFound(job_id.as_str().to_string()))
    }

    /// Mark a job as completed and propagate readiness to its dependents.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if the job is not registered.
    pub fn mark_completed(&mut self, job_id: &JobId) -> Result<()> {
        let key = job_id.as_str().to_string();
        if !self.status.contains_key(&key) {
            return Err(BatchError::JobNotFound(key));
        }
        self.status.insert(key.clone(), DependencyStatus::Completed);

        // Propagate: recompute status for all successors.
        let successors: Vec<String> = self
            .successors
            .get(&key)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
        for succ in &successors {
            self.recompute_status(succ);
        }
        Ok(())
    }

    /// Mark a job as failed.  Downstream dependents remain `Pending` (they
    /// will never become `Ready` unless the failure is cleared).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if the job is not registered.
    pub fn mark_failed(&mut self, job_id: &JobId) -> Result<()> {
        let key = job_id.as_str().to_string();
        if !self.status.contains_key(&key) {
            return Err(BatchError::JobNotFound(key));
        }
        self.status.insert(key.clone(), DependencyStatus::Failed);

        // Cascade: mark all transitive dependents as Failed.
        let dependents = self.transitive_dependents(&key);
        for dep in &dependents {
            self.status.insert(dep.clone(), DependencyStatus::Failed);
        }
        Ok(())
    }

    // -- Queries ------------------------------------------------------------

    /// Return all jobs currently in the `Ready` state.
    #[must_use]
    pub fn ready_jobs(&self) -> Vec<String> {
        self.status
            .iter()
            .filter(|(_, &s)| s == DependencyStatus::Ready)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Return all jobs currently in the `Pending` state.
    #[must_use]
    pub fn pending_jobs(&self) -> Vec<String> {
        self.status
            .iter()
            .filter(|(_, &s)| s == DependencyStatus::Pending)
            .map(|(k, _)| k.clone())
            .collect()
    }

    /// Total number of registered jobs.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.status.len()
    }

    /// Direct predecessors (dependencies) of a job.
    #[must_use]
    pub fn dependencies_of(&self, job_id: &JobId) -> Vec<String> {
        self.predecessors
            .get(job_id.as_str())
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Direct successors (dependents) of a job.
    #[must_use]
    pub fn dependents_of(&self, job_id: &JobId) -> Vec<String> {
        self.successors
            .get(job_id.as_str())
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Fan-out: number of direct successors.
    #[must_use]
    pub fn fan_out(&self, job_id: &JobId) -> usize {
        self.successors
            .get(job_id.as_str())
            .map_or(0, HashSet::len)
    }

    /// Fan-in: number of direct predecessors.
    #[must_use]
    pub fn fan_in(&self, job_id: &JobId) -> usize {
        self.predecessors
            .get(job_id.as_str())
            .map_or(0, HashSet::len)
    }

    // -- Topological sort ---------------------------------------------------

    /// Compute a topological ordering of all registered jobs using Kahn's
    /// algorithm.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::DependencyError`] if the graph contains a cycle
    /// (should not happen if edges were added via `add_dependency`, which
    /// checks for cycles).
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for key in self.status.keys() {
            in_degree.insert(key.as_str(), 0);
        }
        for (_, succs) in &self.successors {
            for s in succs {
                *in_degree.entry(s.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&k, _)| k)
            .collect();

        let mut order: Vec<String> = Vec::with_capacity(self.status.len());

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(succs) = self.successors.get(node) {
                for s in succs {
                    if let Some(deg) = in_degree.get_mut(s.as_str()) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(s.as_str());
                        }
                    }
                }
            }
        }

        if order.len() != self.status.len() {
            return Err(BatchError::DependencyError(
                "Cycle detected during topological sort".to_string(),
            ));
        }

        Ok(order)
    }

    // -- Execution order convenience ----------------------------------------

    /// Return jobs in a valid execution order, respecting dependencies.
    /// This is a wrapper around [`topological_sort`](Self::topological_sort).
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::DependencyError`] if the graph contains a cycle.
    pub fn execution_order(&self) -> Result<Vec<String>> {
        self.topological_sort()
    }

    // -- Internal helpers ---------------------------------------------------

    /// Recompute status for a single node based on its predecessors.
    fn recompute_status(&mut self, node: &str) {
        let preds = match self.predecessors.get(node) {
            Some(p) => p.clone(),
            None => return,
        };

        // If any predecessor is Failed, this node is also Failed.
        let any_failed = preds
            .iter()
            .any(|p| self.status.get(p) == Some(&DependencyStatus::Failed));
        if any_failed {
            self.status.insert(node.to_string(), DependencyStatus::Failed);
            return;
        }

        // If all predecessors are Completed, this node is Ready.
        let all_completed = preds
            .iter()
            .all(|p| self.status.get(p) == Some(&DependencyStatus::Completed));

        if preds.is_empty() || all_completed {
            // Only promote to Ready if currently Pending.
            if self.status.get(node) == Some(&DependencyStatus::Pending) {
                self.status.insert(node.to_string(), DependencyStatus::Ready);
            }
        } else {
            // Some predecessors are not yet completed.
            if self.status.get(node) == Some(&DependencyStatus::Ready) {
                self.status.insert(node.to_string(), DependencyStatus::Pending);
            }
        }
    }

    /// Check whether adding edge `from -> to` would create a cycle.
    /// This is done by checking if there is already a path from `to` to `from`.
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        // BFS from `to` following successor edges.  If we reach `from`, it's
        // a cycle.
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to);
        visited.insert(to.to_string());

        while let Some(current) = queue.pop_front() {
            if current == from {
                return true;
            }
            if let Some(succs) = self.successors.get(current) {
                for s in succs {
                    if visited.insert(s.clone()) {
                        queue.push_back(s.as_str());
                    }
                }
            }
        }
        false
    }

    /// All transitive dependents of a node (BFS through successors).
    fn transitive_dependents(&self, node: &str) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(succs) = self.successors.get(node) {
            for s in succs {
                if visited.insert(s.clone()) {
                    queue.push_back(s.clone());
                }
            }
        }
        while let Some(current) = queue.pop_front() {
            if let Some(succs) = self.successors.get(&current) {
                for s in succs {
                    if visited.insert(s.clone()) {
                        queue.push_back(s.clone());
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

    fn jid(s: &str) -> JobId {
        JobId::from(s)
    }

    #[test]
    fn test_register_job_starts_ready() {
        let mut mgr = JobDependencyManager::new();
        mgr.register_job(&jid("a"));
        assert_eq!(
            mgr.status(&jid("a")).expect("should exist"),
            DependencyStatus::Ready
        );
    }

    #[test]
    fn test_add_dependency_makes_dependent_pending() {
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        assert_eq!(
            mgr.status(&jid("b")).expect("exists"),
            DependencyStatus::Pending
        );
        assert_eq!(
            mgr.status(&jid("a")).expect("exists"),
            DependencyStatus::Ready
        );
    }

    #[test]
    fn test_circular_dependency_detected() {
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        let result = mgr.add_dependency(&jid("a"), &jid("b"));
        assert!(result.is_err());
    }

    #[test]
    fn test_self_dependency_rejected() {
        let mut mgr = JobDependencyManager::new();
        let result = mgr.add_dependency(&jid("x"), &jid("x"));
        assert!(result.is_err());
    }

    #[test]
    fn test_mark_completed_propagates_readiness() {
        let mut mgr = JobDependencyManager::new();
        // b depends on a
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        assert_eq!(
            mgr.status(&jid("b")).expect("exists"),
            DependencyStatus::Pending
        );
        mgr.mark_completed(&jid("a")).expect("ok");
        assert_eq!(
            mgr.status(&jid("b")).expect("exists"),
            DependencyStatus::Ready
        );
    }

    #[test]
    fn test_mark_failed_cascades() {
        let mut mgr = JobDependencyManager::new();
        // c depends on b, b depends on a
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("c"), &jid("b")).expect("ok");
        mgr.mark_failed(&jid("a")).expect("ok");
        assert_eq!(
            mgr.status(&jid("b")).expect("exists"),
            DependencyStatus::Failed
        );
        assert_eq!(
            mgr.status(&jid("c")).expect("exists"),
            DependencyStatus::Failed
        );
    }

    #[test]
    fn test_topological_sort_linear_chain() {
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("c"), &jid("b")).expect("ok");
        let order = mgr.topological_sort().expect("ok");
        let pos_a = order.iter().position(|x| x == "a").expect("a in order");
        let pos_b = order.iter().position(|x| x == "b").expect("b in order");
        let pos_c = order.iter().position(|x| x == "c").expect("c in order");
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_fan_out_pattern() {
        // a -> b, a -> c, a -> d (fan-out from a)
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("c"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("d"), &jid("a")).expect("ok");
        assert_eq!(mgr.fan_out(&jid("a")), 3);
        assert_eq!(mgr.fan_in(&jid("b")), 1);
        // b, c, d should all be Pending
        assert_eq!(mgr.pending_jobs().len(), 3);
        // Completing a makes all three Ready
        mgr.mark_completed(&jid("a")).expect("ok");
        assert_eq!(mgr.ready_jobs().len(), 3);
    }

    #[test]
    fn test_fan_in_pattern() {
        // a, b, c -> d (fan-in to d)
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("d"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("d"), &jid("b")).expect("ok");
        mgr.add_dependency(&jid("d"), &jid("c")).expect("ok");
        assert_eq!(mgr.fan_in(&jid("d")), 3);
        // d is Pending
        assert_eq!(
            mgr.status(&jid("d")).expect("exists"),
            DependencyStatus::Pending
        );
        // Complete a and b — d still pending (c not done)
        mgr.mark_completed(&jid("a")).expect("ok");
        mgr.mark_completed(&jid("b")).expect("ok");
        assert_eq!(
            mgr.status(&jid("d")).expect("exists"),
            DependencyStatus::Pending
        );
        // Complete c — d becomes Ready
        mgr.mark_completed(&jid("c")).expect("ok");
        assert_eq!(
            mgr.status(&jid("d")).expect("exists"),
            DependencyStatus::Ready
        );
    }

    #[test]
    fn test_execution_order_respects_deps() {
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("deploy"), &jid("build")).expect("ok");
        mgr.add_dependency(&jid("deploy"), &jid("test")).expect("ok");
        mgr.add_dependency(&jid("test"), &jid("build")).expect("ok");
        let order = mgr.execution_order().expect("ok");
        let pos = |name: &str| order.iter().position(|x| x == name).expect("in order");
        assert!(pos("build") < pos("test"));
        assert!(pos("test") < pos("deploy"));
    }

    #[test]
    fn test_dependency_status_display() {
        assert_eq!(DependencyStatus::Pending.to_string(), "Pending");
        assert_eq!(DependencyStatus::Ready.to_string(), "Ready");
        assert_eq!(DependencyStatus::Completed.to_string(), "Completed");
        assert_eq!(DependencyStatus::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_status_unknown_job_returns_error() {
        let mgr = JobDependencyManager::new();
        assert!(mgr.status(&jid("nonexistent")).is_err());
    }

    #[test]
    fn test_register_idempotent() {
        let mut mgr = JobDependencyManager::new();
        mgr.register_job(&jid("x"));
        mgr.register_job(&jid("x")); // no-op
        assert_eq!(mgr.job_count(), 1);
    }

    #[test]
    fn test_three_node_cycle_detected() {
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("c"), &jid("b")).expect("ok");
        let result = mgr.add_dependency(&jid("a"), &jid("c"));
        assert!(result.is_err());
    }

    #[test]
    fn test_diamond_dag() {
        //     a
        //    / \
        //   b   c
        //    \ /
        //     d
        let mut mgr = JobDependencyManager::new();
        mgr.add_dependency(&jid("b"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("c"), &jid("a")).expect("ok");
        mgr.add_dependency(&jid("d"), &jid("b")).expect("ok");
        mgr.add_dependency(&jid("d"), &jid("c")).expect("ok");
        assert_eq!(mgr.job_count(), 4);
        let order = mgr.topological_sort().expect("ok");
        assert_eq!(order.len(), 4);
        let pos = |n: &str| order.iter().position(|x| x == n).expect("in order");
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
    }
}
