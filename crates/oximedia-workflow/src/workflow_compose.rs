//! Workflow composition: combine smaller workflows into larger meta-workflows.
//!
//! Provides tools for composing multiple independent workflows into a single
//! execution unit with cross-workflow dependencies, shared variables, and
//! unified lifecycle management.
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::workflow_compose::{MetaWorkflow, SubWorkflowRef, CrossDependency};
//! use oximedia_workflow::workflow::Workflow;
//!
//! let ingest_wf = Workflow::new("ingest");
//! let transcode_wf = Workflow::new("transcode");
//!
//! let mut meta = MetaWorkflow::new("full-pipeline");
//! let ingest_ref = meta.add_sub_workflow(ingest_wf);
//! let transcode_ref = meta.add_sub_workflow(transcode_wf);
//! meta.add_dependency(CrossDependency::new(ingest_ref, transcode_ref));
//!
//! let order = meta.execution_order().expect("should succeed");
//! assert_eq!(order.len(), 2);
//! ```

use crate::workflow::Workflow;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Reference handle to a sub-workflow within a meta-workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubWorkflowRef(usize);

impl SubWorkflowRef {
    /// Get the numeric index.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

impl std::fmt::Display for SubWorkflowRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sub-{}", self.0)
    }
}

/// A dependency between two sub-workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossDependency {
    /// The sub-workflow that must complete first.
    pub predecessor: SubWorkflowRef,
    /// The sub-workflow that depends on the predecessor.
    pub successor: SubWorkflowRef,
    /// Optional condition: only start successor if this condition on the
    /// predecessor's final variables is met.
    pub condition: Option<String>,
    /// Whether to pass variables from predecessor to successor.
    pub pass_variables: bool,
}

impl CrossDependency {
    /// Create a simple ordering dependency.
    #[must_use]
    pub fn new(predecessor: SubWorkflowRef, successor: SubWorkflowRef) -> Self {
        Self {
            predecessor,
            successor,
            condition: None,
            pass_variables: false,
        }
    }

    /// Add a condition to this dependency.
    #[must_use]
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }

    /// Enable variable passing from predecessor to successor.
    #[must_use]
    pub fn with_variable_passing(mut self) -> Self {
        self.pass_variables = true;
        self
    }
}

/// Error from meta-workflow operations.
#[derive(Debug, thiserror::Error)]
pub enum ComposeError {
    /// Cycle detected between sub-workflows.
    #[error("Cycle detected in meta-workflow dependencies")]
    CycleDetected,

    /// Referenced sub-workflow does not exist.
    #[error("Sub-workflow not found: {0}")]
    SubWorkflowNotFound(SubWorkflowRef),

    /// Duplicate sub-workflow name.
    #[error("Duplicate sub-workflow name: {0}")]
    DuplicateName(String),
}

/// Execution status of a sub-workflow within the meta-workflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubWorkflowStatus {
    /// Not yet started.
    Pending,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed(String),
    /// Skipped due to unsatisfied condition.
    Skipped,
}

impl SubWorkflowStatus {
    /// Whether this status is terminal.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Skipped)
    }
}

/// A meta-workflow that composes multiple sub-workflows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaWorkflow {
    /// Name of the meta-workflow.
    pub name: String,
    /// Description.
    pub description: String,
    /// Sub-workflows in insertion order.
    pub sub_workflows: Vec<Workflow>,
    /// Cross-workflow dependencies.
    pub dependencies: Vec<CrossDependency>,
    /// Per-sub-workflow execution status.
    pub statuses: HashMap<SubWorkflowRef, SubWorkflowStatus>,
    /// Shared variables accessible by all sub-workflows.
    pub shared_variables: HashMap<String, serde_json::Value>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl MetaWorkflow {
    /// Create a new empty meta-workflow.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            sub_workflows: Vec::new(),
            dependencies: Vec::new(),
            statuses: HashMap::new(),
            shared_variables: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Add a sub-workflow, returning its reference handle.
    pub fn add_sub_workflow(&mut self, workflow: Workflow) -> SubWorkflowRef {
        let idx = self.sub_workflows.len();
        let ref_handle = SubWorkflowRef(idx);
        self.statuses.insert(ref_handle, SubWorkflowStatus::Pending);
        self.sub_workflows.push(workflow);
        ref_handle
    }

    /// Add a cross-workflow dependency.
    ///
    /// # Errors
    ///
    /// Returns `ComposeError` if either reference is invalid.
    pub fn add_dependency(&mut self, dep: CrossDependency) -> Result<(), ComposeError> {
        if dep.predecessor.0 >= self.sub_workflows.len() {
            return Err(ComposeError::SubWorkflowNotFound(dep.predecessor));
        }
        if dep.successor.0 >= self.sub_workflows.len() {
            return Err(ComposeError::SubWorkflowNotFound(dep.successor));
        }
        self.dependencies.push(dep);
        Ok(())
    }

    /// Set a shared variable.
    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.shared_variables.insert(key.into(), value);
    }

    /// Get a sub-workflow by reference.
    #[must_use]
    pub fn get_sub_workflow(&self, ref_handle: SubWorkflowRef) -> Option<&Workflow> {
        self.sub_workflows.get(ref_handle.0)
    }

    /// Get a mutable sub-workflow.
    pub fn get_sub_workflow_mut(&mut self, ref_handle: SubWorkflowRef) -> Option<&mut Workflow> {
        self.sub_workflows.get_mut(ref_handle.0)
    }

    /// Total number of sub-workflows.
    #[must_use]
    pub fn sub_workflow_count(&self) -> usize {
        self.sub_workflows.len()
    }

    /// Total number of tasks across all sub-workflows.
    #[must_use]
    pub fn total_task_count(&self) -> usize {
        self.sub_workflows.iter().map(|w| w.tasks.len()).sum()
    }

    /// Compute execution order (topological sort of sub-workflows).
    ///
    /// # Errors
    ///
    /// Returns `ComposeError::CycleDetected` if dependencies form a cycle.
    pub fn execution_order(&self) -> Result<Vec<SubWorkflowRef>, ComposeError> {
        let n = self.sub_workflows.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

        for dep in &self.dependencies {
            adj[dep.predecessor.0].push(dep.successor.0);
            in_degree[dep.successor.0] += 1;
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(idx) = queue.pop_front() {
            order.push(SubWorkflowRef(idx));
            for &succ in &adj[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if order.len() != n {
            return Err(ComposeError::CycleDetected);
        }

        Ok(order)
    }

    /// Check if the dependency graph has a cycle.
    #[must_use]
    pub fn has_cycle(&self) -> bool {
        self.execution_order().is_err()
    }

    /// Return predecessors of a sub-workflow.
    #[must_use]
    pub fn predecessors(&self, ref_handle: SubWorkflowRef) -> Vec<SubWorkflowRef> {
        self.dependencies
            .iter()
            .filter(|d| d.successor == ref_handle)
            .map(|d| d.predecessor)
            .collect()
    }

    /// Return successors of a sub-workflow.
    #[must_use]
    pub fn successors(&self, ref_handle: SubWorkflowRef) -> Vec<SubWorkflowRef> {
        self.dependencies
            .iter()
            .filter(|d| d.predecessor == ref_handle)
            .map(|d| d.successor)
            .collect()
    }

    /// Get root sub-workflows (no predecessors).
    #[must_use]
    pub fn root_sub_workflows(&self) -> Vec<SubWorkflowRef> {
        let has_pred: HashSet<usize> = self.dependencies.iter().map(|d| d.successor.0).collect();
        (0..self.sub_workflows.len())
            .filter(|i| !has_pred.contains(i))
            .map(SubWorkflowRef)
            .collect()
    }

    /// Get leaf sub-workflows (no successors).
    #[must_use]
    pub fn leaf_sub_workflows(&self) -> Vec<SubWorkflowRef> {
        let has_succ: HashSet<usize> = self.dependencies.iter().map(|d| d.predecessor.0).collect();
        (0..self.sub_workflows.len())
            .filter(|i| !has_succ.contains(i))
            .map(SubWorkflowRef)
            .collect()
    }

    /// Mark a sub-workflow status.
    pub fn set_status(&mut self, ref_handle: SubWorkflowRef, status: SubWorkflowStatus) {
        self.statuses.insert(ref_handle, status);
    }

    /// Get the status of a sub-workflow.
    #[must_use]
    pub fn get_status(&self, ref_handle: SubWorkflowRef) -> Option<&SubWorkflowStatus> {
        self.statuses.get(&ref_handle)
    }

    /// Check whether all sub-workflows are complete (or skipped).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.statuses.values().all(SubWorkflowStatus::is_terminal)
    }

    /// Merge a finished meta-workflow's sub-workflows into a single flat `Workflow`.
    ///
    /// This creates a new workflow containing all tasks from all sub-workflows,
    /// preserving internal edges and adding edges between leaf tasks of
    /// predecessors and root tasks of successors as per dependencies.
    #[must_use]
    pub fn flatten(&self) -> Workflow {
        use crate::task::TaskId;

        let mut combined = Workflow::new(&self.name);
        combined.description = self.description.clone();
        combined.metadata = self.metadata.clone();

        // Track task ID remapping per sub-workflow
        let mut sub_task_maps: Vec<HashMap<TaskId, TaskId>> = Vec::new();

        for wf in &self.sub_workflows {
            let mut id_map = HashMap::new();

            for task in wf.tasks.values() {
                let mut new_task = task.clone();
                let new_id = TaskId::new();
                new_task.id = new_id;
                new_task.name = format!("{}/{}", wf.name, task.name);
                id_map.insert(task.id, new_id);
                combined.tasks.insert(new_id, new_task);
            }

            // Re-create internal edges with new IDs
            for edge in &wf.edges {
                if let (Some(&new_from), Some(&new_to)) =
                    (id_map.get(&edge.from), id_map.get(&edge.to))
                {
                    combined.edges.push(crate::workflow::Edge {
                        from: new_from,
                        to: new_to,
                        condition: edge.condition.clone(),
                    });
                }
            }

            sub_task_maps.push(id_map);
        }

        // Add cross-dependency edges: leaf tasks of predecessor -> root tasks of successor
        for dep in &self.dependencies {
            let pred_wf = &self.sub_workflows[dep.predecessor.0];
            let succ_wf = &self.sub_workflows[dep.successor.0];
            let pred_map = &sub_task_maps[dep.predecessor.0];
            let succ_map = &sub_task_maps[dep.successor.0];

            let pred_leaves = pred_wf.get_leaf_tasks();
            let succ_roots = succ_wf.get_root_tasks();

            for &leaf_id in &pred_leaves {
                for &root_id in &succ_roots {
                    if let (Some(&new_leaf), Some(&new_root)) =
                        (pred_map.get(&leaf_id), succ_map.get(&root_id))
                    {
                        combined.edges.push(crate::workflow::Edge {
                            from: new_leaf,
                            to: new_root,
                            condition: dep.condition.clone(),
                        });
                    }
                }
            }
        }

        combined
    }

    /// Compute summary metrics for the meta-workflow.
    #[must_use]
    pub fn summary(&self) -> MetaWorkflowSummary {
        let total_sub_workflows = self.sub_workflows.len();
        let total_tasks = self.total_task_count();
        let completed = self
            .statuses
            .values()
            .filter(|s| matches!(s, SubWorkflowStatus::Completed))
            .count();
        let failed = self
            .statuses
            .values()
            .filter(|s| matches!(s, SubWorkflowStatus::Failed(_)))
            .count();
        let pending = self
            .statuses
            .values()
            .filter(|s| matches!(s, SubWorkflowStatus::Pending))
            .count();
        let running = self
            .statuses
            .values()
            .filter(|s| matches!(s, SubWorkflowStatus::Running))
            .count();
        let skipped = self
            .statuses
            .values()
            .filter(|s| matches!(s, SubWorkflowStatus::Skipped))
            .count();

        MetaWorkflowSummary {
            name: self.name.clone(),
            total_sub_workflows,
            total_tasks,
            completed,
            failed,
            pending,
            running,
            skipped,
            is_complete: self.is_complete(),
        }
    }
}

/// Summary of a meta-workflow's execution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaWorkflowSummary {
    /// Name.
    pub name: String,
    /// Total sub-workflows.
    pub total_sub_workflows: usize,
    /// Total tasks across all sub-workflows.
    pub total_tasks: usize,
    /// Completed sub-workflows.
    pub completed: usize,
    /// Failed sub-workflows.
    pub failed: usize,
    /// Pending sub-workflows.
    pub pending: usize,
    /// Running sub-workflows.
    pub running: usize,
    /// Skipped sub-workflows.
    pub skipped: usize,
    /// Whether all sub-workflows are terminal.
    pub is_complete: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{Task, TaskType};
    use std::time::Duration;

    fn make_workflow(name: &str, task_count: usize) -> Workflow {
        let mut wf = Workflow::new(name);
        let mut prev_id = None;
        for i in 0..task_count {
            let task = Task::new(
                format!("{name}-task-{i}"),
                TaskType::Wait {
                    duration: Duration::from_secs(1),
                },
            );
            let id = wf.add_task(task);
            if let Some(prev) = prev_id {
                wf.add_edge(prev, id).expect("add edge");
            }
            prev_id = Some(id);
        }
        wf
    }

    #[test]
    fn test_meta_workflow_creation() {
        let meta = MetaWorkflow::new("pipeline");
        assert_eq!(meta.name, "pipeline");
        assert_eq!(meta.sub_workflow_count(), 0);
    }

    #[test]
    fn test_add_sub_workflow() {
        let mut meta = MetaWorkflow::new("pipeline");
        let wf = make_workflow("ingest", 3);
        let ref_handle = meta.add_sub_workflow(wf);

        assert_eq!(ref_handle.index(), 0);
        assert_eq!(meta.sub_workflow_count(), 1);
        assert!(meta.get_sub_workflow(ref_handle).is_some());
    }

    #[test]
    fn test_cross_dependency() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("ingest", 2));
        let r2 = meta.add_sub_workflow(make_workflow("transcode", 2));

        assert!(meta.add_dependency(CrossDependency::new(r1, r2)).is_ok());
        assert_eq!(meta.predecessors(r2), vec![r1]);
        assert_eq!(meta.successors(r1), vec![r2]);
    }

    #[test]
    fn test_invalid_dependency() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("ingest", 1));
        let bad_ref = SubWorkflowRef(99);

        assert!(meta
            .add_dependency(CrossDependency::new(r1, bad_ref))
            .is_err());
    }

    #[test]
    fn test_execution_order_linear() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("ingest", 1));
        let r2 = meta.add_sub_workflow(make_workflow("transcode", 1));
        let r3 = meta.add_sub_workflow(make_workflow("deliver", 1));

        meta.add_dependency(CrossDependency::new(r1, r2))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r2, r3))
            .expect("dep");

        let order = meta.execution_order().expect("order");
        assert_eq!(order.len(), 3);

        let pos1 = order.iter().position(|&x| x == r1).expect("find r1");
        let pos2 = order.iter().position(|&x| x == r2).expect("find r2");
        let pos3 = order.iter().position(|&x| x == r3).expect("find r3");
        assert!(pos1 < pos2);
        assert!(pos2 < pos3);
    }

    #[test]
    fn test_execution_order_cycle_detection() {
        let mut meta = MetaWorkflow::new("cycle");
        let r1 = meta.add_sub_workflow(make_workflow("a", 1));
        let r2 = meta.add_sub_workflow(make_workflow("b", 1));

        meta.add_dependency(CrossDependency::new(r1, r2))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r2, r1))
            .expect("dep");

        assert!(meta.has_cycle());
        assert!(meta.execution_order().is_err());
    }

    #[test]
    fn test_root_and_leaf_sub_workflows() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("a", 1));
        let r2 = meta.add_sub_workflow(make_workflow("b", 1));
        let r3 = meta.add_sub_workflow(make_workflow("c", 1));

        meta.add_dependency(CrossDependency::new(r1, r2))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r2, r3))
            .expect("dep");

        assert_eq!(meta.root_sub_workflows(), vec![r1]);
        assert_eq!(meta.leaf_sub_workflows(), vec![r3]);
    }

    #[test]
    fn test_total_task_count() {
        let mut meta = MetaWorkflow::new("pipeline");
        meta.add_sub_workflow(make_workflow("ingest", 3));
        meta.add_sub_workflow(make_workflow("transcode", 5));

        assert_eq!(meta.total_task_count(), 8);
    }

    #[test]
    fn test_status_tracking() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("a", 1));
        let r2 = meta.add_sub_workflow(make_workflow("b", 1));

        assert!(!meta.is_complete());

        meta.set_status(r1, SubWorkflowStatus::Completed);
        assert!(!meta.is_complete());

        meta.set_status(r2, SubWorkflowStatus::Completed);
        assert!(meta.is_complete());
    }

    #[test]
    fn test_status_failed() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("a", 1));

        meta.set_status(r1, SubWorkflowStatus::Failed("timeout".to_string()));
        assert!(meta.is_complete());

        let status = meta.get_status(r1).expect("status");
        assert!(matches!(status, SubWorkflowStatus::Failed(msg) if msg == "timeout"));
    }

    #[test]
    fn test_flatten_creates_combined_workflow() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("ingest", 2));
        let r2 = meta.add_sub_workflow(make_workflow("transcode", 2));

        meta.add_dependency(CrossDependency::new(r1, r2))
            .expect("dep");

        let flat = meta.flatten();
        assert_eq!(flat.tasks.len(), 4);
        // Internal edges (2 * 1) + cross edges (1 leaf * 1 root = 1)
        assert_eq!(flat.edges.len(), 3);
        assert!(!flat.has_cycle());
    }

    #[test]
    fn test_flatten_preserves_names() {
        let mut meta = MetaWorkflow::new("pipeline");
        meta.add_sub_workflow(make_workflow("ingest", 1));

        let flat = meta.flatten();
        let task = flat.tasks.values().next().expect("task");
        assert!(task.name.starts_with("ingest/"));
    }

    #[test]
    fn test_flatten_conditional_dependency() {
        let mut meta = MetaWorkflow::new("cond");
        let r1 = meta.add_sub_workflow(make_workflow("check", 1));
        let r2 = meta.add_sub_workflow(make_workflow("process", 1));

        let dep = CrossDependency::new(r1, r2).with_condition("status == success");
        meta.add_dependency(dep).expect("dep");

        let flat = meta.flatten();
        let cross_edge = flat
            .edges
            .iter()
            .find(|e| e.condition.is_some())
            .expect("conditional edge");
        assert_eq!(cross_edge.condition, Some("status == success".to_string()));
    }

    #[test]
    fn test_shared_variables() {
        let mut meta = MetaWorkflow::new("pipeline");
        let input = std::env::temp_dir()
            .join("oximedia-workflow-compose-input.mp4")
            .to_string_lossy()
            .into_owned();
        meta.set_variable("input_path", serde_json::json!(input));

        assert_eq!(
            meta.shared_variables.get("input_path"),
            Some(&serde_json::json!(input))
        );
    }

    #[test]
    fn test_summary() {
        let mut meta = MetaWorkflow::new("pipeline");
        let r1 = meta.add_sub_workflow(make_workflow("a", 3));
        let _r2 = meta.add_sub_workflow(make_workflow("b", 2));

        meta.set_status(r1, SubWorkflowStatus::Completed);

        let summary = meta.summary();
        assert_eq!(summary.total_sub_workflows, 2);
        assert_eq!(summary.total_tasks, 5);
        assert_eq!(summary.completed, 1);
        assert_eq!(summary.pending, 1);
        assert!(!summary.is_complete);
    }

    #[test]
    fn test_diamond_dependency() {
        let mut meta = MetaWorkflow::new("diamond");
        let r1 = meta.add_sub_workflow(make_workflow("start", 1));
        let r2 = meta.add_sub_workflow(make_workflow("branch_a", 1));
        let r3 = meta.add_sub_workflow(make_workflow("branch_b", 1));
        let r4 = meta.add_sub_workflow(make_workflow("join", 1));

        meta.add_dependency(CrossDependency::new(r1, r2))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r1, r3))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r2, r4))
            .expect("dep");
        meta.add_dependency(CrossDependency::new(r3, r4))
            .expect("dep");

        assert!(!meta.has_cycle());
        let order = meta.execution_order().expect("order");
        assert_eq!(order.len(), 4);

        let pos1 = order.iter().position(|&x| x == r1).expect("find");
        let pos2 = order.iter().position(|&x| x == r2).expect("find");
        let pos3 = order.iter().position(|&x| x == r3).expect("find");
        let pos4 = order.iter().position(|&x| x == r4).expect("find");
        assert!(pos1 < pos2 && pos1 < pos3);
        assert!(pos2 < pos4 && pos3 < pos4);
    }

    #[test]
    fn test_sub_workflow_ref_display() {
        let r = SubWorkflowRef(5);
        assert_eq!(r.to_string(), "sub-5");
    }
}
