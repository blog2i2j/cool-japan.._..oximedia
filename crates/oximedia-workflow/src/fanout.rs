//! Fan-out / fan-in parallel execution pattern for OxiMedia workflows.
//!
//! The fan-out / fan-in pattern allows a single parent workflow step to spawn
//! multiple child tasks in parallel and then merge the results back once a
//! configurable completion condition is met.
//!
//! # Merge strategies
//!
//! | Strategy          | Description                                               |
//! |-------------------|-----------------------------------------------------------|
//! | [`MergeStrategy::WaitAll`]  | Block until **every** child task completes.  |
//! | [`MergeStrategy::WaitAny`]  | Unblock as soon as **any one** child completes.|
//! | [`MergeStrategy::WaitN`]    | Unblock when at least `n` children complete.  |
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::fanout::{
//!     FanOutExecutor, FanOutGroup, FanOutTask, MergeStrategy,
//! };
//!
//! let mut exec = FanOutExecutor::new();
//!
//! let group = FanOutGroup {
//!     parent_step: "encode-variants".to_string(),
//!     children: vec![
//!         FanOutTask { id: "t-1080p".to_string(), parent_id: "encode-variants".to_string(), payload: r#"{"profile":"1080p"}"#.to_string() },
//!         FanOutTask { id: "t-720p".to_string(),  parent_id: "encode-variants".to_string(), payload: r#"{"profile":"720p"}"#.to_string()  },
//!         FanOutTask { id: "t-480p".to_string(),  parent_id: "encode-variants".to_string(), payload: r#"{"profile":"480p"}"#.to_string()  },
//!     ],
//!     merge_strategy: MergeStrategy::WaitAll,
//! };
//!
//! exec.submit_group(group);
//! assert!(!exec.is_merge_condition_met("encode-variants"));
//!
//! exec.mark_complete("t-1080p");
//! exec.mark_complete("t-720p");
//! exec.mark_complete("t-480p");
//! assert!(exec.is_merge_condition_met("encode-variants"));
//! ```

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// FanOutTask
// ---------------------------------------------------------------------------

/// A single child task spawned by a fan-out operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FanOutTask {
    /// Unique identifier for this task within the entire workflow.
    pub id: String,
    /// ID of the parent step that spawned this task.
    pub parent_id: String,
    /// Opaque payload passed to the task handler (e.g. a JSON string).
    pub payload: String,
}

impl FanOutTask {
    /// Create a new task with the given id, parent id, and payload.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        parent_id: impl Into<String>,
        payload: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            parent_id: parent_id.into(),
            payload: payload.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// MergeStrategy
// ---------------------------------------------------------------------------

/// Determines when a fan-out group is considered "done" and execution can
/// continue to the fan-in step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Wait until **all** child tasks have completed.
    WaitAll,
    /// Unblock as soon as **any** child task completes.
    WaitAny,
    /// Unblock when at least `n` child tasks have completed.
    WaitN(usize),
}

// ---------------------------------------------------------------------------
// FanOutGroup
// ---------------------------------------------------------------------------

/// A named group of child tasks associated with a single parent step.
#[derive(Debug, Clone)]
pub struct FanOutGroup {
    /// The parent step that owns this fan-out group.  Used as the key in
    /// [`FanOutExecutor`].
    pub parent_step: String,
    /// The child tasks to run in parallel.
    pub children: Vec<FanOutTask>,
    /// The strategy used to decide when the group is done.
    pub merge_strategy: MergeStrategy,
}

// ---------------------------------------------------------------------------
// FanOutGroupState (internal)
// ---------------------------------------------------------------------------

/// Internal runtime state tracked by [`FanOutExecutor`] for one group.
#[derive(Debug)]
struct FanOutGroupState {
    group: FanOutGroup,
    /// IDs of tasks that have reported completion.
    completed: HashSet<String>,
}

impl FanOutGroupState {
    fn new(group: FanOutGroup) -> Self {
        Self {
            group,
            completed: HashSet::new(),
        }
    }

    /// Record that `task_id` has finished.  No-op if the id is unknown.
    fn mark_complete(&mut self, task_id: &str) {
        // Only mark tasks that actually belong to this group.
        if self.group.children.iter().any(|t| t.id == task_id) {
            self.completed.insert(task_id.to_string());
        }
    }

    /// Check whether the merge condition is satisfied.
    fn is_done(&self) -> bool {
        let n_complete = self.completed.len();
        let n_total = self.group.children.len();

        match &self.group.merge_strategy {
            MergeStrategy::WaitAll => n_complete >= n_total,
            MergeStrategy::WaitAny => n_complete >= 1,
            MergeStrategy::WaitN(n) => n_complete >= *n,
        }
    }

    /// Return the number of completed tasks.
    fn completed_count(&self) -> usize {
        self.completed.len()
    }

    /// Return the total number of child tasks in the group.
    fn total_count(&self) -> usize {
        self.group.children.len()
    }
}

// ---------------------------------------------------------------------------
// FanOutExecutor
// ---------------------------------------------------------------------------

/// Manages multiple [`FanOutGroup`]s and tracks their completion state.
///
/// This type is **not** an async runtime; it is a pure bookkeeping layer.
/// The caller is responsible for actually executing child tasks (e.g. via
/// a thread pool or Tokio task set) and calling [`mark_complete`] when each
/// one finishes.
///
/// [`mark_complete`]: FanOutExecutor::mark_complete
#[derive(Debug, Default)]
pub struct FanOutExecutor {
    /// Groups indexed by `parent_step`.
    groups: HashMap<String, FanOutGroupState>,
    /// Reverse map: `task_id -> parent_step` for O(1) mark-complete routing.
    task_to_group: HashMap<String, String>,
}

impl FanOutExecutor {
    /// Create a new, empty executor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new [`FanOutGroup`].
    ///
    /// If a group with the same `parent_step` already exists it is replaced
    /// and all previous completion records for it are discarded.
    pub fn submit_group(&mut self, group: FanOutGroup) {
        // Remove stale task→group mappings for the old group (if any).
        if let Some(old_state) = self.groups.get(&group.parent_step) {
            for task in &old_state.group.children {
                self.task_to_group.remove(&task.id);
            }
        }

        // Build task→group mappings.
        for task in &group.children {
            self.task_to_group
                .insert(task.id.clone(), group.parent_step.clone());
        }

        self.groups
            .insert(group.parent_step.clone(), FanOutGroupState::new(group));
    }

    /// Mark a child task as complete.
    ///
    /// The executor looks up the group the task belongs to and records the
    /// completion there.  Unknown task IDs are silently ignored.
    pub fn mark_complete(&mut self, task_id: &str) {
        if let Some(group_id) = self.task_to_group.get(task_id).cloned() {
            if let Some(state) = self.groups.get_mut(&group_id) {
                state.mark_complete(task_id);
            }
        }
    }

    /// Return `true` when the merge condition for the group identified by
    /// `group_id` (i.e. `parent_step`) is satisfied.
    ///
    /// Returns `false` for unknown groups.
    #[must_use]
    pub fn is_merge_condition_met(&self, group_id: &str) -> bool {
        self.groups
            .get(group_id)
            .map(|s| s.is_done())
            .unwrap_or(false)
    }

    /// Return the number of tasks that have completed in a group.
    ///
    /// Returns `0` for unknown groups.
    #[must_use]
    pub fn completed_count(&self, group_id: &str) -> usize {
        self.groups
            .get(group_id)
            .map(|s| s.completed_count())
            .unwrap_or(0)
    }

    /// Return the total number of child tasks in a group.
    ///
    /// Returns `0` for unknown groups.
    #[must_use]
    pub fn total_count(&self, group_id: &str) -> usize {
        self.groups
            .get(group_id)
            .map(|s| s.total_count())
            .unwrap_or(0)
    }

    /// Return the number of registered groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Remove a group and its task mappings, returning the group if present.
    pub fn remove_group(&mut self, group_id: &str) -> Option<FanOutGroup> {
        if let Some(state) = self.groups.remove(group_id) {
            for task in &state.group.children {
                self.task_to_group.remove(&task.id);
            }
            Some(state.group)
        } else {
            None
        }
    }
}

// ===========================================================================
// FanoutStep — simple fan-out convenience wrapper
// ===========================================================================

/// A simple fan-out step that distributes a single input across multiple named branches.
///
/// Each branch receives a copy of the input tagged with the branch name. This is
/// a higher-level convenience API built on top of [`FanOutExecutor`].
///
/// # Example
///
/// ```rust
/// use oximedia_workflow::fanout::FanoutStep;
///
/// let step = FanoutStep::new(vec!["1080p".to_string(), "720p".to_string()]);
/// let results = step.execute("source.mp4");
/// assert_eq!(results.len(), 2);
/// assert!(results[0].contains("1080p"));
/// ```
#[derive(Debug, Clone)]
pub struct FanoutStep {
    /// Named branches to fan out to.
    pub branches: Vec<String>,
}

impl FanoutStep {
    /// Create a new fan-out step with the given branch names.
    #[must_use]
    pub fn new(branches: Vec<String>) -> Self {
        Self { branches }
    }

    /// Distribute `input` to all branches, returning one result string per branch.
    ///
    /// Each result is formatted as `"{branch}:{input}"` so the caller can identify
    /// which branch produced which output.
    #[must_use]
    pub fn execute(&self, input: &str) -> Vec<String> {
        self.branches
            .iter()
            .map(|branch| format!("{branch}:{input}"))
            .collect()
    }

    /// Return the number of configured branches.
    #[must_use]
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn three_task_group(strategy: MergeStrategy) -> FanOutGroup {
        FanOutGroup {
            parent_step: "encode-variants".to_string(),
            children: vec![
                FanOutTask::new("t-1080p", "encode-variants", r#"{"profile":"1080p"}"#),
                FanOutTask::new("t-720p", "encode-variants", r#"{"profile":"720p"}"#),
                FanOutTask::new("t-480p", "encode-variants", r#"{"profile":"480p"}"#),
            ],
            merge_strategy: strategy,
        }
    }

    // ------------------------------------------------------------------
    // WaitAll
    // ------------------------------------------------------------------

    #[test]
    fn wait_all_not_done_until_all_complete() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));

        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-1080p");
        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-720p");
        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-480p");
        assert!(exec.is_merge_condition_met("encode-variants"));
    }

    #[test]
    fn wait_all_idempotent_double_complete() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));

        exec.mark_complete("t-1080p");
        exec.mark_complete("t-1080p"); // duplicate — should not double-count
        exec.mark_complete("t-720p");
        exec.mark_complete("t-480p");

        assert!(exec.is_merge_condition_met("encode-variants"));
        assert_eq!(exec.completed_count("encode-variants"), 3);
    }

    // ------------------------------------------------------------------
    // WaitAny
    // ------------------------------------------------------------------

    #[test]
    fn wait_any_done_after_first_complete() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAny));

        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-720p");
        assert!(exec.is_merge_condition_met("encode-variants"));
    }

    // ------------------------------------------------------------------
    // WaitN
    // ------------------------------------------------------------------

    #[test]
    fn wait_n_done_after_n_completions() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitN(2)));

        exec.mark_complete("t-1080p");
        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-480p");
        assert!(exec.is_merge_condition_met("encode-variants"));
    }

    #[test]
    fn wait_n_1_same_as_wait_any() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitN(1)));

        assert!(!exec.is_merge_condition_met("encode-variants"));
        exec.mark_complete("t-1080p");
        assert!(exec.is_merge_condition_met("encode-variants"));
    }

    // ------------------------------------------------------------------
    // Unknown group / task
    // ------------------------------------------------------------------

    #[test]
    fn unknown_group_returns_false() {
        let exec = FanOutExecutor::new();
        assert!(!exec.is_merge_condition_met("non-existent-group"));
    }

    #[test]
    fn unknown_task_id_is_silently_ignored() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));
        exec.mark_complete("ghost-task-xyz"); // should not panic or break state
        assert!(!exec.is_merge_condition_met("encode-variants"));
        assert_eq!(exec.completed_count("encode-variants"), 0);
    }

    // ------------------------------------------------------------------
    // Multiple groups
    // ------------------------------------------------------------------

    #[test]
    fn multiple_groups_independent() {
        let mut exec = FanOutExecutor::new();

        exec.submit_group(FanOutGroup {
            parent_step: "group-a".to_string(),
            children: vec![FanOutTask::new("a1", "group-a", "")],
            merge_strategy: MergeStrategy::WaitAll,
        });
        exec.submit_group(FanOutGroup {
            parent_step: "group-b".to_string(),
            children: vec![
                FanOutTask::new("b1", "group-b", ""),
                FanOutTask::new("b2", "group-b", ""),
            ],
            merge_strategy: MergeStrategy::WaitAll,
        });

        exec.mark_complete("a1");
        assert!(exec.is_merge_condition_met("group-a"));
        assert!(!exec.is_merge_condition_met("group-b"));

        exec.mark_complete("b1");
        exec.mark_complete("b2");
        assert!(exec.is_merge_condition_met("group-b"));
    }

    // ------------------------------------------------------------------
    // Group replacement
    // ------------------------------------------------------------------

    #[test]
    fn submit_group_replaces_existing() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));
        exec.mark_complete("t-1080p");
        assert_eq!(exec.completed_count("encode-variants"), 1);

        // Replace with a fresh group — old completion records must be gone.
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));
        assert_eq!(exec.completed_count("encode-variants"), 0);
        assert!(!exec.is_merge_condition_met("encode-variants"));
    }

    // ------------------------------------------------------------------
    // Remove group
    // ------------------------------------------------------------------

    #[test]
    fn remove_group_clears_task_mappings() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));

        let removed = exec.remove_group("encode-variants");
        assert!(removed.is_some());
        assert_eq!(exec.group_count(), 0);

        // Completing a task whose group was removed must be a no-op.
        exec.mark_complete("t-1080p");
        assert!(!exec.is_merge_condition_met("encode-variants"));
    }

    // ------------------------------------------------------------------
    // Empty group
    // ------------------------------------------------------------------

    #[test]
    fn wait_all_on_empty_group_is_immediately_done() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(FanOutGroup {
            parent_step: "empty-group".to_string(),
            children: vec![],
            merge_strategy: MergeStrategy::WaitAll,
        });
        // 0 completed >= 0 total → true
        assert!(exec.is_merge_condition_met("empty-group"));
    }

    #[test]
    fn wait_any_on_empty_group_is_not_done() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(FanOutGroup {
            parent_step: "empty-any".to_string(),
            children: vec![],
            merge_strategy: MergeStrategy::WaitAny,
        });
        // 0 completed < 1 → false
        assert!(!exec.is_merge_condition_met("empty-any"));
    }

    // ------------------------------------------------------------------
    // Count helpers
    // ------------------------------------------------------------------

    #[test]
    fn total_count_and_completed_count() {
        let mut exec = FanOutExecutor::new();
        exec.submit_group(three_task_group(MergeStrategy::WaitAll));

        assert_eq!(exec.total_count("encode-variants"), 3);
        assert_eq!(exec.completed_count("encode-variants"), 0);

        exec.mark_complete("t-480p");
        assert_eq!(exec.completed_count("encode-variants"), 1);
    }

    #[test]
    fn counts_for_unknown_group_are_zero() {
        let exec = FanOutExecutor::new();
        assert_eq!(exec.total_count("ghost"), 0);
        assert_eq!(exec.completed_count("ghost"), 0);
    }
}
