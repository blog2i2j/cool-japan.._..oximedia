//! Workflow checkpoint management for `oximedia-workflow`.
//!
//! [`CheckpointManager`] snapshots workflow execution state at configurable
//! intervals so that a failed workflow can be resumed from the last checkpoint
//! instead of being re-run from scratch.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Checkpoint policy
// ---------------------------------------------------------------------------

/// Describes when checkpoints should be captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CheckpointPolicy {
    /// Never checkpoint (fastest, but no recovery).
    Never,
    /// Checkpoint after every completed task.
    AfterEveryTask,
    /// Checkpoint at a fixed wall-clock interval.
    Interval,
    /// Checkpoint only at explicitly marked tasks.
    Explicit,
}

impl CheckpointPolicy {
    /// Returns a human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Never => "Never",
            Self::AfterEveryTask => "After Every Task",
            Self::Interval => "Interval",
            Self::Explicit => "Explicit",
        }
    }

    /// Returns all variants.
    #[must_use]
    pub const fn all() -> &'static [CheckpointPolicy] {
        &[
            Self::Never,
            Self::AfterEveryTask,
            Self::Interval,
            Self::Explicit,
        ]
    }
}

impl std::fmt::Display for CheckpointPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A snapshot of workflow execution state at a point in time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique checkpoint identifier.
    pub id: u64,
    /// ID of the workflow this checkpoint belongs to.
    pub workflow_id: String,
    /// Unix timestamp (seconds) when the checkpoint was taken.
    pub timestamp_secs: u64,
    /// IDs of tasks that had completed at checkpoint time.
    pub completed_tasks: Vec<String>,
    /// IDs of tasks that were running at checkpoint time.
    pub running_tasks: Vec<String>,
    /// Arbitrary key-value state to persist (e.g. intermediate outputs).
    pub state_data: HashMap<String, String>,
    /// Human-readable label (optional).
    pub label: Option<String>,
}

impl Checkpoint {
    /// Creates a new checkpoint with the current timestamp.
    #[must_use]
    pub fn new(id: u64, workflow_id: impl Into<String>) -> Self {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            id,
            workflow_id: workflow_id.into(),
            timestamp_secs: ts,
            completed_tasks: Vec::new(),
            running_tasks: Vec::new(),
            state_data: HashMap::new(),
            label: None,
        }
    }

    /// Sets a state key-value pair.
    pub fn set_state(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.state_data.insert(key.into(), value.into());
    }

    /// Gets a state value by key.
    #[must_use]
    pub fn get_state(&self, key: &str) -> Option<&String> {
        self.state_data.get(key)
    }

    /// Returns `true` if a given task was completed at this checkpoint.
    #[must_use]
    pub fn is_task_completed(&self, task_id: &str) -> bool {
        self.completed_tasks.iter().any(|t| t == task_id)
    }

    /// Returns the total number of tasks (completed + running).
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.completed_tasks.len() + self.running_tasks.len()
    }
}

// ---------------------------------------------------------------------------
// Checkpoint manager
// ---------------------------------------------------------------------------

/// Manages a series of checkpoints for one workflow.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    /// Active policy.
    policy: CheckpointPolicy,
    /// Interval in seconds (used only with `Interval` policy).
    interval_secs: u64,
    /// Stored checkpoints, newest last.
    checkpoints: Vec<Checkpoint>,
    /// Auto-incrementing ID counter.
    next_id: u64,
    /// Maximum number of checkpoints to retain (0 = unlimited).
    max_retained: usize,
}

impl Default for CheckpointManager {
    fn default() -> Self {
        Self {
            policy: CheckpointPolicy::AfterEveryTask,
            interval_secs: 60,
            checkpoints: Vec::new(),
            next_id: 1,
            max_retained: 0,
        }
    }
}

impl CheckpointManager {
    /// Creates a new manager with the given policy.
    #[must_use]
    pub fn new(policy: CheckpointPolicy) -> Self {
        Self {
            policy,
            ..Default::default()
        }
    }

    /// Creates a manager with a fixed interval policy.
    #[must_use]
    pub fn with_interval(interval_secs: u64) -> Self {
        Self {
            policy: CheckpointPolicy::Interval,
            interval_secs,
            ..Default::default()
        }
    }

    /// Sets the maximum number of checkpoints to retain.
    pub fn set_max_retained(&mut self, max: usize) {
        self.max_retained = max;
    }

    /// Returns the current policy.
    #[must_use]
    pub fn policy(&self) -> CheckpointPolicy {
        self.policy
    }

    /// Returns the number of stored checkpoints.
    #[must_use]
    pub fn count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Creates and stores a new checkpoint for the given workflow.
    pub fn create_checkpoint(&mut self, workflow_id: &str) -> &Checkpoint {
        let id = self.next_id;
        self.next_id += 1;
        let cp = Checkpoint::new(id, workflow_id);
        self.checkpoints.push(cp);
        self.enforce_retention();
        self.checkpoints
            .last()
            .expect("invariant: checkpoint just pushed above")
    }

    /// Returns the latest checkpoint, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&Checkpoint> {
        self.checkpoints.last()
    }

    /// Returns all checkpoints for a given workflow ID.
    #[must_use]
    pub fn checkpoints_for(&self, workflow_id: &str) -> Vec<&Checkpoint> {
        self.checkpoints
            .iter()
            .filter(|c| c.workflow_id == workflow_id)
            .collect()
    }

    /// Removes all checkpoints.
    pub fn clear(&mut self) {
        self.checkpoints.clear();
    }

    /// Returns the configured interval in seconds.
    #[must_use]
    pub fn interval_secs(&self) -> u64 {
        self.interval_secs
    }

    /// Enforces the retention limit by dropping the oldest checkpoints.
    fn enforce_retention(&mut self) {
        if self.max_retained > 0 && self.checkpoints.len() > self.max_retained {
            let excess = self.checkpoints.len() - self.max_retained;
            self.checkpoints.drain(0..excess);
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- CheckpointPolicy ---------------------------------------------------

    #[test]
    fn test_policy_label() {
        assert_eq!(CheckpointPolicy::Never.label(), "Never");
        assert_eq!(CheckpointPolicy::AfterEveryTask.label(), "After Every Task");
    }

    #[test]
    fn test_policy_display() {
        assert_eq!(format!("{}", CheckpointPolicy::Interval), "Interval");
    }

    #[test]
    fn test_policy_all() {
        assert_eq!(CheckpointPolicy::all().len(), 4);
    }

    // -- Checkpoint ---------------------------------------------------------

    #[test]
    fn test_checkpoint_new() {
        let cp = Checkpoint::new(1, "wf-001");
        assert_eq!(cp.id, 1);
        assert_eq!(cp.workflow_id, "wf-001");
        assert!(cp.completed_tasks.is_empty());
    }

    #[test]
    fn test_checkpoint_state() {
        let mut cp = Checkpoint::new(1, "wf-001");
        let out = std::env::temp_dir()
            .join("oximedia-workflow-cp-out.mp4")
            .to_string_lossy()
            .into_owned();
        cp.set_state("output_path", &out);
        assert_eq!(
            cp.get_state("output_path").expect("should succeed in test"),
            &out
        );
        assert!(cp.get_state("missing").is_none());
    }

    #[test]
    fn test_checkpoint_task_completed() {
        let mut cp = Checkpoint::new(1, "wf-001");
        cp.completed_tasks.push("task-a".to_string());
        assert!(cp.is_task_completed("task-a"));
        assert!(!cp.is_task_completed("task-b"));
    }

    #[test]
    fn test_checkpoint_task_count() {
        let mut cp = Checkpoint::new(1, "wf-001");
        cp.completed_tasks.push("a".to_string());
        cp.running_tasks.push("b".to_string());
        assert_eq!(cp.task_count(), 2);
    }

    // -- CheckpointManager --------------------------------------------------

    #[test]
    fn test_manager_default() {
        let m = CheckpointManager::default();
        assert_eq!(m.policy(), CheckpointPolicy::AfterEveryTask);
        assert_eq!(m.count(), 0);
    }

    #[test]
    fn test_manager_create_checkpoint() {
        let mut m = CheckpointManager::new(CheckpointPolicy::AfterEveryTask);
        m.create_checkpoint("wf-001");
        assert_eq!(m.count(), 1);
        assert_eq!(
            m.latest().expect("should succeed in test").workflow_id,
            "wf-001"
        );
    }

    #[test]
    fn test_manager_multiple_checkpoints() {
        let mut m = CheckpointManager::new(CheckpointPolicy::AfterEveryTask);
        m.create_checkpoint("wf-001");
        m.create_checkpoint("wf-001");
        m.create_checkpoint("wf-002");
        assert_eq!(m.count(), 3);
        assert_eq!(m.checkpoints_for("wf-001").len(), 2);
        assert_eq!(m.checkpoints_for("wf-002").len(), 1);
    }

    #[test]
    fn test_manager_retention_limit() {
        let mut m = CheckpointManager::new(CheckpointPolicy::AfterEveryTask);
        m.set_max_retained(2);
        m.create_checkpoint("wf-001");
        m.create_checkpoint("wf-001");
        m.create_checkpoint("wf-001");
        assert_eq!(m.count(), 2);
        // The oldest one (id=1) should have been dropped.
        assert_eq!(m.latest().expect("should succeed in test").id, 3);
    }

    #[test]
    fn test_manager_clear() {
        let mut m = CheckpointManager::new(CheckpointPolicy::AfterEveryTask);
        m.create_checkpoint("wf-001");
        m.clear();
        assert_eq!(m.count(), 0);
        assert!(m.latest().is_none());
    }

    #[test]
    fn test_manager_with_interval() {
        let m = CheckpointManager::with_interval(120);
        assert_eq!(m.policy(), CheckpointPolicy::Interval);
        assert_eq!(m.interval_secs(), 120);
    }

    #[test]
    fn test_manager_latest_none() {
        let m = CheckpointManager::new(CheckpointPolicy::Never);
        assert!(m.latest().is_none());
    }
}
