//! Task checkpointing for fault tolerance in the render farm.
//!
//! Provides types for storing and retrieving task checkpoint data, enabling
//! interrupted tasks to resume from a known good state rather than restarting
//! from the beginning.

#![allow(dead_code)]

/// A snapshot of a task's state at a specific frame or processing step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointData {
    /// Identifier of the task this checkpoint belongs to.
    pub task_id: u64,
    /// The frame index or processing step at which the snapshot was taken.
    pub frame_or_step: u64,
    /// Serialised task state (opaque bytes).
    pub state_bytes: Vec<u8>,
    /// Epoch timestamp (seconds) when this checkpoint was created.
    pub created_epoch: u64,
}

impl CheckpointData {
    /// Create a new checkpoint.
    #[must_use]
    pub fn new(task_id: u64, frame_or_step: u64, state_bytes: Vec<u8>, created_epoch: u64) -> Self {
        Self {
            task_id,
            frame_or_step,
            state_bytes,
            created_epoch,
        }
    }

    /// Return the size of the serialised state in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.state_bytes.len()
    }

    /// Return `true` if this checkpoint was created within `max_age_secs` of `now`.
    #[must_use]
    pub fn is_recent(&self, now: u64, max_age_secs: u64) -> bool {
        now.saturating_sub(self.created_epoch) <= max_age_secs
    }
}

/// In-memory store for task checkpoints.
#[derive(Debug, Clone, Default)]
pub struct CheckpointStore {
    /// All stored checkpoints.
    pub checkpoints: Vec<CheckpointData>,
}

impl CheckpointStore {
    /// Create an empty checkpoint store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Save a checkpoint.  Existing checkpoints for the same task are kept.
    pub fn save(&mut self, cp: CheckpointData) {
        self.checkpoints.push(cp);
    }

    /// Return a reference to the most recent checkpoint for the given task,
    /// i.e. the one with the highest `frame_or_step` value.
    #[must_use]
    pub fn latest_for(&self, task_id: u64) -> Option<&CheckpointData> {
        self.checkpoints
            .iter()
            .filter(|cp| cp.task_id == task_id)
            .max_by_key(|cp| cp.frame_or_step)
    }

    /// Remove all checkpoints for the given task.
    pub fn remove_for(&mut self, task_id: u64) {
        self.checkpoints.retain(|cp| cp.task_id != task_id);
    }

    /// Return the total size of all stored checkpoint state in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.checkpoints
            .iter()
            .map(|cp| cp.size_bytes() as u64)
            .sum()
    }

    /// Return the number of distinct task IDs with at least one checkpoint.
    #[must_use]
    pub fn task_count(&self) -> usize {
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for cp in &self.checkpoints {
            seen.insert(cp.task_id);
        }
        seen.len()
    }
}

/// Policy governing when checkpoints are created and pruned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointPolicy {
    /// Checkpoint every N frames/steps.
    pub interval_frames: u64,
    /// Maximum number of checkpoints to retain per task before pruning.
    pub max_checkpoints_per_task: usize,
    /// Minimum age in seconds before a checkpoint may be pruned.
    pub min_age_secs_to_prune: u64,
}

impl CheckpointPolicy {
    /// Create a sensible default policy.
    #[must_use]
    pub fn default_policy() -> Self {
        Self {
            interval_frames: 100,
            max_checkpoints_per_task: 5,
            min_age_secs_to_prune: 3_600,
        }
    }

    /// Return `true` if a checkpoint should be taken at `current_frame` given
    /// that the last checkpoint was at `last_checkpoint_frame`.
    #[must_use]
    pub fn should_checkpoint(&self, current_frame: u64, last_checkpoint_frame: u64) -> bool {
        if self.interval_frames == 0 {
            return false;
        }
        current_frame.saturating_sub(last_checkpoint_frame) >= self.interval_frames
    }
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- CheckpointData tests ---

    #[test]
    fn test_checkpoint_data_size_bytes() {
        let cp = CheckpointData::new(1, 100, vec![0u8; 256], 1_000);
        assert_eq!(cp.size_bytes(), 256);
    }

    #[test]
    fn test_checkpoint_data_size_bytes_empty() {
        let cp = CheckpointData::new(1, 0, vec![], 0);
        assert_eq!(cp.size_bytes(), 0);
    }

    #[test]
    fn test_checkpoint_is_recent_within_age() {
        let cp = CheckpointData::new(1, 0, vec![], 1_000);
        // now = 1500, max_age = 600  → age = 500 ≤ 600
        assert!(cp.is_recent(1_500, 600));
    }

    #[test]
    fn test_checkpoint_is_not_recent_outside_age() {
        let cp = CheckpointData::new(1, 0, vec![], 1_000);
        // now = 2_700, max_age = 1_600  → age = 1_700 > 1_600
        assert!(!cp.is_recent(2_700, 1_600));
    }

    #[test]
    fn test_checkpoint_is_recent_exact_boundary() {
        let cp = CheckpointData::new(1, 0, vec![], 1_000);
        // now = 1_600, max_age = 600  → age = 600 ≤ 600
        assert!(cp.is_recent(1_600, 600));
    }

    // --- CheckpointStore tests ---

    #[test]
    fn test_store_starts_empty() {
        let store = CheckpointStore::new();
        assert_eq!(store.task_count(), 0);
        assert_eq!(store.total_size_bytes(), 0);
    }

    #[test]
    fn test_store_save_and_task_count() {
        let mut store = CheckpointStore::new();
        store.save(CheckpointData::new(1, 0, vec![0u8; 10], 0));
        store.save(CheckpointData::new(2, 0, vec![0u8; 20], 0));
        assert_eq!(store.task_count(), 2);
    }

    #[test]
    fn test_store_latest_for_returns_highest_frame() {
        let mut store = CheckpointStore::new();
        store.save(CheckpointData::new(1, 50, vec![1], 100));
        store.save(CheckpointData::new(1, 150, vec![2], 200));
        store.save(CheckpointData::new(1, 100, vec![3], 150));
        let latest = store.latest_for(1).expect("latest_for should succeed");
        assert_eq!(latest.frame_or_step, 150);
    }

    #[test]
    fn test_store_latest_for_returns_none_when_absent() {
        let store = CheckpointStore::new();
        assert!(store.latest_for(42).is_none());
    }

    #[test]
    fn test_store_remove_for_clears_task() {
        let mut store = CheckpointStore::new();
        store.save(CheckpointData::new(1, 0, vec![], 0));
        store.save(CheckpointData::new(2, 0, vec![], 0));
        store.remove_for(1);
        assert!(store.latest_for(1).is_none());
        assert!(store.latest_for(2).is_some());
        assert_eq!(store.task_count(), 1);
    }

    #[test]
    fn test_store_total_size_bytes() {
        let mut store = CheckpointStore::new();
        store.save(CheckpointData::new(1, 0, vec![0u8; 100], 0));
        store.save(CheckpointData::new(1, 1, vec![0u8; 200], 1));
        store.save(CheckpointData::new(2, 0, vec![0u8; 50], 0));
        assert_eq!(store.total_size_bytes(), 350);
    }

    #[test]
    fn test_store_same_task_multiple_checkpoints_counts_once() {
        let mut store = CheckpointStore::new();
        store.save(CheckpointData::new(5, 0, vec![], 0));
        store.save(CheckpointData::new(5, 100, vec![], 1));
        store.save(CheckpointData::new(5, 200, vec![], 2));
        assert_eq!(store.task_count(), 1);
    }

    // --- CheckpointPolicy tests ---

    #[test]
    fn test_default_policy_interval_100() {
        let policy = CheckpointPolicy::default();
        assert_eq!(policy.interval_frames, 100);
    }

    #[test]
    fn test_should_checkpoint_at_interval() {
        let policy = CheckpointPolicy::default_policy();
        // last at 0, current at 100 → exactly at interval
        assert!(policy.should_checkpoint(100, 0));
    }

    #[test]
    fn test_should_not_checkpoint_before_interval() {
        let policy = CheckpointPolicy::default_policy();
        assert!(!policy.should_checkpoint(99, 0));
    }

    #[test]
    fn test_should_checkpoint_past_interval() {
        let policy = CheckpointPolicy::default_policy();
        // last at 100, current at 250 → 150 ≥ 100
        assert!(policy.should_checkpoint(250, 100));
    }

    #[test]
    fn test_should_not_checkpoint_zero_interval() {
        let policy = CheckpointPolicy {
            interval_frames: 0,
            max_checkpoints_per_task: 5,
            min_age_secs_to_prune: 3_600,
        };
        assert!(!policy.should_checkpoint(1_000, 0));
    }
}
