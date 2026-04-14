//! Task checkpointing for fault tolerance in the render farm.
//!
//! Provides types for storing and retrieving task checkpoint data, enabling
//! interrupted tasks to resume from a known good state rather than restarting
//! from the beginning.
//!
//! # Incremental checkpointing
//!
//! In addition to full-snapshot checkpointing, this module offers *delta-based*
//! (incremental) checkpointing.  Instead of writing the entire task state on
//! every checkpoint frame, only the bytes that changed since the last base
//! snapshot are stored as `(offset, new_byte)` patches.  This dramatically
//! reduces I/O when the state is large but changes sparsely between frames.
//!
//! ## Usage sketch
//!
//! ```rust,ignore
//! let mut store = IncrementalCheckpointStore::new(task_id, initial_state, IncrementalCheckpointConfig::default());
//! // ... after each frame:
//! store.record(frame_idx, &new_state, epoch_secs)?;
//! // Recover the latest state:
//! let recovered = store.reconstruct_latest()?;
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Full-snapshot checkpointing ───────────────────────────────────────────────

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

// ── Incremental / delta checkpointing ─────────────────────────────────────────

/// A single byte-level patch: `(offset_into_state, new_value)`.
pub type BytePatch = (usize, u8);

/// A set of byte-level patches that describe how the task state changed
/// since the previous base snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointDelta {
    /// Task this delta belongs to.
    pub task_id: u64,
    /// Frame/step at which this delta was recorded.
    pub frame_or_step: u64,
    /// Byte-level patches relative to the previous base state.
    pub patches: Vec<BytePatch>,
    /// Epoch timestamp (seconds) when this delta was recorded.
    pub created_epoch: u64,
    /// Monotonically increasing sequence number within the task's delta chain.
    pub sequence: u32,
}

impl CheckpointDelta {
    /// Create a new delta record.
    #[must_use]
    pub fn new(
        task_id: u64,
        frame_or_step: u64,
        patches: Vec<BytePatch>,
        created_epoch: u64,
        sequence: u32,
    ) -> Self {
        Self {
            task_id,
            frame_or_step,
            patches,
            created_epoch,
            sequence,
        }
    }

    /// Number of bytes changed by this delta.
    #[must_use]
    pub fn patch_count(&self) -> usize {
        self.patches.len()
    }

    /// Estimated I/O saved vs. writing the full state of `full_state_bytes` bytes.
    ///
    /// Returns `0` if `full_state_bytes` is 0 or fewer bytes are stored than would
    /// be patched (pathological case where patching is more expensive).
    #[must_use]
    pub fn io_saving_bytes(&self, full_state_bytes: usize) -> usize {
        // Each patch stores 1 offset (usize = 8 bytes on 64-bit) + 1 byte value.
        let delta_bytes = self.patches.len() * 9;
        full_state_bytes.saturating_sub(delta_bytes)
    }
}

/// Compute the byte-level delta between `base` and `current`.
///
/// Only positions where `base[i] != current[i]` are included.  If `base` and
/// `current` have different lengths this returns an error.
///
/// # Errors
///
/// Returns `Err` if `base.len() != current.len()`.
pub fn compute_delta(
    task_id: u64,
    frame_or_step: u64,
    base: &[u8],
    current: &[u8],
    created_epoch: u64,
    sequence: u32,
) -> Result<CheckpointDelta, String> {
    if base.len() != current.len() {
        return Err(format!(
            "base length {} != current length {}; variable-length states require a full snapshot",
            base.len(),
            current.len()
        ));
    }
    let patches: Vec<BytePatch> = base
        .iter()
        .zip(current.iter())
        .enumerate()
        .filter_map(|(i, (b, c))| if b != c { Some((i, *c)) } else { None })
        .collect();
    Ok(CheckpointDelta::new(
        task_id,
        frame_or_step,
        patches,
        created_epoch,
        sequence,
    ))
}

/// Apply an ordered slice of deltas on top of `base`, returning the reconstructed state.
///
/// Deltas are applied in the order given (ascending `sequence` is assumed by the caller).
///
/// # Errors
///
/// Returns `Err` if any patch offset is out of bounds for the current state length.
pub fn apply_deltas(base: &[u8], deltas: &[CheckpointDelta]) -> Result<Vec<u8>, String> {
    let mut state = base.to_vec();
    for delta in deltas {
        for &(offset, byte) in &delta.patches {
            // Capture length before the mutable borrow to satisfy borrow checker.
            let state_len = state.len();
            *state.get_mut(offset).ok_or_else(|| {
                format!(
                    "patch offset {} out of bounds (state len={}, delta seq={})",
                    offset, state_len, delta.sequence
                )
            })? = byte;
        }
    }
    Ok(state)
}

/// Configuration for the incremental checkpoint store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IncrementalCheckpointConfig {
    /// Number of delta records to accumulate before writing a new base snapshot
    /// and clearing the delta chain.
    pub delta_chain_limit: u32,
    /// If `true`, patches with zero changed bytes (unchanged frames) are still
    /// recorded as empty deltas so that the frame sequence is complete.
    pub record_unchanged: bool,
}

impl IncrementalCheckpointConfig {
    /// Sensible defaults: roll over every 50 deltas, skip unchanged frames.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            delta_chain_limit: 50,
            record_unchanged: false,
        }
    }
}

impl Default for IncrementalCheckpointConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Outcome reported after calling [`IncrementalCheckpointStore::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// A new base snapshot was written (chain rolled over or first record).
    BaseWritten,
    /// A delta record was appended to the current chain.
    DeltaWritten,
    /// The state was unchanged and `record_unchanged` is `false`; nothing stored.
    Unchanged,
    /// The delta chain reached its limit; a new base was written (same as
    /// `BaseWritten` but indicates a scheduled roll-over).
    BaseRolledOver,
}

/// Statistics returned by [`IncrementalCheckpointStore::flush_stats`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlushStats {
    /// Total number of base snapshots written.
    pub base_count: u32,
    /// Total number of delta records written.
    pub delta_count: u32,
    /// Total patch bytes stored across all deltas.
    pub total_patch_bytes: u64,
    /// Estimated bytes saved vs. writing full snapshots every frame.
    pub estimated_io_saved_bytes: u64,
}

/// Per-task incremental checkpoint chain.
struct TaskChain {
    /// Current base snapshot bytes.
    base: Vec<u8>,
    /// Accumulated deltas since the last base snapshot.
    deltas: Vec<CheckpointDelta>,
    /// Next sequence number to assign to a delta.
    next_seq: u32,
    /// Number of base snapshots written (including the initial one).
    base_count: u32,
    /// Total patch bytes stored.
    total_patch_bytes: u64,
    /// Total full-state bytes that *would* have been stored without incremental.
    total_full_bytes: u64,
}

impl TaskChain {
    fn new(initial_state: Vec<u8>) -> Self {
        let base_len = initial_state.len() as u64;
        Self {
            base: initial_state,
            deltas: Vec::new(),
            next_seq: 0,
            base_count: 1,
            total_patch_bytes: 0,
            total_full_bytes: base_len, // the initial base counts as one full write
        }
    }
}

/// In-memory incremental checkpoint store for multiple tasks.
///
/// For each task it maintains a *base snapshot* plus a *delta chain*.  When the
/// chain reaches `config.delta_chain_limit`, the current state is baked into a
/// new base snapshot and the chain is cleared.
pub struct IncrementalCheckpointStore {
    /// Per-task chains, keyed by task ID.
    chains: HashMap<u64, TaskChain>,
    /// Shared configuration.
    config: IncrementalCheckpointConfig,
}

impl IncrementalCheckpointStore {
    /// Create a new store with no tasks registered.
    #[must_use]
    pub fn new(config: IncrementalCheckpointConfig) -> Self {
        Self {
            chains: HashMap::new(),
            config,
        }
    }

    /// Register a task with its initial state.  If the task was already
    /// registered the existing chain is replaced.
    pub fn register_task(&mut self, task_id: u64, initial_state: Vec<u8>) {
        self.chains.insert(task_id, TaskChain::new(initial_state));
    }

    /// Record a new state for `task_id` at `frame_or_step`.
    ///
    /// Internally this computes the delta against the chain's current base and
    /// either appends a delta record or rolls over to a new base snapshot.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - The task is not registered.
    /// - The new state length differs from the base length (variable-length
    ///   states are not supported by delta encoding; use a full snapshot store).
    pub fn record(
        &mut self,
        task_id: u64,
        new_state: &[u8],
        frame_or_step: u64,
        created_epoch: u64,
    ) -> Result<RecordOutcome, String> {
        let config = self.config.clone();
        let chain = self
            .chains
            .get_mut(&task_id)
            .ok_or_else(|| format!("task {task_id} not registered"))?;

        // Reconstruct the current accumulated state so we can diff against it.
        let current_base = chain.base.clone();
        let current_accumulated = apply_deltas(&current_base, &chain.deltas)?;

        // Detect length change — must fall back to a new base snapshot.
        if new_state.len() != current_accumulated.len() {
            // Roll over with a full base write regardless of chain depth.
            chain.base = new_state.to_vec();
            chain.deltas.clear();
            chain.next_seq = 0;
            chain.base_count += 1;
            chain.total_full_bytes += new_state.len() as u64;
            return Ok(RecordOutcome::BaseWritten);
        }

        let delta = compute_delta(
            task_id,
            frame_or_step,
            &current_accumulated,
            new_state,
            created_epoch,
            chain.next_seq,
        )?;

        if delta.patches.is_empty() && !config.record_unchanged {
            return Ok(RecordOutcome::Unchanged);
        }

        // Check whether a roll-over is needed.
        let roll_over = chain.deltas.len() as u32 >= config.delta_chain_limit;
        if roll_over {
            // Bake accumulated state into new base.
            chain.base = new_state.to_vec();
            chain.deltas.clear();
            chain.next_seq = 0;
            chain.base_count += 1;
            chain.total_full_bytes += new_state.len() as u64;
            return Ok(RecordOutcome::BaseRolledOver);
        }

        // Append delta.
        chain.total_patch_bytes += delta.patches.len() as u64 * 9; // offset(8)+byte(1)
        chain.total_full_bytes += new_state.len() as u64;
        chain.next_seq += 1;
        chain.deltas.push(delta);
        Ok(RecordOutcome::DeltaWritten)
    }

    /// Reconstruct the latest state for `task_id` by replaying all deltas.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the task is not registered or a delta is corrupt.
    pub fn reconstruct_latest(&self, task_id: u64) -> Result<Vec<u8>, String> {
        let chain = self
            .chains
            .get(&task_id)
            .ok_or_else(|| format!("task {task_id} not registered"))?;
        apply_deltas(&chain.base, &chain.deltas)
    }

    /// Return aggregate flush statistics across all registered tasks.
    #[must_use]
    pub fn flush_stats(&self) -> FlushStats {
        let mut base_count: u32 = 0;
        let mut delta_count: u32 = 0;
        let mut total_patch_bytes: u64 = 0;
        let mut total_full_bytes: u64 = 0;
        for chain in self.chains.values() {
            base_count += chain.base_count;
            delta_count += chain.deltas.len() as u32;
            total_patch_bytes += chain.total_patch_bytes;
            total_full_bytes += chain.total_full_bytes;
        }
        let estimated_io_saved_bytes = total_full_bytes.saturating_sub(
            // actual bytes written: bases (full) + patches
            total_full_bytes
                .saturating_sub(total_patch_bytes + (base_count as u64) * 64)
                .min(total_full_bytes),
        );
        FlushStats {
            base_count,
            delta_count,
            total_patch_bytes,
            estimated_io_saved_bytes,
        }
    }

    /// Remove all data for `task_id`.
    pub fn deregister_task(&mut self, task_id: u64) {
        self.chains.remove(&task_id);
    }

    /// Number of registered tasks.
    #[must_use]
    pub fn task_count(&self) -> usize {
        self.chains.len()
    }

    /// Return the number of deltas currently held for `task_id`, or `None`
    /// if the task is not registered.
    #[must_use]
    pub fn delta_count_for(&self, task_id: u64) -> Option<usize> {
        self.chains.get(&task_id).map(|c| c.deltas.len())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

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
        let latest = store
            .latest_for(1)
            .expect("latest_for should return a value");
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

    // --- compute_delta tests ---

    #[test]
    fn test_compute_delta_identical_states_empty_patches() {
        let base = vec![1u8, 2, 3, 4, 5];
        let delta = compute_delta(1, 10, &base, &base, 0, 0).expect("compute_delta failed");
        assert_eq!(delta.patch_count(), 0);
    }

    #[test]
    fn test_compute_delta_single_byte_change() {
        let base = vec![0u8; 8];
        let mut current = base.clone();
        current[3] = 42;
        let delta = compute_delta(1, 1, &base, &current, 0, 0).expect("compute_delta failed");
        assert_eq!(delta.patch_count(), 1);
        assert_eq!(delta.patches[0], (3, 42));
    }

    #[test]
    fn test_compute_delta_all_bytes_changed() {
        let base = vec![0u8; 4];
        let current = vec![9u8; 4];
        let delta = compute_delta(1, 1, &base, &current, 0, 0).expect("compute_delta failed");
        assert_eq!(delta.patch_count(), 4);
    }

    #[test]
    fn test_compute_delta_length_mismatch_returns_err() {
        let base = vec![0u8; 4];
        let current = vec![0u8; 5];
        assert!(compute_delta(1, 1, &base, &current, 0, 0).is_err());
    }

    // --- apply_deltas tests ---

    #[test]
    fn test_apply_deltas_no_deltas_returns_base() {
        let base = vec![1u8, 2, 3];
        let result = apply_deltas(&base, &[]).expect("apply_deltas failed");
        assert_eq!(result, base);
    }

    #[test]
    fn test_apply_deltas_single_patch_correct() {
        let base = vec![0u8; 5];
        let delta = CheckpointDelta::new(1, 1, vec![(2, 99)], 0, 0);
        let result = apply_deltas(&base, &[delta]).expect("apply_deltas failed");
        assert_eq!(result[2], 99);
        assert_eq!(&result[..2], &[0, 0]);
        assert_eq!(&result[3..], &[0, 0]);
    }

    #[test]
    fn test_apply_deltas_out_of_bounds_returns_err() {
        let base = vec![0u8; 3];
        let delta = CheckpointDelta::new(1, 1, vec![(10, 5)], 0, 0);
        assert!(apply_deltas(&base, &[delta]).is_err());
    }

    #[test]
    fn test_apply_deltas_multiple_sequential() {
        let base = vec![0u8; 4];
        let d1 = CheckpointDelta::new(1, 1, vec![(0, 10)], 0, 0);
        let d2 = CheckpointDelta::new(1, 2, vec![(1, 20)], 1, 1);
        let d3 = CheckpointDelta::new(1, 3, vec![(0, 30), (3, 40)], 2, 2);
        let result = apply_deltas(&base, &[d1, d2, d3]).expect("apply_deltas failed");
        assert_eq!(result, vec![30, 20, 0, 40]);
    }

    // --- IncrementalCheckpointStore tests ---

    #[test]
    fn test_incremental_store_register_and_reconstruct() {
        let config = IncrementalCheckpointConfig::default_config();
        let mut store = IncrementalCheckpointStore::new(config);
        let initial = vec![0u8, 1, 2, 3];
        store.register_task(1, initial.clone());
        let state = store
            .reconstruct_latest(1)
            .expect("reconstruct_latest failed");
        assert_eq!(state, initial);
    }

    #[test]
    fn test_incremental_store_record_unchanged_skipped_by_default() {
        let config = IncrementalCheckpointConfig::default_config();
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 4]);
        let outcome = store.record(1, &[0u8; 4], 1, 100).expect("record failed");
        assert_eq!(outcome, RecordOutcome::Unchanged);
        assert_eq!(store.delta_count_for(1), Some(0));
    }

    #[test]
    fn test_incremental_store_record_delta_written() {
        let config = IncrementalCheckpointConfig::default_config();
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 4]);
        let mut new_state = vec![0u8; 4];
        new_state[1] = 7;
        let outcome = store.record(1, &new_state, 1, 100).expect("record failed");
        assert_eq!(outcome, RecordOutcome::DeltaWritten);
        assert_eq!(store.delta_count_for(1), Some(1));
    }

    #[test]
    fn test_incremental_store_reconstruct_after_deltas() {
        let config = IncrementalCheckpointConfig::default_config();
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 4]);

        let s1 = vec![1u8, 0, 0, 0];
        store.record(1, &s1, 1, 10).expect("record failed");
        let s2 = vec![1u8, 2, 0, 0];
        store.record(1, &s2, 2, 20).expect("record failed");
        let s3 = vec![1u8, 2, 3, 0];
        store.record(1, &s3, 3, 30).expect("record failed");

        let recovered = store
            .reconstruct_latest(1)
            .expect("reconstruct_latest failed");
        assert_eq!(recovered, s3);
    }

    #[test]
    fn test_incremental_store_rollover_resets_chain() {
        let config = IncrementalCheckpointConfig {
            delta_chain_limit: 3,
            record_unchanged: false,
        };
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 4]);

        // Record 3 distinct changes — each should be DeltaWritten.
        for i in 0..3u8 {
            let mut s = vec![0u8; 4];
            s[0] = i + 1;
            store.record(1, &s, i as u64, 0).expect("record failed");
        }
        assert_eq!(store.delta_count_for(1), Some(3));

        // One more triggers roll-over.
        let s = vec![99u8; 4];
        let outcome = store.record(1, &s, 3, 0).expect("record failed");
        assert_eq!(outcome, RecordOutcome::BaseRolledOver);
        // After roll-over, delta chain is cleared.
        assert_eq!(store.delta_count_for(1), Some(0));
        // And the reconstructed state is correct.
        let recovered = store
            .reconstruct_latest(1)
            .expect("reconstruct_latest failed");
        assert_eq!(recovered, s);
    }

    #[test]
    fn test_incremental_store_unregistered_task_returns_err() {
        let store = IncrementalCheckpointStore::new(IncrementalCheckpointConfig::default());
        assert!(store.reconstruct_latest(99).is_err());
    }

    #[test]
    fn test_incremental_store_deregister_removes_task() {
        let mut store = IncrementalCheckpointStore::new(IncrementalCheckpointConfig::default());
        store.register_task(1, vec![0u8; 4]);
        assert_eq!(store.task_count(), 1);
        store.deregister_task(1);
        assert_eq!(store.task_count(), 0);
        assert!(store.reconstruct_latest(1).is_err());
    }

    #[test]
    fn test_incremental_store_flush_stats_counts_bases_and_deltas() {
        let config = IncrementalCheckpointConfig::default_config();
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 8]);
        // Write 2 deltas.
        store
            .record(1, &[1u8, 0, 0, 0, 0, 0, 0, 0], 1, 0)
            .expect("record failed");
        store
            .record(1, &[1u8, 2, 0, 0, 0, 0, 0, 0], 2, 0)
            .expect("record failed");
        let stats = store.flush_stats();
        // initial base = 1 base; 2 deltas appended
        assert_eq!(stats.base_count, 1);
        assert_eq!(stats.delta_count, 2);
    }

    #[test]
    fn test_checkpoint_delta_io_saving() {
        let delta = CheckpointDelta::new(1, 1, vec![(0, 1), (5, 2)], 0, 0);
        // 2 patches * 9 bytes = 18 bytes; full state of 100 bytes → saving = 82
        assert_eq!(delta.io_saving_bytes(100), 82);
    }

    #[test]
    fn test_incremental_store_record_unchanged_recorded_when_flag_set() {
        let config = IncrementalCheckpointConfig {
            delta_chain_limit: 50,
            record_unchanged: true,
        };
        let mut store = IncrementalCheckpointStore::new(config);
        store.register_task(1, vec![0u8; 4]);
        let outcome = store.record(1, &[0u8; 4], 1, 0).expect("record failed");
        // Should still write a delta (empty patches) because record_unchanged is true.
        assert_eq!(outcome, RecordOutcome::DeltaWritten);
    }
}
