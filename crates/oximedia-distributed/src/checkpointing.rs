//! Distributed checkpoint management.
//!
//! Provides checkpoint creation, storage, retrieval, and recovery planning
//! for fault-tolerant distributed encoding jobs.

#![allow(dead_code)]

use crate::consensus::NodeId;
use std::collections::HashMap;

/// Unique identifier for a checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CheckpointId {
    /// The node that created this checkpoint.
    pub node_id: NodeId,
    /// Monotonically increasing sequence number within the node.
    pub sequence: u64,
}

impl CheckpointId {
    /// Create a new checkpoint ID.
    #[must_use]
    pub fn new(node_id: NodeId, sequence: u64) -> Self {
        Self { node_id, sequence }
    }

    /// Format as a string key: "node-{id}-seq-{seq}".
    #[must_use]
    pub fn to_string(&self) -> String {
        format!("node-{}-seq-{}", self.node_id.inner(), self.sequence)
    }
}

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// A single checkpoint record.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// The checkpoint's unique identifier.
    pub id: CheckpointId,
    /// The job this checkpoint belongs to.
    pub job_id: String,
    /// Size of the serialized state in bytes.
    pub state_size_bytes: u64,
    /// Unix epoch milliseconds when this checkpoint was created.
    pub created_at_ms: u64,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new(
        id: CheckpointId,
        job_id: impl Into<String>,
        state_size_bytes: u64,
        created_at_ms: u64,
    ) -> Self {
        Self {
            id,
            job_id: job_id.into(),
            state_size_bytes,
            created_at_ms,
            metadata: HashMap::new(),
        }
    }

    /// Create a checkpoint with metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// In-memory store of checkpoints, keyed by `CheckpointId`.
#[derive(Debug, Default)]
pub struct CheckpointStore {
    checkpoints: HashMap<String, Checkpoint>,
}

impl CheckpointStore {
    /// Create a new empty checkpoint store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
        }
    }

    /// Save a checkpoint.
    pub fn save(&mut self, checkpoint: Checkpoint) {
        let key = checkpoint.id.to_string();
        self.checkpoints.insert(key, checkpoint);
    }

    /// Load a checkpoint by ID, returning a reference if found.
    #[must_use]
    pub fn load(&self, id: &CheckpointId) -> Option<&Checkpoint> {
        self.checkpoints.get(&id.to_string())
    }

    /// Return the most recently created checkpoint for a job (by `created_at_ms`).
    #[must_use]
    pub fn latest_for_job(&self, job_id: &str) -> Option<&Checkpoint> {
        self.checkpoints
            .values()
            .filter(|c| c.job_id == job_id)
            .max_by_key(|c| c.created_at_ms)
    }

    /// Purge old checkpoints for each job, keeping only the `keep_last_n` most recent.
    pub fn purge_old(&mut self, keep_last_n: usize) {
        // Collect job IDs
        let job_ids: Vec<String> = self
            .checkpoints
            .values()
            .map(|c| c.job_id.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        for job_id in job_ids {
            // Gather keys and timestamps for this job
            let mut job_checkpoints: Vec<(String, u64)> = self
                .checkpoints
                .iter()
                .filter(|(_, c)| c.job_id == job_id)
                .map(|(k, c)| (k.clone(), c.created_at_ms))
                .collect();

            if job_checkpoints.len() <= keep_last_n {
                continue;
            }

            // Sort by timestamp descending; keep first keep_last_n, remove rest
            job_checkpoints.sort_by(|a, b| b.1.cmp(&a.1));
            for (key, _) in job_checkpoints.into_iter().skip(keep_last_n) {
                self.checkpoints.remove(&key);
            }
        }
    }

    /// Return the total number of stored checkpoints.
    #[must_use]
    pub fn len(&self) -> usize {
        self.checkpoints.len()
    }

    /// Return true if the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.checkpoints.is_empty()
    }
}

/// Policy governing checkpoint behaviour.
#[derive(Debug, Clone)]
pub struct CheckpointPolicy {
    /// How often to create checkpoints (seconds).
    pub frequency_secs: u32,
    /// Maximum number of checkpoints to retain per job.
    pub keep_last_n: usize,
    /// Whether to compress checkpoint state.
    pub compress: bool,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            frequency_secs: 300,
            keep_last_n: 5,
            compress: true,
        }
    }
}

/// Finds the best checkpoint to use when recovering a failed job.
pub struct RecoveryPlanner;

impl RecoveryPlanner {
    /// Find the most suitable recovery point for a failed job.
    ///
    /// Returns the latest checkpoint for the given job, if any exists.
    #[must_use]
    pub fn find_recovery_point<'a>(
        store: &'a CheckpointStore,
        failed_job_id: &str,
    ) -> Option<&'a Checkpoint> {
        store.latest_for_job(failed_job_id)
    }
}

/// Estimate of the cost to recover from a given checkpoint.
#[derive(Debug, Clone)]
pub struct RecoveryEstimate {
    /// The checkpoint to recover from.
    pub checkpoint: CheckpointId,
    /// Number of frames that need to be re-processed.
    pub reprocess_frames: u64,
    /// Estimated time in seconds to complete recovery.
    pub estimated_recovery_secs: f64,
}

impl RecoveryEstimate {
    /// Create a recovery estimate.
    #[must_use]
    pub fn new(
        checkpoint: CheckpointId,
        reprocess_frames: u64,
        estimated_recovery_secs: f64,
    ) -> Self {
        Self {
            checkpoint,
            reprocess_frames,
            estimated_recovery_secs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id(node: u64, seq: u64) -> CheckpointId {
        CheckpointId::new(NodeId::new(node), seq)
    }

    fn make_checkpoint(node: u64, seq: u64, job: &str, ts: u64) -> Checkpoint {
        Checkpoint::new(make_id(node, seq), job, 1024, ts)
    }

    #[test]
    fn test_checkpoint_id_to_string() {
        let id = make_id(3, 7);
        assert_eq!(id.to_string(), "node-3-seq-7");
    }

    #[test]
    fn test_checkpoint_id_display() {
        let id = make_id(1, 2);
        assert_eq!(format!("{id}"), "node-1-seq-2");
    }

    #[test]
    fn test_store_save_and_load() {
        let mut store = CheckpointStore::new();
        let cp = make_checkpoint(1, 1, "job-a", 1000);
        let id = cp.id.clone();
        store.save(cp);

        let loaded = store.load(&id);
        assert!(loaded.is_some());
        assert_eq!(loaded.expect("loading should succeed").job_id, "job-a");
    }

    #[test]
    fn test_store_load_missing() {
        let store = CheckpointStore::new();
        assert!(store.load(&make_id(99, 99)).is_none());
    }

    #[test]
    fn test_latest_for_job() {
        let mut store = CheckpointStore::new();
        store.save(make_checkpoint(1, 1, "job-a", 1000));
        store.save(make_checkpoint(1, 2, "job-a", 3000));
        store.save(make_checkpoint(1, 3, "job-a", 2000));

        let latest = store.latest_for_job("job-a");
        assert!(latest.is_some());
        assert_eq!(latest.expect("latest should exist").id.sequence, 2); // ts=3000 is latest
    }

    #[test]
    fn test_latest_for_job_missing() {
        let store = CheckpointStore::new();
        assert!(store.latest_for_job("no-such-job").is_none());
    }

    #[test]
    fn test_purge_old_keeps_n() {
        let mut store = CheckpointStore::new();
        for i in 1..=5 {
            store.save(make_checkpoint(1, i, "job-b", i as u64 * 1000));
        }
        assert_eq!(store.len(), 5);
        store.purge_old(3);
        assert_eq!(store.len(), 3);
    }

    #[test]
    fn test_purge_old_no_op_when_few() {
        let mut store = CheckpointStore::new();
        store.save(make_checkpoint(1, 1, "job-c", 1000));
        store.save(make_checkpoint(1, 2, "job-c", 2000));
        store.purge_old(5);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn test_recovery_planner_finds_latest() {
        let mut store = CheckpointStore::new();
        store.save(make_checkpoint(1, 1, "job-fail", 5000));
        store.save(make_checkpoint(1, 2, "job-fail", 9000));

        let cp = RecoveryPlanner::find_recovery_point(&store, "job-fail");
        assert!(cp.is_some());
        assert_eq!(cp.expect("checkpoint should exist").created_at_ms, 9000);
    }

    #[test]
    fn test_recovery_planner_no_checkpoint() {
        let store = CheckpointStore::new();
        let cp = RecoveryPlanner::find_recovery_point(&store, "missing-job");
        assert!(cp.is_none());
    }

    #[test]
    fn test_recovery_estimate() {
        let est = RecoveryEstimate::new(make_id(1, 5), 1200, 45.5);
        assert_eq!(est.reprocess_frames, 1200);
        assert!((est.estimated_recovery_secs - 45.5).abs() < 1e-9);
    }

    #[test]
    fn test_checkpoint_with_metadata() {
        let cp = Checkpoint::new(make_id(2, 1), "job-meta", 2048, 500)
            .with_metadata("encoder", "av1")
            .with_metadata("pass", "2");
        assert_eq!(cp.metadata.get("encoder").map(String::as_str), Some("av1"));
        assert_eq!(cp.metadata.get("pass").map(String::as_str), Some("2"));
    }
}
