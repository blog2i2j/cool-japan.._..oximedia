//! Job checkpointing: save and restore execution state for resume-on-failure.
//!
//! Checkpoints are serialized as JSON and stored in-memory (with optional
//! path tracking).  They allow a job to resume from where it left off after a
//! transient failure.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};
use crate::types::JobId;

/// The execution progress saved at a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointData {
    /// Job identifier.
    pub job_id: String,
    /// Human-readable checkpoint label.
    pub label: String,
    /// Zero-based step index at which the checkpoint was taken.
    pub step: usize,
    /// Total number of steps expected.
    pub total_steps: usize,
    /// Arbitrary key-value metadata (e.g. last file processed).
    pub metadata: HashMap<String, String>,
    /// Wall-clock timestamp (seconds since Unix epoch).
    pub timestamp_secs: u64,
    /// Number of times this job has been retried.
    pub retry_count: u32,
}

impl CheckpointData {
    /// Create a new checkpoint at a given step.
    #[must_use]
    pub fn new(job_id: &JobId, label: impl Into<String>, step: usize, total_steps: usize) -> Self {
        Self {
            job_id: job_id.as_str().to_string(),
            label: label.into(),
            step,
            total_steps,
            metadata: HashMap::new(),
            timestamp_secs: current_timestamp(),
            retry_count: 0,
        }
    }

    /// Add a metadata entry and return `self` for builder-style chaining.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Completion percentage (0–100).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_pct(&self) -> f64 {
        if self.total_steps == 0 {
            return 0.0;
        }
        (self.step as f64 / self.total_steps as f64) * 100.0
    }

    /// Whether the checkpoint represents a completed job.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.total_steps > 0 && self.step >= self.total_steps
    }
}

/// Serialize a checkpoint to JSON bytes.
///
/// # Errors
///
/// Returns [`BatchError::SerializationError`] if serialization fails.
pub fn serialize_checkpoint(cp: &CheckpointData) -> Result<Vec<u8>> {
    serde_json::to_vec(cp).map_err(BatchError::SerializationError)
}

/// Deserialize a checkpoint from JSON bytes.
///
/// # Errors
///
/// Returns [`BatchError::SerializationError`] if deserialization fails.
pub fn deserialize_checkpoint(data: &[u8]) -> Result<CheckpointData> {
    serde_json::from_slice(data).map_err(BatchError::SerializationError)
}

/// In-memory checkpoint store shared across threads.
#[derive(Debug, Default)]
pub struct CheckpointStore {
    checkpoints: RwLock<HashMap<String, CheckpointData>>,
}

impl CheckpointStore {
    /// Create a new, empty checkpoint store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Save a checkpoint for a job, overwriting any existing one.
    pub fn save(&self, cp: CheckpointData) {
        self.checkpoints.write().insert(cp.job_id.clone(), cp);
    }

    /// Load the checkpoint for a job.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if no checkpoint exists.
    pub fn load(&self, job_id: &JobId) -> Result<CheckpointData> {
        self.checkpoints
            .read()
            .get(job_id.as_str())
            .cloned()
            .ok_or_else(|| {
                BatchError::JobNotFound(format!("No checkpoint for job: {}", job_id.as_str()))
            })
    }

    /// Delete the checkpoint for a job.  Returns `true` if it existed.
    pub fn delete(&self, job_id: &JobId) -> bool {
        self.checkpoints.write().remove(job_id.as_str()).is_some()
    }

    /// Return `true` if a checkpoint exists for the given job.
    #[must_use]
    pub fn exists(&self, job_id: &JobId) -> bool {
        self.checkpoints.read().contains_key(job_id.as_str())
    }

    /// Return the total number of stored checkpoints.
    #[must_use]
    pub fn count(&self) -> usize {
        self.checkpoints.read().len()
    }

    /// List all job IDs that have checkpoints.
    #[must_use]
    pub fn list_ids(&self) -> Vec<String> {
        self.checkpoints.read().keys().cloned().collect()
    }

    /// Clear all checkpoints.
    pub fn clear(&self) {
        self.checkpoints.write().clear();
    }
}

/// Manages checkpointing for a specific job.
pub struct JobCheckpointer {
    job_id: JobId,
    store: Arc<CheckpointStore>,
    total_steps: usize,
}

impl JobCheckpointer {
    /// Create a new checkpointer.
    #[must_use]
    pub fn new(job_id: JobId, store: Arc<CheckpointStore>, total_steps: usize) -> Self {
        Self {
            job_id,
            store,
            total_steps,
        }
    }

    /// Save a checkpoint at the given step.
    pub fn checkpoint(&self, step: usize, label: impl Into<String>) {
        let cp = CheckpointData::new(&self.job_id, label, step, self.total_steps);
        self.store.save(cp);
    }

    /// Save a checkpoint with extra metadata.
    pub fn checkpoint_with_meta(
        &self,
        step: usize,
        label: impl Into<String>,
        meta: HashMap<String, String>,
    ) {
        let mut cp = CheckpointData::new(&self.job_id, label, step, self.total_steps);
        cp.metadata = meta;
        self.store.save(cp);
    }

    /// Resume: return the step to start from (0 if no checkpoint).
    #[must_use]
    pub fn resume_from(&self) -> usize {
        self.store.load(&self.job_id).map(|cp| cp.step).unwrap_or(0)
    }

    /// Delete the checkpoint (call after successful completion).
    pub fn complete(&self) {
        self.store.delete(&self.job_id);
    }
}

/// Policy for how checkpoints are retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetentionPolicy {
    /// Keep only the most recent checkpoint per job.
    #[default]
    Latest,
    /// Keep all checkpoints (useful for audit).
    All,
    /// Delete checkpoint after job completes successfully.
    DeleteOnSuccess,
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jid(s: &str) -> JobId {
        JobId::from(s)
    }

    #[test]
    fn test_checkpoint_data_new() {
        let cp = CheckpointData::new(&jid("job-1"), "step1", 3, 10);
        assert_eq!(cp.job_id, "job-1");
        assert_eq!(cp.step, 3);
        assert_eq!(cp.total_steps, 10);
        assert_eq!(cp.label, "step1");
    }

    #[test]
    fn test_checkpoint_progress_pct() {
        let cp = CheckpointData::new(&jid("j"), "lbl", 5, 10);
        assert!((cp.progress_pct() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_is_complete() {
        let cp = CheckpointData::new(&jid("j"), "done", 10, 10);
        assert!(cp.is_complete());
        let cp2 = CheckpointData::new(&jid("j"), "partial", 5, 10);
        assert!(!cp2.is_complete());
    }

    #[test]
    fn test_checkpoint_with_meta() {
        let cp = CheckpointData::new(&jid("j"), "lbl", 1, 5).with_meta("last_file", "/tmp/foo.mp4");
        assert_eq!(
            cp.metadata.get("last_file").expect("failed to get value"),
            "/tmp/foo.mp4"
        );
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let cp = CheckpointData::new(&jid("j"), "lbl", 2, 8);
        let bytes = serialize_checkpoint(&cp).expect("operation should succeed");
        let cp2 = deserialize_checkpoint(&bytes).expect("operation should succeed");
        assert_eq!(cp2.job_id, cp.job_id);
        assert_eq!(cp2.step, cp.step);
    }

    #[test]
    fn test_store_save_and_load() {
        let store = CheckpointStore::new();
        let cp = CheckpointData::new(&jid("abc"), "lbl", 1, 5);
        store.save(cp);
        let loaded = store.load(&jid("abc")).expect("failed to load");
        assert_eq!(loaded.step, 1);
    }

    #[test]
    fn test_store_load_missing_returns_error() {
        let store = CheckpointStore::new();
        assert!(store.load(&jid("nope")).is_err());
    }

    #[test]
    fn test_store_delete() {
        let store = CheckpointStore::new();
        store.save(CheckpointData::new(&jid("x"), "l", 1, 2));
        assert!(store.delete(&jid("x")));
        assert!(!store.exists(&jid("x")));
    }

    #[test]
    fn test_store_count_and_clear() {
        let store = CheckpointStore::new();
        store.save(CheckpointData::new(&jid("a"), "l", 0, 1));
        store.save(CheckpointData::new(&jid("b"), "l", 0, 1));
        assert_eq!(store.count(), 2);
        store.clear();
        assert_eq!(store.count(), 0);
    }

    #[test]
    fn test_store_list_ids() {
        let store = CheckpointStore::new();
        store.save(CheckpointData::new(&jid("a"), "l", 0, 1));
        store.save(CheckpointData::new(&jid("b"), "l", 0, 1));
        let ids = store.list_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"a".to_string()));
    }

    #[test]
    fn test_job_checkpointer_checkpoint_and_resume() {
        let store = Arc::new(CheckpointStore::new());
        let c = JobCheckpointer::new(jid("job-a"), Arc::clone(&store), 10);
        assert_eq!(c.resume_from(), 0); // no checkpoint
        c.checkpoint(4, "mid");
        assert_eq!(c.resume_from(), 4);
    }

    #[test]
    fn test_job_checkpointer_complete_removes_checkpoint() {
        let store = Arc::new(CheckpointStore::new());
        let c = JobCheckpointer::new(jid("job-b"), Arc::clone(&store), 5);
        c.checkpoint(5, "done");
        assert!(store.exists(&jid("job-b")));
        c.complete();
        assert!(!store.exists(&jid("job-b")));
    }

    #[test]
    fn test_retention_policy_default() {
        let p = RetentionPolicy::default();
        assert_eq!(p, RetentionPolicy::Latest);
    }

    #[test]
    fn test_checkpoint_timestamp_nonzero() {
        let cp = CheckpointData::new(&jid("t"), "l", 0, 1);
        assert!(cp.timestamp_secs > 0);
    }
}
