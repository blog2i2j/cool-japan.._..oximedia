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

// ---------------------------------------------------------------------------
// JobCheckpoint — lightweight struct for raw state-byte checkpoints
// ---------------------------------------------------------------------------

/// A raw-state checkpoint, distinct from [`CheckpointData`].
///
/// `state_bytes` carries arbitrary serialized job state that the caller
/// controls, allowing resume-on-failure for any kind of workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobCheckpoint {
    /// Unique job identifier.
    pub job_id: String,
    /// Step index at which the checkpoint was taken.
    pub step: u32,
    /// Opaque serialized job state (caller-controlled encoding).
    pub state_bytes: Vec<u8>,
    /// Creation time as Unix epoch seconds.
    pub created_at: u64,
}

impl JobCheckpoint {
    /// Create a new checkpoint with the current timestamp.
    #[must_use]
    pub fn new(job_id: impl Into<String>, step: u32, state_bytes: Vec<u8>) -> Self {
        Self {
            job_id: job_id.into(),
            step,
            state_bytes,
            created_at: current_timestamp(),
        }
    }
}

// ---------------------------------------------------------------------------
// PersistentCheckpointStore — file-backed store using temp_dir()
// ---------------------------------------------------------------------------

/// A checkpoint store that persists [`JobCheckpoint`] entries as JSON files
/// under `std::env::temp_dir()`.
///
/// File naming: `oximedia_checkpoint_<job_id>.json`
/// (Non-ASCII / path-unsafe characters in `job_id` are replaced with `_`.)
#[derive(Debug, Clone)]
pub struct PersistentCheckpointStore {
    /// Base directory for checkpoint files.
    base_dir: std::path::PathBuf,
}

impl PersistentCheckpointStore {
    /// Create a store rooted at `std::env::temp_dir()`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_dir: std::env::temp_dir(),
        }
    }

    /// Create a store rooted at a custom directory (useful for tests).
    #[must_use]
    pub fn with_dir(dir: std::path::PathBuf) -> Self {
        Self { base_dir: dir }
    }

    fn checkpoint_path(&self, job_id: &str) -> std::path::PathBuf {
        // Replace characters that are unsafe in file names with underscores.
        let safe_id: String = job_id
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.base_dir
            .join(format!("oximedia_checkpoint_{safe_id}.json"))
    }

    /// Persist a checkpoint to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or the file write fails.
    pub fn save(&self, checkpoint: &JobCheckpoint) -> Result<()> {
        let path = self.checkpoint_path(&checkpoint.job_id);
        let json = serde_json::to_vec(checkpoint).map_err(BatchError::SerializationError)?;
        std::fs::write(&path, &json).map_err(BatchError::IoError)?;
        Ok(())
    }

    /// Load a checkpoint from disk.
    ///
    /// Returns `Ok(None)` if no checkpoint file exists for the given job.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or parsed.
    pub fn load(&self, job_id: &str) -> Result<Option<JobCheckpoint>> {
        let path = self.checkpoint_path(job_id);
        match std::fs::read(&path) {
            Ok(bytes) => {
                let cp: JobCheckpoint =
                    serde_json::from_slice(&bytes).map_err(BatchError::SerializationError)?;
                Ok(Some(cp))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(BatchError::IoError(e)),
        }
    }

    /// Delete a checkpoint file.
    ///
    /// Returns `Ok(true)` if the file existed and was deleted, `Ok(false)` if
    /// it did not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be removed.
    pub fn delete(&self, job_id: &str) -> Result<bool> {
        let path = self.checkpoint_path(job_id);
        match std::fs::remove_file(&path) {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(BatchError::IoError(e)),
        }
    }

    /// List all job IDs that have a checkpoint file in the store directory.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn list(&self) -> Result<Vec<String>> {
        let prefix = "oximedia_checkpoint_";
        let suffix = ".json";
        let entries = std::fs::read_dir(&self.base_dir).map_err(BatchError::IoError)?;
        let mut ids = Vec::new();
        for entry in entries {
            let entry = entry.map_err(BatchError::IoError)?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(prefix) && name_str.ends_with(suffix) {
                let id_part = &name_str[prefix.len()..name_str.len() - suffix.len()];
                ids.push(id_part.to_string());
            }
        }
        Ok(ids)
    }

    /// Return `true` if a checkpoint file exists for the given job.
    #[must_use]
    pub fn exists(&self, job_id: &str) -> bool {
        self.checkpoint_path(job_id).exists()
    }
}

impl Default for PersistentCheckpointStore {
    fn default() -> Self {
        Self::new()
    }
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
        let tmp_file = std::env::temp_dir()
            .join("oximedia-batch-checkpoint-foo.mp4")
            .to_string_lossy()
            .into_owned();
        let cp =
            CheckpointData::new(&jid("j"), "lbl", 1, 5).with_meta("last_file", tmp_file.as_str());
        assert_eq!(
            cp.metadata.get("last_file").expect("failed to get value"),
            &tmp_file
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

    // -----------------------------------------------------------------------
    // JobCheckpoint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_job_checkpoint_new() {
        let cp = JobCheckpoint::new("job-123", 5, b"state data".to_vec());
        assert_eq!(cp.job_id, "job-123");
        assert_eq!(cp.step, 5);
        assert_eq!(cp.state_bytes, b"state data");
        assert!(cp.created_at > 0);
    }

    #[test]
    fn test_job_checkpoint_empty_state() {
        let cp = JobCheckpoint::new("empty-state", 0, Vec::new());
        assert!(cp.state_bytes.is_empty());
    }

    // -----------------------------------------------------------------------
    // PersistentCheckpointStore tests
    // -----------------------------------------------------------------------

    fn temp_store(suffix: &str) -> PersistentCheckpointStore {
        let dir = std::env::temp_dir().join(format!("oximedia_cpstore_{suffix}"));
        std::fs::create_dir_all(&dir).expect("create test dir");
        PersistentCheckpointStore::with_dir(dir)
    }

    #[test]
    fn test_persistent_store_save_and_load() {
        let store = temp_store("save_load");
        let cp = JobCheckpoint::new("job-save", 3, b"saved state".to_vec());
        store.save(&cp).expect("save failed");

        let loaded = store.load("job-save").expect("load failed");
        assert!(loaded.is_some());
        let loaded = loaded.expect("expected Some");
        assert_eq!(loaded.job_id, "job-save");
        assert_eq!(loaded.step, 3);
        assert_eq!(loaded.state_bytes, b"saved state");
        store.delete("job-save").ok();
    }

    #[test]
    fn test_persistent_store_load_missing_returns_none() {
        let store = temp_store("load_none");
        let result = store.load("no_such_job").expect("load failed unexpectedly");
        assert!(result.is_none());
    }

    #[test]
    fn test_persistent_store_delete_existing() {
        let store = temp_store("delete_exist");
        let cp = JobCheckpoint::new("del-job", 1, vec![1, 2, 3]);
        store.save(&cp).expect("save failed");
        assert!(store.exists("del-job"));
        let deleted = store.delete("del-job").expect("delete failed");
        assert!(deleted);
        assert!(!store.exists("del-job"));
    }

    #[test]
    fn test_persistent_store_delete_nonexistent_returns_false() {
        let store = temp_store("delete_none");
        let deleted = store.delete("phantom-job").expect("delete error");
        assert!(!deleted);
    }

    #[test]
    fn test_persistent_store_list_jobs() {
        let store = temp_store("list_jobs");
        store
            .save(&JobCheckpoint::new("list-a", 1, vec![]))
            .expect("save a");
        store
            .save(&JobCheckpoint::new("list-b", 2, vec![]))
            .expect("save b");

        let ids = store.list().expect("list failed");
        assert!(ids.contains(&"list-a".to_string()));
        assert!(ids.contains(&"list-b".to_string()));

        store.delete("list-a").ok();
        store.delete("list-b").ok();
    }

    #[test]
    fn test_persistent_store_overwrite() {
        let store = temp_store("overwrite");
        let cp1 = JobCheckpoint::new("ow-job", 1, b"v1".to_vec());
        store.save(&cp1).expect("save v1");
        let cp2 = JobCheckpoint::new("ow-job", 2, b"v2".to_vec());
        store.save(&cp2).expect("save v2");

        let loaded = store.load("ow-job").expect("load").expect("Some");
        assert_eq!(loaded.step, 2);
        assert_eq!(loaded.state_bytes, b"v2");
        store.delete("ow-job").ok();
    }

    #[test]
    fn test_persistent_store_exists() {
        let store = temp_store("exists_check");
        assert!(!store.exists("ex-job"));
        store
            .save(&JobCheckpoint::new("ex-job", 0, vec![]))
            .expect("save");
        assert!(store.exists("ex-job"));
        store.delete("ex-job").ok();
    }

    #[test]
    fn test_persistent_store_special_chars_in_job_id() {
        let store = temp_store("special_chars");
        // Slashes and spaces should be sanitised to underscores.
        let cp = JobCheckpoint::new("job/with spaces:and!symbols", 0, b"data".to_vec());
        store.save(&cp).expect("save");
        let loaded = store
            .load("job/with spaces:and!symbols")
            .expect("load")
            .expect("Some");
        assert_eq!(loaded.state_bytes, b"data");
        store.delete("job/with spaces:and!symbols").ok();
    }

    #[test]
    fn test_persistent_store_large_state_bytes() {
        let store = temp_store("large_state");
        let large: Vec<u8> = (0u8..=255).cycle().take(65536).collect();
        let cp = JobCheckpoint::new("large-job", 99, large.clone());
        store.save(&cp).expect("save large");
        let loaded = store.load("large-job").expect("load").expect("Some");
        assert_eq!(loaded.state_bytes.len(), 65536);
        assert_eq!(loaded.state_bytes, large);
        store.delete("large-job").ok();
    }

    #[test]
    fn test_persistent_store_default_constructor() {
        let store = PersistentCheckpointStore::default();
        // Should use temp_dir without panicking.
        assert!(store.base_dir.exists() || !store.base_dir.exists()); // always true, just ensure no panic
    }
}
