//! Periodic state checkpointing for crash recovery.
//!
//! [`CheckpointManager`] takes periodic snapshots of queue state, in-progress
//! jobs, and worker assignments.  Snapshots are stored as JSON files with
//! rotation (keep last N) and corrupted-checkpoint detection via SHA-256
//! integrity hashes.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{BatchError, Result};

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

/// A serialisable snapshot of engine state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Unix-epoch timestamp (seconds) when the checkpoint was created.
    pub timestamp_secs: u64,
    /// Queued job IDs (ordered by priority).
    pub queued_jobs: Vec<String>,
    /// Currently in-progress job IDs.
    pub in_progress_jobs: Vec<String>,
    /// Mapping from worker name/ID to the job it is currently executing.
    pub worker_assignments: HashMap<String, String>,
    /// Arbitrary metadata (e.g. engine version, node name).
    pub metadata: HashMap<String, String>,
}

impl Checkpoint {
    /// Create a new checkpoint with the given sequence number and current
    /// timestamp.
    #[must_use]
    pub fn new(sequence: u64) -> Self {
        Self {
            sequence,
            timestamp_secs: current_timestamp(),
            queued_jobs: Vec::new(),
            in_progress_jobs: Vec::new(),
            worker_assignments: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Builder: set queued jobs.
    #[must_use]
    pub fn with_queued(mut self, jobs: Vec<String>) -> Self {
        self.queued_jobs = jobs;
        self
    }

    /// Builder: set in-progress jobs.
    #[must_use]
    pub fn with_in_progress(mut self, jobs: Vec<String>) -> Self {
        self.in_progress_jobs = jobs;
        self
    }

    /// Builder: set worker assignments.
    #[must_use]
    pub fn with_workers(mut self, assignments: HashMap<String, String>) -> Self {
        self.worker_assignments = assignments;
        self
    }

    /// Builder: add metadata.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Total number of jobs referenced (queued + in-progress).
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.queued_jobs.len() + self.in_progress_jobs.len()
    }
}

// ---------------------------------------------------------------------------
// CheckpointEnvelope — checkpoint + integrity hash
// ---------------------------------------------------------------------------

/// A checkpoint together with a SHA-256 integrity hash of its serialised
/// payload.  Used for on-disk storage so that corrupted files can be
/// detected.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointEnvelope {
    /// Hex-encoded SHA-256 of the JSON-serialised `Checkpoint`.
    hash: String,
    /// The checkpoint payload (serialised as nested JSON).
    payload: String,
}

impl CheckpointEnvelope {
    fn wrap(checkpoint: &Checkpoint) -> Result<Self> {
        let payload =
            serde_json::to_string(checkpoint).map_err(BatchError::SerializationError)?;
        let hash = sha256_hex(payload.as_bytes());
        Ok(Self { hash, payload })
    }

    fn unwrap_verified(&self) -> Result<Checkpoint> {
        let computed = sha256_hex(self.payload.as_bytes());
        if computed != self.hash {
            return Err(BatchError::ValidationError(format!(
                "Checkpoint integrity check failed: expected {}, got {}",
                self.hash, computed
            )));
        }
        serde_json::from_str(&self.payload).map_err(BatchError::SerializationError)
    }
}

// ---------------------------------------------------------------------------
// CheckpointManager
// ---------------------------------------------------------------------------

/// Manages periodic checkpoint files in a directory.
///
/// Checkpoints are named `checkpoint_<sequence>.json` and wrapped in a
/// [`CheckpointEnvelope`] for integrity verification.
///
/// Rotation: after writing a new checkpoint the manager deletes the oldest
/// files to keep at most `max_retained` checkpoints on disk.
#[derive(Debug)]
pub struct CheckpointManager {
    /// Directory where checkpoint files are stored.
    dir: PathBuf,
    /// Maximum number of checkpoint files to retain.
    max_retained: usize,
    /// Next sequence number to assign.
    next_sequence: u64,
}

impl CheckpointManager {
    /// Create a new manager.
    ///
    /// # Arguments
    ///
    /// * `dir` — directory in which to store checkpoint files.
    /// * `max_retained` — maximum number of checkpoints to keep on disk.
    ///   Must be at least 1.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be created.
    pub fn new(dir: impl Into<PathBuf>, max_retained: usize) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir).map_err(BatchError::IoError)?;
        let max_retained = max_retained.max(1);

        // Scan existing checkpoints to determine starting sequence number.
        let existing = list_checkpoint_files(&dir)?;
        let next_sequence = existing
            .iter()
            .filter_map(|p| sequence_from_path(p))
            .max()
            .map_or(1, |s| s + 1);

        Ok(Self {
            dir,
            max_retained,
            next_sequence,
        })
    }

    /// Save a checkpoint to disk, rotating old files as needed.
    ///
    /// The checkpoint's `sequence` field is overwritten with the manager's
    /// internal counter.
    ///
    /// # Errors
    ///
    /// Returns an error if serialisation or file I/O fails.
    pub fn save_checkpoint(&mut self, mut checkpoint: Checkpoint) -> Result<u64> {
        let seq = self.next_sequence;
        checkpoint.sequence = seq;
        self.next_sequence += 1;

        let envelope = CheckpointEnvelope::wrap(&checkpoint)?;
        let json = serde_json::to_vec_pretty(&envelope).map_err(BatchError::SerializationError)?;
        let path = self.checkpoint_path(seq);
        std::fs::write(&path, &json).map_err(BatchError::IoError)?;

        self.rotate()?;
        Ok(seq)
    }

    /// Load the most recent valid checkpoint.
    ///
    /// Skips corrupted checkpoints (logs a warning via `tracing`).
    ///
    /// Returns `Ok(None)` if no valid checkpoint exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint directory cannot be read.
    pub fn load_latest(&self) -> Result<Option<Checkpoint>> {
        let mut files = list_checkpoint_files(&self.dir)?;
        // Sort descending by sequence so we try the newest first.
        files.sort_by(|a, b| {
            let sa = sequence_from_path(a).unwrap_or(0);
            let sb = sequence_from_path(b).unwrap_or(0);
            sb.cmp(&sa)
        });

        for path in &files {
            match self.load_from_file(path) {
                Ok(cp) => return Ok(Some(cp)),
                Err(e) => {
                    tracing::warn!(
                        path = %path.display(),
                        error = %e,
                        "Skipping corrupted checkpoint"
                    );
                }
            }
        }
        Ok(None)
    }

    /// Load a specific checkpoint by sequence number.
    ///
    /// # Errors
    ///
    /// Returns an error if the file does not exist, cannot be read, or
    /// fails integrity verification.
    pub fn load_by_sequence(&self, sequence: u64) -> Result<Checkpoint> {
        let path = self.checkpoint_path(sequence);
        self.load_from_file(&path)
    }

    /// Restore engine state from the latest checkpoint.
    ///
    /// This is a convenience wrapper around [`load_latest`](Self::load_latest)
    /// that returns a concrete error when no checkpoint is available.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::JobNotFound`] if no valid checkpoint is found,
    /// or an I/O / serialisation error.
    pub fn restore_from_checkpoint(&self) -> Result<Checkpoint> {
        self.load_latest()?.ok_or_else(|| {
            BatchError::JobNotFound("No valid checkpoint found for restoration".to_string())
        })
    }

    /// Number of checkpoint files currently on disk.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn checkpoint_count(&self) -> Result<usize> {
        Ok(list_checkpoint_files(&self.dir)?.len())
    }

    /// List sequence numbers of all checkpoints on disk (ascending order).
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read.
    pub fn list_sequences(&self) -> Result<Vec<u64>> {
        let files = list_checkpoint_files(&self.dir)?;
        let mut seqs: Vec<u64> = files
            .iter()
            .filter_map(|p| sequence_from_path(p))
            .collect();
        seqs.sort_unstable();
        Ok(seqs)
    }

    /// Delete all checkpoint files.
    ///
    /// # Errors
    ///
    /// Returns an error if file deletion fails.
    pub fn clear(&self) -> Result<()> {
        for path in list_checkpoint_files(&self.dir)? {
            std::fs::remove_file(&path).map_err(BatchError::IoError)?;
        }
        Ok(())
    }

    // -- Internal helpers ---------------------------------------------------

    fn checkpoint_path(&self, sequence: u64) -> PathBuf {
        self.dir.join(format!("checkpoint_{sequence}.json"))
    }

    fn load_from_file(&self, path: &Path) -> Result<Checkpoint> {
        let bytes = std::fs::read(path).map_err(BatchError::IoError)?;
        let envelope: CheckpointEnvelope =
            serde_json::from_slice(&bytes).map_err(BatchError::SerializationError)?;
        envelope.unwrap_verified()
    }

    fn rotate(&self) -> Result<()> {
        let mut files = list_checkpoint_files(&self.dir)?;
        if files.len() <= self.max_retained {
            return Ok(());
        }
        // Sort ascending by sequence so oldest are first.
        files.sort_by(|a, b| {
            let sa = sequence_from_path(a).unwrap_or(0);
            let sb = sequence_from_path(b).unwrap_or(0);
            sa.cmp(&sb)
        });
        let to_remove = files.len() - self.max_retained;
        for path in files.iter().take(to_remove) {
            std::fs::remove_file(path).map_err(BatchError::IoError)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

fn list_checkpoint_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).map_err(BatchError::IoError)?;
    let mut files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(BatchError::IoError)?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("checkpoint_") && name_str.ends_with(".json") {
            files.push(entry.path());
        }
    }
    Ok(files)
}

fn sequence_from_path(path: &Path) -> Option<u64> {
    let stem = path.file_stem()?.to_string_lossy();
    let seq_str = stem.strip_prefix("checkpoint_")?;
    seq_str.parse().ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(suffix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("oximedia_cp_test_{suffix}"));
        let _ = std::fs::remove_dir_all(&dir); // clean up from previous runs
        std::fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    #[test]
    fn test_save_and_load_latest() {
        let dir = temp_dir("save_load");
        let mut mgr = CheckpointManager::new(&dir, 5).expect("new mgr");

        let cp = Checkpoint::new(0)
            .with_queued(vec!["q1".into(), "q2".into()])
            .with_in_progress(vec!["ip1".into()]);
        let seq = mgr.save_checkpoint(cp).expect("save");
        assert_eq!(seq, 1);

        let loaded = mgr.load_latest().expect("load").expect("Some");
        assert_eq!(loaded.sequence, 1);
        assert_eq!(loaded.queued_jobs, vec!["q1", "q2"]);
        assert_eq!(loaded.in_progress_jobs, vec!["ip1"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_restore_from_checkpoint() {
        let dir = temp_dir("restore");
        let mut mgr = CheckpointManager::new(&dir, 5).expect("new mgr");
        let cp = Checkpoint::new(0).with_meta("version", "0.1.3");
        mgr.save_checkpoint(cp).expect("save");

        let restored = mgr.restore_from_checkpoint().expect("restore");
        assert_eq!(
            restored.metadata.get("version").map(String::as_str),
            Some("0.1.3")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_restore_empty_returns_error() {
        let dir = temp_dir("restore_empty");
        let mgr = CheckpointManager::new(&dir, 5).expect("new mgr");
        assert!(mgr.restore_from_checkpoint().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rotation_keeps_max_retained() {
        let dir = temp_dir("rotation");
        let mut mgr = CheckpointManager::new(&dir, 3).expect("new mgr");

        for _ in 0..6 {
            mgr.save_checkpoint(Checkpoint::new(0)).expect("save");
        }

        let count = mgr.checkpoint_count().expect("count");
        assert_eq!(count, 3);

        let seqs = mgr.list_sequences().expect("seqs");
        assert_eq!(seqs, vec![4, 5, 6]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_corrupted_checkpoint_detected() {
        let dir = temp_dir("corrupted");
        let mut mgr = CheckpointManager::new(&dir, 5).expect("new mgr");
        mgr.save_checkpoint(Checkpoint::new(0)).expect("save");

        // Corrupt the file by overwriting part of the payload.
        let path = dir.join("checkpoint_1.json");
        let mut content = std::fs::read_to_string(&path).expect("read");
        content = content.replace("\"hash\"", "\"hash_x\"");
        // Actually corrupt the payload hash: write garbage hash
        let corrupted =
            r#"{"hash":"0000000000000000000000000000000000000000000000000000000000000000","payload":"{\"sequence\":1}"}"#;
        std::fs::write(&path, corrupted).expect("write corrupt");

        // load_by_sequence should fail
        assert!(mgr.load_by_sequence(1).is_err());

        // load_latest should skip the corrupted one and return None
        let result = mgr.load_latest().expect("no io error");
        assert!(result.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_worker_assignments() {
        let dir = temp_dir("workers");
        let mut mgr = CheckpointManager::new(&dir, 5).expect("new mgr");

        let mut assignments = HashMap::new();
        assignments.insert("worker-0".to_string(), "job-a".to_string());
        assignments.insert("worker-1".to_string(), "job-b".to_string());

        let cp = Checkpoint::new(0).with_workers(assignments);
        mgr.save_checkpoint(cp).expect("save");

        let loaded = mgr.load_latest().expect("load").expect("Some");
        assert_eq!(loaded.worker_assignments.len(), 2);
        assert_eq!(
            loaded.worker_assignments.get("worker-0").map(String::as_str),
            Some("job-a")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_checkpoint_total_jobs() {
        let cp = Checkpoint::new(1)
            .with_queued(vec!["a".into(), "b".into()])
            .with_in_progress(vec!["c".into()]);
        assert_eq!(cp.total_jobs(), 3);
    }

    #[test]
    fn test_clear_removes_all() {
        let dir = temp_dir("clear");
        let mut mgr = CheckpointManager::new(&dir, 10).expect("new mgr");
        for _ in 0..4 {
            mgr.save_checkpoint(Checkpoint::new(0)).expect("save");
        }
        assert_eq!(mgr.checkpoint_count().expect("count"), 4);
        mgr.clear().expect("clear");
        assert_eq!(mgr.checkpoint_count().expect("count"), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sequence_numbers_monotonic() {
        let dir = temp_dir("monotonic");
        let mut mgr = CheckpointManager::new(&dir, 10).expect("new mgr");
        let s1 = mgr.save_checkpoint(Checkpoint::new(0)).expect("save 1");
        let s2 = mgr.save_checkpoint(Checkpoint::new(0)).expect("save 2");
        let s3 = mgr.save_checkpoint(Checkpoint::new(0)).expect("save 3");
        assert!(s1 < s2);
        assert!(s2 < s3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_max_retained_clamped_to_one() {
        let dir = temp_dir("clamp");
        let mut mgr = CheckpointManager::new(&dir, 0).expect("new mgr");
        // max_retained=0 is clamped to 1
        mgr.save_checkpoint(Checkpoint::new(0)).expect("save 1");
        mgr.save_checkpoint(Checkpoint::new(0)).expect("save 2");
        assert_eq!(mgr.checkpoint_count().expect("count"), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sha256_hex_deterministic() {
        let h1 = sha256_hex(b"hello");
        let h2 = sha256_hex(b"hello");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // 256 bits = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_checkpoint_envelope_roundtrip() {
        let cp = Checkpoint::new(42).with_meta("k", "v");
        let env = CheckpointEnvelope::wrap(&cp).expect("wrap");
        let restored = env.unwrap_verified().expect("unwrap");
        assert_eq!(restored.sequence, 42);
        assert_eq!(
            restored.metadata.get("k").map(String::as_str),
            Some("v")
        );
    }
}
