//! Render artifact lifecycle management.
//!
//! [`RenderArtifact`] tracks output files produced by render jobs, including
//! their storage locations, content checksums, and retention policies.
//! [`ArtifactStore`] manages a collection of artifacts and enforces lifecycle
//! rules (keep-N, expiry by age).

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// RetentionPolicy
// ---------------------------------------------------------------------------

/// Policy controlling how long an artifact is retained.
#[derive(Debug, Clone, PartialEq)]
pub enum RetentionPolicy {
    /// Keep the artifact indefinitely.
    Forever,
    /// Keep for at most `days` calendar days.
    DaysAfterCreation { days: u32 },
    /// Keep only the most recent `n` artifacts per job.
    KeepLatestN { n: usize },
    /// Delete immediately after the render job transitions to Completed.
    DeleteOnCompletion,
}

// ---------------------------------------------------------------------------
// ArtifactStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a [`RenderArtifact`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactStatus {
    /// Being written by the render worker.
    Uploading,
    /// Fully written and available for download.
    Available,
    /// Marked for deletion (pending GC).
    PendingDelete,
    /// Deleted from backing storage.
    Deleted,
}

// ---------------------------------------------------------------------------
// RenderArtifact
// ---------------------------------------------------------------------------

/// A single render output file managed by the farm.
#[derive(Debug, Clone)]
pub struct RenderArtifact {
    /// Globally unique artifact identifier.
    pub id: u64,
    /// Render job this artifact belongs to.
    pub job_id: u64,
    /// Human-readable file name (e.g. `frame_0042.exr`).
    pub filename: String,
    /// Storage URI (e.g. `s3://bucket/path/frame_0042.exr`).
    pub storage_uri: String,
    /// File size in bytes.
    pub byte_size: u64,
    /// Hex-encoded SHA-256 content checksum.
    pub checksum_sha256: String,
    /// Unix timestamp (seconds) when the artifact was created.
    pub created_at: i64,
    /// Retention policy for this artifact.
    pub retention: RetentionPolicy,
    /// Current lifecycle status.
    pub status: ArtifactStatus,
}

impl RenderArtifact {
    /// Creates a new artifact in `Uploading` state.
    #[must_use]
    pub fn new(
        id: u64,
        job_id: u64,
        filename: impl Into<String>,
        storage_uri: impl Into<String>,
        byte_size: u64,
        checksum_sha256: impl Into<String>,
        created_at: i64,
    ) -> Self {
        Self {
            id,
            job_id,
            filename: filename.into(),
            storage_uri: storage_uri.into(),
            byte_size,
            checksum_sha256: checksum_sha256.into(),
            created_at,
            retention: RetentionPolicy::Forever,
            status: ArtifactStatus::Uploading,
        }
    }

    /// Marks the artifact as fully uploaded and available.
    pub fn mark_available(&mut self) {
        self.status = ArtifactStatus::Available;
    }

    /// Returns `true` when the artifact should be expired at `now` (Unix
    /// seconds) according to its retention policy.
    ///
    /// `KeepLatestN` is evaluated by [`ArtifactStore`] rather than individual
    /// artifacts.
    #[must_use]
    pub fn is_expired_at(&self, now: i64) -> bool {
        match &self.retention {
            RetentionPolicy::DaysAfterCreation { days } => {
                let cutoff = self.created_at + *days as i64 * 86_400;
                now >= cutoff
            }
            RetentionPolicy::DeleteOnCompletion => {
                // Artifact-level flag; ArtifactStore handles the actual trigger
                false
            }
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ArtifactStore
// ---------------------------------------------------------------------------

/// Manages render artifacts with lifecycle enforcement.
pub struct ArtifactStore {
    artifacts: HashMap<u64, RenderArtifact>,
    next_id: u64,
}

impl ArtifactStore {
    /// Creates an empty artifact store.
    #[must_use]
    pub fn new() -> Self {
        Self {
            artifacts: HashMap::new(),
            next_id: 1,
        }
    }

    /// Registers a new artifact and returns its assigned ID.
    pub fn register(
        &mut self,
        job_id: u64,
        filename: impl Into<String>,
        storage_uri: impl Into<String>,
        byte_size: u64,
        checksum_sha256: impl Into<String>,
        created_at: i64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let artifact = RenderArtifact::new(
            id,
            job_id,
            filename,
            storage_uri,
            byte_size,
            checksum_sha256,
            created_at,
        );
        self.artifacts.insert(id, artifact);
        id
    }

    /// Marks artifact `id` as available.
    ///
    /// Returns `true` when the artifact was found and updated.
    pub fn mark_available(&mut self, id: u64) -> bool {
        if let Some(a) = self.artifacts.get_mut(&id) {
            a.mark_available();
            true
        } else {
            false
        }
    }

    /// Sets the retention policy for an artifact.
    ///
    /// Returns `true` when the artifact was found.
    pub fn set_retention(&mut self, id: u64, policy: RetentionPolicy) -> bool {
        if let Some(a) = self.artifacts.get_mut(&id) {
            a.retention = policy;
            true
        } else {
            false
        }
    }

    /// Marks artifacts as `PendingDelete` when their retention policy has
    /// expired at `now`.
    ///
    /// For `KeepLatestN` policies: per-job, only the latest `n` artifacts
    /// (by `created_at`) are kept; older ones are marked for deletion.
    pub fn apply_retention(&mut self, now: i64) {
        // 1. Time-based expiry
        for artifact in self.artifacts.values_mut() {
            if artifact.status == ArtifactStatus::Available && artifact.is_expired_at(now) {
                artifact.status = ArtifactStatus::PendingDelete;
            }
        }

        // 2. KeepLatestN: group by job_id, then mark old ones
        let mut keep_n_jobs: HashMap<u64, usize> = HashMap::new();
        for a in self.artifacts.values() {
            if let RetentionPolicy::KeepLatestN { n } = a.retention {
                keep_n_jobs.entry(a.job_id).or_insert(n);
            }
        }

        for (job_id, n) in &keep_n_jobs {
            let mut job_artifacts: Vec<(u64, i64)> = self
                .artifacts
                .values()
                .filter(|a| {
                    a.job_id == *job_id
                        && matches!(a.retention, RetentionPolicy::KeepLatestN { .. })
                        && a.status == ArtifactStatus::Available
                })
                .map(|a| (a.id, a.created_at))
                .collect();

            // Sort newest first
            job_artifacts.sort_by(|a, b| b.1.cmp(&a.1));

            // Mark anything beyond the Nth as pending delete
            for (id, _) in job_artifacts.iter().skip(*n) {
                if let Some(a) = self.artifacts.get_mut(id) {
                    a.status = ArtifactStatus::PendingDelete;
                }
            }
        }
    }

    /// Permanently removes all `PendingDelete` artifacts from the store.
    ///
    /// Returns the number of artifacts removed.
    pub fn purge_pending(&mut self) -> usize {
        let before = self.artifacts.len();
        self.artifacts
            .retain(|_, a| a.status != ArtifactStatus::PendingDelete);
        before - self.artifacts.len()
    }

    /// Returns artifacts belonging to `job_id`.
    #[must_use]
    pub fn artifacts_for_job(&self, job_id: u64) -> Vec<&RenderArtifact> {
        self.artifacts
            .values()
            .filter(|a| a.job_id == job_id)
            .collect()
    }

    /// Total stored bytes across all available artifacts.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.artifacts
            .values()
            .filter(|a| a.status == ArtifactStatus::Available)
            .map(|a| a.byte_size)
            .sum()
    }

    /// Number of artifacts in the store (all statuses).
    #[must_use]
    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    /// Returns `true` when the store is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }
}

impl Default for ArtifactStore {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn add_available(store: &mut ArtifactStore, job_id: u64, created_at: i64) -> u64 {
        let id = store.register(job_id, "frame.exr", "s3://b/f.exr", 1024, "abc", created_at);
        store.mark_available(id);
        id
    }

    #[test]
    fn test_register_and_availability() {
        let mut store = ArtifactStore::new();
        let id = store.register(1, "f.exr", "s3://b/f.exr", 2048, "deadbeef", 1000);
        assert!(!store.is_empty());
        store.mark_available(id);
        let artifact = store.artifacts_for_job(1);
        assert_eq!(artifact[0].status, ArtifactStatus::Available);
    }

    #[test]
    fn test_total_bytes() {
        let mut store = ArtifactStore::new();
        add_available(&mut store, 1, 1000);
        let id2 = store.register(1, "f2.exr", "s3://b/f2.exr", 512, "cafe", 2000);
        store.mark_available(id2);
        assert_eq!(store.total_bytes(), 1536);
    }

    #[test]
    fn test_time_based_expiry() {
        let mut store = ArtifactStore::new();
        let id = add_available(&mut store, 1, 0);
        store.set_retention(id, RetentionPolicy::DaysAfterCreation { days: 1 });
        // 2 days later
        store.apply_retention(2 * 86_400);
        let a = &store.artifacts_for_job(1)[0];
        assert_eq!(a.status, ArtifactStatus::PendingDelete);
    }

    #[test]
    fn test_keep_latest_n() {
        let mut store = ArtifactStore::new();
        for ts in [100, 200, 300, 400, 500] {
            let id = add_available(&mut store, 7, ts);
            store.set_retention(id, RetentionPolicy::KeepLatestN { n: 2 });
        }
        store.apply_retention(0);
        let pending: Vec<_> = store
            .artifacts_for_job(7)
            .into_iter()
            .filter(|a| a.status == ArtifactStatus::PendingDelete)
            .collect();
        // 5 artifacts, keep 2 → 3 should be pending delete
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_purge_pending() {
        let mut store = ArtifactStore::new();
        let id = add_available(&mut store, 1, 0);
        store.set_retention(id, RetentionPolicy::DaysAfterCreation { days: 1 });
        store.apply_retention(2 * 86_400);
        let purged = store.purge_pending();
        assert_eq!(purged, 1);
        assert!(store.is_empty());
    }

    #[test]
    fn test_retention_forever_not_expired() {
        let mut store = ArtifactStore::new();
        let id = add_available(&mut store, 1, 0);
        store.set_retention(id, RetentionPolicy::Forever);
        store.apply_retention(10_000_000);
        let a = &store.artifacts_for_job(1)[0];
        assert_eq!(a.status, ArtifactStatus::Available);
    }

    #[test]
    fn test_mark_available_nonexistent_returns_false() {
        let mut store = ArtifactStore::new();
        assert!(!store.mark_available(999));
    }

    #[test]
    fn test_set_retention_nonexistent_returns_false() {
        let mut store = ArtifactStore::new();
        assert!(!store.set_retention(999, RetentionPolicy::Forever));
    }
}
