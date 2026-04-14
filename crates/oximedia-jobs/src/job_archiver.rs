// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job archival — cold-storage archiving for completed and failed jobs.
//!
//! `JobArchiver` moves terminal jobs (Completed, Failed, Cancelled) out of the
//! hot queue into a dedicated in-memory cold store, freeing queue capacity while
//! keeping job history accessible for auditing, replay, and reporting.
//!
//! # Design
//! - `ArchivePolicy` controls which statuses are archived and the minimum age
//!   before a job becomes eligible.
//! - `ArchivedJob` wraps a `Job` snapshot with an `archived_at` timestamp and
//!   an optional `archive_reason` string.
//! - `JobArchiver` stores archives in a `HashMap<Uuid, ArchivedJob>` and
//!   enforces a configurable `max_capacity`.  When the store is full, the
//!   oldest entries are evicted.
//! - `PruningPolicy` trims archives by age and/or total count.
//! - `restore` moves an `ArchivedJob` back into a fresh `Job` (Pending state).
//!
//! # Example
//! ```rust
//! use oximedia_jobs::job_archiver::{JobArchiver, ArchivePolicy, PruningPolicy};
//! use oximedia_jobs::{Job, JobPayload, Priority, JobStatus, TranscodeParams};
//!
//! let params = TranscodeParams {
//!     input: "in.mp4".into(), output: "out.mp4".into(),
//!     video_codec: "h264".into(), audio_codec: "aac".into(),
//!     video_bitrate: 4_000_000, audio_bitrate: 128_000,
//!     resolution: None, framerate: None,
//!     preset: "fast".into(), hw_accel: None,
//! };
//! let mut job = Job::new("encode".into(), Priority::Normal, JobPayload::Transcode(params));
//! job.status = JobStatus::Completed;
//!
//! let policy = ArchivePolicy::default();
//! let mut archiver = JobArchiver::new(policy, 1000);
//!
//! archiver.archive(job, None).expect("archive should succeed");
//! assert_eq!(archiver.len(), 1);
//! ```

use crate::job::{Job, JobStatus};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during archival operations.
#[derive(Debug, thiserror::Error)]
pub enum ArchiveError {
    /// The job is in a non-terminal status and cannot be archived.
    #[error("Job {0} is not in a terminal status (current: {1})")]
    NonTerminalStatus(Uuid, String),
    /// The job is too new to be archived (minimum age not met).
    #[error("Job {0} does not meet the minimum age requirement for archival")]
    TooYoung(Uuid),
    /// The requested archive entry was not found.
    #[error("Archived job {0} not found")]
    NotFound(Uuid),
    /// The archive store is full and eviction was disabled.
    #[error("Archive at capacity ({0} entries); enable eviction or increase capacity")]
    AtCapacity(usize),
}

// ---------------------------------------------------------------------------
// ArchivePolicy
// ---------------------------------------------------------------------------

/// Controls which jobs are eligible for archival and when.
#[derive(Debug, Clone)]
pub struct ArchivePolicy {
    /// Archive jobs with `Completed` status.
    pub archive_completed: bool,
    /// Archive jobs with `Failed` status.
    pub archive_failed: bool,
    /// Archive jobs with `Cancelled` status.
    pub archive_cancelled: bool,
    /// Minimum age since `completed_at` (or `created_at` if unknown) before
    /// a job is eligible.  `None` means archive immediately.
    pub min_age: Option<Duration>,
    /// When `true`, automatically evict the oldest entry when the store is
    /// full rather than returning [`ArchiveError::AtCapacity`].
    pub evict_on_full: bool,
}

impl Default for ArchivePolicy {
    fn default() -> Self {
        Self {
            archive_completed: true,
            archive_failed: true,
            archive_cancelled: true,
            min_age: None,
            evict_on_full: true,
        }
    }
}

impl ArchivePolicy {
    /// Return `true` if the given status qualifies under this policy.
    #[must_use]
    pub fn status_eligible(&self, status: JobStatus) -> bool {
        match status {
            JobStatus::Completed => self.archive_completed,
            JobStatus::Failed => self.archive_failed,
            JobStatus::Cancelled => self.archive_cancelled,
            _ => false,
        }
    }

    /// Return `true` if the job meets the minimum age requirement.
    #[must_use]
    pub fn age_eligible(&self, job: &Job) -> bool {
        match self.min_age {
            None => true,
            Some(min_age) => {
                let reference = job.completed_at.unwrap_or(job.created_at);
                Utc::now() - reference >= min_age
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PruningPolicy
// ---------------------------------------------------------------------------

/// Controls how old archives are trimmed.
#[derive(Debug, Clone)]
pub struct PruningPolicy {
    /// Remove archives older than this duration.  `None` disables age-based pruning.
    pub max_age: Option<Duration>,
    /// Keep at most this many archives per status.  `None` disables count pruning.
    pub max_count_per_status: Option<usize>,
}

impl Default for PruningPolicy {
    fn default() -> Self {
        Self {
            max_age: Some(Duration::days(30)),
            max_count_per_status: Some(10_000),
        }
    }
}

// ---------------------------------------------------------------------------
// ArchivedJob
// ---------------------------------------------------------------------------

/// A job snapshot stored in cold archive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivedJob {
    /// The archived job snapshot.
    pub job: Job,
    /// Timestamp when the job was moved into the archive.
    pub archived_at: DateTime<Utc>,
    /// Human-readable reason for archival (e.g. "retention policy", "manual").
    pub archive_reason: Option<String>,
}

impl ArchivedJob {
    /// Create a new `ArchivedJob` from a `Job`.
    #[must_use]
    pub fn new(job: Job, reason: Option<String>) -> Self {
        Self {
            job,
            archived_at: Utc::now(),
            archive_reason: reason,
        }
    }

    /// Age of this archive entry relative to `now`.
    #[must_use]
    pub fn age(&self) -> Duration {
        Utc::now() - self.archived_at
    }
}

// ---------------------------------------------------------------------------
// ArchiveStats
// ---------------------------------------------------------------------------

/// Aggregate statistics over the archive store.
#[derive(Debug, Clone, Default)]
pub struct ArchiveStats {
    /// Total number of archived entries.
    pub total: usize,
    /// Number of completed-job archives.
    pub completed_count: usize,
    /// Number of failed-job archives.
    pub failed_count: usize,
    /// Number of cancelled-job archives.
    pub cancelled_count: usize,
    /// Total entries evicted due to capacity limits.
    pub evictions: u64,
    /// Total entries removed by pruning.
    pub prunes: u64,
}

// ---------------------------------------------------------------------------
// JobArchiver
// ---------------------------------------------------------------------------

/// Cold-storage archive for terminal jobs.
pub struct JobArchiver {
    /// Archive policy governing eligibility.
    policy: ArchivePolicy,
    /// Maximum number of entries in the store.
    max_capacity: usize,
    /// The archive store (job_id → ArchivedJob).
    store: HashMap<Uuid, ArchivedJob>,
    /// Monotonic insertion order (for LRU eviction).
    insertion_order: Vec<Uuid>,
    /// Aggregate statistics.
    stats: ArchiveStats,
}

impl JobArchiver {
    /// Create a new `JobArchiver` with the given policy and capacity.
    #[must_use]
    pub fn new(policy: ArchivePolicy, max_capacity: usize) -> Self {
        Self {
            policy,
            max_capacity,
            store: HashMap::new(),
            insertion_order: Vec::new(),
            stats: ArchiveStats::default(),
        }
    }

    /// Archive a job, optionally with a human-readable reason.
    ///
    /// # Errors
    ///
    /// - [`ArchiveError::NonTerminalStatus`] if the job is still active.
    /// - [`ArchiveError::TooYoung`] if the job does not meet the minimum age.
    /// - [`ArchiveError::AtCapacity`] if the store is full and eviction is disabled.
    pub fn archive(&mut self, job: Job, reason: Option<String>) -> Result<Uuid, ArchiveError> {
        if !self.policy.status_eligible(job.status) {
            return Err(ArchiveError::NonTerminalStatus(
                job.id,
                job.status.to_string(),
            ));
        }

        if !self.policy.age_eligible(&job) {
            return Err(ArchiveError::TooYoung(job.id));
        }

        // Evict oldest entry if at capacity.
        if self.store.len() >= self.max_capacity {
            if self.policy.evict_on_full {
                self.evict_oldest();
            } else {
                return Err(ArchiveError::AtCapacity(self.max_capacity));
            }
        }

        let id = job.id;

        // Update stats.
        match job.status {
            JobStatus::Completed => self.stats.completed_count += 1,
            JobStatus::Failed => self.stats.failed_count += 1,
            JobStatus::Cancelled => self.stats.cancelled_count += 1,
            _ => {}
        }
        self.stats.total += 1;

        let archived = ArchivedJob::new(job, reason);
        self.store.insert(id, archived);
        self.insertion_order.push(id);

        Ok(id)
    }

    /// Retrieve an archived job by ID.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&ArchivedJob> {
        self.store.get(&id)
    }

    /// Restore an archived job as a fresh `Job` in `Pending` status.
    ///
    /// The restored job receives a **new** UUID, resets all execution state,
    /// and retains the original name, priority, payload, tags, and retry policy.
    ///
    /// # Errors
    ///
    /// Returns [`ArchiveError::NotFound`] if the ID is not in the archive.
    pub fn restore(&mut self, id: Uuid) -> Result<Job, ArchiveError> {
        let archived = self
            .store
            .remove(&id)
            .ok_or(ArchiveError::NotFound(id))?;
        self.insertion_order.retain(|&x| x != id);
        self.stats.total = self.stats.total.saturating_sub(1);

        let original = archived.job;
        let mut fresh = Job::new(
            original.name.clone(),
            original.priority,
            original.payload.clone(),
        );
        fresh.tags = original.tags.clone();
        fresh.retry_policy = original.retry_policy.clone();
        fresh.resource_quota = original.resource_quota.clone();
        Ok(fresh)
    }

    /// Remove an archived job without restoring it.
    ///
    /// # Errors
    ///
    /// Returns [`ArchiveError::NotFound`] if the ID does not exist.
    pub fn delete(&mut self, id: Uuid) -> Result<ArchivedJob, ArchiveError> {
        let archived = self
            .store
            .remove(&id)
            .ok_or(ArchiveError::NotFound(id))?;
        self.insertion_order.retain(|&x| x != id);
        self.stats.total = self.stats.total.saturating_sub(1);
        Ok(archived)
    }

    /// Apply a `PruningPolicy`, removing entries that violate age or count
    /// constraints.  Returns the number of entries pruned.
    pub fn prune(&mut self, pruning: &PruningPolicy) -> usize {
        let mut removed = 0usize;

        // Age-based pruning.
        if let Some(max_age) = pruning.max_age {
            let cutoff = Utc::now() - max_age;
            let stale: Vec<Uuid> = self
                .store
                .iter()
                .filter(|(_, a)| a.archived_at < cutoff)
                .map(|(id, _)| *id)
                .collect();
            for id in stale {
                self.store.remove(&id);
                self.insertion_order.retain(|&x| x != id);
                removed += 1;
            }
        }

        // Per-status count pruning: keep the newest `max_count` per status.
        if let Some(max_count) = pruning.max_count_per_status {
            for target_status in [JobStatus::Completed, JobStatus::Failed, JobStatus::Cancelled] {
                let mut by_status: Vec<(Uuid, DateTime<Utc>)> = self
                    .store
                    .iter()
                    .filter(|(_, a)| a.job.status == target_status)
                    .map(|(id, a)| (*id, a.archived_at))
                    .collect();

                if by_status.len() > max_count {
                    // Sort oldest-first.
                    by_status.sort_by_key(|&(_, ts)| ts);
                    let excess = by_status.len() - max_count;
                    for (id, _) in by_status.iter().take(excess) {
                        self.store.remove(id);
                        self.insertion_order.retain(|&x| x != *id);
                        removed += 1;
                    }
                }
            }
        }

        if removed > 0 {
            self.stats.total = self.store.len();
            self.stats.prunes += removed as u64;
        }

        removed
    }

    /// Number of entries currently in the archive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Iterate over all archived jobs.
    pub fn iter(&self) -> impl Iterator<Item = (&Uuid, &ArchivedJob)> {
        self.store.iter()
    }

    /// Returns aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &ArchiveStats {
        &self.stats
    }

    /// List all archived job IDs sorted by archive timestamp (newest first).
    #[must_use]
    pub fn list_by_date(&self) -> Vec<Uuid> {
        let mut entries: Vec<(Uuid, DateTime<Utc>)> = self
            .store
            .iter()
            .map(|(id, a)| (*id, a.archived_at))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1));
        entries.into_iter().map(|(id, _)| id).collect()
    }

    /// List archived jobs filtered by status.
    #[must_use]
    pub fn list_by_status(&self, status: JobStatus) -> Vec<&ArchivedJob> {
        self.store
            .values()
            .filter(|a| a.job.status == status)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn evict_oldest(&mut self) {
        if let Some(oldest_id) = self.insertion_order.first().copied() {
            self.store.remove(&oldest_id);
            self.insertion_order.remove(0);
            self.stats.evictions += 1;
            self.stats.total = self.stats.total.saturating_sub(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobPayload, Priority, TranscodeParams};

    fn make_transcode_job(name: &str, status: JobStatus) -> Job {
        let params = TranscodeParams {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".into(),
            hw_accel: None,
        };
        let mut job = Job::new(name.into(), Priority::Normal, JobPayload::Transcode(params));
        job.status = status;
        job
    }

    #[test]
    fn test_archive_completed_job() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("encode", JobStatus::Completed);
        let id = archiver.archive(job, None).expect("should archive");
        assert_eq!(archiver.len(), 1);
        assert!(archiver.get(id).is_some());
    }

    #[test]
    fn test_archive_failed_job() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("encode-fail", JobStatus::Failed);
        archiver.archive(job, Some("max retries exceeded".into())).expect("should archive");
        assert_eq!(archiver.stats().failed_count, 1);
    }

    #[test]
    fn test_archive_rejects_running_job() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("running", JobStatus::Running);
        let result = archiver.archive(job, None);
        assert!(matches!(result, Err(ArchiveError::NonTerminalStatus(_, _))));
    }

    #[test]
    fn test_archive_rejects_pending_job() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("pending", JobStatus::Pending);
        let result = archiver.archive(job, None);
        assert!(matches!(result, Err(ArchiveError::NonTerminalStatus(_, _))));
    }

    #[test]
    fn test_restore_creates_fresh_job() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let original = make_transcode_job("encode", JobStatus::Completed);
        let original_name = original.name.clone();
        let original_id = original.id;
        let archived_id = archiver.archive(original, None).expect("should archive");

        let fresh = archiver.restore(archived_id).expect("should restore");
        assert_eq!(fresh.name, original_name);
        assert_ne!(fresh.id, original_id); // new UUID
        assert_eq!(fresh.status, JobStatus::Pending);
        assert_eq!(archiver.len(), 0); // removed from archive
    }

    #[test]
    fn test_delete_entry() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("to-delete", JobStatus::Completed);
        let id = archiver.archive(job, None).expect("should archive");
        let deleted = archiver.delete(id).expect("should delete");
        assert_eq!(deleted.job.name, "to-delete");
        assert_eq!(archiver.len(), 0);
    }

    #[test]
    fn test_eviction_when_at_capacity() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 2);
        let j1 = make_transcode_job("j1", JobStatus::Completed);
        let j2 = make_transcode_job("j2", JobStatus::Completed);
        let j3 = make_transcode_job("j3", JobStatus::Completed);
        archiver.archive(j1, None).expect("j1 ok");
        archiver.archive(j2, None).expect("j2 ok");
        archiver.archive(j3, None).expect("j3 ok"); // should evict j1
        assert_eq!(archiver.len(), 2);
        assert_eq!(archiver.stats().evictions, 1);
    }

    #[test]
    fn test_at_capacity_error_when_eviction_disabled() {
        let mut policy = ArchivePolicy::default();
        policy.evict_on_full = false;
        let mut archiver = JobArchiver::new(policy, 1);
        let j1 = make_transcode_job("j1", JobStatus::Completed);
        let j2 = make_transcode_job("j2", JobStatus::Completed);
        archiver.archive(j1, None).expect("j1 ok");
        let result = archiver.archive(j2, None);
        assert!(matches!(result, Err(ArchiveError::AtCapacity(1))));
    }

    #[test]
    fn test_pruning_by_age() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let job = make_transcode_job("old-job", JobStatus::Completed);
        let id = archiver.archive(job, None).expect("should archive");

        // Artificially backdate the archived_at timestamp.
        if let Some(entry) = archiver.store.get_mut(&id) {
            entry.archived_at = Utc::now() - Duration::days(60);
        }

        let pruning = PruningPolicy {
            max_age: Some(Duration::days(30)),
            max_count_per_status: None,
        };
        let pruned = archiver.prune(&pruning);
        assert_eq!(pruned, 1);
        assert_eq!(archiver.len(), 0);
    }

    #[test]
    fn test_pruning_by_count() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 200);
        for i in 0..10 {
            let j = make_transcode_job(&format!("job-{i}"), JobStatus::Completed);
            archiver.archive(j, None).expect("should archive");
        }
        let pruning = PruningPolicy {
            max_age: None,
            max_count_per_status: Some(5),
        };
        let pruned = archiver.prune(&pruning);
        assert_eq!(pruned, 5);
        assert_eq!(archiver.len(), 5);
    }

    #[test]
    fn test_list_by_status() {
        let mut archiver = JobArchiver::new(ArchivePolicy::default(), 100);
        let c = make_transcode_job("c", JobStatus::Completed);
        let f = make_transcode_job("f", JobStatus::Failed);
        archiver.archive(c, None).expect("c ok");
        archiver.archive(f, None).expect("f ok");

        let completed = archiver.list_by_status(JobStatus::Completed);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].job.name, "c");
    }

    #[test]
    fn test_min_age_policy() {
        let mut policy = ArchivePolicy::default();
        policy.min_age = Some(Duration::hours(1));
        let mut archiver = JobArchiver::new(policy, 100);

        // Job completed just now — not old enough.
        let mut job = make_transcode_job("fresh", JobStatus::Completed);
        job.completed_at = Some(Utc::now());
        let result = archiver.archive(job, None);
        assert!(matches!(result, Err(ArchiveError::TooYoung(_))));
    }
}
