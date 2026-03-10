//! Multi-site storage replication for high availability and disaster recovery.
//!
//! This module provides types and a manager for replicating storage objects
//! across multiple sites or providers, supporting synchronous, asynchronous,
//! and scheduled replication modes.

#![allow(dead_code)]

use std::collections::HashMap;

/// Synchronisation mode for a replication target
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncMode {
    /// Wait for the target to acknowledge before returning success
    Synchronous,
    /// Return immediately; replicate in the background
    Asynchronous,
    /// Replicate on a fixed schedule (interval in seconds)
    Scheduled(u32),
}

/// A destination site for replication
#[derive(Debug, Clone)]
pub struct ReplicationTarget {
    /// Unique identifier for the remote site
    pub site_id: String,
    /// URL of the remote storage endpoint
    pub url: String,
    /// Priority (lower number = higher priority)
    pub priority: u8,
    /// Synchronisation mode
    pub sync_mode: SyncMode,
}

impl ReplicationTarget {
    /// Create a new synchronous replication target
    pub fn new_sync(site_id: impl Into<String>, url: impl Into<String>, priority: u8) -> Self {
        Self {
            site_id: site_id.into(),
            url: url.into(),
            priority,
            sync_mode: SyncMode::Synchronous,
        }
    }

    /// Create a new asynchronous replication target
    pub fn new_async(site_id: impl Into<String>, url: impl Into<String>, priority: u8) -> Self {
        Self {
            site_id: site_id.into(),
            url: url.into(),
            priority,
            sync_mode: SyncMode::Asynchronous,
        }
    }
}

/// Current status of a replication job
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicationStatus {
    /// Waiting to start
    Pending,
    /// Currently being replicated
    Running,
    /// Successfully completed
    Completed,
    /// Failed with an error
    Failed,
    /// Cancelled before completion
    Cancelled,
}

impl ReplicationStatus {
    /// Returns `true` if the job is in a terminal state (will not change)
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

impl std::fmt::Display for ReplicationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
        }
    }
}

/// A replication job that copies a source object to one or more targets
#[derive(Debug, Clone)]
pub struct ReplicationJob {
    /// Unique job identifier
    pub id: String,
    /// Source object path or key
    pub source_path: String,
    /// Destination targets
    pub targets: Vec<ReplicationTarget>,
    /// Current job status
    pub status: ReplicationStatus,
    /// Number of bytes successfully replicated so far
    pub bytes_replicated: u64,
}

impl ReplicationJob {
    /// Create a new pending replication job
    pub fn new(
        id: impl Into<String>,
        source_path: impl Into<String>,
        targets: Vec<ReplicationTarget>,
    ) -> Self {
        Self {
            id: id.into(),
            source_path: source_path.into(),
            targets,
            status: ReplicationStatus::Pending,
            bytes_replicated: 0,
        }
    }

    /// Returns `true` if the job is in a terminal state
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }
}

/// Aggregate statistics for all replication activity
#[derive(Debug, Clone, Default)]
pub struct ReplicationStats {
    /// Total bytes that need to be replicated
    pub total_bytes: u64,
    /// Bytes successfully replicated
    pub replicated_bytes: u64,
    /// Bytes that failed to replicate
    pub failed_bytes: u64,
    /// Current throughput in megabytes per second
    pub throughput_mbps: f32,
}

impl ReplicationStats {
    /// Returns the replication completion percentage (0.0–100.0)
    pub fn completion_pct(&self) -> f32 {
        if self.total_bytes == 0 {
            return 100.0;
        }
        (self.replicated_bytes as f32 / self.total_bytes as f32) * 100.0
    }
}

/// Manages the lifecycle of replication jobs
pub struct ReplicationManager {
    jobs: HashMap<String, ReplicationJob>,
}

impl ReplicationManager {
    /// Create a new, empty replication manager
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    /// Submit a new replication job (status must be `Pending`)
    pub fn submit(&mut self, job: ReplicationJob) {
        self.jobs.insert(job.id.clone(), job);
    }

    /// Cancel a job by ID.  Does nothing if the job is already terminal.
    pub fn cancel(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if !job.is_terminal() {
                job.status = ReplicationStatus::Cancelled;
            }
        }
    }

    /// Mark a job as running (simulates starting execution)
    pub fn start(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if job.status == ReplicationStatus::Pending {
                job.status = ReplicationStatus::Running;
            }
        }
    }

    /// Mark a running job as completed
    pub fn complete(&mut self, job_id: &str, bytes_replicated: u64) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if job.status == ReplicationStatus::Running {
                job.status = ReplicationStatus::Completed;
                job.bytes_replicated = bytes_replicated;
            }
        }
    }

    /// Mark a running job as failed
    pub fn fail(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if !job.is_terminal() {
                job.status = ReplicationStatus::Failed;
            }
        }
    }

    /// Returns references to all non-terminal jobs
    pub fn active_jobs(&self) -> Vec<&ReplicationJob> {
        self.jobs.values().filter(|j| !j.is_terminal()).collect()
    }

    /// Returns references to all completed jobs
    pub fn completed_jobs(&self) -> Vec<&ReplicationJob> {
        self.jobs
            .values()
            .filter(|j| j.status == ReplicationStatus::Completed)
            .collect()
    }

    /// Returns the job with the given ID, if it exists
    pub fn get_job(&self, job_id: &str) -> Option<&ReplicationJob> {
        self.jobs.get(job_id)
    }

    /// Returns aggregate statistics across all jobs
    pub fn stats(&self) -> ReplicationStats {
        let mut stats = ReplicationStats::default();
        for job in self.jobs.values() {
            match job.status {
                ReplicationStatus::Completed => {
                    stats.total_bytes += job.bytes_replicated;
                    stats.replicated_bytes += job.bytes_replicated;
                }
                ReplicationStatus::Failed => {
                    stats.failed_bytes += job.bytes_replicated;
                }
                _ => {}
            }
        }
        stats
    }
}

impl Default for ReplicationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifier for checking data consistency between replica sites
pub struct ConsistencyCheck;

impl ConsistencyCheck {
    /// Verify that two checksums match, indicating consistent replicas
    pub fn verify(_path: &str, checksum_a: u64, checksum_b: u64) -> bool {
        checksum_a == checksum_b
    }

    /// Find paths where the two replica checksums differ.
    ///
    /// Each entry in `checksums` is `(path, checksum_a, checksum_b)`.
    /// Returns the paths that are inconsistent.
    pub fn find_inconsistencies(checksums: &[(String, u64, u64)]) -> Vec<String> {
        checksums
            .iter()
            .filter(|(path, a, b)| !Self::verify(path, *a, *b))
            .map(|(path, _, _)| path.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_targets() -> Vec<ReplicationTarget> {
        vec![
            ReplicationTarget::new_sync("site-a", "https://site-a.example.com", 1),
            ReplicationTarget::new_async("site-b", "https://site-b.example.com", 2),
        ]
    }

    // --- SyncMode ---

    #[test]
    fn test_sync_mode_scheduled() {
        let mode = SyncMode::Scheduled(3600);
        assert_eq!(mode, SyncMode::Scheduled(3600));
        assert_ne!(mode, SyncMode::Scheduled(7200));
    }

    // --- ReplicationStatus ---

    #[test]
    fn test_status_terminal_states() {
        assert!(ReplicationStatus::Completed.is_terminal());
        assert!(ReplicationStatus::Failed.is_terminal());
        assert!(ReplicationStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_status_non_terminal_states() {
        assert!(!ReplicationStatus::Pending.is_terminal());
        assert!(!ReplicationStatus::Running.is_terminal());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(ReplicationStatus::Pending.to_string(), "Pending");
        assert_eq!(ReplicationStatus::Running.to_string(), "Running");
        assert_eq!(ReplicationStatus::Completed.to_string(), "Completed");
        assert_eq!(ReplicationStatus::Failed.to_string(), "Failed");
        assert_eq!(ReplicationStatus::Cancelled.to_string(), "Cancelled");
    }

    // --- ReplicationJob ---

    #[test]
    fn test_job_created_as_pending() {
        let job = ReplicationJob::new("job-1", "videos/clip.mp4", make_targets());
        assert_eq!(job.status, ReplicationStatus::Pending);
        assert_eq!(job.bytes_replicated, 0);
    }

    // --- ReplicationManager ---

    #[test]
    fn test_manager_submit_and_active() {
        let mut mgr = ReplicationManager::new();
        let job = ReplicationJob::new("j1", "path/a", make_targets());
        mgr.submit(job);
        assert_eq!(mgr.active_jobs().len(), 1);
        assert_eq!(mgr.completed_jobs().len(), 0);
    }

    #[test]
    fn test_manager_start_and_complete() {
        let mut mgr = ReplicationManager::new();
        mgr.submit(ReplicationJob::new("j2", "path/b", make_targets()));
        mgr.start("j2");
        assert_eq!(
            mgr.get_job("j2").expect("job should exist").status,
            ReplicationStatus::Running
        );
        mgr.complete("j2", 1_024_000);
        assert_eq!(
            mgr.get_job("j2").expect("job should exist").status,
            ReplicationStatus::Completed
        );
        assert_eq!(mgr.active_jobs().len(), 0);
        assert_eq!(mgr.completed_jobs().len(), 1);
    }

    #[test]
    fn test_manager_cancel() {
        let mut mgr = ReplicationManager::new();
        mgr.submit(ReplicationJob::new("j3", "path/c", make_targets()));
        mgr.cancel("j3");
        assert_eq!(
            mgr.get_job("j3").expect("job should exist").status,
            ReplicationStatus::Cancelled
        );
        // Cancelling again has no effect
        mgr.cancel("j3");
        assert_eq!(
            mgr.get_job("j3").expect("job should exist").status,
            ReplicationStatus::Cancelled
        );
    }

    #[test]
    fn test_manager_fail() {
        let mut mgr = ReplicationManager::new();
        mgr.submit(ReplicationJob::new("j4", "path/d", make_targets()));
        mgr.start("j4");
        mgr.fail("j4");
        assert_eq!(
            mgr.get_job("j4").expect("job should exist").status,
            ReplicationStatus::Failed
        );
        assert!(mgr.active_jobs().is_empty());
    }

    #[test]
    fn test_manager_stats() {
        let mut mgr = ReplicationManager::new();
        mgr.submit(ReplicationJob::new("j5", "p1", make_targets()));
        mgr.start("j5");
        mgr.complete("j5", 2_000_000);

        let stats = mgr.stats();
        assert_eq!(stats.replicated_bytes, 2_000_000);
        assert_eq!(stats.failed_bytes, 0);
    }

    // --- ConsistencyCheck ---

    #[test]
    fn test_verify_matching_checksums() {
        assert!(ConsistencyCheck::verify("file.mp4", 12345, 12345));
    }

    #[test]
    fn test_verify_mismatched_checksums() {
        assert!(!ConsistencyCheck::verify("file.mp4", 12345, 99999));
    }

    #[test]
    fn test_find_inconsistencies() {
        let checksums = vec![
            ("a.mp4".to_string(), 111, 111),
            ("b.mp4".to_string(), 222, 333), // inconsistent
            ("c.mp4".to_string(), 444, 444),
        ];
        let bad = ConsistencyCheck::find_inconsistencies(&checksums);
        assert_eq!(bad, vec!["b.mp4"]);
    }

    #[test]
    fn test_find_inconsistencies_all_match() {
        let checksums = vec![("x.mp4".to_string(), 1, 1), ("y.mp4".to_string(), 2, 2)];
        assert!(ConsistencyCheck::find_inconsistencies(&checksums).is_empty());
    }

    #[test]
    fn test_replication_stats_completion_pct_zero_total() {
        let stats = ReplicationStats::default();
        assert!((stats.completion_pct() - 100.0).abs() < f32::EPSILON);
    }
}
