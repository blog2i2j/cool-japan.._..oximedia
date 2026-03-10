#![allow(dead_code)]
//! Archive completed batch jobs for historical querying and analytics.
//!
//! After a job finishes (success or failure) it can be moved into an archive
//! that supports time-range queries, statistics, and retention policies.

use std::collections::HashMap;

/// Status of an archived job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchivedStatus {
    /// The job completed successfully.
    Succeeded,
    /// The job failed.
    Failed,
    /// The job was cancelled.
    Cancelled,
    /// The job timed out.
    TimedOut,
}

impl ArchivedStatus {
    /// Whether the status represents a successful outcome.
    #[must_use]
    pub fn is_success(self) -> bool {
        self == Self::Succeeded
    }
}

/// A single archived job record.
#[derive(Debug, Clone)]
pub struct ArchivedJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Human-readable job name.
    pub name: String,
    /// Final status.
    pub status: ArchivedStatus,
    /// Unix timestamp when the job was submitted.
    pub submitted_at: u64,
    /// Unix timestamp when the job started executing.
    pub started_at: u64,
    /// Unix timestamp when the job finished.
    pub finished_at: u64,
    /// Wall-clock duration in seconds.
    pub duration_secs: f64,
    /// Number of retry attempts that were made.
    pub retries: u32,
    /// Optional error message (for failed jobs).
    pub error_message: Option<String>,
    /// Arbitrary key-value tags for filtering.
    pub tags: HashMap<String, String>,
}

impl ArchivedJob {
    /// Create a new archived job record.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        name: impl Into<String>,
        status: ArchivedStatus,
        submitted_at: u64,
        started_at: u64,
        finished_at: u64,
        retries: u32,
    ) -> Self {
        let finished = finished_at;
        let started = started_at;
        #[allow(clippy::cast_precision_loss)]
        let duration = if finished > started {
            (finished - started) as f64
        } else {
            0.0
        };
        Self {
            job_id: job_id.into(),
            name: name.into(),
            status,
            submitted_at,
            started_at,
            finished_at,
            duration_secs: duration,
            retries,
            error_message: None,
            tags: HashMap::new(),
        }
    }

    /// Set error message.
    #[must_use]
    pub fn with_error(mut self, msg: impl Into<String>) -> Self {
        self.error_message = Some(msg.into());
        self
    }

    /// Add a tag.
    #[must_use]
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Queue wait time in seconds.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn queue_wait_secs(&self) -> f64 {
        if self.started_at > self.submitted_at {
            (self.started_at - self.submitted_at) as f64
        } else {
            0.0
        }
    }
}

/// Statistics computed over a set of archived jobs.
#[derive(Debug, Clone)]
pub struct ArchiveStats {
    /// Total number of jobs.
    pub total: usize,
    /// Number of successful jobs.
    pub succeeded: usize,
    /// Number of failed jobs.
    pub failed: usize,
    /// Number of cancelled jobs.
    pub cancelled: usize,
    /// Number of timed-out jobs.
    pub timed_out: usize,
    /// Average duration in seconds.
    pub avg_duration_secs: f64,
    /// Maximum duration in seconds.
    pub max_duration_secs: f64,
    /// Minimum duration in seconds.
    pub min_duration_secs: f64,
    /// Average queue wait time in seconds.
    pub avg_queue_wait_secs: f64,
    /// Total retries across all jobs.
    pub total_retries: u64,
}

/// Retention policy for the archive.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Maximum age of records in seconds (0 = keep forever).
    pub max_age_secs: u64,
    /// Maximum number of records to keep (0 = unlimited).
    pub max_records: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age_secs: 90 * 24 * 3600, // 90 days
            max_records: 100_000,
        }
    }
}

/// The job archive that stores and queries historical job data.
#[derive(Debug, Clone)]
pub struct JobArchive {
    /// Stored jobs, most recent last.
    jobs: Vec<ArchivedJob>,
    /// Retention configuration.
    retention: RetentionPolicy,
}

impl Default for JobArchive {
    fn default() -> Self {
        Self::new(RetentionPolicy::default())
    }
}

impl JobArchive {
    /// Create a new archive with the given retention policy.
    #[must_use]
    pub fn new(retention: RetentionPolicy) -> Self {
        Self {
            jobs: Vec::new(),
            retention,
        }
    }

    /// Insert a job into the archive.
    pub fn insert(&mut self, job: ArchivedJob) {
        self.jobs.push(job);
    }

    /// Total number of archived records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Apply the retention policy, removing old or excess records.
    pub fn apply_retention(&mut self, current_time: u64) {
        if self.retention.max_age_secs > 0 {
            let cutoff = current_time.saturating_sub(self.retention.max_age_secs);
            self.jobs.retain(|j| j.finished_at >= cutoff);
        }
        if self.retention.max_records > 0 && self.jobs.len() > self.retention.max_records {
            let excess = self.jobs.len() - self.retention.max_records;
            self.jobs.drain(..excess);
        }
    }

    /// Query jobs in a time range (by `finished_at`).
    #[must_use]
    pub fn query_time_range(&self, from: u64, to: u64) -> Vec<&ArchivedJob> {
        self.jobs
            .iter()
            .filter(|j| j.finished_at >= from && j.finished_at <= to)
            .collect()
    }

    /// Query jobs by status.
    #[must_use]
    pub fn query_status(&self, status: ArchivedStatus) -> Vec<&ArchivedJob> {
        self.jobs.iter().filter(|j| j.status == status).collect()
    }

    /// Query jobs by tag key-value pair.
    #[must_use]
    pub fn query_tag(&self, key: &str, value: &str) -> Vec<&ArchivedJob> {
        self.jobs
            .iter()
            .filter(|j| j.tags.get(key).is_some_and(|v| v == value))
            .collect()
    }

    /// Find a specific job by id.
    #[must_use]
    pub fn find_by_id(&self, job_id: &str) -> Option<&ArchivedJob> {
        self.jobs.iter().find(|j| j.job_id == job_id)
    }

    /// Compute statistics over all archived jobs.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn stats(&self) -> ArchiveStats {
        if self.jobs.is_empty() {
            return ArchiveStats {
                total: 0,
                succeeded: 0,
                failed: 0,
                cancelled: 0,
                timed_out: 0,
                avg_duration_secs: 0.0,
                max_duration_secs: 0.0,
                min_duration_secs: 0.0,
                avg_queue_wait_secs: 0.0,
                total_retries: 0,
            };
        }

        let mut succeeded = 0usize;
        let mut failed = 0usize;
        let mut cancelled = 0usize;
        let mut timed_out = 0usize;
        let mut total_duration = 0.0_f64;
        let mut max_dur = f64::MIN;
        let mut min_dur = f64::MAX;
        let mut total_wait = 0.0_f64;
        let mut total_retries = 0_u64;

        for job in &self.jobs {
            match job.status {
                ArchivedStatus::Succeeded => succeeded += 1,
                ArchivedStatus::Failed => failed += 1,
                ArchivedStatus::Cancelled => cancelled += 1,
                ArchivedStatus::TimedOut => timed_out += 1,
            }
            total_duration += job.duration_secs;
            if job.duration_secs > max_dur {
                max_dur = job.duration_secs;
            }
            if job.duration_secs < min_dur {
                min_dur = job.duration_secs;
            }
            total_wait += job.queue_wait_secs();
            total_retries += u64::from(job.retries);
        }

        let count = self.jobs.len() as f64;
        ArchiveStats {
            total: self.jobs.len(),
            succeeded,
            failed,
            cancelled,
            timed_out,
            avg_duration_secs: total_duration / count,
            max_duration_secs: max_dur,
            min_duration_secs: min_dur,
            avg_queue_wait_secs: total_wait / count,
            total_retries,
        }
    }

    /// Return the most recent N jobs.
    #[must_use]
    pub fn recent(&self, n: usize) -> Vec<&ArchivedJob> {
        let start = self.jobs.len().saturating_sub(n);
        self.jobs[start..].iter().collect()
    }

    /// Clear the entire archive.
    pub fn clear(&mut self) {
        self.jobs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str, status: ArchivedStatus, finished: u64) -> ArchivedJob {
        ArchivedJob::new(
            id,
            format!("Job {id}"),
            status,
            finished - 20,
            finished - 10,
            finished,
            0,
        )
    }

    #[test]
    fn test_archived_status_is_success() {
        assert!(ArchivedStatus::Succeeded.is_success());
        assert!(!ArchivedStatus::Failed.is_success());
        assert!(!ArchivedStatus::Cancelled.is_success());
        assert!(!ArchivedStatus::TimedOut.is_success());
    }

    #[test]
    fn test_archived_job_creation() {
        let job = ArchivedJob::new(
            "j1",
            "Test Job",
            ArchivedStatus::Succeeded,
            100,
            110,
            120,
            0,
        );
        assert_eq!(job.job_id, "j1");
        assert!((job.duration_secs - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_archived_job_with_error() {
        let job = ArchivedJob::new("j2", "Fail", ArchivedStatus::Failed, 100, 110, 120, 1)
            .with_error("disk full");
        assert_eq!(job.error_message.as_deref(), Some("disk full"));
    }

    #[test]
    fn test_archived_job_with_tag() {
        let job = make_job("j3", ArchivedStatus::Succeeded, 200).with_tag("codec", "h264");
        assert_eq!(job.tags.get("codec").expect("failed to get value"), "h264");
    }

    #[test]
    fn test_archived_job_queue_wait() {
        let job = ArchivedJob::new("j4", "Wait", ArchivedStatus::Succeeded, 100, 115, 130, 0);
        assert!((job.queue_wait_secs() - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_archive_insert_and_len() {
        let mut archive = JobArchive::default();
        assert!(archive.is_empty());
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        assert_eq!(archive.len(), 1);
        assert!(!archive.is_empty());
    }

    #[test]
    fn test_archive_find_by_id() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Failed, 1100));
        assert!(archive.find_by_id("j1").is_some());
        assert!(archive.find_by_id("j999").is_none());
    }

    #[test]
    fn test_archive_query_status() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Failed, 1100));
        archive.insert(make_job("j3", ArchivedStatus::Succeeded, 1200));
        let ok = archive.query_status(ArchivedStatus::Succeeded);
        assert_eq!(ok.len(), 2);
        let fail = archive.query_status(ArchivedStatus::Failed);
        assert_eq!(fail.len(), 1);
    }

    #[test]
    fn test_archive_query_time_range() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Succeeded, 2000));
        archive.insert(make_job("j3", ArchivedStatus::Succeeded, 3000));
        let range = archive.query_time_range(1500, 2500);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].job_id, "j2");
    }

    #[test]
    fn test_archive_query_tag() {
        let mut archive = JobArchive::default();
        archive
            .insert(make_job("j1", ArchivedStatus::Succeeded, 1000).with_tag("type", "transcode"));
        archive.insert(make_job("j2", ArchivedStatus::Succeeded, 1100).with_tag("type", "copy"));
        let transcode = archive.query_tag("type", "transcode");
        assert_eq!(transcode.len(), 1);
    }

    #[test]
    fn test_archive_stats() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Failed, 1100));
        let stats = archive.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.succeeded, 1);
        assert_eq!(stats.failed, 1);
    }

    #[test]
    fn test_archive_stats_empty() {
        let archive = JobArchive::default();
        let stats = archive.stats();
        assert_eq!(stats.total, 0);
        assert!((stats.avg_duration_secs).abs() < f64::EPSILON);
    }

    #[test]
    fn test_archive_retention_max_records() {
        let policy = RetentionPolicy {
            max_age_secs: 0,
            max_records: 2,
        };
        let mut archive = JobArchive::new(policy);
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Succeeded, 2000));
        archive.insert(make_job("j3", ArchivedStatus::Succeeded, 3000));
        archive.apply_retention(3000);
        assert_eq!(archive.len(), 2);
        // oldest should be removed
        assert!(archive.find_by_id("j1").is_none());
    }

    #[test]
    fn test_archive_retention_max_age() {
        let policy = RetentionPolicy {
            max_age_secs: 500,
            max_records: 0,
        };
        let mut archive = JobArchive::new(policy);
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 100));
        archive.insert(make_job("j2", ArchivedStatus::Succeeded, 900));
        archive.apply_retention(1000);
        assert_eq!(archive.len(), 1);
        assert_eq!(archive.jobs[0].job_id, "j2");
    }

    #[test]
    fn test_archive_recent() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.insert(make_job("j2", ArchivedStatus::Succeeded, 2000));
        archive.insert(make_job("j3", ArchivedStatus::Succeeded, 3000));
        let recent = archive.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].job_id, "j2");
        assert_eq!(recent[1].job_id, "j3");
    }

    #[test]
    fn test_archive_clear() {
        let mut archive = JobArchive::default();
        archive.insert(make_job("j1", ArchivedStatus::Succeeded, 1000));
        archive.clear();
        assert!(archive.is_empty());
    }
}
