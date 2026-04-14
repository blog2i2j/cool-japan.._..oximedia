#![allow(dead_code)]
//! Job migration — move jobs between queue instances.
//!
//! This module provides serialisation/deserialisation helpers for exporting
//! jobs from one queue and importing them into another, along with a
//! `MigrationPlan` that records what was moved and why.

use crate::job::{Job, JobStatus, Priority};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Error type for migration operations.
#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    /// Serialisation failure.
    #[error("Serialise error: {0}")]
    Serialise(String),
    /// Deserialisation failure.
    #[error("Deserialise error: {0}")]
    Deserialise(String),
    /// A job that was expected to exist was not found.
    #[error("Job not found: {0}")]
    JobNotFound(Uuid),
    /// The target queue rejected the job.
    #[error("Import rejected: {0}")]
    ImportRejected(String),
}

/// A job export record — a snapshot of a job at export time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedJob {
    /// The serialised job.
    pub job_json: String,
    /// Source queue identifier.
    pub source_queue: String,
    /// When the job was exported (RFC 3339).
    pub exported_at: String,
    /// Migration reason.
    pub reason: String,
}

impl ExportedJob {
    /// Create an exported job record.
    pub fn new(
        job: &Job,
        source_queue: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<Self, MigrationError> {
        let job_json = serde_json::to_string(job)
            .map_err(|e| MigrationError::Serialise(e.to_string()))?;
        Ok(Self {
            job_json,
            source_queue: source_queue.into(),
            exported_at: chrono::Utc::now().to_rfc3339(),
            reason: reason.into(),
        })
    }

    /// Deserialise back to a `Job`.
    pub fn to_job(&self) -> Result<Job, MigrationError> {
        serde_json::from_str(&self.job_json)
            .map_err(|e| MigrationError::Deserialise(e.to_string()))
    }
}

/// Filter controlling which jobs are exported.
#[derive(Debug, Clone, Default)]
pub struct MigrationFilter {
    /// Only export jobs with at least this priority.
    pub min_priority: Option<Priority>,
    /// Only export jobs with this status.
    pub status_filter: Option<JobStatus>,
    /// Maximum number of jobs to export.
    pub limit: Option<usize>,
}

impl MigrationFilter {
    /// Create a filter that accepts all jobs.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Set a minimum priority level.
    pub fn with_min_priority(mut self, priority: Priority) -> Self {
        self.min_priority = Some(priority);
        self
    }

    /// Set a status filter.
    pub fn with_status(mut self, status: JobStatus) -> Self {
        self.status_filter = Some(status);
        self
    }

    /// Set an export count limit.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Test whether a job passes this filter.
    #[must_use]
    pub fn accepts(&self, job: &Job) -> bool {
        if let Some(min_pri) = self.min_priority {
            if (job.priority as u8) < (min_pri as u8) {
                return false;
            }
        }
        if let Some(ref required_status) = self.status_filter {
            if &job.status != required_status {
                return false;
            }
        }
        true
    }
}

/// Summary of a completed migration operation.
#[derive(Debug, Clone, Default)]
pub struct MigrationReport {
    /// Number of jobs exported.
    pub exported: usize,
    /// Number of jobs successfully imported.
    pub imported: usize,
    /// Number of jobs that failed to import.
    pub failed: usize,
    /// List of job IDs that were migrated.
    pub migrated_ids: Vec<Uuid>,
    /// Error messages for failed jobs.
    pub errors: Vec<String>,
}

impl MigrationReport {
    /// Returns `true` if all exported jobs were imported successfully.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.failed == 0 && self.exported == self.imported
    }
}

/// An in-memory job queue used as a source or target for migrations.
#[derive(Debug, Default)]
pub struct MigratableQueue {
    /// Label for this queue instance.
    pub name: String,
    jobs: Vec<Job>,
}

impl MigratableQueue {
    /// Create a named queue.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            jobs: Vec::new(),
        }
    }

    /// Add a job to the queue.
    pub fn push(&mut self, job: Job) {
        self.jobs.push(job);
    }

    /// Number of jobs in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Export jobs matching the filter, returning `ExportedJob` records.
    pub fn export(
        &self,
        filter: &MigrationFilter,
        reason: &str,
    ) -> Result<Vec<ExportedJob>, MigrationError> {
        let mut exported = Vec::new();
        for job in self.jobs.iter() {
            if !filter.accepts(job) {
                continue;
            }
            if let Some(limit) = filter.limit {
                if exported.len() >= limit {
                    break;
                }
            }
            exported.push(ExportedJob::new(job, &self.name, reason)?);
        }
        Ok(exported)
    }

    /// Import a list of exported jobs.
    ///
    /// Returns a report describing what was imported.
    pub fn import(&mut self, jobs: Vec<ExportedJob>) -> MigrationReport {
        let mut report = MigrationReport::default();
        report.exported = jobs.len();
        for exported in jobs {
            match exported.to_job() {
                Ok(mut job) => {
                    // Reset status to Pending on import
                    job.status = JobStatus::Pending;
                    report.migrated_ids.push(job.id);
                    self.jobs.push(job);
                    report.imported += 1;
                }
                Err(e) => {
                    report.failed += 1;
                    report.errors.push(e.to_string());
                }
            }
        }
        report
    }

    /// Drain all matching jobs, returning them for transfer to another queue.
    pub fn drain_matching(&mut self, filter: &MigrationFilter) -> Vec<Job> {
        let mut drained = Vec::new();
        self.jobs.retain(|j| {
            if filter.accepts(j) {
                drained.push(j.clone());
                false
            } else {
                true
            }
        });
        drained
    }
}

/// High-level orchestrator that transfers jobs from `source` to `target`.
pub struct JobMigrator;

impl JobMigrator {
    /// Migrate matching jobs from `source` to `target`.
    ///
    /// Jobs are exported from `source`, imported into `target`, and removed
    /// from `source` on successful import.  Returns a `MigrationReport`.
    pub fn migrate(
        source: &mut MigratableQueue,
        target: &mut MigratableQueue,
        filter: &MigrationFilter,
        reason: &str,
    ) -> Result<MigrationReport, MigrationError> {
        let exported = source.export(filter, reason)?;
        let mut report = target.import(exported);
        // Remove successfully migrated jobs from source
        let migrated_ids: std::collections::HashSet<Uuid> =
            report.migrated_ids.iter().cloned().collect();
        source.jobs.retain(|j| !migrated_ids.contains(&j.id));
        report.exported = migrated_ids.len() + report.failed;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{AnalysisParams, JobPayload};

    fn make_job(name: &str, priority: Priority) -> Job {
        use crate::job::AnalysisType;
        Job::new(
            name.to_string(),
            priority,
            JobPayload::Analysis(AnalysisParams {
                input: "file.mp4".to_string(),
                analysis_type: AnalysisType::Quality,
                output: None,
            }),
        )
    }

    #[test]
    fn test_migration_filter_all_accepts_any_job() {
        let filter = MigrationFilter::all();
        let job = make_job("j1", Priority::Normal);
        assert!(filter.accepts(&job));
    }

    #[test]
    fn test_migration_filter_min_priority() {
        let filter = MigrationFilter::all().with_min_priority(Priority::High);
        let normal = make_job("j1", Priority::Normal);
        let high = make_job("j2", Priority::High);
        assert!(!filter.accepts(&normal));
        assert!(filter.accepts(&high));
    }

    #[test]
    fn test_migration_filter_status() {
        let filter = MigrationFilter::all().with_status(JobStatus::Pending);
        let mut running = make_job("j1", Priority::Normal);
        running.status = JobStatus::Running;
        let pending = make_job("j2", Priority::Normal);
        assert!(!filter.accepts(&running));
        assert!(filter.accepts(&pending));
    }

    #[test]
    fn test_export_jobs() {
        let mut q = MigratableQueue::new("source");
        q.push(make_job("j1", Priority::Normal));
        q.push(make_job("j2", Priority::High));
        let exported = q
            .export(&MigrationFilter::all(), "test migration")
            .expect("export should succeed");
        assert_eq!(exported.len(), 2);
        assert_eq!(exported[0].source_queue, "source");
    }

    #[test]
    fn test_export_with_limit() {
        let mut q = MigratableQueue::new("source");
        for i in 0..10 {
            q.push(make_job(&format!("j{i}"), Priority::Normal));
        }
        let filter = MigrationFilter::all().with_limit(3);
        let exported = q.export(&filter, "limited").expect("export should succeed");
        assert_eq!(exported.len(), 3);
    }

    #[test]
    fn test_import_resets_status_to_pending() {
        let mut source = MigratableQueue::new("source");
        let mut job = make_job("j1", Priority::Normal);
        job.status = JobStatus::Failed;
        source.push(job);
        let exported = source
            .export(&MigrationFilter::all(), "re-import")
            .expect("export");
        let mut target = MigratableQueue::new("target");
        let report = target.import(exported);
        assert_eq!(report.imported, 1);
        assert_eq!(target.jobs[0].status, JobStatus::Pending);
    }

    #[test]
    fn test_migrator_transfers_and_removes_from_source() {
        let mut source = MigratableQueue::new("source");
        let mut target = MigratableQueue::new("target");
        source.push(make_job("j1", Priority::Normal));
        source.push(make_job("j2", Priority::Normal));
        let report = JobMigrator::migrate(
            &mut source,
            &mut target,
            &MigrationFilter::all(),
            "full migration",
        )
        .expect("migrate should succeed");
        assert_eq!(report.imported, 2);
        assert_eq!(target.len(), 2);
        assert_eq!(source.len(), 0);
    }

    #[test]
    fn test_migrator_partial_migration() {
        let mut source = MigratableQueue::new("source");
        let mut target = MigratableQueue::new("target");
        source.push(make_job("j1", Priority::High));
        source.push(make_job("j2", Priority::Normal));
        let filter = MigrationFilter::all().with_min_priority(Priority::High);
        let report = JobMigrator::migrate(&mut source, &mut target, &filter, "partial")
            .expect("migrate should succeed");
        assert_eq!(report.imported, 1);
        assert_eq!(target.len(), 1);
        // j2 (Normal priority) should remain in source
        assert_eq!(source.len(), 1);
    }

    #[test]
    fn test_exported_job_roundtrip() {
        let job = make_job("roundtrip", Priority::High);
        let exported = ExportedJob::new(&job, "q1", "test").expect("export");
        let restored = exported.to_job().expect("to_job");
        assert_eq!(restored.id, job.id);
        assert_eq!(restored.name, job.name);
    }

    #[test]
    fn test_migration_report_is_complete() {
        let mut report = MigrationReport::default();
        report.exported = 3;
        report.imported = 3;
        assert!(report.is_complete());
        report.failed = 1;
        assert!(!report.is_complete());
    }
}
