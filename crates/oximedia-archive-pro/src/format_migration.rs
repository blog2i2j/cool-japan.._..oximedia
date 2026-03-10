//! Format migration planning and tracking for long-term digital preservation.
//!
//! When a stored media format becomes obsolete or at risk, a migration job must
//! be planned, approved, executed, and validated.  This module provides the
//! data model and planning logic for that workflow, without performing actual
//! file I/O.

#![allow(dead_code)]

use std::path::PathBuf;

/// Life-cycle status of a single format migration job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationStatus {
    /// The job has been created but not yet assessed.
    Pending,
    /// Risk and feasibility have been assessed; awaiting approval.
    Assessed,
    /// Approved and ready to execute.
    Approved,
    /// Currently executing.
    Running,
    /// Migration completed and validation passed.
    Completed,
    /// Validation failed; original asset is intact.
    Failed,
    /// Job was cancelled before completion.
    Cancelled,
}

impl std::fmt::Display for MigrationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pending => "Pending",
            Self::Assessed => "Assessed",
            Self::Approved => "Approved",
            Self::Running => "Running",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        };
        write!(f, "{s}")
    }
}

impl MigrationStatus {
    /// Returns `true` if the job can still transition to [`MigrationStatus::Running`].
    #[must_use]
    pub fn is_actionable(self) -> bool {
        matches!(self, Self::Pending | Self::Assessed | Self::Approved)
    }

    /// Returns `true` if the job is in a terminal state.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

// ---------------------------------------------------------------------------
// MigrationJob
// ---------------------------------------------------------------------------

/// A single format migration job for one source asset.
#[derive(Debug, Clone)]
pub struct MigrationJob {
    /// Unique job identifier.
    pub id: u64,
    /// Path to the source asset.
    pub source: PathBuf,
    /// Source format identifier (e.g. `"video/x-msvideo"`).
    pub source_format: String,
    /// Target preservation format identifier (e.g. `"video/x-matroska"`).
    pub target_format: String,
    /// Desired output path for the migrated asset.
    pub target: PathBuf,
    /// Current life-cycle status.
    pub status: MigrationStatus,
    /// Estimated migration duration in seconds.
    pub estimated_duration_secs: u64,
    /// Notes from the assessor / validator.
    pub notes: Vec<String>,
}

impl MigrationJob {
    /// Create a new pending migration job.
    #[must_use]
    pub fn new(
        id: u64,
        source: PathBuf,
        source_format: impl Into<String>,
        target_format: impl Into<String>,
        target: PathBuf,
    ) -> Self {
        Self {
            id,
            source,
            source_format: source_format.into(),
            target_format: target_format.into(),
            target,
            status: MigrationStatus::Pending,
            estimated_duration_secs: 0,
            notes: Vec::new(),
        }
    }

    /// Add a note to this job.
    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Transition status, returning an error if the transition is invalid.
    pub fn transition(&mut self, next: MigrationStatus) -> Result<(), String> {
        let valid = matches!(
            (&self.status, &next),
            (MigrationStatus::Pending, MigrationStatus::Assessed)
                | (MigrationStatus::Assessed, MigrationStatus::Approved)
                | (MigrationStatus::Approved, MigrationStatus::Running)
                | (MigrationStatus::Running, MigrationStatus::Completed)
                | (MigrationStatus::Running, MigrationStatus::Failed)
                | (_, MigrationStatus::Cancelled)
        );
        if valid {
            self.status = next;
            Ok(())
        } else {
            Err(format!(
                "Invalid transition: {:?} → {:?}",
                self.status, next
            ))
        }
    }
}

// ---------------------------------------------------------------------------
// FormatMigrator
// ---------------------------------------------------------------------------

/// Planner for format migration campaigns across a media collection.
///
/// Tracks all jobs, computes summary statistics, and enforces business rules
/// (e.g. no duplicate jobs for the same source path).
pub struct FormatMigrator {
    jobs: Vec<MigrationJob>,
    next_id: u64,
}

impl FormatMigrator {
    /// Create a new, empty migrator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 1,
        }
    }

    /// Plan a new migration job, returning the assigned job ID.
    ///
    /// Returns an error if a job for `source` already exists and is not in a
    /// terminal state.
    pub fn plan(
        &mut self,
        source: PathBuf,
        source_format: impl Into<String>,
        target_format: impl Into<String>,
        target: PathBuf,
    ) -> Result<u64, String> {
        // Guard: no duplicate active jobs for the same source path.
        if let Some(existing) = self.jobs.iter().find(|j| j.source == source) {
            if !existing.status.is_terminal() {
                return Err(format!(
                    "Active migration job {} already exists for {:?}",
                    existing.id, source
                ));
            }
        }

        let id = self.next_id;
        self.next_id += 1;
        self.jobs.push(MigrationJob::new(
            id,
            source,
            source_format,
            target_format,
            target,
        ));
        Ok(id)
    }

    /// Return a reference to a job by ID.
    #[must_use]
    pub fn job(&self, id: u64) -> Option<&MigrationJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Return a mutable reference to a job by ID.
    pub fn job_mut(&mut self, id: u64) -> Option<&mut MigrationJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Count jobs in each status bucket.
    #[must_use]
    pub fn status_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut map = std::collections::HashMap::new();
        for job in &self.jobs {
            *map.entry(job.status.to_string()).or_insert(0) += 1;
        }
        map
    }

    /// Return all jobs with the given status.
    #[must_use]
    pub fn jobs_by_status(&self, status: MigrationStatus) -> Vec<&MigrationJob> {
        self.jobs.iter().filter(|j| j.status == status).collect()
    }

    /// Total number of jobs tracked.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }
}

impl Default for FormatMigrator {
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

    fn make_job(id: u64) -> MigrationJob {
        MigrationJob::new(
            id,
            PathBuf::from(format!("/archive/source_{id}.avi")),
            "video/x-msvideo",
            "video/x-matroska",
            PathBuf::from(format!("/archive/target_{id}.mkv")),
        )
    }

    #[test]
    fn test_status_display() {
        assert_eq!(MigrationStatus::Pending.to_string(), "Pending");
        assert_eq!(MigrationStatus::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_status_is_actionable() {
        assert!(MigrationStatus::Pending.is_actionable());
        assert!(MigrationStatus::Approved.is_actionable());
        assert!(!MigrationStatus::Completed.is_actionable());
    }

    #[test]
    fn test_status_is_terminal() {
        assert!(MigrationStatus::Completed.is_terminal());
        assert!(MigrationStatus::Failed.is_terminal());
        assert!(MigrationStatus::Cancelled.is_terminal());
        assert!(!MigrationStatus::Running.is_terminal());
    }

    #[test]
    fn test_job_new() {
        let j = make_job(1);
        assert_eq!(j.id, 1);
        assert_eq!(j.status, MigrationStatus::Pending);
        assert!(j.notes.is_empty());
    }

    #[test]
    fn test_job_add_note() {
        let mut j = make_job(1);
        j.add_note("Risk assessed: low");
        assert_eq!(j.notes.len(), 1);
    }

    #[test]
    fn test_job_valid_transition() {
        let mut j = make_job(1);
        assert!(j.transition(MigrationStatus::Assessed).is_ok());
        assert_eq!(j.status, MigrationStatus::Assessed);
    }

    #[test]
    fn test_job_invalid_transition() {
        let mut j = make_job(1);
        // Pending → Running is not a valid direct step
        assert!(j.transition(MigrationStatus::Running).is_err());
    }

    #[test]
    fn test_job_cancel_any_state() {
        let mut j = make_job(1);
        // Can cancel from Pending
        assert!(j.transition(MigrationStatus::Cancelled).is_ok());
    }

    #[test]
    fn test_migrator_plan_returns_id() {
        let mut m = FormatMigrator::new();
        let id = m
            .plan(
                PathBuf::from("/src/a.avi"),
                "video/x-msvideo",
                "video/x-matroska",
                PathBuf::from("/dst/a.mkv"),
            )
            .expect("operation should succeed");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_migrator_plan_increments_id() {
        let mut m = FormatMigrator::new();
        let id1 = m
            .plan(
                PathBuf::from("/s1.avi"),
                "avi",
                "mkv",
                PathBuf::from("/d1.mkv"),
            )
            .expect("operation should succeed");
        let id2 = m
            .plan(
                PathBuf::from("/s2.avi"),
                "avi",
                "mkv",
                PathBuf::from("/d2.mkv"),
            )
            .expect("operation should succeed");
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_migrator_duplicate_active_job_rejected() {
        let mut m = FormatMigrator::new();
        let src = PathBuf::from("/same.avi");
        m.plan(src.clone(), "avi", "mkv", PathBuf::from("/d.mkv"))
            .expect("operation should succeed");
        let result = m.plan(src, "avi", "mkv", PathBuf::from("/d2.mkv"));
        assert!(result.is_err());
    }

    #[test]
    fn test_migrator_job_lookup() {
        let mut m = FormatMigrator::new();
        let id = m
            .plan(
                PathBuf::from("/a.avi"),
                "avi",
                "mkv",
                PathBuf::from("/a.mkv"),
            )
            .expect("operation should succeed");
        assert!(m.job(id).is_some());
        assert!(m.job(999).is_none());
    }

    #[test]
    fn test_migrator_status_counts() {
        let mut m = FormatMigrator::new();
        m.plan(
            PathBuf::from("/b1.avi"),
            "avi",
            "mkv",
            PathBuf::from("/b1.mkv"),
        )
        .expect("operation should succeed");
        m.plan(
            PathBuf::from("/b2.avi"),
            "avi",
            "mkv",
            PathBuf::from("/b2.mkv"),
        )
        .expect("operation should succeed");
        let counts = m.status_counts();
        assert_eq!(counts.get("Pending").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_migrator_jobs_by_status() {
        let mut m = FormatMigrator::new();
        let id = m
            .plan(
                PathBuf::from("/c.avi"),
                "avi",
                "mkv",
                PathBuf::from("/c.mkv"),
            )
            .expect("operation should succeed");
        {
            let j = m.job_mut(id).expect("operation should succeed");
            j.transition(MigrationStatus::Assessed)
                .expect("operation should succeed");
        }
        let assessed = m.jobs_by_status(MigrationStatus::Assessed);
        assert_eq!(assessed.len(), 1);
        let pending = m.jobs_by_status(MigrationStatus::Pending);
        assert!(pending.is_empty());
    }

    #[test]
    fn test_migrator_job_count() {
        let mut m = FormatMigrator::new();
        assert_eq!(m.job_count(), 0);
        m.plan(
            PathBuf::from("/x.avi"),
            "avi",
            "mkv",
            PathBuf::from("/x.mkv"),
        )
        .expect("operation should succeed");
        assert_eq!(m.job_count(), 1);
    }

    #[test]
    fn test_migrator_completed_job_allows_replan() {
        let mut m = FormatMigrator::new();
        let src = PathBuf::from("/rep.avi");
        let id = m
            .plan(src.clone(), "avi", "mkv", PathBuf::from("/rep1.mkv"))
            .expect("operation should succeed");
        // Advance to Completed via valid transitions
        let j = m.job_mut(id).expect("operation should succeed");
        j.transition(MigrationStatus::Assessed)
            .expect("operation should succeed");
        j.transition(MigrationStatus::Approved)
            .expect("operation should succeed");
        j.transition(MigrationStatus::Running)
            .expect("operation should succeed");
        j.transition(MigrationStatus::Completed)
            .expect("operation should succeed");
        // Now re-planning the same source should succeed
        let id2 = m
            .plan(src, "avi", "mkv", PathBuf::from("/rep2.mkv"))
            .expect("operation should succeed");
        assert_ne!(id, id2);
    }
}
