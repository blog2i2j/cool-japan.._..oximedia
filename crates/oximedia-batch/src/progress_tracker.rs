//! Progress tracking for batch jobs.
//!
//! Tracks per-job progress state and provides a monitor that aggregates
//! progress across multiple concurrent batch jobs.

#![allow(dead_code)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// State of a batch job's progress.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressState {
    /// Job has been registered but not yet started.
    Registered,
    /// Job is actively running.
    Running,
    /// Job completed successfully.
    Done,
    /// Job failed.
    Failed,
    /// Job was cancelled.
    Cancelled,
}

impl ProgressState {
    /// Returns `true` if the job is in an active (non-terminal) state.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Registered | Self::Running)
    }
}

/// Progress information for a single batch job.
#[derive(Debug)]
pub struct BatchProgress {
    /// Job identifier
    pub job_id: String,
    /// Current state
    pub state: ProgressState,
    /// Number of work units completed
    pub completed: u64,
    /// Total work units (0 = unknown)
    pub total: u64,
    /// Instant at which this progress record was created
    start: Instant,
    /// Instant of the last update
    last_update: Instant,
}

impl BatchProgress {
    /// Creates a new `BatchProgress` in `Registered` state.
    #[must_use]
    pub fn new(job_id: &str, total: u64) -> Self {
        let now = Instant::now();
        Self {
            job_id: job_id.to_owned(),
            state: ProgressState::Registered,
            completed: 0,
            total,
            start: now,
            last_update: now,
        }
    }

    /// Updates the completed count and transitions state to `Running`.
    pub fn update(&mut self, completed: u64) {
        self.completed = completed;
        self.last_update = Instant::now();
        if self.state == ProgressState::Registered {
            self.state = ProgressState::Running;
        }
    }

    /// Marks the job as done.
    pub fn finish(&mut self) {
        self.completed = self.total;
        self.state = ProgressState::Done;
        self.last_update = Instant::now();
    }

    /// Marks the job as failed.
    pub fn fail(&mut self) {
        self.state = ProgressState::Failed;
        self.last_update = Instant::now();
    }

    /// Returns the completion percentage (0.0–100.0).
    /// Returns `0.0` if total is 0.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn completion_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.completed as f64 / self.total as f64 * 100.0).min(100.0)
    }

    /// Returns elapsed milliseconds since this progress record was created.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Returns the duration since the last update.
    #[must_use]
    pub fn time_since_update(&self) -> Duration {
        self.last_update.elapsed()
    }
}

/// Monitor that tracks progress for multiple batch jobs.
#[derive(Debug, Default)]
pub struct BatchProgressMonitor {
    jobs: HashMap<String, BatchProgress>,
}

impl BatchProgressMonitor {
    /// Creates a new empty monitor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a new job with the given total work units.
    /// If a job with the same id already exists, it is replaced.
    pub fn register(&mut self, job_id: &str, total: u64) {
        self.jobs
            .insert(job_id.to_owned(), BatchProgress::new(job_id, total));
    }

    /// Returns a reference to the progress for the given job, or `None`.
    #[must_use]
    pub fn get(&self, job_id: &str) -> Option<&BatchProgress> {
        self.jobs.get(job_id)
    }

    /// Returns a mutable reference to the progress for the given job, or `None`.
    pub fn get_mut(&mut self, job_id: &str) -> Option<&mut BatchProgress> {
        self.jobs.get_mut(job_id)
    }

    /// Returns the number of jobs in a terminal (`Done`, `Failed`, `Cancelled`) state.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.jobs.values().filter(|p| !p.state.is_active()).count()
    }

    /// Returns the number of registered jobs.
    #[must_use]
    pub fn total_registered(&self) -> usize {
        self.jobs.len()
    }

    /// Returns the overall completion percentage across all jobs with known totals.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn overall_pct(&self) -> f64 {
        let known: Vec<&BatchProgress> = self.jobs.values().filter(|p| p.total > 0).collect();
        if known.is_empty() {
            return 0.0;
        }
        let sum: f64 = known.iter().map(|p| p.completion_pct()).sum();
        sum / known.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_state_registered_is_active() {
        assert!(ProgressState::Registered.is_active());
    }

    #[test]
    fn test_progress_state_running_is_active() {
        assert!(ProgressState::Running.is_active());
    }

    #[test]
    fn test_progress_state_done_not_active() {
        assert!(!ProgressState::Done.is_active());
    }

    #[test]
    fn test_progress_state_failed_not_active() {
        assert!(!ProgressState::Failed.is_active());
    }

    #[test]
    fn test_progress_state_cancelled_not_active() {
        assert!(!ProgressState::Cancelled.is_active());
    }

    #[test]
    fn test_batch_progress_initial_pct_zero() {
        let p = BatchProgress::new("j1", 100);
        assert!((p.completion_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_update_pct() {
        let mut p = BatchProgress::new("j1", 200);
        p.update(100);
        assert!((p.completion_pct() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_progress_update_transitions_to_running() {
        let mut p = BatchProgress::new("j1", 10);
        p.update(1);
        assert_eq!(p.state, ProgressState::Running);
    }

    #[test]
    fn test_batch_progress_finish() {
        let mut p = BatchProgress::new("j1", 10);
        p.finish();
        assert_eq!(p.state, ProgressState::Done);
        assert!((p.completion_pct() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_batch_progress_fail() {
        let mut p = BatchProgress::new("j1", 10);
        p.fail();
        assert_eq!(p.state, ProgressState::Failed);
    }

    #[test]
    fn test_batch_progress_zero_total() {
        let p = BatchProgress::new("j1", 0);
        assert!((p.completion_pct() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_elapsed_ms_non_negative() {
        let p = BatchProgress::new("j1", 50);
        assert!(p.elapsed_ms() < 1_000); // should complete in well under 1 s
    }

    #[test]
    fn test_monitor_register_and_get() {
        let mut mon = BatchProgressMonitor::new();
        mon.register("j1", 100);
        assert!(mon.get("j1").is_some());
    }

    #[test]
    fn test_monitor_get_missing() {
        let mon = BatchProgressMonitor::new();
        assert!(mon.get("nope").is_none());
    }

    #[test]
    fn test_monitor_completed_count() {
        let mut mon = BatchProgressMonitor::new();
        mon.register("j1", 10);
        mon.register("j2", 20);
        mon.get_mut("j2").expect("get_mut should succeed").finish();
        assert_eq!(mon.completed_count(), 1);
    }

    #[test]
    fn test_monitor_total_registered() {
        let mut mon = BatchProgressMonitor::new();
        mon.register("j1", 10);
        mon.register("j2", 20);
        assert_eq!(mon.total_registered(), 2);
    }

    #[test]
    fn test_monitor_overall_pct() {
        let mut mon = BatchProgressMonitor::new();
        mon.register("j1", 100);
        mon.register("j2", 100);
        mon.get_mut("j1")
            .expect("get_mut should succeed")
            .update(50); // 50%
        mon.get_mut("j2")
            .expect("get_mut should succeed")
            .update(100); // 100%
                          // mean = 75%
        assert!((mon.overall_pct() - 75.0).abs() < 1e-9);
    }
}
