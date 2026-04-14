// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Progressive job status tracking with ETA estimation.
//!
//! Provides per-job progress records, an ETA estimator, and a [`JobTracker`]
//! that aggregates progress across all active jobs.
//!
//! # Example
//!
//! ```rust
//! use oximedia_renderfarm::job_tracker::{JobTracker, estimate_eta};
//!
//! let mut tracker = JobTracker::new();
//! tracker.update_progress("job-1", 50.0, "encoding", 10_000);
//!
//! let eta = estimate_eta(50.0, 10_000);
//! assert_eq!(eta, Some(10_000)); // 50% done → same elapsed remaining
//!
//! let active = tracker.all_active();
//! assert_eq!(active.len(), 1);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// JobProgress
// ---------------------------------------------------------------------------

/// Snapshot of a single job's render progress.
#[derive(Debug, Clone)]
pub struct JobProgress {
    /// Unique identifier of the job.
    pub job_id: String,
    /// Completion percentage in the range `[0.0, 100.0]`.
    pub percent: f32,
    /// Wall-clock milliseconds elapsed since the job started.
    pub elapsed_ms: u64,
    /// Estimated remaining milliseconds, or `None` when there is not yet
    /// enough information to produce a reliable estimate (< 5% complete).
    pub eta_ms: Option<u64>,
    /// Human-readable description of what the job is currently doing.
    pub current_step: String,
}

impl JobProgress {
    /// Construct a new `JobProgress` record.
    ///
    /// `eta_ms` is derived automatically from `percent` and `elapsed_ms` via
    /// [`estimate_eta`].
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        percent: f32,
        step: impl Into<String>,
        elapsed_ms: u64,
    ) -> Self {
        let eta_ms = estimate_eta(percent, elapsed_ms);
        Self {
            job_id: job_id.into(),
            percent,
            elapsed_ms,
            eta_ms,
            current_step: step.into(),
        }
    }

    /// Returns `true` when the job is considered complete (≥ 100 %).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.percent >= 100.0
    }
}

// ---------------------------------------------------------------------------
// estimate_eta
// ---------------------------------------------------------------------------

/// Estimate the remaining time (in milliseconds) for a job.
///
/// Uses simple linear extrapolation:
/// ```text
/// eta_ms = elapsed_ms × (100 − percent) / percent
/// ```
///
/// Returns `None` when `percent` is below 5 % because there is too little
/// signal for a reliable estimate, or when `percent` is ≥ 100 (no remaining
/// work).
///
/// # Examples
///
/// ```
/// use oximedia_renderfarm::job_tracker::estimate_eta;
///
/// // 50% complete after 10 000 ms → ~10 000 ms remaining
/// assert_eq!(estimate_eta(50.0, 10_000), Some(10_000));
///
/// // Too early to estimate
/// assert_eq!(estimate_eta(3.0, 1_000), None);
///
/// // Complete
/// assert_eq!(estimate_eta(100.0, 5_000), None);
/// ```
#[must_use]
pub fn estimate_eta(percent: f32, elapsed_ms: u64) -> Option<u64> {
    if percent < 5.0 || percent >= 100.0 {
        return None;
    }
    // Cast to f64 for precision.
    let remaining = (elapsed_ms as f64) * (100.0 - percent as f64) / (percent as f64);
    Some(remaining as u64)
}

// ---------------------------------------------------------------------------
// JobTracker
// ---------------------------------------------------------------------------

/// In-memory registry that tracks progress for all active jobs.
///
/// Jobs that reach 100% completion remain in the map until explicitly
/// removed via [`remove`](Self::remove) so callers can inspect the final
/// state.
pub struct JobTracker {
    jobs: HashMap<String, JobProgress>,
}

impl JobTracker {
    /// Create an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
        }
    }

    /// Insert or update the progress record for `job_id`.
    ///
    /// `now_ms` is the current wall-clock time in milliseconds (caller is
    /// responsible for providing a monotonic source).  The ETA is
    /// recomputed automatically.
    pub fn update_progress(&mut self, job_id: &str, percent: f32, step: &str, now_ms: u64) {
        let entry = self
            .jobs
            .entry(job_id.to_owned())
            .or_insert_with(|| JobProgress {
                job_id: job_id.to_owned(),
                percent: 0.0,
                elapsed_ms: 0,
                eta_ms: None,
                current_step: String::new(),
            });

        entry.percent = percent.clamp(0.0, 100.0);
        entry.elapsed_ms = now_ms;
        entry.current_step = step.to_owned();
        entry.eta_ms = estimate_eta(entry.percent, entry.elapsed_ms);
    }

    /// Return references to all tracked job progress records, including
    /// completed ones.
    #[must_use]
    pub fn all_active(&self) -> Vec<&JobProgress> {
        self.jobs.values().collect()
    }

    /// Return references to jobs that are not yet complete (< 100 %).
    #[must_use]
    pub fn in_progress(&self) -> Vec<&JobProgress> {
        self.jobs.values().filter(|p| !p.is_complete()).collect()
    }

    /// Look up a specific job by ID.
    #[must_use]
    pub fn get(&self, job_id: &str) -> Option<&JobProgress> {
        self.jobs.get(job_id)
    }

    /// Remove a job from the tracker (e.g. after archiving completion).
    ///
    /// Returns the removed record, or `None` if the job was not tracked.
    pub fn remove(&mut self, job_id: &str) -> Option<JobProgress> {
        self.jobs.remove(job_id)
    }

    /// Total number of jobs currently tracked (active + completed).
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` when no jobs are being tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

impl Default for JobTracker {
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

    // --- estimate_eta ---

    #[test]
    fn eta_below_threshold_returns_none() {
        assert_eq!(estimate_eta(4.9, 5_000), None);
    }

    #[test]
    fn eta_at_threshold_returns_some() {
        // 5% done after 1 000 ms → 19 000 ms remaining
        let eta = estimate_eta(5.0, 1_000);
        assert!(eta.is_some());
        // 1000 * 95 / 5 = 19 000
        assert_eq!(eta, Some(19_000));
    }

    #[test]
    fn eta_fifty_percent() {
        assert_eq!(estimate_eta(50.0, 10_000), Some(10_000));
    }

    #[test]
    fn eta_complete_returns_none() {
        assert_eq!(estimate_eta(100.0, 99_999), None);
    }

    #[test]
    fn eta_over_hundred_returns_none() {
        assert_eq!(estimate_eta(110.0, 1_000), None);
    }

    #[test]
    fn eta_zero_elapsed_gives_zero_remaining() {
        // 50% done, 0 ms elapsed → 0 ms remaining (degenerate but valid)
        assert_eq!(estimate_eta(50.0, 0), Some(0));
    }

    // --- JobProgress ---

    #[test]
    fn job_progress_new_sets_eta() {
        let p = JobProgress::new("job-a", 50.0, "encoding", 10_000);
        assert_eq!(p.percent, 50.0);
        assert_eq!(p.elapsed_ms, 10_000);
        assert_eq!(p.eta_ms, Some(10_000));
        assert_eq!(p.current_step, "encoding");
    }

    #[test]
    fn job_progress_complete_flag() {
        let p = JobProgress::new("job-b", 100.0, "done", 60_000);
        assert!(p.is_complete());
    }

    #[test]
    fn job_progress_not_complete_at_99() {
        let p = JobProgress::new("job-c", 99.9, "finalizing", 50_000);
        assert!(!p.is_complete());
    }

    // --- JobTracker ---

    #[test]
    fn tracker_starts_empty() {
        let t = JobTracker::new();
        assert!(t.is_empty());
    }

    #[test]
    fn tracker_update_adds_new_job() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 25.0, "step-1", 5_000);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn tracker_update_overwrites_existing() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 10.0, "step-1", 2_000);
        t.update_progress("j1", 60.0, "step-2", 12_000);
        assert_eq!(t.len(), 1);
        let p = t.get("j1").expect("should exist");
        assert_eq!(p.percent, 60.0);
        assert_eq!(p.current_step, "step-2");
    }

    #[test]
    fn tracker_all_active_returns_all() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 30.0, "a", 3_000);
        t.update_progress("j2", 100.0, "done", 10_000);
        assert_eq!(t.all_active().len(), 2);
    }

    #[test]
    fn tracker_in_progress_excludes_completed() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 30.0, "a", 3_000);
        t.update_progress("j2", 100.0, "done", 10_000);
        assert_eq!(t.in_progress().len(), 1);
    }

    #[test]
    fn tracker_remove_returns_record() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 50.0, "working", 5_000);
        let removed = t.remove("j1");
        assert!(removed.is_some());
        assert!(t.is_empty());
    }

    #[test]
    fn tracker_remove_missing_returns_none() {
        let mut t = JobTracker::new();
        assert!(t.remove("ghost").is_none());
    }

    #[test]
    fn tracker_percent_clamped_above_hundred() {
        let mut t = JobTracker::new();
        t.update_progress("j1", 120.0, "overdone", 1_000);
        let p = t.get("j1").expect("exists");
        assert_eq!(p.percent, 100.0);
    }
}
