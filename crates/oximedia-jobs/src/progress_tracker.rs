// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Fine-grained job progress tracking with ETA estimation.
//!
//! Tracks per-job progress updates and estimates remaining time using
//! linear regression on historical update timestamps versus step counts.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// ProgressUpdate
// ---------------------------------------------------------------------------

/// A single progress update event for a job.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// The job this update belongs to.
    pub job_id: String,
    /// Current step number (0-indexed).
    pub step: u32,
    /// Total number of steps.
    pub total_steps: u32,
    /// Human-readable message describing what is happening.
    pub message: String,
    /// Unix timestamp (seconds) when the update was recorded.
    pub timestamp_secs: u64,
}

// ---------------------------------------------------------------------------
// ProgressTracker
// ---------------------------------------------------------------------------

/// Per-job tracker that maintains a history of progress updates and computes
/// ETA using ordinary least-squares linear regression.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    /// Job identifier.
    pub job_id: String,
    /// History of updates in chronological order.
    updates: Vec<ProgressUpdate>,
}

impl ProgressTracker {
    /// Create a new tracker for `job_id`.
    #[must_use]
    pub fn new(job_id: impl Into<String>) -> Self {
        Self {
            job_id: job_id.into(),
            updates: Vec::new(),
        }
    }

    /// Record a progress update.
    ///
    /// `timestamp_secs` defaults to the current Unix time when `None`.
    pub fn update(
        &mut self,
        step: u32,
        total_steps: u32,
        message: impl Into<String>,
        timestamp_secs: Option<u64>,
    ) {
        let ts = timestamp_secs.unwrap_or_else(current_unix_secs);
        self.updates.push(ProgressUpdate {
            job_id: self.job_id.clone(),
            step,
            total_steps,
            message: message.into(),
            timestamp_secs: ts,
        });
    }

    /// Completion fraction in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` when there are no updates, or when `total_steps` is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn completion_fraction(&self) -> f32 {
        let last = match self.updates.last() {
            Some(u) => u,
            None => return 0.0,
        };
        if last.total_steps == 0 {
            return 0.0;
        }
        (last.step as f64 / last.total_steps as f64).min(1.0) as f32
    }

    /// Estimated remaining seconds based on linear regression of update history.
    ///
    /// Requires at least 2 distinct updates to produce a meaningful estimate.
    /// Returns `None` when insufficient data is available or the job is complete.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn estimated_remaining_secs(&self) -> Option<u64> {
        if self.updates.len() < 2 {
            return None;
        }

        let last = self.updates.last()?;
        if last.total_steps == 0 {
            return None;
        }
        // Already done — no remaining time.
        if last.step >= last.total_steps {
            return Some(0);
        }

        // Build (x = step, y = timestamp_secs) pairs for linear regression.
        // We want to predict the timestamp when step == total_steps.
        let n = self.updates.len() as f64;
        let sum_x: f64 = self.updates.iter().map(|u| u.step as f64).sum();
        let sum_y: f64 = self.updates.iter().map(|u| u.timestamp_secs as f64).sum();
        let sum_xx: f64 = self.updates.iter().map(|u| (u.step as f64).powi(2)).sum();
        let sum_xy: f64 = self
            .updates
            .iter()
            .map(|u| u.step as f64 * u.timestamp_secs as f64)
            .sum();

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < f64::EPSILON {
            // All steps are identical — cannot estimate slope.
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        let target = last.total_steps as f64;
        let predicted_finish = slope * target + intercept;
        let now = last.timestamp_secs as f64;

        if predicted_finish <= now {
            return Some(0);
        }

        let remaining = predicted_finish - now;
        Some(remaining.ceil() as u64)
    }

    /// Returns a snapshot of the latest update, if any.
    #[must_use]
    pub fn latest_update(&self) -> Option<&ProgressUpdate> {
        self.updates.last()
    }

    /// Number of updates recorded.
    #[must_use]
    pub fn update_count(&self) -> usize {
        self.updates.len()
    }

    /// Whether the job is considered complete (step == total_steps and total > 0).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        match self.updates.last() {
            Some(u) if u.total_steps > 0 => u.step >= u.total_steps,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ProgressSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a job's progress.
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    /// Job identifier.
    pub job_id: String,
    /// Completion fraction in `[0.0, 1.0]`.
    pub fraction: f32,
    /// Estimated remaining seconds (None when insufficient history).
    pub eta_secs: Option<u64>,
    /// The message from the most recent update.
    pub last_message: String,
}

// ---------------------------------------------------------------------------
// ProgressRegistry
// ---------------------------------------------------------------------------

/// Central registry managing progress trackers for many jobs simultaneously.
#[derive(Debug, Default)]
pub struct ProgressRegistry {
    trackers: HashMap<String, ProgressTracker>,
}

impl ProgressRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new job.  No-op if the job is already registered.
    pub fn register(&mut self, job_id: impl Into<String>) {
        let id = job_id.into();
        self.trackers
            .entry(id.clone())
            .or_insert_with(|| ProgressTracker::new(id));
    }

    /// Record a progress update for a job.
    ///
    /// The job must have been registered first; returns an error string otherwise.
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message if the job is not registered.
    pub fn update(
        &mut self,
        job_id: &str,
        step: u32,
        total_steps: u32,
        message: impl Into<String>,
    ) -> Result<(), String> {
        let tracker = self
            .trackers
            .get_mut(job_id)
            .ok_or_else(|| format!("job '{job_id}' is not registered"))?;
        tracker.update(step, total_steps, message, None);
        Ok(())
    }

    /// Record a progress update with an explicit timestamp (useful for tests).
    ///
    /// # Errors
    ///
    /// Returns `Err` with a message if the job is not registered.
    pub fn update_with_timestamp(
        &mut self,
        job_id: &str,
        step: u32,
        total_steps: u32,
        message: impl Into<String>,
        timestamp_secs: u64,
    ) -> Result<(), String> {
        let tracker = self
            .trackers
            .get_mut(job_id)
            .ok_or_else(|| format!("job '{job_id}' is not registered"))?;
        tracker.update(step, total_steps, message, Some(timestamp_secs));
        Ok(())
    }

    /// Get the current snapshot for a job.  Returns `None` if not registered
    /// or if no updates have been recorded yet.
    #[must_use]
    pub fn snapshot(&self, job_id: &str) -> Option<ProgressSnapshot> {
        let tracker = self.trackers.get(job_id)?;
        let last = tracker.latest_update()?;
        Some(ProgressSnapshot {
            job_id: job_id.to_string(),
            fraction: tracker.completion_fraction(),
            eta_secs: tracker.estimated_remaining_secs(),
            last_message: last.message.clone(),
        })
    }

    /// Returns a list of job IDs that have reached 100% completion.
    #[must_use]
    pub fn completed_jobs(&self) -> Vec<String> {
        self.trackers
            .values()
            .filter(|t| t.is_complete())
            .map(|t| t.job_id.clone())
            .collect()
    }

    /// Returns the number of registered jobs.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.trackers.len()
    }

    /// Remove a job from the registry.
    pub fn deregister(&mut self, job_id: &str) {
        self.trackers.remove(job_id);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ProgressTracker ---------------------------------------------------

    #[test]
    fn test_tracker_no_updates_fraction_zero() {
        let tracker = ProgressTracker::new("job-1");
        assert_eq!(tracker.completion_fraction(), 0.0);
    }

    #[test]
    fn test_tracker_no_updates_no_eta() {
        let tracker = ProgressTracker::new("job-1");
        assert!(tracker.estimated_remaining_secs().is_none());
    }

    #[test]
    fn test_tracker_single_update_fraction() {
        let mut tracker = ProgressTracker::new("job-2");
        tracker.update(5, 10, "halfway", Some(1000));
        assert!((tracker.completion_fraction() - 0.5_f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_tracker_single_update_no_eta() {
        // Need ≥2 updates for ETA.
        let mut tracker = ProgressTracker::new("job-3");
        tracker.update(5, 10, "halfway", Some(1000));
        assert!(tracker.estimated_remaining_secs().is_none());
    }

    #[test]
    fn test_tracker_linear_progress_eta_accurate() {
        // Steps 0..10, each step takes 10 seconds.
        // At step 5 (t=50) the estimate for remaining 5 steps is ~50 s.
        let mut tracker = ProgressTracker::new("job-4");
        for i in 0u32..=5 {
            tracker.update(i, 10, "step", Some(u64::from(i) * 10));
        }
        let eta = tracker.estimated_remaining_secs().expect("should have ETA");
        // Expected: ~50 s remaining (steps 6-10, 10 s each).
        // Allow ±5 s tolerance for rounding.
        assert!(
            eta >= 45 && eta <= 55,
            "expected ~50, got {eta}"
        );
    }

    #[test]
    fn test_tracker_completion_at_100_percent() {
        let mut tracker = ProgressTracker::new("job-5");
        for i in 0u32..=10 {
            tracker.update(i, 10, "step", Some(u64::from(i) * 5));
        }
        assert!(tracker.is_complete());
        assert_eq!(tracker.completion_fraction(), 1.0_f32);
        // ETA should be 0 when already complete.
        assert_eq!(tracker.estimated_remaining_secs(), Some(0));
    }

    #[test]
    fn test_tracker_total_steps_zero_fraction() {
        let mut tracker = ProgressTracker::new("job-6");
        tracker.update(0, 0, "bad", Some(1000));
        assert_eq!(tracker.completion_fraction(), 0.0);
    }

    #[test]
    fn test_tracker_update_count() {
        let mut tracker = ProgressTracker::new("job-7");
        tracker.update(1, 5, "a", Some(100));
        tracker.update(2, 5, "b", Some(200));
        assert_eq!(tracker.update_count(), 2);
    }

    // ---- ProgressRegistry --------------------------------------------------

    #[test]
    fn test_registry_register_and_snapshot_no_updates() {
        let mut reg = ProgressRegistry::new();
        reg.register("r-job-1");
        // No updates yet, snapshot returns None.
        assert!(reg.snapshot("r-job-1").is_none());
    }

    #[test]
    fn test_registry_update_unregistered_returns_error() {
        let mut reg = ProgressRegistry::new();
        let result = reg.update("ghost-job", 1, 10, "msg");
        assert!(result.is_err());
    }

    #[test]
    fn test_registry_multi_job_independent() {
        let mut reg = ProgressRegistry::new();
        reg.register("job-a");
        reg.register("job-b");

        reg.update_with_timestamp("job-a", 3, 10, "step", 1000).expect("ok");
        reg.update_with_timestamp("job-a", 6, 10, "step", 2000).expect("ok");
        reg.update_with_timestamp("job-b", 1, 5, "step", 500).expect("ok");

        let snap_a = reg.snapshot("job-a").expect("should have snapshot");
        let snap_b = reg.snapshot("job-b").expect("should have snapshot");

        // job-a is 60% done, job-b is 20% done.
        assert!((snap_a.fraction - 0.6_f32).abs() < 1e-5);
        assert!((snap_b.fraction - 0.2_f32).abs() < 1e-5);
    }

    #[test]
    fn test_registry_completed_jobs() {
        let mut reg = ProgressRegistry::new();
        reg.register("complete-job");
        reg.register("incomplete-job");

        reg.update_with_timestamp("complete-job", 10, 10, "done", 1000).expect("ok");
        reg.update_with_timestamp("incomplete-job", 5, 10, "half", 500).expect("ok");

        let completed = reg.completed_jobs();
        assert_eq!(completed.len(), 1);
        assert!(completed.contains(&"complete-job".to_string()));
    }

    #[test]
    fn test_registry_snapshot_message() {
        let mut reg = ProgressRegistry::new();
        reg.register("msg-job");
        reg.update_with_timestamp("msg-job", 1, 10, "hello world", 100).expect("ok");
        let snap = reg.snapshot("msg-job").expect("should exist");
        assert_eq!(snap.last_message, "hello world");
    }
}
