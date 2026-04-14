// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Progress tracking for batch conversions.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Progress tracker for monitoring conversion progress.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    state: Arc<Mutex<ProgressState>>,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new(total_jobs: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(ProgressState {
                total_jobs,
                completed_jobs: 0,
                failed_jobs: 0,
                current_job: None,
                current_progress: 0.0,
                start_time: Instant::now(),
                bytes_processed: 0,
                total_bytes: 0,
            })),
        }
    }

    /// Update the current job being processed.
    pub fn set_current_job(&self, job_name: String) {
        if let Ok(mut state) = self.state.lock() {
            state.current_job = Some(job_name);
            state.current_progress = 0.0;
        }
    }

    /// Update the progress of the current job (0.0 to 1.0).
    pub fn update_progress(&self, progress: f64) {
        if let Ok(mut state) = self.state.lock() {
            state.current_progress = progress.clamp(0.0, 1.0);
        }
    }

    /// Mark a job as completed.
    pub fn complete_job(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.completed_jobs += 1;
            state.current_job = None;
            state.current_progress = 0.0;
        }
    }

    /// Mark a job as failed.
    pub fn fail_job(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.failed_jobs += 1;
            state.current_job = None;
            state.current_progress = 0.0;
        }
    }

    /// Update bytes processed.
    pub fn update_bytes(&self, processed: u64, total: u64) {
        if let Ok(mut state) = self.state.lock() {
            state.bytes_processed = processed;
            state.total_bytes = total;
        }
    }

    /// Get the current progress snapshot.
    #[must_use]
    pub fn snapshot(&self) -> Option<ProgressSnapshot> {
        self.state.lock().ok().map(|state| {
            let elapsed = state.start_time.elapsed();
            let completed = state.completed_jobs;
            let total = state.total_jobs;

            let overall_progress = if total > 0 {
                (completed as f64 + state.current_progress) / total as f64
            } else {
                0.0
            };

            let eta = if overall_progress > 0.0 {
                let total_time = elapsed.as_secs_f64() / overall_progress;
                let remaining = total_time - elapsed.as_secs_f64();
                Some(Duration::from_secs_f64(remaining.max(0.0)))
            } else {
                None
            };

            ProgressSnapshot {
                total_jobs: total,
                completed_jobs: completed,
                failed_jobs: state.failed_jobs,
                current_job: state.current_job.clone(),
                current_progress: state.current_progress,
                overall_progress,
                elapsed,
                eta,
                bytes_processed: state.bytes_processed,
                total_bytes: state.total_bytes,
            }
        })
    }

    /// Get the overall progress as a percentage (0-100).
    #[must_use]
    pub fn overall_progress_percent(&self) -> f64 {
        self.snapshot().map_or(0.0, |s| s.overall_progress * 100.0)
    }

    /// Check if all jobs are complete.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.completed_jobs + state.failed_jobs >= state.total_jobs)
            .unwrap_or(false)
    }

    /// Reset the progress tracker.
    pub fn reset(&self, total_jobs: usize) {
        if let Ok(mut state) = self.state.lock() {
            *state = ProgressState {
                total_jobs,
                completed_jobs: 0,
                failed_jobs: 0,
                current_job: None,
                current_progress: 0.0,
                start_time: Instant::now(),
                bytes_processed: 0,
                total_bytes: 0,
            };
        }
    }
}

impl Default for ProgressTracker {
    fn default() -> Self {
        Self::new(0)
    }
}

#[derive(Debug)]
struct ProgressState {
    total_jobs: usize,
    completed_jobs: usize,
    failed_jobs: usize,
    current_job: Option<String>,
    current_progress: f64,
    start_time: Instant,
    bytes_processed: u64,
    total_bytes: u64,
}

/// A snapshot of the current progress.
#[derive(Debug, Clone)]
pub struct ProgressSnapshot {
    /// Total number of jobs
    pub total_jobs: usize,
    /// Number of completed jobs
    pub completed_jobs: usize,
    /// Number of failed jobs
    pub failed_jobs: usize,
    /// Name of the current job
    pub current_job: Option<String>,
    /// Progress of the current job (0.0 to 1.0)
    pub current_progress: f64,
    /// Overall progress (0.0 to 1.0)
    pub overall_progress: f64,
    /// Elapsed time since start
    pub elapsed: Duration,
    /// Estimated time remaining
    pub eta: Option<Duration>,
    /// Bytes processed
    pub bytes_processed: u64,
    /// Total bytes to process
    pub total_bytes: u64,
}

impl ProgressSnapshot {
    /// Get the number of remaining jobs.
    #[must_use]
    pub fn remaining_jobs(&self) -> usize {
        self.total_jobs
            .saturating_sub(self.completed_jobs + self.failed_jobs)
    }

    /// Get the success rate as a percentage.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let finished = self.completed_jobs + self.failed_jobs;
        if finished == 0 {
            return 100.0;
        }
        (self.completed_jobs as f64 / finished as f64) * 100.0
    }

    /// Get the processing speed in jobs per second.
    #[must_use]
    pub fn jobs_per_second(&self) -> f64 {
        let elapsed_secs = self.elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            self.completed_jobs as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Get the data transfer rate in bytes per second.
    #[must_use]
    pub fn bytes_per_second(&self) -> f64 {
        let elapsed_secs = self.elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            self.bytes_processed as f64 / elapsed_secs
        } else {
            0.0
        }
    }

    /// Format the ETA as a human-readable string.
    #[must_use]
    pub fn eta_formatted(&self) -> String {
        match self.eta {
            Some(duration) => {
                let secs = duration.as_secs();
                let hours = secs / 3600;
                let minutes = (secs % 3600) / 60;
                let seconds = secs % 60;

                if hours > 0 {
                    format!("{hours}h {minutes}m {seconds}s")
                } else if minutes > 0 {
                    format!("{minutes}m {seconds}s")
                } else {
                    format!("{seconds}s")
                }
            }
            None => "Unknown".to_string(),
        }
    }

    /// Format the elapsed time as a human-readable string.
    #[must_use]
    pub fn elapsed_formatted(&self) -> String {
        let secs = self.elapsed.as_secs();
        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;

        if hours > 0 {
            format!("{hours}h {minutes}m {seconds}s")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_tracker_creation() {
        let tracker = ProgressTracker::new(10);
        let snapshot = tracker.snapshot().unwrap();

        assert_eq!(snapshot.total_jobs, 10);
        assert_eq!(snapshot.completed_jobs, 0);
        assert_eq!(snapshot.failed_jobs, 0);
    }

    #[test]
    fn test_progress_updates() {
        let tracker = ProgressTracker::new(5);

        tracker.set_current_job("job1".to_string());
        tracker.update_progress(0.5);

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.current_job, Some("job1".to_string()));
        assert_eq!(snapshot.current_progress, 0.5);

        tracker.complete_job();
        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.completed_jobs, 1);
        assert_eq!(snapshot.current_job, None);
    }

    #[test]
    fn test_progress_completion() {
        let tracker = ProgressTracker::new(2);

        assert!(!tracker.is_complete());

        tracker.complete_job();
        assert!(!tracker.is_complete());

        tracker.complete_job();
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_progress_with_failures() {
        let tracker = ProgressTracker::new(3);

        tracker.complete_job();
        tracker.fail_job();
        tracker.complete_job();

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.completed_jobs, 2);
        assert_eq!(snapshot.failed_jobs, 1);
        assert!((snapshot.success_rate() - 200.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_overall_progress() {
        let tracker = ProgressTracker::new(4);

        tracker.complete_job();
        tracker.set_current_job("job2".to_string());
        tracker.update_progress(0.5);

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.overall_progress, 1.5 / 4.0);
    }

    #[test]
    fn test_reset() {
        let tracker = ProgressTracker::new(5);
        tracker.complete_job();
        tracker.complete_job();

        tracker.reset(10);

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.total_jobs, 10);
        assert_eq!(snapshot.completed_jobs, 0);
    }

    #[test]
    fn test_bytes_tracking() {
        let tracker = ProgressTracker::new(1);
        tracker.update_bytes(5000, 10000);

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.bytes_processed, 5000);
        assert_eq!(snapshot.total_bytes, 10000);
    }

    #[test]
    fn test_snapshot_methods() {
        let tracker = ProgressTracker::new(10);
        tracker.complete_job();
        tracker.complete_job();

        let snapshot = tracker.snapshot().unwrap();
        assert_eq!(snapshot.remaining_jobs(), 8);
        assert!(snapshot.jobs_per_second() >= 0.0);
    }
}
