//! Progressive job status updates with percentage completion and ETA estimation.
//!
//! This module provides a high-fidelity progress tracking system for encoding farm jobs.
//! Each job carries:
//! - A percentage completion value (0.0–100.0)
//! - An ETA estimate computed from a sliding-window throughput rate
//! - A status message and current phase label
//! - A ring-buffer of recent progress samples used for ETA smoothing
//!
//! ## ETA algorithm
//!
//! ETA is computed with an exponentially weighted moving average (EWMA) of the
//! instantaneous throughput (percentage-points per second).  This avoids
//! wild swings caused by momentary slowdowns while still reacting to sustained
//! speed changes.  A minimum of two samples is required before an ETA can be
//! computed.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_farm::job_progress::{JobProgressTracker, ProgressUpdate};
//! use oximedia_farm::JobId;
//!
//! let job_id = JobId::new();
//! let mut tracker = JobProgressTracker::new(job_id);
//!
//! tracker.record_progress(ProgressUpdate {
//!     percent: 25.0,
//!     phase: "encoding".to_string(),
//!     message: "frame 250 / 1000".to_string(),
//!     timestamp_secs: 10,
//! });
//!
//! tracker.record_progress(ProgressUpdate {
//!     percent: 50.0,
//!     phase: "encoding".to_string(),
//!     message: "frame 500 / 1000".to_string(),
//!     timestamp_secs: 20,
//! });
//!
//! let snapshot = tracker.snapshot().expect("has samples");
//! assert!(snapshot.eta_secs.is_some());
//! ```

use crate::JobId;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of progress samples retained in the sliding window.
const MAX_SAMPLES: usize = 16;

/// EWMA smoothing factor α (0 < α ≤ 1).  Higher → more weight on recent samples.
const EWMA_ALPHA: f64 = 0.3;

// ---------------------------------------------------------------------------
// Progress update
// ---------------------------------------------------------------------------

/// A single progress report emitted by a worker.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressUpdate {
    /// Percentage completion in the range `[0.0, 100.0]`.
    pub percent: f64,
    /// Current processing phase (e.g., `"encoding"`, `"muxing"`, `"uploading"`).
    pub phase: String,
    /// Human-readable status message.
    pub message: String,
    /// Unix timestamp (seconds) when this update was emitted.
    pub timestamp_secs: u64,
}

impl ProgressUpdate {
    /// Clamp `percent` to `[0.0, 100.0]`.
    #[must_use]
    pub fn clamped_percent(&self) -> f64 {
        self.percent.clamp(0.0, 100.0)
    }
}

// ---------------------------------------------------------------------------
// Snapshot
// ---------------------------------------------------------------------------

/// A point-in-time view of a job's progress, computed from accumulated samples.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgressSnapshot {
    /// Job this snapshot belongs to.
    pub job_id: JobId,
    /// Current percentage completion `[0.0, 100.0]`.
    pub percent: f64,
    /// Current phase label.
    pub phase: String,
    /// Latest status message.
    pub message: String,
    /// Estimated seconds until completion.  `None` if not enough data.
    pub eta_secs: Option<u64>,
    /// Smoothed throughput in percentage-points per second.  `None` if not enough data.
    pub throughput_pct_per_sec: Option<f64>,
    /// Unix timestamp of the latest update.
    pub last_updated_secs: u64,
    /// Total number of updates recorded so far.
    pub update_count: u64,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during progress tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgressError {
    /// The supplied percentage value is outside `[0.0, 100.0]`.
    InvalidPercent(String),
    /// A new update regresses the percentage below the previous value.
    ProgressRegression {
        /// Previously recorded percentage.
        previous: String,
        /// New (lower) percentage that was rejected.
        attempted: String,
    },
    /// No progress samples recorded yet.
    NoSamples,
}

impl std::fmt::Display for ProgressError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidPercent(v) => write!(f, "invalid percent value: {v}"),
            Self::ProgressRegression { previous, attempted } => write!(
                f,
                "progress regression: attempted {attempted}% but previous was {previous}%"
            ),
            Self::NoSamples => write!(f, "no progress samples recorded"),
        }
    }
}

impl std::error::Error for ProgressError {}

/// Result type for progress operations.
pub type Result<T> = std::result::Result<T, ProgressError>;

// ---------------------------------------------------------------------------
// Internal sample
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct ProgressSample {
    percent: f64,
    timestamp_secs: u64,
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

/// Tracks progressive progress for a single farm job.
///
/// Thread-safety: this struct is not `Sync` on its own.  Wrap in
/// `Arc<Mutex<…>>` if shared across threads.
#[derive(Debug)]
pub struct JobProgressTracker {
    /// Job this tracker belongs to.
    job_id: JobId,
    /// Sliding window of recent samples.
    samples: VecDeque<ProgressSample>,
    /// Latest phase string.
    phase: String,
    /// Latest status message.
    message: String,
    /// EWMA of throughput (pct/s); `None` until second sample arrives.
    ewma_throughput: Option<f64>,
    /// Total number of updates applied (including rejected ones that were clamped).
    update_count: u64,
}

impl JobProgressTracker {
    /// Create a new tracker for the given job.
    #[must_use]
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            samples: VecDeque::with_capacity(MAX_SAMPLES),
            phase: String::new(),
            message: String::new(),
            ewma_throughput: None,
            update_count: 0,
        }
    }

    /// Record a new progress update.
    ///
    /// The percent value is clamped to `[0.0, 100.0]`.  If the clamped value
    /// regresses below the most recent recorded percentage, `ProgressRegression`
    /// is returned and no state change occurs.
    ///
    /// # Errors
    ///
    /// Returns `ProgressError::InvalidPercent` if `percent` is `NaN`.
    /// Returns `ProgressError::ProgressRegression` if the new value is lower
    /// than the previous one.
    pub fn record_progress(&mut self, update: ProgressUpdate) -> Result<()> {
        if update.percent.is_nan() {
            return Err(ProgressError::InvalidPercent("NaN".to_string()));
        }

        let percent = update.percent.clamp(0.0, 100.0);

        // Reject regression
        if let Some(last) = self.samples.back() {
            if percent < last.percent {
                return Err(ProgressError::ProgressRegression {
                    previous: format!("{:.2}", last.percent),
                    attempted: format!("{:.2}", percent),
                });
            }
        }

        // Evict oldest when window is full
        if self.samples.len() == MAX_SAMPLES {
            self.samples.pop_front();
        }

        // Update EWMA throughput from the previous sample
        if let Some(prev) = self.samples.back() {
            let dt = update
                .timestamp_secs
                .saturating_sub(prev.timestamp_secs) as f64;
            if dt > 0.0 {
                let instantaneous = (percent - prev.percent) / dt;
                let smoothed = match self.ewma_throughput {
                    Some(prev_ewma) => EWMA_ALPHA * instantaneous + (1.0 - EWMA_ALPHA) * prev_ewma,
                    None => instantaneous,
                };
                self.ewma_throughput = Some(smoothed.max(0.0));
            }
        }

        self.samples.push_back(ProgressSample {
            percent,
            timestamp_secs: update.timestamp_secs,
        });

        self.phase = update.phase;
        self.message = update.message;
        self.update_count += 1;

        Ok(())
    }

    /// Return a snapshot of the current progress state.
    ///
    /// # Errors
    ///
    /// Returns `ProgressError::NoSamples` if no updates have been recorded yet.
    pub fn snapshot(&self) -> Result<ProgressSnapshot> {
        let last = self.samples.back().ok_or(ProgressError::NoSamples)?;

        let (eta_secs, throughput_pct_per_sec) = if let Some(tp) = self.ewma_throughput {
            if tp > 0.0 {
                let remaining = 100.0 - last.percent;
                let eta = (remaining / tp).ceil() as u64;
                (Some(eta), Some(tp))
            } else {
                (None, Some(tp))
            }
        } else {
            (None, None)
        };

        Ok(ProgressSnapshot {
            job_id: self.job_id,
            percent: last.percent,
            phase: self.phase.clone(),
            message: self.message.clone(),
            eta_secs,
            throughput_pct_per_sec,
            last_updated_secs: last.timestamp_secs,
            update_count: self.update_count,
        })
    }

    /// Return the current completion percentage, or `0.0` if no samples yet.
    #[must_use]
    pub fn current_percent(&self) -> f64 {
        self.samples.back().map_or(0.0, |s| s.percent)
    }

    /// Return `true` if the job is complete (percent ≥ 100.0).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.current_percent() >= 100.0
    }

    /// Reset the tracker, clearing all samples and EWMA state.
    pub fn reset(&mut self) {
        self.samples.clear();
        self.ewma_throughput = None;
        self.update_count = 0;
        self.phase.clear();
        self.message.clear();
    }

    /// Return the job ID this tracker belongs to.
    #[must_use]
    pub fn job_id(&self) -> &JobId {
        &self.job_id
    }
}

// ---------------------------------------------------------------------------
// Multi-job registry
// ---------------------------------------------------------------------------

/// A registry of progress trackers for multiple concurrent jobs.
///
/// Backed by a `dashmap::DashMap` for lock-free concurrent access.
pub struct FarmProgressRegistry {
    trackers: dashmap::DashMap<String, JobProgressTracker>,
}

impl FarmProgressRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            trackers: dashmap::DashMap::new(),
        }
    }

    /// Register a new job tracker.  Replaces any existing tracker for `job_id`.
    pub fn register(&self, job_id: JobId) {
        self.trackers
            .insert(job_id.to_string(), JobProgressTracker::new(job_id));
    }

    /// Apply a progress update to the tracker for `job_id`.
    ///
    /// # Errors
    ///
    /// Returns `ProgressError` if the tracker is not registered or if the
    /// update is invalid.
    pub fn update(&self, job_id: &JobId, update: ProgressUpdate) -> Result<()> {
        let mut tracker = self
            .trackers
            .get_mut(&job_id.to_string())
            .ok_or(ProgressError::NoSamples)?;
        tracker.record_progress(update)
    }

    /// Return a snapshot for `job_id`.
    ///
    /// # Errors
    ///
    /// Returns `ProgressError::NoSamples` if the tracker is not found or has
    /// no samples yet.
    pub fn snapshot(&self, job_id: &JobId) -> Result<ProgressSnapshot> {
        self.trackers
            .get(&job_id.to_string())
            .ok_or(ProgressError::NoSamples)?
            .snapshot()
    }

    /// Remove a job from the registry.
    pub fn remove(&self, job_id: &JobId) {
        self.trackers.remove(&job_id.to_string());
    }

    /// Return the number of tracked jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.trackers.len()
    }

    /// Return `true` if no jobs are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }
}

impl Default for FarmProgressRegistry {
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

    fn make_update(percent: f64, timestamp_secs: u64) -> ProgressUpdate {
        ProgressUpdate {
            percent,
            phase: "encoding".to_string(),
            message: format!("at {percent}%"),
            timestamp_secs,
        }
    }

    #[test]
    fn test_first_update_records_correctly() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(10.0, 5)).expect("should succeed");
        assert!((tracker.current_percent() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_two_updates_produce_eta() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(0.0, 0)).expect("first update");
        tracker.record_progress(make_update(10.0, 10)).expect("second update");
        let snap = tracker.snapshot().expect("snapshot");
        // 10 pct in 10 s → 1 pct/s → 90 s remaining
        assert!(snap.eta_secs.is_some());
        assert_eq!(snap.eta_secs.unwrap(), 90);
    }

    #[test]
    fn test_ewma_smooths_throughput() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(0.0, 0)).expect("update 0");
        tracker.record_progress(make_update(20.0, 10)).expect("update 1"); // 2 pct/s
        tracker.record_progress(make_update(22.0, 11)).expect("update 2"); // 2 pct/s instant
        let snap = tracker.snapshot().expect("snapshot");
        // EWMA should still show > 0 throughput
        assert!(snap.throughput_pct_per_sec.unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn test_regression_rejected() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(50.0, 10)).expect("first");
        let err = tracker.record_progress(make_update(30.0, 20)).unwrap_err();
        assert!(matches!(err, ProgressError::ProgressRegression { .. }));
    }

    #[test]
    fn test_nan_rejected() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        let err = tracker.record_progress(make_update(f64::NAN, 1)).unwrap_err();
        assert!(matches!(err, ProgressError::InvalidPercent(_)));
    }

    #[test]
    fn test_snapshot_without_samples_returns_error() {
        let tracker = JobProgressTracker::new(JobId::new());
        assert!(tracker.snapshot().is_err());
    }

    #[test]
    fn test_is_complete_at_100() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(100.0, 60)).expect("complete");
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_reset_clears_state() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(50.0, 10)).expect("update");
        tracker.reset();
        assert!((tracker.current_percent() - 0.0).abs() < f64::EPSILON);
        assert!(tracker.snapshot().is_err());
    }

    #[test]
    fn test_percent_clamped_above_100() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        tracker.record_progress(make_update(150.0, 10)).expect("over 100");
        assert!((tracker.current_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_registry_register_and_update() {
        let registry = FarmProgressRegistry::new();
        let job_id = JobId::new();
        registry.register(job_id);
        registry.update(&job_id, make_update(0.0, 0)).expect("first");
        registry.update(&job_id, make_update(25.0, 25)).expect("second");
        let snap = registry.snapshot(&job_id).expect("snap");
        assert!((snap.percent - 25.0).abs() < f64::EPSILON);
        assert_eq!(snap.eta_secs, Some(75));
    }

    #[test]
    fn test_registry_remove() {
        let registry = FarmProgressRegistry::new();
        let job_id = JobId::new();
        registry.register(job_id);
        assert_eq!(registry.len(), 1);
        registry.remove(&job_id);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_update_count_increments() {
        let job_id = JobId::new();
        let mut tracker = JobProgressTracker::new(job_id);
        for i in 0u64..5 {
            tracker
                .record_progress(make_update(i as f64 * 10.0, i * 10))
                .expect("update");
        }
        let snap = tracker.snapshot().expect("snap");
        assert_eq!(snap.update_count, 5);
    }

    #[test]
    fn test_progress_error_display() {
        let err = ProgressError::NoSamples;
        assert!(!err.to_string().is_empty());

        let err2 = ProgressError::ProgressRegression {
            previous: "50.00".to_string(),
            attempted: "30.00".to_string(),
        };
        assert!(err2.to_string().contains("50.00"));
    }
}
