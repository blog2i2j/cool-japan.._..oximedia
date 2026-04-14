//! Clip transcode status tracking: proxy generation and transcode progress per clip.
//!
//! This module provides per-clip tracking of ongoing and completed transcode
//! operations.  A single clip may have multiple active transcodes (e.g. a
//! proxy generation job running in parallel with an archive encode).
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::clip_transcode_status::{TranscodeStatusStore, TranscodeJob, TranscodeState};
//!
//! let mut store = TranscodeStatusStore::new();
//! let job = TranscodeJob::new("clip-01".to_string(), "proxy-720p".to_string());
//! let job_id = job.job_id.clone();
//! store.register(job);
//!
//! store.update_progress(&job_id, 0.5);
//! assert_eq!(store.job(&job_id).map(|j| j.state.clone()), Some(TranscodeState::Running));
//!
//! store.complete(&job_id, "/cache/clip-01-proxy.mov".to_string());
//! assert!(matches!(store.job(&job_id).map(|j| &j.state), Some(TranscodeState::Completed { .. })));
//! ```

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// TranscodePreset
// ─────────────────────────────────────────────────────────────────────────────

/// Well-known transcode preset names.
pub mod preset {
    /// Low-resolution editing proxy (e.g. 720p DNxHD).
    pub const PROXY_720P: &str = "proxy-720p";
    /// Ultra-low-resolution offline proxy for slow connections.
    pub const PROXY_360P: &str = "proxy-360p";
    /// Full-resolution delivery master.
    pub const DELIVERY_MASTER: &str = "delivery-master";
    /// H.264 web streaming preset.
    pub const WEB_H264: &str = "web-h264";
    /// AV1 streaming preset.
    pub const WEB_AV1: &str = "web-av1";
    /// Lossless archive (FFV1 + FLAC).
    pub const ARCHIVE_FFV1: &str = "archive-ffv1";
    /// Audio-only waveform extraction.
    pub const AUDIO_WAVEFORM: &str = "audio-waveform";
}

// ─────────────────────────────────────────────────────────────────────────────
// TranscodeState
// ─────────────────────────────────────────────────────────────────────────────

/// The current state of a transcode job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TranscodeState {
    /// Job is queued but has not started yet.
    Queued,
    /// Job is actively transcoding. `progress` is in `[0.0, 1.0]`.
    Running,
    /// Job finished successfully. `output_path` is the destination file.
    Completed {
        /// Path to the output file.
        output_path: String,
    },
    /// Job failed. `reason` describes the error.
    Failed {
        /// Human-readable error description.
        reason: String,
    },
    /// Job was cancelled by the user.
    Cancelled,
}

impl TranscodeState {
    /// Returns `true` if the job has reached a terminal state.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed { .. } | Self::Failed { .. } | Self::Cancelled
        )
    }

    /// Returns `true` if the job succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TranscodeJob
// ─────────────────────────────────────────────────────────────────────────────

/// Represents a single transcode operation for one clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeJob {
    /// Unique job identifier.
    pub job_id: String,
    /// The clip this job belongs to.
    pub clip_id: String,
    /// The preset being applied (e.g. `"proxy-720p"`, `"delivery-master"`).
    pub preset: String,
    /// Current state.
    pub state: TranscodeState,
    /// Progress in `[0.0, 1.0]`.  Meaningless if not in `Running` state.
    pub progress: f32,
    /// When the job was queued.
    pub queued_at: DateTime<Utc>,
    /// When the job started (i.e. transitioned from Queued to Running).
    pub started_at: Option<DateTime<Utc>>,
    /// When the job reached a terminal state.
    pub finished_at: Option<DateTime<Utc>>,
    /// Current encoding speed relative to real-time (e.g. 2.5 = 2.5×).
    pub speed_factor: Option<f32>,
    /// Estimated seconds remaining. `None` if unknown.
    pub eta_secs: Option<u64>,
}

impl TranscodeJob {
    /// Create a new queued job.
    #[must_use]
    pub fn new(clip_id: String, preset: String) -> Self {
        use uuid::Uuid;
        Self {
            job_id: Uuid::new_v4().to_string(),
            clip_id,
            preset,
            state: TranscodeState::Queued,
            progress: 0.0,
            queued_at: Utc::now(),
            started_at: None,
            finished_at: None,
            speed_factor: None,
            eta_secs: None,
        }
    }

    /// Mark the job as started.
    pub fn start(&mut self) {
        if self.state == TranscodeState::Queued {
            self.state = TranscodeState::Running;
            self.started_at = Some(Utc::now());
        }
    }

    /// Update the running progress. Clamps to `[0.0, 1.0]`.
    ///
    /// Automatically transitions to `Running` if still `Queued`.
    pub fn update_progress(&mut self, progress: f32) {
        if self.state == TranscodeState::Queued {
            self.start();
        }
        self.progress = progress.clamp(0.0, 1.0);
    }

    /// Set encoding speed and recompute ETA.
    pub fn update_speed(&mut self, speed_factor: f32, total_duration_secs: Option<f64>) {
        self.speed_factor = Some(speed_factor);
        if let Some(total) = total_duration_secs {
            if speed_factor > 0.0 {
                let remaining_fraction = (1.0 - self.progress as f64).max(0.0);
                let remaining_real_secs = (total * remaining_fraction) / speed_factor as f64;
                self.eta_secs = Some(remaining_real_secs.ceil() as u64);
            }
        }
    }

    /// Mark the job as successfully completed.
    pub fn complete(&mut self, output_path: String) {
        self.state = TranscodeState::Completed { output_path };
        self.progress = 1.0;
        self.finished_at = Some(Utc::now());
        self.eta_secs = None;
    }

    /// Mark the job as failed.
    pub fn fail(&mut self, reason: String) {
        self.state = TranscodeState::Failed { reason };
        self.finished_at = Some(Utc::now());
    }

    /// Cancel the job.
    pub fn cancel(&mut self) {
        if !self.state.is_terminal() {
            self.state = TranscodeState::Cancelled;
            self.finished_at = Some(Utc::now());
        }
    }

    /// Elapsed duration in seconds since job was queued.
    #[must_use]
    pub fn elapsed_secs(&self) -> i64 {
        Utc::now()
            .signed_duration_since(self.queued_at)
            .num_seconds()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TranscodeStatusStore
// ─────────────────────────────────────────────────────────────────────────────

/// Central store tracking all transcode jobs across all clips.
///
/// Jobs are indexed by job ID for O(1) access. A reverse index from
/// clip ID → job IDs allows efficient per-clip queries.
#[derive(Debug, Default)]
pub struct TranscodeStatusStore {
    jobs: HashMap<String, TranscodeJob>,
    /// clip_id → set of job_ids
    clip_index: HashMap<String, Vec<String>>,
}

impl TranscodeStatusStore {
    /// Create an empty status store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new job. The job is stored in its current state (usually
    /// `Queued`).
    pub fn register(&mut self, job: TranscodeJob) {
        self.clip_index
            .entry(job.clip_id.clone())
            .or_default()
            .push(job.job_id.clone());
        self.jobs.insert(job.job_id.clone(), job);
    }

    /// Create and register a new queued job for a clip+preset combination.
    ///
    /// Returns the job ID.
    pub fn create_job(&mut self, clip_id: String, preset: String) -> String {
        let job = TranscodeJob::new(clip_id, preset);
        let id = job.job_id.clone();
        self.register(job);
        id
    }

    /// Retrieve a job by ID (read-only).
    #[must_use]
    pub fn job(&self, job_id: &str) -> Option<&TranscodeJob> {
        self.jobs.get(job_id)
    }

    /// Retrieve a job by ID (mutable).
    pub fn job_mut(&mut self, job_id: &str) -> Option<&mut TranscodeJob> {
        self.jobs.get_mut(job_id)
    }

    /// All jobs for a specific clip (read-only).
    #[must_use]
    pub fn jobs_for_clip(&self, clip_id: &str) -> Vec<&TranscodeJob> {
        self.clip_index
            .get(clip_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.jobs.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Active (non-terminal) jobs for a clip.
    #[must_use]
    pub fn active_jobs_for_clip(&self, clip_id: &str) -> Vec<&TranscodeJob> {
        self.jobs_for_clip(clip_id)
            .into_iter()
            .filter(|j| !j.state.is_terminal())
            .collect()
    }

    /// Start a queued job.
    pub fn start(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.start();
        }
    }

    /// Update progress for a job. Also auto-starts queued jobs.
    pub fn update_progress(&mut self, job_id: &str, progress: f32) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.update_progress(progress);
        }
    }

    /// Mark a job as successfully completed.
    pub fn complete(&mut self, job_id: &str, output_path: String) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.complete(output_path);
        }
    }

    /// Mark a job as failed.
    pub fn fail(&mut self, job_id: &str, reason: String) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.fail(reason);
        }
    }

    /// Cancel a job.
    pub fn cancel(&mut self, job_id: &str) {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.cancel();
        }
    }

    /// Remove all terminal (completed/failed/cancelled) jobs from the store.
    /// Returns the number of jobs removed.
    pub fn purge_terminal(&mut self) -> usize {
        let terminal: Vec<String> = self
            .jobs
            .iter()
            .filter(|(_, j)| j.state.is_terminal())
            .map(|(id, _)| id.clone())
            .collect();
        let count = terminal.len();
        for id in &terminal {
            if let Some(job) = self.jobs.remove(id) {
                if let Some(clip_jobs) = self.clip_index.get_mut(&job.clip_id) {
                    clip_jobs.retain(|jid| jid != id);
                }
            }
        }
        count
    }

    /// Total number of jobs tracked (including terminal).
    #[must_use]
    pub fn total_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// Summary stats: `(queued, running, completed, failed, cancelled)`.
    #[must_use]
    pub fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let mut queued = 0usize;
        let mut running = 0usize;
        let mut completed = 0usize;
        let mut failed = 0usize;
        let mut cancelled = 0usize;
        for job in self.jobs.values() {
            match job.state {
                TranscodeState::Queued => queued += 1,
                TranscodeState::Running => running += 1,
                TranscodeState::Completed { .. } => completed += 1,
                TranscodeState::Failed { .. } => failed += 1,
                TranscodeState::Cancelled => cancelled += 1,
            }
        }
        (queued, running, completed, failed, cancelled)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_lifecycle_queued_running_completed() {
        let mut job = TranscodeJob::new("clip-01".to_string(), preset::PROXY_720P.to_string());
        assert_eq!(job.state, TranscodeState::Queued);

        job.start();
        assert_eq!(job.state, TranscodeState::Running);

        job.update_progress(0.5);
        assert!((job.progress - 0.5).abs() < f32::EPSILON);

        job.complete("/out/clip-01-proxy.mov".to_string());
        assert!(job.state.is_terminal());
        assert!(job.state.is_success());
    }

    #[test]
    fn test_job_failure() {
        let mut job = TranscodeJob::new("c".to_string(), "web".to_string());
        job.start();
        job.fail("Codec not found".to_string());
        assert!(matches!(job.state, TranscodeState::Failed { .. }));
        assert!(job.state.is_terminal());
        assert!(!job.state.is_success());
    }

    #[test]
    fn test_job_cancellation() {
        let mut job = TranscodeJob::new("c".to_string(), "web".to_string());
        job.cancel();
        assert_eq!(job.state, TranscodeState::Cancelled);
    }

    #[test]
    fn test_store_create_and_complete() {
        let mut store = TranscodeStatusStore::new();
        let id = store.create_job("clip-a".to_string(), preset::PROXY_720P.to_string());
        assert_eq!(store.total_jobs(), 1);

        store.update_progress(&id, 0.75);
        assert!((store.job(&id).expect("job").progress - 0.75).abs() < f32::EPSILON);

        store.complete(&id, "/out/a.mov".to_string());
        let j = store.job(&id).expect("job");
        assert!(j.state.is_success());
    }

    #[test]
    fn test_jobs_for_clip() {
        let mut store = TranscodeStatusStore::new();
        let id1 = store.create_job("clip-b".to_string(), preset::PROXY_720P.to_string());
        let id2 = store.create_job("clip-b".to_string(), preset::ARCHIVE_FFV1.to_string());
        store.create_job("clip-c".to_string(), preset::WEB_H264.to_string());

        let b_jobs = store.jobs_for_clip("clip-b");
        assert_eq!(b_jobs.len(), 2);
        let ids: Vec<&str> = b_jobs.iter().map(|j| j.job_id.as_str()).collect();
        assert!(ids.contains(&id1.as_str()));
        assert!(ids.contains(&id2.as_str()));
    }

    #[test]
    fn test_purge_terminal() {
        let mut store = TranscodeStatusStore::new();
        let done = store.create_job("c".to_string(), "p".to_string());
        let running = store.create_job("c".to_string(), "q".to_string());
        store.complete(&done, "/out".to_string());
        store.start(&running);

        let removed = store.purge_terminal();
        assert_eq!(removed, 1);
        assert_eq!(store.total_jobs(), 1);
        assert!(store.job(&running).is_some());
    }

    #[test]
    fn test_stats() {
        let mut store = TranscodeStatusStore::new();
        let id1 = store.create_job("c1".to_string(), "p".to_string());
        let id2 = store.create_job("c2".to_string(), "p".to_string());
        let id3 = store.create_job("c3".to_string(), "p".to_string());
        store.start(&id2);
        store.complete(&id3, "/out".to_string());

        let (queued, running, completed, failed, cancelled) = store.stats();
        assert_eq!(queued, 1);
        assert_eq!(running, 1);
        assert_eq!(completed, 1);
        assert_eq!(failed, 0);
        assert_eq!(cancelled, 0);
        drop(id1);
    }

    #[test]
    fn test_update_progress_auto_starts() {
        let mut store = TranscodeStatusStore::new();
        let id = store.create_job("c".to_string(), "p".to_string());
        store.update_progress(&id, 0.1);
        assert_eq!(
            store.job(&id).expect("job").state,
            TranscodeState::Running
        );
    }

    #[test]
    fn test_eta_computation() {
        let mut job = TranscodeJob::new("c".to_string(), "p".to_string());
        job.start();
        job.update_progress(0.5);
        // 10-second clip at 2×, 50% done → 2.5 s remaining
        job.update_speed(2.0, Some(10.0));
        let eta = job.eta_secs.expect("eta should be set");
        assert_eq!(eta, 3); // ceil(2.5) = 3
    }
}
