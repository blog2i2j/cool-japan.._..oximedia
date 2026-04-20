//! Multi-user export coordination for collaborative sessions.
//!
//! Prevents conflicting concurrent exports of the same project, manages an
//! ordered export queue, tracks real-time progress, and provides a clean
//! audit trail of all export attempts.
//!
//! # Design
//! * **ExportJob** — describes a single export request (format, range, settings).
//! * **ExportQueue** — a FIFO queue of pending jobs; at most one job per
//!   project may be *active* at a time.
//! * **ExportCoordinator** — the top-level entry point.  Thread-safe via
//!   `Arc<ExportCoordinator>`.
//! * Progress is reported as an integer percentage `[0, 100]` stored in an
//!   `AtomicU8` so it can be read without holding any lock.

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::{CollabError, Result};

// ---------------------------------------------------------------------------
// ExportFormat
// ---------------------------------------------------------------------------

/// Output format requested by the export job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// ProRes family (specify codec variant in settings).
    ProRes,
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// AV1.
    Av1,
    /// VP9 in a WebM container.
    Vp9Webm,
    /// DNxHD / DNxHR family.
    DnxHd,
    /// FFV1 lossless.
    Ffv1,
    /// Audio-only — AAC.
    AudioAac,
    /// Audio-only — FLAC lossless.
    AudioFlac,
    /// Proxy / preview — low-resolution H.264.
    ProxyH264,
    /// Image sequence — TIFF frames.
    TiffSequence,
    /// Custom format string — passed verbatim to the codec backend.
    Custom(String),
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ProRes => write!(f, "prores"),
            Self::H264 => write!(f, "h264"),
            Self::H265 => write!(f, "h265"),
            Self::Av1 => write!(f, "av1"),
            Self::Vp9Webm => write!(f, "vp9_webm"),
            Self::DnxHd => write!(f, "dnxhd"),
            Self::Ffv1 => write!(f, "ffv1"),
            Self::AudioAac => write!(f, "audio_aac"),
            Self::AudioFlac => write!(f, "audio_flac"),
            Self::ProxyH264 => write!(f, "proxy_h264"),
            Self::TiffSequence => write!(f, "tiff_sequence"),
            Self::Custom(s) => write!(f, "custom:{s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// ExportRange
// ---------------------------------------------------------------------------

/// Time range to export (milliseconds, half-open `[start, end)`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportRange {
    /// Start of the export range (ms, inclusive).
    pub start_ms: u64,
    /// End of the export range (ms, exclusive).  `None` means end-of-timeline.
    pub end_ms: Option<u64>,
}

impl ExportRange {
    /// Full-timeline export.
    pub fn full() -> Self {
        Self {
            start_ms: 0,
            end_ms: None,
        }
    }

    /// Partial export.
    ///
    /// Returns an error if `end_ms <= start_ms`.
    pub fn partial(start_ms: u64, end_ms: u64) -> Result<Self> {
        if end_ms <= start_ms {
            return Err(CollabError::InvalidOperation(format!(
                "ExportRange: end_ms ({end_ms}) must be > start_ms ({start_ms})"
            )));
        }
        Ok(Self {
            start_ms,
            end_ms: Some(end_ms),
        })
    }
}

// ---------------------------------------------------------------------------
// ExportSettings
// ---------------------------------------------------------------------------

/// Codec and mux settings for an export job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSettings {
    /// Target video bitrate in kbps (0 = format default / lossless).
    pub video_bitrate_kbps: u32,
    /// Target audio bitrate in kbps (0 = format default).
    pub audio_bitrate_kbps: u32,
    /// Constant-rate-factor (0 = disabled / use bitrate).
    pub crf: Option<u8>,
    /// Target frame rate numerator.
    pub fps_num: u32,
    /// Target frame rate denominator.
    pub fps_den: u32,
    /// Output width in pixels (0 = source width).
    pub width: u32,
    /// Output height in pixels (0 = source height).
    pub height: u32,
    /// Output file path or URI.
    pub output_path: String,
    /// Additional format-specific key-value options.
    pub extra: HashMap<String, String>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            video_bitrate_kbps: 0,
            audio_bitrate_kbps: 0,
            crf: None,
            fps_num: 30,
            fps_den: 1,
            width: 0,
            height: 0,
            output_path: String::new(),
            extra: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// ExportStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of an export job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportStatus {
    /// Waiting in queue.
    Queued,
    /// Currently encoding.
    Running,
    /// Finished successfully.
    Completed,
    /// Cancelled by a user.
    Cancelled,
    /// Failed with an error message.
    Failed(String),
}

impl ExportStatus {
    /// Returns `true` if the job has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed(_)
        )
    }
}

impl std::fmt::Display for ExportStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Failed(msg) => write!(f, "failed({msg})"),
        }
    }
}

// ---------------------------------------------------------------------------
// ExportJob
// ---------------------------------------------------------------------------

/// A single export request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportJob {
    /// Unique job identifier.
    pub id: Uuid,
    /// Project the export belongs to.
    pub project_id: Uuid,
    /// Session in which the export was requested.
    pub session_id: Uuid,
    /// User who submitted the export.
    pub submitted_by: Uuid,
    /// Human-readable name of the submitter.
    pub submitted_by_name: String,
    /// Output format.
    pub format: ExportFormat,
    /// Time range to export.
    pub range: ExportRange,
    /// Codec / mux settings.
    pub settings: ExportSettings,
    /// Current job status.
    pub status: ExportStatus,
    /// Wall-clock submission time (epoch ms).
    pub submitted_at_ms: u64,
    /// Wall-clock time when the job started running (epoch ms).
    pub started_at_ms: Option<u64>,
    /// Wall-clock time when the job reached a terminal state (epoch ms).
    pub finished_at_ms: Option<u64>,
    /// Optional label for human display (e.g. "Final cut – delivery").
    pub label: String,
}

impl ExportJob {
    /// Create a new `ExportJob` in `Queued` state.
    pub fn new(
        project_id: Uuid,
        session_id: Uuid,
        submitted_by: Uuid,
        submitted_by_name: impl Into<String>,
        format: ExportFormat,
        range: ExportRange,
        settings: ExportSettings,
        label: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            project_id,
            session_id,
            submitted_by,
            submitted_by_name: submitted_by_name.into(),
            format,
            range,
            settings,
            status: ExportStatus::Queued,
            submitted_at_ms: now_ms,
            started_at_ms: None,
            finished_at_ms: None,
            label: label.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// ExportProgress
// ---------------------------------------------------------------------------

/// Atomic progress tracker for a running job.
///
/// Uses `AtomicU8` so callers can poll without acquiring a lock.
pub struct ExportProgress {
    /// Job ID this progress belongs to.
    pub job_id: Uuid,
    /// Completion percentage `[0, 100]`.
    percent: AtomicU8,
    /// Estimated seconds remaining (u16::MAX = unknown).
    eta_secs: Mutex<Option<u16>>,
    /// Optional human-readable status message.
    message: Mutex<String>,
}

impl ExportProgress {
    fn new(job_id: Uuid) -> Arc<Self> {
        Arc::new(Self {
            job_id,
            percent: AtomicU8::new(0),
            eta_secs: Mutex::new(None),
            message: Mutex::new(String::new()),
        })
    }

    /// Current completion percentage `[0, 100]`.
    pub fn percent(&self) -> u8 {
        self.percent.load(Ordering::Relaxed)
    }

    /// Update progress.
    ///
    /// `percent` is clamped to `[0, 100]`.
    pub fn update(&self, percent: u8, eta_secs: Option<u16>, message: impl Into<String>) {
        self.percent
            .store(percent.min(100), Ordering::Relaxed);
        *self.eta_secs.lock() = eta_secs;
        *self.message.lock() = message.into();
    }

    /// Estimated seconds remaining, if known.
    pub fn eta_secs(&self) -> Option<u16> {
        *self.eta_secs.lock()
    }

    /// Current status message.
    pub fn message(&self) -> String {
        self.message.lock().clone()
    }
}

// ---------------------------------------------------------------------------
// ExportCoordinatorConfig
// ---------------------------------------------------------------------------

/// Configuration for [`ExportCoordinator`].
#[derive(Debug, Clone)]
pub struct ExportCoordinatorConfig {
    /// Maximum number of concurrent active (Running) jobs across all projects.
    pub max_concurrent: usize,
    /// Maximum number of completed/failed jobs to keep in history per project.
    pub history_per_project: usize,
    /// Maximum queue depth across all projects.
    pub max_queue_depth: usize,
}

impl Default for ExportCoordinatorConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 2,
            history_per_project: 100,
            max_queue_depth: 512,
        }
    }
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

struct Inner {
    /// Pending jobs waiting to run, ordered FIFO.
    queue: VecDeque<ExportJob>,
    /// Currently running jobs by job ID.
    running: HashMap<Uuid, (ExportJob, Arc<ExportProgress>)>,
    /// Completed / failed / cancelled jobs per project.
    history: HashMap<Uuid, Vec<ExportJob>>,
}

impl Inner {
    fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            running: HashMap::new(),
            history: HashMap::new(),
        }
    }

    /// Total queued depth.
    fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Number of jobs currently in `Running` state.
    fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Trim history for `project_id` to `max` entries.
    fn trim_history(&mut self, project_id: Uuid, max: usize) {
        if let Some(hist) = self.history.get_mut(&project_id) {
            if hist.len() > max {
                let drain_count = hist.len() - max;
                hist.drain(0..drain_count);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ExportCoordinator
// ---------------------------------------------------------------------------

/// Thread-safe coordinator for multi-user export operations.
pub struct ExportCoordinator {
    config: ExportCoordinatorConfig,
    inner: Arc<RwLock<Inner>>,
}

impl ExportCoordinator {
    /// Create a new coordinator with default configuration.
    pub fn new() -> Self {
        Self::with_config(ExportCoordinatorConfig::default())
    }

    /// Create a new coordinator with a custom configuration.
    pub fn with_config(config: ExportCoordinatorConfig) -> Self {
        Self {
            config,
            inner: Arc::new(RwLock::new(Inner::new())),
        }
    }

    // -----------------------------------------------------------------------
    // Submitting jobs
    // -----------------------------------------------------------------------

    /// Submit a new export job.
    ///
    /// Returns the job's `Uuid` on success.
    ///
    /// Errors:
    /// * `InvalidOperation` — the queue is at capacity.
    /// * `LockFailed` — the project already has a `Running` job (only one
    ///   concurrent export per project is permitted).
    pub fn submit(&self, job: ExportJob) -> Result<Uuid> {
        let mut inner = self.inner.write();

        if inner.queue_len() >= self.config.max_queue_depth {
            return Err(CollabError::InvalidOperation(format!(
                "export queue is full (max {})",
                self.config.max_queue_depth
            )));
        }

        // Only one active export per project at a time.
        let project_already_running = inner
            .running
            .values()
            .any(|(j, _)| j.project_id == job.project_id);
        if project_already_running {
            return Err(CollabError::LockFailed(format!(
                "project {} already has a running export",
                job.project_id
            )));
        }

        let id = job.id;
        inner.queue.push_back(job);
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // Lifecycle transitions
    // -----------------------------------------------------------------------

    /// Dequeue the next job and transition it to `Running`.
    ///
    /// Returns `None` when the queue is empty or the concurrent limit is
    /// already reached.
    pub fn start_next(&self, now_ms: u64) -> Option<(ExportJob, Arc<ExportProgress>)> {
        let mut inner = self.inner.write();

        if inner.running_count() >= self.config.max_concurrent {
            return None;
        }

        // Find the first queued job whose project does not already have a
        // running export.
        let idx = inner.queue.iter().position(|job| {
            !inner
                .running
                .values()
                .any(|(r, _)| r.project_id == job.project_id)
        })?;

        let mut job = inner.queue.remove(idx)?;
        job.status = ExportStatus::Running;
        job.started_at_ms = Some(now_ms);

        let progress = ExportProgress::new(job.id);
        let result = (job.clone(), progress.clone());
        inner.running.insert(job.id, (job, progress));
        Some(result)
    }

    /// Mark a running job as `Completed`.
    pub fn complete(&self, job_id: Uuid, now_ms: u64) -> Result<()> {
        self.finish_job(job_id, ExportStatus::Completed, now_ms)
    }

    /// Mark a running job as `Failed` with a reason.
    pub fn fail(&self, job_id: Uuid, reason: impl Into<String>, now_ms: u64) -> Result<()> {
        self.finish_job(job_id, ExportStatus::Failed(reason.into()), now_ms)
    }

    /// Cancel a job — works for both `Queued` and `Running` jobs.
    ///
    /// Returns `Err(InvalidOperation)` if the job is not found or is already
    /// in a terminal state.
    pub fn cancel(&self, job_id: Uuid, now_ms: u64) -> Result<()> {
        let mut inner = self.inner.write();

        // Check running first.
        if inner.running.contains_key(&job_id) {
            return self.finish_job_inner(&mut inner, job_id, ExportStatus::Cancelled, now_ms);
        }

        // Check queue.
        let pos = inner.queue.iter().position(|j| j.id == job_id);
        match pos {
            None => Err(CollabError::InvalidOperation(format!(
                "export job {job_id} not found"
            ))),
            Some(idx) => {
                let mut job = inner.queue.remove(idx).ok_or_else(|| {
                    CollabError::InvalidOperation(format!("export job {job_id} not found"))
                })?;
                job.status = ExportStatus::Cancelled;
                job.finished_at_ms = Some(now_ms);
                let project_id = job.project_id;
                inner
                    .history
                    .entry(project_id)
                    .or_default()
                    .push(job);
                let max = self.config.history_per_project;
                inner.trim_history(project_id, max);
                Ok(())
            }
        }
    }

    fn finish_job(&self, job_id: Uuid, status: ExportStatus, now_ms: u64) -> Result<()> {
        let mut inner = self.inner.write();
        self.finish_job_inner(&mut inner, job_id, status, now_ms)
    }

    fn finish_job_inner(
        &self,
        inner: &mut Inner,
        job_id: Uuid,
        status: ExportStatus,
        now_ms: u64,
    ) -> Result<()> {
        let (mut job, _progress) = inner.running.remove(&job_id).ok_or_else(|| {
            CollabError::InvalidOperation(format!("running job {job_id} not found"))
        })?;
        job.status = status;
        job.finished_at_ms = Some(now_ms);
        let project_id = job.project_id;
        inner.history.entry(project_id).or_default().push(job);
        let max = self.config.history_per_project;
        inner.trim_history(project_id, max);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------

    /// Return a snapshot of all queued (pending) jobs.
    pub fn queued_jobs(&self) -> Vec<ExportJob> {
        self.inner.read().queue.iter().cloned().collect()
    }

    /// Return a snapshot of all currently running jobs.
    pub fn running_jobs(&self) -> Vec<ExportJob> {
        self.inner
            .read()
            .running
            .values()
            .map(|(j, _)| j.clone())
            .collect()
    }

    /// Return the progress tracker for a running job.
    pub fn progress(&self, job_id: Uuid) -> Option<Arc<ExportProgress>> {
        self.inner
            .read()
            .running
            .get(&job_id)
            .map(|(_, p)| p.clone())
    }

    /// Return the export history for a project (completed / failed / cancelled).
    pub fn project_history(&self, project_id: Uuid) -> Vec<ExportJob> {
        self.inner
            .read()
            .history
            .get(&project_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Total number of queued jobs.
    pub fn queue_len(&self) -> usize {
        self.inner.read().queue_len()
    }

    /// Number of currently running jobs.
    pub fn running_count(&self) -> usize {
        self.inner.read().running_count()
    }

    /// Summary statistics for a project.
    pub fn project_stats(&self, project_id: Uuid) -> ProjectExportStats {
        let inner = self.inner.read();
        let queued = inner
            .queue
            .iter()
            .filter(|j| j.project_id == project_id)
            .count();
        let running = inner
            .running
            .values()
            .filter(|(j, _)| j.project_id == project_id)
            .count();
        let history = inner.history.get(&project_id);
        let completed = history
            .map(|h| {
                h.iter()
                    .filter(|j| j.status == ExportStatus::Completed)
                    .count()
            })
            .unwrap_or(0);
        let failed = history
            .map(|h| {
                h.iter()
                    .filter(|j| matches!(j.status, ExportStatus::Failed(_)))
                    .count()
            })
            .unwrap_or(0);
        let cancelled = history
            .map(|h| {
                h.iter()
                    .filter(|j| j.status == ExportStatus::Cancelled)
                    .count()
            })
            .unwrap_or(0);
        ProjectExportStats {
            project_id,
            queued,
            running,
            completed,
            failed,
            cancelled,
        }
    }
}

impl Default for ExportCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ProjectExportStats
// ---------------------------------------------------------------------------

/// Export statistics for a single project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectExportStats {
    pub project_id: Uuid,
    pub queued: usize,
    pub running: usize,
    pub completed: usize,
    pub failed: usize,
    pub cancelled: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> u64 {
        1_700_000_000_000u64
    }

    fn project() -> Uuid {
        Uuid::parse_str("aaaaaaaa-0000-0000-0000-000000000001").expect("valid UUID literal")
    }

    fn user_alice() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").expect("valid UUID literal")
    }

    fn session() -> Uuid {
        Uuid::parse_str("bbbbbbbb-0000-0000-0000-000000000001").expect("valid UUID literal")
    }

    fn simple_job() -> ExportJob {
        ExportJob::new(
            project(),
            session(),
            user_alice(),
            "Alice",
            ExportFormat::H264,
            ExportRange::full(),
            ExportSettings::default(),
            "test export",
            now(),
        )
    }

    #[test]
    fn test_submit_and_queue_length() {
        let coord = ExportCoordinator::new();
        let job = simple_job();
        let job_id = coord.submit(job).expect("submit should succeed with empty queue");
        assert_eq!(coord.queue_len(), 1);
        assert!(coord.queued_jobs().iter().any(|j| j.id == job_id));
    }

    #[test]
    fn test_start_next_transitions_to_running() {
        let coord = ExportCoordinator::new();
        coord.submit(simple_job()).expect("submit should succeed");
        assert_eq!(coord.queue_len(), 1);

        let result = coord.start_next(now());
        assert!(result.is_some());
        assert_eq!(coord.queue_len(), 0);
        assert_eq!(coord.running_count(), 1);
    }

    #[test]
    fn test_complete_job() {
        let coord = ExportCoordinator::new();
        let job_id = coord.submit(simple_job()).expect("submit should succeed");
        let (running_job, _) = coord.start_next(now()).expect("start_next should return the queued job");
        assert_eq!(running_job.id, job_id);

        coord.complete(job_id, now() + 1000).expect("complete should succeed for running job");
        assert_eq!(coord.running_count(), 0);

        let history = coord.project_history(project());
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, ExportStatus::Completed);
    }

    #[test]
    fn test_fail_job() {
        let coord = ExportCoordinator::new();
        let job_id = coord.submit(simple_job()).expect("submit should succeed");
        coord.start_next(now()).expect("start_next should dequeue job");

        coord.fail(job_id, "codec error", now() + 500).expect("fail should succeed for running job");

        let history = coord.project_history(project());
        assert!(matches!(history[0].status, ExportStatus::Failed(_)));
    }

    #[test]
    fn test_cancel_queued_job() {
        let coord = ExportCoordinator::new();
        let job_id = coord.submit(simple_job()).expect("submit should succeed");

        coord.cancel(job_id, now()).expect("cancel should succeed for queued job");
        assert_eq!(coord.queue_len(), 0);
        let history = coord.project_history(project());
        assert_eq!(history[0].status, ExportStatus::Cancelled);
    }

    #[test]
    fn test_cancel_running_job() {
        let coord = ExportCoordinator::new();
        let job_id = coord.submit(simple_job()).expect("submit should succeed");
        coord.start_next(now()).expect("start_next should dequeue job");

        coord.cancel(job_id, now() + 200).expect("cancel should succeed for running job");
        assert_eq!(coord.running_count(), 0);
        let history = coord.project_history(project());
        assert_eq!(history[0].status, ExportStatus::Cancelled);
    }

    #[test]
    fn test_duplicate_project_export_rejected() {
        let coord = ExportCoordinator::new();
        let j1 = simple_job();
        let j2 = simple_job(); // same project
        coord.submit(j1).expect("submit j1 should succeed");
        coord.start_next(now()).expect("start_next should dequeue j1"); // j1 now Running

        // Submitting j2 for the same project should fail.
        let result = coord.submit(j2);
        assert!(matches!(result, Err(CollabError::LockFailed(_))));
    }

    #[test]
    fn test_export_progress_tracking() {
        let coord = ExportCoordinator::new();
        let job_id = coord.submit(simple_job()).expect("submit should succeed");
        coord.start_next(now()).expect("start_next should dequeue job");

        let progress = coord.progress(job_id).expect("progress should be available for running job");
        progress.update(42, Some(30), "encoding audio");
        assert_eq!(progress.percent(), 42);
        assert_eq!(progress.eta_secs(), Some(30));
        assert_eq!(progress.message(), "encoding audio");
    }

    #[test]
    fn test_project_stats() {
        let coord = ExportCoordinator::new();
        let j1 = simple_job();
        let j2 = simple_job();

        let id1 = coord.submit(j1).expect("submit j1 should succeed");
        coord.submit(j2).expect("submit j2 should succeed"); // queued

        coord.start_next(now()).expect("start_next should dequeue j1"); // id1 now running
        coord.complete(id1, now() + 1000).expect("complete should succeed for running job");

        let stats = coord.project_stats(project());
        assert_eq!(stats.queued, 1);
        assert_eq!(stats.running, 0);
        assert_eq!(stats.completed, 1);
    }

    #[test]
    fn test_export_range_validation() {
        assert!(ExportRange::partial(0, 5000).is_ok());
        assert!(ExportRange::partial(5000, 1000).is_err()); // end < start
        assert!(ExportRange::partial(1000, 1000).is_err()); // end == start
    }

    #[test]
    fn test_export_status_terminal() {
        assert!(ExportStatus::Completed.is_terminal());
        assert!(ExportStatus::Cancelled.is_terminal());
        assert!(ExportStatus::Failed("err".into()).is_terminal());
        assert!(!ExportStatus::Queued.is_terminal());
        assert!(!ExportStatus::Running.is_terminal());
    }
}
