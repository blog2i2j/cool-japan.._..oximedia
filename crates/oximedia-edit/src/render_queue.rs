//! Render queue for batch export of multiple timelines.
//!
//! [`RenderQueue`] holds a priority-ordered collection of [`RenderJob`]s and
//! exposes a simple producer/consumer API:
//!
//! - [`RenderQueue::add_job`] — enqueue a job (highest priority runs first).
//! - [`RenderQueue::next_job`] — dequeue the highest-priority pending job.
//! - [`RenderQueue::remove_job`] — cancel a job by ID regardless of status.
//! - [`RenderQueue::update_status`] — transition a job through its lifecycle.
//! - Query methods: `len`, `is_empty`, `get_job`, `jobs_with_status`.
//!
//! # Priority ordering
//!
//! Jobs with a **higher** `priority` value are dequeued first.  Ties are
//! broken by insertion order (FIFO within the same priority level).

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// JobId
// ─────────────────────────────────────────────────────────────────────────────

/// Unique identifier for a render job.
pub type JobId = u64;

// ─────────────────────────────────────────────────────────────────────────────
// JobStatus
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a render job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobStatus {
    /// Waiting to be picked up by a worker.
    Pending,
    /// Currently being processed by a worker.
    Running,
    /// Successfully completed.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled before it started or while running.
    Cancelled,
}

impl JobStatus {
    /// Returns `true` if the job has reached a terminal state and will not be
    /// dequeued again.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// Returns `true` if the job can still be picked up by a worker.
    #[must_use]
    pub fn is_pending(self) -> bool {
        matches!(self, Self::Pending)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExportConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single export operation.
///
/// Keeps all fields public so callers can customise freely; none of the queue
/// logic depends on the specific values — this is an opaque payload from the
/// queue's perspective.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Output file path (as a string; no I/O is performed by the queue).
    pub output_path: String,
    /// Desired output width in pixels.
    pub width: u32,
    /// Desired output height in pixels.
    pub height: u32,
    /// Target frames per second.
    pub fps: f64,
    /// Whether to export video.
    pub export_video: bool,
    /// Whether to export audio.
    pub export_audio: bool,
    /// Free-form codec / preset label (e.g. `"av1-crf28"`, `"opus-128k"`).
    pub codec_preset: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            output_path: String::new(),
            width: 1920,
            height: 1080,
            fps: 30.0,
            export_video: true,
            export_audio: true,
            codec_preset: "default".to_string(),
        }
    }
}

impl ExportConfig {
    /// Create a new `ExportConfig` with the given output path.
    #[must_use]
    pub fn new(output_path: impl Into<String>) -> Self {
        Self {
            output_path: output_path.into(),
            ..Default::default()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimelineSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight snapshot of a timeline's key metadata.
///
/// In a real implementation this would hold a deep-clone of the timeline (or
/// an Arc reference to an immutable version).  For the queue's purposes we
/// store just the fields needed to describe the job.
#[derive(Debug, Clone)]
pub struct TimelineSnapshot {
    /// Human-readable name of the timeline.
    pub name: String,
    /// Total duration in milliseconds.
    pub duration_ms: i64,
    /// Number of tracks.
    pub track_count: usize,
    /// Total number of clips across all tracks.
    pub clip_count: usize,
}

impl TimelineSnapshot {
    /// Create a `TimelineSnapshot` with the given metadata.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        duration_ms: i64,
        track_count: usize,
        clip_count: usize,
    ) -> Self {
        Self {
            name: name.into(),
            duration_ms,
            track_count,
            clip_count,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderJob
// ─────────────────────────────────────────────────────────────────────────────

/// A single export task in the render queue.
#[derive(Debug, Clone)]
pub struct RenderJob {
    /// Unique job identifier (assigned by the queue).
    pub id: JobId,
    /// Snapshot of the timeline to render.
    pub timeline_snapshot: TimelineSnapshot,
    /// Export configuration.
    pub export_config: ExportConfig,
    /// Priority: higher values are dequeued first.
    pub priority: i32,
    /// Current lifecycle status.
    pub status: JobStatus,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Insertion sequence number — used to break priority ties (lower = earlier).
    pub(crate) seq: u64,
    /// Optional error message set when status transitions to `Failed`.
    pub error_message: Option<String>,
    /// Progress percentage (0–100), updated by the worker.
    pub progress: u8,
}

impl RenderJob {
    /// Create a new `RenderJob`.
    ///
    /// The `id` and `seq` fields are assigned by the owning [`RenderQueue`];
    /// do not set them manually.
    #[must_use]
    pub fn new(
        timeline_snapshot: TimelineSnapshot,
        export_config: ExportConfig,
        priority: i32,
    ) -> Self {
        Self {
            id: 0,
            timeline_snapshot,
            export_config,
            priority,
            status: JobStatus::Pending,
            label: None,
            seq: 0,
            error_message: None,
            progress: 0,
        }
    }

    /// Attach a human-readable label to the job.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns `true` if this job is pending (eligible for `next_job`).
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.status.is_pending()
    }

    /// Returns `true` if this job has finished (any terminal state).
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.status.is_terminal()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RenderQueue
// ─────────────────────────────────────────────────────────────────────────────

/// A priority-based render queue for batch export.
///
/// Jobs are stored in insertion order internally; `next_job` performs an O(n)
/// scan to find the highest-priority pending job.  For typical queue depths
/// (tens to low hundreds of jobs) this is fast enough.  The design is
/// deliberately simple so it can be wrapped in a `Mutex` or `RwLock` without
/// complex data-structure gymnastics.
#[derive(Debug, Default)]
pub struct RenderQueue {
    jobs: Vec<RenderJob>,
    next_id: JobId,
    next_seq: u64,
}

impl RenderQueue {
    /// Create a new empty render queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            next_id: 1,
            next_seq: 0,
        }
    }

    /// Enqueue a job and return its assigned [`JobId`].
    ///
    /// The job's `id`, `seq`, and `status` fields are overwritten by the queue.
    pub fn add_job(&mut self, mut job: RenderJob) -> JobId {
        let id = self.next_id;
        self.next_id += 1;
        job.id = id;
        job.seq = self.next_seq;
        self.next_seq += 1;
        job.status = JobStatus::Pending;
        self.jobs.push(job);
        id
    }

    /// Remove a job by ID (any status) and return it, or `None` if not found.
    pub fn remove_job(&mut self, id: JobId) -> Option<RenderJob> {
        if let Some(pos) = self.jobs.iter().position(|j| j.id == id) {
            Some(self.jobs.remove(pos))
        } else {
            None
        }
    }

    /// Return a reference to the job with the given ID, or `None`.
    #[must_use]
    pub fn get_job(&self, id: JobId) -> Option<&RenderJob> {
        self.jobs.iter().find(|j| j.id == id)
    }

    /// Return a mutable reference to the job with the given ID, or `None`.
    pub fn get_job_mut(&mut self, id: JobId) -> Option<&mut RenderJob> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }

    /// Dequeue the highest-priority pending job, transitioning it to `Running`.
    ///
    /// Returns `None` if there are no pending jobs.
    ///
    /// Ties in priority are broken by insertion order (earliest first).
    pub fn next_job(&mut self) -> Option<JobId> {
        // Find the pending job with the highest (priority, then lowest seq).
        let best_pos = self
            .jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| j.status.is_pending())
            .max_by(|(_, a), (_, b)| {
                a.priority.cmp(&b.priority).then_with(|| b.seq.cmp(&a.seq)) // lower seq = earlier = preferred
            })
            .map(|(i, _)| i);

        let pos = best_pos?;
        self.jobs[pos].status = JobStatus::Running;
        Some(self.jobs[pos].id)
    }

    /// Update the status of a job.
    ///
    /// Returns `true` if the job was found and the transition is valid:
    ///
    /// - `Running` → `Completed` / `Failed` / `Cancelled`
    /// - `Pending`  → `Cancelled`
    ///
    /// Invalid or unknown transitions are silently ignored (returns `false`).
    pub fn update_status(&mut self, id: JobId, new_status: JobStatus) -> bool {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            let valid = matches!(
                (job.status, new_status),
                (JobStatus::Running, JobStatus::Completed)
                    | (JobStatus::Running, JobStatus::Failed)
                    | (JobStatus::Running, JobStatus::Cancelled)
                    | (JobStatus::Pending, JobStatus::Cancelled)
            );
            if valid {
                job.status = new_status;
                return true;
            }
        }
        false
    }

    /// Convenience: mark a running job as completed with a final progress of 100.
    pub fn complete_job(&mut self, id: JobId) -> bool {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            if job.status == JobStatus::Running {
                job.status = JobStatus::Completed;
                job.progress = 100;
                return true;
            }
        }
        false
    }

    /// Convenience: mark a running job as failed with an error message.
    pub fn fail_job(&mut self, id: JobId, error: impl Into<String>) -> bool {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            if job.status == JobStatus::Running {
                job.status = JobStatus::Failed;
                job.error_message = Some(error.into());
                return true;
            }
        }
        false
    }

    /// Update the progress of a running job (clamped to 0–100).
    pub fn set_progress(&mut self, id: JobId, progress: u8) -> bool {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            if job.status == JobStatus::Running {
                job.progress = progress.min(100);
                return true;
            }
        }
        false
    }

    /// Number of jobs in the queue (all statuses).
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Returns `true` if the queue contains no jobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Number of pending jobs.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.jobs.iter().filter(|j| j.is_pending()).count()
    }

    /// All jobs with the given status (in insertion order).
    #[must_use]
    pub fn jobs_with_status(&self, status: JobStatus) -> Vec<&RenderJob> {
        self.jobs.iter().filter(|j| j.status == status).collect()
    }

    /// Status counts for all known states.
    #[must_use]
    pub fn status_counts(&self) -> HashMap<JobStatus, usize> {
        let mut map = HashMap::new();
        for j in &self.jobs {
            *map.entry(j.status).or_insert(0) += 1;
        }
        map
    }

    /// Remove all jobs that have reached a terminal state.
    pub fn purge_completed(&mut self) {
        self.jobs.retain(|j| !j.status.is_terminal());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(priority: i32) -> RenderJob {
        RenderJob::new(
            TimelineSnapshot::new("test", 5000, 2, 4),
            ExportConfig::new("/tmp/out.mkv"),
            priority,
        )
    }

    // ── basic add/remove ──────────────────────────────────────────────────

    #[test]
    fn test_add_job_assigns_sequential_ids() {
        let mut q = RenderQueue::new();
        let id1 = q.add_job(make_job(0));
        let id2 = q.add_job(make_job(0));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_add_job_len_increases() {
        let mut q = RenderQueue::new();
        assert_eq!(q.len(), 0);
        q.add_job(make_job(0));
        assert_eq!(q.len(), 1);
        q.add_job(make_job(0));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_remove_job_returns_correct_job() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(5));
        let removed = q.remove_job(id);
        assert!(removed.is_some());
        assert_eq!(removed.as_ref().map(|j| j.id), Some(id));
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_remove_nonexistent_job_returns_none() {
        let mut q = RenderQueue::new();
        assert!(q.remove_job(99).is_none());
    }

    #[test]
    fn test_get_job_found() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(1));
        assert!(q.get_job(id).is_some());
    }

    #[test]
    fn test_get_job_not_found() {
        let q = RenderQueue::new();
        assert!(q.get_job(42).is_none());
    }

    // ── next_job priority ordering ────────────────────────────────────────

    #[test]
    fn test_next_job_returns_highest_priority() {
        let mut q = RenderQueue::new();
        let _low_id = q.add_job(make_job(1));
        let high_id = q.add_job(make_job(10));
        let _med_id = q.add_job(make_job(5));

        let next = q.next_job().expect("should have a next job");
        assert_eq!(next, high_id);
    }

    #[test]
    fn test_next_job_fifo_within_same_priority() {
        let mut q = RenderQueue::new();
        let first = q.add_job(make_job(3));
        let _second = q.add_job(make_job(3));

        // Complete the first so we can check second
        q.next_job(); // picks first (earlier seq)
        q.complete_job(first);

        let next = q.next_job().expect("should have second");
        assert_eq!(next, _second);
    }

    #[test]
    fn test_next_job_returns_none_when_empty() {
        let mut q = RenderQueue::new();
        assert!(q.next_job().is_none());
    }

    #[test]
    fn test_next_job_transitions_status_to_running() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        let job = q.get_job(id).expect("job should exist");
        assert_eq!(job.status, JobStatus::Running);
    }

    // ── status transitions ────────────────────────────────────────────────

    #[test]
    fn test_complete_job() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        assert!(q.complete_job(id));
        assert_eq!(q.get_job(id).map(|j| j.status), Some(JobStatus::Completed));
        assert_eq!(q.get_job(id).map(|j| j.progress), Some(100));
    }

    #[test]
    fn test_fail_job_sets_error_message() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        assert!(q.fail_job(id, "codec error"));
        let job = q.get_job(id).expect("job should exist");
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error_message.as_deref(), Some("codec error"));
    }

    #[test]
    fn test_cancel_pending_job() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        assert!(q.update_status(id, JobStatus::Cancelled));
        assert_eq!(q.get_job(id).map(|j| j.status), Some(JobStatus::Cancelled));
        // Cancelled job should not be returned by next_job
        assert!(q.next_job().is_none());
    }

    #[test]
    fn test_update_status_invalid_transition_rejected() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        // Pending → Completed is not valid (must go through Running first)
        assert!(!q.update_status(id, JobStatus::Completed));
    }

    // ── set_progress ──────────────────────────────────────────────────────

    #[test]
    fn test_set_progress_clamps_to_100() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        q.set_progress(id, 200);
        assert_eq!(q.get_job(id).map(|j| j.progress), Some(100));
    }

    #[test]
    fn test_set_progress_on_pending_job_fails() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        assert!(!q.set_progress(id, 50));
    }

    // ── status query helpers ───────────────────────────────────────────────

    #[test]
    fn test_pending_count() {
        let mut q = RenderQueue::new();
        q.add_job(make_job(0));
        q.add_job(make_job(0));
        let _id = q.add_job(make_job(0));
        q.next_job(); // id 1 → Running
        assert_eq!(q.pending_count(), 2);
    }

    #[test]
    fn test_jobs_with_status() {
        let mut q = RenderQueue::new();
        let id1 = q.add_job(make_job(1));
        let id2 = q.add_job(make_job(2));
        q.next_job(); // id2 running (highest priority)
        q.complete_job(id2);
        let completed = q.jobs_with_status(JobStatus::Completed);
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].id, id2);
        let pending = q.jobs_with_status(JobStatus::Pending);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, id1);
    }

    #[test]
    fn test_purge_completed_removes_terminal() {
        let mut q = RenderQueue::new();
        let id1 = q.add_job(make_job(0));
        let id2 = q.add_job(make_job(0));
        q.next_job(); // id1
        q.complete_job(id1);
        q.purge_completed();
        assert_eq!(q.len(), 1);
        assert!(q.get_job(id1).is_none());
        assert!(q.get_job(id2).is_some());
    }

    #[test]
    fn test_status_counts() {
        let mut q = RenderQueue::new();
        let _id1 = q.add_job(make_job(0));
        let _id2 = q.add_job(make_job(1));
        q.add_job(make_job(2));
        q.next_job(); // id3 → Running
        q.next_job(); // id2 → Running

        let counts = q.status_counts();
        assert_eq!(counts.get(&JobStatus::Running), Some(&2));
        assert_eq!(counts.get(&JobStatus::Pending), Some(&1));
    }

    // ── label / metadata ──────────────────────────────────────────────────

    #[test]
    fn test_job_label() {
        let mut q = RenderQueue::new();
        let job = make_job(0).with_label("4K export");
        let id = q.add_job(job);
        let j = q.get_job(id).expect("job");
        assert_eq!(j.label.as_deref(), Some("4K export"));
    }

    // ── JobStatus helpers ─────────────────────────────────────────────────

    #[test]
    fn test_job_status_is_terminal() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(JobStatus::Cancelled.is_terminal());
        assert!(!JobStatus::Running.is_terminal());
        assert!(!JobStatus::Pending.is_terminal());
    }

    // ── Additional comprehensive tests ────────────────────────────────────

    #[test]
    fn test_export_config_default_resolution() {
        let cfg = ExportConfig::default();
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.fps, 30.0);
    }

    #[test]
    fn test_export_config_new_sets_path() {
        let cfg = ExportConfig::new("/tmp/test_output.webm");
        assert_eq!(cfg.output_path, "/tmp/test_output.webm");
        assert!(cfg.export_video);
        assert!(cfg.export_audio);
    }

    #[test]
    fn test_timeline_snapshot_fields() {
        let snap = TimelineSnapshot::new("My Timeline", 60_000, 3, 12);
        assert_eq!(snap.name, "My Timeline");
        assert_eq!(snap.duration_ms, 60_000);
        assert_eq!(snap.track_count, 3);
        assert_eq!(snap.clip_count, 12);
    }

    #[test]
    fn test_render_job_initial_progress_is_zero() {
        let job = make_job(0);
        assert_eq!(job.progress, 0);
        assert!(job.error_message.is_none());
    }

    #[test]
    fn test_render_job_is_pending_after_creation() {
        let job = make_job(5);
        assert!(job.is_pending());
        assert!(!job.is_done());
    }

    #[test]
    fn test_render_queue_get_job_mut() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        if let Some(job) = q.get_job_mut(id) {
            job.label = Some("mutated".to_string());
        }
        assert_eq!(
            q.get_job(id).and_then(|j| j.label.as_deref()),
            Some("mutated")
        );
    }

    #[test]
    fn test_fail_job_error_message_stored() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        q.fail_job(id, "encoder crashed");
        let job = q.get_job(id).expect("job should exist");
        assert_eq!(job.error_message.as_deref(), Some("encoder crashed"));
        assert_eq!(job.status, JobStatus::Failed);
    }

    #[test]
    fn test_complete_job_sets_progress_100() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job();
        q.set_progress(id, 75);
        q.complete_job(id);
        assert_eq!(q.get_job(id).map(|j| j.progress), Some(100));
    }

    #[test]
    fn test_cancel_running_job_via_update_status() {
        let mut q = RenderQueue::new();
        let id = q.add_job(make_job(0));
        q.next_job(); // → Running
        assert!(q.update_status(id, JobStatus::Cancelled));
        assert_eq!(q.get_job(id).map(|j| j.status), Some(JobStatus::Cancelled));
    }

    #[test]
    fn test_next_job_skips_cancelled_pending() {
        let mut q = RenderQueue::new();
        let id1 = q.add_job(make_job(5));
        let id2 = q.add_job(make_job(3));
        // Cancel the highest priority job before it starts
        q.update_status(id1, JobStatus::Cancelled);
        let next = q.next_job().expect("id2 should be next");
        assert_eq!(next, id2);
    }

    #[test]
    fn test_purge_completed_retains_pending_and_running() {
        let mut q = RenderQueue::new();
        let id_pending = q.add_job(make_job(1));
        let id_running = q.add_job(make_job(2));
        let id_done = q.add_job(make_job(3));

        // Start and complete id_done
        let started = q.next_job().expect("should start id_done (priority 3)");
        assert_eq!(started, id_done);
        q.complete_job(id_done);

        q.purge_completed();
        assert_eq!(q.len(), 2);
        assert!(q.get_job(id_pending).is_some());
        assert!(q.get_job(id_running).is_some());
        assert!(q.get_job(id_done).is_none());
    }

    #[test]
    fn test_status_counts_all_statuses() {
        let mut q = RenderQueue::new();
        // Add enough jobs to exercise every status
        let id_a = q.add_job(make_job(10));
        let id_b = q.add_job(make_job(9));
        let id_c = q.add_job(make_job(8));
        let _id_d = q.add_job(make_job(1)); // stays pending

        q.next_job(); // id_a → Running
        q.complete_job(id_a);

        q.next_job(); // id_b → Running
        q.fail_job(id_b, "oops");

        q.next_job(); // id_c → Running
        q.update_status(id_c, JobStatus::Cancelled);

        let counts = q.status_counts();
        assert_eq!(counts.get(&JobStatus::Completed), Some(&1));
        assert_eq!(counts.get(&JobStatus::Failed), Some(&1));
        assert_eq!(counts.get(&JobStatus::Cancelled), Some(&1));
        assert_eq!(counts.get(&JobStatus::Pending), Some(&1));
    }
}
