#![allow(dead_code)]
//! Render job queue management for Python bindings.
//!
//! Provides a priority-based render queue that Python callers can use to
//! schedule, cancel, query, and reorder media render operations.

use std::collections::HashMap;

/// Priority level for render jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RenderPriority {
    /// Background / low priority.
    Low,
    /// Normal priority (default).
    Normal,
    /// High priority.
    High,
    /// Critical / rush priority.
    Critical,
}

/// Current state of a render job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobState {
    /// Waiting in the queue.
    Queued,
    /// Currently rendering.
    Running,
    /// Completed successfully.
    Completed,
    /// Failed with an error.
    Failed,
    /// Cancelled by the user.
    Cancelled,
    /// Paused.
    Paused,
}

/// Unique identifier for a render job.
pub type JobId = u64;

/// Output format specification for a render job.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputSpec {
    /// Output file path or URI.
    pub path: String,
    /// Container format (e.g. "mp4", "mkv", "webm").
    pub container: String,
    /// Video codec identifier (e.g. "av1", "vp9").
    pub video_codec: Option<String>,
    /// Audio codec identifier (e.g. "opus", "flac").
    pub audio_codec: Option<String>,
    /// Target width in pixels.
    pub width: Option<u32>,
    /// Target height in pixels.
    pub height: Option<u32>,
    /// Target bitrate in kbps.
    pub bitrate_kbps: Option<u32>,
}

/// A single render job in the queue.
#[derive(Debug, Clone)]
pub struct RenderJob {
    /// Unique job identifier.
    pub id: JobId,
    /// Human-readable job name.
    pub name: String,
    /// Priority level.
    pub priority: RenderPriority,
    /// Current state.
    pub state: JobState,
    /// Output specification.
    pub output: OutputSpec,
    /// Progress percentage (0.0 .. 100.0).
    progress_pct: f64,
    /// Error message if state is Failed.
    pub error: Option<String>,
    /// Tags for grouping / filtering.
    pub tags: Vec<String>,
}

impl RenderJob {
    /// Create a new queued job.
    pub fn new(id: JobId, name: impl Into<String>, output: OutputSpec) -> Self {
        Self {
            id,
            name: name.into(),
            priority: RenderPriority::Normal,
            state: JobState::Queued,
            output,
            progress_pct: 0.0,
            error: None,
            tags: Vec::new(),
        }
    }

    /// Get current progress percentage.
    pub fn progress(&self) -> f64 {
        self.progress_pct
    }

    /// Set progress percentage, clamped to 0..=100.
    #[allow(clippy::cast_precision_loss)]
    pub fn set_progress(&mut self, pct: f64) {
        self.progress_pct = pct.clamp(0.0, 100.0);
    }

    /// Whether the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            JobState::Completed | JobState::Failed | JobState::Cancelled
        )
    }

    /// Mark the job as running.
    pub fn start(&mut self) -> Result<(), RenderQueueError> {
        if self.state != JobState::Queued && self.state != JobState::Paused {
            return Err(RenderQueueError::InvalidTransition {
                from: self.state,
                to: JobState::Running,
            });
        }
        self.state = JobState::Running;
        Ok(())
    }

    /// Mark the job as completed.
    pub fn complete(&mut self) -> Result<(), RenderQueueError> {
        if self.state != JobState::Running {
            return Err(RenderQueueError::InvalidTransition {
                from: self.state,
                to: JobState::Completed,
            });
        }
        self.state = JobState::Completed;
        self.progress_pct = 100.0;
        Ok(())
    }

    /// Mark the job as failed with an error message.
    pub fn fail(&mut self, msg: impl Into<String>) -> Result<(), RenderQueueError> {
        if self.state != JobState::Running {
            return Err(RenderQueueError::InvalidTransition {
                from: self.state,
                to: JobState::Failed,
            });
        }
        self.state = JobState::Failed;
        self.error = Some(msg.into());
        Ok(())
    }

    /// Cancel the job if it is not in a terminal state.
    pub fn cancel(&mut self) -> Result<(), RenderQueueError> {
        if self.is_terminal() {
            return Err(RenderQueueError::InvalidTransition {
                from: self.state,
                to: JobState::Cancelled,
            });
        }
        self.state = JobState::Cancelled;
        Ok(())
    }
}

/// Errors from render queue operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RenderQueueError {
    /// Job not found by ID.
    JobNotFound(JobId),
    /// Invalid state transition.
    InvalidTransition {
        /// Current state.
        from: JobState,
        /// Target state.
        to: JobState,
    },
    /// Queue is empty when an operation requires at least one job.
    QueueEmpty,
}

impl std::fmt::Display for RenderQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JobNotFound(id) => write!(f, "job {id} not found"),
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition {from:?} -> {to:?}")
            }
            Self::QueueEmpty => write!(f, "queue is empty"),
        }
    }
}

impl std::error::Error for RenderQueueError {}

/// Queue statistics summary.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total jobs ever enqueued.
    pub total_enqueued: u64,
    /// Currently queued jobs.
    pub queued: usize,
    /// Currently running jobs.
    pub running: usize,
    /// Completed jobs.
    pub completed: usize,
    /// Failed jobs.
    pub failed: usize,
    /// Cancelled jobs.
    pub cancelled: usize,
}

/// A priority-based render job queue.
#[derive(Debug, Default)]
pub struct RenderQueue {
    /// All jobs, keyed by ID.
    jobs: HashMap<JobId, RenderJob>,
    /// Next job ID.
    next_id: JobId,
    /// Maximum concurrent running jobs.
    max_concurrent: usize,
}

impl RenderQueue {
    /// Create a new empty render queue with default concurrency of 1.
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            next_id: 1,
            max_concurrent: 1,
        }
    }

    /// Create a queue with a specific concurrency limit.
    pub fn with_concurrency(max: usize) -> Self {
        Self {
            jobs: HashMap::new(),
            next_id: 1,
            max_concurrent: max.max(1),
        }
    }

    /// Enqueue a new job and return its ID.
    pub fn enqueue(&mut self, name: impl Into<String>, output: OutputSpec) -> JobId {
        let id = self.next_id;
        self.next_id += 1;
        let job = RenderJob::new(id, name, output);
        self.jobs.insert(id, job);
        id
    }

    /// Enqueue a job with a specific priority.
    pub fn enqueue_with_priority(
        &mut self,
        name: impl Into<String>,
        output: OutputSpec,
        priority: RenderPriority,
    ) -> JobId {
        let id = self.enqueue(name, output);
        if let Some(job) = self.jobs.get_mut(&id) {
            job.priority = priority;
        }
        id
    }

    /// Get a reference to a job.
    pub fn get(&self, id: JobId) -> Option<&RenderJob> {
        self.jobs.get(&id)
    }

    /// Get a mutable reference to a job.
    pub fn get_mut(&mut self, id: JobId) -> Option<&mut RenderJob> {
        self.jobs.get_mut(&id)
    }

    /// Cancel a job by ID.
    pub fn cancel(&mut self, id: JobId) -> Result<(), RenderQueueError> {
        let job = self
            .jobs
            .get_mut(&id)
            .ok_or(RenderQueueError::JobNotFound(id))?;
        job.cancel()
    }

    /// Number of jobs in the queue (all states).
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Return the next job that should run (highest priority queued job).
    pub fn next_runnable(&self) -> Option<JobId> {
        let running = self
            .jobs
            .values()
            .filter(|j| j.state == JobState::Running)
            .count();
        if running >= self.max_concurrent {
            return None;
        }
        self.jobs
            .values()
            .filter(|j| j.state == JobState::Queued)
            .max_by_key(|j| j.priority)
            .map(|j| j.id)
    }

    /// Compute queue statistics.
    pub fn stats(&self) -> QueueStats {
        let mut s = QueueStats::default();
        s.total_enqueued = self.next_id.saturating_sub(1);
        for job in self.jobs.values() {
            match job.state {
                JobState::Queued | JobState::Paused => s.queued += 1,
                JobState::Running => s.running += 1,
                JobState::Completed => s.completed += 1,
                JobState::Failed => s.failed += 1,
                JobState::Cancelled => s.cancelled += 1,
            }
        }
        s
    }

    /// Remove all jobs in terminal states and return how many were removed.
    pub fn purge_completed(&mut self) -> usize {
        let before = self.jobs.len();
        self.jobs.retain(|_, j| !j.is_terminal());
        before - self.jobs.len()
    }

    /// List job IDs filtered by state.
    pub fn list_by_state(&self, state: JobState) -> Vec<JobId> {
        self.jobs
            .values()
            .filter(|j| j.state == state)
            .map(|j| j.id)
            .collect()
    }

    /// Maximum concurrent jobs.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_output() -> OutputSpec {
        OutputSpec {
            path: std::env::temp_dir()
                .join("oximedia-py-rq-out.mp4")
                .to_string_lossy()
                .into_owned(),
            container: "mp4".into(),
            video_codec: Some("av1".into()),
            audio_codec: Some("opus".into()),
            width: Some(1920),
            height: Some(1080),
            bitrate_kbps: Some(5000),
        }
    }

    #[test]
    fn test_enqueue_and_get() {
        let mut q = RenderQueue::new();
        let id = q.enqueue("job1", sample_output());
        assert_eq!(q.len(), 1);
        let job = q.get(id).expect("job should be valid");
        assert_eq!(job.name, "job1");
        assert_eq!(job.state, JobState::Queued);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(RenderPriority::Low < RenderPriority::Normal);
        assert!(RenderPriority::Normal < RenderPriority::High);
        assert!(RenderPriority::High < RenderPriority::Critical);
    }

    #[test]
    fn test_next_runnable_by_priority() {
        let mut q = RenderQueue::new();
        let _low = q.enqueue_with_priority("low", sample_output(), RenderPriority::Low);
        let high = q.enqueue_with_priority("high", sample_output(), RenderPriority::High);
        assert_eq!(q.next_runnable(), Some(high));
    }

    #[test]
    fn test_job_lifecycle() {
        let mut job = RenderJob::new(1, "test", sample_output());
        assert!(!job.is_terminal());
        job.start().expect("start should succeed");
        assert_eq!(job.state, JobState::Running);
        job.set_progress(50.0);
        assert!((job.progress() - 50.0).abs() < f64::EPSILON);
        job.complete().expect("complete should succeed");
        assert!(job.is_terminal());
        assert!((job.progress() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_failure() {
        let mut job = RenderJob::new(1, "fail_test", sample_output());
        job.start().expect("start should succeed");
        job.fail("codec error").expect("fail should succeed");
        assert_eq!(job.state, JobState::Failed);
        assert_eq!(job.error.as_deref(), Some("codec error"));
    }

    #[test]
    fn test_cancel_queued() {
        let mut job = RenderJob::new(1, "cancel_test", sample_output());
        job.cancel().expect("cancel should succeed");
        assert_eq!(job.state, JobState::Cancelled);
    }

    #[test]
    fn test_cancel_completed_fails() {
        let mut job = RenderJob::new(1, "done", sample_output());
        job.start().expect("start should succeed");
        job.complete().expect("complete should succeed");
        let err = job.cancel().unwrap_err();
        assert!(matches!(err, RenderQueueError::InvalidTransition { .. }));
    }

    #[test]
    fn test_queue_cancel_not_found() {
        let mut q = RenderQueue::new();
        let err = q.cancel(999).unwrap_err();
        assert!(matches!(err, RenderQueueError::JobNotFound(999)));
    }

    #[test]
    fn test_concurrency_limit() {
        let mut q = RenderQueue::with_concurrency(1);
        let id1 = q.enqueue("j1", sample_output());
        let _id2 = q.enqueue("j2", sample_output());
        q.get_mut(id1).expect("get_mut should succeed").start().expect("test expectation failed");
        assert!(q.next_runnable().is_none());
    }

    #[test]
    fn test_purge_completed() {
        let mut q = RenderQueue::new();
        let id1 = q.enqueue("j1", sample_output());
        let _id2 = q.enqueue("j2", sample_output());
        q.get_mut(id1).expect("get_mut should succeed").start().expect("test expectation failed");
        q.get_mut(id1).expect("get_mut should succeed").complete().expect("test expectation failed");
        let purged = q.purge_completed();
        assert_eq!(purged, 1);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut q = RenderQueue::new();
        let id1 = q.enqueue("j1", sample_output());
        let _id2 = q.enqueue("j2", sample_output());
        q.get_mut(id1).expect("get_mut should succeed").start().expect("test expectation failed");
        let s = q.stats();
        assert_eq!(s.running, 1);
        assert_eq!(s.queued, 1);
        assert_eq!(s.total_enqueued, 2);
    }

    #[test]
    fn test_list_by_state() {
        let mut q = RenderQueue::new();
        let _id1 = q.enqueue("j1", sample_output());
        let _id2 = q.enqueue("j2", sample_output());
        let queued = q.list_by_state(JobState::Queued);
        assert_eq!(queued.len(), 2);
    }

    #[test]
    fn test_progress_clamp() {
        let mut job = RenderJob::new(1, "clamp", sample_output());
        job.set_progress(150.0);
        assert!((job.progress() - 100.0).abs() < f64::EPSILON);
        job.set_progress(-10.0);
        assert!((job.progress()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_render_queue_error_display() {
        let e = RenderQueueError::QueueEmpty;
        assert!(e.to_string().contains("empty"));
        let e2 = RenderQueueError::JobNotFound(42);
        assert!(e2.to_string().contains("42"));
    }
}
