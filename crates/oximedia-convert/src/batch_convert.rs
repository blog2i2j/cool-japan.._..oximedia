//! Batch conversion orchestration: job queuing, progress tracking, and error
//! recovery.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, VecDeque};

/// Status of a batch conversion job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BatchJobStatus {
    /// Waiting to be picked up
    Queued,
    /// Currently converting
    Running,
    /// Finished successfully
    Done,
    /// Finished with an error
    Failed,
    /// Skipped (e.g., duplicate)
    Skipped,
}

/// A single job within a batch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BatchJob {
    /// Unique job identifier within the batch
    pub job_id: String,
    /// Input file path
    pub input: String,
    /// Output file path
    pub output: String,
    /// Target format/extension
    pub target_format: String,
    /// Current status
    pub status: BatchJobStatus,
    /// Retry count so far
    pub retries: u32,
    /// Maximum allowed retries
    pub max_retries: u32,
    /// Error message if failed
    pub error: Option<String>,
}

impl BatchJob {
    /// Create a new queued batch job.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
        target_format: impl Into<String>,
        max_retries: u32,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            input: input.into(),
            output: output.into(),
            target_format: target_format.into(),
            status: BatchJobStatus::Queued,
            retries: 0,
            max_retries,
            error: None,
        }
    }

    /// Whether this job can be retried.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.status == BatchJobStatus::Failed && self.retries <= self.max_retries
    }

    /// Mark as running.
    pub fn start(&mut self) {
        self.status = BatchJobStatus::Running;
    }

    /// Mark as done.
    pub fn complete(&mut self) {
        self.status = BatchJobStatus::Done;
    }

    /// Record a failure, incrementing retry counter.
    pub fn fail(&mut self, error: impl Into<String>) {
        self.status = BatchJobStatus::Failed;
        self.retries += 1;
        self.error = Some(error.into());
    }

    /// Reset to queued for retry.
    pub fn reset_for_retry(&mut self) {
        self.status = BatchJobStatus::Queued;
    }
}

/// Progress snapshot for a batch.
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Total jobs in batch
    pub total: usize,
    /// Completed jobs (Done + Skipped)
    pub completed: usize,
    /// Failed jobs (not re-queued)
    pub failed: usize,
    /// Currently running
    pub running: usize,
}

impl BatchProgress {
    /// Progress fraction 0.0–1.0.
    #[must_use]
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        (self.completed + self.failed) as f64 / self.total as f64
    }

    /// Whether the batch is fully processed.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.completed + self.failed >= self.total
    }
}

/// FIFO batch conversion queue with error recovery.
#[derive(Debug, Default)]
pub struct BatchConvertQueue {
    jobs: HashMap<String, BatchJob>,
    order: VecDeque<String>,
}

impl BatchConvertQueue {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enqueue a job.
    pub fn enqueue(&mut self, job: BatchJob) {
        self.order.push_back(job.job_id.clone());
        self.jobs.insert(job.job_id.clone(), job);
    }

    /// Dequeue the next queued job for processing.
    pub fn dequeue(&mut self) -> Option<&mut BatchJob> {
        // Find first Queued job in FIFO order
        let id = self
            .order
            .iter()
            .find(|id| {
                self.jobs
                    .get(*id)
                    .is_some_and(|j| j.status == BatchJobStatus::Queued)
            })?
            .clone();
        self.jobs.get_mut(&id)
    }

    /// Mark a job as running by id.
    pub fn start_job(&mut self, job_id: &str) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.start();
            true
        } else {
            false
        }
    }

    /// Mark a job as done.
    pub fn complete_job(&mut self, job_id: &str) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.complete();
            true
        } else {
            false
        }
    }

    /// Mark a job as failed and auto-requeue if retries remain.
    pub fn fail_job(&mut self, job_id: &str, error: impl Into<String>) -> bool {
        let error = error.into();
        if let Some(job) = self.jobs.get_mut(job_id) {
            job.fail(&error);
            if job.can_retry() {
                job.reset_for_retry();
                // Move to back of queue for retry
                if let Some(pos) = self.order.iter().position(|id| id == job_id) {
                    self.order.remove(pos);
                }
                let id = job.job_id.clone();
                self.order.push_back(id);
            }
            true
        } else {
            false
        }
    }

    /// Current progress snapshot.
    #[must_use]
    pub fn progress(&self) -> BatchProgress {
        let total = self.jobs.len();
        let completed = self
            .jobs
            .values()
            .filter(|j| j.status == BatchJobStatus::Done || j.status == BatchJobStatus::Skipped)
            .count();
        let failed = self
            .jobs
            .values()
            .filter(|j| j.status == BatchJobStatus::Failed)
            .count();
        let running = self
            .jobs
            .values()
            .filter(|j| j.status == BatchJobStatus::Running)
            .count();
        BatchProgress {
            total,
            completed,
            failed,
            running,
        }
    }

    /// Total job count.
    #[must_use]
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    /// Number of queued (not-yet-started) jobs.
    #[must_use]
    pub fn queued_count(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| j.status == BatchJobStatus::Queued)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job(id: &str) -> BatchJob {
        BatchJob::new(id, "/in/file.mov", "/out/file.mp4", "mp4", 2)
    }

    #[test]
    fn test_batch_job_initial_status() {
        let job = make_job("j1");
        assert_eq!(job.status, BatchJobStatus::Queued);
        assert_eq!(job.retries, 0);
    }

    #[test]
    fn test_batch_job_start() {
        let mut job = make_job("j2");
        job.start();
        assert_eq!(job.status, BatchJobStatus::Running);
    }

    #[test]
    fn test_batch_job_complete() {
        let mut job = make_job("j3");
        job.start();
        job.complete();
        assert_eq!(job.status, BatchJobStatus::Done);
    }

    #[test]
    fn test_batch_job_fail_increments_retries() {
        let mut job = make_job("j4");
        job.fail("oops");
        assert_eq!(job.retries, 1);
        assert_eq!(job.status, BatchJobStatus::Failed);
    }

    #[test]
    fn test_batch_job_can_retry() {
        let mut job = make_job("j5");
        job.fail("err");
        assert!(job.can_retry());
        job.fail("err");
        assert!(job.can_retry());
        job.fail("err");
        // max_retries = 2, retries = 3 now → cannot retry
        assert!(!job.can_retry());
    }

    #[test]
    fn test_batch_job_reset_for_retry() {
        let mut job = make_job("j6");
        job.fail("err");
        job.reset_for_retry();
        assert_eq!(job.status, BatchJobStatus::Queued);
    }

    #[test]
    fn test_queue_enqueue_and_len() {
        let mut q = BatchConvertQueue::new();
        q.enqueue(make_job("q1"));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_queue_dequeue_fifo() {
        let mut q = BatchConvertQueue::new();
        q.enqueue(make_job("first"));
        q.enqueue(make_job("second"));
        let id = q.dequeue().map(|j| j.job_id.clone()).unwrap();
        assert_eq!(id, "first");
    }

    #[test]
    fn test_queue_complete_job() {
        let mut q = BatchConvertQueue::new();
        q.enqueue(make_job("c1"));
        q.start_job("c1");
        q.complete_job("c1");
        assert_eq!(q.progress().completed, 1);
    }

    #[test]
    fn test_queue_fail_and_requeue() {
        let mut q = BatchConvertQueue::new();
        q.enqueue(make_job("f1"));
        q.start_job("f1");
        q.fail_job("f1", "error");
        // retry count = 1, max = 2, so re-queued
        assert_eq!(q.queued_count(), 1);
    }

    #[test]
    fn test_queue_fail_exhausted_retries() {
        let mut q = BatchConvertQueue::new();
        q.enqueue(BatchJob::new("ex", "/a", "/b", "mp4", 0));
        q.start_job("ex");
        q.fail_job("ex", "err");
        // max_retries = 0, retries = 1 → not re-queued
        assert_eq!(q.queued_count(), 0);
        assert_eq!(q.progress().failed, 1);
    }

    #[test]
    fn test_batch_progress_fraction() {
        let p = BatchProgress {
            total: 4,
            completed: 2,
            failed: 1,
            running: 1,
        };
        assert!((p.fraction() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_batch_progress_complete() {
        let p = BatchProgress {
            total: 3,
            completed: 2,
            failed: 1,
            running: 0,
        };
        assert!(p.is_complete());
    }

    #[test]
    fn test_queue_missing_job_returns_false() {
        let mut q = BatchConvertQueue::new();
        assert!(!q.start_job("nonexistent"));
        assert!(!q.complete_job("nonexistent"));
        assert!(!q.fail_job("nonexistent", "err"));
    }
}
