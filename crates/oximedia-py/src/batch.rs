//! Python-exposed batch processing types.
//!
//! Provides plain Rust structs for managing batch jobs and queues.
//! These can be wrapped with PyO3 annotations later if needed.

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────
//  PyJobStatus
// ─────────────────────────────────────────────────────────────

/// Current status of a batch job.
#[derive(Debug, Clone)]
pub enum PyJobStatus {
    /// Job is waiting to be picked up.
    Pending,
    /// Job is being processed; `progress` is in [0.0, 1.0].
    Running { progress: f32 },
    /// Job finished successfully.
    Completed { output_path: String },
    /// Job failed with an error message.
    Failed { error: String },
}

impl PyJobStatus {
    /// Returns `true` if the job is pending.
    pub fn is_pending(&self) -> bool {
        matches!(self, PyJobStatus::Pending)
    }

    /// Returns `true` if the job is currently running.
    pub fn is_running(&self) -> bool {
        matches!(self, PyJobStatus::Running { .. })
    }

    /// Returns `true` if the job completed successfully.
    pub fn is_completed(&self) -> bool {
        matches!(self, PyJobStatus::Completed { .. })
    }

    /// Returns `true` if the job failed.
    pub fn is_failed(&self) -> bool {
        matches!(self, PyJobStatus::Failed { .. })
    }
}

// ─────────────────────────────────────────────────────────────
//  PyBatchJob
// ─────────────────────────────────────────────────────────────

/// A single batch processing job.
#[derive(Debug, Clone)]
pub struct PyBatchJob {
    /// Unique job identifier.
    pub id: u64,
    /// List of input file paths to process.
    pub input_paths: Vec<String>,
    /// Directory where output files should be written.
    pub output_dir: String,
    /// Name of the encoding preset to use.
    pub preset: String,
    /// Job priority; higher values are processed first.
    pub priority: i32,
}

impl PyBatchJob {
    /// Construct a new `PyBatchJob`.
    pub fn new(
        id: u64,
        input_paths: Vec<String>,
        output_dir: &str,
        preset: &str,
        priority: i32,
    ) -> Self {
        Self {
            id,
            input_paths,
            output_dir: output_dir.to_string(),
            preset: preset.to_string(),
            priority,
        }
    }

    /// Number of input files in this job.
    pub fn input_count(&self) -> usize {
        self.input_paths.len()
    }
}

// ─────────────────────────────────────────────────────────────
//  PyBatchResult
// ─────────────────────────────────────────────────────────────

/// Result produced when a batch job finishes (either way).
#[derive(Debug, Clone)]
pub struct PyBatchResult {
    /// Identifier of the job this result belongs to.
    pub job_id: u64,
    /// Final status of the job.
    pub status: PyJobStatus,
    /// Wall-clock time the job took, in milliseconds.
    pub elapsed_ms: u64,
}

impl PyBatchResult {
    /// Construct a new `PyBatchResult`.
    pub fn new(job_id: u64, status: PyJobStatus, elapsed_ms: u64) -> Self {
        Self {
            job_id,
            status,
            elapsed_ms,
        }
    }

    /// Returns `true` if the job completed successfully.
    pub fn is_success(&self) -> bool {
        self.status.is_completed()
    }
}

// ─────────────────────────────────────────────────────────────
//  PyBatchQueue
// ─────────────────────────────────────────────────────────────

/// A priority-aware queue of batch jobs.
#[derive(Debug, Clone)]
pub struct PyBatchQueue {
    /// All submitted jobs (pending, running, or finished).
    pub jobs: Vec<PyBatchJob>,
    /// Maximum number of jobs that may run concurrently.
    pub max_concurrent: usize,
    /// Monotonically increasing counter used to assign job IDs.
    next_id: u64,
}

impl PyBatchQueue {
    /// Create a new empty queue.
    ///
    /// # Arguments
    ///
    /// * `max_concurrent` - Upper bound on parallel job execution (minimum 1).
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            jobs: Vec::new(),
            max_concurrent: max_concurrent.max(1),
            next_id: 1,
        }
    }

    /// Submit a job to the queue.
    ///
    /// The job's `id` field is overwritten with a freshly allocated identifier.
    /// Returns the assigned job ID.
    pub fn submit(&mut self, mut job: PyBatchJob) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        job.id = id;
        self.jobs.push(job);
        id
    }

    /// Cancel a pending job by ID.
    ///
    /// Returns `true` if a job with that ID existed and was removed.
    pub fn cancel(&mut self, id: u64) -> bool {
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        self.jobs.len() < before
    }

    /// Number of pending (not yet started) jobs.
    ///
    /// In this skeleton all jobs in the queue are considered "pending".
    pub fn pending_count(&self) -> usize {
        self.jobs.len()
    }

    /// Total number of jobs ever submitted (including cancelled ones that have
    /// been removed — tracked via `next_id - 1`).
    pub fn total_submitted(&self) -> usize {
        (self.next_id - 1) as usize
    }
}

// ─────────────────────────────────────────────────────────────
//  estimate_completion_ms
// ─────────────────────────────────────────────────────────────

/// Estimate the wall-clock time (in ms) until all pending jobs complete.
///
/// Uses a simple model: `ceil(pending / max_concurrent) * avg_job_ms`.
///
/// # Arguments
///
/// * `queue`       - The batch queue to inspect.
/// * `avg_job_ms`  - Expected duration of a single job in milliseconds.
pub fn estimate_completion_ms(queue: &PyBatchQueue, avg_job_ms: u64) -> u64 {
    let pending = queue.pending_count();
    if pending == 0 {
        return 0;
    }
    let concurrency = queue.max_concurrent.max(1) as u64;
    let batches = (pending as u64 + concurrency - 1) / concurrency; // ceil division
    batches * avg_job_ms
}

// ─────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-py-batch-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn make_job(priority: i32) -> PyBatchJob {
        PyBatchJob::new(
            0, // id will be overwritten by submit()
            vec!["a.mp4".to_string(), "b.mp4".to_string()],
            &tmp_str("out"),
            "medium",
            priority,
        )
    }

    // ── PyJobStatus ──────────────────────────────────────────

    #[test]
    fn test_job_status_pending() {
        let s = PyJobStatus::Pending;
        assert!(s.is_pending());
        assert!(!s.is_running());
        assert!(!s.is_completed());
        assert!(!s.is_failed());
    }

    #[test]
    fn test_job_status_running() {
        let s = PyJobStatus::Running { progress: 0.5 };
        assert!(!s.is_pending());
        assert!(s.is_running());
    }

    #[test]
    fn test_job_status_completed() {
        let s = PyJobStatus::Completed {
            output_path: tmp_str("out-result.mkv"),
        };
        assert!(s.is_completed());
        assert!(!s.is_failed());
    }

    #[test]
    fn test_job_status_failed() {
        let s = PyJobStatus::Failed {
            error: "codec error".to_string(),
        };
        assert!(s.is_failed());
        assert!(!s.is_completed());
    }

    // ── PyBatchJob ───────────────────────────────────────────

    #[test]
    fn test_batch_job_new() {
        let job = PyBatchJob::new(42, vec!["x.mkv".to_string()], "/out", "slow", 5);
        assert_eq!(job.id, 42);
        assert_eq!(job.input_count(), 1);
        assert_eq!(job.output_dir, "/out");
        assert_eq!(job.preset, "slow");
        assert_eq!(job.priority, 5);
    }

    #[test]
    fn test_batch_job_input_count() {
        let job = PyBatchJob::new(
            1,
            vec![
                "a.mp4".to_string(),
                "b.mp4".to_string(),
                "c.mp4".to_string(),
            ],
            "/out",
            "fast",
            0,
        );
        assert_eq!(job.input_count(), 3);
    }

    // ── PyBatchResult ────────────────────────────────────────

    #[test]
    fn test_batch_result_success() {
        let r = PyBatchResult::new(
            7,
            PyJobStatus::Completed {
                output_path: tmp_str("out.mkv"),
            },
            5000,
        );
        assert_eq!(r.job_id, 7);
        assert!(r.is_success());
        assert_eq!(r.elapsed_ms, 5000);
    }

    #[test]
    fn test_batch_result_failure() {
        let r = PyBatchResult::new(
            8,
            PyJobStatus::Failed {
                error: "io error".to_string(),
            },
            120,
        );
        assert!(!r.is_success());
    }

    // ── PyBatchQueue ─────────────────────────────────────────

    #[test]
    fn test_queue_new() {
        let q = PyBatchQueue::new(4);
        assert_eq!(q.max_concurrent, 4);
        assert_eq!(q.pending_count(), 0);
        assert_eq!(q.total_submitted(), 0);
    }

    #[test]
    fn test_queue_min_concurrent() {
        // Passing 0 should be clamped to 1.
        let q = PyBatchQueue::new(0);
        assert_eq!(q.max_concurrent, 1);
    }

    #[test]
    fn test_queue_submit() {
        let mut q = PyBatchQueue::new(2);
        let id1 = q.submit(make_job(0));
        let id2 = q.submit(make_job(1));
        assert_ne!(id1, id2);
        assert_eq!(q.pending_count(), 2);
        assert_eq!(q.total_submitted(), 2);
    }

    #[test]
    fn test_queue_cancel() {
        let mut q = PyBatchQueue::new(2);
        let id = q.submit(make_job(0));
        assert!(q.cancel(id));
        assert_eq!(q.pending_count(), 0);
        // total_submitted still reflects the original count.
        assert_eq!(q.total_submitted(), 1);
    }

    #[test]
    fn test_queue_cancel_nonexistent() {
        let mut q = PyBatchQueue::new(2);
        q.submit(make_job(0));
        assert!(!q.cancel(9999));
        assert_eq!(q.pending_count(), 1);
    }

    // ── estimate_completion_ms ───────────────────────────────

    #[test]
    fn test_estimate_completion_empty_queue() {
        let q = PyBatchQueue::new(4);
        assert_eq!(estimate_completion_ms(&q, 1000), 0);
    }

    #[test]
    fn test_estimate_completion_single_batch() {
        let mut q = PyBatchQueue::new(4);
        q.submit(make_job(0));
        q.submit(make_job(0));
        // 2 jobs, concurrency 4 → 1 batch of 1000 ms = 1000
        assert_eq!(estimate_completion_ms(&q, 1000), 1000);
    }

    #[test]
    fn test_estimate_completion_multiple_batches() {
        let mut q = PyBatchQueue::new(2);
        for _ in 0..5 {
            q.submit(make_job(0));
        }
        // 5 jobs, concurrency 2 → ceil(5/2) = 3 batches × 500 ms = 1500
        assert_eq!(estimate_completion_ms(&q, 500), 1500);
    }
}
