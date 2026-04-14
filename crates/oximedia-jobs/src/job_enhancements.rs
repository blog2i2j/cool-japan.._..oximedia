//! Enhanced job management: DAG with conditional edges, cancellation tokens,
//! retry with backoff, progress reporting, worker health monitoring,
//! aggregate job metrics, batch processing, and improved priority queue.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Job DAG with conditional edges
// ---------------------------------------------------------------------------

/// Condition under which a dependency edge is satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyCondition {
    /// Run only if the dependency completed successfully.
    OnSuccess,
    /// Run only if the dependency failed.
    OnFailure,
    /// Run regardless of the dependency's outcome.
    Always,
}

/// Status of a completed job, used to evaluate dependency conditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    /// Job completed without error.
    Success,
    /// Job failed with an error.
    Failed,
}

/// A directed-acyclic job graph with conditional edge semantics.
///
/// An edge `(dep_id -> job_id, condition)` means: *job_id* may only run after
/// *dep_id*, and only when *dep_id*'s result satisfies *condition*.
#[derive(Debug, Default)]
pub struct JobDag {
    /// Registered jobs: job_id -> job name.
    jobs: HashMap<u64, String>,
    /// Edges: job_id -> [(depends_on, condition)].
    edges: HashMap<u64, Vec<(u64, DependencyCondition)>>,
}

impl JobDag {
    /// Create an empty DAG.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a job by ID and name.
    pub fn add_job(&mut self, job_id: u64, name: impl Into<String>) {
        self.jobs.insert(job_id, name.into());
    }

    /// Add a conditional dependency.
    ///
    /// `job_id` will only become ready when `depends_on` has completed and
    /// its outcome satisfies `condition`.
    pub fn add_dependency(
        &mut self,
        job_id: u64,
        depends_on: u64,
        condition: DependencyCondition,
    ) {
        self.edges
            .entry(job_id)
            .or_default()
            .push((depends_on, condition));
    }

    /// Return the IDs of all jobs that are currently ready to run.
    ///
    /// A job is "ready" when every one of its dependency edges is satisfied
    /// by the provided `completed` map. Jobs with no dependencies are always
    /// ready (as long as they haven't been completed themselves).
    ///
    /// # Arguments
    ///
    /// * `completed` – Map of `job_id -> JobStatus` for all finished jobs.
    #[must_use]
    pub fn get_ready_jobs(&self, completed: &HashMap<u64, JobStatus>) -> Vec<u64> {
        self.jobs
            .keys()
            .copied()
            .filter(|job_id| {
                // Already completed → not "ready to run"
                if completed.contains_key(job_id) {
                    return false;
                }

                // Check all dependency edges for this job
                let deps = self.edges.get(job_id).map(|v| v.as_slice()).unwrap_or(&[]);
                deps.iter().all(|(dep_id, condition)| {
                    match completed.get(dep_id) {
                        None => false, // dependency not yet done
                        Some(status) => match condition {
                            DependencyCondition::OnSuccess => *status == JobStatus::Success,
                            DependencyCondition::OnFailure => *status == JobStatus::Failed,
                            DependencyCondition::Always => true,
                        },
                    }
                })
            })
            .collect()
    }

    /// Return the dependency edges for a given job.
    #[must_use]
    pub fn dependencies_of(&self, job_id: u64) -> &[(u64, DependencyCondition)] {
        self.edges.get(&job_id).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// Job cancellation token
// ---------------------------------------------------------------------------

/// A handle that can be used to cancel a job.
#[derive(Debug, Clone)]
pub struct CancellationHandle {
    flag: Arc<AtomicBool>,
}

impl CancellationHandle {
    /// Signal cancellation.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Returns `true` if cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// A token observed by the job to check for cancellation.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    flag: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Returns `true` if the corresponding [`CancellationHandle`] has called
    /// [`cancel`](CancellationHandle::cancel).
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Create a linked `(CancellationToken, CancellationHandle)` pair.
#[must_use]
pub fn new_cancellation_pair() -> (CancellationToken, CancellationHandle) {
    let flag = Arc::new(AtomicBool::new(false));
    (
        CancellationToken { flag: Arc::clone(&flag) },
        CancellationHandle { flag },
    )
}

// ---------------------------------------------------------------------------
// Job retry with exponential backoff
// ---------------------------------------------------------------------------

/// A simple retry policy with exponential backoff.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of attempts (including the first try).
    pub max_attempts: u32,
    /// Base delay in milliseconds.
    pub base_delay_ms: u64,
}

impl RetryPolicy {
    /// Create a new retry policy.
    #[must_use]
    pub fn new(max_attempts: u32, base_delay_ms: u64) -> Self {
        Self {
            max_attempts,
            base_delay_ms,
        }
    }

    /// Compute the delay before the given attempt (0-indexed).
    ///
    /// Formula: `base_delay_ms * 2^attempt`.
    ///
    /// Returns 0 for attempt 0 (no delay before the first retry attempt after
    /// failure). Saturates at `u64::MAX`.
    #[must_use]
    pub fn next_delay(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return self.base_delay_ms;
        }
        // base_delay * 2^attempt, with saturation.
        let shift = attempt.min(63);
        let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
        self.base_delay_ms.saturating_mul(multiplier)
    }

    /// Returns `true` if another attempt is allowed after `attempt` failures.
    ///
    /// `attempt` is 0-indexed (0 = the first failure has just occurred).
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt + 1 < self.max_attempts
    }
}

// ---------------------------------------------------------------------------
// Job progress reporting
// ---------------------------------------------------------------------------

/// A progress update snapshot.
#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    /// Job identifier.
    pub job_id: u64,
    /// Completion percentage [0.0, 100.0].
    pub pct: f32,
    /// Human-readable status message.
    pub message: String,
    /// Monotonic instant of this update.
    pub recorded_at: Instant,
}

/// Per-job progress tracker.
#[derive(Debug)]
pub struct JobProgress {
    /// The job being tracked.
    pub job_id: u64,
    last: Mutex<Option<ProgressUpdate>>,
}

impl JobProgress {
    /// Create a new progress tracker for the given job.
    #[must_use]
    pub fn new(job_id: u64) -> Self {
        Self {
            job_id,
            last: Mutex::new(None),
        }
    }

    /// Record a progress update.
    pub fn update(&self, pct: f32, message: &str) {
        let update = ProgressUpdate {
            job_id: self.job_id,
            pct: pct.clamp(0.0, 100.0),
            message: message.to_string(),
            recorded_at: Instant::now(),
        };
        *self.last.lock().expect("progress lock poisoned") = Some(update);
    }

    /// Return the most recent progress update, if any.
    pub fn last_update(&self) -> Option<ProgressUpdate> {
        self.last.lock().expect("progress lock poisoned").clone()
    }
}

// ---------------------------------------------------------------------------
// Worker health monitoring
// ---------------------------------------------------------------------------

/// Tracks worker heartbeats and detects stale workers.
#[derive(Debug, Default)]
pub struct WorkerHealthMonitor {
    /// Map from worker_id → last heartbeat timestamp (ms since arbitrary epoch).
    last_heartbeat: Mutex<HashMap<u64, u64>>,
}

impl WorkerHealthMonitor {
    /// Create a new monitor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a heartbeat from `worker_id` at `now_ms`.
    pub fn record_heartbeat(&self, worker_id: u64, now_ms: u64) {
        self.last_heartbeat
            .lock()
            .expect("health monitor lock poisoned")
            .insert(worker_id, now_ms);
    }

    /// Return the IDs of workers whose last heartbeat was more than
    /// `stale_ms` milliseconds before `now_ms`.
    #[must_use]
    pub fn get_stale_workers(&self, stale_ms: u64, now_ms: u64) -> Vec<u64> {
        let cutoff = now_ms.saturating_sub(stale_ms);
        self.last_heartbeat
            .lock()
            .expect("health monitor lock poisoned")
            .iter()
            .filter_map(|(&wid, &last)| if last < cutoff { Some(wid) } else { None })
            .collect()
    }

    /// Number of tracked workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.last_heartbeat
            .lock()
            .expect("health monitor lock poisoned")
            .len()
    }
}

// ---------------------------------------------------------------------------
// Aggregate job metrics
// ---------------------------------------------------------------------------

/// Lightweight aggregate metrics collector.
#[derive(Debug, Default)]
pub struct JobMetrics {
    completions: Mutex<Vec<(u64, u64, bool)>>, // (job_id, duration_ms, success)
}

impl JobMetrics {
    /// Create a new metrics store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a job completion.
    pub fn record_completion(&self, job_id: u64, duration_ms: u64, success: bool) {
        self.completions
            .lock()
            .expect("job metrics lock poisoned")
            .push((job_id, duration_ms, success));
    }

    /// Average duration across all recorded completions (ms). Returns 0.0 if
    /// no completions have been recorded.
    #[must_use]
    pub fn avg_duration_ms(&self) -> f64 {
        let guard = self.completions.lock().expect("job metrics lock poisoned");
        if guard.is_empty() {
            return 0.0;
        }
        let sum: u64 = guard.iter().map(|(_, d, _)| *d).sum();
        sum as f64 / guard.len() as f64
    }

    /// Success rate in [0.0, 1.0]. Returns 0.0 if no completions recorded.
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let guard = self.completions.lock().expect("job metrics lock poisoned");
        if guard.is_empty() {
            return 0.0;
        }
        let successes = guard.iter().filter(|(_, _, ok)| *ok).count();
        successes as f32 / guard.len() as f32
    }

    /// Total number of recorded completions.
    #[must_use]
    pub fn total_completions(&self) -> usize {
        self.completions
            .lock()
            .expect("job metrics lock poisoned")
            .len()
    }
}

// ---------------------------------------------------------------------------
// Batch job processing
// ---------------------------------------------------------------------------

/// Identifier for a batch of jobs.
pub type BatchId = u64;

/// A simple job descriptor used in batch processing.
#[derive(Debug, Clone)]
pub struct Job {
    /// Job identifier.
    pub id: u64,
    /// Job name.
    pub name: String,
    /// Priority (higher = more urgent).
    pub priority: i32,
}

impl Job {
    /// Create a new job.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, priority: i32) -> Self {
        Self {
            id,
            name: name.into(),
            priority,
        }
    }
}

/// Status of a submitted batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchStatus {
    /// Batch is queued but not yet started.
    Pending,
    /// Batch is currently processing.
    Running,
    /// All jobs in the batch have completed.
    Completed,
    /// The batch was cancelled before completion.
    Cancelled,
}

/// A record of a submitted batch.
#[derive(Debug)]
struct BatchRecord {
    jobs: Vec<Job>,
    status: BatchStatus,
}

/// A simple batch job processor.
///
/// Batches are submitted synchronously; in a real system they would be
/// dispatched to an async worker pool.
#[derive(Debug, Default)]
pub struct BatchJobProcessor {
    batches: Mutex<HashMap<BatchId, BatchRecord>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl BatchJobProcessor {
    /// Create a new processor.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a batch of jobs and return its [`BatchId`].
    pub fn submit_batch(&self, jobs: Vec<Job>) -> BatchId {
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);
        self.batches
            .lock()
            .expect("batch processor lock poisoned")
            .insert(
                id,
                BatchRecord {
                    jobs,
                    status: BatchStatus::Pending,
                },
            );
        id
    }

    /// Poll the status of a batch.
    ///
    /// Returns `None` if the batch ID is not recognised.
    pub fn poll_batch(&self, batch_id: BatchId) -> Option<BatchStatus> {
        self.batches
            .lock()
            .expect("batch processor lock poisoned")
            .get(&batch_id)
            .map(|r| r.status.clone())
    }

    /// Advance a batch to `Running` state (for testing / simulation).
    pub fn start_batch(&self, batch_id: BatchId) {
        if let Some(record) = self
            .batches
            .lock()
            .expect("batch processor lock poisoned")
            .get_mut(&batch_id)
        {
            if record.status == BatchStatus::Pending {
                record.status = BatchStatus::Running;
            }
        }
    }

    /// Mark a batch as completed (for testing / simulation).
    pub fn complete_batch(&self, batch_id: BatchId) {
        if let Some(record) = self
            .batches
            .lock()
            .expect("batch processor lock poisoned")
            .get_mut(&batch_id)
        {
            if record.status == BatchStatus::Running {
                record.status = BatchStatus::Completed;
            }
        }
    }

    /// Return the number of jobs in a batch.
    pub fn batch_job_count(&self, batch_id: BatchId) -> usize {
        self.batches
            .lock()
            .expect("batch processor lock poisoned")
            .get(&batch_id)
            .map(|r| r.jobs.len())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Enhanced priority queue with drain_priority
// ---------------------------------------------------------------------------

/// Priority enumeration for the enhanced queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Lowest priority.
    Low = 0,
    /// Normal priority.
    Normal = 1,
    /// Highest priority.
    High = 2,
}

/// An item in the enhanced priority queue.
#[derive(Debug, Clone)]
pub struct QueueItem<T> {
    /// The stored value.
    pub value: T,
    /// The priority.
    pub priority: Priority,
    /// Insertion sequence for FIFO tie-breaking within the same priority.
    seq: u64,
}

/// An enhanced priority queue with `peek`, `len`, and `drain_priority`.
///
/// Higher-priority items (larger `Priority` variant) dequeue first. Within the
/// same priority, items are dequeued in FIFO order.
#[derive(Debug)]
pub struct EnhancedPriorityQueue<T> {
    items: Vec<QueueItem<T>>,
    counter: u64,
}

impl<T> Default for EnhancedPriorityQueue<T> {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            counter: 0,
        }
    }
}

impl<T> EnhancedPriorityQueue<T> {
    /// Create a new empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an item with the given priority.
    pub fn push(&mut self, value: T, priority: Priority) {
        let seq = self.counter;
        self.counter += 1;
        self.items.push(QueueItem {
            value,
            priority,
            seq,
        });
        // Keep sorted: highest priority first, then lowest seq first.
        self.items.sort_unstable_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| a.seq.cmp(&b.seq))
        });
    }

    /// Pop the highest-priority item (FIFO within same priority).
    pub fn pop(&mut self) -> Option<T> {
        if self.items.is_empty() {
            None
        } else {
            Some(self.items.remove(0).value)
        }
    }

    /// Peek at the highest-priority item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.items.first().map(|item| &item.value)
    }

    /// Return the number of items in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Remove and return all items with the given priority, preserving FIFO
    /// order among them.
    pub fn drain_priority(&mut self, priority: Priority) -> Vec<T> {
        let (matching, remaining): (Vec<QueueItem<T>>, Vec<QueueItem<T>>) =
            std::mem::take(&mut self.items)
                .into_iter()
                .partition(|item| item.priority == priority);
        self.items = remaining;
        matching.into_iter().map(|item| item.value).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── JobDag ────────────────────────────────────────────────────────────

    #[test]
    fn test_dag_ready_jobs_no_deps() {
        let mut dag = JobDag::new();
        dag.add_job(1, "A");
        dag.add_job(2, "B");
        let completed: HashMap<u64, JobStatus> = HashMap::new();
        let mut ready = dag.get_ready_jobs(&completed);
        ready.sort();
        assert_eq!(ready, vec![1, 2]);
    }

    #[test]
    fn test_dag_ready_jobs_on_success_condition_satisfied() {
        let mut dag = JobDag::new();
        dag.add_job(1, "A");
        dag.add_job(2, "B");
        dag.add_dependency(2, 1, DependencyCondition::OnSuccess);

        let mut completed = HashMap::new();
        completed.insert(1u64, JobStatus::Success);

        let ready = dag.get_ready_jobs(&completed);
        assert_eq!(ready, vec![2]);
    }

    #[test]
    fn test_dag_ready_jobs_on_success_condition_not_satisfied_when_failed() {
        let mut dag = JobDag::new();
        dag.add_job(1, "A");
        dag.add_job(2, "B");
        dag.add_dependency(2, 1, DependencyCondition::OnSuccess);

        let mut completed = HashMap::new();
        completed.insert(1u64, JobStatus::Failed);

        let ready = dag.get_ready_jobs(&completed);
        assert!(ready.is_empty(), "OnSuccess should not fire when dep failed");
    }

    #[test]
    fn test_dag_ready_jobs_on_failure_condition() {
        let mut dag = JobDag::new();
        dag.add_job(1, "cleanup");
        dag.add_job(2, "processor");
        dag.add_dependency(1, 2, DependencyCondition::OnFailure);

        let mut completed = HashMap::new();
        completed.insert(2u64, JobStatus::Failed);

        let ready = dag.get_ready_jobs(&completed);
        assert_eq!(ready, vec![1]);
    }

    #[test]
    fn test_dag_ready_jobs_always_condition() {
        let mut dag = JobDag::new();
        dag.add_job(1, "A");
        dag.add_job(2, "B");
        dag.add_dependency(2, 1, DependencyCondition::Always);

        let mut completed = HashMap::new();
        completed.insert(1u64, JobStatus::Failed); // Always ignores the outcome

        let ready = dag.get_ready_jobs(&completed);
        assert_eq!(ready, vec![2]);
    }

    #[test]
    fn test_dag_completed_jobs_not_in_ready() {
        let mut dag = JobDag::new();
        dag.add_job(1, "done");

        let mut completed = HashMap::new();
        completed.insert(1u64, JobStatus::Success);

        let ready = dag.get_ready_jobs(&completed);
        assert!(ready.is_empty());
    }

    // ── Cancellation ─────────────────────────────────────────────────────

    #[test]
    fn test_cancellation_not_cancelled_initially() {
        let (token, _handle) = new_cancellation_pair();
        assert!(!token.is_cancelled());
    }

    #[test]
    fn test_cancellation_handle_cancels_token() {
        let (token, handle) = new_cancellation_pair();
        handle.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_handle_also_reflects_state() {
        let (_, handle) = new_cancellation_pair();
        assert!(!handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());
    }

    #[test]
    fn test_cancellation_clone_shares_state() {
        let (token, handle) = new_cancellation_pair();
        let token2 = token.clone();
        handle.cancel();
        assert!(token2.is_cancelled());
    }

    // ── RetryPolicy ───────────────────────────────────────────────────────

    #[test]
    fn test_retry_policy_exponential_backoff() {
        let policy = RetryPolicy::new(5, 100);
        // attempt 0: 100ms
        assert_eq!(policy.next_delay(0), 100);
        // attempt 1: 100 * 2^1 = 200ms
        assert_eq!(policy.next_delay(1), 200);
        // attempt 2: 100 * 2^2 = 400ms
        assert_eq!(policy.next_delay(2), 400);
        // attempt 3: 100 * 2^3 = 800ms
        assert_eq!(policy.next_delay(3), 800);
    }

    #[test]
    fn test_retry_policy_should_retry() {
        let policy = RetryPolicy::new(3, 50); // max 3 attempts: 0, 1, 2
        assert!(policy.should_retry(0));  // after 1st failure, can retry
        assert!(policy.should_retry(1));  // after 2nd failure, can retry
        assert!(!policy.should_retry(2)); // after 3rd failure, max reached
    }

    #[test]
    fn test_retry_policy_backoff_grows_exponentially() {
        let policy = RetryPolicy::new(10, 10);
        let d0 = policy.next_delay(0);
        let d1 = policy.next_delay(1);
        let d2 = policy.next_delay(2);
        // Each step doubles
        assert_eq!(d1, d0 * 2);
        assert_eq!(d2, d0 * 4);
    }

    // ── JobProgress ───────────────────────────────────────────────────────

    #[test]
    fn test_progress_no_update() {
        let prog = JobProgress::new(1);
        assert!(prog.last_update().is_none());
    }

    #[test]
    fn test_progress_update_and_read() {
        let prog = JobProgress::new(42);
        prog.update(75.0, "three-quarters done");
        let u = prog.last_update().expect("should have update");
        assert_eq!(u.job_id, 42);
        assert!((u.pct - 75.0).abs() < 1e-6);
        assert_eq!(u.message, "three-quarters done");
    }

    #[test]
    fn test_progress_clamps_to_100() {
        let prog = JobProgress::new(1);
        prog.update(150.0, "over 100");
        let u = prog.last_update().expect("should have update");
        assert!((u.pct - 100.0).abs() < 1e-6);
    }

    // ── WorkerHealthMonitor ───────────────────────────────────────────────

    #[test]
    fn test_health_monitor_no_stale_workers() {
        let monitor = WorkerHealthMonitor::new();
        monitor.record_heartbeat(1, 1000);
        monitor.record_heartbeat(2, 999);
        // No workers are stale at now=1000 with stale_ms=500 (cutoff=500)
        let stale = monitor.get_stale_workers(500, 1000);
        assert!(stale.is_empty());
    }

    #[test]
    fn test_health_monitor_detects_stale_worker() {
        let monitor = WorkerHealthMonitor::new();
        monitor.record_heartbeat(1, 100);   // last heartbeat at 100ms
        monitor.record_heartbeat(2, 5000);  // last heartbeat at 5000ms
        // now=6000, stale_ms=2000 → cutoff=4000 → worker 1 (100 < 4000) is stale
        let mut stale = monitor.get_stale_workers(2000, 6000);
        stale.sort();
        assert_eq!(stale, vec![1]);
    }

    #[test]
    fn test_health_monitor_all_stale() {
        let monitor = WorkerHealthMonitor::new();
        monitor.record_heartbeat(1, 0);
        monitor.record_heartbeat(2, 0);
        let mut stale = monitor.get_stale_workers(1, 1000);
        stale.sort();
        assert_eq!(stale, vec![1, 2]);
    }

    // ── JobMetrics ────────────────────────────────────────────────────────

    #[test]
    fn test_job_metrics_avg_duration() {
        let m = JobMetrics::new();
        m.record_completion(1, 100, true);
        m.record_completion(2, 300, true);
        let avg = m.avg_duration_ms();
        assert!((avg - 200.0).abs() < 1e-9, "avg should be 200ms, got {avg}");
    }

    #[test]
    fn test_job_metrics_success_rate() {
        let m = JobMetrics::new();
        m.record_completion(1, 100, true);
        m.record_completion(2, 200, false);
        m.record_completion(3, 150, true);
        let rate = m.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-6, "rate={rate}");
    }

    #[test]
    fn test_job_metrics_empty() {
        let m = JobMetrics::new();
        assert_eq!(m.avg_duration_ms(), 0.0);
        assert_eq!(m.success_rate(), 0.0);
    }

    // ── BatchJobProcessor ─────────────────────────────────────────────────

    #[test]
    fn test_batch_submit_and_poll() {
        let proc = BatchJobProcessor::new();
        let jobs = vec![
            Job::new(1, "encode", 1),
            Job::new(2, "thumbnail", 0),
        ];
        let bid = proc.submit_batch(jobs);
        assert_eq!(proc.poll_batch(bid), Some(BatchStatus::Pending));
    }

    #[test]
    fn test_batch_job_count() {
        let proc = BatchJobProcessor::new();
        let jobs = vec![Job::new(1, "a", 0), Job::new(2, "b", 0), Job::new(3, "c", 0)];
        let bid = proc.submit_batch(jobs);
        assert_eq!(proc.batch_job_count(bid), 3);
    }

    #[test]
    fn test_batch_lifecycle() {
        let proc = BatchJobProcessor::new();
        let bid = proc.submit_batch(vec![Job::new(1, "j", 0)]);
        proc.start_batch(bid);
        assert_eq!(proc.poll_batch(bid), Some(BatchStatus::Running));
        proc.complete_batch(bid);
        assert_eq!(proc.poll_batch(bid), Some(BatchStatus::Completed));
    }

    #[test]
    fn test_batch_missing_returns_none() {
        let proc = BatchJobProcessor::new();
        assert!(proc.poll_batch(9999).is_none());
    }

    // ── EnhancedPriorityQueue ─────────────────────────────────────────────

    #[test]
    fn test_epq_priority_ordering() {
        let mut pq: EnhancedPriorityQueue<&str> = EnhancedPriorityQueue::new();
        pq.push("low", Priority::Low);
        pq.push("high", Priority::High);
        pq.push("normal", Priority::Normal);

        assert_eq!(pq.pop(), Some("high"));
        assert_eq!(pq.pop(), Some("normal"));
        assert_eq!(pq.pop(), Some("low"));
    }

    #[test]
    fn test_epq_peek_does_not_remove() {
        let mut pq: EnhancedPriorityQueue<u32> = EnhancedPriorityQueue::new();
        pq.push(42, Priority::High);
        assert_eq!(pq.peek(), Some(&42));
        assert_eq!(pq.len(), 1);
    }

    #[test]
    fn test_epq_drain_priority() {
        let mut pq: EnhancedPriorityQueue<u32> = EnhancedPriorityQueue::new();
        pq.push(1, Priority::High);
        pq.push(2, Priority::Normal);
        pq.push(3, Priority::High);
        pq.push(4, Priority::Low);

        let high = pq.drain_priority(Priority::High);
        assert_eq!(high.len(), 2);
        assert!(high.contains(&1));
        assert!(high.contains(&3));
        assert_eq!(pq.len(), 2); // Normal and Low remain
    }

    #[test]
    fn test_epq_len_and_empty() {
        let mut pq: EnhancedPriorityQueue<i32> = EnhancedPriorityQueue::new();
        assert!(pq.is_empty());
        assert_eq!(pq.len(), 0);
        pq.push(99, Priority::Normal);
        assert_eq!(pq.len(), 1);
        pq.pop();
        assert!(pq.is_empty());
    }

    #[test]
    fn test_epq_fifo_within_same_priority() {
        let mut pq: EnhancedPriorityQueue<u32> = EnhancedPriorityQueue::new();
        pq.push(10, Priority::Normal);
        pq.push(20, Priority::Normal);
        pq.push(30, Priority::Normal);
        // FIFO within same priority
        assert_eq!(pq.pop(), Some(10));
        assert_eq!(pq.pop(), Some(20));
        assert_eq!(pq.pop(), Some(30));
    }
}
