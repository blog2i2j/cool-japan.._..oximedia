//! Generic `BatchJob<T>`, `BatchQueue`, `JobExecutor`, `TranscodeJobSpec`,
//! `BatchProcessor`, and `JobStats`.

#![allow(dead_code)]

use async_trait::async_trait;
use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, Semaphore};

// ─────────────────────────────────────────────────────────────────────────────
// Job status
// ─────────────────────────────────────────────────────────────────────────────

/// Status of a generic `BatchJob<T>`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Waiting to be picked up
    Pending,
    /// Currently executing
    Running,
    /// Finished successfully
    Completed,
    /// Finished with an error
    Failed(String),
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(e) => write!(f, "Failed({e})"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchJob<T>
// ─────────────────────────────────────────────────────────────────────────────

/// A generic batch job carrying an arbitrary payload `T`.
///
/// Jobs have a priority (higher = more urgent) and a current `JobStatus`.
#[derive(Debug)]
pub struct BatchJobItem<T> {
    /// Unique identifier
    pub id: u64,
    /// Job payload
    pub payload: T,
    /// Scheduling priority (higher = picked first)
    pub priority: u32,
    /// Current status
    pub status: JobStatus,
}

impl<T> BatchJobItem<T> {
    /// Create a new job with the given `id`, `payload`, and `priority`
    #[must_use]
    pub fn new(id: u64, payload: T, priority: u32) -> Self {
        Self {
            id,
            payload,
            priority,
            status: JobStatus::Pending,
        }
    }

    /// Returns `true` if the job has completed (successfully or not)
    #[must_use]
    pub fn is_done(&self) -> bool {
        matches!(self.status, JobStatus::Completed | JobStatus::Failed(_))
    }
}

// For `BinaryHeap` ordering by priority (higher priority first)
impl<T: std::fmt::Debug> Eq for BatchJobItem<T> {}
impl<T: std::fmt::Debug> PartialEq for BatchJobItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.id == other.id
    }
}
impl<T: std::fmt::Debug> PartialOrd for BatchJobItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: std::fmt::Debug> Ord for BatchJobItem<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then(self.id.cmp(&other.id))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchQueue
// ─────────────────────────────────────────────────────────────────────────────

/// Priority queue for `BatchJobItem<T>` with an optional concurrency limit.
pub struct BatchQueue<T: std::fmt::Debug + Send + 'static> {
    inner: Mutex<BinaryHeap<BatchJobItem<T>>>,
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
}

impl<T: std::fmt::Debug + Send + 'static> BatchQueue<T> {
    /// Create a new queue with a maximum concurrent execution limit
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        let cap = if max_concurrent == 0 {
            1
        } else {
            max_concurrent
        };
        Self {
            inner: Mutex::new(BinaryHeap::new()),
            max_concurrent: cap,
            semaphore: Arc::new(Semaphore::new(cap)),
        }
    }

    /// Push a job onto the queue
    pub async fn push(&self, job: BatchJobItem<T>) {
        let mut heap = self.inner.lock().await;
        heap.push(job);
    }

    /// Pop the highest-priority pending job, or `None` if the queue is empty
    pub async fn pop(&self) -> Option<BatchJobItem<T>> {
        let mut heap = self.inner.lock().await;
        heap.pop()
    }

    /// Current number of jobs waiting in the queue
    pub async fn len(&self) -> usize {
        self.inner.lock().await.len()
    }

    /// Returns `true` if the queue is empty
    pub async fn is_empty(&self) -> bool {
        self.inner.lock().await.is_empty()
    }

    /// Maximum concurrent jobs
    #[must_use]
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Acquire a concurrency slot (blocks until one is available).
    ///
    /// The internal semaphore is never explicitly closed, so this can only fail
    /// in the event of programmer error (e.g. closing the semaphore manually).
    pub async fn acquire_slot(&self) -> tokio::sync::OwnedSemaphorePermit {
        Arc::clone(&self.semaphore)
            .acquire_owned()
            .await
            .unwrap_or_else(|_| unreachable!("BatchQueue semaphore is never closed"))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JobExecutor trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait that processes a single job payload
#[async_trait]
pub trait JobExecutor: Send + Sync {
    /// The job payload type this executor handles
    type Job: Send;

    /// Execute a job
    ///
    /// # Errors
    ///
    /// Returns an error string if execution fails.
    async fn execute(&mut self, job: Self::Job) -> Result<(), String>;
}

// ─────────────────────────────────────────────────────────────────────────────
// TranscodeJobSpec
// ─────────────────────────────────────────────────────────────────────────────

/// Target resolution for a transcode job
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolution {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Resolution {
    /// Create a new resolution
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// 1920x1080 Full HD
    #[must_use]
    pub fn fhd() -> Self {
        Self::new(1920, 1080)
    }

    /// 1280x720 HD
    #[must_use]
    pub fn hd() -> Self {
        Self::new(1280, 720)
    }

    /// 3840x2160 4K UHD
    #[must_use]
    pub fn uhd_4k() -> Self {
        Self::new(3840, 2160)
    }
}

impl std::fmt::Display for Resolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// Specification for a single transcode operation
#[derive(Debug, Clone)]
pub struct TranscodeJobSpec {
    /// Input file path
    pub input_path: PathBuf,
    /// Output file path
    pub output_path: PathBuf,
    /// Codec preset name (e.g. "h264-fast", "av1-quality")
    pub codec_preset: String,
    /// Target resolution (None = keep source)
    pub target_resolution: Option<Resolution>,
}

impl TranscodeJobSpec {
    /// Create a new transcode job specification
    #[must_use]
    pub fn new(
        input_path: impl Into<PathBuf>,
        output_path: impl Into<PathBuf>,
        codec_preset: impl Into<String>,
    ) -> Self {
        Self {
            input_path: input_path.into(),
            output_path: output_path.into(),
            codec_preset: codec_preset.into(),
            target_resolution: None,
        }
    }

    /// Set the target resolution
    #[must_use]
    pub fn with_resolution(mut self, res: Resolution) -> Self {
        self.target_resolution = Some(res);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JobStats
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate statistics for a `BatchProcessor` run
#[derive(Debug, Clone, Default)]
pub struct JobStats {
    /// Total jobs submitted
    pub total: u64,
    /// Jobs completed successfully
    pub succeeded: u64,
    /// Jobs that failed
    pub failed: u64,
    /// Elapsed wall-clock time in seconds
    pub elapsed_secs: f64,
}

impl JobStats {
    /// Jobs per second throughput (returns 0 if elapsed is 0)
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let total = self.total as f64;
            total / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Estimated time remaining given `pending` jobs (in seconds)
    #[must_use]
    pub fn eta(&self, pending: u64) -> Option<f64> {
        let done = self.succeeded + self.failed;
        if done == 0 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let rate = done as f64 / self.elapsed_secs;
        if rate > 0.0 {
            #[allow(clippy::cast_precision_loss)]
            let pending_f = pending as f64;
            Some(pending_f / rate)
        } else {
            None
        }
    }

    /// Success rate in [0.0, 1.0] (returns 0 if nothing finished)
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let done = self.succeeded + self.failed;
        if done == 0 {
            0.0
        } else {
            #[allow(clippy::cast_precision_loss)]
            let result = self.succeeded as f64 / done as f64;
            result
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchProcessor
// ─────────────────────────────────────────────────────────────────────────────

/// Progress callback type: receives (completed, total)
pub type ProgressCallback = Arc<dyn Fn(u64, u64) + Send + Sync>;

/// Processes multiple `TranscodeJobSpec`s concurrently
pub struct BatchProcessor {
    max_concurrent: usize,
    progress_callback: Option<ProgressCallback>,
}

impl BatchProcessor {
    /// Create a new processor with a concurrency limit
    #[must_use]
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent: max_concurrent.max(1),
            progress_callback: None,
        }
    }

    /// Attach a progress callback
    #[must_use]
    pub fn with_progress(mut self, cb: ProgressCallback) -> Self {
        self.progress_callback = Some(cb);
        self
    }

    /// Process all specs concurrently, calling the optional progress callback
    /// after each job finishes.  Returns `JobStats`.
    ///
    /// This implementation simulates transcoding via a no-op executor; in
    /// production it would delegate to a real codec pipeline.
    ///
    /// # Panics
    ///
    /// Panics if the internal semaphore has been closed.
    pub async fn process(&self, specs: Vec<TranscodeJobSpec>) -> JobStats {
        let total = specs.len() as u64;
        let completed = Arc::new(AtomicU64::new(0));
        let succeeded = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));

        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        let start = Instant::now();
        let mut handles = Vec::new();

        for spec in specs {
            let permit = Arc::clone(&semaphore)
                .acquire_owned()
                .await
                .unwrap_or_else(|_| unreachable!("local semaphore is never closed"));
            let completed = Arc::clone(&completed);
            let succeeded = Arc::clone(&succeeded);
            let failed = Arc::clone(&failed);
            let cb = self.progress_callback.clone();

            let handle = tokio::spawn(async move {
                let _permit = permit;

                // Simulate the transcode (real impl would call ffmpeg / codec)
                let ok = simulate_transcode(&spec);

                if ok {
                    succeeded.fetch_add(1, Ordering::Relaxed);
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                }
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;

                if let Some(ref callback) = cb {
                    callback(done, total);
                }
            });

            handles.push(handle);
        }

        for h in handles {
            let _ = h.await;
        }

        JobStats {
            total,
            succeeded: succeeded.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            elapsed_secs: start.elapsed().as_secs_f64(),
        }
    }
}

/// No-op transcode simulation used in tests / when no real codec is available
fn simulate_transcode(spec: &TranscodeJobSpec) -> bool {
    // Validate: preset must be non-empty and paths non-empty
    !spec.codec_preset.is_empty()
        && spec.input_path != PathBuf::new()
        && spec.output_path != PathBuf::new()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    // --- BatchJobItem ---

    #[test]
    fn test_batch_job_item_new() {
        let job: BatchJobItem<i32> = BatchJobItem::new(1, 42, 10);
        assert_eq!(job.id, 1);
        assert_eq!(job.payload, 42);
        assert_eq!(job.priority, 10);
        assert_eq!(job.status, JobStatus::Pending);
    }

    #[test]
    fn test_batch_job_item_is_done() {
        let mut job: BatchJobItem<i32> = BatchJobItem::new(1, 0, 0);
        assert!(!job.is_done());

        job.status = JobStatus::Completed;
        assert!(job.is_done());

        job.status = JobStatus::Failed("oops".into());
        assert!(job.is_done());
    }

    #[test]
    fn test_job_status_display() {
        assert_eq!(JobStatus::Pending.to_string(), "Pending");
        assert_eq!(JobStatus::Running.to_string(), "Running");
        assert_eq!(JobStatus::Completed.to_string(), "Completed");
        assert_eq!(JobStatus::Failed("err".into()).to_string(), "Failed(err)");
    }

    // --- BatchQueue ---

    #[tokio::test]
    async fn test_batch_queue_priority_order() {
        let queue: BatchQueue<&str> = BatchQueue::new(4);
        queue.push(BatchJobItem::new(1, "low", 1)).await;
        queue.push(BatchJobItem::new(2, "high", 100)).await;
        queue.push(BatchJobItem::new(3, "medium", 50)).await;

        let first = queue.pop().await.expect("failed to pop");
        assert_eq!(first.priority, 100);
        let second = queue.pop().await.expect("failed to pop");
        assert_eq!(second.priority, 50);
        let third = queue.pop().await.expect("failed to pop");
        assert_eq!(third.priority, 1);
    }

    #[tokio::test]
    async fn test_batch_queue_len_and_empty() {
        let queue: BatchQueue<u32> = BatchQueue::new(2);
        assert!(queue.is_empty().await);
        queue.push(BatchJobItem::new(0, 99, 1)).await;
        assert_eq!(queue.len().await, 1);
        assert!(!queue.is_empty().await);
        queue.pop().await;
        assert!(queue.is_empty().await);
    }

    #[tokio::test]
    async fn test_batch_queue_max_concurrent() {
        let queue: BatchQueue<i32> = BatchQueue::new(3);
        assert_eq!(queue.max_concurrent(), 3);
    }

    #[tokio::test]
    async fn test_batch_queue_pop_empty() {
        let queue: BatchQueue<String> = BatchQueue::new(1);
        assert!(queue.pop().await.is_none());
    }

    // --- Resolution ---

    #[test]
    fn test_resolution_new() {
        let r = Resolution::new(640, 480);
        assert_eq!(r.width, 640);
        assert_eq!(r.height, 480);
    }

    #[test]
    fn test_resolution_presets() {
        let fhd = Resolution::fhd();
        assert_eq!(fhd.width, 1920);
        assert_eq!(fhd.height, 1080);

        let hd = Resolution::hd();
        assert_eq!(hd.width, 1280);
        assert_eq!(hd.height, 720);

        let k4 = Resolution::uhd_4k();
        assert_eq!(k4.width, 3840);
        assert_eq!(k4.height, 2160);
    }

    #[test]
    fn test_resolution_display() {
        assert_eq!(Resolution::fhd().to_string(), "1920x1080");
    }

    // --- TranscodeJobSpec ---

    #[test]
    fn test_transcode_job_spec_new() {
        let spec = TranscodeJobSpec::new("/in/a.mp4", "/out/a.mp4", "h264-fast");
        assert_eq!(spec.codec_preset, "h264-fast");
        assert!(spec.target_resolution.is_none());
    }

    #[test]
    fn test_transcode_job_spec_with_resolution() {
        let spec = TranscodeJobSpec::new("/in/a.mp4", "/out/a.mp4", "av1")
            .with_resolution(Resolution::hd());
        assert_eq!(spec.target_resolution, Some(Resolution::hd()));
    }

    // --- JobStats ---

    #[test]
    fn test_job_stats_throughput() {
        let stats = JobStats {
            total: 10,
            succeeded: 8,
            failed: 2,
            elapsed_secs: 5.0,
        };
        assert!((stats.throughput() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_stats_success_rate() {
        let stats = JobStats {
            total: 10,
            succeeded: 9,
            failed: 1,
            elapsed_secs: 1.0,
        };
        assert!((stats.success_rate() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_stats_success_rate_all_failed() {
        let stats = JobStats {
            total: 5,
            succeeded: 0,
            failed: 5,
            elapsed_secs: 1.0,
        };
        assert!((stats.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_job_stats_eta() {
        let stats = JobStats {
            total: 10,
            succeeded: 5,
            failed: 0,
            elapsed_secs: 10.0,
        };
        let eta = stats.eta(5).expect("eta should succeed");
        assert!((eta - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_job_stats_eta_none_when_no_done() {
        let stats = JobStats {
            total: 10,
            succeeded: 0,
            failed: 0,
            elapsed_secs: 0.0,
        };
        assert!(stats.eta(10).is_none());
    }

    #[test]
    fn test_job_stats_throughput_zero_elapsed() {
        let stats = JobStats {
            total: 5,
            succeeded: 5,
            failed: 0,
            elapsed_secs: 0.0,
        };
        assert_eq!(stats.throughput(), 0.0);
    }

    // --- BatchProcessor ---

    #[tokio::test]
    async fn test_batch_processor_all_succeed() {
        let processor = BatchProcessor::new(2);
        let specs = vec![
            TranscodeJobSpec::new("/in/a.mp4", "/out/a.mp4", "h264"),
            TranscodeJobSpec::new("/in/b.mp4", "/out/b.mp4", "av1"),
            TranscodeJobSpec::new("/in/c.mp4", "/out/c.mp4", "hevc"),
        ];
        let stats = processor.process(specs).await;
        assert_eq!(stats.total, 3);
        assert_eq!(stats.succeeded, 3);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn test_batch_processor_empty_input() {
        let processor = BatchProcessor::new(4);
        let stats = processor.process(vec![]).await;
        assert_eq!(stats.total, 0);
    }

    #[tokio::test]
    async fn test_batch_processor_progress_callback() {
        let counter = Arc::new(StdMutex::new(0u64));
        let counter_clone = Arc::clone(&counter);
        let cb: ProgressCallback = Arc::new(move |done, _total| {
            *counter_clone.lock().expect("lock poisoned") = done;
        });

        let processor = BatchProcessor::new(2).with_progress(cb);
        let specs = vec![
            TranscodeJobSpec::new("/a", "/b", "h264"),
            TranscodeJobSpec::new("/c", "/d", "h264"),
        ];
        let stats = processor.process(specs).await;
        assert_eq!(stats.total, 2);

        let final_count = *counter.lock().expect("lock poisoned");
        assert_eq!(final_count, 2);
    }

    #[tokio::test]
    async fn test_batch_processor_concurrency() {
        // Run 8 jobs with max_concurrent=3 – should complete without deadlock
        let processor = BatchProcessor::new(3);
        let specs: Vec<_> = (0..8)
            .map(|i| TranscodeJobSpec::new(format!("/in/{i}.mp4"), format!("/out/{i}.mp4"), "h264"))
            .collect();
        let stats = processor.process(specs).await;
        assert_eq!(stats.succeeded + stats.failed, 8);
    }

    #[tokio::test]
    async fn test_batch_processor_success_rate_full() {
        let processor = BatchProcessor::new(4);
        let specs: Vec<_> = (0..6)
            .map(|i| {
                TranscodeJobSpec::new(format!("/in/{i}.ts"), format!("/out/{i}.mp4"), "preset")
            })
            .collect();
        let stats = processor.process(specs).await;
        assert!((stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }
}
