#![allow(dead_code)]
//! Per-worker performance metrics for the encoding farm.
//!
//! Tracks job completion rate, average processing time, error rate, and
//! resource utilisation statistics for every registered worker.  All data is
//! collected in-process — no Prometheus or external dependency is required.
//!
//! ## Design
//!
//! Each worker is represented by a [`WorkerMetricsRecord`].  The
//! [`WorkerMetricsRegistry`] aggregates records for all workers and exposes
//! methods to update counters and query derived statistics such as completion
//! rate, error rate, and throughput.
//!
//! Resource snapshots (CPU/memory/GPU) are stored as a fixed-size ring buffer
//! so that callers can compute windowed averages without unbounded growth.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{FarmError, WorkerId};

// ---------------------------------------------------------------------------
// Ring buffer for resource snapshots
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer that keeps the last `N` samples.
#[derive(Debug, Clone)]
pub struct RingBuffer<T: Clone> {
    buf: Vec<T>,
    head: usize,
    len: usize,
    cap: usize,
}

impl<T: Clone> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] when `capacity` is zero.
    pub fn new(capacity: usize) -> crate::Result<Self> {
        if capacity == 0 {
            return Err(FarmError::InvalidConfig(
                "RingBuffer capacity must be > 0".into(),
            ));
        }
        Ok(Self {
            buf: Vec::with_capacity(capacity),
            head: 0,
            len: 0,
            cap: capacity,
        })
    }

    /// Push a new item, evicting the oldest if the buffer is full.
    pub fn push(&mut self, item: T) {
        if self.len < self.cap {
            self.buf.push(item);
            self.len += 1;
        } else {
            self.buf[self.head] = item;
            self.head = (self.head + 1) % self.cap;
        }
    }

    /// Iterate over stored items in insertion order (oldest → newest).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // When the buffer is not yet full, items are packed at [0..len] in
        // insertion order — no wrap-around.
        //
        // When full, `head` is the index of the *oldest* item (the slot that
        // will be overwritten on the *next* push).  We yield
        //   [head..cap]  (oldest … end of backing array)
        //   [0..head]    (beginning of backing array … newest)
        if self.len < self.cap {
            self.buf[..self.len].iter().chain(self.buf[0..0].iter())
        } else {
            let (lo, hi) = self.buf.split_at(self.head);
            // hi = [head..cap]  ← oldest items
            // lo = [0..head]    ← newest items
            hi.iter().chain(lo.iter())
        }
    }

    /// Number of items currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` when no items are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// ---------------------------------------------------------------------------
// Resource snapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of a worker's resource utilisation.
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// CPU utilisation in the range `[0.0, 1.0]`.
    pub cpu_utilisation: f64,
    /// Memory utilisation in the range `[0.0, 1.0]`.
    pub memory_utilisation: f64,
    /// GPU utilisation in the range `[0.0, 1.0]` (0.0 if no GPU).
    pub gpu_utilisation: f64,
    /// Network bandwidth utilisation in bytes per second.
    pub network_bytes_per_sec: u64,
    /// When the snapshot was taken.
    pub captured_at: Instant,
}

impl ResourceSnapshot {
    /// Create a new resource snapshot with the given values.
    #[must_use]
    pub fn new(
        cpu_utilisation: f64,
        memory_utilisation: f64,
        gpu_utilisation: f64,
        network_bytes_per_sec: u64,
    ) -> Self {
        Self {
            cpu_utilisation: cpu_utilisation.clamp(0.0, 1.0),
            memory_utilisation: memory_utilisation.clamp(0.0, 1.0),
            gpu_utilisation: gpu_utilisation.clamp(0.0, 1.0),
            network_bytes_per_sec,
            captured_at: Instant::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Job duration record
// ---------------------------------------------------------------------------

/// A completed-job record used to derive average processing time.
#[derive(Debug, Clone)]
pub struct CompletedJobRecord {
    /// Wall-clock duration from dispatch to completion.
    pub wall_time: Duration,
    /// Whether the job succeeded.
    pub succeeded: bool,
    /// Number of frames processed (may be 0 for non-video jobs).
    pub frames_processed: u64,
}

// ---------------------------------------------------------------------------
// Per-worker record
// ---------------------------------------------------------------------------

/// All metrics accumulated for a single worker.
#[derive(Debug)]
pub struct WorkerMetricsRecord {
    /// Stable worker identifier.
    pub worker_id: WorkerId,

    /// Total jobs assigned to this worker.
    pub jobs_assigned: u64,
    /// Jobs that completed successfully.
    pub jobs_succeeded: u64,
    /// Jobs that finished with an error.
    pub jobs_failed: u64,
    /// Jobs that were cancelled externally.
    pub jobs_cancelled: u64,
    /// Jobs that exceeded the timeout.
    pub jobs_timed_out: u64,

    /// Sum of wall-clock durations for all completed (succeeded or failed) jobs.
    /// Used to derive `mean_processing_time`.
    pub total_processing_time: Duration,

    /// Total frames processed across all successful video jobs.
    pub total_frames_processed: u64,

    /// Ring buffer of recent resource snapshots.
    resource_history: RingBuffer<ResourceSnapshot>,

    /// Ring buffer of recent job durations (kept for latency percentile use).
    duration_history: RingBuffer<CompletedJobRecord>,

    /// When this record was first created (worker registration time).
    pub registered_at: Instant,

    /// When the last job completion was recorded.
    pub last_activity_at: Option<Instant>,
}

impl WorkerMetricsRecord {
    /// Create a new metrics record for the given worker.
    ///
    /// * `resource_window` — how many resource snapshots to retain.
    /// * `duration_window` — how many job-duration records to retain.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if either window is zero.
    pub fn new(
        worker_id: WorkerId,
        resource_window: usize,
        duration_window: usize,
    ) -> crate::Result<Self> {
        Ok(Self {
            worker_id,
            jobs_assigned: 0,
            jobs_succeeded: 0,
            jobs_failed: 0,
            jobs_cancelled: 0,
            jobs_timed_out: 0,
            total_processing_time: Duration::ZERO,
            total_frames_processed: 0,
            resource_history: RingBuffer::new(resource_window)?,
            duration_history: RingBuffer::new(duration_window)?,
            registered_at: Instant::now(),
            last_activity_at: None,
        })
    }

    // --- Counters -----------------------------------------------------------

    /// Record that a job was dispatched to this worker.
    pub fn record_job_assigned(&mut self) {
        self.jobs_assigned += 1;
    }

    /// Record a job completion outcome.
    pub fn record_job_completed(&mut self, record: CompletedJobRecord) {
        if record.succeeded {
            self.jobs_succeeded += 1;
        } else {
            self.jobs_failed += 1;
        }
        self.total_processing_time += record.wall_time;
        self.total_frames_processed += record.frames_processed;
        self.last_activity_at = Some(Instant::now());
        self.duration_history.push(record);
    }

    /// Record a job cancellation.
    pub fn record_job_cancelled(&mut self) {
        self.jobs_cancelled += 1;
        self.last_activity_at = Some(Instant::now());
    }

    /// Record a job timeout.
    pub fn record_job_timed_out(&mut self) {
        self.jobs_timed_out += 1;
        self.last_activity_at = Some(Instant::now());
    }

    /// Push a resource utilisation snapshot.
    pub fn record_resource_snapshot(&mut self, snapshot: ResourceSnapshot) {
        self.resource_history.push(snapshot);
    }

    // --- Derived statistics -------------------------------------------------

    /// Job completion rate = succeeded / assigned.
    ///
    /// Returns `None` when no jobs have been assigned.
    #[must_use]
    pub fn completion_rate(&self) -> Option<f64> {
        if self.jobs_assigned == 0 {
            None
        } else {
            Some(self.jobs_succeeded as f64 / self.jobs_assigned as f64)
        }
    }

    /// Error rate = failed / (succeeded + failed).
    ///
    /// Returns `None` when no jobs have completed.
    #[must_use]
    pub fn error_rate(&self) -> Option<f64> {
        let completed = self.jobs_succeeded + self.jobs_failed;
        if completed == 0 {
            None
        } else {
            Some(self.jobs_failed as f64 / completed as f64)
        }
    }

    /// Mean processing time across all completed (succeeded + failed) jobs.
    ///
    /// Returns `None` when no jobs have completed.
    #[must_use]
    pub fn mean_processing_time(&self) -> Option<Duration> {
        let completed = self.jobs_succeeded + self.jobs_failed;
        if completed == 0 {
            None
        } else {
            Some(self.total_processing_time / completed as u32)
        }
    }

    /// Throughput in frames per second computed over the duration history
    /// window.  Returns `None` when no video jobs have been recorded.
    #[must_use]
    pub fn frames_per_second(&self) -> Option<f64> {
        let total_secs: f64 = self
            .duration_history
            .iter()
            .map(|r| r.wall_time.as_secs_f64())
            .sum();
        let total_frames: u64 = self
            .duration_history
            .iter()
            .map(|r| r.frames_processed)
            .sum();
        if total_secs <= 0.0 || total_frames == 0 {
            None
        } else {
            Some(total_frames as f64 / total_secs)
        }
    }

    /// Average CPU utilisation over the retained resource snapshot window.
    ///
    /// Returns `None` when no snapshots have been recorded.
    #[must_use]
    pub fn avg_cpu_utilisation(&self) -> Option<f64> {
        avg_field(&self.resource_history, |s| s.cpu_utilisation)
    }

    /// Average memory utilisation over the retained resource snapshot window.
    ///
    /// Returns `None` when no snapshots have been recorded.
    #[must_use]
    pub fn avg_memory_utilisation(&self) -> Option<f64> {
        avg_field(&self.resource_history, |s| s.memory_utilisation)
    }

    /// Average GPU utilisation over the retained resource snapshot window.
    ///
    /// Returns `None` when no snapshots have been recorded.
    #[must_use]
    pub fn avg_gpu_utilisation(&self) -> Option<f64> {
        avg_field(&self.resource_history, |s| s.gpu_utilisation)
    }

    /// p50 (median) job wall-clock duration from the duration history window.
    ///
    /// Returns `None` when the history is empty.
    #[must_use]
    pub fn p50_processing_time(&self) -> Option<Duration> {
        percentile_duration(&self.duration_history, 50)
    }

    /// p95 job wall-clock duration from the duration history window.
    ///
    /// Returns `None` when the history is empty.
    #[must_use]
    pub fn p95_processing_time(&self) -> Option<Duration> {
        percentile_duration(&self.duration_history, 95)
    }

    /// Idle time since the last recorded activity.
    ///
    /// Returns `None` if the worker has never processed a job.
    #[must_use]
    pub fn idle_duration(&self) -> Option<Duration> {
        self.last_activity_at.map(|t| t.elapsed())
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn avg_field<T: Clone, F: Fn(&T) -> f64>(buf: &RingBuffer<T>, f: F) -> Option<f64> {
    if buf.is_empty() {
        return None;
    }
    let sum: f64 = buf.iter().map(&f).sum();
    Some(sum / buf.len() as f64)
}

fn percentile_duration(buf: &RingBuffer<CompletedJobRecord>, pct: u8) -> Option<Duration> {
    if buf.is_empty() {
        return None;
    }
    let mut durations: Vec<Duration> = buf.iter().map(|r| r.wall_time).collect();
    durations.sort_unstable();
    let idx = ((pct as usize).saturating_mul(durations.len())).saturating_sub(1) / 100;
    durations.get(idx).copied()
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Registry that holds a [`WorkerMetricsRecord`] for every registered worker.
///
/// Thread-safe by design: wrap in a `Mutex` or `RwLock` for concurrent use.
#[derive(Debug)]
pub struct WorkerMetricsRegistry {
    records: HashMap<WorkerId, WorkerMetricsRecord>,
    resource_window: usize,
    duration_window: usize,
}

impl WorkerMetricsRegistry {
    /// Create a new registry.
    ///
    /// * `resource_window` — resource snapshot ring-buffer size per worker.
    /// * `duration_window` — job-duration ring-buffer size per worker.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] when either window is zero.
    pub fn new(resource_window: usize, duration_window: usize) -> crate::Result<Self> {
        if resource_window == 0 || duration_window == 0 {
            return Err(FarmError::InvalidConfig(
                "resource_window and duration_window must be > 0".into(),
            ));
        }
        Ok(Self {
            records: HashMap::new(),
            resource_window,
            duration_window,
        })
    }

    /// Register a new worker.  Returns an error if the worker is already
    /// registered.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] when the worker is already known.
    pub fn register(&mut self, worker_id: WorkerId) -> crate::Result<()> {
        if self.records.contains_key(&worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {} is already registered",
                worker_id
            )));
        }
        let record = WorkerMetricsRecord::new(
            worker_id.clone(),
            self.resource_window,
            self.duration_window,
        )?;
        self.records.insert(worker_id, record);
        Ok(())
    }

    /// Remove a worker from the registry.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the worker is unknown.
    pub fn deregister(&mut self, worker_id: &WorkerId) -> crate::Result<WorkerMetricsRecord> {
        self.records
            .remove(worker_id)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {} not found", worker_id)))
    }

    /// Retrieve an immutable reference to a worker's record.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the worker is unknown.
    pub fn get(&self, worker_id: &WorkerId) -> crate::Result<&WorkerMetricsRecord> {
        self.records
            .get(worker_id)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {} not found", worker_id)))
    }

    /// Retrieve a mutable reference to a worker's record.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the worker is unknown.
    pub fn get_mut(&mut self, worker_id: &WorkerId) -> crate::Result<&mut WorkerMetricsRecord> {
        self.records
            .get_mut(worker_id)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {} not found", worker_id)))
    }

    /// Number of registered workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.records.len()
    }

    /// Return an iterator over all worker records.
    pub fn iter(&self) -> impl Iterator<Item = (&WorkerId, &WorkerMetricsRecord)> {
        self.records.iter()
    }

    /// Return a [`FarmWideSummary`] aggregating key statistics across all
    /// workers.
    #[must_use]
    pub fn farm_wide_summary(&self) -> FarmWideSummary {
        let worker_count = self.records.len();
        let total_assigned: u64 = self.records.values().map(|r| r.jobs_assigned).sum();
        let total_succeeded: u64 = self.records.values().map(|r| r.jobs_succeeded).sum();
        let total_failed: u64 = self.records.values().map(|r| r.jobs_failed).sum();
        let total_timed_out: u64 = self.records.values().map(|r| r.jobs_timed_out).sum();

        let completion_rate = if total_assigned == 0 {
            None
        } else {
            Some(total_succeeded as f64 / total_assigned as f64)
        };

        let completed = total_succeeded + total_failed;
        let error_rate = if completed == 0 {
            None
        } else {
            Some(total_failed as f64 / completed as f64)
        };

        let avg_cpu = {
            let vals: Vec<f64> = self
                .records
                .values()
                .filter_map(|r| r.avg_cpu_utilisation())
                .collect();
            if vals.is_empty() {
                None
            } else {
                Some(vals.iter().sum::<f64>() / vals.len() as f64)
            }
        };

        FarmWideSummary {
            worker_count,
            total_jobs_assigned: total_assigned,
            total_jobs_succeeded: total_succeeded,
            total_jobs_failed: total_failed,
            total_jobs_timed_out: total_timed_out,
            farm_completion_rate: completion_rate,
            farm_error_rate: error_rate,
            avg_cpu_utilisation: avg_cpu,
        }
    }

    /// Rank workers by completion rate (highest first).  Workers with no jobs
    /// assigned appear last.
    #[must_use]
    pub fn rank_by_completion_rate(&self) -> Vec<(&WorkerId, f64)> {
        let mut entries: Vec<(&WorkerId, f64)> = self
            .records
            .iter()
            .map(|(id, rec)| (id, rec.completion_rate().unwrap_or(0.0)))
            .collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }
}

// ---------------------------------------------------------------------------
// Farm-wide summary
// ---------------------------------------------------------------------------

/// Aggregated statistics across all workers in the farm.
#[derive(Debug, Clone)]
pub struct FarmWideSummary {
    /// Total number of registered workers.
    pub worker_count: usize,
    /// Total jobs assigned across all workers.
    pub total_jobs_assigned: u64,
    /// Total jobs that succeeded.
    pub total_jobs_succeeded: u64,
    /// Total jobs that failed.
    pub total_jobs_failed: u64,
    /// Total jobs that timed out.
    pub total_jobs_timed_out: u64,
    /// Farm-wide completion rate (succeeded / assigned).
    pub farm_completion_rate: Option<f64>,
    /// Farm-wide error rate (failed / completed).
    pub farm_error_rate: Option<f64>,
    /// Average CPU utilisation across all workers with snapshots.
    pub avg_cpu_utilisation: Option<f64>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker() -> WorkerId {
        WorkerId::new("test-worker-1")
    }

    fn make_record() -> WorkerMetricsRecord {
        WorkerMetricsRecord::new(make_worker(), 10, 20).expect("should create record")
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf: RingBuffer<u32> = RingBuffer::new(3).expect("capacity 3 is valid");
        buf.push(1);
        buf.push(2);
        buf.push(3);
        let items: Vec<u32> = buf.iter().copied().collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut buf: RingBuffer<u32> = RingBuffer::new(3).expect("capacity 3 is valid");
        buf.push(1);
        buf.push(2);
        buf.push(3);
        buf.push(4); // evicts 1
        let items: Vec<u32> = buf.iter().copied().collect();
        assert_eq!(items, vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_zero_capacity_errors() {
        assert!(RingBuffer::<u32>::new(0).is_err());
    }

    #[test]
    fn test_completion_rate_no_jobs() {
        let rec = make_record();
        assert!(rec.completion_rate().is_none());
    }

    #[test]
    fn test_completion_rate_with_jobs() {
        let mut rec = make_record();
        rec.record_job_assigned();
        rec.record_job_assigned();
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(10),
            succeeded: true,
            frames_processed: 100,
        });
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(5),
            succeeded: false,
            frames_processed: 0,
        });
        // 1 succeeded, 2 assigned → 0.5
        let rate = rec.completion_rate().expect("should have rate");
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_error_rate() {
        let mut rec = make_record();
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(10),
            succeeded: true,
            frames_processed: 0,
        });
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(10),
            succeeded: false,
            frames_processed: 0,
        });
        let rate = rec.error_rate().expect("should have error rate");
        assert!((rate - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mean_processing_time() {
        let mut rec = make_record();
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(10),
            succeeded: true,
            frames_processed: 0,
        });
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(20),
            succeeded: true,
            frames_processed: 0,
        });
        let mean = rec.mean_processing_time().expect("should have mean");
        assert_eq!(mean, Duration::from_secs(15));
    }

    #[test]
    fn test_frames_per_second() {
        let mut rec = make_record();
        rec.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(10),
            succeeded: true,
            frames_processed: 250,
        });
        let fps = rec.frames_per_second().expect("should have fps");
        assert!((fps - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_resource_snapshot_averaging() {
        let mut rec = make_record();
        rec.record_resource_snapshot(ResourceSnapshot::new(0.4, 0.5, 0.0, 0));
        rec.record_resource_snapshot(ResourceSnapshot::new(0.6, 0.5, 0.0, 0));
        let avg_cpu = rec.avg_cpu_utilisation().expect("should have avg cpu");
        assert!((avg_cpu - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_registry_register_deregister() {
        let mut reg = WorkerMetricsRegistry::new(5, 10).expect("valid windows");
        let wid = WorkerId::new("w1");
        reg.register(wid.clone()).expect("first registration OK");
        assert_eq!(reg.worker_count(), 1);
        // duplicate registration must fail
        assert!(reg.register(wid.clone()).is_err());
        reg.deregister(&wid).expect("deregister OK");
        assert_eq!(reg.worker_count(), 0);
    }

    #[test]
    fn test_farm_wide_summary() {
        let mut reg = WorkerMetricsRegistry::new(5, 10).expect("valid windows");
        let w1 = WorkerId::new("w1");
        let w2 = WorkerId::new("w2");
        reg.register(w1.clone()).expect("ok");
        reg.register(w2.clone()).expect("ok");

        let r1 = reg.get_mut(&w1).expect("w1 exists");
        r1.record_job_assigned();
        r1.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(5),
            succeeded: true,
            frames_processed: 0,
        });

        let r2 = reg.get_mut(&w2).expect("w2 exists");
        r2.record_job_assigned();
        r2.record_job_completed(CompletedJobRecord {
            wall_time: Duration::from_secs(5),
            succeeded: false,
            frames_processed: 0,
        });

        let summary = reg.farm_wide_summary();
        assert_eq!(summary.worker_count, 2);
        assert_eq!(summary.total_jobs_assigned, 2);
        assert_eq!(summary.total_jobs_succeeded, 1);
        assert_eq!(summary.total_jobs_failed, 1);
        let cr = summary.farm_completion_rate.expect("completion rate");
        assert!((cr - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_p50_p95_processing_time() {
        let mut rec = make_record();
        for secs in [1u64, 2, 3, 4, 100] {
            rec.record_job_completed(CompletedJobRecord {
                wall_time: Duration::from_secs(secs),
                succeeded: true,
                frames_processed: 0,
            });
        }
        let p50 = rec.p50_processing_time().expect("p50 exists");
        let p95 = rec.p95_processing_time().expect("p95 exists");
        // p50 should be a middle value, p95 near the max
        assert!(p50 <= p95);
    }
}
