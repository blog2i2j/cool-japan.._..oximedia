//! Priority-aware job queue for the encoding farm coordinator.
//!
//! Jobs are ordered first by `JobPriority` (highest first) and then by
//! submission timestamp (oldest first) so that equal-priority jobs are
//! served in FIFO order.
//!
//! # Progressive job status tracking
//!
//! In addition to the plain `JobQueue`, this module provides
//! `ProgressTrackingQueue` which attaches live progress information to each
//! in-flight job:
//!
//! - Completion percentage (0.0 – 100.0)
//! - EWMA-smoothed throughput (frames / second or units / second)
//! - Estimated time to completion (ETA) derived from throughput and remaining work
//! - Deadline urgency flag (set when ETA > deadline)

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::{Duration, Instant};

// ── Priority ──────────────────────────────────────────────────────────────────

/// Priority level assigned to a farm job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum JobPriority {
    /// Background / bulk processing.
    Low = 0,
    /// Standard workload (default).
    #[default]
    Normal = 1,
    /// Time-sensitive processing.
    High = 2,
    /// Deadline-critical; pre-empts lower-priority work.
    Urgent = 3,
}

impl JobPriority {
    /// Numeric value of the priority level.
    #[must_use]
    pub fn value(self) -> u8 {
        self as u8
    }

    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Urgent => "urgent",
        }
    }
}

// ── FarmJob ───────────────────────────────────────────────────────────────────

/// A job submitted to the farm queue.
#[derive(Debug, Clone)]
pub struct FarmJob {
    /// Unique job identifier.
    pub job_id: String,
    /// Display name for this job.
    pub name: String,
    /// Scheduling priority.
    pub priority: JobPriority,
    /// Instant at which this job was submitted.
    pub submitted_at: Instant,
    /// Optional time-to-live; job is considered expired after this duration.
    pub ttl: Option<Duration>,
    /// Estimated processing time in seconds.
    pub estimated_seconds: u32,
    /// Optional epoch-seconds deadline for completion.
    pub deadline_epoch: Option<u64>,
    /// Total units of work (e.g. frames) in this job.
    pub total_units: u64,
}

impl FarmJob {
    /// Create a new farm job.
    #[must_use]
    pub fn new(
        job_id: impl Into<String>,
        name: impl Into<String>,
        priority: JobPriority,
        ttl: Option<Duration>,
        estimated_seconds: u32,
    ) -> Self {
        Self {
            job_id: job_id.into(),
            name: name.into(),
            priority,
            submitted_at: Instant::now(),
            ttl,
            estimated_seconds,
            deadline_epoch: None,
            total_units: 0,
        }
    }

    /// Builder: set a deadline (epoch seconds) and total unit count.
    #[must_use]
    pub fn with_deadline(mut self, deadline_epoch: u64, total_units: u64) -> Self {
        self.deadline_epoch = Some(deadline_epoch);
        self.total_units = total_units;
        self
    }

    /// Builder: set the total unit count without a deadline.
    #[must_use]
    pub fn with_total_units(mut self, total_units: u64) -> Self {
        self.total_units = total_units;
        self
    }

    /// Returns `true` if the job has exceeded its time-to-live.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            None => false,
            Some(ttl) => self.submitted_at.elapsed() > ttl,
        }
    }

    /// Age of the job since submission.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.submitted_at.elapsed()
    }
}

// ── Ordering wrapper for BinaryHeap ──────────────────────────────────────────

/// Internal heap entry that orders by priority (desc) then age (desc, i.e. older = higher).
struct HeapEntry {
    job: FarmJob,
    /// Negated submission-time nanos for tie-breaking (older jobs rank higher).
    neg_nanos: i128,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.job.priority == other.job.priority && self.neg_nanos == other.neg_nanos
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority wins; if equal, more-negative nanos (older job) wins.
        self.job
            .priority
            .cmp(&other.job.priority)
            .then_with(|| other.neg_nanos.cmp(&self.neg_nanos))
    }
}

// ── JobQueue ──────────────────────────────────────────────────────────────────

/// A max-heap job queue ordered by priority then submission age.
#[derive(Default)]
pub struct JobQueue {
    heap: BinaryHeap<HeapEntry>,
}

impl JobQueue {
    /// Create an empty queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a job to the queue.
    pub fn enqueue(&mut self, job: FarmJob) {
        let neg_nanos = -(job.submitted_at.elapsed().as_nanos() as i128);
        self.heap.push(HeapEntry { job, neg_nanos });
    }

    /// Remove and return the highest-priority job, or `None` if empty.
    pub fn dequeue(&mut self) -> Option<FarmJob> {
        self.heap.pop().map(|e| e.job)
    }

    /// Peek at the priority of the next job without removing it.
    #[must_use]
    pub fn peek_priority(&self) -> Option<JobPriority> {
        self.heap.peek().map(|e| e.job.priority)
    }

    /// Total number of jobs currently in the queue.
    #[must_use]
    pub fn count(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue has no pending jobs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Remove all expired jobs and return how many were purged.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.heap.len();
        let jobs: Vec<HeapEntry> = std::mem::take(&mut self.heap).into_iter().collect();
        self.heap = jobs.into_iter().filter(|e| !e.job.is_expired()).collect();
        before - self.heap.len()
    }
}

// ── Progressive job status tracking ──────────────────────────────────────────

/// Configuration for `ProgressTrackingQueue`.
#[derive(Debug, Clone)]
pub struct ProgressTrackingConfig {
    /// EWMA smoothing factor α ∈ (0, 1].  Higher = faster adaptation.
    pub ewma_alpha: f64,
    /// Number of recent throughput samples to keep in the rolling window.
    pub throughput_window: usize,
    /// If `true`, a job is flagged as urgent when its ETA exceeds its deadline.
    pub flag_deadline_breach: bool,
}

impl ProgressTrackingConfig {
    /// Conservative defaults: slow EWMA, 10-sample window.
    #[must_use]
    pub fn default_config() -> Self {
        Self {
            ewma_alpha: 0.2,
            throughput_window: 10,
            flag_deadline_breach: true,
        }
    }
}

impl Default for ProgressTrackingConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// A point-in-time progress snapshot for a single in-flight job.
#[derive(Debug, Clone)]
pub struct JobStatusSnapshot {
    /// Job identifier.
    pub job_id: String,
    /// Completion percentage in `[0.0, 100.0]`.
    pub percent_complete: f64,
    /// Processed units so far.
    pub units_done: u64,
    /// Total units in the job.
    pub total_units: u64,
    /// EWMA-smoothed throughput (units / second).  `None` if no samples yet.
    pub throughput_units_per_sec: Option<f64>,
    /// Estimated seconds remaining.  `None` if throughput is zero or unknown.
    pub eta_secs: Option<f64>,
    /// Epoch-seconds deadline, if set.
    pub deadline_epoch: Option<u64>,
    /// `true` when deadline is set and ETA would breach it.
    pub deadline_breached: bool,
    /// Epoch seconds when this snapshot was generated.
    pub snapshot_epoch: u64,
}

/// Internal per-job progress state.
struct JobProgress {
    /// Underlying job metadata.
    job: FarmJob,
    /// Processed units at the last update.
    units_done: u64,
    /// EWMA throughput estimate (units / second).
    ewma_throughput: Option<f64>,
    /// Rolling window of recent per-interval throughput samples (units/sec).
    throughput_samples: VecDeque<f64>,
    /// Epoch seconds of the last `update_progress` call.
    last_update_epoch: u64,
}

impl JobProgress {
    fn new(job: FarmJob) -> Self {
        Self {
            job,
            units_done: 0,
            ewma_throughput: None,
            throughput_samples: VecDeque::new(),
            last_update_epoch: 0,
        }
    }

    /// Ingest a new observation: `units_done` processed so far, at `now_epoch`.
    fn observe(&mut self, units_done: u64, now_epoch: u64, config: &ProgressTrackingConfig) {
        let elapsed_secs = now_epoch.saturating_sub(self.last_update_epoch);
        let units_delta = units_done.saturating_sub(self.units_done);

        if elapsed_secs > 0 && units_delta > 0 {
            let instant_rate = units_delta as f64 / elapsed_secs as f64;

            // EWMA update.
            let alpha = config.ewma_alpha.clamp(1e-9, 1.0);
            let new_ewma = match self.ewma_throughput {
                None => instant_rate,
                Some(prev) => alpha * instant_rate + (1.0 - alpha) * prev,
            };
            self.ewma_throughput = Some(new_ewma);

            // Rolling window.
            if self.throughput_samples.len() >= config.throughput_window {
                self.throughput_samples.pop_front();
            }
            self.throughput_samples.push_back(instant_rate);
        }

        self.units_done = units_done;
        self.last_update_epoch = now_epoch;
    }

    /// Build a snapshot at `now_epoch`.
    fn snapshot(&self, now_epoch: u64, config: &ProgressTrackingConfig) -> JobStatusSnapshot {
        let total = self.job.total_units;
        let done = self.units_done.min(total);

        let percent_complete = if total == 0 {
            0.0
        } else {
            (done as f64 / total as f64 * 100.0).clamp(0.0, 100.0)
        };

        let remaining = total.saturating_sub(done);
        let eta_secs = self.ewma_throughput.and_then(|rate| {
            if rate > 0.0 {
                Some(remaining as f64 / rate)
            } else {
                None
            }
        });

        let deadline_breached = config.flag_deadline_breach
            && self
                .job
                .deadline_epoch
                .zip(eta_secs)
                .map(|(deadline, eta)| {
                    // Breach if now + eta exceeds deadline.
                    now_epoch as f64 + eta > deadline as f64
                })
                .unwrap_or(false);

        JobStatusSnapshot {
            job_id: self.job.job_id.clone(),
            percent_complete,
            units_done: done,
            total_units: total,
            throughput_units_per_sec: self.ewma_throughput,
            eta_secs,
            deadline_epoch: self.job.deadline_epoch,
            deadline_breached,
            snapshot_epoch: now_epoch,
        }
    }
}

/// A job queue that tracks live progress, EWMA throughput, and ETA per job.
///
/// Jobs move from *pending* (in the priority heap) to *in-flight* (tracked in
/// a progress map) when dequeued.  Callers push progress updates via
/// [`ProgressTrackingQueue::update_progress`] and retrieve snapshots via [`ProgressTrackingQueue::snapshot`].
pub struct ProgressTrackingQueue {
    pending: BinaryHeap<HeapEntry>,
    in_flight: HashMap<String, JobProgress>,
    config: ProgressTrackingConfig,
}

impl ProgressTrackingQueue {
    /// Create an empty tracking queue.
    #[must_use]
    pub fn new(config: ProgressTrackingConfig) -> Self {
        Self {
            pending: BinaryHeap::new(),
            in_flight: HashMap::new(),
            config,
        }
    }

    /// Add a job to the pending queue.
    pub fn enqueue(&mut self, job: FarmJob) {
        let neg_nanos = -(job.submitted_at.elapsed().as_nanos() as i128);
        self.pending.push(HeapEntry { job, neg_nanos });
    }

    /// Dequeue the highest-priority pending job and move it to in-flight tracking.
    ///
    /// Returns the job ID of the dequeued job, or `None` if the queue is empty.
    pub fn dequeue_to_flight(&mut self, start_epoch: u64) -> Option<String> {
        let entry = self.pending.pop()?;
        let job_id = entry.job.job_id.clone();
        let mut progress = JobProgress::new(entry.job);
        progress.last_update_epoch = start_epoch;
        self.in_flight.insert(job_id.clone(), progress);
        Some(job_id)
    }

    /// Push a progress update for an in-flight job.
    ///
    /// `units_done` is the total processed units so far (monotonically
    /// non-decreasing).  `now_epoch` is the current epoch time in seconds.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `job_id` is not currently in-flight.
    pub fn update_progress(
        &mut self,
        job_id: &str,
        units_done: u64,
        now_epoch: u64,
    ) -> Result<(), String> {
        let config = self.config.clone();
        let progress = self
            .in_flight
            .get_mut(job_id)
            .ok_or_else(|| format!("job '{job_id}' is not in-flight"))?;
        progress.observe(units_done, now_epoch, &config);
        Ok(())
    }

    /// Retrieve a point-in-time snapshot for an in-flight job.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `job_id` is not currently in-flight.
    pub fn snapshot(&self, job_id: &str, now_epoch: u64) -> Result<JobStatusSnapshot, String> {
        let progress = self
            .in_flight
            .get(job_id)
            .ok_or_else(|| format!("job '{job_id}' is not in-flight"))?;
        Ok(progress.snapshot(now_epoch, &self.config))
    }

    /// Mark an in-flight job as complete and remove it from tracking.
    ///
    /// Returns the final snapshot before removal.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `job_id` is not currently in-flight.
    pub fn complete_job(
        &mut self,
        job_id: &str,
        now_epoch: u64,
    ) -> Result<JobStatusSnapshot, String> {
        let config = self.config.clone();
        let progress = self
            .in_flight
            .get(job_id)
            .ok_or_else(|| format!("job '{job_id}' is not in-flight"))?;
        let snap = progress.snapshot(now_epoch, &config);
        self.in_flight.remove(job_id);
        Ok(snap)
    }

    /// Number of pending (not yet dequeued) jobs.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of in-flight jobs.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Snapshots of all in-flight jobs at `now_epoch`.
    #[must_use]
    pub fn all_snapshots(&self, now_epoch: u64) -> Vec<JobStatusSnapshot> {
        self.in_flight
            .values()
            .map(|p| p.snapshot(now_epoch, &self.config))
            .collect()
    }

    /// Return all in-flight job IDs whose deadline is breached at `now_epoch`.
    #[must_use]
    pub fn breached_deadlines(&self, now_epoch: u64) -> Vec<String> {
        self.in_flight
            .values()
            .filter_map(|p| {
                let snap = p.snapshot(now_epoch, &self.config);
                if snap.deadline_breached {
                    Some(snap.job_id)
                } else {
                    None
                }
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn job(id: &str, priority: JobPriority) -> FarmJob {
        FarmJob::new(id, id, priority, None, 60)
    }

    fn expiring_job(id: &str) -> FarmJob {
        // TTL of 1 ns — expires after any measurable elapsed time.
        FarmJob::new(
            id,
            id,
            JobPriority::Normal,
            Some(Duration::from_nanos(1)),
            10,
        )
    }

    fn tracked_job(id: &str, total_units: u64, deadline: Option<u64>) -> FarmJob {
        let j = FarmJob::new(id, id, JobPriority::Normal, None, 60).with_total_units(total_units);
        match deadline {
            Some(d) => j.with_deadline(d, total_units),
            None => j,
        }
    }

    // --- JobPriority tests ---

    #[test]
    fn test_job_priority_value_low() {
        assert_eq!(JobPriority::Low.value(), 0);
    }

    #[test]
    fn test_job_priority_value_urgent() {
        assert_eq!(JobPriority::Urgent.value(), 3);
    }

    #[test]
    fn test_job_priority_label() {
        assert_eq!(JobPriority::High.label(), "high");
        assert_eq!(JobPriority::Normal.label(), "normal");
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Urgent > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    // --- FarmJob tests ---

    #[test]
    fn test_farm_job_not_expired_without_ttl() {
        let j = job("j1", JobPriority::Normal);
        assert!(!j.is_expired());
    }

    #[test]
    fn test_farm_job_expired_with_zero_ttl() {
        let j = expiring_job("j_exp");
        std::thread::sleep(Duration::from_millis(1));
        assert!(j.is_expired());
    }

    // --- JobQueue tests ---

    #[test]
    fn test_queue_starts_empty() {
        let q = JobQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.count(), 0);
    }

    #[test]
    fn test_queue_enqueue_increments_count() {
        let mut q = JobQueue::new();
        q.enqueue(job("j1", JobPriority::Normal));
        assert_eq!(q.count(), 1);
    }

    #[test]
    fn test_queue_dequeue_highest_priority_first() {
        let mut q = JobQueue::new();
        q.enqueue(job("low", JobPriority::Low));
        q.enqueue(job("high", JobPriority::High));
        q.enqueue(job("normal", JobPriority::Normal));
        let first = q.dequeue().expect("failed to dequeue");
        assert_eq!(first.priority, JobPriority::High);
    }

    #[test]
    fn test_queue_dequeue_returns_none_when_empty() {
        let mut q = JobQueue::new();
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_peek_priority() {
        let mut q = JobQueue::new();
        q.enqueue(job("j1", JobPriority::Urgent));
        assert_eq!(q.peek_priority(), Some(JobPriority::Urgent));
    }

    #[test]
    fn test_peek_priority_none_when_empty() {
        let q = JobQueue::new();
        assert!(q.peek_priority().is_none());
    }

    #[test]
    fn test_purge_expired_removes_expired_jobs() {
        let mut q = JobQueue::new();
        q.enqueue(expiring_job("exp1"));
        q.enqueue(expiring_job("exp2"));
        q.enqueue(job("keep", JobPriority::Normal));
        std::thread::sleep(Duration::from_millis(1));
        let purged = q.purge_expired();
        assert_eq!(purged, 2);
        assert_eq!(q.count(), 1);
    }

    #[test]
    fn test_purge_expired_no_expired_jobs() {
        let mut q = JobQueue::new();
        q.enqueue(job("j1", JobPriority::Normal));
        let purged = q.purge_expired();
        assert_eq!(purged, 0);
        assert_eq!(q.count(), 1);
    }

    #[test]
    fn test_multiple_enqueue_dequeue_order() {
        let mut q = JobQueue::new();
        q.enqueue(job("a", JobPriority::Normal));
        q.enqueue(job("b", JobPriority::Urgent));
        q.enqueue(job("c", JobPriority::Low));
        assert_eq!(
            q.dequeue().expect("failed to dequeue").priority,
            JobPriority::Urgent
        );
        assert_eq!(
            q.dequeue().expect("failed to dequeue").priority,
            JobPriority::Normal
        );
        assert_eq!(
            q.dequeue().expect("failed to dequeue").priority,
            JobPriority::Low
        );
    }

    // --- ProgressTrackingQueue tests ---

    #[test]
    fn test_ptq_starts_empty() {
        let ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        assert_eq!(ptq.pending_count(), 0);
        assert_eq!(ptq.in_flight_count(), 0);
    }

    #[test]
    fn test_ptq_enqueue_increments_pending() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 1000, None));
        assert_eq!(ptq.pending_count(), 1);
        assert_eq!(ptq.in_flight_count(), 0);
    }

    #[test]
    fn test_ptq_dequeue_to_flight_moves_job() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 1000, None));
        let id = ptq
            .dequeue_to_flight(1000)
            .expect("dequeue_to_flight failed");
        assert_eq!(id, "j1");
        assert_eq!(ptq.pending_count(), 0);
        assert_eq!(ptq.in_flight_count(), 1);
    }

    #[test]
    fn test_ptq_dequeue_to_flight_none_when_empty() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        assert!(ptq.dequeue_to_flight(0).is_none());
    }

    #[test]
    fn test_ptq_update_progress_and_snapshot_percent() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 1000, None));
        ptq.dequeue_to_flight(1000)
            .expect("dequeue_to_flight failed");
        // 250 units done out of 1000 = 25 %
        ptq.update_progress("j1", 250, 1010)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 1010).expect("snapshot failed");
        assert!((snap.percent_complete - 25.0).abs() < 0.01);
        assert_eq!(snap.units_done, 250);
        assert_eq!(snap.total_units, 1000);
    }

    #[test]
    fn test_ptq_snapshot_err_when_not_in_flight() {
        let ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        assert!(ptq.snapshot("missing", 0).is_err());
    }

    #[test]
    fn test_ptq_update_progress_err_when_not_in_flight() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        assert!(ptq.update_progress("missing", 10, 100).is_err());
    }

    #[test]
    fn test_ptq_ewma_throughput_populated_after_update() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 1000, None));
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        // Simulate 100 units in 10 seconds → 10 units/s instant rate.
        ptq.update_progress("j1", 100, 10)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 10).expect("snapshot failed");
        assert!(snap.throughput_units_per_sec.is_some());
        let tps = snap
            .throughput_units_per_sec
            .expect("throughput should be set");
        assert!(tps > 0.0);
    }

    #[test]
    fn test_ptq_eta_computed_from_throughput() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 1000, None));
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        // 100 units done in 10 seconds → 10 units/s; 900 remaining → ETA ≈ 90 s
        ptq.update_progress("j1", 100, 10)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 10).expect("snapshot failed");
        let eta = snap.eta_secs.expect("eta should be set");
        // EWMA alpha=0.2 so first observation equals instant rate; ETA = 900/10 = 90.
        assert!((eta - 90.0).abs() < 1.0);
    }

    #[test]
    fn test_ptq_deadline_breached_flag() {
        let config = ProgressTrackingConfig {
            ewma_alpha: 1.0, // instant adaption for determinism
            throughput_window: 5,
            flag_deadline_breach: true,
        };
        let mut ptq = ProgressTrackingQueue::new(config);
        // Deadline at t=50; now=10; 100 units done in 10 s → 10 u/s;
        // remaining=900 → ETA=90 s; 10+90=100 > 50 → breached.
        let j = FarmJob::new("j1", "j1", JobPriority::Normal, None, 60).with_deadline(50, 1000);
        ptq.enqueue(j);
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        ptq.update_progress("j1", 100, 10)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 10).expect("snapshot failed");
        assert!(snap.deadline_breached);
    }

    #[test]
    fn test_ptq_no_breach_when_flag_disabled() {
        let config = ProgressTrackingConfig {
            ewma_alpha: 1.0,
            throughput_window: 5,
            flag_deadline_breach: false,
        };
        let mut ptq = ProgressTrackingQueue::new(config);
        let j = FarmJob::new("j1", "j1", JobPriority::Normal, None, 60).with_deadline(50, 1000);
        ptq.enqueue(j);
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        ptq.update_progress("j1", 100, 10)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 10).expect("snapshot failed");
        assert!(!snap.deadline_breached);
    }

    #[test]
    fn test_ptq_complete_job_removes_from_flight() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 100, None));
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        ptq.update_progress("j1", 100, 5)
            .expect("update_progress failed");
        let snap = ptq.complete_job("j1", 5).expect("complete_job failed");
        assert_eq!(snap.job_id, "j1");
        assert_eq!(ptq.in_flight_count(), 0);
    }

    #[test]
    fn test_ptq_all_snapshots_returns_all_in_flight() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        for i in 0..3 {
            ptq.enqueue(tracked_job(&format!("j{i}"), 100, None));
            ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        }
        let snaps = ptq.all_snapshots(0);
        assert_eq!(snaps.len(), 3);
    }

    #[test]
    fn test_ptq_breached_deadlines_returns_correct_ids() {
        let config = ProgressTrackingConfig {
            ewma_alpha: 1.0,
            throughput_window: 5,
            flag_deadline_breach: true,
        };
        let mut ptq = ProgressTrackingQueue::new(config);
        // Job with tight deadline → will breach.
        let j_tight =
            FarmJob::new("tight", "tight", JobPriority::Normal, None, 60).with_deadline(20, 1000);
        // Job with generous deadline → will NOT breach.
        let j_ok =
            FarmJob::new("ok", "ok", JobPriority::Normal, None, 60).with_deadline(10_000, 100);
        ptq.enqueue(j_tight);
        ptq.enqueue(j_ok);
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        // 100 units done in 10 s = 10 u/s
        // tight: 900 remaining → ETA=90; now(10)+90=100>20 → breach
        // ok: 100 units total, 100 done → 0 remaining → no breach
        ptq.update_progress("tight", 100, 10)
            .expect("update_progress failed");
        ptq.update_progress("ok", 100, 10)
            .expect("update_progress failed");
        let breached = ptq.breached_deadlines(10);
        assert_eq!(breached.len(), 1);
        assert_eq!(breached[0], "tight");
    }

    #[test]
    fn test_ptq_percent_complete_100_when_all_done() {
        let mut ptq = ProgressTrackingQueue::new(ProgressTrackingConfig::default());
        ptq.enqueue(tracked_job("j1", 500, None));
        ptq.dequeue_to_flight(0).expect("dequeue_to_flight failed");
        ptq.update_progress("j1", 500, 5)
            .expect("update_progress failed");
        let snap = ptq.snapshot("j1", 5).expect("snapshot failed");
        assert!((snap.percent_complete - 100.0).abs() < 0.01);
    }
}
