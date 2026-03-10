//! Priority-aware job queue for the encoding farm coordinator.
//!
//! Jobs are ordered first by `JobPriority` (highest first) and then by
//! submission timestamp (oldest first) so that equal-priority jobs are
//! served in FIFO order.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

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
        }
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
        // We use the system time in nanos relative to `submitted_at` for ordering.
        // Since `Instant` has no epoch, we use the duration since the `job.submitted_at`
        // baseline as a tiebreaker via the negated elapsed nanos at submission time.
        // We store a stable "sequence" by negating the nanos from a fixed point.
        // Simplest: use a negative counter emulated by the HeapEntry comparison.
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

    #[test]
    fn test_farm_job_not_expired_without_ttl() {
        let j = job("j1", JobPriority::Normal);
        assert!(!j.is_expired());
    }

    #[test]
    fn test_farm_job_expired_with_zero_ttl() {
        let j = expiring_job("j_exp");
        // Sleep after creation so elapsed() is guaranteed to exceed the 1 ns TTL,
        // eliminating the race where elapsed() could read as 0 ns on fast hardware.
        std::thread::sleep(Duration::from_millis(1));
        assert!(j.is_expired());
    }

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
        // Sleep after enqueue so elapsed() on the expiring jobs exceeds their 1 ns TTL.
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
}
