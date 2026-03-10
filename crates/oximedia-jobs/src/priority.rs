// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Priority-based job scheduling for oximedia-jobs.
//!
//! Implements a multi-level feedback queue (MLFQ) where jobs with assigned
//! priorities are dequeued from the highest-priority bucket first.  To
//! prevent starvation, jobs that have been waiting longer than the configured
//! `aging_threshold` are temporarily promoted to the next priority level
//! during the `dequeue` call.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Job priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
    Realtime = 4,
}

impl Priority {
    /// Scheduling weight – used for weighted-fair-queuing strategies.
    ///
    /// Returns 1, 2, 4, 8, or 16 for Low..Realtime respectively.
    #[must_use]
    pub fn weight(self) -> u32 {
        match self {
            Self::Low => 1,
            Self::Normal => 2,
            Self::High => 4,
            Self::Critical => 8,
            Self::Realtime => 16,
        }
    }

    /// Convert from a `u8` value (0 = Low, 4 = Realtime).
    #[must_use]
    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            0 => Some(Self::Low),
            1 => Some(Self::Normal),
            2 => Some(Self::High),
            3 => Some(Self::Critical),
            4 => Some(Self::Realtime),
            _ => None,
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Normal => "Normal",
            Self::High => "High",
            Self::Critical => "Critical",
            Self::Realtime => "Realtime",
        }
    }

    /// Return the priority one level above `self`, saturating at `Realtime`.
    fn promoted(self) -> Self {
        match self {
            Self::Low => Self::Normal,
            Self::Normal => Self::High,
            Self::High => Self::Critical,
            Self::Critical | Self::Realtime => Self::Realtime,
        }
    }

    /// Index into the 5-slot array used by `PriorityJobQueue`.
    fn index(self) -> usize {
        self as usize
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// PriorityJob
// ---------------------------------------------------------------------------

/// Global counter for unique job IDs across all queues in a process.
static GLOBAL_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// A job entry in the priority queue.
#[derive(Debug, Clone)]
pub struct PriorityJob<J> {
    pub job: J,
    pub priority: Priority,
    pub submission_time: Instant,
    pub deadline: Option<Instant>,
    pub job_id: u64,
}

impl<J> PriorityJob<J> {
    /// Wrap `job` at the given `priority`.
    pub fn new(job: J, priority: Priority) -> Self {
        Self {
            job,
            priority,
            submission_time: Instant::now(),
            deadline: None,
            job_id: GLOBAL_JOB_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Set an absolute deadline for this job.
    #[must_use]
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns `true` if the job's deadline has already passed.
    #[must_use]
    pub fn is_past_deadline(&self) -> bool {
        self.deadline.map(|d| d < Instant::now()).unwrap_or(false)
    }

    /// Time elapsed since the job was submitted.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.submission_time.elapsed()
    }
}

impl<J> PartialEq for PriorityJob<J> {
    fn eq(&self, other: &Self) -> bool {
        self.job_id == other.job_id
    }
}

impl<J> Eq for PriorityJob<J> {}

// ---------------------------------------------------------------------------
// PriorityJobQueue
// ---------------------------------------------------------------------------

/// Priority queue for jobs with fairness aging.
///
/// Uses a multi-level feedback queue (MLFQ) approach:
/// - Jobs start at their assigned priority
/// - Jobs that wait longer than `aging_threshold` get temporarily promoted
///   to prevent starvation
pub struct PriorityJobQueue<J> {
    /// One deque per priority level (index 0 = Low … 4 = Realtime).
    queues: [VecDeque<PriorityJob<J>>; 5],
    /// How long a job may wait before being bumped up one level.
    aging_threshold: Duration,
    next_job_id: u64,
    total_enqueued: u64,
    total_dequeued: u64,
}

impl<J: Clone> PriorityJobQueue<J> {
    /// Create a new queue with a 30-second aging threshold.
    pub fn new() -> Self {
        Self {
            queues: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            aging_threshold: Duration::from_secs(30),
            next_job_id: 1,
            total_enqueued: 0,
            total_dequeued: 0,
        }
    }

    /// Override the default aging threshold.
    #[must_use]
    pub fn with_aging_threshold(mut self, threshold: Duration) -> Self {
        self.aging_threshold = threshold;
        self
    }

    /// Enqueue `job` at `priority`.  Returns the assigned job ID.
    pub fn enqueue(&mut self, job: J, priority: Priority) -> u64 {
        let id = self.next_job_id;
        self.next_job_id += 1;
        let mut entry = PriorityJob::new(job, priority);
        entry.job_id = id;
        self.queues[priority.index()].push_back(entry);
        self.total_enqueued += 1;
        id
    }

    /// Enqueue `job` at `priority` with an absolute `deadline`.  Returns the
    /// assigned job ID.
    pub fn enqueue_with_deadline(&mut self, job: J, priority: Priority, deadline: Instant) -> u64 {
        let id = self.next_job_id;
        self.next_job_id += 1;
        let mut entry = PriorityJob::new(job, priority).with_deadline(deadline);
        entry.job_id = id;
        self.queues[priority.index()].push_back(entry);
        self.total_enqueued += 1;
        id
    }

    /// Apply aging: scan all lower-priority queues and promote jobs that have
    /// waited longer than `aging_threshold` to the next level.
    fn apply_aging(&mut self) {
        // Process levels 0..=3 (Low through Critical); Realtime cannot be promoted.
        for level in 0..4usize {
            let threshold = self.aging_threshold;
            // Drain aged jobs into a temporary buffer
            let mut promoted: Vec<PriorityJob<J>> = Vec::new();
            let queue = &mut self.queues[level];
            queue.retain(|job| {
                if job.age() >= threshold {
                    promoted.push(job.clone());
                    false
                } else {
                    true
                }
            });
            // Re-enqueue at the next level
            for mut job in promoted {
                job.priority = job.priority.promoted();
                self.queues[level + 1].push_front(job);
            }
        }
    }

    /// Dequeue the highest-priority job.
    ///
    /// Aging is applied before selection so that long-waiting low-priority
    /// jobs are not starved indefinitely.
    pub fn dequeue(&mut self) -> Option<PriorityJob<J>> {
        self.apply_aging();
        // Search from highest (Realtime=4) to lowest (Low=0)
        for level in (0..5usize).rev() {
            if let Some(job) = self.queues[level].pop_front() {
                self.total_dequeued += 1;
                return Some(job);
            }
        }
        None
    }

    /// Peek at the front of the highest non-empty queue without removing the
    /// job.  Aging is NOT applied during a peek.
    pub fn peek(&self) -> Option<&PriorityJob<J>> {
        for level in (0..5usize).rev() {
            if let Some(job) = self.queues[level].front() {
                return Some(job);
            }
        }
        None
    }

    /// Total number of queued jobs across all priority levels.
    pub fn len(&self) -> usize {
        self.queues.iter().map(VecDeque::len).sum()
    }

    /// Returns `true` if the queue has no pending jobs.
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(VecDeque::is_empty)
    }

    /// Number of jobs at each priority level (index 0 = Low, 4 = Realtime).
    pub fn counts(&self) -> [usize; 5] {
        [
            self.queues[0].len(),
            self.queues[1].len(),
            self.queues[2].len(),
            self.queues[3].len(),
            self.queues[4].len(),
        ]
    }

    /// Remove and discard all jobs whose deadline has already passed.
    ///
    /// Returns the number of jobs that were expired.
    pub fn expire_overdue(&mut self) -> usize {
        let mut count = 0;
        for queue in &mut self.queues {
            let before = queue.len();
            queue.retain(|job| !job.is_past_deadline());
            count += before - queue.len();
        }
        count
    }

    /// Return `(total_enqueued, total_dequeued)` lifetime counters.
    pub fn stats(&self) -> (u64, u64) {
        (self.total_enqueued, self.total_dequeued)
    }
}

impl<J: Clone> Default for PriorityJobQueue<J> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ------------------------------------------------------------------
    // Priority enum
    // ------------------------------------------------------------------

    #[test]
    fn test_priority_variants() {
        // All five variants must exist and be distinct
        let variants = [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
            Priority::Realtime,
        ];
        let names = ["Low", "Normal", "High", "Critical", "Realtime"];
        for (p, n) in variants.iter().zip(names.iter()) {
            assert_eq!(p.name(), *n);
        }
    }

    #[test]
    fn test_priority_from_u8() {
        assert_eq!(Priority::from_u8(0), Some(Priority::Low));
        assert_eq!(Priority::from_u8(1), Some(Priority::Normal));
        assert_eq!(Priority::from_u8(2), Some(Priority::High));
        assert_eq!(Priority::from_u8(3), Some(Priority::Critical));
        assert_eq!(Priority::from_u8(4), Some(Priority::Realtime));
        assert_eq!(Priority::from_u8(5), None);
        assert_eq!(Priority::from_u8(255), None);
    }

    #[test]
    fn test_priority_weight() {
        assert_eq!(Priority::Low.weight(), 1);
        assert_eq!(Priority::Normal.weight(), 2);
        assert_eq!(Priority::High.weight(), 4);
        assert_eq!(Priority::Critical.weight(), 8);
        assert_eq!(Priority::Realtime.weight(), 16);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low < Priority::Normal);
        assert!(Priority::Normal < Priority::High);
        assert!(Priority::High < Priority::Critical);
        assert!(Priority::Critical < Priority::Realtime);
        // Sanity check at extremes
        assert!(Priority::Low < Priority::Critical);
        assert!(Priority::Low < Priority::Realtime);
        assert!(Priority::Normal < Priority::Realtime);
    }

    #[test]
    fn test_priority_default() {
        assert_eq!(Priority::default(), Priority::Normal);
    }

    // ------------------------------------------------------------------
    // PriorityJob
    // ------------------------------------------------------------------

    #[test]
    fn test_priority_job_new() {
        let pj = PriorityJob::new(42u32, Priority::High);
        assert_eq!(pj.job, 42);
        assert_eq!(pj.priority, Priority::High);
        assert!(pj.deadline.is_none());
        assert!(!pj.is_past_deadline());
    }

    #[test]
    fn test_priority_job_with_deadline_future() {
        let future = Instant::now() + Duration::from_secs(60);
        let pj = PriorityJob::new("job", Priority::Normal).with_deadline(future);
        assert!(!pj.is_past_deadline(), "future deadline should not be past");
    }

    #[test]
    fn test_priority_job_with_deadline_past() {
        // Instant::now() is already in the past relative to the "now" checked inside
        // is_past_deadline, so use a timestamp from a tiny bit ago.
        let past = Instant::now() - Duration::from_millis(1);
        let pj = PriorityJob::new("job", Priority::Normal).with_deadline(past);
        assert!(pj.is_past_deadline(), "past deadline should be detected");
    }

    #[test]
    fn test_priority_job_age() {
        let pj = PriorityJob::new("x", Priority::Low);
        // age() must be >= 0 and < 1 second in a unit test
        assert!(pj.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_priority_job_equality() {
        let pj1 = PriorityJob::new(1u32, Priority::Low);
        let pj2 = PriorityJob::new(1u32, Priority::High); // same value, different priority
                                                          // IDs are different so they must be inequal
        assert_ne!(pj1, pj2);

        // A clone shares the same job_id
        let pj3 = pj1.clone();
        assert_eq!(pj1, pj3);
    }

    // ------------------------------------------------------------------
    // PriorityJobQueue
    // ------------------------------------------------------------------

    #[test]
    fn test_queue_enqueue_dequeue_respects_priority() {
        let mut q: PriorityJobQueue<&str> = PriorityJobQueue::new();
        q.enqueue("low", Priority::Low);
        q.enqueue("high", Priority::High);
        q.enqueue("normal", Priority::Normal);

        // Highest priority should come out first
        let first = q.dequeue().expect("first should be valid");
        assert_eq!(first.priority, Priority::High);
        let second = q.dequeue().expect("second should be valid");
        assert_eq!(second.priority, Priority::Normal);
        let third = q.dequeue().expect("third should be valid");
        assert_eq!(third.priority, Priority::Low);
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_queue_len_and_is_empty() {
        let mut q: PriorityJobQueue<u32> = PriorityJobQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.enqueue(1, Priority::Normal);
        q.enqueue(2, Priority::High);
        assert!(!q.is_empty());
        assert_eq!(q.len(), 2);

        q.dequeue();
        assert_eq!(q.len(), 1);
        q.dequeue();
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_counts() {
        let mut q: PriorityJobQueue<u8> = PriorityJobQueue::new();
        q.enqueue(1, Priority::Low);
        q.enqueue(2, Priority::Low);
        q.enqueue(3, Priority::Normal);
        q.enqueue(4, Priority::Realtime);

        let counts = q.counts();
        assert_eq!(counts[0], 2); // Low
        assert_eq!(counts[1], 1); // Normal
        assert_eq!(counts[2], 0); // High
        assert_eq!(counts[3], 0); // Critical
        assert_eq!(counts[4], 1); // Realtime
    }

    #[test]
    fn test_queue_peek() {
        let mut q: PriorityJobQueue<u32> = PriorityJobQueue::new();
        assert!(q.peek().is_none());

        q.enqueue(10, Priority::Low);
        q.enqueue(20, Priority::High);

        let peeked = q.peek().expect("peeked should be valid");
        assert_eq!(peeked.priority, Priority::High);
        // peek must not remove the job
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_stats() {
        let mut q: PriorityJobQueue<u32> = PriorityJobQueue::new();
        let (enq, deq) = q.stats();
        assert_eq!((enq, deq), (0, 0));

        q.enqueue(1, Priority::Normal);
        q.enqueue(2, Priority::Normal);
        let (enq, deq) = q.stats();
        assert_eq!((enq, deq), (2, 0));

        q.dequeue();
        let (enq, deq) = q.stats();
        assert_eq!((enq, deq), (2, 1));
    }

    #[test]
    fn test_queue_expire_overdue() {
        let mut q: PriorityJobQueue<u32> = PriorityJobQueue::new();
        // Job with a future deadline should NOT be expired
        q.enqueue_with_deadline(
            1,
            Priority::Normal,
            Instant::now() + Duration::from_secs(60),
        );
        // Job with a past deadline SHOULD be expired
        q.enqueue_with_deadline(2, Priority::High, Instant::now() - Duration::from_millis(1));
        // Job with no deadline should NOT be expired
        q.enqueue(3, Priority::Low);

        assert_eq!(q.len(), 3);
        let expired = q.expire_overdue();
        assert_eq!(expired, 1);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_aging_promotes_waiting_jobs() {
        // Use a very short aging threshold so we can trigger it instantly
        let mut q: PriorityJobQueue<u32> =
            PriorityJobQueue::new().with_aging_threshold(Duration::from_nanos(1)); // effectively zero

        q.enqueue(99, Priority::Low);
        // After sleeping just a tiny bit the job should be older than 1 ns
        std::thread::sleep(Duration::from_millis(5));

        // dequeue() applies aging before picking; the Low job should have been
        // promoted to Normal (or higher), but we only care it comes out.
        let job = q.dequeue().expect("job should be valid");
        assert_eq!(job.job, 99);
        // Its priority should have been promoted at least once
        assert!(job.priority > Priority::Low);
    }

    #[test]
    fn test_queue_default() {
        let q: PriorityJobQueue<()> = PriorityJobQueue::default();
        assert!(q.is_empty());
    }

    #[test]
    fn test_enqueue_returns_unique_ids() {
        let mut q: PriorityJobQueue<u32> = PriorityJobQueue::new();
        let id1 = q.enqueue(1, Priority::Normal);
        let id2 = q.enqueue(2, Priority::Normal);
        assert_ne!(id1, id2);
    }
}
