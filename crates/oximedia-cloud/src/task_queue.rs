#![allow(dead_code)]
//! Priority-based cloud task queue.
//!
//! Provides a [`CloudTaskQueue`] that accepts [`CloudTask`] items tagged with a
//! [`QueuePriority`] and serves them in priority-then-FIFO order.  A
//! [`dequeue_batch`](CloudTaskQueue::dequeue_batch) helper returns up to *N*
//! items in a single call, which suits bulk worker dispatch.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Priority
// ─────────────────────────────────────────────────────────────────────────────

/// Scheduling priority for a cloud task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QueuePriority {
    /// Lowest priority — run after all others.
    Low = 0,
    /// Normal background workload.
    Normal = 1,
    /// Elevated priority — ahead of Normal tasks.
    High = 2,
    /// Immediately jump to the front of the queue.
    Critical = 3,
}

impl QueuePriority {
    /// Human-readable label for the priority level.
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }

    /// Returns `true` if this priority is at least as high as `other`.
    pub fn at_least(self, other: Self) -> bool {
        self >= other
    }
}

impl std::fmt::Display for QueuePriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CloudTask
// ─────────────────────────────────────────────────────────────────────────────

/// A unit of work submitted to a [`CloudTaskQueue`].
#[derive(Debug, Clone, PartialEq)]
pub struct CloudTask {
    /// Unique task identifier.
    pub id: u64,
    /// Scheduling priority.
    pub priority: QueuePriority,
    /// Arbitrary task payload (serialised command, job spec, …).
    pub payload: Vec<u8>,
    /// Wall-clock time at which the task was submitted.
    pub submitted_at: Instant,
    /// Optional time-to-live; the task is discarded after this duration.
    pub ttl: Option<Duration>,
}

impl CloudTask {
    /// Create a new task with the given id, priority and payload.
    pub fn new(id: u64, priority: QueuePriority, payload: Vec<u8>) -> Self {
        Self {
            id,
            priority,
            payload,
            submitted_at: Instant::now(),
            ttl: None,
        }
    }

    /// Attach a time-to-live to this task.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Returns `true` when the task's TTL has been exceeded.
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            Some(ttl) => self.submitted_at.elapsed() >= ttl,
            None => false,
        }
    }

    /// Age of the task since submission.
    pub fn age(&self) -> Duration {
        self.submitted_at.elapsed()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CloudTaskQueue
// ─────────────────────────────────────────────────────────────────────────────

/// Error variants for [`CloudTaskQueue`] operations.
#[derive(Debug, PartialEq, Eq)]
pub enum TaskQueueError {
    /// The queue has reached its configured capacity.
    QueueFull,
    /// The queue contains no ready tasks.
    QueueEmpty,
    /// The requested batch size is zero.
    InvalidBatchSize,
}

impl std::fmt::Display for TaskQueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "task queue is full"),
            Self::QueueEmpty => write!(f, "task queue is empty"),
            Self::InvalidBatchSize => write!(f, "batch size must be greater than zero"),
        }
    }
}

/// A priority-ordered queue of [`CloudTask`] items.
///
/// Tasks are stored in four internal lanes — one per [`QueuePriority`] — so
/// that higher-priority tasks are always served before lower-priority ones,
/// regardless of submission order.
#[derive(Debug)]
pub struct CloudTaskQueue {
    /// Per-priority lanes: index 0 = Low … index 3 = Critical.
    lanes: [VecDeque<CloudTask>; 4],
    /// Maximum total tasks across all lanes.
    capacity: usize,
    /// Monotonically increasing id counter.
    next_id: u64,
}

impl CloudTaskQueue {
    /// Create a new queue with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            lanes: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            capacity,
            next_id: 1,
        }
    }

    // Lane index for a priority (mirrors the discriminant).
    fn lane(priority: QueuePriority) -> usize {
        priority as usize
    }

    /// Total number of tasks currently held (excluding expired tasks).
    pub fn len(&self) -> usize {
        self.lanes.iter().map(|l| l.len()).sum()
    }

    /// Returns `true` when the queue holds no tasks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Submit a task at the specified priority.
    ///
    /// Expired tasks are purged from the queue before the capacity check so
    /// that the caller can always make room by waiting for TTLs to elapse.
    pub fn enqueue(
        &mut self,
        priority: QueuePriority,
        payload: Vec<u8>,
    ) -> Result<u64, TaskQueueError> {
        self.purge_expired();
        if self.len() >= self.capacity {
            return Err(TaskQueueError::QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        let task = CloudTask::new(id, priority, payload);
        self.lanes[Self::lane(priority)].push_back(task);
        Ok(id)
    }

    /// Submit a pre-built [`CloudTask`] directly.
    pub fn enqueue_task(&mut self, task: CloudTask) -> Result<(), TaskQueueError> {
        self.purge_expired();
        if self.len() >= self.capacity {
            return Err(TaskQueueError::QueueFull);
        }
        let lane = Self::lane(task.priority);
        self.lanes[lane].push_back(task);
        Ok(())
    }

    /// Dequeue the next highest-priority, non-expired task (Critical → Low).
    pub fn dequeue(&mut self) -> Result<CloudTask, TaskQueueError> {
        self.purge_expired();
        // Iterate lanes from highest to lowest priority.
        for lane in (0..4).rev() {
            if let Some(task) = self.lanes[lane].pop_front() {
                return Ok(task);
            }
        }
        Err(TaskQueueError::QueueEmpty)
    }

    /// Dequeue up to `max_count` tasks in priority order.
    ///
    /// Returns an error if `max_count` is zero; returns a (possibly shorter)
    /// `Vec` if fewer than `max_count` tasks are available.
    pub fn dequeue_batch(&mut self, max_count: usize) -> Result<Vec<CloudTask>, TaskQueueError> {
        if max_count == 0 {
            return Err(TaskQueueError::InvalidBatchSize);
        }
        self.purge_expired();
        let mut batch = Vec::with_capacity(max_count);
        'outer: for lane in (0..4).rev() {
            while let Some(task) = self.lanes[lane].pop_front() {
                batch.push(task);
                if batch.len() >= max_count {
                    break 'outer;
                }
            }
        }
        Ok(batch)
    }

    /// Number of tasks waiting in the given priority lane.
    pub fn lane_depth(&self, priority: QueuePriority) -> usize {
        self.lanes[Self::lane(priority)].len()
    }

    /// Remove all expired tasks from every lane.
    fn purge_expired(&mut self) {
        for lane in &mut self.lanes {
            lane.retain(|t| !t.is_expired());
        }
    }

    /// Discard all tasks in every lane.
    pub fn clear(&mut self) {
        for lane in &mut self.lanes {
            lane.clear();
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_queue(cap: usize) -> CloudTaskQueue {
        CloudTaskQueue::new(cap)
    }

    #[test]
    fn test_priority_ordering() {
        assert!(QueuePriority::Critical > QueuePriority::High);
        assert!(QueuePriority::High > QueuePriority::Normal);
        assert!(QueuePriority::Normal > QueuePriority::Low);
    }

    #[test]
    fn test_priority_at_least() {
        assert!(QueuePriority::Critical.at_least(QueuePriority::High));
        assert!(QueuePriority::Normal.at_least(QueuePriority::Normal));
        assert!(!QueuePriority::Low.at_least(QueuePriority::Normal));
    }

    #[test]
    fn test_priority_label() {
        assert_eq!(QueuePriority::Low.label(), "low");
        assert_eq!(QueuePriority::Critical.label(), "critical");
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(QueuePriority::Normal.to_string(), "normal");
    }

    #[test]
    fn test_enqueue_returns_incrementing_ids() {
        let mut q = make_queue(10);
        let id1 = q
            .enqueue(QueuePriority::Normal, b"a".to_vec())
            .expect("id1 should be valid");
        let id2 = q
            .enqueue(QueuePriority::Normal, b"b".to_vec())
            .expect("id2 should be valid");
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_queue_empty_initially() {
        let q = make_queue(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_len_increases_on_enqueue() {
        let mut q = make_queue(10);
        q.enqueue(QueuePriority::Low, b"x".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::High, b"y".to_vec())
            .expect("test expectation failed");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_dequeue_empty_returns_error() {
        let mut q = make_queue(10);
        assert_eq!(q.dequeue(), Err(TaskQueueError::QueueEmpty));
    }

    #[test]
    fn test_dequeue_respects_priority() {
        let mut q = make_queue(10);
        q.enqueue(QueuePriority::Low, b"low".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::Critical, b"crit".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::Normal, b"norm".to_vec())
            .expect("test expectation failed");
        let first = q.dequeue().expect("first should be valid");
        assert_eq!(first.payload, b"crit");
        let second = q.dequeue().expect("second should be valid");
        assert_eq!(second.payload, b"norm");
    }

    #[test]
    fn test_dequeue_batch_returns_correct_count() {
        let mut q = make_queue(20);
        for _ in 0..5 {
            q.enqueue(QueuePriority::Normal, b"item".to_vec())
                .expect("test expectation failed");
        }
        let batch = q.dequeue_batch(3).expect("batch should be valid");
        assert_eq!(batch.len(), 3);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_dequeue_batch_zero_size_error() {
        let mut q = make_queue(10);
        assert_eq!(q.dequeue_batch(0), Err(TaskQueueError::InvalidBatchSize));
    }

    #[test]
    fn test_dequeue_batch_partial_fill() {
        let mut q = make_queue(10);
        q.enqueue(QueuePriority::Normal, b"only".to_vec())
            .expect("test expectation failed");
        let batch = q.dequeue_batch(5).expect("batch should be valid");
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_queue_full_error() {
        let mut q = make_queue(2);
        q.enqueue(QueuePriority::Normal, b"a".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::Normal, b"b".to_vec())
            .expect("test expectation failed");
        assert_eq!(
            q.enqueue(QueuePriority::Normal, b"c".to_vec()),
            Err(TaskQueueError::QueueFull)
        );
    }

    #[test]
    fn test_lane_depth() {
        let mut q = make_queue(20);
        q.enqueue(QueuePriority::High, b"h1".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::High, b"h2".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::Low, b"l1".to_vec())
            .expect("test expectation failed");
        assert_eq!(q.lane_depth(QueuePriority::High), 2);
        assert_eq!(q.lane_depth(QueuePriority::Low), 1);
        assert_eq!(q.lane_depth(QueuePriority::Normal), 0);
    }

    #[test]
    fn test_clear_empties_all_lanes() {
        let mut q = make_queue(20);
        q.enqueue(QueuePriority::Low, b"a".to_vec())
            .expect("test expectation failed");
        q.enqueue(QueuePriority::Critical, b"b".to_vec())
            .expect("test expectation failed");
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_task_not_expired_by_default() {
        let task = CloudTask::new(1, QueuePriority::Normal, b"data".to_vec());
        assert!(!task.is_expired());
    }

    #[test]
    fn test_task_with_long_ttl_not_expired() {
        let task = CloudTask::new(1, QueuePriority::Normal, b"data".to_vec())
            .with_ttl(Duration::from_secs(3600));
        assert!(!task.is_expired());
    }

    #[test]
    fn test_enqueue_task_direct() {
        let mut q = make_queue(10);
        let task = CloudTask::new(99, QueuePriority::High, b"direct".to_vec());
        q.enqueue_task(task).expect("enqueue_task should succeed");
        assert_eq!(q.len(), 1);
        assert_eq!(q.lane_depth(QueuePriority::High), 1);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(TaskQueueError::QueueFull.to_string(), "task queue is full");
        assert_eq!(
            TaskQueueError::QueueEmpty.to_string(),
            "task queue is empty"
        );
        assert_eq!(
            TaskQueueError::InvalidBatchSize.to_string(),
            "batch size must be greater than zero"
        );
    }
}
