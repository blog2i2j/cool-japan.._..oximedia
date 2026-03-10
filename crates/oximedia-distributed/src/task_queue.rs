//! Distributed task queue with priority ordering.
//!
//! This module implements a priority-based task queue for distributing
//! encoding tasks across worker nodes. Tasks are dequeued in priority
//! order, with FIFO ordering within the same priority level.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use uuid::Uuid;

/// Priority levels for distributed tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TaskPriority {
    /// Background tasks, lowest priority
    Background = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Urgent tasks, highest priority
    Urgent = 3,
    /// Critical infrastructure tasks
    Critical = 4,
}

impl TaskPriority {
    /// Returns the numeric value of this priority.
    #[must_use]
    pub fn value(&self) -> u8 {
        *self as u8
    }

    /// Returns true if this priority is higher than the other.
    #[must_use]
    pub fn is_higher_than(&self, other: &Self) -> bool {
        self.value() > other.value()
    }
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().cmp(&other.value())
    }
}

/// Status of a distributed task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TaskStatus {
    /// Task is waiting in the queue
    Pending,
    /// Task is assigned to a worker
    Assigned,
    /// Task is currently being executed
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
}

/// A distributed task that can be queued and assigned to workers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistributedTask {
    /// Unique task identifier
    pub id: Uuid,
    /// Task name/description
    pub name: String,
    /// Task priority
    pub priority: TaskPriority,
    /// Current status
    pub status: TaskStatus,
    /// Payload data (serialized task parameters)
    pub payload: String,
    /// Unix timestamp when the task was enqueued
    pub enqueued_at: i64,
    /// Maximum number of retry attempts
    pub max_retries: u32,
    /// Current retry count
    pub retry_count: u32,
    /// Optional deadline (unix timestamp)
    pub deadline: Option<i64>,
    /// Sequence number for FIFO within same priority
    sequence: u64,
}

impl DistributedTask {
    /// Creates a new distributed task.
    #[must_use]
    pub fn new(name: &str, priority: TaskPriority, payload: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            priority,
            status: TaskStatus::Pending,
            payload: payload.to_string(),
            enqueued_at: chrono::Utc::now().timestamp(),
            max_retries: 3,
            retry_count: 0,
            deadline: None,
            sequence: 0,
        }
    }

    /// Sets the maximum retry count.
    #[must_use]
    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Sets a deadline for the task.
    #[must_use]
    pub fn with_deadline(mut self, deadline: i64) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns true if the task can be retried.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }

    /// Returns true if the task has passed its deadline.
    #[must_use]
    pub fn is_past_deadline(&self, now: i64) -> bool {
        self.deadline.is_some_and(|d| now > d)
    }

    /// Increments the retry count and resets status to Pending.
    pub fn retry(&mut self) {
        self.retry_count += 1;
        self.status = TaskStatus::Pending;
    }

    /// Marks the task as running.
    pub fn mark_running(&mut self) {
        self.status = TaskStatus::Running;
    }

    /// Marks the task as completed.
    pub fn mark_completed(&mut self) {
        self.status = TaskStatus::Completed;
    }

    /// Marks the task as failed.
    pub fn mark_failed(&mut self) {
        self.status = TaskStatus::Failed;
    }
}

impl PartialEq for DistributedTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for DistributedTask {}

impl PartialOrd for DistributedTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistributedTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then lower sequence (FIFO)
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.sequence.cmp(&self.sequence), // lower seq = earlier
            other_ord => other_ord,
        }
    }
}

/// A priority-based task queue for distributed task scheduling.
///
/// Tasks are dequeued in priority order; within the same priority,
/// tasks follow FIFO ordering based on their enqueue sequence.
#[derive(Debug)]
pub struct TaskQueue {
    /// The priority heap
    heap: BinaryHeap<DistributedTask>,
    /// Monotonically increasing sequence counter
    next_sequence: u64,
    /// Maximum queue capacity (0 = unlimited)
    max_capacity: usize,
    /// Total tasks ever enqueued
    total_enqueued: u64,
    /// Total tasks ever dequeued
    total_dequeued: u64,
}

impl TaskQueue {
    /// Creates a new empty task queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_sequence: 0,
            max_capacity: 0,
            total_enqueued: 0,
            total_dequeued: 0,
        }
    }

    /// Creates a new task queue with a capacity limit.
    #[must_use]
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_sequence: 0,
            max_capacity,
            total_enqueued: 0,
            total_dequeued: 0,
        }
    }

    /// Enqueues a task. Returns false if the queue is at capacity.
    pub fn enqueue(&mut self, mut task: DistributedTask) -> bool {
        if self.max_capacity > 0 && self.heap.len() >= self.max_capacity {
            return false;
        }
        task.sequence = self.next_sequence;
        self.next_sequence += 1;
        self.total_enqueued += 1;
        self.heap.push(task);
        true
    }

    /// Dequeues the highest-priority task.
    ///
    /// Returns `None` if the queue is empty.
    pub fn dequeue(&mut self) -> Option<DistributedTask> {
        let task = self.heap.pop()?;
        self.total_dequeued += 1;
        Some(task)
    }

    /// Peeks at the highest-priority task without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&DistributedTask> {
        self.heap.peek()
    }

    /// Returns the number of tasks in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns true if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns total tasks ever enqueued.
    #[must_use]
    pub fn total_enqueued(&self) -> u64 {
        self.total_enqueued
    }

    /// Returns total tasks ever dequeued.
    #[must_use]
    pub fn total_dequeued(&self) -> u64 {
        self.total_dequeued
    }

    /// Drains all tasks from the queue in priority order.
    pub fn drain(&mut self) -> Vec<DistributedTask> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(task) = self.heap.pop() {
            result.push(task);
        }
        self.total_dequeued += result.len() as u64;
        result
    }

    /// Removes tasks that have passed their deadline.
    pub fn remove_expired(&mut self, now: i64) -> Vec<DistributedTask> {
        let mut remaining = Vec::new();
        let mut expired = Vec::new();
        while let Some(task) = self.heap.pop() {
            if task.is_past_deadline(now) {
                expired.push(task);
            } else {
                remaining.push(task);
            }
        }
        for task in remaining {
            self.heap.push(task);
        }
        expired
    }
}

impl Default for TaskQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::Urgent);
        assert!(TaskPriority::Urgent > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Background);
    }

    #[test]
    fn test_task_priority_is_higher_than() {
        assert!(TaskPriority::Critical.is_higher_than(&TaskPriority::High));
        assert!(!TaskPriority::Normal.is_higher_than(&TaskPriority::High));
    }

    #[test]
    fn test_task_creation() {
        let task =
            DistributedTask::new("encode_video", TaskPriority::Normal, "{\"file\":\"a.mp4\"}");
        assert_eq!(task.name, "encode_video");
        assert_eq!(task.priority, TaskPriority::Normal);
        assert_eq!(task.status, TaskStatus::Pending);
        assert_eq!(task.retry_count, 0);
    }

    #[test]
    fn test_task_with_deadline() {
        let task = DistributedTask::new("t1", TaskPriority::High, "{}").with_deadline(9999);
        assert_eq!(task.deadline, Some(9999));
        assert!(!task.is_past_deadline(9998));
        assert!(task.is_past_deadline(10000));
    }

    #[test]
    fn test_task_retry() {
        let mut task = DistributedTask::new("t1", TaskPriority::Normal, "{}").with_max_retries(2);
        assert!(task.can_retry());
        task.retry();
        assert_eq!(task.retry_count, 1);
        task.retry();
        assert!(!task.can_retry());
    }

    #[test]
    fn test_task_lifecycle() {
        let mut task = DistributedTask::new("t1", TaskPriority::Normal, "{}");
        assert_eq!(task.status, TaskStatus::Pending);
        task.mark_running();
        assert_eq!(task.status, TaskStatus::Running);
        task.mark_completed();
        assert_eq!(task.status, TaskStatus::Completed);
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("t1", TaskPriority::Normal, "{}"));
        assert_eq!(q.len(), 1);
        let t = q.dequeue().expect("dequeue should return a task");
        assert_eq!(t.name, "t1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_priority_order() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("low", TaskPriority::Background, "{}"));
        q.enqueue(DistributedTask::new("high", TaskPriority::High, "{}"));
        q.enqueue(DistributedTask::new("normal", TaskPriority::Normal, "{}"));
        let first = q.dequeue().expect("dequeue should return a task");
        assert_eq!(first.name, "high");
        let second = q.dequeue().expect("dequeue should return a task");
        assert_eq!(second.name, "normal");
        let third = q.dequeue().expect("dequeue should return a task");
        assert_eq!(third.name, "low");
    }

    #[test]
    fn test_queue_fifo_within_same_priority() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("first", TaskPriority::Normal, "{}"));
        q.enqueue(DistributedTask::new("second", TaskPriority::Normal, "{}"));
        q.enqueue(DistributedTask::new("third", TaskPriority::Normal, "{}"));
        assert_eq!(
            q.dequeue().expect("dequeue should return a task").name,
            "first"
        );
        assert_eq!(
            q.dequeue().expect("dequeue should return a task").name,
            "second"
        );
        assert_eq!(
            q.dequeue().expect("dequeue should return a task").name,
            "third"
        );
    }

    #[test]
    fn test_queue_capacity_limit() {
        let mut q = TaskQueue::with_capacity(2);
        assert!(q.enqueue(DistributedTask::new("t1", TaskPriority::Normal, "{}")));
        assert!(q.enqueue(DistributedTask::new("t2", TaskPriority::Normal, "{}")));
        assert!(!q.enqueue(DistributedTask::new("t3", TaskPriority::Normal, "{}")));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_peek() {
        let mut q = TaskQueue::new();
        assert!(q.peek().is_none());
        q.enqueue(DistributedTask::new("t1", TaskPriority::High, "{}"));
        assert_eq!(q.peek().expect("peek should return a value").name, "t1");
        assert_eq!(q.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_queue_drain() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("t1", TaskPriority::Normal, "{}"));
        q.enqueue(DistributedTask::new("t2", TaskPriority::High, "{}"));
        let drained = q.drain();
        assert_eq!(drained.len(), 2);
        assert!(q.is_empty());
        assert_eq!(drained[0].name, "t2"); // high priority first
    }

    #[test]
    fn test_queue_remove_expired() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("expired", TaskPriority::Normal, "{}").with_deadline(100));
        q.enqueue(DistributedTask::new("alive", TaskPriority::Normal, "{}").with_deadline(9999));
        q.enqueue(DistributedTask::new(
            "no_deadline",
            TaskPriority::Normal,
            "{}",
        ));
        let expired = q.remove_expired(200);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].name, "expired");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_counters() {
        let mut q = TaskQueue::new();
        q.enqueue(DistributedTask::new("t1", TaskPriority::Normal, "{}"));
        q.enqueue(DistributedTask::new("t2", TaskPriority::Normal, "{}"));
        let _ = q.dequeue();
        assert_eq!(q.total_enqueued(), 2);
        assert_eq!(q.total_dequeued(), 1);
    }

    #[test]
    fn test_task_mark_failed() {
        let mut task = DistributedTask::new("t1", TaskPriority::Normal, "{}");
        task.mark_failed();
        assert_eq!(task.status, TaskStatus::Failed);
    }
}
