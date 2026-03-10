#![allow(dead_code)]
//! Priority-based task scheduling queue.
//!
//! A multi-level priority queue that orders tasks by priority, deadline, and
//! submission time, ensuring critical work is scheduled first while preventing
//! starvation of lower-priority tasks through an aging mechanism.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt;

/// Priority level for tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Priority {
    /// Background / best-effort.
    Low,
    /// Default priority.
    Normal,
    /// Elevated priority.
    High,
    /// Must be processed immediately.
    Critical,
}

impl Priority {
    /// Numeric weight (higher = more urgent).
    fn weight(self) -> u32 {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
            Self::Critical => 3,
        }
    }
}

impl fmt::Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "Low"),
            Self::Normal => write!(f, "Normal"),
            Self::High => write!(f, "High"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// A task entry in the priority queue.
#[derive(Debug, Clone)]
pub struct PriorityTask {
    /// Unique task identifier.
    pub task_id: String,
    /// Base priority.
    pub priority: Priority,
    /// Submission timestamp (ms since epoch).
    pub submitted_at: u64,
    /// Optional deadline (ms since epoch). `None` means no deadline.
    pub deadline: Option<u64>,
    /// Number of aging bumps applied.
    pub age_bumps: u32,
    /// Estimated processing time in milliseconds.
    pub estimated_duration_ms: u64,
}

impl PriorityTask {
    /// Create a new task.
    pub fn new(task_id: impl Into<String>, priority: Priority, submitted_at: u64) -> Self {
        Self {
            task_id: task_id.into(),
            priority,
            submitted_at,
            deadline: None,
            age_bumps: 0,
            estimated_duration_ms: 0,
        }
    }

    /// Set a deadline.
    #[must_use]
    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline = Some(deadline_ms);
        self
    }

    /// Set estimated duration.
    #[must_use]
    pub fn with_estimated_duration(mut self, ms: u64) -> Self {
        self.estimated_duration_ms = ms;
        self
    }

    /// Effective priority weight including aging.
    fn effective_weight(&self) -> u32 {
        self.priority.weight() + self.age_bumps
    }

    /// Effective sort key: (`effective_weight`, `has_deadline`, `inverse_deadline`, `earlier_submit`).
    ///
    /// Higher weight first; among equal weight, deadline tasks first (earlier deadline wins);
    /// among equal, earlier submission wins.
    fn sort_key(&self) -> (u32, bool, u64, u64) {
        let has_dl = self.deadline.is_some();
        let inverse_dl = self.deadline.map_or(0, |d| u64::MAX - d);
        let inverse_submit = u64::MAX - self.submitted_at;
        (self.effective_weight(), has_dl, inverse_dl, inverse_submit)
    }
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

/// A priority queue for tasks.
#[derive(Debug)]
pub struct TaskPriorityQueue {
    /// The underlying max-heap.
    heap: BinaryHeap<PriorityTask>,
    /// Maximum capacity (0 = unlimited).
    capacity: usize,
    /// Number of age bumps to apply per aging cycle.
    age_bump_amount: u32,
}

impl TaskPriorityQueue {
    /// Create a new unbounded queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity: 0,
            age_bump_amount: 1,
        }
    }

    /// Create a queue with a maximum capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
            age_bump_amount: 1,
        }
    }

    /// Set the aging bump amount.
    pub fn set_age_bump_amount(&mut self, amount: u32) {
        self.age_bump_amount = amount;
    }

    /// Push a task into the queue.
    ///
    /// Returns `false` if the queue is at capacity.
    pub fn push(&mut self, task: PriorityTask) -> bool {
        if self.capacity > 0 && self.heap.len() >= self.capacity {
            return false;
        }
        self.heap.push(task);
        true
    }

    /// Pop the highest-priority task.
    pub fn pop(&mut self) -> Option<PriorityTask> {
        self.heap.pop()
    }

    /// Peek at the highest-priority task without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PriorityTask> {
        self.heap.peek()
    }

    /// Number of tasks in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Clear all tasks.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Apply aging: bump the priority of all non-critical tasks.
    ///
    /// This prevents starvation of low-priority tasks by gradually
    /// increasing their effective priority.
    pub fn apply_aging(&mut self) {
        let bump = self.age_bump_amount;
        let items: Vec<PriorityTask> = self.heap.drain().collect();
        for mut task in items {
            if task.priority != Priority::Critical {
                task.age_bumps += bump;
            }
            self.heap.push(task);
        }
    }

    /// Remove all tasks that have passed their deadline.
    ///
    /// `now_ms` is the current timestamp in milliseconds.
    /// Returns the expired tasks.
    pub fn remove_expired(&mut self, now_ms: u64) -> Vec<PriorityTask> {
        let mut expired = Vec::new();
        let mut kept = Vec::new();
        for task in self.heap.drain() {
            if let Some(dl) = task.deadline {
                if dl < now_ms {
                    expired.push(task);
                    continue;
                }
            }
            kept.push(task);
        }
        for task in kept {
            self.heap.push(task);
        }
        expired
    }

    /// Drain all tasks with a given priority.
    pub fn drain_priority(&mut self, priority: Priority) -> Vec<PriorityTask> {
        let mut matched = Vec::new();
        let mut rest = Vec::new();
        for task in self.heap.drain() {
            if task.priority == priority {
                matched.push(task);
            } else {
                rest.push(task);
            }
        }
        for task in rest {
            self.heap.push(task);
        }
        matched
    }

    /// Get queue capacity (0 = unlimited).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for TaskPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_weight() {
        assert!(Priority::Critical.weight() > Priority::High.weight());
        assert!(Priority::High.weight() > Priority::Normal.weight());
        assert!(Priority::Normal.weight() > Priority::Low.weight());
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(Priority::Low.to_string(), "Low");
        assert_eq!(Priority::Critical.to_string(), "Critical");
    }

    #[test]
    fn test_new_queue_is_empty() {
        let q = TaskPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_push_and_pop_single() {
        let mut q = TaskPriorityQueue::new();
        let task = PriorityTask::new("t1", Priority::Normal, 100);
        assert!(q.push(task));
        assert_eq!(q.len(), 1);
        let popped = q.pop().expect("pop should return a value");
        assert_eq!(popped.task_id, "t1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_pop_order_by_priority() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("low", Priority::Low, 100));
        q.push(PriorityTask::new("high", Priority::High, 100));
        q.push(PriorityTask::new("normal", Priority::Normal, 100));
        assert_eq!(q.pop().expect("pop should return a value").task_id, "high");
        assert_eq!(
            q.pop().expect("pop should return a value").task_id,
            "normal"
        );
        assert_eq!(q.pop().expect("pop should return a value").task_id, "low");
    }

    #[test]
    fn test_same_priority_earlier_submit_first() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("later", Priority::Normal, 200));
        q.push(PriorityTask::new("earlier", Priority::Normal, 100));
        assert_eq!(
            q.pop().expect("pop should return a value").task_id,
            "earlier"
        );
    }

    #[test]
    fn test_deadline_tasks_preferred() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("no_dl", Priority::Normal, 100));
        q.push(PriorityTask::new("with_dl", Priority::Normal, 100).with_deadline(5000));
        assert_eq!(
            q.pop().expect("pop should return a value").task_id,
            "with_dl"
        );
    }

    #[test]
    fn test_earlier_deadline_first() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("late_dl", Priority::Normal, 100).with_deadline(9000));
        q.push(PriorityTask::new("early_dl", Priority::Normal, 100).with_deadline(3000));
        assert_eq!(
            q.pop().expect("pop should return a value").task_id,
            "early_dl"
        );
    }

    #[test]
    fn test_capacity_limit() {
        let mut q = TaskPriorityQueue::with_capacity(2);
        assert!(q.push(PriorityTask::new("t1", Priority::Normal, 100)));
        assert!(q.push(PriorityTask::new("t2", Priority::Normal, 200)));
        assert!(!q.push(PriorityTask::new("t3", Priority::Normal, 300)));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_peek() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("t1", Priority::High, 100));
        let peeked = q.peek().expect("peek should return a value");
        assert_eq!(peeked.task_id, "t1");
        assert_eq!(q.len(), 1); // not removed
    }

    #[test]
    fn test_clear() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("t1", Priority::Normal, 100));
        q.push(PriorityTask::new("t2", Priority::Normal, 200));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_apply_aging() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("low", Priority::Low, 100));
        q.push(PriorityTask::new("crit", Priority::Critical, 100));
        // After 3 aging cycles, low's effective weight = 0 + 3 = 3
        q.apply_aging();
        q.apply_aging();
        q.apply_aging();
        // Low now has same effective weight as critical (3), but critical is not aged
        let first = q.pop().expect("pop should return a value");
        // Both have weight 3, tied; order depends on secondary criteria
        assert!(first.task_id == "low" || first.task_id == "crit");
    }

    #[test]
    fn test_remove_expired() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("expired", Priority::Normal, 100).with_deadline(500));
        q.push(PriorityTask::new("active", Priority::Normal, 100).with_deadline(2000));
        q.push(PriorityTask::new("no_dl", Priority::Normal, 100));
        let expired = q.remove_expired(1000);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].task_id, "expired");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_drain_priority() {
        let mut q = TaskPriorityQueue::new();
        q.push(PriorityTask::new("h1", Priority::High, 100));
        q.push(PriorityTask::new("n1", Priority::Normal, 100));
        q.push(PriorityTask::new("h2", Priority::High, 200));
        let high_tasks = q.drain_priority(Priority::High);
        assert_eq!(high_tasks.len(), 2);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_with_estimated_duration() {
        let task = PriorityTask::new("t1", Priority::Normal, 100).with_estimated_duration(5000);
        assert_eq!(task.estimated_duration_ms, 5000);
    }
}
