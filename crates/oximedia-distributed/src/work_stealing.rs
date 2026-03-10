//! Work-stealing queue for task distribution.
//!
//! Implements a work-stealing scheduler where each worker thread has
//! its own double-ended queue. The owner pops tasks from the back
//! (LIFO) while idle workers steal from the front (FIFO) of other queues.

#![allow(dead_code)]

/// A task that can be stolen by other workers.
#[derive(Debug, Clone)]
pub struct StealableTask {
    /// Unique task identifier.
    pub id: u64,
    /// Priority value (higher = more important).
    pub priority: u32,
    /// Task payload (serialized command or descriptor).
    pub payload: String,
}

impl StealableTask {
    /// Create a new stealable task.
    #[must_use]
    pub fn new(id: u64, priority: u32, payload: impl Into<String>) -> Self {
        Self {
            id,
            priority,
            payload: payload.into(),
        }
    }

    /// Returns true if this task has a priority of 10 or higher.
    #[must_use]
    pub fn is_high_priority(&self) -> bool {
        self.priority >= 10
    }
}

/// A double-ended work queue owned by a single worker.
///
/// The owner pushes and pops from the back (LIFO); thieves steal
/// from the front (FIFO) to minimise cache invalidation.
#[derive(Debug)]
pub struct WorkQueue {
    /// The internal deque of tasks (front = steal end, back = owner end).
    pub deque: Vec<StealableTask>,
    /// Identifier of the owning worker.
    pub owner_id: u32,
}

impl WorkQueue {
    /// Create a new empty work queue for the given owner.
    #[must_use]
    pub fn new(owner_id: u32) -> Self {
        Self {
            deque: Vec::new(),
            owner_id,
        }
    }

    /// Push a task onto the owner's end of the queue (back).
    pub fn push(&mut self, task: StealableTask) {
        self.deque.push(task);
    }

    /// Pop a task from the owner's end of the queue (back, LIFO).
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<StealableTask> {
        self.deque.pop()
    }

    /// Steal a task from the thief's end of the queue (front, FIFO).
    ///
    /// Returns `None` if the queue is empty.
    pub fn steal(&mut self) -> Option<StealableTask> {
        if self.deque.is_empty() {
            None
        } else {
            Some(self.deque.remove(0))
        }
    }

    /// Returns the number of tasks in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    /// Returns true if the queue has no tasks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    /// Returns all high-priority tasks (without removing them).
    #[must_use]
    pub fn high_priority_tasks(&self) -> Vec<&StealableTask> {
        self.deque.iter().filter(|t| t.is_high_priority()).collect()
    }
}

/// A work-stealing scheduler managing multiple worker queues.
#[derive(Debug, Default)]
pub struct WorkStealingScheduler {
    /// One queue per worker, indexed by position.
    pub queues: Vec<WorkQueue>,
}

impl WorkStealingScheduler {
    /// Create a new empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self { queues: Vec::new() }
    }

    /// Add a new queue for the given `owner_id`.
    ///
    /// If a queue for `owner_id` already exists this is a no-op.
    pub fn add_queue(&mut self, owner_id: u32) {
        if !self.queues.iter().any(|q| q.owner_id == owner_id) {
            self.queues.push(WorkQueue::new(owner_id));
        }
    }

    /// Submit a task to the queue owned by `owner_id`.
    ///
    /// Returns `false` if no queue for `owner_id` exists.
    pub fn submit_task(&mut self, owner_id: u32, task: StealableTask) -> bool {
        if let Some(queue) = self.queues.iter_mut().find(|q| q.owner_id == owner_id) {
            queue.push(task);
            true
        } else {
            false
        }
    }

    /// Steal a task from the queue owned by `target_id`.
    ///
    /// Returns `None` if the queue doesn't exist or is empty.
    pub fn steal_from(&mut self, target_id: u32) -> Option<StealableTask> {
        self.queues
            .iter_mut()
            .find(|q| q.owner_id == target_id)
            .and_then(WorkQueue::steal)
    }

    /// Pop a task for the given `owner_id` (owner's own pop, LIFO).
    pub fn pop_for(&mut self, owner_id: u32) -> Option<StealableTask> {
        self.queues
            .iter_mut()
            .find(|q| q.owner_id == owner_id)
            .and_then(WorkQueue::pop)
    }

    /// Returns the total number of pending tasks across all queues.
    #[must_use]
    pub fn total_pending(&self) -> usize {
        self.queues.iter().map(WorkQueue::len).sum()
    }

    /// Returns the number of queues registered.
    #[must_use]
    pub fn queue_count(&self) -> usize {
        self.queues.len()
    }

    /// Find the busiest queue (most tasks) and return its owner ID.
    #[must_use]
    pub fn busiest_owner(&self) -> Option<u32> {
        self.queues
            .iter()
            .max_by_key(|q| q.len())
            .filter(|q| !q.is_empty())
            .map(|q| q.owner_id)
    }

    /// Find the most idle queue (fewest tasks) and return its owner ID.
    #[must_use]
    pub fn idlest_owner(&self) -> Option<u32> {
        self.queues
            .iter()
            .min_by_key(|q| q.len())
            .map(|q| q.owner_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: u64, priority: u32) -> StealableTask {
        StealableTask::new(id, priority, format!("payload_{}", id))
    }

    #[test]
    fn test_stealable_task_is_high_priority() {
        assert!(task(1, 10).is_high_priority());
        assert!(task(2, 15).is_high_priority());
        assert!(!task(3, 9).is_high_priority());
        assert!(!task(4, 0).is_high_priority());
    }

    #[test]
    fn test_work_queue_push_pop_lifo() {
        let mut q = WorkQueue::new(1);
        q.push(task(1, 5));
        q.push(task(2, 5));
        q.push(task(3, 5));

        // Owner pops LIFO (last pushed = first out)
        assert_eq!(q.pop().expect("pop should return a value").id, 3);
        assert_eq!(q.pop().expect("pop should return a value").id, 2);
        assert_eq!(q.pop().expect("pop should return a value").id, 1);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_work_queue_steal_fifo() {
        let mut q = WorkQueue::new(1);
        q.push(task(1, 5));
        q.push(task(2, 5));
        q.push(task(3, 5));

        // Thief steals FIFO (first pushed = first stolen)
        assert_eq!(q.steal().expect("steal should return a task").id, 1);
        assert_eq!(q.steal().expect("steal should return a task").id, 2);
        assert_eq!(q.steal().expect("steal should return a task").id, 3);
        assert!(q.steal().is_none());
    }

    #[test]
    fn test_work_queue_len_and_is_empty() {
        let mut q = WorkQueue::new(1);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.push(task(1, 5));
        q.push(task(2, 5));
        assert_eq!(q.len(), 2);
        assert!(!q.is_empty());
    }

    #[test]
    fn test_work_queue_high_priority_tasks() {
        let mut q = WorkQueue::new(1);
        q.push(task(1, 5));
        q.push(task(2, 10));
        q.push(task(3, 15));
        q.push(task(4, 3));

        let hi = q.high_priority_tasks();
        assert_eq!(hi.len(), 2);
    }

    #[test]
    fn test_scheduler_add_queue() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.add_queue(1);
        sched.add_queue(2);
        assert_eq!(sched.queue_count(), 3);

        // Duplicate should be ignored
        sched.add_queue(1);
        assert_eq!(sched.queue_count(), 3);
    }

    #[test]
    fn test_scheduler_submit_task() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);

        assert!(sched.submit_task(0, task(1, 5)));
        assert_eq!(sched.total_pending(), 1);

        // Owner 99 does not exist
        assert!(!sched.submit_task(99, task(2, 5)));
        assert_eq!(sched.total_pending(), 1);
    }

    #[test]
    fn test_scheduler_steal_from() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.submit_task(0, task(1, 5));
        sched.submit_task(0, task(2, 5));

        // Steal FIFO (task 1 was pushed first)
        let stolen = sched.steal_from(0).expect("steal_from should succeed");
        assert_eq!(stolen.id, 1);
        assert_eq!(sched.total_pending(), 1);
    }

    #[test]
    fn test_scheduler_steal_from_empty() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        assert!(sched.steal_from(0).is_none());
    }

    #[test]
    fn test_scheduler_steal_from_missing_queue() {
        let mut sched = WorkStealingScheduler::new();
        assert!(sched.steal_from(42).is_none());
    }

    #[test]
    fn test_scheduler_pop_for() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.submit_task(0, task(1, 5));
        sched.submit_task(0, task(2, 5));

        // Owner pops LIFO (task 2 was pushed last)
        let t = sched.pop_for(0).expect("pop_for should return a task");
        assert_eq!(t.id, 2);
    }

    #[test]
    fn test_scheduler_total_pending() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.add_queue(1);
        sched.submit_task(0, task(1, 5));
        sched.submit_task(0, task(2, 5));
        sched.submit_task(1, task(3, 5));

        assert_eq!(sched.total_pending(), 3);
    }

    #[test]
    fn test_scheduler_busiest_owner() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.add_queue(1);
        sched.submit_task(0, task(1, 5));
        sched.submit_task(0, task(2, 5));
        sched.submit_task(1, task(3, 5));

        assert_eq!(sched.busiest_owner(), Some(0));
    }

    #[test]
    fn test_scheduler_idlest_owner() {
        let mut sched = WorkStealingScheduler::new();
        sched.add_queue(0);
        sched.add_queue(1);
        sched.submit_task(0, task(1, 5));
        sched.submit_task(0, task(2, 5));

        assert_eq!(sched.idlest_owner(), Some(1));
    }
}
