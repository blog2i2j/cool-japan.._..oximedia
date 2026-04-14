//! Async compute queue for overlapping compute and transfer operations.
//!
//! Provides a CPU-side task queue where compute jobs can be submitted by
//! `task_id` and polled for completion.  On the CPU fallback backend, tasks
//! are executed synchronously on submission; the queue abstraction
//! future-proofs the API for true async GPU execution when a WGPU device is
//! available.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::async_compute::AsyncComputeQueue;
//!
//! let mut queue = AsyncComputeQueue::new();
//! queue.submit(1, vec![0x01, 0x02]);
//! let result = queue.poll(1);
//! assert!(result.is_some());
//! assert_eq!(result.unwrap(), vec![0x01, 0x02]);
//! // Polling a second time returns None (result already consumed).
//! assert!(queue.poll(1).is_none());
//! ```

use std::collections::HashMap;

// ── Task state ────────────────────────────────────────────────────────────────

/// Lifecycle state of a submitted compute task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskState {
    /// Submitted, waiting for GPU execution to begin.
    Pending,
    /// GPU execution in progress (stub: immediately transitions to Complete).
    Running,
    /// Execution finished; result data is available.
    Complete,
    /// Execution failed; error message is stored.
    Failed(String),
}

/// Internal record for a tracked compute task.
#[derive(Debug)]
struct TaskRecord {
    state: TaskState,
    /// Payload supplied at submit time (also used as the result on CPU path).
    data: Vec<u8>,
}

// ── AsyncComputeQueue ─────────────────────────────────────────────────────────

/// Lightweight async compute task queue.
///
/// In the CPU-stub backend, tasks complete synchronously; the API is
/// designed to be drop-in replaceable with an actual GPU async queue once
/// a WGPU device is available.
#[derive(Debug, Default)]
pub struct AsyncComputeQueue {
    /// Active tasks, keyed by caller-defined `task_id`.
    tasks: HashMap<u64, TaskRecord>,
    /// Monotonically increasing submission counter.
    pub submission_count: u64,
    /// Number of tasks that have been polled and returned a result.
    pub completed_count: u64,
}

impl AsyncComputeQueue {
    /// Create a new, empty async compute queue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Submit a compute task.
    ///
    /// * `task_id`  – Caller-defined identifier for this task.
    /// * `data`     – Input payload (or pre-computed output on CPU path).
    ///
    /// If a task with the same `task_id` already exists it is replaced.
    pub fn submit(&mut self, task_id: u64, data: Vec<u8>) {
        self.submission_count += 1;
        // On the CPU-stub path, execution is synchronous → mark as Complete
        // immediately so `poll()` can return the result on the next call.
        self.tasks.insert(
            task_id,
            TaskRecord {
                state: TaskState::Complete,
                data,
            },
        );
    }

    /// Poll for the result of a previously submitted task.
    ///
    /// Returns `Some(result)` if the task has completed, consuming the
    /// result from the queue (subsequent polls for the same `task_id`
    /// return `None`).  Returns `None` if the task is still pending/running
    /// or has already been consumed.
    pub fn poll(&mut self, task_id: u64) -> Option<Vec<u8>> {
        if let Some(record) = self.tasks.get(&task_id) {
            if record.state == TaskState::Complete {
                // Remove and return.
                let record = self.tasks.remove(&task_id)?;
                self.completed_count += 1;
                return Some(record.data);
            }
        }
        None
    }

    /// Query the current state of a task without consuming the result.
    ///
    /// Returns `None` if no task with that `task_id` exists (either never
    /// submitted or already consumed by [`Self::poll`]).
    #[must_use]
    pub fn state(&self, task_id: u64) -> Option<&TaskState> {
        self.tasks.get(&task_id).map(|r| &r.state)
    }

    /// Cancel a pending or running task.
    ///
    /// Returns `true` if the task was found and removed.
    pub fn cancel(&mut self, task_id: u64) -> bool {
        self.tasks.remove(&task_id).is_some()
    }

    /// Number of tasks currently tracked (pending, running, or complete).
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.tasks.len()
    }

    /// `true` if no tasks are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Mark a task as failed with an error message (useful for testing error
    /// paths).
    pub fn fail_task(&mut self, task_id: u64, error: String) {
        if let Some(record) = self.tasks.get_mut(&task_id) {
            record.state = TaskState::Failed(error);
        }
    }

    /// Returns `true` if the task with `task_id` has failed.
    #[must_use]
    pub fn is_failed(&self, task_id: u64) -> bool {
        matches!(
            self.tasks.get(&task_id).map(|r| &r.state),
            Some(TaskState::Failed(_))
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submit_and_poll_returns_data() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![10, 20, 30]);
        let result = q.poll(1);
        assert_eq!(result, Some(vec![10, 20, 30]));
    }

    #[test]
    fn test_poll_twice_returns_none_second_time() {
        let mut q = AsyncComputeQueue::new();
        q.submit(42, vec![1]);
        assert!(q.poll(42).is_some());
        assert!(q.poll(42).is_none());
    }

    #[test]
    fn test_poll_unknown_task_returns_none() {
        let mut q = AsyncComputeQueue::new();
        assert!(q.poll(99).is_none());
    }

    #[test]
    fn test_multiple_tasks_independent() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![0xAA]);
        q.submit(2, vec![0xBB]);
        assert_eq!(q.poll(2), Some(vec![0xBB]));
        assert_eq!(q.poll(1), Some(vec![0xAA]));
    }

    #[test]
    fn test_cancel_removes_task() {
        let mut q = AsyncComputeQueue::new();
        q.submit(7, vec![0xFF]);
        assert!(q.cancel(7));
        assert!(q.poll(7).is_none());
    }

    #[test]
    fn test_submission_count_increments() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![]);
        q.submit(2, vec![]);
        assert_eq!(q.submission_count, 2);
    }

    #[test]
    fn test_completed_count_increments_on_poll() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![1]);
        q.poll(1);
        assert_eq!(q.completed_count, 1);
    }

    #[test]
    fn test_state_complete_after_submit() {
        let q = {
            let mut q = AsyncComputeQueue::new();
            q.submit(5, vec![5]);
            q
        };
        assert_eq!(q.state(5), Some(&TaskState::Complete));
    }

    #[test]
    fn test_active_count_decreases_on_poll() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![]);
        q.submit(2, vec![]);
        assert_eq!(q.active_count(), 2);
        q.poll(1);
        assert_eq!(q.active_count(), 1);
    }

    #[test]
    fn test_is_empty_after_all_polled() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![1]);
        q.poll(1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_fail_task_marks_failed() {
        let mut q = AsyncComputeQueue::new();
        q.submit(3, vec![]);
        q.fail_task(3, "shader compile error".into());
        assert!(q.is_failed(3));
    }

    #[test]
    fn test_resubmit_replaces_previous() {
        let mut q = AsyncComputeQueue::new();
        q.submit(1, vec![0x01]);
        q.submit(1, vec![0x02]); // replace
        assert_eq!(q.poll(1), Some(vec![0x02]));
    }

    #[test]
    fn test_empty_payload_allowed() {
        let mut q = AsyncComputeQueue::new();
        q.submit(0, vec![]);
        assert_eq!(q.poll(0), Some(vec![]));
    }
}
