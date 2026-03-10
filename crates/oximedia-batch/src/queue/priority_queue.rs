//! Priority-based job queue

use crate::error::Result;
use crate::job::BatchJob;
use crate::types::JobId;
use parking_lot::Mutex;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Wrapper for priority-based ordering
#[derive(Clone)]
struct PriorityJob {
    job: BatchJob,
}

impl PartialEq for PriorityJob {
    fn eq(&self, other: &Self) -> bool {
        self.job.priority == other.job.priority
    }
}

impl Eq for PriorityJob {}

impl PartialOrd for PriorityJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityJob {
    fn cmp(&self, other: &Self) -> Ordering {
        self.job.priority.cmp(&other.job.priority)
    }
}

/// Priority job queue
pub struct PriorityJobQueue {
    heap: Mutex<BinaryHeap<PriorityJob>>,
}

impl PriorityJobQueue {
    /// Create a new priority job queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
        }
    }

    /// Push a job onto the queue
    ///
    /// # Arguments
    ///
    /// * `job` - The job to enqueue
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails
    pub fn push(&self, job: BatchJob) -> Result<()> {
        self.heap.lock().push(PriorityJob { job });
        Ok(())
    }

    /// Pop the highest priority job from the queue
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails
    pub fn pop(&self) -> Result<Option<BatchJob>> {
        Ok(self.heap.lock().pop().map(|pj| pj.job))
    }

    /// Remove a specific job from the queue
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to remove
    ///
    /// # Errors
    ///
    /// Returns an error if the operation fails
    pub fn remove(&self, job_id: &JobId) -> Result<()> {
        let mut heap = self.heap.lock();
        let jobs: Vec<PriorityJob> = heap.drain().collect();
        heap.clear();

        for pj in jobs {
            if pj.job.id != *job_id {
                heap.push(pj);
            }
        }

        Ok(())
    }

    /// Get the number of jobs in the queue
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.lock().len()
    }

    /// Check if the queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.lock().is_empty()
    }
}

impl Default for PriorityJobQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;
    use crate::types::Priority;

    #[test]
    fn test_priority_queue_creation() {
        let queue = PriorityJobQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_push_and_pop() {
        let queue = PriorityJobQueue::new();

        let mut job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        job.set_priority(Priority::Normal);

        queue.push(job).expect("failed to push");
        assert_eq!(queue.len(), 1);

        let popped = queue.pop().expect("failed to pop");
        assert!(popped.is_some());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let queue = PriorityJobQueue::new();

        let mut job_low = BatchJob::new(
            "low".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        job_low.set_priority(Priority::Low);

        let mut job_high = BatchJob::new(
            "high".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        job_high.set_priority(Priority::High);

        let mut job_normal = BatchJob::new(
            "normal".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );
        job_normal.set_priority(Priority::Normal);

        queue.push(job_low).expect("failed to push");
        queue.push(job_high.clone()).expect("failed to push");
        queue.push(job_normal).expect("failed to push");

        // Should pop high priority first
        let first = queue.pop().expect("failed to pop").expect("failed to pop");
        assert_eq!(first.name, "high");
    }

    #[test]
    fn test_remove_job() {
        let queue = PriorityJobQueue::new();

        let job = BatchJob::new(
            "test".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = job.id.clone();
        queue.push(job).expect("failed to push");
        assert_eq!(queue.len(), 1);

        queue.remove(&job_id).expect("failed to remove");
        assert_eq!(queue.len(), 0);
    }
}
