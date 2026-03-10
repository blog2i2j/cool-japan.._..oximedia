//! Job queue implementation

pub mod priority_queue;
pub mod scheduler;

use crate::error::{BatchError, Result};
use crate::job::BatchJob;
use crate::types::{JobId, JobState};
use dashmap::DashMap;
use parking_lot::RwLock;
use priority_queue::PriorityJobQueue;
use scheduler::Scheduler;
use std::sync::Arc;
use tokio::sync::Notify;

/// Job queue manager
pub struct JobQueue {
    /// Priority queue for jobs
    queue: Arc<PriorityJobQueue>,
    /// Job state tracking
    states: Arc<DashMap<JobId, JobState>>,
    /// Job storage
    jobs: Arc<DashMap<JobId, BatchJob>>,
    /// Scheduler for delayed/recurring jobs
    scheduler: Arc<RwLock<Scheduler>>,
    /// Notification for queue changes
    notify: Arc<Notify>,
}

impl JobQueue {
    /// Create a new job queue
    #[must_use]
    pub fn new() -> Self {
        Self {
            queue: Arc::new(PriorityJobQueue::new()),
            states: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            scheduler: Arc::new(RwLock::new(Scheduler::new())),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Enqueue a job
    ///
    /// # Arguments
    ///
    /// * `job` - The job to enqueue
    ///
    /// # Errors
    ///
    /// Returns an error if enqueueing fails
    #[allow(clippy::unused_async)]
    pub async fn enqueue(&self, job: BatchJob) -> Result<()> {
        let job_id = job.id.clone();

        // Store the job
        self.jobs.insert(job_id.clone(), job.clone());

        // Check schedule
        match &job.schedule.clone() {
            crate::types::Schedule::Immediate => {
                // Queue immediately
                self.queue.push(job)?;
                self.states.insert(job_id, JobState::Queued);
                self.notify.notify_one();
            }
            crate::types::Schedule::At(datetime) => {
                // Schedule for later
                self.scheduler.write().schedule_at(job, *datetime)?;
                self.states.insert(job_id, JobState::Pending);
            }
            crate::types::Schedule::After(secs) => {
                // Schedule after delay
                self.scheduler.write().schedule_after(job, *secs)?;
                self.states.insert(job_id, JobState::Pending);
            }
            crate::types::Schedule::Recurring { expression: _ } => {
                // Schedule recurring
                self.scheduler.write().schedule_recurring(job)?;
                self.states.insert(job_id, JobState::Pending);
            }
        }

        Ok(())
    }

    /// Dequeue the next job
    ///
    /// # Errors
    ///
    /// Returns an error if dequeueing fails
    pub async fn dequeue(&self) -> Result<Option<BatchJob>> {
        loop {
            // Check scheduler for due jobs
            let due_jobs = self.scheduler.write().get_due_jobs();
            for job in due_jobs {
                self.queue.push(job.clone())?;
                self.states.insert(job.id.clone(), JobState::Queued);
            }

            // Try to get a job from the queue
            if let Some(job) = self.queue.pop()? {
                return Ok(Some(job));
            }

            // Wait for notification or timeout
            tokio::select! {
                () = self.notify.notified() => {}
                () = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {}
            }
        }
    }

    /// Get job status
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    #[allow(clippy::unused_async)]
    pub async fn get_job_status(&self, job_id: &JobId) -> Result<JobState> {
        self.states
            .get(job_id)
            .map(|entry| *entry.value())
            .ok_or_else(|| BatchError::JobNotFound(job_id.to_string()))
    }

    /// Update job status
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    /// * `state` - New state
    pub fn update_status(&self, job_id: &JobId, state: JobState) {
        self.states.insert(job_id.clone(), state);
    }

    /// Cancel a job
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to cancel
    ///
    /// # Errors
    ///
    /// Returns an error if cancellation fails
    #[allow(clippy::unused_async)]
    pub async fn cancel_job(&self, job_id: &JobId) -> Result<()> {
        // Update state to cancelled
        self.states.insert(job_id.clone(), JobState::Cancelled);

        // Remove from queue if present
        self.queue.remove(job_id)?;

        // Remove from scheduler if present
        self.scheduler.write().cancel(job_id)?;

        Ok(())
    }

    /// Get all jobs
    #[must_use]
    pub fn get_all_jobs(&self) -> Vec<BatchJob> {
        self.jobs
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Get job by ID
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job
    #[must_use]
    pub fn get_job(&self, job_id: &JobId) -> Option<BatchJob> {
        self.jobs.get(job_id).map(|entry| entry.value().clone())
    }

    /// Get queue size
    #[must_use]
    pub fn size(&self) -> usize {
        self.queue.len()
    }
}

impl Default for JobQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::FileOperation;

    #[tokio::test]
    async fn test_job_queue_creation() {
        let queue = JobQueue::new();
        assert_eq!(queue.size(), 0);
    }

    #[tokio::test]
    async fn test_enqueue_job() {
        let queue = JobQueue::new();

        let job = BatchJob::new(
            "test-job".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let result = queue.enqueue(job).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_job_status() {
        let queue = JobQueue::new();

        let job = BatchJob::new(
            "test-job".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = job.id.clone();
        queue.enqueue(job).await.expect("failed to enqueue");

        let status = queue.get_job_status(&job_id).await;
        assert!(status.is_ok());
        assert_eq!(status.expect("status should be valid"), JobState::Queued);
    }

    #[tokio::test]
    async fn test_cancel_job() {
        let queue = JobQueue::new();

        let job = BatchJob::new(
            "test-job".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = job.id.clone();
        queue.enqueue(job).await.expect("failed to enqueue");
        queue
            .cancel_job(&job_id)
            .await
            .expect("await should be valid");

        let status = queue
            .get_job_status(&job_id)
            .await
            .expect("await should be valid");
        assert_eq!(status, JobState::Cancelled);
    }

    #[tokio::test]
    async fn test_get_all_jobs() {
        let queue = JobQueue::new();

        let job1 = BatchJob::new(
            "job1".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        let job2 = BatchJob::new(
            "job2".to_string(),
            crate::job::BatchOperation::FileOp {
                operation: FileOperation::Copy { overwrite: false },
            },
        );

        queue.enqueue(job1).await.expect("failed to enqueue");
        queue.enqueue(job2).await.expect("failed to enqueue");

        let all_jobs = queue.get_all_jobs();
        assert_eq!(all_jobs.len(), 2);
    }
}
