//! Comprehensive batch processing engine for `OxiMedia`
//!
//! This crate provides a production-ready batch processing system with:
//! - Job queuing and scheduling
//! - Worker pool management
//! - Template-based configuration
//! - Watch folder automation
//! - Distributed processing support
//! - REST API and CLI interfaces

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod api;
pub mod batch_report;
pub mod batch_runner;
pub mod batch_schedule;
pub mod checkpointing;
pub mod cli;
pub mod database;
pub mod dep_graph;
pub mod dependency;
pub mod error;
pub mod examples;
pub mod execution;
pub mod job;
pub mod job_archive;
pub mod metrics;
pub mod monitoring;
pub mod notifications;
pub mod operations;
pub mod output_collector;
pub mod pipeline_validator;
pub mod presets;
pub mod processor;
pub mod progress_tracker;
pub mod queue;
pub mod rate_limiter;
pub mod resource_estimator;
pub mod retry_policy;
pub mod script;
pub mod task_group;
pub mod template;
pub mod throttle;
pub mod types;
pub mod utils;
pub mod watch;

pub use error::{BatchError, Result};
pub use job::{BatchJob, BatchOperation, InputSpec, OutputSpec};
pub use types::{JobId, JobState, Priority, RetryPolicy};

use database::Database;
use execution::ExecutionEngine;
use queue::JobQueue;
use std::sync::Arc;

/// Main batch processing engine
pub struct BatchEngine {
    queue: Arc<JobQueue>,
    engine: Arc<ExecutionEngine>,
    database: Arc<Database>,
}

impl BatchEngine {
    /// Create a new batch processing engine
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to `SQLite` database file
    /// * `worker_count` - Number of worker threads
    ///
    /// # Errors
    ///
    /// Returns an error if database initialization fails
    pub fn new(db_path: &str, worker_count: usize) -> Result<Self> {
        let database = Arc::new(Database::new(db_path)?);
        let queue = Arc::new(JobQueue::new());
        let engine = Arc::new(ExecutionEngine::new(
            worker_count,
            Arc::clone(&queue),
            Arc::clone(&database),
        )?);

        Ok(Self {
            queue,
            engine,
            database,
        })
    }

    /// Submit a job to the queue
    ///
    /// # Arguments
    ///
    /// * `job` - The job to submit
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_job(&self, job: BatchJob) -> Result<JobId> {
        let job_id = job.id.clone();
        self.database.save_job(&job)?;
        self.queue.enqueue(job).await?;
        Ok(job_id)
    }

    /// Get job status
    ///
    /// # Arguments
    ///
    /// * `job_id` - ID of the job to query
    ///
    /// # Errors
    ///
    /// Returns an error if the job is not found
    pub async fn get_job_status(&self, job_id: &JobId) -> Result<JobState> {
        self.queue.get_job_status(job_id).await
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
    pub async fn cancel_job(&self, job_id: &JobId) -> Result<()> {
        self.queue.cancel_job(job_id).await
    }

    /// List all jobs
    ///
    /// # Errors
    ///
    /// Returns an error if database query fails
    pub fn list_jobs(&self) -> Result<Vec<BatchJob>> {
        self.database.list_jobs()
    }

    /// Start the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if engine startup fails
    pub async fn start(&self) -> Result<()> {
        self.engine.start().await
    }

    /// Stop the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if engine shutdown fails
    pub async fn stop(&self) -> Result<()> {
        self.engine.stop().await
    }

    /// Get queue reference
    #[must_use]
    pub fn queue(&self) -> Arc<JobQueue> {
        Arc::clone(&self.queue)
    }

    /// Get engine reference
    #[must_use]
    pub fn engine(&self) -> Arc<ExecutionEngine> {
        Arc::clone(&self.engine)
    }

    /// Get database reference
    #[must_use]
    pub fn database(&self) -> Arc<Database> {
        Arc::clone(&self.database)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_batch_engine_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_job_submission() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await;
        assert!(job_id.is_ok());
    }

    #[tokio::test]
    async fn test_job_status() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await.expect("failed to submit job");
        let status = engine.get_job_status(&job_id).await;
        assert!(status.is_ok());
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let job = BatchJob::new(
            "test-job".to_string(),
            BatchOperation::FileOp {
                operation: operations::FileOperation::Copy { overwrite: false },
            },
        );

        let job_id = engine.submit_job(job).await.expect("failed to submit job");
        let result = engine.cancel_job(&job_id).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_list_jobs() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");
        let engine = BatchEngine::new(db_path, 4).expect("failed to create");

        let jobs = engine.list_jobs();
        assert!(jobs.is_ok());
    }
}
