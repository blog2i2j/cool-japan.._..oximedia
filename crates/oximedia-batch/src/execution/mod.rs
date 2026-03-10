//! Execution engine for batch processing

pub mod worker;

use crate::database::Database;
use crate::error::{BatchError, Result};
use crate::job::BatchJob;
use crate::operations::{
    file_ops::FileOperationExecutor, media_ops::MediaOperationExecutor, pipeline::PipelineExecutor,
    OperationExecutor,
};
use crate::queue::JobQueue;
use crate::types::JobState;
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::task::JoinHandle;
use worker::Worker;

/// Execution engine configuration
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Number of worker threads
    pub worker_count: usize,
    /// Maximum concurrent jobs per worker
    pub max_concurrent_jobs: usize,
    /// Job timeout in seconds
    pub job_timeout_secs: u64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get(),
            max_concurrent_jobs: 1,
            job_timeout_secs: 3600,
        }
    }
}

/// Batch job execution engine
pub struct ExecutionEngine {
    config: ExecutionConfig,
    queue: Arc<JobQueue>,
    database: Arc<Database>,
    workers: Arc<RwLock<Vec<Worker>>>,
    running: Arc<AtomicBool>,
    task_handles: Arc<RwLock<Vec<JoinHandle<()>>>>,
    file_executor: Arc<FileOperationExecutor>,
    media_executor: Arc<MediaOperationExecutor>,
    pipeline_executor: Arc<PipelineExecutor>,
}

impl ExecutionEngine {
    /// Create a new execution engine
    ///
    /// # Arguments
    ///
    /// * `worker_count` - Number of worker threads
    /// * `queue` - Job queue
    /// * `database` - Database for persistence
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn new(worker_count: usize, queue: Arc<JobQueue>, database: Arc<Database>) -> Result<Self> {
        let config = ExecutionConfig {
            worker_count,
            ..Default::default()
        };

        Ok(Self {
            config,
            queue,
            database,
            workers: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            task_handles: Arc::new(RwLock::new(Vec::new())),
            file_executor: Arc::new(FileOperationExecutor::new()),
            media_executor: Arc::new(MediaOperationExecutor::new()),
            pipeline_executor: Arc::new(PipelineExecutor::new()),
        })
    }

    /// Start the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if startup fails
    #[allow(clippy::unused_async)]
    pub async fn start(&self) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(BatchError::ExecutionError(
                "Engine already running".to_string(),
            ));
        }

        self.running.store(true, Ordering::SeqCst);

        // Spawn worker tasks
        for i in 0..self.config.worker_count {
            let worker = Worker::new(i, Arc::clone(&self.queue), Arc::clone(&self.database));
            self.workers.write().push(worker.clone());

            let queue = Arc::clone(&self.queue);
            let running = Arc::clone(&self.running);
            let file_executor = Arc::clone(&self.file_executor);
            let media_executor = Arc::clone(&self.media_executor);
            let pipeline_executor = Arc::clone(&self.pipeline_executor);
            let database = Arc::clone(&self.database);

            let handle = tokio::spawn(async move {
                Self::worker_loop(
                    worker,
                    queue,
                    running,
                    file_executor,
                    media_executor,
                    pipeline_executor,
                    database,
                )
                .await;
            });

            self.task_handles.write().push(handle);
        }

        tracing::info!(
            "Execution engine started with {} workers",
            self.config.worker_count
        );

        Ok(())
    }

    /// Stop the execution engine
    ///
    /// # Errors
    ///
    /// Returns an error if shutdown fails
    pub async fn stop(&self) -> Result<()> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        self.running.store(false, Ordering::SeqCst);

        // Wait for all workers to finish
        let handles = std::mem::take(&mut *self.task_handles.write());
        for handle in handles {
            let _ = handle.await;
        }

        tracing::info!("Execution engine stopped");

        Ok(())
    }

    async fn worker_loop(
        worker: Worker,
        queue: Arc<JobQueue>,
        running: Arc<AtomicBool>,
        file_executor: Arc<FileOperationExecutor>,
        media_executor: Arc<MediaOperationExecutor>,
        pipeline_executor: Arc<PipelineExecutor>,
        database: Arc<Database>,
    ) {
        while running.load(Ordering::SeqCst) {
            match queue.dequeue().await {
                Ok(Some(job)) => {
                    let job_id = job.id.clone();

                    // Update status to running
                    queue.update_status(&job_id, JobState::Running);
                    database.update_job_status(&job_id, JobState::Running).ok();

                    // Execute the job
                    let result = Self::execute_job(
                        &job,
                        &file_executor,
                        &media_executor,
                        &pipeline_executor,
                    )
                    .await;

                    match result {
                        Ok(_) => {
                            queue.update_status(&job_id, JobState::Completed);
                            database
                                .update_job_status(&job_id, JobState::Completed)
                                .ok();
                            tracing::info!("Job completed: {}", job_id);
                        }
                        Err(e) => {
                            queue.update_status(&job_id, JobState::Failed);
                            database.update_job_status(&job_id, JobState::Failed).ok();
                            database.log_job_error(&job_id, &e.to_string()).ok();
                            tracing::error!("Job failed: {} - {}", job_id, e);

                            // Handle retry logic
                            Self::handle_retry(&job, &queue, &database).await;
                        }
                    }
                }
                Ok(None) => {
                    // Queue is empty, wait a bit
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
                Err(e) => {
                    tracing::error!("Worker error: {}", e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                }
            }
        }

        tracing::info!("Worker {} stopped", worker.id());
    }

    async fn execute_job(
        job: &BatchJob,
        file_executor: &Arc<FileOperationExecutor>,
        media_executor: &Arc<MediaOperationExecutor>,
        pipeline_executor: &Arc<PipelineExecutor>,
    ) -> Result<Vec<PathBuf>> {
        // Resolve input files
        let input_files = Self::resolve_input_files(job)?;

        // Select appropriate executor
        match &job.operation {
            crate::job::BatchOperation::FileOp { .. } => {
                file_executor.execute(job, &input_files).await
            }
            crate::job::BatchOperation::Transcode { .. }
            | crate::job::BatchOperation::QualityCheck { .. }
            | crate::job::BatchOperation::Analyze { .. } => {
                media_executor.execute(job, &input_files).await
            }
            crate::job::BatchOperation::Pipeline { .. } => {
                pipeline_executor.execute(job, &input_files).await
            }
            crate::job::BatchOperation::Custom { .. } => Err(BatchError::ExecutionError(
                "Unsupported operation".to_string(),
            )),
        }
    }

    fn resolve_input_files(job: &BatchJob) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        for input_spec in &job.inputs {
            let pattern = &input_spec.pattern;

            // Use glob to find matching files
            let glob_pattern = if let Some(base_dir) = &input_spec.base_dir {
                format!("{}/{}", base_dir.display(), pattern)
            } else {
                pattern.clone()
            };

            for entry in
                glob::glob(&glob_pattern).map_err(|e| BatchError::PatternError(e.to_string()))?
            {
                match entry {
                    Ok(path) => {
                        if path.is_file() {
                            files.push(path);
                        } else if path.is_dir() && input_spec.recursive {
                            Self::collect_files_recursive(&path, &mut files)?;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Glob entry error: {}", e);
                    }
                }
            }
        }

        if files.is_empty() {
            return Err(BatchError::ValidationError(
                "No input files found".to_string(),
            ));
        }

        Ok(files)
    }

    fn collect_files_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) -> Result<()> {
        for entry in walkdir::WalkDir::new(dir) {
            let entry = entry.map_err(|e| BatchError::FileOperationError(e.to_string()))?;
            if entry.path().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }
        Ok(())
    }

    async fn handle_retry(job: &BatchJob, queue: &Arc<JobQueue>, database: &Arc<Database>) {
        if let Some(mut context) = job.context.clone() {
            context.increment_retry();

            if context.retry_count < job.retry.max_attempts {
                let delay = job.retry.get_delay(context.retry_count);
                tracing::info!("Retrying job {} after {} seconds", job.id, delay);

                tokio::time::sleep(tokio::time::Duration::from_secs(delay)).await;

                // Re-enqueue the job
                let mut retry_job = job.clone();
                retry_job.context = Some(context);
                let _ = queue.enqueue(retry_job).await;
            } else {
                tracing::error!("Job {} exhausted all retry attempts", job.id);
                database
                    .log_job_error(&job.id, "Retry attempts exhausted")
                    .ok();
            }
        }
    }

    /// Get number of active workers
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.read().len()
    }

    /// Check if engine is running
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_execution_engine_creation() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let database = Arc::new(Database::new(db_path).expect("failed to create database"));
        let queue = Arc::new(JobQueue::new());

        let engine = ExecutionEngine::new(4, queue, database);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_start_stop_engine() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let database = Arc::new(Database::new(db_path).expect("failed to create database"));
        let queue = Arc::new(JobQueue::new());

        let engine = ExecutionEngine::new(2, queue, database).expect("failed to create");

        assert!(!engine.is_running());

        engine.start().await.expect("await should be valid");
        assert!(engine.is_running());

        engine.stop().await.expect("await should be valid");
        assert!(!engine.is_running());
    }

    #[tokio::test]
    async fn test_worker_count() {
        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let db_path = temp_file
            .path()
            .to_str()
            .expect("path should be valid UTF-8");

        let database = Arc::new(Database::new(db_path).expect("failed to create database"));
        let queue = Arc::new(JobQueue::new());

        let engine = ExecutionEngine::new(4, queue, database).expect("failed to create");
        engine.start().await.expect("await should be valid");

        assert_eq!(engine.worker_count(), 4);

        engine.stop().await.expect("await should be valid");
    }
}
