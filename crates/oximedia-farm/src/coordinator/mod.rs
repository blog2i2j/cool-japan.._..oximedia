//! Coordinator implementation - the heart of the encoding farm

mod job_queue;
pub mod worker_registry;

use crate::communication::FarmCoordinatorService;
use crate::persistence::Database;
use crate::scheduler::{Scheduler, SchedulingStrategy};
use crate::{CoordinatorConfig, FarmError, JobId, Result};
use std::sync::Arc;
use tokio::sync::broadcast;
use tonic::transport::Server;
use tracing::{error, info};

pub use job_queue::{Job, JobProgress, JobQueue, Task};
pub use worker_registry::{GpuInfo, WorkerCapabilities, WorkerRegistration, WorkerRegistry};

/// Main coordinator that manages the encoding farm
pub struct Coordinator {
    config: CoordinatorConfig,
    database: Arc<Database>,
    job_queue: Arc<JobQueue>,
    worker_registry: Arc<WorkerRegistry>,
    scheduler: Arc<Scheduler>,
    shutdown_tx: broadcast::Sender<()>,
}

impl Coordinator {
    /// Create a new coordinator
    pub async fn new(config: CoordinatorConfig) -> Result<Self> {
        info!("Initializing coordinator with config: {:?}", config);

        // Initialize database
        let database = Arc::new(Database::new(&config.database_path)?);
        info!("Database initialized at {}", config.database_path);

        // Create job queue
        let job_queue = Arc::new(JobQueue::new(
            database.clone(),
            config.max_concurrent_jobs,
            config.max_tasks_per_job,
        ));

        // Create worker registry
        let worker_registry = Arc::new(WorkerRegistry::new(config.heartbeat_timeout));

        // Create scheduler
        let scheduler = Arc::new(Scheduler::new(SchedulingStrategy::LeastLoaded));

        // Create shutdown channel
        let (shutdown_tx, _) = broadcast::channel(1);

        Ok(Self {
            config,
            database,
            job_queue,
            worker_registry,
            scheduler,
            shutdown_tx,
        })
    }

    /// Start the coordinator server
    pub async fn start(self: Arc<Self>) -> Result<()> {
        info!("Starting coordinator on {}", self.config.bind_address);

        // Start background tasks
        self.clone().start_background_tasks();

        // Parse bind address
        let addr = self
            .config
            .bind_address
            .parse()
            .map_err(|e| FarmError::InvalidConfig(format!("Invalid bind address: {e}")))?;

        // Create gRPC service
        let farm_service = FarmCoordinatorService::new(
            self.job_queue.clone(),
            self.worker_registry.clone(),
            self.scheduler.clone(),
        );

        // Build server
        let server = Server::builder()
            .add_service(
                crate::pb::farm_coordinator_server::FarmCoordinatorServer::new(farm_service),
            )
            .serve(addr);

        info!("Coordinator listening on {}", addr);

        // Run server
        server.await.map_err(FarmError::from)
    }

    /// Start background tasks
    fn start_background_tasks(self: Arc<Self>) {
        // Task scheduler
        let coord = self.clone();
        tokio::spawn(async move {
            coord.run_task_scheduler().await;
        });

        // Worker health monitor
        let coord = self.clone();
        tokio::spawn(async move {
            coord.run_worker_monitor().await;
        });

        // Job state monitor
        let coord = self.clone();
        tokio::spawn(async move {
            coord.run_job_monitor().await;
        });

        // Metrics collector
        if self.config.enable_metrics {
            let coord = self.clone();
            tokio::spawn(async move {
                coord.run_metrics_collector().await;
            });
        }
    }

    /// Run task scheduler loop
    async fn run_task_scheduler(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.schedule_pending_tasks().await {
                        error!("Error scheduling tasks: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Task scheduler shutting down");
                    break;
                }
            }
        }
    }

    /// Run worker health monitor loop
    async fn run_worker_monitor(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.check_worker_health().await {
                        error!("Error checking worker health: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Worker monitor shutting down");
                    break;
                }
            }
        }
    }

    /// Run job state monitor loop
    async fn run_job_monitor(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.update_job_states().await {
                        error!("Error updating job states: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Job monitor shutting down");
                    break;
                }
            }
        }
    }

    /// Run metrics collector loop
    async fn run_metrics_collector(&self) {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));

        loop {
            tokio::select! {
                _ = interval.tick() => {
                    if let Err(e) = self.collect_metrics().await {
                        error!("Error collecting metrics: {}", e);
                    }
                }
                _ = shutdown_rx.recv() => {
                    info!("Metrics collector shutting down");
                    break;
                }
            }
        }
    }

    /// Schedule pending tasks to available workers
    async fn schedule_pending_tasks(&self) -> Result<()> {
        let pending_tasks = self.job_queue.get_pending_tasks(100).await?;

        for task in pending_tasks {
            match self.scheduler.select_worker(&task) {
                Ok(worker_id) => {
                    if let Err(e) = self.job_queue.assign_task(task.task_id, worker_id).await {
                        error!("Failed to assign task {}: {}", task.task_id, e);
                    }
                }
                Err(e) => {
                    // No worker available, task remains pending
                    tracing::debug!("No worker available for task {}: {}", task.task_id, e);
                }
            }
        }

        Ok(())
    }

    /// Check worker health and mark stale workers as offline
    async fn check_worker_health(&self) -> Result<()> {
        let timeout_duration = chrono::Duration::from_std(self.config.heartbeat_timeout)
            .map_err(|_| FarmError::InvalidConfig("Invalid heartbeat timeout".to_string()))?;
        let stale_workers = self.scheduler.get_stale_workers(timeout_duration);

        for worker_id in stale_workers {
            info!("Worker {} is stale, marking as offline", worker_id);
            self.worker_registry.mark_offline(&worker_id).await?;

            // Reassign tasks from offline worker
            if let Err(e) = self.job_queue.reassign_worker_tasks(&worker_id).await {
                error!("Failed to reassign tasks from worker {}: {}", worker_id, e);
            }
        }

        Ok(())
    }

    /// Update job states based on task completion
    async fn update_job_states(&self) -> Result<()> {
        self.job_queue.update_job_states().await
    }

    /// Collect and record metrics
    async fn collect_metrics(&self) -> Result<()> {
        let stats = self.database.get_job_stats()?;
        let worker_count = self.scheduler.active_worker_count();

        info!(
            "Metrics: {} total jobs, {} pending, {} running, {} completed, {} workers",
            stats.total, stats.pending, stats.running, stats.completed, worker_count
        );

        Ok(())
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down coordinator");
        let _ = self.shutdown_tx.send(());
        Ok(())
    }

    /// Submit a new job
    pub async fn submit_job(&self, job: Job) -> Result<JobId> {
        self.job_queue.submit_job(job).await
    }

    /// Cancel a job
    pub async fn cancel_job(&self, job_id: JobId) -> Result<()> {
        self.job_queue.cancel_job(job_id).await
    }

    /// Get job status
    pub async fn get_job_status(&self, job_id: JobId) -> Result<Job> {
        self.job_queue.get_job(job_id).await
    }

    /// List all jobs
    pub async fn list_jobs(&self) -> Result<Vec<Job>> {
        self.job_queue.list_jobs().await
    }

    /// Get worker count
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.scheduler.active_worker_count()
    }

    /// Get database reference
    #[must_use]
    pub fn database(&self) -> &Arc<Database> {
        &self.database
    }

    /// Get job queue reference
    #[must_use]
    pub fn job_queue(&self) -> &Arc<JobQueue> {
        &self.job_queue
    }

    /// Get worker registry reference
    #[must_use]
    pub fn worker_registry(&self) -> &Arc<WorkerRegistry> {
        &self.worker_registry
    }

    /// Get scheduler reference
    #[must_use]
    pub fn scheduler(&self) -> &Arc<Scheduler> {
        &self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{JobType, Priority};

    async fn create_test_coordinator() -> Arc<Coordinator> {
        let config = CoordinatorConfig {
            database_path: ":memory:".to_string(),
            ..Default::default()
        };
        Arc::new(Coordinator::new(config).await.unwrap())
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coord = create_test_coordinator().await;
        assert_eq!(coord.worker_count(), 0);
    }

    #[tokio::test]
    async fn test_job_submission() {
        let coord = create_test_coordinator().await;

        let job = Job {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
            state: crate::JobState::Pending,
            tasks: vec![],
            deadline: None,
        };

        let job_id = coord.submit_job(job).await.unwrap();
        let retrieved = coord.get_job_status(job_id).await.unwrap();
        assert_eq!(retrieved.id, job_id);
    }

    #[tokio::test]
    async fn test_job_cancellation() {
        let coord = create_test_coordinator().await;

        let job = Job {
            id: JobId::new(),
            job_type: JobType::VideoTranscode,
            priority: Priority::Normal,
            input_path: "/input/test.mp4".to_string(),
            output_path: "/output/test.mp4".to_string(),
            parameters: std::collections::HashMap::new(),
            metadata: std::collections::HashMap::new(),
            state: crate::JobState::Pending,
            tasks: vec![],
            deadline: None,
        };

        let job_id = coord.submit_job(job).await.unwrap();
        coord.cancel_job(job_id).await.unwrap();

        let retrieved = coord.get_job_status(job_id).await.unwrap();
        assert_eq!(retrieved.state, crate::JobState::Cancelled);
    }
}
