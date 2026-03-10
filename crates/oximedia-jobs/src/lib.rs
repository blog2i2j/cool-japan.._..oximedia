// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! `OxiMedia` Jobs - Job queue and worker management for video transcoding
//!
//! This crate provides a comprehensive job queue system with the following features:
//!
//! # Features
//!
//! - **Priority Queue**: Support for high, normal, and low priority jobs
//! - **Job Scheduling**: Schedule jobs for future execution with flexible scheduling options
//! - **Dependencies**: Define job dependencies and execution order
//! - **Retry Logic**: Automatic retry with exponential backoff
//! - **Job Cancellation**: Cancel running or pending jobs
//! - **Persistent Queue**: SQLite-based persistence for job state
//! - **Worker Pool**: Configurable worker pool with load balancing
//! - **Health Monitoring**: Worker health checks and monitoring
//! - **Auto-scaling**: Automatic worker pool scaling based on load
//! - **Job Pipelines**: Chain jobs together in pipelines
//! - **Conditional Execution**: Execute jobs based on conditions
//! - **Resource Quotas**: Limit resource usage per job
//! - **Deadline Scheduling**: Set deadlines for job execution
//! - **Metrics**: Comprehensive metrics and performance tracking
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_jobs::{
//!     Job, JobBuilder, JobPayload, Priority, TranscodeParams,
//!     JobQueue, QueueConfig, WorkerConfig, MetricsCollector,
//!     DefaultExecutor,
//! };
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create metrics collector
//!     let metrics = Arc::new(MetricsCollector::new());
//!
//!     // Create worker configuration
//!     let worker_config = WorkerConfig::default();
//!
//!     // Create job executor
//!     let executor = Arc::new(DefaultExecutor);
//!
//!     // Create queue configuration
//!     let queue_config = QueueConfig::default();
//!
//!     // Create job queue
//!     let queue = JobQueue::new(queue_config, executor, metrics, worker_config)
//!         .expect("Failed to create queue");
//!
//!     // Start the queue
//!     queue.start().await;
//!
//!     // Create a transcoding job
//!     let params = TranscodeParams {
//!         input: "input.mp4".to_string(),
//!         output: "output.mp4".to_string(),
//!         video_codec: "h264".to_string(),
//!         audio_codec: "aac".to_string(),
//!         video_bitrate: 5_000_000,
//!         audio_bitrate: 192_000,
//!         resolution: Some((1920, 1080)),
//!         framerate: Some(30.0),
//!         preset: "medium".to_string(),
//!         hw_accel: None,
//!     };
//!
//!     let job = Job::new(
//!         "Transcode video".to_string(),
//!         Priority::Normal,
//!         JobPayload::Transcode(params),
//!     );
//!
//!     // Submit the job
//!     let job_id = queue.submit(job).await.expect("Failed to submit job");
//!     println!("Submitted job: {}", job_id);
//!
//!     // Wait for a bit
//!     tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
//!
//!     // Shutdown the queue
//!     queue.shutdown().await;
//! }
//! ```

pub mod batch;
pub mod dependency;
pub mod dependency_graph;
pub mod event_log;
pub mod job;
/// Worker affinity and anti-affinity rules for job placement.
pub mod job_affinity;
pub mod job_graph_viz;
pub mod job_history;
/// Per-job metrics collection — outcomes, efficiency, and aggregate statistics.
pub mod job_metrics;
/// Dynamic priority boosting based on wait time, deadlines, and starvation prevention.
pub mod job_priority_boost;
pub mod job_tags;
/// Reusable, parameterised job templates with placeholder substitution and a template registry.
pub mod job_template;
pub mod metrics;
pub mod persistence;
pub mod priority;
pub mod queue;
pub mod quota;
/// Rate limiting — token bucket and fixed-window algorithms for controlling job throughput.
pub mod rate_limiter;
pub mod registry;
pub mod resource_claim;
pub mod resource_estimate;
pub mod resource_limits;
pub mod retry;
/// Retry policy configuration with backoff strategies, circuit breakers, and per-error-class behavior.
pub mod retry_policy;
pub mod scheduler;
/// Scheduling rules for time-window constraints, resource-aware dispatch, and affinity/anti-affinity.
pub mod scheduling_rule;
pub mod telemetry;
/// Real-time throughput tracking and reporting for the job queue.
pub mod throughput_tracker;
pub mod worker;
/// Worker pool management — worker state, utilization tracking, and job assignment.
pub mod worker_pool;

pub use job::{
    AnalysisParams, AnalysisType, BatchParams, Condition, Job, JobBuilder, JobPayload, JobStatus,
    Priority, ResourceQuota, RetryPolicy, SpriteSheetParams, ThumbnailParams, TranscodeParams,
};
pub use metrics::{
    JobMetrics, MetricsCollector, PerformanceReport, QueueStats, WorkerMetrics, WorkerStatus,
};
pub use persistence::{JobPersistence, PersistenceError};
pub use queue::{JobQueue, QueueConfig, QueueError};
pub use registry::{JobId, JobRegistry, PriorityQueue};
pub use scheduler::{
    JobScheduler, Pipeline, PipelineBuilder, PipelineStage, Schedule, SchedulerError,
};
use std::sync::Arc;
pub use worker::{DefaultExecutor, JobExecutor, WorkerConfig, WorkerError, WorkerPool};

/// Job queue service that combines all components
pub struct JobQueueService {
    /// Job queue
    pub queue: Arc<JobQueue>,
    /// Job scheduler
    pub scheduler: JobScheduler,
    /// Metrics collector
    pub metrics: Arc<MetricsCollector>,
}

impl JobQueueService {
    /// Create a new job queue service
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn new(
        queue_config: QueueConfig,
        worker_config: WorkerConfig,
        executor: Arc<dyn JobExecutor>,
    ) -> Result<Self, QueueError> {
        let metrics = Arc::new(MetricsCollector::new());
        let queue = Arc::new(JobQueue::new(
            queue_config,
            executor,
            metrics.clone(),
            worker_config,
        )?);

        let scheduler = JobScheduler::new().with_queue((*queue).clone());

        Ok(Self {
            queue,
            scheduler,
            metrics,
        })
    }

    /// Create a service with in-memory persistence (for testing)
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    pub fn in_memory(
        worker_config: WorkerConfig,
        executor: Arc<dyn JobExecutor>,
    ) -> Result<Self, QueueError> {
        let metrics = Arc::new(MetricsCollector::new());
        let queue = Arc::new(JobQueue::in_memory(
            executor,
            metrics.clone(),
            worker_config,
        )?);

        let scheduler = JobScheduler::new().with_queue((*queue).clone());

        Ok(Self {
            queue,
            scheduler,
            metrics,
        })
    }

    /// Start the service
    pub async fn start(&self) {
        self.queue.start().await;
    }

    /// Shutdown the service
    pub async fn shutdown(&self) {
        self.queue.shutdown().await;
    }

    /// Submit a job
    ///
    /// # Errors
    ///
    /// Returns an error if job submission fails
    pub async fn submit_job(&self, job: Job) -> Result<uuid::Uuid, QueueError> {
        self.queue.submit(job).await
    }

    /// Schedule a job
    ///
    /// # Errors
    ///
    /// Returns an error if job scheduling fails
    pub async fn schedule_job(
        &self,
        job: Job,
        schedule: Schedule,
    ) -> Result<uuid::Uuid, SchedulerError> {
        self.scheduler.schedule_job(job, schedule).await
    }

    /// Schedule a pipeline
    ///
    /// # Errors
    ///
    /// Returns an error if pipeline scheduling fails
    pub async fn schedule_pipeline(
        &self,
        pipeline: Pipeline,
        schedule: Schedule,
    ) -> Result<Vec<uuid::Uuid>, SchedulerError> {
        self.scheduler.schedule_pipeline(pipeline, schedule).await
    }

    /// Cancel a job
    ///
    /// # Errors
    ///
    /// Returns an error if job cancellation fails
    pub async fn cancel_job(&self, job_id: uuid::Uuid) -> Result<(), QueueError> {
        self.queue.cancel(job_id).await
    }

    /// Get job by ID
    ///
    /// # Errors
    ///
    /// Returns an error if job is not found
    pub async fn get_job(&self, job_id: uuid::Uuid) -> Result<Job, QueueError> {
        self.queue.get_job(job_id).await
    }

    /// Get jobs by status
    ///
    /// # Errors
    ///
    /// Returns an error if query fails
    pub async fn get_jobs_by_status(&self, status: JobStatus) -> Result<Vec<Job>, QueueError> {
        self.queue.get_jobs_by_status(status).await
    }

    /// Get queue statistics
    pub async fn get_stats(&self) -> QueueStats {
        self.queue.get_stats().await
    }

    /// Get performance report
    pub async fn get_performance_report(&self, slow_threshold_secs: f64) -> PerformanceReport {
        PerformanceReport::generate(&self.metrics, slow_threshold_secs).await
    }

    /// Update job progress
    ///
    /// # Errors
    ///
    /// Returns an error if update fails
    pub async fn update_progress(
        &self,
        job_id: uuid::Uuid,
        progress: u8,
    ) -> Result<(), QueueError> {
        self.queue.update_progress(job_id, progress).await
    }

    /// Health check
    pub async fn health_check(&self) -> Vec<String> {
        self.queue.worker_pool.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_job_creation() {
        let params = TranscodeParams {
            input: "input.mp4".to_string(),
            output: "output.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 192_000,
            resolution: Some((1920, 1080)),
            framerate: Some(30.0),
            preset: "medium".to_string(),
            hw_accel: None,
        };

        let job = Job::new(
            "Test job".to_string(),
            Priority::Normal,
            JobPayload::Transcode(params),
        );

        assert_eq!(job.name, "Test job");
        assert_eq!(job.priority, Priority::Normal);
        assert_eq!(job.status, JobStatus::Pending);
    }

    #[tokio::test]
    async fn test_job_builder() {
        let params = TranscodeParams {
            input: "input.mp4".to_string(),
            output: "output.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 192_000,
            resolution: Some((1920, 1080)),
            framerate: Some(30.0),
            preset: "medium".to_string(),
            hw_accel: None,
        };

        let job = JobBuilder::new(
            "Test job".to_string(),
            Priority::High,
            JobPayload::Transcode(params),
        )
        .tag("video".to_string())
        .tag("transcode".to_string())
        .build();

        assert_eq!(job.priority, Priority::High);
        assert_eq!(job.tags.len(), 2);
        assert!(job.tags.contains(&"video".to_string()));
    }

    #[tokio::test]
    async fn test_metrics_collector() {
        let metrics = MetricsCollector::new();

        metrics.register_worker("worker-1".to_string()).await;

        let worker_metrics = metrics.get_worker_metrics().await;
        assert!(worker_metrics.contains_key("worker-1"));
    }

    #[tokio::test]
    async fn test_pipeline() {
        let params1 = TranscodeParams {
            input: "input.mp4".to_string(),
            output: "output.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 192_000,
            resolution: Some((1920, 1080)),
            framerate: Some(30.0),
            preset: "medium".to_string(),
            hw_accel: None,
        };

        let params2 = ThumbnailParams {
            input: "output.mp4".to_string(),
            output_dir: "thumbnails".to_string(),
            count: 10,
            width: 320,
            height: 180,
            quality: 80,
        };

        let stage1 = PipelineStage::new("transcode".to_string(), JobPayload::Transcode(params1));
        let stage2 = PipelineStage::new("thumbnail".to_string(), JobPayload::Thumbnail(params2));

        let pipeline = PipelineBuilder::new("video-processing".to_string())
            .stage(stage1)
            .stage(stage2)
            .tag("video".to_string())
            .build();

        let jobs = pipeline.build().expect("jobs should be valid");
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0].next_jobs[0], jobs[1].id);
        assert!(jobs[1].dependencies.contains(&jobs[0].id));
    }

    #[tokio::test]
    async fn test_retry_policy() {
        let policy = RetryPolicy::default();
        let backoff1 = policy.backoff_duration(0);
        let backoff2 = policy.backoff_duration(1);
        let backoff3 = policy.backoff_duration(2);

        assert!(backoff1.num_seconds() == 60);
        assert!(backoff2.num_seconds() == 120);
        assert!(backoff3.num_seconds() == 240);
    }

    #[tokio::test]
    async fn test_job_queue_service() {
        let worker_config = WorkerConfig::default();
        let executor = Arc::new(DefaultExecutor);

        let service =
            JobQueueService::in_memory(worker_config, executor).expect("Failed to create service");

        service.start().await;

        let params = TranscodeParams {
            input: "input.mp4".to_string(),
            output: "output.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 5_000_000,
            audio_bitrate: 192_000,
            resolution: Some((1920, 1080)),
            framerate: Some(30.0),
            preset: "medium".to_string(),
            hw_accel: None,
        };

        let job = Job::new(
            "Test job".to_string(),
            Priority::Normal,
            JobPayload::Transcode(params),
        );

        let job_id = service.submit_job(job).await.expect("Failed to submit job");

        assert!(service.get_job(job_id).await.is_ok());

        service.shutdown().await;
    }
}
