//! Distributed encoding coordinator for `OxiMedia`.
//!
//! This crate provides a distributed video encoding system with:
//! - Central coordinator for job management
//! - Worker nodes for distributed encoding
//! - Multiple splitting strategies (segment, tile, GOP-based)
//! - Load balancing and fault tolerance
//! - gRPC-based communication

pub mod audit_log;
pub mod backpressure;
pub mod checkpointing;
pub mod circuit_breaker;
pub mod cluster;
pub mod compaction;
pub mod connection_pool;
pub mod consensus;
pub mod coordinator;
pub mod discovery;
pub mod distributed_enhancements;
pub mod fault_tolerance;
pub mod heartbeat;
pub mod job_dag;
pub mod job_preemption;
pub mod job_tracker;
pub mod leader_election;
pub mod lease;
pub mod load_balancer;
pub mod membership;
pub mod message_bus;
pub mod message_queue;
pub mod metrics_aggregator;
pub mod node_health;
pub mod node_registry;
pub mod node_topology;
pub mod partition;
pub mod pb;
pub mod raft_primitives;
pub mod replication;
pub mod resource_quota;
pub mod scheduler;
pub mod segment;
pub mod segment_merge;
pub mod shard;
pub mod shard_map;
pub mod snapshot_store;
pub mod task_distribution;
pub mod task_priority_queue;
pub mod task_queue;
pub mod task_retry;
pub mod twopc;
pub mod weighted_round_robin;
pub mod work_stealing;
pub mod worker;
pub mod worker_draining;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Result type for distributed operations
pub type Result<T> = std::result::Result<T, DistributedError>;

/// Errors that can occur in distributed encoding
#[derive(Debug, Error)]
pub enum DistributedError {
    #[error("Worker error: {0}")]
    Worker(String),

    #[error("Coordinator error: {0}")]
    Coordinator(String),

    #[error("Job error: {0}")]
    Job(String),

    #[error("Network error: {0}")]
    Network(#[from] tonic::transport::Error),

    #[error("gRPC status error: {0}")]
    Status(#[from] tonic::Status),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Discovery error: {0}")]
    Discovery(String),

    #[error("Scheduling error: {0}")]
    Scheduling(String),

    #[error("Segmentation error: {0}")]
    Segmentation(String),

    #[error("Timeout error")]
    Timeout,

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Error: {0}")]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl From<Box<dyn std::error::Error + Send + Sync>> for DistributedError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        DistributedError::Other(err)
    }
}

/// Configuration for the distributed encoder
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistributedConfig {
    /// Coordinator address
    pub coordinator_addr: String,

    /// Maximum number of retry attempts
    pub max_retries: u32,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Job timeout
    pub job_timeout: Duration,

    /// Maximum concurrent jobs per worker
    pub max_concurrent_jobs: u32,

    /// Enable fault tolerance
    pub fault_tolerance: bool,

    /// Worker discovery method
    pub discovery_method: DiscoveryMethod,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            coordinator_addr: "127.0.0.1:50051".to_string(),
            max_retries: 3,
            heartbeat_interval: Duration::from_secs(30),
            job_timeout: Duration::from_secs(3600),
            max_concurrent_jobs: 4,
            fault_tolerance: true,
            discovery_method: DiscoveryMethod::Static,
        }
    }
}

/// Worker discovery methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub enum DiscoveryMethod {
    /// Static configuration
    Static,
    /// Multicast DNS
    #[allow(clippy::upper_case_acronyms)]
    MDNS,
    /// etcd-based discovery
    Etcd,
    /// Consul-based discovery
    Consul,
}

/// Job splitting strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SplitStrategy {
    /// Split by time segments
    SegmentBased,
    /// Split by spatial tiles
    TileBased,
    /// Split by GOP (Group of Pictures)
    GopBased,
}

impl From<SplitStrategy> for i32 {
    fn from(strategy: SplitStrategy) -> Self {
        match strategy {
            SplitStrategy::SegmentBased => 0,
            SplitStrategy::TileBased => 1,
            SplitStrategy::GopBased => 2,
        }
    }
}

/// Job priority levels
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum JobPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl From<JobPriority> for u32 {
    fn from(priority: JobPriority) -> Self {
        priority as u32
    }
}

/// Represents a distributed encoding job
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DistributedJob {
    /// Unique job identifier
    pub id: Uuid,

    /// Task identifier (multiple jobs can belong to same task)
    pub task_id: Uuid,

    /// Source video URL
    pub source_url: String,

    /// Target codec
    pub codec: String,

    /// Splitting strategy
    pub strategy: SplitStrategy,

    /// Job priority
    pub priority: JobPriority,

    /// Encoding parameters
    pub params: EncodingParams,

    /// Output destination
    pub output_url: String,

    /// Deadline timestamp (Unix epoch)
    pub deadline: Option<i64>,
}

/// Encoding parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EncodingParams {
    pub bitrate: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub preset: Option<String>,
    pub profile: Option<String>,
    pub crf: Option<u32>,
    pub extra_params: std::collections::HashMap<String, String>,
}

impl Default for EncodingParams {
    fn default() -> Self {
        Self {
            bitrate: None,
            width: None,
            height: None,
            preset: Some("medium".to_string()),
            profile: None,
            crf: Some(23),
            extra_params: std::collections::HashMap::new(),
        }
    }
}

/// Internal record for a submitted job, tracking its lifecycle.
#[derive(Debug, Clone)]
struct JobRecord {
    /// The original job definition.
    #[allow(dead_code)]
    job: DistributedJob,
    /// Current status.
    status: JobStatus,
    /// When the job was submitted.
    submitted_at: Instant,
    /// Number of retry attempts so far.
    retries: u32,
}

/// Main distributed encoder interface.
///
/// Maintains an in-process job store so that `submit_job`, `job_status`, and
/// `cancel_job` operate on real state. In a production deployment the store
/// would be backed by the gRPC coordinator; this implementation provides a
/// fully functional local fallback that exercises the complete lifecycle.
pub struct DistributedEncoder {
    config: DistributedConfig,
    /// Job store keyed by job UUID.
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
}

impl DistributedEncoder {
    /// Create a new distributed encoder with the given configuration
    #[must_use]
    pub fn new(config: DistributedConfig) -> Self {
        Self {
            config,
            jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a new distributed encoder with default configuration
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(DistributedConfig::default())
    }

    /// Get the current configuration
    #[must_use]
    pub fn config(&self) -> &DistributedConfig {
        &self.config
    }

    /// Return the number of currently tracked jobs.
    pub async fn job_count(&self) -> usize {
        self.jobs.read().await.len()
    }

    /// Return the number of active (non-terminal) jobs.
    pub async fn active_job_count(&self) -> usize {
        self.jobs
            .read()
            .await
            .values()
            .filter(|r| {
                matches!(
                    r.status,
                    JobStatus::Pending | JobStatus::Assigned | JobStatus::InProgress
                )
            })
            .count()
    }

    /// Submit a job for distributed encoding.
    ///
    /// Validates the job, checks concurrency limits, registers it in the
    /// internal store, and returns the job ID on success.
    ///
    /// # Arguments
    ///
    /// * `job` - The encoding job to submit
    ///
    /// # Returns
    ///
    /// Returns the job ID on success
    ///
    /// # Errors
    ///
    /// Returns `DistributedError::InvalidConfig` if the job definition is
    /// invalid, or `DistributedError::ResourceExhausted` if the maximum
    /// concurrent job limit has been reached.
    pub async fn submit_job(&self, job: DistributedJob) -> Result<Uuid> {
        // --- validation ---
        if job.source_url.is_empty() {
            return Err(DistributedError::InvalidConfig(
                "source_url must not be empty".to_string(),
            ));
        }
        if job.output_url.is_empty() {
            return Err(DistributedError::InvalidConfig(
                "output_url must not be empty".to_string(),
            ));
        }
        if job.codec.is_empty() {
            return Err(DistributedError::InvalidConfig(
                "codec must not be empty".to_string(),
            ));
        }

        // Check deadline is not already in the past
        if let Some(deadline) = job.deadline {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| DistributedError::Job(format!("System time error: {e}")))?;
            if deadline < now.as_secs() as i64 {
                return Err(DistributedError::Job(
                    "Job deadline is already in the past".to_string(),
                ));
            }
        }

        let mut jobs = self.jobs.write().await;

        // Check for duplicate job ID
        if jobs.contains_key(&job.id) {
            return Err(DistributedError::Job(format!(
                "Job with ID {} already exists",
                job.id
            )));
        }

        // Enforce concurrency limit
        let active_count = jobs
            .values()
            .filter(|r| {
                matches!(
                    r.status,
                    JobStatus::Pending | JobStatus::Assigned | JobStatus::InProgress
                )
            })
            .count();

        if active_count >= self.config.max_concurrent_jobs as usize {
            return Err(DistributedError::ResourceExhausted(format!(
                "Maximum concurrent jobs ({}) reached",
                self.config.max_concurrent_jobs
            )));
        }

        let job_id = job.id;

        tracing::info!(
            "Submitting job {} (codec={}, strategy={:?}, priority={:?}) to coordinator at {}",
            job_id,
            job.codec,
            job.strategy,
            job.priority,
            self.config.coordinator_addr
        );

        jobs.insert(
            job_id,
            JobRecord {
                job,
                status: JobStatus::Pending,
                submitted_at: Instant::now(),
                retries: 0,
            },
        );

        Ok(job_id)
    }

    /// Query the status of a previously submitted job.
    ///
    /// In addition to returning the stored status, this method performs
    /// timeout checking: if a job has been active longer than the configured
    /// `job_timeout` it is automatically marked as `Failed`.
    ///
    /// # Errors
    ///
    /// Returns `DistributedError::Job` if the job ID is not found.
    pub async fn job_status(&self, job_id: Uuid) -> Result<JobStatus> {
        tracing::debug!("Querying status for job {}", job_id);

        let mut jobs = self.jobs.write().await;
        let record = jobs
            .get_mut(&job_id)
            .ok_or_else(|| DistributedError::Job(format!("Job {job_id} not found")))?;

        // Check for timeout on active jobs
        if matches!(
            record.status,
            JobStatus::Pending | JobStatus::Assigned | JobStatus::InProgress
        ) && record.submitted_at.elapsed() > self.config.job_timeout
        {
            tracing::warn!(
                "Job {} has timed out after {:?}",
                job_id,
                self.config.job_timeout
            );
            record.status = JobStatus::Failed;
        }

        Ok(record.status)
    }

    /// Cancel a previously submitted job.
    ///
    /// Only jobs that are not yet in a terminal state (`Completed`, `Failed`,
    /// `Cancelled`) can be cancelled.
    ///
    /// # Errors
    ///
    /// Returns `DistributedError::Job` if the job ID is not found or the job
    /// is already in a terminal state.
    pub async fn cancel_job(&self, job_id: Uuid) -> Result<()> {
        tracing::info!("Cancelling job {}", job_id);

        let mut jobs = self.jobs.write().await;
        let record = jobs
            .get_mut(&job_id)
            .ok_or_else(|| DistributedError::Job(format!("Job {job_id} not found")))?;

        match record.status {
            JobStatus::Completed => {
                return Err(DistributedError::Job(format!(
                    "Job {job_id} is already completed and cannot be cancelled"
                )));
            }
            JobStatus::Failed => {
                return Err(DistributedError::Job(format!(
                    "Job {job_id} has already failed and cannot be cancelled"
                )));
            }
            JobStatus::Cancelled => {
                return Err(DistributedError::Job(format!(
                    "Job {job_id} is already cancelled"
                )));
            }
            _ => {}
        }

        record.status = JobStatus::Cancelled;
        Ok(())
    }

    /// Advance a job to the next logical status (for internal/testing use).
    ///
    /// Transitions: Pending -> Assigned -> InProgress -> Completed
    ///
    /// # Errors
    ///
    /// Returns error if the job is not found or is in a terminal state.
    pub async fn advance_job(&self, job_id: Uuid) -> Result<JobStatus> {
        let mut jobs = self.jobs.write().await;
        let record = jobs
            .get_mut(&job_id)
            .ok_or_else(|| DistributedError::Job(format!("Job {job_id} not found")))?;

        record.status = match record.status {
            JobStatus::Pending => JobStatus::Assigned,
            JobStatus::Assigned => JobStatus::InProgress,
            JobStatus::InProgress => JobStatus::Completed,
            other => {
                return Err(DistributedError::Job(format!(
                    "Cannot advance job in terminal state: {other:?}"
                )));
            }
        };

        Ok(record.status)
    }

    /// Mark a job as failed (for internal/testing use).
    ///
    /// # Errors
    ///
    /// Returns error if the job is not found or already in a terminal state.
    pub async fn fail_job(&self, job_id: Uuid) -> Result<()> {
        let mut jobs = self.jobs.write().await;
        let record = jobs
            .get_mut(&job_id)
            .ok_or_else(|| DistributedError::Job(format!("Job {job_id} not found")))?;

        if matches!(record.status, JobStatus::Completed | JobStatus::Cancelled) {
            return Err(DistributedError::Job(format!(
                "Cannot fail job {job_id} in terminal state: {:?}",
                record.status
            )));
        }

        // Check if we should retry
        if self.config.fault_tolerance && record.retries < self.config.max_retries {
            record.retries += 1;
            record.status = JobStatus::Pending;
            tracing::info!(
                "Retrying job {} (attempt {}/{})",
                job_id,
                record.retries,
                self.config.max_retries
            );
        } else {
            record.status = JobStatus::Failed;
        }

        Ok(())
    }

    /// Get the retry count for a job.
    ///
    /// # Errors
    ///
    /// Returns error if the job is not found.
    pub async fn job_retries(&self, job_id: Uuid) -> Result<u32> {
        let jobs = self.jobs.read().await;
        let record = jobs
            .get(&job_id)
            .ok_or_else(|| DistributedError::Job(format!("Job {job_id} not found")))?;
        Ok(record.retries)
    }

    /// List all job IDs with their current statuses.
    pub async fn list_jobs(&self) -> Vec<(Uuid, JobStatus)> {
        self.jobs
            .read()
            .await
            .iter()
            .map(|(id, record)| (*id, record.status))
            .collect()
    }
}

/// Job execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JobStatus {
    Pending,
    Assigned,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_job() -> DistributedJob {
        DistributedJob {
            id: Uuid::new_v4(),
            task_id: Uuid::new_v4(),
            source_url: "s3://bucket/input.mp4".to_string(),
            codec: "av1".to_string(),
            strategy: SplitStrategy::SegmentBased,
            priority: JobPriority::Normal,
            params: EncodingParams::default(),
            output_url: "s3://bucket/output.mp4".to_string(),
            deadline: None,
        }
    }

    #[test]
    fn test_default_config() {
        let config = DistributedConfig::default();
        assert_eq!(config.coordinator_addr, "127.0.0.1:50051");
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_concurrent_jobs, 4);
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = DistributedEncoder::with_defaults();
        assert_eq!(encoder.config().coordinator_addr, "127.0.0.1:50051");
    }

    #[test]
    fn test_job_priority_ordering() {
        assert!(JobPriority::Critical > JobPriority::High);
        assert!(JobPriority::High > JobPriority::Normal);
        assert!(JobPriority::Normal > JobPriority::Low);
    }

    #[tokio::test]
    async fn test_submit_and_query_job() {
        let encoder = DistributedEncoder::with_defaults();
        let job = make_job();
        let job_id = job.id;

        let returned_id = encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");
        assert_eq!(returned_id, job_id);

        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Pending);
    }

    #[tokio::test]
    async fn test_submit_rejects_empty_source_url() {
        let encoder = DistributedEncoder::with_defaults();
        let mut job = make_job();
        job.source_url = String::new();

        let result = encoder.submit_job(job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_rejects_empty_codec() {
        let encoder = DistributedEncoder::with_defaults();
        let mut job = make_job();
        job.codec = String::new();

        let result = encoder.submit_job(job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_rejects_empty_output_url() {
        let encoder = DistributedEncoder::with_defaults();
        let mut job = make_job();
        job.output_url = String::new();

        let result = encoder.submit_job(job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_submit_rejects_duplicate_id() {
        let encoder = DistributedEncoder::with_defaults();
        let job = make_job();
        let dup = job.clone();

        encoder
            .submit_job(job)
            .await
            .expect("first submit should succeed");
        let result = encoder.submit_job(dup).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cancel_job() {
        let encoder = DistributedEncoder::with_defaults();
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");
        encoder
            .cancel_job(job_id)
            .await
            .expect("cancel should succeed");

        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Cancelled);
    }

    #[tokio::test]
    async fn test_cancel_nonexistent_job_fails() {
        let encoder = DistributedEncoder::with_defaults();
        let result = encoder.cancel_job(Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_cancel_completed_job_fails() {
        let encoder = DistributedEncoder::with_defaults();
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");
        // Advance to Completed
        encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed"); // Assigned
        encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed"); // InProgress
        encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed"); // Completed

        let result = encoder.cancel_job(job_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_advance_job_lifecycle() {
        let encoder = DistributedEncoder::with_defaults();
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");

        let s1 = encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed");
        assert_eq!(s1, JobStatus::Assigned);

        let s2 = encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed");
        assert_eq!(s2, JobStatus::InProgress);

        let s3 = encoder
            .advance_job(job_id)
            .await
            .expect("advance should succeed");
        assert_eq!(s3, JobStatus::Completed);

        // Cannot advance past Completed
        let result = encoder.advance_job(job_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fail_job_with_retry() {
        let config = DistributedConfig {
            max_retries: 2,
            fault_tolerance: true,
            ..DistributedConfig::default()
        };
        let encoder = DistributedEncoder::new(config);
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");

        // First failure: should retry (back to Pending)
        encoder.fail_job(job_id).await.expect("fail should succeed");
        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Pending);
        let retries = encoder
            .job_retries(job_id)
            .await
            .expect("retries should succeed");
        assert_eq!(retries, 1);

        // Second failure: should retry again
        encoder.fail_job(job_id).await.expect("fail should succeed");
        let retries = encoder
            .job_retries(job_id)
            .await
            .expect("retries should succeed");
        assert_eq!(retries, 2);

        // Third failure: max retries exhausted, should be Failed
        encoder.fail_job(job_id).await.expect("fail should succeed");
        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Failed);
    }

    #[tokio::test]
    async fn test_fail_without_fault_tolerance() {
        let config = DistributedConfig {
            fault_tolerance: false,
            ..DistributedConfig::default()
        };
        let encoder = DistributedEncoder::new(config);
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");
        encoder.fail_job(job_id).await.expect("fail should succeed");

        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Failed);
    }

    #[tokio::test]
    async fn test_concurrency_limit() {
        let config = DistributedConfig {
            max_concurrent_jobs: 2,
            ..DistributedConfig::default()
        };
        let encoder = DistributedEncoder::new(config);

        encoder
            .submit_job(make_job())
            .await
            .expect("first should succeed");
        encoder
            .submit_job(make_job())
            .await
            .expect("second should succeed");

        // Third should be rejected
        let result = encoder.submit_job(make_job()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_concurrency_freed_after_cancel() {
        let config = DistributedConfig {
            max_concurrent_jobs: 1,
            ..DistributedConfig::default()
        };
        let encoder = DistributedEncoder::new(config);

        let job = make_job();
        let job_id = job.id;
        encoder.submit_job(job).await.expect("first should succeed");

        // Cannot submit another
        assert!(encoder.submit_job(make_job()).await.is_err());

        // Cancel the first
        encoder
            .cancel_job(job_id)
            .await
            .expect("cancel should succeed");

        // Now we can submit
        encoder
            .submit_job(make_job())
            .await
            .expect("after cancel should succeed");
    }

    #[tokio::test]
    async fn test_list_jobs() {
        let encoder = DistributedEncoder::with_defaults();
        let j1 = make_job();
        let j2 = make_job();
        let id1 = j1.id;
        let id2 = j2.id;

        encoder.submit_job(j1).await.expect("submit should succeed");
        encoder.submit_job(j2).await.expect("submit should succeed");

        let jobs = encoder.list_jobs().await;
        assert_eq!(jobs.len(), 2);

        let ids: Vec<Uuid> = jobs.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[tokio::test]
    async fn test_job_count() {
        let encoder = DistributedEncoder::with_defaults();
        assert_eq!(encoder.job_count().await, 0);

        encoder
            .submit_job(make_job())
            .await
            .expect("submit should succeed");
        assert_eq!(encoder.job_count().await, 1);
        assert_eq!(encoder.active_job_count().await, 1);
    }

    #[tokio::test]
    async fn test_status_nonexistent_job_fails() {
        let encoder = DistributedEncoder::with_defaults();
        let result = encoder.job_status(Uuid::new_v4()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_job_timeout_detection() {
        let config = DistributedConfig {
            job_timeout: Duration::from_millis(1),
            ..DistributedConfig::default()
        };
        let encoder = DistributedEncoder::new(config);
        let job = make_job();
        let job_id = job.id;

        encoder
            .submit_job(job)
            .await
            .expect("submit should succeed");

        // Wait briefly for timeout
        tokio::time::sleep(Duration::from_millis(10)).await;

        let status = encoder
            .job_status(job_id)
            .await
            .expect("status should succeed");
        assert_eq!(status, JobStatus::Failed);
    }

    #[tokio::test]
    async fn test_submit_past_deadline_rejected() {
        let encoder = DistributedEncoder::with_defaults();
        let mut job = make_job();
        job.deadline = Some(0); // epoch = far in the past

        let result = encoder.submit_job(job).await;
        assert!(result.is_err());
    }
}
