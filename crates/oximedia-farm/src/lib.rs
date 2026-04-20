//! `OxiMedia` Farm - Distributed Encoding Farm Coordinator
//!
//! A production-grade distributed encoding farm implementation providing:
//! - Centralized job queue management with priority scheduling
//! - Worker registration, health monitoring, and capability tracking
//! - Intelligent task distribution and load balancing
//! - Comprehensive fault tolerance with automatic retry
//! - Real-time progress aggregation and monitoring
//! - Resource-aware scheduling (CPU, GPU, memory)
//! - gRPC-based communication with TLS support
//! - Persistent state management with `SQLite`
//! - Prometheus metrics and structured logging
//!
//! # Architecture
//!
//! The farm consists of three main components:
//!
//! ## Coordinator (Master)
//! - Central job queue with priority-based scheduling
//! - Worker registry with health monitoring
//! - Task distribution and load balancing
//! - Progress aggregation and reporting
//! - Failure detection and automatic retry
//! - Resource allocation and tracking
//!
//! ## Worker (Agent)
//! - Task execution (transcode, QC, analysis)
//! - Periodic heartbeat to coordinator
//! - Capability advertisement (codecs, hardware)
//! - Real-time progress reporting
//! - Structured log streaming
//! - Graceful shutdown handling
//!
//! ## Communication Layer
//! - gRPC for efficient RPC communication
//! - Protocol Buffers for message serialization
//! - TLS for secure communication (optional)
//! - Authentication and authorization support
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use oximedia_farm::{Coordinator, CoordinatorConfig};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Start a coordinator
//! let config = CoordinatorConfig::default();
//! let coordinator = Coordinator::new(config).await?;
//! Arc::new(coordinator).start().await?;
//! # Ok(())
//! # }
//! ```

#[cfg(not(target_arch = "wasm32"))]
pub mod auto_scaler;
#[cfg(not(target_arch = "wasm32"))]
pub mod capacity_planner;
#[cfg(not(target_arch = "wasm32"))]
pub mod chaos_testing;
#[cfg(not(target_arch = "wasm32"))]
pub mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
pub mod cloud_storage;
#[cfg(not(target_arch = "wasm32"))]
pub mod communication;
#[cfg(not(target_arch = "wasm32"))]
pub mod coordinator;
#[cfg(not(target_arch = "wasm32"))]
pub mod dependency;
#[cfg(not(target_arch = "wasm32"))]
pub mod farm_config;
#[cfg(not(target_arch = "wasm32"))]
pub mod fault_tolerance;
#[cfg(not(target_arch = "wasm32"))]
pub mod health;
#[cfg(not(target_arch = "wasm32"))]
pub mod heartbeat_batch;
#[cfg(not(target_arch = "wasm32"))]
pub mod job_cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod job_queue;
#[cfg(not(target_arch = "wasm32"))]
pub mod job_state_cache;
#[cfg(not(target_arch = "wasm32"))]
pub mod job_template;
#[cfg(not(target_arch = "wasm32"))]
pub mod metrics;
#[cfg(not(target_arch = "wasm32"))]
pub mod node_affinity;
#[cfg(not(target_arch = "wasm32"))]
pub mod node_monitor;
#[cfg(not(target_arch = "wasm32"))]
pub mod notification;
#[cfg(not(target_arch = "wasm32"))]
pub mod output_validator;
#[cfg(not(target_arch = "wasm32"))]
pub mod persistence;
#[cfg(not(target_arch = "wasm32"))]
pub mod priority_queue;
#[cfg(not(target_arch = "wasm32"))]
pub mod progress_stream;
#[cfg(not(target_arch = "wasm32"))]
pub mod render_stats;
#[cfg(not(target_arch = "wasm32"))]
pub mod resource_manager;
#[cfg(not(target_arch = "wasm32"))]
pub mod scheduler;
#[cfg(not(target_arch = "wasm32"))]
pub mod shutdown;
#[cfg(not(target_arch = "wasm32"))]
pub mod task_allocator;
#[cfg(not(target_arch = "wasm32"))]
pub mod task_preemption;
#[cfg(not(target_arch = "wasm32"))]
pub mod tenant;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker_health;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker_health_check;
#[cfg(not(target_arch = "wasm32"))]
pub mod worker_pool;

use std::fmt;
use std::time::Duration;
use thiserror::Error;
use uuid::Uuid;

/// Re-export protobuf generated code
#[cfg(not(target_arch = "wasm32"))]
pub mod pb {
    tonic::include_proto!("oximedia.farm");
}

/// Result type for farm operations
pub type Result<T> = std::result::Result<T, FarmError>;

/// Errors that can occur in the encoding farm
#[derive(Debug, Error)]
pub enum FarmError {
    #[error("Worker error: {0}")]
    Worker(String),

    #[error("Coordinator error: {0}")]
    Coordinator(String),

    #[error("Job error: {0}")]
    Job(String),

    #[error("Task error: {0}")]
    Task(String),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Network error: {0}")]
    Network(#[from] tonic::transport::Error),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("gRPC status error: {0}")]
    Status(#[from] tonic::Status),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(not(target_arch = "wasm32"))]
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("Scheduling error: {0}")]
    Scheduling(String),

    #[error("Timeout error: {0}")]
    Timeout(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Job identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct JobId(pub Uuid);

impl JobId {
    /// Generate a new random job ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for JobId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<JobId> for Uuid {
    fn from(id: JobId) -> Self {
        id.0
    }
}

/// Task identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TaskId(pub Uuid);

impl TaskId {
    /// Generate a new random task ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create from a UUID
    #[must_use]
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID
    #[must_use]
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for TaskId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<TaskId> for Uuid {
    fn from(id: TaskId) -> Self {
        id.0
    }
}

/// Worker identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkerId(pub String);

impl WorkerId {
    /// Create a new worker ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a random worker ID
    #[must_use]
    pub fn random() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Get the ID as a string slice
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for WorkerId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for WorkerId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Job priority levels
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
    Default,
)]
pub enum Priority {
    Low = 0,
    #[default]
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl From<Priority> for i32 {
    fn from(priority: Priority) -> Self {
        priority as i32
    }
}

impl TryFrom<i32> for Priority {
    type Error = FarmError;

    fn try_from(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::Low),
            1 => Ok(Self::Normal),
            2 => Ok(Self::High),
            3 => Ok(Self::Critical),
            _ => Err(FarmError::InvalidConfig(format!(
                "Invalid priority value: {value}"
            ))),
        }
    }
}

/// Job type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum JobType {
    VideoTranscode,
    AudioTranscode,
    ThumbnailGeneration,
    QcValidation,
    MediaAnalysis,
    ContentFingerprinting,
    MultiOutputTranscode,
}

impl fmt::Display for JobType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VideoTranscode => write!(f, "VideoTranscode"),
            Self::AudioTranscode => write!(f, "AudioTranscode"),
            Self::ThumbnailGeneration => write!(f, "ThumbnailGeneration"),
            Self::QcValidation => write!(f, "QcValidation"),
            Self::MediaAnalysis => write!(f, "MediaAnalysis"),
            Self::ContentFingerprinting => write!(f, "ContentFingerprinting"),
            Self::MultiOutputTranscode => write!(f, "MultiOutputTranscode"),
        }
    }
}

/// Job state
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JobState {
    Pending,
    Queued,
    Running,
    Completed,
    /// Job finished but output validation raised warnings.
    CompletedWithWarnings,
    Failed,
    Cancelled,
    Paused,
}

impl fmt::Display for JobState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Queued => write!(f, "Queued"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::CompletedWithWarnings => write!(f, "CompletedWithWarnings"),
            Self::Failed => write!(f, "Failed"),
            Self::Cancelled => write!(f, "Cancelled"),
            Self::Paused => write!(f, "Paused"),
        }
    }
}

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TaskState {
    Pending,
    Assigned,
    Running,
    Completed,
    Failed,
}

impl fmt::Display for TaskState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Assigned => write!(f, "Assigned"),
            Self::Running => write!(f, "Running"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WorkerState {
    Idle,
    Busy,
    Overloaded,
    Draining,
    Offline,
}

impl fmt::Display for WorkerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::Busy => write!(f, "Busy"),
            Self::Overloaded => write!(f, "Overloaded"),
            Self::Draining => write!(f, "Draining"),
            Self::Offline => write!(f, "Offline"),
        }
    }
}

/// Configuration for the coordinator
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoordinatorConfig {
    /// Address to bind the coordinator server
    pub bind_address: String,

    /// Database file path for persistent storage
    pub database_path: String,

    /// Heartbeat timeout duration
    pub heartbeat_timeout: Duration,

    /// Task timeout duration
    pub task_timeout: Duration,

    /// Maximum retry attempts for failed tasks
    pub max_retries: u32,

    /// Enable TLS
    pub enable_tls: bool,

    /// TLS certificate path
    pub tls_cert_path: Option<String>,

    /// TLS key path
    pub tls_key_path: Option<String>,

    /// Maximum concurrent jobs
    pub max_concurrent_jobs: usize,

    /// Maximum tasks per job
    pub max_tasks_per_job: usize,

    /// Enable metrics
    pub enable_metrics: bool,

    /// Metrics port
    pub metrics_port: u16,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:50051".to_string(),
            database_path: "farm.db".to_string(),
            heartbeat_timeout: Duration::from_secs(60),
            task_timeout: Duration::from_secs(3600),
            max_retries: 3,
            enable_tls: false,
            tls_cert_path: None,
            tls_key_path: None,
            max_concurrent_jobs: 1000,
            max_tasks_per_job: 100,
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}

/// Configuration for the worker
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerConfig {
    /// Worker ID (auto-generated if not provided)
    pub worker_id: Option<String>,

    /// Coordinator address
    pub coordinator_address: String,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Maximum concurrent tasks
    pub max_concurrent_tasks: u32,

    /// Enable TLS
    pub enable_tls: bool,

    /// TLS CA certificate path
    pub tls_ca_cert_path: Option<String>,

    /// Work directory for temporary files
    pub work_directory: String,

    /// Supported codecs
    pub supported_codecs: Vec<String>,

    /// Supported formats
    pub supported_formats: Vec<String>,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// Worker tags for scheduling
    pub tags: std::collections::HashMap<String, String>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let max_concurrent_tasks = num_cpus::get() as u32;
        #[cfg(target_arch = "wasm32")]
        let max_concurrent_tasks = 1u32;

        Self {
            worker_id: None,
            coordinator_address: "http://127.0.0.1:50051".to_string(),
            heartbeat_interval: Duration::from_secs(30),
            max_concurrent_tasks,
            enable_tls: false,
            tls_ca_cert_path: None,
            work_directory: std::env::temp_dir()
                .join("oximedia-farm")
                .to_string_lossy()
                .into_owned(),
            supported_codecs: vec![
                "h264".to_string(),
                "h265".to_string(),
                "vp9".to_string(),
                "av1".to_string(),
            ],
            supported_formats: vec![
                "mp4".to_string(),
                "mkv".to_string(),
                "mov".to_string(),
                "webm".to_string(),
            ],
            enable_gpu: false,
            tags: std::collections::HashMap::new(),
        }
    }
}

// Re-export main types (only available on non-wasm targets)
#[cfg(not(target_arch = "wasm32"))]
pub use coordinator::Coordinator;
#[cfg(not(target_arch = "wasm32"))]
pub use worker::Worker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id_creation() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_task_id_creation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_worker_id_creation() {
        let id = WorkerId::new("worker-1");
        assert_eq!(id.as_str(), "worker-1");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Normal);
        assert!(Priority::Normal > Priority::Low);
    }

    #[test]
    fn test_priority_conversion() {
        assert_eq!(i32::from(Priority::Low), 0);
        assert_eq!(i32::from(Priority::Normal), 1);
        assert_eq!(i32::from(Priority::High), 2);
        assert_eq!(i32::from(Priority::Critical), 3);
    }

    #[test]
    fn test_priority_from_i32() {
        assert_eq!(Priority::try_from(0).unwrap(), Priority::Low);
        assert_eq!(Priority::try_from(1).unwrap(), Priority::Normal);
        assert_eq!(Priority::try_from(2).unwrap(), Priority::High);
        assert_eq!(Priority::try_from(3).unwrap(), Priority::Critical);
        assert!(Priority::try_from(4).is_err());
    }

    #[test]
    fn test_default_coordinator_config() {
        let config = CoordinatorConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:50051");
        assert_eq!(config.max_retries, 3);
        assert!(!config.enable_tls);
    }

    #[test]
    fn test_default_worker_config() {
        let config = WorkerConfig::default();
        assert_eq!(config.coordinator_address, "http://127.0.0.1:50051");
        assert!(!config.enable_tls);
    }

    #[test]
    fn test_job_state_display() {
        assert_eq!(JobState::Pending.to_string(), "Pending");
        assert_eq!(JobState::Running.to_string(), "Running");
        assert_eq!(JobState::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_task_state_display() {
        assert_eq!(TaskState::Pending.to_string(), "Pending");
        assert_eq!(TaskState::Running.to_string(), "Running");
        assert_eq!(TaskState::Completed.to_string(), "Completed");
    }

    #[test]
    fn test_worker_state_display() {
        assert_eq!(WorkerState::Idle.to_string(), "Idle");
        assert_eq!(WorkerState::Busy.to_string(), "Busy");
        assert_eq!(WorkerState::Offline.to_string(), "Offline");
    }
}
