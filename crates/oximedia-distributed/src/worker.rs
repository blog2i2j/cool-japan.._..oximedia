//! Worker node implementation for distributed encoding.
//!
//! Workers:
//! - Register with the coordinator
//! - Send periodic heartbeats
//! - Request and execute encoding jobs
//! - Report progress and results
//! - Handle local encoding execution

use crate::pb;
use crate::pb::coordinator_service_client::CoordinatorServiceClient;
use crate::pb::{
    EncodingTask, JobFailure, JobRequest, JobResult, ProgressReport, ResultMetadata,
    WorkerHeartbeat, WorkerRegistration, WorkerUnregistration,
};
use crate::{DistributedError, Result};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tonic::transport::Channel;
use tonic::Request;
use tracing::{debug, error, info};
use uuid::Uuid;

/// Worker node
pub struct Worker {
    /// Worker configuration
    config: WorkerConfig,

    /// Worker ID (assigned by coordinator or self-generated)
    worker_id: Arc<RwLock<String>>,

    /// gRPC client
    client: Arc<RwLock<Option<CoordinatorServiceClient<Channel>>>>,

    /// Active jobs
    active_jobs: Arc<RwLock<HashMap<String, ActiveJob>>>,

    /// Worker state
    state: Arc<WorkerState>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Coordinator address
    pub coordinator_addr: String,

    /// Worker hostname
    pub hostname: String,

    /// Worker IP address
    pub ip_address: String,

    /// Worker listening port
    pub port: u32,

    /// Heartbeat interval
    pub heartbeat_interval: Duration,

    /// Job poll interval
    pub poll_interval: Duration,

    /// Maximum concurrent jobs
    pub max_concurrent_jobs: u32,

    /// Worker capabilities
    pub capabilities: WorkerCapabilities,

    /// Working directory for temporary files
    pub work_dir: PathBuf,

    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            coordinator_addr: "http://127.0.0.1:50051".to_string(),
            hostname: hostname::get()
                .ok()
                .and_then(|h| h.into_string().ok())
                .unwrap_or_else(|| "unknown".to_string()),
            ip_address: "127.0.0.1".to_string(),
            port: 50052,
            heartbeat_interval: Duration::from_secs(30),
            poll_interval: Duration::from_secs(5),
            max_concurrent_jobs: 4,
            capabilities: WorkerCapabilities::detect(),
            work_dir: std::env::temp_dir(),
            enable_gpu: false,
        }
    }
}

/// Worker capabilities detection
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    pub cpu_cores: u32,
    pub memory_bytes: u64,
    pub gpu_devices: Vec<String>,
    pub supported_codecs: Vec<String>,
    pub supported_hwaccels: Vec<String>,
    pub relative_speed: f32,
}

impl WorkerCapabilities {
    /// Detect system capabilities
    #[must_use]
    pub fn detect() -> Self {
        let cpu_cores = num_cpus::get() as u32;
        let memory_bytes = Self::detect_memory();
        let gpu_devices = Self::detect_gpus();
        let supported_codecs = Self::detect_codecs();
        let supported_hwaccels = Self::detect_hwaccels();
        let relative_speed = Self::benchmark_speed();

        Self {
            cpu_cores,
            memory_bytes,
            gpu_devices,
            supported_codecs,
            supported_hwaccels,
            relative_speed,
        }
    }

    fn detect_memory() -> u64 {
        // Simplified memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(info) = sys_info::mem_info() {
                return info.total * 1024;
            }
        }
        4_294_967_296 // Default 4GB
    }

    fn detect_gpus() -> Vec<String> {
        // Simplified GPU detection
        // In production, use proper GPU detection libraries
        Vec::new()
    }

    fn detect_codecs() -> Vec<String> {
        // Return common codecs
        vec![
            "h264".to_string(),
            "h265".to_string(),
            "vp9".to_string(),
            "av1".to_string(),
        ]
    }

    fn detect_hwaccels() -> Vec<String> {
        // Detect hardware acceleration support
        Vec::new()
    }

    fn benchmark_speed() -> f32 {
        // Simple benchmark, return baseline speed
        1.0
    }
}

/// Worker internal state
struct WorkerState {
    status: Arc<RwLock<WorkerStatus>>,
    metrics: LocalWorkerMetrics,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            status: Arc::new(RwLock::new(WorkerStatus::Idle)),
            metrics: LocalWorkerMetrics::new(),
        }
    }
}

/// Worker status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    Idle,
    Busy,
    Full,
    Draining,
    Error,
}

impl From<WorkerStatus> for i32 {
    fn from(status: WorkerStatus) -> Self {
        match status {
            WorkerStatus::Idle => 0,
            WorkerStatus::Busy => 1,
            WorkerStatus::Full => 2,
            WorkerStatus::Draining => 3,
            WorkerStatus::Error => 4,
        }
    }
}

/// Worker metrics tracking
#[derive(Clone)]
struct LocalWorkerMetrics {
    cpu_usage: Arc<AtomicU32>,
    memory_usage: Arc<AtomicU32>,
    gpu_usage: Arc<AtomicU32>,
    bytes_processed: Arc<AtomicU64>,
    frames_encoded: Arc<AtomicU32>,
}

impl LocalWorkerMetrics {
    fn new() -> Self {
        Self {
            cpu_usage: Arc::new(AtomicU32::new(0)),
            memory_usage: Arc::new(AtomicU32::new(0)),
            gpu_usage: Arc::new(AtomicU32::new(0)),
            bytes_processed: Arc::new(AtomicU64::new(0)),
            frames_encoded: Arc::new(AtomicU32::new(0)),
        }
    }

    fn update_system_metrics(&self) {
        // Update CPU and memory usage
        #[cfg(target_os = "linux")]
        {
            if let Ok(load) = sys_info::loadavg() {
                let cpu_percent = (load.one * 100.0) as u32;
                self.cpu_usage.store(cpu_percent, Ordering::Relaxed);
            }

            if let Ok(mem) = sys_info::mem_info() {
                let mem_percent = ((mem.total - mem.avail) * 100 / mem.total) as u32;
                self.memory_usage.store(mem_percent, Ordering::Relaxed);
            }
        }
    }

    fn to_proto(&self) -> pb::WorkerMetrics {
        pb::WorkerMetrics {
            cpu_usage: self.cpu_usage.load(Ordering::Relaxed) as f32 / 100.0,
            memory_usage: self.memory_usage.load(Ordering::Relaxed) as f32 / 100.0,
            gpu_usage: self.gpu_usage.load(Ordering::Relaxed) as f32 / 100.0,
            bytes_processed: self.bytes_processed.load(Ordering::Relaxed),
            frames_encoded: self.frames_encoded.load(Ordering::Relaxed),
        }
    }
}

/// Active job tracking
#[allow(dead_code)]
struct ActiveJob {
    job_id: String,
    task: EncodingTask,
    progress: f32,
    started_at: SystemTime,
    cancel_tx: mpsc::Sender<()>,
}

impl Worker {
    /// Create a new worker with the given configuration
    #[must_use]
    pub fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            worker_id: Arc::new(RwLock::new(Uuid::new_v4().to_string())),
            client: Arc::new(RwLock::new(None)),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(WorkerState::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the worker
    pub async fn run(&self) -> Result<()> {
        info!("Starting worker");

        // Connect to coordinator
        self.connect().await?;

        // Register with coordinator
        self.register().await?;

        // Start background tasks
        let worker = self.clone_arc();
        tokio::spawn(async move {
            worker.heartbeat_loop().await;
        });

        let worker = self.clone_arc();
        tokio::spawn(async move {
            worker.job_poll_loop().await;
        });

        // Wait for shutdown
        while !self.shutdown.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(1)).await;
        }

        // Unregister
        self.unregister().await?;

        Ok(())
    }

    fn clone_arc(&self) -> Arc<Self> {
        // Helper to create Arc clone
        // In real implementation, Worker would be wrapped in Arc
        Arc::new(Self {
            config: self.config.clone(),
            worker_id: self.worker_id.clone(),
            client: self.client.clone(),
            active_jobs: self.active_jobs.clone(),
            state: Arc::new(WorkerState {
                status: self.state.status.clone(),
                metrics: LocalWorkerMetrics {
                    cpu_usage: self.state.metrics.cpu_usage.clone(),
                    memory_usage: self.state.metrics.memory_usage.clone(),
                    gpu_usage: self.state.metrics.gpu_usage.clone(),
                    bytes_processed: self.state.metrics.bytes_processed.clone(),
                    frames_encoded: self.state.metrics.frames_encoded.clone(),
                },
            }),
            shutdown: self.shutdown.clone(),
        })
    }

    /// Connect to coordinator
    async fn connect(&self) -> Result<()> {
        info!(
            "Connecting to coordinator at {}",
            self.config.coordinator_addr
        );

        let client =
            CoordinatorServiceClient::connect(self.config.coordinator_addr.clone()).await?;

        let mut client_lock = self.client.write().await;
        *client_lock = Some(client);

        info!("Connected to coordinator");
        Ok(())
    }

    /// Register with coordinator
    async fn register(&self) -> Result<()> {
        info!("Registering with coordinator");

        let capabilities = pb::WorkerCapabilities {
            cpu_cores: self.config.capabilities.cpu_cores,
            memory_bytes: self.config.capabilities.memory_bytes,
            gpu_devices: self.config.capabilities.gpu_devices.clone(),
            supported_codecs: self.config.capabilities.supported_codecs.clone(),
            supported_hwaccels: self.config.capabilities.supported_hwaccels.clone(),
            relative_speed: self.config.capabilities.relative_speed,
            max_concurrent_jobs: self.config.max_concurrent_jobs,
        };

        let registration = WorkerRegistration {
            worker_id: self.worker_id.read().await.clone(),
            hostname: self.config.hostname.clone(),
            ip_address: self.config.ip_address.clone(),
            port: self.config.port,
            capabilities: Some(Box::new(capabilities)),
            metadata: HashMap::new(),
        };

        let mut client = self.client.write().await;
        if let Some(ref mut c) = *client {
            let response = c
                .register_worker(Request::new(registration))
                .await?
                .into_inner();

            if response.success {
                let mut worker_id = self.worker_id.write().await;
                *worker_id = response.assigned_worker_id;
                info!("Registered with ID: {}", *worker_id);
                Ok(())
            } else {
                Err(DistributedError::Worker(response.message))
            }
        } else {
            Err(DistributedError::Worker("Not connected".to_string()))
        }
    }

    /// Unregister from coordinator
    async fn unregister(&self) -> Result<()> {
        info!("Unregistering from coordinator");

        let unreg = WorkerUnregistration {
            worker_id: self.worker_id.read().await.clone(),
            reason: "Shutdown".to_string(),
        };

        let mut client = self.client.write().await;
        if let Some(ref mut c) = *client {
            c.unregister_worker(Request::new(unreg)).await?;
        }

        Ok(())
    }

    /// Heartbeat loop
    async fn heartbeat_loop(&self) {
        let mut interval = tokio::time::interval(self.config.heartbeat_interval);

        loop {
            interval.tick().await;

            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            if let Err(e) = self.send_heartbeat().await {
                error!("Heartbeat failed: {}", e);
            }
        }
    }

    /// Send heartbeat to coordinator
    async fn send_heartbeat(&self) -> Result<()> {
        debug!("Sending heartbeat");

        // Update metrics
        self.state.metrics.update_system_metrics();

        let active_jobs = self.active_jobs.read().await;
        let active_job_ids: Vec<String> = active_jobs.keys().cloned().collect();

        let status = *self.state.status.read().await;
        let status_proto = pb::WorkerStatus {
            state: i32::from(status),
            active_jobs: active_jobs.len() as u32,
            queued_jobs: 0,
        };

        let heartbeat = WorkerHeartbeat {
            worker_id: self.worker_id.read().await.clone(),
            status: Some(Box::new(status_proto)),
            active_job_ids,
            metrics: Some(Box::new(self.state.metrics.to_proto())),
        };

        let mut client = self.client.write().await;
        if let Some(ref mut c) = *client {
            let response = c.heartbeat(Request::new(heartbeat)).await?.into_inner();

            // Handle coordinator commands
            if response.should_drain {
                info!("Coordinator requested drain");
                let mut status = self.state.status.write().await;
                *status = WorkerStatus::Draining;
            }

            // Handle job cancellations
            for job_id in response.jobs_to_cancel {
                self.cancel_job(&job_id).await;
            }
        }

        Ok(())
    }

    /// Job polling loop
    async fn job_poll_loop(&self) {
        let mut interval = tokio::time::interval(self.config.poll_interval);

        loop {
            interval.tick().await;

            if self.shutdown.load(Ordering::Relaxed) {
                break;
            }

            let status = *self.state.status.read().await;
            if status == WorkerStatus::Draining || status == WorkerStatus::Error {
                continue;
            }

            let active_count = self.active_jobs.read().await.len();
            if active_count >= self.config.max_concurrent_jobs as usize {
                // Update status to Full
                let mut status_lock = self.state.status.write().await;
                *status_lock = WorkerStatus::Full;
                continue;
            }

            // Request jobs
            if let Err(e) = self.request_and_execute_jobs().await {
                error!("Job request failed: {}", e);
            }
        }
    }

    /// Request and execute jobs from coordinator
    async fn request_and_execute_jobs(&self) -> Result<()> {
        let active_count = self.active_jobs.read().await.len() as u32;
        let max_jobs = self.config.max_concurrent_jobs.saturating_sub(active_count);

        if max_jobs == 0 {
            return Ok(());
        }

        let request = JobRequest {
            worker_id: self.worker_id.read().await.clone(),
            max_jobs,
            preferred_codecs: self.config.capabilities.supported_codecs.clone(),
        };

        let mut client = self.client.write().await;
        if let Some(ref mut c) = *client {
            let response = c.request_job(Request::new(request)).await?.into_inner();

            for job in response.jobs {
                if let Some(task) = job.task {
                    info!("Received job: {}", job.job_id);
                    self.execute_job(job.job_id, *task).await;
                }
            }

            // Update status
            drop(client);
            let active_count = self.active_jobs.read().await.len();
            let mut status = self.state.status.write().await;
            *status = if active_count == 0 {
                WorkerStatus::Idle
            } else if active_count >= self.config.max_concurrent_jobs as usize {
                WorkerStatus::Full
            } else {
                WorkerStatus::Busy
            };
        }

        Ok(())
    }

    /// Execute an encoding job
    async fn execute_job(&self, job_id: String, task: EncodingTask) {
        let (cancel_tx, mut cancel_rx) = mpsc::channel(1);

        let active_job = ActiveJob {
            job_id: job_id.clone(),
            task: task.clone(),
            progress: 0.0,
            started_at: SystemTime::now(),
            cancel_tx,
        };

        self.active_jobs
            .write()
            .await
            .insert(job_id.clone(), active_job);

        let worker_id = self.worker_id.read().await.clone();
        let client = self.client.clone();
        let active_jobs = self.active_jobs.clone();
        let metrics = self.state.metrics.clone();

        tokio::spawn(async move {
            let result = Self::run_encoding_task(
                &task,
                &job_id,
                &worker_id,
                client.clone(),
                &mut cancel_rx,
                metrics,
            )
            .await;

            // Remove from active jobs
            active_jobs.write().await.remove(&job_id);

            // Report result
            let mut client_lock = client.write().await;
            if let Some(ref mut c) = *client_lock {
                match result {
                    Ok(output_info) => {
                        let result_msg = JobResult {
                            job_id: job_id.clone(),
                            worker_id: worker_id.clone(),
                            output_url: output_info.output_url,
                            output_size: output_info.output_size,
                            encoding_time: output_info.encoding_time,
                            metadata: Some(Box::new(ResultMetadata {
                                frames_encoded: output_info.frames_encoded,
                                average_bitrate: output_info.average_bitrate,
                                checksum: output_info.checksum,
                                extra_metadata: HashMap::new(),
                            })),
                        };

                        if let Err(e) = c.submit_result(Request::new(result_msg)).await {
                            error!("Failed to submit result for job {}: {}", job_id, e);
                        } else {
                            info!("Job {} completed successfully", job_id);
                        }
                    }
                    Err(e) => {
                        error!("Job {} failed: {}", job_id, e);

                        let failure = JobFailure {
                            job_id: job_id.clone(),
                            worker_id: worker_id.clone(),
                            error_message: e.to_string(),
                            error_code: "ENCODING_ERROR".to_string(),
                            is_transient: false,
                        };

                        let _ = c.report_failure(Request::new(failure)).await;
                    }
                }
            }
        });
    }

    /// Run the actual encoding task
    #[allow(clippy::too_many_arguments)]
    async fn run_encoding_task(
        task: &EncodingTask,
        job_id: &str,
        worker_id: &str,
        client: Arc<RwLock<Option<CoordinatorServiceClient<Channel>>>>,
        cancel_rx: &mut mpsc::Receiver<()>,
        metrics: LocalWorkerMetrics,
    ) -> Result<EncodingOutput> {
        info!("Starting encoding task for job {}", job_id);

        // Simulate encoding work
        // In production, this would call actual encoding functions
        let total_frames = 1000u64;
        let mut frames_encoded = 0u64;

        for _ in 0..10 {
            // Check for cancellation
            if cancel_rx.try_recv().is_ok() {
                return Err(DistributedError::Job("Job cancelled".to_string()));
            }

            // Simulate encoding work
            tokio::time::sleep(Duration::from_millis(100)).await;
            frames_encoded += total_frames / 10;

            let progress = frames_encoded as f32 / total_frames as f32;

            // Report progress
            let progress_report = ProgressReport {
                job_id: job_id.to_string(),
                worker_id: worker_id.to_string(),
                progress,
                frames_encoded,
                bytes_written: frames_encoded * 1024,
                encoding_speed: 30.0,
                estimated_completion_timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
                    + 60,
            };

            let mut client_lock = client.write().await;
            if let Some(ref mut c) = *client_lock {
                let _ = c.report_progress(Request::new(progress_report)).await;
            }
        }

        // Update metrics
        metrics
            .frames_encoded
            .fetch_add(total_frames as u32, Ordering::Relaxed);
        metrics
            .bytes_processed
            .fetch_add(total_frames * 1024, Ordering::Relaxed);

        Ok(EncodingOutput {
            output_url: task.output_url.clone(),
            output_size: total_frames * 1024,
            encoding_time: 1.0,
            frames_encoded: total_frames,
            average_bitrate: 5000.0,
            checksum: "abc123".to_string(),
        })
    }

    /// Cancel a job
    async fn cancel_job(&self, job_id: &str) {
        info!("Cancelling job {}", job_id);

        if let Some(job) = self.active_jobs.write().await.remove(job_id) {
            let _ = job.cancel_tx.send(()).await;
        }
    }

    /// Shutdown the worker
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down worker");
        self.shutdown.store(true, Ordering::Relaxed);

        // Cancel all active jobs
        let active_jobs = self.active_jobs.read().await;
        for (_, job) in active_jobs.iter() {
            let _ = job.cancel_tx.send(()).await;
        }

        Ok(())
    }
}

/// Encoding output information
struct EncodingOutput {
    output_url: String,
    output_size: u64,
    encoding_time: f64,
    frames_encoded: u64,
    average_bitrate: f64,
    checksum: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.max_concurrent_jobs, 4);
        assert!(config.capabilities.cpu_cores > 0);
    }

    #[test]
    fn test_worker_capabilities_detection() {
        let caps = WorkerCapabilities::detect();
        assert!(caps.cpu_cores > 0);
        assert!(caps.memory_bytes > 0);
        assert!(!caps.supported_codecs.is_empty());
    }

    #[test]
    fn test_worker_status_conversion() {
        assert_eq!(i32::from(WorkerStatus::Idle), 0);
        assert_eq!(i32::from(WorkerStatus::Busy), 1);
        assert_eq!(i32::from(WorkerStatus::Full), 2);
    }

    #[test]
    fn test_worker_metrics() {
        let metrics = LocalWorkerMetrics::new();
        assert_eq!(metrics.cpu_usage.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.frames_encoded.load(Ordering::Relaxed), 0);

        metrics.frames_encoded.store(100, Ordering::Relaxed);
        assert_eq!(metrics.frames_encoded.load(Ordering::Relaxed), 100);
    }
}
