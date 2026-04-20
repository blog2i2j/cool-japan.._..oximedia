//! Central coordinator service for distributed encoding.
//!
//! The coordinator manages:
//! - Worker registration and health monitoring
//! - Job distribution and assignment
//! - Progress tracking
//! - Result collection and aggregation
//! - Fault tolerance and job rescheduling

use crate::pb::coordinator_service_server::CoordinatorService;
use crate::pb::{
    job_assignment, job_cancellation, job_status_request, worker_status_request,
    CancellationResponse, EncodingTask, FailureAcknowledgment, HeartbeatResponse, JobAssignment,
    JobCancellation, JobFailure, JobRequest, JobResult, JobStatusRequest, JobStatusResponse,
    ProgressAcknowledgment, ProgressReport, ResultAcknowledgment, UnregistrationResponse,
    WorkerHeartbeat, WorkerInfo, WorkerMetrics, WorkerRegistration, WorkerRegistrationResponse,
    WorkerStatus, WorkerStatusRequest, WorkerStatusResponse, WorkerUnregistration,
};
use crate::scheduler::{JobScheduler, ScheduledJob};
use crate::{JobPriority, Result};
use dashmap::DashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, RwLock};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Coordinator state and configuration
pub struct Coordinator {
    /// Worker registry
    workers: Arc<DashMap<String, WorkerState>>,

    /// Job registry
    jobs: Arc<DashMap<String, JobState>>,

    /// Job scheduler
    scheduler: Arc<RwLock<JobScheduler>>,

    /// Statistics
    stats: Arc<CoordinatorStats>,

    /// Configuration
    config: CoordinatorConfig,

    /// Shutdown signal
    shutdown_tx: mpsc::Sender<()>,
    #[allow(dead_code)]
    shutdown_rx: Arc<RwLock<mpsc::Receiver<()>>>,
}

/// Coordinator configuration
#[derive(Debug, Clone)]
pub struct CoordinatorConfig {
    /// Maximum workers
    pub max_workers: usize,

    /// Worker timeout
    pub worker_timeout: Duration,

    /// Job retry limit
    pub max_job_retries: u32,

    /// Enable preemption
    pub enable_preemption: bool,

    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        Self {
            max_workers: 1000,
            worker_timeout: Duration::from_secs(90),
            max_job_retries: 3,
            enable_preemption: false,
            load_balancing: LoadBalancingStrategy::LeastLoaded,
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Assign to worker with fewest jobs
    LeastLoaded,
    /// Round-robin assignment
    RoundRobin,
    /// Assign to fastest worker
    FastestFirst,
    /// Consider worker capabilities
    CapabilityBased,
}

/// Worker state tracking
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct WorkerState {
    worker_id: String,
    hostname: String,
    ip_address: String,
    port: u32,
    capabilities: worker_capabilities::Capabilities,
    status: worker_status::State,
    active_jobs: Vec<String>,
    metrics: worker_metrics::Metrics,
    last_heartbeat: SystemTime,
    total_jobs_completed: u64,
    total_jobs_failed: u64,
}

mod worker_capabilities {
    #[allow(dead_code)]
    #[derive(Debug, Clone)]
    pub struct Capabilities {
        pub cpu_cores: u32,
        pub memory_bytes: u64,
        pub gpu_devices: Vec<String>,
        pub supported_codecs: Vec<String>,
        pub supported_hwaccels: Vec<String>,
        pub relative_speed: f32,
        pub max_concurrent_jobs: u32,
    }

    impl Default for Capabilities {
        fn default() -> Self {
            Self {
                cpu_cores: 1,
                memory_bytes: 1_073_741_824, // 1 GB
                gpu_devices: Vec::new(),
                supported_codecs: vec!["h264".to_string()],
                supported_hwaccels: Vec::new(),
                relative_speed: 1.0,
                max_concurrent_jobs: 2,
            }
        }
    }
}

mod worker_status {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum State {
        Idle,
        Busy,
        Full,
        Draining,
        Error,
    }

    impl From<i32> for State {
        fn from(value: i32) -> Self {
            match value {
                0 => State::Idle,
                1 => State::Busy,
                2 => State::Full,
                3 => State::Draining,
                4 => State::Error,
                _ => State::Error,
            }
        }
    }

    impl From<State> for i32 {
        fn from(state: State) -> Self {
            match state {
                State::Idle => 0,
                State::Busy => 1,
                State::Full => 2,
                State::Draining => 3,
                State::Error => 4,
            }
        }
    }
}

mod worker_metrics {
    #[allow(dead_code)]
    #[derive(Debug, Clone, Default)]
    pub struct Metrics {
        pub cpu_usage: f32,
        pub memory_usage: f32,
        pub gpu_usage: f32,
        pub bytes_processed: u64,
        pub frames_encoded: u32,
    }
}

/// Job state tracking
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct JobState {
    job_id: String,
    task_id: String,
    assigned_worker: Option<String>,
    status: job_status::State,
    priority: u32,
    retry_count: u32,
    progress: f32,
    created_at: SystemTime,
    started_at: Option<SystemTime>,
    completed_at: Option<SystemTime>,
    encoding_task: Option<EncodingTask>,
}

mod job_status {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum State {
        Pending,
        Assigned,
        InProgress,
        Completed,
        Failed,
        Cancelled,
    }

    impl From<State> for i32 {
        fn from(state: State) -> Self {
            match state {
                State::Pending => 0,
                State::Assigned => 1,
                State::InProgress => 2,
                State::Completed => 3,
                State::Failed => 4,
                State::Cancelled => 5,
            }
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Default)]
struct CoordinatorStats {
    total_workers: AtomicU64,
    active_workers: AtomicU64,
    total_jobs_submitted: AtomicU64,
    total_jobs_completed: AtomicU64,
    total_jobs_failed: AtomicU64,
    total_bytes_processed: AtomicU64,
}

impl Coordinator {
    /// Create a new coordinator
    #[must_use]
    pub fn new(config: CoordinatorConfig) -> Self {
        let (shutdown_tx, shutdown_rx) = mpsc::channel(1);

        Self {
            workers: Arc::new(DashMap::new()),
            jobs: Arc::new(DashMap::new()),
            scheduler: Arc::new(RwLock::new(JobScheduler::new())),
            stats: Arc::new(CoordinatorStats::default()),
            config,
            shutdown_tx,
            shutdown_rx: Arc::new(RwLock::new(shutdown_rx)),
        }
    }

    /// Start the coordinator server
    pub async fn serve(self, addr: SocketAddr) -> Result<()> {
        info!("Starting coordinator on {}", addr);

        let coordinator = Arc::new(self);

        // Start background tasks
        let coord_clone = coordinator.clone();
        tokio::spawn(async move {
            coord_clone.health_check_loop().await;
        });

        let coord_clone = coordinator.clone();
        tokio::spawn(async move {
            coord_clone.job_assignment_loop().await;
        });

        // Start in-process TCP control server on configured port.
        // Accepts basic text commands: "status", "nodes", "jobs" and returns JSON responses.
        {
            let coord_for_tcp = coordinator.clone();
            let tcp_addr = addr;
            std::thread::spawn(move || {
                let listener = match TcpListener::bind(tcp_addr) {
                    Ok(l) => l,
                    Err(e) => {
                        error!("TCP control server failed to bind {}: {}", tcp_addr, e);
                        return;
                    }
                };
                info!("TCP control server listening on {}", tcp_addr);

                for stream in listener.incoming() {
                    match stream {
                        Ok(mut tcp_stream) => {
                            let coord = coord_for_tcp.clone();
                            let cloned = match tcp_stream.try_clone() {
                                Ok(s) => s,
                                Err(e) => {
                                    debug!("TCP stream clone error: {}", e);
                                    continue;
                                }
                            };
                            std::thread::spawn(move || {
                                let reader = BufReader::new(cloned);
                                for line in reader.lines() {
                                    let command = match line {
                                        Ok(l) => l.trim().to_lowercase(),
                                        Err(_) => break,
                                    };

                                    let response = match command.as_str() {
                                        "status" => {
                                            let stats = coord.stats();
                                            format!(
                                                "{{\"total_workers\":{},\"active_workers\":{},\"total_jobs_submitted\":{},\"total_jobs_completed\":{},\"total_jobs_failed\":{},\"total_bytes_processed\":{}}}\n",
                                                stats.total_workers,
                                                stats.active_workers,
                                                stats.total_jobs_submitted,
                                                stats.total_jobs_completed,
                                                stats.total_jobs_failed,
                                                stats.total_bytes_processed,
                                            )
                                        }
                                        "nodes" => {
                                            let workers: Vec<String> = coord
                                                .workers
                                                .iter()
                                                .map(|e| {
                                                    format!(
                                                        "{{\"id\":\"{}\",\"host\":\"{}\",\"jobs\":{}}}",
                                                        e.value().worker_id,
                                                        e.value().hostname,
                                                        e.value().active_jobs.len(),
                                                    )
                                                })
                                                .collect();
                                            format!("[{}]\n", workers.join(","))
                                        }
                                        "jobs" => {
                                            let jobs: Vec<String> = coord
                                                .jobs
                                                .iter()
                                                .map(|e| {
                                                    let state: i32 =
                                                        e.value().status
                                                            .into();
                                                    format!(
                                                        "{{\"id\":\"{}\",\"state\":{},\"progress\":{}}}",
                                                        e.value().job_id,
                                                        state,
                                                        e.value().progress,
                                                    )
                                                })
                                                .collect();
                                            format!("[{}]\n", jobs.join(","))
                                        }
                                        "" => continue,
                                        _ => "{\"error\":\"unknown command; try: status, nodes, jobs\"}\n".to_string(),
                                    };

                                    if tcp_stream.write_all(response.as_bytes()).is_err() {
                                        break;
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            debug!("TCP control accept error: {}", e);
                        }
                    }
                }
            });
        }

        // Wait for shutdown signal
        tokio::signal::ctrl_c().await?;
        info!("Coordinator shutting down");
        Ok(())
    }

    /// Health check loop for monitoring workers
    async fn health_check_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            let now = SystemTime::now();
            let timeout = self.config.worker_timeout;

            // Check for timed-out workers
            let timed_out_workers: Vec<String> = self
                .workers
                .iter()
                .filter_map(|entry| {
                    let worker_id = entry.key().clone();
                    let worker = entry.value();

                    if let Ok(elapsed) = now.duration_since(worker.last_heartbeat) {
                        if elapsed > timeout {
                            return Some(worker_id);
                        }
                    }
                    None
                })
                .collect();

            // Handle timed-out workers
            for worker_id in timed_out_workers {
                warn!("Worker {} timed out, marking as failed", worker_id);
                self.handle_worker_failure(&worker_id).await;
            }

            // Update statistics
            self.stats
                .active_workers
                .store(self.workers.len() as u64, Ordering::Relaxed);

            debug!(
                "Health check: {} active workers, {} jobs",
                self.workers.len(),
                self.jobs.len()
            );
        }
    }

    /// Job assignment loop
    async fn job_assignment_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(5));

        loop {
            interval.tick().await;

            let mut scheduler = self.scheduler.write().await;

            // Get pending jobs from scheduler
            while let Some(scheduled_job) = scheduler.next_job() {
                // Find suitable worker
                if let Some(worker_id) = self.find_suitable_worker(&scheduled_job).await {
                    debug!(
                        "Assigning job {} to worker {}",
                        scheduled_job.job_id, worker_id
                    );

                    // Update job state
                    if let Some(mut job) = self.jobs.get_mut(&scheduled_job.job_id) {
                        job.assigned_worker = Some(worker_id.clone());
                        job.status = job_status::State::Assigned;
                    }

                    // Update worker state
                    if let Some(mut worker) = self.workers.get_mut(&worker_id) {
                        worker.active_jobs.push(scheduled_job.job_id.clone());
                    }
                } else {
                    // No suitable worker, put job back in queue
                    scheduler.enqueue(scheduled_job);
                    break;
                }
            }
        }
    }

    /// Find a suitable worker for a job
    async fn find_suitable_worker(&self, job: &ScheduledJob) -> Option<String> {
        match self.config.load_balancing {
            LoadBalancingStrategy::LeastLoaded => self.find_least_loaded_worker(),
            LoadBalancingStrategy::RoundRobin => self.find_round_robin_worker(),
            LoadBalancingStrategy::FastestFirst => self.find_fastest_worker(),
            LoadBalancingStrategy::CapabilityBased => self.find_capability_worker(job),
        }
    }

    fn find_least_loaded_worker(&self) -> Option<String> {
        self.workers
            .iter()
            .filter(|entry| {
                let worker = entry.value();
                worker.status != worker_status::State::Full
                    && worker.status != worker_status::State::Error
                    && worker.status != worker_status::State::Draining
            })
            .min_by_key(|entry| entry.value().active_jobs.len())
            .map(|entry| entry.key().clone())
    }

    fn find_round_robin_worker(&self) -> Option<String> {
        // Simple round-robin based on total jobs completed
        self.workers
            .iter()
            .filter(|entry| {
                let worker = entry.value();
                worker.status != worker_status::State::Full
                    && worker.status != worker_status::State::Error
            })
            .min_by_key(|entry| entry.value().total_jobs_completed)
            .map(|entry| entry.key().clone())
    }

    fn find_fastest_worker(&self) -> Option<String> {
        self.workers
            .iter()
            .filter(|entry| {
                let worker = entry.value();
                worker.status != worker_status::State::Full
                    && worker.status != worker_status::State::Error
            })
            .max_by(|a, b| {
                a.value()
                    .capabilities
                    .relative_speed
                    .partial_cmp(&b.value().capabilities.relative_speed)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.key().clone())
    }

    fn find_capability_worker(&self, job: &ScheduledJob) -> Option<String> {
        // Find worker with best match for job requirements
        self.workers
            .iter()
            .filter(|entry| {
                let worker = entry.value();
                if worker.status == worker_status::State::Full
                    || worker.status == worker_status::State::Error
                {
                    return false;
                }

                // Check codec support
                worker.capabilities.supported_codecs.iter().any(|codec| {
                    job.encoding_task
                        .as_ref()
                        .is_some_and(|task| task.codec == *codec)
                })
            })
            .max_by_key(|entry| {
                let worker = entry.value();
                // Score based on relative speed and current load
                let load_factor = 1.0
                    - (worker.active_jobs.len() as f32
                        / worker.capabilities.max_concurrent_jobs as f32);
                (worker.capabilities.relative_speed * load_factor * 1000.0) as u64
            })
            .map(|entry| entry.key().clone())
    }

    /// Handle worker failure
    async fn handle_worker_failure(&self, worker_id: &str) {
        if let Some(worker) = self.workers.get(worker_id) {
            let failed_jobs = worker.active_jobs.clone();

            // Reschedule failed jobs
            for job_id in failed_jobs {
                if let Some(mut job) = self.jobs.get_mut(&job_id) {
                    job.retry_count += 1;

                    if job.retry_count < self.config.max_job_retries {
                        info!("Rescheduling job {} after worker failure", job_id);
                        job.assigned_worker = None;
                        job.status = job_status::State::Pending;

                        // Add back to scheduler
                        let mut scheduler = self.scheduler.write().await;
                        if let Some(task) = &job.encoding_task {
                            scheduler.enqueue(ScheduledJob {
                                job_id: job_id.clone(),
                                task_id: job.task_id.clone(),
                                priority: JobPriority::Normal,
                                deadline: None,
                                encoding_task: Some(task.clone()),
                            });
                        }
                    } else {
                        error!("Job {} failed after {} retries", job_id, job.retry_count);
                        job.status = job_status::State::Failed;
                        self.stats.total_jobs_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // Remove worker
        self.workers.remove(worker_id);
        info!("Worker {} removed after failure", worker_id);
    }

    /// Submit a new encoding job
    pub async fn submit_job(&self, task: EncodingTask, priority: u32) -> Result<String> {
        let job_id = Uuid::new_v4().to_string();
        let task_id = task.task_id.clone();

        let job_state = JobState {
            job_id: job_id.clone(),
            task_id: task_id.clone(),
            assigned_worker: None,
            status: job_status::State::Pending,
            priority,
            retry_count: 0,
            progress: 0.0,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            encoding_task: Some(task.clone()),
        };

        self.jobs.insert(job_id.clone(), job_state);

        // Add to scheduler
        let mut scheduler = self.scheduler.write().await;
        let priority_enum = match priority {
            3 => JobPriority::Critical,
            2 => JobPriority::High,
            1 => JobPriority::Normal,
            _ => JobPriority::Low,
        };

        scheduler.enqueue(ScheduledJob {
            job_id: job_id.clone(),
            task_id,
            priority: priority_enum,
            deadline: None,
            encoding_task: Some(task),
        });

        self.stats
            .total_jobs_submitted
            .fetch_add(1, Ordering::Relaxed);

        info!("Job {} submitted with priority {}", job_id, priority);
        Ok(job_id)
    }

    /// Get coordinator statistics
    #[must_use]
    pub fn stats(&self) -> CoordinatorStatistics {
        CoordinatorStatistics {
            total_workers: self.stats.total_workers.load(Ordering::Relaxed),
            active_workers: self.stats.active_workers.load(Ordering::Relaxed),
            total_jobs_submitted: self.stats.total_jobs_submitted.load(Ordering::Relaxed),
            total_jobs_completed: self.stats.total_jobs_completed.load(Ordering::Relaxed),
            total_jobs_failed: self.stats.total_jobs_failed.load(Ordering::Relaxed),
            total_bytes_processed: self.stats.total_bytes_processed.load(Ordering::Relaxed),
        }
    }

    /// Shutdown the coordinator
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down coordinator");
        let _ = self.shutdown_tx.send(()).await;
        Ok(())
    }
}

/// Coordinator statistics snapshot
#[derive(Debug, Clone)]
pub struct CoordinatorStatistics {
    pub total_workers: u64,
    pub active_workers: u64,
    pub total_jobs_submitted: u64,
    pub total_jobs_completed: u64,
    pub total_jobs_failed: u64,
    pub total_bytes_processed: u64,
}

/// gRPC service implementation
#[derive(Clone)]
struct CoordinatorServiceImpl {
    coordinator: Arc<Coordinator>,
}

#[tonic::async_trait]
impl CoordinatorService for CoordinatorServiceImpl {
    async fn register_worker(
        &self,
        request: Request<WorkerRegistration>,
    ) -> std::result::Result<Response<WorkerRegistrationResponse>, Status> {
        let registration = request.into_inner();
        let worker_id = if registration.worker_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            registration.worker_id
        };

        info!("Registering worker: {}", worker_id);

        // Check worker limit
        if self.coordinator.workers.len() >= self.coordinator.config.max_workers {
            return Ok(Response::new(WorkerRegistrationResponse {
                success: false,
                message: "Maximum worker limit reached".to_string(),
                assigned_worker_id: String::new(),
            }));
        }

        let capabilities = registration
            .capabilities
            .map(|c| worker_capabilities::Capabilities {
                cpu_cores: c.cpu_cores,
                memory_bytes: c.memory_bytes,
                gpu_devices: c.gpu_devices,
                supported_codecs: c.supported_codecs,
                supported_hwaccels: c.supported_hwaccels,
                relative_speed: c.relative_speed,
                max_concurrent_jobs: c.max_concurrent_jobs,
            })
            .unwrap_or_default();

        let worker_state = WorkerState {
            worker_id: worker_id.clone(),
            hostname: registration.hostname,
            ip_address: registration.ip_address,
            port: registration.port,
            capabilities,
            status: worker_status::State::Idle,
            active_jobs: Vec::new(),
            metrics: worker_metrics::Metrics::default(),
            last_heartbeat: SystemTime::now(),
            total_jobs_completed: 0,
            total_jobs_failed: 0,
        };

        self.coordinator
            .workers
            .insert(worker_id.clone(), worker_state);
        self.coordinator
            .stats
            .total_workers
            .fetch_add(1, Ordering::Relaxed);

        Ok(Response::new(WorkerRegistrationResponse {
            success: true,
            message: "Worker registered successfully".to_string(),
            assigned_worker_id: worker_id,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<WorkerHeartbeat>,
    ) -> std::result::Result<Response<HeartbeatResponse>, Status> {
        let heartbeat = request.into_inner();
        let worker_id = heartbeat.worker_id;

        if let Some(mut worker) = self.coordinator.workers.get_mut(&worker_id) {
            worker.last_heartbeat = SystemTime::now();

            if let Some(status) = heartbeat.status {
                worker.status = worker_status::State::from(status.state);
            }

            if let Some(metrics) = heartbeat.metrics {
                worker.metrics = worker_metrics::Metrics {
                    cpu_usage: metrics.cpu_usage,
                    memory_usage: metrics.memory_usage,
                    gpu_usage: metrics.gpu_usage,
                    bytes_processed: metrics.bytes_processed,
                    frames_encoded: metrics.frames_encoded,
                };
            }

            worker.active_jobs = heartbeat.active_job_ids;
        } else {
            return Err(Status::not_found(format!("Worker {worker_id} not found")));
        }

        Ok(Response::new(HeartbeatResponse {
            acknowledged: true,
            jobs_to_cancel: Vec::new(),
            should_drain: false,
        }))
    }

    async fn unregister_worker(
        &self,
        request: Request<WorkerUnregistration>,
    ) -> std::result::Result<Response<UnregistrationResponse>, Status> {
        let unreg = request.into_inner();
        info!(
            "Unregistering worker: {} ({})",
            unreg.worker_id, unreg.reason
        );

        self.coordinator.workers.remove(&unreg.worker_id);

        Ok(Response::new(UnregistrationResponse { success: true }))
    }

    async fn request_job(
        &self,
        request: Request<JobRequest>,
    ) -> std::result::Result<Response<JobAssignment>, Status> {
        let req = request.into_inner();
        let worker_id = req.worker_id;

        // Find jobs assigned to this worker
        let assigned_jobs: Vec<job_assignment::Job> = self
            .coordinator
            .jobs
            .iter()
            .filter(|entry| {
                entry.value().assigned_worker.as_ref() == Some(&worker_id)
                    && entry.value().status == job_status::State::Assigned
            })
            .take(req.max_jobs as usize)
            .map(|entry| {
                let job = entry.value();
                job_assignment::Job {
                    job_id: job.job_id.clone(),
                    task: job.encoding_task.clone().map(Box::new),
                    priority: job.priority,
                    deadline_timestamp: 0,
                }
            })
            .collect();

        Ok(Response::new(JobAssignment {
            jobs: assigned_jobs,
            has_more: false,
        }))
    }

    async fn report_progress(
        &self,
        request: Request<ProgressReport>,
    ) -> std::result::Result<Response<ProgressAcknowledgment>, Status> {
        let progress = request.into_inner();

        if let Some(mut job) = self.coordinator.jobs.get_mut(&progress.job_id) {
            job.progress = progress.progress;
            if job.status == job_status::State::Assigned {
                job.status = job_status::State::InProgress;
                job.started_at = Some(SystemTime::now());
            }
        }

        Ok(Response::new(ProgressAcknowledgment { acknowledged: true }))
    }

    async fn submit_result(
        &self,
        request: Request<JobResult>,
    ) -> std::result::Result<Response<ResultAcknowledgment>, Status> {
        let result = request.into_inner();
        info!(
            "Job {} completed by worker {}",
            result.job_id, result.worker_id
        );

        if let Some(mut job) = self.coordinator.jobs.get_mut(&result.job_id) {
            job.status = job_status::State::Completed;
            job.completed_at = Some(SystemTime::now());

            // Update worker stats
            if let Some(mut worker) = self.coordinator.workers.get_mut(&result.worker_id) {
                worker.total_jobs_completed += 1;
                worker.active_jobs.retain(|j| j != &result.job_id);
            }

            self.coordinator
                .stats
                .total_jobs_completed
                .fetch_add(1, Ordering::Relaxed);
            self.coordinator
                .stats
                .total_bytes_processed
                .fetch_add(result.output_size, Ordering::Relaxed);
        }

        Ok(Response::new(ResultAcknowledgment {
            acknowledged: true,
            next_job_id: String::new(),
        }))
    }

    async fn report_failure(
        &self,
        request: Request<JobFailure>,
    ) -> std::result::Result<Response<FailureAcknowledgment>, Status> {
        let failure = request.into_inner();
        error!(
            "Job {} failed on worker {}: {}",
            failure.job_id, failure.worker_id, failure.error_message
        );

        let should_retry = if let Some(mut job) = self.coordinator.jobs.get_mut(&failure.job_id) {
            job.retry_count += 1;

            if failure.is_transient && job.retry_count < self.coordinator.config.max_job_retries {
                job.assigned_worker = None;
                job.status = job_status::State::Pending;
                true
            } else {
                job.status = job_status::State::Failed;
                self.coordinator
                    .stats
                    .total_jobs_failed
                    .fetch_add(1, Ordering::Relaxed);
                false
            }
        } else {
            false
        };

        // Update worker stats
        if let Some(mut worker) = self.coordinator.workers.get_mut(&failure.worker_id) {
            worker.total_jobs_failed += 1;
            worker.active_jobs.retain(|j| j != &failure.job_id);
        }

        Ok(Response::new(FailureAcknowledgment {
            should_retry,
            reassigned_job_id: String::new(),
        }))
    }

    async fn get_worker_status(
        &self,
        request: Request<WorkerStatusRequest>,
    ) -> std::result::Result<Response<WorkerStatusResponse>, Status> {
        let req = request.into_inner();

        let workers = match req.query {
            Some(worker_status_request::Query::WorkerId(id)) => {
                if let Some(worker) = self.coordinator.workers.get(&id) {
                    vec![self.worker_to_info(&worker)]
                } else {
                    Vec::new()
                }
            }
            Some(worker_status_request::Query::AllWorkers(true)) | None => self
                .coordinator
                .workers
                .iter()
                .map(|entry| self.worker_to_info(&entry))
                .collect(),
            _ => Vec::new(),
        };

        Ok(Response::new(WorkerStatusResponse { workers }))
    }

    async fn get_job_status(
        &self,
        request: Request<JobStatusRequest>,
    ) -> std::result::Result<Response<JobStatusResponse>, Status> {
        let req = request.into_inner();

        let job_id = match req.query {
            Some(job_status_request::Query::JobId(id)) => id,
            Some(job_status_request::Query::TaskId(_task_id)) => {
                // Find first job for task
                String::new()
            }
            None => String::new(),
        };

        if let Some(job) = self.coordinator.jobs.get(&job_id) {
            let started = job
                .started_at
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs() as i64);

            let completed = job
                .completed_at
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs() as i64);

            Ok(Response::new(JobStatusResponse {
                job_id: job.job_id.clone(),
                state: job.status.into(),
                assigned_worker_id: job.assigned_worker.clone().unwrap_or_default(),
                progress: job.progress,
                started_timestamp: started,
                completed_timestamp: completed,
            }))
        } else {
            Err(Status::not_found(format!("Job {job_id} not found")))
        }
    }

    async fn cancel_job(
        &self,
        request: Request<JobCancellation>,
    ) -> std::result::Result<Response<CancellationResponse>, Status> {
        let cancellation = request.into_inner();

        let mut cancelled_jobs = Vec::new();

        match cancellation.target {
            Some(job_cancellation::Target::JobId(job_id)) => {
                if let Some(mut job) = self.coordinator.jobs.get_mut(&job_id) {
                    job.status = job_status::State::Cancelled;
                    cancelled_jobs.push(job_id);
                }
            }
            Some(job_cancellation::Target::TaskId(task_id)) => {
                for mut entry in self.coordinator.jobs.iter_mut() {
                    if entry.value().task_id == task_id {
                        entry.value_mut().status = job_status::State::Cancelled;
                        cancelled_jobs.push(entry.key().clone());
                    }
                }
            }
            None => {}
        }

        Ok(Response::new(CancellationResponse {
            success: !cancelled_jobs.is_empty(),
            cancelled_job_ids: cancelled_jobs,
        }))
    }
}

impl CoordinatorServiceImpl {
    #[allow(dead_code)]
    fn worker_to_info(&self, worker: &WorkerState) -> WorkerInfo {
        let last_heartbeat = worker
            .last_heartbeat
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        WorkerInfo {
            worker_id: worker.worker_id.clone(),
            hostname: worker.hostname.clone(),
            status: Some(Box::new(WorkerStatus {
                state: worker.status.into(),
                active_jobs: worker.active_jobs.len() as u32,
                queued_jobs: 0,
            })),
            metrics: Some(Box::new(WorkerMetrics {
                cpu_usage: worker.metrics.cpu_usage,
                memory_usage: worker.metrics.memory_usage,
                gpu_usage: worker.metrics.gpu_usage,
                bytes_processed: worker.metrics.bytes_processed,
                frames_encoded: worker.metrics.frames_encoded,
            })),
            last_heartbeat_timestamp: last_heartbeat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinator_creation() {
        let config = CoordinatorConfig::default();
        let coordinator = Coordinator::new(config);
        assert_eq!(coordinator.workers.len(), 0);
        assert_eq!(coordinator.jobs.len(), 0);
    }

    #[test]
    fn test_worker_status_conversion() {
        assert_eq!(i32::from(worker_status::State::Idle), 0);
        assert_eq!(i32::from(worker_status::State::Busy), 1);
        assert_eq!(worker_status::State::from(0), worker_status::State::Idle);
    }

    #[test]
    fn test_load_balancing_strategies() {
        let config = CoordinatorConfig {
            load_balancing: LoadBalancingStrategy::LeastLoaded,
            ..Default::default()
        };
        assert_eq!(config.load_balancing, LoadBalancingStrategy::LeastLoaded);
    }
}
