//! gRPC service implementation

use crate::coordinator::worker_registry::{WorkerCapabilities, WorkerStatusUpdate};
use crate::coordinator::{JobQueue, WorkerRegistry};
use crate::pb::farm_coordinator_server::FarmCoordinator;
#[cfg(test)]
use crate::pb::WorkerStatus;
use crate::pb::{
    GetTaskRequest, GetTaskResponse, HeartbeatRequest, HeartbeatResponse, ProgressReport,
    ProgressResponse, RegisterWorkerRequest, RegisterWorkerResponse, TaskCompletion,
    TaskCompletionResponse, TaskFailure, TaskFailureResponse, UnregisterWorkerRequest,
    UnregisterWorkerResponse,
};
use crate::scheduler::Scheduler;
use crate::WorkerId;
use std::sync::Arc;
use tonic::{Request, Response, Status};
use tracing::{error, info};

/// Farm coordinator gRPC service implementation
pub struct FarmCoordinatorService {
    #[allow(dead_code)]
    job_queue: Arc<JobQueue>,
    worker_registry: Arc<WorkerRegistry>,
    #[allow(dead_code)]
    scheduler: Arc<Scheduler>,
}

impl FarmCoordinatorService {
    /// Create a new farm coordinator service
    pub fn new(
        job_queue: Arc<JobQueue>,
        worker_registry: Arc<WorkerRegistry>,
        scheduler: Arc<Scheduler>,
    ) -> Self {
        Self {
            job_queue,
            worker_registry,
            scheduler,
        }
    }
}

#[tonic::async_trait]
impl FarmCoordinator for FarmCoordinatorService {
    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::new(req.worker_id);

        info!("Registering worker: {}", worker_id);

        // Convert capabilities
        let capabilities = req
            .capabilities
            .ok_or_else(|| Status::invalid_argument("Missing worker capabilities"))?;

        let worker_caps = WorkerCapabilities {
            cpu_cores: capabilities.cpu_cores,
            memory_bytes: capabilities.memory_bytes,
            supported_codecs: capabilities.supported_codecs,
            supported_formats: capabilities.supported_formats,
            has_gpu: capabilities.has_gpu,
            gpus: capabilities
                .gpus
                .into_iter()
                .map(|g| crate::coordinator::worker_registry::GpuInfo {
                    name: g.name,
                    memory_bytes: g.memory_bytes,
                    vendor: g.vendor,
                    supported_codecs: g.supported_codecs,
                })
                .collect(),
            max_concurrent_tasks: capabilities.max_concurrent_tasks,
            tags: capabilities.tags,
        };

        // Register worker
        match self
            .worker_registry
            .register_worker(worker_id, req.hostname, worker_caps, req.metadata)
            .await
        {
            Ok(()) => Ok(Response::new(RegisterWorkerResponse {
                success: true,
                message: "Worker registered successfully".to_string(),
                heartbeat_interval_secs: 30,
            })),
            Err(e) => {
                error!("Failed to register worker: {}", e);
                Err(Status::internal(format!("Registration failed: {e}")))
            }
        }
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::new(req.worker_id);

        let status = req
            .status
            .ok_or_else(|| Status::invalid_argument("Missing worker status"))?;

        let state = crate::communication::pb_to_worker_state(status.state);

        let status_update = WorkerStatusUpdate {
            cpu_usage: status.cpu_usage,
            memory_used: status.memory_used,
            memory_total: status.memory_total,
            disk_free: status.disk_free,
            active_tasks: status.active_tasks,
            state,
        };

        match self
            .worker_registry
            .heartbeat(&worker_id, status_update)
            .await
        {
            Ok(()) => Ok(Response::new(HeartbeatResponse {
                success: true,
                tasks_to_cancel: vec![],
                should_shutdown: false,
            })),
            Err(e) => {
                error!("Heartbeat failed for worker {}: {}", worker_id, e);
                Err(Status::internal(format!("Heartbeat failed: {e}")))
            }
        }
    }

    async fn get_task(
        &self,
        request: Request<GetTaskRequest>,
    ) -> Result<Response<GetTaskResponse>, Status> {
        let req = request.into_inner();
        let _worker_id = WorkerId::new(req.worker_id);
        let _max_tasks = req.max_tasks;

        // In a real implementation, this would fetch tasks from the job queue
        // For now, return empty response
        Ok(Response::new(GetTaskResponse { tasks: vec![] }))
    }

    async fn report_progress(
        &self,
        request: Request<ProgressReport>,
    ) -> Result<Response<ProgressResponse>, Status> {
        let req = request.into_inner();
        let _worker_id = WorkerId::new(req.worker_id);

        // In a real implementation, this would update task progress
        Ok(Response::new(ProgressResponse { success: true }))
    }

    async fn complete_task(
        &self,
        request: Request<TaskCompletion>,
    ) -> Result<Response<TaskCompletionResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::new(req.worker_id);

        // Increment worker's completed task counter
        if let Err(e) = self
            .worker_registry
            .increment_task_completed(&worker_id)
            .await
        {
            error!("Failed to increment task counter: {}", e);
        }

        Ok(Response::new(TaskCompletionResponse {
            success: true,
            message: "Task completed".to_string(),
        }))
    }

    async fn fail_task(
        &self,
        request: Request<TaskFailure>,
    ) -> Result<Response<TaskFailureResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::new(req.worker_id);

        // Increment worker's failed task counter
        if let Err(e) = self.worker_registry.increment_task_failed(&worker_id).await {
            error!("Failed to increment failed counter: {}", e);
        }

        Ok(Response::new(TaskFailureResponse { success: true }))
    }

    async fn unregister_worker(
        &self,
        request: Request<UnregisterWorkerRequest>,
    ) -> Result<Response<UnregisterWorkerResponse>, Status> {
        let req = request.into_inner();
        let worker_id = WorkerId::new(req.worker_id);

        info!(
            "Unregistering worker: {} (reason: {})",
            worker_id, req.reason
        );

        match self.worker_registry.unregister_worker(&worker_id).await {
            Ok(()) => Ok(Response::new(UnregisterWorkerResponse { success: true })),
            Err(e) => {
                error!("Failed to unregister worker: {}", e);
                Err(Status::internal(format!("Unregistration failed: {e}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::JobQueue;
    use crate::pb;
    use crate::persistence::Database;
    use std::collections::HashMap;

    fn create_test_service() -> FarmCoordinatorService {
        let db = Arc::new(Database::in_memory().unwrap());
        let job_queue = Arc::new(JobQueue::new(db, 100, 100));
        let worker_registry = Arc::new(WorkerRegistry::new(std::time::Duration::from_secs(60)));
        let scheduler = Arc::new(Scheduler::new(
            crate::scheduler::SchedulingStrategy::RoundRobin,
        ));

        FarmCoordinatorService::new(job_queue, worker_registry, scheduler)
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let service = create_test_service();

        let request = Request::new(RegisterWorkerRequest {
            worker_id: "test-worker".to_string(),
            hostname: "test-host".to_string(),
            capabilities: Some(pb::WorkerCapabilities {
                cpu_cores: 4,
                memory_bytes: 8 * 1024 * 1024 * 1024,
                supported_codecs: vec!["h264".to_string()],
                supported_formats: vec!["mp4".to_string()],
                has_gpu: false,
                gpus: vec![],
                max_concurrent_tasks: 2,
                tags: HashMap::new(),
            }),
            metadata: HashMap::new(),
        });

        let response = service.register_worker(request).await.unwrap();
        assert!(response.into_inner().success);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let service = create_test_service();

        // First register the worker
        let reg_request = Request::new(RegisterWorkerRequest {
            worker_id: "test-worker".to_string(),
            hostname: "test-host".to_string(),
            capabilities: Some(pb::WorkerCapabilities {
                cpu_cores: 4,
                memory_bytes: 8 * 1024 * 1024 * 1024,
                supported_codecs: vec!["h264".to_string()],
                supported_formats: vec!["mp4".to_string()],
                has_gpu: false,
                gpus: vec![],
                max_concurrent_tasks: 2,
                tags: HashMap::new(),
            }),
            metadata: HashMap::new(),
        });

        service.register_worker(reg_request).await.unwrap();

        // Send heartbeat
        let hb_request = Request::new(HeartbeatRequest {
            worker_id: "test-worker".to_string(),
            status: Some(WorkerStatus {
                cpu_usage: 0.5,
                memory_used: 4 * 1024 * 1024 * 1024,
                memory_total: 8 * 1024 * 1024 * 1024,
                disk_free: 100 * 1024 * 1024 * 1024,
                active_tasks: 1,
                state: crate::pb::WorkerState::Busy as i32,
            }),
            active_task_ids: vec![],
        });

        let response = service.heartbeat(hb_request).await.unwrap();
        assert!(response.into_inner().success);
    }

    #[tokio::test]
    async fn test_unregister_worker() {
        let service = create_test_service();

        // First register
        let reg_request = Request::new(RegisterWorkerRequest {
            worker_id: "test-worker".to_string(),
            hostname: "test-host".to_string(),
            capabilities: Some(pb::WorkerCapabilities {
                cpu_cores: 4,
                memory_bytes: 8 * 1024 * 1024 * 1024,
                supported_codecs: vec!["h264".to_string()],
                supported_formats: vec!["mp4".to_string()],
                has_gpu: false,
                gpus: vec![],
                max_concurrent_tasks: 2,
                tags: HashMap::new(),
            }),
            metadata: HashMap::new(),
        });

        service.register_worker(reg_request).await.unwrap();

        // Unregister
        let unreg_request = Request::new(UnregisterWorkerRequest {
            worker_id: "test-worker".to_string(),
            reason: "test shutdown".to_string(),
        });

        let response = service.unregister_worker(unreg_request).await.unwrap();
        assert!(response.into_inner().success);
    }
}
