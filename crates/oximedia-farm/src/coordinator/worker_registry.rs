//! Worker registry and management

use crate::{FarmError, Result, WorkerId, WorkerState};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::Duration;

/// Worker registration information
#[derive(Debug, Clone)]
pub struct WorkerRegistration {
    pub worker_id: WorkerId,
    pub hostname: String,
    pub capabilities: WorkerCapabilities,
    pub metadata: HashMap<String, String>,
    pub state: WorkerState,
    pub registered_at: DateTime<Utc>,
    pub last_heartbeat: DateTime<Utc>,
    pub active_tasks: u32,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
}

/// Worker capabilities
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorkerCapabilities {
    pub cpu_cores: u32,
    pub memory_bytes: u64,
    pub supported_codecs: Vec<String>,
    pub supported_formats: Vec<String>,
    pub has_gpu: bool,
    pub gpus: Vec<GpuInfo>,
    pub max_concurrent_tasks: u32,
    pub tags: HashMap<String, String>,
}

/// GPU information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub memory_bytes: u64,
    pub vendor: String,
    pub supported_codecs: Vec<String>,
}

/// Worker status update
#[derive(Debug, Clone)]
pub struct WorkerStatusUpdate {
    pub cpu_usage: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub disk_free: u64,
    pub active_tasks: u32,
    pub state: WorkerState,
}

/// Worker registry manages all connected workers
pub struct WorkerRegistry {
    workers: RwLock<HashMap<WorkerId, WorkerRegistration>>,
    heartbeat_timeout: Duration,
}

impl WorkerRegistry {
    /// Create a new worker registry
    #[must_use]
    pub fn new(heartbeat_timeout: Duration) -> Self {
        Self {
            workers: RwLock::new(HashMap::new()),
            heartbeat_timeout,
        }
    }

    /// Register a new worker
    pub async fn register_worker(
        &self,
        worker_id: WorkerId,
        hostname: String,
        capabilities: WorkerCapabilities,
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        let mut workers = self.workers.write();

        if workers.contains_key(&worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {worker_id} already registered"
            )));
        }

        let registration = WorkerRegistration {
            worker_id: worker_id.clone(),
            hostname,
            capabilities,
            metadata,
            state: WorkerState::Idle,
            registered_at: Utc::now(),
            last_heartbeat: Utc::now(),
            active_tasks: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
        };

        workers.insert(worker_id.clone(), registration);
        tracing::info!("Worker {} registered successfully", worker_id);
        Ok(())
    }

    /// Unregister a worker
    pub async fn unregister_worker(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if workers.remove(worker_id).is_none() {
            return Err(FarmError::NotFound(format!("Worker {worker_id} not found")));
        }

        tracing::info!("Worker {} unregistered", worker_id);
        Ok(())
    }

    /// Update worker heartbeat
    pub async fn heartbeat(&self, worker_id: &WorkerId, status: WorkerStatusUpdate) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.last_heartbeat = Utc::now();
            worker.state = status.state;
            worker.active_tasks = status.active_tasks;
            tracing::debug!("Heartbeat received from worker {}", worker_id);
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Mark worker as offline
    pub async fn mark_offline(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.state = WorkerState::Offline;
            tracing::warn!("Worker {} marked as offline", worker_id);
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Get worker by ID
    pub fn get_worker(&self, worker_id: &WorkerId) -> Option<WorkerRegistration> {
        let workers = self.workers.read();
        workers.get(worker_id).cloned()
    }

    /// List all workers
    pub fn list_workers(&self) -> Vec<WorkerRegistration> {
        let workers = self.workers.read();
        workers.values().cloned().collect()
    }

    /// List workers by state
    pub fn list_workers_by_state(&self, state: WorkerState) -> Vec<WorkerRegistration> {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state == state)
            .cloned()
            .collect()
    }

    /// Get active worker count
    pub fn active_worker_count(&self) -> usize {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state != WorkerState::Offline)
            .count()
    }

    /// Increment task completed count
    pub async fn increment_task_completed(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.total_tasks_completed += 1;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Increment task failed count
    pub async fn increment_task_failed(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.total_tasks_failed += 1;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Get workers that haven't sent heartbeat within timeout
    pub fn get_stale_workers(&self) -> Vec<WorkerId> {
        let workers = self.workers.read();
        let now = Utc::now();
        // chrono::Duration::from_std fails only if the std Duration exceeds
        // chrono's i64-nanosecond range (~292 years). heartbeat_timeout is
        // always a small operational duration, so fall back to empty list on
        // overflow rather than panicking.
        let timeout = match chrono::Duration::from_std(self.heartbeat_timeout) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        workers
            .values()
            .filter(|w| w.state != WorkerState::Offline && (now - w.last_heartbeat) > timeout)
            .map(|w| w.worker_id.clone())
            .collect()
    }

    /// Get worker statistics
    pub fn get_statistics(&self) -> WorkerStatistics {
        let workers = self.workers.read();

        let total = workers.len();
        let idle = workers
            .values()
            .filter(|w| w.state == WorkerState::Idle)
            .count();
        let busy = workers
            .values()
            .filter(|w| w.state == WorkerState::Busy)
            .count();
        let overloaded = workers
            .values()
            .filter(|w| w.state == WorkerState::Overloaded)
            .count();
        let draining = workers
            .values()
            .filter(|w| w.state == WorkerState::Draining)
            .count();
        let offline = workers
            .values()
            .filter(|w| w.state == WorkerState::Offline)
            .count();

        let total_tasks_completed: u64 = workers.values().map(|w| w.total_tasks_completed).sum();
        let total_tasks_failed: u64 = workers.values().map(|w| w.total_tasks_failed).sum();

        WorkerStatistics {
            total,
            idle,
            busy,
            overloaded,
            draining,
            offline,
            total_tasks_completed,
            total_tasks_failed,
        }
    }
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStatistics {
    pub total: usize,
    pub idle: usize,
    pub busy: usize,
    pub overloaded: usize,
    pub draining: usize,
    pub offline: usize,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_capabilities() -> WorkerCapabilities {
        WorkerCapabilities {
            cpu_cores: 8,
            memory_bytes: 16 * 1024 * 1024 * 1024,
            supported_codecs: vec!["h264".to_string(), "h265".to_string()],
            supported_formats: vec!["mp4".to_string(), "mkv".to_string()],
            has_gpu: false,
            gpus: vec![],
            max_concurrent_tasks: 4,
            tags: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_worker_registration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        assert_eq!(registry.active_worker_count(), 1);
        let worker = registry
            .get_worker(&worker_id)
            .expect("get_worker should succeed");
        assert_eq!(worker.hostname, "host1");
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        let result = registry
            .register_worker(
                worker_id,
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_worker_unregistration() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        registry
            .unregister_worker(&worker_id)
            .await
            .expect("await should be valid");
        assert_eq!(registry.active_worker_count(), 0);
    }

    #[tokio::test]
    async fn test_worker_heartbeat() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        let status = WorkerStatusUpdate {
            cpu_usage: 0.5,
            memory_used: 8 * 1024 * 1024 * 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            disk_free: 100 * 1024 * 1024 * 1024,
            active_tasks: 2,
            state: WorkerState::Busy,
        };

        registry
            .heartbeat(&worker_id, status)
            .await
            .expect("await should be valid");

        let worker = registry
            .get_worker(&worker_id)
            .expect("get_worker should succeed");
        assert_eq!(worker.state, WorkerState::Busy);
        assert_eq!(worker.active_tasks, 2);
    }

    #[tokio::test]
    async fn test_list_workers_by_state() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));

        let worker1 = WorkerId::new("worker-1");
        let worker2 = WorkerId::new("worker-2");

        registry
            .register_worker(
                worker1.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        registry
            .register_worker(
                worker2.clone(),
                "host2".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        let status = WorkerStatusUpdate {
            cpu_usage: 0.5,
            memory_used: 8 * 1024 * 1024 * 1024,
            memory_total: 16 * 1024 * 1024 * 1024,
            disk_free: 100 * 1024 * 1024 * 1024,
            active_tasks: 2,
            state: WorkerState::Busy,
        };

        registry
            .heartbeat(&worker2, status)
            .await
            .expect("await should be valid");

        let idle_workers = registry.list_workers_by_state(WorkerState::Idle);
        assert_eq!(idle_workers.len(), 1);

        let busy_workers = registry.list_workers_by_state(WorkerState::Busy);
        assert_eq!(busy_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_task_counters() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));
        let worker_id = WorkerId::new("worker-1");

        registry
            .register_worker(
                worker_id.clone(),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        registry
            .increment_task_completed(&worker_id)
            .await
            .expect("await should be valid");
        registry
            .increment_task_completed(&worker_id)
            .await
            .expect("await should be valid");
        registry
            .increment_task_failed(&worker_id)
            .await
            .expect("await should be valid");

        let worker = registry
            .get_worker(&worker_id)
            .expect("get_worker should succeed");
        assert_eq!(worker.total_tasks_completed, 2);
        assert_eq!(worker.total_tasks_failed, 1);
    }

    #[tokio::test]
    async fn test_statistics() {
        let registry = WorkerRegistry::new(Duration::from_secs(60));

        registry
            .register_worker(
                WorkerId::new("worker-1"),
                "host1".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        registry
            .register_worker(
                WorkerId::new("worker-2"),
                "host2".to_string(),
                create_test_capabilities(),
                HashMap::new(),
            )
            .await
            .expect("operation should succeed");

        let stats = registry.get_statistics();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.idle, 2);
        assert_eq!(stats.busy, 0);
    }
}
