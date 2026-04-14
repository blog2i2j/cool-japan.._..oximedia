//! Task scheduling and load balancing
//!
//! Provides various scheduling strategies:
//! - Round-robin distribution
//! - Least-loaded worker selection
//! - Capability-based routing
//! - Priority-based scheduling
//! - Deadline-aware scheduling

mod strategies;

use crate::{FarmError, Priority, Result, TaskId, WorkerId, WorkerState};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

pub use strategies::{LoadBalancer, SchedulingStrategy};

/// Task scheduling information
#[derive(Debug, Clone)]
pub struct SchedulableTask {
    pub task_id: TaskId,
    pub priority: Priority,
    pub required_capabilities: Vec<String>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub resource_requirements: ResourceRequirements,
}

/// Resource requirements for a task
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub gpu_required: bool,
    pub disk_space_mb: u64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_mb: 1024,
            gpu_required: false,
            disk_space_mb: 1024,
        }
    }
}

/// Worker information for scheduling
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    pub worker_id: WorkerId,
    pub state: WorkerState,
    pub capabilities: WorkerCapabilities,
    pub current_load: WorkerLoad,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
}

/// Worker capabilities
#[derive(Debug, Clone)]
pub struct WorkerCapabilities {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub supported_codecs: Vec<String>,
    pub supported_formats: Vec<String>,
    pub has_gpu: bool,
    pub gpu_count: u32,
    pub tags: HashMap<String, String>,
}

/// Current worker load
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    pub active_tasks: u32,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
}

impl WorkerLoad {
    /// Calculate overall load score (0.0 to 1.0)
    #[must_use]
    pub fn score(&self) -> f64 {
        (self.cpu_usage + self.memory_usage + self.disk_usage) / 3.0
    }
}

/// Task scheduler
pub struct Scheduler {
    #[allow(dead_code)]
    strategy: SchedulingStrategy,
    workers: Arc<RwLock<HashMap<WorkerId, WorkerInfo>>>,
    load_balancer: LoadBalancer,
}

impl Scheduler {
    /// Create a new scheduler with the given strategy
    #[must_use]
    pub fn new(strategy: SchedulingStrategy) -> Self {
        Self {
            strategy,
            workers: Arc::new(RwLock::new(HashMap::new())),
            load_balancer: LoadBalancer::new(strategy),
        }
    }

    /// Register a worker
    pub fn register_worker(&self, worker_info: WorkerInfo) -> Result<()> {
        let mut workers = self.workers.write();
        if workers.contains_key(&worker_info.worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {} already registered",
                worker_info.worker_id
            )));
        }
        workers.insert(worker_info.worker_id.clone(), worker_info);
        Ok(())
    }

    /// Unregister a worker
    pub fn unregister_worker(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();
        if workers.remove(worker_id).is_none() {
            return Err(FarmError::NotFound(format!("Worker {worker_id} not found")));
        }
        Ok(())
    }

    /// Update worker state
    pub fn update_worker_state(&self, worker_id: &WorkerId, state: WorkerState) -> Result<()> {
        let mut workers = self.workers.write();
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.state = state;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Update worker load
    pub fn update_worker_load(&self, worker_id: &WorkerId, load: WorkerLoad) -> Result<()> {
        let mut workers = self.workers.write();
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.current_load = load;
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Update worker heartbeat
    pub fn update_heartbeat(&self, worker_id: &WorkerId) -> Result<()> {
        let mut workers = self.workers.write();
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.last_heartbeat = chrono::Utc::now();
            Ok(())
        } else {
            Err(FarmError::NotFound(format!("Worker {worker_id} not found")))
        }
    }

    /// Select a worker for a task
    pub fn select_worker(&self, task: &SchedulableTask) -> Result<WorkerId> {
        let workers = self.workers.read();

        // Filter workers that can handle the task
        let eligible_workers: Vec<&WorkerInfo> = workers
            .values()
            .filter(|w| self.is_worker_eligible(w, task))
            .collect();

        if eligible_workers.is_empty() {
            return Err(FarmError::ResourceExhausted(
                "No eligible workers available".to_string(),
            ));
        }

        // Use load balancer to select the best worker
        let selected = self.load_balancer.select_worker(&eligible_workers, task)?;
        Ok(selected.worker_id.clone())
    }

    /// Check if a worker is eligible for a task
    fn is_worker_eligible(&self, worker: &WorkerInfo, task: &SchedulableTask) -> bool {
        // Check worker state
        if worker.state != WorkerState::Idle && worker.state != WorkerState::Busy {
            return false;
        }

        // Check if overloaded
        if worker.state == WorkerState::Overloaded {
            return false;
        }

        // Check capabilities
        if task.resource_requirements.gpu_required && !worker.capabilities.has_gpu {
            return false;
        }

        // Check if worker supports required codecs
        for required_codec in &task.required_capabilities {
            if !worker
                .capabilities
                .supported_codecs
                .contains(required_codec)
                && !worker
                    .capabilities
                    .supported_formats
                    .contains(required_codec)
            {
                return false;
            }
        }

        // Check resource availability
        if worker.capabilities.cpu_cores < task.resource_requirements.cpu_cores {
            return false;
        }

        if worker.capabilities.memory_mb < task.resource_requirements.memory_mb {
            return false;
        }

        true
    }

    /// Get all workers
    pub fn get_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read();
        workers.values().cloned().collect()
    }

    /// Get worker by ID
    pub fn get_worker(&self, worker_id: &WorkerId) -> Option<WorkerInfo> {
        let workers = self.workers.read();
        workers.get(worker_id).cloned()
    }

    /// Get active worker count
    pub fn active_worker_count(&self) -> usize {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state == WorkerState::Idle || w.state == WorkerState::Busy)
            .count()
    }

    /// Get workers by state
    pub fn get_workers_by_state(&self, state: WorkerState) -> Vec<WorkerInfo> {
        let workers = self.workers.read();
        workers
            .values()
            .filter(|w| w.state == state)
            .cloned()
            .collect()
    }

    /// Check for stale workers (no heartbeat within timeout)
    pub fn get_stale_workers(&self, timeout: chrono::Duration) -> Vec<WorkerId> {
        let workers = self.workers.read();
        let now = chrono::Utc::now();
        workers
            .values()
            .filter(|w| now - w.last_heartbeat > timeout)
            .map(|w| w.worker_id.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_worker(id: &str, cpu_cores: u32, has_gpu: bool) -> WorkerInfo {
        WorkerInfo {
            worker_id: WorkerId::new(id),
            state: WorkerState::Idle,
            capabilities: WorkerCapabilities {
                cpu_cores,
                memory_mb: 8192,
                supported_codecs: vec!["h264".to_string(), "h265".to_string()],
                supported_formats: vec!["mp4".to_string()],
                has_gpu,
                gpu_count: if has_gpu { 1 } else { 0 },
                tags: HashMap::new(),
            },
            current_load: WorkerLoad {
                active_tasks: 0,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                disk_usage: 0.0,
            },
            last_heartbeat: chrono::Utc::now(),
        }
    }

    fn create_test_task(gpu_required: bool) -> SchedulableTask {
        SchedulableTask {
            task_id: TaskId::new(),
            priority: Priority::Normal,
            required_capabilities: vec!["h264".to_string()],
            deadline: None,
            resource_requirements: ResourceRequirements {
                cpu_cores: 2,
                memory_mb: 2048,
                gpu_required,
                disk_space_mb: 1024,
            },
        }
    }

    #[test]
    fn test_scheduler_creation() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        assert_eq!(scheduler.active_worker_count(), 0);
    }

    #[test]
    fn test_worker_registration() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        let worker = create_test_worker("worker-1", 4, false);

        scheduler.register_worker(worker.clone()).unwrap();
        assert_eq!(scheduler.active_worker_count(), 1);

        let retrieved = scheduler.get_worker(&worker.worker_id).unwrap();
        assert_eq!(retrieved.worker_id, worker.worker_id);
    }

    #[test]
    fn test_duplicate_worker_registration() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        let worker = create_test_worker("worker-1", 4, false);

        scheduler.register_worker(worker.clone()).unwrap();
        let result = scheduler.register_worker(worker);
        assert!(result.is_err());
    }

    #[test]
    fn test_worker_unregistration() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        let worker = create_test_worker("worker-1", 4, false);

        scheduler.register_worker(worker.clone()).unwrap();
        scheduler.unregister_worker(&worker.worker_id).unwrap();
        assert_eq!(scheduler.active_worker_count(), 0);
    }

    #[test]
    fn test_worker_state_update() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        let worker = create_test_worker("worker-1", 4, false);

        scheduler.register_worker(worker.clone()).unwrap();
        scheduler
            .update_worker_state(&worker.worker_id, WorkerState::Busy)
            .unwrap();

        let retrieved = scheduler.get_worker(&worker.worker_id).unwrap();
        assert_eq!(retrieved.state, WorkerState::Busy);
    }

    #[test]
    fn test_worker_selection_cpu_only() {
        let scheduler = Scheduler::new(SchedulingStrategy::LeastLoaded);
        let worker1 = create_test_worker("worker-1", 4, false);
        let worker2 = create_test_worker("worker-2", 8, false);

        scheduler.register_worker(worker1).unwrap();
        scheduler.register_worker(worker2).unwrap();

        let task = create_test_task(false);
        let selected = scheduler.select_worker(&task).unwrap();
        assert!(!selected.as_str().is_empty());
    }

    #[test]
    fn test_worker_selection_gpu_required() {
        let scheduler = Scheduler::new(SchedulingStrategy::LeastLoaded);
        let worker1 = create_test_worker("worker-1", 4, false);
        let worker2 = create_test_worker("worker-2", 4, true);

        scheduler.register_worker(worker1).unwrap();
        scheduler.register_worker(worker2).unwrap();

        let task = create_test_task(true);
        let selected = scheduler.select_worker(&task).unwrap();
        assert_eq!(selected.as_str(), "worker-2");
    }

    #[test]
    fn test_no_eligible_workers() {
        let scheduler = Scheduler::new(SchedulingStrategy::LeastLoaded);
        let task = create_test_task(true);

        let result = scheduler.select_worker(&task);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_workers_by_state() {
        let scheduler = Scheduler::new(SchedulingStrategy::RoundRobin);
        let worker1 = create_test_worker("worker-1", 4, false);
        let worker2 = create_test_worker("worker-2", 4, false);

        scheduler.register_worker(worker1.clone()).unwrap();
        scheduler.register_worker(worker2.clone()).unwrap();
        scheduler
            .update_worker_state(&worker2.worker_id, WorkerState::Busy)
            .unwrap();

        let idle_workers = scheduler.get_workers_by_state(WorkerState::Idle);
        assert_eq!(idle_workers.len(), 1);

        let busy_workers = scheduler.get_workers_by_state(WorkerState::Busy);
        assert_eq!(busy_workers.len(), 1);
    }

    #[test]
    fn test_worker_load_score() {
        let load = WorkerLoad {
            active_tasks: 2,
            cpu_usage: 0.5,
            memory_usage: 0.6,
            disk_usage: 0.4,
        };

        let score = load.score();
        assert!((score - 0.5).abs() < 0.01);
    }
}
