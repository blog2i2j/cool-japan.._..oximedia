#![allow(dead_code)]
//! Worker pool management — worker state, utilization tracking, and job assignment.

use std::collections::HashMap;

/// The current state of a worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerState {
    /// Worker is ready and not processing any job.
    Idle,
    /// Worker is currently executing a job.
    Busy,
    /// Worker is draining (finishing current job, will not accept new ones).
    Draining,
    /// Worker has encountered a non-recoverable error.
    Error,
    /// Worker has been stopped.
    Stopped,
}

impl WorkerState {
    /// Returns true if the worker can accept a new job.
    pub fn is_available(&self) -> bool {
        matches!(self, WorkerState::Idle)
    }

    /// Returns true if the worker is active (Idle or Busy).
    pub fn is_active(&self) -> bool {
        matches!(self, WorkerState::Idle | WorkerState::Busy)
    }

    /// Short string label.
    pub fn label(&self) -> &'static str {
        match self {
            WorkerState::Idle => "idle",
            WorkerState::Busy => "busy",
            WorkerState::Draining => "draining",
            WorkerState::Error => "error",
            WorkerState::Stopped => "stopped",
        }
    }
}

/// Represents a single worker in the pool.
#[derive(Debug, Clone)]
pub struct Worker {
    /// Unique worker identifier.
    pub id: String,
    /// Current operational state.
    pub state: WorkerState,
    /// Total number of jobs this worker has completed.
    pub jobs_completed: u64,
    /// Total number of jobs this worker has failed.
    pub jobs_failed: u64,
    /// ID of the job currently being processed (if any).
    pub current_job: Option<String>,
    /// Maximum number of concurrent job slots for this worker.
    pub capacity: u32,
    /// Number of job slots currently in use.
    pub active_slots: u32,
    /// Worker tags for affinity scheduling.
    pub tags: Vec<String>,
}

impl Worker {
    /// Create a new idle worker with the given capacity.
    pub fn new(id: impl Into<String>, capacity: u32) -> Self {
        Self {
            id: id.into(),
            state: WorkerState::Idle,
            jobs_completed: 0,
            jobs_failed: 0,
            current_job: None,
            capacity,
            active_slots: 0,
            tags: Vec::new(),
        }
    }

    /// Add a scheduling tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Utilization as a fraction of capacity [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.active_slots as f64 / self.capacity as f64
    }

    /// Returns true if this worker can accept another job.
    pub fn can_accept(&self) -> bool {
        self.state.is_available() && self.active_slots < self.capacity
    }

    /// Mark the worker as starting a job.
    pub fn assign_job(&mut self, job_id: impl Into<String>) -> bool {
        if !self.can_accept() {
            return false;
        }
        self.current_job = Some(job_id.into());
        self.active_slots += 1;
        if self.active_slots >= self.capacity {
            self.state = WorkerState::Busy;
        }
        true
    }

    /// Mark the worker as having finished a job (success or failure).
    pub fn complete_job(&mut self, success: bool) {
        if success {
            self.jobs_completed += 1;
        } else {
            self.jobs_failed += 1;
        }
        if self.active_slots > 0 {
            self.active_slots -= 1;
        }
        self.current_job = None;
        if self.active_slots == 0 && self.state != WorkerState::Draining {
            self.state = WorkerState::Idle;
        }
    }

    /// Total jobs attempted (completed + failed).
    pub fn total_jobs(&self) -> u64 {
        self.jobs_completed + self.jobs_failed
    }

    /// Success rate [0.0, 1.0]. Returns 0.0 if no jobs attempted.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_jobs();
        if total == 0 {
            return 0.0;
        }
        self.jobs_completed as f64 / total as f64
    }
}

/// A pool that manages multiple workers.
#[derive(Debug, Default)]
pub struct WorkerPool {
    workers: HashMap<String, Worker>,
}

impl WorkerPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a worker to the pool.
    pub fn add_worker(&mut self, worker: Worker) {
        self.workers.insert(worker.id.clone(), worker);
    }

    /// Remove a worker by ID. Returns the removed worker or None.
    pub fn remove_worker(&mut self, id: &str) -> Option<Worker> {
        self.workers.remove(id)
    }

    /// Total number of workers in the pool.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of workers that are currently available (idle and with free slots).
    pub fn available_count(&self) -> usize {
        self.workers.values().filter(|w| w.can_accept()).count()
    }

    /// Number of workers that are active (Idle or Busy).
    pub fn active_count(&self) -> usize {
        self.workers
            .values()
            .filter(|w| w.state.is_active())
            .count()
    }

    /// Assign a job to the least-loaded available worker.
    /// Returns the worker ID that accepted the job, or None if no worker is available.
    pub fn assign_job(&mut self, job_id: impl Into<String>) -> Option<String> {
        let job_id = job_id.into();

        // Find the available worker with the lowest utilization
        let best_id = self
            .workers
            .values()
            .filter(|w| w.can_accept())
            .min_by(|a, b| {
                a.utilization()
                    .partial_cmp(&b.utilization())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|w| w.id.clone());

        if let Some(id) = best_id {
            if let Some(worker) = self.workers.get_mut(&id) {
                worker.assign_job(&job_id);
                return Some(id);
            }
        }
        None
    }

    /// Signal that a worker has finished a job.
    pub fn complete_job(&mut self, worker_id: &str, success: bool) -> bool {
        if let Some(w) = self.workers.get_mut(worker_id) {
            w.complete_job(success);
            true
        } else {
            false
        }
    }

    /// Get immutable reference to a worker.
    pub fn get_worker(&self, id: &str) -> Option<&Worker> {
        self.workers.get(id)
    }

    /// Average utilization across all workers [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_utilization(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.workers.values().map(|w| w.utilization()).sum();
        sum / self.workers.len() as f64
    }

    /// Total jobs completed across all workers.
    pub fn total_completed(&self) -> u64 {
        self.workers.values().map(|w| w.jobs_completed).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_state_is_available() {
        assert!(WorkerState::Idle.is_available());
        assert!(!WorkerState::Busy.is_available());
        assert!(!WorkerState::Error.is_available());
        assert!(!WorkerState::Stopped.is_available());
        assert!(!WorkerState::Draining.is_available());
    }

    #[test]
    fn test_worker_state_is_active() {
        assert!(WorkerState::Idle.is_active());
        assert!(WorkerState::Busy.is_active());
        assert!(!WorkerState::Error.is_active());
        assert!(!WorkerState::Stopped.is_active());
    }

    #[test]
    fn test_worker_state_label() {
        assert_eq!(WorkerState::Idle.label(), "idle");
        assert_eq!(WorkerState::Busy.label(), "busy");
        assert_eq!(WorkerState::Draining.label(), "draining");
    }

    #[test]
    fn test_worker_utilization_empty() {
        let w = Worker::new("w0", 0);
        assert_eq!(w.utilization(), 0.0);
    }

    #[test]
    fn test_worker_utilization() {
        let mut w = Worker::new("w1", 4);
        w.active_slots = 2;
        assert!((w.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_worker_assign_job() {
        let mut w = Worker::new("w2", 2);
        assert!(w.assign_job("job-1"));
        assert_eq!(w.active_slots, 1);
        assert_eq!(w.state, WorkerState::Idle); // still has free slot
        assert!(w.assign_job("job-2"));
        assert_eq!(w.active_slots, 2);
        assert_eq!(w.state, WorkerState::Busy); // capacity reached
                                                // No more slots
        assert!(!w.assign_job("job-3"));
    }

    #[test]
    fn test_worker_complete_job_success() {
        let mut w = Worker::new("w3", 1);
        w.assign_job("job-1");
        w.complete_job(true);
        assert_eq!(w.jobs_completed, 1);
        assert_eq!(w.jobs_failed, 0);
        assert_eq!(w.active_slots, 0);
        assert_eq!(w.state, WorkerState::Idle);
    }

    #[test]
    fn test_worker_complete_job_failure() {
        let mut w = Worker::new("w4", 1);
        w.assign_job("job-1");
        w.complete_job(false);
        assert_eq!(w.jobs_completed, 0);
        assert_eq!(w.jobs_failed, 1);
    }

    #[test]
    fn test_worker_success_rate() {
        let mut w = Worker::new("w5", 4);
        w.jobs_completed = 3;
        w.jobs_failed = 1;
        assert!((w.success_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_pool_add_and_count() {
        let mut pool = WorkerPool::new();
        pool.add_worker(Worker::new("w1", 2));
        pool.add_worker(Worker::new("w2", 2));
        assert_eq!(pool.worker_count(), 2);
    }

    #[test]
    fn test_pool_available_count() {
        let mut pool = WorkerPool::new();
        pool.add_worker(Worker::new("w1", 1));
        pool.add_worker(Worker::new("w2", 1));
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_pool_assign_job() {
        let mut pool = WorkerPool::new();
        pool.add_worker(Worker::new("w1", 1));
        let assigned = pool.assign_job("job-1");
        assert!(assigned.is_some());
        // Worker is now busy (capacity=1, slot used)
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_pool_assign_no_available() {
        let mut pool = WorkerPool::new();
        // Add a worker with no capacity
        let mut w = Worker::new("w1", 1);
        w.state = WorkerState::Busy;
        w.active_slots = 1;
        pool.add_worker(w);
        let assigned = pool.assign_job("job-x");
        assert!(assigned.is_none());
    }

    #[test]
    fn test_pool_complete_job() {
        let mut pool = WorkerPool::new();
        pool.add_worker(Worker::new("w1", 1));
        pool.assign_job("j1");
        pool.complete_job("w1", true);
        assert_eq!(pool.total_completed(), 1);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_pool_avg_utilization() {
        let pool = WorkerPool::new();
        assert_eq!(pool.avg_utilization(), 0.0);
    }
}
