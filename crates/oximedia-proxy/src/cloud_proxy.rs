#![allow(dead_code)]
//! Cloud proxy generation — offload proxy transcoding to remote workers.
//!
//! This module provides a cloud-based proxy generation system that distributes
//! transcode jobs to remote worker nodes over a logical network. Jobs are queued,
//! dispatched to available workers based on capacity and region affinity, and
//! results are collected asynchronously.
//!
//! The implementation is purely in-memory and does not perform real network I/O;
//! it models the scheduling, routing, and result-collection aspects that a real
//! cloud proxy farm would need.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Worker region and health
// ---------------------------------------------------------------------------

/// Geographic region for a cloud worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CloudRegion {
    /// US East (Virginia).
    UsEast,
    /// US West (Oregon).
    UsWest,
    /// EU West (Ireland).
    EuWest,
    /// AP Southeast (Singapore).
    ApSoutheast,
    /// Custom / unspecified.
    Other,
}

impl CloudRegion {
    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::UsEast => "us-east-1",
            Self::UsWest => "us-west-2",
            Self::EuWest => "eu-west-1",
            Self::ApSoutheast => "ap-southeast-1",
            Self::Other => "other",
        }
    }
}

/// Health status of a remote worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    /// Worker is healthy and accepting jobs.
    Healthy,
    /// Worker is degraded (slow responses, partial failures).
    Degraded,
    /// Worker is unreachable or down.
    Unreachable,
    /// Worker is draining (will not accept new work).
    Draining,
}

// ---------------------------------------------------------------------------
// Cloud worker
// ---------------------------------------------------------------------------

/// A remote worker node capable of transcoding proxies.
#[derive(Debug, Clone)]
pub struct CloudWorker {
    /// Unique worker identifier.
    pub id: String,
    /// Region where the worker is deployed.
    pub region: CloudRegion,
    /// Maximum concurrent jobs this worker can handle.
    pub capacity: u32,
    /// Number of jobs currently assigned.
    pub active_jobs: u32,
    /// Cumulative jobs completed.
    pub completed_jobs: u64,
    /// Current health status.
    pub health: WorkerHealth,
    /// Estimated transcode speed factor (1.0 = real-time, 2.0 = 2x, etc.).
    pub speed_factor: f64,
}

impl CloudWorker {
    /// Create a new healthy worker.
    pub fn new(id: impl Into<String>, region: CloudRegion, capacity: u32) -> Self {
        Self {
            id: id.into(),
            region,
            capacity,
            active_jobs: 0,
            completed_jobs: 0,
            health: WorkerHealth::Healthy,
            speed_factor: 1.0,
        }
    }

    /// Set the speed factor.
    pub fn with_speed(mut self, factor: f64) -> Self {
        self.speed_factor = factor;
        self
    }

    /// Whether this worker can accept another job.
    pub fn has_capacity(&self) -> bool {
        self.health == WorkerHealth::Healthy && self.active_jobs < self.capacity
    }

    /// Remaining capacity (slots available).
    pub fn remaining_capacity(&self) -> u32 {
        if self.health != WorkerHealth::Healthy {
            return 0;
        }
        self.capacity.saturating_sub(self.active_jobs)
    }

    /// Assign a job (increment active count).
    pub fn assign_job(&mut self) -> bool {
        if !self.has_capacity() {
            return false;
        }
        self.active_jobs += 1;
        true
    }

    /// Complete a job (decrement active count, increment completed).
    pub fn complete_job(&mut self) {
        if self.active_jobs > 0 {
            self.active_jobs -= 1;
        }
        self.completed_jobs += 1;
    }

    /// Utilization ratio (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.active_jobs as f64 / self.capacity as f64
    }
}

// ---------------------------------------------------------------------------
// Cloud job
// ---------------------------------------------------------------------------

/// State of a cloud proxy job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloudJobState {
    /// Waiting to be dispatched.
    Pending,
    /// Dispatched to a remote worker.
    Dispatched,
    /// Transcode in progress on the remote worker.
    Transcoding,
    /// Upload of the resulting proxy is in progress.
    Uploading,
    /// Job completed successfully.
    Completed,
    /// Job failed.
    Failed,
}

/// A cloud proxy generation job.
#[derive(Debug, Clone)]
pub struct CloudProxyJob {
    /// Unique job identifier.
    pub id: String,
    /// Source media path or URI.
    pub source: String,
    /// Target codec (e.g. "h264", "vp9").
    pub codec: String,
    /// Target resolution (width, height).
    pub resolution: (u32, u32),
    /// Target bitrate in kbps.
    pub bitrate_kbps: u32,
    /// Preferred region for processing.
    pub preferred_region: Option<CloudRegion>,
    /// Current state.
    pub state: CloudJobState,
    /// Assigned worker ID (once dispatched).
    pub assigned_worker: Option<String>,
    /// Output path (once completed).
    pub output_path: Option<String>,
    /// Error message (if failed).
    pub error: Option<String>,
}

impl CloudProxyJob {
    /// Create a new pending cloud job.
    pub fn new(
        id: impl Into<String>,
        source: impl Into<String>,
        codec: impl Into<String>,
        resolution: (u32, u32),
        bitrate_kbps: u32,
    ) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            codec: codec.into(),
            resolution,
            bitrate_kbps,
            preferred_region: None,
            state: CloudJobState::Pending,
            assigned_worker: None,
            output_path: None,
            error: None,
        }
    }

    /// Set preferred region.
    pub fn with_region(mut self, region: CloudRegion) -> Self {
        self.preferred_region = Some(region);
        self
    }

    /// Whether the job is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, CloudJobState::Completed | CloudJobState::Failed)
    }
}

// ---------------------------------------------------------------------------
// Dispatch strategy
// ---------------------------------------------------------------------------

/// Strategy for selecting a worker to dispatch a job to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchStrategy {
    /// Pick the worker with the most remaining capacity.
    LeastLoaded,
    /// Pick the fastest worker (highest speed_factor).
    Fastest,
    /// Prefer a worker in the job's preferred region, then least-loaded.
    RegionAffinity,
}

// ---------------------------------------------------------------------------
// Cloud proxy manager
// ---------------------------------------------------------------------------

/// Manages cloud proxy generation: workers, jobs, dispatching, and results.
pub struct CloudProxyManager {
    /// Registered workers keyed by ID.
    workers: HashMap<String, CloudWorker>,
    /// All jobs keyed by ID.
    jobs: HashMap<String, CloudProxyJob>,
    /// Pending job queue (FIFO for dispatch).
    pending_queue: VecDeque<String>,
    /// Dispatch strategy.
    strategy: DispatchStrategy,
    /// Total jobs dispatched.
    total_dispatched: u64,
    /// Total jobs completed.
    total_completed: u64,
    /// Total jobs failed.
    total_failed: u64,
}

impl CloudProxyManager {
    /// Create a new manager with the given dispatch strategy.
    pub fn new(strategy: DispatchStrategy) -> Self {
        Self {
            workers: HashMap::new(),
            jobs: HashMap::new(),
            pending_queue: VecDeque::new(),
            strategy,
            total_dispatched: 0,
            total_completed: 0,
            total_failed: 0,
        }
    }

    /// Register a remote worker.
    pub fn register_worker(&mut self, worker: CloudWorker) {
        self.workers.insert(worker.id.clone(), worker);
    }

    /// Remove a worker (only if it has no active jobs).
    pub fn remove_worker(&mut self, id: &str) -> bool {
        if let Some(w) = self.workers.get(id) {
            if w.active_jobs == 0 {
                self.workers.remove(id);
                return true;
            }
        }
        false
    }

    /// Submit a job for cloud processing.
    pub fn submit_job(&mut self, job: CloudProxyJob) -> String {
        let id = job.id.clone();
        self.pending_queue.push_back(id.clone());
        self.jobs.insert(id.clone(), job);
        id
    }

    /// Dispatch pending jobs to available workers.
    ///
    /// Returns the number of jobs dispatched.
    pub fn dispatch(&mut self) -> usize {
        let mut dispatched = 0;

        // Collect pending job IDs
        let pending: Vec<String> = self.pending_queue.drain(..).collect();
        let mut remaining = VecDeque::new();

        for job_id in pending {
            let preferred = self
                .jobs
                .get(&job_id)
                .and_then(|j| {
                    if j.state == CloudJobState::Pending {
                        Some(j.preferred_region)
                    } else {
                        None
                    }
                });

            let preferred = match preferred {
                Some(p) => p,
                None => continue,
            };

            if let Some(worker_id) = self.select_worker(preferred) {
                // Apply immediately so capacity is updated for the next iteration
                if let Some(worker) = self.workers.get_mut(&worker_id) {
                    worker.assign_job();
                }
                if let Some(job) = self.jobs.get_mut(&job_id) {
                    job.state = CloudJobState::Dispatched;
                    job.assigned_worker = Some(worker_id);
                }
                self.total_dispatched += 1;
                dispatched += 1;
            } else {
                remaining.push_back(job_id);
            }
        }

        self.pending_queue = remaining;
        dispatched
    }

    /// Select a worker based on the current strategy.
    fn select_worker(&self, preferred_region: Option<CloudRegion>) -> Option<String> {
        let available: Vec<&CloudWorker> = self
            .workers
            .values()
            .filter(|w| w.has_capacity())
            .collect();

        if available.is_empty() {
            return None;
        }

        match self.strategy {
            DispatchStrategy::LeastLoaded => available
                .iter()
                .max_by_key(|w| w.remaining_capacity())
                .map(|w| w.id.clone()),

            DispatchStrategy::Fastest => available
                .iter()
                .max_by(|a, b| a.speed_factor.partial_cmp(&b.speed_factor).unwrap_or(std::cmp::Ordering::Equal))
                .map(|w| w.id.clone()),

            DispatchStrategy::RegionAffinity => {
                // Prefer workers in the preferred region
                if let Some(region) = preferred_region {
                    let regional: Vec<&&CloudWorker> = available
                        .iter()
                        .filter(|w| w.region == region)
                        .collect();
                    if let Some(w) = regional
                        .iter()
                        .max_by_key(|w| w.remaining_capacity())
                    {
                        return Some(w.id.clone());
                    }
                }
                // Fall back to least-loaded
                available
                    .iter()
                    .max_by_key(|w| w.remaining_capacity())
                    .map(|w| w.id.clone())
            }
        }
    }

    /// Mark a job as completed with an output path.
    pub fn complete_job(&mut self, job_id: &str, output: impl Into<String>) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if job.is_terminal() {
                return false;
            }
            job.state = CloudJobState::Completed;
            job.output_path = Some(output.into());
            if let Some(wid) = &job.assigned_worker {
                if let Some(w) = self.workers.get_mut(wid) {
                    w.complete_job();
                }
            }
            self.total_completed += 1;
            true
        } else {
            false
        }
    }

    /// Mark a job as failed with an error message.
    pub fn fail_job(&mut self, job_id: &str, error: impl Into<String>) -> bool {
        if let Some(job) = self.jobs.get_mut(job_id) {
            if job.is_terminal() {
                return false;
            }
            job.state = CloudJobState::Failed;
            job.error = Some(error.into());
            if let Some(wid) = &job.assigned_worker {
                if let Some(w) = self.workers.get_mut(wid) {
                    w.complete_job();
                }
            }
            self.total_failed += 1;
            true
        } else {
            false
        }
    }

    /// Get a job by ID.
    pub fn get_job(&self, id: &str) -> Option<&CloudProxyJob> {
        self.jobs.get(id)
    }

    /// Get a worker by ID.
    pub fn get_worker(&self, id: &str) -> Option<&CloudWorker> {
        self.workers.get(id)
    }

    /// Number of registered workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Number of pending jobs.
    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    /// Total dispatched jobs.
    pub fn total_dispatched(&self) -> u64 {
        self.total_dispatched
    }

    /// Total completed jobs.
    pub fn total_completed(&self) -> u64 {
        self.total_completed
    }

    /// Total failed jobs.
    pub fn total_failed(&self) -> u64 {
        self.total_failed
    }

    /// Overall fleet utilization.
    #[allow(clippy::cast_precision_loss)]
    pub fn fleet_utilization(&self) -> f64 {
        let total_capacity: u32 = self.workers.values().map(|w| w.capacity).sum();
        let total_active: u32 = self.workers.values().map(|w| w.active_jobs).sum();
        if total_capacity == 0 {
            return 0.0;
        }
        total_active as f64 / total_capacity as f64
    }

    /// Set worker health.
    pub fn set_worker_health(&mut self, id: &str, health: WorkerHealth) -> bool {
        if let Some(w) = self.workers.get_mut(id) {
            w.health = health;
            true
        } else {
            false
        }
    }
}

impl Default for CloudProxyManager {
    fn default() -> Self {
        Self::new(DispatchStrategy::LeastLoaded)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_worker(id: &str, region: CloudRegion, capacity: u32) -> CloudWorker {
        CloudWorker::new(id, region, capacity)
    }

    fn make_job(id: &str, codec: &str) -> CloudProxyJob {
        CloudProxyJob::new(id, format!("/src/{id}.mov"), codec, (1920, 1080), 8_000)
    }

    #[test]
    fn test_worker_creation() {
        let w = make_worker("w1", CloudRegion::UsEast, 4);
        assert_eq!(w.id, "w1");
        assert_eq!(w.region, CloudRegion::UsEast);
        assert_eq!(w.capacity, 4);
        assert!(w.has_capacity());
    }

    #[test]
    fn test_worker_assign_and_complete() {
        let mut w = make_worker("w1", CloudRegion::UsWest, 2);
        assert!(w.assign_job());
        assert!(w.assign_job());
        assert!(!w.assign_job()); // at capacity
        assert_eq!(w.active_jobs, 2);
        w.complete_job();
        assert_eq!(w.active_jobs, 1);
        assert_eq!(w.completed_jobs, 1);
    }

    #[test]
    fn test_worker_utilization() {
        let mut w = make_worker("w1", CloudRegion::EuWest, 4);
        assert!((w.utilization() - 0.0).abs() < f64::EPSILON);
        w.assign_job();
        w.assign_job();
        assert!((w.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_worker_zero_capacity() {
        let w = CloudWorker::new("w0", CloudRegion::Other, 0);
        assert!(!w.has_capacity());
        assert!((w.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cloud_region_label() {
        assert_eq!(CloudRegion::UsEast.label(), "us-east-1");
        assert_eq!(CloudRegion::EuWest.label(), "eu-west-1");
        assert_eq!(CloudRegion::ApSoutheast.label(), "ap-southeast-1");
    }

    #[test]
    fn test_job_creation() {
        let job = make_job("j1", "h264");
        assert_eq!(job.state, CloudJobState::Pending);
        assert!(!job.is_terminal());
        assert!(job.assigned_worker.is_none());
    }

    #[test]
    fn test_job_with_region() {
        let job = make_job("j1", "vp9").with_region(CloudRegion::UsEast);
        assert_eq!(job.preferred_region, Some(CloudRegion::UsEast));
    }

    #[test]
    fn test_manager_submit_and_dispatch() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.submit_job(make_job("j2", "h264"));

        let dispatched = mgr.dispatch();
        assert_eq!(dispatched, 2);
        assert_eq!(mgr.total_dispatched(), 2);
        assert_eq!(mgr.pending_count(), 0);

        let j1 = mgr.get_job("j1").expect("job should exist");
        assert_eq!(j1.state, CloudJobState::Dispatched);
        assert_eq!(j1.assigned_worker.as_deref(), Some("w1"));
    }

    #[test]
    fn test_dispatch_no_capacity() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 1));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.submit_job(make_job("j2", "h264"));

        let dispatched = mgr.dispatch();
        assert_eq!(dispatched, 1);
        assert_eq!(mgr.pending_count(), 1);
    }

    #[test]
    fn test_complete_job_frees_worker() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 1));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.dispatch();

        assert!(mgr.complete_job("j1", "/proxy/j1.mp4"));
        assert_eq!(mgr.total_completed(), 1);

        let w = mgr.get_worker("w1").expect("worker should exist");
        assert_eq!(w.active_jobs, 0);
        assert_eq!(w.completed_jobs, 1);
    }

    #[test]
    fn test_fail_job() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.dispatch();

        assert!(mgr.fail_job("j1", "codec not supported"));
        assert_eq!(mgr.total_failed(), 1);

        let j = mgr.get_job("j1").expect("job should exist");
        assert_eq!(j.state, CloudJobState::Failed);
        assert_eq!(j.error.as_deref(), Some("codec not supported"));
    }

    #[test]
    fn test_cannot_complete_twice() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.dispatch();

        assert!(mgr.complete_job("j1", "/out.mp4"));
        assert!(!mgr.complete_job("j1", "/out2.mp4")); // already terminal
    }

    #[test]
    fn test_region_affinity_dispatch() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::RegionAffinity);
        mgr.register_worker(make_worker("w-us", CloudRegion::UsEast, 2));
        mgr.register_worker(make_worker("w-eu", CloudRegion::EuWest, 2));

        let job = make_job("j1", "h264").with_region(CloudRegion::EuWest);
        mgr.submit_job(job);
        mgr.dispatch();

        let j = mgr.get_job("j1").expect("job should exist");
        assert_eq!(j.assigned_worker.as_deref(), Some("w-eu"));
    }

    #[test]
    fn test_fastest_dispatch() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::Fastest);
        let slow = make_worker("slow", CloudRegion::UsEast, 4).with_speed(1.0);
        let fast = make_worker("fast", CloudRegion::UsEast, 4).with_speed(4.0);
        mgr.register_worker(slow);
        mgr.register_worker(fast);

        mgr.submit_job(make_job("j1", "h264"));
        mgr.dispatch();

        let j = mgr.get_job("j1").expect("job should exist");
        assert_eq!(j.assigned_worker.as_deref(), Some("fast"));
    }

    #[test]
    fn test_fleet_utilization() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 4));
        mgr.register_worker(make_worker("w2", CloudRegion::UsWest, 4));
        // Total capacity = 8
        mgr.submit_job(make_job("j1", "h264"));
        mgr.submit_job(make_job("j2", "h264"));
        mgr.dispatch();
        // 2 active out of 8 = 0.25
        assert!((mgr.fleet_utilization() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remove_worker() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        assert!(mgr.remove_worker("w1"));
        assert_eq!(mgr.worker_count(), 0);
    }

    #[test]
    fn test_remove_busy_worker_fails() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        mgr.submit_job(make_job("j1", "h264"));
        mgr.dispatch();
        assert!(!mgr.remove_worker("w1")); // has active job
    }

    #[test]
    fn test_set_worker_health() {
        let mut mgr = CloudProxyManager::new(DispatchStrategy::LeastLoaded);
        mgr.register_worker(make_worker("w1", CloudRegion::UsEast, 2));
        assert!(mgr.set_worker_health("w1", WorkerHealth::Unreachable));

        // Unreachable worker should not accept jobs
        mgr.submit_job(make_job("j1", "h264"));
        let dispatched = mgr.dispatch();
        assert_eq!(dispatched, 0);
    }

    #[test]
    fn test_default_manager() {
        let mgr = CloudProxyManager::default();
        assert_eq!(mgr.worker_count(), 0);
        assert_eq!(mgr.pending_count(), 0);
    }
}
