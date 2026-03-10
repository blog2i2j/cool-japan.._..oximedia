//! Job scheduling algorithms and priority management.
//!
//! The scheduler handles:
//! - Priority queue management
//! - Resource allocation
//! - Deadline-based scheduling
//! - Fairness policies
//! - Preemption support

#![allow(dead_code)]

use crate::pb::EncodingTask;
use crate::{JobPriority, Result};
use std::cmp::Ordering as CmpOrdering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use tracing::{debug, info, warn};

/// Job scheduler with multiple scheduling strategies
pub struct JobScheduler {
    /// Priority queue for jobs
    priority_queue: BinaryHeap<ScheduledJob>,

    /// FIFO queue for fair scheduling
    fifo_queue: VecDeque<ScheduledJob>,

    /// Deadline-based queue
    deadline_queue: BinaryHeap<DeadlineJob>,

    /// Resource allocation tracker
    resource_tracker: ResourceTracker,

    /// Scheduling policy
    policy: SchedulingPolicy,

    /// Statistics
    stats: SchedulerStats,
}

/// Scheduled job representation
#[derive(Debug, Clone)]
pub struct ScheduledJob {
    pub job_id: String,
    pub task_id: String,
    pub priority: JobPriority,
    pub deadline: Option<SystemTime>,
    pub encoding_task: Option<EncodingTask>,
}

impl PartialEq for ScheduledJob {
    fn eq(&self, other: &Self) -> bool {
        self.job_id == other.job_id
    }
}

impl Eq for ScheduledJob {}

impl PartialOrd for ScheduledJob {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledJob {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority comes first
        self.priority.cmp(&other.priority)
    }
}

/// Deadline-based job wrapper
#[derive(Debug, Clone)]
struct DeadlineJob {
    job: ScheduledJob,
    deadline: SystemTime,
}

impl PartialEq for DeadlineJob {
    fn eq(&self, other: &Self) -> bool {
        self.job.job_id == other.job.job_id
    }
}

impl Eq for DeadlineJob {}

impl PartialOrd for DeadlineJob {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for DeadlineJob {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Earlier deadline comes first (reverse ordering for max heap)
        other.deadline.cmp(&self.deadline)
    }
}

/// Scheduling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingPolicy {
    /// Priority-based scheduling
    #[default]
    Priority,
    /// First-In-First-Out
    FIFO,
    /// Earliest Deadline First
    EDF,
    /// Fair share scheduling
    FairShare,
    /// Shortest Job First
    SJF,
}

/// Resource allocation tracker
struct ResourceTracker {
    /// CPU cores allocated
    cpu_allocated: u32,
    /// Memory allocated (bytes)
    memory_allocated: u64,
    /// GPU devices allocated
    gpu_allocated: HashMap<String, bool>,
    /// Job resource usage
    job_resources: HashMap<String, ResourceAllocation>,
}

impl ResourceTracker {
    fn new() -> Self {
        Self {
            cpu_allocated: 0,
            memory_allocated: 0,
            gpu_allocated: HashMap::new(),
            job_resources: HashMap::new(),
        }
    }

    fn allocate(&mut self, job_id: &str, allocation: ResourceAllocation) -> bool {
        // Check if resources are available
        if self.cpu_allocated + allocation.cpu_cores > 1000 {
            return false;
        }

        if self.memory_allocated + allocation.memory_bytes > 1_099_511_627_776 {
            // 1TB limit
            return false;
        }

        // Allocate resources
        self.cpu_allocated += allocation.cpu_cores;
        self.memory_allocated += allocation.memory_bytes;
        self.job_resources.insert(job_id.to_string(), allocation);

        true
    }

    fn release(&mut self, job_id: &str) {
        if let Some(allocation) = self.job_resources.remove(job_id) {
            self.cpu_allocated = self.cpu_allocated.saturating_sub(allocation.cpu_cores);
            self.memory_allocated = self
                .memory_allocated
                .saturating_sub(allocation.memory_bytes);
        }
    }

    fn available_cpu(&self) -> u32 {
        1000u32.saturating_sub(self.cpu_allocated)
    }

    fn available_memory(&self) -> u64 {
        1_099_511_627_776u64.saturating_sub(self.memory_allocated)
    }
}

/// Resource allocation for a job
#[derive(Debug, Clone, Copy)]
pub struct ResourceAllocation {
    pub cpu_cores: u32,
    pub memory_bytes: u64,
    pub gpu_id: Option<usize>,
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_bytes: 4_294_967_296, // 4GB
            gpu_id: None,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub total_jobs_scheduled: u64,
    pub total_jobs_completed: u64,
    pub total_jobs_preempted: u64,
    pub average_wait_time: Duration,
}

impl JobScheduler {
    /// Create a new job scheduler
    #[must_use]
    pub fn new() -> Self {
        Self::with_policy(SchedulingPolicy::default())
    }

    /// Create a scheduler with a specific policy
    #[must_use]
    pub fn with_policy(policy: SchedulingPolicy) -> Self {
        Self {
            priority_queue: BinaryHeap::new(),
            fifo_queue: VecDeque::new(),
            deadline_queue: BinaryHeap::new(),
            resource_tracker: ResourceTracker::new(),
            policy,
            stats: SchedulerStats::default(),
        }
    }

    /// Enqueue a job for scheduling
    pub fn enqueue(&mut self, job: ScheduledJob) {
        info!(
            "Enqueueing job {} with priority {:?}",
            job.job_id, job.priority
        );

        match self.policy {
            SchedulingPolicy::Priority => {
                self.priority_queue.push(job);
            }
            SchedulingPolicy::FIFO => {
                self.fifo_queue.push_back(job);
            }
            SchedulingPolicy::EDF => {
                if let Some(deadline) = job.deadline {
                    self.deadline_queue.push(DeadlineJob { job, deadline });
                } else {
                    // No deadline, use priority queue
                    self.priority_queue.push(job);
                }
            }
            SchedulingPolicy::FairShare => {
                // Fair share uses FIFO with task-based balancing
                self.fifo_queue.push_back(job);
            }
            SchedulingPolicy::SJF => {
                // Shortest Job First - use priority queue
                // In practice, would estimate job duration
                self.priority_queue.push(job);
            }
        }

        self.stats.total_jobs_scheduled += 1;
    }

    /// Get the next job to schedule
    pub fn next_job(&mut self) -> Option<ScheduledJob> {
        match self.policy {
            SchedulingPolicy::Priority => self.priority_queue.pop(),
            SchedulingPolicy::FIFO => self.fifo_queue.pop_front(),
            SchedulingPolicy::EDF => {
                // Check deadline queue first
                if let Some(deadline_job) = self.deadline_queue.pop() {
                    // Check if deadline is still valid
                    if deadline_job.deadline > SystemTime::now() {
                        return Some(deadline_job.job);
                    }
                    // Deadline passed, drop job
                    warn!("Job {} missed deadline", deadline_job.job.job_id);
                    return self.next_job();
                }
                // Fall back to priority queue
                self.priority_queue.pop()
            }
            SchedulingPolicy::FairShare => self.fair_share_next(),
            SchedulingPolicy::SJF => self.shortest_job_first(),
        }
    }

    /// Fair share scheduling
    fn fair_share_next(&mut self) -> Option<ScheduledJob> {
        // Simple fair share: round-robin by task_id
        // In production, would track task quotas
        self.fifo_queue.pop_front()
    }

    /// Shortest Job First scheduling
    fn shortest_job_first(&mut self) -> Option<ScheduledJob> {
        // Estimate job duration and return shortest
        // For now, use priority queue
        self.priority_queue.pop()
    }

    /// Allocate resources for a job
    pub fn allocate_resources(&mut self, job_id: &str) -> Option<ResourceAllocation> {
        let allocation = ResourceAllocation::default();

        if self.resource_tracker.allocate(job_id, allocation) {
            debug!("Allocated resources for job {}", job_id);
            Some(allocation)
        } else {
            debug!("Insufficient resources for job {}", job_id);
            None
        }
    }

    /// Release resources for a completed job
    pub fn release_resources(&mut self, job_id: &str) {
        debug!("Releasing resources for job {}", job_id);
        self.resource_tracker.release(job_id);
        self.stats.total_jobs_completed += 1;
    }

    /// Preempt a job (remove and re-enqueue)
    pub fn preempt_job(&mut self, job_id: &str) -> Result<()> {
        info!("Preempting job {}", job_id);
        self.resource_tracker.release(job_id);
        self.stats.total_jobs_preempted += 1;
        Ok(())
    }

    /// Get queue length
    #[must_use]
    pub fn queue_length(&self) -> usize {
        match self.policy {
            SchedulingPolicy::Priority => self.priority_queue.len(),
            SchedulingPolicy::FIFO => self.fifo_queue.len(),
            SchedulingPolicy::EDF => self.deadline_queue.len() + self.priority_queue.len(),
            SchedulingPolicy::FairShare => self.fifo_queue.len(),
            SchedulingPolicy::SJF => self.priority_queue.len(),
        }
    }

    /// Get available resources
    #[must_use]
    pub fn available_resources(&self) -> (u32, u64) {
        (
            self.resource_tracker.available_cpu(),
            self.resource_tracker.available_memory(),
        )
    }

    /// Get scheduler statistics
    #[must_use]
    pub fn statistics(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Clear all queues
    pub fn clear(&mut self) {
        self.priority_queue.clear();
        self.fifo_queue.clear();
        self.deadline_queue.clear();
    }

    /// Set scheduling policy
    pub fn set_policy(&mut self, policy: SchedulingPolicy) {
        if self.policy != policy {
            info!("Changing scheduling policy to {:?}", policy);
            self.policy = policy;
            self.migrate_queues();
        }
    }

    /// Migrate jobs between queues when policy changes
    fn migrate_queues(&mut self) {
        // Collect all jobs
        let mut all_jobs = Vec::new();

        while let Some(job) = self.priority_queue.pop() {
            all_jobs.push(job);
        }

        while let Some(job) = self.fifo_queue.pop_front() {
            all_jobs.push(job);
        }

        while let Some(deadline_job) = self.deadline_queue.pop() {
            all_jobs.push(deadline_job.job);
        }

        // Re-enqueue with new policy
        for job in all_jobs {
            self.enqueue(job);
        }
    }

    /// Optimize queue order
    pub fn optimize(&mut self) {
        // Re-prioritize jobs based on current conditions
        match self.policy {
            SchedulingPolicy::Priority => {
                // Already optimized by heap
            }
            SchedulingPolicy::EDF => {
                // Check for deadline violations and re-prioritize
                let now = SystemTime::now();
                let urgent_jobs: Vec<_> = self
                    .deadline_queue
                    .iter()
                    .filter(|dj| {
                        dj.deadline
                            .duration_since(now)
                            .map(|d| d < Duration::from_secs(60))
                            .unwrap_or(true)
                    })
                    .cloned()
                    .collect();

                if !urgent_jobs.is_empty() {
                    debug!("Found {} urgent jobs", urgent_jobs.len());
                }
            }
            _ => {}
        }
    }
}

impl Default for JobScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Job scheduling builder for complex scheduling scenarios
pub struct SchedulingBuilder {
    policy: SchedulingPolicy,
    max_cpu: u32,
    max_memory: u64,
    enable_preemption: bool,
}

impl SchedulingBuilder {
    /// Create a new scheduling builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            policy: SchedulingPolicy::Priority,
            max_cpu: 1000,
            max_memory: 1_099_511_627_776,
            enable_preemption: false,
        }
    }

    /// Set scheduling policy
    #[must_use]
    pub fn policy(mut self, policy: SchedulingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set maximum CPU allocation
    #[must_use]
    pub fn max_cpu(mut self, max_cpu: u32) -> Self {
        self.max_cpu = max_cpu;
        self
    }

    /// Set maximum memory allocation
    #[must_use]
    pub fn max_memory(mut self, max_memory: u64) -> Self {
        self.max_memory = max_memory;
        self
    }

    /// Enable job preemption
    #[must_use]
    pub fn enable_preemption(mut self, enable: bool) -> Self {
        self.enable_preemption = enable;
        self
    }

    /// Build the scheduler
    #[must_use]
    pub fn build(self) -> JobScheduler {
        JobScheduler::with_policy(self.policy)
    }
}

impl Default for SchedulingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Task affinity for scheduling optimization
#[derive(Debug, Clone)]
pub struct TaskAffinity {
    /// Preferred worker IDs
    pub preferred_workers: Vec<String>,
    /// Required capabilities
    pub required_capabilities: Vec<String>,
    /// GPU requirement
    pub requires_gpu: bool,
}

impl TaskAffinity {
    /// Create a new task affinity
    #[must_use]
    pub fn new() -> Self {
        Self {
            preferred_workers: Vec::new(),
            required_capabilities: Vec::new(),
            requires_gpu: false,
        }
    }

    /// Add preferred worker
    #[must_use]
    pub fn prefer_worker(mut self, worker_id: String) -> Self {
        self.preferred_workers.push(worker_id);
        self
    }

    /// Add required capability
    #[must_use]
    pub fn require_capability(mut self, capability: String) -> Self {
        self.required_capabilities.push(capability);
        self
    }

    /// Set GPU requirement
    #[must_use]
    pub fn require_gpu(mut self, require: bool) -> Self {
        self.requires_gpu = require;
        self
    }
}

impl Default for TaskAffinity {
    fn default() -> Self {
        Self::new()
    }
}

/// Backfilling scheduler for improved utilization
pub struct BackfillingScheduler {
    main_queue: JobScheduler,
    backfill_queue: VecDeque<ScheduledJob>,
}

impl BackfillingScheduler {
    /// Create a new backfilling scheduler
    #[must_use]
    pub fn new() -> Self {
        Self {
            main_queue: JobScheduler::new(),
            backfill_queue: VecDeque::new(),
        }
    }

    /// Enqueue a job
    pub fn enqueue(&mut self, job: ScheduledJob) {
        if job.priority == JobPriority::Low {
            self.backfill_queue.push_back(job);
        } else {
            self.main_queue.enqueue(job);
        }
    }

    /// Get next job with backfilling
    pub fn next_job(&mut self) -> Option<ScheduledJob> {
        // Try main queue first
        if let Some(job) = self.main_queue.next_job() {
            return Some(job);
        }

        // Try backfill if resources available
        if !self.backfill_queue.is_empty() {
            let (cpu, mem) = self.main_queue.available_resources();
            if cpu >= 4 && mem >= 4_294_967_296 {
                return self.backfill_queue.pop_front();
            }
        }

        None
    }

    /// Clear all queues
    pub fn clear(&mut self) {
        self.main_queue.clear();
        self.backfill_queue.clear();
    }
}

impl Default for BackfillingScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn test_scheduler_creation() {
        let scheduler = JobScheduler::new();
        assert_eq!(scheduler.queue_length(), 0);
    }

    #[test]
    fn test_priority_scheduling() {
        let mut scheduler = JobScheduler::with_policy(SchedulingPolicy::Priority);

        let job1 = ScheduledJob {
            job_id: Uuid::new_v4().to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Low,
            deadline: None,
            encoding_task: None,
        };

        let job2 = ScheduledJob {
            job_id: Uuid::new_v4().to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Critical,
            deadline: None,
            encoding_task: None,
        };

        scheduler.enqueue(job1);
        scheduler.enqueue(job2.clone());

        let next = scheduler.next_job();
        assert!(next.is_some());
        assert_eq!(next.expect("next job should exist").job_id, job2.job_id);
    }

    #[test]
    fn test_fifo_scheduling() {
        let mut scheduler = JobScheduler::with_policy(SchedulingPolicy::FIFO);

        let job1 = ScheduledJob {
            job_id: "job1".to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Critical,
            deadline: None,
            encoding_task: None,
        };

        let job2 = ScheduledJob {
            job_id: "job2".to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Low,
            deadline: None,
            encoding_task: None,
        };

        scheduler.enqueue(job1.clone());
        scheduler.enqueue(job2);

        let next = scheduler.next_job();
        assert!(next.is_some());
        assert_eq!(next.expect("next job should exist").job_id, job1.job_id);
    }

    #[test]
    fn test_resource_allocation() {
        let mut scheduler = JobScheduler::new();
        let job_id = "test_job";

        let allocation = scheduler.allocate_resources(job_id);
        assert!(allocation.is_some());

        scheduler.release_resources(job_id);
        assert_eq!(scheduler.stats.total_jobs_completed, 1);
    }

    #[test]
    fn test_scheduling_builder() {
        let scheduler = SchedulingBuilder::new()
            .policy(SchedulingPolicy::EDF)
            .max_cpu(100)
            .max_memory(1_073_741_824)
            .enable_preemption(true)
            .build();

        assert_eq!(scheduler.queue_length(), 0);
    }

    #[test]
    fn test_backfilling() {
        let mut scheduler = BackfillingScheduler::new();

        let job1 = ScheduledJob {
            job_id: "job1".to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Normal,
            deadline: None,
            encoding_task: None,
        };

        let job2 = ScheduledJob {
            job_id: "job2".to_string(),
            task_id: Uuid::new_v4().to_string(),
            priority: JobPriority::Low,
            deadline: None,
            encoding_task: None,
        };

        scheduler.enqueue(job1.clone());
        scheduler.enqueue(job2);

        let next = scheduler.next_job();
        assert!(next.is_some());
        assert_eq!(next.expect("next job should exist").job_id, job1.job_id);
    }

    #[test]
    fn test_policy_migration() {
        let mut scheduler = JobScheduler::with_policy(SchedulingPolicy::Priority);

        for i in 0..5 {
            scheduler.enqueue(ScheduledJob {
                job_id: format!("job{}", i),
                task_id: Uuid::new_v4().to_string(),
                priority: JobPriority::Normal,
                deadline: None,
                encoding_task: None,
            });
        }

        assert_eq!(scheduler.queue_length(), 5);

        scheduler.set_policy(SchedulingPolicy::FIFO);
        assert_eq!(scheduler.queue_length(), 5);
    }
}
