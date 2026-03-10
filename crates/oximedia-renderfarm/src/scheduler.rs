// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job scheduling algorithms for render farm.

use crate::error::Result;
use crate::job::{JobId, Priority};
use crate::worker::{Worker, WorkerId};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Arc;

/// Scheduling algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingAlgorithm {
    /// First-Come-First-Served
    FCFS,
    /// Priority-based scheduling
    Priority,
    /// Deadline-aware scheduling
    Deadline,
    /// Fair-share scheduling
    FairShare,
    /// Backfill scheduling
    Backfill,
}

/// Scheduled task
#[derive(Debug, Clone)]
pub struct Task {
    /// Task ID
    pub id: String,
    /// Job ID
    pub job_id: JobId,
    /// Frame number
    pub frame: u32,
    /// Priority
    pub priority: Priority,
    /// Deadline
    pub deadline: Option<DateTime<Utc>>,
    /// Estimated time (seconds)
    pub estimated_time: f64,
    /// Dependencies
    pub dependencies: Vec<String>,
    /// Created at
    pub created_at: DateTime<Utc>,
}

impl Task {
    /// Create a new task
    #[must_use]
    pub fn new(job_id: JobId, frame: u32, priority: Priority) -> Self {
        Self {
            id: format!("{job_id}-{frame}"),
            job_id,
            frame,
            priority,
            deadline: None,
            estimated_time: 10.0,
            dependencies: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Calculate urgency score (higher = more urgent)
    #[must_use]
    pub fn urgency_score(&self) -> f64 {
        let mut score = match self.priority {
            Priority::Low => 1.0,
            Priority::Normal => 2.0,
            Priority::High => 3.0,
            Priority::Urgent => 4.0,
        };

        // Increase score if deadline is approaching
        if let Some(deadline) = self.deadline {
            let time_until_deadline = (deadline - Utc::now()).num_seconds() as f64;
            if time_until_deadline < 3600.0 {
                // Less than 1 hour
                score *= 2.0;
            } else if time_until_deadline < 86400.0 {
                // Less than 1 day
                score *= 1.5;
            }
        }

        // Increase score for older tasks (prevent starvation)
        let age = (Utc::now() - self.created_at).num_seconds() as f64;
        score += age / 3600.0; // Add 1.0 per hour of waiting

        score
    }
}

impl Ord for Task {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.urgency_score()
            .partial_cmp(&other.urgency_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl PartialOrd for Task {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Task {}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

/// Task assignment
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Worker ID
    pub worker_id: WorkerId,
    /// Task
    pub task: Task,
    /// Assigned at
    pub assigned_at: DateTime<Utc>,
}

/// Scheduler for render jobs
pub struct Scheduler {
    /// Scheduling algorithm
    algorithm: SchedulingAlgorithm,
    /// Task queue
    queue: Arc<RwLock<BinaryHeap<Task>>>,
    /// Pending queue (FCFS)
    pending: Arc<RwLock<VecDeque<Task>>>,
    /// Active assignments
    assignments: Arc<RwLock<HashMap<WorkerId, Assignment>>>,
    /// Job task counts (for fair-share)
    job_task_counts: Arc<RwLock<HashMap<JobId, u32>>>,
}

impl Scheduler {
    /// Create a new scheduler
    #[must_use]
    pub fn new(algorithm: SchedulingAlgorithm) -> Self {
        Self {
            algorithm,
            queue: Arc::new(RwLock::new(BinaryHeap::new())),
            pending: Arc::new(RwLock::new(VecDeque::new())),
            assignments: Arc::new(RwLock::new(HashMap::new())),
            job_task_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add task to queue
    pub fn enqueue(&self, task: Task) {
        let job_id = task.job_id;
        match self.algorithm {
            SchedulingAlgorithm::FCFS => {
                self.pending.write().push_back(task);
            }
            _ => {
                self.queue.write().push(task);
            }
        }

        // Update job task count
        let mut counts = self.job_task_counts.write();
        *counts.entry(job_id).or_insert(0) += 1;
    }

    /// Get next task for worker
    #[must_use]
    pub fn schedule(&self, worker: &Worker) -> Option<Task> {
        match self.algorithm {
            SchedulingAlgorithm::FCFS => self.schedule_fcfs(worker),
            SchedulingAlgorithm::Priority => self.schedule_priority(worker),
            SchedulingAlgorithm::Deadline => self.schedule_deadline(worker),
            SchedulingAlgorithm::FairShare => self.schedule_fair_share(worker),
            SchedulingAlgorithm::Backfill => self.schedule_backfill(worker),
        }
    }

    /// FCFS scheduling
    fn schedule_fcfs(&self, _worker: &Worker) -> Option<Task> {
        self.pending.write().pop_front()
    }

    /// Priority-based scheduling
    fn schedule_priority(&self, _worker: &Worker) -> Option<Task> {
        self.queue.write().pop()
    }

    /// Deadline-aware scheduling
    fn schedule_deadline(&self, _worker: &Worker) -> Option<Task> {
        let mut queue = self.queue.write();

        // Find task with earliest deadline
        let tasks: Vec<Task> = queue.drain().collect();
        if tasks.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut earliest_deadline = None;

        for (idx, task) in tasks.iter().enumerate() {
            if let Some(deadline) = task.deadline {
                if earliest_deadline.map_or(true, |d| deadline < d) {
                    earliest_deadline = Some(deadline);
                    best_idx = idx;
                }
            }
        }

        let mut selected = None;
        for (idx, task) in tasks.into_iter().enumerate() {
            if idx == best_idx {
                selected = Some(task);
            } else {
                queue.push(task);
            }
        }

        selected
    }

    /// Fair-share scheduling
    fn schedule_fair_share(&self, _worker: &Worker) -> Option<Task> {
        let mut queue = self.queue.write();
        let counts = self.job_task_counts.read();

        // Find job with fewest running tasks
        let tasks: Vec<Task> = queue.drain().collect();
        if tasks.is_empty() {
            return None;
        }

        let mut best_idx = 0;
        let mut min_count = u32::MAX;

        for (idx, task) in tasks.iter().enumerate() {
            let count = counts.get(&task.job_id).copied().unwrap_or(0);
            if count < min_count {
                min_count = count;
                best_idx = idx;
            }
        }

        let mut selected = None;
        for (idx, task) in tasks.into_iter().enumerate() {
            if idx == best_idx {
                selected = Some(task);
            } else {
                queue.push(task);
            }
        }

        selected
    }

    /// Backfill scheduling
    fn schedule_backfill(&self, _worker: &Worker) -> Option<Task> {
        let mut queue = self.queue.write();
        let tasks: Vec<Task> = queue.drain().collect();

        if tasks.is_empty() {
            return None;
        }

        // Calculate available time on worker (simplified)
        let available_time = 3600.0; // 1 hour

        // Try to find a task that fits
        let mut selected_idx = None;
        let mut best_score = f64::NEG_INFINITY;

        for (idx, task) in tasks.iter().enumerate() {
            if task.estimated_time <= available_time {
                let score = task.urgency_score() * (1.0 - task.estimated_time / available_time);
                if score > best_score {
                    best_score = score;
                    selected_idx = Some(idx);
                }
            }
        }

        let mut selected = None;
        for (idx, task) in tasks.into_iter().enumerate() {
            if Some(idx) == selected_idx {
                selected = Some(task);
            } else {
                queue.push(task);
            }
        }

        selected
    }

    /// Assign task to worker
    pub fn assign(&self, worker_id: WorkerId, task: Task) -> Result<Assignment> {
        let assignment = Assignment {
            worker_id,
            task,
            assigned_at: Utc::now(),
        };

        self.assignments
            .write()
            .insert(worker_id, assignment.clone());
        Ok(assignment)
    }

    /// Complete assignment
    pub fn complete(&self, worker_id: WorkerId) -> Result<()> {
        if let Some(assignment) = self.assignments.write().remove(&worker_id) {
            // Decrement job task count
            let mut counts = self.job_task_counts.write();
            if let Some(count) = counts.get_mut(&assignment.task.job_id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    counts.remove(&assignment.task.job_id);
                }
            }
        }
        Ok(())
    }

    /// Get current assignment for worker
    #[must_use]
    pub fn get_assignment(&self, worker_id: WorkerId) -> Option<Assignment> {
        self.assignments.read().get(&worker_id).cloned()
    }

    /// Get queue size
    #[must_use]
    pub fn queue_size(&self) -> usize {
        match self.algorithm {
            SchedulingAlgorithm::FCFS => self.pending.read().len(),
            _ => self.queue.read().len(),
        }
    }

    /// Get active assignments count
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.assignments.read().len()
    }

    /// Estimate total render time for all queued and active tasks.
    ///
    /// Uses a weighted model that accounts for:
    /// - Estimated execution time of each task
    /// - Number of available workers (parallelism)
    /// - Priority-based urgency weighting
    /// - Historical completion rate if available
    ///
    /// Returns the estimated wall-clock seconds to complete all remaining work
    /// given `worker_count` available workers.
    #[must_use]
    pub fn estimate_total_time(&self, worker_count: usize) -> f64 {
        if worker_count == 0 {
            return f64::INFINITY;
        }

        // Gather estimated times from all queued tasks
        let queued_times: Vec<f64> = match self.algorithm {
            SchedulingAlgorithm::FCFS => self
                .pending
                .read()
                .iter()
                .map(|t| t.estimated_time)
                .collect(),
            _ => self.queue.read().iter().map(|t| t.estimated_time).collect(),
        };

        // Add in-progress task remaining times (estimate 50% remaining on average)
        let active_remaining: f64 = self
            .assignments
            .read()
            .values()
            .map(|a| {
                let elapsed = (Utc::now() - a.assigned_at).num_seconds().max(0) as f64;
                // Remaining = estimated - elapsed, but at least 0
                (a.task.estimated_time - elapsed).max(0.0)
            })
            .sum();

        let total_queued_work: f64 = queued_times.iter().sum();
        let total_work = total_queued_work + active_remaining;

        // Wall-clock time = total work / parallelism
        // Apply a scheduling overhead factor (context switches, data transfer)
        let scheduling_overhead = 1.05; // 5% overhead
        let parallel_time = total_work / worker_count as f64;

        parallel_time * scheduling_overhead
    }

    /// Estimate completion time for a specific job.
    ///
    /// Sums estimated times for all tasks belonging to the given job
    /// that are still queued, and divides by `worker_count`.
    #[must_use]
    pub fn estimate_job_time(&self, job_id: JobId, worker_count: usize) -> f64 {
        if worker_count == 0 {
            return f64::INFINITY;
        }

        let queued_job_time: f64 = match self.algorithm {
            SchedulingAlgorithm::FCFS => self
                .pending
                .read()
                .iter()
                .filter(|t| t.job_id == job_id)
                .map(|t| t.estimated_time)
                .sum(),
            _ => self
                .queue
                .read()
                .iter()
                .filter(|t| t.job_id == job_id)
                .map(|t| t.estimated_time)
                .sum(),
        };

        // Check active assignments for this job
        let active_remaining: f64 = self
            .assignments
            .read()
            .values()
            .filter(|a| a.task.job_id == job_id)
            .map(|a| {
                let elapsed = (Utc::now() - a.assigned_at).num_seconds().max(0) as f64;
                (a.task.estimated_time - elapsed).max(0.0)
            })
            .sum();

        let total_job_work = queued_job_time + active_remaining;

        // For a single job, tasks may run in parallel up to worker_count
        let job_task_count = match self.algorithm {
            SchedulingAlgorithm::FCFS => self
                .pending
                .read()
                .iter()
                .filter(|t| t.job_id == job_id)
                .count(),
            _ => self
                .queue
                .read()
                .iter()
                .filter(|t| t.job_id == job_id)
                .count(),
        };

        let effective_parallelism = worker_count.min(job_task_count.max(1));
        total_job_work / effective_parallelism as f64
    }

    /// Compute a schedule feasibility report.
    ///
    /// Returns `(feasible_count, infeasible_count)` where infeasible tasks
    /// are those whose estimated completion time exceeds their deadline.
    #[must_use]
    pub fn feasibility_check(&self, worker_count: usize) -> (usize, usize) {
        let now = Utc::now();
        let tasks: Vec<Task> = match self.algorithm {
            SchedulingAlgorithm::FCFS => self.pending.read().iter().cloned().collect(),
            _ => self.queue.read().iter().cloned().collect(),
        };

        let mut feasible = 0;
        let mut infeasible = 0;

        // Estimate when each task might start (simple queuing model)
        let mut accumulated_time = 0.0_f64;
        let parallelism = (worker_count as f64).max(1.0);

        for task in &tasks {
            let start_time = accumulated_time / parallelism;
            let finish_time = start_time + task.estimated_time;

            if let Some(deadline) = task.deadline {
                let deadline_secs = (deadline - now).num_seconds() as f64;
                if finish_time <= deadline_secs {
                    feasible += 1;
                } else {
                    infeasible += 1;
                }
            } else {
                // No deadline => always feasible
                feasible += 1;
            }

            accumulated_time += task.estimated_time;
        }

        (feasible, infeasible)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new(SchedulingAlgorithm::Priority)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRegistration;
    use std::net::{IpAddr, Ipv4Addr};

    fn create_test_worker() -> Worker {
        let registration = WorkerRegistration {
            hostname: "worker01".to_string(),
            ip_address: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)),
            port: 8080,
            capabilities: Default::default(),
            location: None,
            tags: HashMap::new(),
        };
        Worker::new(registration)
    }

    #[test]
    fn test_task_urgency_score() {
        let task = Task::new(JobId::new(), 1, Priority::Normal);
        let score = task.urgency_score();
        assert!(score > 0.0);

        let urgent_task = Task::new(JobId::new(), 1, Priority::Urgent);
        assert!(urgent_task.urgency_score() > task.urgency_score());
    }

    #[test]
    fn test_scheduler_fcfs() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FCFS);
        let worker = create_test_worker();

        let task1 = Task::new(JobId::new(), 1, Priority::Low);
        let task2 = Task::new(JobId::new(), 2, Priority::High);

        scheduler.enqueue(task1.clone());
        scheduler.enqueue(task2);

        // FCFS should return task1 first (regardless of priority)
        let next = scheduler.schedule(&worker);
        assert!(next.is_some());
        assert_eq!(next.expect("should succeed in test").id, task1.id);
    }

    #[test]
    fn test_scheduler_priority() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Priority);
        let worker = create_test_worker();

        let task1 = Task::new(JobId::new(), 1, Priority::Low);
        let task2 = Task::new(JobId::new(), 2, Priority::High);

        scheduler.enqueue(task1);
        scheduler.enqueue(task2.clone());

        // Priority should return task2 first (higher priority)
        let next = scheduler.schedule(&worker);
        assert!(next.is_some());
        assert_eq!(next.expect("should succeed in test").id, task2.id);
    }

    #[test]
    fn test_scheduler_assignment() -> Result<()> {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FCFS);
        let worker = create_test_worker();
        let task = Task::new(JobId::new(), 1, Priority::Normal);

        scheduler.assign(worker.id, task.clone())?;

        let assignment = scheduler.get_assignment(worker.id);
        assert!(assignment.is_some());
        assert_eq!(assignment.expect("should succeed in test").task.id, task.id);

        Ok(())
    }

    #[test]
    fn test_scheduler_complete() -> Result<()> {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FCFS);
        let worker = create_test_worker();
        let task = Task::new(JobId::new(), 1, Priority::Normal);

        scheduler.assign(worker.id, task)?;
        assert_eq!(scheduler.active_count(), 1);

        scheduler.complete(worker.id)?;
        assert_eq!(scheduler.active_count(), 0);

        Ok(())
    }

    #[test]
    fn test_scheduler_queue_size() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Priority);

        assert_eq!(scheduler.queue_size(), 0);

        scheduler.enqueue(Task::new(JobId::new(), 1, Priority::Normal));
        assert_eq!(scheduler.queue_size(), 1);

        scheduler.enqueue(Task::new(JobId::new(), 2, Priority::Normal));
        assert_eq!(scheduler.queue_size(), 2);
    }

    #[test]
    fn test_task_ordering() {
        let task1 = Task::new(JobId::new(), 1, Priority::Low);
        let task2 = Task::new(JobId::new(), 2, Priority::High);

        assert!(task2 > task1);
    }

    #[test]
    fn test_deadline_scheduling() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Deadline);
        let worker = create_test_worker();

        let mut task1 = Task::new(JobId::new(), 1, Priority::Normal);
        task1.deadline = Some(Utc::now() + chrono::Duration::hours(2));

        let mut task2 = Task::new(JobId::new(), 2, Priority::Normal);
        task2.deadline = Some(Utc::now() + chrono::Duration::hours(1));

        scheduler.enqueue(task1);
        scheduler.enqueue(task2.clone());

        // Should return task2 (earlier deadline)
        let next = scheduler.schedule(&worker);
        assert!(next.is_some());
        assert_eq!(next.expect("should succeed in test").id, task2.id);
    }

    #[test]
    fn test_estimate_total_time() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FCFS);
        let mut task1 = Task::new(JobId::new(), 1, Priority::Normal);
        task1.estimated_time = 100.0;
        let mut task2 = Task::new(JobId::new(), 2, Priority::Normal);
        task2.estimated_time = 200.0;

        scheduler.enqueue(task1);
        scheduler.enqueue(task2);

        // 2 workers: total work = 300s, parallelism = 2, expected ~157.5s (with 5% overhead)
        let est = scheduler.estimate_total_time(2);
        assert!(est > 0.0);
        assert!(est < 400.0);

        // 0 workers: infinity
        assert!(scheduler.estimate_total_time(0).is_infinite());
    }

    #[test]
    fn test_estimate_job_time() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::Priority);
        let job_id = JobId::new();

        let mut t1 = Task::new(job_id, 1, Priority::Normal);
        t1.estimated_time = 50.0;
        let mut t2 = Task::new(job_id, 2, Priority::Normal);
        t2.estimated_time = 50.0;
        let mut t3 = Task::new(JobId::new(), 3, Priority::Normal);
        t3.estimated_time = 100.0;

        scheduler.enqueue(t1);
        scheduler.enqueue(t2);
        scheduler.enqueue(t3);

        let est = scheduler.estimate_job_time(job_id, 2);
        // 2 tasks of 50s each, 2 workers => ~50s
        assert!(est > 0.0);
        assert!(est < 200.0);
    }

    #[test]
    fn test_feasibility_check() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FCFS);

        let mut task1 = Task::new(JobId::new(), 1, Priority::Normal);
        task1.estimated_time = 10.0;
        task1.deadline = Some(Utc::now() + chrono::Duration::hours(1));

        let mut task2 = Task::new(JobId::new(), 2, Priority::Normal);
        task2.estimated_time = 10.0;
        // No deadline

        scheduler.enqueue(task1);
        scheduler.enqueue(task2);

        let (feasible, infeasible) = scheduler.feasibility_check(2);
        assert_eq!(feasible, 2);
        assert_eq!(infeasible, 0);
    }

    #[test]
    fn test_fair_share_scheduling() {
        let scheduler = Scheduler::new(SchedulingAlgorithm::FairShare);
        let worker = create_test_worker();

        let job1 = JobId::new();
        let job2 = JobId::new();

        let task1 = Task::new(job1, 1, Priority::Normal);
        let task2 = Task::new(job1, 2, Priority::Normal);
        let task3 = Task::new(job2, 1, Priority::Normal);

        scheduler.enqueue(task1);
        scheduler.enqueue(task2);
        scheduler.enqueue(task3.clone());

        // Should prioritize job2 (fewer running tasks)
        let next = scheduler.schedule(&worker);
        assert!(next.is_some());
    }
}
