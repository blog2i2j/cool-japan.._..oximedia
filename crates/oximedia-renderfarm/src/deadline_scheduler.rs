// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Deadline-aware scheduling for render tasks.
//!
//! Implements Earliest-Deadline-First (EDF) scheduling with:
//! - Priority inversion prevention via priority inheritance
//! - Slack-time computation for time-to-deadline headroom
//! - Overload detection and admission control

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// Priority level for a render task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Background work, no hard deadline.
    Background = 0,
    /// Normal render task.
    Normal = 1,
    /// High-priority rush job.
    High = 2,
    /// Real-time or broadcast-critical task.
    Critical = 3,
}

/// Unique identifier for a schedulable task.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(pub String);

impl TaskId {
    /// Create a task ID from a string slice.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A render task with deadline and resource requirements.
#[derive(Debug, Clone)]
pub struct DeadlineTask {
    /// Unique task identifier.
    pub id: TaskId,
    /// Human-readable task name.
    pub name: String,
    /// Base priority level.
    pub priority: TaskPriority,
    /// Absolute deadline (monotonic clock).
    pub deadline: Instant,
    /// Estimated execution duration.
    pub estimated_duration: Duration,
    /// CPU cores required.
    pub required_cores: u32,
    /// RAM required in megabytes.
    pub required_ram_mb: u64,
    /// Effective priority after inheritance adjustments (internal).
    effective_priority: TaskPriority,
}

impl DeadlineTask {
    /// Create a new deadline task.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        priority: TaskPriority,
        deadline: Instant,
        estimated_duration: Duration,
        required_cores: u32,
        required_ram_mb: u64,
    ) -> Self {
        Self {
            id: TaskId::new(id),
            name: name.into(),
            priority,
            deadline,
            estimated_duration,
            required_cores,
            required_ram_mb,
            effective_priority: priority,
        }
    }

    /// Compute the remaining slack time: `deadline - now - estimated_duration`.
    ///
    /// Negative slack means the task cannot finish before its deadline even if
    /// started immediately.
    #[must_use]
    pub fn slack(&self, now: Instant) -> Duration {
        let time_to_deadline = self.deadline.saturating_duration_since(now);
        time_to_deadline.saturating_sub(self.estimated_duration)
    }

    /// Returns `true` if the deadline has already passed.
    #[must_use]
    pub fn is_expired(&self, now: Instant) -> bool {
        now > self.deadline
    }

    /// Elevate the effective priority (priority inheritance).
    pub fn inherit_priority(&mut self, donor: TaskPriority) {
        if donor > self.effective_priority {
            self.effective_priority = donor;
        }
    }

    /// Reset effective priority back to base.
    pub fn reset_priority(&mut self) {
        self.effective_priority = self.priority;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EDF queue internals
// ─────────────────────────────────────────────────────────────────────────────

/// Entry stored in the EDF binary heap.
///
/// Ordering: earlier deadline wins; ties broken by higher effective priority.
#[derive(Debug)]
struct EdfEntry {
    deadline: Instant,
    effective_priority: TaskPriority,
    id: TaskId,
}

impl PartialEq for EdfEntry {
    fn eq(&self, other: &Self) -> bool {
        self.deadline == other.deadline && self.effective_priority == other.effective_priority
    }
}

impl Eq for EdfEntry {}

impl PartialOrd for EdfEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdfEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; we want earliest deadline at top, so reverse.
        other
            .deadline
            .cmp(&self.deadline)
            .then_with(|| self.effective_priority.cmp(&other.effective_priority))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EDF Scheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Error type for deadline scheduler operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// Task with the given ID is not in the ready queue.
    TaskNotFound(String),
    /// Task was rejected by admission control.
    AdmissionRejected(String),
    /// Scheduler has no capacity for new tasks.
    QueueFull,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TaskNotFound(id) => write!(f, "task not found: {id}"),
            Self::AdmissionRejected(reason) => write!(f, "admission rejected: {reason}"),
            Self::QueueFull => write!(f, "scheduler queue is full"),
        }
    }
}

/// Earliest-Deadline-First scheduler with priority inversion prevention.
#[derive(Debug)]
pub struct EdfScheduler {
    ready_queue: BinaryHeap<EdfEntry>,
    tasks: HashMap<TaskId, DeadlineTask>,
    max_queue_size: usize,
    /// Accumulated total utilisation (sum of `estimated / deadline_window`).
    utilisation: f64,
}

impl EdfScheduler {
    /// Create a new EDF scheduler with an optional maximum queue size.
    #[must_use]
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            ready_queue: BinaryHeap::new(),
            tasks: HashMap::new(),
            max_queue_size,
            utilisation: 0.0,
        }
    }

    /// Attempt to admit a task into the ready queue.
    ///
    /// Uses a simple utilisation bound check: if adding this task would push
    /// the total utilisation above 1.0, admission is denied.
    pub fn admit(&mut self, task: DeadlineTask) -> Result<(), SchedulerError> {
        if self.tasks.len() >= self.max_queue_size {
            return Err(SchedulerError::QueueFull);
        }

        // Utilisation contribution: c_i / d_i  (from now)
        let now = Instant::now();
        let deadline_window = task.deadline.saturating_duration_since(now).as_secs_f64();
        let task_util = if deadline_window > 0.0 {
            task.estimated_duration.as_secs_f64() / deadline_window
        } else {
            // Deadline already expired or too tight
            return Err(SchedulerError::AdmissionRejected(
                "deadline already expired".into(),
            ));
        };

        if self.utilisation + task_util > 1.0 {
            return Err(SchedulerError::AdmissionRejected(format!(
                "would exceed utilisation bound ({:.2})",
                self.utilisation + task_util
            )));
        }

        self.utilisation += task_util;
        let entry = EdfEntry {
            deadline: task.deadline,
            effective_priority: task.effective_priority,
            id: task.id.clone(),
        };
        self.tasks.insert(task.id.clone(), task);
        self.ready_queue.push(entry);
        Ok(())
    }

    /// Dequeue the highest-priority (earliest deadline) task.
    ///
    /// Expired tasks are silently discarded and the next candidate is returned.
    pub fn next_task(&mut self) -> Option<DeadlineTask> {
        let now = Instant::now();
        loop {
            let entry = self.ready_queue.pop()?;
            if let Some(task) = self.tasks.remove(&entry.id) {
                if task.is_expired(now) {
                    // Discard expired tasks
                    continue;
                }
                return Some(task);
            }
        }
    }

    /// Apply priority inheritance: raise the effective priority of `holder_id`
    /// to the effective priority of `waiter_id`.
    pub fn inherit_priority(
        &mut self,
        holder_id: &TaskId,
        waiter_id: &TaskId,
    ) -> Result<(), SchedulerError> {
        let waiter_prio = self
            .tasks
            .get(waiter_id)
            .ok_or_else(|| SchedulerError::TaskNotFound(waiter_id.to_string()))?
            .effective_priority;

        let holder = self
            .tasks
            .get_mut(holder_id)
            .ok_or_else(|| SchedulerError::TaskNotFound(holder_id.to_string()))?;

        holder.inherit_priority(waiter_prio);
        Ok(())
    }

    /// Return the number of tasks in the ready queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Returns `true` when the ready queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Current aggregate utilisation (0.0 – 1.0+).
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        self.utilisation
    }

    /// Peek at the task with the earliest deadline without removing it.
    #[must_use]
    pub fn peek_next(&self) -> Option<&DeadlineTask> {
        let entry = self.ready_queue.peek()?;
        self.tasks.get(&entry.id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slack time analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Summary of slack-time analysis across a set of tasks.
#[derive(Debug, Clone)]
pub struct SlackAnalysis {
    /// Number of tasks analysed.
    pub task_count: usize,
    /// Tasks with negative slack (cannot meet deadline).
    pub infeasible_count: usize,
    /// Minimum slack (can be negative).
    pub min_slack_secs: f64,
    /// Average slack.
    pub avg_slack_secs: f64,
    /// Maximum slack.
    pub max_slack_secs: f64,
}

impl SlackAnalysis {
    /// Analyse the given tasks at the given instant.
    #[must_use]
    pub fn analyse(tasks: &[&DeadlineTask], now: Instant) -> Self {
        if tasks.is_empty() {
            return Self {
                task_count: 0,
                infeasible_count: 0,
                min_slack_secs: 0.0,
                avg_slack_secs: 0.0,
                max_slack_secs: 0.0,
            };
        }

        let slacks: Vec<f64> = tasks
            .iter()
            .map(|t| {
                let time_left = t.deadline.saturating_duration_since(now).as_secs_f64();
                let est = t.estimated_duration.as_secs_f64();
                time_left - est
            })
            .collect();

        let infeasible = slacks.iter().filter(|&&s| s < 0.0).count();
        let min = slacks.iter().copied().fold(f64::INFINITY, f64::min);
        let max = slacks.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg = slacks.iter().sum::<f64>() / slacks.len() as f64;

        Self {
            task_count: tasks.len(),
            infeasible_count: infeasible,
            min_slack_secs: min,
            avg_slack_secs: avg,
            max_slack_secs: max,
        }
    }

    /// Returns `true` if all tasks have non-negative slack.
    #[must_use]
    pub fn is_feasible(&self) -> bool {
        self.infeasible_count == 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_task(
        id: &str,
        priority: TaskPriority,
        secs_until_deadline: u64,
        est_secs: u64,
    ) -> DeadlineTask {
        DeadlineTask::new(
            id,
            id,
            priority,
            Instant::now() + Duration::from_secs(secs_until_deadline),
            Duration::from_secs(est_secs),
            4,
            8192,
        )
    }

    #[test]
    fn test_task_slack_positive() {
        let task = make_task("t1", TaskPriority::Normal, 100, 10);
        let slack = task.slack(Instant::now());
        // At least 89 seconds of slack (100 - 10 - epsilon)
        assert!(slack.as_secs() >= 89);
    }

    #[test]
    fn test_task_slack_zero_when_tight() {
        let task = make_task("t1", TaskPriority::Normal, 10, 10);
        // slack = 0 (best case)
        let slack = task.slack(Instant::now());
        assert!(slack.as_secs() <= 1);
    }

    #[test]
    fn test_task_is_expired() {
        let task = DeadlineTask::new(
            "t1",
            "t1",
            TaskPriority::Normal,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test expectation failed"), // past
            Duration::from_secs(5),
            4,
            8192,
        );
        assert!(task.is_expired(Instant::now()));
    }

    #[test]
    fn test_task_not_expired() {
        let task = make_task("t1", TaskPriority::High, 60, 5);
        assert!(!task.is_expired(Instant::now()));
    }

    #[test]
    fn test_priority_inheritance() {
        let mut task = make_task("t1", TaskPriority::Normal, 60, 5);
        task.inherit_priority(TaskPriority::Critical);
        assert_eq!(task.effective_priority, TaskPriority::Critical);
    }

    #[test]
    fn test_priority_inheritance_no_downgrade() {
        let mut task = make_task("t1", TaskPriority::High, 60, 5);
        task.inherit_priority(TaskPriority::Normal); // lower — should not apply
        assert_eq!(task.effective_priority, TaskPriority::High);
    }

    #[test]
    fn test_reset_priority() {
        let mut task = make_task("t1", TaskPriority::Normal, 60, 5);
        task.inherit_priority(TaskPriority::Critical);
        task.reset_priority();
        assert_eq!(task.effective_priority, TaskPriority::Normal);
    }

    #[test]
    fn test_edf_admit_and_dequeue_order() {
        let mut sched = EdfScheduler::new(100);
        let t_far = make_task("far", TaskPriority::Normal, 200, 10);
        let t_near = make_task("near", TaskPriority::Normal, 50, 5);
        sched.admit(t_far).expect("should succeed in test");
        sched.admit(t_near).expect("should succeed in test");
        // Earliest deadline (near) should come first
        let first = sched.next_task().expect("should succeed in test");
        assert_eq!(first.id.0, "near");
    }

    #[test]
    fn test_edf_queue_full() {
        let mut sched = EdfScheduler::new(1);
        sched
            .admit(make_task("t1", TaskPriority::Normal, 100, 5))
            .expect("should succeed in test");
        let res = sched.admit(make_task("t2", TaskPriority::Normal, 200, 5));
        assert_eq!(res, Err(SchedulerError::QueueFull));
    }

    #[test]
    fn test_edf_admission_expired() {
        let mut sched = EdfScheduler::new(10);
        let expired = DeadlineTask::new(
            "exp",
            "exp",
            TaskPriority::Normal,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("test expectation failed"),
            Duration::from_secs(5),
            4,
            8192,
        );
        let res = sched.admit(expired);
        assert!(matches!(res, Err(SchedulerError::AdmissionRejected(_))));
    }

    #[test]
    fn test_edf_len_and_is_empty() {
        let mut sched = EdfScheduler::new(10);
        assert!(sched.is_empty());
        sched
            .admit(make_task("t1", TaskPriority::Normal, 60, 5))
            .expect("should succeed in test");
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn test_edf_utilisation_increases() {
        let mut sched = EdfScheduler::new(10);
        let before = sched.utilisation();
        sched
            .admit(make_task("t1", TaskPriority::Normal, 100, 10))
            .expect("should succeed in test");
        assert!(sched.utilisation() > before);
    }

    #[test]
    fn test_slack_analysis_feasible() {
        let t1 = make_task("t1", TaskPriority::Normal, 100, 10);
        let t2 = make_task("t2", TaskPriority::High, 200, 20);
        let analysis = SlackAnalysis::analyse(&[&t1, &t2], Instant::now());
        assert!(analysis.is_feasible());
        assert_eq!(analysis.task_count, 2);
        assert_eq!(analysis.infeasible_count, 0);
    }

    #[test]
    fn test_slack_analysis_infeasible() {
        let tight = DeadlineTask::new(
            "tight",
            "tight",
            TaskPriority::Critical,
            Instant::now() + Duration::from_secs(3),
            Duration::from_secs(10), // needs 10s but only 3s window
            4,
            8192,
        );
        let analysis = SlackAnalysis::analyse(&[&tight], Instant::now());
        assert!(!analysis.is_feasible());
        assert_eq!(analysis.infeasible_count, 1);
    }

    #[test]
    fn test_slack_analysis_empty() {
        let analysis = SlackAnalysis::analyse(&[], Instant::now());
        assert_eq!(analysis.task_count, 0);
        assert!(analysis.is_feasible());
    }
}
