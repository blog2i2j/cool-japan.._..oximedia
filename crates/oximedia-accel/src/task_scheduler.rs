#![allow(dead_code)]
//! Task scheduler for hardware-accelerated operations in `oximedia-accel`.
//!
//! Provides a priority queue-based scheduler that orders `AccelTask` items
//! by priority and estimated cost, choosing the optimal hardware target for
//! each task based on registered `AccelProfile` hints.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Task priority levels — higher numeric value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    /// Background, deferred work.
    Low = 0,
    /// Normal interactive processing.
    Normal = 1,
    /// High-priority, time-sensitive work.
    High = 2,
    /// Real-time critical path (e.g. live broadcast).
    RealTime = 3,
}

impl TaskPriority {
    /// Returns a human-readable name for this priority.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Normal => "normal",
            Self::High => "high",
            Self::RealTime => "realtime",
        }
    }

    /// Returns `true` if this priority is considered time-critical.
    #[must_use]
    pub fn is_time_critical(&self) -> bool {
        matches!(self, Self::High | Self::RealTime)
    }
}

/// The kind of work an `AccelTask` represents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskKind {
    /// Image scaling / resizing.
    Scale,
    /// Colour space conversion.
    ColorConvert,
    /// Motion estimation.
    MotionEstimate,
    /// Generic compute kernel invocation.
    ComputeKernel(String),
    /// ML inference pass.
    Inference,
}

impl TaskKind {
    /// Whether this task kind benefits significantly from GPU acceleration.
    #[must_use]
    pub fn prefers_gpu(&self) -> bool {
        matches!(
            self,
            Self::Scale | Self::ColorConvert | Self::MotionEstimate | Self::Inference
        )
    }
}

/// A single unit of accelerated work.
#[derive(Debug, Clone)]
pub struct AccelTask {
    /// Unique task identifier.
    pub id: u64,
    /// The kind of work.
    pub kind: TaskKind,
    /// Task priority.
    pub priority: TaskPriority,
    /// Estimated compute cost in arbitrary units (higher = more expensive).
    pub cost: u32,
    /// Optional deadline (Unix timestamp in milliseconds).
    pub deadline_ms: Option<u64>,
}

impl AccelTask {
    /// Create a new task.
    #[must_use]
    pub fn new(id: u64, kind: TaskKind, priority: TaskPriority, cost: u32) -> Self {
        Self {
            id,
            kind,
            priority,
            cost,
            deadline_ms: None,
        }
    }

    /// Attach a deadline to this task.
    #[must_use]
    pub fn with_deadline(mut self, deadline_ms: u64) -> Self {
        self.deadline_ms = Some(deadline_ms);
        self
    }

    /// Returns `true` if this task has missed its deadline relative to `now_ms`.
    #[must_use]
    pub fn is_overdue(&self, now_ms: u64) -> bool {
        self.deadline_ms.is_some_and(|d| now_ms > d)
    }
}

/// Wrapper for `BinaryHeap` ordering — higher priority first, then lower cost.
#[derive(Debug)]
struct QueueEntry(AccelTask);

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.priority == other.0.priority && self.0.cost == other.0.cost
    }
}
impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; break ties by lower cost first.
        self.0
            .priority
            .cmp(&other.0.priority)
            .then(other.0.cost.cmp(&self.0.cost))
    }
}

/// Statistics collected by the scheduler.
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total tasks submitted.
    pub submitted: u64,
    /// Total tasks dispatched.
    pub dispatched: u64,
    /// Tasks that were overdue when dispatched.
    pub overdue_dispatched: u64,
}

/// Priority-queue task scheduler for accelerated operations.
///
/// Tasks are enqueued with a priority and cost estimate.  `schedule()` pops
/// the next highest-priority task and returns it with the recommended
/// hardware target.
#[derive(Debug)]
pub struct AccelScheduler {
    queue: BinaryHeap<QueueEntry>,
    gpu_available: bool,
    npu_available: bool,
    stats: SchedulerStats,
}

impl AccelScheduler {
    /// Create a scheduler.
    ///
    /// - `gpu_available`: whether GPU dispatch is possible.
    /// - `npu_available`: whether NPU dispatch is possible.
    #[must_use]
    pub fn new(gpu_available: bool, npu_available: bool) -> Self {
        Self {
            queue: BinaryHeap::new(),
            gpu_available,
            npu_available,
            stats: SchedulerStats::default(),
        }
    }

    /// Enqueue a task for scheduling.
    pub fn enqueue(&mut self, task: AccelTask) {
        self.stats.submitted += 1;
        self.queue.push(QueueEntry(task));
    }

    /// Pop and return the next task along with the recommended target.
    ///
    /// Returns `None` when the queue is empty.
    pub fn schedule(
        &mut self,
        now_ms: u64,
    ) -> Option<(AccelTask, crate::accel_profile::AccelTarget)> {
        let QueueEntry(task) = self.queue.pop()?;
        self.stats.dispatched += 1;
        if task.is_overdue(now_ms) {
            self.stats.overdue_dispatched += 1;
        }
        let target = self.choose_target(&task);
        Some((task, target))
    }

    /// Returns a snapshot of scheduler statistics.
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        self.stats.clone()
    }

    /// Returns the number of tasks currently in the queue.
    #[must_use]
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Drain all tasks from the queue without dispatching (e.g. for shutdown).
    pub fn drain(&mut self) -> Vec<AccelTask> {
        let mut tasks = Vec::new();
        while let Some(QueueEntry(t)) = self.queue.pop() {
            tasks.push(t);
        }
        tasks
    }

    // ── private helpers ──────────────────────────────────────────────────────

    fn choose_target(&self, task: &AccelTask) -> crate::accel_profile::AccelTarget {
        use crate::accel_profile::AccelTarget;
        if task.kind == TaskKind::Inference && self.npu_available {
            return AccelTarget::Npu;
        }
        if task.kind.prefers_gpu() && self.gpu_available {
            return AccelTarget::Gpu;
        }
        AccelTarget::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accel_profile::AccelTarget;

    fn task(id: u64, priority: TaskPriority, cost: u32) -> AccelTask {
        AccelTask::new(id, TaskKind::Scale, priority, cost)
    }

    #[test]
    fn test_enqueue_and_schedule() {
        let mut s = AccelScheduler::new(true, false);
        s.enqueue(task(1, TaskPriority::Normal, 100));
        let result = s.schedule(0);
        assert!(result.is_some());
    }

    #[test]
    fn test_schedule_empty_returns_none() {
        let mut s = AccelScheduler::new(false, false);
        assert!(s.schedule(0).is_none());
    }

    #[test]
    fn test_priority_ordering_high_before_low() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Low, 10));
        s.enqueue(task(2, TaskPriority::High, 10));
        let (first, _) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(first.priority, TaskPriority::High);
    }

    #[test]
    fn test_realtime_dispatched_first() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Normal, 10));
        s.enqueue(task(2, TaskPriority::RealTime, 50));
        let (first, _) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(first.priority, TaskPriority::RealTime);
    }

    #[test]
    fn test_same_priority_lower_cost_first() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Normal, 200));
        s.enqueue(task(2, TaskPriority::Normal, 50));
        let (first, _) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(first.cost, 50);
    }

    #[test]
    fn test_gpu_target_for_scale_when_available() {
        let mut s = AccelScheduler::new(true, false);
        s.enqueue(AccelTask::new(
            1,
            TaskKind::Scale,
            TaskPriority::Normal,
            100,
        ));
        let (_, target) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(target, AccelTarget::Gpu);
    }

    #[test]
    fn test_cpu_target_when_gpu_unavailable() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(AccelTask::new(
            1,
            TaskKind::Scale,
            TaskPriority::Normal,
            100,
        ));
        let (_, target) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(target, AccelTarget::Cpu);
    }

    #[test]
    fn test_npu_target_for_inference() {
        let mut s = AccelScheduler::new(true, true);
        s.enqueue(AccelTask::new(
            1,
            TaskKind::Inference,
            TaskPriority::Normal,
            100,
        ));
        let (_, target) = s.schedule(0).expect("schedule should succeed");
        assert_eq!(target, AccelTarget::Npu);
    }

    #[test]
    fn test_stats_submitted_count() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Low, 10));
        s.enqueue(task(2, TaskPriority::Low, 10));
        assert_eq!(s.stats().submitted, 2);
    }

    #[test]
    fn test_stats_dispatched_count() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Normal, 10));
        s.schedule(0);
        assert_eq!(s.stats().dispatched, 1);
    }

    #[test]
    fn test_overdue_task_counted() {
        let mut s = AccelScheduler::new(false, false);
        let t = task(1, TaskPriority::Normal, 10).with_deadline(100);
        s.enqueue(t);
        s.schedule(200); // now > deadline
        assert_eq!(s.stats().overdue_dispatched, 1);
    }

    #[test]
    fn test_queue_depth() {
        let mut s = AccelScheduler::new(false, false);
        assert_eq!(s.queue_depth(), 0);
        s.enqueue(task(1, TaskPriority::Low, 10));
        assert_eq!(s.queue_depth(), 1);
    }

    #[test]
    fn test_drain_empties_queue() {
        let mut s = AccelScheduler::new(false, false);
        s.enqueue(task(1, TaskPriority::Low, 10));
        s.enqueue(task(2, TaskPriority::Normal, 20));
        let drained = s.drain();
        assert_eq!(drained.len(), 2);
        assert!(s.is_empty());
    }

    #[test]
    fn test_task_priority_name() {
        assert_eq!(TaskPriority::RealTime.name(), "realtime");
    }

    #[test]
    fn test_task_priority_is_time_critical() {
        assert!(TaskPriority::High.is_time_critical());
        assert!(!TaskPriority::Low.is_time_critical());
    }

    #[test]
    fn test_task_kind_prefers_gpu() {
        assert!(TaskKind::ColorConvert.prefers_gpu());
        assert!(TaskKind::Inference.prefers_gpu());
        assert!(!TaskKind::ComputeKernel("custom".to_string()).prefers_gpu());
    }
}
