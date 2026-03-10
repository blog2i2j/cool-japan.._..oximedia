#![allow(dead_code)]
//! Batch scheduling — schedule types, individual schedules, and a scheduler registry.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Classification of when a batch should run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScheduleType {
    /// Run as soon as possible.
    Immediate,
    /// Run after a fixed delay from submission.
    Delayed(Duration),
    /// Run repeatedly with a fixed interval between executions.
    Recurring(Duration),
}

impl ScheduleType {
    /// Returns `true` if this schedule repeats.
    #[must_use]
    pub fn is_recurring(&self) -> bool {
        matches!(self, Self::Recurring(_))
    }

    /// Returns `true` if this schedule runs immediately.
    #[must_use]
    pub fn is_immediate(&self) -> bool {
        matches!(self, Self::Immediate)
    }

    /// Return the interval duration, if applicable.
    #[must_use]
    pub fn interval(&self) -> Option<Duration> {
        match self {
            Self::Delayed(d) | Self::Recurring(d) => Some(*d),
            Self::Immediate => None,
        }
    }
}

/// An individual batch schedule entry.
#[derive(Debug, Clone)]
pub struct BatchSchedule {
    /// Unique identifier for this schedule.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Schedule type controlling when runs are triggered.
    pub schedule_type: ScheduleType,
    /// When this schedule was created.
    created_at: Instant,
    /// How many times this schedule has triggered.
    run_count: u64,
}

impl BatchSchedule {
    /// Create a new batch schedule.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        schedule_type: ScheduleType,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            schedule_type,
            created_at: Instant::now(),
            run_count: 0,
        }
    }

    /// Compute the `Instant` at which the next run should occur.
    ///
    /// For `Immediate`, returns the creation time.
    /// For `Delayed` and `Recurring`, returns `created_at + interval * (run_count + 1)`.
    #[must_use]
    pub fn next_run_at(&self) -> Instant {
        match &self.schedule_type {
            ScheduleType::Immediate => self.created_at,
            ScheduleType::Delayed(d) => self.created_at + *d,
            ScheduleType::Recurring(d) => self.created_at + *d * (self.run_count as u32 + 1),
        }
    }

    /// Return the number of times this schedule has been triggered.
    #[must_use]
    pub fn run_count(&self) -> u64 {
        self.run_count
    }

    /// Record that this schedule has triggered once.
    pub fn mark_triggered(&mut self) {
        self.run_count += 1;
    }

    /// Returns `true` if the schedule is currently due (i.e. `next_run_at` is in the past).
    #[must_use]
    pub fn is_due(&self) -> bool {
        Instant::now() >= self.next_run_at()
    }
}

/// Manages a collection of [`BatchSchedule`] entries.
#[derive(Debug, Default)]
pub struct BatchScheduler {
    schedules: HashMap<String, BatchSchedule>,
}

impl BatchScheduler {
    /// Create an empty scheduler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a schedule. Returns `false` if a schedule with the same id already exists.
    pub fn schedule(&mut self, s: BatchSchedule) -> bool {
        if self.schedules.contains_key(&s.id) {
            return false;
        }
        self.schedules.insert(s.id.clone(), s);
        true
    }

    /// Remove a schedule by id. Returns `true` if it was present.
    pub fn remove(&mut self, id: &str) -> bool {
        self.schedules.remove(id).is_some()
    }

    /// Return ids of all schedules that are currently due.
    #[must_use]
    pub fn due_now(&self) -> Vec<&str> {
        self.schedules
            .values()
            .filter(|s| s.is_due())
            .map(|s| s.id.as_str())
            .collect()
    }

    /// Number of registered schedules.
    #[must_use]
    pub fn len(&self) -> usize {
        self.schedules.len()
    }

    /// Returns `true` when no schedules are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.schedules.is_empty()
    }

    /// Get a reference to a schedule by id.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&BatchSchedule> {
        self.schedules.get(id)
    }

    /// Get a mutable reference to a schedule by id.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut BatchSchedule> {
        self.schedules.get_mut(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_type_immediate_not_recurring() {
        assert!(!ScheduleType::Immediate.is_recurring());
        assert!(ScheduleType::Immediate.is_immediate());
    }

    #[test]
    fn test_schedule_type_recurring_is_recurring() {
        let st = ScheduleType::Recurring(Duration::from_secs(60));
        assert!(st.is_recurring());
        assert!(!st.is_immediate());
    }

    #[test]
    fn test_schedule_type_delayed_not_recurring() {
        let st = ScheduleType::Delayed(Duration::from_secs(30));
        assert!(!st.is_recurring());
        assert!(!st.is_immediate());
    }

    #[test]
    fn test_schedule_type_interval_immediate_none() {
        assert!(ScheduleType::Immediate.interval().is_none());
    }

    #[test]
    fn test_schedule_type_interval_delayed_some() {
        let d = Duration::from_secs(10);
        assert_eq!(ScheduleType::Delayed(d).interval(), Some(d));
    }

    #[test]
    fn test_schedule_type_interval_recurring_some() {
        let d = Duration::from_secs(5);
        assert_eq!(ScheduleType::Recurring(d).interval(), Some(d));
    }

    #[test]
    fn test_batch_schedule_immediate_is_due_immediately() {
        let s = BatchSchedule::new("s1", "Immediate run", ScheduleType::Immediate);
        assert!(s.is_due());
    }

    #[test]
    fn test_batch_schedule_delayed_not_immediately_due() {
        let s = BatchSchedule::new(
            "s2",
            "Delayed run",
            ScheduleType::Delayed(Duration::from_secs(3600)),
        );
        assert!(!s.is_due());
    }

    #[test]
    fn test_batch_schedule_run_count_increments() {
        let mut s = BatchSchedule::new("s3", "Counter", ScheduleType::Immediate);
        assert_eq!(s.run_count(), 0);
        s.mark_triggered();
        s.mark_triggered();
        assert_eq!(s.run_count(), 2);
    }

    #[test]
    fn test_batch_schedule_next_run_at_immediate() {
        let s = BatchSchedule::new("s4", "Now", ScheduleType::Immediate);
        // next_run_at should be at or before now
        assert!(s.next_run_at() <= Instant::now());
    }

    #[test]
    fn test_scheduler_schedule_and_len() {
        let mut sched = BatchScheduler::new();
        assert!(sched.is_empty());
        let s = BatchSchedule::new("id1", "first", ScheduleType::Immediate);
        assert!(sched.schedule(s));
        assert_eq!(sched.len(), 1);
        assert!(!sched.is_empty());
    }

    #[test]
    fn test_scheduler_schedule_duplicate_returns_false() {
        let mut sched = BatchScheduler::new();
        let s1 = BatchSchedule::new("dup", "first", ScheduleType::Immediate);
        let s2 = BatchSchedule::new("dup", "second", ScheduleType::Immediate);
        assert!(sched.schedule(s1));
        assert!(!sched.schedule(s2));
        assert_eq!(sched.len(), 1);
    }

    #[test]
    fn test_scheduler_remove() {
        let mut sched = BatchScheduler::new();
        sched.schedule(BatchSchedule::new(
            "r1",
            "removable",
            ScheduleType::Immediate,
        ));
        assert!(sched.remove("r1"));
        assert!(!sched.remove("r1")); // already removed
        assert!(sched.is_empty());
    }

    #[test]
    fn test_scheduler_due_now_includes_immediate() {
        let mut sched = BatchScheduler::new();
        sched.schedule(BatchSchedule::new(
            "now",
            "immediate",
            ScheduleType::Immediate,
        ));
        sched.schedule(BatchSchedule::new(
            "later",
            "delayed",
            ScheduleType::Delayed(Duration::from_secs(3600)),
        ));
        let due = sched.due_now();
        assert_eq!(due.len(), 1);
        assert_eq!(due[0], "now");
    }

    #[test]
    fn test_scheduler_get_and_get_mut() {
        let mut sched = BatchScheduler::new();
        sched.schedule(BatchSchedule::new(
            "g1",
            "gettable",
            ScheduleType::Immediate,
        ));
        assert!(sched.get("g1").is_some());
        assert!(sched.get("missing").is_none());
        let entry = sched.get_mut("g1").expect("get_mut should succeed");
        entry.mark_triggered();
        assert_eq!(sched.get("g1").expect("failed to get value").run_count(), 1);
    }
}
