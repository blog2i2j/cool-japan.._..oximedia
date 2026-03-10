// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Schedule entries, scheduling policies, and time-window utilities.
//!
//! This module provides lightweight, pure-Rust scheduling primitives that
//! complement the higher-level [`crate::scheduler`] module. It operates
//! exclusively on millisecond timestamps so that it remains independent of any
//! particular clock source.

#![allow(dead_code)]

/// A single entry in a job schedule queue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleEntry {
    /// Unique job identifier.
    pub job_id: u64,
    /// Scheduling priority (higher = more urgent).
    pub priority: u32,
    /// Wall-clock time at which the entry was enqueued (ms since epoch).
    pub scheduled_at_ms: u64,
    /// Optional deadline; if `Some(t)`, the job must start before `t`.
    pub deadline_ms: Option<u64>,
}

impl ScheduleEntry {
    /// Create a new schedule entry.
    #[must_use]
    pub fn new(job_id: u64, priority: u32, scheduled_at_ms: u64, deadline_ms: Option<u64>) -> Self {
        Self {
            job_id,
            priority,
            scheduled_at_ms,
            deadline_ms,
        }
    }

    /// Returns `true` if the entry has a deadline and `now_ms` has passed it.
    #[must_use]
    pub fn is_overdue(&self, now_ms: u64) -> bool {
        self.deadline_ms.map_or(false, |d| now_ms > d)
    }

    /// Waiting time in milliseconds from `scheduled_at_ms` to `now_ms`.
    #[must_use]
    pub fn wait_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.scheduled_at_ms)
    }
}

/// Strategy used by [`EntryScheduler`] to choose the next entry to dequeue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulePolicy {
    /// First-in, first-out – oldest entry leaves first.
    Fifo,
    /// Highest-priority entry leaves first (ties broken by arrival order).
    Priority,
    /// Earliest-deadline-first; entries without a deadline are treated last.
    Edf,
    /// Strict round-robin across all waiting entries.
    RoundRobin,
}

impl SchedulePolicy {
    /// Short lowercase name for the policy.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Fifo => "fifo",
            Self::Priority => "priority",
            Self::Edf => "edf",
            Self::RoundRobin => "round_robin",
        }
    }
}

/// Scheduler that maintains a queue of [`ScheduleEntry`] values and dequeues
/// them according to the configured [`SchedulePolicy`].
#[derive(Debug)]
pub struct EntryScheduler {
    /// Active scheduling policy.
    pub policy: SchedulePolicy,
    /// Pending entries (insertion-ordered).
    queue: Vec<ScheduleEntry>,
    /// Internal cursor for round-robin dispatch.
    round_robin_idx: usize,
}

impl EntryScheduler {
    /// Create a new scheduler with the given policy.
    #[must_use]
    pub fn new(policy: SchedulePolicy) -> Self {
        Self {
            policy,
            queue: Vec::new(),
            round_robin_idx: 0,
        }
    }

    /// Add an entry to the back of the internal queue.
    pub fn enqueue(&mut self, entry: ScheduleEntry) {
        self.queue.push(entry);
    }

    /// Remove and return the next entry according to the active policy.
    pub fn dequeue(&mut self) -> Option<ScheduleEntry> {
        if self.queue.is_empty() {
            return None;
        }
        let idx = match self.policy {
            SchedulePolicy::Fifo => 0,
            SchedulePolicy::Priority => {
                self.queue
                    .iter()
                    .enumerate()
                    .max_by_key(|(pos, e)| (e.priority, u64::MAX - *pos as u64))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            SchedulePolicy::Edf => {
                // Entries with deadlines come first; among them the earliest deadline wins.
                // Entries without deadlines are treated as if their deadline is u64::MAX.
                self.queue
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e.deadline_ms.unwrap_or(u64::MAX))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            SchedulePolicy::RoundRobin => {
                let len = self.queue.len();
                let idx = self.round_robin_idx % len;
                self.round_robin_idx = self.round_robin_idx.wrapping_add(1);
                idx
            }
        };
        Some(self.queue.remove(idx))
    }

    /// Number of entries currently waiting.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Entries whose deadline has been exceeded at `now_ms`.
    #[must_use]
    pub fn overdue_jobs(&self, now_ms: u64) -> Vec<&ScheduleEntry> {
        self.queue.iter().filter(|e| e.is_overdue(now_ms)).collect()
    }

    /// Drain all overdue entries from the queue and return them.
    pub fn drain_overdue(&mut self, now_ms: u64) -> Vec<ScheduleEntry> {
        let mut overdue = Vec::new();
        let mut i = 0;
        while i < self.queue.len() {
            if self.queue[i].is_overdue(now_ms) {
                overdue.push(self.queue.remove(i));
            } else {
                i += 1;
            }
        }
        overdue
    }
}

/// A half-open time interval `[start_ms, end_ms)` expressed in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeWindow {
    /// Inclusive start of the window.
    pub start_ms: u64,
    /// Exclusive end of the window.
    pub end_ms: u64,
}

impl TimeWindow {
    /// Create a new time window. Panics in debug mode if `start_ms > end_ms`.
    #[must_use]
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        debug_assert!(start_ms <= end_ms, "start_ms must not exceed end_ms");
        Self { start_ms, end_ms }
    }

    /// Duration of the window in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }

    /// Returns `true` if this window overlaps with `other`.
    ///
    /// Two windows overlap when neither ends before the other starts.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Returns `true` if `ts_ms` falls within `[start_ms, end_ms)`.
    #[must_use]
    pub fn contains(&self, ts_ms: u64) -> bool {
        ts_ms >= self.start_ms && ts_ms < self.end_ms
    }

    /// Returns the intersection of this window with `other`, or `None` if they
    /// do not overlap.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start = self.start_ms.max(other.start_ms);
        let end = self.end_ms.min(other.end_ms);
        if start < end {
            Some(Self::new(start, end))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScheduleEntry ─────────────────────────────────────────────────────────

    #[test]
    fn test_entry_not_overdue_without_deadline() {
        let e = ScheduleEntry::new(1, 10, 0, None);
        assert!(!e.is_overdue(u64::MAX));
    }

    #[test]
    fn test_entry_overdue_with_deadline() {
        let e = ScheduleEntry::new(1, 10, 0, Some(5000));
        assert!(!e.is_overdue(5000)); // boundary: not overdue AT deadline
        assert!(e.is_overdue(5001));
    }

    #[test]
    fn test_entry_wait_ms() {
        let e = ScheduleEntry::new(1, 10, 1000, None);
        assert_eq!(e.wait_ms(3000), 2000);
        assert_eq!(e.wait_ms(500), 0); // saturating
    }

    // ── SchedulePolicy ────────────────────────────────────────────────────────

    #[test]
    fn test_policy_names() {
        assert_eq!(SchedulePolicy::Fifo.name(), "fifo");
        assert_eq!(SchedulePolicy::Priority.name(), "priority");
        assert_eq!(SchedulePolicy::Edf.name(), "edf");
        assert_eq!(SchedulePolicy::RoundRobin.name(), "round_robin");
    }

    // ── EntryScheduler – FIFO ─────────────────────────────────────────────────

    #[test]
    fn test_fifo_order() {
        let mut s = EntryScheduler::new(SchedulePolicy::Fifo);
        s.enqueue(ScheduleEntry::new(1, 5, 100, None));
        s.enqueue(ScheduleEntry::new(2, 5, 200, None));
        s.enqueue(ScheduleEntry::new(3, 5, 300, None));

        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 1);
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 2);
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 3);
        assert!(s.dequeue().is_none());
    }

    // ── EntryScheduler – Priority ─────────────────────────────────────────────

    #[test]
    fn test_priority_order() {
        let mut s = EntryScheduler::new(SchedulePolicy::Priority);
        s.enqueue(ScheduleEntry::new(1, 1, 100, None)); // low prio
        s.enqueue(ScheduleEntry::new(2, 9, 200, None)); // high prio
        s.enqueue(ScheduleEntry::new(3, 5, 300, None)); // mid prio

        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 2); // highest prio first
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 3);
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 1);
    }

    // ── EntryScheduler – EDF ──────────────────────────────────────────────────

    #[test]
    fn test_edf_order() {
        let mut s = EntryScheduler::new(SchedulePolicy::Edf);
        s.enqueue(ScheduleEntry::new(1, 1, 100, Some(5000)));
        s.enqueue(ScheduleEntry::new(2, 1, 100, Some(1000))); // earliest deadline
        s.enqueue(ScheduleEntry::new(3, 1, 100, None)); // no deadline – last

        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 2);
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 1);
        assert_eq!(s.dequeue().expect("dequeue should succeed").job_id, 3);
    }

    // ── EntryScheduler – RoundRobin ───────────────────────────────────────────

    #[test]
    fn test_round_robin_order() {
        let mut s = EntryScheduler::new(SchedulePolicy::RoundRobin);
        s.enqueue(ScheduleEntry::new(10, 1, 0, None));
        s.enqueue(ScheduleEntry::new(20, 1, 0, None));
        s.enqueue(ScheduleEntry::new(30, 1, 0, None));

        // RoundRobin picks index 0 first (round_robin_idx starts at 0)
        let first = s.dequeue().expect("first should be valid").job_id;
        // After first removal the queue shrinks; just check we get something valid
        let second = s.dequeue().expect("second should be valid").job_id;
        let third = s.dequeue().expect("third should be valid").job_id;
        let ids = [first, second, third];
        let mut sorted = ids;
        sorted.sort_unstable();
        assert_eq!(sorted, [10, 20, 30]);
    }

    // ── EntryScheduler – overdue ──────────────────────────────────────────────

    #[test]
    fn test_overdue_jobs() {
        let mut s = EntryScheduler::new(SchedulePolicy::Fifo);
        s.enqueue(ScheduleEntry::new(1, 1, 0, Some(100)));
        s.enqueue(ScheduleEntry::new(2, 1, 0, Some(9999)));
        s.enqueue(ScheduleEntry::new(3, 1, 0, None));

        let overdue = s.overdue_jobs(500);
        assert_eq!(overdue.len(), 1);
        assert_eq!(overdue[0].job_id, 1);
    }

    #[test]
    fn test_drain_overdue() {
        let mut s = EntryScheduler::new(SchedulePolicy::Fifo);
        s.enqueue(ScheduleEntry::new(1, 1, 0, Some(100)));
        s.enqueue(ScheduleEntry::new(2, 1, 0, Some(200)));
        s.enqueue(ScheduleEntry::new(3, 1, 0, Some(9999)));

        let drained = s.drain_overdue(500);
        assert_eq!(drained.len(), 2);
        assert_eq!(s.pending_count(), 1);
    }

    #[test]
    fn test_pending_count() {
        let mut s = EntryScheduler::new(SchedulePolicy::Fifo);
        assert_eq!(s.pending_count(), 0);
        s.enqueue(ScheduleEntry::new(1, 1, 0, None));
        assert_eq!(s.pending_count(), 1);
        s.dequeue();
        assert_eq!(s.pending_count(), 0);
    }

    // ── TimeWindow ────────────────────────────────────────────────────────────

    #[test]
    fn test_time_window_duration() {
        let w = TimeWindow::new(1000, 3000);
        assert_eq!(w.duration_ms(), 2000);
    }

    #[test]
    fn test_time_window_zero_duration() {
        let w = TimeWindow::new(500, 500);
        assert_eq!(w.duration_ms(), 0);
    }

    #[test]
    fn test_time_window_overlaps_true() {
        let a = TimeWindow::new(0, 200);
        let b = TimeWindow::new(100, 300);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_time_window_overlaps_false() {
        let a = TimeWindow::new(0, 100);
        let b = TimeWindow::new(100, 200);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_time_window_contains() {
        let w = TimeWindow::new(100, 200);
        assert!(w.contains(100));
        assert!(w.contains(150));
        assert!(!w.contains(200)); // exclusive end
        assert!(!w.contains(99));
    }

    #[test]
    fn test_time_window_intersection_some() {
        let a = TimeWindow::new(0, 300);
        let b = TimeWindow::new(100, 500);
        let inter = a.intersection(&b).expect("inter should be valid");
        assert_eq!(inter.start_ms, 100);
        assert_eq!(inter.end_ms, 300);
    }

    #[test]
    fn test_time_window_intersection_none() {
        let a = TimeWindow::new(0, 100);
        let b = TimeWindow::new(100, 200);
        assert!(a.intersection(&b).is_none());
    }
}
