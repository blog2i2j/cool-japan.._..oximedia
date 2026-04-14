#![allow(dead_code)]
//! Priority queue for farm job scheduling.
//!
//! Implements a max-heap based priority queue that orders farm jobs by
//! their [`crate::Priority`] level, submission time, and optional
//! deadline. Jobs with higher priority are dequeued first; among jobs
//! with equal priority the oldest submission wins (FIFO within a
//! priority tier).

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Queue entry
// ---------------------------------------------------------------------------

/// A single entry in the priority queue.
#[derive(Debug, Clone)]
pub struct PriorityEntry {
    /// Numeric priority (higher = more urgent).
    pub priority: u32,
    /// Monotonic submission timestamp for FIFO tie-breaking.
    pub submitted_at: Instant,
    /// Optional hard deadline.
    pub deadline: Option<Instant>,
    /// Opaque job identifier.
    pub job_id: String,
    /// Estimated processing duration (for scheduling hints).
    pub estimated_duration: Duration,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.job_id == other.job_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first
        self.priority
            .cmp(&other.priority)
            // Earlier submission first (reverse because BinaryHeap is max-heap)
            .then_with(|| other.submitted_at.cmp(&self.submitted_at))
    }
}

// ---------------------------------------------------------------------------
// Priority queue
// ---------------------------------------------------------------------------

/// A max-heap priority queue for scheduling farm jobs.
#[derive(Debug)]
pub struct FarmPriorityQueue {
    /// The underlying max-heap.
    heap: BinaryHeap<PriorityEntry>,
    /// Maximum capacity (0 = unlimited).
    capacity: usize,
}

impl FarmPriorityQueue {
    /// Create a new unbounded priority queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity: 0,
        }
    }

    /// Create a new priority queue with a maximum capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            capacity,
        }
    }

    /// Push an entry onto the queue.
    ///
    /// Returns `false` if the queue is at capacity and the entry was not added.
    pub fn push(&mut self, entry: PriorityEntry) -> bool {
        if self.capacity > 0 && self.heap.len() >= self.capacity {
            return false;
        }
        self.heap.push(entry);
        true
    }

    /// Pop the highest-priority entry.
    pub fn pop(&mut self) -> Option<PriorityEntry> {
        self.heap.pop()
    }

    /// Peek at the highest-priority entry without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&PriorityEntry> {
        self.heap.peek()
    }

    /// Number of entries currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Remove all entries from the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Drain all entries that have passed their deadline.
    ///
    /// Returns the expired entries.
    pub fn drain_expired(&mut self) -> Vec<PriorityEntry> {
        let now = Instant::now();
        let mut remaining = BinaryHeap::new();
        let mut expired = Vec::new();
        while let Some(entry) = self.heap.pop() {
            if let Some(deadline) = entry.deadline {
                if now > deadline {
                    expired.push(entry);
                    continue;
                }
            }
            remaining.push(entry);
        }
        self.heap = remaining;
        expired
    }

    /// Count entries at a specific priority level.
    #[must_use]
    pub fn count_at_priority(&self, priority: u32) -> usize {
        self.heap.iter().filter(|e| e.priority == priority).count()
    }

    /// Get the maximum configured capacity (0 means unlimited).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Update the priority of the entry identified by `job_id`.
    ///
    /// Internally this performs a remove + reinsert using the standard
    /// `BinaryHeap` rebuild strategy: rebuild the entire heap from the
    /// remaining entries with the modified priority.  This is O(n) but
    /// acceptable because the queue is bounded by `capacity`.
    ///
    /// Returns `true` if the entry was found and updated, `false` otherwise.
    pub fn update_priority(&mut self, job_id: &str, new_priority: u32) -> bool {
        // Drain the heap into a Vec, modify the target, then rebuild.
        let mut entries: Vec<PriorityEntry> = self.heap.drain().collect();
        let found = entries.iter_mut().find(|e| e.job_id == job_id);

        match found {
            Some(entry) => {
                entry.priority = new_priority;
                self.heap = entries.into_iter().collect();
                true
            }
            None => {
                // Restore all entries if not found.
                self.heap = entries.into_iter().collect();
                false
            }
        }
    }
}

impl Default for FarmPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Builder for [`PriorityEntry`].
#[derive(Debug)]
pub struct PriorityEntryBuilder {
    /// Numeric priority.
    priority: u32,
    /// Job identifier.
    job_id: String,
    /// Estimated processing time.
    estimated_duration: Duration,
    /// Optional hard deadline.
    deadline: Option<Instant>,
}

impl PriorityEntryBuilder {
    /// Create a new builder with required fields.
    #[must_use]
    pub fn new(job_id: impl Into<String>, priority: u32) -> Self {
        Self {
            priority,
            job_id: job_id.into(),
            estimated_duration: Duration::ZERO,
            deadline: None,
        }
    }

    /// Set the estimated processing duration.
    #[must_use]
    pub fn estimated_duration(mut self, d: Duration) -> Self {
        self.estimated_duration = d;
        self
    }

    /// Set the hard deadline.
    #[must_use]
    pub fn deadline(mut self, d: Instant) -> Self {
        self.deadline = Some(d);
        self
    }

    /// Build the [`PriorityEntry`].
    #[must_use]
    pub fn build(self) -> PriorityEntry {
        PriorityEntry {
            priority: self.priority,
            submitted_at: Instant::now(),
            deadline: self.deadline,
            job_id: self.job_id,
            estimated_duration: self.estimated_duration,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn make_entry(job_id: &str, priority: u32) -> PriorityEntry {
        PriorityEntry {
            priority,
            submitted_at: Instant::now(),
            deadline: None,
            job_id: job_id.to_string(),
            estimated_duration: Duration::from_secs(10),
        }
    }

    #[test]
    fn test_push_pop_single() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("j1", 1));
        let e = q.pop().unwrap();
        assert_eq!(e.job_id, "j1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("low", 1));
        q.push(make_entry("high", 3));
        q.push(make_entry("mid", 2));
        assert_eq!(q.pop().unwrap().job_id, "high");
        assert_eq!(q.pop().unwrap().job_id, "mid");
        assert_eq!(q.pop().unwrap().job_id, "low");
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut q = FarmPriorityQueue::new();
        let e1 = make_entry("first", 5);
        thread::sleep(Duration::from_millis(2));
        let e2 = make_entry("second", 5);
        q.push(e1);
        q.push(e2);
        assert_eq!(q.pop().unwrap().job_id, "first");
        assert_eq!(q.pop().unwrap().job_id, "second");
    }

    #[test]
    fn test_capacity_limit() {
        let mut q = FarmPriorityQueue::with_capacity(2);
        assert!(q.push(make_entry("j1", 1)));
        assert!(q.push(make_entry("j2", 1)));
        assert!(!q.push(make_entry("j3", 1))); // rejected
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_peek() {
        let mut q = FarmPriorityQueue::new();
        assert!(q.peek().is_none());
        q.push(make_entry("j1", 10));
        assert_eq!(q.peek().unwrap().job_id, "j1");
        assert_eq!(q.len(), 1); // peek didn't remove
    }

    #[test]
    fn test_clear() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("j1", 1));
        q.push(make_entry("j2", 2));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_drain_expired() {
        let mut q = FarmPriorityQueue::new();
        // Entry that is already expired
        let mut expired_entry = make_entry("expired", 5);
        expired_entry.deadline = Some(Instant::now() - Duration::from_secs(1));
        q.push(expired_entry);
        // Entry that is still valid
        let mut valid_entry = make_entry("valid", 3);
        valid_entry.deadline = Some(Instant::now() + Duration::from_secs(3600));
        q.push(valid_entry);
        // Entry with no deadline (always valid)
        q.push(make_entry("no_deadline", 1));

        let expired = q.drain_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].job_id, "expired");
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_count_at_priority() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("a", 1));
        q.push(make_entry("b", 1));
        q.push(make_entry("c", 2));
        assert_eq!(q.count_at_priority(1), 2);
        assert_eq!(q.count_at_priority(2), 1);
        assert_eq!(q.count_at_priority(99), 0);
    }

    #[test]
    fn test_default() {
        let q = FarmPriorityQueue::default();
        assert!(q.is_empty());
        assert_eq!(q.capacity(), 0);
    }

    #[test]
    fn test_builder_basic() {
        let entry = PriorityEntryBuilder::new("build-job", 7)
            .estimated_duration(Duration::from_secs(60))
            .build();
        assert_eq!(entry.job_id, "build-job");
        assert_eq!(entry.priority, 7);
        assert_eq!(entry.estimated_duration, Duration::from_secs(60));
        assert!(entry.deadline.is_none());
    }

    #[test]
    fn test_builder_with_deadline() {
        let dl = Instant::now() + Duration::from_secs(300);
        let entry = PriorityEntryBuilder::new("dl-job", 2).deadline(dl).build();
        assert!(entry.deadline.is_some());
    }

    #[test]
    fn test_pop_empty() {
        let mut q = FarmPriorityQueue::new();
        assert!(q.pop().is_none());
    }

    // ── update_priority tests (Task G) ────────────────────────────────────────

    #[test]
    fn test_update_priority_changes_order() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("low", 1));
        q.push(make_entry("high", 3));

        // Boost "low" above "high".
        assert!(q.update_priority("low", 5));
        // "low" (now priority 5) should come out first.
        assert_eq!(q.pop().expect("pop").job_id, "low");
        assert_eq!(q.pop().expect("pop").job_id, "high");
    }

    #[test]
    fn test_update_priority_demotes_entry() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("alpha", 10));
        q.push(make_entry("beta", 5));

        // Demote "alpha" below "beta".
        assert!(q.update_priority("alpha", 1));
        assert_eq!(q.pop().expect("pop").job_id, "beta");
        assert_eq!(q.pop().expect("pop").job_id, "alpha");
    }

    #[test]
    fn test_update_priority_not_found_returns_false() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("exists", 5));
        assert!(!q.update_priority("ghost", 10));
        // Existing entry still intact.
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_update_priority_preserves_len() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("a", 1));
        q.push(make_entry("b", 2));
        q.push(make_entry("c", 3));
        q.update_priority("b", 5);
        assert_eq!(q.len(), 3);
    }

    #[test]
    fn test_update_priority_same_value_is_noop() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("x", 7));
        assert!(q.update_priority("x", 7));
        assert_eq!(q.pop().expect("pop").priority, 7);
    }

    #[test]
    fn test_ordering_correctness_high_before_low() {
        let mut q = FarmPriorityQueue::new();
        for i in 0u32..5 {
            q.push(make_entry(&format!("j{i}"), i));
        }
        let mut last_priority = u32::MAX;
        while let Some(e) = q.pop() {
            assert!(
                e.priority <= last_priority,
                "priority {} appeared after {}",
                e.priority,
                last_priority
            );
            last_priority = e.priority;
        }
    }

    #[test]
    fn test_fifo_same_priority_stable() {
        let mut q = FarmPriorityQueue::new();
        let first = make_entry("first", 5);
        thread::sleep(Duration::from_millis(2));
        let second = make_entry("second", 5);
        q.push(first);
        q.push(second);
        assert_eq!(q.pop().expect("pop").job_id, "first");
        assert_eq!(q.pop().expect("pop").job_id, "second");
    }

    #[test]
    fn test_len_and_empty() {
        let mut q = FarmPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
        q.push(make_entry("a", 1));
        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
        q.pop();
        assert!(q.is_empty());
    }

    #[test]
    fn test_update_priority_on_empty_queue_returns_false() {
        let mut q = FarmPriorityQueue::new();
        assert!(!q.update_priority("nobody", 10));
        assert!(q.is_empty());
    }

    #[test]
    fn test_update_priority_multiple_entries_same_priority_group() {
        let mut q = FarmPriorityQueue::new();
        q.push(make_entry("a", 5));
        q.push(make_entry("b", 5));
        q.push(make_entry("c", 5));

        // Boost "c" to priority 9.
        assert!(q.update_priority("c", 9));
        assert_eq!(q.pop().expect("pop").job_id, "c");
        // Remaining two should still have priority 5.
        assert_eq!(q.pop().expect("pop").priority, 5);
        assert_eq!(q.pop().expect("pop").priority, 5);
    }
}
