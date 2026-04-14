#![allow(dead_code)]
// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Work-stealing scheduler for distributing jobs across workers.
//!
//! Each worker owns a local double-ended queue (deque).  When a worker's
//! local queue is empty it attempts to *steal* work from other workers'
//! queues, starting with the most loaded peer.  This minimises idle time
//! and naturally balances load without a central dispatcher bottleneck.
//!
//! # Design
//!
//! * **`WorkerDeque<T>`** – per-worker deque supporting `push_back`,
//!   `pop_front` (owner takes from front), and `steal_back` (thief takes
//!   from back).
//! * **`WorkStealingPool<T>`** – manages a collection of worker deques and
//!   exposes `submit`, `try_steal`, and aggregate statistics.
//! * **`StealPolicy`** – configurable policies for when and how stealing
//!   occurs (threshold, max batch, randomised victim selection, etc.).

use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// Steal policy
// ---------------------------------------------------------------------------

/// Policy controlling work-stealing behavior.
#[derive(Debug, Clone)]
pub struct StealPolicy {
    /// Minimum number of items a victim must have before a steal is allowed.
    pub victim_min_items: usize,
    /// Maximum number of items to steal in one batch.
    pub max_steal_batch: usize,
    /// If true, the thief steals from the most loaded worker; otherwise it
    /// round-robins across peers.
    pub steal_from_most_loaded: bool,
    /// Maximum number of steal attempts before giving up.
    pub max_steal_attempts: usize,
}

impl Default for StealPolicy {
    fn default() -> Self {
        Self {
            victim_min_items: 2,
            max_steal_batch: 4,
            steal_from_most_loaded: true,
            max_steal_attempts: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Worker deque
// ---------------------------------------------------------------------------

/// A double-ended work queue owned by a single worker.
///
/// The owner pushes and pops from the front; thieves steal from the back.
#[derive(Debug)]
pub struct WorkerDeque<T> {
    id: usize,
    deque: VecDeque<T>,
    items_pushed: u64,
    items_popped: u64,
    items_stolen_from: u64,
}

impl<T> WorkerDeque<T> {
    /// Create a new empty deque for the worker with the given `id`.
    #[must_use]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            deque: VecDeque::new(),
            items_pushed: 0,
            items_popped: 0,
            items_stolen_from: 0,
        }
    }

    /// Worker ID.
    #[must_use]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Number of items currently in the deque.
    #[must_use]
    pub fn len(&self) -> usize {
        self.deque.len()
    }

    /// Whether the deque is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    /// Owner pushes a new item to the back of the deque.
    pub fn push_back(&mut self, item: T) {
        self.deque.push_back(item);
        self.items_pushed += 1;
    }

    /// Owner pops an item from the front (FIFO for the owner).
    pub fn pop_front(&mut self) -> Option<T> {
        let item = self.deque.pop_front();
        if item.is_some() {
            self.items_popped += 1;
        }
        item
    }

    /// Thief steals up to `count` items from the **back** of the deque.
    ///
    /// Returns the stolen items (may be fewer than `count` if the deque
    /// doesn't have enough).  A minimum of `keep` items are always retained
    /// so the owner isn't fully drained.
    pub fn steal_back(&mut self, count: usize, keep: usize) -> Vec<T> {
        let available = self.deque.len().saturating_sub(keep);
        let to_steal = count.min(available);
        let mut stolen = Vec::with_capacity(to_steal);
        for _ in 0..to_steal {
            if let Some(item) = self.deque.pop_back() {
                stolen.push(item);
                self.items_stolen_from += 1;
            }
        }
        stolen
    }

    /// Total items ever pushed.
    #[must_use]
    pub fn total_pushed(&self) -> u64 {
        self.items_pushed
    }

    /// Total items ever popped by the owner.
    #[must_use]
    pub fn total_popped(&self) -> u64 {
        self.items_popped
    }

    /// Total items stolen from this deque by other workers.
    #[must_use]
    pub fn total_stolen_from(&self) -> u64 {
        self.items_stolen_from
    }
}

// ---------------------------------------------------------------------------
// Steal result
// ---------------------------------------------------------------------------

/// Outcome of a steal attempt.
#[derive(Debug)]
pub enum StealResult<T> {
    /// Successfully stole items from the given worker.
    Success {
        victim_id: usize,
        items: Vec<T>,
    },
    /// No items could be stolen (all peers too light or empty).
    Empty,
    /// The thief's own queue is not empty — stealing unnecessary.
    NotNeeded,
}

impl<T: fmt::Debug> fmt::Display for StealResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success { victim_id, items } => {
                write!(f, "Stole {} items from worker {}", items.len(), victim_id)
            }
            Self::Empty => write!(f, "No items available to steal"),
            Self::NotNeeded => write!(f, "Steal not needed (local queue non-empty)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------

/// Aggregate statistics for the work-stealing pool.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total items submitted across all workers.
    pub total_submitted: u64,
    /// Total items completed (popped by owner).
    pub total_completed: u64,
    /// Total items stolen.
    pub total_stolen: u64,
    /// Total steal attempts.
    pub total_steal_attempts: u64,
    /// Total successful steals.
    pub total_successful_steals: u64,
    /// Number of workers.
    pub worker_count: usize,
}

// ---------------------------------------------------------------------------
// Work-stealing pool
// ---------------------------------------------------------------------------

/// A pool of worker deques with work-stealing support.
#[derive(Debug)]
pub struct WorkStealingPool<T> {
    workers: Vec<WorkerDeque<T>>,
    policy: StealPolicy,
    stats: PoolStats,
}

impl<T: fmt::Debug> WorkStealingPool<T> {
    /// Create a pool with `worker_count` workers and the default steal policy.
    #[must_use]
    pub fn new(worker_count: usize) -> Self {
        Self::with_policy(worker_count, StealPolicy::default())
    }

    /// Create a pool with a custom steal policy.
    #[must_use]
    pub fn with_policy(worker_count: usize, policy: StealPolicy) -> Self {
        let workers = (0..worker_count).map(WorkerDeque::new).collect();
        Self {
            workers,
            policy,
            stats: PoolStats {
                worker_count,
                ..Default::default()
            },
        }
    }

    /// Number of workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Submit an item to a specific worker.
    ///
    /// Returns `Ok(())` if the worker exists, `Err(item)` if the worker ID
    /// is out of range.
    pub fn submit_to(&mut self, worker_id: usize, item: T) -> Result<(), T> {
        if worker_id >= self.workers.len() {
            return Err(item);
        }
        self.workers[worker_id].push_back(item);
        self.stats.total_submitted += 1;
        Ok(())
    }

    /// Submit an item to the least-loaded worker.
    pub fn submit(&mut self, item: T) {
        if self.workers.is_empty() {
            return;
        }
        let target = self
            .workers
            .iter()
            .enumerate()
            .min_by_key(|(_, w)| w.len())
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.workers[target].push_back(item);
        self.stats.total_submitted += 1;
    }

    /// Worker `thief_id` pops from its own queue. If empty, tries to steal.
    pub fn take(&mut self, thief_id: usize) -> Option<T> {
        if thief_id >= self.workers.len() {
            return None;
        }
        // Try own queue first.
        if let Some(item) = self.workers[thief_id].pop_front() {
            self.stats.total_completed += 1;
            return Some(item);
        }
        // Own queue empty — try stealing.
        let stolen = self.try_steal(thief_id);
        match stolen {
            StealResult::Success { items, .. } => {
                let mut iter = items.into_iter();
                let first = iter.next();
                // Push remaining stolen items into thief's local queue.
                for item in iter {
                    self.workers[thief_id].push_back(item);
                }
                if first.is_some() {
                    self.stats.total_completed += 1;
                }
                first
            }
            _ => None,
        }
    }

    /// Attempt to steal work for `thief_id` from the best victim.
    pub fn try_steal(&mut self, thief_id: usize) -> StealResult<T> {
        if thief_id >= self.workers.len() {
            return StealResult::Empty;
        }

        if !self.workers[thief_id].is_empty() {
            return StealResult::NotNeeded;
        }

        self.stats.total_steal_attempts += 1;

        // Find best victim.
        let victim_id = if self.policy.steal_from_most_loaded {
            self.find_most_loaded_victim(thief_id)
        } else {
            self.find_round_robin_victim(thief_id)
        };

        let victim_id = match victim_id {
            Some(id) => id,
            None => return StealResult::Empty,
        };

        let items = self.workers[victim_id].steal_back(
            self.policy.max_steal_batch,
            self.policy.victim_min_items.saturating_sub(1),
        );

        if items.is_empty() {
            StealResult::Empty
        } else {
            self.stats.total_stolen += items.len() as u64;
            self.stats.total_successful_steals += 1;
            StealResult::Success { victim_id, items }
        }
    }

    /// Get the length of a worker's local queue.
    #[must_use]
    pub fn worker_queue_len(&self, worker_id: usize) -> Option<usize> {
        self.workers.get(worker_id).map(|w| w.len())
    }

    /// Total items across all worker queues.
    #[must_use]
    pub fn total_pending(&self) -> usize {
        self.workers.iter().map(|w| w.len()).sum()
    }

    /// Get aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Returns the distribution of queue lengths as a vec of `(worker_id, len)`.
    pub fn queue_distribution(&self) -> Vec<(usize, usize)> {
        self.workers.iter().map(|w| (w.id(), w.len())).collect()
    }

    /// Imbalance ratio: `max_len / avg_len`. Returns 0.0 if pool is empty.
    pub fn imbalance_ratio(&self) -> f64 {
        if self.workers.is_empty() {
            return 0.0;
        }
        let lengths: Vec<usize> = self.workers.iter().map(|w| w.len()).collect();
        let max = lengths.iter().copied().max().unwrap_or(0);
        let total: usize = lengths.iter().sum();
        let avg = total as f64 / self.workers.len() as f64;
        if avg < f64::EPSILON {
            0.0
        } else {
            max as f64 / avg
        }
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn find_most_loaded_victim(&self, thief_id: usize) -> Option<usize> {
        self.workers
            .iter()
            .enumerate()
            .filter(|(i, w)| *i != thief_id && w.len() >= self.policy.victim_min_items)
            .max_by_key(|(_, w)| w.len())
            .map(|(i, _)| i)
    }

    fn find_round_robin_victim(&self, thief_id: usize) -> Option<usize> {
        // Simple scan starting after thief_id.
        let n = self.workers.len();
        for offset in 1..n {
            let candidate = (thief_id + offset) % n;
            if self.workers[candidate].len() >= self.policy.victim_min_items {
                return Some(candidate);
            }
        }
        None
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_deque_push_pop() {
        let mut d = WorkerDeque::new(0);
        d.push_back(1);
        d.push_back(2);
        d.push_back(3);
        assert_eq!(d.len(), 3);
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(d.pop_front(), Some(2));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn test_worker_deque_steal() {
        let mut d = WorkerDeque::new(0);
        for i in 0..10 {
            d.push_back(i);
        }
        // Steal 3, keeping at least 2
        let stolen = d.steal_back(3, 2);
        assert_eq!(stolen.len(), 3);
        assert_eq!(d.len(), 7);
        // Stolen from back: 9, 8, 7
        assert_eq!(stolen, vec![9, 8, 7]);
    }

    #[test]
    fn test_worker_deque_steal_respects_keep() {
        let mut d = WorkerDeque::new(0);
        d.push_back(1);
        d.push_back(2);
        // Try to steal 5, but keep 2 means 0 available
        let stolen = d.steal_back(5, 2);
        assert!(stolen.is_empty());
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn test_pool_submit_and_take() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(3);
        pool.submit_to(0, 10).ok();
        pool.submit_to(0, 20).ok();
        assert_eq!(pool.take(0), Some(10));
        assert_eq!(pool.take(0), Some(20));
        assert_eq!(pool.take(0), None);
    }

    #[test]
    fn test_pool_submit_to_least_loaded() {
        let mut pool: WorkStealingPool<&str> = WorkStealingPool::new(3);
        // Worker 0 gets 5 items
        for _ in 0..5 {
            pool.submit_to(0, "a").ok();
        }
        // Auto-submit should go to worker 1 or 2 (both empty)
        pool.submit("b");
        assert!(pool.worker_queue_len(1) == Some(1) || pool.worker_queue_len(2) == Some(1));
    }

    #[test]
    fn test_pool_steal_from_most_loaded() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        // Load worker 0 with 10 items
        for i in 0..10 {
            pool.submit_to(0, i).ok();
        }
        // Worker 1 is empty, should steal
        let result = pool.try_steal(1);
        match result {
            StealResult::Success { victim_id, items } => {
                assert_eq!(victim_id, 0);
                assert!(!items.is_empty());
                assert!(items.len() <= 4); // default max_steal_batch
            }
            _ => panic!("Expected successful steal"),
        }
    }

    #[test]
    fn test_pool_steal_not_needed() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        pool.submit_to(0, 1).ok();
        // Worker 0 has items, so steal is not needed
        let result = pool.try_steal(0);
        assert!(matches!(result, StealResult::NotNeeded));
    }

    #[test]
    fn test_pool_take_with_steal() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        for i in 0..8 {
            pool.submit_to(0, i).ok();
        }
        // Worker 1 takes — should steal from worker 0
        let item = pool.take(1);
        assert!(item.is_some());
    }

    #[test]
    fn test_pool_stats() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        for i in 0..6 {
            pool.submit_to(0, i).ok();
        }
        pool.take(1); // steal
        let stats = pool.stats();
        assert_eq!(stats.total_submitted, 6);
        assert_eq!(stats.worker_count, 2);
        assert!(stats.total_steal_attempts >= 1);
    }

    #[test]
    fn test_pool_total_pending() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(3);
        pool.submit_to(0, 1).ok();
        pool.submit_to(1, 2).ok();
        pool.submit_to(2, 3).ok();
        assert_eq!(pool.total_pending(), 3);
        pool.take(0);
        assert_eq!(pool.total_pending(), 2);
    }

    #[test]
    fn test_pool_queue_distribution() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(3);
        pool.submit_to(0, 1).ok();
        pool.submit_to(0, 2).ok();
        pool.submit_to(2, 3).ok();
        let dist = pool.queue_distribution();
        assert_eq!(dist, vec![(0, 2), (1, 0), (2, 1)]);
    }

    #[test]
    fn test_pool_imbalance_ratio_balanced() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(3);
        pool.submit_to(0, 1).ok();
        pool.submit_to(1, 2).ok();
        pool.submit_to(2, 3).ok();
        // All queues have 1 item: ratio = 1/1 = 1.0
        let ratio = pool.imbalance_ratio();
        assert!((ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pool_imbalance_ratio_empty() {
        let pool: WorkStealingPool<i32> = WorkStealingPool::new(3);
        assert!((pool.imbalance_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_submit_to_invalid_worker() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        let result = pool.submit_to(5, 42);
        assert!(result.is_err());
        assert_eq!(result.err(), Some(42));
    }

    #[test]
    fn test_steal_empty_pool() {
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::new(2);
        let result = pool.try_steal(0);
        assert!(matches!(result, StealResult::Empty));
    }

    #[test]
    fn test_round_robin_victim_selection() {
        let policy = StealPolicy {
            victim_min_items: 1,
            max_steal_batch: 2,
            steal_from_most_loaded: false,
            max_steal_attempts: 4,
        };
        let mut pool: WorkStealingPool<i32> = WorkStealingPool::with_policy(3, policy);
        // Worker 1 has items
        pool.submit_to(1, 10).ok();
        pool.submit_to(1, 20).ok();
        // Worker 0 steals — round-robin starts at worker 1
        let result = pool.try_steal(0);
        match result {
            StealResult::Success { victim_id, .. } => assert_eq!(victim_id, 1),
            _ => panic!("Expected steal from worker 1"),
        }
    }

    #[test]
    fn test_worker_deque_statistics() {
        let mut d = WorkerDeque::new(42);
        d.push_back("a");
        d.push_back("b");
        d.push_back("c");
        d.pop_front();
        d.steal_back(1, 0);
        assert_eq!(d.total_pushed(), 3);
        assert_eq!(d.total_popped(), 1);
        assert_eq!(d.total_stolen_from(), 1);
        assert_eq!(d.id(), 42);
    }

    #[test]
    fn test_steal_result_display() {
        let success: StealResult<i32> = StealResult::Success {
            victim_id: 2,
            items: vec![1, 2, 3],
        };
        let msg = format!("{success}");
        assert!(msg.contains("3 items"));
        assert!(msg.contains("worker 2"));

        let empty: StealResult<i32> = StealResult::Empty;
        assert!(format!("{empty}").contains("No items"));
    }
}
