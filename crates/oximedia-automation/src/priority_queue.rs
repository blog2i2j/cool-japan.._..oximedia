#![allow(dead_code)]
//! Priority-based event and task queue for broadcast automation scheduling.
//!
//! Manages a priority queue where broadcast events and automation tasks are
//! ordered by urgency and scheduled time. Supports multiple priority tiers,
//! deadline enforcement, coalescing of duplicate events, and fair scheduling
//! across channels.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

/// Priority tier for queued items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityTier {
    /// Critical: EAS alerts, failover events.
    Critical,
    /// High: live switching, timed events.
    High,
    /// Normal: scheduled playlist items.
    Normal,
    /// Low: background maintenance tasks.
    Low,
    /// Deferred: non-urgent housekeeping.
    Deferred,
}

impl PriorityTier {
    /// Returns the numeric weight of this tier (higher = more urgent).
    #[allow(clippy::cast_precision_loss)]
    fn weight(self) -> u32 {
        match self {
            Self::Critical => 100,
            Self::High => 75,
            Self::Normal => 50,
            Self::Low => 25,
            Self::Deferred => 10,
        }
    }
}

impl fmt::Display for PriorityTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
            Self::Deferred => write!(f, "Deferred"),
        }
    }
}

impl PartialOrd for PriorityTier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTier {
    fn cmp(&self, other: &Self) -> Ordering {
        self.weight().cmp(&other.weight())
    }
}

/// A unique identifier for a queued item.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QueueItemId(pub String);

impl fmt::Display for QueueItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Category of a queued event for coalescing purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EventCategory(pub String);

/// A single item in the priority queue.
#[derive(Debug, Clone)]
pub struct QueueItem {
    /// Unique identifier.
    pub id: QueueItemId,
    /// Priority tier.
    pub priority: PriorityTier,
    /// Scheduled timestamp (epoch millis). Earlier = higher urgency at same tier.
    pub scheduled_at_ms: i64,
    /// Optional deadline (epoch millis). After this the item is stale.
    pub deadline_ms: Option<i64>,
    /// Event category (for coalescing duplicate events).
    pub category: Option<EventCategory>,
    /// Channel identifier this item belongs to (for fair scheduling).
    pub channel_id: Option<String>,
    /// Description of the task/event.
    pub description: String,
    /// Insertion sequence number (tie-breaker for equal priority/time).
    sequence: u64,
}

impl PartialEq for QueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for QueueItem {}

impl PartialOrd for QueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority tier first.
        let tier_cmp = self.priority.cmp(&other.priority);
        if tier_cmp != Ordering::Equal {
            return tier_cmp;
        }
        // Earlier scheduled time first (reverse because BinaryHeap is max-heap).
        let time_cmp = other.scheduled_at_ms.cmp(&self.scheduled_at_ms);
        if time_cmp != Ordering::Equal {
            return time_cmp;
        }
        // Earlier insertion first.
        other.sequence.cmp(&self.sequence)
    }
}

/// Statistics about the priority queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total items currently in the queue.
    pub total_items: usize,
    /// Items per priority tier.
    pub items_per_tier: HashMap<String, usize>,
    /// Number of stale (past-deadline) items.
    pub stale_items: usize,
    /// Total items ever enqueued.
    pub total_enqueued: u64,
    /// Total items ever dequeued.
    pub total_dequeued: u64,
}

/// Priority queue for broadcast automation events and tasks.
#[derive(Debug)]
pub struct AutomationPriorityQueue {
    /// The underlying max-heap.
    heap: BinaryHeap<QueueItem>,
    /// Monotonic sequence counter for insertion ordering.
    sequence: u64,
    /// Set of active item IDs for dedup.
    active_ids: HashMap<String, PriorityTier>,
    /// Lifetime enqueue counter.
    total_enqueued: u64,
    /// Lifetime dequeue counter.
    total_dequeued: u64,
    /// Maximum queue capacity (0 = unlimited).
    max_capacity: usize,
}

impl AutomationPriorityQueue {
    /// Create a new priority queue with no capacity limit.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            sequence: 0,
            active_ids: HashMap::new(),
            total_enqueued: 0,
            total_dequeued: 0,
            max_capacity: 0,
        }
    }

    /// Create a priority queue with a maximum capacity.
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(max_capacity),
            sequence: 0,
            active_ids: HashMap::new(),
            total_enqueued: 0,
            total_dequeued: 0,
            max_capacity,
        }
    }

    /// Enqueue an item. Returns `false` if the queue is at capacity or the ID is duplicate.
    pub fn enqueue(
        &mut self,
        id: &str,
        priority: PriorityTier,
        scheduled_at_ms: i64,
        description: &str,
    ) -> bool {
        if self.max_capacity > 0 && self.heap.len() >= self.max_capacity {
            return false;
        }
        if self.active_ids.contains_key(id) {
            return false;
        }
        let item = QueueItem {
            id: QueueItemId(id.to_string()),
            priority,
            scheduled_at_ms,
            deadline_ms: None,
            category: None,
            channel_id: None,
            description: description.to_string(),
            sequence: self.sequence,
        };
        self.sequence += 1;
        self.active_ids.insert(id.to_string(), priority);
        self.heap.push(item);
        self.total_enqueued += 1;
        true
    }

    /// Enqueue an item with a deadline.
    pub fn enqueue_with_deadline(
        &mut self,
        id: &str,
        priority: PriorityTier,
        scheduled_at_ms: i64,
        deadline_ms: i64,
        description: &str,
    ) -> bool {
        if self.max_capacity > 0 && self.heap.len() >= self.max_capacity {
            return false;
        }
        if self.active_ids.contains_key(id) {
            return false;
        }
        let item = QueueItem {
            id: QueueItemId(id.to_string()),
            priority,
            scheduled_at_ms,
            deadline_ms: Some(deadline_ms),
            category: None,
            channel_id: None,
            description: description.to_string(),
            sequence: self.sequence,
        };
        self.sequence += 1;
        self.active_ids.insert(id.to_string(), priority);
        self.heap.push(item);
        self.total_enqueued += 1;
        true
    }

    /// Dequeue the highest-priority item.
    pub fn dequeue(&mut self) -> Option<QueueItem> {
        let item = self.heap.pop()?;
        self.active_ids.remove(&item.id.0);
        self.total_dequeued += 1;
        Some(item)
    }

    /// Peek at the highest-priority item without removing it.
    pub fn peek(&self) -> Option<&QueueItem> {
        self.heap.peek()
    }

    /// Number of items currently in the queue.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Remove stale items whose deadline has passed.
    pub fn purge_stale(&mut self, current_time_ms: i64) -> usize {
        let before = self.heap.len();
        let old_heap = std::mem::take(&mut self.heap);
        for item in old_heap {
            if let Some(deadline) = item.deadline_ms {
                if current_time_ms > deadline {
                    self.active_ids.remove(&item.id.0);
                    continue;
                }
            }
            self.heap.push(item);
        }
        before - self.heap.len()
    }

    /// Check if an item with the given ID is in the queue.
    pub fn contains(&self, id: &str) -> bool {
        self.active_ids.contains_key(id)
    }

    /// Get queue statistics.
    pub fn stats(&self, current_time_ms: i64) -> QueueStats {
        let mut items_per_tier: HashMap<String, usize> = HashMap::new();
        let mut stale_items = 0;
        for item in &self.heap {
            *items_per_tier.entry(item.priority.to_string()).or_insert(0) += 1;
            if let Some(deadline) = item.deadline_ms {
                if current_time_ms > deadline {
                    stale_items += 1;
                }
            }
        }
        QueueStats {
            total_items: self.heap.len(),
            items_per_tier,
            stale_items,
            total_enqueued: self.total_enqueued,
            total_dequeued: self.total_dequeued,
        }
    }

    /// Clear all items from the queue.
    pub fn clear(&mut self) {
        self.heap.clear();
        self.active_ids.clear();
    }

    /// Count items at a specific priority tier.
    pub fn count_at_tier(&self, tier: PriorityTier) -> usize {
        self.active_ids.values().filter(|&&t| t == tier).count()
    }
}

impl Default for AutomationPriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_tier_ordering() {
        assert!(PriorityTier::Critical > PriorityTier::High);
        assert!(PriorityTier::High > PriorityTier::Normal);
        assert!(PriorityTier::Normal > PriorityTier::Low);
        assert!(PriorityTier::Low > PriorityTier::Deferred);
    }

    #[test]
    fn test_priority_tier_display() {
        assert_eq!(PriorityTier::Critical.to_string(), "Critical");
        assert_eq!(PriorityTier::Deferred.to_string(), "Deferred");
    }

    #[test]
    fn test_new_queue_is_empty() {
        let q = AutomationPriorityQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_enqueue_dequeue_basic() {
        let mut q = AutomationPriorityQueue::new();
        assert!(q.enqueue("e1", PriorityTier::Normal, 1000, "event 1"));
        assert_eq!(q.len(), 1);
        let item = q.dequeue().expect("dequeue should succeed");
        assert_eq!(item.id.0, "e1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_priority_ordering_dequeue() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("low", PriorityTier::Low, 1000, "low priority");
        q.enqueue("crit", PriorityTier::Critical, 2000, "critical");
        q.enqueue("norm", PriorityTier::Normal, 500, "normal");

        let first = q.dequeue().expect("dequeue should succeed");
        assert_eq!(first.id.0, "crit");
        let second = q.dequeue().expect("dequeue should succeed");
        assert_eq!(second.id.0, "norm");
        let third = q.dequeue().expect("dequeue should succeed");
        assert_eq!(third.id.0, "low");
    }

    #[test]
    fn test_same_priority_earlier_time_first() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("later", PriorityTier::Normal, 2000, "later");
        q.enqueue("earlier", PriorityTier::Normal, 1000, "earlier");

        let first = q.dequeue().expect("dequeue should succeed");
        assert_eq!(first.id.0, "earlier");
    }

    #[test]
    fn test_duplicate_id_rejected() {
        let mut q = AutomationPriorityQueue::new();
        assert!(q.enqueue("dup", PriorityTier::Normal, 1000, "first"));
        assert!(!q.enqueue("dup", PriorityTier::High, 500, "second"));
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_capacity_limit() {
        let mut q = AutomationPriorityQueue::with_capacity(2);
        assert!(q.enqueue("a", PriorityTier::Normal, 1000, "a"));
        assert!(q.enqueue("b", PriorityTier::Normal, 2000, "b"));
        assert!(!q.enqueue("c", PriorityTier::Critical, 500, "c"));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_purge_stale_items() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue_with_deadline("stale", PriorityTier::Normal, 100, 500, "will expire");
        q.enqueue_with_deadline("fresh", PriorityTier::Normal, 100, 2000, "still valid");
        q.enqueue("no_deadline", PriorityTier::Low, 100, "no deadline");

        let removed = q.purge_stale(1000);
        assert_eq!(removed, 1);
        assert_eq!(q.len(), 2);
        assert!(!q.contains("stale"));
        assert!(q.contains("fresh"));
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("peek_me", PriorityTier::High, 100, "peek test");
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_clear_empties_queue() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("a", PriorityTier::Normal, 100, "a");
        q.enqueue("b", PriorityTier::High, 200, "b");
        q.clear();
        assert!(q.is_empty());
        assert!(!q.contains("a"));
    }

    #[test]
    fn test_stats_report() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("c1", PriorityTier::Critical, 100, "crit 1");
        q.enqueue("n1", PriorityTier::Normal, 200, "norm 1");
        q.enqueue("n2", PriorityTier::Normal, 300, "norm 2");
        q.dequeue(); // dequeue one

        let stats = q.stats(0);
        assert_eq!(stats.total_items, 2);
        assert_eq!(stats.total_enqueued, 3);
        assert_eq!(stats.total_dequeued, 1);
    }

    #[test]
    fn test_count_at_tier() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("h1", PriorityTier::High, 100, "h1");
        q.enqueue("h2", PriorityTier::High, 200, "h2");
        q.enqueue("n1", PriorityTier::Normal, 300, "n1");
        assert_eq!(q.count_at_tier(PriorityTier::High), 2);
        assert_eq!(q.count_at_tier(PriorityTier::Normal), 1);
        assert_eq!(q.count_at_tier(PriorityTier::Low), 0);
    }

    #[test]
    fn test_contains_after_dequeue() {
        let mut q = AutomationPriorityQueue::new();
        q.enqueue("item", PriorityTier::Normal, 100, "item");
        assert!(q.contains("item"));
        q.dequeue();
        assert!(!q.contains("item"));
    }

    #[test]
    fn test_queue_item_id_display() {
        let id = QueueItemId("abc-123".to_string());
        assert_eq!(id.to_string(), "abc-123");
    }
}
