//! A bounded, priority-aware work queue for media pipeline tasks.
//!
//! [`WorkQueue`] holds [`WorkItem`]s ordered by priority (higher value = more
//! urgent).  [`QueueStats`] tracks lifetime throughput counters.
//!
//! # Examples
//!
//! ```
//! use oximedia_core::work_queue::{WorkItem, WorkQueue};
//!
//! let mut q: WorkQueue<u32> = WorkQueue::new(8);
//! q.push(WorkItem::new(42_u32, 10))?;
//! q.push(WorkItem::new(99_u32, 20))?;
//! // Highest-priority item comes out first.
//! let item = q.pop()?;
//! assert_eq!(item.payload, 99_u32);
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

/// A single work item carrying an arbitrary payload and a scheduling priority.
///
/// Higher `priority` values are dequeued first.
///
/// # Examples
///
/// ```
/// use oximedia_core::work_queue::WorkItem;
///
/// let item = WorkItem::new("transcode frame 7", 5_u32);
/// assert_eq!(item.priority, 5);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkItem<T> {
    /// The payload to be processed.
    pub payload: T,
    /// Scheduling priority (higher = dequeued sooner).
    pub priority: u32,
}

impl<T> WorkItem<T> {
    /// Creates a new [`WorkItem`] with the given payload and priority.
    #[must_use]
    pub const fn new(payload: T, priority: u32) -> Self {
        Self { payload, priority }
    }
}

/// Error returned by queue operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueError {
    /// The queue has reached its capacity limit.
    Full,
    /// A batch request exceeded the queue's capacity.
    BatchTooLarge,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "work queue is full"),
            Self::BatchTooLarge => write!(f, "batch size exceeds queue capacity"),
        }
    }
}

impl std::error::Error for QueueError {}

/// Lifetime throughput statistics for a [`WorkQueue`].
///
/// # Examples
///
/// ```
/// use oximedia_core::work_queue::{WorkItem, WorkQueue};
///
/// let mut q: WorkQueue<()> = WorkQueue::new(16);
/// q.push(WorkItem::new((), 1))?;
/// let _ = q.pop();
/// let stats = q.stats();
/// assert_eq!(stats.total_pushed, 1);
/// assert_eq!(stats.total_popped, 1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QueueStats {
    /// Cumulative number of items successfully pushed.
    pub total_pushed: u64,
    /// Cumulative number of items removed via pop or `pop_batch`.
    pub total_popped: u64,
    /// Cumulative number of push attempts that were rejected due to capacity.
    pub total_rejected: u64,
}

impl QueueStats {
    /// Returns the number of items pushed but not yet popped.
    ///
    /// This is a lower-bound estimate; it saturates at 0 to avoid wrapping.
    #[inline]
    #[must_use]
    pub fn in_flight(&self) -> u64 {
        self.total_pushed.saturating_sub(self.total_popped)
    }
}

/// A bounded, priority-ordered work queue.
///
/// Items are stored in a `Vec` sorted in ascending priority order so that
/// the highest-priority item is always at the back and can be removed in O(1).
/// Pushes maintain sorted order via binary search insertion at O(n) in the
/// worst case, which is acceptable for the typical small queue sizes used in
/// media pipeline scheduling.
///
/// # Examples
///
/// ```
/// use oximedia_core::work_queue::{WorkItem, WorkQueue};
///
/// let mut q: WorkQueue<i32> = WorkQueue::new(4);
/// q.push(WorkItem::new(1, 5))?;
/// q.push(WorkItem::new(2, 1))?;
/// q.push(WorkItem::new(3, 9))?;
/// assert_eq!(q.pop()?.payload, 3); // priority 9 first
/// assert_eq!(q.len(), 2);
/// ```
#[derive(Debug)]
pub struct WorkQueue<T> {
    /// Items stored in ascending priority order (highest priority at the back).
    items: Vec<WorkItem<T>>,
    capacity: usize,
    stats: QueueStats,
}

impl<T> WorkQueue<T> {
    /// Creates a new queue with the given maximum capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "WorkQueue capacity must be > 0");
        Self {
            items: Vec::with_capacity(capacity.min(256)),
            capacity,
            stats: QueueStats::default(),
        }
    }

    /// Returns the maximum number of items this queue can hold.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the current number of items in the queue.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the queue contains no items.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns `true` if the queue has reached its capacity.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Pushes a work item into the queue in priority order.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::Full`] if the queue is at capacity.
    pub fn push(&mut self, item: WorkItem<T>) -> Result<(), QueueError> {
        if self.is_full() {
            self.stats.total_rejected += 1;
            return Err(QueueError::Full);
        }
        // Binary search for the insertion position (ascending by priority).
        let pos = self
            .items
            .partition_point(|existing| existing.priority <= item.priority);
        self.items.insert(pos, item);
        self.stats.total_pushed += 1;
        Ok(())
    }

    /// Removes and returns the highest-priority item, or `None` if empty.
    pub fn pop(&mut self) -> Option<WorkItem<T>> {
        let item = self.items.pop(); // highest-priority is at the back
        if item.is_some() {
            self.stats.total_popped += 1;
        }
        item
    }

    /// Removes and returns up to `n` highest-priority items.
    ///
    /// # Errors
    ///
    /// Returns [`QueueError::BatchTooLarge`] if `n > capacity`.
    pub fn pop_batch(&mut self, n: usize) -> Result<Vec<WorkItem<T>>, QueueError> {
        if n > self.capacity {
            return Err(QueueError::BatchTooLarge);
        }
        let take = n.min(self.items.len());
        let start = self.items.len().saturating_sub(take);
        let batch: Vec<WorkItem<T>> = self.items.drain(start..).rev().collect();
        self.stats.total_popped += batch.len() as u64;
        Ok(batch)
    }

    /// Peeks at the highest-priority item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&WorkItem<T>> {
        self.items.last()
    }

    /// Removes all items from the queue (statistics are preserved).
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Returns a snapshot of the lifetime statistics.
    #[must_use]
    pub fn stats(&self) -> QueueStats {
        self.stats
    }

    /// Returns an iterator over all items in ascending priority order.
    pub fn iter(&self) -> impl Iterator<Item = &WorkItem<T>> {
        self.items.iter()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_queue_is_empty() {
        let q: WorkQueue<u32> = WorkQueue::new(10);
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn push_and_pop_single_item() {
        let mut q: WorkQueue<u32> = WorkQueue::new(4);
        q.push(WorkItem::new(7_u32, 1))
            .expect("push should succeed");
        let item = q.pop().expect("pop should return item");
        assert_eq!(item.payload, 7);
        assert!(q.is_empty());
    }

    #[test]
    fn pop_respects_priority_order() {
        let mut q: WorkQueue<u32> = WorkQueue::new(8);
        q.push(WorkItem::new(1_u32, 5))
            .expect("push should succeed");
        q.push(WorkItem::new(2_u32, 1))
            .expect("push should succeed");
        q.push(WorkItem::new(3_u32, 9))
            .expect("push should succeed");
        assert_eq!(q.pop().expect("pop should return item").payload, 3); // priority 9
        assert_eq!(q.pop().expect("pop should return item").payload, 1); // priority 5
        assert_eq!(q.pop().expect("pop should return item").payload, 2); // priority 1
    }

    #[test]
    fn pop_empty_returns_none() {
        let mut q: WorkQueue<()> = WorkQueue::new(4);
        assert!(q.pop().is_none());
    }

    #[test]
    fn push_at_capacity_returns_error() {
        let mut q: WorkQueue<u32> = WorkQueue::new(2);
        q.push(WorkItem::new(1_u32, 1))
            .expect("push should succeed");
        q.push(WorkItem::new(2_u32, 2))
            .expect("push should succeed");
        let err = q.push(WorkItem::new(3_u32, 3));
        assert_eq!(err, Err(QueueError::Full));
    }

    #[test]
    fn is_full_and_capacity() {
        let mut q: WorkQueue<u32> = WorkQueue::new(1);
        assert!(!q.is_full());
        q.push(WorkItem::new(0_u32, 1))
            .expect("push should succeed");
        assert!(q.is_full());
        assert_eq!(q.capacity(), 1);
    }

    #[test]
    fn peek_does_not_remove() {
        let mut q: WorkQueue<u32> = WorkQueue::new(4);
        q.push(WorkItem::new(42_u32, 10))
            .expect("push should succeed");
        assert_eq!(q.peek().expect("peek should return item").payload, 42);
        assert_eq!(q.len(), 1); // still there
    }

    #[test]
    fn clear_empties_queue() {
        let mut q: WorkQueue<u32> = WorkQueue::new(8);
        q.push(WorkItem::new(1_u32, 1))
            .expect("push should succeed");
        q.push(WorkItem::new(2_u32, 2))
            .expect("push should succeed");
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn pop_batch_returns_highest_first() {
        let mut q: WorkQueue<u32> = WorkQueue::new(8);
        for i in 0_u32..5 {
            q.push(WorkItem::new(i, i)).expect("push should succeed");
        }
        let batch = q.pop_batch(3).expect("pop_batch should succeed");
        assert_eq!(batch.len(), 3);
        assert_eq!(batch[0].priority, 4); // highest
        assert_eq!(batch[1].priority, 3);
        assert_eq!(batch[2].priority, 2);
    }

    #[test]
    fn pop_batch_too_large_returns_error() {
        let mut q: WorkQueue<u32> = WorkQueue::new(4);
        let err = q.pop_batch(5);
        assert_eq!(err, Err(QueueError::BatchTooLarge));
    }

    #[test]
    fn stats_track_push_and_pop() {
        let mut q: WorkQueue<u32> = WorkQueue::new(8);
        q.push(WorkItem::new(1_u32, 1))
            .expect("push should succeed");
        q.push(WorkItem::new(2_u32, 2))
            .expect("push should succeed");
        let _ = q.pop();
        let s = q.stats();
        assert_eq!(s.total_pushed, 2);
        assert_eq!(s.total_popped, 1);
        assert_eq!(s.in_flight(), 1);
    }

    #[test]
    fn stats_count_rejected_pushes() {
        let mut q: WorkQueue<u32> = WorkQueue::new(1);
        q.push(WorkItem::new(1_u32, 1))
            .expect("push should succeed");
        let _ = q.push(WorkItem::new(2_u32, 2)); // rejected
        assert_eq!(q.stats().total_rejected, 1);
    }

    #[test]
    fn queue_error_display() {
        assert!(!QueueError::Full.to_string().is_empty());
        assert!(!QueueError::BatchTooLarge.to_string().is_empty());
    }

    #[test]
    fn iter_yields_all_items() {
        let mut q: WorkQueue<u32> = WorkQueue::new(8);
        for i in 0_u32..4 {
            q.push(WorkItem::new(i, i)).expect("push should succeed");
        }
        assert_eq!(q.iter().count(), 4);
    }

    #[test]
    fn work_item_is_clone() {
        let a = WorkItem::new(String::from("hello"), 5_u32);
        let b = a.clone();
        assert_eq!(a.payload, b.payload);
        assert_eq!(a.priority, b.priority);
    }
}
