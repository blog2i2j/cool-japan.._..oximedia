#![allow(dead_code)]
//! Deduplication work queue with priority scheduling.
//!
//! Provides a priority queue for scheduling deduplication tasks across
//! media files, allowing high-priority items (large files, user requests)
//! to be processed before low-priority background scans.

use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

/// Priority level for dedup tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DedupPriority {
    /// Critical priority (user-requested immediate scan).
    Critical,
    /// High priority (newly ingested files).
    High,
    /// Normal priority (scheduled background scan).
    Normal,
    /// Low priority (periodic re-verification).
    Low,
    /// Background priority (idle-time processing).
    Background,
}

impl DedupPriority {
    /// Convert priority to a numeric value (higher = more urgent).
    fn numeric(&self) -> u8 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Normal => 2,
            Self::Low => 1,
            Self::Background => 0,
        }
    }

    /// Get display name for the priority.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Critical => "critical",
            Self::High => "high",
            Self::Normal => "normal",
            Self::Low => "low",
            Self::Background => "background",
        }
    }
}

impl PartialOrd for DedupPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DedupPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numeric().cmp(&other.numeric())
    }
}

/// Type of dedup task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DedupTaskKind {
    /// Hash a file for exact duplicate detection.
    HashFile,
    /// Compute perceptual hash for visual similarity.
    PerceptualHash,
    /// Full similarity comparison between two items.
    Compare,
    /// Re-verify an existing duplicate entry.
    Verify,
    /// Clean up stale entries.
    Cleanup,
}

impl DedupTaskKind {
    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::HashFile => "hash_file",
            Self::PerceptualHash => "perceptual_hash",
            Self::Compare => "compare",
            Self::Verify => "verify",
            Self::Cleanup => "cleanup",
        }
    }
}

/// A single dedup task in the queue.
#[derive(Debug, Clone)]
pub struct DedupTask {
    /// Unique task identifier.
    pub id: u64,
    /// Priority.
    pub priority: DedupPriority,
    /// Task kind.
    pub kind: DedupTaskKind,
    /// File path or identifier.
    pub target: String,
    /// Optional second target for comparison tasks.
    pub compare_target: Option<String>,
    /// File size hint (for scheduling).
    pub size_hint: u64,
    /// Timestamp when the task was created (epoch millis).
    pub created_at: u64,
    /// Number of retry attempts.
    pub retries: u32,
    /// Maximum retries allowed.
    pub max_retries: u32,
}

impl DedupTask {
    /// Create a new dedup task.
    pub fn new(id: u64, priority: DedupPriority, kind: DedupTaskKind, target: String) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            id,
            priority,
            kind,
            target,
            compare_target: None,
            size_hint: 0,
            created_at,
            retries: 0,
            max_retries: 3,
        }
    }

    /// Set the size hint.
    pub fn with_size_hint(mut self, size: u64) -> Self {
        self.size_hint = size;
        self
    }

    /// Set the compare target.
    pub fn with_compare_target(mut self, target: String) -> Self {
        self.compare_target = Some(target);
        self
    }

    /// Set the maximum retries.
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Check if this task can be retried.
    pub fn can_retry(&self) -> bool {
        self.retries < self.max_retries
    }

    /// Increment the retry counter and return a new task for retry.
    pub fn retry(&self) -> Option<Self> {
        if !self.can_retry() {
            return None;
        }
        let mut task = self.clone();
        task.retries += 1;
        Some(task)
    }
}

impl PartialEq for DedupTask {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for DedupTask {}

impl PartialOrd for DedupTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DedupTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then older tasks first (lower created_at)
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.created_at.cmp(&self.created_at))
    }
}

/// Priority queue for dedup tasks.
#[derive(Debug)]
pub struct DedupQueue {
    /// The priority queue.
    heap: BinaryHeap<DedupTask>,
    /// Next task ID.
    next_id: u64,
    /// Total tasks ever enqueued.
    total_enqueued: u64,
    /// Total tasks completed.
    total_completed: u64,
    /// Total tasks failed.
    total_failed: u64,
}

impl DedupQueue {
    /// Create a new empty dedup queue.
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_id: 1,
            total_enqueued: 0,
            total_completed: 0,
            total_failed: 0,
        }
    }

    /// Enqueue a task, returning the assigned ID.
    pub fn enqueue(
        &mut self,
        priority: DedupPriority,
        kind: DedupTaskKind,
        target: String,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let task = DedupTask::new(id, priority, kind, target);
        self.heap.push(task);
        self.total_enqueued += 1;
        id
    }

    /// Enqueue a pre-built task.
    pub fn enqueue_task(&mut self, task: DedupTask) {
        self.heap.push(task);
        self.total_enqueued += 1;
    }

    /// Dequeue the highest-priority task.
    pub fn dequeue(&mut self) -> Option<DedupTask> {
        self.heap.pop()
    }

    /// Peek at the highest-priority task without removing it.
    pub fn peek(&self) -> Option<&DedupTask> {
        self.heap.peek()
    }

    /// Get the number of pending tasks.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Record a task completion.
    pub fn record_completed(&mut self) {
        self.total_completed += 1;
    }

    /// Record a task failure.
    pub fn record_failed(&mut self) {
        self.total_failed += 1;
    }

    /// Get queue statistics.
    pub fn stats(&self) -> QueueStats {
        QueueStats {
            pending: self.heap.len(),
            total_enqueued: self.total_enqueued,
            total_completed: self.total_completed,
            total_failed: self.total_failed,
        }
    }

    /// Clear all pending tasks.
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Drain up to `n` tasks from the queue.
    pub fn drain_batch(&mut self, n: usize) -> Vec<DedupTask> {
        let mut batch = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(task) = self.heap.pop() {
                batch.push(task);
            } else {
                break;
            }
        }
        batch
    }
}

impl Default for DedupQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Queue statistics.
#[derive(Debug, Clone)]
pub struct QueueStats {
    /// Number of pending tasks.
    pub pending: usize,
    /// Total tasks ever enqueued.
    pub total_enqueued: u64,
    /// Total tasks completed.
    pub total_completed: u64,
    /// Total tasks failed.
    pub total_failed: u64,
}

impl QueueStats {
    /// Get success rate as a fraction.
    #[allow(clippy::cast_precision_loss)]
    pub fn success_rate(&self) -> f64 {
        let total = self.total_completed + self.total_failed;
        if total == 0 {
            return 1.0;
        }
        self.total_completed as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        assert!(DedupPriority::Critical > DedupPriority::High);
        assert!(DedupPriority::High > DedupPriority::Normal);
        assert!(DedupPriority::Normal > DedupPriority::Low);
        assert!(DedupPriority::Low > DedupPriority::Background);
    }

    #[test]
    fn test_priority_name() {
        assert_eq!(DedupPriority::Critical.name(), "critical");
        assert_eq!(DedupPriority::Background.name(), "background");
    }

    #[test]
    fn test_task_kind_name() {
        assert_eq!(DedupTaskKind::HashFile.name(), "hash_file");
        assert_eq!(DedupTaskKind::Cleanup.name(), "cleanup");
    }

    #[test]
    fn test_task_creation() {
        let task = DedupTask::new(
            1,
            DedupPriority::Normal,
            DedupTaskKind::HashFile,
            "test.mp4".to_string(),
        );
        assert_eq!(task.id, 1);
        assert_eq!(task.priority, DedupPriority::Normal);
        assert_eq!(task.kind, DedupTaskKind::HashFile);
        assert_eq!(task.target, "test.mp4");
        assert!(task.compare_target.is_none());
        assert_eq!(task.retries, 0);
    }

    #[test]
    fn test_task_builders() {
        let task = DedupTask::new(1, DedupPriority::High, DedupTaskKind::Compare, "a.mp4".to_string())
            .with_size_hint(1024)
            .with_compare_target("b.mp4".to_string())
            .with_max_retries(5);
        assert_eq!(task.size_hint, 1024);
        assert_eq!(task.compare_target.as_deref(), Some("b.mp4"));
        assert_eq!(task.max_retries, 5);
    }

    #[test]
    fn test_task_retry() {
        let task = DedupTask::new(1, DedupPriority::Normal, DedupTaskKind::HashFile, "x".to_string())
            .with_max_retries(2);
        assert!(task.can_retry());

        let r1 = task.retry().expect("operation should succeed");
        assert_eq!(r1.retries, 1);
        assert!(r1.can_retry());

        let r2 = r1.retry().expect("operation should succeed");
        assert_eq!(r2.retries, 2);
        assert!(!r2.can_retry());
        assert!(r2.retry().is_none());
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut q = DedupQueue::new();
        assert!(q.is_empty());

        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "a.mp4".to_string());
        q.enqueue(DedupPriority::High, DedupTaskKind::HashFile, "b.mp4".to_string());
        q.enqueue(DedupPriority::Low, DedupTaskKind::HashFile, "c.mp4".to_string());

        assert_eq!(q.len(), 3);

        // Should dequeue in priority order
        let t = q.dequeue().expect("operation should succeed");
        assert_eq!(t.priority, DedupPriority::High);
        let t = q.dequeue().expect("operation should succeed");
        assert_eq!(t.priority, DedupPriority::Normal);
        let t = q.dequeue().expect("operation should succeed");
        assert_eq!(t.priority, DedupPriority::Low);
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_queue_peek() {
        let mut q = DedupQueue::new();
        assert!(q.peek().is_none());

        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "x".to_string());
        assert!(q.peek().is_some());
        assert_eq!(q.len(), 1); // peek doesn't remove
    }

    #[test]
    fn test_queue_stats() {
        let mut q = DedupQueue::new();
        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "a".to_string());
        q.enqueue(DedupPriority::High, DedupTaskKind::HashFile, "b".to_string());
        let _ = q.dequeue();
        q.record_completed();
        q.record_failed();

        let stats = q.stats();
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.total_enqueued, 2);
        assert_eq!(stats.total_completed, 1);
        assert_eq!(stats.total_failed, 1);
    }

    #[test]
    fn test_queue_clear() {
        let mut q = DedupQueue::new();
        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "a".to_string());
        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "b".to_string());
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_drain_batch() {
        let mut q = DedupQueue::new();
        for i in 0..5 {
            q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, format!("f{i}"));
        }
        let batch = q.drain_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_drain_batch_more_than_available() {
        let mut q = DedupQueue::new();
        q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "a".to_string());
        let batch = q.drain_batch(10);
        assert_eq!(batch.len(), 1);
        assert!(q.is_empty());
    }

    #[test]
    fn test_success_rate() {
        let stats = QueueStats {
            pending: 0,
            total_enqueued: 10,
            total_completed: 8,
            total_failed: 2,
        };
        assert!((stats.success_rate() - 0.8).abs() < f64::EPSILON);

        let empty_stats = QueueStats {
            pending: 0,
            total_enqueued: 0,
            total_completed: 0,
            total_failed: 0,
        };
        assert!((empty_stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_queue_id_autoincrement() {
        let mut q = DedupQueue::new();
        let id1 = q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "a".to_string());
        let id2 = q.enqueue(DedupPriority::Normal, DedupTaskKind::HashFile, "b".to_string());
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }
}
