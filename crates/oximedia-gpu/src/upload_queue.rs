#![allow(dead_code)]
//! Asynchronous upload queue for staging CPU data to GPU buffers.
//!
//! This module provides a batched upload mechanism that stages host-side data
//! into a queue and flushes it to GPU-side buffers in optimized batches,
//! reducing per-transfer overhead.

use std::collections::VecDeque;
use std::time::Instant;

/// Unique identifier for an upload request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UploadId(
    /// Inner identifier value.
    pub u64,
);

/// Priority level for upload requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum UploadPriority {
    /// Low priority - background uploads.
    Low = 0,
    /// Normal priority - standard uploads.
    #[default]
    Normal = 1,
    /// High priority - latency-sensitive uploads.
    High = 2,
    /// Critical priority - must be processed immediately.
    Critical = 3,
}

/// The current state of an upload request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UploadState {
    /// Queued and waiting to be processed.
    Queued,
    /// Currently being transferred.
    Transferring,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed.
    Failed,
    /// Transfer was cancelled.
    Cancelled,
}

/// Describes the destination for an upload.
#[derive(Debug, Clone)]
pub struct UploadTarget {
    /// Target buffer identifier.
    pub buffer_id: u64,
    /// Byte offset within the target buffer.
    pub offset: usize,
    /// Expected total size of the target buffer.
    pub buffer_size: usize,
}

impl UploadTarget {
    /// Create a new upload target.
    #[must_use]
    pub fn new(buffer_id: u64, offset: usize, buffer_size: usize) -> Self {
        Self {
            buffer_id,
            offset,
            buffer_size,
        }
    }

    /// Check whether the given data size fits within the target at the offset.
    #[must_use]
    pub fn fits(&self, data_size: usize) -> bool {
        self.offset + data_size <= self.buffer_size
    }
}

/// A single upload request in the queue.
#[derive(Debug, Clone)]
pub struct UploadRequest {
    /// Unique identifier for this request.
    pub id: UploadId,
    /// The data to upload.
    pub data: Vec<u8>,
    /// Target destination for the data.
    pub target: UploadTarget,
    /// Priority level.
    pub priority: UploadPriority,
    /// Current state.
    pub state: UploadState,
    /// Timestamp when the request was enqueued.
    pub enqueue_time: Instant,
    /// Timestamp when the transfer completed.
    pub complete_time: Option<Instant>,
}

impl UploadRequest {
    /// Create a new upload request.
    #[must_use]
    pub fn new(
        id: UploadId,
        data: Vec<u8>,
        target: UploadTarget,
        priority: UploadPriority,
    ) -> Self {
        Self {
            id,
            data,
            target,
            priority,
            state: UploadState::Queued,
            enqueue_time: Instant::now(),
            complete_time: None,
        }
    }

    /// Return the size of the data in bytes.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.data.len()
    }

    /// Return the transfer latency if the upload is completed.
    #[must_use]
    pub fn latency(&self) -> Option<std::time::Duration> {
        self.complete_time
            .map(|t| t.duration_since(self.enqueue_time))
    }
}

/// Configuration for the upload queue.
#[derive(Debug, Clone)]
pub struct UploadQueueConfig {
    /// Maximum number of pending requests.
    pub max_pending: usize,
    /// Maximum total bytes that can be queued.
    pub max_queued_bytes: usize,
    /// Batch size for flush operations.
    pub flush_batch_size: usize,
    /// Whether to sort by priority before flushing.
    pub priority_sort: bool,
}

impl Default for UploadQueueConfig {
    fn default() -> Self {
        Self {
            max_pending: 256,
            max_queued_bytes: 256 * 1024 * 1024, // 256 MB
            flush_batch_size: 32,
            priority_sort: true,
        }
    }
}

/// Statistics for the upload queue.
#[derive(Debug, Clone, Default)]
pub struct UploadQueueStats {
    /// Number of requests currently in the queue.
    pub queued_count: usize,
    /// Total bytes currently queued.
    pub queued_bytes: usize,
    /// Total number of requests processed.
    pub total_processed: u64,
    /// Total bytes transferred.
    pub total_bytes_transferred: u64,
    /// Number of failed requests.
    pub failed_count: u64,
    /// Number of cancelled requests.
    pub cancelled_count: u64,
    /// Average transfer throughput in bytes per second.
    pub avg_throughput_bps: f64,
}

impl UploadQueueStats {
    /// Return queued bytes in megabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn queued_mb(&self) -> f64 {
        self.queued_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Return total transferred in megabytes.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn transferred_mb(&self) -> f64 {
        self.total_bytes_transferred as f64 / (1024.0 * 1024.0)
    }
}

/// An asynchronous upload queue that batches host-to-device transfers.
pub struct UploadQueue {
    /// Pending upload requests.
    queue: VecDeque<UploadRequest>,
    /// Completed upload requests (for stats).
    completed: Vec<UploadRequest>,
    /// Configuration.
    config: UploadQueueConfig,
    /// Counter for generating unique upload IDs.
    next_id: u64,
    /// Total bytes transferred.
    total_bytes_transferred: u64,
    /// Failed request count.
    failed_count: u64,
    /// Cancelled request count.
    cancelled_count: u64,
}

impl UploadQueue {
    /// Create a new upload queue with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(UploadQueueConfig::default())
    }

    /// Create a new upload queue with the given configuration.
    #[must_use]
    pub fn with_config(config: UploadQueueConfig) -> Self {
        Self {
            queue: VecDeque::new(),
            completed: Vec::new(),
            config,
            next_id: 0,
            total_bytes_transferred: 0,
            failed_count: 0,
            cancelled_count: 0,
        }
    }

    /// Enqueue an upload request. Returns the `UploadId` if successful,
    /// or `None` if the queue is full.
    pub fn enqueue(
        &mut self,
        data: Vec<u8>,
        target: UploadTarget,
        priority: UploadPriority,
    ) -> Option<UploadId> {
        let current_bytes: usize = self.queue.iter().map(|r| r.data.len()).sum();
        if self.queue.len() >= self.config.max_pending {
            return None;
        }
        if current_bytes + data.len() > self.config.max_queued_bytes {
            return None;
        }
        let id = UploadId(self.next_id);
        self.next_id += 1;
        let request = UploadRequest::new(id, data, target, priority);
        self.queue.push_back(request);
        Some(id)
    }

    /// Flush a batch of uploads, simulating the transfer to GPU.
    /// Returns the IDs of requests that were processed.
    pub fn flush(&mut self) -> Vec<UploadId> {
        if self.config.priority_sort {
            let mut items: Vec<UploadRequest> = self.queue.drain(..).collect();
            items.sort_by(|a, b| b.priority.cmp(&a.priority));
            self.queue = items.into_iter().collect();
        }

        let batch_size = self.config.flush_batch_size.min(self.queue.len());
        let mut processed = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if let Some(mut request) = self.queue.pop_front() {
                if request.target.fits(request.data.len()) {
                    request.state = UploadState::Completed;
                    request.complete_time = Some(Instant::now());
                    self.total_bytes_transferred += request.data.len() as u64;
                } else {
                    request.state = UploadState::Failed;
                    self.failed_count += 1;
                }
                processed.push(request.id);
                self.completed.push(request);
            }
        }

        processed
    }

    /// Cancel a pending upload by ID.
    pub fn cancel(&mut self, id: UploadId) -> bool {
        if let Some(pos) = self.queue.iter().position(|r| r.id == id) {
            let mut request = match self.queue.remove(pos) {
                Some(r) => r,
                None => return false,
            };
            request.state = UploadState::Cancelled;
            self.cancelled_count += 1;
            self.completed.push(request);
            true
        } else {
            false
        }
    }

    /// Return the number of pending requests.
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Return the total queued bytes.
    #[must_use]
    pub fn queued_bytes(&self) -> usize {
        self.queue.iter().map(|r| r.data.len()).sum()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Return the state of a request by ID.
    #[must_use]
    pub fn request_state(&self, id: UploadId) -> Option<UploadState> {
        self.queue
            .iter()
            .find(|r| r.id == id)
            .map(|r| r.state)
            .or_else(|| self.completed.iter().find(|r| r.id == id).map(|r| r.state))
    }

    /// Return statistics about the queue.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn stats(&self) -> UploadQueueStats {
        let queued_bytes: usize = self.queue.iter().map(|r| r.data.len()).sum();
        let total_processed = self.completed.len() as u64;
        let latencies: Vec<f64> = self
            .completed
            .iter()
            .filter_map(UploadRequest::latency)
            .map(|d| d.as_secs_f64())
            .collect();
        let avg_throughput_bps = if latencies.is_empty() {
            0.0
        } else {
            let total_secs: f64 = latencies.iter().sum();
            if total_secs > 0.0 {
                self.total_bytes_transferred as f64 / total_secs
            } else {
                0.0
            }
        };

        UploadQueueStats {
            queued_count: self.queue.len(),
            queued_bytes,
            total_processed,
            total_bytes_transferred: self.total_bytes_transferred,
            failed_count: self.failed_count,
            cancelled_count: self.cancelled_count,
            avg_throughput_bps,
        }
    }

    /// Clear all pending requests (marking them cancelled).
    pub fn clear(&mut self) {
        while let Some(mut req) = self.queue.pop_front() {
            req.state = UploadState::Cancelled;
            self.cancelled_count += 1;
            self.completed.push(req);
        }
    }

    /// Peek at the next request that would be processed.
    #[must_use]
    pub fn peek_next(&self) -> Option<&UploadRequest> {
        self.queue.front()
    }
}

impl Default for UploadQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_target(size: usize) -> UploadTarget {
        UploadTarget::new(1, 0, size)
    }

    #[test]
    fn test_create_queue() {
        let queue = UploadQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn test_enqueue_request() {
        let mut queue = UploadQueue::new();
        let data = vec![0u8; 1024];
        let target = make_target(2048);
        let id = queue.enqueue(data, target, UploadPriority::Normal);
        assert!(id.is_some());
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_enqueue_limit() {
        let config = UploadQueueConfig {
            max_pending: 2,
            ..Default::default()
        };
        let mut queue = UploadQueue::with_config(config);
        let t1 = make_target(4096);
        let t2 = make_target(4096);
        let t3 = make_target(4096);
        assert!(queue
            .enqueue(vec![0; 100], t1, UploadPriority::Normal)
            .is_some());
        assert!(queue
            .enqueue(vec![0; 100], t2, UploadPriority::Normal)
            .is_some());
        assert!(queue
            .enqueue(vec![0; 100], t3, UploadPriority::Normal)
            .is_none());
    }

    #[test]
    fn test_enqueue_byte_limit() {
        let config = UploadQueueConfig {
            max_queued_bytes: 200,
            ..Default::default()
        };
        let mut queue = UploadQueue::with_config(config);
        let t1 = make_target(4096);
        let t2 = make_target(4096);
        assert!(queue
            .enqueue(vec![0; 150], t1, UploadPriority::Normal)
            .is_some());
        assert!(queue
            .enqueue(vec![0; 100], t2, UploadPriority::Normal)
            .is_none());
    }

    #[test]
    fn test_flush_batch() {
        let mut queue = UploadQueue::new();
        for i in 0..5 {
            let target = make_target(4096);
            queue.enqueue(vec![0u8; 100], target, UploadPriority::Normal);
            let _ = i;
        }
        let processed = queue.flush();
        assert!(!processed.is_empty());
    }

    #[test]
    fn test_flush_priority_ordering() {
        let config = UploadQueueConfig {
            priority_sort: true,
            flush_batch_size: 1,
            ..Default::default()
        };
        let mut queue = UploadQueue::with_config(config);
        let t1 = make_target(4096);
        let t2 = make_target(4096);
        let _low_id = queue
            .enqueue(vec![1], t1, UploadPriority::Low)
            .expect("enqueue should succeed");
        let high_id = queue
            .enqueue(vec![2], t2, UploadPriority::Critical)
            .expect("operation should succeed in test");
        let processed = queue.flush();
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0], high_id);
    }

    #[test]
    fn test_cancel_request() {
        let mut queue = UploadQueue::new();
        let target = make_target(4096);
        let id = queue
            .enqueue(vec![0; 100], target, UploadPriority::Normal)
            .expect("operation should succeed in test");
        assert!(queue.cancel(id));
        assert!(queue.is_empty());
        assert_eq!(queue.request_state(id), Some(UploadState::Cancelled));
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut queue = UploadQueue::new();
        assert!(!queue.cancel(UploadId(999)));
    }

    #[test]
    fn test_request_state_tracking() {
        let mut queue = UploadQueue::new();
        let target = make_target(4096);
        let id = queue
            .enqueue(vec![0; 100], target, UploadPriority::Normal)
            .expect("operation should succeed in test");
        assert_eq!(queue.request_state(id), Some(UploadState::Queued));
        queue.flush();
        assert_eq!(queue.request_state(id), Some(UploadState::Completed));
    }

    #[test]
    fn test_failed_transfer() {
        let mut queue = UploadQueue::new();
        // Target too small for data
        let target = UploadTarget::new(1, 0, 10);
        let id = queue
            .enqueue(vec![0; 100], target, UploadPriority::Normal)
            .expect("operation should succeed in test");
        queue.flush();
        assert_eq!(queue.request_state(id), Some(UploadState::Failed));
    }

    #[test]
    fn test_queued_bytes() {
        let mut queue = UploadQueue::new();
        let t1 = make_target(4096);
        let t2 = make_target(4096);
        queue.enqueue(vec![0; 100], t1, UploadPriority::Normal);
        queue.enqueue(vec![0; 200], t2, UploadPriority::Normal);
        assert_eq!(queue.queued_bytes(), 300);
    }

    #[test]
    fn test_stats() {
        let mut queue = UploadQueue::new();
        let target = make_target(4096);
        queue.enqueue(vec![0; 1024], target, UploadPriority::Normal);
        queue.flush();
        let stats = queue.stats();
        assert_eq!(stats.total_processed, 1);
        assert_eq!(stats.total_bytes_transferred, 1024);
    }

    #[test]
    fn test_stats_mb_conversion() {
        let stats = UploadQueueStats {
            queued_bytes: 1_048_576,
            total_bytes_transferred: 2_097_152,
            ..Default::default()
        };
        assert!((stats.queued_mb() - 1.0).abs() < 0.001);
        assert!((stats.transferred_mb() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_clear_queue() {
        let mut queue = UploadQueue::new();
        for _ in 0..5 {
            let target = make_target(4096);
            queue.enqueue(vec![0; 100], target, UploadPriority::Normal);
        }
        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.stats().cancelled_count, 5);
    }

    #[test]
    fn test_peek_next() {
        let mut queue = UploadQueue::new();
        assert!(queue.peek_next().is_none());
        let target = make_target(4096);
        let id = queue
            .enqueue(vec![0; 100], target, UploadPriority::Normal)
            .expect("operation should succeed in test");
        let peeked = queue.peek_next().expect("peek should return next item");
        assert_eq!(peeked.id, id);
    }

    #[test]
    fn test_upload_target_fits() {
        let target = UploadTarget::new(1, 10, 100);
        assert!(target.fits(90));
        assert!(target.fits(0));
        assert!(!target.fits(91));
    }
}
