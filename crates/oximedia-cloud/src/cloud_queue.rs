#![allow(dead_code)]
//! Simple cloud message queue abstraction.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for a `CloudQueue`.
#[derive(Debug, Clone)]
pub struct CloudQueueConfig {
    /// Maximum number of messages the queue may hold.
    pub capacity: usize,
    /// Maximum time in seconds a message is retained before being discarded.
    pub max_retention_secs: u64,
    /// Maximum number of bytes per message payload.
    pub max_message_bytes: usize,
}

impl CloudQueueConfig {
    /// Create a config with the given capacity and retention.
    pub fn new(capacity: usize, max_retention_secs: u64) -> Self {
        Self {
            capacity,
            max_retention_secs,
            max_message_bytes: 256 * 1024, // 256 KiB default
        }
    }

    /// Returns the maximum retention duration in seconds.
    pub fn max_retention_secs(&self) -> u64 {
        self.max_retention_secs
    }
}

impl Default for CloudQueueConfig {
    fn default() -> Self {
        Self::new(10_000, 3600)
    }
}

/// A message held in the queue.
#[derive(Debug, Clone, PartialEq)]
pub struct QueueMessage {
    /// Unique identifier.
    pub id: u64,
    /// Message payload bytes.
    pub payload: Vec<u8>,
    /// Wall-clock instant at which the message was enqueued.
    pub enqueued_at: Instant,
    /// Configured retention for this message.
    pub retention: Duration,
}

impl QueueMessage {
    /// Create a new message.
    pub fn new(id: u64, payload: Vec<u8>, retention_secs: u64) -> Self {
        Self {
            id,
            payload,
            enqueued_at: Instant::now(),
            retention: Duration::from_secs(retention_secs),
        }
    }

    /// Returns `true` if this message has exceeded its retention window.
    pub fn is_expired(&self) -> bool {
        self.enqueued_at.elapsed() >= self.retention
    }

    /// Returns the payload length in bytes.
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }
}

/// Errors that can occur when interacting with a `CloudQueue`.
#[derive(Debug, PartialEq, Eq)]
pub enum QueueError {
    /// Queue has reached maximum capacity.
    QueueFull,
    /// Message payload exceeds the configured limit.
    PayloadTooLarge,
    /// Queue is empty; no message to dequeue.
    QueueEmpty,
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "queue is full"),
            Self::PayloadTooLarge => write!(f, "message payload exceeds limit"),
            Self::QueueEmpty => write!(f, "queue is empty"),
        }
    }
}

/// A first-in, first-out cloud message queue.
#[derive(Debug)]
pub struct CloudQueue {
    config: CloudQueueConfig,
    messages: VecDeque<QueueMessage>,
    next_id: u64,
}

impl CloudQueue {
    /// Create a new queue with the given configuration.
    pub fn new(config: CloudQueueConfig) -> Self {
        Self {
            messages: VecDeque::with_capacity(config.capacity),
            config,
            next_id: 1,
        }
    }

    /// Enqueue a message payload.
    ///
    /// Returns the assigned message ID on success, or a `QueueError` on failure.
    pub fn enqueue(&mut self, payload: Vec<u8>) -> Result<u64, QueueError> {
        if payload.len() > self.config.max_message_bytes {
            return Err(QueueError::PayloadTooLarge);
        }
        self.purge_expired();
        if self.messages.len() >= self.config.capacity {
            return Err(QueueError::QueueFull);
        }
        let id = self.next_id;
        self.next_id += 1;
        self.messages.push_back(QueueMessage::new(
            id,
            payload,
            self.config.max_retention_secs,
        ));
        Ok(id)
    }

    /// Dequeue and return the oldest non-expired message.
    pub fn dequeue(&mut self) -> Result<QueueMessage, QueueError> {
        self.purge_expired();
        self.messages.pop_front().ok_or(QueueError::QueueEmpty)
    }

    /// Returns the current number of messages in the queue (excluding expired).
    pub fn depth(&self) -> usize {
        self.messages.iter().filter(|m| !m.is_expired()).count()
    }

    /// Purge all expired messages from the front of the queue.
    fn purge_expired(&mut self) {
        while let Some(front) = self.messages.front() {
            if front.is_expired() {
                self.messages.pop_front();
            } else {
                break;
            }
        }
    }

    /// Returns `true` if the queue has no messages.
    pub fn is_empty(&self) -> bool {
        self.depth() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_queue(cap: usize) -> CloudQueue {
        CloudQueue::new(CloudQueueConfig::new(cap, 3600))
    }

    #[test]
    fn test_config_max_retention() {
        let cfg = CloudQueueConfig::new(100, 7200);
        assert_eq!(cfg.max_retention_secs(), 7200);
    }

    #[test]
    fn test_config_default() {
        let cfg = CloudQueueConfig::default();
        assert_eq!(cfg.capacity, 10_000);
        assert_eq!(cfg.max_retention_secs, 3600);
    }

    #[test]
    fn test_enqueue_returns_id() {
        let mut q = make_queue(10);
        let id = q.enqueue(b"hello".to_vec()).expect("id should be valid");
        assert_eq!(id, 1);
    }

    #[test]
    fn test_enqueue_increments_id() {
        let mut q = make_queue(10);
        let id1 = q.enqueue(b"a".to_vec()).expect("id1 should be valid");
        let id2 = q.enqueue(b"b".to_vec()).expect("id2 should be valid");
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn test_depth_after_enqueue() {
        let mut q = make_queue(10);
        q.enqueue(b"x".to_vec()).expect("test expectation failed");
        q.enqueue(b"y".to_vec()).expect("test expectation failed");
        assert_eq!(q.depth(), 2);
    }

    #[test]
    fn test_dequeue_fifo_order() {
        let mut q = make_queue(10);
        q.enqueue(b"first".to_vec())
            .expect("test expectation failed");
        q.enqueue(b"second".to_vec())
            .expect("test expectation failed");
        let msg = q.dequeue().expect("msg should be valid");
        assert_eq!(msg.payload, b"first");
    }

    #[test]
    fn test_dequeue_empty_error() {
        let mut q = make_queue(10);
        assert_eq!(q.dequeue(), Err(QueueError::QueueEmpty));
    }

    #[test]
    fn test_queue_full_error() {
        let mut q = make_queue(2);
        q.enqueue(b"a".to_vec()).expect("test expectation failed");
        q.enqueue(b"b".to_vec()).expect("test expectation failed");
        assert_eq!(q.enqueue(b"c".to_vec()), Err(QueueError::QueueFull));
    }

    #[test]
    fn test_payload_too_large_error() {
        let cfg = CloudQueueConfig {
            capacity: 10,
            max_retention_secs: 3600,
            max_message_bytes: 5,
        };
        let mut q = CloudQueue::new(cfg);
        assert_eq!(
            q.enqueue(b"toolongpayload".to_vec()),
            Err(QueueError::PayloadTooLarge)
        );
    }

    #[test]
    fn test_is_empty_initially() {
        let q = make_queue(10);
        assert!(q.is_empty());
    }

    #[test]
    fn test_is_empty_after_enqueue() {
        let mut q = make_queue(10);
        q.enqueue(b"data".to_vec())
            .expect("test expectation failed");
        assert!(!q.is_empty());
    }

    #[test]
    fn test_message_not_expired_immediately() {
        let msg = QueueMessage::new(1, b"data".to_vec(), 3600);
        assert!(!msg.is_expired());
    }

    #[test]
    fn test_message_payload_len() {
        let msg = QueueMessage::new(1, b"hello".to_vec(), 60);
        assert_eq!(msg.payload_len(), 5);
    }

    #[test]
    fn test_depth_decreases_after_dequeue() {
        let mut q = make_queue(10);
        q.enqueue(b"a".to_vec()).expect("test expectation failed");
        q.enqueue(b"b".to_vec()).expect("test expectation failed");
        q.dequeue().expect("dequeue should succeed");
        assert_eq!(q.depth(), 1);
    }

    #[test]
    fn test_queue_error_display() {
        assert_eq!(QueueError::QueueFull.to_string(), "queue is full");
        assert_eq!(QueueError::QueueEmpty.to_string(), "queue is empty");
        assert_eq!(
            QueueError::PayloadTooLarge.to_string(),
            "message payload exceeds limit"
        );
    }
}
