#![allow(dead_code)]
//! Distributed message queue primitives for `OxiMedia`.
//!
//! Provides a lightweight priority-based in-memory message queue suitable for
//! inter-node communication in a distributed encoding cluster.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Priority level of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MessagePriority {
    /// Background / bulk transfer
    Low = 0,
    /// Normal operational messages
    Normal = 1,
    /// Important control messages
    High = 2,
    /// Time-critical or error-recovery messages
    Critical = 3,
}

impl MessagePriority {
    /// Numeric representation (0 = lowest, 3 = highest).
    #[must_use]
    pub fn numeric_value(self) -> u8 {
        self as u8
    }

    /// Return the next higher priority level, if one exists.
    #[must_use]
    pub fn escalate(self) -> Self {
        match self {
            MessagePriority::Low => MessagePriority::Normal,
            MessagePriority::Normal => MessagePriority::High,
            MessagePriority::High | MessagePriority::Critical => MessagePriority::Critical,
        }
    }
}

/// An envelope wrapping a message payload for the distributed queue.
#[derive(Debug, Clone)]
pub struct DistributedMessage {
    /// Unique message identifier.
    pub id: u64,
    /// Priority of this message.
    pub priority: MessagePriority,
    /// Opaque byte payload.
    pub payload: Vec<u8>,
    /// When the message was enqueued.
    pub enqueued_at: Instant,
    /// Optional TTL: message is expired after this duration.
    pub ttl: Option<Duration>,
    /// Source node identifier.
    pub source: String,
    /// Destination node identifier.
    pub destination: String,
}

impl DistributedMessage {
    /// Create a new message.
    #[must_use]
    pub fn new(
        id: u64,
        priority: MessagePriority,
        payload: Vec<u8>,
        source: impl Into<String>,
        destination: impl Into<String>,
    ) -> Self {
        Self {
            id,
            priority,
            payload,
            enqueued_at: Instant::now(),
            ttl: None,
            source: source.into(),
            destination: destination.into(),
        }
    }

    /// Attach a time-to-live to the message.
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Return `true` if the message's TTL has elapsed relative to `now`.
    ///
    /// Messages without a TTL never expire.
    #[must_use]
    pub fn is_expired(&self, now: Instant) -> bool {
        match self.ttl {
            None => false,
            Some(ttl) => now.saturating_duration_since(self.enqueued_at) >= ttl,
        }
    }

    /// Payload length in bytes.
    #[must_use]
    pub fn payload_len(&self) -> usize {
        self.payload.len()
    }
}

/// Wrapper that makes `DistributedMessage` orderable for `BinaryHeap`
/// (max-heap by priority, then by insertion order — lower id = older = higher urgency).
#[derive(Debug)]
struct QueueEntry {
    message: DistributedMessage,
    seq: u64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.message.priority == other.message.priority && self.seq == other.seq
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
        // Higher priority first; for equal priority, lower seq (older) first.
        match self.message.priority.cmp(&other.message.priority) {
            Ordering::Equal => other.seq.cmp(&self.seq),
            ord => ord,
        }
    }
}

/// A bounded, priority-ordered distributed message queue.
#[derive(Debug)]
pub struct MessageQueue {
    heap: BinaryHeap<QueueEntry>,
    capacity: usize,
    seq_counter: u64,
    enqueued_total: u64,
    dropped_total: u64,
}

impl MessageQueue {
    /// Create a new queue with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity,
            seq_counter: 0,
            enqueued_total: 0,
            dropped_total: 0,
        }
    }

    /// Number of messages currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Return `true` when the queue contains no messages.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Enqueue a message.
    ///
    /// Returns `false` and increments the dropped counter when the queue is
    /// at capacity.
    pub fn enqueue(&mut self, message: DistributedMessage) -> bool {
        if self.heap.len() >= self.capacity {
            self.dropped_total += 1;
            return false;
        }
        let seq = self.seq_counter;
        self.seq_counter += 1;
        self.enqueued_total += 1;
        self.heap.push(QueueEntry { message, seq });
        true
    }

    /// Dequeue the highest-priority message, or `None` if the queue is empty.
    pub fn dequeue(&mut self) -> Option<DistributedMessage> {
        self.heap.pop().map(|e| e.message)
    }

    /// Peek at the priority of the next message without removing it.
    #[must_use]
    pub fn peek_priority(&self) -> Option<MessagePriority> {
        self.heap.peek().map(|e| e.message.priority)
    }

    /// Total messages successfully enqueued since creation.
    #[must_use]
    pub fn enqueued_total(&self) -> u64 {
        self.enqueued_total
    }

    /// Total messages dropped due to capacity overflow.
    #[must_use]
    pub fn dropped_total(&self) -> u64 {
        self.dropped_total
    }

    /// Drain all expired messages (as of `now`) from the queue, returning the
    /// count of messages removed.
    pub fn drain_expired(&mut self, now: Instant) -> usize {
        let before = self.heap.len();
        let entries: Vec<QueueEntry> = self.heap.drain().collect();
        for entry in entries {
            if !entry.message.is_expired(now) {
                self.heap.push(entry);
            }
        }
        before - self.heap.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn msg(id: u64, priority: MessagePriority) -> DistributedMessage {
        DistributedMessage::new(id, priority, vec![1, 2, 3], "src", "dst")
    }

    #[test]
    fn test_priority_numeric_values() {
        assert_eq!(MessagePriority::Low.numeric_value(), 0);
        assert_eq!(MessagePriority::Normal.numeric_value(), 1);
        assert_eq!(MessagePriority::High.numeric_value(), 2);
        assert_eq!(MessagePriority::Critical.numeric_value(), 3);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(MessagePriority::Critical > MessagePriority::High);
        assert!(MessagePriority::High > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
    }

    #[test]
    fn test_priority_escalate() {
        assert_eq!(MessagePriority::Low.escalate(), MessagePriority::Normal);
        assert_eq!(MessagePriority::Normal.escalate(), MessagePriority::High);
        assert_eq!(MessagePriority::High.escalate(), MessagePriority::Critical);
        assert_eq!(
            MessagePriority::Critical.escalate(),
            MessagePriority::Critical
        );
    }

    #[test]
    fn test_message_not_expired_without_ttl() {
        let m = msg(1, MessagePriority::Normal);
        assert!(!m.is_expired(Instant::now()));
    }

    #[test]
    fn test_message_not_expired_within_ttl() {
        let m = msg(1, MessagePriority::Normal).with_ttl(Duration::from_secs(60));
        assert!(!m.is_expired(Instant::now()));
    }

    #[test]
    fn test_message_is_expired_after_ttl() {
        let past = Instant::now() - Duration::from_secs(10);
        let mut m = msg(1, MessagePriority::Normal);
        m.enqueued_at = past;
        let m = m.with_ttl(Duration::from_secs(5));
        assert!(m.is_expired(Instant::now()));
    }

    #[test]
    fn test_message_payload_len() {
        let m = DistributedMessage::new(1, MessagePriority::Low, vec![0u8; 100], "a", "b");
        assert_eq!(m.payload_len(), 100);
    }

    #[test]
    fn test_queue_enqueue_and_len() {
        let mut q = MessageQueue::new(10);
        assert!(q.is_empty());
        q.enqueue(msg(1, MessagePriority::Low));
        q.enqueue(msg(2, MessagePriority::High));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_dequeue_priority_order() {
        let mut q = MessageQueue::new(10);
        q.enqueue(msg(1, MessagePriority::Low));
        q.enqueue(msg(2, MessagePriority::Critical));
        q.enqueue(msg(3, MessagePriority::Normal));

        assert_eq!(
            q.dequeue().expect("dequeue should return a task").priority,
            MessagePriority::Critical
        );
        assert_eq!(
            q.dequeue().expect("dequeue should return a task").priority,
            MessagePriority::Normal
        );
        assert_eq!(
            q.dequeue().expect("dequeue should return a task").priority,
            MessagePriority::Low
        );
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_queue_peek_priority() {
        let mut q = MessageQueue::new(5);
        assert!(q.peek_priority().is_none());
        q.enqueue(msg(1, MessagePriority::Low));
        q.enqueue(msg(2, MessagePriority::High));
        assert_eq!(q.peek_priority(), Some(MessagePriority::High));
    }

    #[test]
    fn test_queue_capacity_overflow() {
        let mut q = MessageQueue::new(2);
        assert!(q.enqueue(msg(1, MessagePriority::Low)));
        assert!(q.enqueue(msg(2, MessagePriority::Low)));
        assert!(!q.enqueue(msg(3, MessagePriority::Low)));
        assert_eq!(q.dropped_total(), 1);
    }

    #[test]
    fn test_queue_enqueued_total() {
        let mut q = MessageQueue::new(10);
        q.enqueue(msg(1, MessagePriority::Normal));
        q.enqueue(msg(2, MessagePriority::Normal));
        assert_eq!(q.enqueued_total(), 2);
    }

    #[test]
    fn test_drain_expired() {
        let mut q = MessageQueue::new(10);
        let past = Instant::now() - Duration::from_secs(10);

        let mut expired_msg = msg(1, MessagePriority::Low);
        expired_msg.enqueued_at = past;
        let expired_msg = expired_msg.with_ttl(Duration::from_secs(5));
        q.enqueue(expired_msg);

        let fresh = msg(2, MessagePriority::High).with_ttl(Duration::from_secs(60));
        q.enqueue(fresh);

        let removed = q.drain_expired(Instant::now());
        assert_eq!(removed, 1);
        assert_eq!(q.len(), 1);
    }
}
