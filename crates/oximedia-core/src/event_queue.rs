//! Priority-aware media event queue.
//!
//! Provides an in-memory queue of [`MediaEvent`] items ordered by
//! [`EventPriority`], suitable for coordinating asynchronous media pipeline
//! stages without an external message broker.

#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Scheduling priority for a media event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventPriority {
    /// Low-priority background work (e.g. index updates).
    Low = 0,
    /// Normal pipeline processing.
    Normal = 1,
    /// High-priority control signals (e.g. EOS, flush).
    High = 2,
    /// Critical errors that must be handled immediately.
    Critical = 3,
}

impl EventPriority {
    /// Returns a numeric value for this priority level (higher = more urgent).
    #[must_use]
    pub fn value(self) -> u8 {
        self as u8
    }
}

/// A media pipeline event with an associated payload string and priority.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MediaEvent {
    /// Human-readable event kind (e.g. `"frame.ready"`, `"eos"`).
    pub kind: String,
    /// Optional payload associated with this event.
    pub payload: Option<String>,
    /// Scheduling priority.
    pub priority: EventPriority,
    /// Sequence number for FIFO ordering among same-priority events.
    pub(crate) seq: u64,
}

impl MediaEvent {
    /// Create a new [`MediaEvent`].
    #[must_use]
    pub fn new(kind: impl Into<String>, priority: EventPriority) -> Self {
        Self {
            kind: kind.into(),
            payload: None,
            priority,
            seq: 0,
        }
    }

    /// Attach a payload to this event, returning `self` for chaining.
    #[must_use]
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Returns `true` if this event has [`EventPriority::High`] or
    /// [`EventPriority::Critical`] priority.
    #[must_use]
    pub fn is_high_priority(&self) -> bool {
        matches!(self.priority, EventPriority::High | EventPriority::Critical)
    }
}

// BinaryHeap is a max-heap; we want higher EventPriority::value() to come out
// first, then lower seq (FIFO within the same priority).
impl Ord for MediaEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq)) // lower seq = older = first
    }
}

impl PartialOrd for MediaEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A bounded, priority-ordered queue of [`MediaEvent`]s.
#[derive(Debug)]
pub struct EventQueue {
    heap: BinaryHeap<MediaEvent>,
    capacity: usize,
    next_seq: u64,
}

impl EventQueue {
    /// Create a new [`EventQueue`] with the given maximum `capacity`.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::new(),
            capacity,
            next_seq: 0,
        }
    }

    /// Push an event onto the queue.
    ///
    /// Returns `false` if the queue is full (event is discarded).
    pub fn push(&mut self, mut event: MediaEvent) -> bool {
        if self.heap.len() >= self.capacity {
            return false;
        }
        event.seq = self.next_seq;
        self.next_seq += 1;
        self.heap.push(event);
        true
    }

    /// Pop the highest-priority event from the queue.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<MediaEvent> {
        self.heap.pop()
    }

    /// Drain all events with [`EventPriority::High`] or higher into a `Vec`.
    ///
    /// The remaining events stay in the queue.
    pub fn drain_high_priority(&mut self) -> Vec<MediaEvent> {
        let mut high: Vec<MediaEvent> = Vec::new();
        let mut rest: Vec<MediaEvent> = Vec::new();

        while let Some(ev) = self.heap.pop() {
            if ev.is_high_priority() {
                high.push(ev);
            } else {
                rest.push(ev);
            }
        }
        for ev in rest {
            self.heap.push(ev);
        }
        high
    }

    /// Returns the number of events currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue contains no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(kind: &str, prio: EventPriority) -> MediaEvent {
        MediaEvent::new(kind, prio)
    }

    #[test]
    fn test_priority_value_ordering() {
        assert!(EventPriority::Critical.value() > EventPriority::High.value());
        assert!(EventPriority::High.value() > EventPriority::Normal.value());
        assert!(EventPriority::Normal.value() > EventPriority::Low.value());
    }

    #[test]
    fn test_event_is_high_priority() {
        assert!(make_event("flush", EventPriority::High).is_high_priority());
        assert!(make_event("err", EventPriority::Critical).is_high_priority());
        assert!(!make_event("frame", EventPriority::Normal).is_high_priority());
        assert!(!make_event("bg", EventPriority::Low).is_high_priority());
    }

    #[test]
    fn test_event_with_payload() {
        let ev = MediaEvent::new("test", EventPriority::Normal).with_payload("data");
        assert_eq!(ev.payload.as_deref(), Some("data"));
    }

    #[test]
    fn test_queue_push_pop_single() {
        let mut q = EventQueue::new(8);
        let ev = make_event("eos", EventPriority::High);
        assert!(q.push(ev));
        assert_eq!(q.len(), 1);
        let popped = q.pop().expect("pop should return item");
        assert_eq!(popped.kind, "eos");
        assert!(q.is_empty());
    }

    #[test]
    fn test_queue_priority_ordering() {
        let mut q = EventQueue::new(8);
        q.push(make_event("low", EventPriority::Low));
        q.push(make_event("high", EventPriority::High));
        q.push(make_event("normal", EventPriority::Normal));

        assert_eq!(q.pop().expect("pop should return item").kind, "high");
        assert_eq!(q.pop().expect("pop should return item").kind, "normal");
        assert_eq!(q.pop().expect("pop should return item").kind, "low");
    }

    #[test]
    fn test_queue_fifo_same_priority() {
        let mut q = EventQueue::new(8);
        q.push(make_event("first", EventPriority::Normal));
        q.push(make_event("second", EventPriority::Normal));
        q.push(make_event("third", EventPriority::Normal));

        assert_eq!(q.pop().expect("pop should return item").kind, "first");
        assert_eq!(q.pop().expect("pop should return item").kind, "second");
        assert_eq!(q.pop().expect("pop should return item").kind, "third");
    }

    #[test]
    fn test_queue_capacity_limit() {
        let mut q = EventQueue::new(2);
        assert!(q.push(make_event("a", EventPriority::Low)));
        assert!(q.push(make_event("b", EventPriority::Low)));
        assert!(!q.push(make_event("c", EventPriority::Low))); // rejected
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_queue_drain_high_priority() {
        let mut q = EventQueue::new(8);
        q.push(make_event("low1", EventPriority::Low));
        q.push(make_event("high1", EventPriority::High));
        q.push(make_event("normal1", EventPriority::Normal));
        q.push(make_event("crit1", EventPriority::Critical));

        let high = q.drain_high_priority();
        assert_eq!(high.len(), 2);
        assert_eq!(q.len(), 2); // low1 + normal1 remain
    }

    #[test]
    fn test_queue_empty_pop() {
        let mut q = EventQueue::new(4);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_queue_capacity_accessor() {
        let q = EventQueue::new(16);
        assert_eq!(q.capacity(), 16);
    }

    #[test]
    fn test_event_kind_stored() {
        let ev = MediaEvent::new("frame.ready", EventPriority::Normal);
        assert_eq!(ev.kind, "frame.ready");
    }

    #[test]
    fn test_drain_high_priority_empty() {
        let mut q = EventQueue::new(8);
        q.push(make_event("low", EventPriority::Low));
        let high = q.drain_high_priority();
        assert!(high.is_empty());
        assert_eq!(q.len(), 1);
    }
}
