#![allow(dead_code)]
//! Storage event bus — publish, subscribe, and query storage lifecycle events.

use std::collections::VecDeque;

/// Events emitted by the storage subsystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageEvent {
    /// A new object was stored.
    ObjectCreated {
        /// Object key.
        key: String,
        /// Size in bytes.
        size_bytes: u64,
    },
    /// An object was removed.
    ObjectDeleted {
        /// Object key.
        key: String,
    },
    /// The storage backend reported it is full.
    StorageFull {
        /// Backend / bucket identifier.
        backend: String,
        /// Total capacity in bytes.
        capacity_bytes: u64,
    },
    /// A quota limit is approaching.
    QuotaWarning {
        /// Namespace or account identifier.
        namespace: String,
        /// Current usage in bytes.
        used_bytes: u64,
        /// Quota limit in bytes.
        limit_bytes: u64,
    },
    /// An object was successfully replicated to another location.
    ObjectReplicated {
        /// Object key.
        key: String,
        /// Destination backend.
        destination: String,
    },
    /// An error occurred during a storage operation.
    OperationError {
        /// Operation type.
        operation: String,
        /// Error description.
        message: String,
    },
}

impl StorageEvent {
    /// Whether this event represents a warning or degraded condition.
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            Self::StorageFull { .. } | Self::QuotaWarning { .. } | Self::OperationError { .. }
        )
    }

    /// Whether this is a creation event.
    pub fn is_creation(&self) -> bool {
        matches!(self, Self::ObjectCreated { .. })
    }

    /// Whether this is a deletion event.
    pub fn is_deletion(&self) -> bool {
        matches!(self, Self::ObjectDeleted { .. })
    }

    /// Short description of the event type.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::ObjectCreated { .. } => "ObjectCreated",
            Self::ObjectDeleted { .. } => "ObjectDeleted",
            Self::StorageFull { .. } => "StorageFull",
            Self::QuotaWarning { .. } => "QuotaWarning",
            Self::ObjectReplicated { .. } => "ObjectReplicated",
            Self::OperationError { .. } => "OperationError",
        }
    }
}

/// A subscriber record holding events that match its filter.
#[derive(Debug)]
pub struct Subscriber {
    /// Subscriber identifier.
    pub id: String,
    /// Events delivered to this subscriber.
    pub inbox: VecDeque<StorageEvent>,
    /// Optional event-type filter; `None` = receive all.
    pub filter: Option<fn(&StorageEvent) -> bool>,
}

impl Subscriber {
    /// Create a new subscriber with no filter.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            inbox: VecDeque::new(),
            filter: None,
        }
    }

    /// Create a new subscriber with a filter predicate.
    pub fn with_filter(id: impl Into<String>, filter: fn(&StorageEvent) -> bool) -> Self {
        Self {
            id: id.into(),
            inbox: VecDeque::new(),
            filter: Some(filter),
        }
    }

    /// Drain all events from the inbox.
    pub fn drain(&mut self) -> Vec<StorageEvent> {
        self.inbox.drain(..).collect()
    }
}

/// Central event bus for storage events.
#[derive(Debug, Default)]
pub struct StorageEventBus {
    /// Ring buffer of recent events (capped at `max_recent`).
    recent: VecDeque<StorageEvent>,
    /// Maximum number of recent events to retain.
    max_recent: usize,
    /// Registered subscribers.
    subscribers: Vec<Subscriber>,
    /// Total events published since creation.
    total_published: u64,
}

impl StorageEventBus {
    /// Create a new bus retaining up to `max_recent` recent events.
    pub fn new(max_recent: usize) -> Self {
        Self {
            recent: VecDeque::new(),
            max_recent: max_recent.max(1),
            subscribers: Vec::new(),
            total_published: 0,
        }
    }

    /// Publish an event, fan-out to subscribers, and record in the recent ring buffer.
    pub fn publish(&mut self, event: StorageEvent) {
        // Fan out to subscribers.
        for sub in &mut self.subscribers {
            let deliver = sub.filter.is_none_or(|f| f(&event));
            if deliver {
                sub.inbox.push_back(event.clone());
            }
        }

        // Record in recent ring.
        if self.recent.len() >= self.max_recent {
            self.recent.pop_front();
        }
        self.recent.push_back(event);
        self.total_published += 1;
    }

    /// Register a subscriber. Returns the subscriber index.
    pub fn subscribe_events(&mut self, subscriber: Subscriber) -> usize {
        let idx = self.subscribers.len();
        self.subscribers.push(subscriber);
        idx
    }

    /// Retrieve a snapshot of recent events (oldest first).
    pub fn recent_events(&self) -> Vec<&StorageEvent> {
        self.recent.iter().collect()
    }

    /// Count of recent events retained.
    pub fn recent_count(&self) -> usize {
        self.recent.len()
    }

    /// Total events published over the lifetime of this bus.
    pub fn total_published(&self) -> u64 {
        self.total_published
    }

    /// Count of events marked as warnings in the recent buffer.
    pub fn warning_count(&self) -> usize {
        self.recent.iter().filter(|e| e.is_warning()).count()
    }

    /// Drain a subscriber's inbox by index. Returns `None` if index is out of bounds.
    pub fn drain_subscriber(&mut self, idx: usize) -> Option<Vec<StorageEvent>> {
        self.subscribers.get_mut(idx).map(Subscriber::drain)
    }

    /// Number of registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_created(key: &str, size: u64) -> StorageEvent {
        StorageEvent::ObjectCreated {
            key: key.to_owned(),
            size_bytes: size,
        }
    }

    fn make_deleted(key: &str) -> StorageEvent {
        StorageEvent::ObjectDeleted {
            key: key.to_owned(),
        }
    }

    fn make_quota_warning(ns: &str) -> StorageEvent {
        StorageEvent::QuotaWarning {
            namespace: ns.to_owned(),
            used_bytes: 900,
            limit_bytes: 1000,
        }
    }

    fn make_full(backend: &str) -> StorageEvent {
        StorageEvent::StorageFull {
            backend: backend.to_owned(),
            capacity_bytes: 1024,
        }
    }

    #[test]
    fn test_is_warning_quota() {
        assert!(make_quota_warning("ns").is_warning());
    }

    #[test]
    fn test_is_warning_full() {
        assert!(make_full("b1").is_warning());
    }

    #[test]
    fn test_is_warning_false_for_created() {
        assert!(!make_created("k", 100).is_warning());
    }

    #[test]
    fn test_is_creation() {
        assert!(make_created("k", 1).is_creation());
        assert!(!make_deleted("k").is_creation());
    }

    #[test]
    fn test_is_deletion() {
        assert!(make_deleted("k").is_deletion());
        assert!(!make_created("k", 1).is_deletion());
    }

    #[test]
    fn test_event_type_names() {
        assert_eq!(make_created("k", 1).event_type(), "ObjectCreated");
        assert_eq!(make_deleted("k").event_type(), "ObjectDeleted");
        assert_eq!(make_quota_warning("ns").event_type(), "QuotaWarning");
        assert_eq!(make_full("b").event_type(), "StorageFull");
    }

    #[test]
    fn test_publish_increments_total() {
        let mut bus = StorageEventBus::new(10);
        bus.publish(make_created("a", 100));
        bus.publish(make_deleted("b"));
        assert_eq!(bus.total_published(), 2);
    }

    #[test]
    fn test_recent_events_capped() {
        let mut bus = StorageEventBus::new(3);
        for i in 0..5 {
            bus.publish(make_created(&format!("k{i}"), i as u64));
        }
        assert_eq!(bus.recent_count(), 3);
    }

    #[test]
    fn test_recent_events_order() {
        let mut bus = StorageEventBus::new(5);
        bus.publish(make_created("first", 1));
        bus.publish(make_deleted("second"));
        let events = bus.recent_events();
        assert_eq!(events[0].event_type(), "ObjectCreated");
        assert_eq!(events[1].event_type(), "ObjectDeleted");
    }

    #[test]
    fn test_subscribe_receives_events() {
        let mut bus = StorageEventBus::new(10);
        let idx = bus.subscribe_events(Subscriber::new("sub1"));
        bus.publish(make_created("obj", 512));
        let received = bus.drain_subscriber(idx).expect("drain should succeed");
        assert_eq!(received.len(), 1);
    }

    #[test]
    fn test_subscriber_filter() {
        let mut bus = StorageEventBus::new(10);
        // Only warnings.
        let sub = Subscriber::with_filter("warn-only", |e| e.is_warning());
        let idx = bus.subscribe_events(sub);
        bus.publish(make_created("obj", 1));
        bus.publish(make_quota_warning("ns"));
        let received = bus.drain_subscriber(idx).expect("drain should succeed");
        assert_eq!(received.len(), 1);
        assert!(received[0].is_warning());
    }

    #[test]
    fn test_warning_count() {
        let mut bus = StorageEventBus::new(10);
        bus.publish(make_created("a", 1));
        bus.publish(make_quota_warning("ns"));
        bus.publish(make_full("b"));
        assert_eq!(bus.warning_count(), 2);
    }

    #[test]
    fn test_drain_out_of_bounds() {
        let mut bus = StorageEventBus::new(5);
        assert!(bus.drain_subscriber(99).is_none());
    }

    #[test]
    fn test_subscriber_count() {
        let mut bus = StorageEventBus::new(5);
        bus.subscribe_events(Subscriber::new("a"));
        bus.subscribe_events(Subscriber::new("b"));
        assert_eq!(bus.subscriber_count(), 2);
    }
}
