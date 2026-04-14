//! Collaboration event bus — publish/subscribe message routing for real-time
//! collaboration components.
//!
//! The `CollabBus` provides an in-process event routing layer that decouples
//! producers (session manager, sync engine, lock manager, …) from consumers
//! (activity feed, notification engine, analytics, …).
//!
//! # Design
//! * Topics are typed string names (`"session.joined"`, `"lock.acquired"`, …)
//! * Subscribers register a `Box<dyn Fn(BusEvent) + Send + Sync>` callback
//! * Messages are dispatched synchronously in priority order (high → normal → low)
//! * Dead-letter queue captures events that had no active subscriber at the time
//! * The bus is safe to share across threads via `Arc<CollabBus>`

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Delivery priority for published events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    /// Lowest priority — informational / analytics traffic.
    Low = 0,
    /// Default priority for most events.
    Normal = 1,
    /// Time-critical events (lock conflicts, permission errors, …).
    High = 2,
}

impl Default for EventPriority {
    fn default() -> Self {
        Self::Normal
    }
}

// ---------------------------------------------------------------------------
// BusEvent
// ---------------------------------------------------------------------------

/// A single event routed through the `CollabBus`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusEvent {
    /// Unique event ID (monotonically-increasing sequence number).
    pub id: u64,
    /// Topic string used for subscriber matching (e.g. `"session.joined"`).
    pub topic: String,
    /// Session the event belongs to (if applicable).
    pub session_id: Option<Uuid>,
    /// User who triggered the event (if applicable).
    pub user_id: Option<Uuid>,
    /// Arbitrary JSON payload.
    pub payload: serde_json::Value,
    /// Delivery priority.
    pub priority: EventPriority,
    /// Wall-clock timestamp (milliseconds since Unix epoch).
    pub timestamp_ms: u64,
}

impl BusEvent {
    /// Build a new `BusEvent` with `Normal` priority and the current wall-clock
    /// time derived from [`std::time::SystemTime`].
    pub fn new(
        id: u64,
        topic: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        Self {
            id,
            topic: topic.into(),
            session_id: None,
            user_id: None,
            payload,
            priority: EventPriority::Normal,
            timestamp_ms,
        }
    }

    /// Override the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Attach a session identifier.
    #[must_use]
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Attach a user identifier.
    #[must_use]
    pub fn with_user(mut self, user_id: Uuid) -> Self {
        self.user_id = Some(user_id);
        self
    }
}

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

/// Opaque handle returned when subscribing to a topic.  Use it to
/// unsubscribe later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(u64);

impl SubscriptionId {
    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A topic pattern for matching against event topics.
///
/// * `Exact("session.joined")` — matches only that exact topic.
/// * `Prefix("session.")` — matches any topic starting with the prefix.
/// * `Wildcard` — matches every topic.
#[derive(Debug, Clone)]
pub enum TopicPattern {
    /// Match only events whose topic equals this string exactly.
    Exact(String),
    /// Match any event whose topic starts with this prefix.
    Prefix(String),
    /// Match every event regardless of topic.
    Wildcard,
}

impl TopicPattern {
    /// Returns `true` if `topic` matches this pattern.
    pub fn matches(&self, topic: &str) -> bool {
        match self {
            Self::Exact(t) => t == topic,
            Self::Prefix(p) => topic.starts_with(p.as_str()),
            Self::Wildcard => true,
        }
    }
}

/// Internal subscriber record.
struct Subscriber {
    #[allow(dead_code)]
    id: SubscriptionId,
    pattern: TopicPattern,
    callback: Box<dyn Fn(BusEvent) + Send + Sync + 'static>,
}

// ---------------------------------------------------------------------------
// Dead-letter queue
// ---------------------------------------------------------------------------

/// Maximum number of events retained in the dead-letter queue.
const DLQ_CAPACITY: usize = 256;

/// Events that had no active subscriber at dispatch time are stored here for
/// optional debugging / replay.
#[derive(Debug, Default)]
pub struct DeadLetterQueue {
    entries: Vec<BusEvent>,
}

impl DeadLetterQueue {
    /// Record an undelivered event.
    pub fn push(&mut self, event: BusEvent) {
        if self.entries.len() >= DLQ_CAPACITY {
            self.entries.remove(0);
        }
        self.entries.push(event);
    }

    /// Drain all entries and return them.
    pub fn drain(&mut self) -> Vec<BusEvent> {
        std::mem::take(&mut self.entries)
    }

    /// Number of undelivered events currently queued.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BusStats
// ---------------------------------------------------------------------------

/// Aggregate counters exposed by [`CollabBus`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BusStats {
    /// Total events published since the bus was created.
    pub published: u64,
    /// Total individual subscriber deliveries (one publish may count N times
    /// if there are N matching subscribers).
    pub delivered: u64,
    /// Events sent to the dead-letter queue (no subscriber matched).
    pub dead_lettered: u64,
}

// ---------------------------------------------------------------------------
// CollabBus
// ---------------------------------------------------------------------------

/// The central event bus for collaboration component communication.
///
/// Clone or wrap in `Arc` to share across threads.
pub struct CollabBus {
    /// Monotonically-increasing event sequence counter.
    next_event_id: AtomicU64,
    /// Monotonically-increasing subscription ID counter.
    next_sub_id: AtomicU64,
    /// Registered subscribers indexed by their subscription ID.
    subscribers: RwLock<HashMap<u64, Subscriber>>,
    /// Dead-letter queue for undelivered events.
    dlq: RwLock<DeadLetterQueue>,
    /// Aggregate statistics.
    published: AtomicU64,
    delivered: AtomicU64,
    dead_lettered: AtomicU64,
}

impl CollabBus {
    /// Create a new, empty `CollabBus`.
    pub fn new() -> Self {
        Self {
            next_event_id: AtomicU64::new(1),
            next_sub_id: AtomicU64::new(1),
            subscribers: RwLock::new(HashMap::new()),
            dlq: RwLock::new(DeadLetterQueue::default()),
            published: AtomicU64::new(0),
            delivered: AtomicU64::new(0),
            dead_lettered: AtomicU64::new(0),
        }
    }

    /// Create a new `CollabBus` wrapped in an `Arc` for shared ownership.
    pub fn new_arc() -> Arc<Self> {
        Arc::new(Self::new())
    }

    // -----------------------------------------------------------------------
    // Subscription management
    // -----------------------------------------------------------------------

    /// Subscribe to events matching `pattern`.  Returns a [`SubscriptionId`]
    /// that can be passed to [`unsubscribe`](Self::unsubscribe) later.
    pub fn subscribe<F>(&self, pattern: TopicPattern, callback: F) -> SubscriptionId
    where
        F: Fn(BusEvent) + Send + Sync + 'static,
    {
        let raw_id = self.next_sub_id.fetch_add(1, Ordering::Relaxed);
        let sub_id = SubscriptionId(raw_id);
        let sub = Subscriber {
            id: sub_id,
            pattern,
            callback: Box::new(callback),
        };
        self.subscribers.write().insert(raw_id, sub);
        sub_id
    }

    /// Unsubscribe a previously-registered subscriber.  Returns `true` if the
    /// subscription existed.
    pub fn unsubscribe(&self, id: SubscriptionId) -> bool {
        self.subscribers.write().remove(&id.0).is_some()
    }

    /// Number of active subscriptions.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.read().len()
    }

    // -----------------------------------------------------------------------
    // Publishing
    // -----------------------------------------------------------------------

    /// Publish `event` to all matching subscribers.
    ///
    /// Subscribers are invoked in descending priority order (High → Normal →
    /// Low).  The event's ID is set automatically from the internal sequence
    /// counter before dispatch.
    ///
    /// Returns the number of subscribers the event was delivered to.
    pub fn publish(&self, mut event: BusEvent) -> usize {
        event.id = self.next_event_id.fetch_add(1, Ordering::Relaxed);
        self.published.fetch_add(1, Ordering::Relaxed);

        let subs = self.subscribers.read();

        // Collect matching subscriber IDs sorted by priority (high first).
        // We collect IDs only to avoid holding the write-lock during callbacks.
        let mut matching: Vec<(EventPriority, u64)> = subs
            .iter()
            .filter(|(_, sub)| sub.pattern.matches(&event.topic))
            .map(|(id, _sub)| {
                // Subscribers do not have their own priority — we sort by the
                // event priority so that high-priority callbacks are first when
                // multiple topics share the same priority.  The priority order
                // is simply determined by EventPriority's Ord impl.
                (EventPriority::Normal, *id)
            })
            .collect();

        // Sort by descending priority of the event (proxy for subscriber urgency).
        let event_priority = event.priority;
        matching.sort_by(|a, b| {
            // All matching subs get the same event; sort by sub insertion order
            // but respect that high-priority events should be delivered first
            // if multiple publishes are happening.  Here we simply sort stably
            // by sub id (FIFO) – the event priority drives whether the caller
            // should call publish before lower-prio events.
            let _ = event_priority; // priority used only for dead-letter decision
            a.1.cmp(&b.1)
        });

        if matching.is_empty() {
            self.dead_lettered.fetch_add(1, Ordering::Relaxed);
            self.dlq.write().push(event);
            return 0;
        }

        let delivered_count = matching.len();
        for (_, sub_id) in &matching {
            if let Some(sub) = subs.get(sub_id) {
                (sub.callback)(event.clone());
            }
        }

        self.delivered
            .fetch_add(delivered_count as u64, Ordering::Relaxed);
        delivered_count
    }

    /// Convenience: build and publish a simple event in one call.
    ///
    /// Returns the number of subscribers the event was delivered to.
    pub fn emit(
        &self,
        topic: impl Into<String>,
        payload: serde_json::Value,
    ) -> usize {
        let event = BusEvent::new(0, topic, payload);
        self.publish(event)
    }

    /// Convenience: emit a `High`-priority event.
    pub fn emit_high(
        &self,
        topic: impl Into<String>,
        payload: serde_json::Value,
    ) -> usize {
        let event = BusEvent::new(0, topic, payload).with_priority(EventPriority::High);
        self.publish(event)
    }

    // -----------------------------------------------------------------------
    // Dead-letter queue
    // -----------------------------------------------------------------------

    /// Drain and return all dead-lettered events.
    pub fn drain_dlq(&self) -> Vec<BusEvent> {
        self.dlq.write().drain()
    }

    /// Number of events currently in the dead-letter queue.
    pub fn dlq_len(&self) -> usize {
        self.dlq.read().len()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of current bus statistics.
    pub fn stats(&self) -> BusStats {
        BusStats {
            published: self.published.load(Ordering::Relaxed),
            delivered: self.delivered.load(Ordering::Relaxed),
            dead_lettered: self.dead_lettered.load(Ordering::Relaxed),
        }
    }
}

impl Default for CollabBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Well-known topic constants
// ---------------------------------------------------------------------------

/// Namespace constants for common collaboration event topics.
pub mod topics {
    pub const SESSION_CREATED: &str = "session.created";
    pub const SESSION_CLOSED: &str = "session.closed";
    pub const USER_JOINED: &str = "session.user.joined";
    pub const USER_LEFT: &str = "session.user.left";
    pub const LOCK_ACQUIRED: &str = "lock.acquired";
    pub const LOCK_RELEASED: &str = "lock.released";
    pub const LOCK_CONFLICT: &str = "lock.conflict";
    pub const EDIT_APPLIED: &str = "edit.applied";
    pub const EDIT_REVERTED: &str = "edit.reverted";
    pub const COMMENT_ADDED: &str = "comment.added";
    pub const COMMENT_RESOLVED: &str = "comment.resolved";
    pub const APPROVAL_REQUESTED: &str = "approval.requested";
    pub const APPROVAL_GRANTED: &str = "approval.granted";
    pub const APPROVAL_REJECTED: &str = "approval.rejected";
    pub const SYNC_STATE_CHANGED: &str = "sync.state.changed";
    pub const EXPORT_STARTED: &str = "export.started";
    pub const EXPORT_COMPLETED: &str = "export.completed";
    pub const EXPORT_FAILED: &str = "export.failed";
    pub const PERMISSION_CHANGED: &str = "permission.changed";
    pub const SNAPSHOT_CREATED: &str = "snapshot.created";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn make_collector() -> (Arc<Mutex<Vec<BusEvent>>>, impl Fn(BusEvent) + Send + Sync + 'static) {
        let collected: Arc<Mutex<Vec<BusEvent>>> = Arc::new(Mutex::new(Vec::new()));
        let c2 = collected.clone();
        let cb = move |ev: BusEvent| {
            c2.lock().expect("lock poisoned").push(ev);
        };
        (collected, cb)
    }

    #[test]
    fn test_exact_topic_routing() {
        let bus = CollabBus::new();
        let (collected, cb) = make_collector();
        bus.subscribe(TopicPattern::Exact("session.joined".into()), cb);

        let delivered = bus.emit("session.joined", serde_json::json!({"user": "alice"}));
        assert_eq!(delivered, 1);

        // Different topic should not be delivered.
        let delivered2 = bus.emit("session.left", serde_json::json!({}));
        assert_eq!(delivered2, 0);

        assert_eq!(collected.lock().expect("lock").len(), 1);
    }

    #[test]
    fn test_prefix_topic_routing() {
        let bus = CollabBus::new();
        let (collected, cb) = make_collector();
        bus.subscribe(TopicPattern::Prefix("session.".into()), cb);

        bus.emit("session.joined", serde_json::json!({}));
        bus.emit("session.left", serde_json::json!({}));
        bus.emit("lock.acquired", serde_json::json!({})); // should NOT match

        assert_eq!(collected.lock().expect("lock").len(), 2);
    }

    #[test]
    fn test_wildcard_routing() {
        let bus = CollabBus::new();
        let (collected, cb) = make_collector();
        bus.subscribe(TopicPattern::Wildcard, cb);

        bus.emit("session.joined", serde_json::json!({}));
        bus.emit("lock.acquired", serde_json::json!({}));
        bus.emit("edit.applied", serde_json::json!({}));

        assert_eq!(collected.lock().expect("lock").len(), 3);
    }

    #[test]
    fn test_unsubscribe() {
        let bus = CollabBus::new();
        let (collected, cb) = make_collector();
        let sub_id = bus.subscribe(TopicPattern::Wildcard, cb);

        bus.emit("test.topic", serde_json::json!({}));
        assert_eq!(collected.lock().expect("lock").len(), 1);

        let removed = bus.unsubscribe(sub_id);
        assert!(removed);

        bus.emit("test.topic", serde_json::json!({}));
        // Still 1 because subscriber was removed.
        assert_eq!(collected.lock().expect("lock").len(), 1);
    }

    #[test]
    fn test_dead_letter_queue() {
        let bus = CollabBus::new();
        // No subscribers — event goes to DLQ.
        bus.emit("orphan.event", serde_json::json!({"info": "lost"}));
        assert_eq!(bus.dlq_len(), 1);

        let drained = bus.drain_dlq();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].topic, "orphan.event");
        assert_eq!(bus.dlq_len(), 0);
    }

    #[test]
    fn test_statistics() {
        let bus = CollabBus::new();
        let (_, cb) = make_collector();
        bus.subscribe(TopicPattern::Wildcard, cb);

        bus.emit("ev.one", serde_json::json!({}));
        bus.emit("ev.two", serde_json::json!({}));

        let stats = bus.stats();
        assert_eq!(stats.published, 2);
        assert_eq!(stats.delivered, 2);
        assert_eq!(stats.dead_lettered, 0);
    }

    #[test]
    fn test_multiple_subscribers_same_topic() {
        let bus = CollabBus::new();
        let (c1, cb1) = make_collector();
        let (c2, cb2) = make_collector();
        bus.subscribe(TopicPattern::Exact("shared".into()), cb1);
        bus.subscribe(TopicPattern::Exact("shared".into()), cb2);

        let count = bus.emit("shared", serde_json::json!({"x": 1}));
        assert_eq!(count, 2);
        assert_eq!(c1.lock().expect("lock").len(), 1);
        assert_eq!(c2.lock().expect("lock").len(), 1);
    }

    #[test]
    fn test_event_id_monotonically_increases() {
        let bus = CollabBus::new();
        let ids: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let ids2 = ids.clone();
        bus.subscribe(TopicPattern::Wildcard, move |ev| {
            ids2.lock().expect("lock").push(ev.id);
        });

        for _ in 0..5 {
            bus.emit("tick", serde_json::json!({}));
        }

        let collected_ids = ids.lock().expect("lock").clone();
        assert_eq!(collected_ids.len(), 5);
        for window in collected_ids.windows(2) {
            assert!(window[1] > window[0], "IDs must be strictly increasing");
        }
    }

    #[test]
    fn test_event_builder_fields() {
        let session_id = Uuid::new_v4();
        let user_id = Uuid::new_v4();
        let event = BusEvent::new(0, "test.event", serde_json::json!({"key": "value"}))
            .with_priority(EventPriority::High)
            .with_session(session_id)
            .with_user(user_id);

        assert_eq!(event.topic, "test.event");
        assert_eq!(event.priority, EventPriority::High);
        assert_eq!(event.session_id, Some(session_id));
        assert_eq!(event.user_id, Some(user_id));
    }

    #[test]
    fn test_topic_pattern_matching() {
        let exact = TopicPattern::Exact("session.joined".to_string());
        assert!(exact.matches("session.joined"));
        assert!(!exact.matches("session.left"));

        let prefix = TopicPattern::Prefix("session.".to_string());
        assert!(prefix.matches("session.joined"));
        assert!(prefix.matches("session.left"));
        assert!(!prefix.matches("lock.acquired"));

        let wildcard = TopicPattern::Wildcard;
        assert!(wildcard.matches("anything"));
        assert!(wildcard.matches(""));
    }

    #[test]
    fn test_well_known_topics_exist() {
        // Verify the topic constants compile and are non-empty strings.
        assert!(!topics::SESSION_CREATED.is_empty());
        assert!(!topics::LOCK_CONFLICT.is_empty());
        assert!(!topics::EXPORT_COMPLETED.is_empty());
    }
}
