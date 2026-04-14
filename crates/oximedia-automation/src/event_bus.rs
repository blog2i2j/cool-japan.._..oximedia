//! Automation event bus for broadcast workflow messaging.
//!
//! Provides a publish/subscribe event system for decoupled automation components.
//!
//! Two implementations are available:
//!
//! * [`EventBus`] — simple, synchronous, single-threaded bus suitable for
//!   unit tests and low-throughput scenarios.
//! * [`LockFreeEventBus`] — multi-producer, multi-consumer bus backed by
//!   `crossbeam-channel` bounded queues.  Each call to
//!   [`LockFreeEventBus::subscribe`] creates an independent receiver that
//!   receives every subsequent event published via
//!   [`LockFreeEventBus::publish`].

use std::sync::{Arc, Mutex};

/// An automation event published to the event bus.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationEvent {
    /// Unique event identifier.
    pub id: u64,
    /// Category or type of the event.
    pub event_type: String,
    /// Source component that generated the event.
    pub source: String,
    /// Arbitrary payload associated with the event.
    pub payload: String,
    /// Millisecond timestamp when the event was created.
    pub timestamp_ms: u64,
}

impl AutomationEvent {
    /// Create a new automation event.
    pub fn new(id: u64, event_type: &str, source: &str, payload: &str, timestamp_ms: u64) -> Self {
        Self {
            id,
            event_type: event_type.to_string(),
            source: source.to_string(),
            payload: payload.to_string(),
            timestamp_ms,
        }
    }

    /// Return how many milliseconds ago this event was created relative to `now`.
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// A subscription that determines which events a subscriber receives.
#[derive(Debug, Clone)]
pub struct EventSubscription {
    /// Identifier for the subscribing component.
    pub subscriber_id: u64,
    /// If `Some`, only events with a matching `event_type` are delivered.
    pub event_type_filter: Option<String>,
    /// If `Some`, only events with a matching `source` are delivered.
    pub source_filter: Option<String>,
}

impl EventSubscription {
    /// Create a new subscription.
    pub fn new(
        subscriber_id: u64,
        event_type_filter: Option<&str>,
        source_filter: Option<&str>,
    ) -> Self {
        Self {
            subscriber_id,
            event_type_filter: event_type_filter.map(str::to_string),
            source_filter: source_filter.map(str::to_string),
        }
    }

    /// Return `true` if `event` satisfies this subscription's filters.
    pub fn matches(&self, event: &AutomationEvent) -> bool {
        if let Some(ref et) = self.event_type_filter {
            if &event.event_type != et {
                return false;
            }
        }
        if let Some(ref sf) = self.source_filter {
            if &event.source != sf {
                return false;
            }
        }
        true
    }
}

/// A simple in-process event bus for automation components.
#[derive(Debug, Default)]
pub struct EventBus {
    events: Vec<AutomationEvent>,
    subscriptions: Vec<EventSubscription>,
    next_id: u64,
}

impl EventBus {
    /// Create a new, empty event bus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish an event and return the assigned event id.
    pub fn publish(&mut self, event_type: &str, source: &str, payload: &str) -> u64 {
        self.publish_at(event_type, source, payload, 0)
    }

    /// Publish an event with an explicit timestamp and return the assigned event id.
    pub fn publish_at(
        &mut self,
        event_type: &str,
        source: &str,
        payload: &str,
        timestamp_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.events.push(AutomationEvent::new(
            id,
            event_type,
            source,
            payload,
            timestamp_ms,
        ));
        id
    }

    /// Register a subscription and return the subscription's `subscriber_id`.
    pub fn subscribe(&mut self, sub: EventSubscription) -> u64 {
        let id = sub.subscriber_id;
        self.subscriptions.push(sub);
        id
    }

    /// Return all events that match the subscription registered for `subscriber_id`.
    pub fn events_for(&self, subscriber_id: u64) -> Vec<&AutomationEvent> {
        let sub = self
            .subscriptions
            .iter()
            .find(|s| s.subscriber_id == subscriber_id);
        match sub {
            None => vec![],
            Some(sub) => self.events.iter().filter(|e| sub.matches(e)).collect(),
        }
    }

    /// Remove events older than `max_age_ms` milliseconds relative to `now_ms`.
    pub fn clear_old(&mut self, max_age_ms: u64, now_ms: u64) {
        self.events.retain(|e| e.age_ms(now_ms) <= max_age_ms);
    }

    /// Return the total number of events currently held in the bus.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Return the total number of subscriptions registered.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }
}

// ── Lock-free event bus ────────────────────────────────────────────────────────

/// Maximum number of events that may be buffered in each subscriber's channel
/// before the oldest events are dropped on overflow.
pub const DEFAULT_BUS_CAPACITY: usize = 1024;

/// A receiver end for a [`LockFreeEventBus`] subscription.
///
/// Each [`EventReceiver`] is independent — events published on the bus are
/// delivered to all currently-registered receivers.  Cloning an
/// [`EventReceiver`] creates a second receiver that shares the same underlying
/// crossbeam channel; i.e. the two clones compete for messages.  To create a
/// fully independent subscriber call [`LockFreeEventBus::subscribe`] again.
pub struct EventReceiver {
    rx: crossbeam_channel::Receiver<AutomationEvent>,
}

impl EventReceiver {
    /// Block until an event arrives or the bus is closed.
    ///
    /// Returns `None` if the bus has been dropped and no further events will
    /// arrive.
    pub fn recv(&self) -> Option<AutomationEvent> {
        self.rx.recv().ok()
    }

    /// Non-blocking receive — returns `None` immediately if no event is
    /// available.
    pub fn try_recv(&self) -> Option<AutomationEvent> {
        self.rx.try_recv().ok()
    }

    /// Return `true` if at least one event is currently queued.
    pub fn has_pending(&self) -> bool {
        !self.rx.is_empty()
    }

    /// Drain all currently-queued events into a `Vec`.
    pub fn drain(&self) -> Vec<AutomationEvent> {
        self.rx.try_iter().collect()
    }
}

/// Internal shared state for [`LockFreeEventBus`].
struct BusInner {
    /// All active subscriber send-halves.
    senders: Vec<crossbeam_channel::Sender<AutomationEvent>>,
    /// Monotonically increasing event ID.
    next_id: u64,
}

/// A lock-free, multi-subscriber event bus backed by `crossbeam-channel`.
///
/// # Design
///
/// Each call to [`subscribe`](Self::subscribe) creates a new bounded channel
/// with capacity [`DEFAULT_BUS_CAPACITY`].  The `Sender` half is stored
/// internally; the `Receiver` half is returned as an [`EventReceiver`].
///
/// When [`publish`](Self::publish) is called, the event is sent to every
/// registered sender via [`try_send`](crossbeam_channel::Sender::try_send).
/// If a subscriber's channel is full the event is silently dropped for that
/// subscriber (back-pressure via drop), keeping publishers non-blocking.
///
/// The internal sender list is protected by a `Mutex<BusInner>` only during
/// registration and publication — the event delivery path through the channel
/// is fully lock-free.
///
/// # Example
///
/// ```rust
/// use oximedia_automation::event_bus::LockFreeEventBus;
///
/// let bus = LockFreeEventBus::new(128);
/// let rx = bus.subscribe();
///
/// bus.publish("clip_start", "playout", "id=1");
/// let ev = rx.try_recv().expect("event should be available");
/// assert_eq!(ev.event_type, "clip_start");
/// ```
pub struct LockFreeEventBus {
    inner: Arc<Mutex<BusInner>>,
    capacity: usize,
}

impl LockFreeEventBus {
    /// Create a new lock-free event bus with the given per-subscriber channel
    /// capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BusInner {
                senders: Vec::new(),
                next_id: 0,
            })),
            capacity,
        }
    }

    /// Create with the default capacity ([`DEFAULT_BUS_CAPACITY`]).
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_BUS_CAPACITY)
    }

    /// Register a new subscriber and return an [`EventReceiver`] that will
    /// receive all future events.
    ///
    /// Subscribers created before an event is published will receive that
    /// event.  Subscribers created *after* an event is published will not
    /// receive it retroactively.
    pub fn subscribe(&self) -> EventReceiver {
        let (tx, rx) = crossbeam_channel::bounded(self.capacity);
        if let Ok(mut guard) = self.inner.lock() {
            guard.senders.push(tx);
        }
        EventReceiver { rx }
    }

    /// Publish an event to all currently-registered subscribers.
    ///
    /// Dead subscribers (those whose [`EventReceiver`] has been dropped) are
    /// pruned from the internal sender list automatically.  If a subscriber's
    /// channel buffer is full the event is dropped for that subscriber only.
    ///
    /// Returns the auto-assigned event ID.
    pub fn publish(&self, event_type: &str, source: &str, payload: &str) -> u64 {
        self.publish_at(event_type, source, payload, 0)
    }

    /// Publish an event with an explicit timestamp.
    pub fn publish_at(
        &self,
        event_type: &str,
        source: &str,
        payload: &str,
        timestamp_ms: u64,
    ) -> u64 {
        let mut guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return 0, // Mutex poisoned — degrade gracefully.
        };

        let id = guard.next_id;
        guard.next_id += 1;

        let event = AutomationEvent::new(id, event_type, source, payload, timestamp_ms);

        // Send to all subscribers and collect indices of dead channels.
        let mut dead: Vec<usize> = Vec::new();
        for (idx, tx) in guard.senders.iter().enumerate() {
            match tx.try_send(event.clone()) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(_)) => {
                    // Subscriber is slow — drop this event for them.
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                    // Subscriber has been dropped — mark for removal.
                    dead.push(idx);
                }
            }
        }

        // Remove dead senders in reverse order to preserve indices.
        for idx in dead.into_iter().rev() {
            guard.senders.swap_remove(idx);
        }

        id
    }

    /// Return the number of currently-active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.inner.lock().map(|g| g.senders.len()).unwrap_or(0)
    }

    /// Return the number of events that have been published so far.
    pub fn published_count(&self) -> u64 {
        self.inner.lock().map(|g| g.next_id).unwrap_or(0)
    }
}

impl Clone for LockFreeEventBus {
    /// Clone returns a new handle to the **same** underlying bus.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            capacity: self.capacity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_new() {
        let ev = AutomationEvent::new(1, "clip_start", "playout", "clip_id=42", 1000);
        assert_eq!(ev.id, 1);
        assert_eq!(ev.event_type, "clip_start");
        assert_eq!(ev.source, "playout");
        assert_eq!(ev.payload, "clip_id=42");
        assert_eq!(ev.timestamp_ms, 1000);
    }

    #[test]
    fn test_event_age_ms_normal() {
        let ev = AutomationEvent::new(0, "t", "s", "", 500);
        assert_eq!(ev.age_ms(1000), 500);
    }

    #[test]
    fn test_event_age_ms_zero_when_same() {
        let ev = AutomationEvent::new(0, "t", "s", "", 500);
        assert_eq!(ev.age_ms(500), 0);
    }

    #[test]
    fn test_event_age_ms_saturating() {
        let ev = AutomationEvent::new(0, "t", "s", "", 1000);
        // now < timestamp => saturating_sub returns 0
        assert_eq!(ev.age_ms(500), 0);
    }

    #[test]
    fn test_subscription_matches_no_filter() {
        let sub = EventSubscription::new(1, None, None);
        let ev = AutomationEvent::new(0, "any_type", "any_source", "", 0);
        assert!(sub.matches(&ev));
    }

    #[test]
    fn test_subscription_matches_event_type_filter_pass() {
        let sub = EventSubscription::new(1, Some("clip_start"), None);
        let ev = AutomationEvent::new(0, "clip_start", "playout", "", 0);
        assert!(sub.matches(&ev));
    }

    #[test]
    fn test_subscription_matches_event_type_filter_fail() {
        let sub = EventSubscription::new(1, Some("clip_start"), None);
        let ev = AutomationEvent::new(0, "clip_end", "playout", "", 0);
        assert!(!sub.matches(&ev));
    }

    #[test]
    fn test_subscription_matches_source_filter_pass() {
        let sub = EventSubscription::new(2, None, Some("switcher"));
        let ev = AutomationEvent::new(0, "cut", "switcher", "", 0);
        assert!(sub.matches(&ev));
    }

    #[test]
    fn test_subscription_matches_source_filter_fail() {
        let sub = EventSubscription::new(2, None, Some("switcher"));
        let ev = AutomationEvent::new(0, "cut", "router", "", 0);
        assert!(!sub.matches(&ev));
    }

    #[test]
    fn test_subscription_matches_both_filters() {
        let sub = EventSubscription::new(3, Some("alert"), Some("eas"));
        let good = AutomationEvent::new(0, "alert", "eas", "", 0);
        let bad_type = AutomationEvent::new(1, "info", "eas", "", 0);
        let bad_src = AutomationEvent::new(2, "alert", "monitor", "", 0);
        assert!(sub.matches(&good));
        assert!(!sub.matches(&bad_type));
        assert!(!sub.matches(&bad_src));
    }

    #[test]
    fn test_bus_publish_increments_ids() {
        let mut bus = EventBus::new();
        let id0 = bus.publish("a", "s", "p");
        let id1 = bus.publish("b", "s", "p");
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(bus.event_count(), 2);
    }

    #[test]
    fn test_bus_subscribe_returns_subscriber_id() {
        let mut bus = EventBus::new();
        let sub = EventSubscription::new(99, None, None);
        let returned = bus.subscribe(sub);
        assert_eq!(returned, 99);
        assert_eq!(bus.subscription_count(), 1);
    }

    #[test]
    fn test_bus_events_for_no_subscription() {
        let bus = EventBus::new();
        let events = bus.events_for(0);
        assert!(events.is_empty());
    }

    #[test]
    fn test_bus_events_for_filtered() {
        let mut bus = EventBus::new();
        bus.publish_at("clip_start", "playout", "", 100);
        bus.publish_at("clip_end", "playout", "", 200);
        bus.publish_at("clip_start", "playout", "", 300);

        let sub = EventSubscription::new(1, Some("clip_start"), None);
        bus.subscribe(sub);

        let events = bus.events_for(1);
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|e| e.event_type == "clip_start"));
    }

    #[test]
    fn test_bus_clear_old() {
        let mut bus = EventBus::new();
        bus.publish_at("a", "s", "", 100);
        bus.publish_at("b", "s", "", 500);
        bus.publish_at("c", "s", "", 900);

        // now = 1000; max_age = 400 => keep events with age <= 400, i.e. timestamp >= 600
        bus.clear_old(400, 1000);
        assert_eq!(bus.event_count(), 1);
    }

    // ── LockFreeEventBus tests ─────────────────────────────────────────────────

    #[test]
    fn test_lock_free_bus_publish_then_receive() {
        let bus = LockFreeEventBus::new(32);
        let rx = bus.subscribe();

        let id = bus.publish("clip_start", "playout", "id=42");
        assert_eq!(id, 0);

        let ev = rx
            .try_recv()
            .expect("event should be available immediately");
        assert_eq!(ev.event_type, "clip_start");
        assert_eq!(ev.source, "playout");
        assert_eq!(ev.payload, "id=42");
        assert_eq!(ev.id, 0);
    }

    #[test]
    fn test_lock_free_bus_multiple_subscribers_each_get_event() {
        let bus = LockFreeEventBus::new(32);
        let rx1 = bus.subscribe();
        let rx2 = bus.subscribe();
        let rx3 = bus.subscribe();

        bus.publish("alert", "eas", "tornado");

        let ev1 = rx1.try_recv().expect("rx1 should receive event");
        let ev2 = rx2.try_recv().expect("rx2 should receive event");
        let ev3 = rx3.try_recv().expect("rx3 should receive event");

        assert_eq!(ev1.event_type, "alert");
        assert_eq!(ev2.event_type, "alert");
        assert_eq!(ev3.event_type, "alert");
    }

    #[test]
    fn test_lock_free_bus_multiple_events_ordered() {
        let bus = LockFreeEventBus::new(64);
        let rx = bus.subscribe();

        bus.publish("a", "s", "1");
        bus.publish("b", "s", "2");
        bus.publish("c", "s", "3");

        let events = rx.drain();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, "a");
        assert_eq!(events[1].event_type, "b");
        assert_eq!(events[2].event_type, "c");
    }

    #[test]
    fn test_lock_free_bus_full_channel_drops_gracefully() {
        // Capacity of 2; publish 5 events — the bus must not block or panic.
        let bus = LockFreeEventBus::new(2);
        let rx = bus.subscribe();

        for i in 0u64..5 {
            bus.publish_at("ev", "src", "data", i);
        }
        // At most 2 events will be in the buffer; the rest were dropped.
        let received: Vec<_> = rx.drain();
        assert!(
            received.len() <= 2,
            "expected at most 2 buffered events, got {}",
            received.len()
        );
        // Importantly, the bus did not panic.
    }

    #[test]
    fn test_lock_free_bus_dead_subscriber_pruned() {
        let bus = LockFreeEventBus::new(8);
        {
            let _rx = bus.subscribe(); // dropped at end of block
        }
        assert_eq!(
            bus.subscriber_count(),
            1,
            "before publish, dead not yet pruned"
        );

        // Publish triggers pruning of the dead sender.
        bus.publish("x", "s", "");
        assert_eq!(
            bus.subscriber_count(),
            0,
            "dead subscriber should be pruned"
        );
    }

    #[test]
    fn test_lock_free_bus_published_count() {
        let bus = LockFreeEventBus::new(32);
        let _rx = bus.subscribe();

        assert_eq!(bus.published_count(), 0);
        bus.publish("a", "s", "");
        bus.publish("b", "s", "");
        assert_eq!(bus.published_count(), 2);
    }

    #[test]
    fn test_lock_free_bus_clone_shares_state() {
        let bus = LockFreeEventBus::new(32);
        let bus2 = bus.clone();
        let rx = bus.subscribe();

        // Publish via the clone — receiver on original bus should get it.
        bus2.publish("via_clone", "src", "payload");
        let ev = rx.try_recv().expect("event from clone should arrive");
        assert_eq!(ev.event_type, "via_clone");
    }

    #[test]
    fn test_lock_free_bus_subscriber_added_after_publish_misses_old_event() {
        let bus = LockFreeEventBus::new(32);
        bus.publish("old", "s", "");

        let rx_late = bus.subscribe();
        // The late subscriber must not see the old event.
        assert!(
            rx_late.try_recv().is_none(),
            "late subscriber should not see past events"
        );

        bus.publish("new", "s", "");
        let ev = rx_late
            .try_recv()
            .expect("late subscriber should see new event");
        assert_eq!(ev.event_type, "new");
    }

    #[test]
    fn test_lock_free_bus_has_pending() {
        let bus = LockFreeEventBus::new(32);
        let rx = bus.subscribe();

        assert!(!rx.has_pending(), "should be empty before publish");
        bus.publish("x", "s", "");
        assert!(rx.has_pending(), "should have pending after publish");
        rx.try_recv();
        assert!(!rx.has_pending(), "should be empty after consume");
    }
}
