//! Cloud event routing – event patterns, routing rules, target dispatch, and
//! dead-letter queue (DLQ) handling.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

/// The source service that emitted an event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum EventSource {
    /// Events emitted by the transcoding pipeline.
    Transcoding,
    /// Events emitted by the storage layer.
    Storage,
    /// Events emitted by the CDN.
    Cdn,
    /// Events emitted by auto-scaling.
    AutoScaling,
    /// A custom application-defined source.
    Custom(String),
}

/// Severity / type classification of an event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    /// A job or task completed successfully.
    Completed,
    /// A job or task failed.
    Failed,
    /// A warning condition was detected.
    Warning,
    /// An informational lifecycle event.
    Info,
    /// A scaling action was taken.
    Scaled,
}

/// A cloud event that flows through the event bridge.
#[derive(Debug, Clone)]
pub struct CloudEvent {
    /// Unique identifier for this event.
    pub id: String,
    /// Source that generated the event.
    pub source: EventSource,
    /// Classification of the event.
    pub event_type: EventType,
    /// Free-form payload attached to the event.
    pub payload: HashMap<String, String>,
}

impl CloudEvent {
    /// Constructs a new event with an empty payload.
    #[must_use]
    pub fn new(id: impl Into<String>, source: EventSource, event_type: EventType) -> Self {
        Self {
            id: id.into(),
            source,
            event_type,
            payload: HashMap::new(),
        }
    }

    /// Attaches a key-value pair to the event payload.
    pub fn with_payload(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.payload.insert(key.into(), value.into());
        self
    }
}

/// A routing rule that matches events and forwards them to a named target.
#[derive(Debug, Clone)]
pub struct RoutingRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// If `Some`, the rule only matches events from this source.
    pub source_filter: Option<EventSource>,
    /// If `Some`, the rule only matches events of this type.
    pub type_filter: Option<EventType>,
    /// Name of the target to deliver matching events to.
    pub target: String,
}

impl RoutingRule {
    /// Creates a rule that matches any event and sends it to `target`.
    #[must_use]
    pub fn catch_all(name: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_filter: None,
            type_filter: None,
            target: target.into(),
        }
    }

    /// Returns `true` if this rule matches the provided event.
    #[must_use]
    pub fn matches(&self, event: &CloudEvent) -> bool {
        let source_ok = self
            .source_filter
            .as_ref()
            .map_or(true, |s| s == &event.source);
        let type_ok = self
            .type_filter
            .as_ref()
            .map_or(true, |t| t == &event.event_type);
        source_ok && type_ok
    }
}

/// An entry in the dead-letter queue.
#[derive(Debug, Clone)]
pub struct DeadLetterEntry {
    /// The event that could not be delivered.
    pub event: CloudEvent,
    /// The name of the target that failed.
    pub failed_target: String,
    /// Human-readable reason for the failure.
    pub reason: String,
}

/// Delivery outcome returned by a target handler.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryResult {
    /// The event was delivered successfully.
    Ok,
    /// Delivery failed; includes a reason.
    Failed(String),
}

/// A registered event target identified by name.
#[derive(Debug, Clone)]
pub struct EventTarget {
    /// Unique name for this target.
    pub name: String,
    /// Events successfully delivered to this target (for inspection in tests).
    pub received: Vec<CloudEvent>,
}

impl EventTarget {
    /// Creates a new target with the given name.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            received: Vec::new(),
        }
    }

    /// Accepts a delivered event.
    pub fn deliver(&mut self, event: CloudEvent) -> DeliveryResult {
        self.received.push(event);
        DeliveryResult::Ok
    }
}

/// The event bridge: routes events through rules to targets and parks
/// undeliverable events in a dead-letter queue.
#[derive(Debug)]
pub struct EventBridge {
    rules: Vec<RoutingRule>,
    targets: HashMap<String, EventTarget>,
    dlq: Vec<DeadLetterEntry>,
    /// Total events that passed through the bridge.
    pub events_processed: u64,
}

impl EventBridge {
    /// Creates a new, empty event bridge.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            targets: HashMap::new(),
            dlq: Vec::new(),
            events_processed: 0,
        }
    }

    /// Registers a routing rule.
    pub fn add_rule(&mut self, rule: RoutingRule) {
        self.rules.push(rule);
    }

    /// Registers a target.
    pub fn add_target(&mut self, target: EventTarget) {
        self.targets.insert(target.name.clone(), target);
    }

    /// Dispatches an event through all matching rules, delivering to targets.
    /// Events with no matching rule, or whose target is unknown, go to the DLQ.
    pub fn dispatch(&mut self, event: CloudEvent) {
        self.events_processed += 1;
        let mut matched = false;
        // Collect matching targets first to avoid borrow-checker issues.
        let matching_targets: Vec<String> = self
            .rules
            .iter()
            .filter(|r| r.matches(&event))
            .map(|r| r.target.clone())
            .collect();

        for target_name in matching_targets {
            matched = true;
            if let Some(target) = self.targets.get_mut(&target_name) {
                target.deliver(event.clone());
            } else {
                self.dlq.push(DeadLetterEntry {
                    event: event.clone(),
                    failed_target: target_name.clone(),
                    reason: "Target not registered".to_string(),
                });
            }
        }

        if !matched {
            self.dlq.push(DeadLetterEntry {
                event,
                failed_target: String::new(),
                reason: "No matching rule".to_string(),
            });
        }
    }

    /// Returns all entries currently in the dead-letter queue.
    #[must_use]
    pub fn dead_letter_queue(&self) -> &[DeadLetterEntry] {
        &self.dlq
    }

    /// Returns the number of events in the dead-letter queue.
    #[must_use]
    pub fn dlq_len(&self) -> usize {
        self.dlq.len()
    }

    /// Drains and returns all dead-letter entries, clearing the DLQ.
    pub fn drain_dlq(&mut self) -> Vec<DeadLetterEntry> {
        std::mem::take(&mut self.dlq)
    }

    /// Returns a reference to a target by name.
    #[must_use]
    pub fn target(&self, name: &str) -> Option<&EventTarget> {
        self.targets.get(name)
    }
}

impl Default for EventBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str) -> CloudEvent {
        CloudEvent::new(id, EventSource::Transcoding, EventType::Completed)
    }

    fn make_bridge() -> EventBridge {
        let mut bridge = EventBridge::new();
        bridge.add_rule(RoutingRule::catch_all("all", "sink"));
        bridge.add_target(EventTarget::new("sink"));
        bridge
    }

    #[test]
    fn test_dispatch_to_registered_target() {
        let mut bridge = make_bridge();
        bridge.dispatch(make_event("evt-1"));
        assert_eq!(
            bridge
                .target("sink")
                .expect("target should succeed")
                .received
                .len(),
            1
        );
    }

    #[test]
    fn test_no_dlq_on_successful_delivery() {
        let mut bridge = make_bridge();
        bridge.dispatch(make_event("evt-2"));
        assert_eq!(bridge.dlq_len(), 0);
    }

    #[test]
    fn test_unmatched_event_goes_to_dlq() {
        let mut bridge = EventBridge::new();
        // No rules registered
        bridge.dispatch(make_event("evt-3"));
        assert_eq!(bridge.dlq_len(), 1);
        assert_eq!(bridge.dead_letter_queue()[0].reason, "No matching rule");
    }

    #[test]
    fn test_unknown_target_goes_to_dlq() {
        let mut bridge = EventBridge::new();
        bridge.add_rule(RoutingRule::catch_all("r", "missing-target"));
        // target "missing-target" is NOT registered
        bridge.dispatch(make_event("evt-4"));
        assert_eq!(bridge.dlq_len(), 1);
        assert!(bridge.dead_letter_queue()[0]
            .reason
            .contains("not registered"));
    }

    #[test]
    fn test_source_filter_matches() {
        let mut bridge = EventBridge::new();
        bridge.add_rule(RoutingRule {
            name: "only-storage".into(),
            source_filter: Some(EventSource::Storage),
            type_filter: None,
            target: "sink".into(),
        });
        bridge.add_target(EventTarget::new("sink"));

        let storage_evt = CloudEvent::new("s1", EventSource::Storage, EventType::Info);
        let cdn_evt = CloudEvent::new("c1", EventSource::Cdn, EventType::Info);

        bridge.dispatch(storage_evt);
        bridge.dispatch(cdn_evt);

        assert_eq!(
            bridge
                .target("sink")
                .expect("target should succeed")
                .received
                .len(),
            1
        );
        assert_eq!(bridge.dlq_len(), 1); // cdn event unmatched
    }

    #[test]
    fn test_type_filter_matches() {
        let mut bridge = EventBridge::new();
        bridge.add_rule(RoutingRule {
            name: "only-failed".into(),
            source_filter: None,
            type_filter: Some(EventType::Failed),
            target: "alerts".into(),
        });
        bridge.add_target(EventTarget::new("alerts"));

        bridge.dispatch(CloudEvent::new(
            "f1",
            EventSource::Transcoding,
            EventType::Failed,
        ));
        bridge.dispatch(CloudEvent::new(
            "ok1",
            EventSource::Transcoding,
            EventType::Completed,
        ));

        assert_eq!(
            bridge
                .target("alerts")
                .expect("target should succeed")
                .received
                .len(),
            1
        );
    }

    #[test]
    fn test_multiple_rules_can_deliver_to_multiple_targets() {
        let mut bridge = EventBridge::new();
        bridge.add_rule(RoutingRule::catch_all("r1", "t1"));
        bridge.add_rule(RoutingRule::catch_all("r2", "t2"));
        bridge.add_target(EventTarget::new("t1"));
        bridge.add_target(EventTarget::new("t2"));

        bridge.dispatch(make_event("multi"));
        assert_eq!(
            bridge
                .target("t1")
                .expect("target should succeed")
                .received
                .len(),
            1
        );
        assert_eq!(
            bridge
                .target("t2")
                .expect("target should succeed")
                .received
                .len(),
            1
        );
    }

    #[test]
    fn test_events_processed_counter() {
        let mut bridge = make_bridge();
        bridge.dispatch(make_event("a"));
        bridge.dispatch(make_event("b"));
        assert_eq!(bridge.events_processed, 2);
    }

    #[test]
    fn test_drain_dlq_clears_queue() {
        let mut bridge = EventBridge::new();
        bridge.dispatch(make_event("x"));
        let drained = bridge.drain_dlq();
        assert_eq!(drained.len(), 1);
        assert_eq!(bridge.dlq_len(), 0);
    }

    #[test]
    fn test_payload_roundtrip() {
        let event =
            CloudEvent::new("p1", EventSource::Cdn, EventType::Info).with_payload("key", "value");
        assert_eq!(event.payload.get("key").map(String::as_str), Some("value"));
    }

    #[test]
    fn test_routing_rule_catch_all_always_matches() {
        let rule = RoutingRule::catch_all("all", "t");
        let events = vec![
            CloudEvent::new("e1", EventSource::Storage, EventType::Completed),
            CloudEvent::new("e2", EventSource::Cdn, EventType::Failed),
            CloudEvent::new("e3", EventSource::AutoScaling, EventType::Scaled),
        ];
        for e in &events {
            assert!(rule.matches(e));
        }
    }

    #[test]
    fn test_default_trait() {
        let bridge = EventBridge::default();
        assert_eq!(bridge.events_processed, 0);
        assert_eq!(bridge.dlq_len(), 0);
    }

    #[test]
    fn test_custom_source_roundtrip() {
        let src = EventSource::Custom("my-app".to_string());
        let event = CloudEvent::new("c", src.clone(), EventType::Info);
        assert_eq!(event.source, src);
    }
}
