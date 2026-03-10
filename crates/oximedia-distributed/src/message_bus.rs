//! Publish/subscribe message bus for distributed coordination.
//!
//! This module implements an in-process message bus that supports
//! typed topics, publish/subscribe patterns, and message filtering
//! for coordinating distributed encoding nodes.

#![allow(dead_code)]

use std::collections::HashMap;
use uuid::Uuid;

/// Types of messages that can be sent on the bus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MessageType {
    /// A task has been submitted
    TaskSubmitted,
    /// A task has been assigned to a worker
    TaskAssigned,
    /// A task has completed
    TaskCompleted,
    /// A task has failed
    TaskFailed,
    /// A node has joined the cluster
    NodeJoined,
    /// A node has left the cluster
    NodeLeft,
    /// A node's health status changed
    HealthChanged,
    /// A heartbeat from a node
    Heartbeat,
    /// Cluster configuration changed
    ConfigChanged,
    /// Custom application-level message
    Custom,
}

impl MessageType {
    /// Returns the topic string for this message type.
    #[must_use]
    pub fn topic(&self) -> &'static str {
        match self {
            Self::TaskSubmitted => "task.submitted",
            Self::TaskAssigned => "task.assigned",
            Self::TaskCompleted => "task.completed",
            Self::TaskFailed => "task.failed",
            Self::NodeJoined => "node.joined",
            Self::NodeLeft => "node.left",
            Self::HealthChanged => "node.health",
            Self::Heartbeat => "node.heartbeat",
            Self::ConfigChanged => "cluster.config",
            Self::Custom => "custom",
        }
    }

    /// Returns true if this is a task-related message.
    #[must_use]
    pub fn is_task_event(&self) -> bool {
        matches!(
            self,
            Self::TaskSubmitted | Self::TaskAssigned | Self::TaskCompleted | Self::TaskFailed
        )
    }

    /// Returns true if this is a node-related message.
    #[must_use]
    pub fn is_node_event(&self) -> bool {
        matches!(
            self,
            Self::NodeJoined | Self::NodeLeft | Self::HealthChanged | Self::Heartbeat
        )
    }
}

/// A message that flows through the bus.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BusMessage {
    /// Unique message identifier
    pub id: Uuid,
    /// Message type / topic
    pub message_type: MessageType,
    /// Source node that sent the message
    pub source: Uuid,
    /// Serialized payload
    pub payload: String,
    /// Unix timestamp when the message was created
    pub timestamp: i64,
    /// Optional correlation ID for request-reply patterns
    pub correlation_id: Option<Uuid>,
    /// Message headers (metadata)
    pub headers: HashMap<String, String>,
}

impl BusMessage {
    /// Creates a new bus message.
    #[must_use]
    pub fn new(message_type: MessageType, source: Uuid, payload: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type,
            source,
            payload: payload.to_string(),
            timestamp: chrono::Utc::now().timestamp(),
            correlation_id: None,
            headers: HashMap::new(),
        }
    }

    /// Sets a correlation ID for request-reply patterns.
    #[must_use]
    pub fn with_correlation(mut self, correlation_id: Uuid) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    /// Adds a header to the message.
    #[must_use]
    pub fn with_header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Returns the age of the message in seconds relative to a reference time.
    #[must_use]
    pub fn age_secs(&self, now: i64) -> i64 {
        now - self.timestamp
    }
}

/// Subscription identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(Uuid);

impl SubscriptionId {
    /// Creates a new subscription ID.
    #[must_use]
    fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

/// A subscription to specific message types.
#[derive(Debug, Clone)]
struct Subscription {
    /// Subscription identifier
    id: SubscriptionId,
    /// Subscriber identifier
    subscriber_id: Uuid,
    /// Message types this subscription listens for
    types: Vec<MessageType>,
}

/// Bus statistics for monitoring.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct BusStats {
    /// Total messages published
    pub messages_published: u64,
    /// Total messages delivered to subscribers
    pub messages_delivered: u64,
    /// Total messages dropped (no subscribers)
    pub messages_dropped: u64,
    /// Current number of active subscriptions
    pub active_subscriptions: u64,
}

/// An in-process publish/subscribe message bus.
///
/// Supports topic-based subscriptions where subscribers can listen
/// for specific message types. Messages published to the bus are
/// stored in per-subscriber mailboxes for retrieval.
#[derive(Debug)]
pub struct MessageBus {
    /// Active subscriptions
    subscriptions: Vec<Subscription>,
    /// Per-subscriber mailbox
    mailboxes: HashMap<SubscriptionId, Vec<BusMessage>>,
    /// Maximum mailbox size
    max_mailbox_size: usize,
    /// Statistics
    stats: BusStats,
}

impl MessageBus {
    /// Creates a new message bus.
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscriptions: Vec::new(),
            mailboxes: HashMap::new(),
            max_mailbox_size: 1000,
            stats: BusStats::default(),
        }
    }

    /// Creates a new message bus with a custom mailbox size limit.
    #[must_use]
    pub fn with_mailbox_size(max_size: usize) -> Self {
        Self {
            subscriptions: Vec::new(),
            mailboxes: HashMap::new(),
            max_mailbox_size: max_size,
            stats: BusStats::default(),
        }
    }

    /// Subscribes to specific message types.
    ///
    /// Returns a `SubscriptionId` that can be used to receive messages
    /// or unsubscribe later.
    pub fn subscribe(&mut self, subscriber_id: Uuid, types: Vec<MessageType>) -> SubscriptionId {
        let sub_id = SubscriptionId::new();
        self.subscriptions.push(Subscription {
            id: sub_id,
            subscriber_id,
            types,
        });
        self.mailboxes.insert(sub_id, Vec::new());
        self.stats.active_subscriptions += 1;
        sub_id
    }

    /// Unsubscribes and removes the subscription.
    pub fn unsubscribe(&mut self, sub_id: &SubscriptionId) -> bool {
        let before = self.subscriptions.len();
        self.subscriptions.retain(|s| s.id != *sub_id);
        self.mailboxes.remove(sub_id);
        let removed = self.subscriptions.len() < before;
        if removed {
            self.stats.active_subscriptions = self.stats.active_subscriptions.saturating_sub(1);
        }
        removed
    }

    /// Publishes a message to all matching subscribers.
    ///
    /// Returns the number of subscribers the message was delivered to.
    pub fn publish(&mut self, message: BusMessage) -> usize {
        self.stats.messages_published += 1;

        let matching_subs: Vec<SubscriptionId> = self
            .subscriptions
            .iter()
            .filter(|sub| sub.types.contains(&message.message_type))
            .map(|sub| sub.id)
            .collect();

        if matching_subs.is_empty() {
            self.stats.messages_dropped += 1;
            return 0;
        }

        let mut delivered = 0;
        for sub_id in &matching_subs {
            if let Some(mailbox) = self.mailboxes.get_mut(sub_id) {
                if mailbox.len() < self.max_mailbox_size {
                    mailbox.push(message.clone());
                    delivered += 1;
                }
            }
        }

        self.stats.messages_delivered += delivered as u64;
        delivered
    }

    /// Receives all pending messages for a subscription.
    ///
    /// Drains the mailbox, returning all messages.
    pub fn receive(&mut self, sub_id: &SubscriptionId) -> Vec<BusMessage> {
        self.mailboxes
            .get_mut(sub_id)
            .map(std::mem::take)
            .unwrap_or_default()
    }

    /// Returns the number of pending messages for a subscription.
    #[must_use]
    pub fn pending_count(&self, sub_id: &SubscriptionId) -> usize {
        self.mailboxes.get(sub_id).map_or(0, Vec::len)
    }

    /// Returns the total number of active subscriptions.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Returns bus statistics.
    #[must_use]
    pub fn stats(&self) -> &BusStats {
        &self.stats
    }

    /// Clears all mailboxes without removing subscriptions.
    pub fn clear_mailboxes(&mut self) {
        for mailbox in self.mailboxes.values_mut() {
            mailbox.clear();
        }
    }
}

impl Default for MessageBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_message_type_topics() {
        assert_eq!(MessageType::TaskSubmitted.topic(), "task.submitted");
        assert_eq!(MessageType::NodeJoined.topic(), "node.joined");
        assert_eq!(MessageType::Custom.topic(), "custom");
    }

    #[test]
    fn test_message_type_is_task_event() {
        assert!(MessageType::TaskSubmitted.is_task_event());
        assert!(MessageType::TaskCompleted.is_task_event());
        assert!(!MessageType::NodeJoined.is_task_event());
        assert!(!MessageType::Heartbeat.is_task_event());
    }

    #[test]
    fn test_message_type_is_node_event() {
        assert!(MessageType::NodeJoined.is_node_event());
        assert!(MessageType::Heartbeat.is_node_event());
        assert!(!MessageType::TaskSubmitted.is_node_event());
    }

    #[test]
    fn test_bus_message_creation() {
        let s = src();
        let msg = BusMessage::new(MessageType::TaskSubmitted, s, "{\"task_id\":\"abc\"}");
        assert_eq!(msg.message_type, MessageType::TaskSubmitted);
        assert_eq!(msg.source, s);
        assert!(msg.correlation_id.is_none());
    }

    #[test]
    fn test_bus_message_with_correlation() {
        let corr = Uuid::new_v4();
        let msg = BusMessage::new(MessageType::TaskCompleted, src(), "{}").with_correlation(corr);
        assert_eq!(msg.correlation_id, Some(corr));
    }

    #[test]
    fn test_bus_message_with_header() {
        let msg = BusMessage::new(MessageType::Custom, src(), "{}")
            .with_header("priority", "high")
            .with_header("region", "us-east");
        assert_eq!(msg.headers.len(), 2);
        assert_eq!(
            msg.headers
                .get("priority")
                .expect("get should return a value"),
            "high"
        );
    }

    #[test]
    fn test_bus_message_age() {
        let mut msg = BusMessage::new(MessageType::Heartbeat, src(), "{}");
        msg.timestamp = 1000;
        assert_eq!(msg.age_secs(1050), 50);
    }

    #[test]
    fn test_subscribe_and_publish() {
        let mut bus = MessageBus::new();
        let sub = bus.subscribe(src(), vec![MessageType::TaskSubmitted]);
        let msg = BusMessage::new(MessageType::TaskSubmitted, src(), "{}");
        let delivered = bus.publish(msg);
        assert_eq!(delivered, 1);
        let received = bus.receive(&sub);
        assert_eq!(received.len(), 1);
    }

    #[test]
    fn test_publish_no_subscribers() {
        let mut bus = MessageBus::new();
        let msg = BusMessage::new(MessageType::TaskSubmitted, src(), "{}");
        let delivered = bus.publish(msg);
        assert_eq!(delivered, 0);
        assert_eq!(bus.stats().messages_dropped, 1);
    }

    #[test]
    fn test_subscribe_filters_by_type() {
        let mut bus = MessageBus::new();
        let sub = bus.subscribe(src(), vec![MessageType::TaskCompleted]);
        // Publish a different type
        bus.publish(BusMessage::new(MessageType::TaskSubmitted, src(), "{}"));
        let received = bus.receive(&sub);
        assert!(received.is_empty());
    }

    #[test]
    fn test_multiple_subscribers() {
        let mut bus = MessageBus::new();
        let sub1 = bus.subscribe(src(), vec![MessageType::NodeJoined]);
        let sub2 = bus.subscribe(src(), vec![MessageType::NodeJoined]);
        bus.publish(BusMessage::new(MessageType::NodeJoined, src(), "{}"));
        assert_eq!(bus.receive(&sub1).len(), 1);
        assert_eq!(bus.receive(&sub2).len(), 1);
    }

    #[test]
    fn test_unsubscribe() {
        let mut bus = MessageBus::new();
        let sub = bus.subscribe(src(), vec![MessageType::Heartbeat]);
        assert_eq!(bus.subscription_count(), 1);
        assert!(bus.unsubscribe(&sub));
        assert_eq!(bus.subscription_count(), 0);
    }

    #[test]
    fn test_pending_count() {
        let mut bus = MessageBus::new();
        let sub = bus.subscribe(src(), vec![MessageType::TaskFailed]);
        assert_eq!(bus.pending_count(&sub), 0);
        bus.publish(BusMessage::new(MessageType::TaskFailed, src(), "{}"));
        bus.publish(BusMessage::new(MessageType::TaskFailed, src(), "{}"));
        assert_eq!(bus.pending_count(&sub), 2);
    }

    #[test]
    fn test_mailbox_size_limit() {
        let mut bus = MessageBus::with_mailbox_size(2);
        let sub = bus.subscribe(src(), vec![MessageType::Heartbeat]);
        bus.publish(BusMessage::new(MessageType::Heartbeat, src(), "1"));
        bus.publish(BusMessage::new(MessageType::Heartbeat, src(), "2"));
        bus.publish(BusMessage::new(MessageType::Heartbeat, src(), "3"));
        assert_eq!(bus.pending_count(&sub), 2); // capped at 2
    }

    #[test]
    fn test_clear_mailboxes() {
        let mut bus = MessageBus::new();
        let sub = bus.subscribe(src(), vec![MessageType::TaskAssigned]);
        bus.publish(BusMessage::new(MessageType::TaskAssigned, src(), "{}"));
        bus.clear_mailboxes();
        assert_eq!(bus.pending_count(&sub), 0);
        assert_eq!(bus.subscription_count(), 1); // sub still exists
    }

    #[test]
    fn test_bus_stats() {
        let mut bus = MessageBus::new();
        let _sub = bus.subscribe(src(), vec![MessageType::TaskSubmitted]);
        bus.publish(BusMessage::new(MessageType::TaskSubmitted, src(), "{}"));
        bus.publish(BusMessage::new(MessageType::NodeLeft, src(), "{}")); // no subscriber
        assert_eq!(bus.stats().messages_published, 2);
        assert_eq!(bus.stats().messages_delivered, 1);
        assert_eq!(bus.stats().messages_dropped, 1);
    }
}
