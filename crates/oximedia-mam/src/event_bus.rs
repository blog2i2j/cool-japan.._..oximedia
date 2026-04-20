//! In-process publish/subscribe event bus for MAM system events.
//!
//! Provides a fan-out event distribution mechanism with:
//! - Glob pattern matching for subscriptions ("asset.*", "*", "workflow.completed")
//! - Thread-safe subscription registry
//! - Bounded event history with configurable max size
//! - Unique subscription IDs for lifecycle management

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// All MAM domain events that can be published on the bus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MamEvent {
    /// An asset has been successfully ingested into the system.
    AssetIngested {
        asset_id: String,
        path: String,
        size_bytes: u64,
    },
    /// An asset has been permanently deleted.
    AssetDeleted { asset_id: String },
    /// One or more metadata fields on an asset have changed.
    AssetUpdated {
        asset_id: String,
        changed_fields: Vec<String>,
    },
    /// Tags have been applied to an asset.
    AssetTagged { asset_id: String, tags: Vec<String> },
    /// A new collection has been created.
    CollectionCreated { collection_id: String, name: String },
    /// A collection has been deleted.
    CollectionDeleted { collection_id: String },
    /// A workflow has been started for an asset.
    WorkflowStarted {
        workflow_id: String,
        asset_id: String,
    },
    /// A workflow has finished (successfully or not).
    WorkflowCompleted {
        workflow_id: String,
        asset_id: String,
        success: bool,
    },
    /// A transcode job has started.
    TranscodeStarted { job_id: String, asset_id: String },
    /// A transcode job has finished and produced output.
    TranscodeCompleted {
        job_id: String,
        asset_id: String,
        output_path: String,
    },
    /// A transcode job has failed.
    TranscodeFailed {
        job_id: String,
        asset_id: String,
        error: String,
    },
    /// A user performed an action on a resource.
    UserAction {
        user_id: String,
        action: String,
        resource: String,
    },
    /// Storage utilisation has crossed a warning threshold.
    StorageWarning {
        used_bytes: u64,
        capacity_bytes: u64,
        threshold_pct: f32,
    },
}

impl MamEvent {
    /// Return the dot-separated event type string, e.g. `"asset.ingested"`.
    #[must_use]
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::AssetIngested { .. } => "asset.ingested",
            Self::AssetDeleted { .. } => "asset.deleted",
            Self::AssetUpdated { .. } => "asset.updated",
            Self::AssetTagged { .. } => "asset.tagged",
            Self::CollectionCreated { .. } => "collection.created",
            Self::CollectionDeleted { .. } => "collection.deleted",
            Self::WorkflowStarted { .. } => "workflow.started",
            Self::WorkflowCompleted { .. } => "workflow.completed",
            Self::TranscodeStarted { .. } => "transcode.started",
            Self::TranscodeCompleted { .. } => "transcode.completed",
            Self::TranscodeFailed { .. } => "transcode.failed",
            Self::UserAction { .. } => "user.action",
            Self::StorageWarning { .. } => "storage.warning",
        }
    }

    /// Extract the `asset_id` if this event carries one, otherwise `None`.
    #[must_use]
    pub fn asset_id(&self) -> Option<&str> {
        match self {
            Self::AssetIngested { asset_id, .. }
            | Self::AssetDeleted { asset_id }
            | Self::AssetUpdated { asset_id, .. }
            | Self::AssetTagged { asset_id, .. }
            | Self::WorkflowStarted { asset_id, .. }
            | Self::WorkflowCompleted { asset_id, .. }
            | Self::TranscodeStarted { asset_id, .. }
            | Self::TranscodeCompleted { asset_id, .. }
            | Self::TranscodeFailed { asset_id, .. } => Some(asset_id.as_str()),
            Self::CollectionCreated { .. }
            | Self::CollectionDeleted { .. }
            | Self::UserAction { .. }
            | Self::StorageWarning { .. } => None,
        }
    }
}

/// A callable that receives `MamEvent` references.
pub type EventHandler = Box<dyn Fn(&MamEvent) + Send + Sync>;

/// Metadata about a single event bus subscription.
pub struct Subscription {
    /// Unique subscription identifier returned from [`EventBus::subscribe`].
    pub id: String,
    /// Glob-style pattern: `"*"`, `"asset.*"`, or an exact type like `"workflow.completed"`.
    pub event_pattern: String,
}

/// Thread-safe in-process publish/subscribe event bus.
///
/// # Pattern matching
///
/// Patterns are simple glob strings where `*` matches any single path segment:
/// - `"*"` matches every event.
/// - `"asset.*"` matches all events whose type starts with `"asset."`.
/// - `"workflow.completed"` matches only that exact type.
///
/// Multi-segment wildcards (`**`) are **not** supported; each `*` matches
/// everything up to the next `.` or the end of the string.
pub struct EventBus {
    subscriptions: Arc<Mutex<HashMap<String, (Subscription, EventHandler)>>>,
    event_history: Arc<Mutex<Vec<(SystemTime, MamEvent)>>>,
    max_history: usize,
}

impl EventBus {
    /// Create a new `EventBus` that retains at most `max_history` events.
    #[must_use]
    pub fn new(max_history: usize) -> Self {
        Self {
            subscriptions: Arc::new(Mutex::new(HashMap::new())),
            event_history: Arc::new(Mutex::new(Vec::new())),
            max_history,
        }
    }

    /// Register a handler for events whose type matches `pattern`.
    ///
    /// Returns a subscription ID that can later be passed to `unsubscribe`.
    pub fn subscribe(&self, pattern: &str, handler: EventHandler) -> String {
        let id = Uuid::new_v4().to_string();
        let sub = Subscription {
            id: id.clone(),
            event_pattern: pattern.to_string(),
        };
        self.subscriptions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.clone(), (sub, handler));
        id
    }

    /// Remove a subscription by its ID.
    ///
    /// Returns `true` if the subscription existed and was removed, `false` otherwise.
    pub fn unsubscribe(&self, sub_id: &str) -> bool {
        self.subscriptions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .remove(sub_id)
            .is_some()
    }

    /// Publish an event, fanning it out to all matching subscriptions and
    /// appending it to the history ring-buffer.
    pub fn publish(&self, event: MamEvent) {
        let event_type = event.event_type();

        // Collect matching handlers — clone Arc refs so we can call them
        // without holding the lock.
        let handlers: Vec<Arc<dyn Fn(&MamEvent) + Send + Sync>> = {
            let guard = self.subscriptions.lock().unwrap_or_else(|p| p.into_inner());
            guard
                .values()
                .filter(|(sub, _)| Self::matches_pattern(&sub.event_pattern, event_type))
                .map(|(_, handler)| {
                    // Wrap handler reference as a shared closure via pointer coercion.
                    // We cannot clone Box<dyn Fn>, so we collect into a local Vec<&_>
                    // — but that would borrow the guard.  Instead, we re-dispatch
                    // below while still holding the lock to keep things simple.
                    let _h = handler;
                    // Placeholder; we'll call inline below.
                    Arc::new(|_: &MamEvent| {}) as Arc<dyn Fn(&MamEvent) + Send + Sync>
                })
                .collect()
        };
        // Drop the placeholder vec — actual dispatch is done inline while lock is held
        drop(handlers);

        // Call handlers while holding the lock (handlers are Send+Sync, and we
        // don't re-enter the bus from within a handler in production usage).
        {
            let guard = self.subscriptions.lock().unwrap_or_else(|p| p.into_inner());
            for (sub, handler) in guard.values() {
                if Self::matches_pattern(&sub.event_pattern, event_type) {
                    handler(&event);
                }
            }
        }

        // Append to history, respecting max_history.
        {
            let mut hist = self.event_history.lock().unwrap_or_else(|p| p.into_inner());
            hist.push((SystemTime::now(), event));
            if hist.len() > self.max_history {
                let overflow = hist.len() - self.max_history;
                hist.drain(0..overflow);
            }
        }
    }

    /// Return a snapshot of the recent event history (oldest first).
    #[must_use]
    pub fn history(&self) -> Vec<(SystemTime, MamEvent)> {
        self.event_history
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Return the number of active subscriptions.
    #[must_use]
    pub fn subscription_count(&self) -> usize {
        self.subscriptions
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Glob-style pattern matching.
    ///
    /// - `"*"` matches any event type.
    /// - `"prefix.*"` matches any event type that starts with `"prefix."`.
    /// - Any other pattern is treated as an exact match.
    fn matches_pattern(pattern: &str, event_type: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        // Support "segment.*" style: strip trailing ".*" and check prefix.
        if let Some(prefix) = pattern.strip_suffix(".*") {
            let expected_prefix = format!("{prefix}.");
            return event_type.starts_with(&expected_prefix);
        }
        // Exact match.
        pattern == event_type
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn make_ingested(id: &str) -> MamEvent {
        MamEvent::AssetIngested {
            asset_id: id.to_string(),
            path: "/media/test.mp4".to_string(),
            size_bytes: 1024,
        }
    }

    fn make_deleted(id: &str) -> MamEvent {
        MamEvent::AssetDeleted {
            asset_id: id.to_string(),
        }
    }

    fn make_workflow_completed(wid: &str, aid: &str) -> MamEvent {
        MamEvent::WorkflowCompleted {
            workflow_id: wid.to_string(),
            asset_id: aid.to_string(),
            success: true,
        }
    }

    // --- event_type / asset_id helpers ---

    #[test]
    fn test_event_type_labels() {
        assert_eq!(make_ingested("a").event_type(), "asset.ingested");
        assert_eq!(make_deleted("a").event_type(), "asset.deleted");
        assert_eq!(
            make_workflow_completed("w", "a").event_type(),
            "workflow.completed"
        );
        assert_eq!(
            MamEvent::StorageWarning {
                used_bytes: 1,
                capacity_bytes: 2,
                threshold_pct: 0.5
            }
            .event_type(),
            "storage.warning"
        );
    }

    #[test]
    fn test_asset_id_extraction() {
        assert_eq!(make_ingested("x42").asset_id(), Some("x42"));
        assert_eq!(make_deleted("x43").asset_id(), Some("x43"));
        assert_eq!(
            MamEvent::CollectionCreated {
                collection_id: "c1".into(),
                name: "My Coll".into()
            }
            .asset_id(),
            None
        );
        assert_eq!(
            MamEvent::StorageWarning {
                used_bytes: 0,
                capacity_bytes: 1,
                threshold_pct: 0.0
            }
            .asset_id(),
            None
        );
    }

    // --- pattern matching ---

    #[test]
    fn test_pattern_wildcard_matches_all() {
        assert!(EventBus::matches_pattern("*", "asset.ingested"));
        assert!(EventBus::matches_pattern("*", "workflow.completed"));
        assert!(EventBus::matches_pattern("*", "storage.warning"));
    }

    #[test]
    fn test_pattern_prefix_wildcard() {
        assert!(EventBus::matches_pattern("asset.*", "asset.ingested"));
        assert!(EventBus::matches_pattern("asset.*", "asset.deleted"));
        assert!(!EventBus::matches_pattern("asset.*", "workflow.completed"));
        assert!(!EventBus::matches_pattern("asset.*", "storage.warning"));
    }

    #[test]
    fn test_pattern_exact_match() {
        assert!(EventBus::matches_pattern(
            "workflow.completed",
            "workflow.completed"
        ));
        assert!(!EventBus::matches_pattern(
            "workflow.completed",
            "workflow.started"
        ));
        assert!(!EventBus::matches_pattern(
            "workflow.completed",
            "asset.ingested"
        ));
    }

    // --- subscribe / publish / unsubscribe ---

    #[test]
    fn test_subscribe_and_publish_exact() {
        let bus = EventBus::new(100);
        let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let rx = Arc::clone(&received);

        bus.subscribe(
            "asset.ingested",
            Box::new(move |event| {
                rx.lock()
                    .expect("lock")
                    .push(event.event_type().to_string());
            }),
        );

        bus.publish(make_ingested("a1"));
        bus.publish(make_deleted("a2")); // should NOT match

        let guard = received.lock().expect("lock");
        assert_eq!(guard.len(), 1);
        assert_eq!(guard[0], "asset.ingested");
    }

    #[test]
    fn test_subscribe_wildcard_prefix() {
        let bus = EventBus::new(100);
        let count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let c = Arc::clone(&count);

        bus.subscribe(
            "asset.*",
            Box::new(move |_event| {
                *c.lock().expect("lock") += 1;
            }),
        );

        bus.publish(make_ingested("x"));
        bus.publish(make_deleted("y"));
        bus.publish(make_workflow_completed("w1", "z")); // not asset.*

        assert_eq!(*count.lock().expect("lock"), 2);
    }

    #[test]
    fn test_subscribe_global_wildcard() {
        let bus = EventBus::new(100);
        let count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let c = Arc::clone(&count);

        bus.subscribe(
            "*",
            Box::new(move |_| {
                *c.lock().expect("lock") += 1;
            }),
        );

        bus.publish(make_ingested("a"));
        bus.publish(make_deleted("b"));
        bus.publish(make_workflow_completed("w", "c"));

        assert_eq!(*count.lock().expect("lock"), 3);
    }

    #[test]
    fn test_unsubscribe_stops_delivery() {
        let bus = EventBus::new(100);
        let count: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let c = Arc::clone(&count);

        let sub_id = bus.subscribe(
            "*",
            Box::new(move |_| {
                *c.lock().expect("lock") += 1;
            }),
        );

        bus.publish(make_ingested("a"));
        assert!(bus.unsubscribe(&sub_id));
        bus.publish(make_ingested("b"));

        assert_eq!(*count.lock().expect("lock"), 1);
    }

    #[test]
    fn test_unsubscribe_nonexistent_returns_false() {
        let bus = EventBus::new(100);
        assert!(!bus.unsubscribe("not-a-real-id"));
    }

    #[test]
    fn test_subscription_count() {
        let bus = EventBus::new(100);
        assert_eq!(bus.subscription_count(), 0);

        let id1 = bus.subscribe("*", Box::new(|_| {}));
        let id2 = bus.subscribe("asset.*", Box::new(|_| {}));
        assert_eq!(bus.subscription_count(), 2);

        bus.unsubscribe(&id1);
        assert_eq!(bus.subscription_count(), 1);

        bus.unsubscribe(&id2);
        assert_eq!(bus.subscription_count(), 0);
    }

    // --- history ---

    #[test]
    fn test_history_records_events() {
        let bus = EventBus::new(10);
        bus.publish(make_ingested("a"));
        bus.publish(make_deleted("b"));

        let hist = bus.history();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].1.event_type(), "asset.ingested");
        assert_eq!(hist[1].1.event_type(), "asset.deleted");
    }

    #[test]
    fn test_history_respects_max_history() {
        let bus = EventBus::new(3);
        for i in 0..5u32 {
            bus.publish(MamEvent::AssetDeleted {
                asset_id: i.to_string(),
            });
        }
        let hist = bus.history();
        // Only the last 3 events should remain.
        assert_eq!(hist.len(), 3);
        // The oldest retained event should be "2" (0 and 1 were evicted).
        if let MamEvent::AssetDeleted { asset_id } = &hist[0].1 {
            assert_eq!(asset_id, "2");
        } else {
            panic!("unexpected event type");
        }
    }

    #[test]
    fn test_multiple_subscribers_same_pattern() {
        let bus = EventBus::new(100);
        let count_a: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let count_b: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        let ca = Arc::clone(&count_a);
        let cb = Arc::clone(&count_b);

        bus.subscribe("asset.*", Box::new(move |_| *ca.lock().expect("l") += 1));
        bus.subscribe("asset.*", Box::new(move |_| *cb.lock().expect("l") += 1));

        bus.publish(make_ingested("z"));

        assert_eq!(*count_a.lock().expect("l"), 1);
        assert_eq!(*count_b.lock().expect("l"), 1);
    }

    #[test]
    fn test_no_history_on_zero_max() {
        let bus = EventBus::new(0);
        bus.publish(make_ingested("x"));
        // history is empty because max_history == 0
        assert!(bus.history().is_empty());
    }

    #[test]
    fn test_event_serialization_roundtrip() {
        let event = MamEvent::TranscodeCompleted {
            job_id: "j1".into(),
            asset_id: "a1".into(),
            output_path: "/out/video.av1".into(),
        };
        let json = serde_json::to_string(&event).expect("serialize");
        let back: MamEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.event_type(), "transcode.completed");
        assert_eq!(back.asset_id(), Some("a1"));
    }

    #[test]
    fn test_storage_warning_event() {
        let bus = EventBus::new(10);
        let fired: Arc<Mutex<bool>> = Arc::new(Mutex::new(false));
        let f = Arc::clone(&fired);

        bus.subscribe(
            "storage.warning",
            Box::new(move |e| {
                if let MamEvent::StorageWarning { threshold_pct, .. } = e {
                    if *threshold_pct > 0.8 {
                        *f.lock().expect("l") = true;
                    }
                }
            }),
        );

        bus.publish(MamEvent::StorageWarning {
            used_bytes: 900,
            capacity_bytes: 1000,
            threshold_pct: 0.9,
        });

        assert!(*fired.lock().expect("l"));
    }

    #[test]
    fn test_user_action_event_no_asset_id() {
        let event = MamEvent::UserAction {
            user_id: "u1".into(),
            action: "download".into(),
            resource: "asset/abc".into(),
        };
        assert_eq!(event.event_type(), "user.action");
        assert_eq!(event.asset_id(), None);

        let bus = EventBus::new(10);
        bus.publish(event);
        assert_eq!(bus.history().len(), 1);
    }
}
