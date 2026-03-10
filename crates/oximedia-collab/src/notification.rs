//! Collaboration notification system.
//!
//! Provides typed notification kinds, per-notification metadata, and an inbox
//! that supports delivery, bulk-read, and per-recipient filtering.

#![allow(dead_code)]

/// The category of a collaboration notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotificationKind {
    /// A collaborator mentioned this user.
    Mention,
    /// A collaborator replied to a comment by this user.
    Reply,
    /// The status of a review item changed.
    StatusChange,
    /// Work was assigned to this user.
    Assignment,
    /// A deadline is approaching or was missed.
    Deadline,
}

impl NotificationKind {
    /// Return `true` for high-priority notifications that require immediate attention.
    #[must_use]
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::Mention | Self::Assignment | Self::Deadline)
    }
}

/// A single collaboration notification.
#[derive(Debug, Clone)]
pub struct CollabNotification {
    /// Unique identifier within the inbox.
    pub id: u64,
    /// User id of the intended recipient.
    pub recipient_id: String,
    /// User id of the sender.
    pub sender_id: String,
    /// Category of the notification.
    pub kind: NotificationKind,
    /// Human-readable notification message.
    pub message: String,
    /// Identifier of the resource this notification relates to.
    pub resource_id: String,
    /// Whether the recipient has read this notification.
    pub read: bool,
    /// Creation time in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
}

impl CollabNotification {
    /// Mark this notification as read.
    pub fn mark_read(&mut self) {
        self.read = true;
    }

    /// Return the age of this notification relative to `now` (milliseconds).
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

/// An inbox that stores and manages `CollabNotification`s.
#[derive(Debug, Default)]
pub struct NotificationInbox {
    /// All stored notifications in delivery order.
    pub notifications: Vec<CollabNotification>,
    /// Counter used to assign unique ids.
    pub next_id: u64,
}

impl NotificationInbox {
    /// Create an empty `NotificationInbox`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Deliver a new notification and return its assigned id.
    #[allow(clippy::too_many_arguments)]
    pub fn deliver(
        &mut self,
        recipient: impl Into<String>,
        sender: impl Into<String>,
        kind: NotificationKind,
        message: impl Into<String>,
        resource: impl Into<String>,
        now_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.notifications.push(CollabNotification {
            id,
            recipient_id: recipient.into(),
            sender_id: sender.into(),
            kind,
            message: message.into(),
            resource_id: resource.into(),
            read: false,
            timestamp_ms: now_ms,
        });
        id
    }

    /// Mark all notifications addressed to `recipient_id` as read.
    pub fn mark_all_read(&mut self, recipient_id: &str) {
        for n in &mut self.notifications {
            if n.recipient_id == recipient_id {
                n.mark_read();
            }
        }
    }

    /// Return the number of unread notifications for `recipient_id`.
    #[must_use]
    pub fn unread_count(&self, recipient_id: &str) -> usize {
        self.notifications
            .iter()
            .filter(|n| n.recipient_id == recipient_id && !n.read)
            .count()
    }

    /// Return all notifications addressed to `id` in delivery order.
    #[must_use]
    pub fn for_recipient(&self, id: &str) -> Vec<&CollabNotification> {
        self.notifications
            .iter()
            .filter(|n| n.recipient_id == id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deliver_one(inbox: &mut NotificationInbox, recipient: &str) -> u64 {
        inbox.deliver(
            recipient,
            "system",
            NotificationKind::Mention,
            "You were mentioned",
            "res-1",
            1_000,
        )
    }

    // ---- NotificationKind ----

    #[test]
    fn test_mention_is_urgent() {
        assert!(NotificationKind::Mention.is_urgent());
    }

    #[test]
    fn test_assignment_is_urgent() {
        assert!(NotificationKind::Assignment.is_urgent());
    }

    #[test]
    fn test_deadline_is_urgent() {
        assert!(NotificationKind::Deadline.is_urgent());
    }

    #[test]
    fn test_reply_not_urgent() {
        assert!(!NotificationKind::Reply.is_urgent());
    }

    #[test]
    fn test_status_change_not_urgent() {
        assert!(!NotificationKind::StatusChange.is_urgent());
    }

    // ---- CollabNotification ----

    #[test]
    fn test_mark_read_sets_flag() {
        let mut inbox = NotificationInbox::new();
        let id = deliver_one(&mut inbox, "alice");
        let n = inbox
            .notifications
            .iter_mut()
            .find(|n| n.id == id)
            .expect("collab test operation should succeed");
        assert!(!n.read);
        n.mark_read();
        assert!(n.read);
    }

    #[test]
    fn test_age_ms_positive() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        let n = &inbox.notifications[0];
        assert_eq!(n.age_ms(3_000), 2_000);
    }

    #[test]
    fn test_age_ms_before_creation_is_zero() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        let n = &inbox.notifications[0];
        // now < timestamp_ms → saturating_sub → 0
        assert_eq!(n.age_ms(500), 0);
    }

    // ---- NotificationInbox ----

    #[test]
    fn test_deliver_returns_incrementing_ids() {
        let mut inbox = NotificationInbox::new();
        let id1 = deliver_one(&mut inbox, "alice");
        let id2 = deliver_one(&mut inbox, "bob");
        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_new_inbox_is_empty() {
        let inbox = NotificationInbox::new();
        assert!(inbox.notifications.is_empty());
    }

    #[test]
    fn test_unread_count_increments_on_deliver() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "alice");
        assert_eq!(inbox.unread_count("alice"), 2);
    }

    #[test]
    fn test_unread_count_zero_for_other_user() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        assert_eq!(inbox.unread_count("bob"), 0);
    }

    #[test]
    fn test_mark_all_read_clears_unread() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "alice");
        inbox.mark_all_read("alice");
        assert_eq!(inbox.unread_count("alice"), 0);
    }

    #[test]
    fn test_mark_all_read_only_affects_recipient() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "bob");
        inbox.mark_all_read("alice");
        assert_eq!(inbox.unread_count("alice"), 0);
        assert_eq!(inbox.unread_count("bob"), 1); // bob untouched
    }

    #[test]
    fn test_for_recipient_filters_correctly() {
        let mut inbox = NotificationInbox::new();
        deliver_one(&mut inbox, "alice");
        deliver_one(&mut inbox, "bob");
        deliver_one(&mut inbox, "alice");
        let alice_notifs = inbox.for_recipient("alice");
        assert_eq!(alice_notifs.len(), 2);
        assert!(alice_notifs.iter().all(|n| n.recipient_id == "alice"));
    }

    #[test]
    fn test_for_recipient_empty_for_unknown() {
        let inbox = NotificationInbox::new();
        assert!(inbox.for_recipient("ghost").is_empty());
    }

    #[test]
    fn test_notification_fields_stored() {
        let mut inbox = NotificationInbox::new();
        inbox.deliver(
            "alice",
            "bob",
            NotificationKind::Reply,
            "Great work!",
            "res-42",
            9_999,
        );
        let n = &inbox.notifications[0];
        assert_eq!(n.recipient_id, "alice");
        assert_eq!(n.sender_id, "bob");
        assert_eq!(n.kind, NotificationKind::Reply);
        assert_eq!(n.message, "Great work!");
        assert_eq!(n.resource_id, "res-42");
        assert_eq!(n.timestamp_ms, 9_999);
        assert!(!n.read);
    }
}
