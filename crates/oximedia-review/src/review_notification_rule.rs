#![allow(dead_code)]
//! Configurable notification rules for review events.
//!
//! Allows users to define conditions under which they receive notifications
//! about review activity, with support for channel selection, batching,
//! and quiet hours.

use std::collections::HashSet;

/// Notification delivery channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NotificationChannel {
    /// Email notification.
    Email,
    /// In-app notification.
    InApp,
    /// Webhook (HTTP POST).
    Webhook,
    /// Slack integration.
    Slack,
    /// SMS notification.
    Sms,
    /// Push notification (mobile).
    Push,
}

impl std::fmt::Display for NotificationChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Email => "email",
            Self::InApp => "in_app",
            Self::Webhook => "webhook",
            Self::Slack => "slack",
            Self::Sms => "sms",
            Self::Push => "push",
        };
        write!(f, "{label}")
    }
}

/// The event type that triggers a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NotificationEvent {
    /// A new comment was added.
    CommentAdded,
    /// A comment was resolved.
    CommentResolved,
    /// The approval status changed.
    ApprovalChanged,
    /// A new version was uploaded.
    VersionUploaded,
    /// A task was assigned to the user.
    TaskAssigned,
    /// A task deadline is approaching.
    TaskDeadline,
    /// A reviewer was added.
    ReviewerAdded,
    /// The session status changed.
    StatusChanged,
    /// A mention (@user) occurred.
    UserMentioned,
    /// The session deadline is approaching.
    SessionDeadline,
}

/// Priority level for a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NotificationPriority {
    /// Low priority - can be batched.
    Low,
    /// Normal priority.
    Normal,
    /// High priority - deliver immediately.
    High,
    /// Critical - deliver immediately and escalate.
    Critical,
}

/// Batching strategy for notifications.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStrategy {
    /// Deliver immediately.
    Immediate,
    /// Batch every N minutes.
    IntervalMinutes(u32),
    /// Deliver as a daily digest.
    DailyDigest,
    /// Deliver as a weekly digest.
    WeeklyDigest,
}

/// Quiet hours configuration (hour-of-day based, 0-23).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuietHours {
    /// Start hour (0-23).
    pub start_hour: u8,
    /// End hour (0-23).
    pub end_hour: u8,
    /// Whether quiet hours are enabled.
    pub enabled: bool,
}

impl QuietHours {
    /// Create a new quiet hours range.
    #[must_use]
    pub fn new(start_hour: u8, end_hour: u8) -> Self {
        Self {
            start_hour: start_hour.min(23),
            end_hour: end_hour.min(23),
            enabled: true,
        }
    }

    /// Check if a given hour falls within quiet hours.
    #[must_use]
    pub fn is_quiet(&self, hour: u8) -> bool {
        if !self.enabled {
            return false;
        }
        if self.start_hour <= self.end_hour {
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // Wraps around midnight
            hour >= self.start_hour || hour < self.end_hour
        }
    }

    /// Disable quiet hours.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable quiet hours.
    pub fn enable(&mut self) {
        self.enabled = true;
    }
}

/// A single notification rule.
#[derive(Debug, Clone)]
pub struct NotificationRule {
    /// Human-readable name for this rule.
    pub name: String,
    /// Events that trigger this rule.
    pub events: HashSet<NotificationEvent>,
    /// Channels to deliver on.
    pub channels: HashSet<NotificationChannel>,
    /// Minimum priority for this rule to fire.
    pub min_priority: NotificationPriority,
    /// Batching strategy.
    pub batch_strategy: BatchStrategy,
    /// Whether the rule is active.
    pub enabled: bool,
    /// Optional session IDs to scope this rule to.
    pub session_filter: Option<Vec<String>>,
}

impl NotificationRule {
    /// Create a new rule with given name, event, and channel.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        event: NotificationEvent,
        channel: NotificationChannel,
    ) -> Self {
        let mut events = HashSet::new();
        events.insert(event);
        let mut channels = HashSet::new();
        channels.insert(channel);
        Self {
            name: name.into(),
            events,
            channels,
            min_priority: NotificationPriority::Low,
            batch_strategy: BatchStrategy::Immediate,
            enabled: true,
            session_filter: None,
        }
    }

    /// Add an additional event trigger.
    pub fn add_event(&mut self, event: NotificationEvent) {
        self.events.insert(event);
    }

    /// Add an additional delivery channel.
    pub fn add_channel(&mut self, channel: NotificationChannel) {
        self.channels.insert(channel);
    }

    /// Set the minimum priority threshold.
    pub fn set_min_priority(&mut self, priority: NotificationPriority) {
        self.min_priority = priority;
    }

    /// Set the batching strategy.
    pub fn set_batch_strategy(&mut self, strategy: BatchStrategy) {
        self.batch_strategy = strategy;
    }

    /// Check whether this rule matches a given event and priority.
    #[must_use]
    pub fn matches(&self, event: NotificationEvent, priority: NotificationPriority) -> bool {
        self.enabled && self.events.contains(&event) && priority >= self.min_priority
    }

    /// Check whether this rule applies to a given session.
    #[must_use]
    pub fn applies_to_session(&self, session_id: &str) -> bool {
        match &self.session_filter {
            None => true,
            Some(ids) => ids.iter().any(|id| id == session_id),
        }
    }
}

/// A collection of notification rules for a user.
#[derive(Debug)]
pub struct NotificationRuleSet {
    /// User ID this set belongs to.
    user_id: String,
    /// All configured rules.
    rules: Vec<NotificationRule>,
    /// Quiet hours configuration.
    quiet_hours: Option<QuietHours>,
    /// Global enabled flag.
    enabled: bool,
}

impl NotificationRuleSet {
    /// Create a new rule set for a user.
    #[must_use]
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            rules: Vec::new(),
            quiet_hours: None,
            enabled: true,
        }
    }

    /// Get the user ID.
    #[must_use]
    pub fn user_id(&self) -> &str {
        &self.user_id
    }

    /// Add a rule to the set.
    pub fn add_rule(&mut self, rule: NotificationRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by name.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.name != name);
        self.rules.len() < before
    }

    /// Get all rules.
    #[must_use]
    pub fn rules(&self) -> &[NotificationRule] {
        &self.rules
    }

    /// Get number of rules.
    #[must_use]
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Set quiet hours.
    pub fn set_quiet_hours(&mut self, quiet: QuietHours) {
        self.quiet_hours = Some(quiet);
    }

    /// Check if the user is currently in quiet hours.
    #[must_use]
    pub fn is_quiet_hour(&self, hour: u8) -> bool {
        self.quiet_hours
            .as_ref()
            .map_or(false, |qh| qh.is_quiet(hour))
    }

    /// Find all matching rules for a given event, priority, and session.
    #[must_use]
    pub fn matching_rules(
        &self,
        event: NotificationEvent,
        priority: NotificationPriority,
        session_id: &str,
    ) -> Vec<&NotificationRule> {
        if !self.enabled {
            return Vec::new();
        }
        self.rules
            .iter()
            .filter(|r| r.matches(event, priority) && r.applies_to_session(session_id))
            .collect()
    }

    /// Collect all channels from matching rules (deduplicated).
    #[must_use]
    pub fn channels_for_event(
        &self,
        event: NotificationEvent,
        priority: NotificationPriority,
        session_id: &str,
    ) -> HashSet<NotificationChannel> {
        let matching = self.matching_rules(event, priority, session_id);
        let mut channels = HashSet::new();
        for rule in matching {
            for ch in &rule.channels {
                channels.insert(*ch);
            }
        }
        channels
    }

    /// Disable all notifications globally.
    pub fn disable_all(&mut self) {
        self.enabled = false;
    }

    /// Enable all notifications globally.
    pub fn enable_all(&mut self) {
        self.enabled = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiet_hours_normal_range() {
        let qh = QuietHours::new(22, 7);
        assert!(qh.is_quiet(22));
        assert!(qh.is_quiet(0));
        assert!(qh.is_quiet(3));
        assert!(!qh.is_quiet(7));
        assert!(!qh.is_quiet(12));
    }

    #[test]
    fn test_quiet_hours_same_day_range() {
        let qh = QuietHours::new(9, 17);
        assert!(qh.is_quiet(9));
        assert!(qh.is_quiet(12));
        assert!(!qh.is_quiet(17));
        assert!(!qh.is_quiet(8));
        assert!(!qh.is_quiet(20));
    }

    #[test]
    fn test_quiet_hours_disabled() {
        let mut qh = QuietHours::new(22, 7);
        qh.disable();
        assert!(!qh.is_quiet(23));
        assert!(!qh.is_quiet(3));
    }

    #[test]
    fn test_notification_rule_matches() {
        let rule = NotificationRule::new(
            "comment rule",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        );
        assert!(rule.matches(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal
        ));
        assert!(!rule.matches(
            NotificationEvent::VersionUploaded,
            NotificationPriority::Normal
        ));
    }

    #[test]
    fn test_notification_rule_priority_filter() {
        let mut rule = NotificationRule::new(
            "high only",
            NotificationEvent::ApprovalChanged,
            NotificationChannel::Slack,
        );
        rule.set_min_priority(NotificationPriority::High);
        assert!(!rule.matches(
            NotificationEvent::ApprovalChanged,
            NotificationPriority::Normal
        ));
        assert!(rule.matches(
            NotificationEvent::ApprovalChanged,
            NotificationPriority::High
        ));
        assert!(rule.matches(
            NotificationEvent::ApprovalChanged,
            NotificationPriority::Critical
        ));
    }

    #[test]
    fn test_notification_rule_disabled() {
        let mut rule = NotificationRule::new(
            "disabled",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        );
        rule.enabled = false;
        assert!(!rule.matches(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal
        ));
    }

    #[test]
    fn test_notification_rule_session_filter() {
        let mut rule = NotificationRule::new(
            "scoped",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        );
        rule.session_filter = Some(vec!["session-1".to_string(), "session-2".to_string()]);
        assert!(rule.applies_to_session("session-1"));
        assert!(!rule.applies_to_session("session-3"));
    }

    #[test]
    fn test_rule_set_add_and_remove() {
        let mut set = NotificationRuleSet::new("user-1");
        set.add_rule(NotificationRule::new(
            "r1",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        ));
        set.add_rule(NotificationRule::new(
            "r2",
            NotificationEvent::VersionUploaded,
            NotificationChannel::Slack,
        ));
        assert_eq!(set.rule_count(), 2);
        assert!(set.remove_rule("r1"));
        assert_eq!(set.rule_count(), 1);
        assert!(!set.remove_rule("nonexistent"));
    }

    #[test]
    fn test_rule_set_matching_rules() {
        let mut set = NotificationRuleSet::new("user-1");
        set.add_rule(NotificationRule::new(
            "r1",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        ));
        set.add_rule(NotificationRule::new(
            "r2",
            NotificationEvent::VersionUploaded,
            NotificationChannel::Slack,
        ));
        let matches = set.matching_rules(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal,
            "s1",
        );
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].name, "r1");
    }

    #[test]
    fn test_rule_set_channels_for_event() {
        let mut set = NotificationRuleSet::new("user-1");
        let mut rule = NotificationRule::new(
            "r1",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        );
        rule.add_channel(NotificationChannel::Push);
        set.add_rule(rule);
        set.add_rule(NotificationRule::new(
            "r2",
            NotificationEvent::CommentAdded,
            NotificationChannel::Slack,
        ));
        let channels = set.channels_for_event(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal,
            "s1",
        );
        assert_eq!(channels.len(), 3);
        assert!(channels.contains(&NotificationChannel::Email));
        assert!(channels.contains(&NotificationChannel::Push));
        assert!(channels.contains(&NotificationChannel::Slack));
    }

    #[test]
    fn test_rule_set_disabled_globally() {
        let mut set = NotificationRuleSet::new("user-1");
        set.add_rule(NotificationRule::new(
            "r1",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        ));
        set.disable_all();
        let matches = set.matching_rules(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal,
            "s1",
        );
        assert!(matches.is_empty());
    }

    #[test]
    fn test_rule_set_quiet_hours() {
        let mut set = NotificationRuleSet::new("user-1");
        assert!(!set.is_quiet_hour(23));
        set.set_quiet_hours(QuietHours::new(22, 7));
        assert!(set.is_quiet_hour(23));
        assert!(!set.is_quiet_hour(12));
    }

    #[test]
    fn test_notification_channel_display() {
        assert_eq!(format!("{}", NotificationChannel::Email), "email");
        assert_eq!(format!("{}", NotificationChannel::Slack), "slack");
        assert_eq!(format!("{}", NotificationChannel::Push), "push");
        assert_eq!(format!("{}", NotificationChannel::InApp), "in_app");
    }

    #[test]
    fn test_add_multiple_events_to_rule() {
        let mut rule = NotificationRule::new(
            "multi",
            NotificationEvent::CommentAdded,
            NotificationChannel::Email,
        );
        rule.add_event(NotificationEvent::VersionUploaded);
        rule.add_event(NotificationEvent::ApprovalChanged);
        assert!(rule.matches(
            NotificationEvent::CommentAdded,
            NotificationPriority::Normal
        ));
        assert!(rule.matches(
            NotificationEvent::VersionUploaded,
            NotificationPriority::Normal
        ));
        assert!(rule.matches(
            NotificationEvent::ApprovalChanged,
            NotificationPriority::Normal
        ));
        assert!(!rule.matches(
            NotificationEvent::TaskAssigned,
            NotificationPriority::Normal
        ));
    }
}
