//! Clearance workflow escalation notifications.
//!
//! When a [`crate::clearance_workflow::ClearanceRequest`] remains in `Pending`
//! state for longer than a configurable threshold, this module raises escalation
//! notifications through progressively higher levels of urgency.
//!
//! # Overview
//! - [`EscalationConfig`] – defines the pending-age threshold and the ordered
//!   list of escalation levels (e.g. `["reminder", "manager", "legal"]`).
//! - [`ClearanceNotification`] – a single escalation event, recording which
//!   clearance was escalated, to which level, and when.
//! - [`ClearanceNotifier`] – stateless helper that computes whether an
//!   escalation is due for a clearance given its submission time and the current
//!   time.
//! - [`NotificationLog`] – accumulates all notifications that have been sent,
//!   with helpers for querying per-clearance history.

#![allow(dead_code)]

// ── EscalationConfig ────────────────────────────────────────────────────────

/// Configuration for the clearance escalation policy.
///
/// The `pending_threshold_ms` is the minimum age (in milliseconds since
/// submission) before any escalation is triggered. Subsequent escalations
/// advance through `escalation_levels` one at a time each time the threshold
/// elapses again.
///
/// # Example
/// ```
/// use oximedia_rights::clearance_notifications::EscalationConfig;
///
/// let cfg = EscalationConfig::new(
///     // 24 hours in ms
///     24 * 3600 * 1000,
///     vec!["reminder".into(), "manager".into(), "legal".into()],
/// );
/// assert_eq!(cfg.escalation_levels.len(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Minimum pending age in milliseconds before the first escalation fires.
    pub pending_threshold_ms: u64,
    /// Ordered list of escalation level names.
    /// Level index 0 = first escalation, 1 = second, etc.
    pub escalation_levels: Vec<String>,
}

impl EscalationConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(pending_threshold_ms: u64, escalation_levels: Vec<String>) -> Self {
        Self {
            pending_threshold_ms,
            escalation_levels,
        }
    }

    /// Return the level name for a given 0-based index, or `None` if the index
    /// is out of range.
    #[must_use]
    pub fn level_name(&self, index: usize) -> Option<&str> {
        self.escalation_levels.get(index).map(String::as_str)
    }

    /// Total number of escalation levels defined.
    #[must_use]
    pub fn level_count(&self) -> usize {
        self.escalation_levels.len()
    }
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self::new(
            24 * 3_600_000, // 24 h in ms
            vec![
                "reminder".to_string(),
                "manager_alert".to_string(),
                "legal_escalation".to_string(),
            ],
        )
    }
}

// ── ClearanceNotification ───────────────────────────────────────────────────

/// A single escalation notification for a clearance request.
#[derive(Debug, Clone)]
pub struct ClearanceNotification {
    /// Identifier of the [`crate::clearance_workflow::ClearanceRequest`].
    pub clearance_id: String,
    /// 0-based escalation level that was triggered.
    pub level: usize,
    /// Human-readable escalation message.
    pub message: String,
    /// Unix timestamp in milliseconds when the notification was generated.
    pub sent_at_ms: u64,
}

impl ClearanceNotification {
    /// Create a new notification.
    #[must_use]
    pub fn new(
        clearance_id: impl Into<String>,
        level: usize,
        message: impl Into<String>,
        sent_at_ms: u64,
    ) -> Self {
        Self {
            clearance_id: clearance_id.into(),
            level,
            message: message.into(),
            sent_at_ms,
        }
    }
}

// ── ClearanceNotifier ───────────────────────────────────────────────────────

/// Stateless helper for computing clearance escalations.
///
/// Given a clearance ID, the time it was submitted, the current time,
/// and the escalation config, `check_escalation` decides whether a new
/// notification should be raised — and if so, which level.
///
/// The caller is responsible for persisting emitted notifications (e.g. via
/// [`NotificationLog`]) and for ensuring each level is only sent once per
/// clearance.
pub struct ClearanceNotifier;

impl ClearanceNotifier {
    /// Check whether an escalation notification should be raised for a
    /// clearance that was submitted at `submitted_at_ms` and is still pending
    /// at `now_ms`.
    ///
    /// The level is determined by how many multiples of
    /// `config.pending_threshold_ms` have elapsed since submission, capped at
    /// the number of defined levels.
    ///
    /// Returns `Some(notification)` when an escalation is due, or `None` when
    /// the clearance is not yet old enough or all levels have been exhausted.
    ///
    /// # Parameters
    /// - `clearance_id` – ID of the clearance request (used in the notification).
    /// - `submitted_at_ms` – Unix timestamp in milliseconds when the request was submitted.
    /// - `now_ms` – Current Unix timestamp in milliseconds.
    /// - `config` – Escalation configuration.
    #[must_use]
    pub fn check_escalation(
        clearance_id: &str,
        submitted_at_ms: u64,
        now_ms: u64,
        config: &EscalationConfig,
    ) -> Option<ClearanceNotification> {
        if config.pending_threshold_ms == 0 || config.escalation_levels.is_empty() {
            return None;
        }
        let elapsed_ms = now_ms.saturating_sub(submitted_at_ms);
        if elapsed_ms < config.pending_threshold_ms {
            return None;
        }
        // Compute which level index would apply based on elapsed time.
        let raw_level = (elapsed_ms / config.pending_threshold_ms) as usize;
        // Cap to the last defined level (no level beyond the list).
        let level = raw_level.min(config.escalation_levels.len()).saturating_sub(1);
        let level_name = config.level_name(level)?;
        let message = format!(
            "Clearance '{}' has been pending for {} ms — escalating to level {} ({})",
            clearance_id, elapsed_ms, level, level_name
        );
        Some(ClearanceNotification::new(
            clearance_id,
            level,
            message,
            now_ms,
        ))
    }
}

// ── NotificationLog ─────────────────────────────────────────────────────────

/// Persistent log of all escalation notifications that have been raised.
///
/// # Example
/// ```
/// use oximedia_rights::clearance_notifications::{
///     ClearanceNotification, NotificationLog,
/// };
///
/// let mut log = NotificationLog::new();
/// log.record(ClearanceNotification::new("clr-1", 0, "reminder sent", 5_000));
/// assert_eq!(log.total_count(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct NotificationLog {
    entries: Vec<ClearanceNotification>,
}

impl NotificationLog {
    /// Create an empty log.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a notification.
    pub fn record(&mut self, notification: ClearanceNotification) {
        self.entries.push(notification);
    }

    /// Total number of recorded notifications.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// All notifications for a specific clearance ID.
    #[must_use]
    pub fn for_clearance(&self, clearance_id: &str) -> Vec<&ClearanceNotification> {
        self.entries
            .iter()
            .filter(|n| n.clearance_id == clearance_id)
            .collect()
    }

    /// The highest escalation level that has already been recorded for a
    /// clearance, or `None` if no notifications exist for it.
    #[must_use]
    pub fn highest_level_sent(&self, clearance_id: &str) -> Option<usize> {
        self.entries
            .iter()
            .filter(|n| n.clearance_id == clearance_id)
            .map(|n| n.level)
            .max()
    }

    /// Whether a particular level has already been recorded for a clearance.
    #[must_use]
    pub fn already_sent(&self, clearance_id: &str, level: usize) -> bool {
        self.entries
            .iter()
            .any(|n| n.clearance_id == clearance_id && n.level == level)
    }

    /// Return all notifications, ordered by `sent_at_ms` ascending.
    #[must_use]
    pub fn sorted_by_time(&self) -> Vec<&ClearanceNotification> {
        let mut sorted: Vec<&ClearanceNotification> = self.entries.iter().collect();
        sorted.sort_by_key(|n| n.sent_at_ms);
        sorted
    }

    /// Clear all recorded notifications.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> EscalationConfig {
        EscalationConfig::new(
            1_000, // 1 second in ms
            vec![
                "reminder".to_string(),
                "manager_alert".to_string(),
                "legal_escalation".to_string(),
            ],
        )
    }

    // ── EscalationConfig ──

    #[test]
    fn test_config_level_name_in_range() {
        let cfg = default_config();
        assert_eq!(cfg.level_name(0), Some("reminder"));
        assert_eq!(cfg.level_name(2), Some("legal_escalation"));
    }

    #[test]
    fn test_config_level_name_out_of_range() {
        let cfg = default_config();
        assert!(cfg.level_name(99).is_none());
    }

    #[test]
    fn test_config_level_count() {
        assert_eq!(default_config().level_count(), 3);
    }

    // ── ClearanceNotifier ──

    #[test]
    fn test_notifier_no_escalation_before_threshold() {
        let cfg = default_config();
        // submitted at 0, now at 500 ms (threshold is 1000 ms)
        let result = ClearanceNotifier::check_escalation("clr-1", 0, 500, &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_notifier_first_escalation_at_threshold() {
        let cfg = default_config();
        // exactly at threshold
        let result = ClearanceNotifier::check_escalation("clr-1", 0, 1_000, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        assert_eq!(notif.clearance_id, "clr-1");
        assert_eq!(notif.level, 0); // first level
    }

    #[test]
    fn test_notifier_second_escalation() {
        let cfg = default_config();
        // 2× threshold elapsed → level 1
        let result = ClearanceNotifier::check_escalation("clr-2", 0, 2_000, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        assert_eq!(notif.level, 1);
    }

    #[test]
    fn test_notifier_capped_at_last_level() {
        let cfg = default_config();
        // 100× threshold elapsed but only 3 levels → stays at level 2
        let result = ClearanceNotifier::check_escalation("clr-3", 0, 100_000, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        assert_eq!(notif.level, 2);
    }

    #[test]
    fn test_notifier_empty_levels_returns_none() {
        let cfg = EscalationConfig::new(1_000, vec![]);
        let result = ClearanceNotifier::check_escalation("clr-4", 0, 5_000, &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_notifier_notification_contains_id() {
        let cfg = default_config();
        let result = ClearanceNotifier::check_escalation("my-clearance", 0, 1_500, &cfg);
        let notif = result.expect("expected notification");
        assert!(notif.message.contains("my-clearance"));
        assert_eq!(notif.sent_at_ms, 1_500);
    }

    // ── NotificationLog ──

    #[test]
    fn test_log_record_and_count() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 0, "reminder", 1_000));
        log.record(ClearanceNotification::new("clr-1", 1, "escalated", 2_000));
        assert_eq!(log.total_count(), 2);
    }

    #[test]
    fn test_log_for_clearance_filter() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 0, "A", 1_000));
        log.record(ClearanceNotification::new("clr-2", 0, "B", 2_000));
        let clr1 = log.for_clearance("clr-1");
        assert_eq!(clr1.len(), 1);
        assert_eq!(clr1[0].message, "A");
    }

    #[test]
    fn test_log_highest_level_sent() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 0, "level 0", 1_000));
        log.record(ClearanceNotification::new("clr-1", 2, "level 2", 3_000));
        assert_eq!(log.highest_level_sent("clr-1"), Some(2));
        assert!(log.highest_level_sent("unknown").is_none());
    }

    #[test]
    fn test_log_already_sent() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 0, "reminder", 1_000));
        assert!(log.already_sent("clr-1", 0));
        assert!(!log.already_sent("clr-1", 1));
    }

    #[test]
    fn test_log_sorted_by_time() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 1, "second", 2_000));
        log.record(ClearanceNotification::new("clr-2", 0, "first", 1_000));
        let sorted = log.sorted_by_time();
        assert_eq!(sorted[0].sent_at_ms, 1_000);
        assert_eq!(sorted[1].sent_at_ms, 2_000);
    }

    #[test]
    fn test_log_clear() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("clr-1", 0, "x", 1_000));
        log.clear();
        assert_eq!(log.total_count(), 0);
    }

    // ── Additional tests for clearance workflow escalation (TODO item) ──

    #[test]
    fn test_config_default_has_three_levels() {
        let cfg = EscalationConfig::default();
        assert_eq!(cfg.level_count(), 3);
        assert_eq!(cfg.level_name(0), Some("reminder"));
        assert_eq!(cfg.level_name(1), Some("manager_alert"));
        assert_eq!(cfg.level_name(2), Some("legal_escalation"));
    }

    #[test]
    fn test_config_default_threshold_is_24h() {
        let cfg = EscalationConfig::default();
        // Default threshold should be 24 hours in milliseconds
        assert_eq!(cfg.pending_threshold_ms, 24 * 3_600_000);
    }

    #[test]
    fn test_notifier_zero_threshold_returns_none() {
        let cfg = EscalationConfig::new(0, vec!["reminder".to_string()]);
        // Zero threshold is a degenerate config: should not escalate
        let result = ClearanceNotifier::check_escalation("clr-x", 0, 9_999_999, &cfg);
        assert!(result.is_none());
    }

    #[test]
    fn test_notifier_single_level_config_at_threshold() {
        let cfg = EscalationConfig::new(500, vec!["only_level".to_string()]);
        let result = ClearanceNotifier::check_escalation("clr-single", 0, 500, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        assert_eq!(notif.level, 0);
    }

    #[test]
    fn test_notifier_single_level_capped_after_multiple_thresholds() {
        // With only one level, any elapsed time ≥ threshold should still give level 0
        let cfg = EscalationConfig::new(100, vec!["single".to_string()]);
        let result = ClearanceNotifier::check_escalation("clr-s", 0, 9_900, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        assert_eq!(notif.level, 0, "should be capped at last (only) level");
    }

    #[test]
    fn test_notifier_level_message_mentions_elapsed() {
        let cfg = EscalationConfig::new(1_000, vec!["reminder".to_string()]);
        let result = ClearanceNotifier::check_escalation("clr-elapsed", 5_000, 8_000, &cfg);
        assert!(result.is_some());
        let notif = result.expect("expected notification");
        // Message should mention the elapsed time (3000 ms)
        assert!(
            notif.message.contains("3000"),
            "message should contain elapsed time: {}",
            notif.message
        );
    }

    #[test]
    fn test_log_multiple_clearances_independent() {
        let mut log = NotificationLog::new();
        for i in 0..5u64 {
            log.record(ClearanceNotification::new(
                &format!("clr-{i}"),
                0,
                "reminder",
                i * 1_000,
            ));
        }
        // Each clearance should have exactly 1 notification
        for i in 0..5u64 {
            assert_eq!(log.for_clearance(&format!("clr-{i}")).len(), 1);
        }
        assert_eq!(log.total_count(), 5);
    }

    #[test]
    fn test_log_already_sent_false_for_empty_log() {
        let log = NotificationLog::new();
        assert!(!log.already_sent("any-clr", 0));
        assert!(!log.already_sent("any-clr", 99));
    }

    #[test]
    fn test_log_sorted_by_time_single_entry() {
        let mut log = NotificationLog::new();
        log.record(ClearanceNotification::new("c1", 0, "only", 42_000));
        let sorted = log.sorted_by_time();
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].sent_at_ms, 42_000);
    }

    #[test]
    fn test_notifier_large_elapsed_exact_level_cap() {
        // 99 thresholds elapsed but only 3 levels → level should be 2 (last)
        let cfg = EscalationConfig::new(1_000, vec!["a".into(), "b".into(), "c".into()]);
        let result = ClearanceNotifier::check_escalation("x", 0, 99_000, &cfg);
        let notif = result.expect("expected notification");
        assert_eq!(notif.level, 2);
    }
}
