// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Job completion notifications — rule-based event dispatching with history.
//!
//! `NotificationManager` evaluates registered `NotificationRule`s whenever a
//! job transitions to a terminal state (Completed, Failed, Cancelled).  Each
//! rule targets a `NotificationDestination` and filters by one or more
//! `TriggerEvent`s and optional job-tag predicates.
//!
//! Notifications are dispatched via the `NotificationSender` trait, enabling
//! in-process test doubles without spawning real HTTP/email connections.
//!
//! # Example
//! ```rust
//! use oximedia_jobs::job_notification::{
//!     NotificationManager, NotificationRule, NotificationDestination,
//!     TriggerEvent, NoopSender,
//! };
//! use oximedia_jobs::{Job, JobPayload, Priority, JobStatus, TranscodeParams};
//! use std::sync::Arc;
//!
//! let params = TranscodeParams {
//!     input: "in.mp4".into(), output: "out.mp4".into(),
//!     video_codec: "h264".into(), audio_codec: "aac".into(),
//!     video_bitrate: 4_000_000, audio_bitrate: 128_000,
//!     resolution: None, framerate: None,
//!     preset: "fast".into(), hw_accel: None,
//! };
//! let mut job = Job::new("encode".into(), Priority::Normal, JobPayload::Transcode(params));
//! job.status = JobStatus::Completed;
//!
//! let rule = NotificationRule::new(
//!     "on-complete",
//!     vec![TriggerEvent::OnCompletion],
//!     NotificationDestination::Log,
//! );
//! let sender = Arc::new(NoopSender);
//! let mut manager = NotificationManager::new(sender);
//! manager.add_rule(rule);
//! manager.notify(&job);
//! assert_eq!(manager.history().len(), 1);
//! ```

use crate::job::{Job, JobStatus};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// TriggerEvent
// ---------------------------------------------------------------------------

/// The job lifecycle event that triggers a notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TriggerEvent {
    /// Fire when the job completes successfully.
    OnSuccess,
    /// Fire when the job fails (exhausts retries or hard failure).
    OnFailure,
    /// Fire when the job is cancelled.
    OnCancellation,
    /// Fire on any terminal outcome (Success | Failure | Cancellation).
    OnCompletion,
}

impl TriggerEvent {
    /// Whether this event matches the given job status.
    #[must_use]
    pub fn matches(&self, status: JobStatus) -> bool {
        match self {
            Self::OnSuccess => status == JobStatus::Completed,
            Self::OnFailure => status == JobStatus::Failed,
            Self::OnCancellation => status == JobStatus::Cancelled,
            Self::OnCompletion => matches!(
                status,
                JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// NotificationDestination
// ---------------------------------------------------------------------------

/// Where to send the notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationDestination {
    /// Write notification to the tracing log.
    Log,
    /// POST to an HTTP endpoint.
    Webhook { url: String },
    /// Send to an email address.
    Email { address: String },
    /// Forward to an in-process channel identified by name.
    InProcess { channel: String },
}

// ---------------------------------------------------------------------------
// NotificationRule
// ---------------------------------------------------------------------------

/// A rule that specifies when and where to send notifications for a job.
#[derive(Debug, Clone)]
pub struct NotificationRule {
    /// Unique rule identifier.
    pub id: Uuid,
    /// Human-readable rule name.
    pub name: String,
    /// Events that trigger this rule.
    pub trigger_events: Vec<TriggerEvent>,
    /// Destination for the notification.
    pub destination: NotificationDestination,
    /// If non-empty, the rule only fires for jobs carrying at least one of
    /// these tags.
    pub tag_filter: Vec<String>,
    /// Whether the rule is currently active.
    pub enabled: bool,
}

impl NotificationRule {
    /// Create a new enabled rule with no tag filter.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        trigger_events: Vec<TriggerEvent>,
        destination: NotificationDestination,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            trigger_events,
            destination,
            tag_filter: Vec::new(),
            enabled: true,
        }
    }

    /// Restrict this rule to jobs carrying at least one of the given tags.
    #[must_use]
    pub fn with_tag_filter(mut self, tags: Vec<String>) -> Self {
        self.tag_filter = tags;
        self
    }

    /// Check whether this rule fires for the given job.
    #[must_use]
    pub fn fires_for(&self, job: &Job) -> bool {
        if !self.enabled {
            return false;
        }

        let event_matches = self
            .trigger_events
            .iter()
            .any(|e| e.matches(job.status));

        if !event_matches {
            return false;
        }

        if self.tag_filter.is_empty() {
            return true;
        }

        // At least one of the filter tags must be present on the job.
        self.tag_filter
            .iter()
            .any(|t| job.tags.contains(t))
    }
}

// ---------------------------------------------------------------------------
// NotificationPayload
// ---------------------------------------------------------------------------

/// The data sent with each notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPayload {
    /// ID of the triggering job.
    pub job_id: Uuid,
    /// Name of the triggering job.
    pub job_name: String,
    /// Final status of the job.
    pub job_status: String,
    /// Tags on the job at the time of notification.
    pub job_tags: Vec<String>,
    /// Error message, if the job failed.
    pub error: Option<String>,
    /// The event that caused this notification.
    pub event: TriggerEvent,
    /// Timestamp of the notification.
    pub notified_at: DateTime<Utc>,
}

impl NotificationPayload {
    fn from_job(job: &Job, event: TriggerEvent) -> Self {
        Self {
            job_id: job.id,
            job_name: job.name.clone(),
            job_status: job.status.to_string(),
            job_tags: job.tags.clone(),
            error: job.error.clone(),
            event,
            notified_at: Utc::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// NotificationRecord
// ---------------------------------------------------------------------------

/// A historical record of a dispatched notification.
#[derive(Debug, Clone)]
pub struct NotificationRecord {
    /// Unique record ID.
    pub id: Uuid,
    /// ID of the rule that fired.
    pub rule_id: Uuid,
    /// Name of the rule that fired.
    pub rule_name: String,
    /// The notification payload.
    pub payload: NotificationPayload,
    /// Whether the notification was successfully sent.
    pub success: bool,
    /// Error message if sending failed.
    pub send_error: Option<String>,
}

// ---------------------------------------------------------------------------
// NotificationSender trait
// ---------------------------------------------------------------------------

/// Trait for dispatching a notification to a destination.
///
/// Implementors handle the actual I/O (HTTP POST, email, etc.).  The method is
/// synchronous so it can be used without requiring an async runtime.
pub trait NotificationSender: Send + Sync {
    /// Send a notification to the given destination.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error string on failure.
    fn send(
        &self,
        destination: &NotificationDestination,
        payload: &NotificationPayload,
    ) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// NoopSender
// ---------------------------------------------------------------------------

/// A `NotificationSender` that silently discards all notifications.
///
/// Useful for tests and scenarios where notification side-effects are not
/// desired.
pub struct NoopSender;

impl NotificationSender for NoopSender {
    fn send(
        &self,
        _destination: &NotificationDestination,
        _payload: &NotificationPayload,
    ) -> Result<(), String> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CapturingSender
// ---------------------------------------------------------------------------

/// A `NotificationSender` that records every notification for later inspection.
///
/// Thread-safe via an internal `Mutex`.
pub struct CapturingSender {
    captured: std::sync::Mutex<Vec<(NotificationDestination, NotificationPayload)>>,
}

impl CapturingSender {
    /// Create an empty `CapturingSender`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            captured: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Return a snapshot of all captured notifications.
    #[must_use]
    pub fn captured(&self) -> Vec<(NotificationDestination, NotificationPayload)> {
        self.captured
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }

    /// Number of captured notifications.
    #[must_use]
    pub fn count(&self) -> usize {
        self.captured
            .lock()
            .map(|g| g.len())
            .unwrap_or(0)
    }
}

impl Default for CapturingSender {
    fn default() -> Self {
        Self::new()
    }
}

impl NotificationSender for CapturingSender {
    fn send(
        &self,
        destination: &NotificationDestination,
        payload: &NotificationPayload,
    ) -> Result<(), String> {
        if let Ok(mut guard) = self.captured.lock() {
            guard.push((destination.clone(), payload.clone()));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FailingSender
// ---------------------------------------------------------------------------

/// A `NotificationSender` that always returns an error.  Useful for testing
/// error-handling paths in `NotificationManager`.
pub struct FailingSender {
    /// The error message to return.
    pub error_message: String,
}

impl FailingSender {
    /// Create a new `FailingSender` with the given error message.
    #[must_use]
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            error_message: msg.into(),
        }
    }
}

impl NotificationSender for FailingSender {
    fn send(
        &self,
        _destination: &NotificationDestination,
        _payload: &NotificationPayload,
    ) -> Result<(), String> {
        Err(self.error_message.clone())
    }
}

// ---------------------------------------------------------------------------
// NotificationStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the notification manager.
#[derive(Debug, Clone, Default)]
pub struct NotificationStats {
    /// Total notifications fired (across all rules).
    pub total_fired: u64,
    /// Total notifications successfully delivered.
    pub total_delivered: u64,
    /// Total notifications that failed to deliver.
    pub total_failed: u64,
    /// Counts per rule name.
    pub per_rule: HashMap<String, u64>,
}

// ---------------------------------------------------------------------------
// NotificationManager
// ---------------------------------------------------------------------------

/// Central manager that evaluates rules and dispatches notifications.
pub struct NotificationManager {
    /// Registered rules, keyed by rule ID.
    rules: Vec<NotificationRule>,
    /// Sender used for I/O.
    sender: Arc<dyn NotificationSender>,
    /// Notification history (bounded by `max_history`).
    history: Vec<NotificationRecord>,
    /// Maximum number of history entries to retain.
    max_history: usize,
    /// Aggregate statistics.
    stats: NotificationStats,
}

impl NotificationManager {
    /// Create a new manager with the given sender and a default history limit
    /// of 10 000 entries.
    #[must_use]
    pub fn new(sender: Arc<dyn NotificationSender>) -> Self {
        Self {
            rules: Vec::new(),
            sender,
            history: Vec::new(),
            max_history: 10_000,
            stats: NotificationStats::default(),
        }
    }

    /// Override the maximum number of history entries retained.
    #[must_use]
    pub fn with_max_history(mut self, limit: usize) -> Self {
        self.max_history = limit;
        self
    }

    /// Register a new notification rule.
    pub fn add_rule(&mut self, rule: NotificationRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by ID.  Returns `true` if a rule was removed.
    pub fn remove_rule(&mut self, rule_id: Uuid) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.id != rule_id);
        self.rules.len() < before
    }

    /// Enable or disable a rule by ID.  Returns `true` if the rule was found.
    pub fn set_rule_enabled(&mut self, rule_id: Uuid, enabled: bool) -> bool {
        for rule in &mut self.rules {
            if rule.id == rule_id {
                rule.enabled = enabled;
                return true;
            }
        }
        false
    }

    /// Evaluate all rules for the given job and dispatch matching notifications.
    /// Returns the number of notifications dispatched.
    pub fn notify(&mut self, job: &Job) -> usize {
        let mut dispatched = 0;

        // Collect matching (rule, event) pairs to avoid borrow conflicts.
        let matches: Vec<(Uuid, String, Vec<TriggerEvent>, NotificationDestination)> = self
            .rules
            .iter()
            .filter(|r| r.fires_for(job))
            .map(|r| {
                (
                    r.id,
                    r.name.clone(),
                    r.trigger_events.clone(),
                    r.destination.clone(),
                )
            })
            .collect();

        for (rule_id, rule_name, events, destination) in matches {
            // Pick the first matching event for the payload.
            let event = events
                .iter()
                .find(|e| e.matches(job.status))
                .copied()
                .unwrap_or(TriggerEvent::OnCompletion);

            let payload = NotificationPayload::from_job(job, event);

            let result = self.sender.send(&destination, &payload);
            let success = result.is_ok();
            let send_error = result.err();

            let record = NotificationRecord {
                id: Uuid::new_v4(),
                rule_id,
                rule_name: rule_name.clone(),
                payload,
                success,
                send_error,
            };

            // Trim history if needed.
            if self.history.len() >= self.max_history {
                self.history.remove(0);
            }
            self.history.push(record);

            // Update stats.
            self.stats.total_fired += 1;
            if success {
                self.stats.total_delivered += 1;
            } else {
                self.stats.total_failed += 1;
            }
            *self.stats.per_rule.entry(rule_name).or_insert(0) += 1;

            dispatched += 1;
        }

        dispatched
    }

    /// Return the notification history slice.
    #[must_use]
    pub fn history(&self) -> &[NotificationRecord] {
        &self.history
    }

    /// Return aggregate statistics.
    #[must_use]
    pub fn stats(&self) -> &NotificationStats {
        &self.stats
    }

    /// Return a reference to the registered rules.
    #[must_use]
    pub fn rules(&self) -> &[NotificationRule] {
        &self.rules
    }

    /// Clear all history entries.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobPayload, Priority, TranscodeParams};

    fn make_job(status: JobStatus, tags: Vec<&str>) -> Job {
        let params = TranscodeParams {
            input: "in.mp4".into(),
            output: "out.mp4".into(),
            video_codec: "h264".into(),
            audio_codec: "aac".into(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".into(),
            hw_accel: None,
        };
        let mut job = Job::new("encode".into(), Priority::Normal, JobPayload::Transcode(params));
        job.status = status;
        job.tags = tags.into_iter().map(|s| s.to_string()).collect();
        job
    }

    #[test]
    fn test_noop_sender_does_not_error() {
        let sender = Arc::new(NoopSender);
        let rule = NotificationRule::new(
            "all-done",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        let mut mgr = NotificationManager::new(sender);
        mgr.add_rule(rule);
        let job = make_job(JobStatus::Completed, vec![]);
        let dispatched = mgr.notify(&job);
        assert_eq!(dispatched, 1);
        assert_eq!(mgr.stats().total_delivered, 1);
    }

    #[test]
    fn test_on_success_fires_only_for_completed() {
        let sender = Arc::new(CapturingSender::new());
        let rule = NotificationRule::new(
            "success-only",
            vec![TriggerEvent::OnSuccess],
            NotificationDestination::Log,
        );
        let sender_clone = Arc::clone(&sender);
        let mut mgr = NotificationManager::new(sender_clone);
        mgr.add_rule(rule);

        let completed = make_job(JobStatus::Completed, vec![]);
        let failed = make_job(JobStatus::Failed, vec![]);
        mgr.notify(&completed);
        mgr.notify(&failed);

        assert_eq!(sender.count(), 1);
    }

    #[test]
    fn test_on_failure_fires_only_for_failed() {
        let sender = Arc::new(CapturingSender::new());
        let rule = NotificationRule::new(
            "fail-rule",
            vec![TriggerEvent::OnFailure],
            NotificationDestination::Log,
        );
        let sender_clone = Arc::clone(&sender);
        let mut mgr = NotificationManager::new(sender_clone);
        mgr.add_rule(rule);

        let completed = make_job(JobStatus::Completed, vec![]);
        let failed = make_job(JobStatus::Failed, vec![]);
        mgr.notify(&completed);
        mgr.notify(&failed);

        assert_eq!(sender.count(), 1);
        assert_eq!(
            sender.captured()[0].1.event,
            TriggerEvent::OnFailure
        );
    }

    #[test]
    fn test_tag_filter_restricts_notifications() {
        let sender = Arc::new(CapturingSender::new());
        let rule = NotificationRule::new(
            "video-only",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        )
        .with_tag_filter(vec!["video".into()]);
        let sender_clone = Arc::clone(&sender);
        let mut mgr = NotificationManager::new(sender_clone);
        mgr.add_rule(rule);

        let tagged = make_job(JobStatus::Completed, vec!["video"]);
        let untagged = make_job(JobStatus::Completed, vec!["audio"]);
        mgr.notify(&tagged);
        mgr.notify(&untagged);

        assert_eq!(sender.count(), 1);
    }

    #[test]
    fn test_disabled_rule_does_not_fire() {
        let sender = Arc::new(CapturingSender::new());
        let mut rule = NotificationRule::new(
            "disabled",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        rule.enabled = false;
        let sender_clone = Arc::clone(&sender);
        let mut mgr = NotificationManager::new(sender_clone);
        mgr.add_rule(rule);

        mgr.notify(&make_job(JobStatus::Completed, vec![]));
        assert_eq!(sender.count(), 0);
    }

    #[test]
    fn test_failing_sender_records_error() {
        let sender = Arc::new(FailingSender::new("network error"));
        let rule = NotificationRule::new(
            "fail-send",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Webhook { url: "http://example.com".into() },
        );
        let mut mgr = NotificationManager::new(sender);
        mgr.add_rule(rule);
        mgr.notify(&make_job(JobStatus::Completed, vec![]));

        let stats = mgr.stats();
        assert_eq!(stats.total_fired, 1);
        assert_eq!(stats.total_failed, 1);
        assert!(mgr.history()[0].send_error.is_some());
    }

    #[test]
    fn test_remove_rule() {
        let sender = Arc::new(CapturingSender::new());
        let rule = NotificationRule::new(
            "removable",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        let rule_id = rule.id;
        let sender_clone = Arc::clone(&sender);
        let mut mgr = NotificationManager::new(sender_clone);
        mgr.add_rule(rule);
        assert_eq!(mgr.rules().len(), 1);
        let removed = mgr.remove_rule(rule_id);
        assert!(removed);
        assert_eq!(mgr.rules().len(), 0);
    }

    #[test]
    fn test_history_bounded() {
        let sender = Arc::new(NoopSender);
        let rule = NotificationRule::new(
            "many",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        let mut mgr = NotificationManager::new(sender).with_max_history(3);
        mgr.add_rule(rule);

        for _ in 0..5 {
            mgr.notify(&make_job(JobStatus::Completed, vec![]));
        }
        assert_eq!(mgr.history().len(), 3);
    }

    #[test]
    fn test_per_rule_stats() {
        let sender = Arc::new(NoopSender);
        let rule = NotificationRule::new(
            "tracked",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        let mut mgr = NotificationManager::new(sender);
        mgr.add_rule(rule);
        mgr.notify(&make_job(JobStatus::Completed, vec![]));
        mgr.notify(&make_job(JobStatus::Completed, vec![]));
        assert_eq!(mgr.stats().per_rule.get("tracked").copied(), Some(2));
    }

    #[test]
    fn test_clear_history() {
        let sender = Arc::new(NoopSender);
        let rule = NotificationRule::new(
            "r",
            vec![TriggerEvent::OnCompletion],
            NotificationDestination::Log,
        );
        let mut mgr = NotificationManager::new(sender);
        mgr.add_rule(rule);
        mgr.notify(&make_job(JobStatus::Completed, vec![]));
        assert_eq!(mgr.history().len(), 1);
        mgr.clear_history();
        assert_eq!(mgr.history().len(), 0);
    }
}
