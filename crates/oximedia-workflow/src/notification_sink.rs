//! Notification sink integrations for workflow events.
//!
//! This module provides a [`NotificationSink`] trait and three concrete
//! implementations:
//!
//! - [`SlackSink`]: sends messages to a Slack webhook URL with configurable
//!   channel, username, and icon.
//! - [`EmailSink`]: produces RFC 5321-formatted SMTP messages (transport-agnostic;
//!   does not initiate TCP connections — callers relay the [`EmailMessage`]).
//! - [`PagerDutySink`]: builds PagerDuty Events API v2 JSON payloads with
//!   severity mapping and deduplication keys.
//!
//! [`NotificationRouter`] dispatches a single [`SinkEvent`] to multiple sinks
//! concurrently and collects per-sink outcomes.
//!
//! # Design
//!
//! All sinks are **transport-agnostic**: they build and return payloads /
//! messages rather than performing actual network I/O.  This keeps the module
//! free of async HTTP clients and makes it trivially testable.  Callers provide
//! their own transport (reqwest, hyper, `tokio::net::TcpStream`, etc.).
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::notification_sink::{
//!     NotificationRouter, SlackSink, EmailSink, PagerDutySink,
//!     SmtpConfig, PagerDutyConfig, SinkEvent, PagerDutySeverity,
//! };
//!
//! let slack = SlackSink::new("https://hooks.slack.com/T000/B000/xxx")
//!     .with_channel("#ops")
//!     .with_username("OxiMedia Bot");
//!
//! let email = EmailSink::new(SmtpConfig {
//!     from: "ops@example.com".to_string(),
//!     to: vec!["alerts@example.com".to_string()],
//!     subject_prefix: "[OxiMedia]".to_string(),
//! });
//!
//! let pager = PagerDutySink::new(PagerDutyConfig {
//!     routing_key: "abc123".to_string(),
//!     default_severity: PagerDutySeverity::Error,
//!     service_name: "OxiMedia Pipeline".to_string(),
//! });
//!
//! let mut router = NotificationRouter::new();
//! router.add_sink(Box::new(slack));
//! router.add_sink(Box::new(email));
//! router.add_sink(Box::new(pager));
//!
//! let event = SinkEvent::workflow_failed("wf-001", "transcode-pipeline", "disk full");
//! let outcomes = router.dispatch(&event);
//! assert_eq!(outcomes.outcomes.len(), 3);
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// SinkEvent
// ---------------------------------------------------------------------------

/// Severity level for a sink event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EventSeverity {
    /// Informational.
    Info,
    /// Non-critical warning.
    Warning,
    /// Recoverable error.
    Error,
    /// Unrecoverable / page-worthy.
    Critical,
}

impl EventSeverity {
    /// Short lowercase label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }

    /// Numeric level (0 = info, 3 = critical).
    #[must_use]
    pub const fn level(self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }
}

impl std::fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// An event emitted by the workflow engine that should be forwarded to sinks.
#[derive(Debug, Clone)]
pub struct SinkEvent {
    /// Short machine-readable event type (e.g. `"workflow.failed"`).
    pub event_type: String,
    /// Human-readable title for the notification.
    pub title: String,
    /// Detailed description / error message.
    pub body: String,
    /// Severity level used for routing and PagerDuty mapping.
    pub severity: EventSeverity,
    /// Arbitrary key-value metadata (workflow_id, task_name, etc.).
    pub metadata: HashMap<String, String>,
    /// Unix epoch in milliseconds.
    pub timestamp_ms: u64,
    /// Optional deduplication key (used by PagerDuty and idempotent sinks).
    pub dedup_key: Option<String>,
}

impl SinkEvent {
    /// Create a fully specified event.
    #[must_use]
    pub fn new(
        event_type: impl Into<String>,
        title: impl Into<String>,
        body: impl Into<String>,
        severity: EventSeverity,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            title: title.into(),
            body: body.into(),
            severity,
            metadata: HashMap::new(),
            timestamp_ms: current_timestamp_ms(),
            dedup_key: None,
        }
    }

    /// Convenience constructor for a workflow-completed event.
    #[must_use]
    pub fn workflow_completed(workflow_id: &str, workflow_name: &str) -> Self {
        let mut ev = Self::new(
            "workflow.completed",
            format!("Workflow completed: {workflow_name}"),
            format!("Workflow '{workflow_name}' (id={workflow_id}) completed successfully."),
            EventSeverity::Info,
        );
        ev.metadata
            .insert("workflow_id".to_string(), workflow_id.to_string());
        ev.metadata
            .insert("workflow_name".to_string(), workflow_name.to_string());
        ev
    }

    /// Convenience constructor for a workflow-failed event.
    #[must_use]
    pub fn workflow_failed(workflow_id: &str, workflow_name: &str, reason: &str) -> Self {
        let mut ev = Self::new(
            "workflow.failed",
            format!("Workflow FAILED: {workflow_name}"),
            format!("Workflow '{workflow_name}' (id={workflow_id}) failed: {reason}"),
            EventSeverity::Error,
        );
        ev.metadata
            .insert("workflow_id".to_string(), workflow_id.to_string());
        ev.metadata
            .insert("workflow_name".to_string(), workflow_name.to_string());
        ev.metadata.insert("reason".to_string(), reason.to_string());
        ev.dedup_key = Some(format!("wf-failed-{workflow_id}"));
        ev
    }

    /// Convenience constructor for a task-failed event.
    #[must_use]
    pub fn task_failed(workflow_id: &str, task_name: &str, reason: &str) -> Self {
        let mut ev = Self::new(
            "task.failed",
            format!("Task FAILED: {task_name}"),
            format!("Task '{task_name}' in workflow {workflow_id} failed: {reason}"),
            EventSeverity::Warning,
        );
        ev.metadata
            .insert("workflow_id".to_string(), workflow_id.to_string());
        ev.metadata
            .insert("task_name".to_string(), task_name.to_string());
        ev
    }

    /// Builder: add a metadata key-value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Builder: set a deduplication key.
    #[must_use]
    pub fn with_dedup_key(mut self, key: impl Into<String>) -> Self {
        self.dedup_key = Some(key.into());
        self
    }
}

fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// NotificationSink trait
// ---------------------------------------------------------------------------

/// Outcome of a single sink dispatch attempt.
#[derive(Debug, Clone)]
pub struct SinkOutcome {
    /// Name of the sink (for logging and debugging).
    pub sink_name: String,
    /// Whether the sink accepted and processed the event.
    pub success: bool,
    /// Human-readable summary of what the sink produced or the error message.
    pub message: String,
}

impl SinkOutcome {
    /// Create a successful outcome.
    #[must_use]
    pub fn ok(sink_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            sink_name: sink_name.into(),
            success: true,
            message: message.into(),
        }
    }

    /// Create a failed outcome.
    #[must_use]
    pub fn err(sink_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            sink_name: sink_name.into(),
            success: false,
            message: message.into(),
        }
    }
}

/// A destination that can receive workflow notification events.
///
/// Implementations are transport-agnostic: they build payloads and return
/// [`SinkOutcome`]s describing what would be sent.
pub trait NotificationSink: Send + Sync {
    /// Short, stable name for this sink (e.g. `"slack"`, `"email"`).
    fn name(&self) -> &str;

    /// Dispatch `event` to this sink and return the outcome.
    ///
    /// Implementations should never panic; all errors must be captured in the
    /// returned [`SinkOutcome`].
    fn dispatch(&self, event: &SinkEvent) -> SinkOutcome;

    /// Return `true` when this sink should receive events of the given severity.
    ///
    /// The default implementation accepts all severities.
    fn accepts_severity(&self, _severity: EventSeverity) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// SlackSink
// ---------------------------------------------------------------------------

/// A Slack Block Kit / incoming-webhook message ready to POST.
#[derive(Debug, Clone)]
pub struct SlackMessage {
    /// Slack channel override (e.g. `"#ops"`).
    pub channel: Option<String>,
    /// Bot username shown in Slack.
    pub username: Option<String>,
    /// Emoji icon (e.g. `":rotating_light:"`).
    pub icon_emoji: Option<String>,
    /// Serialized JSON body of the Slack API payload.
    pub payload_json: String,
}

/// Slack incoming-webhook sink.
///
/// Builds a JSON payload suitable for Slack's incoming-webhook API.
/// Does **not** perform HTTP POST — callers forward [`SlackMessage::payload_json`].
#[derive(Debug, Clone)]
pub struct SlackSink {
    /// Incoming-webhook URL (e.g. `"https://hooks.slack.com/T…/B…/xxx"`).
    pub webhook_url: String,
    /// Optional channel override.
    pub channel: Option<String>,
    /// Optional bot username.
    pub username: Option<String>,
    /// Optional emoji icon.
    pub icon_emoji: Option<String>,
    /// Minimum severity to forward (default: Info).
    pub min_severity: EventSeverity,
}

impl SlackSink {
    /// Create a new Slack sink for the given webhook URL.
    #[must_use]
    pub fn new(webhook_url: impl Into<String>) -> Self {
        Self {
            webhook_url: webhook_url.into(),
            channel: None,
            username: None,
            icon_emoji: None,
            min_severity: EventSeverity::Info,
        }
    }

    /// Set the target channel.
    #[must_use]
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    /// Set the bot username.
    #[must_use]
    pub fn with_username(mut self, username: impl Into<String>) -> Self {
        self.username = Some(username.into());
        self
    }

    /// Set the emoji icon.
    #[must_use]
    pub fn with_icon(mut self, emoji: impl Into<String>) -> Self {
        self.icon_emoji = Some(emoji.into());
        self
    }

    /// Set the minimum severity for this sink.
    #[must_use]
    pub fn with_min_severity(mut self, min: EventSeverity) -> Self {
        self.min_severity = min;
        self
    }

    /// Build the Slack message for `event` without sending it.
    #[must_use]
    pub fn build_message(&self, event: &SinkEvent) -> SlackMessage {
        let color = match event.severity {
            EventSeverity::Info => "#36a64f",
            EventSeverity::Warning => "#ffcc00",
            EventSeverity::Error => "#ff4444",
            EventSeverity::Critical => "#cc0000",
        };

        let attachment = serde_json::json!({
            "color": color,
            "title": event.title,
            "text": event.body,
            "footer": format!("severity={} type={}", event.severity.label(), event.event_type),
            "ts": event.timestamp_ms / 1000,
        });

        let mut payload = serde_json::json!({
            "attachments": [attachment],
        });

        if let Some(ch) = &self.channel {
            payload["channel"] = serde_json::json!(ch);
        }
        if let Some(un) = &self.username {
            payload["username"] = serde_json::json!(un);
        }
        if let Some(ic) = &self.icon_emoji {
            payload["icon_emoji"] = serde_json::json!(ic);
        }

        SlackMessage {
            channel: self.channel.clone(),
            username: self.username.clone(),
            icon_emoji: self.icon_emoji.clone(),
            payload_json: payload.to_string(),
        }
    }
}

impl NotificationSink for SlackSink {
    fn name(&self) -> &str {
        "slack"
    }

    fn dispatch(&self, event: &SinkEvent) -> SinkOutcome {
        if !self.accepts_severity(event.severity) {
            return SinkOutcome::ok(
                "slack",
                format!(
                    "skipped: severity {} below minimum {}",
                    event.severity.label(),
                    self.min_severity.label()
                ),
            );
        }
        let msg = self.build_message(event);
        SinkOutcome::ok(
            "slack",
            format!(
                "POST {} payload_len={}",
                self.webhook_url,
                msg.payload_json.len()
            ),
        )
    }

    fn accepts_severity(&self, severity: EventSeverity) -> bool {
        severity >= self.min_severity
    }
}

// ---------------------------------------------------------------------------
// EmailSink
// ---------------------------------------------------------------------------

/// SMTP configuration for the email sink.
#[derive(Debug, Clone)]
pub struct SmtpConfig {
    /// Sender address.
    pub from: String,
    /// List of recipient addresses.
    pub to: Vec<String>,
    /// Subject line prefix (e.g. `"[OxiMedia]"`).
    pub subject_prefix: String,
}

/// An RFC 5321-formatted email message ready to relay.
#[derive(Debug, Clone)]
pub struct EmailMessage {
    /// MAIL FROM envelope address.
    pub from: String,
    /// RCPT TO addresses.
    pub to: Vec<String>,
    /// Subject line.
    pub subject: String,
    /// Plain-text body.
    pub body: String,
}

/// Email notification sink.
///
/// Produces [`EmailMessage`] values formatted for SMTP delivery.
/// Transport (connection, TLS, AUTH) is left to the caller.
#[derive(Debug, Clone)]
pub struct EmailSink {
    /// SMTP configuration.
    pub config: SmtpConfig,
    /// Minimum severity to forward (default: Warning).
    pub min_severity: EventSeverity,
}

impl EmailSink {
    /// Create a new email sink.
    #[must_use]
    pub fn new(config: SmtpConfig) -> Self {
        Self {
            config,
            min_severity: EventSeverity::Warning,
        }
    }

    /// Set the minimum severity.
    #[must_use]
    pub fn with_min_severity(mut self, min: EventSeverity) -> Self {
        self.min_severity = min;
        self
    }

    /// Build the email message for `event` without sending it.
    #[must_use]
    pub fn build_message(&self, event: &SinkEvent) -> EmailMessage {
        let subject = format!(
            "{} [{}] {}",
            self.config.subject_prefix,
            event.severity.label().to_uppercase(),
            event.title
        );

        let mut body_lines = vec![
            event.body.clone(),
            String::new(),
            format!("Event type : {}", event.event_type),
            format!("Severity   : {}", event.severity.label()),
            format!("Timestamp  : {}ms", event.timestamp_ms),
        ];

        if !event.metadata.is_empty() {
            body_lines.push(String::new());
            body_lines.push("Metadata:".to_string());
            let mut sorted: Vec<(&String, &String)> = event.metadata.iter().collect();
            sorted.sort_by_key(|(k, _)| k.as_str());
            for (k, v) in sorted {
                body_lines.push(format!("  {k}: {v}"));
            }
        }

        EmailMessage {
            from: self.config.from.clone(),
            to: self.config.to.clone(),
            subject,
            body: body_lines.join("\n"),
        }
    }
}

impl NotificationSink for EmailSink {
    fn name(&self) -> &str {
        "email"
    }

    fn dispatch(&self, event: &SinkEvent) -> SinkOutcome {
        if !self.accepts_severity(event.severity) {
            return SinkOutcome::ok(
                "email",
                format!(
                    "skipped: severity {} below minimum {}",
                    event.severity.label(),
                    self.min_severity.label()
                ),
            );
        }
        let msg = self.build_message(event);
        SinkOutcome::ok(
            "email",
            format!(
                "SMTP from={} to={} subject={:?}",
                msg.from,
                msg.to.join(","),
                msg.subject
            ),
        )
    }

    fn accepts_severity(&self, severity: EventSeverity) -> bool {
        severity >= self.min_severity
    }
}

// ---------------------------------------------------------------------------
// PagerDutySink
// ---------------------------------------------------------------------------

/// PagerDuty Events API v2 severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PagerDutySeverity {
    /// PD `info`.
    Info,
    /// PD `warning`.
    Warning,
    /// PD `error`.
    Error,
    /// PD `critical`.
    Critical,
}

impl PagerDutySeverity {
    /// The lowercase string used in PD API payloads.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }

    /// Map an [`EventSeverity`] to the closest PagerDuty severity.
    #[must_use]
    pub fn from_event_severity(s: EventSeverity) -> Self {
        match s {
            EventSeverity::Info => Self::Info,
            EventSeverity::Warning => Self::Warning,
            EventSeverity::Error => Self::Error,
            EventSeverity::Critical => Self::Critical,
        }
    }
}

/// Configuration for the PagerDuty Events API v2 sink.
#[derive(Debug, Clone)]
pub struct PagerDutyConfig {
    /// PagerDuty Integration Key (aka routing key).
    pub routing_key: String,
    /// Default severity when the event does not carry one.
    pub default_severity: PagerDutySeverity,
    /// Human-readable service / component name included in the payload.
    pub service_name: String,
}

/// A PagerDuty Events API v2 payload ready to POST to
/// `https://events.pagerduty.com/v2/enqueue`.
#[derive(Debug, Clone)]
pub struct PagerDutyPayload {
    /// JSON body string.
    pub body_json: String,
    /// The deduplication key embedded in the payload.
    pub dedup_key: String,
}

/// PagerDuty notification sink.
///
/// Builds Events API v2 payloads.  Does **not** perform HTTP POST — callers
/// must send [`PagerDutyPayload::body_json`] to the PD enqueue endpoint.
///
/// Only events with severity >= [`EventSeverity::Warning`] are forwarded by
/// default; override with [`PagerDutySink::with_min_severity`].
#[derive(Debug, Clone)]
pub struct PagerDutySink {
    /// PagerDuty configuration.
    pub config: PagerDutyConfig,
    /// Minimum severity to forward (default: Warning).
    pub min_severity: EventSeverity,
}

impl PagerDutySink {
    /// Create a new PagerDuty sink.
    #[must_use]
    pub fn new(config: PagerDutyConfig) -> Self {
        Self {
            config,
            min_severity: EventSeverity::Warning,
        }
    }

    /// Set the minimum severity.
    #[must_use]
    pub fn with_min_severity(mut self, min: EventSeverity) -> Self {
        self.min_severity = min;
        self
    }

    /// Build the PD Events API v2 payload for `event`.
    #[must_use]
    pub fn build_payload(&self, event: &SinkEvent) -> PagerDutyPayload {
        let pd_severity = PagerDutySeverity::from_event_severity(event.severity);
        let dedup_key = event
            .dedup_key
            .clone()
            .unwrap_or_else(|| format!("oximedia-{}", event.event_type));

        let mut custom_details = serde_json::Map::new();
        for (k, v) in &event.metadata {
            custom_details.insert(k.clone(), serde_json::json!(v));
        }
        custom_details.insert(
            "event_type".to_string(),
            serde_json::json!(event.event_type),
        );

        let body = serde_json::json!({
            "routing_key": self.config.routing_key,
            "event_action": "trigger",
            "dedup_key": dedup_key,
            "payload": {
                "summary": event.title,
                "severity": pd_severity.as_str(),
                "source": self.config.service_name,
                "timestamp": event.timestamp_ms,
                "custom_details": custom_details,
            },
            "client": "OxiMedia Workflow",
            "client_url": "https://github.com/cool-japan/oximedia",
        });

        PagerDutyPayload {
            body_json: body.to_string(),
            dedup_key,
        }
    }
}

impl NotificationSink for PagerDutySink {
    fn name(&self) -> &str {
        "pagerduty"
    }

    fn dispatch(&self, event: &SinkEvent) -> SinkOutcome {
        if !self.accepts_severity(event.severity) {
            return SinkOutcome::ok(
                "pagerduty",
                format!(
                    "skipped: severity {} below minimum {}",
                    event.severity.label(),
                    self.min_severity.label()
                ),
            );
        }
        let payload = self.build_payload(event);
        SinkOutcome::ok(
            "pagerduty",
            format!(
                "POST events.pagerduty.com/v2/enqueue dedup_key={} payload_len={}",
                payload.dedup_key,
                payload.body_json.len()
            ),
        )
    }

    fn accepts_severity(&self, severity: EventSeverity) -> bool {
        severity >= self.min_severity
    }
}

// ---------------------------------------------------------------------------
// NotificationRouter
// ---------------------------------------------------------------------------

/// Summary of a router dispatch round.
#[derive(Debug, Clone, Default)]
pub struct RouterSummary {
    /// Per-sink outcomes.
    pub outcomes: Vec<SinkOutcome>,
    /// Number of sinks that succeeded.
    pub success_count: usize,
    /// Number of sinks that failed.
    pub failure_count: usize,
    /// Number of sinks that skipped the event (severity filter).
    pub skip_count: usize,
}

impl RouterSummary {
    /// Returns `true` when all dispatches succeeded (none failed).
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failure_count == 0
    }
}

/// Dispatches a single [`SinkEvent`] to multiple [`NotificationSink`]s.
///
/// Sinks are called sequentially (no async / thread overhead).  For async
/// fan-out combine with [`crate::fan_pattern::FanExecutor`].
#[derive(Default)]
pub struct NotificationRouter {
    sinks: Vec<Box<dyn NotificationSink>>,
}

impl std::fmt::Debug for NotificationRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.sinks.iter().map(|s| s.name()).collect();
        f.debug_struct("NotificationRouter")
            .field("sinks", &names)
            .finish()
    }
}

impl NotificationRouter {
    /// Create an empty router.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sink.
    pub fn add_sink(&mut self, sink: Box<dyn NotificationSink>) {
        self.sinks.push(sink);
    }

    /// Number of registered sinks.
    #[must_use]
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }

    /// Dispatch `event` to all registered sinks and return a [`RouterSummary`].
    pub fn dispatch(&self, event: &SinkEvent) -> RouterSummary {
        let mut summary = RouterSummary::default();

        for sink in &self.sinks {
            let outcome = sink.dispatch(event);
            if outcome.success {
                if outcome.message.starts_with("skipped:") {
                    summary.skip_count += 1;
                } else {
                    summary.success_count += 1;
                }
            } else {
                summary.failure_count += 1;
            }
            summary.outcomes.push(outcome);
        }

        summary
    }

    /// Names of all registered sinks.
    #[must_use]
    pub fn sink_names(&self) -> Vec<&str> {
        self.sinks.iter().map(|s| s.name()).collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // EventSeverity
    // -----------------------------------------------------------------------

    #[test]
    fn severity_ordering() {
        assert!(EventSeverity::Info < EventSeverity::Warning);
        assert!(EventSeverity::Warning < EventSeverity::Error);
        assert!(EventSeverity::Error < EventSeverity::Critical);
    }

    #[test]
    fn severity_labels() {
        assert_eq!(EventSeverity::Info.label(), "info");
        assert_eq!(EventSeverity::Warning.label(), "warning");
        assert_eq!(EventSeverity::Error.label(), "error");
        assert_eq!(EventSeverity::Critical.label(), "critical");
    }

    // -----------------------------------------------------------------------
    // SinkEvent constructors
    // -----------------------------------------------------------------------

    #[test]
    fn sink_event_workflow_completed() {
        let ev = SinkEvent::workflow_completed("wf-001", "transcode");
        assert_eq!(ev.event_type, "workflow.completed");
        assert_eq!(ev.severity, EventSeverity::Info);
        assert_eq!(
            ev.metadata.get("workflow_id").map(String::as_str),
            Some("wf-001")
        );
    }

    #[test]
    fn sink_event_workflow_failed_has_dedup_key() {
        let ev = SinkEvent::workflow_failed("wf-002", "ingest", "disk full");
        assert!(ev.dedup_key.is_some());
        assert!(ev.dedup_key.as_deref().unwrap().contains("wf-002"));
        assert_eq!(ev.severity, EventSeverity::Error);
    }

    #[test]
    fn sink_event_task_failed() {
        let ev = SinkEvent::task_failed("wf-003", "qc-check", "bitrate too high");
        assert_eq!(ev.event_type, "task.failed");
        assert_eq!(ev.severity, EventSeverity::Warning);
        assert_eq!(
            ev.metadata.get("task_name").map(String::as_str),
            Some("qc-check")
        );
    }

    // -----------------------------------------------------------------------
    // SlackSink
    // -----------------------------------------------------------------------

    #[test]
    fn slack_sink_builds_valid_json_payload() {
        let sink = SlackSink::new("https://hooks.slack.com/T/B/xxx")
            .with_channel("#ops")
            .with_username("Bot");
        let ev = SinkEvent::workflow_failed("wf-1", "pipe", "oom");
        let msg = sink.build_message(&ev);
        let parsed: serde_json::Value =
            serde_json::from_str(&msg.payload_json).expect("valid JSON");
        assert!(parsed["attachments"].is_array());
        assert_eq!(parsed["channel"], "#ops");
        assert_eq!(parsed["username"], "Bot");
    }

    #[test]
    fn slack_sink_skips_below_min_severity() {
        let sink = SlackSink::new("https://x").with_min_severity(EventSeverity::Error);
        let ev = SinkEvent::workflow_completed("wf-1", "pipe");
        let outcome = sink.dispatch(&ev);
        assert!(outcome.success);
        assert!(outcome.message.starts_with("skipped:"));
    }

    #[test]
    fn slack_sink_dispatches_above_min_severity() {
        let sink = SlackSink::new("https://hooks.slack.com/x");
        let ev = SinkEvent::workflow_failed("wf-1", "pipe", "err");
        let outcome = sink.dispatch(&ev);
        assert!(outcome.success);
        assert!(outcome.message.contains("POST"));
    }

    // -----------------------------------------------------------------------
    // EmailSink
    // -----------------------------------------------------------------------

    #[test]
    fn email_sink_builds_correct_subject() {
        let sink = EmailSink::new(SmtpConfig {
            from: "ops@example.com".to_string(),
            to: vec!["alerts@example.com".to_string()],
            subject_prefix: "[OxiMedia]".to_string(),
        });
        let ev = SinkEvent::workflow_failed("wf-5", "encode", "crash");
        let msg = sink.build_message(&ev);
        assert!(msg.subject.starts_with("[OxiMedia]"));
        assert!(msg.subject.contains("ERROR"));
    }

    #[test]
    fn email_sink_body_contains_metadata() {
        let sink = EmailSink::new(SmtpConfig {
            from: "a@b.com".to_string(),
            to: vec!["c@d.com".to_string()],
            subject_prefix: "[X]".to_string(),
        });
        let ev = SinkEvent::workflow_failed("wf-6", "pipe", "oom").with_meta("region", "eu-west-1");
        let msg = sink.build_message(&ev);
        assert!(msg.body.contains("region"));
        assert!(msg.body.contains("eu-west-1"));
    }

    #[test]
    fn email_sink_skips_info_by_default() {
        let sink = EmailSink::new(SmtpConfig {
            from: "a@b.com".to_string(),
            to: vec![],
            subject_prefix: "X".to_string(),
        });
        let ev = SinkEvent::workflow_completed("wf-7", "pipe");
        let outcome = sink.dispatch(&ev);
        assert!(outcome.success);
        assert!(outcome.message.starts_with("skipped:"));
    }

    // -----------------------------------------------------------------------
    // PagerDutySink
    // -----------------------------------------------------------------------

    #[test]
    fn pagerduty_sink_payload_is_valid_json() {
        let sink = PagerDutySink::new(PagerDutyConfig {
            routing_key: "abc123".to_string(),
            default_severity: PagerDutySeverity::Error,
            service_name: "OxiMedia".to_string(),
        });
        let ev = SinkEvent::workflow_failed("wf-8", "pipe", "disk full");
        let payload = sink.build_payload(&ev);
        let parsed: serde_json::Value =
            serde_json::from_str(&payload.body_json).expect("valid JSON");
        assert_eq!(parsed["routing_key"], "abc123");
        assert_eq!(parsed["event_action"], "trigger");
        assert_eq!(parsed["payload"]["severity"], "error");
        assert_eq!(parsed["payload"]["source"], "OxiMedia");
    }

    #[test]
    fn pagerduty_sink_uses_event_dedup_key() {
        let sink = PagerDutySink::new(PagerDutyConfig {
            routing_key: "r".to_string(),
            default_severity: PagerDutySeverity::Warning,
            service_name: "S".to_string(),
        });
        let ev = SinkEvent::workflow_failed("wf-9", "p", "e");
        let payload = sink.build_payload(&ev);
        assert!(payload.dedup_key.contains("wf-9"));
    }

    #[test]
    fn pagerduty_severity_mapping() {
        assert_eq!(
            PagerDutySeverity::from_event_severity(EventSeverity::Critical).as_str(),
            "critical"
        );
        assert_eq!(
            PagerDutySeverity::from_event_severity(EventSeverity::Info).as_str(),
            "info"
        );
    }

    // -----------------------------------------------------------------------
    // NotificationRouter
    // -----------------------------------------------------------------------

    #[test]
    fn router_dispatches_to_all_sinks() {
        let mut router = NotificationRouter::new();
        router.add_sink(Box::new(
            SlackSink::new("https://hooks.slack.com/x").with_min_severity(EventSeverity::Info),
        ));
        router.add_sink(Box::new(
            EmailSink::new(SmtpConfig {
                from: "a@b.com".to_string(),
                to: vec!["c@d.com".to_string()],
                subject_prefix: "[X]".to_string(),
            })
            .with_min_severity(EventSeverity::Info),
        ));
        router.add_sink(Box::new(
            PagerDutySink::new(PagerDutyConfig {
                routing_key: "r".to_string(),
                default_severity: PagerDutySeverity::Error,
                service_name: "S".to_string(),
            })
            .with_min_severity(EventSeverity::Info),
        ));

        let ev = SinkEvent::workflow_failed("wf-x", "pipe", "crash");
        let summary = router.dispatch(&ev);
        assert_eq!(summary.outcomes.len(), 3);
        assert_eq!(summary.success_count, 3);
        assert_eq!(summary.failure_count, 0);
        assert!(summary.all_succeeded());
    }

    #[test]
    fn router_counts_skips_correctly() {
        let mut router = NotificationRouter::new();
        // Email default min = Warning; Info event should be skipped.
        router.add_sink(Box::new(EmailSink::new(SmtpConfig {
            from: "a@b.com".to_string(),
            to: vec![],
            subject_prefix: "[X]".to_string(),
        })));

        let ev = SinkEvent::workflow_completed("wf-z", "pipe");
        let summary = router.dispatch(&ev);
        assert_eq!(summary.skip_count, 1);
        assert_eq!(summary.success_count, 0);
    }

    #[test]
    fn router_sink_names() {
        let mut router = NotificationRouter::new();
        router.add_sink(Box::new(SlackSink::new("u")));
        router.add_sink(Box::new(PagerDutySink::new(PagerDutyConfig {
            routing_key: "r".to_string(),
            default_severity: PagerDutySeverity::Error,
            service_name: "s".to_string(),
        })));
        let names = router.sink_names();
        assert_eq!(names, vec!["slack", "pagerduty"]);
    }
}
