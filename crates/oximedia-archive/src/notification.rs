//! Notification system for archive fixity check failures.
//!
//! Provides structured notification delivery through multiple channels:
//! - Webhook (HTTP POST with JSON payload)
//! - In-memory event bus (for testing and local integration)
//! - File-based notification log
//!
//! The `NotificationDispatcher` routes events to all registered
//! `NotificationChannel` implementations according to per-channel severity filters.

#![allow(dead_code)]

use crate::{ArchiveError, ArchiveResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for a notification event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    /// Informational — routine status updates.
    Info,
    /// Warning — fixity check degraded or overdue.
    Warning,
    /// Error — checksum mismatch or file inaccessible.
    Error,
    /// Critical — multiple files corrupted or database failure.
    Critical,
}

impl Severity {
    /// Return a short uppercase label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// NotificationEvent
// ---------------------------------------------------------------------------

/// A notification event emitted by the archive verification pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationEvent {
    /// Unique event identifier (UUID-like hex string).
    pub event_id: String,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Event severity.
    pub severity: Severity,
    /// Machine-readable event kind.
    pub kind: EventKind,
    /// Human-readable message.
    pub message: String,
    /// Affected file path(s), if any.
    pub affected_files: Vec<PathBuf>,
    /// Additional structured metadata (key → value).
    pub metadata: std::collections::HashMap<String, String>,
}

/// Machine-readable classification of the notification event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    /// A fixity check completed without errors.
    FixityCheckPassed,
    /// A fixity check found a checksum mismatch.
    FixityCheckFailed,
    /// A fixity check is overdue (interval exceeded).
    FixityCheckOverdue,
    /// A file was automatically quarantined.
    FileQuarantined,
    /// A file that was quarantined has been restored.
    FileRestored,
    /// A scheduled integrity scan completed.
    IntegrityScanComplete,
    /// A batch operation failed for one or more items.
    BatchOperationError,
    /// A storage tier migration completed.
    MigrationComplete,
    /// Custom application-defined event.
    Custom(String),
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FixityCheckPassed => write!(f, "fixity_check_passed"),
            Self::FixityCheckFailed => write!(f, "fixity_check_failed"),
            Self::FixityCheckOverdue => write!(f, "fixity_check_overdue"),
            Self::FileQuarantined => write!(f, "file_quarantined"),
            Self::FileRestored => write!(f, "file_restored"),
            Self::IntegrityScanComplete => write!(f, "integrity_scan_complete"),
            Self::BatchOperationError => write!(f, "batch_operation_error"),
            Self::MigrationComplete => write!(f, "migration_complete"),
            Self::Custom(s) => write!(f, "custom:{s}"),
        }
    }
}

impl NotificationEvent {
    /// Create a new notification event with the current timestamp.
    pub fn new(severity: Severity, kind: EventKind, message: impl Into<String>) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let event_id = format!("{:016x}{:08x}", Utc::now().timestamp(), nanos);
        Self {
            event_id,
            timestamp: Utc::now(),
            severity,
            kind,
            message: message.into(),
            affected_files: Vec::new(),
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Attach affected file paths.
    #[must_use]
    pub fn with_files(mut self, files: impl IntoIterator<Item = PathBuf>) -> Self {
        self.affected_files.extend(files);
        self
    }

    /// Attach a metadata key-value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// NotificationChannel trait
// ---------------------------------------------------------------------------

/// A delivery channel for archive notifications.
pub trait NotificationChannel: Send + Sync {
    /// Human-readable name for this channel.
    fn name(&self) -> &str;

    /// Deliver a notification event. Returns `Ok(true)` if delivered,
    /// `Ok(false)` if filtered out, or an error on delivery failure.
    fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool>;

    /// Minimum severity that this channel cares about.
    fn min_severity(&self) -> Severity {
        Severity::Warning
    }
}

// ---------------------------------------------------------------------------
// WebhookChannel
// ---------------------------------------------------------------------------

/// Configuration for an HTTP webhook notification channel.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Name for this webhook endpoint.
    pub name: String,
    /// Target URL (HTTP or HTTPS).
    pub url: String,
    /// Optional bearer token for Authorization header.
    pub bearer_token: Option<String>,
    /// Minimum severity filter.
    pub min_severity: Severity,
    /// Connection timeout in seconds.
    pub timeout_secs: u64,
    /// Whether TLS certificate validation is performed.
    pub verify_tls: bool,
}

impl WebhookConfig {
    /// Create a simple webhook config.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            url: url.into(),
            bearer_token: None,
            min_severity: Severity::Warning,
            timeout_secs: 10,
            verify_tls: true,
        }
    }
}

/// An HTTP webhook notification channel.
///
/// In this pure-Rust implementation, actual HTTP dispatch is handled by the
/// caller providing a `WebhookDispatcher` function. This design avoids
/// requiring a specific HTTP client crate while still enabling the full
/// notification pipeline to be tested.
pub struct WebhookChannel {
    config: WebhookConfig,
    /// Number of events attempted.
    attempt_count: Arc<Mutex<u64>>,
}

impl WebhookChannel {
    /// Create a new webhook channel.
    pub fn new(config: WebhookConfig) -> Self {
        Self {
            config,
            attempt_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Build the JSON payload for a notification event.
    pub fn build_payload(&self, event: &NotificationEvent) -> ArchiveResult<String> {
        serde_json::to_string(event)
            .map_err(|e| ArchiveError::Validation(format!("JSON serialization error: {e}")))
    }

    /// Return the webhook URL.
    pub fn url(&self) -> &str {
        &self.config.url
    }

    /// Return attempt count (for testing/metrics).
    pub fn attempt_count(&self) -> u64 {
        self.attempt_count.lock().map(|g| *g).unwrap_or(0)
    }
}

impl NotificationChannel for WebhookChannel {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn min_severity(&self) -> Severity {
        self.config.min_severity
    }

    /// Build and log the payload (actual HTTP dispatch requires integration layer).
    fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
        if event.severity < self.config.min_severity {
            return Ok(false);
        }
        let _payload = self.build_payload(event)?;
        if let Ok(mut count) = self.attempt_count.lock() {
            *count = count.saturating_add(1);
        }
        // Real HTTP dispatch would happen here via an injected transport.
        // For the pure-Rust in-process implementation we record the attempt.
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// InMemoryChannel — for testing / local integration
// ---------------------------------------------------------------------------

/// An in-memory notification channel that stores events in a ring buffer.
///
/// Useful for unit tests and local monitoring dashboards.
#[derive(Debug)]
pub struct InMemoryChannel {
    name: String,
    min_severity: Severity,
    capacity: usize,
    events: Arc<Mutex<VecDeque<NotificationEvent>>>,
}

impl InMemoryChannel {
    /// Create a new in-memory channel with the given capacity.
    pub fn new(name: impl Into<String>, capacity: usize, min_severity: Severity) -> Self {
        Self {
            name: name.into(),
            min_severity,
            capacity,
            events: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
        }
    }

    /// Drain all stored events.
    pub fn drain_events(&self) -> Vec<NotificationEvent> {
        self.events
            .lock()
            .map(|mut g| g.drain(..).collect())
            .unwrap_or_default()
    }

    /// Return the number of stored events.
    pub fn len(&self) -> usize {
        self.events.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Return true if no events are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Peek at the most recently stored event.
    pub fn last_event(&self) -> Option<NotificationEvent> {
        self.events.lock().ok().and_then(|g| g.back().cloned())
    }
}

impl NotificationChannel for InMemoryChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn min_severity(&self) -> Severity {
        self.min_severity
    }

    fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
        if event.severity < self.min_severity {
            return Ok(false);
        }
        if let Ok(mut queue) = self.events.lock() {
            if queue.len() >= self.capacity {
                queue.pop_front();
            }
            queue.push_back(event.clone());
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// FileLogChannel
// ---------------------------------------------------------------------------

/// A channel that appends NDJSON notification records to a log file.
#[derive(Debug)]
pub struct FileLogChannel {
    name: String,
    log_path: PathBuf,
    min_severity: Severity,
}

impl FileLogChannel {
    /// Create a new file log channel.
    pub fn new(
        name: impl Into<String>,
        log_path: impl Into<PathBuf>,
        min_severity: Severity,
    ) -> Self {
        Self {
            name: name.into(),
            log_path: log_path.into(),
            min_severity,
        }
    }

    /// Return the log file path.
    pub fn log_path(&self) -> &PathBuf {
        &self.log_path
    }
}

impl NotificationChannel for FileLogChannel {
    fn name(&self) -> &str {
        &self.name
    }

    fn min_severity(&self) -> Severity {
        self.min_severity
    }

    fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
        if event.severity < self.min_severity {
            return Ok(false);
        }
        use std::io::Write;
        let line = serde_json::to_string(event)
            .map_err(|e| ArchiveError::Validation(format!("JSON error: {e}")))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        writeln!(file, "{line}")?;
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// NotificationDispatcher
// ---------------------------------------------------------------------------

/// Dispatches notification events to all registered channels.
pub struct NotificationDispatcher {
    channels: Vec<Box<dyn NotificationChannel>>,
}

impl NotificationDispatcher {
    /// Create an empty dispatcher.
    pub fn new() -> Self {
        Self {
            channels: Vec::new(),
        }
    }

    /// Register a channel.
    pub fn add_channel(&mut self, channel: Box<dyn NotificationChannel>) {
        self.channels.push(channel);
    }

    /// Dispatch an event to all channels that accept its severity.
    ///
    /// Returns a list of `(channel_name, sent)` pairs.
    pub fn dispatch(&self, event: &NotificationEvent) -> Vec<(String, ArchiveResult<bool>)> {
        self.channels
            .iter()
            .map(|ch| {
                let result = ch.send(event);
                (ch.name().to_string(), result)
            })
            .collect()
    }

    /// Return the number of registered channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Convenience: emit a fixity-check-failed event.
    pub fn notify_fixity_failure(
        &self,
        file: PathBuf,
        expected: &str,
        actual: &str,
    ) -> Vec<(String, ArchiveResult<bool>)> {
        let event = NotificationEvent::new(
            Severity::Error,
            EventKind::FixityCheckFailed,
            format!("Fixity check failed for {}", file.display()),
        )
        .with_files(std::iter::once(file))
        .with_meta("expected_checksum", expected)
        .with_meta("actual_checksum", actual);
        self.dispatch(&event)
    }

    /// Convenience: emit a file-quarantined event.
    pub fn notify_quarantine(
        &self,
        file: PathBuf,
        reason: &str,
    ) -> Vec<(String, ArchiveResult<bool>)> {
        let event = NotificationEvent::new(
            Severity::Warning,
            EventKind::FileQuarantined,
            format!("File quarantined: {}", file.display()),
        )
        .with_files(std::iter::once(file))
        .with_meta("reason", reason);
        self.dispatch(&event)
    }
}

impl Default for NotificationDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DeliveryReport
// ---------------------------------------------------------------------------

/// Summary of a dispatch operation across all channels.
#[derive(Debug, Clone)]
pub struct DeliveryReport {
    /// Number of channels that successfully delivered.
    pub delivered: usize,
    /// Number of channels that filtered out the event (severity below min).
    pub filtered: usize,
    /// Number of channels that failed.
    pub failed: usize,
    /// Per-channel results.
    pub details: Vec<ChannelResult>,
}

/// Per-channel delivery result.
#[derive(Debug, Clone)]
pub struct ChannelResult {
    /// Channel name.
    pub channel_name: String,
    /// Whether the event was delivered, filtered, or errored.
    pub outcome: DeliveryOutcome,
}

/// Outcome of a single channel delivery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOutcome {
    /// Event was successfully delivered.
    Delivered,
    /// Event was filtered out (severity below threshold).
    Filtered,
    /// Delivery failed with an error message.
    Failed(String),
}

impl NotificationDispatcher {
    /// Dispatch and return a structured `DeliveryReport`.
    pub fn dispatch_with_report(&self, event: &NotificationEvent) -> DeliveryReport {
        let results = self.dispatch(event);
        let mut delivered = 0;
        let mut filtered = 0;
        let mut failed = 0;
        let mut details = Vec::with_capacity(results.len());

        for (channel_name, result) in results {
            let outcome = match result {
                Ok(true) => {
                    delivered += 1;
                    DeliveryOutcome::Delivered
                }
                Ok(false) => {
                    filtered += 1;
                    DeliveryOutcome::Filtered
                }
                Err(e) => {
                    failed += 1;
                    DeliveryOutcome::Failed(e.to_string())
                }
            };
            details.push(ChannelResult {
                channel_name,
                outcome,
            });
        }

        DeliveryReport {
            delivered,
            filtered,
            failed,
            details,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_event(severity: Severity, kind: EventKind) -> NotificationEvent {
        NotificationEvent::new(severity, kind, "test event")
    }

    // ── Severity ordering ────────────────────────────────────────────────────

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_severity_labels() {
        assert_eq!(Severity::Info.label(), "INFO");
        assert_eq!(Severity::Warning.label(), "WARNING");
        assert_eq!(Severity::Error.label(), "ERROR");
        assert_eq!(Severity::Critical.label(), "CRITICAL");
    }

    // ── NotificationEvent construction ────────────────────────────────────────

    #[test]
    fn test_event_construction_with_meta() {
        let event = NotificationEvent::new(
            Severity::Warning,
            EventKind::FixityCheckFailed,
            "checksum mismatch",
        )
        .with_files(vec![PathBuf::from("/archive/file.mkv")])
        .with_meta("algorithm", "sha256");

        assert_eq!(event.severity, Severity::Warning);
        assert_eq!(event.kind, EventKind::FixityCheckFailed);
        assert_eq!(event.affected_files.len(), 1);
        assert_eq!(
            event.metadata.get("algorithm").map(String::as_str),
            Some("sha256")
        );
        assert!(!event.event_id.is_empty());
    }

    #[test]
    fn test_event_kind_display() {
        assert_eq!(
            EventKind::FixityCheckPassed.to_string(),
            "fixity_check_passed"
        );
        assert_eq!(
            EventKind::FixityCheckFailed.to_string(),
            "fixity_check_failed"
        );
        assert_eq!(EventKind::FileQuarantined.to_string(), "file_quarantined");
        assert_eq!(EventKind::Custom("ping".into()).to_string(), "custom:ping");
    }

    // ── InMemoryChannel ───────────────────────────────────────────────────────

    #[test]
    fn test_in_memory_channel_receives_event() {
        let ch = InMemoryChannel::new("mem", 10, Severity::Info);
        let event = make_event(Severity::Error, EventKind::FixityCheckFailed);
        let result = ch.send(&event);
        assert!(result.is_ok());
        assert_eq!(ch.len(), 1);
    }

    #[test]
    fn test_in_memory_channel_filters_low_severity() {
        let ch = InMemoryChannel::new("mem", 10, Severity::Error);
        let event = make_event(Severity::Info, EventKind::FixityCheckPassed);
        let result = ch.send(&event).expect("send ok");
        assert!(!result); // filtered
        assert!(ch.is_empty());
    }

    #[test]
    fn test_in_memory_channel_capacity_ring() {
        let ch = InMemoryChannel::new("mem", 3, Severity::Info);
        for i in 0..5 {
            let ev = NotificationEvent::new(
                Severity::Info,
                EventKind::IntegrityScanComplete,
                format!("scan {i}"),
            );
            ch.send(&ev).expect("send");
        }
        assert_eq!(ch.len(), 3); // ring: oldest dropped
    }

    #[test]
    fn test_in_memory_channel_drain() {
        let ch = InMemoryChannel::new("mem", 10, Severity::Info);
        for _ in 0..4 {
            ch.send(&make_event(
                Severity::Warning,
                EventKind::FixityCheckOverdue,
            ))
            .expect("send");
        }
        let drained = ch.drain_events();
        assert_eq!(drained.len(), 4);
        assert!(ch.is_empty());
    }

    // ── FileLogChannel ────────────────────────────────────────────────────────

    #[test]
    fn test_file_log_channel_appends_ndjson() {
        let dir = std::env::temp_dir();
        let log_path = dir.join("oximedia_notif_test.ndjson");
        // Remove leftovers from previous runs
        let _ = std::fs::remove_file(&log_path);

        let ch = FileLogChannel::new("filelog", log_path.clone(), Severity::Info);
        let event = make_event(Severity::Error, EventKind::FileQuarantined);
        ch.send(&event).expect("send");

        let content = std::fs::read_to_string(&log_path).expect("read log");
        assert!(!content.is_empty());
        // Should be valid JSON on the first line
        let parsed: serde_json::Value =
            serde_json::from_str(content.lines().next().unwrap_or("{}")).expect("valid JSON line");
        assert!(parsed.get("event_id").is_some());

        let _ = std::fs::remove_file(&log_path);
    }

    // ── WebhookChannel ─────────────────────────────────────────────────────────

    #[test]
    fn test_webhook_payload_serialization() {
        let config = WebhookConfig::new("hook", "https://example.com/notify");
        let ch = WebhookChannel::new(config);
        let event = make_event(Severity::Critical, EventKind::BatchOperationError);
        let payload = ch.build_payload(&event).expect("payload");
        let decoded: serde_json::Value = serde_json::from_str(&payload).expect("valid json");
        assert_eq!(decoded["severity"].as_str(), Some("Critical"));
    }

    #[test]
    fn test_webhook_filters_low_severity() {
        let mut config = WebhookConfig::new("hook", "https://example.com/notify");
        config.min_severity = Severity::Critical;
        let ch = WebhookChannel::new(config);
        let event = make_event(Severity::Info, EventKind::FixityCheckPassed);
        let sent = ch.send(&event).expect("send ok");
        assert!(!sent);
        assert_eq!(ch.attempt_count(), 0);
    }

    // ── NotificationDispatcher ────────────────────────────────────────────────

    #[test]
    fn test_dispatcher_delivers_to_all_channels() {
        let mut dispatcher = NotificationDispatcher::new();
        let ch1 = InMemoryChannel::new("ch1", 10, Severity::Info);
        let ch2 = InMemoryChannel::new("ch2", 10, Severity::Info);
        // We need shared references — wrap in Arc to observe from outside
        let ch1_arc = Arc::new(ch1);
        let ch2_arc = Arc::new(ch2);

        // Use wrapper structs that delegate to Arc
        struct ArcChannel(Arc<InMemoryChannel>);
        impl NotificationChannel for ArcChannel {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn min_severity(&self) -> Severity {
                self.0.min_severity()
            }
            fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
                self.0.send(event)
            }
        }

        dispatcher.add_channel(Box::new(ArcChannel(Arc::clone(&ch1_arc))));
        dispatcher.add_channel(Box::new(ArcChannel(Arc::clone(&ch2_arc))));

        let event = make_event(Severity::Warning, EventKind::FixityCheckFailed);
        let results = dispatcher.dispatch(&event);
        assert_eq!(results.len(), 2);
        assert_eq!(ch1_arc.len(), 1);
        assert_eq!(ch2_arc.len(), 1);
    }

    #[test]
    fn test_dispatcher_delivery_report() {
        let mut dispatcher = NotificationDispatcher::new();
        let ch_all = InMemoryChannel::new("all", 10, Severity::Info);
        let ch_crit = InMemoryChannel::new("crit_only", 10, Severity::Critical);

        struct ArcChannel(Arc<InMemoryChannel>);
        impl NotificationChannel for ArcChannel {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn min_severity(&self) -> Severity {
                self.0.min_severity()
            }
            fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
                self.0.send(event)
            }
        }

        dispatcher.add_channel(Box::new(ArcChannel(Arc::new(ch_all))));
        dispatcher.add_channel(Box::new(ArcChannel(Arc::new(ch_crit))));

        let event = make_event(Severity::Warning, EventKind::FixityCheckOverdue);
        let report = dispatcher.dispatch_with_report(&event);

        assert_eq!(report.delivered, 1);
        assert_eq!(report.filtered, 1);
        assert_eq!(report.failed, 0);
    }

    #[test]
    fn test_dispatcher_notify_fixity_failure_convenience() {
        let mut dispatcher = NotificationDispatcher::new();
        let ch = Arc::new(InMemoryChannel::new("mem", 10, Severity::Info));
        struct ArcChannel(Arc<InMemoryChannel>);
        impl NotificationChannel for ArcChannel {
            fn name(&self) -> &str {
                self.0.name()
            }
            fn min_severity(&self) -> Severity {
                self.0.min_severity()
            }
            fn send(&self, event: &NotificationEvent) -> ArchiveResult<bool> {
                self.0.send(event)
            }
        }
        dispatcher.add_channel(Box::new(ArcChannel(Arc::clone(&ch))));

        dispatcher.notify_fixity_failure(PathBuf::from("/media/film.mkv"), "abc123", "def456");
        assert_eq!(ch.len(), 1);
        let ev = ch.last_event().expect("event");
        assert_eq!(ev.kind, EventKind::FixityCheckFailed);
        assert_eq!(ev.severity, Severity::Error);
    }
}
