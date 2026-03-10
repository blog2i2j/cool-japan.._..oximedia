#![allow(dead_code)]
//! Audit trail tracking for archive preservation operations.
//!
//! Provides immutable, append-only logging of every operation performed on archived
//! media assets. Supports chain-of-custody verification and compliance reporting.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// Unique identifier for an audit event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AuditEventId(String);

impl AuditEventId {
    /// Creates a new audit event identifier from the given string.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the inner identifier string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Categories of auditable operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditAction {
    /// Asset was ingested into the archive.
    Ingest,
    /// Asset metadata was modified.
    MetadataUpdate,
    /// Asset was accessed or read.
    Access,
    /// Asset was exported or downloaded.
    Export,
    /// Asset was migrated to a new format.
    FormatMigration,
    /// Fixity check was performed.
    FixityCheck,
    /// Asset was replicated to another location.
    Replication,
    /// Asset was deleted.
    Deletion,
    /// Retention policy was applied.
    RetentionApplied,
    /// Integrity verification was performed.
    IntegrityVerify,
}

impl AuditAction {
    /// Returns a human-readable label for this action.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Ingest => "ingest",
            Self::MetadataUpdate => "metadata_update",
            Self::Access => "access",
            Self::Export => "export",
            Self::FormatMigration => "format_migration",
            Self::FixityCheck => "fixity_check",
            Self::Replication => "replication",
            Self::Deletion => "deletion",
            Self::RetentionApplied => "retention_applied",
            Self::IntegrityVerify => "integrity_verify",
        }
    }
}

/// Severity level for an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AuditSeverity {
    /// Informational event.
    Info,
    /// Warning event.
    Warning,
    /// Error event.
    Error,
    /// Critical event requiring immediate attention.
    Critical,
}

impl AuditSeverity {
    /// Returns the numeric priority (higher is more severe).
    #[must_use]
    pub const fn priority(&self) -> u8 {
        match self {
            Self::Info => 0,
            Self::Warning => 1,
            Self::Error => 2,
            Self::Critical => 3,
        }
    }
}

/// Outcome of an audited operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditOutcome {
    /// Operation completed successfully.
    Success,
    /// Operation failed.
    Failure,
    /// Operation was partially completed.
    Partial,
    /// Operation was denied due to policy.
    Denied,
}

/// A single audit event in the trail.
#[derive(Debug, Clone)]
pub struct AuditEvent {
    /// Unique event identifier.
    pub id: AuditEventId,
    /// Timestamp when the event occurred.
    pub timestamp: SystemTime,
    /// The action that was performed.
    pub action: AuditAction,
    /// The severity of this event.
    pub severity: AuditSeverity,
    /// The outcome of the operation.
    pub outcome: AuditOutcome,
    /// The user or system actor that performed the action.
    pub actor: String,
    /// The asset identifier that was affected.
    pub asset_id: String,
    /// Free-form description of the event.
    pub description: String,
    /// Additional key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl AuditEvent {
    /// Creates a new audit event with the given parameters.
    #[must_use]
    pub fn new(
        id: AuditEventId,
        action: AuditAction,
        actor: impl Into<String>,
        asset_id: impl Into<String>,
    ) -> Self {
        Self {
            id,
            timestamp: SystemTime::now(),
            action,
            severity: AuditSeverity::Info,
            outcome: AuditOutcome::Success,
            actor: actor.into(),
            asset_id: asset_id.into(),
            description: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Sets the severity for this event.
    #[must_use]
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Sets the outcome for this event.
    #[must_use]
    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Sets a description for this event.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Adds a metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns whether this event represents a failure.
    #[must_use]
    pub fn is_failure(&self) -> bool {
        matches!(self.outcome, AuditOutcome::Failure | AuditOutcome::Denied)
    }
}

/// Query filter for retrieving audit events.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by action type.
    pub action: Option<AuditAction>,
    /// Filter by actor.
    pub actor: Option<String>,
    /// Filter by asset identifier.
    pub asset_id: Option<String>,
    /// Filter by minimum severity.
    pub min_severity: Option<AuditSeverity>,
    /// Filter events after this time.
    pub after: Option<SystemTime>,
    /// Filter events before this time.
    pub before: Option<SystemTime>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl AuditQuery {
    /// Creates a new empty query.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filters by action type.
    #[must_use]
    pub fn with_action(mut self, action: AuditAction) -> Self {
        self.action = Some(action);
        self
    }

    /// Filters by actor.
    #[must_use]
    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }

    /// Filters by asset identifier.
    #[must_use]
    pub fn with_asset(mut self, asset_id: impl Into<String>) -> Self {
        self.asset_id = Some(asset_id.into());
        self
    }

    /// Filters by minimum severity.
    #[must_use]
    pub fn with_min_severity(mut self, severity: AuditSeverity) -> Self {
        self.min_severity = Some(severity);
        self
    }

    /// Sets a result limit.
    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Returns whether a given event matches this query.
    #[must_use]
    pub fn matches(&self, event: &AuditEvent) -> bool {
        if let Some(action) = self.action {
            if event.action != action {
                return false;
            }
        }
        if let Some(ref actor) = self.actor {
            if event.actor != *actor {
                return false;
            }
        }
        if let Some(ref asset_id) = self.asset_id {
            if event.asset_id != *asset_id {
                return false;
            }
        }
        if let Some(min_sev) = self.min_severity {
            if event.severity < min_sev {
                return false;
            }
        }
        if let Some(after) = self.after {
            if event.timestamp < after {
                return false;
            }
        }
        if let Some(before) = self.before {
            if event.timestamp > before {
                return false;
            }
        }
        true
    }
}

/// Summary statistics for a set of audit events.
#[derive(Debug, Clone, Default)]
pub struct AuditSummary {
    /// Total number of events.
    pub total_events: usize,
    /// Number of successful operations.
    pub successes: usize,
    /// Number of failures.
    pub failures: usize,
    /// Number of denied operations.
    pub denials: usize,
    /// Counts per action type.
    pub action_counts: HashMap<&'static str, usize>,
}

/// Append-only audit trail log.
#[derive(Debug, Default)]
pub struct AuditTrail {
    /// Sequential log of events.
    events: Vec<AuditEvent>,
    /// Next event sequence number.
    next_seq: u64,
}

impl AuditTrail {
    /// Creates a new empty audit trail.
    #[must_use]
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_seq: 1,
        }
    }

    /// Records a new audit event and returns its sequence number.
    pub fn record(&mut self, event: AuditEvent) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.events.push(event);
        seq
    }

    /// Returns the total number of recorded events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns whether the trail is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Queries the trail with the given filter, returning matching events.
    #[must_use]
    pub fn query(&self, q: &AuditQuery) -> Vec<&AuditEvent> {
        let mut results: Vec<&AuditEvent> = self.events.iter().filter(|e| q.matches(e)).collect();
        if let Some(limit) = q.limit {
            results.truncate(limit);
        }
        results
    }

    /// Returns a summary of all events in the trail.
    #[must_use]
    pub fn summarize(&self) -> AuditSummary {
        let mut summary = AuditSummary {
            total_events: self.events.len(),
            ..Default::default()
        };
        for event in &self.events {
            match event.outcome {
                AuditOutcome::Success => summary.successes += 1,
                AuditOutcome::Failure => summary.failures += 1,
                AuditOutcome::Denied => summary.denials += 1,
                AuditOutcome::Partial => {}
            }
            *summary
                .action_counts
                .entry(event.action.label())
                .or_insert(0) += 1;
        }
        summary
    }

    /// Returns the most recent event, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&AuditEvent> {
        self.events.last()
    }

    /// Returns all events for a specific asset.
    #[must_use]
    pub fn events_for_asset(&self, asset_id: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.asset_id == asset_id)
            .collect()
    }

    /// Returns the elapsed time since the last event, or `None` if empty.
    #[must_use]
    pub fn time_since_last_event(&self) -> Option<Duration> {
        self.events.last().and_then(|e| e.timestamp.elapsed().ok())
    }

    /// Returns the count of events with a given severity or above.
    #[must_use]
    pub fn count_at_severity(&self, min_severity: AuditSeverity) -> usize {
        self.events
            .iter()
            .filter(|e| e.severity >= min_severity)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(action: AuditAction, actor: &str, asset: &str) -> AuditEvent {
        AuditEvent::new(
            AuditEventId::new(format!("evt-{actor}-{asset}")),
            action,
            actor,
            asset,
        )
    }

    #[test]
    fn test_audit_event_id() {
        let id = AuditEventId::new("test-123");
        assert_eq!(id.as_str(), "test-123");
    }

    #[test]
    fn test_action_labels() {
        assert_eq!(AuditAction::Ingest.label(), "ingest");
        assert_eq!(AuditAction::Export.label(), "export");
        assert_eq!(AuditAction::Deletion.label(), "deletion");
        assert_eq!(AuditAction::FixityCheck.label(), "fixity_check");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(AuditSeverity::Info < AuditSeverity::Warning);
        assert!(AuditSeverity::Warning < AuditSeverity::Error);
        assert!(AuditSeverity::Error < AuditSeverity::Critical);
    }

    #[test]
    fn test_severity_priority() {
        assert_eq!(AuditSeverity::Info.priority(), 0);
        assert_eq!(AuditSeverity::Critical.priority(), 3);
    }

    #[test]
    fn test_event_builder() {
        let event = make_event(AuditAction::Ingest, "alice", "asset-001")
            .with_severity(AuditSeverity::Warning)
            .with_outcome(AuditOutcome::Failure)
            .with_description("failed checksum")
            .with_metadata("size", "1024");

        assert_eq!(event.severity, AuditSeverity::Warning);
        assert_eq!(event.outcome, AuditOutcome::Failure);
        assert!(event.is_failure());
        assert_eq!(event.description, "failed checksum");
        assert_eq!(
            event
                .metadata
                .get("size")
                .expect("operation should succeed"),
            "1024"
        );
    }

    #[test]
    fn test_event_is_failure() {
        let success = make_event(AuditAction::Access, "bob", "a1");
        assert!(!success.is_failure());

        let denied =
            make_event(AuditAction::Access, "bob", "a1").with_outcome(AuditOutcome::Denied);
        assert!(denied.is_failure());
    }

    #[test]
    fn test_trail_record_and_len() {
        let mut trail = AuditTrail::new();
        assert!(trail.is_empty());

        let seq = trail.record(make_event(AuditAction::Ingest, "alice", "a1"));
        assert_eq!(seq, 1);
        assert_eq!(trail.len(), 1);

        let seq2 = trail.record(make_event(AuditAction::Access, "bob", "a2"));
        assert_eq!(seq2, 2);
        assert_eq!(trail.len(), 2);
    }

    #[test]
    fn test_trail_latest() {
        let mut trail = AuditTrail::new();
        assert!(trail.latest().is_none());

        trail.record(make_event(AuditAction::Ingest, "alice", "a1"));
        trail.record(make_event(AuditAction::Export, "bob", "a2"));

        let latest = trail.latest().expect("operation should succeed");
        assert_eq!(latest.action, AuditAction::Export);
        assert_eq!(latest.actor, "bob");
    }

    #[test]
    fn test_trail_events_for_asset() {
        let mut trail = AuditTrail::new();
        trail.record(make_event(AuditAction::Ingest, "alice", "a1"));
        trail.record(make_event(AuditAction::Access, "bob", "a2"));
        trail.record(make_event(AuditAction::Export, "carol", "a1"));

        let a1_events = trail.events_for_asset("a1");
        assert_eq!(a1_events.len(), 2);
        assert_eq!(a1_events[0].action, AuditAction::Ingest);
        assert_eq!(a1_events[1].action, AuditAction::Export);
    }

    #[test]
    fn test_query_by_action() {
        let mut trail = AuditTrail::new();
        trail.record(make_event(AuditAction::Ingest, "alice", "a1"));
        trail.record(make_event(AuditAction::Access, "bob", "a2"));
        trail.record(make_event(AuditAction::Ingest, "carol", "a3"));

        let q = AuditQuery::new().with_action(AuditAction::Ingest);
        let results = trail.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_actor() {
        let mut trail = AuditTrail::new();
        trail.record(make_event(AuditAction::Ingest, "alice", "a1"));
        trail.record(make_event(AuditAction::Access, "alice", "a2"));
        trail.record(make_event(AuditAction::Ingest, "bob", "a3"));

        let q = AuditQuery::new().with_actor("alice");
        let results = trail.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_with_limit() {
        let mut trail = AuditTrail::new();
        for i in 0..10 {
            trail.record(make_event(AuditAction::Access, "user", &format!("a{i}")));
        }

        let q = AuditQuery::new().with_limit(3);
        let results = trail.query(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_summarize() {
        let mut trail = AuditTrail::new();
        trail.record(make_event(AuditAction::Ingest, "a", "a1"));
        trail
            .record(make_event(AuditAction::Access, "b", "a2").with_outcome(AuditOutcome::Failure));
        trail.record(make_event(AuditAction::Export, "c", "a3").with_outcome(AuditOutcome::Denied));
        trail.record(make_event(AuditAction::Ingest, "d", "a4"));

        let summary = trail.summarize();
        assert_eq!(summary.total_events, 4);
        assert_eq!(summary.successes, 2);
        assert_eq!(summary.failures, 1);
        assert_eq!(summary.denials, 1);
        assert_eq!(summary.action_counts["ingest"], 2);
    }

    #[test]
    fn test_count_at_severity() {
        let mut trail = AuditTrail::new();
        trail.record(make_event(AuditAction::Ingest, "a", "a1"));
        trail.record(
            make_event(AuditAction::Access, "b", "a2").with_severity(AuditSeverity::Warning),
        );
        trail.record(
            make_event(AuditAction::Export, "c", "a3").with_severity(AuditSeverity::Critical),
        );

        assert_eq!(trail.count_at_severity(AuditSeverity::Info), 3);
        assert_eq!(trail.count_at_severity(AuditSeverity::Warning), 2);
        assert_eq!(trail.count_at_severity(AuditSeverity::Error), 1);
        assert_eq!(trail.count_at_severity(AuditSeverity::Critical), 1);
    }
}
