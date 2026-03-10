#![allow(dead_code)]
//! Immutable audit trail for broadcast automation compliance.
//!
//! Records every automation action, configuration change, and operator
//! interaction with full provenance (who, what, when, why). Designed for
//! regulatory compliance (FCC, Ofcom) and post-incident forensics.
//! Entries are append-only and integrity-protected with sequential hashing.

use std::collections::HashMap;
use std::fmt;

/// Category of audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditCategory {
    /// Automation command executed.
    CommandExecution,
    /// Configuration changed.
    ConfigChange,
    /// Operator login/logout.
    OperatorSession,
    /// Failover event.
    FailoverEvent,
    /// Emergency alert activity.
    EmergencyAlert,
    /// Device state change.
    DeviceStateChange,
    /// System health event.
    SystemHealth,
    /// Interlock triggered.
    InterlockTriggered,
}

impl fmt::Display for AuditCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandExecution => write!(f, "CommandExecution"),
            Self::ConfigChange => write!(f, "ConfigChange"),
            Self::OperatorSession => write!(f, "OperatorSession"),
            Self::FailoverEvent => write!(f, "FailoverEvent"),
            Self::EmergencyAlert => write!(f, "EmergencyAlert"),
            Self::DeviceStateChange => write!(f, "DeviceStateChange"),
            Self::SystemHealth => write!(f, "SystemHealth"),
            Self::InterlockTriggered => write!(f, "InterlockTriggered"),
        }
    }
}

/// Severity level for an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditSeverity {
    /// Informational.
    Info,
    /// Warning-level.
    Warning,
    /// Error-level.
    Error,
    /// Critical-level.
    Critical,
}

impl fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Outcome of the audited action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditOutcome {
    /// Action succeeded.
    Success,
    /// Action failed.
    Failure,
    /// Action was blocked by an interlock or policy.
    Blocked,
    /// Action was partially completed.
    Partial,
}

impl fmt::Display for AuditOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::Failure => write!(f, "FAILURE"),
            Self::Blocked => write!(f, "BLOCKED"),
            Self::Partial => write!(f, "PARTIAL"),
        }
    }
}

/// A single audit trail entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Timestamp in milliseconds since epoch.
    pub timestamp_ms: u64,
    /// Category of the event.
    pub category: AuditCategory,
    /// Severity level.
    pub severity: AuditSeverity,
    /// Outcome of the action.
    pub outcome: AuditOutcome,
    /// Operator or system principal that initiated the action.
    pub principal: String,
    /// Channel or scope this event relates to.
    pub scope: String,
    /// Human-readable description.
    pub description: String,
    /// Simple integrity hash linking to previous entry.
    pub chain_hash: u64,
    /// Key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl AuditEntry {
    /// Returns true if this entry represents a successful action.
    pub fn is_success(&self) -> bool {
        self.outcome == AuditOutcome::Success
    }

    /// Returns true if the severity is at least Warning.
    pub fn is_notable(&self) -> bool {
        self.severity >= AuditSeverity::Warning
    }
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{seq:06}] {ts}ms {sev} {cat} [{scope}] {principal}: {desc} => {outcome}",
            seq = self.sequence,
            ts = self.timestamp_ms,
            sev = self.severity,
            cat = self.category,
            scope = self.scope,
            principal = self.principal,
            desc = self.description,
            outcome = self.outcome,
        )
    }
}

/// Builder for constructing audit entries.
#[derive(Debug)]
pub struct AuditEntryBuilder {
    /// Category.
    category: AuditCategory,
    /// Severity.
    severity: AuditSeverity,
    /// Outcome.
    outcome: AuditOutcome,
    /// Principal.
    principal: String,
    /// Scope.
    scope: String,
    /// Description.
    description: String,
    /// Metadata.
    metadata: HashMap<String, String>,
}

impl AuditEntryBuilder {
    /// Start building an audit entry.
    pub fn new(category: AuditCategory) -> Self {
        Self {
            category,
            severity: AuditSeverity::Info,
            outcome: AuditOutcome::Success,
            principal: String::new(),
            scope: String::new(),
            description: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set severity.
    pub fn severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set outcome.
    pub fn outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Set principal (who).
    pub fn principal(mut self, principal: impl Into<String>) -> Self {
        self.principal = principal.into();
        self
    }

    /// Set scope (channel/device).
    pub fn scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = scope.into();
        self
    }

    /// Set description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Add a metadata key-value pair.
    pub fn meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build the entry (internal use: trail assigns sequence/hash).
    fn build(self, sequence: u64, timestamp_ms: u64, chain_hash: u64) -> AuditEntry {
        AuditEntry {
            sequence,
            timestamp_ms,
            category: self.category,
            severity: self.severity,
            outcome: self.outcome,
            principal: self.principal,
            scope: self.scope,
            description: self.description,
            chain_hash,
            metadata: self.metadata,
        }
    }
}

/// Simple hash combining function for chaining entries.
fn chain(prev_hash: u64, sequence: u64, timestamp_ms: u64) -> u64 {
    let mut h = prev_hash;
    h = h.wrapping_mul(6_364_136_223_846_793_005);
    h = h.wrapping_add(sequence);
    h = h.wrapping_mul(6_364_136_223_846_793_005);
    h = h.wrapping_add(timestamp_ms);
    h
}

/// Append-only audit trail with integrity chain.
#[derive(Debug)]
pub struct AuditTrail {
    /// All entries in order.
    entries: Vec<AuditEntry>,
    /// Current chain hash.
    current_hash: u64,
    /// Next sequence number.
    next_sequence: u64,
    /// Maximum number of entries to retain (0 = unlimited).
    max_entries: usize,
}

impl AuditTrail {
    /// Create a new audit trail.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            current_hash: 0,
            next_sequence: 1,
            max_entries: 0,
        }
    }

    /// Create a new audit trail with a maximum entry limit.
    pub fn with_max_entries(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            current_hash: 0,
            next_sequence: 1,
            max_entries,
        }
    }

    /// Record an audit entry from a builder.
    pub fn record(&mut self, builder: AuditEntryBuilder, timestamp_ms: u64) -> u64 {
        let seq = self.next_sequence;
        let new_hash = chain(self.current_hash, seq, timestamp_ms);
        let entry = builder.build(seq, timestamp_ms, new_hash);
        self.entries.push(entry);
        self.current_hash = new_hash;
        self.next_sequence += 1;

        // Enforce capacity
        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        seq
    }

    /// Get the total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the trail is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by sequence number.
    pub fn get_by_sequence(&self, seq: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.sequence == seq)
    }

    /// Get the most recent entry.
    pub fn latest(&self) -> Option<&AuditEntry> {
        self.entries.last()
    }

    /// Query entries by category.
    pub fn query_by_category(&self, category: AuditCategory) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }

    /// Query entries by severity at or above the given level.
    pub fn query_by_min_severity(&self, min_severity: AuditSeverity) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity >= min_severity)
            .collect()
    }

    /// Query entries in a time range.
    pub fn query_by_time_range(&self, start_ms: u64, end_ms: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Verify the integrity chain from the beginning.
    pub fn verify_integrity(&self) -> bool {
        let mut expected_hash: u64 = 0;
        for entry in &self.entries {
            expected_hash = chain(expected_hash, entry.sequence, entry.timestamp_ms);
            if entry.chain_hash != expected_hash {
                return false;
            }
        }
        true
    }

    /// Get the current chain hash.
    pub fn current_hash(&self) -> u64 {
        self.current_hash
    }

    /// Get all entries (read-only).
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Count entries matching a predicate.
    pub fn count_where<F: Fn(&AuditEntry) -> bool>(&self, predicate: F) -> usize {
        self.entries.iter().filter(|e| predicate(e)).count()
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_builder(cat: AuditCategory) -> AuditEntryBuilder {
        AuditEntryBuilder::new(cat)
            .principal("operator1")
            .scope("CH1")
            .description("test action")
    }

    #[test]
    fn test_audit_category_display() {
        assert_eq!(
            AuditCategory::CommandExecution.to_string(),
            "CommandExecution"
        );
        assert_eq!(AuditCategory::FailoverEvent.to_string(), "FailoverEvent");
        assert_eq!(AuditCategory::EmergencyAlert.to_string(), "EmergencyAlert");
    }

    #[test]
    fn test_audit_severity_display_and_ordering() {
        assert_eq!(AuditSeverity::Info.to_string(), "INFO");
        assert_eq!(AuditSeverity::Critical.to_string(), "CRITICAL");
        assert!(AuditSeverity::Info < AuditSeverity::Warning);
        assert!(AuditSeverity::Warning < AuditSeverity::Error);
        assert!(AuditSeverity::Error < AuditSeverity::Critical);
    }

    #[test]
    fn test_audit_outcome_display() {
        assert_eq!(AuditOutcome::Success.to_string(), "SUCCESS");
        assert_eq!(AuditOutcome::Blocked.to_string(), "BLOCKED");
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut trail = AuditTrail::new();
        let seq = trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        assert_eq!(seq, 1);
        assert_eq!(trail.len(), 1);
        let entry = trail
            .get_by_sequence(1)
            .expect("get_by_sequence should succeed");
        assert_eq!(entry.principal, "operator1");
    }

    #[test]
    fn test_latest_entry() {
        let mut trail = AuditTrail::new();
        trail.record(make_builder(AuditCategory::ConfigChange), 1000);
        trail.record(make_builder(AuditCategory::FailoverEvent), 2000);
        let latest = trail.latest().expect("latest should succeed");
        assert_eq!(latest.sequence, 2);
        assert_eq!(latest.timestamp_ms, 2000);
    }

    #[test]
    fn test_query_by_category() {
        let mut trail = AuditTrail::new();
        trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        trail.record(make_builder(AuditCategory::ConfigChange), 2000);
        trail.record(make_builder(AuditCategory::CommandExecution), 3000);
        let cmds = trail.query_by_category(AuditCategory::CommandExecution);
        assert_eq!(cmds.len(), 2);
    }

    #[test]
    fn test_query_by_severity() {
        let mut trail = AuditTrail::new();
        trail.record(
            make_builder(AuditCategory::CommandExecution).severity(AuditSeverity::Info),
            1000,
        );
        trail.record(
            make_builder(AuditCategory::FailoverEvent).severity(AuditSeverity::Error),
            2000,
        );
        trail.record(
            make_builder(AuditCategory::EmergencyAlert).severity(AuditSeverity::Critical),
            3000,
        );
        let notable = trail.query_by_min_severity(AuditSeverity::Error);
        assert_eq!(notable.len(), 2);
    }

    #[test]
    fn test_query_by_time_range() {
        let mut trail = AuditTrail::new();
        trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        trail.record(make_builder(AuditCategory::CommandExecution), 2000);
        trail.record(make_builder(AuditCategory::CommandExecution), 3000);
        let ranged = trail.query_by_time_range(1500, 2500);
        assert_eq!(ranged.len(), 1);
    }

    #[test]
    fn test_integrity_verification() {
        let mut trail = AuditTrail::new();
        trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        trail.record(make_builder(AuditCategory::ConfigChange), 2000);
        trail.record(make_builder(AuditCategory::FailoverEvent), 3000);
        assert!(trail.verify_integrity());
    }

    #[test]
    fn test_integrity_detects_tampering() {
        let mut trail = AuditTrail::new();
        trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        trail.record(make_builder(AuditCategory::ConfigChange), 2000);
        // Tamper with an entry
        if let Some(entry) = trail.entries.first_mut() {
            entry.chain_hash = 999_999;
        }
        assert!(!trail.verify_integrity());
    }

    #[test]
    fn test_max_entries_enforcement() {
        let mut trail = AuditTrail::with_max_entries(2);
        trail.record(make_builder(AuditCategory::CommandExecution), 1000);
        trail.record(make_builder(AuditCategory::CommandExecution), 2000);
        trail.record(make_builder(AuditCategory::CommandExecution), 3000);
        assert_eq!(trail.len(), 2);
        // Oldest entry should have been removed
        assert!(trail.get_by_sequence(1).is_none());
    }

    #[test]
    fn test_entry_builder_metadata() {
        let mut trail = AuditTrail::new();
        let builder = AuditEntryBuilder::new(AuditCategory::DeviceStateChange)
            .principal("system")
            .scope("VTR-01")
            .description("Device went offline")
            .severity(AuditSeverity::Warning)
            .outcome(AuditOutcome::Failure)
            .meta("device_type", "VTR")
            .meta("error_code", "E_TIMEOUT");
        trail.record(builder, 5000);
        let entry = trail.latest().expect("latest should succeed");
        assert_eq!(
            entry
                .metadata
                .get("device_type")
                .expect("get should succeed"),
            "VTR"
        );
        assert_eq!(
            entry
                .metadata
                .get("error_code")
                .expect("get should succeed"),
            "E_TIMEOUT"
        );
    }

    #[test]
    fn test_entry_is_success_and_notable() {
        let mut trail = AuditTrail::new();
        trail.record(
            make_builder(AuditCategory::CommandExecution)
                .severity(AuditSeverity::Info)
                .outcome(AuditOutcome::Success),
            1000,
        );
        trail.record(
            make_builder(AuditCategory::FailoverEvent)
                .severity(AuditSeverity::Error)
                .outcome(AuditOutcome::Failure),
            2000,
        );
        let entries = trail.entries();
        assert!(entries[0].is_success());
        assert!(!entries[0].is_notable());
        assert!(!entries[1].is_success());
        assert!(entries[1].is_notable());
    }

    #[test]
    fn test_count_where() {
        let mut trail = AuditTrail::new();
        trail.record(
            make_builder(AuditCategory::CommandExecution).outcome(AuditOutcome::Success),
            1000,
        );
        trail.record(
            make_builder(AuditCategory::CommandExecution).outcome(AuditOutcome::Failure),
            2000,
        );
        trail.record(
            make_builder(AuditCategory::CommandExecution).outcome(AuditOutcome::Success),
            3000,
        );
        assert_eq!(trail.count_where(super::AuditEntry::is_success), 2);
    }

    #[test]
    fn test_empty_trail() {
        let trail = AuditTrail::new();
        assert!(trail.is_empty());
        assert_eq!(trail.len(), 0);
        assert!(trail.latest().is_none());
        assert!(trail.verify_integrity());
    }
}
