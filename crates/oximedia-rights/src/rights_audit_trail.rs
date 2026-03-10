#![allow(dead_code)]
//! Comprehensive audit trail for all rights changes.
//!
//! Tracks every modification to rights, licenses, and permissions
//! with immutable log entries, enabling compliance reporting and
//! forensic analysis of rights history.

use std::collections::HashMap;

/// Type of action recorded in the audit trail.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AuditAction {
    /// A new right was granted.
    Grant,
    /// An existing right was revoked.
    Revoke,
    /// A right was modified (e.g., territory change).
    Modify,
    /// A right was transferred to another party.
    Transfer,
    /// A right was renewed or extended.
    Renew,
    /// A right expired naturally.
    Expire,
    /// An access check was performed.
    AccessCheck,
    /// A license was created.
    LicenseCreate,
    /// A license was terminated.
    LicenseTerminate,
    /// A custom action type.
    Custom(String),
}

impl AuditAction {
    /// Return a human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            AuditAction::Grant => "grant",
            AuditAction::Revoke => "revoke",
            AuditAction::Modify => "modify",
            AuditAction::Transfer => "transfer",
            AuditAction::Renew => "renew",
            AuditAction::Expire => "expire",
            AuditAction::AccessCheck => "access_check",
            AuditAction::LicenseCreate => "license_create",
            AuditAction::LicenseTerminate => "license_terminate",
            AuditAction::Custom(name) => name.as_str(),
        }
    }

    /// Check if this is a destructive action (revoke, terminate, expire).
    #[must_use]
    pub fn is_destructive(&self) -> bool {
        matches!(
            self,
            AuditAction::Revoke | AuditAction::LicenseTerminate | AuditAction::Expire
        )
    }
}

/// Severity level for an audit entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AuditSeverity {
    /// Informational — routine operation.
    Info,
    /// Warning — unusual but permitted.
    Warning,
    /// Critical — compliance-relevant action.
    Critical,
}

impl AuditSeverity {
    /// Return a label for the severity.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            AuditSeverity::Info => "info",
            AuditSeverity::Warning => "warning",
            AuditSeverity::Critical => "critical",
        }
    }
}

/// A single immutable audit trail entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Unique entry identifier.
    pub entry_id: u64,
    /// Timestamp as ISO 8601 string.
    pub timestamp: String,
    /// The action performed.
    pub action: AuditAction,
    /// Severity of this entry.
    pub severity: AuditSeverity,
    /// User or system that performed the action.
    pub actor: String,
    /// The right or license ID affected.
    pub target_id: String,
    /// Human-readable description.
    pub description: String,
    /// Previous value (if applicable).
    pub old_value: String,
    /// New value (if applicable).
    pub new_value: String,
    /// IP address or system identifier of the actor.
    pub source_ip: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl AuditEntry {
    /// Create a new audit entry.
    #[must_use]
    pub fn new(entry_id: u64, action: AuditAction, actor: &str, target_id: &str) -> Self {
        Self {
            entry_id,
            timestamp: String::new(),
            action,
            severity: AuditSeverity::Info,
            actor: actor.to_string(),
            target_id: target_id.to_string(),
            description: String::new(),
            old_value: String::new(),
            new_value: String::new(),
            source_ip: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Set the severity.
    #[must_use]
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set old and new values for a modification.
    #[must_use]
    pub fn with_change(mut self, old: &str, new: &str) -> Self {
        self.old_value = old.to_string();
        self.new_value = new.to_string();
        self
    }

    /// Set source IP.
    #[must_use]
    pub fn with_source_ip(mut self, ip: &str) -> Self {
        self.source_ip = ip.to_string();
        self
    }

    /// Add metadata key-value pair.
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Check if this entry records a value change.
    #[must_use]
    pub fn has_change(&self) -> bool {
        !self.old_value.is_empty() || !self.new_value.is_empty()
    }
}

/// Immutable audit trail log.
pub struct AuditTrail {
    /// All entries in chronological order.
    entries: Vec<AuditEntry>,
    /// Next entry ID to assign.
    next_id: u64,
}

impl AuditTrail {
    /// Create a new empty audit trail.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
        }
    }

    /// Append a new entry to the trail. Returns the assigned entry ID.
    pub fn append(&mut self, mut entry: AuditEntry) -> u64 {
        let id = self.next_id;
        entry.entry_id = id;
        self.next_id += 1;
        self.entries.push(entry);
        id
    }

    /// Get all entries.
    #[must_use]
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Get the total number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the trail is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Find entries by target ID.
    #[must_use]
    pub fn find_by_target(&self, target_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.target_id == target_id)
            .collect()
    }

    /// Find entries by actor.
    #[must_use]
    pub fn find_by_actor(&self, actor: &str) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| e.actor == actor).collect()
    }

    /// Find entries by action type.
    #[must_use]
    pub fn find_by_action(&self, action: &AuditAction) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.action == action)
            .collect()
    }

    /// Find entries at or above a severity level.
    #[must_use]
    pub fn find_by_severity(&self, min_severity: AuditSeverity) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.severity >= min_severity)
            .collect()
    }

    /// Find all destructive actions.
    #[must_use]
    pub fn find_destructive(&self) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.action.is_destructive())
            .collect()
    }

    /// Get the last N entries.
    #[must_use]
    pub fn recent(&self, n: usize) -> &[AuditEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Count entries by action type.
    #[must_use]
    pub fn count_by_action(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            let key = entry.action.label().to_string();
            *counts.entry(key).or_insert(0) += 1;
        }
        counts
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

    #[test]
    fn test_audit_action_label() {
        assert_eq!(AuditAction::Grant.label(), "grant");
        assert_eq!(AuditAction::Revoke.label(), "revoke");
        assert_eq!(AuditAction::Custom("check".into()).label(), "check");
    }

    #[test]
    fn test_audit_action_is_destructive() {
        assert!(AuditAction::Revoke.is_destructive());
        assert!(AuditAction::LicenseTerminate.is_destructive());
        assert!(AuditAction::Expire.is_destructive());
        assert!(!AuditAction::Grant.is_destructive());
        assert!(!AuditAction::Modify.is_destructive());
    }

    #[test]
    fn test_audit_severity_ordering() {
        assert!(AuditSeverity::Critical > AuditSeverity::Warning);
        assert!(AuditSeverity::Warning > AuditSeverity::Info);
    }

    #[test]
    fn test_audit_severity_label() {
        assert_eq!(AuditSeverity::Info.label(), "info");
        assert_eq!(AuditSeverity::Warning.label(), "warning");
        assert_eq!(AuditSeverity::Critical.label(), "critical");
    }

    #[test]
    fn test_audit_entry_creation() {
        let entry = AuditEntry::new(1, AuditAction::Grant, "admin", "right-001")
            .with_timestamp("2024-01-15T10:30:00Z")
            .with_description("Granted broadcast rights")
            .with_source_ip("192.168.1.1")
            .with_severity(AuditSeverity::Info);

        assert_eq!(entry.actor, "admin");
        assert_eq!(entry.target_id, "right-001");
        assert!(!entry.has_change());
    }

    #[test]
    fn test_audit_entry_with_change() {
        let entry = AuditEntry::new(2, AuditAction::Modify, "editor", "right-002")
            .with_change("US", "US,CA");

        assert!(entry.has_change());
        assert_eq!(entry.old_value, "US");
        assert_eq!(entry.new_value, "US,CA");
    }

    #[test]
    fn test_audit_entry_metadata() {
        let entry = AuditEntry::new(3, AuditAction::Transfer, "sys", "right-003")
            .with_metadata("from_owner", "Alice")
            .with_metadata("to_owner", "Bob");

        assert_eq!(
            entry
                .metadata
                .get("from_owner")
                .expect("rights test operation should succeed"),
            "Alice"
        );
        assert_eq!(
            entry
                .metadata
                .get("to_owner")
                .expect("rights test operation should succeed"),
            "Bob"
        );
    }

    #[test]
    fn test_audit_trail_append() {
        let mut trail = AuditTrail::new();
        assert!(trail.is_empty());

        let id = trail.append(AuditEntry::new(0, AuditAction::Grant, "admin", "r1"));
        assert_eq!(id, 1);
        assert_eq!(trail.len(), 1);
    }

    #[test]
    fn test_audit_trail_auto_increment_id() {
        let mut trail = AuditTrail::new();
        let id1 = trail.append(AuditEntry::new(0, AuditAction::Grant, "a", "r1"));
        let id2 = trail.append(AuditEntry::new(0, AuditAction::Revoke, "a", "r1"));
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_audit_trail_find_by_target() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEntry::new(0, AuditAction::Grant, "a", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Grant, "b", "r2"));
        trail.append(AuditEntry::new(0, AuditAction::Modify, "a", "r1"));

        let r1_entries = trail.find_by_target("r1");
        assert_eq!(r1_entries.len(), 2);
    }

    #[test]
    fn test_audit_trail_find_by_actor() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEntry::new(0, AuditAction::Grant, "alice", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Grant, "bob", "r2"));

        let alice_entries = trail.find_by_actor("alice");
        assert_eq!(alice_entries.len(), 1);
    }

    #[test]
    fn test_audit_trail_find_by_action() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEntry::new(0, AuditAction::Grant, "a", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Revoke, "a", "r2"));
        trail.append(AuditEntry::new(0, AuditAction::Grant, "b", "r3"));

        let grants = trail.find_by_action(&AuditAction::Grant);
        assert_eq!(grants.len(), 2);
    }

    #[test]
    fn test_audit_trail_find_destructive() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEntry::new(0, AuditAction::Grant, "a", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Revoke, "a", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Expire, "sys", "r2"));

        let destructive = trail.find_destructive();
        assert_eq!(destructive.len(), 2);
    }

    #[test]
    fn test_audit_trail_find_by_severity() {
        let mut trail = AuditTrail::new();
        trail.append(
            AuditEntry::new(0, AuditAction::Grant, "a", "r1").with_severity(AuditSeverity::Info),
        );
        trail.append(
            AuditEntry::new(0, AuditAction::Revoke, "a", "r1")
                .with_severity(AuditSeverity::Critical),
        );

        let critical = trail.find_by_severity(AuditSeverity::Critical);
        assert_eq!(critical.len(), 1);
        let warning_plus = trail.find_by_severity(AuditSeverity::Warning);
        assert_eq!(warning_plus.len(), 1);
    }

    #[test]
    fn test_audit_trail_recent() {
        let mut trail = AuditTrail::new();
        for i in 0..10 {
            trail.append(AuditEntry::new(
                0,
                AuditAction::AccessCheck,
                "sys",
                &format!("r{i}"),
            ));
        }
        let recent = trail.recent(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_audit_trail_count_by_action() {
        let mut trail = AuditTrail::new();
        trail.append(AuditEntry::new(0, AuditAction::Grant, "a", "r1"));
        trail.append(AuditEntry::new(0, AuditAction::Grant, "b", "r2"));
        trail.append(AuditEntry::new(0, AuditAction::Revoke, "a", "r3"));

        let counts = trail.count_by_action();
        assert_eq!(
            *counts
                .get("grant")
                .expect("rights test operation should succeed"),
            2
        );
        assert_eq!(
            *counts
                .get("revoke")
                .expect("rights test operation should succeed"),
            1
        );
    }
}
