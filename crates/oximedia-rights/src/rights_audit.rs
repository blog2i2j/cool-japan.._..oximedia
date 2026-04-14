//! Rights audit log.
//!
//! [`RightsAuditLog`] records timestamped actions performed on rights records
//! and allows recent-entry retrieval.

/// A single entry in the rights audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Freeform description of the action performed.
    pub action: String,
    /// ID of the rights record that was acted upon.
    pub rights_id: u64,
    /// ID of the actor (user/system) that performed the action.
    pub actor: u64,
    /// Unix timestamp in seconds when the action occurred.
    pub ts: u64,
}

impl AuditEntry {
    /// Create a new audit entry.
    pub fn new(action: impl Into<String>, rights_id: u64, actor: u64, ts: u64) -> Self {
        Self {
            action: action.into(),
            rights_id,
            actor,
            ts,
        }
    }
}

/// Append-only audit log for rights management operations.
///
/// # Example
/// ```
/// use oximedia_rights::rights_audit::RightsAuditLog;
///
/// let mut log = RightsAuditLog::new();
/// log.record("created", 42, 1, 1_700_000_000);
/// log.record("approved", 42, 2, 1_700_000_060);
/// assert_eq!(log.recent(1)[0].action, "approved");
/// ```
#[derive(Debug, Default)]
pub struct RightsAuditLog {
    entries: Vec<AuditEntry>,
}

impl RightsAuditLog {
    /// Create an empty audit log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new action in the log.
    pub fn record(&mut self, action: &str, rights_id: u64, actor: u64, ts: u64) {
        self.entries
            .push(AuditEntry::new(action, rights_id, actor, ts));
    }

    /// Return the `n` most-recently-recorded entries (latest first).
    ///
    /// If `n` is larger than the total number of entries all entries are
    /// returned.
    pub fn recent(&self, n: usize) -> Vec<&AuditEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Total number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// All entries for a specific rights record, in insertion order.
    pub fn for_rights(&self, rights_id: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.rights_id == rights_id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_log_is_empty() {
        let log = RightsAuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_record_and_recent() {
        let mut log = RightsAuditLog::new();
        log.record("created", 1, 10, 100);
        log.record("updated", 1, 10, 200);
        log.record("approved", 1, 20, 300);

        let recent = log.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].action, "approved");
        assert_eq!(recent[1].action, "updated");
    }

    #[test]
    fn test_recent_more_than_len() {
        let mut log = RightsAuditLog::new();
        log.record("a", 1, 1, 1);
        let recent = log.recent(100);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_for_rights_filter() {
        let mut log = RightsAuditLog::new();
        log.record("created", 1, 1, 10);
        log.record("created", 2, 1, 20);
        log.record("updated", 1, 2, 30);

        let entries = log.for_rights(1);
        assert_eq!(entries.len(), 2);
        assert!(entries.iter().all(|e| e.rights_id == 1));
    }

    #[test]
    fn test_record_fields() {
        let mut log = RightsAuditLog::new();
        log.record("revoked", 99, 42, 1_234_567);
        let entry = &log.recent(1)[0];
        assert_eq!(entry.action, "revoked");
        assert_eq!(entry.rights_id, 99);
        assert_eq!(entry.actor, 42);
        assert_eq!(entry.ts, 1_234_567);
    }
}
