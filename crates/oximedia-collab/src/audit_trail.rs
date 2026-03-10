//! Audit trail for collaborative sessions.
//!
//! Records every significant action taken within a session, providing
//! traceability for compliance, debugging, and forensic review.

#![allow(dead_code)]

use std::collections::HashMap;

/// The type of action that was performed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    // Read-like actions
    /// A user joined the session.
    UserJoined,
    /// A user left the session.
    UserLeft,
    /// A resource was viewed.
    Viewed { resource_id: String },

    // Write-like actions
    /// A clip or resource was created.
    Created { resource_id: String },
    /// A resource was modified.
    Modified { resource_id: String, field: String },
    /// A resource was deleted.
    Deleted { resource_id: String },
    /// A lock was acquired on a resource.
    LockAcquired { resource_id: String },
    /// A lock was released on a resource.
    LockReleased { resource_id: String },
    /// A comment was added.
    CommentAdded { comment_id: String },
    /// An approval decision was made.
    ApprovalDecision { decision: String },
    /// An invite link was created.
    InviteCreated { token: String },
    /// An invite link was revoked.
    InviteRevoked { token: String },
}

impl AuditAction {
    /// Returns `true` if this action represents a write / mutation.
    #[must_use]
    pub fn is_write(&self) -> bool {
        !matches!(
            self,
            Self::UserJoined | Self::UserLeft | Self::Viewed { .. }
        )
    }

    /// Returns a short, human-readable label for the action.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::UserJoined => "user_joined",
            Self::UserLeft => "user_left",
            Self::Viewed { .. } => "viewed",
            Self::Created { .. } => "created",
            Self::Modified { .. } => "modified",
            Self::Deleted { .. } => "deleted",
            Self::LockAcquired { .. } => "lock_acquired",
            Self::LockReleased { .. } => "lock_released",
            Self::CommentAdded { .. } => "comment_added",
            Self::ApprovalDecision { .. } => "approval_decision",
            Self::InviteCreated { .. } => "invite_created",
            Self::InviteRevoked { .. } => "invite_revoked",
        }
    }
}

/// A single entry in the audit trail.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Monotonically increasing sequence number within the trail.
    pub seq: u64,
    /// Unix timestamp (seconds) when the action occurred.
    pub timestamp: u64,
    /// ID of the user who performed the action.
    pub user_id: String,
    /// Display name of the user (snapshot at time of action).
    pub user_name: String,
    /// The action that was performed.
    pub action: AuditAction,
    /// Optional freeform notes attached to this entry.
    pub notes: Option<String>,
}

impl AuditEntry {
    /// Creates a new entry.
    #[must_use]
    pub fn new(
        seq: u64,
        timestamp: u64,
        user_id: String,
        user_name: String,
        action: AuditAction,
    ) -> Self {
        Self {
            seq,
            timestamp,
            user_id,
            user_name,
            action,
            notes: None,
        }
    }

    /// Attaches notes to this entry.
    #[must_use]
    pub fn with_notes(mut self, notes: String) -> Self {
        self.notes = Some(notes);
        self
    }

    /// Returns a human-readable description of this entry.
    #[must_use]
    pub fn description(&self) -> String {
        format!(
            "[{}] seq={} user={} action={}",
            self.timestamp,
            self.seq,
            self.user_name,
            self.action.label(),
        )
    }
}

/// An append-only audit trail for a collaboration session.
#[derive(Debug, Default)]
pub struct AuditTrail {
    entries: Vec<AuditEntry>,
    next_seq: u64,
}

impl AuditTrail {
    /// Creates an empty trail.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a new action.  Assigns the next sequence number and returns it.
    pub fn record(
        &mut self,
        timestamp: u64,
        user_id: String,
        user_name: String,
        action: AuditAction,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries
            .push(AuditEntry::new(seq, timestamp, user_id, user_name, action));
        seq
    }

    /// Returns all entries as a slice (oldest first).
    #[must_use]
    pub fn all_entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Returns all entries authored by `user_id`.
    #[must_use]
    pub fn entries_by_user(&self, user_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.user_id == user_id)
            .collect()
    }

    /// Returns entries where `action.is_write()` is `true`, limited to the
    /// most recent `limit` entries (or all if `limit == 0`).
    #[must_use]
    pub fn recent_writes(&self, limit: usize) -> Vec<&AuditEntry> {
        let writes: Vec<&AuditEntry> = self
            .entries
            .iter()
            .filter(|e| e.action.is_write())
            .collect();
        if limit == 0 || writes.len() <= limit {
            writes
        } else {
            writes[writes.len() - limit..].to_vec()
        }
    }

    /// Returns the total number of recorded entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no entries have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns entries within the inclusive timestamp range `[from, to]`.
    #[must_use]
    pub fn entries_in_range(&self, from: u64, to: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect()
    }

    /// Returns a per-user summary: map from user_id to action count.
    #[must_use]
    pub fn user_action_counts(&self) -> HashMap<&str, usize> {
        let mut map: HashMap<&str, usize> = HashMap::new();
        for e in &self.entries {
            *map.entry(e.user_id.as_str()).or_insert(0) += 1;
        }
        map
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trail() -> AuditTrail {
        let mut trail = AuditTrail::new();
        trail.record(10, "u1".into(), "Alice".into(), AuditAction::UserJoined);
        trail.record(
            20,
            "u1".into(),
            "Alice".into(),
            AuditAction::Created {
                resource_id: "c1".into(),
            },
        );
        trail.record(30, "u2".into(), "Bob".into(), AuditAction::UserJoined);
        trail.record(
            40,
            "u2".into(),
            "Bob".into(),
            AuditAction::Modified {
                resource_id: "c1".into(),
                field: "duration".into(),
            },
        );
        trail.record(
            50,
            "u1".into(),
            "Alice".into(),
            AuditAction::Deleted {
                resource_id: "c2".into(),
            },
        );
        trail
    }

    #[test]
    fn audit_action_is_write_for_mutations() {
        assert!(AuditAction::Created {
            resource_id: "x".into()
        }
        .is_write());
        assert!(AuditAction::Modified {
            resource_id: "x".into(),
            field: "f".into()
        }
        .is_write());
        assert!(AuditAction::Deleted {
            resource_id: "x".into()
        }
        .is_write());
    }

    #[test]
    fn audit_action_is_not_write_for_reads() {
        assert!(!AuditAction::UserJoined.is_write());
        assert!(!AuditAction::UserLeft.is_write());
        assert!(!AuditAction::Viewed {
            resource_id: "x".into()
        }
        .is_write());
    }

    #[test]
    fn audit_action_labels() {
        assert_eq!(AuditAction::UserJoined.label(), "user_joined");
        assert_eq!(
            AuditAction::Deleted {
                resource_id: "x".into()
            }
            .label(),
            "deleted"
        );
        assert_eq!(
            AuditAction::CommentAdded {
                comment_id: "c".into()
            }
            .label(),
            "comment_added"
        );
    }

    #[test]
    fn audit_entry_description_contains_user() {
        let entry = AuditEntry::new(0, 100, "u1".into(), "Alice".into(), AuditAction::UserJoined);
        assert!(entry.description().contains("Alice"));
    }

    #[test]
    fn audit_entry_with_notes() {
        let entry = AuditEntry::new(0, 100, "u1".into(), "Alice".into(), AuditAction::UserJoined)
            .with_notes("first join".into());
        assert_eq!(
            entry.notes.expect("collab test operation should succeed"),
            "first join"
        );
    }

    #[test]
    fn trail_len_after_records() {
        let trail = make_trail();
        assert_eq!(trail.len(), 5);
    }

    #[test]
    fn trail_is_empty_initially() {
        let trail = AuditTrail::new();
        assert!(trail.is_empty());
    }

    #[test]
    fn trail_sequence_numbers_monotonic() {
        let trail = make_trail();
        let seqs: Vec<u64> = trail.all_entries().iter().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn trail_entries_by_user() {
        let trail = make_trail();
        let alice_entries = trail.entries_by_user("u1");
        assert_eq!(alice_entries.len(), 3);
        assert!(alice_entries.iter().all(|e| e.user_id == "u1"));
    }

    #[test]
    fn trail_recent_writes_limit() {
        let trail = make_trail();
        let writes = trail.recent_writes(1);
        assert_eq!(writes.len(), 1);
        // Should be the last write entry
        assert_eq!(writes[0].action.label(), "deleted");
    }

    #[test]
    fn trail_recent_writes_all_when_limit_zero() {
        let trail = make_trail();
        let writes = trail.recent_writes(0);
        // created + modified + deleted = 3 writes
        assert_eq!(writes.len(), 3);
    }

    #[test]
    fn trail_entries_in_range() {
        let trail = make_trail();
        let in_range = trail.entries_in_range(20, 40);
        assert_eq!(in_range.len(), 3); // seq 1,2,3 (t=20,30,40)
    }

    #[test]
    fn trail_user_action_counts() {
        let trail = make_trail();
        let counts = trail.user_action_counts();
        assert_eq!(
            *counts
                .get("u1")
                .expect("collab test operation should succeed"),
            3
        );
        assert_eq!(
            *counts
                .get("u2")
                .expect("collab test operation should succeed"),
            2
        );
    }
}
