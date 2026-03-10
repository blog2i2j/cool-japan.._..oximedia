#![allow(dead_code)]
//! Storage access logging and audit trail.
//!
//! Records every read, write, delete, and list operation for auditing,
//! billing analysis, and anomaly detection.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AccessKind
// ---------------------------------------------------------------------------

/// The kind of storage operation that was performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessKind {
    /// Object read / download.
    Read,
    /// Object write / upload.
    Write,
    /// Object delete.
    Delete,
    /// List objects.
    List,
    /// Copy object.
    Copy,
    /// Head / metadata request.
    Head,
}

impl std::fmt::Display for AccessKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "READ"),
            Self::Write => write!(f, "WRITE"),
            Self::Delete => write!(f, "DELETE"),
            Self::List => write!(f, "LIST"),
            Self::Copy => write!(f, "COPY"),
            Self::Head => write!(f, "HEAD"),
        }
    }
}

// ---------------------------------------------------------------------------
// AccessStatus
// ---------------------------------------------------------------------------

/// Outcome of a storage operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessStatus {
    /// Operation succeeded.
    Success,
    /// Operation was denied (authorization / quota).
    Denied,
    /// The target object was not found.
    NotFound,
    /// An internal error occurred.
    Error,
}

impl std::fmt::Display for AccessStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "SUCCESS"),
            Self::Denied => write!(f, "DENIED"),
            Self::NotFound => write!(f, "NOT_FOUND"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ---------------------------------------------------------------------------
// AccessEntry
// ---------------------------------------------------------------------------

/// A single access log entry.
#[derive(Debug, Clone)]
pub struct AccessEntry {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Epoch timestamp (seconds).
    pub timestamp_epoch: u64,
    /// Kind of operation.
    pub kind: AccessKind,
    /// Object key involved (empty for LIST).
    pub key: String,
    /// Identity of the caller (user, service, etc.).
    pub caller: String,
    /// Outcome status.
    pub status: AccessStatus,
    /// Bytes transferred (0 when not applicable).
    pub bytes_transferred: u64,
    /// Duration of the operation in microseconds.
    pub duration_us: u64,
}

impl AccessEntry {
    /// Create a new access entry with all fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        seq: u64,
        timestamp_epoch: u64,
        kind: AccessKind,
        key: impl Into<String>,
        caller: impl Into<String>,
        status: AccessStatus,
        bytes_transferred: u64,
        duration_us: u64,
    ) -> Self {
        Self {
            seq,
            timestamp_epoch,
            kind,
            key: key.into(),
            caller: caller.into(),
            status,
            bytes_transferred,
            duration_us,
        }
    }
}

// ---------------------------------------------------------------------------
// AccessLog
// ---------------------------------------------------------------------------

/// In-memory access log with query helpers.
#[derive(Debug, Default)]
pub struct AccessLog {
    /// All entries in insertion order.
    entries: Vec<AccessEntry>,
    /// Next sequence number.
    next_seq: u64,
}

impl AccessLog {
    /// Create an empty access log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a new entry and return its sequence number.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        timestamp_epoch: u64,
        kind: AccessKind,
        key: impl Into<String>,
        caller: impl Into<String>,
        status: AccessStatus,
        bytes_transferred: u64,
        duration_us: u64,
    ) -> u64 {
        let seq = self.next_seq;
        self.entries.push(AccessEntry::new(
            seq,
            timestamp_epoch,
            kind,
            key,
            caller,
            status,
            bytes_transferred,
            duration_us,
        ));
        self.next_seq += 1;
        seq
    }

    /// Number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by sequence number.
    pub fn get(&self, seq: u64) -> Option<&AccessEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Filter entries by kind.
    pub fn by_kind(&self, kind: AccessKind) -> Vec<&AccessEntry> {
        self.entries.iter().filter(|e| e.kind == kind).collect()
    }

    /// Filter entries by caller.
    pub fn by_caller(&self, caller: &str) -> Vec<&AccessEntry> {
        self.entries.iter().filter(|e| e.caller == caller).collect()
    }

    /// Filter entries by status.
    pub fn by_status(&self, status: AccessStatus) -> Vec<&AccessEntry> {
        self.entries.iter().filter(|e| e.status == status).collect()
    }

    /// Total bytes transferred across all entries.
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.bytes_transferred).sum()
    }

    /// Count entries per kind.
    pub fn counts_by_kind(&self) -> HashMap<AccessKind, usize> {
        let mut counts = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.kind).or_insert(0) += 1;
        }
        counts
    }

    /// Entries in a timestamp range (inclusive).
    pub fn in_range(&self, start_epoch: u64, end_epoch: u64) -> Vec<&AccessEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_epoch >= start_epoch && e.timestamp_epoch <= end_epoch)
            .collect()
    }

    /// Clear all log entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.next_seq = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_kind_display() {
        assert_eq!(AccessKind::Read.to_string(), "READ");
        assert_eq!(AccessKind::Write.to_string(), "WRITE");
        assert_eq!(AccessKind::Delete.to_string(), "DELETE");
        assert_eq!(AccessKind::List.to_string(), "LIST");
        assert_eq!(AccessKind::Copy.to_string(), "COPY");
        assert_eq!(AccessKind::Head.to_string(), "HEAD");
    }

    #[test]
    fn test_access_status_display() {
        assert_eq!(AccessStatus::Success.to_string(), "SUCCESS");
        assert_eq!(AccessStatus::Denied.to_string(), "DENIED");
        assert_eq!(AccessStatus::NotFound.to_string(), "NOT_FOUND");
        assert_eq!(AccessStatus::Error.to_string(), "ERROR");
    }

    #[test]
    fn test_log_record_and_len() {
        let mut log = AccessLog::new();
        assert!(log.is_empty());
        let seq = log.record(
            1000,
            AccessKind::Read,
            "obj/a",
            "user1",
            AccessStatus::Success,
            512,
            50,
        );
        assert_eq!(seq, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_log_get() {
        let mut log = AccessLog::new();
        log.record(
            1000,
            AccessKind::Write,
            "k1",
            "u",
            AccessStatus::Success,
            100,
            10,
        );
        let entry = log.get(0).expect("get should succeed");
        assert_eq!(entry.kind, AccessKind::Write);
        assert_eq!(entry.key, "k1");
        assert!(log.get(99).is_none());
    }

    #[test]
    fn test_log_by_kind() {
        let mut log = AccessLog::new();
        log.record(1, AccessKind::Read, "a", "u", AccessStatus::Success, 0, 0);
        log.record(2, AccessKind::Write, "b", "u", AccessStatus::Success, 0, 0);
        log.record(3, AccessKind::Read, "c", "u", AccessStatus::Success, 0, 0);
        assert_eq!(log.by_kind(AccessKind::Read).len(), 2);
        assert_eq!(log.by_kind(AccessKind::Write).len(), 1);
        assert_eq!(log.by_kind(AccessKind::Delete).len(), 0);
    }

    #[test]
    fn test_log_by_caller() {
        let mut log = AccessLog::new();
        log.record(
            1,
            AccessKind::Read,
            "a",
            "alice",
            AccessStatus::Success,
            0,
            0,
        );
        log.record(2, AccessKind::Read, "b", "bob", AccessStatus::Success, 0, 0);
        assert_eq!(log.by_caller("alice").len(), 1);
        assert_eq!(log.by_caller("charlie").len(), 0);
    }

    #[test]
    fn test_log_by_status() {
        let mut log = AccessLog::new();
        log.record(1, AccessKind::Read, "a", "u", AccessStatus::Success, 0, 0);
        log.record(2, AccessKind::Read, "b", "u", AccessStatus::Denied, 0, 0);
        assert_eq!(log.by_status(AccessStatus::Success).len(), 1);
        assert_eq!(log.by_status(AccessStatus::Denied).len(), 1);
    }

    #[test]
    fn test_log_total_bytes() {
        let mut log = AccessLog::new();
        log.record(1, AccessKind::Read, "a", "u", AccessStatus::Success, 100, 0);
        log.record(
            2,
            AccessKind::Write,
            "b",
            "u",
            AccessStatus::Success,
            200,
            0,
        );
        assert_eq!(log.total_bytes(), 300);
    }

    #[test]
    fn test_log_counts_by_kind() {
        let mut log = AccessLog::new();
        log.record(1, AccessKind::Read, "a", "u", AccessStatus::Success, 0, 0);
        log.record(2, AccessKind::Read, "b", "u", AccessStatus::Success, 0, 0);
        log.record(3, AccessKind::Delete, "c", "u", AccessStatus::Success, 0, 0);
        let counts = log.counts_by_kind();
        assert_eq!(
            *counts.get(&AccessKind::Read).expect("get should succeed"),
            2
        );
        assert_eq!(
            *counts.get(&AccessKind::Delete).expect("get should succeed"),
            1
        );
    }

    #[test]
    fn test_log_in_range() {
        let mut log = AccessLog::new();
        log.record(10, AccessKind::Read, "a", "u", AccessStatus::Success, 0, 0);
        log.record(20, AccessKind::Read, "b", "u", AccessStatus::Success, 0, 0);
        log.record(30, AccessKind::Read, "c", "u", AccessStatus::Success, 0, 0);
        assert_eq!(log.in_range(15, 25).len(), 1);
        assert_eq!(log.in_range(10, 30).len(), 3);
        assert_eq!(log.in_range(50, 100).len(), 0);
    }

    #[test]
    fn test_log_clear() {
        let mut log = AccessLog::new();
        log.record(1, AccessKind::Read, "a", "u", AccessStatus::Success, 0, 0);
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_access_entry_new() {
        let e = AccessEntry::new(
            5,
            999,
            AccessKind::Head,
            "key",
            "svc",
            AccessStatus::NotFound,
            0,
            123,
        );
        assert_eq!(e.seq, 5);
        assert_eq!(e.kind, AccessKind::Head);
        assert_eq!(e.status, AccessStatus::NotFound);
        assert_eq!(e.duration_us, 123);
    }
}
