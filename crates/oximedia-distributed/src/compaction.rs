//! Raft log compaction via snapshotting.
//!
//! Provides `LogCompactor` for truncating the Raft log up to a committed
//! snapshot index, and `Snapshot` to carry the state-machine snapshot data.

use crate::distributed_enhancements::LogEntry;

/// A snapshot of the replicated state machine at a given log index.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The log index that this snapshot covers (inclusive).
    pub last_included_index: u64,
    /// The term of the last included log entry.
    pub last_included_term: u64,
    /// Encoded state-machine data at `last_included_index`.
    pub data: Vec<u8>,
}

/// Handles compaction of the Raft log by replacing committed entries with a
/// snapshot, thereby bounding log growth.
pub struct LogCompactor;

impl LogCompactor {
    /// Create a new `LogCompactor`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Compact the Raft log up to (and including) `snapshot_idx`.
    ///
    /// All log entries with `index <= snapshot_idx` are removed from `log`.
    /// Returns a `Snapshot` that captures the term of the last removed entry
    /// (or 0 if no entry was found at `snapshot_idx`) and an empty state-data
    /// payload.  Callers are expected to replace the payload with the actual
    /// serialised state-machine snapshot before persisting.
    ///
    /// # Arguments
    ///
    /// * `log`          - Mutable reference to the Raft log.
    /// * `snapshot_idx` - The log index up to which compaction should proceed.
    ///
    /// # Returns
    ///
    /// A [`Snapshot`] representing the compaction boundary.
    #[must_use]
    pub fn compact(log: &mut Vec<LogEntry>, snapshot_idx: u64) -> Snapshot {
        // Find the term of the last entry being snapshotted
        let last_included_term = log
            .iter()
            .rev()
            .find(|e| e.index <= snapshot_idx)
            .map(|e| e.term)
            .unwrap_or(0);

        // Remove all entries covered by the snapshot
        log.retain(|e| e.index > snapshot_idx);

        Snapshot {
            last_included_index: snapshot_idx,
            last_included_term,
            data: Vec::new(),
        }
    }

    /// Apply a received snapshot to a log: truncate the entire log if the
    /// snapshot is newer than all existing entries.
    ///
    /// Returns `true` if the snapshot was applied (log was truncated), `false`
    /// if the log already has entries beyond the snapshot (snapshot is stale).
    pub fn install_snapshot(log: &mut Vec<LogEntry>, snapshot: &Snapshot) -> bool {
        let log_last_index = log.last().map(|e| e.index).unwrap_or(0);
        if snapshot.last_included_index >= log_last_index {
            log.clear();
            true
        } else {
            // Partial install: remove only the covered prefix
            log.retain(|e| e.index > snapshot.last_included_index);
            false
        }
    }
}

impl Default for LogCompactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_log(count: u64) -> Vec<LogEntry> {
        (1..=count)
            .map(|i| LogEntry {
                term: (i + 1) / 2,
                index: i,
                command: format!("cmd-{i}"),
            })
            .collect()
    }

    #[test]
    fn test_compact_removes_entries_up_to_index() {
        let mut log = make_log(10);
        let snap = LogCompactor::compact(&mut log, 5);
        assert_eq!(snap.last_included_index, 5);
        assert!(log.iter().all(|e| e.index > 5));
        assert_eq!(log.len(), 5);
    }

    #[test]
    fn test_compact_all_entries() {
        let mut log = make_log(5);
        let snap = LogCompactor::compact(&mut log, 5);
        assert_eq!(snap.last_included_index, 5);
        assert!(log.is_empty());
    }

    #[test]
    fn test_compact_no_matching_entries() {
        let mut log = make_log(5);
        // snapshot_idx 0 → nothing removed (all have index >= 1)
        let snap = LogCompactor::compact(&mut log, 0);
        assert_eq!(snap.last_included_index, 0);
        assert_eq!(snap.last_included_term, 0);
        assert_eq!(log.len(), 5);
    }

    #[test]
    fn test_compact_empty_log() {
        let mut log: Vec<LogEntry> = Vec::new();
        let snap = LogCompactor::compact(&mut log, 10);
        assert_eq!(snap.last_included_index, 10);
        assert_eq!(snap.last_included_term, 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_install_snapshot_truncates_older_log() {
        let mut log = make_log(5);
        let snap = Snapshot {
            last_included_index: 10,
            last_included_term: 5,
            data: Vec::new(),
        };
        let applied = LogCompactor::install_snapshot(&mut log, &snap);
        assert!(applied);
        assert!(log.is_empty());
    }

    #[test]
    fn test_install_snapshot_stale_does_not_clear_all() {
        let mut log = make_log(10);
        let snap = Snapshot {
            last_included_index: 5,
            last_included_term: 3,
            data: Vec::new(),
        };
        LogCompactor::install_snapshot(&mut log, &snap);
        // Entries 6–10 remain
        assert!(log.iter().all(|e| e.index > 5));
    }
}
