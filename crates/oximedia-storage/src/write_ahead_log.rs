#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
//! Write-ahead log (WAL) for crash-safe storage operations.
//!
//! Provides an append-only, sequentially numbered log of storage mutations.
//! Entries can be replayed on recovery to bring the storage backend to a
//! consistent state after an unexpected crash or restart.

use std::collections::VecDeque;
use std::fmt;

/// Maximum number of entries kept in-memory before compaction is recommended.
const DEFAULT_MAX_ENTRIES: usize = 10_000;

/// Unique, monotonically increasing sequence number for WAL entries.
pub type Lsn = u64;

/// The kind of storage mutation recorded in the WAL.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WalOperation {
    /// A new object was created or overwritten.
    Put {
        /// Object key.
        key: String,
        /// Size in bytes of the object.
        size: u64,
    },
    /// An object was deleted.
    Delete {
        /// Object key.
        key: String,
    },
    /// An object was copied.
    Copy {
        /// Source key.
        src: String,
        /// Destination key.
        dst: String,
    },
    /// Metadata was updated.
    UpdateMeta {
        /// Object key.
        key: String,
        /// Metadata field name.
        field: String,
        /// New value.
        value: String,
    },
}

impl fmt::Display for WalOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Put { key, size } => write!(f, "PUT {key} ({size} bytes)"),
            Self::Delete { key } => write!(f, "DELETE {key}"),
            Self::Copy { src, dst } => write!(f, "COPY {src} -> {dst}"),
            Self::UpdateMeta { key, field, value } => {
                write!(f, "META {key} {field}={value}")
            }
        }
    }
}

/// A single entry in the write-ahead log.
#[derive(Debug, Clone)]
pub struct WalEntry {
    /// Log sequence number.
    pub lsn: Lsn,
    /// Timestamp (milliseconds since an epoch).
    pub timestamp_ms: u64,
    /// The operation.
    pub operation: WalOperation,
    /// Whether this entry has been flushed to durable storage.
    pub flushed: bool,
}

impl WalEntry {
    /// Create a new unflushed entry.
    pub fn new(lsn: Lsn, timestamp_ms: u64, operation: WalOperation) -> Self {
        Self {
            lsn,
            timestamp_ms,
            operation,
            flushed: false,
        }
    }

    /// Mark as flushed.
    pub fn mark_flushed(&mut self) {
        self.flushed = true;
    }

    /// Returns the object key affected by this entry.
    pub fn affected_key(&self) -> &str {
        match &self.operation {
            WalOperation::Put { key, .. }
            | WalOperation::Delete { key }
            | WalOperation::UpdateMeta { key, .. } => key,
            WalOperation::Copy { src, .. } => src,
        }
    }
}

impl fmt::Display for WalEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.flushed { "F" } else { "U" };
        write!(f, "[{}][{}] {}", self.lsn, status, self.operation)
    }
}

/// The in-memory write-ahead log.
#[derive(Debug)]
pub struct WriteAheadLog {
    /// Ordered entries.
    entries: VecDeque<WalEntry>,
    /// Next LSN to assign.
    next_lsn: Lsn,
    /// Maximum entries before compaction is recommended.
    max_entries: usize,
    /// Current simulated timestamp source.
    current_time_ms: u64,
}

impl WriteAheadLog {
    /// Create a new empty WAL.
    pub fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            next_lsn: 1,
            max_entries: DEFAULT_MAX_ENTRIES,
            current_time_ms: 0,
        }
    }

    /// Create a WAL with a custom max-entries limit.
    pub fn with_max_entries(max: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            next_lsn: 1,
            max_entries: max,
            current_time_ms: 0,
        }
    }

    /// Advance the internal clock (for testing / deterministic replay).
    pub fn set_time_ms(&mut self, ms: u64) {
        self.current_time_ms = ms;
    }

    /// Append an operation, returning the assigned LSN.
    pub fn append(&mut self, op: WalOperation) -> Lsn {
        let lsn = self.next_lsn;
        self.next_lsn += 1;
        let entry = WalEntry::new(lsn, self.current_time_ms, op);
        self.entries.push_back(entry);
        lsn
    }

    /// Number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether compaction is recommended.
    pub fn needs_compaction(&self) -> bool {
        self.entries.len() >= self.max_entries
    }

    /// Get an entry by LSN.
    pub fn get(&self, lsn: Lsn) -> Option<&WalEntry> {
        self.entries.iter().find(|e| e.lsn == lsn)
    }

    /// Mark an entry as flushed.
    pub fn mark_flushed(&mut self, lsn: Lsn) -> bool {
        if let Some(e) = self.entries.iter_mut().find(|e| e.lsn == lsn) {
            e.mark_flushed();
            true
        } else {
            false
        }
    }

    /// Count unflushed entries.
    pub fn unflushed_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.flushed).count()
    }

    /// Compact: remove all flushed entries from the front of the log.
    pub fn compact(&mut self) -> usize {
        let mut removed = 0;
        while let Some(front) = self.entries.front() {
            if front.flushed {
                self.entries.pop_front();
                removed += 1;
            } else {
                break;
            }
        }
        removed
    }

    /// Return entries that need replay (unflushed, in LSN order).
    pub fn replay_entries(&self) -> Vec<&WalEntry> {
        self.entries.iter().filter(|e| !e.flushed).collect()
    }

    /// Return all entries affecting a given key.
    pub fn entries_for_key(&self, key: &str) -> Vec<&WalEntry> {
        self.entries
            .iter()
            .filter(|e| e.affected_key() == key)
            .collect()
    }

    /// The highest assigned LSN so far (0 if none).
    pub fn latest_lsn(&self) -> Lsn {
        self.next_lsn.saturating_sub(1)
    }

    /// Truncate all entries with LSN greater than the given value.
    pub fn truncate_after(&mut self, lsn: Lsn) {
        self.entries.retain(|e| e.lsn <= lsn);
        self.next_lsn = lsn + 1;
    }

    /// Bytes referenced by Put operations in the log (sum of sizes).
    pub fn total_put_bytes(&self) -> u64 {
        self.entries
            .iter()
            .filter_map(|e| {
                if let WalOperation::Put { size, .. } = &e.operation {
                    Some(*size)
                } else {
                    None
                }
            })
            .sum()
    }
}

impl Default for WriteAheadLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_op(key: &str, size: u64) -> WalOperation {
        WalOperation::Put {
            key: key.to_string(),
            size,
        }
    }

    fn del_op(key: &str) -> WalOperation {
        WalOperation::Delete {
            key: key.to_string(),
        }
    }

    // ── WalOperation ───────────────────────────────────────────────────────

    #[test]
    fn test_op_display_put() {
        let op = put_op("file.mp4", 1024);
        let s = op.to_string();
        assert!(s.contains("PUT"));
        assert!(s.contains("file.mp4"));
        assert!(s.contains("1024"));
    }

    #[test]
    fn test_op_display_delete() {
        let op = del_op("old.bin");
        assert!(op.to_string().contains("DELETE"));
    }

    #[test]
    fn test_op_display_copy() {
        let op = WalOperation::Copy {
            src: "a".into(),
            dst: "b".into(),
        };
        let s = op.to_string();
        assert!(s.contains("COPY"));
        assert!(s.contains("->"));
    }

    // ── WalEntry ───────────────────────────────────────────────────────────

    #[test]
    fn test_entry_new() {
        let e = WalEntry::new(1, 1000, put_op("k", 10));
        assert_eq!(e.lsn, 1);
        assert!(!e.flushed);
        assert_eq!(e.affected_key(), "k");
    }

    #[test]
    fn test_entry_mark_flushed() {
        let mut e = WalEntry::new(1, 0, del_op("k"));
        e.mark_flushed();
        assert!(e.flushed);
    }

    #[test]
    fn test_entry_display() {
        let e = WalEntry::new(5, 0, put_op("x", 100));
        let s = e.to_string();
        assert!(s.contains("[5]"));
        assert!(s.contains("[U]")); // unflushed
    }

    // ── WriteAheadLog ──────────────────────────────────────────────────────

    #[test]
    fn test_wal_empty() {
        let wal = WriteAheadLog::new();
        assert!(wal.is_empty());
        assert_eq!(wal.len(), 0);
        assert_eq!(wal.latest_lsn(), 0);
    }

    #[test]
    fn test_wal_append() {
        let mut wal = WriteAheadLog::new();
        let lsn1 = wal.append(put_op("a", 10));
        let lsn2 = wal.append(del_op("b"));
        assert_eq!(lsn1, 1);
        assert_eq!(lsn2, 2);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.latest_lsn(), 2);
    }

    #[test]
    fn test_wal_get() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("x", 50));
        let entry = wal.get(1).expect("get should succeed");
        assert_eq!(entry.affected_key(), "x");
    }

    #[test]
    fn test_wal_mark_flushed() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("f", 1));
        assert!(wal.mark_flushed(1));
        assert!(!wal.mark_flushed(999));
        assert_eq!(wal.unflushed_count(), 0);
    }

    #[test]
    fn test_wal_unflushed_count() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("a", 1));
        wal.append(put_op("b", 2));
        wal.mark_flushed(1);
        assert_eq!(wal.unflushed_count(), 1);
    }

    #[test]
    fn test_wal_compact() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("a", 1));
        wal.append(put_op("b", 2));
        wal.append(put_op("c", 3));
        wal.mark_flushed(1);
        wal.mark_flushed(2);
        let removed = wal.compact();
        assert_eq!(removed, 2);
        assert_eq!(wal.len(), 1);
    }

    #[test]
    fn test_wal_replay_entries() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("a", 1));
        wal.append(del_op("b"));
        wal.mark_flushed(1);
        let replay = wal.replay_entries();
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].lsn, 2);
    }

    #[test]
    fn test_wal_entries_for_key() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("x", 10));
        wal.append(del_op("y"));
        wal.append(put_op("x", 20));
        let xs = wal.entries_for_key("x");
        assert_eq!(xs.len(), 2);
    }

    #[test]
    fn test_wal_truncate_after() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("a", 1));
        wal.append(put_op("b", 2));
        wal.append(put_op("c", 3));
        wal.truncate_after(2);
        assert_eq!(wal.len(), 2);
        assert_eq!(wal.latest_lsn(), 2);
    }

    #[test]
    fn test_wal_total_put_bytes() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("a", 100));
        wal.append(del_op("b"));
        wal.append(put_op("c", 200));
        assert_eq!(wal.total_put_bytes(), 300);
    }

    #[test]
    fn test_wal_needs_compaction() {
        let mut wal = WriteAheadLog::with_max_entries(3);
        wal.append(put_op("a", 1));
        wal.append(put_op("b", 2));
        assert!(!wal.needs_compaction());
        wal.append(put_op("c", 3));
        assert!(wal.needs_compaction());
    }

    #[test]
    fn test_wal_set_time() {
        let mut wal = WriteAheadLog::new();
        wal.set_time_ms(5000);
        wal.append(put_op("t", 1));
        assert_eq!(wal.get(1).expect("get should succeed").timestamp_ms, 5000);
    }

    // ── Concurrent WAL tests ────────────────────────────────────────────────

    #[test]
    fn test_wal_concurrent_writers_no_corruption() {
        use std::sync::{Arc, Mutex};
        // 8 writers × 50 ops = 400 entries
        let wal = Arc::new(Mutex::new(WriteAheadLog::new()));
        let handles: Vec<_> = (0..8u64)
            .map(|i| {
                let w = wal.clone();
                std::thread::spawn(move || {
                    for j in 0..50u64 {
                        let key = format!("worker-{i}-op-{j}");
                        let mut guard = w.lock().expect("wal lock");
                        guard.append(WalOperation::Put { key, size: j });
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread should not panic");
        }
        let guard = wal.lock().expect("wal lock");
        assert_eq!(guard.len(), 400, "expected 400 entries from 8 * 50 writes");
    }

    #[test]
    fn test_wal_concurrent_lsn_monotonic() {
        use std::sync::{Arc, Mutex};
        let wal = Arc::new(Mutex::new(WriteAheadLog::new()));
        let handles: Vec<_> = (0..4u64)
            .map(|i| {
                let w = wal.clone();
                std::thread::spawn(move || {
                    let mut guard = w.lock().expect("wal lock");
                    guard.append(put_op(&format!("t-{i}"), i))
                })
            })
            .collect();
        let lsns: Vec<u64> = handles
            .into_iter()
            .map(|h| h.join().expect("thread"))
            .collect();
        // All LSNs should be unique (monotonic)
        let mut sorted = lsns.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), lsns.len(), "LSNs should be unique");
    }

    #[test]
    fn test_wal_concurrent_flush_and_compact() {
        use std::sync::{Arc, Mutex};
        let wal = Arc::new(Mutex::new(WriteAheadLog::new()));
        // Writer thread
        let w = wal.clone();
        let writer = std::thread::spawn(move || {
            for i in 0..100u64 {
                let mut guard = w.lock().expect("wal lock");
                guard.append(put_op(&format!("k{i}"), i));
            }
        });
        writer.join().expect("writer thread");

        // Mark first 50 as flushed
        {
            let mut guard = wal.lock().expect("wal lock");
            for lsn in 1..=50 {
                guard.mark_flushed(lsn);
            }
            let removed = guard.compact();
            assert_eq!(removed, 50);
            assert_eq!(guard.len(), 50);
        }
    }

    #[test]
    fn test_wal_replay_after_partial_flush() {
        let mut wal = WriteAheadLog::new();
        for i in 0..10u64 {
            wal.append(put_op(&format!("file-{i}"), i * 100));
        }
        // Mark even-LSN entries as flushed
        for lsn in [2u64, 4, 6, 8, 10] {
            wal.mark_flushed(lsn);
        }
        let replay = wal.replay_entries();
        // Odd-LSN entries (1,3,5,7,9) should be in replay
        assert_eq!(replay.len(), 5);
        for entry in &replay {
            assert!(!entry.flushed);
        }
    }

    #[test]
    fn test_wal_sequential_ordering_preserved() {
        let mut wal = WriteAheadLog::new();
        let keys = ["alpha", "beta", "gamma", "delta"];
        for &k in &keys {
            wal.append(put_op(k, 10));
        }
        let all: Vec<&WalEntry> = wal.entries.iter().collect();
        // LSNs must be strictly increasing
        for window in all.windows(2) {
            assert!(window[1].lsn > window[0].lsn);
        }
    }

    #[test]
    fn test_wal_crash_recovery_replay_all() {
        // Simulate crash: write entries without flushing, then replay all
        let mut wal = WriteAheadLog::new();
        for i in 0..20u64 {
            wal.append(put_op(&format!("obj-{i}"), i * 512));
        }
        // Simulate restart: all entries are unflushed → replay everything
        let replay = wal.replay_entries();
        assert_eq!(replay.len(), 20, "all 20 entries should be replayed");
        assert_eq!(wal.unflushed_count(), 20);
    }

    #[test]
    fn test_wal_truncate_preserves_lsn_order() {
        let mut wal = WriteAheadLog::new();
        for i in 0..10u64 {
            wal.append(del_op(&format!("del-{i}")));
        }
        wal.truncate_after(5);
        assert_eq!(wal.len(), 5);
        // latest_lsn should now be 5 (was reset to 5+1)
        assert_eq!(wal.latest_lsn(), 5);
    }

    #[test]
    fn test_wal_multiple_ops_per_key() {
        let mut wal = WriteAheadLog::new();
        wal.append(put_op("video.mp4", 1024));
        wal.append(WalOperation::UpdateMeta {
            key: "video.mp4".to_string(),
            field: "title".to_string(),
            value: "My Video".to_string(),
        });
        wal.append(WalOperation::Copy {
            src: "video.mp4".to_string(),
            dst: "backup/video.mp4".to_string(),
        });
        wal.append(del_op("video.mp4"));
        let entries = wal.entries_for_key("video.mp4");
        // Put, UpdateMeta, Delete all reference "video.mp4" directly.
        // Copy references via src.
        assert_eq!(entries.len(), 4);
    }
}
