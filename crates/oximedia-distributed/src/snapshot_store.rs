#![allow(dead_code)]
//! Snapshot storage primitives for `OxiMedia` distributed cluster.
//!
//! Provides a lightweight in-memory snapshot store that can save, load, prune,
//! and validate distributed-state snapshots (e.g. Raft snapshots).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Identifies a snapshot by term + index (Raft-style).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotId {
    /// Election term in which the snapshot was taken.
    pub term: u64,
    /// Log index at which the snapshot was taken.
    pub index: u64,
}

impl SnapshotId {
    /// Create a new snapshot identifier.
    #[must_use]
    pub fn new(term: u64, index: u64) -> Self {
        Self { term, index }
    }

    /// A valid snapshot id must have both term > 0 and index > 0.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.term > 0 && self.index > 0
    }

    /// Return `true` if this snapshot is more recent than `other`
    /// (higher term, or same term and higher index).
    #[must_use]
    pub fn is_newer_than(&self, other: &SnapshotId) -> bool {
        self > other
    }
}

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "T{}:I{}", self.term, self.index)
    }
}

/// A captured snapshot of cluster state.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Unique identifier.
    pub id: SnapshotId,
    /// Serialised state data.
    pub data: Vec<u8>,
    /// Wall-clock time at which the snapshot was taken.
    pub captured_at: Instant,
    /// Node that produced this snapshot.
    pub source_node: String,
    /// Checksum (simple sum of bytes for illustration).
    pub checksum: u64,
}

impl Snapshot {
    /// Create a new snapshot.
    ///
    /// The checksum is computed automatically.
    #[must_use]
    pub fn new(id: SnapshotId, data: Vec<u8>, source_node: impl Into<String>) -> Self {
        let checksum = data.iter().map(|&b| u64::from(b)).sum();
        Self {
            id,
            data,
            captured_at: Instant::now(),
            source_node: source_node.into(),
            checksum,
        }
    }

    /// Size of the snapshot data in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if the snapshot is older than `max_age` relative to `now`.
    #[must_use]
    pub fn is_stale_at(&self, now: Instant, max_age: Duration) -> bool {
        now.saturating_duration_since(self.captured_at) >= max_age
    }

    /// Verify the checksum matches the stored data.
    #[must_use]
    pub fn verify_checksum(&self) -> bool {
        let expected: u64 = self.data.iter().map(|&b| u64::from(b)).sum();
        expected == self.checksum
    }
}

/// In-memory snapshot store with optional retention policy.
#[derive(Debug, Default)]
pub struct SnapshotStore {
    /// Snapshots keyed by their id.
    snapshots: BTreeMap<SnapshotId, Snapshot>,
    /// Total bytes stored.
    total_bytes: usize,
}

impl SnapshotStore {
    /// Create an empty snapshot store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Persist a snapshot.
    ///
    /// If a snapshot with the same id already exists it is replaced.
    pub fn save(&mut self, snapshot: Snapshot) {
        // If replacing, subtract old size.
        if let Some(old) = self.snapshots.get(&snapshot.id) {
            self.total_bytes -= old.size_bytes();
        }
        self.total_bytes += snapshot.size_bytes();
        self.snapshots.insert(snapshot.id, snapshot);
    }

    /// Retrieve a snapshot by id.
    #[must_use]
    pub fn load(&self, id: &SnapshotId) -> Option<&Snapshot> {
        self.snapshots.get(id)
    }

    /// Total number of snapshots in the store.
    #[must_use]
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Total bytes currently stored.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Return the most recent (highest id) snapshot, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.values().next_back()
    }

    /// Remove snapshots that are older than `max_age` relative to `now`.
    ///
    /// Returns the number of snapshots removed.
    pub fn prune_old(&mut self, now: Instant, max_age: Duration) -> usize {
        let stale_ids: Vec<SnapshotId> = self
            .snapshots
            .values()
            .filter(|s| s.is_stale_at(now, max_age))
            .map(|s| s.id)
            .collect();

        let removed = stale_ids.len();
        for id in stale_ids {
            if let Some(s) = self.snapshots.remove(&id) {
                self.total_bytes -= s.size_bytes();
            }
        }
        removed
    }

    /// Keep only the `keep_count` most recent snapshots, removing the rest.
    ///
    /// Returns the number of snapshots removed.
    pub fn retain_latest(&mut self, keep_count: usize) -> usize {
        if self.snapshots.len() <= keep_count {
            return 0;
        }
        let to_remove = self.snapshots.len() - keep_count;
        let old_ids: Vec<SnapshotId> = self.snapshots.keys().take(to_remove).copied().collect();
        for id in &old_ids {
            if let Some(s) = self.snapshots.remove(id) {
                self.total_bytes -= s.size_bytes();
            }
        }
        old_ids.len()
    }

    /// Return all snapshot ids in ascending order.
    #[must_use]
    pub fn all_ids(&self) -> Vec<SnapshotId> {
        self.snapshots.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn snap(term: u64, index: u64, data: Vec<u8>) -> Snapshot {
        Snapshot::new(SnapshotId::new(term, index), data, "node1")
    }

    #[test]
    fn test_snapshot_id_is_valid() {
        assert!(SnapshotId::new(1, 1).is_valid());
        assert!(!SnapshotId::new(0, 1).is_valid());
        assert!(!SnapshotId::new(1, 0).is_valid());
        assert!(!SnapshotId::new(0, 0).is_valid());
    }

    #[test]
    fn test_snapshot_id_ordering() {
        let a = SnapshotId::new(1, 5);
        let b = SnapshotId::new(2, 1);
        let c = SnapshotId::new(1, 10);
        assert!(b > a);
        assert!(c > a);
        assert!(b > c);
    }

    #[test]
    fn test_snapshot_id_is_newer_than() {
        let older = SnapshotId::new(1, 5);
        let newer = SnapshotId::new(2, 1);
        assert!(newer.is_newer_than(&older));
        assert!(!older.is_newer_than(&newer));
    }

    #[test]
    fn test_snapshot_id_display() {
        let id = SnapshotId::new(3, 42);
        assert_eq!(format!("{id}"), "T3:I42");
    }

    #[test]
    fn test_snapshot_checksum() {
        let s = snap(1, 1, vec![1u8, 2, 3]);
        assert!(s.verify_checksum());
        assert_eq!(s.checksum, 6);
    }

    #[test]
    fn test_snapshot_size_bytes() {
        let s = snap(1, 1, vec![0u8; 512]);
        assert_eq!(s.size_bytes(), 512);
    }

    #[test]
    fn test_snapshot_is_stale_at() {
        let mut s = snap(1, 1, vec![0]);
        // Backdate captured_at
        s.captured_at = Instant::now() - Duration::from_secs(20);
        assert!(s.is_stale_at(Instant::now(), Duration::from_secs(10)));
        assert!(!s.is_stale_at(Instant::now(), Duration::from_secs(30)));
    }

    #[test]
    fn test_store_save_and_load() {
        let mut store = SnapshotStore::new();
        let id = SnapshotId::new(1, 10);
        store.save(snap(1, 10, vec![7, 8, 9]));
        let loaded = store.load(&id).expect("loading should succeed");
        assert_eq!(loaded.checksum, 7 + 8 + 9);
    }

    #[test]
    fn test_store_count() {
        let mut store = SnapshotStore::new();
        assert_eq!(store.count(), 0);
        store.save(snap(1, 1, vec![1]));
        store.save(snap(1, 2, vec![2]));
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_store_total_bytes() {
        let mut store = SnapshotStore::new();
        store.save(snap(1, 1, vec![0u8; 100]));
        store.save(snap(1, 2, vec![0u8; 200]));
        assert_eq!(store.total_bytes(), 300);
    }

    #[test]
    fn test_store_replace_updates_bytes() {
        let mut store = SnapshotStore::new();
        store.save(snap(1, 1, vec![0u8; 100]));
        store.save(snap(1, 1, vec![0u8; 50])); // replace
        assert_eq!(store.count(), 1);
        assert_eq!(store.total_bytes(), 50);
    }

    #[test]
    fn test_store_latest() {
        let mut store = SnapshotStore::new();
        store.save(snap(1, 5, vec![1]));
        store.save(snap(2, 1, vec![2]));
        store.save(snap(1, 10, vec![3]));
        // Latest by BTreeMap ordering: highest SnapshotId = (2,1)
        assert_eq!(
            store.latest().expect("latest should exist").id,
            SnapshotId::new(2, 1)
        );
    }

    #[test]
    fn test_store_prune_old() {
        let mut store = SnapshotStore::new();
        let mut s_old = snap(1, 1, vec![1]);
        s_old.captured_at = Instant::now() - Duration::from_secs(20);
        store.save(s_old);
        store.save(snap(1, 2, vec![2])); // fresh

        let removed = store.prune_old(Instant::now(), Duration::from_secs(10));
        assert_eq!(removed, 1);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_store_retain_latest() {
        let mut store = SnapshotStore::new();
        store.save(snap(1, 1, vec![1]));
        store.save(snap(1, 2, vec![2]));
        store.save(snap(1, 3, vec![3]));
        store.save(snap(2, 1, vec![4]));

        let removed = store.retain_latest(2);
        assert_eq!(removed, 2);
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_store_all_ids_ordered() {
        let mut store = SnapshotStore::new();
        store.save(snap(2, 5, vec![]));
        store.save(snap(1, 1, vec![]));
        store.save(snap(1, 10, vec![]));
        let ids = store.all_ids();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
    }
}
