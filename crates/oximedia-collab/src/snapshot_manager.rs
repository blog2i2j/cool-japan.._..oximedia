#![allow(dead_code)]
//! Session state snapshots for recovery and time-travel.
//!
//! Provides the ability to capture, store, compare, and restore
//! point-in-time snapshots of a collaboration session's state.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Unique identifier for a snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SnapshotId(u64);

impl SnapshotId {
    /// Create a new snapshot id.
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the underlying value.
    pub fn value(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SnapshotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "snap-{}", self.0)
    }
}

/// Kind of snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotKind {
    /// Automatically taken on a schedule.
    Automatic,
    /// Manually triggered by a user.
    Manual,
    /// Taken before a potentially destructive operation.
    PreOperation,
    /// Taken as a recovery checkpoint.
    Checkpoint,
}

impl std::fmt::Display for SnapshotKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => write!(f, "Automatic"),
            Self::Manual => write!(f, "Manual"),
            Self::PreOperation => write!(f, "PreOperation"),
            Self::Checkpoint => write!(f, "Checkpoint"),
        }
    }
}

/// A key-value representation of session state.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionState {
    /// Opaque state entries keyed by path / property name.
    pub entries: HashMap<String, Vec<u8>>,
    /// Version counter of the session when snapshotted.
    pub version: u64,
    /// Number of active users at snapshot time.
    pub user_count: usize,
}

impl SessionState {
    /// Create an empty session state.
    pub fn new(version: u64, user_count: usize) -> Self {
        Self {
            entries: HashMap::new(),
            version,
            user_count,
        }
    }

    /// Insert a state entry.
    pub fn insert(&mut self, key: &str, value: Vec<u8>) {
        self.entries.insert(key.to_string(), value);
    }

    /// Size in bytes of all state entries.
    pub fn size_bytes(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }
}

/// A point-in-time snapshot.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Unique identifier.
    pub id: SnapshotId,
    /// When the snapshot was created.
    pub created_at: Instant,
    /// Kind of snapshot.
    pub kind: SnapshotKind,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// The captured state.
    pub state: SessionState,
    /// Who created it (user id or "system").
    pub created_by: String,
}

/// Difference between two snapshots.
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Keys only in the first snapshot.
    pub removed_keys: Vec<String>,
    /// Keys only in the second snapshot.
    pub added_keys: Vec<String>,
    /// Keys present in both but with different values.
    pub changed_keys: Vec<String>,
    /// Keys present in both with identical values.
    pub unchanged_keys: Vec<String>,
}

impl SnapshotDiff {
    /// Total number of differences.
    pub fn diff_count(&self) -> usize {
        self.removed_keys.len() + self.added_keys.len() + self.changed_keys.len()
    }

    /// Whether the two snapshots are identical.
    pub fn is_identical(&self) -> bool {
        self.diff_count() == 0
    }
}

/// Configuration for the snapshot manager.
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum number of snapshots to keep.
    pub max_snapshots: usize,
    /// Automatic snapshot interval.
    pub auto_interval: Duration,
    /// Whether automatic snapshots are enabled.
    pub auto_enabled: bool,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 100,
            auto_interval: Duration::from_secs(300),
            auto_enabled: true,
        }
    }
}

/// Manages session snapshots.
#[derive(Debug)]
pub struct SnapshotManager {
    /// Configuration.
    config: SnapshotConfig,
    /// Stored snapshots, ordered by creation time.
    snapshots: Vec<Snapshot>,
    /// Next snapshot id.
    next_id: u64,
    /// When the last automatic snapshot was taken.
    last_auto: Option<Instant>,
}

impl SnapshotManager {
    /// Create a new snapshot manager.
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            snapshots: Vec::new(),
            next_id: 1,
            last_auto: None,
        }
    }

    /// Take a snapshot.
    pub fn take_snapshot(
        &mut self,
        kind: SnapshotKind,
        state: SessionState,
        created_by: &str,
        label: Option<&str>,
    ) -> SnapshotId {
        let id = SnapshotId::new(self.next_id);
        self.next_id += 1;
        let snapshot = Snapshot {
            id,
            created_at: Instant::now(),
            kind,
            label: label.map(String::from),
            state,
            created_by: created_by.to_string(),
        };
        self.snapshots.push(snapshot);
        // Evict oldest if over limit
        while self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.remove(0);
        }
        if kind == SnapshotKind::Automatic {
            self.last_auto = Some(Instant::now());
        }
        id
    }

    /// Check if an automatic snapshot is due.
    pub fn is_auto_due(&self) -> bool {
        if !self.config.auto_enabled {
            return false;
        }
        match self.last_auto {
            None => true,
            Some(last) => last.elapsed() >= self.config.auto_interval,
        }
    }

    /// Get a snapshot by id.
    pub fn get(&self, id: SnapshotId) -> Option<&Snapshot> {
        self.snapshots.iter().find(|s| s.id == id)
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&Snapshot> {
        self.snapshots.last()
    }

    /// List all snapshot ids.
    pub fn list_ids(&self) -> Vec<SnapshotId> {
        self.snapshots.iter().map(|s| s.id).collect()
    }

    /// Number of stored snapshots.
    pub fn count(&self) -> usize {
        self.snapshots.len()
    }

    /// Delete a snapshot by id.
    pub fn delete(&mut self, id: SnapshotId) -> bool {
        let before = self.snapshots.len();
        self.snapshots.retain(|s| s.id != id);
        self.snapshots.len() < before
    }

    /// Compute the diff between two snapshots.
    pub fn diff(&self, a: SnapshotId, b: SnapshotId) -> Option<SnapshotDiff> {
        let snap_a = self.get(a)?;
        let snap_b = self.get(b)?;
        let keys_a: std::collections::HashSet<&String> = snap_a.state.entries.keys().collect();
        let keys_b: std::collections::HashSet<&String> = snap_b.state.entries.keys().collect();

        let removed_keys: Vec<String> = keys_a.difference(&keys_b).map(|k| (*k).clone()).collect();
        let added_keys: Vec<String> = keys_b.difference(&keys_a).map(|k| (*k).clone()).collect();
        let mut changed_keys = Vec::new();
        let mut unchanged_keys = Vec::new();
        for key in keys_a.intersection(&keys_b) {
            if snap_a.state.entries[*key] == snap_b.state.entries[*key] {
                unchanged_keys.push((*key).clone());
            } else {
                changed_keys.push((*key).clone());
            }
        }

        Some(SnapshotDiff {
            removed_keys,
            added_keys,
            changed_keys,
            unchanged_keys,
        })
    }

    /// Restore: returns a clone of the state from the given snapshot.
    pub fn restore(&self, id: SnapshotId) -> Option<SessionState> {
        self.get(id).map(|s| s.state.clone())
    }

    /// Total bytes consumed by all snapshots.
    pub fn total_size_bytes(&self) -> usize {
        self.snapshots.iter().map(|s| s.state.size_bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state(version: u64) -> SessionState {
        let mut state = SessionState::new(version, 2);
        state.insert("timeline", vec![1, 2, 3]);
        state.insert("tracks", vec![4, 5, 6]);
        state
    }

    #[test]
    fn test_snapshot_id_display() {
        let id = SnapshotId::new(42);
        assert_eq!(id.to_string(), "snap-42");
        assert_eq!(id.value(), 42);
    }

    #[test]
    fn test_snapshot_kind_display() {
        assert_eq!(SnapshotKind::Automatic.to_string(), "Automatic");
        assert_eq!(SnapshotKind::Manual.to_string(), "Manual");
        assert_eq!(SnapshotKind::PreOperation.to_string(), "PreOperation");
        assert_eq!(SnapshotKind::Checkpoint.to_string(), "Checkpoint");
    }

    #[test]
    fn test_session_state_size() {
        let state = sample_state(1);
        assert_eq!(state.size_bytes(), 6); // 3 + 3
    }

    #[test]
    fn test_take_snapshot() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let id = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "alice", Some("v1"));
        assert_eq!(mgr.count(), 1);
        let snap = mgr.get(id).expect("collab test operation should succeed");
        assert_eq!(snap.label.as_deref(), Some("v1"));
        assert_eq!(snap.created_by, "alice");
    }

    #[test]
    fn test_latest() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let _id1 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        let id2 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(2), "b", None);
        assert_eq!(
            mgr.latest()
                .expect("collab test operation should succeed")
                .id,
            id2
        );
    }

    #[test]
    fn test_delete_snapshot() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let id = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        assert!(mgr.delete(id));
        assert_eq!(mgr.count(), 0);
        assert!(!mgr.delete(id)); // already deleted
    }

    #[test]
    fn test_eviction_on_max() {
        let config = SnapshotConfig {
            max_snapshots: 3,
            ..SnapshotConfig::default()
        };
        let mut mgr = SnapshotManager::new(config);
        for i in 0..5 {
            mgr.take_snapshot(SnapshotKind::Automatic, sample_state(i), "sys", None);
        }
        assert_eq!(mgr.count(), 3);
    }

    #[test]
    fn test_diff_identical() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let id1 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        let id2 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        let diff = mgr
            .diff(id1, id2)
            .expect("collab test operation should succeed");
        assert!(diff.is_identical());
    }

    #[test]
    fn test_diff_added_removed() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let mut s1 = SessionState::new(1, 1);
        s1.insert("alpha", vec![1]);
        s1.insert("beta", vec![2]);
        let mut s2 = SessionState::new(2, 1);
        s2.insert("beta", vec![2]);
        s2.insert("gamma", vec![3]);
        let id1 = mgr.take_snapshot(SnapshotKind::Manual, s1, "a", None);
        let id2 = mgr.take_snapshot(SnapshotKind::Manual, s2, "a", None);
        let diff = mgr
            .diff(id1, id2)
            .expect("collab test operation should succeed");
        assert!(diff.removed_keys.contains(&"alpha".to_string()));
        assert!(diff.added_keys.contains(&"gamma".to_string()));
        assert!(diff.unchanged_keys.contains(&"beta".to_string()));
    }

    #[test]
    fn test_diff_changed() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let mut s1 = SessionState::new(1, 1);
        s1.insert("key", vec![1]);
        let mut s2 = SessionState::new(2, 1);
        s2.insert("key", vec![2]);
        let id1 = mgr.take_snapshot(SnapshotKind::Manual, s1, "a", None);
        let id2 = mgr.take_snapshot(SnapshotKind::Manual, s2, "a", None);
        let diff = mgr
            .diff(id1, id2)
            .expect("collab test operation should succeed");
        assert!(diff.changed_keys.contains(&"key".to_string()));
        assert_eq!(diff.diff_count(), 1);
    }

    #[test]
    fn test_restore() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let state = sample_state(42);
        let id = mgr.take_snapshot(SnapshotKind::Checkpoint, state.clone(), "sys", None);
        let restored = mgr
            .restore(id)
            .expect("collab test operation should succeed");
        assert_eq!(restored.version, 42);
        assert_eq!(restored.entries.len(), 2);
    }

    #[test]
    fn test_list_ids() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        let id1 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        let id2 = mgr.take_snapshot(SnapshotKind::Manual, sample_state(2), "a", None);
        let ids = mgr.list_ids();
        assert_eq!(ids, vec![id1, id2]);
    }

    #[test]
    fn test_total_size_bytes() {
        let mut mgr = SnapshotManager::new(SnapshotConfig::default());
        mgr.take_snapshot(SnapshotKind::Manual, sample_state(1), "a", None);
        mgr.take_snapshot(SnapshotKind::Manual, sample_state(2), "a", None);
        assert_eq!(mgr.total_size_bytes(), 12); // 6 bytes each
    }

    #[test]
    fn test_is_auto_due_initially() {
        let mgr = SnapshotManager::new(SnapshotConfig::default());
        assert!(mgr.is_auto_due());
    }

    #[test]
    fn test_is_auto_due_disabled() {
        let config = SnapshotConfig {
            auto_enabled: false,
            ..SnapshotConfig::default()
        };
        let mgr = SnapshotManager::new(config);
        assert!(!mgr.is_auto_due());
    }
}
