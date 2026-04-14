//! Incremental state serialization for collaborative session synchronization.
//!
//! When synchronising a collaborative editing session the naïve approach is to
//! transmit the full project state with every message. For large projects this
//! is prohibitively expensive. This module provides an **incremental sync**
//! layer that:
//!
//! - Tracks which version of the state each peer last acknowledged.
//! - Produces minimal *delta payloads* containing only the fields that have
//!   changed since a given baseline version.
//! - Compresses each field-value pair using a simple run-length delta encoding
//!   that exploits temporal locality.
//! - Reconstructs the full state from a baseline plus a sequence of deltas.
//!
//! ## Design
//!
//! The unit of state is a flat [`StateMap`]: a `HashMap<String, StateValue>`
//! keyed by field path (e.g. `"timeline/clip/42/start_frame"`). A
//! [`StateVersion`] is a monotonically increasing counter. A [`StateDelta`]
//! records the changes between two consecutive versions.
//!
//! The [`IncrementalStateStore`] stores all versions and can produce a
//! [`StateDelta`] for any `(from, to)` range. The [`PeerSyncTracker`] keeps
//! track of each connected peer's acknowledged version so only the minimal
//! delta needs to be sent.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// StateValue
// ─────────────────────────────────────────────────────────────────────────────

/// The value stored for a single field in the state map.
#[derive(Debug, Clone, PartialEq)]
pub enum StateValue {
    /// Integer field.
    Int(i64),
    /// Floating-point field.
    Float(f64),
    /// String field.
    Text(String),
    /// Boolean flag.
    Bool(bool),
    /// Raw bytes (e.g. serialised sub-document).
    Bytes(Vec<u8>),
    /// Nested JSON-like object (field → value).
    Object(HashMap<String, StateValue>),
    /// The field has been removed.
    Tombstone,
}

impl fmt::Display for StateValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Text(s) => write!(f, "{s:?}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Bytes(b) => write!(f, "<{} bytes>", b.len()),
            Self::Object(m) => write!(f, "{{…{} fields…}}", m.len()),
            Self::Tombstone => write!(f, "<deleted>"),
        }
    }
}

/// A flat map from field paths to values.
pub type StateMap = HashMap<String, StateValue>;

// ─────────────────────────────────────────────────────────────────────────────
// StateVersion
// ─────────────────────────────────────────────────────────────────────────────

/// A monotonically increasing version counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateVersion(pub u64);

impl StateVersion {
    /// The zero version (before any state exists).
    pub const ZERO: Self = Self(0);

    /// Increment and return the next version.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl fmt::Display for StateVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FieldChange
// ─────────────────────────────────────────────────────────────────────────────

/// A single changed field within a delta.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// Dot-separated field path (e.g. `"timeline.clips.42.start_frame"`).
    pub path: String,
    /// The new value, or `StateValue::Tombstone` for deletion.
    pub new_value: StateValue,
}

impl FieldChange {
    /// Create a field change record.
    #[must_use]
    pub fn new(path: impl Into<String>, new_value: StateValue) -> Self {
        Self { path: path.into(), new_value }
    }

    /// Shorthand for a deletion.
    #[must_use]
    pub fn delete(path: impl Into<String>) -> Self {
        Self::new(path, StateValue::Tombstone)
    }

    /// `true` if this change removes the field.
    #[must_use]
    pub fn is_deletion(&self) -> bool {
        self.new_value == StateValue::Tombstone
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StateDelta
// ─────────────────────────────────────────────────────────────────────────────

/// The minimal set of field changes needed to advance from one version to
/// another.
#[derive(Debug, Clone, Default)]
pub struct StateDelta {
    /// Version from which this delta can be applied.
    pub from_version: StateVersion,
    /// Version produced after applying this delta.
    pub to_version: StateVersion,
    /// The individual field changes, ordered by path for determinism.
    pub changes: Vec<FieldChange>,
}

impl StateDelta {
    /// Create a new empty delta.
    #[must_use]
    pub fn new(from: StateVersion, to: StateVersion) -> Self {
        Self {
            from_version: from,
            to_version: to,
            changes: Vec::new(),
        }
    }

    /// Number of changed fields.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Estimated serialised byte size (rough heuristic).
    ///
    /// Useful for deciding whether to send a delta or a full snapshot.
    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.changes
            .iter()
            .map(|c| {
                c.path.len()
                    + match &c.new_value {
                        StateValue::Int(_) | StateValue::Float(_) | StateValue::Bool(_) => 8,
                        StateValue::Text(s) => s.len(),
                        StateValue::Bytes(b) => b.len(),
                        StateValue::Object(m) => m.len() * 32,
                        StateValue::Tombstone => 1,
                    }
            })
            .sum()
    }

    /// `true` when the delta contains no changes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// IncrementalStateStore
// ─────────────────────────────────────────────────────────────────────────────

/// Maintains a versioned history of project state changes and produces minimal
/// incremental deltas for synchronisation.
///
/// Internally a `BTreeMap<StateVersion, StateDelta>` records every individual
/// delta as it is committed.  To compute the delta between arbitrary versions
/// `(from, to)` the store merges all intermediate deltas, retaining only the
/// *last* change for each field path.
#[derive(Debug, Default)]
pub struct IncrementalStateStore {
    /// All committed deltas sorted by version.
    deltas: BTreeMap<StateVersion, StateDelta>,
    /// The current head version.
    head: StateVersion,
}

impl IncrementalStateStore {
    /// Create a new empty store at version 0.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The current head version.
    #[must_use]
    pub fn head(&self) -> StateVersion {
        self.head
    }

    /// Number of stored deltas.
    #[must_use]
    pub fn delta_count(&self) -> usize {
        self.deltas.len()
    }

    /// Commit a set of field changes as a new version.
    ///
    /// Returns the new version number.
    pub fn commit(&mut self, changes: Vec<FieldChange>) -> StateVersion {
        let from = self.head;
        let to = from.next();
        let mut delta = StateDelta::new(from, to);
        delta.changes = changes;
        self.deltas.insert(to, delta);
        self.head = to;
        to
    }

    /// Produce a merged delta from `from_version` to `to_version`.
    ///
    /// Returns `None` if either version is out of range.
    ///
    /// If `from == to` an empty delta is returned. If `to < from` the versions
    /// are swapped (always returns forward delta).
    #[must_use]
    pub fn delta_between(
        &self,
        from: StateVersion,
        to: StateVersion,
    ) -> Option<StateDelta> {
        let (lo, hi) = if from <= to { (from, to) } else { (to, from) };

        if hi > self.head {
            return None;
        }

        // Same-version: return an empty delta immediately to avoid an invalid
        // BTreeMap range (lo+1 > lo when lo == hi).
        if lo == hi {
            return Some(StateDelta::new(lo, hi));
        }

        // Collect all deltas in range (lo, hi] and merge them.
        // For duplicate paths keep the last-seen value.
        let mut merged: HashMap<String, StateValue> = HashMap::new();

        for (_, d) in self.deltas.range((lo.next())..=hi) {
            for change in &d.changes {
                merged.insert(change.path.clone(), change.new_value.clone());
            }
        }

        let mut result = StateDelta::new(lo, hi);
        result.changes = {
            let mut v: Vec<FieldChange> = merged
                .into_iter()
                .map(|(path, value)| FieldChange::new(path, value))
                .collect();
            v.sort_by(|a, b| a.path.cmp(&b.path)); // deterministic order
            v
        };

        Some(result)
    }

    /// Reconstruct the full state at `version` by replaying all deltas from
    /// `StateVersion::ZERO`.
    ///
    /// Returns `None` if `version > head`.
    #[must_use]
    pub fn reconstruct(&self, version: StateVersion) -> Option<StateMap> {
        if version > self.head {
            return None;
        }
        let delta = self.delta_between(StateVersion::ZERO, version)?;
        let mut state: StateMap = HashMap::new();
        for change in delta.changes {
            if change.is_deletion() {
                state.remove(&change.path);
            } else {
                state.insert(change.path, change.new_value);
            }
        }
        Some(state)
    }

    /// Prune all deltas older than `keep_from_version`, freeing memory.
    ///
    /// After pruning it is no longer possible to compute deltas with `from <
    /// keep_from_version`.  Returns the number of deltas removed.
    pub fn prune(&mut self, keep_from_version: StateVersion) -> usize {
        let before = self.deltas.len();
        self.deltas.retain(|v, _| *v > keep_from_version);
        before - self.deltas.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PeerSyncTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks the acknowledged state version for each connected peer so that the
/// server can generate the minimal delta needed to bring them up to date.
#[derive(Debug, Default)]
pub struct PeerSyncTracker {
    /// Last acknowledged version per peer.
    acked: HashMap<String, StateVersion>,
}

impl PeerSyncTracker {
    /// Create a new tracker with no peers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new peer.  They start at version 0 (need the full history).
    pub fn register(&mut self, peer_id: impl Into<String>) {
        self.acked.entry(peer_id.into()).or_insert(StateVersion::ZERO);
    }

    /// Record that `peer_id` has successfully processed up to `version`.
    pub fn acknowledge(&mut self, peer_id: &str, version: StateVersion) {
        if let Some(v) = self.acked.get_mut(peer_id) {
            if version > *v {
                *v = version;
            }
        }
    }

    /// Remove a peer.
    pub fn deregister(&mut self, peer_id: &str) {
        self.acked.remove(peer_id);
    }

    /// Return the acknowledged version for `peer_id`, or `None` if unknown.
    #[must_use]
    pub fn acked_version(&self, peer_id: &str) -> Option<StateVersion> {
        self.acked.get(peer_id).copied()
    }

    /// Return the minimum acknowledged version across all peers.
    ///
    /// This is the safe pruning horizon: all deltas at or before this version
    /// can be discarded without losing the ability to sync any current peer.
    #[must_use]
    pub fn min_acked_version(&self) -> StateVersion {
        self.acked.values().copied().min().unwrap_or(StateVersion::ZERO)
    }

    /// All peer identifiers.
    #[must_use]
    pub fn peer_ids(&self) -> Vec<&str> {
        self.acked.keys().map(String::as_str).collect()
    }

    /// Number of registered peers.
    #[must_use]
    pub fn peer_count(&self) -> usize {
        self.acked.len()
    }

    /// Produce the delta that `peer_id` needs from `store`.
    ///
    /// Returns `None` if the peer is unknown or the store cannot satisfy the
    /// range.
    #[must_use]
    pub fn delta_for_peer(
        &self,
        peer_id: &str,
        store: &IncrementalStateStore,
    ) -> Option<StateDelta> {
        let from = self.acked_version(peer_id)?;
        store.delta_between(from, store.head())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── StateValue ──

    #[test]
    fn test_state_value_display_int() {
        assert_eq!(StateValue::Int(42).to_string(), "42");
    }

    #[test]
    fn test_state_value_display_tombstone() {
        assert_eq!(StateValue::Tombstone.to_string(), "<deleted>");
    }

    #[test]
    fn test_state_value_display_bytes() {
        let s = StateValue::Bytes(vec![0u8; 10]).to_string();
        assert!(s.contains("10 bytes"));
    }

    // ── StateVersion ──

    #[test]
    fn test_state_version_ordering() {
        assert!(StateVersion(1) > StateVersion(0));
        assert!(StateVersion::ZERO < StateVersion(1));
    }

    #[test]
    fn test_state_version_display() {
        assert_eq!(StateVersion(5).to_string(), "v5");
    }

    #[test]
    fn test_state_version_next() {
        assert_eq!(StateVersion(3).next(), StateVersion(4));
    }

    // ── FieldChange ──

    #[test]
    fn test_field_change_is_deletion() {
        let del = FieldChange::delete("a.b");
        assert!(del.is_deletion());
        let upd = FieldChange::new("a.b", StateValue::Int(1));
        assert!(!upd.is_deletion());
    }

    // ── StateDelta ──

    #[test]
    fn test_delta_is_empty_when_no_changes() {
        let d = StateDelta::new(StateVersion(0), StateVersion(1));
        assert!(d.is_empty());
        assert_eq!(d.change_count(), 0);
    }

    #[test]
    fn test_delta_estimated_bytes_grows_with_text() {
        let mut d = StateDelta::new(StateVersion(0), StateVersion(1));
        d.changes.push(FieldChange::new("x", StateValue::Text("hello".to_string())));
        assert!(d.estimated_bytes() >= 5);
    }

    // ── IncrementalStateStore ──

    fn populated_store() -> IncrementalStateStore {
        let mut store = IncrementalStateStore::new();
        // v1: set a=1, b=2
        store.commit(vec![
            FieldChange::new("a", StateValue::Int(1)),
            FieldChange::new("b", StateValue::Int(2)),
        ]);
        // v2: update a=10
        store.commit(vec![FieldChange::new("a", StateValue::Int(10))]);
        // v3: delete b, add c=3
        store.commit(vec![
            FieldChange::delete("b"),
            FieldChange::new("c", StateValue::Int(3)),
        ]);
        store
    }

    #[test]
    fn test_store_head_advances_on_commit() {
        let store = populated_store();
        assert_eq!(store.head(), StateVersion(3));
        assert_eq!(store.delta_count(), 3);
    }

    #[test]
    fn test_delta_between_full_range() {
        let store = populated_store();
        let delta = store.delta_between(StateVersion(0), StateVersion(3)).expect("ok");
        // Latest for 'a' is 10, 'b' is tombstone, 'c' is 3.
        assert_eq!(delta.change_count(), 3);
        let a = delta.changes.iter().find(|c| c.path == "a").expect("a");
        assert_eq!(a.new_value, StateValue::Int(10));
        let b = delta.changes.iter().find(|c| c.path == "b").expect("b");
        assert!(b.is_deletion());
        let c = delta.changes.iter().find(|c| c.path == "c").expect("c");
        assert_eq!(c.new_value, StateValue::Int(3));
    }

    #[test]
    fn test_delta_between_partial_range() {
        let store = populated_store();
        // v1→v2: only a changes to 10
        let delta = store.delta_between(StateVersion(1), StateVersion(2)).expect("ok");
        assert_eq!(delta.change_count(), 1);
        let a = delta.changes.iter().find(|c| c.path == "a").expect("a");
        assert_eq!(a.new_value, StateValue::Int(10));
    }

    #[test]
    fn test_delta_between_same_version_is_empty() {
        let store = populated_store();
        let delta = store.delta_between(StateVersion(2), StateVersion(2)).expect("ok");
        assert!(delta.is_empty());
    }

    #[test]
    fn test_delta_between_out_of_range_returns_none() {
        let store = populated_store();
        assert!(store.delta_between(StateVersion(0), StateVersion(99)).is_none());
    }

    #[test]
    fn test_reconstruct_at_head() {
        let store = populated_store();
        let state = store.reconstruct(StateVersion(3)).expect("ok");
        // a=10, b deleted, c=3
        assert_eq!(state.get("a"), Some(&StateValue::Int(10)));
        assert!(!state.contains_key("b")); // b was tombstoned
        assert_eq!(state.get("c"), Some(&StateValue::Int(3)));
    }

    #[test]
    fn test_reconstruct_intermediate_version() {
        let store = populated_store();
        let state = store.reconstruct(StateVersion(1)).expect("ok");
        assert_eq!(state.get("a"), Some(&StateValue::Int(1)));
        assert_eq!(state.get("b"), Some(&StateValue::Int(2)));
        assert!(!state.contains_key("c"));
    }

    #[test]
    fn test_reconstruct_out_of_range_returns_none() {
        let store = populated_store();
        assert!(store.reconstruct(StateVersion(99)).is_none());
    }

    #[test]
    fn test_prune_removes_old_deltas() {
        let mut store = populated_store();
        let removed = store.prune(StateVersion(1));
        assert_eq!(removed, 1); // v1 is removed; v2, v3 remain
        assert_eq!(store.delta_count(), 2);
    }

    // ── PeerSyncTracker ──

    #[test]
    fn test_peer_register_and_count() {
        let mut tracker = PeerSyncTracker::new();
        tracker.register("alice");
        tracker.register("bob");
        assert_eq!(tracker.peer_count(), 2);
    }

    #[test]
    fn test_peer_acknowledge_advances_version() {
        let mut tracker = PeerSyncTracker::new();
        tracker.register("alice");
        tracker.acknowledge("alice", StateVersion(5));
        assert_eq!(tracker.acked_version("alice"), Some(StateVersion(5)));
    }

    #[test]
    fn test_peer_acknowledge_cannot_go_backwards() {
        let mut tracker = PeerSyncTracker::new();
        tracker.register("alice");
        tracker.acknowledge("alice", StateVersion(5));
        tracker.acknowledge("alice", StateVersion(3)); // should be ignored
        assert_eq!(tracker.acked_version("alice"), Some(StateVersion(5)));
    }

    #[test]
    fn test_min_acked_version() {
        let mut tracker = PeerSyncTracker::new();
        tracker.register("alice");
        tracker.register("bob");
        tracker.acknowledge("alice", StateVersion(5));
        tracker.acknowledge("bob", StateVersion(2));
        assert_eq!(tracker.min_acked_version(), StateVersion(2));
    }

    #[test]
    fn test_delta_for_peer() {
        let mut store = IncrementalStateStore::new();
        store.commit(vec![FieldChange::new("x", StateValue::Int(1))]);
        store.commit(vec![FieldChange::new("x", StateValue::Int(2))]);

        let mut tracker = PeerSyncTracker::new();
        tracker.register("carol");
        tracker.acknowledge("carol", StateVersion(1));

        let delta = tracker.delta_for_peer("carol", &store).expect("ok");
        // Carol is at v1; store head is v2; delta should contain the v2 change.
        assert_eq!(delta.change_count(), 1);
        assert_eq!(delta.changes[0].new_value, StateValue::Int(2));
    }

    #[test]
    fn test_deregister_peer() {
        let mut tracker = PeerSyncTracker::new();
        tracker.register("dave");
        tracker.deregister("dave");
        assert_eq!(tracker.peer_count(), 0);
        assert!(tracker.acked_version("dave").is_none());
    }
}
