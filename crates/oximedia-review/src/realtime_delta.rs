//! Delta-based real-time synchronisation for annotation updates.
//!
//! Instead of broadcasting the full annotation state on every change,
//! this module tracks a compact sequence of [`AnnotationDelta`] operations
//! and transmits only the mutations that occurred since each client's last
//! acknowledged sequence number.
//!
//! # Architecture
//!
//! ```text
//! Client A                 DeltaBroadcaster               Client B
//!   |-- patch op -------->  |                               |
//!                           |-- DeltaMessage(seq=42) -----> |
//!                           |                               | (ack seq=42)
//!                           |<-- AckMessage(seq=42) --------|
//! ```
//!
//! The [`DeltaLog`] stores an append-only ring of at most `capacity` entries.
//! Clients that fall behind by more than `capacity` entries receive a
//! `DeltaMessage::Resync` signal, instructing them to fetch the full snapshot.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Monotonically increasing sequence number for delta log entries.
pub type Seq = u64;

/// Identifier for a connected client.
pub type ClientId = String;

/// A single mutation applied to the annotation state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnnotationDelta {
    /// A new annotation was added.
    Add {
        /// Unique annotation identifier.
        annotation_id: String,
        /// Serialised annotation payload (e.g. JSON).
        payload: String,
        /// The user who performed the action.
        author: String,
    },
    /// An existing annotation was updated in-place.
    Update {
        /// Annotation being modified.
        annotation_id: String,
        /// Field that changed.
        field: String,
        /// New value (serialised).
        value: String,
        /// The user who performed the action.
        author: String,
    },
    /// An annotation was removed.
    Remove {
        /// Annotation being removed.
        annotation_id: String,
        /// The user who performed the action.
        author: String,
    },
    /// A comment was resolved.
    Resolve {
        /// Annotation / comment identifier.
        annotation_id: String,
        /// The user who resolved it.
        resolver: String,
    },
}

/// An entry in the delta log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaEntry {
    /// Log sequence number (monotonically increasing, 1-based).
    pub seq: Seq,
    /// When the mutation was recorded.
    pub timestamp: DateTime<Utc>,
    /// The mutation itself.
    pub delta: AnnotationDelta,
}

// ---------------------------------------------------------------------------
// DeltaLog
// ---------------------------------------------------------------------------

/// Append-only, bounded ring of delta entries.
///
/// Entries with sequence numbers older than `head_seq - capacity` are
/// evicted.  Clients that have fallen behind receive a `Resync` signal.
#[derive(Debug)]
pub struct DeltaLog {
    entries: Vec<DeltaEntry>,
    capacity: usize,
    next_seq: Seq,
}

impl DeltaLog {
    /// Create a new log with the given maximum capacity.
    ///
    /// `capacity` must be at least 1.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            next_seq: 1,
        }
    }

    /// Append a delta and return its assigned sequence number.
    pub fn push(&mut self, delta: AnnotationDelta) -> Seq {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries.push(DeltaEntry {
            seq,
            timestamp: Utc::now(),
            delta,
        });
        // Evict entries beyond capacity (keep the most recent `capacity` entries).
        if self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(..excess);
        }
        seq
    }

    /// Return the sequence number of the most recent entry, or 0 if empty.
    #[must_use]
    pub fn latest_seq(&self) -> Seq {
        self.entries.last().map(|e| e.seq).unwrap_or(0)
    }

    /// Return the sequence number of the oldest retained entry, or 0 if empty.
    #[must_use]
    pub fn oldest_seq(&self) -> Seq {
        self.entries.first().map(|e| e.seq).unwrap_or(0)
    }

    /// Return all entries with `seq > since`, or `None` if `since` is older
    /// than the oldest retained entry (signalling a required resync).
    #[must_use]
    pub fn since(&self, since: Seq) -> Option<&[DeltaEntry]> {
        if self.entries.is_empty() {
            return Some(&[]);
        }
        let oldest = self.oldest_seq();
        // If the client's last-ack is *before* the oldest retained entry it
        // cannot catch up incrementally — the caller must resync.
        if since > 0 && since < oldest.saturating_sub(1) {
            return None;
        }
        let pos = self.entries.partition_point(|e| e.seq <= since);
        Some(&self.entries[pos..])
    }

    /// Total number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Client cursor tracking
// ---------------------------------------------------------------------------

/// Tracks the last-acknowledged sequence number for each connected client.
#[derive(Debug, Default)]
pub struct ClientCursors {
    cursors: HashMap<ClientId, Seq>,
}

impl ClientCursors {
    /// Create an empty cursor map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new client at sequence `seq` (typically 0 for a fresh join).
    pub fn register(&mut self, client_id: impl Into<ClientId>, seq: Seq) {
        self.cursors.insert(client_id.into(), seq);
    }

    /// Record that `client_id` has acknowledged up to `seq`.
    ///
    /// Returns `false` if the client was not previously registered.
    pub fn ack(&mut self, client_id: &str, seq: Seq) -> bool {
        match self.cursors.get_mut(client_id) {
            Some(current) => {
                if seq > *current {
                    *current = seq;
                }
                true
            }
            None => false,
        }
    }

    /// Unregister a client (e.g. on disconnect).
    pub fn remove(&mut self, client_id: &str) {
        self.cursors.remove(client_id);
    }

    /// Return the last-acknowledged seq for `client_id`.
    #[must_use]
    pub fn cursor_for(&self, client_id: &str) -> Option<Seq> {
        self.cursors.get(client_id).copied()
    }

    /// Return the minimum cursor across all registered clients.
    ///
    /// Useful for safe log truncation: entries before this seq are not
    /// needed by any client.
    #[must_use]
    pub fn min_cursor(&self) -> Seq {
        self.cursors.values().copied().min().unwrap_or(0)
    }

    /// Number of registered clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.cursors.len()
    }
}

// ---------------------------------------------------------------------------
// DeltaBroadcaster
// ---------------------------------------------------------------------------

/// Outcome of a [`DeltaBroadcaster::fetch`] call.
#[derive(Debug, Clone)]
pub enum FetchResult<'a> {
    /// The client is up to date (no new entries).
    UpToDate,
    /// New entries the client has not yet seen.
    Deltas(&'a [DeltaEntry]),
    /// Client cursor is too old; it must fetch a full snapshot and re-register.
    ResyncRequired,
}

/// Coordinates delta generation and per-client delivery.
#[derive(Debug)]
pub struct DeltaBroadcaster {
    log: DeltaLog,
    cursors: ClientCursors,
}

impl DeltaBroadcaster {
    /// Create a broadcaster with the given log capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            log: DeltaLog::new(capacity),
            cursors: ClientCursors::new(),
        }
    }

    /// Apply a delta and notify the log; returns the assigned sequence number.
    pub fn apply(&mut self, delta: AnnotationDelta) -> Seq {
        self.log.push(delta)
    }

    /// Register a new client.  Use `seq = 0` for a client that has just
    /// performed a full snapshot fetch, or the client's last-known seq.
    pub fn connect(&mut self, client_id: impl Into<ClientId>, seq: Seq) {
        self.cursors.register(client_id, seq);
    }

    /// Disconnect a client.
    pub fn disconnect(&mut self, client_id: &str) {
        self.cursors.remove(client_id);
    }

    /// Fetch deltas for `client_id` since its last acknowledged sequence.
    pub fn fetch<'a>(&'a self, client_id: &str) -> FetchResult<'a> {
        let since = match self.cursors.cursor_for(client_id) {
            Some(s) => s,
            None => return FetchResult::ResyncRequired,
        };
        match self.log.since(since) {
            None => FetchResult::ResyncRequired,
            Some([]) => FetchResult::UpToDate,
            Some(entries) => FetchResult::Deltas(entries),
        }
    }

    /// Acknowledge that `client_id` has processed up to `seq`.
    ///
    /// Returns `false` if the client is unknown.
    pub fn ack(&mut self, client_id: &str, seq: Seq) -> bool {
        self.cursors.ack(client_id, seq)
    }

    /// Number of entries in the log.
    #[must_use]
    pub fn log_len(&self) -> usize {
        self.log.len()
    }

    /// Latest sequence number assigned.
    #[must_use]
    pub fn latest_seq(&self) -> Seq {
        self.log.latest_seq()
    }

    /// Number of currently connected clients.
    #[must_use]
    pub fn client_count(&self) -> usize {
        self.cursors.client_count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_add(id: &str) -> AnnotationDelta {
        AnnotationDelta::Add {
            annotation_id: id.to_string(),
            payload: format!("{{\"id\":\"{id}\"}}"),
            author: "alice".to_string(),
        }
    }

    fn make_update(id: &str) -> AnnotationDelta {
        AnnotationDelta::Update {
            annotation_id: id.to_string(),
            field: "text".to_string(),
            value: "updated".to_string(),
            author: "bob".to_string(),
        }
    }

    fn make_remove(id: &str) -> AnnotationDelta {
        AnnotationDelta::Remove {
            annotation_id: id.to_string(),
            author: "carol".to_string(),
        }
    }

    // 1. Empty log returns empty slice for seq=0.
    #[test]
    fn test_delta_log_empty_since_zero() {
        let log = DeltaLog::new(10);
        let result = log.since(0);
        assert!(matches!(result, Some(entries) if entries.is_empty()));
    }

    // 2. Pushing entries increments seq monotonically.
    #[test]
    fn test_delta_log_seq_increments() {
        let mut log = DeltaLog::new(10);
        let s1 = log.push(make_add("a1"));
        let s2 = log.push(make_add("a2"));
        let s3 = log.push(make_add("a3"));
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
        assert_eq!(log.latest_seq(), 3);
    }

    // 3. `since` returns only entries after the given cursor.
    #[test]
    fn test_delta_log_since_returns_tail() {
        let mut log = DeltaLog::new(10);
        log.push(make_add("a1"));
        log.push(make_add("a2"));
        log.push(make_add("a3"));
        let entries = log.since(1).expect("should return entries");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 2);
        assert_eq!(entries[1].seq, 3);
    }

    // 4. Capacity eviction triggers resync for stale clients.
    #[test]
    fn test_delta_log_eviction_triggers_resync() {
        let mut log = DeltaLog::new(3);
        for i in 0..5u32 {
            log.push(make_add(&i.to_string()));
        }
        // oldest seq retained should be 3 (5 - 3 + 1)
        assert_eq!(log.oldest_seq(), 3);
        // client that last saw seq=1 is now behind the oldest retained entry
        let result = log.since(1);
        assert!(result.is_none(), "stale client should get None (resync)");
    }

    // 5. Client cursor registration and ack.
    #[test]
    fn test_client_cursors_register_and_ack() {
        let mut cursors = ClientCursors::new();
        cursors.register("c1", 0);
        assert_eq!(cursors.cursor_for("c1"), Some(0));
        let ok = cursors.ack("c1", 5);
        assert!(ok);
        assert_eq!(cursors.cursor_for("c1"), Some(5));
    }

    // 6. Ack for unknown client returns false.
    #[test]
    fn test_client_cursors_ack_unknown_returns_false() {
        let mut cursors = ClientCursors::new();
        assert!(!cursors.ack("ghost", 10));
    }

    // 7. min_cursor returns lowest cursor across all clients.
    #[test]
    fn test_client_cursors_min_cursor() {
        let mut cursors = ClientCursors::new();
        cursors.register("c1", 5);
        cursors.register("c2", 10);
        cursors.register("c3", 3);
        assert_eq!(cursors.min_cursor(), 3);
    }

    // 8. DeltaBroadcaster: apply, connect, fetch, ack round-trip.
    #[test]
    fn test_broadcaster_full_round_trip() {
        let mut bc = DeltaBroadcaster::new(100);
        // Client connects with no prior history.
        bc.connect("alice", 0);
        // Producer pushes two deltas.
        let s1 = bc.apply(make_add("ann-1"));
        let s2 = bc.apply(make_update("ann-1"));
        // Fetch for alice should yield both deltas.
        let entries = match bc.fetch("alice") {
            FetchResult::Deltas(e) => e.to_vec(),
            other => panic!("expected Deltas, got {other:?}"),
        };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, s1);
        assert_eq!(entries[1].seq, s2);
        // Alice acks the latest.
        bc.ack("alice", s2);
        // Now fetch should return UpToDate.
        assert!(matches!(bc.fetch("alice"), FetchResult::UpToDate));
    }

    // 9. Disconnected client receives ResyncRequired.
    #[test]
    fn test_broadcaster_disconnected_client_resync() {
        let mut bc = DeltaBroadcaster::new(10);
        bc.connect("bob", 0);
        bc.disconnect("bob");
        assert!(matches!(bc.fetch("bob"), FetchResult::ResyncRequired));
    }

    // 10. Remove delta is recorded correctly.
    #[test]
    fn test_broadcaster_remove_delta() {
        let mut bc = DeltaBroadcaster::new(10);
        bc.connect("dave", 0);
        bc.apply(make_add("x"));
        let seq = bc.apply(make_remove("x"));
        let entries = match bc.fetch("dave") {
            FetchResult::Deltas(e) => e.to_vec(),
            other => panic!("expected Deltas, got {other:?}"),
        };
        let last = entries.last().expect("must have last entry");
        assert_eq!(last.seq, seq);
        assert!(matches!(
            &last.delta,
            AnnotationDelta::Remove { annotation_id, .. } if annotation_id == "x"
        ));
    }

    // 11. Resolve delta is serialisable.
    #[test]
    fn test_resolve_delta_serialize() {
        let delta = AnnotationDelta::Resolve {
            annotation_id: "ann-99".to_string(),
            resolver: "manager".to_string(),
        };
        let json = serde_json::to_string(&delta).expect("serialize");
        assert!(json.contains("ann-99"));
        let back: AnnotationDelta = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, delta);
    }

    // 12. log_len and client_count reflect state.
    #[test]
    fn test_broadcaster_metadata() {
        let mut bc = DeltaBroadcaster::new(50);
        bc.connect("u1", 0);
        bc.connect("u2", 0);
        assert_eq!(bc.client_count(), 2);
        bc.apply(make_add("z"));
        bc.apply(make_add("y"));
        assert_eq!(bc.log_len(), 2);
        assert_eq!(bc.latest_seq(), 2);
    }
}
