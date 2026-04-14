//! Real-time change feed for collaborative video editing sessions.
//!
//! Provides an ordered, append-only log of changes with:
//! * [`ChangeFeed`] — the core log that assigns monotonic sequence numbers.
//! * [`ChangeEntry`] — a typed change record with kind, author, resource, and payload.
//! * [`FeedSubscriber`] — a subscriber cursor that replays missed changes on reconnect.
//! * [`FeedSnapshot`] — a point-in-time snapshot of the feed for catch-up replay.
//!
//! # Design overview
//!
//! All changes are appended with a monotonically increasing sequence number
//! (`seq`).  Subscribers remember the last `seq` they have processed; on
//! reconnect they call [`ChangeFeed::since`] to get all missed entries and
//! replay them in order.  This avoids lost updates without requiring persistent
//! storage — the feed is purely in-memory and bounded by a configurable
//! capacity limit.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Change kinds
// ─────────────────────────────────────────────────────────────────────────────

/// The category of change recorded in the feed.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChangeKind {
    /// A clip was inserted into the timeline.
    ClipInserted,
    /// A clip was removed from the timeline.
    ClipRemoved,
    /// A clip attribute (position, duration, volume…) was modified.
    ClipModified,
    /// A track was added.
    TrackAdded,
    /// A track was removed.
    TrackRemoved,
    /// A track attribute was modified.
    TrackModified,
    /// An effect was applied to a clip or track.
    EffectApplied,
    /// An effect was removed.
    EffectRemoved,
    /// A text/metadata field was updated.
    MetadataChanged,
    /// A user joined the session.
    UserJoined,
    /// A user left the session.
    UserLeft,
    /// A lock was acquired on a resource.
    LockAcquired,
    /// A lock was released.
    LockReleased,
    /// A custom application-defined change.
    Custom(String),
}

impl std::fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::ClipInserted => "clip_inserted",
            Self::ClipRemoved => "clip_removed",
            Self::ClipModified => "clip_modified",
            Self::TrackAdded => "track_added",
            Self::TrackRemoved => "track_removed",
            Self::TrackModified => "track_modified",
            Self::EffectApplied => "effect_applied",
            Self::EffectRemoved => "effect_removed",
            Self::MetadataChanged => "metadata_changed",
            Self::UserJoined => "user_joined",
            Self::UserLeft => "user_left",
            Self::LockAcquired => "lock_acquired",
            Self::LockReleased => "lock_released",
            Self::Custom(s) => s.as_str(),
        };
        write!(f, "{s}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Change entry
// ─────────────────────────────────────────────────────────────────────────────

/// A single entry in the change feed.
#[derive(Debug, Clone)]
pub struct ChangeEntry {
    /// Monotonically increasing sequence number assigned by the feed.
    pub seq: u64,
    /// Category of this change.
    pub kind: ChangeKind,
    /// Identifier of the user who made the change.
    pub author_id: String,
    /// Identifier of the resource that was changed (e.g. `"clip:42"`).
    pub resource_id: String,
    /// Wall-clock timestamp in milliseconds (Unix epoch).
    pub timestamp_ms: u64,
    /// Optional free-form payload (e.g. serialised diff, new value).
    pub payload: Option<String>,
    /// Optional key-value tags for routing or filtering.
    pub tags: HashMap<String, String>,
}

impl ChangeEntry {
    /// Convenience constructor.
    pub fn new(
        seq: u64,
        kind: ChangeKind,
        author_id: impl Into<String>,
        resource_id: impl Into<String>,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            seq,
            kind,
            author_id: author_id.into(),
            resource_id: resource_id.into(),
            timestamp_ms,
            payload: None,
            tags: HashMap::new(),
        }
    }

    /// Attach a free-form payload string.
    pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
        self.payload = Some(payload.into());
        self
    }

    /// Attach a key-value tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

impl std::fmt::Display for ChangeEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "seq={} kind={} author={} resource={} ts={}",
            self.seq, self.kind, self.author_id, self.resource_id, self.timestamp_ms
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feed error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the change feed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeFeedError {
    /// The requested sequence number is older than the oldest entry retained.
    SequenceTooOld { requested: u64, oldest: u64 },
    /// The subscriber id is not registered with this feed.
    UnknownSubscriber(String),
    /// The feed capacity is zero, which is not a valid configuration.
    InvalidCapacity,
}

impl std::fmt::Display for ChangeFeedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceTooOld { requested, oldest } => write!(
                f,
                "sequence {requested} is older than oldest retained {oldest}"
            ),
            Self::UnknownSubscriber(id) => write!(f, "unknown subscriber: {id}"),
            Self::InvalidCapacity => write!(f, "feed capacity must be > 0"),
        }
    }
}

impl std::error::Error for ChangeFeedError {}

// ─────────────────────────────────────────────────────────────────────────────
// Feed snapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of a contiguous range of changes.
///
/// Clients that need to catch up can request a `FeedSnapshot` and replay
/// the entries in `seq` order.
#[derive(Debug, Clone)]
pub struct FeedSnapshot {
    /// The first sequence number in this snapshot (inclusive).
    pub from_seq: u64,
    /// The last sequence number in this snapshot (inclusive).
    pub to_seq: u64,
    /// Entries in ascending sequence order.
    pub entries: Vec<ChangeEntry>,
}

impl FeedSnapshot {
    /// Whether the snapshot contains any entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of entries in the snapshot.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feed subscriber cursor
// ─────────────────────────────────────────────────────────────────────────────

/// A subscriber cursor that tracks the last sequence number consumed.
///
/// On reconnect, call [`ChangeFeed::catch_up`] with the subscriber id to
/// retrieve all entries missed since the cursor.
#[derive(Debug, Clone)]
pub struct FeedSubscriber {
    /// Unique identifier for this subscriber.
    pub id: String,
    /// Identifier of the associated user.
    pub user_id: String,
    /// Last sequence number successfully processed by this subscriber.
    /// `None` means the subscriber has not processed any entry yet.
    pub last_seq: Option<u64>,
    /// Whether the subscriber is currently connected.
    pub connected: bool,
}

impl FeedSubscriber {
    /// Create a new subscriber with no history.
    pub fn new(id: impl Into<String>, user_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            user_id: user_id.into(),
            last_seq: None,
            connected: true,
        }
    }

    /// Mark the subscriber as disconnected.
    pub fn disconnect(&mut self) {
        self.connected = false;
    }

    /// Mark the subscriber as reconnected.
    pub fn reconnect(&mut self) {
        self.connected = true;
    }

    /// Advance the cursor to `seq`.
    pub fn advance(&mut self, seq: u64) {
        self.last_seq = Some(seq);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Change feed
// ─────────────────────────────────────────────────────────────────────────────

/// Ordered, append-only change log with subscriber tracking.
///
/// Entries are retained up to `capacity` entries; older entries are evicted
/// using a ring-buffer approach when the limit is reached.
#[derive(Debug)]
pub struct ChangeFeed {
    /// All retained entries in ascending sequence order.
    entries: Vec<ChangeEntry>,
    /// Sequence number to assign to the next append.
    next_seq: u64,
    /// Maximum number of entries to retain (0 means unlimited).
    capacity: usize,
    /// Registered subscribers, keyed by subscriber id.
    subscribers: HashMap<String, FeedSubscriber>,
}

impl ChangeFeed {
    /// Create a new change feed with a capacity limit.
    ///
    /// Returns `Err(ChangeFeedError::InvalidCapacity)` when `capacity` is 0.
    pub fn new(capacity: usize) -> Result<Self, ChangeFeedError> {
        if capacity == 0 {
            return Err(ChangeFeedError::InvalidCapacity);
        }
        Ok(Self {
            entries: Vec::new(),
            next_seq: 1,
            capacity,
            subscribers: HashMap::new(),
        })
    }

    /// Create a new change feed with no capacity limit.
    pub fn unbounded() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 1,
            capacity: 0,
            subscribers: HashMap::new(),
        }
    }

    /// Append a change to the feed and return the assigned sequence number.
    pub fn append(
        &mut self,
        kind: ChangeKind,
        author_id: impl Into<String>,
        resource_id: impl Into<String>,
        timestamp_ms: u64,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = ChangeEntry::new(seq, kind, author_id, resource_id, timestamp_ms);
        self.entries.push(entry);
        self.enforce_capacity();
        seq
    }

    /// Append a fully built `ChangeEntry` (its `seq` field will be
    /// overwritten with the assigned sequence number).
    pub fn append_entry(&mut self, mut entry: ChangeEntry) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        entry.seq = seq;
        self.entries.push(entry);
        self.enforce_capacity();
        seq
    }

    /// Return all entries with `seq > after_seq`, in ascending order.
    ///
    /// Returns `Err(ChangeFeedError::SequenceTooOld)` when `after_seq` is
    /// older than the oldest retained entry (and the feed is non-empty).
    pub fn since(&self, after_seq: u64) -> Result<Vec<&ChangeEntry>, ChangeFeedError> {
        if let Some(oldest) = self.entries.first() {
            // The caller is asking for entries starting at `after_seq + 1`.
            // If the oldest retained entry is newer than that, the gap cannot
            // be filled.
            if oldest.seq > after_seq + 1 {
                return Err(ChangeFeedError::SequenceTooOld {
                    requested: after_seq + 1,
                    oldest: oldest.seq,
                });
            }
        }
        Ok(self.entries.iter().filter(|e| e.seq > after_seq).collect())
    }

    /// Return a snapshot of all entries in the range `[from_seq, to_seq]`.
    pub fn snapshot(&self, from_seq: u64, to_seq: u64) -> FeedSnapshot {
        let entries: Vec<ChangeEntry> = self
            .entries
            .iter()
            .filter(|e| e.seq >= from_seq && e.seq <= to_seq)
            .cloned()
            .collect();
        FeedSnapshot {
            from_seq,
            to_seq,
            entries,
        }
    }

    /// Return the latest `n` entries (or fewer if the feed has fewer).
    pub fn latest(&self, n: usize) -> Vec<&ChangeEntry> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].iter().collect()
    }

    // ── Subscriber management ──────────────────────────────────────────────

    /// Register a new subscriber.  Returns the subscriber id.
    pub fn subscribe(&mut self, sub: FeedSubscriber) -> String {
        let id = sub.id.clone();
        self.subscribers.insert(id.clone(), sub);
        id
    }

    /// Unregister a subscriber.  Returns `true` if found.
    pub fn unsubscribe(&mut self, id: &str) -> bool {
        self.subscribers.remove(id).is_some()
    }

    /// Retrieve all entries the subscriber has not yet seen and advance its
    /// cursor to the latest sequence number.
    pub fn catch_up(&mut self, subscriber_id: &str) -> Result<Vec<ChangeEntry>, ChangeFeedError> {
        let sub = self
            .subscribers
            .get(subscriber_id)
            .ok_or_else(|| ChangeFeedError::UnknownSubscriber(subscriber_id.to_string()))?;

        let last_seq = sub.last_seq.unwrap_or(0);
        let missed: Result<Vec<&ChangeEntry>, ChangeFeedError> = self.since(last_seq);

        let entries: Vec<ChangeEntry> = missed?.iter().map(|e| (*e).clone()).collect();

        // Advance the cursor to the highest seq we're returning.
        if let Some(last) = entries.last() {
            let new_seq = last.seq;
            if let Some(sub_mut) = self.subscribers.get_mut(subscriber_id) {
                sub_mut.advance(new_seq);
            }
        }
        Ok(entries)
    }

    /// Mark a subscriber as disconnected.
    pub fn disconnect_subscriber(&mut self, id: &str) -> Result<(), ChangeFeedError> {
        self.subscribers
            .get_mut(id)
            .ok_or_else(|| ChangeFeedError::UnknownSubscriber(id.to_string()))
            .map(|s| s.disconnect())
    }

    /// Mark a subscriber as reconnected.
    pub fn reconnect_subscriber(&mut self, id: &str) -> Result<(), ChangeFeedError> {
        self.subscribers
            .get_mut(id)
            .ok_or_else(|| ChangeFeedError::UnknownSubscriber(id.to_string()))
            .map(|s| s.reconnect())
    }

    // ── Metadata ───────────────────────────────────────────────────────────

    /// Total number of entries in the feed (retained).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the feed has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The sequence number that will be assigned to the next entry.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// The oldest retained sequence number, or `None` if the feed is empty.
    pub fn oldest_seq(&self) -> Option<u64> {
        self.entries.first().map(|e| e.seq)
    }

    /// The newest retained sequence number, or `None` if the feed is empty.
    pub fn newest_seq(&self) -> Option<u64> {
        self.entries.last().map(|e| e.seq)
    }

    /// Number of registered subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn enforce_capacity(&mut self) {
        if self.capacity > 0 && self.entries.len() > self.capacity {
            let excess = self.entries.len() - self.capacity;
            self.entries.drain(0..excess);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feed() -> ChangeFeed {
        ChangeFeed::unbounded()
    }

    fn append_n(feed: &mut ChangeFeed, n: u64) {
        for i in 0..n {
            feed.append(
                ChangeKind::ClipModified,
                "user:1",
                format!("clip:{i}"),
                i * 1000,
            );
        }
    }

    // ── Append / basic ─────────────────────────────────────────────────────

    #[test]
    fn test_append_assigns_incrementing_seqs() {
        let mut feed = make_feed();
        let s1 = feed.append(ChangeKind::ClipInserted, "alice", "clip:1", 1000);
        let s2 = feed.append(ChangeKind::ClipModified, "alice", "clip:1", 2000);
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
    }

    #[test]
    fn test_feed_len_and_is_empty() {
        let mut feed = make_feed();
        assert!(feed.is_empty());
        feed.append(ChangeKind::TrackAdded, "bob", "track:1", 0);
        assert_eq!(feed.len(), 1);
        assert!(!feed.is_empty());
    }

    #[test]
    fn test_newest_oldest_seq() {
        let mut feed = make_feed();
        assert_eq!(feed.oldest_seq(), None);
        assert_eq!(feed.newest_seq(), None);
        append_n(&mut feed, 3);
        assert_eq!(feed.oldest_seq(), Some(1));
        assert_eq!(feed.newest_seq(), Some(3));
    }

    // ── Capacity eviction ──────────────────────────────────────────────────

    #[test]
    fn test_capacity_evicts_oldest() {
        let mut feed = ChangeFeed::new(3).expect("capacity should be valid");
        append_n(&mut feed, 5);
        // capacity=3 → only entries 3,4,5 retained
        assert_eq!(feed.len(), 3);
        assert_eq!(feed.oldest_seq(), Some(3));
        assert_eq!(feed.newest_seq(), Some(5));
    }

    #[test]
    fn test_zero_capacity_is_error() {
        let result = ChangeFeed::new(0);
        assert!(result.is_err());
        assert!(matches!(result, Err(ChangeFeedError::InvalidCapacity)));
    }

    // ── since ──────────────────────────────────────────────────────────────

    #[test]
    fn test_since_returns_entries_after_seq() {
        let mut feed = make_feed();
        append_n(&mut feed, 5);
        let entries = feed.since(3).expect("since should succeed");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 4);
        assert_eq!(entries[1].seq, 5);
    }

    #[test]
    fn test_since_zero_returns_all() {
        let mut feed = make_feed();
        append_n(&mut feed, 4);
        let entries = feed.since(0).expect("since should succeed");
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn test_since_evicted_seq_returns_error() {
        let mut feed = ChangeFeed::new(3).expect("capacity should be valid");
        append_n(&mut feed, 5); // entries 3,4,5 retained; 1,2 evicted
        // Asking for entries after seq=0 means wanting seq 1+ but oldest is 3
        let result = feed.since(0);
        assert!(matches!(
            result,
            Err(ChangeFeedError::SequenceTooOld { .. })
        ));
    }

    // ── Snapshot ───────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_returns_range() {
        let mut feed = make_feed();
        append_n(&mut feed, 5);
        let snap = feed.snapshot(2, 4);
        assert_eq!(snap.len(), 3);
        assert_eq!(snap.entries[0].seq, 2);
        assert_eq!(snap.entries[2].seq, 4);
    }

    #[test]
    fn test_snapshot_empty_range() {
        let feed = make_feed();
        let snap = feed.snapshot(10, 20);
        assert!(snap.is_empty());
    }

    // ── Latest ─────────────────────────────────────────────────────────────

    #[test]
    fn test_latest_returns_n_newest() {
        let mut feed = make_feed();
        append_n(&mut feed, 6);
        let latest = feed.latest(3);
        assert_eq!(latest.len(), 3);
        assert_eq!(latest[0].seq, 4);
        assert_eq!(latest[2].seq, 6);
    }

    // ── Subscriber catch-up ────────────────────────────────────────────────

    #[test]
    fn test_catch_up_returns_missed_entries() {
        let mut feed = make_feed();
        let sub = FeedSubscriber::new("sub:1", "user:1");
        feed.subscribe(sub);

        // Append 3 entries before subscriber catches up.
        append_n(&mut feed, 3);

        let missed = feed.catch_up("sub:1").expect("catch-up should succeed");
        assert_eq!(missed.len(), 3);
        // Cursor should now be at seq 3.
        let sub = feed.subscribers.get("sub:1").expect("should exist");
        assert_eq!(sub.last_seq, Some(3));
    }

    #[test]
    fn test_catch_up_unknown_subscriber_returns_error() {
        let mut feed = make_feed();
        let result = feed.catch_up("ghost:99");
        assert!(matches!(
            result,
            Err(ChangeFeedError::UnknownSubscriber(_))
        ));
    }

    #[test]
    fn test_subscribe_and_unsubscribe() {
        let mut feed = make_feed();
        feed.subscribe(FeedSubscriber::new("sub:1", "user:1"));
        assert_eq!(feed.subscriber_count(), 1);
        assert!(feed.unsubscribe("sub:1"));
        assert_eq!(feed.subscriber_count(), 0);
        assert!(!feed.unsubscribe("sub:1")); // second removal → false
    }

    #[test]
    fn test_disconnect_and_reconnect_subscriber() {
        let mut feed = make_feed();
        feed.subscribe(FeedSubscriber::new("sub:1", "user:1"));
        feed.disconnect_subscriber("sub:1").expect("should succeed");
        let sub = feed.subscribers.get("sub:1").expect("should exist");
        assert!(!sub.connected);
        feed.reconnect_subscriber("sub:1").expect("should succeed");
        let sub = feed.subscribers.get("sub:1").expect("should exist");
        assert!(sub.connected);
    }

    // ── ChangeKind display ─────────────────────────────────────────────────

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::ClipInserted.to_string(), "clip_inserted");
        assert_eq!(ChangeKind::MetadataChanged.to_string(), "metadata_changed");
        assert_eq!(
            ChangeKind::Custom("my_event".to_string()).to_string(),
            "my_event"
        );
    }

    // ── ChangeEntry builder ────────────────────────────────────────────────

    #[test]
    fn test_entry_builder_payload_and_tag() {
        let entry = ChangeEntry::new(1, ChangeKind::ClipInserted, "alice", "clip:7", 9999)
            .with_payload(r#"{"pos": 42}"#)
            .with_tag("session", "s1");
        assert_eq!(entry.payload, Some(r#"{"pos": 42}"#.to_string()));
        assert_eq!(entry.tags.get("session"), Some(&"s1".to_string()));
    }
}
