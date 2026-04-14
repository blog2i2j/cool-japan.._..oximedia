#![allow(dead_code)]
//! Route audit trail for tracking routing changes over time.
//!
//! Every configuration change (connect, disconnect, gain change, preset
//! recall, etc.) is recorded as an [`AuditEntry`] in an [`AuditLog`].
//! The log supports querying by time range, action type, and source/dest
//! identifiers, and can generate summary reports.
//!
//! In addition to the event log, this module provides [`RoutingSnapshot`] for
//! capturing a point-in-time view of all active routing connections and
//! [`SnapshotDiff`] for comparing two snapshots to understand what changed.

use std::fmt;

/// Kind of routing action that was performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditAction {
    /// A new route was connected.
    Connect,
    /// A route was disconnected.
    Disconnect,
    /// Gain was changed on an existing route.
    GainChange,
    /// A preset was recalled.
    PresetRecall,
    /// A preset was saved.
    PresetSave,
    /// A route was muted.
    Mute,
    /// A route was un-muted.
    Unmute,
    /// A failover event occurred.
    Failover,
}

impl fmt::Display for AuditAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect => write!(f, "CONNECT"),
            Self::Disconnect => write!(f, "DISCONNECT"),
            Self::GainChange => write!(f, "GAIN_CHANGE"),
            Self::PresetRecall => write!(f, "PRESET_RECALL"),
            Self::PresetSave => write!(f, "PRESET_SAVE"),
            Self::Mute => write!(f, "MUTE"),
            Self::Unmute => write!(f, "UNMUTE"),
            Self::Failover => write!(f, "FAILOVER"),
        }
    }
}

/// A single audit log entry.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Timestamp in microseconds since log creation.
    pub timestamp_us: u64,
    /// The action that was performed.
    pub action: AuditAction,
    /// Source id involved (if applicable).
    pub source: Option<u32>,
    /// Destination id involved (if applicable).
    pub destination: Option<u32>,
    /// Human-readable detail string.
    pub detail: String,
    /// User / operator who triggered the action.
    pub user: String,
}

impl AuditEntry {
    /// Create a new audit entry.
    pub fn new(seq: u64, timestamp_us: u64, action: AuditAction, detail: &str, user: &str) -> Self {
        Self {
            seq,
            timestamp_us,
            action,
            source: None,
            destination: None,
            detail: detail.to_owned(),
            user: user.to_owned(),
        }
    }

    /// Set source and destination ids.
    pub fn with_route(mut self, source: u32, destination: u32) -> Self {
        self.source = Some(source);
        self.destination = Some(destination);
        self
    }

    /// Set source id only.
    pub fn with_source(mut self, source: u32) -> Self {
        self.source = Some(source);
        self
    }

    /// Timestamp in milliseconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn timestamp_ms(&self) -> f64 {
        self.timestamp_us as f64 / 1_000.0
    }
}

impl fmt::Display for AuditEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:06}] {:.3}ms {} by '{}': {}",
            self.seq,
            self.timestamp_ms(),
            self.action,
            self.user,
            self.detail,
        )
    }
}

/// Query filter for searching the audit log.
#[derive(Debug, Clone, Default)]
pub struct AuditQuery {
    /// Filter by action type.
    pub action: Option<AuditAction>,
    /// Filter by source id.
    pub source: Option<u32>,
    /// Filter by destination id.
    pub destination: Option<u32>,
    /// Filter by user name (exact match).
    pub user: Option<String>,
    /// Start timestamp (inclusive).
    pub from_us: Option<u64>,
    /// End timestamp (inclusive).
    pub to_us: Option<u64>,
    /// Maximum number of results.
    pub limit: Option<usize>,
}

impl AuditQuery {
    /// Create an empty query (matches everything).
    pub fn all() -> Self {
        Self::default()
    }

    /// Filter by action.
    pub fn with_action(mut self, action: AuditAction) -> Self {
        self.action = Some(action);
        self
    }

    /// Filter by source.
    pub fn with_source(mut self, source: u32) -> Self {
        self.source = Some(source);
        self
    }

    /// Filter by destination.
    pub fn with_destination(mut self, dest: u32) -> Self {
        self.destination = Some(dest);
        self
    }

    /// Filter by user.
    pub fn with_user(mut self, user: &str) -> Self {
        self.user = Some(user.to_owned());
        self
    }

    /// Filter by time range.
    pub fn with_time_range(mut self, from_us: u64, to_us: u64) -> Self {
        self.from_us = Some(from_us);
        self.to_us = Some(to_us);
        self
    }

    /// Limit results.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check whether a single entry matches this query.
    pub fn matches(&self, entry: &AuditEntry) -> bool {
        if let Some(a) = self.action {
            if entry.action != a {
                return false;
            }
        }
        if let Some(s) = self.source {
            if entry.source != Some(s) {
                return false;
            }
        }
        if let Some(d) = self.destination {
            if entry.destination != Some(d) {
                return false;
            }
        }
        if let Some(ref u) = self.user {
            if entry.user != *u {
                return false;
            }
        }
        if let Some(from) = self.from_us {
            if entry.timestamp_us < from {
                return false;
            }
        }
        if let Some(to) = self.to_us {
            if entry.timestamp_us > to {
                return false;
            }
        }
        true
    }
}

/// Per-action count summary.
#[derive(Debug, Clone)]
pub struct ActionCount {
    /// Action type.
    pub action: AuditAction,
    /// Number of occurrences.
    pub count: usize,
}

/// The audit log.
#[derive(Debug)]
pub struct AuditLog {
    /// All entries in chronological order.
    entries: Vec<AuditEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Maximum entries to retain (0 = unlimited).
    max_entries: usize,
}

impl AuditLog {
    /// Create a new empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
            max_entries: 0,
        }
    }

    /// Create with a maximum entry limit. Oldest entries are evicted first.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 0,
            max_entries,
        }
    }

    /// Record a new entry. Returns the sequence number.
    pub fn record(
        &mut self,
        timestamp_us: u64,
        action: AuditAction,
        detail: &str,
        user: &str,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.entries
            .push(AuditEntry::new(seq, timestamp_us, action, detail, user));

        // Evict oldest if needed
        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        seq
    }

    /// Record an entry with route information.
    pub fn record_route(
        &mut self,
        timestamp_us: u64,
        action: AuditAction,
        source: u32,
        destination: u32,
        detail: &str,
        user: &str,
    ) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let entry = AuditEntry::new(seq, timestamp_us, action, detail, user)
            .with_route(source, destination);
        self.entries.push(entry);

        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        seq
    }

    /// Total entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get an entry by sequence number.
    pub fn get(&self, seq: u64) -> Option<&AuditEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Query the log with a filter.
    pub fn query(&self, q: &AuditQuery) -> Vec<&AuditEntry> {
        let mut results: Vec<&AuditEntry> = self.entries.iter().filter(|e| q.matches(e)).collect();
        if let Some(limit) = q.limit {
            results.truncate(limit);
        }
        results
    }

    /// Count entries per action type.
    pub fn action_counts(&self) -> Vec<ActionCount> {
        use std::collections::HashMap;
        let mut map: HashMap<AuditAction, usize> = HashMap::new();
        for e in &self.entries {
            *map.entry(e.action).or_default() += 1;
        }
        let mut out: Vec<ActionCount> = map
            .into_iter()
            .map(|(action, count)| ActionCount { action, count })
            .collect();
        out.sort_by(|a, b| b.count.cmp(&a.count));
        out
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Most recent entry.
    pub fn last_entry(&self) -> Option<&AuditEntry> {
        self.entries.last()
    }

    /// Generate a text report of the log.
    pub fn report(&self) -> String {
        let mut lines = vec![format!("Audit Log ({} entries)", self.entries.len())];
        for entry in &self.entries {
            lines.push(format!("  {entry}"));
        }
        lines.join("\n")
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RoutingSnapshot — a point-in-time capture of the full routing state
// ---------------------------------------------------------------------------

/// A connection stored in a routing snapshot.
///
/// Gain is stored as an integer (dBFS × 100) to allow exact equality
/// comparison without floating-point rounding surprises.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SnapshotConnection {
    /// Source (input) index.
    pub source: u32,
    /// Destination (output) index.
    pub destination: u32,
    /// Gain in dBFS × 100 (i.e. −6 dB stored as −600).
    pub gain_db_x100: i32,
}

impl SnapshotConnection {
    /// Creates a new connection record from floating-point dB.
    pub fn new(source: u32, destination: u32, gain_db: f32) -> Self {
        Self {
            source,
            destination,
            gain_db_x100: (gain_db * 100.0).round() as i32,
        }
    }

    /// Returns the gain as a floating-point dB value.
    pub fn gain_db(&self) -> f32 {
        self.gain_db_x100 as f32 / 100.0
    }
}

/// Metadata stored alongside a routing snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotMeta {
    /// Human-readable label.
    pub label: String,
    /// Snapshot timestamp in microseconds.
    pub timestamp_us: u64,
    /// Operator who created the snapshot.
    pub user: String,
}

/// A point-in-time capture of all active routing connections and their gains.
///
/// Snapshots can be compared with [`RoutingSnapshot::diff`] to produce a
/// [`SnapshotDiff`] that describes exactly what changed between the two states.
#[derive(Debug, Clone)]
pub struct RoutingSnapshot {
    /// Metadata for this snapshot.
    pub meta: SnapshotMeta,
    /// Sorted list of active connections.
    connections: Vec<SnapshotConnection>,
}

impl RoutingSnapshot {
    /// Creates an empty snapshot.
    pub fn new(label: impl Into<String>, timestamp_us: u64, user: impl Into<String>) -> Self {
        Self {
            meta: SnapshotMeta {
                label: label.into(),
                timestamp_us,
                user: user.into(),
            },
            connections: Vec::new(),
        }
    }

    /// Inserts a connection into the snapshot.
    ///
    /// If a connection with the same `(source, destination)` already exists it
    /// is replaced; otherwise the new connection is inserted and the list is
    /// re-sorted.
    pub fn insert(&mut self, conn: SnapshotConnection) {
        if let Some(existing) = self
            .connections
            .iter_mut()
            .find(|c| c.source == conn.source && c.destination == conn.destination)
        {
            *existing = conn;
        } else {
            self.connections.push(conn);
            self.connections.sort();
        }
    }

    /// Convenience wrapper around [`Self::insert`] that builds the connection.
    pub fn add(&mut self, source: u32, destination: u32, gain_db: f32) {
        self.insert(SnapshotConnection::new(source, destination, gain_db));
    }

    /// Returns the sorted slice of all connections.
    pub fn connections(&self) -> &[SnapshotConnection] {
        &self.connections
    }

    /// Number of active connections in this snapshot.
    pub fn len(&self) -> usize {
        self.connections.len()
    }

    /// Returns `true` if the snapshot has no connections.
    pub fn is_empty(&self) -> bool {
        self.connections.is_empty()
    }

    /// Returns the connection for the given source/destination pair, if any.
    pub fn find(&self, source: u32, destination: u32) -> Option<&SnapshotConnection> {
        self.connections
            .iter()
            .find(|c| c.source == source && c.destination == destination)
    }

    /// Computes a diff between `self` (the "before" snapshot) and `other`
    /// (the "after" snapshot).
    ///
    /// - **Added**: connections present in `other` but absent in `self`.
    /// - **Removed**: connections present in `self` but absent in `other`.
    /// - **Changed**: connections present in both, but with a different gain.
    pub fn diff<'a>(&'a self, other: &'a RoutingSnapshot) -> SnapshotDiff<'a> {
        let mut added: Vec<&'a SnapshotConnection> = Vec::new();
        let mut removed: Vec<&'a SnapshotConnection> = Vec::new();
        let mut changed: Vec<GainChanged> = Vec::new();

        for after_conn in &other.connections {
            match self.find(after_conn.source, after_conn.destination) {
                None => added.push(after_conn),
                Some(before_conn) => {
                    if before_conn.gain_db_x100 != after_conn.gain_db_x100 {
                        changed.push(GainChanged {
                            source: after_conn.source,
                            destination: after_conn.destination,
                            before_gain_db: before_conn.gain_db(),
                            after_gain_db: after_conn.gain_db(),
                        });
                    }
                }
            }
        }

        for before_conn in &self.connections {
            if other
                .find(before_conn.source, before_conn.destination)
                .is_none()
            {
                removed.push(before_conn);
            }
        }

        SnapshotDiff {
            before: self,
            after: other,
            added,
            removed,
            changed,
        }
    }
}

// ---------------------------------------------------------------------------
// GainChanged — a gain change record within a diff
// ---------------------------------------------------------------------------

/// Records a gain change on an existing connection between two snapshots.
#[derive(Debug, Clone, PartialEq)]
pub struct GainChanged {
    /// Source index.
    pub source: u32,
    /// Destination index.
    pub destination: u32,
    /// Gain before the change in dB.
    pub before_gain_db: f32,
    /// Gain after the change in dB.
    pub after_gain_db: f32,
}

impl GainChanged {
    /// Returns the change delta (after − before) in dB.
    pub fn delta_db(&self) -> f32 {
        self.after_gain_db - self.before_gain_db
    }
}

// ---------------------------------------------------------------------------
// SnapshotDiff — result of comparing two snapshots
// ---------------------------------------------------------------------------

/// The result of diffing two [`RoutingSnapshot`]s.
///
/// Captures which connections were **added**, **removed**, and which had their
/// **gain changed** between the before and after states.
pub struct SnapshotDiff<'a> {
    /// The snapshot taken before the change.
    pub before: &'a RoutingSnapshot,
    /// The snapshot taken after the change.
    pub after: &'a RoutingSnapshot,
    /// Connections present in `after` that were absent in `before`.
    pub added: Vec<&'a SnapshotConnection>,
    /// Connections present in `before` that are absent in `after`.
    pub removed: Vec<&'a SnapshotConnection>,
    /// Connections present in both but with different gain.
    pub changed: Vec<GainChanged>,
}

impl SnapshotDiff<'_> {
    /// Returns `true` if there are no differences between the two snapshots.
    pub fn is_identical(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }

    /// Total number of changes (added + removed + gain-changed).
    pub fn change_count(&self) -> usize {
        self.added.len() + self.removed.len() + self.changed.len()
    }

    /// Generates a human-readable diff report.
    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Routing diff: '{}' \u{2192} '{}' ({} changes)",
            self.before.meta.label,
            self.after.meta.label,
            self.change_count()
        ));
        for conn in &self.added {
            lines.push(format!(
                "  + ADDED   src={} dst={} gain={:.1} dB",
                conn.source,
                conn.destination,
                conn.gain_db()
            ));
        }
        for conn in &self.removed {
            lines.push(format!(
                "  - REMOVED src={} dst={} gain={:.1} dB",
                conn.source,
                conn.destination,
                conn.gain_db()
            ));
        }
        for change in &self.changed {
            lines.push(format!(
                "  ~ CHANGED src={} dst={} {:.1}\u{2192}{:.1} dB (\u{0394}{:.1})",
                change.source,
                change.destination,
                change.before_gain_db,
                change.after_gain_db,
                change.delta_db()
            ));
        }
        lines.join("\n")
    }

    /// Records each change in this diff into the supplied audit log.
    ///
    /// Added connections are recorded as [`AuditAction::Connect`], removed as
    /// [`AuditAction::Disconnect`], and gain changes as
    /// [`AuditAction::GainChange`].
    pub fn record_to_log(&self, log: &mut AuditLog, timestamp_us: u64, user: &str) {
        for conn in &self.added {
            log.record_route(
                timestamp_us,
                AuditAction::Connect,
                conn.source,
                conn.destination,
                &format!(
                    "snapshot diff: added src={} dst={}",
                    conn.source, conn.destination
                ),
                user,
            );
        }
        for conn in &self.removed {
            log.record_route(
                timestamp_us,
                AuditAction::Disconnect,
                conn.source,
                conn.destination,
                &format!(
                    "snapshot diff: removed src={} dst={}",
                    conn.source, conn.destination
                ),
                user,
            );
        }
        for change in &self.changed {
            log.record_route(
                timestamp_us,
                AuditAction::GainChange,
                change.source,
                change.destination,
                &format!(
                    "snapshot diff: gain {:.1}\u{2192}{:.1} dB",
                    change.before_gain_db, change.after_gain_db
                ),
                user,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_new() {
        let e = AuditEntry::new(0, 1000, AuditAction::Connect, "connected", "admin");
        assert_eq!(e.seq, 0);
        assert_eq!(e.action, AuditAction::Connect);
        assert_eq!(e.user, "admin");
    }

    #[test]
    fn test_entry_with_route() {
        let e = AuditEntry::new(0, 0, AuditAction::Connect, "route", "op").with_route(1, 2);
        assert_eq!(e.source, Some(1));
        assert_eq!(e.destination, Some(2));
    }

    #[test]
    fn test_entry_timestamp_ms() {
        let e = AuditEntry::new(0, 5_000, AuditAction::Connect, "", "");
        assert!((e.timestamp_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_entry_display() {
        let e = AuditEntry::new(1, 2000, AuditAction::GainChange, "gain set", "tech");
        let s = format!("{e}");
        assert!(s.contains("GAIN_CHANGE"));
        assert!(s.contains("tech"));
    }

    #[test]
    fn test_log_record() {
        let mut log = AuditLog::new();
        let seq = log.record(100, AuditAction::Connect, "route added", "admin");
        assert_eq!(seq, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_log_record_route() {
        let mut log = AuditLog::new();
        log.record_route(100, AuditAction::Connect, 0, 1, "S0->D1", "op");
        let entry = log.get(0).expect("should succeed in test");
        assert_eq!(entry.source, Some(0));
        assert_eq!(entry.destination, Some(1));
    }

    #[test]
    fn test_log_capacity_eviction() {
        let mut log = AuditLog::with_capacity(2);
        log.record(100, AuditAction::Connect, "first", "a");
        log.record(200, AuditAction::Connect, "second", "a");
        log.record(300, AuditAction::Connect, "third", "a");
        assert_eq!(log.len(), 2);
        // First entry should have been evicted
        assert!(log.get(0).is_none());
        assert!(log.get(1).is_some());
    }

    #[test]
    fn test_query_all() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "x");
        log.record(200, AuditAction::Disconnect, "b", "y");
        let results = log.query(&AuditQuery::all());
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_action() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "x");
        log.record(200, AuditAction::Disconnect, "b", "y");
        log.record(300, AuditAction::Connect, "c", "z");
        let q = AuditQuery::all().with_action(AuditAction::Connect);
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_by_user() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "alice");
        log.record(200, AuditAction::Connect, "b", "bob");
        let q = AuditQuery::all().with_user("alice");
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_by_time_range() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "x");
        log.record(500, AuditAction::Connect, "b", "x");
        log.record(900, AuditAction::Connect, "c", "x");
        let q = AuditQuery::all().with_time_range(200, 600);
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_query_with_limit() {
        let mut log = AuditLog::new();
        for i in 0..10 {
            log.record(i * 100, AuditAction::Connect, "x", "x");
        }
        let q = AuditQuery::all().with_limit(3);
        let results = log.query(&q);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_action_counts() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "x");
        log.record(200, AuditAction::Connect, "b", "x");
        log.record(300, AuditAction::Disconnect, "c", "x");
        let counts = log.action_counts();
        let connect_count = counts.iter().find(|c| c.action == AuditAction::Connect);
        assert_eq!(connect_count.expect("should succeed in test").count, 2);
    }

    #[test]
    fn test_clear() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "a", "x");
        log.clear();
        assert!(log.is_empty());
    }

    #[test]
    fn test_last_entry() {
        let mut log = AuditLog::new();
        assert!(log.last_entry().is_none());
        log.record(100, AuditAction::Connect, "first", "x");
        log.record(200, AuditAction::Mute, "second", "y");
        assert_eq!(
            log.last_entry().expect("should succeed in test").action,
            AuditAction::Mute
        );
    }

    #[test]
    fn test_report() {
        let mut log = AuditLog::new();
        log.record(100, AuditAction::Connect, "route added", "admin");
        let r = log.report();
        assert!(r.contains("Audit Log"));
        assert!(r.contains("CONNECT"));
    }

    #[test]
    fn test_audit_action_display() {
        assert_eq!(format!("{}", AuditAction::Failover), "FAILOVER");
        assert_eq!(format!("{}", AuditAction::PresetRecall), "PRESET_RECALL");
    }

    // -----------------------------------------------------------------------
    // RoutingSnapshot tests
    // -----------------------------------------------------------------------

    fn make_before() -> RoutingSnapshot {
        let mut snap = RoutingSnapshot::new("before", 1_000_000, "admin");
        snap.add(0, 0, 0.0);
        snap.add(1, 1, -6.0);
        snap.add(2, 2, -12.0);
        snap
    }

    fn make_after() -> RoutingSnapshot {
        let mut snap = RoutingSnapshot::new("after", 2_000_000, "admin");
        // src 0→0 removed
        // src 1→1 gain changed from -6 to -3
        snap.add(1, 1, -3.0);
        // src 2→2 unchanged
        snap.add(2, 2, -12.0);
        // src 3→3 added
        snap.add(3, 3, 0.0);
        snap
    }

    #[test]
    fn test_snapshot_new_empty() {
        let snap = RoutingSnapshot::new("test", 0, "op");
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
        assert_eq!(snap.meta.label, "test");
    }

    #[test]
    fn test_snapshot_add_and_find() {
        let mut snap = RoutingSnapshot::new("s", 0, "op");
        snap.add(1, 2, -3.0);
        let conn = snap.find(1, 2);
        assert!(conn.is_some());
        assert!((conn.expect("should find").gain_db() - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_find_missing() {
        let snap = RoutingSnapshot::new("s", 0, "op");
        assert!(snap.find(99, 99).is_none());
    }

    #[test]
    fn test_snapshot_insert_replaces_existing() {
        let mut snap = RoutingSnapshot::new("s", 0, "op");
        snap.add(0, 0, -6.0);
        snap.add(0, 0, -12.0); // same src/dst, different gain
        assert_eq!(snap.len(), 1);
        assert!((snap.find(0, 0).expect("exists").gain_db() - (-12.0)).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_connections_sorted() {
        let mut snap = RoutingSnapshot::new("s", 0, "op");
        snap.add(3, 3, 0.0);
        snap.add(1, 1, 0.0);
        snap.add(2, 2, 0.0);
        let conns = snap.connections();
        assert!(conns[0].source <= conns[1].source);
        assert!(conns[1].source <= conns[2].source);
    }

    #[test]
    fn test_diff_identical_snapshots() {
        let before = make_before();
        let after = make_before(); // same data
        let diff = before.diff(&after);
        assert!(diff.is_identical());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_diff_added_connection() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].source, 3);
        assert_eq!(diff.added[0].destination, 3);
    }

    #[test]
    fn test_diff_removed_connection() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].source, 0);
        assert_eq!(diff.removed[0].destination, 0);
    }

    #[test]
    fn test_diff_gain_changed() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        assert_eq!(diff.changed.len(), 1);
        assert_eq!(diff.changed[0].source, 1);
        assert!((diff.changed[0].before_gain_db - (-6.0)).abs() < 0.01);
        assert!((diff.changed[0].after_gain_db - (-3.0)).abs() < 0.01);
        assert!((diff.changed[0].delta_db() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_diff_change_count() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        // 1 added + 1 removed + 1 changed = 3
        assert_eq!(diff.change_count(), 3);
    }

    #[test]
    fn test_diff_report_contains_keywords() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        let report = diff.report();
        assert!(report.contains("ADDED"));
        assert!(report.contains("REMOVED"));
        assert!(report.contains("CHANGED"));
        assert!(report.contains("before"));
        assert!(report.contains("after"));
    }

    #[test]
    fn test_diff_record_to_log() {
        let before = make_before();
        let after = make_after();
        let diff = before.diff(&after);
        let mut log = AuditLog::new();
        diff.record_to_log(&mut log, 9_000_000, "system");
        // 1 Connect + 1 Disconnect + 1 GainChange = 3 entries
        assert_eq!(log.len(), 3);
        let connects = log.query(&AuditQuery::all().with_action(AuditAction::Connect));
        assert_eq!(connects.len(), 1);
        let disconnects = log.query(&AuditQuery::all().with_action(AuditAction::Disconnect));
        assert_eq!(disconnects.len(), 1);
        let gains = log.query(&AuditQuery::all().with_action(AuditAction::GainChange));
        assert_eq!(gains.len(), 1);
    }

    #[test]
    fn test_snapshot_connection_gain_db_x100() {
        let conn = SnapshotConnection::new(0, 1, -6.5);
        assert_eq!(conn.gain_db_x100, -650);
        assert!((conn.gain_db() - (-6.5)).abs() < 0.01);
    }

    #[test]
    fn test_gain_changed_delta_db() {
        let gc = GainChanged {
            source: 0,
            destination: 1,
            before_gain_db: -12.0,
            after_gain_db: -6.0,
        };
        assert!((gc.delta_db() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_diff_empty_snapshots_identical() {
        let before = RoutingSnapshot::new("a", 0, "op");
        let after = RoutingSnapshot::new("b", 1, "op");
        let diff = before.diff(&after);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_diff_all_added_when_before_empty() {
        let before = RoutingSnapshot::new("empty", 0, "op");
        let mut after = RoutingSnapshot::new("full", 1, "op");
        after.add(0, 0, 0.0);
        after.add(1, 1, -3.0);
        let diff = before.diff(&after);
        assert_eq!(diff.added.len(), 2);
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_diff_all_removed_when_after_empty() {
        let mut before = RoutingSnapshot::new("full", 0, "op");
        before.add(0, 0, 0.0);
        before.add(1, 1, -3.0);
        let after = RoutingSnapshot::new("empty", 1, "op");
        let diff = before.diff(&after);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 2);
        assert!(diff.changed.is_empty());
    }
}
