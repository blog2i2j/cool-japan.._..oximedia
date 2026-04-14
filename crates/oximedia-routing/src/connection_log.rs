#![allow(dead_code)]
//! NMOS IS-05 connection change log.
//!
//! Every connection-level event — activation, deactivation, staging,
//! immediate single-point changes, and rollback — is captured as a
//! [`ConnectionEntry`] in a [`ConnectionLog`].  The log supports
//! pagination, filtering by sender/receiver/reason, and atomic rollback
//! tracking so engineers can reconstruct the full history of a session.
//!
//! # Design
//!
//! - [`ChangeReason`] — machine-readable reason code for a connection change.
//! - [`ConnectionEntry`] — a single immutable audit record.
//! - [`ConnectionLog`] — ordered store of entries with query helpers.
//! - [`RollbackTracker`] — lightweight helper for recording and replaying
//!   rollback checkpoints.

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ChangeReason
// ---------------------------------------------------------------------------

/// Machine-readable reason code explaining why a connection change occurred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChangeReason {
    /// Manual operator request via the IS-05 REST API.
    OperatorRequest,
    /// Automated activation at a scheduled time (IS-05 activation mode).
    ScheduledActivation,
    /// Immediate (non-scheduled) activation.
    ImmediateActivation,
    /// Automatic failover triggered by signal loss or node failure.
    AutoFailover,
    /// Configuration was rolled back to a previous checkpoint.
    Rollback,
    /// Connection was staged but not yet activated.
    Staged,
    /// Connection was deactivated (sender side went offline or receiver disconnected).
    Deactivated,
    /// Health check / keep-alive update.
    HealthCheck,
    /// Reason is not known or was not recorded.
    Unknown,
}

impl ChangeReason {
    /// Returns a short machine-readable string.
    pub fn code(&self) -> &'static str {
        match self {
            Self::OperatorRequest => "operator_request",
            Self::ScheduledActivation => "scheduled_activation",
            Self::ImmediateActivation => "immediate_activation",
            Self::AutoFailover => "auto_failover",
            Self::Rollback => "rollback",
            Self::Staged => "staged",
            Self::Deactivated => "deactivated",
            Self::HealthCheck => "health_check",
            Self::Unknown => "unknown",
        }
    }

    /// Returns `true` if this reason indicates an error or unexpected condition.
    pub fn is_fault_related(&self) -> bool {
        matches!(self, Self::AutoFailover | Self::Rollback)
    }
}

impl fmt::Display for ChangeReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

// ---------------------------------------------------------------------------
// ConnectionState
// ---------------------------------------------------------------------------

/// The resulting connection state after a change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// The connection is fully active.
    Active,
    /// Configuration has been staged but not activated.
    Staged,
    /// The connection has been torn down.
    Inactive,
    /// An error occurred during the change attempt.
    Error,
}

impl fmt::Display for ConnectionState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Staged => write!(f, "staged"),
            Self::Inactive => write!(f, "inactive"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ---------------------------------------------------------------------------
// TransportParams
// ---------------------------------------------------------------------------

/// A snapshot of key transport parameters for a single-direction leg.
#[derive(Debug, Clone, Default)]
pub struct TransportParams {
    /// Destination IP address or hostname.
    pub destination_ip: Option<String>,
    /// Destination UDP port.
    pub destination_port: Option<u16>,
    /// Source IP (for ST 2110 multicast joins).
    pub source_ip: Option<String>,
    /// RTP payload type identifier.
    pub rtp_payload_type: Option<u8>,
    /// SSRC (Synchronisation Source) identifier.
    pub ssrc: Option<u32>,
}

impl TransportParams {
    /// Creates an empty parameter snapshot.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the destination address.
    pub fn with_destination(mut self, ip: impl Into<String>, port: u16) -> Self {
        self.destination_ip = Some(ip.into());
        self.destination_port = Some(port);
        self
    }

    /// Sets the source IP.
    pub fn with_source_ip(mut self, ip: impl Into<String>) -> Self {
        self.source_ip = Some(ip.into());
        self
    }

    /// Sets the RTP payload type.
    pub fn with_rtp_payload_type(mut self, pt: u8) -> Self {
        self.rtp_payload_type = Some(pt);
        self
    }
}

// ---------------------------------------------------------------------------
// ConnectionEntry
// ---------------------------------------------------------------------------

/// A single immutable record in the connection change log.
#[derive(Debug, Clone)]
pub struct ConnectionEntry {
    /// Monotonically increasing sequence number.
    pub seq: u64,
    /// Timestamp in microseconds since epoch (or log creation).
    pub timestamp_us: u64,
    /// IS-05 sender id involved in the change.
    pub sender_id: String,
    /// IS-05 receiver id involved in the change.
    pub receiver_id: String,
    /// Reason the change occurred.
    pub reason: ChangeReason,
    /// Connection state *after* this change.
    pub result_state: ConnectionState,
    /// Optional transport parameter snapshot for this leg.
    pub transport: Option<TransportParams>,
    /// Optional human-readable notes.
    pub notes: String,
    /// Whether this entry is the target of a rollback operation.
    pub is_rollback_target: bool,
    /// Sequence number of a related entry (e.g., the original entry being rolled back).
    pub related_seq: Option<u64>,
}

impl ConnectionEntry {
    /// Creates a new entry.
    pub fn new(
        seq: u64,
        timestamp_us: u64,
        sender_id: impl Into<String>,
        receiver_id: impl Into<String>,
        reason: ChangeReason,
        result_state: ConnectionState,
    ) -> Self {
        Self {
            seq,
            timestamp_us,
            sender_id: sender_id.into(),
            receiver_id: receiver_id.into(),
            reason,
            result_state,
            transport: None,
            notes: String::new(),
            is_rollback_target: false,
            related_seq: None,
        }
    }

    /// Attaches transport parameters to this entry.
    pub fn with_transport(mut self, params: TransportParams) -> Self {
        self.transport = Some(params);
        self
    }

    /// Attaches a note.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Marks this entry as a rollback target for a later rollback operation.
    pub fn mark_rollback_target(mut self) -> Self {
        self.is_rollback_target = true;
        self
    }

    /// Links this entry to a related sequence number.
    pub fn with_related(mut self, related_seq: u64) -> Self {
        self.related_seq = Some(related_seq);
        self
    }

    /// Returns the timestamp in milliseconds.
    pub fn timestamp_ms(&self) -> f64 {
        self.timestamp_us as f64 / 1_000.0
    }
}

impl fmt::Display for ConnectionEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{seq:06}] {ts:.1}ms {sender}->{receiver} {reason} => {state}",
            seq = self.seq,
            ts = self.timestamp_ms(),
            sender = self.sender_id,
            receiver = self.receiver_id,
            reason = self.reason,
            state = self.result_state,
        )
    }
}

// ---------------------------------------------------------------------------
// LogQuery
// ---------------------------------------------------------------------------

/// Filter for [`ConnectionLog::query`].
#[derive(Debug, Clone, Default)]
pub struct LogQuery {
    /// Filter by sender id.
    pub sender_id: Option<String>,
    /// Filter by receiver id.
    pub receiver_id: Option<String>,
    /// Filter by reason.
    pub reason: Option<ChangeReason>,
    /// Filter by result state.
    pub result_state: Option<ConnectionState>,
    /// Only return entries from this timestamp forward (inclusive).
    pub from_us: Option<u64>,
    /// Only return entries up to this timestamp (inclusive).
    pub to_us: Option<u64>,
    /// Only return rollback-target entries.
    pub only_rollback_targets: bool,
    /// Maximum number of results (0 = unlimited).
    pub limit: usize,
}

impl LogQuery {
    /// Creates a query that matches everything.
    pub fn all() -> Self {
        Self::default()
    }

    /// Filter by sender.
    pub fn with_sender(mut self, sender_id: impl Into<String>) -> Self {
        self.sender_id = Some(sender_id.into());
        self
    }

    /// Filter by receiver.
    pub fn with_receiver(mut self, receiver_id: impl Into<String>) -> Self {
        self.receiver_id = Some(receiver_id.into());
        self
    }

    /// Filter by reason.
    pub fn with_reason(mut self, reason: ChangeReason) -> Self {
        self.reason = Some(reason);
        self
    }

    /// Filter by result state.
    pub fn with_state(mut self, state: ConnectionState) -> Self {
        self.result_state = Some(state);
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
        self.limit = limit;
        self
    }

    /// Only rollback targets.
    pub fn rollback_targets_only(mut self) -> Self {
        self.only_rollback_targets = true;
        self
    }

    fn matches(&self, entry: &ConnectionEntry) -> bool {
        if let Some(ref s) = self.sender_id {
            if &entry.sender_id != s {
                return false;
            }
        }
        if let Some(ref r) = self.receiver_id {
            if &entry.receiver_id != r {
                return false;
            }
        }
        if let Some(reason) = self.reason {
            if entry.reason != reason {
                return false;
            }
        }
        if let Some(state) = self.result_state {
            if entry.result_state != state {
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
        if self.only_rollback_targets && !entry.is_rollback_target {
            return false;
        }
        true
    }
}

// ---------------------------------------------------------------------------
// ConnectionLog
// ---------------------------------------------------------------------------

/// Ordered log of IS-05 connection change events.
#[derive(Debug, Default)]
pub struct ConnectionLog {
    entries: Vec<ConnectionEntry>,
    /// Next sequence number.
    next_seq: u64,
    /// Maximum entries retained (0 = unlimited).
    max_entries: usize,
}

impl ConnectionLog {
    /// Creates an unlimited log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a log that retains at most `max_entries` entries.
    /// Oldest entries are evicted first when the limit is exceeded.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            max_entries,
            ..Default::default()
        }
    }

    /// Appends a new entry and returns its sequence number.
    pub fn append(&mut self, entry: ConnectionEntry) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        let mut e = entry;
        e.seq = seq;
        self.entries.push(e);

        if self.max_entries > 0 && self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }

        seq
    }

    /// Convenience: record a change directly without pre-building an entry.
    pub fn record(
        &mut self,
        timestamp_us: u64,
        sender_id: impl Into<String>,
        receiver_id: impl Into<String>,
        reason: ChangeReason,
        result_state: ConnectionState,
    ) -> u64 {
        let seq = self.next_seq;
        let entry = ConnectionEntry::new(
            seq,
            timestamp_us,
            sender_id,
            receiver_id,
            reason,
            result_state,
        );
        self.append(entry)
    }

    /// Returns total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up an entry by sequence number.
    pub fn get(&self, seq: u64) -> Option<&ConnectionEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    /// Returns the most recent entry.
    pub fn last(&self) -> Option<&ConnectionEntry> {
        self.entries.last()
    }

    /// Queries entries matching the filter.
    pub fn query(&self, q: &LogQuery) -> Vec<&ConnectionEntry> {
        let mut results: Vec<&ConnectionEntry> =
            self.entries.iter().filter(|e| q.matches(e)).collect();
        if q.limit > 0 {
            results.truncate(q.limit);
        }
        results
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns counts per [`ChangeReason`].
    pub fn reason_counts(&self) -> HashMap<ChangeReason, usize> {
        let mut map = HashMap::new();
        for e in &self.entries {
            *map.entry(e.reason).or_default() += 1;
        }
        map
    }

    /// Generates a human-readable text report.
    pub fn report(&self) -> String {
        let mut lines = vec![format!("ConnectionLog ({} entries)", self.entries.len())];
        for e in &self.entries {
            lines.push(format!("  {e}"));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// RollbackTracker
// ---------------------------------------------------------------------------

/// Tracks rollback checkpoints so a connection session can be restored to a
/// known-good state.
///
/// A checkpoint captures the sequence number of the most recent `Active`
/// entry for each `(sender_id, receiver_id)` pair.  To roll back, callers
/// retrieve the checkpointed entry from the [`ConnectionLog`] and re-apply
/// the recorded transport parameters.
#[derive(Debug, Default)]
pub struct RollbackTracker {
    /// Maps (sender_id, receiver_id) → checkpoint sequence number.
    checkpoints: HashMap<(String, String), u64>,
    /// Number of rollback operations performed.
    rollback_count: u32,
}

impl RollbackTracker {
    /// Creates an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a checkpoint for the given pair.
    pub fn record_checkpoint(
        &mut self,
        sender_id: impl Into<String>,
        receiver_id: impl Into<String>,
        seq: u64,
    ) {
        self.checkpoints
            .insert((sender_id.into(), receiver_id.into()), seq);
    }

    /// Returns the checkpoint sequence number for the given pair, if any.
    pub fn checkpoint_for(&self, sender_id: &str, receiver_id: &str) -> Option<u64> {
        self.checkpoints
            .get(&(sender_id.to_string(), receiver_id.to_string()))
            .copied()
    }

    /// Removes the checkpoint for the given pair.
    pub fn clear_checkpoint(&mut self, sender_id: &str, receiver_id: &str) {
        self.checkpoints
            .remove(&(sender_id.to_string(), receiver_id.to_string()));
    }

    /// Returns the number of stored checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Increments the rollback counter and returns the new count.
    pub fn record_rollback(&mut self) -> u32 {
        self.rollback_count += 1;
        self.rollback_count
    }

    /// Returns the total number of rollback operations recorded.
    pub fn rollback_count(&self) -> u32 {
        self.rollback_count
    }

    /// Produces a rollback entry for the [`ConnectionLog`] based on the
    /// checkpoint for the given pair.  Returns `None` if no checkpoint exists.
    pub fn build_rollback_entry(
        &self,
        timestamp_us: u64,
        sender_id: &str,
        receiver_id: &str,
        log: &ConnectionLog,
    ) -> Option<ConnectionEntry> {
        let checkpoint_seq = self.checkpoint_for(sender_id, receiver_id)?;
        let original = log.get(checkpoint_seq)?;

        // Clone transport params from the original active entry
        let transport = original.transport.clone();

        let mut entry = ConnectionEntry::new(
            0, // will be assigned by log.append
            timestamp_us,
            sender_id,
            receiver_id,
            ChangeReason::Rollback,
            ConnectionState::Active,
        )
        .with_related(checkpoint_seq);

        if let Some(tp) = transport {
            entry = entry.with_transport(tp);
        }

        Some(entry)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_entry(
        seq: u64,
        ts: u64,
        reason: ChangeReason,
        state: ConnectionState,
    ) -> ConnectionEntry {
        ConnectionEntry::new(seq, ts, "snd-001", "rcv-001", reason, state)
    }

    #[test]
    fn test_log_append_and_len() {
        let mut log = ConnectionLog::new();
        let seq = log.record(
            1000,
            "snd-A",
            "rcv-B",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        assert_eq!(seq, 0);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn test_log_get_by_seq() {
        let mut log = ConnectionLog::new();
        log.record(
            100,
            "snd-1",
            "rcv-1",
            ChangeReason::Staged,
            ConnectionState::Staged,
        );
        log.record(
            200,
            "snd-1",
            "rcv-1",
            ChangeReason::ImmediateActivation,
            ConnectionState::Active,
        );
        assert!(log.get(0).is_some());
        assert_eq!(log.get(1).unwrap().result_state, ConnectionState::Active);
    }

    #[test]
    fn test_log_capacity_evicts_oldest() {
        let mut log = ConnectionLog::with_capacity(2);
        log.record(
            100,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            200,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            300,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        assert_eq!(log.len(), 2);
        // seq 0 should be evicted
        assert!(log.get(0).is_none());
        assert!(log.get(1).is_some());
    }

    #[test]
    fn test_query_by_sender() {
        let mut log = ConnectionLog::new();
        log.record(
            100,
            "snd-A",
            "rcv-X",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            200,
            "snd-B",
            "rcv-X",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        let q = LogQuery::all().with_sender("snd-A");
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].sender_id, "snd-A");
    }

    #[test]
    fn test_query_by_reason() {
        let mut log = ConnectionLog::new();
        log.record(
            100,
            "s",
            "r",
            ChangeReason::AutoFailover,
            ConnectionState::Active,
        );
        log.record(
            200,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            300,
            "s",
            "r",
            ChangeReason::AutoFailover,
            ConnectionState::Active,
        );
        let q = LogQuery::all().with_reason(ChangeReason::AutoFailover);
        let results = log.query(&q);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_query_time_range() {
        let mut log = ConnectionLog::new();
        log.record(
            100,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            500,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            900,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        let q = LogQuery::all().with_time_range(200, 700);
        assert_eq!(log.query(&q).len(), 1);
    }

    #[test]
    fn test_query_with_limit() {
        let mut log = ConnectionLog::new();
        for ts in 0..10 {
            log.record(
                ts * 100,
                "s",
                "r",
                ChangeReason::HealthCheck,
                ConnectionState::Active,
            );
        }
        let q = LogQuery::all().with_limit(3);
        assert_eq!(log.query(&q).len(), 3);
    }

    #[test]
    fn test_rollback_tracker_checkpoint() {
        let mut tracker = RollbackTracker::new();
        tracker.record_checkpoint("snd-1", "rcv-1", 42);
        assert_eq!(tracker.checkpoint_for("snd-1", "rcv-1"), Some(42));
        assert_eq!(tracker.checkpoint_count(), 1);
    }

    #[test]
    fn test_rollback_tracker_clear() {
        let mut tracker = RollbackTracker::new();
        tracker.record_checkpoint("snd-1", "rcv-1", 5);
        tracker.clear_checkpoint("snd-1", "rcv-1");
        assert_eq!(tracker.checkpoint_for("snd-1", "rcv-1"), None);
        assert_eq!(tracker.checkpoint_count(), 0);
    }

    #[test]
    fn test_rollback_tracker_build_entry() {
        let mut log = ConnectionLog::new();
        let seq = log.append(
            ConnectionEntry::new(
                0,
                1000,
                "snd-2",
                "rcv-2",
                ChangeReason::ImmediateActivation,
                ConnectionState::Active,
            )
            .with_transport(TransportParams::new().with_destination("239.1.1.1", 5000)),
        );

        let mut tracker = RollbackTracker::new();
        tracker.record_checkpoint("snd-2", "rcv-2", seq);

        let entry = tracker.build_rollback_entry(9000, "snd-2", "rcv-2", &log);
        assert!(entry.is_some());
        let e = entry.unwrap();
        assert_eq!(e.reason, ChangeReason::Rollback);
        assert_eq!(e.related_seq, Some(seq));
        assert!(e.transport.is_some());
        assert_eq!(
            e.transport.unwrap().destination_ip.as_deref(),
            Some("239.1.1.1")
        );
    }

    #[test]
    fn test_rollback_counter() {
        let mut tracker = RollbackTracker::new();
        assert_eq!(tracker.rollback_count(), 0);
        tracker.record_rollback();
        tracker.record_rollback();
        assert_eq!(tracker.rollback_count(), 2);
    }

    #[test]
    fn test_change_reason_fault_related() {
        assert!(ChangeReason::AutoFailover.is_fault_related());
        assert!(ChangeReason::Rollback.is_fault_related());
        assert!(!ChangeReason::OperatorRequest.is_fault_related());
        assert!(!ChangeReason::HealthCheck.is_fault_related());
    }

    #[test]
    fn test_reason_counts() {
        let mut log = ConnectionLog::new();
        log.record(
            100,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            200,
            "s",
            "r",
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        );
        log.record(
            300,
            "s",
            "r",
            ChangeReason::AutoFailover,
            ConnectionState::Active,
        );
        let counts = log.reason_counts();
        assert_eq!(counts.get(&ChangeReason::OperatorRequest).copied(), Some(2));
        assert_eq!(counts.get(&ChangeReason::AutoFailover).copied(), Some(1));
    }

    #[test]
    fn test_entry_display() {
        let e = ConnectionEntry::new(
            7,
            2_500_000,
            "snd-disp",
            "rcv-disp",
            ChangeReason::Staged,
            ConnectionState::Staged,
        );
        let s = format!("{e}");
        assert!(s.contains("snd-disp"));
        assert!(s.contains("staged"));
    }

    #[test]
    fn test_rollback_targets_only_query() {
        let mut log = ConnectionLog::new();
        log.append(simple_entry(
            0,
            100,
            ChangeReason::OperatorRequest,
            ConnectionState::Active,
        ));
        log.append(
            simple_entry(0, 200, ChangeReason::Rollback, ConnectionState::Active)
                .mark_rollback_target(),
        );
        let q = LogQuery::all().rollback_targets_only();
        let results = log.query(&q);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_rollback_target);
    }
}
