//! Edit history recorder and player for session replay.
//!
//! Provides a lightweight, dependency-free record of collaboration actions
//! that can be persisted as NDJSON and replayed at arbitrary points in time.

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// EditHistoryEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single recorded action in the edit history.
#[derive(Debug, Clone, PartialEq)]
pub struct EditHistoryEntry {
    /// Wall-clock time of the action in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Identifier of the user who performed the action.
    pub user_id: String,
    /// Short label identifying the kind of action (e.g. "insert", "delete",
    /// "move", "color_grade").
    pub action_type: String,
    /// JSON-encoded action payload (opaque to this module).
    pub payload: String,
}

impl EditHistoryEntry {
    /// Serialise this entry as a single-line JSON object (no trailing newline).
    ///
    /// The serialisation is hand-written so no additional dependencies are
    /// needed.
    pub fn to_json_line(&self) -> String {
        let ts = self.timestamp_ms;
        let uid = escape_json_str(&self.user_id);
        let at = escape_json_str(&self.action_type);
        // payload is already a JSON fragment — embed it verbatim.
        format!(
            r#"{{"timestamp_ms":{ts},"user_id":"{uid}","action_type":"{at}","payload":{payload}}}"#,
            ts = ts,
            uid = uid,
            at = at,
            payload = self.payload,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HistoryRecorder
// ─────────────────────────────────────────────────────────────────────────────

/// Accumulates `EditHistoryEntry` values during an active recording session.
#[derive(Debug, Default)]
pub struct HistoryRecorder {
    /// All recorded entries in chronological order.
    pub entries: Vec<EditHistoryEntry>,
    /// Whether new entries are being accepted.
    pub recording: bool,
}

impl HistoryRecorder {
    /// Create a new, stopped recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start recording.  Has no effect if already recording.
    pub fn start(&mut self) {
        self.recording = true;
    }

    /// Stop recording.  Has no effect if already stopped.
    pub fn stop(&mut self) {
        self.recording = false;
    }

    /// Record a new entry.
    ///
    /// Returns `true` if the entry was appended, `false` if recording is
    /// currently stopped.
    pub fn record(
        &mut self,
        user_id: String,
        action: String,
        payload: String,
        now_ms: u64,
    ) -> bool {
        if !self.recording {
            return false;
        }
        self.entries.push(EditHistoryEntry {
            timestamp_ms: now_ms,
            user_id,
            action_type: action,
            payload,
        });
        true
    }

    /// Total number of recorded entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Drain entries into a `HistoryPlayer`.
    pub fn into_player(self) -> HistoryPlayer {
        HistoryPlayer::from_entries(self.entries)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HistoryPlayer
// ─────────────────────────────────────────────────────────────────────────────

/// Read-only view of a recorded history that supports time-based and
/// thread-based queries, plus NDJSON export.
#[derive(Debug)]
pub struct HistoryPlayer {
    entries: Vec<EditHistoryEntry>,
}

impl HistoryPlayer {
    /// Build a player from an existing collection of entries.
    ///
    /// The entries are sorted by `timestamp_ms` ascending so that
    /// `replay_at` returns a stable prefix regardless of insertion order.
    pub fn from_entries(mut entries: Vec<EditHistoryEntry>) -> Self {
        entries.sort_by_key(|e| e.timestamp_ms);
        Self { entries }
    }

    /// Return all entries with `timestamp_ms <= target_ms`.
    pub fn replay_at(&self, target_ms: u64) -> Vec<&EditHistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp_ms <= target_ms)
            .collect()
    }

    /// Return all entries authored by `user_id`.
    pub fn by_user(&self, user_id: &str) -> Vec<&EditHistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.user_id == user_id)
            .collect()
    }

    /// Return all entries with `action_type == action`.
    pub fn by_action(&self, action: &str) -> Vec<&EditHistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.action_type == action)
            .collect()
    }

    /// Serialise the full history as NDJSON (one JSON object per line).
    ///
    /// The output ends with a trailing newline.
    pub fn to_json_log(&self) -> String {
        self.entries
            .iter()
            .map(|e| e.to_json_line())
            .collect::<Vec<_>>()
            .join("\n")
            + if self.entries.is_empty() { "" } else { "\n" }
    }

    /// Total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the player has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Escape a string for embedding inside a JSON double-quoted value.
fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(ts: u64, user: &str, action: &str) -> EditHistoryEntry {
        EditHistoryEntry {
            timestamp_ms: ts,
            user_id: user.to_string(),
            action_type: action.to_string(),
            payload: r#"{"v":1}"#.to_string(),
        }
    }

    // ── HistoryRecorder ──────────────────────────────────────────────────────

    #[test]
    fn test_recorder_start_stop() {
        let mut rec = HistoryRecorder::new();
        assert!(!rec.recording);
        rec.start();
        assert!(rec.recording);
        rec.stop();
        assert!(!rec.recording);
    }

    #[test]
    fn test_recorder_does_not_record_while_stopped() {
        let mut rec = HistoryRecorder::new();
        let accepted = rec.record("u1".into(), "insert".into(), r#"{"x":1}"#.into(), 1_000);
        assert!(!accepted);
        assert_eq!(rec.entry_count(), 0);
    }

    #[test]
    fn test_recorder_records_while_running() {
        let mut rec = HistoryRecorder::new();
        rec.start();
        let ok = rec.record("alice".into(), "cut".into(), r#"{"clip":"c1"}"#.into(), 1_000);
        assert!(ok);
        assert_eq!(rec.entry_count(), 1);
    }

    #[test]
    fn test_recorder_multiple_entries_in_order() {
        let mut rec = HistoryRecorder::new();
        rec.start();
        rec.record("a".into(), "insert".into(), r#"{"i":0}"#.into(), 100);
        rec.record("b".into(), "delete".into(), r#"{"i":1}"#.into(), 200);
        rec.record("a".into(), "move".into(), r#"{"i":2}"#.into(), 300);
        assert_eq!(rec.entry_count(), 3);
        assert_eq!(rec.entries[0].user_id, "a");
        assert_eq!(rec.entries[1].user_id, "b");
    }

    // ── HistoryPlayer ────────────────────────────────────────────────────────

    #[test]
    fn test_player_replay_at_all_before() {
        let entries = vec![
            make_entry(100, "u1", "insert"),
            make_entry(200, "u2", "delete"),
            make_entry(300, "u1", "move"),
        ];
        let player = HistoryPlayer::from_entries(entries);
        let at_250 = player.replay_at(250);
        assert_eq!(at_250.len(), 2);
        assert!(at_250.iter().all(|e| e.timestamp_ms <= 250));
    }

    #[test]
    fn test_player_replay_at_none_before() {
        let entries = vec![make_entry(500, "u1", "insert")];
        let player = HistoryPlayer::from_entries(entries);
        let result = player.replay_at(100);
        assert!(result.is_empty());
    }

    #[test]
    fn test_player_replay_at_inclusive_boundary() {
        let entries = vec![make_entry(500, "u1", "insert")];
        let player = HistoryPlayer::from_entries(entries);
        let result = player.replay_at(500);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_player_by_user() {
        let entries = vec![
            make_entry(100, "alice", "insert"),
            make_entry(200, "bob", "delete"),
            make_entry(300, "alice", "move"),
        ];
        let player = HistoryPlayer::from_entries(entries);
        let alice = player.by_user("alice");
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|e| e.user_id == "alice"));
    }

    #[test]
    fn test_player_by_action() {
        let entries = vec![
            make_entry(100, "u1", "insert"),
            make_entry(200, "u1", "delete"),
            make_entry(300, "u2", "insert"),
        ];
        let player = HistoryPlayer::from_entries(entries);
        let inserts = player.by_action("insert");
        assert_eq!(inserts.len(), 2);
    }

    #[test]
    fn test_player_entries_sorted_by_timestamp() {
        // Supply entries out of order; player must sort them.
        let entries = vec![
            make_entry(300, "u1", "c"),
            make_entry(100, "u1", "a"),
            make_entry(200, "u1", "b"),
        ];
        let player = HistoryPlayer::from_entries(entries);
        assert_eq!(player.entries[0].timestamp_ms, 100);
        assert_eq!(player.entries[1].timestamp_ms, 200);
        assert_eq!(player.entries[2].timestamp_ms, 300);
    }

    #[test]
    fn test_to_json_log_ndjson_format() {
        let entries = vec![
            make_entry(100, "alice", "cut"),
            make_entry(200, "bob", "paste"),
        ];
        let player = HistoryPlayer::from_entries(entries);
        let log = player.to_json_log();
        let lines: Vec<&str> = log.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line must be valid JSON-ish (contains expected fields).
        assert!(lines[0].contains("\"timestamp_ms\":100"));
        assert!(lines[0].contains("\"user_id\":\"alice\""));
        assert!(lines[0].contains("\"action_type\":\"cut\""));
        assert!(lines[1].contains("\"timestamp_ms\":200"));
    }

    #[test]
    fn test_to_json_log_empty() {
        let player = HistoryPlayer::from_entries(vec![]);
        assert_eq!(player.to_json_log(), "");
    }

    #[test]
    fn test_edit_history_entry_to_json_line_escaping() {
        let entry = EditHistoryEntry {
            timestamp_ms: 42,
            user_id: r#"user"with"quotes"#.to_string(),
            action_type: "insert".to_string(),
            payload: r#"{"k":"v"}"#.to_string(),
        };
        let line = entry.to_json_line();
        // The quote in the user_id must be escaped.
        assert!(line.contains(r#"user\"with\"quotes"#));
    }

    #[test]
    fn test_recorder_into_player() {
        let mut rec = HistoryRecorder::new();
        rec.start();
        rec.record("u1".into(), "op".into(), r#"{}"#.into(), 1_000);
        rec.stop();
        let player = rec.into_player();
        assert_eq!(player.len(), 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HistoryReplayer — the high-level API requested by the TODO
// ─────────────────────────────────────────────────────────────────────────────

/// A typed event payload that can be applied to produce an incremental state.
///
/// The [`HistoryReplayer`] is intentionally generic over the event / state
/// types so callers can map raw [`EditHistoryEntry`] records to domain objects.
/// When the simpler `Vec<EditHistoryEntry>`-based API is sufficient, use
/// [`HistoryReplayer::new`] directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    /// Timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Author of this event.
    pub user_id: String,
    /// Opaque action kind label.
    pub action_type: String,
    /// JSON-encoded payload (opaque to this module).
    pub payload: String,
}

impl Event {
    /// Build an `Event` from an [`EditHistoryEntry`].
    pub fn from_entry(entry: &EditHistoryEntry) -> Self {
        Self {
            timestamp_ms: entry.timestamp_ms,
            user_id: entry.user_id.clone(),
            action_type: entry.action_type.clone(),
            payload: entry.payload.clone(),
        }
    }
}

/// Accumulated state produced by replaying a sequence of [`Event`]s up to a
/// given timestamp.
///
/// The state is intentionally simple: it accumulates the list of applied
/// events so higher-level code can fold them into a domain-specific
/// representation (e.g., reconstruct the timeline, undo history, etc.).
#[derive(Debug, Clone, Default)]
pub struct State {
    /// All events that were applied to produce this state, in chronological order.
    pub applied_events: Vec<Event>,
}

impl State {
    /// Create an empty initial state.
    pub fn empty() -> Self {
        Self {
            applied_events: Vec::new(),
        }
    }

    /// Number of events accumulated.
    pub fn event_count(&self) -> usize {
        self.applied_events.len()
    }

    /// Latest timestamp among all applied events, or `0` when empty.
    pub fn latest_timestamp_ms(&self) -> u64 {
        self.applied_events
            .iter()
            .map(|e| e.timestamp_ms)
            .max()
            .unwrap_or(0)
    }
}

/// High-level history replayer.
///
/// Stores a chronologically sorted list of [`Event`]s and produces a
/// [`State`] snapshot for any target timestamp via [`replay_to`](Self::replay_to).
///
/// # Example
///
/// ```
/// use oximedia_collab::history_replay::{Event, HistoryReplayer};
///
/// let events = vec![
///     Event { timestamp_ms: 100, user_id: "alice".into(), action_type: "insert".into(), payload: "{}".into() },
///     Event { timestamp_ms: 200, user_id: "bob".into(),   action_type: "delete".into(), payload: "{}".into() },
///     Event { timestamp_ms: 300, user_id: "alice".into(), action_type: "move".into(),   payload: "{}".into() },
/// ];
///
/// let replayer = HistoryReplayer::new(events);
/// let state = replayer.replay_to(200);
/// assert_eq!(state.event_count(), 2);
/// ```
#[derive(Debug)]
pub struct HistoryReplayer {
    /// Sorted events.
    events: Vec<Event>,
}

impl HistoryReplayer {
    /// Create a new replayer from a list of events.
    ///
    /// Events are sorted ascending by `timestamp_ms` so that
    /// [`replay_to`](Self::replay_to) always returns a stable prefix.
    pub fn new(mut events: Vec<Event>) -> Self {
        events.sort_by_key(|e| e.timestamp_ms);
        Self { events }
    }

    /// Build a replayer from raw [`EditHistoryEntry`] records.
    pub fn from_entries(entries: Vec<EditHistoryEntry>) -> Self {
        let events = entries.iter().map(Event::from_entry).collect();
        Self::new(events)
    }

    /// Replay all events with `timestamp_ms <= ts` and return the resulting
    /// [`State`].
    ///
    /// Calling `replay_to(u64::MAX)` returns the complete final state.
    pub fn replay_to(&self, ts: u64) -> State {
        let applied_events = self
            .events
            .iter()
            .filter(|e| e.timestamp_ms <= ts)
            .cloned()
            .collect();
        State { applied_events }
    }

    /// Total number of events in the replayer.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Whether the replayer has no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod replayer_tests {
    use super::*;

    fn make_event(ts: u64, user: &str, action: &str) -> Event {
        Event {
            timestamp_ms: ts,
            user_id: user.to_string(),
            action_type: action.to_string(),
            payload: r#"{"v":1}"#.to_string(),
        }
    }

    #[test]
    fn test_replayer_new_sorts_events() {
        let events = vec![
            make_event(300, "u1", "c"),
            make_event(100, "u1", "a"),
            make_event(200, "u1", "b"),
        ];
        let r = HistoryReplayer::new(events);
        assert_eq!(r.events[0].timestamp_ms, 100);
        assert_eq!(r.events[1].timestamp_ms, 200);
        assert_eq!(r.events[2].timestamp_ms, 300);
    }

    #[test]
    fn test_replay_to_all_before() {
        let events = vec![
            make_event(100, "u1", "insert"),
            make_event(200, "u2", "delete"),
            make_event(300, "u1", "move"),
        ];
        let r = HistoryReplayer::new(events);
        let state = r.replay_to(250);
        assert_eq!(state.event_count(), 2);
    }

    #[test]
    fn test_replay_to_none_before() {
        let events = vec![make_event(500, "u1", "insert")];
        let r = HistoryReplayer::new(events);
        let state = r.replay_to(100);
        assert_eq!(state.event_count(), 0);
    }

    #[test]
    fn test_replay_to_inclusive_boundary() {
        let events = vec![make_event(500, "u1", "insert")];
        let r = HistoryReplayer::new(events);
        let state = r.replay_to(500);
        assert_eq!(state.event_count(), 1);
    }

    #[test]
    fn test_replay_to_full_history() {
        let events = vec![
            make_event(100, "u1", "a"),
            make_event(200, "u2", "b"),
            make_event(300, "u1", "c"),
        ];
        let r = HistoryReplayer::new(events);
        let state = r.replay_to(u64::MAX);
        assert_eq!(state.event_count(), 3);
    }

    #[test]
    fn test_state_latest_timestamp() {
        let events = vec![
            make_event(100, "u1", "a"),
            make_event(200, "u2", "b"),
        ];
        let r = HistoryReplayer::new(events);
        let state = r.replay_to(u64::MAX);
        assert_eq!(state.latest_timestamp_ms(), 200);
    }

    #[test]
    fn test_state_empty_latest_timestamp_is_zero() {
        let state = State::empty();
        assert_eq!(state.latest_timestamp_ms(), 0);
    }

    #[test]
    fn test_from_entries_converts_correctly() {
        let entries = vec![EditHistoryEntry {
            timestamp_ms: 42,
            user_id: "alice".to_string(),
            action_type: "cut".to_string(),
            payload: r#"{"clip":"c1"}"#.to_string(),
        }];
        let r = HistoryReplayer::from_entries(entries);
        let state = r.replay_to(u64::MAX);
        assert_eq!(state.event_count(), 1);
        assert_eq!(state.applied_events[0].user_id, "alice");
        assert_eq!(state.applied_events[0].action_type, "cut");
    }

    #[test]
    fn test_event_from_entry() {
        let entry = EditHistoryEntry {
            timestamp_ms: 999,
            user_id: "bob".to_string(),
            action_type: "paste".to_string(),
            payload: r#"{}"#.to_string(),
        };
        let ev = Event::from_entry(&entry);
        assert_eq!(ev.timestamp_ms, 999);
        assert_eq!(ev.user_id, "bob");
    }

    #[test]
    fn test_replayer_is_empty() {
        let r = HistoryReplayer::new(vec![]);
        assert!(r.is_empty());
        assert_eq!(r.event_count(), 0);
    }
}
