//! Gaming session statistics for `oximedia-gaming`.
//!
//! Tracks individual session events (kills, deaths, points, etc.) and provides
//! aggregate analytics across multiple sessions for a player.

#![allow(dead_code)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

// ---------------------------------------------------------------------------
// SessionEvent
// ---------------------------------------------------------------------------

/// A timestamped event that occurred during a game session.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    /// Free-form event type label (e.g. `"kill"`, `"death"`, `"score"`).
    pub event_type: String,
    /// Unix timestamp in milliseconds when the event occurred.
    pub timestamp_ms: u64,
    /// Numeric value associated with the event (e.g. points scored).
    pub value: f64,
}

impl SessionEvent {
    /// Create a new session event.
    pub fn new(event_type: &str, timestamp_ms: u64, value: f64) -> Self {
        Self {
            event_type: event_type.to_string(),
            timestamp_ms,
            value,
        }
    }

    /// Age of this event relative to `now` in milliseconds.
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.timestamp_ms)
    }
}

// ---------------------------------------------------------------------------
// GameSession
// ---------------------------------------------------------------------------

/// A single gameplay session for one player.
#[derive(Debug, Clone)]
pub struct GameSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Player who played the session.
    pub player_id: String,
    /// Game title / identifier.
    pub game_id: String,
    /// When the session started (milliseconds since Unix epoch).
    pub started_ms: u64,
    /// When the session ended, or `None` if still active.
    pub ended_ms: Option<u64>,
    /// All events recorded during the session.
    pub events: Vec<SessionEvent>,
}

impl GameSession {
    /// Create a new active session.
    pub fn new(session_id: &str, player_id: &str, game_id: &str, started_ms: u64) -> Self {
        Self {
            session_id: session_id.to_string(),
            player_id: player_id.to_string(),
            game_id: game_id.to_string(),
            started_ms,
            ended_ms: None,
            events: Vec::new(),
        }
    }

    /// Duration of the session in milliseconds, or `None` if still active.
    pub fn duration_ms(&self) -> Option<u64> {
        self.ended_ms.map(|end| end.saturating_sub(self.started_ms))
    }

    /// Returns `true` when the session has not yet ended.
    pub fn is_active(&self) -> bool {
        self.ended_ms.is_none()
    }

    /// Total number of events recorded in this session.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Average `value` of events whose `event_type` matches `event_type`.
    ///
    /// Returns `None` if no matching events exist.
    pub fn avg_event_value(&self, event_type: &str) -> Option<f64> {
        let matching: Vec<f64> = self
            .events
            .iter()
            .filter(|e| e.event_type == event_type)
            .map(|e| e.value)
            .collect();
        if matching.is_empty() {
            return None;
        }
        Some(matching.iter().sum::<f64>() / matching.len() as f64)
    }

    /// Append an event to this session.
    pub fn record(&mut self, event: SessionEvent) {
        self.events.push(event);
    }

    /// End the session at `ended_ms`.
    pub fn end(&mut self, ended_ms: u64) {
        self.ended_ms = Some(ended_ms);
    }
}

// ---------------------------------------------------------------------------
// SessionStats
// ---------------------------------------------------------------------------

/// Aggregated statistics across multiple [`GameSession`] instances.
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// All sessions tracked (any player, any game).
    pub sessions: Vec<GameSession>,
}

impl SessionStats {
    /// Create an empty stats tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a session to the tracker.
    pub fn add(&mut self, s: GameSession) {
        self.sessions.push(s);
    }

    /// Total playtime in milliseconds for `player_id` across all their ended
    /// sessions.
    pub fn total_playtime_ms(&self, player_id: &str) -> u64 {
        self.sessions
            .iter()
            .filter(|s| s.player_id == player_id)
            .filter_map(GameSession::duration_ms)
            .sum()
    }

    /// Number of sessions for `player_id`.
    pub fn session_count(&self, player_id: &str) -> usize {
        self.sessions
            .iter()
            .filter(|s| s.player_id == player_id)
            .count()
    }

    /// Average session duration in milliseconds for `player_id`.
    ///
    /// Only ended sessions with a measurable duration are included.  Returns
    /// `0.0` if there are no qualifying sessions.
    pub fn avg_session_ms(&self, player_id: &str) -> f64 {
        let durations: Vec<u64> = self
            .sessions
            .iter()
            .filter(|s| s.player_id == player_id)
            .filter_map(GameSession::duration_ms)
            .collect();
        if durations.is_empty() {
            return 0.0;
        }
        durations.iter().sum::<u64>() as f64 / durations.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: &str, player: &str, start: u64, end: Option<u64>) -> GameSession {
        let mut s = GameSession::new(id, player, "game_x", start);
        if let Some(e) = end {
            s.end(e);
        }
        s
    }

    #[test]
    fn test_session_event_age_ms() {
        let ev = SessionEvent::new("kill", 1_000, 1.0);
        assert_eq!(ev.age_ms(3_000), 2_000);
    }

    #[test]
    fn test_session_event_age_ms_saturates() {
        let ev = SessionEvent::new("kill", 5_000, 1.0);
        assert_eq!(ev.age_ms(1_000), 0); // saturating_sub, no underflow
    }

    #[test]
    fn test_session_is_active_true() {
        let s = make_session("s1", "p1", 0, None);
        assert!(s.is_active());
    }

    #[test]
    fn test_session_is_active_false_after_end() {
        let s = make_session("s1", "p1", 0, Some(60_000));
        assert!(!s.is_active());
    }

    #[test]
    fn test_session_duration_ms_some() {
        let s = make_session("s1", "p1", 1_000, Some(61_000));
        assert_eq!(s.duration_ms(), Some(60_000));
    }

    #[test]
    fn test_session_duration_ms_none() {
        let s = make_session("s1", "p1", 0, None);
        assert_eq!(s.duration_ms(), None);
    }

    #[test]
    fn test_session_event_count() {
        let mut s = make_session("s1", "p1", 0, None);
        s.record(SessionEvent::new("kill", 100, 10.0));
        s.record(SessionEvent::new("kill", 200, 5.0));
        assert_eq!(s.event_count(), 2);
    }

    #[test]
    fn test_session_avg_event_value_some() {
        let mut s = make_session("s1", "p1", 0, None);
        s.record(SessionEvent::new("score", 100, 10.0));
        s.record(SessionEvent::new("score", 200, 20.0));
        let avg = s
            .avg_event_value("score")
            .expect("avg value should succeed");
        assert!((avg - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_session_avg_event_value_none_no_match() {
        let s = make_session("s1", "p1", 0, None);
        assert!(s.avg_event_value("nonexistent").is_none());
    }

    #[test]
    fn test_stats_total_playtime() {
        let mut stats = SessionStats::new();
        stats.add(make_session("s1", "alice", 0, Some(60_000)));
        stats.add(make_session("s2", "alice", 100_000, Some(160_000)));
        stats.add(make_session("s3", "bob", 0, Some(30_000)));
        assert_eq!(stats.total_playtime_ms("alice"), 120_000);
        assert_eq!(stats.total_playtime_ms("bob"), 30_000);
    }

    #[test]
    fn test_stats_session_count() {
        let mut stats = SessionStats::new();
        stats.add(make_session("s1", "alice", 0, Some(1)));
        stats.add(make_session("s2", "alice", 2, Some(3)));
        stats.add(make_session("s3", "bob", 0, Some(1)));
        assert_eq!(stats.session_count("alice"), 2);
        assert_eq!(stats.session_count("bob"), 1);
        assert_eq!(stats.session_count("nobody"), 0);
    }

    #[test]
    fn test_stats_avg_session_ms() {
        let mut stats = SessionStats::new();
        stats.add(make_session("s1", "alice", 0, Some(100)));
        stats.add(make_session("s2", "alice", 200, Some(400)));
        // durations: 100 and 200 → avg = 150
        assert!((stats.avg_session_ms("alice") - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_avg_session_ms_no_sessions() {
        let stats = SessionStats::new();
        assert_eq!(stats.avg_session_ms("nobody"), 0.0);
    }

    #[test]
    fn test_stats_ignores_active_sessions_in_playtime() {
        let mut stats = SessionStats::new();
        stats.add(make_session("s1", "alice", 0, Some(1_000)));
        stats.add(make_session("s2", "alice", 2_000, None)); // still active
        assert_eq!(stats.total_playtime_ms("alice"), 1_000);
    }
}
