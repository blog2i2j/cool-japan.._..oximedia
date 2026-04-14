//! Collaboration session lifecycle management.
//!
//! Covers the full lifetime of a collaborative editing session:
//!
//! * [`SessionState`] — state machine transitions (Created → Active → Idle →
//!   Suspended → Closed).
//! * [`ParticipantInfo`] — join/leave records for each user.
//! * [`ReconnectToken`] — opaque token issued on disconnect that allows
//!   seamless re-entry without losing progress.
//! * [`SessionLifecycle`] — the stateful coordinator that manages participants,
//!   state transitions, and reconnect token issuance.
//!
//! # State machine
//!
//! ```text
//! Created ──► Active ──► Idle ──► Suspended ──► Closed
//!                │                    │
//!                └────────────────────┘ (re-activate)
//!                         │
//!                         └──► Closed (force-close)
//! ```
//!
//! Transitions enforce invariants (e.g. a closed session cannot be re-opened)
//! and return `SessionLifecycleError` on invalid transitions.

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// State machine
// ─────────────────────────────────────────────────────────────────────────────

/// Lifecycle state of a collaboration session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// The session has been created but has not yet received its first
    /// participant.
    Created,
    /// At least one participant is connected and actively editing.
    Active,
    /// No participants are connected but the session is still open.
    Idle,
    /// The session has been explicitly suspended (e.g. server maintenance).
    /// Participants may reconnect after the suspension is lifted.
    Suspended,
    /// The session is permanently closed.  No further operations are allowed.
    Closed,
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Created => "created",
            Self::Active => "active",
            Self::Idle => "idle",
            Self::Suspended => "suspended",
            Self::Closed => "closed",
        };
        write!(f, "{s}")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Participant info
// ─────────────────────────────────────────────────────────────────────────────

/// Connection status of a participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticipantStatus {
    /// Currently connected to the session.
    Connected,
    /// Temporarily disconnected; holds a valid reconnect token.
    Disconnected,
    /// Has permanently left the session.
    Left,
}

impl std::fmt::Display for ParticipantStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Connected => "connected",
            Self::Disconnected => "disconnected",
            Self::Left => "left",
        };
        write!(f, "{s}")
    }
}

/// Information about a single session participant.
#[derive(Debug, Clone)]
pub struct ParticipantInfo {
    /// Unique user identifier.
    pub user_id: String,
    /// Display name.
    pub display_name: String,
    /// Current connection status.
    pub status: ParticipantStatus,
    /// Wall-clock time (ms) when the participant joined.
    pub joined_at_ms: u64,
    /// Wall-clock time (ms) of the most recent status change.
    pub last_seen_ms: u64,
    /// Number of times this participant has reconnected.
    pub reconnect_count: u32,
}

impl ParticipantInfo {
    /// Create a new connected participant record.
    pub fn new(
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        now_ms: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            status: ParticipantStatus::Connected,
            joined_at_ms: now_ms,
            last_seen_ms: now_ms,
            reconnect_count: 0,
        }
    }

    /// Duration since the participant joined, in milliseconds.
    pub fn session_duration_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.joined_at_ms)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reconnect token
// ─────────────────────────────────────────────────────────────────────────────

/// An opaque token issued to a disconnecting participant.
///
/// The token can be presented on reconnect to restore session state without
/// re-authenticating from scratch.  Tokens expire after `valid_until_ms`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconnectToken {
    /// The token string (treated as opaque by the lifecycle manager).
    pub token: String,
    /// User ID the token was issued for.
    pub user_id: String,
    /// Session ID this token belongs to.
    pub session_id: String,
    /// Wall-clock expiry timestamp in milliseconds.
    pub valid_until_ms: u64,
}

impl ReconnectToken {
    /// Create a new reconnect token.
    pub fn new(
        token: impl Into<String>,
        user_id: impl Into<String>,
        session_id: impl Into<String>,
        valid_until_ms: u64,
    ) -> Self {
        Self {
            token: token.into(),
            user_id: user_id.into(),
            session_id: session_id.into(),
            valid_until_ms,
        }
    }

    /// Whether the token is still valid at `now_ms`.
    pub fn is_valid(&self, now_ms: u64) -> bool {
        now_ms <= self.valid_until_ms
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the session lifecycle manager.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionLifecycleError {
    /// The requested state transition is not allowed from the current state.
    InvalidTransition {
        from: SessionState,
        to: SessionState,
    },
    /// The session is closed and no further operations are accepted.
    SessionClosed,
    /// The specified participant was not found.
    ParticipantNotFound(String),
    /// The participant has already left the session.
    ParticipantAlreadyLeft(String),
    /// The reconnect token is invalid or has expired.
    InvalidToken,
    /// A participant with the same user id is already in the session.
    DuplicateParticipant(String),
}

impl std::fmt::Display for SessionLifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidTransition { from, to } => {
                write!(f, "invalid transition: {from} → {to}")
            }
            Self::SessionClosed => write!(f, "session is closed"),
            Self::ParticipantNotFound(id) => write!(f, "participant not found: {id}"),
            Self::ParticipantAlreadyLeft(id) => write!(f, "participant already left: {id}"),
            Self::InvalidToken => write!(f, "reconnect token is invalid or expired"),
            Self::DuplicateParticipant(id) => {
                write!(f, "participant already in session: {id}")
            }
        }
    }
}

impl std::error::Error for SessionLifecycleError {}

// ─────────────────────────────────────────────────────────────────────────────
// Lifecycle manager
// ─────────────────────────────────────────────────────────────────────────────

/// Token TTL: 24 hours in milliseconds.
const DEFAULT_TOKEN_TTL_MS: u64 = 24 * 60 * 60 * 1000;

/// Manages the lifecycle of a single collaboration session.
#[derive(Debug)]
pub struct SessionLifecycle {
    /// Unique identifier of this session.
    session_id: String,
    /// Current lifecycle state.
    state: SessionState,
    /// Wall-clock time when the session was created (ms).
    created_at_ms: u64,
    /// Wall-clock time of the last state transition (ms).
    last_transition_ms: u64,
    /// Registered participants, keyed by user id.
    participants: HashMap<String, ParticipantInfo>,
    /// Active reconnect tokens, keyed by token string.
    reconnect_tokens: HashMap<String, ReconnectToken>,
    /// Counter used to generate simple token strings.
    token_counter: u64,
    /// TTL for reconnect tokens in milliseconds.
    token_ttl_ms: u64,
    /// Running log of `(timestamp_ms, event_description)` pairs.
    event_log: Vec<(u64, String)>,
}

impl SessionLifecycle {
    /// Create a new session lifecycle in the `Created` state.
    pub fn new(session_id: impl Into<String>, now_ms: u64) -> Self {
        Self {
            session_id: session_id.into(),
            state: SessionState::Created,
            created_at_ms: now_ms,
            last_transition_ms: now_ms,
            participants: HashMap::new(),
            reconnect_tokens: HashMap::new(),
            token_counter: 0,
            token_ttl_ms: DEFAULT_TOKEN_TTL_MS,
            event_log: Vec::new(),
        }
    }

    /// Override the reconnect token TTL.
    pub fn with_token_ttl_ms(mut self, ttl_ms: u64) -> Self {
        self.token_ttl_ms = ttl_ms;
        self
    }

    // ── State accessors ────────────────────────────────────────────────────

    /// Current lifecycle state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Session identifier.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Whether the session is accepting new participants.
    pub fn is_open(&self) -> bool {
        !matches!(self.state, SessionState::Closed)
    }

    // ── Participant management ─────────────────────────────────────────────

    /// Add a new participant to the session.
    ///
    /// Automatically transitions from `Created` → `Active` when the first
    /// participant joins.
    pub fn add_participant(
        &mut self,
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        now_ms: u64,
    ) -> Result<(), SessionLifecycleError> {
        if matches!(self.state, SessionState::Closed) {
            return Err(SessionLifecycleError::SessionClosed);
        }
        let uid = user_id.into();
        if let Some(existing) = self.participants.get(&uid) {
            if existing.status != ParticipantStatus::Left {
                return Err(SessionLifecycleError::DuplicateParticipant(uid));
            }
        }
        let info = ParticipantInfo::new(uid.clone(), display_name, now_ms);
        self.participants.insert(uid.clone(), info);
        self.log(now_ms, format!("participant {uid} joined"));

        // Auto-transition: Created → Active on first join.
        if matches!(self.state, SessionState::Created | SessionState::Idle) {
            self.transition_to(SessionState::Active, now_ms)?;
        }
        Ok(())
    }

    /// Mark a participant as having left permanently.
    ///
    /// Automatically transitions to `Idle` if no connected participants remain.
    pub fn remove_participant(
        &mut self,
        user_id: &str,
        now_ms: u64,
    ) -> Result<(), SessionLifecycleError> {
        let info = self
            .participants
            .get_mut(user_id)
            .ok_or_else(|| SessionLifecycleError::ParticipantNotFound(user_id.to_string()))?;

        if info.status == ParticipantStatus::Left {
            return Err(SessionLifecycleError::ParticipantAlreadyLeft(
                user_id.to_string(),
            ));
        }

        info.status = ParticipantStatus::Left;
        info.last_seen_ms = now_ms;
        self.log(now_ms, format!("participant {user_id} left"));

        // Revoke any tokens held by this user.
        self.reconnect_tokens
            .retain(|_, t| t.user_id != user_id);

        // Transition to Idle if no connected participants remain.
        if self.connected_count() == 0 && matches!(self.state, SessionState::Active) {
            self.transition_to(SessionState::Idle, now_ms)?;
        }
        Ok(())
    }

    /// Temporarily disconnect a participant and issue a reconnect token.
    ///
    /// The participant's status is set to `Disconnected` and a token is
    /// returned.  The token expires after `self.token_ttl_ms` milliseconds.
    pub fn disconnect_participant(
        &mut self,
        user_id: &str,
        now_ms: u64,
    ) -> Result<ReconnectToken, SessionLifecycleError> {
        let info = self
            .participants
            .get_mut(user_id)
            .ok_or_else(|| SessionLifecycleError::ParticipantNotFound(user_id.to_string()))?;

        if info.status == ParticipantStatus::Left {
            return Err(SessionLifecycleError::ParticipantAlreadyLeft(
                user_id.to_string(),
            ));
        }

        info.status = ParticipantStatus::Disconnected;
        info.last_seen_ms = now_ms;

        // Mint a token.
        self.token_counter += 1;
        let token_str = format!(
            "tok-{}-{}-{}",
            self.session_id, user_id, self.token_counter
        );
        let token = ReconnectToken::new(
            &token_str,
            user_id,
            &self.session_id,
            now_ms + self.token_ttl_ms,
        );
        self.reconnect_tokens.insert(token_str.clone(), token.clone());
        self.log(now_ms, format!("participant {user_id} disconnected"));

        // Transition to Idle if no connected participants remain.
        if self.connected_count() == 0 && matches!(self.state, SessionState::Active) {
            // Best-effort; ignore errors here (state may already be Idle).
            let _ = self.transition_to(SessionState::Idle, now_ms);
        }

        Ok(token)
    }

    /// Reconnect a participant using a valid reconnect token.
    ///
    /// Returns the participant's display name on success.
    pub fn reconnect_participant(
        &mut self,
        token_str: &str,
        now_ms: u64,
    ) -> Result<String, SessionLifecycleError> {
        if matches!(self.state, SessionState::Closed) {
            return Err(SessionLifecycleError::SessionClosed);
        }

        let token = self
            .reconnect_tokens
            .get(token_str)
            .ok_or(SessionLifecycleError::InvalidToken)?;

        if !token.is_valid(now_ms) {
            return Err(SessionLifecycleError::InvalidToken);
        }

        let user_id = token.user_id.clone();
        // Remove token (one-time use).
        self.reconnect_tokens.remove(token_str);

        let info = self
            .participants
            .get_mut(&user_id)
            .ok_or_else(|| SessionLifecycleError::ParticipantNotFound(user_id.clone()))?;

        info.status = ParticipantStatus::Connected;
        info.last_seen_ms = now_ms;
        info.reconnect_count += 1;
        let display_name = info.display_name.clone();
        self.log(now_ms, format!("participant {user_id} reconnected"));

        // Transition back to Active if we were Idle or Suspended.
        if matches!(self.state, SessionState::Idle | SessionState::Suspended) {
            self.transition_to(SessionState::Active, now_ms)?;
        }

        Ok(display_name)
    }

    // ── State transitions ──────────────────────────────────────────────────

    /// Explicitly suspend the session.
    pub fn suspend(&mut self, now_ms: u64) -> Result<(), SessionLifecycleError> {
        self.transition_to(SessionState::Suspended, now_ms)
    }

    /// Re-activate a suspended or idle session.
    pub fn reactivate(&mut self, now_ms: u64) -> Result<(), SessionLifecycleError> {
        self.transition_to(SessionState::Active, now_ms)
    }

    /// Permanently close the session.
    pub fn close(&mut self, now_ms: u64) -> Result<(), SessionLifecycleError> {
        self.transition_to(SessionState::Closed, now_ms)
    }

    // ── Queries ────────────────────────────────────────────────────────────

    /// Number of currently connected participants.
    pub fn connected_count(&self) -> usize {
        self.participants
            .values()
            .filter(|p| p.status == ParticipantStatus::Connected)
            .count()
    }

    /// All participants (any status).
    pub fn all_participants(&self) -> Vec<&ParticipantInfo> {
        self.participants.values().collect()
    }

    /// Connected participants only.
    pub fn connected_participants(&self) -> Vec<&ParticipantInfo> {
        self.participants
            .values()
            .filter(|p| p.status == ParticipantStatus::Connected)
            .collect()
    }

    /// Look up a participant by user id.
    pub fn participant(&self, user_id: &str) -> Option<&ParticipantInfo> {
        self.participants.get(user_id)
    }

    /// Number of currently active (unexpired) reconnect tokens.
    pub fn active_token_count(&self, now_ms: u64) -> usize {
        self.reconnect_tokens
            .values()
            .filter(|t| t.is_valid(now_ms))
            .count()
    }

    /// Purge expired reconnect tokens.
    pub fn purge_expired_tokens(&mut self, now_ms: u64) {
        self.reconnect_tokens.retain(|_, t| t.is_valid(now_ms));
    }

    /// Wall-clock time the session was created (ms).
    pub fn created_at_ms(&self) -> u64 {
        self.created_at_ms
    }

    /// Session age in milliseconds.
    pub fn age_ms(&self, now_ms: u64) -> u64 {
        now_ms.saturating_sub(self.created_at_ms)
    }

    /// Read-only access to the event log.
    pub fn event_log(&self) -> &[(u64, String)] {
        &self.event_log
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn transition_to(
        &mut self,
        target: SessionState,
        now_ms: u64,
    ) -> Result<(), SessionLifecycleError> {
        let allowed = match self.state {
            SessionState::Created => matches!(target, SessionState::Active | SessionState::Closed),
            SessionState::Active => matches!(
                target,
                SessionState::Idle | SessionState::Suspended | SessionState::Closed
            ),
            SessionState::Idle => matches!(
                target,
                SessionState::Active | SessionState::Suspended | SessionState::Closed
            ),
            SessionState::Suspended => {
                matches!(target, SessionState::Active | SessionState::Closed)
            }
            SessionState::Closed => false,
        };

        if !allowed {
            return Err(SessionLifecycleError::InvalidTransition {
                from: self.state,
                to: target,
            });
        }

        self.log(
            now_ms,
            format!("state transition: {} → {}", self.state, target),
        );
        self.state = target;
        self.last_transition_ms = now_ms;
        Ok(())
    }

    fn log(&mut self, now_ms: u64, msg: impl Into<String>) {
        self.event_log.push((now_ms, msg.into()));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session() -> SessionLifecycle {
        SessionLifecycle::new("session:1", 0)
    }

    // ── State transitions ──────────────────────────────────────────────────

    #[test]
    fn test_initial_state_is_created() {
        let s = make_session();
        assert_eq!(s.state(), SessionState::Created);
    }

    #[test]
    fn test_add_participant_transitions_to_active() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("add should succeed");
        assert_eq!(s.state(), SessionState::Active);
    }

    #[test]
    fn test_remove_all_participants_transitions_to_idle() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("add should succeed");
        s.remove_participant("alice", 200)
            .expect("remove should succeed");
        assert_eq!(s.state(), SessionState::Idle);
    }

    #[test]
    fn test_close_session() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("should succeed");
        s.close(500).expect("should succeed");
        assert_eq!(s.state(), SessionState::Closed);
        assert!(!s.is_open());
    }

    #[test]
    fn test_add_participant_to_closed_session_fails() {
        let mut s = make_session();
        s.close(0).expect("should succeed");
        let result = s.add_participant("bob", "Bob", 100);
        assert!(matches!(result, Err(SessionLifecycleError::SessionClosed)));
    }

    #[test]
    fn test_suspend_and_reactivate() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("should succeed");
        s.suspend(200).expect("should succeed");
        assert_eq!(s.state(), SessionState::Suspended);
        s.reactivate(300).expect("should succeed");
        assert_eq!(s.state(), SessionState::Active);
    }

    #[test]
    fn test_invalid_transition_from_closed() {
        let mut s = make_session();
        s.close(0).expect("should succeed");
        let result = s.reactivate(100);
        assert!(matches!(
            result,
            Err(SessionLifecycleError::InvalidTransition { .. })
        ));
    }

    // ── Participant management ─────────────────────────────────────────────

    #[test]
    fn test_duplicate_participant_rejected() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("first add should succeed");
        let result = s.add_participant("alice", "Alice2", 200);
        assert!(matches!(
            result,
            Err(SessionLifecycleError::DuplicateParticipant(_))
        ));
    }

    #[test]
    fn test_remove_unknown_participant_fails() {
        let mut s = make_session();
        let result = s.remove_participant("ghost", 0);
        assert!(matches!(
            result,
            Err(SessionLifecycleError::ParticipantNotFound(_))
        ));
    }

    #[test]
    fn test_connected_count() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 0)
            .expect("should succeed");
        s.add_participant("bob", "Bob", 0)
            .expect("should succeed");
        assert_eq!(s.connected_count(), 2);
        s.remove_participant("alice", 100)
            .expect("should succeed");
        assert_eq!(s.connected_count(), 1);
    }

    // ── Reconnect tokens ───────────────────────────────────────────────────

    #[test]
    fn test_disconnect_and_reconnect() {
        let mut s = SessionLifecycle::new("session:1", 0).with_token_ttl_ms(60_000);
        s.add_participant("alice", "Alice", 0)
            .expect("should succeed");
        let token = s
            .disconnect_participant("alice", 1000)
            .expect("should succeed");
        assert!(token.is_valid(1000));
        assert!(!token.is_valid(1_000_000)); // well past expiry

        // Session should be Idle now (no connected participants).
        assert_eq!(s.state(), SessionState::Idle);

        // Reconnect using the token.
        let name = s
            .reconnect_participant(&token.token, 2000)
            .expect("reconnect should succeed");
        assert_eq!(name, "Alice");
        assert_eq!(s.state(), SessionState::Active);

        // Token should be consumed (one-time use).
        let result = s.reconnect_participant(&token.token, 3000);
        assert!(matches!(result, Err(SessionLifecycleError::InvalidToken)));
    }

    #[test]
    fn test_expired_token_rejected() {
        let mut s = SessionLifecycle::new("session:1", 0).with_token_ttl_ms(100);
        s.add_participant("alice", "Alice", 0)
            .expect("should succeed");
        let token = s
            .disconnect_participant("alice", 0)
            .expect("should succeed");
        // Token expires at ms=100; try to use at ms=200.
        let result = s.reconnect_participant(&token.token, 200);
        assert!(matches!(result, Err(SessionLifecycleError::InvalidToken)));
    }

    #[test]
    fn test_purge_expired_tokens() {
        let mut s = SessionLifecycle::new("session:1", 0).with_token_ttl_ms(100);
        s.add_participant("alice", "Alice", 0)
            .expect("should succeed");
        s.disconnect_participant("alice", 0)
            .expect("should succeed");
        assert_eq!(s.active_token_count(50), 1); // not yet expired
        s.purge_expired_tokens(200); // expire everything
        assert_eq!(s.active_token_count(200), 0);
    }

    // ── Event log ──────────────────────────────────────────────────────────

    #[test]
    fn test_event_log_records_activity() {
        let mut s = make_session();
        s.add_participant("alice", "Alice", 100)
            .expect("should succeed");
        s.remove_participant("alice", 200)
            .expect("should succeed");
        let log = s.event_log();
        // Should have at least: join + Created→Active + leave + Active→Idle
        assert!(log.len() >= 2);
        assert!(log.iter().any(|(_, msg)| msg.contains("joined")));
        assert!(log.iter().any(|(_, msg)| msg.contains("left")));
    }

    // ── Age ────────────────────────────────────────────────────────────────

    #[test]
    fn test_age_ms() {
        let s = SessionLifecycle::new("session:1", 1000);
        assert_eq!(s.age_ms(6000), 5000);
        assert_eq!(s.age_ms(500), 0); // saturating_sub
    }

    // ── SessionState display ───────────────────────────────────────────────

    #[test]
    fn test_state_display() {
        assert_eq!(SessionState::Created.to_string(), "created");
        assert_eq!(SessionState::Active.to_string(), "active");
        assert_eq!(SessionState::Idle.to_string(), "idle");
        assert_eq!(SessionState::Suspended.to_string(), "suspended");
        assert_eq!(SessionState::Closed.to_string(), "closed");
    }
}
