#![allow(dead_code)]
//! Session management for collaborative editing sessions.
//!
//! Provides high-level session lifecycle management, state tracking,
//! and metrics for active collaboration sessions.

use std::collections::HashMap;

/// State of a collaboration session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionState {
    /// Session is being initialized.
    Initializing,
    /// Session is active and accepting participants.
    Active,
    /// Session is paused (no edits accepted).
    Paused,
    /// Session is in the process of closing.
    Closing,
    /// Session has been closed.
    Closed,
}

impl SessionState {
    /// Returns true if the session can accept new participants.
    pub fn accepts_participants(&self) -> bool {
        matches!(self, SessionState::Active)
    }

    /// Returns true if the session accepts edit operations.
    pub fn accepts_edits(&self) -> bool {
        matches!(self, SessionState::Active)
    }

    /// Returns true if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionState::Closed)
    }
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Initializing => write!(f, "Initializing"),
            SessionState::Active => write!(f, "Active"),
            SessionState::Paused => write!(f, "Paused"),
            SessionState::Closing => write!(f, "Closing"),
            SessionState::Closed => write!(f, "Closed"),
        }
    }
}

/// Unique session identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionId(pub String);

impl SessionId {
    /// Create a new session ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Return the inner string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A collaboration session entry tracked by the manager.
#[derive(Debug, Clone)]
pub struct CollabSession {
    /// Unique identifier for this session.
    pub id: SessionId,
    /// Current state of the session.
    pub state: SessionState,
    /// Name of the project being edited.
    pub project_name: String,
    /// Number of participants currently in the session.
    pub participant_count: usize,
    /// Maximum participants allowed.
    pub max_participants: usize,
    /// Timestamp when the session was created (Unix seconds).
    pub created_at: u64,
}

impl CollabSession {
    /// Create a new session in `Initializing` state.
    pub fn new(
        id: SessionId,
        project_name: impl Into<String>,
        max_participants: usize,
        created_at: u64,
    ) -> Self {
        Self {
            id,
            state: SessionState::Initializing,
            project_name: project_name.into(),
            participant_count: 0,
            max_participants,
            created_at,
        }
    }

    /// Transition the session to `Active`.
    pub fn activate(&mut self) {
        if self.state == SessionState::Initializing {
            self.state = SessionState::Active;
        }
    }

    /// Pause the session.
    pub fn pause(&mut self) {
        if self.state == SessionState::Active {
            self.state = SessionState::Paused;
        }
    }

    /// Resume a paused session.
    pub fn resume(&mut self) {
        if self.state == SessionState::Paused {
            self.state = SessionState::Active;
        }
    }

    /// Begin closing the session.
    pub fn begin_close(&mut self) {
        if !self.state.is_terminal() {
            self.state = SessionState::Closing;
        }
    }

    /// Finalize closing the session.
    pub fn close(&mut self) {
        self.state = SessionState::Closed;
        self.participant_count = 0;
    }

    /// Returns true if the session has capacity for another participant.
    pub fn has_capacity(&self) -> bool {
        self.participant_count < self.max_participants
    }

    /// Add a participant if capacity allows.  Returns `false` if full.
    pub fn add_participant(&mut self) -> bool {
        if self.state.accepts_participants() && self.has_capacity() {
            self.participant_count += 1;
            true
        } else {
            false
        }
    }

    /// Remove a participant.  Saturating at zero.
    pub fn remove_participant(&mut self) {
        self.participant_count = self.participant_count.saturating_sub(1);
    }
}

/// Manages the lifecycle of multiple collaboration sessions.
#[derive(Debug, Default)]
pub struct SessionManager {
    sessions: HashMap<SessionId, CollabSession>,
}

impl SessionManager {
    /// Create a new, empty session manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new session.  Returns `false` if the ID is already in use.
    pub fn register(&mut self, session: CollabSession) -> bool {
        if self.sessions.contains_key(&session.id) {
            return false;
        }
        self.sessions.insert(session.id.clone(), session);
        true
    }

    /// Retrieve an immutable reference to a session.
    pub fn get(&self, id: &SessionId) -> Option<&CollabSession> {
        self.sessions.get(id)
    }

    /// Retrieve a mutable reference to a session.
    pub fn get_mut(&mut self, id: &SessionId) -> Option<&mut CollabSession> {
        self.sessions.get_mut(id)
    }

    /// Remove a session from the manager.
    pub fn remove(&mut self, id: &SessionId) -> Option<CollabSession> {
        self.sessions.remove(id)
    }

    /// Returns the number of currently tracked sessions.
    pub fn active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state == SessionState::Active)
            .count()
    }

    /// Returns the total number of sessions (all states).
    pub fn total_count(&self) -> usize {
        self.sessions.len()
    }

    /// Returns the total number of participants across all active sessions.
    pub fn total_participants(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.state == SessionState::Active)
            .map(|s| s.participant_count)
            .sum()
    }

    /// Collect IDs of all sessions in a given state.
    pub fn sessions_in_state(&self, state: SessionState) -> Vec<&SessionId> {
        self.sessions
            .values()
            .filter(|s| s.state == state)
            .map(|s| &s.id)
            .collect()
    }

    /// Close all sessions that are in `Closing` state.
    pub fn finalize_closing(&mut self) {
        for session in self.sessions.values_mut() {
            if session.state == SessionState::Closing {
                session.close();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: &str, max: usize) -> CollabSession {
        CollabSession::new(SessionId::new(id), "TestProject", max, 0)
    }

    #[test]
    fn test_session_state_accepts_participants() {
        assert!(!SessionState::Initializing.accepts_participants());
        assert!(SessionState::Active.accepts_participants());
        assert!(!SessionState::Paused.accepts_participants());
        assert!(!SessionState::Closing.accepts_participants());
        assert!(!SessionState::Closed.accepts_participants());
    }

    #[test]
    fn test_session_state_accepts_edits() {
        assert!(SessionState::Active.accepts_edits());
        assert!(!SessionState::Paused.accepts_edits());
        assert!(!SessionState::Closed.accepts_edits());
    }

    #[test]
    fn test_session_state_is_terminal() {
        assert!(!SessionState::Active.is_terminal());
        assert!(SessionState::Closed.is_terminal());
    }

    #[test]
    fn test_session_state_display() {
        assert_eq!(SessionState::Active.to_string(), "Active");
        assert_eq!(SessionState::Closed.to_string(), "Closed");
        assert_eq!(SessionState::Paused.to_string(), "Paused");
    }

    #[test]
    fn test_session_id_display() {
        let id = SessionId::new("sess-001");
        assert_eq!(id.to_string(), "sess-001");
        assert_eq!(id.as_str(), "sess-001");
    }

    #[test]
    fn test_session_activate() {
        let mut s = make_session("s1", 10);
        assert_eq!(s.state, SessionState::Initializing);
        s.activate();
        assert_eq!(s.state, SessionState::Active);
    }

    #[test]
    fn test_session_pause_resume() {
        let mut s = make_session("s2", 10);
        s.activate();
        s.pause();
        assert_eq!(s.state, SessionState::Paused);
        s.resume();
        assert_eq!(s.state, SessionState::Active);
    }

    #[test]
    fn test_session_close_lifecycle() {
        let mut s = make_session("s3", 10);
        s.activate();
        s.begin_close();
        assert_eq!(s.state, SessionState::Closing);
        s.close();
        assert_eq!(s.state, SessionState::Closed);
        assert_eq!(s.participant_count, 0);
    }

    #[test]
    fn test_session_add_remove_participant() {
        let mut s = make_session("s4", 2);
        s.activate();
        assert!(s.add_participant());
        assert!(s.add_participant());
        assert!(!s.add_participant()); // at capacity
        s.remove_participant();
        assert!(s.add_participant()); // space again
    }

    #[test]
    fn test_session_manager_active_count() {
        let mut mgr = SessionManager::new();
        let mut s1 = make_session("a1", 5);
        s1.activate();
        let s2 = make_session("a2", 5); // stays Initializing
        mgr.register(s1);
        mgr.register(s2);
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.total_count(), 2);
    }

    #[test]
    fn test_session_manager_register_duplicate() {
        let mut mgr = SessionManager::new();
        let s = make_session("dup", 5);
        let s2 = make_session("dup", 5);
        assert!(mgr.register(s));
        assert!(!mgr.register(s2)); // duplicate rejected
    }

    #[test]
    fn test_session_manager_remove() {
        let mut mgr = SessionManager::new();
        mgr.register(make_session("rm1", 5));
        let id = SessionId::new("rm1");
        assert!(mgr.remove(&id).is_some());
        assert_eq!(mgr.total_count(), 0);
    }

    #[test]
    fn test_session_manager_total_participants() {
        let mut mgr = SessionManager::new();
        let mut s1 = make_session("tp1", 5);
        s1.activate();
        s1.add_participant();
        s1.add_participant();
        let mut s2 = make_session("tp2", 5);
        s2.activate();
        s2.add_participant();
        mgr.register(s1);
        mgr.register(s2);
        assert_eq!(mgr.total_participants(), 3);
    }

    #[test]
    fn test_session_manager_finalize_closing() {
        let mut mgr = SessionManager::new();
        let mut s = make_session("fc1", 5);
        s.activate();
        s.begin_close();
        mgr.register(s);
        mgr.finalize_closing();
        let id = SessionId::new("fc1");
        assert_eq!(
            mgr.get(&id)
                .expect("collab test operation should succeed")
                .state,
            SessionState::Closed
        );
    }

    #[test]
    fn test_session_manager_sessions_in_state() {
        let mut mgr = SessionManager::new();
        let mut s1 = make_session("st1", 5);
        s1.activate();
        let s2 = make_session("st2", 5);
        mgr.register(s1);
        mgr.register(s2);
        let active = mgr.sessions_in_state(SessionState::Active);
        assert_eq!(active.len(), 1);
        let init = mgr.sessions_in_state(SessionState::Initializing);
        assert_eq!(init.len(), 1);
    }

    #[test]
    fn test_session_has_capacity() {
        let mut s = make_session("cap1", 1);
        s.activate();
        assert!(s.has_capacity());
        s.add_participant();
        assert!(!s.has_capacity());
    }
}
