//! User session management for media access control.
//!
//! Provides `SessionStatus`, `UserSession`, and `SessionManager`.

#![allow(dead_code)]

use std::collections::HashMap;

/// The lifecycle state of a user session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    /// Session is valid and usable.
    Active,
    /// Session has been explicitly ended by the user or server.
    Expired,
    /// Session was revoked due to a security event.
    Revoked,
    /// Session timed out due to inactivity.
    TimedOut,
}

impl SessionStatus {
    /// Returns `true` if the session can be used to authenticate requests.
    #[must_use]
    pub fn is_active(&self) -> bool {
        matches!(self, SessionStatus::Active)
    }

    /// Human-readable description.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            SessionStatus::Active => "Active",
            SessionStatus::Expired => "Expired",
            SessionStatus::Revoked => "Revoked",
            SessionStatus::TimedOut => "Timed Out",
        }
    }
}

/// A single authenticated user session.
#[derive(Debug, Clone)]
pub struct UserSession {
    /// Unique session token (opaque string).
    pub token: String,
    /// The user identifier this session belongs to.
    pub user_id: String,
    /// Unix timestamp (seconds) when this session was created.
    pub created_at: u64,
    /// Unix timestamp (seconds) when this session expires.
    pub expires_at: u64,
    /// Current status of the session.
    pub status: SessionStatus,
    /// Optional display name for the client application.
    pub client_name: Option<String>,
}

impl UserSession {
    /// Create a new active session.
    pub fn new(
        token: impl Into<String>,
        user_id: impl Into<String>,
        created_at: u64,
        expires_at: u64,
    ) -> Self {
        Self {
            token: token.into(),
            user_id: user_id.into(),
            created_at,
            expires_at,
            status: SessionStatus::Active,
            client_name: None,
        }
    }

    /// Attach a client application name to this session.
    pub fn with_client_name(mut self, name: impl Into<String>) -> Self {
        self.client_name = Some(name.into());
        self
    }

    /// Returns `true` when the session has passed its expiry time.
    ///
    /// `now` is the current Unix timestamp in seconds.
    #[must_use]
    pub fn is_expired_at(&self, now: u64) -> bool {
        now >= self.expires_at || !self.status.is_active()
    }

    /// Session duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> u64 {
        self.expires_at.saturating_sub(self.created_at)
    }
}

/// Central manager for creating and tracking user sessions.
#[derive(Debug, Default)]
pub struct SessionManager {
    /// Map from session token to session object.
    sessions: HashMap<String, UserSession>,
}

impl SessionManager {
    /// Create a new, empty `SessionManager`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create and register a new session.
    ///
    /// Returns the session token.
    pub fn create(
        &mut self,
        token: impl Into<String>,
        user_id: impl Into<String>,
        created_at: u64,
        expires_at: u64,
    ) -> String {
        let session = UserSession::new(token, user_id, created_at, expires_at);
        let t = session.token.clone();
        self.sessions.insert(t.clone(), session);
        t
    }

    /// Look up a session by token.
    #[must_use]
    pub fn get(&self, token: &str) -> Option<&UserSession> {
        self.sessions.get(token)
    }

    /// Expire (invalidate) a session by token.
    ///
    /// Returns `true` if the session was found and expired.
    pub fn expire(&mut self, token: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(token) {
            session.status = SessionStatus::Expired;
            true
        } else {
            false
        }
    }

    /// Revoke a session (security event).
    ///
    /// Returns `true` if the session was found and revoked.
    pub fn revoke(&mut self, token: &str) -> bool {
        if let Some(session) = self.sessions.get_mut(token) {
            session.status = SessionStatus::Revoked;
            true
        } else {
            false
        }
    }

    /// Count of sessions currently in `Active` status.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.sessions
            .values()
            .filter(|s| s.status.is_active())
            .count()
    }

    /// Total number of sessions regardless of status.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.sessions.len()
    }

    /// Remove all sessions that have expired relative to `now`.
    pub fn purge_expired(&mut self, now: u64) {
        self.sessions.retain(|_, s| !s.is_expired_at(now));
    }

    /// Returns all active sessions for a given user id.
    #[must_use]
    pub fn sessions_for_user(&self, user_id: &str) -> Vec<&UserSession> {
        self.sessions
            .values()
            .filter(|s| s.user_id == user_id && s.status.is_active())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(token: &str, user: &str, created: u64, expires: u64) -> UserSession {
        UserSession::new(token, user, created, expires)
    }

    #[test]
    fn session_status_active_is_active() {
        assert!(SessionStatus::Active.is_active());
    }

    #[test]
    fn session_status_expired_not_active() {
        assert!(!SessionStatus::Expired.is_active());
    }

    #[test]
    fn session_status_revoked_not_active() {
        assert!(!SessionStatus::Revoked.is_active());
    }

    #[test]
    fn session_status_description() {
        assert_eq!(SessionStatus::TimedOut.description(), "Timed Out");
        assert_eq!(SessionStatus::Active.description(), "Active");
    }

    #[test]
    fn user_session_is_expired_at_before_expiry() {
        let session = make_session("tok", "user1", 1000, 2000);
        assert!(!session.is_expired_at(1500));
    }

    #[test]
    fn user_session_is_expired_at_after_expiry() {
        let session = make_session("tok", "user1", 1000, 2000);
        assert!(session.is_expired_at(2000));
        assert!(session.is_expired_at(3000));
    }

    #[test]
    fn user_session_duration_secs() {
        let session = make_session("tok", "user1", 1000, 4600);
        assert_eq!(session.duration_secs(), 3600);
    }

    #[test]
    fn user_session_with_client_name() {
        let session = make_session("tok", "user1", 0, 9999).with_client_name("OxiMedia Desktop");
        assert_eq!(session.client_name.as_deref(), Some("OxiMedia Desktop"));
    }

    #[test]
    fn session_manager_create_and_get() {
        let mut mgr = SessionManager::new();
        let token = mgr.create("abc123", "alice", 0, 9999);
        let session = mgr.get(&token);
        assert!(session.is_some());
        assert_eq!(session.expect("test expectation failed").user_id, "alice");
    }

    #[test]
    fn session_manager_get_missing_returns_none() {
        let mgr = SessionManager::new();
        assert!(mgr.get("no-such-token").is_none());
    }

    #[test]
    fn session_manager_expire() {
        let mut mgr = SessionManager::new();
        let token = mgr.create("tok1", "bob", 0, 9999);
        assert!(mgr.expire(&token));
        assert!(!mgr
            .get(&token)
            .expect("get should succeed")
            .status
            .is_active());
    }

    #[test]
    fn session_manager_active_count() {
        let mut mgr = SessionManager::new();
        let t1 = mgr.create("t1", "alice", 0, 9999);
        mgr.create("t2", "bob", 0, 9999);
        mgr.expire(&t1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn session_manager_purge_expired() {
        let mut mgr = SessionManager::new();
        mgr.create("t1", "alice", 0, 100);
        mgr.create("t2", "bob", 0, 9999);
        mgr.purge_expired(200);
        assert_eq!(mgr.total_count(), 1);
    }

    #[test]
    fn session_manager_sessions_for_user() {
        let mut mgr = SessionManager::new();
        mgr.create("t1", "alice", 0, 9999);
        mgr.create("t2", "alice", 0, 9999);
        mgr.create("t3", "bob", 0, 9999);
        let alice_sessions = mgr.sessions_for_user("alice");
        assert_eq!(alice_sessions.len(), 2);
    }

    #[test]
    fn session_manager_revoke() {
        let mut mgr = SessionManager::new();
        let token = mgr.create("tok", "charlie", 0, 9999);
        assert!(mgr.revoke(&token));
        assert_eq!(
            mgr.get(&token).expect("get should succeed").status,
            SessionStatus::Revoked
        );
    }
}
