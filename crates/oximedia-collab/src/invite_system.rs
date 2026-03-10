#![allow(dead_code)]
//! Invite system for collaboration sessions.
//!
//! Handles generation, distribution, and acceptance of invitations
//! for users to join a collaborative editing session.

use std::collections::HashMap;

/// Current status of an invitation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InviteStatus {
    /// Invite has been created but not yet sent.
    Pending,
    /// Invite was sent and is awaiting a response.
    Sent,
    /// Recipient accepted the invite.
    Accepted,
    /// Recipient declined the invite.
    Declined,
    /// Invite has expired without a response.
    Expired,
    /// Invite was revoked by the sender.
    Revoked,
}

impl InviteStatus {
    /// Returns true if the invite is still actionable (can be accepted or declined).
    pub fn is_actionable(&self) -> bool {
        matches!(self, InviteStatus::Sent | InviteStatus::Pending)
    }

    /// Returns true if the invite has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            InviteStatus::Accepted
                | InviteStatus::Declined
                | InviteStatus::Expired
                | InviteStatus::Revoked
        )
    }
}

impl std::fmt::Display for InviteStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InviteStatus::Pending => write!(f, "Pending"),
            InviteStatus::Sent => write!(f, "Sent"),
            InviteStatus::Accepted => write!(f, "Accepted"),
            InviteStatus::Declined => write!(f, "Declined"),
            InviteStatus::Expired => write!(f, "Expired"),
            InviteStatus::Revoked => write!(f, "Revoked"),
        }
    }
}

/// Role granted to the invitee upon accepting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InviteRole {
    /// Full editor privileges.
    Editor,
    /// Read-only access.
    Viewer,
    /// Comment-only access.
    Commenter,
}

impl std::fmt::Display for InviteRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InviteRole::Editor => write!(f, "Editor"),
            InviteRole::Viewer => write!(f, "Viewer"),
            InviteRole::Commenter => write!(f, "Commenter"),
        }
    }
}

/// A unique token used to identify an invitation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InviteToken(pub String);

impl InviteToken {
    /// Create an invite token from a string.
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }

    /// Return the inner token string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for InviteToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An invitation to join a collaboration session.
#[derive(Debug, Clone)]
pub struct Invite {
    /// Unique token for this invitation.
    pub token: InviteToken,
    /// Session the invitee will join.
    pub session_id: String,
    /// Email or username of the invitee.
    pub invitee: String,
    /// Role the invitee will receive.
    pub role: InviteRole,
    /// Current status of this invite.
    pub status: InviteStatus,
    /// Unix timestamp of invite creation.
    pub created_at: u64,
    /// Unix timestamp after which the invite expires (None = no expiry).
    pub expires_at: Option<u64>,
}

impl Invite {
    /// Create a new pending invite.
    pub fn new(
        token: InviteToken,
        session_id: impl Into<String>,
        invitee: impl Into<String>,
        role: InviteRole,
        created_at: u64,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            token,
            session_id: session_id.into(),
            invitee: invitee.into(),
            role,
            status: InviteStatus::Pending,
            created_at,
            expires_at,
        }
    }

    /// Mark the invite as sent.
    pub fn mark_sent(&mut self) {
        if self.status == InviteStatus::Pending {
            self.status = InviteStatus::Sent;
        }
    }

    /// Check whether this invite has expired given the current timestamp.
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.map(|exp| now >= exp).unwrap_or(false)
    }

    /// Attempt to expire the invite if the current time is past its expiry.
    /// Returns `true` if the invite was newly expired.
    pub fn try_expire(&mut self, now: u64) -> bool {
        if self.status.is_actionable() && self.is_expired(now) {
            self.status = InviteStatus::Expired;
            return true;
        }
        false
    }
}

/// Error type for invite operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InviteError {
    /// No invite found for the given token.
    NotFound,
    /// The invite is not in a state that allows this action.
    InvalidState(InviteStatus),
    /// The invite has already expired.
    Expired,
}

impl std::fmt::Display for InviteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InviteError::NotFound => write!(f, "Invite not found"),
            InviteError::InvalidState(s) => write!(f, "Invalid invite state: {}", s),
            InviteError::Expired => write!(f, "Invite has expired"),
        }
    }
}

/// Manages invite creation, dispatch, and acceptance for collaboration sessions.
#[derive(Debug, Default)]
pub struct InviteSystem {
    invites: HashMap<InviteToken, Invite>,
    /// Monotonic counter used to generate simple sequential tokens in tests.
    counter: u64,
}

impl InviteSystem {
    /// Create a new, empty invite system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate a simple sequential token (suitable for deterministic testing).
    fn next_token(&mut self) -> InviteToken {
        self.counter += 1;
        InviteToken::new(format!("invite-{:06}", self.counter))
    }

    /// Issue a new invitation, returning the generated token.
    pub fn issue(
        &mut self,
        session_id: impl Into<String>,
        invitee: impl Into<String>,
        role: InviteRole,
        now: u64,
        expires_at: Option<u64>,
    ) -> InviteToken {
        let token = self.next_token();
        let invite = Invite::new(token.clone(), session_id, invitee, role, now, expires_at);
        self.invites.insert(token.clone(), invite);
        token
    }

    /// Send a pending invite (transitions Pending → Sent).
    pub fn send(&mut self, token: &InviteToken) -> Result<(), InviteError> {
        let invite = self.invites.get_mut(token).ok_or(InviteError::NotFound)?;
        if invite.status != InviteStatus::Pending {
            return Err(InviteError::InvalidState(invite.status));
        }
        invite.mark_sent();
        Ok(())
    }

    /// Accept an invite, transitioning it to `Accepted`.
    ///
    /// Returns the role that was granted, or an error.
    pub fn accept(&mut self, token: &InviteToken, now: u64) -> Result<InviteRole, InviteError> {
        let invite = self.invites.get_mut(token).ok_or(InviteError::NotFound)?;
        if invite.try_expire(now) {
            return Err(InviteError::Expired);
        }
        if invite.status != InviteStatus::Sent {
            return Err(InviteError::InvalidState(invite.status));
        }
        let role = invite.role;
        invite.status = InviteStatus::Accepted;
        Ok(role)
    }

    /// Decline an invite, transitioning it to `Declined`.
    pub fn decline(&mut self, token: &InviteToken, now: u64) -> Result<(), InviteError> {
        let invite = self.invites.get_mut(token).ok_or(InviteError::NotFound)?;
        if invite.try_expire(now) {
            return Err(InviteError::Expired);
        }
        if invite.status != InviteStatus::Sent {
            return Err(InviteError::InvalidState(invite.status));
        }
        invite.status = InviteStatus::Declined;
        Ok(())
    }

    /// Revoke an invite regardless of its current actionable state.
    pub fn revoke(&mut self, token: &InviteToken) -> Result<(), InviteError> {
        let invite = self.invites.get_mut(token).ok_or(InviteError::NotFound)?;
        if invite.status.is_terminal() {
            return Err(InviteError::InvalidState(invite.status));
        }
        invite.status = InviteStatus::Revoked;
        Ok(())
    }

    /// Expire all invites whose expiry timestamp is <= `now`.
    /// Returns the count of newly expired invites.
    pub fn expire_stale(&mut self, now: u64) -> usize {
        let mut count = 0;
        for invite in self.invites.values_mut() {
            if invite.try_expire(now) {
                count += 1;
            }
        }
        count
    }

    /// Look up an invite by token.
    pub fn get(&self, token: &InviteToken) -> Option<&Invite> {
        self.invites.get(token)
    }

    /// Total number of invites tracked.
    pub fn total_count(&self) -> usize {
        self.invites.len()
    }

    /// Number of invites currently in `Accepted` state.
    pub fn accepted_count(&self) -> usize {
        self.invites
            .values()
            .filter(|i| i.status == InviteStatus::Accepted)
            .count()
    }

    /// Number of invites currently in an actionable state.
    pub fn pending_count(&self) -> usize {
        self.invites
            .values()
            .filter(|i| i.status.is_actionable())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invite_status_is_actionable() {
        assert!(InviteStatus::Pending.is_actionable());
        assert!(InviteStatus::Sent.is_actionable());
        assert!(!InviteStatus::Accepted.is_actionable());
        assert!(!InviteStatus::Declined.is_actionable());
        assert!(!InviteStatus::Expired.is_actionable());
        assert!(!InviteStatus::Revoked.is_actionable());
    }

    #[test]
    fn test_invite_status_is_terminal() {
        assert!(!InviteStatus::Pending.is_terminal());
        assert!(!InviteStatus::Sent.is_terminal());
        assert!(InviteStatus::Accepted.is_terminal());
        assert!(InviteStatus::Expired.is_terminal());
        assert!(InviteStatus::Revoked.is_terminal());
    }

    #[test]
    fn test_invite_status_display() {
        assert_eq!(InviteStatus::Accepted.to_string(), "Accepted");
        assert_eq!(InviteStatus::Revoked.to_string(), "Revoked");
    }

    #[test]
    fn test_invite_role_display() {
        assert_eq!(InviteRole::Editor.to_string(), "Editor");
        assert_eq!(InviteRole::Viewer.to_string(), "Viewer");
        assert_eq!(InviteRole::Commenter.to_string(), "Commenter");
    }

    #[test]
    fn test_invite_token_display() {
        let t = InviteToken::new("tok-001");
        assert_eq!(t.to_string(), "tok-001");
        assert_eq!(t.as_str(), "tok-001");
    }

    #[test]
    fn test_invite_is_expired() {
        let inv = Invite::new(
            InviteToken::new("t"),
            "s1",
            "user@example.com",
            InviteRole::Editor,
            100,
            Some(200),
        );
        assert!(!inv.is_expired(150));
        assert!(inv.is_expired(200));
        assert!(inv.is_expired(300));
    }

    #[test]
    fn test_invite_no_expiry() {
        let inv = Invite::new(
            InviteToken::new("t2"),
            "s2",
            "user",
            InviteRole::Viewer,
            0,
            None,
        );
        assert!(!inv.is_expired(u64::MAX));
    }

    #[test]
    fn test_issue_and_send() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess1", "alice", InviteRole::Editor, 0, None);
        assert_eq!(
            sys.get(&tok)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Pending
        );
        sys.send(&tok)
            .expect("collab test operation should succeed");
        assert_eq!(
            sys.get(&tok)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Sent
        );
    }

    #[test]
    fn test_accept_returns_role() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess2", "bob", InviteRole::Viewer, 0, None);
        sys.send(&tok)
            .expect("collab test operation should succeed");
        let role = sys
            .accept(&tok, 1)
            .expect("collab test operation should succeed");
        assert_eq!(role, InviteRole::Viewer);
        assert_eq!(sys.accepted_count(), 1);
    }

    #[test]
    fn test_accept_expired_invite() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess3", "charlie", InviteRole::Editor, 0, Some(100));
        sys.send(&tok)
            .expect("collab test operation should succeed");
        let err = sys.accept(&tok, 200).unwrap_err();
        assert_eq!(err, InviteError::Expired);
    }

    #[test]
    fn test_decline() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess4", "dave", InviteRole::Commenter, 0, None);
        sys.send(&tok)
            .expect("collab test operation should succeed");
        sys.decline(&tok, 0)
            .expect("collab test operation should succeed");
        assert_eq!(
            sys.get(&tok)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Declined
        );
    }

    #[test]
    fn test_revoke() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess5", "eve", InviteRole::Editor, 0, None);
        sys.revoke(&tok)
            .expect("collab test operation should succeed");
        assert_eq!(
            sys.get(&tok)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Revoked
        );
    }

    #[test]
    fn test_revoke_terminal_state_fails() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("sess6", "frank", InviteRole::Viewer, 0, None);
        sys.send(&tok)
            .expect("collab test operation should succeed");
        sys.accept(&tok, 0)
            .expect("collab test operation should succeed");
        let err = sys.revoke(&tok).unwrap_err();
        assert!(matches!(err, InviteError::InvalidState(_)));
    }

    #[test]
    fn test_expire_stale() {
        let mut sys = InviteSystem::new();
        let tok1 = sys.issue("s", "u1", InviteRole::Editor, 0, Some(50));
        let tok2 = sys.issue("s", "u2", InviteRole::Editor, 0, Some(200));
        sys.send(&tok1)
            .expect("collab test operation should succeed");
        sys.send(&tok2)
            .expect("collab test operation should succeed");
        let expired = sys.expire_stale(100);
        assert_eq!(expired, 1);
        assert_eq!(
            sys.get(&tok1)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Expired
        );
        assert_eq!(
            sys.get(&tok2)
                .expect("collab test operation should succeed")
                .status,
            InviteStatus::Sent
        );
    }

    #[test]
    fn test_pending_count() {
        let mut sys = InviteSystem::new();
        let tok = sys.issue("s", "u", InviteRole::Viewer, 0, None);
        assert_eq!(sys.pending_count(), 1); // Pending is actionable
        sys.send(&tok)
            .expect("collab test operation should succeed");
        assert_eq!(sys.pending_count(), 1); // Sent is also actionable
        sys.accept(&tok, 0)
            .expect("collab test operation should succeed");
        assert_eq!(sys.pending_count(), 0);
    }

    #[test]
    fn test_not_found_error() {
        let mut sys = InviteSystem::new();
        let tok = InviteToken::new("nonexistent");
        assert_eq!(sys.send(&tok).unwrap_err(), InviteError::NotFound);
        assert_eq!(sys.accept(&tok, 0).unwrap_err(), InviteError::NotFound);
        assert_eq!(sys.decline(&tok, 0).unwrap_err(), InviteError::NotFound);
        assert_eq!(sys.revoke(&tok).unwrap_err(), InviteError::NotFound);
    }
}
