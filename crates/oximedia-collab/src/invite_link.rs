//! Invite link management for collaboration sessions.
//!
//! Provides short-lived, role-specific invite links that grant access to
//! collaboration sessions without requiring direct user lookup.

#![allow(dead_code)]

use std::collections::HashMap;

/// The role granted to a user who joins via an invite link.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InviteRole {
    /// Full edit permissions.
    Editor,
    /// Read-only access.
    Viewer,
    /// Comment-only access.
    Commenter,
}

impl InviteRole {
    /// Returns `true` if this role grants edit (write) capabilities.
    #[must_use]
    pub const fn can_edit(self) -> bool {
        matches!(self, Self::Editor)
    }

    /// Returns a short string identifier for the role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Editor => "editor",
            Self::Viewer => "viewer",
            Self::Commenter => "commenter",
        }
    }
}

/// A single invite link.
#[derive(Debug, Clone)]
pub struct InviteLink {
    /// Unique token used in the invite URL.
    pub token: String,
    /// The session (or project) this link grants access to.
    pub session_id: String,
    /// The role granted to the invitee.
    pub role: InviteRole,
    /// Maximum number of times this link may be used (0 = unlimited).
    pub max_uses: u32,
    /// Number of times the link has been used so far.
    pub use_count: u32,
    /// Unix timestamp (seconds) at which this link expires (0 = never).
    pub expires_at: u64,
    /// Whether the link has been explicitly revoked.
    pub revoked: bool,
}

impl InviteLink {
    /// Creates a new invite link.
    #[must_use]
    pub fn new(
        token: String,
        session_id: String,
        role: InviteRole,
        max_uses: u32,
        expires_at: u64,
    ) -> Self {
        Self {
            token,
            session_id,
            role,
            max_uses,
            use_count: 0,
            expires_at,
            revoked: false,
        }
    }

    /// Returns `true` if this link has expired relative to `now_secs` (Unix time).
    ///
    /// A link with `expires_at == 0` never expires.
    #[must_use]
    pub fn is_expired_at(&self, now_secs: u64) -> bool {
        self.expires_at != 0 && now_secs >= self.expires_at
    }

    /// Returns `true` if this link has reached its maximum use count.
    ///
    /// A link with `max_uses == 0` has unlimited uses.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.max_uses != 0 && self.use_count >= self.max_uses
    }

    /// Returns `true` if this link can still be used at `now_secs`.
    #[must_use]
    pub fn is_usable_at(&self, now_secs: u64) -> bool {
        !self.revoked && !self.is_expired_at(now_secs) && !self.is_exhausted()
    }
}

/// Error type for invite-link operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InviteError {
    /// The token was not found.
    NotFound(String),
    /// The link has expired.
    Expired,
    /// The link has been fully used.
    Exhausted,
    /// The link has been revoked.
    Revoked,
}

impl std::fmt::Display for InviteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(t) => write!(f, "invite link not found: {t}"),
            Self::Expired => write!(f, "invite link has expired"),
            Self::Exhausted => write!(f, "invite link use count exhausted"),
            Self::Revoked => write!(f, "invite link has been revoked"),
        }
    }
}

/// Manager that stores and validates invite links.
#[derive(Debug, Default)]
pub struct InviteLinkManager {
    links: HashMap<String, InviteLink>,
}

impl InviteLinkManager {
    /// Creates an empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates and stores a new invite link.  Returns the token.
    pub fn create(
        &mut self,
        session_id: String,
        role: InviteRole,
        max_uses: u32,
        expires_at: u64,
    ) -> String {
        // Simple deterministic token for tests; real code would use UUID/random.
        let token = format!("inv-{}-{}", session_id, self.links.len());
        let link = InviteLink::new(token.clone(), session_id, role, max_uses, expires_at);
        self.links.insert(token.clone(), link);
        token
    }

    /// Attempts to "use" the invite link identified by `token` at `now_secs`.
    ///
    /// On success, increments the use counter and returns the granted role.
    pub fn use_invite(&mut self, token: &str, now_secs: u64) -> Result<InviteRole, InviteError> {
        let link = self
            .links
            .get_mut(token)
            .ok_or_else(|| InviteError::NotFound(token.to_string()))?;

        if link.revoked {
            return Err(InviteError::Revoked);
        }
        if link.is_expired_at(now_secs) {
            return Err(InviteError::Expired);
        }
        if link.is_exhausted() {
            return Err(InviteError::Exhausted);
        }

        link.use_count += 1;
        Ok(link.role)
    }

    /// Revokes the invite link identified by `token`.
    ///
    /// Returns `false` if the token was not found.
    pub fn revoke(&mut self, token: &str) -> bool {
        if let Some(link) = self.links.get_mut(token) {
            link.revoked = true;
            true
        } else {
            false
        }
    }

    /// Returns the number of currently active (non-revoked, non-expired, non-exhausted)
    /// links, evaluated at `now_secs`.
    #[must_use]
    pub fn active_count(&self, now_secs: u64) -> usize {
        self.links
            .values()
            .filter(|l| l.is_usable_at(now_secs))
            .count()
    }

    /// Returns a reference to the link for `token`, if present.
    #[must_use]
    pub fn get(&self, token: &str) -> Option<&InviteLink> {
        self.links.get(token)
    }

    /// Returns the total number of links (including revoked/expired ones).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.links.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invite_role_can_edit() {
        assert!(InviteRole::Editor.can_edit());
        assert!(!InviteRole::Viewer.can_edit());
        assert!(!InviteRole::Commenter.can_edit());
    }

    #[test]
    fn invite_role_as_str() {
        assert_eq!(InviteRole::Editor.as_str(), "editor");
        assert_eq!(InviteRole::Viewer.as_str(), "viewer");
        assert_eq!(InviteRole::Commenter.as_str(), "commenter");
    }

    #[test]
    fn invite_link_not_expired_at_zero() {
        let link = InviteLink::new("tok".into(), "sess".into(), InviteRole::Viewer, 0, 0);
        assert!(!link.is_expired_at(u64::MAX));
    }

    #[test]
    fn invite_link_expired() {
        let link = InviteLink::new("tok".into(), "sess".into(), InviteRole::Viewer, 0, 100);
        assert!(link.is_expired_at(100));
        assert!(link.is_expired_at(200));
        assert!(!link.is_expired_at(99));
    }

    #[test]
    fn invite_link_exhausted() {
        let mut link = InviteLink::new("tok".into(), "sess".into(), InviteRole::Editor, 2, 0);
        link.use_count = 2;
        assert!(link.is_exhausted());
    }

    #[test]
    fn invite_link_unlimited_uses() {
        let link = InviteLink::new("tok".into(), "sess".into(), InviteRole::Editor, 0, 0);
        assert!(!link.is_exhausted());
    }

    #[test]
    fn manager_create_and_get() {
        let mut mgr = InviteLinkManager::new();
        let token = mgr.create("s1".into(), InviteRole::Editor, 5, 0);
        let link = mgr
            .get(&token)
            .expect("collab test operation should succeed");
        assert_eq!(link.role, InviteRole::Editor);
    }

    #[test]
    fn manager_use_invite_increments_count() {
        let mut mgr = InviteLinkManager::new();
        let token = mgr.create("s1".into(), InviteRole::Editor, 5, 0);
        let role = mgr
            .use_invite(&token, 0)
            .expect("collab test operation should succeed");
        assert_eq!(role, InviteRole::Editor);
        assert_eq!(
            mgr.get(&token)
                .expect("collab test operation should succeed")
                .use_count,
            1
        );
    }

    #[test]
    fn manager_use_invite_exhausted() {
        let mut mgr = InviteLinkManager::new();
        let token = mgr.create("s1".into(), InviteRole::Viewer, 1, 0);
        mgr.use_invite(&token, 0)
            .expect("collab test operation should succeed");
        let err = mgr.use_invite(&token, 0).unwrap_err();
        assert_eq!(err, InviteError::Exhausted);
    }

    #[test]
    fn manager_use_invite_expired() {
        let mut mgr = InviteLinkManager::new();
        let token = mgr.create("s1".into(), InviteRole::Viewer, 0, 50);
        let err = mgr.use_invite(&token, 100).unwrap_err();
        assert_eq!(err, InviteError::Expired);
    }

    #[test]
    fn manager_revoke() {
        let mut mgr = InviteLinkManager::new();
        let token = mgr.create("s1".into(), InviteRole::Editor, 0, 0);
        assert!(mgr.revoke(&token));
        let err = mgr.use_invite(&token, 0).unwrap_err();
        assert_eq!(err, InviteError::Revoked);
    }

    #[test]
    fn manager_active_count() {
        let mut mgr = InviteLinkManager::new();
        mgr.create("s1".into(), InviteRole::Editor, 0, 0);
        let expired_tok = mgr.create("s1".into(), InviteRole::Viewer, 0, 10);
        assert_eq!(mgr.active_count(0), 2);
        assert_eq!(mgr.active_count(11), 1); // expired_tok no longer active
        let _ = expired_tok; // suppress unused warning
    }

    #[test]
    fn manager_total_count() {
        let mut mgr = InviteLinkManager::new();
        mgr.create("s1".into(), InviteRole::Editor, 0, 0);
        mgr.create("s2".into(), InviteRole::Viewer, 0, 0);
        assert_eq!(mgr.total_count(), 2);
    }

    #[test]
    fn invite_error_display() {
        let e = InviteError::NotFound("tok".to_string());
        assert!(e.to_string().contains("tok"));
    }
}
