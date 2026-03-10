#![allow(dead_code)]
//! Workspace membership and role management for collaborative projects.
//!
//! A `Workspace` holds a set of `WorkspaceMember`s each carrying a `WorkspaceRole`.
//! Roles gate management operations so only privileged members can invite or remove others.

use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// The role a member holds within a workspace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceRole {
    /// Full administrative control over the workspace.
    Admin,
    /// Can create and edit content but cannot manage membership.
    Editor,
    /// Read-only access; cannot modify anything.
    Viewer,
    /// Guest with temporary, scoped access.
    Guest,
}

impl WorkspaceRole {
    /// Returns `true` if this role may manage (add/remove) workspace members.
    pub fn can_manage(&self) -> bool {
        matches!(self, WorkspaceRole::Admin)
    }

    /// Returns `true` if this role may create or modify content.
    pub fn can_edit(&self) -> bool {
        matches!(self, WorkspaceRole::Admin | WorkspaceRole::Editor)
    }

    /// Human-readable role label.
    pub fn label(&self) -> &'static str {
        match self {
            WorkspaceRole::Admin => "Admin",
            WorkspaceRole::Editor => "Editor",
            WorkspaceRole::Viewer => "Viewer",
            WorkspaceRole::Guest => "Guest",
        }
    }
}

/// A single member of a workspace.
#[derive(Debug, Clone)]
pub struct WorkspaceMember {
    /// Unique user identifier.
    pub user_id: String,
    /// Display name for the member.
    pub display_name: String,
    /// Role in the workspace.
    pub role: WorkspaceRole,
    /// When this user joined the workspace.
    pub joined_at: DateTime<Utc>,
    /// Whether the membership is currently active.
    pub active: bool,
}

impl WorkspaceMember {
    /// Create a new, active workspace member.
    pub fn new(
        user_id: impl Into<String>,
        display_name: impl Into<String>,
        role: WorkspaceRole,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            role,
            joined_at: Utc::now(),
            active: true,
        }
    }

    /// Returns `true` when the member has not been deactivated.
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// A workspace containing multiple members with enforced role-based membership operations.
#[derive(Debug)]
pub struct Workspace {
    /// Workspace identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Members keyed by user_id.
    members: HashMap<String, WorkspaceMember>,
}

impl Workspace {
    /// Create a new workspace. The first member is the initial owner/admin.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            members: HashMap::new(),
        }
    }

    /// Add a member to the workspace.
    ///
    /// `requester_id` must belong to an active Admin member; otherwise returns `Err`.
    /// If `requester_id` is `None` the call is treated as a bootstrapping/internal operation
    /// and always succeeds (used to seed the first admin).
    pub fn add_member(
        &mut self,
        requester_id: Option<&str>,
        new_member: WorkspaceMember,
    ) -> Result<(), String> {
        if let Some(rid) = requester_id {
            let requester = self
                .members
                .get(rid)
                .ok_or_else(|| format!("Requester '{}' not found", rid))?;
            if !requester.is_active() || !requester.role.can_manage() {
                return Err(format!(
                    "Requester '{}' does not have permission to add members",
                    rid
                ));
            }
        }
        self.members.insert(new_member.user_id.clone(), new_member);
        Ok(())
    }

    /// Remove (deactivate) a member from the workspace.
    ///
    /// `requester_id` must be an active Admin. Returns `Err` on permission failure.
    pub fn remove_member(
        &mut self,
        requester_id: &str,
        target_user_id: &str,
    ) -> Result<(), String> {
        {
            let requester = self
                .members
                .get(requester_id)
                .ok_or_else(|| format!("Requester '{}' not found", requester_id))?;
            if !requester.is_active() || !requester.role.can_manage() {
                return Err(format!(
                    "Requester '{}' does not have permission to remove members",
                    requester_id
                ));
            }
        }
        match self.members.get_mut(target_user_id) {
            Some(m) => {
                m.active = false;
                Ok(())
            }
            None => Err(format!("Target user '{}' not found", target_user_id)),
        }
    }

    /// Count of members that are currently active.
    pub fn active_member_count(&self) -> usize {
        self.members.values().filter(|m| m.active).count()
    }

    /// Retrieve a member by user_id.
    pub fn get_member(&self, user_id: &str) -> Option<&WorkspaceMember> {
        self.members.get(user_id)
    }

    /// All active members.
    pub fn active_members(&self) -> Vec<&WorkspaceMember> {
        self.members.values().filter(|m| m.active).collect()
    }

    /// Total member count (active + inactive).
    pub fn total_member_count(&self) -> usize {
        self.members.len()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn admin(id: &str) -> WorkspaceMember {
        WorkspaceMember::new(id, "Admin User", WorkspaceRole::Admin)
    }

    fn editor(id: &str) -> WorkspaceMember {
        WorkspaceMember::new(id, "Editor User", WorkspaceRole::Editor)
    }

    fn viewer(id: &str) -> WorkspaceMember {
        WorkspaceMember::new(id, "Viewer User", WorkspaceRole::Viewer)
    }

    // WorkspaceRole tests

    #[test]
    fn test_admin_can_manage() {
        assert!(WorkspaceRole::Admin.can_manage());
    }

    #[test]
    fn test_editor_cannot_manage() {
        assert!(!WorkspaceRole::Editor.can_manage());
    }

    #[test]
    fn test_viewer_cannot_manage() {
        assert!(!WorkspaceRole::Viewer.can_manage());
    }

    #[test]
    fn test_guest_cannot_manage() {
        assert!(!WorkspaceRole::Guest.can_manage());
    }

    #[test]
    fn test_admin_and_editor_can_edit() {
        assert!(WorkspaceRole::Admin.can_edit());
        assert!(WorkspaceRole::Editor.can_edit());
    }

    #[test]
    fn test_viewer_cannot_edit() {
        assert!(!WorkspaceRole::Viewer.can_edit());
    }

    #[test]
    fn test_role_labels_non_empty() {
        for role in &[
            WorkspaceRole::Admin,
            WorkspaceRole::Editor,
            WorkspaceRole::Viewer,
            WorkspaceRole::Guest,
        ] {
            assert!(!role.label().is_empty());
        }
    }

    // WorkspaceMember tests

    #[test]
    fn test_new_member_is_active() {
        let m = editor("e1");
        assert!(m.is_active());
    }

    #[test]
    fn test_deactivated_member() {
        let mut m = viewer("v1");
        m.active = false;
        assert!(!m.is_active());
    }

    // Workspace tests

    #[test]
    fn test_add_member_bootstrap() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        assert_eq!(ws.active_member_count(), 1);
    }

    #[test]
    fn test_add_member_by_admin() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), editor("e1"))
            .expect("collab test operation should succeed");
        assert_eq!(ws.active_member_count(), 2);
    }

    #[test]
    fn test_add_member_by_non_admin_fails() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), editor("e1"))
            .expect("collab test operation should succeed");
        let result = ws.add_member(Some("e1"), viewer("v1"));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_member_by_admin() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), editor("e1"))
            .expect("collab test operation should succeed");
        ws.remove_member("a1", "e1")
            .expect("collab test operation should succeed");
        assert_eq!(ws.active_member_count(), 1);
    }

    #[test]
    fn test_remove_member_by_editor_fails() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), editor("e1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), viewer("v1"))
            .expect("collab test operation should succeed");
        let result = ws.remove_member("e1", "v1");
        assert!(result.is_err());
    }

    #[test]
    fn test_active_member_count_excludes_removed() {
        let mut ws = Workspace::new("ws1", "My Workspace");
        ws.add_member(None, admin("a1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), editor("e1"))
            .expect("collab test operation should succeed");
        ws.add_member(Some("a1"), viewer("v1"))
            .expect("collab test operation should succeed");
        ws.remove_member("a1", "v1")
            .expect("collab test operation should succeed");
        assert_eq!(ws.active_member_count(), 2);
        assert_eq!(ws.total_member_count(), 3);
    }
}
