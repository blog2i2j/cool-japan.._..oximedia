#![allow(dead_code)]
//! Team role management for collaborative editing sessions.
//!
//! Provides fine-grained role definitions, capability matrices, role
//! hierarchies, and role assignment/revocation workflows that go beyond
//! the simple Owner/Editor/Viewer model.

use std::collections::{HashMap, HashSet};
use std::fmt;

/// Capability that can be granted to a role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    /// View project content.
    View,
    /// Edit timeline content.
    EditTimeline,
    /// Edit audio tracks.
    EditAudio,
    /// Apply color grading.
    ColorGrade,
    /// Add and manage effects.
    ManageEffects,
    /// Export the final output.
    Export,
    /// Manage project metadata.
    ManageMetadata,
    /// Approve or reject changes.
    Approve,
    /// Invite other users.
    Invite,
    /// Assign roles to other users.
    AssignRoles,
    /// Delete project assets.
    Delete,
    /// Administer the project (full control).
    Admin,
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Capability::View => "view",
            Capability::EditTimeline => "edit_timeline",
            Capability::EditAudio => "edit_audio",
            Capability::ColorGrade => "color_grade",
            Capability::ManageEffects => "manage_effects",
            Capability::Export => "export",
            Capability::ManageMetadata => "manage_metadata",
            Capability::Approve => "approve",
            Capability::Invite => "invite",
            Capability::AssignRoles => "assign_roles",
            Capability::Delete => "delete",
            Capability::Admin => "admin",
        };
        write!(f, "{s}")
    }
}

/// A named role with a set of capabilities.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamRole {
    /// Unique name of the role.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Capabilities granted by this role.
    pub capabilities: HashSet<Capability>,
    /// Hierarchy level (lower = more privileged; 0 = top).
    pub level: u8,
    /// Whether this is a built-in role that cannot be deleted.
    pub built_in: bool,
}

impl TeamRole {
    /// Create a new custom role.
    pub fn new(name: impl Into<String>, description: impl Into<String>, level: u8) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            capabilities: HashSet::new(),
            level,
            built_in: false,
        }
    }

    /// Add a capability to the role.
    pub fn with_capability(mut self, cap: Capability) -> Self {
        self.capabilities.insert(cap);
        self
    }

    /// Add multiple capabilities.
    pub fn with_capabilities(mut self, caps: &[Capability]) -> Self {
        for cap in caps {
            self.capabilities.insert(*cap);
        }
        self
    }

    /// Mark this role as built-in.
    pub fn as_built_in(mut self) -> Self {
        self.built_in = true;
        self
    }

    /// Check if this role has a specific capability.
    pub fn has_capability(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap) || self.capabilities.contains(&Capability::Admin)
    }

    /// Check if this role outranks (is more privileged than) another.
    pub fn outranks(&self, other: &TeamRole) -> bool {
        self.level < other.level
    }

    /// Return the number of capabilities in this role.
    pub fn capability_count(&self) -> usize {
        self.capabilities.len()
    }
}

impl fmt::Display for TeamRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (level {})", self.name, self.level)
    }
}

/// Predefined roles.
pub fn admin_role() -> TeamRole {
    TeamRole::new("admin", "Full project administrator", 0)
        .with_capability(Capability::Admin)
        .as_built_in()
}

/// Predefined editor role.
pub fn editor_role() -> TeamRole {
    TeamRole::new("editor", "Can edit timeline and audio", 10)
        .with_capabilities(&[
            Capability::View,
            Capability::EditTimeline,
            Capability::EditAudio,
            Capability::ManageEffects,
            Capability::Export,
            Capability::ManageMetadata,
        ])
        .as_built_in()
}

/// Predefined colorist role.
pub fn colorist_role() -> TeamRole {
    TeamRole::new("colorist", "Color grading specialist", 15)
        .with_capabilities(&[Capability::View, Capability::ColorGrade, Capability::Export])
        .as_built_in()
}

/// Predefined reviewer role.
pub fn reviewer_role() -> TeamRole {
    TeamRole::new("reviewer", "Can view and approve", 20)
        .with_capabilities(&[Capability::View, Capability::Approve])
        .as_built_in()
}

/// Predefined viewer role.
pub fn viewer_role() -> TeamRole {
    TeamRole::new("viewer", "Read-only access", 30)
        .with_capability(Capability::View)
        .as_built_in()
}

/// Assignment of a role to a user within a project.
#[derive(Debug, Clone)]
pub struct RoleAssignment {
    /// The user ID this assignment belongs to.
    pub user_id: String,
    /// The role name assigned.
    pub role_name: String,
    /// Who assigned this role.
    pub assigned_by: String,
    /// Timestamp of assignment (epoch seconds).
    pub assigned_at: u64,
    /// Optional expiry timestamp.
    pub expires_at: Option<u64>,
}

impl RoleAssignment {
    /// Create a new role assignment.
    pub fn new(
        user_id: impl Into<String>,
        role_name: impl Into<String>,
        assigned_by: impl Into<String>,
        assigned_at: u64,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            role_name: role_name.into(),
            assigned_by: assigned_by.into(),
            assigned_at,
            expires_at: None,
        }
    }

    /// Set an expiry time.
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Check if the assignment has expired.
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires_at.map_or(false, |exp| now >= exp)
    }
}

/// Error type for role management operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RoleError {
    /// Role not found.
    NotFound(String),
    /// Cannot delete a built-in role.
    BuiltInRole(String),
    /// Duplicate role name.
    Duplicate(String),
    /// Insufficient privilege to perform the operation.
    InsufficientPrivilege,
    /// User already has this role.
    AlreadyAssigned(String),
}

impl fmt::Display for RoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoleError::NotFound(n) => write!(f, "Role not found: {n}"),
            RoleError::BuiltInRole(n) => write!(f, "Cannot modify built-in role: {n}"),
            RoleError::Duplicate(n) => write!(f, "Duplicate role: {n}"),
            RoleError::InsufficientPrivilege => write!(f, "Insufficient privilege"),
            RoleError::AlreadyAssigned(u) => write!(f, "User already has role: {u}"),
        }
    }
}

/// Manager for roles and role assignments.
#[derive(Debug)]
pub struct RoleManager {
    /// Defined roles keyed by name.
    roles: HashMap<String, TeamRole>,
    /// Role assignments: user_id -> list of assignments.
    assignments: HashMap<String, Vec<RoleAssignment>>,
}

impl RoleManager {
    /// Create a new role manager with predefined built-in roles.
    pub fn new() -> Self {
        let mut roles = HashMap::new();
        for role in [
            admin_role(),
            editor_role(),
            colorist_role(),
            reviewer_role(),
            viewer_role(),
        ] {
            roles.insert(role.name.clone(), role);
        }
        Self {
            roles,
            assignments: HashMap::new(),
        }
    }

    /// Add a custom role.
    pub fn add_role(&mut self, role: TeamRole) -> Result<(), RoleError> {
        if self.roles.contains_key(&role.name) {
            return Err(RoleError::Duplicate(role.name));
        }
        self.roles.insert(role.name.clone(), role);
        Ok(())
    }

    /// Remove a custom role (built-in roles cannot be removed).
    pub fn remove_role(&mut self, name: &str) -> Result<TeamRole, RoleError> {
        let role = self
            .roles
            .get(name)
            .ok_or_else(|| RoleError::NotFound(name.to_string()))?;
        if role.built_in {
            return Err(RoleError::BuiltInRole(name.to_string()));
        }
        // The key's existence was verified by get() above; remove() is always
        // Some here.  ok_or_else provides a safe fallback without panicking.
        self.roles
            .remove(name)
            .ok_or_else(|| RoleError::NotFound(name.to_string()))
    }

    /// Get a role by name.
    pub fn get_role(&self, name: &str) -> Option<&TeamRole> {
        self.roles.get(name)
    }

    /// List all role names.
    pub fn list_roles(&self) -> Vec<&str> {
        self.roles.keys().map(|k| k.as_str()).collect()
    }

    /// Assign a role to a user.
    pub fn assign_role(&mut self, assignment: RoleAssignment) -> Result<(), RoleError> {
        if !self.roles.contains_key(&assignment.role_name) {
            return Err(RoleError::NotFound(assignment.role_name.clone()));
        }
        let entries = self
            .assignments
            .entry(assignment.user_id.clone())
            .or_default();
        if entries.iter().any(|a| a.role_name == assignment.role_name) {
            return Err(RoleError::AlreadyAssigned(assignment.user_id));
        }
        entries.push(assignment);
        Ok(())
    }

    /// Revoke a role from a user.
    pub fn revoke_role(&mut self, user_id: &str, role_name: &str) -> Result<(), RoleError> {
        let entries = self
            .assignments
            .get_mut(user_id)
            .ok_or_else(|| RoleError::NotFound(user_id.to_string()))?;
        let idx = entries
            .iter()
            .position(|a| a.role_name == role_name)
            .ok_or_else(|| RoleError::NotFound(role_name.to_string()))?;
        entries.remove(idx);
        Ok(())
    }

    /// Get all roles assigned to a user.
    pub fn user_roles(&self, user_id: &str) -> Vec<&TeamRole> {
        self.assignments
            .get(user_id)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|a| self.roles.get(&a.role_name))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if a user has a specific capability (through any of their roles).
    pub fn user_has_capability(&self, user_id: &str, cap: Capability) -> bool {
        self.user_roles(user_id)
            .iter()
            .any(|role| role.has_capability(cap))
    }

    /// Return total number of assignments.
    pub fn total_assignments(&self) -> usize {
        self.assignments.values().map(|v| v.len()).sum()
    }
}

impl Default for RoleManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_role_has_all_caps() {
        let admin = admin_role();
        assert!(admin.has_capability(Capability::View));
        assert!(admin.has_capability(Capability::Delete));
        assert!(admin.has_capability(Capability::Admin));
    }

    #[test]
    fn test_editor_role_capabilities() {
        let editor = editor_role();
        assert!(editor.has_capability(Capability::EditTimeline));
        assert!(editor.has_capability(Capability::EditAudio));
        assert!(!editor.has_capability(Capability::Admin));
        assert!(!editor.has_capability(Capability::Delete));
    }

    #[test]
    fn test_viewer_role_limited() {
        let viewer = viewer_role();
        assert!(viewer.has_capability(Capability::View));
        assert!(!viewer.has_capability(Capability::EditTimeline));
    }

    #[test]
    fn test_role_outranks() {
        let admin = admin_role();
        let editor = editor_role();
        let viewer = viewer_role();
        assert!(admin.outranks(&editor));
        assert!(editor.outranks(&viewer));
        assert!(!viewer.outranks(&admin));
    }

    #[test]
    fn test_custom_role() {
        let role = TeamRole::new("sound_designer", "Sound design specialist", 12)
            .with_capabilities(&[Capability::View, Capability::EditAudio]);
        assert_eq!(role.name, "sound_designer");
        assert_eq!(role.capability_count(), 2);
        assert!(!role.built_in);
    }

    #[test]
    fn test_role_display() {
        let role = editor_role();
        assert_eq!(role.to_string(), "editor (level 10)");
    }

    #[test]
    fn test_capability_display() {
        assert_eq!(Capability::EditTimeline.to_string(), "edit_timeline");
        assert_eq!(Capability::Admin.to_string(), "admin");
    }

    #[test]
    fn test_role_manager_builtin_roles() {
        let mgr = RoleManager::new();
        assert!(mgr.get_role("admin").is_some());
        assert!(mgr.get_role("editor").is_some());
        assert!(mgr.get_role("viewer").is_some());
        assert!(mgr.get_role("colorist").is_some());
        assert!(mgr.get_role("reviewer").is_some());
    }

    #[test]
    fn test_role_manager_add_custom() {
        let mut mgr = RoleManager::new();
        let role = TeamRole::new("intern", "Intern role", 25).with_capability(Capability::View);
        mgr.add_role(role)
            .expect("collab test operation should succeed");
        assert!(mgr.get_role("intern").is_some());
    }

    #[test]
    fn test_role_manager_duplicate() {
        let mut mgr = RoleManager::new();
        let role = TeamRole::new("admin", "dup", 0);
        assert!(matches!(mgr.add_role(role), Err(RoleError::Duplicate(_))));
    }

    #[test]
    fn test_role_manager_cannot_remove_builtin() {
        let mut mgr = RoleManager::new();
        assert!(matches!(
            mgr.remove_role("admin"),
            Err(RoleError::BuiltInRole(_))
        ));
    }

    #[test]
    fn test_assign_and_check_capability() {
        let mut mgr = RoleManager::new();
        let assignment = RoleAssignment::new("user1", "editor", "admin", 1000);
        mgr.assign_role(assignment)
            .expect("collab test operation should succeed");
        assert!(mgr.user_has_capability("user1", Capability::EditTimeline));
        assert!(!mgr.user_has_capability("user1", Capability::Admin));
    }

    #[test]
    fn test_revoke_role() {
        let mut mgr = RoleManager::new();
        let assignment = RoleAssignment::new("user1", "editor", "admin", 1000);
        mgr.assign_role(assignment)
            .expect("collab test operation should succeed");
        mgr.revoke_role("user1", "editor")
            .expect("collab test operation should succeed");
        assert!(!mgr.user_has_capability("user1", Capability::EditTimeline));
    }

    #[test]
    fn test_assignment_expiry() {
        let a = RoleAssignment::new("u", "r", "a", 100).with_expiry(200);
        assert!(!a.is_expired(150));
        assert!(a.is_expired(200));
        assert!(a.is_expired(300));
    }

    #[test]
    fn test_total_assignments() {
        let mut mgr = RoleManager::new();
        mgr.assign_role(RoleAssignment::new("u1", "editor", "admin", 1000))
            .expect("collab test operation should succeed");
        mgr.assign_role(RoleAssignment::new("u2", "viewer", "admin", 1000))
            .expect("collab test operation should succeed");
        assert_eq!(mgr.total_assignments(), 2);
    }
}
