//! Role-based access control for media resources.
//!
//! Provides roles, permissions, and role inheritance for controlling
//! access to media production resources and operations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::{HashMap, HashSet};

/// A permission that can be granted to a role.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Permission {
    /// Read media files
    ReadMedia,
    /// Write (upload/modify) media files
    WriteMedia,
    /// Delete media files
    DeleteMedia,
    /// Transcode media
    Transcode,
    /// Export media
    Export,
    /// Manage users
    ManageUsers,
    /// Manage roles
    ManageRoles,
    /// View audit logs
    ViewAuditLogs,
    /// Administer system
    AdministerSystem,
    /// Custom permission by name
    Custom(String),
}

impl Permission {
    /// Returns the string name of the permission.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::ReadMedia => "read_media".to_string(),
            Self::WriteMedia => "write_media".to_string(),
            Self::DeleteMedia => "delete_media".to_string(),
            Self::Transcode => "transcode".to_string(),
            Self::Export => "export".to_string(),
            Self::ManageUsers => "manage_users".to_string(),
            Self::ManageRoles => "manage_roles".to_string(),
            Self::ViewAuditLogs => "view_audit_logs".to_string(),
            Self::AdministerSystem => "administer_system".to_string(),
            Self::Custom(name) => name.clone(),
        }
    }
}

/// A role that groups permissions together.
#[derive(Debug, Clone)]
pub struct Role {
    /// Unique role identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Permissions granted by this role
    pub permissions: HashSet<Permission>,
    /// Roles this role inherits from
    pub parent_roles: Vec<String>,
}

impl Role {
    /// Create a new role with no permissions.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            permissions: HashSet::new(),
            parent_roles: Vec::new(),
        }
    }

    /// Add a permission to this role.
    pub fn grant(&mut self, perm: Permission) {
        self.permissions.insert(perm);
    }

    /// Remove a permission from this role.
    pub fn revoke(&mut self, perm: &Permission) {
        self.permissions.remove(perm);
    }

    /// Check if this role directly has a permission (not inherited).
    #[must_use]
    pub fn has_direct_permission(&self, perm: &Permission) -> bool {
        self.permissions.contains(perm)
    }

    /// Add a parent role to inherit permissions from.
    pub fn add_parent(&mut self, parent_id: impl Into<String>) {
        self.parent_roles.push(parent_id.into());
    }
}

/// The RBAC registry that manages roles and user assignments.
#[derive(Debug, Default)]
pub struct RbacRegistry {
    roles: HashMap<String, Role>,
    /// Maps `user_id` -> set of `role_ids`
    user_roles: HashMap<String, HashSet<String>>,
}

impl RbacRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a role.
    pub fn register_role(&mut self, role: Role) {
        self.roles.insert(role.id.clone(), role);
    }

    /// Assign a role to a user.
    pub fn assign_role(&mut self, user_id: impl Into<String>, role_id: impl Into<String>) {
        self.user_roles
            .entry(user_id.into())
            .or_default()
            .insert(role_id.into());
    }

    /// Remove a role from a user.
    pub fn remove_role(&mut self, user_id: &str, role_id: &str) {
        if let Some(roles) = self.user_roles.get_mut(user_id) {
            roles.remove(role_id);
        }
    }

    /// Get all role ids assigned to a user.
    #[must_use]
    pub fn user_role_ids(&self, user_id: &str) -> HashSet<String> {
        self.user_roles.get(user_id).cloned().unwrap_or_default()
    }

    /// Collect all permissions for a role, including inherited ones.
    fn collect_permissions(
        &self,
        role_id: &str,
        visited: &mut HashSet<String>,
    ) -> HashSet<Permission> {
        if visited.contains(role_id) {
            return HashSet::new();
        }
        visited.insert(role_id.to_string());

        let Some(role) = self.roles.get(role_id) else {
            return HashSet::new();
        };

        let mut perms = role.permissions.clone();
        for parent_id in &role.parent_roles.clone() {
            let parent_perms = self.collect_permissions(parent_id, visited);
            perms.extend(parent_perms);
        }
        perms
    }

    /// Check if a user has a specific permission (direct or inherited).
    #[must_use]
    pub fn check_permission(&self, user_id: &str, perm: &Permission) -> bool {
        let role_ids = self.user_role_ids(user_id);
        for role_id in &role_ids {
            let mut visited = HashSet::new();
            let perms = self.collect_permissions(role_id, &mut visited);
            if perms.contains(perm) {
                return true;
            }
        }
        false
    }

    /// Get all effective permissions for a user.
    #[must_use]
    pub fn effective_permissions(&self, user_id: &str) -> HashSet<Permission> {
        let role_ids = self.user_role_ids(user_id);
        let mut all_perms = HashSet::new();
        for role_id in &role_ids {
            let mut visited = HashSet::new();
            all_perms.extend(self.collect_permissions(role_id, &mut visited));
        }
        all_perms
    }

    /// Count registered roles.
    #[must_use]
    pub fn role_count(&self) -> usize {
        self.roles.len()
    }

    /// Count users with any role assigned.
    #[must_use]
    pub fn user_count(&self) -> usize {
        self.user_roles.len()
    }

    /// Get a role by id.
    #[must_use]
    pub fn get_role(&self, role_id: &str) -> Option<&Role> {
        self.roles.get(role_id)
    }
}

/// Access decision result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessDecision {
    /// Access is granted
    Granted,
    /// Access is denied
    Denied(String),
}

impl AccessDecision {
    /// Returns true if access was granted.
    #[must_use]
    pub fn is_granted(&self) -> bool {
        matches!(self, Self::Granted)
    }
}

/// Evaluates access requests against the RBAC registry.
#[derive(Debug)]
pub struct AccessEvaluator<'a> {
    registry: &'a RbacRegistry,
}

impl<'a> AccessEvaluator<'a> {
    /// Create a new evaluator backed by the given registry.
    #[must_use]
    pub fn new(registry: &'a RbacRegistry) -> Self {
        Self { registry }
    }

    /// Evaluate whether a user may perform an action requiring a permission.
    #[must_use]
    pub fn evaluate(&self, user_id: &str, perm: &Permission) -> AccessDecision {
        if self.registry.check_permission(user_id, perm) {
            AccessDecision::Granted
        } else {
            AccessDecision::Denied(format!(
                "User '{}' lacks permission '{}'",
                user_id,
                perm.name()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_editor_role() -> Role {
        let mut role = Role::new("editor", "Editor");
        role.grant(Permission::ReadMedia);
        role.grant(Permission::WriteMedia);
        role.grant(Permission::Transcode);
        role
    }

    fn make_viewer_role() -> Role {
        let mut role = Role::new("viewer", "Viewer");
        role.grant(Permission::ReadMedia);
        role
    }

    #[test]
    fn test_role_new_has_no_permissions() {
        let role = Role::new("test", "Test Role");
        assert!(role.permissions.is_empty());
        assert!(role.parent_roles.is_empty());
    }

    #[test]
    fn test_role_grant_permission() {
        let mut role = Role::new("editor", "Editor");
        role.grant(Permission::ReadMedia);
        assert!(role.has_direct_permission(&Permission::ReadMedia));
    }

    #[test]
    fn test_role_revoke_permission() {
        let mut role = Role::new("editor", "Editor");
        role.grant(Permission::WriteMedia);
        role.revoke(&Permission::WriteMedia);
        assert!(!role.has_direct_permission(&Permission::WriteMedia));
    }

    #[test]
    fn test_role_parent_inheritance() {
        let mut child = Role::new("senior_editor", "Senior Editor");
        child.add_parent("editor");
        assert_eq!(child.parent_roles, vec!["editor"]);
    }

    #[test]
    fn test_registry_register_and_count() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());
        registry.register_role(make_editor_role());
        assert_eq!(registry.role_count(), 2);
    }

    #[test]
    fn test_assign_and_remove_role() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());
        registry.assign_role("alice", "viewer");
        assert!(registry.user_role_ids("alice").contains("viewer"));
        registry.remove_role("alice", "viewer");
        assert!(!registry.user_role_ids("alice").contains("viewer"));
    }

    #[test]
    fn test_check_permission_direct() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_editor_role());
        registry.assign_role("bob", "editor");
        assert!(registry.check_permission("bob", &Permission::WriteMedia));
        assert!(!registry.check_permission("bob", &Permission::DeleteMedia));
    }

    #[test]
    fn test_check_permission_inherited() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());

        let mut senior = Role::new("senior_viewer", "Senior Viewer");
        senior.grant(Permission::Export);
        senior.add_parent("viewer");
        registry.register_role(senior);

        registry.assign_role("carol", "senior_viewer");
        // Inherited from viewer
        assert!(registry.check_permission("carol", &Permission::ReadMedia));
        // Direct on senior_viewer
        assert!(registry.check_permission("carol", &Permission::Export));
    }

    #[test]
    fn test_effective_permissions_multi_role() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());
        registry.register_role(make_editor_role());
        registry.assign_role("dave", "viewer");
        registry.assign_role("dave", "editor");
        let perms = registry.effective_permissions("dave");
        assert!(perms.contains(&Permission::ReadMedia));
        assert!(perms.contains(&Permission::WriteMedia));
        assert!(perms.contains(&Permission::Transcode));
    }

    #[test]
    fn test_user_with_no_roles_denied() {
        let registry = RbacRegistry::new();
        assert!(!registry.check_permission("nobody", &Permission::ReadMedia));
    }

    #[test]
    fn test_access_evaluator_granted() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_editor_role());
        registry.assign_role("eve", "editor");
        let eval = AccessEvaluator::new(&registry);
        let decision = eval.evaluate("eve", &Permission::Transcode);
        assert!(decision.is_granted());
    }

    #[test]
    fn test_access_evaluator_denied() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());
        registry.assign_role("frank", "viewer");
        let eval = AccessEvaluator::new(&registry);
        let decision = eval.evaluate("frank", &Permission::DeleteMedia);
        assert!(!decision.is_granted());
        assert!(matches!(decision, AccessDecision::Denied(_)));
    }

    #[test]
    fn test_custom_permission() {
        let mut role = Role::new("custom_role", "Custom");
        role.grant(Permission::Custom("stream_live".to_string()));
        assert!(role.has_direct_permission(&Permission::Custom("stream_live".to_string())));
        assert!(!role.has_direct_permission(&Permission::Custom("other".to_string())));
    }

    #[test]
    fn test_cycle_in_inheritance_does_not_panic() {
        let mut registry = RbacRegistry::new();
        let mut role_a = Role::new("a", "A");
        role_a.grant(Permission::ReadMedia);
        role_a.add_parent("b");
        let mut role_b = Role::new("b", "B");
        role_b.grant(Permission::WriteMedia);
        role_b.add_parent("a"); // cycle
        registry.register_role(role_a);
        registry.register_role(role_b);
        registry.assign_role("user1", "a");
        // Should not hang or panic
        let perms = registry.effective_permissions("user1");
        assert!(perms.contains(&Permission::ReadMedia));
    }

    #[test]
    fn test_permission_name() {
        assert_eq!(Permission::ReadMedia.name(), "read_media");
        assert_eq!(Permission::AdministerSystem.name(), "administer_system");
        assert_eq!(Permission::Custom("foo".to_string()).name(), "foo");
    }

    #[test]
    fn test_user_count() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_viewer_role());
        registry.assign_role("u1", "viewer");
        registry.assign_role("u2", "viewer");
        assert_eq!(registry.user_count(), 2);
    }

    #[test]
    fn test_get_role() {
        let mut registry = RbacRegistry::new();
        registry.register_role(make_editor_role());
        let role = registry.get_role("editor");
        assert!(role.is_some());
        assert_eq!(role.expect("test expectation failed").name, "Editor");
        assert!(registry.get_role("nonexistent").is_none());
    }
}
