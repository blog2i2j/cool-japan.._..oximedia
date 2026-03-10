//! User group management for access control in `OxiMedia`.
//!
//! Supports hierarchical group definitions, membership management,
//! and permission inheritance across groups.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

/// Identifier for a user group.
pub type GroupId = u32;

/// The type of a user group, determining its permission behaviour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupType {
    /// Top-level administrator group with all permissions.
    Admin,
    /// Standard editors with production access.
    Editor,
    /// Viewers with read-only access.
    Viewer,
    /// A custom group with explicitly defined permissions.
    Custom(String),
}

impl GroupType {
    /// Returns `true` if this group type inherits permissions from a parent group.
    #[must_use]
    pub fn inherits_permissions(&self) -> bool {
        match self {
            Self::Admin => false,
            Self::Editor => true,
            Self::Viewer => true,
            Self::Custom(_) => true,
        }
    }

    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Admin => "Admin",
            Self::Editor => "Editor",
            Self::Viewer => "Viewer",
            Self::Custom(name) => name.as_str(),
        }
    }
}

/// A named group of users.
#[derive(Debug, Clone)]
pub struct UserGroup {
    /// Unique identifier.
    pub id: GroupId,
    /// Human-readable name.
    pub name: String,
    /// Type determining permission behaviour.
    pub group_type: GroupType,
    /// Set of member usernames.
    members: HashSet<String>,
    /// Optional parent group ID for permission inheritance.
    pub parent_id: Option<GroupId>,
}

impl UserGroup {
    /// Create a new empty group.
    #[must_use]
    pub fn new(id: GroupId, name: impl Into<String>, group_type: GroupType) -> Self {
        Self {
            id,
            name: name.into(),
            group_type,
            members: HashSet::new(),
            parent_id: None,
        }
    }

    /// Add a member to this group.
    pub fn add_member(&mut self, username: impl Into<String>) {
        self.members.insert(username.into());
    }

    /// Remove a member from this group.
    pub fn remove_member(&mut self, username: &str) {
        self.members.remove(username);
    }

    /// Returns `true` if `username` is a member.
    #[must_use]
    pub fn is_member(&self, username: &str) -> bool {
        self.members.contains(username)
    }

    /// Returns the number of members in this group.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Returns an iterator over all member usernames.
    pub fn members(&self) -> impl Iterator<Item = &str> {
        self.members.iter().map(String::as_str)
    }
}

/// Manages a collection of user groups.
#[derive(Debug, Default)]
pub struct GroupManager {
    groups: HashMap<GroupId, UserGroup>,
    next_id: GroupId,
}

impl GroupManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            groups: HashMap::new(),
            next_id: 1,
        }
    }

    /// Create a new group and register it, returning its assigned ID.
    pub fn create(&mut self, name: impl Into<String>, group_type: GroupType) -> GroupId {
        let id = self.next_id;
        self.next_id += 1;
        let group = UserGroup::new(id, name, group_type);
        self.groups.insert(id, group);
        id
    }

    /// Add `username` to the group identified by `group_id`.
    /// Returns `false` if the group does not exist.
    pub fn add_member(&mut self, group_id: GroupId, username: impl Into<String>) -> bool {
        if let Some(g) = self.groups.get_mut(&group_id) {
            g.add_member(username);
            true
        } else {
            false
        }
    }

    /// Remove `username` from the group identified by `group_id`.
    pub fn remove_member(&mut self, group_id: GroupId, username: &str) {
        if let Some(g) = self.groups.get_mut(&group_id) {
            g.remove_member(username);
        }
    }

    /// Retrieve an immutable reference to a group.
    #[must_use]
    pub fn get(&self, group_id: GroupId) -> Option<&UserGroup> {
        self.groups.get(&group_id)
    }

    /// Returns a list of group IDs that `username` belongs to.
    #[must_use]
    pub fn groups_for_user(&self, username: &str) -> Vec<GroupId> {
        self.groups
            .values()
            .filter(|g| g.is_member(username))
            .map(|g| g.id)
            .collect()
    }

    /// Returns the total number of registered groups.
    #[must_use]
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_admin_does_not_inherit() {
        assert!(!GroupType::Admin.inherits_permissions());
    }

    #[test]
    fn test_editor_inherits() {
        assert!(GroupType::Editor.inherits_permissions());
    }

    #[test]
    fn test_viewer_inherits() {
        assert!(GroupType::Viewer.inherits_permissions());
    }

    #[test]
    fn test_custom_inherits() {
        assert!(GroupType::Custom("ops".to_string()).inherits_permissions());
    }

    #[test]
    fn test_group_type_label() {
        assert_eq!(GroupType::Admin.label(), "Admin");
        assert_eq!(GroupType::Custom("ops".to_string()).label(), "ops");
    }

    #[test]
    fn test_user_group_is_member_after_add() {
        let mut g = UserGroup::new(1, "editors", GroupType::Editor);
        g.add_member("alice");
        assert!(g.is_member("alice"));
    }

    #[test]
    fn test_user_group_not_member_before_add() {
        let g = UserGroup::new(1, "editors", GroupType::Editor);
        assert!(!g.is_member("bob"));
    }

    #[test]
    fn test_user_group_remove_member() {
        let mut g = UserGroup::new(1, "editors", GroupType::Editor);
        g.add_member("carol");
        g.remove_member("carol");
        assert!(!g.is_member("carol"));
    }

    #[test]
    fn test_user_group_member_count() {
        let mut g = UserGroup::new(1, "editors", GroupType::Editor);
        g.add_member("dave");
        g.add_member("eve");
        assert_eq!(g.member_count(), 2);
    }

    #[test]
    fn test_manager_create_returns_id() {
        let mut mgr = GroupManager::new();
        let id = mgr.create("admins", GroupType::Admin);
        assert_eq!(id, 1);
    }

    #[test]
    fn test_manager_group_count() {
        let mut mgr = GroupManager::new();
        mgr.create("a", GroupType::Editor);
        mgr.create("b", GroupType::Viewer);
        assert_eq!(mgr.group_count(), 2);
    }

    #[test]
    fn test_manager_add_member() {
        let mut mgr = GroupManager::new();
        let id = mgr.create("editors", GroupType::Editor);
        assert!(mgr.add_member(id, "frank"));
        assert!(mgr.get(id).expect("get should succeed").is_member("frank"));
    }

    #[test]
    fn test_manager_add_member_missing_group() {
        let mut mgr = GroupManager::new();
        assert!(!mgr.add_member(999, "ghost"));
    }

    #[test]
    fn test_groups_for_user_single() {
        let mut mgr = GroupManager::new();
        let id = mgr.create("editors", GroupType::Editor);
        mgr.add_member(id, "grace");
        let groups = mgr.groups_for_user("grace");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], id);
    }

    #[test]
    fn test_groups_for_user_multiple() {
        let mut mgr = GroupManager::new();
        let id1 = mgr.create("editors", GroupType::Editor);
        let id2 = mgr.create("viewers", GroupType::Viewer);
        mgr.add_member(id1, "heidi");
        mgr.add_member(id2, "heidi");
        let mut groups = mgr.groups_for_user("heidi");
        groups.sort_unstable();
        assert_eq!(groups, vec![id1, id2]);
    }

    #[test]
    fn test_groups_for_user_none() {
        let mgr = GroupManager::new();
        assert!(mgr.groups_for_user("nobody").is_empty());
    }
}
