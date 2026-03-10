//! Catalog of automation actions available to the broadcast system.
//!
//! Provides `ActionCategory`, `ActionDef`, and `ActionCatalog` for
//! registering, discovering, and filtering automation actions.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ActionCategory
// ---------------------------------------------------------------------------

/// Broad grouping for an automation action.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActionCategory {
    /// Controls video/audio playout devices.
    DeviceControl,
    /// Manipulates the current rundown or playlist.
    PlaylistManagement,
    /// Triggers live switching operations.
    LiveSwitching,
    /// Sends notifications or external messages.
    Notification,
    /// Manages file or asset operations.
    AssetManagement,
    /// System-level operations (start, stop, restart).
    SystemControl,
    /// Custom / user-defined category.
    Custom(String),
}

impl ActionCategory {
    /// Short display label for the category.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::DeviceControl => "Device Control".to_string(),
            Self::PlaylistManagement => "Playlist Management".to_string(),
            Self::LiveSwitching => "Live Switching".to_string(),
            Self::Notification => "Notification".to_string(),
            Self::AssetManagement => "Asset Management".to_string(),
            Self::SystemControl => "System Control".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Returns `true` for built-in (non-custom) categories.
    #[must_use]
    pub fn is_builtin(&self) -> bool {
        !matches!(self, Self::Custom(_))
    }
}

// ---------------------------------------------------------------------------
// ActionDef
// ---------------------------------------------------------------------------

/// Definition of a single automation action.
#[derive(Debug, Clone)]
pub struct ActionDef {
    /// Unique identifier for the action.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Category this action belongs to.
    pub category: ActionCategory,
    /// Short description of what the action does.
    pub description: String,
    /// Whether this action can be undone.
    reversible: bool,
    /// Minimum privilege level required (0 = anyone, higher = more restricted).
    pub min_privilege: u8,
}

impl ActionDef {
    /// Create a new action definition.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        category: ActionCategory,
        description: impl Into<String>,
        reversible: bool,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            category,
            description: description.into(),
            reversible,
            min_privilege: 0,
        }
    }

    /// Set the minimum privilege level.
    #[must_use]
    pub fn with_privilege(mut self, level: u8) -> Self {
        self.min_privilege = level;
        self
    }

    /// Returns `true` if this action supports undo.
    #[must_use]
    pub fn is_reversible(&self) -> bool {
        self.reversible
    }
}

// ---------------------------------------------------------------------------
// ActionCatalog
// ---------------------------------------------------------------------------

/// Registry of all known automation actions.
#[derive(Debug, Default)]
pub struct ActionCatalog {
    actions: HashMap<String, ActionDef>,
}

impl ActionCatalog {
    /// Create an empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an action definition. Overwrites any existing entry with the
    /// same ID.
    pub fn register(&mut self, action: ActionDef) {
        self.actions.insert(action.id.clone(), action);
    }

    /// Look up an action by ID.
    #[must_use]
    pub fn find(&self, id: &str) -> Option<&ActionDef> {
        self.actions.get(id)
    }

    /// Return all actions belonging to the given category.
    #[must_use]
    pub fn by_category(&self, category: &ActionCategory) -> Vec<&ActionDef> {
        self.actions
            .values()
            .filter(|a| &a.category == category)
            .collect()
    }

    /// Return all reversible actions.
    #[must_use]
    pub fn reversible(&self) -> Vec<&ActionDef> {
        self.actions
            .values()
            .filter(|a| a.is_reversible())
            .collect()
    }

    /// Total number of registered actions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns `true` if no actions have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Return all actions sorted by name.
    #[must_use]
    pub fn all_sorted(&self) -> Vec<&ActionDef> {
        let mut v: Vec<&ActionDef> = self.actions.values().collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_def(id: &str, category: ActionCategory, reversible: bool) -> ActionDef {
        ActionDef::new(id, id, category, "desc", reversible)
    }

    #[test]
    fn test_category_label_builtin() {
        assert_eq!(ActionCategory::DeviceControl.label(), "Device Control");
        assert_eq!(ActionCategory::LiveSwitching.label(), "Live Switching");
    }

    #[test]
    fn test_category_label_custom() {
        let c = ActionCategory::Custom("My Category".to_string());
        assert_eq!(c.label(), "My Category");
    }

    #[test]
    fn test_category_is_builtin() {
        assert!(ActionCategory::DeviceControl.is_builtin());
        assert!(!ActionCategory::Custom("x".to_string()).is_builtin());
    }

    #[test]
    fn test_action_def_is_reversible_true() {
        let a = make_def("a", ActionCategory::DeviceControl, true);
        assert!(a.is_reversible());
    }

    #[test]
    fn test_action_def_is_reversible_false() {
        let a = make_def("b", ActionCategory::SystemControl, false);
        assert!(!a.is_reversible());
    }

    #[test]
    fn test_action_def_with_privilege() {
        let a = make_def("c", ActionCategory::SystemControl, false).with_privilege(3);
        assert_eq!(a.min_privilege, 3);
    }

    #[test]
    fn test_catalog_register_and_find() {
        let mut cat = ActionCatalog::new();
        cat.register(make_def("switch", ActionCategory::LiveSwitching, true));
        assert!(cat.find("switch").is_some());
        assert!(cat.find("unknown").is_none());
    }

    #[test]
    fn test_catalog_len() {
        let mut cat = ActionCatalog::new();
        assert_eq!(cat.len(), 0);
        cat.register(make_def("a", ActionCategory::Notification, false));
        cat.register(make_def("b", ActionCategory::Notification, false));
        assert_eq!(cat.len(), 2);
    }

    #[test]
    fn test_catalog_is_empty() {
        let cat = ActionCatalog::new();
        assert!(cat.is_empty());
    }

    #[test]
    fn test_catalog_by_category() {
        let mut cat = ActionCatalog::new();
        cat.register(make_def("a", ActionCategory::DeviceControl, false));
        cat.register(make_def("b", ActionCategory::DeviceControl, true));
        cat.register(make_def("c", ActionCategory::Notification, false));
        let dc = cat.by_category(&ActionCategory::DeviceControl);
        assert_eq!(dc.len(), 2);
    }

    #[test]
    fn test_catalog_reversible() {
        let mut cat = ActionCatalog::new();
        cat.register(make_def("rev", ActionCategory::LiveSwitching, true));
        cat.register(make_def("irrev", ActionCategory::SystemControl, false));
        let rev = cat.reversible();
        assert_eq!(rev.len(), 1);
        assert_eq!(rev[0].id, "rev");
    }

    #[test]
    fn test_catalog_overwrite() {
        let mut cat = ActionCatalog::new();
        let a1 = ActionDef::new("id1", "Old Name", ActionCategory::Notification, "d", false);
        let a2 = ActionDef::new("id1", "New Name", ActionCategory::Notification, "d", true);
        cat.register(a1);
        cat.register(a2);
        assert_eq!(cat.len(), 1);
        assert_eq!(
            cat.find("id1").expect("find should succeed").name,
            "New Name"
        );
    }

    #[test]
    fn test_catalog_all_sorted() {
        let mut cat = ActionCatalog::new();
        cat.register(make_def("z_action", ActionCategory::Notification, false));
        cat.register(make_def("a_action", ActionCategory::Notification, false));
        cat.register(make_def("m_action", ActionCategory::Notification, false));
        let sorted = cat.all_sorted();
        assert_eq!(sorted[0].name, "a_action");
        assert_eq!(sorted[2].name, "z_action");
    }

    #[test]
    fn test_action_def_fields() {
        let a = ActionDef::new("id", "Name", ActionCategory::AssetManagement, "About", true);
        assert_eq!(a.id, "id");
        assert_eq!(a.name, "Name");
        assert_eq!(a.description, "About");
    }
}
