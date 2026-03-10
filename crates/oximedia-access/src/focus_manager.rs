#![allow(dead_code)]
//! Focus management for accessible media player controls.
//!
//! Implements a focus-ring system that tracks which interactive element
//! currently holds keyboard focus, supports tab-order navigation, and
//! provides focus-trap regions for modal dialogs.

use std::collections::HashMap;

/// The type of focusable element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FocusableKind {
    /// A clickable button.
    Button,
    /// A slider control (volume, timeline).
    Slider,
    /// A menu or dropdown.
    Menu,
    /// A text input field.
    TextInput,
    /// A toggle/checkbox.
    Toggle,
    /// A list item in a list box.
    ListItem,
    /// A custom interactive region.
    Custom,
}

impl FocusableKind {
    /// Return the ARIA role string for this kind.
    #[must_use]
    pub fn aria_role(&self) -> &'static str {
        match self {
            Self::Button => "button",
            Self::Slider => "slider",
            Self::Menu => "menu",
            Self::TextInput => "textbox",
            Self::Toggle => "checkbox",
            Self::ListItem => "option",
            Self::Custom => "group",
        }
    }
}

/// A single focusable element in the UI.
#[derive(Debug, Clone)]
pub struct FocusableElement {
    /// Unique identifier.
    pub id: String,
    /// Type of element.
    pub kind: FocusableKind,
    /// Tab index for ordering (lower = earlier).
    pub tab_index: i32,
    /// Whether the element is currently enabled.
    pub enabled: bool,
    /// Whether the element is visible.
    pub visible: bool,
    /// Accessible label for screen readers.
    pub label: String,
    /// Optional group this element belongs to.
    pub group: Option<String>,
}

impl FocusableElement {
    /// Create a new focusable element.
    #[must_use]
    pub fn new(id: impl Into<String>, kind: FocusableKind, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            tab_index: 0,
            enabled: true,
            visible: true,
            label: label.into(),
            group: None,
        }
    }

    /// Set the tab index.
    #[must_use]
    pub fn with_tab_index(mut self, index: i32) -> Self {
        self.tab_index = index;
        self
    }

    /// Set the group name.
    #[must_use]
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Set whether the element is enabled.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Check whether this element can receive focus.
    #[must_use]
    pub fn is_focusable(&self) -> bool {
        self.enabled && self.visible
    }
}

/// Direction of focus navigation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDirection {
    /// Move forward in tab order.
    Forward,
    /// Move backward in tab order.
    Backward,
}

/// A focus trap that constrains tab navigation within a region.
#[derive(Debug, Clone)]
pub struct FocusTrap {
    /// Name of the trap region.
    pub name: String,
    /// IDs of elements within this trap.
    pub element_ids: Vec<String>,
    /// Whether the trap is currently active.
    pub active: bool,
}

impl FocusTrap {
    /// Create a new focus trap.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            element_ids: Vec::new(),
            active: false,
        }
    }

    /// Add an element ID to the trap.
    pub fn add_element(&mut self, id: impl Into<String>) {
        self.element_ids.push(id.into());
    }

    /// Activate the trap.
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate the trap.
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Check whether an element is inside this trap.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.element_ids.iter().any(|e| e == id)
    }

    /// Return the number of elements in the trap.
    #[must_use]
    pub fn len(&self) -> usize {
        self.element_ids.len()
    }

    /// Check whether the trap is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.element_ids.is_empty()
    }
}

/// Manages focus state and navigation across all focusable elements.
#[derive(Debug)]
pub struct FocusManager {
    /// All registered focusable elements, keyed by ID.
    elements: HashMap<String, FocusableElement>,
    /// Sorted order of element IDs by tab index.
    order: Vec<String>,
    /// Index into `order` of the currently focused element.
    current_index: Option<usize>,
    /// Active focus trap, if any.
    trap: Option<FocusTrap>,
    /// Whether order needs to be rebuilt.
    dirty: bool,
}

impl FocusManager {
    /// Create a new focus manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            elements: HashMap::new(),
            order: Vec::new(),
            current_index: None,
            trap: None,
            dirty: false,
        }
    }

    /// Register a focusable element.
    pub fn register(&mut self, element: FocusableElement) {
        self.elements.insert(element.id.clone(), element);
        self.dirty = true;
    }

    /// Unregister an element by ID.
    pub fn unregister(&mut self, id: &str) {
        self.elements.remove(id);
        self.dirty = true;
    }

    /// Rebuild the sorted order from registered elements.
    fn rebuild_order(&mut self) {
        let mut entries: Vec<_> = self
            .elements
            .values()
            .filter(|e| e.is_focusable())
            .collect();
        entries.sort_by(|a, b| a.tab_index.cmp(&b.tab_index).then(a.id.cmp(&b.id)));
        self.order = entries.iter().map(|e| e.id.clone()).collect();
        // Adjust current_index if needed
        if let Some(idx) = self.current_index {
            if idx >= self.order.len() {
                self.current_index = if self.order.is_empty() {
                    None
                } else {
                    Some(self.order.len() - 1)
                };
            }
        }
        self.dirty = false;
    }

    /// Return the ID of the currently focused element.
    #[must_use]
    pub fn focused(&mut self) -> Option<&str> {
        if self.dirty {
            self.rebuild_order();
        }
        self.current_index
            .and_then(|i| self.order.get(i))
            .map(String::as_str)
    }

    /// Set focus to a specific element by ID.
    pub fn focus(&mut self, id: &str) -> bool {
        if self.dirty {
            self.rebuild_order();
        }
        if let Some(pos) = self.order.iter().position(|e| e == id) {
            self.current_index = Some(pos);
            true
        } else {
            false
        }
    }

    /// Move focus in the given direction, wrapping around.
    pub fn navigate(&mut self, direction: FocusDirection) -> Option<&str> {
        if self.dirty {
            self.rebuild_order();
        }
        let effective_order = self.effective_order();
        if effective_order.is_empty() {
            return None;
        }
        let current = self
            .current_index
            .and_then(|i| self.order.get(i))
            .and_then(|id| effective_order.iter().position(|e| e == id));
        let next_pos = match (current, direction) {
            (Some(pos), FocusDirection::Forward) => (pos + 1) % effective_order.len(),
            (Some(0), FocusDirection::Backward) => effective_order.len() - 1,
            (Some(pos), FocusDirection::Backward) => pos - 1,
            (None, _) => 0,
        };
        let next_id = effective_order[next_pos].clone();
        // Map back to global order
        if let Some(global_pos) = self.order.iter().position(|e| *e == next_id) {
            self.current_index = Some(global_pos);
        }
        self.order
            .get(self.current_index.unwrap_or(0))
            .map(String::as_str)
    }

    /// Return the effective element order, respecting an active trap.
    fn effective_order(&self) -> Vec<String> {
        if let Some(trap) = &self.trap {
            if trap.active {
                return self
                    .order
                    .iter()
                    .filter(|id| trap.contains(id))
                    .cloned()
                    .collect();
            }
        }
        self.order.clone()
    }

    /// Set a focus trap. While active, navigation is confined to its elements.
    pub fn set_trap(&mut self, trap: FocusTrap) {
        self.trap = Some(trap);
    }

    /// Remove and deactivate the current focus trap.
    pub fn clear_trap(&mut self) {
        self.trap = None;
    }

    /// Return the total number of registered elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.elements.len()
    }

    /// Return the number of elements that can currently receive focus.
    #[must_use]
    pub fn focusable_count(&self) -> usize {
        self.elements.values().filter(|e| e.is_focusable()).count()
    }
}

impl Default for FocusManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_button(id: &str, tab: i32) -> FocusableElement {
        FocusableElement::new(id, FocusableKind::Button, id).with_tab_index(tab)
    }

    #[test]
    fn test_focusable_kind_aria_role() {
        assert_eq!(FocusableKind::Button.aria_role(), "button");
        assert_eq!(FocusableKind::Slider.aria_role(), "slider");
        assert_eq!(FocusableKind::Toggle.aria_role(), "checkbox");
    }

    #[test]
    fn test_focusable_element_creation() {
        let el = FocusableElement::new("play", FocusableKind::Button, "Play")
            .with_tab_index(1)
            .with_group("controls");
        assert_eq!(el.id, "play");
        assert_eq!(el.tab_index, 1);
        assert_eq!(el.group, Some("controls".to_string()));
        assert!(el.is_focusable());
    }

    #[test]
    fn test_disabled_not_focusable() {
        let el = FocusableElement::new("x", FocusableKind::Button, "X").with_enabled(false);
        assert!(!el.is_focusable());
    }

    #[test]
    fn test_focus_manager_register() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        assert_eq!(fm.element_count(), 2);
        assert_eq!(fm.focusable_count(), 2);
    }

    #[test]
    fn test_focus_manager_navigate_forward() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        fm.register(make_button("c", 2));
        let first = fm
            .navigate(FocusDirection::Forward)
            .expect("first should be valid")
            .to_string();
        assert_eq!(first, "a");
        let second = fm
            .navigate(FocusDirection::Forward)
            .expect("second should be valid")
            .to_string();
        assert_eq!(second, "b");
        let third = fm
            .navigate(FocusDirection::Forward)
            .expect("third should be valid")
            .to_string();
        assert_eq!(third, "c");
        // Wrap around
        let wrap = fm
            .navigate(FocusDirection::Forward)
            .expect("wrap should be valid")
            .to_string();
        assert_eq!(wrap, "a");
    }

    #[test]
    fn test_focus_manager_navigate_backward() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        fm.focus("b");
        let prev = fm
            .navigate(FocusDirection::Backward)
            .expect("prev should be valid")
            .to_string();
        assert_eq!(prev, "a");
    }

    #[test]
    fn test_focus_manager_set_focus() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        assert!(fm.focus("b"));
        assert_eq!(fm.focused().expect("focused should succeed"), "b");
        assert!(!fm.focus("nonexistent"));
    }

    #[test]
    fn test_focus_trap() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        fm.register(make_button("c", 2));
        let mut trap = FocusTrap::new("modal");
        trap.add_element("b");
        trap.add_element("c");
        trap.activate();
        fm.set_trap(trap);
        fm.focus("b");
        let next = fm
            .navigate(FocusDirection::Forward)
            .expect("next should be valid")
            .to_string();
        assert_eq!(next, "c");
        let wrap = fm
            .navigate(FocusDirection::Forward)
            .expect("wrap should be valid")
            .to_string();
        assert_eq!(wrap, "b");
    }

    #[test]
    fn test_focus_trap_contains() {
        let mut trap = FocusTrap::new("dialog");
        trap.add_element("ok");
        trap.add_element("cancel");
        assert!(trap.contains("ok"));
        assert!(!trap.contains("other"));
        assert_eq!(trap.len(), 2);
    }

    #[test]
    fn test_focus_manager_unregister() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        fm.register(make_button("b", 1));
        fm.unregister("a");
        assert_eq!(fm.element_count(), 1);
    }

    #[test]
    fn test_focus_manager_clear_trap() {
        let mut fm = FocusManager::new();
        fm.register(make_button("a", 0));
        let mut trap = FocusTrap::new("t");
        trap.activate();
        fm.set_trap(trap);
        fm.clear_trap();
        // Navigate normally now
        let r = fm.navigate(FocusDirection::Forward);
        assert!(r.is_some());
    }

    #[test]
    fn test_focus_manager_empty_navigate() {
        let mut fm = FocusManager::new();
        assert!(fm.navigate(FocusDirection::Forward).is_none());
    }

    #[test]
    fn test_focus_trap_deactivate() {
        let mut trap = FocusTrap::new("modal");
        trap.activate();
        assert!(trap.active);
        trap.deactivate();
        assert!(!trap.active);
    }
}
