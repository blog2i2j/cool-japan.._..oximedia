#![allow(dead_code)]
//! Edit context management for scoped editing operations.

/// Scope of an edit operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EditScope {
    /// Affects only the targeted clip segment.
    Local,
    /// Affects all clips on the same track.
    Track,
    /// Affects all tracks in the timeline.
    Global,
}

impl EditScope {
    /// Returns true if this scope causes downstream clips to shift.
    #[must_use]
    pub fn affects_downstream(&self) -> bool {
        matches!(self, EditScope::Track | EditScope::Global)
    }

    /// Human-readable label for the scope.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            EditScope::Local => "Local",
            EditScope::Track => "Track",
            EditScope::Global => "Global",
        }
    }
}

/// A single editing context capturing scope, description, and whether it is reversible.
#[derive(Debug, Clone)]
pub struct EditContext {
    /// Unique identifier for this context.
    pub id: u64,
    /// Human-readable description (e.g. "Trim clip end").
    pub description: String,
    /// Scope of the context.
    pub scope: EditScope,
    /// Whether this context can be undone.
    pub reversible: bool,
    /// Epoch timestamp (ms) when the context was created.
    pub created_at_ms: u64,
}

impl EditContext {
    /// Creates a new edit context.
    pub fn new(
        id: u64,
        description: impl Into<String>,
        scope: EditScope,
        reversible: bool,
        created_at_ms: u64,
    ) -> Self {
        Self {
            id,
            description: description.into(),
            scope,
            reversible,
            created_at_ms,
        }
    }

    /// Returns `true` if an undo entry exists for this context.
    #[must_use]
    pub fn has_undo(&self) -> bool {
        self.reversible
    }
}

/// Manages a stack of edit contexts, supporting push/pop operations.
#[derive(Debug, Default)]
pub struct EditContextManager {
    stack: Vec<EditContext>,
    next_id: u64,
}

impl EditContextManager {
    /// Creates a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            next_id: 1,
        }
    }

    /// Pushes a new context onto the stack, returning its assigned id.
    pub fn push_context(
        &mut self,
        description: impl Into<String>,
        scope: EditScope,
        reversible: bool,
        timestamp_ms: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.stack.push(EditContext::new(
            id,
            description,
            scope,
            reversible,
            timestamp_ms,
        ));
        id
    }

    /// Pops the most recent context from the stack.
    pub fn pop_context(&mut self) -> Option<EditContext> {
        self.stack.pop()
    }

    /// Returns a reference to the current (topmost) context, if any.
    #[must_use]
    pub fn current(&self) -> Option<&EditContext> {
        self.stack.last()
    }

    /// Number of contexts currently on the stack.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Returns all reversible contexts, most recent first.
    #[must_use]
    pub fn reversible_contexts(&self) -> Vec<&EditContext> {
        self.stack.iter().rev().filter(|c| c.reversible).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_local_not_downstream() {
        assert!(!EditScope::Local.affects_downstream());
    }

    #[test]
    fn test_scope_track_downstream() {
        assert!(EditScope::Track.affects_downstream());
    }

    #[test]
    fn test_scope_global_downstream() {
        assert!(EditScope::Global.affects_downstream());
    }

    #[test]
    fn test_scope_labels() {
        assert_eq!(EditScope::Local.label(), "Local");
        assert_eq!(EditScope::Track.label(), "Track");
        assert_eq!(EditScope::Global.label(), "Global");
    }

    #[test]
    fn test_edit_context_has_undo_true() {
        let ctx = EditContext::new(1, "Trim", EditScope::Local, true, 0);
        assert!(ctx.has_undo());
    }

    #[test]
    fn test_edit_context_has_undo_false() {
        let ctx = EditContext::new(2, "Export", EditScope::Global, false, 0);
        assert!(!ctx.has_undo());
    }

    #[test]
    fn test_manager_new_empty() {
        let mgr = EditContextManager::new();
        assert_eq!(mgr.depth(), 0);
        assert!(mgr.current().is_none());
    }

    #[test]
    fn test_manager_push_returns_id() {
        let mut mgr = EditContextManager::new();
        let id = mgr.push_context("Cut", EditScope::Local, true, 100);
        assert_eq!(id, 1);
        let id2 = mgr.push_context("Ripple", EditScope::Track, true, 200);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_manager_current_is_latest() {
        let mut mgr = EditContextManager::new();
        mgr.push_context("First", EditScope::Local, true, 0);
        mgr.push_context("Second", EditScope::Track, false, 1);
        let cur = mgr.current().expect("cur should be valid");
        assert_eq!(cur.description, "Second");
    }

    #[test]
    fn test_manager_pop_returns_context() {
        let mut mgr = EditContextManager::new();
        mgr.push_context("Op", EditScope::Global, true, 50);
        let popped = mgr.pop_context().expect("popped should be valid");
        assert_eq!(popped.description, "Op");
        assert_eq!(mgr.depth(), 0);
    }

    #[test]
    fn test_manager_pop_empty_is_none() {
        let mut mgr = EditContextManager::new();
        assert!(mgr.pop_context().is_none());
    }

    #[test]
    fn test_manager_depth_grows() {
        let mut mgr = EditContextManager::new();
        for i in 0..5 {
            mgr.push_context(format!("Op{i}"), EditScope::Local, true, i as u64);
        }
        assert_eq!(mgr.depth(), 5);
    }

    #[test]
    fn test_reversible_contexts_filtered() {
        let mut mgr = EditContextManager::new();
        mgr.push_context("A", EditScope::Local, true, 0);
        mgr.push_context("B", EditScope::Track, false, 1);
        mgr.push_context("C", EditScope::Global, true, 2);
        let rev = mgr.reversible_contexts();
        assert_eq!(rev.len(), 2);
        // Most recent first
        assert_eq!(rev[0].description, "C");
        assert_eq!(rev[1].description, "A");
    }

    #[test]
    fn test_context_scope_stored() {
        let ctx = EditContext::new(10, "Test", EditScope::Track, true, 999);
        assert_eq!(ctx.scope, EditScope::Track);
        assert_eq!(ctx.id, 10);
    }
}
