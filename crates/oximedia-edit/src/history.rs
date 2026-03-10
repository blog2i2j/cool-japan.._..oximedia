//! Edit history for undo/redo functionality.
//!
//! This module provides a complete undo/redo system for timeline editing operations.

#![allow(dead_code)]

/// An action that was performed on the timeline and can be undone/redone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditAction {
    /// A clip was added to the timeline.
    AddClip {
        /// The ID of the clip that was added.
        clip_id: u64,
    },
    /// A clip was removed from the timeline.
    RemoveClip {
        /// The ID of the clip that was removed.
        clip_id: u64,
    },
    /// A clip was moved to a new position.
    MoveClip {
        /// The ID of the clip that was moved.
        clip_id: u64,
        /// The original start position (in timebase units).
        old_pos: u64,
        /// The new start position (in timebase units).
        new_pos: u64,
    },
    /// A clip was trimmed (in or out point changed).
    TrimClip {
        /// The ID of the clip that was trimmed.
        clip_id: u64,
        /// The original in-point.
        old_in: u64,
        /// The original out-point.
        old_out: u64,
        /// The new in-point.
        new_in: u64,
        /// The new out-point.
        new_out: u64,
    },
    /// A transition was added between two clips.
    AddTransition {
        /// The ID of the first clip (A-side).
        clip_a: u64,
        /// The ID of the second clip (B-side).
        clip_b: u64,
        /// The duration of the transition (in timebase units).
        duration: u64,
    },
}

impl EditAction {
    /// Returns a human-readable description of this action.
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::AddClip { .. } => "Add clip",
            Self::RemoveClip { .. } => "Remove clip",
            Self::MoveClip { .. } => "Move clip",
            Self::TrimClip { .. } => "Trim clip",
            Self::AddTransition { .. } => "Add transition",
        }
    }

    /// Returns the inverse of this action (what undo would do).
    #[must_use]
    pub fn inverse(&self) -> Self {
        match *self {
            Self::AddClip { clip_id } => Self::RemoveClip { clip_id },
            Self::RemoveClip { clip_id } => Self::AddClip { clip_id },
            Self::MoveClip {
                clip_id,
                old_pos,
                new_pos,
            } => Self::MoveClip {
                clip_id,
                old_pos: new_pos,
                new_pos: old_pos,
            },
            Self::TrimClip {
                clip_id,
                old_in,
                old_out,
                new_in,
                new_out,
            } => Self::TrimClip {
                clip_id,
                old_in: new_in,
                old_out: new_out,
                new_in: old_in,
                new_out: old_out,
            },
            Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            } => Self::AddTransition {
                clip_a,
                clip_b,
                duration,
            },
        }
    }
}

/// Manages the undo/redo history for editing operations.
///
/// Maintains two stacks: an undo stack (past actions) and a redo stack
/// (actions that were undone and can be redone). Each `push` of a new action
/// clears the redo stack.
#[derive(Debug, Clone)]
pub struct EditHistory {
    /// Stack of actions that can be undone (most recent at back).
    undo_stack: Vec<EditAction>,
    /// Stack of actions that can be redone (most recent at back).
    redo_stack: Vec<EditAction>,
    /// Maximum number of actions to keep in the undo stack.
    max_depth: usize,
}

impl EditHistory {
    /// Creates a new `EditHistory` with the given maximum undo depth.
    ///
    /// # Arguments
    ///
    /// * `max_depth` - Maximum number of actions to remember. Once exceeded,
    ///   the oldest entries are dropped.
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    /// Pushes a new action onto the undo stack and clears the redo stack.
    ///
    /// If the undo stack exceeds `max_depth`, the oldest action is dropped.
    pub fn push(&mut self, action: EditAction) {
        self.redo_stack.clear();
        self.undo_stack.push(action);
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Pops the most recent action from the undo stack and pushes it onto
    /// the redo stack. Returns the action to be undone.
    pub fn undo(&mut self) -> Option<EditAction> {
        let action = self.undo_stack.pop()?;
        self.redo_stack.push(action.clone());
        Some(action)
    }

    /// Pops the most recent action from the redo stack and pushes it onto
    /// the undo stack. Returns the action to be redone.
    pub fn redo(&mut self) -> Option<EditAction> {
        let action = self.redo_stack.pop()?;
        self.undo_stack.push(action.clone());
        Some(action)
    }

    /// Returns `true` if there are any actions that can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns `true` if there are any actions that can be redone.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clears both the undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Returns the number of actions currently in the undo stack.
    #[must_use]
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the number of actions currently in the redo stack.
    #[must_use]
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Returns the maximum undo depth for this history.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns a slice of all actions in the undo stack (oldest first).
    #[must_use]
    pub fn undo_stack(&self) -> &[EditAction] {
        &self.undo_stack
    }

    /// Returns a slice of all actions in the redo stack (oldest first).
    #[must_use]
    pub fn redo_stack(&self) -> &[EditAction] {
        &self.redo_stack
    }
}

impl Default for EditHistory {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_history_is_empty() {
        let h = EditHistory::new(50);
        assert!(!h.can_undo());
        assert!(!h.can_redo());
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 0);
    }

    #[test]
    fn test_push_enables_undo() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        assert!(h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn test_undo_returns_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 42 });
        let action = h.undo();
        assert!(action.is_some());
        assert_eq!(
            action.expect("test expectation failed"),
            EditAction::AddClip { clip_id: 42 }
        );
    }

    #[test]
    fn test_undo_enables_redo() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.undo();
        assert!(!h.can_undo());
        assert!(h.can_redo());
    }

    #[test]
    fn test_redo_returns_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::RemoveClip { clip_id: 7 });
        h.undo();
        let redone = h.redo();
        assert!(redone.is_some());
        assert_eq!(
            redone.expect("test expectation failed"),
            EditAction::RemoveClip { clip_id: 7 }
        );
    }

    #[test]
    fn test_push_clears_redo_stack() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.undo();
        assert!(h.can_redo());
        h.push(EditAction::AddClip { clip_id: 2 });
        assert!(!h.can_redo(), "pushing a new action must clear redo stack");
    }

    #[test]
    fn test_max_depth_enforced() {
        let mut h = EditHistory::new(3);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.push(EditAction::AddClip { clip_id: 3 });
        h.push(EditAction::AddClip { clip_id: 4 }); // should evict clip_id 1
        assert_eq!(h.undo_count(), 3);
        // The oldest remaining action should be clip_id=2
        assert_eq!(h.undo_stack()[0], EditAction::AddClip { clip_id: 2 });
    }

    #[test]
    fn test_clear_resets_both_stacks() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.undo();
        h.clear();
        assert!(!h.can_undo());
        assert!(!h.can_redo());
    }

    #[test]
    fn test_undo_empty_returns_none() {
        let mut h = EditHistory::new(50);
        assert!(h.undo().is_none());
    }

    #[test]
    fn test_redo_empty_returns_none() {
        let mut h = EditHistory::new(50);
        assert!(h.redo().is_none());
    }

    #[test]
    fn test_move_clip_action() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::MoveClip {
            clip_id: 5,
            old_pos: 100,
            new_pos: 200,
        });
        let action = h.undo().expect("action should be valid");
        assert_eq!(
            action,
            EditAction::MoveClip {
                clip_id: 5,
                old_pos: 100,
                new_pos: 200,
            }
        );
    }

    #[test]
    fn test_trim_clip_action() {
        let action = EditAction::TrimClip {
            clip_id: 10,
            old_in: 0,
            old_out: 1000,
            new_in: 100,
            new_out: 900,
        };
        let inv = action.inverse();
        assert_eq!(
            inv,
            EditAction::TrimClip {
                clip_id: 10,
                old_in: 100,
                old_out: 900,
                new_in: 0,
                new_out: 1000,
            }
        );
    }

    #[test]
    fn test_action_descriptions() {
        assert_eq!(EditAction::AddClip { clip_id: 1 }.description(), "Add clip");
        assert_eq!(
            EditAction::RemoveClip { clip_id: 1 }.description(),
            "Remove clip"
        );
        assert_eq!(
            EditAction::MoveClip {
                clip_id: 1,
                old_pos: 0,
                new_pos: 0
            }
            .description(),
            "Move clip"
        );
        assert_eq!(
            EditAction::TrimClip {
                clip_id: 1,
                old_in: 0,
                old_out: 0,
                new_in: 0,
                new_out: 0
            }
            .description(),
            "Trim clip"
        );
        assert_eq!(
            EditAction::AddTransition {
                clip_a: 1,
                clip_b: 2,
                duration: 30
            }
            .description(),
            "Add transition"
        );
    }

    #[test]
    fn test_multiple_undo_redo_cycle() {
        let mut h = EditHistory::new(50);
        h.push(EditAction::AddClip { clip_id: 1 });
        h.push(EditAction::AddClip { clip_id: 2 });
        h.push(EditAction::AddClip { clip_id: 3 });

        // Undo all three
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 3 }
        );
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 2 }
        );
        assert_eq!(
            h.undo().expect("undo should succeed"),
            EditAction::AddClip { clip_id: 1 }
        );
        assert!(!h.can_undo());

        // Redo all three
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 1 }
        );
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 2 }
        );
        assert_eq!(
            h.redo().expect("redo should succeed"),
            EditAction::AddClip { clip_id: 3 }
        );
        assert!(!h.can_redo());
    }

    #[test]
    fn test_default_max_depth() {
        let h = EditHistory::default();
        assert_eq!(h.max_depth(), 100);
    }
}
