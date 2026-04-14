//! Undo/redo history for playlist editing operations.
//!
//! This module provides a [`PlaylistHistory`] stack that records [`PlaylistEdit`]
//! operations so that any edit can be rolled back (undo) or re-applied (redo)
//! with correct semantic inversion.
//!
//! # Inversion semantics
//!
//! Each `undo()` call pops the most-recently-pushed edit, computes its inverse,
//! and pushes the inverse onto the redo stack (so that `redo()` can replay the
//! original edit).  The inverse of each variant is:
//!
//! | Variant                 | Inverse                                                      |
//! |-------------------------|--------------------------------------------------------------|
//! | `InsertItem`            | `RemoveItem` (same index and item_id)                        |
//! | `RemoveItem`            | `InsertItem` (same index and item_id)                        |
//! | `MoveItem { from, to }` | `MoveItem { from: to, to: from }`                            |
//! | `UpdateDuration`        | `UpdateDuration` (old and new swapped)                       |
//! | `SetCrossfade`          | `SetCrossfade` (no clean inverse; returned unchanged)        |

#![allow(dead_code)]

/// A single playlist editing operation that can be recorded, undone, or redone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlaylistEdit {
    /// An item was inserted at the given zero-based index.
    InsertItem {
        /// Index at which the item was inserted.
        index: usize,
        /// Identifier of the inserted item.
        item_id: String,
    },
    /// An item was removed from the given zero-based index.
    RemoveItem {
        /// Index from which the item was removed.
        index: usize,
        /// Identifier of the removed item.
        item_id: String,
    },
    /// An item was moved between two positions.
    MoveItem {
        /// Original position before the move.
        from: usize,
        /// Target position after the move.
        to: usize,
    },
    /// An item's duration was changed.
    UpdateDuration {
        /// Identifier of the affected item.
        item_id: String,
        /// The duration in milliseconds before the update.
        old_duration_ms: u64,
        /// The duration in milliseconds after the update.
        new_duration_ms: u64,
    },
    /// A crossfade duration was set on an item.
    SetCrossfade {
        /// Identifier of the affected item.
        item_id: String,
        /// Crossfade duration in milliseconds.
        duration_ms: u64,
    },
}

impl PlaylistEdit {
    /// Compute the semantic inverse of this edit.
    ///
    /// The inverse is the edit that must be applied to the playlist in order to
    /// restore it to the state it had *before* the original edit was applied.
    ///
    /// `SetCrossfade` has no clean inverse (the previous value is not stored in
    /// the edit itself), so the same edit is returned unchanged — callers should
    /// be aware that re-applying a `SetCrossfade` will overwrite whatever
    /// crossfade is currently active.
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            Self::InsertItem { index, item_id } => Self::RemoveItem {
                index: *index,
                item_id: item_id.clone(),
            },
            Self::RemoveItem { index, item_id } => Self::InsertItem {
                index: *index,
                item_id: item_id.clone(),
            },
            Self::MoveItem { from, to } => Self::MoveItem {
                from: *to,
                to: *from,
            },
            Self::UpdateDuration {
                item_id,
                old_duration_ms,
                new_duration_ms,
            } => Self::UpdateDuration {
                item_id: item_id.clone(),
                old_duration_ms: *new_duration_ms,
                new_duration_ms: *old_duration_ms,
            },
            // No clean inverse — return unchanged.
            Self::SetCrossfade { item_id, duration_ms } => Self::SetCrossfade {
                item_id: item_id.clone(),
                duration_ms: *duration_ms,
            },
        }
    }
}

/// Undo/redo history manager for a single playlist.
///
/// Internally maintains two bounded stacks — an undo stack (recording edits in
/// chronological order) and a redo stack (recording edits that have been undone
/// and can be re-applied).  Both stacks are bounded to `max_depth` entries;
/// when the undo stack overflows the *oldest* entry is discarded (i.e., the
/// edit that was furthest back in history is forgotten).
///
/// # Example
///
/// ```rust
/// use oximedia_playlist::playlist_history::{PlaylistHistory, PlaylistEdit};
///
/// let mut hist = PlaylistHistory::new(10);
/// hist.push(PlaylistEdit::InsertItem { index: 0, item_id: "song_a".into() });
///
/// // Undo returns the *inverse* — what to apply to revert the insert.
/// let inv = hist.undo().expect("should have undo");
/// assert!(matches!(inv, PlaylistEdit::RemoveItem { index: 0, .. }));
///
/// // Redo returns the original edit to re-apply it.
/// let redo = hist.redo().expect("should have redo");
/// assert!(matches!(redo, PlaylistEdit::InsertItem { index: 0, .. }));
/// ```
#[derive(Debug, Clone)]
pub struct PlaylistHistory {
    undo_stack: Vec<PlaylistEdit>,
    redo_stack: Vec<PlaylistEdit>,
    max_depth: usize,
}

impl PlaylistHistory {
    /// Create a new `PlaylistHistory` with the given maximum stack depth.
    ///
    /// A `max_depth` of `0` means unlimited (no trimming).
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
        }
    }

    /// Push a new edit onto the undo stack.
    ///
    /// This clears the redo stack (a new edit invalidates any previously
    /// undone sequence), and trims the undo stack to `max_depth` by
    /// discarding the oldest entry if needed.
    pub fn push(&mut self, edit: PlaylistEdit) {
        self.redo_stack.clear();
        self.undo_stack.push(edit);
        if self.max_depth > 0 && self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the most recent edit.
    ///
    /// Pops the top of the undo stack, computes its inverse, pushes the
    /// *original* edit onto the redo stack (so `redo()` can replay it), and
    /// returns the inverse edit (i.e., what the caller should apply to the
    /// playlist to reverse the operation).
    ///
    /// Returns `None` if the undo stack is empty.
    pub fn undo(&mut self) -> Option<PlaylistEdit> {
        let edit = self.undo_stack.pop()?;
        let inverse = edit.inverse();
        // Push the original onto the redo stack so it can be re-applied.
        self.redo_stack.push(edit);
        Some(inverse)
    }

    /// Redo the most recently undone edit.
    ///
    /// Pops the top of the redo stack, computes its inverse (to push onto
    /// the undo stack — conceptually "re-doing" the original edit means we
    /// can undo it again), and returns the original edit that the caller
    /// should re-apply to the playlist.
    ///
    /// Returns `None` if the redo stack is empty.
    pub fn redo(&mut self) -> Option<PlaylistEdit> {
        let edit = self.redo_stack.pop()?;
        // Re-push onto the undo stack so it can be undone again.
        self.undo_stack.push(edit.clone());
        // Return the edit itself (the caller applies it to the playlist).
        Some(edit)
    }

    /// Return `true` if there is at least one edit that can be undone.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Return `true` if there is at least one edit that can be redone.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clear both the undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Number of edits currently on the undo stack.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of edits currently on the redo stack.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn insert(index: usize, id: &str) -> PlaylistEdit {
        PlaylistEdit::InsertItem {
            index,
            item_id: id.into(),
        }
    }

    fn remove(index: usize, id: &str) -> PlaylistEdit {
        PlaylistEdit::RemoveItem {
            index,
            item_id: id.into(),
        }
    }

    // T1: push clears redo stack
    #[test]
    fn test_push_clears_redo_stack() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(insert(0, "a"));
        hist.undo();
        assert!(hist.can_redo(), "redo should be available after undo");
        // A new push must clear the redo stack.
        hist.push(insert(1, "b"));
        assert!(!hist.can_redo(), "redo stack should be cleared after new push");
    }

    // T2: undo returns inverse edit
    #[test]
    fn test_undo_returns_inverse_insert() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(insert(2, "song_x"));
        let inv = hist.undo().expect("undo should return Some");
        assert_eq!(inv, remove(2, "song_x"));
    }

    #[test]
    fn test_undo_returns_inverse_remove() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(remove(3, "song_y"));
        let inv = hist.undo().expect("undo should return Some");
        assert_eq!(inv, insert(3, "song_y"));
    }

    #[test]
    fn test_undo_returns_inverse_move() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(PlaylistEdit::MoveItem { from: 1, to: 4 });
        let inv = hist.undo().expect("undo should return Some");
        assert_eq!(inv, PlaylistEdit::MoveItem { from: 4, to: 1 });
    }

    #[test]
    fn test_undo_returns_inverse_update_duration() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(PlaylistEdit::UpdateDuration {
            item_id: "clip_001".into(),
            old_duration_ms: 5000,
            new_duration_ms: 8000,
        });
        let inv = hist.undo().expect("undo should return Some");
        assert_eq!(
            inv,
            PlaylistEdit::UpdateDuration {
                item_id: "clip_001".into(),
                old_duration_ms: 8000,
                new_duration_ms: 5000,
            }
        );
    }

    // T3: redo re-applies the original edit
    #[test]
    fn test_redo_after_undo() {
        let mut hist = PlaylistHistory::new(10);
        let original = insert(0, "track_a");
        hist.push(original.clone());
        hist.undo();
        let redone = hist.redo().expect("redo should return Some");
        // redo returns the original edit that should be re-applied.
        assert_eq!(redone, original);
    }

    // T4: max_depth trims oldest undo entry
    #[test]
    fn test_max_depth_trims_oldest() {
        let mut hist = PlaylistHistory::new(3);
        hist.push(insert(0, "first"));
        hist.push(insert(1, "second"));
        hist.push(insert(2, "third"));
        // This fourth push should evict "first".
        hist.push(insert(3, "fourth"));
        assert_eq!(hist.undo_depth(), 3);
        // The oldest remaining should be "second" (first was evicted).
        // Undo three times, the last inverse should correspond to insert(1, "second").
        hist.undo(); // undoes "fourth"
        hist.undo(); // undoes "third"
        let last_inv = hist.undo().expect("should still have entry for second");
        assert_eq!(last_inv, remove(1, "second"));
    }

    // T5: can_undo and can_redo
    #[test]
    fn test_can_undo_redo_flags() {
        let mut hist = PlaylistHistory::new(10);
        assert!(!hist.can_undo());
        assert!(!hist.can_redo());
        hist.push(insert(0, "a"));
        assert!(hist.can_undo());
        assert!(!hist.can_redo());
        hist.undo();
        assert!(!hist.can_undo());
        assert!(hist.can_redo());
    }

    // T6: clear resets both stacks
    #[test]
    fn test_clear_resets_both_stacks() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(insert(0, "a"));
        hist.push(insert(1, "b"));
        hist.undo();
        assert!(hist.can_undo());
        assert!(hist.can_redo());
        hist.clear();
        assert!(!hist.can_undo());
        assert!(!hist.can_redo());
        assert_eq!(hist.undo_depth(), 0);
        assert_eq!(hist.redo_depth(), 0);
    }

    // T7: undo at empty returns None
    #[test]
    fn test_undo_at_empty_returns_none() {
        let mut hist = PlaylistHistory::new(10);
        assert!(hist.undo().is_none());
    }

    // T8: redo after undo then push (redo stack cleared again)
    #[test]
    fn test_redo_after_undo_then_push() {
        let mut hist = PlaylistHistory::new(10);
        hist.push(insert(0, "alpha"));
        hist.undo();
        assert!(hist.can_redo());
        // A new push clears redo.
        hist.push(insert(0, "beta"));
        assert!(!hist.can_redo(), "redo must be unavailable after new push");
        assert!(hist.can_undo());
    }

    // Depth tracking
    #[test]
    fn test_depth_tracking() {
        let mut hist = PlaylistHistory::new(10);
        assert_eq!(hist.undo_depth(), 0);
        assert_eq!(hist.redo_depth(), 0);
        hist.push(insert(0, "x"));
        hist.push(insert(1, "y"));
        assert_eq!(hist.undo_depth(), 2);
        hist.undo();
        assert_eq!(hist.undo_depth(), 1);
        assert_eq!(hist.redo_depth(), 1);
    }
}
