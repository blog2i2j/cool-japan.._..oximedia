#![allow(dead_code)]
//! Multi-clip selection and group operations.
//!
//! This module provides group operations on multiple selected clips:
//! move, trim, delete, copy, and paste.  A [`Selection`] holds the set
//! of `(TrackId, ClipId)` pairs that are currently selected, and
//! [`SelectionOp`] describes what to do with them.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::clip::ClipId;
use crate::track::TrackId;
use crate::types::Position;

/// A set of selected clips identified by `(TrackId, ClipId)` pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Selection {
    items: HashSet<(TrackId, ClipId)>,
}

impl Selection {
    /// Create an empty selection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a clip to the selection.
    pub fn add(&mut self, track_id: TrackId, clip_id: ClipId) {
        self.items.insert((track_id, clip_id));
    }

    /// Remove a clip from the selection.
    pub fn remove(&mut self, track_id: TrackId, clip_id: ClipId) {
        self.items.remove(&(track_id, clip_id));
    }

    /// Toggle a clip's selection state.
    pub fn toggle(&mut self, track_id: TrackId, clip_id: ClipId) {
        let key = (track_id, clip_id);
        if self.items.contains(&key) {
            self.items.remove(&key);
        } else {
            self.items.insert(key);
        }
    }

    /// Returns `true` if the given clip is selected.
    #[must_use]
    pub fn contains(&self, track_id: TrackId, clip_id: ClipId) -> bool {
        self.items.contains(&(track_id, clip_id))
    }

    /// Clear all selected clips.
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Number of selected clips.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if no clips are selected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over `(TrackId, ClipId)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(TrackId, ClipId)> {
        self.items.iter()
    }

    /// Collect items into a sorted vec (deterministic order for tests).
    #[must_use]
    pub fn to_sorted_vec(&self) -> Vec<(TrackId, ClipId)> {
        let mut v: Vec<_> = self.items.iter().copied().collect();
        v.sort_by_key(|(tid, cid)| (tid.as_uuid().to_string(), cid.as_uuid().to_string()));
        v
    }
}

/// A single group operation applied to the current [`Selection`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionOp {
    /// Move all selected clips by `delta` frames.  Positive = forward in time.
    Move {
        /// Frame delta (positive = forward in time).
        delta: i64,
    },
    /// Trim the in-point of all selected clips by `delta` frames.
    TrimIn {
        /// Frame delta for the in-point.
        delta: i64,
    },
    /// Trim the out-point of all selected clips by `delta` frames.
    TrimOut {
        /// Frame delta for the out-point.
        delta: i64,
    },
    /// Delete all selected clips.
    Delete,
    /// Copy selected clips to the clipboard (clipboard is opaque String JSON).
    Copy,
    /// Paste clipboard clips at the given timeline position.
    Paste {
        /// Timeline position at which to paste.
        at: Position,
    },
    /// Set the volume of all selected audio clips to `gain` (0.0–1.0).
    SetVolume {
        /// Gain value (0.0–1.0).
        gain: f32,
    },
    /// Disable/enable all selected clips.
    SetEnabled {
        /// Whether clips should be enabled.
        enabled: bool,
    },
}

/// Describes a pending move instruction for a single clip.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClipMoveInstruction {
    /// Track containing the clip.
    pub track_id: TrackId,
    /// Clip to move.
    pub clip_id: ClipId,
    /// Frame delta (positive = later in timeline).
    pub delta: i64,
}

/// Describes a pending trim instruction for a single clip.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClipTrimInstruction {
    /// Track containing the clip.
    pub track_id: TrackId,
    /// Clip to trim.
    pub clip_id: ClipId,
    /// Frame delta for the edit point.
    pub delta: i64,
    /// Whether to trim the in-point (`true`) or the out-point (`false`).
    pub trim_in: bool,
}

/// Result of executing a group operation on a selection.
#[derive(Debug, Default)]
pub struct SelectionOpResult {
    /// Move instructions generated (for `SelectionOp::Move`).
    pub moves: Vec<ClipMoveInstruction>,
    /// Trim instructions generated (for `TrimIn`/`TrimOut`).
    pub trims: Vec<ClipTrimInstruction>,
    /// Clip IDs to delete (for `Delete`).
    pub deletes: Vec<(TrackId, ClipId)>,
    /// JSON-serialised clipboard payload (for `Copy`).
    pub clipboard: Option<String>,
    /// Clips to create from paste (track_id, at_position, source_clip_json).
    pub pastes: Vec<(TrackId, Position, String)>,
}

/// Executes group operations on a [`Selection`].
///
/// This is a pure data-transformation layer — it produces structured
/// instruction lists that the caller (the timeline editor) applies to the
/// actual data model.  This separation makes the multi-select logic testable
/// without a full timeline instance.
#[derive(Debug, Default)]
pub struct MultiSelect {
    /// Current clipboard contents (JSON-encoded clip list).
    clipboard: Option<String>,
}

impl MultiSelect {
    /// Create a new `MultiSelect` instance.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute `op` on `selection` and return the resulting instructions.
    ///
    /// The caller is responsible for applying the instructions to the timeline.
    pub fn execute(
        &mut self,
        selection: &Selection,
        op: SelectionOp,
    ) -> SelectionOpResult {
        let mut result = SelectionOpResult::default();

        match op {
            SelectionOp::Move { delta } => {
                for &(track_id, clip_id) in selection.iter() {
                    result.moves.push(ClipMoveInstruction {
                        track_id,
                        clip_id,
                        delta,
                    });
                }
            }
            SelectionOp::TrimIn { delta } => {
                for &(track_id, clip_id) in selection.iter() {
                    result.trims.push(ClipTrimInstruction {
                        track_id,
                        clip_id,
                        delta,
                        trim_in: true,
                    });
                }
            }
            SelectionOp::TrimOut { delta } => {
                for &(track_id, clip_id) in selection.iter() {
                    result.trims.push(ClipTrimInstruction {
                        track_id,
                        clip_id,
                        delta,
                        trim_in: false,
                    });
                }
            }
            SelectionOp::Delete => {
                for &pair in selection.iter() {
                    result.deletes.push(pair);
                }
            }
            SelectionOp::Copy => {
                // Encode selected clip IDs to JSON.
                let pairs: Vec<_> = selection.to_sorted_vec();
                let json = serde_json::to_string(&pairs).unwrap_or_default();
                self.clipboard = Some(json.clone());
                result.clipboard = Some(json);
            }
            SelectionOp::Paste { at } => {
                if let Some(ref payload) = self.clipboard {
                    // Each clipboard entry is pasted on the same track at the new position.
                    if let Ok(pairs) = serde_json::from_str::<Vec<(TrackId, ClipId)>>(payload) {
                        for (offset, (track_id, _clip_id)) in pairs.into_iter().enumerate() {
                            result.pastes.push((
                                track_id,
                                Position::new(at.0 + offset as i64),
                                payload.clone(),
                            ));
                        }
                    }
                }
            }
            SelectionOp::SetVolume { .. } | SelectionOp::SetEnabled { .. } => {
                // These are metadata-only operations; the caller applies them
                // directly using the selection membership list.
                // Returning an empty result is intentional.
            }
        }

        result
    }

    /// Return the current clipboard contents, if any.
    #[must_use]
    pub fn clipboard(&self) -> Option<&str> {
        self.clipboard.as_deref()
    }

    /// Clear the clipboard.
    pub fn clear_clipboard(&mut self) {
        self.clipboard = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ids() -> (TrackId, ClipId) {
        (TrackId::new(), ClipId::new())
    }

    #[test]
    fn test_selection_add_contains() {
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);
        assert!(sel.contains(tid, cid));
        assert_eq!(sel.len(), 1);
    }

    #[test]
    fn test_selection_remove() {
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);
        sel.remove(tid, cid);
        assert!(!sel.contains(tid, cid));
        assert!(sel.is_empty());
    }

    #[test]
    fn test_selection_toggle() {
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.toggle(tid, cid);
        assert!(sel.contains(tid, cid));
        sel.toggle(tid, cid);
        assert!(!sel.contains(tid, cid));
    }

    #[test]
    fn test_selection_clear() {
        let mut sel = Selection::new();
        let (t1, c1) = make_ids();
        let (t2, c2) = make_ids();
        sel.add(t1, c1);
        sel.add(t2, c2);
        sel.clear();
        assert!(sel.is_empty());
    }

    #[test]
    fn test_execute_move_produces_instructions() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);
        let result = ms.execute(&sel, SelectionOp::Move { delta: 10 });
        assert_eq!(result.moves.len(), 1);
        assert_eq!(result.moves[0].delta, 10);
    }

    #[test]
    fn test_execute_delete() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (t1, c1) = make_ids();
        let (t2, c2) = make_ids();
        sel.add(t1, c1);
        sel.add(t2, c2);
        let result = ms.execute(&sel, SelectionOp::Delete);
        assert_eq!(result.deletes.len(), 2);
    }

    #[test]
    fn test_execute_trim_in_trim_out() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);

        let r_in = ms.execute(&sel, SelectionOp::TrimIn { delta: 5 });
        assert_eq!(r_in.trims.len(), 1);
        assert!(r_in.trims[0].trim_in);

        let r_out = ms.execute(&sel, SelectionOp::TrimOut { delta: -3 });
        assert_eq!(r_out.trims.len(), 1);
        assert!(!r_out.trims[0].trim_in);
    }

    #[test]
    fn test_execute_copy_sets_clipboard() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);
        let result = ms.execute(&sel, SelectionOp::Copy);
        assert!(result.clipboard.is_some());
        assert!(ms.clipboard().is_some());
    }

    #[test]
    fn test_execute_paste_uses_clipboard() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);

        // Copy first
        ms.execute(&sel, SelectionOp::Copy);

        // Now paste at position 100
        let result = ms.execute(&sel, SelectionOp::Paste { at: Position::new(100) });
        assert!(!result.pastes.is_empty());
    }

    #[test]
    fn test_clear_clipboard() {
        let mut ms = MultiSelect::new();
        let mut sel = Selection::new();
        let (tid, cid) = make_ids();
        sel.add(tid, cid);
        ms.execute(&sel, SelectionOp::Copy);
        ms.clear_clipboard();
        assert!(ms.clipboard().is_none());
    }
}
