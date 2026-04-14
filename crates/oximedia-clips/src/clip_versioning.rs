//! Clip versioning module for tracking edit history with undo/redo support.
//!
//! Provides a per-clip version stack that records every mutating operation,
//! enabling full undo/redo over the clip's editing history. Each
//! [`ClipEditOperation`] captures the field modified and the before/after
//! values so that the change can be reversed (undo) or re-applied (redo).
//!
//! # Example
//!
//! ```rust
//! use oximedia_clips::clip_versioning::{ClipVersionHistory, ClipEditOperation};
//!
//! let mut history = ClipVersionHistory::new("clip-01".to_string());
//! history.push(ClipEditOperation::NameChanged {
//!     before: "raw_take_1.mov".to_string(),
//!     after:  "Interview A".to_string(),
//! });
//! assert_eq!(history.undo_count(), 1);
//! let op = history.undo().expect("should have undo");
//! assert_eq!(history.undo_count(), 0);
//! assert_eq!(history.redo_count(), 1);
//! drop(op); // suppress unused-variable warning
//! ```

#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Edit operation variants
// ─────────────────────────────────────────────────────────────────────────────

/// A single reversible editing operation applied to a clip.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClipEditOperation {
    /// The clip's display name was changed.
    NameChanged {
        /// Value before the change.
        before: String,
        /// Value after the change.
        after: String,
    },

    /// The clip's description was changed.
    DescriptionChanged {
        /// Value before the change.
        before: Option<String>,
        /// Value after the change.
        after: Option<String>,
    },

    /// Star rating was changed.
    RatingChanged {
        /// Rating value before the change (serialised as `u8`).
        before: u8,
        /// Rating value after the change.
        after: u8,
    },

    /// A keyword was added.
    KeywordAdded {
        /// The keyword that was added.
        keyword: String,
    },

    /// A keyword was removed.
    KeywordRemoved {
        /// The keyword that was removed.
        keyword: String,
    },

    /// In-point was changed (frame number, `None` = cleared).
    InPointChanged {
        /// Previous in-point.
        before: Option<i64>,
        /// New in-point.
        after: Option<i64>,
    },

    /// Out-point was changed (frame number, `None` = cleared).
    OutPointChanged {
        /// Previous out-point.
        before: Option<i64>,
        /// New out-point.
        after: Option<i64>,
    },

    /// Favourite flag was toggled.
    FavoriteToggled {
        /// State before the toggle.
        before: bool,
        /// State after the toggle.
        after: bool,
    },

    /// Rejected flag was toggled.
    RejectedToggled {
        /// State before the toggle.
        before: bool,
        /// State after the toggle.
        after: bool,
    },

    /// Custom metadata was changed.
    CustomMetadataChanged {
        /// Key whose value was changed.
        key: String,
        /// Value before (serialised as JSON string or `None`).
        before: Option<String>,
        /// Value after.
        after: Option<String>,
    },
}

impl ClipEditOperation {
    /// Returns the inverse of this operation (for undo).
    #[must_use]
    pub fn invert(&self) -> Self {
        match self {
            Self::NameChanged { before, after } => Self::NameChanged {
                before: after.clone(),
                after: before.clone(),
            },
            Self::DescriptionChanged { before, after } => Self::DescriptionChanged {
                before: after.clone(),
                after: before.clone(),
            },
            Self::RatingChanged { before, after } => Self::RatingChanged {
                before: *after,
                after: *before,
            },
            Self::KeywordAdded { keyword } => Self::KeywordRemoved {
                keyword: keyword.clone(),
            },
            Self::KeywordRemoved { keyword } => Self::KeywordAdded {
                keyword: keyword.clone(),
            },
            Self::InPointChanged { before, after } => Self::InPointChanged {
                before: *after,
                after: *before,
            },
            Self::OutPointChanged { before, after } => Self::OutPointChanged {
                before: *after,
                after: *before,
            },
            Self::FavoriteToggled { before, after } => Self::FavoriteToggled {
                before: *after,
                after: *before,
            },
            Self::RejectedToggled { before, after } => Self::RejectedToggled {
                before: *after,
                after: *before,
            },
            Self::CustomMetadataChanged { key, before, after } => Self::CustomMetadataChanged {
                key: key.clone(),
                before: after.clone(),
                after: before.clone(),
            },
        }
    }

    /// Human-readable description of the operation.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            Self::NameChanged { before, after } => {
                format!("Rename \"{before}\" → \"{after}\"")
            }
            Self::DescriptionChanged { .. } => "Change description".to_string(),
            Self::RatingChanged { before, after } => {
                format!("Change rating {before}★ → {after}★")
            }
            Self::KeywordAdded { keyword } => format!("Add keyword \"{keyword}\""),
            Self::KeywordRemoved { keyword } => format!("Remove keyword \"{keyword}\""),
            Self::InPointChanged { after, .. } => match after {
                Some(f) => format!("Set in-point to frame {f}"),
                None => "Clear in-point".to_string(),
            },
            Self::OutPointChanged { after, .. } => match after {
                Some(f) => format!("Set out-point to frame {f}"),
                None => "Clear out-point".to_string(),
            },
            Self::FavoriteToggled { after, .. } => {
                if *after {
                    "Mark as favourite".to_string()
                } else {
                    "Unmark favourite".to_string()
                }
            }
            Self::RejectedToggled { after, .. } => {
                if *after {
                    "Mark as rejected".to_string()
                } else {
                    "Unmark rejected".to_string()
                }
            }
            Self::CustomMetadataChanged { key, .. } => {
                format!("Change metadata field \"{key}\"")
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Version entry
// ─────────────────────────────────────────────────────────────────────────────

/// A timestamped entry in the edit history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    /// Sequence number (monotonically increasing per history).
    pub sequence: u64,
    /// The operation that was applied.
    pub operation: ClipEditOperation,
    /// When the operation occurred.
    pub timestamp: DateTime<Utc>,
    /// Optional human-readable label (e.g., "v2 – colour grade").
    pub label: Option<String>,
}

impl VersionEntry {
    fn new(sequence: u64, operation: ClipEditOperation, label: Option<String>) -> Self {
        Self {
            sequence,
            operation,
            timestamp: Utc::now(),
            label,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipVersionHistory
// ─────────────────────────────────────────────────────────────────────────────

/// Per-clip edit history supporting unlimited undo/redo.
///
/// Operations are stored in an *undo stack*. `undo()` pops the most recent
/// entry, inverts it, pushes it onto the *redo stack*, and returns the
/// inverse operation so the caller can apply it to the clip. `redo()` reverses
/// this process.
///
/// When `push()` is called after one or more undos the entire redo stack is
/// discarded (branching edit history is not supported; only linear history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipVersionHistory {
    /// ID of the clip this history belongs to.
    pub clip_id: String,
    /// Operations that can be undone (most recent last).
    undo_stack: Vec<VersionEntry>,
    /// Operations that can be redone (most recent last).
    redo_stack: Vec<VersionEntry>,
    /// Monotonic counter for version sequence numbers.
    next_seq: u64,
    /// Optional maximum undo depth. `None` means unlimited.
    max_depth: Option<usize>,
}

impl ClipVersionHistory {
    /// Create a new, empty version history for the given clip.
    #[must_use]
    pub fn new(clip_id: String) -> Self {
        Self {
            clip_id,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            next_seq: 1,
            max_depth: None,
        }
    }

    /// Create a history with a maximum undo depth.
    ///
    /// Once the stack reaches `max_depth` entries the oldest entry is
    /// discarded when a new operation is pushed.
    #[must_use]
    pub fn with_max_depth(clip_id: String, max_depth: usize) -> Self {
        Self {
            clip_id,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            next_seq: 1,
            max_depth: Some(max_depth),
        }
    }

    /// Record a new edit operation.
    ///
    /// Clears the redo stack (branching history is not supported).
    pub fn push(&mut self, operation: ClipEditOperation) {
        self.push_labeled(operation, None);
    }

    /// Record a new edit operation with an optional descriptive label.
    pub fn push_labeled(&mut self, operation: ClipEditOperation, label: Option<String>) {
        let seq = self.next_seq;
        self.next_seq += 1;
        self.redo_stack.clear();
        self.undo_stack.push(VersionEntry::new(seq, operation, label));
        // Trim to max depth
        if let Some(max) = self.max_depth {
            if self.undo_stack.len() > max {
                let excess = self.undo_stack.len() - max;
                self.undo_stack.drain(..excess);
            }
        }
    }

    /// Undo the most recent operation.
    ///
    /// Returns the *inverse* operation (i.e., the operation the caller should
    /// apply to revert the clip to its previous state), or `None` if the
    /// undo stack is empty.
    pub fn undo(&mut self) -> Option<ClipEditOperation> {
        let entry = self.undo_stack.pop()?;
        let inverse = entry.operation.invert();
        let seq = self.next_seq;
        self.next_seq += 1;
        self.redo_stack.push(VersionEntry::new(seq, entry.operation, entry.label));
        Some(inverse)
    }

    /// Redo the most recently undone operation.
    ///
    /// Returns the operation that should be re-applied to the clip, or `None`
    /// if the redo stack is empty.
    pub fn redo(&mut self) -> Option<ClipEditOperation> {
        let entry = self.redo_stack.pop()?;
        let op = entry.operation.clone();
        let seq = self.next_seq;
        self.next_seq += 1;
        self.undo_stack.push(VersionEntry::new(seq, entry.operation, entry.label));
        Some(op)
    }

    /// Number of operations available on the undo stack.
    #[must_use]
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of operations available on the redo stack.
    #[must_use]
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }

    /// Whether undo is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Whether redo is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clear both undo and redo stacks.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// The full undo history in chronological order (oldest first).
    #[must_use]
    pub fn history(&self) -> &[VersionEntry] {
        &self.undo_stack
    }

    /// The most recent entry in the undo stack, if any.
    #[must_use]
    pub fn latest(&self) -> Option<&VersionEntry> {
        self.undo_stack.last()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-clip version store
// ─────────────────────────────────────────────────────────────────────────────

/// Manages version histories for multiple clips in a single session.
///
/// Keyed by clip ID string for efficient lookup. Histories are created
/// on demand when the first operation is recorded for a clip.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VersionStore {
    histories: HashMap<String, ClipVersionHistory>,
    /// Optional global undo depth limit applied to new histories.
    default_max_depth: Option<usize>,
}

impl VersionStore {
    /// Create a new, empty version store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a version store where every history is capped at `max_depth`.
    #[must_use]
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            histories: HashMap::new(),
            default_max_depth: Some(max_depth),
        }
    }

    /// Return (or create) the history for a clip.
    pub fn history_mut(&mut self, clip_id: &str) -> &mut ClipVersionHistory {
        self.histories
            .entry(clip_id.to_string())
            .or_insert_with(|| match self.default_max_depth {
                Some(d) => ClipVersionHistory::with_max_depth(clip_id.to_string(), d),
                None => ClipVersionHistory::new(clip_id.to_string()),
            })
    }

    /// Return the history for a clip (read-only), or `None` if not present.
    #[must_use]
    pub fn history(&self, clip_id: &str) -> Option<&ClipVersionHistory> {
        self.histories.get(clip_id)
    }

    /// Record an operation for a clip.
    pub fn push(&mut self, clip_id: &str, operation: ClipEditOperation) {
        self.history_mut(clip_id).push(operation);
    }

    /// Undo the latest operation for a clip.
    pub fn undo(&mut self, clip_id: &str) -> Option<ClipEditOperation> {
        self.history_mut(clip_id).undo()
    }

    /// Redo the latest undone operation for a clip.
    pub fn redo(&mut self, clip_id: &str) -> Option<ClipEditOperation> {
        self.history_mut(clip_id).redo()
    }

    /// Remove the history for a clip.
    pub fn remove(&mut self, clip_id: &str) {
        self.histories.remove(clip_id);
    }

    /// Total number of clips that have histories.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.histories.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_undo() {
        let mut h = ClipVersionHistory::new("clip-01".to_string());
        h.push(ClipEditOperation::NameChanged {
            before: "old".to_string(),
            after: "new".to_string(),
        });
        assert_eq!(h.undo_count(), 1);
        let inv = h.undo().expect("should undo");
        assert!(matches!(inv, ClipEditOperation::NameChanged { before, after } if before == "new" && after == "old"));
        assert_eq!(h.undo_count(), 0);
        assert_eq!(h.redo_count(), 1);
    }

    #[test]
    fn test_redo_after_undo() {
        let mut h = ClipVersionHistory::new("c1".to_string());
        h.push(ClipEditOperation::FavoriteToggled {
            before: false,
            after: true,
        });
        h.undo();
        let op = h.redo().expect("should redo");
        assert!(matches!(op, ClipEditOperation::FavoriteToggled { after: true, .. }));
        assert_eq!(h.undo_count(), 1);
        assert_eq!(h.redo_count(), 0);
    }

    #[test]
    fn test_push_clears_redo() {
        let mut h = ClipVersionHistory::new("c1".to_string());
        h.push(ClipEditOperation::KeywordAdded {
            keyword: "outdoor".to_string(),
        });
        h.undo();
        assert_eq!(h.redo_count(), 1);
        // New push discards redo
        h.push(ClipEditOperation::KeywordAdded {
            keyword: "indoor".to_string(),
        });
        assert_eq!(h.redo_count(), 0);
    }

    #[test]
    fn test_max_depth_trims_oldest() {
        let mut h = ClipVersionHistory::with_max_depth("c1".to_string(), 2);
        h.push(ClipEditOperation::RatingChanged { before: 0, after: 3 });
        h.push(ClipEditOperation::RatingChanged { before: 3, after: 4 });
        h.push(ClipEditOperation::RatingChanged { before: 4, after: 5 });
        // Only the last 2 entries should remain
        assert_eq!(h.undo_count(), 2);
    }

    #[test]
    fn test_invert_keyword_added() {
        let op = ClipEditOperation::KeywordAdded { keyword: "kw".to_string() };
        let inv = op.invert();
        assert!(matches!(inv, ClipEditOperation::KeywordRemoved { keyword } if keyword == "kw"));
    }

    #[test]
    fn test_version_store_multi_clip() {
        let mut store = VersionStore::new();
        store.push("clip-a", ClipEditOperation::NameChanged {
            before: "a".to_string(),
            after: "A".to_string(),
        });
        store.push("clip-b", ClipEditOperation::NameChanged {
            before: "b".to_string(),
            after: "B".to_string(),
        });
        assert_eq!(store.clip_count(), 2);
        let inv = store.undo("clip-a").expect("should undo clip-a");
        assert!(matches!(inv, ClipEditOperation::NameChanged { after, .. } if after == "a"));
    }

    #[test]
    fn test_empty_undo_returns_none() {
        let mut h = ClipVersionHistory::new("c1".to_string());
        assert!(h.undo().is_none());
        assert!(h.redo().is_none());
    }

    #[test]
    fn test_operation_description() {
        let op = ClipEditOperation::NameChanged {
            before: "x".to_string(),
            after: "y".to_string(),
        };
        let desc = op.description();
        assert!(desc.contains("x"));
        assert!(desc.contains("y"));
    }
}
