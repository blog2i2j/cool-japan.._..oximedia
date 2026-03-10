#![allow(dead_code)]
//! Clip edit history and undo/redo support.
//!
//! This module provides a non-destructive edit history system for clips,
//! tracking all changes (trims, metadata edits, rating changes, etc.)
//! with full undo/redo support. Each change is recorded as an action
//! in a linear history stack.

use std::collections::VecDeque;
use std::fmt;

/// Type of edit action performed on a clip.
#[derive(Debug, Clone, PartialEq)]
pub enum EditAction {
    /// Trim in-point changed.
    TrimIn {
        /// Previous in-point in frames.
        old_frame: u64,
        /// New in-point in frames.
        new_frame: u64,
    },
    /// Trim out-point changed.
    TrimOut {
        /// Previous out-point in frames.
        old_frame: u64,
        /// New out-point in frames.
        new_frame: u64,
    },
    /// Rating changed.
    RatingChange {
        /// Previous rating.
        old_rating: u8,
        /// New rating.
        new_rating: u8,
    },
    /// Name / label changed.
    Rename {
        /// Previous name.
        old_name: String,
        /// New name.
        new_name: String,
    },
    /// Keyword added.
    KeywordAdd {
        /// The keyword that was added.
        keyword: String,
    },
    /// Keyword removed.
    KeywordRemove {
        /// The keyword that was removed.
        keyword: String,
    },
    /// Marker added at a frame position.
    MarkerAdd {
        /// Frame position of the marker.
        frame: u64,
        /// Marker label.
        label: String,
    },
    /// Marker removed.
    MarkerRemove {
        /// Frame position of the marker.
        frame: u64,
        /// Marker label.
        label: String,
    },
    /// Note / comment changed.
    NoteChange {
        /// Previous note text.
        old_text: String,
        /// New note text.
        new_text: String,
    },
    /// Color label changed.
    ColorLabel {
        /// Previous color label.
        old_color: String,
        /// New color label.
        new_color: String,
    },
}

impl fmt::Display for EditAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrimIn { new_frame, .. } => write!(f, "Set In Point: {new_frame}"),
            Self::TrimOut { new_frame, .. } => write!(f, "Set Out Point: {new_frame}"),
            Self::RatingChange { new_rating, .. } => write!(f, "Rate: {new_rating} stars"),
            Self::Rename { new_name, .. } => write!(f, "Rename: {new_name}"),
            Self::KeywordAdd { keyword } => write!(f, "Add Keyword: {keyword}"),
            Self::KeywordRemove { keyword } => write!(f, "Remove Keyword: {keyword}"),
            Self::MarkerAdd { frame, label } => {
                write!(f, "Add Marker at {frame}: {label}")
            }
            Self::MarkerRemove { frame, label } => {
                write!(f, "Remove Marker at {frame}: {label}")
            }
            Self::NoteChange { .. } => write!(f, "Edit Note"),
            Self::ColorLabel { new_color, .. } => write!(f, "Color: {new_color}"),
        }
    }
}

/// A recorded entry in the edit history.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// The action performed.
    pub action: EditAction,
    /// Clip ID affected.
    pub clip_id: u64,
    /// Timestamp of the action (seconds since epoch).
    pub timestamp: u64,
    /// User or session that performed the action.
    pub user: String,
}

impl HistoryEntry {
    /// Creates a new history entry.
    #[must_use]
    pub fn new(action: EditAction, clip_id: u64, timestamp: u64, user: impl Into<String>) -> Self {
        Self {
            action,
            clip_id,
            timestamp,
            user: user.into(),
        }
    }

    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        format!("[{}] Clip {}: {}", self.user, self.clip_id, self.action)
    }
}

/// Edit history with undo/redo support.
#[derive(Debug)]
pub struct ClipEditHistory {
    /// Stack of performed actions (undo stack).
    undo_stack: Vec<HistoryEntry>,
    /// Stack of undone actions (redo stack).
    redo_stack: Vec<HistoryEntry>,
    /// Maximum history depth.
    max_depth: usize,
    /// Whether history recording is active.
    recording: bool,
}

impl ClipEditHistory {
    /// Creates a new empty edit history.
    #[must_use]
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_depth,
            recording: true,
        }
    }

    /// Creates a history with default max depth (1000).
    #[must_use]
    pub fn with_default_depth() -> Self {
        Self::new(1000)
    }

    /// Records a new action. Clears the redo stack.
    pub fn record(&mut self, entry: HistoryEntry) {
        if !self.recording {
            return;
        }
        self.redo_stack.clear();
        self.undo_stack.push(entry);
        // Trim to max depth
        while self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
    }

    /// Returns the most recent action without undoing it.
    #[must_use]
    pub fn peek_undo(&self) -> Option<&HistoryEntry> {
        self.undo_stack.last()
    }

    /// Undoes the most recent action and returns it.
    pub fn undo(&mut self) -> Option<HistoryEntry> {
        if let Some(entry) = self.undo_stack.pop() {
            self.redo_stack.push(entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    /// Redoes the most recently undone action and returns it.
    pub fn redo(&mut self) -> Option<HistoryEntry> {
        if let Some(entry) = self.redo_stack.pop() {
            self.undo_stack.push(entry.clone());
            Some(entry)
        } else {
            None
        }
    }

    /// Returns true if undo is available.
    #[must_use]
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns true if redo is available.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Returns the undo stack depth.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// Returns the redo stack depth.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }

    /// Clears all history.
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// Pauses history recording.
    pub fn pause(&mut self) {
        self.recording = false;
    }

    /// Resumes history recording.
    pub fn resume(&mut self) {
        self.recording = true;
    }

    /// Returns whether recording is active.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Returns all entries in the undo stack (oldest first).
    #[must_use]
    pub fn undo_entries(&self) -> &[HistoryEntry] {
        &self.undo_stack
    }

    /// Returns the history for a specific clip.
    #[must_use]
    pub fn clip_history(&self, clip_id: u64) -> Vec<&HistoryEntry> {
        self.undo_stack
            .iter()
            .filter(|e| e.clip_id == clip_id)
            .collect()
    }
}

/// A batch of edits grouped as a single undoable unit.
#[derive(Debug, Clone)]
pub struct EditBatch {
    /// Label for this batch.
    pub label: String,
    /// Actions in this batch.
    pub actions: Vec<HistoryEntry>,
}

impl EditBatch {
    /// Creates a new edit batch.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            actions: Vec::new(),
        }
    }

    /// Adds an action to the batch.
    pub fn add(&mut self, entry: HistoryEntry) {
        self.actions.push(entry);
    }

    /// Returns the number of actions in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Returns true if the batch is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// History log that persists a chronological record of all edits.
#[derive(Debug)]
pub struct HistoryLog {
    /// All logged entries.
    entries: VecDeque<HistoryEntry>,
    /// Maximum number of entries to keep.
    capacity: usize,
}

impl HistoryLog {
    /// Creates a new history log with the given capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
        }
    }

    /// Appends an entry to the log.
    pub fn append(&mut self, entry: HistoryEntry) {
        self.entries.push_back(entry);
        while self.entries.len() > self.capacity {
            self.entries.pop_front();
        }
    }

    /// Returns the number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the log is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries (oldest first).
    #[must_use]
    pub fn entries(&self) -> &VecDeque<HistoryEntry> {
        &self.entries
    }

    /// Returns entries for a specific clip.
    #[must_use]
    pub fn entries_for_clip(&self, clip_id: u64) -> Vec<&HistoryEntry> {
        self.entries
            .iter()
            .filter(|e| e.clip_id == clip_id)
            .collect()
    }

    /// Clears the log.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(action: EditAction, clip_id: u64) -> HistoryEntry {
        HistoryEntry::new(action, clip_id, 1000, "editor")
    }

    #[test]
    fn test_edit_action_display() {
        let action = EditAction::TrimIn {
            old_frame: 0,
            new_frame: 100,
        };
        let display = format!("{action}");
        assert!(display.contains("100"));
    }

    #[test]
    fn test_history_entry_description() {
        let entry = make_entry(
            EditAction::Rename {
                old_name: "Old".into(),
                new_name: "New".into(),
            },
            42,
        );
        let desc = entry.description();
        assert!(desc.contains("42"));
        assert!(desc.contains("editor"));
    }

    #[test]
    fn test_history_record_and_undo() {
        let mut history = ClipEditHistory::new(100);
        let entry = make_entry(
            EditAction::RatingChange {
                old_rating: 0,
                new_rating: 5,
            },
            1,
        );
        history.record(entry);
        assert!(history.can_undo());
        assert!(!history.can_redo());
        assert_eq!(history.undo_depth(), 1);

        let undone = history.undo().expect("undo should succeed");
        assert_eq!(undone.clip_id, 1);
        assert!(!history.can_undo());
        assert!(history.can_redo());
    }

    #[test]
    fn test_history_redo() {
        let mut history = ClipEditHistory::new(100);
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "test".into(),
            },
            1,
        ));
        history.undo();
        assert!(history.can_redo());

        let redone = history.redo().expect("redo should succeed");
        assert_eq!(redone.clip_id, 1);
        assert!(history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn test_new_action_clears_redo() {
        let mut history = ClipEditHistory::new(100);
        history.record(make_entry(
            EditAction::TrimIn {
                old_frame: 0,
                new_frame: 10,
            },
            1,
        ));
        history.undo();
        assert!(history.can_redo());

        // New action should clear redo
        history.record(make_entry(
            EditAction::TrimOut {
                old_frame: 100,
                new_frame: 90,
            },
            1,
        ));
        assert!(!history.can_redo());
    }

    #[test]
    fn test_max_depth() {
        let mut history = ClipEditHistory::new(3);
        for i in 0..5 {
            history.record(make_entry(
                EditAction::RatingChange {
                    old_rating: 0,
                    new_rating: i as u8,
                },
                1,
            ));
        }
        assert_eq!(history.undo_depth(), 3);
    }

    #[test]
    fn test_clear_history() {
        let mut history = ClipEditHistory::new(100);
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "a".into(),
            },
            1,
        ));
        history.undo();
        history.clear();
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn test_pause_resume() {
        let mut history = ClipEditHistory::new(100);
        history.pause();
        assert!(!history.is_recording());
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "x".into(),
            },
            1,
        ));
        assert_eq!(history.undo_depth(), 0); // not recorded

        history.resume();
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "y".into(),
            },
            1,
        ));
        assert_eq!(history.undo_depth(), 1);
    }

    #[test]
    fn test_clip_history_filter() {
        let mut history = ClipEditHistory::new(100);
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "a".into(),
            },
            1,
        ));
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "b".into(),
            },
            2,
        ));
        history.record(make_entry(
            EditAction::KeywordAdd {
                keyword: "c".into(),
            },
            1,
        ));
        let clip1_hist = history.clip_history(1);
        assert_eq!(clip1_hist.len(), 2);
    }

    #[test]
    fn test_edit_batch() {
        let mut batch = EditBatch::new("Batch Rating");
        assert!(batch.is_empty());
        batch.add(make_entry(
            EditAction::RatingChange {
                old_rating: 0,
                new_rating: 5,
            },
            1,
        ));
        batch.add(make_entry(
            EditAction::RatingChange {
                old_rating: 0,
                new_rating: 4,
            },
            2,
        ));
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_history_log() {
        let mut log = HistoryLog::new(5);
        assert!(log.is_empty());
        for i in 0..7 {
            log.append(make_entry(
                EditAction::RatingChange {
                    old_rating: 0,
                    new_rating: i as u8,
                },
                i,
            ));
        }
        assert_eq!(log.len(), 5); // capped at capacity
    }

    #[test]
    fn test_history_log_entries_for_clip() {
        let mut log = HistoryLog::new(100);
        log.append(make_entry(
            EditAction::KeywordAdd {
                keyword: "a".into(),
            },
            10,
        ));
        log.append(make_entry(
            EditAction::KeywordAdd {
                keyword: "b".into(),
            },
            20,
        ));
        log.append(make_entry(
            EditAction::KeywordAdd {
                keyword: "c".into(),
            },
            10,
        ));
        let entries = log.entries_for_clip(10);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_peek_undo() {
        let mut history = ClipEditHistory::new(100);
        assert!(history.peek_undo().is_none());
        history.record(make_entry(
            EditAction::Rename {
                old_name: "A".into(),
                new_name: "B".into(),
            },
            1,
        ));
        let peeked = history.peek_undo().expect("peek_undo should succeed");
        assert_eq!(peeked.clip_id, 1);
        // peek should not consume
        assert_eq!(history.undo_depth(), 1);
    }
}
