//! Operator action journal with undo/redo capability.
//!
//! This module records every operator action performed during a broadcast
//! session, maintaining a full history that supports:
//!
//! - **Audit trail**: Chronological record of who did what and when.
//! - **Undo/Redo**: Reverse or replay individual actions during production.
//! - **Session replay**: Re-examine decisions after broadcast for training.
//!
//! # Design
//!
//! Actions are stored in a `VecDeque` acting as a fixed-capacity ring buffer.
//! A cursor divides the buffer into "past" (undone) and "applied" (live)
//! sections, enabling standard linear undo/redo semantics.  Each action
//! carries an inverse description used when undoing.
//!
//! # Example
//!
//! ```rust
//! use oximedia_automation::operator_journal::{
//!     OperatorJournal, JournalEntry, OperatorActionKind,
//! };
//!
//! let mut journal = OperatorJournal::new(100);
//! journal.record(JournalEntry::new(
//!     "op_admin",
//!     OperatorActionKind::SwitchSource {
//!         from_source: "CAM1".to_string(),
//!         to_source: "CAM2".to_string(),
//!     },
//!     1_000_000,
//! ));
//!
//! assert_eq!(journal.applied_count(), 1);
//! let undone = journal.undo();
//! assert!(undone.is_some());
//! assert_eq!(journal.applied_count(), 0);
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// OperatorActionKind
// ---------------------------------------------------------------------------

/// Describes the kind of action an operator performed.
#[derive(Debug, Clone, PartialEq)]
pub enum OperatorActionKind {
    /// Switched the live video source.
    SwitchSource {
        /// Source that was active before the switch.
        from_source: String,
        /// Source that was switched to.
        to_source: String,
    },
    /// Started playout of a clip.
    PlayClip {
        /// Clip identifier.
        clip_id: String,
    },
    /// Stopped playout of the current clip.
    StopPlayout {
        /// Clip that was stopped.
        clip_id: String,
    },
    /// Manually triggered a cue point.
    ManualCue {
        /// Cue identifier.
        cue_id: String,
    },
    /// Changed audio level on a channel.
    AudioLevelChange {
        /// Channel identifier.
        channel: String,
        /// Previous level in dB.
        from_db: f64,
        /// New level in dB.
        to_db: f64,
    },
    /// Inserted a graphic overlay.
    GraphicInsert {
        /// Graphic template identifier.
        template_id: String,
    },
    /// Removed a graphic overlay.
    GraphicRemove {
        /// Graphic template identifier.
        template_id: String,
    },
    /// Approved a rundown item.
    ApproveItem {
        /// Item identifier.
        item_id: String,
    },
    /// Skipped a rundown item.
    SkipItem {
        /// Item identifier.
        item_id: String,
    },
    /// Overrode an interlock to allow a blocked action.
    InterlockOverride {
        /// Rule identifier that was overridden.
        rule_id: String,
        /// Description of the overridden action.
        action_description: String,
    },
    /// Triggered an emergency alert.
    EmergencyAlert {
        /// Alert type description.
        alert_type: String,
    },
    /// Custom operator action with an arbitrary description.
    Custom {
        /// Short label for the action.
        label: String,
        /// Additional details.
        details: String,
    },
}

impl OperatorActionKind {
    /// Returns a short label describing this action.
    pub fn label(&self) -> &str {
        match self {
            Self::SwitchSource { .. } => "SwitchSource",
            Self::PlayClip { .. } => "PlayClip",
            Self::StopPlayout { .. } => "StopPlayout",
            Self::ManualCue { .. } => "ManualCue",
            Self::AudioLevelChange { .. } => "AudioLevelChange",
            Self::GraphicInsert { .. } => "GraphicInsert",
            Self::GraphicRemove { .. } => "GraphicRemove",
            Self::ApproveItem { .. } => "ApproveItem",
            Self::SkipItem { .. } => "SkipItem",
            Self::InterlockOverride { .. } => "InterlockOverride",
            Self::EmergencyAlert { .. } => "EmergencyAlert",
            Self::Custom { label, .. } => label.as_str(),
        }
    }

    /// Returns a human-readable description of how to undo this action, or
    /// `None` if the action is not reversible.
    pub fn undo_description(&self) -> Option<String> {
        match self {
            Self::SwitchSource { from_source, to_source } => Some(format!(
                "Switch source back from {to_source} to {from_source}"
            )),
            Self::PlayClip { clip_id } => Some(format!("Stop clip {clip_id}")),
            Self::StopPlayout { clip_id } => Some(format!("Resume clip {clip_id}")),
            Self::GraphicInsert { template_id } => {
                Some(format!("Remove graphic {template_id}"))
            }
            Self::GraphicRemove { template_id } => {
                Some(format!("Re-insert graphic {template_id}"))
            }
            Self::AudioLevelChange { channel, from_db, .. } => {
                Some(format!("Restore audio level on {channel} to {from_db:.1} dB"))
            }
            Self::ApproveItem { item_id } => {
                Some(format!("Revert approval of item {item_id}"))
            }
            // These cannot be undone in a meaningful way:
            Self::ManualCue { .. } => None,
            Self::SkipItem { .. } => None,
            Self::InterlockOverride { .. } => None,
            Self::EmergencyAlert { .. } => None,
            Self::Custom { .. } => None,
        }
    }

    /// Returns `true` if this action can be undone.
    pub fn is_reversible(&self) -> bool {
        self.undo_description().is_some()
    }
}

// ---------------------------------------------------------------------------
// JournalEntry
// ---------------------------------------------------------------------------

/// A single entry in the operator action journal.
#[derive(Debug, Clone)]
pub struct JournalEntry {
    /// Operator login name or identifier.
    pub operator: String,
    /// The action that was performed.
    pub kind: OperatorActionKind,
    /// Timestamp in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// Optional comment provided by the operator.
    pub comment: Option<String>,
    /// Whether this entry has been undone (still in history but inactive).
    pub(crate) undone: bool,
}

impl JournalEntry {
    /// Create a new journal entry.
    pub fn new(
        operator: impl Into<String>,
        kind: OperatorActionKind,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            operator: operator.into(),
            kind,
            timestamp_ms,
            comment: None,
            undone: false,
        }
    }

    /// Builder: attach a comment.
    pub fn with_comment(mut self, comment: impl Into<String>) -> Self {
        self.comment = Some(comment.into());
        self
    }

    /// Whether this entry has been marked as undone.
    pub fn is_undone(&self) -> bool {
        self.undone
    }

    /// Whether this action can be undone.
    pub fn is_reversible(&self) -> bool {
        self.kind.is_reversible()
    }
}

// ---------------------------------------------------------------------------
// UndoRecord
// ---------------------------------------------------------------------------

/// Description of an undone action returned from [`OperatorJournal::undo`].
#[derive(Debug, Clone)]
pub struct UndoRecord {
    /// Operator who originally performed the action.
    pub operator: String,
    /// Label of the action that was undone.
    pub action_label: String,
    /// Human-readable description of what was undone.
    pub undo_description: String,
    /// Original timestamp of the action.
    pub original_timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// OperatorJournal
// ---------------------------------------------------------------------------

/// A bounded journal of operator actions with linear undo/redo.
///
/// The journal holds at most `capacity` entries.  When the capacity is
/// exceeded the oldest entries are discarded.  Undo/redo operate on a cursor
/// within the journal — undoing the most-recently-applied reversible action
/// and redoing the most-recently-undone action.
#[derive(Debug)]
pub struct OperatorJournal {
    /// All entries (both applied and undone).
    entries: VecDeque<JournalEntry>,
    /// Maximum number of entries to retain.
    capacity: usize,
    /// Index past the last *applied* entry.  Entries at `[cursor..]` are
    /// in the "future" (redo stack).
    cursor: usize,
}

impl OperatorJournal {
    /// Create a new journal with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity.min(4096)),
            capacity,
            cursor: 0,
        }
    }

    /// Record a new operator action.
    ///
    /// Recording a new action after undos discards the redo history (all
    /// entries above the cursor) to maintain a linear timeline.
    pub fn record(&mut self, entry: JournalEntry) {
        // Discard redo stack (entries above cursor).
        while self.entries.len() > self.cursor {
            self.entries.pop_back();
        }

        // Enforce capacity: evict oldest if needed.
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
            // Cursor shifts left because we removed from the front.
            self.cursor = self.cursor.saturating_sub(1);
        }

        self.entries.push_back(entry);
        self.cursor = self.entries.len();
    }

    /// Undo the most recently applied reversible action.
    ///
    /// Returns an [`UndoRecord`] describing what was undone, or `None` if
    /// there is nothing to undo (no reversible actions in the applied history).
    pub fn undo(&mut self) -> Option<UndoRecord> {
        // Walk backwards from cursor to find the last reversible, not-yet-undone entry.
        let mut idx = self.cursor;
        loop {
            if idx == 0 {
                return None;
            }
            idx -= 1;
            let entry = &self.entries[idx];
            if !entry.undone && entry.is_reversible() {
                break;
            }
        }

        let undo_desc = self.entries[idx].kind.undo_description()?;
        let entry = &mut self.entries[idx];
        entry.undone = true;

        let record = UndoRecord {
            operator: entry.operator.clone(),
            action_label: entry.kind.label().to_string(),
            undo_description: undo_desc,
            original_timestamp_ms: entry.timestamp_ms,
        };

        // Move cursor to the position of the undone entry so redo knows where
        // to pick up.  The cursor marks the boundary of applied entries.
        self.cursor = idx;

        Some(record)
    }

    /// Redo the most recently undone action.
    ///
    /// Returns the [`JournalEntry`] that was re-applied, or `None` if there
    /// is nothing to redo.
    pub fn redo(&mut self) -> Option<&JournalEntry> {
        // Walk forward from cursor to find the next undone entry.
        let mut idx = self.cursor;
        loop {
            if idx >= self.entries.len() {
                return None;
            }
            if self.entries[idx].undone {
                break;
            }
            idx += 1;
        }
        if idx >= self.entries.len() {
            return None;
        }

        self.entries[idx].undone = false;
        self.cursor = idx + 1;
        Some(&self.entries[idx])
    }

    /// Return the number of applied (non-undone) entries.
    pub fn applied_count(&self) -> usize {
        self.entries.iter().take(self.cursor).filter(|e| !e.undone).count()
    }

    /// Return the number of entries that can be undone.
    pub fn undo_depth(&self) -> usize {
        self.entries.iter().take(self.cursor).filter(|e| !e.undone && e.is_reversible()).count()
    }

    /// Return the number of entries that can be redone.
    pub fn redo_depth(&self) -> usize {
        self.entries.iter().skip(self.cursor).filter(|e| e.undone).count()
    }

    /// Total number of entries stored (applied + undone / redo stack).
    pub fn total_count(&self) -> usize {
        self.entries.len()
    }

    /// Return a slice of all entries in chronological order.
    pub fn entries(&self) -> &VecDeque<JournalEntry> {
        &self.entries
    }

    /// Return the N most recent applied entries (most-recent last).
    pub fn recent_applied(&self, n: usize) -> Vec<&JournalEntry> {
        let applied: Vec<&JournalEntry> = self
            .entries
            .iter()
            .take(self.cursor)
            .filter(|e| !e.undone)
            .collect();
        let skip = applied.len().saturating_sub(n);
        applied.into_iter().skip(skip).collect()
    }

    /// Return all entries by a specific operator.
    pub fn entries_by_operator(&self, operator: &str) -> Vec<&JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.operator == operator)
            .collect()
    }

    /// Return all entries of a specific action kind label.
    pub fn entries_by_label(&self, label: &str) -> Vec<&JournalEntry> {
        self.entries
            .iter()
            .filter(|e| e.kind.label() == label)
            .collect()
    }

    /// Clear all entries and reset the journal.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.cursor = 0;
    }

    /// Whether undo is available.
    pub fn can_undo(&self) -> bool {
        self.undo_depth() > 0
    }

    /// Whether redo is available.
    pub fn can_redo(&self) -> bool {
        self.redo_depth() > 0
    }
}

impl Default for OperatorJournal {
    fn default() -> Self {
        Self::new(1000)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn switch(from: &str, to: &str) -> OperatorActionKind {
        OperatorActionKind::SwitchSource {
            from_source: from.to_string(),
            to_source: to.to_string(),
        }
    }

    fn play(clip: &str) -> OperatorActionKind {
        OperatorActionKind::PlayClip { clip_id: clip.to_string() }
    }

    fn cue(id: &str) -> OperatorActionKind {
        OperatorActionKind::ManualCue { cue_id: id.to_string() }
    }

    fn entry(op: &str, kind: OperatorActionKind, ts: u64) -> JournalEntry {
        JournalEntry::new(op, kind, ts)
    }

    #[test]
    fn test_journal_record_and_applied_count() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("CAM1", "CAM2"), 1000));
        j.record(entry("op1", play("clip_a"), 2000));
        assert_eq!(j.applied_count(), 2);
        assert_eq!(j.total_count(), 2);
    }

    #[test]
    fn test_undo_returns_undo_record() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("admin", switch("CAM1", "CAM2"), 1000));
        let undone = j.undo();
        assert!(undone.is_some());
        let rec = undone.unwrap();
        assert_eq!(rec.action_label, "SwitchSource");
        assert_eq!(rec.operator, "admin");
        assert!(rec.undo_description.contains("CAM1"));
    }

    #[test]
    fn test_undo_decrements_applied_count() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("A", "B"), 1000));
        assert_eq!(j.applied_count(), 1);
        j.undo();
        assert_eq!(j.applied_count(), 0);
    }

    #[test]
    fn test_undo_irreversible_action_returns_none() {
        let mut j = OperatorJournal::new(50);
        // ManualCue is not reversible
        j.record(entry("op1", cue("cue-001"), 1000));
        let result = j.undo();
        assert!(result.is_none());
    }

    #[test]
    fn test_redo_after_undo() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("X", "Y"), 1000));
        j.undo();
        assert_eq!(j.applied_count(), 0);
        assert!(j.can_redo());
        let redone = j.redo();
        assert!(redone.is_some());
        assert_eq!(j.applied_count(), 1);
    }

    #[test]
    fn test_redo_depth_zero_when_no_undos() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("A", "B"), 1000));
        assert_eq!(j.redo_depth(), 0);
        assert!(!j.can_redo());
    }

    #[test]
    fn test_new_record_after_undo_clears_redo_stack() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("A", "B"), 1000));
        j.record(entry("op1", switch("B", "C"), 2000));
        j.undo(); // undo A->B switch (the second one)
        assert_eq!(j.redo_depth(), 1);

        // Now record a new action — should clear redo stack
        j.record(entry("op1", play("new_clip"), 3000));
        assert_eq!(j.redo_depth(), 0);
    }

    #[test]
    fn test_capacity_enforced() {
        let mut j = OperatorJournal::new(3);
        j.record(entry("op", switch("A", "B"), 1));
        j.record(entry("op", switch("B", "C"), 2));
        j.record(entry("op", switch("C", "D"), 3));
        // Adding a 4th should evict the oldest
        j.record(entry("op", switch("D", "E"), 4));
        assert!(j.total_count() <= 3);
    }

    #[test]
    fn test_undo_depth() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op1", switch("A", "B"), 1000)); // reversible
        j.record(entry("op1", cue("c1"), 2000));        // not reversible
        j.record(entry("op1", switch("B", "C"), 3000)); // reversible
        assert_eq!(j.undo_depth(), 2);
    }

    #[test]
    fn test_entries_by_operator() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("alice", switch("A", "B"), 1000));
        j.record(entry("bob", play("clip"), 2000));
        j.record(entry("alice", cue("cue1"), 3000));
        let alice_entries = j.entries_by_operator("alice");
        assert_eq!(alice_entries.len(), 2);
        assert!(alice_entries.iter().all(|e| e.operator == "alice"));
    }

    #[test]
    fn test_entries_by_label() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op", switch("A", "B"), 1));
        j.record(entry("op", switch("B", "C"), 2));
        j.record(entry("op", play("c"), 3));
        let switches = j.entries_by_label("SwitchSource");
        assert_eq!(switches.len(), 2);
    }

    #[test]
    fn test_recent_applied_returns_last_n() {
        let mut j = OperatorJournal::new(50);
        for i in 0..10u64 {
            j.record(entry("op", play(&format!("clip_{i}")), i * 100));
        }
        let recent = j.recent_applied(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_clear_resets_journal() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op", play("a"), 100));
        j.record(entry("op", play("b"), 200));
        j.clear();
        assert_eq!(j.total_count(), 0);
        assert_eq!(j.applied_count(), 0);
        assert!(!j.can_undo());
        assert!(!j.can_redo());
    }

    #[test]
    fn test_comment_on_entry() {
        let e = entry("op", play("clip"), 1000).with_comment("Emergency fill");
        assert_eq!(e.comment.as_deref(), Some("Emergency fill"));
    }

    #[test]
    fn test_action_kind_is_reversible() {
        assert!(switch("A", "B").is_reversible());
        assert!(play("x").is_reversible());
        assert!(!cue("c").is_reversible());
        assert!(!OperatorActionKind::EmergencyAlert { alert_type: "t".to_string() }.is_reversible());
    }

    #[test]
    fn test_audio_level_change_reversible() {
        let kind = OperatorActionKind::AudioLevelChange {
            channel: "CH1".to_string(),
            from_db: -20.0,
            to_db: -14.0,
        };
        assert!(kind.is_reversible());
        let desc = kind.undo_description().unwrap();
        assert!(desc.contains("CH1"));
        assert!(desc.contains("-20.0"));
    }

    #[test]
    fn test_multiple_undo_redo_cycle() {
        let mut j = OperatorJournal::new(50);
        j.record(entry("op", switch("A", "B"), 1));
        j.record(entry("op", switch("B", "C"), 2));

        // Undo both
        assert!(j.undo().is_some());
        assert!(j.undo().is_some());
        assert_eq!(j.applied_count(), 0);

        // Redo both
        assert!(j.redo().is_some());
        assert!(j.redo().is_some());
        assert_eq!(j.applied_count(), 2);

        // No more redo
        assert!(j.redo().is_none());
    }

    #[test]
    fn test_graphic_insert_remove_reversible() {
        let insert = OperatorActionKind::GraphicInsert { template_id: "lower_third".to_string() };
        let remove = OperatorActionKind::GraphicRemove { template_id: "lower_third".to_string() };
        assert!(insert.is_reversible());
        assert!(remove.is_reversible());
    }
}
