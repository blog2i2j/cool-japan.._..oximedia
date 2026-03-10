#![allow(dead_code)]
//! Track changes and diffs between two EDL versions.
//!
//! This module compares two EDL event lists and produces a structured
//! changelist describing additions, removals, and modifications.

use crate::event::EdlEvent;

/// The kind of change detected between two EDL versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// Event was added in the new version.
    Added,
    /// Event was removed from the old version.
    Removed,
    /// Event was modified (same event number, different content).
    Modified,
    /// Event number was reassigned (reel moved to a different position).
    Renumbered {
        /// Original event number.
        old_number: u32,
        /// New event number.
        new_number: u32,
    },
}

/// A single change entry in the changelist.
#[derive(Debug, Clone)]
pub struct ChangeEntry {
    /// The kind of change.
    pub kind: ChangeKind,
    /// Event number in the old EDL (if applicable).
    pub old_event_number: Option<u32>,
    /// Event number in the new EDL (if applicable).
    pub new_event_number: Option<u32>,
    /// Human-readable description of the change.
    pub description: String,
}

/// A complete changelist between two EDL versions.
#[derive(Debug, Clone)]
pub struct Changelist {
    /// Individual change entries.
    pub entries: Vec<ChangeEntry>,
}

impl Changelist {
    /// Number of changes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the changelist is empty (no differences).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count of added events.
    #[must_use]
    pub fn added_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind == ChangeKind::Added)
            .count()
    }

    /// Count of removed events.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind == ChangeKind::Removed)
            .count()
    }

    /// Count of modified events.
    #[must_use]
    pub fn modified_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.kind == ChangeKind::Modified)
            .count()
    }

    /// Format the changelist as a human-readable string.
    #[must_use]
    pub fn to_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "EDL Changelist: {} change(s)",
            self.entries.len()
        ));
        lines.push(format!(
            "  Added: {}, Removed: {}, Modified: {}",
            self.added_count(),
            self.removed_count(),
            self.modified_count()
        ));
        lines.push(String::new());
        for entry in &self.entries {
            let prefix = match &entry.kind {
                ChangeKind::Added => "+",
                ChangeKind::Removed => "-",
                ChangeKind::Modified => "~",
                ChangeKind::Renumbered { .. } => "#",
            };
            lines.push(format!("{prefix} {}", entry.description));
        }
        lines.join("\n")
    }
}

/// Compare two event lists and produce a changelist.
///
/// Events are matched by event number. If an event number exists in `old`
/// but not in `new`, it is marked as removed, and vice versa. If it exists
/// in both but differs, it is marked as modified.
#[must_use]
pub fn diff_events(old: &[EdlEvent], new: &[EdlEvent]) -> Changelist {
    let mut entries = Vec::new();

    // Build lookup maps by event number
    let old_map: std::collections::HashMap<u32, &EdlEvent> =
        old.iter().map(|e| (e.number, e)).collect();
    let new_map: std::collections::HashMap<u32, &EdlEvent> =
        new.iter().map(|e| (e.number, e)).collect();

    // Detect removed and modified
    for old_ev in old {
        match new_map.get(&old_ev.number) {
            None => {
                entries.push(ChangeEntry {
                    kind: ChangeKind::Removed,
                    old_event_number: Some(old_ev.number),
                    new_event_number: None,
                    description: format!(
                        "Event {} removed (reel: {})",
                        old_ev.number, old_ev.reel
                    ),
                });
            }
            Some(new_ev) => {
                if events_differ(old_ev, new_ev) {
                    let desc = describe_modification(old_ev, new_ev);
                    entries.push(ChangeEntry {
                        kind: ChangeKind::Modified,
                        old_event_number: Some(old_ev.number),
                        new_event_number: Some(new_ev.number),
                        description: desc,
                    });
                }
            }
        }
    }

    // Detect added
    for new_ev in new {
        if !old_map.contains_key(&new_ev.number) {
            entries.push(ChangeEntry {
                kind: ChangeKind::Added,
                old_event_number: None,
                new_event_number: Some(new_ev.number),
                description: format!(
                    "Event {} added (reel: {})",
                    new_ev.number, new_ev.reel
                ),
            });
        }
    }

    // Sort by event number for deterministic output
    entries.sort_by_key(|e| e.new_event_number.or(e.old_event_number).unwrap_or(0));

    Changelist { entries }
}

/// Check if two events differ in any meaningful way.
fn events_differ(a: &EdlEvent, b: &EdlEvent) -> bool {
    a.reel != b.reel
        || a.track != b.track
        || a.edit_type != b.edit_type
        || a.source_in != b.source_in
        || a.source_out != b.source_out
        || a.record_in != b.record_in
        || a.record_out != b.record_out
        || a.transition_duration != b.transition_duration
        || a.clip_name != b.clip_name
}

/// Build a human-readable description of what changed between two events.
fn describe_modification(old: &EdlEvent, new: &EdlEvent) -> String {
    let mut parts = Vec::new();
    if old.reel != new.reel {
        parts.push(format!("reel: {} -> {}", old.reel, new.reel));
    }
    if old.source_in != new.source_in || old.source_out != new.source_out {
        parts.push("source timecodes changed".to_string());
    }
    if old.record_in != new.record_in || old.record_out != new.record_out {
        parts.push("record timecodes changed".to_string());
    }
    if old.edit_type != new.edit_type {
        parts.push(format!("edit type: {} -> {}", old.edit_type, new.edit_type));
    }
    if old.clip_name != new.clip_name {
        parts.push("clip name changed".to_string());
    }
    let detail = if parts.is_empty() {
        "content changed".to_string()
    } else {
        parts.join(", ")
    };
    format!("Event {} modified: {detail}", old.number)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn make_tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("failed to create")
    }

    fn make_event(num: u32, reel: &str) -> EdlEvent {
        let tc1 = make_tc(1, 0, 0, 0);
        let tc2 = make_tc(1, 0, 5, 0);
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        )
    }

    #[test]
    fn test_diff_no_changes() {
        let events = vec![make_event(1, "A001"), make_event(2, "A002")];
        let cl = diff_events(&events, &events);
        assert!(cl.is_empty());
    }

    #[test]
    fn test_diff_added_event() {
        let old = vec![make_event(1, "A001")];
        let new = vec![make_event(1, "A001"), make_event(2, "A002")];
        let cl = diff_events(&old, &new);
        assert_eq!(cl.added_count(), 1);
        assert_eq!(cl.removed_count(), 0);
    }

    #[test]
    fn test_diff_removed_event() {
        let old = vec![make_event(1, "A001"), make_event(2, "A002")];
        let new = vec![make_event(1, "A001")];
        let cl = diff_events(&old, &new);
        assert_eq!(cl.removed_count(), 1);
        assert_eq!(cl.added_count(), 0);
    }

    #[test]
    fn test_diff_modified_event() {
        let old = vec![make_event(1, "A001")];
        let new = vec![make_event(1, "B001")];
        let cl = diff_events(&old, &new);
        assert_eq!(cl.modified_count(), 1);
    }

    #[test]
    fn test_diff_mixed_changes() {
        let old = vec![make_event(1, "A001"), make_event(2, "A002")];
        let new = vec![make_event(1, "B001"), make_event(3, "A003")];
        let cl = diff_events(&old, &new);
        assert_eq!(cl.modified_count(), 1); // event 1 changed
        assert_eq!(cl.removed_count(), 1); // event 2 removed
        assert_eq!(cl.added_count(), 1); // event 3 added
    }

    #[test]
    fn test_changelist_len() {
        let old = vec![make_event(1, "A001")];
        let new = vec![make_event(1, "A001"), make_event(2, "A002")];
        let cl = diff_events(&old, &new);
        assert_eq!(cl.len(), 1);
    }

    #[test]
    fn test_changelist_report() {
        let old = vec![make_event(1, "A001")];
        let new = vec![make_event(1, "B001")];
        let cl = diff_events(&old, &new);
        let report = cl.to_report();
        assert!(report.contains("1 change(s)"));
        assert!(report.contains("Modified: 1"));
    }

    #[test]
    fn test_events_differ_same() {
        let a = make_event(1, "A001");
        let b = make_event(1, "A001");
        assert!(!events_differ(&a, &b));
    }

    #[test]
    fn test_events_differ_reel() {
        let a = make_event(1, "A001");
        let b = make_event(1, "B001");
        assert!(events_differ(&a, &b));
    }

    #[test]
    fn test_describe_modification_reel() {
        let a = make_event(1, "A001");
        let b = make_event(1, "B001");
        let desc = describe_modification(&a, &b);
        assert!(desc.contains("reel: A001 -> B001"));
    }

    #[test]
    fn test_diff_empty_lists() {
        let cl = diff_events(&[], &[]);
        assert!(cl.is_empty());
    }

    #[test]
    fn test_change_kind_equality() {
        assert_eq!(ChangeKind::Added, ChangeKind::Added);
        assert_ne!(ChangeKind::Added, ChangeKind::Removed);
    }
}
