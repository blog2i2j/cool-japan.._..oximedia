//! EDL change tracking and diff.
//!
//! This module compares two `Edl` structures and produces a list of
//! `EdlChange` variants describing what was Added, Removed, Modified,
//! or Moved between versions.

use crate::event::EdlEvent;
use crate::Edl;
use std::collections::HashMap;

/// A single change detected between two EDL versions.
#[derive(Debug, Clone, PartialEq)]
pub enum EdlChange {
    /// An event was added in the new EDL.
    Added {
        /// The event that was added.
        event: EdlEvent,
    },
    /// An event was removed from the old EDL.
    Removed {
        /// The event that was removed.
        event: EdlEvent,
    },
    /// An event was modified (same event number, different content).
    Modified {
        /// The old version of the event.
        old_event: EdlEvent,
        /// The new version of the event.
        new_event: EdlEvent,
        /// Descriptions of what fields changed.
        field_changes: Vec<FieldChange>,
    },
    /// An event was moved to a different position (same reel+timecodes, different number).
    Moved {
        /// The event in its old position.
        old_event: EdlEvent,
        /// The event in its new position.
        new_event: EdlEvent,
    },
}

impl EdlChange {
    /// Get a human-readable summary of this change.
    #[must_use]
    pub fn summary(&self) -> String {
        match self {
            Self::Added { event } => {
                format!(
                    "Added event {} (reel: {}, {})",
                    event.number, event.reel, event.edit_type
                )
            }
            Self::Removed { event } => {
                format!(
                    "Removed event {} (reel: {}, {})",
                    event.number, event.reel, event.edit_type
                )
            }
            Self::Modified {
                old_event,
                field_changes,
                ..
            } => {
                let changes: Vec<&str> = field_changes.iter().map(|c| c.field_name()).collect();
                format!(
                    "Modified event {}: {}",
                    old_event.number,
                    changes.join(", ")
                )
            }
            Self::Moved {
                old_event,
                new_event,
            } => {
                format!(
                    "Moved event {} -> {} (reel: {})",
                    old_event.number, new_event.number, old_event.reel
                )
            }
        }
    }
}

/// Description of a specific field that changed in a Modified event.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldChange {
    /// Reel name changed.
    Reel {
        /// Old reel name.
        old: String,
        /// New reel name.
        new: String,
    },
    /// Edit type changed.
    EditType {
        /// Old edit type description.
        old: String,
        /// New edit type description.
        new: String,
    },
    /// Track type changed.
    TrackType {
        /// Old track type description.
        old: String,
        /// New track type description.
        new: String,
    },
    /// Source in timecode changed.
    SourceIn {
        /// Old source-in frame count.
        old_frames: u64,
        /// New source-in frame count.
        new_frames: u64,
    },
    /// Source out timecode changed.
    SourceOut {
        /// Old source-out frame count.
        old_frames: u64,
        /// New source-out frame count.
        new_frames: u64,
    },
    /// Record in timecode changed.
    RecordIn {
        /// Old record-in frame count.
        old_frames: u64,
        /// New record-in frame count.
        new_frames: u64,
    },
    /// Record out timecode changed.
    RecordOut {
        /// Old record-out frame count.
        old_frames: u64,
        /// New record-out frame count.
        new_frames: u64,
    },
    /// Clip name changed.
    ClipName {
        /// Old clip name.
        old: Option<String>,
        /// New clip name.
        new: Option<String>,
    },
    /// Transition duration changed.
    TransitionDuration {
        /// Old duration.
        old: Option<u32>,
        /// New duration.
        new: Option<u32>,
    },
}

impl FieldChange {
    /// Get the field name for display.
    #[must_use]
    pub const fn field_name(&self) -> &'static str {
        match self {
            Self::Reel { .. } => "reel",
            Self::EditType { .. } => "edit_type",
            Self::TrackType { .. } => "track_type",
            Self::SourceIn { .. } => "source_in",
            Self::SourceOut { .. } => "source_out",
            Self::RecordIn { .. } => "record_in",
            Self::RecordOut { .. } => "record_out",
            Self::ClipName { .. } => "clip_name",
            Self::TransitionDuration { .. } => "transition_duration",
        }
    }
}

/// Result of diffing two EDLs.
#[derive(Debug, Clone)]
pub struct EdlDiff {
    /// Title of the old EDL.
    pub old_title: Option<String>,
    /// Title of the new EDL.
    pub new_title: Option<String>,
    /// List of changes detected.
    pub changes: Vec<EdlChange>,
}

impl EdlDiff {
    /// Number of total changes.
    #[must_use]
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Whether the two EDLs are identical (no changes).
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.changes.is_empty()
    }

    /// Count of added events.
    #[must_use]
    pub fn added_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, EdlChange::Added { .. }))
            .count()
    }

    /// Count of removed events.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, EdlChange::Removed { .. }))
            .count()
    }

    /// Count of modified events.
    #[must_use]
    pub fn modified_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, EdlChange::Modified { .. }))
            .count()
    }

    /// Count of moved events.
    #[must_use]
    pub fn moved_count(&self) -> usize {
        self.changes
            .iter()
            .filter(|c| matches!(c, EdlChange::Moved { .. }))
            .count()
    }

    /// Generate a human-readable report of all changes.
    #[must_use]
    pub fn to_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("EDL Diff: {} change(s)", self.changes.len()));
        lines.push(format!(
            "  Added: {}, Removed: {}, Modified: {}, Moved: {}",
            self.added_count(),
            self.removed_count(),
            self.modified_count(),
            self.moved_count(),
        ));

        if self.old_title != self.new_title {
            lines.push(format!(
                "  Title: {:?} -> {:?}",
                self.old_title, self.new_title
            ));
        }

        lines.push(String::new());
        for change in &self.changes {
            let prefix = match change {
                EdlChange::Added { .. } => "+",
                EdlChange::Removed { .. } => "-",
                EdlChange::Modified { .. } => "~",
                EdlChange::Moved { .. } => ">",
            };
            lines.push(format!("{prefix} {}", change.summary()));
        }
        lines.join("\n")
    }

    /// Get only the added events.
    #[must_use]
    pub fn added_events(&self) -> Vec<&EdlEvent> {
        self.changes
            .iter()
            .filter_map(|c| match c {
                EdlChange::Added { event } => Some(event),
                _ => None,
            })
            .collect()
    }

    /// Get only the removed events.
    #[must_use]
    pub fn removed_events(&self) -> Vec<&EdlEvent> {
        self.changes
            .iter()
            .filter_map(|c| match c {
                EdlChange::Removed { event } => Some(event),
                _ => None,
            })
            .collect()
    }
}

/// Compare two EDLs and produce a diff.
///
/// Events are matched primarily by event number. If an event number exists
/// in the old EDL but not in the new, we check if the same reel+timecodes
/// appear under a different number (detecting moves). Otherwise it is
/// classified as removed, and new event numbers are classified as added.
#[must_use]
pub fn diff_edls(old: &Edl, new: &Edl) -> EdlDiff {
    diff_event_lists(&old.events, &new.events, old.title.clone(), new.title.clone())
}

/// Compare two event lists and produce a diff.
#[must_use]
pub fn diff_event_lists(
    old: &[EdlEvent],
    new: &[EdlEvent],
    old_title: Option<String>,
    new_title: Option<String>,
) -> EdlDiff {
    let mut changes = Vec::new();

    let _old_by_num: HashMap<u32, &EdlEvent> = old.iter().map(|e| (e.number, e)).collect();
    let new_by_num: HashMap<u32, &EdlEvent> = new.iter().map(|e| (e.number, e)).collect();

    // Track which new events have been matched (to detect added events later)
    let mut matched_new: std::collections::HashSet<u32> = std::collections::HashSet::new();

    // Process old events
    for old_ev in old {
        if let Some(new_ev) = new_by_num.get(&old_ev.number) {
            // Same event number exists in both — check for modifications
            let field_changes = compute_field_changes(old_ev, new_ev);
            if !field_changes.is_empty() {
                changes.push(EdlChange::Modified {
                    old_event: old_ev.clone(),
                    new_event: (*new_ev).clone(),
                    field_changes,
                });
            }
            matched_new.insert(new_ev.number);
        } else {
            // Event number doesn't exist in new — check if it moved
            if let Some(moved_to) = find_moved_event(old_ev, new, &matched_new) {
                changes.push(EdlChange::Moved {
                    old_event: old_ev.clone(),
                    new_event: moved_to.clone(),
                });
                matched_new.insert(moved_to.number);
            } else {
                changes.push(EdlChange::Removed {
                    event: old_ev.clone(),
                });
            }
        }
    }

    // Any new events not yet matched are "Added"
    for new_ev in new {
        if !matched_new.contains(&new_ev.number) {
            changes.push(EdlChange::Added {
                event: new_ev.clone(),
            });
        }
    }

    // Sort changes by event number for deterministic output
    changes.sort_by_key(|c| match c {
        EdlChange::Added { event } => event.number,
        EdlChange::Removed { event } => event.number,
        EdlChange::Modified { old_event, .. } => old_event.number,
        EdlChange::Moved { old_event, .. } => old_event.number,
    });

    EdlDiff {
        old_title,
        new_title,
        changes,
    }
}

/// Check if an old event appears in the new list under a different number
/// (same reel, same source/record timecodes, same edit type).
fn find_moved_event<'a>(
    old_ev: &EdlEvent,
    new_events: &'a [EdlEvent],
    already_matched: &std::collections::HashSet<u32>,
) -> Option<&'a EdlEvent> {
    new_events.iter().find(|ne| {
        !already_matched.contains(&ne.number)
            && ne.number != old_ev.number
            && ne.reel == old_ev.reel
            && ne.source_in == old_ev.source_in
            && ne.source_out == old_ev.source_out
            && ne.record_in == old_ev.record_in
            && ne.record_out == old_ev.record_out
            && ne.edit_type == old_ev.edit_type
    })
}

/// Compute the list of field-level changes between two events.
fn compute_field_changes(old: &EdlEvent, new: &EdlEvent) -> Vec<FieldChange> {
    let mut changes = Vec::new();

    if old.reel != new.reel {
        changes.push(FieldChange::Reel {
            old: old.reel.clone(),
            new: new.reel.clone(),
        });
    }

    if old.edit_type != new.edit_type {
        changes.push(FieldChange::EditType {
            old: old.edit_type.to_string(),
            new: new.edit_type.to_string(),
        });
    }

    if old.track != new.track {
        changes.push(FieldChange::TrackType {
            old: old.track.to_string(),
            new: new.track.to_string(),
        });
    }

    if old.source_in != new.source_in {
        changes.push(FieldChange::SourceIn {
            old_frames: old.source_in.to_frames(),
            new_frames: new.source_in.to_frames(),
        });
    }

    if old.source_out != new.source_out {
        changes.push(FieldChange::SourceOut {
            old_frames: old.source_out.to_frames(),
            new_frames: new.source_out.to_frames(),
        });
    }

    if old.record_in != new.record_in {
        changes.push(FieldChange::RecordIn {
            old_frames: old.record_in.to_frames(),
            new_frames: new.record_in.to_frames(),
        });
    }

    if old.record_out != new.record_out {
        changes.push(FieldChange::RecordOut {
            old_frames: old.record_out.to_frames(),
            new_frames: new.record_out.to_frames(),
        });
    }

    if old.clip_name != new.clip_name {
        changes.push(FieldChange::ClipName {
            old: old.clip_name.clone(),
            new: new.clip_name.clone(),
        });
    }

    if old.transition_duration != new.transition_duration {
        changes.push(FieldChange::TransitionDuration {
            old: old.transition_duration,
            new: new.transition_duration,
        });
    }

    changes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::EdlFormat;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("valid timecode")
    }

    fn make_event(num: u32, reel: &str) -> EdlEvent {
        let tc1 = tc(1, 0, 0, 0);
        let tc2 = tc(1, 0, 5, 0);
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

    fn make_event_at(num: u32, reel: &str, s_in: u8, s_out: u8) -> EdlEvent {
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            tc(1, 0, s_in, 0),
            tc(1, 0, s_out, 0),
            tc(1, 0, s_in, 0),
            tc(1, 0, s_out, 0),
        )
    }

    fn make_edl(events: Vec<EdlEvent>) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        edl.events = events;
        edl
    }

    #[test]
    fn test_identical_edls() {
        let old = make_edl(vec![make_event(1, "A001"), make_event(2, "A002")]);
        let new = make_edl(vec![make_event(1, "A001"), make_event(2, "A002")]);
        let diff = diff_edls(&old, &new);
        assert!(diff.is_identical());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_added_event() {
        let old = make_edl(vec![make_event(1, "A001")]);
        let new = make_edl(vec![make_event(1, "A001"), make_event(2, "A002")]);
        let diff = diff_edls(&old, &new);
        assert_eq!(diff.added_count(), 1);
        assert_eq!(diff.removed_count(), 0);
        let added = diff.added_events();
        assert_eq!(added[0].number, 2);
        assert_eq!(added[0].reel, "A002");
    }

    #[test]
    fn test_removed_event() {
        let old = make_edl(vec![make_event(1, "A001"), make_event(2, "A002")]);
        let new = make_edl(vec![make_event(1, "A001")]);
        let diff = diff_edls(&old, &new);
        assert_eq!(diff.removed_count(), 1);
        assert_eq!(diff.added_count(), 0);
        let removed = diff.removed_events();
        assert_eq!(removed[0].number, 2);
    }

    #[test]
    fn test_modified_event_reel_change() {
        let old = make_edl(vec![make_event(1, "A001")]);
        let new = make_edl(vec![make_event(1, "B001")]);
        let diff = diff_edls(&old, &new);
        assert_eq!(diff.modified_count(), 1);
        if let EdlChange::Modified { field_changes, .. } = &diff.changes[0] {
            assert_eq!(field_changes.len(), 1);
            assert!(matches!(&field_changes[0], FieldChange::Reel { old, new } if old == "A001" && new == "B001"));
        } else {
            panic!("Expected Modified change");
        }
    }

    #[test]
    fn test_moved_event() {
        let old = make_edl(vec![
            make_event_at(1, "A001", 0, 5),
            make_event_at(2, "A002", 5, 10),
        ]);
        // Same events but renumbered
        let new = make_edl(vec![
            make_event_at(3, "A001", 0, 5),
            make_event_at(4, "A002", 5, 10),
        ]);
        let diff = diff_edls(&old, &new);
        assert_eq!(diff.moved_count(), 2);
    }

    #[test]
    fn test_mixed_changes() {
        let old = make_edl(vec![
            make_event(1, "A001"),
            make_event(2, "A002"),
            make_event(3, "A003"),
        ]);
        let new = make_edl(vec![
            make_event(1, "A001"), // unchanged
            make_event(2, "B002"), // modified
            make_event(4, "A004"), // added (3 removed)
        ]);
        let diff = diff_edls(&old, &new);
        assert_eq!(diff.modified_count(), 1);
        assert_eq!(diff.removed_count(), 1);
        assert_eq!(diff.added_count(), 1);
    }

    #[test]
    fn test_diff_report() {
        let old = make_edl(vec![make_event(1, "A001")]);
        let new = make_edl(vec![make_event(1, "B001")]);
        let diff = diff_edls(&old, &new);
        let report = diff.to_report();
        assert!(report.contains("1 change(s)"));
        assert!(report.contains("Modified"));
        assert!(report.contains("reel"));
    }

    #[test]
    fn test_empty_edls() {
        let old = make_edl(vec![]);
        let new = make_edl(vec![]);
        let diff = diff_edls(&old, &new);
        assert!(diff.is_identical());
    }

    #[test]
    fn test_field_change_names() {
        assert_eq!(
            FieldChange::Reel {
                old: "a".to_string(),
                new: "b".to_string()
            }
            .field_name(),
            "reel"
        );
        assert_eq!(
            FieldChange::SourceIn {
                old_frames: 0,
                new_frames: 1
            }
            .field_name(),
            "source_in"
        );
        assert_eq!(
            FieldChange::RecordOut {
                old_frames: 0,
                new_frames: 1
            }
            .field_name(),
            "record_out"
        );
    }

    #[test]
    fn test_change_summary_text() {
        let ev = make_event(1, "A001");
        let change = EdlChange::Added {
            event: ev.clone(),
        };
        let summary = change.summary();
        assert!(summary.contains("Added"));
        assert!(summary.contains("A001"));

        let removed = EdlChange::Removed { event: ev };
        assert!(removed.summary().contains("Removed"));
    }

    #[test]
    fn test_multiple_field_changes() {
        let mut old_ev = make_event(1, "A001");
        old_ev.set_clip_name("old_clip.mov".to_string());

        let mut new_ev = make_event(1, "B001");
        new_ev.set_clip_name("new_clip.mov".to_string());

        let changes = compute_field_changes(&old_ev, &new_ev);
        assert_eq!(changes.len(), 2); // reel + clip_name
        let field_names: Vec<&str> = changes.iter().map(|c| c.field_name()).collect();
        assert!(field_names.contains(&"reel"));
        assert!(field_names.contains(&"clip_name"));
    }

    #[test]
    fn test_diff_title_change_in_report() {
        let mut old = make_edl(vec![]);
        old.set_title("Old Title".to_string());
        let mut new = make_edl(vec![]);
        new.set_title("New Title".to_string());

        let diff = diff_edls(&old, &new);
        let report = diff.to_report();
        assert!(report.contains("Old Title"));
        assert!(report.contains("New Title"));
    }

    #[test]
    fn test_diff_event_lists_directly() {
        let old = vec![make_event(1, "A001")];
        let new = vec![make_event(1, "A001"), make_event(2, "A002")];
        let diff = diff_event_lists(&old, &new, None, None);
        assert_eq!(diff.added_count(), 1);
    }
}
