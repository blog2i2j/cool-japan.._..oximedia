#![allow(dead_code)]
//! EDL comparison utilities for detecting differences between two EDLs.
//!
//! This module provides structural comparison of EDL files, identifying
//! added, removed, and modified events between two versions.

use crate::event::EdlEvent;
use crate::Edl;
use std::collections::HashMap;
use std::fmt;

/// The type of change detected between two EDL versions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    /// An event was added in the new EDL.
    Added,
    /// An event was removed from the old EDL.
    Removed,
    /// An event was modified between versions.
    Modified,
    /// Source timecode was changed.
    SourceTimecodeChanged,
    /// Record timecode was changed.
    RecordTimecodeChanged,
    /// Reel assignment was changed.
    ReelChanged,
    /// Edit type was changed (e.g. Cut -> Dissolve).
    EditTypeChanged,
    /// Track type was changed.
    TrackChanged,
    /// Clip name was changed.
    ClipNameChanged,
}

impl fmt::Display for ChangeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Added => "ADDED",
            Self::Removed => "REMOVED",
            Self::Modified => "MODIFIED",
            Self::SourceTimecodeChanged => "SRC_TC_CHANGED",
            Self::RecordTimecodeChanged => "REC_TC_CHANGED",
            Self::ReelChanged => "REEL_CHANGED",
            Self::EditTypeChanged => "EDIT_TYPE_CHANGED",
            Self::TrackChanged => "TRACK_CHANGED",
            Self::ClipNameChanged => "CLIP_NAME_CHANGED",
        };
        write!(f, "{label}")
    }
}

/// A single difference detected between two EDLs.
#[derive(Debug, Clone)]
pub struct EdlDiff {
    /// Event number in the old EDL (if applicable).
    pub old_event_number: Option<u32>,
    /// Event number in the new EDL (if applicable).
    pub new_event_number: Option<u32>,
    /// The kind of change.
    pub kind: ChangeKind,
    /// Human-readable description of the change.
    pub description: String,
}

impl fmt::Display for EdlDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let old = self
            .old_event_number
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        let new = self
            .new_event_number
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        write!(
            f,
            "[{kind}] old={old} new={new}: {desc}",
            kind = self.kind,
            desc = self.description
        )
    }
}

/// Result of comparing two EDLs.
#[derive(Debug, Clone)]
pub struct CompareResult {
    /// List of differences found.
    pub diffs: Vec<EdlDiff>,
    /// Total events in the old EDL.
    pub old_event_count: usize,
    /// Total events in the new EDL.
    pub new_event_count: usize,
}

impl CompareResult {
    /// Returns `true` if the two EDLs are identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.diffs.is_empty()
    }

    /// Count of added events.
    #[must_use]
    pub fn added_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| d.kind == ChangeKind::Added)
            .count()
    }

    /// Count of removed events.
    #[must_use]
    pub fn removed_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| d.kind == ChangeKind::Removed)
            .count()
    }

    /// Count of modified events.
    #[must_use]
    pub fn modified_count(&self) -> usize {
        self.diffs
            .iter()
            .filter(|d| !matches!(d.kind, ChangeKind::Added | ChangeKind::Removed))
            .count()
    }

    /// Produce a human-readable summary string.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "added={}, removed={}, modified={}, old_events={}, new_events={}",
            self.added_count(),
            self.removed_count(),
            self.modified_count(),
            self.old_event_count,
            self.new_event_count,
        )
    }
}

/// Strategy used to match events between two EDLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchStrategy {
    /// Match events by their event number.
    ByEventNumber,
    /// Match events by record-in timecode.
    ByRecordIn,
    /// Match events by reel name and source-in timecode.
    ByReelAndSource,
}

/// Compare two EDLs and produce a list of differences.
///
/// # Arguments
///
/// * `old` - The baseline EDL.
/// * `new` - The updated EDL.
/// * `strategy` - How to match events between the two.
#[must_use]
pub fn compare_edls(old: &Edl, new: &Edl, strategy: MatchStrategy) -> CompareResult {
    let mut diffs = Vec::new();

    match strategy {
        MatchStrategy::ByEventNumber => {
            compare_by_event_number(old, new, &mut diffs);
        }
        MatchStrategy::ByRecordIn => {
            compare_by_record_in(old, new, &mut diffs);
        }
        MatchStrategy::ByReelAndSource => {
            compare_by_reel_and_source(old, new, &mut diffs);
        }
    }

    CompareResult {
        diffs,
        old_event_count: old.events.len(),
        new_event_count: new.events.len(),
    }
}

/// Compare events matched by event number.
fn compare_by_event_number(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let old_map: HashMap<u32, &EdlEvent> = old.events.iter().map(|e| (e.number, e)).collect();
    let new_map: HashMap<u32, &EdlEvent> = new.events.iter().map(|e| (e.number, e)).collect();

    // Check for removed / modified
    for (&num, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(&num) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(num),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!("Event {num} removed"),
            });
        }
    }

    // Check for added
    for &num in new_map.keys() {
        if !old_map.contains_key(&num) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(num),
                kind: ChangeKind::Added,
                description: format!("Event {num} added"),
            });
        }
    }
}

/// Compare events matched by record-in timecode.
fn compare_by_record_in(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let old_map: HashMap<u64, &EdlEvent> = old
        .events
        .iter()
        .map(|e| (e.record_in.to_frames(), e))
        .collect();
    let new_map: HashMap<u64, &EdlEvent> = new
        .events
        .iter()
        .map(|e| (e.record_in.to_frames(), e))
        .collect();

    for (&tc, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(&tc) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(old_evt.number),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!("Event {} at record_in frame {} removed", old_evt.number, tc),
            });
        }
    }

    for (&tc, &new_evt) in &new_map {
        if !old_map.contains_key(&tc) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(new_evt.number),
                kind: ChangeKind::Added,
                description: format!("Event {} at record_in frame {} added", new_evt.number, tc),
            });
        }
    }
}

/// Compare events matched by reel name + source-in.
fn compare_by_reel_and_source(old: &Edl, new: &Edl, diffs: &mut Vec<EdlDiff>) {
    let key_fn = |e: &EdlEvent| -> (String, u64) { (e.reel.clone(), e.source_in.to_frames()) };

    let old_map: HashMap<(String, u64), &EdlEvent> =
        old.events.iter().map(|e| (key_fn(e), e)).collect();
    let new_map: HashMap<(String, u64), &EdlEvent> =
        new.events.iter().map(|e| (key_fn(e), e)).collect();

    for (key, &old_evt) in &old_map {
        if let Some(&new_evt) = new_map.get(key) {
            diff_events(old_evt, new_evt, diffs);
        } else {
            diffs.push(EdlDiff {
                old_event_number: Some(old_evt.number),
                new_event_number: None,
                kind: ChangeKind::Removed,
                description: format!(
                    "Event {} (reel={}, src_in frame={}) removed",
                    old_evt.number, key.0, key.1
                ),
            });
        }
    }

    for (key, &new_evt) in &new_map {
        if !old_map.contains_key(key) {
            diffs.push(EdlDiff {
                old_event_number: None,
                new_event_number: Some(new_evt.number),
                kind: ChangeKind::Added,
                description: format!(
                    "Event {} (reel={}, src_in frame={}) added",
                    new_evt.number, key.0, key.1
                ),
            });
        }
    }
}

/// Produce field-level diffs between two matched events.
fn diff_events(old: &EdlEvent, new: &EdlEvent, diffs: &mut Vec<EdlDiff>) {
    if old.source_in != new.source_in || old.source_out != new.source_out {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::SourceTimecodeChanged,
            description: format!(
                "Source TC changed: {}-{} -> {}-{}",
                old.source_in, old.source_out, new.source_in, new.source_out
            ),
        });
    }

    if old.record_in != new.record_in || old.record_out != new.record_out {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::RecordTimecodeChanged,
            description: format!(
                "Record TC changed: {}-{} -> {}-{}",
                old.record_in, old.record_out, new.record_in, new.record_out
            ),
        });
    }

    if old.reel != new.reel {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::ReelChanged,
            description: format!("Reel changed: {} -> {}", old.reel, new.reel),
        });
    }

    if old.edit_type != new.edit_type {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::EditTypeChanged,
            description: format!("Edit type changed: {} -> {}", old.edit_type, new.edit_type),
        });
    }

    if old.track != new.track {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::TrackChanged,
            description: format!("Track changed: {} -> {}", old.track, new.track),
        });
    }

    if old.clip_name != new.clip_name {
        diffs.push(EdlDiff {
            old_event_number: Some(old.number),
            new_event_number: Some(new.number),
            kind: ChangeKind::ClipNameChanged,
            description: format!(
                "Clip name changed: {:?} -> {:?}",
                old.clip_name, new.clip_name
            ),
        });
    }
}

/// Compute a numeric similarity score (0.0 .. 1.0) between two EDLs.
///
/// The score considers event count, matching events, and timecode proximity.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn similarity_score(old: &Edl, new: &Edl) -> f64 {
    if old.events.is_empty() && new.events.is_empty() {
        return 1.0;
    }
    let total = (old.events.len() + new.events.len()) as f64;
    if total == 0.0 {
        return 1.0;
    }

    let result = compare_edls(old, new, MatchStrategy::ByEventNumber);
    let diff_count = result.diffs.len() as f64;

    // Clamp to [0, 1]
    (1.0 - diff_count / total).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::{Edl, EdlFormat};

    fn make_event(num: u32, reel: &str, src_in_sec: u8, src_out_sec: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, src_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_out_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, src_out_sec, 0, fr).expect("failed to create"),
        )
    }

    fn make_edl(events: Vec<EdlEvent>) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);
        for e in events {
            edl.events.push(e);
        }
        edl
    }

    #[test]
    fn test_identical_edls() {
        let e1 = make_event(1, "A001", 0, 5);
        let edl_a = make_edl(vec![e1.clone()]);
        let edl_b = make_edl(vec![e1]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result.is_identical());
    }

    #[test]
    fn test_added_event() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert_eq!(result.added_count(), 1);
        assert_eq!(result.removed_count(), 0);
    }

    #[test]
    fn test_removed_event() {
        let edl_a = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert_eq!(result.removed_count(), 1);
    }

    #[test]
    fn test_reel_changed() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "B001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::ReelChanged));
    }

    #[test]
    fn test_edit_type_changed() {
        let mut evt = make_event(1, "A001", 0, 5);
        let edl_a = make_edl(vec![evt.clone()]);
        evt.edit_type = EditType::Dissolve;
        evt.transition_duration = Some(30);
        let edl_b = make_edl(vec![evt]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::EditTypeChanged));
    }

    #[test]
    fn test_source_timecode_changed() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 1, 6)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::SourceTimecodeChanged));
    }

    #[test]
    fn test_compare_by_record_in() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByRecordIn);
        assert!(result.is_identical());
    }

    #[test]
    fn test_compare_by_reel_and_source() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByReelAndSource);
        assert!(result.is_identical());
    }

    #[test]
    fn test_compare_both_empty() {
        let edl_a = make_edl(vec![]);
        let edl_b = make_edl(vec![]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result.is_identical());
    }

    #[test]
    fn test_summary_string() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        let s = result.summary();
        assert!(s.contains("added=1"));
    }

    #[test]
    fn test_similarity_identical() {
        let edl_a = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let edl_b = make_edl(vec![make_event(1, "A001", 0, 5)]);
        let score = similarity_score(&edl_a, &edl_b);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_similarity_empty() {
        let edl_a = make_edl(vec![]);
        let edl_b = make_edl(vec![]);
        let score = similarity_score(&edl_a, &edl_b);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_change_kind_display() {
        assert_eq!(ChangeKind::Added.to_string(), "ADDED");
        assert_eq!(ChangeKind::Removed.to_string(), "REMOVED");
        assert_eq!(ChangeKind::Modified.to_string(), "MODIFIED");
    }

    #[test]
    fn test_diff_display() {
        let diff = EdlDiff {
            old_event_number: Some(1),
            new_event_number: None,
            kind: ChangeKind::Removed,
            description: "Event 1 removed".to_string(),
        };
        let s = diff.to_string();
        assert!(s.contains("REMOVED"));
        assert!(s.contains("old=1"));
    }

    #[test]
    fn test_clip_name_changed() {
        let mut evt_a = make_event(1, "A001", 0, 5);
        evt_a.clip_name = Some("clip_v1.mov".to_string());
        let mut evt_b = make_event(1, "A001", 0, 5);
        evt_b.clip_name = Some("clip_v2.mov".to_string());
        let edl_a = make_edl(vec![evt_a]);
        let edl_b = make_edl(vec![evt_b]);
        let result = compare_edls(&edl_a, &edl_b, MatchStrategy::ByEventNumber);
        assert!(result
            .diffs
            .iter()
            .any(|d| d.kind == ChangeKind::ClipNameChanged));
    }
}
