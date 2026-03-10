//! Marker editing operations for timeline annotation management.
//!
//! Extends the basic marker system with editing-specific operations such as
//! marker nudge, snap-to-marker, marker filtering, range selection from
//! markers, and batch marker manipulation.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Marker category
// ---------------------------------------------------------------------------

/// Category label for organising markers visually and semantically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarkerCategory {
    /// General-purpose marker.
    General,
    /// Comment / annotation marker.
    Comment,
    /// Chapter / section boundary.
    Chapter,
    /// Sync point for audio alignment.
    SyncPoint,
    /// Cue for playback automation.
    Cue,
    /// Error / issue flag.
    Error,
    /// Review note.
    Review,
    /// To-do item.
    Todo,
}

impl fmt::Display for MarkerCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => write!(f, "general"),
            Self::Comment => write!(f, "comment"),
            Self::Chapter => write!(f, "chapter"),
            Self::SyncPoint => write!(f, "sync"),
            Self::Cue => write!(f, "cue"),
            Self::Error => write!(f, "error"),
            Self::Review => write!(f, "review"),
            Self::Todo => write!(f, "todo"),
        }
    }
}

// ---------------------------------------------------------------------------
// Editable marker
// ---------------------------------------------------------------------------

/// A timeline marker with editing metadata.
#[derive(Debug, Clone)]
pub struct EditMarker {
    /// Unique ID within the timeline.
    pub id: u64,
    /// Position on the timeline (in timebase units).
    pub position: u64,
    /// Optional end position for range markers.
    pub end_position: Option<u64>,
    /// Category tag.
    pub category: MarkerCategory,
    /// Short label (shown on timeline).
    pub label: String,
    /// Extended description / notes.
    pub notes: String,
    /// RGBA colour for display.
    pub color: u32,
    /// Whether this marker is locked against edits.
    pub locked: bool,
}

impl EditMarker {
    /// Create a new point marker at `position`.
    pub fn new(id: u64, position: u64, label: impl Into<String>) -> Self {
        Self {
            id,
            position,
            end_position: None,
            category: MarkerCategory::General,
            label: label.into(),
            notes: String::new(),
            color: 0xFFFF00FF, // yellow
            locked: false,
        }
    }

    /// Create a range marker spanning `[start, end)`.
    pub fn range(id: u64, start: u64, end: u64, label: impl Into<String>) -> Self {
        Self {
            id,
            position: start,
            end_position: Some(end),
            category: MarkerCategory::General,
            label: label.into(),
            notes: String::new(),
            color: 0x00FF00FF,
            locked: false,
        }
    }

    /// Builder: set category.
    #[must_use]
    pub fn with_category(mut self, cat: MarkerCategory) -> Self {
        self.category = cat;
        self
    }

    /// Builder: set notes.
    pub fn with_notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = notes.into();
        self
    }

    /// Builder: set color.
    #[must_use]
    pub fn with_color(mut self, rgba: u32) -> Self {
        self.color = rgba;
        self
    }

    /// Returns the duration of a range marker, or 0 for point markers.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end_position
            .map_or(0, |end| end.saturating_sub(self.position))
    }

    /// Returns `true` if this is a range marker.
    #[must_use]
    pub fn is_range(&self) -> bool {
        self.end_position.is_some()
    }

    /// Returns `true` if the given position falls within this marker's range.
    /// For point markers, this checks exact equality.
    #[must_use]
    pub fn contains_position(&self, pos: u64) -> bool {
        match self.end_position {
            Some(end) => pos >= self.position && pos < end,
            None => pos == self.position,
        }
    }

    /// Nudge the marker by a signed offset. Clamps at zero.
    pub fn nudge(&mut self, offset: i64) {
        if self.locked {
            return;
        }
        let new_pos = (self.position as i64).saturating_add(offset).max(0) as u64;
        if let Some(ref mut end) = self.end_position {
            let delta = new_pos as i64 - self.position as i64;
            *end = (*end as i64).saturating_add(delta).max(0) as u64;
        }
        self.position = new_pos;
    }
}

// ---------------------------------------------------------------------------
// Snap helper
// ---------------------------------------------------------------------------

/// Result of a snap-to-marker operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapResult {
    /// The marker ID that was snapped to.
    pub marker_id: u64,
    /// The snapped position.
    pub position: u64,
    /// Distance from the original position to the snap target.
    pub distance: u64,
}

/// Find the nearest marker to `pos` within `threshold` (in timebase units).
/// Returns `None` if no marker is close enough.
#[must_use]
pub fn snap_to_nearest(markers: &[EditMarker], pos: u64, threshold: u64) -> Option<SnapResult> {
    let mut best: Option<SnapResult> = None;
    for m in markers {
        let dist = pos.abs_diff(m.position);
        if dist <= threshold && best.as_ref().map_or(true, |b| dist < b.distance) {
            best = Some(SnapResult {
                marker_id: m.id,
                position: m.position,
                distance: dist,
            });
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Marker editor
// ---------------------------------------------------------------------------

/// Manages a collection of editable markers on a timeline.
#[derive(Debug, Clone)]
pub struct MarkerEditor {
    /// All markers, keyed by ID.
    markers: HashMap<u64, EditMarker>,
    /// Auto-increment counter for marker IDs.
    next_id: u64,
}

impl MarkerEditor {
    /// Create a new, empty marker editor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            markers: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a point marker and return its ID.
    pub fn add_point(&mut self, position: u64, label: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.markers
            .insert(id, EditMarker::new(id, position, label));
        id
    }

    /// Add a range marker and return its ID.
    pub fn add_range(&mut self, start: u64, end: u64, label: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.markers
            .insert(id, EditMarker::range(id, start, end, label));
        id
    }

    /// Remove a marker by ID.
    pub fn remove(&mut self, id: u64) -> Option<EditMarker> {
        self.markers.remove(&id)
    }

    /// Get a marker by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&EditMarker> {
        self.markers.get(&id)
    }

    /// Get a mutable reference to a marker by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut EditMarker> {
        self.markers.get_mut(&id)
    }

    /// Return all markers sorted by position.
    #[must_use]
    pub fn sorted(&self) -> Vec<&EditMarker> {
        let mut v: Vec<&EditMarker> = self.markers.values().collect();
        v.sort_by_key(|m| m.position);
        v
    }

    /// Filter markers by category.
    #[must_use]
    pub fn filter_by_category(&self, cat: MarkerCategory) -> Vec<&EditMarker> {
        self.markers
            .values()
            .filter(|m| m.category == cat)
            .collect()
    }

    /// Nudge all unlocked markers by a signed offset.
    pub fn nudge_all(&mut self, offset: i64) {
        for marker in self.markers.values_mut() {
            marker.nudge(offset);
        }
    }

    /// Delete all markers that match a given category.
    pub fn delete_by_category(&mut self, cat: MarkerCategory) -> usize {
        let to_remove: Vec<u64> = self
            .markers
            .values()
            .filter(|m| m.category == cat)
            .map(|m| m.id)
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            self.markers.remove(&id);
        }
        count
    }

    /// Returns the total number of markers.
    #[must_use]
    pub fn count(&self) -> usize {
        self.markers.len()
    }

    /// Returns `true` if there are no markers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Clear all markers.
    pub fn clear(&mut self) {
        self.markers.clear();
    }

    /// Find all markers whose range contains the given position.
    #[must_use]
    pub fn markers_at(&self, pos: u64) -> Vec<&EditMarker> {
        self.markers
            .values()
            .filter(|m| m.contains_position(pos))
            .collect()
    }
}

impl Default for MarkerEditor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_category_display() {
        assert_eq!(MarkerCategory::Chapter.to_string(), "chapter");
        assert_eq!(MarkerCategory::Todo.to_string(), "todo");
    }

    #[test]
    fn test_edit_marker_point() {
        let m = EditMarker::new(1, 1000, "Take 1");
        assert_eq!(m.position, 1000);
        assert!(!m.is_range());
        assert_eq!(m.duration(), 0);
    }

    #[test]
    fn test_edit_marker_range() {
        let m = EditMarker::range(1, 100, 500, "Scene");
        assert!(m.is_range());
        assert_eq!(m.duration(), 400);
    }

    #[test]
    fn test_edit_marker_contains_position_point() {
        let m = EditMarker::new(1, 50, "x");
        assert!(m.contains_position(50));
        assert!(!m.contains_position(51));
    }

    #[test]
    fn test_edit_marker_contains_position_range() {
        let m = EditMarker::range(1, 100, 200, "r");
        assert!(m.contains_position(100));
        assert!(m.contains_position(199));
        assert!(!m.contains_position(200));
        assert!(!m.contains_position(99));
    }

    #[test]
    fn test_edit_marker_nudge() {
        let mut m = EditMarker::new(1, 100, "n");
        m.nudge(50);
        assert_eq!(m.position, 150);
    }

    #[test]
    fn test_edit_marker_nudge_negative_clamps() {
        let mut m = EditMarker::new(1, 10, "n");
        m.nudge(-100);
        assert_eq!(m.position, 0);
    }

    #[test]
    fn test_edit_marker_nudge_locked() {
        let mut m = EditMarker::new(1, 100, "locked");
        m.locked = true;
        m.nudge(50);
        assert_eq!(m.position, 100);
    }

    #[test]
    fn test_edit_marker_nudge_range() {
        let mut m = EditMarker::range(1, 100, 200, "r");
        m.nudge(50);
        assert_eq!(m.position, 150);
        assert_eq!(m.end_position, Some(250));
    }

    #[test]
    fn test_snap_to_nearest_found() {
        let markers = vec![EditMarker::new(1, 100, "a"), EditMarker::new(2, 200, "b")];
        let result = snap_to_nearest(&markers, 105, 10);
        assert!(result.is_some());
        assert_eq!(result.expect("test expectation failed").marker_id, 1);
        assert_eq!(result.expect("test expectation failed").distance, 5);
    }

    #[test]
    fn test_snap_to_nearest_not_found() {
        let markers = vec![EditMarker::new(1, 100, "a")];
        let result = snap_to_nearest(&markers, 200, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_marker_editor_add_and_get() {
        let mut ed = MarkerEditor::new();
        let id = ed.add_point(500, "Point");
        assert_eq!(ed.count(), 1);
        assert!(ed.get(id).is_some());
        assert_eq!(ed.get(id).expect("get should succeed").label, "Point");
    }

    #[test]
    fn test_marker_editor_remove() {
        let mut ed = MarkerEditor::new();
        let id = ed.add_point(100, "X");
        assert!(ed.remove(id).is_some());
        assert!(ed.is_empty());
    }

    #[test]
    fn test_marker_editor_sorted() {
        let mut ed = MarkerEditor::new();
        ed.add_point(300, "c");
        ed.add_point(100, "a");
        ed.add_point(200, "b");
        let sorted = ed.sorted();
        assert_eq!(sorted[0].position, 100);
        assert_eq!(sorted[1].position, 200);
        assert_eq!(sorted[2].position, 300);
    }

    #[test]
    fn test_marker_editor_filter_by_category() {
        let mut ed = MarkerEditor::new();
        let id1 = ed.add_point(100, "ch1");
        ed.get_mut(id1).expect("get_mut should succeed").category = MarkerCategory::Chapter;
        let _id2 = ed.add_point(200, "gen");
        let chapters = ed.filter_by_category(MarkerCategory::Chapter);
        assert_eq!(chapters.len(), 1);
    }

    #[test]
    fn test_marker_editor_nudge_all() {
        let mut ed = MarkerEditor::new();
        ed.add_point(100, "a");
        ed.add_point(200, "b");
        ed.nudge_all(50);
        let sorted = ed.sorted();
        assert_eq!(sorted[0].position, 150);
        assert_eq!(sorted[1].position, 250);
    }

    #[test]
    fn test_marker_editor_delete_by_category() {
        let mut ed = MarkerEditor::new();
        let id = ed.add_point(100, "err");
        ed.get_mut(id).expect("get_mut should succeed").category = MarkerCategory::Error;
        ed.add_point(200, "gen");
        let removed = ed.delete_by_category(MarkerCategory::Error);
        assert_eq!(removed, 1);
        assert_eq!(ed.count(), 1);
    }

    #[test]
    fn test_marker_editor_markers_at() {
        let mut ed = MarkerEditor::new();
        ed.add_range(100, 300, "range");
        ed.add_point(200, "point");
        let at_200 = ed.markers_at(200);
        assert_eq!(at_200.len(), 2);
    }

    #[test]
    fn test_marker_editor_clear() {
        let mut ed = MarkerEditor::new();
        ed.add_point(10, "x");
        ed.clear();
        assert!(ed.is_empty());
    }

    #[test]
    fn test_marker_editor_default() {
        let ed = MarkerEditor::default();
        assert!(ed.is_empty());
    }

    #[test]
    fn test_marker_builders() {
        let m = EditMarker::new(1, 0, "t")
            .with_category(MarkerCategory::Cue)
            .with_notes("hello")
            .with_color(0xFF0000FF);
        assert_eq!(m.category, MarkerCategory::Cue);
        assert_eq!(m.notes, "hello");
        assert_eq!(m.color, 0xFF0000FF);
    }
}
