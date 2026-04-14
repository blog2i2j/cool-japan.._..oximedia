//! Timeline markers, cue points, and chapter markers for media assets.
//!
//! Provides a rich set of marker types that can be attached to a media asset's
//! timeline:
//!
//! * `MarkerKind` – enumerated type for the class of marker.
//! * `MarkerColor` – visual colour hint for UI rendering.
//! * `TimelineMarker` – a single point or range marker on the timeline.
//! * `ChapterMarker` – a named chapter with optional thumbnail and description.
//! * `CuePoint` – a named timecode cue used for playback control or live events.
//! * `MarkerSet` – a sorted collection of markers attached to an asset.
//! * `MarkerRegistry` – maps asset ids to their `MarkerSet`.
//!
//! All timecodes are stored as millisecond offsets from the media start.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MarkerKind
// ---------------------------------------------------------------------------

/// The class of a timeline marker.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerKind {
    /// A simple in/out range for editing.
    InOut,
    /// A chapter boundary.
    Chapter,
    /// A named cue point (for playback automation or live events).
    Cue,
    /// A comment or annotation anchored to a timecode.
    Comment,
    /// A quality issue or defect flag.
    Defect,
    /// A content advisory marker (e.g. "violence" or "explicit").
    ContentAdvisory,
    /// A legal or compliance marker (e.g. "restricted territory window").
    Legal,
    /// A highlight or "best moment" selection.
    Highlight,
    /// A synchronisation marker (used for multi-camera alignment).
    Sync,
    /// An ad-break / commercial insertion point.
    AdBreak,
    /// Custom marker type.
    Custom(String),
}

impl MarkerKind {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::InOut => "In/Out",
            Self::Chapter => "Chapter",
            Self::Cue => "Cue",
            Self::Comment => "Comment",
            Self::Defect => "Defect",
            Self::ContentAdvisory => "Content Advisory",
            Self::Legal => "Legal",
            Self::Highlight => "Highlight",
            Self::Sync => "Sync",
            Self::AdBreak => "Ad Break",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// MarkerColor
// ---------------------------------------------------------------------------

/// A colour hint for rendering markers in UI timelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarkerColor {
    /// Red – often used for errors/defects.
    Red,
    /// Orange – warnings, advisories.
    Orange,
    /// Yellow – in/out selections.
    Yellow,
    /// Green – approved / cleared.
    Green,
    /// Cyan – sync points.
    Cyan,
    /// Blue – chapters / cues.
    Blue,
    /// Purple – highlights.
    Purple,
    /// Magenta – ad breaks / splice points.
    Magenta,
    /// White – neutral / generic.
    White,
}

impl MarkerColor {
    /// Hex colour string (CSS format).
    #[must_use]
    pub const fn hex(&self) -> &'static str {
        match self {
            Self::Red => "#FF4444",
            Self::Orange => "#FF8C00",
            Self::Yellow => "#FFD700",
            Self::Green => "#44CC44",
            Self::Cyan => "#00CED1",
            Self::Blue => "#4488FF",
            Self::Purple => "#9370DB",
            Self::Magenta => "#FF00FF",
            Self::White => "#FFFFFF",
        }
    }
}

// ---------------------------------------------------------------------------
// TimelineMarker
// ---------------------------------------------------------------------------

/// A single marker or range on a media asset's timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineMarker {
    /// Unique identifier.
    pub id: Uuid,
    /// The asset this marker belongs to.
    pub asset_id: Uuid,
    /// Classification of the marker.
    pub kind: MarkerKind,
    /// Start timecode in milliseconds.
    pub timecode_ms: u64,
    /// End timecode in milliseconds (for range markers; `None` = point marker).
    pub duration_ms: Option<u64>,
    /// Short label (e.g. chapter name, cue name).
    pub label: String,
    /// Optional longer description.
    pub description: Option<String>,
    /// Colour hint for UI rendering.
    pub color: MarkerColor,
    /// The user who created this marker.
    pub created_by: Option<Uuid>,
    /// When the marker was created.
    pub created_at: DateTime<Utc>,
    /// When the marker was last modified.
    pub updated_at: DateTime<Utc>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl TimelineMarker {
    /// Create a new point marker.
    #[must_use]
    pub fn point(
        asset_id: Uuid,
        kind: MarkerKind,
        timecode_ms: u64,
        label: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            asset_id,
            kind,
            timecode_ms,
            duration_ms: None,
            label: label.into(),
            description: None,
            color: MarkerColor::Blue,
            created_by: None,
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Create a new range marker.
    #[must_use]
    pub fn range(
        asset_id: Uuid,
        kind: MarkerKind,
        timecode_ms: u64,
        duration_ms: u64,
        label: impl Into<String>,
    ) -> Self {
        let mut m = Self::point(asset_id, kind, timecode_ms, label);
        m.duration_ms = Some(duration_ms);
        m
    }

    /// Builder: set the colour.
    #[must_use]
    pub fn with_color(mut self, color: MarkerColor) -> Self {
        self.color = color;
        self
    }

    /// Builder: set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Builder: set the creator.
    #[must_use]
    pub fn with_created_by(mut self, user_id: Uuid) -> Self {
        self.created_by = Some(user_id);
        self
    }

    /// Add or update a metadata key.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.updated_at = Utc::now();
        self.metadata.insert(key.into(), value.into());
    }

    /// Returns `true` if this is a range marker.
    #[must_use]
    pub fn is_range(&self) -> bool {
        self.duration_ms.is_some()
    }

    /// End timecode in milliseconds (start + duration for range markers; same
    /// as `timecode_ms` for point markers).
    #[must_use]
    pub fn end_timecode_ms(&self) -> u64 {
        self.timecode_ms + self.duration_ms.unwrap_or(0)
    }

    /// Returns `true` if the given timecode (ms) falls within this marker.
    ///
    /// For point markers, an exact match is required.
    #[must_use]
    pub fn contains(&self, query_ms: u64) -> bool {
        if let Some(dur) = self.duration_ms {
            query_ms >= self.timecode_ms && query_ms < self.timecode_ms + dur
        } else {
            query_ms == self.timecode_ms
        }
    }
}

// ---------------------------------------------------------------------------
// ChapterMarker
// ---------------------------------------------------------------------------

/// A named chapter with rich metadata.
///
/// Chapter markers are a specialisation of `TimelineMarker` with additional
/// fields: chapter number, thumbnail URL, and a longer synopsis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterMarker {
    /// Underlying timeline marker (kind is always `MarkerKind::Chapter`).
    pub marker: TimelineMarker,
    /// Sequential chapter number (1-based).
    pub chapter_number: u32,
    /// Optional URL to a chapter thumbnail image.
    pub thumbnail_url: Option<String>,
    /// Longer synopsis for the chapter.
    pub synopsis: Option<String>,
}

impl ChapterMarker {
    /// Create a new chapter marker.
    #[must_use]
    pub fn new(
        asset_id: Uuid,
        timecode_ms: u64,
        chapter_number: u32,
        title: impl Into<String>,
    ) -> Self {
        let marker =
            TimelineMarker::point(asset_id, MarkerKind::Chapter, timecode_ms, title)
                .with_color(MarkerColor::Blue);
        Self {
            marker,
            chapter_number,
            thumbnail_url: None,
            synopsis: None,
        }
    }

    /// Builder: set the thumbnail URL.
    #[must_use]
    pub fn with_thumbnail(mut self, url: impl Into<String>) -> Self {
        self.thumbnail_url = Some(url.into());
        self
    }

    /// Builder: set the synopsis.
    #[must_use]
    pub fn with_synopsis(mut self, s: impl Into<String>) -> Self {
        self.synopsis = Some(s.into());
        self
    }

    /// Short title from the underlying marker label.
    #[must_use]
    pub fn title(&self) -> &str {
        &self.marker.label
    }
}

// ---------------------------------------------------------------------------
// CuePoint
// ---------------------------------------------------------------------------

/// A named cue point for playback control or live event triggering.
///
/// Cue points are often used in live production to automate graphics, start
/// ad breaks, or synchronise external systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuePoint {
    /// Underlying timeline marker (kind is always `MarkerKind::Cue`).
    pub marker: TimelineMarker,
    /// Optional trigger action string (e.g. `"START_AD_BREAK"`, `"SHOW_GRAPHIC"`).
    pub action: Option<String>,
    /// Optional payload for the trigger action (JSON encoded).
    pub action_payload: Option<serde_json::Value>,
    /// Whether this cue point has been triggered / consumed.
    pub triggered: bool,
    /// When the cue was triggered (if triggered).
    pub triggered_at: Option<DateTime<Utc>>,
}

impl CuePoint {
    /// Create a new cue point.
    #[must_use]
    pub fn new(asset_id: Uuid, timecode_ms: u64, name: impl Into<String>) -> Self {
        let marker =
            TimelineMarker::point(asset_id, MarkerKind::Cue, timecode_ms, name)
                .with_color(MarkerColor::Magenta);
        Self {
            marker,
            action: None,
            action_payload: None,
            triggered: false,
            triggered_at: None,
        }
    }

    /// Builder: set the action string.
    #[must_use]
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.action = Some(action.into());
        self
    }

    /// Builder: set the action payload.
    #[must_use]
    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.action_payload = Some(payload);
        self
    }

    /// Mark the cue as triggered at the given time.
    pub fn trigger(&mut self, at: DateTime<Utc>) {
        self.triggered = true;
        self.triggered_at = Some(at);
    }
}

// ---------------------------------------------------------------------------
// MarkerSet
// ---------------------------------------------------------------------------

/// A sorted collection of timeline markers for a single asset.
///
/// Markers are kept sorted by `timecode_ms` ascending so that range queries
/// and chapter navigation are efficient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerSet {
    /// Asset id.
    pub asset_id: Uuid,
    /// Markers sorted by timecode.
    markers: Vec<TimelineMarker>,
}

impl MarkerSet {
    /// Create an empty marker set for an asset.
    #[must_use]
    pub fn new(asset_id: Uuid) -> Self {
        Self {
            asset_id,
            markers: Vec::new(),
        }
    }

    /// Insert a marker, maintaining timecode order.
    pub fn insert(&mut self, marker: TimelineMarker) {
        let pos = self
            .markers
            .partition_point(|m| m.timecode_ms <= marker.timecode_ms);
        self.markers.insert(pos, marker);
    }

    /// Remove a marker by id.  Returns `true` if found and removed.
    pub fn remove(&mut self, id: Uuid) -> bool {
        if let Some(pos) = self.markers.iter().position(|m| m.id == id) {
            self.markers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Look up a marker by id.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<&TimelineMarker> {
        self.markers.iter().find(|m| m.id == id)
    }

    /// Look up a marker mutably by id.
    #[must_use]
    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut TimelineMarker> {
        self.markers.iter_mut().find(|m| m.id == id)
    }

    /// All markers as an ordered slice.
    #[must_use]
    pub fn all(&self) -> &[TimelineMarker] {
        &self.markers
    }

    /// Markers of a specific kind.
    #[must_use]
    pub fn by_kind(&self, kind: &MarkerKind) -> Vec<&TimelineMarker> {
        self.markers.iter().filter(|m| &m.kind == kind).collect()
    }

    /// Markers that fall within the given time window [start_ms, end_ms).
    #[must_use]
    pub fn in_window(&self, start_ms: u64, end_ms: u64) -> Vec<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.timecode_ms >= start_ms && m.timecode_ms < end_ms)
            .collect()
    }

    /// Markers that contain the given timecode (point or range containment).
    #[must_use]
    pub fn containing(&self, timecode_ms: u64) -> Vec<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.contains(timecode_ms))
            .collect()
    }

    /// Total number of markers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Returns `true` if there are no markers.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
    }

    /// Chapter markers sorted by chapter number.
    ///
    /// Only markers of kind `MarkerKind::Chapter` are returned; chapter number
    /// is derived from label parsing where no explicit number is stored.
    #[must_use]
    pub fn chapters(&self) -> Vec<&TimelineMarker> {
        let mut chapters: Vec<&TimelineMarker> = self
            .markers
            .iter()
            .filter(|m| m.kind == MarkerKind::Chapter)
            .collect();
        // Keep them in timecode order (already sorted in the vec).
        chapters.sort_by_key(|m| m.timecode_ms);
        chapters
    }

    /// Next chapter after the given timecode, if any.
    #[must_use]
    pub fn next_chapter(&self, after_ms: u64) -> Option<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.kind == MarkerKind::Chapter && m.timecode_ms > after_ms)
            .min_by_key(|m| m.timecode_ms)
    }

    /// Previous chapter at or before the given timecode, if any.
    #[must_use]
    pub fn prev_chapter(&self, before_ms: u64) -> Option<&TimelineMarker> {
        self.markers
            .iter()
            .filter(|m| m.kind == MarkerKind::Chapter && m.timecode_ms <= before_ms)
            .max_by_key(|m| m.timecode_ms)
    }
}

// ---------------------------------------------------------------------------
// MarkerRegistry
// ---------------------------------------------------------------------------

/// In-memory store of `MarkerSet`s indexed by asset id.
#[derive(Debug, Default)]
pub struct MarkerRegistry {
    sets: HashMap<Uuid, MarkerSet>,
}

impl MarkerRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create the marker set for an asset.
    pub fn set_for_asset(&mut self, asset_id: Uuid) -> &mut MarkerSet {
        self.sets
            .entry(asset_id)
            .or_insert_with(|| MarkerSet::new(asset_id))
    }

    /// Insert a marker for an asset (creates the set if needed).
    pub fn insert(&mut self, marker: TimelineMarker) {
        self.set_for_asset(marker.asset_id).insert(marker);
    }

    /// Get a read-only view of the marker set for an asset.
    #[must_use]
    pub fn get_set(&self, asset_id: Uuid) -> Option<&MarkerSet> {
        self.sets.get(&asset_id)
    }

    /// Remove a marker by asset id and marker id.  Returns `true` if removed.
    pub fn remove_marker(&mut self, asset_id: Uuid, marker_id: Uuid) -> bool {
        self.sets
            .get_mut(&asset_id)
            .map(|s| s.remove(marker_id))
            .unwrap_or(false)
    }

    /// Total number of assets with markers.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.sets.len()
    }

    /// Total number of markers across all assets.
    #[must_use]
    pub fn total_marker_count(&self) -> usize {
        self.sets.values().map(|s| s.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_asset_id() -> Uuid {
        Uuid::new_v4()
    }

    #[test]
    fn test_marker_color_hex() {
        assert_eq!(MarkerColor::Red.hex(), "#FF4444");
        assert_eq!(MarkerColor::Blue.hex(), "#4488FF");
        assert_eq!(MarkerColor::White.hex(), "#FFFFFF");
    }

    #[test]
    fn test_marker_kind_label() {
        assert_eq!(MarkerKind::Chapter.label(), "Chapter");
        assert_eq!(MarkerKind::AdBreak.label(), "Ad Break");
        assert_eq!(MarkerKind::Custom("X".to_string()).label(), "X");
    }

    #[test]
    fn test_point_marker_is_not_range() {
        let asset_id = sample_asset_id();
        let m = TimelineMarker::point(asset_id, MarkerKind::Cue, 5_000, "Cue 1");
        assert!(!m.is_range());
        assert_eq!(m.end_timecode_ms(), 5_000);
    }

    #[test]
    fn test_range_marker_contains() {
        let asset_id = sample_asset_id();
        let m = TimelineMarker::range(asset_id, MarkerKind::InOut, 10_000, 5_000, "Sel");
        assert!(m.is_range());
        assert!(m.contains(10_000));
        assert!(m.contains(12_000));
        assert!(!m.contains(15_000)); // exclusive end
        assert!(!m.contains(9_999));
    }

    #[test]
    fn test_timeline_marker_set_metadata() {
        let asset_id = sample_asset_id();
        let mut m = TimelineMarker::point(asset_id, MarkerKind::Comment, 0, "Note");
        m.set_metadata("reviewer", "alice");
        assert_eq!(m.metadata.get("reviewer").map(String::as_str), Some("alice"));
    }

    #[test]
    fn test_chapter_marker_construction() {
        let asset_id = sample_asset_id();
        let ch = ChapterMarker::new(asset_id, 60_000, 1, "Intro")
            .with_synopsis("Opening scene")
            .with_thumbnail("https://cdn.example/thumb1.jpg");
        assert_eq!(ch.chapter_number, 1);
        assert_eq!(ch.title(), "Intro");
        assert!(ch.synopsis.is_some());
        assert!(ch.thumbnail_url.is_some());
    }

    #[test]
    fn test_cue_point_trigger() {
        let asset_id = sample_asset_id();
        let mut cue = CuePoint::new(asset_id, 30_000, "Break 1")
            .with_action("START_AD_BREAK");
        assert!(!cue.triggered);
        let now = Utc::now();
        cue.trigger(now);
        assert!(cue.triggered);
        assert_eq!(cue.triggered_at, Some(now));
    }

    #[test]
    fn test_marker_set_insert_maintains_order() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 30_000, "C"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 10_000, "A"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 20_000, "B"));
        let markers = set.all();
        assert_eq!(markers[0].label, "A");
        assert_eq!(markers[1].label, "B");
        assert_eq!(markers[2].label, "C");
    }

    #[test]
    fn test_marker_set_by_kind() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 0, "Ch1"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 5_000, "Cue1"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 60_000, "Ch2"));
        let chapters = set.by_kind(&MarkerKind::Chapter);
        assert_eq!(chapters.len(), 2);
        let cues = set.by_kind(&MarkerKind::Cue);
        assert_eq!(cues.len(), 1);
    }

    #[test]
    fn test_marker_set_in_window() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 5_000, "A"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 15_000, "B"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Cue, 25_000, "C"));
        let window = set.in_window(10_000, 20_000);
        assert_eq!(window.len(), 1);
        assert_eq!(window[0].label, "B");
    }

    #[test]
    fn test_marker_set_next_chapter() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 0, "Intro"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 60_000, "Act 1"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 120_000, "Act 2"));
        let next = set.next_chapter(30_000);
        assert!(next.is_some());
        assert_eq!(next.unwrap().label, "Act 1");
    }

    #[test]
    fn test_marker_set_prev_chapter() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 0, "Intro"));
        set.insert(TimelineMarker::point(asset_id, MarkerKind::Chapter, 60_000, "Act 1"));
        let prev = set.prev_chapter(90_000);
        assert!(prev.is_some());
        assert_eq!(prev.unwrap().label, "Act 1");
    }

    #[test]
    fn test_marker_set_remove() {
        let asset_id = sample_asset_id();
        let mut set = MarkerSet::new(asset_id);
        let m = TimelineMarker::point(asset_id, MarkerKind::Cue, 5_000, "Temp");
        let id = m.id;
        set.insert(m);
        assert_eq!(set.len(), 1);
        assert!(set.remove(id));
        assert_eq!(set.len(), 0);
        assert!(!set.remove(id)); // idempotent
    }

    #[test]
    fn test_marker_registry_total_count() {
        let a1 = Uuid::new_v4();
        let a2 = Uuid::new_v4();
        let mut registry = MarkerRegistry::new();
        registry.insert(TimelineMarker::point(a1, MarkerKind::Chapter, 0, "Ch1"));
        registry.insert(TimelineMarker::point(a1, MarkerKind::Chapter, 60_000, "Ch2"));
        registry.insert(TimelineMarker::point(a2, MarkerKind::Cue, 10_000, "Cue1"));
        assert_eq!(registry.asset_count(), 2);
        assert_eq!(registry.total_marker_count(), 3);
    }

    #[test]
    fn test_marker_registry_remove_marker() {
        let asset_id = Uuid::new_v4();
        let mut registry = MarkerRegistry::new();
        let m = TimelineMarker::point(asset_id, MarkerKind::Comment, 0, "Note");
        let mid = m.id;
        registry.insert(m);
        assert!(registry.remove_marker(asset_id, mid));
        assert_eq!(
            registry
                .get_set(asset_id)
                .map(|s| s.len())
                .unwrap_or(0),
            0
        );
    }
}
