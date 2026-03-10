#![allow(dead_code)]
//! Track layout management for audio post-production sessions.
//!
//! Provides structures for organizing audio tracks in a session with
//! support for track ordering, grouping, color coding, visibility
//! control, and template-based layout creation.

use std::collections::HashMap;

/// Unique identifier for a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(u32);

impl TrackId {
    /// Create a new track identifier.
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw identifier value.
    pub fn value(self) -> u32 {
        self.0
    }
}

/// Type of audio track.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackType {
    /// Dialogue / voice track.
    Dialogue,
    /// Music track.
    Music,
    /// Sound effects track.
    Sfx,
    /// Foley track.
    Foley,
    /// Ambience / atmosphere track.
    Ambience,
    /// ADR (automated dialogue replacement) track.
    Adr,
    /// Narration / voice-over track.
    Narration,
    /// Aux / return track.
    Aux,
    /// Master track.
    Master,
    /// Generic audio track.
    Generic,
}

/// Channel format for a track.
#[derive(Debug, Clone, PartialEq)]
pub enum TrackFormat {
    /// Mono track.
    Mono,
    /// Stereo track.
    Stereo,
    /// Multi-channel (e.g. 5.1, 7.1).
    Multichannel(u32),
}

impl TrackFormat {
    /// Return the channel count.
    pub fn channel_count(&self) -> u32 {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Multichannel(n) => *n,
        }
    }
}

/// Color associated with a track for UI display.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackColor {
    /// Red component (0..255).
    pub r: u8,
    /// Green component (0..255).
    pub g: u8,
    /// Blue component (0..255).
    pub b: u8,
}

impl TrackColor {
    /// Create a new color.
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Standard blue for dialogue tracks.
    pub fn dialogue() -> Self {
        Self::new(70, 130, 220)
    }

    /// Standard green for music tracks.
    pub fn music() -> Self {
        Self::new(80, 180, 80)
    }

    /// Standard orange for SFX tracks.
    pub fn sfx() -> Self {
        Self::new(230, 150, 50)
    }

    /// Standard purple for foley tracks.
    pub fn foley() -> Self {
        Self::new(160, 80, 200)
    }

    /// Standard teal for ambience tracks.
    pub fn ambience() -> Self {
        Self::new(60, 190, 190)
    }

    /// Convert to a hex color string.
    pub fn to_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }
}

/// A single track in the layout.
#[derive(Debug, Clone)]
pub struct Track {
    /// Unique identifier.
    pub id: TrackId,
    /// Display name.
    pub name: String,
    /// Track type.
    pub track_type: TrackType,
    /// Channel format.
    pub format: TrackFormat,
    /// Display color.
    pub color: TrackColor,
    /// Whether the track is visible in the UI.
    pub visible: bool,
    /// Whether the track is locked (cannot be edited).
    pub locked: bool,
    /// Whether the track is record-armed.
    pub armed: bool,
    /// Track height in UI units.
    pub height: u32,
    /// Optional group this track belongs to.
    pub group: Option<String>,
    /// Order index for sorting.
    pub order: u32,
    /// Metadata.
    pub tags: HashMap<String, String>,
}

impl Track {
    /// Create a new track.
    pub fn new(
        id: TrackId,
        name: impl Into<String>,
        track_type: TrackType,
        format: TrackFormat,
    ) -> Self {
        let color = default_color_for_type(&track_type);
        Self {
            id,
            name: name.into(),
            track_type,
            format,
            color,
            visible: true,
            locked: false,
            armed: false,
            height: 60,
            group: None,
            order: 0,
            tags: HashMap::new(),
        }
    }

    /// Set the display color.
    pub fn with_color(mut self, color: TrackColor) -> Self {
        self.color = color;
        self
    }

    /// Set the group name.
    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Set the order index.
    pub fn with_order(mut self, order: u32) -> Self {
        self.order = order;
        self
    }

    /// Set the track height.
    pub fn with_height(mut self, height: u32) -> Self {
        self.height = height.max(20);
        self
    }

    /// Toggle visibility.
    pub fn toggle_visible(&mut self) {
        self.visible = !self.visible;
    }

    /// Toggle lock state.
    pub fn toggle_locked(&mut self) {
        self.locked = !self.locked;
    }

    /// Toggle record-arm.
    pub fn toggle_armed(&mut self) {
        self.armed = !self.armed;
    }

    /// Add a metadata tag.
    pub fn set_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }
}

/// Return a sensible default color for a track type.
fn default_color_for_type(tt: &TrackType) -> TrackColor {
    match tt {
        TrackType::Dialogue | TrackType::Adr | TrackType::Narration => TrackColor::dialogue(),
        TrackType::Music => TrackColor::music(),
        TrackType::Sfx => TrackColor::sfx(),
        TrackType::Foley => TrackColor::foley(),
        TrackType::Ambience => TrackColor::ambience(),
        TrackType::Aux | TrackType::Master | TrackType::Generic => TrackColor::new(128, 128, 128),
    }
}

/// A group of tracks that can be shown/hidden and reordered together.
#[derive(Debug, Clone)]
pub struct TrackGroup {
    /// Group name.
    pub name: String,
    /// Whether the group is expanded in the UI.
    pub expanded: bool,
    /// Color for the group header.
    pub color: TrackColor,
}

impl TrackGroup {
    /// Create a new track group.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            expanded: true,
            color: TrackColor::new(100, 100, 100),
        }
    }

    /// Set the group color.
    pub fn with_color(mut self, color: TrackColor) -> Self {
        self.color = color;
        self
    }

    /// Toggle expansion.
    pub fn toggle_expanded(&mut self) {
        self.expanded = !self.expanded;
    }
}

/// A complete track layout for a session.
#[derive(Debug)]
pub struct TrackLayout {
    /// All tracks, ordered.
    tracks: Vec<Track>,
    /// Named track groups.
    groups: HashMap<String, TrackGroup>,
    /// Counter for generating track IDs.
    next_id: u32,
}

impl Default for TrackLayout {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackLayout {
    /// Create an empty track layout.
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            groups: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a track to the layout, returning its ID.
    pub fn add_track(
        &mut self,
        name: impl Into<String>,
        track_type: TrackType,
        format: TrackFormat,
    ) -> TrackId {
        let id = TrackId::new(self.next_id);
        self.next_id += 1;
        let order = self.tracks.len() as u32;
        let track = Track::new(id, name, track_type, format).with_order(order);
        self.tracks.push(track);
        id
    }

    /// Add a track to a specific group, creating the group if needed.
    pub fn add_track_to_group(
        &mut self,
        name: impl Into<String>,
        track_type: TrackType,
        format: TrackFormat,
        group_name: impl Into<String>,
    ) -> TrackId {
        let group = group_name.into();
        if !self.groups.contains_key(&group) {
            self.groups.insert(group.clone(), TrackGroup::new(&group));
        }
        let id = TrackId::new(self.next_id);
        self.next_id += 1;
        let order = self.tracks.len() as u32;
        let track = Track::new(id, name, track_type, format)
            .with_order(order)
            .with_group(&group);
        self.tracks.push(track);
        id
    }

    /// Get a track by ID.
    pub fn get_track(&self, id: TrackId) -> Option<&Track> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Get a mutable track by ID.
    pub fn get_track_mut(&mut self, id: TrackId) -> Option<&mut Track> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }

    /// Remove a track by ID.
    pub fn remove_track(&mut self, id: TrackId) -> Option<Track> {
        if let Some(pos) = self.tracks.iter().position(|t| t.id == id) {
            Some(self.tracks.remove(pos))
        } else {
            None
        }
    }

    /// Return the total number of tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Return all tracks in order.
    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    /// Return tracks belonging to a specific group.
    pub fn tracks_in_group(&self, group_name: &str) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| t.group.as_deref() == Some(group_name))
            .collect()
    }

    /// Sort tracks by their order index.
    pub fn sort_by_order(&mut self) {
        self.tracks.sort_by_key(|t| t.order);
    }

    /// Sort tracks by type (dialogue first, then music, sfx, foley, ambience, others).
    pub fn sort_by_type(&mut self) {
        self.tracks.sort_by_key(|t| type_sort_key(&t.track_type));
    }

    /// Get a group by name.
    pub fn get_group(&self, name: &str) -> Option<&TrackGroup> {
        self.groups.get(name)
    }

    /// Return the number of groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// Create a standard film post-production layout template.
    pub fn film_template() -> Self {
        let mut layout = Self::new();
        // Dialogue stem
        for i in 1..=4 {
            layout.add_track_to_group(
                format!("DLG {i}"),
                TrackType::Dialogue,
                TrackFormat::Mono,
                "Dialogue",
            );
        }
        // ADR
        for i in 1..=2 {
            layout.add_track_to_group(format!("ADR {i}"), TrackType::Adr, TrackFormat::Mono, "ADR");
        }
        // Music
        for i in 1..=2 {
            layout.add_track_to_group(
                format!("MX {i}"),
                TrackType::Music,
                TrackFormat::Stereo,
                "Music",
            );
        }
        // SFX
        for i in 1..=4 {
            layout.add_track_to_group(
                format!("SFX {i}"),
                TrackType::Sfx,
                TrackFormat::Stereo,
                "SFX",
            );
        }
        // Foley
        for i in 1..=2 {
            layout.add_track_to_group(
                format!("FLY {i}"),
                TrackType::Foley,
                TrackFormat::Mono,
                "Foley",
            );
        }
        // Ambience
        layout.add_track_to_group("AMB", TrackType::Ambience, TrackFormat::Stereo, "Ambience");

        layout
    }
}

/// Sorting key for track types.
fn type_sort_key(tt: &TrackType) -> u32 {
    match tt {
        TrackType::Dialogue => 0,
        TrackType::Adr => 1,
        TrackType::Narration => 2,
        TrackType::Music => 3,
        TrackType::Sfx => 4,
        TrackType::Foley => 5,
        TrackType::Ambience => 6,
        TrackType::Aux => 7,
        TrackType::Master => 8,
        TrackType::Generic => 9,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_id() {
        let id = TrackId::new(5);
        assert_eq!(id.value(), 5);
    }

    #[test]
    fn test_track_format_channel_count() {
        assert_eq!(TrackFormat::Mono.channel_count(), 1);
        assert_eq!(TrackFormat::Stereo.channel_count(), 2);
        assert_eq!(TrackFormat::Multichannel(6).channel_count(), 6);
    }

    #[test]
    fn test_track_color_hex() {
        let c = TrackColor::new(255, 128, 0);
        assert_eq!(c.to_hex(), "#FF8000");
    }

    #[test]
    fn test_preset_colors() {
        let dlg = TrackColor::dialogue();
        let mus = TrackColor::music();
        assert_ne!(dlg, mus);
    }

    #[test]
    fn test_new_track_defaults() {
        let t = Track::new(
            TrackId::new(1),
            "DLG 1",
            TrackType::Dialogue,
            TrackFormat::Mono,
        );
        assert_eq!(t.name, "DLG 1");
        assert!(t.visible);
        assert!(!t.locked);
        assert!(!t.armed);
        assert_eq!(t.height, 60);
    }

    #[test]
    fn test_track_toggles() {
        let mut t = Track::new(TrackId::new(1), "T", TrackType::Generic, TrackFormat::Mono);
        t.toggle_visible();
        assert!(!t.visible);
        t.toggle_locked();
        assert!(t.locked);
        t.toggle_armed();
        assert!(t.armed);
    }

    #[test]
    fn test_track_with_group() {
        let t =
            Track::new(TrackId::new(1), "T", TrackType::Sfx, TrackFormat::Stereo).with_group("SFX");
        assert_eq!(t.group.as_deref(), Some("SFX"));
    }

    #[test]
    fn test_track_with_color() {
        let t = Track::new(TrackId::new(1), "T", TrackType::Music, TrackFormat::Stereo)
            .with_color(TrackColor::new(255, 0, 0));
        assert_eq!(t.color.r, 255);
    }

    #[test]
    fn test_track_group() {
        let mut g = TrackGroup::new("Dialogue");
        assert!(g.expanded);
        g.toggle_expanded();
        assert!(!g.expanded);
    }

    #[test]
    fn test_layout_add_track() {
        let mut layout = TrackLayout::new();
        let id = layout.add_track("DLG 1", TrackType::Dialogue, TrackFormat::Mono);
        assert_eq!(layout.track_count(), 1);
        assert!(layout.get_track(id).is_some());
    }

    #[test]
    fn test_layout_add_to_group() {
        let mut layout = TrackLayout::new();
        let id = layout.add_track_to_group("SFX 1", TrackType::Sfx, TrackFormat::Stereo, "SFX");
        assert_eq!(layout.track_count(), 1);
        assert_eq!(layout.group_count(), 1);
        let track = layout.get_track(id).expect("get_track should succeed");
        assert_eq!(track.group.as_deref(), Some("SFX"));
    }

    #[test]
    fn test_layout_remove_track() {
        let mut layout = TrackLayout::new();
        let id = layout.add_track("T", TrackType::Generic, TrackFormat::Mono);
        let removed = layout.remove_track(id);
        assert!(removed.is_some());
        assert_eq!(layout.track_count(), 0);
    }

    #[test]
    fn test_tracks_in_group() {
        let mut layout = TrackLayout::new();
        layout.add_track_to_group("D1", TrackType::Dialogue, TrackFormat::Mono, "DLG");
        layout.add_track_to_group("D2", TrackType::Dialogue, TrackFormat::Mono, "DLG");
        layout.add_track("M1", TrackType::Music, TrackFormat::Stereo);
        assert_eq!(layout.tracks_in_group("DLG").len(), 2);
        assert_eq!(layout.tracks_in_group("Music").len(), 0);
    }

    #[test]
    fn test_sort_by_type() {
        let mut layout = TrackLayout::new();
        layout.add_track("SFX", TrackType::Sfx, TrackFormat::Stereo);
        layout.add_track("DLG", TrackType::Dialogue, TrackFormat::Mono);
        layout.add_track("MUS", TrackType::Music, TrackFormat::Stereo);
        layout.sort_by_type();
        assert_eq!(layout.tracks()[0].track_type, TrackType::Dialogue);
        assert_eq!(layout.tracks()[1].track_type, TrackType::Music);
        assert_eq!(layout.tracks()[2].track_type, TrackType::Sfx);
    }

    #[test]
    fn test_film_template() {
        let layout = TrackLayout::film_template();
        assert!(layout.track_count() >= 15);
        assert!(layout.group_count() >= 4);
        assert!(layout.get_group("Dialogue").is_some());
        assert!(layout.get_group("Music").is_some());
        assert!(layout.get_group("SFX").is_some());
        assert!(layout.get_group("Foley").is_some());
    }

    #[test]
    fn test_default_layout() {
        let layout = TrackLayout::default();
        assert_eq!(layout.track_count(), 0);
    }

    #[test]
    fn test_track_tags() {
        let mut t = Track::new(TrackId::new(1), "T", TrackType::Generic, TrackFormat::Mono);
        t.set_tag("scene", "42");
        assert_eq!(t.tags.get("scene").map(|s| s.as_str()), Some("42"));
    }

    #[test]
    fn test_default_color_for_types() {
        let dlg = default_color_for_type(&TrackType::Dialogue);
        let sfx = default_color_for_type(&TrackType::Sfx);
        assert_ne!(dlg, sfx);
    }

    #[test]
    fn test_track_height_min() {
        let t =
            Track::new(TrackId::new(1), "T", TrackType::Generic, TrackFormat::Mono).with_height(5); // below minimum
        assert_eq!(t.height, 20);
    }
}
