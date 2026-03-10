#![allow(dead_code)]
//! Spatial presence tracking for collaboration sessions.
//!
//! Tracks where each user is looking/working in the timeline, including
//! cursor position, viewport range, active track, and selection state.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A position in the timeline expressed in frames and track index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimelinePosition {
    /// Frame number (zero-based).
    pub frame: u64,
    /// Track index (zero-based).
    pub track: u32,
}

impl TimelinePosition {
    /// Create a new timeline position.
    pub fn new(frame: u64, track: u32) -> Self {
        Self { frame, track }
    }
}

impl std::fmt::Display for TimelinePosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frame={} track={}", self.frame, self.track)
    }
}

/// A viewport range on the timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewportRange {
    /// Start frame (inclusive).
    pub start_frame: u64,
    /// End frame (exclusive).
    pub end_frame: u64,
    /// Zoom level (1.0 = default).
    pub zoom: f64,
}

impl ViewportRange {
    /// Create a new viewport range.
    pub fn new(start_frame: u64, end_frame: u64, zoom: f64) -> Self {
        Self {
            start_frame,
            end_frame,
            zoom,
        }
    }

    /// Get the width of the viewport in frames.
    pub fn width_frames(&self) -> u64 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Check if a frame is within this viewport.
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frame && frame < self.end_frame
    }

    /// Check if two viewports overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_frame < other.end_frame && other.start_frame < self.end_frame
    }
}

/// Selection state of a user.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionState {
    /// Nothing selected.
    None,
    /// A time range is selected on one or more tracks.
    TimeRange {
        /// Start frame.
        start_frame: u64,
        /// End frame.
        end_frame: u64,
        /// Selected tracks.
        tracks: Vec<u32>,
    },
    /// One or more clips are selected by id.
    Clips {
        /// Selected clip identifiers.
        clip_ids: Vec<String>,
    },
}

impl SelectionState {
    /// Whether the user has something selected.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// Full presence information for a single user.
#[derive(Debug, Clone)]
pub struct UserPresence {
    /// User identifier.
    pub user_id: String,
    /// Display name.
    pub display_name: String,
    /// Assigned color (hex string).
    pub color: String,
    /// Current cursor position.
    pub cursor: TimelinePosition,
    /// Current viewport.
    pub viewport: ViewportRange,
    /// Current selection state.
    pub selection: SelectionState,
    /// Whether the user is actively editing.
    pub is_editing: bool,
    /// Last time the presence was updated.
    pub last_updated: Instant,
}

impl UserPresence {
    /// Create new user presence.
    pub fn new(user_id: &str, display_name: &str, color: &str) -> Self {
        Self {
            user_id: user_id.to_string(),
            display_name: display_name.to_string(),
            color: color.to_string(),
            cursor: TimelinePosition::new(0, 0),
            viewport: ViewportRange::new(0, 1000, 1.0),
            selection: SelectionState::None,
            is_editing: false,
            last_updated: Instant::now(),
        }
    }

    /// Time since last update.
    pub fn idle_duration(&self) -> Duration {
        self.last_updated.elapsed()
    }

    /// Whether this user is considered idle (no updates for given threshold).
    pub fn is_idle(&self, threshold: Duration) -> bool {
        self.idle_duration() >= threshold
    }
}

/// Configuration for the presence map.
#[derive(Debug, Clone)]
pub struct PresenceMapConfig {
    /// Duration after which a user is considered idle.
    pub idle_threshold: Duration,
    /// Duration after which an idle user is automatically removed.
    pub expiry_threshold: Duration,
}

impl Default for PresenceMapConfig {
    fn default() -> Self {
        Self {
            idle_threshold: Duration::from_secs(30),
            expiry_threshold: Duration::from_secs(300),
        }
    }
}

/// Manages spatial presence for all users in a session.
#[derive(Debug)]
pub struct UserPresenceMap {
    /// Configuration.
    config: PresenceMapConfig,
    /// Presence entries keyed by user_id.
    entries: HashMap<String, UserPresence>,
}

impl UserPresenceMap {
    /// Create a new presence map.
    pub fn new(config: PresenceMapConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    /// Add or update a user's presence.
    pub fn upsert(&mut self, presence: UserPresence) {
        self.entries.insert(presence.user_id.clone(), presence);
    }

    /// Update a user's cursor position.
    pub fn update_cursor(&mut self, user_id: &str, cursor: TimelinePosition) {
        if let Some(entry) = self.entries.get_mut(user_id) {
            entry.cursor = cursor;
            entry.last_updated = Instant::now();
        }
    }

    /// Update a user's viewport.
    pub fn update_viewport(&mut self, user_id: &str, viewport: ViewportRange) {
        if let Some(entry) = self.entries.get_mut(user_id) {
            entry.viewport = viewport;
            entry.last_updated = Instant::now();
        }
    }

    /// Update a user's selection.
    pub fn update_selection(&mut self, user_id: &str, selection: SelectionState) {
        if let Some(entry) = self.entries.get_mut(user_id) {
            entry.selection = selection;
            entry.last_updated = Instant::now();
        }
    }

    /// Set editing flag for a user.
    pub fn set_editing(&mut self, user_id: &str, editing: bool) {
        if let Some(entry) = self.entries.get_mut(user_id) {
            entry.is_editing = editing;
            entry.last_updated = Instant::now();
        }
    }

    /// Remove a user.
    pub fn remove(&mut self, user_id: &str) -> Option<UserPresence> {
        self.entries.remove(user_id)
    }

    /// Get a user's presence.
    pub fn get(&self, user_id: &str) -> Option<&UserPresence> {
        self.entries.get(user_id)
    }

    /// Get all user ids.
    pub fn user_ids(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Number of tracked users.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Get all users whose viewport overlaps the given range.
    pub fn users_viewing_range(&self, range: &ViewportRange) -> Vec<&UserPresence> {
        self.entries
            .values()
            .filter(|p| p.viewport.overlaps(range))
            .collect()
    }

    /// Get all users on a given track.
    pub fn users_on_track(&self, track: u32) -> Vec<&UserPresence> {
        self.entries
            .values()
            .filter(|p| p.cursor.track == track)
            .collect()
    }

    /// Get all idle users.
    pub fn idle_users(&self) -> Vec<&UserPresence> {
        self.entries
            .values()
            .filter(|p| p.is_idle(self.config.idle_threshold))
            .collect()
    }

    /// Remove users who have exceeded the expiry threshold.
    pub fn expire_stale(&mut self) -> Vec<String> {
        let threshold = self.config.expiry_threshold;
        let expired: Vec<String> = self
            .entries
            .iter()
            .filter(|(_, p)| p.idle_duration() >= threshold)
            .map(|(id, _)| id.clone())
            .collect();
        for id in &expired {
            self.entries.remove(id);
        }
        expired
    }

    /// Find users whose cursors are near a given frame (within `radius` frames).
    pub fn users_near_frame(&self, frame: u64, radius: u64) -> Vec<&UserPresence> {
        self.entries
            .values()
            .filter(|p| {
                let diff = if p.cursor.frame > frame {
                    p.cursor.frame - frame
                } else {
                    frame - p.cursor.frame
                };
                diff <= radius
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_presence(id: &str, frame: u64, track: u32) -> UserPresence {
        let mut p = UserPresence::new(id, id, "#FF0000");
        p.cursor = TimelinePosition::new(frame, track);
        p
    }

    #[test]
    fn test_timeline_position_display() {
        let pos = TimelinePosition::new(100, 3);
        assert_eq!(pos.to_string(), "frame=100 track=3");
    }

    #[test]
    fn test_viewport_width() {
        let vp = ViewportRange::new(100, 500, 1.0);
        assert_eq!(vp.width_frames(), 400);
    }

    #[test]
    fn test_viewport_contains() {
        let vp = ViewportRange::new(100, 500, 1.0);
        assert!(vp.contains_frame(100));
        assert!(vp.contains_frame(499));
        assert!(!vp.contains_frame(500));
        assert!(!vp.contains_frame(50));
    }

    #[test]
    fn test_viewport_overlaps() {
        let a = ViewportRange::new(0, 100, 1.0);
        let b = ViewportRange::new(50, 150, 1.0);
        let c = ViewportRange::new(100, 200, 1.0);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_selection_is_active() {
        assert!(!SelectionState::None.is_active());
        assert!(SelectionState::TimeRange {
            start_frame: 0,
            end_frame: 10,
            tracks: vec![0]
        }
        .is_active());
        assert!(SelectionState::Clips {
            clip_ids: vec!["c1".to_string()]
        }
        .is_active());
    }

    #[test]
    fn test_upsert_and_get() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("alice", 100, 0));
        assert_eq!(map.count(), 1);
        assert!(map.get("alice").is_some());
    }

    #[test]
    fn test_update_cursor() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("bob", 0, 0));
        map.update_cursor("bob", TimelinePosition::new(500, 2));
        let p = map
            .get("bob")
            .expect("collab test operation should succeed");
        assert_eq!(p.cursor.frame, 500);
        assert_eq!(p.cursor.track, 2);
    }

    #[test]
    fn test_update_viewport() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("c", 0, 0));
        map.update_viewport("c", ViewportRange::new(200, 800, 2.0));
        let p = map.get("c").expect("collab test operation should succeed");
        assert_eq!(p.viewport.start_frame, 200);
    }

    #[test]
    fn test_update_selection() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("d", 0, 0));
        map.update_selection(
            "d",
            SelectionState::Clips {
                clip_ids: vec!["clip1".to_string()],
            },
        );
        let p = map.get("d").expect("collab test operation should succeed");
        assert!(p.selection.is_active());
    }

    #[test]
    fn test_remove_user() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("e", 0, 0));
        let removed = map.remove("e");
        assert!(removed.is_some());
        assert_eq!(map.count(), 0);
    }

    #[test]
    fn test_users_on_track() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("a", 0, 1));
        map.upsert(make_presence("b", 100, 1));
        map.upsert(make_presence("c", 200, 2));
        let on_track1 = map.users_on_track(1);
        assert_eq!(on_track1.len(), 2);
    }

    #[test]
    fn test_users_near_frame() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("a", 100, 0));
        map.upsert(make_presence("b", 200, 0));
        map.upsert(make_presence("c", 500, 0));
        let near = map.users_near_frame(150, 60);
        assert_eq!(near.len(), 2); // a(100) and b(200) are within 60 frames of 150
    }

    #[test]
    fn test_users_viewing_range() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        let mut p1 = make_presence("a", 0, 0);
        p1.viewport = ViewportRange::new(0, 500, 1.0);
        let mut p2 = make_presence("b", 0, 0);
        p2.viewport = ViewportRange::new(600, 900, 1.0);
        map.upsert(p1);
        map.upsert(p2);
        let range = ViewportRange::new(400, 700, 1.0);
        let viewing = map.users_viewing_range(&range);
        assert_eq!(viewing.len(), 2); // both overlap
    }

    #[test]
    fn test_set_editing() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("x", 0, 0));
        map.set_editing("x", true);
        assert!(
            map.get("x")
                .expect("collab test operation should succeed")
                .is_editing
        );
    }

    #[test]
    fn test_user_ids() {
        let mut map = UserPresenceMap::new(PresenceMapConfig::default());
        map.upsert(make_presence("a", 0, 0));
        map.upsert(make_presence("b", 0, 0));
        let mut ids = map.user_ids();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }
}
