//! Mix session management for audio post-production.
//!
//! Provides tracks, groups, automation, and session management for
//! professional audio mixing workflows.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// A single mix track with volume, pan, mute, solo, and group membership.
#[derive(Debug, Clone)]
pub struct MixTrack {
    /// Unique track identifier.
    pub id: u32,
    /// Human-readable track name.
    pub name: String,
    /// Volume level in dB (0 dB = unity gain).
    pub volume_db: f32,
    /// Pan position: –1.0 = hard left, 0.0 = center, +1.0 = hard right.
    pub pan: f32,
    /// Whether this track is muted.
    pub muted: bool,
    /// Whether this track is soloed.
    pub solo: bool,
    /// Optional group this track belongs to.
    pub group_id: Option<u32>,
}

impl MixTrack {
    /// Create a new track at unity gain, centered, neither muted nor soloed.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            volume_db: 0.0,
            pan: 0.0,
            muted: false,
            solo: false,
            group_id: None,
        }
    }

    /// Compute the effective linear gain of this track.
    ///
    /// - If `any_solo` is `true` and this track is not soloed → gain is 0.0.
    /// - If the track is muted → gain is 0.0.
    /// - Otherwise the gain is `10^(volume_db / 20)`.
    #[must_use]
    pub fn effective_gain(&self, any_solo: bool) -> f32 {
        if self.muted {
            return 0.0;
        }
        if any_solo && !self.solo {
            return 0.0;
        }
        10_f32.powf(self.volume_db / 20.0)
    }
}

/// A named group of tracks with a shared volume control.
#[derive(Debug, Clone)]
pub struct TrackGroup {
    /// Unique group identifier.
    pub id: u32,
    /// Human-readable group name.
    pub name: String,
    /// Group volume in dB (applied in addition to individual track volumes).
    pub volume_db: f32,
    /// IDs of tracks belonging to this group.
    pub track_ids: Vec<u32>,
}

impl TrackGroup {
    /// Create a new empty track group at unity gain.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            volume_db: 0.0,
            track_ids: Vec::new(),
        }
    }

    /// Add a track to this group.
    pub fn add_track(&mut self, id: u32) {
        if !self.track_ids.contains(&id) {
            self.track_ids.push(id);
        }
    }

    /// Remove a track from this group.
    ///
    /// Returns `true` if the track was present and removed.
    pub fn remove_track(&mut self, id: u32) -> bool {
        let before = self.track_ids.len();
        self.track_ids.retain(|&t| t != id);
        self.track_ids.len() < before
    }

    /// Number of tracks currently in this group.
    #[must_use]
    pub fn member_count(&self) -> usize {
        self.track_ids.len()
    }
}

/// Parameter automation for a single track, stored as time-sorted keyframes.
#[derive(Debug, Clone)]
pub struct MixAutomation {
    /// The track this automation belongs to.
    pub track_id: u32,
    /// Name of the automated parameter (e.g. "volume", "pan").
    pub parameter: String,
    /// Keyframes as (time_ms, value) pairs, sorted by time ascending.
    pub keyframes: Vec<(u64, f32)>,
}

impl MixAutomation {
    /// Create a new automation lane with no keyframes.
    #[must_use]
    pub fn new(track_id: u32, parameter: impl Into<String>) -> Self {
        Self {
            track_id,
            parameter: parameter.into(),
            keyframes: Vec::new(),
        }
    }

    /// Insert a keyframe. The list is kept sorted by time.
    pub fn add_keyframe(&mut self, time_ms: u64, value: f32) {
        // Remove any existing keyframe at the exact same time
        self.keyframes.retain(|(t, _)| *t != time_ms);
        self.keyframes.push((time_ms, value));
        self.keyframes.sort_by_key(|(t, _)| *t);
    }

    /// Compute the interpolated value at `time_ms` using linear interpolation.
    ///
    /// - Returns the first keyframe value if `time_ms` is before all keyframes.
    /// - Returns the last keyframe value if `time_ms` is after all keyframes.
    /// - Returns 0.0 if there are no keyframes.
    #[must_use]
    pub fn value_at(&self, time_ms: u64) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if time_ms <= self.keyframes[0].0 {
            return self.keyframes[0].1;
        }
        if let Some(last) = self.keyframes.last() {
            if time_ms >= last.0 {
                return last.1;
            }
        }
        // Find surrounding keyframes
        let idx = self
            .keyframes
            .partition_point(|(t, _)| *t <= time_ms)
            .saturating_sub(1);
        let (t0, v0) = self.keyframes[idx];
        let (t1, v1) = self.keyframes[idx + 1];
        if t1 == t0 {
            return v0;
        }
        let frac = (time_ms - t0) as f32 / (t1 - t0) as f32;
        v0 + frac * (v1 - v0)
    }
}

/// A complete mix session containing tracks, groups, and automation.
#[derive(Debug, Default)]
pub struct MixSession {
    /// Name of this mix session.
    pub name: String,
    /// All tracks in the session.
    pub tracks: Vec<MixTrack>,
    /// Track groups.
    pub groups: Vec<TrackGroup>,
    /// Automation lanes.
    pub automation: Vec<MixAutomation>,
}

impl MixSession {
    /// Create a new, empty mix session.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Add a track to the session.
    pub fn add_track(&mut self, track: MixTrack) {
        self.tracks.push(track);
    }

    /// Add a group to the session.
    pub fn add_group(&mut self, group: TrackGroup) {
        self.groups.push(group);
    }

    /// Add an automation lane to the session.
    pub fn add_automation(&mut self, automation: MixAutomation) {
        self.automation.push(automation);
    }

    /// Mute the track with the given id (no-op if not found).
    pub fn mute_track(&mut self, id: u32) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == id) {
            track.muted = true;
        }
    }

    /// Unmute the track with the given id (no-op if not found).
    pub fn unmute_track(&mut self, id: u32) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == id) {
            track.muted = false;
        }
    }

    /// Solo the track with the given id (no-op if not found).
    pub fn solo_track(&mut self, id: u32) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == id) {
            track.solo = true;
        }
    }

    /// Un-solo the track with the given id (no-op if not found).
    pub fn unsolo_track(&mut self, id: u32) {
        if let Some(track) = self.tracks.iter_mut().find(|t| t.id == id) {
            track.solo = false;
        }
    }

    /// Whether any track in the session is currently soloed.
    #[must_use]
    pub fn any_solo(&self) -> bool {
        self.tracks.iter().any(|t| t.solo)
    }

    /// Get the group volume (in dB) for a given group id.
    ///
    /// Returns `None` if the group does not exist.
    #[must_use]
    pub fn group_volume(&self, group_id: u32) -> Option<f32> {
        self.groups
            .iter()
            .find(|g| g.id == group_id)
            .map(|g| g.volume_db)
    }

    /// Get a reference to the track with the given id.
    ///
    /// Returns `None` if no track with that id exists.
    #[must_use]
    pub fn track_by_id(&self, id: u32) -> Option<&MixTrack> {
        self.tracks.iter().find(|t| t.id == id)
    }

    /// Get a mutable reference to the track with the given id.
    pub fn track_by_id_mut(&mut self, id: u32) -> Option<&mut MixTrack> {
        self.tracks.iter_mut().find(|t| t.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MixTrack tests ──────────────────────────────────────────────────────

    #[test]
    fn test_mix_track_unity_gain() {
        let track = MixTrack::new(1, "Dialogue");
        let gain = track.effective_gain(false);
        assert!((gain - 1.0).abs() < 1e-4, "Unity gain expected, got {gain}");
    }

    #[test]
    fn test_mix_track_muted_returns_zero() {
        let mut track = MixTrack::new(1, "FX");
        track.muted = true;
        assert_eq!(track.effective_gain(false), 0.0);
    }

    #[test]
    fn test_mix_track_solo_exclusion() {
        let track = MixTrack::new(1, "Music");
        // any_solo=true but this track is not soloed
        assert_eq!(track.effective_gain(true), 0.0);
    }

    #[test]
    fn test_mix_track_solo_included() {
        let mut track = MixTrack::new(1, "Dialogue");
        track.solo = true;
        let gain = track.effective_gain(true);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_mix_track_volume_db() {
        let mut track = MixTrack::new(1, "T");
        track.volume_db = 20.0; // +20 dB = 10x linear
        let gain = track.effective_gain(false);
        assert!((gain - 10.0).abs() < 0.01, "gain={gain}");
    }

    // ── TrackGroup tests ────────────────────────────────────────────────────

    #[test]
    fn test_track_group_add_and_count() {
        let mut group = TrackGroup::new(1, "Strings");
        group.add_track(10);
        group.add_track(11);
        assert_eq!(group.member_count(), 2);
    }

    #[test]
    fn test_track_group_no_duplicate() {
        let mut group = TrackGroup::new(1, "Brass");
        group.add_track(5);
        group.add_track(5);
        assert_eq!(group.member_count(), 1);
    }

    #[test]
    fn test_track_group_remove_existing() {
        let mut group = TrackGroup::new(1, "Perc");
        group.add_track(3);
        assert!(group.remove_track(3));
        assert_eq!(group.member_count(), 0);
    }

    #[test]
    fn test_track_group_remove_missing() {
        let mut group = TrackGroup::new(1, "Perc");
        group.add_track(3);
        assert!(!group.remove_track(99));
        assert_eq!(group.member_count(), 1);
    }

    // ── MixAutomation tests ─────────────────────────────────────────────────

    #[test]
    fn test_automation_empty_returns_zero() {
        let auto = MixAutomation::new(1, "volume");
        assert_eq!(auto.value_at(1000), 0.0);
    }

    #[test]
    fn test_automation_before_first_keyframe() {
        let mut auto = MixAutomation::new(1, "volume");
        auto.add_keyframe(1000, 5.0);
        assert_eq!(auto.value_at(0), 5.0);
    }

    #[test]
    fn test_automation_after_last_keyframe() {
        let mut auto = MixAutomation::new(1, "pan");
        auto.add_keyframe(1000, 0.5);
        assert_eq!(auto.value_at(9999), 0.5);
    }

    #[test]
    fn test_automation_linear_interpolation() {
        let mut auto = MixAutomation::new(1, "volume");
        auto.add_keyframe(0, 0.0);
        auto.add_keyframe(1000, 10.0);
        let v = auto.value_at(500);
        assert!((v - 5.0).abs() < 0.01, "Interpolated value={v}");
    }

    #[test]
    fn test_automation_exact_keyframe() {
        let mut auto = MixAutomation::new(1, "volume");
        auto.add_keyframe(500, 7.0);
        auto.add_keyframe(1000, 3.0);
        assert!((auto.value_at(500) - 7.0).abs() < 1e-4);
        assert!((auto.value_at(1000) - 3.0).abs() < 1e-4);
    }

    // ── MixSession tests ────────────────────────────────────────────────────

    #[test]
    fn test_session_add_and_find_track() {
        let mut session = MixSession::new("Scene 1");
        session.add_track(MixTrack::new(1, "Dialogue"));
        assert!(session.track_by_id(1).is_some());
        assert!(session.track_by_id(99).is_none());
    }

    #[test]
    fn test_session_mute_track() {
        let mut session = MixSession::new("S");
        session.add_track(MixTrack::new(2, "Music"));
        session.mute_track(2);
        assert!(
            session
                .track_by_id(2)
                .expect("track_by_id should succeed")
                .muted
        );
    }

    #[test]
    fn test_session_solo_track() {
        let mut session = MixSession::new("S");
        session.add_track(MixTrack::new(3, "FX"));
        session.solo_track(3);
        assert!(session.any_solo());
    }

    #[test]
    fn test_session_group_volume() {
        let mut session = MixSession::new("S");
        let mut group = TrackGroup::new(10, "Strings");
        group.volume_db = -3.0;
        session.add_group(group);
        assert_eq!(session.group_volume(10), Some(-3.0));
        assert_eq!(session.group_volume(99), None);
    }
}
