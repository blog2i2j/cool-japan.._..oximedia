//! Timeline and track structures.
//!
//! The timeline is a multi-track structure containing video, audio, and subtitle clips.

use oximedia_core::Rational;
use std::collections::HashMap;

use crate::clip::{Clip, ClipId, ClipSelection, ClipType};
use crate::error::{EditError, EditResult};
use crate::group::{GroupManager, LinkManager};
use crate::marker::{InOutPoints, MarkerManager, RegionManager};
use crate::transition::TransitionManager;

/// A multi-track timeline.
#[derive(Debug)]
pub struct Timeline {
    /// Timeline tracks.
    pub tracks: Vec<Track>,
    /// Timeline timebase (e.g., 1/1000 for milliseconds).
    pub timebase: Rational,
    /// Timeline frame rate (for video).
    pub frame_rate: Rational,
    /// Timeline duration (in timebase units).
    pub duration: i64,
    /// Current playhead position.
    pub playhead: i64,
    /// Clip selection.
    pub selection: ClipSelection,
    /// Transition manager.
    pub transitions: TransitionManager,
    /// Marker manager.
    pub markers: MarkerManager,
    /// Region manager.
    pub regions: RegionManager,
    /// In/Out points.
    pub in_out: InOutPoints,
    /// Group manager.
    pub groups: GroupManager,
    /// Link manager.
    pub links: LinkManager,
    /// Next clip ID.
    pub next_clip_id: u64,
    /// Clip lookup by ID.
    pub clip_map: HashMap<ClipId, (usize, usize)>, // (track_index, clip_index)
}

impl Timeline {
    /// Create a new timeline.
    #[must_use]
    pub fn new(timebase: Rational, frame_rate: Rational) -> Self {
        Self {
            tracks: Vec::new(),
            timebase,
            frame_rate,
            duration: 0,
            playhead: 0,
            selection: ClipSelection::new(),
            transitions: TransitionManager::new(),
            markers: MarkerManager::new(),
            regions: RegionManager::new(),
            in_out: InOutPoints::new(),
            groups: GroupManager::new(),
            links: LinkManager::new(),
            next_clip_id: 1,
            clip_map: HashMap::new(),
        }
    }

    /// Create a timeline with default settings (1ms timebase, 30fps).
    #[must_use]
    pub fn default_settings() -> Self {
        Self::new(Rational::new(1, 1000), Rational::new(30, 1))
    }

    /// Add a new track.
    pub fn add_track(&mut self, track_type: TrackType) -> usize {
        let index = self.tracks.len();
        self.tracks.push(Track::new(index, track_type));
        index
    }

    /// Remove a track by index.
    pub fn remove_track(&mut self, index: usize) -> EditResult<Track> {
        if index >= self.tracks.len() {
            return Err(EditError::InvalidTrackIndex(index, self.tracks.len()));
        }

        // Remove clips from this track from the clip map
        let track = &self.tracks[index];
        for clip in &track.clips {
            self.clip_map.remove(&clip.id);
        }

        let track = self.tracks.remove(index);

        // Update track indices
        for (i, t) in self.tracks.iter_mut().enumerate() {
            t.index = i;
        }

        // Update clip map indices
        self.rebuild_clip_map();

        Ok(track)
    }

    /// Get a track by index.
    #[must_use]
    pub fn get_track(&self, index: usize) -> Option<&Track> {
        self.tracks.get(index)
    }

    /// Get a mutable track by index.
    pub fn get_track_mut(&mut self, index: usize) -> Option<&mut Track> {
        self.tracks.get_mut(index)
    }

    /// Add a clip to a track.
    pub fn add_clip(&mut self, track_index: usize, mut clip: Clip) -> EditResult<ClipId> {
        if track_index >= self.tracks.len() {
            return Err(EditError::InvalidTrackIndex(track_index, self.tracks.len()));
        }

        // Assign clip ID
        clip.id = self.next_clip_id;
        self.next_clip_id += 1;
        let clip_id = clip.id;

        // Check for overlaps
        let track = &self.tracks[track_index];
        for existing in &track.clips {
            if existing.overlaps(clip.timeline_start, clip.timeline_end()) {
                return Err(EditError::ClipOverlap(clip.timeline_start, track_index));
            }
        }

        // Add clip to track
        let track = &mut self.tracks[track_index];
        let _clip_index = track.clips.len();
        track.clips.push(clip);
        track.sort_clips();

        // Update clip map
        let clip_index = track
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .expect("clip was just inserted into this track");
        self.clip_map.insert(clip_id, (track_index, clip_index));

        // Update timeline duration
        self.update_duration();

        Ok(clip_id)
    }

    /// Remove a clip by ID.
    pub fn remove_clip(&mut self, clip_id: ClipId) -> EditResult<Clip> {
        let (track_index, _) = self
            .clip_map
            .get(&clip_id)
            .copied()
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let track = &mut self.tracks[track_index];
        let clip_index = track
            .clips
            .iter()
            .position(|c| c.id == clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let clip = track.clips.remove(clip_index);
        self.clip_map.remove(&clip_id);
        self.rebuild_clip_map();
        self.update_duration();

        Ok(clip)
    }

    /// Get a clip by ID.
    #[must_use]
    pub fn get_clip(&self, clip_id: ClipId) -> Option<&Clip> {
        let (track_index, clip_index) = self.clip_map.get(&clip_id)?;
        self.tracks.get(*track_index)?.clips.get(*clip_index)
    }

    /// Get a mutable clip by ID.
    pub fn get_clip_mut(&mut self, clip_id: ClipId) -> Option<&mut Clip> {
        let (track_index, clip_index) = self.clip_map.get(&clip_id).copied()?;
        self.tracks.get_mut(track_index)?.clips.get_mut(clip_index)
    }

    /// Move a clip to a new position on the timeline.
    pub fn move_clip(&mut self, clip_id: ClipId, new_start: i64) -> EditResult<()> {
        let clip = self
            .get_clip_mut(clip_id)
            .ok_or(EditError::ClipNotFound(clip_id))?;

        let duration = clip.timeline_duration;
        clip.timeline_start = new_start;

        // Re-sort the track
        let (track_index, _) = self
            .clip_map
            .get(&clip_id)
            .copied()
            .expect("clip_id confirmed present via get_clip_mut check above");
        self.tracks[track_index].sort_clips();
        self.rebuild_clip_map();

        // Check for overlaps
        let track = &self.tracks[track_index];
        for (i, clip) in track.clips.iter().enumerate() {
            if clip.id == clip_id {
                // Check against previous clip
                if i > 0 && track.clips[i - 1].timeline_end() > new_start {
                    return Err(EditError::ClipOverlap(new_start, track_index));
                }
                // Check against next clip
                if i < track.clips.len() - 1
                    && track.clips[i + 1].timeline_start < new_start + duration
                {
                    return Err(EditError::ClipOverlap(new_start, track_index));
                }
            }
        }

        self.update_duration();
        Ok(())
    }

    /// Move a clip to a different track.
    pub fn move_clip_to_track(&mut self, clip_id: ClipId, target_track: usize) -> EditResult<()> {
        if target_track >= self.tracks.len() {
            return Err(EditError::InvalidTrackIndex(
                target_track,
                self.tracks.len(),
            ));
        }

        let clip = self.remove_clip(clip_id)?;
        self.add_clip(target_track, clip)?;
        Ok(())
    }

    /// Get all clips at a specific timeline position.
    #[must_use]
    pub fn get_clips_at(&self, position: i64) -> Vec<(usize, &Clip)> {
        let mut result = Vec::new();
        for track in &self.tracks {
            if let Some(clip) = track.get_clip_at(position) {
                result.push((track.index, clip));
            }
        }
        result
    }

    /// Get all clips in a time range.
    #[must_use]
    pub fn get_clips_in_range(&self, start: i64, end: i64) -> Vec<(usize, &Clip)> {
        let mut result = Vec::new();
        for track in &self.tracks {
            for clip in &track.clips {
                if clip.overlaps(start, end) {
                    result.push((track.index, clip));
                }
            }
        }
        result
    }

    /// Rebuild the clip map from scratch.
    pub fn rebuild_clip_map(&mut self) {
        self.clip_map.clear();
        for (track_index, track) in self.tracks.iter().enumerate() {
            for (clip_index, clip) in track.clips.iter().enumerate() {
                self.clip_map.insert(clip.id, (track_index, clip_index));
            }
        }
    }

    /// Update timeline duration based on clips.
    fn update_duration(&mut self) {
        let mut max_end = 0i64;
        for track in &self.tracks {
            if let Some(clip) = track.clips.last() {
                max_end = max_end.max(clip.timeline_end());
            }
        }
        self.duration = max_end;
    }

    /// Set playhead position.
    pub fn set_playhead(&mut self, position: i64) {
        self.playhead = position.clamp(0, self.duration);
    }

    /// Move playhead forward by delta.
    pub fn move_playhead(&mut self, delta: i64) {
        self.set_playhead(self.playhead + delta);
    }

    /// Seek to start of timeline.
    pub fn seek_to_start(&mut self) {
        self.playhead = 0;
    }

    /// Seek to end of timeline.
    pub fn seek_to_end(&mut self) {
        self.playhead = self.duration;
    }

    /// Get timeline duration in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        let timestamp = oximedia_core::Timestamp::new(self.duration, self.timebase);
        timestamp.to_seconds()
    }

    /// Get video tracks.
    #[must_use]
    pub fn video_tracks(&self) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| matches!(t.track_type, TrackType::Video))
            .collect()
    }

    /// Get audio tracks.
    #[must_use]
    pub fn audio_tracks(&self) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| matches!(t.track_type, TrackType::Audio))
            .collect()
    }

    /// Get subtitle tracks.
    #[must_use]
    pub fn subtitle_tracks(&self) -> Vec<&Track> {
        self.tracks
            .iter()
            .filter(|t| matches!(t.track_type, TrackType::Subtitle))
            .collect()
    }

    /// Get total number of clips.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.tracks.iter().map(|t| t.clips.len()).sum()
    }

    /// Clear all tracks and clips.
    pub fn clear(&mut self) {
        self.tracks.clear();
        self.clip_map.clear();
        self.selection.clear();
        self.transitions.clear();
        self.markers.clear();
        self.regions.clear();
        self.in_out.clear();
        self.groups.clear();
        self.links.clear();
        self.playhead = 0;
        self.duration = 0;
        self.next_clip_id = 1;
    }
}

impl Default for Timeline {
    fn default() -> Self {
        Self::default_settings()
    }
}

/// A track in the timeline.
#[derive(Clone, Debug)]
pub struct Track {
    /// Track index.
    pub index: usize,
    /// Track type (video, audio, or subtitle).
    pub track_type: TrackType,
    /// Clips in this track (sorted by start time).
    pub clips: Vec<Clip>,
    /// Track is muted.
    pub muted: bool,
    /// Track is soloed.
    pub solo: bool,
    /// Track is locked (cannot be edited).
    pub locked: bool,
    /// Track height (for UI).
    pub height: u32,
    /// Track name.
    pub name: Option<String>,
    /// Track color (for UI).
    pub color: Option<String>,
}

impl Track {
    /// Create a new track.
    #[must_use]
    pub fn new(index: usize, track_type: TrackType) -> Self {
        Self {
            index,
            track_type,
            clips: Vec::new(),
            muted: false,
            solo: false,
            locked: false,
            height: 60,
            name: None,
            color: None,
        }
    }

    /// Sort clips by timeline start position.
    pub fn sort_clips(&mut self) {
        self.clips.sort_by_key(|c| c.timeline_start);
    }

    /// Get clip at a specific timeline position.
    #[must_use]
    pub fn get_clip_at(&self, position: i64) -> Option<&Clip> {
        self.clips.iter().find(|c| c.contains(position))
    }

    /// Get mutable clip at a specific timeline position.
    pub fn get_clip_at_mut(&mut self, position: i64) -> Option<&mut Clip> {
        self.clips.iter_mut().find(|c| c.contains(position))
    }

    /// Get all clips in a time range.
    #[must_use]
    pub fn get_clips_in_range(&self, start: i64, end: i64) -> Vec<&Clip> {
        self.clips
            .iter()
            .filter(|c| c.overlaps(start, end))
            .collect()
    }

    /// Check if this is a video track.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(self.track_type, TrackType::Video)
    }

    /// Check if this is an audio track.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(self.track_type, TrackType::Audio)
    }

    /// Check if this is a subtitle track.
    #[must_use]
    pub fn is_subtitle(&self) -> bool {
        matches!(self.track_type, TrackType::Subtitle)
    }

    /// Get track duration (end of last clip).
    #[must_use]
    pub fn duration(&self) -> i64 {
        self.clips.last().map_or(0, super::clip::Clip::timeline_end)
    }

    /// Check if track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Get clip count.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }
}

/// Type of track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
}

impl TrackType {
    /// Check if this track type matches a clip type.
    #[must_use]
    pub fn matches_clip(&self, clip_type: ClipType) -> bool {
        matches!(
            (self, clip_type),
            (Self::Video, ClipType::Video)
                | (Self::Audio, ClipType::Audio)
                | (Self::Subtitle, ClipType::Subtitle)
        )
    }
}

/// Playback state of the timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PlaybackState {
    /// Timeline is stopped.
    #[default]
    Stopped,
    /// Timeline is playing.
    Playing,
    /// Timeline is paused.
    Paused,
    /// Timeline is seeking.
    Seeking,
}

/// Timeline configuration.
#[derive(Clone, Debug)]
pub struct TimelineConfig {
    /// Timeline timebase.
    pub timebase: Rational,
    /// Video frame rate.
    pub frame_rate: Rational,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Audio sample rate.
    pub sample_rate: u32,
    /// Audio channels.
    pub channels: u32,
}

impl Default for TimelineConfig {
    fn default() -> Self {
        Self {
            timebase: Rational::new(1, 1000),
            frame_rate: Rational::new(30, 1),
            width: 1920,
            height: 1080,
            sample_rate: 48000,
            channels: 2,
        }
    }
}

impl TimelineConfig {
    /// Create a configuration for 1080p 30fps.
    #[must_use]
    pub fn hd_1080p_30() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: Rational::new(30, 1),
            ..Default::default()
        }
    }

    /// Create a configuration for 1080p 60fps.
    #[must_use]
    pub fn hd_1080p_60() -> Self {
        Self {
            width: 1920,
            height: 1080,
            frame_rate: Rational::new(60, 1),
            ..Default::default()
        }
    }

    /// Create a configuration for 4K 30fps.
    #[must_use]
    pub fn uhd_4k_30() -> Self {
        Self {
            width: 3840,
            height: 2160,
            frame_rate: Rational::new(30, 1),
            ..Default::default()
        }
    }

    /// Create a configuration for 4K 60fps.
    #[must_use]
    pub fn uhd_4k_60() -> Self {
        Self {
            width: 3840,
            height: 2160,
            frame_rate: Rational::new(60, 1),
            ..Default::default()
        }
    }
}
