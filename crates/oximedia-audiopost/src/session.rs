//! Session management and project organization for audio post-production.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Post-production session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostProductionSession {
    /// Session ID
    pub id: Uuid,
    /// Session name
    pub name: String,
    /// Project directory
    pub project_dir: PathBuf,
    /// Sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
    /// Frame rate
    pub frame_rate: f64,
    /// Video reference path
    pub video_reference: Option<PathBuf>,
    /// Tracks
    tracks: HashMap<usize, Track>,
    /// Next track ID
    next_track_id: usize,
    /// Markers
    markers: Vec<Marker>,
    /// Regions
    regions: Vec<Region>,
    /// Created timestamp
    pub created: chrono::DateTime<chrono::Utc>,
    /// Modified timestamp
    pub modified: chrono::DateTime<chrono::Utc>,
    /// Notes
    pub notes: String,
}

impl PostProductionSession {
    /// Create a new post-production session
    ///
    /// # Errors
    ///
    /// Returns an error if sample rate or bit depth is invalid
    pub fn new(
        name: &str,
        project_dir: PathBuf,
        sample_rate: u32,
        bit_depth: u16,
        frame_rate: f64,
    ) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if bit_depth != 16 && bit_depth != 24 && bit_depth != 32 {
            return Err(AudioPostError::Generic(
                "Bit depth must be 16, 24, or 32".to_string(),
            ));
        }

        let now = chrono::Utc::now();

        Ok(Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            project_dir,
            sample_rate,
            bit_depth,
            frame_rate,
            video_reference: None,
            tracks: HashMap::new(),
            next_track_id: 1,
            markers: Vec::new(),
            regions: Vec::new(),
            created: now,
            modified: now,
            notes: String::new(),
        })
    }

    /// Add a track
    pub fn add_track(&mut self, track: Track) -> usize {
        let id = self.next_track_id;
        self.tracks.insert(id, track);
        self.next_track_id += 1;
        self.update_modified();
        id
    }

    /// Get a track
    ///
    /// # Errors
    ///
    /// Returns an error if track is not found
    pub fn get_track(&self, id: usize) -> AudioPostResult<&Track> {
        self.tracks
            .get(&id)
            .ok_or(AudioPostError::Generic(format!("Track {id} not found")))
    }

    /// Get a mutable track
    ///
    /// # Errors
    ///
    /// Returns an error if track is not found
    pub fn get_track_mut(&mut self, id: usize) -> AudioPostResult<&mut Track> {
        self.tracks
            .get_mut(&id)
            .ok_or(AudioPostError::Generic(format!("Track {id} not found")))
    }

    /// Remove a track
    ///
    /// # Errors
    ///
    /// Returns an error if track is not found
    pub fn remove_track(&mut self, id: usize) -> AudioPostResult<Track> {
        self.update_modified();
        self.tracks
            .remove(&id)
            .ok_or(AudioPostError::Generic(format!("Track {id} not found")))
    }

    /// Get all tracks
    #[must_use]
    pub fn get_all_tracks(&self) -> Vec<(usize, &Track)> {
        let mut tracks: Vec<_> = self.tracks.iter().map(|(id, track)| (*id, track)).collect();
        tracks.sort_by_key(|(_, track)| track.index);
        tracks
    }

    /// Get track count
    #[must_use]
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Add a marker
    pub fn add_marker(&mut self, marker: Marker) {
        self.markers.push(marker);
        self.markers.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.update_modified();
    }

    /// Get all markers
    #[must_use]
    pub fn get_markers(&self) -> &[Marker] {
        &self.markers
    }

    /// Add a region
    pub fn add_region(&mut self, region: Region) {
        self.regions.push(region);
        self.update_modified();
    }

    /// Get all regions
    #[must_use]
    pub fn get_regions(&self) -> &[Region] {
        &self.regions
    }

    /// Update modified timestamp
    fn update_modified(&mut self) {
        self.modified = chrono::Utc::now();
    }

    /// Get session duration (longest track or region end)
    #[must_use]
    pub fn get_duration(&self) -> f64 {
        let track_duration = self
            .tracks
            .values()
            .filter_map(|track| track.clips.last().map(|clip| clip.end_time))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let region_duration = self
            .regions
            .iter()
            .map(|region| region.end_time)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        track_duration.max(region_duration)
    }
}

/// Audio track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Track {
    /// Track name
    pub name: String,
    /// Track type
    pub track_type: TrackType,
    /// Track index/position
    pub index: usize,
    /// Audio clips on this track
    pub clips: Vec<AudioClip>,
    /// Muted flag
    pub muted: bool,
    /// Solo flag
    pub solo: bool,
    /// Record enabled
    pub record_enabled: bool,
    /// Volume (dB)
    pub volume: f32,
    /// Pan (-1.0 to 1.0)
    pub pan: f32,
    /// Color (RGB)
    pub color: Option<(u8, u8, u8)>,
}

impl Track {
    /// Create a new track
    #[must_use]
    pub fn new(name: &str, track_type: TrackType, index: usize) -> Self {
        Self {
            name: name.to_string(),
            track_type,
            index,
            clips: Vec::new(),
            muted: false,
            solo: false,
            record_enabled: false,
            volume: 0.0,
            pan: 0.0,
            color: None,
        }
    }

    /// Add a clip to the track
    pub fn add_clip(&mut self, clip: AudioClip) {
        self.clips.push(clip);
        self.clips.sort_by(|a, b| {
            a.start_time
                .partial_cmp(&b.start_time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Remove a clip
    ///
    /// # Errors
    ///
    /// Returns an error if clip is not found
    pub fn remove_clip(&mut self, clip_id: &Uuid) -> AudioPostResult<AudioClip> {
        let index = self
            .clips
            .iter()
            .position(|clip| clip.id == *clip_id)
            .ok_or(AudioPostError::Generic("Clip not found".to_string()))?;
        Ok(self.clips.remove(index))
    }

    /// Get clip count
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.clips.len()
    }

    /// Get total duration of all clips
    #[must_use]
    pub fn total_duration(&self) -> f64 {
        self.clips.last().map_or(0.0, |clip| clip.end_time)
    }
}

/// Track type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackType {
    /// Dialogue track
    Dialogue,
    /// Music track
    Music,
    /// Effects track
    Effects,
    /// Foley track
    Foley,
    /// Ambience track
    Ambience,
    /// ADR track
    Adr,
    /// Voiceover track
    Voiceover,
    /// Master track
    Master,
    /// Aux/Bus track
    Aux,
}

impl TrackType {
    /// Get display name
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dialogue => "Dialogue",
            Self::Music => "Music",
            Self::Effects => "Effects",
            Self::Foley => "Foley",
            Self::Ambience => "Ambience",
            Self::Adr => "ADR",
            Self::Voiceover => "Voiceover",
            Self::Master => "Master",
            Self::Aux => "Aux",
        }
    }
}

/// Audio clip on a track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioClip {
    /// Clip ID
    pub id: Uuid,
    /// Clip name
    pub name: String,
    /// Source file path
    pub source_file: PathBuf,
    /// Start time on timeline (seconds)
    pub start_time: f64,
    /// End time on timeline (seconds)
    pub end_time: f64,
    /// Offset in source file (seconds)
    pub source_offset: f64,
    /// Fade in duration (seconds)
    pub fade_in: f64,
    /// Fade out duration (seconds)
    pub fade_out: f64,
    /// Gain adjustment (dB)
    pub gain: f32,
    /// Muted flag
    pub muted: bool,
    /// Locked flag (prevents editing)
    pub locked: bool,
}

impl AudioClip {
    /// Create a new audio clip
    #[must_use]
    pub fn new(name: &str, source_file: PathBuf, start_time: f64, duration: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            source_file,
            start_time,
            end_time: start_time + duration,
            source_offset: 0.0,
            fade_in: 0.0,
            fade_out: 0.0,
            gain: 0.0,
            muted: false,
            locked: false,
        }
    }

    /// Get clip duration
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Set fade in duration
    pub fn set_fade_in(&mut self, duration: f64) {
        self.fade_in = duration.max(0.0).min(self.duration());
    }

    /// Set fade out duration
    pub fn set_fade_out(&mut self, duration: f64) {
        self.fade_out = duration.max(0.0).min(self.duration());
    }

    /// Move clip to new start time
    pub fn move_to(&mut self, new_start: f64) {
        let duration = self.duration();
        self.start_time = new_start;
        self.end_time = new_start + duration;
    }

    /// Trim clip start
    pub fn trim_start(&mut self, new_start: f64) {
        if new_start < self.end_time {
            let diff = new_start - self.start_time;
            self.start_time = new_start;
            self.source_offset += diff;
        }
    }

    /// Trim clip end
    pub fn trim_end(&mut self, new_end: f64) {
        if new_end > self.start_time {
            self.end_time = new_end;
        }
    }
}

/// Timeline marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Marker {
    /// Marker ID
    pub id: Uuid,
    /// Marker name
    pub name: String,
    /// Time position (seconds)
    pub time: f64,
    /// Marker type
    pub marker_type: MarkerType,
    /// Comment
    pub comment: String,
    /// Color (RGB)
    pub color: Option<(u8, u8, u8)>,
}

impl Marker {
    /// Create a new marker
    #[must_use]
    pub fn new(name: &str, time: f64, marker_type: MarkerType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            time,
            marker_type,
            comment: String::new(),
            color: None,
        }
    }
}

/// Marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarkerType {
    /// Generic marker
    Generic,
    /// Cue point
    Cue,
    /// Sync point
    Sync,
    /// Chapter marker
    Chapter,
    /// Edit point
    Edit,
    /// Note marker
    Note,
}

/// Timeline region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Region {
    /// Region ID
    pub id: Uuid,
    /// Region name
    pub name: String,
    /// Start time (seconds)
    pub start_time: f64,
    /// End time (seconds)
    pub end_time: f64,
    /// Region type
    pub region_type: RegionType,
    /// Color (RGB)
    pub color: Option<(u8, u8, u8)>,
}

impl Region {
    /// Create a new region
    #[must_use]
    pub fn new(name: &str, start_time: f64, end_time: f64, region_type: RegionType) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            start_time,
            end_time,
            region_type,
            color: None,
        }
    }

    /// Get region duration
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }
}

/// Region type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionType {
    /// Selection region
    Selection,
    /// Loop region
    Loop,
    /// Export region
    Export,
    /// Work region
    Work,
}

/// Project template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectTemplate {
    /// Template name
    pub name: String,
    /// Track templates
    pub tracks: Vec<TrackTemplate>,
    /// Sample rate
    pub sample_rate: u32,
    /// Bit depth
    pub bit_depth: u16,
    /// Frame rate
    pub frame_rate: f64,
}

impl ProjectTemplate {
    /// Create a new project template
    #[must_use]
    pub fn new(name: &str, sample_rate: u32, bit_depth: u16, frame_rate: f64) -> Self {
        Self {
            name: name.to_string(),
            tracks: Vec::new(),
            sample_rate,
            bit_depth,
            frame_rate,
        }
    }

    /// Add a track template
    pub fn add_track(&mut self, track: TrackTemplate) {
        self.tracks.push(track);
    }

    /// Create standard film/TV template
    #[must_use]
    pub fn film_tv_standard() -> Self {
        let mut template = Self::new("Film/TV Standard", 48000, 24, 24.0);

        template.add_track(TrackTemplate::new("Dialogue 1", TrackType::Dialogue));
        template.add_track(TrackTemplate::new("Dialogue 2", TrackType::Dialogue));
        template.add_track(TrackTemplate::new("Dialogue 3", TrackType::Dialogue));
        template.add_track(TrackTemplate::new("ADR 1", TrackType::Adr));
        template.add_track(TrackTemplate::new("ADR 2", TrackType::Adr));
        template.add_track(TrackTemplate::new("Foley 1", TrackType::Foley));
        template.add_track(TrackTemplate::new("Foley 2", TrackType::Foley));
        template.add_track(TrackTemplate::new("Effects 1", TrackType::Effects));
        template.add_track(TrackTemplate::new("Effects 2", TrackType::Effects));
        template.add_track(TrackTemplate::new("Ambience", TrackType::Ambience));
        template.add_track(TrackTemplate::new("Music 1", TrackType::Music));
        template.add_track(TrackTemplate::new("Music 2", TrackType::Music));

        template
    }

    /// Create podcast template
    #[must_use]
    pub fn podcast() -> Self {
        let mut template = Self::new("Podcast", 44100, 16, 30.0);

        template.add_track(TrackTemplate::new("Host", TrackType::Dialogue));
        template.add_track(TrackTemplate::new("Guest", TrackType::Dialogue));
        template.add_track(TrackTemplate::new("Music", TrackType::Music));
        template.add_track(TrackTemplate::new("Effects", TrackType::Effects));

        template
    }

    /// Create music mixing template
    #[must_use]
    pub fn music_mixing() -> Self {
        let mut template = Self::new("Music Mixing", 48000, 24, 30.0);

        for i in 1..=8 {
            template.add_track(TrackTemplate::new(&format!("Track {i}"), TrackType::Music));
        }

        template
    }
}

/// Track template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackTemplate {
    /// Track name
    pub name: String,
    /// Track type
    pub track_type: TrackType,
    /// Default volume (dB)
    pub default_volume: f32,
    /// Default pan
    pub default_pan: f32,
    /// Color (RGB)
    pub color: Option<(u8, u8, u8)>,
}

impl TrackTemplate {
    /// Create a new track template
    #[must_use]
    pub fn new(name: &str, track_type: TrackType) -> Self {
        Self {
            name: name.to_string(),
            track_type,
            default_volume: 0.0,
            default_pan: 0.0,
            color: None,
        }
    }
}

/// Session backup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBackup {
    /// Backup ID
    pub id: Uuid,
    /// Session snapshot
    pub session: PostProductionSession,
    /// Backup timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Backup description
    pub description: String,
}

impl SessionBackup {
    /// Create a new session backup
    #[must_use]
    pub fn new(session: PostProductionSession, description: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            session,
            timestamp: chrono::Utc::now(),
            description: description.to_string(),
        }
    }
}

/// Session manager
#[derive(Debug)]
pub struct SessionManager {
    /// Active sessions
    sessions: HashMap<Uuid, PostProductionSession>,
    /// Session backups
    backups: HashMap<Uuid, Vec<SessionBackup>>,
}

impl SessionManager {
    /// Create a new session manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            backups: HashMap::new(),
        }
    }

    /// Create and add a new session
    ///
    /// # Errors
    ///
    /// Returns an error if session creation fails
    pub fn create_session(
        &mut self,
        name: &str,
        project_dir: PathBuf,
        sample_rate: u32,
        bit_depth: u16,
        frame_rate: f64,
    ) -> AudioPostResult<Uuid> {
        let session =
            PostProductionSession::new(name, project_dir, sample_rate, bit_depth, frame_rate)?;
        let id = session.id;
        self.sessions.insert(id, session);
        Ok(id)
    }

    /// Get a session
    ///
    /// # Errors
    ///
    /// Returns an error if session is not found
    pub fn get_session(&self, id: &Uuid) -> AudioPostResult<&PostProductionSession> {
        self.sessions
            .get(id)
            .ok_or(AudioPostError::Generic("Session not found".to_string()))
    }

    /// Get a mutable session
    ///
    /// # Errors
    ///
    /// Returns an error if session is not found
    pub fn get_session_mut(&mut self, id: &Uuid) -> AudioPostResult<&mut PostProductionSession> {
        self.sessions
            .get_mut(id)
            .ok_or(AudioPostError::Generic("Session not found".to_string()))
    }

    /// Close a session
    ///
    /// # Errors
    ///
    /// Returns an error if session is not found
    pub fn close_session(&mut self, id: &Uuid) -> AudioPostResult<PostProductionSession> {
        self.sessions
            .remove(id)
            .ok_or(AudioPostError::Generic("Session not found".to_string()))
    }

    /// Create a backup of a session
    ///
    /// # Errors
    ///
    /// Returns an error if session is not found
    pub fn create_backup(&mut self, session_id: &Uuid, description: &str) -> AudioPostResult<Uuid> {
        let session = self.get_session(session_id)?.clone();
        let backup = SessionBackup::new(session, description);
        let backup_id = backup.id;

        self.backups.entry(*session_id).or_default().push(backup);

        Ok(backup_id)
    }

    /// Get session count
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get all session IDs
    #[must_use]
    pub fn get_session_ids(&self) -> Vec<Uuid> {
        self.sessions.keys().copied().collect()
    }
}

impl Default for SessionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let session = PostProductionSession::new(
            "Test Session",
            std::env::temp_dir().join("oximedia-audiopost-session-test"),
            48000,
            24,
            24.0,
        )
        .expect("operation should succeed");
        assert_eq!(session.name, "Test Session");
        assert_eq!(session.sample_rate, 48000);
    }

    #[test]
    fn test_invalid_sample_rate() {
        assert!(PostProductionSession::new("Test", std::env::temp_dir(), 0, 24, 24.0).is_err());
    }

    #[test]
    fn test_invalid_bit_depth() {
        assert!(PostProductionSession::new("Test", std::env::temp_dir(), 48000, 8, 24.0).is_err());
    }

    #[test]
    fn test_add_track() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let track = Track::new("Dialogue", TrackType::Dialogue, 0);
        let id = session.add_track(track);
        assert_eq!(id, 1);
        assert_eq!(session.track_count(), 1);
    }

    #[test]
    fn test_get_track() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let track = Track::new("Dialogue", TrackType::Dialogue, 0);
        let id = session.add_track(track);
        assert!(session.get_track(id).is_ok());
    }

    #[test]
    fn test_remove_track() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let track = Track::new("Dialogue", TrackType::Dialogue, 0);
        let id = session.add_track(track);
        assert!(session.remove_track(id).is_ok());
        assert_eq!(session.track_count(), 0);
    }

    #[test]
    fn test_track_creation() {
        let track = Track::new("Test Track", TrackType::Music, 0);
        assert_eq!(track.name, "Test Track");
        assert_eq!(track.track_type, TrackType::Music);
    }

    #[test]
    fn test_add_clip_to_track() {
        let mut track = Track::new("Test Track", TrackType::Music, 0);
        let clip = AudioClip::new("Clip 1", PathBuf::from("/audio.wav"), 0.0, 10.0);
        track.add_clip(clip);
        assert_eq!(track.clip_count(), 1);
    }

    #[test]
    fn test_remove_clip() {
        let mut track = Track::new("Test Track", TrackType::Music, 0);
        let clip = AudioClip::new("Clip 1", PathBuf::from("/audio.wav"), 0.0, 10.0);
        let clip_id = clip.id;
        track.add_clip(clip);
        assert!(track.remove_clip(&clip_id).is_ok());
        assert_eq!(track.clip_count(), 0);
    }

    #[test]
    fn test_audio_clip_creation() {
        let clip = AudioClip::new("Test Clip", PathBuf::from("/audio.wav"), 5.0, 10.0);
        assert_eq!(clip.name, "Test Clip");
        assert_eq!(clip.start_time, 5.0);
        assert_eq!(clip.duration(), 10.0);
    }

    #[test]
    fn test_clip_fade_in() {
        let mut clip = AudioClip::new("Test", PathBuf::from("/audio.wav"), 0.0, 10.0);
        clip.set_fade_in(2.0);
        assert_eq!(clip.fade_in, 2.0);
    }

    #[test]
    fn test_clip_fade_out() {
        let mut clip = AudioClip::new("Test", PathBuf::from("/audio.wav"), 0.0, 10.0);
        clip.set_fade_out(2.0);
        assert_eq!(clip.fade_out, 2.0);
    }

    #[test]
    fn test_clip_move_to() {
        let mut clip = AudioClip::new("Test", PathBuf::from("/audio.wav"), 0.0, 10.0);
        clip.move_to(5.0);
        assert_eq!(clip.start_time, 5.0);
        assert_eq!(clip.end_time, 15.0);
    }

    #[test]
    fn test_clip_trim_start() {
        let mut clip = AudioClip::new("Test", PathBuf::from("/audio.wav"), 0.0, 10.0);
        clip.trim_start(2.0);
        assert_eq!(clip.start_time, 2.0);
        assert_eq!(clip.duration(), 8.0);
    }

    #[test]
    fn test_clip_trim_end() {
        let mut clip = AudioClip::new("Test", PathBuf::from("/audio.wav"), 0.0, 10.0);
        clip.trim_end(8.0);
        assert_eq!(clip.end_time, 8.0);
        assert_eq!(clip.duration(), 8.0);
    }

    #[test]
    fn test_marker_creation() {
        let marker = Marker::new("Test Marker", 10.0, MarkerType::Cue);
        assert_eq!(marker.name, "Test Marker");
        assert_eq!(marker.time, 10.0);
    }

    #[test]
    fn test_add_marker() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let marker = Marker::new("Marker 1", 10.0, MarkerType::Cue);
        session.add_marker(marker);
        assert_eq!(session.get_markers().len(), 1);
    }

    #[test]
    fn test_region_creation() {
        let region = Region::new("Test Region", 0.0, 10.0, RegionType::Selection);
        assert_eq!(region.name, "Test Region");
        assert_eq!(region.duration(), 10.0);
    }

    #[test]
    fn test_add_region() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let region = Region::new("Region 1", 0.0, 10.0, RegionType::Loop);
        session.add_region(region);
        assert_eq!(session.get_regions().len(), 1);
    }

    #[test]
    fn test_project_template() {
        let template = ProjectTemplate::new("Test Template", 48000, 24, 24.0);
        assert_eq!(template.name, "Test Template");
    }

    #[test]
    fn test_film_tv_template() {
        let template = ProjectTemplate::film_tv_standard();
        assert!(template.tracks.len() > 0);
    }

    #[test]
    fn test_podcast_template() {
        let template = ProjectTemplate::podcast();
        assert_eq!(template.tracks.len(), 4);
    }

    #[test]
    fn test_music_mixing_template() {
        let template = ProjectTemplate::music_mixing();
        assert_eq!(template.tracks.len(), 8);
    }

    #[test]
    fn test_track_template() {
        let template = TrackTemplate::new("Dialogue", TrackType::Dialogue);
        assert_eq!(template.name, "Dialogue");
    }

    #[test]
    fn test_session_backup() {
        let session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let backup = SessionBackup::new(session, "Test backup");
        assert_eq!(backup.description, "Test backup");
    }

    #[test]
    fn test_session_manager() {
        let manager = SessionManager::new();
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_create_session_in_manager() {
        let mut manager = SessionManager::new();
        let id = manager
            .create_session("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("operation should succeed");
        assert_eq!(manager.session_count(), 1);
        assert!(manager.get_session(&id).is_ok());
    }

    #[test]
    fn test_close_session() {
        let mut manager = SessionManager::new();
        let id = manager
            .create_session("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("operation should succeed");
        assert!(manager.close_session(&id).is_ok());
        assert_eq!(manager.session_count(), 0);
    }

    #[test]
    fn test_create_backup_in_manager() {
        let mut manager = SessionManager::new();
        let id = manager
            .create_session("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("operation should succeed");
        assert!(manager.create_backup(&id, "Backup 1").is_ok());
    }

    #[test]
    fn test_get_session_ids() {
        let mut manager = SessionManager::new();
        manager
            .create_session(
                "Test 1",
                std::env::temp_dir().join("oximedia-audiopost-test1"),
                48000,
                24,
                24.0,
            )
            .expect("operation should succeed");
        manager
            .create_session(
                "Test 2",
                std::env::temp_dir().join("oximedia-audiopost-test2"),
                48000,
                24,
                24.0,
            )
            .expect("operation should succeed");
        assert_eq!(manager.get_session_ids().len(), 2);
    }

    #[test]
    fn test_session_duration() {
        let mut session = PostProductionSession::new("Test", std::env::temp_dir(), 48000, 24, 24.0)
            .expect("failed to create");
        let mut track = Track::new("Test", TrackType::Music, 0);
        track.add_clip(AudioClip::new(
            "Clip",
            PathBuf::from("/audio.wav"),
            0.0,
            10.0,
        ));
        session.add_track(track);
        assert_eq!(session.get_duration(), 10.0);
    }

    #[test]
    fn test_track_type_as_str() {
        assert_eq!(TrackType::Dialogue.as_str(), "Dialogue");
        assert_eq!(TrackType::Music.as_str(), "Music");
    }
}
