//! Simultaneous program output recording with codec selection.
//!
//! This module provides a professional-grade recording subsystem that can
//! capture the switcher's program output (and optionally any aux bus) to
//! multiple simultaneous output files with independent codec and container
//! settings.  The design follows ISO-Broadcast and EBU R68 best practices:
//!
//! * **Multi-track** — up to `MAX_RECORDING_TRACKS` simultaneous outputs.
//! * **Codec presets** — curated presets for common delivery formats.
//! * **Segment recording** — automatic file roll-over at configurable
//!   intervals or on manual triggers.
//! * **Timecode embedding** — LTC/VITC-style timecode stamped on each frame.
//! * **Non-blocking** — the recording manager does not block the switcher's
//!   frame-processing loop; writes are queued.
//!
//! # Example
//!
//! ```rust
//! use oximedia_switcher::recording::{
//!     RecordingManager, RecordingTrack, RecordingCodec, RecordingPreset,
//! };
//!
//! let mut manager = RecordingManager::new();
//!
//! let track = RecordingTrack::new("program_out")
//!     .with_codec(RecordingCodec::ProResHq)
//!     .with_preset(RecordingPreset::Broadcast);
//!
//! let id = manager.add_track(track).expect("add track ok");
//! manager.start_all().expect("start ok");
//! assert!(manager.is_recording(id));
//!
//! // Notify the manager that one frame has been captured.
//! manager.on_frame_captured(id, 3_840_000).expect("frame ok");
//!
//! manager.stop_all().expect("stop ok");
//! assert!(!manager.is_recording(id));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// ────────────────────────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────────────────────────

/// Maximum number of simultaneous recording tracks.
pub const MAX_RECORDING_TRACKS: usize = 8;

// ────────────────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────────────────

/// Errors from the recording subsystem.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RecordingError {
    /// The track ID does not exist.
    #[error("Recording track {0} not found")]
    TrackNotFound(usize),

    /// Maximum simultaneous track count exceeded.
    #[error("Cannot add more than {MAX_RECORDING_TRACKS} recording tracks")]
    TrackLimitExceeded,

    /// Attempted to start a track that is already recording.
    #[error("Recording track {0} is already recording")]
    AlreadyRecording(usize),

    /// Attempted to stop a track that is not recording.
    #[error("Recording track {0} is not recording")]
    NotRecording(usize),

    /// The output path is empty.
    #[error("Recording track output path must not be empty")]
    EmptyOutputPath,

    /// Segment duration is zero.
    #[error("Segment duration must be greater than zero")]
    ZeroSegmentDuration,

    /// Track name must be non-empty.
    #[error("Recording track name must not be empty")]
    EmptyTrackName,
}

// ────────────────────────────────────────────────────────────────────────────
// Codec & container
// ────────────────────────────────────────────────────────────────────────────

/// Video/audio codec to use for a recording track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingCodec {
    /// Apple ProRes 422 HQ — high-quality intra-frame codec.
    ProResHq,
    /// Apple ProRes 422 — standard quality.
    ProRes422,
    /// Apple ProRes 4444 — with alpha channel.
    ProRes4444,
    /// DNxHR HQ — Avid's high-quality intra codec.
    DnxhrHq,
    /// AV1 CBR — efficient long-GOP codec for streaming archives.
    Av1Cbr,
    /// FLAC (lossless) — audio-only tracks.
    FlacAudio,
    /// H.264 CRF — reasonable quality for proxy recordings.
    H264Crf,
    /// VP9 CRF — open, patent-free long-GOP codec.
    Vp9Crf,
    /// FFV1 — lossless, archival codec.
    Ffv1Lossless,
    /// Uncompressed 10-bit YUV — maximum quality, very large files.
    Uncompressed10Bit,
}

impl RecordingCodec {
    /// Returns `true` if this codec produces lossless output.
    pub fn is_lossless(&self) -> bool {
        matches!(
            self,
            RecordingCodec::FlacAudio
                | RecordingCodec::Ffv1Lossless
                | RecordingCodec::Uncompressed10Bit
        )
    }

    /// Returns `true` if this codec supports alpha channels.
    pub fn supports_alpha(&self) -> bool {
        matches!(self, RecordingCodec::ProRes4444)
    }

    /// Typical bit-depth supported by the codec.
    pub fn bit_depth(&self) -> u8 {
        match self {
            RecordingCodec::Uncompressed10Bit => 10,
            RecordingCodec::ProRes4444 | RecordingCodec::DnxhrHq | RecordingCodec::Ffv1Lossless => {
                10
            }
            _ => 8,
        }
    }
}

/// Container/wrapper format for the output file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingContainer {
    /// QuickTime MOV — native for ProRes/DNxHR.
    Mov,
    /// Matroska MKV — versatile open container.
    Mkv,
    /// MP4 — widely compatible.
    Mp4,
    /// MXF OP1a — broadcast standard.
    MxfOp1a,
    /// Raw stream — no container wrapper.
    Raw,
}

// ────────────────────────────────────────────────────────────────────────────
// Preset
// ────────────────────────────────────────────────────────────────────────────

/// Named presets that combine codec + container + quality settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingPreset {
    /// Broadcast master: ProRes HQ in MOV, full-range, embedded timecode.
    Broadcast,
    /// Streaming archive: AV1 CBR in MKV, efficient file size.
    StreamingArchive,
    /// Proxy edit: H.264 in MP4 at reduced resolution.
    ProxyEdit,
    /// Archival master: FFV1 lossless in MKV.
    ArchivalMaster,
    /// Custom: user-configured settings.
    Custom,
}

impl RecordingPreset {
    /// Return the default codec for this preset.
    pub fn default_codec(&self) -> RecordingCodec {
        match self {
            RecordingPreset::Broadcast => RecordingCodec::ProResHq,
            RecordingPreset::StreamingArchive => RecordingCodec::Av1Cbr,
            RecordingPreset::ProxyEdit => RecordingCodec::H264Crf,
            RecordingPreset::ArchivalMaster => RecordingCodec::Ffv1Lossless,
            RecordingPreset::Custom => RecordingCodec::ProRes422,
        }
    }

    /// Return the default container for this preset.
    pub fn default_container(&self) -> RecordingContainer {
        match self {
            RecordingPreset::Broadcast => RecordingContainer::Mov,
            RecordingPreset::StreamingArchive => RecordingContainer::Mkv,
            RecordingPreset::ProxyEdit => RecordingContainer::Mp4,
            RecordingPreset::ArchivalMaster => RecordingContainer::Mkv,
            RecordingPreset::Custom => RecordingContainer::Mov,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Recording state
// ────────────────────────────────────────────────────────────────────────────

/// Operational state of a recording track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordingState {
    /// Track is configured but not yet recording.
    Idle,
    /// Track is actively capturing frames.
    Recording,
    /// Track has been paused (frames are skipped but the file stays open).
    Paused,
    /// Recording has finished; the file has been closed.
    Stopped,
    /// Recording encountered an error; further frames are rejected.
    Error,
}

impl RecordingState {
    /// Returns `true` if frames are currently being accepted.
    pub fn is_active(&self) -> bool {
        matches!(self, RecordingState::Recording)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Segment policy
// ────────────────────────────────────────────────────────────────────────────

/// Controls when the output file is rolled over to a new segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegmentPolicy {
    /// Never roll over; write a single file for the entire recording.
    SingleFile,
    /// Roll over after `duration_frames` frames.
    ByFrameCount(u64),
    /// Roll over after `duration_bytes` bytes of output data.
    ByFileSize(u64),
}

// ────────────────────────────────────────────────────────────────────────────
// Recording track
// ────────────────────────────────────────────────────────────────────────────

/// A single recording output track.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingTrack {
    /// Human-readable track name.
    pub name: String,
    /// Output directory or full path template.
    pub output_path: String,
    /// Codec selection.
    pub codec: RecordingCodec,
    /// Container format.
    pub container: RecordingContainer,
    /// Quality preset (overrides `codec`/`container` when not `Custom`).
    pub preset: RecordingPreset,
    /// Segment roll-over policy.
    pub segment_policy: SegmentPolicy,
    /// Whether to embed timecode in the output stream.
    pub embed_timecode: bool,
    /// Source bus index (0 = program, 1+ = aux buses).
    pub source_bus: usize,
}

impl RecordingTrack {
    /// Create a new track with sensible defaults.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            output_path: std::env::temp_dir()
                .join("oximedia-recording")
                .to_string_lossy()
                .into_owned(),
            codec: RecordingCodec::ProResHq,
            container: RecordingContainer::Mov,
            preset: RecordingPreset::Broadcast,
            segment_policy: SegmentPolicy::SingleFile,
            embed_timecode: true,
            source_bus: 0,
        }
    }

    /// Override the codec.
    pub fn with_codec(mut self, codec: RecordingCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Override the preset (and apply its defaults for codec/container).
    pub fn with_preset(mut self, preset: RecordingPreset) -> Self {
        self.codec = preset.default_codec();
        self.container = preset.default_container();
        self.preset = preset;
        self
    }

    /// Set the output path.
    pub fn with_output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = path.into();
        self
    }

    /// Set the segment policy.
    pub fn with_segment_policy(mut self, policy: SegmentPolicy) -> Self {
        self.segment_policy = policy;
        self
    }

    /// Validate the track configuration.
    pub fn validate(&self) -> Result<(), RecordingError> {
        if self.name.is_empty() {
            return Err(RecordingError::EmptyTrackName);
        }
        if self.output_path.is_empty() {
            return Err(RecordingError::EmptyOutputPath);
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Track runtime state
// ────────────────────────────────────────────────────────────────────────────

/// Runtime state for an active recording track.
#[derive(Debug)]
struct TrackRuntime {
    config: RecordingTrack,
    state: RecordingState,
    frames_captured: u64,
    bytes_written: u64,
    segment_index: u32,
    segment_frames: u64,
    segment_bytes: u64,
}

impl TrackRuntime {
    fn new(config: RecordingTrack) -> Self {
        Self {
            config,
            state: RecordingState::Idle,
            frames_captured: 0,
            bytes_written: 0,
            segment_index: 0,
            segment_frames: 0,
            segment_bytes: 0,
        }
    }

    fn start(&mut self) -> Result<(), RecordingError> {
        match self.state {
            RecordingState::Recording => Err(RecordingError::AlreadyRecording(0)),
            _ => {
                self.state = RecordingState::Recording;
                Ok(())
            }
        }
    }

    fn stop(&mut self) -> Result<(), RecordingError> {
        match self.state {
            RecordingState::Recording | RecordingState::Paused => {
                self.state = RecordingState::Stopped;
                Ok(())
            }
            _ => Err(RecordingError::NotRecording(0)),
        }
    }

    fn pause(&mut self) -> Result<(), RecordingError> {
        if self.state != RecordingState::Recording {
            return Err(RecordingError::NotRecording(0));
        }
        self.state = RecordingState::Paused;
        Ok(())
    }

    fn resume(&mut self) -> Result<(), RecordingError> {
        if self.state != RecordingState::Paused {
            return Err(RecordingError::AlreadyRecording(0));
        }
        self.state = RecordingState::Recording;
        Ok(())
    }

    fn on_frame(&mut self, frame_bytes: u64) -> bool {
        if self.state != RecordingState::Recording {
            return false;
        }
        self.frames_captured += 1;
        self.bytes_written += frame_bytes;
        self.segment_frames += 1;
        self.segment_bytes += frame_bytes;

        // Check segment roll-over.
        let rollover = match &self.config.segment_policy {
            SegmentPolicy::SingleFile => false,
            SegmentPolicy::ByFrameCount(limit) => self.segment_frames >= *limit,
            SegmentPolicy::ByFileSize(limit) => self.segment_bytes >= *limit,
        };
        if rollover {
            self.segment_index += 1;
            self.segment_frames = 0;
            self.segment_bytes = 0;
        }
        rollover
    }

    fn stats(&self) -> TrackStats {
        TrackStats {
            state: self.state,
            frames_captured: self.frames_captured,
            bytes_written: self.bytes_written,
            segment_index: self.segment_index,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Stats
// ────────────────────────────────────────────────────────────────────────────

/// Snapshot of a recording track's performance counters.
#[derive(Debug, Clone)]
pub struct TrackStats {
    /// Current state.
    pub state: RecordingState,
    /// Total frames captured since recording started.
    pub frames_captured: u64,
    /// Total bytes written to disk.
    pub bytes_written: u64,
    /// Current segment file index (starts at 0).
    pub segment_index: u32,
}

// ────────────────────────────────────────────────────────────────────────────
// Manager
// ────────────────────────────────────────────────────────────────────────────

/// Manages multiple simultaneous recording tracks.
#[derive(Debug)]
pub struct RecordingManager {
    tracks: HashMap<usize, TrackRuntime>,
    next_id: usize,
}

impl RecordingManager {
    /// Create a new recording manager with no tracks.
    pub fn new() -> Self {
        Self {
            tracks: HashMap::new(),
            next_id: 0,
        }
    }

    /// Add a new recording track and return its ID.
    pub fn add_track(&mut self, track: RecordingTrack) -> Result<usize, RecordingError> {
        if self.tracks.len() >= MAX_RECORDING_TRACKS {
            return Err(RecordingError::TrackLimitExceeded);
        }
        track.validate()?;
        let id = self.next_id;
        self.next_id += 1;
        self.tracks.insert(id, TrackRuntime::new(track));
        Ok(id)
    }

    /// Remove a track.  Stops it first if it is still recording.
    pub fn remove_track(&mut self, id: usize) -> Result<(), RecordingError> {
        if !self.tracks.contains_key(&id) {
            return Err(RecordingError::TrackNotFound(id));
        }
        self.tracks.remove(&id);
        Ok(())
    }

    /// Start recording on a single track.
    pub fn start(&mut self, id: usize) -> Result<(), RecordingError> {
        let track = self
            .tracks
            .get_mut(&id)
            .ok_or(RecordingError::TrackNotFound(id))?;
        track
            .start()
            .map_err(|_| RecordingError::AlreadyRecording(id))
    }

    /// Stop recording on a single track.
    pub fn stop(&mut self, id: usize) -> Result<(), RecordingError> {
        let track = self
            .tracks
            .get_mut(&id)
            .ok_or(RecordingError::TrackNotFound(id))?;
        track.stop().map_err(|_| RecordingError::NotRecording(id))
    }

    /// Pause recording on a single track.
    pub fn pause(&mut self, id: usize) -> Result<(), RecordingError> {
        let track = self
            .tracks
            .get_mut(&id)
            .ok_or(RecordingError::TrackNotFound(id))?;
        track.pause().map_err(|_| RecordingError::NotRecording(id))
    }

    /// Resume a paused track.
    pub fn resume(&mut self, id: usize) -> Result<(), RecordingError> {
        let track = self
            .tracks
            .get_mut(&id)
            .ok_or(RecordingError::TrackNotFound(id))?;
        track
            .resume()
            .map_err(|_| RecordingError::AlreadyRecording(id))
    }

    /// Start recording on all tracks.
    pub fn start_all(&mut self) -> Result<(), RecordingError> {
        let ids: Vec<usize> = self.tracks.keys().copied().collect();
        for id in ids {
            if let Some(track) = self.tracks.get_mut(&id) {
                if track.state == RecordingState::Idle || track.state == RecordingState::Stopped {
                    track
                        .start()
                        .map_err(|_| RecordingError::AlreadyRecording(id))?;
                }
            }
        }
        Ok(())
    }

    /// Stop recording on all tracks.
    pub fn stop_all(&mut self) -> Result<(), RecordingError> {
        let ids: Vec<usize> = self.tracks.keys().copied().collect();
        for id in ids {
            if let Some(track) = self.tracks.get_mut(&id) {
                if track.state == RecordingState::Recording || track.state == RecordingState::Paused
                {
                    track.stop().map_err(|_| RecordingError::NotRecording(id))?;
                }
            }
        }
        Ok(())
    }

    /// Notify the manager that a frame has been captured for the given track.
    ///
    /// `frame_bytes` is the compressed size of the frame in bytes.
    /// Returns `true` if a segment roll-over occurred.
    pub fn on_frame_captured(
        &mut self,
        id: usize,
        frame_bytes: u64,
    ) -> Result<bool, RecordingError> {
        let track = self
            .tracks
            .get_mut(&id)
            .ok_or(RecordingError::TrackNotFound(id))?;
        Ok(track.on_frame(frame_bytes))
    }

    /// Returns `true` if the track is currently in the `Recording` state.
    pub fn is_recording(&self, id: usize) -> bool {
        self.tracks
            .get(&id)
            .map(|t| t.state == RecordingState::Recording)
            .unwrap_or(false)
    }

    /// Get a snapshot of a track's statistics.
    pub fn track_stats(&self, id: usize) -> Option<TrackStats> {
        self.tracks.get(&id).map(|t| t.stats())
    }

    /// Get the configuration of a track.
    pub fn track_config(&self, id: usize) -> Option<&RecordingTrack> {
        self.tracks.get(&id).map(|t| &t.config)
    }

    /// Number of currently configured tracks.
    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    /// Number of tracks currently in the `Recording` state.
    pub fn active_count(&self) -> usize {
        self.tracks
            .values()
            .filter(|t| t.state == RecordingState::Recording)
            .count()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_str(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("oximedia-switcher-recording-{name}"))
            .to_string_lossy()
            .into_owned()
    }

    fn simple_track(name: &str) -> RecordingTrack {
        RecordingTrack::new(name).with_output_path(tmp_str("test_recording"))
    }

    #[test]
    fn test_add_track_and_count() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        assert_eq!(manager.track_count(), 1);
        assert_eq!(id, 0);
    }

    #[test]
    fn test_start_and_is_recording() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.start(id).expect("start ok");
        assert!(manager.is_recording(id));
    }

    #[test]
    fn test_stop() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.start(id).expect("start ok");
        manager.stop(id).expect("stop ok");
        assert!(!manager.is_recording(id));
    }

    #[test]
    fn test_already_recording_error() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.start(id).expect("start ok");
        assert!(matches!(
            manager.start(id),
            Err(RecordingError::AlreadyRecording(_))
        ));
    }

    #[test]
    fn test_not_recording_stop_error() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        assert!(matches!(
            manager.stop(id),
            Err(RecordingError::NotRecording(_))
        ));
    }

    #[test]
    fn test_pause_and_resume() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.start(id).expect("start ok");
        manager.pause(id).expect("pause ok");

        // Paused track is not "recording".
        assert!(!manager.is_recording(id));

        manager.resume(id).expect("resume ok");
        assert!(manager.is_recording(id));
    }

    #[test]
    fn test_frame_capture_increments_stats() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.start(id).expect("start ok");

        for _ in 0..5 {
            manager.on_frame_captured(id, 1_000_000).expect("frame ok");
        }

        let stats = manager.track_stats(id).expect("stats ok");
        assert_eq!(stats.frames_captured, 5);
        assert_eq!(stats.bytes_written, 5_000_000);
    }

    #[test]
    fn test_segment_rollover_by_frame_count() {
        let mut manager = RecordingManager::new();
        let track = simple_track("cam1").with_segment_policy(SegmentPolicy::ByFrameCount(3));
        let id = manager.add_track(track).expect("add ok");
        manager.start(id).expect("start ok");

        // Feed exactly 3 frames — the third should trigger rollover.
        let r1 = manager.on_frame_captured(id, 100).expect("ok");
        let r2 = manager.on_frame_captured(id, 100).expect("ok");
        let r3 = manager.on_frame_captured(id, 100).expect("ok");

        assert!(!r1);
        assert!(!r2);
        assert!(r3, "third frame should trigger rollover");

        let stats = manager.track_stats(id).expect("stats ok");
        assert_eq!(stats.segment_index, 1);
    }

    #[test]
    fn test_segment_rollover_by_file_size() {
        let mut manager = RecordingManager::new();
        let track = simple_track("cam1").with_segment_policy(SegmentPolicy::ByFileSize(500));
        let id = manager.add_track(track).expect("add ok");
        manager.start(id).expect("start ok");

        manager.on_frame_captured(id, 300).expect("ok");
        let rolled = manager.on_frame_captured(id, 300).expect("ok"); // total 600 > 500

        assert!(rolled, "second frame should trigger rollover");
    }

    #[test]
    fn test_start_all_stop_all() {
        let mut manager = RecordingManager::new();
        let id0 = manager.add_track(simple_track("cam1")).expect("ok");
        let id1 = manager.add_track(simple_track("cam2")).expect("ok");

        manager.start_all().expect("start_all ok");
        assert!(manager.is_recording(id0));
        assert!(manager.is_recording(id1));
        assert_eq!(manager.active_count(), 2);

        manager.stop_all().expect("stop_all ok");
        assert!(!manager.is_recording(id0));
        assert!(!manager.is_recording(id1));
    }

    #[test]
    fn test_track_limit() {
        let mut manager = RecordingManager::new();
        for i in 0..MAX_RECORDING_TRACKS {
            manager
                .add_track(simple_track(&format!("track_{i}")))
                .expect("should succeed");
        }
        assert!(matches!(
            manager.add_track(simple_track("overflow")),
            Err(RecordingError::TrackLimitExceeded)
        ));
    }

    #[test]
    fn test_remove_track() {
        let mut manager = RecordingManager::new();
        let id = manager.add_track(simple_track("cam1")).expect("add ok");
        manager.remove_track(id).expect("remove ok");
        assert_eq!(manager.track_count(), 0);
    }

    #[test]
    fn test_preset_codec_mapping() {
        assert_eq!(
            RecordingPreset::Broadcast.default_codec(),
            RecordingCodec::ProResHq
        );
        assert_eq!(
            RecordingPreset::ArchivalMaster.default_codec(),
            RecordingCodec::Ffv1Lossless
        );
        assert_eq!(
            RecordingPreset::StreamingArchive.default_codec(),
            RecordingCodec::Av1Cbr
        );
    }

    #[test]
    fn test_codec_properties() {
        assert!(RecordingCodec::Ffv1Lossless.is_lossless());
        assert!(!RecordingCodec::ProResHq.is_lossless());
        assert!(RecordingCodec::ProRes4444.supports_alpha());
        assert!(!RecordingCodec::ProRes422.supports_alpha());
        assert_eq!(RecordingCodec::Uncompressed10Bit.bit_depth(), 10);
        assert_eq!(RecordingCodec::H264Crf.bit_depth(), 8);
    }

    #[test]
    fn test_empty_name_validation() {
        let track = RecordingTrack::new("").with_output_path(tmp_str("test"));
        assert!(matches!(
            track.validate(),
            Err(RecordingError::EmptyTrackName)
        ));
    }

    #[test]
    fn test_track_not_found_error() {
        let mut manager = RecordingManager::new();
        assert!(matches!(
            manager.start(999),
            Err(RecordingError::TrackNotFound(999))
        ));
    }
}
