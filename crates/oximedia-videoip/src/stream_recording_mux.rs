//! Stream recording multiplexer: records incoming VideoIP streams to containers.
//!
//! The `StreamRecordingMux` captures live video and audio frames from VideoIP
//! receiver streams and writes them into container formats (MKV/WebM).  This is
//! used for:
//!
//! - **ISO recording**: continuous recording of programme output for compliance
//! - **Highlights / clip capture**: on-demand start/stop recording of any source
//! - **Multi-track recording**: recording multiple camera angles into separate
//!   tracks within a single container
//!
//! # Container support
//!
//! | Format | Extension | Video codecs   | Audio codecs  |
//! |--------|-----------|----------------|---------------|
//! | WebM   | `.webm`   | VP9, AV1       | Opus, Vorbis  |
//! | MKV    | `.mkv`    | VP9, AV1, FFV1 | Opus, PCM     |
//!
//! # Design
//!
//! The muxer uses a **segment-based** architecture:
//! - Frames are grouped into segments of configurable duration
//! - Each segment is independently seekable within the container
//! - Segment boundaries are placed at keyframes to ensure clean cuts
//! - Metadata (timecode, source info) is embedded as container tags

#![allow(dead_code)]

use std::collections::HashMap;

/// Container format for the recording output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerFormat {
    /// Matroska container (`.mkv`), supports VP9/AV1/FFV1 + Opus/PCM.
    Mkv,
    /// WebM container (`.webm`), limited to VP9/AV1 + Opus/Vorbis.
    WebM,
}

impl ContainerFormat {
    /// Returns the standard file extension including the leading dot.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Mkv => ".mkv",
            Self::WebM => ".webm",
        }
    }

    /// Returns `true` if the format supports the given video codec.
    #[must_use]
    pub fn supports_video_codec(&self, codec: VideoCodecId) -> bool {
        match self {
            Self::Mkv => matches!(codec, VideoCodecId::Vp9 | VideoCodecId::Av1 | VideoCodecId::Ffv1),
            Self::WebM => matches!(codec, VideoCodecId::Vp9 | VideoCodecId::Av1),
        }
    }

    /// Returns `true` if the format supports the given audio codec.
    #[must_use]
    pub fn supports_audio_codec(&self, codec: AudioCodecId) -> bool {
        match self {
            Self::Mkv => matches!(
                codec,
                AudioCodecId::Opus | AudioCodecId::Pcm16 | AudioCodecId::Pcm24 | AudioCodecId::Vorbis
            ),
            Self::WebM => matches!(codec, AudioCodecId::Opus | AudioCodecId::Vorbis),
        }
    }
}

/// Video codec identifiers (patent-free only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VideoCodecId {
    /// VP9 (patent-free, Web standard).
    Vp9,
    /// AV1 (patent-free, next-gen).
    Av1,
    /// FFV1 (lossless archival codec).
    Ffv1,
}

/// Audio codec identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioCodecId {
    /// Opus (low-latency, high-quality).
    Opus,
    /// Vorbis (legacy, WebM-compatible).
    Vorbis,
    /// 16-bit PCM (uncompressed).
    Pcm16,
    /// 24-bit PCM (uncompressed, broadcast).
    Pcm24,
}

/// State of the recording session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordingState {
    /// Muxer is configured but not yet recording.
    Idle,
    /// Actively recording frames.
    Recording,
    /// Recording is paused (frames are not written but the file remains open).
    Paused,
    /// Recording has been finalised and the file is closed.
    Finalised,
    /// An error occurred and the recording was aborted.
    Aborted,
}

/// A video track definition within the container.
#[derive(Debug, Clone)]
pub struct VideoTrack {
    /// Track identifier (unique within the mux session).
    pub track_id: u32,
    /// Video codec.
    pub codec: VideoCodecId,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate (frames per second).
    pub frame_rate: f64,
    /// Source stream identifier (maps to VideoIP source).
    pub source_id: String,
}

/// An audio track definition within the container.
#[derive(Debug, Clone)]
pub struct AudioTrack {
    /// Track identifier (unique within the mux session).
    pub track_id: u32,
    /// Audio codec.
    pub codec: AudioCodecId,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u16,
    /// Bit depth (relevant for PCM; ignored for Opus/Vorbis).
    pub bit_depth: u16,
    /// Source stream identifier.
    pub source_id: String,
}

/// Configuration for a recording session.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Output container format.
    pub format: ContainerFormat,
    /// Base file path (without extension; the muxer appends the correct ext).
    pub output_path: String,
    /// Target segment duration in seconds (0 = single segment / no splitting).
    pub segment_duration_secs: f64,
    /// Maximum file size in bytes before rolling to a new file (0 = unlimited).
    pub max_file_size: u64,
    /// Video tracks to record.
    pub video_tracks: Vec<VideoTrack>,
    /// Audio tracks to record.
    pub audio_tracks: Vec<AudioTrack>,
    /// Optional recording title / description embedded as container metadata.
    pub title: Option<String>,
    /// Whether to split files at the `max_file_size` boundary.
    pub auto_split: bool,
}

impl RecordingConfig {
    /// Creates a minimal recording configuration with one video and one audio
    /// track targeting WebM.
    #[must_use]
    pub fn simple_webm(
        output_path: impl Into<String>,
        width: u32,
        height: u32,
        frame_rate: f64,
        sample_rate: u32,
        channels: u16,
    ) -> Self {
        Self {
            format: ContainerFormat::WebM,
            output_path: output_path.into(),
            segment_duration_secs: 0.0,
            max_file_size: 0,
            video_tracks: vec![VideoTrack {
                track_id: 1,
                codec: VideoCodecId::Vp9,
                width,
                height,
                frame_rate,
                source_id: "default".to_owned(),
            }],
            audio_tracks: vec![AudioTrack {
                track_id: 2,
                codec: AudioCodecId::Opus,
                sample_rate,
                channels,
                bit_depth: 16,
                source_id: "default".to_owned(),
            }],
            title: None,
            auto_split: false,
        }
    }

    /// Creates a recording configuration for MKV with FFV1 (lossless video)
    /// and 24-bit PCM audio — suitable for archival.
    #[must_use]
    pub fn archival_mkv(
        output_path: impl Into<String>,
        width: u32,
        height: u32,
        frame_rate: f64,
        sample_rate: u32,
        channels: u16,
    ) -> Self {
        Self {
            format: ContainerFormat::Mkv,
            output_path: output_path.into(),
            segment_duration_secs: 5.0,
            max_file_size: 0,
            video_tracks: vec![VideoTrack {
                track_id: 1,
                codec: VideoCodecId::Ffv1,
                width,
                height,
                frame_rate,
                source_id: "default".to_owned(),
            }],
            audio_tracks: vec![AudioTrack {
                track_id: 2,
                codec: AudioCodecId::Pcm24,
                sample_rate,
                channels,
                bit_depth: 24,
                source_id: "default".to_owned(),
            }],
            title: None,
            auto_split: false,
        }
    }
}

/// Error type for recording operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RecordingError {
    /// The container format does not support the requested codec.
    #[error("codec {codec:?} not supported by {format:?}")]
    UnsupportedCodec {
        /// The codec that was rejected.
        codec: String,
        /// The container format.
        format: ContainerFormat,
    },
    /// Invalid track configuration (e.g., zero dimensions).
    #[error("invalid track: {0}")]
    InvalidTrack(String),
    /// Recording is not in a valid state for the requested operation.
    #[error("invalid state: expected {expected:?}, got {actual:?}")]
    InvalidState {
        /// The expected state.
        expected: RecordingState,
        /// The actual state.
        actual: RecordingState,
    },
    /// Duplicate track ID.
    #[error("duplicate track ID {0}")]
    DuplicateTrack(u32),
    /// I/O error description.
    #[error("I/O error: {0}")]
    Io(String),
    /// The file size limit has been reached.
    #[error("file size limit reached ({limit} bytes)")]
    FileSizeLimit {
        /// The configured limit.
        limit: u64,
    },
}

/// Result type for recording operations.
pub type RecordingResult<T> = Result<T, RecordingError>;

/// Statistics about the current recording session.
#[derive(Debug, Clone, Default)]
pub struct RecordingStats {
    /// Total video frames written across all tracks.
    pub video_frames_written: u64,
    /// Total audio frames (blocks) written across all tracks.
    pub audio_frames_written: u64,
    /// Total bytes written to the output (all segments combined).
    pub bytes_written: u64,
    /// Number of segments completed.
    pub segments_completed: u32,
    /// Number of keyframes encountered.
    pub keyframes_seen: u64,
    /// Duration of the recording in seconds.
    pub duration_secs: f64,
    /// Number of dropped frames (arrived too late or out of order).
    pub dropped_frames: u64,
    /// Number of output files produced (for auto-split mode).
    pub files_produced: u32,
}

/// A frame submitted for recording.
#[derive(Debug, Clone)]
pub struct MuxFrame {
    /// Track ID this frame belongs to.
    pub track_id: u32,
    /// Presentation timestamp in microseconds (relative to recording start).
    pub pts_us: i64,
    /// `true` if this is a keyframe (video) or the start of a new audio block.
    pub is_keyframe: bool,
    /// Encoded frame data (codec-compressed bytes).
    pub data: Vec<u8>,
}

/// Per-track bookkeeping within the muxer.
#[derive(Debug, Clone)]
struct TrackState {
    /// Track kind.
    kind: TrackKind,
    /// Number of frames written.
    frames: u64,
    /// Last PTS seen (for ordering checks).
    last_pts_us: i64,
    /// Total bytes written for this track.
    bytes: u64,
}

/// Discriminator for track type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrackKind {
    Video,
    Audio,
}

/// Internal segment tracking.
#[derive(Debug, Clone)]
struct Segment {
    /// Segment index (0-based).
    index: u32,
    /// Start PTS of the segment in microseconds.
    start_pts_us: i64,
    /// Number of frames in this segment.
    frame_count: u64,
    /// Byte size of this segment.
    byte_size: u64,
}

/// Stream recording multiplexer.
///
/// Accepts encoded video and audio frames and writes them into a container
/// (MKV or WebM).  The muxer tracks per-track statistics, manages segments,
/// and validates codec compatibility.
#[derive(Debug)]
pub struct StreamRecordingMux {
    /// Configuration.
    config: RecordingConfig,
    /// Current state.
    state: RecordingState,
    /// Per-track state indexed by track ID.
    tracks: HashMap<u32, TrackState>,
    /// Accumulated statistics.
    stats: RecordingStats,
    /// Current segment.
    current_segment: Segment,
    /// Completed segments.
    completed_segments: Vec<Segment>,
    /// Output buffer (simulates file I/O for the muxer layer).
    output_buffer: Vec<u8>,
    /// Timestamp of recording start (first frame PTS).
    start_pts_us: Option<i64>,
}

impl StreamRecordingMux {
    /// Creates a new muxer from the given configuration.
    ///
    /// Validates that all codecs are compatible with the chosen container format
    /// and that track IDs are unique.
    pub fn new(config: RecordingConfig) -> RecordingResult<Self> {
        // Validate video tracks.
        let mut seen_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for vt in &config.video_tracks {
            if !seen_ids.insert(vt.track_id) {
                return Err(RecordingError::DuplicateTrack(vt.track_id));
            }
            if vt.width == 0 || vt.height == 0 {
                return Err(RecordingError::InvalidTrack(format!(
                    "video track {} has zero dimensions",
                    vt.track_id
                )));
            }
            if vt.frame_rate <= 0.0 {
                return Err(RecordingError::InvalidTrack(format!(
                    "video track {} has invalid frame rate {}",
                    vt.track_id, vt.frame_rate
                )));
            }
            if !config.format.supports_video_codec(vt.codec) {
                return Err(RecordingError::UnsupportedCodec {
                    codec: format!("{:?}", vt.codec),
                    format: config.format,
                });
            }
        }
        // Validate audio tracks.
        for at in &config.audio_tracks {
            if !seen_ids.insert(at.track_id) {
                return Err(RecordingError::DuplicateTrack(at.track_id));
            }
            if at.sample_rate == 0 || at.channels == 0 {
                return Err(RecordingError::InvalidTrack(format!(
                    "audio track {} has invalid sample_rate={} channels={}",
                    at.track_id, at.sample_rate, at.channels
                )));
            }
            if !config.format.supports_audio_codec(at.codec) {
                return Err(RecordingError::UnsupportedCodec {
                    codec: format!("{:?}", at.codec),
                    format: config.format,
                });
            }
        }

        let mut tracks = HashMap::new();
        for vt in &config.video_tracks {
            tracks.insert(
                vt.track_id,
                TrackState {
                    kind: TrackKind::Video,
                    frames: 0,
                    last_pts_us: i64::MIN,
                    bytes: 0,
                },
            );
        }
        for at in &config.audio_tracks {
            tracks.insert(
                at.track_id,
                TrackState {
                    kind: TrackKind::Audio,
                    frames: 0,
                    last_pts_us: i64::MIN,
                    bytes: 0,
                },
            );
        }

        Ok(Self {
            config,
            state: RecordingState::Idle,
            tracks,
            stats: RecordingStats::default(),
            current_segment: Segment {
                index: 0,
                start_pts_us: 0,
                frame_count: 0,
                byte_size: 0,
            },
            completed_segments: Vec::new(),
            output_buffer: Vec::new(),
            start_pts_us: None,
        })
    }

    /// Starts the recording session.
    pub fn start(&mut self) -> RecordingResult<()> {
        if self.state != RecordingState::Idle {
            return Err(RecordingError::InvalidState {
                expected: RecordingState::Idle,
                actual: self.state,
            });
        }
        self.state = RecordingState::Recording;
        self.stats.files_produced = 1;
        // Write container header bytes (simplified placeholder).
        let header = self.build_header();
        self.output_buffer.extend_from_slice(&header);
        self.stats.bytes_written += header.len() as u64;
        Ok(())
    }

    /// Pauses recording. Frames submitted while paused are dropped silently.
    pub fn pause(&mut self) -> RecordingResult<()> {
        if self.state != RecordingState::Recording {
            return Err(RecordingError::InvalidState {
                expected: RecordingState::Recording,
                actual: self.state,
            });
        }
        self.state = RecordingState::Paused;
        Ok(())
    }

    /// Resumes recording after a pause.
    pub fn resume(&mut self) -> RecordingResult<()> {
        if self.state != RecordingState::Paused {
            return Err(RecordingError::InvalidState {
                expected: RecordingState::Paused,
                actual: self.state,
            });
        }
        self.state = RecordingState::Recording;
        Ok(())
    }

    /// Writes a frame to the recording.
    pub fn write_frame(&mut self, frame: &MuxFrame) -> RecordingResult<()> {
        match self.state {
            RecordingState::Recording => {}
            RecordingState::Paused => {
                self.stats.dropped_frames += 1;
                return Ok(());
            }
            other => {
                return Err(RecordingError::InvalidState {
                    expected: RecordingState::Recording,
                    actual: other,
                });
            }
        }

        // Validate track exists (without holding a mutable borrow on self).
        if !self.tracks.contains_key(&frame.track_id) {
            return Err(RecordingError::InvalidTrack(format!(
                "unknown track ID {}",
                frame.track_id
            )));
        }

        // Check file size limit.
        if self.config.max_file_size > 0
            && self.stats.bytes_written + frame.data.len() as u64 > self.config.max_file_size
        {
            if self.config.auto_split {
                self.roll_file()?;
            } else {
                return Err(RecordingError::FileSizeLimit {
                    limit: self.config.max_file_size,
                });
            }
        }

        // Set start PTS on first frame.
        if self.start_pts_us.is_none() {
            self.start_pts_us = Some(frame.pts_us);
            self.current_segment.start_pts_us = frame.pts_us;
        }

        // Check for segment boundary.
        if self.config.segment_duration_secs > 0.0 && frame.is_keyframe {
            let relative_pts = frame.pts_us - self.current_segment.start_pts_us;
            let segment_dur_us = (self.config.segment_duration_secs * 1_000_000.0) as i64;
            if relative_pts >= segment_dur_us && self.current_segment.frame_count > 0 {
                self.finish_segment(frame.pts_us);
            }
        }

        // Write frame data to buffer.
        let frame_header = self.encode_frame_header(frame);
        self.output_buffer.extend_from_slice(&frame_header);
        self.output_buffer.extend_from_slice(&frame.data);

        let total_bytes = (frame_header.len() + frame.data.len()) as u64;

        // Now get mutable track reference for updates.
        if let Some(track) = self.tracks.get_mut(&frame.track_id) {
            track.frames += 1;
            track.last_pts_us = frame.pts_us;
            track.bytes += total_bytes;

            match track.kind {
                TrackKind::Video => {
                    self.stats.video_frames_written += 1;
                    if frame.is_keyframe {
                        self.stats.keyframes_seen += 1;
                    }
                }
                TrackKind::Audio => {
                    self.stats.audio_frames_written += 1;
                }
            }
        }

        self.stats.bytes_written += total_bytes;
        self.current_segment.frame_count += 1;
        self.current_segment.byte_size += total_bytes;

        // Update duration.
        if let Some(start) = self.start_pts_us {
            self.stats.duration_secs = (frame.pts_us - start) as f64 / 1_000_000.0;
        }

        Ok(())
    }

    /// Finalises the recording, flushes all buffers, and writes the container
    /// footer / seek index.
    pub fn finalise(&mut self) -> RecordingResult<()> {
        if !matches!(self.state, RecordingState::Recording | RecordingState::Paused) {
            return Err(RecordingError::InvalidState {
                expected: RecordingState::Recording,
                actual: self.state,
            });
        }

        // Finalise the last segment.
        let last_pts = self.tracks.values().map(|t| t.last_pts_us).max().unwrap_or(0);
        self.finish_segment(last_pts);

        // Write container footer (seek index placeholder).
        let footer = self.build_footer();
        self.output_buffer.extend_from_slice(&footer);
        self.stats.bytes_written += footer.len() as u64;

        self.state = RecordingState::Finalised;
        Ok(())
    }

    /// Aborts the recording without writing a proper footer.
    pub fn abort(&mut self) {
        self.state = RecordingState::Aborted;
    }

    /// Returns the current recording state.
    #[must_use]
    pub fn state(&self) -> RecordingState {
        self.state
    }

    /// Returns a snapshot of the recording statistics.
    #[must_use]
    pub fn stats(&self) -> &RecordingStats {
        &self.stats
    }

    /// Returns the output buffer contents (for testing / in-memory muxing).
    #[must_use]
    pub fn output_data(&self) -> &[u8] {
        &self.output_buffer
    }

    /// Returns the number of completed segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.completed_segments.len()
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &RecordingConfig {
        &self.config
    }

    /// Returns the output file path with the correct extension.
    #[must_use]
    pub fn output_path_with_ext(&self) -> String {
        format!("{}{}", self.config.output_path, self.config.format.extension())
    }

    /// Returns per-track frame counts as `(track_id, frame_count)` pairs.
    #[must_use]
    pub fn track_frame_counts(&self) -> Vec<(u32, u64)> {
        let mut counts: Vec<(u32, u64)> = self.tracks.iter().map(|(&id, ts)| (id, ts.frames)).collect();
        counts.sort_by_key(|&(id, _)| id);
        counts
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn finish_segment(&mut self, next_start_pts_us: i64) {
        if self.current_segment.frame_count == 0 {
            return;
        }
        self.stats.segments_completed += 1;
        self.completed_segments.push(self.current_segment.clone());
        self.current_segment = Segment {
            index: self.completed_segments.len() as u32,
            start_pts_us: next_start_pts_us,
            frame_count: 0,
            byte_size: 0,
        };
    }

    fn roll_file(&mut self) -> RecordingResult<()> {
        // In a real implementation this would close the current file and open
        // a new one with an incremented suffix.  Here we just reset the byte
        // counter and segment state.
        let last_pts = self.tracks.values().map(|t| t.last_pts_us).max().unwrap_or(0);
        self.finish_segment(last_pts);
        self.stats.files_produced += 1;
        // Reset per-file tracking but keep cumulative stats.
        self.current_segment = Segment {
            index: 0,
            start_pts_us: last_pts,
            frame_count: 0,
            byte_size: 0,
        };
        Ok(())
    }

    fn build_header(&self) -> Vec<u8> {
        // Simplified EBML-like header placeholder.  A real implementation would
        // emit proper Matroska/WebM EBML elements.
        let mut hdr = Vec::with_capacity(64);
        // Magic bytes.
        match self.config.format {
            ContainerFormat::Mkv => hdr.extend_from_slice(b"\x1a\x45\xdf\xa3MKV"),
            ContainerFormat::WebM => hdr.extend_from_slice(b"\x1a\x45\xdf\xa3WEBM"),
        }
        // Track count (1 byte video + 1 byte audio).
        hdr.push(self.config.video_tracks.len() as u8);
        hdr.push(self.config.audio_tracks.len() as u8);
        hdr
    }

    fn build_footer(&self) -> Vec<u8> {
        // Simplified seek-index placeholder.
        let mut footer = Vec::with_capacity(32);
        footer.extend_from_slice(b"SEEK");
        let seg_count = self.completed_segments.len() as u32;
        footer.extend_from_slice(&seg_count.to_le_bytes());
        footer
    }

    fn encode_frame_header(&self, frame: &MuxFrame) -> Vec<u8> {
        // Minimal frame header: track_id (4) + pts_us (8) + flags (1) + data_len (4) = 17 bytes.
        let mut hdr = Vec::with_capacity(17);
        hdr.extend_from_slice(&frame.track_id.to_le_bytes());
        hdr.extend_from_slice(&frame.pts_us.to_le_bytes());
        let flags: u8 = if frame.is_keyframe { 0x01 } else { 0x00 };
        hdr.push(flags);
        hdr.extend_from_slice(&(frame.data.len() as u32).to_le_bytes());
        hdr
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config() -> RecordingConfig {
        RecordingConfig::simple_webm("/tmp/test_recording", 1920, 1080, 30.0, 48000, 2)
    }

    fn make_video_frame(track_id: u32, pts_us: i64, keyframe: bool) -> MuxFrame {
        MuxFrame {
            track_id,
            pts_us,
            is_keyframe: keyframe,
            data: vec![0xAA; 4096], // simulated compressed frame
        }
    }

    fn make_audio_frame(track_id: u32, pts_us: i64) -> MuxFrame {
        MuxFrame {
            track_id,
            pts_us,
            is_keyframe: false,
            data: vec![0xBB; 960], // simulated Opus packet
        }
    }

    #[test]
    fn test_create_muxer_simple_webm() {
        let mux = StreamRecordingMux::new(simple_config());
        assert!(mux.is_ok());
        let mux = mux.expect("should create mux");
        assert_eq!(mux.state(), RecordingState::Idle);
        assert!(mux.output_path_with_ext().ends_with(".webm"));
    }

    #[test]
    fn test_create_muxer_archival_mkv() {
        let config = RecordingConfig::archival_mkv("/tmp/archive", 3840, 2160, 25.0, 96000, 8);
        let mux = StreamRecordingMux::new(config);
        assert!(mux.is_ok());
        let mux = mux.expect("should create mux");
        assert!(mux.output_path_with_ext().ends_with(".mkv"));
    }

    #[test]
    fn test_unsupported_codec_rejected() {
        let mut config = simple_config();
        // FFV1 is not supported in WebM.
        config.video_tracks[0].codec = VideoCodecId::Ffv1;
        let result = StreamRecordingMux::new(config);
        assert!(matches!(result, Err(RecordingError::UnsupportedCodec { .. })));
    }

    #[test]
    fn test_duplicate_track_id_rejected() {
        let mut config = simple_config();
        // Both video and audio have track_id=1 → conflict.
        config.audio_tracks[0].track_id = 1;
        let result = StreamRecordingMux::new(config);
        assert!(matches!(result, Err(RecordingError::DuplicateTrack(1))));
    }

    #[test]
    fn test_zero_dimension_video_rejected() {
        let mut config = simple_config();
        config.video_tracks[0].width = 0;
        let result = StreamRecordingMux::new(config);
        assert!(matches!(result, Err(RecordingError::InvalidTrack(_))));
    }

    #[test]
    fn test_start_and_write_frames() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        assert_eq!(mux.state(), RecordingState::Recording);

        // Write 3 video frames and 3 audio frames.
        for i in 0..3 {
            let pts = i * 33_333; // ~30fps
            mux.write_frame(&make_video_frame(1, pts, i == 0)).expect("video write ok");
            mux.write_frame(&make_audio_frame(2, pts)).expect("audio write ok");
        }

        assert_eq!(mux.stats().video_frames_written, 3);
        assert_eq!(mux.stats().audio_frames_written, 3);
        assert_eq!(mux.stats().keyframes_seen, 1);
        assert!(mux.stats().bytes_written > 0);
    }

    #[test]
    fn test_pause_drops_frames() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        mux.write_frame(&make_video_frame(1, 0, true)).expect("ok");
        mux.pause().expect("pause ok");
        assert_eq!(mux.state(), RecordingState::Paused);

        // Frame while paused should be dropped.
        mux.write_frame(&make_video_frame(1, 33_333, false)).expect("ok");
        assert_eq!(mux.stats().dropped_frames, 1);
        assert_eq!(mux.stats().video_frames_written, 1); // only the pre-pause frame

        mux.resume().expect("resume ok");
        mux.write_frame(&make_video_frame(1, 66_666, false)).expect("ok");
        assert_eq!(mux.stats().video_frames_written, 2);
    }

    #[test]
    fn test_finalise_writes_footer() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        mux.write_frame(&make_video_frame(1, 0, true)).expect("ok");
        let bytes_before = mux.stats().bytes_written;
        mux.finalise().expect("finalise ok");
        assert_eq!(mux.state(), RecordingState::Finalised);
        // Footer should add some bytes.
        assert!(mux.stats().bytes_written > bytes_before);
    }

    #[test]
    fn test_segment_splitting() {
        let mut config = simple_config();
        config.segment_duration_secs = 1.0; // 1 second segments
        let mut mux = StreamRecordingMux::new(config).expect("config ok");
        mux.start().expect("start ok");

        // Write frames spanning 4 seconds at 25fps with keyframe every second.
        // Use 25fps (40_000 us/frame) so keyframes land exactly on 1s boundaries.
        for i in 0..100 {
            let pts = i * 40_000_i64; // 25fps → 4 seconds total
            let keyframe = i % 25 == 0; // keyframe every 25 frames (every 1s)
            mux.write_frame(&make_video_frame(1, pts, keyframe)).expect("write ok");
        }

        // Keyframes at pts 0, 1_000_000, 2_000_000, 3_000_000.
        // Segments finish at 2nd, 3rd, 4th keyframes -> at least 2 completed segments.
        assert!(
            mux.segment_count() >= 2,
            "expected >=2 segments, got {}",
            mux.segment_count()
        );
    }

    #[test]
    fn test_file_size_limit_without_auto_split() {
        let mut config = simple_config();
        config.max_file_size = 5000; // very small limit
        let mut mux = StreamRecordingMux::new(config).expect("config ok");
        mux.start().expect("start ok");

        // First frame should fit.
        mux.write_frame(&make_video_frame(1, 0, true)).expect("ok");
        // Eventually exceed the limit.
        let result = mux.write_frame(&make_video_frame(1, 33_333, false));
        // Should fail because auto_split is false.
        assert!(
            matches!(result, Err(RecordingError::FileSizeLimit { .. })),
            "expected FileSizeLimit error, got {:?}",
            result
        );
    }

    #[test]
    fn test_file_size_limit_with_auto_split() {
        let mut config = simple_config();
        config.max_file_size = 5000;
        config.auto_split = true;
        let mut mux = StreamRecordingMux::new(config).expect("config ok");
        mux.start().expect("start ok");

        // Write several frames; auto-split should handle the limit.
        for i in 0..5 {
            let pts = i * 33_333;
            let result = mux.write_frame(&make_video_frame(1, pts, i == 0));
            assert!(result.is_ok(), "frame {i} should succeed with auto-split");
        }

        assert!(mux.stats().files_produced >= 2, "should have split to multiple files");
    }

    #[test]
    fn test_abort_sets_state() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        mux.abort();
        assert_eq!(mux.state(), RecordingState::Aborted);
    }

    #[test]
    fn test_container_format_codec_support() {
        // WebM rejects FFV1.
        assert!(!ContainerFormat::WebM.supports_video_codec(VideoCodecId::Ffv1));
        assert!(ContainerFormat::WebM.supports_video_codec(VideoCodecId::Vp9));
        assert!(ContainerFormat::WebM.supports_video_codec(VideoCodecId::Av1));
        // WebM rejects PCM.
        assert!(!ContainerFormat::WebM.supports_audio_codec(AudioCodecId::Pcm16));
        // MKV supports everything.
        assert!(ContainerFormat::Mkv.supports_video_codec(VideoCodecId::Ffv1));
        assert!(ContainerFormat::Mkv.supports_audio_codec(AudioCodecId::Pcm24));
    }

    #[test]
    fn test_track_frame_counts() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        for i in 0..5 {
            mux.write_frame(&make_video_frame(1, i * 33_333, i == 0)).expect("ok");
        }
        for i in 0..3 {
            mux.write_frame(&make_audio_frame(2, i * 33_333)).expect("ok");
        }
        let counts = mux.track_frame_counts();
        assert_eq!(counts.len(), 2);
        // Track 1 (video) = 5, Track 2 (audio) = 3.
        let video_count = counts.iter().find(|&&(id, _)| id == 1).map(|&(_, c)| c);
        let audio_count = counts.iter().find(|&&(id, _)| id == 2).map(|&(_, c)| c);
        assert_eq!(video_count, Some(5));
        assert_eq!(audio_count, Some(3));
    }

    #[test]
    fn test_invalid_state_transitions() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        // Can't pause before starting.
        assert!(mux.pause().is_err());
        // Can't resume before pausing.
        assert!(mux.resume().is_err());
        // Can't finalise before starting.
        assert!(mux.finalise().is_err());
    }

    #[test]
    fn test_unknown_track_id_rejected() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        let result = mux.write_frame(&make_video_frame(99, 0, true));
        assert!(matches!(result, Err(RecordingError::InvalidTrack(_))));
    }

    #[test]
    fn test_duration_tracking() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        mux.write_frame(&make_video_frame(1, 0, true)).expect("ok");
        mux.write_frame(&make_video_frame(1, 2_000_000, false)).expect("ok"); // 2 seconds later
        let dur = mux.stats().duration_secs;
        assert!(
            (dur - 2.0).abs() < 0.01,
            "expected ~2.0s duration, got {dur}"
        );
    }

    #[test]
    fn test_pcm24_in_webm_rejected() {
        let mut config = simple_config();
        config.audio_tracks[0].codec = AudioCodecId::Pcm24;
        let result = StreamRecordingMux::new(config);
        assert!(matches!(result, Err(RecordingError::UnsupportedCodec { .. })));
    }

    #[test]
    fn test_output_data_contains_header() {
        let mut mux = StreamRecordingMux::new(simple_config()).expect("config ok");
        mux.start().expect("start ok");
        let data = mux.output_data();
        // Should start with EBML magic.
        assert!(data.len() >= 4);
        assert_eq!(&data[0..4], b"\x1a\x45\xdf\xa3");
    }
}
