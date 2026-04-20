#![allow(dead_code)]
//! NDI stream recording to local storage for `oximedia-ndi`.
//!
//! Provides configurable recording of incoming NDI streams: segmented
//! recordings by size or duration, naming conventions, and state tracking.
//! Recording sessions can be started, paused, resumed, and stopped.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// RecordingState
// ---------------------------------------------------------------------------

/// The current state of a recording session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordingState {
    /// Idle — no recording in progress.
    Idle,
    /// Actively recording.
    Recording,
    /// Recording is paused (can be resumed).
    Paused,
    /// Recording was stopped.
    Stopped,
    /// An error occurred; recording halted.
    Error,
}

// ---------------------------------------------------------------------------
// SegmentPolicy
// ---------------------------------------------------------------------------

/// Policy that controls when a new segment file is started.
#[derive(Debug, Clone)]
pub enum SegmentPolicy {
    /// Never split — write one continuous file.
    None,
    /// Split after a fixed duration.
    Duration(Duration),
    /// Split after a fixed number of bytes.
    MaxBytes(u64),
    /// Split after a fixed number of frames.
    MaxFrames(u64),
}

impl Default for SegmentPolicy {
    fn default() -> Self {
        Self::None
    }
}

// ---------------------------------------------------------------------------
// RecordingConfig
// ---------------------------------------------------------------------------

/// Configuration for an NDI recording session.
#[derive(Debug, Clone)]
pub struct RecordingConfig {
    /// Base output directory.
    pub output_dir: PathBuf,
    /// File prefix for segment names.
    pub file_prefix: String,
    /// File extension (e.g. `"mkv"`, `"mov"`).
    pub extension: String,
    /// How to split output into segments.
    pub segment_policy: SegmentPolicy,
    /// Maximum total disk usage before recording stops (0 = unlimited).
    pub max_total_bytes: u64,
    /// Whether to include audio in the recording.
    pub record_audio: bool,
    /// Whether to include video in the recording.
    pub record_video: bool,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            file_prefix: "ndi_rec".to_string(),
            extension: "mkv".to_string(),
            segment_policy: SegmentPolicy::None,
            max_total_bytes: 0,
            record_audio: true,
            record_video: true,
        }
    }
}

impl RecordingConfig {
    /// Create a config targeting the given directory.
    pub fn with_dir(dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: dir.into(),
            ..Default::default()
        }
    }

    /// Set the segment policy.
    pub fn segment(mut self, policy: SegmentPolicy) -> Self {
        self.segment_policy = policy;
        self
    }
}

// ---------------------------------------------------------------------------
// SegmentInfo
// ---------------------------------------------------------------------------

/// Metadata for a completed recording segment.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Segment index (0-based).
    pub index: u32,
    /// File path of the segment.
    pub path: PathBuf,
    /// Number of frames written.
    pub frames: u64,
    /// Number of bytes written.
    pub bytes: u64,
    /// Duration of the segment.
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// RecordingStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for a recording session.
#[derive(Debug, Clone)]
pub struct RecordingStats {
    /// Total frames written across all segments.
    pub total_frames: u64,
    /// Total bytes written across all segments.
    pub total_bytes: u64,
    /// Total recording duration (excluding pauses).
    pub total_duration: Duration,
    /// Number of completed segments.
    pub segments_completed: u32,
    /// Number of frames that were dropped during recording.
    pub dropped_frames: u64,
}

impl Default for RecordingStats {
    fn default() -> Self {
        Self {
            total_frames: 0,
            total_bytes: 0,
            total_duration: Duration::ZERO,
            segments_completed: 0,
            dropped_frames: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// RecordingSession
// ---------------------------------------------------------------------------

/// Manages the lifecycle of recording an NDI stream to disk.
///
/// Handles state transitions (start/pause/resume/stop), segment rotation, and
/// statistics.
#[derive(Debug)]
pub struct RecordingSession {
    /// Session configuration.
    config: RecordingConfig,
    /// Current state.
    state: RecordingState,
    /// When the session was started.
    started_at: Option<Instant>,
    /// When the current segment was started.
    segment_start: Option<Instant>,
    /// Accumulated recording time (excluding pauses).
    accumulated_duration: Duration,
    /// When the session was last paused.
    paused_at: Option<Instant>,
    /// Frames written to the current segment.
    segment_frames: u64,
    /// Bytes written to the current segment.
    segment_bytes: u64,
    /// Overall statistics.
    stats: RecordingStats,
    /// Completed segments.
    segments: VecDeque<SegmentInfo>,
    /// Current segment index.
    current_segment_index: u32,
}

impl RecordingSession {
    /// Create a new recording session with the given configuration.
    pub fn new(config: RecordingConfig) -> Self {
        Self {
            config,
            state: RecordingState::Idle,
            started_at: None,
            segment_start: None,
            accumulated_duration: Duration::ZERO,
            paused_at: None,
            segment_frames: 0,
            segment_bytes: 0,
            stats: RecordingStats::default(),
            segments: VecDeque::new(),
            current_segment_index: 0,
        }
    }

    /// Return the current recording state.
    pub fn state(&self) -> RecordingState {
        self.state
    }

    /// Return the recording configuration.
    pub fn config(&self) -> &RecordingConfig {
        &self.config
    }

    /// Start recording. Returns `false` if the session is already recording.
    pub fn start(&mut self) -> bool {
        if self.state != RecordingState::Idle && self.state != RecordingState::Stopped {
            return false;
        }
        let now = Instant::now();
        self.state = RecordingState::Recording;
        self.started_at = Some(now);
        self.segment_start = Some(now);
        self.accumulated_duration = Duration::ZERO;
        self.segment_frames = 0;
        self.segment_bytes = 0;
        self.current_segment_index = 0;
        true
    }

    /// Pause a running recording. Returns `false` if not currently recording.
    pub fn pause(&mut self) -> bool {
        if self.state != RecordingState::Recording {
            return false;
        }
        self.paused_at = Some(Instant::now());
        if let Some(seg_start) = self.segment_start {
            if let Some(paused_at) = self.paused_at {
                self.accumulated_duration += paused_at.duration_since(seg_start);
            }
        }
        self.state = RecordingState::Paused;
        true
    }

    /// Resume a paused recording. Returns `false` if not paused.
    pub fn resume(&mut self) -> bool {
        if self.state != RecordingState::Paused {
            return false;
        }
        self.segment_start = Some(Instant::now());
        self.paused_at = None;
        self.state = RecordingState::Recording;
        true
    }

    /// Stop recording and finalise the current segment.
    pub fn stop(&mut self) -> bool {
        if self.state != RecordingState::Recording && self.state != RecordingState::Paused {
            return false;
        }
        self.finalise_segment();
        self.state = RecordingState::Stopped;
        true
    }

    /// Write a frame to the recording. Returns whether a segment rotation
    /// occurred.
    pub fn write_frame(&mut self, byte_count: u64) -> bool {
        if self.state != RecordingState::Recording {
            self.stats.dropped_frames += 1;
            return false;
        }
        self.segment_frames += 1;
        self.segment_bytes += byte_count;
        self.stats.total_frames += 1;
        self.stats.total_bytes += byte_count;

        if self.should_rotate() {
            self.finalise_segment();
            self.current_segment_index += 1;
            self.segment_start = Some(Instant::now());
            self.segment_frames = 0;
            self.segment_bytes = 0;
            return true;
        }
        false
    }

    /// Record a dropped frame.
    pub fn record_drop(&mut self) {
        self.stats.dropped_frames += 1;
    }

    /// Check whether the current segment should be rotated.
    fn should_rotate(&self) -> bool {
        match &self.config.segment_policy {
            SegmentPolicy::None => false,
            SegmentPolicy::Duration(max_dur) => {
                if let Some(seg_start) = self.segment_start {
                    Instant::now().duration_since(seg_start) >= *max_dur
                } else {
                    false
                }
            }
            SegmentPolicy::MaxBytes(max_bytes) => self.segment_bytes >= *max_bytes,
            SegmentPolicy::MaxFrames(max_frames) => self.segment_frames >= *max_frames,
        }
    }

    /// Finalise the current segment and push it to the completed list.
    fn finalise_segment(&mut self) {
        let duration = if let Some(seg_start) = self.segment_start {
            Instant::now().duration_since(seg_start)
        } else {
            Duration::ZERO
        };

        let path = self.segment_path(self.current_segment_index);
        self.segments.push_back(SegmentInfo {
            index: self.current_segment_index,
            path,
            frames: self.segment_frames,
            bytes: self.segment_bytes,
            duration,
        });
        self.stats.segments_completed += 1;
        self.stats.total_duration += duration;
    }

    /// Build the file path for a segment.
    fn segment_path(&self, index: u32) -> PathBuf {
        let name = format!(
            "{}_{:04}.{}",
            self.config.file_prefix, index, self.config.extension
        );
        self.config.output_dir.join(name)
    }

    /// Return aggregate recording statistics.
    pub fn stats(&self) -> &RecordingStats {
        &self.stats
    }

    /// Return all completed segments.
    pub fn segments(&self) -> &VecDeque<SegmentInfo> {
        &self.segments
    }

    /// Return the path of the most recently completed segment.
    pub fn last_segment_path(&self) -> Option<&Path> {
        self.segments.back().map(|s| s.path.as_path())
    }

    /// Return whether disk-quota has been exceeded.
    pub fn quota_exceeded(&self) -> bool {
        self.config.max_total_bytes > 0 && self.stats.total_bytes >= self.config.max_total_bytes
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_session() -> RecordingSession {
        RecordingSession::new(RecordingConfig::default())
    }

    #[test]
    fn test_initial_state() {
        let s = default_session();
        assert_eq!(s.state(), RecordingState::Idle);
    }

    #[test]
    fn test_start_stop_cycle() {
        let mut s = default_session();
        assert!(s.start());
        assert_eq!(s.state(), RecordingState::Recording);
        assert!(s.stop());
        assert_eq!(s.state(), RecordingState::Stopped);
    }

    #[test]
    fn test_cannot_double_start() {
        let mut s = default_session();
        assert!(s.start());
        assert!(!s.start());
    }

    #[test]
    fn test_pause_resume() {
        let mut s = default_session();
        s.start();
        assert!(s.pause());
        assert_eq!(s.state(), RecordingState::Paused);
        assert!(s.resume());
        assert_eq!(s.state(), RecordingState::Recording);
    }

    #[test]
    fn test_cannot_pause_when_idle() {
        let mut s = default_session();
        assert!(!s.pause());
    }

    #[test]
    fn test_cannot_resume_when_recording() {
        let mut s = default_session();
        s.start();
        assert!(!s.resume());
    }

    #[test]
    fn test_write_frame_updates_stats() {
        let mut s = default_session();
        s.start();
        s.write_frame(1024);
        s.write_frame(2048);
        assert_eq!(s.stats().total_frames, 2);
        assert_eq!(s.stats().total_bytes, 3072);
    }

    #[test]
    fn test_write_frame_while_idle_is_dropped() {
        let mut s = default_session();
        s.write_frame(100);
        assert_eq!(s.stats().total_frames, 0);
        assert_eq!(s.stats().dropped_frames, 1);
    }

    #[test]
    fn test_segment_rotation_by_frames() {
        let config = RecordingConfig {
            segment_policy: SegmentPolicy::MaxFrames(3),
            ..Default::default()
        };
        let mut s = RecordingSession::new(config);
        s.start();
        assert!(!s.write_frame(100));
        assert!(!s.write_frame(100));
        assert!(s.write_frame(100)); // 3rd frame triggers rotation
        assert_eq!(s.stats().segments_completed, 1);
    }

    #[test]
    fn test_segment_rotation_by_bytes() {
        let config = RecordingConfig {
            segment_policy: SegmentPolicy::MaxBytes(200),
            ..Default::default()
        };
        let mut s = RecordingSession::new(config);
        s.start();
        s.write_frame(100);
        assert!(s.write_frame(150)); // total >= 200, rotate
    }

    #[test]
    fn test_segment_path_format() {
        let dir = std::env::temp_dir().join("oximedia-ndi-recordings");
        let config = RecordingConfig {
            output_dir: dir.clone(),
            file_prefix: "stream".to_string(),
            extension: "mov".to_string(),
            ..Default::default()
        };
        let s = RecordingSession::new(config);
        let p = s.segment_path(7);
        assert_eq!(p, dir.join("stream_0007.mov"));
    }

    #[test]
    fn test_stop_creates_final_segment() {
        let mut s = default_session();
        s.start();
        s.write_frame(500);
        s.stop();
        assert_eq!(s.segments().len(), 1);
    }

    #[test]
    fn test_quota_exceeded() {
        let config = RecordingConfig {
            max_total_bytes: 1000,
            ..Default::default()
        };
        let mut s = RecordingSession::new(config);
        s.start();
        s.write_frame(500);
        assert!(!s.quota_exceeded());
        s.write_frame(600);
        assert!(s.quota_exceeded());
    }

    #[test]
    fn test_last_segment_path() {
        let mut s = default_session();
        assert!(s.last_segment_path().is_none());
        s.start();
        s.write_frame(100);
        s.stop();
        assert!(s.last_segment_path().is_some());
    }

    #[test]
    fn test_config_builder() {
        let dir = std::env::temp_dir().join("oximedia-ndi-rec-out");
        let cfg = RecordingConfig::with_dir(dir.clone())
            .segment(SegmentPolicy::MaxFrames(100));
        assert_eq!(cfg.output_dir, dir);
        assert!(matches!(cfg.segment_policy, SegmentPolicy::MaxFrames(100)));
    }
}
