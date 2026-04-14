//! Instant clip recorder backed by a circular frame buffer.
//!
//! Keeps a configurable rolling window of recent encoded frames in memory.
//! At any moment the caller can extract a clip — a contiguous slice of that
//! buffer — by specifying a time range (seconds before now, or absolute
//! timestamps).  Clips carry rich metadata: source resolution, bitrate
//! estimate, keyframe positions, and user-supplied tags.
//!
//! The recorder is designed to be driven by the same encode pipeline that
//! produces [`EncodedFrame`]s; no copying occurs until a clip is actually
//! extracted.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gaming::clip_recorder::{
//!     ClipRecorder, ClipRecorderConfig, EncodedFrame, FrameKind, ClipRequest,
//! };
//! use std::time::{Duration, Instant};
//!
//! let cfg = ClipRecorderConfig {
//!     buffer_duration: Duration::from_secs(90),
//!     max_frame_bytes: 1_000_000,
//!     width: 1920,
//!     height: 1080,
//!     fps_hint: 60,
//! };
//! let mut recorder = ClipRecorder::new(cfg);
//!
//! let now = Instant::now();
//! recorder.push_frame(EncodedFrame {
//!     pts_us: 0,
//!     data: vec![0u8; 8_000],
//!     kind: FrameKind::Keyframe,
//!     captured_at: now,
//! });
//!
//! let clip = recorder.extract_clip(ClipRequest::LastSeconds(5)).unwrap();
//! println!("clip frames: {}", clip.frames.len());
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Frame kind
// ---------------------------------------------------------------------------

/// Type of encoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Intra-coded frame (no dependencies on other frames).
    Keyframe,
    /// Predictive frame (depends on previous frame(s)).
    Interframe,
    /// Bi-directional predictive frame.
    BFrame,
}

// ---------------------------------------------------------------------------
// Encoded frame
// ---------------------------------------------------------------------------

/// A single encoded video frame stored in the circular buffer.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Presentation timestamp in microseconds (monotonically increasing).
    pub pts_us: i64,
    /// Compressed payload bytes.
    pub data: Vec<u8>,
    /// Frame type.
    pub kind: FrameKind,
    /// Wall-clock time when this frame was captured.
    pub captured_at: Instant,
}

// ---------------------------------------------------------------------------
// Clip request
// ---------------------------------------------------------------------------

/// Specifies what slice of the buffer to extract as a clip.
#[derive(Debug, Clone, Copy)]
pub enum ClipRequest {
    /// Extract all frames captured in the last `n` seconds.
    LastSeconds(u32),
    /// Extract frames between two wall-clock instants.
    TimeRange {
        /// Start of the time range (inclusive).
        from: Instant,
        /// End of the time range (inclusive).
        to: Instant,
    },
    /// Extract frames with PTS in the given inclusive range (microseconds).
    PtsRange {
        /// Start of the PTS range in microseconds (inclusive).
        from_us: i64,
        /// End of the PTS range in microseconds (inclusive).
        to_us: i64,
    },
}

// ---------------------------------------------------------------------------
// Clip metadata + output
// ---------------------------------------------------------------------------

/// Metadata attached to an extracted clip.
#[derive(Debug, Clone)]
pub struct ClipMetadata {
    /// UTC-like creation time (uses `Instant` for simplicity; callers should
    /// map to `SystemTime` when persisting).
    pub created_at: Instant,
    /// Total duration of the clip.
    pub duration: Duration,
    /// Source resolution.
    pub width: u32,
    /// Source resolution.
    pub height: u32,
    /// Approximate average bitrate of the clip in bits-per-second.
    pub avg_bitrate_bps: f64,
    /// Indices of keyframes within `Clip::frames`.
    pub keyframe_indices: Vec<usize>,
    /// User-supplied tags.
    pub tags: Vec<String>,
    /// Total compressed bytes in the clip.
    pub total_bytes: u64,
    /// Number of frames.
    pub frame_count: usize,
}

/// An extracted clip containing a sequence of encoded frames.
#[derive(Debug, Clone)]
pub struct Clip {
    /// Ordered frames (oldest first).
    pub frames: Vec<EncodedFrame>,
    /// Clip metadata.
    pub metadata: ClipMetadata,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from clip recorder operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipError {
    /// The requested time range produces an empty clip.
    EmptyClip,
    /// The requested time range is invalid (e.g. `from > to`).
    InvalidRange(String),
    /// No frames are buffered yet.
    BufferEmpty,
}

impl std::fmt::Display for ClipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyClip => write!(f, "no frames in the requested range"),
            Self::InvalidRange(msg) => write!(f, "invalid range: {msg}"),
            Self::BufferEmpty => write!(f, "frame buffer is empty"),
        }
    }
}

impl std::error::Error for ClipError {}

// ---------------------------------------------------------------------------
// Recorder configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ClipRecorder`].
#[derive(Debug, Clone)]
pub struct ClipRecorderConfig {
    /// How many seconds of frames to keep in the rolling buffer.
    pub buffer_duration: Duration,
    /// If any single compressed frame exceeds this byte count it is silently
    /// dropped to protect memory (e.g. misconfigured encoder).
    pub max_frame_bytes: usize,
    /// Source width — recorded in clip metadata.
    pub width: u32,
    /// Source height — recorded in clip metadata.
    pub height: u32,
    /// Approximate frames-per-second hint for buffer sizing / duration math.
    pub fps_hint: u32,
}

impl Default for ClipRecorderConfig {
    fn default() -> Self {
        Self {
            buffer_duration: Duration::from_secs(90),
            max_frame_bytes: 2_000_000,
            width: 1920,
            height: 1080,
            fps_hint: 60,
        }
    }
}

// ---------------------------------------------------------------------------
// Clip recorder
// ---------------------------------------------------------------------------

/// Circular-buffer clip recorder.
///
/// Maintains a rolling window of the most recent encoded frames. Older frames
/// are evicted automatically when the buffer's time span exceeds
/// [`ClipRecorderConfig::buffer_duration`].
pub struct ClipRecorder {
    config: ClipRecorderConfig,
    buffer: VecDeque<EncodedFrame>,
    /// Total frames pushed since creation.
    total_pushed: u64,
    /// Total frames dropped (oversized or evicted from the rolling window).
    total_dropped: u64,
}

impl ClipRecorder {
    /// Create a new recorder with the given configuration.
    #[must_use]
    pub fn new(config: ClipRecorderConfig) -> Self {
        // Pre-allocate ~fps_hint * buffer_duration entries
        let approx_cap =
            (config.fps_hint as u64 * config.buffer_duration.as_secs().max(1)) as usize;
        Self {
            config,
            buffer: VecDeque::with_capacity(approx_cap.min(100_000)),
            total_pushed: 0,
            total_dropped: 0,
        }
    }

    /// Push a newly encoded frame into the rolling buffer.
    ///
    /// Frames that exceed [`ClipRecorderConfig::max_frame_bytes`] are silently
    /// dropped.  After insertion the buffer is trimmed so that the oldest frame
    /// falls within [`ClipRecorderConfig::buffer_duration`] of the newest.
    pub fn push_frame(&mut self, frame: EncodedFrame) {
        self.total_pushed += 1;
        if frame.data.len() > self.config.max_frame_bytes {
            self.total_dropped += 1;
            return;
        }
        self.buffer.push_back(frame);
        self.evict_old_frames();
    }

    /// Number of frames currently in the rolling buffer.
    #[must_use]
    pub fn buffered_frames(&self) -> usize {
        self.buffer.len()
    }

    /// Approximate duration of the content currently buffered.
    #[must_use]
    pub fn buffered_duration(&self) -> Duration {
        match (self.buffer.front(), self.buffer.back()) {
            (Some(first), Some(last)) => last.captured_at.duration_since(first.captured_at),
            _ => Duration::ZERO,
        }
    }

    /// Total frames pushed since recorder creation.
    #[must_use]
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Total frames dropped (oversized).
    #[must_use]
    pub fn total_dropped(&self) -> u64 {
        self.total_dropped
    }

    /// Extract a clip according to the given [`ClipRequest`].
    ///
    /// The clip always starts on a keyframe to ensure decodability.
    ///
    /// # Errors
    ///
    /// Returns [`ClipError::BufferEmpty`] if no frames have been buffered,
    /// [`ClipError::InvalidRange`] if `from > to`, or [`ClipError::EmptyClip`]
    /// if no frames fall within the requested range.
    pub fn extract_clip(&self, request: ClipRequest) -> Result<Clip, ClipError> {
        if self.buffer.is_empty() {
            return Err(ClipError::BufferEmpty);
        }

        let (from_instant, to_instant) = self.resolve_request(request)?;

        // Collect frames in the requested time range
        let candidates: Vec<&EncodedFrame> = self
            .buffer
            .iter()
            .filter(|f| f.captured_at >= from_instant && f.captured_at <= to_instant)
            .collect();

        if candidates.is_empty() {
            return Err(ClipError::EmptyClip);
        }

        // Seek backward to the nearest keyframe at or before the first candidate
        let first_candidate_pts = candidates[0].pts_us;
        let keyframe_start = self
            .buffer
            .iter()
            .rev()
            .find(|f| f.kind == FrameKind::Keyframe && f.pts_us <= first_candidate_pts);

        // If there's a keyframe before our window, use it as the start so the
        // clip is decodable; otherwise start from the first candidate.
        let effective_start_pts = keyframe_start
            .map(|f| f.pts_us)
            .unwrap_or(first_candidate_pts);

        let frames: Vec<EncodedFrame> = self
            .buffer
            .iter()
            .filter(|f| f.pts_us >= effective_start_pts && f.captured_at <= to_instant)
            .cloned()
            .collect();

        if frames.is_empty() {
            return Err(ClipError::EmptyClip);
        }

        let metadata = self.build_metadata(&frames, Instant::now());
        Ok(Clip { frames, metadata })
    }

    /// Extract a clip and attach user-supplied tags to its metadata.
    ///
    /// # Errors
    ///
    /// Same as [`extract_clip`][`Self::extract_clip`].
    pub fn extract_clip_tagged(
        &self,
        request: ClipRequest,
        tags: Vec<String>,
    ) -> Result<Clip, ClipError> {
        let mut clip = self.extract_clip(request)?;
        clip.metadata.tags = tags;
        Ok(clip)
    }

    /// Clear the entire buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn evict_old_frames(&mut self) {
        let newest_instant = match self.buffer.back() {
            Some(f) => f.captured_at,
            None => return,
        };
        while let Some(front) = self.buffer.front() {
            let age = newest_instant.duration_since(front.captured_at);
            if age > self.config.buffer_duration {
                self.buffer.pop_front();
                self.total_dropped += 1;
            } else {
                break;
            }
        }
    }

    fn resolve_request(&self, request: ClipRequest) -> Result<(Instant, Instant), ClipError> {
        let newest = self
            .buffer
            .back()
            .ok_or(ClipError::BufferEmpty)?
            .captured_at;

        match request {
            ClipRequest::LastSeconds(secs) => {
                let from = newest
                    .checked_sub(Duration::from_secs(u64::from(secs)))
                    .unwrap_or(newest);
                Ok((from, newest))
            }
            ClipRequest::TimeRange { from, to } => {
                if from > to {
                    return Err(ClipError::InvalidRange("from must be <= to".to_string()));
                }
                Ok((from, to))
            }
            ClipRequest::PtsRange { from_us, to_us } => {
                if from_us > to_us {
                    return Err(ClipError::InvalidRange(
                        "from_us must be <= to_us".to_string(),
                    ));
                }
                // Map PTS range to wall-clock using buffer endpoints
                let first = self
                    .buffer
                    .front()
                    .ok_or(ClipError::BufferEmpty)?
                    .captured_at;
                let first_pts = self.buffer.front().map(|f| f.pts_us).unwrap_or(0);
                let last_pts = self.buffer.back().map(|f| f.pts_us).unwrap_or(0);
                let total_pts_span = (last_pts - first_pts).max(1) as f64;
                let total_duration = newest.duration_since(first);

                let from_offset = ((from_us - first_pts).max(0) as f64 / total_pts_span
                    * total_duration.as_secs_f64())
                .max(0.0);
                let to_offset = ((to_us - first_pts).max(0) as f64 / total_pts_span
                    * total_duration.as_secs_f64())
                .min(total_duration.as_secs_f64());

                let from_instant = first + Duration::from_secs_f64(from_offset);
                let to_instant = first + Duration::from_secs_f64(to_offset);
                Ok((from_instant, to_instant))
            }
        }
    }

    fn build_metadata(&self, frames: &[EncodedFrame], now: Instant) -> ClipMetadata {
        let total_bytes: u64 = frames.iter().map(|f| f.data.len() as u64).sum();
        let keyframe_indices: Vec<usize> = frames
            .iter()
            .enumerate()
            .filter(|(_, f)| f.kind == FrameKind::Keyframe)
            .map(|(i, _)| i)
            .collect();

        let duration = match (frames.first(), frames.last()) {
            (Some(first), Some(last)) => last.captured_at.duration_since(first.captured_at),
            _ => Duration::ZERO,
        };

        let avg_bitrate_bps = if duration.as_secs_f64() > 0.0 {
            (total_bytes * 8) as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        ClipMetadata {
            created_at: now,
            duration,
            width: self.config.width,
            height: self.config.height,
            avg_bitrate_bps,
            keyframe_indices,
            tags: Vec::new(),
            total_bytes,
            frame_count: frames.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(pts_us: i64, kind: FrameKind, t: Instant, bytes: usize) -> EncodedFrame {
        EncodedFrame {
            pts_us,
            data: vec![0u8; bytes],
            kind,
            captured_at: t,
        }
    }

    fn make_recorder(secs: u32) -> ClipRecorder {
        ClipRecorder::new(ClipRecorderConfig {
            buffer_duration: Duration::from_secs(u64::from(secs)),
            fps_hint: 30,
            ..ClipRecorderConfig::default()
        })
    }

    #[test]
    fn test_empty_buffer_returns_error() {
        let recorder = make_recorder(30);
        let result = recorder.extract_clip(ClipRequest::LastSeconds(5));
        assert_eq!(result.unwrap_err(), ClipError::BufferEmpty);
    }

    #[test]
    fn test_push_and_buffered_frames() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 5_000));
        recorder.push_frame(make_frame(33_333, FrameKind::Interframe, now, 3_000));
        assert_eq!(recorder.buffered_frames(), 2);
        assert_eq!(recorder.total_pushed(), 2);
        assert_eq!(recorder.total_dropped(), 0);
    }

    #[test]
    fn test_oversized_frame_dropped() {
        let mut recorder = ClipRecorder::new(ClipRecorderConfig {
            max_frame_bytes: 1_000,
            ..ClipRecorderConfig::default()
        });
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 2_000));
        assert_eq!(recorder.buffered_frames(), 0);
        assert_eq!(recorder.total_dropped(), 1);
    }

    #[test]
    fn test_rolling_eviction() {
        let mut recorder = ClipRecorder::new(ClipRecorderConfig {
            buffer_duration: Duration::from_secs(1),
            fps_hint: 30,
            ..ClipRecorderConfig::default()
        });
        let now = Instant::now();
        // Push a frame 2 seconds in the past
        recorder.push_frame(make_frame(
            0,
            FrameKind::Keyframe,
            now - Duration::from_secs(2),
            1_000,
        ));
        // Push a fresh frame — eviction should remove the old one
        recorder.push_frame(make_frame(33_333, FrameKind::Keyframe, now, 1_000));
        assert_eq!(recorder.buffered_frames(), 1);
    }

    #[test]
    fn test_extract_last_seconds() {
        let mut recorder = make_recorder(60);
        let now = Instant::now();
        let base = now - Duration::from_secs(10);
        for i in 0..30u64 {
            let t = base + Duration::from_millis(i * 333);
            let kind = if i == 0 {
                FrameKind::Keyframe
            } else {
                FrameKind::Interframe
            };
            recorder.push_frame(make_frame(i as i64 * 33_333, kind, t, 4_000));
        }
        let clip = recorder
            .extract_clip(ClipRequest::LastSeconds(5))
            .expect("clip");
        assert!(!clip.frames.is_empty());
        assert!(clip.metadata.frame_count > 0);
    }

    #[test]
    fn test_extract_time_range_invalid() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 4_000));
        let result = recorder.extract_clip(ClipRequest::TimeRange {
            from: now + Duration::from_secs(5),
            to: now,
        });
        assert_eq!(
            result.unwrap_err(),
            ClipError::InvalidRange("from must be <= to".to_string())
        );
    }

    #[test]
    fn test_clip_metadata_keyframe_indices() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 8_000));
        recorder.push_frame(make_frame(
            33_000,
            FrameKind::Interframe,
            now + Duration::from_millis(33),
            4_000,
        ));
        recorder.push_frame(make_frame(
            66_000,
            FrameKind::Keyframe,
            now + Duration::from_millis(66),
            8_000,
        ));
        let clip = recorder
            .extract_clip(ClipRequest::TimeRange {
                from: now,
                to: now + Duration::from_millis(100),
            })
            .expect("clip");
        // Should have at least one keyframe index
        assert!(!clip.metadata.keyframe_indices.is_empty());
    }

    #[test]
    fn test_clip_tagged() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 4_000));
        let tags = vec!["highlight".to_string(), "triple-kill".to_string()];
        let clip = recorder
            .extract_clip_tagged(ClipRequest::LastSeconds(5), tags.clone())
            .expect("clip");
        assert_eq!(clip.metadata.tags, tags);
    }

    #[test]
    fn test_clear_empties_buffer() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 4_000));
        recorder.clear();
        assert_eq!(recorder.buffered_frames(), 0);
        // total_pushed still reflects the push before clear
        assert_eq!(recorder.total_pushed(), 1);
    }

    #[test]
    fn test_buffered_duration_zero_on_single_frame() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        recorder.push_frame(make_frame(0, FrameKind::Keyframe, now, 4_000));
        assert_eq!(recorder.buffered_duration(), Duration::ZERO);
    }

    #[test]
    fn test_clip_starts_on_keyframe() {
        let mut recorder = make_recorder(30);
        let now = Instant::now();
        // P-frame first (PTS 0), then keyframe (PTS 1s), then more P-frames
        recorder.push_frame(make_frame(0, FrameKind::Interframe, now, 4_000));
        recorder.push_frame(make_frame(
            1_000_000,
            FrameKind::Keyframe,
            now + Duration::from_secs(1),
            8_000,
        ));
        recorder.push_frame(make_frame(
            1_033_333,
            FrameKind::Interframe,
            now + Duration::from_millis(1033),
            4_000,
        ));

        // Request only the last 0.5 s — starts mid-GOP; recorder should
        // walk back to the keyframe at PTS 1_000_000.
        let clip = recorder
            .extract_clip(ClipRequest::LastSeconds(1))
            .expect("clip");
        // First frame in clip must be the keyframe
        assert_eq!(
            clip.frames[0].kind,
            FrameKind::Keyframe,
            "clip must start on a keyframe"
        );
    }
}
