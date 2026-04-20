//! Media stream packaging into timed segments.
//!
//! Provides [`SegmentPackager`] which accumulates [`MediaUnit`]s and flushes
//! them into [`PackagedSegment`]s either when the configured segment duration
//! is reached or when explicitly flushed.

use std::collections::VecDeque;
use std::path::PathBuf;

use crate::StreamError;

// ─── StreamType ───────────────────────────────────────────────────────────────

/// Elementary stream type carried by a [`MediaUnit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// Video elementary stream.
    Video,
    /// Audio elementary stream.
    Audio,
    /// Data / metadata stream (subtitles, ID3, etc.).
    Data,
}

// ─── MediaUnit ────────────────────────────────────────────────────────────────

/// A single encoded media access unit (NAL unit, audio frame, etc.).
#[derive(Debug, Clone)]
pub struct MediaUnit {
    /// Presentation timestamp in milliseconds.
    pub pts_ms: i64,
    /// Decode timestamp in milliseconds.
    pub dts_ms: i64,
    /// Raw encoded bytes for this unit.
    pub data: Vec<u8>,
    /// Whether this unit is a random-access (key) frame.
    pub is_keyframe: bool,
    /// The elementary stream this unit belongs to.
    pub stream_type: StreamType,
}

// ─── PackagedSegment ─────────────────────────────────────────────────────────

/// A finalized media segment ready for writing or manifest registration.
#[derive(Debug, Clone)]
pub struct PackagedSegment {
    /// Monotonically increasing sequence number.
    pub sequence_number: u64,
    /// Duration of this segment in milliseconds.
    pub duration_ms: u32,
    /// Concatenated encoded data.
    pub data: Vec<u8>,
    /// Whether the segment is aligned to a key frame at its start.
    pub keyframe_aligned: bool,
}

// ─── PackagerConfig ───────────────────────────────────────────────────────────

/// Configuration for [`SegmentPackager`].
#[derive(Debug, Clone)]
pub struct PackagerConfig {
    /// Target segment duration in milliseconds.
    pub segment_duration_ms: u32,
    /// Maximum allowed segment duration before forced flush (milliseconds).
    pub target_duration_ms: u32,
    /// HLS version to target (affects output format hints).
    pub hls_version: u8,
    /// Enable CMAF-compatible packaging (single-track fragmented MP4).
    pub enable_cmaf: bool,
}

impl Default for PackagerConfig {
    fn default() -> Self {
        Self {
            segment_duration_ms: 6_000,
            target_duration_ms: 10_000,
            hls_version: 7,
            enable_cmaf: false,
        }
    }
}

// ─── SegmentPackager ─────────────────────────────────────────────────────────

/// Accumulates [`MediaUnit`]s and emits [`PackagedSegment`]s.
///
/// A new segment is started on the first keyframe encountered after the
/// configured `segment_duration_ms` has elapsed since the last segment
/// boundary.
pub struct SegmentPackager {
    /// Packager configuration.
    pub config: PackagerConfig,
    /// Monotonically increasing output segment counter.
    pub segment_counter: u64,
    /// Units buffered for the current in-progress segment.
    pub pending_frames: VecDeque<MediaUnit>,
    /// PTS (ms) of the first unit in the current segment.
    segment_start_pts: Option<i64>,
}

impl SegmentPackager {
    /// Create a packager with the given config.
    pub fn new(config: PackagerConfig) -> Self {
        Self {
            config,
            segment_counter: 0,
            pending_frames: VecDeque::new(),
            segment_start_pts: None,
        }
    }

    /// Push a [`MediaUnit`] into the packager.
    ///
    /// Returns `Some(PackagedSegment)` if the unit triggered a segment
    /// boundary, otherwise `None`.
    pub fn push(&mut self, unit: MediaUnit) -> Option<PackagedSegment> {
        // Track the PTS of the first unit ever pushed.
        if self.segment_start_pts.is_none() {
            self.segment_start_pts = Some(unit.pts_ms);
        }

        let elapsed_ms = unit
            .pts_ms
            .saturating_sub(self.segment_start_pts.unwrap_or(unit.pts_ms))
            .max(0) as u64;

        // Determine whether to cut a new segment.
        let over_target = elapsed_ms >= self.config.segment_duration_ms as u64;
        let over_max = elapsed_ms >= self.config.target_duration_ms as u64;
        let should_cut =
            (over_target && unit.is_keyframe && unit.stream_type == StreamType::Video) || over_max;

        if should_cut && !self.pending_frames.is_empty() {
            // Flush current segment before accepting this unit.
            let seg = self.flush_internal();
            // Start new segment with this unit.
            self.segment_start_pts = Some(unit.pts_ms);
            self.pending_frames.push_back(unit);
            return seg;
        }

        self.pending_frames.push_back(unit);
        None
    }

    /// Force-flush any buffered units into a [`PackagedSegment`].
    ///
    /// Returns `None` if there are no pending units.
    pub fn flush(&mut self) -> Option<PackagedSegment> {
        self.flush_internal()
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn flush_internal(&mut self) -> Option<PackagedSegment> {
        if self.pending_frames.is_empty() {
            return None;
        }

        let units: Vec<MediaUnit> = self.pending_frames.drain(..).collect();
        let segment = pack_segment_from_units(&units, self.segment_counter);
        self.segment_counter += 1;
        self.segment_start_pts = None;
        Some(segment)
    }
}

// ─── pack_segment ─────────────────────────────────────────────────────────────

/// Assemble a slice of [`MediaUnit`]s into a [`PackagedSegment`].
///
/// - Duration is derived from `max(pts_ms) - min(pts_ms)`.
/// - `keyframe_aligned` is `true` when the first video unit is a keyframe.
pub fn pack_segment(units: &[MediaUnit]) -> PackagedSegment {
    pack_segment_from_units(units, 0)
}

fn pack_segment_from_units(units: &[MediaUnit], sequence_number: u64) -> PackagedSegment {
    if units.is_empty() {
        return PackagedSegment {
            sequence_number,
            duration_ms: 0,
            data: Vec::new(),
            keyframe_aligned: false,
        };
    }

    let min_pts = units.iter().map(|u| u.pts_ms).min().unwrap_or(0);
    let max_pts = units.iter().map(|u| u.pts_ms).max().unwrap_or(0);
    let duration_ms = (max_pts - min_pts).max(0) as u32;

    let keyframe_aligned = units
        .iter()
        .find(|u| u.stream_type == StreamType::Video)
        .map(|u| u.is_keyframe)
        .unwrap_or(false);

    // Concatenate all data buffers.
    let total_len: usize = units.iter().map(|u| u.data.len()).sum();
    let mut data = Vec::with_capacity(total_len);
    for unit in units {
        data.extend_from_slice(&unit.data);
    }

    PackagedSegment {
        sequence_number,
        duration_ms,
        data,
        keyframe_aligned,
    }
}

// ─── SegmentWriter trait ──────────────────────────────────────────────────────

/// Sink that accepts finished [`PackagedSegment`]s.
pub trait SegmentWriter {
    /// Write a single segment.
    fn write(&mut self, segment: &PackagedSegment) -> Result<(), StreamError>;
}

// ─── FileSegmentWriter ────────────────────────────────────────────────────────

/// Writes [`PackagedSegment`]s to files in a directory.
///
/// Files are named `{prefix}{sequence_number:08}.seg`.
pub struct FileSegmentWriter {
    /// Directory to write segment files into.
    pub output_dir: PathBuf,
    /// Filename prefix, e.g. `"segment_"`.
    pub prefix: String,
}

impl FileSegmentWriter {
    /// Construct a new writer.
    pub fn new(output_dir: impl Into<PathBuf>, prefix: impl Into<String>) -> Self {
        Self {
            output_dir: output_dir.into(),
            prefix: prefix.into(),
        }
    }

    /// Compute the full path for a given sequence number.
    pub fn segment_path(&self, sequence_number: u64) -> PathBuf {
        self.output_dir
            .join(format!("{}{:08}.seg", self.prefix, sequence_number))
    }
}

impl SegmentWriter for FileSegmentWriter {
    fn write(&mut self, segment: &PackagedSegment) -> Result<(), StreamError> {
        let path = self.segment_path(segment.sequence_number);
        std::fs::write(&path, &segment.data).map_err(|e| {
            StreamError::IoError(format!("failed to write segment {}: {e}", path.display()))
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_video_keyframe(pts_ms: i64) -> MediaUnit {
        MediaUnit {
            pts_ms,
            dts_ms: pts_ms,
            data: vec![0xAA, 0xBB, 0xCC],
            is_keyframe: true,
            stream_type: StreamType::Video,
        }
    }

    fn make_video_delta(pts_ms: i64) -> MediaUnit {
        MediaUnit {
            pts_ms,
            dts_ms: pts_ms,
            data: vec![0x11, 0x22],
            is_keyframe: false,
            stream_type: StreamType::Video,
        }
    }

    fn make_audio(pts_ms: i64) -> MediaUnit {
        MediaUnit {
            pts_ms,
            dts_ms: pts_ms,
            data: vec![0xFF, 0xF1, 0x50],
            is_keyframe: false,
            stream_type: StreamType::Audio,
        }
    }

    // ── PackagerConfig ────────────────────────────────────────────────────────

    #[test]
    fn test_packager_config_default_values() {
        let cfg = PackagerConfig::default();
        assert_eq!(cfg.segment_duration_ms, 6_000);
        assert_eq!(cfg.target_duration_ms, 10_000);
        assert_eq!(cfg.hls_version, 7);
        assert!(!cfg.enable_cmaf);
    }

    // ── pack_segment ──────────────────────────────────────────────────────────

    #[test]
    fn test_pack_segment_empty_input() {
        let seg = pack_segment(&[]);
        assert_eq!(seg.duration_ms, 0);
        assert!(seg.data.is_empty());
        assert!(!seg.keyframe_aligned);
    }

    #[test]
    fn test_pack_segment_single_keyframe() {
        let units = vec![make_video_keyframe(0)];
        let seg = pack_segment(&units);
        assert_eq!(seg.duration_ms, 0); // single unit — min == max
        assert!(seg.keyframe_aligned);
        assert_eq!(seg.data.len(), 3);
    }

    #[test]
    fn test_pack_segment_duration_calculated() {
        let units = vec![make_video_keyframe(0), make_video_delta(3000)];
        let seg = pack_segment(&units);
        assert_eq!(seg.duration_ms, 3000);
    }

    #[test]
    fn test_pack_segment_data_concatenated() {
        let units = vec![make_video_keyframe(0), make_audio(500)];
        let seg = pack_segment(&units);
        assert_eq!(seg.data.len(), 3 + 3); // 3 video + 3 audio bytes
    }

    #[test]
    fn test_pack_segment_keyframe_aligned_false_when_delta_first() {
        let units = vec![make_video_delta(0), make_video_keyframe(3000)];
        let seg = pack_segment(&units);
        assert!(!seg.keyframe_aligned);
    }

    #[test]
    fn test_pack_segment_no_video_keyframe_alignment_false() {
        let units = vec![make_audio(0), make_audio(1000)];
        let seg = pack_segment(&units);
        // No video units — keyframe_aligned should be false
        assert!(!seg.keyframe_aligned);
    }

    // ── SegmentPackager::push ─────────────────────────────────────────────────

    #[test]
    fn test_packager_no_segment_before_duration_exceeded() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        // Push several frames well within 6 s
        for ms in [0i64, 33, 66, 100] {
            let result = p.push(make_video_delta(ms));
            assert!(result.is_none(), "too early to segment at pts={ms}ms");
        }
    }

    #[test]
    fn test_packager_segments_on_keyframe_after_target() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        // Push delta frames for 6 seconds
        for i in 0..60i64 {
            let _ = p.push(make_video_delta(i * 100));
        }
        // Push a keyframe at 6000 ms — should trigger segment flush
        let seg = p.push(make_video_keyframe(6000));
        assert!(
            seg.is_some(),
            "keyframe after target duration should produce a segment"
        );
    }

    #[test]
    fn test_packager_segment_sequence_increments() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        for i in 0..60i64 {
            let _ = p.push(make_video_delta(i * 100));
        }
        let seg1 = p.push(make_video_keyframe(6000));
        for i in 0..60i64 {
            let _ = p.push(make_video_delta(6000 + i * 100));
        }
        let seg2 = p.push(make_video_keyframe(12000));
        let n1 = seg1.expect("seg1").sequence_number;
        let n2 = seg2.expect("seg2").sequence_number;
        assert_eq!(n2, n1 + 1);
    }

    #[test]
    fn test_packager_forced_flush_when_over_max() {
        let config = PackagerConfig {
            segment_duration_ms: 6_000,
            target_duration_ms: 8_000,
            ..PackagerConfig::default()
        };
        let mut p = SegmentPackager::new(config);
        // Push delta frames up to 8001 ms without a keyframe
        for i in 0..81i64 {
            let result = p.push(make_video_delta(i * 100));
            if i == 80 {
                assert!(result.is_some(), "forced flush at max duration");
            }
        }
    }

    // ── SegmentPackager::flush ────────────────────────────────────────────────

    #[test]
    fn test_flush_returns_none_when_empty() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        assert!(p.flush().is_none());
    }

    #[test]
    fn test_flush_drains_pending_frames() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        p.push(make_video_keyframe(0));
        p.push(make_video_delta(33));
        let seg = p.flush().expect("should produce a segment on flush");
        assert!(seg.data.len() > 0);
        assert_eq!(p.pending_frames.len(), 0);
    }

    #[test]
    fn test_flush_twice_returns_none_second_time() {
        let mut p = SegmentPackager::new(PackagerConfig::default());
        p.push(make_video_keyframe(0));
        let _seg = p.flush();
        assert!(p.flush().is_none());
    }

    // ── FileSegmentWriter ──────────────────────────────────────────────────────

    #[test]
    fn test_file_segment_writer_path_format() {
        let writer = FileSegmentWriter::new(
            std::env::temp_dir().join("oximedia-stream-pkgr-segs"),
            "seg_",
        );
        let path = writer.segment_path(7);
        assert_eq!(
            path.file_name().and_then(|n| n.to_str()),
            Some("seg_00000007.seg")
        );
    }

    #[test]
    fn test_file_segment_writer_write_and_read_back() {
        let dir = std::env::temp_dir().join(format!(
            "oximedia_packager_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let mut writer = FileSegmentWriter::new(&dir, "test_");
        let seg = PackagedSegment {
            sequence_number: 0,
            duration_ms: 6000,
            data: vec![1, 2, 3, 4, 5],
            keyframe_aligned: true,
        };
        writer.write(&seg).expect("write segment");

        let path = writer.segment_path(0);
        let read_back = std::fs::read(&path).expect("read back");
        assert_eq!(read_back, vec![1, 2, 3, 4, 5]);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_file_segment_writer_write_to_nonexistent_dir_fails() {
        let mut writer = FileSegmentWriter::new("/nonexistent/path/that/cannot/exist", "seg_");
        let seg = PackagedSegment {
            sequence_number: 0,
            duration_ms: 0,
            data: vec![],
            keyframe_aligned: false,
        };
        assert!(writer.write(&seg).is_err());
    }

    // ── StreamType ────────────────────────────────────────────────────────────

    #[test]
    fn test_stream_type_eq() {
        assert_eq!(StreamType::Video, StreamType::Video);
        assert_ne!(StreamType::Video, StreamType::Audio);
        assert_ne!(StreamType::Audio, StreamType::Data);
    }

    // ── PackagedSegment data integrity ────────────────────────────────────────

    #[test]
    fn test_pack_segment_sequence_number_zero_by_default() {
        let units = vec![make_video_keyframe(0)];
        let seg = pack_segment(&units);
        assert_eq!(seg.sequence_number, 0);
    }

    #[test]
    fn test_pack_segment_large_pts_difference() {
        let units = vec![make_video_keyframe(0), make_video_delta(60_000)];
        let seg = pack_segment(&units);
        assert_eq!(seg.duration_ms, 60_000);
    }
}
