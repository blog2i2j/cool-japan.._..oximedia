//! Timecode-based multi-camera synchronization.
//!
//! This module provides tools for parsing SMPTE timecodes, aligning multiple camera
//! streams using their embedded timecodes, and validating the consistency of the
//! resulting offsets.

#![allow(dead_code)]

/// Associates a camera with a timecode and a frame index in its local timeline.
#[derive(Debug, Clone)]
pub struct CameraTimecodeRecord {
    /// Camera identifier.
    pub camera_id: u32,
    /// SMPTE timecode string at the reference point (HH:MM:SS:FF).
    pub timecode: String,
    /// Local frame index corresponding to this timecode.
    pub frame_idx: u64,
}

impl CameraTimecodeRecord {
    /// Creates a new `CameraTimecodeRecord`.
    pub fn new(camera_id: u32, timecode: impl Into<String>, frame_idx: u64) -> Self {
        Self {
            camera_id,
            timecode: timecode.into(),
            frame_idx,
        }
    }
}

/// Provides SMPTE timecode parsing and formatting utilities.
pub struct TimecodeParser;

impl TimecodeParser {
    /// Converts a SMPTE timecode string (HH:MM:SS:FF) to a frame count.
    ///
    /// # Arguments
    /// * `tc` - SMPTE timecode string, e.g. `"01:02:03:12"`.
    /// * `fps` - Frames per second (e.g. 25.0, 29.97, 30.0).
    ///
    /// # Returns
    /// Total frame count, or `0` if the timecode cannot be parsed.
    #[must_use]
    pub fn from_smpte(tc: &str, fps: f32) -> u64 {
        if fps <= 0.0 {
            return 0;
        }

        let parts: Vec<&str> = tc.split(':').collect();
        if parts.len() != 4 {
            return 0;
        }

        let hours: u64 = parts[0].parse().unwrap_or(0);
        let minutes: u64 = parts[1].parse().unwrap_or(0);
        let seconds: u64 = parts[2].parse().unwrap_or(0);
        let frames: u64 = parts[3].parse().unwrap_or(0);

        let fps_round = fps.round() as u64;
        hours * 3600 * fps_round + minutes * 60 * fps_round + seconds * fps_round + frames
    }

    /// Converts a frame count to a SMPTE timecode string (HH:MM:SS:FF).
    ///
    /// # Arguments
    /// * `frames` - Total frame count.
    /// * `fps` - Frames per second.
    ///
    /// # Returns
    /// SMPTE timecode string.
    #[must_use]
    pub fn to_smpte(frames: u64, fps: f32) -> String {
        if fps <= 0.0 {
            return "00:00:00:00".to_string();
        }

        let fps_round = fps.round() as u64;
        let fps_round = fps_round.max(1);

        let ff = frames % fps_round;
        let total_seconds = frames / fps_round;
        let ss = total_seconds % 60;
        let total_minutes = total_seconds / 60;
        let mm = total_minutes % 60;
        let hh = total_minutes / 60;

        format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
    }
}

/// Provides camera alignment from timecode records.
pub struct TimecodeSync;

impl TimecodeSync {
    /// Aligns multiple cameras by computing frame offsets relative to camera 0.
    ///
    /// The reference camera is the one with `camera_id == 0`. If no such camera
    /// exists, the first camera in the slice is used as the reference.
    ///
    /// # Arguments
    /// * `records` - Slice of per-camera timecode records (one per camera).
    /// * `fps` - Frames per second, used to convert timecodes to frame counts.
    ///
    /// # Returns
    /// A vector of `(camera_id, offset_frames)` tuples. A positive offset means
    /// the camera is ahead of the reference and should be delayed.
    #[must_use]
    pub fn align_cameras(records: &[CameraTimecodeRecord], fps: f32) -> Vec<(u32, i64)> {
        if records.is_empty() {
            return Vec::new();
        }

        // Find reference camera (id == 0) or fall back to first
        let reference = records
            .iter()
            .find(|r| r.camera_id == 0)
            .unwrap_or(&records[0]);

        let ref_tc_frames = TimecodeParser::from_smpte(&reference.timecode, fps);
        let ref_frame_idx = reference.frame_idx;

        records
            .iter()
            .map(|r| {
                let tc_frames = TimecodeParser::from_smpte(&r.timecode, fps);
                // Timecode difference (in frames) between this camera and reference
                let tc_diff = tc_frames as i64 - ref_tc_frames as i64;
                // Local frame index difference
                let idx_diff = r.frame_idx as i64 - ref_frame_idx as i64;
                // Offset = how many frames this camera is ahead of reference at sync point
                let offset = tc_diff - idx_diff;
                (r.camera_id, offset)
            })
            .collect()
    }
}

/// Records a timecode offset with associated confidence.
#[derive(Debug, Clone)]
pub struct TimecodeOffset {
    /// Camera ID.
    pub camera_id: u32,
    /// Signed frame offset relative to reference camera.
    pub offset_frames: i64,
    /// Confidence score in the range [0.0, 1.0].
    pub confidence: f32,
}

impl TimecodeOffset {
    /// Creates a new `TimecodeOffset`.
    #[must_use]
    pub fn new(camera_id: u32, offset_frames: i64, confidence: f32) -> Self {
        Self {
            camera_id,
            offset_frames,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

/// Validates timecode offsets for consistency.
pub struct SyncValidator;

impl SyncValidator {
    /// Checks whether the provided offsets are consistent with each other.
    ///
    /// Generates warning strings for:
    /// - Large absolute offsets (> 10 frames).
    /// - Low confidence values (< 0.5).
    ///
    /// # Returns
    /// A vector of warning messages; empty if everything looks consistent.
    #[must_use]
    pub fn check_consistency(offsets: &[TimecodeOffset]) -> Vec<String> {
        let mut warnings = Vec::new();

        for offset in offsets {
            if offset.offset_frames.unsigned_abs() > 10 {
                warnings.push(format!(
                    "Camera {}: large offset of {} frames detected",
                    offset.camera_id, offset.offset_frames
                ));
            }

            if offset.confidence < 0.5 {
                warnings.push(format!(
                    "Camera {}: low synchronization confidence {:.2}",
                    offset.camera_id, offset.confidence
                ));
            }
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_from_smpte_zero() {
        assert_eq!(TimecodeParser::from_smpte("00:00:00:00", 25.0), 0);
    }

    #[test]
    fn test_timecode_from_smpte_one_second() {
        // 1 second at 25fps = 25 frames
        assert_eq!(TimecodeParser::from_smpte("00:00:01:00", 25.0), 25);
    }

    #[test]
    fn test_timecode_from_smpte_frames() {
        // "00:00:00:12" at 25fps = 12 frames
        assert_eq!(TimecodeParser::from_smpte("00:00:00:12", 25.0), 12);
    }

    #[test]
    fn test_timecode_from_smpte_invalid() {
        assert_eq!(TimecodeParser::from_smpte("invalid", 25.0), 0);
    }

    #[test]
    fn test_timecode_to_smpte_zero() {
        assert_eq!(TimecodeParser::to_smpte(0, 25.0), "00:00:00:00");
    }

    #[test]
    fn test_timecode_to_smpte_one_second() {
        assert_eq!(TimecodeParser::to_smpte(25, 25.0), "00:00:01:00");
    }

    #[test]
    fn test_timecode_roundtrip() {
        let tc = "01:02:03:15";
        let frames = TimecodeParser::from_smpte(tc, 25.0);
        let result = TimecodeParser::to_smpte(frames, 25.0);
        assert_eq!(result, tc);
    }

    #[test]
    fn test_align_cameras_no_offset() {
        let records = vec![
            CameraTimecodeRecord::new(0, "00:00:10:00", 250),
            CameraTimecodeRecord::new(1, "00:00:10:00", 250),
        ];
        let offsets = TimecodeSync::align_cameras(&records, 25.0);
        assert_eq!(offsets.len(), 2);

        let cam1 = offsets
            .iter()
            .find(|(id, _)| *id == 1)
            .expect("multicam test operation should succeed");
        assert_eq!(cam1.1, 0); // no offset
    }

    #[test]
    fn test_align_cameras_with_offset() {
        // Camera 1 is 10 frames ahead in timecode but at the same local frame index
        let records = vec![
            CameraTimecodeRecord::new(0, "00:00:10:00", 250),
            CameraTimecodeRecord::new(1, "00:00:10:10", 250),
        ];
        let offsets = TimecodeSync::align_cameras(&records, 25.0);
        let cam1 = offsets
            .iter()
            .find(|(id, _)| *id == 1)
            .expect("multicam test operation should succeed");
        assert_eq!(cam1.1, 10);
    }

    #[test]
    fn test_align_cameras_empty() {
        let offsets = TimecodeSync::align_cameras(&[], 25.0);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_timecode_offset_creation() {
        let off = TimecodeOffset::new(2, -5, 0.9);
        assert_eq!(off.camera_id, 2);
        assert_eq!(off.offset_frames, -5);
        assert!((off.confidence - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sync_validator_no_warnings() {
        let offsets = vec![
            TimecodeOffset::new(0, 0, 1.0),
            TimecodeOffset::new(1, 2, 0.95),
        ];
        let warnings = SyncValidator::check_consistency(&offsets);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_sync_validator_large_offset_warning() {
        let offsets = vec![TimecodeOffset::new(1, 50, 0.9)];
        let warnings = SyncValidator::check_consistency(&offsets);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("large offset"));
    }

    #[test]
    fn test_sync_validator_low_confidence_warning() {
        let offsets = vec![TimecodeOffset::new(2, 0, 0.3)];
        let warnings = SyncValidator::check_consistency(&offsets);
        assert!(!warnings.is_empty());
        assert!(warnings[0].contains("confidence"));
    }
}
