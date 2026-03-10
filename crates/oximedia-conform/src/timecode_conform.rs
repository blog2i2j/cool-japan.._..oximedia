//! Timecode conformance checking and offset analysis.
//!
//! Parses SMPTE-style timecodes, converts between frame numbers and timecode
//! strings, and reports timecode conflicts across a set of clips.

#![allow(dead_code)]

/// A timecode conflict between expected and found values for a single clip.
#[derive(Debug, Clone)]
pub struct TimecodeConflict {
    /// Unique identifier for the clip.
    pub clip_id: u64,
    /// The timecode that was expected for this clip.
    pub expected_tc: String,
    /// The timecode that was actually found.
    pub found_tc: String,
    /// The difference expressed in frames (positive = found is later).
    pub offset_frames: i32,
}

impl TimecodeConflict {
    /// Returns `true` when the absolute frame offset meets or exceeds `threshold`.
    #[must_use]
    pub fn is_significant(&self, threshold: i32) -> bool {
        self.offset_frames.unsigned_abs() >= threshold as u32
    }
}

/// Performs timecode parsing, conversion, and offset calculation.
pub struct TimecodeConformer {
    /// Nominal frame rate of the timeline.
    pub frame_rate: f64,
    /// Whether to use drop-frame (DF) counting (29.97 / 59.94 only).
    pub drop_frame: bool,
}

impl TimecodeConformer {
    /// Create a new conformer.
    #[must_use]
    pub fn new(frame_rate: f64, drop_frame: bool) -> Self {
        Self {
            frame_rate,
            drop_frame,
        }
    }

    /// Parse a timecode string `HH:MM:SS:FF` (or `HH:MM:SS;FF` for DF) into
    /// an absolute frame count.
    ///
    /// Returns `None` if the string is malformed or the values are out of range.
    #[must_use]
    pub fn parse_tc(&self, s: &str) -> Option<u64> {
        // Accept both ':' and ';' as frame separator
        let s = s.replace(';', ":");
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 4 {
            return None;
        }
        let hh: u64 = parts[0].parse().ok()?;
        let mm: u64 = parts[1].parse().ok()?;
        let ss: u64 = parts[2].parse().ok()?;
        let ff: u64 = parts[3].parse().ok()?;

        let fps = self.frame_rate.round() as u64;
        if ff >= fps || ss >= 60 || mm >= 60 {
            return None;
        }

        if self.drop_frame && (fps == 30 || fps == 60) {
            // Drop-frame formula (30 DF version; scale for 60 DF)
            let drop_per_min = if fps == 30 { 2u64 } else { 4u64 };
            let total_minutes = hh * 60 + mm;
            let drop_frames = drop_per_min * (total_minutes - total_minutes / 10);
            let frame_number = fps * 3600 * hh + fps * 60 * mm + fps * ss + ff - drop_frames;
            Some(frame_number)
        } else {
            Some(hh * 3600 * fps + mm * 60 * fps + ss * fps + ff)
        }
    }

    /// Convert an absolute frame count to a timecode string `HH:MM:SS:FF`.
    #[must_use]
    pub fn frames_to_tc(&self, frames: u64) -> String {
        let fps = self.frame_rate.round() as u64;
        if fps == 0 {
            return "00:00:00:00".to_string();
        }

        if self.drop_frame && (fps == 30 || fps == 60) {
            let drop_per_min = if fps == 30 { 2u64 } else { 4u64 };
            let frames_per_10min = fps * 60 * 10 - drop_per_min * 9;
            let d = frames / frames_per_10min;
            let m = frames % frames_per_10min;
            let frames_per_min = fps * 60 - drop_per_min;
            let adjusted = if m < drop_per_min {
                frames + drop_per_min * 9 * d
            } else {
                frames + drop_per_min * 9 * d + drop_per_min * ((m - drop_per_min) / frames_per_min)
            };
            let ff = adjusted % fps;
            let total_secs = adjusted / fps;
            let ss = total_secs % 60;
            let total_mins = total_secs / 60;
            let mm = total_mins % 60;
            let hh = total_mins / 60;
            format!("{hh:02}:{mm:02}:{ss:02};{ff:02}")
        } else {
            let ff = frames % fps;
            let total_secs = frames / fps;
            let ss = total_secs % 60;
            let total_mins = total_secs / 60;
            let mm = total_mins % 60;
            let hh = total_mins / 60;
            format!("{hh:02}:{mm:02}:{ss:02}:{ff:02}")
        }
    }

    /// Compute the signed frame offset between two timecode strings
    /// (tc2 − tc1).
    ///
    /// Returns `None` if either string fails to parse.
    #[must_use]
    pub fn offset_between(&self, tc1: &str, tc2: &str) -> Option<i32> {
        let f1 = self.parse_tc(tc1)?;
        let f2 = self.parse_tc(tc2)?;
        Some(f2 as i32 - f1 as i32)
    }
}

/// Aggregated timecode conformance report.
#[derive(Debug)]
pub struct TimecodeConformReport {
    /// All detected timecode conflicts.
    pub conflicts: Vec<TimecodeConflict>,
    /// Total number of clips examined.
    pub total_clips: usize,
}

impl TimecodeConformReport {
    /// Create a new report.
    #[must_use]
    pub fn new(conflicts: Vec<TimecodeConflict>, total_clips: usize) -> Self {
        Self {
            conflicts,
            total_clips,
        }
    }

    /// Returns the fraction of clips that have conflicts (0.0 – 1.0).
    #[must_use]
    pub fn conflict_rate(&self) -> f64 {
        if self.total_clips == 0 {
            return 0.0;
        }
        self.conflicts.len() as f64 / self.total_clips as f64
    }

    /// Returns `true` when at least one conflict was found.
    #[must_use]
    pub fn has_conflicts(&self) -> bool {
        !self.conflicts.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ndf25() -> TimecodeConformer {
        TimecodeConformer::new(25.0, false)
    }

    fn ndf30() -> TimecodeConformer {
        TimecodeConformer::new(30.0, false)
    }

    #[test]
    fn test_parse_tc_simple() {
        let tc = ndf25();
        let frames = tc.parse_tc("00:00:01:00");
        assert_eq!(frames, Some(25));
    }

    #[test]
    fn test_parse_tc_one_minute() {
        let tc = ndf25();
        let frames = tc.parse_tc("00:01:00:00");
        assert_eq!(frames, Some(25 * 60));
    }

    #[test]
    fn test_parse_tc_one_hour() {
        let tc = ndf25();
        let frames = tc.parse_tc("01:00:00:00");
        assert_eq!(frames, Some(25 * 3600));
    }

    #[test]
    fn test_parse_tc_mixed() {
        let tc = ndf25();
        // 01:02:03:04 = 3600*25 + 2*60*25 + 3*25 + 4 = 90000 + 3000 + 75 + 4
        let expected = 25 * 3600 + 2 * 60 * 25 + 3 * 25 + 4;
        assert_eq!(tc.parse_tc("01:02:03:04"), Some(expected));
    }

    #[test]
    fn test_parse_tc_malformed() {
        let tc = ndf25();
        assert!(tc.parse_tc("bad").is_none());
        assert!(tc.parse_tc("00:00:00").is_none());
    }

    #[test]
    fn test_parse_tc_out_of_range_frames() {
        let tc = ndf25();
        // 25 frames is out of range for 25fps (valid: 00–24)
        assert!(tc.parse_tc("00:00:00:25").is_none());
    }

    #[test]
    fn test_frames_to_tc_roundtrip() {
        let tc = ndf25();
        let original = "01:02:03:04";
        let frames = tc.parse_tc(original).expect("frames should be valid");
        let back = tc.frames_to_tc(frames);
        assert_eq!(back, original);
    }

    #[test]
    fn test_frames_to_tc_zero() {
        let tc = ndf30();
        assert_eq!(tc.frames_to_tc(0), "00:00:00:00");
    }

    #[test]
    fn test_offset_between_positive() {
        let tc = ndf25();
        // 00:00:01:00 → 25 frames; 00:00:02:00 → 50 frames; offset = +25
        let offset = tc.offset_between("00:00:01:00", "00:00:02:00");
        assert_eq!(offset, Some(25));
    }

    #[test]
    fn test_offset_between_negative() {
        let tc = ndf25();
        let offset = tc.offset_between("00:00:02:00", "00:00:01:00");
        assert_eq!(offset, Some(-25));
    }

    #[test]
    fn test_offset_between_same() {
        let tc = ndf25();
        assert_eq!(tc.offset_between("00:01:00:00", "00:01:00:00"), Some(0));
    }

    #[test]
    fn test_timecode_conflict_is_significant() {
        let conflict = TimecodeConflict {
            clip_id: 1,
            expected_tc: "00:00:01:00".to_string(),
            found_tc: "00:00:01:10".to_string(),
            offset_frames: 10,
        };
        assert!(conflict.is_significant(5));
        assert!(!conflict.is_significant(15));
    }

    #[test]
    fn test_report_no_conflicts() {
        let report = TimecodeConformReport::new(vec![], 10);
        assert!(!report.has_conflicts());
        assert!((report.conflict_rate()).abs() < 1e-9);
    }

    #[test]
    fn test_report_conflict_rate() {
        let conflicts = vec![TimecodeConflict {
            clip_id: 1,
            expected_tc: "00:00:00:00".to_string(),
            found_tc: "00:00:01:00".to_string(),
            offset_frames: 25,
        }];
        let report = TimecodeConformReport::new(conflicts, 4);
        assert!(report.has_conflicts());
        assert!((report.conflict_rate() - 0.25).abs() < 1e-9);
    }
}
