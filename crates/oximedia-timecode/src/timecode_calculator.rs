//! Timecode arithmetic and conversion utilities.
//!
//! Provides `TimecodeValue` for frame-accurate timecode calculations,
//! including addition, subtraction, and string formatting.

use std::fmt;

/// A timecode value with arithmetic support.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub struct TimecodeValue {
    /// Hours component (0-23)
    pub hh: u8,
    /// Minutes component (0-59)
    pub mm: u8,
    /// Seconds component (0-59)
    pub ss: u8,
    /// Frames component (0 to fps-1)
    pub ff: u8,
    /// Frames per second (e.g. 25.0, 29.97, 30.0)
    pub fps: f32,
    /// Whether this timecode uses drop-frame notation
    pub drop_frame: bool,
}

impl TimecodeValue {
    /// Create a new timecode value.
    #[must_use]
    pub fn new(hh: u8, mm: u8, ss: u8, ff: u8, fps: f32, drop_frame: bool) -> Self {
        Self {
            hh,
            mm,
            ss,
            ff,
            fps,
            drop_frame,
        }
    }

    /// Get the integer frames per second.
    #[must_use]
    fn fps_int(&self) -> u64 {
        self.fps.ceil() as u64
    }

    /// Get total frames per day (wrapping boundary).
    #[must_use]
    fn frames_per_day(&self) -> u64 {
        self.fps_int() * 3600 * 24
    }

    /// Convert to total frame count since 00:00:00:00.
    ///
    /// For drop-frame timecode, applies the standard SMPTE drop-frame adjustment.
    #[must_use]
    pub fn to_frame_count(&self) -> u64 {
        let fps = self.fps_int();
        let hh = u64::from(self.hh);
        let mm = u64::from(self.mm);
        let ss = u64::from(self.ss);
        let ff = u64::from(self.ff);

        let raw = hh * 3600 * fps + mm * 60 * fps + ss * fps + ff;

        if self.drop_frame {
            let total_minutes = hh * 60 + mm;
            let dropped = 2 * (total_minutes - total_minutes / 10);
            raw - dropped
        } else {
            raw
        }
    }

    /// Create a timecode value from a frame count.
    #[must_use]
    pub fn from_frame_count(frames: u64, fps: f32, drop_frame: bool) -> Self {
        let fps_int = fps.ceil() as u64;
        let mut remaining = frames;

        // Apply drop-frame adjustment
        if drop_frame {
            let frames_per_min = fps_int * 60 - 2;
            let frames_per_10_min = frames_per_min * 9 + fps_int * 60;

            let ten_min_blocks = remaining / frames_per_10_min;
            remaining += ten_min_blocks * 18;

            let remaining_in_block = remaining % frames_per_10_min;
            if remaining_in_block >= fps_int * 60 {
                let extra_minutes = (remaining_in_block - fps_int * 60) / frames_per_min;
                remaining += (extra_minutes + 1) * 2;
            }
        }

        let hh = ((remaining / (fps_int * 3600)) % 24) as u8;
        remaining %= fps_int * 3600;
        let mm = (remaining / (fps_int * 60)) as u8;
        remaining %= fps_int * 60;
        let ss = (remaining / fps_int) as u8;
        let ff = (remaining % fps_int) as u8;

        Self::new(hh, mm, ss, ff, fps, drop_frame)
    }

    /// Add a number of frames (positive or negative), wrapping at 24 hours.
    #[must_use]
    pub fn add_frames(&self, frames: i64) -> Self {
        let total = self.to_frame_count() as i64;
        let frames_per_day = self.frames_per_day() as i64;

        // Add frames and wrap within [0, frames_per_day)
        let new_total = ((total + frames) % frames_per_day + frames_per_day) % frames_per_day;

        Self::from_frame_count(new_total as u64, self.fps, self.drop_frame)
    }

    /// Compute the signed frame difference between self and another timecode.
    ///
    /// Returns a positive value if self is later than `other`.
    #[must_use]
    pub fn subtract(&self, other: &Self) -> i64 {
        self.to_frame_count() as i64 - other.to_frame_count() as i64
    }

    /// Convert to a formatted timecode string.
    ///
    /// Uses colons for non-drop-frame and semicolons for drop-frame.
    #[must_use]
    pub fn to_string_formatted(&self) -> String {
        let sep = if self.drop_frame { ';' } else { ':' };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hh, self.mm, self.ss, sep, self.ff
        )
    }

    /// Parse a timecode string.
    ///
    /// Detects drop-frame from the presence of a semicolon before the frame count.
    /// Format: `HH:MM:SS:FF` (NDF) or `HH:MM:SS;FF` (DF).
    #[must_use]
    pub fn parse(s: &str, fps: f32) -> Option<Self> {
        // Detect drop frame: last separator is ';'
        let drop_frame = s.contains(';');

        // Replace semicolons with colons for uniform splitting
        let normalized = s.replace(';', ":");
        let parts: Vec<&str> = normalized.split(':').collect();

        if parts.len() != 4 {
            return None;
        }

        let hh: u8 = parts[0].parse().ok()?;
        let mm: u8 = parts[1].parse().ok()?;
        let ss: u8 = parts[2].parse().ok()?;
        let ff: u8 = parts[3].parse().ok()?;

        if hh > 23 || mm > 59 || ss > 59 {
            return None;
        }

        Some(Self::new(hh, mm, ss, ff, fps, drop_frame))
    }
}

impl fmt::Display for TimecodeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_formatted())
    }
}

/// Duration utilities for converting timecodes to wall-clock time.
#[allow(dead_code)]
pub struct Duration;

impl Duration {
    /// Convert a timecode value to elapsed time in seconds.
    ///
    /// For 29.97 fps drop-frame, this gives accurate real-time seconds.
    #[must_use]
    pub fn from_timecode(tc: &TimecodeValue) -> f64 {
        let frame_count = tc.to_frame_count();
        f64::from(frame_count as u32) / f64::from(tc.fps)
    }

    /// Convert seconds to a timecode value.
    #[must_use]
    pub fn to_timecode(seconds: f64, fps: f32, drop_frame: bool) -> TimecodeValue {
        let total_frames = (seconds * f64::from(fps)).round() as u64;
        TimecodeValue::from_frame_count(total_frames, fps, drop_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_value_new() {
        let tc = TimecodeValue::new(1, 2, 3, 4, 25.0, false);
        assert_eq!(tc.hh, 1);
        assert_eq!(tc.mm, 2);
        assert_eq!(tc.ss, 3);
        assert_eq!(tc.ff, 4);
        assert!((tc.fps - 25.0).abs() < f32::EPSILON);
        assert!(!tc.drop_frame);
    }

    #[test]
    fn test_to_frame_count_ndf() {
        let tc = TimecodeValue::new(0, 0, 1, 0, 25.0, false);
        assert_eq!(tc.to_frame_count(), 25);
    }

    #[test]
    fn test_to_frame_count_one_hour() {
        let tc = TimecodeValue::new(1, 0, 0, 0, 30.0, false);
        assert_eq!(tc.to_frame_count(), 3600 * 30);
    }

    #[test]
    fn test_from_frame_count_ndf() {
        let tc = TimecodeValue::from_frame_count(25, 25.0, false);
        assert_eq!(tc.hh, 0);
        assert_eq!(tc.mm, 0);
        assert_eq!(tc.ss, 1);
        assert_eq!(tc.ff, 0);
    }

    #[test]
    fn test_frame_count_roundtrip_ndf() {
        let original = TimecodeValue::new(1, 30, 45, 12, 25.0, false);
        let frames = original.to_frame_count();
        let recovered = TimecodeValue::from_frame_count(frames, 25.0, false);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_add_frames_forward() {
        let tc = TimecodeValue::new(0, 0, 0, 0, 25.0, false);
        let tc2 = tc.add_frames(25);
        assert_eq!(tc2.ss, 1);
        assert_eq!(tc2.ff, 0);
    }

    #[test]
    fn test_add_frames_backward() {
        let tc = TimecodeValue::new(0, 0, 1, 0, 25.0, false);
        let tc2 = tc.add_frames(-25);
        assert_eq!(tc2.hh, 0);
        assert_eq!(tc2.mm, 0);
        assert_eq!(tc2.ss, 0);
        assert_eq!(tc2.ff, 0);
    }

    #[test]
    fn test_add_frames_wrap_at_24h() {
        // 24h in 25fps = 24 * 3600 * 25 frames
        let tc = TimecodeValue::new(23, 59, 59, 24, 25.0, false);
        let tc2 = tc.add_frames(1); // Should wrap to 00:00:00:00
        assert_eq!(tc2.hh, 0);
        assert_eq!(tc2.mm, 0);
        assert_eq!(tc2.ss, 0);
        assert_eq!(tc2.ff, 0);
    }

    #[test]
    fn test_subtract() {
        let tc1 = TimecodeValue::new(0, 0, 1, 0, 25.0, false);
        let tc2 = TimecodeValue::new(0, 0, 0, 0, 25.0, false);
        assert_eq!(tc1.subtract(&tc2), 25);
        assert_eq!(tc2.subtract(&tc1), -25);
    }

    #[test]
    fn test_display_ndf() {
        let tc = TimecodeValue::new(1, 2, 3, 4, 25.0, false);
        assert_eq!(tc.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_display_df() {
        let tc = TimecodeValue::new(1, 2, 3, 4, 29.97, true);
        assert_eq!(tc.to_string(), "01:02:03;04");
    }

    #[test]
    fn test_parse_ndf() {
        let tc = TimecodeValue::parse("01:02:03:04", 25.0).expect("valid timecode value");
        assert_eq!(tc.hh, 1);
        assert_eq!(tc.mm, 2);
        assert_eq!(tc.ss, 3);
        assert_eq!(tc.ff, 4);
        assert!(!tc.drop_frame);
    }

    #[test]
    fn test_parse_df() {
        let tc = TimecodeValue::parse("01:02:03;04", 29.97).expect("valid timecode value");
        assert_eq!(tc.hh, 1);
        assert_eq!(tc.mm, 2);
        assert_eq!(tc.ss, 3);
        assert_eq!(tc.ff, 4);
        assert!(tc.drop_frame);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(TimecodeValue::parse("invalid", 25.0).is_none());
        assert!(TimecodeValue::parse("25:00:00:00", 25.0).is_none()); // hours > 23
        assert!(TimecodeValue::parse("", 25.0).is_none());
    }

    #[test]
    fn test_duration_from_timecode() {
        let tc = TimecodeValue::new(0, 0, 1, 0, 25.0, false);
        let secs = Duration::from_timecode(&tc);
        assert!((secs - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_duration_to_timecode() {
        let tc = Duration::to_timecode(1.0, 25.0, false);
        assert_eq!(tc.ss, 1);
        assert_eq!(tc.ff, 0);
    }

    #[test]
    fn test_parse_display_roundtrip() {
        let original = "01:30:45:12";
        let tc = TimecodeValue::parse(original, 25.0).expect("valid timecode value");
        assert_eq!(tc.to_string(), original);
    }
}
