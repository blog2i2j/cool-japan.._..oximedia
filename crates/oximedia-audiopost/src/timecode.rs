//! SMPTE timecode support for audio post-production.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// SMPTE timecode representation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0-fps)
    pub frames: u8,
    /// Frame rate
    pub fps: f64,
    /// Drop frame flag
    pub drop_frame: bool,
}

impl Timecode {
    /// Create a new timecode
    ///
    /// # Errors
    ///
    /// Returns an error if the timecode values are invalid
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        fps: f64,
        drop_frame: bool,
    ) -> AudioPostResult<Self> {
        if hours > 23 {
            return Err(AudioPostError::InvalidTimecode(format!(
                "Hours must be 0-23, got {hours}"
            )));
        }
        if minutes > 59 {
            return Err(AudioPostError::InvalidTimecode(format!(
                "Minutes must be 0-59, got {minutes}"
            )));
        }
        if seconds > 59 {
            return Err(AudioPostError::InvalidTimecode(format!(
                "Seconds must be 0-59, got {seconds}"
            )));
        }
        if f64::from(frames) >= fps {
            return Err(AudioPostError::InvalidTimecode(format!(
                "Frames must be 0-{}, got {frames}",
                fps as u8
            )));
        }

        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            fps,
            drop_frame,
        })
    }

    /// Create timecode from total frames
    pub fn from_frames(total_frames: u64, fps: f64) -> Self {
        let frames_per_second = fps as u64;
        let frames_per_minute = frames_per_second * 60;
        let frames_per_hour = frames_per_minute * 60;

        let hours = (total_frames / frames_per_hour) as u8;
        let remaining = total_frames % frames_per_hour;
        let minutes = (remaining / frames_per_minute) as u8;
        let remaining = remaining % frames_per_minute;
        let seconds = (remaining / frames_per_second) as u8;
        let frames = (remaining % frames_per_second) as u8;

        Self {
            hours,
            minutes,
            seconds,
            frames,
            fps,
            drop_frame: false,
        }
    }

    /// Create timecode from seconds
    pub fn from_seconds(total_seconds: f64, fps: f64) -> Self {
        let total_frames = (total_seconds * fps) as u64;
        Self::from_frames(total_frames, fps)
    }

    /// Convert timecode to total frames
    #[must_use]
    pub fn to_frames(&self) -> u64 {
        let fps = self.fps as u64;
        u64::from(self.hours) * fps * 3600
            + u64::from(self.minutes) * fps * 60
            + u64::from(self.seconds) * fps
            + u64::from(self.frames)
    }

    /// Convert timecode to seconds
    #[must_use]
    pub fn to_seconds(&self) -> f64 {
        self.to_frames() as f64 / self.fps
    }

    /// Add an offset to this timecode
    #[must_use]
    pub fn add_offset(&self, offset_frames: i64) -> Self {
        let current_frames = self.to_frames() as i64;
        let new_frames = (current_frames + offset_frames).max(0) as u64;
        Self::from_frames(new_frames, self.fps)
    }

    /// Calculate the difference between two timecodes in frames
    #[must_use]
    pub fn difference(&self, other: &Self) -> i64 {
        self.to_frames() as i64 - other.to_frames() as i64
    }
}

impl fmt::Display for Timecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let separator = if self.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

impl PartialOrd for Timecode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.to_frames().cmp(&other.to_frames()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = Timecode::new(1, 30, 45, 12, 24.0, false).expect("failed to create");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);
    }

    #[test]
    fn test_timecode_from_frames() {
        let tc = Timecode::from_frames(3600 * 24, 24.0);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 0);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_timecode_to_frames() {
        let tc = Timecode::new(1, 0, 0, 0, 24.0, false).expect("failed to create");
        assert_eq!(tc.to_frames(), 3600 * 24);
    }

    #[test]
    fn test_timecode_to_seconds() {
        let tc = Timecode::new(0, 1, 0, 0, 24.0, false).expect("failed to create");
        assert!((tc.to_seconds() - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_timecode_add_offset() {
        let tc = Timecode::from_frames(100, 24.0);
        let new_tc = tc.add_offset(50);
        assert_eq!(new_tc.to_frames(), 150);
    }

    #[test]
    fn test_timecode_difference() {
        let tc1 = Timecode::from_frames(100, 24.0);
        let tc2 = Timecode::from_frames(50, 24.0);
        assert_eq!(tc1.difference(&tc2), 50);
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 30, 45, 12, 24.0, false).expect("failed to create");
        assert_eq!(format!("{tc}"), "01:30:45:12");
    }

    #[test]
    fn test_timecode_display_drop_frame() {
        let tc = Timecode::new(1, 30, 45, 12, 29.97, true).expect("failed to create");
        assert_eq!(format!("{tc}"), "01:30:45;12");
    }

    #[test]
    fn test_invalid_timecode() {
        assert!(Timecode::new(24, 0, 0, 0, 24.0, false).is_err());
        assert!(Timecode::new(0, 60, 0, 0, 24.0, false).is_err());
        assert!(Timecode::new(0, 0, 60, 0, 24.0, false).is_err());
        assert!(Timecode::new(0, 0, 0, 24, 24.0, false).is_err());
    }

    #[test]
    fn test_timecode_comparison() {
        let tc1 = Timecode::from_frames(100, 24.0);
        let tc2 = Timecode::from_frames(50, 24.0);
        assert!(tc1 > tc2);
    }
}
