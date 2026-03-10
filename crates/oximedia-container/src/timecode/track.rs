//! Timecode track handling.
//!
//! Professional timecode support for broadcast workflows.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};

/// Timecode format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeFormat {
    /// 24 fps non-drop.
    Fps24,
    /// 25 fps (PAL).
    Fps25,
    /// 30 fps non-drop.
    Fps30NonDrop,
    /// 30 fps drop-frame (29.97).
    Fps30Drop,
    /// 60 fps non-drop.
    Fps60NonDrop,
    /// 60 fps drop-frame (59.94).
    Fps60Drop,
}

impl TimecodeFormat {
    /// Returns the frame rate as frames per second.
    #[must_use]
    pub const fn fps(&self) -> f64 {
        match self {
            Self::Fps24 => 24.0,
            Self::Fps25 => 25.0,
            Self::Fps30NonDrop => 30.0,
            Self::Fps30Drop => 29.97,
            Self::Fps60NonDrop => 60.0,
            Self::Fps60Drop => 59.94,
        }
    }

    /// Returns true if this is a drop-frame format.
    #[must_use]
    pub const fn is_drop_frame(&self) -> bool {
        matches!(self, Self::Fps30Drop | Self::Fps60Drop)
    }
}

/// A timecode value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    /// Hours (0-23).
    pub hours: u8,
    /// Minutes (0-59).
    pub minutes: u8,
    /// Seconds (0-59).
    pub seconds: u8,
    /// Frames (0-fps-1).
    pub frames: u8,
    /// Format.
    pub format: TimecodeFormat,
}

impl Timecode {
    /// Creates a new timecode.
    ///
    /// # Errors
    ///
    /// Returns `Err` if hours >= 24, minutes >= 60, seconds >= 60, or frames >= fps.
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        format: TimecodeFormat,
    ) -> OxiResult<Self> {
        // Validate ranges
        if hours >= 24 {
            return Err(OxiError::InvalidData("Hours must be 0-23".into()));
        }
        if minutes >= 60 {
            return Err(OxiError::InvalidData("Minutes must be 0-59".into()));
        }
        if seconds >= 60 {
            return Err(OxiError::InvalidData("Seconds must be 0-59".into()));
        }

        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let max_frames = format.fps() as u8;

        if frames >= max_frames {
            return Err(OxiError::InvalidData(format!(
                "Frames must be 0-{}",
                max_frames - 1
            )));
        }

        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            format,
        })
    }

    /// Creates a timecode from frame count.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_frame_count(frame_count: u64, format: TimecodeFormat) -> Self {
        let fps = format.fps() as u64;
        let total_seconds = frame_count / fps;
        let frames = (frame_count % fps) as u8;

        let hours = (total_seconds / 3600) as u8;
        let minutes = ((total_seconds % 3600) / 60) as u8;
        let seconds = (total_seconds % 60) as u8;

        Self {
            hours,
            minutes,
            seconds,
            frames,
            format,
        }
    }

    /// Converts timecode to frame count.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn to_frame_count(&self) -> u64 {
        let fps = self.format.fps();
        let total_seconds =
            u64::from(self.hours) * 3600 + u64::from(self.minutes) * 60 + u64::from(self.seconds);
        (total_seconds as f64 * fps) as u64 + u64::from(self.frames)
    }

    /// Formats the timecode as a string.
    #[must_use]
    pub fn format_string(&self) -> String {
        let separator = if self.format.is_drop_frame() {
            ';'
        } else {
            ':'
        };
        format!(
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }

    /// Parses a timecode string.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the string is not in `HH:MM:SS:FF` format or contains invalid values.
    pub fn from_string(s: &str, format: TimecodeFormat) -> OxiResult<Self> {
        let parts: Vec<&str> = s.split([':', ';']).collect();

        if parts.len() != 4 {
            return Err(OxiError::InvalidData(
                "Timecode must be in format HH:MM:SS:FF".into(),
            ));
        }

        let hours = parts[0]
            .parse()
            .map_err(|_| OxiError::InvalidData("Invalid hours".into()))?;
        let minutes = parts[1]
            .parse()
            .map_err(|_| OxiError::InvalidData("Invalid minutes".into()))?;
        let seconds = parts[2]
            .parse()
            .map_err(|_| OxiError::InvalidData("Invalid seconds".into()))?;
        let frames = parts[3]
            .parse()
            .map_err(|_| OxiError::InvalidData("Invalid frames".into()))?;

        Self::new(hours, minutes, seconds, frames, format)
    }
}

/// Timecode track in a container.
#[derive(Debug, Clone)]
pub struct TimecodeTrack {
    format: TimecodeFormat,
    start_timecode: Timecode,
    timecodes: Vec<(u64, Timecode)>, // (sample_number, timecode)
}

impl TimecodeTrack {
    /// Creates a new timecode track.
    #[must_use]
    pub const fn new(format: TimecodeFormat, start_timecode: Timecode) -> Self {
        Self {
            format,
            start_timecode,
            timecodes: Vec::new(),
        }
    }

    /// Adds a timecode at a specific sample.
    pub fn add_timecode(&mut self, sample_number: u64, timecode: Timecode) {
        self.timecodes.push((sample_number, timecode));
    }

    /// Gets the timecode at a specific sample.
    #[must_use]
    pub fn get_timecode(&self, sample_number: u64) -> Option<&Timecode> {
        self.timecodes
            .iter()
            .rev()
            .find(|(s, _)| *s <= sample_number)
            .map(|(_, tc)| tc)
    }

    /// Returns the start timecode.
    #[must_use]
    pub const fn start_timecode(&self) -> &Timecode {
        &self.start_timecode
    }

    /// Returns the format.
    #[must_use]
    pub const fn format(&self) -> TimecodeFormat {
        self.format
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_format() {
        assert_eq!(TimecodeFormat::Fps24.fps(), 24.0);
        assert_eq!(TimecodeFormat::Fps30Drop.fps(), 29.97);
        assert!(TimecodeFormat::Fps30Drop.is_drop_frame());
        assert!(!TimecodeFormat::Fps24.is_drop_frame());
    }

    #[test]
    fn test_timecode_creation() {
        let tc =
            Timecode::new(1, 30, 45, 12, TimecodeFormat::Fps24).expect("operation should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);

        // Invalid values
        assert!(Timecode::new(24, 0, 0, 0, TimecodeFormat::Fps24).is_err());
        assert!(Timecode::new(0, 60, 0, 0, TimecodeFormat::Fps24).is_err());
    }

    #[test]
    fn test_timecode_frame_count() {
        let tc = Timecode::from_frame_count(100, TimecodeFormat::Fps24);
        assert_eq!(tc.seconds, 4);
        assert_eq!(tc.frames, 4);

        let frame_count = tc.to_frame_count();
        assert_eq!(frame_count, 100);
    }

    #[test]
    fn test_timecode_string() {
        let tc =
            Timecode::new(1, 30, 45, 12, TimecodeFormat::Fps24).expect("operation should succeed");
        assert_eq!(tc.format_string(), "01:30:45:12");

        let tc_drop = Timecode::new(1, 30, 45, 12, TimecodeFormat::Fps30Drop)
            .expect("operation should succeed");
        assert_eq!(tc_drop.format_string(), "01:30:45;12");
    }

    #[test]
    fn test_timecode_parse() {
        let tc = Timecode::from_string("01:30:45:12", TimecodeFormat::Fps24)
            .expect("operation should succeed");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 30);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 12);

        assert!(Timecode::from_string("invalid", TimecodeFormat::Fps24).is_err());
    }

    #[test]
    fn test_timecode_track() {
        let start_tc =
            Timecode::new(0, 0, 0, 0, TimecodeFormat::Fps24).expect("operation should succeed");
        let mut track = TimecodeTrack::new(TimecodeFormat::Fps24, start_tc);

        let tc1 =
            Timecode::new(0, 0, 1, 0, TimecodeFormat::Fps24).expect("operation should succeed");
        track.add_timecode(24, tc1);

        let retrieved = track.get_timecode(24);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("operation should succeed").seconds, 1);
    }
}
