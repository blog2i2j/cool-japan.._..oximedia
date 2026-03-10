//! Timecode handling for EDL operations.
//!
//! This module provides EDL-specific timecode parsing and formatting,
//! building on the core timecode functionality from `oximedia-timecode`.

use crate::error::{EdlError, EdlResult};
use oximedia_timecode::{FrameRate, Timecode as CoreTimecode};
use std::fmt;
use std::str::FromStr;

/// EDL timecode wrapper with additional formatting and parsing capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdlTimecode {
    /// The underlying timecode.
    pub tc: CoreTimecode,
}

impl EdlTimecode {
    /// Create a new EDL timecode.
    ///
    /// # Errors
    ///
    /// Returns an error if the timecode values are invalid.
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: EdlFrameRate,
    ) -> EdlResult<Self> {
        let tc = CoreTimecode::new(hours, minutes, seconds, frames, frame_rate.to_frame_rate())
            .map_err(|e| EdlError::InvalidTimecode {
                line: 0,
                message: format!("{e}"),
            })?;

        Ok(Self { tc })
    }

    /// Parse a timecode string in EDL format (HH:MM:SS:FF or HH:MM:SS;FF).
    ///
    /// # Errors
    ///
    /// Returns an error if the timecode string is invalid.
    pub fn parse(s: &str, frame_rate: EdlFrameRate) -> EdlResult<Self> {
        let parts: Vec<&str> = s.split([':', ';']).collect();

        if parts.len() != 4 {
            return Err(EdlError::InvalidTimecode {
                line: 0,
                message: format!("Expected format HH:MM:SS:FF, got: {s}"),
            });
        }

        let hours = parts[0]
            .parse::<u8>()
            .map_err(|_| EdlError::InvalidTimecode {
                line: 0,
                message: format!("Invalid hours: {}", parts[0]),
            })?;

        let minutes = parts[1]
            .parse::<u8>()
            .map_err(|_| EdlError::InvalidTimecode {
                line: 0,
                message: format!("Invalid minutes: {}", parts[1]),
            })?;

        let seconds = parts[2]
            .parse::<u8>()
            .map_err(|_| EdlError::InvalidTimecode {
                line: 0,
                message: format!("Invalid seconds: {}", parts[2]),
            })?;

        let frames = parts[3]
            .parse::<u8>()
            .map_err(|_| EdlError::InvalidTimecode {
                line: 0,
                message: format!("Invalid frames: {}", parts[3]),
            })?;

        // The separator (';' vs ':') is optional notation; the frame_rate parameter
        // already carries drop-frame information from the FCM line, so don't reject
        // timecodes that use ':' in a drop-frame EDL (CMX3600 standard).
        Self::new(hours, minutes, seconds, frames, frame_rate)
    }

    /// Get the hours component.
    #[must_use]
    pub const fn hours(&self) -> u8 {
        self.tc.hours
    }

    /// Get the minutes component.
    #[must_use]
    pub const fn minutes(&self) -> u8 {
        self.tc.minutes
    }

    /// Get the seconds component.
    #[must_use]
    pub const fn seconds(&self) -> u8 {
        self.tc.seconds
    }

    /// Get the frames component.
    #[must_use]
    pub const fn frames(&self) -> u8 {
        self.tc.frames
    }

    /// Check if this is a drop frame timecode.
    #[must_use]
    pub const fn is_drop_frame(&self) -> bool {
        self.tc.frame_rate.drop_frame
    }

    /// Convert to total frames.
    #[must_use]
    pub fn to_frames(&self) -> u64 {
        self.tc.to_frames()
    }

    /// Create from total frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the frame count is invalid.
    pub fn from_frames(frames: u64, frame_rate: EdlFrameRate) -> EdlResult<Self> {
        let tc = CoreTimecode::from_frames(frames, frame_rate.to_frame_rate()).map_err(|e| {
            EdlError::InvalidTimecode {
                line: 0,
                message: format!("{e}"),
            }
        })?;

        Ok(Self { tc })
    }
}

impl fmt::Display for EdlTimecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.tc)
    }
}

impl PartialOrd for EdlTimecode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdlTimecode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_frames().cmp(&other.to_frames())
    }
}

/// EDL frame rate enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EdlFrameRate {
    /// 23.976 fps (film transferred to NTSC)
    Fps23976,
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps drop frame (NTSC)
    Fps2997DF,
    /// 29.97 fps non-drop frame (NTSC)
    Fps2997NDF,
    /// 30 fps
    Fps30,
    /// 50 fps (PAL progressive)
    Fps50,
    /// 59.94 fps (NTSC progressive)
    Fps5994,
    /// 60 fps
    Fps60,
}

impl EdlFrameRate {
    /// Convert to the core `FrameRate` type.
    #[must_use]
    pub const fn to_frame_rate(self) -> FrameRate {
        match self {
            Self::Fps23976 => FrameRate::Fps23976,
            Self::Fps24 => FrameRate::Fps24,
            Self::Fps25 => FrameRate::Fps25,
            Self::Fps2997DF => FrameRate::Fps2997DF,
            Self::Fps2997NDF => FrameRate::Fps2997NDF,
            Self::Fps30 => FrameRate::Fps30,
            Self::Fps50 => FrameRate::Fps50,
            Self::Fps5994 => FrameRate::Fps5994,
            Self::Fps60 => FrameRate::Fps60,
        }
    }

    /// Check if this is a drop frame rate.
    #[must_use]
    pub const fn is_drop_frame(self) -> bool {
        matches!(self, Self::Fps2997DF)
    }

    /// Get the frame rate as a float.
    #[must_use]
    pub fn as_float(self) -> f64 {
        self.to_frame_rate().as_float()
    }

    /// Get the nominal frames per second.
    #[must_use]
    pub const fn fps(self) -> u32 {
        match self {
            Self::Fps23976 | Self::Fps24 => 24,
            Self::Fps25 => 25,
            Self::Fps2997DF | Self::Fps2997NDF | Self::Fps30 => 30,
            Self::Fps50 => 50,
            Self::Fps5994 | Self::Fps60 => 60,
        }
    }

    /// Get the CMX EDL frame count mode string.
    #[must_use]
    pub const fn fcm_string(self) -> &'static str {
        match self {
            Self::Fps2997DF => "DROP FRAME",
            _ => "NON-DROP FRAME",
        }
    }
}

impl FromStr for EdlFrameRate {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "23.976" | "23976" => Ok(Self::Fps23976),
            "24" | "24.0" => Ok(Self::Fps24),
            "25" | "25.0" => Ok(Self::Fps25),
            "29.97DF" | "2997DF" | "DROP FRAME" => Ok(Self::Fps2997DF),
            "29.97NDF" | "2997NDF" | "29.97" | "NON-DROP FRAME" => Ok(Self::Fps2997NDF),
            "30" | "30.0" => Ok(Self::Fps30),
            "50" | "50.0" => Ok(Self::Fps50),
            "59.94" | "5994" => Ok(Self::Fps5994),
            "60" | "60.0" => Ok(Self::Fps60),
            _ => Err(EdlError::InvalidFrameRate(s.to_string())),
        }
    }
}

impl fmt::Display for EdlFrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Fps23976 => "23.976",
            Self::Fps24 => "24",
            Self::Fps25 => "25",
            Self::Fps2997DF => "29.97 DF",
            Self::Fps2997NDF => "29.97 NDF",
            Self::Fps30 => "30",
            Self::Fps50 => "50",
            Self::Fps5994 => "59.94",
            Self::Fps60 => "60",
        };
        write!(f, "{s}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = EdlTimecode::new(1, 2, 3, 4, EdlFrameRate::Fps25).expect("failed to create");
        assert_eq!(tc.hours(), 1);
        assert_eq!(tc.minutes(), 2);
        assert_eq!(tc.seconds(), 3);
        assert_eq!(tc.frames(), 4);
    }

    #[test]
    fn test_timecode_parsing() {
        let tc = EdlTimecode::parse("01:02:03:04", EdlFrameRate::Fps25).expect("failed to parse");
        assert_eq!(tc.hours(), 1);
        assert_eq!(tc.minutes(), 2);
        assert_eq!(tc.seconds(), 3);
        assert_eq!(tc.frames(), 4);
    }

    #[test]
    fn test_drop_frame_parsing() {
        let tc =
            EdlTimecode::parse("01:02:03;04", EdlFrameRate::Fps2997DF).expect("failed to parse");
        assert_eq!(tc.hours(), 1);
        assert!(tc.is_drop_frame());
    }

    #[test]
    fn test_timecode_display() {
        let tc = EdlTimecode::new(1, 2, 3, 4, EdlFrameRate::Fps25).expect("failed to create");
        assert_eq!(tc.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_drop_frame_display() {
        let tc = EdlTimecode::new(1, 2, 3, 4, EdlFrameRate::Fps2997DF).expect("failed to create");
        assert_eq!(tc.to_string(), "01:02:03;04");
    }

    #[test]
    fn test_frame_rate_parsing() {
        assert_eq!(
            "25".parse::<EdlFrameRate>()
                .expect("operation should succeed"),
            EdlFrameRate::Fps25
        );
        assert_eq!(
            "DROP FRAME"
                .parse::<EdlFrameRate>()
                .expect("operation should succeed"),
            EdlFrameRate::Fps2997DF
        );
    }

    #[test]
    fn test_timecode_comparison() {
        let tc1 = EdlTimecode::new(0, 0, 0, 10, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(0, 0, 0, 20, EdlFrameRate::Fps25).expect("failed to create");
        assert!(tc1 < tc2);
    }

    #[test]
    fn test_frame_conversion() {
        let tc = EdlTimecode::new(0, 0, 1, 0, EdlFrameRate::Fps25).expect("failed to create");
        let frames = tc.to_frames();
        assert_eq!(frames, 25);

        let tc2 = EdlTimecode::from_frames(frames, EdlFrameRate::Fps25)
            .expect("operation should succeed");
        assert_eq!(tc, tc2);
    }
}
