//! OxiMedia Timecode - LTC and VITC reading and writing
//!
//! This crate provides SMPTE 12M compliant timecode reading and writing for:
//! - LTC (Linear Timecode) - audio-based timecode
//! - VITC (Vertical Interval Timecode) - video line-based timecode
//!
//! # Features
//! - All standard frame rates (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)
//! - Drop frame and non-drop frame support
//! - User bits encoding/decoding
//! - Real-time capable
//! - No unsafe code
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    dead_code,
    clippy::pedantic
)]

pub mod burn_in;
pub mod continuity;
pub mod drop_frame;
pub mod duration;
pub mod frame_offset;
pub mod frame_rate;
pub mod ltc;
pub mod ltc_encoder;
pub mod ltc_parser;
pub mod midi_timecode;
pub mod reader;
pub mod sync;
pub mod sync_map;
pub mod tc_calculator;
pub mod tc_compare;
pub mod tc_convert;
pub mod tc_drift;
pub mod tc_interpolate;
pub mod tc_math;
pub mod tc_metadata;
pub mod tc_range;
pub mod tc_smpte_ranges;
pub mod tc_validator;
pub mod timecode_calculator;
pub mod timecode_format;
pub mod timecode_range;
pub mod vitc;

use std::fmt;

/// SMPTE timecode frame rates
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameRate {
    /// 23.976 fps (film transferred to NTSC)
    Fps23976,
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps (NTSC drop frame)
    Fps2997DF,
    /// 29.97 fps (NTSC non-drop frame)
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

impl FrameRate {
    /// Get the nominal frame rate as a float
    pub fn as_float(&self) -> f64 {
        match self {
            FrameRate::Fps23976 => 23.976,
            FrameRate::Fps24 => 24.0,
            FrameRate::Fps25 => 25.0,
            FrameRate::Fps2997DF | FrameRate::Fps2997NDF => 29.97,
            FrameRate::Fps30 => 30.0,
            FrameRate::Fps50 => 50.0,
            FrameRate::Fps5994 => 59.94,
            FrameRate::Fps60 => 60.0,
        }
    }

    /// Get the exact frame rate as a rational (numerator, denominator)
    pub fn as_rational(&self) -> (u32, u32) {
        match self {
            FrameRate::Fps23976 => (24000, 1001),
            FrameRate::Fps24 => (24, 1),
            FrameRate::Fps25 => (25, 1),
            FrameRate::Fps2997DF | FrameRate::Fps2997NDF => (30000, 1001),
            FrameRate::Fps30 => (30, 1),
            FrameRate::Fps50 => (50, 1),
            FrameRate::Fps5994 => (60000, 1001),
            FrameRate::Fps60 => (60, 1),
        }
    }

    /// Check if this is a drop frame rate
    pub fn is_drop_frame(&self) -> bool {
        matches!(self, FrameRate::Fps2997DF)
    }

    /// Get the number of frames per second (rounded)
    pub fn frames_per_second(&self) -> u32 {
        match self {
            FrameRate::Fps23976 => 24,
            FrameRate::Fps24 => 24,
            FrameRate::Fps25 => 25,
            FrameRate::Fps2997DF | FrameRate::Fps2997NDF => 30,
            FrameRate::Fps30 => 30,
            FrameRate::Fps50 => 50,
            FrameRate::Fps5994 => 60,
            FrameRate::Fps60 => 60,
        }
    }
}

/// SMPTE timecode structure
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0 to frame_rate - 1)
    pub frames: u8,
    /// Frame rate
    pub frame_rate: FrameRateInfo,
    /// User bits (32 bits)
    pub user_bits: u32,
}

/// Frame rate information for timecode
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FrameRateInfo {
    /// Frames per second (rounded)
    pub fps: u8,
    /// Drop frame flag
    pub drop_frame: bool,
}

impl Timecode {
    /// Create a new timecode
    pub fn new(
        hours: u8,
        minutes: u8,
        seconds: u8,
        frames: u8,
        frame_rate: FrameRate,
    ) -> Result<Self, TimecodeError> {
        let fps = frame_rate.frames_per_second() as u8;

        if hours > 23 {
            return Err(TimecodeError::InvalidHours);
        }
        if minutes > 59 {
            return Err(TimecodeError::InvalidMinutes);
        }
        if seconds > 59 {
            return Err(TimecodeError::InvalidSeconds);
        }
        if frames >= fps {
            return Err(TimecodeError::InvalidFrames);
        }

        // Validate drop frame rules
        if frame_rate.is_drop_frame() {
            // Frames 0 and 1 are dropped at the start of each minute, except minutes 0, 10, 20, 30, 40, 50
            if seconds == 0 && frames < 2 && !minutes.is_multiple_of(10) {
                return Err(TimecodeError::InvalidDropFrame);
            }
        }

        Ok(Timecode {
            hours,
            minutes,
            seconds,
            frames,
            frame_rate: FrameRateInfo {
                fps,
                drop_frame: frame_rate.is_drop_frame(),
            },
            user_bits: 0,
        })
    }

    /// Create timecode with user bits
    pub fn with_user_bits(mut self, user_bits: u32) -> Self {
        self.user_bits = user_bits;
        self
    }

    /// Convert to total frames since midnight
    pub fn to_frames(&self) -> u64 {
        let fps = self.frame_rate.fps as u64;
        let mut total = self.hours as u64 * 3600 * fps;
        total += self.minutes as u64 * 60 * fps;
        total += self.seconds as u64 * fps;
        total += self.frames as u64;

        // Adjust for drop frame
        if self.frame_rate.drop_frame {
            // Drop 2 frames per minute except every 10th minute
            let total_minutes = self.hours as u64 * 60 + self.minutes as u64;
            let dropped_frames = 2 * (total_minutes - total_minutes / 10);
            total -= dropped_frames;
        }

        total
    }

    /// Create from total frames since midnight
    pub fn from_frames(frames: u64, frame_rate: FrameRate) -> Result<Self, TimecodeError> {
        let fps = frame_rate.frames_per_second() as u64;
        let mut remaining = frames;

        // Adjust for drop frame
        if frame_rate.is_drop_frame() {
            // This is an approximation; exact calculation is complex
            let frames_per_minute = fps * 60 - 2;
            let frames_per_10_minutes = frames_per_minute * 9 + fps * 60;

            let ten_minute_blocks = remaining / frames_per_10_minutes;
            remaining += ten_minute_blocks * 18;

            let remaining_in_block = remaining % frames_per_10_minutes;
            if remaining_in_block >= fps * 60 {
                let extra_minutes = (remaining_in_block - fps * 60) / frames_per_minute;
                remaining += (extra_minutes + 1) * 2;
            }
        }

        let hours = (remaining / (fps * 3600)) as u8;
        remaining %= fps * 3600;
        let minutes = (remaining / (fps * 60)) as u8;
        remaining %= fps * 60;
        let seconds = (remaining / fps) as u8;
        let frame = (remaining % fps) as u8;

        Self::new(hours, minutes, seconds, frame, frame_rate)
    }

    /// Increment by one frame
    pub fn increment(&mut self) -> Result<(), TimecodeError> {
        self.frames += 1;

        if self.frames >= self.frame_rate.fps {
            self.frames = 0;
            self.seconds += 1;

            if self.seconds >= 60 {
                self.seconds = 0;
                self.minutes += 1;

                // Handle drop frame
                if self.frame_rate.drop_frame && !self.minutes.is_multiple_of(10) {
                    self.frames = 2; // Skip frames 0 and 1
                }

                if self.minutes >= 60 {
                    self.minutes = 0;
                    self.hours += 1;

                    if self.hours >= 24 {
                        self.hours = 0;
                    }
                }
            }
        }

        Ok(())
    }

    /// Decrement by one frame
    pub fn decrement(&mut self) -> Result<(), TimecodeError> {
        if self.frames > 0 {
            self.frames -= 1;

            // Check if we're in a drop frame position
            if self.frame_rate.drop_frame
                && self.seconds == 0
                && self.frames == 1
                && !self.minutes.is_multiple_of(10)
            {
                self.frames = self.frame_rate.fps - 1;
                if self.seconds > 0 {
                    self.seconds -= 1;
                } else {
                    self.seconds = 59;
                    if self.minutes > 0 {
                        self.minutes -= 1;
                    } else {
                        self.minutes = 59;
                        if self.hours > 0 {
                            self.hours -= 1;
                        } else {
                            self.hours = 23;
                        }
                    }
                }
            }
        } else if self.seconds > 0 {
            self.seconds -= 1;
            self.frames = self.frame_rate.fps - 1;
        } else {
            self.seconds = 59;
            self.frames = self.frame_rate.fps - 1;

            if self.minutes > 0 {
                self.minutes -= 1;
            } else {
                self.minutes = 59;
                if self.hours > 0 {
                    self.hours -= 1;
                } else {
                    self.hours = 23;
                }
            }
        }

        Ok(())
    }
}

impl fmt::Display for Timecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let separator = if self.frame_rate.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

/// Timecode reader trait
pub trait TimecodeReader {
    /// Read the next timecode from the source
    fn read_timecode(&mut self) -> Result<Option<Timecode>, TimecodeError>;

    /// Get the current frame rate
    fn frame_rate(&self) -> FrameRate;

    /// Check if the reader is synchronized
    fn is_synchronized(&self) -> bool;
}

/// Timecode writer trait
pub trait TimecodeWriter {
    /// Write a timecode to the output
    fn write_timecode(&mut self, timecode: &Timecode) -> Result<(), TimecodeError>;

    /// Get the current frame rate
    fn frame_rate(&self) -> FrameRate;

    /// Flush any buffered data
    fn flush(&mut self) -> Result<(), TimecodeError>;
}

/// Timecode errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimecodeError {
    /// Invalid hours value
    InvalidHours,
    /// Invalid minutes value
    InvalidMinutes,
    /// Invalid seconds value
    InvalidSeconds,
    /// Invalid frames value
    InvalidFrames,
    /// Invalid drop frame timecode
    InvalidDropFrame,
    /// Sync word not found
    SyncNotFound,
    /// CRC error
    CrcError,
    /// Buffer too small
    BufferTooSmall,
    /// Invalid configuration
    InvalidConfiguration,
    /// IO error
    IoError(String),
    /// Not synchronized
    NotSynchronized,
}

impl fmt::Display for TimecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimecodeError::InvalidHours => write!(f, "Invalid hours value"),
            TimecodeError::InvalidMinutes => write!(f, "Invalid minutes value"),
            TimecodeError::InvalidSeconds => write!(f, "Invalid seconds value"),
            TimecodeError::InvalidFrames => write!(f, "Invalid frames value"),
            TimecodeError::InvalidDropFrame => write!(f, "Invalid drop frame timecode"),
            TimecodeError::SyncNotFound => write!(f, "Sync word not found"),
            TimecodeError::CrcError => write!(f, "CRC error"),
            TimecodeError::BufferTooSmall => write!(f, "Buffer too small"),
            TimecodeError::InvalidConfiguration => write!(f, "Invalid configuration"),
            TimecodeError::IoError(e) => write!(f, "IO error: {}", e),
            TimecodeError::NotSynchronized => write!(f, "Not synchronized"),
        }
    }
}

impl std::error::Error for TimecodeError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_creation() {
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("valid timecode");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
    }

    #[test]
    fn test_timecode_display() {
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("valid timecode");
        assert_eq!(tc.to_string(), "01:02:03:04");

        let tc_df = Timecode::new(1, 2, 3, 4, FrameRate::Fps2997DF).expect("valid timecode");
        assert_eq!(tc_df.to_string(), "01:02:03;04");
    }

    #[test]
    fn test_timecode_increment() {
        let mut tc = Timecode::new(0, 0, 0, 24, FrameRate::Fps25).expect("valid timecode");
        tc.increment().expect("increment should succeed");
        assert_eq!(tc.frames, 0);
        assert_eq!(tc.seconds, 1);
    }

    #[test]
    fn test_frame_rate() {
        assert_eq!(FrameRate::Fps25.as_float(), 25.0);
        assert_eq!(FrameRate::Fps2997DF.as_float(), 29.97);
        assert!(FrameRate::Fps2997DF.is_drop_frame());
        assert!(!FrameRate::Fps2997NDF.is_drop_frame());
    }
}
