//! Fractional (sub-frame) timecode representation.
//!
//! Extends the classic SMPTE timecode model to support fractional frame counts,
//! which arise in variable-frame-rate and high-precision audio workflows.
//! Both drop-frame (DF) and non-drop-frame (NDF) string formatting is supported.

use std::fmt;

/// A timecode with fractional frame support.
///
/// Components:
/// - `hours`, `minutes`, `seconds` – the wall-clock portion
/// - `frame` – integer frame count within the second (0-indexed)
/// - `rate` – nominal frame rate (e.g. 24.0, 29.97, 30.0)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FractionalTimecode {
    /// Hours component (0–23).
    pub hours: u8,
    /// Minutes component (0–59).
    pub minutes: u8,
    /// Seconds component (0–59).
    pub seconds: u8,
    /// Frame number within the second (0-indexed).
    pub frame: u32,
    /// Nominal frame rate (frames per second).
    pub rate: f32,
}

impl FractionalTimecode {
    /// Create a new `FractionalTimecode`.
    ///
    /// `frame` is clamped to the valid range `[0, floor(rate) - 1]`.
    ///
    /// # Arguments
    ///
    /// * `hours`   – Hour component (0–23).
    /// * `minutes` – Minute component (0–59).
    /// * `seconds` – Second component (0–59).
    /// * `frame`   – Frame number within the second.
    /// * `rate`    – Frame rate in frames per second (must be > 0).
    #[must_use]
    pub fn new(hours: u8, minutes: u8, seconds: u8, frame: u32, rate: f32) -> Self {
        let max_frame = if rate > 0.0 {
            (rate.floor() as u32).saturating_sub(1)
        } else {
            0
        };
        Self {
            hours,
            minutes: minutes.min(59),
            seconds: seconds.min(59),
            frame: frame.min(max_frame),
            rate,
        }
    }

    /// Format the timecode in drop-frame (DF) notation: `HH:MM:SS;FF`.
    ///
    /// Drop-frame uses a semicolon before the frame count.
    #[must_use]
    pub fn to_string_df(&self) -> String {
        format!(
            "{:02}:{:02}:{:02};{:02}",
            self.hours, self.minutes, self.seconds, self.frame
        )
    }

    /// Format the timecode in non-drop-frame (NDF) notation: `HH:MM:SS:FF`.
    ///
    /// Non-drop-frame uses a colon before the frame count.
    #[must_use]
    pub fn to_string_ndf(&self) -> String {
        format!(
            "{:02}:{:02}:{:02}:{:02}",
            self.hours, self.minutes, self.seconds, self.frame
        )
    }

    /// Convert the timecode to an absolute frame number.
    ///
    /// Uses integer arithmetic based on `floor(rate)`.
    #[must_use]
    pub fn to_frame_number(&self) -> u64 {
        let fps = self.rate.floor() as u64;
        let total_seconds = self.hours as u64 * 3600
            + self.minutes as u64 * 60
            + self.seconds as u64;
        total_seconds * fps + self.frame as u64
    }

    /// Construct a `FractionalTimecode` from an absolute frame number and a rate.
    ///
    /// Uses integer arithmetic based on `floor(rate)`.
    #[must_use]
    pub fn from_frame_number(frame_number: u64, rate: f32) -> Self {
        let fps = rate.floor() as u64;
        if fps == 0 {
            return Self {
                hours: 0,
                minutes: 0,
                seconds: 0,
                frame: 0,
                rate,
            };
        }
        let total_seconds = frame_number / fps;
        let frame = (frame_number % fps) as u32;
        let seconds = (total_seconds % 60) as u8;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u8;
        let hours = ((total_minutes / 60) % 24) as u8;
        Self {
            hours,
            minutes,
            seconds,
            frame,
            rate,
        }
    }

    /// Return `true` if all components are within their valid ranges.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let max_frame = if self.rate > 0.0 {
            self.rate.floor() as u32
        } else {
            return false;
        };
        self.minutes <= 59
            && self.seconds <= 59
            && self.frame < max_frame
    }
}

impl fmt::Display for FractionalTimecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_string_ndf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_string_ndf() {
        let tc = FractionalTimecode::new(1, 2, 3, 15, 24.0);
        assert_eq!(tc.to_string_ndf(), "01:02:03:15");
    }

    #[test]
    fn test_to_string_df() {
        let tc = FractionalTimecode::new(1, 2, 3, 15, 29.97);
        assert_eq!(tc.to_string_df(), "01:02:03;15");
    }

    #[test]
    fn test_display_uses_ndf() {
        let tc = FractionalTimecode::new(0, 0, 5, 12, 25.0);
        assert_eq!(format!("{tc}"), "00:00:05:12");
    }

    #[test]
    fn test_frame_clamp() {
        // Rate 24 fps → max frame = 23
        let tc = FractionalTimecode::new(0, 0, 0, 30, 24.0);
        assert_eq!(tc.frame, 23);
    }

    #[test]
    fn test_to_frame_number_zero() {
        let tc = FractionalTimecode::new(0, 0, 0, 0, 24.0);
        assert_eq!(tc.to_frame_number(), 0);
    }

    #[test]
    fn test_to_frame_number_one_second() {
        let tc = FractionalTimecode::new(0, 0, 1, 0, 24.0);
        assert_eq!(tc.to_frame_number(), 24);
    }

    #[test]
    fn test_to_frame_number_one_minute() {
        let tc = FractionalTimecode::new(0, 1, 0, 0, 25.0);
        assert_eq!(tc.to_frame_number(), 25 * 60);
    }

    #[test]
    fn test_roundtrip_from_frame_number() {
        let fps = 24.0_f32;
        for frame_num in [0_u64, 1, 23, 24, 1440, 86400, 86400 * 24 - 1] {
            let tc = FractionalTimecode::from_frame_number(frame_num, fps);
            assert_eq!(
                tc.to_frame_number(),
                frame_num,
                "roundtrip failed for frame_num={frame_num}"
            );
        }
    }

    #[test]
    fn test_is_valid() {
        let tc = FractionalTimecode::new(0, 0, 0, 0, 25.0);
        assert!(tc.is_valid());
    }

    #[test]
    fn test_is_valid_zero_rate() {
        let tc = FractionalTimecode {
            hours: 0,
            minutes: 0,
            seconds: 0,
            frame: 0,
            rate: 0.0,
        };
        assert!(!tc.is_valid());
    }

    #[test]
    fn test_from_frame_number_zero_rate() {
        let tc = FractionalTimecode::from_frame_number(100, 0.0);
        assert_eq!(tc.frame, 0);
    }

    #[test]
    fn test_ntsc_29_97_df_format() {
        let tc = FractionalTimecode::new(0, 59, 59, 29, 29.97);
        let df = tc.to_string_df();
        assert!(df.contains(';'), "DF should use semicolon: {df}");
    }
}
