//! Timecode mathematical operations.
//!
//! Provides operations for multiplying and dividing timecode durations,
//! computing midpoints, and performing percentage-based offset calculations.
//! All operations respect drop-frame rules and 24-hour boundaries.

#![allow(dead_code)]

use crate::{FrameRate, Timecode, TimecodeError};

// -- TcDuration --------------------------------------------------------------

/// A duration expressed in timecode frames.
///
/// Unlike [`Timecode`] this is not anchored to a time-of-day and can exceed
/// 24 hours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TcDuration {
    /// Total frames in this duration.
    pub frames: u64,
    /// Frames per second (rounded integer).
    pub fps: u8,
}

impl TcDuration {
    /// Create a duration from a frame count at a given fps.
    pub fn from_frames(frames: u64, fps: u8) -> Self {
        Self { frames, fps }
    }

    /// Create a duration from hours, minutes, seconds, and frames.
    pub fn from_hmsf(hours: u32, minutes: u32, seconds: u32, frames: u32, fps: u8) -> Self {
        let total = hours as u64 * 3600 * fps as u64
            + minutes as u64 * 60 * fps as u64
            + seconds as u64 * fps as u64
            + frames as u64;
        Self { frames: total, fps }
    }

    /// Convert to (hours, minutes, seconds, frames) tuple.
    pub fn to_hmsf(&self) -> (u32, u32, u32, u32) {
        let fps = self.fps as u64;
        let total = self.frames;
        let hours = (total / (fps * 3600)) as u32;
        let rem = total % (fps * 3600);
        let minutes = (rem / (fps * 60)) as u32;
        let rem = rem % (fps * 60);
        let seconds = (rem / fps) as u32;
        let frames = (rem % fps) as u32;
        (hours, minutes, seconds, frames)
    }

    /// Duration in seconds (floating point).
    #[allow(clippy::cast_precision_loss)]
    pub fn as_seconds(&self) -> f64 {
        self.frames as f64 / self.fps as f64
    }

    /// Multiply the duration by an integer factor.
    pub fn multiply(&self, factor: u64) -> Self {
        Self {
            frames: self.frames * factor,
            fps: self.fps,
        }
    }

    /// Divide the duration by an integer divisor.
    /// Returns `None` if divisor is zero.
    pub fn divide(&self, divisor: u64) -> Option<Self> {
        if divisor == 0 {
            return None;
        }
        Some(Self {
            frames: self.frames / divisor,
            fps: self.fps,
        })
    }

    /// Scale the duration by a floating-point factor (e.g. speed change).
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn scale(&self, factor: f64) -> Self {
        let scaled = (self.frames as f64 * factor).round() as u64;
        Self {
            frames: scaled,
            fps: self.fps,
        }
    }

    /// Compute the midpoint between zero and this duration.
    pub fn midpoint(&self) -> Self {
        Self {
            frames: self.frames / 2,
            fps: self.fps,
        }
    }

    /// Add two durations together.
    pub fn add(&self, other: &TcDuration) -> Self {
        Self {
            frames: self.frames + other.frames,
            fps: self.fps,
        }
    }

    /// Subtract another duration (saturating at zero).
    pub fn subtract(&self, other: &TcDuration) -> Self {
        Self {
            frames: self.frames.saturating_sub(other.frames),
            fps: self.fps,
        }
    }

    /// Return `true` if the duration is zero.
    pub fn is_zero(&self) -> bool {
        self.frames == 0
    }
}

impl std::fmt::Display for TcDuration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (h, m, s, fr) = self.to_hmsf();
        write!(f, "{h:02}:{m:02}:{s:02}:{fr:02}")
    }
}

// -- TcMath ------------------------------------------------------------------

/// Stateless utility for timecode mathematical operations.
///
/// # Example
/// ```
/// use oximedia_timecode::tc_math::{TcMath, TcDuration};
///
/// let dur = TcDuration::from_hmsf(0, 1, 0, 0, 25); // 1 minute
/// let mid = TcMath::midpoint_between_durations(&TcDuration::from_frames(0, 25), &dur);
/// assert_eq!(mid.frames, 750); // 30 seconds at 25fps
/// ```
pub struct TcMath;

impl TcMath {
    /// Compute the duration between two timecodes (absolute value).
    pub fn duration_between(a: &Timecode, b: &Timecode) -> TcDuration {
        let fa = a.to_frames();
        let fb = b.to_frames();
        let diff = fa.abs_diff(fb);
        TcDuration::from_frames(diff, a.frame_rate.fps)
    }

    /// Compute the midpoint timecode between two timecodes.
    pub fn midpoint(
        a: &Timecode,
        b: &Timecode,
        rate: FrameRate,
    ) -> Result<Timecode, TimecodeError> {
        let fa = a.to_frames();
        let fb = b.to_frames();
        let mid = (fa + fb) / 2;
        Timecode::from_frames(mid, rate)
    }

    /// Offset a timecode by a percentage of a duration.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn offset_by_percentage(
        tc: &Timecode,
        duration: &TcDuration,
        pct: f64,
        rate: FrameRate,
    ) -> Result<Timecode, TimecodeError> {
        let offset_frames = (duration.frames as f64 * pct / 100.0).round() as u64;
        let target = tc.to_frames() + offset_frames;
        Timecode::from_frames(target, rate)
    }

    /// Compute the midpoint between two durations (not anchored to a TOD).
    pub fn midpoint_between_durations(a: &TcDuration, b: &TcDuration) -> TcDuration {
        TcDuration::from_frames((a.frames + b.frames) / 2, a.fps)
    }

    /// Compute a percentage position of a timecode within a range.
    #[allow(clippy::cast_precision_loss)]
    pub fn position_percentage(tc: &Timecode, start: &Timecode, end: &Timecode) -> f64 {
        let pos = tc.to_frames();
        let s = start.to_frames();
        let e = end.to_frames();
        if e <= s {
            return 0.0;
        }
        ((pos - s) as f64 / (e - s) as f64) * 100.0
    }

    /// Compute the frame rate conversion factor between two rates.
    #[allow(clippy::cast_precision_loss)]
    pub fn rate_conversion_factor(from: FrameRate, to: FrameRate) -> f64 {
        to.as_float() / from.as_float()
    }

    /// Convert a frame count from one frame rate to another.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn convert_frame_count(frames: u64, from: FrameRate, to: FrameRate) -> u64 {
        let factor = Self::rate_conversion_factor(from, to);
        (frames as f64 * factor).round() as u64
    }
}

// -- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tc25(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_duration_from_hmsf() {
        let d = TcDuration::from_hmsf(1, 0, 0, 0, 25);
        assert_eq!(d.frames, 90000); // 1h * 3600s * 25fps
    }

    #[test]
    fn test_duration_to_hmsf() {
        let d = TcDuration::from_frames(90000, 25);
        let (h, m, s, f) = d.to_hmsf();
        assert_eq!((h, m, s, f), (1, 0, 0, 0));
    }

    #[test]
    fn test_duration_as_seconds() {
        let d = TcDuration::from_frames(50, 25);
        let secs = d.as_seconds();
        assert!((secs - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_duration_multiply() {
        let d = TcDuration::from_frames(100, 25);
        let result = d.multiply(3);
        assert_eq!(result.frames, 300);
    }

    #[test]
    fn test_duration_divide() {
        let d = TcDuration::from_frames(300, 25);
        let result = d.divide(3).expect("divide should succeed");
        assert_eq!(result.frames, 100);
    }

    #[test]
    fn test_duration_divide_by_zero() {
        let d = TcDuration::from_frames(100, 25);
        assert!(d.divide(0).is_none());
    }

    #[test]
    fn test_duration_scale() {
        let d = TcDuration::from_frames(100, 25);
        let result = d.scale(1.5);
        assert_eq!(result.frames, 150);
    }

    #[test]
    fn test_duration_midpoint() {
        let d = TcDuration::from_frames(200, 25);
        assert_eq!(d.midpoint().frames, 100);
    }

    #[test]
    fn test_duration_add_subtract() {
        let a = TcDuration::from_frames(100, 25);
        let b = TcDuration::from_frames(50, 25);
        assert_eq!(a.add(&b).frames, 150);
        assert_eq!(a.subtract(&b).frames, 50);
        assert_eq!(b.subtract(&a).frames, 0); // saturates
    }

    #[test]
    fn test_duration_display() {
        let d = TcDuration::from_hmsf(1, 2, 3, 4, 25);
        assert_eq!(d.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_math_duration_between() {
        let a = tc25(0, 0, 0, 0);
        let b = tc25(0, 0, 2, 0);
        let dur = TcMath::duration_between(&a, &b);
        assert_eq!(dur.frames, 50);
    }

    #[test]
    fn test_math_midpoint() {
        let a = tc25(0, 0, 0, 0);
        let b = tc25(0, 0, 4, 0);
        let mid = TcMath::midpoint(&a, &b, FrameRate::Fps25).expect("midpoint should succeed");
        assert_eq!(mid.seconds, 2);
        assert_eq!(mid.frames, 0);
    }

    #[test]
    fn test_math_offset_by_percentage() {
        let tc = tc25(0, 0, 0, 0);
        let dur = TcDuration::from_frames(100, 25);
        let result = TcMath::offset_by_percentage(&tc, &dur, 50.0, FrameRate::Fps25)
            .expect("offset by percentage should succeed");
        assert_eq!(result.to_frames(), 50);
    }

    #[test]
    fn test_math_position_percentage() {
        let start = tc25(0, 0, 0, 0);
        let end = tc25(0, 0, 4, 0); // 100 frames
        let pos = tc25(0, 0, 2, 0); // 50 frames
        let pct = TcMath::position_percentage(&pos, &start, &end);
        assert!((pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_math_rate_conversion_factor() {
        let factor = TcMath::rate_conversion_factor(FrameRate::Fps25, FrameRate::Fps50);
        assert!((factor - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_math_convert_frame_count() {
        let result = TcMath::convert_frame_count(100, FrameRate::Fps25, FrameRate::Fps50);
        assert_eq!(result, 200);
    }

    #[test]
    fn test_duration_is_zero() {
        assert!(TcDuration::from_frames(0, 25).is_zero());
        assert!(!TcDuration::from_frames(1, 25).is_zero());
    }
}
