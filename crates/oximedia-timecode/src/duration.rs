//! Timecode-based duration calculations
//!
//! Provides `TcDuration`, `DurationRange`, and helper functions for computing
//! durations relative to a given frame rate.

#[allow(dead_code)]
/// A duration expressed as a frame count at a given frame rate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcDuration {
    /// Signed frame count (negative = before reference point)
    pub frames: i64,
    /// Frame rate numerator
    pub frame_rate_num: u32,
    /// Frame rate denominator
    pub frame_rate_den: u32,
}

impl TcDuration {
    /// Create a new `TcDuration`
    #[must_use]
    pub fn new(frames: i64, frame_rate_num: u32, frame_rate_den: u32) -> Self {
        Self {
            frames,
            frame_rate_num,
            frame_rate_den,
        }
    }

    /// Convert to floating-point seconds
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_seconds(&self) -> f64 {
        if self.frame_rate_num == 0 {
            return 0.0;
        }
        self.frames as f64 * self.frame_rate_den as f64 / self.frame_rate_num as f64
    }

    /// Convert to milliseconds
    #[must_use]
    pub fn to_milliseconds(&self) -> f64 {
        self.to_seconds() * 1000.0
    }

    /// Add two durations (must share the same frame rate)
    #[must_use]
    pub fn add(&self, other: &TcDuration) -> TcDuration {
        TcDuration {
            frames: self.frames + other.frames,
            frame_rate_num: self.frame_rate_num,
            frame_rate_den: self.frame_rate_den,
        }
    }

    /// Subtract `other` from `self`. Returns `None` if the result would be negative.
    #[must_use]
    pub fn subtract(&self, other: &TcDuration) -> Option<TcDuration> {
        let result = self.frames - other.frames;
        if result < 0 {
            None
        } else {
            Some(TcDuration {
                frames: result,
                frame_rate_num: self.frame_rate_num,
                frame_rate_den: self.frame_rate_den,
            })
        }
    }

    /// Returns `true` when the frame count is negative
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.frames < 0
    }
}

#[allow(dead_code)]
/// A half-open frame range `[start_frames, end_frames)`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DurationRange {
    /// First frame of the range (inclusive)
    pub start_frames: i64,
    /// Last frame of the range (exclusive)
    pub end_frames: i64,
}

impl DurationRange {
    /// Create a new `DurationRange`
    #[must_use]
    pub fn new(start_frames: i64, end_frames: i64) -> Self {
        Self {
            start_frames,
            end_frames,
        }
    }

    /// Return the total number of frames covered by the range
    #[must_use]
    pub fn duration_frames(&self) -> i64 {
        self.end_frames - self.start_frames
    }

    /// Returns `true` when `frame` falls within `[start_frames, end_frames)`
    #[must_use]
    pub fn contains(&self, frame: i64) -> bool {
        frame >= self.start_frames && frame < self.end_frames
    }

    /// Returns `true` when this range overlaps with `other`
    #[must_use]
    pub fn overlaps(&self, other: &DurationRange) -> bool {
        self.start_frames < other.end_frames && other.start_frames < self.end_frames
    }
}

/// Compute the signed difference `tc1_frames - tc2_frames` as a `TcDuration`
#[must_use]
pub fn timecode_subtract(
    tc1_frames: i64,
    tc2_frames: i64,
    fps_num: u32,
    fps_den: u32,
) -> TcDuration {
    TcDuration::new(tc1_frames - tc2_frames, fps_num, fps_den)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dur(frames: i64) -> TcDuration {
        TcDuration::new(frames, 25, 1)
    }

    #[test]
    fn test_to_seconds_25fps() {
        // 25 frames at 25 fps = 1 second
        assert!((dur(25).to_seconds() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_to_seconds_zero_fps() {
        let d = TcDuration::new(100, 0, 1);
        assert_eq!(d.to_seconds(), 0.0);
    }

    #[test]
    fn test_to_milliseconds() {
        // 25 frames at 25 fps = 1000 ms
        assert!((dur(25).to_milliseconds() - 1000.0).abs() < 1e-9);
    }

    #[test]
    fn test_add() {
        let result = dur(10).add(&dur(15));
        assert_eq!(result.frames, 25);
    }

    #[test]
    fn test_subtract_positive() {
        let result = dur(30).subtract(&dur(10));
        assert!(result.is_some());
        assert_eq!(result.expect("result should be ok").frames, 20);
    }

    #[test]
    fn test_subtract_to_zero() {
        let result = dur(10).subtract(&dur(10));
        assert!(result.is_some());
        assert_eq!(result.expect("result should be ok").frames, 0);
    }

    #[test]
    fn test_subtract_negative_returns_none() {
        let result = dur(5).subtract(&dur(10));
        assert!(result.is_none());
    }

    #[test]
    fn test_is_negative_false() {
        assert!(!dur(10).is_negative());
    }

    #[test]
    fn test_is_negative_true() {
        assert!(TcDuration::new(-1, 25, 1).is_negative());
    }

    #[test]
    fn test_duration_range_duration_frames() {
        let r = DurationRange::new(100, 200);
        assert_eq!(r.duration_frames(), 100);
    }

    #[test]
    fn test_duration_range_contains_true() {
        let r = DurationRange::new(100, 200);
        assert!(r.contains(100));
        assert!(r.contains(150));
        assert!(r.contains(199));
    }

    #[test]
    fn test_duration_range_contains_false() {
        let r = DurationRange::new(100, 200);
        assert!(!r.contains(99));
        assert!(!r.contains(200));
    }

    #[test]
    fn test_duration_range_overlaps_true() {
        let a = DurationRange::new(0, 100);
        let b = DurationRange::new(50, 150);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_duration_range_overlaps_false() {
        let a = DurationRange::new(0, 50);
        let b = DurationRange::new(50, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_timecode_subtract_positive() {
        let d = timecode_subtract(100, 40, 25, 1);
        assert_eq!(d.frames, 60);
        assert!(!d.is_negative());
    }

    #[test]
    fn test_timecode_subtract_negative() {
        let d = timecode_subtract(40, 100, 25, 1);
        assert_eq!(d.frames, -60);
        assert!(d.is_negative());
    }
}
