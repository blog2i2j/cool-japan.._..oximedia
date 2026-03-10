//! SMPTE timecode range definitions and boundary checking.
//!
//! Provides [`SmpteRange`] to represent a closed interval of timecodes and
//! utilities for checking whether a timecode falls inside a particular
//! SMPTE-defined boundary (e.g. a valid programme segment).

#![allow(dead_code)]

use crate::{FrameRate, FrameRateInfo, Timecode};

// -- SmpteBoundary -----------------------------------------------------------

/// A named SMPTE boundary definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmpteBoundary {
    /// Standard 24-hour day (00:00:00:00 .. 23:59:59:ff).
    FullDay,
    /// First 12 hours (00:00:00:00 .. 11:59:59:ff).
    FirstHalf,
    /// Second 12 hours (12:00:00:00 .. 23:59:59:ff).
    SecondHalf,
    /// A single hour starting at `hour`.
    SingleHour(u8),
}

impl std::fmt::Display for SmpteBoundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FullDay => write!(f, "full-day"),
            Self::FirstHalf => write!(f, "first-half"),
            Self::SecondHalf => write!(f, "second-half"),
            Self::SingleHour(h) => write!(f, "hour-{h:02}"),
        }
    }
}

// -- SmpteRange --------------------------------------------------------------

/// A closed timecode range `[start, end]` expressed in total frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmpteRange {
    /// Inclusive start position in frames since midnight.
    pub start_frames: u64,
    /// Inclusive end position in frames since midnight.
    pub end_frames: u64,
    /// Frame-rate context for display / validation.
    pub fps: u8,
}

impl SmpteRange {
    /// Create a range from two `Timecode` values.
    pub fn from_timecodes(start: &Timecode, end: &Timecode) -> Self {
        Self {
            start_frames: start.to_frames(),
            end_frames: end.to_frames(),
            fps: start.frame_rate.fps,
        }
    }

    /// Create a range from raw frame positions.
    pub fn from_frames(start: u64, end: u64, fps: u8) -> Self {
        Self {
            start_frames: start,
            end_frames: end,
            fps,
        }
    }

    /// Create a range covering the full 24-hour day for a given frame rate.
    pub fn full_day(frame_rate: FrameRate) -> Self {
        let fps = frame_rate.frames_per_second();
        let max_frame = (fps as u64) * 86400 - 1;
        Self {
            start_frames: 0,
            end_frames: max_frame,
            fps: fps as u8,
        }
    }

    /// Create a range covering a named SMPTE boundary.
    pub fn from_boundary(boundary: SmpteBoundary, frame_rate: FrameRate) -> Self {
        let fps = frame_rate.frames_per_second() as u64;
        match boundary {
            SmpteBoundary::FullDay => Self::from_frames(0, fps * 86400 - 1, fps as u8),
            SmpteBoundary::FirstHalf => Self::from_frames(0, fps * 43200 - 1, fps as u8),
            SmpteBoundary::SecondHalf => Self::from_frames(fps * 43200, fps * 86400 - 1, fps as u8),
            SmpteBoundary::SingleHour(h) => {
                let start = fps * 3600 * h as u64;
                let end = start + fps * 3600 - 1;
                Self::from_frames(start, end, fps as u8)
            }
        }
    }

    /// Check whether a `Timecode` falls inside this range (inclusive).
    pub fn contains(&self, tc: &Timecode) -> bool {
        let pos = tc.to_frames();
        pos >= self.start_frames && pos <= self.end_frames
    }

    /// Duration of the range in frames (inclusive).
    pub fn duration_frames(&self) -> u64 {
        if self.end_frames >= self.start_frames {
            self.end_frames - self.start_frames + 1
        } else {
            0
        }
    }

    /// Duration in seconds (approximate, based on fps).
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_frames() as f64 / self.fps as f64
    }

    /// Check whether two ranges overlap.
    pub fn overlaps(&self, other: &SmpteRange) -> bool {
        self.start_frames <= other.end_frames && other.start_frames <= self.end_frames
    }

    /// Compute the intersection of two ranges. Returns `None` if they do not overlap.
    pub fn intersection(&self, other: &SmpteRange) -> Option<SmpteRange> {
        if !self.overlaps(other) {
            return None;
        }
        Some(SmpteRange {
            start_frames: self.start_frames.max(other.start_frames),
            end_frames: self.end_frames.min(other.end_frames),
            fps: self.fps,
        })
    }

    /// Return `true` if this range entirely contains `other`.
    pub fn encompasses(&self, other: &SmpteRange) -> bool {
        self.start_frames <= other.start_frames && self.end_frames >= other.end_frames
    }

    /// Split the range at a given frame position into two sub-ranges.
    /// The split point becomes the last frame of the first range and the
    /// first frame of the second range is `split_at + 1`.
    pub fn split_at(&self, split_at: u64) -> Option<(SmpteRange, SmpteRange)> {
        if split_at < self.start_frames || split_at >= self.end_frames {
            return None;
        }
        let left = SmpteRange::from_frames(self.start_frames, split_at, self.fps);
        let right = SmpteRange::from_frames(split_at + 1, self.end_frames, self.fps);
        Some((left, right))
    }
}

// -- BoundaryChecker ---------------------------------------------------------

/// Checks timecodes against a set of allowed SMPTE ranges.
#[derive(Debug, Clone)]
pub struct BoundaryChecker {
    /// Allowed ranges.
    ranges: Vec<SmpteRange>,
}

impl BoundaryChecker {
    /// Create a checker with no allowed ranges (everything is out-of-bounds).
    pub fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    /// Add an allowed range.
    pub fn add_range(&mut self, range: SmpteRange) {
        self.ranges.push(range);
    }

    /// Create a checker from a slice of ranges.
    pub fn from_ranges(ranges: &[SmpteRange]) -> Self {
        Self {
            ranges: ranges.to_vec(),
        }
    }

    /// Check whether a timecode is inside any of the allowed ranges.
    pub fn is_allowed(&self, tc: &Timecode) -> bool {
        self.ranges.iter().any(|r| r.contains(tc))
    }

    /// Return how many allowed ranges contain the given timecode.
    pub fn matching_ranges(&self, tc: &Timecode) -> usize {
        self.ranges.iter().filter(|r| r.contains(tc)).count()
    }
}

impl Default for BoundaryChecker {
    fn default() -> Self {
        Self::new()
    }
}

// -- helper ------------------------------------------------------------------

/// Build a `Timecode` from raw fields (bypasses constructor checks).
fn raw_tc(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u8) -> Timecode {
    Timecode {
        hours,
        minutes,
        seconds,
        frames,
        frame_rate: FrameRateInfo {
            fps,
            drop_frame: false,
        },
        user_bits: 0,
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
    fn test_range_from_timecodes() {
        let start = tc25(1, 0, 0, 0);
        let end = tc25(1, 0, 10, 0);
        let range = SmpteRange::from_timecodes(&start, &end);
        assert!(range.start_frames < range.end_frames);
        assert_eq!(range.fps, 25);
    }

    #[test]
    fn test_range_contains_inside() {
        let range = SmpteRange::from_frames(100, 200, 25);
        let tc = raw_tc(0, 0, 6, 0, 25); // 150 frames
        assert!(range.contains(&tc));
    }

    #[test]
    fn test_range_contains_outside() {
        let range = SmpteRange::from_frames(100, 200, 25);
        let tc = raw_tc(0, 0, 0, 5, 25); // 5 frames
        assert!(!range.contains(&tc));
    }

    #[test]
    fn test_range_contains_on_boundary() {
        let range = SmpteRange::from_frames(100, 200, 25);
        let tc = raw_tc(0, 0, 4, 0, 25); // 100 frames
        assert!(range.contains(&tc));
    }

    #[test]
    fn test_duration_frames() {
        let range = SmpteRange::from_frames(0, 99, 25);
        assert_eq!(range.duration_frames(), 100);
    }

    #[test]
    fn test_duration_seconds() {
        let range = SmpteRange::from_frames(0, 49, 25);
        let dur = range.duration_seconds();
        assert!((dur - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_full_day_range() {
        let range = SmpteRange::full_day(FrameRate::Fps25);
        assert_eq!(range.start_frames, 0);
        assert_eq!(range.duration_frames(), 25 * 86400);
    }

    #[test]
    fn test_from_boundary_first_half() {
        let range = SmpteRange::from_boundary(SmpteBoundary::FirstHalf, FrameRate::Fps25);
        assert_eq!(range.start_frames, 0);
        assert_eq!(range.duration_frames(), 25 * 43200);
    }

    #[test]
    fn test_from_boundary_single_hour() {
        let range = SmpteRange::from_boundary(SmpteBoundary::SingleHour(2), FrameRate::Fps25);
        assert_eq!(range.start_frames, 25 * 3600 * 2);
        assert_eq!(range.duration_frames(), 25 * 3600);
    }

    #[test]
    fn test_overlaps_true() {
        let a = SmpteRange::from_frames(0, 100, 25);
        let b = SmpteRange::from_frames(50, 150, 25);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_false() {
        let a = SmpteRange::from_frames(0, 50, 25);
        let b = SmpteRange::from_frames(100, 200, 25);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_intersection_some() {
        let a = SmpteRange::from_frames(0, 100, 25);
        let b = SmpteRange::from_frames(50, 150, 25);
        let inter = a.intersection(&b).expect("intersection should succeed");
        assert_eq!(inter.start_frames, 50);
        assert_eq!(inter.end_frames, 100);
    }

    #[test]
    fn test_intersection_none() {
        let a = SmpteRange::from_frames(0, 50, 25);
        let b = SmpteRange::from_frames(100, 200, 25);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_encompasses() {
        let outer = SmpteRange::from_frames(0, 1000, 25);
        let inner = SmpteRange::from_frames(100, 500, 25);
        assert!(outer.encompasses(&inner));
        assert!(!inner.encompasses(&outer));
    }

    #[test]
    fn test_split_at() {
        let range = SmpteRange::from_frames(0, 200, 25);
        let (left, right) = range.split_at(100).expect("split should succeed");
        assert_eq!(left.end_frames, 100);
        assert_eq!(right.start_frames, 101);
    }

    #[test]
    fn test_split_at_invalid() {
        let range = SmpteRange::from_frames(100, 200, 25);
        assert!(range.split_at(50).is_none());
        assert!(range.split_at(200).is_none());
    }

    #[test]
    fn test_boundary_checker_allowed() {
        let range = SmpteRange::from_frames(0, 1000, 25);
        let checker = BoundaryChecker::from_ranges(&[range]);
        let tc = tc25(0, 0, 1, 0);
        assert!(checker.is_allowed(&tc));
    }

    #[test]
    fn test_boundary_checker_not_allowed() {
        let range = SmpteRange::from_frames(0, 10, 25);
        let checker = BoundaryChecker::from_ranges(&[range]);
        let tc = tc25(1, 0, 0, 0);
        assert!(!checker.is_allowed(&tc));
    }

    #[test]
    fn test_boundary_checker_matching_ranges() {
        let r1 = SmpteRange::from_frames(0, 1000, 25);
        let r2 = SmpteRange::from_frames(500, 2000, 25);
        let checker = BoundaryChecker::from_ranges(&[r1, r2]);
        let tc = raw_tc(0, 0, 24, 0, 25); // 600 frames
        assert_eq!(checker.matching_ranges(&tc), 2);
    }

    #[test]
    fn test_boundary_display() {
        assert_eq!(SmpteBoundary::FullDay.to_string(), "full-day");
        assert_eq!(SmpteBoundary::SingleHour(5).to_string(), "hour-05");
    }

    #[test]
    fn test_empty_range_duration() {
        let range = SmpteRange::from_frames(200, 100, 25);
        assert_eq!(range.duration_frames(), 0);
    }
}
