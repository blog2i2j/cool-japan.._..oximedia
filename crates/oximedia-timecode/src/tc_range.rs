#![allow(dead_code)]
//! Timecode range operations for defining and manipulating spans of timecode.
//!
//! Provides `TcRange` for representing a contiguous span of timecode values,
//! with support for iteration, containment checks, overlap detection,
//! splitting, and merging.

use crate::{FrameRateInfo, Timecode, TimecodeError};

/// A contiguous range of timecodes defined by an inclusive start and exclusive end.
///
/// The range is measured in absolute frame counts for a given frame rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcRange {
    /// Start timecode (inclusive)
    start_frames: u64,
    /// End timecode (exclusive)
    end_frames: u64,
    /// Frames per second (rounded)
    fps: u8,
    /// Whether drop-frame accounting is active
    drop_frame: bool,
}

/// Result of splitting a range at a given timecode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitResult {
    /// The portion before the split point
    pub before: Option<TcRange>,
    /// The portion from the split point onward
    pub after: Option<TcRange>,
}

/// Describes the overlap relationship between two ranges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapKind {
    /// No overlap at all
    None,
    /// The ranges are exactly equal
    Equal,
    /// The first range fully contains the second
    Contains,
    /// The first range is fully contained by the second
    ContainedBy,
    /// Partial overlap (neither contains the other)
    Partial,
}

impl TcRange {
    /// Creates a new timecode range from two timecodes.
    ///
    /// # Errors
    ///
    /// Returns `TimecodeError::InvalidConfiguration` if start >= end or frame rate info differs.
    pub fn new(start: &Timecode, end: &Timecode) -> Result<Self, TimecodeError> {
        if start.frame_rate != end.frame_rate {
            return Err(TimecodeError::InvalidConfiguration);
        }
        let s = start.to_frames();
        let e = end.to_frames();
        if s >= e {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self {
            start_frames: s,
            end_frames: e,
            fps: start.frame_rate.fps,
            drop_frame: start.frame_rate.drop_frame,
        })
    }

    /// Creates a range from raw frame numbers.
    ///
    /// # Errors
    ///
    /// Returns `TimecodeError::InvalidConfiguration` if start >= end.
    pub fn from_frames(
        start: u64,
        end: u64,
        fps: u8,
        drop_frame: bool,
    ) -> Result<Self, TimecodeError> {
        if start >= end {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self {
            start_frames: start,
            end_frames: end,
            fps,
            drop_frame,
        })
    }

    /// Returns the start frame number (inclusive).
    pub fn start_frames(&self) -> u64 {
        self.start_frames
    }

    /// Returns the end frame number (exclusive).
    pub fn end_frames(&self) -> u64 {
        self.end_frames
    }

    /// Returns the duration in frames.
    pub fn duration_frames(&self) -> u64 {
        self.end_frames - self.start_frames
    }

    /// Returns the duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.duration_frames() as f64 / self.fps as f64
    }

    /// Checks whether the given frame number falls within this range.
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start_frames && frame < self.end_frames
    }

    /// Checks whether the given timecode falls within this range.
    pub fn contains_timecode(&self, tc: &Timecode) -> bool {
        self.contains_frame(tc.to_frames())
    }

    /// Returns the `FrameRateInfo` associated with this range.
    pub fn frame_rate_info(&self) -> FrameRateInfo {
        FrameRateInfo {
            fps: self.fps,
            drop_frame: self.drop_frame,
        }
    }

    /// Checks whether two ranges overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_frames < other.end_frames && other.start_frames < self.end_frames
    }

    /// Classifies the overlap between two ranges.
    pub fn overlap_kind(&self, other: &Self) -> OverlapKind {
        if self == other {
            return OverlapKind::Equal;
        }
        if !self.overlaps(other) {
            return OverlapKind::None;
        }
        if self.start_frames <= other.start_frames && self.end_frames >= other.end_frames {
            return OverlapKind::Contains;
        }
        if other.start_frames <= self.start_frames && other.end_frames >= self.end_frames {
            return OverlapKind::ContainedBy;
        }
        OverlapKind::Partial
    }

    /// Returns the intersection of two ranges, if any.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        if !self.overlaps(other) {
            return None;
        }
        let s = self.start_frames.max(other.start_frames);
        let e = self.end_frames.min(other.end_frames);
        Some(Self {
            start_frames: s,
            end_frames: e,
            fps: self.fps,
            drop_frame: self.drop_frame,
        })
    }

    /// Merges two ranges into one if they overlap or are adjacent.
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.end_frames < other.start_frames || other.end_frames < self.start_frames {
            return None;
        }
        let s = self.start_frames.min(other.start_frames);
        let e = self.end_frames.max(other.end_frames);
        Some(Self {
            start_frames: s,
            end_frames: e,
            fps: self.fps,
            drop_frame: self.drop_frame,
        })
    }

    /// Splits the range at the given frame number.
    pub fn split_at_frame(&self, frame: u64) -> SplitResult {
        if frame <= self.start_frames {
            SplitResult {
                before: None,
                after: Some(self.clone()),
            }
        } else if frame >= self.end_frames {
            SplitResult {
                before: Some(self.clone()),
                after: None,
            }
        } else {
            SplitResult {
                before: Some(Self {
                    start_frames: self.start_frames,
                    end_frames: frame,
                    fps: self.fps,
                    drop_frame: self.drop_frame,
                }),
                after: Some(Self {
                    start_frames: frame,
                    end_frames: self.end_frames,
                    fps: self.fps,
                    drop_frame: self.drop_frame,
                }),
            }
        }
    }

    /// Offsets the range by a signed number of frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the result would be negative.
    pub fn offset(&self, delta: i64) -> Result<Self, TimecodeError> {
        let s = if delta >= 0 {
            self.start_frames + delta as u64
        } else {
            let abs = (-delta) as u64;
            if abs > self.start_frames {
                return Err(TimecodeError::InvalidFrames);
            }
            self.start_frames - abs
        };
        let e = if delta >= 0 {
            self.end_frames + delta as u64
        } else {
            let abs = (-delta) as u64;
            if abs > self.end_frames {
                return Err(TimecodeError::InvalidFrames);
            }
            self.end_frames - abs
        };
        Ok(Self {
            start_frames: s,
            end_frames: e,
            fps: self.fps,
            drop_frame: self.drop_frame,
        })
    }

    /// Returns a list of frame numbers in this range.
    pub fn frame_iter(&self) -> impl Iterator<Item = u64> {
        self.start_frames..self.end_frames
    }

    /// Extends the range by the given number of frames on each side.
    pub fn extend(&self, head_frames: u64, tail_frames: u64) -> Self {
        let s = self.start_frames.saturating_sub(head_frames);
        let e = self.end_frames.saturating_add(tail_frames);
        Self {
            start_frames: s,
            end_frames: e,
            fps: self.fps,
            drop_frame: self.drop_frame,
        }
    }

    /// Trims the range by the given number of frames on each side.
    ///
    /// Returns `None` if the range would become empty.
    pub fn trim(&self, head_frames: u64, tail_frames: u64) -> Option<Self> {
        let s = self.start_frames.saturating_add(head_frames);
        let e = self.end_frames.saturating_sub(tail_frames);
        if s >= e {
            return None;
        }
        Some(Self {
            start_frames: s,
            end_frames: e,
            fps: self.fps,
            drop_frame: self.drop_frame,
        })
    }
}

/// Merges a list of potentially overlapping ranges into a sorted, non-overlapping set.
pub fn merge_ranges(mut ranges: Vec<TcRange>) -> Vec<TcRange> {
    if ranges.is_empty() {
        return vec![];
    }
    ranges.sort_by_key(|r| r.start_frames);
    let mut merged: Vec<TcRange> = vec![ranges[0].clone()];
    for r in &ranges[1..] {
        let last = merged
            .last_mut()
            .expect("merged is non-empty: element 0 was pushed above");
        if let Some(u) = last.union(r) {
            *last = u;
        } else {
            merged.push(r.clone());
        }
    }
    merged
}

/// Computes the total number of frames covered by a set of ranges (after merging).
pub fn total_coverage(ranges: Vec<TcRange>) -> u64 {
    merge_ranges(ranges)
        .iter()
        .map(|r| r.duration_frames())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_range(start: u64, end: u64) -> TcRange {
        TcRange::from_frames(start, end, 25, false).expect("valid timecode range")
    }

    #[test]
    fn test_create_range() {
        let r = make_range(0, 100);
        assert_eq!(r.start_frames(), 0);
        assert_eq!(r.end_frames(), 100);
    }

    #[test]
    fn test_duration_frames() {
        let r = make_range(10, 110);
        assert_eq!(r.duration_frames(), 100);
    }

    #[test]
    fn test_duration_seconds() {
        let r = make_range(0, 25);
        let d = r.duration_seconds();
        assert!((d - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_contains_frame() {
        let r = make_range(10, 20);
        assert!(r.contains_frame(10));
        assert!(r.contains_frame(19));
        assert!(!r.contains_frame(20));
        assert!(!r.contains_frame(9));
    }

    #[test]
    fn test_overlaps() {
        let a = make_range(0, 50);
        let b = make_range(25, 75);
        let c = make_range(50, 100);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_overlap_kind_equal() {
        let a = make_range(0, 100);
        let b = make_range(0, 100);
        assert_eq!(a.overlap_kind(&b), OverlapKind::Equal);
    }

    #[test]
    fn test_overlap_kind_contains() {
        let a = make_range(0, 100);
        let b = make_range(10, 50);
        assert_eq!(a.overlap_kind(&b), OverlapKind::Contains);
    }

    #[test]
    fn test_overlap_kind_partial() {
        let a = make_range(0, 50);
        let b = make_range(25, 75);
        assert_eq!(a.overlap_kind(&b), OverlapKind::Partial);
    }

    #[test]
    fn test_intersect() {
        let a = make_range(0, 50);
        let b = make_range(25, 75);
        let inter = a.intersect(&b).expect("intersect should succeed");
        assert_eq!(inter.start_frames(), 25);
        assert_eq!(inter.end_frames(), 50);
    }

    #[test]
    fn test_intersect_none() {
        let a = make_range(0, 10);
        let b = make_range(20, 30);
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn test_union() {
        let a = make_range(0, 50);
        let b = make_range(50, 100);
        let u = a.union(&b).expect("union should succeed");
        assert_eq!(u.start_frames(), 0);
        assert_eq!(u.end_frames(), 100);
    }

    #[test]
    fn test_split_at_frame() {
        let r = make_range(0, 100);
        let split = r.split_at_frame(50);
        let before = split.before.expect("should succeed");
        let after = split.after.expect("should succeed");
        assert_eq!(before.duration_frames(), 50);
        assert_eq!(after.duration_frames(), 50);
    }

    #[test]
    fn test_offset_positive() {
        let r = make_range(10, 20);
        let shifted = r.offset(5).expect("offset should succeed");
        assert_eq!(shifted.start_frames(), 15);
        assert_eq!(shifted.end_frames(), 25);
    }

    #[test]
    fn test_offset_negative() {
        let r = make_range(10, 20);
        let shifted = r.offset(-5).expect("offset should succeed");
        assert_eq!(shifted.start_frames(), 5);
        assert_eq!(shifted.end_frames(), 15);
    }

    #[test]
    fn test_extend_and_trim() {
        let r = make_range(50, 100);
        let ext = r.extend(10, 10);
        assert_eq!(ext.start_frames(), 40);
        assert_eq!(ext.end_frames(), 110);
        let trimmed = ext.trim(10, 10).expect("trim should succeed");
        assert_eq!(trimmed.start_frames(), 50);
        assert_eq!(trimmed.end_frames(), 100);
    }

    #[test]
    fn test_merge_ranges() {
        let ranges = vec![make_range(0, 30), make_range(20, 50), make_range(60, 80)];
        let merged = merge_ranges(ranges);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].start_frames(), 0);
        assert_eq!(merged[0].end_frames(), 50);
        assert_eq!(merged[1].start_frames(), 60);
    }

    #[test]
    fn test_total_coverage() {
        let ranges = vec![make_range(0, 30), make_range(20, 50), make_range(60, 80)];
        assert_eq!(total_coverage(ranges), 70); // 50 + 20
    }

    #[test]
    fn test_invalid_range() {
        assert!(TcRange::from_frames(100, 50, 25, false).is_err());
    }

    #[test]
    fn test_frame_iter_count() {
        let r = make_range(0, 10);
        assert_eq!(r.frame_iter().count(), 10);
    }
}
