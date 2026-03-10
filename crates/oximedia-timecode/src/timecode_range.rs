#![allow(dead_code)]
//! Timecode range operations for start/end interval management.

/// A range defined by a start frame and an end frame (inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimecodeRange {
    /// Start frame (inclusive).
    pub start: u64,
    /// End frame (inclusive).
    pub end: u64,
}

impl TimecodeRange {
    /// Create a new [`TimecodeRange`].
    ///
    /// Returns `None` if `end < start`.
    pub fn new(start: u64, end: u64) -> Option<Self> {
        if end < start {
            None
        } else {
            Some(Self { start, end })
        }
    }

    /// Number of frames contained in this range (inclusive on both ends).
    pub fn duration_frames(&self) -> u64 {
        self.end - self.start + 1
    }

    /// Whether `frame` is within `[start, end]`.
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start && frame <= self.end
    }

    /// Whether this range overlaps with `other`.
    pub fn overlaps(&self, other: &TimecodeRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Split this range at `frame`, returning two sub-ranges.
    ///
    /// The split point becomes the last frame of the first range and the
    /// first frame of the second.  Returns `None` if `frame` is outside
    /// `[start, end - 1]` (there must be at least one frame on each side).
    pub fn split_at(&self, frame: u64) -> Option<(TimecodeRange, TimecodeRange)> {
        if frame < self.start || frame >= self.end {
            return None;
        }
        let left = TimecodeRange {
            start: self.start,
            end: frame,
        };
        let right = TimecodeRange {
            start: frame + 1,
            end: self.end,
        };
        Some((left, right))
    }
}

/// A collection of [`TimecodeRange`] intervals.
#[derive(Debug, Clone, Default)]
pub struct TimecodeRangeList {
    ranges: Vec<TimecodeRange>,
}

impl TimecodeRangeList {
    /// Create an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a range to the list.
    pub fn add(&mut self, range: TimecodeRange) {
        self.ranges.push(range);
    }

    /// Total frame count across all ranges.
    pub fn total_frames(&self) -> u64 {
        self.ranges.iter().map(|r| r.duration_frames()).sum()
    }

    /// Merge consecutive / overlapping ranges that are adjacent (end + 1 == next.start).
    pub fn merge_adjacent(&self) -> TimecodeRangeList {
        let mut sorted = self.ranges.clone();
        sorted.sort_by_key(|r| r.start);

        let mut merged: Vec<TimecodeRange> = Vec::new();
        for range in sorted {
            if let Some(last) = merged.last_mut() {
                if range.start <= last.end + 1 {
                    // Extend the last range if necessary.
                    if range.end > last.end {
                        last.end = range.end;
                    }
                    continue;
                }
            }
            merged.push(range);
        }

        TimecodeRangeList { ranges: merged }
    }

    /// Iterate over the stored ranges.
    pub fn iter(&self) -> std::slice::Iter<'_, TimecodeRange> {
        self.ranges.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let r = TimecodeRange::new(10, 20).expect("valid timecode range");
        assert_eq!(r.start, 10);
        assert_eq!(r.end, 20);
    }

    #[test]
    fn test_new_invalid_returns_none() {
        assert!(TimecodeRange::new(20, 10).is_none());
    }

    #[test]
    fn test_new_same_start_end() {
        let r = TimecodeRange::new(5, 5).expect("valid timecode range");
        assert_eq!(r.duration_frames(), 1);
    }

    #[test]
    fn test_duration_frames() {
        let r = TimecodeRange::new(0, 24).expect("valid timecode range");
        assert_eq!(r.duration_frames(), 25);
    }

    #[test]
    fn test_contains_frame_inside() {
        let r = TimecodeRange::new(10, 20).expect("valid timecode range");
        assert!(r.contains_frame(10));
        assert!(r.contains_frame(15));
        assert!(r.contains_frame(20));
    }

    #[test]
    fn test_contains_frame_outside() {
        let r = TimecodeRange::new(10, 20).expect("valid timecode range");
        assert!(!r.contains_frame(9));
        assert!(!r.contains_frame(21));
    }

    #[test]
    fn test_overlaps_true() {
        let a = TimecodeRange::new(0, 10).expect("valid timecode range");
        let b = TimecodeRange::new(5, 15).expect("valid timecode range");
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_overlaps_adjacent_no_overlap() {
        let a = TimecodeRange::new(0, 9).expect("valid timecode range");
        let b = TimecodeRange::new(10, 20).expect("valid timecode range");
        // Adjacent but not overlapping (end of a == start of b - 1)
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_overlaps_touching() {
        let a = TimecodeRange::new(0, 10).expect("valid timecode range");
        let b = TimecodeRange::new(10, 20).expect("valid timecode range");
        // They share frame 10 → overlapping
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_split_at_valid() {
        let r = TimecodeRange::new(0, 9).expect("valid timecode range");
        let (left, right) = r.split_at(4).expect("split should succeed");
        assert_eq!(left.start, 0);
        assert_eq!(left.end, 4);
        assert_eq!(right.start, 5);
        assert_eq!(right.end, 9);
    }

    #[test]
    fn test_split_at_boundary_invalid() {
        let r = TimecodeRange::new(0, 9).expect("valid timecode range");
        // Cannot split at the last frame (end = 9)
        assert!(r.split_at(9).is_none());
        // Cannot split before start
        assert!(r.split_at(u64::MAX).is_none());
    }

    #[test]
    fn test_list_total_frames() {
        let mut list = TimecodeRangeList::new();
        list.add(TimecodeRange::new(0, 9).expect("valid timecode range")); // 10 frames
        list.add(TimecodeRange::new(20, 24).expect("valid timecode range")); // 5 frames
        assert_eq!(list.total_frames(), 15);
    }

    #[test]
    fn test_list_merge_adjacent() {
        let mut list = TimecodeRangeList::new();
        list.add(TimecodeRange::new(10, 20).expect("valid timecode range"));
        list.add(TimecodeRange::new(21, 30).expect("valid timecode range")); // adjacent → merge
        list.add(TimecodeRange::new(50, 60).expect("valid timecode range")); // gap → separate
        let merged = list.merge_adjacent();
        let ranges: Vec<_> = merged.iter().cloned().collect();
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].start, 10);
        assert_eq!(ranges[0].end, 30);
        assert_eq!(ranges[1].start, 50);
        assert_eq!(ranges[1].end, 60);
    }

    #[test]
    fn test_list_merge_overlapping() {
        let mut list = TimecodeRangeList::new();
        list.add(TimecodeRange::new(0, 15).expect("valid timecode range"));
        list.add(TimecodeRange::new(10, 25).expect("valid timecode range")); // overlapping → merge
        let merged = list.merge_adjacent();
        let ranges: Vec<_> = merged.iter().cloned().collect();
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].end, 25);
    }
}
