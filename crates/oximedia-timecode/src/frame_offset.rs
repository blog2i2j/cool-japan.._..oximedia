//! Frame offset calculation and conversion utilities.
//!
//! Handles absolute frame offsets, cross-rate conversions, and timestamp arithmetic.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{FrameRate, Timecode, TimecodeError};

/// Absolute frame offset from the epoch (midnight).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct FrameOffset {
    frames: u64,
}

impl FrameOffset {
    /// Create a new frame offset.
    pub fn new(frames: u64) -> Self {
        Self { frames }
    }

    /// Get the raw frame count.
    pub fn as_frames(&self) -> u64 {
        self.frames
    }

    /// Add a number of frames.
    pub fn add_frames(self, n: u64) -> Self {
        Self {
            frames: self.frames + n,
        }
    }

    /// Subtract frames, saturating at zero.
    pub fn sub_frames(self, n: u64) -> Self {
        Self {
            frames: self.frames.saturating_sub(n),
        }
    }

    /// Compute the difference in frames.
    pub fn diff(self, other: FrameOffset) -> i64 {
        self.frames as i64 - other.frames as i64
    }

    /// Convert to a timecode at the given frame rate.
    pub fn to_timecode(self, frame_rate: FrameRate) -> Result<Timecode, TimecodeError> {
        Timecode::from_frames(self.frames, frame_rate)
    }

    /// Convert to wall-clock time in seconds.
    pub fn to_seconds(self, frame_rate: FrameRate) -> f64 {
        let (num, den) = frame_rate.as_rational();
        self.frames as f64 * den as f64 / num as f64
    }

    /// Create from wall-clock time in seconds.
    pub fn from_seconds(seconds: f64, frame_rate: FrameRate) -> Self {
        let (num, den) = frame_rate.as_rational();
        let frames = (seconds * num as f64 / den as f64).round() as u64;
        Self { frames }
    }

    /// Create from a timecode.
    pub fn from_timecode(tc: &Timecode) -> Self {
        Self {
            frames: tc.to_frames(),
        }
    }
}

impl From<u64> for FrameOffset {
    fn from(n: u64) -> Self {
        Self::new(n)
    }
}

/// Cross-rate frame conversion.
///
/// Converts frame offsets between different frame rates.
#[derive(Debug, Clone)]
pub struct CrossRateConverter {
    src_rate: FrameRate,
    dst_rate: FrameRate,
}

impl CrossRateConverter {
    /// Create a new converter.
    pub fn new(src_rate: FrameRate, dst_rate: FrameRate) -> Self {
        Self { src_rate, dst_rate }
    }

    /// Convert a frame offset from source rate to destination rate.
    pub fn convert(&self, offset: FrameOffset) -> FrameOffset {
        // Use rational arithmetic to avoid floating-point drift
        let (src_num, src_den) = self.src_rate.as_rational();
        let (dst_num, dst_den) = self.dst_rate.as_rational();
        // dst_frames = src_frames * (src_den/src_num) * (dst_num/dst_den)
        //            = src_frames * src_den * dst_num / (src_num * dst_den)
        let numerator = offset.frames as u128 * src_den as u128 * dst_num as u128;
        let denominator = src_num as u128 * dst_den as u128;
        let dst_frames = (numerator + denominator / 2) / denominator;
        FrameOffset::new(dst_frames as u64)
    }

    /// Convert seconds to frame offset at destination rate.
    pub fn seconds_to_offset(&self, seconds: f64) -> FrameOffset {
        FrameOffset::from_seconds(seconds, self.dst_rate)
    }
}

/// Timecode offset table for fast lookup.
#[derive(Debug, Clone)]
pub struct OffsetTable {
    frame_rate: FrameRate,
    entries: Vec<OffsetEntry>,
}

/// A single entry in an offset table.
#[derive(Debug, Clone, Copy)]
pub struct OffsetEntry {
    /// Source timecode frame position.
    pub src_frame: u64,
    /// Destination/record timecode frame position.
    pub dst_frame: u64,
    /// Edit type for this region.
    pub edit_type: EditType,
}

/// Type of edit at a timecode offset entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EditType {
    /// Normal continuous recording.
    Continuous,
    /// Cut edit.
    Cut,
    /// Dissolve transition.
    Dissolve,
    /// Discontinuity in source.
    Discontinuity,
}

impl OffsetTable {
    /// Create a new empty offset table.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            entries: Vec::new(),
        }
    }

    /// Add an offset entry.
    pub fn add_entry(&mut self, src_frame: u64, dst_frame: u64, edit_type: EditType) {
        self.entries.push(OffsetEntry {
            src_frame,
            dst_frame,
            edit_type,
        });
        self.entries.sort_by_key(|e| e.src_frame);
    }

    /// Look up the destination frame for a source frame (nearest-lower match).
    pub fn lookup(&self, src_frame: u64) -> Option<&OffsetEntry> {
        // Binary search for the last entry where src_frame <= query
        let pos = self.entries.partition_point(|e| e.src_frame <= src_frame);
        if pos == 0 {
            None
        } else {
            Some(&self.entries[pos - 1])
        }
    }

    /// Get the number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the frame rate.
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Compute the offset (dst - src) at a given source frame.
    pub fn offset_at(&self, src_frame: u64) -> Option<i64> {
        let entry = self.lookup(src_frame)?;
        Some(entry.dst_frame as i64 - entry.src_frame as i64)
    }
}

/// Duration arithmetic between two frame offsets.
pub fn frame_duration(start: FrameOffset, end: FrameOffset, frame_rate: FrameRate) -> f64 {
    let frames = end.as_frames().saturating_sub(start.as_frames());
    let (num, den) = frame_rate.as_rational();
    frames as f64 * den as f64 / num as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_offset_arithmetic() {
        let a = FrameOffset::new(100);
        let b = a.add_frames(50);
        assert_eq!(b.as_frames(), 150);
        let c = b.sub_frames(30);
        assert_eq!(c.as_frames(), 120);
    }

    #[test]
    fn test_frame_offset_diff() {
        let a = FrameOffset::new(200);
        let b = FrameOffset::new(150);
        assert_eq!(a.diff(b), 50);
        assert_eq!(b.diff(a), -50);
    }

    #[test]
    fn test_frame_offset_to_seconds_25fps() {
        let offset = FrameOffset::new(25);
        let secs = offset.to_seconds(FrameRate::Fps25);
        assert!((secs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_offset_from_seconds_25fps() {
        let offset = FrameOffset::from_seconds(2.0, FrameRate::Fps25);
        assert_eq!(offset.as_frames(), 50);
    }

    #[test]
    fn test_frame_offset_from_timecode() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let offset = FrameOffset::from_timecode(&tc);
        assert_eq!(offset.as_frames(), 25);
    }

    #[test]
    fn test_frame_offset_to_timecode() {
        let offset = FrameOffset::new(25);
        let tc = offset
            .to_timecode(FrameRate::Fps25)
            .expect("to_timecode should succeed");
        assert_eq!(tc.seconds, 1);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_cross_rate_same_rate() {
        let conv = CrossRateConverter::new(FrameRate::Fps25, FrameRate::Fps25);
        let offset = FrameOffset::new(100);
        let converted = conv.convert(offset);
        assert_eq!(converted.as_frames(), 100);
    }

    #[test]
    fn test_cross_rate_25_to_50() {
        let conv = CrossRateConverter::new(FrameRate::Fps25, FrameRate::Fps50);
        let offset = FrameOffset::new(25);
        let converted = conv.convert(offset);
        assert_eq!(converted.as_frames(), 50);
    }

    #[test]
    fn test_cross_rate_50_to_25() {
        let conv = CrossRateConverter::new(FrameRate::Fps50, FrameRate::Fps25);
        let offset = FrameOffset::new(50);
        let converted = conv.convert(offset);
        assert_eq!(converted.as_frames(), 25);
    }

    #[test]
    fn test_offset_table_lookup() {
        let mut table = OffsetTable::new(FrameRate::Fps25);
        table.add_entry(0, 0, EditType::Continuous);
        table.add_entry(100, 200, EditType::Cut);
        table.add_entry(300, 400, EditType::Dissolve);

        assert!(table.lookup(0).is_some());
        let entry = table.lookup(150).expect("lookup should succeed");
        assert_eq!(entry.src_frame, 100);
        assert_eq!(entry.dst_frame, 200);
    }

    #[test]
    fn test_offset_table_offset_at() {
        let mut table = OffsetTable::new(FrameRate::Fps25);
        table.add_entry(0, 10, EditType::Continuous);
        assert_eq!(table.offset_at(0), Some(10));
        assert_eq!(table.offset_at(50), Some(10));
    }

    #[test]
    fn test_offset_table_empty_lookup() {
        let table = OffsetTable::new(FrameRate::Fps25);
        assert!(table.lookup(0).is_none());
        assert!(table.is_empty());
    }

    #[test]
    fn test_frame_duration() {
        let start = FrameOffset::new(0);
        let end = FrameOffset::new(25);
        let dur = frame_duration(start, end, FrameRate::Fps25);
        assert!((dur - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_frame_offset_sub_saturate() {
        let a = FrameOffset::new(5);
        let b = a.sub_frames(100);
        assert_eq!(b.as_frames(), 0);
    }
}
