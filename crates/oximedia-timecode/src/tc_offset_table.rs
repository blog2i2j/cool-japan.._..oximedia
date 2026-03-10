#![allow(dead_code)]
//! Pre-computed offset look-up tables for fast timecode-to-frame conversion.
//!
//! Converting timecodes to absolute frame numbers (especially with drop-frame
//! adjustments) involves repeated arithmetic.  For performance-critical paths
//! such as real-time LTC decoders, this module provides a pre-built table that
//! maps every minute boundary to its absolute frame offset, enabling O(1)
//! look-ups plus a small linear interpolation.

use crate::{FrameRate, Timecode, TimecodeError};

/// Maximum number of minutes in a 24-hour day.
const MAX_MINUTES: usize = 24 * 60;

/// Pre-computed offset table for a particular frame rate.
#[derive(Debug, Clone)]
pub struct OffsetTable {
    /// Frame rate this table was built for.
    rate: FrameRate,
    /// `minute_offsets[m]` = absolute frame number at hour:min = m/60 : m%60, second=0, frame=0.
    minute_offsets: Vec<u64>,
    /// Frames per second (rounded integer).
    fps: u64,
}

impl OffsetTable {
    /// Build the offset table for the given frame rate.
    ///
    /// The table covers the full 24-hour range (1440 minute entries).
    #[allow(clippy::cast_precision_loss)]
    pub fn build(rate: FrameRate) -> Self {
        let fps = rate.frames_per_second() as u64;
        let is_df = rate.is_drop_frame();
        let mut offsets = Vec::with_capacity(MAX_MINUTES);

        let mut cumulative: u64 = 0;
        for m in 0..MAX_MINUTES {
            offsets.push(cumulative);
            // Frames in this minute
            let frames_in_minute = fps * 60;
            if is_df && m > 0 {
                // Drop-frame: skip 2 frames at the start of every minute
                // except multiples of 10.
                let next_m = m + 1;
                if next_m % 10 != 0 {
                    cumulative += frames_in_minute - 2;
                } else {
                    cumulative += frames_in_minute;
                }
            } else {
                cumulative += frames_in_minute;
            }
        }

        Self {
            rate,
            minute_offsets: offsets,
            fps,
        }
    }

    /// Convert a timecode to its absolute frame offset using the table.
    ///
    /// # Errors
    ///
    /// Returns [`TimecodeError::InvalidHours`] if the timecode is out of range.
    pub fn timecode_to_frame(&self, tc: &Timecode) -> Result<u64, TimecodeError> {
        let minute_idx = tc.hours as usize * 60 + tc.minutes as usize;
        if minute_idx >= MAX_MINUTES {
            return Err(TimecodeError::InvalidHours);
        }
        let base = self.minute_offsets[minute_idx];
        let extra = tc.seconds as u64 * self.fps + tc.frames as u64;
        Ok(base + extra)
    }

    /// Convert an absolute frame offset back to a timecode using the table.
    ///
    /// Binary-searches the minute table then linearly computes seconds/frames.
    ///
    /// # Errors
    ///
    /// Returns [`TimecodeError::InvalidFrames`] if the frame number exceeds the
    /// 24-hour range.
    pub fn frame_to_timecode(&self, frame: u64) -> Result<Timecode, TimecodeError> {
        // Binary search for the minute bucket
        let mut lo: usize = 0;
        let mut hi: usize = self.minute_offsets.len();
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if self.minute_offsets[mid] <= frame {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let minute_idx = lo;
        let remaining = frame - self.minute_offsets[minute_idx];

        let hours = (minute_idx / 60) as u8;
        let minutes = (minute_idx % 60) as u8;
        let seconds = (remaining / self.fps) as u8;
        let frames = (remaining % self.fps) as u8;

        Timecode::new(hours, minutes, seconds, frames, self.rate)
    }

    /// Get the frame rate this table was built for.
    pub fn rate(&self) -> FrameRate {
        self.rate
    }

    /// Total number of frames in a full 24-hour day.
    pub fn total_day_frames(&self) -> u64 {
        let last_idx = MAX_MINUTES - 1;
        let base = self.minute_offsets[last_idx];
        // Add frames for the last minute
        if self.rate.is_drop_frame() {
            // Last minute (23:59) — minute 1439 — 1439 % 10 != 0 → drop
            base + self.fps * 60 - 2
        } else {
            base + self.fps * 60
        }
    }

    /// Look up the offset for a given minute index directly.
    pub fn minute_offset(&self, minute: usize) -> Option<u64> {
        self.minute_offsets.get(minute).copied()
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.minute_offsets.len()
    }

    /// Returns `true` when the table is empty (should never happen after build).
    pub fn is_empty(&self) -> bool {
        self.minute_offsets.is_empty()
    }
}

/// Compute the frame-accurate distance between two timecodes using the table.
///
/// The result is signed: positive when `b` is after `a`, negative otherwise.
pub fn signed_frame_distance(
    table: &OffsetTable,
    a: &Timecode,
    b: &Timecode,
) -> Result<i64, TimecodeError> {
    let fa = table.timecode_to_frame(a)?;
    let fb = table.timecode_to_frame(b)?;
    Ok(fb as i64 - fa as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_25fps() {
        let table = OffsetTable::build(FrameRate::Fps25);
        assert_eq!(table.len(), MAX_MINUTES);
        assert_eq!(table.minute_offset(0), Some(0));
        // At minute 1: 25*60 = 1500
        assert_eq!(table.minute_offset(1), Some(1500));
    }

    #[test]
    fn test_roundtrip_ndf() {
        let table = OffsetTable::build(FrameRate::Fps25);
        let tc = Timecode::new(1, 30, 15, 12, FrameRate::Fps25).expect("valid timecode");
        let frame = table.timecode_to_frame(&tc).expect("timecode should exist");
        let tc2 = table.frame_to_timecode(frame).expect("frame to timecode should succeed");
        assert_eq!(tc.hours, tc2.hours);
        assert_eq!(tc.minutes, tc2.minutes);
        assert_eq!(tc.seconds, tc2.seconds);
        assert_eq!(tc.frames, tc2.frames);
    }

    #[test]
    fn test_roundtrip_30fps() {
        let table = OffsetTable::build(FrameRate::Fps30);
        let tc = Timecode::new(10, 45, 22, 18, FrameRate::Fps30).expect("valid timecode");
        let frame = table.timecode_to_frame(&tc).expect("timecode should exist");
        let tc2 = table.frame_to_timecode(frame).expect("frame to timecode should succeed");
        assert_eq!(tc.hours, tc2.hours);
        assert_eq!(tc.minutes, tc2.minutes);
        assert_eq!(tc.seconds, tc2.seconds);
        assert_eq!(tc.frames, tc2.frames);
    }

    #[test]
    fn test_zero_timecode() {
        let table = OffsetTable::build(FrameRate::Fps25);
        let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        assert_eq!(table.timecode_to_frame(&tc).expect("timecode should exist"), 0);
    }

    #[test]
    fn test_total_day_frames_25() {
        let table = OffsetTable::build(FrameRate::Fps25);
        // 24h * 3600s * 25fps = 2_160_000
        assert_eq!(table.total_day_frames(), 2_160_000);
    }

    #[test]
    fn test_total_day_frames_30() {
        let table = OffsetTable::build(FrameRate::Fps30);
        // 24h * 3600s * 30fps = 2_592_000
        assert_eq!(table.total_day_frames(), 2_592_000);
    }

    #[test]
    fn test_signed_distance_positive() {
        let table = OffsetTable::build(FrameRate::Fps25);
        let a = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let b = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let dist = signed_frame_distance(&table, &a, &b).expect("signed distance should succeed");
        assert_eq!(dist, 25);
    }

    #[test]
    fn test_signed_distance_negative() {
        let table = OffsetTable::build(FrameRate::Fps25);
        let a = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let b = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let dist = signed_frame_distance(&table, &a, &b).expect("signed distance should succeed");
        assert_eq!(dist, -25);
    }

    #[test]
    fn test_table_is_not_empty() {
        let table = OffsetTable::build(FrameRate::Fps24);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_minute_offset_out_of_range() {
        let table = OffsetTable::build(FrameRate::Fps25);
        assert!(table.minute_offset(MAX_MINUTES).is_none());
    }

    #[test]
    fn test_frame_to_timecode_minute_boundary() {
        let table = OffsetTable::build(FrameRate::Fps25);
        // Frame 1500 = minute 1 = 00:01:00:00
        let tc = table.frame_to_timecode(1500).expect("frame to timecode should succeed");
        assert_eq!(tc.hours, 0);
        assert_eq!(tc.minutes, 1);
        assert_eq!(tc.seconds, 0);
        assert_eq!(tc.frames, 0);
    }

    #[test]
    fn test_rate_accessor() {
        let table = OffsetTable::build(FrameRate::Fps50);
        assert_eq!(table.rate(), FrameRate::Fps50);
    }
}
