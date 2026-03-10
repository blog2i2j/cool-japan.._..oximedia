//! Media timing primitives: `MediaTime`, `TimeRange`, and `MediaTimeCalc`.
//!
//! Provides lightweight time representation using integer numerator/denominator
//! pairs for frame-accurate arithmetic without floating-point rounding errors.

#![allow(dead_code)]

use std::fmt;

/// A media timestamp expressed as `ticks / time_base`.
///
/// Avoids floating-point representation to preserve frame accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MediaTime {
    /// Tick count (numerator).
    pub ticks: i64,
    /// Time base denominator (e.g. 90000 for MPEG, 48000 for audio).
    pub time_base: u64,
}

impl MediaTime {
    /// Creates a `MediaTime` at exactly zero.
    #[must_use]
    pub const fn zero(time_base: u64) -> Self {
        Self {
            ticks: 0,
            time_base,
        }
    }

    /// Creates a `MediaTime` from whole seconds.
    ///
    /// # Panics
    ///
    /// Panics if `time_base` is zero.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn from_secs(secs: f64, time_base: u64) -> Self {
        assert!(time_base > 0, "time_base must be non-zero");
        let ticks = (secs * time_base as f64).round() as i64;
        Self { ticks, time_base }
    }

    /// Converts this `MediaTime` to seconds as a 64-bit float.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn to_secs(&self) -> f64 {
        self.ticks as f64 / self.time_base as f64
    }

    /// Adds an offset in ticks (same time base) and returns a new `MediaTime`.
    #[must_use]
    pub const fn add_offset(&self, offset_ticks: i64) -> Self {
        Self {
            ticks: self.ticks + offset_ticks,
            time_base: self.time_base,
        }
    }

    /// Returns `true` if this time is strictly before `other`.
    ///
    /// Both times must share the same time base; otherwise this compares
    /// the raw tick values which may be misleading.
    #[must_use]
    pub fn is_before(&self, other: &Self) -> bool {
        if self.time_base == other.time_base {
            self.ticks < other.ticks
        } else {
            self.to_secs() < other.to_secs()
        }
    }

    /// Returns `true` if this time is strictly after `other`.
    #[must_use]
    pub fn is_after(&self, other: &Self) -> bool {
        other.is_before(self)
    }

    /// Returns the absolute difference between two `MediaTime` values in seconds.
    #[must_use]
    pub fn abs_diff_secs(&self, other: &Self) -> f64 {
        (self.to_secs() - other.to_secs()).abs()
    }

    /// Rescales this `MediaTime` to a different time base.
    ///
    /// # Panics
    ///
    /// Panics if `new_base` is zero.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[must_use]
    pub fn rescale(&self, new_base: u64) -> Self {
        assert!(new_base > 0, "time_base must be non-zero");
        let new_ticks =
            (self.ticks as f64 * new_base as f64 / self.time_base as f64).round() as i64;
        Self {
            ticks: new_ticks,
            time_base: new_base,
        }
    }
}

impl fmt::Display for MediaTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}s", self.to_secs())
    }
}

/// A half-open time range `[start, end)` in the same time base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// Inclusive start.
    pub start: MediaTime,
    /// Exclusive end.
    pub end: MediaTime,
}

impl TimeRange {
    /// Creates a new `TimeRange`.
    ///
    /// # Panics
    /// Panics if `start` is after `end`.
    #[must_use]
    pub fn new(start: MediaTime, end: MediaTime) -> Self {
        assert!(
            !end.is_before(&start),
            "TimeRange: start must not be after end"
        );
        Self { start, end }
    }

    /// Returns the duration of this range in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end.to_secs() - self.start.to_secs()
    }

    /// Returns `true` if `time` falls within `[start, end)`.
    #[must_use]
    pub fn contains(&self, time: &MediaTime) -> bool {
        !time.is_before(&self.start) && time.is_before(&self.end)
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.is_before(&other.end) && other.start.is_before(&self.end)
    }

    /// Returns the overlap range between `self` and `other`, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start_secs = self.start.to_secs().max(other.start.to_secs());
        let end_secs = self.end.to_secs().min(other.end.to_secs());
        if end_secs <= start_secs {
            return None;
        }
        let tb = self.start.time_base;
        Some(Self::new(
            MediaTime::from_secs(start_secs, tb),
            MediaTime::from_secs(end_secs, tb),
        ))
    }
}

impl fmt::Display for TimeRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

/// Helper for common PTS/DTS calculations.
///
/// Converts presentation timestamps to decode timestamps given a constant
/// B-frame delay (measured in codec time-base ticks).
#[derive(Debug, Clone, Copy)]
pub struct MediaTimeCalc {
    /// Number of B-frames in the codec's lookahead (determines PTS→DTS offset).
    pub b_frame_delay: u32,
    /// Codec time base.
    pub time_base: u64,
}

impl MediaTimeCalc {
    /// Creates a `MediaTimeCalc` with the given parameters.
    #[must_use]
    pub const fn new(b_frame_delay: u32, time_base: u64) -> Self {
        Self {
            b_frame_delay,
            time_base,
        }
    }

    /// Converts a PTS (presentation timestamp in ticks) to a DTS (decode
    /// timestamp in ticks) by subtracting the B-frame delay.
    #[must_use]
    pub fn pts_to_dts(&self, pts: MediaTime) -> MediaTime {
        let delay = i64::from(self.b_frame_delay);
        pts.add_offset(-delay)
    }

    /// Converts a DTS back to a PTS by adding the B-frame delay.
    #[must_use]
    pub fn dts_to_pts(&self, dts: MediaTime) -> MediaTime {
        let delay = i64::from(self.b_frame_delay);
        dts.add_offset(delay)
    }

    /// Returns the minimum DTS offset in seconds caused by the B-frame delay.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn dts_offset_secs(&self) -> f64 {
        f64::from(self.b_frame_delay) / self.time_base as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let t = MediaTime::zero(90_000);
        assert_eq!(t.ticks, 0);
        assert_eq!(t.to_secs(), 0.0);
    }

    #[test]
    fn test_from_secs_to_secs() {
        let t = MediaTime::from_secs(1.0, 90_000);
        assert!((t.to_secs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_secs_half() {
        let t = MediaTime::from_secs(0.5, 90_000);
        assert!((t.to_secs() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_add_offset() {
        let t = MediaTime::from_secs(1.0, 90_000);
        let t2 = t.add_offset(90_000);
        assert!((t2.to_secs() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_before() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(2.0, 90_000);
        assert!(t1.is_before(&t2));
        assert!(!t2.is_before(&t1));
    }

    #[test]
    fn test_is_after() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(2.0, 90_000);
        assert!(t2.is_after(&t1));
        assert!(!t1.is_after(&t2));
    }

    #[test]
    fn test_abs_diff_secs() {
        let t1 = MediaTime::from_secs(1.0, 90_000);
        let t2 = MediaTime::from_secs(3.0, 90_000);
        assert!((t1.abs_diff_secs(&t2) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_rescale() {
        let t = MediaTime::from_secs(1.0, 90_000);
        let t2 = t.rescale(48_000);
        assert!((t2.to_secs() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_display() {
        let t = MediaTime::from_secs(1.5, 90_000);
        let s = format!("{t}");
        assert!(s.contains("1.5"));
    }

    #[test]
    fn test_time_range_duration() {
        let s = MediaTime::from_secs(0.0, 90_000);
        let e = MediaTime::from_secs(2.0, 90_000);
        let r = TimeRange::new(s, e);
        assert!((r.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_range_contains() {
        let s = MediaTime::from_secs(1.0, 90_000);
        let e = MediaTime::from_secs(5.0, 90_000);
        let r = TimeRange::new(s, e);
        let mid = MediaTime::from_secs(3.0, 90_000);
        assert!(r.contains(&mid));
        assert!(!r.contains(&MediaTime::from_secs(0.5, 90_000)));
        // End is exclusive
        assert!(!r.contains(&e));
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(4.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(2.0, 90_000),
            MediaTime::from_secs(6.0, 90_000),
        );
        assert!(r1.overlaps(&r2));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(2.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(3.0, 90_000),
            MediaTime::from_secs(5.0, 90_000),
        );
        assert!(!r1.overlaps(&r2));
    }

    #[test]
    fn test_time_range_intersection() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(4.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(2.0, 90_000),
            MediaTime::from_secs(6.0, 90_000),
        );
        let inter = r1.intersection(&r2).expect("intersection should exist");
        assert!((inter.start.to_secs() - 2.0).abs() < 1e-4);
        assert!((inter.end.to_secs() - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_time_range_intersection_none() {
        let r1 = TimeRange::new(
            MediaTime::from_secs(0.0, 90_000),
            MediaTime::from_secs(2.0, 90_000),
        );
        let r2 = TimeRange::new(
            MediaTime::from_secs(3.0, 90_000),
            MediaTime::from_secs(5.0, 90_000),
        );
        assert!(r1.intersection(&r2).is_none());
    }

    #[test]
    fn test_pts_to_dts_no_delay() {
        let calc = MediaTimeCalc::new(0, 90_000);
        let pts = MediaTime::from_secs(1.0, 90_000);
        let dts = calc.pts_to_dts(pts);
        assert_eq!(dts, pts);
    }

    #[test]
    fn test_pts_to_dts_with_delay() {
        let calc = MediaTimeCalc::new(2, 90_000);
        let pts = MediaTime {
            ticks: 100,
            time_base: 90_000,
        };
        let dts = calc.pts_to_dts(pts);
        assert_eq!(dts.ticks, 98);
    }

    #[test]
    fn test_dts_to_pts_roundtrip() {
        let calc = MediaTimeCalc::new(3, 90_000);
        let pts = MediaTime::from_secs(5.0, 90_000);
        let dts = calc.pts_to_dts(pts);
        let pts2 = calc.dts_to_pts(dts);
        assert_eq!(pts, pts2);
    }

    #[test]
    fn test_dts_offset_secs() {
        let calc = MediaTimeCalc::new(2, 90_000);
        let offset = calc.dts_offset_secs();
        assert!((offset - 2.0 / 90_000.0).abs() < 1e-12);
    }
}
