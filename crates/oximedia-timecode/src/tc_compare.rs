#![allow(dead_code)]
//! Timecode comparison and distance utilities.
//!
//! Provides utilities for comparing timecodes, computing distances,
//! sorting, and checking various temporal relationships between timecodes.

use crate::{FrameRate, Timecode, TimecodeError};
use std::cmp::Ordering;

/// The result of comparing two timecodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcRelation {
    /// The first timecode is earlier.
    Before,
    /// The timecodes are identical.
    Equal,
    /// The first timecode is later.
    After,
}

/// Compares two timecodes by total frame count.
pub fn compare(a: &Timecode, b: &Timecode) -> TcRelation {
    let fa = a.to_frames();
    let fb = b.to_frames();
    match fa.cmp(&fb) {
        Ordering::Less => TcRelation::Before,
        Ordering::Equal => TcRelation::Equal,
        Ordering::Greater => TcRelation::After,
    }
}

/// Returns the absolute distance in frames between two timecodes.
pub fn distance_frames(a: &Timecode, b: &Timecode) -> u64 {
    let fa = a.to_frames();
    let fb = b.to_frames();
    fa.abs_diff(fb)
}

/// Returns the distance between two timecodes in seconds (approximate for non-integer rates).
#[allow(clippy::cast_precision_loss)]
pub fn distance_seconds(a: &Timecode, b: &Timecode, frame_rate: FrameRate) -> f64 {
    let d = distance_frames(a, b);
    d as f64 / frame_rate.as_float()
}

/// Checks whether timecode `tc` falls within the range [start, end] (inclusive).
pub fn is_within_range(tc: &Timecode, start: &Timecode, end: &Timecode) -> bool {
    let f = tc.to_frames();
    let fs = start.to_frames();
    let fe = end.to_frames();
    f >= fs && f <= fe
}

/// Returns the midpoint timecode between two timecodes.
pub fn midpoint(
    a: &Timecode,
    b: &Timecode,
    frame_rate: FrameRate,
) -> Result<Timecode, TimecodeError> {
    let fa = a.to_frames();
    let fb = b.to_frames();
    let mid = (fa + fb) / 2;
    Timecode::from_frames(mid, frame_rate)
}

/// A timecode span defined by an in-point and an out-point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TcSpan {
    /// The in-point timecode.
    pub tc_in: Timecode,
    /// The out-point timecode.
    pub tc_out: Timecode,
}

impl TcSpan {
    /// Creates a new span. `tc_in` must not be after `tc_out`.
    pub fn new(tc_in: Timecode, tc_out: Timecode) -> Result<Self, TimecodeError> {
        if tc_in.to_frames() > tc_out.to_frames() {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self { tc_in, tc_out })
    }

    /// Returns the duration of the span in frames.
    pub fn duration_frames(&self) -> u64 {
        self.tc_out.to_frames() - self.tc_in.to_frames()
    }

    /// Returns the duration in seconds.
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, frame_rate: FrameRate) -> f64 {
        self.duration_frames() as f64 / frame_rate.as_float()
    }

    /// Checks whether a timecode falls within this span (inclusive).
    pub fn contains(&self, tc: &Timecode) -> bool {
        is_within_range(tc, &self.tc_in, &self.tc_out)
    }

    /// Checks whether this span overlaps with another.
    pub fn overlaps(&self, other: &TcSpan) -> bool {
        let a_start = self.tc_in.to_frames();
        let a_end = self.tc_out.to_frames();
        let b_start = other.tc_in.to_frames();
        let b_end = other.tc_out.to_frames();
        a_start <= b_end && b_start <= a_end
    }

    /// Returns the intersection of two spans, or `None` if they don't overlap.
    pub fn intersection(
        &self,
        other: &TcSpan,
        frame_rate: FrameRate,
    ) -> Option<Result<TcSpan, TimecodeError>> {
        if !self.overlaps(other) {
            return None;
        }
        let start = self.tc_in.to_frames().max(other.tc_in.to_frames());
        let end = self.tc_out.to_frames().min(other.tc_out.to_frames());
        let tc_in = match Timecode::from_frames(start, frame_rate) {
            Ok(tc) => tc,
            Err(e) => return Some(Err(e)),
        };
        let tc_out = match Timecode::from_frames(end, frame_rate) {
            Ok(tc) => tc,
            Err(e) => return Some(Err(e)),
        };
        Some(TcSpan::new(tc_in, tc_out))
    }
}

/// Sorts a slice of timecodes by frame count (ascending).
pub fn sort_timecodes(tcs: &mut [Timecode]) {
    tcs.sort_by_key(Timecode::to_frames);
}

/// Returns the earliest timecode from a non-empty slice.
pub fn earliest(tcs: &[Timecode]) -> Option<&Timecode> {
    tcs.iter().min_by_key(|tc| tc.to_frames())
}

/// Returns the latest timecode from a non-empty slice.
pub fn latest(tcs: &[Timecode]) -> Option<&Timecode> {
    tcs.iter().max_by_key(|tc| tc.to_frames())
}

/// Checks if a sequence of timecodes is strictly ascending (no duplicates).
pub fn is_ascending(tcs: &[Timecode]) -> bool {
    tcs.windows(2).all(|w| w[0].to_frames() < w[1].to_frames())
}

/// Checks if a sequence of timecodes is contiguous (each consecutive pair differs by exactly 1 frame).
pub fn is_contiguous(tcs: &[Timecode]) -> bool {
    tcs.windows(2)
        .all(|w| w[1].to_frames() == w[0].to_frames() + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_compare_before() {
        assert_eq!(
            compare(&tc(0, 0, 0, 0), &tc(0, 0, 0, 1)),
            TcRelation::Before
        );
    }

    #[test]
    fn test_compare_equal() {
        assert_eq!(compare(&tc(1, 2, 3, 4), &tc(1, 2, 3, 4)), TcRelation::Equal);
    }

    #[test]
    fn test_compare_after() {
        assert_eq!(
            compare(&tc(0, 0, 1, 0), &tc(0, 0, 0, 24)),
            TcRelation::After
        );
    }

    #[test]
    fn test_distance_frames_same() {
        assert_eq!(distance_frames(&tc(0, 0, 0, 0), &tc(0, 0, 0, 0)), 0);
    }

    #[test]
    fn test_distance_frames_one_second() {
        assert_eq!(distance_frames(&tc(0, 0, 0, 0), &tc(0, 0, 1, 0)), 25);
    }

    #[test]
    fn test_distance_frames_symmetric() {
        let a = tc(0, 0, 0, 10);
        let b = tc(0, 0, 1, 5);
        assert_eq!(distance_frames(&a, &b), distance_frames(&b, &a));
    }

    #[test]
    fn test_distance_seconds() {
        let d = distance_seconds(&tc(0, 0, 0, 0), &tc(0, 0, 1, 0), FrameRate::Fps25);
        assert!((d - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_is_within_range() {
        let start = tc(0, 0, 0, 0);
        let end = tc(0, 0, 2, 0);
        assert!(is_within_range(&tc(0, 0, 1, 0), &start, &end));
        assert!(is_within_range(&start, &start, &end));
        assert!(is_within_range(&end, &start, &end));
        assert!(!is_within_range(&tc(0, 0, 3, 0), &start, &end));
    }

    #[test]
    fn test_midpoint() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 2, 0); // 50 frames
        let mid = midpoint(&a, &b, FrameRate::Fps25).expect("midpoint should succeed");
        assert_eq!(mid.to_frames(), 25); // 1 second
    }

    #[test]
    fn test_tc_span_creation() {
        let span = TcSpan::new(tc(0, 0, 0, 0), tc(0, 0, 1, 0)).expect("valid timecode span");
        assert_eq!(span.duration_frames(), 25);
    }

    #[test]
    fn test_tc_span_invalid() {
        let result = TcSpan::new(tc(0, 0, 1, 0), tc(0, 0, 0, 0));
        assert!(result.is_err());
    }

    #[test]
    fn test_tc_span_contains() {
        let span = TcSpan::new(tc(0, 0, 0, 0), tc(0, 0, 2, 0)).expect("valid timecode span");
        assert!(span.contains(&tc(0, 0, 1, 0)));
        assert!(!span.contains(&tc(0, 0, 3, 0)));
    }

    #[test]
    fn test_tc_span_overlaps() {
        let a = TcSpan::new(tc(0, 0, 0, 0), tc(0, 0, 2, 0)).expect("valid timecode span");
        let b = TcSpan::new(tc(0, 0, 1, 0), tc(0, 0, 3, 0)).expect("valid timecode span");
        assert!(a.overlaps(&b));
        let c = TcSpan::new(tc(0, 0, 5, 0), tc(0, 0, 6, 0)).expect("valid timecode span");
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_tc_span_intersection() {
        let a = TcSpan::new(tc(0, 0, 0, 0), tc(0, 0, 2, 0)).expect("valid timecode span");
        let b = TcSpan::new(tc(0, 0, 1, 0), tc(0, 0, 3, 0)).expect("valid timecode span");
        let inter = a
            .intersection(&b, FrameRate::Fps25)
            .expect("intersection should succeed")
            .expect("intersection should succeed");
        assert_eq!(inter.tc_in.to_frames(), tc(0, 0, 1, 0).to_frames());
        assert_eq!(inter.tc_out.to_frames(), tc(0, 0, 2, 0).to_frames());
    }

    #[test]
    fn test_sort_timecodes() {
        let mut tcs = vec![tc(0, 0, 2, 0), tc(0, 0, 0, 0), tc(0, 0, 1, 0)];
        sort_timecodes(&mut tcs);
        assert_eq!(tcs[0].to_frames(), tc(0, 0, 0, 0).to_frames());
        assert_eq!(tcs[1].to_frames(), tc(0, 0, 1, 0).to_frames());
        assert_eq!(tcs[2].to_frames(), tc(0, 0, 2, 0).to_frames());
    }

    #[test]
    fn test_earliest_latest() {
        let tcs = vec![tc(0, 0, 2, 0), tc(0, 0, 0, 0), tc(0, 0, 1, 0)];
        assert_eq!(
            earliest(&tcs).expect("should succeed").to_frames(),
            tc(0, 0, 0, 0).to_frames()
        );
        assert_eq!(
            latest(&tcs).expect("should succeed").to_frames(),
            tc(0, 0, 2, 0).to_frames()
        );
    }

    #[test]
    fn test_is_ascending() {
        let asc = vec![tc(0, 0, 0, 0), tc(0, 0, 0, 1), tc(0, 0, 0, 2)];
        assert!(is_ascending(&asc));
        let not_asc = vec![tc(0, 0, 0, 1), tc(0, 0, 0, 0)];
        assert!(!is_ascending(&not_asc));
    }

    #[test]
    fn test_is_contiguous() {
        let contig = vec![tc(0, 0, 0, 0), tc(0, 0, 0, 1), tc(0, 0, 0, 2)];
        assert!(is_contiguous(&contig));
        let gap = vec![tc(0, 0, 0, 0), tc(0, 0, 0, 5)];
        assert!(!is_contiguous(&gap));
    }
}
