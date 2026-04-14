// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Timecode comparison helpers.
//!
//! The [`Timecode`] struct already implements `PartialOrd` and `Ord` based on
//! total frame count (see `lib.rs`).  This module adds named helper methods
//! that read more naturally in production code and provides distance and
//! clamping utilities.

use crate::Timecode;

/// Extension trait adding named comparison helpers to `Timecode`.
///
/// The trait is sealed; it is only implemented for `Timecode`.
pub trait TimecodeCompare: private::Sealed {
    /// Return `true` if this timecode comes before `other` in time.
    fn is_earlier_than(&self, other: &Self) -> bool;

    /// Return `true` if this timecode comes after `other` in time.
    fn is_later_than(&self, other: &Self) -> bool;

    /// Return `true` if this timecode is identical to `other` (same frame count).
    fn is_same_frame_as(&self, other: &Self) -> bool;

    /// Return the absolute distance between two timecodes in frames.
    fn frames_distance(&self, other: &Self) -> u64;

    /// Clamp `self` to the range `[lo, hi]` (inclusive, by frame count).
    ///
    /// If `lo > hi` the result is `lo`.
    fn clamp_to_range<'a>(&'a self, lo: &'a Self, hi: &'a Self) -> &'a Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::Timecode {}
}

impl TimecodeCompare for Timecode {
    /// Return `true` if `self` is strictly earlier than `other`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use oximedia_timecode::{Timecode, FrameRate};
    /// use oximedia_timecode::compare::TimecodeCompare;
    ///
    /// let tc1 = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).unwrap();
    /// let tc2 = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).unwrap();
    /// assert!(tc1.is_earlier_than(&tc2));
    /// ```
    fn is_earlier_than(&self, other: &Timecode) -> bool {
        self.to_frames() < other.to_frames()
    }

    /// Return `true` if `self` is strictly later than `other`.
    fn is_later_than(&self, other: &Timecode) -> bool {
        self.to_frames() > other.to_frames()
    }

    /// Return `true` if `self` and `other` refer to the same frame index.
    fn is_same_frame_as(&self, other: &Timecode) -> bool {
        self.to_frames() == other.to_frames()
    }

    /// Absolute difference in frames between `self` and `other`.
    fn frames_distance(&self, other: &Timecode) -> u64 {
        self.to_frames().abs_diff(other.to_frames())
    }

    /// Clamp `self` into the inclusive range `[lo, hi]`.
    fn clamp_to_range<'a>(&'a self, lo: &'a Timecode, hi: &'a Timecode) -> &'a Timecode {
        if self.is_earlier_than(lo) {
            lo
        } else if self.is_later_than(hi) {
            hi
        } else {
            self
        }
    }
}

// Standalone comparison helpers (for use without the trait).

/// Return `true` if `a` comes before `b`.
pub fn is_earlier_than(a: &Timecode, b: &Timecode) -> bool {
    a.to_frames() < b.to_frames()
}

/// Return `true` if `a` comes after `b`.
pub fn is_later_than(a: &Timecode, b: &Timecode) -> bool {
    a.to_frames() > b.to_frames()
}

/// Return `true` if `a` and `b` represent the same frame.
pub fn is_same_frame(a: &Timecode, b: &Timecode) -> bool {
    a.to_frames() == b.to_frames()
}

/// Return the frame with the earlier position.
pub fn earlier<'a>(a: &'a Timecode, b: &'a Timecode) -> &'a Timecode {
    if a.to_frames() <= b.to_frames() {
        a
    } else {
        b
    }
}

/// Return the frame with the later position.
pub fn later<'a>(a: &'a Timecode, b: &'a Timecode) -> &'a Timecode {
    if a.to_frames() >= b.to_frames() {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid tc")
    }

    #[test]
    fn is_earlier_than_true() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 1, 0);
        assert!(a.is_earlier_than(&b));
    }

    #[test]
    fn is_earlier_than_false_when_equal() {
        let a = tc(0, 0, 1, 0);
        assert!(!a.is_earlier_than(&a));
    }

    #[test]
    fn is_later_than_true() {
        let a = tc(0, 0, 1, 0);
        let b = tc(0, 0, 0, 0);
        assert!(a.is_later_than(&b));
    }

    #[test]
    fn is_same_frame_as() {
        let a = tc(1, 2, 3, 4);
        let b = tc(1, 2, 3, 4);
        assert!(a.is_same_frame_as(&b));
    }

    #[test]
    fn frames_distance() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 1, 0); // 25 frames at 25fps
        assert_eq!(a.frames_distance(&b), 25);
    }

    #[test]
    fn clamp_to_range_below() {
        let lo = tc(0, 0, 1, 0);
        let hi = tc(0, 0, 5, 0);
        let x = tc(0, 0, 0, 0); // below lo
        let result = x.clamp_to_range(&lo, &hi);
        assert!(result.is_same_frame_as(&lo));
    }

    #[test]
    fn clamp_to_range_above() {
        let lo = tc(0, 0, 1, 0);
        let hi = tc(0, 0, 5, 0);
        let x = tc(0, 0, 9, 0); // above hi
        let result = x.clamp_to_range(&lo, &hi);
        assert!(result.is_same_frame_as(&hi));
    }

    #[test]
    fn clamp_to_range_within() {
        let lo = tc(0, 0, 1, 0);
        let hi = tc(0, 0, 5, 0);
        let x = tc(0, 0, 3, 0);
        let result = x.clamp_to_range(&lo, &hi);
        assert!(result.is_same_frame_as(&x));
    }

    #[test]
    fn standalone_is_earlier_than() {
        assert!(is_earlier_than(&tc(0, 0, 0, 0), &tc(0, 0, 0, 1)));
        assert!(!is_earlier_than(&tc(0, 0, 0, 1), &tc(0, 0, 0, 0)));
    }

    #[test]
    fn standalone_earlier_returns_min() {
        let a = tc(0, 0, 2, 0);
        let b = tc(0, 0, 1, 0);
        assert!(earlier(&a, &b).is_same_frame_as(&b));
    }

    #[test]
    fn partial_ord_uses_frame_count() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 0, 1);
        assert!(a < b);
        assert!(b > a);
    }
}
