//! Timestamp arithmetic operations with overflow protection.
//!
//! This module adds `add`, `sub`, and `scale` (multiply by [`Rational`])
//! operations for [`Timestamp`], all of which are overflow-safe via
//! checked 128-bit intermediate arithmetic.
//!
//! # Design
//!
//! - All operations return `Option<Timestamp>` — they yield `None` rather than
//!   panicking or wrapping on overflow.
//! - Mixed-timebase arithmetic (e.g. adding two timestamps with different
//!   timebases) is supported: the second operand is first *rescaled* into the
//!   first operand's timebase before the operation is performed.
//! - The returned `Timestamp` always uses the **first** operand's timebase.
//!
//! # Example
//!
//! ```
//! use oximedia_core::types::{Timestamp, Rational};
//! use oximedia_core::timestamp_arith::TimestampArith;
//!
//! let tb = Rational::new(1, 1000); // 1 ms per tick
//! let t1 = Timestamp::new(1_000, tb); // 1 second
//! let t2 = Timestamp::new(500, tb);   // 0.5 seconds
//!
//! let sum = TimestampArith::add(&t1, &t2).expect("no overflow");
//! assert_eq!(sum.pts, 1_500);
//!
//! let diff = TimestampArith::sub(&t1, &t2).expect("no overflow");
//! assert_eq!(diff.pts, 500);
//!
//! let doubled = TimestampArith::scale(&t1, Rational::new(2, 1)).expect("no overflow");
//! assert_eq!(doubled.pts, 2_000);
//! ```

#![allow(dead_code)]

use crate::types::{Rational, Timestamp};

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise from timestamp arithmetic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimestampArithError {
    /// The result overflowed the `i64` range of a PTS / DTS tick value.
    Overflow,
    /// Division by zero occurred during rescaling (zero-denominator Rational).
    ZeroDenominator,
}

impl std::fmt::Display for TimestampArithError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "timestamp arithmetic overflow"),
            Self::ZeroDenominator => write!(f, "zero denominator in timestamp rational"),
        }
    }
}

impl std::error::Error for TimestampArithError {}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rescale a single tick value from `src_base` to `dst_base` using 128-bit
/// arithmetic.  Returns `None` on overflow or zero-denominator.
fn rescale_ticks(ticks: i64, src: Rational, dst: Rational) -> Option<i64> {
    if src.den == 0 || dst.den == 0 || dst.num == 0 {
        return None;
    }
    if src == dst {
        return Some(ticks);
    }
    // new_ticks = ticks * (src.num / src.den) / (dst.num / dst.den)
    //           = ticks * src.num * dst.den / (src.den * dst.num)
    let num = i128::from(src.num).checked_mul(i128::from(dst.den))?;
    let den = i128::from(src.den).checked_mul(i128::from(dst.num))?;
    if den == 0 {
        return None;
    }
    // Round to nearest: add half-denominator before dividing.
    let half = den / 2;
    let result = i128::from(ticks)
        .checked_mul(num)?
        .checked_add(half)?
        .checked_div(den)?;
    i64::try_from(result).ok()
}

/// Rescale a `Timestamp` to a new timebase.  Returns `None` on overflow.
fn rescale_ts(ts: &Timestamp, dst: Rational) -> Option<Timestamp> {
    let pts = rescale_ticks(ts.pts, ts.timebase, dst)?;
    let dts = match ts.dts {
        Some(d) => Some(rescale_ticks(d, ts.timebase, dst)?),
        None => None,
    };
    let duration = match ts.duration {
        Some(d) => Some(rescale_ticks(d, ts.timebase, dst)?),
        None => None,
    };
    Some(Timestamp::with_dts(pts, dts, dst, duration))
}

// ─────────────────────────────────────────────────────────────────────────────
// TimestampArith
// ─────────────────────────────────────────────────────────────────────────────

/// Namespace for overflow-safe [`Timestamp`] arithmetic operations.
///
/// All methods are free functions (no state); they operate on references and
/// return `Option<Timestamp>`.
pub struct TimestampArith;

impl TimestampArith {
    // ── add ──────────────────────────────────────────────────────────────

    /// Adds two timestamps.
    ///
    /// If the timebases differ, `rhs` is rescaled into `lhs`'s timebase first.
    /// The result uses `lhs`'s timebase.
    ///
    /// Returns `None` on overflow or zero-denominator timebase.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    /// use oximedia_core::timestamp_arith::TimestampArith;
    ///
    /// let tb = Rational::new(1, 1000);
    /// let a = Timestamp::new(1_000, tb);
    /// let b = Timestamp::new(500, tb);
    /// let sum = TimestampArith::add(&a, &b).expect("ok");
    /// assert_eq!(sum.pts, 1_500);
    /// ```
    #[must_use]
    pub fn add(lhs: &Timestamp, rhs: &Timestamp) -> Option<Timestamp> {
        let rhs_rebased = rescale_ts(rhs, lhs.timebase)?;
        let pts = lhs.pts.checked_add(rhs_rebased.pts)?;
        let dts = match (lhs.dts, rhs_rebased.dts) {
            (Some(a), Some(b)) => Some(a.checked_add(b)?),
            (Some(a), None) => Some(a.checked_add(rhs_rebased.pts)?),
            (None, Some(b)) => Some(lhs.pts.checked_add(b)?),
            (None, None) => None,
        };
        let duration = match (lhs.duration, rhs_rebased.duration) {
            (Some(a), Some(b)) => Some(a.checked_add(b)?),
            (other, None) | (None, other) => other,
        };
        Some(Timestamp::with_dts(pts, dts, lhs.timebase, duration))
    }

    // ── sub ──────────────────────────────────────────────────────────────

    /// Subtracts `rhs` from `lhs` (`lhs - rhs`).
    ///
    /// If the timebases differ, `rhs` is rescaled into `lhs`'s timebase first.
    /// The result uses `lhs`'s timebase.
    ///
    /// Returns `None` on overflow or zero-denominator timebase.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    /// use oximedia_core::timestamp_arith::TimestampArith;
    ///
    /// let tb = Rational::new(1, 90_000);
    /// let a = Timestamp::new(180_000, tb); // 2 s
    /// let b = Timestamp::new( 90_000, tb); // 1 s
    /// let diff = TimestampArith::sub(&a, &b).expect("ok");
    /// assert_eq!(diff.pts, 90_000);
    /// ```
    #[must_use]
    pub fn sub(lhs: &Timestamp, rhs: &Timestamp) -> Option<Timestamp> {
        let rhs_rebased = rescale_ts(rhs, lhs.timebase)?;
        let pts = lhs.pts.checked_sub(rhs_rebased.pts)?;
        let dts = match (lhs.dts, rhs_rebased.dts) {
            (Some(a), Some(b)) => Some(a.checked_sub(b)?),
            (Some(a), None) => Some(a.checked_sub(rhs_rebased.pts)?),
            (None, Some(b)) => Some(lhs.pts.checked_sub(b)?),
            (None, None) => None,
        };
        // Duration semantics: take lhs duration unchanged on subtraction.
        Some(Timestamp::with_dts(pts, dts, lhs.timebase, lhs.duration))
    }

    // ── scale ─────────────────────────────────────────────────────────────

    /// Multiplies a timestamp's tick values by a [`Rational`] scale factor.
    ///
    /// The timebase is **not** changed; only the tick values are scaled.
    /// This is useful for, e.g., applying a 2× speed-up to a timeline:
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    /// use oximedia_core::timestamp_arith::TimestampArith;
    ///
    /// let tb = Rational::new(1, 1000);
    /// let t = Timestamp::new(3_000, tb);
    /// let half_speed = TimestampArith::scale(&t, Rational::new(1, 2)).expect("ok");
    /// assert_eq!(half_speed.pts, 1_500);
    /// ```
    ///
    /// Returns `None` on overflow or zero-denominator scale factor.
    #[must_use]
    pub fn scale(ts: &Timestamp, factor: Rational) -> Option<Timestamp> {
        if factor.den == 0 {
            return None;
        }
        let scale_tick = |v: i64| -> Option<i64> {
            let num128 = i128::from(v).checked_mul(i128::from(factor.num))?;
            let half = i128::from(factor.den) / 2;
            let result = num128
                .checked_add(half)?
                .checked_div(i128::from(factor.den))?;
            i64::try_from(result).ok()
        };
        let pts = scale_tick(ts.pts)?;
        let dts = match ts.dts {
            Some(d) => Some(scale_tick(d)?),
            None => None,
        };
        let duration = match ts.duration {
            Some(d) => Some(scale_tick(d)?),
            None => None,
        };
        Some(Timestamp::with_dts(pts, dts, ts.timebase, duration))
    }

    // ── rescale ───────────────────────────────────────────────────────────

    /// Rescales a [`Timestamp`] from its current timebase to `new_base`.
    ///
    /// This is a lossless (rounding) rescale using 128-bit arithmetic.
    ///
    /// Returns `None` on overflow or zero-denominator.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    /// use oximedia_core::timestamp_arith::TimestampArith;
    ///
    /// let ts = Timestamp::new(1_000, Rational::new(1, 1_000)); // 1 second
    /// let rescaled = TimestampArith::rescale(&ts, Rational::new(1, 90_000)).expect("ok");
    /// assert_eq!(rescaled.pts, 90_000);
    /// ```
    #[must_use]
    pub fn rescale(ts: &Timestamp, new_base: Rational) -> Option<Timestamp> {
        rescale_ts(ts, new_base)
    }

    // ── clamp ─────────────────────────────────────────────────────────────

    /// Clamps a timestamp's PTS to the range `[lo, hi]`.
    ///
    /// If `lo` or `hi` have different timebases they are first rescaled into
    /// `ts`'s timebase.  Returns `None` if rescaling fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_core::types::{Timestamp, Rational};
    /// use oximedia_core::timestamp_arith::TimestampArith;
    ///
    /// let tb = Rational::new(1, 1000);
    /// let ts  = Timestamp::new(5_000, tb);
    /// let lo  = Timestamp::new(1_000, tb);
    /// let hi  = Timestamp::new(3_000, tb);
    /// let clamped = TimestampArith::clamp(&ts, &lo, &hi).expect("ok");
    /// assert_eq!(clamped.pts, 3_000);
    /// ```
    #[must_use]
    pub fn clamp(ts: &Timestamp, lo: &Timestamp, hi: &Timestamp) -> Option<Timestamp> {
        let lo_rebased = rescale_ts(lo, ts.timebase)?;
        let hi_rebased = rescale_ts(hi, ts.timebase)?;
        let pts = ts.pts.clamp(lo_rebased.pts, hi_rebased.pts);
        let dts = ts.dts.map(|d| d.clamp(lo_rebased.pts, hi_rebased.pts));
        Some(Timestamp::with_dts(pts, dts, ts.timebase, ts.duration))
    }

    // ── compare ───────────────────────────────────────────────────────────

    /// Compares two timestamps by their wall-clock PTS seconds.
    ///
    /// Returns [`std::cmp::Ordering`] without requiring the same timebase.
    /// Falls back to comparing floating-point seconds, which introduces a very
    /// small rounding error only for extremely large tick values.
    #[must_use]
    pub fn cmp_pts(a: &Timestamp, b: &Timestamp) -> std::cmp::Ordering {
        if a.timebase == b.timebase {
            a.pts.cmp(&b.pts)
        } else {
            let a_secs = a.to_seconds();
            let b_secs = b.to_seconds();
            a_secs
                .partial_cmp(&b_secs)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Rational, Timestamp};

    fn ms_tb() -> Rational {
        Rational::new(1, 1_000)
    }

    fn tb_90k() -> Rational {
        Rational::new(1, 90_000)
    }

    // 1. add: same timebase
    #[test]
    fn test_add_same_base() {
        let tb = ms_tb();
        let a = Timestamp::new(1_000, tb);
        let b = Timestamp::new(500, tb);
        let sum = TimestampArith::add(&a, &b).expect("add ok");
        assert_eq!(sum.pts, 1_500);
        assert_eq!(sum.timebase, tb);
    }

    // 2. sub: same timebase
    #[test]
    fn test_sub_same_base() {
        let tb = ms_tb();
        let a = Timestamp::new(2_000, tb);
        let b = Timestamp::new(750, tb);
        let diff = TimestampArith::sub(&a, &b).expect("sub ok");
        assert_eq!(diff.pts, 1_250);
    }

    // 3. scale: by 2/1
    #[test]
    fn test_scale_double() {
        let tb = ms_tb();
        let t = Timestamp::new(1_000, tb);
        let doubled = TimestampArith::scale(&t, Rational::new(2, 1)).expect("scale ok");
        assert_eq!(doubled.pts, 2_000);
    }

    // 4. scale: by 1/2
    #[test]
    fn test_scale_half() {
        let tb = ms_tb();
        let t = Timestamp::new(3_000, tb);
        let halved = TimestampArith::scale(&t, Rational::new(1, 2)).expect("scale ok");
        assert_eq!(halved.pts, 1_500);
    }

    // 5. rescale: 1s in 1kHz → 90kHz
    #[test]
    fn test_rescale_1k_to_90k() {
        let ts = Timestamp::new(1_000, ms_tb());
        let r = TimestampArith::rescale(&ts, tb_90k()).expect("rescale ok");
        assert_eq!(r.pts, 90_000);
        assert_eq!(r.timebase, tb_90k());
    }

    // 6. rescale preserves DTS
    #[test]
    fn test_rescale_preserves_dts() {
        let ts = Timestamp::with_dts(1_000, Some(900), ms_tb(), Some(33));
        let r = TimestampArith::rescale(&ts, tb_90k()).expect("rescale ok");
        // 1000 ms = 90000 ticks, 900 ms = 81000 ticks, 33 ms = 2970 ticks
        assert_eq!(r.pts, 90_000);
        assert_eq!(r.dts, Some(81_000));
        assert_eq!(r.duration, Some(2_970));
    }

    // 7. add: cross timebase (ms + 90k → ms result)
    #[test]
    fn test_add_cross_base() {
        let a = Timestamp::new(1_000, ms_tb()); // 1 second
        let b = Timestamp::new(45_000, tb_90k()); // 0.5 seconds
        let sum = TimestampArith::add(&a, &b).expect("add ok");
        assert_eq!(sum.timebase, ms_tb());
        // 1000 + 500 = 1500 ms
        assert_eq!(sum.pts, 1_500);
    }

    // 8. sub: cross timebase
    #[test]
    fn test_sub_cross_base() {
        let a = Timestamp::new(2_000, ms_tb()); // 2 s
        let b = Timestamp::new(90_000, tb_90k()); // 1 s
        let diff = TimestampArith::sub(&a, &b).expect("sub ok");
        assert_eq!(diff.timebase, ms_tb());
        assert_eq!(diff.pts, 1_000);
    }

    // 9. add overflow returns None
    #[test]
    fn test_add_overflow() {
        let tb = ms_tb();
        let max_ts = Timestamp::new(i64::MAX, tb);
        let one = Timestamp::new(1, tb);
        assert!(TimestampArith::add(&max_ts, &one).is_none());
    }

    // 10. sub underflow returns None
    #[test]
    fn test_sub_underflow() {
        let tb = ms_tb();
        let min_ts = Timestamp::new(i64::MIN, tb);
        let one = Timestamp::new(1, tb);
        assert!(TimestampArith::sub(&min_ts, &one).is_none());
    }

    // 11. scale by zero-denominator returns None
    #[test]
    fn test_scale_zero_denominator() {
        let tb = ms_tb();
        let ts = Timestamp::new(1_000, tb);
        // Rational::new would panic on den=0; build it directly
        let zero_den = Rational { num: 1, den: 0 };
        assert!(TimestampArith::scale(&ts, zero_den).is_none());
    }

    // 12. clamp: value above hi is clamped to hi
    #[test]
    fn test_clamp_above_hi() {
        let tb = ms_tb();
        let ts = Timestamp::new(5_000, tb);
        let lo = Timestamp::new(0, tb);
        let hi = Timestamp::new(3_000, tb);
        let c = TimestampArith::clamp(&ts, &lo, &hi).expect("clamp ok");
        assert_eq!(c.pts, 3_000);
    }

    // 13. clamp: value below lo is clamped to lo
    #[test]
    fn test_clamp_below_lo() {
        let tb = ms_tb();
        let ts = Timestamp::new(-500, tb);
        let lo = Timestamp::new(0, tb);
        let hi = Timestamp::new(3_000, tb);
        let c = TimestampArith::clamp(&ts, &lo, &hi).expect("clamp ok");
        assert_eq!(c.pts, 0);
    }

    // 14. clamp: value within range unchanged
    #[test]
    fn test_clamp_within_range() {
        let tb = ms_tb();
        let ts = Timestamp::new(1_500, tb);
        let lo = Timestamp::new(0, tb);
        let hi = Timestamp::new(3_000, tb);
        let c = TimestampArith::clamp(&ts, &lo, &hi).expect("clamp ok");
        assert_eq!(c.pts, 1_500);
    }

    // 15. cmp_pts: same base comparison
    #[test]
    fn test_cmp_pts_same_base() {
        use std::cmp::Ordering;
        let tb = ms_tb();
        let a = Timestamp::new(1_000, tb);
        let b = Timestamp::new(2_000, tb);
        assert_eq!(TimestampArith::cmp_pts(&a, &b), Ordering::Less);
        assert_eq!(TimestampArith::cmp_pts(&b, &a), Ordering::Greater);
        assert_eq!(TimestampArith::cmp_pts(&a, &a), Ordering::Equal);
    }

    // 16. cmp_pts: cross base (1000 ms vs 90000/90k = 1s)
    #[test]
    fn test_cmp_pts_cross_base_equal() {
        use std::cmp::Ordering;
        let a = Timestamp::new(1_000, ms_tb()); // 1 second
        let b = Timestamp::new(90_000, tb_90k()); // also 1 second
        assert_eq!(TimestampArith::cmp_pts(&a, &b), Ordering::Equal);
    }

    // 17. add: DTS is summed when both present
    #[test]
    fn test_add_dts_summed() {
        let tb = ms_tb();
        let a = Timestamp::with_dts(1_000, Some(900), tb, None);
        let b = Timestamp::with_dts(500, Some(450), tb, None);
        let sum = TimestampArith::add(&a, &b).expect("add ok");
        assert_eq!(sum.pts, 1_500);
        assert_eq!(sum.dts, Some(1_350));
    }

    // 18. rescale: no-op when source == dest timebase
    #[test]
    fn test_rescale_noop_same_base() {
        let tb = ms_tb();
        let ts = Timestamp::new(42_000, tb);
        let r = TimestampArith::rescale(&ts, tb).expect("rescale ok");
        assert_eq!(r.pts, 42_000);
        assert_eq!(r.timebase, tb);
    }

    // 19. scale: by 1/1 is identity
    #[test]
    fn test_scale_identity() {
        let tb = ms_tb();
        let ts = Timestamp::with_dts(3_000, Some(2_700), tb, Some(33));
        let same = TimestampArith::scale(&ts, Rational::new(1, 1)).expect("scale ok");
        assert_eq!(same.pts, 3_000);
        assert_eq!(same.dts, Some(2_700));
        assert_eq!(same.duration, Some(33));
    }

    // 20. TimestampArithError display.
    #[test]
    fn test_error_display() {
        assert!(format!("{}", TimestampArithError::Overflow).contains("overflow"));
        assert!(format!("{}", TimestampArithError::ZeroDenominator).contains("zero"));
    }

    // 21. rescale: 90kHz → 48kHz (audio rebase)
    #[test]
    fn test_rescale_90k_to_48k() {
        // 90000 ticks @ 90kHz = 1 second → should give 48000 ticks @ 48kHz
        let ts = Timestamp::new(90_000, tb_90k());
        let audio_tb = Rational::new(1, 48_000);
        let r = TimestampArith::rescale(&ts, audio_tb).expect("rescale ok");
        assert_eq!(r.pts, 48_000);
        assert_eq!(r.timebase, audio_tb);
    }

    // 22. scale: by 3/2 (1.5×) speed-up
    #[test]
    fn test_scale_three_halves() {
        let tb = ms_tb();
        let ts = Timestamp::new(2_000, tb); // 2 seconds
        let scaled = TimestampArith::scale(&ts, Rational::new(3, 2)).expect("scale ok");
        // 2000 * 3/2 = 3000
        assert_eq!(scaled.pts, 3_000);
    }

    // 23. add: zero-value timestamp is identity
    #[test]
    fn test_add_zero_is_identity() {
        let tb = ms_tb();
        let ts = Timestamp::new(5_432, tb);
        let zero = Timestamp::new(0, tb);
        let sum = TimestampArith::add(&ts, &zero).expect("add ok");
        assert_eq!(sum.pts, 5_432);
    }

    // 24. sub: subtracting self yields zero
    #[test]
    fn test_sub_self_yields_zero() {
        let tb = ms_tb();
        let ts = Timestamp::new(10_000, tb);
        let diff = TimestampArith::sub(&ts, &ts).expect("sub ok");
        assert_eq!(diff.pts, 0);
    }

    // 25. clamp: cross-timebase (ms ts, 90k bounds)
    #[test]
    fn test_clamp_cross_base() {
        // ts = 5000 ms = 5 s; lo = 1 s = 90000 ticks@90k; hi = 3 s = 270000 ticks@90k
        let ts = Timestamp::new(5_000, ms_tb());
        let lo = Timestamp::new(90_000, tb_90k()); // 1 s
        let hi = Timestamp::new(270_000, tb_90k()); // 3 s
        let c = TimestampArith::clamp(&ts, &lo, &hi).expect("clamp ok");
        // result is in ms timebase; should be clamped to 3000 ms
        assert_eq!(c.timebase, ms_tb());
        assert_eq!(c.pts, 3_000);
    }

    // 26. scale: scaling duration field
    #[test]
    fn test_scale_duration_field() {
        let tb = ms_tb();
        let ts = Timestamp::with_dts(1_000, None, tb, Some(100)); // 100ms duration
        let half = TimestampArith::scale(&ts, Rational::new(1, 2)).expect("scale ok");
        assert_eq!(half.duration, Some(50));
    }

    // 27. add: duration is summed when both present
    #[test]
    fn test_add_duration_summed() {
        let tb = ms_tb();
        let a = Timestamp::with_dts(0, None, tb, Some(500));
        let b = Timestamp::with_dts(500, None, tb, Some(300));
        let sum = TimestampArith::add(&a, &b).expect("add ok");
        assert_eq!(sum.pts, 500);
        assert_eq!(sum.duration, Some(800));
    }

    // 28. rescale: 48kHz → 1kHz (round-trip sanity)
    #[test]
    fn test_rescale_48k_to_1k() {
        let audio_tb = Rational::new(1, 48_000);
        // 48000 ticks @ 48kHz = 1 second → 1000 ticks @ 1kHz
        let ts = Timestamp::new(48_000, audio_tb);
        let r = TimestampArith::rescale(&ts, ms_tb()).expect("rescale ok");
        assert_eq!(r.pts, 1_000);
    }
}
