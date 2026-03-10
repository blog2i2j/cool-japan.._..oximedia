//! Timecode arithmetic: add/subtract frame counts with overflow handling.
//!
//! `TcCalculator` converts between the display timecode and an absolute
//! frame count so that arithmetic on timecodes always respects drop-frame
//! rules and 24-hour wrap-around.

#![allow(dead_code)]

use crate::{FrameRate, Timecode, TimecodeError};

// ── Operation enum ────────────────────────────────────────────────────────────

/// An arithmetic operation to apply to a timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcOperation {
    /// Add a positive number of frames.
    AddFrames(u64),
    /// Subtract frames, clamping at zero (no underflow panic).
    SubtractFrames(u64),
    /// Add whole seconds.
    AddSeconds(u32),
    /// Subtract whole seconds, clamping at zero.
    SubtractSeconds(u32),
}

impl std::fmt::Display for TcOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddFrames(n) => write!(f, "+{n} frames"),
            Self::SubtractFrames(n) => write!(f, "-{n} frames"),
            Self::AddSeconds(s) => write!(f, "+{s} seconds"),
            Self::SubtractSeconds(s) => write!(f, "-{s} seconds"),
        }
    }
}

// ── Result type ───────────────────────────────────────────────────────────────

/// The result of a timecode calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcResult {
    /// The resulting timecode.
    pub timecode: Timecode,
    /// Whether 24-hour wrap-around occurred.
    pub wrapped: bool,
    /// How many full 24-hour periods were crossed (useful for multi-day spans).
    pub days_wrapped: u32,
}

impl TcResult {
    /// Convenience: return the inner `Timecode`.
    pub fn tc(&self) -> &Timecode {
        &self.timecode
    }
}

// ── Calculator ────────────────────────────────────────────────────────────────

/// Performs timecode arithmetic with drop-frame awareness.
///
/// # Example
/// ```
/// use oximedia_timecode::{Timecode, FrameRate};
/// use oximedia_timecode::tc_calculator::{TcCalculator, TcOperation};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let calc = TcCalculator::new(FrameRate::Fps25);
/// let tc = Timecode::new(0, 0, 0, 0, FrameRate::Fps25)?;
/// let result = calc.apply(&tc, TcOperation::AddFrames(50))?;
/// assert_eq!(result.timecode.seconds, 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TcCalculator {
    frame_rate: FrameRate,
}

impl TcCalculator {
    /// Create a new calculator for the given frame rate.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self { frame_rate }
    }

    /// Total number of frames in a 24-hour day for this frame rate.
    pub fn frames_per_day(&self) -> u64 {
        let fps = self.frame_rate.frames_per_second() as u64;
        if self.frame_rate.is_drop_frame() {
            // 29.97 DF: 24*60*60*30 - 2*(24*60 - 24*6) = 2589408 frames/day (exact)
            // General formula: (fps * 3600 * 24) - 2 * (24*60 - 24*60/10)
            let total_minutes = 24u64 * 60;
            let non_tenth_minutes = total_minutes - total_minutes / 10;
            fps * 3600 * 24 - 2 * non_tenth_minutes
        } else {
            fps * 3600 * 24
        }
    }

    /// Apply an operation to a timecode.
    pub fn apply(&self, tc: &Timecode, op: TcOperation) -> Result<TcResult, TimecodeError> {
        let fpd = self.frames_per_day();
        let current = tc.to_frames();

        let (raw_target, wrapped, days_wrapped) = match op {
            TcOperation::AddFrames(n) => {
                let total = current + n;
                let days = (total / fpd) as u32;
                let pos = total % fpd;
                (pos, days > 0, days)
            }
            TcOperation::SubtractFrames(n) => {
                if n <= current {
                    (current - n, false, 0)
                } else {
                    // Wrap backwards
                    let deficit = n - current;
                    let days = deficit.div_ceil(fpd) as u32;
                    let pos = fpd - (deficit % fpd);
                    let pos = if pos == fpd { 0 } else { pos };
                    (pos, true, days)
                }
            }
            TcOperation::AddSeconds(s) => {
                let fps = self.frame_rate.frames_per_second() as u64;
                self.apply(tc, TcOperation::AddFrames(s as u64 * fps))?;
                // Re-route through AddFrames
                return self.apply(tc, TcOperation::AddFrames(s as u64 * fps));
            }
            TcOperation::SubtractSeconds(s) => {
                let fps = self.frame_rate.frames_per_second() as u64;
                return self.apply(tc, TcOperation::SubtractFrames(s as u64 * fps));
            }
        };

        let result_tc = Timecode::from_frames(raw_target, self.frame_rate)?;
        Ok(TcResult {
            timecode: result_tc,
            wrapped,
            days_wrapped,
        })
    }

    /// Compute the signed frame difference `b - a`.
    /// Positive means `b` is later; negative means `b` is earlier (wrap is ignored).
    pub fn difference(&self, a: &Timecode, b: &Timecode) -> i64 {
        b.to_frames() as i64 - a.to_frames() as i64
    }

    /// Return the later of two timecodes.
    pub fn max_tc<'a>(&self, a: &'a Timecode, b: &'a Timecode) -> &'a Timecode {
        if a.to_frames() >= b.to_frames() {
            a
        } else {
            b
        }
    }

    /// Return the earlier of two timecodes.
    pub fn min_tc<'a>(&self, a: &'a Timecode, b: &'a Timecode) -> &'a Timecode {
        if a.to_frames() <= b.to_frames() {
            a
        } else {
            b
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    fn calc() -> TcCalculator {
        TcCalculator::new(FrameRate::Fps25)
    }

    #[test]
    fn test_add_frames_basic() {
        let result = calc()
            .apply(&tc(0, 0, 0, 0), TcOperation::AddFrames(25))
            .expect("should succeed");
        assert_eq!(result.timecode.seconds, 1);
        assert_eq!(result.timecode.frames, 0);
        assert!(!result.wrapped);
    }

    #[test]
    fn test_add_frames_wraps_hour() {
        let result = calc()
            .apply(&tc(23, 59, 59, 24), TcOperation::AddFrames(1))
            .expect("should succeed");
        // Should wrap to midnight
        assert_eq!(result.timecode.hours, 0);
        assert!(result.wrapped);
        assert_eq!(result.days_wrapped, 1);
    }

    #[test]
    fn test_subtract_frames_basic() {
        let result = calc()
            .apply(&tc(0, 0, 2, 0), TcOperation::SubtractFrames(25))
            .expect("should succeed");
        assert_eq!(result.timecode.seconds, 1);
        assert!(!result.wrapped);
    }

    #[test]
    fn test_subtract_frames_wraps_backwards() {
        let result = calc()
            .apply(&tc(0, 0, 0, 0), TcOperation::SubtractFrames(25))
            .expect("should succeed");
        // Should wrap to 23:59:59:00
        assert_eq!(result.timecode.hours, 23);
        assert!(result.wrapped);
    }

    #[test]
    fn test_add_seconds() {
        let result = calc()
            .apply(&tc(0, 0, 0, 0), TcOperation::AddSeconds(3))
            .expect("should succeed");
        assert_eq!(result.timecode.seconds, 3);
    }

    #[test]
    fn test_subtract_seconds() {
        let result = calc()
            .apply(&tc(0, 0, 5, 0), TcOperation::SubtractSeconds(3))
            .expect("should succeed");
        assert_eq!(result.timecode.seconds, 2);
    }

    #[test]
    fn test_difference_positive() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 1, 0);
        assert_eq!(calc().difference(&a, &b), 25);
    }

    #[test]
    fn test_difference_negative() {
        let a = tc(0, 0, 1, 0);
        let b = tc(0, 0, 0, 0);
        assert_eq!(calc().difference(&a, &b), -25);
    }

    #[test]
    fn test_difference_zero() {
        let a = tc(1, 2, 3, 4);
        let b = tc(1, 2, 3, 4);
        assert_eq!(calc().difference(&a, &b), 0);
    }

    #[test]
    fn test_max_tc() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 1, 0);
        let m = calc().max_tc(&a, &b);
        assert_eq!(m.seconds, 1);
    }

    #[test]
    fn test_min_tc() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 1, 0);
        let m = calc().min_tc(&a, &b);
        assert_eq!(m.seconds, 0);
    }

    #[test]
    fn test_frames_per_day_25fps() {
        let c = TcCalculator::new(FrameRate::Fps25);
        assert_eq!(c.frames_per_day(), 25 * 3600 * 24);
    }

    #[test]
    fn test_frames_per_day_30fps() {
        let c = TcCalculator::new(FrameRate::Fps30);
        assert_eq!(c.frames_per_day(), 30 * 3600 * 24);
    }

    #[test]
    fn test_tc_result_tc_accessor() {
        let result = calc()
            .apply(&tc(0, 0, 1, 0), TcOperation::AddFrames(0))
            .expect("should succeed");
        assert_eq!(result.tc().seconds, 1);
    }

    #[test]
    fn test_operation_display() {
        assert_eq!(TcOperation::AddFrames(10).to_string(), "+10 frames");
        assert_eq!(TcOperation::SubtractSeconds(5).to_string(), "-5 seconds");
    }

    #[test]
    fn test_add_zero_frames() {
        let original = tc(1, 2, 3, 4);
        let result = calc()
            .apply(&original, TcOperation::AddFrames(0))
            .expect("operation should succeed");
        assert_eq!(result.timecode, original);
        assert!(!result.wrapped);
    }
}
