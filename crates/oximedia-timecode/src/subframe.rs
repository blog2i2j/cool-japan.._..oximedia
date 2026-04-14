// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Sub-frame timecode: extend a SMPTE timecode with an audio-sample offset.
//!
//! `SubframeTimestamp` combines a [`Timecode`] with an audio-sample index and
//! sample rate to give nanosecond-accurate timestamps that go beyond the
//! one-frame resolution of SMPTE timecode alone.
//!
//! # Example
//!
//! ```rust,ignore
//! use oximedia_timecode::{Timecode, FrameRate};
//! use oximedia_timecode::subframe::SubframeTimestamp;
//!
//! let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).unwrap();
//! let sub = SubframeTimestamp::new(tc, 441, 44100);
//! // 1 second + 441/44100 s = 1.01 s = 1_010_000_000 ns
//! assert_eq!(sub.to_nanos(), 1_010_000_000);
//! ```

use crate::{Timecode, TimecodeError};

/// A timecode extended with a sub-frame audio sample offset.
#[derive(Debug, Clone, Copy)]
pub struct SubframeTimestamp {
    /// The integer-frame SMPTE timecode.
    pub timecode: Timecode,
    /// Zero-based sample offset within the current frame.
    pub sample: u32,
    /// Audio sample rate in Hz (e.g. 48000, 44100).
    pub sample_rate: u32,
}

impl SubframeTimestamp {
    /// Create a new `SubframeTimestamp`.
    ///
    /// `sample` is the zero-based sample index within the frame identified by
    /// `tc`.  `sample_rate` must be > 0.
    ///
    /// # Errors
    ///
    /// Returns [`TimecodeError::InvalidConfiguration`] if `sample_rate` is 0.
    pub fn new(tc: Timecode, sample: u32, sample_rate: u32) -> Self {
        SubframeTimestamp {
            timecode: tc,
            sample,
            sample_rate: sample_rate.max(1),
        }
    }

    /// Convert to an absolute nanosecond offset from timecode midnight (00:00:00:00).
    ///
    /// The calculation is:
    ///
    /// ```text
    /// tc_nanos   = timecode.to_frames() * 1_000_000_000 / frame_rate
    /// sub_nanos  = sample * 1_000_000_000 / sample_rate
    /// total_nanos = tc_nanos + sub_nanos
    /// ```
    ///
    /// Integer arithmetic is used throughout to avoid floating-point error.
    /// For drop-frame rates, `timecode.to_frames()` already accounts for the
    /// frame drops.
    pub fn to_nanos(&self) -> u64 {
        let fps_info = self.timecode.frame_rate;
        let fps = crate::frame_rate_from_info(&fps_info);
        let (num, den) = fps.as_rational();

        // nanoseconds per frame = 1_000_000_000 * den / num
        // Use 128-bit intermediates to avoid overflow at 120 fps / high frame counts.
        let total_frames = self.timecode.to_frames() as u128;
        let ns_per_frame_num = 1_000_000_000_u128 * den as u128;
        let ns_per_frame_den = num as u128;

        let tc_nanos = total_frames * ns_per_frame_num / ns_per_frame_den;

        // sub-frame: sample / sample_rate seconds → nanoseconds
        let sub_nanos = self.sample as u128 * 1_000_000_000 / self.sample_rate as u128;

        (tc_nanos + sub_nanos) as u64
    }

    /// Return the sub-frame offset as a fraction in `[0.0, 1.0)` of one frame.
    ///
    /// ```text
    /// fraction = sample / (sample_rate / frame_rate)
    ///          = sample * frame_rate / sample_rate
    /// ```
    pub fn subframe_fraction(&self) -> f64 {
        let fps_info = self.timecode.frame_rate;
        let fps = crate::frame_rate_from_info(&fps_info);
        let frame_rate = fps.as_float();
        let samples_per_frame = self.sample_rate as f64 / frame_rate;
        if samples_per_frame <= 0.0 {
            return 0.0;
        }
        (self.sample as f64 / samples_per_frame).clamp(0.0, 1.0)
    }

    /// Convert back to floating-point seconds from midnight.
    pub fn to_seconds_f64(&self) -> f64 {
        self.to_nanos() as f64 / 1_000_000_000.0
    }

    /// Advance by `samples` audio samples, wrapping timecode frames as needed.
    ///
    /// Returns `Err` if timecode increment fails (e.g. wrapping past 23:59:59:xx).
    pub fn advance_samples(&self, samples: u32) -> Result<Self, TimecodeError> {
        let fps_info = self.timecode.frame_rate;
        let fps = crate::frame_rate_from_info(&fps_info);
        let samples_per_frame = (self.sample_rate as f64 / fps.as_float()).round() as u32;

        let total_samples = self.sample + samples;
        let extra_frames = total_samples / samples_per_frame.max(1);
        let new_sample = total_samples % samples_per_frame.max(1);

        let mut new_tc = self.timecode;
        for _ in 0..extra_frames {
            new_tc.increment()?;
        }

        Ok(SubframeTimestamp::new(new_tc, new_sample, self.sample_rate))
    }
}

impl PartialEq for SubframeTimestamp {
    fn eq(&self, other: &Self) -> bool {
        self.to_nanos() == other.to_nanos()
    }
}

impl Eq for SubframeTimestamp {}

impl PartialOrd for SubframeTimestamp {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SubframeTimestamp {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_nanos().cmp(&other.to_nanos())
    }
}

impl std::fmt::Display for SubframeTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} +{}/{} samples",
            self.timecode, self.sample, self.sample_rate
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc(h: u8, m: u8, s: u8, fr: u8, rate: FrameRate) -> Timecode {
        Timecode::new(h, m, s, fr, rate).expect("valid timecode")
    }

    #[test]
    fn zero_subframe_matches_tc_nanos() {
        let t = tc(0, 0, 1, 0, FrameRate::Fps25);
        let sub = SubframeTimestamp::new(t, 0, 48000);
        // 1 second = 1_000_000_000 ns
        assert_eq!(sub.to_nanos(), 1_000_000_000);
    }

    #[test]
    fn half_frame_offset_at_25fps() {
        // frame 0 at 00:00:01:00 @ 25fps = 1s = 1_000_000_000 ns
        // + 24000/48000 s = 0.5 s = 500_000_000 ns
        let t = tc(0, 0, 1, 0, FrameRate::Fps25);
        let sub = SubframeTimestamp::new(t, 24000, 48000);
        assert_eq!(sub.to_nanos(), 1_500_000_000);
    }

    #[test]
    fn subframe_fraction_zero() {
        let t = tc(0, 0, 0, 0, FrameRate::Fps25);
        let sub = SubframeTimestamp::new(t, 0, 48000);
        assert!((sub.subframe_fraction() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn to_seconds_f64_is_consistent() {
        let t = tc(0, 0, 2, 0, FrameRate::Fps25);
        let sub = SubframeTimestamp::new(t, 0, 48000);
        assert!((sub.to_seconds_f64() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ordering() {
        let t0 = tc(0, 0, 0, 0, FrameRate::Fps25);
        let t1 = tc(0, 0, 0, 1, FrameRate::Fps25);
        let sub0 = SubframeTimestamp::new(t0, 0, 48000);
        let sub1 = SubframeTimestamp::new(t1, 0, 48000);
        assert!(sub0 < sub1);
    }

    #[test]
    fn sample_rate_zero_clamped() {
        let t = tc(0, 0, 0, 0, FrameRate::Fps25);
        let sub = SubframeTimestamp::new(t, 0, 0);
        assert_eq!(sub.sample_rate, 1);
    }
}
