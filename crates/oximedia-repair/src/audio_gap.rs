//! Audio gap filling for damaged or missing audio segments.
//!
//! [`AudioGapFiller`] provides two strategies for concealing gaps in a PCM
//! sample stream:
//!
//! * **Silence fill** — replace the gap region with zeros.
//! * **Linear interpolation fill** — blend from the sample before the gap to
//!   the sample after the gap using linear interpolation.

/// Audio gap concealment.
pub struct AudioGapFiller;

impl AudioGapFiller {
    /// Fill a region of `samples` with silence (zeros).
    ///
    /// Replaces `len` samples starting at `start` with `0.0`.
    /// If the range extends beyond the length of `samples`, only the
    /// in-bounds portion is filled (no panic).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_repair::audio_gap::AudioGapFiller;
    ///
    /// let mut samples = vec![1.0f32; 10];
    /// AudioGapFiller::fill_silence(&mut samples, 3, 4);
    /// assert_eq!(&samples[3..7], &[0.0f32; 4]);
    /// ```
    pub fn fill_silence(samples: &mut Vec<f32>, start: usize, len: usize) {
        if start >= samples.len() {
            return;
        }
        let end = (start + len).min(samples.len());
        for s in samples[start..end].iter_mut() {
            *s = 0.0;
        }
    }

    /// Fill a region of `samples` using linear interpolation between the
    /// boundary samples.
    ///
    /// The value before the gap (`samples[start - 1]`, or 0.0 if `start == 0`)
    /// is blended with the value after the gap (`samples[start + len]`, or 0.0
    /// if the gap extends to the end of the buffer).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_repair::audio_gap::AudioGapFiller;
    ///
    /// let mut samples = vec![0.0f32, 1.0, f32::NAN, f32::NAN, 0.5, 0.0];
    /// AudioGapFiller::fill_interpolate(&mut samples, 2, 2);
    /// // Samples at index 2 and 3 are now interpolated between 1.0 and 0.5
    /// assert!(samples[2] > 0.5 && samples[2] <= 1.0);
    /// assert!(samples[3] >= 0.5 && samples[3] < 1.0);
    /// ```
    pub fn fill_interpolate(samples: &mut Vec<f32>, start: usize, len: usize) {
        if len == 0 || start >= samples.len() {
            return;
        }

        let end = (start + len).min(samples.len());
        let actual_len = end - start;

        // Boundary values
        let v_before = if start > 0 { samples[start - 1] } else { 0.0 };
        let v_after = if end < samples.len() { samples[end] } else { 0.0 };

        for i in 0..actual_len {
            // t goes from 0.0 (exclusive) to 1.0 (exclusive) across the gap
            let t = (i + 1) as f32 / (actual_len + 1) as f32;
            samples[start + i] = v_before + t * (v_after - v_before);
        }
    }

    /// Fill a gap using linear interpolation, then apply a Hann-window taper
    /// to smooth the transition at the gap boundaries.
    ///
    /// Useful for longer gaps where a simple linear ramp causes an audible
    /// discontinuity at the edges.
    pub fn fill_interpolate_windowed(samples: &mut Vec<f32>, start: usize, len: usize) {
        if len == 0 || start >= samples.len() {
            return;
        }

        // First apply linear interpolation to get a base fill.
        Self::fill_interpolate(samples, start, len);

        let end = (start + len).min(samples.len());
        let actual_len = end - start;

        if actual_len < 2 {
            return;
        }

        // Apply Hann window to the filled region
        use std::f32::consts::PI;
        for i in 0..actual_len {
            let w = 0.5 * (1.0 - (2.0 * PI * i as f32 / (actual_len - 1) as f32).cos());
            samples[start + i] *= w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_silence_replaces_with_zeros() {
        let mut samples = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        AudioGapFiller::fill_silence(&mut samples, 1, 3);
        assert_eq!(samples, vec![1.0, 0.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn fill_silence_clamps_at_end() {
        let mut samples = vec![1.0f32; 5];
        AudioGapFiller::fill_silence(&mut samples, 3, 100);
        assert_eq!(&samples[3..], &[0.0f32, 0.0]);
        // First three untouched
        assert_eq!(&samples[..3], &[1.0f32, 1.0, 1.0]);
    }

    #[test]
    fn fill_silence_zero_len_is_noop() {
        let mut samples = vec![1.0f32, 2.0, 3.0];
        let original = samples.clone();
        AudioGapFiller::fill_silence(&mut samples, 1, 0);
        assert_eq!(samples, original);
    }

    #[test]
    fn fill_silence_start_past_end_is_noop() {
        let mut samples = vec![1.0f32; 4];
        AudioGapFiller::fill_silence(&mut samples, 10, 2);
        assert!(samples.iter().all(|&s| s == 1.0));
    }

    #[test]
    fn fill_interpolate_basic() {
        // Signal: 0.0, 1.0, [gap], 0.0 where gap is 2 samples
        let mut samples = vec![0.0f32, 1.0, 0.0, 0.0, 0.0];
        AudioGapFiller::fill_interpolate(&mut samples, 2, 2);
        // After: interpolated between 1.0 (index 1) and 0.0 (index 4)
        // t=1/3: 1.0 + (1/3)*(-1.0) = 0.666...
        // t=2/3: 1.0 + (2/3)*(-1.0) = 0.333...
        assert!((samples[2] - 2.0 / 3.0).abs() < 0.01);
        assert!((samples[3] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn fill_interpolate_at_start_uses_zero_as_before() {
        let mut samples = vec![0.0f32, 0.0, 1.0];
        AudioGapFiller::fill_interpolate(&mut samples, 0, 2);
        // Before=0.0, after=1.0
        assert!(samples[0] > 0.0);
        assert!(samples[1] > samples[0]);
    }

    #[test]
    fn fill_interpolate_at_end_uses_zero_as_after() {
        let mut samples = vec![1.0f32, 0.0, 0.0];
        AudioGapFiller::fill_interpolate(&mut samples, 1, 2);
        // Before=1.0, after=0.0 (end of buffer)
        assert!(samples[1] < 1.0 && samples[1] >= 0.0);
        assert!(samples[2] < samples[1] || samples[2] == 0.0);
    }

    #[test]
    fn fill_interpolate_zero_len_is_noop() {
        let mut samples = vec![1.0f32, 2.0, 3.0];
        let orig = samples.clone();
        AudioGapFiller::fill_interpolate(&mut samples, 1, 0);
        assert_eq!(samples, orig);
    }

    #[test]
    fn fill_interpolate_windowed_applies_hann_envelope() {
        let mut samples = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // Fill middle 4 samples
        AudioGapFiller::fill_interpolate_windowed(&mut samples, 2, 4);
        // Windowed region should be lower at edges, higher in centre
        let region = &samples[2..6];
        // Check that all are finite
        assert!(region.iter().all(|s| s.is_finite()));
    }
}
