//! Mid-Side (M/S) stereo processing.
//!
//! Mid-Side encoding separates a stereo signal into two components:
//!
//! - **Mid** (`M`): the sum of left and right, representing the mono-compatible
//!   centre content.
//! - **Side** (`S`): the difference between left and right, representing the
//!   stereo width content.
//!
//! By scaling the Side channel before decoding you can continuously control the
//! perceived stereo width without touching the mono sum, which is useful for
//! mastering, broadcast delivery, and creative effects.
//!
//! # Signal flow
//!
//! ```text
//!  encode:   M = (L + R) / 2
//!            S = (L - R) / 2
//!
//!  (optionally scale S by width factor)
//!
//!  decode:   L = M + S
//!            R = M - S
//! ```

use thiserror::Error;

// ─── MsError ─────────────────────────────────────────────────────────────────

/// Errors that can occur during M/S processing.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MsError {
    /// The left and right slices have different lengths.
    #[error("Length mismatch: left={0}, right={1}")]
    LengthMismatch(usize, usize),

    /// An empty input was provided where samples are required.
    #[error("Input is empty")]
    EmptyInput,

    /// The requested stereo width is outside the valid range `[0.0, 4.0]`.
    #[error("Invalid width: {0} (must be in [0.0, 4.0])")]
    InvalidWidth(f32),
}

// ─── MsEncoder ───────────────────────────────────────────────────────────────

/// Converts a stereo signal into its Mid and Side components.
pub struct MsEncoder;

impl MsEncoder {
    /// Encode a stereo pair `(left, right)` into `(mid, side)`.
    ///
    /// ```text
    /// mid[n]  = (left[n] + right[n]) / 2
    /// side[n] = (left[n] - right[n]) / 2
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] if either slice is empty, or
    /// [`MsError::LengthMismatch`] if they differ in length.
    pub fn encode(left: &[f32], right: &[f32]) -> Result<(Vec<f32>, Vec<f32>), MsError> {
        if left.is_empty() || right.is_empty() {
            return Err(MsError::EmptyInput);
        }
        if left.len() != right.len() {
            return Err(MsError::LengthMismatch(left.len(), right.len()));
        }

        let mid: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| (l + r) * 0.5)
            .collect();
        let side: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(&l, &r)| (l - r) * 0.5)
            .collect();

        Ok((mid, side))
    }
}

// ─── MsDecoder ───────────────────────────────────────────────────────────────

/// Reconstructs a stereo signal from its Mid and Side components.
pub struct MsDecoder;

impl MsDecoder {
    /// Decode `(mid, side)` back to `(left, right)`.
    ///
    /// ```text
    /// left[n]  = mid[n] + side[n]
    /// right[n] = mid[n] - side[n]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] or [`MsError::LengthMismatch`].
    pub fn decode(mid: &[f32], side: &[f32]) -> Result<(Vec<f32>, Vec<f32>), MsError> {
        if mid.is_empty() || side.is_empty() {
            return Err(MsError::EmptyInput);
        }
        if mid.len() != side.len() {
            return Err(MsError::LengthMismatch(mid.len(), side.len()));
        }

        let left: Vec<f32> = mid
            .iter()
            .zip(side.iter())
            .map(|(&m, &s)| m + s)
            .collect();
        let right: Vec<f32> = mid
            .iter()
            .zip(side.iter())
            .map(|(&m, &s)| m - s)
            .collect();

        Ok((left, right))
    }
}

// ─── MsProcessor ─────────────────────────────────────────────────────────────

/// Applies M/S width processing to an in-place stereo buffer.
///
/// | Width | Effect                                      |
/// |-------|---------------------------------------------|
/// | 0.0   | Mono — Side channel zeroed                  |
/// | 1.0   | Original stereo unchanged                   |
/// | 2.0   | Double width — Side scaled by 2             |
/// | >2.0  | Extra-wide (use with care; may clip)        |
#[derive(Debug, Clone)]
pub struct MsProcessor {
    /// Stereo width multiplier applied to the Side channel.
    /// Valid range: `[0.0, 4.0]`.
    pub width: f32,
}

impl MsProcessor {
    /// Create a new `MsProcessor`.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::InvalidWidth`] when `width` is outside `[0.0, 4.0]`.
    pub fn new(width: f32) -> Result<Self, MsError> {
        if !(0.0..=4.0).contains(&width) {
            return Err(MsError::InvalidWidth(width));
        }
        Ok(Self { width })
    }

    /// Set a new width value.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::InvalidWidth`] when the value is out of range.
    pub fn set_width(&mut self, width: f32) -> Result<(), MsError> {
        if !(0.0..=4.0).contains(&width) {
            return Err(MsError::InvalidWidth(width));
        }
        self.width = width;
        Ok(())
    }

    /// Apply M/S width processing to `left` and `right` in-place.
    ///
    /// Internally this encodes to M/S, scales the Side channel by [`Self::width`],
    /// then decodes back to L/R.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] or [`MsError::LengthMismatch`].
    pub fn process(&self, left: &mut [f32], right: &mut [f32]) -> Result<(), MsError> {
        if left.is_empty() || right.is_empty() {
            return Err(MsError::EmptyInput);
        }
        if left.len() != right.len() {
            return Err(MsError::LengthMismatch(left.len(), right.len()));
        }

        let w = self.width;
        for (l, r) in left.iter_mut().zip(right.iter_mut()) {
            let mid = (*l + *r) * 0.5;
            let side = (*l - *r) * 0.5 * w;
            *l = mid + side;
            *r = mid - side;
        }
        Ok(())
    }
}

impl Default for MsProcessor {
    fn default() -> Self {
        Self { width: 1.0 }
    }
}

// ─── MsStereoAnalyzer ─────────────────────────────────────────────────────────

/// Measures stereo field properties of a signal.
pub struct MsStereoAnalyzer;

impl MsStereoAnalyzer {
    /// Compute the **Pearson correlation coefficient** between two channels.
    ///
    /// Returns a value in `[-1.0, 1.0]`:
    /// - `+1.0` — channels are identical (mono).
    /// - `0.0`  — channels are uncorrelated.
    /// - `-1.0` — channels are perfectly out-of-phase.
    ///
    /// Returns `0.0` for constant (zero-variance) signals to avoid division by zero.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] or [`MsError::LengthMismatch`].
    pub fn correlation(left: &[f32], right: &[f32]) -> Result<f32, MsError> {
        if left.is_empty() || right.is_empty() {
            return Err(MsError::EmptyInput);
        }
        if left.len() != right.len() {
            return Err(MsError::LengthMismatch(left.len(), right.len()));
        }

        let n = left.len() as f64;
        let mean_l = left.iter().map(|&s| f64::from(s)).sum::<f64>() / n;
        let mean_r = right.iter().map(|&s| f64::from(s)).sum::<f64>() / n;

        let mut cov = 0.0f64;
        let mut var_l = 0.0f64;
        let mut var_r = 0.0f64;

        for (&l, &r) in left.iter().zip(right.iter()) {
            let dl = f64::from(l) - mean_l;
            let dr = f64::from(r) - mean_r;
            cov += dl * dr;
            var_l += dl * dl;
            var_r += dr * dr;
        }

        let denom = (var_l * var_r).sqrt();
        if denom < f64::EPSILON {
            return Ok(0.0);
        }
        Ok((cov / denom).clamp(-1.0, 1.0) as f32)
    }

    /// Estimate the **stereo width** as a normalised value.
    ///
    /// Uses the ratio of Side RMS to the sum of Mid and Side RMS:
    ///
    /// ```text
    /// width = rms(side) / (rms(mid) + rms(side))
    /// ```
    ///
    /// Returns `0.0` when the signal is mono (no Side energy) and approaches
    /// `1.0` as Side energy increases.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] or [`MsError::LengthMismatch`].
    pub fn stereo_width(left: &[f32], right: &[f32]) -> Result<f32, MsError> {
        let (mid, side) = MsEncoder::encode(left, right)?;
        let mid_rms = rms(&mid);
        let side_rms = rms(&side);
        let total = mid_rms + side_rms;
        if total < f32::EPSILON {
            return Ok(0.0);
        }
        Ok((side_rms / total).clamp(0.0, 1.0))
    }

    /// Returns `true` when the stereo field has significant **phase problems**.
    ///
    /// Phase issues are detected when the channel correlation drops below
    /// `−0.5`, which indicates that left and right are substantially out of
    /// phase and will largely cancel when summed to mono.
    ///
    /// # Errors
    ///
    /// Returns [`MsError::EmptyInput`] or [`MsError::LengthMismatch`].
    pub fn phase_issues(left: &[f32], right: &[f32]) -> Result<bool, MsError> {
        let corr = Self::correlation(left, right)?;
        Ok(corr < -0.5)
    }
}

// ─── DSP utility ─────────────────────────────────────────────────────────────

#[inline]
fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, amp: f32, n: usize, fs: u32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs as f32).sin())
            .collect()
    }

    // ── Encode / decode roundtrip ──────────────────────────────────────────

    #[test]
    fn encode_decode_roundtrip() {
        let left = sine(440.0, 0.8, 1024, 48_000);
        let right = sine(880.0, 0.5, 1024, 48_000);
        let (mid, side) = MsEncoder::encode(&left, &right).expect("encode ok");
        let (left2, right2) = MsDecoder::decode(&mid, &side).expect("decode ok");
        for i in 0..1024 {
            let diff_l = (left[i] - left2[i]).abs();
            let diff_r = (right[i] - right2[i]).abs();
            assert!(diff_l < 1e-6, "L roundtrip error at {i}: {diff_l}");
            assert!(diff_r < 1e-6, "R roundtrip error at {i}: {diff_r}");
        }
    }

    // ── Mono signal → zero side channel ───────────────────────────────────

    #[test]
    fn mono_signal_produces_zero_side() {
        let mono = sine(1_000.0, 0.7, 512, 48_000);
        let (_, side) = MsEncoder::encode(&mono, &mono).expect("encode ok");
        for (i, &s) in side.iter().enumerate() {
            assert!(s.abs() < 1e-6, "side should be zero at sample {i}, got {s}");
        }
    }

    // ── Width = 0 → mono output ───────────────────────────────────────────

    #[test]
    fn width_zero_produces_mono_output() {
        let proc = MsProcessor::new(0.0).expect("valid width");
        let mut left = sine(440.0, 0.8, 512, 48_000);
        let mut right = sine(880.0, 0.5, 512, 48_000);
        proc.process(&mut left, &mut right).expect("process ok");
        // After width=0: L = mid, R = mid → they must be equal
        for i in 0..512 {
            let diff = (left[i] - right[i]).abs();
            assert!(diff < 1e-6, "L and R should be equal at {i}: L={} R={}", left[i], right[i]);
        }
    }

    // ── Width = 1 → original stereo ───────────────────────────────────────

    #[test]
    fn width_one_preserves_original_stereo() {
        let proc = MsProcessor::new(1.0).expect("valid width");
        let original_l = sine(440.0, 0.8, 512, 48_000);
        let original_r = sine(880.0, 0.5, 512, 48_000);
        let mut left = original_l.clone();
        let mut right = original_r.clone();
        proc.process(&mut left, &mut right).expect("process ok");
        for i in 0..512 {
            let diff_l = (left[i] - original_l[i]).abs();
            let diff_r = (right[i] - original_r[i]).abs();
            assert!(diff_l < 1e-5, "L changed at {i}: {diff_l}");
            assert!(diff_r < 1e-5, "R changed at {i}: {diff_r}");
        }
    }

    // ── Correlation: identical signals → +1.0 ─────────────────────────────

    #[test]
    fn correlation_identical_signals_is_one() {
        let sig = sine(440.0, 0.7, 1024, 48_000);
        let corr = MsStereoAnalyzer::correlation(&sig, &sig).expect("ok");
        assert!((corr - 1.0).abs() < 1e-4, "correlation should be ~1.0, got {corr}");
    }

    // ── Correlation: inverted signals → -1.0 ─────────────────────────────

    #[test]
    fn correlation_inverted_signals_is_minus_one() {
        let sig = sine(440.0, 0.7, 1024, 48_000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        let corr = MsStereoAnalyzer::correlation(&sig, &inv).expect("ok");
        assert!((corr + 1.0).abs() < 1e-4, "correlation should be ~-1.0, got {corr}");
    }

    // ── Phase issues detected for inverted signal ─────────────────────────

    #[test]
    fn phase_issues_detected_for_inverted_signal() {
        let sig = sine(440.0, 0.7, 1024, 48_000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        let issues = MsStereoAnalyzer::phase_issues(&sig, &inv).expect("ok");
        assert!(issues, "inverted signals should trigger phase issue flag");
    }

    // ── No phase issues for identical signals ─────────────────────────────

    #[test]
    fn no_phase_issues_for_identical_signals() {
        let sig = sine(440.0, 0.7, 1024, 48_000);
        let issues = MsStereoAnalyzer::phase_issues(&sig, &sig).expect("ok");
        assert!(!issues, "identical signals should not trigger phase issue flag");
    }

    // ── Error: empty input ────────────────────────────────────────────────

    #[test]
    fn encode_empty_returns_error() {
        assert!(matches!(MsEncoder::encode(&[], &[]), Err(MsError::EmptyInput)));
    }

    #[test]
    fn decode_empty_returns_error() {
        assert!(matches!(MsDecoder::decode(&[], &[]), Err(MsError::EmptyInput)));
    }

    // ── Error: length mismatch ────────────────────────────────────────────

    #[test]
    fn encode_length_mismatch_returns_error() {
        let left = vec![0.0f32; 10];
        let right = vec![0.0f32; 20];
        assert!(matches!(
            MsEncoder::encode(&left, &right),
            Err(MsError::LengthMismatch(10, 20))
        ));
    }

    // ── Invalid width ────────────────────────────────────────────────────

    #[test]
    fn invalid_width_returns_error() {
        assert!(matches!(MsProcessor::new(-0.1), Err(MsError::InvalidWidth(_))));
        assert!(matches!(MsProcessor::new(4.1), Err(MsError::InvalidWidth(_))));
    }

    // ── Stereo width: mono signal → 0.0 ──────────────────────────────────

    #[test]
    fn stereo_width_mono_is_zero() {
        let sig = sine(440.0, 0.8, 512, 48_000);
        let width = MsStereoAnalyzer::stereo_width(&sig, &sig).expect("ok");
        assert!(width < 1e-4, "mono signal should have zero width, got {width}");
    }

    // ── Stereo width: fully opposed signals → max width ───────────────────

    #[test]
    fn stereo_width_opposed_signals_is_max() {
        let sig = sine(440.0, 0.8, 512, 48_000);
        let inv: Vec<f32> = sig.iter().map(|&s| -s).collect();
        // L+R = 0 → mid = 0, side = L → width approaches 1.0
        let width = MsStereoAnalyzer::stereo_width(&sig, &inv).expect("ok");
        assert!(width > 0.9, "fully opposed signals should have high width, got {width}");
    }

    // ── MsProcessor: process errors ──────────────────────────────────────

    #[test]
    fn processor_empty_returns_error() {
        let proc = MsProcessor::default();
        assert!(matches!(
            proc.process(&mut [], &mut []),
            Err(MsError::EmptyInput)
        ));
    }

    #[test]
    fn processor_length_mismatch_returns_error() {
        let proc = MsProcessor::default();
        let mut l = vec![0.0f32; 10];
        let mut r = vec![0.0f32; 5];
        assert!(matches!(
            proc.process(&mut l, &mut r),
            Err(MsError::LengthMismatch(10, 5))
        ));
    }
}
