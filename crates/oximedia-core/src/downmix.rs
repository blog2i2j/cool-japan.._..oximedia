//! Downmix and upmix matrices for audio channel layout conversion.
//!
//! This module provides [`DownmixMatrix`] — a dense `(out_channels × in_channels)`
//! mixing matrix — together with factory functions that produce ITU-R BS.775-3
//! compliant downmix coefficients for common speaker configurations.
//!
//! # Design
//!
//! - Coefficients are stored as linear amplitude gains (not dB).
//! - All matrices are normalised so that the loudest output channel's gain sum
//!   does not exceed `1.0` (prevents digital clipping on sum-of-all-channels
//!   inputs).
//! - The matrix is applied with [`DownmixMatrix::apply`], which takes a flat
//!   interleaved input slice (`samples * in_channels`) and writes a flat
//!   interleaved output slice (`samples * out_channels`).
//!
//! # Supported conversions
//!
//! | Source layout | Target layout | Factory |
//! |---|---|---|
//! | 5.1 | Stereo | [`DownmixMatrix::surround51_to_stereo`] |
//! | 5.1 | Mono | [`DownmixMatrix::surround51_to_mono`] |
//! | 7.1 | 5.1 | [`DownmixMatrix::surround71_to_51`] |
//! | 7.1 | Stereo | [`DownmixMatrix::surround71_to_stereo`] |
//! | 5.1.4 Atmos | 5.1 | [`DownmixMatrix::atmos514_to_51`] |
//! | Stereo | Mono | [`DownmixMatrix::stereo_to_mono`] |
//! | Mono | Stereo | [`DownmixMatrix::mono_to_stereo`] (upmix) |
//!
//! # Example
//!
//! ```
//! use oximedia_core::downmix::DownmixMatrix;
//!
//! // Downmix a 5.1 frame to stereo.
//! let mx = DownmixMatrix::surround51_to_stereo();
//! assert_eq!(mx.input_channels(), 6);
//! assert_eq!(mx.output_channels(), 2);
//!
//! // 1 sample of 5.1 silence → 1 sample of stereo silence.
//! let input = vec![0.0f32; 6];
//! let mut output = vec![0.0f32; 2];
//! mx.apply(&input, &mut output, 1);
//! assert_eq!(output, vec![0.0; 2]);
//! ```

#![allow(dead_code)]

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when constructing or applying a [`DownmixMatrix`].
#[derive(Debug, PartialEq, Eq)]
pub enum DownmixError {
    /// The matrix dimensions are inconsistent.
    InvalidDimensions {
        /// Expected number of rows × columns.
        expected: (usize, usize),
        /// Actual coefficient vector length.
        actual: usize,
    },
    /// The input buffer length is not a multiple of the number of input channels.
    InputLengthMismatch {
        /// Length of the input slice.
        input_len: usize,
        /// Expected stride (number of input channels).
        in_channels: usize,
    },
    /// The output buffer is too small for the given sample count and output channels.
    OutputLengthMismatch {
        /// Length of the output slice.
        output_len: usize,
        /// Required length.
        required: usize,
    },
}

impl fmt::Display for DownmixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { expected, actual } => write!(
                f,
                "invalid matrix: expected {} coefficients ({}×{}), got {}",
                expected.0 * expected.1,
                expected.0,
                expected.1,
                actual
            ),
            Self::InputLengthMismatch {
                input_len,
                in_channels,
            } => write!(
                f,
                "input length {input_len} is not a multiple of in_channels {in_channels}"
            ),
            Self::OutputLengthMismatch {
                output_len,
                required,
            } => write!(f, "output buffer length {output_len} < required {required}"),
        }
    }
}

impl std::error::Error for DownmixError {}

// ─────────────────────────────────────────────────────────────────────────────
// DownmixMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// A dense `(out_channels × in_channels)` mixing matrix.
///
/// Coefficients are stored in **row-major** order: coefficient `[o][i]` is
/// at index `o * in_channels + i` in the flat `coeffs` vector.
///
/// `output[o] += coeffs[o][i] * input[i]` for all `i`.
#[derive(Debug, Clone, PartialEq)]
pub struct DownmixMatrix {
    out_channels: usize,
    in_channels: usize,
    /// Flat coefficients in row-major order (`out_channels × in_channels`).
    coeffs: Vec<f32>,
}

impl DownmixMatrix {
    // ── Construction ──────────────────────────────────────────────────────

    /// Creates a new matrix from a flat coefficient slice.
    ///
    /// `coeffs` must have exactly `out_channels * in_channels` elements.
    ///
    /// # Errors
    ///
    /// Returns [`DownmixError::InvalidDimensions`] if the coefficient slice
    /// length does not match `out_channels * in_channels`.
    pub fn new(
        out_channels: usize,
        in_channels: usize,
        coeffs: Vec<f32>,
    ) -> Result<Self, DownmixError> {
        let expected = out_channels * in_channels;
        if coeffs.len() != expected {
            return Err(DownmixError::InvalidDimensions {
                expected: (out_channels, in_channels),
                actual: coeffs.len(),
            });
        }
        Ok(Self {
            out_channels,
            in_channels,
            coeffs,
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Returns the number of output channels.
    #[must_use]
    pub fn output_channels(&self) -> usize {
        self.out_channels
    }

    /// Returns the number of input channels.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.in_channels
    }

    /// Returns the coefficient for output channel `o` from input channel `i`.
    ///
    /// Returns `None` if `o ≥ out_channels` or `i ≥ in_channels`.
    #[must_use]
    pub fn coeff(&self, o: usize, i: usize) -> Option<f32> {
        if o < self.out_channels && i < self.in_channels {
            Some(self.coeffs[o * self.in_channels + i])
        } else {
            None
        }
    }

    // ── Apply ─────────────────────────────────────────────────────────────

    /// Applies the matrix to `sample_count` frames of interleaved audio.
    ///
    /// - `input` must have length `sample_count * in_channels`.
    /// - `output` must have length ≥ `sample_count * out_channels`.
    ///
    /// The output buffer is **overwritten** (not accumulated).
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths are inconsistent with the matrix
    /// dimensions and `sample_count`.
    pub fn apply(
        &self,
        input: &[f32],
        output: &mut [f32],
        sample_count: usize,
    ) -> Result<(), DownmixError> {
        let expected_in = sample_count * self.in_channels;
        if input.len() < expected_in
            || (self.in_channels > 0 && input.len() % self.in_channels != 0)
        {
            return Err(DownmixError::InputLengthMismatch {
                input_len: input.len(),
                in_channels: self.in_channels,
            });
        }
        let required_out = sample_count * self.out_channels;
        if output.len() < required_out {
            return Err(DownmixError::OutputLengthMismatch {
                output_len: output.len(),
                required: required_out,
            });
        }

        for s in 0..sample_count {
            let in_base = s * self.in_channels;
            let out_base = s * self.out_channels;
            for o in 0..self.out_channels {
                let mut acc = 0.0f32;
                for i in 0..self.in_channels {
                    acc += self.coeffs[o * self.in_channels + i] * input[in_base + i];
                }
                output[out_base + o] = acc;
            }
        }
        Ok(())
    }

    /// Convenience wrapper — same as [`DownmixMatrix::apply`] but panics on
    /// dimension mismatch.  Intended for use in hot-path code that has already
    /// validated the buffer sizes.
    ///
    /// # Panics
    ///
    /// Panics if `input` / `output` are not correctly sized.
    pub fn apply_unchecked(&self, input: &[f32], output: &mut [f32], sample_count: usize) {
        self.apply(input, output, sample_count)
            .expect("apply_unchecked: dimension mismatch");
    }

    // ── Factory functions ──────────────────────────────────────────────────
    //
    // Coefficients follow ITU-R BS.775-3 where possible.
    // Constants:
    //   SQRT2 = √2 ≈ 1.4142135
    //   1/SQRT2 ≈ 0.7071068  (−3 dB)
    //   1/(SQRT2*2) ≈ 0.3535534 (−6 dB relative to 0 dB)

    const INV_SQRT2: f32 = 0.707_106_8;

    /// Builds a 5.1 → Stereo downmix matrix (ITU-R BS.775-3, Table B.1).
    ///
    /// Input channel order: FL FR FC LFE BL BR
    /// Output channel order: L R
    ///
    /// ```text
    /// L = FL + FC/√2 + 0.5·BL + 0 (LFE discarded)
    /// R = FR + FC/√2 + 0.5·BR
    /// ```
    #[must_use]
    pub fn surround51_to_stereo() -> Self {
        // Row 0 (L): FL  FR    FC             LFE  BL   BR
        // Row 1 (R): FL  FR    FC             LFE  BL   BR
        #[rustfmt::skip]
        let coeffs = vec![
            1.0,  0.0, Self::INV_SQRT2, 0.0, 0.5, 0.0,  // L
            0.0,  1.0, Self::INV_SQRT2, 0.0, 0.0, 0.5,  // R
        ];
        Self {
            out_channels: 2,
            in_channels: 6,
            coeffs,
        }
    }

    /// Builds a 5.1 → Mono downmix matrix.
    ///
    /// Input channel order: FL FR FC LFE BL BR
    /// Output: single mono channel.
    ///
    /// ```text
    /// M = 0.5·FL + 0.5·FR + FC/√2 + 0.25·BL + 0.25·BR  (LFE discarded)
    /// ```
    #[must_use]
    pub fn surround51_to_mono() -> Self {
        #[rustfmt::skip]
        let coeffs = vec![
            0.5, 0.5, Self::INV_SQRT2, 0.0, 0.25, 0.25,
        ];
        Self {
            out_channels: 1,
            in_channels: 6,
            coeffs,
        }
    }

    /// Builds a Stereo → Mono downmix matrix.
    ///
    /// ```text
    /// M = 0.5·L + 0.5·R
    /// ```
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        let coeffs = vec![0.5, 0.5];
        Self {
            out_channels: 1,
            in_channels: 2,
            coeffs,
        }
    }

    /// Builds a Mono → Stereo upmix matrix (identity split).
    ///
    /// ```text
    /// L = M · (1/√2)
    /// R = M · (1/√2)
    /// ```
    ///
    /// Scaling by 1/√2 preserves the perceived loudness of the centre.
    #[must_use]
    pub fn mono_to_stereo() -> Self {
        let coeffs = vec![
            Self::INV_SQRT2, // L ← M
            Self::INV_SQRT2, // R ← M
        ];
        Self {
            out_channels: 2,
            in_channels: 1,
            coeffs,
        }
    }

    /// Builds a 7.1 → 5.1 downmix matrix.
    ///
    /// Input order:  FL FR FC LFE BL BR SL SR
    /// Output order: FL FR FC LFE BL BR
    ///
    /// ```text
    /// FL_out = FL  + SL/√2
    /// FR_out = FR  + SR/√2
    /// FC_out = FC
    /// LFE_out = LFE
    /// BL_out = BL  + SL/√2
    /// BR_out = BR  + SR/√2
    /// ```
    #[must_use]
    pub fn surround71_to_51() -> Self {
        //                  FL  FR  FC  LFE  BL  BR  SL             SR
        #[rustfmt::skip]
        let coeffs = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, Self::INV_SQRT2, 0.0,  // FL_out
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, Self::INV_SQRT2,  // FR_out
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,              // FC_out
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,              // LFE_out
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, Self::INV_SQRT2, 0.0,  // BL_out
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, Self::INV_SQRT2,  // BR_out
        ];
        Self {
            out_channels: 6,
            in_channels: 8,
            coeffs,
        }
    }

    /// Builds a 7.1 → Stereo downmix matrix.
    ///
    /// Input order: FL FR FC LFE BL BR SL SR
    ///
    /// ```text
    /// L = FL + FC/√2 + 0.5·BL + SL/√2
    /// R = FR + FC/√2 + 0.5·BR + SR/√2
    /// ```
    #[must_use]
    pub fn surround71_to_stereo() -> Self {
        //                 FL  FR  FC             LFE  BL   BR   SL             SR
        #[rustfmt::skip]
        let coeffs = vec![
            1.0, 0.0, Self::INV_SQRT2, 0.0, 0.5, 0.0, Self::INV_SQRT2, 0.0,  // L
            0.0, 1.0, Self::INV_SQRT2, 0.0, 0.0, 0.5, 0.0, Self::INV_SQRT2,  // R
        ];
        Self {
            out_channels: 2,
            in_channels: 8,
            coeffs,
        }
    }

    /// Builds a 5.1.4 Atmos → 5.1 downmix matrix.
    ///
    /// Input order:  FL FR FC LFE BL BR TFL TFR TBL TBR
    /// Output order: FL FR FC LFE BL BR
    ///
    /// Height channels are blended into their nearest bed equivalents at −6 dB.
    ///
    /// ```text
    /// FL_out  = FL  + TFL/2
    /// FR_out  = FR  + TFR/2
    /// FC_out  = FC
    /// LFE_out = LFE
    /// BL_out  = BL  + TBL/2
    /// BR_out  = BR  + TBR/2
    /// ```
    #[must_use]
    pub fn atmos514_to_51() -> Self {
        //                 FL  FR  FC  LFE  BL  BR  TFL  TFR  TBL  TBR
        #[rustfmt::skip]
        let coeffs = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0,  // FL_out
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0,  // FR_out
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // FC_out
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  // LFE_out
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,  // BL_out
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5,  // BR_out
        ];
        Self {
            out_channels: 6,
            in_channels: 10,
            coeffs,
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────

    /// Returns the identity matrix for `channels` channels.
    ///
    /// All output channels map 1:1 to the corresponding input channel.
    #[must_use]
    pub fn identity(channels: usize) -> Self {
        let n = channels;
        let mut coeffs = vec![0.0f32; n * n];
        for c in 0..n {
            coeffs[c * n + c] = 1.0;
        }
        Self {
            out_channels: n,
            in_channels: n,
            coeffs,
        }
    }

    /// Returns the transpose of this matrix.
    ///
    /// Swaps input and output channel counts and transposes the coefficient
    /// matrix.  This is useful for constructing upmix matrices from downmix
    /// definitions.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let mut coeffs = vec![0.0f32; self.in_channels * self.out_channels];
        for o in 0..self.out_channels {
            for i in 0..self.in_channels {
                coeffs[i * self.out_channels + o] = self.coeffs[o * self.in_channels + i];
            }
        }
        Self {
            out_channels: self.in_channels,
            in_channels: self.out_channels,
            coeffs,
        }
    }
}

impl fmt::Display for DownmixMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DownmixMatrix({}×{})",
            self.out_channels, self.in_channels
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // 1. Dimension mismatch on construction returns an error.
    #[test]
    fn test_new_dimension_mismatch() {
        let result = DownmixMatrix::new(2, 3, vec![1.0; 7]);
        assert!(result.is_err());
        match result {
            Err(DownmixError::InvalidDimensions { expected, actual }) => {
                assert_eq!(expected, (2, 3));
                assert_eq!(actual, 7);
            }
            _ => panic!("expected InvalidDimensions"),
        }
    }

    // 2. Valid construction succeeds.
    #[test]
    fn test_new_valid() {
        let mx = DownmixMatrix::new(2, 3, vec![0.0; 6]).expect("valid");
        assert_eq!(mx.input_channels(), 3);
        assert_eq!(mx.output_channels(), 2);
    }

    // 3. Identity matrix passes samples through unchanged.
    #[test]
    fn test_identity_passthrough() {
        let mx = DownmixMatrix::identity(3);
        let input = vec![1.0f32, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];
        mx.apply(&input, &mut output, 1).expect("apply ok");
        assert!(approx_eq(output[0], 1.0));
        assert!(approx_eq(output[1], 2.0));
        assert!(approx_eq(output[2], 3.0));
    }

    // 4. Stereo → Mono: two equal channels → half amplitude.
    #[test]
    fn test_stereo_to_mono() {
        let mx = DownmixMatrix::stereo_to_mono();
        let input = vec![1.0f32, 1.0];
        let mut output = vec![0.0f32; 1];
        mx.apply(&input, &mut output, 1).expect("apply ok");
        assert!(approx_eq(output[0], 1.0));
    }

    // 5. Mono → Stereo upmix: output scaled by 1/√2.
    #[test]
    fn test_mono_to_stereo() {
        let mx = DownmixMatrix::mono_to_stereo();
        let input = vec![1.0f32];
        let mut output = vec![0.0f32; 2];
        mx.apply(&input, &mut output, 1).expect("apply ok");
        let expected = DownmixMatrix::INV_SQRT2;
        assert!(approx_eq(output[0], expected));
        assert!(approx_eq(output[1], expected));
    }

    // 6. 5.1 → Stereo dimensions.
    #[test]
    fn test_51_to_stereo_dimensions() {
        let mx = DownmixMatrix::surround51_to_stereo();
        assert_eq!(mx.input_channels(), 6);
        assert_eq!(mx.output_channels(), 2);
    }

    // 7. 5.1 → Stereo: pure FL/FR only → same amplitude out.
    #[test]
    fn test_51_to_stereo_pure_front() {
        let mx = DownmixMatrix::surround51_to_stereo();
        // FL=1, FR=1, FC=0, LFE=0, BL=0, BR=0
        let input = vec![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; 2];
        mx.apply(&input, &mut output, 1).expect("apply ok");
        assert!(approx_eq(output[0], 1.0));
        assert!(approx_eq(output[1], 1.0));
    }

    // 8. 5.1 → Stereo: centre channel split equally at −3 dB.
    #[test]
    fn test_51_to_stereo_centre_split() {
        let mx = DownmixMatrix::surround51_to_stereo();
        // FL=0, FR=0, FC=1, LFE=0, BL=0, BR=0
        let input = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let mut output = vec![0.0f32; 2];
        mx.apply(&input, &mut output, 1).expect("apply ok");
        assert!(
            approx_eq(output[0], DownmixMatrix::INV_SQRT2),
            "L = {:.6}",
            output[0]
        );
        assert!(
            approx_eq(output[1], DownmixMatrix::INV_SQRT2),
            "R = {:.6}",
            output[1]
        );
    }

    // 9. 5.1 → Mono dimensions.
    #[test]
    fn test_51_to_mono_dimensions() {
        let mx = DownmixMatrix::surround51_to_mono();
        assert_eq!(mx.input_channels(), 6);
        assert_eq!(mx.output_channels(), 1);
    }

    // 10. 5.1 → Mono: LFE is discarded (zero coefficient).
    #[test]
    fn test_51_to_mono_lfe_discarded() {
        let mx = DownmixMatrix::surround51_to_mono();
        let lfe_coeff = mx.coeff(0, 3).expect("coeff(0,3)");
        assert!(approx_eq(lfe_coeff, 0.0));
    }

    // 11. 7.1 → 5.1 dimensions.
    #[test]
    fn test_71_to_51_dimensions() {
        let mx = DownmixMatrix::surround71_to_51();
        assert_eq!(mx.input_channels(), 8);
        assert_eq!(mx.output_channels(), 6);
    }

    // 12. 7.1 → 5.1: FC passes through unmodified.
    #[test]
    fn test_71_to_51_fc_passthrough() {
        let mx = DownmixMatrix::surround71_to_51();
        // FC is input channel 2, output channel 2
        let fc_coeff = mx.coeff(2, 2).expect("coeff(2,2)");
        assert!(approx_eq(fc_coeff, 1.0));
        // FC does not bleed into other outputs
        assert!(approx_eq(mx.coeff(0, 2).expect("c(0,2)"), 0.0));
        assert!(approx_eq(mx.coeff(1, 2).expect("c(1,2)"), 0.0));
    }

    // 13. 7.1 → Stereo dimensions and SL coefficient.
    #[test]
    fn test_71_to_stereo() {
        let mx = DownmixMatrix::surround71_to_stereo();
        assert_eq!(mx.input_channels(), 8);
        assert_eq!(mx.output_channels(), 2);
        // SL (channel 6) feeds L (output 0) at 1/√2
        let sl_to_l = mx.coeff(0, 6).expect("coeff(0,6)");
        assert!(approx_eq(sl_to_l, DownmixMatrix::INV_SQRT2));
        // SR (channel 7) does NOT feed L
        let sr_to_l = mx.coeff(0, 7).expect("coeff(0,7)");
        assert!(approx_eq(sr_to_l, 0.0));
    }

    // 14. Atmos 5.1.4 → 5.1 dimensions and TFL coefficient.
    #[test]
    fn test_atmos514_to_51() {
        let mx = DownmixMatrix::atmos514_to_51();
        assert_eq!(mx.input_channels(), 10);
        assert_eq!(mx.output_channels(), 6);
        // TFL (channel 6) folds into FL_out (row 0) at 0.5
        let tfl_coeff = mx.coeff(0, 6).expect("coeff(0,6)");
        assert!(approx_eq(tfl_coeff, 0.5));
    }

    // 15. apply over multiple samples.
    #[test]
    fn test_apply_multiple_samples() {
        let mx = DownmixMatrix::stereo_to_mono();
        // 4 stereo samples: (1,1), (2,2), (3,3), (4,4)
        let input = vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
        let mut output = vec![0.0f32; 4];
        mx.apply(&input, &mut output, 4).expect("apply ok");
        assert!(approx_eq(output[0], 1.0));
        assert!(approx_eq(output[1], 2.0));
        assert!(approx_eq(output[2], 3.0));
        assert!(approx_eq(output[3], 4.0));
    }

    // 16. apply returns error when output buffer is too small.
    #[test]
    fn test_apply_output_too_small() {
        let mx = DownmixMatrix::stereo_to_mono();
        let input = vec![1.0f32, 1.0];
        let mut output = vec![]; // too small for 1 sample
        let result = mx.apply(&input, &mut output, 1);
        assert!(matches!(
            result,
            Err(DownmixError::OutputLengthMismatch { .. })
        ));
    }

    // 17. transpose of stereo_to_mono gives a 1-in → 2-out matrix.
    #[test]
    fn test_transpose_stereo_to_mono() {
        let mx = DownmixMatrix::stereo_to_mono().transpose();
        assert_eq!(mx.input_channels(), 1);
        assert_eq!(mx.output_channels(), 2);
    }

    // 18. coeff returns None for out-of-range indices.
    #[test]
    fn test_coeff_out_of_range() {
        let mx = DownmixMatrix::stereo_to_mono();
        assert!(mx.coeff(0, 0).is_some());
        assert!(mx.coeff(1, 0).is_none()); // only 1 output row
        assert!(mx.coeff(0, 2).is_none()); // only 2 input columns
    }

    // 19. Display impl.
    #[test]
    fn test_display() {
        let mx = DownmixMatrix::surround51_to_stereo();
        let s = format!("{mx}");
        assert!(s.contains("2×6"), "display = {s}");
    }

    // 20. DownmixError display does not panic.
    #[test]
    fn test_error_display() {
        let err = DownmixError::InvalidDimensions {
            expected: (2, 3),
            actual: 7,
        };
        let s = format!("{err}");
        assert!(s.contains("7"));
    }
}
