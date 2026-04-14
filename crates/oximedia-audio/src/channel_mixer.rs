//! Channel mixing matrix for N-to-M channel conversion.
//!
//! Provides a flexible mixing matrix that maps an input signal with N channels
//! to an output signal with M channels, with per-element gain coefficients.
//! Standard downmix (5.1 → stereo, 7.1 → 5.1, etc.) and upmix (mono → stereo,
//! stereo → 5.1, etc.) matrices are provided as defaults, and fully custom
//! matrices may be constructed with [`MixMatrix::new`].
//!
//! # Quick start
//!
//! ```
//! use oximedia_audio::channel_mixer::{ChannelMixer, MixMatrix};
//!
//! // Downmix stereo → mono
//! let matrix = MixMatrix::stereo_to_mono();
//! let mut mixer = ChannelMixer::new(matrix);
//!
//! let stereo_frame = vec![0.8_f32, 0.6_f32, 0.4_f32, 0.2_f32]; // L R L R
//! let mono = mixer.process_interleaved(&stereo_frame);
//! assert_eq!(mono.len(), 2); // 2 mono samples
//! ```
//!
//! # Supported standard matrices
//!
//! | Name | Function |
//! |---|---|
//! | [`MixMatrix::mono_to_stereo`] | 1→2: duplicate mono to L+R |
//! | [`MixMatrix::stereo_to_mono`] | 2→1: average L and R |
//! | [`MixMatrix::stereo_to_51`] | 2→6: stereo to 5.1 surround |
//! | [`MixMatrix::surround51_to_stereo`] | 6→2: 5.1 to stereo downmix (ITU-R BS.775) |
//! | [`MixMatrix::surround71_to_51`] | 8→6: 7.1 to 5.1 downmix |
//! | [`MixMatrix::surround51_to_71`] | 6→8: 5.1 to 7.1 upmix |

#![forbid(unsafe_code)]

use crate::{AudioError, AudioResult};

// ─────────────────────────────────────────────────────────────────────────────
// MixMatrix
// ─────────────────────────────────────────────────────────────────────────────

/// A gain matrix mapping N input channels to M output channels.
///
/// Element `[out_ch][in_ch]` is the linear gain applied from input channel
/// `in_ch` to output channel `out_ch`.
#[derive(Clone, Debug)]
pub struct MixMatrix {
    /// Number of input channels.
    pub input_channels: usize,
    /// Number of output channels.
    pub output_channels: usize,
    /// Row-major matrix: `coeffs[out][in]`.
    coeffs: Vec<Vec<f32>>,
}

impl MixMatrix {
    /// Create a new mixing matrix initialised to silence (all zeros).
    ///
    /// Set coefficients via [`MixMatrix::set`] before use.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if either dimension is zero.
    pub fn new(input_channels: usize, output_channels: usize) -> AudioResult<Self> {
        if input_channels == 0 || output_channels == 0 {
            return Err(AudioError::InvalidParameter(
                "channel counts must be greater than zero".into(),
            ));
        }
        let coeffs = vec![vec![0.0_f32; input_channels]; output_channels];
        Ok(Self {
            input_channels,
            output_channels,
            coeffs,
        })
    }

    /// Set the gain for a specific (output, input) pair.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if indices are out of bounds.
    pub fn set(&mut self, out_ch: usize, in_ch: usize, gain: f32) -> AudioResult<()> {
        if out_ch >= self.output_channels || in_ch >= self.input_channels {
            return Err(AudioError::InvalidParameter(format!(
                "index ({out_ch}, {in_ch}) out of bounds ({} output × {} input)",
                self.output_channels, self.input_channels
            )));
        }
        self.coeffs[out_ch][in_ch] = gain;
        Ok(())
    }

    /// Read the gain for a specific (output, input) pair.
    ///
    /// Returns `0.0` for out-of-bounds indices rather than panicking.
    #[must_use]
    pub fn get(&self, out_ch: usize, in_ch: usize) -> f32 {
        self.coeffs
            .get(out_ch)
            .and_then(|row| row.get(in_ch))
            .copied()
            .unwrap_or(0.0)
    }

    /// Apply the matrix to one multi-channel sample (one sample per channel).
    ///
    /// `input` must have length ≥ `self.input_channels`.
    /// Returns a `Vec<f32>` of length `self.output_channels`.
    #[must_use]
    pub fn apply_sample(&self, input: &[f32]) -> Vec<f32> {
        let in_len = input.len().min(self.input_channels);
        let mut output = vec![0.0_f32; self.output_channels];
        for (out_ch, out_val) in output.iter_mut().enumerate() {
            for in_ch in 0..in_len {
                *out_val += self.coeffs[out_ch][in_ch] * input[in_ch];
            }
        }
        output
    }

    // ── Standard matrices ─────────────────────────────────────────────────────

    /// Mono → Stereo: duplicate the single channel to L and R at unity gain.
    #[must_use]
    pub fn mono_to_stereo() -> Self {
        let mut m = Self {
            input_channels: 1,
            output_channels: 2,
            coeffs: vec![vec![0.0_f32; 1]; 2],
        };
        // L = in0, R = in0
        m.coeffs[0][0] = 1.0;
        m.coeffs[1][0] = 1.0;
        m
    }

    /// Stereo → Mono: sum L and R with equal gain (`0.5` each) to prevent
    /// clipping on correlated material.
    #[must_use]
    pub fn stereo_to_mono() -> Self {
        let mut m = Self {
            input_channels: 2,
            output_channels: 1,
            coeffs: vec![vec![0.0_f32; 2]; 1],
        };
        // mono = 0.5 * L + 0.5 * R
        m.coeffs[0][0] = 0.5;
        m.coeffs[0][1] = 0.5;
        m
    }

    /// Stereo → 5.1 upmix.
    ///
    /// Channel order: `[L, R, C, LFE, Ls, Rs]`
    ///
    /// - Front L/R at unity.
    /// - Centre derived as −3 dB mix of L+R.
    /// - Surround Ls/Rs derived as −3 dB.
    /// - LFE is silent.
    #[must_use]
    pub fn stereo_to_51() -> Self {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2; // ≈ 0.707
        let mut m = Self {
            input_channels: 2,
            output_channels: 6,
            coeffs: vec![vec![0.0_f32; 2]; 6],
        };
        // L → L
        m.coeffs[0][0] = 1.0;
        // R → R
        m.coeffs[1][1] = 1.0;
        // C = (L + R) * 1/√2
        m.coeffs[2][0] = INV_SQRT2;
        m.coeffs[2][1] = INV_SQRT2;
        // LFE = 0
        // Ls ← L * 1/√2
        m.coeffs[4][0] = INV_SQRT2;
        // Rs ← R * 1/√2
        m.coeffs[5][1] = INV_SQRT2;
        m
    }

    /// 5.1 → Stereo downmix (ITU-R BS.775 coefficients).
    ///
    /// Channel order: `[L, R, C, LFE, Ls, Rs]`
    ///
    /// ```text
    /// Lo = L  + C/√2 + Ls/√2  (LFE ignored)
    /// Ro = R  + C/√2 + Rs/√2
    /// ```
    #[must_use]
    pub fn surround51_to_stereo() -> Self {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let mut m = Self {
            input_channels: 6,
            output_channels: 2,
            coeffs: vec![vec![0.0_f32; 6]; 2],
        };
        // Lo = L  + C/√2 + Ls/√2
        m.coeffs[0][0] = 1.0; // L
        m.coeffs[0][2] = INV_SQRT2; // C
        m.coeffs[0][4] = INV_SQRT2; // Ls
        // Ro = R  + C/√2 + Rs/√2
        m.coeffs[1][1] = 1.0; // R
        m.coeffs[1][2] = INV_SQRT2; // C
        m.coeffs[1][5] = INV_SQRT2; // Rs
        m
    }

    /// 7.1 → 5.1 downmix.
    ///
    /// Channel order input:  `[L, R, C, LFE, Ls, Rs, Lss, Rss]`
    /// Channel order output: `[L, R, C, LFE, Ls, Rs]`
    ///
    /// Side surround (Lss/Rss) are folded into rear surround at −3 dB.
    #[must_use]
    pub fn surround71_to_51() -> Self {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let mut m = Self {
            input_channels: 8,
            output_channels: 6,
            coeffs: vec![vec![0.0_f32; 8]; 6],
        };
        // Pass-through: L R C LFE
        for ch in 0..4 {
            m.coeffs[ch][ch] = 1.0;
        }
        // Ls_out = Ls_in + Lss * 1/√2
        m.coeffs[4][4] = 1.0;
        m.coeffs[4][6] = INV_SQRT2;
        // Rs_out = Rs_in + Rss * 1/√2
        m.coeffs[5][5] = 1.0;
        m.coeffs[5][7] = INV_SQRT2;
        m
    }

    /// 5.1 → 7.1 upmix.
    ///
    /// Channel order input:  `[L, R, C, LFE, Ls, Rs]`
    /// Channel order output: `[L, R, C, LFE, Ls, Rs, Lss, Rss]`
    ///
    /// Rear surrounds are derived from Ls/Rs at −3 dB.
    #[must_use]
    pub fn surround51_to_71() -> Self {
        const INV_SQRT2: f32 = std::f32::consts::FRAC_1_SQRT_2;
        let mut m = Self {
            input_channels: 6,
            output_channels: 8,
            coeffs: vec![vec![0.0_f32; 6]; 8],
        };
        // Pass-through: L R C LFE Ls Rs
        for ch in 0..6 {
            m.coeffs[ch][ch] = 1.0;
        }
        // Lss ← Ls * 1/√2
        m.coeffs[6][4] = INV_SQRT2;
        // Rss ← Rs * 1/√2
        m.coeffs[7][5] = INV_SQRT2;
        m
    }

    /// Per-channel gain adjustment matrix (identity with per-channel scaling).
    ///
    /// `gains` must have exactly `channels` entries (one per channel).
    ///
    /// # Errors
    ///
    /// Returns an error if `gains` is empty.
    pub fn per_channel_gain(gains: &[f32]) -> AudioResult<Self> {
        let n = gains.len();
        if n == 0 {
            return Err(AudioError::InvalidParameter(
                "gains slice must not be empty".into(),
            ));
        }
        let mut m = Self {
            input_channels: n,
            output_channels: n,
            coeffs: vec![vec![0.0_f32; n]; n],
        };
        for (ch, &g) in gains.iter().enumerate() {
            m.coeffs[ch][ch] = g;
        }
        Ok(m)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChannelMixer
// ─────────────────────────────────────────────────────────────────────────────

/// Channel mixer that applies a [`MixMatrix`] to interleaved or planar audio.
#[derive(Clone, Debug)]
pub struct ChannelMixer {
    matrix: MixMatrix,
}

impl ChannelMixer {
    /// Create a new channel mixer with the given matrix.
    #[must_use]
    pub fn new(matrix: MixMatrix) -> Self {
        Self { matrix }
    }

    /// Replace the mixing matrix at runtime.
    pub fn set_matrix(&mut self, matrix: MixMatrix) {
        self.matrix = matrix;
    }

    /// Return a reference to the current matrix.
    #[must_use]
    pub fn matrix(&self) -> &MixMatrix {
        &self.matrix
    }

    /// Process a buffer of interleaved samples.
    ///
    /// Input must contain an integer multiple of `input_channels` samples.
    /// Returns interleaved output with `output_channels` channels per frame.
    ///
    /// Silently pads the last incomplete frame with zeros if necessary.
    #[must_use]
    pub fn process_interleaved(&self, input: &[f32]) -> Vec<f32> {
        let n_in = self.matrix.input_channels;
        let n_out = self.matrix.output_channels;

        if n_in == 0 || n_out == 0 || input.is_empty() {
            return Vec::new();
        }

        let total_frames = (input.len() + n_in - 1) / n_in;
        let mut output = vec![0.0_f32; total_frames * n_out];

        let mut frame_buf = vec![0.0_f32; n_in];

        for frame_idx in 0..total_frames {
            let in_start = frame_idx * n_in;
            let in_end = (in_start + n_in).min(input.len());

            // Copy available samples; pad the rest with zero.
            let available = in_end - in_start;
            frame_buf[..available].copy_from_slice(&input[in_start..in_end]);
            for v in &mut frame_buf[available..] {
                *v = 0.0;
            }

            let mixed = self.matrix.apply_sample(&frame_buf);
            let out_start = frame_idx * n_out;
            output[out_start..out_start + n_out].copy_from_slice(&mixed);
        }

        output
    }

    /// Process planar audio: a `Vec` of per-channel sample slices.
    ///
    /// All input planes must have the same length.  Returns a `Vec` of output
    /// planes, each with the same number of samples as the input planes.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if the number of input planes
    /// does not match `matrix.input_channels`, or if plane lengths differ.
    pub fn process_planar(&self, input_planes: &[Vec<f32>]) -> AudioResult<Vec<Vec<f32>>> {
        let n_in = self.matrix.input_channels;
        let n_out = self.matrix.output_channels;

        if input_planes.len() != n_in {
            return Err(AudioError::InvalidParameter(format!(
                "expected {n_in} input planes, got {}",
                input_planes.len()
            )));
        }

        let n_samples = if input_planes.is_empty() {
            0
        } else {
            let first_len = input_planes[0].len();
            for (i, plane) in input_planes.iter().enumerate().skip(1) {
                if plane.len() != first_len {
                    return Err(AudioError::InvalidParameter(format!(
                        "plane 0 has {} samples but plane {i} has {}",
                        first_len,
                        plane.len()
                    )));
                }
            }
            first_len
        };

        let mut output_planes: Vec<Vec<f32>> = vec![vec![0.0_f32; n_samples]; n_out];

        let mut sample_buf = vec![0.0_f32; n_in];
        for s in 0..n_samples {
            for (ch, plane) in input_planes.iter().enumerate() {
                sample_buf[ch] = plane[s];
            }
            let mixed = self.matrix.apply_sample(&sample_buf);
            for (out_ch, &val) in mixed.iter().enumerate() {
                output_planes[out_ch][s] = val;
            }
        }

        Ok(output_planes)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a constant interleaved buffer: `value` repeated for every sample
    /// across `frames` frames with `channels` channels.
    fn const_interleaved(channels: usize, frames: usize, value: f32) -> Vec<f32> {
        vec![value; channels * frames]
    }

    // ── MixMatrix::new ────────────────────────────────────────────────────────

    #[test]
    fn test_matrix_new_valid() {
        let m = MixMatrix::new(2, 2).expect("valid");
        assert_eq!(m.input_channels, 2);
        assert_eq!(m.output_channels, 2);
    }

    #[test]
    fn test_matrix_new_zero_input_fails() {
        assert!(MixMatrix::new(0, 2).is_err());
    }

    #[test]
    fn test_matrix_new_zero_output_fails() {
        assert!(MixMatrix::new(2, 0).is_err());
    }

    // ── MixMatrix::set / get ──────────────────────────────────────────────────

    #[test]
    fn test_matrix_set_and_get() {
        let mut m = MixMatrix::new(3, 2).expect("valid");
        m.set(0, 1, 0.75).expect("in-bounds");
        assert!((m.get(0, 1) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_set_oob_returns_err() {
        let mut m = MixMatrix::new(2, 2).expect("valid");
        assert!(m.set(2, 0, 1.0).is_err()); // out_ch == 2, max is 1
    }

    #[test]
    fn test_matrix_get_oob_returns_zero() {
        let m = MixMatrix::new(2, 2).expect("valid");
        assert_eq!(m.get(99, 0), 0.0);
    }

    // ── Stereo → Mono ─────────────────────────────────────────────────────────

    #[test]
    fn test_stereo_to_mono_average() {
        let mixer = ChannelMixer::new(MixMatrix::stereo_to_mono());
        // L = 1.0, R = 0.0  → mono should be 0.5
        let input = vec![1.0_f32, 0.0_f32];
        let output = mixer.process_interleaved(&input);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 0.5).abs() < 1e-6, "Expected 0.5, got {}", output[0]);
    }

    // ── Mono → Stereo ─────────────────────────────────────────────────────────

    #[test]
    fn test_mono_to_stereo_duplicates() {
        let mixer = ChannelMixer::new(MixMatrix::mono_to_stereo());
        let input = vec![0.6_f32, 0.6_f32]; // 2 mono frames
        let output = mixer.process_interleaved(&input);
        // 2 frames × 2 output channels = 4 samples
        assert_eq!(output.len(), 4);
        for s in output.chunks(2) {
            assert!((s[0] - 0.6).abs() < 1e-6);
            assert!((s[1] - 0.6).abs() < 1e-6);
        }
    }

    // ── 5.1 → Stereo ──────────────────────────────────────────────────────────

    #[test]
    fn test_51_to_stereo_centre_contribution() {
        let mixer = ChannelMixer::new(MixMatrix::surround51_to_stereo());
        // One frame: L=0, R=0, C=1, LFE=0, Ls=0, Rs=0
        let input = vec![0.0_f32, 0.0, 1.0, 0.0, 0.0, 0.0];
        let output = mixer.process_interleaved(&input);
        assert_eq!(output.len(), 2);
        // Both L and R should get C/√2
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((output[0] - expected).abs() < 1e-5, "L={}", output[0]);
        assert!((output[1] - expected).abs() < 1e-5, "R={}", output[1]);
    }

    // ── 7.1 → 5.1 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_71_to_51_passthrough_channels() {
        let mixer = ChannelMixer::new(MixMatrix::surround71_to_51());
        // One frame: L R C LFE all at 1.0, sides silent
        let input = vec![1.0_f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let output = mixer.process_interleaved(&input);
        assert_eq!(output.len(), 6);
        for &v in &output[0..4] {
            assert!((v - 1.0).abs() < 1e-6, "pass-through channel should be 1.0, got {v}");
        }
    }

    // ── 5.1 → 7.1 ─────────────────────────────────────────────────────────────

    #[test]
    fn test_51_to_71_side_derived_from_surround() {
        let mixer = ChannelMixer::new(MixMatrix::surround51_to_71());
        // One frame: all channels 0 except Ls=1, Rs=1
        let input = vec![0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0];
        let output = mixer.process_interleaved(&input);
        assert_eq!(output.len(), 8);
        // Lss (ch 6) = Ls * 1/√2, Rss (ch 7) = Rs * 1/√2
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((output[6] - expected).abs() < 1e-5, "Lss={}", output[6]);
        assert!((output[7] - expected).abs() < 1e-5, "Rss={}", output[7]);
    }

    // ── Per-channel gain ──────────────────────────────────────────────────────

    #[test]
    fn test_per_channel_gain_scales_independently() {
        let gains = vec![2.0_f32, 0.5];
        let mixer = ChannelMixer::new(MixMatrix::per_channel_gain(&gains).expect("valid"));
        let input = vec![1.0_f32, 1.0]; // one stereo frame
        let output = mixer.process_interleaved(&input);
        assert_eq!(output.len(), 2);
        assert!((output[0] - 2.0).abs() < 1e-6, "L scaled: {}", output[0]);
        assert!((output[1] - 0.5).abs() < 1e-6, "R scaled: {}", output[1]);
    }

    #[test]
    fn test_per_channel_gain_empty_fails() {
        assert!(MixMatrix::per_channel_gain(&[]).is_err());
    }

    // ── Planar processing ─────────────────────────────────────────────────────

    #[test]
    fn test_planar_stereo_to_mono() {
        let matrix = MixMatrix::stereo_to_mono();
        let mixer = ChannelMixer::new(matrix);
        let planes = vec![vec![1.0_f32, 0.8], vec![0.0_f32, 0.4]];
        let out = mixer.process_planar(&planes).expect("ok");
        assert_eq!(out.len(), 1); // 1 output plane
        assert_eq!(out[0].len(), 2); // 2 samples
        assert!((out[0][0] - 0.5).abs() < 1e-6);
        assert!((out[0][1] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_planar_wrong_plane_count_fails() {
        let matrix = MixMatrix::stereo_to_mono(); // expects 2 in
        let mixer = ChannelMixer::new(matrix);
        let planes = vec![vec![1.0_f32]]; // only 1 plane
        assert!(mixer.process_planar(&planes).is_err());
    }

    #[test]
    fn test_planar_mismatched_lengths_fails() {
        let matrix = MixMatrix::stereo_to_mono();
        let mixer = ChannelMixer::new(matrix);
        let planes = vec![vec![1.0_f32, 2.0], vec![1.0_f32]]; // different lengths
        assert!(mixer.process_planar(&planes).is_err());
    }

    // ── Silence / edge cases ──────────────────────────────────────────────────

    #[test]
    fn test_process_interleaved_empty_input() {
        let mixer = ChannelMixer::new(MixMatrix::stereo_to_mono());
        let output = mixer.process_interleaved(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_stereo_to_51_output_channel_count() {
        let mixer = ChannelMixer::new(MixMatrix::stereo_to_51());
        let input = const_interleaved(2, 8, 0.5);
        let output = mixer.process_interleaved(&input);
        // 8 stereo frames → 8 × 6 output samples
        assert_eq!(output.len(), 48);
    }

    #[test]
    fn test_matrix_apply_sample_length() {
        let m = MixMatrix::stereo_to_51();
        let out = m.apply_sample(&[0.5_f32, 0.5]);
        assert_eq!(out.len(), 6);
    }
}
