//! Tape saturation effect with configurable drive and soft-clipping transfer function.
//!
//! Tape saturation emulates the warm, soft-clipping characteristic of analog
//! magnetic tape recorders. Unlike hard clipping, tape saturation uses a
//! smooth non-linear transfer function that progressively limits signal peaks
//! while preserving the character of the audio at lower levels.
//!
//! # Algorithm
//!
//! The core transfer function combines hyperbolic tangent (tanh) saturation
//! with a subtle asymmetric 2nd-order term to emulate even-harmonic tape
//! coloration:
//!
//! ```text
//! driven = x * (1 + drive * k)          // amplify into saturation
//! sat    = tanh(driven)                  // primary soft-clip
//! asym   = sat + drive * 0.05 * sat²    // subtle even harmonics
//! output = asym / (1 + drive * 0.05)    // normalize
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_effects::tape_sat::TapeSaturation;
//!
//! let mut sat = TapeSaturation::new(0.6);
//! let input = vec![0.0_f32, 0.5, -0.5, 1.0, -1.0];
//! let output = sat.process(&input);
//! assert_eq!(output.len(), input.len());
//! // Tape saturation should not exceed ±1.0 for large inputs (soft clipping).
//! for &s in &output {
//!     assert!(s.is_finite());
//!     assert!(s.abs() <= 1.0 + 1e-4, "output should be soft-clipped: {s}");
//! }
//! ```

#![allow(dead_code)]

/// Configuration for the tape saturation effect.
#[derive(Debug, Clone)]
pub struct TapeSaturationConfig {
    /// Drive (saturation amount) in `[0.0, 1.0]`.
    ///
    /// - `0.0` = bypass (linear, no saturation)
    /// - `1.0` = maximum tape saturation
    pub drive: f32,
    /// Wet/dry mix `[0.0, 1.0]`.
    ///
    /// - `0.0` = dry signal only
    /// - `1.0` = fully saturated signal
    pub mix: f32,
    /// Asymmetry amount `[0.0, 1.0]`.
    ///
    /// Non-zero values add even harmonics (2nd, 4th) characteristic of real tape.
    pub asymmetry: f32,
    /// Input gain before saturation (linear).
    pub input_gain: f32,
    /// Output gain after saturation (linear, compensates for loudness change).
    pub output_gain: f32,
}

impl Default for TapeSaturationConfig {
    fn default() -> Self {
        Self {
            drive: 0.5,
            mix: 1.0,
            asymmetry: 0.2,
            input_gain: 1.0,
            output_gain: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// TapeSaturation
// ---------------------------------------------------------------------------

/// Tape saturation effect with smooth soft-clipping and even-harmonic coloring.
///
/// # Usage
///
/// ```rust
/// use oximedia_effects::tape_sat::TapeSaturation;
///
/// let mut sat = TapeSaturation::new(0.7);
/// let samples = vec![0.8_f32, -0.9, 0.3, -0.6];
/// let out = sat.process(&samples);
/// assert_eq!(out.len(), samples.len());
/// ```
pub struct TapeSaturation {
    config: TapeSaturationConfig,
    /// Parameter-smoothed drive value to prevent zipper noise.
    smooth_drive: f32,
    /// One-pole smoothing coefficient (~10 ms at 48 kHz).
    smooth_coeff: f32,
    sample_rate: f32,
}

impl TapeSaturation {
    // ── constructors ──────────────────────────────────────────────────────────

    /// Create a tape saturation effect with the given drive amount `[0.0, 1.0]`.
    ///
    /// Uses `mix = 1.0`, `asymmetry = 0.2`, unity gain.
    #[must_use]
    pub fn new(drive: f32) -> Self {
        let config = TapeSaturationConfig {
            drive: drive.clamp(0.0, 1.0),
            ..TapeSaturationConfig::default()
        };
        Self::with_config(config, 48_000.0)
    }

    /// Create a tape saturation effect with full configuration and sample rate.
    #[must_use]
    pub fn with_config(config: TapeSaturationConfig, sample_rate: f32) -> Self {
        let smooth_coeff = (-1.0_f32 / (0.010 * sample_rate.max(1.0))).exp();
        let smooth_drive = config.drive;
        Self {
            config,
            smooth_drive,
            smooth_coeff,
            sample_rate,
        }
    }

    // ── parameter setters ─────────────────────────────────────────────────────

    /// Set the drive amount `[0.0, 1.0]`.
    pub fn set_drive(&mut self, drive: f32) {
        self.config.drive = drive.clamp(0.0, 1.0);
    }

    /// Set the wet/dry mix `[0.0, 1.0]`.
    pub fn set_mix(&mut self, mix: f32) {
        self.config.mix = mix.clamp(0.0, 1.0);
    }

    /// Return the current drive setting.
    #[must_use]
    pub fn drive(&self) -> f32 {
        self.config.drive
    }

    /// Return the current mix setting.
    #[must_use]
    pub fn mix(&self) -> f32 {
        self.config.mix
    }

    // ── core transfer function ────────────────────────────────────────────────

    /// Apply the tape saturation transfer function to a single value.
    ///
    /// This is a pure function with no state side-effects; it is exposed for
    /// testing and use in custom signal chains.
    #[must_use]
    #[inline]
    pub fn saturate(x: f32, drive: f32, asymmetry: f32) -> f32 {
        if drive < f32::EPSILON {
            return x;
        }
        // Scale factor: maps drive [0,1] to a gain of [1, ~9] before tanh.
        let k = 1.0 + drive * 8.0;
        let driven = x * k;
        let sat = driven.tanh(); // primary soft-clip (bounded ±1)

        // Asymmetric 2nd-order term: adds subtle even harmonics.
        // The coefficient is kept small so the signal stays within ±1 post-normalization.
        let asym = sat + asymmetry * drive * 0.15 * sat * sat;

        // Normalize: divide by the maximum possible value of |asym| so output
        // stays in (approximately) ±1.  In practice `|asym|` is bounded by
        // |tanh(k)| + asym*drive*0.15*tanh(k)² ≤ 1 + 0.15 ≈ 1.15.
        let norm = 1.0 + asymmetry * drive * 0.15;
        asym / norm.max(f32::EPSILON)
    }

    // ── sample processing ─────────────────────────────────────────────────────

    /// Process a single sample with parameter smoothing.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Smooth drive parameter.
        self.smooth_drive =
            self.smooth_drive * self.smooth_coeff + self.config.drive * (1.0 - self.smooth_coeff);

        let x = input * self.config.input_gain;
        let sat = Self::saturate(x, self.smooth_drive, self.config.asymmetry);
        let wet = sat * self.config.output_gain;

        // Wet/dry blend.
        let dry = input;
        dry + self.config.mix * (wet - dry)
    }

    /// Process a slice of samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset parameter smoother to the current drive target.
    pub fn reset(&mut self) {
        self.smooth_drive = self.config.drive;
    }

    /// Update sample rate and recompute the smoothing coefficient.
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.smooth_coeff = (-1.0_f32 / (0.010 * sample_rate.max(1.0))).exp();
    }
}

impl crate::AudioEffect for TapeSaturation {
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process_sample(input)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.set_sample_rate(sample_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clamps_drive() {
        let s = TapeSaturation::new(5.0);
        assert!((s.drive() - 1.0).abs() < 1e-6);
        let s2 = TapeSaturation::new(-0.3);
        assert!((s2.drive() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_length_matches_input() {
        let mut sat = TapeSaturation::new(0.5);
        let input = vec![0.3_f32; 128];
        let output = sat.process(&input);
        assert_eq!(output.len(), 128);
    }

    #[test]
    fn test_soft_clip_bounds_output() {
        let mut sat = TapeSaturation::new(1.0);
        // Large amplitudes should be soft-clipped to approximately ±1.
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let output = sat.process(&input);
        for &s in &output {
            assert!(s.is_finite(), "Output is not finite: {s}");
            assert!(s.abs() <= 1.05, "Soft-clip exceeded 1.05: {s}");
        }
    }

    #[test]
    fn test_drive_zero_mix_one_is_identity() {
        let mut sat = TapeSaturation::with_config(
            TapeSaturationConfig {
                drive: 0.0,
                mix: 1.0,
                asymmetry: 0.0,
                input_gain: 1.0,
                output_gain: 1.0,
            },
            48_000.0,
        );
        sat.smooth_drive = 0.0;
        let input = vec![0.3_f32, -0.5, 0.7, 0.1, -0.9];
        let output = sat.process(&input);
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-4,
                "drive=0 should be identity at {i}: in={inp}, out={out}"
            );
        }
    }

    #[test]
    fn test_mix_zero_passes_dry() {
        let mut sat = TapeSaturation::with_config(
            TapeSaturationConfig {
                drive: 1.0,
                mix: 0.0,
                asymmetry: 0.0,
                input_gain: 1.0,
                output_gain: 1.0,
            },
            48_000.0,
        );
        let input = vec![0.4_f32, -0.3, 0.9];
        let output = sat.process(&input);
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-5,
                "mix=0 should pass dry signal at {i}: in={inp}, out={out}"
            );
        }
    }

    #[test]
    fn test_saturate_static_function() {
        // tanh(0) = 0 → output is 0 for any drive.
        assert!((TapeSaturation::saturate(0.0, 0.5, 0.2)).abs() < 1e-6);
        // tanh at high drive should be bounded.
        let out = TapeSaturation::saturate(100.0, 1.0, 0.2);
        assert!(
            out.abs() <= 1.1,
            "saturate(100, 1.0) should be ~1.0, got {out}"
        );
    }

    #[test]
    fn test_all_outputs_finite() {
        let mut sat = TapeSaturation::new(0.8);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let output = sat.process(&input);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "Sample {i} is not finite: {s}");
        }
    }

    #[test]
    fn test_reset_snaps_smoother() {
        let mut sat = TapeSaturation::new(0.7);
        sat.smooth_drive = 0.0;
        sat.reset();
        assert!(
            (sat.smooth_drive - 0.7).abs() < 1e-6,
            "reset should snap smooth_drive to target"
        );
    }

    #[test]
    fn test_set_mix_clamps() {
        let mut sat = TapeSaturation::new(0.5);
        sat.set_mix(5.0);
        assert!((sat.mix() - 1.0).abs() < 1e-6);
        sat.set_mix(-1.0);
        assert!((sat.mix() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_effect_trait() {
        let mut sat = TapeSaturation::new(0.6);
        let out = sat.process_sample(0.5);
        assert!(out.is_finite());
    }

    #[test]
    fn test_set_sample_rate() {
        let mut sat = TapeSaturation::new(0.5);
        sat.set_sample_rate(44_100.0);
        assert!((sat.sample_rate - 44_100.0).abs() < 1.0);
    }
}
