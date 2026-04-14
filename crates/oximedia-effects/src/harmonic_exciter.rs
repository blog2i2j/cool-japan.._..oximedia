//! Harmonic exciter effect.
//!
//! A harmonic exciter adds controlled harmonic distortion to a signal,
//! typically enhancing perceived clarity and presence by synthesizing
//! upper harmonics (2nd, 3rd, etc.) from the input signal.
//!
//! # Algorithm
//!
//! This implementation generates the 2nd harmonic by squaring the input
//! after band-pass filtering at the drive frequency range, then blending
//! the harmonic content back with the original signal.
//!
//! # Example
//!
//! ```
//! use oximedia_effects::harmonic_exciter::HarmonicExciter;
//!
//! let mut exciter = HarmonicExciter::new(0.5);
//! let input = vec![0.3_f32, -0.2, 0.5, -0.4, 0.1];
//! let output = exciter.process(&input);
//! assert_eq!(output.len(), input.len());
//! for &s in &output {
//!     assert!(s.is_finite());
//! }
//! ```

#![allow(dead_code, clippy::cast_precision_loss)]

use std::f32::consts::PI;

/// Configuration for the harmonic exciter.
#[derive(Debug, Clone)]
pub struct HarmonicExciterConfig {
    /// Drive amount in `[0.0, 1.0]`. Higher = more harmonic content.
    pub drive: f32,
    /// Mix of harmonic content with dry signal `[0.0, 1.0]`.
    /// `0.0` = all dry, `1.0` = maximum harmonics.
    pub mix: f32,
    /// High-pass filter cutoff for the harmonic generation path (Hz).
    /// Frequencies below this are excluded from harmonic synthesis.
    pub hp_cutoff_hz: f32,
    /// Sample rate (Hz).
    pub sample_rate: f32,
}

impl Default for HarmonicExciterConfig {
    fn default() -> Self {
        Self {
            drive: 0.5,
            mix: 0.25,
            hp_cutoff_hz: 2_000.0,
            sample_rate: 48_000.0,
        }
    }
}

// ---------------------------------------------------------------------------
// One-pole high-pass filter state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct OnePoleHp {
    prev_in: f32,
    prev_out: f32,
    coeff: f32, // pole coefficient
}

impl OnePoleHp {
    fn new(cutoff_hz: f32, sample_rate: f32) -> Self {
        // Bilinear-transform first-order HPF coefficient.
        let rc = 1.0 / (2.0 * PI * cutoff_hz.max(1.0));
        let dt = 1.0 / sample_rate.max(1.0);
        let coeff = rc / (rc + dt);
        Self {
            prev_in: 0.0,
            prev_out: 0.0,
            coeff,
        }
    }

    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        // y[n] = coeff * (y[n-1] + x[n] - x[n-1])
        let y = self.coeff * (self.prev_out + x - self.prev_in);
        self.prev_in = x;
        self.prev_out = y;
        y
    }

    fn reset(&mut self) {
        self.prev_in = 0.0;
        self.prev_out = 0.0;
    }
}

// ---------------------------------------------------------------------------
// HarmonicExciter
// ---------------------------------------------------------------------------

/// Harmonic exciter — adds synthesized 2nd (and mild 3rd) harmonic content.
///
/// The exciter separates the high-frequency portion of the signal (via an
/// internal high-pass filter), applies a non-linear drive, and blends the
/// resulting harmonic content back with the original dry signal.
///
/// # Parameters
///
/// - **drive** `[0.0, 1.0]` — amount of harmonic distortion in the synthesis path.
/// - **mix** `[0.0, 1.0]` — blend of synthesized harmonics into the output.
///
/// Higher `drive` values generate more pronounced even harmonics (2nd, 4th);
/// the transfer function is a soft polynomial saturation so the output remains
/// bounded.
pub struct HarmonicExciter {
    config: HarmonicExciterConfig,
    /// High-pass filter isolating the frequency range for harmonic synthesis.
    hp: OnePoleHp,
    /// Smoothed drive coefficient (parameter smoothing, 10 ms time-constant).
    smooth_drive: f32,
    /// One-pole smoothing coefficient.
    smooth_coeff: f32,
}

impl HarmonicExciter {
    /// Create a new harmonic exciter with `drive` in `[0.0, 1.0]`.
    ///
    /// Uses default settings: `mix = 0.25`, `hp_cutoff = 2000 Hz`,
    /// `sample_rate = 48 000 Hz`.
    #[must_use]
    pub fn new(drive: f32) -> Self {
        let config = HarmonicExciterConfig {
            drive: drive.clamp(0.0, 1.0),
            ..HarmonicExciterConfig::default()
        };
        Self::with_config(config)
    }

    /// Create a harmonic exciter with full configuration.
    #[must_use]
    pub fn with_config(config: HarmonicExciterConfig) -> Self {
        let hp = OnePoleHp::new(config.hp_cutoff_hz, config.sample_rate);
        // 10 ms parameter smoothing
        let smooth_coeff = (-1.0_f32 / (0.010 * config.sample_rate.max(1.0))).exp();
        Self {
            smooth_drive: config.drive,
            hp,
            smooth_coeff,
            config,
        }
    }

    /// Set the drive amount `[0.0, 1.0]`.
    pub fn set_drive(&mut self, drive: f32) {
        self.config.drive = drive.clamp(0.0, 1.0);
    }

    /// Set the harmonic blend mix `[0.0, 1.0]`.
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

    /// Process a single sample.
    ///
    /// Applies parameter smoothing to the drive value.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Smooth drive parameter.
        self.smooth_drive =
            self.smooth_drive * self.smooth_coeff + self.config.drive * (1.0 - self.smooth_coeff);

        // High-pass filter the signal for harmonic generation.
        let hp_signal = self.hp.process(input);

        // Generate 2nd harmonic via polynomial: x^2 - DC offset approximation.
        // Use sign-preserving non-linearity to preserve polarity information:
        //   h2(x) = x * |x| (gives predominantly 2nd-order distortion)
        let driven = hp_signal * (1.0 + self.smooth_drive * 8.0);
        let harmonic = driven * driven.abs(); // 2nd harmonic surrogate
        let harmonic_norm = harmonic.tanh(); // bound to ±1

        // Blend harmonic content with original.
        input + self.config.mix * self.smooth_drive * harmonic_norm
    }

    /// Process a slice of samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Reset internal filter and smoother state.
    pub fn reset(&mut self) {
        self.hp.reset();
        self.smooth_drive = self.config.drive;
    }
}

impl crate::AudioEffect for HarmonicExciter {
    fn process_sample(&mut self, input: f32) -> f32 {
        self.process_sample(input)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_sample_rate(&mut self, sample_rate: f32) {
        self.config.sample_rate = sample_rate;
        self.hp = OnePoleHp::new(self.config.hp_cutoff_hz, sample_rate);
        self.smooth_coeff = (-1.0_f32 / (0.010 * sample_rate.max(1.0))).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq_hz: f32, sr: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    fn rms(buf: &[f32]) -> f32 {
        if buf.is_empty() {
            return 0.0;
        }
        (buf.iter().map(|&s| s * s).sum::<f32>() / buf.len() as f32).sqrt()
    }

    #[test]
    fn test_new_clamps_drive() {
        let exc = HarmonicExciter::new(5.0);
        assert!((exc.drive() - 1.0).abs() < 1e-6);

        let exc2 = HarmonicExciter::new(-0.5);
        assert!((exc2.drive() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_output_length_matches_input() {
        let mut exc = HarmonicExciter::new(0.5);
        let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let output = exc.process(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_all_outputs_finite() {
        let mut exc = HarmonicExciter::new(0.8);
        let sine = make_sine(440.0, 48_000.0, 1024);
        let output = exc.process(&sine);
        for (i, &s) in output.iter().enumerate() {
            assert!(s.is_finite(), "Sample {i} is not finite: {s}");
        }
    }

    #[test]
    fn test_drive_zero_near_identity() {
        // With drive=0 the harmonic contribution is zero → output ≈ input.
        let mut exc = HarmonicExciter::new(0.0);
        let input = vec![0.3_f32, -0.2, 0.5, 0.0, -0.4];
        let output = exc.process(&input);
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert!(
                (inp - out).abs() < 1e-4,
                "drive=0 should be near-identity at sample {i}: in={inp}, out={out}"
            );
        }
    }

    #[test]
    fn test_drive_high_adds_energy() {
        // With high drive, the output should have more energy than the input.
        let mut exc = HarmonicExciter::new(1.0);
        exc.set_mix(1.0);
        // Allow settling time.
        let settle = make_sine(5_000.0, 48_000.0, 2048);
        let _ = exc.process(&settle);

        let input = make_sine(5_000.0, 48_000.0, 1024);
        let output = exc.process(&input);

        assert!(
            rms(&output) > rms(&input) * 0.95,
            "High drive should add or maintain energy: in_rms={:.4}, out_rms={:.4}",
            rms(&input),
            rms(&output)
        );
    }

    #[test]
    fn test_reset_clears_state() {
        let mut exc = HarmonicExciter::new(0.7);
        let loud = vec![0.9_f32; 256];
        let _ = exc.process(&loud);
        exc.reset();
        // After reset, a zero input should produce zero output (no DC residual).
        let zeros = vec![0.0_f32; 64];
        let out = exc.process(&zeros);
        for &s in &out {
            assert!(s.abs() < 1e-5, "After reset, zero in → ~zero out, got {s}");
        }
    }

    #[test]
    fn test_set_sample_rate() {
        use crate::AudioEffect;
        let mut exc = HarmonicExciter::new(0.5);
        exc.set_sample_rate(44_100.0);
        assert!((exc.config.sample_rate - 44_100.0).abs() < 1.0);
    }

    #[test]
    fn test_audio_effect_trait_process_sample() {
        let mut exc = HarmonicExciter::new(0.5);
        let out = exc.process_sample(0.5);
        assert!(out.is_finite());
    }
}
