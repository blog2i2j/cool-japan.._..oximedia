//! Auto-level processor for maintaining consistent output loudness.
//!
//! Analyses a window of audio to estimate the current signal level and
//! applies a smoothed gain so the output remains near a configurable target.
//! Unlike a traditional compressor, the gain is updated on a coarse time scale
//! (measured in blocks / windows) rather than sample-by-sample, which avoids
//! audible pumping while still correcting slow level drift.
//!
//! # Design
//!
//! 1. **Measurement** – The RMS level is computed over a sliding window of
//!    `window_samples` samples using an exponentially-weighted moving average.
//! 2. **Gain computation** – A target gain `G = target_rms / measured_rms` is
//!    derived.  The gain is clamped to `[min_gain, max_gain]` to prevent
//!    extreme amplification of silence or attenuation of loud passages.
//! 3. **Smoothing** – The gain is smoothed with a first-order IIR whose
//!    time-constant is controlled by `attack_coeff` and `release_coeff`.
//!    Gain *decreases* (signal increasing) use `attack_coeff`; gain *increases*
//!    (signal decreasing) use `release_coeff`.
//! 4. **Lookahead** – An optional lookahead delay (in samples) can be added so
//!    the gain adjustment anticipates sudden level increases.
//!
//! # Quick start
//!
//! ```
//! use oximedia_audio::auto_level::{AutoLevel, AutoLevelConfig};
//!
//! let config = AutoLevelConfig {
//!     target_rms: 0.2,
//!     ..AutoLevelConfig::default()
//! };
//! let mut al = AutoLevel::new(48_000, config);
//!
//! let input: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
//! let output = al.process_block(&input);
//! assert_eq!(output.len(), input.len());
//! ```

#![forbid(unsafe_code)]

use crate::{AudioError, AudioResult};

// ─────────────────────────────────────────────────────────────────────────────
// AutoLevelConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the auto-level processor.
#[derive(Clone, Debug)]
pub struct AutoLevelConfig {
    /// Target RMS level in linear scale (e.g. `0.2` ≈ −14 dBFS).
    pub target_rms: f32,

    /// Minimum gain factor applied to the signal (prevents extreme boost).
    /// Must be in `[0, max_gain]`.
    pub min_gain: f32,

    /// Maximum gain factor applied to the signal (prevents extreme attenuation).
    /// Must be ≥ `min_gain`.
    pub max_gain: f32,

    /// Attack time constant: determines how quickly gain can decrease
    /// when the signal becomes louder.  In seconds.
    pub attack_seconds: f32,

    /// Release time constant: determines how quickly gain can increase
    /// when the signal becomes quieter.  In seconds.
    pub release_seconds: f32,

    /// RMS measurement window in seconds.  A longer window gives a steadier
    /// estimate at the cost of slower response to level changes.
    pub window_seconds: f32,

    /// Lookahead delay in samples.  Zero disables lookahead.
    pub lookahead_samples: usize,
}

impl Default for AutoLevelConfig {
    fn default() -> Self {
        Self {
            target_rms: 0.2,
            min_gain: 0.1,
            max_gain: 10.0,
            attack_seconds: 0.3,
            release_seconds: 1.0,
            window_seconds: 0.4,
            lookahead_samples: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AutoLevel
// ─────────────────────────────────────────────────────────────────────────────

/// Auto-level processor.
///
/// See the [module-level documentation](self) for algorithm details.
#[derive(Clone, Debug)]
pub struct AutoLevel {
    /// Sample rate used to convert time constants to coefficients.
    sample_rate: u32,
    /// Working configuration.
    config: AutoLevelConfig,
    /// Exponentially weighted RMS accumulator (power, not amplitude).
    rms_power: f32,
    /// Smoothing coefficient for the RMS estimator.
    rms_coeff: f32,
    /// Current smoothed gain.
    current_gain: f32,
    /// IIR coefficient when gain is decreasing (attack).
    attack_coeff: f32,
    /// IIR coefficient when gain is increasing (release).
    release_coeff: f32,
    /// Lookahead delay line.
    lookahead: Vec<f32>,
    /// Current write head in the lookahead ring.
    lookahead_write: usize,
}

impl AutoLevel {
    /// Create a new `AutoLevel` processor.
    ///
    /// # Panics
    ///
    /// Panics if `sample_rate` is zero.
    #[must_use]
    pub fn new(sample_rate: u32, config: AutoLevelConfig) -> Self {
        assert!(sample_rate > 0, "sample_rate must be non-zero");

        let sr = sample_rate as f32;

        // RMS window coefficient: α = exp(−1 / (window_samples))
        let window_samples = (config.window_seconds * sr).max(1.0);
        let rms_coeff = (-1.0_f32 / window_samples).exp();

        // Attack/release IIR coefficients.
        let attack_coeff = Self::time_to_coeff(config.attack_seconds, sr);
        let release_coeff = Self::time_to_coeff(config.release_seconds, sr);

        let lookahead = vec![0.0_f32; config.lookahead_samples.max(1)];

        Self {
            sample_rate,
            config: config.clone(),
            rms_power: 0.0,
            rms_coeff,
            current_gain: 1.0,
            attack_coeff,
            release_coeff,
            lookahead,
            lookahead_write: 0,
        }
    }

    /// Convert a time constant in seconds to a first-order IIR coefficient.
    fn time_to_coeff(seconds: f32, sample_rate: f32) -> f32 {
        if seconds <= 0.0 {
            return 0.0; // instantaneous
        }
        (-1.0_f32 / (seconds * sample_rate)).exp()
    }

    /// Apply the auto-level processor to a single sample.
    ///
    /// Updates the internal RMS estimate, computes the target gain, and
    /// returns the output sample with the smoothed gain applied.
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // --- Lookahead delay ---
        let lookahead_len = self.lookahead.len();
        // The sample that exits the delay line is used for analysis.
        let read_pos = (self.lookahead_write + 1) % lookahead_len;
        let delayed_sample = self.lookahead[read_pos];
        self.lookahead[self.lookahead_write] = input;
        self.lookahead_write = (self.lookahead_write + 1) % lookahead_len;

        // Signal to analyse (lookahead enabled: analyse incoming; apply to delayed)
        let analyse = if self.config.lookahead_samples > 0 {
            input
        } else {
            delayed_sample
        };

        // --- RMS estimation (exponentially weighted power) ---
        self.rms_power = self.rms_coeff * self.rms_power + (1.0 - self.rms_coeff) * analyse * analyse;
        let rms = self.rms_power.sqrt().max(1e-10);

        // --- Target gain ---
        let target_gain = (self.config.target_rms / rms)
            .clamp(self.config.min_gain, self.config.max_gain);

        // --- Smooth gain with attack/release ---
        let coeff = if target_gain < self.current_gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.current_gain = coeff * self.current_gain + (1.0 - coeff) * target_gain;

        // Output: apply gain to the delayed (or direct) signal.
        let output_sample = if self.config.lookahead_samples > 0 {
            delayed_sample
        } else {
            analyse
        };

        output_sample * self.current_gain
    }

    /// Process a block of samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process_block(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Process in-place (mutates `samples`).
    pub fn process_inplace(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            *s = self.process_sample(*s);
        }
    }

    /// Return the current smoothed gain value.
    #[must_use]
    pub fn current_gain(&self) -> f32 {
        self.current_gain
    }

    /// Return the current RMS estimate (amplitude, not power).
    #[must_use]
    pub fn current_rms(&self) -> f32 {
        self.rms_power.sqrt()
    }

    /// Reset all state to silence / unity gain.
    pub fn reset(&mut self) {
        self.rms_power = 0.0;
        self.current_gain = 1.0;
        for v in &mut self.lookahead {
            *v = 0.0;
        }
        self.lookahead_write = 0;
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &AutoLevelConfig {
        &self.config
    }

    /// Update the target RMS at runtime without resetting state.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if `target_rms` ≤ 0.
    pub fn set_target_rms(&mut self, target_rms: f32) -> AudioResult<()> {
        if target_rms <= 0.0 {
            return Err(AudioError::InvalidParameter(
                "target_rms must be positive".into(),
            ));
        }
        self.config.target_rms = target_rms;
        Ok(())
    }

    /// Update gain limits at runtime.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if `min_gain > max_gain` or
    /// either is negative.
    pub fn set_gain_limits(&mut self, min_gain: f32, max_gain: f32) -> AudioResult<()> {
        if min_gain < 0.0 || max_gain < 0.0 {
            return Err(AudioError::InvalidParameter(
                "gain limits must be non-negative".into(),
            ));
        }
        if min_gain > max_gain {
            return Err(AudioError::InvalidParameter(
                "min_gain must not exceed max_gain".into(),
            ));
        }
        self.config.min_gain = min_gain;
        self.config.max_gain = max_gain;
        Ok(())
    }

    /// Return the sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Compute the RMS of a slice (utility function).
    #[must_use]
    pub fn rms_of(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
        (sum_sq / samples.len() as f32).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(frequency_hz: f32, amplitude: f32, num_samples: usize, sample_rate: u32) -> Vec<f32> {
        (0..num_samples)
            .map(|i| {
                (i as f32 * frequency_hz * std::f32::consts::TAU / sample_rate as f32).sin()
                    * amplitude
            })
            .collect()
    }

    // ── Construction ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_default_config() {
        let al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert_eq!(al.sample_rate(), 48_000);
        // Initial gain should be 1.0
        assert!((al.current_gain() - 1.0).abs() < 1e-6);
    }

    // ── Output length ─────────────────────────────────────────────────────────

    #[test]
    fn test_process_block_length_preserved() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        let input = vec![0.5_f32; 1024];
        let output = al.process_block(&input);
        assert_eq!(output.len(), 1024);
    }

    // ── Output finiteness ─────────────────────────────────────────────────────

    #[test]
    fn test_output_all_finite() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        let input = sine_wave(440.0, 0.8, 4800, 48_000);
        let output = al.process_block(&input);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    // ── Silence in → output bounded ───────────────────────────────────────────

    #[test]
    fn test_silence_bounded_by_max_gain() {
        let config = AutoLevelConfig {
            max_gain: 5.0,
            ..AutoLevelConfig::default()
        };
        let mut al = AutoLevel::new(48_000, config);
        let input = vec![0.0_f32; 2048];
        let output = al.process_block(&input);
        for &v in &output {
            assert!(
                v.abs() <= 5.0 + 1e-4,
                "output exceeds max_gain: {v}"
            );
        }
    }

    // ── Target RMS convergence (very rough) ───────────────────────────────────

    #[test]
    fn test_level_converges_toward_target() {
        let target = 0.2_f32;
        let config = AutoLevelConfig {
            target_rms: target,
            attack_seconds: 0.05,
            release_seconds: 0.2,
            window_seconds: 0.1,
            min_gain: 0.01,
            max_gain: 20.0,
            lookahead_samples: 0,
        };
        let mut al = AutoLevel::new(48_000, config);
        // Feed a loud signal (amplitude = 0.8, rms ≈ 0.566)
        let loud = sine_wave(440.0, 0.8, 48_000, 48_000);
        let output = al.process_block(&loud);

        // After a full second at 48 kHz the output RMS should be reasonably close to target.
        let tail = &output[output.len() / 2..]; // second half
        let out_rms = AutoLevel::rms_of(tail);
        assert!(
            (out_rms - target).abs() < 0.15,
            "Output RMS {out_rms:.4} not near target {target:.4}"
        );
    }

    // ── set_target_rms error handling ─────────────────────────────────────────

    #[test]
    fn test_set_target_rms_zero_fails() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_target_rms(0.0).is_err());
    }

    #[test]
    fn test_set_target_rms_negative_fails() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_target_rms(-0.1).is_err());
    }

    #[test]
    fn test_set_target_rms_valid() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_target_rms(0.3).is_ok());
        assert!((al.config().target_rms - 0.3).abs() < 1e-6);
    }

    // ── set_gain_limits error handling ────────────────────────────────────────

    #[test]
    fn test_gain_limits_inverted_fails() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_gain_limits(5.0, 2.0).is_err());
    }

    #[test]
    fn test_gain_limits_negative_fails() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_gain_limits(-1.0, 2.0).is_err());
    }

    #[test]
    fn test_gain_limits_valid() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        assert!(al.set_gain_limits(0.5, 4.0).is_ok());
    }

    // ── Reset clears state ────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_state() {
        let mut al = AutoLevel::new(48_000, AutoLevelConfig::default());
        let input = sine_wave(440.0, 0.9, 9600, 48_000);
        let _ = al.process_block(&input);
        al.reset();
        assert!((al.current_gain() - 1.0).abs() < 1e-6);
        assert!((al.current_rms() - 0.0).abs() < 1e-6);
    }

    // ── Process inplace matches process_block ─────────────────────────────────

    #[test]
    fn test_inplace_matches_block() {
        let input = sine_wave(220.0, 0.5, 512, 48_000);

        let mut al1 = AutoLevel::new(48_000, AutoLevelConfig::default());
        let block_out = al1.process_block(&input);

        let mut al2 = AutoLevel::new(48_000, AutoLevelConfig::default());
        let mut inplace = input.clone();
        al2.process_inplace(&mut inplace);

        for (a, b) in block_out.iter().zip(inplace.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    // ── rms_of utility ────────────────────────────────────────────────────────

    #[test]
    fn test_rms_of_empty() {
        assert_eq!(AutoLevel::rms_of(&[]), 0.0);
    }

    #[test]
    fn test_rms_of_constant() {
        let v = vec![1.0_f32; 1000];
        let rms = AutoLevel::rms_of(&v);
        assert!((rms - 1.0).abs() < 1e-6, "rms={rms}");
    }

    // ── Lookahead: output length preserved ───────────────────────────────────

    #[test]
    fn test_lookahead_output_length() {
        let config = AutoLevelConfig {
            lookahead_samples: 64,
            ..AutoLevelConfig::default()
        };
        let mut al = AutoLevel::new(48_000, config);
        let input = sine_wave(440.0, 0.4, 512, 48_000);
        let output = al.process_block(&input);
        assert_eq!(output.len(), 512);
    }

    // ── Gain bounded within limits ────────────────────────────────────────────

    #[test]
    fn test_gain_stays_within_limits() {
        let config = AutoLevelConfig {
            min_gain: 0.5,
            max_gain: 3.0,
            ..AutoLevelConfig::default()
        };
        let mut al = AutoLevel::new(48_000, config);
        // Very loud signal: should not go below min_gain.
        let loud: Vec<f32> = vec![1.0_f32; 4800];
        let _ = al.process_block(&loud);
        // Very quiet: should not exceed max_gain.
        let quiet: Vec<f32> = vec![0.001_f32; 4800];
        let _ = al.process_block(&quiet);
        let g = al.current_gain();
        assert!(g >= 0.5 - 1e-3 && g <= 3.0 + 1e-3, "gain={g} out of limits");
    }
}
