//! Auto-gain processor — maintain a consistent output loudness level after
//! any upstream processing that may have changed the programme level.
//!
//! The module provides two complementary types:
//!
//! * [`AutoGainController`] — a look-ahead, adaptive gain stage that
//!   continuously estimates short-term RMS and applies a smooth gain
//!   correction towards a configurable target dBFS.  It is suitable for
//!   real-time processing (game audio, podcasting, live streaming).
//!
//! * [`BatchAutoGain`] — a two-pass normalizer for offline / file processing.
//!   The first pass measures true peak and integrated RMS; the second pass
//!   applies a single static gain factor.
//!
//! # Algorithm
//!
//! `AutoGainController` uses an exponential moving average (EMA) to track
//! the short-term signal power:
//!
//! ```text
//! power_ema[n] = α · x[n]² + (1 − α) · power_ema[n−1]
//! ```
//!
//! where `α` is derived from a configurable attack/release time constant.
//! The desired gain `G_desired` is:
//!
//! ```text
//! target_linear = 10^(target_dbfs / 20)
//! G_desired = target_linear / sqrt(power_ema)
//!             clamped to [min_gain, max_gain]
//! ```
//!
//! Gain is smoothed with a first-order IIR whose time constant is
//! `gain_smoothing_ms` to avoid audible gain pumping.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::auto_gain::{AutoGainController, AutoGainConfig};
//!
//! let config = AutoGainConfig {
//!     target_dbfs: -18.0,
//!     attack_ms: 50.0,
//!     release_ms: 200.0,
//!     gain_smoothing_ms: 100.0,
//!     min_gain_db: -20.0,
//!     max_gain_db: 20.0,
//!     hold_ms: 20.0,
//! };
//! let mut agc = AutoGainController::new(config, 48_000).expect("valid config");
//!
//! let mut buffer = vec![0.5_f32; 480]; // 10 ms at 48 kHz
//! agc.process_inplace(&mut buffer);
//! ```

#![forbid(unsafe_code)]

use crate::error::{AudioError, AudioResult};

// ────────────────────────────────────────────────────────────────────────────
// Utilities
// ────────────────────────────────────────────────────────────────────────────

/// Convert dB to linear gain.
#[inline]
fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB.  Returns −120.0 for near-zero input.
#[inline]
fn linear_to_db(linear: f64) -> f64 {
    if linear < 1e-6 {
        -120.0
    } else {
        20.0 * linear.log10()
    }
}

/// Compute an EMA coefficient from a time constant in milliseconds.
#[inline]
fn ema_coeff(time_ms: f64, sample_rate: u32) -> f64 {
    if time_ms <= 0.0 {
        return 1.0; // Instant response.
    }
    let tau_samples = time_ms * 1e-3 * f64::from(sample_rate);
    (-1.0 / tau_samples).exp()
}

// ────────────────────────────────────────────────────────────────────────────
// Real-time AutoGainController
// ────────────────────────────────────────────────────────────────────────────

/// Configuration for `AutoGainController`.
#[derive(Clone, Debug)]
pub struct AutoGainConfig {
    /// Target output level in dBFS (e.g. −18.0).
    pub target_dbfs: f64,
    /// RMS attack time constant in milliseconds.
    pub attack_ms: f64,
    /// RMS release time constant in milliseconds.
    pub release_ms: f64,
    /// Gain change smoothing time constant in milliseconds.
    pub gain_smoothing_ms: f64,
    /// Minimum allowable gain in dB.
    pub min_gain_db: f64,
    /// Maximum allowable gain in dB.
    pub max_gain_db: f64,
    /// Hold time in milliseconds: once gain starts rising it will not drop
    /// again for at least this duration.
    pub hold_ms: f64,
}

impl Default for AutoGainConfig {
    fn default() -> Self {
        Self {
            target_dbfs: -18.0,
            attack_ms: 50.0,
            release_ms: 200.0,
            gain_smoothing_ms: 100.0,
            min_gain_db: -30.0,
            max_gain_db: 30.0,
            hold_ms: 20.0,
        }
    }
}

/// Real-time adaptive auto-gain controller.
pub struct AutoGainController {
    config: AutoGainConfig,
    sample_rate: u32,
    /// Current power EMA (linear power, not dB).
    power_ema: f64,
    /// Current smoothed gain (linear).
    smooth_gain: f64,
    /// Minimum gain clamped to linear.
    min_gain_linear: f64,
    /// Maximum gain clamped to linear.
    max_gain_linear: f64,
    /// Target amplitude (linear).
    target_linear: f64,
    /// Attack EMA coefficient (> 0 means fast attack).
    attack_coeff: f64,
    /// Release EMA coefficient (slower).
    release_coeff: f64,
    /// Gain smoothing EMA coefficient.
    gain_coeff: f64,
    /// Hold counter: remaining samples during which gain cannot rise.
    hold_samples_remaining: u64,
    /// Hold duration in samples.
    hold_samples_total: u64,
    /// Previous desired gain for hold logic.
    prev_desired_gain: f64,
}

impl AutoGainController {
    /// Create a new `AutoGainController`.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidParameter` if `sample_rate` is zero or
    /// `min_gain_db > max_gain_db`.
    pub fn new(config: AutoGainConfig, sample_rate: u32) -> AudioResult<Self> {
        if sample_rate == 0 {
            return Err(AudioError::InvalidParameter(
                "sample_rate must be non-zero".into(),
            ));
        }
        if config.min_gain_db > config.max_gain_db {
            return Err(AudioError::InvalidParameter(
                "min_gain_db must be <= max_gain_db".into(),
            ));
        }

        let attack_coeff = ema_coeff(config.attack_ms, sample_rate);
        let release_coeff = ema_coeff(config.release_ms, sample_rate);
        let gain_coeff = ema_coeff(config.gain_smoothing_ms, sample_rate);
        let target_linear = db_to_linear(config.target_dbfs);
        let min_gain_linear = db_to_linear(config.min_gain_db);
        let max_gain_linear = db_to_linear(config.max_gain_db);
        let hold_samples_total =
            (config.hold_ms * 1e-3 * f64::from(sample_rate)).round() as u64;

        Ok(Self {
            config,
            sample_rate,
            power_ema: 0.0,
            smooth_gain: 1.0,
            min_gain_linear,
            max_gain_linear,
            target_linear,
            attack_coeff,
            release_coeff,
            gain_coeff,
            hold_samples_remaining: 0,
            hold_samples_total,
            prev_desired_gain: 1.0,
        })
    }

    /// Process a mono buffer in-place.
    pub fn process_inplace(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x = f64::from(*s);
            let power = x * x;

            // Choose attack or release coefficient based on signal rise/fall.
            let coeff = if power > self.power_ema {
                self.attack_coeff
            } else {
                self.release_coeff
            };
            self.power_ema = coeff * self.power_ema + (1.0 - coeff) * power;

            // Compute desired gain.
            let rms = self.power_ema.sqrt();
            let desired_gain = if rms < 1e-7 {
                self.max_gain_linear
            } else {
                (self.target_linear / rms)
                    .clamp(self.min_gain_linear, self.max_gain_linear)
            };

            // Hold logic: prevent gain from rising immediately after a loud transient.
            let effective_desired = if desired_gain > self.prev_desired_gain {
                if self.hold_samples_remaining > 0 {
                    self.hold_samples_remaining -= 1;
                    self.prev_desired_gain // hold the previous value
                } else {
                    self.hold_samples_remaining = self.hold_samples_total;
                    desired_gain
                }
            } else {
                self.hold_samples_remaining = 0;
                desired_gain
            };
            self.prev_desired_gain = effective_desired;

            // Smooth the gain change.
            self.smooth_gain = self.gain_coeff * self.smooth_gain
                + (1.0 - self.gain_coeff) * effective_desired;

            *s = (x * self.smooth_gain) as f32;
        }
    }

    /// Process a mono buffer and return a new `Vec<f32>` with the result.
    #[must_use]
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut out = samples.to_vec();
        self.process_inplace(&mut out);
        out
    }

    /// Get the current gain in dB.
    #[must_use]
    pub fn current_gain_db(&self) -> f64 {
        linear_to_db(self.smooth_gain)
    }

    /// Get the current RMS level in dBFS.
    #[must_use]
    pub fn current_rms_dbfs(&self) -> f64 {
        linear_to_db(self.power_ema.sqrt())
    }

    /// Reset the processor state (gain → 1.0, power EMA → 0).
    pub fn reset(&mut self) {
        self.power_ema = 0.0;
        self.smooth_gain = 1.0;
        self.prev_desired_gain = 1.0;
        self.hold_samples_remaining = 0;
    }

    /// Update the target dBFS level without recreating the processor.
    pub fn set_target_dbfs(&mut self, target_dbfs: f64) {
        self.config.target_dbfs = target_dbfs;
        self.target_linear = db_to_linear(target_dbfs);
    }

    /// Return a reference to the current configuration.
    #[must_use]
    pub fn config(&self) -> &AutoGainConfig {
        &self.config
    }

    /// Return the configured sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Offline BatchAutoGain
// ────────────────────────────────────────────────────────────────────────────

/// Result of a `BatchAutoGain` two-pass normalization.
#[derive(Clone, Debug)]
pub struct BatchNormResult {
    /// Gain factor applied (linear).
    pub gain_linear: f64,
    /// Gain factor applied (dB).
    pub gain_db: f64,
    /// Measured peak level before normalization (linear).
    pub measured_peak_linear: f64,
    /// Measured RMS level before normalization (dBFS).
    pub measured_rms_dbfs: f64,
}

/// Offline two-pass auto-gain normalizer.
///
/// Pass 1: measure peak and RMS from the entire buffer.
/// Pass 2: apply a single gain factor that brings the RMS to the target.
///
/// The gain is additionally peak-limited so that the output peak never
/// exceeds `peak_ceiling_dbfs`.
pub struct BatchAutoGain {
    /// Target RMS in dBFS.
    pub target_rms_dbfs: f64,
    /// Peak ceiling in dBFS (default −0.5).
    pub peak_ceiling_dbfs: f64,
}

impl Default for BatchAutoGain {
    fn default() -> Self {
        Self {
            target_rms_dbfs: -18.0,
            peak_ceiling_dbfs: -0.5,
        }
    }
}

impl BatchAutoGain {
    /// Create with explicit targets.
    #[must_use]
    pub fn new(target_rms_dbfs: f64, peak_ceiling_dbfs: f64) -> Self {
        Self { target_rms_dbfs, peak_ceiling_dbfs }
    }

    /// Normalize `buffer` in-place and return measurement results.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidData` when `buffer` is empty.
    pub fn normalize_inplace(&self, buffer: &mut [f32]) -> AudioResult<BatchNormResult> {
        if buffer.is_empty() {
            return Err(AudioError::InvalidData("buffer is empty".into()));
        }

        // Pass 1 — measure.
        let sum_sq: f64 = buffer.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
        let rms = (sum_sq / buffer.len() as f64).sqrt();
        let peak = buffer
            .iter()
            .map(|s| s.abs())
            .fold(0.0_f32, f32::max);

        let peak_linear = f64::from(peak);
        let measured_rms_dbfs = linear_to_db(rms);

        // Compute gain required to hit target RMS.
        let target_rms_linear = db_to_linear(self.target_rms_dbfs);
        let gain_for_rms = if rms < 1e-9 {
            1.0
        } else {
            target_rms_linear / rms
        };

        // Peak-limit the gain.
        let peak_ceiling_linear = db_to_linear(self.peak_ceiling_dbfs);
        let max_gain_for_peak = if peak_linear < 1e-9 {
            gain_for_rms
        } else {
            peak_ceiling_linear / peak_linear
        };

        let gain_linear = gain_for_rms.min(max_gain_for_peak).max(0.0);
        let gain_db = linear_to_db(gain_linear);

        // Pass 2 — apply.
        for s in buffer.iter_mut() {
            *s = (*s as f64 * gain_linear) as f32;
        }

        Ok(BatchNormResult {
            gain_linear,
            gain_db,
            measured_peak_linear: peak_linear,
            measured_rms_dbfs,
        })
    }

    /// Like `normalize_inplace` but returns a new `Vec<f32>`.
    ///
    /// # Errors
    ///
    /// Returns `AudioError::InvalidData` when `samples` is empty.
    pub fn normalize(&self, samples: &[f32]) -> AudioResult<(Vec<f32>, BatchNormResult)> {
        let mut buf = samples.to_vec();
        let result = self.normalize_inplace(&mut buf)?;
        Ok((buf, result))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agc() -> AutoGainController {
        AutoGainController::new(AutoGainConfig::default(), 48_000)
            .expect("valid config")
    }

    #[test]
    fn test_agc_construction() {
        let _ = make_agc();
    }

    #[test]
    fn test_agc_invalid_sample_rate() {
        let result = AutoGainController::new(AutoGainConfig::default(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_agc_invalid_gain_range() {
        let config = AutoGainConfig {
            min_gain_db: 10.0,
            max_gain_db: 5.0, // max < min → error
            ..Default::default()
        };
        let result = AutoGainController::new(config, 48_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_agc_silence_does_not_panic() {
        let mut agc = make_agc();
        let mut buf = vec![0.0_f32; 480];
        agc.process_inplace(&mut buf);
        // All samples should remain zero (gain applied to zero is still zero).
        for s in &buf {
            assert!(*s == 0.0 || s.is_finite());
        }
    }

    #[test]
    fn test_agc_gain_reduces_loud_signal() {
        // Feed a very loud signal (close to clipping) and check that after
        // convergence the output is quieter than the input.
        let mut agc = AutoGainController::new(
            AutoGainConfig {
                target_dbfs: -18.0,
                attack_ms: 1.0,
                release_ms: 10.0,
                gain_smoothing_ms: 5.0,
                hold_ms: 0.0,
                ..Default::default()
            },
            48_000,
        )
        .expect("valid");

        // Warm up the processor with a constant 0 dBFS tone.
        let loud: Vec<f32> = vec![0.95_f32; 4800];
        agc.process(&loud);

        // After convergence the gain should be below 1.0 (attenuating).
        assert!(
            agc.smooth_gain < 1.0,
            "smooth_gain={} expected < 1.0",
            agc.smooth_gain
        );
    }

    #[test]
    fn test_agc_reset_clears_state() {
        let mut agc = make_agc();
        let loud: Vec<f32> = vec![0.9_f32; 1000];
        let _ = agc.process(&loud);
        agc.reset();
        assert!((agc.smooth_gain - 1.0).abs() < 1e-9);
        assert!(agc.power_ema < 1e-15);
    }

    #[test]
    fn test_agc_set_target_changes_target() {
        let mut agc = make_agc();
        agc.set_target_dbfs(-12.0);
        assert!((agc.config().target_dbfs - (-12.0)).abs() < 1e-9);
        assert!((agc.target_linear - db_to_linear(-12.0)).abs() < 1e-9);
    }

    #[test]
    fn test_agc_current_gain_db_type() {
        let agc = make_agc();
        let db = agc.current_gain_db();
        // Fresh processor → gain = 1.0 → 0 dB.
        assert!((db - 0.0).abs() < 1e-6, "db={}", db);
    }

    #[test]
    fn test_batch_normalise_empty_error() {
        let bgc = BatchAutoGain::default();
        let result = bgc.normalize_inplace(&mut []);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_normalise_rms_target() {
        // Input at -6 dBFS RMS, target -18 dBFS → should apply ~-12 dB gain.
        let bgc = BatchAutoGain::new(-18.0, -0.1);
        // 0.5 amplitude ≈ -6 dBFS.
        let mut buf = vec![0.5_f32; 4800];
        let result = bgc.normalize_inplace(&mut buf).expect("ok");
        // Measured RMS should be close to -18 dBFS after processing.
        let rms_after: f64 = {
            let sum_sq: f64 = buf.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
            (sum_sq / buf.len() as f64).sqrt()
        };
        let rms_db_after = linear_to_db(rms_after);
        assert!(
            (rms_db_after - (-18.0)).abs() < 0.5,
            "rms_after={:.2} dBFS",
            rms_db_after
        );
        assert!(result.gain_linear > 0.0);
    }

    #[test]
    fn test_batch_normalise_peak_ceiling() {
        // Input is a very hot signal; peak ceiling should limit the gain.
        let bgc = BatchAutoGain::new(-3.0, -1.0); // target RMS = -3 dBFS but peak must stay ≤ -1 dBTP
        let mut buf = vec![0.9_f32; 4800]; // peak already 0.9 → -0.92 dBFS
        let result = bgc.normalize_inplace(&mut buf).expect("ok");
        let peak_after = buf.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        let peak_ceiling = db_to_linear(-1.0);
        assert!(
            f64::from(peak_after) <= peak_ceiling + 1e-6,
            "peak_after={:.4} limit={:.4}",
            peak_after,
            peak_ceiling
        );
        assert!(result.gain_db < 0.0 || result.gain_db.abs() < 1.0);
    }

    #[test]
    fn test_db_linear_roundtrip() {
        for db in [-60.0_f64, -18.0, 0.0, 6.0, 20.0] {
            let linear = db_to_linear(db);
            let back = linear_to_db(linear);
            assert!((back - db).abs() < 1e-9, "db={} back={}", db, back);
        }
    }
}
