//! Professional gain staging and dB conversion utilities for the `OxiMedia` mixer.
//!
//! Provides:
//! - Accurate dB ↔ linear amplitude conversions with configurable reference levels.
//! - A [`GainComputer`] that applies dynamic-range gain reduction with soft-knee,
//!   ballistic attack/release smoothing, and make-up gain.
//! - A [`LoudnessNormalizer`] for target-loudness matching (EBU R128 / ITU-R BS.1770-4
//!   compliant integrated loudness measurement using a K-weighted RMS approximation).
//! - A [`GainStage`] component that chains input trim, gain computer output, make-up
//!   gain, and output limiter ceiling into a single processing unit.
//!
//! All processing is sample-accurate and allocation-free after construction.

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// dB ↔ Linear conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a decibel value to linear amplitude.
///
/// Returns `0.0` for any value ≤ `MIN_DB_FLOOR` (−150 dB).
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    const MIN_DB_FLOOR: f64 = -150.0;
    if db <= MIN_DB_FLOOR {
        0.0
    } else {
        10.0_f64.powf(db / 20.0)
    }
}

/// Convert a linear amplitude to decibels.
///
/// Returns `f64::NEG_INFINITY` for zero or negative values.
#[must_use]
pub fn linear_to_db(linear: f64) -> f64 {
    if linear <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * linear.log10()
    }
}

/// Convert a power ratio to decibels (dB = 10 · log₁₀(power)).
#[must_use]
pub fn power_to_db(power: f64) -> f64 {
    if power <= 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * power.log10()
    }
}

/// Convert decibels (power domain) back to a power ratio.
#[must_use]
pub fn db_to_power(db: f64) -> f64 {
    10.0_f64.powf(db / 10.0)
}

/// Compute the gain coefficient for a time-constant `tau` at `sample_rate`.
///
/// Returns the one-pole IIR coefficient α = exp(−1 / (tau_samples)).
#[must_use]
pub fn time_constant_coeff(time_ms: f64, sample_rate: u32) -> f64 {
    if time_ms <= 0.0 || sample_rate == 0 {
        return 0.0;
    }
    let tau_samples = time_ms * 0.001 * f64::from(sample_rate);
    (-1.0 / tau_samples).exp()
}

// ─────────────────────────────────────────────────────────────────────────────
// GainComputerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the gain computer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GainComputerConfig {
    /// Threshold in dBFS above which gain reduction is applied.
    pub threshold_db: f64,
    /// Compression/expansion ratio.  `> 1.0` = compression, `< 1.0` = expansion.
    pub ratio: f64,
    /// Soft-knee half-width in dB.  `0.0` = hard knee.
    pub knee_db: f64,
    /// Attack time constant in milliseconds.
    pub attack_ms: f64,
    /// Release time constant in milliseconds.
    pub release_ms: f64,
    /// Make-up gain added after compression in dB.
    pub makeup_gain_db: f64,
}

impl Default for GainComputerConfig {
    fn default() -> Self {
        Self {
            threshold_db: -18.0,
            ratio: 4.0,
            knee_db: 6.0,
            attack_ms: 5.0,
            release_ms: 80.0,
            makeup_gain_db: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GainComputer
// ─────────────────────────────────────────────────────────────────────────────

/// Sample-accurate gain computer with soft-knee and ballistic smoothing.
///
/// Implements the AES standard gain-computer characteristic:
///
/// ```text
/// yG =  xG                                    (xG < T - W/2)
/// yG =  xG + (1/R − 1)(xG − T + W/2)² / 2W   (T−W/2 ≤ xG ≤ T+W/2)
/// yG =  T + (xG − T) / R                      (xG > T + W/2)
/// ```
#[derive(Debug, Clone)]
pub struct GainComputer {
    config: GainComputerConfig,
    /// Current smoothed gain level in dB.
    gain_db: f64,
    /// Attack coefficient (IIR one-pole).
    attack_coeff: f64,
    /// Release coefficient (IIR one-pole).
    release_coeff: f64,
}

impl GainComputer {
    /// Create a new gain computer.
    #[must_use]
    pub fn new(config: GainComputerConfig, sample_rate: u32) -> Self {
        let attack_coeff = time_constant_coeff(config.attack_ms, sample_rate);
        let release_coeff = time_constant_coeff(config.release_ms, sample_rate);
        Self {
            config,
            gain_db: 0.0,
            attack_coeff,
            release_coeff,
        }
    }

    /// Compute the static (unsmoothed) gain reduction in dB for a given input level in dB.
    #[must_use]
    pub fn static_gain_db(&self, input_db: f64) -> f64 {
        let t = self.config.threshold_db;
        let r = self.config.ratio;
        let k = self.config.knee_db.max(0.0);
        let half_k = k / 2.0;

        if k > 0.0 && (input_db - t).abs() <= half_k {
            // Soft knee region.
            let num = (input_db - t + half_k).powi(2);
            let denom = 2.0 * k;
            (1.0 / r - 1.0) * num / denom
        } else if input_db > t + half_k {
            // Above knee — full compression.
            t + (input_db - t) / r - input_db
        } else {
            // Below threshold — no gain reduction.
            0.0
        }
    }

    /// Process one sample, returning the linear output gain multiplier.
    ///
    /// This applies one-pole ballistic smoothing to the gain reduction.
    #[must_use]
    pub fn process_sample(&mut self, input_linear: f64) -> f64 {
        let input_db = linear_to_db(input_linear.abs());
        let static_gr = self.static_gain_db(input_db);

        // Smooth with attack/release ballistics.
        let coeff = if static_gr < self.gain_db {
            // Gain needs to go down (more reduction) → attack.
            self.attack_coeff
        } else {
            // Gain going back up → release.
            self.release_coeff
        };
        self.gain_db = coeff * self.gain_db + (1.0 - coeff) * static_gr;

        // Apply makeup gain and return as linear multiplier.
        db_to_linear(self.gain_db + self.config.makeup_gain_db)
    }

    /// Process a buffer in-place, multiplying each sample by the computed gain.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            let gain = self.process_sample(f64::from(*sample));
            *sample = (*sample as f64 * gain) as f32;
        }
    }

    /// Current smoothed gain reduction in dB (debug/metering use).
    #[must_use]
    pub fn current_gain_reduction_db(&self) -> f64 {
        self.gain_db
    }

    /// Reset internal state (gain to 0 dB — no reduction).
    pub fn reset(&mut self) {
        self.gain_db = 0.0;
    }

    /// Update attack and release coefficients when sample rate changes.
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.attack_coeff = time_constant_coeff(self.config.attack_ms, sample_rate);
        self.release_coeff = time_constant_coeff(self.config.release_ms, sample_rate);
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &GainComputerConfig {
        &self.config
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoudnessNormalizer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the loudness normalizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoudnessNormalizerConfig {
    /// Target integrated loudness in LUFS (e.g. −23.0 for EBU R128).
    pub target_lufs: f64,
    /// Maximum gain to apply in dB (prevents over-amplification of quiet content).
    pub max_gain_db: f64,
    /// Minimum gain to apply in dB (prevents excessive attenuation).
    pub min_gain_db: f64,
    /// Whether to apply a true-peak limiter ceiling in dBTP.
    pub true_peak_limit_db: Option<f64>,
}

impl Default for LoudnessNormalizerConfig {
    fn default() -> Self {
        Self {
            target_lufs: -23.0,
            max_gain_db: 20.0,
            min_gain_db: -40.0,
            true_peak_limit_db: Some(-1.0),
        }
    }
}

/// EBU R128 / BS.1770-4 loudness normalizer.
///
/// Measures integrated loudness over a sliding window using a simplified
/// K-weighted power measurement (the full pre-filter is approximated with
/// a first-order high-shelf at 1500 Hz and 4 dB boost).
///
/// The normalizer accumulates loudness over time and computes the corrective
/// gain needed to reach the target LUFS.
#[derive(Debug, Clone)]
pub struct LoudnessNormalizer {
    config: LoudnessNormalizerConfig,
    sample_rate: u32,
    /// Accumulated power sum for the current measurement window.
    power_sum: f64,
    /// Number of samples accumulated.
    sample_count: u64,
    /// Current estimated integrated loudness in LUFS.
    integrated_lufs: Option<f64>,
    /// Pre-filter state (first-order high-shelf).
    shelf_state: f64,
}

impl LoudnessNormalizer {
    /// Create a new loudness normalizer.
    #[must_use]
    pub fn new(config: LoudnessNormalizerConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
            power_sum: 0.0,
            sample_count: 0,
            integrated_lufs: None,
            shelf_state: 0.0,
        }
    }

    /// Feed mono samples into the loudness accumulator.
    ///
    /// Call this continuously during playback, then read [`integrated_lufs`].
    pub fn push_samples(&mut self, samples: &[f32]) {
        // Simple K-weighting approximation: high-shelf pre-filter at ~1500 Hz, +4 dB.
        // Coefficient computed for a first-order shelf.
        #[allow(clippy::cast_precision_loss)]
        let fc = 1500.0_f64 / self.sample_rate as f64;
        let g = db_to_linear(4.0); // ~1.585
        let coeff = 1.0 - (2.0 * std::f64::consts::PI * fc).exp().recip();

        for &s in samples {
            let x = f64::from(s);
            // High-shelf: y = g·x + (g−1)·shelf_state (running mean subtracted).
            self.shelf_state = self.shelf_state + coeff * (x - self.shelf_state);
            let weighted = x * g - self.shelf_state * (g - 1.0);
            self.power_sum += weighted * weighted;
            self.sample_count += 1;
        }
    }

    /// Compute and return the current integrated loudness estimate in LUFS.
    ///
    /// Returns `None` if fewer than 400 ms of audio has been analysed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn integrated_lufs(&mut self) -> Option<f64> {
        let min_samples = (0.4 * self.sample_rate as f64) as u64;
        if self.sample_count < min_samples {
            return None;
        }
        let mean_square = self.power_sum / self.sample_count as f64;
        // LUFS = -0.691 + 10·log₁₀(mean_square)
        let lufs = -0.691 + power_to_db(mean_square);
        self.integrated_lufs = Some(lufs);
        Some(lufs)
    }

    /// Compute the gain correction in dB needed to hit the target LUFS.
    ///
    /// Returns `None` if loudness has not yet been estimated.
    #[must_use]
    pub fn gain_correction_db(&mut self) -> Option<f64> {
        let lufs = self.integrated_lufs()?;
        let correction = self.config.target_lufs - lufs;
        Some(
            correction
                .max(self.config.min_gain_db)
                .min(self.config.max_gain_db),
        )
    }

    /// Apply the loudness correction gain to a buffer in-place.
    ///
    /// Does nothing if loudness measurement is not yet available.
    pub fn normalize_buffer(&mut self, buffer: &mut [f32]) {
        if let Some(gain_db) = self.gain_correction_db() {
            let gain_linear = db_to_linear(gain_db) as f32;
            // Apply optional true-peak ceiling.
            let ceiling = self
                .config
                .true_peak_limit_db
                .map(|db| db_to_linear(db) as f32)
                .unwrap_or(f32::MAX);
            for sample in buffer.iter_mut() {
                *sample = (*sample * gain_linear).clamp(-ceiling, ceiling);
            }
        }
    }

    /// Reset all accumulated measurements.
    pub fn reset(&mut self) {
        self.power_sum = 0.0;
        self.sample_count = 0;
        self.integrated_lufs = None;
        self.shelf_state = 0.0;
    }

    /// Return the target LUFS.
    #[must_use]
    pub fn target_lufs(&self) -> f64 {
        self.config.target_lufs
    }

    /// Return the number of samples accumulated.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GainStage
// ─────────────────────────────────────────────────────────────────────────────

/// A complete gain staging chain: input trim → gain computer → output ceiling.
#[derive(Debug, Clone)]
pub struct GainStage {
    /// Input trim gain in dB.
    pub input_trim_db: f64,
    /// Gain computer for dynamic range control.
    computer: GainComputer,
    /// Hard output ceiling in dBFS (brickwall limiter).
    pub output_ceiling_db: f64,
}

impl GainStage {
    /// Create a new gain stage.
    #[must_use]
    pub fn new(
        input_trim_db: f64,
        computer_config: GainComputerConfig,
        output_ceiling_db: f64,
        sample_rate: u32,
    ) -> Self {
        Self {
            input_trim_db,
            computer: GainComputer::new(computer_config, sample_rate),
            output_ceiling_db,
        }
    }

    /// Process a single sample through the full gain stage.
    #[must_use]
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        let trimmed = (sample as f64) * db_to_linear(self.input_trim_db);
        let compressed = trimmed * self.computer.process_sample(trimmed);
        let ceiling = db_to_linear(self.output_ceiling_db);
        compressed.clamp(-ceiling, ceiling) as f32
    }

    /// Process a buffer in-place through the full gain stage.
    pub fn process_buffer(&mut self, buffer: &mut [f32]) {
        for sample in buffer.iter_mut() {
            *sample = self.process_sample(*sample);
        }
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.computer.reset();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── dB / linear helpers ──────────────────────────────────────────────────

    #[test]
    fn test_db_to_linear_zero_db_is_one() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_db_to_linear_6db_is_approx_two() {
        assert!((db_to_linear(6.0206) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_db_to_linear_floor() {
        assert!((db_to_linear(-200.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_to_db_one_is_zero() {
        assert!((linear_to_db(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_linear_to_db_zero_is_neg_inf() {
        assert!(linear_to_db(0.0).is_infinite());
    }

    #[test]
    fn test_roundtrip_db_linear() {
        let orig_db = -12.3;
        let roundtripped = linear_to_db(db_to_linear(orig_db));
        assert!((roundtripped - orig_db).abs() < 1e-9);
    }

    #[test]
    fn test_power_db_roundtrip() {
        let orig = -6.0;
        let roundtripped = power_to_db(db_to_power(orig));
        assert!((roundtripped - orig).abs() < 1e-9);
    }

    // ── time_constant_coeff ──────────────────────────────────────────────────

    #[test]
    fn test_time_constant_zero_ms_returns_zero() {
        assert!((time_constant_coeff(0.0, 48000)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_time_constant_returns_value_in_0_1() {
        let coeff = time_constant_coeff(10.0, 48000);
        assert!(coeff > 0.0 && coeff < 1.0);
    }

    // ── GainComputer ─────────────────────────────────────────────────────────

    #[test]
    fn test_gain_computer_below_threshold_no_reduction() {
        let config = GainComputerConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 0.0,
            ..Default::default()
        };
        let comp = GainComputer::new(config, 48000);
        let gr = comp.static_gain_db(-40.0); // 20 dB below threshold
        assert!(gr.abs() < 1e-6, "should be no gain reduction, got {gr}");
    }

    #[test]
    fn test_gain_computer_above_threshold_reduces_gain() {
        let config = GainComputerConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 0.0,
            ..Default::default()
        };
        let comp = GainComputer::new(config, 48000);
        let gr = comp.static_gain_db(0.0); // 20 dB above threshold
        // Expected: gain change = T + (0 - T)/R - 0 = -20 + 20/4 = -20 + 5 = -15 dB
        assert!((gr - (-15.0)).abs() < 0.1, "gain_change={gr}, expected ~-15");
    }

    #[test]
    fn test_gain_computer_soft_knee_in_region() {
        let config = GainComputerConfig {
            threshold_db: -20.0,
            ratio: 4.0,
            knee_db: 10.0, // ±5 dB knee around threshold
            ..Default::default()
        };
        let comp = GainComputer::new(config, 48000);
        // At threshold exactly — should be non-zero soft knee gain.
        let gr = comp.static_gain_db(-20.0);
        assert!(gr <= 0.0, "soft knee at threshold should be ≤ 0");
    }

    #[test]
    fn test_gain_computer_reset() {
        let config = GainComputerConfig::default();
        let mut comp = GainComputer::new(config, 48000);
        comp.process_sample(0.5);
        comp.reset();
        assert!((comp.current_gain_reduction_db()).abs() < 1e-10);
    }

    #[test]
    fn test_gain_computer_process_buffer() {
        let config = GainComputerConfig {
            threshold_db: 0.0,
            ratio: 100.0,
            knee_db: 0.0,
            makeup_gain_db: 0.0,
            ..Default::default()
        };
        let mut comp = GainComputer::new(config, 48000);
        // Feed silence — should come through unity.
        let mut buf = vec![0.0f32; 64];
        comp.process_buffer(&mut buf);
        for s in &buf {
            assert!(s.abs() < 1e-4, "silence should pass unchanged, got {s}");
        }
    }

    // ── LoudnessNormalizer ────────────────────────────────────────────────────

    #[test]
    fn test_loudness_normalizer_insufficient_samples_returns_none() {
        let config = LoudnessNormalizerConfig::default();
        let mut norm = LoudnessNormalizer::new(config, 48000);
        // Push only 100 samples (far below 400 ms = 19200 samples at 48 kHz).
        let samples = vec![0.1f32; 100];
        norm.push_samples(&samples);
        assert!(norm.integrated_lufs().is_none());
    }

    #[test]
    fn test_loudness_normalizer_sufficient_samples_returns_some() {
        let config = LoudnessNormalizerConfig::default();
        let mut norm = LoudnessNormalizer::new(config, 48000);
        // 0.5 s of audio at 48 kHz = 24000 samples
        let samples = vec![0.1f32; 24000];
        norm.push_samples(&samples);
        assert!(norm.integrated_lufs().is_some());
    }

    #[test]
    fn test_loudness_normalizer_reset_clears_state() {
        let config = LoudnessNormalizerConfig::default();
        let mut norm = LoudnessNormalizer::new(config, 48000);
        let samples = vec![0.1f32; 24000];
        norm.push_samples(&samples);
        norm.reset();
        assert_eq!(norm.sample_count(), 0);
        assert!(norm.integrated_lufs().is_none());
    }

    #[test]
    fn test_loudness_normalizer_gain_correction_direction() {
        // A signal that is louder than target should produce a negative correction.
        let config = LoudnessNormalizerConfig {
            target_lufs: -23.0,
            ..Default::default()
        };
        let mut norm = LoudnessNormalizer::new(config, 48000);
        // Loud signal (near 0 dBFS)
        let samples = vec![0.9f32; 48000];
        norm.push_samples(&samples);
        let correction = norm.gain_correction_db().unwrap();
        assert!(correction < 0.0, "loud signal should get attenuated, correction={correction}");
    }

    #[test]
    fn test_loudness_normalizer_clamped_max_gain() {
        let config = LoudnessNormalizerConfig {
            target_lufs: -23.0,
            max_gain_db: 3.0,
            ..Default::default()
        };
        let mut norm = LoudnessNormalizer::new(config, 48000);
        // Very quiet signal.
        let samples = vec![0.0001f32; 48000];
        norm.push_samples(&samples);
        let correction = norm.gain_correction_db().unwrap();
        assert!(correction <= 3.0, "gain should be clamped to max_gain_db");
    }

    // ── GainStage ─────────────────────────────────────────────────────────────

    #[test]
    fn test_gain_stage_unity_passes_signal() {
        let mut stage = GainStage::new(
            0.0,
            GainComputerConfig {
                threshold_db: 0.0,
                ratio: 1.0, // unity compression ratio = no compression
                ..Default::default()
            },
            0.0, // ceiling at 0 dBFS
            48000,
        );
        let input = 0.5f32;
        let output = stage.process_sample(input);
        assert!(
            (output - input).abs() < 0.05,
            "unity stage should pass signal, got {output}"
        );
    }

    #[test]
    fn test_gain_stage_trim_attenuates() {
        let mut stage = GainStage::new(
            -6.0, // −6 dB trim
            GainComputerConfig {
                ratio: 1.0, // no compression
                threshold_db: 0.0,
                ..Default::default()
            },
            0.0,
            48000,
        );
        let input = 1.0f32;
        let output = stage.process_sample(input);
        // After −6 dB trim, expected ~0.5
        assert!(
            (output - 0.5).abs() < 0.05,
            "−6 dB trim should halve level, got {output}"
        );
    }

    #[test]
    fn test_gain_stage_ceiling_clips() {
        let mut stage = GainStage::new(
            0.0,
            GainComputerConfig {
                ratio: 1.0,
                threshold_db: 0.0,
                ..Default::default()
            },
            -6.0, // ceiling at −6 dBFS ≈ 0.5
            48000,
        );
        let output = stage.process_sample(1.0f32);
        assert!(
            output <= 0.52,
            "output should be clamped to ceiling, got {output}"
        );
    }

    #[test]
    fn test_gain_stage_reset() {
        let mut stage = GainStage::new(0.0, GainComputerConfig::default(), 0.0, 48000);
        stage.process_sample(1.0);
        stage.reset();
        // After reset, gain state should be back to 0.
        assert!(stage.computer.current_gain_reduction_db().abs() < 1e-9);
    }
}
