//! Broadcast lookahead limiter with threshold-based API.
//!
//! A companion to [`crate::dynamics::lookahead_limiter::LookaheadLimiter`] that
//! uses `threshold_db` / `release_ms` terminology (common in broadcast-spec
//! documentation such as EBU R128 and ATSC A/85) and exposes a
//! `(delayed_output, applied_gain_db)` tuple from `process_sample`.
//!
//! ## Algorithm
//! 1. Push the incoming sample into a fixed-length delay line.
//! 2. Compute the gain needed so `|input| ≤ threshold_linear`:
//!    `gain_needed = threshold / |input|.max(ε)`, clamped to ≤ 1.0.
//! 3. Apply instant attack (minimum-tracking):
//!    `current_gain = current_gain.min(gain_needed)`.
//! 4. Apply one-pole release towards unity:
//!    `current_gain += (1.0 − current_gain) × (1.0 − release_coeff)`.
//! 5. Output the *delayed* sample multiplied by `current_gain`.

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::collections::VecDeque;

/// Configuration for the broadcast lookahead limiter.
#[derive(Debug, Clone)]
pub struct BroadcastLimiterConfig {
    /// True-peak ceiling in dBFS (e.g. `-1.0` for EBU R128 headroom).
    pub threshold_db: f32,
    /// Lookahead window in milliseconds (delay introduced to the output).
    pub lookahead_ms: f32,
    /// Gain recovery time in milliseconds.
    pub release_ms: f32,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
}

impl BroadcastLimiterConfig {
    /// EBU R128 / broadcast standard preset:
    /// ceiling = −1.0 dBFS, lookahead = 5 ms, release = 200 ms.
    #[must_use]
    pub fn broadcast_standard(sample_rate: u32) -> Self {
        Self {
            threshold_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 200.0,
            sample_rate,
        }
    }

    /// Streaming / online delivery preset:
    /// ceiling = −2.0 dBFS, lookahead = 3 ms, release = 100 ms.
    #[must_use]
    pub fn streaming(sample_rate: u32) -> Self {
        Self {
            threshold_db: -2.0,
            lookahead_ms: 3.0,
            release_ms: 100.0,
            sample_rate,
        }
    }

    /// Validate configuration parameters.
    ///
    /// Returns `Err` if any parameter is out of its valid range.
    pub fn validate(&self) -> Result<(), String> {
        if self.threshold_db > 0.0 {
            return Err(format!(
                "threshold_db must be ≤ 0.0 dBFS, got {:.2}",
                self.threshold_db
            ));
        }
        if !(0.0..=100.0).contains(&self.lookahead_ms) {
            return Err(format!(
                "lookahead_ms must be in [0, 100], got {:.2}",
                self.lookahead_ms
            ));
        }
        if self.release_ms <= 0.0 {
            return Err(format!(
                "release_ms must be > 0.0, got {:.2}",
                self.release_ms
            ));
        }
        if self.sample_rate == 0 {
            return Err("sample_rate must be > 0".to_string());
        }
        Ok(())
    }
}

/// Broadcast lookahead limiter.
///
/// Introduces a latency of `lookahead_samples` samples (accessible via
/// [`latency_samples`](BroadcastLimiter::latency_samples)).
pub struct BroadcastLimiter {
    config: BroadcastLimiterConfig,
    /// Delay line for input samples.
    delay_buffer: VecDeque<f32>,
    /// Pre-computed gain values (one per lookahead sample).
    gain_buffer: VecDeque<f32>,
    /// Current instantaneous gain (linear, 0.0 … 1.0).
    current_gain: f32,
    /// Number of samples in the lookahead window.
    lookahead_samples: usize,
    /// One-pole release coefficient.
    release_coeff: f32,
    /// Threshold in linear scale.
    threshold_linear: f32,
}

impl BroadcastLimiter {
    /// Create a new broadcast limiter from the given configuration.
    ///
    /// # Panics
    /// Will not panic; all arithmetic is guarded.
    #[must_use]
    pub fn new(config: BroadcastLimiterConfig) -> Self {
        let sr = config.sample_rate as f32;
        let lookahead_samples =
            ((config.lookahead_ms * sr / 1000.0) as usize).max(1);
        let release_samples = config.release_ms * sr / 1000.0;
        let release_coeff = if release_samples > 0.0 {
            (-1.0_f32 / release_samples).exp()
        } else {
            0.0
        };
        let threshold_linear = 10.0_f32.powf(config.threshold_db / 20.0);

        // Pre-fill delay line and gain buffer with unity / silence.
        let delay_buffer = VecDeque::from(vec![0.0_f32; lookahead_samples]);
        let gain_buffer = VecDeque::from(vec![1.0_f32; lookahead_samples]);

        Self {
            config,
            delay_buffer,
            gain_buffer,
            current_gain: 1.0,
            lookahead_samples,
            release_coeff,
            threshold_linear,
        }
    }

    // ── core processing ───────────────────────────────────────────────────────

    /// Process one sample.
    ///
    /// Returns `(delayed_output, applied_gain_db)`.
    /// `applied_gain_db` is ≤ 0 when limiting is active.
    pub fn process_sample(&mut self, input: f32) -> (f32, f32) {
        // 1. Compute required gain for this input sample.
        let abs_in = input.abs().max(1e-10);
        let gain_needed = if abs_in > self.threshold_linear {
            (self.threshold_linear / abs_in).min(1.0)
        } else {
            1.0
        };

        // 2. Instant attack: clamp current gain down to required.
        self.current_gain = self.current_gain.min(gain_needed);

        // 3. Release: smoothly recover toward unity.
        self.current_gain +=
            (1.0 - self.current_gain) * (1.0 - self.release_coeff);
        self.current_gain = self.current_gain.min(1.0);

        // 4. Store current gain into the gain buffer; pop the oldest gain.
        self.gain_buffer.push_back(self.current_gain);
        let output_gain = self.gain_buffer.pop_front().unwrap_or(1.0);

        // 5. Push input into the delay line; pop the oldest (most delayed) sample.
        self.delay_buffer.push_back(input);
        let delayed = self.delay_buffer.pop_front().unwrap_or(0.0);

        // 6. Apply gain to the delayed sample.
        let output = delayed * output_gain;
        let gain_db = 20.0 * output_gain.max(f32::EPSILON).log10();

        (output, gain_db)
    }

    /// Process a mono buffer of samples, returning a new `Vec<f32>`.
    #[must_use]
    pub fn process_buffer(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process_sample(s).0).collect()
    }

    /// Current gain reduction in dB (≤ 0 when limiting, 0 when not limiting).
    #[must_use]
    pub fn gain_reduction_db(&self) -> f32 {
        20.0 * self.current_gain.max(f32::EPSILON).log10()
    }

    /// Number of samples of latency introduced by the lookahead delay.
    #[must_use]
    pub fn latency_samples(&self) -> usize {
        self.lookahead_samples
    }

    /// Reset delay lines and gain state.
    pub fn reset(&mut self) {
        for v in self.delay_buffer.iter_mut() {
            *v = 0.0;
        }
        for v in self.gain_buffer.iter_mut() {
            *v = 1.0;
        }
        self.current_gain = 1.0;
    }

    /// Expose the configuration.
    #[must_use]
    pub fn config(&self) -> &BroadcastLimiterConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    fn make_sine(freq_hz: f32, sr: f32, n: usize) -> Vec<f32> {
        use std::f32::consts::TAU;
        (0..n)
            .map(|i| (i as f32 * TAU * freq_hz / sr).sin())
            .collect()
    }

    // ── config validation ─────────────────────────────────────────────────────

    #[test]
    fn test_broadcast_standard_preset_values() {
        let cfg = BroadcastLimiterConfig::broadcast_standard(SR);
        assert!((cfg.threshold_db - (-1.0)).abs() < 1e-6);
        assert!((cfg.lookahead_ms - 5.0).abs() < 1e-6);
        assert!((cfg.release_ms - 200.0).abs() < 1e-6);
        assert_eq!(cfg.sample_rate, SR);
    }

    #[test]
    fn test_streaming_preset_values() {
        let cfg = BroadcastLimiterConfig::streaming(SR);
        assert!((cfg.threshold_db - (-2.0)).abs() < 1e-6);
        assert!((cfg.lookahead_ms - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_validate_ok_on_valid_config() {
        let cfg = BroadcastLimiterConfig::broadcast_standard(SR);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_positive_threshold() {
        let cfg = BroadcastLimiterConfig {
            threshold_db: 1.0,
            lookahead_ms: 5.0,
            release_ms: 200.0,
            sample_rate: SR,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_zero_sample_rate() {
        let cfg = BroadcastLimiterConfig {
            threshold_db: -1.0,
            lookahead_ms: 5.0,
            release_ms: 200.0,
            sample_rate: 0,
        };
        assert!(cfg.validate().is_err());
    }

    // ── limiter behaviour ─────────────────────────────────────────────────────

    #[test]
    fn test_no_clipping_below_threshold() {
        // A signal at 0.5 linear is well below the −1 dBFS ceiling (≈ 0.891).
        let mut lim = BroadcastLimiter::new(BroadcastLimiterConfig::broadcast_standard(SR));
        let input = make_sine(440.0, SR as f32, 2048);
        let output = lim.process_buffer(&input);
        let threshold_lin = 10.0_f32.powf(-1.0 / 20.0);
        for &s in &output {
            assert!(
                s.abs() <= threshold_lin + 1e-4,
                "Output sample {s} exceeds threshold {threshold_lin}"
            );
        }
    }

    #[test]
    fn test_clips_are_handled() {
        let cfg = BroadcastLimiterConfig::broadcast_standard(SR);
        let threshold_lin = 10.0_f32.powf(cfg.threshold_db / 20.0);
        let mut lim = BroadcastLimiter::new(cfg);
        // Very loud signal — should be limited.
        let input = vec![2.0_f32; 2048];
        let output = lim.process_buffer(&input);
        // After the lookahead settles the limiter should enforce the ceiling.
        // Allow 0.5% relative tolerance for floating-point rounding in gain computation.
        let settle = lim.lookahead_samples;
        let ceiling = threshold_lin * 1.005;
        for (i, &s) in output[settle..].iter().enumerate() {
            assert!(
                s.abs() <= ceiling,
                "Sample {i} should be limited: got {s:.7}, ceiling {ceiling:.7}"
            );
        }
    }

    #[test]
    fn test_gain_reduction_db_negative_when_limiting() {
        let mut lim =
            BroadcastLimiter::new(BroadcastLimiterConfig::broadcast_standard(SR));
        // Feed a loud signal to trigger gain reduction.
        for _ in 0..1024 {
            lim.process_sample(2.0);
        }
        let gr = lim.gain_reduction_db();
        assert!(
            gr < 0.0,
            "gain_reduction_db should be negative when limiting, got {gr}"
        );
    }

    #[test]
    fn test_process_buffer_preserves_length() {
        let mut lim =
            BroadcastLimiter::new(BroadcastLimiterConfig::broadcast_standard(SR));
        let input = vec![0.5_f32; 333];
        let output = lim.process_buffer(&input);
        assert_eq!(output.len(), 333);
    }

    #[test]
    fn test_reset_zeros_state() {
        let mut lim =
            BroadcastLimiter::new(BroadcastLimiterConfig::broadcast_standard(SR));
        for _ in 0..512 {
            lim.process_sample(2.0);
        }
        lim.reset();
        // After reset gain should be back to unity.
        assert!(
            (lim.current_gain - 1.0).abs() < 1e-6,
            "current_gain after reset: {}",
            lim.current_gain
        );
        // Delay buffers should be zero.
        for &s in &lim.delay_buffer {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn test_process_sample_returns_gain_db_tuple() {
        let mut lim =
            BroadcastLimiter::new(BroadcastLimiterConfig::broadcast_standard(SR));
        let (output, gain_db) = lim.process_sample(0.1);
        assert!(output.is_finite());
        assert!(gain_db.is_finite());
        // At low level no limiting → gain ≈ 0 dBFS.
        assert!(
            gain_db >= -0.5,
            "No limiting expected for quiet signal; gain_db={gain_db}"
        );
    }

    #[test]
    fn test_latency_equals_lookahead_samples() {
        let cfg = BroadcastLimiterConfig::broadcast_standard(SR);
        let expected = (cfg.lookahead_ms * SR as f32 / 1000.0) as usize;
        let lim = BroadcastLimiter::new(cfg);
        assert_eq!(lim.latency_samples(), expected.max(1));
    }
}
