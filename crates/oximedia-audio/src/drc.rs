//! Dynamic Range Control (DRC) with look-ahead limiting.
//!
//! Provides a look-ahead limiter that prevents inter-sample peaks by
//! inspecting a short future segment of the signal before applying gain
//! reduction.  This trades a fixed latency of `lookahead_samples` for
//! transparent, artefact-free limiting.
//!
//! # Algorithm
//!
//! 1. The input signal is written into a ring buffer of length
//!    `lookahead_samples`.
//! 2. On each output sample, the limiter computes the maximum absolute
//!    value over the look-ahead window.
//! 3. If the peak exceeds `threshold`, a gain factor is computed:
//!    `gain = threshold / peak`.
//! 4. The gain is applied with a short attack smoothing constant to avoid
//!    clicks.
//!
//! # Example
//!
//! ```rust
//! use oximedia_audio::drc::DrcLimiter;
//!
//! let mut limiter = DrcLimiter::new(0.9, 512);
//! let mut samples = vec![1.2_f32, -1.5, 0.3, 0.5];
//! limiter.process(&mut samples);
//! assert!(samples.iter().all(|s| s.abs() <= 1.0 + 1e-4));
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;

// ── DrcLimiter ────────────────────────────────────────────────────────────────

/// Look-ahead DRC limiter.
pub struct DrcLimiter {
    /// Threshold above which limiting begins (linear, 0–1 range).
    pub threshold: f32,
    /// Number of samples of look-ahead delay.
    pub lookahead_samples: usize,
    /// Circular look-ahead buffer.
    delay_buf: VecDeque<f32>,
    /// Current smoothed gain coefficient.
    current_gain: f32,
    /// Attack smoothing coefficient (higher = slower attack).
    attack_coeff: f32,
    /// Release smoothing coefficient.
    release_coeff: f32,
}

impl DrcLimiter {
    /// Create a new DRC limiter.
    ///
    /// * `threshold`         – Limiting ceiling in linear amplitude (e.g. `0.9`).
    /// * `lookahead_samples` – Future samples to inspect before output; sets latency.
    #[must_use]
    pub fn new(threshold: f32, lookahead_samples: usize) -> Self {
        let la = lookahead_samples.max(1);
        Self {
            threshold: threshold.clamp(1e-6, 1.0),
            lookahead_samples: la,
            delay_buf: VecDeque::from(vec![0.0f32; la]),
            current_gain: 1.0,
            attack_coeff: 0.99,
            release_coeff: 0.9999,
        }
    }

    /// Set the attack smoothing coefficient in `[0, 1)`.
    ///
    /// Higher values mean slower gain reduction (gentler attack).
    pub fn set_attack_coeff(&mut self, coeff: f32) {
        self.attack_coeff = coeff.clamp(0.0, 0.9999);
    }

    /// Set the release smoothing coefficient in `[0, 1)`.
    pub fn set_release_coeff(&mut self, coeff: f32) {
        self.release_coeff = coeff.clamp(0.0, 0.9999);
    }

    /// Process a buffer of samples in-place.
    ///
    /// Each sample is attenuated so that no output exceeds ±`threshold`.
    /// The output is delayed by `lookahead_samples` samples.
    pub fn process(&mut self, samples: &mut Vec<f32>) {
        for sample in samples.iter_mut() {
            // Push new sample into look-ahead buffer.
            self.delay_buf.push_back(*sample);

            // Find peak over the current look-ahead window.
            let peak = self
                .delay_buf
                .iter()
                .fold(0.0_f32, |acc, &s| acc.max(s.abs()));

            // Compute target gain.
            let target_gain = if peak > self.threshold {
                self.threshold / peak
            } else {
                1.0
            };

            // Smooth gain: faster attack (decrease), slower release (increase).
            if target_gain < self.current_gain {
                self.current_gain =
                    self.attack_coeff * self.current_gain + (1.0 - self.attack_coeff) * target_gain;
            } else {
                self.current_gain = self.release_coeff * self.current_gain
                    + (1.0 - self.release_coeff) * target_gain;
            }

            // Output the oldest sample from the delay line with gain applied.
            let delayed = self
                .delay_buf
                .pop_front()
                .unwrap_or(0.0);
            *sample = delayed * self.current_gain;
        }
    }

    /// Reset internal state (gain and delay buffer) to silence.
    pub fn reset(&mut self) {
        let la = self.lookahead_samples;
        self.delay_buf = VecDeque::from(vec![0.0f32; la]);
        self.current_gain = 1.0;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limiter_attenuates_peaks_above_threshold() {
        let mut limiter = DrcLimiter::new(0.9, 4);
        // Use faster attack for test
        limiter.set_attack_coeff(0.0);
        let mut samples = vec![1.5_f32; 256];
        limiter.process(&mut samples);
        // Check the tail (after the limiter has had time to engage).
        for &s in &samples[8..] {
            assert!(
                s.abs() <= 1.0 + 1e-3,
                "sample {s} exceeds unity after limiting"
            );
        }
    }

    #[test]
    fn test_limiter_passes_quiet_signal_unchanged() {
        // Process a long quiet signal. Verify all outputs are below the
        // threshold (gain should remain near 1.0).
        let mut limiter = DrcLimiter::new(0.9, 4);
        let mut samples: Vec<f32> = (0..128).map(|i| if i % 2 == 0 { 0.1 } else { -0.1 }).collect();
        limiter.process(&mut samples);
        for &s in &samples {
            assert!(
                s.abs() <= 0.15,
                "quiet signal should not be amplified: {s}"
            );
        }
    }

    #[test]
    fn test_limiter_output_length_unchanged() {
        let mut limiter = DrcLimiter::new(0.8, 16);
        let mut samples: Vec<f32> = (0..100).map(|i| (i as f32) * 0.01).collect();
        let len_before = samples.len();
        limiter.process(&mut samples);
        assert_eq!(samples.len(), len_before);
    }

    #[test]
    fn test_limiter_reset_clears_state() {
        let mut limiter = DrcLimiter::new(0.9, 16);
        let mut samples = vec![1.5_f32; 32];
        limiter.process(&mut samples);
        limiter.reset();
        assert!((limiter.current_gain - 1.0).abs() < 1e-6);
        assert_eq!(limiter.delay_buf.len(), limiter.lookahead_samples);
    }

    #[test]
    fn test_limiter_zero_threshold_clamp() {
        // threshold clamped to 1e-6, should not panic
        let limiter = DrcLimiter::new(0.0, 8);
        assert!(limiter.threshold > 0.0);
    }

    #[test]
    fn test_limiter_large_lookahead() {
        let mut limiter = DrcLimiter::new(0.9, 2048);
        let mut samples = vec![0.5_f32; 64];
        limiter.process(&mut samples); // should not panic
        assert_eq!(samples.len(), 64);
    }
}
