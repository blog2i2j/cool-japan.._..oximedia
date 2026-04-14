//! Dropout regularization layer.
//!
//! During training, each activation is zeroed with probability `rate` and
//! the remaining activations are scaled by `1 / (1 - rate)` (inverted dropout)
//! so that the expected value is preserved.
//!
//! Randomness uses a pure-Rust LCG — no external dependencies.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::dropout::Dropout;
//!
//! let dropout = Dropout::new(0.5);
//! let input = vec![1.0_f32; 100];
//!
//! // In training mode some values are zeroed
//! let (out_train, _seed) = dropout.forward(&input, true, 42);
//! let zeros = out_train.iter().filter(|&&v| v == 0.0).count();
//! assert!(zeros > 0, "expect some zeroed activations during training");
//!
//! // In inference mode output equals input
//! let (out_infer, _) = dropout.forward(&input, false, 0);
//! assert_eq!(out_infer, input);
//! ```

use crate::error::NeuralError;

// ─────────────────────────────────────────────────────────────────────────────
// LCG helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Advance the LCG and return the next state.
#[inline]
fn lcg_step(state: u64) -> u64 {
    state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Return a value in [0, 1) from an LCG state.
#[inline]
fn lcg_f32(state: u64) -> f32 {
    (state >> 32) as f32 / u32::MAX as f32
}

// ─────────────────────────────────────────────────────────────────────────────
// Dropout
// ─────────────────────────────────────────────────────────────────────────────

/// Dropout regularization layer.
///
/// * `rate` — fraction of units to drop (0.0 = no dropout, 1.0 = drop all).
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Drop probability in [0, 1).
    pub rate: f32,
}

impl Dropout {
    /// Create a new dropout layer with the given drop rate.
    ///
    /// `rate` is clamped to `[0, 1)`.
    #[must_use]
    pub fn new(rate: f32) -> Self {
        Self {
            rate: rate.clamp(0.0, 0.999_999),
        }
    }

    /// Apply dropout to `input`.
    ///
    /// Returns `(output, next_seed)` so callers can chain the seed across
    /// multiple dropout layers or steps.
    ///
    /// * `training` — when `false`, input is returned unchanged.
    /// * `seed` — initial LCG state.
    ///
    /// The output uses **inverted dropout**: surviving activations are scaled
    /// by `1 / (1 - rate)` so the expected value of each unit is preserved.
    #[must_use]
    pub fn forward(&self, input: &[f32], training: bool, seed: u64) -> (Vec<f32>, u64) {
        if !training || self.rate == 0.0 {
            return (input.to_vec(), seed);
        }

        let scale = 1.0 / (1.0 - self.rate);
        let mut state = seed;
        let mut out = Vec::with_capacity(input.len());

        for &x in input {
            state = lcg_step(state);
            let u = lcg_f32(state);
            if u < self.rate {
                out.push(0.0);
            } else {
                out.push(x * scale);
            }
        }

        (out, state)
    }

    /// Apply dropout in-place, returning the updated seed.
    ///
    /// No-op when `training` is `false`.
    pub fn forward_inplace(&self, data: &mut [f32], training: bool, seed: u64) -> u64 {
        if !training || self.rate == 0.0 {
            return seed;
        }

        let scale = 1.0 / (1.0 - self.rate);
        let mut state = seed;

        for x in data.iter_mut() {
            state = lcg_step(state);
            let u = lcg_f32(state);
            if u < self.rate {
                *x = 0.0;
            } else {
                *x *= scale;
            }
        }

        state
    }

    /// Apply dropout and return only the output `Vec` (discards next seed).
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidShape`] if `input` is empty and training
    /// is requested (guard for accidental misuse).
    pub fn apply(&self, input: &[f32], training: bool, seed: u64) -> Result<Vec<f32>, NeuralError> {
        if input.is_empty() && training {
            return Err(NeuralError::InvalidShape(
                "Dropout::apply: empty input".to_string(),
            ));
        }
        let (out, _) = self.forward(input, training, seed);
        Ok(out)
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_returns_input_unchanged() {
        let d = Dropout::new(0.5);
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let (out, _) = d.forward(&input, false, 99);
        assert_eq!(out, input);
    }

    #[test]
    fn test_training_zeroes_some_activations() {
        let d = Dropout::new(0.5);
        let input = vec![1.0_f32; 200];
        let (out, _) = d.forward(&input, true, 42);
        let zeros = out.iter().filter(|&&v| v == 0.0).count();
        // Statistically, ~50% should be zero; allow wide band
        assert!(zeros > 20, "expected many zeros, got {zeros}");
        assert!(zeros < 180, "expected some survivors, got {zeros}");
    }

    #[test]
    fn test_inverted_dropout_preserves_expected_value() {
        let d = Dropout::new(0.5);
        let n = 10_000;
        let input = vec![1.0_f32; n];
        let (out, _) = d.forward(&input, true, 7);
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        // Expected ~1.0 (inverted dropout)
        assert!(
            (mean - 1.0).abs() < 0.1,
            "mean={mean}, expected ~1.0 (inverted dropout)"
        );
    }

    #[test]
    fn test_zero_rate_is_identity() {
        let d = Dropout::new(0.0);
        let input = vec![5.0_f32; 10];
        let (out, _) = d.forward(&input, true, 1);
        assert_eq!(out, input);
    }

    #[test]
    fn test_seed_chaining() {
        let d = Dropout::new(0.3);
        let input = vec![1.0_f32; 50];
        let (out1, seed1) = d.forward(&input, true, 100);
        let (out2, _) = d.forward(&input, true, seed1);
        // Two consecutive passes with different seeds should differ (statistically)
        let identical = out1.iter().zip(out2.iter()).filter(|(a, b)| a == b).count();
        // It's extremely unlikely all 50 are identical with rate=0.3
        assert!(
            identical < 50,
            "two passes with different seeds should not be identical"
        );
    }

    #[test]
    fn test_inplace_same_as_forward() {
        let d = Dropout::new(0.4);
        let input = vec![2.0_f32; 20];
        let (expected, _) = d.forward(&input, true, 55);
        let mut data = input.clone();
        d.forward_inplace(&mut data, true, 55);
        for (a, b) in expected.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_apply_error_on_empty_training() {
        let d = Dropout::new(0.5);
        let result = d.apply(&[], true, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_ok_empty_inference() {
        let d = Dropout::new(0.5);
        let result = d.apply(&[], false, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
