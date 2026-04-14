//! Gradient-descent optimizers: SGD and Adam.
//!
//! This module re-exports the full optimizer API from [`crate::training`] and
//! adds a convenience-oriented thin wrapper surface so callers can do:
//!
//! ```rust
//! use oximedia_neural::optimizer::{SgdOptimizer, AdamOptimizer};
//!
//! let mut sgd = SgdOptimizer::new(0.01);
//! let mut params = vec![1.0_f32, 2.0, 3.0];
//! let grads = vec![0.1_f32, 0.2, 0.3];
//! sgd.update(&mut params, &grads).unwrap();
//! assert!(params[0] < 1.0);
//!
//! let mut adam = AdamOptimizer::new(0.001, 0.9, 0.999);
//! let mut p2 = vec![0.5_f32];
//! let g2 = vec![0.1_f32];
//! adam.update(&mut p2, &g2, 1).unwrap();
//! ```

use crate::error::NeuralError;

// ─────────────────────────────────────────────────────────────────────────────
// SGD
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent optimizer.
///
/// Update rule (vanilla SGD, no momentum):
///
/// ```text
/// params[i] -= lr * grads[i]
/// ```
pub struct SgdOptimizer {
    /// Learning rate.
    pub lr: f32,
}

impl SgdOptimizer {
    /// Create a new SGD optimizer with the given learning rate.
    #[must_use]
    pub const fn new(lr: f32) -> Self {
        Self { lr }
    }

    /// Apply one gradient-descent step.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidShape`] when `params` and `grads` have
    /// different lengths.
    pub fn update(&mut self, params: &mut Vec<f32>, grads: &[f32]) -> Result<(), NeuralError> {
        if params.len() != grads.len() {
            return Err(NeuralError::InvalidShape(format!(
                "SgdOptimizer::update: params.len()={} != grads.len()={}",
                params.len(),
                grads.len()
            )));
        }
        let lr = self.lr;
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= lr * g;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Update rule:
///
/// ```text
/// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
/// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
/// m̂_t = m_t / (1 - beta1^t)
/// v̂_t = v_t / (1 - beta2^t)
/// params -= lr * m̂_t / (sqrt(v̂_t) + epsilon)
/// ```
pub struct AdamOptimizer {
    /// Learning rate.
    pub lr: f32,
    /// First-moment decay rate (typically 0.9).
    pub beta1: f32,
    /// Second-moment decay rate (typically 0.999).
    pub beta2: f32,
    /// Numerical stability epsilon.
    pub epsilon: f32,
    /// First-moment buffer.
    m: Vec<f32>,
    /// Second-moment buffer.
    v: Vec<f32>,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer.
    #[must_use]
    pub fn new(lr: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon: 1e-8,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    /// Apply one Adam step.
    ///
    /// `t` is the **1-based** time step (used for bias correction).
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidShape`] when `params` and `grads` have
    /// different lengths.
    pub fn update(
        &mut self,
        params: &mut Vec<f32>,
        grads: &[f32],
        t: u64,
    ) -> Result<(), NeuralError> {
        if params.len() != grads.len() {
            return Err(NeuralError::InvalidShape(format!(
                "AdamOptimizer::update: params.len()={} != grads.len()={}",
                params.len(),
                grads.len()
            )));
        }
        let n = params.len();
        // Initialise moment buffers on first call or if the parameter size changed.
        if self.m.len() != n {
            self.m = vec![0.0_f32; n];
            self.v = vec![0.0_f32; n];
        }

        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let lr = self.lr;

        // Bias-correction factors
        let t_f = t as f32;
        let bc1 = 1.0 - b1.powf(t_f);
        let bc2 = 1.0 - b2.powf(t_f);

        for i in 0..n {
            let g = grads[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        Ok(())
    }

    /// Reset moment buffers.
    pub fn zero_grad(&mut self) {
        for v in &mut self.m {
            *v = 0.0;
        }
        for v in &mut self.v {
            *v = 0.0;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_decreases_param() {
        let mut opt = SgdOptimizer::new(0.1);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32];
        opt.update(&mut params, &grads).unwrap();
        assert!((params[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_shape_mismatch() {
        let mut opt = SgdOptimizer::new(0.1);
        let mut params = vec![1.0_f32, 2.0];
        let grads = vec![1.0_f32];
        assert!(opt.update(&mut params, &grads).is_err());
    }

    #[test]
    fn test_adam_decreases_param() {
        let mut opt = AdamOptimizer::new(0.01, 0.9, 0.999);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32];
        opt.update(&mut params, &grads, 1).unwrap();
        assert!(params[0] < 1.0);
    }

    #[test]
    fn test_adam_shape_mismatch() {
        let mut opt = AdamOptimizer::new(0.01, 0.9, 0.999);
        let mut params = vec![1.0_f32, 2.0];
        let grads = vec![1.0_f32];
        assert!(opt.update(&mut params, &grads, 1).is_err());
    }

    #[test]
    fn test_adam_zero_grad() {
        let mut opt = AdamOptimizer::new(0.01, 0.9, 0.999);
        let mut params = vec![1.0_f32];
        let grads = vec![0.5_f32];
        opt.update(&mut params, &grads, 1).unwrap();
        opt.zero_grad();
        // After zero_grad, moment buffers should be reset
        assert!(opt.m.iter().all(|&v| v == 0.0));
        assert!(opt.v.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sgd_multiple_params() {
        let mut opt = SgdOptimizer::new(0.1);
        let mut params = vec![2.0_f32, 3.0, 4.0];
        let grads = vec![1.0_f32, 2.0, 3.0];
        opt.update(&mut params, &grads).unwrap();
        assert!((params[0] - 1.9).abs() < 1e-6);
        assert!((params[1] - 2.8).abs() < 1e-6);
        assert!((params[2] - 3.7).abs() < 1e-6);
    }
}
