//! Training utilities: optimizers (SGD, Adam), learning rate schedulers,
//! gradient clipping, and a simple training loop.
//!
//! All RNG uses a pure-Rust LCG (no external crates).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::training::{SgdOptimizer, Optimizer, TrainingConfig, train_step};
//! use oximedia_neural::tensor::Tensor;
//!
//! // Build a small parameter set (weights for a 2→1 linear layer)
//! let mut weights = vec![0.5_f32, -0.3];
//! let grads    = vec![0.1_f32,  0.2];
//!
//! let mut opt  = SgdOptimizer::new(0.01, 0.0);
//! opt.step(&mut weights, &grads);
//! // weights are updated: w -= lr * g
//! assert!((weights[0] - 0.499).abs() < 1e-4);
//! ```

use crate::error::NeuralError;

// ─────────────────────────────────────────────────────────────────────────────
// Optimizer trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for parameter optimizers.
pub trait Optimizer: Send {
    /// Apply one gradient-descent step.
    ///
    /// * `params` – flat slice of trainable parameters (updated in-place).
    /// * `grads`  – gradient for each parameter.
    ///
    /// # Errors
    ///
    /// Returns [`NeuralError::InvalidShape`] when `params` and `grads` have
    /// different lengths.
    fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), NeuralError>;

    /// Zero internal gradient state (e.g. momentum buffers).
    fn zero_grad(&mut self);

    /// Current learning rate.
    fn lr(&self) -> f32;

    /// Set learning rate (used by schedulers).
    fn set_lr(&mut self, lr: f32);
}

// ─────────────────────────────────────────────────────────────────────────────
// SGD
// ─────────────────────────────────────────────────────────────────────────────

/// Stochastic Gradient Descent with optional momentum and L2 weight decay.
///
/// Update rule (with momentum `β` and weight-decay `λ`):
///
/// ```text
/// v_t = β * v_{t-1} + (g_t + λ * θ_t)
/// θ_t = θ_{t-1} - lr * v_t
/// ```
pub struct SgdOptimizer {
    /// Learning rate.
    pub learning_rate: f32,
    /// Momentum coefficient (0 = no momentum).
    pub momentum: f32,
    /// L2 weight-decay coefficient.
    pub weight_decay: f32,
    /// Velocity buffer for momentum.
    velocity: Vec<f32>,
}

impl SgdOptimizer {
    /// Create a new SGD optimizer.
    #[must_use]
    pub fn new(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            velocity: Vec::new(),
        }
    }

    /// Create with weight decay.
    #[must_use]
    pub fn with_weight_decay(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for SgdOptimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), NeuralError> {
        if params.len() != grads.len() {
            return Err(NeuralError::InvalidShape(format!(
                "SgdOptimizer::step: params.len()={} != grads.len()={}",
                params.len(),
                grads.len()
            )));
        }

        // Ensure velocity buffer is big enough
        if self.velocity.len() < params.len() {
            self.velocity.resize(params.len(), 0.0);
        }

        let lr = self.learning_rate;
        let beta = self.momentum;
        let decay = self.weight_decay;

        for i in 0..params.len() {
            let g = grads[i] + decay * params[i];
            self.velocity[i] = beta * self.velocity[i] + g;
            params[i] -= lr * self.velocity[i];
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for v in &mut self.velocity {
            *v = 0.0;
        }
    }

    fn lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam
// ─────────────────────────────────────────────────────────────────────────────

/// Adam (Adaptive Moment Estimation) optimizer.
///
/// Update rule:
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t²
/// m̂_t = m_t / (1 - β1^t)
/// v̂_t = v_t / (1 - β2^t)
/// θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
/// ```
pub struct AdamOptimizer {
    /// Learning rate.
    pub learning_rate: f32,
    /// First-moment decay coefficient (default 0.9).
    pub beta1: f32,
    /// Second-moment decay coefficient (default 0.999).
    pub beta2: f32,
    /// Numerical stability epsilon (default 1e-8).
    pub epsilon: f32,
    /// L2 weight-decay coefficient.
    pub weight_decay: f32,
    /// First moment buffer.
    m: Vec<f32>,
    /// Second moment buffer.
    v: Vec<f32>,
    /// Step counter (for bias correction).
    t: u64,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer with default hyperparameters.
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Create with custom beta values.
    #[must_use]
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) -> Result<(), NeuralError> {
        if params.len() != grads.len() {
            return Err(NeuralError::InvalidShape(format!(
                "AdamOptimizer::step: params.len()={} != grads.len()={}",
                params.len(),
                grads.len()
            )));
        }

        if self.m.len() < params.len() {
            self.m.resize(params.len(), 0.0);
            self.v.resize(params.len(), 0.0);
        }

        self.t += 1;
        let t = self.t as f32;
        let lr = self.learning_rate;
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.epsilon;
        let decay = self.weight_decay;

        // Bias-corrected step size
        let alpha = lr * (1.0 - b2.powf(t)).sqrt() / (1.0 - b1.powf(t));

        for i in 0..params.len() {
            let g = grads[i] + decay * params[i];
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g * g;
            params[i] -= alpha * self.m[i] / (self.v[i].sqrt() + eps);
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for m in &mut self.m {
            *m = 0.0;
        }
        for v in &mut self.v {
            *v = 0.0;
        }
        self.t = 0;
    }

    fn lr(&self) -> f32 {
        self.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient clipping
// ─────────────────────────────────────────────────────────────────────────────

/// Clip gradients by global L2 norm.
///
/// When the L2 norm of the gradient vector exceeds `max_norm`, all gradients
/// are scaled down so that the norm equals `max_norm`.
///
/// Returns the original L2 norm (before clipping).
pub fn clip_grad_norm(grads: &mut [f32], max_norm: f32) -> f32 {
    let norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
    norm
}

/// Clip gradients element-wise to `[-clip_value, clip_value]`.
pub fn clip_grad_value(grads: &mut [f32], clip_value: f32) {
    let abs_clip = clip_value.abs();
    for g in grads.iter_mut() {
        *g = g.clamp(-abs_clip, abs_clip);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Learning rate schedulers
// ─────────────────────────────────────────────────────────────────────────────

/// Learning rate scheduler trait.
pub trait LrScheduler: Send {
    /// Compute the learning rate for the given step/epoch.
    fn get_lr(&self, step: u64) -> f32;

    /// Step the scheduler and update the optimizer's learning rate.
    fn step_scheduler(&self, optimizer: &mut dyn Optimizer, step: u64) {
        optimizer.set_lr(self.get_lr(step));
    }
}

/// Step decay: multiply LR by `gamma` every `step_size` epochs.
///
/// `lr(t) = lr_initial * gamma^(floor(t / step_size))`
pub struct StepDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Multiplicative decay factor.
    pub gamma: f32,
    /// Epoch interval between decays.
    pub step_size: u64,
}

impl StepDecayScheduler {
    /// Create a new step decay scheduler.
    #[must_use]
    pub fn new(initial_lr: f32, gamma: f32, step_size: u64) -> Self {
        Self {
            initial_lr,
            gamma,
            step_size: step_size.max(1),
        }
    }

    /// Return the learning rate at `epoch`.
    ///
    /// Alias for [`LrScheduler::get_lr`] with a more descriptive name.
    #[must_use]
    pub fn lr_at(&self, epoch: u64) -> f32 {
        self.get_lr(epoch)
    }
}

impl LrScheduler for StepDecayScheduler {
    fn get_lr(&self, step: u64) -> f32 {
        let decay_steps = step / self.step_size;
        self.initial_lr * self.gamma.powi(decay_steps as i32)
    }
}

/// Cosine annealing: `lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))`
pub struct CosineAnnealingScheduler {
    /// Maximum learning rate.
    pub lr_max: f32,
    /// Minimum learning rate.
    pub lr_min: f32,
    /// Period of one cosine cycle in steps.
    pub t_max: u64,
}

impl CosineAnnealingScheduler {
    /// Create a new cosine annealing scheduler.
    #[must_use]
    pub fn new(lr_max: f32, lr_min: f32, t_max: u64) -> Self {
        Self {
            lr_max,
            lr_min,
            t_max: t_max.max(1),
        }
    }
}

impl LrScheduler for CosineAnnealingScheduler {
    fn get_lr(&self, step: u64) -> f32 {
        let t = (step % self.t_max) as f32;
        let cos = (std::f32::consts::PI * t / self.t_max as f32).cos();
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos)
    }
}

/// Exponential decay: `lr(t) = lr_initial * gamma^t`
pub struct ExponentialDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Decay factor per step.
    pub gamma: f32,
}

impl ExponentialDecayScheduler {
    /// Create a new exponential decay scheduler.
    #[must_use]
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LrScheduler for ExponentialDecayScheduler {
    fn get_lr(&self, step: u64) -> f32 {
        self.initial_lr * self.gamma.powi(step as i32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Training loop helper
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the training loop.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs.
    pub epochs: u64,
    /// Log every N steps (0 = no logging).
    pub log_interval: u64,
    /// Gradient clipping max-norm (0 = disabled).
    pub grad_clip_norm: f32,
    /// Gradient clipping value (0 = disabled).
    pub grad_clip_value: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            log_interval: 100,
            grad_clip_norm: 0.0,
            grad_clip_value: 0.0,
        }
    }
}

/// Result of a single training step.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Loss value for this step.
    pub loss: f32,
    /// Gradient L2 norm before clipping.
    pub grad_norm: f32,
    /// Step counter.
    pub step: u64,
}

/// Perform a single training step: compute loss, clip gradients, update params.
///
/// The `forward_and_grad` closure receives the current parameters and returns
/// `(loss, gradients)`.  Gradients are clipped according to `config` before
/// the optimizer step.
///
/// # Errors
///
/// Propagates errors from `forward_and_grad` or the optimizer.
pub fn train_step<F, E>(
    params: &mut Vec<f32>,
    optimizer: &mut dyn Optimizer,
    config: &TrainingConfig,
    step: u64,
    forward_and_grad: F,
) -> Result<StepResult, E>
where
    F: FnOnce(&[f32]) -> Result<(f32, Vec<f32>), E>,
    E: From<NeuralError>,
{
    let (loss, mut grads) = forward_and_grad(params)?;

    let grad_norm = if config.grad_clip_norm > 0.0 {
        clip_grad_norm(&mut grads, config.grad_clip_norm)
    } else if config.grad_clip_value > 0.0 {
        clip_grad_value(&mut grads, config.grad_clip_value);
        grads.iter().map(|g| g * g).sum::<f32>().sqrt()
    } else {
        grads.iter().map(|g| g * g).sum::<f32>().sqrt()
    };

    optimizer.step(params, &grads).map_err(E::from)?;

    Ok(StepResult {
        loss,
        grad_norm,
        step,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_no_momentum() {
        let mut opt = SgdOptimizer::new(0.1, 0.0);
        let mut params = vec![1.0_f32, 2.0];
        let grads = vec![0.5_f32, -0.5];
        opt.step(&mut params, &grads).expect("step ok");
        assert!((params[0] - 0.95).abs() < 1e-5, "p[0]={}", params[0]);
        assert!((params[1] - 2.05).abs() < 1e-5, "p[1]={}", params[1]);
    }

    #[test]
    fn test_sgd_momentum() {
        let mut opt = SgdOptimizer::new(0.1, 0.9);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32];
        // Step 1: v = 0*0.9 + 1 = 1; θ = 1 - 0.1*1 = 0.9
        opt.step(&mut params, &grads).expect("step ok");
        assert!((params[0] - 0.9).abs() < 1e-5);
        // Step 2: v = 0.9*1 + 1 = 1.9; θ = 0.9 - 0.1*1.9 = 0.71
        opt.step(&mut params, &grads).expect("step ok");
        assert!((params[0] - 0.71).abs() < 1e-5, "p[0]={}", params[0]);
    }

    #[test]
    fn test_sgd_shape_mismatch() {
        let mut opt = SgdOptimizer::new(0.1, 0.0);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32, 2.0];
        let result = opt.step(&mut params, &grads);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_single_step() {
        let mut opt = AdamOptimizer::new(0.001);
        let mut params = vec![0.5_f32, -0.5];
        let grads = vec![0.1_f32, -0.1];
        opt.step(&mut params, &grads).expect("step ok");
        // Params should have changed
        assert!((params[0] - 0.5).abs() > 1e-6);
    }

    #[test]
    fn test_adam_converges_toward_zero() {
        let mut opt = AdamOptimizer::new(0.1);
        let mut params = vec![1.0_f32];
        // Gradient = param (trying to minimize 0.5 * param^2)
        for _ in 0..200 {
            let g = vec![params[0]];
            opt.step(&mut params, &g).expect("step ok");
        }
        assert!(params[0].abs() < 0.05, "param={}", params[0]);
    }

    #[test]
    fn test_adam_shape_mismatch() {
        let mut opt = AdamOptimizer::new(0.01);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32, 2.0];
        assert!(opt.step(&mut params, &grads).is_err());
    }

    #[test]
    fn test_clip_grad_norm_above_threshold() {
        let mut grads = vec![3.0_f32, 4.0]; // L2 norm = 5
        let original_norm = clip_grad_norm(&mut grads, 1.0);
        assert!((original_norm - 5.0).abs() < 1e-4, "norm={original_norm}");
        let new_norm: f32 = grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5, "new_norm={new_norm}");
    }

    #[test]
    fn test_clip_grad_norm_below_threshold() {
        let mut grads = vec![0.3_f32, 0.4]; // L2 norm = 0.5
        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 0.5).abs() < 1e-4);
        // Should be unchanged
        assert!((grads[0] - 0.3).abs() < 1e-5);
        assert!((grads[1] - 0.4).abs() < 1e-5);
    }

    #[test]
    fn test_clip_grad_value() {
        let mut grads = vec![0.5_f32, -0.8, 1.5];
        clip_grad_value(&mut grads, 0.7);
        assert!((grads[0] - 0.5).abs() < 1e-5);
        assert!((grads[1] - (-0.7)).abs() < 1e-5);
        assert!((grads[2] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_step_decay_scheduler() {
        let sched = StepDecayScheduler::new(0.1, 0.5, 10);
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-6);
        assert!((sched.get_lr(10) - 0.05).abs() < 1e-6);
        assert!((sched.get_lr(20) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_scheduler() {
        let sched = CosineAnnealingScheduler::new(0.1, 0.0, 100);
        // At t=0: lr = lr_max = 0.1
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-5);
        // At t=T/2 (t=50): cos(pi*50/100)=cos(pi/2)=0, lr = 0.5*(0.1)*(1+0) = 0.05
        let mid = sched.get_lr(50);
        assert!((mid - 0.05).abs() < 1e-4, "mid={mid}");
        // At t=T (t=100): wraps around (step % t_max = 0), lr = lr_max = 0.1
        let end = sched.get_lr(100);
        assert!((end - 0.1).abs() < 1e-5, "end={end}");
    }

    #[test]
    fn test_exponential_decay_scheduler() {
        let sched = ExponentialDecayScheduler::new(0.1, 0.9);
        assert!((sched.get_lr(0) - 0.1).abs() < 1e-6);
        assert!((sched.get_lr(1) - 0.09).abs() < 1e-6);
        assert!((sched.get_lr(2) - 0.081).abs() < 1e-5);
    }

    #[test]
    fn test_sgd_zero_grad() {
        let mut opt = SgdOptimizer::new(0.1, 0.9);
        let mut params = vec![1.0_f32];
        let grads = vec![1.0_f32];
        opt.step(&mut params, &grads).expect("ok");
        // Velocity should be non-zero now
        assert!(opt.velocity[0].abs() > 0.0);
        opt.zero_grad();
        assert!((opt.velocity[0]).abs() < 1e-10);
    }

    #[test]
    fn test_train_step_converges() {
        let config = TrainingConfig {
            epochs: 1,
            log_interval: 0,
            grad_clip_norm: 5.0,
            grad_clip_value: 0.0,
        };
        // Minimize f(w) = w^2; gradient = 2w
        let mut params = vec![1.0_f32];
        let mut opt = SgdOptimizer::new(0.1, 0.0);

        for step in 0..50u64 {
            let result = train_step::<_, NeuralError>(&mut params, &mut opt, &config, step, |p| {
                Ok((p[0] * p[0], vec![2.0 * p[0]]))
            });
            assert!(result.is_ok());
        }
        assert!(params[0].abs() < 0.01, "param={}", params[0]);
    }

    #[test]
    fn test_scheduler_updates_optimizer_lr() {
        let mut opt = SgdOptimizer::new(0.1, 0.0);
        let sched = StepDecayScheduler::new(0.1, 0.5, 10);
        sched.step_scheduler(&mut opt, 10);
        assert!((opt.lr() - 0.05).abs() < 1e-6);
    }
}
