//! Learning rate schedulers.
//!
//! Provides [`StepDecayScheduler`] (and re-exports from [`crate::training`]).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::scheduler::StepDecayScheduler;
//!
//! let sched = StepDecayScheduler::new(0.1, 0.5, 10);
//! assert!((sched.lr_at(0) - 0.1).abs() < 1e-6);
//! assert!((sched.lr_at(10) - 0.05).abs() < 1e-6);
//! assert!((sched.lr_at(20) - 0.025).abs() < 1e-6);
//! ```

/// Step-decay learning rate scheduler.
///
/// `lr(epoch) = initial_lr * decay^(floor(epoch / step_size))`
#[derive(Debug, Clone)]
pub struct StepDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Multiplicative decay factor applied every `step_size` epochs.
    pub decay: f32,
    /// Number of epochs per decay step.
    pub step_size: u64,
}

impl StepDecayScheduler {
    /// Create a new step-decay scheduler.
    ///
    /// * `initial_lr` — starting learning rate.
    /// * `decay`      — multiplicative factor (`< 1.0` to decrease LR).
    /// * `step_size`  — apply decay every this many epochs (clamped to 1 minimum).
    #[must_use]
    pub fn new(initial_lr: f32, decay: f32, step_size: u64) -> Self {
        Self {
            initial_lr,
            decay,
            step_size: step_size.max(1),
        }
    }

    /// Return the learning rate at the given epoch.
    ///
    /// `lr = initial_lr * decay^(epoch / step_size)`
    #[must_use]
    pub fn lr_at(&self, epoch: u64) -> f32 {
        let decay_steps = epoch / self.step_size;
        self.initial_lr * self.decay.powi(decay_steps as i32)
    }
}

/// Cosine annealing scheduler.
///
/// Smoothly decreases the LR from `lr_max` to `lr_min` following a cosine curve.
#[derive(Debug, Clone)]
pub struct CosineAnnealingScheduler {
    /// Maximum (initial) learning rate.
    pub lr_max: f32,
    /// Minimum learning rate.
    pub lr_min: f32,
    /// Total number of epochs (half-period of the cosine).
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

    /// Return the learning rate at epoch `t`.
    #[must_use]
    pub fn lr_at(&self, t: u64) -> f32 {
        let t = (t % (2 * self.t_max)) as f32;
        let t_max = self.t_max as f32;
        let cos_val = (std::f32::consts::PI * t / t_max).cos();
        self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + cos_val)
    }
}

/// Exponential decay scheduler.
///
/// `lr(epoch) = initial_lr * gamma^epoch`
#[derive(Debug, Clone)]
pub struct ExponentialDecayScheduler {
    /// Initial learning rate.
    pub initial_lr: f32,
    /// Per-epoch multiplicative factor.
    pub gamma: f32,
}

impl ExponentialDecayScheduler {
    /// Create a new exponential decay scheduler.
    #[must_use]
    pub const fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }

    /// Return the learning rate at the given epoch.
    #[must_use]
    pub fn lr_at(&self, epoch: u64) -> f32 {
        self.initial_lr * self.gamma.powi(epoch as i32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_decay_epoch_zero() {
        let s = StepDecayScheduler::new(0.1, 0.5, 10);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_step_decay_after_one_step() {
        let s = StepDecayScheduler::new(0.1, 0.5, 10);
        assert!((s.lr_at(10) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_step_decay_after_two_steps() {
        let s = StepDecayScheduler::new(0.1, 0.5, 10);
        assert!((s.lr_at(20) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_step_decay_midstep() {
        // epoch 15 is still in the first decay window after epoch 10
        let s = StepDecayScheduler::new(0.1, 0.5, 10);
        assert!((s.lr_at(15) - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_step_decay_step_size_one() {
        // step_size=1 means decay every epoch
        let s = StepDecayScheduler::new(1.0, 0.5, 1);
        assert!((s.lr_at(0) - 1.0).abs() < 1e-6);
        assert!((s.lr_at(1) - 0.5).abs() < 1e-6);
        assert!((s.lr_at(2) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_lr_at_zero() {
        let s = CosineAnnealingScheduler::new(0.1, 0.0, 100);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_lr_at_tmax() {
        let s = CosineAnnealingScheduler::new(0.1, 0.0, 100);
        assert!((s.lr_at(100) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_exponential_decay_epoch_zero() {
        let s = ExponentialDecayScheduler::new(0.1, 0.9);
        assert!((s.lr_at(0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_exponential_decay_decreasing() {
        let s = ExponentialDecayScheduler::new(1.0, 0.9);
        let lr0 = s.lr_at(0);
        let lr1 = s.lr_at(1);
        let lr10 = s.lr_at(10);
        assert!(lr1 < lr0, "LR should decrease each epoch");
        assert!(lr10 < lr1, "LR should continue decreasing");
    }
}
