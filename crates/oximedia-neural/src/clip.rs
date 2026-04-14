//! Gradient clipping utilities.
//!
//! Provides [`clip_gradients`] (L2 global norm clipping) and
//! [`clip_gradients_by_value`] (element-wise clipping).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::clip::{clip_gradients, clip_gradients_by_value};
//!
//! let mut grads = vec![3.0_f32, 4.0]; // norm = 5.0
//! clip_gradients(&mut grads, 1.0);   // rescale to norm 1
//! assert!((grads[0] - 0.6).abs() < 1e-5);
//! assert!((grads[1] - 0.8).abs() < 1e-5);
//!
//! let mut grads2 = vec![-5.0_f32, 2.0, 10.0];
//! clip_gradients_by_value(&mut grads2, 3.0);
//! assert_eq!(grads2, vec![-3.0, 2.0, 3.0]);
//! ```

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Clip gradients by global L2 norm.
///
/// If the L2 norm of `grads` exceeds `max_norm`, all gradients are scaled down
/// so that the new norm equals `max_norm`:
///
/// ```text
/// grads *= max_norm / max(norm, max_norm)
/// ```
///
/// No-op when `grads` is empty or `max_norm <= 0`.
pub fn clip_gradients(grads: &mut Vec<f32>, max_norm: f32) {
    if grads.is_empty() || max_norm <= 0.0 {
        return;
    }
    let norm_sq: f32 = grads.iter().map(|&g| g * g).sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grads.iter_mut() {
            *g *= scale;
        }
    }
}

/// Clip each gradient element to the range `[-clip_value, clip_value]`.
///
/// No-op when `clip_value <= 0`.
pub fn clip_gradients_by_value(grads: &mut Vec<f32>, clip_value: f32) {
    if clip_value <= 0.0 {
        return;
    }
    for g in grads.iter_mut() {
        *g = g.clamp(-clip_value, clip_value);
    }
}

/// Return the L2 norm of a gradient slice without modifying it.
#[must_use]
pub fn grad_norm(grads: &[f32]) -> f32 {
    grads.iter().map(|&g| g * g).sum::<f32>().sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_gradients_scales_down() {
        let mut grads = vec![3.0_f32, 4.0]; // norm = 5.0
        clip_gradients(&mut grads, 1.0);
        // After clipping: grads *= 1.0 / 5.0 = 0.2
        assert!((grads[0] - 0.6).abs() < 1e-5, "grads[0]={}", grads[0]);
        assert!((grads[1] - 0.8).abs() < 1e-5, "grads[1]={}", grads[1]);
    }

    #[test]
    fn test_clip_gradients_no_op_if_below() {
        let original = vec![0.1_f32, 0.2];
        let mut grads = original.clone();
        clip_gradients(&mut grads, 10.0); // norm ≈ 0.22 < 10
        assert_eq!(grads, original, "norm already below max_norm → no change");
    }

    #[test]
    fn test_clip_gradients_exact_threshold() {
        let mut grads = vec![3.0_f32, 4.0]; // norm = 5.0
        clip_gradients(&mut grads, 5.0); // max_norm == actual norm → no scale
        assert!((grads[0] - 3.0).abs() < 1e-5);
        assert!((grads[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_gradients_empty() {
        let mut grads: Vec<f32> = Vec::new();
        clip_gradients(&mut grads, 1.0); // should not panic
        assert!(grads.is_empty());
    }

    #[test]
    fn test_clip_gradients_zero_max_norm_is_noop() {
        let original = vec![1.0_f32, 2.0];
        let mut grads = original.clone();
        clip_gradients(&mut grads, 0.0);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_gradients_preserves_norm_direction() {
        let mut grads = vec![6.0_f32, 8.0]; // norm = 10.0
        clip_gradients(&mut grads, 5.0); // target norm = 5.0
        let new_norm = grad_norm(&grads);
        assert!((new_norm - 5.0).abs() < 1e-4, "new_norm={new_norm}");
        // Direction preserved: grads should still be proportional to [6, 8]
        assert!((grads[0] / grads[1] - 6.0 / 8.0).abs() < 1e-5);
    }

    #[test]
    fn test_clip_by_value_clamps() {
        let mut grads = vec![-5.0_f32, 2.0, 10.0];
        clip_gradients_by_value(&mut grads, 3.0);
        assert_eq!(grads, vec![-3.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clip_by_value_no_op_within_range() {
        let original = vec![1.0_f32, -1.0, 0.5];
        let mut grads = original.clone();
        clip_gradients_by_value(&mut grads, 5.0);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_clip_by_value_zero_clip_is_noop() {
        let original = vec![100.0_f32, -200.0];
        let mut grads = original.clone();
        clip_gradients_by_value(&mut grads, 0.0);
        assert_eq!(grads, original);
    }

    #[test]
    fn test_grad_norm_3_4_5() {
        let grads = vec![3.0_f32, 4.0];
        assert!((grad_norm(&grads) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_grad_norm_empty() {
        assert!((grad_norm(&[])).abs() < 1e-10);
    }
}
