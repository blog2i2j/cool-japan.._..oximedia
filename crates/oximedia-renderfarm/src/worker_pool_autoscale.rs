// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker-pool auto-scaling with configurable thresholds and cooldown.
//!
//! Provides [`AutoScaleConfig`], [`AutoScaleDecision`], and the free function
//! [`evaluate_autoscale`] that decides whether to scale up, scale down, or
//! leave the pool unchanged.
//!
//! # Design
//!
//! The decision logic is **stateless** — all mutable state (e.g. the last
//! scale timestamp) is owned by the caller.  This makes the function easy to
//! test and to integrate with async coordinators that already own shared state.
//!
//! ## Decision priority
//!
//! 1. **Cooldown**: if fewer than `cooldown_ms` milliseconds have elapsed
//!    since the last scaling action, return [`AutoScaleDecision::NoChange`].
//! 2. **Scale up**: if `queue_depth > scale_up_threshold × active_workers`
//!    *and* `active_workers < max_workers`, return
//!    `AutoScaleDecision::ScaleUp(1)`.
//! 3. **Scale down**: if `queue_depth < scale_down_threshold` *and*
//!    `active_workers > min_workers`, return
//!    `AutoScaleDecision::ScaleDown(1)`.
//! 4. Otherwise: [`AutoScaleDecision::NoChange`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_renderfarm::worker_pool_autoscale::{
//!     AutoScaleConfig, AutoScaleDecision, evaluate_autoscale,
//! };
//!
//! let config = AutoScaleConfig {
//!     min_workers: 1,
//!     max_workers: 8,
//!     scale_up_threshold: 3,
//!     scale_down_threshold: 1,
//!     cooldown_ms: 60_000,
//! };
//!
//! // Queue has 12 items with 3 active workers → 12 > 3×3 → scale up
//! let decision = evaluate_autoscale(12, 3, &config, 0, 120_000);
//! assert_eq!(decision, AutoScaleDecision::ScaleUp(1));
//! ```

// ---------------------------------------------------------------------------
// AutoScaleConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for the worker-pool auto-scaler.
#[derive(Debug, Clone)]
pub struct AutoScaleConfig {
    /// Minimum number of workers to keep in the pool at all times.
    pub min_workers: usize,
    /// Hard ceiling on the number of provisioned workers.
    pub max_workers: usize,
    /// Multiplier: scale up when `queue_depth > scale_up_threshold * active_workers`.
    pub scale_up_threshold: usize,
    /// Absolute depth: scale down when `queue_depth < scale_down_threshold`.
    pub scale_down_threshold: usize,
    /// Minimum milliseconds that must pass between consecutive scale actions.
    pub cooldown_ms: u64,
}

impl Default for AutoScaleConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 16,
            scale_up_threshold: 3,
            scale_down_threshold: 1,
            cooldown_ms: 60_000,
        }
    }
}

// ---------------------------------------------------------------------------
// AutoScaleDecision
// ---------------------------------------------------------------------------

/// A scaling recommendation produced by [`evaluate_autoscale`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutoScaleDecision {
    /// Add `n` workers to the pool.
    ScaleUp(usize),
    /// Remove `n` workers from the pool.
    ScaleDown(usize),
    /// No change warranted at this time.
    NoChange,
}

impl AutoScaleDecision {
    /// Returns `true` when the decision is a scale-up.
    #[must_use]
    pub fn is_scale_up(&self) -> bool {
        matches!(self, Self::ScaleUp(_))
    }

    /// Returns `true` when the decision is a scale-down.
    #[must_use]
    pub fn is_scale_down(&self) -> bool {
        matches!(self, Self::ScaleDown(_))
    }

    /// Returns the absolute number of workers to add or remove, or 0 for
    /// [`NoChange`](Self::NoChange).
    #[must_use]
    pub fn delta(&self) -> usize {
        match self {
            Self::ScaleUp(n) | Self::ScaleDown(n) => *n,
            Self::NoChange => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// evaluate_autoscale
// ---------------------------------------------------------------------------

/// Evaluate the current queue and worker state and return an [`AutoScaleDecision`].
///
/// # Arguments
///
/// * `queue_depth`     – Number of jobs currently waiting to be dispatched.
/// * `active_workers`  – Number of workers currently provisioned.
/// * `config`          – Scaling thresholds and bounds.
/// * `last_scale_ms`   – Timestamp (ms) when the last scaling action occurred.
///   Pass `0` to indicate no prior scaling.
/// * `now_ms`          – Current timestamp in milliseconds.
///
/// # Returns
///
/// An [`AutoScaleDecision`] indicating what action, if any, should be taken.
#[must_use]
pub fn evaluate_autoscale(
    queue_depth: usize,
    active_workers: usize,
    config: &AutoScaleConfig,
    last_scale_ms: u64,
    now_ms: u64,
) -> AutoScaleDecision {
    // 1. Enforce cooldown.
    let elapsed_since_scale = now_ms.saturating_sub(last_scale_ms);
    if elapsed_since_scale < config.cooldown_ms {
        return AutoScaleDecision::NoChange;
    }

    // 2. Scale up: queue is deep relative to current worker count.
    let up_threshold = config.scale_up_threshold.saturating_mul(active_workers);
    if queue_depth > up_threshold && active_workers < config.max_workers {
        return AutoScaleDecision::ScaleUp(1);
    }

    // 3. Scale down: queue is shallow and we have spare workers.
    if queue_depth < config.scale_down_threshold && active_workers > config.min_workers {
        return AutoScaleDecision::ScaleDown(1);
    }

    AutoScaleDecision::NoChange
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AutoScaleConfig {
        AutoScaleConfig {
            min_workers: 1,
            max_workers: 8,
            scale_up_threshold: 3,
            scale_down_threshold: 1,
            cooldown_ms: 60_000,
        }
    }

    // --- AutoScaleDecision helpers ---

    #[test]
    fn decision_is_scale_up() {
        assert!(AutoScaleDecision::ScaleUp(1).is_scale_up());
        assert!(!AutoScaleDecision::ScaleDown(1).is_scale_up());
        assert!(!AutoScaleDecision::NoChange.is_scale_up());
    }

    #[test]
    fn decision_is_scale_down() {
        assert!(AutoScaleDecision::ScaleDown(1).is_scale_down());
        assert!(!AutoScaleDecision::ScaleUp(1).is_scale_down());
        assert!(!AutoScaleDecision::NoChange.is_scale_down());
    }

    #[test]
    fn decision_delta() {
        assert_eq!(AutoScaleDecision::ScaleUp(3).delta(), 3);
        assert_eq!(AutoScaleDecision::ScaleDown(2).delta(), 2);
        assert_eq!(AutoScaleDecision::NoChange.delta(), 0);
    }

    // --- evaluate_autoscale: cooldown ---

    #[test]
    fn cooldown_blocks_scale_up() {
        let config = default_config();
        // last_scale was 30 s ago, cooldown is 60 s → should not scale
        let decision = evaluate_autoscale(100, 2, &config, 90_000, 120_000);
        assert_eq!(decision, AutoScaleDecision::NoChange, "still in cooldown");
    }

    #[test]
    fn cooldown_expired_allows_scale_up() {
        let config = default_config();
        // last_scale_ms = 0, now = 120_000 (2 min) → cooldown elapsed
        // queue_depth (20) > threshold (3) * workers (2) = 6 → scale up
        let decision = evaluate_autoscale(20, 2, &config, 0, 120_000);
        assert_eq!(decision, AutoScaleDecision::ScaleUp(1));
    }

    // --- evaluate_autoscale: scale up ---

    #[test]
    fn scale_up_when_queue_exceeds_threshold() {
        let config = default_config();
        // 3 workers, threshold = 3 → needs queue_depth > 9 to scale up
        let decision = evaluate_autoscale(10, 3, &config, 0, 120_000);
        assert_eq!(decision, AutoScaleDecision::ScaleUp(1));
    }

    #[test]
    fn no_scale_up_at_max_workers() {
        let config = AutoScaleConfig {
            max_workers: 3,
            ..default_config()
        };
        // already at max
        let decision = evaluate_autoscale(1_000, 3, &config, 0, 120_000);
        assert_ne!(decision, AutoScaleDecision::ScaleUp(1));
    }

    #[test]
    fn no_scale_up_below_threshold() {
        let config = default_config();
        // 2 workers, threshold = 3 → needs queue_depth > 6; depth is 5
        let decision = evaluate_autoscale(5, 2, &config, 0, 120_000);
        // 5 <= 6 → no scale-up; depth >= scale_down_threshold → no scale-down
        assert_eq!(decision, AutoScaleDecision::NoChange);
    }

    // --- evaluate_autoscale: scale down ---

    #[test]
    fn scale_down_when_queue_is_shallow() {
        let config = AutoScaleConfig {
            scale_down_threshold: 2,
            ..default_config()
        };
        // queue_depth = 1 < 2, and workers (3) > min (1)
        let decision = evaluate_autoscale(1, 3, &config, 0, 120_000);
        assert_eq!(decision, AutoScaleDecision::ScaleDown(1));
    }

    #[test]
    fn no_scale_down_at_min_workers() {
        let config = AutoScaleConfig {
            scale_down_threshold: 5,
            ..default_config()
        };
        // queue_depth = 0, but workers == min (1) → cannot go lower
        let decision = evaluate_autoscale(0, 1, &config, 0, 120_000);
        assert_eq!(decision, AutoScaleDecision::NoChange);
    }

    #[test]
    fn scale_up_takes_priority_over_scale_down() {
        // Construct a pathological config where both conditions would fire.
        let config = AutoScaleConfig {
            min_workers: 1,
            max_workers: 8,
            scale_up_threshold: 0, // queue_depth > 0 × workers = 0 → always true
            scale_down_threshold: 100, // queue_depth < 100 → always true
            cooldown_ms: 0,
        };
        // scale-up condition fires first.
        let decision = evaluate_autoscale(10, 2, &config, 0, 0);
        assert_eq!(decision, AutoScaleDecision::ScaleUp(1));
    }

    // --- zero_workers edge case ---

    #[test]
    fn zero_active_workers_does_not_panic() {
        let config = default_config();
        // With 0 active workers, up_threshold = 0 → queue_depth (1) > 0, scale up
        let decision = evaluate_autoscale(1, 0, &config, 0, 120_000);
        assert_eq!(decision, AutoScaleDecision::ScaleUp(1));
    }
}
