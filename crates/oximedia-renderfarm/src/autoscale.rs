// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Auto-scaling logic for the render-farm worker pool.
//!
//! [`AutoScaler`] evaluates queue depth and worker idle time to decide whether
//! to add or remove workers.  Cooldown periods prevent oscillation.  Min/max
//! bounds are always enforced.
//!
//! Usage:
//! ```rust
//! use oximedia_renderfarm::autoscale::{AutoScaleConfig, AutoScaler, ScaleEvent};
//!
//! let config = AutoScaleConfig::new(1, 8);
//! let mut scaler = AutoScaler::new(config).expect("valid config");
//!
//! // Simulate a busy queue (use a non-zero timestamp so cooldown does not
//! // block the very first evaluation, which uses last_scale_ms = 0)
//! let now_ms: u64 = 120_000; // 2 min after epoch, past the 60 s cooldown
//! let event = scaler.evaluate(10, now_ms);
//! scaler.apply_event(&event, now_ms);
//! assert_eq!(scaler.current_workers(), 2); // started at 1, scaled up by 1
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AutoScaleConfig
// ---------------------------------------------------------------------------

/// Configuration for the auto-scaling policy.
#[derive(Debug, Clone)]
pub struct AutoScaleConfig {
    /// Minimum number of workers to keep alive at all times.
    pub min_workers: u32,
    /// Maximum number of workers the farm may provision.
    pub max_workers: u32,
    /// Add a worker when the pending-job queue depth exceeds this value.
    pub scale_up_queue_threshold: u32,
    /// Remove a worker when it has been idle for at least this many seconds.
    pub scale_down_idle_threshold_secs: u64,
    /// Minimum number of seconds between any two scaling decisions.
    pub cooldown_secs: u64,
}

impl AutoScaleConfig {
    /// Create a configuration with the given bounds and sensible defaults for
    /// the remaining knobs.
    ///
    /// Defaults: scale-up at queue depth > 5, scale-down after 300 s idle,
    /// 60 s cooldown between events.
    pub fn new(min: u32, max: u32) -> Self {
        Self {
            min_workers: min,
            max_workers: max,
            scale_up_queue_threshold: 5,
            scale_down_idle_threshold_secs: 300,
            cooldown_secs: 60,
        }
    }

    /// Validate that the configuration is internally consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.min_workers == 0 {
            return Err("min_workers must be >= 1".to_owned());
        }
        if self.min_workers > self.max_workers {
            return Err(format!(
                "min_workers ({}) must be <= max_workers ({})",
                self.min_workers, self.max_workers
            ));
        }
        Ok(())
    }

    /// Conservative preset: small pool, scale up only when the queue is very deep.
    pub fn conservative() -> Self {
        Self {
            min_workers: 1,
            max_workers: 4,
            scale_up_queue_threshold: 10,
            scale_down_idle_threshold_secs: 600,
            cooldown_secs: 120,
        }
    }

    /// Aggressive preset: larger pool, scale up quickly on modest queue growth.
    pub fn aggressive() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            scale_up_queue_threshold: 2,
            scale_down_idle_threshold_secs: 60,
            cooldown_secs: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// ScaleEvent
// ---------------------------------------------------------------------------

/// A scaling decision produced by [`AutoScaler::evaluate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScaleEvent {
    /// Provision `count` additional workers.
    ScaleUp { count: u32 },
    /// Decommission `count` idle workers.
    ScaleDown { count: u32 },
    /// No change warranted.
    NoChange,
}

// ---------------------------------------------------------------------------
// AutoScaler
// ---------------------------------------------------------------------------

/// Stateful auto-scaler for a render-farm worker pool.
pub struct AutoScaler {
    config: AutoScaleConfig,
    current_workers: u32,
    last_scale_ms: u64,
    /// Maps worker_id → timestamp (ms) when the worker became idle.
    worker_idle_since: HashMap<String, u64>,
}

impl AutoScaler {
    /// Construct an [`AutoScaler`] from a validated config.
    ///
    /// Returns `Err` if the config fails validation.  The initial worker count
    /// is set to `config.min_workers`.
    pub fn new(config: AutoScaleConfig) -> Result<Self, String> {
        config.validate()?;
        let initial = config.min_workers;
        Ok(Self {
            config,
            current_workers: initial,
            last_scale_ms: 0,
            worker_idle_since: HashMap::new(),
        })
    }

    /// Evaluate the current state and return a recommended [`ScaleEvent`].
    ///
    /// Decision logic (in priority order):
    /// 1. If the cooldown has not elapsed since the last scaling event → [`ScaleEvent::NoChange`].
    /// 2. If `queue_depth > scale_up_queue_threshold` and `current < max` → [`ScaleEvent::ScaleUp`] by 1.
    /// 3. If any worker has been idle longer than `scale_down_idle_threshold_secs` and `current > min` → [`ScaleEvent::ScaleDown`] by 1.
    /// 4. Otherwise → [`ScaleEvent::NoChange`].
    pub fn evaluate(&mut self, queue_depth: u32, now_ms: u64) -> ScaleEvent {
        let cooldown_ms = self.config.cooldown_secs * 1000;
        if now_ms.saturating_sub(self.last_scale_ms) < cooldown_ms {
            return ScaleEvent::NoChange;
        }

        // Scale up takes priority over scale down.
        if queue_depth > self.config.scale_up_queue_threshold
            && self.current_workers < self.config.max_workers
        {
            return ScaleEvent::ScaleUp { count: 1 };
        }

        // Scale down if at least one worker has been idle long enough.
        if self.current_workers > self.config.min_workers {
            let threshold_ms = self.config.scale_down_idle_threshold_secs * 1000;
            let has_idle = self
                .worker_idle_since
                .values()
                .any(|&idle_since| now_ms.saturating_sub(idle_since) >= threshold_ms);
            if has_idle {
                return ScaleEvent::ScaleDown { count: 1 };
            }
        }

        ScaleEvent::NoChange
    }

    /// Apply a scale event by updating internal state.
    ///
    /// `current_workers` is clamped to `[min_workers, max_workers]`.  The
    /// `last_scale_ms` clock is only advanced when an actual change occurs.
    pub fn apply_event(&mut self, event: &ScaleEvent, now_ms: u64) {
        match event {
            ScaleEvent::ScaleUp { count } => {
                let new_count = (self.current_workers + count).min(self.config.max_workers);
                if new_count != self.current_workers {
                    self.current_workers = new_count;
                    self.last_scale_ms = now_ms;
                }
            }
            ScaleEvent::ScaleDown { count } => {
                let new_count = self
                    .current_workers
                    .saturating_sub(*count)
                    .max(self.config.min_workers);
                if new_count != self.current_workers {
                    self.current_workers = new_count;
                    self.last_scale_ms = now_ms;
                    // Remove one idle worker from the tracking map.
                    self.evict_oldest_idle_worker(now_ms);
                }
            }
            ScaleEvent::NoChange => {}
        }
    }

    /// Record that a worker has become idle (i.e. finished all assigned jobs).
    pub fn mark_worker_idle(&mut self, worker_id: impl Into<String>, now_ms: u64) {
        self.worker_idle_since.insert(worker_id.into(), now_ms);
    }

    /// Record that a worker has been assigned a new job (no longer idle).
    pub fn mark_worker_busy(&mut self, worker_id: impl Into<String>) {
        self.worker_idle_since.remove(&worker_id.into());
    }

    /// Current number of workers in the pool.
    pub fn current_workers(&self) -> u32 {
        self.current_workers
    }

    /// IDs of workers that have been idle for at least `scale_down_idle_threshold_secs`.
    pub fn idle_workers(&self, now_ms: u64) -> Vec<String> {
        let threshold_ms = self.config.scale_down_idle_threshold_secs * 1000;
        let mut ids: Vec<String> = self
            .worker_idle_since
            .iter()
            .filter(|(_, &idle_since)| now_ms.saturating_sub(idle_since) >= threshold_ms)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Remove the worker that has been idle the longest from the tracking map.
    fn evict_oldest_idle_worker(&mut self, now_ms: u64) {
        // Find the worker idle the longest (smallest idle_since value).
        let oldest_id = self
            .worker_idle_since
            .iter()
            .filter(|(_, &idle_since)| {
                now_ms.saturating_sub(idle_since)
                    >= self.config.scale_down_idle_threshold_secs * 1000
            })
            .min_by_key(|(_, &idle_since)| idle_since)
            .map(|(id, _)| id.clone());

        if let Some(id) = oldest_id {
            self.worker_idle_since.remove(&id);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scaler(min: u32, max: u32) -> AutoScaler {
        AutoScaler::new(AutoScaleConfig::new(min, max)).expect("valid config")
    }

    #[test]
    fn scale_up_when_queue_exceeds_threshold() {
        let mut scaler = make_scaler(1, 8);
        // Default threshold = 5; queue_depth = 6 → ScaleUp
        let event = scaler.evaluate(6, 1_000_000);
        assert_eq!(event, ScaleEvent::ScaleUp { count: 1 });
    }

    #[test]
    fn no_scale_when_in_cooldown() {
        let mut scaler = make_scaler(1, 8);
        let t0: u64 = 1_000_000;
        let event = scaler.evaluate(10, t0);
        scaler.apply_event(&event, t0);

        // Within the 60-second cooldown window
        let t1 = t0 + 30_000;
        let event2 = scaler.evaluate(10, t1);
        assert_eq!(event2, ScaleEvent::NoChange, "should be in cooldown");
    }

    #[test]
    fn scale_down_when_worker_idle_long_enough() {
        let cfg = AutoScaleConfig {
            min_workers: 1,
            max_workers: 4,
            scale_up_queue_threshold: 5,
            scale_down_idle_threshold_secs: 300,
            cooldown_secs: 0, // no cooldown for this test
        };
        let mut scaler = AutoScaler::new(cfg).expect("valid");
        // Manually boost current_workers to 2
        scaler.current_workers = 2;

        let t0: u64 = 0;
        scaler.mark_worker_idle("worker-a", t0);

        // Advance past idle threshold
        let t1 = t0 + 300_001;
        let event = scaler.evaluate(0, t1);
        assert_eq!(event, ScaleEvent::ScaleDown { count: 1 });
    }

    #[test]
    fn min_boundary_enforced_on_scale_down() {
        let cfg = AutoScaleConfig {
            min_workers: 1,
            max_workers: 4,
            scale_up_queue_threshold: 5,
            scale_down_idle_threshold_secs: 10,
            cooldown_secs: 0,
        };
        let mut scaler = AutoScaler::new(cfg).expect("valid");
        // current = min = 1; even with idle workers, should not go below 1
        scaler.mark_worker_idle("w1", 0);
        let event = scaler.evaluate(0, 20_000);
        assert_eq!(event, ScaleEvent::NoChange, "cannot go below min");
    }

    #[test]
    fn max_boundary_enforced_on_scale_up() {
        let cfg = AutoScaleConfig {
            min_workers: 1,
            max_workers: 2,
            scale_up_queue_threshold: 1,
            scale_down_idle_threshold_secs: 300,
            cooldown_secs: 0,
        };
        let mut scaler = AutoScaler::new(cfg).expect("valid");
        scaler.current_workers = 2; // already at max

        let event = scaler.evaluate(100, 0);
        assert_eq!(event, ScaleEvent::NoChange, "already at max");
    }

    #[test]
    fn apply_event_updates_worker_count() {
        let mut scaler = make_scaler(1, 8);
        assert_eq!(scaler.current_workers(), 1);
        scaler.apply_event(&ScaleEvent::ScaleUp { count: 1 }, 0);
        assert_eq!(scaler.current_workers(), 2);
        scaler.apply_event(&ScaleEvent::ScaleDown { count: 1 }, 0);
        assert_eq!(scaler.current_workers(), 1);
    }

    #[test]
    fn validate_rejects_min_greater_than_max() {
        let cfg = AutoScaleConfig::new(5, 3); // invalid
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_min() {
        let cfg = AutoScaleConfig::new(0, 4);
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn idle_workers_returns_correct_ids() {
        let cfg = AutoScaleConfig {
            min_workers: 1,
            max_workers: 8,
            scale_up_queue_threshold: 5,
            scale_down_idle_threshold_secs: 60,
            cooldown_secs: 0,
        };
        let mut scaler = AutoScaler::new(cfg).expect("valid");

        scaler.mark_worker_idle("worker-b", 0);
        scaler.mark_worker_idle("worker-a", 0);

        // Not idle long enough yet
        let still_fresh = scaler.idle_workers(30_000);
        assert!(still_fresh.is_empty(), "too soon to consider idle");

        // Now past the threshold
        let stale = scaler.idle_workers(65_000);
        assert_eq!(stale, vec!["worker-a", "worker-b"]); // sorted
    }
}
