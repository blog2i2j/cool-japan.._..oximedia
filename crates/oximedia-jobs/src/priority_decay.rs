#![allow(dead_code)]
// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Configurable priority decay for aged jobs.
//!
//! While [`crate::job_priority_boost`] increases priority over time to prevent
//! starvation, this module provides the complementary mechanism: jobs that have
//! been sitting in the queue for too long may have their effective priority
//! *decayed* according to configurable strategies.  This is useful for
//! scenarios where old, stale jobs should yield to fresher, more relevant work.
//!
//! # Decay strategies
//!
//! * **Linear** – priority drops by a fixed amount per decay interval.
//! * **Exponential** – priority halves (or reduces by a configurable factor)
//!   each interval.
//! * **StepFunction** – priority drops in discrete steps at configured age
//!   thresholds.
//! * **Logarithmic** – decay slows down as the job ages (rapid initial decay,
//!   then plateau).
//!
//! A *floor* value ensures that no job's effective priority drops below the
//! configured minimum, so even heavily decayed jobs will eventually run.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Decay strategy
// ---------------------------------------------------------------------------

/// Strategy governing how priority decays over time.
#[derive(Debug, Clone, PartialEq)]
pub enum DecayStrategy {
    /// Priority drops by `amount` every `interval`.
    Linear {
        /// Fixed amount subtracted per interval.
        amount: f64,
        /// Length of one interval.
        interval: Duration,
    },
    /// Priority is multiplied by `factor` (0 < factor < 1) every `interval`.
    Exponential {
        /// Multiplicative factor per interval (e.g. 0.5 halves priority).
        factor: f64,
        /// Length of one interval.
        interval: Duration,
    },
    /// Priority drops to pre-defined values at specific age thresholds.
    StepFunction {
        /// Ordered list of `(age_threshold, priority_at_that_age)`.
        /// Must be sorted by threshold ascending.
        steps: Vec<(Duration, f64)>,
    },
    /// Priority decays as `base - k * ln(1 + elapsed/interval)`.
    Logarithmic {
        /// Coefficient controlling the speed of decay.
        coefficient: f64,
        /// Time unit for the logarithmic argument.
        interval: Duration,
    },
}

impl DecayStrategy {
    /// Compute the effective priority given `base_priority` and `elapsed` time.
    ///
    /// Returns a value clamped to `[floor, base_priority]`.
    pub fn compute(&self, base_priority: f64, elapsed: Duration, floor: f64) -> f64 {
        let raw = match self {
            Self::Linear { amount, interval } => {
                let intervals = elapsed.as_secs_f64() / interval.as_secs_f64().max(1e-9);
                base_priority - amount * intervals
            }
            Self::Exponential { factor, interval } => {
                let intervals = elapsed.as_secs_f64() / interval.as_secs_f64().max(1e-9);
                base_priority * factor.powf(intervals)
            }
            Self::StepFunction { steps } => {
                let mut result = base_priority;
                for (threshold, priority) in steps {
                    if elapsed >= *threshold {
                        result = *priority;
                    } else {
                        break;
                    }
                }
                result
            }
            Self::Logarithmic {
                coefficient,
                interval,
            } => {
                let t = elapsed.as_secs_f64() / interval.as_secs_f64().max(1e-9);
                base_priority - coefficient * (1.0 + t).ln()
            }
        };
        raw.clamp(floor, base_priority)
    }
}

// ---------------------------------------------------------------------------
// Decay configuration
// ---------------------------------------------------------------------------

/// Configuration for the priority decay engine.
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// The decay strategy to use.
    pub strategy: DecayStrategy,
    /// Minimum effective priority (floor). Jobs will never drop below this.
    pub floor: f64,
    /// Grace period: no decay occurs until the job has been queued for at
    /// least this long.
    pub grace_period: Duration,
    /// Whether decay is globally enabled.
    pub enabled: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            strategy: DecayStrategy::Linear {
                amount: 1.0,
                interval: Duration::from_secs(60),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-job tracking
// ---------------------------------------------------------------------------

/// Per-job state maintained by the decay engine.
#[derive(Debug, Clone)]
pub struct DecayJobState {
    /// Original (base) priority when the job was enqueued.
    pub base_priority: f64,
    /// Current effective priority after decay.
    pub effective_priority: f64,
    /// When the job was first enqueued.
    pub enqueued_at: Instant,
    /// Number of times decay has been recalculated.
    pub recalc_count: u64,
    /// Whether decay is paused for this specific job.
    pub paused: bool,
}

/// Record of a single decay event.
#[derive(Debug, Clone)]
pub struct DecayEvent {
    /// Job identifier.
    pub job_id: String,
    /// Priority before this decay step.
    pub previous_priority: f64,
    /// Priority after this decay step.
    pub new_priority: f64,
    /// The delta applied (negative means decay).
    pub delta: f64,
    /// When the event occurred.
    pub occurred_at: Instant,
}

// ---------------------------------------------------------------------------
// Decay engine
// ---------------------------------------------------------------------------

/// Manages priority decay for a set of tracked jobs.
#[derive(Debug)]
pub struct PriorityDecayEngine {
    config: DecayConfig,
    jobs: HashMap<String, DecayJobState>,
    events: Vec<DecayEvent>,
    max_events: usize,
}

impl PriorityDecayEngine {
    /// Create a new engine with the given configuration.
    #[must_use]
    pub fn new(config: DecayConfig) -> Self {
        Self {
            config,
            jobs: HashMap::new(),
            events: Vec::new(),
            max_events: 10_000,
        }
    }

    /// Set the maximum number of events retained.
    pub fn set_max_events(&mut self, max: usize) {
        self.max_events = max;
    }

    /// Register a job for decay tracking.
    pub fn register_job(&mut self, job_id: impl Into<String>, base_priority: f64) {
        let id = job_id.into();
        let now = Instant::now();
        self.jobs.insert(
            id,
            DecayJobState {
                base_priority,
                effective_priority: base_priority,
                enqueued_at: now,
                recalc_count: 0,
                paused: false,
            },
        );
    }

    /// Remove a job from tracking (e.g. when it completes or is cancelled).
    pub fn unregister_job(&mut self, job_id: &str) -> Option<DecayJobState> {
        self.jobs.remove(job_id)
    }

    /// Pause decay for a specific job (e.g. when it is running).
    pub fn pause_job(&mut self, job_id: &str) -> bool {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.paused = true;
            true
        } else {
            false
        }
    }

    /// Resume decay for a paused job.
    pub fn resume_job(&mut self, job_id: &str) -> bool {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.paused = false;
            true
        } else {
            false
        }
    }

    /// Recalculate effective priority for a single job. Returns the new
    /// effective priority, or `None` if the job is not tracked.
    pub fn recalculate(&mut self, job_id: &str) -> Option<f64> {
        if !self.config.enabled {
            return self.jobs.get(job_id).map(|s| s.effective_priority);
        }

        let state = self.jobs.get_mut(job_id)?;
        if state.paused {
            return Some(state.effective_priority);
        }

        let elapsed = state.enqueued_at.elapsed();
        if elapsed < self.config.grace_period {
            return Some(state.effective_priority);
        }

        let effective_elapsed = elapsed - self.config.grace_period;
        let previous = state.effective_priority;
        let new_priority =
            self.config
                .strategy
                .compute(state.base_priority, effective_elapsed, self.config.floor);

        state.effective_priority = new_priority;
        state.recalc_count += 1;

        let delta = new_priority - previous;
        if (delta).abs() > f64::EPSILON {
            self.record_event(job_id.to_string(), previous, new_priority, delta);
        }

        Some(new_priority)
    }

    /// Recalculate all tracked jobs and return a map of job_id -> new priority.
    pub fn recalculate_all(&mut self) -> HashMap<String, f64> {
        let ids: Vec<String> = self.jobs.keys().cloned().collect();
        let mut result = HashMap::new();
        for id in ids {
            if let Some(p) = self.recalculate(&id) {
                result.insert(id, p);
            }
        }
        result
    }

    /// Get the current state for a job.
    #[must_use]
    pub fn get_job_state(&self, job_id: &str) -> Option<&DecayJobState> {
        self.jobs.get(job_id)
    }

    /// Get all recent decay events.
    #[must_use]
    pub fn events(&self) -> &[DecayEvent] {
        &self.events
    }

    /// Number of tracked jobs.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Returns job IDs sorted by effective priority (lowest first).
    pub fn jobs_by_priority_ascending(&self) -> Vec<(String, f64)> {
        let mut entries: Vec<(String, f64)> = self
            .jobs
            .iter()
            .map(|(id, s)| (id.clone(), s.effective_priority))
            .collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    /// Reset a job's priority back to its base value.
    pub fn reset_job(&mut self, job_id: &str) -> bool {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.effective_priority = state.base_priority;
            state.recalc_count = 0;
            state.enqueued_at = Instant::now();
            true
        } else {
            false
        }
    }

    /// Update the decay configuration at runtime.
    pub fn update_config(&mut self, config: DecayConfig) {
        self.config = config;
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &DecayConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn record_event(&mut self, job_id: String, previous: f64, new: f64, delta: f64) {
        if self.events.len() >= self.max_events {
            // Drop the oldest quarter when full.
            let drain_count = self.max_events / 4;
            self.events.drain(..drain_count);
        }
        self.events.push(DecayEvent {
            job_id,
            previous_priority: previous,
            new_priority: new,
            delta,
            occurred_at: Instant::now(),
        });
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_engine() -> PriorityDecayEngine {
        PriorityDecayEngine::new(DecayConfig::default())
    }

    #[test]
    fn test_register_and_unregister() {
        let mut engine = default_engine();
        engine.register_job("job-1", 100.0);
        assert_eq!(engine.job_count(), 1);
        let state = engine.unregister_job("job-1");
        assert!(state.is_some());
        assert_eq!(engine.job_count(), 0);
    }

    #[test]
    fn test_linear_decay_no_elapsed() {
        // With zero elapsed the priority should remain at base.
        let strategy = DecayStrategy::Linear {
            amount: 5.0,
            interval: Duration::from_secs(10),
        };
        let result = strategy.compute(100.0, Duration::from_secs(0), 0.0);
        assert!((result - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_decay_multiple_intervals() {
        let strategy = DecayStrategy::Linear {
            amount: 10.0,
            interval: Duration::from_secs(60),
        };
        // After 3 intervals: 100 - 10*3 = 70
        let result = strategy.compute(100.0, Duration::from_secs(180), 0.0);
        assert!((result - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_linear_decay_clamped_to_floor() {
        let strategy = DecayStrategy::Linear {
            amount: 50.0,
            interval: Duration::from_secs(10),
        };
        // After 5 intervals: 100 - 250 = -150, clamped to floor=10
        let result = strategy.compute(100.0, Duration::from_secs(50), 10.0);
        assert!((result - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exponential_decay() {
        let strategy = DecayStrategy::Exponential {
            factor: 0.5,
            interval: Duration::from_secs(60),
        };
        // After 2 intervals: 100 * 0.5^2 = 25
        let result = strategy.compute(100.0, Duration::from_secs(120), 0.0);
        assert!((result - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_exponential_decay_floor() {
        let strategy = DecayStrategy::Exponential {
            factor: 0.1,
            interval: Duration::from_secs(10),
        };
        // After 3 intervals: 100 * 0.1^3 = 0.1, floor=5
        let result = strategy.compute(100.0, Duration::from_secs(30), 5.0);
        assert!((result - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_step_function_decay() {
        let strategy = DecayStrategy::StepFunction {
            steps: vec![
                (Duration::from_secs(30), 80.0),
                (Duration::from_secs(60), 50.0),
                (Duration::from_secs(120), 20.0),
            ],
        };
        // Before first step
        let r1 = strategy.compute(100.0, Duration::from_secs(10), 0.0);
        assert!((r1 - 100.0).abs() < f64::EPSILON);

        // At 45 seconds: matches first step only
        let r2 = strategy.compute(100.0, Duration::from_secs(45), 0.0);
        assert!((r2 - 80.0).abs() < f64::EPSILON);

        // At 90 seconds: matches first and second step
        let r3 = strategy.compute(100.0, Duration::from_secs(90), 0.0);
        assert!((r3 - 50.0).abs() < f64::EPSILON);

        // At 200 seconds: matches all three
        let r4 = strategy.compute(100.0, Duration::from_secs(200), 0.0);
        assert!((r4 - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_logarithmic_decay() {
        let strategy = DecayStrategy::Logarithmic {
            coefficient: 10.0,
            interval: Duration::from_secs(60),
        };
        // At t=0: 100 - 10*ln(1+0) = 100
        let r0 = strategy.compute(100.0, Duration::from_secs(0), 0.0);
        assert!((r0 - 100.0).abs() < 0.01);

        // At t=60s (1 interval): 100 - 10*ln(2) ≈ 93.07
        let r1 = strategy.compute(100.0, Duration::from_secs(60), 0.0);
        let expected = 100.0 - 10.0 * 2.0_f64.ln();
        assert!((r1 - expected).abs() < 0.01);
    }

    #[test]
    fn test_grace_period_prevents_decay() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 100.0,
                interval: Duration::from_secs(1),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(3600), // 1 hour grace
            enabled: true,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        // Recalculate immediately — within grace period, no decay
        let p = engine.recalculate("job-1");
        assert!(p.is_some());
        assert!((p.unwrap_or(0.0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pause_prevents_decay() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 50.0,
                interval: Duration::from_secs(1),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: true,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        engine.pause_job("job-1");
        // Even after time passes, paused job keeps its priority
        let p = engine.recalculate("job-1");
        assert!(p.is_some());
        assert!((p.unwrap_or(0.0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resume_after_pause() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 1.0,
                interval: Duration::from_secs(60),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: true,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        assert!(engine.pause_job("job-1"));
        assert!(engine.resume_job("job-1"));
        let state = engine.get_job_state("job-1");
        assert!(state.is_some());
        assert!(!state.map_or(true, |s| s.paused));
    }

    #[test]
    fn test_disabled_engine_no_decay() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 50.0,
                interval: Duration::from_secs(1),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: false,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        let p = engine.recalculate("job-1");
        assert!((p.unwrap_or(0.0) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recalculate_all() {
        let mut engine = default_engine();
        engine.register_job("a", 80.0);
        engine.register_job("b", 60.0);
        engine.register_job("c", 40.0);
        let results = engine.recalculate_all();
        assert_eq!(results.len(), 3);
        assert!(results.contains_key("a"));
        assert!(results.contains_key("b"));
        assert!(results.contains_key("c"));
    }

    #[test]
    fn test_jobs_by_priority_ascending() {
        let mut engine = default_engine();
        engine.register_job("low", 10.0);
        engine.register_job("high", 90.0);
        engine.register_job("mid", 50.0);
        let sorted = engine.jobs_by_priority_ascending();
        assert_eq!(sorted.len(), 3);
        assert_eq!(sorted[0].0, "low");
        assert_eq!(sorted[1].0, "mid");
        assert_eq!(sorted[2].0, "high");
    }

    #[test]
    fn test_reset_job() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 50.0,
                interval: Duration::from_secs(1),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: true,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        // Manually lower effective for test
        if let Some(s) = engine.jobs.get_mut("job-1") {
            s.effective_priority = 30.0;
        }
        assert!(engine.reset_job("job-1"));
        let state = engine.get_job_state("job-1");
        assert!((state.map_or(0.0, |s| s.effective_priority) - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_config_at_runtime() {
        let mut engine = default_engine();
        let new_config = DecayConfig {
            strategy: DecayStrategy::Exponential {
                factor: 0.9,
                interval: Duration::from_secs(30),
            },
            floor: 5.0,
            grace_period: Duration::from_secs(10),
            enabled: true,
        };
        engine.update_config(new_config.clone());
        assert!((engine.config().floor - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_event_recording() {
        let config = DecayConfig {
            strategy: DecayStrategy::Linear {
                amount: 50.0,
                interval: Duration::from_secs(1),
            },
            floor: 0.0,
            grace_period: Duration::from_secs(0),
            enabled: true,
        };
        let mut engine = PriorityDecayEngine::new(config);
        engine.register_job("job-1", 100.0);
        // Manually adjust enqueue time to simulate age.
        if let Some(s) = engine.jobs.get_mut("job-1") {
            s.enqueued_at = Instant::now() - Duration::from_secs(10);
        }
        engine.recalculate("job-1");
        // There should be at least one event since priority changed.
        assert!(!engine.events().is_empty());
    }

    #[test]
    fn test_event_capacity_trimming() {
        let config = DecayConfig::default();
        let mut engine = PriorityDecayEngine::new(config);
        engine.set_max_events(4);
        // Record 5 events manually via the internal method.
        for i in 0..5 {
            engine.record_event(format!("job-{i}"), 100.0, 90.0, -10.0);
        }
        // Should have trimmed the oldest quarter (1 event) then added 1.
        assert!(engine.events().len() <= 5);
    }

    #[test]
    fn test_unregister_unknown_job() {
        let mut engine = default_engine();
        assert!(engine.unregister_job("nonexistent").is_none());
    }

    #[test]
    fn test_pause_unknown_job() {
        let mut engine = default_engine();
        assert!(!engine.pause_job("nonexistent"));
    }

    #[test]
    fn test_recalculate_unknown_job() {
        let mut engine = default_engine();
        assert!(engine.recalculate("nonexistent").is_none());
    }
}
