#![allow(dead_code)]
//! Dynamic priority boosting for queued jobs.
//!
//! Implements several strategies for adjusting job priority based on factors such as
//! wait time, approaching deadlines, starvation prevention, and dependency completion.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Strategy used to boost a job's priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoostStrategy {
    /// Boost based on how long a job has been waiting.
    WaitTime,
    /// Boost when a deadline is approaching.
    DeadlineProximity,
    /// Boost when a job has been starved (passed over repeatedly).
    StarvationPrevention,
    /// Boost when all upstream dependencies have completed.
    DependencyCompletion,
    /// Manual boost applied by an operator.
    Manual,
}

impl std::fmt::Display for BoostStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WaitTime => write!(f, "WaitTime"),
            Self::DeadlineProximity => write!(f, "DeadlineProximity"),
            Self::StarvationPrevention => write!(f, "StarvationPrevention"),
            Self::DependencyCompletion => write!(f, "DependencyCompletion"),
            Self::Manual => write!(f, "Manual"),
        }
    }
}

/// A record of a single priority boost event.
#[derive(Debug, Clone)]
pub struct BoostEvent {
    /// Identifier for the boosted job.
    pub job_id: String,
    /// Strategy that triggered the boost.
    pub strategy: BoostStrategy,
    /// Priority delta applied (positive = higher priority).
    pub delta: i32,
    /// When the boost was applied.
    pub applied_at: Instant,
    /// Human-readable reason.
    pub reason: String,
}

/// Configuration for the priority booster.
#[derive(Debug, Clone)]
pub struct BoostConfig {
    /// How long a job waits before receiving a wait-time boost.
    pub wait_threshold: Duration,
    /// Priority increment per wait-time boost cycle.
    pub wait_boost_increment: i32,
    /// Maximum total boost from wait time.
    pub max_wait_boost: i32,
    /// How close to deadline (in seconds) before deadline boost kicks in.
    pub deadline_proximity_secs: u64,
    /// Priority increment for deadline proximity.
    pub deadline_boost_increment: i32,
    /// Number of times a job must be passed over before starvation boost.
    pub starvation_pass_count: u32,
    /// Priority increment for starvation prevention.
    pub starvation_boost_increment: i32,
    /// Priority increment when all dependencies complete.
    pub dependency_complete_boost: i32,
    /// Ceiling on the effective priority value.
    pub priority_ceiling: i32,
}

impl Default for BoostConfig {
    fn default() -> Self {
        Self {
            wait_threshold: Duration::from_secs(60),
            wait_boost_increment: 5,
            max_wait_boost: 50,
            deadline_proximity_secs: 300,
            deadline_boost_increment: 20,
            starvation_pass_count: 10,
            starvation_boost_increment: 10,
            dependency_complete_boost: 15,
            priority_ceiling: 100,
        }
    }
}

/// Per-job state tracked by the booster.
#[derive(Debug, Clone)]
pub struct JobBoostState {
    /// Original base priority.
    pub base_priority: i32,
    /// Current effective priority (base + boosts).
    pub effective_priority: i32,
    /// Total accumulated boost.
    pub total_boost: i32,
    /// When the job entered the queue.
    pub enqueued_at: Instant,
    /// Optional hard deadline.
    pub deadline: Option<Instant>,
    /// Number of times this job was passed over during scheduling.
    pub pass_count: u32,
    /// Whether all dependencies are satisfied.
    pub dependencies_complete: bool,
    /// History of boost events.
    pub boost_history: Vec<BoostEvent>,
}

impl JobBoostState {
    /// Create a new job boost state.
    pub fn new(base_priority: i32, deadline: Option<Instant>) -> Self {
        Self {
            base_priority,
            effective_priority: base_priority,
            total_boost: 0,
            enqueued_at: Instant::now(),
            deadline,
            pass_count: 0,
            dependencies_complete: false,
            boost_history: Vec::new(),
        }
    }
}

/// Priority booster that manages dynamic priority adjustments.
#[derive(Debug)]
pub struct PriorityBooster {
    /// Configuration.
    pub config: BoostConfig,
    /// Per-job state.
    jobs: HashMap<String, JobBoostState>,
    /// Aggregate statistics: total boosts applied.
    total_boosts_applied: u64,
}

impl PriorityBooster {
    /// Create a new priority booster.
    pub fn new(config: BoostConfig) -> Self {
        Self {
            config,
            jobs: HashMap::new(),
            total_boosts_applied: 0,
        }
    }

    /// Register a job for priority boosting.
    pub fn register_job(&mut self, job_id: &str, base_priority: i32, deadline: Option<Instant>) {
        self.jobs.insert(
            job_id.to_string(),
            JobBoostState::new(base_priority, deadline),
        );
    }

    /// Remove a job (e.g. after completion).
    pub fn unregister_job(&mut self, job_id: &str) -> Option<JobBoostState> {
        self.jobs.remove(job_id)
    }

    /// Get the current effective priority for a job.
    pub fn effective_priority(&self, job_id: &str) -> Option<i32> {
        self.jobs.get(job_id).map(|s| s.effective_priority)
    }

    /// Record that a job was passed over during scheduling.
    pub fn record_pass(&mut self, job_id: &str) {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.pass_count += 1;
        }
    }

    /// Mark a job's dependencies as fully satisfied.
    pub fn mark_dependencies_complete(&mut self, job_id: &str) {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.dependencies_complete = true;
        }
    }

    /// Apply a single boost to a job, clamping to the ceiling.
    fn apply_boost(&mut self, job_id: &str, strategy: BoostStrategy, delta: i32, reason: &str) {
        if let Some(state) = self.jobs.get_mut(job_id) {
            let clamped = (state.effective_priority + delta).min(self.config.priority_ceiling);
            let actual_delta = clamped - state.effective_priority;
            if actual_delta > 0 {
                state.effective_priority = clamped;
                state.total_boost += actual_delta;
                state.boost_history.push(BoostEvent {
                    job_id: job_id.to_string(),
                    strategy,
                    delta: actual_delta,
                    applied_at: Instant::now(),
                    reason: reason.to_string(),
                });
                self.total_boosts_applied += 1;
            }
        }
    }

    /// Evaluate and apply wait-time boosts for a single job.
    pub fn evaluate_wait_time(&mut self, job_id: &str) {
        let (should_boost, current_wait_boost) = {
            if let Some(state) = self.jobs.get(job_id) {
                let waited = state.enqueued_at.elapsed();
                let cycles = waited.as_secs() / self.config.wait_threshold.as_secs().max(1);
                #[allow(clippy::cast_precision_loss)]
                let expected_boost = (cycles as i32) * self.config.wait_boost_increment;
                let current_wait = state.total_boost;
                (
                    expected_boost > current_wait && current_wait < self.config.max_wait_boost,
                    current_wait,
                )
            } else {
                return;
            }
        };
        if should_boost {
            let increment = self
                .config
                .wait_boost_increment
                .min(self.config.max_wait_boost - current_wait_boost);
            self.apply_boost(
                job_id,
                BoostStrategy::WaitTime,
                increment,
                "Job waited beyond threshold",
            );
        }
    }

    /// Evaluate deadline proximity for a single job.
    pub fn evaluate_deadline(&mut self, job_id: &str) {
        let should_boost = {
            if let Some(state) = self.jobs.get(job_id) {
                if let Some(deadline) = state.deadline {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    remaining.as_secs() <= self.config.deadline_proximity_secs
                } else {
                    false
                }
            } else {
                false
            }
        };
        if should_boost {
            self.apply_boost(
                job_id,
                BoostStrategy::DeadlineProximity,
                self.config.deadline_boost_increment,
                "Deadline approaching",
            );
        }
    }

    /// Evaluate starvation for a single job.
    pub fn evaluate_starvation(&mut self, job_id: &str) {
        let should_boost = {
            if let Some(state) = self.jobs.get(job_id) {
                state.pass_count >= self.config.starvation_pass_count
            } else {
                false
            }
        };
        if should_boost {
            self.apply_boost(
                job_id,
                BoostStrategy::StarvationPrevention,
                self.config.starvation_boost_increment,
                "Job was starved",
            );
            // Reset pass count after boost
            if let Some(state) = self.jobs.get_mut(job_id) {
                state.pass_count = 0;
            }
        }
    }

    /// Evaluate dependency completion for a single job.
    pub fn evaluate_dependency_completion(&mut self, job_id: &str) {
        let should_boost = {
            if let Some(state) = self.jobs.get(job_id) {
                state.dependencies_complete
            } else {
                false
            }
        };
        if should_boost {
            self.apply_boost(
                job_id,
                BoostStrategy::DependencyCompletion,
                self.config.dependency_complete_boost,
                "All dependencies completed",
            );
        }
    }

    /// Apply a manual boost.
    pub fn manual_boost(&mut self, job_id: &str, delta: i32, reason: &str) {
        self.apply_boost(job_id, BoostStrategy::Manual, delta, reason);
    }

    /// Run all evaluations for every registered job.
    pub fn evaluate_all(&mut self) {
        let ids: Vec<String> = self.jobs.keys().cloned().collect();
        for id in ids {
            self.evaluate_wait_time(&id);
            self.evaluate_deadline(&id);
            self.evaluate_starvation(&id);
            self.evaluate_dependency_completion(&id);
        }
    }

    /// Get the number of tracked jobs.
    pub fn tracked_job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Get total boosts applied across all jobs.
    pub fn total_boosts_applied(&self) -> u64 {
        self.total_boosts_applied
    }

    /// Get the boost history for a specific job.
    pub fn boost_history(&self, job_id: &str) -> Option<&[BoostEvent]> {
        self.jobs.get(job_id).map(|s| s.boost_history.as_slice())
    }

    /// Reset a job's priority back to its base.
    pub fn reset_priority(&mut self, job_id: &str) {
        if let Some(state) = self.jobs.get_mut(job_id) {
            state.effective_priority = state.base_priority;
            state.total_boost = 0;
        }
    }

    /// Get a sorted list of jobs by effective priority (descending).
    pub fn jobs_by_priority(&self) -> Vec<(String, i32)> {
        let mut result: Vec<(String, i32)> = self
            .jobs
            .iter()
            .map(|(id, state)| (id.clone(), state.effective_priority))
            .collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boost_config_default() {
        let config = BoostConfig::default();
        assert_eq!(config.wait_boost_increment, 5);
        assert_eq!(config.max_wait_boost, 50);
        assert_eq!(config.priority_ceiling, 100);
    }

    #[test]
    fn test_register_and_effective_priority() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        assert_eq!(booster.effective_priority("job-1"), Some(10));
        assert_eq!(booster.tracked_job_count(), 1);
    }

    #[test]
    fn test_unregister_job() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        let state = booster.unregister_job("job-1");
        assert!(state.is_some());
        assert_eq!(booster.tracked_job_count(), 0);
        assert_eq!(booster.effective_priority("job-1"), None);
    }

    #[test]
    fn test_manual_boost() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        booster.manual_boost("job-1", 20, "urgent");
        assert_eq!(booster.effective_priority("job-1"), Some(30));
        assert_eq!(booster.total_boosts_applied(), 1);
    }

    #[test]
    fn test_manual_boost_clamps_to_ceiling() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 90, None);
        booster.manual_boost("job-1", 50, "max-out");
        assert_eq!(booster.effective_priority("job-1"), Some(100));
    }

    #[test]
    fn test_starvation_boost() {
        let config = BoostConfig {
            starvation_pass_count: 3,
            starvation_boost_increment: 10,
            ..BoostConfig::default()
        };
        let mut booster = PriorityBooster::new(config);
        booster.register_job("job-1", 10, None);
        for _ in 0..3 {
            booster.record_pass("job-1");
        }
        booster.evaluate_starvation("job-1");
        assert_eq!(booster.effective_priority("job-1"), Some(20));
    }

    #[test]
    fn test_dependency_completion_boost() {
        let config = BoostConfig {
            dependency_complete_boost: 15,
            ..BoostConfig::default()
        };
        let mut booster = PriorityBooster::new(config);
        booster.register_job("job-1", 10, None);
        booster.mark_dependencies_complete("job-1");
        booster.evaluate_dependency_completion("job-1");
        assert_eq!(booster.effective_priority("job-1"), Some(25));
    }

    #[test]
    fn test_deadline_boost() {
        let config = BoostConfig {
            deadline_proximity_secs: 600,
            deadline_boost_increment: 20,
            ..BoostConfig::default()
        };
        let mut booster = PriorityBooster::new(config);
        // Deadline in 5 seconds (well within 600s proximity)
        let deadline = Instant::now() + Duration::from_secs(5);
        booster.register_job("job-1", 10, Some(deadline));
        booster.evaluate_deadline("job-1");
        assert_eq!(booster.effective_priority("job-1"), Some(30));
    }

    #[test]
    fn test_no_deadline_no_boost() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        booster.evaluate_deadline("job-1");
        assert_eq!(booster.effective_priority("job-1"), Some(10));
    }

    #[test]
    fn test_reset_priority() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        booster.manual_boost("job-1", 30, "test");
        assert_eq!(booster.effective_priority("job-1"), Some(40));
        booster.reset_priority("job-1");
        assert_eq!(booster.effective_priority("job-1"), Some(10));
    }

    #[test]
    fn test_boost_history() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("job-1", 10, None);
        booster.manual_boost("job-1", 5, "first");
        booster.manual_boost("job-1", 10, "second");
        let history = booster
            .boost_history("job-1")
            .expect("history should be valid");
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].strategy, BoostStrategy::Manual);
        assert_eq!(history[1].delta, 10);
    }

    #[test]
    fn test_jobs_by_priority_sorted() {
        let mut booster = PriorityBooster::new(BoostConfig::default());
        booster.register_job("low", 5, None);
        booster.register_job("high", 50, None);
        booster.register_job("mid", 25, None);
        let sorted = booster.jobs_by_priority();
        assert_eq!(sorted[0].0, "high");
        assert_eq!(sorted[1].0, "mid");
        assert_eq!(sorted[2].0, "low");
    }

    #[test]
    fn test_boost_strategy_display() {
        assert_eq!(BoostStrategy::WaitTime.to_string(), "WaitTime");
        assert_eq!(
            BoostStrategy::DeadlineProximity.to_string(),
            "DeadlineProximity"
        );
        assert_eq!(
            BoostStrategy::StarvationPrevention.to_string(),
            "StarvationPrevention"
        );
        assert_eq!(
            BoostStrategy::DependencyCompletion.to_string(),
            "DependencyCompletion"
        );
        assert_eq!(BoostStrategy::Manual.to_string(), "Manual");
    }

    #[test]
    fn test_evaluate_all_runs_without_panic() {
        let config = BoostConfig {
            starvation_pass_count: 2,
            deadline_proximity_secs: 600,
            ..BoostConfig::default()
        };
        let mut booster = PriorityBooster::new(config);
        let deadline = Instant::now() + Duration::from_secs(10);
        booster.register_job("j1", 10, Some(deadline));
        booster.register_job("j2", 20, None);
        booster.mark_dependencies_complete("j2");
        for _ in 0..3 {
            booster.record_pass("j1");
        }
        booster.evaluate_all();
        // j1 should have gotten at least deadline + starvation boosts
        assert!(
            booster
                .effective_priority("j1")
                .expect("effective_priority should succeed")
                > 10
        );
        // j2 should have gotten dependency completion boost
        assert!(
            booster
                .effective_priority("j2")
                .expect("effective_priority should succeed")
                > 20
        );
    }

    #[test]
    fn test_job_boost_state_new() {
        let state = JobBoostState::new(42, None);
        assert_eq!(state.base_priority, 42);
        assert_eq!(state.effective_priority, 42);
        assert_eq!(state.total_boost, 0);
        assert_eq!(state.pass_count, 0);
        assert!(!state.dependencies_complete);
        assert!(state.boost_history.is_empty());
    }
}
