#![allow(dead_code)]
//! Job timeout management with deadline tracking, escalation policies, and grace periods.
//!
//! Provides per-job timeout configuration, stall detection, and configurable
//! escalation actions when jobs exceed their time budgets.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Action to take when a job times out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutAction {
    /// Cancel the job immediately.
    Cancel,
    /// Send a warning notification but let the job continue.
    Warn,
    /// Retry the job from scratch.
    Retry,
    /// Escalate to a higher-priority queue.
    Escalate,
    /// Pause the job and wait for manual intervention.
    Pause,
}

/// Escalation level after repeated timeouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EscalationLevel {
    /// No escalation yet.
    None,
    /// First warning issued.
    Warning,
    /// Supervisor notified.
    Supervisor,
    /// Hard kill issued.
    Critical,
}

/// Per-job timeout configuration.
#[derive(Debug, Clone)]
pub struct TimeoutConfig {
    /// Maximum wall-clock duration for the job.
    pub max_duration: Duration,
    /// Grace period after the soft deadline before hard action.
    pub grace_period: Duration,
    /// Interval at which to check for stalls (no progress).
    pub stall_check_interval: Duration,
    /// Minimum progress delta expected per stall check interval.
    pub min_progress_delta: f64,
    /// Action to take on soft timeout.
    pub soft_action: TimeoutAction,
    /// Action to take on hard timeout (after grace period).
    pub hard_action: TimeoutAction,
    /// Maximum number of escalation steps before forced cancel.
    pub max_escalations: u32,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_secs(3600),
            grace_period: Duration::from_secs(300),
            stall_check_interval: Duration::from_secs(60),
            min_progress_delta: 0.01,
            soft_action: TimeoutAction::Warn,
            hard_action: TimeoutAction::Cancel,
            max_escalations: 3,
        }
    }
}

/// Tracks the timeout state for a single job.
#[derive(Debug, Clone)]
pub struct JobTimeoutState {
    /// Unique job identifier (string key).
    pub job_id: String,
    /// When the job started executing.
    pub started_at: Instant,
    /// Last recorded progress (0.0 to 1.0).
    pub last_progress: f64,
    /// When progress was last updated.
    pub last_progress_at: Instant,
    /// Current escalation level.
    pub escalation_level: EscalationLevel,
    /// Number of escalations performed.
    pub escalation_count: u32,
    /// Whether the soft timeout has fired.
    pub soft_timeout_fired: bool,
    /// Whether the hard timeout has fired.
    pub hard_timeout_fired: bool,
    /// Configuration for this job.
    pub config: TimeoutConfig,
}

impl JobTimeoutState {
    /// Create a new timeout state for a job.
    pub fn new(job_id: String, config: TimeoutConfig) -> Self {
        let now = Instant::now();
        Self {
            job_id,
            started_at: now,
            last_progress: 0.0,
            last_progress_at: now,
            escalation_level: EscalationLevel::None,
            escalation_count: 0,
            soft_timeout_fired: false,
            hard_timeout_fired: false,
            config,
        }
    }

    /// Record a progress update.
    #[allow(clippy::cast_precision_loss)]
    pub fn update_progress(&mut self, progress: f64) {
        let clamped = progress.clamp(0.0, 1.0);
        self.last_progress = clamped;
        self.last_progress_at = Instant::now();
    }

    /// Elapsed time since the job started.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Time since the last progress update.
    pub fn time_since_progress(&self) -> Duration {
        self.last_progress_at.elapsed()
    }

    /// Check whether the soft deadline has been exceeded.
    pub fn is_soft_timeout(&self) -> bool {
        self.elapsed() >= self.config.max_duration
    }

    /// Check whether the hard deadline (max_duration + grace) has been exceeded.
    pub fn is_hard_timeout(&self) -> bool {
        self.elapsed() >= self.config.max_duration + self.config.grace_period
    }

    /// Check whether the job appears stalled (no progress in stall_check_interval).
    #[allow(clippy::cast_precision_loss)]
    pub fn is_stalled(&self) -> bool {
        self.time_since_progress() >= self.config.stall_check_interval
            && self.last_progress < 1.0
    }

    /// Remaining time before soft timeout, or zero if already exceeded.
    pub fn remaining(&self) -> Duration {
        self.config
            .max_duration
            .checked_sub(self.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    /// Escalate the timeout state and return the new level.
    pub fn escalate(&mut self) -> EscalationLevel {
        if self.escalation_count >= self.config.max_escalations {
            self.escalation_level = EscalationLevel::Critical;
        } else {
            self.escalation_count += 1;
            self.escalation_level = match self.escalation_count {
                1 => EscalationLevel::Warning,
                2 => EscalationLevel::Supervisor,
                _ => EscalationLevel::Critical,
            };
        }
        self.escalation_level
    }

    /// Evaluate what action, if any, should be taken right now.
    pub fn evaluate(&mut self) -> Option<TimeoutAction> {
        if self.is_hard_timeout() && !self.hard_timeout_fired {
            self.hard_timeout_fired = true;
            return Some(self.config.hard_action);
        }
        if self.is_soft_timeout() && !self.soft_timeout_fired {
            self.soft_timeout_fired = true;
            return Some(self.config.soft_action);
        }
        if self.is_stalled() {
            let level = self.escalate();
            if level == EscalationLevel::Critical {
                return Some(TimeoutAction::Cancel);
            }
            return Some(TimeoutAction::Warn);
        }
        None
    }
}

/// Manager that tracks timeout state for many jobs.
#[derive(Debug)]
pub struct TimeoutManager {
    /// Map from job-id to its timeout state.
    states: HashMap<String, JobTimeoutState>,
    /// Default config for jobs that don't supply their own.
    default_config: TimeoutConfig,
}

impl TimeoutManager {
    /// Create a new timeout manager with the given default config.
    pub fn new(default_config: TimeoutConfig) -> Self {
        Self {
            states: HashMap::new(),
            default_config,
        }
    }

    /// Register a job with an explicit timeout config.
    pub fn register(&mut self, job_id: &str, config: TimeoutConfig) {
        self.states
            .insert(job_id.to_string(), JobTimeoutState::new(job_id.to_string(), config));
    }

    /// Register a job using the default timeout config.
    pub fn register_default(&mut self, job_id: &str) {
        let config = self.default_config.clone();
        self.register(job_id, config);
    }

    /// Remove a job from tracking.
    pub fn unregister(&mut self, job_id: &str) -> Option<JobTimeoutState> {
        self.states.remove(job_id)
    }

    /// Update progress for a tracked job.
    pub fn update_progress(&mut self, job_id: &str, progress: f64) -> bool {
        if let Some(state) = self.states.get_mut(job_id) {
            state.update_progress(progress);
            true
        } else {
            false
        }
    }

    /// Evaluate all tracked jobs and return a list of (job_id, action) pairs.
    pub fn evaluate_all(&mut self) -> Vec<(String, TimeoutAction)> {
        let mut actions = Vec::new();
        for (job_id, state) in &mut self.states {
            if let Some(action) = state.evaluate() {
                actions.push((job_id.clone(), action));
            }
        }
        actions
    }

    /// Number of tracked jobs.
    pub fn tracked_count(&self) -> usize {
        self.states.len()
    }

    /// Get a reference to a job's timeout state.
    pub fn get_state(&self, job_id: &str) -> Option<&JobTimeoutState> {
        self.states.get(job_id)
    }

    /// Collect statistics: (total, soft_timed_out, hard_timed_out, stalled).
    #[allow(clippy::cast_precision_loss)]
    pub fn stats(&self) -> TimeoutStats {
        let total = self.states.len() as u64;
        let soft = self.states.values().filter(|s| s.soft_timeout_fired).count() as u64;
        let hard = self.states.values().filter(|s| s.hard_timeout_fired).count() as u64;
        let stalled = self.states.values().filter(|s| s.is_stalled()).count() as u64;
        TimeoutStats {
            total_tracked: total,
            soft_timeouts: soft,
            hard_timeouts: hard,
            stalled_jobs: stalled,
        }
    }
}

/// Summary statistics for timeouts across tracked jobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutStats {
    /// Total number of tracked jobs.
    pub total_tracked: u64,
    /// Number of jobs that hit soft timeout.
    pub soft_timeouts: u64,
    /// Number of jobs that hit hard timeout.
    pub hard_timeouts: u64,
    /// Number of currently stalled jobs.
    pub stalled_jobs: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quick_config(max_secs: u64, grace_secs: u64) -> TimeoutConfig {
        TimeoutConfig {
            max_duration: Duration::from_secs(max_secs),
            grace_period: Duration::from_secs(grace_secs),
            stall_check_interval: Duration::from_millis(10),
            min_progress_delta: 0.01,
            soft_action: TimeoutAction::Warn,
            hard_action: TimeoutAction::Cancel,
            max_escalations: 3,
        }
    }

    #[test]
    fn test_timeout_config_default() {
        let cfg = TimeoutConfig::default();
        assert_eq!(cfg.max_duration, Duration::from_secs(3600));
        assert_eq!(cfg.grace_period, Duration::from_secs(300));
        assert_eq!(cfg.soft_action, TimeoutAction::Warn);
        assert_eq!(cfg.hard_action, TimeoutAction::Cancel);
    }

    #[test]
    fn test_state_new_initial_values() {
        let state = JobTimeoutState::new("j1".into(), TimeoutConfig::default());
        assert_eq!(state.job_id, "j1");
        assert!((state.last_progress - 0.0).abs() < f64::EPSILON);
        assert_eq!(state.escalation_level, EscalationLevel::None);
        assert!(!state.soft_timeout_fired);
        assert!(!state.hard_timeout_fired);
    }

    #[test]
    fn test_update_progress_clamps() {
        let mut state = JobTimeoutState::new("j2".into(), TimeoutConfig::default());
        state.update_progress(1.5);
        assert!((state.last_progress - 1.0).abs() < f64::EPSILON);
        state.update_progress(-0.5);
        assert!((state.last_progress - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_remaining_decreases() {
        let cfg = quick_config(1, 0);
        let state = JobTimeoutState::new("j3".into(), cfg);
        let r = state.remaining();
        assert!(r <= Duration::from_secs(1));
    }

    #[test]
    fn test_escalation_levels() {
        let mut state = JobTimeoutState::new("j4".into(), TimeoutConfig::default());
        assert_eq!(state.escalate(), EscalationLevel::Warning);
        assert_eq!(state.escalate(), EscalationLevel::Supervisor);
        assert_eq!(state.escalate(), EscalationLevel::Critical);
        // Further escalations stay critical
        assert_eq!(state.escalate(), EscalationLevel::Critical);
    }

    #[test]
    fn test_timeout_action_variants() {
        assert_ne!(TimeoutAction::Cancel, TimeoutAction::Warn);
        assert_ne!(TimeoutAction::Retry, TimeoutAction::Escalate);
        assert_eq!(TimeoutAction::Pause, TimeoutAction::Pause);
    }

    #[test]
    fn test_manager_register_and_count() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        mgr.register_default("a");
        mgr.register_default("b");
        assert_eq!(mgr.tracked_count(), 2);
    }

    #[test]
    fn test_manager_unregister() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        mgr.register_default("a");
        let removed = mgr.unregister("a");
        assert!(removed.is_some());
        assert_eq!(mgr.tracked_count(), 0);
        assert!(mgr.unregister("nonexistent").is_none());
    }

    #[test]
    fn test_manager_update_progress_unknown_job() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        assert!(!mgr.update_progress("missing", 0.5));
    }

    #[test]
    fn test_manager_update_progress_known_job() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        mgr.register_default("x");
        assert!(mgr.update_progress("x", 0.75));
        let state = mgr.get_state("x").expect("state should be valid");
        assert!((state.last_progress - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_stats_initial() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        mgr.register_default("a");
        mgr.register_default("b");
        let s = mgr.stats();
        assert_eq!(s.total_tracked, 2);
        assert_eq!(s.soft_timeouts, 0);
        assert_eq!(s.hard_timeouts, 0);
    }

    #[test]
    fn test_evaluate_no_action_when_fresh() {
        let mut mgr = TimeoutManager::new(TimeoutConfig::default());
        mgr.register("j", quick_config(3600, 300));
        mgr.update_progress("j", 0.5);
        let actions = mgr.evaluate_all();
        // Should have no actions since deadline is far away and progress is recent
        assert!(actions.is_empty());
    }
}
