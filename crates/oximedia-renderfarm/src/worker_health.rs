// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker health scoring with automatic quarantine.
//!
//! Each worker in the render farm is monitored via a [`WorkerHealthScorer`].
//! The scorer maintains rolling success/failure counters, tracks consecutive
//! failures, and derives a [`WorkerHealthStatus`] according to a configurable
//! [`HealthConfig`].  Workers that exceed the consecutive-failure threshold are
//! automatically quarantined for a cooldown period and excluded from job
//! dispatch until released (manually or after the quarantine expires).

// ---------------------------------------------------------------------------
// WorkerHealthStatus
// ---------------------------------------------------------------------------

/// The health classification of a render-farm worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerHealthStatus {
    /// Failure rate is below the degraded threshold and no recent issues.
    Healthy,
    /// Elevated failure rate but still usable for new jobs.
    Degraded,
    /// Failure rate exceeds the unhealthy threshold; avoid for new jobs.
    Unhealthy,
    /// Temporarily removed from the pool after too many consecutive failures.
    Quarantined,
    /// No telemetry received yet, or the last event is too old to trust.
    Unknown,
}

// ---------------------------------------------------------------------------
// HealthMetrics
// ---------------------------------------------------------------------------

/// Accumulated performance counters for a single worker.
#[derive(Debug, Clone)]
pub struct HealthMetrics {
    /// Total jobs completed successfully.
    pub success_count: u64,
    /// Total jobs that ended in failure.
    pub failure_count: u64,
    /// Number of failures in an unbroken run (reset to 0 on any success).
    pub consecutive_failures: u32,
    /// Wall-clock timestamp of the last successful job completion.
    pub last_success_ms: Option<u64>,
    /// Wall-clock timestamp of the last job failure.
    pub last_failure_ms: Option<u64>,
    /// Exponentially-weighted moving average of completed-job durations.
    pub avg_job_duration_ms: f64,
    /// Jobs currently assigned to this worker and not yet finished.
    pub jobs_in_flight: u32,
}

impl HealthMetrics {
    /// Create zeroed-out metrics.
    pub fn new() -> Self {
        Self {
            success_count: 0,
            failure_count: 0,
            consecutive_failures: 0,
            last_success_ms: None,
            last_failure_ms: None,
            avg_job_duration_ms: 0.0,
            jobs_in_flight: 0,
        }
    }

    /// Record one successful job completion.
    ///
    /// Updates the EWMA of job duration (α = 0.1) and resets the consecutive
    /// failure counter.
    pub fn record_success(&mut self, duration_ms: u64, now_ms: u64) {
        self.success_count += 1;
        self.consecutive_failures = 0;
        self.last_success_ms = Some(now_ms);

        // EWMA update: new_avg = (1-α)*old_avg + α*sample
        const ALPHA: f64 = 0.1;
        if self.avg_job_duration_ms == 0.0 {
            self.avg_job_duration_ms = duration_ms as f64;
        } else {
            self.avg_job_duration_ms =
                (1.0 - ALPHA) * self.avg_job_duration_ms + ALPHA * duration_ms as f64;
        }
    }

    /// Record one job failure.  Increments the consecutive failure counter.
    pub fn record_failure(&mut self, now_ms: u64) {
        self.failure_count += 1;
        self.consecutive_failures += 1;
        self.last_failure_ms = Some(now_ms);
    }

    /// Fraction of completed jobs that succeeded.
    ///
    /// Returns `1.0` when no jobs have been observed (optimistic default).
    #[must_use]
    pub fn success_rate(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 1.0;
        }
        self.success_count as f32 / total as f32
    }
}

impl Default for HealthMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthConfig
// ---------------------------------------------------------------------------

/// Thresholds that drive the health-status transitions.
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Failure rate above which the worker is considered `Degraded`.  Default: 0.10.
    pub degraded_failure_rate: f32,
    /// Failure rate above which the worker is considered `Unhealthy`.  Default: 0.30.
    pub unhealthy_failure_rate: f32,
    /// Consecutive failures required before the worker is quarantined.  Default: 5.
    pub quarantine_consecutive: u32,
    /// How long (ms) a quarantine lasts before a worker may be reconsidered.  Default: 300 000 (5 min).
    pub quarantine_duration_ms: u64,
    /// If the last activity is older than this many milliseconds, status is `Unknown`.  Default: 60 000 (1 min).
    pub stale_threshold_ms: u64,
}

impl HealthConfig {
    /// Default configuration: tolerates up to 10 % errors before degraded,
    /// 30 % before unhealthy, 5 consecutive failures → 5-minute quarantine.
    pub fn default() -> Self {
        Self {
            degraded_failure_rate: 0.10,
            unhealthy_failure_rate: 0.30,
            quarantine_consecutive: 5,
            quarantine_duration_ms: 300_000,
            stale_threshold_ms: 60_000,
        }
    }

    /// Stricter thresholds suitable for high-SLA deployments.
    pub fn strict() -> Self {
        Self {
            degraded_failure_rate: 0.05,
            unhealthy_failure_rate: 0.15,
            quarantine_consecutive: 3,
            quarantine_duration_ms: 600_000,
            stale_threshold_ms: 30_000,
        }
    }

    /// Lenient thresholds suitable for experimental or development clusters.
    pub fn lenient() -> Self {
        Self {
            degraded_failure_rate: 0.20,
            unhealthy_failure_rate: 0.50,
            quarantine_consecutive: 10,
            quarantine_duration_ms: 60_000,
            stale_threshold_ms: 300_000,
        }
    }
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// WorkerHealthScorer
// ---------------------------------------------------------------------------

/// Tracks and scores the health of a single render-farm worker.
pub struct WorkerHealthScorer {
    /// Worker identifier (opaque string, e.g. UUID or hostname).
    pub worker_id: String,
    metrics: HealthMetrics,
    config: HealthConfig,
    /// Quarantine expiry timestamp.  `None` if not quarantined.
    quarantined_until_ms: Option<u64>,
}

impl WorkerHealthScorer {
    /// Create a new scorer with an `Unknown` initial state.
    pub fn new(worker_id: impl Into<String>, config: HealthConfig) -> Self {
        Self {
            worker_id: worker_id.into(),
            metrics: HealthMetrics::new(),
            config,
            quarantined_until_ms: None,
        }
    }

    /// Notify the scorer of a successful job completion.
    pub fn record_success(&mut self, duration_ms: u64, now_ms: u64) {
        self.metrics.record_success(duration_ms, now_ms);
    }

    /// Notify the scorer of a job failure.
    ///
    /// If the consecutive-failure threshold is reached the worker is
    /// automatically quarantined.
    pub fn record_failure(&mut self, now_ms: u64) {
        self.metrics.record_failure(now_ms);
        if self.metrics.consecutive_failures >= self.config.quarantine_consecutive {
            let until = now_ms + self.config.quarantine_duration_ms;
            self.quarantined_until_ms = Some(until);
        }
    }

    /// Derive the current [`WorkerHealthStatus`] at the given wall-clock time.
    ///
    /// Priority order:
    /// 1. Quarantined (explicit hold)
    /// 2. Unknown (no data or stale)
    /// 3. Unhealthy (consecutive threshold OR high failure rate)
    /// 4. Degraded (elevated failure rate)
    /// 5. Healthy
    pub fn status(&self, now_ms: u64) -> WorkerHealthStatus {
        // 1. Quarantine check (may have been set by record_failure)
        if let Some(until) = self.quarantined_until_ms {
            if now_ms < until {
                return WorkerHealthStatus::Quarantined;
            }
        }

        // 2. No data / stale
        let last_activity = match (self.metrics.last_success_ms, self.metrics.last_failure_ms) {
            (None, None) => return WorkerHealthStatus::Unknown,
            (Some(s), None) => s,
            (None, Some(f)) => f,
            (Some(s), Some(f)) => s.max(f),
        };
        if now_ms.saturating_sub(last_activity) > self.config.stale_threshold_ms {
            return WorkerHealthStatus::Unknown;
        }

        // 3. Unhealthy: consecutive failures at or above threshold
        if self.metrics.consecutive_failures >= self.config.quarantine_consecutive {
            return WorkerHealthStatus::Unhealthy;
        }

        let failure_rate = 1.0 - self.metrics.success_rate();

        if failure_rate > self.config.unhealthy_failure_rate {
            return WorkerHealthStatus::Unhealthy;
        }

        // 4. Degraded
        if failure_rate > self.config.degraded_failure_rate {
            return WorkerHealthStatus::Degraded;
        }

        // 5. Healthy
        WorkerHealthStatus::Healthy
    }

    /// Returns `true` if the worker may receive new jobs
    /// (`Healthy` or `Degraded` status).
    pub fn is_usable(&self, now_ms: u64) -> bool {
        matches!(
            self.status(now_ms),
            WorkerHealthStatus::Healthy | WorkerHealthStatus::Degraded
        )
    }

    /// Read-only access to accumulated metrics.
    pub fn metrics(&self) -> &HealthMetrics {
        &self.metrics
    }

    /// Lift the active quarantine immediately (e.g. after manual intervention).
    pub fn remove_quarantine(&mut self) {
        self.quarantined_until_ms = None;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000_000;

    fn default_scorer(id: &str) -> WorkerHealthScorer {
        WorkerHealthScorer::new(id, HealthConfig::default())
    }

    #[test]
    fn new_scorer_is_unknown() {
        let scorer = default_scorer("w1");
        assert_eq!(scorer.status(NOW), WorkerHealthStatus::Unknown);
    }

    #[test]
    fn after_first_success_healthy() {
        let mut scorer = default_scorer("w1");
        scorer.record_success(1000, NOW);
        assert_eq!(scorer.status(NOW), WorkerHealthStatus::Healthy);
        assert!(scorer.is_usable(NOW));
    }

    #[test]
    fn high_failure_rate_is_unhealthy() {
        let mut scorer = default_scorer("w1");
        // 6 failures, 1 success → ~85 % failure rate → Unhealthy
        for i in 0..6u64 {
            scorer.record_failure(NOW + i);
        }
        // Reset consecutive manually by recording a success then more failures
        // We need a fresh scorer to test pure rate-based Unhealthy
        let mut scorer2 = WorkerHealthScorer::new(
            "w2",
            HealthConfig {
                quarantine_consecutive: 100, // very high so rate triggers first
                ..HealthConfig::default()
            },
        );
        for i in 0..7u64 {
            scorer2.record_failure(NOW + i);
        }
        scorer2.record_success(500, NOW + 8);
        // 7 failures, 1 success → 87.5 % failure rate → Unhealthy
        assert_eq!(scorer2.status(NOW + 8), WorkerHealthStatus::Unhealthy);
        assert!(!scorer2.is_usable(NOW + 8));
    }

    #[test]
    fn consecutive_failures_triggers_quarantine() {
        let mut scorer = default_scorer("w1");
        let cfg = HealthConfig::default();
        for i in 0..cfg.quarantine_consecutive {
            scorer.record_failure(NOW + i as u64 * 100);
        }
        assert_eq!(scorer.status(NOW + 500), WorkerHealthStatus::Quarantined);
    }

    #[test]
    fn quarantined_worker_is_not_usable() {
        let mut scorer = default_scorer("w1");
        for i in 0..5u64 {
            scorer.record_failure(NOW + i);
        }
        assert!(!scorer.is_usable(NOW + 10));
    }

    #[test]
    fn remove_quarantine_makes_worker_usable_again() {
        let mut scorer = default_scorer("w1");
        for i in 0..5u64 {
            scorer.record_failure(NOW + i);
        }
        assert_eq!(scorer.status(NOW + 10), WorkerHealthStatus::Quarantined);
        scorer.remove_quarantine();
        // After removing quarantine, consecutive is still ≥ threshold so status
        // would be Unhealthy rather than Quarantined (no active quarantine window).
        let s = scorer.status(NOW + 10);
        assert_ne!(s, WorkerHealthStatus::Quarantined);
    }

    #[test]
    fn success_rate_formula() {
        let mut m = HealthMetrics::new();
        m.record_success(100, 1000);
        m.record_success(200, 1001);
        m.record_failure(1002);
        // 2 successes, 1 failure → 2/3 ≈ 0.667
        let rate = m.success_rate();
        assert!((rate - 2.0 / 3.0).abs() < 1e-5, "unexpected rate: {rate}");
    }

    #[test]
    fn record_failure_increments_consecutive() {
        let mut m = HealthMetrics::new();
        m.record_failure(1000);
        m.record_failure(1001);
        assert_eq!(m.consecutive_failures, 2);
    }

    #[test]
    fn success_resets_consecutive_failures() {
        let mut m = HealthMetrics::new();
        m.record_failure(1000);
        m.record_failure(1001);
        m.record_failure(1002);
        assert_eq!(m.consecutive_failures, 3);
        m.record_success(500, 1003);
        assert_eq!(m.consecutive_failures, 0, "success must reset consecutive");
    }

    #[test]
    fn stale_worker_is_unknown() {
        let mut scorer = default_scorer("w1");
        scorer.record_success(100, 0); // activity at t=0
                                       // Now + stale_threshold + 1 ms → stale
        let stale_time = HealthConfig::default().stale_threshold_ms + 1;
        assert_eq!(scorer.status(stale_time), WorkerHealthStatus::Unknown);
    }

    #[test]
    fn degraded_status_for_moderate_failure_rate() {
        let cfg = HealthConfig {
            quarantine_consecutive: 100, // disable quarantine by consecutive
            ..HealthConfig::default()
        };
        let mut scorer = WorkerHealthScorer::new("w-deg", cfg);
        // 1 failure, 8 successes → 11 % failure rate → Degraded (>10 % but ≤30 %)
        scorer.record_failure(NOW);
        for i in 1..=8u64 {
            scorer.record_success(100, NOW + i);
        }
        assert_eq!(scorer.status(NOW + 8), WorkerHealthStatus::Degraded);
        assert!(
            scorer.is_usable(NOW + 8),
            "degraded workers are still usable"
        );
    }
}
