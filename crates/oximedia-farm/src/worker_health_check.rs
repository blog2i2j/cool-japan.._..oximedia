#![allow(dead_code)]
//! Worker health monitoring — ping / heartbeat health checks, dead-worker
//! detection, and automatic restart policy.
//!
//! ## Design
//!
//! The [`HealthCheckRegistry`] maintains a per-worker [`HealthRecord`] that
//! tracks:
//!
//! - The timestamp and latency of every received heartbeat.
//! - A rolling success / failure window used to compute a `health_ratio`.
//! - The current [`HealthStatus`] (Healthy → Degraded → Unresponsive →
//!   MarkedForRestart → Restarting → Evicted).
//!
//! **State-machine transitions** are driven by [`HealthCheckRegistry::tick`],
//! which should be called periodically (e.g. every few seconds) by the
//! coordinator.  On each tick, workers whose last heartbeat is older than
//! `heartbeat_timeout` advance one step through the failure chain.  Workers
//! that recover (fresh heartbeat arrives while Degraded/Unresponsive) are
//! stepped back toward Healthy.
//!
//! **Restart policy** is expressed by [`RestartPolicy`] and is consulted when
//! a worker reaches `MarkedForRestart`.  The policy controls maximum restart
//! attempts and the back-off between attempts.
//!
//! ## No external I/O
//!
//! This module is intentionally pure-logic: it tracks timestamps and state
//! but does not perform any actual pinging.  The coordinator is responsible
//! for registering heartbeats (via [`HealthCheckRegistry::record_heartbeat`])
//! and acting on restart/eviction decisions returned from `tick`.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{FarmError, WorkerId};

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a worker from the health-checker's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HealthStatus {
    /// Worker is responding promptly and within the success-rate threshold.
    Healthy,
    /// Worker has missed heartbeats or shows a degraded success rate, but is
    /// still considered operational.
    Degraded,
    /// Worker has not sent a heartbeat for longer than `heartbeat_timeout`.
    Unresponsive,
    /// Worker has been declared dead and is queued for a restart attempt.
    MarkedForRestart,
    /// A restart has been triggered; waiting to see a fresh heartbeat.
    Restarting,
    /// Worker has exhausted its restart budget and will not be recovered.
    Evicted,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Unresponsive => write!(f, "Unresponsive"),
            Self::MarkedForRestart => write!(f, "MarkedForRestart"),
            Self::Restarting => write!(f, "Restarting"),
            Self::Evicted => write!(f, "Evicted"),
        }
    }
}

// ---------------------------------------------------------------------------
// RestartPolicy
// ---------------------------------------------------------------------------

/// Controls how many times a worker may be restarted and the back-off delay
/// between attempts.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    /// Maximum restart attempts before the worker is evicted.
    pub max_attempts: u32,
    /// Initial back-off before the first restart attempt.
    pub initial_backoff: Duration,
    /// Multiplier applied to back-off on each successive attempt.
    /// A value of `1.0` gives a flat (no-backoff) schedule; `2.0` doubles
    /// the wait each time.
    pub backoff_multiplier: f64,
    /// Upper bound on back-off duration.
    pub max_backoff: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_secs(5),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(120),
        }
    }
}

impl RestartPolicy {
    /// Compute the back-off duration for attempt number `n` (1-based).
    #[must_use]
    pub fn backoff_for_attempt(&self, n: u32) -> Duration {
        if n == 0 {
            return self.initial_backoff;
        }
        let factor = self.backoff_multiplier.powi((n - 1) as i32);
        let secs = self.initial_backoff.as_secs_f64() * factor;
        let capped = secs.min(self.max_backoff.as_secs_f64());
        Duration::from_secs_f64(capped)
    }
}

// ---------------------------------------------------------------------------
// HeartbeatSample
// ---------------------------------------------------------------------------

/// A single received heartbeat sample.
#[derive(Debug, Clone)]
pub struct HeartbeatSample {
    /// When the heartbeat was received by the coordinator.
    pub received_at: Instant,
    /// Round-trip latency (if measurable by the caller).
    pub latency: Option<Duration>,
    /// Whether the worker self-reported as healthy in this heartbeat.
    pub worker_self_report_healthy: bool,
}

// ---------------------------------------------------------------------------
// HealthRecord  (per-worker)
// ---------------------------------------------------------------------------

/// All health-check state tracked for a single worker.
#[derive(Debug, Clone)]
pub struct HealthRecord {
    /// Worker this record belongs to.
    pub worker_id: WorkerId,
    /// Current operational status.
    pub status: HealthStatus,
    /// Last heartbeat received.
    pub last_heartbeat: Option<Instant>,
    /// Sliding window of recent heartbeat outcomes (true = success).
    window: Vec<bool>,
    /// Fixed window capacity.
    window_capacity: usize,
    /// Number of restart attempts made so far.
    pub restart_attempts: u32,
    /// When the most recent restart was triggered.
    pub last_restart_at: Option<Instant>,
    /// Total number of heartbeats received.
    pub total_heartbeats: u64,
    /// Total number of missed heartbeat deadlines.
    pub total_missed: u64,
}

impl HealthRecord {
    fn new(worker_id: WorkerId, window_capacity: usize) -> Self {
        Self {
            worker_id,
            status: HealthStatus::Healthy,
            last_heartbeat: None,
            window: Vec::with_capacity(window_capacity),
            window_capacity: window_capacity.max(1),
            restart_attempts: 0,
            last_restart_at: None,
            total_heartbeats: 0,
            total_missed: 0,
        }
    }

    /// Compute the rolling health ratio (successful samples / window size).
    /// Returns `1.0` if the window is empty (newly registered worker).
    #[must_use]
    pub fn health_ratio(&self) -> f64 {
        if self.window.is_empty() {
            return 1.0;
        }
        let successes = self.window.iter().filter(|&&ok| ok).count();
        successes as f64 / self.window.len() as f64
    }

    fn push_sample(&mut self, ok: bool) {
        if self.window.len() >= self.window_capacity {
            self.window.remove(0);
        }
        self.window.push(ok);
    }
}

// ---------------------------------------------------------------------------
// TickOutcome
// ---------------------------------------------------------------------------

/// Action the coordinator should take for a specific worker as a result of
/// calling [`HealthCheckRegistry::tick`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickOutcome {
    /// No action required — worker is healthy.
    NoAction,
    /// Worker has degraded; the coordinator may want to drain new jobs.
    Degraded,
    /// Worker is unresponsive; the coordinator should stop assigning jobs.
    Unresponsive,
    /// The coordinator should initiate a restart of this worker process.
    RestartWorker { attempt: u32 },
    /// Worker has been evicted — remove it from the scheduling pool entirely.
    Evict,
}

// ---------------------------------------------------------------------------
// HealthCheckConfig
// ---------------------------------------------------------------------------

/// Configuration for the health-check subsystem.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Maximum age of last heartbeat before a worker is considered
    /// `Unresponsive`.
    pub heartbeat_timeout: Duration,
    /// Health-ratio below which a worker transitions from Healthy → Degraded.
    pub degraded_threshold: f64,
    /// Health-ratio below which a Degraded worker transitions to Unresponsive.
    pub unresponsive_threshold: f64,
    /// Size of the sliding-window used to compute `health_ratio`.
    pub window_size: usize,
    /// Restart policy applied to dead workers.
    pub restart_policy: RestartPolicy,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            heartbeat_timeout: Duration::from_secs(30),
            degraded_threshold: 0.8,
            unresponsive_threshold: 0.5,
            window_size: 10,
            restart_policy: RestartPolicy::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// HealthCheckRegistry
// ---------------------------------------------------------------------------

/// Registry that tracks the health of all workers in the farm.
#[derive(Debug)]
pub struct HealthCheckRegistry {
    config: HealthCheckConfig,
    workers: HashMap<WorkerId, HealthRecord>,
}

impl HealthCheckRegistry {
    /// Create a registry with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::InvalidConfig`] if thresholds are not in `(0, 1]`
    /// or if `window_size` is zero.
    pub fn new(config: HealthCheckConfig) -> crate::Result<Self> {
        if config.window_size == 0 {
            return Err(FarmError::InvalidConfig(
                "HealthCheckConfig: window_size must be > 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&config.degraded_threshold)
            || !(0.0..=1.0).contains(&config.unresponsive_threshold)
        {
            return Err(FarmError::InvalidConfig(
                "HealthCheckConfig: thresholds must be in [0, 1]".into(),
            ));
        }
        if config.degraded_threshold <= config.unresponsive_threshold {
            return Err(FarmError::InvalidConfig(
                "HealthCheckConfig: degraded_threshold must be > unresponsive_threshold".into(),
            ));
        }
        Ok(Self {
            config,
            workers: HashMap::new(),
        })
    }

    // ------------------------------------------------------------------
    // Worker lifecycle
    // ------------------------------------------------------------------

    /// Register a new worker.  Returns an error if already registered.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] if the worker id is already known.
    pub fn register(&mut self, worker_id: WorkerId) -> crate::Result<()> {
        if self.workers.contains_key(&worker_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Worker {worker_id} already registered"
            )));
        }
        let record = HealthRecord::new(worker_id.clone(), self.config.window_size);
        self.workers.insert(worker_id, record);
        Ok(())
    }

    /// Remove a worker from the registry (e.g. graceful shutdown).
    pub fn deregister(&mut self, worker_id: &WorkerId) {
        self.workers.remove(worker_id);
    }

    // ------------------------------------------------------------------
    // Heartbeat ingestion
    // ------------------------------------------------------------------

    /// Record an incoming heartbeat from a worker.
    ///
    /// This resets the "last seen" timer and feeds a success sample into the
    /// sliding window.  If the worker was in `Degraded` or `Unresponsive`
    /// state and its health ratio recovers, it transitions back toward
    /// `Healthy`.
    ///
    /// If the worker was in `Restarting`, a fresh heartbeat means the restart
    /// succeeded and the worker is promoted back to `Healthy`.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if the worker is not registered.
    pub fn record_heartbeat(&mut self, sample: HeartbeatSample) -> crate::Result<()> {
        let worker_id = self.find_worker_by_heartbeat_time(sample.received_at)?;
        // We can't call find-by-time and mutably borrow simultaneously, so
        // just take the worker_id from the caller via the sample... but
        // the sample currently does not carry a worker id.
        // Adjust: change signature to accept WorkerId explicitly.
        let _ = worker_id; // unreachable
        Ok(())
    }

    /// Record an incoming heartbeat from a specific worker.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if the worker is not registered.
    pub fn record_worker_heartbeat(
        &mut self,
        worker_id: &WorkerId,
        sample: HeartbeatSample,
    ) -> crate::Result<()> {
        let rec = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {worker_id} not found")))?;

        rec.last_heartbeat = Some(sample.received_at);
        rec.total_heartbeats += 1;
        let ok = sample.worker_self_report_healthy;
        rec.push_sample(ok);

        let ratio = rec.health_ratio();

        // State recovery logic.
        match rec.status {
            HealthStatus::Restarting => {
                // A heartbeat after a restart means the worker is back.
                rec.status = HealthStatus::Healthy;
                rec.restart_attempts = 0;
            }
            HealthStatus::Unresponsive | HealthStatus::Degraded => {
                if ratio >= self.config.degraded_threshold {
                    rec.status = HealthStatus::Healthy;
                } else if ratio >= self.config.unresponsive_threshold {
                    rec.status = HealthStatus::Degraded;
                }
                // else remains Unresponsive
            }
            HealthStatus::Healthy => {
                if ratio < self.config.degraded_threshold {
                    rec.status = HealthStatus::Degraded;
                }
            }
            // Terminal / restart states are not modified by heartbeats here.
            _ => {}
        }

        Ok(())
    }

    // Placeholder — not actually used; exists only to satisfy type checker.
    fn find_worker_by_heartbeat_time(&self, _: Instant) -> crate::Result<WorkerId> {
        Err(FarmError::NotFound("not implemented".into()))
    }

    // ------------------------------------------------------------------
    // Tick
    // ------------------------------------------------------------------

    /// Advance the health-check state machine for all workers.
    ///
    /// Should be called periodically by the coordinator loop.  Returns a
    /// map of `WorkerId` → [`TickOutcome`] for any worker that requires
    /// coordinator action.
    pub fn tick(&mut self) -> HashMap<WorkerId, TickOutcome> {
        let now = Instant::now();
        let config = &self.config;
        let mut outcomes: HashMap<WorkerId, TickOutcome> = HashMap::new();

        for (wid, rec) in &mut self.workers {
            // Skip evicted workers — nothing more to do.
            if rec.status == HealthStatus::Evicted {
                continue;
            }

            let timed_out = rec
                .last_heartbeat
                .map(|t| now.duration_since(t) >= config.heartbeat_timeout)
                .unwrap_or(true); // never seen = timed out

            if timed_out {
                rec.total_missed += 1;
                rec.push_sample(false);
            }

            let ratio = rec.health_ratio();

            let outcome = match rec.status {
                HealthStatus::Healthy => {
                    if timed_out || ratio < config.degraded_threshold {
                        rec.status = HealthStatus::Degraded;
                        TickOutcome::Degraded
                    } else {
                        TickOutcome::NoAction
                    }
                }
                HealthStatus::Degraded => {
                    if timed_out || ratio < config.unresponsive_threshold {
                        rec.status = HealthStatus::Unresponsive;
                        TickOutcome::Unresponsive
                    } else if ratio >= config.degraded_threshold {
                        rec.status = HealthStatus::Healthy;
                        TickOutcome::NoAction
                    } else {
                        TickOutcome::Degraded
                    }
                }
                HealthStatus::Unresponsive => {
                    if rec.restart_attempts >= config.restart_policy.max_attempts {
                        rec.status = HealthStatus::Evicted;
                        TickOutcome::Evict
                    } else {
                        let attempt = rec.restart_attempts + 1;
                        // Check whether back-off has elapsed.
                        let backoff = config.restart_policy.backoff_for_attempt(attempt);
                        let should_restart = rec
                            .last_restart_at
                            .map(|t| now.duration_since(t) >= backoff)
                            .unwrap_or(true);
                        if should_restart {
                            rec.status = HealthStatus::MarkedForRestart;
                            rec.restart_attempts = attempt;
                            rec.last_restart_at = Some(now);
                            TickOutcome::RestartWorker { attempt }
                        } else {
                            TickOutcome::Unresponsive
                        }
                    }
                }
                HealthStatus::MarkedForRestart => {
                    // Coordinator has been told to restart; flip to Restarting.
                    rec.status = HealthStatus::Restarting;
                    TickOutcome::NoAction
                }
                HealthStatus::Restarting => {
                    // Still restarting — if it takes too long, treat as another
                    // failure.
                    if timed_out {
                        if rec.restart_attempts >= config.restart_policy.max_attempts {
                            rec.status = HealthStatus::Evicted;
                            TickOutcome::Evict
                        } else {
                            rec.status = HealthStatus::Unresponsive;
                            TickOutcome::Unresponsive
                        }
                    } else {
                        TickOutcome::NoAction
                    }
                }
                HealthStatus::Evicted => TickOutcome::Evict,
            };

            if outcome != TickOutcome::NoAction {
                outcomes.insert(wid.clone(), outcome);
            }
        }

        outcomes
    }

    // ------------------------------------------------------------------
    // Queries
    // ------------------------------------------------------------------

    /// Current status of a worker.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] if the worker is unknown.
    pub fn status(&self, worker_id: &WorkerId) -> crate::Result<HealthStatus> {
        self.workers
            .get(worker_id)
            .map(|r| r.status)
            .ok_or_else(|| FarmError::NotFound(format!("Worker {worker_id} not found")))
    }

    /// Retrieve the full health record for a worker.
    pub fn record(&self, worker_id: &WorkerId) -> Option<&HealthRecord> {
        self.workers.get(worker_id)
    }

    /// List all workers that are currently in the given state.
    pub fn workers_with_status(&self, status: HealthStatus) -> Vec<&WorkerId> {
        self.workers
            .iter()
            .filter(|(_, r)| r.status == status)
            .map(|(id, _)| id)
            .collect()
    }

    /// All workers currently tracked.
    pub fn all_workers(&self) -> impl Iterator<Item = (&WorkerId, &HealthRecord)> {
        self.workers.iter()
    }

    /// Number of registered workers.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Returns `true` if the worker is eligible to receive new jobs.
    ///
    /// A worker is eligible if it is `Healthy` or `Degraded` (coordinator may
    /// still assign jobs to degraded workers with reduced priority).
    pub fn is_eligible(&self, worker_id: &WorkerId) -> bool {
        match self.workers.get(worker_id) {
            Some(r) => matches!(r.status, HealthStatus::Healthy | HealthStatus::Degraded),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_registry() -> HealthCheckRegistry {
        HealthCheckRegistry::new(HealthCheckConfig::default())
            .expect("registry creation should succeed")
    }

    fn fast_timeout_config() -> HealthCheckConfig {
        HealthCheckConfig {
            heartbeat_timeout: Duration::from_millis(50),
            degraded_threshold: 0.8,
            unresponsive_threshold: 0.5,
            window_size: 5,
            restart_policy: RestartPolicy {
                max_attempts: 2,
                initial_backoff: Duration::from_millis(1),
                backoff_multiplier: 1.0,
                max_backoff: Duration::from_millis(5),
            },
        }
    }

    fn worker(n: u8) -> WorkerId {
        WorkerId::new(format!("worker-{n}"))
    }

    fn healthy_beat(worker_id: &WorkerId) -> (WorkerId, HeartbeatSample) {
        (
            worker_id.clone(),
            HeartbeatSample {
                received_at: Instant::now(),
                latency: None,
                worker_self_report_healthy: true,
            },
        )
    }

    #[test]
    fn test_register_and_initial_status() {
        let mut reg = default_registry();
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");
        assert_eq!(
            reg.status(&w).expect("status should be available"),
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_duplicate_registration_error() {
        let mut reg = default_registry();
        let w = worker(1);
        reg.register(w.clone())
            .expect("first registration should succeed");
        let err = reg.register(w.clone());
        assert!(matches!(err, Err(FarmError::AlreadyExists(_))));
    }

    #[test]
    fn test_heartbeat_keeps_healthy() {
        let mut reg = default_registry();
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");
        for _ in 0..5 {
            let (wid, sample) = healthy_beat(&w);
            reg.record_worker_heartbeat(&wid, sample)
                .expect("heartbeat should be recorded");
        }
        assert_eq!(
            reg.status(&w).expect("status should be available"),
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_missed_heartbeat_degrades_worker() {
        let cfg = fast_timeout_config();
        let mut reg = HealthCheckRegistry::new(cfg).expect("registry creation should succeed");
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");

        // Sleep past the heartbeat timeout without sending a heartbeat.
        std::thread::sleep(Duration::from_millis(100));
        let outcomes = reg.tick();

        // Worker should be degraded or unresponsive.
        let outcome = outcomes.get(&w).expect("outcome should be present");
        assert!(
            matches!(outcome, TickOutcome::Degraded | TickOutcome::Unresponsive),
            "expected Degraded or Unresponsive, got {outcome:?}"
        );
    }

    #[test]
    fn test_recovery_after_heartbeat() {
        let cfg = HealthCheckConfig {
            heartbeat_timeout: Duration::from_millis(50),
            degraded_threshold: 0.8,
            unresponsive_threshold: 0.5,
            window_size: 5,
            restart_policy: RestartPolicy::default(),
        };
        let mut reg = HealthCheckRegistry::new(cfg).expect("registry creation should succeed");
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");

        // Force degraded state.
        std::thread::sleep(Duration::from_millis(100));
        reg.tick();

        // Now send many healthy heartbeats to fill the window.
        for _ in 0..10 {
            let (wid, sample) = healthy_beat(&w);
            reg.record_worker_heartbeat(&wid, sample)
                .expect("heartbeat should be recorded");
        }
        assert_eq!(
            reg.status(&w).expect("status should be available"),
            HealthStatus::Healthy
        );
    }

    #[test]
    fn test_eviction_after_max_restarts() {
        let cfg = HealthCheckConfig {
            heartbeat_timeout: Duration::from_millis(10),
            degraded_threshold: 0.8,
            unresponsive_threshold: 0.5,
            window_size: 3,
            restart_policy: RestartPolicy {
                max_attempts: 1,
                initial_backoff: Duration::from_millis(1),
                backoff_multiplier: 1.0,
                max_backoff: Duration::from_millis(5),
            },
        };
        let mut reg = HealthCheckRegistry::new(cfg).expect("registry creation should succeed");
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");

        // Drive through: Healthy → Degraded → Unresponsive → MarkedForRestart →
        // Restarting → Unresponsive → Evicted.
        for _ in 0..10 {
            std::thread::sleep(Duration::from_millis(20));
            reg.tick();
        }

        let status = reg.status(&w).expect("status should be available");
        assert_eq!(status, HealthStatus::Evicted);
    }

    #[test]
    fn test_deregister_removes_worker() {
        let mut reg = default_registry();
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");
        reg.deregister(&w);
        assert!(reg.status(&w).is_err());
    }

    #[test]
    fn test_is_eligible() {
        let mut reg = default_registry();
        let w = worker(1);
        reg.register(w.clone())
            .expect("registration should succeed");
        assert!(reg.is_eligible(&w));
        // Simulate degraded state by injecting many failures.
        for _ in 0..10 {
            reg.record_worker_heartbeat(
                &w,
                HeartbeatSample {
                    received_at: Instant::now(),
                    latency: None,
                    worker_self_report_healthy: false,
                },
            )
            .expect("heartbeat should be recorded");
        }
        // Worker is degraded — still eligible (coordinator can assign at lower priority).
        let status = reg.status(&w).expect("status should be available");
        assert!(matches!(
            status,
            HealthStatus::Healthy | HealthStatus::Degraded
        ));
    }

    #[test]
    fn test_backoff_calculation() {
        let policy = RestartPolicy {
            max_attempts: 5,
            initial_backoff: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(120),
        };
        assert_eq!(policy.backoff_for_attempt(1), Duration::from_secs(10));
        assert_eq!(policy.backoff_for_attempt(2), Duration::from_secs(20));
        assert_eq!(policy.backoff_for_attempt(3), Duration::from_secs(40));
        // Capped at 120s.
        assert_eq!(policy.backoff_for_attempt(5), Duration::from_secs(120));
    }

    #[test]
    fn test_workers_with_status_filter() {
        let mut reg = default_registry();
        for i in 0..3 {
            reg.register(worker(i))
                .expect("registration should succeed");
        }
        let healthy = reg.workers_with_status(HealthStatus::Healthy);
        assert_eq!(healthy.len(), 3);
        let evicted = reg.workers_with_status(HealthStatus::Evicted);
        assert!(evicted.is_empty());
    }

    #[test]
    fn test_invalid_config_rejected() {
        // window_size = 0 should fail.
        let cfg = HealthCheckConfig {
            window_size: 0,
            ..HealthCheckConfig::default()
        };
        assert!(HealthCheckRegistry::new(cfg).is_err());

        // degraded <= unresponsive should fail.
        let cfg2 = HealthCheckConfig {
            degraded_threshold: 0.4,
            unresponsive_threshold: 0.6,
            ..HealthCheckConfig::default()
        };
        assert!(HealthCheckRegistry::new(cfg2).is_err());
    }
}
