#![allow(dead_code)]
//! Farm-level job retry policies for the encoding farm.
//!
//! This module provides a flexible, composable retry framework that controls
//! how failed jobs are re-queued after transient errors.  Key capabilities:
//!
//! - Configurable maximum attempt count per job.
//! - Pluggable back-off strategies: fixed, linear, exponential, and jittered
//!   exponential.
//! - Optional worker blacklisting: after a worker causes `N` consecutive
//!   failures for a job it is removed from the eligible set for that job.
//! - Per-job retry state tracking so the coordinator can query how many
//!   attempts remain and which workers have been blacklisted.
//! - Hard deadline enforcement: once a job's deadline passes it is never
//!   retried regardless of remaining attempts.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::{FarmError, JobId, WorkerId};

// ---------------------------------------------------------------------------
// Back-off strategy
// ---------------------------------------------------------------------------

/// Strategy used to compute the delay before the next retry attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Always wait the same fixed duration.
    Fixed {
        /// Constant delay between attempts.
        delay: Duration,
    },
    /// Increase delay linearly: `base + attempt * step`.
    Linear {
        /// Base delay for attempt 0.
        base: Duration,
        /// Amount added per additional attempt.
        step: Duration,
    },
    /// Double the delay on every attempt: `base * 2^attempt`, capped at `max`.
    Exponential {
        /// Delay for the first retry (attempt 0).
        base: Duration,
        /// Upper cap on the computed delay.
        max: Duration,
    },
    /// Exponential back-off with full jitter (uniform in `[0, base * 2^attempt]`).
    ///
    /// The jitter uses a deterministic LCG seeded from the `JobId`'s UUID
    /// bytes so that behaviour is reproducible in tests while still spreading
    /// retries in production.
    JitteredExponential {
        /// Base delay multiplied by `2^attempt`.
        base: Duration,
        /// Upper cap before jitter is applied.
        max: Duration,
    },
}

impl BackoffStrategy {
    /// Compute the back-off duration for the given attempt index (0-based).
    ///
    /// `job_seed` is used only for jittered strategies — a simple but
    /// deterministic pseudo-random value derived from the job identifier.
    #[must_use]
    pub fn delay_for(&self, attempt: u32, job_seed: u64) -> Duration {
        match self {
            Self::Fixed { delay } => *delay,
            Self::Linear { base, step } => {
                base.saturating_add(step.saturating_mul(attempt))
            }
            Self::Exponential { base, max } => {
                let factor = 1u64.checked_shl(attempt.min(62)).unwrap_or(u64::MAX);
                let nanos = base
                    .as_nanos()
                    .saturating_mul(factor as u128)
                    .min(max.as_nanos());
                Duration::from_nanos(nanos as u64)
            }
            Self::JitteredExponential { base, max } => {
                let factor = 1u64.checked_shl(attempt.min(62)).unwrap_or(u64::MAX);
                let cap_nanos = base
                    .as_nanos()
                    .saturating_mul(factor as u128)
                    .min(max.as_nanos()) as u64;
                // Deterministic LCG: avoids pulling in rand crate
                let rand_val = lcg_rand(job_seed.wrapping_add(attempt as u64));
                let jitter = if cap_nanos == 0 {
                    0
                } else {
                    rand_val % cap_nanos
                };
                Duration::from_nanos(jitter)
            }
        }
    }
}

/// Minimal linear-congruential PRNG (Knuth's constants).
fn lcg_rand(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ---------------------------------------------------------------------------
// Retry policy configuration
// ---------------------------------------------------------------------------

/// Configuration for the farm-level retry policy applied to all jobs.
#[derive(Debug, Clone)]
pub struct RetryPolicyConfig {
    /// Maximum number of attempts (first attempt + retries).
    ///
    /// A value of `1` means no retries.
    pub max_attempts: u32,

    /// Back-off strategy to use between retries.
    pub backoff: BackoffStrategy,

    /// After this many consecutive failures **on the same worker**, that
    /// worker is blacklisted for the specific job.  `None` disables per-job
    /// blacklisting.
    pub blacklist_after_consecutive_failures: Option<u32>,

    /// Do not retry after this absolute wall-clock deadline.  `None` means no
    /// deadline.
    pub retry_deadline: Option<Duration>,
}

impl Default for RetryPolicyConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                base: Duration::from_secs(5),
                max: Duration::from_secs(300),
            },
            blacklist_after_consecutive_failures: Some(2),
            retry_deadline: Some(Duration::from_secs(3600)),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-job retry state
// ---------------------------------------------------------------------------

/// Outcome of a single job execution attempt on a specific worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The attempt succeeded.
    Success,
    /// A transient failure — job should be retried.
    TransientFailure,
    /// A permanent/fatal failure — job must not be retried.
    PermanentFailure,
    /// The attempt timed out.
    Timeout,
}

impl AttemptOutcome {
    /// Whether this outcome warrants a retry.
    #[must_use]
    pub fn is_retryable(self) -> bool {
        matches!(self, Self::TransientFailure | Self::Timeout)
    }
}

/// Runtime state tracking retries for a single job.
#[derive(Debug)]
pub struct JobRetryState {
    /// The job being tracked.
    pub job_id: JobId,
    /// Number of attempts made so far (including the initial attempt).
    pub attempts_made: u32,
    /// Whether the job has permanently failed (no more retries allowed).
    pub permanently_failed: bool,
    /// When the first attempt was started (for deadline enforcement).
    pub first_attempt_at: Instant,
    /// When the last attempt was started.
    pub last_attempt_at: Option<Instant>,
    /// Consecutive failure count per worker (for blacklisting).
    consecutive_failures: HashMap<WorkerId, u32>,
    /// Workers blacklisted for this job.
    blacklisted_workers: HashSet<WorkerId>,
}

impl JobRetryState {
    /// Create a new retry-state record for a job.
    #[must_use]
    pub fn new(job_id: JobId) -> Self {
        Self {
            job_id,
            attempts_made: 0,
            permanently_failed: false,
            first_attempt_at: Instant::now(),
            last_attempt_at: None,
            consecutive_failures: HashMap::new(),
            blacklisted_workers: HashSet::new(),
        }
    }

    /// Record the start of an attempt on the given worker.
    pub fn record_attempt_started(&mut self, worker_id: &WorkerId) {
        self.attempts_made += 1;
        self.last_attempt_at = Some(Instant::now());
        // Reset consecutive failure counter for this worker when a new
        // attempt begins (it will be incremented on failure if needed).
        let _ = worker_id; // used only for blacklist tracking on outcome
    }

    /// Record the outcome of the latest attempt on the given worker.
    ///
    /// Returns `true` when the worker was just blacklisted as a result of this
    /// outcome.
    pub fn record_attempt_outcome(
        &mut self,
        worker_id: &WorkerId,
        outcome: AttemptOutcome,
        policy: &RetryPolicyConfig,
    ) -> bool {
        if outcome == AttemptOutcome::PermanentFailure {
            self.permanently_failed = true;
        }

        let newly_blacklisted = if !outcome.is_retryable() || outcome == AttemptOutcome::Success {
            // Success or permanent failure: reset streak
            self.consecutive_failures.remove(worker_id);
            false
        } else {
            // Transient failure or timeout: increment streak
            let streak = self
                .consecutive_failures
                .entry(worker_id.clone())
                .or_insert(0);
            *streak += 1;
            if let Some(threshold) = policy.blacklist_after_consecutive_failures {
                if *streak >= threshold && !self.blacklisted_workers.contains(worker_id) {
                    self.blacklisted_workers.insert(worker_id.clone());
                    return true;
                }
            }
            false
        };
        newly_blacklisted
    }

    /// Whether the given worker is blacklisted for this job.
    #[must_use]
    pub fn is_blacklisted(&self, worker_id: &WorkerId) -> bool {
        self.blacklisted_workers.contains(worker_id)
    }

    /// All workers currently blacklisted for this job.
    #[must_use]
    pub fn blacklisted_workers(&self) -> &HashSet<WorkerId> {
        &self.blacklisted_workers
    }
}

// ---------------------------------------------------------------------------
// Policy evaluator
// ---------------------------------------------------------------------------

/// Evaluates retry eligibility for jobs and computes the next retry delay.
///
/// One instance is typically shared across the coordinator.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    config: RetryPolicyConfig,
}

/// Decision returned by [`RetryPolicy::evaluate`].
#[derive(Debug, Clone, PartialEq)]
pub enum RetryDecision {
    /// The job should be retried after the specified delay.
    Retry {
        /// How long to wait before re-queuing the job.
        delay: Duration,
        /// Attempt number that will be made next (1-based).
        next_attempt: u32,
    },
    /// The job has exhausted all retries or hit a hard stop.
    Abandon {
        /// Human-readable reason.
        reason: AbandonReason,
    },
}

/// Why a job was abandoned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbandonReason {
    /// The attempt returned a permanent/fatal error.
    PermanentFailure,
    /// The maximum number of attempts has been reached.
    MaxAttemptsExceeded,
    /// The retry deadline has passed.
    DeadlineExceeded,
    /// The job succeeded — no retry needed.
    Succeeded,
}

impl RetryPolicy {
    /// Create a new policy from the given configuration.
    #[must_use]
    pub fn new(config: RetryPolicyConfig) -> Self {
        Self { config }
    }

    /// Create a policy with default settings.
    #[must_use]
    pub fn default_policy() -> Self {
        Self::new(RetryPolicyConfig::default())
    }

    /// Evaluate whether a job should be retried after the given outcome.
    ///
    /// * `state` — mutable per-job retry state (updated in place).
    /// * `worker_id` — the worker that just ran the attempt.
    /// * `outcome` — what happened.
    /// * `job_seed` — deterministic seed derived from the job id (used for
    ///   jitter).
    ///
    /// The method advances `state.attempts_made` and applies blacklisting
    /// before computing the decision.
    pub fn evaluate(
        &self,
        state: &mut JobRetryState,
        worker_id: &WorkerId,
        outcome: AttemptOutcome,
        job_seed: u64,
    ) -> RetryDecision {
        // Record the outcome and possibly blacklist the worker.
        state.record_attempt_outcome(worker_id, outcome, &self.config);

        // Short-circuit: success
        if outcome == AttemptOutcome::Success {
            return RetryDecision::Abandon {
                reason: AbandonReason::Succeeded,
            };
        }

        // Short-circuit: permanent failure
        if state.permanently_failed {
            return RetryDecision::Abandon {
                reason: AbandonReason::PermanentFailure,
            };
        }

        // Check max attempts
        if state.attempts_made >= self.config.max_attempts {
            return RetryDecision::Abandon {
                reason: AbandonReason::MaxAttemptsExceeded,
            };
        }

        // Check deadline
        if let Some(deadline_dur) = self.config.retry_deadline {
            if state.first_attempt_at.elapsed() >= deadline_dur {
                return RetryDecision::Abandon {
                    reason: AbandonReason::DeadlineExceeded,
                };
            }
        }

        // Compute next delay (attempt index is 0-based for back-off purposes)
        let attempt_idx = state.attempts_made.saturating_sub(1);
        let delay = self.config.backoff.delay_for(attempt_idx, job_seed);

        RetryDecision::Retry {
            delay,
            next_attempt: state.attempts_made + 1,
        }
    }

    /// Expose a reference to the underlying configuration.
    #[must_use]
    pub fn config(&self) -> &RetryPolicyConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Registry: tracks per-job retry state for all in-flight jobs
// ---------------------------------------------------------------------------

/// Coordinator-level store for [`JobRetryState`] records.
#[derive(Debug, Default)]
pub struct RetryStateStore {
    states: HashMap<JobId, JobRetryState>,
}

impl RetryStateStore {
    /// Create an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a new retry state for a job.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::AlreadyExists`] if the job is already tracked.
    pub fn insert(&mut self, state: JobRetryState) -> crate::Result<()> {
        if self.states.contains_key(&state.job_id) {
            return Err(FarmError::AlreadyExists(format!(
                "Retry state for job {} already exists",
                state.job_id
            )));
        }
        self.states.insert(state.job_id, state);
        Ok(())
    }

    /// Retrieve a mutable reference to the retry state for a job.
    ///
    /// # Errors
    ///
    /// Returns [`FarmError::NotFound`] when the job is unknown.
    pub fn get_mut(&mut self, job_id: &JobId) -> crate::Result<&mut JobRetryState> {
        self.states
            .get_mut(job_id)
            .ok_or_else(|| FarmError::NotFound(format!("No retry state for job {}", job_id)))
    }

    /// Remove and return the retry state for a completed/abandoned job.
    pub fn remove(&mut self, job_id: &JobId) -> Option<JobRetryState> {
        self.states.remove(job_id)
    }

    /// Number of jobs currently tracked.
    #[must_use]
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// `true` when no jobs are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JobId;

    fn make_job() -> JobId {
        JobId::new()
    }

    fn make_worker(name: &str) -> WorkerId {
        WorkerId::new(name)
    }

    // Helper: run N transient-failure attempts and collect decisions
    fn run_failures(
        policy: &RetryPolicy,
        state: &mut JobRetryState,
        worker: &WorkerId,
        count: u32,
    ) -> Vec<RetryDecision> {
        (0..count)
            .map(|_| {
                state.record_attempt_started(worker);
                policy.evaluate(state, worker, AttemptOutcome::TransientFailure, 42)
            })
            .collect()
    }

    #[test]
    fn test_success_returns_succeeded() {
        let policy = RetryPolicy::default_policy();
        let mut state = JobRetryState::new(make_job());
        let worker = make_worker("w1");
        state.record_attempt_started(&worker);
        let decision = policy.evaluate(&mut state, &worker, AttemptOutcome::Success, 0);
        assert_eq!(
            decision,
            RetryDecision::Abandon {
                reason: AbandonReason::Succeeded
            }
        );
    }

    #[test]
    fn test_permanent_failure_abandons_immediately() {
        let policy = RetryPolicy::default_policy();
        let mut state = JobRetryState::new(make_job());
        let worker = make_worker("w1");
        state.record_attempt_started(&worker);
        let decision =
            policy.evaluate(&mut state, &worker, AttemptOutcome::PermanentFailure, 0);
        assert_eq!(
            decision,
            RetryDecision::Abandon {
                reason: AbandonReason::PermanentFailure
            }
        );
    }

    #[test]
    fn test_max_attempts_exceeded() {
        let config = RetryPolicyConfig {
            max_attempts: 2,
            backoff: BackoffStrategy::Fixed {
                delay: Duration::from_millis(10),
            },
            blacklist_after_consecutive_failures: None,
            retry_deadline: None,
        };
        let policy = RetryPolicy::new(config);
        let mut state = JobRetryState::new(make_job());
        let worker = make_worker("w1");

        let decisions = run_failures(&policy, &mut state, &worker, 2);
        // First failure → retry; second failure → abandon (max=2)
        assert!(matches!(decisions[0], RetryDecision::Retry { .. }));
        assert_eq!(
            decisions[1],
            RetryDecision::Abandon {
                reason: AbandonReason::MaxAttemptsExceeded
            }
        );
    }

    #[test]
    fn test_exponential_backoff_delays_increase() {
        let strat = BackoffStrategy::Exponential {
            base: Duration::from_secs(1),
            max: Duration::from_secs(60),
        };
        let d0 = strat.delay_for(0, 0);
        let d1 = strat.delay_for(1, 0);
        let d2 = strat.delay_for(2, 0);
        assert!(d0 <= d1);
        assert!(d1 <= d2);
    }

    #[test]
    fn test_linear_backoff() {
        let strat = BackoffStrategy::Linear {
            base: Duration::from_secs(5),
            step: Duration::from_secs(5),
        };
        assert_eq!(strat.delay_for(0, 0), Duration::from_secs(5));
        assert_eq!(strat.delay_for(1, 0), Duration::from_secs(10));
        assert_eq!(strat.delay_for(2, 0), Duration::from_secs(15));
    }

    #[test]
    fn test_worker_blacklisting() {
        let config = RetryPolicyConfig {
            max_attempts: 10,
            backoff: BackoffStrategy::Fixed {
                delay: Duration::from_millis(1),
            },
            blacklist_after_consecutive_failures: Some(2),
            retry_deadline: None,
        };
        let policy = RetryPolicy::new(config);
        let mut state = JobRetryState::new(make_job());
        let worker = make_worker("bad-worker");

        // Two consecutive transient failures
        for _ in 0..2 {
            state.record_attempt_started(&worker);
            policy.evaluate(&mut state, &worker, AttemptOutcome::TransientFailure, 0);
        }
        assert!(state.is_blacklisted(&worker));
    }

    #[test]
    fn test_jittered_exponential_does_not_exceed_max() {
        let strat = BackoffStrategy::JitteredExponential {
            base: Duration::from_secs(1),
            max: Duration::from_secs(30),
        };
        for attempt in 0..10u32 {
            let d = strat.delay_for(attempt, 12345);
            assert!(d <= Duration::from_secs(30), "delay {d:?} exceeded max at attempt {attempt}");
        }
    }

    #[test]
    fn test_retry_state_store_insert_get() {
        let mut store = RetryStateStore::new();
        let job_id = make_job();
        let state = JobRetryState::new(job_id);
        store.insert(state).expect("insert succeeds");
        assert_eq!(store.len(), 1);
        // Duplicate insert should fail
        let state2 = JobRetryState::new(job_id);
        assert!(store.insert(state2).is_err());
        // get_mut should work
        let _ = store.get_mut(&job_id).expect("found");
        store.remove(&job_id);
        assert!(store.is_empty());
    }

    #[test]
    fn test_deadline_exceeded() {
        // Set an artificially short deadline in the past by using elapsed time
        let config = RetryPolicyConfig {
            max_attempts: 10,
            backoff: BackoffStrategy::Fixed {
                delay: Duration::from_millis(1),
            },
            blacklist_after_consecutive_failures: None,
            // Effectively zero — will always be exceeded after first attempt
            retry_deadline: Some(Duration::ZERO),
        };
        let policy = RetryPolicy::new(config);
        let mut state = JobRetryState::new(make_job());
        let worker = make_worker("w1");
        state.record_attempt_started(&worker);
        // Sleep is not needed: Duration::ZERO means elapsed() >= 0 is always true
        let decision =
            policy.evaluate(&mut state, &worker, AttemptOutcome::TransientFailure, 0);
        assert_eq!(
            decision,
            RetryDecision::Abandon {
                reason: AbandonReason::DeadlineExceeded
            }
        );
    }
}
