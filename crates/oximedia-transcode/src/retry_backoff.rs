//! Exponential back-off retry scheduling for failed transcode jobs.
//!
//! Provides [`BackoffPolicy`], [`RetryState`], and [`RetryScheduler`] for
//! computing per-attempt delays with optional full-jitter randomisation.
//!
//! The delay formula is:
//! ```text
//! base_delay * multiplier^(attempt - 1), clamped to max_delay
//! ```
//! When jitter is enabled, the actual delay is drawn uniformly from
//! `[0, computed_delay]` (full-jitter strategy), which avoids thundering-herd
//! scenarios when many jobs fail simultaneously.

use std::time::Duration;

/// Strategy used to spread retry delays across concurrent failing jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JitterStrategy {
    /// No jitter; delay is deterministic.
    None,
    /// Full jitter: delay drawn uniformly from `[0, computed_delay]`.
    Full,
    /// Equal jitter: `computed_delay / 2 + random * computed_delay / 2`.
    Equal,
}

/// Policy governing how retry delays are computed.
#[derive(Debug, Clone)]
pub struct BackoffPolicy {
    /// Delay before the first retry attempt.
    pub base_delay: Duration,
    /// Multiplicative factor applied per attempt (must be ≥ 1.0).
    pub multiplier: f64,
    /// Upper bound on computed delay before jitter.
    pub max_delay: Duration,
    /// Maximum number of retry attempts (None = unlimited).
    pub max_attempts: Option<u32>,
    /// Jitter strategy to apply after computing the base delay.
    pub jitter: JitterStrategy,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            base_delay: Duration::from_millis(500),
            multiplier: 2.0,
            max_delay: Duration::from_secs(30),
            max_attempts: Some(5),
            jitter: JitterStrategy::Full,
        }
    }
}

impl BackoffPolicy {
    /// Creates a new policy with the given base delay and exponential multiplier.
    #[must_use]
    pub fn new(base_delay: Duration, multiplier: f64) -> Self {
        Self {
            base_delay,
            multiplier,
            ..Self::default()
        }
    }

    /// Builder: sets the maximum delay ceiling.
    #[must_use]
    pub fn with_max_delay(mut self, max: Duration) -> Self {
        self.max_delay = max;
        self
    }

    /// Builder: sets the maximum number of attempts.
    #[must_use]
    pub fn with_max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = Some(n);
        self
    }

    /// Builder: disables the maximum-attempts cap.
    #[must_use]
    pub fn unlimited(mut self) -> Self {
        self.max_attempts = None;
        self
    }

    /// Builder: sets the jitter strategy.
    #[must_use]
    pub fn with_jitter(mut self, jitter: JitterStrategy) -> Self {
        self.jitter = jitter;
        self
    }

    /// Computes the deterministic (pre-jitter) delay for attempt number `n` (1-based).
    ///
    /// Returns `None` when `n` exceeds `max_attempts`.
    #[must_use]
    pub fn deterministic_delay(&self, attempt: u32) -> Option<Duration> {
        if let Some(max) = self.max_attempts {
            if attempt > max {
                return None;
            }
        }
        if attempt == 0 {
            return Some(Duration::ZERO);
        }
        let exp = u32::from(attempt.saturating_sub(1));
        let scale = self.multiplier.powi(exp as i32);
        let base_us = self.base_delay.as_micros() as f64;
        let delay_us = (base_us * scale).min(self.max_delay.as_micros() as f64);
        Some(Duration::from_micros(delay_us.round() as u64))
    }

    /// Returns `true` when `attempt` is still within the allowed window.
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        self.max_attempts.map_or(true, |max| attempt <= max)
    }
}

/// Tracks the state for a single retrying operation.
#[derive(Debug, Clone)]
pub struct RetryState {
    /// Number of attempts made so far (starts at 0 before the first try).
    pub attempt: u32,
    /// Cumulative time spent sleeping between retries.
    pub total_delay: Duration,
    /// The policy governing this retry sequence.
    pub policy: BackoffPolicy,
}

impl RetryState {
    /// Creates a new retry state with the given policy.
    #[must_use]
    pub fn new(policy: BackoffPolicy) -> Self {
        Self {
            attempt: 0,
            total_delay: Duration::ZERO,
            policy,
        }
    }

    /// Creates a retry state with the default policy.
    #[must_use]
    pub fn with_default_policy() -> Self {
        Self::new(BackoffPolicy::default())
    }

    /// Returns `true` if another retry is permitted under the policy.
    #[must_use]
    pub fn can_retry(&self) -> bool {
        // attempt is the number of failures so far; next would be attempt+1
        self.policy.should_retry(self.attempt + 1)
    }

    /// Computes the next delay without advancing state.
    ///
    /// Returns `None` when no further retries are allowed.
    #[must_use]
    pub fn next_delay_deterministic(&self) -> Option<Duration> {
        self.policy.deterministic_delay(self.attempt + 1)
    }

    /// Advances the state, recording a failure. Returns the delay to wait before
    /// the next attempt, or `None` if the retry limit has been reached.
    pub fn record_failure(&mut self) -> Option<Duration> {
        self.attempt += 1;
        let delay = self.policy.deterministic_delay(self.attempt)?;
        self.total_delay += delay;
        Some(delay)
    }

    /// Resets attempt counter (e.g. after a successful intermediate step).
    pub fn reset(&mut self) {
        self.attempt = 0;
        self.total_delay = Duration::ZERO;
    }

    /// Returns how many attempts remain, or `None` for unlimited policies.
    #[must_use]
    pub fn remaining_attempts(&self) -> Option<u32> {
        self.policy
            .max_attempts
            .map(|max| max.saturating_sub(self.attempt))
    }
}

/// Schedules retry operations and computes jittered delays using an LCG-based
/// pseudo-random generator (no external dependency required).
///
/// Uses a 64-bit linear congruential generator seeded from elapsed time.
pub struct RetryScheduler {
    policy: BackoffPolicy,
    /// LCG state for full-jitter randomisation.
    rng_state: u64,
}

impl RetryScheduler {
    /// Creates a scheduler with the given policy, seeding the LCG from `seed`.
    #[must_use]
    pub fn new(policy: BackoffPolicy, seed: u64) -> Self {
        Self {
            policy,
            rng_state: seed | 1, // ensure odd for maximal period
        }
    }

    /// Creates a scheduler with the default policy and a time-derived seed.
    #[must_use]
    pub fn with_default_policy() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0xdead_beef_cafe_babe);
        Self::new(BackoffPolicy::default(), seed)
    }

    /// Returns the next LCG-generated value in `[0, 1)`.
    fn next_rand(&mut self) -> f64 {
        // Knuth's MMIX constants
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Extract upper 53 bits for float
        let hi = self.rng_state >> 11;
        hi as f64 / (1u64 << 53) as f64
    }

    /// Computes the (possibly jittered) delay for the given attempt number (1-based).
    ///
    /// Returns `None` when the attempt exceeds the allowed limit.
    pub fn delay_for_attempt(&mut self, attempt: u32) -> Option<Duration> {
        let det = self.policy.deterministic_delay(attempt)?;
        let delay = match self.policy.jitter {
            JitterStrategy::None => det,
            JitterStrategy::Full => {
                let r = self.next_rand();
                Duration::from_micros((det.as_micros() as f64 * r).round() as u64)
            }
            JitterStrategy::Equal => {
                let half = det.as_micros() as f64 / 2.0;
                let r = self.next_rand();
                Duration::from_micros((half + r * half).round() as u64)
            }
        };
        Some(delay)
    }

    /// Returns `true` if another retry is permitted.
    #[must_use]
    pub fn should_retry(&self, attempt: u32) -> bool {
        self.policy.should_retry(attempt)
    }

    /// Returns a reference to the underlying policy.
    #[must_use]
    pub fn policy(&self) -> &BackoffPolicy {
        &self.policy
    }
}

/// Computes the total delay over all retry attempts up to `n` under a policy
/// (without jitter).  Useful for estimating worst-case wait time.
///
/// Returns the sum of delays from attempt 1 through `n` (or until `max_attempts`).
#[must_use]
pub fn total_delay_estimate(policy: &BackoffPolicy, n: u32) -> Duration {
    let limit = policy
        .max_attempts
        .map_or(n, |max| n.min(max));
    (1..=limit)
        .filter_map(|a| policy.deterministic_delay(a))
        .fold(Duration::ZERO, |acc, d| acc + d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------- BackoffPolicy ----------

    #[test]
    fn test_default_policy_parameters() {
        let p = BackoffPolicy::default();
        assert_eq!(p.base_delay, Duration::from_millis(500));
        assert!((p.multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(p.max_delay, Duration::from_secs(30));
        assert_eq!(p.max_attempts, Some(5));
        assert_eq!(p.jitter, JitterStrategy::Full);
    }

    #[test]
    fn test_deterministic_delay_first_attempt() {
        let p = BackoffPolicy::new(Duration::from_secs(1), 2.0);
        let d = p.deterministic_delay(1).expect("attempt 1 is always valid");
        assert_eq!(d, Duration::from_secs(1));
    }

    #[test]
    fn test_deterministic_delay_doubles_each_attempt() {
        let p = BackoffPolicy::new(Duration::from_secs(1), 2.0)
            .with_max_delay(Duration::from_secs(1000))
            .with_max_attempts(10);
        let d1 = p.deterministic_delay(1).expect("attempt 1 valid");
        let d2 = p.deterministic_delay(2).expect("attempt 2 valid");
        let d3 = p.deterministic_delay(3).expect("attempt 3 valid");
        assert_eq!(d2, d1 * 2);
        assert_eq!(d3, d1 * 4);
    }

    #[test]
    fn test_deterministic_delay_capped_at_max() {
        let p = BackoffPolicy::new(Duration::from_secs(1), 4.0)
            .with_max_delay(Duration::from_secs(5))
            .with_max_attempts(20);
        // Attempt 5 would be 4^4 = 256 seconds, but max is 5
        let d = p.deterministic_delay(5).expect("attempt 5 valid");
        assert_eq!(d, Duration::from_secs(5));
    }

    #[test]
    fn test_deterministic_delay_none_beyond_max_attempts() {
        let p = BackoffPolicy::default(); // max_attempts = 5
        assert!(p.deterministic_delay(6).is_none());
    }

    #[test]
    fn test_should_retry_within_limit() {
        let p = BackoffPolicy::default(); // max_attempts = 5
        assert!(p.should_retry(1));
        assert!(p.should_retry(5));
        assert!(!p.should_retry(6));
    }

    #[test]
    fn test_unlimited_policy_always_retries() {
        let p = BackoffPolicy::default().unlimited();
        assert!(p.should_retry(1000));
        assert!(p.deterministic_delay(1000).is_some());
    }

    // ---------- RetryState ----------

    #[test]
    fn test_retry_state_initial() {
        let s = RetryState::with_default_policy();
        assert_eq!(s.attempt, 0);
        assert_eq!(s.total_delay, Duration::ZERO);
        assert!(s.can_retry());
    }

    #[test]
    fn test_retry_state_record_failure_increments_attempt() {
        let mut s = RetryState::with_default_policy();
        s.record_failure().expect("first failure allowed");
        assert_eq!(s.attempt, 1);
    }

    #[test]
    fn test_retry_state_record_failure_accumulates_delay() {
        let policy = BackoffPolicy::new(Duration::from_millis(100), 2.0)
            .with_max_delay(Duration::from_secs(60))
            .with_max_attempts(5)
            .with_jitter(JitterStrategy::None);
        let mut s = RetryState::new(policy);

        let d1 = s.record_failure().expect("attempt 1");
        let d2 = s.record_failure().expect("attempt 2");
        assert_eq!(d1, Duration::from_millis(100));
        assert_eq!(d2, Duration::from_millis(200));
        assert_eq!(s.total_delay, Duration::from_millis(300));
    }

    #[test]
    fn test_retry_state_none_after_exhaustion() {
        let policy = BackoffPolicy::default().with_max_attempts(2);
        let mut s = RetryState::new(policy);
        s.record_failure().expect("attempt 1");
        s.record_failure().expect("attempt 2");
        assert!(s.record_failure().is_none());
    }

    #[test]
    fn test_retry_state_reset() {
        let mut s = RetryState::with_default_policy();
        s.record_failure().expect("attempt 1");
        s.reset();
        assert_eq!(s.attempt, 0);
        assert_eq!(s.total_delay, Duration::ZERO);
    }

    #[test]
    fn test_retry_state_remaining_attempts() {
        let policy = BackoffPolicy::default().with_max_attempts(3);
        let mut s = RetryState::new(policy);
        assert_eq!(s.remaining_attempts(), Some(3));
        s.record_failure().expect("attempt 1");
        assert_eq!(s.remaining_attempts(), Some(2));
    }

    // ---------- RetryScheduler ----------

    #[test]
    fn test_scheduler_no_jitter_deterministic() {
        let policy = BackoffPolicy::new(Duration::from_millis(200), 2.0)
            .with_jitter(JitterStrategy::None)
            .with_max_attempts(5)
            .with_max_delay(Duration::from_secs(60));
        let mut sched = RetryScheduler::new(policy, 42);
        let d1 = sched.delay_for_attempt(1).expect("attempt 1");
        let d2 = sched.delay_for_attempt(2).expect("attempt 2");
        assert_eq!(d1, Duration::from_millis(200));
        assert_eq!(d2, Duration::from_millis(400));
    }

    #[test]
    fn test_scheduler_full_jitter_within_bounds() {
        let base = Duration::from_secs(1);
        let policy = BackoffPolicy::new(base, 2.0)
            .with_jitter(JitterStrategy::Full)
            .with_max_delay(Duration::from_secs(60))
            .with_max_attempts(10);
        let mut sched = RetryScheduler::new(policy, 0xfeed_beef);
        for attempt in 1..=5 {
            let d = sched
                .delay_for_attempt(attempt)
                .expect("should succeed in test");
            let det = BackoffPolicy::new(base, 2.0)
                .with_max_delay(Duration::from_secs(60))
                .with_max_attempts(10)
                .deterministic_delay(attempt)
                .expect("should succeed in test");
            assert!(d <= det, "jittered delay {d:?} must be ≤ det {det:?}");
        }
    }

    #[test]
    fn test_scheduler_returns_none_beyond_limit() {
        let policy = BackoffPolicy::default().with_max_attempts(3);
        let mut sched = RetryScheduler::new(policy, 1);
        assert!(sched.delay_for_attempt(4).is_none());
    }

    // ---------- total_delay_estimate ----------

    #[test]
    fn test_total_delay_estimate_no_jitter() {
        let policy = BackoffPolicy::new(Duration::from_secs(1), 2.0)
            .with_jitter(JitterStrategy::None)
            .with_max_delay(Duration::from_secs(1000))
            .with_max_attempts(4);
        // Attempts: 1s + 2s + 4s + 8s = 15s
        let total = total_delay_estimate(&policy, 4);
        assert_eq!(total, Duration::from_secs(15));
    }

    #[test]
    fn test_total_delay_capped_by_max_attempts() {
        let policy = BackoffPolicy::new(Duration::from_secs(1), 2.0)
            .with_jitter(JitterStrategy::None)
            .with_max_delay(Duration::from_secs(1000))
            .with_max_attempts(2);
        // Only 2 attempts even though we ask for 10
        let total = total_delay_estimate(&policy, 10);
        assert_eq!(total, Duration::from_secs(3)); // 1 + 2
    }

    #[test]
    fn test_equal_jitter_midpoint_range() {
        let base = Duration::from_secs(2);
        let policy = BackoffPolicy::new(base, 1.0)
            .with_jitter(JitterStrategy::Equal)
            .with_max_delay(Duration::from_secs(60))
            .with_max_attempts(10);
        let mut sched = RetryScheduler::new(policy, 12345);
        for _ in 0..20 {
            let d = sched
                .delay_for_attempt(1)
                .expect("should succeed in test");
            assert!(d >= Duration::from_secs(1), "equal-jitter low bound");
            assert!(d <= Duration::from_secs(2), "equal-jitter high bound");
        }
    }
}
