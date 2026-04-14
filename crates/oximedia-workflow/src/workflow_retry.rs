#![allow(dead_code)]
//! Advanced retry orchestration for workflow tasks.
//!
//! This module provides sophisticated retry strategies beyond simple count-based
//! retries, including exponential backoff, circuit breakers, and retry budgets
//! that prevent cascading failures in large workflow DAGs.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Backoff strategy for retry delays.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Constant(Duration),
    /// Linearly increasing delay: `base * attempt`.
    Linear {
        /// Base delay multiplied by the attempt number.
        base: Duration,
    },
    /// Exponentially increasing delay: `base * 2^attempt`, capped at `max`.
    Exponential {
        /// Initial delay.
        base: Duration,
        /// Maximum delay cap.
        max: Duration,
    },
    /// Fibonacci-sequence based delay: fib(attempt) * base.
    Fibonacci {
        /// Base unit of delay.
        base: Duration,
    },
}

impl BackoffStrategy {
    /// Compute the delay for a given attempt number (0-indexed).
    #[allow(clippy::cast_precision_loss)]
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            Self::Constant(d) => *d,
            Self::Linear { base } => *base * attempt.max(1),
            Self::Exponential { base, max } => {
                let multiplier = 2u64.saturating_pow(attempt);
                let delay = base.saturating_mul(multiplier as u32);
                if delay > *max {
                    *max
                } else {
                    delay
                }
            }
            Self::Fibonacci { base } => {
                let fib = fibonacci(attempt);
                base.saturating_mul(fib)
            }
        }
    }
}

/// Compute the n-th Fibonacci number (capped to avoid overflow).
fn fibonacci(n: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut a: u32 = 1;
    let mut b: u32 = 1;
    for _ in 1..n {
        let next = a.saturating_add(b);
        a = b;
        b = next;
    }
    b
}

/// Circuit breaker state for a task or task group.
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    /// Normal operation; requests pass through.
    Closed,
    /// Failures exceeded threshold; all requests are rejected.
    Open {
        /// When the circuit was opened.
        opened_at: Instant,
        /// How long to wait before transitioning to half-open.
        cooldown: Duration,
    },
    /// Testing phase; limited requests allowed.
    HalfOpen {
        /// Number of successes required to close the circuit.
        successes_needed: u32,
        /// Current number of consecutive successes.
        current_successes: u32,
    },
}

/// Circuit breaker configuration and state.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Failure threshold before opening the circuit.
    pub failure_threshold: u32,
    /// Duration to wait before half-opening.
    pub cooldown: Duration,
    /// Number of successes in half-open state to close.
    pub recovery_threshold: u32,
    /// Current consecutive failure count.
    pub consecutive_failures: u32,
    /// Current state.
    pub state: CircuitState,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with the given thresholds.
    pub fn new(failure_threshold: u32, cooldown: Duration, recovery_threshold: u32) -> Self {
        Self {
            failure_threshold,
            cooldown,
            recovery_threshold,
            consecutive_failures: 0,
            state: CircuitState::Closed,
        }
    }

    /// Check whether a request should be allowed.
    pub fn allow_request(&mut self) -> bool {
        match &self.state {
            CircuitState::Closed => true,
            CircuitState::Open {
                opened_at,
                cooldown,
            } => {
                if opened_at.elapsed() >= *cooldown {
                    self.state = CircuitState::HalfOpen {
                        successes_needed: self.recovery_threshold,
                        current_successes: 0,
                    };
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen { .. } => true,
        }
    }

    /// Record a successful request.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        match &self.state {
            CircuitState::HalfOpen {
                successes_needed,
                current_successes,
            } => {
                let next = current_successes + 1;
                if next >= *successes_needed {
                    self.state = CircuitState::Closed;
                } else {
                    self.state = CircuitState::HalfOpen {
                        successes_needed: *successes_needed,
                        current_successes: next,
                    };
                }
            }
            _ => {
                self.state = CircuitState::Closed;
            }
        }
    }

    /// Record a failed request.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        match &self.state {
            CircuitState::Closed => {
                if self.consecutive_failures >= self.failure_threshold {
                    self.state = CircuitState::Open {
                        opened_at: Instant::now(),
                        cooldown: self.cooldown,
                    };
                }
            }
            CircuitState::HalfOpen { .. } => {
                // Any failure in half-open re-opens the circuit.
                self.state = CircuitState::Open {
                    opened_at: Instant::now(),
                    cooldown: self.cooldown,
                };
            }
            CircuitState::Open { .. } => {}
        }
    }
}

/// A budget that limits how many retries can occur within a time window.
#[derive(Debug, Clone)]
pub struct RetryBudget {
    /// Maximum retries allowed within the window.
    pub max_retries: u32,
    /// The time window length.
    pub window: Duration,
    /// Timestamps of retries within the window.
    entries: Vec<Instant>,
}

impl RetryBudget {
    /// Create a new retry budget.
    pub fn new(max_retries: u32, window: Duration) -> Self {
        Self {
            max_retries,
            window,
            entries: Vec::new(),
        }
    }

    /// Check if a retry is allowed and, if so, record it.
    pub fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        self.entries
            .retain(|t| now.duration_since(*t) < self.window);
        if self.entries.len() < self.max_retries as usize {
            self.entries.push(now);
            true
        } else {
            false
        }
    }

    /// Return how many retries remain in the current window.
    pub fn remaining(&mut self) -> u32 {
        let now = Instant::now();
        self.entries
            .retain(|t| now.duration_since(*t) < self.window);
        self.max_retries.saturating_sub(self.entries.len() as u32)
    }
}

/// Outcome of a single retry attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum RetryOutcome {
    /// The task succeeded.
    Success,
    /// The task failed but should be retried.
    RetryableFailure(String),
    /// The task failed and should not be retried.
    PermanentFailure(String),
}

/// An orchestrator that combines backoff, circuit breaker, and retry budget.
#[derive(Debug)]
pub struct RetryOrchestrator {
    /// Backoff strategy.
    pub backoff: BackoffStrategy,
    /// Maximum number of attempts.
    pub max_attempts: u32,
    /// Optional circuit breaker per task key.
    breakers: HashMap<String, CircuitBreaker>,
    /// Default circuit breaker config.
    breaker_config: Option<(u32, Duration, u32)>,
    /// Global retry budget.
    pub budget: Option<RetryBudget>,
    /// Current attempt count per task.
    attempts: HashMap<String, u32>,
}

impl RetryOrchestrator {
    /// Create a new retry orchestrator with the given backoff and max attempts.
    pub fn new(backoff: BackoffStrategy, max_attempts: u32) -> Self {
        Self {
            backoff,
            max_attempts,
            breakers: HashMap::new(),
            breaker_config: None,
            budget: None,
            attempts: HashMap::new(),
        }
    }

    /// Enable circuit breakers with the given configuration.
    pub fn with_circuit_breaker(
        mut self,
        failure_threshold: u32,
        cooldown: Duration,
        recovery_threshold: u32,
    ) -> Self {
        self.breaker_config = Some((failure_threshold, cooldown, recovery_threshold));
        self
    }

    /// Enable a global retry budget.
    pub fn with_budget(mut self, max_retries: u32, window: Duration) -> Self {
        self.budget = Some(RetryBudget::new(max_retries, window));
        self
    }

    /// Get or create a circuit breaker for the given task key.
    fn get_breaker(&mut self, key: &str) -> Option<&mut CircuitBreaker> {
        if let Some((ft, cd, rt)) = self.breaker_config {
            if !self.breakers.contains_key(key) {
                self.breakers
                    .insert(key.to_string(), CircuitBreaker::new(ft, cd, rt));
            }
            self.breakers.get_mut(key)
        } else {
            None
        }
    }

    /// Determine if a retry should be attempted for the given task.
    ///
    /// Returns `Some(delay)` if a retry is allowed, or `None` if retries
    /// are exhausted or blocked.
    pub fn should_retry(&mut self, task_key: &str, outcome: &RetryOutcome) -> Option<Duration> {
        if *outcome == RetryOutcome::Success {
            if let Some(breaker) = self.get_breaker(task_key) {
                breaker.record_success();
            }
            self.attempts.remove(task_key);
            return None;
        }

        if let RetryOutcome::PermanentFailure(_) = outcome {
            if let Some(breaker) = self.get_breaker(task_key) {
                breaker.record_failure();
            }
            self.attempts.remove(task_key);
            return None;
        }

        // Check circuit breaker.
        if let Some(breaker) = self.get_breaker(task_key) {
            breaker.record_failure();
            if !breaker.allow_request() {
                return None;
            }
        }

        // Check attempt count.
        let attempt = self.attempts.entry(task_key.to_string()).or_insert(0);
        *attempt += 1;
        if *attempt >= self.max_attempts {
            return None;
        }

        // Check budget.
        if let Some(ref mut budget) = self.budget {
            if !budget.try_acquire() {
                return None;
            }
        }

        Some(self.backoff.delay_for_attempt(*attempt))
    }

    /// Reset all state for a task key.
    pub fn reset(&mut self, task_key: &str) {
        self.attempts.remove(task_key);
        self.breakers.remove(task_key);
    }

    /// Get the current attempt count for a task.
    pub fn attempt_count(&self, task_key: &str) -> u32 {
        self.attempts.get(task_key).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_backoff() {
        let strategy = BackoffStrategy::Constant(Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(5));
        assert_eq!(strategy.delay_for_attempt(100), Duration::from_secs(5));
    }

    #[test]
    fn test_linear_backoff() {
        let strategy = BackoffStrategy::Linear {
            base: Duration::from_secs(2),
        };
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_secs(2));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_secs(6));
        assert_eq!(strategy.delay_for_attempt(5), Duration::from_secs(10));
    }

    #[test]
    fn test_exponential_backoff() {
        let strategy = BackoffStrategy::Exponential {
            base: Duration::from_millis(100),
            max: Duration::from_secs(10),
        };
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_millis(400));
        // Should be capped at max
        assert!(strategy.delay_for_attempt(30) <= Duration::from_secs(10));
    }

    #[test]
    fn test_fibonacci_backoff() {
        let strategy = BackoffStrategy::Fibonacci {
            base: Duration::from_millis(100),
        };
        // fib: 1, 1, 2, 3, 5, 8, 13, ...
        assert_eq!(strategy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(strategy.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(strategy.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(strategy.delay_for_attempt(3), Duration::from_millis(300));
        assert_eq!(strategy.delay_for_attempt(4), Duration::from_millis(500));
    }

    #[test]
    fn test_fibonacci_function() {
        assert_eq!(fibonacci(0), 1);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(2), 2);
        assert_eq!(fibonacci(3), 3);
        assert_eq!(fibonacci(4), 5);
        assert_eq!(fibonacci(5), 8);
        assert_eq!(fibonacci(6), 13);
    }

    #[test]
    fn test_circuit_breaker_closed() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(10), 2);
        assert!(cb.allow_request());
        cb.record_failure();
        assert!(cb.allow_request());
        cb.record_failure();
        assert!(cb.allow_request());
        // Third failure should open the circuit.
        cb.record_failure();
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_recovery() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(1), 1);
        cb.record_failure();
        cb.record_failure();
        // Circuit is open.
        assert!(!cb.allow_request());
        // Wait for cooldown.
        std::thread::sleep(Duration::from_millis(5));
        // Should transition to half-open.
        assert!(cb.allow_request());
        cb.record_success();
        assert_eq!(cb.state, CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(1), 2);
        cb.record_failure();
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(5));
        assert!(cb.allow_request()); // half-open
        cb.record_failure(); // failure in half-open -> re-open
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_retry_budget_allows() {
        let mut budget = RetryBudget::new(3, Duration::from_secs(60));
        assert!(budget.try_acquire());
        assert!(budget.try_acquire());
        assert!(budget.try_acquire());
        assert!(!budget.try_acquire());
    }

    #[test]
    fn test_retry_budget_remaining() {
        let mut budget = RetryBudget::new(5, Duration::from_secs(60));
        assert_eq!(budget.remaining(), 5);
        budget.try_acquire();
        budget.try_acquire();
        assert_eq!(budget.remaining(), 3);
    }

    #[test]
    fn test_orchestrator_basic_retry() {
        let mut orch =
            RetryOrchestrator::new(BackoffStrategy::Constant(Duration::from_millis(100)), 3);
        let fail = RetryOutcome::RetryableFailure("err".into());
        // First retry.
        let delay = orch.should_retry("task-1", &fail);
        assert!(delay.is_some());
        // Second retry.
        let delay = orch.should_retry("task-1", &fail);
        assert!(delay.is_some());
        // Third retry should be denied (max 3 attempts).
        let delay = orch.should_retry("task-1", &fail);
        assert!(delay.is_none());
    }

    #[test]
    fn test_orchestrator_success_resets() {
        let mut orch =
            RetryOrchestrator::new(BackoffStrategy::Constant(Duration::from_millis(50)), 3);
        let fail = RetryOutcome::RetryableFailure("err".into());
        orch.should_retry("t1", &fail);
        assert_eq!(orch.attempt_count("t1"), 1);
        orch.should_retry("t1", &RetryOutcome::Success);
        assert_eq!(orch.attempt_count("t1"), 0);
    }

    #[test]
    fn test_orchestrator_permanent_failure() {
        let mut orch =
            RetryOrchestrator::new(BackoffStrategy::Constant(Duration::from_millis(50)), 5);
        let result = orch.should_retry("t1", &RetryOutcome::PermanentFailure("fatal".into()));
        assert!(result.is_none());
    }

    #[test]
    fn test_orchestrator_with_budget() {
        let mut orch =
            RetryOrchestrator::new(BackoffStrategy::Constant(Duration::from_millis(50)), 100)
                .with_budget(2, Duration::from_secs(60));
        let fail = RetryOutcome::RetryableFailure("err".into());
        // Budget allows 2 retries.
        assert!(orch.should_retry("t1", &fail).is_some());
        assert!(orch.should_retry("t2", &fail).is_some());
        // Budget exhausted.
        assert!(orch.should_retry("t3", &fail).is_none());
    }

    #[test]
    fn test_orchestrator_reset() {
        let mut orch =
            RetryOrchestrator::new(BackoffStrategy::Constant(Duration::from_millis(50)), 3)
                .with_circuit_breaker(3, Duration::from_secs(10), 1);
        let fail = RetryOutcome::RetryableFailure("err".into());
        orch.should_retry("t1", &fail);
        orch.should_retry("t1", &fail);
        assert_eq!(orch.attempt_count("t1"), 2);
        orch.reset("t1");
        assert_eq!(orch.attempt_count("t1"), 0);
    }
}
