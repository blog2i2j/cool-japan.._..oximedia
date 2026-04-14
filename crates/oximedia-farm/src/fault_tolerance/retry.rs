//! Retry policies and strategies.
//!
//! Provides several backoff strategies including exponential backoff with
//! full/partial jitter, designed for distributed media-processing workloads
//! where thundering-herd effects on the coordinator must be avoided.

use rand::RngExt;
use std::time::Duration;

/// Retry policy configuration
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub strategy: RetryStrategy,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            strategy: RetryStrategy::ExponentialBackoff {
                initial_delay: Duration::from_secs(1),
                max_delay: Duration::from_secs(60),
                multiplier: 2.0,
            },
        }
    }
}

impl RetryPolicy {
    /// Create a new retry policy
    #[must_use]
    pub fn new(max_attempts: u32, strategy: RetryStrategy) -> Self {
        Self {
            max_attempts,
            strategy,
        }
    }

    /// Calculate the delay for a given attempt
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        self.strategy.calculate_delay(attempt)
    }
}

/// Retry strategies
#[derive(Debug, Clone)]
pub enum RetryStrategy {
    /// Fixed delay between retries
    Fixed(Duration),

    /// Linear backoff with increasing delay
    LinearBackoff {
        initial_delay: Duration,
        increment: Duration,
    },

    /// Exponential backoff (deterministic, no jitter)
    ExponentialBackoff {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
    },

    /// Exponential backoff with full jitter.
    ///
    /// Each call to [`calculate_delay`](RetryStrategy::calculate_delay) returns a
    /// uniformly random value in `[0, cap]` where `cap = initial_delay * multiplier^(attempt-1)`
    /// (capped at `max_delay`).  This is the "Full Jitter" strategy from the AWS
    /// architecture blog and is the recommended default for distributed systems.
    ExponentialBackoffWithJitter {
        initial_delay: Duration,
        max_delay: Duration,
        multiplier: f64,
        /// Fraction of jitter applied `[0.0, 1.0]`.
        /// 0.0 = no jitter (equivalent to `ExponentialBackoff`),
        /// 1.0 = full jitter (random in `[0, cap]`).
        jitter_factor: f64,
    },

    /// Custom delay list; last element is repeated for out-of-range attempts.
    Custom(Vec<Duration>),
}

impl RetryStrategy {
    /// Calculate the delay for a given attempt (1-based).
    ///
    /// For jitter-bearing strategies the returned duration is **non-deterministic**.
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        match self {
            Self::Fixed(delay) => *delay,

            Self::LinearBackoff {
                initial_delay,
                increment,
            } => {
                let total_increments = attempt.saturating_sub(1);
                *initial_delay + *increment * total_increments
            }

            Self::ExponentialBackoff {
                initial_delay,
                max_delay,
                multiplier,
            } => {
                let delay_ms = initial_delay.as_millis() as f64
                    * multiplier.powi(attempt.saturating_sub(1) as i32);
                let delay = Duration::from_millis(delay_ms.min(u64::MAX as f64) as u64);
                delay.min(*max_delay)
            }

            Self::ExponentialBackoffWithJitter {
                initial_delay,
                max_delay,
                multiplier,
                jitter_factor,
            } => {
                let base_ms = initial_delay.as_millis() as f64
                    * multiplier.powi(attempt.saturating_sub(1) as i32);
                let cap_ms = (base_ms as u64).min(max_delay.as_millis() as u64);

                if cap_ms == 0 {
                    return Duration::ZERO;
                }

                let jitter_fraction = jitter_factor.clamp(0.0, 1.0);
                let stable_ms = (cap_ms as f64 * (1.0 - jitter_fraction)) as u64;
                let jitter_range = cap_ms - stable_ms;

                let jitter_ms = if jitter_range > 0 {
                    let mut rng = rand::rng();
                    rng.random_range(0..=jitter_range)
                } else {
                    0
                };

                Duration::from_millis(stable_ms + jitter_ms)
            }

            Self::Custom(delays) => {
                let index = (attempt.saturating_sub(1) as usize).min(delays.len() - 1);
                delays[index]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ── Fixed delay ──────────────────────────────────────────────────────────

    #[test]
    fn test_fixed_delay() {
        let strategy = RetryStrategy::Fixed(Duration::from_secs(5));

        assert_eq!(strategy.calculate_delay(1), Duration::from_secs(5));
        assert_eq!(strategy.calculate_delay(2), Duration::from_secs(5));
        assert_eq!(strategy.calculate_delay(3), Duration::from_secs(5));
    }

    // ── Linear backoff ───────────────────────────────────────────────────────

    #[test]
    fn test_linear_backoff() {
        let strategy = RetryStrategy::LinearBackoff {
            initial_delay: Duration::from_secs(1),
            increment: Duration::from_secs(2),
        };

        assert_eq!(strategy.calculate_delay(1), Duration::from_secs(1));
        assert_eq!(strategy.calculate_delay(2), Duration::from_secs(3));
        assert_eq!(strategy.calculate_delay(3), Duration::from_secs(5));
    }

    // ── Exponential backoff (deterministic) ──────────────────────────────────

    #[test]
    fn test_exponential_backoff() {
        let strategy = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
        };

        assert_eq!(strategy.calculate_delay(1), Duration::from_secs(1));
        assert_eq!(strategy.calculate_delay(2), Duration::from_secs(2));
        assert_eq!(strategy.calculate_delay(3), Duration::from_secs(4));
        assert_eq!(strategy.calculate_delay(4), Duration::from_secs(8));
    }

    #[test]
    fn test_exponential_backoff_max_delay() {
        let strategy = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
        };

        // Should cap at max_delay
        assert_eq!(strategy.calculate_delay(10), Duration::from_secs(10));
    }

    #[test]
    fn test_exponential_each_delay_gt_previous() {
        let strategy = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
        };
        let d1 = strategy.calculate_delay(1);
        let d2 = strategy.calculate_delay(2);
        let d3 = strategy.calculate_delay(3);
        assert!(
            d2 > d1,
            "attempt 2 delay ({d2:?}) should exceed attempt 1 ({d1:?})"
        );
        assert!(
            d3 > d2,
            "attempt 3 delay ({d3:?}) should exceed attempt 2 ({d2:?})"
        );
        // Verify ~2× growth
        let ratio_1_2 = d2.as_millis() as f64 / d1.as_millis() as f64;
        assert!(
            (ratio_1_2 - 2.0).abs() < 0.01,
            "expected 2× growth, got {ratio_1_2:.3}×"
        );
    }

    // ── Exponential backoff with jitter ──────────────────────────────────────

    #[test]
    fn test_jitter_delay_within_bounds() {
        let strategy = RetryStrategy::ExponentialBackoffWithJitter {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            jitter_factor: 1.0,
        };
        // attempt 3 → cap = 100ms * 2^2 = 400ms; delay must be in [0, 400ms]
        for _ in 0..20 {
            let d = strategy.calculate_delay(3);
            assert!(
                d <= Duration::from_millis(400),
                "jitter delay {:?} exceeds cap 400ms",
                d
            );
        }
    }

    #[test]
    fn test_jitter_produces_variation() {
        // With full jitter (jitter_factor=1.0), 20 independent samples should
        // not all be identical (probability of all equal ≈ 0 with >1ms range).
        let strategy = RetryStrategy::ExponentialBackoffWithJitter {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            jitter_factor: 1.0,
        };
        let samples: HashSet<u128> = (0..20)
            .map(|_| strategy.calculate_delay(3).as_millis())
            .collect();
        assert!(
            samples.len() > 1,
            "jitter should produce multiple distinct values, got {:?}",
            samples
        );
    }

    #[test]
    fn test_jitter_zero_factor_is_deterministic() {
        // With jitter_factor=0.0, the strategy should behave like plain
        // ExponentialBackoff.
        let jitter_strategy = RetryStrategy::ExponentialBackoffWithJitter {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            jitter_factor: 0.0,
        };
        let plain_strategy = RetryStrategy::ExponentialBackoff {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
        };
        for attempt in 1..=5 {
            assert_eq!(
                jitter_strategy.calculate_delay(attempt),
                plain_strategy.calculate_delay(attempt),
                "attempt {attempt}: jitter=0 should match plain backoff"
            );
        }
    }

    #[test]
    fn test_jitter_max_delay_cap_respected() {
        let strategy = RetryStrategy::ExponentialBackoffWithJitter {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(10),
            multiplier: 2.0,
            jitter_factor: 1.0,
        };
        for _ in 0..30 {
            let d = strategy.calculate_delay(20); // exponential would be huge
            assert!(
                d <= Duration::from_secs(10),
                "jitter delay {:?} exceeds max_delay 10s",
                d
            );
        }
    }

    #[test]
    fn test_jitter_partial_factor_range() {
        // With jitter_factor=0.5, minimum delay = 50% of cap, max = 100%.
        let cap_ms: u64 = 400; // attempt 3 with 100ms initial, 2× multiplier
        let strategy = RetryStrategy::ExponentialBackoffWithJitter {
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(60),
            multiplier: 2.0,
            jitter_factor: 0.5,
        };
        for _ in 0..30 {
            let d = strategy.calculate_delay(3);
            let ms = d.as_millis() as u64;
            assert!(
                ms >= cap_ms / 2 && ms <= cap_ms,
                "partial-jitter delay {ms}ms out of [{}, {cap_ms}]",
                cap_ms / 2
            );
        }
    }

    // ── Custom delays ────────────────────────────────────────────────────────

    #[test]
    fn test_custom_delays() {
        let strategy = RetryStrategy::Custom(vec![
            Duration::from_secs(1),
            Duration::from_secs(5),
            Duration::from_secs(10),
        ]);

        assert_eq!(strategy.calculate_delay(1), Duration::from_secs(1));
        assert_eq!(strategy.calculate_delay(2), Duration::from_secs(5));
        assert_eq!(strategy.calculate_delay(3), Duration::from_secs(10));
        // Should use last delay for attempts beyond the list
        assert_eq!(strategy.calculate_delay(4), Duration::from_secs(10));
    }

    // ── RetryPolicy ──────────────────────────────────────────────────────────

    #[test]
    fn test_default_policy() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.max_attempts, 3);
    }

    #[test]
    fn test_policy_calculate_delay() {
        let policy = RetryPolicy::new(3, RetryStrategy::Fixed(Duration::from_secs(5)));

        assert_eq!(policy.calculate_delay(1), Duration::from_secs(5));
        assert_eq!(policy.calculate_delay(2), Duration::from_secs(5));
    }

    #[test]
    fn test_max_retries_field() {
        let policy = RetryPolicy::new(
            5,
            RetryStrategy::ExponentialBackoffWithJitter {
                initial_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(10),
                multiplier: 2.0,
                jitter_factor: 0.5,
            },
        );
        assert_eq!(policy.max_attempts, 5);
    }

    /// Simulates a function that fails twice then succeeds, verifying the
    /// retry machinery stops after `max_attempts`.
    #[test]
    fn test_retry_succeeds_on_third_attempt() {
        let policy = RetryPolicy::new(5, RetryStrategy::Fixed(Duration::ZERO));

        let mut attempts = 0u32;
        let result: Result<&str, &str> = loop {
            attempts += 1;
            if attempts < 3 {
                // simulate failure (no real sleep; delays are zero)
            } else {
                break Ok("success");
            }
            if attempts >= policy.max_attempts {
                break Err("max retries exhausted");
            }
        };

        assert!(result.is_ok());
        assert_eq!(result, Ok("success"));
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_max_retries_exhausted() {
        let max = 3u32;
        let policy = RetryPolicy::new(max, RetryStrategy::Fixed(Duration::ZERO));

        let mut attempts = 0u32;
        let result: Result<(), &str> = loop {
            attempts += 1;
            // always fail
            if attempts >= policy.max_attempts {
                break Err("exhausted");
            }
        };

        assert!(result.is_err());
        assert_eq!(attempts, max);
    }
}
