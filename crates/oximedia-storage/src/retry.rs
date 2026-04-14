//! Generic async retry infrastructure with exponential back-off and jitter.
//!
//! Provides:
//! - `RetryableError` trait to indicate whether an error is transient.
//! - `with_retry` async helper that wraps any fallible async closure.
//! - [`RetryConfig`] re-exported from the crate root for uniform configuration.
//!
//! # Jitter
//!
//! Rather than depending on the `rand` crate, the jitter fraction is derived
//! from a deterministic hash of `(attempt, timestamp_nanos)`.  This keeps the
//! computation pure-Rust and dependency-free while still spreading retry
//! bursts across time.
#![allow(dead_code)]

pub use crate::RetryConfig;
use std::future::Future;
use std::time::SystemTime;

// ─── RetryableError trait ─────────────────────────────────────────────────────

/// Implemented by error types that can signal whether an operation should be
/// retried automatically.
pub trait RetryableError {
    /// Return `true` if the error represents a transient failure that may
    /// resolve on a subsequent attempt.
    fn is_retryable(&self) -> bool;
}

// ─── Jitter helper ────────────────────────────────────────────────────────────

/// Compute a deterministic pseudo-random fraction in `[0.0, 1.0)` from
/// `attempt` and the current nanosecond timestamp.
///
/// Uses two rounds of Murmur3-inspired bit-mixing so that:
/// - Different attempt numbers at the same time produce different values.
/// - The same attempt number at different times produces different values.
#[must_use]
fn jitter_fraction(attempt: u32) -> f64 {
    let now_ns = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0) as u64;

    let mut v = (attempt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    v ^= now_ns.wrapping_mul(0x6c62_272e_07bb_0142);
    v ^= v >> 30;
    v = v.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94d0_49bb_1331_11eb);
    v ^= v >> 31;
    // Map to [0.0, 1.0)
    (v as f64) / (u64::MAX as f64 + 1.0)
}

// ─── Delay computation ────────────────────────────────────────────────────────

/// Compute the wait in milliseconds before attempt `n` (0-indexed).
///
/// Formula: `min(initial * multiplier^n, max) * (1 + jitter_factor * rand)`
///
/// where `rand` is a deterministic pseudo-random value in `[0, 1)`.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
pub fn compute_delay_ms(config: &RetryConfig, attempt: u32) -> u64 {
    let multiplier = config.backoff_multiplier.max(1.0);
    let base = config.initial_backoff_ms as f64 * multiplier.powi(attempt as i32);
    let capped = base.min(config.max_backoff_ms as f64);
    let jitter = config.jitter_factor.clamp(0.0, 1.0);
    let rand_frac = jitter_fraction(attempt);
    let with_jitter = capped * (1.0 + jitter * rand_frac);
    with_jitter.round() as u64
}

// ─── with_retry ───────────────────────────────────────────────────────────────

/// Execute `f` up to `config.max_retries + 1` times, waiting between attempts
/// with exponential back-off and jitter.
///
/// On each failure the error's [`RetryableError::is_retryable`] result is
/// consulted.  If the error is **not** retryable the function returns
/// immediately without sleeping.
///
/// # Type Parameters
/// * `F` — factory closure that produces a fresh `Future` for each attempt.
/// * `Fut` — the `Future` returned by `F`.
/// * `T` — the success type.
/// * `E` — the error type; must implement [`RetryableError`].
pub async fn with_retry<F, Fut, T, E>(config: &RetryConfig, mut f: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: RetryableError,
{
    let max_attempts = (config.max_retries as u64) + 1;

    for attempt in 0..max_attempts {
        match f().await {
            Ok(val) => return Ok(val),
            Err(err) => {
                let is_last = attempt + 1 >= max_attempts;
                if is_last || !err.is_retryable() {
                    return Err(err);
                }

                let delay_ms = compute_delay_ms(config, attempt as u32);
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }

    // Unreachable: the loop always returns on the last attempt.
    unreachable!("with_retry loop should have returned")
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    // Minimal test error type
    #[derive(Debug, PartialEq)]
    enum TestError {
        Transient(String),
        Permanent(String),
    }

    impl RetryableError for TestError {
        fn is_retryable(&self) -> bool {
            matches!(self, Self::Transient(_))
        }
    }

    impl std::fmt::Display for TestError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Transient(m) | Self::Permanent(m) => write!(f, "{m}"),
            }
        }
    }

    fn fast_config() -> RetryConfig {
        RetryConfig {
            max_retries: 3,
            initial_backoff_ms: 1,
            max_backoff_ms: 10,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
            retry_on_transient_only: true,
        }
    }

    // ── compute_delay_ms tests ────────────────────────────────────────────────

    #[test]
    fn test_backoff_increases_with_attempt() {
        let cfg = RetryConfig {
            initial_backoff_ms: 100,
            backoff_multiplier: 2.0,
            max_backoff_ms: 60_000,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        let d0 = compute_delay_ms(&cfg, 0);
        let d1 = compute_delay_ms(&cfg, 1);
        let d2 = compute_delay_ms(&cfg, 2);
        // Without jitter: 100, 200, 400
        assert_eq!(d0, 100);
        assert_eq!(d1, 200);
        assert_eq!(d2, 400);
    }

    #[test]
    fn test_backoff_capped_at_max() {
        let cfg = RetryConfig {
            initial_backoff_ms: 100,
            backoff_multiplier: 2.0,
            max_backoff_ms: 500,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        let d10 = compute_delay_ms(&cfg, 10);
        // 100 * 2^10 = 102_400 → capped at 500
        assert_eq!(d10, 500);
    }

    #[test]
    fn test_backoff_multiplier_less_than_one_clamped() {
        let cfg = RetryConfig {
            initial_backoff_ms: 100,
            backoff_multiplier: 0.5, // should clamp to 1.0
            max_backoff_ms: 10_000,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        // With multiplier clamped to 1.0: always 100ms
        let d0 = compute_delay_ms(&cfg, 0);
        let d5 = compute_delay_ms(&cfg, 5);
        assert_eq!(d0, 100);
        assert_eq!(d5, 100);
    }

    #[test]
    fn test_jitter_within_bounds() {
        let cfg = RetryConfig {
            initial_backoff_ms: 1000,
            backoff_multiplier: 1.0,
            max_backoff_ms: 10_000,
            jitter_factor: 0.1,
            ..RetryConfig::default()
        };
        for attempt in 0u32..10 {
            let d = compute_delay_ms(&cfg, attempt);
            // base = 1000, jitter_factor = 0.1 → delay in [1000, 1100]
            assert!(
                d >= 1000 && d <= 1110,
                "attempt {attempt}: delay {d} out of [1000, 1110]"
            );
        }
    }

    #[test]
    fn test_jitter_zero_gives_exact_delay() {
        let cfg = RetryConfig {
            initial_backoff_ms: 200,
            backoff_multiplier: 2.0,
            max_backoff_ms: 60_000,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        assert_eq!(compute_delay_ms(&cfg, 0), 200);
        assert_eq!(compute_delay_ms(&cfg, 1), 400);
    }

    #[test]
    fn test_jitter_fraction_is_in_unit_interval() {
        for attempt in 0u32..20 {
            let f = jitter_fraction(attempt);
            assert!(
                (0.0..1.0).contains(&f),
                "attempt {attempt}: fraction {f} out of [0,1)"
            );
        }
    }

    // ── with_retry async tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_with_retry_succeeds_immediately() {
        let cfg = fast_config();
        let result: Result<i32, TestError> = with_retry(&cfg, || async { Ok(42) }).await;
        assert_eq!(result, Ok(42));
    }

    #[tokio::test]
    async fn test_with_retry_retries_transient_then_succeeds() {
        let cfg = fast_config();
        let attempts = Arc::new(AtomicU32::new(0));
        let a = attempts.clone();

        let result: Result<i32, TestError> = with_retry(&cfg, || {
            let a = a.clone();
            async move {
                let n = a.fetch_add(1, Ordering::SeqCst);
                if n < 2 {
                    Err(TestError::Transient("transient".into()))
                } else {
                    Ok(99)
                }
            }
        })
        .await;

        assert_eq!(result, Ok(99));
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_with_retry_non_retryable_errors_fail_fast() {
        let cfg = fast_config();
        let attempts = Arc::new(AtomicU32::new(0));
        let a = attempts.clone();

        let result: Result<i32, TestError> = with_retry(&cfg, || {
            let a = a.clone();
            async move {
                a.fetch_add(1, Ordering::SeqCst);
                Err(TestError::Permanent("fatal".into()))
            }
        })
        .await;

        assert!(result.is_err());
        // Should only have tried once (non-retryable).
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_retry_exhausts_max_retries() {
        let cfg = RetryConfig {
            max_retries: 2,
            ..fast_config()
        };
        let attempts = Arc::new(AtomicU32::new(0));
        let a = attempts.clone();

        let result: Result<i32, TestError> = with_retry(&cfg, || {
            let a = a.clone();
            async move {
                a.fetch_add(1, Ordering::SeqCst);
                Err(TestError::Transient("always fails".into()))
            }
        })
        .await;

        assert!(result.is_err());
        // 1 initial + 2 retries = 3 total attempts.
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }
}
