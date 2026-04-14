//! Segment retry policy with configurable exponential back-off.
//!
//! [`SegmentRetryPolicy`] provides a deterministic back-off strategy for
//! transient segment fetch failures.  No external time source or `rand` crate
//! is required: jitter is derived from the attempt number via a Weyl-sequence
//! hash mix, keeping the policy pure-Rust and `no_std`-compatible.
//!
//! # Example
//!
//! ```rust
//! use oximedia_stream::retry::SegmentRetryPolicy;
//!
//! let policy = SegmentRetryPolicy::new(5, 500);
//!
//! // Compute successive delays (ms) without sleeping
//! for attempt in 0..5 {
//!     let delay_ms = policy.next_delay(attempt);
//!     println!("attempt {attempt}: wait {delay_ms} ms");
//! }
//! ```

// ─── SegmentRetryPolicy ───────────────────────────────────────────────────────

/// Exponential back-off retry policy for segment fetch operations.
///
/// Delays grow as `base_delay_ms × 2^attempt`, capped at an internal maximum
/// of 60 000 ms.  A deterministic jitter of up to 20 % is added to smooth
/// retry storms.
#[derive(Debug, Clone)]
pub struct SegmentRetryPolicy {
    /// Maximum number of fetch attempts (including the first attempt).
    pub max_attempts: u32,
    /// Base delay in milliseconds before the first retry.
    pub base_delay_ms: u64,
    /// Hard ceiling for computed delays in milliseconds.
    pub max_delay_ms: u64,
    /// Jitter fraction in [0, 1].  0.0 disables jitter; 1.0 allows up to 100%
    /// additional delay.
    pub jitter_fraction: f64,
}

impl SegmentRetryPolicy {
    /// Create a new policy with the given `max_attempts` and `base_delay_ms`.
    ///
    /// Jitter defaults to 0.20 (20%) and the maximum delay defaults to 60 s.
    pub fn new(max_attempts: u32, base_delay_ms: u64) -> Self {
        Self {
            max_attempts,
            base_delay_ms,
            max_delay_ms: 60_000,
            jitter_fraction: 0.20,
        }
    }

    /// Override the maximum delay ceiling.
    pub fn with_max_delay(mut self, max_delay_ms: u64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }

    /// Override the jitter fraction (clamped to `[0.0, 1.0]`).
    pub fn with_jitter(mut self, fraction: f64) -> Self {
        self.jitter_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Compute the delay in milliseconds before retry `attempt` (0-indexed).
    ///
    /// `attempt = 0` corresponds to the first retry after the initial failure.
    ///
    /// Returns 0 when `attempt >= max_attempts` (no more retries should occur).
    #[must_use]
    pub fn next_delay(&self, attempt: u32) -> u64 {
        if attempt >= self.max_attempts {
            return 0;
        }
        // Exponential base: base_delay × 2^attempt, capped at max_delay.
        let exponent = attempt.min(31); // prevent overflow for large attempts
        let base = self
            .base_delay_ms
            .saturating_mul(1u64 << exponent)
            .min(self.max_delay_ms);

        // Deterministic jitter via a Weyl-sequence hash of the attempt number.
        let jitter_fraction = self.jitter_fraction.clamp(0.0, 1.0);
        let pseudo_rand = pseudo_rand_f64(attempt);
        let jitter_ms = (base as f64 * jitter_fraction * pseudo_rand) as u64;

        (base + jitter_ms).min(self.max_delay_ms)
    }

    /// Returns `true` if another retry is allowed after `attempt` failures.
    ///
    /// `attempt` is the number of failures so far (0 = no failures yet).
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_attempts
    }
}

/// Deterministic pseudo-random f64 in [0, 1) derived from `n` using a
/// Weyl-sequence / SplitMix64 hash mix.
fn pseudo_rand_f64(n: u32) -> f64 {
    let mut v = (n as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
    v ^= v >> 30;
    v = v.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    v ^= v >> 27;
    v = v.wrapping_mul(0x94d0_49bb_1331_11eb);
    v ^= v >> 31;
    // Map to [0, 1)
    (v as f64) / (u64::MAX as f64 + 1.0)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_is_zero_when_attempt_exceeds_max() {
        let p = SegmentRetryPolicy::new(3, 200);
        assert_eq!(p.next_delay(3), 0);
        assert_eq!(p.next_delay(99), 0);
    }

    #[test]
    fn test_delay_grows_exponentially() {
        let p = SegmentRetryPolicy::new(10, 100).with_jitter(0.0);
        let d0 = p.next_delay(0);
        let d1 = p.next_delay(1);
        let d2 = p.next_delay(2);
        assert_eq!(d0, 100);
        assert_eq!(d1, 200);
        assert_eq!(d2, 400);
    }

    #[test]
    fn test_delay_capped_at_max() {
        let p = SegmentRetryPolicy::new(10, 10_000)
            .with_max_delay(15_000)
            .with_jitter(0.0);
        // attempt 2 → 10_000 × 4 = 40_000 → capped at 15_000
        let d = p.next_delay(2);
        assert_eq!(d, 15_000, "expected cap at max_delay=15000 got {d}");
    }

    #[test]
    fn test_should_retry() {
        let p = SegmentRetryPolicy::new(3, 100);
        assert!(p.should_retry(0));
        assert!(p.should_retry(2));
        assert!(!p.should_retry(3));
        assert!(!p.should_retry(10));
    }

    #[test]
    fn test_jitter_adds_non_negative_amount() {
        let p = SegmentRetryPolicy::new(10, 500).with_jitter(0.5);
        let base_no_jitter = SegmentRetryPolicy::new(10, 500).with_jitter(0.0);
        for attempt in 0..5u32 {
            let d_jitter = p.next_delay(attempt);
            let d_plain = base_no_jitter.next_delay(attempt);
            assert!(
                d_jitter >= d_plain,
                "jitter must not reduce delay: attempt={attempt} jitter={d_jitter} plain={d_plain}"
            );
        }
    }

    #[test]
    fn test_with_jitter_clamps_above_one() {
        let p = SegmentRetryPolicy::new(5, 1000).with_jitter(5.0);
        assert!((p.jitter_fraction - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_deterministic_same_seed_same_result() {
        let p = SegmentRetryPolicy::new(10, 200).with_jitter(0.3);
        for attempt in 0..10u32 {
            assert_eq!(
                p.next_delay(attempt),
                p.next_delay(attempt),
                "same attempt must yield same delay"
            );
        }
    }

    #[test]
    fn test_base_delay_zero_always_zero() {
        let p = SegmentRetryPolicy::new(5, 0).with_jitter(0.0);
        for attempt in 0..5u32 {
            assert_eq!(p.next_delay(attempt), 0);
        }
    }
}
