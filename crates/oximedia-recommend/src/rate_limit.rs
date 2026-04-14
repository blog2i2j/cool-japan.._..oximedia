//! Token-bucket rate limiter for the recommendation engine.
//!
//! Prevents API abuse by enforcing per-user (and global) call-rate quotas.
//! Uses the classic **token-bucket** algorithm:
//! - Each identity has a bucket of capacity `max_tokens`.
//! - Tokens refill at `refill_rate` tokens per second.
//! - Each call consumes one token; calls when the bucket is empty are rejected.
//!
//! All timestamps are Unix seconds (i64), supplied by the caller to keep the
//! module fully deterministic and test-friendly.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Bucket state
// ---------------------------------------------------------------------------

/// State of a single rate-limit token bucket.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Current number of available tokens (fractional).
    available: f64,
    /// Maximum capacity of the bucket.
    capacity: f64,
    /// Refill rate in tokens per second.
    refill_rate: f64,
    /// Unix timestamp (seconds) of the last refill.
    last_refill: i64,
    /// Total calls that consumed a token.
    total_consumed: u64,
    /// Total calls that were rejected (bucket empty).
    total_rejected: u64,
}

impl TokenBucket {
    /// Create a new token bucket, starting full.
    #[must_use]
    pub fn new(capacity: f64, refill_rate: f64, now: i64) -> Self {
        Self {
            available: capacity,
            capacity,
            refill_rate,
            last_refill: now,
            total_consumed: 0,
            total_rejected: 0,
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self, now: i64) {
        let elapsed = (now - self.last_refill).max(0) as f64;
        self.available = (self.available + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    /// Attempt to consume `tokens` from the bucket.
    ///
    /// Returns `true` if successful (tokens available), `false` if the bucket
    /// would be overdrawn.
    pub fn try_consume(&mut self, tokens: f64, now: i64) -> bool {
        self.refill(now);
        if self.available >= tokens {
            self.available -= tokens;
            self.total_consumed += 1;
            true
        } else {
            self.total_rejected += 1;
            false
        }
    }

    /// Current available tokens (after refill).
    #[must_use]
    pub fn available(&self) -> f64 {
        self.available
    }

    /// Total calls successfully allowed.
    #[must_use]
    pub fn total_consumed(&self) -> u64 {
        self.total_consumed
    }

    /// Total calls rejected.
    #[must_use]
    pub fn total_rejected(&self) -> u64 {
        self.total_rejected
    }
}

// ---------------------------------------------------------------------------
// Rate-limiter configuration
// ---------------------------------------------------------------------------

/// Configuration for the recommendation-engine rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum tokens per user bucket.
    pub per_user_capacity: f64,
    /// Token refill rate per user (tokens / second).
    pub per_user_refill_rate: f64,
    /// Maximum tokens in the global bucket (shared across all users).
    pub global_capacity: f64,
    /// Global token refill rate (tokens / second).
    pub global_refill_rate: f64,
    /// Tokens consumed per recommendation request (default: 1).
    pub tokens_per_request: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            per_user_capacity: 20.0,
            per_user_refill_rate: 2.0, // 2 req/s sustained, burst up to 20
            global_capacity: 500.0,
            global_refill_rate: 100.0,
            tokens_per_request: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Rate limiter
// ---------------------------------------------------------------------------

/// Rate-limit outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitDecision {
    /// Request is within limits and allowed.
    Allowed,
    /// Per-user bucket exhausted.
    UserLimitExceeded,
    /// Global bucket exhausted.
    GlobalLimitExceeded,
}

impl RateLimitDecision {
    /// Returns `true` if the request is allowed.
    #[must_use]
    pub fn is_allowed(self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Token-bucket rate limiter for the recommendation engine.
///
/// Maintains per-user buckets and one global bucket.  Both must have tokens
/// for a request to be approved.
#[derive(Debug)]
pub struct RecommendationRateLimiter {
    /// Per-user buckets keyed by user identity string.
    user_buckets: HashMap<String, TokenBucket>,
    /// Global shared bucket.
    global_bucket: TokenBucket,
    /// Configuration.
    config: RateLimitConfig,
}

impl RecommendationRateLimiter {
    /// Create a new rate limiter.
    #[must_use]
    pub fn new(config: RateLimitConfig, now: i64) -> Self {
        let global_bucket =
            TokenBucket::new(config.global_capacity, config.global_refill_rate, now);
        Self {
            user_buckets: HashMap::new(),
            global_bucket,
            config,
        }
    }

    /// Decide whether to allow a recommendation request for `user_id` at time `now`.
    ///
    /// Consumes tokens from both the per-user and global buckets atomically.
    /// If the global bucket would be exceeded, the user bucket is **not** consumed.
    pub fn check_and_consume(&mut self, user_id: &str, now: i64) -> RateLimitDecision {
        let tokens = self.config.tokens_per_request;

        // Peek at global first (refill only)
        self.global_bucket.refill(now);
        if self.global_bucket.available < tokens {
            self.global_bucket.total_rejected += 1;
            return RateLimitDecision::GlobalLimitExceeded;
        }

        // Check per-user bucket
        let user_bucket = self
            .user_buckets
            .entry(user_id.to_string())
            .or_insert_with(|| {
                TokenBucket::new(
                    self.config.per_user_capacity,
                    self.config.per_user_refill_rate,
                    now,
                )
            });

        if !user_bucket.try_consume(tokens, now) {
            return RateLimitDecision::UserLimitExceeded;
        }

        // Consume from global
        self.global_bucket.available -= tokens;
        self.global_bucket.total_consumed += 1;

        RateLimitDecision::Allowed
    }

    /// Get the current available tokens for a user (without consuming).
    ///
    /// Returns `None` if the user has no bucket yet (i.e., they haven't made any requests).
    #[must_use]
    pub fn user_available_tokens(&self, user_id: &str) -> Option<f64> {
        self.user_buckets.get(user_id).map(|b| b.available)
    }

    /// Get the current available global tokens.
    #[must_use]
    pub fn global_available_tokens(&self) -> f64 {
        self.global_bucket.available
    }

    /// Total per-user rejections across all users.
    #[must_use]
    pub fn total_user_rejections(&self) -> u64 {
        self.user_buckets.values().map(|b| b.total_rejected).sum()
    }

    /// Total global rejections.
    #[must_use]
    pub fn total_global_rejections(&self) -> u64 {
        self.global_bucket.total_rejected
    }

    /// Number of distinct users tracked.
    #[must_use]
    pub fn tracked_users(&self) -> usize {
        self.user_buckets.len()
    }

    /// Reset a user's bucket, refilling it to capacity.
    pub fn reset_user(&mut self, user_id: &str, now: i64) {
        self.user_buckets.insert(
            user_id.to_string(),
            TokenBucket::new(
                self.config.per_user_capacity,
                self.config.per_user_refill_rate,
                now,
            ),
        );
    }
}

impl Default for RecommendationRateLimiter {
    fn default() -> Self {
        Self::new(RateLimitConfig::default(), 0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter(capacity: f64, rate: f64) -> RecommendationRateLimiter {
        let config = RateLimitConfig {
            per_user_capacity: capacity,
            per_user_refill_rate: rate,
            global_capacity: 1000.0,
            global_refill_rate: 500.0,
            tokens_per_request: 1.0,
        };
        RecommendationRateLimiter::new(config, 0)
    }

    #[test]
    fn test_new_limiter_allows_first_request() {
        let mut rl = make_limiter(5.0, 1.0);
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
    }

    #[test]
    fn test_burst_then_reject() {
        let mut rl = make_limiter(3.0, 0.0); // no refill
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
        // 4th should be rejected
        assert_eq!(
            rl.check_and_consume("alice", 0),
            RateLimitDecision::UserLimitExceeded
        );
    }

    #[test]
    fn test_refill_after_time() {
        let mut rl = make_limiter(2.0, 2.0); // 2 tokens/sec
                                             // Drain bucket
        rl.check_and_consume("alice", 0);
        rl.check_and_consume("alice", 0);
        assert_eq!(
            rl.check_and_consume("alice", 0),
            RateLimitDecision::UserLimitExceeded
        );
        // After 1 second, should have 2 new tokens
        assert_eq!(rl.check_and_consume("alice", 1), RateLimitDecision::Allowed);
    }

    #[test]
    fn test_different_users_independent() {
        let mut rl = make_limiter(1.0, 0.0);
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
        // alice exhausted, but bob has full bucket
        assert_eq!(rl.check_and_consume("bob", 0), RateLimitDecision::Allowed);
        // both exhausted
        assert_eq!(
            rl.check_and_consume("alice", 0),
            RateLimitDecision::UserLimitExceeded
        );
        assert_eq!(
            rl.check_and_consume("bob", 0),
            RateLimitDecision::UserLimitExceeded
        );
    }

    #[test]
    fn test_global_limit_blocks_all_users() {
        let config = RateLimitConfig {
            per_user_capacity: 1000.0,
            per_user_refill_rate: 0.0,
            global_capacity: 2.0,
            global_refill_rate: 0.0,
            tokens_per_request: 1.0,
        };
        let mut rl = RecommendationRateLimiter::new(config, 0);
        rl.check_and_consume("alice", 0);
        rl.check_and_consume("bob", 0);
        // Global exhausted
        let decision = rl.check_and_consume("charlie", 0);
        assert_eq!(decision, RateLimitDecision::GlobalLimitExceeded);
    }

    #[test]
    fn test_user_available_tokens_before_first_request() {
        let rl = make_limiter(5.0, 1.0);
        assert!(rl.user_available_tokens("unknown").is_none());
    }

    #[test]
    fn test_user_available_tokens_after_request() {
        let mut rl = make_limiter(5.0, 1.0);
        rl.check_and_consume("alice", 0);
        let avail = rl
            .user_available_tokens("alice")
            .expect("bucket should exist");
        assert!((avail - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tracked_users_increases() {
        let mut rl = make_limiter(5.0, 1.0);
        assert_eq!(rl.tracked_users(), 0);
        rl.check_and_consume("alice", 0);
        assert_eq!(rl.tracked_users(), 1);
        rl.check_and_consume("bob", 0);
        assert_eq!(rl.tracked_users(), 2);
    }

    #[test]
    fn test_total_user_rejections() {
        let mut rl = make_limiter(1.0, 0.0);
        rl.check_and_consume("alice", 0);
        rl.check_and_consume("alice", 0); // rejected
        rl.check_and_consume("alice", 0); // rejected
        assert_eq!(rl.total_user_rejections(), 2);
    }

    #[test]
    fn test_reset_user_refills_bucket() {
        let mut rl = make_limiter(1.0, 0.0);
        rl.check_and_consume("alice", 0);
        assert_eq!(
            rl.check_and_consume("alice", 0),
            RateLimitDecision::UserLimitExceeded
        );
        rl.reset_user("alice", 0);
        assert_eq!(rl.check_and_consume("alice", 0), RateLimitDecision::Allowed);
    }

    #[test]
    fn test_token_bucket_available_increases_after_refill() {
        let mut bucket = TokenBucket::new(10.0, 5.0, 0);
        bucket.try_consume(5.0, 0); // drain 5
        assert!((bucket.available() - 5.0).abs() < f64::EPSILON);
        bucket.refill(2); // +10 tokens but capped at 10
        assert!((bucket.available() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_limit_decision_is_allowed() {
        assert!(RateLimitDecision::Allowed.is_allowed());
        assert!(!RateLimitDecision::UserLimitExceeded.is_allowed());
        assert!(!RateLimitDecision::GlobalLimitExceeded.is_allowed());
    }

    #[test]
    fn test_token_bucket_stats() {
        let mut bucket = TokenBucket::new(3.0, 0.0, 0);
        bucket.try_consume(1.0, 0);
        bucket.try_consume(1.0, 0);
        bucket.try_consume(1.0, 0);
        bucket.try_consume(1.0, 0); // rejected
        assert_eq!(bucket.total_consumed(), 3);
        assert_eq!(bucket.total_rejected(), 1);
    }

    #[test]
    fn test_global_available_tokens_decreases() {
        let mut rl = make_limiter(100.0, 10.0);
        let before = rl.global_available_tokens();
        rl.check_and_consume("alice", 0);
        let after = rl.global_available_tokens();
        assert!((before - after - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert!(config.per_user_capacity > 0.0);
        assert!(config.per_user_refill_rate > 0.0);
        assert!(config.global_capacity > 0.0);
        assert!((config.tokens_per_request - 1.0).abs() < f64::EPSILON);
    }
}
