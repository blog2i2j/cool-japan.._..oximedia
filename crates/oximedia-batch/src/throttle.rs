//! Rate throttling for batch job execution.
//!
//! Provides a token-bucket rate limiter, burst control, and a concurrency
//! limiter so batch workers respect external API quotas and resource limits.

#![allow(dead_code)]

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Configuration for the token-bucket throttle.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    /// Maximum tokens in the bucket (burst capacity).
    pub capacity: u64,
    /// Token refill rate (tokens per second).
    pub refill_rate: f64,
    /// Maximum concurrent operations allowed (0 = unlimited).
    pub max_concurrent: usize,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            refill_rate: 10.0,
            max_concurrent: 0,
        }
    }
}

impl ThrottleConfig {
    /// Create a new throttle configuration.
    #[must_use]
    pub fn new(capacity: u64, refill_rate: f64, max_concurrent: usize) -> Self {
        Self {
            capacity,
            refill_rate,
            max_concurrent,
        }
    }
}

/// Token-bucket rate limiter with optional concurrency limit.
#[derive(Debug)]
pub struct Throttle {
    config: ThrottleConfig,
    tokens: Mutex<f64>,
    last_refill: Mutex<Instant>,
    concurrent: AtomicUsize,
}

impl Throttle {
    /// Create a new throttle from configuration.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        Self {
            tokens: Mutex::new(config.capacity as f64),
            last_refill: Mutex::new(Instant::now()),
            concurrent: AtomicUsize::new(0),
            config,
        }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_config() -> Self {
        Self::new(ThrottleConfig::default())
    }

    /// Refill tokens based on elapsed time (internal).
    fn refill(&self) {
        let mut last = self.last_refill.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        *last = now;
        drop(last);

        let mut tokens = self.tokens.lock();
        let new_tokens = elapsed * self.config.refill_rate;
        *tokens = (*tokens + new_tokens).min(self.config.capacity as f64);
    }

    /// Try to acquire `n` tokens without blocking.
    ///
    /// Returns `true` if the tokens were available and consumed.
    pub fn try_acquire(&self, n: u64) -> bool {
        self.refill();
        let mut tokens = self.tokens.lock();
        if *tokens >= n as f64 {
            *tokens -= n as f64;
            true
        } else {
            false
        }
    }

    /// Acquire `n` tokens, blocking (spinning) until they are available.
    ///
    /// In production you would use async sleep; here we use a short spin
    /// suitable for unit tests and CPU-light operations.
    pub fn acquire(&self, n: u64) {
        loop {
            if self.try_acquire(n) {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Return the current number of available tokens (approximate).
    #[must_use]
    pub fn available_tokens(&self) -> f64 {
        self.refill();
        *self.tokens.lock()
    }

    /// Try to enter a concurrent section.  Returns `true` if the slot was
    /// available (or if `max_concurrent == 0`).
    pub fn try_enter_concurrent(&self) -> bool {
        if self.config.max_concurrent == 0 {
            return true;
        }
        let prev = self.concurrent.fetch_add(1, Ordering::SeqCst);
        if prev < self.config.max_concurrent {
            true
        } else {
            self.concurrent.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    /// Leave a concurrent section.
    pub fn leave_concurrent(&self) {
        if self.config.max_concurrent > 0 {
            self.concurrent.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Return the current concurrency count.
    #[must_use]
    pub fn concurrent_count(&self) -> usize {
        self.concurrent.load(Ordering::SeqCst)
    }

    /// Capacity (burst limit) of the token bucket.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.config.capacity
    }

    /// Configured refill rate (tokens / second).
    #[must_use]
    pub fn refill_rate(&self) -> f64 {
        self.config.refill_rate
    }
}

/// A shared, cloneable wrapper around [`Throttle`].
#[derive(Debug, Clone)]
pub struct SharedThrottle(Arc<Throttle>);

impl SharedThrottle {
    /// Create a new shared throttle.
    #[must_use]
    pub fn new(config: ThrottleConfig) -> Self {
        Self(Arc::new(Throttle::new(config)))
    }

    /// Try to acquire tokens without blocking.
    #[must_use]
    pub fn try_acquire(&self, n: u64) -> bool {
        self.0.try_acquire(n)
    }

    /// Acquire tokens, blocking until available.
    pub fn acquire(&self, n: u64) {
        self.0.acquire(n);
    }

    /// Current available tokens.
    #[must_use]
    pub fn available_tokens(&self) -> f64 {
        self.0.available_tokens()
    }
}

/// Simple concurrency limiter backed by an atomic counter.
#[derive(Debug)]
pub struct ConcurrencyLimiter {
    limit: usize,
    current: AtomicUsize,
}

impl ConcurrencyLimiter {
    /// Create a new limiter with the given maximum concurrency.
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            current: AtomicUsize::new(0),
        }
    }

    /// Try to acquire a slot.  Returns `true` on success.
    pub fn try_acquire(&self) -> bool {
        let prev = self.current.fetch_add(1, Ordering::SeqCst);
        if prev < self.limit {
            true
        } else {
            self.current.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    /// Release a previously acquired slot.
    pub fn release(&self) {
        self.current.fetch_sub(1, Ordering::SeqCst);
    }

    /// Current number of held slots.
    #[must_use]
    pub fn current(&self) -> usize {
        self.current.load(Ordering::SeqCst)
    }

    /// Maximum allowed concurrency.
    #[must_use]
    pub fn limit(&self) -> usize {
        self.limit
    }
}

/// Sliding-window request counter for rate-limiting by time window.
#[derive(Debug)]
pub struct SlidingWindowCounter {
    window: Duration,
    limit: u64,
    requests: Mutex<Vec<Instant>>,
}

impl SlidingWindowCounter {
    /// Create a new sliding-window counter.
    #[must_use]
    pub fn new(window: Duration, limit: u64) -> Self {
        Self {
            window,
            limit,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Record a request and return `true` if within the rate limit.
    pub fn record(&self) -> bool {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(self.window)
            .expect("window duration does not exceed Instant range");
        let mut reqs = self.requests.lock();
        reqs.retain(|&t| t > cutoff);
        if (reqs.len() as u64) < self.limit {
            reqs.push(now);
            true
        } else {
            false
        }
    }

    /// Number of requests recorded in the current window.
    #[must_use]
    pub fn count_in_window(&self) -> usize {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(self.window)
            .expect("window duration does not exceed Instant range");
        let mut reqs = self.requests.lock();
        reqs.retain(|&t| t > cutoff);
        reqs.len()
    }
}

/// Tracks statistics about throttle events.
#[derive(Debug, Default)]
pub struct ThrottleStats {
    /// Total acquire attempts.
    pub attempts: AtomicI64,
    /// Successful acquires.
    pub successes: AtomicI64,
    /// Rejected / throttled attempts.
    pub rejections: AtomicI64,
}

impl ThrottleStats {
    /// Create new stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an attempt outcome.
    pub fn record(&self, success: bool) {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rejections.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Rejection ratio (0.0–1.0).
    #[must_use]
    pub fn rejection_ratio(&self) -> f64 {
        let attempts = self.attempts.load(Ordering::Relaxed);
        if attempts == 0 {
            return 0.0;
        }
        self.rejections.load(Ordering::Relaxed) as f64 / attempts as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throttle_config_default() {
        let cfg = ThrottleConfig::default();
        assert_eq!(cfg.capacity, 100);
        assert!(cfg.refill_rate > 0.0);
        assert_eq!(cfg.max_concurrent, 0);
    }

    #[test]
    fn test_throttle_initial_tokens() {
        let t = Throttle::new(ThrottleConfig {
            capacity: 50,
            refill_rate: 5.0,
            max_concurrent: 0,
        });
        // Initially full
        assert!(t.available_tokens() > 49.0);
    }

    #[test]
    fn test_throttle_try_acquire_success() {
        let t = Throttle::new(ThrottleConfig {
            capacity: 10,
            refill_rate: 1.0,
            max_concurrent: 0,
        });
        assert!(t.try_acquire(5));
        assert!(t.try_acquire(5));
    }

    #[test]
    fn test_throttle_try_acquire_fail_when_empty() {
        let t = Throttle::new(ThrottleConfig {
            capacity: 3,
            refill_rate: 0.001,
            max_concurrent: 0,
        });
        assert!(t.try_acquire(3));
        // Bucket is now empty; next acquire should fail
        assert!(!t.try_acquire(1));
    }

    #[test]
    fn test_throttle_capacity() {
        let t = Throttle::new(ThrottleConfig::new(42, 1.0, 0));
        assert_eq!(t.capacity(), 42);
    }

    #[test]
    fn test_throttle_refill_rate() {
        let t = Throttle::new(ThrottleConfig::new(10, 3.5, 0));
        assert!((t.refill_rate() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_concurrent_limit_acquire_release() {
        let t = Throttle::new(ThrottleConfig {
            capacity: 100,
            refill_rate: 10.0,
            max_concurrent: 2,
        });
        assert!(t.try_enter_concurrent());
        assert!(t.try_enter_concurrent());
        assert!(!t.try_enter_concurrent()); // at limit
        t.leave_concurrent();
        assert!(t.try_enter_concurrent());
    }

    #[test]
    fn test_concurrent_count() {
        let t = Throttle::new(ThrottleConfig {
            capacity: 100,
            refill_rate: 10.0,
            max_concurrent: 5,
        });
        assert_eq!(t.concurrent_count(), 0);
        t.try_enter_concurrent();
        assert_eq!(t.concurrent_count(), 1);
        t.leave_concurrent();
        assert_eq!(t.concurrent_count(), 0);
    }

    #[test]
    fn test_concurrency_limiter_basics() {
        let l = ConcurrencyLimiter::new(3);
        assert!(l.try_acquire());
        assert!(l.try_acquire());
        assert!(l.try_acquire());
        assert!(!l.try_acquire());
        l.release();
        assert!(l.try_acquire());
    }

    #[test]
    fn test_concurrency_limiter_current_and_limit() {
        let l = ConcurrencyLimiter::new(5);
        l.try_acquire();
        assert_eq!(l.current(), 1);
        assert_eq!(l.limit(), 5);
    }

    #[test]
    fn test_sliding_window_counter_allows_within_limit() {
        let sw = SlidingWindowCounter::new(Duration::from_secs(60), 5);
        for _ in 0..5 {
            assert!(sw.record());
        }
        assert!(!sw.record()); // over limit
    }

    #[test]
    fn test_sliding_window_count_in_window() {
        let sw = SlidingWindowCounter::new(Duration::from_secs(60), 10);
        sw.record();
        sw.record();
        assert_eq!(sw.count_in_window(), 2);
    }

    #[test]
    fn test_throttle_stats_record() {
        let stats = ThrottleStats::new();
        stats.record(true);
        stats.record(true);
        stats.record(false);
        assert_eq!(stats.attempts.load(Ordering::Relaxed), 3);
        assert_eq!(stats.successes.load(Ordering::Relaxed), 2);
        assert_eq!(stats.rejections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_throttle_stats_rejection_ratio() {
        let stats = ThrottleStats::new();
        assert!((stats.rejection_ratio() - 0.0).abs() < 1e-6);
        stats.record(true);
        stats.record(false);
        assert!((stats.rejection_ratio() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_shared_throttle_clone() {
        let st = SharedThrottle::new(ThrottleConfig::new(20, 2.0, 0));
        let st2 = st.clone();
        assert!(st.try_acquire(10));
        // Both share the same bucket
        assert!(st2.try_acquire(10));
        assert!(!st.try_acquire(1));
    }
}
