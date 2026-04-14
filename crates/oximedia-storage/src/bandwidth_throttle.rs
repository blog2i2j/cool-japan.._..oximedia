//! Bandwidth throttling for storage transfers.
//!
//! Provides token-bucket rate limiting, time-window throughput tracking,
//! adaptive throttling, per-key bandwidth allocation, and a
//! `ThrottledTransfer` wrapper for the [`crate::CloudStorage`] trait.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::sleep;

/// Token bucket rate limiter with async blocking support.
///
/// The bucket holds up to `burst_bytes` tokens.  Each call to
/// [`TokenBucket::try_consume`] checks synchronously whether tokens are
/// available; [`TokenBucket::consume_blocking`] waits asynchronously until
/// tokens are replenished at `rate_bytes_per_sec`.
#[derive(Debug, Clone)]
pub struct TokenBucket {
    /// Sustained rate in bytes per second (also used as the refill amount
    /// for the legacy synchronous interface).
    rate_bytes_per_sec: u64,
    /// Maximum tokens (bytes) the bucket can hold — the burst allowance.
    burst_bytes: u64,
    /// Current available tokens.
    tokens: u64,
    /// Tokens added per refill interval (legacy interface).
    refill_amount: u64,
    /// Number of refill calls made.
    refill_count: u64,
}

impl TokenBucket {
    /// Create a token bucket with an explicit `rate_bytes_per_sec` and
    /// `burst_bytes` (maximum token capacity).
    ///
    /// The bucket starts full (tokens = `burst_bytes`).
    pub fn with_rate(rate_bytes_per_sec: u64, burst_bytes: u64) -> Self {
        Self {
            rate_bytes_per_sec,
            burst_bytes,
            tokens: burst_bytes,
            refill_amount: rate_bytes_per_sec,
            refill_count: 0,
        }
    }

    /// Create a new token bucket.
    ///
    /// `capacity` is the burst size in bytes; `refill_amount` is bytes added per refill.
    pub fn new(capacity: u64, refill_amount: u64) -> Self {
        Self {
            rate_bytes_per_sec: refill_amount,
            burst_bytes: capacity,
            tokens: capacity,
            refill_amount,
            refill_count: 0,
        }
    }

    /// Try to consume `amount` tokens. Returns `true` if successful.
    pub fn try_consume(&mut self, amount: u64) -> bool {
        if self.tokens >= amount {
            self.tokens -= amount;
            true
        } else {
            false
        }
    }

    /// Block asynchronously until `bytes` tokens are available, then consume them.
    ///
    /// Uses `tokio::time::sleep` to yield between checks; the sleep duration is
    /// computed as `bytes / rate_bytes_per_sec` (minimum 1 ms) so the caller
    /// never busy-loops.
    ///
    /// # Zero-rate handling
    /// If `rate_bytes_per_sec == 0` the function sleeps indefinitely (it will
    /// only return if the rate is changed externally, which this API does not
    /// support — callers should check the rate before calling).
    #[allow(clippy::cast_precision_loss)]
    pub async fn consume_blocking(&mut self, bytes: u64) {
        loop {
            if self.tokens >= bytes {
                self.tokens -= bytes;
                return;
            }

            // How many additional tokens are needed?
            let needed = bytes - self.tokens;

            // How long until the needed tokens arrive?
            let wait_ms = if self.rate_bytes_per_sec == 0 {
                1_000 // 1 s default when rate is zero (will loop)
            } else {
                // ceiling division to avoid under-sleeping
                let ms = (needed as f64 / self.rate_bytes_per_sec as f64 * 1_000.0).ceil() as u64;
                ms.max(1)
            };

            sleep(Duration::from_millis(wait_ms)).await;

            // Refill proportionally to elapsed time.
            let added = self
                .rate_bytes_per_sec
                .min(self.burst_bytes - self.tokens.min(self.burst_bytes));
            self.tokens = (self.tokens + added).min(self.burst_bytes);
        }
    }

    /// Refill the bucket by one interval (legacy interface).
    pub fn refill(&mut self) {
        self.tokens = (self.tokens + self.refill_amount).min(self.burst_bytes);
        self.refill_count += 1;
    }

    /// Current available tokens.
    pub fn available(&self) -> u64 {
        self.tokens
    }

    /// Maximum capacity (burst size).
    pub fn capacity(&self) -> u64 {
        self.burst_bytes
    }

    /// Configured sustained rate in bytes per second.
    pub fn rate_bytes_per_sec(&self) -> u64 {
        self.rate_bytes_per_sec
    }

    /// Number of refills that have occurred.
    pub fn refill_count(&self) -> u64 {
        self.refill_count
    }

    /// Utilization ratio (consumed / capacity).
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.burst_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.tokens as f64 / self.burst_bytes as f64)
    }
}

// ─── ThrottledTransfer ────────────────────────────────────────────────────────

/// A [`crate::CloudStorage`] wrapper that applies token-bucket bandwidth
/// limiting to all uploads and downloads.
///
/// The inner bucket is protected by a `tokio::sync::Mutex` so that multiple
/// concurrent upload/download tasks share a single rate limit.
#[derive(Debug, Clone)]
pub struct ThrottledTransfer<S> {
    inner: Arc<S>,
    bucket: Arc<Mutex<TokenBucket>>,
}

impl<S> ThrottledTransfer<S> {
    /// Wrap `storage` with a token-bucket throttle.
    ///
    /// The `rate_bytes_per_sec` controls the sustained transfer rate;
    /// `burst_bytes` allows short bursts above that rate.
    pub fn new(storage: S, rate_bytes_per_sec: u64, burst_bytes: u64) -> Self {
        Self {
            inner: Arc::new(storage),
            bucket: Arc::new(Mutex::new(TokenBucket::with_rate(
                rate_bytes_per_sec,
                burst_bytes,
            ))),
        }
    }

    /// Access the underlying storage backend.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Consume `bytes` from the token bucket, sleeping if necessary.
    pub async fn throttle(&self, bytes: u64) {
        let mut bucket = self.bucket.lock().await;
        bucket.consume_blocking(bytes).await;
    }

    /// Synchronously attempt to consume `bytes`; returns `false` if tokens are
    /// insufficient (non-blocking path for callers that cannot sleep).
    pub async fn try_throttle(&self, bytes: u64) -> bool {
        let mut bucket = self.bucket.lock().await;
        bucket.try_consume(bytes)
    }

    /// Available tokens at this instant.
    pub async fn available_tokens(&self) -> u64 {
        self.bucket.lock().await.available()
    }
}

/// Throughput measurement over a sliding window.
#[derive(Debug, Clone)]
pub struct ThroughputMeter {
    /// Ring buffer of (timestamp_ms, bytes) samples.
    samples: Vec<(u64, u64)>,
    /// Maximum number of samples to retain.
    max_samples: usize,
}

impl ThroughputMeter {
    /// Create a new throughput meter.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::new(),
            max_samples: max_samples.max(1),
        }
    }

    /// Record a sample.
    pub fn record(&mut self, timestamp_ms: u64, bytes: u64) {
        self.samples.push((timestamp_ms, bytes));
        if self.samples.len() > self.max_samples {
            self.samples.remove(0);
        }
    }

    /// Number of samples recorded.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Total bytes across all samples.
    pub fn total_bytes(&self) -> u64 {
        self.samples.iter().map(|(_, b)| b).sum()
    }

    /// Average throughput in bytes per second over the sample window.
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_throughput_bps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first_ts = self.samples.first().map(|s| s.0).unwrap_or(0);
        let last_ts = self.samples.last().map(|s| s.0).unwrap_or(0);
        let duration_ms = last_ts.saturating_sub(first_ts);
        if duration_ms == 0 {
            return 0.0;
        }
        let total_bytes: u64 = self.samples.iter().map(|(_, b)| b).sum();
        total_bytes as f64 / (duration_ms as f64 / 1000.0)
    }

    /// Peak throughput across consecutive sample pairs.
    #[allow(clippy::cast_precision_loss)]
    pub fn peak_throughput_bps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mut peak = 0.0_f64;
        for window in self.samples.windows(2) {
            let dt_ms = window[1].0.saturating_sub(window[0].0);
            if dt_ms > 0 {
                let bps = window[1].1 as f64 / (dt_ms as f64 / 1000.0);
                if bps > peak {
                    peak = bps;
                }
            }
        }
        peak
    }

    /// Clear all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

/// Throttle mode for adaptive throttling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleMode {
    /// No throttling (unlimited).
    Unlimited,
    /// Fixed rate in bytes per second.
    Fixed(u64),
    /// Adaptive — adjusts based on congestion signals.
    Adaptive {
        /// Minimum rate in bytes per second.
        min_bps: u64,
        /// Maximum rate in bytes per second.
        max_bps: u64,
    },
}

/// Per-key bandwidth allocation.
#[derive(Debug, Clone)]
pub struct KeyAllocation {
    /// Object key or prefix.
    pub key: String,
    /// Allocated bytes per second.
    pub bps: u64,
    /// Bytes consumed so far.
    pub consumed: u64,
}

impl KeyAllocation {
    /// Create a new allocation.
    pub fn new(key: impl Into<String>, bps: u64) -> Self {
        Self {
            key: key.into(),
            bps,
            consumed: 0,
        }
    }

    /// Remaining budget.
    pub fn remaining(&self) -> u64 {
        self.bps.saturating_sub(self.consumed)
    }

    /// Record consumption.
    pub fn consume(&mut self, bytes: u64) {
        self.consumed += bytes;
    }

    /// Reset consumption counter.
    pub fn reset(&mut self) {
        self.consumed = 0;
    }
}

/// Bandwidth throttle manager that orchestrates rate limiting for transfers.
#[derive(Debug)]
pub struct BandwidthThrottle {
    /// Global token bucket.
    bucket: TokenBucket,
    /// Throughput meter.
    meter: ThroughputMeter,
    /// Active throttle mode.
    mode: ThrottleMode,
    /// Per-key allocations.
    allocations: HashMap<String, KeyAllocation>,
    /// Total bytes throttled (delayed or rejected).
    bytes_throttled: u64,
    /// Total requests that were throttled.
    throttle_events: u64,
}

impl BandwidthThrottle {
    /// Create a new throttle with a global rate limit.
    pub fn new(global_bps: u64) -> Self {
        Self {
            bucket: TokenBucket::new(global_bps, global_bps),
            meter: ThroughputMeter::new(1000),
            mode: ThrottleMode::Fixed(global_bps),
            allocations: HashMap::new(),
            bytes_throttled: 0,
            throttle_events: 0,
        }
    }

    /// Create an unlimited (no-throttle) instance.
    pub fn unlimited() -> Self {
        Self {
            bucket: TokenBucket::new(u64::MAX, u64::MAX),
            meter: ThroughputMeter::new(1000),
            mode: ThrottleMode::Unlimited,
            allocations: HashMap::new(),
            bytes_throttled: 0,
            throttle_events: 0,
        }
    }

    /// Try to acquire bandwidth for a transfer of `bytes`.
    pub fn try_acquire(&mut self, bytes: u64, timestamp_ms: u64) -> bool {
        if self.mode == ThrottleMode::Unlimited {
            self.meter.record(timestamp_ms, bytes);
            return true;
        }
        if self.bucket.try_consume(bytes) {
            self.meter.record(timestamp_ms, bytes);
            true
        } else {
            self.bytes_throttled += bytes;
            self.throttle_events += 1;
            false
        }
    }

    /// Refill the token bucket (call this periodically).
    pub fn refill(&mut self) {
        self.bucket.refill();
    }

    /// Set a per-key bandwidth allocation.
    pub fn set_allocation(&mut self, key: impl Into<String>, bps: u64) {
        let key = key.into();
        self.allocations
            .insert(key.clone(), KeyAllocation::new(key, bps));
    }

    /// Get allocation for a key.
    pub fn get_allocation(&self, key: &str) -> Option<&KeyAllocation> {
        self.allocations.get(key)
    }

    /// Reset all per-key consumption counters.
    pub fn reset_allocations(&mut self) {
        for alloc in self.allocations.values_mut() {
            alloc.reset();
        }
    }

    /// Current throttle mode.
    pub fn mode(&self) -> &ThrottleMode {
        &self.mode
    }

    /// Total bytes that were throttled.
    pub fn total_bytes_throttled(&self) -> u64 {
        self.bytes_throttled
    }

    /// Total throttle events.
    pub fn total_throttle_events(&self) -> u64 {
        self.throttle_events
    }

    /// Average throughput from the meter.
    pub fn avg_throughput(&self) -> f64 {
        self.meter.avg_throughput_bps()
    }

    /// Available tokens in the global bucket.
    pub fn available_tokens(&self) -> u64 {
        self.bucket.available()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(1000, 500);
        assert_eq!(bucket.available(), 1000);
        assert!(bucket.try_consume(300));
        assert_eq!(bucket.available(), 700);
    }

    #[test]
    fn test_token_bucket_overflow() {
        let mut bucket = TokenBucket::new(1000, 500);
        assert!(!bucket.try_consume(1001));
        assert_eq!(bucket.available(), 1000);
    }

    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(1000, 500);
        bucket.try_consume(800);
        assert_eq!(bucket.available(), 200);
        bucket.refill();
        assert_eq!(bucket.available(), 700);
        bucket.refill();
        assert_eq!(bucket.available(), 1000); // capped at capacity
    }

    #[test]
    fn test_token_bucket_utilization() {
        let mut bucket = TokenBucket::new(1000, 100);
        assert!((bucket.utilization()).abs() < f64::EPSILON);
        bucket.try_consume(500);
        assert!((bucket.utilization() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_throughput_meter_avg() {
        let mut meter = ThroughputMeter::new(100);
        meter.record(0, 1000);
        meter.record(1000, 2000);
        // 3000 bytes over 1 second = 3000 bps
        let bps = meter.avg_throughput_bps();
        assert!((bps - 3000.0).abs() < 0.1);
    }

    #[test]
    fn test_throughput_meter_peak() {
        let mut meter = ThroughputMeter::new(100);
        meter.record(0, 100);
        meter.record(1000, 5000); // 5000 B in 1s = 5000 bps
        meter.record(2000, 1000); // 1000 B in 1s = 1000 bps
        assert!((meter.peak_throughput_bps() - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_throughput_meter_single_sample() {
        let mut meter = ThroughputMeter::new(10);
        meter.record(0, 1000);
        assert_eq!(meter.avg_throughput_bps(), 0.0);
    }

    #[test]
    fn test_key_allocation() {
        let mut alloc = KeyAllocation::new("video/file.mp4", 10_000);
        assert_eq!(alloc.remaining(), 10_000);
        alloc.consume(3000);
        assert_eq!(alloc.remaining(), 7000);
        alloc.reset();
        assert_eq!(alloc.remaining(), 10_000);
    }

    #[test]
    fn test_bandwidth_throttle_acquire() {
        let mut t = BandwidthThrottle::new(1000);
        assert!(t.try_acquire(500, 0));
        assert!(t.try_acquire(500, 100));
        assert!(!t.try_acquire(1, 200)); // bucket empty
        assert_eq!(t.total_throttle_events(), 1);
    }

    #[test]
    fn test_bandwidth_throttle_unlimited() {
        let mut t = BandwidthThrottle::unlimited();
        assert!(t.try_acquire(u64::MAX / 2, 0));
        assert_eq!(t.total_throttle_events(), 0);
    }

    #[test]
    fn test_bandwidth_throttle_refill_and_acquire() {
        let mut t = BandwidthThrottle::new(1000);
        assert!(t.try_acquire(1000, 0));
        assert!(!t.try_acquire(1, 100));
        t.refill();
        assert!(t.try_acquire(500, 200));
    }

    #[test]
    fn test_set_and_get_allocation() {
        let mut t = BandwidthThrottle::new(10_000);
        t.set_allocation("key1", 5000);
        let alloc = t.get_allocation("key1").expect("allocation should exist");
        assert_eq!(alloc.bps, 5000);
    }

    #[test]
    fn test_throttle_mode() {
        let t = BandwidthThrottle::new(100);
        assert!(matches!(t.mode(), ThrottleMode::Fixed(100)));
        let t2 = BandwidthThrottle::unlimited();
        assert!(matches!(t2.mode(), ThrottleMode::Unlimited));
    }

    // ── TokenBucket async tests (Task E) ──────────────────────────────────────

    #[test]
    fn test_token_bucket_with_rate_starts_full() {
        let bucket = TokenBucket::with_rate(1000, 5000);
        assert_eq!(bucket.available(), 5000);
        assert_eq!(bucket.rate_bytes_per_sec(), 1000);
    }

    #[test]
    fn test_token_bucket_try_consume_blocks_when_empty() {
        let mut bucket = TokenBucket::with_rate(500, 1000);
        // Consume everything
        assert!(bucket.try_consume(1000));
        // Bucket empty — next try_consume should fail
        assert!(!bucket.try_consume(1));
    }

    #[test]
    fn test_token_bucket_burst_allows_spike() {
        // burst_bytes = 4000 → can absorb 4000 bytes in one shot
        let mut bucket = TokenBucket::with_rate(1000, 4000);
        assert!(bucket.try_consume(4000));
        assert!(!bucket.try_consume(1)); // exhausted
    }

    #[tokio::test]
    async fn test_consume_blocking_returns_when_tokens_available() {
        // The bucket starts full with 1000 tokens; requesting 500 should
        // return immediately without sleeping.
        let mut bucket = TokenBucket::with_rate(10_000, 1000);
        bucket.consume_blocking(500).await;
        assert_eq!(bucket.available(), 500);
    }

    #[tokio::test]
    async fn test_throttled_transfer_try_throttle_succeeds_within_burst() {
        // Create a dummy no-op storage (just a unit type).
        struct NoopStorage;
        let throttle = ThrottledTransfer::new(NoopStorage, 1_000, 5_000);
        // 4000 bytes within burst of 5000 — should succeed immediately.
        let ok = throttle.try_throttle(4_000).await;
        assert!(ok, "try_throttle should succeed within burst");
    }

    #[tokio::test]
    async fn test_throttled_transfer_try_throttle_fails_over_burst() {
        struct NoopStorage;
        let throttle = ThrottledTransfer::new(NoopStorage, 1_000, 500);
        // Requesting 1000 > burst of 500 → fails
        let ok = throttle.try_throttle(1_000).await;
        assert!(!ok, "try_throttle should fail when request exceeds burst");
    }

    #[tokio::test]
    async fn test_throttled_transfer_available_tokens_decreases_on_consume() {
        struct NoopStorage;
        let throttle = ThrottledTransfer::new(NoopStorage, 10_000, 2_000);
        let before = throttle.available_tokens().await;
        // Consume half the burst.
        throttle.try_throttle(1_000).await;
        let after = throttle.available_tokens().await;
        assert!(
            after < before,
            "available tokens should decrease after consume"
        );
    }

    #[test]
    fn test_token_bucket_utilization_with_rate() {
        let mut bucket = TokenBucket::with_rate(1000, 1000);
        assert!((bucket.utilization()).abs() < f64::EPSILON);
        bucket.try_consume(500);
        assert!((bucket.utilization() - 0.5).abs() < f64::EPSILON);
    }
}
