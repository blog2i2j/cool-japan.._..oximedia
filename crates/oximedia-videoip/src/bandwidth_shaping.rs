//! Traffic shaping and QoS prioritisation per stream.
//!
//! The `BandwidthShaper` rate-limits and prioritises media packets to prevent
//! network saturation and ensure high-priority streams (e.g., programme output)
//! are not starved by lower-priority feeds (e.g., preview feeds).
//!
//! # Design
//!
//! The shaper uses a **Token Bucket** algorithm per stream: each stream has a
//! bucket that refills at the configured rate.  Packets are admitted when enough
//! tokens are available; otherwise they are queued or dropped based on the
//! configured policy.
//!
//! Streams are assigned a priority class (1–4, lower = higher priority) used by
//! the scheduler to prefer high-priority traffic when the aggregate bandwidth
//! ceiling is approached.

#![allow(dead_code)]

use std::collections::HashMap;

/// Priority class for a stream (1 = highest, 4 = lowest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u8);

impl Priority {
    /// Highest priority (programme output, live events).
    pub const HIGH: Self = Self(1);
    /// Elevated priority (preview, monitoring).
    pub const MEDIUM: Self = Self(2);
    /// Normal priority.
    pub const NORMAL: Self = Self(3);
    /// Low / best-effort priority (background ingest).
    pub const LOW: Self = Self(4);
}

impl Default for Priority {
    fn default() -> Self {
        Self::NORMAL
    }
}

/// Policy applied when a stream's token bucket is empty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaperPolicy {
    /// Queue the packet until tokens are available.
    Queue,
    /// Drop the packet immediately.
    Drop,
    /// Allow the packet (burst) up to `burst_bytes` beyond the bucket.
    Burst,
}

impl Default for ShaperPolicy {
    fn default() -> Self {
        Self::Queue
    }
}

/// Token bucket state for a single stream.
#[derive(Debug, Clone)]
struct TokenBucket {
    /// Maximum number of tokens (= burst size in bytes).
    capacity: u64,
    /// Current number of tokens.
    tokens: u64,
    /// Refill rate in bytes per second.
    rate_bps: u64,
    /// Last refill timestamp in microseconds.
    last_refill_us: u64,
}

impl TokenBucket {
    fn new(rate_bps: u64, burst_bytes: u64) -> Self {
        Self {
            capacity: burst_bytes,
            tokens: burst_bytes,
            rate_bps,
            last_refill_us: 0,
        }
    }

    /// Adds tokens proportional to elapsed time.
    fn refill(&mut self, now_us: u64) {
        if now_us <= self.last_refill_us {
            return;
        }
        let elapsed_s = (now_us - self.last_refill_us) as f64 / 1_000_000.0;
        let new_tokens = (self.rate_bps as f64 * elapsed_s) as u64;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill_us = now_us;
    }

    /// Attempts to consume `bytes` tokens.  Returns `true` if successful.
    fn consume(&mut self, bytes: u64) -> bool {
        if self.tokens >= bytes {
            self.tokens -= bytes;
            true
        } else {
            false
        }
    }
}

/// Configuration for one shaped stream.
#[derive(Debug, Clone)]
pub struct StreamShapeConfig {
    /// Target rate in bytes per second.
    pub rate_bps: u64,
    /// Burst allowance in bytes (maximum instantaneous burst above the rate).
    pub burst_bytes: u64,
    /// Priority class.
    pub priority: Priority,
    /// Policy when the bucket is empty.
    pub policy: ShaperPolicy,
    /// Maximum queue depth (packets) for the `Queue` policy.
    pub max_queue_depth: usize,
}

impl Default for StreamShapeConfig {
    fn default() -> Self {
        Self {
            rate_bps: 1_000_000, // 1 MB/s
            burst_bytes: 65_536,
            priority: Priority::NORMAL,
            policy: ShaperPolicy::Queue,
            max_queue_depth: 256,
        }
    }
}

/// A shaped packet waiting in the queue.
#[derive(Debug, Clone)]
pub struct QueuedPacket {
    /// The raw packet bytes.
    pub data: Vec<u8>,
    /// Enqueue timestamp in microseconds.
    pub enqueued_at_us: u64,
}

/// Per-stream statistics.
#[derive(Debug, Clone, Default)]
pub struct StreamShapeStats {
    /// Total packets admitted.
    pub packets_admitted: u64,
    /// Total packets queued.
    pub packets_queued: u64,
    /// Total packets dropped.
    pub packets_dropped: u64,
    /// Total bytes admitted.
    pub bytes_admitted: u64,
}

/// Internal state for a shaped stream.
#[derive(Debug)]
struct ShapedStream {
    config: StreamShapeConfig,
    bucket: TokenBucket,
    queue: std::collections::VecDeque<QueuedPacket>,
    stats: StreamShapeStats,
}

impl ShapedStream {
    fn new(config: StreamShapeConfig) -> Self {
        let bucket = TokenBucket::new(config.rate_bps, config.burst_bytes);
        Self {
            config,
            bucket,
            queue: std::collections::VecDeque::new(),
            stats: StreamShapeStats::default(),
        }
    }
}

/// Error type for bandwidth shaping operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ShaperError {
    /// Stream not registered.
    #[error("stream '{0}' not registered")]
    StreamNotFound(String),
    /// Stream already registered.
    #[error("stream '{0}' already registered")]
    StreamExists(String),
    /// Aggregate rate limit exceeded.
    #[error("aggregate rate limit of {limit_bps} bps exceeded")]
    AggregateLimitExceeded {
        /// The configured aggregate limit.
        limit_bps: u64,
    },
}

/// Result type for bandwidth shaping operations.
pub type ShaperResult<T> = Result<T, ShaperError>;

/// Result of a shaper admission decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    /// Packet was admitted immediately.
    Admitted,
    /// Packet was queued for later transmission.
    Queued,
    /// Packet was dropped.
    Dropped,
}

/// A bandwidth shaper managing multiple streams with QoS prioritisation.
#[derive(Debug, Default)]
pub struct BandwidthShaper {
    /// Shaped streams indexed by stream ID (queue-aware API).
    streams: HashMap<String, ShapedStream>,
    /// Optional aggregate bandwidth ceiling in bytes per second (0 = unlimited).
    aggregate_limit_bps: u64,
    /// Current bytes per second consumed across all streams (sliding estimate).
    current_aggregate_bps: u64,
    /// Internal clock in microseconds.
    now_us: u64,
    /// Streams managed by the simple token-bucket API (`add_stream` / `record_send`).
    simple_streams: HashMap<String, SimpleStream>,
}

impl BandwidthShaper {
    /// Creates a new shaper with an optional aggregate rate limit.
    #[must_use]
    pub fn new(aggregate_limit_bps: u64) -> Self {
        Self {
            streams: HashMap::new(),
            aggregate_limit_bps,
            current_aggregate_bps: 0,
            now_us: 0,
            simple_streams: HashMap::new(),
        }
    }

    /// Advances the internal clock to `now_us`.
    pub fn set_time_us(&mut self, now_us: u64) {
        self.now_us = now_us;
        // Refill all buckets.
        for stream in self.streams.values_mut() {
            stream.bucket.refill(now_us);
        }
    }

    /// Registers a stream with the given shaping configuration.
    pub fn register_stream(
        &mut self,
        stream_id: impl Into<String>,
        config: StreamShapeConfig,
    ) -> ShaperResult<()> {
        let id = stream_id.into();
        if self.streams.contains_key(&id) {
            return Err(ShaperError::StreamExists(id));
        }
        self.streams.insert(id, ShapedStream::new(config));
        Ok(())
    }

    /// Removes a stream from the shaper.
    pub fn unregister_stream(&mut self, stream_id: &str) -> ShaperResult<()> {
        self.streams
            .remove(stream_id)
            .map(|_| ())
            .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))
    }

    /// Submits a packet for shaping.
    ///
    /// Returns the admission decision.
    pub fn submit_packet(&mut self, stream_id: &str, packet: Vec<u8>) -> ShaperResult<AdmissionDecision> {
        let byte_count = packet.len() as u64;

        // Check aggregate limit.
        if self.aggregate_limit_bps > 0
            && self.current_aggregate_bps + byte_count * 8 > self.aggregate_limit_bps
        {
            // Check if any lower-priority stream can be preempted.
            // For now, just reject if we're at the aggregate limit.
            let stream = self
                .streams
                .get_mut(stream_id)
                .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))?;
            stream.stats.packets_dropped += 1;
            return Ok(AdmissionDecision::Dropped);
        }

        let now = self.now_us;
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))?;

        stream.bucket.refill(now);

        let admitted = stream.bucket.consume(byte_count);
        if admitted {
            stream.stats.packets_admitted += 1;
            stream.stats.bytes_admitted += byte_count;
            self.current_aggregate_bps =
                self.current_aggregate_bps.saturating_add(byte_count * 8);
            Ok(AdmissionDecision::Admitted)
        } else {
            match stream.config.policy {
                ShaperPolicy::Drop => {
                    stream.stats.packets_dropped += 1;
                    Ok(AdmissionDecision::Dropped)
                }
                ShaperPolicy::Queue => {
                    if stream.queue.len() >= stream.config.max_queue_depth {
                        // Queue full → drop oldest.
                        stream.queue.pop_front();
                        stream.stats.packets_dropped += 1;
                    }
                    stream.queue.push_back(QueuedPacket {
                        data: packet,
                        enqueued_at_us: now,
                    });
                    stream.stats.packets_queued += 1;
                    Ok(AdmissionDecision::Queued)
                }
                ShaperPolicy::Burst => {
                    // Allow burst beyond bucket; set tokens to 0.
                    stream.bucket.tokens = 0;
                    stream.stats.packets_admitted += 1;
                    stream.stats.bytes_admitted += byte_count;
                    Ok(AdmissionDecision::Admitted)
                }
            }
        }
    }

    /// Drains queued packets that can now be admitted (bucket has refilled).
    ///
    /// Returns a sorted-by-priority list of `(stream_id, packet)` tuples.
    pub fn drain_ready(&mut self) -> Vec<(String, Vec<u8>)> {
        // Collect streams sorted by priority.
        let mut ready: Vec<(String, Vec<u8>)> = Vec::new();
        let now = self.now_us;

        let mut ids: Vec<String> = self.streams.keys().cloned().collect();
        // Sort by priority (lower number = higher priority).
        ids.sort_by_key(|id| {
            self.streams.get(id).map_or(u8::MAX, |s| s.config.priority.0)
        });

        for id in ids {
            let stream = match self.streams.get_mut(&id) {
                Some(s) => s,
                None => continue,
            };
            stream.bucket.refill(now);
            while let Some(pkt) = stream.queue.front() {
                let byte_count = pkt.data.len() as u64;
                if stream.bucket.consume(byte_count) {
                    // SAFETY: we just checked `front()` returned `Some`, so `pop_front()` is guaranteed.
                    let Some(pkt) = stream.queue.pop_front() else { break };
                    stream.stats.packets_admitted += 1;
                    stream.stats.bytes_admitted += byte_count;
                    ready.push((id.clone(), pkt.data));
                } else {
                    break;
                }
            }
        }
        ready
    }

    /// Returns statistics for a stream.
    pub fn stream_stats(&self, stream_id: &str) -> ShaperResult<StreamShapeStats> {
        self.streams
            .get(stream_id)
            .map(|s| s.stats.clone())
            .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))
    }

    /// Returns the current queue depth for a stream.
    pub fn queue_depth(&self, stream_id: &str) -> ShaperResult<usize> {
        self.streams
            .get(stream_id)
            .map(|s| s.queue.len())
            .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))
    }

    /// Updates the rate limit for a stream at runtime.
    pub fn update_rate(&mut self, stream_id: &str, new_rate_bps: u64) -> ShaperResult<()> {
        let stream = self
            .streams
            .get_mut(stream_id)
            .ok_or_else(|| ShaperError::StreamNotFound(stream_id.to_owned()))?;
        stream.config.rate_bps = new_rate_bps;
        stream.bucket.rate_bps = new_rate_bps;
        Ok(())
    }

    /// Returns the number of registered streams.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }
}

// ── Public Token Bucket API ───────────────────────────────────────────────────

/// A public token-bucket rate limiter that can be used standalone or embedded
/// inside a [`BandwidthShaper`] for per-stream egress control.
///
/// Tokens represent bytes. The bucket refills at `rate_bps` bytes per second
/// up to a ceiling of `burst_bytes`. Callers consume tokens when sending
/// packets; when insufficient tokens are present the caller must wait.
#[derive(Debug, Clone)]
pub struct RateLimiter {
    /// Sustained send rate in bytes per second.
    pub rate_bps: u64,
    /// Maximum instantaneous burst in bytes (bucket ceiling).
    pub burst_bytes: u64,
    /// Current number of available tokens (bytes).
    pub tokens: f64,
    /// Timestamp of the last refill in microseconds.
    pub last_refill_us: u64,
}

impl RateLimiter {
    /// Creates a new rate limiter, fully charged to `burst_bytes`.
    #[must_use]
    pub fn new_bucket(rate_bps: u64, burst_bytes: u64) -> Self {
        Self {
            rate_bps,
            burst_bytes,
            tokens: burst_bytes as f64,
            last_refill_us: 0,
        }
    }

    /// Refills tokens based on elapsed time since the last call.
    ///
    /// Tokens are capped at `burst_bytes`.
    pub fn refill(&mut self, now_us: u64) {
        if now_us <= self.last_refill_us {
            return;
        }
        let elapsed_s = (now_us - self.last_refill_us) as f64 / 1_000_000.0;
        let added = self.rate_bps as f64 * elapsed_s;
        self.tokens = (self.tokens + added).min(self.burst_bytes as f64);
        self.last_refill_us = now_us;
    }

    /// Attempts to consume `bytes` tokens at time `now_us`.
    ///
    /// Implicitly calls [`refill`](Self::refill) before testing availability.
    /// Returns `true` and deducts tokens if enough are available; returns
    /// `false` without modifying state otherwise.
    pub fn try_consume(&mut self, bytes: u64, now_us: u64) -> bool {
        self.refill(now_us);
        let need = bytes as f64;
        if self.tokens >= need {
            self.tokens -= need;
            true
        } else {
            false
        }
    }

    /// Returns the number of microseconds the caller must wait before `bytes`
    /// tokens will be available at `now_us`.
    ///
    /// Returns `0` if tokens are already available.
    #[must_use]
    pub fn wait_time_us(&mut self, bytes: u64, now_us: u64) -> u64 {
        self.refill(now_us);
        let need = bytes as f64;
        if self.tokens >= need {
            return 0;
        }
        let deficit = need - self.tokens;
        if self.rate_bps == 0 {
            return u64::MAX;
        }
        // deficit bytes / (rate bytes/s) → seconds → microseconds
        (deficit / self.rate_bps as f64 * 1_000_000.0).ceil() as u64
    }
}

// ── ShaperStats ───────────────────────────────────────────────────────────────

/// Per-stream statistics produced by [`BandwidthShaper`].
#[derive(Debug, Clone, Default)]
pub struct ShaperStats {
    /// Total bytes allowed through for this stream.
    pub allowed_bytes: u64,
    /// Total bytes dropped for this stream.
    pub dropped_bytes: u64,
    /// Estimated current throughput in bits per second (EMA).
    pub current_rate_bps: f64,
}

// ── Extended BandwidthShaper API ──────────────────────────────────────────────

/// Internal state for a stream managed by the simple per-stream shaper API.
#[derive(Debug)]
struct SimpleStream {
    bucket: RateLimiter,
    stats: ShaperStats,
    /// Last timestamp seen, for rate estimation.
    last_ts_us: u64,
    /// EMA rate estimator (bytes/s).
    rate_ema: f64,
}

impl SimpleStream {
    fn new(rate_bps: u64, burst_bytes: u64) -> Self {
        Self {
            bucket: RateLimiter::new_bucket(rate_bps, burst_bytes),
            stats: ShaperStats::default(),
            last_ts_us: 0,
            rate_ema: 0.0,
        }
    }

    fn update_rate_ema(&mut self, bytes: u64, now_us: u64) {
        if self.last_ts_us > 0 && now_us > self.last_ts_us {
            let elapsed_s = (now_us - self.last_ts_us) as f64 / 1_000_000.0;
            let instant_bps = (bytes as f64 * 8.0) / elapsed_s;
            const ALPHA: f64 = 0.125;
            self.rate_ema = (1.0 - ALPHA) * self.rate_ema + ALPHA * instant_bps;
        }
        self.last_ts_us = now_us;
    }
}

impl BandwidthShaper {
    /// Adds a new stream with the given rate and burst parameters.
    ///
    /// Does nothing if a stream with `stream_id` already exists under the
    /// simple-stream API (use [`register_stream`](Self::register_stream) for
    /// the queue-aware API).
    pub fn add_stream(&mut self, stream_id: impl Into<String>, rate_bps: u64, burst_bytes: u64) {
        let id = stream_id.into();
        self.simple_streams
            .entry(id)
            .or_insert_with(|| SimpleStream::new(rate_bps, burst_bytes));
    }

    /// Removes a stream added via [`add_stream`].
    pub fn remove_stream(&mut self, stream_id: &str) {
        self.simple_streams.remove(stream_id);
    }

    /// Returns `true` if `stream_id` may send `bytes` at time `now_us`.
    ///
    /// Does **not** consume tokens; call [`record_send`](Self::record_send)
    /// after the packet is actually transmitted.
    #[must_use]
    pub fn can_send(&mut self, stream_id: &str, bytes: u64, now_us: u64) -> bool {
        if let Some(stream) = self.simple_streams.get_mut(stream_id) {
            let mut bucket_clone = stream.bucket.clone();
            bucket_clone.try_consume(bytes, now_us)
        } else {
            false
        }
    }

    /// Records a successful send of `bytes` bytes for `stream_id` at `now_us`.
    ///
    /// Consumes tokens from the stream's bucket and updates statistics.
    pub fn record_send(&mut self, stream_id: &str, bytes: u64, now_us: u64) {
        if let Some(stream) = self.simple_streams.get_mut(stream_id) {
            let consumed = stream.bucket.try_consume(bytes, now_us);
            if consumed {
                stream.stats.allowed_bytes += bytes;
            } else {
                stream.stats.dropped_bytes += bytes;
            }
            stream.stats.current_rate_bps = {
                stream.update_rate_ema(bytes, now_us);
                stream.rate_ema
            };
        }
    }

    /// Returns a snapshot of statistics for `stream_id`, or `None` if the
    /// stream is not registered via [`add_stream`].
    #[must_use]
    pub fn stats(&self, stream_id: &str) -> Option<ShaperStats> {
        self.simple_streams
            .get(stream_id)
            .map(|s| s.stats.clone())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config(rate_bps: u64) -> StreamShapeConfig {
        StreamShapeConfig {
            rate_bps,
            burst_bytes: rate_bps, // 1 second burst
            priority: Priority::NORMAL,
            policy: ShaperPolicy::Queue,
            max_queue_depth: 16,
        }
    }

    #[test]
    fn test_admit_within_rate() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.register_stream("cam1", default_config(1_000_000)).unwrap();
        shaper.set_time_us(0);
        let decision = shaper.submit_packet("cam1", vec![0u8; 1000]).unwrap();
        assert_eq!(decision, AdmissionDecision::Admitted);
    }

    #[test]
    fn test_queue_when_bucket_empty() {
        let mut shaper = BandwidthShaper::new(0);
        // Tiny bucket: 100 bytes
        let cfg = StreamShapeConfig {
            rate_bps: 100,
            burst_bytes: 100,
            policy: ShaperPolicy::Queue,
            max_queue_depth: 16,
            ..Default::default()
        };
        shaper.register_stream("cam1", cfg).unwrap();
        shaper.set_time_us(0);
        // First 100-byte packet consumes bucket.
        shaper.submit_packet("cam1", vec![0u8; 100]).unwrap();
        // Second packet should be queued.
        let d = shaper.submit_packet("cam1", vec![0u8; 10]).unwrap();
        assert_eq!(d, AdmissionDecision::Queued);
    }

    #[test]
    fn test_drop_when_bucket_empty() {
        let mut shaper = BandwidthShaper::new(0);
        let cfg = StreamShapeConfig {
            rate_bps: 100,
            burst_bytes: 50,
            policy: ShaperPolicy::Drop,
            max_queue_depth: 16,
            ..Default::default()
        };
        shaper.register_stream("cam1", cfg).unwrap();
        shaper.set_time_us(0);
        shaper.submit_packet("cam1", vec![0u8; 50]).unwrap();
        let d = shaper.submit_packet("cam1", vec![0u8; 10]).unwrap();
        assert_eq!(d, AdmissionDecision::Dropped);
    }

    #[test]
    fn test_drain_ready_after_refill() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.register_stream("cam1", default_config(1000)).unwrap();
        shaper.set_time_us(0);
        // Exhaust bucket.
        shaper.submit_packet("cam1", vec![0u8; 1000]).unwrap();
        // Queue one more packet.
        let d = shaper.submit_packet("cam1", vec![0u8; 100]).unwrap();
        assert_eq!(d, AdmissionDecision::Queued);
        // Advance 1 second → bucket refills.
        shaper.set_time_us(1_000_000);
        let drained = shaper.drain_ready();
        assert_eq!(drained.len(), 1);
    }

    #[test]
    fn test_update_rate() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.register_stream("cam1", default_config(1000)).unwrap();
        shaper.update_rate("cam1", 2000).unwrap();
        let stats = shaper.stream_stats("cam1");
        assert!(stats.is_ok());
    }

    #[test]
    fn test_unknown_stream_error() {
        let mut shaper = BandwidthShaper::new(0);
        let result = shaper.submit_packet("nope", vec![0u8; 10]);
        assert!(matches!(result, Err(ShaperError::StreamNotFound(_))));
    }

    #[test]
    fn test_duplicate_stream_rejected() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.register_stream("cam1", default_config(1000)).unwrap();
        let result = shaper.register_stream("cam1", default_config(1000));
        assert!(matches!(result, Err(ShaperError::StreamExists(_))));
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::HIGH < Priority::LOW);
        assert!(Priority::MEDIUM < Priority::NORMAL);
    }

    #[test]
    fn test_stream_count() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.register_stream("a", default_config(1000)).unwrap();
        shaper.register_stream("b", default_config(1000)).unwrap();
        assert_eq!(shaper.stream_count(), 2);
        shaper.unregister_stream("a").unwrap();
        assert_eq!(shaper.stream_count(), 1);
    }

    // ── TokenBucket public API tests ──────────────────────────────────────────

    // 9. Token refill rate: tokens accumulate at the configured rate.
    #[test]
    fn test_token_refill_rate() {
        let mut bucket = RateLimiter::new_bucket(1_000, 2_000); // 1 KB/s, 2 KB burst
        // Drain all tokens.
        bucket.tokens = 0.0;
        bucket.last_refill_us = 0;
        // Advance 1 second → should refill 1_000 bytes.
        bucket.refill(1_000_000);
        assert!(
            bucket.tokens >= 990.0,
            "expected ~1000 tokens, got {}",
            bucket.tokens
        );
    }

    // 10. Burst allowed: initial burst lets large send through.
    #[test]
    fn test_burst_allowed() {
        let mut bucket = RateLimiter::new_bucket(100, 10_000); // 100 B/s, 10 KB burst
        // Bucket starts full at 10_000.
        assert!(bucket.try_consume(10_000, 0), "burst should be allowed");
        assert!(!bucket.try_consume(1, 0), "no tokens left after burst");
    }

    // 11. Sustained rate limited: after burst, rate limits apply.
    #[test]
    fn test_sustained_rate_limited() {
        let rate = 1_000_u64; // 1 KB/s
        let mut bucket = RateLimiter::new_bucket(rate, rate); // 1 s burst = rate bytes
        // Consume entire burst.
        assert!(bucket.try_consume(rate, 0));
        // Immediately: no tokens remaining.
        assert!(!bucket.try_consume(1, 0));
        // After 0.5 s → ~500 bytes refilled.
        assert!(bucket.try_consume(499, 500_000));
    }

    // 12. Stream independence: two streams don't interfere.
    #[test]
    fn test_stream_independence() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.add_stream("s1", 1_000, 1_000);
        shaper.add_stream("s2", 1_000, 1_000);
        // Drain s1.
        assert!(shaper.can_send("s1", 1_000, 0));
        shaper.record_send("s1", 1_000, 0);
        // s2 should still have full bucket.
        assert!(shaper.can_send("s2", 1_000, 0));
    }

    // 13. Wait time calculation: correct wait before bytes available.
    #[test]
    fn test_wait_time_calculation() {
        let mut bucket = RateLimiter::new_bucket(1_000, 1_000); // 1 KB/s
        bucket.tokens = 0.0;
        bucket.last_refill_us = 0;
        // Want 500 bytes at time 0 → need to wait 500 ms = 500_000 us.
        let wait = bucket.wait_time_us(500, 0);
        assert!(
            wait >= 499_000 && wait <= 501_000,
            "wait_time_us should be ~500_000 us, got {wait}"
        );
    }

    // 14. add_stream / remove_stream lifecycle.
    #[test]
    fn test_add_remove_stream_lifecycle() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.add_stream("cam1", 1_000_000, 65_536);
        assert!(shaper.stats("cam1").is_some());
        shaper.remove_stream("cam1");
        assert!(shaper.stats("cam1").is_none());
    }

    // 15. can_send + record_send tracks allowed bytes in stats.
    #[test]
    fn test_can_send_and_record_send_stats() {
        let mut shaper = BandwidthShaper::new(0);
        shaper.add_stream("cam1", 1_000_000, 100_000);
        assert!(shaper.can_send("cam1", 1_000, 0));
        shaper.record_send("cam1", 1_000, 0);
        let stats = shaper.stats("cam1").expect("stream exists");
        assert_eq!(stats.allowed_bytes, 1_000);
        assert_eq!(stats.dropped_bytes, 0);
    }

    // 16. stats returns None for unknown stream.
    #[test]
    fn test_stats_unknown_stream() {
        let shaper = BandwidthShaper::new(0);
        assert!(shaper.stats("ghost").is_none());
    }
}
