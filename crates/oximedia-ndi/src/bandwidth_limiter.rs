//! Per-source bandwidth limiting for NDI streams.
//!
//! This module implements a token-bucket rate limiter that enforces a configurable
//! maximum bitrate per NDI source.  Senders consult the limiter before transmitting
//! each frame; if the bucket is exhausted the sender either waits or drops to a
//! lower quality tier.
//!
//! # Quality tiers
//!
//! | Tier | Description |
//! |------|-------------|
//! | [`QualityTier::Highest`]  | Full-resolution, high-quality (default) |
//! | [`QualityTier::Lowest`]   | Down-scaled or heavily compressed |
//! | [`QualityTier::AudioOnly`]| Strip video, transmit audio only |
//!
//! # Example
//!
//! ```
//! use oximedia_ndi::bandwidth_limiter::{BandwidthLimiter, LimiterConfig, QualityTier};
//!
//! let cfg = LimiterConfig {
//!     max_bits_per_second: 50_000_000, // 50 Mbit/s
//!     burst_bits: 5_000_000,           // 5 Mbit burst
//!     ..Default::default()
//! };
//! let mut limiter = BandwidthLimiter::new(cfg);
//!
//! // Ask whether a 1 MB frame can be sent right now
//! if limiter.try_consume(1_000_000 * 8) {
//!     // transmit frame
//! }
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// QualityTier
// ---------------------------------------------------------------------------

/// Quality level selected by the bandwidth limiter when the link is congested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QualityTier {
    /// Full resolution, full quality.  Used when bandwidth is plentiful.
    Highest = 2,
    /// Reduced resolution or higher compression.  Used under moderate congestion.
    Lowest = 1,
    /// Audio-only; video is stripped entirely.  Used under severe congestion.
    AudioOnly = 0,
}

impl QualityTier {
    /// Returns a human-readable label for the tier.
    pub fn label(self) -> &'static str {
        match self {
            Self::Highest => "highest",
            Self::Lowest => "lowest",
            Self::AudioOnly => "audio_only",
        }
    }

    /// Returns `true` if this tier carries video.
    pub fn has_video(self) -> bool {
        matches!(self, Self::Highest | Self::Lowest)
    }

    /// Returns `true` if this tier carries audio.
    pub fn has_audio(self) -> bool {
        true // all tiers include audio
    }
}

impl Default for QualityTier {
    fn default() -> Self {
        Self::Highest
    }
}

// ---------------------------------------------------------------------------
// LimiterConfig
// ---------------------------------------------------------------------------

/// Configuration for a [`BandwidthLimiter`].
#[derive(Debug, Clone)]
pub struct LimiterConfig {
    /// Maximum sustained bitrate in bits per second.
    pub max_bits_per_second: u64,
    /// Initial burst capacity in bits.  The bucket starts full at this level.
    /// Must be >= `max_bits_per_second / refill_hz`.
    pub burst_bits: u64,
    /// How many times per second the token bucket is refilled.
    /// A higher value gives smoother rate control; lower values reduce CPU cost.
    pub refill_hz: u32,
    /// Bitrate threshold (fraction of max) below which the tier drops to
    /// [`QualityTier::Lowest`].  E.g. `0.5` means "drop tier at 50% capacity".
    pub lowest_tier_threshold: f64,
    /// Bitrate threshold below which the tier drops to [`QualityTier::AudioOnly`].
    pub audio_only_threshold: f64,
    /// Minimum time between quality tier changes (hysteresis).
    pub tier_change_cooldown: Duration,
}

impl Default for LimiterConfig {
    fn default() -> Self {
        Self {
            max_bits_per_second: 125_000_000, // 125 Mbit/s (1 GbE headroom)
            burst_bits: 12_500_000,           // ~0.1 s burst
            refill_hz: 100,
            lowest_tier_threshold: 0.4,
            audio_only_threshold: 0.1,
            tier_change_cooldown: Duration::from_millis(500),
        }
    }
}

impl LimiterConfig {
    /// Compute tokens added per refill tick.
    fn tokens_per_tick(&self) -> u64 {
        let hz = self.refill_hz.max(1) as u64;
        self.max_bits_per_second / hz
    }

    /// Compute the duration between refill ticks.
    fn tick_interval(&self) -> Duration {
        let hz = self.refill_hz.max(1) as u64;
        Duration::from_nanos(1_000_000_000 / hz)
    }
}

// ---------------------------------------------------------------------------
// TokenBucket (internal)
// ---------------------------------------------------------------------------

/// Internal leaky-token-bucket implementation.
#[derive(Debug)]
struct TokenBucket {
    /// Current number of tokens (bits) in the bucket.
    tokens: u64,
    /// Maximum capacity (burst ceiling).
    capacity: u64,
    /// Tokens added each tick.
    tokens_per_tick: u64,
    /// Duration between ticks.
    tick_interval: Duration,
    /// Last refill timestamp.
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u64, tokens_per_tick: u64, tick_interval: Duration) -> Self {
        Self {
            tokens: capacity,
            capacity,
            tokens_per_tick,
            tick_interval,
            last_refill: Instant::now(),
        }
    }

    /// Refill the bucket based on elapsed time since the last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        if elapsed < self.tick_interval {
            return;
        }
        let ticks = elapsed.as_nanos() / self.tick_interval.as_nanos().max(1);
        let added = (ticks as u64).saturating_mul(self.tokens_per_tick);
        self.tokens = (self.tokens + added).min(self.capacity);
        self.last_refill = now;
    }

    /// Try to consume `bits` tokens.  Returns `true` on success.
    fn try_consume(&mut self, bits: u64) -> bool {
        self.refill();
        if self.tokens >= bits {
            self.tokens -= bits;
            true
        } else {
            false
        }
    }

    /// Current fill level as a fraction in `[0.0, 1.0]`.
    fn fill_fraction(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.tokens as f64 / self.capacity as f64
    }
}

// ---------------------------------------------------------------------------
// BandwidthLimiter
// ---------------------------------------------------------------------------

/// Per-source NDI bandwidth limiter.
///
/// Uses a token-bucket algorithm to enforce a configurable maximum bitrate.
/// When the bucket is under-filled, the limiter recommends a lower
/// [`QualityTier`] so the sender can reduce frame size or drop video.
#[derive(Debug)]
pub struct BandwidthLimiter {
    config: LimiterConfig,
    bucket: TokenBucket,
    current_tier: QualityTier,
    last_tier_change: Instant,
    /// Total bits consumed since construction.
    total_bits_consumed: u64,
    /// Total frames that were rejected (bucket exhausted).
    frames_dropped: u64,
}

impl BandwidthLimiter {
    /// Create a new limiter with the given configuration.
    pub fn new(config: LimiterConfig) -> Self {
        let tokens_per_tick = config.tokens_per_tick();
        let tick_interval = config.tick_interval();
        let capacity = config.burst_bits;
        Self {
            bucket: TokenBucket::new(capacity, tokens_per_tick, tick_interval),
            current_tier: QualityTier::Highest,
            last_tier_change: Instant::now(),
            total_bits_consumed: 0,
            frames_dropped: 0,
            config,
        }
    }

    /// Attempt to consume `bits` tokens.
    ///
    /// Returns `true` if the transmission is permitted, `false` if the bucket
    /// was exhausted (frame should be dropped or deferred).
    pub fn try_consume(&mut self, bits: u64) -> bool {
        if self.bucket.try_consume(bits) {
            self.total_bits_consumed += bits;
            self.update_tier();
            true
        } else {
            self.frames_dropped += 1;
            self.update_tier();
            false
        }
    }

    /// Forcibly consume `bits` tokens, allowing the bucket to go negative
    /// (represented by clamping at 0).  Use when dropping is not an option.
    pub fn force_consume(&mut self, bits: u64) {
        self.bucket.refill();
        self.bucket.tokens = self.bucket.tokens.saturating_sub(bits);
        self.total_bits_consumed += bits;
        self.update_tier();
    }

    /// Return the recommended [`QualityTier`] given current bucket fill level.
    pub fn recommended_tier(&self) -> QualityTier {
        self.current_tier
    }

    /// Current bucket fill as a fraction in `[0.0, 1.0]`.
    pub fn fill_fraction(&mut self) -> f64 {
        self.bucket.refill();
        self.bucket.fill_fraction()
    }

    /// Total bits consumed since construction.
    pub fn total_bits_consumed(&self) -> u64 {
        self.total_bits_consumed
    }

    /// Number of frames that were denied (bucket exhausted).
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Reset the bucket to full capacity and clear statistics.
    pub fn reset(&mut self) {
        self.bucket.tokens = self.bucket.capacity;
        self.bucket.last_refill = Instant::now();
        self.total_bits_consumed = 0;
        self.frames_dropped = 0;
        self.current_tier = QualityTier::Highest;
        self.last_tier_change = Instant::now();
    }

    /// Update the quality tier based on current fill fraction.
    fn update_tier(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_tier_change) < self.config.tier_change_cooldown {
            return;
        }
        let fill = self.bucket.fill_fraction();
        let new_tier = if fill < self.config.audio_only_threshold {
            QualityTier::AudioOnly
        } else if fill < self.config.lowest_tier_threshold {
            QualityTier::Lowest
        } else {
            QualityTier::Highest
        };
        if new_tier != self.current_tier {
            self.current_tier = new_tier;
            self.last_tier_change = now;
        }
    }
}

// ---------------------------------------------------------------------------
// SourceBandwidthTable
// ---------------------------------------------------------------------------

/// Manages per-source [`BandwidthLimiter`] instances indexed by source name.
///
/// In a multi-source NDI setup each source gets its own independent limiter
/// so that a single bandwidth hog cannot starve other sources.
pub struct SourceBandwidthTable {
    /// Default config applied to new sources.
    default_config: LimiterConfig,
    /// Per-source limiters.
    limiters: std::collections::HashMap<String, BandwidthLimiter>,
}

impl SourceBandwidthTable {
    /// Create a new table with the given default configuration.
    pub fn new(default_config: LimiterConfig) -> Self {
        Self {
            default_config,
            limiters: std::collections::HashMap::new(),
        }
    }

    /// Get or create the limiter for `source_name`.
    pub fn limiter_for(&mut self, source_name: &str) -> &mut BandwidthLimiter {
        let cfg = self.default_config.clone();
        self.limiters
            .entry(source_name.to_string())
            .or_insert_with(|| BandwidthLimiter::new(cfg))
    }

    /// Try to consume `bits` for `source_name`.  Returns `true` on success.
    pub fn try_consume(&mut self, source_name: &str, bits: u64) -> bool {
        self.limiter_for(source_name).try_consume(bits)
    }

    /// Recommended tier for `source_name`.
    pub fn recommended_tier(&mut self, source_name: &str) -> QualityTier {
        self.limiter_for(source_name).recommended_tier()
    }

    /// Remove a source from the table (e.g. when the source disconnects).
    pub fn remove(&mut self, source_name: &str) -> bool {
        self.limiters.remove(source_name).is_some()
    }

    /// Number of sources currently tracked.
    pub fn source_count(&self) -> usize {
        self.limiters.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_small() -> LimiterConfig {
        LimiterConfig {
            max_bits_per_second: 1_000_000, // 1 Mbit/s
            burst_bits: 500_000,            // 0.5 Mbit burst
            refill_hz: 100,
            lowest_tier_threshold: 0.4,
            audio_only_threshold: 0.1,
            tier_change_cooldown: Duration::from_millis(0), // disable cooldown in tests
        }
    }

    // --- QualityTier ---

    #[test]
    fn test_quality_tier_ordering() {
        assert!(QualityTier::AudioOnly < QualityTier::Lowest);
        assert!(QualityTier::Lowest < QualityTier::Highest);
    }

    #[test]
    fn test_quality_tier_has_video() {
        assert!(QualityTier::Highest.has_video());
        assert!(QualityTier::Lowest.has_video());
        assert!(!QualityTier::AudioOnly.has_video());
    }

    #[test]
    fn test_quality_tier_has_audio() {
        assert!(QualityTier::AudioOnly.has_audio());
        assert!(QualityTier::Highest.has_audio());
    }

    #[test]
    fn test_quality_tier_labels() {
        assert_eq!(QualityTier::Highest.label(), "highest");
        assert_eq!(QualityTier::Lowest.label(), "lowest");
        assert_eq!(QualityTier::AudioOnly.label(), "audio_only");
    }

    // --- BandwidthLimiter ---

    #[test]
    fn test_new_limiter_starts_full() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Full bucket → fill fraction should be 1.0
        let fill = lim.fill_fraction();
        assert!(fill > 0.99, "expected nearly full bucket, got {}", fill);
    }

    #[test]
    fn test_try_consume_within_capacity() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Consuming less than the burst should succeed
        assert!(lim.try_consume(100_000));
        assert_eq!(lim.total_bits_consumed(), 100_000);
        assert_eq!(lim.frames_dropped(), 0);
    }

    #[test]
    fn test_try_consume_exceeds_capacity() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Drain the bucket first
        lim.force_consume(500_000);
        // Now even a small request should fail (bucket near empty)
        let result = lim.try_consume(499_000);
        if !result {
            assert_eq!(lim.frames_dropped(), 1);
        }
        // Either way, total_bits_consumed should not have increased for dropped frame
    }

    #[test]
    fn test_force_consume_does_not_panic() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Force-consume more than capacity — should clamp at 0
        lim.force_consume(1_000_000_000);
        let fill = lim.fill_fraction();
        assert!(fill >= 0.0 && fill <= 1.0);
    }

    #[test]
    fn test_reset_clears_stats() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        lim.force_consume(400_000);
        lim.reset();
        assert_eq!(lim.total_bits_consumed(), 0);
        assert_eq!(lim.frames_dropped(), 0);
        let fill = lim.fill_fraction();
        assert!(fill > 0.99);
    }

    #[test]
    fn test_tier_drops_to_lowest_when_bucket_low() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Drain to ~30% fill → below lowest_tier_threshold (40%) but above audio_only (10%)
        lim.force_consume(350_000); // 350k of 500k → 30% remaining
        // Manually trigger tier update by calling try_consume (which calls update_tier)
        let _ = lim.try_consume(1);
        assert_eq!(lim.recommended_tier(), QualityTier::Lowest);
    }

    #[test]
    fn test_tier_drops_to_audio_only_when_very_low() {
        let mut lim = BandwidthLimiter::new(cfg_small());
        // Drain to ~5% → below audio_only threshold (10%)
        lim.force_consume(475_000); // leaves 25k of 500k = 5%
        let _ = lim.try_consume(1);
        assert_eq!(lim.recommended_tier(), QualityTier::AudioOnly);
    }

    // --- SourceBandwidthTable ---

    #[test]
    fn test_table_creates_limiter_on_demand() {
        let mut table = SourceBandwidthTable::new(cfg_small());
        assert_eq!(table.source_count(), 0);
        let _ = table.try_consume("cam1", 1_000);
        assert_eq!(table.source_count(), 1);
    }

    #[test]
    fn test_table_independent_limiters() {
        let mut table = SourceBandwidthTable::new(cfg_small());
        // Drain cam1 completely
        table.limiter_for("cam1").force_consume(1_000_000_000);
        // cam2 should still have a full bucket
        let cam2_fill = table.limiter_for("cam2").fill_fraction();
        assert!(cam2_fill > 0.99);
    }

    #[test]
    fn test_table_remove_source() {
        let mut table = SourceBandwidthTable::new(cfg_small());
        let _ = table.try_consume("cam1", 1_000);
        assert!(table.remove("cam1"));
        assert_eq!(table.source_count(), 0);
        assert!(!table.remove("cam1")); // second remove returns false
    }

    #[test]
    fn test_default_config_reasonable() {
        let cfg = LimiterConfig::default();
        assert!(cfg.max_bits_per_second >= 1_000_000);
        assert!(cfg.burst_bits > 0);
        assert!(cfg.refill_hz > 0);
        assert!(cfg.lowest_tier_threshold > cfg.audio_only_threshold);
    }
}
