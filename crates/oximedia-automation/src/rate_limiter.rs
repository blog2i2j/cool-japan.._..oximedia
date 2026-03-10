#![allow(dead_code)]
//! Rate limiting for broadcast automation commands.
//!
//! Prevents overwhelming devices and downstream systems by enforcing
//! per-channel and per-device rate limits on automation commands.
//! Supports sliding-window and token-bucket algorithms, with configurable
//! burst allowances and cooldown periods.

use std::collections::HashMap;
use std::fmt;

/// Policy determining how rate limits are enforced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitPolicy {
    /// Fixed window: counts events in discrete time windows.
    FixedWindow,
    /// Sliding window: uses a rolling time window.
    SlidingWindow,
    /// Token bucket: allows configurable burst.
    TokenBucket,
}

impl fmt::Display for RateLimitPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FixedWindow => write!(f, "FixedWindow"),
            Self::SlidingWindow => write!(f, "SlidingWindow"),
            Self::TokenBucket => write!(f, "TokenBucket"),
        }
    }
}

/// Configuration for a single rate limit rule.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of commands allowed in the window.
    pub max_commands: u64,
    /// Window duration in milliseconds.
    pub window_ms: u64,
    /// Maximum burst size (for token bucket).
    pub burst_size: u64,
    /// Cooldown period in ms after limit is hit.
    pub cooldown_ms: u64,
    /// Policy to use.
    pub policy: RateLimitPolicy,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_commands: 60,
            window_ms: 1000,
            burst_size: 10,
            cooldown_ms: 500,
            policy: RateLimitPolicy::SlidingWindow,
        }
    }
}

/// Result of a rate limit check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitResult {
    /// Command is allowed.
    Allowed,
    /// Command is rate-limited; must wait.
    Limited {
        /// Milliseconds until the next command is allowed.
        retry_after_ms: u64,
    },
    /// In cooldown period after a burst.
    Cooldown {
        /// Milliseconds remaining in cooldown.
        remaining_ms: u64,
    },
}

impl RateLimitResult {
    /// Returns true if the command is allowed.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

impl fmt::Display for RateLimitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allowed => write!(f, "Allowed"),
            Self::Limited { retry_after_ms } => {
                write!(f, "Limited (retry after {retry_after_ms}ms)")
            }
            Self::Cooldown { remaining_ms } => {
                write!(f, "Cooldown ({remaining_ms}ms remaining)")
            }
        }
    }
}

/// Internal state for a sliding-window rate limiter.
#[derive(Debug, Clone)]
struct SlidingWindowState {
    /// Timestamps (in ms) of recent commands.
    timestamps: Vec<u64>,
    /// Cooldown end timestamp (ms), or 0 if not in cooldown.
    cooldown_until: u64,
}

impl SlidingWindowState {
    fn new() -> Self {
        Self {
            timestamps: Vec::new(),
            cooldown_until: 0,
        }
    }
}

/// Internal state for a token-bucket rate limiter.
#[derive(Debug, Clone)]
struct TokenBucketState {
    /// Current number of tokens.
    tokens: u64,
    /// Timestamp of last refill (ms).
    last_refill_ms: u64,
    /// Cooldown end timestamp (ms), or 0 if not in cooldown.
    cooldown_until: u64,
}

impl TokenBucketState {
    fn new(initial_tokens: u64, now_ms: u64) -> Self {
        Self {
            tokens: initial_tokens,
            last_refill_ms: now_ms,
            cooldown_until: 0,
        }
    }
}

/// Statistics for a rate limiter scope.
#[derive(Debug, Clone, Default)]
pub struct RateLimitStats {
    /// Total commands checked.
    pub total_checked: u64,
    /// Commands allowed.
    pub total_allowed: u64,
    /// Commands rate-limited.
    pub total_limited: u64,
    /// Cooldown periods triggered.
    pub cooldowns_triggered: u64,
}

impl RateLimitStats {
    /// Returns the rejection rate as a fraction (0.0-1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rejection_rate(&self) -> f64 {
        if self.total_checked == 0 {
            return 0.0;
        }
        self.total_limited as f64 / self.total_checked as f64
    }
}

/// A rate limiter engine that manages per-scope rate limits.
///
/// Each scope (e.g. channel ID, device ID) has its own independent
/// rate-limiting state.
#[derive(Debug)]
pub struct RateLimiterEngine {
    /// Configuration.
    config: RateLimitConfig,
    /// Per-scope sliding window state.
    sliding_states: HashMap<String, SlidingWindowState>,
    /// Per-scope token bucket state.
    bucket_states: HashMap<String, TokenBucketState>,
    /// Per-scope statistics.
    stats: HashMap<String, RateLimitStats>,
}

impl RateLimiterEngine {
    /// Create a new rate limiter engine with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            sliding_states: HashMap::new(),
            bucket_states: HashMap::new(),
            stats: HashMap::new(),
        }
    }

    /// Check whether a command is allowed for the given scope at time `now_ms`.
    pub fn check(&mut self, scope: &str, now_ms: u64) -> RateLimitResult {
        let stats = self.stats.entry(scope.to_string()).or_default();
        stats.total_checked += 1;

        let result = match self.config.policy {
            RateLimitPolicy::SlidingWindow | RateLimitPolicy::FixedWindow => {
                self.check_sliding(scope, now_ms)
            }
            RateLimitPolicy::TokenBucket => self.check_token_bucket(scope, now_ms),
        };

        let stats = self.stats.entry(scope.to_string()).or_default();
        match &result {
            RateLimitResult::Allowed => stats.total_allowed += 1,
            RateLimitResult::Limited { .. } => stats.total_limited += 1,
            RateLimitResult::Cooldown { .. } => {
                stats.total_limited += 1;
                stats.cooldowns_triggered += 1;
            }
        }

        result
    }

    /// Sliding window check.
    fn check_sliding(&mut self, scope: &str, now_ms: u64) -> RateLimitResult {
        let state = self
            .sliding_states
            .entry(scope.to_string())
            .or_insert_with(SlidingWindowState::new);

        // Check cooldown
        if now_ms < state.cooldown_until {
            return RateLimitResult::Cooldown {
                remaining_ms: state.cooldown_until - now_ms,
            };
        }

        // Prune timestamps outside the window
        let window_start = now_ms.saturating_sub(self.config.window_ms);
        state.timestamps.retain(|&ts| ts >= window_start);

        if state.timestamps.len() as u64 >= self.config.max_commands {
            // Enter cooldown
            state.cooldown_until = now_ms + self.config.cooldown_ms;
            let oldest = state.timestamps.first().copied().unwrap_or(now_ms);
            let retry = oldest + self.config.window_ms - now_ms;
            return RateLimitResult::Limited {
                retry_after_ms: retry.max(1),
            };
        }

        state.timestamps.push(now_ms);
        RateLimitResult::Allowed
    }

    /// Token bucket check.
    fn check_token_bucket(&mut self, scope: &str, now_ms: u64) -> RateLimitResult {
        let config = &self.config;
        let state = self
            .bucket_states
            .entry(scope.to_string())
            .or_insert_with(|| TokenBucketState::new(config.burst_size, now_ms));

        // Check cooldown
        if now_ms < state.cooldown_until {
            return RateLimitResult::Cooldown {
                remaining_ms: state.cooldown_until - now_ms,
            };
        }

        // Refill tokens
        let elapsed = now_ms.saturating_sub(state.last_refill_ms);
        if self.config.window_ms > 0 {
            let refill = elapsed * self.config.max_commands / self.config.window_ms;
            state.tokens = (state.tokens + refill).min(self.config.burst_size);
        }
        state.last_refill_ms = now_ms;

        if state.tokens == 0 {
            state.cooldown_until = now_ms + self.config.cooldown_ms;
            return RateLimitResult::Limited {
                retry_after_ms: self.config.cooldown_ms.max(1),
            };
        }

        state.tokens -= 1;
        RateLimitResult::Allowed
    }

    /// Get statistics for a given scope.
    pub fn stats(&self, scope: &str) -> Option<&RateLimitStats> {
        self.stats.get(scope)
    }

    /// Reset all state and statistics.
    pub fn reset(&mut self) {
        self.sliding_states.clear();
        self.bucket_states.clear();
        self.stats.clear();
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Update the configuration (does not clear existing state).
    pub fn set_config(&mut self, config: RateLimitConfig) {
        self.config = config;
    }

    /// Number of tracked scopes.
    pub fn scope_count(&self) -> usize {
        self.stats.len()
    }
}

impl Default for RateLimiterEngine {
    fn default() -> Self {
        Self::new(RateLimitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_policy_display() {
        assert_eq!(RateLimitPolicy::FixedWindow.to_string(), "FixedWindow");
        assert_eq!(RateLimitPolicy::SlidingWindow.to_string(), "SlidingWindow");
        assert_eq!(RateLimitPolicy::TokenBucket.to_string(), "TokenBucket");
    }

    #[test]
    fn test_rate_limit_config_default() {
        let cfg = RateLimitConfig::default();
        assert_eq!(cfg.max_commands, 60);
        assert_eq!(cfg.window_ms, 1000);
        assert_eq!(cfg.burst_size, 10);
        assert_eq!(cfg.cooldown_ms, 500);
        assert_eq!(cfg.policy, RateLimitPolicy::SlidingWindow);
    }

    #[test]
    fn test_rate_limit_result_display() {
        assert_eq!(RateLimitResult::Allowed.to_string(), "Allowed");
        let limited = RateLimitResult::Limited { retry_after_ms: 42 };
        assert!(limited.to_string().contains("42"));
        let cooldown = RateLimitResult::Cooldown { remaining_ms: 10 };
        assert!(cooldown.to_string().contains("10"));
    }

    #[test]
    fn test_rate_limit_result_is_allowed() {
        assert!(RateLimitResult::Allowed.is_allowed());
        assert!(!RateLimitResult::Limited { retry_after_ms: 1 }.is_allowed());
        assert!(!RateLimitResult::Cooldown { remaining_ms: 1 }.is_allowed());
    }

    #[test]
    fn test_sliding_window_allows_within_limit() {
        let config = RateLimitConfig {
            max_commands: 3,
            window_ms: 1000,
            cooldown_ms: 100,
            ..Default::default()
        };
        let mut engine = RateLimiterEngine::new(config);
        assert!(engine.check("ch1", 100).is_allowed());
        assert!(engine.check("ch1", 200).is_allowed());
        assert!(engine.check("ch1", 300).is_allowed());
    }

    #[test]
    fn test_sliding_window_limits_over_max() {
        let config = RateLimitConfig {
            max_commands: 2,
            window_ms: 1000,
            cooldown_ms: 100,
            ..Default::default()
        };
        let mut engine = RateLimiterEngine::new(config);
        assert!(engine.check("ch1", 100).is_allowed());
        assert!(engine.check("ch1", 200).is_allowed());
        let result = engine.check("ch1", 300);
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_sliding_window_allows_after_window_passes() {
        let config = RateLimitConfig {
            max_commands: 2,
            window_ms: 100,
            cooldown_ms: 50,
            ..Default::default()
        };
        let mut engine = RateLimiterEngine::new(config);
        assert!(engine.check("ch1", 0).is_allowed());
        assert!(engine.check("ch1", 10).is_allowed());
        // Blocked + cooldown at t=20
        assert!(!engine.check("ch1", 20).is_allowed());
        // After cooldown (50ms) and window (100ms) pass
        assert!(engine.check("ch1", 200).is_allowed());
    }

    #[test]
    fn test_token_bucket_allows_burst() {
        let config = RateLimitConfig {
            max_commands: 10,
            window_ms: 1000,
            burst_size: 3,
            cooldown_ms: 100,
            policy: RateLimitPolicy::TokenBucket,
        };
        let mut engine = RateLimiterEngine::new(config);
        // All three burst tokens should be available
        assert!(engine.check("dev1", 0).is_allowed());
        assert!(engine.check("dev1", 0).is_allowed());
        assert!(engine.check("dev1", 0).is_allowed());
    }

    #[test]
    fn test_token_bucket_limits_after_burst() {
        let config = RateLimitConfig {
            max_commands: 10,
            window_ms: 1000,
            burst_size: 2,
            cooldown_ms: 50,
            policy: RateLimitPolicy::TokenBucket,
        };
        let mut engine = RateLimiterEngine::new(config);
        assert!(engine.check("dev1", 0).is_allowed());
        assert!(engine.check("dev1", 0).is_allowed());
        let result = engine.check("dev1", 0);
        assert!(!result.is_allowed());
    }

    #[test]
    fn test_independent_scopes() {
        let config = RateLimitConfig {
            max_commands: 1,
            window_ms: 1000,
            cooldown_ms: 100,
            ..Default::default()
        };
        let mut engine = RateLimiterEngine::new(config);
        assert!(engine.check("ch1", 0).is_allowed());
        assert!(engine.check("ch2", 0).is_allowed());
        // ch1 should now be limited
        assert!(!engine.check("ch1", 0).is_allowed());
        // ch2 should also be limited
        assert!(!engine.check("ch2", 0).is_allowed());
    }

    #[test]
    fn test_stats_tracking() {
        let config = RateLimitConfig {
            max_commands: 1,
            window_ms: 1000,
            cooldown_ms: 100,
            ..Default::default()
        };
        let mut engine = RateLimiterEngine::new(config);
        engine.check("ch1", 0);
        engine.check("ch1", 0);
        let stats = engine.stats("ch1").expect("stats should succeed");
        assert_eq!(stats.total_checked, 2);
        assert_eq!(stats.total_allowed, 1);
        assert_eq!(stats.total_limited, 1);
    }

    #[test]
    fn test_stats_rejection_rate() {
        let mut stats = RateLimitStats::default();
        assert!((stats.rejection_rate() - 0.0).abs() < f64::EPSILON);
        stats.total_checked = 4;
        stats.total_limited = 1;
        assert!((stats.rejection_rate() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_engine_reset() {
        let mut engine = RateLimiterEngine::default();
        engine.check("ch1", 0);
        assert_eq!(engine.scope_count(), 1);
        engine.reset();
        assert_eq!(engine.scope_count(), 0);
    }

    #[test]
    fn test_engine_set_config() {
        let mut engine = RateLimiterEngine::default();
        assert_eq!(engine.config().max_commands, 60);
        let new_cfg = RateLimitConfig {
            max_commands: 100,
            ..Default::default()
        };
        engine.set_config(new_cfg);
        assert_eq!(engine.config().max_commands, 100);
    }
}
