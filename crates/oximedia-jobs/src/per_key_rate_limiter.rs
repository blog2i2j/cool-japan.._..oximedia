// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Per-user / per-tag job submission rate limiting.
//!
//! Implements a sliding-window counter that tracks request timestamps over a
//! rolling time window for each `RateLimitKey`.  Independent limits can be
//! applied at three levels: per user, per tag, and globally.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// RateLimitKey
// ---------------------------------------------------------------------------

/// Identifies the subject of a rate limit check.
///
/// Either `user_id`, `tag`, or both may be `Some`.  The string representation
/// is used as the hash map key, so callers may also construct arbitrary keys.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RateLimitKey {
    /// Optional user identifier.
    pub user_id: Option<String>,
    /// Optional job tag.
    pub tag: Option<String>,
}

impl RateLimitKey {
    /// Create a key scoped to a specific user.
    #[must_use]
    pub fn user(user_id: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            tag: None,
        }
    }

    /// Create a key scoped to a specific tag.
    #[must_use]
    pub fn tag(tag: impl Into<String>) -> Self {
        Self {
            user_id: None,
            tag: Some(tag.into()),
        }
    }

    /// Create a key scoped to both a user and a tag.
    #[must_use]
    pub fn user_tag(user_id: impl Into<String>, tag: impl Into<String>) -> Self {
        Self {
            user_id: Some(user_id.into()),
            tag: Some(tag.into()),
        }
    }

    /// Create the global (catch-all) key.
    #[must_use]
    pub fn global() -> Self {
        Self {
            user_id: None,
            tag: None,
        }
    }

    /// Canonical string representation used as the HashMap key.
    #[must_use]
    pub fn as_key_string(&self) -> String {
        match (&self.user_id, &self.tag) {
            (Some(u), Some(t)) => format!("user:{u};tag:{t}"),
            (Some(u), None) => format!("user:{u}"),
            (None, Some(t)) => format!("tag:{t}"),
            (None, None) => "global".to_string(),
        }
    }
}

impl fmt::Display for RateLimitKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_key_string())
    }
}

// ---------------------------------------------------------------------------
// RateLimit
// ---------------------------------------------------------------------------

/// A rate limit specification: at most `max_per_window` requests within
/// `window_secs` seconds.
#[derive(Debug, Clone)]
pub struct RateLimit {
    /// Maximum allowed requests inside the window.
    pub max_per_window: u32,
    /// Length of the sliding window in seconds.
    pub window_secs: u64,
}

impl RateLimit {
    /// Create a new rate limit.
    #[must_use]
    pub fn new(max_per_window: u32, window_secs: u64) -> Self {
        Self {
            max_per_window,
            window_secs,
        }
    }
}

impl Default for RateLimit {
    /// Default: 100 requests per 60-second window.
    fn default() -> Self {
        Self::new(100, 60)
    }
}

// ---------------------------------------------------------------------------
// RateLimitError
// ---------------------------------------------------------------------------

/// Errors produced by the rate limiter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RateLimitError {
    /// The caller exceeded the configured rate limit.
    LimitExceeded {
        /// String representation of the rate-limit key that was exceeded.
        key: String,
        /// How many seconds to wait before retrying.
        retry_after_secs: u64,
    },
    /// The configuration is invalid (e.g. `window_secs == 0`).
    ConfigError(String),
}

impl fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded { key, retry_after_secs } => {
                write!(f, "rate limit exceeded for '{key}'; retry after {retry_after_secs}s")
            }
            Self::ConfigError(msg) => write!(f, "rate limit config error: {msg}"),
        }
    }
}

// ---------------------------------------------------------------------------
// RateLimiterConfig
// ---------------------------------------------------------------------------

/// Composite configuration governing three independent limit layers.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Limit applied to each unique `user_id`.
    pub per_user_limit: RateLimit,
    /// Limit applied to each unique `tag`.
    pub per_tag_limit: RateLimit,
    /// Limit shared across all keys (global).
    pub global_limit: RateLimit,
}

impl RateLimiterConfig {
    /// Create a config from three explicit limits.
    #[must_use]
    pub fn new(per_user_limit: RateLimit, per_tag_limit: RateLimit, global_limit: RateLimit) -> Self {
        Self {
            per_user_limit,
            per_tag_limit,
            global_limit,
        }
    }
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            per_user_limit: RateLimit::new(50, 60),
            per_tag_limit: RateLimit::new(200, 60),
            global_limit: RateLimit::new(1000, 60),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal sliding-window state
// ---------------------------------------------------------------------------

/// Stores the timestamps (Unix seconds) of recent requests for one key.
#[derive(Debug, Default)]
struct WindowState {
    /// Sorted timestamp log; old entries are pruned lazily on each access.
    timestamps: Vec<u64>,
}

impl WindowState {
    /// Prune timestamps outside the current window and return the count
    /// of remaining (in-window) requests.
    fn prune_and_count(&mut self, now: u64, window_secs: u64) -> u32 {
        let cutoff = now.saturating_sub(window_secs);
        self.timestamps.retain(|&t| t > cutoff);
        self.timestamps.len() as u32
    }

    /// Record a new request at `now`.
    fn push(&mut self, now: u64) {
        self.timestamps.push(now);
    }

    /// Earliest timestamp still inside the window (used for retry-after).
    fn oldest_in_window(&self, now: u64, window_secs: u64) -> Option<u64> {
        let cutoff = now.saturating_sub(window_secs);
        self.timestamps.iter().find(|&&t| t > cutoff).copied()
    }
}

// ---------------------------------------------------------------------------
// RateLimiter
// ---------------------------------------------------------------------------

/// Per-user / per-tag / global sliding-window rate limiter.
///
/// On each [`check_and_consume`](RateLimiter::check_and_consume) call the
/// limiter evaluates up to three windows (user, tag, global) and rejects the
/// request if any one of them is exhausted.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimiterConfig,
    /// State keyed by `RateLimitKey::as_key_string()`.
    windows: HashMap<String, WindowState>,
}

impl RateLimiter {
    /// Create a new limiter from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`RateLimitError::ConfigError`] if any window has `window_secs == 0`.
    pub fn new(config: RateLimiterConfig) -> Result<Self, RateLimitError> {
        if config.per_user_limit.window_secs == 0
            || config.per_tag_limit.window_secs == 0
            || config.global_limit.window_secs == 0
        {
            return Err(RateLimitError::ConfigError(
                "window_secs must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            config,
            windows: HashMap::new(),
        })
    }

    /// Create a limiter with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        // Safety: default config has non-zero windows.
        Self {
            config: RateLimiterConfig::default(),
            windows: HashMap::new(),
        }
    }

    // --- internal helpers --------------------------------------------------

    fn check_limit(
        windows: &mut HashMap<String, WindowState>,
        raw_key: &str,
        limit: &RateLimit,
        now: u64,
    ) -> Result<(), RateLimitError> {
        let state = windows.entry(raw_key.to_string()).or_default();
        let count = state.prune_and_count(now, limit.window_secs);
        if count >= limit.max_per_window {
            // Calculate retry-after: time until the oldest in-window request expires.
            let oldest = state.oldest_in_window(now, limit.window_secs).unwrap_or(now);
            let retry_after_secs = (oldest + limit.window_secs).saturating_sub(now);
            return Err(RateLimitError::LimitExceeded {
                key: raw_key.to_string(),
                retry_after_secs,
            });
        }
        Ok(())
    }

    fn record(windows: &mut HashMap<String, WindowState>, raw_key: &str, now: u64) {
        windows.entry(raw_key.to_string()).or_default().push(now);
    }

    // --- public API --------------------------------------------------------

    /// Attempt to consume one request slot for `key`.
    ///
    /// Checks are applied in order: per-user → per-tag → global.
    /// The request is recorded only when all checks pass.
    ///
    /// # Errors
    ///
    /// Returns [`RateLimitError::LimitExceeded`] when any limit is exhausted.
    pub fn check_and_consume(
        &mut self,
        key: &RateLimitKey,
    ) -> Result<(), RateLimitError> {
        self.check_and_consume_at(key, current_unix_secs())
    }

    /// Same as [`check_and_consume`](Self::check_and_consume) but uses an
    /// explicit timestamp — useful for deterministic tests.
    ///
    /// # Errors
    ///
    /// Returns [`RateLimitError::LimitExceeded`] when any limit is exhausted.
    pub fn check_and_consume_at(
        &mut self,
        key: &RateLimitKey,
        now: u64,
    ) -> Result<(), RateLimitError> {
        // --- Check phase (no mutations to timestamp log) -------------------
        if let Some(user_id) = &key.user_id {
            let raw = format!("user:{user_id}");
            Self::check_limit(&mut self.windows, &raw, &self.config.per_user_limit.clone(), now)?;
        }
        if let Some(tag) = &key.tag {
            let raw = format!("tag:{tag}");
            Self::check_limit(&mut self.windows, &raw, &self.config.per_tag_limit.clone(), now)?;
        }
        // Always check global limit.
        Self::check_limit(&mut self.windows, "global", &self.config.global_limit.clone(), now)?;

        // --- Record phase (all checks passed) ------------------------------
        if let Some(user_id) = &key.user_id {
            Self::record(&mut self.windows, &format!("user:{user_id}"), now);
        }
        if let Some(tag) = &key.tag {
            Self::record(&mut self.windows, &format!("tag:{tag}"), now);
        }
        Self::record(&mut self.windows, "global", now);

        Ok(())
    }

    /// Remaining allowed requests for this key within the current window.
    ///
    /// Returns the minimum of remaining slots across user, tag, and global
    /// windows.
    #[must_use]
    pub fn remaining(&self, key: &RateLimitKey) -> u32 {
        self.remaining_at(key, current_unix_secs())
    }

    /// Same as [`remaining`](Self::remaining) but at explicit `now`.
    #[must_use]
    pub fn remaining_at(&self, key: &RateLimitKey, now: u64) -> u32 {
        let mut min_remaining = u32::MAX;

        if let Some(user_id) = &key.user_id {
            let raw = format!("user:{user_id}");
            let used = count_window(&self.windows, &raw, now, self.config.per_user_limit.window_secs);
            let rem = self.config.per_user_limit.max_per_window.saturating_sub(used);
            min_remaining = min_remaining.min(rem);
        }
        if let Some(tag) = &key.tag {
            let raw = format!("tag:{tag}");
            let used = count_window(&self.windows, &raw, now, self.config.per_tag_limit.window_secs);
            let rem = self.config.per_tag_limit.max_per_window.saturating_sub(used);
            min_remaining = min_remaining.min(rem);
        }
        // Global
        let used = count_window(&self.windows, "global", now, self.config.global_limit.window_secs);
        let global_rem = self.config.global_limit.max_per_window.saturating_sub(used);
        min_remaining = min_remaining.min(global_rem);

        min_remaining
    }

    /// How many seconds until the oldest in-window request for `key` expires,
    /// restoring at least one request slot.  Returns `0` when the key has no
    /// in-window requests.
    #[must_use]
    pub fn reset_after_secs(&self, key: &RateLimitKey) -> u64 {
        self.reset_after_secs_at(key, current_unix_secs())
    }

    /// Same as [`reset_after_secs`](Self::reset_after_secs) but at explicit `now`.
    #[must_use]
    pub fn reset_after_secs_at(&self, key: &RateLimitKey, now: u64) -> u64 {
        // Use the user window as the reference when present, else tag, else global.
        let (raw, window_secs) = if let Some(user_id) = &key.user_id {
            (format!("user:{user_id}"), self.config.per_user_limit.window_secs)
        } else if let Some(tag) = &key.tag {
            (format!("tag:{tag}"), self.config.per_tag_limit.window_secs)
        } else {
            ("global".to_string(), self.config.global_limit.window_secs)
        };

        if let Some(state) = self.windows.get(&raw) {
            if let Some(oldest) = state.oldest_in_window(now, window_secs) {
                return (oldest + window_secs).saturating_sub(now);
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn count_window(
    windows: &HashMap<String, WindowState>,
    raw_key: &str,
    now: u64,
    window_secs: u64,
) -> u32 {
    windows
        .get(raw_key)
        .map(|s| {
            let cutoff = now.saturating_sub(window_secs);
            s.timestamps.iter().filter(|&&t| t > cutoff).count() as u32
        })
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_limiter(max_per_window: u32, window_secs: u64) -> RateLimiter {
        let limit = RateLimit::new(max_per_window, window_secs);
        let config = RateLimiterConfig::new(limit.clone(), limit.clone(), limit);
        RateLimiter::new(config).expect("valid config")
    }

    #[test]
    fn test_within_limit_passes() {
        let mut limiter = make_limiter(5, 60);
        let key = RateLimitKey::user("alice");
        for _ in 0..5 {
            assert!(limiter.check_and_consume_at(&key, 1000).is_ok());
        }
    }

    #[test]
    fn test_exceeds_limit_returns_error() {
        let mut limiter = make_limiter(3, 60);
        let key = RateLimitKey::user("bob");
        for _ in 0..3 {
            limiter.check_and_consume_at(&key, 1000).expect("should pass");
        }
        let err = limiter.check_and_consume_at(&key, 1000).expect_err("should fail");
        assert!(matches!(err, RateLimitError::LimitExceeded { .. }));
    }

    #[test]
    fn test_window_resets_after_expiry() {
        let mut limiter = make_limiter(2, 10);
        let key = RateLimitKey::user("carol");
        limiter.check_and_consume_at(&key, 100).expect("ok");
        limiter.check_and_consume_at(&key, 100).expect("ok");
        // Still within window — should fail.
        assert!(limiter.check_and_consume_at(&key, 105).is_err());
        // After the window (100 + 10 = 110) timestamps expire, slot is free.
        assert!(limiter.check_and_consume_at(&key, 111).is_ok());
    }

    #[test]
    fn test_different_users_are_independent() {
        // Per-user limit of 1, but global limit is high enough not to interfere.
        let per_user = RateLimit::new(1, 60);
        let per_tag = RateLimit::new(100, 60);
        let global = RateLimit::new(100, 60);
        let config = RateLimiterConfig::new(per_user, per_tag, global);
        let mut limiter = RateLimiter::new(config).expect("valid config");

        let key_a = RateLimitKey::user("user-a");
        let key_b = RateLimitKey::user("user-b");
        limiter.check_and_consume_at(&key_a, 1000).expect("ok for a");
        limiter.check_and_consume_at(&key_b, 1000).expect("ok for b");
        // Each user has consumed their 1 allowed request.
        assert!(limiter.check_and_consume_at(&key_a, 1001).is_err());
        assert!(limiter.check_and_consume_at(&key_b, 1001).is_err());
    }

    #[test]
    fn test_different_tags_are_independent() {
        let mut limiter = make_limiter(2, 60);
        let key_a = RateLimitKey::tag("transcode");
        let key_b = RateLimitKey::tag("thumbnail");
        limiter.check_and_consume_at(&key_a, 1000).expect("ok");
        limiter.check_and_consume_at(&key_b, 1000).expect("ok");
    }

    #[test]
    fn test_remaining_decrements() {
        let mut limiter = make_limiter(5, 60);
        let key = RateLimitKey::user("dave");
        assert_eq!(limiter.remaining_at(&key, 1000), 5);
        limiter.check_and_consume_at(&key, 1000).expect("ok");
        assert_eq!(limiter.remaining_at(&key, 1000), 4);
    }

    #[test]
    fn test_reset_after_secs_nonzero_when_exhausted() {
        let mut limiter = make_limiter(1, 30);
        let key = RateLimitKey::user("eve");
        limiter.check_and_consume_at(&key, 1000).expect("ok");
        let reset = limiter.reset_after_secs_at(&key, 1005);
        // Oldest request was at t=1000, window is 30 s → expires at 1030 → 25 s from now.
        assert_eq!(reset, 25);
    }

    #[test]
    fn test_config_error_zero_window() {
        let bad = RateLimit::new(10, 0);
        let config = RateLimiterConfig::new(bad.clone(), bad.clone(), bad);
        let err = RateLimiter::new(config).expect_err("should fail");
        assert!(matches!(err, RateLimitError::ConfigError(_)));
    }

    #[test]
    fn test_global_limit_shared_across_keys() {
        // Global limit of 2; each request from any key consumes from it.
        let user_limit = RateLimit::new(100, 60);
        let tag_limit = RateLimit::new(100, 60);
        let global_limit = RateLimit::new(2, 60);
        let config = RateLimiterConfig::new(user_limit, tag_limit, global_limit);
        let mut limiter = RateLimiter::new(config).expect("ok");

        let key_a = RateLimitKey::user("f1");
        let key_b = RateLimitKey::user("f2");
        limiter.check_and_consume_at(&key_a, 1000).expect("first ok");
        limiter.check_and_consume_at(&key_b, 1000).expect("second ok");
        // Global exhausted regardless of which user.
        assert!(limiter.check_and_consume_at(&key_a, 1001).is_err());
    }

    #[test]
    fn test_rate_limit_key_display() {
        let k = RateLimitKey::user_tag("alice", "video");
        assert_eq!(k.as_key_string(), "user:alice;tag:video");
        let k2 = RateLimitKey::global();
        assert_eq!(k2.as_key_string(), "global");
    }
}
