//! Short-TTL cache for rights-check results.
//!
//! Wraps a [`RightsChecker`] with an in-memory cache keyed on
//! `(asset_id, action, territory, platform)`.  Cache entries expire after
//! a configurable number of seconds, preventing stale grants from being
//! served indefinitely while still amortising the cost of repeated identical
//! queries (e.g. multiple users streaming the same asset from the same
//! territory).
//!
//! # Design
//!
//! - **Key**: `CacheKey` — a 4-tuple of `(asset_id, action, territory, platform)`.
//!   The caller's `now` timestamp is **not** part of the key; instead, each
//!   entry records its own evaluation timestamp and a TTL.
//! - **Eviction**: lazy — stale entries are only removed when a cache hit would
//!   have been returned.  [`QueryCache::evict_expired`] may be called
//!   periodically to reclaim memory.
//! - **Clock injection**: the cache accepts an explicit `now` parameter so that
//!   tests are hermetic without requiring `std::time`.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_rights::query_cache::{QueryCache, CacheConfig};
//! use oximedia_rights::rights_check::{ActionKind, CheckRequest, CheckResult, RightsChecker, RightsGrant};
//!
//! let mut checker = RightsChecker::new();
//! checker.add_grant(
//!     RightsGrant::new("g1", "asset-A")
//!         .with_action(ActionKind::Stream),
//! );
//! let config = CacheConfig { ttl_seconds: 60 };
//! let mut cache = QueryCache::new(checker, config);
//!
//! let req = CheckRequest::new("asset-A", ActionKind::Stream, "US", "web", 1_000);
//! // First call — cache miss, evaluates checker
//! let result = cache.check(&req, 1_000);
//! assert!(result.is_allowed());
//! // Second call — cache hit
//! let result2 = cache.check(&req, 1_030);
//! assert!(result2.is_allowed());
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

use crate::rights_check::{ActionKind, CheckRequest, CheckResult, RightsChecker};

// ── CacheConfig ───────────────────────────────────────────────────────────────

/// Configuration for the [`QueryCache`].
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Number of seconds a cached result remains valid.
    /// Set to `0` to disable caching (every call goes to the checker).
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self { ttl_seconds: 30 }
    }
}

impl CacheConfig {
    /// Create a configuration with the given TTL.
    #[must_use]
    pub fn with_ttl(ttl_seconds: u64) -> Self {
        Self { ttl_seconds }
    }

    /// Whether caching is enabled (TTL > 0).
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.ttl_seconds > 0
    }
}

// ── CacheKey ──────────────────────────────────────────────────────────────────

/// Composite cache key derived from a [`CheckRequest`].
///
/// The `now` field of the request is intentionally excluded — the same logical
/// query should hit the cache even if the caller's clock has advanced slightly.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Asset identifier.
    pub asset_id: String,
    /// Requested action.
    pub action: ActionKind,
    /// Normalised territory code (upper-case).
    pub territory: String,
    /// Normalised platform identifier (lower-case).
    pub platform: String,
}

impl CacheKey {
    /// Derive a key from a check request.
    #[must_use]
    pub fn from_request(req: &CheckRequest) -> Self {
        Self {
            asset_id: req.asset_id.clone(),
            action: req.action,
            territory: req.territory.to_uppercase(),
            platform: req.platform.to_lowercase(),
        }
    }
}

// ── CacheEntry ────────────────────────────────────────────────────────────────

/// A single cached rights-check result.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached result.
    result: CheckResult,
    /// Unix seconds at which the result was evaluated.
    evaluated_at: u64,
    /// TTL in seconds (copied from config at insertion time).
    ttl_seconds: u64,
}

impl CacheEntry {
    /// Whether the entry is still valid at `now`.
    fn is_valid_at(&self, now: u64) -> bool {
        now < self.evaluated_at.saturating_add(self.ttl_seconds)
    }
}

// ── CacheStats ────────────────────────────────────────────────────────────────

/// Cumulative cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total requests served (hits + misses).
    pub total_requests: u64,
    /// Requests answered from the cache.
    pub hits: u64,
    /// Requests that required evaluating the checker.
    pub misses: u64,
    /// Entries that were found stale and evicted at access time.
    pub stale_evictions: u64,
    /// Entries removed by explicit [`QueryCache::evict_expired`] calls.
    pub periodic_evictions: u64,
}

impl CacheStats {
    /// Hit ratio in the range `[0.0, 1.0]`. Returns `0.0` when no requests
    /// have been served yet.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }
}

// ── QueryCache ────────────────────────────────────────────────────────────────

/// A caching wrapper around [`RightsChecker`] with a short TTL.
///
/// See the [module-level documentation](self) for design notes.
pub struct QueryCache {
    checker: RightsChecker,
    config: CacheConfig,
    store: HashMap<CacheKey, CacheEntry>,
    stats: CacheStats,
}

impl QueryCache {
    /// Create a new cache wrapping `checker` with the given `config`.
    #[must_use]
    pub fn new(checker: RightsChecker, config: CacheConfig) -> Self {
        Self {
            checker,
            config,
            store: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with the default configuration (30-second TTL).
    #[must_use]
    pub fn with_defaults(checker: RightsChecker) -> Self {
        Self::new(checker, CacheConfig::default())
    }

    // ── Core check API ────────────────────────────────────────────────────────

    /// Evaluate or retrieve a rights-check result for `req` at `now`.
    ///
    /// If a non-stale entry exists in the cache it is returned directly.
    /// Otherwise the underlying [`RightsChecker`] is evaluated, the result is
    /// stored, and the stats are updated.
    ///
    /// # Parameters
    /// - `req` — the check request.
    /// - `now` — current Unix seconds (used for TTL evaluation and for passing
    ///   to the checker on a miss).
    #[must_use]
    pub fn check(&mut self, req: &CheckRequest, now: u64) -> CheckResult {
        self.stats.total_requests += 1;

        if !self.config.is_enabled() {
            self.stats.misses += 1;
            return self.checker.check(req);
        }

        let key = CacheKey::from_request(req);

        if let Some(entry) = self.store.get(&key) {
            if entry.is_valid_at(now) {
                self.stats.hits += 1;
                return entry.result.clone();
            }
            // Stale — fall through to re-evaluate and update.
            self.stats.stale_evictions += 1;
        }

        // Cache miss: evaluate the checker, then store.
        self.stats.misses += 1;
        let result = self.checker.check(req);
        self.store.insert(
            key,
            CacheEntry {
                result: result.clone(),
                evaluated_at: now,
                ttl_seconds: self.config.ttl_seconds,
            },
        );
        result
    }

    /// Convenience: check whether the action is allowed (see [`check`]).
    #[must_use]
    pub fn is_allowed(&mut self, req: &CheckRequest, now: u64) -> bool {
        self.check(req, now).is_allowed()
    }

    // ── Maintenance ───────────────────────────────────────────────────────────

    /// Remove all expired entries from the cache.
    ///
    /// Returns the number of entries removed.  Call this periodically (e.g.
    /// every few minutes) to prevent unbounded memory growth.
    pub fn evict_expired(&mut self, now: u64) -> usize {
        let before = self.store.len();
        self.store.retain(|_, entry| entry.is_valid_at(now));
        let removed = before - self.store.len();
        self.stats.periodic_evictions += removed as u64;
        removed
    }

    /// Immediately remove a specific asset's cache entries.
    ///
    /// Use this when a grant is revoked or modified so that the next check
    /// reflects the updated state without waiting for TTL expiry.
    pub fn invalidate_asset(&mut self, asset_id: &str) -> usize {
        let before = self.store.len();
        self.store.retain(|key, _| key.asset_id != asset_id);
        before - self.store.len()
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    /// Number of entries currently in the cache (including stale ones not yet
    /// evicted).
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.store.len()
    }

    /// A snapshot of the cache statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Access the underlying [`RightsChecker`].
    #[must_use]
    pub fn checker(&self) -> &RightsChecker {
        &self.checker
    }

    /// Mutable access to the underlying [`RightsChecker`] (e.g. to add new
    /// grants). The cache is **not** automatically cleared; call
    /// [`invalidate_asset`](Self::invalidate_asset) or [`clear`](Self::clear)
    /// after modifying the checker if consistency is required.
    pub fn checker_mut(&mut self) -> &mut RightsChecker {
        &mut self.checker
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rights_check::{ActionKind, CheckRequest, RightsGrant};

    fn make_checker() -> RightsChecker {
        let mut checker = RightsChecker::new();
        checker.add_grant(
            RightsGrant::new("g1", "asset-A")
                .with_action(ActionKind::Stream)
                .with_action(ActionKind::Download),
        );
        checker.add_grant(
            RightsGrant::new("g2", "asset-B")
                .with_action(ActionKind::Broadcast)
                .with_territory("US")
                .with_window(1_000, 9_000),
        );
        checker
    }

    fn stream_req(asset: &str, now: u64) -> CheckRequest {
        CheckRequest::new(asset, ActionKind::Stream, "US", "web", now)
    }

    fn broadcast_us_req(now: u64) -> CheckRequest {
        CheckRequest::new("asset-B", ActionKind::Broadcast, "US", "tv", now)
    }

    // ── CacheConfig ──

    #[test]
    fn test_config_default_ttl() {
        let cfg = CacheConfig::default();
        assert_eq!(cfg.ttl_seconds, 30);
        assert!(cfg.is_enabled());
    }

    #[test]
    fn test_config_zero_ttl_disabled() {
        let cfg = CacheConfig::with_ttl(0);
        assert!(!cfg.is_enabled());
    }

    #[test]
    fn test_config_with_ttl() {
        let cfg = CacheConfig::with_ttl(120);
        assert_eq!(cfg.ttl_seconds, 120);
    }

    // ── CacheKey ──

    #[test]
    fn test_cache_key_normalisation() {
        let req = CheckRequest::new("asset-X", ActionKind::Stream, "gb", "Web", 0);
        let key = CacheKey::from_request(&req);
        assert_eq!(key.territory, "GB");
        assert_eq!(key.platform, "web");
    }

    #[test]
    fn test_cache_key_equality() {
        let r1 = CheckRequest::new("A", ActionKind::Download, "US", "web", 100);
        let r2 = CheckRequest::new("A", ActionKind::Download, "us", "Web", 200);
        assert_eq!(CacheKey::from_request(&r1), CacheKey::from_request(&r2));
    }

    // ── QueryCache — basic hit/miss ──

    #[test]
    fn test_cache_miss_evaluates_checker() {
        let mut cache = QueryCache::with_defaults(make_checker());
        let req = stream_req("asset-A", 500);
        let result = cache.check(&req, 500);
        assert!(result.is_allowed());
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);
    }

    #[test]
    fn test_cache_hit_on_second_call() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(60));
        let req = stream_req("asset-A", 500);
        cache.check(&req, 500);
        cache.check(&req, 510); // within TTL → hit
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_stale_entry_re_evaluated() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(10));
        let req = stream_req("asset-A", 100);
        cache.check(&req, 100); // miss, stored at t=100
        cache.check(&req, 111); // t=111 > 100+10 → stale, re-evaluated
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().stale_evictions, 1);
    }

    #[test]
    fn test_disabled_cache_always_evaluates() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(0));
        let req = stream_req("asset-A", 1);
        cache.check(&req, 1);
        cache.check(&req, 2);
        assert_eq!(cache.stats().misses, 2);
        assert_eq!(cache.stats().hits, 0);
    }

    // ── QueryCache — denied result cached ──

    #[test]
    fn test_denied_result_is_cached() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(60));
        let req = CheckRequest::new("no-such-asset", ActionKind::Stream, "US", "web", 50);
        let r1 = cache.check(&req, 50);
        assert!(r1.is_denied());
        let r2 = cache.check(&req, 55);
        assert!(r2.is_denied());
        assert_eq!(cache.stats().hits, 1);
    }

    // ── QueryCache — eviction ──

    #[test]
    fn test_evict_expired_removes_stale() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(5));
        cache.check(&stream_req("asset-A", 0), 0);
        assert_eq!(cache.entry_count(), 1);
        // At t=6 the entry is stale
        let removed = cache.evict_expired(6);
        assert_eq!(removed, 1);
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.stats().periodic_evictions, 1);
    }

    #[test]
    fn test_invalidate_asset() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(60));
        cache.check(&stream_req("asset-A", 0), 0);
        cache.check(&broadcast_us_req(2_000), 2_000);
        assert_eq!(cache.entry_count(), 2);
        let removed = cache.invalidate_asset("asset-A");
        assert_eq!(removed, 1);
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn test_clear() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(60));
        cache.check(&stream_req("asset-A", 0), 0);
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
    }

    // ── CacheStats ──

    #[test]
    fn test_hit_ratio_zero_requests() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_ratio(), 0.0);
    }

    #[test]
    fn test_hit_ratio_all_hits() {
        let mut cache = QueryCache::new(make_checker(), CacheConfig::with_ttl(120));
        let req = stream_req("asset-A", 0);
        cache.check(&req, 0);
        cache.check(&req, 1);
        cache.check(&req, 2);
        let ratio = cache.stats().hit_ratio();
        // 1 miss + 2 hits = 2/3 ≈ 0.666
        assert!((ratio - 2.0 / 3.0).abs() < 1e-9);
    }

    // ── is_allowed convenience ──

    #[test]
    fn test_is_allowed_convenience() {
        let mut cache = QueryCache::with_defaults(make_checker());
        assert!(cache.is_allowed(&stream_req("asset-A", 0), 0));
        assert!(!cache.is_allowed(&stream_req("asset-Z", 0), 0));
    }
}
