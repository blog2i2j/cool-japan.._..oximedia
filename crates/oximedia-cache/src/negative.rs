//! Negative (miss) caching.
//!
//! [`NegativeCache`] records keys for which a lookup has previously returned
//! no result (a *miss*).  Subsequent lookups can check `is_known_miss` to
//! avoid hitting the origin for keys that are known to be absent.
//!
//! Each miss entry is stamped with the time it was recorded and expires after
//! a configurable TTL (in milliseconds).  Expired entries are considered
//! "unknown" again.
//!
//! # Example
//!
//! ```
//! use oximedia_cache::negative::NegativeCache;
//!
//! let mut nc = NegativeCache::new(5_000); // 5-second TTL
//! nc.insert_miss("missing-asset", 1_000_000);
//!
//! assert!(nc.is_known_miss("missing-asset", 1_002_000)); // within TTL
//! assert!(!nc.is_known_miss("missing-asset", 1_006_000)); // expired
//! ```

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// NegativeCache
// ---------------------------------------------------------------------------

/// A TTL-keyed negative result cache.
#[derive(Debug, Clone)]
pub struct NegativeCache {
    /// How long (in milliseconds) a miss entry is considered valid.
    ttl_ms: u64,
    /// Map from key to the timestamp (ms) at which the miss was recorded.
    entries: HashMap<String, u64>,
}

impl NegativeCache {
    /// Create a new `NegativeCache` with the given TTL in milliseconds.
    #[must_use]
    pub fn new(ttl_ms: u64) -> Self {
        Self {
            ttl_ms,
            entries: HashMap::new(),
        }
    }

    /// Record `key` as a known miss at timestamp `now_ms`.
    ///
    /// If an entry for `key` already exists it is overwritten with the new
    /// timestamp (refreshing the TTL).
    pub fn insert_miss(&mut self, key: &str, now_ms: u64) {
        self.entries.insert(key.to_string(), now_ms);
    }

    /// Returns `true` when `key` is a known miss that has not yet expired.
    ///
    /// A miss entry at `recorded_ms` expires when `now_ms >= recorded_ms + ttl_ms`.
    #[must_use]
    pub fn is_known_miss(&self, key: &str, now_ms: u64) -> bool {
        match self.entries.get(key) {
            Some(&recorded_ms) => {
                let expiry = recorded_ms.saturating_add(self.ttl_ms);
                now_ms < expiry
            }
            None => false,
        }
    }

    /// Remove the entry for `key` (e.g. after it has been successfully
    /// populated in the origin cache).
    pub fn remove(&mut self, key: &str) {
        self.entries.remove(key);
    }

    /// Prune all entries that have expired by `now_ms`.
    pub fn evict_expired(&mut self, now_ms: u64) {
        self.entries.retain(|_, &mut recorded_ms| {
            let expiry = recorded_ms.saturating_add(self.ttl_ms);
            now_ms < expiry
        });
    }

    /// Number of currently tracked negative entries (including expired ones
    /// that have not been pruned yet).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no entries are tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The configured TTL in milliseconds.
    #[must_use]
    pub fn ttl_ms(&self) -> u64 {
        self.ttl_ms
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── new ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_new_is_empty() {
        let nc = NegativeCache::new(5_000);
        assert!(nc.is_empty());
        assert_eq!(nc.len(), 0);
    }

    #[test]
    fn test_ttl_getter() {
        let nc = NegativeCache::new(10_000);
        assert_eq!(nc.ttl_ms(), 10_000);
    }

    // ── insert_miss / is_known_miss ──────────────────────────────────────────

    #[test]
    fn test_is_known_miss_before_expiry() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("key1", 1_000_000);
        assert!(nc.is_known_miss("key1", 1_002_000));
    }

    #[test]
    fn test_is_known_miss_at_expiry_boundary() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("k", 1_000_000);
        // Exactly at expiry (recorded + ttl = 1_005_000) → expired
        assert!(!nc.is_known_miss("k", 1_005_000));
    }

    #[test]
    fn test_is_known_miss_after_expiry() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("k", 1_000_000);
        assert!(!nc.is_known_miss("k", 1_100_000));
    }

    #[test]
    fn test_is_known_miss_unknown_key_returns_false() {
        let nc = NegativeCache::new(5_000);
        assert!(!nc.is_known_miss("absent", 0));
    }

    #[test]
    fn test_insert_miss_overwrites_existing() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("k", 1_000_000);
        // Override with a later timestamp
        nc.insert_miss("k", 2_000_000);
        assert!(nc.is_known_miss("k", 2_001_000));
        // Old entry would have expired, but new one is valid
        assert!(!nc.is_known_miss("k", 2_006_000));
    }

    // ── remove ───────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_makes_key_unknown() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("k", 0);
        nc.remove("k");
        assert!(!nc.is_known_miss("k", 1_000));
    }

    #[test]
    fn test_remove_nonexistent_is_noop() {
        let mut nc = NegativeCache::new(5_000);
        nc.remove("ghost"); // should not panic
        assert!(nc.is_empty());
    }

    // ── evict_expired ────────────────────────────────────────────────────────

    #[test]
    fn test_evict_expired_removes_old_entries() {
        let mut nc = NegativeCache::new(1_000);
        nc.insert_miss("old", 0);          // expires at 1_000
        nc.insert_miss("young", 5_000);    // expires at 6_000
        nc.evict_expired(2_000);           // prune at t=2_000
        assert_eq!(nc.len(), 1, "Only 'young' should remain");
        assert!(!nc.is_known_miss("old", 500));
    }

    #[test]
    fn test_evict_expired_keeps_valid_entries() {
        let mut nc = NegativeCache::new(10_000);
        nc.insert_miss("a", 1_000);
        nc.evict_expired(5_000); // 5_000 < 1_000 + 10_000 = 11_000 → still valid
        assert_eq!(nc.len(), 1);
    }

    // ── clear ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_removes_all_entries() {
        let mut nc = NegativeCache::new(5_000);
        nc.insert_miss("a", 1);
        nc.insert_miss("b", 2);
        nc.clear();
        assert!(nc.is_empty());
    }

    // ── zero ttl ─────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_ttl_expires_immediately() {
        let mut nc = NegativeCache::new(0);
        nc.insert_miss("k", 1_000);
        // Any now_ms >= 1_000 → expired
        assert!(!nc.is_known_miss("k", 1_000));
    }
}
