//! CDN and edge-cache optimization for media asset delivery.
//!
//! This module helps select appropriate caching policies, build normalized cache keys,
//! and manage a simple in-memory cache inventory with eviction support.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

/// Caching policy applied to a media asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Do not store the response in any cache.
    NoStore,
    /// Very short-lived cache (e.g. live manifest, 5 s).
    ShortLived,
    /// Standard media segment cache (e.g. 30 s).
    Standard,
    /// Long-lived cache for stable assets (e.g. 24 h).
    LongLived,
    /// Immutable versioned asset — cache forever (1 year).
    Immutable,
}

impl CachePolicy {
    /// Returns the maximum cache age in seconds.
    #[must_use]
    pub fn max_age_secs(&self) -> u32 {
        match self {
            Self::NoStore => 0,
            Self::ShortLived => 5,
            Self::Standard => 30,
            Self::LongLived => 86_400,
            Self::Immutable => 31_536_000,
        }
    }

    /// Returns `true` if the client should revalidate with the origin before using
    /// a cached response (stale-while-revalidate or must-revalidate semantics).
    #[must_use]
    pub fn should_revalidate(&self) -> bool {
        matches!(self, Self::ShortLived | Self::Standard)
    }
}

/// A normalized cache key composed of a URL path and varying request headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheKey {
    /// The URL path component of the cache key.
    pub url_path: String,
    /// Headers that vary the cached response (e.g. `Accept-Encoding`).
    pub vary_headers: Vec<String>,
}

impl CacheKey {
    /// Returns a normalized string representation: lowercased path + sorted, lowercased headers.
    #[must_use]
    pub fn normalize(&self) -> String {
        let path = self.url_path.to_lowercase();
        let mut headers: Vec<String> = self.vary_headers.iter().map(|h| h.to_lowercase()).collect();
        headers.sort();
        if headers.is_empty() {
            path
        } else {
            format!("{}?vary={}", path, headers.join(","))
        }
    }

    /// Returns `true` if there are any vary headers.
    #[must_use]
    pub fn has_vary(&self) -> bool {
        !self.vary_headers.is_empty()
    }
}

/// A single cached entry with its associated policy and access statistics.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Cache key for this entry.
    pub key: CacheKey,
    /// Size of the cached content in bytes.
    pub size_bytes: u32,
    /// Caching policy applied to this entry.
    pub policy: CachePolicy,
    /// Unix epoch timestamp when this entry was cached.
    pub cached_at: u64,
    /// Number of cache hits served from this entry.
    pub hit_count: u32,
}

impl CacheEntry {
    /// Returns `true` if the entry has expired according to its policy's `max_age_secs`.
    #[must_use]
    pub fn is_expired(&self, now: u64) -> bool {
        let max_age = u64::from(self.policy.max_age_secs());
        if max_age == 0 {
            return true;
        }
        now.saturating_sub(self.cached_at) >= max_age
    }

    /// Returns how old the entry is, in seconds, at the given epoch timestamp.
    #[must_use]
    pub fn age_secs(&self, now: u64) -> u64 {
        now.saturating_sub(self.cached_at)
    }
}

/// In-memory cache inventory with hit-rate tracking and LRU-style eviction.
#[derive(Debug, Default)]
pub struct CacheOptimizer {
    entries: Vec<CacheEntry>,
}

impl CacheOptimizer {
    /// Creates a new, empty `CacheOptimizer`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Looks up a non-expired entry by URL path.
    #[must_use]
    pub fn get(&self, path: &str, now: u64) -> Option<&CacheEntry> {
        let normalized = path.to_lowercase();
        self.entries
            .iter()
            .find(|e| e.key.url_path.to_lowercase() == normalized && !e.is_expired(now))
    }

    /// Inserts or replaces a cache entry. If an entry with the same normalized key
    /// already exists, it is replaced.
    pub fn put(&mut self, entry: CacheEntry) {
        let key_norm = entry.key.normalize();
        if let Some(pos) = self
            .entries
            .iter()
            .position(|e| e.key.normalize() == key_norm)
        {
            self.entries[pos] = entry;
        } else {
            self.entries.push(entry);
        }
    }

    /// Removes all expired entries and returns the count of removed entries.
    pub fn evict_expired(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|e| !e.is_expired(now));
        before - self.entries.len()
    }

    /// Returns the hit rate as a fraction (hits / total entries).
    /// Returns `0.0` if there are no entries.
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let total_hits: u32 = self.entries.iter().map(|e| e.hit_count).sum();
        let total_requests: u32 = self.entries.iter().map(|e| e.hit_count + 1).sum();
        if total_requests == 0 {
            0.0
        } else {
            total_hits as f32 / total_requests as f32
        }
    }

    /// Returns the total size of all cached entries in bytes.
    #[must_use]
    pub fn total_size_bytes(&self) -> u64 {
        self.entries.iter().map(|e| u64::from(e.size_bytes)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(path: &str, policy: CachePolicy, cached_at: u64, hits: u32) -> CacheEntry {
        CacheEntry {
            key: CacheKey {
                url_path: path.to_string(),
                vary_headers: vec![],
            },
            size_bytes: 1024,
            policy,
            cached_at,
            hit_count: hits,
        }
    }

    #[test]
    fn test_no_store_max_age() {
        assert_eq!(CachePolicy::NoStore.max_age_secs(), 0);
    }

    #[test]
    fn test_immutable_max_age() {
        assert_eq!(CachePolicy::Immutable.max_age_secs(), 31_536_000);
    }

    #[test]
    fn test_should_revalidate_short_lived() {
        assert!(CachePolicy::ShortLived.should_revalidate());
    }

    #[test]
    fn test_should_not_revalidate_immutable() {
        assert!(!CachePolicy::Immutable.should_revalidate());
    }

    #[test]
    fn test_cache_key_normalize_lowercase_sorted() {
        let key = CacheKey {
            url_path: "/Media/Segment.ts".to_string(),
            vary_headers: vec!["Accept-Encoding".to_string(), "Accept-Language".to_string()],
        };
        let norm = key.normalize();
        assert!(norm.starts_with("/media/segment.ts"));
        assert!(norm.contains("accept-encoding"));
        assert!(norm.contains("accept-language"));
    }

    #[test]
    fn test_cache_key_no_vary() {
        let key = CacheKey {
            url_path: "/segment.ts".to_string(),
            vary_headers: vec![],
        };
        assert_eq!(key.normalize(), "/segment.ts");
        assert!(!key.has_vary());
    }

    #[test]
    fn test_cache_key_has_vary() {
        let key = CacheKey {
            url_path: "/seg.ts".to_string(),
            vary_headers: vec!["Accept-Encoding".to_string()],
        };
        assert!(key.has_vary());
    }

    #[test]
    fn test_entry_is_expired() {
        // Standard policy has 30s max-age
        let entry = make_entry("/seg.ts", CachePolicy::Standard, 1000, 0);
        assert!(!entry.is_expired(1020)); // 20s old — fresh
        assert!(entry.is_expired(1030)); // 30s old — expired
    }

    #[test]
    fn test_no_store_always_expired() {
        let entry = make_entry("/live.m3u8", CachePolicy::NoStore, 1000, 0);
        assert!(entry.is_expired(1000));
    }

    #[test]
    fn test_entry_age_secs() {
        let entry = make_entry("/seg.ts", CachePolicy::LongLived, 1000, 0);
        assert_eq!(entry.age_secs(1050), 50);
    }

    #[test]
    fn test_cache_optimizer_put_and_get() {
        let mut cache = CacheOptimizer::new();
        cache.put(make_entry("/seg.ts", CachePolicy::Standard, 1000, 5));
        let entry = cache.get("/seg.ts", 1010);
        assert!(entry.is_some());
        assert_eq!(
            entry.expect("operation should succeed in test").hit_count,
            5
        );
    }

    #[test]
    fn test_cache_optimizer_get_expired_returns_none() {
        let mut cache = CacheOptimizer::new();
        cache.put(make_entry("/seg.ts", CachePolicy::Standard, 1000, 0));
        // 31s later — expired
        assert!(cache.get("/seg.ts", 1031).is_none());
    }

    #[test]
    fn test_cache_optimizer_evict_expired() {
        let mut cache = CacheOptimizer::new();
        cache.put(make_entry("/old.ts", CachePolicy::Standard, 1000, 0));
        cache.put(make_entry("/new.ts", CachePolicy::LongLived, 1000, 0));
        let evicted = cache.evict_expired(1031);
        assert_eq!(evicted, 1); // only old.ts expired
    }

    #[test]
    fn test_cache_optimizer_total_size() {
        let mut cache = CacheOptimizer::new();
        cache.put(make_entry("/a.ts", CachePolicy::Immutable, 0, 0));
        cache.put(make_entry("/b.ts", CachePolicy::Immutable, 0, 0));
        assert_eq!(cache.total_size_bytes(), 2048);
    }

    #[test]
    fn test_cache_optimizer_hit_rate() {
        let mut cache = CacheOptimizer::new();
        cache.put(make_entry("/a.ts", CachePolicy::Immutable, 0, 9));
        // 1 entry: 9 hits, 1 total_request count from denominator (hit_count + 1 = 10)
        let rate = cache.hit_rate();
        assert!((rate - 0.9).abs() < 1e-4, "expected 0.9, got {rate}");
    }
}
