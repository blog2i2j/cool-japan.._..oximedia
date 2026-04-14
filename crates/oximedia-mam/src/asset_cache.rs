//! In-memory caching layer for frequently accessed assets and search results.
//!
//! Provides a TTL-aware, capacity-bounded LRU cache that sits in front of the
//! database and search index to reduce round-trip latency for hot assets and
//! popular queries.  The implementation is intentionally pure-Rust with no
//! external service dependency so the MAM system can operate without Redis
//! while still exhibiting Redis-like semantics when those are needed.
//!
//! # Components
//!
//! - [`CacheEntry`] — a generic value wrapper with expiry metadata.
//! - [`CachePolicy`] — TTL and capacity settings for each cache tier.
//! - [`AssetCacheKey`] — typed key variants (asset by ID, search results, etc.).
//! - [`AssetCache`] — the main LRU cache structure with TTL eviction.
//! - [`CacheStats`] — runtime hit/miss/eviction counters.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Cache policy
// ---------------------------------------------------------------------------

/// Configuration controlling TTL and capacity for the cache.
#[derive(Debug, Clone)]
pub struct CachePolicy {
    /// Time-to-live for each entry before it is considered stale.
    pub ttl: Duration,
    /// Maximum number of entries the cache will hold before evicting LRU.
    pub max_entries: usize,
    /// Whether to reset the TTL on every cache hit (sliding window).
    pub sliding_ttl: bool,
}

impl Default for CachePolicy {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(120),
            max_entries: 1_000,
            sliding_ttl: false,
        }
    }
}

impl CachePolicy {
    /// Create a policy with the given TTL and capacity.
    #[must_use]
    pub fn new(ttl: Duration, max_entries: usize) -> Self {
        Self { ttl, max_entries, ..Self::default() }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A cached value paired with expiry metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value.
    pub value: V,
    /// Absolute expiry instant.
    pub expires_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Total number of times this entry has been accessed.
    pub access_count: u64,
}

impl<V: Clone> CacheEntry<V> {
    /// Create a new cache entry that expires `ttl` from now.
    #[must_use]
    pub fn new(value: V, ttl: Duration) -> Self {
        let now = Instant::now();
        Self {
            value,
            expires_at: now + ttl,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Returns `true` if the entry is still within its TTL.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        Instant::now() < self.expires_at
    }

    /// Record an access, optionally extending the TTL by `extension`.
    pub fn touch(&mut self, extension: Option<Duration>) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        if let Some(ext) = extension {
            self.expires_at = Instant::now() + ext;
        }
    }
}

// ---------------------------------------------------------------------------
// Typed cache keys
// ---------------------------------------------------------------------------

/// Typed keys for the asset cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AssetCacheKey {
    /// A single asset document keyed by its string ID.
    AssetById(String),
    /// Full search result set keyed by a normalised query string.
    SearchResult(String),
    /// A collection (list of asset IDs) keyed by collection ID.
    Collection(String),
    /// A user's permission set keyed by user ID.
    UserPermissions(String),
    /// An arbitrary application-level key for extension.
    Custom(String),
}

impl AssetCacheKey {
    /// Return a string representation of the key (useful for logging).
    #[must_use]
    pub fn as_str_repr(&self) -> String {
        match self {
            Self::AssetById(id) => format!("asset:{id}"),
            Self::SearchResult(q) => format!("search:{q}"),
            Self::Collection(id) => format!("collection:{id}"),
            Self::UserPermissions(uid) => format!("perms:{uid}"),
            Self::Custom(k) => format!("custom:{k}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Runtime statistics for an [`AssetCache`] instance.
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total entries evicted due to LRU or TTL expiry.
    pub evictions: u64,
    /// Total entries inserted.
    pub inserts: u64,
    /// Current number of live entries.
    pub live_entries: usize,
}

impl CacheStats {
    /// Hit ratio in range `[0.0, 1.0]`.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Miss ratio in range `[0.0, 1.0]`.
    #[must_use]
    pub fn miss_ratio(&self) -> f64 {
        1.0 - self.hit_ratio()
    }
}

// ---------------------------------------------------------------------------
// LRU ordering tracker
// ---------------------------------------------------------------------------

/// Maintains insertion-order for LRU eviction.
///
/// Keys are stored in a [`VecDeque`]; on every access the key is moved to
/// the back (most-recently-used position).  The front is the
/// least-recently-used candidate for eviction.
#[derive(Debug, Default)]
struct LruOrder {
    order: VecDeque<AssetCacheKey>,
}

impl LruOrder {
    fn touch(&mut self, key: &AssetCacheKey) {
        // Remove existing occurrence, then push to back.
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }

    fn evict_lru(&mut self) -> Option<AssetCacheKey> {
        self.order.pop_front()
    }

    fn remove(&mut self, key: &AssetCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.order.len()
    }
}

// ---------------------------------------------------------------------------
// Main cache
// ---------------------------------------------------------------------------

/// TTL-aware, capacity-bounded LRU cache for asset data and search results.
///
/// Values are stored as `serde_json::Value` so that heterogeneous data
/// (asset documents, search result lists, permission sets) can share one
/// cache instance without generics propagating throughout the codebase.
pub struct AssetCache {
    policy: CachePolicy,
    store: HashMap<AssetCacheKey, CacheEntry<serde_json::Value>>,
    lru: LruOrder,
    stats: CacheStats,
}

impl AssetCache {
    /// Create a new cache with the provided policy.
    #[must_use]
    pub fn new(policy: CachePolicy) -> Self {
        Self {
            policy,
            store: HashMap::new(),
            lru: LruOrder::default(),
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with default policy settings.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(CachePolicy::default())
    }

    // -----------------------------------------------------------------------
    // Core operations
    // -----------------------------------------------------------------------

    /// Insert `value` under `key`, evicting the LRU entry if at capacity.
    pub fn insert(&mut self, key: AssetCacheKey, value: serde_json::Value) {
        // If already present, remove old LRU record
        if self.store.contains_key(&key) {
            self.lru.remove(&key);
        } else {
            // Enforce capacity — evict LRU until we have room
            while self.store.len() >= self.policy.max_entries {
                if let Some(lru_key) = self.lru.evict_lru() {
                    self.store.remove(&lru_key);
                    self.stats.evictions += 1;
                } else {
                    break;
                }
            }
        }
        let entry = CacheEntry::new(value, self.policy.ttl);
        self.store.insert(key.clone(), entry);
        self.lru.touch(&key);
        self.stats.inserts += 1;
        self.stats.live_entries = self.store.len();
    }

    /// Retrieve the value for `key`.
    ///
    /// Returns `None` if the key is absent or its entry has expired.
    pub fn get(&mut self, key: &AssetCacheKey) -> Option<&serde_json::Value> {
        // Check for expiry
        let is_expired = self
            .store
            .get(key)
            .map(|e| !e.is_valid())
            .unwrap_or(true);

        if is_expired {
            if self.store.remove(key).is_some() {
                self.lru.remove(key);
                self.stats.evictions += 1;
                self.stats.live_entries = self.store.len();
            }
            self.stats.misses += 1;
            return None;
        }

        // Valid hit — update LRU and optionally slide TTL
        self.stats.hits += 1;
        self.lru.touch(key);
        let sliding = self.policy.sliding_ttl;
        let ttl = self.policy.ttl;
        if let Some(entry) = self.store.get_mut(key) {
            entry.touch(if sliding { Some(ttl) } else { None });
        }
        self.store.get(key).map(|e| &e.value)
    }

    /// Remove the entry for `key`, returning `true` if it was present.
    pub fn remove(&mut self, key: &AssetCacheKey) -> bool {
        let removed = self.store.remove(key).is_some();
        if removed {
            self.lru.remove(key);
            self.stats.live_entries = self.store.len();
        }
        removed
    }

    /// Returns `true` if `key` is present and not expired.
    pub fn contains(&mut self, key: &AssetCacheKey) -> bool {
        self.get(key).is_some()
    }

    // -----------------------------------------------------------------------
    // Bulk / maintenance operations
    // -----------------------------------------------------------------------

    /// Evict all entries whose TTL has elapsed.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_expired(&mut self) -> usize {
        let expired_keys: Vec<AssetCacheKey> = self
            .store
            .iter()
            .filter(|(_, e)| !e.is_valid())
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for key in &expired_keys {
            self.store.remove(key);
            self.lru.remove(key);
        }
        self.stats.evictions += count as u64;
        self.stats.live_entries = self.store.len();
        count
    }

    /// Remove all entries whose key matches the given prefix type.
    ///
    /// For example, passing `AssetCacheKey::SearchResult("".into())` will
    /// clear all search-result entries regardless of query string.
    pub fn invalidate_by_type(&mut self, key_type: CacheKeyType) -> usize {
        let to_remove: Vec<AssetCacheKey> = self
            .store
            .keys()
            .filter(|k| key_type.matches(k))
            .cloned()
            .collect();
        let count = to_remove.len();
        for key in &to_remove {
            self.store.remove(key);
            self.lru.remove(key);
        }
        self.stats.evictions += count as u64;
        self.stats.live_entries = self.store.len();
        count
    }

    /// Clear the entire cache.
    pub fn clear(&mut self) {
        self.store.clear();
        self.lru = LruOrder::default();
        self.stats.live_entries = 0;
    }

    // -----------------------------------------------------------------------
    // Introspection
    // -----------------------------------------------------------------------

    /// Current number of entries in the cache (including potentially expired).
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return a snapshot of the runtime statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Return the active cache policy.
    #[must_use]
    pub fn policy(&self) -> &CachePolicy {
        &self.policy
    }

    /// Return the number of live (non-expired) entries.
    ///
    /// This is a linear scan — prefer [`len`](Self::len) for O(1) approximate
    /// size and call [`evict_expired`](Self::evict_expired) first for accuracy.
    #[must_use]
    pub fn live_count(&self) -> usize {
        self.store.values().filter(|e| e.is_valid()).count()
    }

    /// List all keys currently in the cache (including expired).
    #[must_use]
    pub fn keys(&self) -> Vec<&AssetCacheKey> {
        self.store.keys().collect()
    }
}

// ---------------------------------------------------------------------------
// Key type discriminant for bulk invalidation
// ---------------------------------------------------------------------------

/// Discriminant used with [`AssetCache::invalidate_by_type`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKeyType {
    /// Match only `AssetById` keys.
    Asset,
    /// Match only `SearchResult` keys.
    Search,
    /// Match only `Collection` keys.
    Collection,
    /// Match only `UserPermissions` keys.
    UserPermissions,
    /// Match only `Custom` keys.
    Custom,
    /// Match all key types.
    All,
}

impl CacheKeyType {
    fn matches(&self, key: &AssetCacheKey) -> bool {
        match self {
            Self::Asset => matches!(key, AssetCacheKey::AssetById(_)),
            Self::Search => matches!(key, AssetCacheKey::SearchResult(_)),
            Self::Collection => matches!(key, AssetCacheKey::Collection(_)),
            Self::UserPermissions => matches!(key, AssetCacheKey::UserPermissions(_)),
            Self::Custom => matches!(key, AssetCacheKey::Custom(_)),
            Self::All => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn short_ttl_policy() -> CachePolicy {
        CachePolicy {
            ttl: Duration::from_millis(50),
            max_entries: 10,
            sliding_ttl: false,
        }
    }

    #[test]
    fn test_insert_and_get_hit() {
        let mut cache = AssetCache::with_defaults();
        let key = AssetCacheKey::AssetById("asset-1".into());
        cache.insert(key.clone(), json!({"title": "Reel One"}));
        let val = cache.get(&key);
        assert!(val.is_some());
        assert_eq!(val.expect("should be Some")["title"], "Reel One");
    }

    #[test]
    fn test_get_miss_absent_key() {
        let mut cache = AssetCache::with_defaults();
        let key = AssetCacheKey::AssetById("no-such-asset".into());
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_entry_expires_after_ttl() {
        let mut cache = AssetCache::new(short_ttl_policy());
        let key = AssetCacheKey::SearchResult("news".into());
        cache.insert(key.clone(), json!(["doc-1", "doc-2"]));
        std::thread::sleep(Duration::from_millis(60));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_lru_eviction_at_capacity() {
        let policy = CachePolicy { max_entries: 3, ..CachePolicy::default() };
        let mut cache = AssetCache::new(policy);
        for i in 0..4u32 {
            cache.insert(AssetCacheKey::AssetById(i.to_string()), json!(i));
        }
        // Cache should still respect max_entries
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_remove_existing_key() {
        let mut cache = AssetCache::with_defaults();
        let key = AssetCacheKey::AssetById("a1".into());
        cache.insert(key.clone(), json!(null));
        assert!(cache.remove(&key));
        assert!(!cache.contains(&key.clone()));
    }

    #[test]
    fn test_remove_absent_key_returns_false() {
        let mut cache = AssetCache::with_defaults();
        let key = AssetCacheKey::AssetById("ghost".into());
        assert!(!cache.remove(&key));
    }

    #[test]
    fn test_invalidate_by_type_clears_matching() {
        let mut cache = AssetCache::with_defaults();
        cache.insert(AssetCacheKey::AssetById("a1".into()), json!(1));
        cache.insert(AssetCacheKey::SearchResult("news".into()), json!([]));
        cache.insert(AssetCacheKey::Collection("c1".into()), json!([]));

        let evicted = cache.invalidate_by_type(CacheKeyType::Search);
        assert_eq!(evicted, 1);
        // Asset and collection keys should still be present
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_evict_expired_removes_stale() {
        let mut cache = AssetCache::new(short_ttl_policy());
        cache.insert(AssetCacheKey::AssetById("a1".into()), json!(1));
        cache.insert(AssetCacheKey::AssetById("a2".into()), json!(2));
        std::thread::sleep(Duration::from_millis(60));
        let evicted = cache.evict_expired();
        assert_eq!(evicted, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_clear_empties_cache() {
        let mut cache = AssetCache::with_defaults();
        cache.insert(AssetCacheKey::AssetById("a1".into()), json!(1));
        cache.insert(AssetCacheKey::UserPermissions("user-42".into()), json!(["read"]));
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_stats_hit_ratio() {
        let mut cache = AssetCache::with_defaults();
        let key = AssetCacheKey::AssetById("k1".into());
        cache.insert(key.clone(), json!("v"));
        cache.get(&key); // hit
        cache.get(&AssetCacheKey::AssetById("missing".into())); // miss
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        let ratio = stats.hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_key_as_str_repr() {
        assert_eq!(
            AssetCacheKey::AssetById("abc".into()).as_str_repr(),
            "asset:abc"
        );
        assert_eq!(
            AssetCacheKey::SearchResult("news".into()).as_str_repr(),
            "search:news"
        );
    }

    #[test]
    fn test_live_count_excludes_expired() {
        let mut cache = AssetCache::new(short_ttl_policy());
        cache.insert(AssetCacheKey::AssetById("x".into()), json!(1));
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(cache.live_count(), 0);
    }

    #[test]
    fn test_cache_policy_defaults() {
        let p = CachePolicy::default();
        assert_eq!(p.max_entries, 1_000);
        assert!(!p.sliding_ttl);
    }

    #[test]
    fn test_invalidate_all_clears_everything() {
        let mut cache = AssetCache::with_defaults();
        cache.insert(AssetCacheKey::AssetById("a".into()), json!(1));
        cache.insert(AssetCacheKey::SearchResult("q".into()), json!(2));
        let evicted = cache.invalidate_by_type(CacheKeyType::All);
        assert_eq!(evicted, 2);
        assert!(cache.is_empty());
    }
}
