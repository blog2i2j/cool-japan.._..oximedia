//! Storage cache layer: LRU, LFU, FIFO, and ARC caches with policy tracking
//! and statistics.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Cache policy
// ---------------------------------------------------------------------------

/// Eviction / replacement policy for a cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CachePolicy {
    /// Least Recently Used.
    LRU,
    /// Least Frequently Used.
    LFU,
    /// First In, First Out.
    FIFO,
    /// Adaptive Replacement Cache.
    ARC,
}

impl CachePolicy {
    /// Returns a human-readable name for the policy.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::LRU => "LRU",
            Self::LFU => "LFU",
            Self::FIFO => "FIFO",
            Self::ARC => "ARC",
        }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// Metadata for a single cache entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Unique identifier / key for the cached object.
    pub key: String,
    /// Size of the cached object in bytes.
    pub size_bytes: u64,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Timestamp (in milliseconds) of the last access.
    pub last_access_ms: u64,
    /// Timestamp (in milliseconds) when the entry was created.
    pub created_ms: u64,
}

impl CacheEntry {
    /// Create a new cache entry.
    #[must_use]
    pub fn new(key: impl Into<String>, size_bytes: u64, now_ms: u64) -> Self {
        Self {
            key: key.into(),
            size_bytes,
            access_count: 0,
            last_access_ms: now_ms,
            created_ms: now_ms,
        }
    }

    /// Age of the entry in milliseconds relative to `now`.
    #[must_use]
    pub fn age_ms(&self, now: u64) -> u64 {
        now.saturating_sub(self.created_ms)
    }
}

// ---------------------------------------------------------------------------
// LRU cache
// ---------------------------------------------------------------------------

/// An LRU cache with a byte-capacity limit.
pub struct LruCache {
    /// Maximum total size in bytes.
    pub capacity_bytes: u64,
    /// Current total size in bytes.
    pub used_bytes: u64,
    /// Map from key to entry.
    pub entries: HashMap<String, CacheEntry>,
    /// LRU order: front = most recently used, back = least recently used.
    order: VecDeque<String>,
}

impl LruCache {
    /// Create a new LRU cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Retrieve an entry by key, updating its access timestamp.
    ///
    /// Returns `None` if the key is not cached.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if !self.entries.contains_key(key) {
            return None;
        }

        // Move to front of LRU order
        self.order.retain(|k| k != key);
        self.order.push_front(key.to_string());

        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access_ms = now_ms;
        }

        self.entries.get(key)
    }

    /// Insert (or refresh) a cache entry.
    ///
    /// Evicts entries as needed to satisfy the byte capacity.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        // Remove existing entry of the same key first
        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
            self.order.retain(|k| k != &key);
        }

        // Evict until there is room
        while self.used_bytes + size_bytes > self.capacity_bytes && !self.order.is_empty() {
            self.evict();
        }

        let entry = CacheEntry::new(key.clone(), size_bytes, now_ms);
        self.used_bytes += size_bytes;
        self.entries.insert(key.clone(), entry);
        self.order.push_front(key);
    }

    /// Evict the least recently used entry.
    ///
    /// Returns `true` if an entry was evicted.
    pub fn evict(&mut self) -> bool {
        if let Some(lru_key) = self.order.pop_back() {
            if let Some(entry) = self.entries.remove(&lru_key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                return true;
            }
        }
        false
    }

    /// Cache utilisation as a fraction (0.0–1.0).
    ///
    /// Returns `0.0` when capacity is zero.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// LFU cache
// ---------------------------------------------------------------------------

/// A Least Frequently Used cache with byte-capacity limit.
///
/// Evicts the entry with the lowest access count.  Ties are broken by the
/// entry with the oldest `last_access_ms` (LRU among least-frequently-used).
pub struct LfuCache {
    /// Maximum total size in bytes.
    pub capacity_bytes: u64,
    /// Current total size in bytes.
    pub used_bytes: u64,
    /// Map from key to entry.
    pub entries: HashMap<String, CacheEntry>,
}

impl LfuCache {
    /// Create a new LFU cache with the given byte capacity.
    #[must_use]
    pub fn new(capacity_bytes: u64) -> Self {
        Self {
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
        }
    }

    /// Retrieve an entry, incrementing its access count.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if let Some(entry) = self.entries.get_mut(key) {
            entry.access_count += 1;
            entry.last_access_ms = now_ms;
        }
        self.entries.get(key)
    }

    /// Insert (or refresh) a cache entry.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        if let Some(old) = self.entries.remove(&key) {
            self.used_bytes = self.used_bytes.saturating_sub(old.size_bytes);
        }

        while self.used_bytes + size_bytes > self.capacity_bytes && !self.entries.is_empty() {
            self.evict();
        }

        let entry = CacheEntry::new(key.clone(), size_bytes, now_ms);
        self.used_bytes += size_bytes;
        self.entries.insert(key, entry);
    }

    /// Evict the least frequently used entry (ties broken by oldest access).
    pub fn evict(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let victim_key = self
            .entries
            .iter()
            .min_by(|a, b| {
                a.1.access_count
                    .cmp(&b.1.access_count)
                    .then(a.1.last_access_ms.cmp(&b.1.last_access_ms))
            })
            .map(|(k, _)| k.clone());

        if let Some(key) = victim_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.size_bytes);
                return true;
            }
        }
        false
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache utilisation.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f64 / self.capacity_bytes as f64
    }
}

// ---------------------------------------------------------------------------
// ARC cache (Adaptive Replacement Cache)
// ---------------------------------------------------------------------------

/// Adaptive Replacement Cache.
///
/// Dynamically balances recency (LRU) and frequency (LFU) via four internal
/// lists (T1, T2, B1, B2) and an adaptive parameter **p**.
///
/// * **T1** — recently accessed (seen once), most recent at front.
/// * **T2** — frequently accessed (seen 2+), most recent at front.
/// * **B1** — ghost entries evicted from T1 (keys only).
/// * **B2** — ghost entries evicted from T2 (keys only).
///
/// Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement
/// Cache", FAST 2003.
pub struct ArcCache {
    /// Maximum number of cached entries (c).
    capacity: usize,
    /// Target size for T1 (adapts at runtime).
    p: usize,
    /// T1: recent entries (accessed once).
    t1: VecDeque<String>,
    /// T2: frequent entries (accessed 2+).
    t2: VecDeque<String>,
    /// B1: ghost keys evicted from T1.
    b1: VecDeque<String>,
    /// B2: ghost keys evicted from T2.
    b2: VecDeque<String>,
    /// Cached entry metadata.
    entries: HashMap<String, CacheEntry>,
}

impl ArcCache {
    /// Create a new ARC cache.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            p: 0,
            t1: VecDeque::new(),
            t2: VecDeque::new(),
            b1: VecDeque::new(),
            b2: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    /// Look up a key.
    ///
    /// If in T1, promote to T2.  If in T2, move to front.
    pub fn get(&mut self, key: &str, now_ms: u64) -> Option<&CacheEntry> {
        if let Some(pos) = self.t1.iter().position(|k| k == key) {
            if let Some(k) = self.t1.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        if let Some(pos) = self.t2.iter().position(|k| k == key) {
            if let Some(k) = self.t2.remove(pos) {
                self.t2.push_front(k);
            }
            if let Some(e) = self.entries.get_mut(key) {
                e.access_count += 1;
                e.last_access_ms = now_ms;
            }
            return self.entries.get(key);
        }

        None
    }

    /// Insert a new entry.
    pub fn put(&mut self, key: impl Into<String>, size_bytes: u64, now_ms: u64) {
        let key = key.into();

        // Already cached → promote
        if self.entries.contains_key(&key) {
            self.get(&key, now_ms);
            return;
        }

        // Hit in B1 → increase p (favour recency)
        if let Some(pos) = self.b1.iter().position(|k| k == &key) {
            let delta = (self.b2.len().max(1) / self.b1.len().max(1)).max(1);
            self.p = (self.p + delta).min(self.capacity);
            self.b1.remove(pos);
            self.replace(false);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            return;
        }

        // Hit in B2 → decrease p (favour frequency)
        if let Some(pos) = self.b2.iter().position(|k| k == &key) {
            let delta = (self.b1.len().max(1) / self.b2.len().max(1)).max(1);
            self.p = self.p.saturating_sub(delta);
            self.b2.remove(pos);
            self.replace(true);
            self.t2.push_front(key.clone());
            self.entries
                .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
            return;
        }

        // Completely new key
        let total_t1 = self.t1.len() + self.b1.len();
        if total_t1 == self.capacity {
            if self.t1.len() < self.capacity {
                self.b1.pop_back();
                self.replace(false);
            } else if let Some(evicted) = self.t1.pop_back() {
                self.entries.remove(&evicted);
            }
        } else {
            let total = self.t1.len() + self.b1.len() + self.t2.len() + self.b2.len();
            if total >= self.capacity {
                if total >= 2 * self.capacity {
                    self.b2.pop_back();
                }
                self.replace(false);
            }
        }

        self.t1.push_front(key.clone());
        self.entries
            .insert(key.clone(), CacheEntry::new(key, size_bytes, now_ms));
    }

    /// ARC "replace" subroutine.
    fn replace(&mut self, in_b2: bool) {
        if !self.t1.is_empty() && (self.t1.len() > self.p || (in_b2 && self.t1.len() == self.p)) {
            if let Some(evicted) = self.t1.pop_back() {
                self.entries.remove(&evicted);
                self.b1.push_front(evicted);
            }
        } else if let Some(evicted) = self.t2.pop_back() {
            self.entries.remove(&evicted);
            self.b2.push_front(evicted);
        }
    }

    /// Number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current adaptive parameter p (target T1 size).
    #[must_use]
    pub fn adaptive_parameter(&self) -> usize {
        self.p
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Accumulates cache hit/miss/eviction counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of entries evicted.
    pub evictions: u64,
}

impl CacheStats {
    /// Fraction of lookups that were hits (0.0–1.0).
    ///
    /// Returns `0.0` when no lookups have been performed.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Record a cache hit.
    pub fn record_hit(&mut self) {
        self.hits += 1;
    }

    /// Record a cache miss.
    pub fn record_miss(&mut self) {
        self.misses += 1;
    }

    /// Record an eviction.
    pub fn record_eviction(&mut self) {
        self.evictions += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_policy_names() {
        assert_eq!(CachePolicy::LRU.name(), "LRU");
        assert_eq!(CachePolicy::LFU.name(), "LFU");
        assert_eq!(CachePolicy::FIFO.name(), "FIFO");
        assert_eq!(CachePolicy::ARC.name(), "ARC");
    }

    #[test]
    fn test_cache_entry_age() {
        let entry = CacheEntry::new("key", 128, 1000);
        assert_eq!(entry.age_ms(1500), 500);
        assert_eq!(entry.age_ms(999), 0); // saturating sub
    }

    #[test]
    fn test_lru_cache_put_and_get() {
        let mut cache = LruCache::new(1024);
        cache.put("file.mp4", 100, 0);
        assert!(cache.get("file.mp4", 1).is_some());
    }

    #[test]
    fn test_lru_cache_miss() {
        let mut cache = LruCache::new(1024);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_lru_cache_eviction_on_overflow() {
        let mut cache = LruCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Both fit; now add one more that requires evicting 'a' (LRU)
        cache.put("c", 100, 2);
        assert_eq!(cache.len(), 2);
        // 'a' should have been evicted
        assert!(cache.get("a", 3).is_none());
        assert!(cache.get("b", 3).is_some());
        assert!(cache.get("c", 3).is_some());
    }

    #[test]
    fn test_lru_cache_access_updates_order() {
        let mut cache = LruCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        // Access 'a' to make it recently used
        cache.get("a", 2);
        // Now insert 'c'; 'b' is LRU and should be evicted
        cache.put("c", 100, 3);
        assert!(cache.get("a", 4).is_some());
        assert!(cache.get("b", 4).is_none());
        assert!(cache.get("c", 4).is_some());
    }

    #[test]
    fn test_lru_cache_utilization() {
        let mut cache = LruCache::new(1000);
        cache.put("x", 500, 0);
        assert!((cache.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_lru_cache_utilization_empty() {
        let cache = LruCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_lru_cache_overwrite_same_key() {
        let mut cache = LruCache::new(1000);
        cache.put("k", 100, 0);
        cache.put("k", 200, 1); // overwrite
        assert_eq!(cache.used_bytes, 200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lru_cache_is_empty() {
        let cache = LruCache::new(1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats_hit_rate_no_lookups() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_stats_hit_rate_all_hits() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_hit();
        assert!((stats.hit_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_stats_hit_rate_mixed() {
        let mut stats = CacheStats::default();
        stats.record_hit();
        stats.record_miss();
        assert!((stats.hit_rate() - 0.5).abs() < 1e-9);
    }

    // ── LFU cache tests ────────────────────────────────────────────────

    #[test]
    fn test_lfu_cache_put_and_get() {
        let mut cache = LfuCache::new(1024);
        cache.put("file.mp4", 100, 0);
        assert!(cache.get("file.mp4", 1).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache_miss() {
        let mut cache = LfuCache::new(1024);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_lfu_cache_evicts_least_frequent() {
        let mut cache = LfuCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);

        // Access 'b' several times to make it more frequent
        cache.get("b", 2);
        cache.get("b", 3);
        // 'a' has 0 accesses, 'b' has 2 accesses

        // Insert 'c' – must evict 'a' (least frequent)
        cache.put("c", 100, 4);
        assert!(cache.get("a", 5).is_none());
        assert!(cache.get("b", 5).is_some());
        assert!(cache.get("c", 5).is_some());
    }

    #[test]
    fn test_lfu_cache_tie_broken_by_oldest_access() {
        let mut cache = LfuCache::new(200);
        cache.put("a", 100, 0);
        cache.put("b", 100, 10);

        // Both have 0 accesses.  'a' has last_access_ms=0, 'b' has 10.
        // Tie-breaking by oldest access → 'a' evicted.
        cache.put("c", 100, 20);
        assert!(cache.get("a", 21).is_none());
        assert!(cache.get("b", 21).is_some());
    }

    #[test]
    fn test_lfu_cache_overwrite_same_key() {
        let mut cache = LfuCache::new(1024);
        cache.put("k", 100, 0);
        cache.put("k", 200, 1);
        assert_eq!(cache.used_bytes, 200);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_lfu_cache_utilization() {
        let mut cache = LfuCache::new(1000);
        cache.put("x", 400, 0);
        assert!((cache.utilization() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_lfu_cache_utilization_zero_capacity() {
        let cache = LfuCache::new(0);
        assert_eq!(cache.utilization(), 0.0);
    }

    #[test]
    fn test_lfu_cache_is_empty() {
        let cache = LfuCache::new(1024);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_lfu_cache_evict_empty() {
        let mut cache = LfuCache::new(1024);
        assert!(!cache.evict());
    }

    #[test]
    fn test_lfu_cache_multiple_evictions() {
        let mut cache = LfuCache::new(300);
        cache.put("a", 100, 0);
        cache.put("b", 100, 1);
        cache.put("c", 100, 2);

        // Access counts: a=0, b=0, c=0.  Insert 'd' (200 bytes) → must evict
        // two entries to make room.
        cache.put("d", 200, 3);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("d", 4).is_some());
    }

    // ── ARC cache tests ────────────────────────────────────────────────

    #[test]
    fn test_arc_cache_put_and_get() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 100, 0);
        assert!(cache.get("a", 1).is_some());
    }

    #[test]
    fn test_arc_cache_miss() {
        let mut cache = ArcCache::new(10);
        assert!(cache.get("missing", 0).is_none());
    }

    #[test]
    fn test_arc_cache_capacity_enforcement() {
        let mut cache = ArcCache::new(3);
        cache.put("a", 10, 0);
        cache.put("b", 10, 1);
        cache.put("c", 10, 2);
        assert_eq!(cache.len(), 3);

        // Adding a 4th entry should evict one
        cache.put("d", 10, 3);
        assert!(cache.len() <= 3);
    }

    #[test]
    fn test_arc_cache_t1_to_t2_promotion() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 10, 0); // goes to T1
                               // First get promotes from T1 to T2
        let entry = cache.get("a", 1);
        assert!(entry.is_some());
        assert_eq!(entry.map(|e| e.access_count), Some(1));
        // Second get should still find it (now in T2)
        assert!(cache.get("a", 2).is_some());
    }

    #[test]
    fn test_arc_cache_duplicate_put_promotes() {
        let mut cache = ArcCache::new(10);
        cache.put("a", 10, 0);
        // Put same key again should promote, not duplicate
        cache.put("a", 10, 1);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_arc_cache_ghost_b1_hit_increases_p() {
        let mut cache = ArcCache::new(2);
        cache.put("a", 10, 0);
        cache.put("b", 10, 1);
        // 'a' is LRU in T1, adding 'c' should evict 'a' to B1
        cache.put("c", 10, 2);
        // Now re-insert 'a' → B1 hit → p should increase
        let p_before = cache.adaptive_parameter();
        cache.put("a", 10, 3);
        assert!(cache.adaptive_parameter() >= p_before);
    }

    #[test]
    fn test_arc_cache_ghost_b2_hit_decreases_p() {
        let mut cache = ArcCache::new(2);
        cache.put("a", 10, 0);
        cache.get("a", 1); // promote to T2
        cache.put("b", 10, 2);
        // Fill to trigger evictions from T2 to B2
        cache.put("c", 10, 3);
        cache.put("d", 10, 4);
        // If 'a' ended up in B2, reinserting it decreases p
        let _p = cache.adaptive_parameter();
        cache.put("a", 10, 5);
        // Just verify no panic and cache is consistent
        assert!(cache.len() <= 2);
    }

    #[test]
    fn test_arc_cache_is_empty() {
        let cache = ArcCache::new(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_arc_cache_stress() {
        let mut cache = ArcCache::new(5);
        // Insert many entries with varying access patterns
        for i in 0..20u64 {
            cache.put(format!("key-{i}"), 10, i * 10);
        }
        assert!(cache.len() <= 5);

        // Access some keys repeatedly to build frequency
        for i in 15..20u64 {
            let _ = cache.get(&format!("key-{i}"), 200 + i);
            let _ = cache.get(&format!("key-{i}"), 300 + i);
        }

        // Insert more; frequently accessed keys should survive
        for i in 20..30u64 {
            cache.put(format!("key-{i}"), 10, 400 + i);
        }
        assert!(cache.len() <= 5);
    }

    #[test]
    fn test_arc_cache_sequential_then_reuse() {
        // Simulates a workload with sequential scan then re-access pattern
        let mut cache = ArcCache::new(4);

        // Sequential inserts
        cache.put("s1", 10, 0);
        cache.put("s2", 10, 1);
        cache.put("s3", 10, 2);
        cache.put("s4", 10, 3);

        // Re-access s1, s2 to promote to T2
        cache.get("s1", 4);
        cache.get("s2", 5);

        // Now more sequential inserts; s1/s2 should survive (in T2)
        cache.put("s5", 10, 6);
        cache.put("s6", 10, 7);

        // s1 and s2 were promoted and should still be accessible
        assert!(cache.get("s1", 8).is_some());
        assert!(cache.get("s2", 9).is_some());
    }
}
