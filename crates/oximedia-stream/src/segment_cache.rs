//! In-memory segment cache for the streaming pipeline.
//!
//! Provides a capacity-bounded, LRU-evicting cache that maps segment IDs
//! (or URL paths) to raw byte payloads.  By serving repeated requests from
//! memory the packager and CDN origin avoid redundant disk I/O while the
//! segment is still hot.
//!
//! # Design
//!
//! * **LRU eviction** — when the cache is at capacity the least-recently-used
//!   entry is evicted automatically on every [`SegmentCache::insert`] call.
//! * **Byte-budget enforcement** — besides the entry-count limit an optional
//!   total-byte budget caps memory usage; entries are evicted (oldest first)
//!   until the budget is satisfied.
//! * **Statistics** — hit/miss counters let callers measure effectiveness.
//! * **No unsafe code, no `unwrap()`** — all operations return `Option` or
//!   `Result` as appropriate.

use std::collections::HashMap;

// ─── Cache entry ──────────────────────────────────────────────────────────────

/// A cached segment payload together with access-order bookkeeping.
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Raw segment bytes.
    data: Vec<u8>,
    /// Monotonically increasing generation counter set on every access.
    /// Lower values are older (i.e. less-recently used).
    last_access: u64,
}

// ─── Cache statistics ─────────────────────────────────────────────────────────

/// Cumulative cache performance counters.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of [`SegmentCache::get`] calls that returned `Some(…)`.
    pub hits: u64,
    /// Number of [`SegmentCache::get`] calls that returned `None`.
    pub misses: u64,
    /// Number of entries evicted due to capacity or byte-budget limits.
    pub evictions: u64,
    /// Total bytes currently held in the cache.
    pub bytes_used: usize,
    /// Number of entries currently held in the cache.
    pub entry_count: usize,
}

impl CacheStats {
    /// Cache hit ratio in the range `[0.0, 1.0]`, or `0.0` if no lookups yet.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ─── SegmentCache ─────────────────────────────────────────────────────────────

/// In-memory LRU cache for raw segment payloads.
///
/// ```
/// use oximedia_stream::segment_cache::SegmentCache;
///
/// let mut cache = SegmentCache::new(128, 4 * 1024 * 1024); // 128 entries, 4 MiB
/// cache.insert("seg-0001".to_string(), vec![0u8; 1024]);
/// assert!(cache.get("seg-0001").is_some());
/// assert!(cache.get("seg-9999").is_none());
/// ```
#[derive(Debug)]
pub struct SegmentCache {
    /// Maximum number of entries before LRU eviction kicks in.
    max_entries: usize,
    /// Maximum total bytes before byte-budget eviction kicks in.
    max_bytes: usize,
    /// Monotonic clock used to order accesses.
    clock: u64,
    /// The actual cached data, keyed by segment ID / URL path.
    store: HashMap<String, CacheEntry>,
    /// Hit counter.
    hits: u64,
    /// Miss counter.
    misses: u64,
    /// Eviction counter.
    evictions: u64,
    /// Current byte usage.
    bytes_used: usize,
}

impl SegmentCache {
    /// Create a new cache.
    ///
    /// # Parameters
    ///
    /// * `max_entries` — maximum number of entries (minimum 1).
    /// * `max_bytes`   — maximum total bytes across all entries (minimum 1).
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            max_bytes: max_bytes.max(1),
            clock: 0,
            store: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            bytes_used: 0,
        }
    }

    /// Insert (or replace) a segment payload, evicting old entries as needed.
    ///
    /// If `data` alone exceeds `max_bytes` the entry is **not** inserted and
    /// the function returns `false`.  Returns `true` when the entry is stored
    /// successfully.
    pub fn insert(&mut self, key: String, data: Vec<u8>) -> bool {
        if data.len() > self.max_bytes {
            return false;
        }

        // If the key already exists, remove its old byte contribution.
        if let Some(old) = self.store.remove(&key) {
            self.bytes_used = self.bytes_used.saturating_sub(old.data.len());
        }

        // Evict LRU entries until there is room (entry count + byte budget).
        while (self.store.len() >= self.max_entries)
            || (self.bytes_used + data.len() > self.max_bytes)
        {
            if self.store.is_empty() {
                break;
            }
            self.evict_lru();
        }

        // Final guard: if we still can't fit the entry, reject it.
        if self.bytes_used + data.len() > self.max_bytes {
            return false;
        }

        self.clock = self.clock.wrapping_add(1);
        self.bytes_used += data.len();
        self.store.insert(
            key,
            CacheEntry {
                data,
                last_access: self.clock,
            },
        );
        true
    }

    /// Retrieve a segment payload by key, updating its access time.
    ///
    /// Returns `None` on a cache miss.
    pub fn get(&mut self, key: &str) -> Option<&[u8]> {
        if let Some(entry) = self.store.get_mut(key) {
            self.clock = self.clock.wrapping_add(1);
            entry.last_access = self.clock;
            self.hits += 1;
            Some(entry.data.as_slice())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Remove a single entry by key, returning `true` if it existed.
    pub fn remove(&mut self, key: &str) -> bool {
        if let Some(old) = self.store.remove(key) {
            self.bytes_used = self.bytes_used.saturating_sub(old.data.len());
            true
        } else {
            false
        }
    }

    /// Evict all entries from the cache.
    pub fn clear(&mut self) {
        self.store.clear();
        self.bytes_used = 0;
    }

    /// Whether the cache contains an entry for `key`.
    pub fn contains(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    /// Current number of entries held in the cache.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// `true` if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Current total bytes used by all cached payloads.
    pub fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    /// Return a snapshot of all cumulative statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            bytes_used: self.bytes_used,
            entry_count: self.store.len(),
        }
    }

    /// Reset hit/miss/eviction counters without clearing cached data.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Evict the single least-recently-used entry.
    fn evict_lru(&mut self) {
        // Find the key with the smallest `last_access` value.
        let lru_key = self
            .store
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = self.store.remove(&key) {
                self.bytes_used = self.bytes_used.saturating_sub(entry.data.len());
                self.evictions += 1;
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_payload(size: usize) -> Vec<u8> {
        vec![0xAB_u8; size]
    }

    #[test]
    fn test_insert_and_get_basic() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("seg-001".to_string(), make_payload(512));
        let data = cache.get("seg-001");
        assert!(data.is_some());
        assert_eq!(data.unwrap().len(), 512);
    }

    #[test]
    fn test_cache_miss_returns_none() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_hit_miss_counters() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("k1".to_string(), make_payload(64));
        cache.get("k1");
        cache.get("k1");
        cache.get("nope");
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_hit_ratio_calculation() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("a".to_string(), make_payload(128));
        cache.get("a");
        cache.get("b"); // miss
        let stats = cache.stats();
        let ratio = stats.hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-9, "expected 0.5 got {ratio}");
    }

    #[test]
    fn test_lru_eviction_on_capacity() {
        let mut cache = SegmentCache::new(3, 1024 * 1024);
        cache.insert("a".to_string(), make_payload(64));
        cache.insert("b".to_string(), make_payload(64));
        cache.insert("c".to_string(), make_payload(64));
        // Access "a" and "b" to make "a" more recent than "b"
        cache.get("b");
        cache.get("a");
        // Inserting "d" should evict the LRU entry, which is "c" (never accessed after insert)
        cache.insert("d".to_string(), make_payload(64));
        assert_eq!(cache.len(), 3);
        assert!(!cache.contains("c"), "'c' should have been evicted");
        assert!(cache.contains("a"));
        assert!(cache.contains("b"));
        assert!(cache.contains("d"));
    }

    #[test]
    fn test_byte_budget_eviction() {
        // Budget: 256 bytes, 100 entries max
        let mut cache = SegmentCache::new(100, 256);
        cache.insert("a".to_string(), make_payload(100));
        cache.insert("b".to_string(), make_payload(100));
        // Third insert should evict "a" (LRU) to fit
        cache.insert("c".to_string(), make_payload(100));
        assert!(cache.bytes_used() <= 256);
    }

    #[test]
    fn test_oversized_entry_rejected() {
        let mut cache = SegmentCache::new(10, 512);
        let inserted = cache.insert("giant".to_string(), make_payload(1024));
        assert!(!inserted, "oversized entry must be rejected");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_replace_existing_key_updates_bytes() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("k".to_string(), make_payload(200));
        let bytes_before = cache.bytes_used();
        cache.insert("k".to_string(), make_payload(100));
        let bytes_after = cache.bytes_used();
        assert_eq!(bytes_before, 200);
        assert_eq!(bytes_after, 100, "replacing entry should update byte count");
    }

    #[test]
    fn test_remove_entry() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("r".to_string(), make_payload(128));
        assert!(cache.contains("r"));
        let removed = cache.remove("r");
        assert!(removed);
        assert!(!cache.contains("r"));
        assert_eq!(cache.bytes_used(), 0);
    }

    #[test]
    fn test_remove_nonexistent_returns_false() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        assert!(!cache.remove("nope"));
    }

    #[test]
    fn test_clear_empties_cache() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("x".to_string(), make_payload(256));
        cache.insert("y".to_string(), make_payload(256));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.bytes_used(), 0);
    }

    #[test]
    fn test_eviction_counter_increments() {
        let mut cache = SegmentCache::new(2, 1024 * 1024);
        cache.insert("a".to_string(), make_payload(64));
        cache.insert("b".to_string(), make_payload(64));
        cache.insert("c".to_string(), make_payload(64)); // evicts one
        let stats = cache.stats();
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn test_stats_entry_count_matches_len() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("p".to_string(), make_payload(64));
        cache.insert("q".to_string(), make_payload(64));
        let stats = cache.stats();
        assert_eq!(stats.entry_count, cache.len());
        assert_eq!(stats.entry_count, 2);
    }

    #[test]
    fn test_reset_stats_preserves_data() {
        let mut cache = SegmentCache::new(10, 1024 * 1024);
        cache.insert("m".to_string(), make_payload(64));
        cache.get("m");
        cache.get("missing");
        cache.reset_stats();
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        // Data still present
        assert!(cache.contains("m"));
    }

    #[test]
    fn test_hit_ratio_zero_when_no_lookups() {
        let cache = SegmentCache::new(10, 1024 * 1024);
        assert_eq!(cache.stats().hit_ratio(), 0.0);
    }
}
