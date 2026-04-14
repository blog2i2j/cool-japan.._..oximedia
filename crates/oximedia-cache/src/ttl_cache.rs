//! TTL-based cache with automatic expiry.
//!
//! This module provides [`TtlCache`], a capacity-bounded cache where each
//! entry carries an expiry deadline expressed as seconds since an arbitrary
//! epoch (e.g. `SystemTime::UNIX_EPOCH`).  Expired entries are never returned
//! and can be purged in bulk via [`TtlCache::remove_expired`].
//!
//! The caller is responsible for supplying monotonically-increasing
//! `now_secs` timestamps; this keeps the implementation deterministic and
//! easy to unit-test without `std::time` side-effects.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::ttl_cache::TtlCache;
//!
//! let mut cache: TtlCache<&str, Vec<u8>> = TtlCache::new(64, 30);
//! cache.insert("frame-001", vec![0u8; 128], 1_000);
//! assert!(cache.get(&"frame-001", 1_020).is_some()); // within TTL
//! assert!(cache.get(&"frame-001", 1_031).is_none()); // expired
//! ```

use std::collections::HashMap;

// ── TtlEntry ─────────────────────────────────────────────────────────────────

/// A single cache entry with an embedded TTL deadline.
#[derive(Debug, Clone)]
pub struct TtlEntry<V> {
    /// The cached value.
    pub value: V,
    /// Epoch-seconds at which this entry was inserted.
    pub inserted_at_secs: u64,
    /// Lifetime in seconds from insertion.
    pub ttl_secs: u64,
}

impl<V> TtlEntry<V> {
    /// Create a new entry with the given parameters.
    #[inline]
    pub fn new(value: V, inserted_at_secs: u64, ttl_secs: u64) -> Self {
        Self {
            value,
            inserted_at_secs,
            ttl_secs,
        }
    }

    /// Returns `true` when `now_secs` has passed the entry's deadline.
    ///
    /// A TTL of `0` means the entry never expires.
    #[inline]
    pub fn is_expired(&self, now_secs: u64) -> bool {
        if self.ttl_secs == 0 {
            return false;
        }
        now_secs >= self.inserted_at_secs.saturating_add(self.ttl_secs)
    }

    /// Seconds remaining before expiry, saturating at zero once expired.
    ///
    /// Returns `u64::MAX` for entries with TTL == 0 (no expiry).
    #[inline]
    pub fn remaining_secs(&self, now_secs: u64) -> u64 {
        if self.ttl_secs == 0 {
            return u64::MAX;
        }
        let deadline = self.inserted_at_secs.saturating_add(self.ttl_secs);
        deadline.saturating_sub(now_secs)
    }
}

// ── TtlCacheStats ─────────────────────────────────────────────────────────────

/// Snapshot of [`TtlCache`] statistics.
#[derive(Debug, Clone, Default)]
pub struct TtlCacheStats {
    /// Total number of entries in the underlying map (including expired ones
    /// that have not yet been purged).
    pub total_entries: usize,
    /// Number of entries that are currently expired (lazy-not-yet-purged).
    pub expired_entries: usize,
    /// Cumulative successful lookups (non-expired key found).
    pub hit_count: u64,
    /// Cumulative failed lookups (key absent or expired).
    pub miss_count: u64,
}

// ── TtlCache ──────────────────────────────────────────────────────────────────

/// Capacity-bounded TTL cache.
///
/// Entries are evicted in FIFO insertion order when the cache reaches
/// `capacity`.  Expired entries are returned as absent on [`get`] and can be
/// bulk-purged via [`remove_expired`].
///
/// # Type parameters
/// * `K` – key type; must implement `Eq + Hash + Clone`.
/// * `V` – value type.
///
/// [`get`]: TtlCache::get
/// [`remove_expired`]: TtlCache::remove_expired
pub struct TtlCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    capacity: usize,
    default_ttl_secs: u64,
    /// Primary storage.
    entries: HashMap<K, TtlEntry<V>>,
    /// Insertion-order record for FIFO eviction.
    insertion_order: Vec<K>,
    hit_count: u64,
    miss_count: u64,
}

impl<K, V> TtlCache<K, V>
where
    K: Eq + std::hash::Hash + Clone,
{
    /// Create a new `TtlCache` with the given `capacity` and `default_ttl_secs`.
    ///
    /// A `default_ttl_secs` of `0` means entries inserted via [`insert`] never
    /// expire; per-entry TTL overrides via [`insert_with_ttl`] still apply.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    ///
    /// [`insert`]: TtlCache::insert
    /// [`insert_with_ttl`]: TtlCache::insert_with_ttl
    pub fn new(capacity: usize, default_ttl_secs: u64) -> Self {
        assert!(capacity > 0, "TtlCache capacity must be non-zero");
        Self {
            capacity,
            default_ttl_secs,
            entries: HashMap::with_capacity(capacity.min(1024)),
            insertion_order: Vec::with_capacity(capacity.min(1024)),
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Insert a key-value pair using the cache's default TTL.
    ///
    /// If the key already exists it is overwritten.  When the cache is at
    /// `capacity`, the oldest entry (by insertion order) is evicted first.
    pub fn insert(&mut self, key: K, value: V, now_secs: u64) {
        self.insert_with_ttl(key, value, self.default_ttl_secs, now_secs);
    }

    /// Insert a key-value pair with an explicit `ttl_secs` override.
    ///
    /// A `ttl_secs` of `0` means this specific entry never expires.
    pub fn insert_with_ttl(&mut self, key: K, value: V, ttl_secs: u64, now_secs: u64) {
        // If we're replacing an existing key, remove the old insertion-order record.
        if self.entries.contains_key(&key) {
            self.insertion_order.retain(|k| k != &key);
        } else if self.entries.len() >= self.capacity {
            // Evict the oldest non-replaced entry.
            self.evict_oldest();
        }

        let entry = TtlEntry::new(value, now_secs, ttl_secs);
        self.entries.insert(key.clone(), entry);
        self.insertion_order.push(key);
    }

    /// Look up `key`.
    ///
    /// Returns `None` if the key is absent **or** if the entry has expired.
    /// Expired entries are lazy: they remain in memory until either
    /// [`remove_expired`] or a new insertion triggers eviction.
    ///
    /// [`remove_expired`]: TtlCache::remove_expired
    pub fn get(&mut self, key: &K, now_secs: u64) -> Option<&V> {
        match self.entries.get(key) {
            None => {
                self.miss_count += 1;
                None
            }
            Some(entry) if entry.is_expired(now_secs) => {
                self.miss_count += 1;
                None
            }
            Some(entry) => {
                self.hit_count += 1;
                Some(&entry.value)
            }
        }
    }

    /// Explicitly remove an entry from the cache regardless of expiry.
    ///
    /// Returns `true` if the key was present (even if expired).
    pub fn remove(&mut self, key: &K) -> bool {
        if self.entries.remove(key).is_some() {
            self.insertion_order.retain(|k| k != key);
            true
        } else {
            false
        }
    }

    /// Purge all expired entries and return the number removed.
    pub fn remove_expired(&mut self, now_secs: u64) -> usize {
        let expired_keys: Vec<K> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(now_secs))
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_keys.len();
        for k in &expired_keys {
            self.entries.remove(k);
        }
        self.insertion_order.retain(|k| !expired_keys.contains(k));
        count
    }

    /// Count of non-expired entries at `now_secs`.
    pub fn len(&self, now_secs: u64) -> usize {
        self.entries
            .values()
            .filter(|e| !e.is_expired(now_secs))
            .count()
    }

    /// Returns `true` when there are no non-expired entries.
    pub fn is_empty(&self, now_secs: u64) -> bool {
        self.len(now_secs) == 0
    }

    /// Return a statistics snapshot for the given `now_secs`.
    pub fn stats(&self, now_secs: u64) -> TtlCacheStats {
        let total = self.entries.len();
        let expired = self
            .entries
            .values()
            .filter(|e| e.is_expired(now_secs))
            .count();
        TtlCacheStats {
            total_entries: total,
            expired_entries: expired,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
        }
    }

    /// Configured maximum number of entries.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Remove the first entry in insertion order (oldest).
    fn evict_oldest(&mut self) {
        if self.insertion_order.is_empty() {
            return;
        }
        let oldest = self.insertion_order.remove(0);
        self.entries.remove(&oldest);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_within_ttl() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(10, 60);
        cache.insert("a", 42, 1000);
        assert_eq!(cache.get(&"a", 1059), Some(&42));
    }

    #[test]
    fn test_expired_entry_returns_none() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(10, 30);
        cache.insert("b", 99, 1000);
        // exactly at deadline: expired
        assert_eq!(cache.get(&"b", 1030), None);
        // after deadline
        assert_eq!(cache.get(&"b", 1100), None);
    }

    #[test]
    fn test_entry_at_expiry_boundary() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(10, 10);
        cache.insert("c", 7, 500);
        assert_eq!(cache.get(&"c", 509), Some(&7)); // 509 < 510
        assert_eq!(cache.get(&"c", 510), None); // exactly at deadline
    }

    #[test]
    fn test_remove_expired_count() {
        let mut cache: TtlCache<u32, &str> = TtlCache::new(20, 5);
        for i in 0..10 {
            cache.insert(i, "v", 0);
        }
        // all expire at t=5
        let purged = cache.remove_expired(5);
        assert_eq!(purged, 10);
        assert_eq!(cache.len(5), 0);
    }

    #[test]
    fn test_remove_expired_partial() {
        let mut cache: TtlCache<u32, u32> = TtlCache::new(20, 100);
        // short-lived entries
        for i in 0..5 {
            cache.insert_with_ttl(i, i * 10, 10, 0);
        }
        // long-lived entries
        for i in 5..10 {
            cache.insert_with_ttl(i, i * 10, 1000, 0);
        }
        let purged = cache.remove_expired(11);
        assert_eq!(purged, 5);
        assert_eq!(cache.len(11), 5);
    }

    #[test]
    fn test_per_entry_ttl_override() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(10, 60);
        cache.insert_with_ttl("short", 1, 5, 1000);
        cache.insert("long", 2, 1000); // uses default 60s
        assert_eq!(cache.get(&"short", 1006), None); // expired
        assert_eq!(cache.get(&"long", 1059), Some(&2)); // still alive
    }

    #[test]
    fn test_stats_hit_miss_expired() {
        let mut cache: TtlCache<u32, u32> = TtlCache::new(10, 50);
        cache.insert(1, 10, 0); // expires at t=50
        cache.insert(2, 20, 0); // expires at t=50

        let _ = cache.get(&1, 10); // hit (within TTL)
        let _ = cache.get(&3, 10); // miss (absent)
        let _ = cache.get(&2, 51); // miss (expired)

        // At t=51 both entries are expired (inserted_at=0, ttl=50 → deadline=50).
        let s = cache.stats(51);
        assert_eq!(s.hit_count, 1);
        assert_eq!(s.miss_count, 2);
        // Both entries 1 and 2 are past their deadline; neither has been purged.
        assert_eq!(s.expired_entries, 2);
    }

    #[test]
    fn test_capacity_eviction_fifo() {
        let mut cache: TtlCache<u32, u32> = TtlCache::new(3, 3600);
        cache.insert(1, 1, 0);
        cache.insert(2, 2, 0);
        cache.insert(3, 3, 0);
        // inserting 4th should evict key 1 (oldest)
        cache.insert(4, 4, 0);
        assert_eq!(cache.get(&1, 0), None);
        assert_eq!(cache.get(&2, 0), Some(&2));
        assert_eq!(cache.get(&3, 0), Some(&3));
        assert_eq!(cache.get(&4, 0), Some(&4));
    }

    #[test]
    fn test_zero_ttl_never_expires() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(10, 0);
        cache.insert("immortal", 55, 0);
        // far in the future
        assert_eq!(cache.get(&"immortal", u64::MAX / 2), Some(&55));
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut cache: TtlCache<&str, u32> = TtlCache::new(5, 100);
        cache.insert("k", 1, 0);
        cache.insert("k", 2, 0); // overwrite
        assert_eq!(cache.get(&"k", 0), Some(&2));
        // insertion order has only one record for "k" after overwrite
        assert_eq!(cache.len(0), 1);
    }

    #[test]
    fn test_remove_entry() {
        let mut cache: TtlCache<u32, u32> = TtlCache::new(10, 100);
        cache.insert(7, 77, 0);
        assert!(cache.remove(&7));
        assert!(!cache.remove(&7)); // already gone
        assert_eq!(cache.get(&7, 0), None);
    }

    #[test]
    fn test_is_empty() {
        let mut cache: TtlCache<u32, u32> = TtlCache::new(4, 10);
        assert!(cache.is_empty(0));
        cache.insert(1, 1, 0);
        assert!(!cache.is_empty(0));
        assert!(cache.is_empty(11)); // all expired
    }
}
