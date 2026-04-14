#![allow(dead_code)]
//! In-memory LRU caching layer for frequently accessed job state.
//!
//! Wraps a fixed-capacity LRU cache that sits in front of the SQLite
//! persistence layer.  Cache entries are invalidated on every write so
//! readers always see up-to-date data after a brief publication lag.
//!
//! ## Design
//!
//! - **Fixed capacity**: the cache holds at most `capacity` entries.  When
//!   full, the least-recently-used entry is evicted to make room.
//! - **TTL expiry**: each entry carries a creation timestamp.  Entries older
//!   than `ttl` are treated as stale and bypassed (the caller is expected to
//!   refresh from the authoritative store).
//! - **Write-through invalidation**: the cache exposes an explicit
//!   `invalidate` method; callers must call it whenever the underlying state
//!   changes.
//! - **Pure Rust**: implemented with a `VecDeque`-backed ordered list to
//!   avoid additional crate dependencies.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A single cached item with eviction metadata.
struct CacheEntry<V> {
    /// Cached value.
    value: V,
    /// Wall-clock instant when this entry was inserted.
    inserted_at: Instant,
    /// LRU access counter; higher = more recently used.
    access_seq: u64,
}

impl<V> CacheEntry<V> {
    fn new(value: V, seq: u64) -> Self {
        Self {
            value,
            inserted_at: Instant::now(),
            access_seq: seq,
        }
    }

    fn is_stale(&self, ttl: Duration) -> bool {
        self.inserted_at.elapsed() > ttl
    }
}

// ---------------------------------------------------------------------------
// LRU cache
// ---------------------------------------------------------------------------

/// Metrics counters for observability.
#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    /// Number of successful cache lookups.
    pub hits: u64,
    /// Number of cache misses (key absent or entry stale).
    pub misses: u64,
    /// Number of entries evicted due to capacity pressure.
    pub evictions: u64,
    /// Number of explicit invalidations.
    pub invalidations: u64,
    /// Number of insertions.
    pub insertions: u64,
}

impl CacheMetrics {
    /// Hit rate as a fraction in `[0.0, 1.0]`.  Returns `0.0` on no traffic.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// A capacity-bounded, TTL-aware, LRU eviction cache keyed by `K`.
///
/// This is a general-purpose cache; see [`JobStateCache`] for the
/// farm-specific wrapper around [`crate::coordinator::Job`].
pub struct LruCache<K, V> {
    entries: HashMap<K, CacheEntry<V>>,
    capacity: usize,
    ttl: Duration,
    /// Monotonically increasing sequence counter for LRU ordering.
    seq: u64,
    /// Aggregated metrics.
    pub metrics: CacheMetrics,
}

impl<K, V> LruCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new cache with the given capacity and TTL.
    ///
    /// `capacity` must be ≥ 1.  `ttl` of `Duration::MAX` effectively disables
    /// expiry.
    #[must_use]
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        assert!(capacity > 0, "cache capacity must be at least 1");
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            ttl,
            seq: 0,
            metrics: CacheMetrics::default(),
        }
    }

    /// Insert or replace a value.
    ///
    /// If inserting would exceed `capacity`, the LRU entry is evicted first.
    pub fn insert(&mut self, key: K, value: V) {
        // Evict the LRU entry if at capacity and the key is not already present.
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        self.seq += 1;
        let entry = CacheEntry::new(value, self.seq);
        self.entries.insert(key, entry);
        self.metrics.insertions += 1;
    }

    /// Retrieve a value by key.
    ///
    /// Returns `None` when the key is absent or the entry has expired.  On a
    /// hit, the entry's access sequence is refreshed (LRU update).
    pub fn get(&mut self, key: &K) -> Option<V> {
        let ttl = self.ttl;
        match self.entries.get_mut(key) {
            Some(entry) if !entry.is_stale(ttl) => {
                self.seq += 1;
                entry.access_seq = self.seq;
                self.metrics.hits += 1;
                Some(entry.value.clone())
            }
            Some(_stale) => {
                self.entries.remove(key);
                self.metrics.misses += 1;
                None
            }
            None => {
                self.metrics.misses += 1;
                None
            }
        }
    }

    /// Remove a key from the cache (explicit invalidation).
    pub fn invalidate(&mut self, key: &K) {
        if self.entries.remove(key).is_some() {
            self.metrics.invalidations += 1;
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Number of entries currently in the cache (including potentially stale ones).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict all stale (TTL-expired) entries proactively.
    ///
    /// Call periodically to reclaim capacity without waiting for a miss.
    pub fn purge_stale(&mut self) {
        let ttl = self.ttl;
        self.entries.retain(|_, e| !e.is_stale(ttl));
    }

    /// Evict the single least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.access_seq)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&lru_key);
            self.metrics.evictions += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Farm-specific cache: JobStateCache
// ---------------------------------------------------------------------------

use crate::JobId;

/// A serialised job snapshot stored in the cache.
///
/// Using a `serde_json::Value` snapshot decouples the cache from the full
/// `Job` type (which is only available under the `sqlite` feature gate) and
/// allows the cache to be used in any configuration.
#[derive(Debug, Clone)]
pub struct JobSnapshot {
    /// Job identifier.
    pub job_id: JobId,
    /// JSON-serialised job state.
    pub data: serde_json::Value,
}

/// A thin LRU cache specialised for job state snapshots.
///
/// Sits in front of the SQLite persistence layer to serve repeated reads of
/// frequently accessed jobs without hitting the database on every call.
pub struct JobStateCache {
    inner: LruCache<JobId, JobSnapshot>,
}

impl JobStateCache {
    /// Create a cache with `capacity` slots and a `ttl` per entry.
    #[must_use]
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: LruCache::new(capacity, ttl),
        }
    }

    /// Store a job snapshot in the cache.
    pub fn put(&mut self, snapshot: JobSnapshot) {
        self.inner.insert(snapshot.job_id, snapshot);
    }

    /// Retrieve a job snapshot from the cache.
    ///
    /// Returns `None` on a miss or when the cached entry has expired.
    pub fn get(&mut self, job_id: &JobId) -> Option<JobSnapshot> {
        self.inner.get(job_id)
    }

    /// Invalidate the cache entry for `job_id` (e.g., after a state update).
    pub fn invalidate(&mut self, job_id: &JobId) {
        self.inner.invalidate(job_id);
    }

    /// Number of entries currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return `true` when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Accumulated cache metrics.
    #[must_use]
    pub fn metrics(&self) -> &CacheMetrics {
        &self.inner.metrics
    }

    /// Proactively evict stale entries.
    pub fn purge_stale(&mut self) {
        self.inner.purge_stale();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(cap: usize, ttl_secs: u64) -> LruCache<u32, String> {
        LruCache::new(cap, Duration::from_secs(ttl_secs))
    }

    // ── Basic get/insert ──────────────────────────────────────────────────────

    #[test]
    fn test_insert_and_get() {
        let mut c = make_cache(8, 60);
        c.insert(1, "hello".to_string());
        assert_eq!(c.get(&1).as_deref(), Some("hello"));
        assert_eq!(c.metrics.hits, 1);
        assert_eq!(c.metrics.misses, 0);
    }

    #[test]
    fn test_miss_on_absent_key() {
        let mut c = make_cache(8, 60);
        assert!(c.get(&99).is_none());
        assert_eq!(c.metrics.misses, 1);
    }

    // ── Capacity / LRU eviction ───────────────────────────────────────────────

    #[test]
    fn test_capacity_evicts_lru() {
        let mut c = make_cache(3, 3600);
        c.insert(1, "a".to_string());
        c.insert(2, "b".to_string());
        c.insert(3, "c".to_string());
        // Access 1 and 2 to make 3 the LRU.
        c.get(&1);
        c.get(&2);
        // Inserting 4 should evict key 3 (least recently used).
        c.insert(4, "d".to_string());
        assert_eq!(c.len(), 3);
        assert!(c.get(&3).is_none(), "key 3 should have been evicted");
        assert!(c.get(&4).is_some());
    }

    #[test]
    fn test_eviction_metric_incremented() {
        let mut c = make_cache(2, 3600);
        c.insert(1, "a".to_string());
        c.insert(2, "b".to_string());
        c.insert(3, "c".to_string()); // evicts one
        assert_eq!(c.metrics.evictions, 1);
    }

    // ── TTL expiry ────────────────────────────────────────────────────────────

    #[test]
    fn test_stale_entry_returns_none() {
        let mut c: LruCache<u32, String> = LruCache::new(8, Duration::from_millis(5));
        c.insert(1, "value".to_string());
        std::thread::sleep(Duration::from_millis(20));
        assert!(c.get(&1).is_none(), "entry should have expired");
        assert_eq!(c.metrics.misses, 1);
    }

    #[test]
    fn test_purge_stale_removes_expired() {
        let mut c: LruCache<u32, String> = LruCache::new(8, Duration::from_millis(5));
        c.insert(1, "old".to_string());
        c.insert(2, "fresh".to_string()); // will expire too, but we just check count
        std::thread::sleep(Duration::from_millis(20));
        c.purge_stale();
        assert_eq!(c.len(), 0);
    }

    // ── Invalidation ──────────────────────────────────────────────────────────

    #[test]
    fn test_invalidate_removes_entry() {
        let mut c = make_cache(8, 3600);
        c.insert(1, "v".to_string());
        c.invalidate(&1);
        assert!(c.get(&1).is_none());
        assert_eq!(c.metrics.invalidations, 1);
    }

    #[test]
    fn test_invalidate_nonexistent_key_no_panic() {
        let mut c = make_cache(8, 3600);
        c.invalidate(&999); // should not panic
        assert_eq!(c.metrics.invalidations, 0);
    }

    // ── Metrics ───────────────────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_zero_on_empty() {
        let c: LruCache<u32, String> = make_cache(8, 3600);
        assert_eq!(c.metrics.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_correct() {
        let mut c = make_cache(8, 3600);
        c.insert(1, "x".to_string());
        c.get(&1); // hit
        c.get(&2); // miss
        assert!((c.metrics.hit_rate() - 0.5).abs() < 1e-9);
    }

    // ── JobStateCache ─────────────────────────────────────────────────────────

    fn make_snapshot(job_id: crate::JobId) -> JobSnapshot {
        JobSnapshot {
            job_id,
            data: serde_json::json!({ "state": "Queued" }),
        }
    }

    #[test]
    fn test_job_state_cache_put_and_get() {
        let mut cache = JobStateCache::new(16, Duration::from_secs(60));
        let id = crate::JobId::new();
        cache.put(make_snapshot(id));
        let retrieved = cache.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("snapshot").job_id, id);
    }

    #[test]
    fn test_job_state_cache_invalidate() {
        let mut cache = JobStateCache::new(16, Duration::from_secs(60));
        let id = crate::JobId::new();
        cache.put(make_snapshot(id));
        cache.invalidate(&id);
        assert!(cache.get(&id).is_none());
        assert_eq!(cache.metrics().invalidations, 1);
    }

    #[test]
    fn test_clear_empties_cache() {
        let mut c = make_cache(8, 3600);
        c.insert(1, "a".to_string());
        c.insert(2, "b".to_string());
        c.clear();
        assert!(c.is_empty());
    }
}
