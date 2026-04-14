//! In-memory caching layer for frequently accessed job state.
//!
//! This module provides a fast, bounded LRU cache that sits in front of the
//! persistent `SQLite` store.  Hot job records are served from DRAM without a
//! database round-trip; evicted entries are transparently removed from the
//! cache (but not from the backing store).
//!
//! ## Design
//!
//! The cache is implemented as a combined `HashMap` + doubly-linked list
//! (a hand-rolled LRU list via `VecDeque` of keys in access order).  For
//! thread-safe use across coordinator tasks the cache is wrapped in
//! `Arc<parking_lot::Mutex<…>>` — see [`SharedJobStateCache`].
//!
//! ## Features
//!
//! - Bounded capacity with configurable eviction watermark
//! - TTL-based expiry: entries stale beyond `ttl_secs` are treated as misses
//! - Per-entry dirty flag to track unsynchronised writes
//! - Statistics: hit/miss/eviction counters with `AtomicU64`
//! - Batch invalidation by prefix or by state
//!
//! ## Example
//!
//! ```rust
//! use oximedia_farm::job_state_cache::{JobStateCache, CachedJobState};
//! use oximedia_farm::{JobId, JobState};
//!
//! let mut cache = JobStateCache::new(512, 300);
//! let id = JobId::new();
//! cache.insert(id, CachedJobState::new(JobState::Running, Some("worker-1")));
//! assert!(cache.get(&id).is_some());
//! ```

use crate::{JobId, JobState};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Cached entry
// ---------------------------------------------------------------------------

/// The state payload stored in the cache for a single job.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedJobState {
    /// Current logical state of the job.
    pub state: JobState,
    /// The worker ID currently processing this job, if any.
    pub assigned_worker: Option<String>,
    /// Human-readable status or error message.
    pub status_message: Option<String>,
    /// Completion percentage `[0.0, 100.0]`.
    pub percent_complete: f64,
    /// Whether this entry has been modified since the last write-back.
    #[serde(skip)]
    pub dirty: bool,
}

impl CachedJobState {
    /// Create a new cache entry in the given state.
    #[must_use]
    pub fn new(state: JobState, assigned_worker: Option<&str>) -> Self {
        Self {
            state,
            assigned_worker: assigned_worker.map(str::to_string),
            status_message: None,
            percent_complete: 0.0,
            dirty: false,
        }
    }

    /// Mark the entry as dirty (pending write-back).
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Clear the dirty flag (after a successful write-back).
    pub fn clear_dirty(&mut self) {
        self.dirty = false;
    }
}

// ---------------------------------------------------------------------------
// Internal cache line
// ---------------------------------------------------------------------------

struct CacheLine {
    value: CachedJobState,
    /// `Instant` at which this entry was inserted or refreshed.
    inserted_at: Instant,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Atomic counters for cache performance monitoring.
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Number of successful cache lookups.
    pub hits: AtomicU64,
    /// Number of failed cache lookups (miss or stale).
    pub misses: AtomicU64,
    /// Number of entries evicted due to capacity pressure.
    pub evictions: AtomicU64,
    /// Number of entries expired due to TTL.
    pub expirations: AtomicU64,
}

impl CacheStats {
    /// Return a point-in-time snapshot as plain numbers.
    #[must_use]
    pub fn snapshot(&self) -> CacheStatsSnapshot {
        CacheStatsSnapshot {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            expirations: self.expirations.load(Ordering::Relaxed),
        }
    }

    /// Compute the hit ratio `hits / (hits + misses)`.
    ///
    /// Returns `0.0` when no lookups have been performed.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let h = self.hits.load(Ordering::Relaxed) as f64;
        let m = self.misses.load(Ordering::Relaxed) as f64;
        let total = h + m;
        if total == 0.0 {
            0.0
        } else {
            h / total
        }
    }
}

/// A plain (non-atomic) snapshot of `CacheStats`.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CacheStatsSnapshot {
    /// Successful lookups.
    pub hits: u64,
    /// Failed lookups.
    pub misses: u64,
    /// Capacity-driven evictions.
    pub evictions: u64,
    /// TTL-driven expirations.
    pub expirations: u64,
}

// ---------------------------------------------------------------------------
// JobStateCache
// ---------------------------------------------------------------------------

/// A bounded, TTL-aware, LRU in-memory cache for job state.
///
/// This is **not** `Sync` on its own.  Use [`SharedJobStateCache`] for
/// concurrent access from multiple threads.
pub struct JobStateCache {
    /// Maximum number of entries before eviction kicks in.
    capacity: usize,
    /// Time-to-live per entry.
    ttl: Duration,
    /// Primary store: `JobId` string → cache line.
    store: HashMap<String, CacheLine>,
    /// LRU order: front = most recently used, back = least recently used.
    lru_order: VecDeque<String>,
    /// Performance counters.
    pub stats: Arc<CacheStats>,
}

impl JobStateCache {
    /// Create a cache with `capacity` entries and a TTL of `ttl_secs` seconds.
    ///
    /// `ttl_secs = 0` disables TTL expiry (entries live until evicted).
    #[must_use]
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        let cap = capacity.max(1);
        Self {
            capacity: cap,
            ttl: Duration::from_secs(ttl_secs),
            store: HashMap::with_capacity(cap),
            lru_order: VecDeque::with_capacity(cap),
            stats: Arc::new(CacheStats::default()),
        }
    }

    /// Return the configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the current number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Insert or replace a job state entry.
    ///
    /// If the cache is at capacity, the least-recently-used entry is evicted.
    pub fn insert(&mut self, job_id: JobId, state: CachedJobState) {
        let key = job_id.to_string();

        if self.store.contains_key(&key) {
            // Refresh existing entry: move to front of LRU list
            self.lru_order.retain(|k| k != &key);
        } else if self.store.len() >= self.capacity {
            // Evict LRU entry
            if let Some(evict_key) = self.lru_order.pop_back() {
                self.store.remove(&evict_key);
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            }
        }

        self.lru_order.push_front(key.clone());
        self.store.insert(
            key,
            CacheLine {
                value: state,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Look up a job state by ID.
    ///
    /// Returns `None` on a miss or if the entry has expired.  On a hit the
    /// entry is moved to the front of the LRU list.
    pub fn get(&mut self, job_id: &JobId) -> Option<&CachedJobState> {
        let key = job_id.to_string();

        // Check existence and TTL before promoting
        let expired = if let Some(line) = self.store.get(&key) {
            self.ttl.as_secs() > 0 && line.inserted_at.elapsed() > self.ttl
        } else {
            false
        };

        if expired {
            self.store.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.stats.expirations.fetch_add(1, Ordering::Relaxed);
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if self.store.contains_key(&key) {
            // Promote to front of LRU
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push_front(key.clone());
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            self.store.get(&key).map(|l| &l.value)
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Mutably access a cached entry.
    ///
    /// The entry is automatically marked dirty.  Returns `None` on miss or
    /// expiry.
    pub fn get_mut(&mut self, job_id: &JobId) -> Option<&mut CachedJobState> {
        let key = job_id.to_string();

        let expired = if let Some(line) = self.store.get(&key) {
            self.ttl.as_secs() > 0 && line.inserted_at.elapsed() > self.ttl
        } else {
            false
        };

        if expired {
            self.store.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.stats.expirations.fetch_add(1, Ordering::Relaxed);
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if self.store.contains_key(&key) {
            self.lru_order.retain(|k| k != &key);
            self.lru_order.push_front(key.clone());
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            self.store.get_mut(&key).map(|l| {
                l.value.mark_dirty();
                &mut l.value
            })
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Explicitly remove an entry.
    pub fn invalidate(&mut self, job_id: &JobId) {
        let key = job_id.to_string();
        self.store.remove(&key);
        self.lru_order.retain(|k| k != &key);
    }

    /// Remove all entries whose `JobState` matches `state`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_state(&mut self, state: JobState) -> usize {
        let to_remove: Vec<String> = self
            .store
            .iter()
            .filter(|(_, l)| matches_state(&l.value.state, state))
            .map(|(k, _)| k.clone())
            .collect();
        let count = to_remove.len();
        for key in to_remove {
            self.store.remove(&key);
            self.lru_order.retain(|k| k != &key);
        }
        count
    }

    /// Remove all entries whose job-ID string begins with `prefix`.
    ///
    /// Returns the number of entries removed.
    pub fn invalidate_by_prefix(&mut self, prefix: &str) -> usize {
        let to_remove: Vec<String> = self
            .store
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        let count = to_remove.len();
        for key in to_remove {
            self.store.remove(&key);
            self.lru_order.retain(|k| k != &key);
        }
        count
    }

    /// Collect all entries flagged as dirty (pending write-back).
    ///
    /// The returned list contains cloned snapshots; the dirty flags in the
    /// live cache are **not** cleared.
    #[must_use]
    pub fn dirty_entries(&self) -> Vec<(JobId, CachedJobState)> {
        self.store
            .iter()
            .filter(|(_, l)| l.value.dirty)
            .filter_map(|(k, l)| {
                k.parse::<uuid::Uuid>()
                    .ok()
                    .map(|uuid| (JobId::from(uuid), l.value.clone()))
            })
            .collect()
    }

    /// Clear the dirty flag on all entries (after a successful flush).
    pub fn mark_all_clean(&mut self) {
        for line in self.store.values_mut() {
            line.value.clear_dirty();
        }
    }

    /// Purge all expired entries proactively.
    ///
    /// Returns the number of entries purged.
    pub fn purge_expired(&mut self) -> usize {
        if self.ttl.as_secs() == 0 {
            return 0;
        }
        let expired_keys: Vec<String> = self
            .store
            .iter()
            .filter(|(_, l)| l.inserted_at.elapsed() > self.ttl)
            .map(|(k, _)| k.clone())
            .collect();
        let count = expired_keys.len();
        for key in expired_keys {
            self.store.remove(&key);
            self.lru_order.retain(|k| k != &key);
            self.stats.expirations.fetch_add(1, Ordering::Relaxed);
        }
        count
    }

    /// Return a reference to the statistics counters.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }
}

/// Helper: compare two `JobState` values by discriminant.
fn matches_state(a: &JobState, b: JobState) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(&b)
}

// ---------------------------------------------------------------------------
// Thread-safe wrapper
// ---------------------------------------------------------------------------

/// A thread-safe job state cache backed by a `parking_lot::Mutex`.
#[derive(Clone)]
pub struct SharedJobStateCache {
    inner: Arc<parking_lot::Mutex<JobStateCache>>,
}

impl SharedJobStateCache {
    /// Create a new shared cache.
    #[must_use]
    pub fn new(capacity: usize, ttl_secs: u64) -> Self {
        Self {
            inner: Arc::new(parking_lot::Mutex::new(JobStateCache::new(
                capacity, ttl_secs,
            ))),
        }
    }

    /// Insert or replace a job state entry.
    pub fn insert(&self, job_id: JobId, state: CachedJobState) {
        self.inner.lock().insert(job_id, state);
    }

    /// Look up a job state, returning a cloned snapshot.
    pub fn get(&self, job_id: &JobId) -> Option<CachedJobState> {
        self.inner.lock().get(job_id).cloned()
    }

    /// Invalidate a single entry.
    pub fn invalidate(&self, job_id: &JobId) {
        self.inner.lock().invalidate(job_id);
    }

    /// Return a statistics snapshot.
    #[must_use]
    pub fn stats_snapshot(&self) -> CacheStatsSnapshot {
        self.inner.lock().stats().snapshot()
    }

    /// Return the number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Return `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn running_state() -> CachedJobState {
        CachedJobState::new(JobState::Running, Some("worker-1"))
    }

    fn completed_state() -> CachedJobState {
        CachedJobState::new(JobState::Completed, None)
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = JobStateCache::new(32, 0);
        let id = JobId::new();
        cache.insert(id, running_state());
        assert!(cache.get(&id).is_some());
        let entry = cache.get(&id).expect("should be cached");
        assert!(matches!(entry.state, JobState::Running));
    }

    #[test]
    fn test_miss_returns_none() {
        let mut cache = JobStateCache::new(32, 0);
        let id = JobId::new();
        assert!(cache.get(&id).is_none());
    }

    #[test]
    fn test_lru_eviction_at_capacity() {
        let mut cache = JobStateCache::new(2, 0);
        let id1 = JobId::new();
        let id2 = JobId::new();
        let id3 = JobId::new();
        cache.insert(id1, running_state());
        cache.insert(id2, running_state());
        // id1 is the LRU entry — access id1 to make id2 the LRU
        let _ = cache.get(&id1);
        // Insert id3 → id2 should be evicted
        cache.insert(id3, running_state());
        assert!(cache.get(&id2).is_none(), "id2 should have been evicted");
        assert!(cache.get(&id1).is_some(), "id1 should still be present");
        assert!(cache.get(&id3).is_some(), "id3 should be present");
    }

    #[test]
    fn test_eviction_counter_increments() {
        let mut cache = JobStateCache::new(1, 0);
        let id1 = JobId::new();
        let id2 = JobId::new();
        cache.insert(id1, running_state());
        cache.insert(id2, running_state()); // evicts id1
        assert_eq!(cache.stats().evictions.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_ttl_expiry_on_get() {
        // TTL of 0 means no expiry — use a very short TTL simulation via direct
        // manipulation. We create a cache with 1s TTL and a fake inserted_at.
        let mut cache = JobStateCache::new(32, 1);
        let id = JobId::new();
        cache.insert(id, running_state());

        // Force the insertion time to be in the past by overwriting the entry
        // with a custom CacheLine that has an old timestamp.
        let key = id.to_string();
        if let Some(line) = cache.store.get_mut(&key) {
            // Subtract 2 seconds from inserted_at by replacing it
            line.inserted_at = Instant::now() - Duration::from_secs(2);
        }

        let result = cache.get(&id);
        assert!(result.is_none(), "entry should have expired");
        assert_eq!(cache.stats().expirations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_hit_and_miss_counters() {
        let mut cache = JobStateCache::new(32, 0);
        let id = JobId::new();
        cache.insert(id, running_state());
        let _ = cache.get(&id); // hit
        let _ = cache.get(&JobId::new()); // miss
        let snap = cache.stats().snapshot();
        assert_eq!(snap.hits, 1);
        assert_eq!(snap.misses, 1);
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = JobStateCache::new(32, 0);
        let id = JobId::new();
        cache.insert(id, running_state());
        cache.invalidate(&id);
        assert!(cache.get(&id).is_none());
    }

    #[test]
    fn test_invalidate_by_state() {
        let mut cache = JobStateCache::new(32, 0);
        let id1 = JobId::new();
        let id2 = JobId::new();
        cache.insert(id1, running_state());
        cache.insert(id2, completed_state());
        let removed = cache.invalidate_by_state(JobState::Completed);
        assert_eq!(removed, 1);
        assert!(cache.get(&id1).is_some(), "running entry should remain");
        assert!(cache.get(&id2).is_none(), "completed entry should be gone");
    }

    #[test]
    fn test_dirty_entries_tracking() {
        let mut cache = JobStateCache::new(32, 0);
        let id = JobId::new();
        cache.insert(id, running_state());
        let _ = cache.get_mut(&id); // marks dirty
        let dirty = cache.dirty_entries();
        assert_eq!(dirty.len(), 1);
        cache.mark_all_clean();
        let dirty_after = cache.dirty_entries();
        assert!(dirty_after.is_empty());
    }

    #[test]
    fn test_shared_cache_thread_safe() {
        let shared = SharedJobStateCache::new(64, 0);
        let id = JobId::new();
        shared.insert(id, running_state());
        assert!(shared.get(&id).is_some());
        assert_eq!(shared.len(), 1);
        shared.invalidate(&id);
        assert!(shared.is_empty());
    }

    #[test]
    fn test_hit_ratio_no_lookups() {
        let cache = JobStateCache::new(32, 0);
        assert!((cache.stats().hit_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_purge_expired_removes_stale() {
        let mut cache = JobStateCache::new(32, 1);
        let id1 = JobId::new();
        let id2 = JobId::new();
        cache.insert(id1, running_state());
        cache.insert(id2, running_state());

        // Age id1 artificially
        let key1 = id1.to_string();
        if let Some(line) = cache.store.get_mut(&key1) {
            line.inserted_at = Instant::now() - Duration::from_secs(2);
        }

        let purged = cache.purge_expired();
        assert_eq!(purged, 1);
        assert_eq!(cache.len(), 1);
    }
}
