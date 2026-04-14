//! Write-through cache that synchronously persists every write to a backing store.
//!
//! Unlike write-*behind* (write-back) caching — where dirty entries are
//! flushed lazily — a write-through cache propagates every [`put`] to the
//! backing store **before** returning to the caller.  This guarantees that
//! the backing store is always consistent with the cache at the cost of
//! write latency.
//!
//! # Read path (write-allocate)
//!
//! On a cache miss, [`get`] falls back to the backing store.  If the store
//! returns a value, the entry is inserted into the cache (write-allocate) so
//! subsequent reads are served locally.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::write_through::{WriteThroughCache, InMemoryStore};
//!
//! let store: InMemoryStore<String, Vec<u8>> = InMemoryStore::new();
//! let mut cache = WriteThroughCache::new(32, store);
//!
//! cache.put("key".to_string(), vec![1, 2, 3]).expect("put failed");
//! assert!(cache.get(&"key".to_string()).is_some());
//! ```
//!
//! [`put`]: WriteThroughCache::put
//! [`get`]: WriteThroughCache::get

use std::collections::HashMap;
use std::fmt;

// ── StoreError ────────────────────────────────────────────────────────────────

/// Errors that can occur during backing-store operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// The backing store has no remaining capacity.
    StoreFull,
    /// The requested key does not exist in the store.
    KeyNotFound,
    /// A generic write failure with a descriptive message.
    WriteFailure(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::StoreFull => write!(f, "backing store is full"),
            StoreError::KeyNotFound => write!(f, "key not found in backing store"),
            StoreError::WriteFailure(msg) => write!(f, "write failure: {msg}"),
        }
    }
}

impl std::error::Error for StoreError {}

// ── BackingStore trait ────────────────────────────────────────────────────────

/// Abstraction over a synchronous persistent store.
///
/// Implementations are expected to be synchronous.  For async I/O the caller
/// can provide a blocking adapter.
pub trait BackingStore<K, V> {
    /// Persist `(key, value)` to the backing store.
    fn store(&mut self, key: &K, value: &V) -> Result<(), StoreError>;

    /// Load the value associated with `key`, returning `None` if absent.
    fn load(&self, key: &K) -> Option<V>;

    /// Remove `key` from the backing store.
    ///
    /// Returns `true` if the key existed and was removed.
    fn remove(&mut self, key: &K) -> bool;
}

// ── InMemoryStore ─────────────────────────────────────────────────────────────

/// A simple in-memory [`BackingStore`] backed by a `HashMap`.
///
/// Useful for testing and as a reference implementation.
#[derive(Debug, Clone, Default)]
pub struct InMemoryStore<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    data: HashMap<K, V>,
    /// Optional hard limit on number of entries (0 = unlimited).
    max_entries: usize,
    /// Number of times `store` was called.
    write_count: u64,
    /// Number of times `load` was called.
    read_count: u64,
}

impl<K, V> InMemoryStore<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    /// Create an unbounded in-memory store.
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            max_entries: 0,
            write_count: 0,
            read_count: 0,
        }
    }

    /// Create an in-memory store with a hard capacity limit.
    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            data: HashMap::with_capacity(max_entries),
            max_entries,
            write_count: 0,
            read_count: 0,
        }
    }

    /// Number of entries currently in the store.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Total number of write operations performed.
    pub fn write_count(&self) -> u64 {
        self.write_count
    }

    /// Total number of read operations performed.
    pub fn read_count(&self) -> u64 {
        self.read_count
    }
}

impl<K, V> BackingStore<K, V> for InMemoryStore<K, V>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
{
    fn store(&mut self, key: &K, value: &V) -> Result<(), StoreError> {
        self.write_count += 1;
        if self.max_entries > 0
            && self.data.len() >= self.max_entries
            && !self.data.contains_key(key)
        {
            return Err(StoreError::StoreFull);
        }
        self.data.insert(key.clone(), value.clone());
        Ok(())
    }

    fn load(&self, key: &K) -> Option<V> {
        self.data.get(key).cloned()
    }

    fn remove(&mut self, key: &K) -> bool {
        self.data.remove(key).is_some()
    }
}

// ── WriteThroughStats ─────────────────────────────────────────────────────────

/// Snapshot of [`WriteThroughCache`] statistics.
#[derive(Debug, Clone, Default)]
pub struct WriteThroughStats {
    /// Number of reads served directly from the in-process cache.
    pub cache_hits: u64,
    /// Number of reads served from the backing store (write-allocate path).
    pub backing_store_hits: u64,
    /// Number of reads that found no value in either cache or backing store.
    pub misses: u64,
    /// Total number of successful [`put`] operations (cache + store written).
    ///
    /// [`put`]: WriteThroughCache::put
    pub writes: u64,
    /// Number of [`put`] operations that failed due to a backing-store error.
    pub write_errors: u64,
}

// ── WriteThroughCache ─────────────────────────────────────────────────────────

/// Write-through cache that synchronously persists every write.
///
/// # Type parameters
/// * `K` – key type; must implement `Eq + Hash + Clone`.
/// * `V` – value type; must implement `Clone`.
/// * `S` – backing store; must implement [`BackingStore<K, V>`].
pub struct WriteThroughCache<K, V, S>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
    S: BackingStore<K, V>,
{
    capacity: usize,
    cache: HashMap<K, V>,
    /// Insertion order for FIFO eviction.
    insertion_order: Vec<K>,
    store: S,
    stats: WriteThroughStats,
    /// Phantom data to tie K and V to the struct without owning them.
    _marker: std::marker::PhantomData<(K, V)>,
}

impl<K, V, S> WriteThroughCache<K, V, S>
where
    K: Eq + std::hash::Hash + Clone,
    V: Clone,
    S: BackingStore<K, V>,
{
    /// Create a new `WriteThroughCache` with the given `capacity` and backing
    /// `store`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    pub fn new(capacity: usize, store: S) -> Self {
        assert!(capacity > 0, "WriteThroughCache capacity must be non-zero");
        Self {
            capacity,
            cache: HashMap::with_capacity(capacity.min(1024)),
            insertion_order: Vec::with_capacity(capacity.min(1024)),
            store,
            stats: WriteThroughStats::default(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Write `(key, value)` to both the in-process cache and the backing store.
    ///
    /// If the write to the backing store fails, the cache is **not** updated and
    /// an error is returned — ensuring both are consistent.
    ///
    /// When the cache has reached `capacity`, the oldest entry is evicted
    /// (from the cache only; the backing store is unaffected).
    pub fn put(&mut self, key: K, value: V) -> Result<(), StoreError> {
        // Attempt backing-store write first so that failure is visible before
        // we mutate the cache.
        if let Err(e) = self.store.store(&key, &value) {
            self.stats.write_errors += 1;
            return Err(e);
        }

        // Evict oldest if needed (and key is new).
        if !self.cache.contains_key(&key) && self.cache.len() >= self.capacity {
            self.evict_oldest();
        } else if self.cache.contains_key(&key) {
            // Update in place; refresh insertion-order position.
            self.insertion_order.retain(|k| k != &key);
        }

        self.cache.insert(key.clone(), value);
        self.insertion_order.push(key);
        self.stats.writes += 1;
        Ok(())
    }

    /// Retrieve the value for `key`.
    ///
    /// Returns a reference to the cached value on a cache hit.  On a cache
    /// miss, falls back to the backing store; if the store has the value it is
    /// inserted into the cache (write-allocate) and a reference is returned.
    ///
    /// Returns `None` only when neither cache nor store contains the key.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        if self.cache.contains_key(key) {
            self.stats.cache_hits += 1;
            return self.cache.get(key);
        }

        // Cache miss — try backing store.
        match self.store.load(key) {
            Some(value) => {
                self.stats.backing_store_hits += 1;
                // Write-allocate: populate the cache.
                if self.cache.len() >= self.capacity {
                    self.evict_oldest();
                }
                self.cache.insert(key.clone(), value);
                self.insertion_order.push(key.clone());
                self.cache.get(key)
            }
            None => {
                self.stats.misses += 1;
                None
            }
        }
    }

    /// Invalidate `key` from both the cache and the backing store.
    ///
    /// Returns `true` if the key was present in either location.
    pub fn invalidate(&mut self, key: &K) -> bool {
        let in_cache = self.cache.remove(key).is_some();
        if in_cache {
            self.insertion_order.retain(|k| k != key);
        }
        let in_store = self.store.remove(key);
        in_cache || in_store
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> WriteThroughStats {
        self.stats.clone()
    }

    /// Number of entries currently in the in-process cache.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Configured cache capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Shared reference to the backing store.
    pub fn backing_store(&self) -> &S {
        &self.store
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn evict_oldest(&mut self) {
        if self.insertion_order.is_empty() {
            return;
        }
        let oldest = self.insertion_order.remove(0);
        self.cache.remove(&oldest);
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(cap: usize) -> WriteThroughCache<String, Vec<u8>, InMemoryStore<String, Vec<u8>>> {
        let store = InMemoryStore::new();
        WriteThroughCache::new(cap, store)
    }

    #[test]
    fn test_put_and_get_from_cache() {
        let mut cache = make_cache(10);
        cache.put("key1".to_string(), vec![1, 2, 3]).expect("put");
        let v = cache.get(&"key1".to_string());
        assert_eq!(v, Some(&vec![1u8, 2, 3]));
        assert_eq!(cache.stats().cache_hits, 1);
    }

    #[test]
    fn test_put_persists_to_backing_store() {
        let mut cache = make_cache(10);
        cache.put("k".to_string(), vec![9]).expect("put");
        // Peek into backing store directly.
        let loaded = cache.backing_store().load(&"k".to_string());
        assert_eq!(loaded, Some(vec![9u8]));
    }

    #[test]
    fn test_get_fallback_to_backing_store() {
        let store = InMemoryStore::new();
        let mut cache: WriteThroughCache<String, u32, InMemoryStore<String, u32>> =
            WriteThroughCache::new(10, store);

        // Manually pre-populate the backing store via a put, then clear the
        // in-process cache by creating a fresh cache wrapping the store data.
        // Simulate backing-store hit by putting a value and then evicting it
        // from cache by filling with other entries.
        cache.put("target".to_string(), 42u32).expect("put");

        // Fill cache beyond capacity so "target" gets evicted (cap=2 here).
        let store2 = InMemoryStore::new();
        let mut cache2: WriteThroughCache<String, u32, InMemoryStore<String, u32>> =
            WriteThroughCache::new(2, store2);
        cache2.put("target".to_string(), 42).expect("put");
        cache2.put("a".to_string(), 1).expect("put");
        cache2.put("b".to_string(), 2).expect("put"); // evicts "target" from cache

        // "target" should now be served from backing store.
        let v = cache2.get(&"target".to_string());
        assert_eq!(v, Some(&42));
        assert_eq!(cache2.stats().backing_store_hits, 1);
    }

    #[test]
    fn test_invalidate_removes_from_both() {
        let mut cache = make_cache(10);
        cache.put("x".to_string(), vec![0]).expect("put");
        let removed = cache.invalidate(&"x".to_string());
        assert!(removed);
        assert_eq!(cache.get(&"x".to_string()), None);
        assert_eq!(cache.backing_store().load(&"x".to_string()), None);
    }

    #[test]
    fn test_invalidate_absent_key_returns_false() {
        let mut cache = make_cache(10);
        assert!(!cache.invalidate(&"ghost".to_string()));
    }

    #[test]
    fn test_stats_writes_and_errors() {
        let store = InMemoryStore::<String, u32>::with_capacity(1);
        let mut cache: WriteThroughCache<String, u32, InMemoryStore<String, u32>> =
            WriteThroughCache::new(10, store);

        cache.put("first".to_string(), 1).expect("first put ok");
        // second distinct key overflows the store capacity of 1.
        let result = cache.put("second".to_string(), 2);
        assert_eq!(result, Err(StoreError::StoreFull));

        let s = cache.stats();
        assert_eq!(s.writes, 1);
        assert_eq!(s.write_errors, 1);
    }

    #[test]
    fn test_miss_when_not_in_either() {
        let mut cache = make_cache(10);
        assert_eq!(cache.get(&"nonexistent".to_string()), None);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_capacity_eviction() {
        let mut cache = make_cache(3);
        cache.put("a".to_string(), vec![1]).expect("put");
        cache.put("b".to_string(), vec![2]).expect("put");
        cache.put("c".to_string(), vec![3]).expect("put");
        // "a" is oldest, will be evicted from in-process cache (not store).
        cache.put("d".to_string(), vec![4]).expect("put");
        // "a" should still be retrievable from the backing store.
        assert_eq!(cache.stats().cache_hits, 0);
        let v = cache.get(&"a".to_string());
        assert_eq!(v, Some(&vec![1u8]));
        assert_eq!(cache.stats().backing_store_hits, 1);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let mut cache = make_cache(5);
        cache.put("k".to_string(), vec![1]).expect("put");
        cache.put("k".to_string(), vec![2]).expect("put"); // overwrite
        assert_eq!(cache.get(&"k".to_string()), Some(&vec![2u8]));
        assert_eq!(cache.cache_len(), 1);
    }

    #[test]
    fn test_write_failure_does_not_pollute_cache() {
        // capacity(1) means the store accepts exactly 1 entry; the 2nd distinct key fails.
        let store = InMemoryStore::<String, i32>::with_capacity(1);
        let mut cache: WriteThroughCache<String, i32, InMemoryStore<String, i32>> =
            WriteThroughCache::new(10, store);
        // First write succeeds.
        cache.put("key1".to_string(), 1).expect("first ok");
        // Second distinct key overflows the store's capacity of 1.
        let result = cache.put("key2".to_string(), 2);
        assert!(result.is_err(), "second put should fail");
        // Cache must NOT contain "key2" because the backing-store write failed.
        assert_eq!(cache.get(&"key2".to_string()), None);
    }
}
