//! Intermediate render output caching.
//!
//! [`RenderCache`] stores intermediate render outputs (lighting passes, AO
//! bakes, ray-traced GI) so that subsequent jobs sharing the same scene or
//! camera setup can skip expensive re-computation.
//!
//! Cache entries are keyed by a 64-bit content hash and are evicted according
//! to an LRU policy once the cache reaches its configured byte capacity.

#![allow(dead_code)]

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// CacheKey / CacheEntry
// ---------------------------------------------------------------------------

/// 64-bit content-addressable key for a render cache entry.
pub type CacheKey = u64;

/// A single cached render output.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Stable key derived from scene hash + pass type.
    pub key: CacheKey,
    /// Human-readable label (e.g. "lighting_pass_frame_0042").
    pub label: String,
    /// Raw pixel data or encoded intermediate buffer.
    pub data: Vec<u8>,
    /// Monotonically increasing access counter (for LRU ordering).
    pub last_accessed: u64,
}

impl CacheEntry {
    /// Creates a new cache entry.
    #[must_use]
    pub fn new(key: CacheKey, label: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            key,
            label: label.into(),
            data,
            last_accessed: 0,
        }
    }

    /// Returns the byte size of the stored data.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// RenderCache
// ---------------------------------------------------------------------------

/// LRU-evicting cache for intermediate render outputs.
///
/// When the total byte usage would exceed `max_bytes`, the least recently
/// accessed entry is evicted before the new one is inserted.
pub struct RenderCache {
    entries: HashMap<CacheKey, CacheEntry>,
    max_bytes: usize,
    current_bytes: usize,
    /// Monotonically increasing clock for LRU tracking.
    clock: u64,
    /// Total entries inserted (including evicted ones).
    insert_count: u64,
    /// Total evictions performed.
    eviction_count: u64,
}

impl RenderCache {
    /// Creates a new cache with a given byte capacity.
    ///
    /// If `max_bytes` is 0 the cache accepts no entries.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_bytes,
            current_bytes: 0,
            clock: 0,
            insert_count: 0,
            eviction_count: 0,
        }
    }

    /// Inserts or updates an entry.
    ///
    /// If the data alone exceeds `max_bytes`, the entry is silently dropped
    /// (the cache cannot hold it regardless of evictions).
    ///
    /// Returns `true` when the entry was successfully stored.
    pub fn insert(&mut self, key: CacheKey, label: impl Into<String>, data: Vec<u8>) -> bool {
        let byte_size = data.len();
        if byte_size > self.max_bytes {
            return false;
        }

        // Remove existing entry with the same key first
        if let Some(old) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(old.byte_size());
        }

        // Evict LRU entries until there is space
        while !self.entries.is_empty() && self.current_bytes + byte_size > self.max_bytes {
            self.evict_lru();
        }

        self.clock += 1;
        let mut entry = CacheEntry::new(key, label, data);
        entry.last_accessed = self.clock;
        self.current_bytes += byte_size;
        self.entries.insert(key, entry);
        self.insert_count += 1;
        true
    }

    /// Retrieves an entry by key, updating its LRU timestamp.
    ///
    /// Returns `None` when the key is not present.
    pub fn get(&mut self, key: CacheKey) -> Option<&CacheEntry> {
        self.clock += 1;
        let clock = self.clock;
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_accessed = clock;
            Some(&*entry)
        } else {
            None
        }
    }

    /// Removes an entry from the cache, returning it if present.
    pub fn invalidate(&mut self, key: CacheKey) -> Option<CacheEntry> {
        if let Some(entry) = self.entries.remove(&key) {
            self.current_bytes = self.current_bytes.saturating_sub(entry.byte_size());
            Some(entry)
        } else {
            None
        }
    }

    /// Removes all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_bytes = 0;
    }

    /// Returns `true` when the cache contains an entry for `key`.
    #[must_use]
    pub fn contains(&self, key: CacheKey) -> bool {
        self.entries.contains_key(&key)
    }

    /// Number of entries currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current total byte usage.
    #[must_use]
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Configured maximum byte capacity.
    #[must_use]
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Total entries inserted since creation (including evicted ones).
    #[must_use]
    pub fn insert_count(&self) -> u64 {
        self.insert_count
    }

    /// Total LRU evictions performed since creation.
    #[must_use]
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn evict_lru(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(&k, _)| k);

        if let Some(key) = lru_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_bytes = self.current_bytes.saturating_sub(entry.byte_size());
                self.eviction_count += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut cache = RenderCache::new(1024);
        assert!(cache.insert(1, "pass_a", vec![0u8; 100]));
        assert!(cache.contains(1));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_get_missing_returns_none() {
        let mut cache = RenderCache::new(1024);
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn test_eviction_on_overflow() {
        let mut cache = RenderCache::new(200);
        cache.insert(1, "a", vec![0u8; 100]);
        cache.insert(2, "b", vec![0u8; 100]);
        // No room for a 3rd 100-byte entry without eviction
        cache.insert(3, "c", vec![0u8; 100]);
        assert_eq!(cache.len(), 2, "one entry should have been evicted");
        assert!(cache.eviction_count() >= 1);
    }

    #[test]
    fn test_oversized_entry_not_inserted() {
        let mut cache = RenderCache::new(50);
        let inserted = cache.insert(1, "big", vec![0u8; 100]);
        assert!(!inserted, "entry larger than max_bytes should be rejected");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = RenderCache::new(1024);
        cache.insert(5, "x", vec![1u8; 64]);
        assert!(cache.contains(5));
        let removed = cache.invalidate(5);
        assert!(removed.is_some());
        assert!(!cache.contains(5));
    }

    #[test]
    fn test_clear() {
        let mut cache = RenderCache::new(1024);
        cache.insert(1, "a", vec![0u8; 64]);
        cache.insert(2, "b", vec![0u8; 64]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn test_byte_accounting() {
        let mut cache = RenderCache::new(4096);
        cache.insert(1, "p", vec![0u8; 300]);
        cache.insert(2, "q", vec![0u8; 500]);
        assert_eq!(cache.current_bytes(), 800);
    }

    #[test]
    fn test_lru_evicts_least_recently_used() {
        // Cache fits exactly 2 entries of 100 bytes each (capacity 200)
        let mut cache = RenderCache::new(200);
        cache.insert(1, "old", vec![0u8; 100]);
        cache.insert(2, "new", vec![0u8; 100]);
        // Access key 1 to make it more recent than key 2
        let _ = cache.get(1);
        // Now insert a 3rd entry; key 2 should be evicted (it's the LRU)
        cache.insert(3, "third", vec![0u8; 100]);
        assert!(cache.contains(1), "key 1 should still be present");
        assert!(cache.contains(3), "key 3 should be present");
        assert!(!cache.contains(2), "key 2 (LRU) should have been evicted");
    }
}
