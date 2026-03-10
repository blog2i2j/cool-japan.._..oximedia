#![allow(dead_code)]
//! Caching system for graph node outputs.
//!
//! This module provides a caching layer for intermediate results produced
//! by graph nodes. This avoids redundant computation when multiple
//! downstream nodes consume the same data or when re-evaluating a graph
//! with unchanged inputs.

use std::collections::HashMap;

/// Unique key for a cached result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Node identifier.
    pub node_id: u64,
    /// Output port index.
    pub port: u32,
    /// Generation/version counter for invalidation.
    pub generation: u64,
}

impl CacheKey {
    /// Create a new cache key.
    pub fn new(node_id: u64, port: u32, generation: u64) -> Self {
        Self {
            node_id,
            port,
            generation,
        }
    }
}

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}@{}", self.node_id, self.port, self.generation)
    }
}

/// A cached data entry.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached data as raw bytes.
    pub data: Vec<u8>,
    /// Size in bytes.
    pub size: usize,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Creation timestamp (monotonic counter).
    pub created_at: u64,
    /// Last access timestamp (monotonic counter).
    pub last_accessed: u64,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(data: Vec<u8>, timestamp: u64) -> Self {
        let size = data.len();
        Self {
            data,
            size,
            access_count: 0,
            created_at: timestamp,
            last_accessed: timestamp,
        }
    }

    /// Record an access to this entry.
    pub fn touch(&mut self, timestamp: u64) {
        self.access_count += 1;
        self.last_accessed = timestamp;
    }
}

/// Eviction policy for the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used.
    Lru,
    /// Least Frequently Used.
    Lfu,
    /// First In, First Out.
    Fifo,
}

/// Cache statistics.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CacheStatistics {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Number of evictions.
    pub evictions: u64,
    /// Number of explicit invalidations.
    pub invalidations: u64,
    /// Current number of entries.
    pub entry_count: usize,
    /// Current total memory usage in bytes.
    pub memory_bytes: usize,
}

impl CacheStatistics {
    /// Compute hit rate as a ratio (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }
}

/// A cache for graph node outputs.
pub struct NodeCache {
    /// Stored entries.
    entries: HashMap<CacheKey, CacheEntry>,
    /// Maximum memory budget in bytes.
    max_memory: usize,
    /// Current memory usage.
    current_memory: usize,
    /// Eviction policy.
    policy: EvictionPolicy,
    /// Monotonic clock for timestamps.
    clock: u64,
    /// Statistics.
    stats: CacheStatistics,
}

impl NodeCache {
    /// Create a new node cache with a given memory budget and eviction policy.
    pub fn new(max_memory: usize, policy: EvictionPolicy) -> Self {
        Self {
            entries: HashMap::new(),
            max_memory,
            current_memory: 0,
            policy,
            clock: 0,
            stats: CacheStatistics::default(),
        }
    }

    /// Create a cache with LRU policy.
    pub fn lru(max_memory: usize) -> Self {
        Self::new(max_memory, EvictionPolicy::Lru)
    }

    /// Insert data into the cache.
    ///
    /// May evict existing entries if the memory budget is exceeded.
    pub fn insert(&mut self, key: CacheKey, data: Vec<u8>) {
        let entry_size = data.len();
        // Evict if necessary
        while self.current_memory + entry_size > self.max_memory && !self.entries.is_empty() {
            self.evict_one();
        }
        // If single entry exceeds budget, do not insert
        if entry_size > self.max_memory {
            return;
        }
        self.clock += 1;
        let entry = CacheEntry::new(data, self.clock);
        self.current_memory += entry_size;
        // If replacing existing entry, subtract old size
        if let Some(old) = self.entries.insert(key, entry) {
            self.current_memory -= old.size;
        }
        self.stats.entry_count = self.entries.len();
        self.stats.memory_bytes = self.current_memory;
    }

    /// Get a reference to cached data.
    pub fn get(&mut self, key: &CacheKey) -> Option<&[u8]> {
        self.clock += 1;
        let ts = self.clock;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch(ts);
            self.stats.hits += 1;
            Some(&entry.data)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Check if a key is present without updating access stats.
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove a specific key from the cache.
    pub fn invalidate(&mut self, key: &CacheKey) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.current_memory -= entry.size;
            self.stats.invalidations += 1;
            self.stats.entry_count = self.entries.len();
            self.stats.memory_bytes = self.current_memory;
            true
        } else {
            false
        }
    }

    /// Invalidate all entries for a given node.
    pub fn invalidate_node(&mut self, node_id: u64) {
        let keys_to_remove: Vec<CacheKey> = self
            .entries
            .keys()
            .filter(|k| k.node_id == node_id)
            .cloned()
            .collect();
        for key in keys_to_remove {
            self.invalidate(&key);
        }
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_memory = 0;
        self.stats.entry_count = 0;
        self.stats.memory_bytes = 0;
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current memory usage.
    pub fn memory_usage(&self) -> usize {
        self.current_memory
    }

    /// Get cache statistics.
    pub fn statistics(&self) -> &CacheStatistics {
        &self.stats
    }

    /// Evict a single entry according to the eviction policy.
    fn evict_one(&mut self) {
        let victim = match self.policy {
            EvictionPolicy::Lru => self.find_lru(),
            EvictionPolicy::Lfu => self.find_lfu(),
            EvictionPolicy::Fifo => self.find_fifo(),
        };
        if let Some(key) = victim {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_memory -= entry.size;
                self.stats.evictions += 1;
                self.stats.entry_count = self.entries.len();
                self.stats.memory_bytes = self.current_memory;
            }
        }
    }

    /// Find the least recently used entry.
    fn find_lru(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.last_accessed)
            .map(|(k, _)| k.clone())
    }

    /// Find the least frequently used entry.
    fn find_lfu(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.access_count)
            .map(|(k, _)| k.clone())
    }

    /// Find the oldest entry (FIFO).
    fn find_fifo(&self) -> Option<CacheKey> {
        self.entries
            .iter()
            .min_by_key(|(_, e)| e.created_at)
            .map(|(k, _)| k.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_display() {
        let key = CacheKey::new(42, 0, 7);
        assert_eq!(format!("{key}"), "42:0@7");
    }

    #[test]
    fn test_cache_insert_and_get() {
        let mut cache = NodeCache::lru(1024);
        let key = CacheKey::new(1, 0, 1);
        cache.insert(key.clone(), vec![1, 2, 3, 4]);
        let data = cache.get(&key).expect("get should succeed");
        assert_eq!(data, &[1, 2, 3, 4]);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = NodeCache::lru(1024);
        let key = CacheKey::new(1, 0, 1);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.statistics().misses, 1);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = NodeCache::lru(1024);
        let key = CacheKey::new(1, 0, 1);
        cache.insert(key.clone(), vec![1, 2, 3]);
        cache.get(&key); // hit
        cache.get(&CacheKey::new(2, 0, 1)); // miss
        let rate = cache.statistics().hit_rate();
        assert!((rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_eviction_lru() {
        let mut cache = NodeCache::lru(10); // 10 bytes max
        cache.insert(CacheKey::new(1, 0, 1), vec![0; 5]);
        cache.insert(CacheKey::new(2, 0, 1), vec![0; 5]);
        // Cache is full (10 bytes)
        assert_eq!(cache.len(), 2);
        // Insert another, should evict LRU (key 1)
        cache.insert(CacheKey::new(3, 0, 1), vec![0; 5]);
        assert_eq!(cache.len(), 2);
        assert!(!cache.contains(&CacheKey::new(1, 0, 1)));
        assert!(cache.contains(&CacheKey::new(3, 0, 1)));
    }

    #[test]
    fn test_cache_eviction_fifo() {
        let mut cache = NodeCache::new(10, EvictionPolicy::Fifo);
        cache.insert(CacheKey::new(1, 0, 1), vec![0; 5]);
        cache.insert(CacheKey::new(2, 0, 1), vec![0; 5]);
        cache.insert(CacheKey::new(3, 0, 1), vec![0; 5]);
        // Should have evicted the first inserted
        assert!(!cache.contains(&CacheKey::new(1, 0, 1)));
    }

    #[test]
    fn test_cache_eviction_lfu() {
        let mut cache = NodeCache::new(15, EvictionPolicy::Lfu);
        cache.insert(CacheKey::new(1, 0, 1), vec![0; 5]);
        cache.insert(CacheKey::new(2, 0, 1), vec![0; 5]);
        cache.insert(CacheKey::new(3, 0, 1), vec![0; 5]);
        // Access key 1 and 3 multiple times
        cache.get(&CacheKey::new(1, 0, 1));
        cache.get(&CacheKey::new(1, 0, 1));
        cache.get(&CacheKey::new(3, 0, 1));
        // Now insert something that forces eviction of LFU (key 2)
        cache.insert(CacheKey::new(4, 0, 1), vec![0; 5]);
        assert!(!cache.contains(&CacheKey::new(2, 0, 1)));
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = NodeCache::lru(1024);
        let key = CacheKey::new(1, 0, 1);
        cache.insert(key.clone(), vec![1, 2, 3]);
        assert!(cache.invalidate(&key));
        assert!(!cache.contains(&key));
        assert_eq!(cache.statistics().invalidations, 1);
    }

    #[test]
    fn test_cache_invalidate_node() {
        let mut cache = NodeCache::lru(1024);
        cache.insert(CacheKey::new(1, 0, 1), vec![1]);
        cache.insert(CacheKey::new(1, 1, 1), vec![2]);
        cache.insert(CacheKey::new(2, 0, 1), vec![3]);
        cache.invalidate_node(1);
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&CacheKey::new(2, 0, 1)));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = NodeCache::lru(1024);
        cache.insert(CacheKey::new(1, 0, 1), vec![1, 2]);
        cache.insert(CacheKey::new(2, 0, 1), vec![3, 4]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.memory_usage(), 0);
    }

    #[test]
    fn test_cache_memory_tracking() {
        let mut cache = NodeCache::lru(1024);
        cache.insert(CacheKey::new(1, 0, 1), vec![0; 100]);
        assert_eq!(cache.memory_usage(), 100);
        cache.insert(CacheKey::new(2, 0, 1), vec![0; 200]);
        assert_eq!(cache.memory_usage(), 300);
        cache.invalidate(&CacheKey::new(1, 0, 1));
        assert_eq!(cache.memory_usage(), 200);
    }

    #[test]
    fn test_cache_oversize_entry_not_inserted() {
        let mut cache = NodeCache::lru(10);
        cache.insert(CacheKey::new(1, 0, 1), vec![0; 20]);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_statistics_default() {
        let stats = CacheStatistics::default();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_entry_touch() {
        let mut entry = CacheEntry::new(vec![1, 2, 3], 10);
        assert_eq!(entry.access_count, 0);
        entry.touch(20);
        assert_eq!(entry.access_count, 1);
        assert_eq!(entry.last_accessed, 20);
    }
}
