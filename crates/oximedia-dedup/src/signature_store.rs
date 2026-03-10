#![allow(dead_code)]
//! In-memory signature storage with lookup and expiration.
//!
//! Provides a time-aware store for media content signatures (hashes,
//! fingerprints) with automatic expiration and efficient lookup,
//! suitable for caching deduplication results.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default time-to-live for signature entries (1 hour).
const DEFAULT_TTL_SECS: u64 = 3600;

/// Maximum number of entries before forced eviction.
const DEFAULT_MAX_ENTRIES: usize = 100_000;

/// A stored signature entry with metadata.
#[derive(Debug, Clone)]
pub struct SignatureEntry {
    /// The signature bytes.
    pub signature: Vec<u8>,
    /// File size of the source.
    pub file_size: u64,
    /// Timestamp when the entry was created.
    created_at: Instant,
    /// Timestamp of the last access.
    last_accessed: Instant,
    /// Number of times this entry was accessed.
    access_count: u64,
    /// Time-to-live for this entry.
    ttl: Duration,
}

impl SignatureEntry {
    /// Create a new signature entry.
    pub fn new(signature: Vec<u8>, file_size: u64, ttl: Duration) -> Self {
        let now = Instant::now();
        Self {
            signature,
            file_size,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl,
        }
    }

    /// Check if this entry has expired.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    /// Get the age of this entry.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get the time since last access.
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }

    /// Get the access count.
    pub fn access_count(&self) -> u64 {
        self.access_count
    }

    /// Record an access.
    pub fn record_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }

    /// Get the signature bytes.
    pub fn signature(&self) -> &[u8] {
        &self.signature
    }
}

/// Configuration for the signature store.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// Default time-to-live for entries.
    pub default_ttl: Duration,
    /// Maximum number of entries.
    pub max_entries: usize,
    /// Whether to use LRU eviction when full.
    pub lru_eviction: bool,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(DEFAULT_TTL_SECS),
            max_entries: DEFAULT_MAX_ENTRIES,
            lru_eviction: true,
        }
    }
}

impl StoreConfig {
    /// Set the default TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = ttl;
        self
    }

    /// Set the max entries.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max.max(1);
        self
    }

    /// Enable or disable LRU eviction.
    pub fn with_lru_eviction(mut self, enabled: bool) -> Self {
        self.lru_eviction = enabled;
        self
    }
}

/// In-memory signature store.
#[derive(Debug)]
pub struct SignatureStore {
    /// Stored entries keyed by an identifier.
    entries: HashMap<String, SignatureEntry>,
    /// Configuration.
    config: StoreConfig,
    /// Total number of inserts.
    total_inserts: u64,
    /// Total number of lookups.
    total_lookups: u64,
    /// Total number of hits.
    total_hits: u64,
    /// Total number of evictions.
    total_evictions: u64,
}

impl SignatureStore {
    /// Create a new signature store with default configuration.
    pub fn new() -> Self {
        Self::with_config(StoreConfig::default())
    }

    /// Create a new signature store with custom configuration.
    pub fn with_config(config: StoreConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            total_inserts: 0,
            total_lookups: 0,
            total_hits: 0,
            total_evictions: 0,
        }
    }

    /// Insert a signature into the store.
    pub fn insert(&mut self, key: String, signature: Vec<u8>, file_size: u64) {
        // Evict if needed
        if self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }
        let entry = SignatureEntry::new(signature, file_size, self.config.default_ttl);
        self.entries.insert(key, entry);
        self.total_inserts += 1;
    }

    /// Insert with a custom TTL.
    pub fn insert_with_ttl(
        &mut self,
        key: String,
        signature: Vec<u8>,
        file_size: u64,
        ttl: Duration,
    ) {
        if self.entries.len() >= self.config.max_entries {
            self.evict_one();
        }
        let entry = SignatureEntry::new(signature, file_size, ttl);
        self.entries.insert(key, entry);
        self.total_inserts += 1;
    }

    /// Look up a signature by key.
    pub fn get(&mut self, key: &str) -> Option<&[u8]> {
        self.total_lookups += 1;
        // Remove if expired
        if self.entries.get(key).map_or(false, |e| e.is_expired()) {
            self.entries.remove(key);
            return None;
        }
        if let Some(entry) = self.entries.get_mut(key) {
            entry.record_access();
            self.total_hits += 1;
            Some(entry.signature())
        } else {
            None
        }
    }

    /// Check if a key exists and is not expired.
    pub fn contains(&self, key: &str) -> bool {
        self.entries.get(key).map_or(false, |e| !e.is_expired())
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &str) -> bool {
        self.entries.remove(key).is_some()
    }

    /// Get the number of entries (including potentially expired ones).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all expired entries.
    pub fn purge_expired(&mut self) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| !entry.is_expired());
        let removed = before - self.entries.len();
        self.total_evictions += removed as u64;
        removed
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get store statistics.
    pub fn stats(&self) -> StoreStats {
        let total_signature_bytes: u64 = self
            .entries
            .values()
            .map(|e| e.signature.len() as u64)
            .sum();
        StoreStats {
            entries: self.entries.len(),
            total_inserts: self.total_inserts,
            total_lookups: self.total_lookups,
            total_hits: self.total_hits,
            total_evictions: self.total_evictions,
            total_signature_bytes,
        }
    }

    /// Evict one entry (LRU or oldest).
    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        // First, try to evict an expired entry
        let expired_key = self
            .entries
            .iter()
            .find(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone());

        if let Some(key) = expired_key {
            self.entries.remove(&key);
            self.total_evictions += 1;
            return;
        }

        // Otherwise, evict the least recently used entry
        if self.config.lru_eviction {
            let lru_key = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone());
            if let Some(key) = lru_key {
                self.entries.remove(&key);
                self.total_evictions += 1;
            }
        } else {
            // Evict the oldest entry
            let oldest_key = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.created_at)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                self.entries.remove(&key);
                self.total_evictions += 1;
            }
        }
    }

    /// Find entries whose signatures match a given signature.
    pub fn find_matching(&self, signature: &[u8]) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, e)| !e.is_expired() && e.signature == signature)
            .map(|(k, _)| k.as_str())
            .collect()
    }
}

impl Default for SignatureStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Store statistics.
#[derive(Debug, Clone)]
pub struct StoreStats {
    /// Number of current entries.
    pub entries: usize,
    /// Total inserts performed.
    pub total_inserts: u64,
    /// Total lookups performed.
    pub total_lookups: u64,
    /// Total cache hits.
    pub total_hits: u64,
    /// Total evictions.
    pub total_evictions: u64,
    /// Total bytes of stored signatures.
    pub total_signature_bytes: u64,
}

impl StoreStats {
    /// Get the hit rate as a fraction.
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.total_hits as f64 / self.total_lookups as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_entry_new() {
        let entry = SignatureEntry::new(vec![1, 2, 3], 1024, Duration::from_secs(60));
        assert_eq!(entry.signature(), &[1, 2, 3]);
        assert_eq!(entry.file_size, 1024);
        assert_eq!(entry.access_count(), 0);
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_signature_entry_access() {
        let mut entry = SignatureEntry::new(vec![1], 100, Duration::from_secs(600));
        entry.record_access();
        entry.record_access();
        assert_eq!(entry.access_count(), 2);
    }

    #[test]
    fn test_store_config_default() {
        let cfg = StoreConfig::default();
        assert_eq!(cfg.default_ttl, Duration::from_secs(DEFAULT_TTL_SECS));
        assert_eq!(cfg.max_entries, DEFAULT_MAX_ENTRIES);
        assert!(cfg.lru_eviction);
    }

    #[test]
    fn test_store_config_builders() {
        let cfg = StoreConfig::default()
            .with_ttl(Duration::from_secs(300))
            .with_max_entries(500)
            .with_lru_eviction(false);
        assert_eq!(cfg.default_ttl, Duration::from_secs(300));
        assert_eq!(cfg.max_entries, 500);
        assert!(!cfg.lru_eviction);
    }

    #[test]
    fn test_store_insert_and_get() {
        let mut store = SignatureStore::new();
        store.insert("file1".to_string(), vec![0xAB, 0xCD], 2048);
        let sig = store.get("file1").expect("operation should succeed");
        assert_eq!(sig, &[0xAB, 0xCD]);
    }

    #[test]
    fn test_store_get_nonexistent() {
        let mut store = SignatureStore::new();
        assert!(store.get("nope").is_none());
    }

    #[test]
    fn test_store_contains() {
        let mut store = SignatureStore::new();
        store.insert("key1".to_string(), vec![1], 100);
        assert!(store.contains("key1"));
        assert!(!store.contains("key2"));
    }

    #[test]
    fn test_store_remove() {
        let mut store = SignatureStore::new();
        store.insert("key1".to_string(), vec![1], 100);
        assert!(store.remove("key1"));
        assert!(!store.remove("key1"));
        assert!(!store.contains("key1"));
    }

    #[test]
    fn test_store_clear() {
        let mut store = SignatureStore::new();
        store.insert("a".to_string(), vec![1], 100);
        store.insert("b".to_string(), vec![2], 200);
        store.clear();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_store_eviction_on_max_entries() {
        let config = StoreConfig::default().with_max_entries(3);
        let mut store = SignatureStore::with_config(config);
        store.insert("a".to_string(), vec![1], 100);
        store.insert("b".to_string(), vec![2], 100);
        store.insert("c".to_string(), vec![3], 100);
        // This should evict one entry
        store.insert("d".to_string(), vec![4], 100);
        assert_eq!(store.len(), 3);
        assert!(store.contains("d"));
    }

    #[test]
    fn test_store_stats() {
        let mut store = SignatureStore::new();
        store.insert("a".to_string(), vec![1, 2, 3], 100);
        let _ = store.get("a");
        let _ = store.get("b"); // miss

        let stats = store.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_inserts, 1);
        assert_eq!(stats.total_lookups, 2);
        assert_eq!(stats.total_hits, 1);
        assert_eq!(stats.total_signature_bytes, 3);
    }

    #[test]
    fn test_store_hit_rate() {
        let stats = StoreStats {
            entries: 10,
            total_inserts: 10,
            total_lookups: 100,
            total_hits: 75,
            total_evictions: 0,
            total_signature_bytes: 100,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_store_hit_rate_empty() {
        let stats = StoreStats {
            entries: 0,
            total_inserts: 0,
            total_lookups: 0,
            total_hits: 0,
            total_evictions: 0,
            total_signature_bytes: 0,
        };
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_matching() {
        let mut store = SignatureStore::new();
        store.insert("f1".to_string(), vec![1, 2, 3], 100);
        store.insert("f2".to_string(), vec![1, 2, 3], 200);
        store.insert("f3".to_string(), vec![4, 5, 6], 300);

        let matches = store.find_matching(&[1, 2, 3]);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&"f1"));
        assert!(matches.contains(&"f2"));
    }

    #[test]
    fn test_insert_with_custom_ttl() {
        let mut store = SignatureStore::new();
        store.insert_with_ttl("k".to_string(), vec![9], 50, Duration::from_secs(1));
        assert!(store.contains("k"));
    }

    #[test]
    fn test_expired_entry_not_returned() {
        let mut store = SignatureStore::with_config(
            StoreConfig::default().with_ttl(Duration::from_millis(0)),
        );
        store.insert("expired".to_string(), vec![1], 100);
        // With 0ms TTL, the entry should be expired immediately (or within the test)
        // We use a sleep-free approach: the TTL is 0, so elapsed > 0 => expired
        std::thread::sleep(Duration::from_millis(1));
        assert!(store.get("expired").is_none());
    }
}
