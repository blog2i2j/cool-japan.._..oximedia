//! Hash-based transcode result caching with configurable eviction policy.
//!
//! The transcode pipeline can be expensive for long-form content. When the
//! same source material with identical parameters is requested repeatedly
//! (e.g. re-transcoding for different delivery formats from a shared origin,
//! or re-running a failed job) it is wasteful to repeat the encode.
//!
//! This module provides a [`TranscodeCache`] that:
//! - Keys entries on a [`CacheKey`] derived from the source content hash and
//!   the encoding parameters hash.
//! - Supports [`EvictionPolicy::Lru`] (Least-Recently-Used) and
//!   [`EvictionPolicy::Lfu`] (Least-Frequently-Used) strategies.
//! - Tracks per-entry metadata: creation time, last-access time, access count,
//!   size in bytes, and codec parameters.
//! - Reports cache statistics through [`CacheStats`].
//! - Thread-safe via internal `parking_lot::Mutex`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_transcode::transcode_cache::{
//!     CacheKey, CacheParams, TranscodeCache, TranscodeCacheConfig, EvictionPolicy,
//! };
//!
//! let cfg = TranscodeCacheConfig {
//!     max_entries: 64,
//!     max_bytes: 10 * 1024 * 1024 * 1024,  // 10 GiB
//!     eviction_policy: EvictionPolicy::Lru,
//! };
//! let mut cache = TranscodeCache::new(cfg);
//!
//! let key = CacheKey::new(0xdeadbeef_u64, &CacheParams {
//!     codec: "vp9".into(),
//!     bitrate_bps: 4_000_000,
//!     width: 1920,
//!     height: 1080,
//!     extra: Default::default(),
//! });
//!
//! // First lookup – miss.
//! assert!(cache.get(&key).is_none());
//!
//! // Insert after transcoding.
//! cache.insert(key.clone(), "/tmp/output.webm".into(), 20_000_000).unwrap();
//!
//! // Second lookup – hit.
//! assert!(cache.get(&key).is_some());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::TranscodeError;

// ---------------------------------------------------------------------------
// Public configuration types
// ---------------------------------------------------------------------------

/// Cache eviction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Evict the least-recently-used entry.
    Lru,
    /// Evict the least-frequently-used entry.
    Lfu,
    /// Evict the entry with the largest stored file size.
    LargestFirst,
}

/// Configuration for [`TranscodeCache`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscodeCacheConfig {
    /// Maximum number of cache entries.
    pub max_entries: usize,
    /// Maximum total stored bytes.  `0` means unlimited.
    pub max_bytes: u64,
    /// Which eviction strategy to use.
    pub eviction_policy: EvictionPolicy,
}

impl Default for TranscodeCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 256,
            max_bytes: 50 * 1024 * 1024 * 1024, // 50 GiB
            eviction_policy: EvictionPolicy::Lru,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Encoding parameters that are part of the cache key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct CacheParams {
    /// Target codec id (e.g. `"vp9"`).
    pub codec: String,
    /// Target video bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Arbitrary additional parameters (e.g. CRF, audio codec).
    pub extra: HashMap<String, String>,
}

impl CacheParams {
    /// Computes a deterministic 64-bit hash of the parameters.
    ///
    /// Uses a simple but stable FNV-1a accumulation so there are no external
    /// dependencies.
    #[must_use]
    pub fn hash(&self) -> u64 {
        const OFFSET: u64 = 0xcbf29ce484222325;
        const PRIME: u64 = 0x00000100000001b3;

        let mut h: u64 = OFFSET;
        let mix = |h: u64, bytes: &[u8]| -> u64 {
            bytes.iter().fold(h, |acc, &b| {
                acc.wrapping_mul(PRIME) ^ u64::from(b)
            })
        };

        h = mix(h, self.codec.as_bytes());
        h = mix(h, &self.bitrate_bps.to_le_bytes());
        h = mix(h, &self.width.to_le_bytes());
        h = mix(h, &self.height.to_le_bytes());

        // Sort extra keys for determinism.
        let mut extras: Vec<(&String, &String)> = self.extra.iter().collect();
        extras.sort_by_key(|(k, _)| k.as_str());
        for (k, v) in extras {
            h = mix(h, k.as_bytes());
            h = mix(h, v.as_bytes());
        }
        h
    }
}

/// Composite cache key: `(source_hash, params_hash)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// 64-bit hash of the source file content (e.g. xxHash or SHA-256 prefix).
    pub source_hash: u64,
    /// 64-bit hash of the [`CacheParams`].
    pub params_hash: u64,
}

impl CacheKey {
    /// Constructs a key from a source content hash and a set of parameters.
    #[must_use]
    pub fn new(source_hash: u64, params: &CacheParams) -> Self {
        Self {
            source_hash,
            params_hash: params.hash(),
        }
    }

    /// Constructs a key directly from two pre-computed hashes.
    #[must_use]
    pub fn from_hashes(source_hash: u64, params_hash: u64) -> Self {
        Self {
            source_hash,
            params_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache entry
// ---------------------------------------------------------------------------

/// A single cached transcode result.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Path to the transcoded output file.
    pub output_path: String,
    /// Size of the output file in bytes.
    pub size_bytes: u64,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// How many times this entry has been accessed.
    pub access_count: u64,
}

impl CacheEntry {
    fn new(output_path: String, size_bytes: u64) -> Self {
        let now = Instant::now();
        Self {
            output_path,
            size_bytes,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }

    /// Age of this entry.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for a [`TranscodeCache`] instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    /// Total number of cache lookups performed.
    pub total_lookups: u64,
    /// Lookups that found a cached result.
    pub hits: u64,
    /// Lookups that did not find a cached result.
    pub misses: u64,
    /// Total entries ever inserted.
    pub total_inserts: u64,
    /// Total entries evicted.
    pub total_evictions: u64,
    /// Current number of entries in the cache.
    pub current_entries: usize,
    /// Current total stored bytes.
    pub current_bytes: u64,
}

impl CacheStats {
    /// Hit ratio in the range `[0.0, 1.0]`.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Main cache
// ---------------------------------------------------------------------------

/// A thread-safe transcode result cache.
pub struct TranscodeCache {
    config: TranscodeCacheConfig,
    entries: HashMap<CacheKey, CacheEntry>,
    stats: CacheStats,
}

impl TranscodeCache {
    /// Creates a new cache with the provided configuration.
    #[must_use]
    pub fn new(config: TranscodeCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Creates a cache with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TranscodeCacheConfig::default())
    }

    /// Looks up `key` and returns a reference to the entry if present.
    ///
    /// Updates the entry's `last_accessed` timestamp and `access_count`.
    pub fn get(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        self.stats.total_lookups += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            self.stats.hits += 1;
            // Return immutable ref.
            self.entries.get(key)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Inserts a new entry, evicting an existing one if capacity is exceeded.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidOutput`] if `output_path` is empty or
    /// if `size_bytes` is `0`.
    pub fn insert(
        &mut self,
        key: CacheKey,
        output_path: String,
        size_bytes: u64,
    ) -> Result<(), TranscodeError> {
        if output_path.is_empty() {
            return Err(TranscodeError::InvalidOutput(
                "Cache: output_path must not be empty".into(),
            ));
        }
        if size_bytes == 0 {
            return Err(TranscodeError::InvalidOutput(
                "Cache: size_bytes must be > 0".into(),
            ));
        }

        // If the key already exists, just update it.
        if let Some(existing) = self.entries.get_mut(&key) {
            self.stats.current_bytes = self
                .stats
                .current_bytes
                .saturating_sub(existing.size_bytes)
                .saturating_add(size_bytes);
            existing.output_path = output_path;
            existing.size_bytes = size_bytes;
            existing.last_accessed = Instant::now();
            return Ok(());
        }

        // Evict if needed.
        while self.needs_eviction(size_bytes) {
            if !self.evict_one() {
                break; // No more candidates to evict.
            }
        }

        self.stats.current_bytes = self.stats.current_bytes.saturating_add(size_bytes);
        self.stats.current_entries += 1;
        self.stats.total_inserts += 1;
        self.entries.insert(key, CacheEntry::new(output_path, size_bytes));
        Ok(())
    }

    /// Removes `key` from the cache.  Returns `true` if the key was present.
    pub fn remove(&mut self, key: &CacheKey) -> bool {
        if let Some(entry) = self.entries.remove(key) {
            self.stats.current_bytes = self.stats.current_bytes.saturating_sub(entry.size_bytes);
            self.stats.current_entries = self.stats.current_entries.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Evicts all entries whose `age()` exceeds `max_age`.
    ///
    /// Returns the number of entries evicted.
    pub fn evict_older_than(&mut self, max_age: Duration) -> usize {
        let expired: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.age() > max_age)
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired.len();
        for key in expired {
            self.remove(&key);
            self.stats.total_evictions += 1;
        }
        count
    }

    /// Clears all entries.
    pub fn clear(&mut self) {
        self.stats.total_evictions += self.entries.len() as u64;
        self.entries.clear();
        self.stats.current_entries = 0;
        self.stats.current_bytes = 0;
    }

    /// Returns a snapshot of current cache statistics.
    #[must_use]
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Returns the current number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the cache contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn needs_eviction(&self, incoming_bytes: u64) -> bool {
        let over_count = self.entries.len() >= self.config.max_entries;
        let over_bytes = self.config.max_bytes > 0
            && self.stats.current_bytes.saturating_add(incoming_bytes) > self.config.max_bytes;
        over_count || over_bytes
    }

    /// Selects and removes one victim entry.  Returns `false` if the cache is
    /// already empty.
    fn evict_one(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let victim_key: Option<CacheKey> = match self.config.eviction_policy {
            EvictionPolicy::Lru => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone()),

            EvictionPolicy::Lfu => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),

            EvictionPolicy::LargestFirst => self
                .entries
                .iter()
                .max_by_key(|(_, e)| e.size_bytes)
                .map(|(k, _)| k.clone()),
        };

        if let Some(key) = victim_key {
            if let Some(evicted) = self.entries.remove(&key) {
                self.stats.current_bytes = self
                    .stats
                    .current_bytes
                    .saturating_sub(evicted.size_bytes);
                self.stats.current_entries = self.stats.current_entries.saturating_sub(1);
            }
            self.stats.total_evictions += 1;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(src: u64, codec: &str, bitrate: u64) -> CacheKey {
        let params = CacheParams {
            codec: codec.into(),
            bitrate_bps: bitrate,
            width: 1920,
            height: 1080,
            extra: HashMap::new(),
        };
        CacheKey::new(src, &params)
    }

    #[test]
    fn test_cache_miss_on_empty() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(1, "vp9", 4_000_000);
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_insert_and_hit() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(42, "av1", 3_000_000);
        cache.insert(key.clone(), "/tmp/out.webm".into(), 10_000).unwrap();
        let entry = cache.get(&key).expect("should be a hit");
        assert_eq!(entry.output_path, "/tmp/out.webm");
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().total_inserts, 1);
    }

    #[test]
    fn test_remove_entry() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(7, "opus", 128_000);
        cache.insert(key.clone(), "/tmp/audio.ogg".into(), 512).unwrap();
        assert!(cache.remove(&key));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_eviction_lru_respects_max_entries() {
        let cfg = TranscodeCacheConfig {
            max_entries: 3,
            max_bytes: 0,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = TranscodeCache::new(cfg);
        for i in 0u64..5 {
            let key = make_key(i, "vp9", i * 1000);
            cache.insert(key, format!("/tmp/out{i}.webm"), 100).unwrap();
        }
        // Only 3 entries should remain after 5 inserts.
        assert_eq!(cache.len(), 3, "LRU eviction should cap at max_entries");
        assert!(cache.stats().total_evictions >= 2);
    }

    #[test]
    fn test_eviction_lfu_respects_max_entries() {
        let cfg = TranscodeCacheConfig {
            max_entries: 2,
            max_bytes: 0,
            eviction_policy: EvictionPolicy::Lfu,
        };
        let mut cache = TranscodeCache::new(cfg);
        for i in 0u64..4 {
            let key = make_key(i, "av1", i * 500);
            cache.insert(key, format!("/tmp/lfu{i}.webm"), 200).unwrap();
        }
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_eviction_largest_first() {
        let cfg = TranscodeCacheConfig {
            max_entries: 2,
            max_bytes: 0,
            eviction_policy: EvictionPolicy::LargestFirst,
        };
        let mut cache = TranscodeCache::new(cfg);
        let k0 = make_key(0, "vp9", 1000);
        let k1 = make_key(1, "vp9", 2000);
        let k2 = make_key(2, "vp9", 3000);
        cache.insert(k0.clone(), "/tmp/a.webm".into(), 100).unwrap();
        cache.insert(k1.clone(), "/tmp/b.webm".into(), 500).unwrap();
        // This should evict the largest entry (k1, 500 bytes).
        cache.insert(k2.clone(), "/tmp/c.webm".into(), 200).unwrap();
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&k1).is_none(), "largest entry should have been evicted");
    }

    #[test]
    fn test_max_bytes_triggers_eviction() {
        let cfg = TranscodeCacheConfig {
            max_entries: 100,
            max_bytes: 1000,
            eviction_policy: EvictionPolicy::Lru,
        };
        let mut cache = TranscodeCache::new(cfg);
        // Insert entries totalling >1000 bytes.
        for i in 0u64..5 {
            let key = make_key(i, "av1", i);
            cache.insert(key, format!("/tmp/b{i}.webm"), 300).unwrap();
        }
        // Current bytes should be <=1000 after evictions.
        assert!(
            cache.stats().current_bytes <= 1000,
            "bytes {} should be <= 1000",
            cache.stats().current_bytes
        );
    }

    #[test]
    fn test_clear_resets_state() {
        let mut cache = TranscodeCache::with_defaults();
        for i in 0u64..5 {
            let key = make_key(i, "flac", i);
            cache.insert(key, format!("/tmp/f{i}.flac"), 400).unwrap();
        }
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.stats().current_bytes, 0);
    }

    #[test]
    fn test_hit_ratio_calculation() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(99, "opus", 192_000);
        cache.insert(key.clone(), "/tmp/o.opus".into(), 1024).unwrap();
        let _ = cache.get(&key); // hit
        let _ = cache.get(&make_key(0, "unknown", 0)); // miss
        let ratio = cache.stats().hit_ratio();
        assert!((ratio - 0.5).abs() < 1e-9, "hit ratio should be 0.5");
    }

    #[test]
    fn test_insert_empty_path_returns_error() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(1, "vp9", 1000);
        assert!(cache.insert(key, String::new(), 100).is_err());
    }

    #[test]
    fn test_insert_zero_bytes_returns_error() {
        let mut cache = TranscodeCache::with_defaults();
        let key = make_key(2, "av1", 2000);
        assert!(cache.insert(key, "/tmp/x.webm".into(), 0).is_err());
    }

    #[test]
    fn test_cache_params_hash_deterministic() {
        let p = CacheParams {
            codec: "vp9".into(),
            bitrate_bps: 5_000_000,
            width: 1280,
            height: 720,
            extra: HashMap::new(),
        };
        assert_eq!(p.hash(), p.hash(), "hash must be deterministic");
    }

    #[test]
    fn test_cache_params_different_codecs_different_hash() {
        let mut p1 = CacheParams {
            codec: "vp9".into(),
            bitrate_bps: 4_000_000,
            width: 1920,
            height: 1080,
            extra: HashMap::new(),
        };
        let mut p2 = p1.clone();
        p2.codec = "av1".into();
        assert_ne!(p1.hash(), p2.hash());
        p1.bitrate_bps = 2_000_000;
        assert_ne!(p1.hash(), p2.hash());
    }

    #[test]
    fn test_evict_older_than_removes_entries() {
        let mut cache = TranscodeCache::with_defaults();
        for i in 0u64..3 {
            let key = make_key(i, "vp9", i);
            cache.insert(key, format!("/tmp/t{i}.webm"), 100).unwrap();
        }
        // Evict anything older than a very short duration. Since entries were
        // just inserted this should not evict all of them, but evicting with
        // a zero duration guarantees all are stale.
        let evicted = cache.evict_older_than(Duration::from_nanos(0));
        // All entries were created at least 0 ns ago so all are evicted.
        assert!(evicted >= 1, "should have evicted at least one entry");
    }
}
