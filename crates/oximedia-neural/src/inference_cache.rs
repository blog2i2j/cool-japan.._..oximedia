//! Inference result cache to avoid redundant neural-network computation.
//!
//! [`InferenceCache`] is a time-bounded, capacity-bounded LRU-style cache that
//! maps `(model_id, input_hash)` pairs to pre-computed inference outputs.
//! Entries expire after a configurable TTL and are evicted either lazily on
//! lookup or explicitly via [`InferenceCache::evict_expired`].
//!
//! ## FNV-1a hashing
//!
//! Input bytes are fingerprinted with the **FNV-1a 64-bit** algorithm, chosen
//! for its excellent distribution and zero-allocation inline implementation.
//! See [`fnv1a_hash`] for the standalone function.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::inference_cache::{
//!     InferenceCache, InferenceCacheConfig, InferenceKey, InferenceResult, fnv1a_hash,
//! };
//!
//! let config = InferenceCacheConfig { max_entries: 128, ttl_secs: 60 };
//! let mut cache = InferenceCache::new(config);
//!
//! let key = InferenceKey {
//!     model_id: "scene_v1".to_string(),
//!     input_hash: fnv1a_hash(b"my_frame_bytes"),
//! };
//! let result = InferenceResult {
//!     output: vec![0.1, 0.9],
//!     inference_time_us: 1234,
//!     model_id: "scene_v1".to_string(),
//! };
//!
//! cache.store(key.clone(), result, 0);
//! assert!(cache.lookup(&key, 0).is_some());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// FNV-1a 64-bit hash
// ─────────────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of arbitrary bytes.
///
/// This implementation follows the reference algorithm from the [FNV authors]:
/// - Offset basis: `0xcbf29ce484222325`
/// - Prime: `0x100000001b3`
///
/// The function is deterministic across platforms and compilations for the
/// same input bytes.
///
/// [FNV authors]: http://www.isthe.com/chongo/tech/comp/fnv/
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceKey
// ─────────────────────────────────────────────────────────────────────────────

/// Cache key identifying a specific model + input combination.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferenceKey {
    /// Identifier of the model that produced the result.
    pub model_id: String,
    /// FNV-1a hash of the raw input bytes fed to the model.
    pub input_hash: u64,
}

impl InferenceKey {
    /// Creates a new `InferenceKey`.
    pub fn new(model_id: impl Into<String>, input_hash: u64) -> Self {
        Self {
            model_id: model_id.into(),
            input_hash,
        }
    }

    /// Creates a key by hashing `input_bytes` with FNV-1a.
    pub fn from_bytes(model_id: impl Into<String>, input_bytes: &[u8]) -> Self {
        Self::new(model_id, fnv1a_hash(input_bytes))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceResult
// ─────────────────────────────────────────────────────────────────────────────

/// The cached output of a single model inference pass.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Model output tensor (flattened).
    pub output: Vec<f32>,
    /// Wall-clock time taken by the inference in microseconds.
    pub inference_time_us: u64,
    /// ID of the model that produced this result (redundant with key but
    /// useful for debugging).
    pub model_id: String,
}

impl InferenceResult {
    /// Creates a new `InferenceResult`.
    pub fn new(model_id: impl Into<String>, output: Vec<f32>, inference_time_us: u64) -> Self {
        Self {
            output,
            inference_time_us,
            model_id: model_id.into(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceCacheConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`InferenceCache`].
#[derive(Debug, Clone)]
pub struct InferenceCacheConfig {
    /// Maximum number of entries the cache will hold before evicting the
    /// oldest entry on insert.
    pub max_entries: usize,
    /// Time-to-live in seconds.  An entry older than this (relative to the
    /// `now_secs` supplied to [`InferenceCache::lookup`]) is treated as
    /// expired.
    pub ttl_secs: u64,
}

impl Default for InferenceCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            ttl_secs: 300,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal entry
// ─────────────────────────────────────────────────────────────────────────────

struct CacheEntry {
    result: InferenceResult,
    /// Unix-epoch seconds when this entry was inserted.
    inserted_at: u64,
    /// Insertion order (monotonically increasing); used as an LRU proxy when
    /// eviction is needed.
    seq: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// CacheStats
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of cache operation counters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Total number of successful cache lookups.
    pub hits: u64,
    /// Total number of failed cache lookups (key not found or expired).
    pub misses: u64,
    /// Current number of live entries in the cache.
    pub entries: usize,
    /// Total number of entries that have been removed (capacity or TTL
    /// eviction).
    pub evictions: u64,
}

impl CacheStats {
    /// Returns the hit rate as a value in `[0, 1]`, or `0.0` if no lookups
    /// have been performed.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceCache
// ─────────────────────────────────────────────────────────────────────────────

/// A bounded, TTL-expiring cache for neural-network inference results.
///
/// ## Eviction policy
///
/// When [`InferenceCache::store`] would exceed `max_entries`, the entry with
/// the **smallest insertion sequence number** (i.e., the oldest entry) is
/// evicted first.  Expired entries are also opportunistically removed during
/// [`InferenceCache::lookup`].
pub struct InferenceCache {
    config: InferenceCacheConfig,
    store: HashMap<InferenceKey, CacheEntry>,
    hits: u64,
    misses: u64,
    evictions: u64,
    seq: u64,
}

impl InferenceCache {
    /// Creates a new, empty `InferenceCache` with the given configuration.
    pub fn new(config: InferenceCacheConfig) -> Self {
        let capacity = config.max_entries;
        Self {
            config,
            store: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
            seq: 0,
        }
    }

    /// Inserts or replaces the result for `key`.
    ///
    /// `now_secs` is the current wall-clock time in Unix-epoch seconds.  It
    /// is used to timestamp the entry for TTL expiry.
    ///
    /// If the cache is at capacity, the oldest entry (by insertion order) is
    /// evicted first.
    pub fn store(&mut self, key: InferenceKey, result: InferenceResult, now_secs: u64) {
        // Evict the oldest entry if we are at capacity and this is a new key.
        if self.store.len() >= self.config.max_entries && !self.store.contains_key(&key) {
            self.evict_oldest();
        }

        self.seq += 1;
        let entry = CacheEntry {
            result,
            inserted_at: now_secs,
            seq: self.seq,
        };
        self.store.insert(key, entry);
    }

    /// Looks up the cached result for `key`.
    ///
    /// Returns `None` if the key is absent **or** the entry has expired (i.e.,
    /// `now_secs - inserted_at > ttl_secs`).  Expired entries are removed on
    /// access (lazy eviction).
    pub fn lookup(&mut self, key: &InferenceKey, now_secs: u64) -> Option<&InferenceResult> {
        // Check existence and expiry before borrowing.
        let expired = match self.store.get(key) {
            None => {
                self.misses += 1;
                return None;
            }
            Some(entry) => now_secs.saturating_sub(entry.inserted_at) > self.config.ttl_secs,
        };

        if expired {
            self.store.remove(key);
            self.evictions += 1;
            self.misses += 1;
            return None;
        }

        self.hits += 1;
        // Re-borrow immutably for the return value.
        self.store.get(key).map(|e| &e.result)
    }

    /// Computes the cache **hit rate** as a fraction of total lookups.
    ///
    /// Returns `0.0` if no lookups have been performed yet.
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }

    /// Removes all entries whose age exceeds the configured TTL.
    ///
    /// Returns the number of entries that were evicted.
    pub fn evict_expired(&mut self, now_secs: u64) -> usize {
        let ttl = self.config.ttl_secs;
        let before = self.store.len();
        self.store
            .retain(|_, entry| now_secs.saturating_sub(entry.inserted_at) <= ttl);
        let removed = before - self.store.len();
        self.evictions += removed as u64;
        removed
    }

    /// Returns a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.store.len(),
            evictions: self.evictions,
        }
    }

    /// Returns the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Clears all entries and resets all counters.
    pub fn clear(&mut self) {
        self.store.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
        self.seq = 0;
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Evicts the single entry with the smallest sequence number (oldest
    /// insertion).
    fn evict_oldest(&mut self) {
        if self.store.is_empty() {
            return;
        }
        // Find the key with the minimum sequence number.
        let oldest_key = self
            .store
            .iter()
            .min_by_key(|(_, e)| e.seq)
            .map(|(k, _)| k.clone());

        if let Some(key) = oldest_key {
            self.store.remove(&key);
            self.evictions += 1;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_key(model: &str, hash: u64) -> InferenceKey {
        InferenceKey::new(model, hash)
    }

    fn make_result(model: &str, output: Vec<f32>) -> InferenceResult {
        InferenceResult::new(model, output, 0)
    }

    // ── fnv1a_hash ───────────────────────────────────────────────────────────

    #[test]
    fn test_fnv1a_empty_returns_offset_basis() {
        const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        assert_eq!(fnv1a_hash(b""), OFFSET_BASIS);
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"hello world");
        let h2 = fnv1a_hash(b"hello world");
        assert_eq!(h1, h2, "FNV-1a must be deterministic");
    }

    #[test]
    fn test_fnv1a_different_inputs_different_hashes() {
        let h1 = fnv1a_hash(b"frame_a");
        let h2 = fnv1a_hash(b"frame_b");
        assert_ne!(h1, h2, "Different inputs must produce different hashes");
    }

    #[test]
    fn test_fnv1a_known_value() {
        // Known FNV-1a 64-bit hash of "foobar" = 0x85944171f73967e8
        assert_eq!(fnv1a_hash(b"foobar"), 0x85944171f73967e8_u64);
    }

    // ── InferenceCache: store and retrieve ───────────────────────────────────

    #[test]
    fn test_store_and_lookup_success() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 60,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("model_a", 42);
        let result = make_result("model_a", vec![0.1, 0.9]);
        cache.store(key.clone(), result, 0);
        let found = cache.lookup(&key, 0);
        assert!(found.is_some(), "Stored entry must be found on lookup");
        let r = found.expect("result exists");
        assert_eq!(r.output, vec![0.1, 0.9]);
    }

    #[test]
    fn test_lookup_missing_key_returns_none() {
        let mut cache = InferenceCache::new(InferenceCacheConfig::default());
        let key = make_key("no_model", 999);
        assert!(cache.lookup(&key, 0).is_none());
    }

    // ── TTL expiry ───────────────────────────────────────────────────────────

    #[test]
    fn test_ttl_expiry_returns_none() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 10,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("model_b", 1);
        cache.store(key.clone(), make_result("model_b", vec![1.0]), 0);
        // Look up after TTL has elapsed (now = 100, inserted at 0, ttl = 10).
        let found = cache.lookup(&key, 100);
        assert!(found.is_none(), "Expired entry must not be returned");
    }

    #[test]
    fn test_ttl_not_yet_expired() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 100,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("model_c", 2);
        cache.store(key.clone(), make_result("model_c", vec![0.5]), 0);
        // Now = 50; inserted at 0; ttl = 100 → 50 <= 100 → still valid.
        assert!(cache.lookup(&key, 50).is_some());
    }

    // ── evict_expired ────────────────────────────────────────────────────────

    #[test]
    fn test_evict_expired_removes_old_entries() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 5,
        };
        let mut cache = InferenceCache::new(config);
        for i in 0u64..5 {
            cache.store(
                make_key("m", i),
                make_result("m", vec![i as f32]),
                0, // inserted at t=0
            );
        }
        // At t=10 all entries are expired (age=10 > ttl=5).
        let removed = cache.evict_expired(10);
        assert_eq!(removed, 5, "All 5 entries must be evicted");
        assert!(cache.is_empty());
    }

    #[test]
    fn test_evict_expired_keeps_live_entries() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 50,
        };
        let mut cache = InferenceCache::new(config);
        // Insert 3 old entries (at t=0) and 2 fresh entries (at t=90).
        for i in 0u64..3 {
            cache.store(make_key("old", i), make_result("old", vec![0.0]), 0);
        }
        for i in 0u64..2 {
            cache.store(make_key("fresh", i), make_result("fresh", vec![1.0]), 90);
        }
        // At t=100: old entries have age=100 > 50 (expired); fresh have age=10 <= 50.
        let removed = cache.evict_expired(100);
        assert_eq!(removed, 3);
        assert_eq!(cache.len(), 2);
    }

    // ── hit rate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hit_rate_no_lookups_is_zero() {
        let cache = InferenceCache::new(InferenceCacheConfig::default());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_hit_rate_all_hits() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 60,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("x", 0);
        cache.store(key.clone(), make_result("x", vec![0.0]), 0);
        cache.lookup(&key, 0);
        cache.lookup(&key, 0);
        assert!(
            (cache.hit_rate() - 1.0).abs() < 1e-6,
            "All-hit scenario: rate must be 1.0"
        );
    }

    #[test]
    fn test_hit_rate_mixed() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 60,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("y", 7);
        cache.store(key.clone(), make_result("y", vec![]), 0);
        cache.lookup(&key, 0); // hit
        cache.lookup(&make_key("z", 99), 0); // miss
                                             // 1 hit / 2 lookups = 0.5
        assert!((cache.hit_rate() - 0.5).abs() < 1e-5);
    }

    // ── capacity eviction ────────────────────────────────────────────────────

    #[test]
    fn test_capacity_eviction_does_not_exceed_max() {
        let config = InferenceCacheConfig {
            max_entries: 3,
            ttl_secs: 1000,
        };
        let mut cache = InferenceCache::new(config);
        for i in 0u64..10 {
            cache.store(make_key("cap", i), make_result("cap", vec![i as f32]), 0);
        }
        assert!(
            cache.len() <= 3,
            "Cache must not exceed max_entries (len={})",
            cache.len()
        );
    }

    // ── CacheStats ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_reflects_operations() {
        let config = InferenceCacheConfig {
            max_entries: 10,
            ttl_secs: 60,
        };
        let mut cache = InferenceCache::new(config);
        let key = make_key("stats_m", 55);
        cache.store(key.clone(), make_result("stats_m", vec![1.0, 2.0]), 0);
        cache.lookup(&key, 0); // hit
        cache.lookup(&make_key("missing", 0), 0); // miss
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_key_from_bytes() {
        let k1 = InferenceKey::from_bytes("model", b"input_data");
        let k2 = InferenceKey::from_bytes("model", b"input_data");
        assert_eq!(k1, k2);
        let k3 = InferenceKey::from_bytes("model", b"other_data");
        assert_ne!(k1, k3);
    }
}
