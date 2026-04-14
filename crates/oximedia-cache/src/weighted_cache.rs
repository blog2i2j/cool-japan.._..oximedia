//! Weighted cache scoring with configurable per-media-type weight factors.
//!
//! Standard LRU eviction treats all entries equally. For multimedia workloads
//! different content types benefit from different retention policies. This
//! module provides [`WeightConfig`] — a table of per-type weights (recency,
//! size-efficiency, priority) — and [`WeightedCache`], an LRU-ordered cache
//! that uses the blended score to decide which entry to evict next.
//!
//! # Scoring formula
//!
//! For each candidate entry the eviction score is:
//!
//! ```text
//! score = recency_weight * recency_factor
//!       + priority_weight * priority_factor
//!       - size_weight * size_factor
//! ```
//!
//! where
//!
//! - `recency_factor` is `1.0 - (age_ns / max_age_ns)` clamped to `[0, 1]`
//! - `priority_factor` is the entry's priority normalised to `[0, 1]`
//! - `size_factor` is `entry_bytes / max_entry_bytes` normalised to `[0, 1]`
//!
//! The entry with the **lowest** score is the best eviction candidate.
//!
//! # Example
//!
//! ```rust
//! use oximedia_cache::weighted_cache::{WeightConfig, WeightedCache, CacheMediaType};
//!
//! let weights = WeightConfig::default();
//! let mut cache = WeightedCache::new(256, weights);
//! cache.insert("seg-001", vec![0u8; 64], CacheMediaType::VideoSegment, 7);
//! assert!(cache.get("seg-001").is_some());
//! ```

use std::collections::HashMap;
use std::time::Instant;

use thiserror::Error;

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors produced by [`WeightedCache`] operations.
#[derive(Debug, Error)]
pub enum WeightedCacheError {
    /// The supplied `WeightConfig` failed validation.
    #[error("invalid weight config: {0}")]
    InvalidConfig(String),
}

// ── CacheMediaType ────────────────────────────────────────────────────────────

/// The media type label attached to a cached entry.
///
/// Used by [`WeightConfig`] to look up per-type weight overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheMediaType {
    /// HLS/DASH video segment.
    VideoSegment,
    /// Audio-only segment.
    AudioSegment,
    /// Still image or frame.
    Image,
    /// Streaming manifest / playlist.
    Manifest,
    /// Thumbnail preview.
    Thumbnail,
    /// Metadata sidecar.
    Metadata,
    /// Generic / unclassified entry.
    Generic,
}

// ── TypeWeights ───────────────────────────────────────────────────────────────

/// Weight factors for one media type.
///
/// Each factor scales the corresponding component of the eviction score.
/// All weights should be non-negative; they are automatically normalised
/// so their sum is 1.0 before scoring.
#[derive(Debug, Clone, Copy)]
pub struct TypeWeights {
    /// How much to value recency (freshness) of the entry.  Higher = keep
    /// recently-accessed entries longer.
    pub recency: f64,
    /// How much to value the entry's priority label.  Higher = keep
    /// high-priority entries longer.
    pub priority: f64,
    /// How much to penalise large entries.  Higher = prefer to evict large
    /// entries first.
    pub size_penalty: f64,
}

impl TypeWeights {
    /// Normalise the three weights so they sum to 1.0.
    ///
    /// If all weights are zero the result is `(1/3, 1/3, 1/3)`.
    #[must_use]
    fn normalise(self) -> Self {
        let sum = self.recency + self.priority + self.size_penalty;
        if sum == 0.0 {
            let third = 1.0 / 3.0;
            return Self {
                recency: third,
                priority: third,
                size_penalty: third,
            };
        }
        Self {
            recency: self.recency / sum,
            priority: self.priority / sum,
            size_penalty: self.size_penalty / sum,
        }
    }
}

impl Default for TypeWeights {
    fn default() -> Self {
        Self {
            recency: 0.5,
            priority: 0.3,
            size_penalty: 0.2,
        }
    }
}

// ── WeightConfig ──────────────────────────────────────────────────────────────

/// Table of per-media-type weight overrides.
///
/// Any type not present in the table falls back to `default_weights`.
#[derive(Debug, Clone)]
pub struct WeightConfig {
    /// Default weights applied when the entry type has no specific override.
    pub default_weights: TypeWeights,
    /// Per-type weight overrides.  Inserted values are normalised automatically.
    overrides: HashMap<CacheMediaType, TypeWeights>,
}

impl Default for WeightConfig {
    fn default() -> Self {
        let mut cfg = Self {
            default_weights: TypeWeights::default(),
            overrides: HashMap::new(),
        };
        // Manifests: very high recency and priority, size is negligible.
        cfg.set_weights(
            CacheMediaType::Manifest,
            TypeWeights { recency: 0.5, priority: 0.45, size_penalty: 0.05 },
        );
        // Thumbnails: moderate priority, low size penalty (they're small).
        cfg.set_weights(
            CacheMediaType::Thumbnail,
            TypeWeights { recency: 0.4, priority: 0.4, size_penalty: 0.2 },
        );
        // Video segments: balanced but with a higher size penalty.
        cfg.set_weights(
            CacheMediaType::VideoSegment,
            TypeWeights { recency: 0.4, priority: 0.25, size_penalty: 0.35 },
        );
        // Audio segments: similar to video but smaller.
        cfg.set_weights(
            CacheMediaType::AudioSegment,
            TypeWeights { recency: 0.4, priority: 0.3, size_penalty: 0.3 },
        );
        // Images: size penalty matters more.
        cfg.set_weights(
            CacheMediaType::Image,
            TypeWeights { recency: 0.35, priority: 0.25, size_penalty: 0.40 },
        );
        // Metadata: tiny; keep for a long time.
        cfg.set_weights(
            CacheMediaType::Metadata,
            TypeWeights { recency: 0.5, priority: 0.35, size_penalty: 0.15 },
        );
        cfg
    }
}

impl WeightConfig {
    /// Create a blank config (all types use `default_weights`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_weights: TypeWeights::default(),
            overrides: HashMap::new(),
        }
    }

    /// Insert or replace the weight overrides for `media_type`.
    pub fn set_weights(&mut self, media_type: CacheMediaType, weights: TypeWeights) {
        self.overrides.insert(media_type, weights.normalise());
    }

    /// Look up the weights for `media_type`, falling back to `default_weights`.
    #[must_use]
    pub fn weights_for(&self, media_type: CacheMediaType) -> TypeWeights {
        self.overrides
            .get(&media_type)
            .copied()
            .unwrap_or_else(|| self.default_weights.normalise())
    }

    /// Validate that all weights are finite and non-negative.
    pub fn validate(&self) -> Result<(), WeightedCacheError> {
        let check = |w: TypeWeights, label: &str| {
            if w.recency < 0.0 || !w.recency.is_finite() {
                return Err(WeightedCacheError::InvalidConfig(format!(
                    "{label}.recency must be finite and >= 0"
                )));
            }
            if w.priority < 0.0 || !w.priority.is_finite() {
                return Err(WeightedCacheError::InvalidConfig(format!(
                    "{label}.priority must be finite and >= 0"
                )));
            }
            if w.size_penalty < 0.0 || !w.size_penalty.is_finite() {
                return Err(WeightedCacheError::InvalidConfig(format!(
                    "{label}.size_penalty must be finite and >= 0"
                )));
            }
            Ok(())
        };
        check(self.default_weights, "default_weights")?;
        for (mt, w) in &self.overrides {
            check(*w, &format!("{mt:?}"))?;
        }
        Ok(())
    }
}

// ── CacheEntry (internal) ─────────────────────────────────────────────────────

struct Entry {
    value: Vec<u8>,
    media_type: CacheMediaType,
    priority: u8,
    last_accessed: Instant,
    size_bytes: usize,
}

// ── WeightedCache ─────────────────────────────────────────────────────────────

/// Capacity-bounded cache that uses per-media-type weight factors to choose
/// eviction candidates.
///
/// Entries are scored on each eviction pass; the entry with the lowest score
/// (least worth keeping) is removed first.
pub struct WeightedCache {
    /// Maximum number of entries the cache will hold.
    capacity: usize,
    /// Weight configuration table.
    weights: WeightConfig,
    /// Key → entry storage.
    entries: HashMap<String, Entry>,
    /// Hit counter.
    hits: u64,
    /// Miss counter.
    misses: u64,
    /// Eviction counter.
    evictions: u64,
}

impl WeightedCache {
    /// Create a new `WeightedCache` with the given capacity and weight config.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is 0.
    #[must_use]
    pub fn new(capacity: usize, weights: WeightConfig) -> Self {
        assert!(capacity > 0, "WeightedCache: capacity must be > 0");
        Self {
            capacity,
            weights,
            entries: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Insert `(key, value)` into the cache with the given `media_type` and
    /// `priority` label (0–255, higher = more important).
    ///
    /// If the key already exists it is overwritten.  If the cache is at
    /// capacity, the lowest-scored entry is evicted first.
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: Vec<u8>,
        media_type: CacheMediaType,
        priority: u8,
    ) {
        let key = key.into();
        let size_bytes = value.len();
        let now = Instant::now();

        // Overwrite existing entry without eviction.
        self.entries.insert(
            key,
            Entry {
                value,
                media_type,
                priority,
                last_accessed: now,
                size_bytes,
            },
        );

        // Evict if over capacity.
        while self.entries.len() > self.capacity {
            self.evict_one();
        }
    }

    /// Look up `key` and return a reference to its value.
    ///
    /// Records a hit or miss and updates `last_accessed` on hit.
    pub fn get(&mut self, key: &str) -> Option<&[u8]> {
        if let Some(entry) = self.entries.get_mut(key) {
            self.hits += 1;
            entry.last_accessed = Instant::now();
            Some(&entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Remove the entry for `key` and return it (if present).
    pub fn remove(&mut self, key: &str) -> Option<Vec<u8>> {
        self.entries.remove(key).map(|e| e.value)
    }

    /// Return `true` when `key` is present.
    #[must_use]
    pub fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }

    /// Number of entries currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` when no entries are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum number of entries the cache will hold.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total cache hits recorded.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total cache misses recorded.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Total evictions performed.
    #[must_use]
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Hit rate as a fraction `[0.0, 1.0]`.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Resize the cache.  If `new_capacity` is smaller than the current entry
    /// count, entries are evicted until the count fits.
    pub fn resize(&mut self, new_capacity: usize) {
        assert!(new_capacity > 0, "WeightedCache: capacity must be > 0");
        self.capacity = new_capacity;
        while self.entries.len() > self.capacity {
            self.evict_one();
        }
    }

    /// Clear all entries and reset statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }

    /// Compute the eviction score for an entry.
    ///
    /// **Lower** scores → better eviction candidates.
    fn score(&self, entry: &Entry, max_age_ns: u64, max_size: usize) -> f64 {
        let w = self.weights.weights_for(entry.media_type);

        // Recency factor: recently accessed entries score higher (harder to evict).
        let age_ns = entry.last_accessed.elapsed().as_nanos() as f64;
        let max_age = max_age_ns as f64;
        let recency_factor = if max_age == 0.0 {
            1.0
        } else {
            (1.0 - (age_ns / max_age)).clamp(0.0, 1.0)
        };

        // Priority factor: higher priority → harder to evict.
        let priority_factor = f64::from(entry.priority) / 255.0;

        // Size factor: larger entries → easier to evict (free more space).
        let size_factor = if max_size == 0 {
            0.0
        } else {
            (entry.size_bytes as f64 / max_size as f64).clamp(0.0, 1.0)
        };

        w.recency * recency_factor + w.priority * priority_factor - w.size_penalty * size_factor
    }

    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        // Compute score context: max age and max size across all entries.
        let max_age_ns = self
            .entries
            .values()
            .map(|e| e.last_accessed.elapsed().as_nanos() as u64)
            .max()
            .unwrap_or(1);
        let max_size = self
            .entries
            .values()
            .map(|e| e.size_bytes)
            .max()
            .unwrap_or(1);

        // Find the key with the minimum score (best eviction candidate).
        let victim_key = self
            .entries
            .iter()
            .map(|(k, e)| (k.clone(), self.score(e, max_age_ns, max_size)))
            .min_by(|(_, s1), (_, s2)| s1.partial_cmp(s2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k);

        if let Some(key) = victim_key {
            self.entries.remove(&key);
            self.evictions += 1;
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache(cap: usize) -> WeightedCache {
        WeightedCache::new(cap, WeightConfig::default())
    }

    // 1. New cache is empty
    #[test]
    fn test_new_cache_is_empty() {
        let cache = default_cache(8);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    // 2. insert + get returns value
    #[test]
    fn test_insert_and_get() {
        let mut cache = default_cache(4);
        cache.insert("key1", vec![1, 2, 3], CacheMediaType::Generic, 5);
        let val = cache.get("key1").expect("should find key1");
        assert_eq!(val, &[1u8, 2, 3]);
    }

    // 3. get on absent key returns None
    #[test]
    fn test_get_absent_returns_none() {
        let mut cache = default_cache(4);
        assert!(cache.get("absent").is_none());
    }

    // 4. hit/miss counters
    #[test]
    fn test_hit_miss_counters() {
        let mut cache = default_cache(4);
        cache.insert("k", vec![0], CacheMediaType::Generic, 1);
        let _ = cache.get("k");
        let _ = cache.get("missing");
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
    }

    // 5. capacity is respected — eviction happens
    #[test]
    fn test_capacity_eviction() {
        let mut cache = default_cache(3);
        cache.insert("a", vec![0; 100], CacheMediaType::VideoSegment, 3);
        cache.insert("b", vec![0; 100], CacheMediaType::VideoSegment, 3);
        cache.insert("c", vec![0; 100], CacheMediaType::VideoSegment, 3);
        cache.insert("d", vec![0; 100], CacheMediaType::VideoSegment, 3);
        assert_eq!(cache.len(), 3, "cache should still be at capacity");
        assert!(cache.evictions() > 0);
    }

    // 6. high-priority Manifest is preferred over low-priority Generic
    #[test]
    fn test_high_priority_survives_eviction() {
        let mut cfg = WeightConfig::new();
        // Manifest: heavily prioritised, low size penalty
        cfg.set_weights(
            CacheMediaType::Manifest,
            TypeWeights { recency: 0.1, priority: 0.85, size_penalty: 0.05 },
        );
        // Generic: low priority weight
        cfg.set_weights(
            CacheMediaType::Generic,
            TypeWeights { recency: 0.5, priority: 0.05, size_penalty: 0.45 },
        );
        let mut cache = WeightedCache::new(2, cfg);

        // Insert manifest with priority 255 and generic with priority 0.
        cache.insert("manifest", vec![0u8; 10], CacheMediaType::Manifest, 255);
        cache.insert("generic", vec![0u8; 10], CacheMediaType::Generic, 0);
        // Trigger eviction by inserting a third entry.
        cache.insert("third", vec![0u8; 10], CacheMediaType::Generic, 0);

        // The manifest should survive because it has the highest priority weight.
        assert!(
            cache.contains("manifest"),
            "Manifest should survive eviction"
        );
    }

    // 7. remove() works
    #[test]
    fn test_remove() {
        let mut cache = default_cache(4);
        cache.insert("k", vec![9], CacheMediaType::Metadata, 1);
        let removed = cache.remove("k");
        assert_eq!(removed, Some(vec![9]));
        assert!(!cache.contains("k"));
    }

    // 8. remove() on absent key returns None
    #[test]
    fn test_remove_absent() {
        let mut cache = default_cache(4);
        assert!(cache.remove("nope").is_none());
    }

    // 9. overwrite existing key
    #[test]
    fn test_overwrite_key() {
        let mut cache = default_cache(4);
        cache.insert("k", vec![1], CacheMediaType::Generic, 1);
        cache.insert("k", vec![2, 3], CacheMediaType::Generic, 5);
        assert_eq!(cache.len(), 1, "overwrite should not duplicate");
        let val = cache.get("k").expect("should exist");
        assert_eq!(val, &[2u8, 3]);
    }

    // 10. hit_rate calculation
    #[test]
    fn test_hit_rate() {
        let mut cache = default_cache(4);
        cache.insert("a", vec![0], CacheMediaType::Generic, 1);
        let _ = cache.get("a"); // hit
        let _ = cache.get("a"); // hit
        let _ = cache.get("b"); // miss
        // 2 hits / 3 total = 0.666…
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    // 11. resize() shrinks cache via eviction
    #[test]
    fn test_resize_shrinks() {
        let mut cache = default_cache(5);
        for i in 0..5u8 {
            cache.insert(format!("k{i}"), vec![i], CacheMediaType::Generic, i);
        }
        assert_eq!(cache.len(), 5);
        cache.resize(3);
        assert_eq!(cache.len(), 3);
    }

    // 12. WeightConfig::validate() catches negative weights
    #[test]
    fn test_validate_rejects_negative_weights() {
        let mut cfg = WeightConfig::new();
        cfg.default_weights = TypeWeights { recency: -0.1, priority: 0.5, size_penalty: 0.5 };
        assert!(cfg.validate().is_err());
    }

    // 13. WeightConfig::set_weights normalises to sum 1.0
    #[test]
    fn test_set_weights_normalises() {
        let mut cfg = WeightConfig::new();
        cfg.set_weights(
            CacheMediaType::Image,
            TypeWeights { recency: 2.0, priority: 2.0, size_penalty: 6.0 },
        );
        let w = cfg.weights_for(CacheMediaType::Image);
        let sum = w.recency + w.priority + w.size_penalty;
        assert!((sum - 1.0).abs() < 1e-9, "weights should normalise to 1.0, got {sum}");
    }

    // 14. clear() resets everything
    #[test]
    fn test_clear() {
        let mut cache = default_cache(4);
        cache.insert("x", vec![1], CacheMediaType::Image, 3);
        let _ = cache.get("x");
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
        assert_eq!(cache.evictions(), 0);
    }

    // 15. multiple media types coexist
    #[test]
    fn test_multiple_media_types() {
        let mut cache = default_cache(10);
        cache.insert("m", vec![0; 5], CacheMediaType::Manifest, 10);
        cache.insert("v", vec![0; 200], CacheMediaType::VideoSegment, 5);
        cache.insert("t", vec![0; 8], CacheMediaType::Thumbnail, 8);
        cache.insert("a", vec![0; 50], CacheMediaType::AudioSegment, 4);
        assert_eq!(cache.len(), 4);
    }

    // 16. evictions counter tracks total evictions
    #[test]
    fn test_evictions_counter() {
        let mut cache = default_cache(2);
        cache.insert("a", vec![0], CacheMediaType::Generic, 1);
        cache.insert("b", vec![0], CacheMediaType::Generic, 1);
        // c causes first eviction
        cache.insert("c", vec![0], CacheMediaType::Generic, 1);
        // d causes second eviction
        cache.insert("d", vec![0], CacheMediaType::Generic, 1);
        assert_eq!(cache.evictions(), 2);
    }

    // 17. capacity() returns configured value
    #[test]
    fn test_capacity_getter() {
        let cache = default_cache(42);
        assert_eq!(cache.capacity(), 42);
    }

    // 18. WeightConfig fallback to default for unknown types
    #[test]
    fn test_default_fallback_weights() {
        let cfg = WeightConfig::new(); // no overrides
        let w = cfg.weights_for(CacheMediaType::VideoSegment);
        let sum = w.recency + w.priority + w.size_penalty;
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
