//! Pipeline object cache for the GPU crate.
//!
//! Avoids redundant pipeline compilation by keying compiled pipeline
//! binaries on a `u64` hash.  On a cache miss, a user-supplied closure
//! is called to produce the binary and the result is stored for subsequent
//! hits.
//!
//! The cache is entirely in-memory; for disk persistence see the
//! higher-level `shader_cache` module.
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::pipeline_cache::PipelineCache;
//!
//! let mut cache = PipelineCache::new();
//! let binary = cache.get_or_create(0xDEAD_BEEF, || vec![0x01, 0x02, 0x03]);
//! assert_eq!(binary, &[0x01, 0x02, 0x03]);
//! // Second call returns cached value without invoking the closure.
//! let again = cache.get_or_create(0xDEAD_BEEF, || panic!("should not be called"));
//! assert_eq!(again, &[0x01, 0x02, 0x03]);
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── PipelineCache ─────────────────────────────────────────────────────────────

/// In-memory cache mapping pipeline keys to compiled pipeline binaries.
///
/// `key` is a caller-defined `u64` (typically a hash of the pipeline
/// descriptor / shader source).  `value` is an opaque `Vec<u8>` that
/// represents the compiled pipeline object — format is backend-specific.
#[derive(Debug, Default)]
pub struct PipelineCache {
    entries: HashMap<u64, Vec<u8>>,
    /// Total number of cache lookups (hits + misses).
    pub total_lookups: u64,
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses (compilations triggered).
    pub misses: u64,
}

impl PipelineCache {
    /// Create a new, empty pipeline cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a reference to the cached binary for `key`, compiling and
    /// caching it first if not already present.
    ///
    /// `create_fn` is invoked **at most once** per unique `key`.
    pub fn get_or_create(&mut self, key: u64, create_fn: impl Fn() -> Vec<u8>) -> &Vec<u8> {
        self.total_lookups += 1;
        if !self.entries.contains_key(&key) {
            self.misses += 1;
            let binary = create_fn();
            self.entries.insert(key, binary);
        } else {
            self.hits += 1;
        }
        // Safety: we just inserted if absent.
        self.entries.get(&key).expect("entry was just inserted")
    }

    /// Return the cached binary for `key`, or `None` if not present.
    #[must_use]
    pub fn get(&self, key: u64) -> Option<&Vec<u8>> {
        self.entries.get(&key)
    }

    /// Remove a cached entry (e.g. when the shader source changes).
    pub fn invalidate(&mut self, key: u64) -> bool {
        self.entries.remove(&key).is_some()
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_lookups = 0;
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of cached pipelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hit ratio in `[0.0, 1.0]`.  Returns `0.0` if no lookups have occurred.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        if self.total_lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_lookups as f64
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_or_create_miss_then_hit() {
        let mut cache = PipelineCache::new();
        let b1 = cache.get_or_create(1, || vec![0xAA]);
        assert_eq!(b1, &[0xAA]);
        // Second call: closure returns different value but must not be invoked.
        let b2 = cache.get_or_create(1, || vec![0xBB]);
        assert_eq!(
            b2,
            &[0xAA],
            "cached value must be returned, not new closure result"
        );
        // Verify miss count is 1 (only one compilation).
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 1);
    }

    #[test]
    fn test_different_keys_stored_separately() {
        let mut cache = PipelineCache::new();
        cache.get_or_create(10, || vec![1]);
        cache.get_or_create(20, || vec![2]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(10), Some(&vec![1]));
        assert_eq!(cache.get(20), Some(&vec![2]));
    }

    #[test]
    fn test_invalidate_removes_entry() {
        let mut cache = PipelineCache::new();
        cache.get_or_create(42, || vec![0x42]);
        assert!(cache.invalidate(42));
        assert!(cache.get(42).is_none());
        assert!(!cache.invalidate(42), "already removed");
    }

    #[test]
    fn test_clear_resets_state() {
        let mut cache = PipelineCache::new();
        cache.get_or_create(1, || vec![1]);
        cache.get_or_create(2, || vec![2]);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.total_lookups, 0);
    }

    #[test]
    fn test_hit_ratio_calculation() {
        let mut cache = PipelineCache::new();
        cache.get_or_create(1, || vec![1]); // miss
        cache.get_or_create(1, || vec![1]); // hit
        cache.get_or_create(1, || vec![1]); // hit
        assert!((cache.hit_ratio() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_hit_ratio_no_lookups_returns_zero() {
        let cache = PipelineCache::new();
        assert_eq!(cache.hit_ratio(), 0.0);
    }

    #[test]
    fn test_empty_binary_cached() {
        let mut cache = PipelineCache::new();
        let b = cache.get_or_create(0, || vec![]);
        assert!(b.is_empty());
        assert_eq!(cache.len(), 1);
    }
}
