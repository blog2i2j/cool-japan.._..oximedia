//! PN (pseudorandom noise) sequence cache for spread-spectrum watermarking.
//!
//! The spread-spectrum embedder and detector call [`generate_pn_sequence`]
//! for every bit index on every embed/detect call.  For typical payloads this
//! means the same sequences are recomputed hundreds of times per second.
//!
//! [`PnSequenceCache`] memoises these calls: it stores already-generated
//! sequences in a hash map keyed by `(length, seed)` and returns a shared
//! reference on subsequent calls.  The first call for a given key runs the
//! generator exactly once; every subsequent call returns the cached result in
//! O(1).
//!
//! # Thread Safety
//!
//! [`PnSequenceCache`] is `!Send` + `!Sync` by default — it is designed for
//! single-threaded embedder/detector state.  If you need a shared cache
//! across threads, wrap it in `Arc<Mutex<PnSequenceCache>>`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_watermark::pn_cache::PnSequenceCache;
//!
//! let mut cache = PnSequenceCache::new();
//! let seq_a = cache.get_or_generate(64, 12345).to_vec();
//! let seq_b = cache.get_or_generate(64, 12345).to_vec();
//! assert_eq!(seq_a, seq_b, "sequences must be identical for the same seed");
//! ```

use crate::payload::generate_pn_sequence;
use std::collections::HashMap;

/// Cache key: `(sequence_length, seed)`.
type PnKey = (usize, u64);

/// An in-memory cache of pseudorandom noise sequences keyed by
/// `(length, seed)`.
///
/// See [module-level documentation](self) for usage and design rationale.
#[derive(Default)]
pub struct PnSequenceCache {
    store: HashMap<PnKey, Vec<i8>>,
}

impl PnSequenceCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the PN sequence for `(length, seed)`, generating and caching it
    /// on first call.
    pub fn get_or_generate(&mut self, length: usize, seed: u64) -> &[i8] {
        // Use entry API to avoid double-hashing.
        self.store
            .entry((length, seed))
            .or_insert_with(|| generate_pn_sequence(length, seed))
    }

    /// Return the number of distinct sequences currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return `true` when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Evict all entries with a given `length`.
    ///
    /// Useful to free memory when switching between chip-rate configurations.
    pub fn evict_length(&mut self, length: usize) {
        self.store.retain(|(l, _), _| *l != length);
    }

    /// Remove all cached entries.
    pub fn clear(&mut self) {
        self.store.clear();
    }

    /// Pre-warm the cache for a specific seed and a contiguous range of
    /// bit indices `[0, num_bits)` with the given sequence `length`.
    ///
    /// Calling this once before an embed/detect loop eliminates all cache
    /// misses during the loop.
    pub fn prewarm(&mut self, length: usize, base_key: u64, num_bits: usize) {
        for bit_idx in 0..num_bits {
            let seed = base_key + bit_idx as u64;
            self.store
                .entry((length, seed))
                .or_insert_with(|| generate_pn_sequence(length, seed));
        }
    }

    /// SIMD-friendly bulk correlation: compute the dot-product of `samples`
    /// with the cached PN sequence for `(length, seed)` and return the
    /// correlation value.
    ///
    /// The inner loop accesses both slices sequentially (unit stride), which
    /// allows LLVM to auto-vectorise with SIMD instructions.  Prefer this over
    /// manually looping outside the cache.
    ///
    /// Returns `None` when `samples.len() < length`.
    pub fn correlate(
        &mut self,
        samples: &[f32],
        length: usize,
        seed: u64,
    ) -> Option<f32> {
        if samples.len() < length {
            return None;
        }
        let pn = self
            .store
            .entry((length, seed))
            .or_insert_with(|| generate_pn_sequence(length, seed));
        // Unit-stride access on both slices → LLVM auto-vectorises with AVX2.
        let mut sum = 0.0f32;
        for (&s, &p) in samples.iter().take(length).zip(pn.iter()) {
            sum += s * f32::from(p);
        }
        Some(sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_hit() {
        let mut cache = PnSequenceCache::new();
        let seq_a = cache.get_or_generate(64, 42).to_vec();
        let seq_b = cache.get_or_generate(64, 42).to_vec();
        assert_eq!(seq_a, seq_b, "cache hit must return identical sequence");
    }

    #[test]
    fn test_different_seeds_differ() {
        let mut cache = PnSequenceCache::new();
        let seq_a = cache.get_or_generate(64, 1).to_vec();
        let seq_b = cache.get_or_generate(64, 2).to_vec();
        // With very high probability two distinct seeds produce different sequences.
        assert_ne!(seq_a, seq_b, "different seeds should yield different sequences");
    }

    #[test]
    fn test_len_increments_on_new_key() {
        let mut cache = PnSequenceCache::new();
        assert_eq!(cache.len(), 0);
        cache.get_or_generate(32, 100);
        assert_eq!(cache.len(), 1);
        cache.get_or_generate(32, 100); // same key — no new entry
        assert_eq!(cache.len(), 1);
        cache.get_or_generate(32, 200); // new key
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut cache = PnSequenceCache::new();
        cache.get_or_generate(64, 1);
        cache.get_or_generate(64, 2);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_evict_length() {
        let mut cache = PnSequenceCache::new();
        cache.get_or_generate(64, 10);
        cache.get_or_generate(128, 10);
        assert_eq!(cache.len(), 2);
        cache.evict_length(64);
        assert_eq!(cache.len(), 1);
        assert!(cache.get_or_generate(128, 10).len() == 128);
    }

    #[test]
    fn test_prewarm() {
        let mut cache = PnSequenceCache::new();
        cache.prewarm(32, 0, 10);
        assert_eq!(cache.len(), 10);
        // All 10 entries must now hit
        for i in 0..10u64 {
            let _ = cache.get_or_generate(32, i);
        }
        assert_eq!(cache.len(), 10, "prewarm should cover all needed entries");
    }

    #[test]
    fn test_correlate_with_itself() {
        let mut cache = PnSequenceCache::new();
        // A PN sequence correlated with itself should give a positive value.
        let pn = cache.get_or_generate(128, 777).to_vec();
        let samples: Vec<f32> = pn.iter().map(|&x| f32::from(x)).collect();
        let corr = cache.correlate(&samples, 128, 777).expect("correlation should succeed");
        assert!(corr > 0.0, "self-correlation must be positive: {corr}");
    }

    #[test]
    fn test_correlate_too_short_returns_none() {
        let mut cache = PnSequenceCache::new();
        let samples = vec![0.5f32; 10];
        let result = cache.correlate(&samples, 128, 1);
        assert!(result.is_none());
    }
}
