//! Cached parameter track with memoization for repeated same-time queries.
//!
//! [`CachedParameterTrack`] wraps a [`ParameterTrack`] and caches the most
//! recent evaluation results.  In typical VFX rendering, the same time value is
//! queried many times per frame (once per pixel row, once per effect in a
//! chain, etc.), so this cache avoids redundant binary-search + interpolation.
//!
//! The cache uses a small fixed-size ring buffer (LRU-like) that holds the last
//! N query results keyed by a quantised time value.  Cache hits are O(1);
//! misses fall through to the underlying `ParameterTrack::evaluate()`.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::{ParameterTrack, EasingFunction};
//! use oximedia_vfx::param_track_cache::CachedParameterTrack;
//!
//! let mut track = ParameterTrack::new();
//! track.add_keyframe(0.0, 0.0, EasingFunction::Linear);
//! track.add_keyframe(1.0, 100.0, EasingFunction::Linear);
//!
//! let mut cached = CachedParameterTrack::new(track);
//! let v1 = cached.evaluate(0.5);
//! let v2 = cached.evaluate(0.5); // cache hit
//! assert_eq!(v1, v2);
//! ```

use crate::{EasingFunction, ParameterTrack};
use serde::{Deserialize, Serialize};

/// Default number of cache slots in the ring buffer.
const DEFAULT_CACHE_SIZE: usize = 8;

/// Time quantisation precision: values within this epsilon are considered the
/// same time and will share a cache entry.
const TIME_EPSILON: f64 = 1e-9;

// ─────────────────────────────────────────────────────────────────────────────
// CacheEntry
// ─────────────────────────────────────────────────────────────────────────────

/// A single cached evaluation result.
#[derive(Debug, Clone, Copy)]
struct CacheEntry {
    /// Quantised time key.
    time: f64,
    /// Cached evaluation result.
    value: Option<f32>,
    /// Generation counter (for LRU eviction).
    generation: u64,
}

impl Default for CacheEntry {
    fn default() -> Self {
        Self {
            time: f64::NAN,
            value: None,
            generation: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CachedParameterTrack
// ─────────────────────────────────────────────────────────────────────────────

/// A parameter track with a fixed-size memoization cache.
///
/// Provides the same API as [`ParameterTrack`] but caches recent evaluations
/// for O(1) repeated lookups at the same time.
#[derive(Debug, Clone)]
pub struct CachedParameterTrack {
    inner: ParameterTrack,
    cache: Vec<CacheEntry>,
    generation: u64,
    /// Total number of evaluate() calls.
    total_queries: u64,
    /// Number of cache hits.
    cache_hits: u64,
}

impl CachedParameterTrack {
    /// Wrap a `ParameterTrack` with the default cache size.
    #[must_use]
    pub fn new(track: ParameterTrack) -> Self {
        Self::with_cache_size(track, DEFAULT_CACHE_SIZE)
    }

    /// Wrap with a custom cache size (minimum 1).
    #[must_use]
    pub fn with_cache_size(track: ParameterTrack, size: usize) -> Self {
        let size = size.max(1);
        Self {
            inner: track,
            cache: vec![CacheEntry::default(); size],
            generation: 0,
            total_queries: 0,
            cache_hits: 0,
        }
    }

    /// Evaluate the track at the given time, using the cache.
    #[must_use]
    pub fn evaluate(&mut self, time: f64) -> Option<f32> {
        self.total_queries += 1;

        // Check cache for a hit
        for entry in &self.cache {
            if (entry.time - time).abs() < TIME_EPSILON && entry.generation > 0 {
                self.cache_hits += 1;
                return entry.value;
            }
        }

        // Cache miss — compute
        let value = self.inner.evaluate(time);

        // Store in the least-recently-used slot
        self.generation += 1;
        let lru_idx = self
            .cache
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.generation)
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.cache[lru_idx] = CacheEntry {
            time,
            value,
            generation: self.generation,
        };

        value
    }

    /// Invalidate the cache (e.g. after adding/removing keyframes).
    pub fn invalidate(&mut self) {
        for entry in &mut self.cache {
            *entry = CacheEntry::default();
        }
        self.generation = 0;
    }

    /// Add a keyframe and invalidate the cache.
    pub fn add_keyframe(&mut self, time: f64, value: f32, easing: EasingFunction) {
        self.inner.add_keyframe(time, value, easing);
        self.invalidate();
    }

    /// Get the underlying track (read-only).
    #[must_use]
    pub fn inner(&self) -> &ParameterTrack {
        &self.inner
    }

    /// Get the underlying track (mutable).  Caller is responsible for calling
    /// `invalidate()` if keyframes are modified.
    pub fn inner_mut(&mut self) -> &mut ParameterTrack {
        &mut self.inner
    }

    /// Number of keyframes in the underlying track.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.inner.len()
    }

    /// Whether the underlying track is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Total number of `evaluate()` calls since creation.
    #[must_use]
    pub fn total_queries(&self) -> u64 {
        self.total_queries
    }

    /// Number of cache hits since creation.
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache_hits
    }

    /// Cache hit ratio (0.0 to 1.0).
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        if self.total_queries == 0 {
            0.0
        } else {
            self.cache_hits as f64 / self.total_queries as f64
        }
    }

    /// Current cache size (number of slots).
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Reset statistics counters.
    pub fn reset_stats(&mut self) {
        self.total_queries = 0;
        self.cache_hits = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BatchEvaluator
// ─────────────────────────────────────────────────────────────────────────────

/// Batch evaluator that evaluates multiple tracks at the same time efficiently.
///
/// When many tracks need to be evaluated at the same frame time, this avoids
/// per-track cache lookup overhead by evaluating all tracks in a single pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEvaluator {
    tracks: Vec<ParameterTrack>,
    names: Vec<String>,
}

impl BatchEvaluator {
    /// Create an empty batch evaluator.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tracks: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add a named track.
    pub fn add_track(&mut self, name: impl Into<String>, track: ParameterTrack) {
        self.names.push(name.into());
        self.tracks.push(track);
    }

    /// Evaluate all tracks at a single time, returning a vector of
    /// `(name, value)` pairs.
    #[must_use]
    pub fn evaluate_all(&self, time: f64) -> Vec<(&str, Option<f32>)> {
        self.names
            .iter()
            .zip(self.tracks.iter())
            .map(|(name, track)| (name.as_str(), track.evaluate(time)))
            .collect()
    }

    /// Number of tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether there are no tracks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Get a track by name.
    #[must_use]
    pub fn get_track(&self, name: &str) -> Option<&ParameterTrack> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|idx| &self.tracks[idx])
    }
}

impl Default for BatchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_track() -> ParameterTrack {
        let mut t = ParameterTrack::new();
        t.add_keyframe(0.0, 0.0, EasingFunction::Linear);
        t.add_keyframe(1.0, 100.0, EasingFunction::Linear);
        t
    }

    fn multi_keyframe_track() -> ParameterTrack {
        let mut t = ParameterTrack::new();
        t.add_keyframe(0.0, 0.0, EasingFunction::Linear);
        t.add_keyframe(0.5, 50.0, EasingFunction::EaseIn);
        t.add_keyframe(1.0, 100.0, EasingFunction::Linear);
        t.add_keyframe(2.0, 0.0, EasingFunction::EaseOut);
        t
    }

    // ── CachedParameterTrack ────────────────────────────────────────────

    #[test]
    fn test_cached_track_basic_evaluation() {
        let mut ct = CachedParameterTrack::new(linear_track());
        let v = ct.evaluate(0.5);
        assert!(v.is_some());
        let val = v.expect("value");
        assert!((val - 50.0).abs() < 1.0, "expected ~50, got {val}");
    }

    #[test]
    fn test_cached_track_cache_hit() {
        let mut ct = CachedParameterTrack::new(linear_track());
        let v1 = ct.evaluate(0.5);
        let v2 = ct.evaluate(0.5);
        assert_eq!(v1, v2);
        assert_eq!(ct.cache_hits(), 1);
        assert_eq!(ct.total_queries(), 2);
    }

    #[test]
    fn test_cached_track_multiple_times() {
        let mut ct = CachedParameterTrack::new(linear_track());
        let _ = ct.evaluate(0.0);
        let _ = ct.evaluate(0.25);
        let _ = ct.evaluate(0.5);
        let _ = ct.evaluate(0.75);
        let _ = ct.evaluate(1.0);
        assert_eq!(ct.total_queries(), 5);
        assert_eq!(ct.cache_hits(), 0); // All distinct times
    }

    #[test]
    fn test_cached_track_repeated_queries() {
        let mut ct = CachedParameterTrack::new(linear_track());
        for _ in 0..100 {
            ct.evaluate(0.5);
        }
        assert_eq!(ct.total_queries(), 100);
        assert_eq!(ct.cache_hits(), 99); // First is a miss
        assert!(ct.hit_ratio() > 0.98);
    }

    #[test]
    fn test_cached_track_invalidation() {
        let mut ct = CachedParameterTrack::new(linear_track());
        let v1 = ct.evaluate(0.5);
        ct.invalidate();
        let v2 = ct.evaluate(0.5);
        assert_eq!(v1, v2);
        // After invalidation, second call should be a cache miss
        assert_eq!(ct.cache_hits(), 0); // invalidation resets generation
    }

    #[test]
    fn test_cached_track_add_keyframe_invalidates() {
        let mut ct = CachedParameterTrack::new(linear_track());
        let _ = ct.evaluate(0.5);
        ct.add_keyframe(0.5, 75.0, EasingFunction::Linear);
        // After adding keyframe, cache is invalidated
        let v = ct.evaluate(0.5).expect("value");
        assert!((v - 75.0).abs() < 1.0, "should be ~75 after new keyframe, got {v}");
    }

    #[test]
    fn test_cached_track_empty() {
        let mut ct = CachedParameterTrack::new(ParameterTrack::new());
        assert!(ct.is_empty());
        assert!(ct.evaluate(0.5).is_none());
    }

    #[test]
    fn test_cached_track_single_keyframe() {
        let mut track = ParameterTrack::new();
        track.add_keyframe(0.0, 42.0, EasingFunction::Linear);
        let mut ct = CachedParameterTrack::new(track);
        // Single keyframe returns its value regardless of time
        let v = ct.evaluate(99.0).expect("value");
        assert!((v - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_cached_track_lru_eviction() {
        let mut ct = CachedParameterTrack::with_cache_size(linear_track(), 2);
        assert_eq!(ct.cache_size(), 2);

        // Fill cache with two entries
        ct.evaluate(0.0);
        ct.evaluate(1.0);
        // This should evict the LRU entry (0.0)
        ct.evaluate(0.5);
        // Query 0.0 again — should be a miss now
        let hits_before = ct.cache_hits();
        ct.evaluate(0.0);
        assert_eq!(ct.cache_hits(), hits_before, "0.0 should have been evicted");
    }

    #[test]
    fn test_cached_track_hit_ratio_zero_queries() {
        let ct = CachedParameterTrack::new(linear_track());
        assert_eq!(ct.hit_ratio(), 0.0);
    }

    #[test]
    fn test_cached_track_reset_stats() {
        let mut ct = CachedParameterTrack::new(linear_track());
        ct.evaluate(0.5);
        ct.evaluate(0.5);
        assert!(ct.total_queries() > 0);
        ct.reset_stats();
        assert_eq!(ct.total_queries(), 0);
        assert_eq!(ct.cache_hits(), 0);
    }

    #[test]
    fn test_cached_track_inner_access() {
        let ct = CachedParameterTrack::new(linear_track());
        assert_eq!(ct.keyframe_count(), 2);
        assert!(!ct.is_empty());
        assert_eq!(ct.inner().len(), 2);
    }

    // ── BatchEvaluator ──────────────────────────────────────────────────

    #[test]
    fn test_batch_evaluator_basic() {
        let mut batch = BatchEvaluator::new();
        batch.add_track("opacity", linear_track());
        batch.add_track("scale", multi_keyframe_track());
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_evaluator_evaluate_all() {
        let mut batch = BatchEvaluator::new();
        batch.add_track("opacity", linear_track());
        let results = batch.evaluate_all(0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "opacity");
        let val = results[0].1.expect("value");
        assert!((val - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_batch_evaluator_get_track() {
        let mut batch = BatchEvaluator::new();
        batch.add_track("opacity", linear_track());
        assert!(batch.get_track("opacity").is_some());
        assert!(batch.get_track("nonexistent").is_none());
    }

    #[test]
    fn test_batch_evaluator_empty() {
        let batch = BatchEvaluator::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        let results = batch.evaluate_all(0.5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_evaluator_multi_track() {
        let mut batch = BatchEvaluator::new();
        let mut t1 = ParameterTrack::new();
        t1.add_keyframe(0.0, 10.0, EasingFunction::Linear);
        t1.add_keyframe(1.0, 20.0, EasingFunction::Linear);

        let mut t2 = ParameterTrack::new();
        t2.add_keyframe(0.0, 100.0, EasingFunction::Linear);
        t2.add_keyframe(1.0, 200.0, EasingFunction::Linear);

        batch.add_track("a", t1);
        batch.add_track("b", t2);

        let results = batch.evaluate_all(0.5);
        let a_val = results[0].1.expect("a value");
        let b_val = results[1].1.expect("b value");
        assert!((a_val - 15.0).abs() < 1.0);
        assert!((b_val - 150.0).abs() < 1.0);
    }
}
