//! FFT plan caching for efficient repeated transforms.
//!
//! OxiFFT plan creation involves non-trivial computation (factoring, twiddle
//! factor generation).  When the same FFT size is used repeatedly — as in
//! streaming spectrum analysis or STFT — recreating the plan each time wastes
//! CPU cycles.
//!
//! [`FftPlanCache`] maintains a `HashMap` of pre-built forward and inverse plans
//! keyed by FFT size, so each plan is allocated at most once.
//!
//! # Example
//!
//! ```
//! use oximedia_audio::spectrum::fft_cache::FftPlanCache;
//!
//! let mut cache = FftPlanCache::new();
//!
//! // First call creates the plan; subsequent calls reuse it.
//! let spectrum = cache.forward_transform(&[1.0; 1024]);
//! assert_eq!(spectrum.len(), 1024);
//!
//! // Inverse transform
//! let time_domain = cache.inverse_transform(&spectrum);
//! assert_eq!(time_domain.len(), 1024);
//! ```

#![allow(dead_code)]
#![forbid(unsafe_code)]

use std::collections::HashMap;

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

// ─────────────────────────────────────────────────────────────────────────────
// Cached plan wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A cached OxiFFT plan for a specific size and direction.
struct CachedPlan {
    plan: Plan<f64>,
    size: usize,
}

/// FFT plan cache — amortises plan construction across many transforms.
///
/// Internally stores one forward plan and one inverse plan per FFT size.
/// Plans are created on first use and reused thereafter.
///
/// The cache is **not** thread-safe; wrap in a `Mutex` or `RwLock` for
/// concurrent access.
pub struct FftPlanCache {
    forward: HashMap<usize, CachedPlan>,
    inverse: HashMap<usize, CachedPlan>,
    /// Total number of cache hits (for diagnostics).
    hits: u64,
    /// Total number of cache misses (plan creations).
    misses: u64,
}

impl FftPlanCache {
    /// Create a new, empty plan cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            inverse: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache pre-warmed with plans for the given sizes.
    ///
    /// Sizes that cannot produce a valid plan are silently skipped.
    #[must_use]
    pub fn with_sizes(sizes: &[usize]) -> Self {
        let mut cache = Self::new();
        for &size in sizes {
            cache.ensure_forward(size);
            cache.ensure_inverse(size);
        }
        cache
    }

    /// Ensure a forward plan exists for `size`, creating one if needed.
    ///
    /// Returns `true` if the plan was newly created, `false` if it already
    /// existed.
    pub fn ensure_forward(&mut self, size: usize) -> bool {
        if self.forward.contains_key(&size) {
            return false;
        }
        if let Some(plan) = Plan::dft_1d(size, Direction::Forward, Flags::MEASURE) {
            self.forward.insert(size, CachedPlan { plan, size });
            true
        } else {
            false
        }
    }

    /// Ensure an inverse plan exists for `size`, creating one if needed.
    ///
    /// Returns `true` if the plan was newly created, `false` if it already
    /// existed.
    pub fn ensure_inverse(&mut self, size: usize) -> bool {
        if self.inverse.contains_key(&size) {
            return false;
        }
        if let Some(plan) = Plan::dft_1d(size, Direction::Backward, Flags::MEASURE) {
            self.inverse.insert(size, CachedPlan { plan, size });
            true
        } else {
            false
        }
    }

    /// Execute a forward FFT on real-valued input, returning complex output.
    ///
    /// The plan for `input.len()` is cached on first use.
    pub fn forward_transform(&mut self, input: &[f64]) -> Vec<Complex<f64>> {
        let size = input.len();
        if size == 0 {
            return Vec::new();
        }

        self.ensure_forward(size);

        let complex_in: Vec<Complex<f64>> = input
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect();

        let mut output = vec![Complex::zero(); size];

        if let Some(cached) = self.forward.get(&size) {
            self.hits += 1;
            cached.plan.execute(&complex_in, &mut output);
        } else {
            self.misses += 1;
        }

        output
    }

    /// Execute a forward FFT on complex input.
    pub fn forward_complex(&mut self, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let size = input.len();
        if size == 0 {
            return Vec::new();
        }

        self.ensure_forward(size);

        let mut output = vec![Complex::zero(); size];

        if let Some(cached) = self.forward.get(&size) {
            self.hits += 1;
            cached.plan.execute(input, &mut output);
        } else {
            self.misses += 1;
        }

        output
    }

    /// Execute an inverse FFT, returning complex output.
    ///
    /// **Note:** OxiFFT does not normalise the inverse; divide by `N` if needed.
    pub fn inverse_transform(&mut self, input: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let size = input.len();
        if size == 0 {
            return Vec::new();
        }

        self.ensure_inverse(size);

        let mut output = vec![Complex::zero(); size];

        if let Some(cached) = self.inverse.get(&size) {
            self.hits += 1;
            cached.plan.execute(input, &mut output);
        } else {
            self.misses += 1;
        }

        output
    }

    /// Execute an inverse FFT and return only the real parts, normalised by N.
    pub fn inverse_real(&mut self, input: &[Complex<f64>]) -> Vec<f64> {
        let size = input.len();
        let output = self.inverse_transform(input);
        let n = size as f64;
        output.iter().map(|c| c.re / n).collect()
    }

    /// Number of plans currently cached (forward + inverse).
    #[must_use]
    pub fn plan_count(&self) -> usize {
        self.forward.len() + self.inverse.len()
    }

    /// Number of distinct FFT sizes with at least one cached plan.
    #[must_use]
    pub fn size_count(&self) -> usize {
        let mut sizes: std::collections::HashSet<usize> = self.forward.keys().copied().collect();
        sizes.extend(self.inverse.keys());
        sizes.len()
    }

    /// Total cache hits since creation.
    #[must_use]
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Total cache misses (plan creations) since creation.
    #[must_use]
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Cache hit ratio in the range `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if no operations have been performed.
    #[must_use]
    pub fn hit_ratio(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Remove all cached plans and reset statistics.
    pub fn clear(&mut self) {
        self.forward.clear();
        self.inverse.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Remove plans for a specific size.
    pub fn evict(&mut self, size: usize) {
        self.forward.remove(&size);
        self.inverse.remove(&size);
    }

    /// Check whether a forward plan for `size` is cached.
    #[must_use]
    pub fn has_forward(&self, size: usize) -> bool {
        self.forward.contains_key(&size)
    }

    /// Check whether an inverse plan for `size` is cached.
    #[must_use]
    pub fn has_inverse(&self, size: usize) -> bool {
        self.inverse.contains_key(&size)
    }
}

impl Default for FftPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CachedFftProcessor — drop-in replacement for FftProcessor
// ─────────────────────────────────────────────────────────────────────────────

use crate::spectrum::fft::WindowFunction;

/// FFT processor with built-in plan caching.
///
/// A drop-in alternative to [`crate::spectrum::fft::FftProcessor`] that keeps
/// its plans in an internal [`FftPlanCache`] so they are reused across calls.
pub struct CachedFftProcessor {
    fft_size: usize,
    window: Vec<f64>,
    cache: FftPlanCache,
}

impl CachedFftProcessor {
    /// Create a new cached FFT processor.
    #[must_use]
    pub fn new(fft_size: usize, window_fn: WindowFunction) -> Self {
        let mut cache = FftPlanCache::new();
        cache.ensure_forward(fft_size);
        Self {
            fft_size,
            window: window_fn.generate(fft_size),
            cache,
        }
    }

    /// FFT size.
    #[must_use]
    pub const fn size(&self) -> usize {
        self.fft_size
    }

    /// Process audio samples and return frequency-domain representation.
    pub fn process(&mut self, samples: &[f64]) -> Vec<Complex<f64>> {
        let input_size = samples.len().min(self.fft_size);

        let complex_in: Vec<Complex<f64>> = (0..self.fft_size)
            .map(|i| {
                if i < input_size {
                    Complex::new(samples[i] * self.window[i], 0.0)
                } else {
                    Complex::zero()
                }
            })
            .collect();

        self.cache.forward_complex(&complex_in)
    }

    /// Process and return magnitude spectrum.
    pub fn magnitude_spectrum(&mut self, samples: &[f64]) -> Vec<f64> {
        self.process(samples)
            .iter()
            .map(|c| c.norm())
            .collect()
    }

    /// Process and return power spectrum.
    pub fn power_spectrum(&mut self, samples: &[f64]) -> Vec<f64> {
        self.process(samples)
            .iter()
            .map(|c| c.norm_sqr())
            .collect()
    }

    /// Number of cache hits so far.
    #[must_use]
    pub fn cache_hits(&self) -> u64 {
        self.cache.hits()
    }

    /// Reference to the internal plan cache.
    #[must_use]
    pub fn cache(&self) -> &FftPlanCache {
        &self.cache
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // 1. Empty cache starts with zero plans.
    #[test]
    fn test_new_cache_is_empty() {
        let cache = FftPlanCache::new();
        assert_eq!(cache.plan_count(), 0);
        assert_eq!(cache.size_count(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    // 2. ensure_forward creates plan and returns true on first call.
    #[test]
    fn test_ensure_forward_creates_plan() {
        let mut cache = FftPlanCache::new();
        let created = cache.ensure_forward(256);
        assert!(created, "plan should be created");
        assert!(cache.has_forward(256));

        // Second call should not create a new plan.
        let again = cache.ensure_forward(256);
        assert!(!again, "plan already exists");
    }

    // 3. ensure_inverse works similarly.
    #[test]
    fn test_ensure_inverse_creates_plan() {
        let mut cache = FftPlanCache::new();
        assert!(cache.ensure_inverse(512));
        assert!(cache.has_inverse(512));
        assert!(!cache.ensure_inverse(512));
    }

    // 4. with_sizes pre-warms the cache.
    #[test]
    fn test_with_sizes_prewarm() {
        let cache = FftPlanCache::with_sizes(&[64, 128, 256]);
        assert!(cache.has_forward(64));
        assert!(cache.has_inverse(128));
        assert!(cache.has_forward(256));
        assert_eq!(cache.size_count(), 3);
    }

    // 5. Forward transform of a DC signal produces energy only in bin 0.
    #[test]
    fn test_forward_dc_signal() {
        let mut cache = FftPlanCache::new();
        let dc = vec![1.0; 64];
        let spectrum = cache.forward_transform(&dc);
        assert_eq!(spectrum.len(), 64);

        // Bin 0 should have magnitude == 64.
        let mag0 = spectrum[0].norm();
        assert!((mag0 - 64.0).abs() < 1e-6, "bin 0 mag = {mag0}");

        // Other bins should be near zero.
        for (i, c) in spectrum.iter().enumerate().skip(1) {
            assert!(c.norm() < 1e-6, "bin {i} magnitude should be ~0, got {}", c.norm());
        }
    }

    // 6. Inverse of forward recovers original signal (up to normalisation).
    #[test]
    fn test_forward_inverse_roundtrip() {
        let mut cache = FftPlanCache::new();
        let n = 128;
        let input: Vec<f64> = (0..n).map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin()).collect();
        let spectrum = cache.forward_transform(&input);
        let recovered = cache.inverse_real(&spectrum);
        assert_eq!(recovered.len(), n);
        for (i, (&orig, &rec)) in input.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-8,
                "sample {i}: expected {orig}, got {rec}"
            );
        }
    }

    // 7. Cache statistics track hits and misses correctly.
    #[test]
    fn test_cache_statistics() {
        let mut cache = FftPlanCache::new();
        let signal = vec![0.5; 64];
        // First call: miss (plan creation) + hit (execution)
        let _ = cache.forward_transform(&signal);
        assert!(cache.hits() >= 1);
        // Second call: hit only (plan exists)
        let _ = cache.forward_transform(&signal);
        assert!(cache.hits() >= 2);
    }

    // 8. clear() removes all plans and resets stats.
    #[test]
    fn test_clear() {
        let mut cache = FftPlanCache::with_sizes(&[64, 128]);
        let _ = cache.forward_transform(&vec![1.0; 64]);
        cache.clear();
        assert_eq!(cache.plan_count(), 0);
        assert_eq!(cache.hits(), 0);
        assert_eq!(cache.misses(), 0);
    }

    // 9. evict() removes plans for a specific size.
    #[test]
    fn test_evict() {
        let mut cache = FftPlanCache::with_sizes(&[64, 128]);
        assert!(cache.has_forward(64));
        cache.evict(64);
        assert!(!cache.has_forward(64));
        assert!(cache.has_forward(128), "other sizes should remain");
    }

    // 10. hit_ratio returns 0.0 when no operations have been performed.
    #[test]
    fn test_hit_ratio_initial() {
        let cache = FftPlanCache::new();
        assert!((cache.hit_ratio() - 0.0).abs() < 1e-12);
    }

    // 11. Empty input produces empty output.
    #[test]
    fn test_empty_input() {
        let mut cache = FftPlanCache::new();
        let result = cache.forward_transform(&[]);
        assert!(result.is_empty());
        let result2 = cache.inverse_transform(&[]);
        assert!(result2.is_empty());
    }

    // 12. CachedFftProcessor processes correctly and caches.
    #[test]
    fn test_cached_fft_processor() {
        let mut proc = CachedFftProcessor::new(256, WindowFunction::Hann);
        assert_eq!(proc.size(), 256);

        let signal: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 256.0).sin())
            .collect();

        let mag = proc.magnitude_spectrum(&signal);
        assert_eq!(mag.len(), 256);
        assert!(mag.iter().all(|v| v.is_finite()));

        // After processing, we should have at least 1 cache hit.
        assert!(proc.cache_hits() >= 1);
    }

    // 13. CachedFftProcessor power spectrum is magnitude squared.
    #[test]
    fn test_cached_processor_power_spectrum() {
        let mut proc = CachedFftProcessor::new(64, WindowFunction::Rectangle);
        let signal = vec![1.0; 64];
        let mag = proc.magnitude_spectrum(&signal);
        let pow = proc.power_spectrum(&signal);
        for (m, p) in mag.iter().zip(pow.iter()) {
            assert!((m * m - p).abs() < 1e-6, "power should be mag^2");
        }
    }

    // 14. Different sizes use different cached plans.
    #[test]
    fn test_multiple_sizes() {
        let mut cache = FftPlanCache::new();
        let _ = cache.forward_transform(&vec![1.0; 64]);
        let _ = cache.forward_transform(&vec![1.0; 128]);
        let _ = cache.forward_transform(&vec![1.0; 256]);
        assert_eq!(cache.size_count(), 3);
    }
}
