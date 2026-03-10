//! Caching optimization strategies for `OxiMedia`.
//!
//! Provides abstractions for analyzing memory access patterns and recommending
//! optimal cache configurations to improve encoder/decoder throughput.

#![allow(dead_code)]

use std::collections::VecDeque;

/// Describes the memory access pattern for a data stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessPattern {
    /// Data is accessed one element after another.
    Sequential,
    /// Data is accessed in an unpredictable order.
    Random,
    /// Data is accessed every N elements.
    Strided(u32),
    /// Data is accessed repeatedly within a short time window.
    Temporal,
}

impl AccessPattern {
    /// Returns the recommended prefetch distance (in cache lines) for this pattern.
    #[must_use]
    pub fn prefetch_distance(&self) -> usize {
        match self {
            Self::Sequential => 8,
            Self::Random => 0,
            Self::Strided(stride) => (*stride as usize).min(16),
            Self::Temporal => 2,
        }
    }
}

/// A hint used by the optimizer to schedule prefetches.
#[derive(Debug, Clone)]
pub struct CacheHint {
    /// Identifier for the cached item.
    pub key: String,
    /// Expected access pattern.
    pub pattern: AccessPattern,
    /// Priority level (0 = lowest, 255 = highest).
    pub priority: u8,
}

impl CacheHint {
    /// Returns `true` if this hint has a priority of 200 or above.
    #[must_use]
    pub fn is_high_priority(&self) -> bool {
        self.priority >= 200
    }
}

/// A bounded FIFO queue of `CacheHint` items.
#[derive(Debug)]
pub struct PrefetchQueue {
    hints: VecDeque<CacheHint>,
    max_size: usize,
}

impl PrefetchQueue {
    /// Creates a new `PrefetchQueue` with the given capacity.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            hints: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Pushes a hint onto the queue.
    ///
    /// If the queue is full, the push is silently ignored.
    pub fn push(&mut self, hint: CacheHint) {
        if !self.is_full() {
            self.hints.push_back(hint);
        }
    }

    /// Pops the oldest hint from the queue.
    pub fn pop(&mut self) -> Option<CacheHint> {
        self.hints.pop_front()
    }

    /// Returns `true` if the queue has reached its capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.hints.len() >= self.max_size
    }

    /// Returns the number of hints currently in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.hints.len()
    }

    /// Returns `true` if the queue contains no hints.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hints.is_empty()
    }
}

/// Estimates the benefit of caching for a given workload profile.
#[derive(Debug, Clone)]
pub struct CacheOptimizer {
    /// Observed cache hit rate in `[0.0, 1.0]`.
    pub hit_rate: f64,
    /// Average miss penalty in milliseconds.
    pub miss_penalty_ms: f64,
}

impl CacheOptimizer {
    /// Estimates the speedup factor from caching.
    ///
    /// `access_count` is the total number of memory accesses.
    /// `cacheable_pct` is the fraction (0–1) of accesses that are cacheable.
    ///
    /// Speedup = 1 / (1 − cacheable_pct × hit_rate × miss_penalty_ratio).
    #[must_use]
    pub fn expected_speedup(&self, access_count: u64, cacheable_pct: f64) -> f64 {
        if access_count == 0 || self.miss_penalty_ms <= 0.0 {
            return 1.0;
        }
        let cacheable = cacheable_pct.max(0.0).min(1.0);
        let hit = self.hit_rate.max(0.0).min(1.0);
        // Fraction of total time saved by cache hits
        let saved_fraction = cacheable * hit;
        1.0 / (1.0 - saved_fraction).max(f64::EPSILON)
    }

    /// Recommends a cache size in MiB given the working set size.
    ///
    /// Recommended cache = working_set × (1 + (1 − hit_rate)) to hold the
    /// working set plus extra for mis-predicted misses.
    #[must_use]
    pub fn recommended_cache_size_mb(&self, working_set_mb: f64) -> f64 {
        let miss_rate = 1.0 - self.hit_rate.max(0.0).min(1.0);
        working_set_mb * (1.0 + miss_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_pattern_prefetch_sequential() {
        assert_eq!(AccessPattern::Sequential.prefetch_distance(), 8);
    }

    #[test]
    fn test_access_pattern_prefetch_random() {
        assert_eq!(AccessPattern::Random.prefetch_distance(), 0);
    }

    #[test]
    fn test_access_pattern_prefetch_strided_small() {
        assert_eq!(AccessPattern::Strided(4).prefetch_distance(), 4);
    }

    #[test]
    fn test_access_pattern_prefetch_strided_large_capped() {
        assert_eq!(AccessPattern::Strided(100).prefetch_distance(), 16);
    }

    #[test]
    fn test_access_pattern_prefetch_temporal() {
        assert_eq!(AccessPattern::Temporal.prefetch_distance(), 2);
    }

    #[test]
    fn test_cache_hint_high_priority() {
        let h = CacheHint {
            key: "frame_buf".to_owned(),
            pattern: AccessPattern::Sequential,
            priority: 200,
        };
        assert!(h.is_high_priority());
    }

    #[test]
    fn test_cache_hint_not_high_priority() {
        let h = CacheHint {
            key: "metadata".to_owned(),
            pattern: AccessPattern::Random,
            priority: 50,
        };
        assert!(!h.is_high_priority());
    }

    #[test]
    fn test_prefetch_queue_push_pop() {
        let mut q = PrefetchQueue::new(4);
        q.push(CacheHint {
            key: "a".to_owned(),
            pattern: AccessPattern::Sequential,
            priority: 10,
        });
        assert_eq!(q.len(), 1);
        let hint = q.pop().expect("pop should return a value");
        assert_eq!(hint.key, "a");
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_prefetch_queue_is_full() {
        let mut q = PrefetchQueue::new(2);
        for i in 0..2 {
            q.push(CacheHint {
                key: i.to_string(),
                pattern: AccessPattern::Random,
                priority: 0,
            });
        }
        assert!(q.is_full());
        // Push to full queue is silently ignored
        q.push(CacheHint {
            key: "overflow".to_owned(),
            pattern: AccessPattern::Random,
            priority: 0,
        });
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_prefetch_queue_fifo_order() {
        let mut q = PrefetchQueue::new(10);
        for i in 0u8..3 {
            q.push(CacheHint {
                key: i.to_string(),
                pattern: AccessPattern::Sequential,
                priority: i,
            });
        }
        assert_eq!(q.pop().expect("pop should return a value").key, "0");
        assert_eq!(q.pop().expect("pop should return a value").key, "1");
        assert_eq!(q.pop().expect("pop should return a value").key, "2");
    }

    #[test]
    fn test_cache_optimizer_expected_speedup_no_penalty() {
        let opt = CacheOptimizer {
            hit_rate: 0.9,
            miss_penalty_ms: 0.0,
        };
        let s = opt.expected_speedup(1000, 0.8);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cache_optimizer_expected_speedup_positive() {
        let opt = CacheOptimizer {
            hit_rate: 0.8,
            miss_penalty_ms: 5.0,
        };
        let s = opt.expected_speedup(10_000, 0.6);
        assert!(s > 1.0, "expected speedup > 1, got {s}");
    }

    #[test]
    fn test_cache_optimizer_recommended_size() {
        let opt = CacheOptimizer {
            hit_rate: 0.75,
            miss_penalty_ms: 10.0,
        };
        // miss_rate = 0.25 → recommended = 100 * 1.25 = 125
        let size = opt.recommended_cache_size_mb(100.0);
        assert!((size - 125.0).abs() < 1e-4, "got {size}");
    }

    #[test]
    fn test_cache_optimizer_zero_accesses() {
        let opt = CacheOptimizer {
            hit_rate: 0.9,
            miss_penalty_ms: 10.0,
        };
        assert!((opt.expected_speedup(0, 0.5) - 1.0).abs() < 1e-9);
    }
}
