//! Pluggable eviction policy abstraction.
//!
//! This module defines an [`EvictionStrategy`] enum and a
//! [`create_eviction_fn`] factory that returns a boxed eviction closure.
//!
//! Each closure accepts a mutable `Vec<(K, u64)>` (a vector of `(key, access_time)`
//! pairs sorted or unsorted) and returns the key that should be evicted next.
//! The vector is not modified by the closure — callers are responsible for
//! removing the returned entry.
//!
//! # Policies
//!
//! | Policy | Behaviour |
//! |--------|-----------|
//! | `Lru`  | Evict the entry with the **smallest** (oldest) `access_time`. |
//! | `Lfu`  | Evict the entry with the **smallest** `access_time` (access count used as proxy). |
//! | `Fifo` | Evict the entry at **index 0** (oldest insertion position). |
//!
//! # Example
//!
//! ```
//! use oximedia_cache::eviction::{EvictionStrategy, create_eviction_fn};
//!
//! let evict = create_eviction_fn(EvictionStrategy::Lru);
//! let entries: Vec<(String, u64)> = vec![
//!     ("frame-001".to_string(), 1000),
//!     ("frame-002".to_string(), 500),
//!     ("frame-003".to_string(), 2000),
//! ];
//! let to_evict = evict(&entries);
//! assert_eq!(to_evict, Some("frame-002".to_string()));
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// EvictionStrategy
// ---------------------------------------------------------------------------

/// Discriminated union of the supported eviction strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionStrategy {
    /// Least Recently Used — evict the entry with the oldest access timestamp.
    Lru,
    /// Least Frequently Used — evict the entry with the lowest access count
    /// (represented here as the smallest access timestamp; in a real system
    /// a separate frequency counter would be maintained).
    Lfu,
    /// First In, First Out — evict the entry at position 0 in the slice.
    Fifo,
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a boxed eviction closure for the given `policy`.
///
/// The returned closure takes a `&Vec<(K, u64)>` and returns
/// `Option<K>` — the key that should be removed.
///
/// Returns `None` when the entry list is empty.
pub fn create_eviction_fn<K>(
    policy: EvictionStrategy,
) -> Box<dyn Fn(&Vec<(K, u64)>) -> Option<K>>
where
    K: Clone + Eq + 'static,
{
    match policy {
        EvictionStrategy::Lru | EvictionStrategy::Lfu => Box::new(|entries: &Vec<(K, u64)>| {
            entries
                .iter()
                .min_by_key(|(_, ts)| *ts)
                .map(|(k, _)| k.clone())
        }),
        EvictionStrategy::Fifo => Box::new(|entries: &Vec<(K, u64)>| {
            entries.first().map(|(k, _)| k.clone())
        }),
    }
}

// ---------------------------------------------------------------------------
// EvictionChooser — concrete helper wrapping the closure
// ---------------------------------------------------------------------------

/// A convenience wrapper around an eviction closure.
pub struct EvictionChooser<K: Clone + Eq + 'static> {
    /// The active strategy.
    pub strategy: EvictionStrategy,
    evict: Box<dyn Fn(&Vec<(K, u64)>) -> Option<K>>,
}

impl<K: Clone + Eq + 'static> EvictionChooser<K> {
    /// Create a new `EvictionChooser` for the given strategy.
    #[must_use]
    pub fn new(strategy: EvictionStrategy) -> Self {
        Self {
            strategy,
            evict: create_eviction_fn(strategy),
        }
    }

    /// Choose the next key to evict from `entries`.
    #[must_use]
    pub fn choose(&self, entries: &Vec<(K, u64)>) -> Option<K> {
        (self.evict)(entries)
    }
}

impl<K: Clone + Eq + std::fmt::Debug + 'static> std::fmt::Debug for EvictionChooser<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EvictionChooser")
            .field("strategy", &self.strategy)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    type Entries = Vec<(String, u64)>;

    fn e(key: &str, ts: u64) -> (String, u64) {
        (key.to_string(), ts)
    }

    // ── LRU ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lru_evicts_oldest() {
        let evict = create_eviction_fn::<String>(EvictionStrategy::Lru);
        let entries: Entries = vec![e("b", 1000), e("c", 500), e("a", 2000)];
        assert_eq!(evict(&entries), Some("c".to_string()));
    }

    #[test]
    fn test_lru_single_entry() {
        let evict = create_eviction_fn::<String>(EvictionStrategy::Lru);
        let entries: Entries = vec![e("only", 999)];
        assert_eq!(evict(&entries), Some("only".to_string()));
    }

    #[test]
    fn test_lru_empty_returns_none() {
        let evict = create_eviction_fn::<String>(EvictionStrategy::Lru);
        let entries: Entries = vec![];
        assert_eq!(evict(&entries), None);
    }

    // ── LFU ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lfu_evicts_lowest_count() {
        // LFU uses access_time as proxy for frequency (lowest = least frequent)
        let evict = create_eviction_fn::<String>(EvictionStrategy::Lfu);
        let entries: Entries = vec![e("hot", 100), e("cold", 1), e("warm", 50)];
        assert_eq!(evict(&entries), Some("cold".to_string()));
    }

    // ── FIFO ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_fifo_evicts_first_inserted() {
        let evict = create_eviction_fn::<String>(EvictionStrategy::Fifo);
        let entries: Entries = vec![e("first", 9999), e("second", 1), e("third", 5000)];
        // FIFO returns index 0 regardless of timestamp
        assert_eq!(evict(&entries), Some("first".to_string()));
    }

    #[test]
    fn test_fifo_empty_returns_none() {
        let evict = create_eviction_fn::<String>(EvictionStrategy::Fifo);
        let entries: Entries = vec![];
        assert_eq!(evict(&entries), None);
    }

    // ── EvictionChooser ────────────────────────────────────────────────────────

    #[test]
    fn test_eviction_chooser_lru() {
        let chooser: EvictionChooser<String> = EvictionChooser::new(EvictionStrategy::Lru);
        let entries: Entries = vec![e("x", 300), e("y", 100), e("z", 200)];
        assert_eq!(chooser.choose(&entries), Some("y".to_string()));
    }

    #[test]
    fn test_eviction_chooser_fifo() {
        let chooser: EvictionChooser<String> = EvictionChooser::new(EvictionStrategy::Fifo);
        let entries: Entries = vec![e("alpha", 1), e("beta", 2)];
        assert_eq!(chooser.choose(&entries), Some("alpha".to_string()));
    }

    #[test]
    fn test_eviction_chooser_strategy_field() {
        let chooser: EvictionChooser<String> = EvictionChooser::new(EvictionStrategy::Lfu);
        assert_eq!(chooser.strategy, EvictionStrategy::Lfu);
    }
}
