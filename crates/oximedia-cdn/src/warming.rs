//! Cache warming — pre-fetching popular content to edges before demand arrives.
//!
//! [`CacheWarmer`] maintains a priority queue of URLs to warm and allows
//! callers to schedule and dequeue them in priority order.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ── WarmEntry ──────────────────────────────────────────────────────────────

/// A single cache warming task.
#[derive(Debug, Clone, Eq, PartialEq)]
struct WarmEntry {
    /// Priority (higher number = higher priority).
    priority: u8,
    /// Stable insertion counter used to break ties (lower = inserted earlier).
    seq: u64,
    /// URL to warm.
    url: String,
}

impl Ord for WarmEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; on equal priority, earlier insertion first.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for WarmEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ── CacheWarmer ────────────────────────────────────────────────────────────

/// Priority-ordered cache warming scheduler.
///
/// # Example
/// ```
/// use oximedia_cdn::warming::CacheWarmer;
///
/// let mut warmer = CacheWarmer::new();
/// warmer.schedule("https://cdn.example.com/a.mp4", 10);
/// warmer.schedule("https://cdn.example.com/b.mp4", 5);
/// // Higher priority first
/// assert_eq!(warmer.warm_next().as_deref(), Some("https://cdn.example.com/a.mp4"));
/// assert_eq!(warmer.warm_next().as_deref(), Some("https://cdn.example.com/b.mp4"));
/// assert!(warmer.warm_next().is_none());
/// ```
#[derive(Debug, Default)]
pub struct CacheWarmer {
    heap: BinaryHeap<WarmEntry>,
    seq: u64,
}

impl CacheWarmer {
    /// Create an empty warmer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule a URL to be warmed with the given priority.
    ///
    /// Higher `priority` values are warmed first.
    pub fn schedule(&mut self, url: &str, priority: u8) {
        let seq = self.seq;
        self.seq += 1;
        self.heap.push(WarmEntry {
            priority,
            seq,
            url: url.to_string(),
        });
    }

    /// Dequeue and return the highest-priority URL, or `None` if empty.
    pub fn warm_next(&mut self) -> Option<String> {
        self.heap.pop().map(|e| e.url)
    }

    /// Number of URLs currently scheduled.
    pub fn pending_count(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if there are no pending warming tasks.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_warmer() {
        let mut w = CacheWarmer::new();
        assert!(w.warm_next().is_none());
        assert_eq!(w.pending_count(), 0);
        assert!(w.is_empty());
    }

    #[test]
    fn test_priority_order() {
        let mut w = CacheWarmer::new();
        w.schedule("low", 1);
        w.schedule("high", 10);
        w.schedule("mid", 5);
        assert_eq!(w.warm_next().as_deref(), Some("high"));
        assert_eq!(w.warm_next().as_deref(), Some("mid"));
        assert_eq!(w.warm_next().as_deref(), Some("low"));
    }

    #[test]
    fn test_equal_priority_fifo() {
        let mut w = CacheWarmer::new();
        w.schedule("first", 5);
        w.schedule("second", 5);
        // Earlier insertion should come first
        assert_eq!(w.warm_next().as_deref(), Some("first"));
        assert_eq!(w.warm_next().as_deref(), Some("second"));
    }

    #[test]
    fn test_pending_count() {
        let mut w = CacheWarmer::new();
        w.schedule("a", 1);
        w.schedule("b", 2);
        assert_eq!(w.pending_count(), 2);
        w.warm_next();
        assert_eq!(w.pending_count(), 1);
    }
}
