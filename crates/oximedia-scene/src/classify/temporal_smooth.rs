//! Temporal smoothing for classification results.
//!
//! Provides a generic [`TemporalSmoother`] that maintains a sliding window of
//! classification results and returns the most frequent class (mode) across the
//! window, which reduces flickering when a classifier alternates between adjacent
//! categories on consecutive frames.

use std::collections::VecDeque;

/// Generic temporal smoother that tracks the *mode* (most frequent element)
/// of a sliding window of classification results.
///
/// # Type parameter
///
/// `T` must implement `Clone`, `Eq`, and `Hash` so that window entries can be
/// duplicated and counted.
///
/// # Example
///
/// ```
/// use oximedia_scene::classify::temporal_smooth::TemporalSmoother;
///
/// let mut smoother: TemporalSmoother<&str> = TemporalSmoother::new(3);
/// smoother.push("cat");
/// smoother.push("dog");
/// smoother.push("cat");
/// assert_eq!(smoother.current_class(), Some(&"cat")); // "cat" appears twice
/// ```
#[derive(Debug, Clone)]
pub struct TemporalSmoother<T: Clone + Eq + std::hash::Hash> {
    /// Sliding window of recent classifications.
    window: VecDeque<T>,
    /// Maximum number of entries to keep.
    pub window_size: usize,
}

impl<T: Clone + Eq + std::hash::Hash> TemporalSmoother<T> {
    /// Create a new smoother with the given window size.
    ///
    /// A `window_size` of 1 disables smoothing (every push immediately becomes
    /// the current class).
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        let effective = window_size.max(1);
        Self {
            window: VecDeque::with_capacity(effective),
            window_size: effective,
        }
    }

    /// Push a new classification result into the window.
    ///
    /// If the window is already at capacity the oldest entry is dropped.
    pub fn push(&mut self, class: T) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(class);
    }

    /// Return a reference to the most frequent class in the current window
    /// (the *mode*).
    ///
    /// When multiple classes appear the same number of times, the one that was
    /// most recently pushed is returned.  Returns `None` when the window is
    /// empty.
    #[must_use]
    pub fn current_class(&self) -> Option<&T> {
        if self.window.is_empty() {
            return None;
        }

        // Count occurrences of each unique value, tracking the last occurrence
        // index so we can break ties in favour of the most recent entry.
        //
        // Strategy: iterate the window once, building a list of
        // (representative_ref, count, last_index) entries.  We identify
        // "same class" via Eq.
        let items: Vec<&T> = self.window.iter().collect();

        // List of (representative value ref, count, last_occurrence_index)
        let mut groups: Vec<(&T, usize, usize)> = Vec::new();

        for (i, item) in items.iter().enumerate() {
            if let Some(g) = groups.iter_mut().find(|(rep, _, _)| *rep == *item) {
                g.1 += 1;
                g.2 = i; // update last-occurrence index
            } else {
                groups.push((*item, 1, i));
            }
        }

        // Pick the group with the highest count; break ties by last_occurrence.
        groups
            .into_iter()
            .max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.2.cmp(&b.2)))
            .map(|(rep, _, _)| rep)
    }

    /// Return the number of entries currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Return `true` if the window contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Clear the window.
    pub fn clear(&mut self) {
        self.window.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_window_returns_none() {
        let smoother: TemporalSmoother<u8> = TemporalSmoother::new(4);
        assert_eq!(smoother.current_class(), None);
    }

    #[test]
    fn test_single_push_returns_that_class() {
        let mut smoother: TemporalSmoother<&str> = TemporalSmoother::new(5);
        smoother.push("indoor");
        assert_eq!(smoother.current_class(), Some(&"indoor"));
    }

    #[test]
    fn test_mode_is_most_frequent() {
        let mut smoother: TemporalSmoother<&str> = TemporalSmoother::new(5);
        smoother.push("outdoor");
        smoother.push("indoor");
        smoother.push("outdoor");
        smoother.push("outdoor");
        // "outdoor" appears 3 times, "indoor" once
        assert_eq!(smoother.current_class(), Some(&"outdoor"));
    }

    #[test]
    fn test_window_eviction() {
        // Window of size 3; after 4 pushes the oldest entry is gone
        let mut smoother: TemporalSmoother<u32> = TemporalSmoother::new(3);
        smoother.push(1); // will be evicted after 4th push
        smoother.push(2);
        smoother.push(2);
        smoother.push(3); // evicts 1; window = [2, 2, 3]
        assert_eq!(smoother.len(), 3);
        // "2" appears twice, "3" once
        assert_eq!(smoother.current_class(), Some(&2));
    }

    #[test]
    fn test_clear_resets_window() {
        let mut smoother: TemporalSmoother<i32> = TemporalSmoother::new(4);
        smoother.push(42);
        smoother.push(42);
        smoother.clear();
        assert!(smoother.is_empty());
        assert_eq!(smoother.current_class(), None);
    }

    #[test]
    fn test_window_size_one_immediate_result() {
        let mut smoother: TemporalSmoother<bool> = TemporalSmoother::new(1);
        smoother.push(true);
        assert_eq!(smoother.current_class(), Some(&true));
        smoother.push(false);
        assert_eq!(smoother.current_class(), Some(&false));
    }

    #[test]
    fn test_len_grows_then_stays_at_capacity() {
        let mut smoother: TemporalSmoother<u8> = TemporalSmoother::new(3);
        assert_eq!(smoother.len(), 0);
        smoother.push(1);
        assert_eq!(smoother.len(), 1);
        smoother.push(2);
        assert_eq!(smoother.len(), 2);
        smoother.push(3);
        assert_eq!(smoother.len(), 3);
        smoother.push(4); // evicts oldest
        assert_eq!(smoother.len(), 3);
    }
}
