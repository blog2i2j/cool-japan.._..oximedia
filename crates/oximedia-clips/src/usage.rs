//! Clip usage tracking across sequences.
//!
//! [`ClipUsageTracker`] records every occurrence of a clip inside a sequence
//! and allows reverse-lookup (which sequences use a given clip).  This
//! enables features such as "find all sequences that reference this clip"
//! and conflict detection before replacing or deleting a clip.
//!
//! # Example
//!
//! ```
//! use oximedia_clips::usage::ClipUsageTracker;
//!
//! let mut tracker = ClipUsageTracker::new();
//! tracker.record_use(10, 1001);
//! tracker.record_use(10, 1002);
//! tracker.record_use(20, 1001);
//!
//! let seqs = tracker.usages_of(10);
//! assert!(seqs.contains(&1001));
//! assert!(seqs.contains(&1002));
//! assert!(!seqs.contains(&9999));
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// ClipUsageTracker
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks which sequences reference each clip.
///
/// Internally a `HashMap<clip_id, Vec<sequence_id>>` is maintained.
/// Duplicate `(clip_id, sequence_id)` pairs are allowed; use
/// [`usages_unique`](Self::usages_unique) to deduplicate.
#[derive(Debug, Default)]
pub struct ClipUsageTracker {
    /// Map from clip ID to the list of sequence IDs in which the clip appears.
    usages: HashMap<u64, Vec<u64>>,
}

impl ClipUsageTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self {
            usages: HashMap::new(),
        }
    }

    /// Record a single use of `clip_id` inside `sequence_id`.
    ///
    /// Duplicate calls with the same pair are accepted and appended; to
    /// deduplicate see [`usages_unique`](Self::usages_unique).
    pub fn record_use(&mut self, clip_id: u64, sequence_id: u64) {
        self.usages.entry(clip_id).or_default().push(sequence_id);
    }

    /// Return all sequence IDs (with possible duplicates) that reference
    /// `clip_id`.
    ///
    /// Returns an empty `Vec` when the clip has never been recorded.
    pub fn usages_of(&self, clip_id: u64) -> Vec<u64> {
        self.usages
            .get(&clip_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Return the deduplicated set of sequence IDs that reference `clip_id`.
    pub fn usages_unique(&self, clip_id: u64) -> Vec<u64> {
        let mut v = self.usages_of(clip_id);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Total number of recorded usages (with duplicates) for `clip_id`.
    pub fn use_count(&self, clip_id: u64) -> usize {
        self.usages.get(&clip_id).map(Vec::len).unwrap_or(0)
    }

    /// Return all clip IDs that have at least one recorded usage.
    pub fn used_clip_ids(&self) -> Vec<u64> {
        self.usages.keys().copied().collect()
    }

    /// Remove all usage records for `clip_id`.
    pub fn clear_clip(&mut self, clip_id: u64) {
        self.usages.remove(&clip_id);
    }

    /// Whether `clip_id` is used in any sequence.
    pub fn is_used(&self, clip_id: u64) -> bool {
        self.usages
            .get(&clip_id)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_is_empty() {
        let t = ClipUsageTracker::new();
        assert!(t.used_clip_ids().is_empty());
    }

    #[test]
    fn test_record_use_and_usages_of() {
        let mut t = ClipUsageTracker::new();
        t.record_use(10, 1001);
        t.record_use(10, 1002);
        let usages = t.usages_of(10);
        assert!(usages.contains(&1001));
        assert!(usages.contains(&1002));
    }

    #[test]
    fn test_usages_of_unknown_clip() {
        let t = ClipUsageTracker::new();
        assert!(t.usages_of(999).is_empty());
    }

    #[test]
    fn test_usages_unique_deduplicates() {
        let mut t = ClipUsageTracker::new();
        t.record_use(5, 100);
        t.record_use(5, 100); // duplicate
        t.record_use(5, 200);
        let unique = t.usages_unique(5);
        assert_eq!(unique.len(), 2);
        assert!(unique.contains(&100));
        assert!(unique.contains(&200));
    }

    #[test]
    fn test_use_count() {
        let mut t = ClipUsageTracker::new();
        t.record_use(1, 10);
        t.record_use(1, 10);
        t.record_use(1, 20);
        assert_eq!(t.use_count(1), 3);
    }

    #[test]
    fn test_use_count_zero_for_unknown() {
        let t = ClipUsageTracker::new();
        assert_eq!(t.use_count(42), 0);
    }

    #[test]
    fn test_is_used() {
        let mut t = ClipUsageTracker::new();
        assert!(!t.is_used(7));
        t.record_use(7, 100);
        assert!(t.is_used(7));
    }

    #[test]
    fn test_clear_clip() {
        let mut t = ClipUsageTracker::new();
        t.record_use(3, 10);
        t.record_use(3, 20);
        t.clear_clip(3);
        assert!(!t.is_used(3));
        assert!(t.usages_of(3).is_empty());
    }

    #[test]
    fn test_used_clip_ids_contains_all() {
        let mut t = ClipUsageTracker::new();
        t.record_use(1, 10);
        t.record_use(2, 20);
        t.record_use(3, 30);
        let ids = t.used_clip_ids();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
        assert!(ids.contains(&3));
    }

    #[test]
    fn test_multiple_clips_independent() {
        let mut t = ClipUsageTracker::new();
        t.record_use(1, 100);
        t.record_use(2, 200);
        assert_eq!(t.usages_of(1), vec![100]);
        assert_eq!(t.usages_of(2), vec![200]);
    }
}
