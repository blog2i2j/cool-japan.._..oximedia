//! Cue / cluster position cache for fast random-access seeking in Matroska.
//!
//! Matroska's cue points (stored in the `Cues` element) map timestamps to
//! byte offsets within the `Segment`.  Parsing cue entries on every seek is
//! expensive for files with many cue points.  This module provides
//! [`CuePositionCache`], a sorted, in-memory cache that allows O(log n)
//! lookup of the nearest cue point before or at a given timestamp.
//!
//! # Usage
//!
//! 1. Populate the cache once during the initial `probe()` pass from the cue
//!    entries decoded by the Matroska parser.
//! 2. On every seek request, call [`CuePositionCache::find_cluster_before`] to
//!    obtain the byte offset of the nearest cluster, then seek the underlying
//!    `MediaSource` to that offset.

use std::collections::BTreeMap;

/// A single cue entry: timestamp → byte offset within the Matroska Segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CueEntry {
    /// Cue timestamp in the track's timescale (nanoseconds for Matroska).
    pub timestamp_ns: u64,
    /// Byte offset of the corresponding cluster from the start of the Segment.
    pub cluster_offset: u64,
    /// Optional track number that this cue entry applies to.
    pub track_number: Option<u64>,
}

impl CueEntry {
    /// Create a new cue entry.
    #[must_use]
    pub fn new(timestamp_ns: u64, cluster_offset: u64) -> Self {
        Self {
            timestamp_ns,
            cluster_offset,
            track_number: None,
        }
    }

    /// Attach an optional track number.
    #[must_use]
    pub fn with_track(mut self, track_number: u64) -> Self {
        self.track_number = Some(track_number);
        self
    }
}

/// In-memory cache of Matroska cue entries indexed by timestamp.
///
/// The cache holds one entry per unique timestamp (latest wins on conflict)
/// in a sorted `BTreeMap` so that predecessor lookups are O(log n).
#[derive(Debug, Clone, Default)]
pub struct CuePositionCache {
    /// Map from timestamp_ns → [`CueEntry`].
    entries: BTreeMap<u64, CueEntry>,
}

impl CuePositionCache {
    /// Create a new, empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a cue entry.
    pub fn insert(&mut self, entry: CueEntry) {
        self.entries.insert(entry.timestamp_ns, entry);
    }

    /// Bulk-insert from an iterator.
    pub fn extend(&mut self, entries: impl IntoIterator<Item = CueEntry>) {
        for entry in entries {
            self.insert(entry);
        }
    }

    /// Returns the number of cue entries in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Find the nearest cluster byte offset at or before `timestamp_ns`.
    ///
    /// Returns `None` when the cache is empty or `timestamp_ns` precedes all
    /// known cue points.
    #[must_use]
    pub fn find_cluster_before(&self, timestamp_ns: u64) -> Option<&CueEntry> {
        self.entries
            .range(..=timestamp_ns)
            .next_back()
            .map(|(_, v)| v)
    }

    /// Find the nearest cluster byte offset strictly after `timestamp_ns`.
    ///
    /// Useful for bidirectional seek validation or "next chapter" navigation.
    #[must_use]
    pub fn find_cluster_after(&self, timestamp_ns: u64) -> Option<&CueEntry> {
        self.entries
            .range((std::ops::Bound::Excluded(timestamp_ns), std::ops::Bound::Unbounded))
            .next()
            .map(|(_, v)| v)
    }

    /// Return the entry with the smallest timestamp (first cluster hint).
    #[must_use]
    pub fn first(&self) -> Option<&CueEntry> {
        self.entries.values().next()
    }

    /// Return the entry with the largest timestamp (last cluster hint).
    #[must_use]
    pub fn last(&self) -> Option<&CueEntry> {
        self.entries.values().next_back()
    }

    /// Iterate over all entries in ascending timestamp order.
    pub fn iter(&self) -> impl Iterator<Item = &CueEntry> {
        self.entries.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache() -> CuePositionCache {
        let mut c = CuePositionCache::new();
        // timestamps at 0 s, 1 s, 2 s, 5 s (in ns)
        for (ts, offset) in [(0, 1000), (1_000_000_000, 5000), (2_000_000_000, 12000), (5_000_000_000, 30000)] {
            c.insert(CueEntry::new(ts, offset));
        }
        c
    }

    #[test]
    fn test_cache_len() {
        let c = make_cache();
        assert_eq!(c.len(), 4);
    }

    #[test]
    fn test_find_before_exact() {
        let c = make_cache();
        let entry = c.find_cluster_before(1_000_000_000).expect("should find");
        assert_eq!(entry.cluster_offset, 5000);
    }

    #[test]
    fn test_find_before_between() {
        let c = make_cache();
        // 1.5 s — should return the 1 s cue entry
        let entry = c.find_cluster_before(1_500_000_000).expect("should find");
        assert_eq!(entry.cluster_offset, 5000);
    }

    #[test]
    fn test_find_before_at_start() {
        let c = make_cache();
        // Query before all cue points
        let entry = c.find_cluster_before(0);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().cluster_offset, 1000);
    }

    #[test]
    fn test_find_before_before_first() {
        let mut c = CuePositionCache::new();
        c.insert(CueEntry::new(1_000_000_000, 5000));
        // Query before the only cue point
        let entry = c.find_cluster_before(500_000_000);
        assert!(entry.is_none());
    }

    #[test]
    fn test_find_after() {
        let c = make_cache();
        let entry = c.find_cluster_after(1_000_000_000).expect("should find after");
        assert_eq!(entry.cluster_offset, 12000);
    }

    #[test]
    fn test_find_after_past_last() {
        let c = make_cache();
        let entry = c.find_cluster_after(5_000_000_000);
        assert!(entry.is_none());
    }

    #[test]
    fn test_first_and_last() {
        let c = make_cache();
        assert_eq!(c.first().map(|e| e.cluster_offset), Some(1000));
        assert_eq!(c.last().map(|e| e.cluster_offset), Some(30000));
    }

    #[test]
    fn test_empty_cache() {
        let c = CuePositionCache::new();
        assert!(c.is_empty());
        assert!(c.find_cluster_before(1_000_000_000).is_none());
    }

    #[test]
    fn test_insert_overwrites() {
        let mut c = CuePositionCache::new();
        c.insert(CueEntry::new(1_000_000_000, 5000));
        c.insert(CueEntry::new(1_000_000_000, 9999));
        assert_eq!(c.len(), 1);
        assert_eq!(c.find_cluster_before(1_000_000_000).unwrap().cluster_offset, 9999);
    }

    #[test]
    fn test_extend() {
        let mut c = CuePositionCache::new();
        c.extend([CueEntry::new(0, 0), CueEntry::new(1_000_000_000, 100)]);
        assert_eq!(c.len(), 2);
    }
}
