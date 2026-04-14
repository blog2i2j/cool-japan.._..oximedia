//! Sync point management for multi-camera timeline anchors.
//!
//! A *sync point* is a named timestamp anchor on the timeline used to
//! align camera angles, mark important events (slate clap, flash, audio
//! burst), or serve as chapter markers.
//!
//! [`SyncPointManager`] maintains an ordered collection of [`SyncPoint`]
//! values and provides nearest-match lookup in O(n) time, which is
//! sufficient for the typical number of sync points in a production
//! (tens to low hundreds).
//!
//! # Example
//!
//! ```rust
//! use oximedia_multicam::sync_points::SyncPointManager;
//!
//! let mut mgr = SyncPointManager::new();
//! mgr.add(0, "slate");
//! mgr.add(30_000, "scene_2");
//! mgr.add(90_000, "scene_3_end");
//!
//! let nearest = mgr.nearest(31_000).expect("should find nearest");
//! assert_eq!(nearest.label, "scene_2");
//! ```

/// A named timestamp on the multi-camera timeline.
#[derive(Debug, Clone, PartialEq)]
pub struct SyncPoint {
    /// Timestamp of the sync point (milliseconds since programme start).
    pub ts_ms: u64,
    /// Human-readable label describing this sync point.
    pub label: String,
}

impl SyncPoint {
    /// Create a new sync point.
    #[must_use]
    pub fn new(ts_ms: u64, label: impl Into<String>) -> Self {
        Self {
            ts_ms,
            label: label.into(),
        }
    }
}

/// Manages an ordered collection of [`SyncPoint`] values for a session.
#[derive(Debug, Default)]
pub struct SyncPointManager {
    points: Vec<SyncPoint>,
}

impl SyncPointManager {
    /// Create a new, empty manager.
    #[must_use]
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Add a sync point with the given timestamp and label.
    ///
    /// Points are kept sorted by timestamp; inserting at an already-existing
    /// timestamp is allowed (two sync points can share a millisecond).
    pub fn add(&mut self, ts_ms: u64, label: &str) {
        let sp = SyncPoint::new(ts_ms, label);
        let pos = self.points.partition_point(|p| p.ts_ms <= ts_ms);
        self.points.insert(pos, sp);
    }

    /// Remove a sync point by exact timestamp and label match.
    ///
    /// Returns `true` if a matching point was found and removed.
    pub fn remove(&mut self, ts_ms: u64, label: &str) -> bool {
        if let Some(pos) = self
            .points
            .iter()
            .position(|p| p.ts_ms == ts_ms && p.label == label)
        {
            self.points.remove(pos);
            true
        } else {
            false
        }
    }

    /// Return the sync point nearest to `ts_ms` in absolute time.
    ///
    /// Returns `None` if the manager is empty.
    #[must_use]
    pub fn nearest(&self, ts_ms: u64) -> Option<&SyncPoint> {
        self.points.iter().min_by_key(|p| {
            if p.ts_ms >= ts_ms {
                p.ts_ms - ts_ms
            } else {
                ts_ms - p.ts_ms
            }
        })
    }

    /// Return all sync points whose timestamps fall within `[start_ms, end_ms]`.
    #[must_use]
    pub fn in_range(&self, start_ms: u64, end_ms: u64) -> Vec<&SyncPoint> {
        self.points
            .iter()
            .filter(|p| p.ts_ms >= start_ms && p.ts_ms <= end_ms)
            .collect()
    }

    /// Return the total number of sync points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return `true` if no sync points have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterate over all sync points in timestamp order.
    #[must_use]
    pub fn iter(&self) -> std::slice::Iter<'_, SyncPoint> {
        self.points.iter()
    }

    /// Return the sync point immediately before `ts_ms`, if any.
    #[must_use]
    pub fn preceding(&self, ts_ms: u64) -> Option<&SyncPoint> {
        self.points
            .iter()
            .rev()
            .find(|p| p.ts_ms < ts_ms)
    }

    /// Return the sync point immediately at or after `ts_ms`, if any.
    #[must_use]
    pub fn following(&self, ts_ms: u64) -> Option<&SyncPoint> {
        self.points.iter().find(|p| p.ts_ms >= ts_ms)
    }

    /// Clear all sync points.
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_ordered() {
        let mut mgr = SyncPointManager::new();
        mgr.add(10_000, "b");
        mgr.add(0, "a");
        mgr.add(20_000, "c");
        let ts: Vec<u64> = mgr.iter().map(|p| p.ts_ms).collect();
        assert_eq!(ts, vec![0, 10_000, 20_000]);
    }

    #[test]
    fn test_nearest_exact() {
        let mut mgr = SyncPointManager::new();
        mgr.add(1_000, "one");
        mgr.add(5_000, "five");
        let sp = mgr.nearest(5_000).expect("should find nearest");
        assert_eq!(sp.label, "five");
    }

    #[test]
    fn test_nearest_between() {
        let mut mgr = SyncPointManager::new();
        mgr.add(0, "start");
        mgr.add(10_000, "mid");
        mgr.add(20_000, "end");
        // 4_000 is closer to 0 than to 10_000 (4_000 < 6_000).
        let sp = mgr.nearest(4_000).expect("should find nearest");
        assert_eq!(sp.label, "start");
        // 6_001 is closer to 10_000.
        let sp2 = mgr.nearest(6_001).expect("should find nearest");
        assert_eq!(sp2.label, "mid");
    }

    #[test]
    fn test_nearest_empty_returns_none() {
        let mgr = SyncPointManager::new();
        assert!(mgr.nearest(1_000).is_none());
    }

    #[test]
    fn test_in_range() {
        let mut mgr = SyncPointManager::new();
        mgr.add(0, "a");
        mgr.add(5_000, "b");
        mgr.add(10_000, "c");
        mgr.add(15_000, "d");
        let pts = mgr.in_range(4_000, 11_000);
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].label, "b");
        assert_eq!(pts[1].label, "c");
    }

    #[test]
    fn test_remove_existing() {
        let mut mgr = SyncPointManager::new();
        mgr.add(1_000, "x");
        assert!(mgr.remove(1_000, "x"));
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut mgr = SyncPointManager::new();
        mgr.add(1_000, "x");
        assert!(!mgr.remove(2_000, "x"));
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_preceding_and_following() {
        let mut mgr = SyncPointManager::new();
        mgr.add(0, "first");
        mgr.add(10_000, "second");
        mgr.add(20_000, "third");

        let pre = mgr.preceding(12_000).expect("should have preceding");
        assert_eq!(pre.label, "second");

        let fol = mgr.following(5_000).expect("should have following");
        assert_eq!(fol.label, "second");
    }

    #[test]
    fn test_clear() {
        let mut mgr = SyncPointManager::new();
        mgr.add(0, "a");
        mgr.add(1_000, "b");
        mgr.clear();
        assert!(mgr.is_empty());
    }

    #[test]
    fn test_len() {
        let mut mgr = SyncPointManager::new();
        assert_eq!(mgr.len(), 0);
        mgr.add(0, "a");
        mgr.add(1_000, "b");
        assert_eq!(mgr.len(), 2);
    }
}
