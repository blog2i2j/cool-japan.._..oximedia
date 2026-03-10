//! Marker range support: named frame ranges with color coding and overlap detection.

#![allow(dead_code)]

use std::collections::HashMap;

/// A named frame range with an optional color.
#[derive(Debug, Clone, PartialEq)]
pub struct MarkerRange {
    /// Unique identifier within a clip.
    pub id: u64,
    /// Human-readable name.
    pub name: String,
    /// First frame of the range (inclusive).
    pub start: i64,
    /// Last frame of the range (inclusive).
    pub end: i64,
    /// Color label in `#RRGGBB` hex format.
    pub color: Option<String>,
    /// Optional category tag.
    pub category: Option<String>,
}

impl MarkerRange {
    /// Create a new marker range.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `start > end`.
    #[must_use]
    pub fn new(id: u64, name: impl Into<String>, start: i64, end: i64) -> Self {
        debug_assert!(start <= end, "start must not exceed end");
        Self {
            id,
            name: name.into(),
            start,
            end,
            color: None,
            category: None,
        }
    }

    /// Duration of the range in frames.
    #[must_use]
    pub fn duration(&self) -> i64 {
        (self.end - self.start).max(0) + 1
    }

    /// Returns `true` if the given frame is within the range (inclusive).
    #[must_use]
    pub fn contains(&self, frame: i64) -> bool {
        frame >= self.start && frame <= self.end
    }

    /// Returns `true` if this range overlaps with `other`.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Set the color label.
    pub fn set_color(&mut self, color: impl Into<String>) {
        self.color = Some(color.into());
    }

    /// Set the category tag.
    pub fn set_category(&mut self, category: impl Into<String>) {
        self.category = Some(category.into());
    }
}

/// Collection of marker ranges for a single clip.
pub struct RangeCollection {
    ranges: HashMap<u64, MarkerRange>,
    next_id: u64,
}

impl RangeCollection {
    /// Create an empty collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ranges: HashMap::new(),
            next_id: 1,
        }
    }

    /// Add a new range and return its assigned ID.
    pub fn add(&mut self, name: impl Into<String>, start: i64, end: i64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.ranges
            .insert(id, MarkerRange::new(id, name, start, end));
        id
    }

    /// Remove a range by ID. Returns `true` if it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        self.ranges.remove(&id).is_some()
    }

    /// Get a range by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&MarkerRange> {
        self.ranges.get(&id)
    }

    /// Get a mutable reference to a range by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut MarkerRange> {
        self.ranges.get_mut(&id)
    }

    /// All ranges sorted by start frame.
    #[must_use]
    pub fn sorted(&self) -> Vec<&MarkerRange> {
        let mut v: Vec<&MarkerRange> = self.ranges.values().collect();
        v.sort_by_key(|r| r.start);
        v
    }

    /// Find all ranges that contain the given frame.
    #[must_use]
    pub fn at_frame(&self, frame: i64) -> Vec<&MarkerRange> {
        self.ranges.values().filter(|r| r.contains(frame)).collect()
    }

    /// Find all overlapping pairs.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<(u64, u64)> {
        let sorted = self.sorted();
        let mut pairs = Vec::new();
        for (i, a) in sorted.iter().enumerate() {
            for b in sorted.iter().skip(i + 1) {
                if a.overlaps(b) {
                    pairs.push((a.id, b.id));
                }
            }
        }
        pairs
    }

    /// Number of ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Returns `true` when there are no ranges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Export all ranges as CSV lines (`id,name,start,end,color`).
    #[must_use]
    pub fn to_csv(&self) -> String {
        let mut lines = vec!["id,name,start,end,color".to_string()];
        for r in self.sorted() {
            lines.push(format!(
                "{},{},{},{},{}",
                r.id,
                r.name,
                r.start,
                r.end,
                r.color.as_deref().unwrap_or("")
            ));
        }
        lines.join("\n")
    }
}

impl Default for RangeCollection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marker_range_new() {
        let r = MarkerRange::new(1, "Act 1", 0, 99);
        assert_eq!(r.start, 0);
        assert_eq!(r.end, 99);
        assert_eq!(r.name, "Act 1");
    }

    #[test]
    fn test_marker_range_duration() {
        let r = MarkerRange::new(1, "Scene", 10, 19);
        assert_eq!(r.duration(), 10); // inclusive: frames 10..=19
    }

    #[test]
    fn test_marker_range_duration_single_frame() {
        let r = MarkerRange::new(1, "Snap", 5, 5);
        assert_eq!(r.duration(), 1);
    }

    #[test]
    fn test_marker_range_contains() {
        let r = MarkerRange::new(1, "Range", 10, 20);
        assert!(r.contains(10));
        assert!(r.contains(15));
        assert!(r.contains(20));
        assert!(!r.contains(9));
        assert!(!r.contains(21));
    }

    #[test]
    fn test_marker_range_overlaps_yes() {
        let a = MarkerRange::new(1, "A", 0, 10);
        let b = MarkerRange::new(2, "B", 5, 15);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_marker_range_overlaps_adjacent() {
        let a = MarkerRange::new(1, "A", 0, 10);
        let b = MarkerRange::new(2, "B", 10, 20);
        // Frame 10 is shared => overlap
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_marker_range_no_overlap() {
        let a = MarkerRange::new(1, "A", 0, 9);
        let b = MarkerRange::new(2, "B", 10, 20);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_collection_add_and_get() {
        let mut col = RangeCollection::new();
        let id = col.add("Scene A", 0, 100);
        assert_eq!(col.len(), 1);
        let r = col.get(id).expect("get should succeed");
        assert_eq!(r.name, "Scene A");
    }

    #[test]
    fn test_collection_remove() {
        let mut col = RangeCollection::new();
        let id = col.add("X", 0, 10);
        assert!(col.remove(id));
        assert!(col.get(id).is_none());
        assert!(!col.remove(id)); // already gone
    }

    #[test]
    fn test_collection_at_frame() {
        let mut col = RangeCollection::new();
        col.add("A", 0, 50);
        col.add("B", 30, 100);
        col.add("C", 60, 90);

        let hits = col.at_frame(40);
        assert_eq!(hits.len(), 2); // A and B
    }

    #[test]
    fn test_collection_find_overlaps() {
        let mut col = RangeCollection::new();
        let id1 = col.add("A", 0, 50);
        let id2 = col.add("B", 40, 100);
        let _id3 = col.add("C", 200, 300);

        let pairs = col.find_overlaps();
        assert_eq!(pairs.len(), 1);
        let (a, b) = pairs[0];
        assert!((a == id1 && b == id2) || (a == id2 && b == id1));
    }

    #[test]
    fn test_collection_sorted() {
        let mut col = RangeCollection::new();
        col.add("Later", 100, 200);
        col.add("Earlier", 0, 50);
        let sorted = col.sorted();
        assert_eq!(sorted[0].name, "Earlier");
        assert_eq!(sorted[1].name, "Later");
    }

    #[test]
    fn test_collection_to_csv() {
        let mut col = RangeCollection::new();
        let id = col.add("MyRange", 10, 20);
        if let Some(r) = col.get_mut(id) {
            r.set_color("#FF0000");
        }
        let csv = col.to_csv();
        assert!(csv.contains("id,name,start,end,color"));
        assert!(csv.contains("MyRange"));
        assert!(csv.contains("#FF0000"));
    }

    #[test]
    fn test_collection_default() {
        let col: RangeCollection = Default::default();
        assert!(col.is_empty());
    }
}
