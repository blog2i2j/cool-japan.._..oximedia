//! Star rating and colour label system for clips.
//!
//! [`ClipRating`] is a lightweight, in-memory store that associates each clip
//! ID with an optional **star rating** (0–5) and an optional **colour label**
//! (free-form string such as `"red"`, `"green"`, `"blue"`, `"yellow"`, …).
//!
//! Both attributes are independent: a clip may have a star rating without a
//! colour, a colour without stars, both, or neither.
//!
//! # Example
//!
//! ```
//! use oximedia_clips::rating_store::ClipRating;
//!
//! let mut rating = ClipRating::new();
//! rating.set_stars(1, 4);
//! rating.set_color(1, "green");
//!
//! let (stars, color) = rating.get(1).expect("clip 1 should have a rating");
//! assert_eq!(stars, 4);
//! assert_eq!(color, "green");
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Internal record
// ─────────────────────────────────────────────────────────────────────────────

/// Combined star + colour record for a single clip.
#[derive(Debug, Clone, Default)]
struct ClipRatingRecord {
    stars: Option<u8>,
    color: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// ClipRating
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory store for per-clip star ratings and colour labels.
#[derive(Debug, Default)]
pub struct ClipRating {
    records: HashMap<u64, ClipRatingRecord>,
}

impl ClipRating {
    /// Create an empty rating store.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
        }
    }

    /// Set the star rating for `clip_id`.
    ///
    /// Values above 5 are clamped to 5.  A star rating of 0 is valid (it
    /// represents an "unrated" or explicitly zero-star clip).
    pub fn set_stars(&mut self, clip_id: u64, stars: u8) {
        let clamped = stars.min(5);
        let rec = self.records.entry(clip_id).or_default();
        rec.stars = Some(clamped);
    }

    /// Set the colour label for `clip_id`.
    ///
    /// The value is stored as-is; callers are responsible for normalising the
    /// string (e.g. lowercasing).  Common values: `"red"`, `"orange"`,
    /// `"yellow"`, `"green"`, `"blue"`, `"purple"`, `"grey"`.
    pub fn set_color(&mut self, clip_id: u64, color: &str) {
        let rec = self.records.entry(clip_id).or_default();
        rec.color = Some(color.to_string());
    }

    /// Get the `(stars, color)` pair for `clip_id`.
    ///
    /// Returns `None` when neither a star rating nor a colour has been set.
    /// Returns `Some((stars, color_str))` when at least one attribute is set;
    /// missing attributes are represented as `0` stars and `""` colour
    /// respectively so the return signature is always a concrete value pair.
    pub fn get(&self, clip_id: u64) -> Option<(u8, &str)> {
        let rec = self.records.get(&clip_id)?;
        // Only return Some when the record actually exists (was ever set).
        let stars = rec.stars.unwrap_or(0);
        let color = rec.color.as_deref().unwrap_or("");
        Some((stars, color))
    }

    /// Return the star rating for `clip_id`, or `None` if not set.
    pub fn stars(&self, clip_id: u64) -> Option<u8> {
        self.records.get(&clip_id)?.stars
    }

    /// Return the colour label for `clip_id`, or `None` if not set.
    pub fn color(&self, clip_id: u64) -> Option<&str> {
        self.records.get(&clip_id)?.color.as_deref()
    }

    /// Remove all rating data for `clip_id`.
    pub fn clear(&mut self, clip_id: u64) {
        self.records.remove(&clip_id);
    }

    /// Number of clips with any rating data.
    pub fn rated_count(&self) -> usize {
        self.records.len()
    }

    /// Return the IDs of all clips that have a star rating of at least `min_stars`.
    pub fn clips_with_min_stars(&self, min_stars: u8) -> Vec<u64> {
        self.records
            .iter()
            .filter_map(|(id, rec)| {
                if rec.stars.unwrap_or(0) >= min_stars {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the IDs of all clips labelled with `color`.
    pub fn clips_with_color(&self, color: &str) -> Vec<u64> {
        self.records
            .iter()
            .filter_map(|(id, rec)| {
                if rec.color.as_deref() == Some(color) {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect()
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
        let r = ClipRating::new();
        assert_eq!(r.rated_count(), 0);
    }

    #[test]
    fn test_set_and_get_stars_and_color() {
        let mut r = ClipRating::new();
        r.set_stars(1, 4);
        r.set_color(1, "green");
        let (s, c) = r.get(1).expect("clip 1 should have a rating");
        assert_eq!(s, 4);
        assert_eq!(c, "green");
    }

    #[test]
    fn test_get_none_when_not_set() {
        let r = ClipRating::new();
        assert!(r.get(999).is_none());
    }

    #[test]
    fn test_stars_clamped_to_5() {
        let mut r = ClipRating::new();
        r.set_stars(1, 10);
        assert_eq!(r.stars(1), Some(5));
    }

    #[test]
    fn test_stars_zero_is_valid() {
        let mut r = ClipRating::new();
        r.set_stars(2, 0);
        assert_eq!(r.stars(2), Some(0));
    }

    #[test]
    fn test_stars_only_color_defaults_to_empty() {
        let mut r = ClipRating::new();
        r.set_stars(3, 3);
        let (_, color) = r.get(3).expect("should have record");
        assert_eq!(color, "");
    }

    #[test]
    fn test_color_only_stars_defaults_to_zero() {
        let mut r = ClipRating::new();
        r.set_color(4, "red");
        let (stars, _) = r.get(4).expect("should have record");
        assert_eq!(stars, 0);
    }

    #[test]
    fn test_stars_method() {
        let mut r = ClipRating::new();
        r.set_stars(5, 3);
        assert_eq!(r.stars(5), Some(3));
        assert!(r.stars(999).is_none());
    }

    #[test]
    fn test_color_method() {
        let mut r = ClipRating::new();
        r.set_color(6, "blue");
        assert_eq!(r.color(6), Some("blue"));
        assert!(r.color(999).is_none());
    }

    #[test]
    fn test_clear() {
        let mut r = ClipRating::new();
        r.set_stars(7, 5);
        r.clear(7);
        assert!(r.get(7).is_none());
    }

    #[test]
    fn test_clips_with_min_stars() {
        let mut r = ClipRating::new();
        r.set_stars(1, 5);
        r.set_stars(2, 3);
        r.set_stars(3, 1);
        let top = r.clips_with_min_stars(4);
        assert!(top.contains(&1));
        assert!(!top.contains(&2));
        assert!(!top.contains(&3));
    }

    #[test]
    fn test_clips_with_color() {
        let mut r = ClipRating::new();
        r.set_color(10, "red");
        r.set_color(11, "green");
        r.set_color(12, "red");
        let red = r.clips_with_color("red");
        assert_eq!(red.len(), 2);
        assert!(red.contains(&10));
        assert!(red.contains(&12));
        assert!(!red.contains(&11));
    }

    #[test]
    fn test_rated_count() {
        let mut r = ClipRating::new();
        r.set_stars(1, 3);
        r.set_stars(2, 4);
        assert_eq!(r.rated_count(), 2);
    }
}
