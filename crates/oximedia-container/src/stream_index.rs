//! Container stream index for fast, O(log n) seeking.
//!
//! A [`StreamIndex`] stores a sorted list of [`SeekPoint`] entries, each
//! associating a presentation timestamp (`pts`) with a byte offset inside the
//! container file.  The index also records whether each entry is a keyframe,
//! enabling keyframe-accurate seeking without scanning the entire file.
//!
//! # Building an index
//!
//! Use [`IndexBuilder`] to construct an index incrementally:
//!
//! ```
//! use oximedia_container::stream_index::{IndexBuilder, SeekPoint};
//!
//! let mut builder = IndexBuilder::new(1, 90_000); // stream 0, 90 kHz timescale
//! builder.add(0, 0, true);          // pts=0,  byte_pos=0,     keyframe
//! builder.add(3000, 4096, false);   // pts=3000, byte_pos=4096, non-keyframe
//! builder.add(6000, 8192, true);    // pts=6000, byte_pos=8192, keyframe
//!
//! let index = builder.build();
//! let kf = index.nearest_keyframe_before(5000).expect("should find keyframe");
//! assert_eq!(kf.pts, 0);
//! ```
//!
//! # Querying
//!
//! - [`StreamIndex::nearest_keyframe_before`] — O(log n) binary search for the
//!   last keyframe at or before the given PTS.
//! - [`StreamIndex::nearest_point`] — O(log n) for the closest point (keyframe
//!   or not) to the given PTS.
//! - [`StreamIndex::points_in_range`] — O(log n + k) for all points in a PTS
//!   window.

use std::cmp::Ordering;
use thiserror::Error;

/// Errors produced by stream-index operations.
#[derive(Debug, Error)]
pub enum StreamIndexError {
    /// The index contains no keyframes at or before the requested PTS.
    #[error("no keyframe found at or before pts={0}")]
    NoKeyframeBefore(i64),

    /// The index is completely empty.
    #[error("the stream index is empty")]
    Empty,
}

/// A single entry in a [`StreamIndex`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeekPoint {
    /// Byte offset of this point inside the container file.
    pub pos_bytes: u64,
    /// Presentation timestamp in timebase units.
    pub pts: i64,
    /// Numerator of the timebase fraction (pts × timebase_num / timebase_den = seconds).
    pub timebase_num: u32,
    /// Denominator of the timebase fraction.
    pub timebase_den: u32,
    /// `true` if the packet at this position is a keyframe (random-access point).
    pub is_keyframe: bool,
}

impl SeekPoint {
    /// Create a new seek point.
    #[must_use]
    pub const fn new(
        pts: i64,
        pos_bytes: u64,
        timebase_num: u32,
        timebase_den: u32,
        is_keyframe: bool,
    ) -> Self {
        Self {
            pos_bytes,
            pts,
            timebase_num,
            timebase_den,
            is_keyframe,
        }
    }

    /// Convert this point's PTS to seconds.
    ///
    /// Returns `None` if `timebase_den` is zero.
    #[must_use]
    pub fn pts_seconds(&self) -> Option<f64> {
        if self.timebase_den == 0 {
            return None;
        }
        Some(self.pts as f64 * self.timebase_num as f64 / self.timebase_den as f64)
    }
}

impl PartialOrd for SeekPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SeekPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        self.pts.cmp(&other.pts)
    }
}

/// A sorted, immutable collection of [`SeekPoint`] entries for one stream.
///
/// All points are kept in ascending PTS order.  Binary search is used for all
/// query operations, giving O(log n) time complexity.
#[derive(Clone, Debug, Default)]
pub struct StreamIndex {
    stream_id: usize,
    points: Vec<SeekPoint>,
}

impl StreamIndex {
    /// Create an empty index for the given stream.
    #[must_use]
    pub fn new(stream_id: usize) -> Self {
        Self {
            stream_id,
            points: Vec::new(),
        }
    }

    /// Return the stream ID this index belongs to.
    #[must_use]
    pub fn stream_id(&self) -> usize {
        self.stream_id
    }

    /// Return the number of entries in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return `true` if the index has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Return a slice of all seek points in PTS order.
    #[must_use]
    pub fn points(&self) -> &[SeekPoint] {
        &self.points
    }

    /// Add a [`SeekPoint`] and maintain sorted order.
    ///
    /// If a point with the same `pts` already exists it is replaced.
    pub fn add_point(&mut self, point: SeekPoint) {
        match self.points.binary_search_by_key(&point.pts, |p| p.pts) {
            Ok(i) => self.points[i] = point,
            Err(i) => self.points.insert(i, point),
        }
    }

    /// Find the nearest keyframe at or before `pts`.
    ///
    /// Performs a binary search to locate the insertion position for `pts`, then
    /// scans backwards for the last keyframe.
    ///
    /// # Errors
    ///
    /// Returns [`StreamIndexError::NoKeyframeBefore`] if no keyframe exists at
    /// or before `pts`, or [`StreamIndexError::Empty`] if the index is empty.
    pub fn nearest_keyframe_before(&self, pts: i64) -> Result<&SeekPoint, StreamIndexError> {
        if self.points.is_empty() {
            return Err(StreamIndexError::Empty);
        }

        // Find the index of the last point with pts <= target.
        let upper = match self.points.binary_search_by_key(&pts, |p| p.pts) {
            Ok(i) => i,
            Err(0) => return Err(StreamIndexError::NoKeyframeBefore(pts)),
            Err(i) => i - 1,
        };

        // Scan backwards for a keyframe.
        for idx in (0..=upper).rev() {
            if self.points[idx].is_keyframe {
                return Ok(&self.points[idx]);
            }
        }

        Err(StreamIndexError::NoKeyframeBefore(pts))
    }

    /// Find the point whose PTS is closest to `pts`.
    ///
    /// When two points are equidistant the earlier one (lower PTS) is returned.
    ///
    /// # Errors
    ///
    /// Returns [`StreamIndexError::Empty`] if the index has no entries.
    pub fn nearest_point(&self, pts: i64) -> Result<&SeekPoint, StreamIndexError> {
        if self.points.is_empty() {
            return Err(StreamIndexError::Empty);
        }

        match self.points.binary_search_by_key(&pts, |p| p.pts) {
            Ok(i) => Ok(&self.points[i]),
            Err(0) => Ok(&self.points[0]),
            Err(i) if i >= self.points.len() => self.points.last().ok_or(StreamIndexError::Empty),
            Err(i) => {
                let before = &self.points[i - 1];
                let after = &self.points[i];
                let dist_before = pts - before.pts;
                let dist_after = after.pts - pts;
                // Prefer the earlier point when equidistant.
                if dist_before <= dist_after {
                    Ok(before)
                } else {
                    Ok(after)
                }
            }
        }
    }

    /// Return all points whose PTS falls in the half-open interval `[start, end)`.
    ///
    /// Returns an empty slice when `start >= end` or no points lie in the range.
    #[must_use]
    pub fn points_in_range(&self, start: i64, end: i64) -> &[SeekPoint] {
        if start >= end || self.points.is_empty() {
            return &[];
        }

        // Binary search for the first index with pts >= start.
        let lo = match self.points.binary_search_by_key(&start, |p| p.pts) {
            Ok(i) => i,
            Err(i) => i,
        };

        // Binary search for the first index with pts >= end.
        let hi = match self.points.binary_search_by_key(&end, |p| p.pts) {
            Ok(i) => i,
            Err(i) => i,
        };

        &self.points[lo..hi]
    }

    /// Return the first seek point (lowest PTS).
    ///
    /// Returns `None` if the index is empty.
    #[must_use]
    pub fn first(&self) -> Option<&SeekPoint> {
        self.points.first()
    }

    /// Return the last seek point (highest PTS).
    ///
    /// Returns `None` if the index is empty.
    #[must_use]
    pub fn last(&self) -> Option<&SeekPoint> {
        self.points.last()
    }

    /// Count how many entries are keyframes.
    #[must_use]
    pub fn keyframe_count(&self) -> usize {
        self.points.iter().filter(|p| p.is_keyframe).count()
    }

    /// Estimate the average keyframe interval in PTS units.
    ///
    /// Returns `None` if there are fewer than two keyframes.
    #[must_use]
    pub fn avg_keyframe_interval(&self) -> Option<i64> {
        let kf_pts: Vec<i64> = self
            .points
            .iter()
            .filter(|p| p.is_keyframe)
            .map(|p| p.pts)
            .collect();
        if kf_pts.len() < 2 {
            return None;
        }
        let total: i64 = kf_pts.windows(2).map(|w| w[1] - w[0]).sum();
        Some(total / (kf_pts.len() as i64 - 1))
    }

    /// Merge another index into this one.
    ///
    /// Duplicate PTS values from `other` overwrite existing entries in `self`.
    pub fn merge(&mut self, other: &StreamIndex) {
        for point in &other.points {
            self.add_point(point.clone());
        }
    }
}

// ─── IndexBuilder ────────────────────────────────────────────────────────────

/// Incremental builder for a [`StreamIndex`].
///
/// Accepts `(pts, byte_pos, is_keyframe)` triples in any order and produces a
/// fully sorted [`StreamIndex`] when [`IndexBuilder::build`] is called.
#[derive(Debug)]
pub struct IndexBuilder {
    stream_id: usize,
    timebase_num: u32,
    timebase_den: u32,
    points: Vec<SeekPoint>,
}

impl IndexBuilder {
    /// Create a new builder for `stream_id` with the given timebase.
    ///
    /// `timebase_den` is the timebase denominator (e.g. 90_000 for 90 kHz).
    /// `timebase_num` is always 1 for standard media timebases; set it to
    /// a different value for rational timebases like 1001/30000.
    #[must_use]
    pub fn new(stream_id: usize, timebase_den: u32) -> Self {
        Self {
            stream_id,
            timebase_num: 1,
            timebase_den,
            points: Vec::new(),
        }
    }

    /// Create a builder with an explicit rational timebase `num/den`.
    #[must_use]
    pub fn with_timebase(stream_id: usize, timebase_num: u32, timebase_den: u32) -> Self {
        Self {
            stream_id,
            timebase_num,
            timebase_den,
            points: Vec::new(),
        }
    }

    /// Add an entry to the builder.
    ///
    /// Points may be added in any order; [`IndexBuilder::build`] will sort them.
    pub fn add(&mut self, pts: i64, pos_bytes: u64, is_keyframe: bool) {
        self.points.push(SeekPoint::new(
            pts,
            pos_bytes,
            self.timebase_num,
            self.timebase_den,
            is_keyframe,
        ));
    }

    /// Reserve capacity for `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        self.points.reserve(additional);
    }

    /// Consume the builder and return a sorted [`StreamIndex`].
    #[must_use]
    pub fn build(mut self) -> StreamIndex {
        self.points.sort_unstable_by_key(|p| p.pts);
        // Deduplicate: keep the last entry for each pts (later entries win).
        self.points.dedup_by(|a, b| {
            if a.pts == b.pts {
                // keep `b` (the earlier-encountered, i.e. first in original order after sort)
                // dedup_by keeps `b` when returning true
                *b = a.clone();
                true
            } else {
                false
            }
        });
        StreamIndex {
            stream_id: self.stream_id,
            points: self.points,
        }
    }

    /// Return the number of entries accumulated so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Return `true` if no entries have been added yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

// ─── Multi-stream index ───────────────────────────────────────────────────────

/// A collection of per-stream indices, keyed by stream ID.
#[derive(Clone, Debug, Default)]
pub struct MultiStreamIndex {
    indices: Vec<StreamIndex>,
}

impl MultiStreamIndex {
    /// Create a new empty multi-stream index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace the index for a stream.
    pub fn insert(&mut self, index: StreamIndex) {
        let id = index.stream_id();
        match self.indices.iter_mut().find(|idx| idx.stream_id() == id) {
            Some(existing) => *existing = index,
            None => self.indices.push(index),
        }
    }

    /// Return the index for `stream_id`, if present.
    #[must_use]
    pub fn get(&self, stream_id: usize) -> Option<&StreamIndex> {
        self.indices.iter().find(|idx| idx.stream_id() == stream_id)
    }

    /// Return the number of stream indices stored.
    #[must_use]
    pub fn stream_count(&self) -> usize {
        self.indices.len()
    }

    /// Iterate over all stream indices.
    pub fn iter(&self) -> impl Iterator<Item = &StreamIndex> {
        self.indices.iter()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_sample_index() -> StreamIndex {
        let mut b = IndexBuilder::new(0, 90_000);
        b.add(0, 0, true);
        b.add(3000, 4096, false);
        b.add(6000, 8192, true);
        b.add(9000, 12288, false);
        b.add(12000, 16384, false);
        b.add(15000, 20480, true);
        b.build()
    }

    #[test]
    fn test_builder_creates_sorted_index() {
        let mut b = IndexBuilder::new(0, 90_000);
        b.add(6000, 8192, true);
        b.add(0, 0, true);
        b.add(3000, 4096, false);
        let idx = b.build();
        let pts_list: Vec<i64> = idx.points().iter().map(|p| p.pts).collect();
        assert_eq!(pts_list, vec![0, 3000, 6000]);
    }

    #[test]
    fn test_nearest_keyframe_before_exact() {
        let idx = build_sample_index();
        let kf = idx.nearest_keyframe_before(6000).expect("keyframe found");
        assert_eq!(kf.pts, 6000);
        assert!(kf.is_keyframe);
    }

    #[test]
    fn test_nearest_keyframe_before_between() {
        let idx = build_sample_index();
        // pts=10000 is between kf@6000 and kf@15000 — should return kf@6000.
        let kf = idx.nearest_keyframe_before(10000).expect("keyframe found");
        assert_eq!(kf.pts, 6000);
    }

    #[test]
    fn test_nearest_keyframe_before_none() {
        let idx = build_sample_index();
        // pts = -1 is before the first entry.
        let result = idx.nearest_keyframe_before(-1);
        assert!(matches!(result, Err(StreamIndexError::NoKeyframeBefore(_))));
    }

    #[test]
    fn test_nearest_point() {
        let idx = build_sample_index();
        // Exact match.
        let p = idx.nearest_point(9000).expect("point found");
        assert_eq!(p.pts, 9000);
        // Midway between 3000 and 6000 (4500): equidistant, pick earlier.
        let p2 = idx.nearest_point(4500).expect("point found");
        assert_eq!(p2.pts, 3000);
        // Closer to 6000.
        let p3 = idx.nearest_point(5999).expect("point found");
        assert_eq!(p3.pts, 6000);
    }

    #[test]
    fn test_nearest_point_empty() {
        let idx = StreamIndex::new(0);
        assert!(matches!(idx.nearest_point(0), Err(StreamIndexError::Empty)));
    }

    #[test]
    fn test_points_in_range() {
        let idx = build_sample_index();
        let range = idx.points_in_range(3000, 12000);
        let pts_list: Vec<i64> = range.iter().map(|p| p.pts).collect();
        assert_eq!(pts_list, vec![3000, 6000, 9000]);
    }

    #[test]
    fn test_points_in_range_empty_when_start_ge_end() {
        let idx = build_sample_index();
        let range = idx.points_in_range(6000, 6000);
        assert!(range.is_empty());
    }

    #[test]
    fn test_add_point_overwrites_duplicate_pts() {
        let mut idx = StreamIndex::new(0);
        idx.add_point(SeekPoint::new(1000, 100, 1, 90_000, true));
        idx.add_point(SeekPoint::new(1000, 200, 1, 90_000, false));
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.points()[0].pos_bytes, 200);
        assert!(!idx.points()[0].is_keyframe);
    }

    #[test]
    fn test_keyframe_count() {
        let idx = build_sample_index();
        assert_eq!(idx.keyframe_count(), 3);
    }

    #[test]
    fn test_avg_keyframe_interval() {
        let idx = build_sample_index();
        // Keyframes at 0, 6000, 15000 → intervals: 6000, 9000 → avg = 7500.
        let avg = idx.avg_keyframe_interval().expect("at least 2 keyframes");
        assert_eq!(avg, 7500);
    }

    #[test]
    fn test_pts_seconds() {
        let p = SeekPoint::new(90_000, 0, 1, 90_000, true);
        let secs = p.pts_seconds().expect("valid timebase");
        assert!((secs - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge() {
        let mut a = StreamIndex::new(0);
        a.add_point(SeekPoint::new(0, 0, 1, 90_000, true));
        a.add_point(SeekPoint::new(3000, 100, 1, 90_000, false));

        let mut b = StreamIndex::new(0);
        b.add_point(SeekPoint::new(6000, 200, 1, 90_000, true));

        a.merge(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.last().expect("last").pts, 6000);
    }

    #[test]
    fn test_multi_stream_index() {
        let mut multi = MultiStreamIndex::new();

        let mut b0 = IndexBuilder::new(0, 90_000);
        b0.add(0, 0, true);
        multi.insert(b0.build());

        let mut b1 = IndexBuilder::new(1, 48_000);
        b1.add(0, 0, true);
        b1.add(1024, 512, false);
        multi.insert(b1.build());

        assert_eq!(multi.stream_count(), 2);
        let idx1 = multi.get(1).expect("stream 1 present");
        assert_eq!(idx1.len(), 2);
    }
}
