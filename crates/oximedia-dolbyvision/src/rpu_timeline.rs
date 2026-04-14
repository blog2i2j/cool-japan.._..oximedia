//! RPU metadata timeline editing.
//!
//! Provides facilities to insert, delete, and retime Dolby Vision RPU entries
//! in a timeline.  An RPU timeline is an ordered sequence of
//! [`TimedRpu`] entries, each associated with a presentation timestamp (PTS)
//! expressed in a configurable time base.
//!
//! # Example
//!
//! ```rust
//! use oximedia_dolbyvision::rpu_timeline::{RpuTimeline, TimedRpu, TimeBase};
//! use oximedia_dolbyvision::{DolbyVisionRpu, Profile};
//!
//! let mut timeline = RpuTimeline::new(TimeBase::new(1, 90000));
//!
//! // Insert some RPU entries
//! for pts in [0u64, 3000, 6000] {
//!     let rpu = DolbyVisionRpu::new(Profile::Profile8);
//!     timeline.insert(TimedRpu::new(pts, rpu));
//! }
//!
//! assert_eq!(timeline.len(), 3);
//!
//! // Delete the entry at PTS 3000
//! timeline.delete_at_pts(3000);
//! assert_eq!(timeline.len(), 2);
//!
//! // Retime: shift all entries by +1000 ticks
//! timeline.shift(1000);
//! assert_eq!(timeline.entry_at_index(0).map(|e| e.pts), Some(1000));
//! ```

#![forbid(unsafe_code)]

use crate::DolbyVisionRpu;

/// A time base expressed as a rational number (numerator / denominator).
///
/// For SMPTE 29.97 use `TimeBase::new(1001, 30000)`.
/// For 90 kHz MPEG timestamps use `TimeBase::new(1, 90000)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeBase {
    /// Numerator of the time base fraction.
    pub num: u64,
    /// Denominator of the time base fraction.
    pub den: u64,
}

impl TimeBase {
    /// Create a new time base.
    ///
    /// # Panics
    ///
    /// Panics if `den` is zero (would cause division by zero in `ticks_to_seconds`).
    #[must_use]
    pub fn new(num: u64, den: u64) -> Self {
        assert!(den > 0, "TimeBase denominator must be non-zero");
        Self { num, den }
    }

    /// Convert a tick count to seconds.
    #[must_use]
    pub fn ticks_to_seconds(&self, ticks: u64) -> f64 {
        (ticks as f64 * self.num as f64) / self.den as f64
    }

    /// Convert seconds to the nearest tick count.
    #[must_use]
    pub fn seconds_to_ticks(&self, seconds: f64) -> u64 {
        ((seconds * self.den as f64) / self.num as f64).round() as u64
    }
}

impl Default for TimeBase {
    /// Default time base is 90 kHz (common for MPEG-2/H.264/H.265 streams).
    fn default() -> Self {
        Self::new(1, 90_000)
    }
}

/// A single RPU entry associated with a presentation timestamp.
#[derive(Debug, Clone)]
pub struct TimedRpu {
    /// Presentation timestamp in the timeline's time base.
    pub pts: u64,
    /// The Dolby Vision RPU metadata for this timestamp.
    pub rpu: DolbyVisionRpu,
}

impl TimedRpu {
    /// Create a new timed RPU entry.
    #[must_use]
    pub fn new(pts: u64, rpu: DolbyVisionRpu) -> Self {
        Self { pts, rpu }
    }
}

/// Edit operation that can be applied to an [`RpuTimeline`].
#[derive(Debug, Clone)]
pub enum TimelineEdit {
    /// Insert a new RPU at the given PTS, replacing any existing entry at that PTS.
    Insert(TimedRpu),
    /// Delete the RPU entry at the given PTS (no-op if absent).
    DeleteAtPts(u64),
    /// Delete RPU entries inside the closed interval `[start_pts, end_pts]`.
    DeleteRange {
        /// Inclusive start PTS.
        start_pts: u64,
        /// Inclusive end PTS.
        end_pts: u64,
    },
    /// Shift all PTS values by adding `delta` (signed, saturating at 0).
    Shift(i64),
    /// Scale all PTS values by `numerator / denominator` (re-time).
    Scale {
        /// Scale numerator.
        numerator: u64,
        /// Scale denominator (must be non-zero).
        denominator: u64,
    },
    /// Replace all RPU entries in `[start_pts, end_pts]` with a single constant RPU.
    Flatten {
        /// Inclusive start PTS.
        start_pts: u64,
        /// Inclusive end PTS.
        end_pts: u64,
        /// The replacement RPU (will be cloned for each replaced entry).
        replacement: DolbyVisionRpu,
    },
}

/// Ordered timeline of Dolby Vision RPU metadata entries.
///
/// Entries are kept sorted by PTS in ascending order.  The timeline supports
/// insert, delete, retime, and bulk-edit operations.
#[derive(Debug, Clone)]
pub struct RpuTimeline {
    /// Ordered list of timed RPU entries (sorted by PTS ascending).
    entries: Vec<TimedRpu>,
    /// Time base for this timeline.
    pub time_base: TimeBase,
}

impl RpuTimeline {
    /// Create an empty timeline with the given time base.
    #[must_use]
    pub fn new(time_base: TimeBase) -> Self {
        Self {
            entries: Vec::new(),
            time_base,
        }
    }

    /// Return the number of entries in the timeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the timeline has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Insert a new timed RPU entry.
    ///
    /// If an entry already exists at the same PTS it is replaced.  Otherwise
    /// the entry is inserted such that the timeline stays sorted by PTS.
    pub fn insert(&mut self, entry: TimedRpu) {
        match self.entries.binary_search_by_key(&entry.pts, |e| e.pts) {
            Ok(idx) => {
                self.entries[idx] = entry;
            }
            Err(idx) => {
                self.entries.insert(idx, entry);
            }
        }
    }

    /// Delete the entry at an exact PTS value.
    ///
    /// Returns `true` if an entry was removed, `false` if no entry existed at
    /// that PTS.
    pub fn delete_at_pts(&mut self, pts: u64) -> bool {
        if let Ok(idx) = self.entries.binary_search_by_key(&pts, |e| e.pts) {
            self.entries.remove(idx);
            true
        } else {
            false
        }
    }

    /// Delete all entries in the closed interval `[start_pts, end_pts]`.
    ///
    /// Returns the number of entries removed.
    pub fn delete_range(&mut self, start_pts: u64, end_pts: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|e| e.pts < start_pts || e.pts > end_pts);
        before - self.entries.len()
    }

    /// Shift all PTS values by `delta` ticks.
    ///
    /// Negative deltas are applied with saturation at 0 so no PTS underflows.
    pub fn shift(&mut self, delta: i64) {
        for entry in &mut self.entries {
            if delta >= 0 {
                entry.pts = entry.pts.saturating_add(delta as u64);
            } else {
                entry.pts = entry.pts.saturating_sub(delta.unsigned_abs());
            }
        }
        // Re-sort in case saturation collapsed some entries
        self.entries.sort_unstable_by_key(|e| e.pts);
    }

    /// Re-time all entries by multiplying PTS by `numerator / denominator`.
    ///
    /// Useful for changing frame rate (e.g. 23.976 → 25 fps).
    ///
    /// # Panics
    ///
    /// Panics if `denominator` is zero.
    pub fn scale(&mut self, numerator: u64, denominator: u64) {
        assert!(denominator > 0, "scale denominator must be non-zero");
        for entry in &mut self.entries {
            entry.pts = entry.pts.saturating_mul(numerator) / denominator;
        }
        self.entries.sort_unstable_by_key(|e| e.pts);
    }

    /// Replace all entries in `[start_pts, end_pts]` with clones of `replacement`.
    ///
    /// The PTSes of the affected entries are preserved; only their RPU payloads
    /// are replaced.
    pub fn flatten(&mut self, start_pts: u64, end_pts: u64, replacement: &DolbyVisionRpu) {
        for entry in &mut self.entries {
            if entry.pts >= start_pts && entry.pts <= end_pts {
                entry.rpu = replacement.clone();
            }
        }
    }

    /// Apply a batch of edits in the order supplied.
    ///
    /// # Errors
    ///
    /// Returns an error string if a scale denominator is zero.
    pub fn apply_edits(&mut self, edits: &[TimelineEdit]) -> Result<(), String> {
        for edit in edits {
            match edit {
                TimelineEdit::Insert(entry) => self.insert(entry.clone()),
                TimelineEdit::DeleteAtPts(pts) => {
                    self.delete_at_pts(*pts);
                }
                TimelineEdit::DeleteRange {
                    start_pts,
                    end_pts,
                } => {
                    self.delete_range(*start_pts, *end_pts);
                }
                TimelineEdit::Shift(delta) => self.shift(*delta),
                TimelineEdit::Scale {
                    numerator,
                    denominator,
                } => {
                    if *denominator == 0 {
                        return Err("scale denominator must be non-zero".to_owned());
                    }
                    self.scale(*numerator, *denominator);
                }
                TimelineEdit::Flatten {
                    start_pts,
                    end_pts,
                    replacement,
                } => self.flatten(*start_pts, *end_pts, replacement),
            }
        }
        Ok(())
    }

    /// Return a reference to the entry at the given index, or `None`.
    #[must_use]
    pub fn entry_at_index(&self, index: usize) -> Option<&TimedRpu> {
        self.entries.get(index)
    }

    /// Return a mutable reference to the entry at the given index, or `None`.
    pub fn entry_at_index_mut(&mut self, index: usize) -> Option<&mut TimedRpu> {
        self.entries.get_mut(index)
    }

    /// Return a reference to the entry with the given PTS, or `None`.
    #[must_use]
    pub fn entry_at_pts(&self, pts: u64) -> Option<&TimedRpu> {
        self.entries
            .binary_search_by_key(&pts, |e| e.pts)
            .ok()
            .and_then(|idx| self.entries.get(idx))
    }

    /// Return the entry whose PTS is closest to `pts` (ties broken by lower PTS).
    #[must_use]
    pub fn nearest_entry(&self, pts: u64) -> Option<&TimedRpu> {
        if self.entries.is_empty() {
            return None;
        }
        match self.entries.binary_search_by_key(&pts, |e| e.pts) {
            Ok(idx) => self.entries.get(idx),
            Err(0) => self.entries.first(),
            Err(idx) if idx >= self.entries.len() => self.entries.last(),
            Err(idx) => {
                let prev = &self.entries[idx - 1];
                let next = &self.entries[idx];
                if pts.abs_diff(prev.pts) <= pts.abs_diff(next.pts) {
                    Some(prev)
                } else {
                    Some(next)
                }
            }
        }
    }

    /// Return an iterator over all entries in PTS order.
    pub fn iter(&self) -> std::slice::Iter<'_, TimedRpu> {
        self.entries.iter()
    }

    /// Return all entries in a PTS range as a slice.
    #[must_use]
    pub fn entries_in_range(&self, start_pts: u64, end_pts: u64) -> Vec<&TimedRpu> {
        self.entries
            .iter()
            .filter(|e| e.pts >= start_pts && e.pts <= end_pts)
            .collect()
    }

    /// Return the duration of the timeline in ticks (last PTS − first PTS).
    ///
    /// Returns `0` if the timeline has fewer than two entries.
    #[must_use]
    pub fn duration_ticks(&self) -> u64 {
        match (self.entries.first(), self.entries.last()) {
            (Some(first), Some(last)) if last.pts > first.pts => last.pts - first.pts,
            _ => 0,
        }
    }

    /// Return the duration of the timeline in seconds.
    #[must_use]
    pub fn duration_seconds(&self) -> f64 {
        self.time_base.ticks_to_seconds(self.duration_ticks())
    }
}

impl Default for RpuTimeline {
    fn default() -> Self {
        Self::new(TimeBase::default())
    }
}

impl<'a> IntoIterator for &'a RpuTimeline {
    type Item = &'a TimedRpu;
    type IntoIter = std::slice::Iter<'a, TimedRpu>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Profile;

    fn make_rpu() -> DolbyVisionRpu {
        DolbyVisionRpu::new(Profile::Profile8)
    }

    #[test]
    fn test_insert_and_len() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn test_insert_replaces_duplicate_pts() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(0, make_rpu())); // duplicate PTS
        assert_eq!(t.len(), 1, "duplicate PTS should replace, not append");
    }

    #[test]
    fn test_insert_maintains_order() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(6000, make_rpu()));
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        let pts: Vec<u64> = t.iter().map(|e| e.pts).collect();
        assert_eq!(pts, vec![0, 3000, 6000]);
    }

    #[test]
    fn test_delete_at_pts() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        let removed = t.delete_at_pts(3000);
        assert!(removed);
        assert_eq!(t.len(), 1);
        let not_removed = t.delete_at_pts(9999);
        assert!(!not_removed);
    }

    #[test]
    fn test_delete_range() {
        let mut t = RpuTimeline::default();
        for pts in [0u64, 1000, 2000, 3000, 4000] {
            t.insert(TimedRpu::new(pts, make_rpu()));
        }
        let count = t.delete_range(1000, 3000);
        assert_eq!(count, 3);
        assert_eq!(t.len(), 2);
        assert_eq!(t.entry_at_index(0).map(|e| e.pts), Some(0));
        assert_eq!(t.entry_at_index(1).map(|e| e.pts), Some(4000));
    }

    #[test]
    fn test_shift_positive() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        t.shift(1000);
        let pts: Vec<u64> = t.iter().map(|e| e.pts).collect();
        assert_eq!(pts, vec![1000, 4000]);
    }

    #[test]
    fn test_shift_negative_saturates_at_zero() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(500, make_rpu()));
        t.shift(-1000); // would go negative → clamp to 0
        assert_eq!(t.entry_at_index(0).map(|e| e.pts), Some(0));
    }

    #[test]
    fn test_scale() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(9000, make_rpu())); // 90000 ticks/s → 0.1 s
        t.scale(25, 30); // 30 fps → 25 fps
        let pts: Vec<u64> = t.iter().map(|e| e.pts).collect();
        assert_eq!(pts[0], 0);
        // 9000 * 25 / 30 = 7500
        assert_eq!(pts[1], 7500);
    }

    #[test]
    fn test_nearest_entry() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        t.insert(TimedRpu::new(6000, make_rpu()));

        let e = t.nearest_entry(2800).expect("should find nearest");
        assert_eq!(e.pts, 3000);

        let e2 = t.nearest_entry(100).expect("should find nearest");
        assert_eq!(e2.pts, 0);
    }

    #[test]
    fn test_flatten() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, DolbyVisionRpu::new(Profile::Profile5)));
        t.insert(TimedRpu::new(3000, DolbyVisionRpu::new(Profile::Profile5)));
        t.insert(TimedRpu::new(6000, DolbyVisionRpu::new(Profile::Profile5)));

        let replacement = DolbyVisionRpu::new(Profile::Profile8);
        t.flatten(0, 3000, &replacement);

        assert_eq!(
            t.entry_at_index(0).map(|e| e.rpu.profile),
            Some(Profile::Profile8)
        );
        assert_eq!(
            t.entry_at_index(1).map(|e| e.rpu.profile),
            Some(Profile::Profile8)
        );
        // Entry at 6000 should be unchanged
        assert_eq!(
            t.entry_at_index(2).map(|e| e.rpu.profile),
            Some(Profile::Profile5)
        );
    }

    #[test]
    fn test_apply_edits_batch() {
        let mut t = RpuTimeline::default();
        t.insert(TimedRpu::new(0, make_rpu()));
        t.insert(TimedRpu::new(3000, make_rpu()));
        t.insert(TimedRpu::new(6000, make_rpu()));

        let edits = vec![
            TimelineEdit::DeleteAtPts(3000),
            TimelineEdit::Shift(500),
        ];
        t.apply_edits(&edits).expect("edits should succeed");

        assert_eq!(t.len(), 2);
        assert_eq!(t.entry_at_index(0).map(|e| e.pts), Some(500));
    }

    #[test]
    fn test_apply_edits_zero_denominator_returns_error() {
        let mut t = RpuTimeline::default();
        let edits = vec![TimelineEdit::Scale {
            numerator: 1,
            denominator: 0,
        }];
        assert!(t.apply_edits(&edits).is_err());
    }

    #[test]
    fn test_duration_ticks() {
        let mut t = RpuTimeline::default();
        assert_eq!(t.duration_ticks(), 0);
        t.insert(TimedRpu::new(0, make_rpu()));
        assert_eq!(t.duration_ticks(), 0);
        t.insert(TimedRpu::new(9000, make_rpu()));
        assert_eq!(t.duration_ticks(), 9000);
    }

    #[test]
    fn test_time_base_conversion() {
        let tb = TimeBase::new(1, 90_000);
        assert!((tb.ticks_to_seconds(90_000) - 1.0).abs() < 1e-9);
        assert_eq!(tb.seconds_to_ticks(1.0), 90_000);
    }

    #[test]
    fn test_entries_in_range() {
        let mut t = RpuTimeline::default();
        for pts in [0u64, 1000, 2000, 3000] {
            t.insert(TimedRpu::new(pts, make_rpu()));
        }
        let in_range = t.entries_in_range(1000, 2000);
        assert_eq!(in_range.len(), 2);
    }
}
