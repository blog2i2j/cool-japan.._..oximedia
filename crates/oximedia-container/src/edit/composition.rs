//! Composition time offsets.
//!
//! Handles composition time offsets (CTS) for B-frames and complex frame reordering.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};
use std::collections::HashMap;

/// Composition time offset for a sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompositionTimeOffset {
    /// Decode timestamp.
    pub dts: i64,
    /// Composition/presentation timestamp.
    pub pts: i64,
}

impl CompositionTimeOffset {
    /// Creates a new composition time offset.
    #[must_use]
    pub const fn new(dts: i64, pts: i64) -> Self {
        Self { dts, pts }
    }

    /// Returns the offset (pts - dts).
    #[must_use]
    pub const fn offset(&self) -> i64 {
        self.pts - self.dts
    }

    /// Returns true if PTS equals DTS (no offset).
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.pts == self.dts
    }
}

/// Composition time offset table for a track.
#[derive(Debug, Clone)]
pub struct CompositionTimeTable {
    offsets: Vec<CompositionTimeOffset>,
    sample_to_offset: HashMap<usize, usize>,
}

impl CompositionTimeTable {
    /// Creates a new composition time table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            offsets: Vec::new(),
            sample_to_offset: HashMap::new(),
        }
    }

    /// Adds a composition time offset for a sample.
    pub fn add_offset(&mut self, sample_index: usize, offset: CompositionTimeOffset) {
        let offset_index = self.offsets.len();
        self.offsets.push(offset);
        self.sample_to_offset.insert(sample_index, offset_index);
    }

    /// Gets the composition time offset for a sample.
    #[must_use]
    pub fn get_offset(&self, sample_index: usize) -> Option<&CompositionTimeOffset> {
        self.sample_to_offset
            .get(&sample_index)
            .and_then(|&offset_index| self.offsets.get(offset_index))
    }

    /// Returns all offsets.
    #[must_use]
    pub fn offsets(&self) -> &[CompositionTimeOffset] {
        &self.offsets
    }

    /// Returns true if all offsets are zero.
    #[must_use]
    pub fn all_zero(&self) -> bool {
        self.offsets.iter().all(CompositionTimeOffset::is_zero)
    }

    /// Clears all offsets.
    pub fn clear(&mut self) {
        self.offsets.clear();
        self.sample_to_offset.clear();
    }

    /// Returns the number of offsets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// Returns true if there are no offsets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }
}

impl Default for CompositionTimeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for composition time tables.
pub struct CompositionTimeBuilder {
    table: CompositionTimeTable,
    current_sample: usize,
}

impl CompositionTimeBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: CompositionTimeTable::new(),
            current_sample: 0,
        }
    }

    /// Adds a frame with specified DTS and PTS.
    pub fn add_frame(&mut self, dts: i64, pts: i64) -> &mut Self {
        let offset = CompositionTimeOffset::new(dts, pts);
        self.table.add_offset(self.current_sample, offset);
        self.current_sample += 1;
        self
    }

    /// Adds a frame with no offset (PTS == DTS).
    pub fn add_frame_no_offset(&mut self, timestamp: i64) -> &mut Self {
        self.add_frame(timestamp, timestamp)
    }

    /// Builds the composition time table.
    #[must_use]
    pub fn build(self) -> CompositionTimeTable {
        self.table
    }
}

impl Default for CompositionTimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for working with composition times.
pub struct CompositionTimeUtils;

impl CompositionTimeUtils {
    /// Calculates the maximum composition time offset in a table.
    #[must_use]
    pub fn max_offset(table: &CompositionTimeTable) -> Option<i64> {
        table
            .offsets()
            .iter()
            .map(CompositionTimeOffset::offset)
            .max()
    }

    /// Calculates the minimum composition time offset in a table.
    #[must_use]
    pub fn min_offset(table: &CompositionTimeTable) -> Option<i64> {
        table
            .offsets()
            .iter()
            .map(CompositionTimeOffset::offset)
            .min()
    }

    /// Checks if a table needs composition time offsets (has non-zero offsets).
    #[must_use]
    pub fn needs_ctts(table: &CompositionTimeTable) -> bool {
        !table.all_zero()
    }

    /// Validates that timestamps are monotonically increasing.
    ///
    /// # Errors
    ///
    /// Returns `Err` if decode timestamps are not monotonically increasing.
    pub fn validate_monotonic(table: &CompositionTimeTable) -> OxiResult<()> {
        let mut last_dts = None;

        for offset in table.offsets() {
            if let Some(prev_dts) = last_dts {
                if offset.dts < prev_dts {
                    return Err(OxiError::InvalidData(
                        "Decode timestamps are not monotonically increasing".into(),
                    ));
                }
            }
            last_dts = Some(offset.dts);
        }

        Ok(())
    }

    /// Generates a composition time table from separate DTS and PTS arrays.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the DTS and PTS arrays have different lengths.
    pub fn from_arrays(dts: &[i64], pts: &[i64]) -> OxiResult<CompositionTimeTable> {
        if dts.len() != pts.len() {
            return Err(OxiError::InvalidData(
                "DTS and PTS arrays must have the same length".into(),
            ));
        }

        let mut table = CompositionTimeTable::new();

        for (i, (&dts_val, &pts_val)) in dts.iter().zip(pts.iter()).enumerate() {
            table.add_offset(i, CompositionTimeOffset::new(dts_val, pts_val));
        }

        Ok(table)
    }

    /// Extracts DTS and PTS arrays from a composition time table.
    #[must_use]
    pub fn to_arrays(table: &CompositionTimeTable) -> (Vec<i64>, Vec<i64>) {
        let dts: Vec<i64> = table.offsets().iter().map(|o| o.dts).collect();
        let pts: Vec<i64> = table.offsets().iter().map(|o| o.pts).collect();
        (dts, pts)
    }
}

/// Frame reordering helper for streams with B-frames.
pub struct FrameReorderer {
    reorder_buffer: Vec<(usize, i64, i64)>, // (index, dts, pts)
    max_reorder: usize,
}

impl FrameReorderer {
    /// Creates a new frame reorderer.
    #[must_use]
    pub const fn new(max_reorder: usize) -> Self {
        Self {
            reorder_buffer: Vec::new(),
            max_reorder,
        }
    }

    /// Adds a frame to the reorder buffer.
    pub fn add_frame(&mut self, index: usize, dts: i64, pts: i64) {
        self.reorder_buffer.push((index, dts, pts));
    }

    /// Gets the next frame in presentation order.
    pub fn get_next_frame(&mut self) -> Option<(usize, i64, i64)> {
        if self.reorder_buffer.len() < self.max_reorder {
            return None;
        }

        // Sort by PTS and return the first one
        self.reorder_buffer.sort_by_key(|(_, _, pts)| *pts);
        Some(self.reorder_buffer.remove(0))
    }

    /// Flushes all remaining frames.
    pub fn flush(&mut self) -> Vec<(usize, i64, i64)> {
        self.reorder_buffer.sort_by_key(|(_, _, pts)| *pts);
        std::mem::take(&mut self.reorder_buffer)
    }

    /// Returns true if the buffer is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.reorder_buffer.len() >= self.max_reorder
    }

    /// Returns the current buffer size.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.reorder_buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_time_offset() {
        let offset = CompositionTimeOffset::new(1000, 1500);
        assert_eq!(offset.dts, 1000);
        assert_eq!(offset.pts, 1500);
        assert_eq!(offset.offset(), 500);
        assert!(!offset.is_zero());

        let zero = CompositionTimeOffset::new(1000, 1000);
        assert!(zero.is_zero());
    }

    #[test]
    fn test_composition_time_table() {
        let mut table = CompositionTimeTable::new();
        table.add_offset(0, CompositionTimeOffset::new(0, 0));
        table.add_offset(1, CompositionTimeOffset::new(1000, 2000));

        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());
        assert!(!table.all_zero());

        let offset = table.get_offset(1).expect("operation should succeed");
        assert_eq!(offset.dts, 1000);
        assert_eq!(offset.pts, 2000);
    }

    #[test]
    fn test_composition_time_builder() {
        let mut builder = CompositionTimeBuilder::new();
        builder.add_frame(0, 0);
        builder.add_frame(1000, 3000);
        builder.add_frame(2000, 1000);
        builder.add_frame(3000, 2000);

        let table = builder.build();
        assert_eq!(table.len(), 4);
    }

    #[test]
    fn test_composition_time_utils() {
        let mut table = CompositionTimeTable::new();
        table.add_offset(0, CompositionTimeOffset::new(0, 0));
        table.add_offset(1, CompositionTimeOffset::new(1000, 3000));
        table.add_offset(2, CompositionTimeOffset::new(2000, 1000));

        assert_eq!(CompositionTimeUtils::max_offset(&table), Some(2000));
        assert_eq!(CompositionTimeUtils::min_offset(&table), Some(-1000));
        assert!(CompositionTimeUtils::needs_ctts(&table));
    }

    #[test]
    fn test_validate_monotonic() {
        let mut table = CompositionTimeTable::new();
        table.add_offset(0, CompositionTimeOffset::new(0, 0));
        table.add_offset(1, CompositionTimeOffset::new(1000, 3000));
        table.add_offset(2, CompositionTimeOffset::new(2000, 1000));

        assert!(CompositionTimeUtils::validate_monotonic(&table).is_ok());

        // Non-monotonic DTS
        let mut bad_table = CompositionTimeTable::new();
        bad_table.add_offset(0, CompositionTimeOffset::new(1000, 1000));
        bad_table.add_offset(1, CompositionTimeOffset::new(500, 500));

        assert!(CompositionTimeUtils::validate_monotonic(&bad_table).is_err());
    }

    #[test]
    fn test_from_to_arrays() {
        let dts = vec![0, 1000, 2000];
        let pts = vec![0, 3000, 1000];

        let table =
            CompositionTimeUtils::from_arrays(&dts, &pts).expect("operation should succeed");
        assert_eq!(table.len(), 3);

        let (dts_out, pts_out) = CompositionTimeUtils::to_arrays(&table);
        assert_eq!(dts_out, dts);
        assert_eq!(pts_out, pts);
    }

    #[test]
    fn test_frame_reorderer() {
        let mut reorderer = FrameReorderer::new(3);

        reorderer.add_frame(0, 0, 0);
        reorderer.add_frame(1, 1000, 3000);
        reorderer.add_frame(2, 2000, 1000);

        let next = reorderer.get_next_frame();
        assert!(next.is_some());
        let (index, _, pts) = next.expect("operation should succeed");
        assert_eq!(index, 0);
        assert_eq!(pts, 0);

        let flushed = reorderer.flush();
        assert_eq!(flushed.len(), 2);
        assert_eq!(flushed[0].0, 2); // Frame with PTS 1000
        assert_eq!(flushed[1].0, 1); // Frame with PTS 3000
    }
}
