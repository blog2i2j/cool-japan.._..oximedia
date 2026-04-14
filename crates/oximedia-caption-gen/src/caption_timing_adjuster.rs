//! Caption timing adjustment: shift and stretch caption timecodes to match
//! edited or re-timed video content.
//!
//! ## Use cases
//!
//! - **Offset shift**: the video was trimmed at the beginning; all captions
//!   need to move earlier by N ms.
//! - **Linear stretch/compression**: the video was speed-adjusted; captions
//!   need their start/end times scaled by a factor.
//! - **Segment remap**: a list of edit-decision-list (EDL) cuts maps old
//!   time ranges to new ones; captions that span a cut are either split or
//!   dropped.
//! - **Snap-to-frame**: timestamps are quantised to the nearest frame boundary
//!   at a given frame rate.
//!
//! All operations are lossless with respect to caption text; only timestamps
//! are modified.

use crate::alignment::CaptionBlock;

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors produced by timing-adjustment operations.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TimingAdjusterError {
    /// A stretch factor of 0.0 or negative was supplied.
    #[error("stretch factor must be positive (got {factor})")]
    InvalidStretchFactor { factor: f64 },

    /// A frame rate ≤ 0.0 was supplied to the snap-to-frame operation.
    #[error("frame rate must be positive (got {fps})")]
    InvalidFrameRate { fps: f32 },

    /// An EDL entry has an invalid time range (start_ms >= end_ms).
    #[error("EDL entry has invalid range: [{src_start_ms}, {src_end_ms})")]
    InvalidEdlRange { src_start_ms: u64, src_end_ms: u64 },
}

// ─── Timing adjuster ─────────────────────────────────────────────────────────

/// Adjusts caption block timestamps.
pub struct CaptionTimingAdjuster;

impl CaptionTimingAdjuster {
    // ─── Offset shift ──────────────────────────────────────────────────────────

    /// Shift all caption timestamps by `offset_ms`.
    ///
    /// A positive offset moves captions later in the timeline; a negative
    /// offset moves them earlier.  Blocks whose adjusted `start_ms` would
    /// become negative are clamped to 0; if the adjusted `end_ms` falls at or
    /// before `start_ms`, the block is dropped from the output.
    pub fn shift(blocks: &[CaptionBlock], offset_ms: i64) -> Vec<CaptionBlock> {
        let mut result: Vec<CaptionBlock> = Vec::with_capacity(blocks.len());

        for block in blocks {
            let new_start = apply_offset(block.start_ms, offset_ms);
            let new_end = apply_offset(block.end_ms, offset_ms);

            if new_end <= new_start {
                // Block disappears after shift.
                continue;
            }

            let mut adjusted = block.clone();
            adjusted.start_ms = new_start;
            adjusted.end_ms = new_end;
            result.push(adjusted);
        }

        result
    }

    // ─── Linear stretch ────────────────────────────────────────────────────────

    /// Stretch (or compress) all caption timestamps by multiplying by `factor`.
    ///
    /// A `factor` of `1.0` is a no-op; `2.0` doubles all durations; `0.5`
    /// halves them.
    ///
    /// # Errors
    /// Returns [`TimingAdjusterError::InvalidStretchFactor`] when `factor ≤ 0`.
    pub fn stretch(
        blocks: &[CaptionBlock],
        factor: f64,
    ) -> Result<Vec<CaptionBlock>, TimingAdjusterError> {
        if factor <= 0.0 {
            return Err(TimingAdjusterError::InvalidStretchFactor { factor });
        }

        let result = blocks
            .iter()
            .map(|block| {
                let mut adjusted = block.clone();
                adjusted.start_ms = (block.start_ms as f64 * factor).round() as u64;
                adjusted.end_ms = (block.end_ms as f64 * factor).round() as u64;
                // Ensure end > start after rounding.
                if adjusted.end_ms <= adjusted.start_ms {
                    adjusted.end_ms = adjusted.start_ms + 1;
                }
                adjusted
            })
            .collect();

        Ok(result)
    }

    // ─── Stretch around an anchor point ────────────────────────────────────────

    /// Stretch timestamps around an `anchor_ms` pivot point.
    ///
    /// Each timestamp `t` is transformed to `anchor_ms + (t - anchor_ms) * factor`.
    /// This is equivalent to a zoom into/out of a specific moment on the timeline.
    ///
    /// # Errors
    /// Returns [`TimingAdjusterError::InvalidStretchFactor`] when `factor ≤ 0`.
    pub fn stretch_around(
        blocks: &[CaptionBlock],
        factor: f64,
        anchor_ms: u64,
    ) -> Result<Vec<CaptionBlock>, TimingAdjusterError> {
        if factor <= 0.0 {
            return Err(TimingAdjusterError::InvalidStretchFactor { factor });
        }

        let result = blocks
            .iter()
            .map(|block| {
                let mut adjusted = block.clone();
                adjusted.start_ms = stretch_around_anchor(block.start_ms, anchor_ms, factor);
                adjusted.end_ms = stretch_around_anchor(block.end_ms, anchor_ms, factor);
                if adjusted.end_ms <= adjusted.start_ms {
                    adjusted.end_ms = adjusted.start_ms + 1;
                }
                adjusted
            })
            .collect();

        Ok(result)
    }

    // ─── Snap to frame ─────────────────────────────────────────────────────────

    /// Quantise all timestamps to the nearest frame boundary at `fps`.
    ///
    /// # Errors
    /// Returns [`TimingAdjusterError::InvalidFrameRate`] when `fps ≤ 0`.
    pub fn snap_to_frame(
        blocks: &[CaptionBlock],
        fps: f32,
    ) -> Result<Vec<CaptionBlock>, TimingAdjusterError> {
        if fps <= 0.0 {
            return Err(TimingAdjusterError::InvalidFrameRate { fps });
        }

        let ms_per_frame = 1000.0 / fps as f64;

        let result = blocks
            .iter()
            .map(|block| {
                let mut adjusted = block.clone();
                adjusted.start_ms = snap(block.start_ms, ms_per_frame);
                adjusted.end_ms = snap(block.end_ms, ms_per_frame);
                // Guard against degenerate result after snapping.
                if adjusted.end_ms <= adjusted.start_ms {
                    adjusted.end_ms = adjusted.start_ms + ms_per_frame.ceil() as u64;
                }
                adjusted
            })
            .collect();

        Ok(result)
    }

    // ─── EDL remap ─────────────────────────────────────────────────────────────

    /// Remap caption timings according to an Edit Decision List (EDL).
    ///
    /// Each [`EdlEntry`] maps a source time range to a destination start
    /// time.  Captions that fall entirely within a source range are remapped;
    /// captions that span an EDL cut point are split at the boundary.
    /// Captions outside all EDL entries are dropped.
    ///
    /// # Errors
    /// Returns [`TimingAdjusterError::InvalidEdlRange`] if any EDL entry has
    /// `src_start_ms >= src_end_ms`.
    pub fn remap_edl(
        blocks: &[CaptionBlock],
        edl: &[EdlEntry],
    ) -> Result<Vec<CaptionBlock>, TimingAdjusterError> {
        // Validate EDL entries.
        for entry in edl {
            if entry.src_start_ms >= entry.src_end_ms {
                return Err(TimingAdjusterError::InvalidEdlRange {
                    src_start_ms: entry.src_start_ms,
                    src_end_ms: entry.src_end_ms,
                });
            }
        }

        let mut result: Vec<CaptionBlock> = Vec::new();
        let mut next_id = 1u32;

        for block in blocks {
            // Find EDL entries that overlap with this block.
            for entry in edl {
                // Compute overlap between block [start, end) and EDL [src_start, src_end).
                let overlap_start = block.start_ms.max(entry.src_start_ms);
                let overlap_end = block.end_ms.min(entry.src_end_ms);

                if overlap_end <= overlap_start {
                    continue;
                }

                // Map to destination timeline.
                let dst_start = entry.dst_start_ms + (overlap_start - entry.src_start_ms);
                let dst_end = entry.dst_start_ms + (overlap_end - entry.src_start_ms);

                let mut remapped = block.clone();
                remapped.id = next_id;
                remapped.start_ms = dst_start;
                remapped.end_ms = dst_end;
                result.push(remapped);
                next_id += 1;
            }
        }

        // Sort by destination start time.
        result.sort_by_key(|b| b.start_ms);

        Ok(result)
    }

    // ─── Clamp to range ────────────────────────────────────────────────────────

    /// Remove or trim caption blocks to fit within `[start_ms, end_ms)`.
    ///
    /// Blocks that overlap the range are trimmed to fit.  Blocks entirely
    /// outside the range are removed.  The returned blocks have sequential
    /// IDs starting at 1.
    pub fn clamp_to_range(
        blocks: &[CaptionBlock],
        range_start_ms: u64,
        range_end_ms: u64,
    ) -> Vec<CaptionBlock> {
        if range_end_ms <= range_start_ms {
            return Vec::new();
        }

        let mut result: Vec<CaptionBlock> = Vec::new();
        let mut next_id = 1u32;

        for block in blocks {
            // Skip blocks entirely outside range.
            if block.end_ms <= range_start_ms || block.start_ms >= range_end_ms {
                continue;
            }

            let clamped_start = block.start_ms.max(range_start_ms);
            let clamped_end = block.end_ms.min(range_end_ms);

            if clamped_end > clamped_start {
                let mut clamped = block.clone();
                clamped.id = next_id;
                clamped.start_ms = clamped_start;
                clamped.end_ms = clamped_end;
                result.push(clamped);
                next_id += 1;
            }
        }

        result
    }
}

// ─── EDL entry ────────────────────────────────────────────────────────────────

/// A single entry in an Edit Decision List.
#[derive(Debug, Clone, PartialEq)]
pub struct EdlEntry {
    /// Start of the source clip range (inclusive).
    pub src_start_ms: u64,
    /// End of the source clip range (exclusive).
    pub src_end_ms: u64,
    /// Destination start time in the output timeline.
    pub dst_start_ms: u64,
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn apply_offset(ts: u64, offset: i64) -> u64 {
    if offset >= 0 {
        ts.saturating_add(offset as u64)
    } else {
        ts.saturating_sub((-offset) as u64)
    }
}

fn stretch_around_anchor(ts: u64, anchor: u64, factor: f64) -> u64 {
    let delta = ts as f64 - anchor as f64;
    let new_ts = anchor as f64 + delta * factor;
    new_ts.round().max(0.0) as u64
}

fn snap(ts: u64, ms_per_frame: f64) -> u64 {
    let frame = (ts as f64 / ms_per_frame).round();
    (frame * ms_per_frame).round() as u64
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, start_ms: u64, end_ms: u64) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![format!("block {}", id)],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // ─── shift ────────────────────────────────────────────────────────────────

    #[test]
    fn shift_positive_offset_moves_later() {
        let blocks = vec![make_block(1, 0, 2000)];
        let result = CaptionTimingAdjuster::shift(&blocks, 500);
        assert_eq!(result[0].start_ms, 500);
        assert_eq!(result[0].end_ms, 2500);
    }

    #[test]
    fn shift_negative_offset_moves_earlier() {
        let blocks = vec![make_block(1, 2000, 4000)];
        let result = CaptionTimingAdjuster::shift(&blocks, -1000);
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    #[test]
    fn shift_clamps_to_zero() {
        let blocks = vec![make_block(1, 500, 1500)];
        let result = CaptionTimingAdjuster::shift(&blocks, -2000);
        // start_ms → max(0, 500 - 2000) = 0; end_ms → max(0, 1500 - 2000) = 0.
        // Block dropped because end_ms (0) <= start_ms (0).
        assert!(result.is_empty());
    }

    #[test]
    fn shift_drops_blocks_with_zero_duration_after_shift() {
        let blocks = vec![make_block(1, 1000, 1000)]; // already zero dur
        let result = CaptionTimingAdjuster::shift(&blocks, 0);
        // Adjusted end == adjusted start → dropped.
        assert!(result.is_empty());
    }

    #[test]
    fn shift_zero_offset_is_noop() {
        let blocks = vec![make_block(1, 1000, 3000)];
        let result = CaptionTimingAdjuster::shift(&blocks, 0);
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    #[test]
    fn shift_empty_input_returns_empty() {
        let result = CaptionTimingAdjuster::shift(&[], 1000);
        assert!(result.is_empty());
    }

    // ─── stretch ──────────────────────────────────────────────────────────────

    #[test]
    fn stretch_factor_two_doubles_timestamps() {
        let blocks = vec![make_block(1, 1000, 2000)];
        let result = CaptionTimingAdjuster::stretch(&blocks, 2.0).expect("stretch should succeed");
        assert_eq!(result[0].start_ms, 2000);
        assert_eq!(result[0].end_ms, 4000);
    }

    #[test]
    fn stretch_factor_half_halves_timestamps() {
        let blocks = vec![make_block(1, 2000, 4000)];
        let result = CaptionTimingAdjuster::stretch(&blocks, 0.5).expect("stretch should succeed");
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 2000);
    }

    #[test]
    fn stretch_factor_one_is_noop() {
        let blocks = vec![make_block(1, 1000, 3000)];
        let result = CaptionTimingAdjuster::stretch(&blocks, 1.0).expect("stretch should succeed");
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    #[test]
    fn stretch_negative_factor_returns_error() {
        let blocks = vec![make_block(1, 0, 1000)];
        let err = CaptionTimingAdjuster::stretch(&blocks, -1.0).unwrap_err();
        assert!(matches!(
            err,
            TimingAdjusterError::InvalidStretchFactor { .. }
        ));
    }

    #[test]
    fn stretch_zero_factor_returns_error() {
        let err = CaptionTimingAdjuster::stretch(&[], 0.0).unwrap_err();
        assert!(matches!(
            err,
            TimingAdjusterError::InvalidStretchFactor { .. }
        ));
    }

    // ─── stretch_around ───────────────────────────────────────────────────────

    #[test]
    fn stretch_around_anchor_at_start() {
        let blocks = vec![make_block(1, 0, 2000)];
        // Factor 2.0 around anchor 0: timestamps double.
        let result = CaptionTimingAdjuster::stretch_around(&blocks, 2.0, 0)
            .expect("stretch around should succeed");
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 4000);
    }

    #[test]
    fn stretch_around_anchor_in_middle() {
        let blocks = vec![make_block(1, 1000, 3000)];
        // Anchor = 2000, factor = 2.0:
        // start: 2000 + (1000 - 2000) * 2 = 0
        // end:   2000 + (3000 - 2000) * 2 = 4000
        let result = CaptionTimingAdjuster::stretch_around(&blocks, 2.0, 2000)
            .expect("stretch around should succeed");
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 4000);
    }

    // ─── snap_to_frame ────────────────────────────────────────────────────────

    #[test]
    fn snap_to_frame_25fps() {
        let blocks = vec![make_block(1, 10, 2010)];
        // At 25fps, ms_per_frame = 40ms.
        // 10ms → nearest multiple of 40 = 0.
        // 2010ms → nearest = 2000.
        let result = CaptionTimingAdjuster::snap_to_frame(&blocks, 25.0)
            .expect("snap to frame should succeed");
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 2000);
    }

    #[test]
    fn snap_to_frame_invalid_fps_returns_error() {
        let err = CaptionTimingAdjuster::snap_to_frame(&[], 0.0).unwrap_err();
        assert!(matches!(err, TimingAdjusterError::InvalidFrameRate { .. }));
    }

    #[test]
    fn snap_to_frame_already_on_frame() {
        let blocks = vec![make_block(1, 0, 1000)];
        // At 25fps: ms_per_frame = 40. 0 and 1000 are both multiples of 40.
        let result = CaptionTimingAdjuster::snap_to_frame(&blocks, 25.0)
            .expect("snap to frame should succeed");
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 1000);
    }

    // ─── remap_edl ────────────────────────────────────────────────────────────

    #[test]
    fn edl_remap_simple() {
        let blocks = vec![make_block(1, 1000, 3000)];
        let edl = vec![EdlEntry {
            src_start_ms: 0,
            src_end_ms: 5000,
            dst_start_ms: 0,
        }];
        let result =
            CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap edl should succeed");
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    #[test]
    fn edl_remap_shifts_destination() {
        let blocks = vec![make_block(1, 1000, 3000)];
        let edl = vec![EdlEntry {
            src_start_ms: 1000,
            src_end_ms: 5000,
            dst_start_ms: 500, // destination 500ms earlier
        }];
        let result =
            CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap edl should succeed");
        assert_eq!(result[0].start_ms, 500);
        assert_eq!(result[0].end_ms, 2500);
    }

    #[test]
    fn edl_remap_drops_blocks_outside_edl() {
        let blocks = vec![make_block(1, 10_000, 12_000)];
        let edl = vec![EdlEntry {
            src_start_ms: 0,
            src_end_ms: 5000,
            dst_start_ms: 0,
        }];
        let result =
            CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap edl should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn edl_remap_invalid_entry_returns_error() {
        let blocks = vec![make_block(1, 0, 1000)];
        let edl = vec![EdlEntry {
            src_start_ms: 5000,
            src_end_ms: 1000, // invalid: start > end
            dst_start_ms: 0,
        }];
        let err = CaptionTimingAdjuster::remap_edl(&blocks, &edl).unwrap_err();
        assert!(matches!(err, TimingAdjusterError::InvalidEdlRange { .. }));
    }

    #[test]
    fn edl_remap_block_spans_cut_is_split() {
        let blocks = vec![make_block(1, 0, 6000)];
        let edl = vec![
            EdlEntry {
                src_start_ms: 0,
                src_end_ms: 3000,
                dst_start_ms: 0,
            },
            EdlEntry {
                src_start_ms: 3000,
                src_end_ms: 6000,
                dst_start_ms: 5000, // jump in destination
            },
        ];
        let result =
            CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap edl should succeed");
        // One block maps to [0, 3000) and another to [5000, 8000).
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[0].end_ms, 3000);
        assert_eq!(result[1].start_ms, 5000);
        assert_eq!(result[1].end_ms, 8000);
    }

    // ─── clamp_to_range ───────────────────────────────────────────────────────

    #[test]
    fn clamp_removes_out_of_range_blocks() {
        let blocks = vec![make_block(1, 0, 1000), make_block(2, 5000, 7000)];
        // Range [2000, 8000).
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 2000, 8000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 5000);
    }

    #[test]
    fn clamp_trims_overlapping_blocks() {
        let blocks = vec![make_block(1, 500, 3000)];
        // Range [1000, 2000).
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 1000, 2000);
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 2000);
    }

    #[test]
    fn clamp_renumbers_blocks() {
        let blocks = vec![make_block(5, 1000, 2000), make_block(6, 3000, 4000)];
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 0, 10000);
        assert_eq!(result[0].id, 1);
        assert_eq!(result[1].id, 2);
    }

    #[test]
    fn clamp_invalid_range_returns_empty() {
        let blocks = vec![make_block(1, 0, 1000)];
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 5000, 1000);
        assert!(result.is_empty());
    }

    // ─── TimingAdjusterError display ─────────────────────────────────────────

    #[test]
    fn error_display_invalid_factor() {
        let e = TimingAdjusterError::InvalidStretchFactor { factor: -1.0 };
        assert!(e.to_string().contains("positive"));
    }

    #[test]
    fn error_display_invalid_fps() {
        let e = TimingAdjusterError::InvalidFrameRate { fps: 0.0 };
        assert!(e.to_string().contains("positive"));
    }

    // ─── Additional tests ─────────────────────────────────────────────────────

    #[test]
    fn shift_preserves_block_text_and_id() {
        let blocks = vec![make_block(5, 1000, 3000)];
        let result = CaptionTimingAdjuster::shift(&blocks, 200);
        assert_eq!(result[0].id, 5);
        assert_eq!(result[0].lines[0], "block 5");
    }

    #[test]
    fn shift_multiple_blocks_all_shifted() {
        let blocks = vec![
            make_block(1, 0, 1000),
            make_block(2, 2000, 3000),
            make_block(3, 4000, 5000),
        ];
        let result = CaptionTimingAdjuster::shift(&blocks, 500);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].start_ms, 500);
        assert_eq!(result[1].start_ms, 2500);
        assert_eq!(result[2].start_ms, 4500);
    }

    #[test]
    fn stretch_factor_quarter_compresses_timestamps() {
        let blocks = vec![make_block(1, 4000, 8000)];
        let result = CaptionTimingAdjuster::stretch(&blocks, 0.25).expect("stretch");
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 2000);
    }

    #[test]
    fn stretch_preserves_number_of_blocks() {
        let blocks: Vec<CaptionBlock> = (1..=5)
            .map(|i| make_block(i, i as u64 * 1000, i as u64 * 1000 + 500))
            .collect();
        let result = CaptionTimingAdjuster::stretch(&blocks, 1.5).expect("stretch");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn snap_to_frame_30fps() {
        // At 30fps, ms_per_frame ≈ 33.33ms.
        // 50ms → nearest frame = frame 2 (50/33.33 ≈ 1.5 → round to 2 → 66ms)
        let blocks = vec![make_block(1, 50, 1050)];
        let result = CaptionTimingAdjuster::snap_to_frame(&blocks, 30.0).expect("snap");
        // Verify snapped values are multiples of ~33.33ms (within rounding).
        let ms_per_frame = 1000.0 / 30.0_f64;
        let start_frame = (result[0].start_ms as f64 / ms_per_frame).round();
        assert!((result[0].start_ms as f64 - start_frame * ms_per_frame).abs() < 1.0);
    }

    #[test]
    fn edl_remap_multiple_blocks_multiple_edl_entries() {
        let blocks = vec![make_block(1, 500, 1500), make_block(2, 2000, 3000)];
        let edl = vec![
            EdlEntry {
                src_start_ms: 0,
                src_end_ms: 2000,
                dst_start_ms: 0,
            },
            EdlEntry {
                src_start_ms: 2000,
                src_end_ms: 4000,
                dst_start_ms: 10000,
            },
        ];
        let result = CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap");
        assert_eq!(result.len(), 2);
        // Block 1 maps via entry 1, block 2 maps via entry 2.
        assert_eq!(result[0].start_ms, 500);
        assert_eq!(result[1].start_ms, 10000);
    }

    #[test]
    fn clamp_block_exactly_at_range_start_included() {
        let blocks = vec![make_block(1, 1000, 2000)];
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 1000, 3000);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].start_ms, 1000);
    }

    #[test]
    fn clamp_block_exactly_at_range_end_excluded() {
        let blocks = vec![make_block(1, 3000, 4000)];
        // Block starts at range end → entirely outside → dropped.
        let result = CaptionTimingAdjuster::clamp_to_range(&blocks, 1000, 3000);
        assert!(result.is_empty());
    }

    #[test]
    fn stretch_around_factor_one_is_noop() {
        let blocks = vec![make_block(1, 1000, 3000)];
        let result = CaptionTimingAdjuster::stretch_around(&blocks, 1.0, 500).expect("stretch");
        assert_eq!(result[0].start_ms, 1000);
        assert_eq!(result[0].end_ms, 3000);
    }

    #[test]
    fn edl_remap_reassigns_sequential_ids() {
        let blocks = vec![make_block(10, 0, 1000), make_block(20, 1000, 2000)];
        let edl = vec![EdlEntry {
            src_start_ms: 0,
            src_end_ms: 5000,
            dst_start_ms: 0,
        }];
        let result = CaptionTimingAdjuster::remap_edl(&blocks, &edl).expect("remap");
        assert_eq!(result[0].id, 1);
        assert_eq!(result[1].id, 2);
    }

    #[test]
    fn error_display_invalid_edl_range() {
        let e = TimingAdjusterError::InvalidEdlRange {
            src_start_ms: 5000,
            src_end_ms: 1000,
        };
        assert!(e.to_string().contains("5000"));
        assert!(e.to_string().contains("1000"));
    }
}
