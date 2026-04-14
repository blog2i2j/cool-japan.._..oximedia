//! Clip arrangement operations: distribute, align, close gaps, snap to grid.
//!
//! Provides batch operations on a set of clips within a track, common in NLE
//! workflows for quickly organising material on the timeline.
//!
//! All functions operate on `&mut [Clip]` slices and never re-allocate.  They
//! assume clips are sorted by `timeline_start` on entry and maintain that
//! invariant on exit.
//!
//! # Example
//! ```rust
//! use oximedia_edit::clip::{Clip, ClipType};
//! use oximedia_edit::clip_arrange;
//!
//! let mut clips = vec![
//!     Clip::new(1, ClipType::Video, 0,   100),
//!     Clip::new(2, ClipType::Video, 200, 100),
//!     Clip::new(3, ClipType::Video, 500, 100),
//! ];
//! clip_arrange::close_gaps(&mut clips);
//! assert_eq!(clips[1].timeline_start, 100);
//! assert_eq!(clips[2].timeline_start, 200);
//! ```

#![allow(dead_code)]

use crate::clip::Clip;

// ─────────────────────────────────────────────────────────────────────────────
// close_gaps
// ─────────────────────────────────────────────────────────────────────────────

/// Shift clips leftward so that each clip's start equals the previous clip's
/// end, eliminating all gaps.  The first clip's start position is preserved.
///
/// Clips must be sorted by `timeline_start` (ascending) on entry.
pub fn close_gaps(clips: &mut [Clip]) {
    if clips.len() < 2 {
        return;
    }
    for i in 1..clips.len() {
        let prev_end = clips[i - 1].timeline_start + clips[i - 1].timeline_duration;
        clips[i].timeline_start = prev_end;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// distribute_evenly
// ─────────────────────────────────────────────────────────────────────────────

/// Distribute clips with equal spacing so that they span exactly `[range_start,
/// range_end)`.
///
/// The total clip duration stays the same; only gaps between clips change.
/// If there is only one clip or the total clip duration already exceeds the
/// range, clips are simply packed from `range_start` with no gaps.
///
/// Clips must be sorted by `timeline_start`.
pub fn distribute_evenly(clips: &mut [Clip], range_start: i64, range_end: i64) {
    if clips.is_empty() {
        return;
    }

    let total_clip_dur: i64 = clips.iter().map(|c| c.timeline_duration).sum();
    let range_dur = range_end - range_start;

    let n = clips.len();

    if n <= 1 || total_clip_dur >= range_dur {
        // Just pack from range_start
        let mut pos = range_start;
        for clip in clips.iter_mut() {
            clip.timeline_start = pos;
            pos += clip.timeline_duration;
        }
        return;
    }

    // We have n clips and (n-1) gaps.
    let total_gap = range_dur - total_clip_dur;
    let gap_count = (n - 1) as i64;
    let base_gap = total_gap / gap_count;
    let remainder = total_gap % gap_count;

    let mut pos = range_start;
    for (i, clip) in clips.iter_mut().enumerate() {
        clip.timeline_start = pos;
        pos += clip.timeline_duration;
        if (i as i64) < gap_count {
            pos += base_gap;
            // Distribute remainder one unit at a time to early gaps.
            if (i as i64) < remainder {
                pos += 1;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// align_starts
// ─────────────────────────────────────────────────────────────────────────────

/// Set all clips' `timeline_start` to `position` (stack them at the same
/// start point).  Useful for aligning heads of clips across different tracks
/// before manual adjustment.
pub fn align_starts(clips: &mut [Clip], position: i64) {
    for clip in clips.iter_mut() {
        clip.timeline_start = position;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// align_ends
// ─────────────────────────────────────────────────────────────────────────────

/// Set all clips so that their `timeline_end()` equals `position`.
pub fn align_ends(clips: &mut [Clip], position: i64) {
    for clip in clips.iter_mut() {
        clip.timeline_start = position - clip.timeline_duration;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// snap_to_grid
// ─────────────────────────────────────────────────────────────────────────────

/// Round each clip's `timeline_start` to the nearest multiple of `grid_size`.
///
/// `grid_size` must be > 0; if zero or negative, clips are left unchanged.
pub fn snap_to_grid(clips: &mut [Clip], grid_size: i64) {
    if grid_size <= 0 {
        return;
    }
    for clip in clips.iter_mut() {
        let remainder = clip.timeline_start % grid_size;
        if remainder == 0 {
            continue;
        }
        // Round to nearest grid line.
        if remainder.abs() * 2 >= grid_size {
            clip.timeline_start += grid_size - remainder;
        } else {
            clip.timeline_start -= remainder;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ripple_insert_gap
// ─────────────────────────────────────────────────────────────────────────────

/// Insert a gap of `gap_duration` at `position` by shifting all clips that
/// start at or after `position` to the right.
pub fn ripple_insert_gap(clips: &mut [Clip], position: i64, gap_duration: i64) {
    for clip in clips.iter_mut() {
        if clip.timeline_start >= position {
            clip.timeline_start += gap_duration;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// reverse_order
// ─────────────────────────────────────────────────────────────────────────────

/// Reverse the order of clips while keeping the same start position as the
/// first original clip and packing them contiguously.
///
/// The first clip in the reversed sequence starts at `clips[0].timeline_start`.
pub fn reverse_order(clips: &mut [Clip]) {
    if clips.len() < 2 {
        return;
    }
    let start = clips[0].timeline_start;

    // Collect durations in reverse order.
    let durations: Vec<i64> = clips.iter().rev().map(|c| c.timeline_duration).collect();

    // Reverse the slice in-place.
    clips.reverse();

    // Re-assign timeline positions contiguously.
    let mut pos = start;
    for (clip, &dur) in clips.iter_mut().zip(durations.iter()) {
        clip.timeline_start = pos;
        clip.timeline_duration = dur;
        pos += dur;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// total_duration
// ─────────────────────────────────────────────────────────────────────────────

/// Return the span from the earliest start to the latest end across all clips.
///
/// Returns 0 for an empty slice.
#[must_use]
pub fn total_span(clips: &[Clip]) -> i64 {
    if clips.is_empty() {
        return 0;
    }
    let min_start = clips.iter().map(|c| c.timeline_start).min().unwrap_or(0);
    let max_end = clips
        .iter()
        .map(|c| c.timeline_start + c.timeline_duration)
        .max()
        .unwrap_or(0);
    max_end - min_start
}

/// Count the total gap frames between clips (assumes sorted by start).
#[must_use]
pub fn total_gap(clips: &[Clip]) -> i64 {
    if clips.len() < 2 {
        return 0;
    }
    let mut gap = 0i64;
    for pair in clips.windows(2) {
        let end = pair[0].timeline_start + pair[0].timeline_duration;
        let next_start = pair[1].timeline_start;
        if next_start > end {
            gap += next_start - end;
        }
    }
    gap
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipType;

    fn make_clips(specs: &[(i64, i64)]) -> Vec<Clip> {
        specs
            .iter()
            .enumerate()
            .map(|(i, &(start, dur))| Clip::new((i + 1) as u64, ClipType::Video, start, dur))
            .collect()
    }

    // ── close_gaps ──────────────────────────────────────────────────────

    #[test]
    fn test_close_gaps_removes_all_gaps() {
        let mut clips = make_clips(&[(0, 100), (200, 100), (500, 100)]);
        close_gaps(&mut clips);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].timeline_start, 100);
        assert_eq!(clips[2].timeline_start, 200);
    }

    #[test]
    fn test_close_gaps_single_clip_noop() {
        let mut clips = make_clips(&[(50, 100)]);
        close_gaps(&mut clips);
        assert_eq!(clips[0].timeline_start, 50);
    }

    #[test]
    fn test_close_gaps_already_contiguous() {
        let mut clips = make_clips(&[(0, 100), (100, 100)]);
        close_gaps(&mut clips);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].timeline_start, 100);
    }

    // ── distribute_evenly ───────────────────────────────────────────────

    #[test]
    fn test_distribute_evenly_two_clips() {
        // Two 100-unit clips in a 400-unit range → 200-unit gap between them.
        let mut clips = make_clips(&[(0, 100), (100, 100)]);
        distribute_evenly(&mut clips, 0, 400);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].timeline_start, 300); // 0+100+200=300
    }

    #[test]
    fn test_distribute_evenly_overflow_packs() {
        // Total clip dur (300) exceeds range (200) → pack from start.
        let mut clips = make_clips(&[(0, 100), (0, 100), (0, 100)]);
        distribute_evenly(&mut clips, 0, 200);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].timeline_start, 100);
        assert_eq!(clips[2].timeline_start, 200);
    }

    #[test]
    fn test_distribute_evenly_single_clip() {
        let mut clips = make_clips(&[(50, 100)]);
        distribute_evenly(&mut clips, 0, 1000);
        assert_eq!(clips[0].timeline_start, 0);
    }

    // ── align_starts / align_ends ───────────────────────────────────────

    #[test]
    fn test_align_starts() {
        let mut clips = make_clips(&[(0, 100), (500, 200), (1000, 50)]);
        align_starts(&mut clips, 42);
        assert!(clips.iter().all(|c| c.timeline_start == 42));
    }

    #[test]
    fn test_align_ends() {
        let mut clips = make_clips(&[(0, 100), (500, 200), (1000, 50)]);
        align_ends(&mut clips, 1000);
        for clip in &clips {
            assert_eq!(clip.timeline_start + clip.timeline_duration, 1000);
        }
    }

    // ── snap_to_grid ────────────────────────────────────────────────────

    #[test]
    fn test_snap_to_grid_rounds_nearest() {
        let mut clips = make_clips(&[(7, 100), (23, 100), (30, 100)]);
        snap_to_grid(&mut clips, 10);
        assert_eq!(clips[0].timeline_start, 10); // 7 rounds up
        assert_eq!(clips[1].timeline_start, 20); // 23 rounds down
        assert_eq!(clips[2].timeline_start, 30); // already aligned
    }

    #[test]
    fn test_snap_to_grid_zero_grid_noop() {
        let mut clips = make_clips(&[(7, 100)]);
        snap_to_grid(&mut clips, 0);
        assert_eq!(clips[0].timeline_start, 7);
    }

    // ── ripple_insert_gap ───────────────────────────────────────────────

    #[test]
    fn test_ripple_insert_gap() {
        let mut clips = make_clips(&[(0, 100), (100, 100), (200, 100)]);
        ripple_insert_gap(&mut clips, 100, 50);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].timeline_start, 150);
        assert_eq!(clips[2].timeline_start, 250);
    }

    // ── reverse_order ───────────────────────────────────────────────────

    #[test]
    fn test_reverse_order_three_clips() {
        let mut clips = make_clips(&[(0, 100), (100, 200), (300, 50)]);
        reverse_order(&mut clips);
        // After reversal: clip 3, clip 2, clip 1 packed from 0
        assert_eq!(clips[0].id, 3);
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[1].id, 2);
        assert_eq!(clips[1].timeline_start, 50);
        assert_eq!(clips[2].id, 1);
        assert_eq!(clips[2].timeline_start, 250);
    }

    #[test]
    fn test_reverse_order_single_noop() {
        let mut clips = make_clips(&[(10, 100)]);
        reverse_order(&mut clips);
        assert_eq!(clips[0].timeline_start, 10);
    }

    // ── total_span / total_gap ──────────────────────────────────────────

    #[test]
    fn test_total_span() {
        let clips = make_clips(&[(10, 100), (200, 50)]);
        assert_eq!(total_span(&clips), 240); // 250 - 10
    }

    #[test]
    fn test_total_span_empty() {
        let clips: Vec<Clip> = Vec::new();
        assert_eq!(total_span(&clips), 0);
    }

    #[test]
    fn test_total_gap_with_gaps() {
        let clips = make_clips(&[(0, 100), (200, 100), (400, 100)]);
        assert_eq!(total_gap(&clips), 200);
    }

    #[test]
    fn test_total_gap_contiguous() {
        let clips = make_clips(&[(0, 100), (100, 100)]);
        assert_eq!(total_gap(&clips), 0);
    }

    #[test]
    fn test_total_gap_empty() {
        let clips: Vec<Clip> = Vec::new();
        assert_eq!(total_gap(&clips), 0);
    }
}
