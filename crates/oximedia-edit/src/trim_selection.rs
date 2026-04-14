//! Trim-to-selection: extract a contiguous sub-range from a clip list.
//!
//! Given a list of [`crate::clip::Clip`]s and an in/out timecode range
//! (in the same timebase units as the clip `timeline_start` / `timeline_duration`
//! fields), `trim_to_selection` returns a new `Vec<Clip>` containing only the
//! clips—or portions thereof—that fall within `[in_tc, out_tc)`.  Clips that
//! straddle a boundary are trimmed and their `source_in`/`source_out` are
//! updated to keep the source mapping consistent.
//!
//! # Example
//! ```rust
//! use oximedia_edit::clip::{Clip, ClipType};
//! use oximedia_edit::trim_selection::trim_to_selection;
//!
//! let clips = vec![
//!     Clip::new(1, ClipType::Video, 0,   500),
//!     Clip::new(2, ClipType::Video, 500, 500),
//!     Clip::new(3, ClipType::Video, 1000, 500),
//! ];
//! let trimmed = trim_to_selection(&clips, 250, 750);
//! assert_eq!(trimmed.len(), 2); // partial clip 1 and partial clip 2
//! ```

#![allow(dead_code)]

use crate::clip::Clip;

/// Extract all clips (or partial clips) whose timeline range intersects
/// `[in_tc, out_tc)`.
///
/// Clips are returned in their original order. Each returned clip has:
/// - `timeline_start` clamped to `max(original_start, in_tc)`
/// - `timeline_duration` adjusted so `timeline_start + duration == min(original_end, out_tc)`
/// - `source_in` incremented by the left trim amount
/// - `source_out` decremented by the right trim amount
///
/// Clips entirely outside `[in_tc, out_tc)` are silently dropped.
/// If `in_tc >= out_tc` an empty vector is returned.
#[must_use]
pub fn trim_to_selection(clips: &[Clip], in_tc: u64, out_tc: u64) -> Vec<Clip> {
    if in_tc >= out_tc {
        return Vec::new();
    }

    let sel_in = in_tc as i64;
    let sel_out = out_tc as i64;

    let mut result = Vec::new();

    for clip in clips {
        let clip_start = clip.timeline_start;
        let clip_end = clip_start + clip.timeline_duration;

        // Skip clips entirely outside the selection.
        if clip_end <= sel_in || clip_start >= sel_out {
            continue;
        }

        let mut trimmed = clip.clone();

        // Left trim: clip starts before selection in-point.
        if clip_start < sel_in {
            let left_trim = sel_in - clip_start;
            trimmed.timeline_start = sel_in;
            trimmed.timeline_duration -= left_trim;
            trimmed.source_in += left_trim;
        }

        // Right trim: clip ends after selection out-point.
        let trimmed_end = trimmed.timeline_start + trimmed.timeline_duration;
        if trimmed_end > sel_out {
            let right_trim = trimmed_end - sel_out;
            trimmed.timeline_duration -= right_trim;
            trimmed.source_out -= right_trim;
        }

        if trimmed.timeline_duration > 0 {
            result.push(trimmed);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::ClipType;

    fn make_clip(id: u64, start: i64, dur: i64) -> Clip {
        let mut c = Clip::new(id, ClipType::Video, start, dur);
        c.source_in = 0;
        c.source_out = dur;
        c
    }

    #[test]
    fn test_trim_empty_range() {
        let clips = vec![make_clip(1, 0, 100)];
        assert!(trim_to_selection(&clips, 50, 50).is_empty());
        assert!(trim_to_selection(&clips, 100, 50).is_empty());
    }

    #[test]
    fn test_trim_clips_fully_inside() {
        let clips = vec![make_clip(1, 100, 100), make_clip(2, 200, 100)];
        let result = trim_to_selection(&clips, 50, 400);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timeline_start, 100);
        assert_eq!(result[0].timeline_duration, 100);
        assert_eq!(result[1].timeline_start, 200);
    }

    #[test]
    fn test_trim_clips_fully_outside() {
        let clips = vec![make_clip(1, 0, 100), make_clip(2, 500, 100)];
        let result = trim_to_selection(&clips, 200, 400);
        assert!(result.is_empty());
    }

    #[test]
    fn test_trim_left_boundary() {
        // Clip [0, 200), selection [100, 300) → trimmed clip [100, 200)
        let clips = vec![make_clip(1, 0, 200)];
        let result = trim_to_selection(&clips, 100, 300);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timeline_start, 100);
        assert_eq!(result[0].timeline_duration, 100);
        assert_eq!(result[0].source_in, 100);
        assert_eq!(result[0].source_out, 200);
    }

    #[test]
    fn test_trim_right_boundary() {
        // Clip [0, 300), selection [0, 150) → trimmed [0, 150)
        let clips = vec![make_clip(1, 0, 300)];
        let result = trim_to_selection(&clips, 0, 150);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timeline_duration, 150);
        assert_eq!(result[0].source_out, 150);
    }

    #[test]
    fn test_trim_both_boundaries() {
        // Clip [0, 1000), selection [300, 700) → [300, 700), dur=400
        let clips = vec![make_clip(1, 0, 1000)];
        let result = trim_to_selection(&clips, 300, 700);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].timeline_start, 300);
        assert_eq!(result[0].timeline_duration, 400);
        assert_eq!(result[0].source_in, 300);
        assert_eq!(result[0].source_out, 700);
    }

    #[test]
    fn test_trim_multiple_clips_mixed() {
        let clips = vec![
            make_clip(1, 0, 100),   // before selection
            make_clip(2, 100, 200), // straddles left edge at 150
            make_clip(3, 300, 200), // fully inside [150, 400)... wait let's use 150-400
            make_clip(4, 400, 100), // partially inside [150, 420)
            make_clip(5, 500, 100), // after selection
        ];
        // Selection [150, 420)
        let result = trim_to_selection(&clips, 150, 420);
        assert_eq!(result.len(), 3); // clips 2, 3, 4
                                     // Clip 2: [100,300) → trimmed to [150,300), dur=150, src_in=50
        assert_eq!(result[0].timeline_start, 150);
        assert_eq!(result[0].timeline_duration, 150);
        assert_eq!(result[0].source_in, 50);
        // Clip 3: [300,500) → trimmed to [300,420), dur=120
        assert_eq!(result[2].timeline_duration, 20);
    }
}
