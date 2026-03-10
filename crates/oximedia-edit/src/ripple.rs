//! Ripple editing operations for the timeline.
//!
//! Ripple edits automatically shift downstream clips when a clip is inserted,
//! removed, or trimmed on a specific track.

#![allow(dead_code)]

/// A clip positioned on the timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineClip {
    /// Unique identifier for this clip.
    pub id: u64,
    /// Timeline start position in timebase units.
    pub start: u64,
    /// Duration of this clip in timebase units.
    pub duration: u64,
    /// Source media in-point in timebase units.
    pub source_in: u64,
    /// The track index this clip lives on.
    pub track: u32,
}

impl TimelineClip {
    /// Creates a new `TimelineClip`.
    #[must_use]
    pub const fn new(id: u64, start: u64, duration: u64, source_in: u64, track: u32) -> Self {
        Self {
            id,
            start,
            duration,
            source_in,
            track,
        }
    }

    /// Returns the timeline end position (exclusive) of this clip.
    #[must_use]
    pub fn end(&self) -> u64 {
        self.start.saturating_add(self.duration)
    }
}

/// Controls how ripple editing inserts or removes time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RippleMode {
    /// Insert a gap and push subsequent clips to the right.
    Insert,
    /// Overwrite existing content without moving other clips.
    Overwrite,
    /// Remove content and leave a gap (no shift).
    Lift,
    /// Remove content and close the gap by pulling subsequent clips left.
    Extract,
}

/// Inserts a gap at `insert_at` on `track` and shifts all clips that start
/// at or after that position to the right by `duration`.
pub fn ripple_insert(clips: &mut Vec<TimelineClip>, insert_at: u64, duration: u64, track: u32) {
    for clip in clips.iter_mut() {
        if clip.track == track && clip.start >= insert_at {
            clip.start = clip.start.saturating_add(duration);
        }
    }
}

/// Deletes the region `[start, end)` on `track` in Extract mode:
/// clips fully inside the region are removed, clips partially overlapping
/// are trimmed, and clips after the region are shifted left.
///
/// Returns the number of timeline units that were removed (i.e., `end - start`
/// clamped to what was actually deleted on the track).
pub fn ripple_delete(clips: &mut Vec<TimelineClip>, start: u64, end: u64, track: u32) -> u64 {
    if end <= start {
        return 0;
    }
    let deleted_duration = end - start;

    // 1. Remove clips entirely inside the range.
    clips.retain(|c| {
        if c.track != track {
            return true;
        }
        // Keep the clip unless it is fully within [start, end)
        !(c.start >= start && c.end() <= end)
    });

    // 2. Trim clips that partially overlap the region.
    for clip in clips.iter_mut() {
        if clip.track != track {
            continue;
        }
        let clip_end = clip.end();
        if clip.start < start && clip_end > start {
            // Clip spans the left edge – trim its right side.
            let new_duration = start - clip.start;
            clip.duration = new_duration;
        } else if clip.start < end && clip.start >= start {
            // Clip starts inside but we haven't removed it – trim its left side.
            let trim = end - clip.start;
            clip.source_in = clip.source_in.saturating_add(trim);
            clip.duration = clip.duration.saturating_sub(trim);
            clip.start = end;
        }
    }

    // 3. Shift clips after the deleted region to the left.
    for clip in clips.iter_mut() {
        if clip.track == track && clip.start >= end {
            clip.start = clip.start.saturating_sub(deleted_duration);
        }
    }

    deleted_duration
}

/// Trims the right (out) edge of a clip identified by `clip_id`.
///
/// `new_out` is the new *end* position (exclusive) of the clip on the
/// timeline.  All clips on the same track that come after the trimmed clip
/// are shifted accordingly (ripple).
///
/// # Errors
///
/// Returns `Err` if `clip_id` is not found or if `new_out` would make the
/// clip have zero or negative duration.
pub fn ripple_trim_right(
    clips: &mut Vec<TimelineClip>,
    clip_id: u64,
    new_out: u64,
) -> Result<(), String> {
    let (idx, old_end, track) = clips
        .iter()
        .enumerate()
        .find(|(_, c)| c.id == clip_id)
        .map(|(i, c)| (i, c.end(), c.track))
        .ok_or_else(|| format!("clip {clip_id} not found"))?;

    let clip_start = clips[idx].start;
    if new_out <= clip_start {
        return Err(format!(
            "new_out ({new_out}) must be greater than clip start ({clip_start})"
        ));
    }

    let delta: i64 = new_out as i64 - old_end as i64;
    clips[idx].duration = new_out - clip_start;

    // Shift clips on the same track that start at or after the old end.
    for clip in clips.iter_mut() {
        if clip.track == track && clip.start >= old_end && clip.id != clip_id {
            clip.start = (clip.start as i64 + delta).max(0) as u64;
        }
    }

    Ok(())
}

/// Trims the left (in) edge of a clip identified by `clip_id`.
///
/// `new_in` is the new *start* position on the timeline. All clips on the
/// same track that come before this clip are shifted accordingly (ripple).
///
/// # Errors
///
/// Returns `Err` if `clip_id` is not found or if `new_in` would make the
/// clip have zero or negative duration.
pub fn ripple_trim_left(
    clips: &mut Vec<TimelineClip>,
    clip_id: u64,
    new_in: u64,
) -> Result<(), String> {
    let (idx, old_start, clip_end, track) = clips
        .iter()
        .enumerate()
        .find(|(_, c)| c.id == clip_id)
        .map(|(i, c)| (i, c.start, c.end(), c.track))
        .ok_or_else(|| format!("clip {clip_id} not found"))?;

    if new_in >= clip_end {
        return Err(format!(
            "new_in ({new_in}) must be less than clip end ({clip_end})"
        ));
    }

    let delta: i64 = new_in as i64 - old_start as i64;
    let trim_amount = (new_in as i64 - old_start as i64).unsigned_abs();

    clips[idx].start = new_in;
    if delta > 0 {
        // Moving in-point right → clip gets shorter
        clips[idx].source_in = clips[idx].source_in.saturating_add(trim_amount);
        clips[idx].duration = clips[idx].duration.saturating_sub(trim_amount);
    } else {
        // Moving in-point left → clip gets longer
        clips[idx].source_in = clips[idx].source_in.saturating_sub(trim_amount);
        clips[idx].duration = clips[idx].duration.saturating_add(trim_amount);
    }

    // Shift clips on the same track that end at or before the old start.
    for clip in clips.iter_mut() {
        if clip.track == track && clip.end() <= old_start && clip.id != clip_id {
            clip.start = (clip.start as i64 + delta).max(0) as u64;
        }
    }

    Ok(())
}

/// Performs a three-point edit, placing source material `[src_in, src_out)`
/// at record position `rec_in` using the given `RippleMode`.
///
/// Returns the newly created `TimelineClip` (with id = 0; callers should
/// assign a real ID).
///
/// # Errors
///
/// Returns `Err` if `src_out <= src_in`.
#[allow(clippy::too_many_arguments)]
pub fn three_point_edit(
    clips: &mut Vec<TimelineClip>,
    src_in: u64,
    src_out: u64,
    rec_in: u64,
    mode: RippleMode,
) -> Result<TimelineClip, String> {
    if src_out <= src_in {
        return Err(format!(
            "src_out ({src_out}) must be greater than src_in ({src_in})"
        ));
    }

    let duration = src_out - src_in;
    let new_clip = TimelineClip::new(0, rec_in, duration, src_in, 0);

    match mode {
        RippleMode::Insert => {
            ripple_insert(clips, rec_in, duration, 0);
            clips.push(new_clip.clone());
        }
        RippleMode::Overwrite => {
            // Overwrite: just place the clip, no shift.
            clips.push(new_clip.clone());
        }
        RippleMode::Lift | RippleMode::Extract => {
            // For three-point edit these behave like overwrite.
            clips.push(new_clip.clone());
        }
    }

    Ok(new_clip)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clip(id: u64, start: u64, duration: u64, track: u32) -> TimelineClip {
        TimelineClip::new(id, start, duration, 0, track)
    }

    #[test]
    fn test_clip_end() {
        let c = make_clip(1, 100, 200, 0);
        assert_eq!(c.end(), 300);
    }

    #[test]
    fn test_ripple_insert_shifts_clips_after_point() {
        let mut clips = vec![
            make_clip(1, 0, 100, 0),
            make_clip(2, 100, 100, 0),
            make_clip(3, 200, 100, 0),
        ];
        ripple_insert(&mut clips, 100, 50, 0);
        assert_eq!(clips[0].start, 0); // before insert_at → unchanged
        assert_eq!(clips[1].start, 150); // at insert_at → shifted
        assert_eq!(clips[2].start, 250); // after → shifted
    }

    #[test]
    fn test_ripple_insert_different_track_unchanged() {
        let mut clips = vec![
            make_clip(1, 100, 100, 0),
            make_clip(2, 100, 100, 1), // different track
        ];
        ripple_insert(&mut clips, 50, 100, 0);
        assert_eq!(clips[0].start, 200); // same track, shifted
        assert_eq!(clips[1].start, 100); // different track, unchanged
    }

    #[test]
    fn test_ripple_delete_removes_clips_inside_range() {
        let mut clips = vec![
            make_clip(1, 0, 50, 0),
            make_clip(2, 50, 50, 0),  // inside [50, 150)
            make_clip(3, 100, 50, 0), // inside
            make_clip(4, 150, 50, 0),
        ];
        ripple_delete(&mut clips, 50, 150, 0);
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].id, 1);
        assert_eq!(clips[1].id, 4);
        // Clip 4 was at 150 → shifted left by 100 → 50
        assert_eq!(clips[1].start, 50);
    }

    #[test]
    fn test_ripple_delete_returns_deleted_duration() {
        let mut clips = vec![make_clip(1, 200, 100, 0)];
        let deleted = ripple_delete(&mut clips, 0, 50, 0);
        assert_eq!(deleted, 50);
    }

    #[test]
    fn test_ripple_delete_zero_range() {
        let mut clips = vec![make_clip(1, 0, 100, 0)];
        let deleted = ripple_delete(&mut clips, 50, 50, 0);
        assert_eq!(deleted, 0);
        assert_eq!(clips.len(), 1);
    }

    #[test]
    fn test_ripple_trim_right_shortens_clip() {
        let mut clips = vec![make_clip(1, 0, 200, 0), make_clip(2, 200, 100, 0)];
        ripple_trim_right(&mut clips, 1, 150).expect("test expectation failed");
        assert_eq!(clips[0].duration, 150);
        // Clip 2 was at 200 → shifted left by 50 → 150
        assert_eq!(clips[1].start, 150);
    }

    #[test]
    fn test_ripple_trim_right_clip_not_found() {
        let mut clips = vec![make_clip(1, 0, 100, 0)];
        let result = ripple_trim_right(&mut clips, 99, 50);
        assert!(result.is_err());
    }

    #[test]
    fn test_ripple_trim_right_zero_duration_error() {
        let mut clips = vec![make_clip(1, 50, 100, 0)];
        let result = ripple_trim_right(&mut clips, 1, 50); // new_out == clip.start
        assert!(result.is_err());
    }

    #[test]
    fn test_ripple_trim_left_shortens_clip() {
        let mut clips = vec![make_clip(1, 0, 200, 0)];
        ripple_trim_left(&mut clips, 1, 50).expect("test expectation failed");
        assert_eq!(clips[0].start, 50);
        assert_eq!(clips[0].duration, 150);
        assert_eq!(clips[0].source_in, 50);
    }

    #[test]
    fn test_ripple_trim_left_clip_not_found() {
        let mut clips = vec![make_clip(1, 0, 100, 0)];
        let result = ripple_trim_left(&mut clips, 99, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_three_point_edit_insert() {
        let mut clips = vec![make_clip(1, 200, 100, 0)];
        let new_clip = three_point_edit(&mut clips, 0, 50, 0, RippleMode::Insert)
            .expect("new_clip should be valid");
        assert_eq!(new_clip.duration, 50);
        assert_eq!(new_clip.start, 0);
        // Original clip shifted right by 50
        let orig = clips
            .iter()
            .find(|c| c.id == 1)
            .expect("orig should be valid");
        assert_eq!(orig.start, 250);
    }

    #[test]
    fn test_three_point_edit_invalid_range() {
        let mut clips = vec![];
        let result = three_point_edit(&mut clips, 100, 50, 0, RippleMode::Insert);
        assert!(result.is_err());
    }

    #[test]
    fn test_ripple_mode_overwrite_no_shift() {
        let mut clips = vec![make_clip(1, 100, 100, 0)];
        three_point_edit(&mut clips, 0, 50, 0, RippleMode::Overwrite)
            .expect("test expectation failed");
        // Original clip must NOT be shifted
        let orig = clips
            .iter()
            .find(|c| c.id == 1)
            .expect("orig should be valid");
        assert_eq!(orig.start, 100);
    }
}
