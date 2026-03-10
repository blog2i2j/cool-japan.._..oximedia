#![allow(dead_code)]
//! Duration arithmetic and range calculations for EDL events.
//!
//! This module provides utilities for computing durations, time ranges,
//! and performing arithmetic on EDL timecodes and event spans.

use crate::error::{EdlError, EdlResult};
use crate::event::EdlEvent;
use crate::timecode::{EdlFrameRate, EdlTimecode};

/// A time range defined by in and out timecodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeRange {
    /// Start timecode.
    pub tc_in: EdlTimecode,
    /// End timecode.
    pub tc_out: EdlTimecode,
}

impl TimeRange {
    /// Create a new time range.
    ///
    /// # Errors
    ///
    /// Returns an error if `tc_in >= tc_out`.
    pub fn new(tc_in: EdlTimecode, tc_out: EdlTimecode) -> EdlResult<Self> {
        if tc_in >= tc_out {
            return Err(EdlError::validation("tc_in must be before tc_out"));
        }
        Ok(Self { tc_in, tc_out })
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration_frames(&self) -> u64 {
        self.tc_out.to_frames() - self.tc_in.to_frames()
    }

    /// Duration in seconds for the given frame rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, fps: EdlFrameRate) -> f64 {
        self.duration_frames() as f64 / fps.as_float()
    }

    /// Check if this range contains a given timecode.
    #[must_use]
    pub fn contains(&self, tc: &EdlTimecode) -> bool {
        *tc >= self.tc_in && *tc < self.tc_out
    }

    /// Check if this range overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        !(self.tc_out <= other.tc_in || self.tc_in >= other.tc_out)
    }

    /// Compute the intersection of two ranges, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.overlaps(other) {
            return None;
        }
        let start = if self.tc_in >= other.tc_in {
            self.tc_in
        } else {
            other.tc_in
        };
        let end = if self.tc_out <= other.tc_out {
            self.tc_out
        } else {
            other.tc_out
        };
        Some(Self {
            tc_in: start,
            tc_out: end,
        })
    }

    /// Compute the gap in frames between this range and another.
    /// Returns 0 if the ranges overlap or are contiguous.
    #[must_use]
    pub fn gap_frames(&self, other: &Self) -> u64 {
        if self.overlaps(other) {
            return 0;
        }
        if self.tc_out <= other.tc_in {
            other.tc_in.to_frames() - self.tc_out.to_frames()
        } else {
            self.tc_in.to_frames() - other.tc_out.to_frames()
        }
    }
}

/// Summary of duration statistics for a list of events.
#[derive(Debug, Clone)]
pub struct DurationSummary {
    /// Total duration in frames.
    pub total_frames: u64,
    /// Minimum event duration in frames.
    pub min_frames: u64,
    /// Maximum event duration in frames.
    pub max_frames: u64,
    /// Average event duration in frames.
    pub avg_frames: f64,
    /// Number of events.
    pub event_count: usize,
}

/// Compute a duration summary from a slice of events.
///
/// # Errors
///
/// Returns an error if `events` is empty.
#[allow(clippy::cast_precision_loss)]
pub fn compute_duration_summary(events: &[EdlEvent]) -> EdlResult<DurationSummary> {
    if events.is_empty() {
        return Err(EdlError::validation("No events to summarize"));
    }
    let mut total: u64 = 0;
    let mut min = u64::MAX;
    let mut max: u64 = 0;
    for ev in events {
        let d = ev.duration_frames();
        total += d;
        if d < min {
            min = d;
        }
        if d > max {
            max = d;
        }
    }
    let avg = total as f64 / events.len() as f64;
    Ok(DurationSummary {
        total_frames: total,
        min_frames: min,
        max_frames: max,
        avg_frames: avg,
        event_count: events.len(),
    })
}

/// Offset all events in a list by a frame delta (positive = shift forward).
///
/// # Errors
///
/// Returns an error if the resulting timecodes are invalid.
pub fn offset_events(
    events: &[EdlEvent],
    delta_frames: i64,
    fps: EdlFrameRate,
) -> EdlResult<Vec<EdlEvent>> {
    let mut result = Vec::with_capacity(events.len());
    for ev in events {
        let mut cloned = ev.clone();
        cloned.source_in = offset_tc(ev.source_in, delta_frames, fps)?;
        cloned.source_out = offset_tc(ev.source_out, delta_frames, fps)?;
        cloned.record_in = offset_tc(ev.record_in, delta_frames, fps)?;
        cloned.record_out = offset_tc(ev.record_out, delta_frames, fps)?;
        result.push(cloned);
    }
    Ok(result)
}

/// Helper: apply a signed frame offset to a timecode.
fn offset_tc(
    tc: EdlTimecode,
    delta: i64,
    fps: EdlFrameRate,
) -> EdlResult<EdlTimecode> {
    let frames = tc.to_frames() as i64 + delta;
    if frames < 0 {
        return Err(EdlError::TimecodeOutOfRange(
            "Offset would make timecode negative".to_string(),
        ));
    }
    EdlTimecode::from_frames(frames as u64, fps)
}

/// Compute the total programme duration (record-side) from a sorted event list.
///
/// Uses the earliest record_in and latest record_out.
///
/// # Errors
///
/// Returns an error if `events` is empty.
pub fn programme_duration_frames(events: &[EdlEvent]) -> EdlResult<u64> {
    if events.is_empty() {
        return Err(EdlError::validation("No events provided"));
    }
    let first_in = events
        .iter()
        .map(|e| e.record_in.to_frames())
        .min()
        .unwrap_or(0);
    let last_out = events
        .iter()
        .map(|e| e.record_out.to_frames())
        .max()
        .unwrap_or(0);
    Ok(last_out - first_in)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};

    fn make_tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("failed to create")
    }

    fn make_event(num: u32, ri_s: u8, ro_s: u8) -> EdlEvent {
        let si = make_tc(1, 0, 0, 0);
        let so = make_tc(1, 0, ro_s - ri_s, 0);
        let ri = make_tc(1, 0, ri_s, 0);
        let ro = make_tc(1, 0, ro_s, 0);
        EdlEvent::new(
            num,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            si,
            so,
            ri,
            ro,
        )
    }

    #[test]
    fn test_time_range_creation() {
        let r = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 5, 0));
        assert!(r.is_ok());
    }

    #[test]
    fn test_time_range_invalid() {
        let tc = make_tc(0, 0, 5, 0);
        let r = TimeRange::new(tc, tc);
        assert!(r.is_err());
    }

    #[test]
    fn test_time_range_duration_frames() {
        let r = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 2, 0)).expect("failed to create");
        assert_eq!(r.duration_frames(), 50); // 2 sec * 25 fps
    }

    #[test]
    fn test_time_range_duration_seconds() {
        let r = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 4, 0)).expect("failed to create");
        let secs = r.duration_seconds(EdlFrameRate::Fps25);
        assert!((secs - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_time_range_contains() {
        let r = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 10, 0)).expect("failed to create");
        assert!(r.contains(&make_tc(0, 0, 5, 0)));
        assert!(!r.contains(&make_tc(0, 0, 10, 0)));
    }

    #[test]
    fn test_time_range_overlaps() {
        let a = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 10, 0)).expect("failed to create");
        let b = TimeRange::new(make_tc(0, 0, 5, 0), make_tc(0, 0, 15, 0)).expect("failed to create");
        assert!(a.overlaps(&b));
    }

    #[test]
    fn test_time_range_no_overlap() {
        let a = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 5, 0)).expect("failed to create");
        let b = TimeRange::new(make_tc(0, 0, 10, 0), make_tc(0, 0, 15, 0)).expect("failed to create");
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_time_range_intersection() {
        let a = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 10, 0)).expect("failed to create");
        let b = TimeRange::new(make_tc(0, 0, 5, 0), make_tc(0, 0, 15, 0)).expect("failed to create");
        let i = a.intersection(&b).expect("intersection should succeed");
        assert_eq!(i.tc_in, make_tc(0, 0, 5, 0));
        assert_eq!(i.tc_out, make_tc(0, 0, 10, 0));
    }

    #[test]
    fn test_time_range_gap_frames() {
        let a = TimeRange::new(make_tc(0, 0, 0, 0), make_tc(0, 0, 5, 0)).expect("failed to create");
        let b = TimeRange::new(make_tc(0, 0, 10, 0), make_tc(0, 0, 15, 0)).expect("failed to create");
        assert_eq!(a.gap_frames(&b), 125); // 5 sec * 25
    }

    #[test]
    fn test_duration_summary() {
        let events = vec![make_event(1, 0, 5), make_event(2, 5, 15)];
        let s = compute_duration_summary(&events).expect("operation should succeed");
        assert_eq!(s.event_count, 2);
        assert_eq!(s.min_frames, 125);
        assert_eq!(s.max_frames, 250);
        assert_eq!(s.total_frames, 375);
    }

    #[test]
    fn test_duration_summary_empty() {
        let result = compute_duration_summary(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_events() {
        let events = vec![make_event(1, 0, 5)];
        let shifted = offset_events(&events, 25, EdlFrameRate::Fps25).expect("operation should succeed");
        assert_eq!(shifted[0].record_in.to_frames(), events[0].record_in.to_frames() + 25);
    }

    #[test]
    fn test_offset_events_negative() {
        let events = vec![make_event(1, 0, 5)];
        let result = offset_events(&events, -999_999, EdlFrameRate::Fps25);
        assert!(result.is_err());
    }

    #[test]
    fn test_programme_duration_frames() {
        let events = vec![make_event(1, 0, 5), make_event(2, 10, 20)];
        let d = programme_duration_frames(&events).expect("operation should succeed");
        assert_eq!(d, 500); // from 0s to 20s = 20*25
    }

    #[test]
    fn test_programme_duration_empty() {
        let result = programme_duration_frames(&[]);
        assert!(result.is_err());
    }
}
