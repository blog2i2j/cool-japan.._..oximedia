//! Schedule gap detector for broadcast automation.
//!
//! Detects empty gaps in broadcast playout schedules and categorises them
//! so that fill content can be automatically inserted.  A gap is any
//! contiguous period within a broadcast window that contains no scheduled
//! content item.
//!
//! # Design
//!
//! The detector accepts an ordered list of [`ScheduleSegment`]s (each with a
//! start frame and duration), sorts them by start frame, and identifies
//! uncovered regions within a given broadcast window.  Each uncovered region
//! is returned as a [`ScheduleGap`] with additional metadata about how the gap
//! should be handled.
//!
//! # Example
//!
//! ```rust
//! use oximedia_automation::gap_detector::{GapDetector, ScheduleSegment};
//!
//! let segments = vec![
//!     ScheduleSegment::new("seg-1", 0, 500),
//!     ScheduleSegment::new("seg-2", 600, 400), // gap from 500..600
//! ];
//!
//! let detector = GapDetector::new(25);
//! let gaps = detector.detect(&segments, 0, 1000);
//! assert_eq!(gaps.len(), 1);
//! assert_eq!(gaps[0].start_frame, 500);
//! assert_eq!(gaps[0].duration_frames, 100);
//! ```

use std::cmp;

// ---------------------------------------------------------------------------
// FillStrategy
// ---------------------------------------------------------------------------

/// Recommended strategy for filling a detected gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillStrategy {
    /// Insert a standard filler clip (bumper, countdown, etc.).
    FillerClip,
    /// Repeat the previous segment if it supports looping.
    LoopPrevious,
    /// Fade to black and hold until the next segment.
    FadeToBlack,
    /// Insert a slate / test card.
    Slate,
    /// No fill — leave the gap empty (rare, only for very short gaps).
    None,
}

impl FillStrategy {
    /// Return a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            Self::FillerClip => "Insert filler clip",
            Self::LoopPrevious => "Loop previous segment",
            Self::FadeToBlack => "Fade to black",
            Self::Slate => "Display slate / test card",
            Self::None => "Leave gap empty",
        }
    }
}

// ---------------------------------------------------------------------------
// GapSeverity
// ---------------------------------------------------------------------------

/// Operational severity of a detected gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GapSeverity {
    /// Less than 1 second — likely a frame-accurate transition; acceptable.
    Minor,
    /// 1–30 seconds — should be filled but not immediately critical.
    Moderate,
    /// 30 seconds to 5 minutes — significant gap requiring fill content.
    Significant,
    /// More than 5 minutes — critical gap that may result in dead air.
    Critical,
}

impl GapSeverity {
    /// Determine severity from the gap duration in frames at the given fps.
    pub fn from_frames(duration_frames: u64, fps: u32) -> Self {
        if fps == 0 {
            return Self::Critical;
        }
        let seconds = duration_frames / fps as u64;
        if seconds < 1 {
            Self::Minor
        } else if seconds < 30 {
            Self::Moderate
        } else if seconds < 300 {
            Self::Significant
        } else {
            Self::Critical
        }
    }
}

// ---------------------------------------------------------------------------
// ScheduleSegment
// ---------------------------------------------------------------------------

/// A single scheduled content segment within a broadcast window.
#[derive(Debug, Clone)]
pub struct ScheduleSegment {
    /// Segment identifier (clip ID, programme ID, etc.).
    pub id: String,
    /// First frame of this segment (inclusive).
    pub start_frame: u64,
    /// Duration in frames.
    pub duration_frames: u64,
}

impl ScheduleSegment {
    /// Create a new segment.
    pub fn new(id: impl Into<String>, start_frame: u64, duration_frames: u64) -> Self {
        Self {
            id: id.into(),
            start_frame,
            duration_frames,
        }
    }

    /// Last frame of this segment (exclusive end).
    pub fn end_frame(&self) -> u64 {
        self.start_frame + self.duration_frames
    }
}

// ---------------------------------------------------------------------------
// ScheduleGap
// ---------------------------------------------------------------------------

/// A detected gap in the broadcast schedule.
#[derive(Debug, Clone)]
pub struct ScheduleGap {
    /// First frame of the gap (inclusive).
    pub start_frame: u64,
    /// Gap duration in frames.
    pub duration_frames: u64,
    /// Operational severity of this gap.
    pub severity: GapSeverity,
    /// Recommended strategy for filling this gap.
    pub fill_strategy: FillStrategy,
    /// Identifier of the segment that precedes this gap, if any.
    pub preceding_segment_id: Option<String>,
    /// Identifier of the segment that follows this gap, if any.
    pub following_segment_id: Option<String>,
}

impl ScheduleGap {
    /// Exclusive end frame of the gap.
    pub fn end_frame(&self) -> u64 {
        self.start_frame + self.duration_frames
    }

    /// Whether this gap falls at the start of the broadcast window
    /// (no preceding segment).
    pub fn is_leading_gap(&self) -> bool {
        self.preceding_segment_id.is_none()
    }

    /// Whether this gap falls at the end of the broadcast window
    /// (no following segment).
    pub fn is_trailing_gap(&self) -> bool {
        self.following_segment_id.is_none()
    }
}

// ---------------------------------------------------------------------------
// OverlapReport
// ---------------------------------------------------------------------------

/// A detected overlap between two schedule segments.
#[derive(Debug, Clone)]
pub struct OverlapReport {
    /// First segment's identifier.
    pub segment_a_id: String,
    /// Second segment's identifier.
    pub segment_b_id: String,
    /// First frame of the overlap (inclusive).
    pub overlap_start: u64,
    /// Duration of the overlap in frames.
    pub overlap_frames: u64,
}

// ---------------------------------------------------------------------------
// GapDetector
// ---------------------------------------------------------------------------

/// Detects gaps and overlaps in broadcast playout schedules.
///
/// Create once with [`GapDetector::new`], then call [`detect`][GapDetector::detect]
/// or [`detect_overlaps`][GapDetector::detect_overlaps] as needed.
#[derive(Debug, Clone)]
pub struct GapDetector {
    /// Frame rate used for severity classification.
    fps: u32,
    /// Minimum gap size (in frames) to report.  Gaps smaller than this are
    /// silently ignored.  Defaults to 0 (report all gaps).
    min_gap_frames: u64,
}

impl GapDetector {
    /// Create a new gap detector for the given frame rate.
    pub fn new(fps: u32) -> Self {
        Self { fps, min_gap_frames: 0 }
    }

    /// Set the minimum gap size in frames.  Gaps shorter than this are ignored.
    pub fn with_min_gap_frames(mut self, frames: u64) -> Self {
        self.min_gap_frames = frames;
        self
    }

    /// Detect gaps in `segments` within the broadcast window
    /// `[window_start, window_end)`.
    ///
    /// Segments outside the window are clipped.  Overlapping segments are
    /// merged before gap detection so that overlaps do not produce false gaps.
    ///
    /// Returns gaps sorted by `start_frame` in ascending order.
    pub fn detect(
        &self,
        segments: &[ScheduleSegment],
        window_start: u64,
        window_end: u64,
    ) -> Vec<ScheduleGap> {
        if window_start >= window_end {
            return Vec::new();
        }

        // Sort a working copy by start frame.
        let mut sorted: Vec<&ScheduleSegment> = segments.iter().collect();
        sorted.sort_by_key(|s| s.start_frame);

        // Build merged intervals clipped to the window.
        let mut intervals: Vec<(u64, u64, String, String)> = Vec::new(); // (start, end, first_id, last_id)

        for seg in &sorted {
            let seg_start = cmp::max(seg.start_frame, window_start);
            let seg_end = cmp::min(seg.end_frame(), window_end);
            if seg_start >= seg_end {
                continue; // entirely outside window
            }

            if let Some(last) = intervals.last_mut() {
                if seg_start <= last.1 {
                    // Overlaps or is adjacent — extend the merged interval.
                    if seg_end > last.1 {
                        last.1 = seg_end;
                        last.3 = seg.id.clone();
                    }
                    continue;
                }
            }
            intervals.push((seg_start, seg_end, seg.id.clone(), seg.id.clone()));
        }

        // Walk the merged intervals to find gaps.
        let mut gaps = Vec::new();
        let mut cursor = window_start;

        for (idx, (start, end, first_id, _last_id)) in intervals.iter().enumerate() {
            if *start > cursor {
                let gap_dur = start - cursor;
                if gap_dur >= self.min_gap_frames {
                    let preceding = if idx == 0 {
                        None
                    } else {
                        Some(intervals[idx - 1].3.clone())
                    };
                    let following = Some(first_id.clone());
                    gaps.push(self.build_gap(cursor, gap_dur, preceding, following));
                }
            }
            cursor = cmp::max(cursor, *end);
        }

        // Trailing gap after the last segment.
        if cursor < window_end {
            let gap_dur = window_end - cursor;
            if gap_dur >= self.min_gap_frames {
                let preceding = intervals.last().map(|(_, _, _, last_id)| last_id.clone());
                gaps.push(self.build_gap(cursor, gap_dur, preceding, None));
            }
        }

        gaps
    }

    /// Detect overlaps between segments.
    ///
    /// Returns a list of [`OverlapReport`]s describing pairs of segments that
    /// overlap in time.  The input is sorted by start frame before checking.
    pub fn detect_overlaps(&self, segments: &[ScheduleSegment]) -> Vec<OverlapReport> {
        let mut sorted: Vec<&ScheduleSegment> = segments.iter().collect();
        sorted.sort_by_key(|s| s.start_frame);

        let mut overlaps = Vec::new();

        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                let a = sorted[i];
                let b = sorted[j];
                if b.start_frame >= a.end_frame() {
                    break; // b starts after a ends — no more overlaps for a
                }
                let overlap_start = b.start_frame;
                let overlap_end = cmp::min(a.end_frame(), b.end_frame());
                let overlap_frames = overlap_end.saturating_sub(overlap_start);
                if overlap_frames > 0 {
                    overlaps.push(OverlapReport {
                        segment_a_id: a.id.clone(),
                        segment_b_id: b.id.clone(),
                        overlap_start,
                        overlap_frames,
                    });
                }
            }
        }

        overlaps
    }

    /// Calculate total coverage (sum of non-overlapping scheduled frames) within the window.
    pub fn coverage_frames(
        &self,
        segments: &[ScheduleSegment],
        window_start: u64,
        window_end: u64,
    ) -> u64 {
        if window_start >= window_end {
            return 0;
        }
        let window_size = window_end - window_start;
        let gaps = self.detect(segments, window_start, window_end);
        let gap_total: u64 = gaps.iter().map(|g| g.duration_frames).sum();
        window_size.saturating_sub(gap_total)
    }

    /// Calculate fill coverage as a percentage (0.0 – 100.0).
    pub fn coverage_percent(
        &self,
        segments: &[ScheduleSegment],
        window_start: u64,
        window_end: u64,
    ) -> f64 {
        if window_start >= window_end {
            return 100.0;
        }
        let window_size = (window_end - window_start) as f64;
        let covered = self.coverage_frames(segments, window_start, window_end) as f64;
        (covered / window_size) * 100.0
    }

    // --- Private helpers ---

    fn build_gap(
        &self,
        start: u64,
        duration: u64,
        preceding: Option<String>,
        following: Option<String>,
    ) -> ScheduleGap {
        let severity = GapSeverity::from_frames(duration, self.fps);
        let fill_strategy = Self::choose_fill_strategy(severity, preceding.is_some());
        ScheduleGap {
            start_frame: start,
            duration_frames: duration,
            severity,
            fill_strategy,
            preceding_segment_id: preceding,
            following_segment_id: following,
        }
    }

    fn choose_fill_strategy(severity: GapSeverity, has_preceding: bool) -> FillStrategy {
        match severity {
            GapSeverity::Minor => FillStrategy::None,
            GapSeverity::Moderate => {
                if has_preceding {
                    FillStrategy::LoopPrevious
                } else {
                    FillStrategy::FillerClip
                }
            }
            GapSeverity::Significant => FillStrategy::FillerClip,
            GapSeverity::Critical => FillStrategy::Slate,
        }
    }
}

impl Default for GapDetector {
    fn default() -> Self {
        Self::new(25)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: &str, start: u64, dur: u64) -> ScheduleSegment {
        ScheduleSegment::new(id, start, dur)
    }

    #[test]
    fn test_no_segments_is_one_full_gap() {
        let d = GapDetector::new(25);
        let gaps = d.detect(&[], 0, 1000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 0);
        assert_eq!(gaps[0].duration_frames, 1000);
        assert!(gaps[0].is_leading_gap());
        assert!(gaps[0].is_trailing_gap());
    }

    #[test]
    fn test_full_coverage_no_gaps() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 500), seg("b", 500, 500)];
        let gaps = d.detect(&segs, 0, 1000);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_gap_between_two_segments() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 400), seg("b", 500, 400)];
        let gaps = d.detect(&segs, 0, 900);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 400);
        assert_eq!(gaps[0].duration_frames, 100);
        assert_eq!(gaps[0].preceding_segment_id.as_deref(), Some("a"));
        assert_eq!(gaps[0].following_segment_id.as_deref(), Some("b"));
    }

    #[test]
    fn test_leading_gap() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 100, 900)];
        let gaps = d.detect(&segs, 0, 1000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 0);
        assert_eq!(gaps[0].duration_frames, 100);
        assert!(gaps[0].is_leading_gap());
        assert!(!gaps[0].is_trailing_gap());
    }

    #[test]
    fn test_trailing_gap() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 800)];
        let gaps = d.detect(&segs, 0, 1000);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start_frame, 800);
        assert_eq!(gaps[0].duration_frames, 200);
        assert!(!gaps[0].is_leading_gap());
        assert!(gaps[0].is_trailing_gap());
    }

    #[test]
    fn test_overlapping_segments_merged() {
        let d = GapDetector::new(25);
        // Segments overlap from 400..600
        let segs = vec![seg("a", 0, 600), seg("b", 400, 600)];
        let gaps = d.detect(&segs, 0, 1000);
        // Merged interval is 0..1000; no gaps
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_segments_outside_window_ignored() {
        let d = GapDetector::new(25);
        let segs = vec![
            seg("before", 0, 100),   // entirely before window
            seg("inside", 500, 200), // inside window
            seg("after", 900, 200),  // entirely after window
        ];
        let gaps = d.detect(&segs, 200, 900);
        // Inside segment: 500..700; gap before: 200..500; gap after: 700..900
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].start_frame, 200);
        assert_eq!(gaps[0].duration_frames, 300);
        assert_eq!(gaps[1].start_frame, 700);
        assert_eq!(gaps[1].duration_frames, 200);
    }

    #[test]
    fn test_min_gap_frames_filter() {
        let d = GapDetector::new(25).with_min_gap_frames(50);
        let segs = vec![seg("a", 0, 490), seg("b", 500, 500)];
        // gap is 10 frames < min_gap_frames(50) — should be filtered
        let gaps = d.detect(&segs, 0, 1000);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_severity_minor_under_1s() {
        let sev = GapSeverity::from_frames(24, 25); // just under 1 second
        assert_eq!(sev, GapSeverity::Minor);
    }

    #[test]
    fn test_severity_moderate_1_to_30s() {
        let sev = GapSeverity::from_frames(25, 25); // exactly 1 second
        assert_eq!(sev, GapSeverity::Moderate);
        let sev2 = GapSeverity::from_frames(25 * 29, 25); // 29 seconds
        assert_eq!(sev2, GapSeverity::Moderate);
    }

    #[test]
    fn test_severity_significant_30s_to_5min() {
        let sev = GapSeverity::from_frames(25 * 30, 25); // 30 seconds
        assert_eq!(sev, GapSeverity::Significant);
    }

    #[test]
    fn test_severity_critical_over_5min() {
        let sev = GapSeverity::from_frames(25 * 300, 25); // 5 minutes
        assert_eq!(sev, GapSeverity::Critical);
    }

    #[test]
    fn test_coverage_frames_full_coverage() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 1000)];
        assert_eq!(d.coverage_frames(&segs, 0, 1000), 1000);
    }

    #[test]
    fn test_coverage_percent() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 500)];
        let pct = d.coverage_percent(&segs, 0, 1000);
        assert!((pct - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_detect_overlaps_none() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 100), seg("b", 100, 100)];
        let overlaps = d.detect_overlaps(&segs);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_detect_overlaps_found() {
        let d = GapDetector::new(25);
        let segs = vec![seg("a", 0, 200), seg("b", 100, 200)];
        let overlaps = d.detect_overlaps(&segs);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].overlap_start, 100);
        assert_eq!(overlaps[0].overlap_frames, 100);
    }

    #[test]
    fn test_empty_window_returns_no_gaps() {
        let d = GapDetector::new(25);
        let gaps = d.detect(&[], 100, 100);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_fill_strategy_descriptions_non_empty() {
        for strat in &[
            FillStrategy::FillerClip,
            FillStrategy::LoopPrevious,
            FillStrategy::FadeToBlack,
            FillStrategy::Slate,
            FillStrategy::None,
        ] {
            assert!(!strat.description().is_empty());
        }
    }

    #[test]
    fn test_default_gap_detector_fps_25() {
        let d = GapDetector::default();
        // Should not panic; verify by running a basic detection
        let gaps = d.detect(&[], 0, 25);
        assert_eq!(gaps.len(), 1);
    }
}
