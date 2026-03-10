//! Timecode continuity checking and gap detection.
//!
//! Provides tools for detecting discontinuities, gaps, and overlaps in timecode streams.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{FrameRate, Timecode, TimecodeError};

/// A detected gap or discontinuity in a timecode sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimecodegGap {
    /// Timecode immediately before the gap.
    pub before: Timecode,
    /// Timecode immediately after the gap.
    pub after: Timecode,
    /// Size of the gap in frames (positive = gap, negative = overlap).
    pub gap_frames: i64,
}

/// Result of a continuity check on a single transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContinuityResult {
    /// Timecodes are exactly one frame apart (continuous).
    Continuous,
    /// A gap of N frames was detected.
    Gap(u64),
    /// An overlap (reverse discontinuity) of N frames was detected.
    Overlap(u64),
    /// Same timecode repeated.
    Repeat,
}

/// Check continuity between two consecutive timecodes.
pub fn check_continuity(prev: &Timecode, next: &Timecode) -> ContinuityResult {
    let prev_f = prev.to_frames();
    let next_f = next.to_frames();

    match next_f.cmp(&(prev_f + 1)) {
        std::cmp::Ordering::Equal => ContinuityResult::Continuous,
        std::cmp::Ordering::Greater => ContinuityResult::Gap(next_f - prev_f - 1),
        std::cmp::Ordering::Less => {
            if next_f == prev_f {
                ContinuityResult::Repeat
            } else {
                ContinuityResult::Overlap(prev_f + 1 - next_f)
            }
        }
    }
}

/// Continuity monitor for a stream of timecodes.
#[derive(Debug, Clone)]
pub struct ContinuityMonitor {
    frame_rate: FrameRate,
    last_tc: Option<Timecode>,
    gaps: Vec<TimecodegGap>,
    gap_count: u32,
    overlap_count: u32,
    repeat_count: u32,
    frame_count: u64,
}

impl ContinuityMonitor {
    /// Create a new continuity monitor.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            last_tc: None,
            gaps: Vec::new(),
            gap_count: 0,
            overlap_count: 0,
            repeat_count: 0,
            frame_count: 0,
        }
    }

    /// Feed a timecode to the monitor and return the continuity result.
    pub fn feed(&mut self, tc: Timecode) -> ContinuityResult {
        self.frame_count += 1;
        let result = if let Some(ref last) = self.last_tc {
            let r = check_continuity(last, &tc);
            match &r {
                ContinuityResult::Gap(n) => {
                    self.gap_count += 1;
                    self.gaps.push(TimecodegGap {
                        before: *last,
                        after: tc,
                        gap_frames: *n as i64,
                    });
                }
                ContinuityResult::Overlap(n) => {
                    self.overlap_count += 1;
                    self.gaps.push(TimecodegGap {
                        before: *last,
                        after: tc,
                        gap_frames: -(*n as i64),
                    });
                }
                ContinuityResult::Repeat => {
                    self.repeat_count += 1;
                }
                ContinuityResult::Continuous => {}
            }
            r
        } else {
            ContinuityResult::Continuous
        };
        self.last_tc = Some(tc);
        result
    }

    /// Get number of gaps detected.
    pub fn gap_count(&self) -> u32 {
        self.gap_count
    }

    /// Get number of overlaps detected.
    pub fn overlap_count(&self) -> u32 {
        self.overlap_count
    }

    /// Get number of repeated timecodes.
    pub fn repeat_count(&self) -> u32 {
        self.repeat_count
    }

    /// Get total frames processed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get all recorded gaps and overlaps.
    pub fn gaps(&self) -> &[TimecodegGap] {
        &self.gaps
    }

    /// Reset the monitor state.
    pub fn reset(&mut self) {
        self.last_tc = None;
        self.gaps.clear();
        self.gap_count = 0;
        self.overlap_count = 0;
        self.repeat_count = 0;
        self.frame_count = 0;
    }

    /// Get the last seen timecode.
    pub fn last_timecode(&self) -> Option<&Timecode> {
        self.last_tc.as_ref()
    }

    /// Generate a summary report string.
    pub fn report(&self) -> String {
        format!(
            "Frames: {}, Gaps: {}, Overlaps: {}, Repeats: {}",
            self.frame_count, self.gap_count, self.overlap_count, self.repeat_count
        )
    }
}

/// Expected frame count between two timecodes.
pub fn expected_frame_count(start: &Timecode, end: &Timecode) -> u64 {
    let s = start.to_frames();
    let e = end.to_frames();
    e.saturating_sub(s)
}

/// Find all gaps in a slice of timecodes.
pub fn find_gaps(timecodes: &[Timecode]) -> Vec<TimecodegGap> {
    let mut gaps = Vec::new();
    for window in timecodes.windows(2) {
        let result = check_continuity(&window[0], &window[1]);
        match result {
            ContinuityResult::Gap(n) => {
                gaps.push(TimecodegGap {
                    before: window[0],
                    after: window[1],
                    gap_frames: n as i64,
                });
            }
            ContinuityResult::Overlap(n) => {
                gaps.push(TimecodegGap {
                    before: window[0],
                    after: window[1],
                    gap_frames: -(n as i64),
                });
            }
            _ => {}
        }
    }
    gaps
}

/// Timecode range representing a continuous segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimecodeRange {
    /// Start of the range.
    pub start: Timecode,
    /// End of the range (inclusive).
    pub end: Timecode,
}

impl TimecodeRange {
    /// Create a new range.
    pub fn new(start: Timecode, end: Timecode) -> Result<Self, TimecodeError> {
        if end.to_frames() < start.to_frames() {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self { start, end })
    }

    /// Get the duration in frames.
    pub fn duration_frames(&self) -> u64 {
        self.end.to_frames().saturating_sub(self.start.to_frames()) + 1
    }

    /// Check if a timecode falls within this range.
    pub fn contains(&self, tc: &Timecode) -> bool {
        let f = tc.to_frames();
        f >= self.start.to_frames() && f <= self.end.to_frames()
    }

    /// Check if two ranges overlap.
    pub fn overlaps(&self, other: &TimecodeRange) -> bool {
        self.start.to_frames() <= other.end.to_frames()
            && other.start.to_frames() <= self.end.to_frames()
    }
}

/// Split a slice of timecodes into continuous segments.
pub fn split_into_segments(timecodes: &[Timecode]) -> Vec<Vec<Timecode>> {
    if timecodes.is_empty() {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut current = vec![timecodes[0]];

    for window in timecodes.windows(2) {
        match check_continuity(&window[0], &window[1]) {
            ContinuityResult::Continuous => {
                current.push(window[1]);
            }
            _ => {
                segments.push(current);
                current = vec![window[1]];
            }
        }
    }
    segments.push(current);
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_check_continuity_continuous() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 0, 1);
        assert_eq!(check_continuity(&a, &b), ContinuityResult::Continuous);
    }

    #[test]
    fn test_check_continuity_gap() {
        let a = tc(0, 0, 0, 0);
        let b = tc(0, 0, 0, 5);
        assert_eq!(check_continuity(&a, &b), ContinuityResult::Gap(4));
    }

    #[test]
    fn test_check_continuity_overlap() {
        let a = tc(0, 0, 0, 5);
        let b = tc(0, 0, 0, 3);
        assert_eq!(check_continuity(&a, &b), ContinuityResult::Overlap(3));
    }

    #[test]
    fn test_check_continuity_repeat() {
        let a = tc(0, 0, 0, 5);
        let b = tc(0, 0, 0, 5);
        assert_eq!(check_continuity(&a, &b), ContinuityResult::Repeat);
    }

    #[test]
    fn test_monitor_continuous() {
        let mut mon = ContinuityMonitor::new(FrameRate::Fps25);
        for f in 0u8..10 {
            let t = tc(0, 0, 0, f);
            mon.feed(t);
        }
        assert_eq!(mon.gap_count(), 0);
        assert_eq!(mon.frame_count(), 10);
    }

    #[test]
    fn test_monitor_gap_detection() {
        let mut mon = ContinuityMonitor::new(FrameRate::Fps25);
        mon.feed(tc(0, 0, 0, 0));
        mon.feed(tc(0, 0, 0, 5)); // gap of 4
        assert_eq!(mon.gap_count(), 1);
        assert_eq!(mon.gaps()[0].gap_frames, 4);
    }

    #[test]
    fn test_monitor_reset() {
        let mut mon = ContinuityMonitor::new(FrameRate::Fps25);
        mon.feed(tc(0, 0, 0, 0));
        mon.feed(tc(0, 0, 0, 5));
        mon.reset();
        assert_eq!(mon.gap_count(), 0);
        assert_eq!(mon.frame_count(), 0);
        assert!(mon.last_timecode().is_none());
    }

    #[test]
    fn test_monitor_report() {
        let mut mon = ContinuityMonitor::new(FrameRate::Fps25);
        mon.feed(tc(0, 0, 0, 0));
        let report = mon.report();
        assert!(report.contains("Frames: 1"));
    }

    #[test]
    fn test_find_gaps() {
        let tcs = vec![
            tc(0, 0, 0, 0),
            tc(0, 0, 0, 1),
            tc(0, 0, 0, 5),
            tc(0, 0, 0, 6),
        ];
        let gaps = find_gaps(&tcs);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].gap_frames, 3);
    }

    #[test]
    fn test_timecode_range_contains() {
        let start = tc(0, 0, 0, 0);
        let end = tc(0, 0, 1, 0);
        let range = TimecodeRange::new(start, end).expect("valid timecode range");
        assert!(range.contains(&tc(0, 0, 0, 10)));
        assert!(!range.contains(&tc(0, 0, 2, 0)));
    }

    #[test]
    fn test_timecode_range_duration() {
        let start = tc(0, 0, 0, 0);
        let end = tc(0, 0, 0, 24);
        let range = TimecodeRange::new(start, end).expect("valid timecode range");
        assert_eq!(range.duration_frames(), 25);
    }

    #[test]
    fn test_timecode_range_overlaps() {
        let r1 = TimecodeRange::new(tc(0, 0, 0, 0), tc(0, 0, 0, 10)).expect("valid timecode range");
        let r2 = TimecodeRange::new(tc(0, 0, 0, 5), tc(0, 0, 0, 20)).expect("valid timecode range");
        let r3 = TimecodeRange::new(tc(0, 0, 1, 0), tc(0, 0, 1, 10)).expect("valid timecode range");
        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_split_into_segments() {
        let tcs = vec![
            tc(0, 0, 0, 0),
            tc(0, 0, 0, 1),
            tc(0, 0, 0, 2),
            tc(0, 0, 1, 0),
            tc(0, 0, 1, 1),
        ];
        let segments = split_into_segments(&tcs);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 3);
        assert_eq!(segments[1].len(), 2);
    }

    #[test]
    fn test_expected_frame_count() {
        let start = tc(0, 0, 0, 0);
        let end = tc(0, 0, 1, 0);
        assert_eq!(expected_frame_count(&start, &end), 25);
    }
}
