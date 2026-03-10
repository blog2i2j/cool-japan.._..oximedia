#![allow(dead_code)]
//! EDL timeline analysis utilities.
//!
//! This module provides tools for analyzing the temporal structure of EDL files,
//! including gap detection, overlap detection, coverage computation, and
//! timeline region analysis.

/// A time range in the timeline expressed in frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    /// Start frame (inclusive).
    pub start: u64,
    /// End frame (exclusive).
    pub end: u64,
}

impl FrameRange {
    /// Create a new frame range.
    ///
    /// If `start >= end`, the range is considered empty.
    #[must_use]
    pub fn new(start: u64, end: u64) -> Self {
        Self { start, end }
    }

    /// Duration of this range in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }

    /// Check if this range is empty (zero or negative duration).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    /// Check if this range overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if this range contains a specific frame.
    #[must_use]
    pub fn contains_frame(&self, frame: u64) -> bool {
        frame >= self.start && frame < self.end
    }

    /// Compute the intersection of two ranges, if any.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        if start < end {
            Some(Self { start, end })
        } else {
            None
        }
    }

    /// Compute the union of two overlapping or adjacent ranges.
    ///
    /// Returns `None` if the ranges are not overlapping or adjacent.
    #[must_use]
    pub fn union(&self, other: &Self) -> Option<Self> {
        if self.overlaps(other) || self.end == other.start || other.end == self.start {
            Some(Self {
                start: self.start.min(other.start),
                end: self.end.max(other.end),
            })
        } else {
            None
        }
    }
}

impl PartialOrd for FrameRange {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrameRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.start.cmp(&other.start).then(self.end.cmp(&other.end))
    }
}

/// A timeline event used for analysis (simplified representation).
#[derive(Debug, Clone)]
pub struct TimelineEvent {
    /// Event identifier.
    pub event_id: u32,
    /// Record in frame.
    pub record_in: u64,
    /// Record out frame.
    pub record_out: u64,
    /// Optional label (e.g., clip name).
    pub label: Option<String>,
}

impl TimelineEvent {
    /// Create a new timeline event.
    #[must_use]
    pub fn new(event_id: u32, record_in: u64, record_out: u64) -> Self {
        Self {
            event_id,
            record_in,
            record_out,
            label: None,
        }
    }

    /// Create a timeline event with a label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Get the frame range of this event.
    #[must_use]
    pub fn range(&self) -> FrameRange {
        FrameRange::new(self.record_in, self.record_out)
    }

    /// Duration in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.range().duration()
    }
}

/// A detected gap in the timeline.
#[derive(Debug, Clone)]
pub struct TimelineGap {
    /// Frame range of the gap.
    pub range: FrameRange,
    /// Event ID before the gap.
    pub before_event: u32,
    /// Event ID after the gap.
    pub after_event: u32,
}

/// A detected overlap in the timeline.
#[derive(Debug, Clone)]
pub struct TimelineOverlap {
    /// Frame range of the overlap.
    pub range: FrameRange,
    /// First overlapping event ID.
    pub event_a: u32,
    /// Second overlapping event ID.
    pub event_b: u32,
}

/// Timeline analyzer for detecting gaps, overlaps, and computing coverage.
pub struct TimelineAnalyzer {
    /// Events to analyze.
    events: Vec<TimelineEvent>,
}

impl TimelineAnalyzer {
    /// Create a new timeline analyzer.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event to the analyzer.
    pub fn add_event(&mut self, event: TimelineEvent) {
        self.events.push(event);
    }

    /// Add multiple events.
    pub fn add_events(&mut self, events: impl IntoIterator<Item = TimelineEvent>) {
        self.events.extend(events);
    }

    /// Get sorted events (by record_in).
    fn sorted_events(&self) -> Vec<&TimelineEvent> {
        let mut sorted: Vec<&TimelineEvent> = self.events.iter().collect();
        sorted.sort_by_key(|e| e.record_in);
        sorted
    }

    /// Detect gaps in the timeline.
    #[must_use]
    pub fn find_gaps(&self) -> Vec<TimelineGap> {
        let sorted = self.sorted_events();
        let mut gaps = Vec::new();

        for window in sorted.windows(2) {
            let prev = window[0];
            let next = window[1];
            if prev.record_out < next.record_in {
                gaps.push(TimelineGap {
                    range: FrameRange::new(prev.record_out, next.record_in),
                    before_event: prev.event_id,
                    after_event: next.event_id,
                });
            }
        }

        gaps
    }

    /// Detect overlaps in the timeline.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<TimelineOverlap> {
        let sorted = self.sorted_events();
        let mut overlaps = Vec::new();

        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                let a = sorted[i];
                let b = sorted[j];
                // Since sorted by start, if b starts after a ends, no overlap possible
                if b.record_in >= a.record_out {
                    break;
                }
                let overlap_start = b.record_in;
                let overlap_end = a.record_out.min(b.record_out);
                if overlap_start < overlap_end {
                    overlaps.push(TimelineOverlap {
                        range: FrameRange::new(overlap_start, overlap_end),
                        event_a: a.event_id,
                        event_b: b.event_id,
                    });
                }
            }
        }

        overlaps
    }

    /// Compute the total timeline coverage in frames (handling overlaps).
    #[must_use]
    pub fn total_coverage_frames(&self) -> u64 {
        let merged = self.merged_ranges();
        merged.iter().map(|r| r.duration()).sum()
    }

    /// Compute merged (non-overlapping) ranges covering all events.
    #[must_use]
    pub fn merged_ranges(&self) -> Vec<FrameRange> {
        if self.events.is_empty() {
            return Vec::new();
        }

        let mut ranges: Vec<FrameRange> = self.events.iter().map(|e| e.range()).collect();
        ranges.sort();

        let mut merged: Vec<FrameRange> = Vec::new();
        let mut current = ranges[0];

        for &r in &ranges[1..] {
            if let Some(u) = current.union(&r) {
                current = u;
            } else {
                merged.push(current);
                current = r;
            }
        }
        merged.push(current);
        merged
    }

    /// Compute coverage percentage over a given total range.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn coverage_percentage(&self, total_range: &FrameRange) -> f64 {
        if total_range.is_empty() {
            return 0.0;
        }
        let covered = self.total_coverage_frames();
        covered as f64 / total_range.duration() as f64 * 100.0
    }

    /// Return the number of events loaded.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl Default for TimelineAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_range_basic() {
        let r = FrameRange::new(0, 100);
        assert_eq!(r.duration(), 100);
        assert!(!r.is_empty());
    }

    #[test]
    fn test_frame_range_empty() {
        let r = FrameRange::new(100, 50);
        assert!(r.is_empty());
        assert_eq!(r.duration(), 0);
    }

    #[test]
    fn test_frame_range_overlaps() {
        let a = FrameRange::new(0, 100);
        let b = FrameRange::new(50, 150);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn test_frame_range_no_overlap() {
        let a = FrameRange::new(0, 50);
        let b = FrameRange::new(50, 100);
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn test_frame_range_contains_frame() {
        let r = FrameRange::new(10, 20);
        assert!(r.contains_frame(10));
        assert!(r.contains_frame(15));
        assert!(!r.contains_frame(20));
        assert!(!r.contains_frame(9));
    }

    #[test]
    fn test_frame_range_intersection() {
        let a = FrameRange::new(0, 100);
        let b = FrameRange::new(50, 150);
        let inter = a.intersection(&b).expect("intersection should succeed");
        assert_eq!(inter.start, 50);
        assert_eq!(inter.end, 100);
    }

    #[test]
    fn test_frame_range_intersection_none() {
        let a = FrameRange::new(0, 50);
        let b = FrameRange::new(50, 100);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_frame_range_union() {
        let a = FrameRange::new(0, 50);
        let b = FrameRange::new(50, 100);
        let u = a.union(&b).expect("union should succeed");
        assert_eq!(u.start, 0);
        assert_eq!(u.end, 100);
    }

    #[test]
    fn test_frame_range_union_disjoint() {
        let a = FrameRange::new(0, 40);
        let b = FrameRange::new(60, 100);
        assert!(a.union(&b).is_none());
    }

    #[test]
    fn test_timeline_event_creation() {
        let ev = TimelineEvent::new(1, 0, 125);
        assert_eq!(ev.event_id, 1);
        assert_eq!(ev.duration(), 125);
        assert!(ev.label.is_none());
    }

    #[test]
    fn test_timeline_event_with_label() {
        let ev = TimelineEvent::new(1, 0, 100).with_label("Shot A");
        assert_eq!(ev.label.as_deref(), Some("Shot A"));
    }

    #[test]
    fn test_find_gaps() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 150, 250));

        let gaps = analyzer.find_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].range.start, 100);
        assert_eq!(gaps[0].range.end, 150);
        assert_eq!(gaps[0].before_event, 1);
        assert_eq!(gaps[0].after_event, 2);
    }

    #[test]
    fn test_find_no_gaps() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 100, 200));

        let gaps = analyzer.find_gaps();
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_find_overlaps() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 80, 180));

        let overlaps = analyzer.find_overlaps();
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0].range.start, 80);
        assert_eq!(overlaps[0].range.end, 100);
    }

    #[test]
    fn test_find_no_overlaps() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 100, 200));

        let overlaps = analyzer.find_overlaps();
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_total_coverage_no_overlap() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 100, 200));

        assert_eq!(analyzer.total_coverage_frames(), 200);
    }

    #[test]
    fn test_total_coverage_with_overlap() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 50, 150));

        assert_eq!(analyzer.total_coverage_frames(), 150);
    }

    #[test]
    fn test_coverage_percentage() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 50));
        let total = FrameRange::new(0, 100);
        let pct = analyzer.coverage_percentage(&total);
        assert!((pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merged_ranges() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        analyzer.add_event(TimelineEvent::new(2, 50, 150));
        analyzer.add_event(TimelineEvent::new(3, 200, 300));

        let merged = analyzer.merged_ranges();
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], FrameRange::new(0, 150));
        assert_eq!(merged[1], FrameRange::new(200, 300));
    }

    #[test]
    fn test_analyzer_clear() {
        let mut analyzer = TimelineAnalyzer::new();
        analyzer.add_event(TimelineEvent::new(1, 0, 100));
        assert_eq!(analyzer.event_count(), 1);
        analyzer.clear();
        assert_eq!(analyzer.event_count(), 0);
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = TimelineAnalyzer::default();
        assert_eq!(analyzer.event_count(), 0);
    }
}
