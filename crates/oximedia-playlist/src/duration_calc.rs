#![allow(dead_code)]

//! Duration calculation and gap analysis for playlists.
//!
//! Provides utilities to compute total playlist durations, detect gaps
//! between items, and convert between timing units used in broadcast.

use std::time::Duration;

/// Unit of duration measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DurationUnit {
    /// Duration expressed in frames at a given frame rate.
    Frames,
    /// Duration expressed in milliseconds.
    Milliseconds,
    /// Duration expressed in seconds.
    Seconds,
    /// Duration expressed in SMPTE timecode (HH:MM:SS:FF).
    Timecode,
}

/// A single timed entry for duration analysis.
#[derive(Debug, Clone)]
pub struct TimedEntry {
    /// Human-readable label.
    pub label: String,
    /// Start offset from the playlist origin.
    pub start: Duration,
    /// Duration of this entry.
    pub duration: Duration,
}

impl TimedEntry {
    /// Create a new timed entry.
    pub fn new(label: impl Into<String>, start: Duration, duration: Duration) -> Self {
        Self {
            label: label.into(),
            start,
            duration,
        }
    }

    /// End time of this entry (start + duration).
    pub fn end(&self) -> Duration {
        self.start + self.duration
    }
}

/// Describes a gap detected between two consecutive entries.
#[derive(Debug, Clone)]
pub struct GapInfo {
    /// Index of the entry before the gap.
    pub before_index: usize,
    /// Index of the entry after the gap.
    pub after_index: usize,
    /// Size of the gap.
    pub gap_duration: Duration,
}

/// Describes an overlap between two consecutive entries.
#[derive(Debug, Clone)]
pub struct OverlapInfo {
    /// Index of the first overlapping entry.
    pub first_index: usize,
    /// Index of the second overlapping entry.
    pub second_index: usize,
    /// Amount of overlap.
    pub overlap_duration: Duration,
}

/// Report produced by gap analysis.
#[derive(Debug, Clone)]
pub struct GapReport {
    /// Total duration of all entries combined.
    pub total_content_duration: Duration,
    /// Effective span from first start to last end.
    pub span_duration: Duration,
    /// Detected gaps.
    pub gaps: Vec<GapInfo>,
    /// Detected overlaps.
    pub overlaps: Vec<OverlapInfo>,
}

/// Calculator for playlist duration and gap analysis.
#[derive(Debug, Clone)]
pub struct DurationCalculator {
    entries: Vec<TimedEntry>,
    frame_rate: f64,
}

impl DurationCalculator {
    /// Create a new calculator with a given frame rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(frame_rate: f64) -> Self {
        Self {
            entries: Vec::new(),
            frame_rate,
        }
    }

    /// Add an entry.
    pub fn add_entry(&mut self, entry: TimedEntry) {
        self.entries.push(entry);
    }

    /// Number of entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Total content duration (sum of all individual durations).
    pub fn total_content_duration(&self) -> Duration {
        self.entries.iter().map(|e| e.duration).sum()
    }

    /// Span from earliest start to latest end.
    pub fn span_duration(&self) -> Duration {
        if self.entries.is_empty() {
            return Duration::ZERO;
        }
        let min_start = self
            .entries
            .iter()
            .map(|e| e.start)
            .min()
            .unwrap_or_default();
        let max_end = self
            .entries
            .iter()
            .map(TimedEntry::end)
            .max()
            .unwrap_or_default();
        max_end.saturating_sub(min_start)
    }

    /// Convert a duration to the specified unit.
    #[allow(clippy::cast_precision_loss)]
    pub fn convert(&self, dur: Duration, unit: DurationUnit) -> f64 {
        let secs = dur.as_secs_f64();
        match unit {
            DurationUnit::Frames => secs * self.frame_rate,
            DurationUnit::Milliseconds => secs * 1000.0,
            DurationUnit::Seconds => secs,
            DurationUnit::Timecode => secs, // returns raw seconds for timecode formatting
        }
    }

    /// Convert frames to duration.
    #[allow(clippy::cast_precision_loss)]
    pub fn frames_to_duration(&self, frames: u64) -> Duration {
        if self.frame_rate <= 0.0 {
            return Duration::ZERO;
        }
        let secs = frames as f64 / self.frame_rate;
        Duration::from_secs_f64(secs)
    }

    /// Run gap analysis on entries sorted by start time.
    pub fn gap_analysis(&self) -> GapReport {
        let mut sorted: Vec<(usize, &TimedEntry)> = self.entries.iter().enumerate().collect();
        sorted.sort_by(|a, b| a.1.start.cmp(&b.1.start));

        let mut gaps = Vec::new();
        let mut overlaps = Vec::new();

        for window in sorted.windows(2) {
            let (i, prev) = window[0];
            let (j, next) = window[1];
            let prev_end = prev.end();
            if prev_end < next.start {
                gaps.push(GapInfo {
                    before_index: i,
                    after_index: j,
                    gap_duration: next.start.checked_sub(prev_end).unwrap_or(Duration::ZERO),
                });
            } else if prev_end > next.start {
                overlaps.push(OverlapInfo {
                    first_index: i,
                    second_index: j,
                    overlap_duration: prev_end.checked_sub(next.start).unwrap_or(Duration::ZERO),
                });
            }
        }

        GapReport {
            total_content_duration: self.total_content_duration(),
            span_duration: self.span_duration(),
            gaps,
            overlaps,
        }
    }

    /// Average entry duration.
    pub fn average_duration(&self) -> Duration {
        if self.entries.is_empty() {
            return Duration::ZERO;
        }
        self.total_content_duration() / self.entries.len() as u32
    }

    /// Longest entry.
    pub fn longest_entry(&self) -> Option<&TimedEntry> {
        self.entries.iter().max_by_key(|e| e.duration)
    }

    /// Shortest entry.
    pub fn shortest_entry(&self) -> Option<&TimedEntry> {
        self.entries.iter().min_by_key(|e| e.duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    #[test]
    fn test_timed_entry_end() {
        let e = TimedEntry::new("A", secs(10), secs(5));
        assert_eq!(e.end(), secs(15));
    }

    #[test]
    fn test_empty_calculator() {
        let calc = DurationCalculator::new(25.0);
        assert_eq!(calc.entry_count(), 0);
        assert_eq!(calc.total_content_duration(), Duration::ZERO);
        assert_eq!(calc.span_duration(), Duration::ZERO);
    }

    #[test]
    fn test_add_entry() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        assert_eq!(calc.entry_count(), 1);
    }

    #[test]
    fn test_total_content_duration() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        calc.add_entry(TimedEntry::new("B", secs(10), secs(20)));
        assert_eq!(calc.total_content_duration(), secs(30));
    }

    #[test]
    fn test_span_duration() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(5), secs(10)));
        calc.add_entry(TimedEntry::new("B", secs(20), secs(10)));
        // span = 30 - 5 = 25
        assert_eq!(calc.span_duration(), secs(25));
    }

    #[test]
    fn test_gap_detection() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        // gap from 10 to 15
        calc.add_entry(TimedEntry::new("B", secs(15), secs(10)));
        let report = calc.gap_analysis();
        assert_eq!(report.gaps.len(), 1);
        assert_eq!(report.gaps[0].gap_duration, secs(5));
        assert!(report.overlaps.is_empty());
    }

    #[test]
    fn test_overlap_detection() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(15)));
        calc.add_entry(TimedEntry::new("B", secs(10), secs(10)));
        let report = calc.gap_analysis();
        assert!(report.gaps.is_empty());
        assert_eq!(report.overlaps.len(), 1);
        assert_eq!(report.overlaps[0].overlap_duration, secs(5));
    }

    #[test]
    fn test_no_gap_no_overlap() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        calc.add_entry(TimedEntry::new("B", secs(10), secs(10)));
        let report = calc.gap_analysis();
        assert!(report.gaps.is_empty());
        assert!(report.overlaps.is_empty());
    }

    #[test]
    fn test_convert_to_frames() {
        let calc = DurationCalculator::new(25.0);
        let frames = calc.convert(secs(2), DurationUnit::Frames);
        assert!((frames - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_convert_to_milliseconds() {
        let calc = DurationCalculator::new(25.0);
        let ms = calc.convert(secs(3), DurationUnit::Milliseconds);
        assert!((ms - 3000.0).abs() < 0.01);
    }

    #[test]
    fn test_frames_to_duration() {
        let calc = DurationCalculator::new(25.0);
        let dur = calc.frames_to_duration(50);
        assert_eq!(dur, secs(2));
    }

    #[test]
    fn test_average_duration() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        calc.add_entry(TimedEntry::new("B", secs(10), secs(30)));
        assert_eq!(calc.average_duration(), secs(20));
    }

    #[test]
    fn test_longest_entry() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("Short", secs(0), secs(5)));
        calc.add_entry(TimedEntry::new("Long", secs(5), secs(60)));
        let longest = calc.longest_entry().expect("should succeed in test");
        assert_eq!(longest.label, "Long");
    }

    #[test]
    fn test_shortest_entry() {
        let mut calc = DurationCalculator::new(25.0);
        calc.add_entry(TimedEntry::new("Short", secs(0), secs(3)));
        calc.add_entry(TimedEntry::new("Long", secs(3), secs(60)));
        let shortest = calc.shortest_entry().expect("should succeed in test");
        assert_eq!(shortest.label, "Short");
    }

    #[test]
    fn test_frames_to_duration_zero_rate() {
        let calc = DurationCalculator::new(0.0);
        let dur = calc.frames_to_duration(100);
        assert_eq!(dur, Duration::ZERO);
    }

    #[test]
    fn test_gap_report_totals() {
        let mut calc = DurationCalculator::new(30.0);
        calc.add_entry(TimedEntry::new("A", secs(0), secs(10)));
        calc.add_entry(TimedEntry::new("B", secs(12), secs(8)));
        let report = calc.gap_analysis();
        assert_eq!(report.total_content_duration, secs(18));
        assert_eq!(report.span_duration, secs(20));
    }
}
