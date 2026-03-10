#![allow(dead_code)]
//! Timecode sequence management for contiguous and non-contiguous frame runs.
//!
//! Provides utilities for building, iterating, and validating ordered sequences
//! of timecodes that represent edit lists, playlists, or continuous recordings.

use crate::{FrameRate, FrameRateInfo, Timecode, TimecodeError};

/// A contiguous run of timecodes sharing the same frame rate.
#[derive(Debug, Clone, PartialEq)]
pub struct TimecodeRun {
    /// Starting timecode of the run.
    pub start: Timecode,
    /// Number of frames in this run (inclusive of the start frame).
    pub frame_count: u64,
}

impl TimecodeRun {
    /// Create a new timecode run beginning at `start` spanning `frame_count` frames.
    ///
    /// # Errors
    ///
    /// Returns [`TimecodeError::InvalidConfiguration`] when `frame_count` is zero.
    pub fn new(start: Timecode, frame_count: u64) -> Result<Self, TimecodeError> {
        if frame_count == 0 {
            return Err(TimecodeError::InvalidConfiguration);
        }
        Ok(Self { start, frame_count })
    }

    /// Duration of the run in seconds (approximate for non-integer frame rates).
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(&self) -> f64 {
        let fps = self.start.frame_rate.fps as f64;
        self.frame_count as f64 / fps
    }

    /// Returns `true` if `tc` falls within this run.
    pub fn contains(&self, tc: &Timecode) -> bool {
        if tc.frame_rate != self.start.frame_rate {
            return false;
        }
        let start_f = self.start.to_frames();
        let tc_f = tc.to_frames();
        tc_f >= start_f && tc_f < start_f + self.frame_count
    }

    /// Compute the end timecode (last frame in the run).
    ///
    /// # Errors
    ///
    /// Returns an error if the resulting timecode exceeds 24-hour bounds.
    pub fn end_timecode(&self, rate: FrameRate) -> Result<Timecode, TimecodeError> {
        let end_frame = self.start.to_frames() + self.frame_count - 1;
        Timecode::from_frames(end_frame, rate)
    }
}

/// An ordered sequence of [`TimecodeRun`] entries representing a playlist or
/// edit decision list.
#[derive(Debug, Clone)]
pub struct TimecodeSequence {
    /// Runs in this sequence.
    runs: Vec<TimecodeRun>,
    /// Cached total frame count.
    total_frames: u64,
}

impl TimecodeSequence {
    /// Create an empty sequence.
    pub fn new() -> Self {
        Self {
            runs: Vec::new(),
            total_frames: 0,
        }
    }

    /// Append a run to the sequence.
    pub fn push(&mut self, run: TimecodeRun) {
        self.total_frames += run.frame_count;
        self.runs.push(run);
    }

    /// Total number of frames across all runs.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Number of runs in the sequence.
    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    /// Get a run by index.
    pub fn get(&self, index: usize) -> Option<&TimecodeRun> {
        self.runs.get(index)
    }

    /// Check whether two adjacent runs are contiguous (end of run N + 1 ==
    /// start of run N+1).
    pub fn is_contiguous(&self, a_idx: usize, b_idx: usize) -> bool {
        let (Some(a), Some(b)) = (self.runs.get(a_idx), self.runs.get(b_idx)) else {
            return false;
        };
        if a.start.frame_rate != b.start.frame_rate {
            return false;
        }
        let a_end = a.start.to_frames() + a.frame_count;
        let b_start = b.start.to_frames();
        a_end == b_start
    }

    /// Return the total duration (seconds) of the whole sequence.
    #[allow(clippy::cast_precision_loss)]
    pub fn total_duration_secs(&self) -> f64 {
        self.runs.iter().map(TimecodeRun::duration_secs).sum()
    }

    /// Find the run containing a given absolute frame offset into the sequence.
    /// Returns `(run_index, offset_within_run)`.
    pub fn find_run_at_offset(&self, offset: u64) -> Option<(usize, u64)> {
        let mut cumulative = 0u64;
        for (i, run) in self.runs.iter().enumerate() {
            if offset < cumulative + run.frame_count {
                return Some((i, offset - cumulative));
            }
            cumulative += run.frame_count;
        }
        None
    }

    /// Merge adjacent contiguous runs into a single run where possible.
    pub fn compact(&mut self) {
        if self.runs.len() < 2 {
            return;
        }
        let mut merged: Vec<TimecodeRun> = Vec::with_capacity(self.runs.len());
        merged.push(self.runs[0].clone());
        for run in self.runs.iter().skip(1) {
            let last = merged
                .last_mut()
                .expect("merged is non-empty: initial element was pushed before this loop");
            if last.start.frame_rate == run.start.frame_rate {
                let last_end = last.start.to_frames() + last.frame_count;
                if last_end == run.start.to_frames() {
                    last.frame_count += run.frame_count;
                    continue;
                }
            }
            merged.push(run.clone());
        }
        self.runs = merged;
    }

    /// Return an iterator over all runs.
    pub fn iter(&self) -> std::slice::Iter<'_, TimecodeRun> {
        self.runs.iter()
    }

    /// Detect gaps between adjacent runs and return the gap durations in frames.
    pub fn detect_gaps(&self) -> Vec<(usize, i64)> {
        let mut gaps = Vec::new();
        for i in 0..self.runs.len().saturating_sub(1) {
            let a = &self.runs[i];
            let b = &self.runs[i + 1];
            if a.start.frame_rate == b.start.frame_rate {
                let a_end = a.start.to_frames() + a.frame_count;
                let b_start = b.start.to_frames();
                let gap = b_start as i64 - a_end as i64;
                if gap != 0 {
                    gaps.push((i, gap));
                }
            }
        }
        gaps
    }
}

impl Default for TimecodeSequence {
    fn default() -> Self {
        Self::new()
    }
}

/// Build a [`TimecodeSequence`] from an iterator of `(start_tc, frame_count)` pairs.
pub fn build_sequence(
    items: &[(Timecode, u64)],
) -> Result<TimecodeSequence, TimecodeError> {
    let mut seq = TimecodeSequence::new();
    for (tc, count) in items {
        let run = TimecodeRun::new(*tc, *count)?;
        seq.push(run);
    }
    Ok(seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tc(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_run_creation() {
        let run = TimecodeRun::new(tc(1, 0, 0, 0), 100).expect("valid timecode run");
        assert_eq!(run.frame_count, 100);
    }

    #[test]
    fn test_run_zero_frames_error() {
        let result = TimecodeRun::new(tc(0, 0, 0, 0), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_duration_secs() {
        let run = TimecodeRun::new(tc(0, 0, 0, 0), 50).expect("valid timecode run");
        let dur = run.duration_secs();
        assert!((dur - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_contains() {
        let run = TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run");
        assert!(run.contains(&tc(0, 0, 0, 10)));
        assert!(run.contains(&tc(0, 0, 0, 24)));
        assert!(!run.contains(&tc(0, 0, 1, 0)));
    }

    #[test]
    fn test_sequence_push_and_total() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 25).expect("valid timecode run"));
        assert_eq!(seq.total_frames(), 50);
        assert_eq!(seq.run_count(), 2);
    }

    #[test]
    fn test_sequence_is_contiguous() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 25).expect("valid timecode run"));
        assert!(seq.is_contiguous(0, 1));
    }

    #[test]
    fn test_sequence_not_contiguous() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 10).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 10).expect("valid timecode run"));
        assert!(!seq.is_contiguous(0, 1));
    }

    #[test]
    fn test_find_run_at_offset() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 25).expect("valid timecode run"));
        assert_eq!(seq.find_run_at_offset(0), Some((0, 0)));
        assert_eq!(seq.find_run_at_offset(24), Some((0, 24)));
        assert_eq!(seq.find_run_at_offset(25), Some((1, 0)));
        assert_eq!(seq.find_run_at_offset(50), None);
    }

    #[test]
    fn test_compact_merges_contiguous() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 25).expect("valid timecode run"));
        assert_eq!(seq.run_count(), 2);
        seq.compact();
        assert_eq!(seq.run_count(), 1);
        assert_eq!(seq.total_frames(), 50);
    }

    #[test]
    fn test_compact_preserves_gaps() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 10).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 2, 0), 10).expect("valid timecode run"));
        seq.compact();
        assert_eq!(seq.run_count(), 2);
    }

    #[test]
    fn test_detect_gaps() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 10).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 10).expect("valid timecode run"));
        let gaps = seq.detect_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].0, 0);
        assert_eq!(gaps[0].1, 15); // gap of 15 frames
    }

    #[test]
    fn test_build_sequence() {
        let items = vec![(tc(0, 0, 0, 0), 25u64), (tc(0, 0, 1, 0), 50)];
        let seq = build_sequence(&items).expect("build sequence should succeed");
        assert_eq!(seq.run_count(), 2);
        assert_eq!(seq.total_frames(), 75);
    }

    #[test]
    fn test_total_duration_secs() {
        let mut seq = TimecodeSequence::new();
        seq.push(TimecodeRun::new(tc(0, 0, 0, 0), 25).expect("valid timecode run"));
        seq.push(TimecodeRun::new(tc(0, 0, 1, 0), 50).expect("valid timecode run"));
        let dur = seq.total_duration_secs();
        assert!((dur - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_end_timecode() {
        let run = TimecodeRun::new(tc(0, 0, 0, 0), 26).expect("valid timecode run");
        let end = run.end_timecode(FrameRate::Fps25).expect("end timecode should succeed");
        assert_eq!(end.hours, 0);
        assert_eq!(end.minutes, 0);
        assert_eq!(end.seconds, 1);
        assert_eq!(end.frames, 0);
    }
}
