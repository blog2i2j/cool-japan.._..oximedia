//! Timing and synchronization for audio description cue points.
//!
//! This module provides frame-accurate timing synchronization, cue point
//! validation, gap detection, overlap handling, and subtitle synchronization
//! for audio description tracks.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

use super::metadata::DescriptionCue;
use crate::AudioResult;
use std::collections::BTreeMap;

/// Timing precision for synchronization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimingPrecision {
    /// Frame-accurate (±1 frame).
    #[default]
    Frame,
    /// Sample-accurate (±1 sample).
    Sample,
    /// Millisecond-accurate (±1ms).
    Millisecond,
}

/// Configuration for timing synchronization.
#[derive(Clone, Debug)]
pub struct TimingConfig {
    /// Timing precision.
    pub precision: TimingPrecision,
    /// Minimum gap between cues in seconds.
    pub min_gap_seconds: f64,
    /// Maximum allowed overlap in seconds.
    pub max_overlap_seconds: f64,
    /// Enable automatic gap filling.
    pub auto_fill_gaps: bool,
    /// Enable overlap detection.
    pub detect_overlaps: bool,
    /// Snap to frame boundaries.
    pub snap_to_frames: bool,
    /// Frame rate for frame snapping.
    pub frame_rate: f64,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            precision: TimingPrecision::Frame,
            min_gap_seconds: 0.1,
            max_overlap_seconds: 0.0,
            auto_fill_gaps: false,
            detect_overlaps: true,
            snap_to_frames: true,
            frame_rate: 30.0,
        }
    }
}

impl TimingConfig {
    /// Create a new timing configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timing precision.
    #[must_use]
    pub fn with_precision(mut self, precision: TimingPrecision) -> Self {
        self.precision = precision;
        self
    }

    /// Set minimum gap between cues.
    #[must_use]
    pub fn with_min_gap(mut self, seconds: f64) -> Self {
        self.min_gap_seconds = seconds.max(0.0);
        self
    }

    /// Set maximum allowed overlap.
    #[must_use]
    pub fn with_max_overlap(mut self, seconds: f64) -> Self {
        self.max_overlap_seconds = seconds.max(0.0);
        self
    }

    /// Enable automatic gap filling.
    #[must_use]
    pub fn with_auto_fill_gaps(mut self, enabled: bool) -> Self {
        self.auto_fill_gaps = enabled;
        self
    }

    /// Enable overlap detection.
    #[must_use]
    pub fn with_overlap_detection(mut self, enabled: bool) -> Self {
        self.detect_overlaps = enabled;
        self
    }

    /// Set frame rate for frame snapping.
    #[must_use]
    pub fn with_frame_rate(mut self, frame_rate: f64) -> Self {
        self.frame_rate = frame_rate.max(1.0);
        self.snap_to_frames = true;
        self
    }

    /// Broadcast timing preset (strict timing).
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            precision: TimingPrecision::Frame,
            min_gap_seconds: 0.0,
            max_overlap_seconds: 0.0,
            auto_fill_gaps: false,
            detect_overlaps: true,
            snap_to_frames: true,
            frame_rate: 30.0,
        }
    }

    /// Flexible timing preset (lenient timing).
    #[must_use]
    pub fn flexible() -> Self {
        Self {
            precision: TimingPrecision::Millisecond,
            min_gap_seconds: 0.0,
            max_overlap_seconds: 0.5,
            auto_fill_gaps: true,
            detect_overlaps: false,
            snap_to_frames: false,
            frame_rate: 30.0,
        }
    }
}

/// Timing validation result.
#[derive(Clone, Debug)]
pub struct TimingValidation {
    /// Whether timing is valid.
    pub is_valid: bool,
    /// List of timing issues.
    pub issues: Vec<TimingIssue>,
}

impl TimingValidation {
    /// Create a valid timing validation result.
    #[must_use]
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            issues: Vec::new(),
        }
    }

    /// Create an invalid timing validation result.
    #[must_use]
    pub fn invalid(issues: Vec<TimingIssue>) -> Self {
        Self {
            is_valid: false,
            issues,
        }
    }

    /// Add a timing issue.
    pub fn add_issue(&mut self, issue: TimingIssue) {
        self.is_valid = false;
        self.issues.push(issue);
    }
}

/// Timing issue type.
#[derive(Clone, Debug)]
pub enum TimingIssue {
    /// Gap between cues is too small.
    GapTooSmall {
        /// Previous cue ID.
        prev_cue_id: String,
        /// Next cue ID.
        next_cue_id: String,
        /// Actual gap in seconds.
        gap: f64,
        /// Minimum required gap.
        min_gap: f64,
    },
    /// Cues overlap.
    Overlap {
        /// First cue ID.
        cue_id_1: String,
        /// Second cue ID.
        cue_id_2: String,
        /// Overlap duration in seconds.
        overlap: f64,
    },
    /// Cue duration is invalid.
    InvalidDuration {
        /// Cue ID.
        cue_id: String,
        /// Duration in seconds.
        duration: f64,
    },
    /// Cue timing is out of bounds.
    OutOfBounds {
        /// Cue ID.
        cue_id: String,
        /// Start time.
        start_time: f64,
        /// End time.
        end_time: f64,
    },
}

/// Timeline manager for audio description cues.
pub struct Timeline {
    /// Configuration.
    config: TimingConfig,
    /// Sample rate.
    sample_rate: f64,
    /// Cues indexed by start time.
    cues: BTreeMap<u64, DescriptionCue>,
    /// Total duration in seconds.
    total_duration: f64,
}

impl Timeline {
    /// Create a new timeline.
    #[must_use]
    pub fn new(config: TimingConfig, sample_rate: f64) -> Self {
        Self {
            config,
            sample_rate,
            cues: BTreeMap::new(),
            total_duration: 0.0,
        }
    }

    /// Set total duration.
    pub fn set_duration(&mut self, duration: f64) {
        self.total_duration = duration;
    }

    /// Add a cue to the timeline.
    pub fn add_cue(&mut self, mut cue: DescriptionCue) -> AudioResult<()> {
        cue.validate()?;

        if self.config.snap_to_frames {
            cue.start_time = self.snap_to_frame(cue.start_time);
            cue.end_time = self.snap_to_frame(cue.end_time);
        }

        let key = self.time_to_key(cue.start_time);
        self.cues.insert(key, cue);

        Ok(())
    }

    /// Remove a cue from the timeline.
    pub fn remove_cue(&mut self, cue_id: &str) -> bool {
        let key_to_remove = self
            .cues
            .iter()
            .find(|(_, cue)| cue.id == cue_id)
            .map(|(k, _)| *k);

        if let Some(key) = key_to_remove {
            self.cues.remove(&key);
            true
        } else {
            false
        }
    }

    /// Get cue at specific time.
    #[must_use]
    pub fn get_cue_at(&self, time: f64) -> Option<&DescriptionCue> {
        self.cues
            .values()
            .find(|cue| time >= cue.start_time && time < cue.end_time)
    }

    /// Get all cues in time range.
    #[must_use]
    pub fn get_cues_in_range(&self, start: f64, end: f64) -> Vec<&DescriptionCue> {
        self.cues
            .values()
            .filter(|cue| {
                (cue.start_time >= start && cue.start_time < end)
                    || (cue.end_time > start && cue.end_time <= end)
                    || (cue.start_time < start && cue.end_time > end)
            })
            .collect()
    }

    /// Get all cues sorted by start time.
    #[must_use]
    pub fn get_all_cues(&self) -> Vec<&DescriptionCue> {
        self.cues.values().collect()
    }

    /// Validate timeline timing.
    pub fn validate(&self) -> TimingValidation {
        let mut validation = TimingValidation::valid();

        let cues: Vec<_> = self.cues.values().collect();

        for cue in &cues {
            if cue.duration() <= 0.0 {
                validation.add_issue(TimingIssue::InvalidDuration {
                    cue_id: cue.id.clone(),
                    duration: cue.duration(),
                });
            }

            if self.total_duration > 0.0 && cue.end_time > self.total_duration {
                validation.add_issue(TimingIssue::OutOfBounds {
                    cue_id: cue.id.clone(),
                    start_time: cue.start_time,
                    end_time: cue.end_time,
                });
            }
        }

        if self.config.detect_overlaps || self.config.min_gap_seconds > 0.0 {
            for i in 0..cues.len().saturating_sub(1) {
                let current = cues[i];
                let next = cues[i + 1];

                if current.end_time > next.start_time {
                    let overlap = current.end_time - next.start_time;

                    if self.config.detect_overlaps && overlap > self.config.max_overlap_seconds {
                        validation.add_issue(TimingIssue::Overlap {
                            cue_id_1: current.id.clone(),
                            cue_id_2: next.id.clone(),
                            overlap,
                        });
                    }
                } else {
                    let gap = next.start_time - current.end_time;

                    if gap < self.config.min_gap_seconds {
                        validation.add_issue(TimingIssue::GapTooSmall {
                            prev_cue_id: current.id.clone(),
                            next_cue_id: next.id.clone(),
                            gap,
                            min_gap: self.config.min_gap_seconds,
                        });
                    }
                }
            }
        }

        validation
    }

    /// Detect gaps in the timeline.
    #[must_use]
    pub fn detect_gaps(&self) -> Vec<TimeGap> {
        let mut gaps = Vec::new();
        let cues: Vec<_> = self.cues.values().collect();

        if cues.is_empty() {
            if self.total_duration > 0.0 {
                gaps.push(TimeGap {
                    start: 0.0,
                    end: self.total_duration,
                    duration: self.total_duration,
                });
            }
            return gaps;
        }

        if cues[0].start_time > 0.0 {
            gaps.push(TimeGap {
                start: 0.0,
                end: cues[0].start_time,
                duration: cues[0].start_time,
            });
        }

        for i in 0..cues.len().saturating_sub(1) {
            let current = cues[i];
            let next = cues[i + 1];

            if next.start_time > current.end_time {
                gaps.push(TimeGap {
                    start: current.end_time,
                    end: next.start_time,
                    duration: next.start_time - current.end_time,
                });
            }
        }

        if self.total_duration > 0.0 {
            if let Some(last) = cues.last() {
                if last.end_time < self.total_duration {
                    gaps.push(TimeGap {
                        start: last.end_time,
                        end: self.total_duration,
                        duration: self.total_duration - last.end_time,
                    });
                }
            }
        }

        gaps
    }

    /// Convert time to sample index.
    #[must_use]
    pub fn time_to_sample(&self, time: f64) -> usize {
        (time * self.sample_rate) as usize
    }

    /// Convert sample index to time.
    #[must_use]
    pub fn sample_to_time(&self, sample: usize) -> f64 {
        sample as f64 / self.sample_rate
    }

    /// Snap time to frame boundary.
    fn snap_to_frame(&self, time: f64) -> f64 {
        if !self.config.snap_to_frames {
            return time;
        }

        let frame_duration = 1.0 / self.config.frame_rate;
        let frame_number = (time / frame_duration).round();
        frame_number * frame_duration
    }

    /// Convert time to key for BTreeMap.
    fn time_to_key(&self, time: f64) -> u64 {
        match self.config.precision {
            TimingPrecision::Sample => (time * self.sample_rate) as u64,
            TimingPrecision::Frame => ((time * self.config.frame_rate).round() * 1000.0) as u64,
            TimingPrecision::Millisecond => (time * 1000.0) as u64,
        }
    }

    /// Clear all cues.
    pub fn clear(&mut self) {
        self.cues.clear();
    }

    /// Get number of cues.
    #[must_use]
    pub fn cue_count(&self) -> usize {
        self.cues.len()
    }
}

/// Time gap in the timeline.
#[derive(Clone, Debug)]
pub struct TimeGap {
    /// Gap start time in seconds.
    pub start: f64,
    /// Gap end time in seconds.
    pub end: f64,
    /// Gap duration in seconds.
    pub duration: f64,
}

/// Subtitle synchronization manager.
pub struct SubtitleSync {
    /// Timeline.
    timeline: Timeline,
    /// Subtitle offset in seconds.
    subtitle_offset: f64,
}

impl SubtitleSync {
    /// Create a new subtitle sync manager.
    #[must_use]
    pub fn new(config: TimingConfig, sample_rate: f64) -> Self {
        Self {
            timeline: Timeline::new(config, sample_rate),
            subtitle_offset: 0.0,
        }
    }

    /// Set subtitle offset.
    pub fn set_offset(&mut self, offset: f64) {
        self.subtitle_offset = offset;
    }

    /// Get subtitle offset.
    #[must_use]
    pub fn offset(&self) -> f64 {
        self.subtitle_offset
    }

    /// Synchronize cue with subtitle timing.
    pub fn sync_cue(&mut self, mut cue: DescriptionCue) -> AudioResult<()> {
        cue.start_time += self.subtitle_offset;
        cue.end_time += self.subtitle_offset;
        self.timeline.add_cue(cue)
    }

    /// Get synchronized cue at time.
    #[must_use]
    pub fn get_cue_at(&self, time: f64) -> Option<&DescriptionCue> {
        self.timeline.get_cue_at(time + self.subtitle_offset)
    }

    /// Get timeline reference.
    #[must_use]
    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    /// Get mutable timeline reference.
    pub fn timeline_mut(&mut self) -> &mut Timeline {
        &mut self.timeline
    }
}

/// Cue point scheduler for real-time playback.
pub struct CueScheduler {
    /// Timeline.
    timeline: Timeline,
    /// Current playback position in seconds.
    position: f64,
    /// Active cue.
    active_cue: Option<DescriptionCue>,
    /// Lookahead time in seconds.
    lookahead: f64,
}

impl CueScheduler {
    /// Create a new cue scheduler.
    #[must_use]
    pub fn new(timeline: Timeline, lookahead: f64) -> Self {
        Self {
            timeline,
            position: 0.0,
            active_cue: None,
            lookahead,
        }
    }

    /// Update playback position.
    pub fn update_position(&mut self, position: f64) {
        self.position = position;

        if let Some(ref cue) = self.active_cue {
            if position >= cue.end_time {
                self.active_cue = None;
            }
        }

        if self.active_cue.is_none() {
            self.active_cue = self.timeline.get_cue_at(position).cloned();
        }
    }

    /// Get active cue.
    #[must_use]
    pub fn active_cue(&self) -> Option<&DescriptionCue> {
        self.active_cue.as_ref()
    }

    /// Get upcoming cues within lookahead window.
    #[must_use]
    pub fn upcoming_cues(&self) -> Vec<&DescriptionCue> {
        self.timeline
            .get_cues_in_range(self.position, self.position + self.lookahead)
    }

    /// Check if cue is active at current position.
    #[must_use]
    pub fn is_cue_active(&self) -> bool {
        self.active_cue.is_some()
    }

    /// Seek to position.
    pub fn seek(&mut self, position: f64) {
        self.position = position;
        self.active_cue = self.timeline.get_cue_at(position).cloned();
    }

    /// Reset scheduler.
    pub fn reset(&mut self) {
        self.position = 0.0;
        self.active_cue = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timing_config() {
        let config = TimingConfig::new().with_min_gap(0.5);
        assert!((config.min_gap_seconds - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timeline_add_cue() {
        let config = TimingConfig::default();
        let mut timeline = Timeline::new(config, 48000.0);

        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test");
        timeline.add_cue(cue).expect("should succeed");

        assert_eq!(timeline.cue_count(), 1);
    }

    #[test]
    fn test_timeline_get_cue_at() {
        let config = TimingConfig::default();
        let mut timeline = Timeline::new(config, 48000.0);

        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test");
        timeline.add_cue(cue).expect("should succeed");

        assert!(timeline.get_cue_at(1.5).is_some());
        assert!(timeline.get_cue_at(0.5).is_none());
        assert!(timeline.get_cue_at(2.5).is_none());
    }

    #[test]
    fn test_timeline_detect_gaps() {
        let config = TimingConfig::default();
        let mut timeline = Timeline::new(config, 48000.0);
        timeline.set_duration(10.0);

        let cue1 = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test 1");
        let cue2 = DescriptionCue::new("cue2", 3.0, 4.0).with_text("Test 2");

        timeline.add_cue(cue1).expect("should succeed");
        timeline.add_cue(cue2).expect("should succeed");

        let gaps = timeline.detect_gaps();
        assert_eq!(gaps.len(), 3);
    }

    #[test]
    fn test_timeline_validation() {
        let config = TimingConfig::default();
        let mut timeline = Timeline::new(config, 48000.0);

        let cue1 = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test 1");
        let cue2 = DescriptionCue::new("cue2", 1.5, 2.5).with_text("Test 2");

        timeline.add_cue(cue1).expect("should succeed");
        timeline.add_cue(cue2).expect("should succeed");

        let validation = timeline.validate();
        assert!(!validation.is_valid);
        assert!(!validation.issues.is_empty());
    }

    #[test]
    fn test_cue_scheduler() {
        let config = TimingConfig::default();
        let mut timeline = Timeline::new(config, 48000.0);

        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test");
        timeline.add_cue(cue).expect("should succeed");

        let mut scheduler = CueScheduler::new(timeline, 1.0);

        scheduler.update_position(1.5);
        assert!(scheduler.is_cue_active());

        scheduler.update_position(2.5);
        assert!(!scheduler.is_cue_active());
    }
}
