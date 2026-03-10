//! Automatic editing operations.
//!
//! Provides algorithms for automatic clip sequencing, audio ducking,
//! gap removal, and beat-driven editing.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

/// Strategy for auto-sequencing clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStrategy {
    /// Place clips end-to-end with no gap.
    BackToBack,
    /// Insert a fixed gap (in timebase units) between clips.
    FixedGap,
    /// Overlap clips by a fixed amount for cross-dissolve.
    Overlap,
}

/// Configuration for automatic sequencing.
#[derive(Debug, Clone)]
pub struct AutoSequenceConfig {
    /// Sequencing strategy.
    pub strategy: SequenceStrategy,
    /// Gap or overlap amount in timebase units.
    pub spacing: u64,
    /// Whether to sort clips by source timecode before sequencing.
    pub sort_by_timecode: bool,
    /// Target track index to place clips on.
    pub target_track: u32,
}

impl AutoSequenceConfig {
    /// Create a default back-to-back configuration.
    #[must_use]
    pub fn back_to_back(target_track: u32) -> Self {
        Self {
            strategy: SequenceStrategy::BackToBack,
            spacing: 0,
            sort_by_timecode: false,
            target_track,
        }
    }

    /// Create a configuration with fixed gaps.
    #[must_use]
    pub fn with_gap(target_track: u32, gap: u64) -> Self {
        Self {
            strategy: SequenceStrategy::FixedGap,
            spacing: gap,
            sort_by_timecode: false,
            target_track,
        }
    }

    /// Create a configuration with overlap for dissolves.
    #[must_use]
    pub fn with_overlap(target_track: u32, overlap: u64) -> Self {
        Self {
            strategy: SequenceStrategy::Overlap,
            spacing: overlap,
            sort_by_timecode: false,
            target_track,
        }
    }
}

/// A clip reference for auto-editing operations.
#[derive(Debug, Clone)]
pub struct AutoClip {
    /// Clip identifier.
    pub id: u64,
    /// Source in-point (timebase units).
    pub source_in: u64,
    /// Source out-point (timebase units).
    pub source_out: u64,
    /// Source timecode for sorting (if available).
    pub source_timecode: Option<u64>,
    /// Audio level in dB (for ducking).
    pub audio_level_db: f64,
}

impl AutoClip {
    /// Create a new auto-clip reference.
    #[must_use]
    pub fn new(id: u64, source_in: u64, source_out: u64) -> Self {
        Self {
            id,
            source_in,
            source_out,
            source_timecode: None,
            audio_level_db: 0.0,
        }
    }

    /// Duration in timebase units.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.source_out.saturating_sub(self.source_in)
    }
}

/// Placement result for a single clip after auto-sequencing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipPlacement {
    /// Clip identifier.
    pub clip_id: u64,
    /// Timeline position (timebase units).
    pub timeline_pos: u64,
    /// Duration on timeline (timebase units).
    pub duration: u64,
    /// Target track index.
    pub track: u32,
}

/// Compute clip placements using the given auto-sequence configuration.
#[must_use]
pub fn auto_sequence(clips: &[AutoClip], config: &AutoSequenceConfig) -> Vec<ClipPlacement> {
    if clips.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<&AutoClip> = clips.iter().collect();
    if config.sort_by_timecode {
        sorted.sort_by_key(|c| c.source_timecode.unwrap_or(0));
    }

    let mut result = Vec::with_capacity(sorted.len());
    let mut cursor: u64 = 0;

    for clip in &sorted {
        let dur = clip.duration();
        result.push(ClipPlacement {
            clip_id: clip.id,
            timeline_pos: cursor,
            duration: dur,
            track: config.target_track,
        });
        match config.strategy {
            SequenceStrategy::BackToBack => cursor += dur,
            SequenceStrategy::FixedGap => cursor += dur + config.spacing,
            SequenceStrategy::Overlap => cursor += dur.saturating_sub(config.spacing),
        }
    }
    result
}

/// Audio ducking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuckMode {
    /// Reduce music volume when dialogue is present.
    DialogueOverMusic,
    /// Reduce all other audio when narration is present.
    NarrationPriority,
}

/// Configuration for auto audio-ducking.
#[derive(Debug, Clone)]
pub struct AutoDuckConfig {
    /// Ducking mode.
    pub mode: DuckMode,
    /// Threshold in dB below which the side-chain triggers.
    pub threshold_db: f64,
    /// Amount of gain reduction in dB when ducking is active.
    pub reduction_db: f64,
    /// Attack time in milliseconds.
    pub attack_ms: f64,
    /// Release time in milliseconds.
    pub release_ms: f64,
}

impl Default for AutoDuckConfig {
    fn default() -> Self {
        Self {
            mode: DuckMode::DialogueOverMusic,
            threshold_db: -20.0,
            reduction_db: -12.0,
            attack_ms: 50.0,
            release_ms: 200.0,
        }
    }
}

impl AutoDuckConfig {
    /// Create a new auto-duck configuration.
    #[must_use]
    pub fn new(mode: DuckMode) -> Self {
        Self {
            mode,
            ..Default::default()
        }
    }

    /// Compute the ducked level for a given input level.
    #[must_use]
    pub fn ducked_level(&self, input_db: f64) -> f64 {
        if input_db > self.threshold_db {
            input_db + self.reduction_db
        } else {
            input_db
        }
    }
}

/// A detected gap in the timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineGap {
    /// Start position of the gap (timebase units).
    pub start: u64,
    /// End position of the gap (timebase units).
    pub end: u64,
    /// Track index where the gap exists.
    pub track: u32,
}

impl TimelineGap {
    /// Duration of the gap.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// Detect gaps in a set of clip placements on a single track.
#[must_use]
pub fn detect_gaps(placements: &[ClipPlacement], track: u32) -> Vec<TimelineGap> {
    let mut on_track: Vec<&ClipPlacement> =
        placements.iter().filter(|p| p.track == track).collect();
    on_track.sort_by_key(|p| p.timeline_pos);

    let mut gaps = Vec::new();
    for i in 1..on_track.len() {
        let prev_end = on_track[i - 1].timeline_pos + on_track[i - 1].duration;
        let curr_start = on_track[i].timeline_pos;
        if curr_start > prev_end {
            gaps.push(TimelineGap {
                start: prev_end,
                end: curr_start,
                track,
            });
        }
    }
    gaps
}

/// Compute ripple-close offsets that would remove all gaps on a track.
///
/// Returns a map of `clip_id` to new timeline position.
#[must_use]
pub fn ripple_close_gaps(placements: &[ClipPlacement], track: u32) -> HashMap<u64, u64> {
    let mut on_track: Vec<&ClipPlacement> =
        placements.iter().filter(|p| p.track == track).collect();
    on_track.sort_by_key(|p| p.timeline_pos);

    let mut result = HashMap::new();
    let mut cursor: u64 = on_track.first().map_or(0, |p| p.timeline_pos);
    for p in &on_track {
        result.insert(p.clip_id, cursor);
        cursor += p.duration;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_clip_duration() {
        let c = AutoClip::new(1, 100, 500);
        assert_eq!(c.duration(), 400);
    }

    #[test]
    fn test_auto_clip_zero_duration() {
        let c = AutoClip::new(1, 500, 500);
        assert_eq!(c.duration(), 0);
    }

    #[test]
    fn test_sequence_back_to_back() {
        let clips = vec![AutoClip::new(1, 0, 100), AutoClip::new(2, 0, 200)];
        let cfg = AutoSequenceConfig::back_to_back(0);
        let result = auto_sequence(&clips, &cfg);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timeline_pos, 0);
        assert_eq!(result[0].duration, 100);
        assert_eq!(result[1].timeline_pos, 100);
        assert_eq!(result[1].duration, 200);
    }

    #[test]
    fn test_sequence_fixed_gap() {
        let clips = vec![AutoClip::new(1, 0, 100), AutoClip::new(2, 0, 100)];
        let cfg = AutoSequenceConfig::with_gap(0, 50);
        let result = auto_sequence(&clips, &cfg);
        assert_eq!(result[0].timeline_pos, 0);
        assert_eq!(result[1].timeline_pos, 150); // 100 + 50 gap
    }

    #[test]
    fn test_sequence_overlap() {
        let clips = vec![AutoClip::new(1, 0, 100), AutoClip::new(2, 0, 100)];
        let cfg = AutoSequenceConfig::with_overlap(0, 20);
        let result = auto_sequence(&clips, &cfg);
        assert_eq!(result[0].timeline_pos, 0);
        assert_eq!(result[1].timeline_pos, 80); // 100 - 20 overlap
    }

    #[test]
    fn test_sequence_empty() {
        let cfg = AutoSequenceConfig::back_to_back(0);
        let result = auto_sequence(&[], &cfg);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sequence_sort_by_timecode() {
        let mut c1 = AutoClip::new(1, 0, 100);
        c1.source_timecode = Some(200);
        let mut c2 = AutoClip::new(2, 0, 100);
        c2.source_timecode = Some(100);
        let mut cfg = AutoSequenceConfig::back_to_back(0);
        cfg.sort_by_timecode = true;
        let result = auto_sequence(&[c1, c2], &cfg);
        assert_eq!(result[0].clip_id, 2); // Earlier timecode first
        assert_eq!(result[1].clip_id, 1);
    }

    #[test]
    fn test_duck_config_default() {
        let cfg = AutoDuckConfig::default();
        assert_eq!(cfg.mode, DuckMode::DialogueOverMusic);
        assert!(cfg.threshold_db < 0.0);
    }

    #[test]
    fn test_ducked_level_above_threshold() {
        let cfg = AutoDuckConfig {
            threshold_db: -20.0,
            reduction_db: -12.0,
            ..Default::default()
        };
        let ducked = cfg.ducked_level(-10.0);
        assert!((ducked - (-22.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ducked_level_below_threshold() {
        let cfg = AutoDuckConfig::default();
        let level = -30.0;
        assert!((cfg.ducked_level(level) - level).abs() < f64::EPSILON);
    }

    #[test]
    fn test_detect_gaps() {
        let placements = vec![
            ClipPlacement {
                clip_id: 1,
                timeline_pos: 0,
                duration: 100,
                track: 0,
            },
            ClipPlacement {
                clip_id: 2,
                timeline_pos: 150,
                duration: 100,
                track: 0,
            },
            ClipPlacement {
                clip_id: 3,
                timeline_pos: 250,
                duration: 50,
                track: 0,
            },
        ];
        let gaps = detect_gaps(&placements, 0);
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].start, 100);
        assert_eq!(gaps[0].end, 150);
        assert_eq!(gaps[0].duration(), 50);
    }

    #[test]
    fn test_detect_no_gaps() {
        let placements = vec![
            ClipPlacement {
                clip_id: 1,
                timeline_pos: 0,
                duration: 100,
                track: 0,
            },
            ClipPlacement {
                clip_id: 2,
                timeline_pos: 100,
                duration: 100,
                track: 0,
            },
        ];
        let gaps = detect_gaps(&placements, 0);
        assert!(gaps.is_empty());
    }

    #[test]
    fn test_ripple_close_gaps() {
        let placements = vec![
            ClipPlacement {
                clip_id: 1,
                timeline_pos: 0,
                duration: 100,
                track: 0,
            },
            ClipPlacement {
                clip_id: 2,
                timeline_pos: 200,
                duration: 50,
                track: 0,
            },
            ClipPlacement {
                clip_id: 3,
                timeline_pos: 400,
                duration: 80,
                track: 0,
            },
        ];
        let new_pos = ripple_close_gaps(&placements, 0);
        assert_eq!(*new_pos.get(&1).expect("get should succeed"), 0);
        assert_eq!(*new_pos.get(&2).expect("get should succeed"), 100);
        assert_eq!(*new_pos.get(&3).expect("get should succeed"), 150);
    }
}
