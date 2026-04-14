//! Audio description track metadata — AD track marking, timing synchronization, and priority.
//!
//! This module provides structured metadata for audio description (AD) tracks, including:
//! - Track identification and language tagging
//! - Timing synchronization windows relative to video timecode
//! - Priority and fallback strategies when multiple AD tracks are available
//! - Gap detection to identify video segments lacking coverage

use std::collections::BTreeMap;
use std::fmt;

use crate::{AccessError, AccessResult};

// ─── Types ────────────────────────────────────────────────────────────────────

/// Style of audio description delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioDescriptionStyle {
    /// Standard AD: narration placed in natural pauses in the dialogue.
    Standard,
    /// Extended AD: programme paused to allow longer descriptions.
    Extended,
    /// Descriptive video service format (North American broadcast standard).
    DvsBroadcast,
    /// Integrated AD where narration is woven into the original dialogue.
    Integrated,
}

impl AudioDescriptionStyle {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Extended => "Extended",
            Self::DvsBroadcast => "DVS Broadcast",
            Self::Integrated => "Integrated",
        }
    }
}

impl fmt::Display for AudioDescriptionStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Priority tier for selecting between multiple AD tracks.
///
/// Lower ordinal = higher priority; ties broken by insertion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AdPriority {
    /// Primary track — highest priority.
    Primary = 0,
    /// Secondary track (e.g. alternate voice).
    Secondary = 1,
    /// Supplemental track (e.g. educational commentary).
    Supplemental = 2,
    /// Fallback track — lowest priority.
    Fallback = 3,
}

impl fmt::Display for AdPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Primary => "Primary",
            Self::Secondary => "Secondary",
            Self::Supplemental => "Supplemental",
            Self::Fallback => "Fallback",
        };
        f.write_str(s)
    }
}

/// A half-open time window `[start_ms, end_ms)` in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TimeWindow {
    /// Start time in milliseconds (inclusive).
    pub start_ms: u64,
    /// End time in milliseconds (exclusive).
    pub end_ms: u64,
}

impl TimeWindow {
    /// Create a new time window.
    ///
    /// # Errors
    /// Returns [`AccessError::InvalidTiming`] when `end_ms <= start_ms`.
    pub fn new(start_ms: u64, end_ms: u64) -> AccessResult<Self> {
        if end_ms <= start_ms {
            return Err(AccessError::InvalidTiming(format!(
                "end_ms ({end_ms}) must be greater than start_ms ({start_ms})"
            )));
        }
        Ok(Self { start_ms, end_ms })
    }

    /// Duration of the window in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }

    /// Whether this window overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start_ms < other.end_ms && other.start_ms < self.end_ms
    }

    /// Whether `ms` falls within this window.
    #[must_use]
    pub fn contains(&self, ms: u64) -> bool {
        ms >= self.start_ms && ms < self.end_ms
    }
}

impl fmt::Display for TimeWindow {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ms, {} ms)", self.start_ms, self.end_ms)
    }
}

/// A single audio-description segment: one narration cue with timing and content.
#[derive(Debug, Clone)]
pub struct AdSegment {
    /// Unique identifier within the track.
    pub id: String,
    /// Timing window during which this cue is delivered.
    pub window: TimeWindow,
    /// Narration text for this cue.
    pub text: String,
    /// Whether this cue extends the programme (pauses video).
    pub extended: bool,
    /// Confidence score in `[0.0, 1.0]` for automatic placement.
    pub placement_confidence: f32,
}

impl AdSegment {
    /// Create a new segment.
    ///
    /// # Errors
    /// Propagates [`AccessError::InvalidTiming`] from [`TimeWindow::new`].
    pub fn new(
        id: impl Into<String>,
        start_ms: u64,
        end_ms: u64,
        text: impl Into<String>,
    ) -> AccessResult<Self> {
        let window = TimeWindow::new(start_ms, end_ms)?;
        Ok(Self {
            id: id.into(),
            window,
            text: text.into(),
            extended: false,
            placement_confidence: 1.0,
        })
    }

    /// Mark this segment as an extended description (video is paused).
    #[must_use]
    pub fn with_extended(mut self, extended: bool) -> Self {
        self.extended = extended;
        self
    }

    /// Set the placement confidence score.
    ///
    /// # Errors
    /// Returns [`AccessError::Other`] when `score` is outside `[0.0, 1.0]`.
    pub fn with_placement_confidence(mut self, score: f32) -> AccessResult<Self> {
        if !(0.0..=1.0).contains(&score) {
            return Err(AccessError::Other(format!(
                "placement_confidence must be in [0.0, 1.0], got {score}"
            )));
        }
        self.placement_confidence = score;
        Ok(self)
    }

    /// Word count of the narration text.
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Estimated speaking rate in words-per-minute for this segment.
    ///
    /// Returns `None` when the window duration is zero (shouldn't happen after validation).
    #[must_use]
    pub fn estimated_wpm(&self) -> Option<f64> {
        let dur_min = self.window.duration_ms() as f64 / 60_000.0;
        if dur_min == 0.0 {
            return None;
        }
        Some(self.word_count() as f64 / dur_min)
    }
}

/// Metadata and segment list for one audio-description track.
#[derive(Debug, Clone)]
pub struct AudioDescriptionTrack {
    /// Track identifier (e.g. `"ad-en-001"`).
    pub track_id: String,
    /// BCP-47 language tag (e.g. `"en-GB"`).
    pub language: String,
    /// Delivery style.
    pub style: AudioDescriptionStyle,
    /// Selection priority.
    pub priority: AdPriority,
    /// Human-readable label shown in player UIs.
    pub display_label: Option<String>,
    /// Narration segments ordered by start time.
    segments: BTreeMap<u64, AdSegment>,
    /// Total video duration this track is designed for, in milliseconds.
    pub video_duration_ms: Option<u64>,
}

impl AudioDescriptionTrack {
    /// Create a new, empty AD track.
    #[must_use]
    pub fn new(
        track_id: impl Into<String>,
        language: impl Into<String>,
        style: AudioDescriptionStyle,
        priority: AdPriority,
    ) -> Self {
        Self {
            track_id: track_id.into(),
            language: language.into(),
            style,
            priority,
            display_label: None,
            segments: BTreeMap::new(),
            video_duration_ms: None,
        }
    }

    /// Set a human-readable display label.
    #[must_use]
    pub fn with_display_label(mut self, label: impl Into<String>) -> Self {
        self.display_label = Some(label.into());
        self
    }

    /// Set the total video duration (used for gap analysis).
    #[must_use]
    pub fn with_video_duration_ms(mut self, duration_ms: u64) -> Self {
        self.video_duration_ms = Some(duration_ms);
        self
    }

    /// Add a segment.
    ///
    /// # Errors
    /// Returns [`AccessError::SyncError`] if the new segment overlaps an existing one.
    pub fn add_segment(&mut self, segment: AdSegment) -> AccessResult<()> {
        for existing in self.segments.values() {
            if existing.window.overlaps(&segment.window) {
                return Err(AccessError::SyncError(format!(
                    "Segment '{}' overlaps with existing segment '{}' in track '{}'",
                    segment.id, existing.id, self.track_id
                )));
            }
        }
        self.segments.insert(segment.window.start_ms, segment);
        Ok(())
    }

    /// Remove a segment by id, returning it if found.
    pub fn remove_segment(&mut self, segment_id: &str) -> Option<AdSegment> {
        let key = self
            .segments
            .iter()
            .find(|(_, s)| s.id == segment_id)
            .map(|(k, _)| *k);
        key.and_then(|k| self.segments.remove(&k))
    }

    /// Return segments ordered by start time.
    #[must_use]
    pub fn segments(&self) -> Vec<&AdSegment> {
        self.segments.values().collect()
    }

    /// Total narration coverage in milliseconds.
    #[must_use]
    pub fn coverage_ms(&self) -> u64 {
        self.segments.values().map(|s| s.window.duration_ms()).sum()
    }

    /// Coverage fraction `[0.0, 1.0]` relative to `video_duration_ms`.
    ///
    /// Returns `None` when `video_duration_ms` is not set or is zero.
    #[must_use]
    pub fn coverage_fraction(&self) -> Option<f64> {
        let total = self.video_duration_ms?;
        if total == 0 {
            return None;
        }
        Some(self.coverage_ms() as f64 / total as f64)
    }

    /// Find gaps (uncovered intervals) larger than `min_gap_ms` milliseconds.
    ///
    /// Gaps are computed from `0` to `video_duration_ms` (if set), or to the
    /// last segment end otherwise.
    #[must_use]
    pub fn gaps(&self, min_gap_ms: u64) -> Vec<TimeWindow> {
        if self.segments.is_empty() {
            if let Some(dur) = self.video_duration_ms {
                if dur > min_gap_ms {
                    return vec![TimeWindow {
                        start_ms: 0,
                        end_ms: dur,
                    }];
                }
            }
            return vec![];
        }

        let mut gaps = Vec::new();
        let mut cursor: u64 = 0;
        let end_of_program = self
            .video_duration_ms
            .unwrap_or_else(|| self.segments.values().last().map(|s| s.window.end_ms).unwrap_or(0));

        for segment in self.segments.values() {
            let gap_end = segment.window.start_ms;
            if gap_end > cursor && gap_end - cursor >= min_gap_ms {
                gaps.push(TimeWindow {
                    start_ms: cursor,
                    end_ms: gap_end,
                });
            }
            cursor = segment.window.end_ms;
        }

        // Trailing gap
        if end_of_program > cursor && end_of_program - cursor >= min_gap_ms {
            gaps.push(TimeWindow {
                start_ms: cursor,
                end_ms: end_of_program,
            });
        }
        gaps
    }

    /// Find the active segment at a given timecode, if any.
    #[must_use]
    pub fn active_at(&self, timecode_ms: u64) -> Option<&AdSegment> {
        self.segments
            .values()
            .find(|s| s.window.contains(timecode_ms))
    }

    /// Count of all segments.
    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// A registry of AD tracks for a single media asset, supporting multi-track selection.
#[derive(Debug, Clone, Default)]
pub struct AudioDescriptionRegistry {
    tracks: Vec<AudioDescriptionTrack>,
}

impl AudioDescriptionRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self { tracks: Vec::new() }
    }

    /// Register a track.
    pub fn register(&mut self, track: AudioDescriptionTrack) {
        self.tracks.push(track);
    }

    /// Return the number of registered tracks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Select the highest-priority track for the given BCP-47 language.
    ///
    /// When multiple tracks share the same priority, the first registered wins.
    #[must_use]
    pub fn select_track(&self, language: &str) -> Option<&AudioDescriptionTrack> {
        self.tracks
            .iter()
            .filter(|t| t.language == language)
            .min_by_key(|t| t.priority)
    }

    /// Return all tracks for a given language, ordered by priority.
    #[must_use]
    pub fn tracks_for_language(&self, language: &str) -> Vec<&AudioDescriptionTrack> {
        let mut matching: Vec<&AudioDescriptionTrack> = self
            .tracks
            .iter()
            .filter(|t| t.language == language)
            .collect();
        matching.sort_by_key(|t| t.priority);
        matching
    }

    /// Return all registered language codes (deduplicated).
    #[must_use]
    pub fn languages(&self) -> Vec<&str> {
        let mut langs: Vec<&str> = self.tracks.iter().map(|t| t.language.as_str()).collect();
        langs.sort_unstable();
        langs.dedup();
        langs
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_segment(id: &str, start: u64, end: u64, text: &str) -> AdSegment {
        AdSegment::new(id, start, end, text).expect("valid segment")
    }

    #[test]
    fn test_time_window_valid() {
        let w = TimeWindow::new(0, 1000).expect("valid window");
        assert_eq!(w.duration_ms(), 1000);
    }

    #[test]
    fn test_time_window_invalid_reversed() {
        let result = TimeWindow::new(5000, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_time_window_overlap() {
        let a = TimeWindow::new(0, 3000).unwrap();
        let b = TimeWindow::new(2000, 5000).unwrap();
        let c = TimeWindow::new(3000, 6000).unwrap();
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_time_window_contains() {
        let w = TimeWindow::new(1000, 4000).unwrap();
        assert!(w.contains(1000));
        assert!(w.contains(3999));
        assert!(!w.contains(4000));
        assert!(!w.contains(500));
    }

    #[test]
    fn test_ad_segment_word_count_and_wpm() {
        let seg = make_segment("s1", 0, 6_000, "A sunset over the mountains");
        assert_eq!(seg.word_count(), 5);
        let wpm = seg.estimated_wpm().expect("wpm available");
        // 5 words / 0.1 min = 50 wpm
        assert!((wpm - 50.0).abs() < 0.01, "wpm={wpm}");
    }

    #[test]
    fn test_ad_segment_placement_confidence_range() {
        assert!(make_segment("s1", 0, 1000, "text").with_placement_confidence(1.1).is_err());
        assert!(make_segment("s1", 0, 1000, "text").with_placement_confidence(0.0).is_ok());
        assert!(make_segment("s1", 0, 1000, "text").with_placement_confidence(0.75).is_ok());
    }

    #[test]
    fn test_track_overlap_rejection() {
        let mut track = AudioDescriptionTrack::new(
            "t1",
            "en-GB",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        );
        track.add_segment(make_segment("s1", 0, 3000, "First cue")).unwrap();
        let result = track.add_segment(make_segment("s2", 2000, 5000, "Overlapping cue"));
        assert!(result.is_err());
    }

    #[test]
    fn test_track_coverage_and_gaps() {
        let mut track = AudioDescriptionTrack::new(
            "t1",
            "en-GB",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        )
        .with_video_duration_ms(20_000);

        track.add_segment(make_segment("s1", 0, 3_000, "Opening")).unwrap();
        track.add_segment(make_segment("s2", 8_000, 11_000, "Mid")).unwrap();

        assert_eq!(track.coverage_ms(), 6_000);
        let frac = track.coverage_fraction().expect("fraction available");
        assert!((frac - 0.3).abs() < 1e-9, "frac={frac}");

        let gaps = track.gaps(1_000);
        assert_eq!(gaps.len(), 2);
        assert_eq!(gaps[0].start_ms, 3_000);
        assert_eq!(gaps[0].end_ms, 8_000);
        assert_eq!(gaps[1].start_ms, 11_000);
        assert_eq!(gaps[1].end_ms, 20_000);
    }

    #[test]
    fn test_track_active_at() {
        let mut track = AudioDescriptionTrack::new(
            "t1",
            "en",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        );
        track.add_segment(make_segment("s1", 1_000, 4_000, "Hello")).unwrap();
        assert!(track.active_at(2_000).is_some());
        assert!(track.active_at(500).is_none());
        assert!(track.active_at(4_000).is_none());
    }

    #[test]
    fn test_registry_priority_selection() {
        let mut reg = AudioDescriptionRegistry::new();
        reg.register(
            AudioDescriptionTrack::new(
                "fallback",
                "en-GB",
                AudioDescriptionStyle::Standard,
                AdPriority::Fallback,
            )
        );
        reg.register(
            AudioDescriptionTrack::new(
                "primary",
                "en-GB",
                AudioDescriptionStyle::Extended,
                AdPriority::Primary,
            )
        );

        let selected = reg.select_track("en-GB").expect("track found");
        assert_eq!(selected.track_id, "primary");
    }

    #[test]
    fn test_registry_languages() {
        let mut reg = AudioDescriptionRegistry::new();
        reg.register(AudioDescriptionTrack::new(
            "a",
            "en-GB",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        ));
        reg.register(AudioDescriptionTrack::new(
            "b",
            "ja-JP",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        ));
        reg.register(AudioDescriptionTrack::new(
            "c",
            "en-GB",
            AudioDescriptionStyle::Extended,
            AdPriority::Secondary,
        ));

        let langs = reg.languages();
        assert_eq!(langs.len(), 2);
        assert!(langs.contains(&"en-GB"));
        assert!(langs.contains(&"ja-JP"));
    }

    #[test]
    fn test_remove_segment() {
        let mut track = AudioDescriptionTrack::new(
            "t1",
            "en",
            AudioDescriptionStyle::Standard,
            AdPriority::Primary,
        );
        track.add_segment(make_segment("s1", 0, 2_000, "Cue one")).unwrap();
        let removed = track.remove_segment("s1");
        assert!(removed.is_some());
        assert_eq!(track.segment_count(), 0);
    }

    #[test]
    fn test_ad_style_label() {
        assert_eq!(AudioDescriptionStyle::Standard.label(), "Standard");
        assert_eq!(AudioDescriptionStyle::Extended.label(), "Extended");
        assert_eq!(AudioDescriptionStyle::DvsBroadcast.label(), "DVS Broadcast");
    }
}
