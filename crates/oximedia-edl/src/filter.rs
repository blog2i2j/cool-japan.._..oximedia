//! EDL event filtering with builder pattern.
//!
//! This module provides `EdlFilter` and `EdlFilterBuilder` for filtering EDL events
//! directly against `EdlEvent` (rather than the simplified `FilterableEvent`).
//! Supports filtering by reel name pattern, event/edit type, track assignment,
//! timecode range, and transition type.

use crate::event::{EditType, EdlEvent, TrackType};
use crate::timecode::EdlTimecode;

/// A predicate-based filter for EDL events.
///
/// `EdlFilter` holds a list of predicates. An event passes the filter if and
/// only if *all* predicates return `true` (logical AND).
#[derive(Default)]
pub struct EdlFilter {
    predicates: Vec<Box<dyn Fn(&EdlEvent) -> bool>>,
}

impl EdlFilter {
    /// Create a new empty filter that matches everything.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom predicate to the filter chain.
    pub fn add_predicate(&mut self, predicate: impl Fn(&EdlEvent) -> bool + 'static) {
        self.predicates.push(Box::new(predicate));
    }

    /// Test whether a single event matches all predicates.
    #[must_use]
    pub fn matches(&self, event: &EdlEvent) -> bool {
        self.predicates.iter().all(|p| p(event))
    }

    /// Apply the filter to a slice of events, returning references to matches.
    #[must_use]
    pub fn apply<'a>(&self, events: &'a [EdlEvent]) -> Vec<&'a EdlEvent> {
        events.iter().filter(|e| self.matches(e)).collect()
    }

    /// Apply the filter and return owned clones of matching events.
    #[must_use]
    pub fn apply_owned(&self, events: &[EdlEvent]) -> Vec<EdlEvent> {
        events
            .iter()
            .filter(|e| self.matches(e))
            .cloned()
            .collect()
    }

    /// Count events matching all predicates.
    #[must_use]
    pub fn count(&self, events: &[EdlEvent]) -> usize {
        events.iter().filter(|e| self.matches(e)).count()
    }

    /// Check if any event matches the filter.
    #[must_use]
    pub fn any_match(&self, events: &[EdlEvent]) -> bool {
        events.iter().any(|e| self.matches(e))
    }
}

impl std::fmt::Debug for EdlFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdlFilter")
            .field("predicate_count", &self.predicates.len())
            .finish()
    }
}

/// Builder for constructing `EdlFilter` with a fluent chaining API.
///
/// Each chained method adds a predicate; the final `build()` call
/// combines them with logical AND.
///
/// # Example
///
/// ```ignore
/// let filter = EdlFilterBuilder::new()
///     .reel_pattern("A0")
///     .edit_type(EditType::Cut)
///     .video_only()
///     .build();
/// let matches = filter.apply(&edl.events);
/// ```
#[derive(Default)]
pub struct EdlFilterBuilder {
    predicates: Vec<Box<dyn Fn(&EdlEvent) -> bool>>,
}

impl EdlFilterBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter events whose reel name contains `pattern` (case-insensitive substring).
    #[must_use]
    pub fn reel_pattern(mut self, pattern: impl Into<String>) -> Self {
        let pat = pattern.into().to_lowercase();
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.reel.to_lowercase().contains(&pat)));
        self
    }

    /// Filter events whose reel name matches exactly.
    #[must_use]
    pub fn reel_exact(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.reel == name));
        self
    }

    /// Filter events by edit/transition type (Cut, Dissolve, Wipe, Key).
    #[must_use]
    pub fn edit_type(mut self, edit_type: EditType) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.edit_type == edit_type));
        self
    }

    /// Filter events that are transitions (Dissolve or Wipe — not Cut, not Key).
    #[must_use]
    pub fn transitions_only(mut self) -> Self {
        self.predicates.push(Box::new(|e: &EdlEvent| {
            matches!(e.edit_type, EditType::Dissolve | EditType::Wipe)
        }));
        self
    }

    /// Filter events that have video on their track.
    #[must_use]
    pub fn video_only(mut self) -> Self {
        self.predicates
            .push(Box::new(|e: &EdlEvent| e.track.has_video()));
        self
    }

    /// Filter events that have audio on their track.
    #[must_use]
    pub fn audio_only(mut self) -> Self {
        self.predicates
            .push(Box::new(|e: &EdlEvent| e.track.has_audio()));
        self
    }

    /// Filter events by exact track type.
    #[must_use]
    pub fn track(mut self, track_type: TrackType) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.track == track_type));
        self
    }

    /// Filter events whose record-in timecode is at or after `start`.
    #[must_use]
    pub fn record_in_from(mut self, start: EdlTimecode) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.record_in >= start));
        self
    }

    /// Filter events whose record-out timecode is at or before `end`.
    #[must_use]
    pub fn record_out_until(mut self, end: EdlTimecode) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.record_out <= end));
        self
    }

    /// Filter events whose record range overlaps `[start, end)` in frames.
    #[must_use]
    pub fn record_range(mut self, start: EdlTimecode, end: EdlTimecode) -> Self {
        self.predicates.push(Box::new(move |e: &EdlEvent| {
            e.record_in.to_frames() < end.to_frames()
                && e.record_out.to_frames() > start.to_frames()
        }));
        self
    }

    /// Filter events with a minimum duration (in frames).
    #[must_use]
    pub fn min_duration_frames(mut self, min_frames: u64) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.duration_frames() >= min_frames));
        self
    }

    /// Filter events with a maximum duration (in frames).
    #[must_use]
    pub fn max_duration_frames(mut self, max_frames: u64) -> Self {
        self.predicates
            .push(Box::new(move |e: &EdlEvent| e.duration_frames() <= max_frames));
        self
    }

    /// Filter events that have a clip name set.
    #[must_use]
    pub fn has_clip_name(mut self) -> Self {
        self.predicates
            .push(Box::new(|e: &EdlEvent| e.clip_name.is_some()));
        self
    }

    /// Filter events whose clip name contains a substring (case-insensitive).
    #[must_use]
    pub fn clip_name_contains(mut self, pattern: impl Into<String>) -> Self {
        let pat = pattern.into().to_lowercase();
        self.predicates.push(Box::new(move |e: &EdlEvent| {
            e.clip_name
                .as_ref()
                .is_some_and(|n| n.to_lowercase().contains(&pat))
        }));
        self
    }

    /// Filter events that have a motion effect.
    #[must_use]
    pub fn has_motion_effect(mut self) -> Self {
        self.predicates
            .push(Box::new(|e: &EdlEvent| e.motion_effect.is_some()));
        self
    }

    /// Invert the *last* predicate added (logical NOT).
    /// Does nothing if the builder is empty.
    #[must_use]
    pub fn negate_last(mut self) -> Self {
        if let Some(pred) = self.predicates.pop() {
            self.predicates
                .push(Box::new(move |e: &EdlEvent| !pred(e)));
        }
        self
    }

    /// Consume the builder and produce an `EdlFilter`.
    #[must_use]
    pub fn build(self) -> EdlFilter {
        EdlFilter {
            predicates: self.predicates,
        }
    }
}

impl std::fmt::Debug for EdlFilterBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EdlFilterBuilder")
            .field("predicate_count", &self.predicates.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioChannel;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("valid timecode")
    }

    fn make_event(num: u32, reel: &str, track: TrackType, edit: EditType) -> EdlEvent {
        let t_in = tc(1, 0, (num as u8).saturating_mul(5), 0);
        let t_out = tc(1, 0, (num as u8).saturating_mul(5) + 5, 0);
        EdlEvent::new(num, reel.to_string(), track, edit, t_in, t_out, t_in, t_out)
    }

    fn sample_events() -> Vec<EdlEvent> {
        vec![
            make_event(1, "A001", TrackType::Video, EditType::Cut),
            make_event(2, "A002", TrackType::Audio(AudioChannel::A1), EditType::Cut),
            make_event(3, "B001", TrackType::Video, EditType::Dissolve),
            make_event(4, "A001", TrackType::AudioWithVideo, EditType::Cut),
            make_event(5, "C001", TrackType::Video, EditType::Cut),
        ]
    }

    #[test]
    fn test_empty_filter_matches_all() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().build();
        assert_eq!(filter.count(&events), 5);
    }

    #[test]
    fn test_filter_by_reel_pattern() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().reel_pattern("A0").build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 3);
        for m in &matches {
            assert!(m.reel.contains("A0"));
        }
    }

    #[test]
    fn test_filter_by_reel_exact() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().reel_exact("B001").build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].reel, "B001");
    }

    #[test]
    fn test_filter_by_edit_type() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().edit_type(EditType::Dissolve).build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 3);
    }

    #[test]
    fn test_filter_video_only() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().video_only().build();
        let matches = filter.apply(&events);
        // Video, AudioWithVideo, Video, Video => 4
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn test_filter_audio_only() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().audio_only().build();
        let matches = filter.apply(&events);
        // Audio(A1), AudioWithVideo => 2
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_filter_by_track_exact() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new()
            .track(TrackType::Audio(AudioChannel::A1))
            .build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 2);
    }

    #[test]
    fn test_filter_by_record_range() {
        let events = sample_events();
        let start = tc(1, 0, 5, 0);
        let end = tc(1, 0, 20, 0);
        let filter = EdlFilterBuilder::new().record_range(start, end).build();
        let matches = filter.apply(&events);
        // events 2 (5..10), 3 (10..15) overlap [5..20); event 4 (15..20) record_in < 20 and record_out > 5 => yes
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_combined_reel_and_edit() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new()
            .reel_pattern("A0")
            .edit_type(EditType::Cut)
            .build();
        let matches = filter.apply(&events);
        // A001 Cut, A002 Cut, A001 Cut => 3
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_transitions_only() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().transitions_only().build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].edit_type, EditType::Dissolve);
    }

    #[test]
    fn test_filter_min_duration() {
        let events = sample_events();
        // Each event is 5 seconds = 125 frames at 25fps
        let filter = EdlFilterBuilder::new()
            .min_duration_frames(125)
            .build();
        assert_eq!(filter.count(&events), 5);

        let filter2 = EdlFilterBuilder::new()
            .min_duration_frames(126)
            .build();
        assert_eq!(filter2.count(&events), 0);
    }

    #[test]
    fn test_filter_negate_last() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new()
            .edit_type(EditType::Cut)
            .negate_last()
            .build();
        let matches = filter.apply(&events);
        // everything that is NOT Cut => Dissolve only
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].edit_type, EditType::Dissolve);
    }

    #[test]
    fn test_filter_apply_owned() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().reel_exact("C001").build();
        let owned = filter.apply_owned(&events);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].reel, "C001");
    }

    #[test]
    fn test_filter_any_match() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().reel_exact("C001").build();
        assert!(filter.any_match(&events));

        let filter_none = EdlFilterBuilder::new().reel_exact("ZZZZ").build();
        assert!(!filter_none.any_match(&events));
    }

    #[test]
    fn test_filter_clip_name_contains() {
        let mut events = sample_events();
        events[0].set_clip_name("interview_take1.mov".to_string());
        events[1].set_clip_name("broll_park.mov".to_string());

        let filter = EdlFilterBuilder::new()
            .clip_name_contains("interview")
            .build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 1);
    }

    #[test]
    fn test_filter_has_clip_name() {
        let mut events = sample_events();
        events[0].set_clip_name("shot1.mov".to_string());

        let filter = EdlFilterBuilder::new().has_clip_name().build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 1);
    }

    #[test]
    fn test_filter_record_in_from() {
        let events = sample_events();
        let start = tc(1, 0, 15, 0);
        let filter = EdlFilterBuilder::new().record_in_from(start).build();
        let matches = filter.apply(&events);
        // event 3 (15..20), event 4 (20..25), event 5 (25..30) have record_in >= 1:00:15:00
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_case_insensitive_reel() {
        let events = sample_events();
        let filter = EdlFilterBuilder::new().reel_pattern("a001").build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 2); // A001 appears twice
    }

    #[test]
    fn test_filter_debug_impl() {
        let filter = EdlFilterBuilder::new().reel_exact("X").build();
        let debug_str = format!("{filter:?}");
        assert!(debug_str.contains("EdlFilter"));
        assert!(debug_str.contains("predicate_count"));
    }
}
