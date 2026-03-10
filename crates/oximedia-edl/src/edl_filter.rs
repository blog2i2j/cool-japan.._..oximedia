#![allow(dead_code)]
//! EDL event filtering and selection utilities.
//!
//! This module provides a flexible filtering system for EDL events,
//! allowing selection by reel name, edit type, track type, duration range,
//! timecode range, and compound filter expressions.

use std::collections::HashSet;

/// A filterable event record (simplified for filtering purposes).
#[derive(Debug, Clone)]
pub struct FilterableEvent {
    /// Event number.
    pub number: u32,
    /// Reel name.
    pub reel: String,
    /// Edit type name (e.g., "Cut", "Dissolve", "Wipe").
    pub edit_type: String,
    /// Track type name (e.g., "Video", "Audio", "Both").
    pub track_type: String,
    /// Record in frame.
    pub record_in: u64,
    /// Record out frame.
    pub record_out: u64,
    /// Source in frame.
    pub source_in: u64,
    /// Source out frame.
    pub source_out: u64,
    /// Optional clip name.
    pub clip_name: Option<String>,
}

impl FilterableEvent {
    /// Create a new filterable event.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        number: u32,
        reel: &str,
        edit_type: &str,
        track_type: &str,
        record_in: u64,
        record_out: u64,
        source_in: u64,
        source_out: u64,
    ) -> Self {
        Self {
            number,
            reel: reel.to_string(),
            edit_type: edit_type.to_string(),
            track_type: track_type.to_string(),
            record_in,
            record_out,
            source_in,
            source_out,
            clip_name: None,
        }
    }

    /// Set the clip name.
    #[must_use]
    pub fn with_clip_name(mut self, name: impl Into<String>) -> Self {
        self.clip_name = Some(name.into());
        self
    }

    /// Record duration in frames.
    #[must_use]
    pub fn record_duration(&self) -> u64 {
        self.record_out.saturating_sub(self.record_in)
    }

    /// Source duration in frames.
    #[must_use]
    pub fn source_duration(&self) -> u64 {
        self.source_out.saturating_sub(self.source_in)
    }
}

/// Criteria for filtering events.
#[derive(Debug, Clone)]
pub enum FilterCriterion {
    /// Match events by reel name (exact match).
    Reel(String),
    /// Match events by one of several reel names.
    ReelSet(HashSet<String>),
    /// Match events by edit type name.
    EditType(String),
    /// Match events by track type name.
    TrackType(String),
    /// Match events with record duration in a range (inclusive bounds, in frames).
    DurationRange {
        /// Minimum duration (inclusive).
        min_frames: u64,
        /// Maximum duration (inclusive).
        max_frames: u64,
    },
    /// Match events whose record range overlaps a given frame range.
    RecordRange {
        /// Start frame (inclusive).
        start: u64,
        /// End frame (exclusive).
        end: u64,
    },
    /// Match events with a clip name containing the given substring.
    ClipNameContains(String),
    /// Match events with event number in a given range (inclusive).
    EventNumberRange {
        /// Minimum event number (inclusive).
        min: u32,
        /// Maximum event number (inclusive).
        max: u32,
    },
    /// Logical NOT of another criterion.
    Not(Box<FilterCriterion>),
    /// Logical AND of two criteria.
    And(Box<FilterCriterion>, Box<FilterCriterion>),
    /// Logical OR of two criteria.
    Or(Box<FilterCriterion>, Box<FilterCriterion>),
}

impl FilterCriterion {
    /// Check if an event matches this criterion.
    #[must_use]
    pub fn matches(&self, event: &FilterableEvent) -> bool {
        match self {
            Self::Reel(name) => event.reel == *name,
            Self::ReelSet(names) => names.contains(&event.reel),
            Self::EditType(et) => event.edit_type == *et,
            Self::TrackType(tt) => event.track_type == *tt,
            Self::DurationRange {
                min_frames,
                max_frames,
            } => {
                let dur = event.record_duration();
                dur >= *min_frames && dur < *max_frames
            }
            Self::RecordRange { start, end } => event.record_in < *end && event.record_out > *start,
            Self::ClipNameContains(substring) => event
                .clip_name
                .as_ref()
                .is_some_and(|name| name.contains(substring.as_str())),
            Self::EventNumberRange { min, max } => event.number >= *min && event.number <= *max,
            Self::Not(inner) => !inner.matches(event),
            Self::And(a, b) => a.matches(event) && b.matches(event),
            Self::Or(a, b) => a.matches(event) || b.matches(event),
        }
    }

    /// Create a NOT criterion.
    #[must_use]
    pub fn not(criterion: Self) -> Self {
        Self::Not(Box::new(criterion))
    }

    /// Create an AND criterion.
    #[must_use]
    pub fn and(a: Self, b: Self) -> Self {
        Self::And(Box::new(a), Box::new(b))
    }

    /// Create an OR criterion.
    #[must_use]
    pub fn or(a: Self, b: Self) -> Self {
        Self::Or(Box::new(a), Box::new(b))
    }
}

/// An event filter that applies criteria to select events.
pub struct EventFilter {
    /// The filtering criterion.
    criterion: FilterCriterion,
}

impl EventFilter {
    /// Create a new event filter with a criterion.
    #[must_use]
    pub fn new(criterion: FilterCriterion) -> Self {
        Self { criterion }
    }

    /// Apply the filter to a slice of events, returning matching events.
    #[must_use]
    pub fn apply<'a>(&self, events: &'a [FilterableEvent]) -> Vec<&'a FilterableEvent> {
        events
            .iter()
            .filter(|e| self.criterion.matches(e))
            .collect()
    }

    /// Apply the filter and return owned copies of matching events.
    #[must_use]
    pub fn apply_owned(&self, events: &[FilterableEvent]) -> Vec<FilterableEvent> {
        events
            .iter()
            .filter(|e| self.criterion.matches(e))
            .cloned()
            .collect()
    }

    /// Count events matching the filter.
    #[must_use]
    pub fn count(&self, events: &[FilterableEvent]) -> usize {
        events.iter().filter(|e| self.criterion.matches(e)).count()
    }

    /// Check if any event matches the filter.
    #[must_use]
    pub fn any_match(&self, events: &[FilterableEvent]) -> bool {
        events.iter().any(|e| self.criterion.matches(e))
    }
}

/// Builder for constructing compound filters.
pub struct FilterBuilder {
    /// Accumulated criteria.
    criteria: Vec<FilterCriterion>,
}

impl FilterBuilder {
    /// Create a new filter builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            criteria: Vec::new(),
        }
    }

    /// Add a reel name filter.
    #[must_use]
    pub fn reel(mut self, name: impl Into<String>) -> Self {
        self.criteria.push(FilterCriterion::Reel(name.into()));
        self
    }

    /// Add an edit type filter.
    #[must_use]
    pub fn edit_type(mut self, edit_type: impl Into<String>) -> Self {
        self.criteria
            .push(FilterCriterion::EditType(edit_type.into()));
        self
    }

    /// Add a track type filter.
    #[must_use]
    pub fn track_type(mut self, track_type: impl Into<String>) -> Self {
        self.criteria
            .push(FilterCriterion::TrackType(track_type.into()));
        self
    }

    /// Add a duration range filter.
    #[must_use]
    pub fn duration_range(mut self, min_frames: u64, max_frames: u64) -> Self {
        self.criteria.push(FilterCriterion::DurationRange {
            min_frames,
            max_frames,
        });
        self
    }

    /// Add a record range filter.
    #[must_use]
    pub fn record_range(mut self, start: u64, end: u64) -> Self {
        self.criteria
            .push(FilterCriterion::RecordRange { start, end });
        self
    }

    /// Build the filter (AND-combining all criteria).
    #[must_use]
    pub fn build(self) -> EventFilter {
        if self.criteria.is_empty() {
            // Match everything with a trivially true criterion
            return EventFilter::new(FilterCriterion::DurationRange {
                min_frames: 0,
                max_frames: u64::MAX,
            });
        }

        let mut iter = self.criteria.into_iter();
        // Safety: guarded by the `is_empty()` check above.
        let first = match iter.next() {
            Some(c) => c,
            None => {
                return EventFilter::new(FilterCriterion::DurationRange {
                    min_frames: 0,
                    max_frames: u64::MAX,
                })
            }
        };
        let combined = iter.fold(first, FilterCriterion::and);
        EventFilter::new(combined)
    }
}

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(
        num: u32,
        reel: &str,
        edit: &str,
        track: &str,
        rec_in: u64,
        rec_out: u64,
    ) -> FilterableEvent {
        FilterableEvent::new(num, reel, edit, track, rec_in, rec_out, rec_in, rec_out)
    }

    fn sample_events() -> Vec<FilterableEvent> {
        vec![
            make_event(1, "A001", "Cut", "Video", 0, 100),
            make_event(2, "A002", "Dissolve", "Video", 100, 250),
            make_event(3, "A001", "Cut", "Audio", 250, 400),
            make_event(4, "A003", "Wipe", "Video", 400, 600),
            make_event(5, "A002", "Cut", "Both", 600, 700),
        ]
    }

    #[test]
    fn test_filter_by_reel() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::Reel("A001".to_string()));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|e| e.reel == "A001"));
    }

    #[test]
    fn test_filter_by_reel_set() {
        let events = sample_events();
        let mut reels = HashSet::new();
        reels.insert("A001".to_string());
        reels.insert("A003".to_string());
        let f = EventFilter::new(FilterCriterion::ReelSet(reels));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_by_edit_type() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::EditType("Cut".to_string()));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_by_track_type() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::TrackType("Video".to_string()));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_by_duration_range() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::DurationRange {
            min_frames: 100,
            max_frames: 200,
        });
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 4);
    }

    #[test]
    fn test_filter_by_record_range() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::RecordRange {
            start: 200,
            end: 500,
        });
        let matches = f.apply(&events);
        // Events 2 (100..250), 3 (250..400), 4 (400..600) overlap [200..500)
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_by_clip_name_contains() {
        let mut events = sample_events();
        events[0] = events[0].clone().with_clip_name("interview_take1.mov");
        events[1] = events[1].clone().with_clip_name("broll_park.mov");

        let f = EventFilter::new(FilterCriterion::ClipNameContains("interview".to_string()));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 1);
    }

    #[test]
    fn test_filter_by_event_number_range() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::EventNumberRange { min: 2, max: 4 });
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_filter_not() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::not(FilterCriterion::EditType(
            "Cut".to_string(),
        )));
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 2); // Dissolve, Wipe
    }

    #[test]
    fn test_filter_and() {
        let events = sample_events();
        let criterion = FilterCriterion::and(
            FilterCriterion::Reel("A001".to_string()),
            FilterCriterion::TrackType("Video".to_string()),
        );
        let f = EventFilter::new(criterion);
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 1);
    }

    #[test]
    fn test_filter_or() {
        let events = sample_events();
        let criterion = FilterCriterion::or(
            FilterCriterion::EditType("Dissolve".to_string()),
            FilterCriterion::EditType("Wipe".to_string()),
        );
        let f = EventFilter::new(criterion);
        let matches = f.apply(&events);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_filter_count() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::EditType("Cut".to_string()));
        assert_eq!(f.count(&events), 3);
    }

    #[test]
    fn test_filter_any_match() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::EditType("Wipe".to_string()));
        assert!(f.any_match(&events));

        let f2 = EventFilter::new(FilterCriterion::EditType("Key".to_string()));
        assert!(!f2.any_match(&events));
    }

    #[test]
    fn test_filter_apply_owned() {
        let events = sample_events();
        let f = EventFilter::new(FilterCriterion::Reel("A003".to_string()));
        let owned = f.apply_owned(&events);
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].reel, "A003");
    }

    #[test]
    fn test_filter_builder() {
        let events = sample_events();
        let filter = FilterBuilder::new()
            .reel("A001")
            .track_type("Video")
            .build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].number, 1);
    }

    #[test]
    fn test_filter_builder_empty() {
        let events = sample_events();
        let filter = FilterBuilder::new().build();
        let matches = filter.apply(&events);
        assert_eq!(matches.len(), 5); // matches all
    }

    #[test]
    fn test_filterable_event_durations() {
        let ev = FilterableEvent::new(1, "R1", "Cut", "Video", 100, 200, 50, 175);
        assert_eq!(ev.record_duration(), 100);
        assert_eq!(ev.source_duration(), 125);
    }

    #[test]
    fn test_filter_builder_default() {
        let builder = FilterBuilder::default();
        let events = sample_events();
        let filter = builder.build();
        assert_eq!(filter.count(&events), 5);
    }
}
