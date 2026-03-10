#![allow(dead_code)]
//! Clip timeline placement and sequencing.
//!
//! This module provides tools for placing clips on a timeline with precise
//! frame-level positioning, sequencing, gap management, and overlap detection.
//! Supports multi-track timelines where clips can be arranged on different
//! video and audio tracks.

use std::collections::BTreeMap;

/// Unique identifier for a timeline entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimelineEntryId(pub u64);

/// Unique identifier for a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TrackId(pub u32);

/// Track type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Title/graphics track.
    Title,
    /// Data/metadata track.
    Data,
}

/// A track in the timeline.
#[derive(Debug, Clone)]
pub struct TimelineTrack {
    /// Track identifier.
    pub id: TrackId,
    /// Track name.
    pub name: String,
    /// Track type.
    pub track_type: TrackType,
    /// Whether the track is locked (no edits allowed).
    pub locked: bool,
    /// Whether the track is muted/hidden.
    pub muted: bool,
}

impl TimelineTrack {
    /// Creates a new track.
    #[must_use]
    pub fn new(id: TrackId, name: impl Into<String>, track_type: TrackType) -> Self {
        Self {
            id,
            name: name.into(),
            track_type,
            locked: false,
            muted: false,
        }
    }
}

/// A clip placed on the timeline at a specific position.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// Unique entry identifier.
    pub id: TimelineEntryId,
    /// The track this entry is placed on.
    pub track_id: TrackId,
    /// Clip reference identifier (external clip ID).
    pub clip_ref: u64,
    /// Timeline position in frames (start of the clip on the timeline).
    pub timeline_start: u64,
    /// Duration on the timeline in frames.
    pub duration: u64,
    /// Source in-point within the original clip (in frames).
    pub source_in: u64,
    /// Source out-point within the original clip (in frames).
    pub source_out: u64,
    /// Playback speed multiplier (1.0 = normal).
    pub speed: f64,
    /// Optional label for the entry.
    pub label: String,
}

impl TimelineEntry {
    /// Creates a new timeline entry.
    #[must_use]
    pub fn new(
        id: TimelineEntryId,
        track_id: TrackId,
        clip_ref: u64,
        timeline_start: u64,
        duration: u64,
    ) -> Self {
        Self {
            id,
            track_id,
            clip_ref,
            timeline_start,
            duration,
            source_in: 0,
            source_out: duration,
            speed: 1.0,
            label: String::new(),
        }
    }

    /// Returns the timeline end position (exclusive).
    #[must_use]
    pub fn timeline_end(&self) -> u64 {
        self.timeline_start + self.duration
    }

    /// Returns true if this entry overlaps with a given time range.
    #[must_use]
    pub fn overlaps_range(&self, start: u64, end: u64) -> bool {
        self.timeline_start < end && self.timeline_end() > start
    }

    /// Returns true if this entry overlaps with another.
    #[must_use]
    pub fn overlaps_with(&self, other: &Self) -> bool {
        self.track_id == other.track_id
            && self.timeline_start < other.timeline_end()
            && self.timeline_end() > other.timeline_start
    }

    /// Returns the source duration after speed adjustment.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn effective_source_duration(&self) -> u64 {
        if self.speed <= 0.0 {
            return self.duration;
        }
        (self.duration as f64 * self.speed) as u64
    }
}

/// Describes a gap between two entries on a track.
#[derive(Debug, Clone)]
pub struct TimelineGap {
    /// Track where the gap exists.
    pub track_id: TrackId,
    /// Start frame of the gap.
    pub start: u64,
    /// End frame of the gap (exclusive).
    pub end: u64,
}

impl TimelineGap {
    /// Returns the duration of the gap in frames.
    #[must_use]
    pub fn duration(&self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// Multi-track clip timeline manager.
#[derive(Debug)]
pub struct ClipTimeline {
    /// Tracks in the timeline.
    tracks: BTreeMap<TrackId, TimelineTrack>,
    /// All entries, keyed by entry ID.
    entries: BTreeMap<TimelineEntryId, TimelineEntry>,
    /// Next available entry ID.
    next_entry_id: u64,
    /// Timeline frame rate.
    frame_rate: f64,
}

impl ClipTimeline {
    /// Creates a new empty timeline.
    #[must_use]
    pub fn new(frame_rate: f64) -> Self {
        Self {
            tracks: BTreeMap::new(),
            entries: BTreeMap::new(),
            next_entry_id: 1,
            frame_rate: frame_rate.max(1.0),
        }
    }

    /// Adds a track to the timeline.
    pub fn add_track(&mut self, track: TimelineTrack) {
        self.tracks.insert(track.id, track);
    }

    /// Returns a reference to all tracks.
    #[must_use]
    pub fn tracks(&self) -> &BTreeMap<TrackId, TimelineTrack> {
        &self.tracks
    }

    /// Adds a clip entry to the timeline, returning the assigned entry ID.
    pub fn add_entry(
        &mut self,
        track_id: TrackId,
        clip_ref: u64,
        timeline_start: u64,
        duration: u64,
    ) -> TimelineEntryId {
        let id = TimelineEntryId(self.next_entry_id);
        self.next_entry_id += 1;
        let entry = TimelineEntry::new(id, track_id, clip_ref, timeline_start, duration);
        self.entries.insert(id, entry);
        id
    }

    /// Removes an entry from the timeline.
    pub fn remove_entry(&mut self, id: TimelineEntryId) -> Option<TimelineEntry> {
        self.entries.remove(&id)
    }

    /// Returns a reference to an entry.
    #[must_use]
    pub fn get_entry(&self, id: TimelineEntryId) -> Option<&TimelineEntry> {
        self.entries.get(&id)
    }

    /// Returns a mutable reference to an entry.
    pub fn get_entry_mut(&mut self, id: TimelineEntryId) -> Option<&mut TimelineEntry> {
        self.entries.get_mut(&id)
    }

    /// Returns the total number of entries.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total timeline duration (end of the last entry).
    #[must_use]
    pub fn total_duration(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.timeline_end())
            .max()
            .unwrap_or(0)
    }

    /// Returns the total duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> f64 {
        self.total_duration() as f64 / self.frame_rate
    }

    /// Returns all entries on a given track, sorted by timeline position.
    #[must_use]
    pub fn entries_on_track(&self, track_id: TrackId) -> Vec<&TimelineEntry> {
        let mut entries: Vec<&TimelineEntry> = self
            .entries
            .values()
            .filter(|e| e.track_id == track_id)
            .collect();
        entries.sort_by_key(|e| e.timeline_start);
        entries
    }

    /// Detects overlapping entries on each track.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<(TimelineEntryId, TimelineEntryId)> {
        let mut overlaps = Vec::new();
        for track_id in self.tracks.keys() {
            let track_entries = self.entries_on_track(*track_id);
            for i in 0..track_entries.len() {
                for j in (i + 1)..track_entries.len() {
                    if track_entries[i].overlaps_with(track_entries[j]) {
                        overlaps.push((track_entries[i].id, track_entries[j].id));
                    }
                }
            }
        }
        overlaps
    }

    /// Finds gaps on a given track.
    #[must_use]
    pub fn find_gaps(&self, track_id: TrackId) -> Vec<TimelineGap> {
        let entries = self.entries_on_track(track_id);
        let mut gaps = Vec::new();

        if entries.is_empty() {
            return gaps;
        }

        // Check gap before the first entry
        if entries[0].timeline_start > 0 {
            gaps.push(TimelineGap {
                track_id,
                start: 0,
                end: entries[0].timeline_start,
            });
        }

        // Check gaps between entries
        for pair in entries.windows(2) {
            let end_a = pair[0].timeline_end();
            let start_b = pair[1].timeline_start;
            if end_a < start_b {
                gaps.push(TimelineGap {
                    track_id,
                    start: end_a,
                    end: start_b,
                });
            }
        }

        gaps
    }

    /// Moves an entry to a new timeline start position.
    pub fn move_entry(&mut self, id: TimelineEntryId, new_start: u64) -> bool {
        if let Some(entry) = self.entries.get_mut(&id) {
            entry.timeline_start = new_start;
            true
        } else {
            false
        }
    }

    /// Returns the frame rate.
    #[must_use]
    pub fn frame_rate(&self) -> f64 {
        self.frame_rate
    }

    /// Returns all entry IDs sorted by timeline position.
    #[must_use]
    pub fn sorted_entry_ids(&self) -> Vec<TimelineEntryId> {
        let mut entries: Vec<(&TimelineEntryId, &TimelineEntry)> = self.entries.iter().collect();
        entries.sort_by_key(|(_, e)| e.timeline_start);
        entries.into_iter().map(|(id, _)| *id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_timeline() -> ClipTimeline {
        let mut tl = ClipTimeline::new(24.0);
        tl.add_track(TimelineTrack::new(TrackId(1), "V1", TrackType::Video));
        tl.add_track(TimelineTrack::new(TrackId(2), "A1", TrackType::Audio));
        tl
    }

    #[test]
    fn test_timeline_track_new() {
        let t = TimelineTrack::new(TrackId(1), "Video 1", TrackType::Video);
        assert_eq!(t.id, TrackId(1));
        assert_eq!(t.name, "Video 1");
        assert!(!t.locked);
    }

    #[test]
    fn test_timeline_entry_end() {
        let e = TimelineEntry::new(TimelineEntryId(1), TrackId(1), 100, 0, 50);
        assert_eq!(e.timeline_end(), 50);
    }

    #[test]
    fn test_timeline_entry_overlaps_range() {
        let e = TimelineEntry::new(TimelineEntryId(1), TrackId(1), 100, 10, 20);
        assert!(e.overlaps_range(15, 25));
        assert!(e.overlaps_range(0, 15));
        assert!(!e.overlaps_range(30, 40));
        assert!(!e.overlaps_range(0, 10));
    }

    #[test]
    fn test_timeline_entry_overlaps_with() {
        let a = TimelineEntry::new(TimelineEntryId(1), TrackId(1), 100, 0, 20);
        let b = TimelineEntry::new(TimelineEntryId(2), TrackId(1), 101, 15, 20);
        assert!(a.overlaps_with(&b));
        let c = TimelineEntry::new(TimelineEntryId(3), TrackId(1), 102, 20, 10);
        assert!(!a.overlaps_with(&c));
    }

    #[test]
    fn test_timeline_entry_no_overlap_different_tracks() {
        let a = TimelineEntry::new(TimelineEntryId(1), TrackId(1), 100, 0, 20);
        let b = TimelineEntry::new(TimelineEntryId(2), TrackId(2), 101, 0, 20);
        assert!(!a.overlaps_with(&b));
    }

    #[test]
    fn test_effective_source_duration() {
        let mut e = TimelineEntry::new(TimelineEntryId(1), TrackId(1), 100, 0, 100);
        e.speed = 2.0;
        assert_eq!(e.effective_source_duration(), 200);
    }

    #[test]
    fn test_timeline_gap_duration() {
        let g = TimelineGap {
            track_id: TrackId(1),
            start: 50,
            end: 100,
        };
        assert_eq!(g.duration(), 50);
    }

    #[test]
    fn test_clip_timeline_add_entry() {
        let mut tl = make_timeline();
        let id = tl.add_entry(TrackId(1), 100, 0, 50);
        assert_eq!(tl.entry_count(), 1);
        let entry = tl.get_entry(id).expect("get_entry should succeed");
        assert_eq!(entry.clip_ref, 100);
        assert_eq!(entry.duration, 50);
    }

    #[test]
    fn test_clip_timeline_total_duration() {
        let mut tl = make_timeline();
        tl.add_entry(TrackId(1), 100, 0, 50);
        tl.add_entry(TrackId(1), 101, 60, 40);
        assert_eq!(tl.total_duration(), 100);
    }

    #[test]
    fn test_clip_timeline_duration_seconds() {
        let mut tl = make_timeline();
        tl.add_entry(TrackId(1), 100, 0, 240);
        assert!((tl.duration_seconds() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clip_timeline_find_overlaps() {
        let mut tl = make_timeline();
        tl.add_entry(TrackId(1), 100, 0, 50);
        tl.add_entry(TrackId(1), 101, 40, 30); // overlaps
        let overlaps = tl.find_overlaps();
        assert_eq!(overlaps.len(), 1);
    }

    #[test]
    fn test_clip_timeline_find_gaps() {
        let mut tl = make_timeline();
        tl.add_entry(TrackId(1), 100, 10, 20); // frames 10-30
        tl.add_entry(TrackId(1), 101, 50, 20); // frames 50-70
        let gaps = tl.find_gaps(TrackId(1));
        assert_eq!(gaps.len(), 2); // gap 0-10 and 30-50
        assert_eq!(gaps[0].start, 0);
        assert_eq!(gaps[0].end, 10);
        assert_eq!(gaps[1].start, 30);
        assert_eq!(gaps[1].end, 50);
    }

    #[test]
    fn test_clip_timeline_move_entry() {
        let mut tl = make_timeline();
        let id = tl.add_entry(TrackId(1), 100, 0, 50);
        assert!(tl.move_entry(id, 100));
        assert_eq!(
            tl.get_entry(id)
                .expect("get_entry should succeed")
                .timeline_start,
            100
        );
    }

    #[test]
    fn test_clip_timeline_remove_entry() {
        let mut tl = make_timeline();
        let id = tl.add_entry(TrackId(1), 100, 0, 50);
        assert_eq!(tl.entry_count(), 1);
        let removed = tl.remove_entry(id);
        assert!(removed.is_some());
        assert_eq!(tl.entry_count(), 0);
    }

    #[test]
    fn test_sorted_entry_ids() {
        let mut tl = make_timeline();
        let id2 = tl.add_entry(TrackId(1), 101, 50, 20);
        let id1 = tl.add_entry(TrackId(1), 100, 0, 30);
        let sorted = tl.sorted_entry_ids();
        assert_eq!(sorted[0], id1);
        assert_eq!(sorted[1], id2);
    }

    #[test]
    fn test_entries_on_track() {
        let mut tl = make_timeline();
        tl.add_entry(TrackId(1), 100, 0, 30);
        tl.add_entry(TrackId(2), 200, 0, 20);
        tl.add_entry(TrackId(1), 101, 40, 20);
        let v_entries = tl.entries_on_track(TrackId(1));
        assert_eq!(v_entries.len(), 2);
        let a_entries = tl.entries_on_track(TrackId(2));
        assert_eq!(a_entries.len(), 1);
    }
}
