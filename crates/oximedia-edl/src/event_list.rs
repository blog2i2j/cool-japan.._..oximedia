#![allow(dead_code)]
//! Indexed event list for efficient EDL event management.
//!
//! This module provides an `EventList` container that wraps a `Vec<EdlEvent>`
//! with indexed lookups by event number, reel name, and record-in timecode,
//! as well as filtering and slicing helpers.

use crate::event::{EditType, EdlEvent, TrackType};
use std::collections::HashMap;

/// An indexed, ordered collection of EDL events.
#[derive(Debug, Clone)]
pub struct EventList {
    /// The underlying events in order.
    events: Vec<EdlEvent>,
}

impl EventList {
    /// Create an empty event list.
    #[must_use]
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Create an event list from an existing vector of events.
    #[must_use]
    pub fn from_events(events: Vec<EdlEvent>) -> Self {
        Self { events }
    }

    /// Return the number of events.
    #[must_use]
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return `true` if there are no events.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Push a new event.
    pub fn push(&mut self, event: EdlEvent) {
        self.events.push(event);
    }

    /// Get an event by its position index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&EdlEvent> {
        self.events.get(index)
    }

    /// Get an event by event number.
    #[must_use]
    pub fn find_by_number(&self, number: u32) -> Option<&EdlEvent> {
        self.events.iter().find(|e| e.number == number)
    }

    /// Get all events that reference a given reel name.
    #[must_use]
    pub fn find_by_reel(&self, reel: &str) -> Vec<&EdlEvent> {
        self.events.iter().filter(|e| e.reel == reel).collect()
    }

    /// Get all events of a given edit type.
    #[must_use]
    pub fn find_by_edit_type(&self, edit_type: EditType) -> Vec<&EdlEvent> {
        self.events
            .iter()
            .filter(|e| e.edit_type == edit_type)
            .collect()
    }

    /// Get all events with the given track type.
    #[must_use]
    pub fn find_by_track(&self, track: &TrackType) -> Vec<&EdlEvent> {
        self.events.iter().filter(|e| &e.track == track).collect()
    }

    /// Return a sub-list of events whose record-in falls within `[start, end)` in frames.
    #[must_use]
    pub fn slice_by_record_range(&self, start_frames: u64, end_frames: u64) -> Vec<&EdlEvent> {
        self.events
            .iter()
            .filter(|e| {
                let f = e.record_in.to_frames();
                f >= start_frames && f < end_frames
            })
            .collect()
    }

    /// Build a reel-name index: maps reel name to a list of event numbers.
    #[must_use]
    pub fn reel_index(&self) -> HashMap<String, Vec<u32>> {
        let mut map: HashMap<String, Vec<u32>> = HashMap::new();
        for e in &self.events {
            map.entry(e.reel.clone()).or_default().push(e.number);
        }
        map
    }

    /// Return a list of unique reel names in the order they first appear.
    #[must_use]
    pub fn unique_reels(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for e in &self.events {
            if !seen.contains(&e.reel) {
                seen.push(e.reel.clone());
            }
        }
        seen
    }

    /// Compute the total duration in frames across all events.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.events.iter().map(|e| e.duration_frames()).sum()
    }

    /// Sort events by record-in timecode (ascending).
    pub fn sort_by_record_in(&mut self) {
        self.events.sort_by_key(|e| e.record_in.to_frames());
    }

    /// Renumber events sequentially starting from the given start number.
    pub fn renumber(&mut self, start: u32) {
        for (i, event) in self.events.iter_mut().enumerate() {
            event.number = start + i as u32;
        }
    }

    /// Remove an event by event number. Returns the removed event if found.
    pub fn remove_by_number(&mut self, number: u32) -> Option<EdlEvent> {
        if let Some(pos) = self.events.iter().position(|e| e.number == number) {
            Some(self.events.remove(pos))
        } else {
            None
        }
    }

    /// Return a reference to the inner slice.
    #[must_use]
    pub fn as_slice(&self) -> &[EdlEvent] {
        &self.events
    }

    /// Consume the list and return the inner vector.
    #[must_use]
    pub fn into_inner(self) -> Vec<EdlEvent> {
        self.events
    }

    /// Detect overlapping events on the same track and return pairs.
    #[must_use]
    pub fn find_overlaps(&self) -> Vec<(u32, u32)> {
        let mut overlaps = Vec::new();
        for i in 0..self.events.len() {
            for j in (i + 1)..self.events.len() {
                if self.events[i].overlaps_with(&self.events[j]) {
                    overlaps.push((self.events[i].number, self.events[j].number));
                }
            }
        }
        overlaps
    }
}

impl Default for EventList {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for EventList {
    type Item = EdlEvent;
    type IntoIter = std::vec::IntoIter<EdlEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.into_iter()
    }
}

impl<'a> IntoIterator for &'a EventList {
    type Item = &'a EdlEvent;
    type IntoIter = std::slice::Iter<'a, EdlEvent>;

    fn into_iter(self) -> Self::IntoIter {
        self.events.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn make_event(num: u32, reel: &str, rec_in_sec: u8, rec_out_sec: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, rec_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, rec_out_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, rec_in_sec, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, rec_out_sec, 0, fr).expect("failed to create"),
        )
    }

    #[test]
    fn test_new_is_empty() {
        let list = EventList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_push_and_len() {
        let mut list = EventList::new();
        list.push(make_event(1, "A001", 0, 5));
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_find_by_number() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        assert!(list.find_by_number(1).is_some());
        assert!(list.find_by_number(3).is_none());
    }

    #[test]
    fn test_find_by_reel() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A001", 5, 10),
            make_event(3, "B001", 10, 15),
        ]);
        assert_eq!(list.find_by_reel("A001").len(), 2);
        assert_eq!(list.find_by_reel("B001").len(), 1);
        assert_eq!(list.find_by_reel("C001").len(), 0);
    }

    #[test]
    fn test_find_by_edit_type() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        assert_eq!(list.find_by_edit_type(EditType::Cut).len(), 2);
        assert_eq!(list.find_by_edit_type(EditType::Dissolve).len(), 0);
    }

    #[test]
    fn test_unique_reels() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "B001", 5, 10),
            make_event(3, "A001", 10, 15),
        ]);
        let reels = list.unique_reels();
        assert_eq!(reels, vec!["A001", "B001"]);
    }

    #[test]
    fn test_reel_index() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A001", 5, 10),
            make_event(3, "B001", 10, 15),
        ]);
        let idx = list.reel_index();
        assert_eq!(idx["A001"], vec![1, 2]);
        assert_eq!(idx["B001"], vec![3]);
    }

    #[test]
    fn test_sort_by_record_in() {
        let mut list = EventList::from_events(vec![
            make_event(2, "A002", 5, 10),
            make_event(1, "A001", 0, 5),
        ]);
        list.sort_by_record_in();
        assert_eq!(list.get(0).expect("failed to get value").number, 1);
        assert_eq!(list.get(1).expect("failed to get value").number, 2);
    }

    #[test]
    fn test_renumber() {
        let mut list = EventList::from_events(vec![
            make_event(10, "A001", 0, 5),
            make_event(20, "A002", 5, 10),
        ]);
        list.renumber(1);
        assert_eq!(list.get(0).expect("failed to get value").number, 1);
        assert_eq!(list.get(1).expect("failed to get value").number, 2);
    }

    #[test]
    fn test_remove_by_number() {
        let mut list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 5, 10),
        ]);
        let removed = list.remove_by_number(1);
        assert!(removed.is_some());
        assert_eq!(list.len(), 1);
        assert!(list.remove_by_number(99).is_none());
    }

    #[test]
    fn test_total_duration_frames() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),  // 5s = 125 frames
            make_event(2, "A002", 5, 10), // 5s = 125 frames
        ]);
        assert_eq!(list.total_duration_frames(), 250);
    }

    #[test]
    fn test_slice_by_record_range() {
        let list = EventList::from_events(vec![
            make_event(1, "A001", 0, 5),
            make_event(2, "A002", 10, 15),
            make_event(3, "A003", 20, 25),
        ]);
        // record_in for event 1 = 1h0m0s = 90000 frames
        // record_in for event 2 = 1h0m10s = 90250 frames
        // record_in for event 3 = 1h0m20s = 90500 frames
        let slice = list.slice_by_record_range(90000, 90300);
        assert_eq!(slice.len(), 2);
    }

    #[test]
    fn test_into_iter() {
        let list = EventList::from_events(vec![make_event(1, "A001", 0, 5)]);
        let collected: Vec<_> = list.into_iter().collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_find_overlaps() {
        let fr = EdlFrameRate::Fps25;
        let e1 = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, 0, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 10, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 0, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 10, 0, fr).expect("failed to create"),
        );
        let e2 = EdlEvent::new(
            2,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, 5, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 15, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 5, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, 15, 0, fr).expect("failed to create"),
        );
        let list = EventList::from_events(vec![e1, e2]);
        let overlaps = list.find_overlaps();
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0], (1, 2));
    }

    #[test]
    fn test_default() {
        let list = EventList::default();
        assert!(list.is_empty());
    }
}
