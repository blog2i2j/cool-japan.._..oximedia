#![allow(dead_code)]
//! Landmark-based navigation for media content accessibility.
//!
//! Defines semantic landmarks within media timelines (chapters, scenes,
//! key moments) to enable efficient non-visual navigation and assistive
//! technology integration.

use std::fmt;

/// Type of navigation landmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LandmarkKind {
    /// A chapter boundary.
    Chapter,
    /// A scene change.
    SceneChange,
    /// A key spoken dialogue moment.
    Dialogue,
    /// An important visual event.
    VisualEvent,
    /// A music cue or change.
    MusicCue,
    /// A sound effect of note.
    SoundEffect,
    /// A title card or text overlay.
    TitleCard,
    /// A credit sequence.
    Credits,
    /// A user-defined bookmark.
    Bookmark,
    /// An ad break or interstitial.
    AdBreak,
}

impl fmt::Display for LandmarkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Chapter => "Chapter",
            Self::SceneChange => "Scene Change",
            Self::Dialogue => "Dialogue",
            Self::VisualEvent => "Visual Event",
            Self::MusicCue => "Music Cue",
            Self::SoundEffect => "Sound Effect",
            Self::TitleCard => "Title Card",
            Self::Credits => "Credits",
            Self::Bookmark => "Bookmark",
            Self::AdBreak => "Ad Break",
        };
        write!(f, "{label}")
    }
}

/// A single landmark in the media timeline.
#[derive(Debug, Clone)]
pub struct Landmark {
    /// Unique identifier.
    pub id: u64,
    /// Kind of landmark.
    pub kind: LandmarkKind,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// Optional end time in milliseconds (for range landmarks).
    pub end_ms: Option<u64>,
    /// Human-readable label.
    pub label: String,
    /// Detailed description for screen readers.
    pub description: String,
    /// Importance level (1 = least, 10 = most).
    pub importance: u8,
}

impl Landmark {
    /// Create a new point landmark (no duration).
    #[must_use]
    pub fn point(id: u64, kind: LandmarkKind, start_ms: u64, label: impl Into<String>) -> Self {
        Self {
            id,
            kind,
            start_ms,
            end_ms: None,
            label: label.into(),
            description: String::new(),
            importance: 5,
        }
    }

    /// Create a range landmark with start and end.
    #[must_use]
    pub fn range(
        id: u64,
        kind: LandmarkKind,
        start_ms: u64,
        end_ms: u64,
        label: impl Into<String>,
    ) -> Self {
        Self {
            id,
            kind,
            start_ms,
            end_ms: Some(end_ms),
            label: label.into(),
            description: String::new(),
            importance: 5,
        }
    }

    /// Set the description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the importance level (clamped to 1-10).
    #[must_use]
    pub fn with_importance(mut self, importance: u8) -> Self {
        self.importance = importance.clamp(1, 10);
        self
    }

    /// Get the duration in milliseconds (0 for point landmarks).
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.map_or(0, |end| end.saturating_sub(self.start_ms))
    }

    /// Check if a timestamp falls within this landmark.
    #[must_use]
    pub fn contains_time(&self, time_ms: u64) -> bool {
        match self.end_ms {
            Some(end) => time_ms >= self.start_ms && time_ms <= end,
            None => time_ms == self.start_ms,
        }
    }
}

/// A collection of landmarks forming a navigation index for media content.
#[derive(Debug)]
pub struct LandmarkIndex {
    /// Total media duration in milliseconds.
    total_duration_ms: u64,
    /// All landmarks, sorted by start time.
    landmarks: Vec<Landmark>,
    /// Next ID to assign.
    next_id: u64,
}

impl LandmarkIndex {
    /// Create a new empty landmark index.
    #[must_use]
    pub fn new(total_duration_ms: u64) -> Self {
        Self {
            total_duration_ms,
            landmarks: Vec::new(),
            next_id: 1,
        }
    }

    /// Add a landmark and return its assigned ID.
    pub fn add(&mut self, kind: LandmarkKind, start_ms: u64, label: impl Into<String>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let landmark = Landmark::point(id, kind, start_ms, label);
        self.landmarks.push(landmark);
        self.landmarks.sort_by_key(|l| l.start_ms);
        id
    }

    /// Add a range landmark and return its assigned ID.
    pub fn add_range(
        &mut self,
        kind: LandmarkKind,
        start_ms: u64,
        end_ms: u64,
        label: impl Into<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let landmark = Landmark::range(id, kind, start_ms, end_ms, label);
        self.landmarks.push(landmark);
        self.landmarks.sort_by_key(|l| l.start_ms);
        id
    }

    /// Get total number of landmarks.
    #[must_use]
    pub fn len(&self) -> usize {
        self.landmarks.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.landmarks.is_empty()
    }

    /// Get a landmark by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.id == id)
    }

    /// Remove a landmark by ID.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.landmarks.len();
        self.landmarks.retain(|l| l.id != id);
        self.landmarks.len() < before
    }

    /// Get all landmarks of a given kind.
    #[must_use]
    pub fn filter_by_kind(&self, kind: LandmarkKind) -> Vec<&Landmark> {
        self.landmarks.iter().filter(|l| l.kind == kind).collect()
    }

    /// Get all landmarks with importance >= threshold.
    #[must_use]
    pub fn filter_by_importance(&self, min_importance: u8) -> Vec<&Landmark> {
        self.landmarks
            .iter()
            .filter(|l| l.importance >= min_importance)
            .collect()
    }

    /// Find the nearest landmark at or after a given time.
    #[must_use]
    pub fn next_landmark(&self, time_ms: u64) -> Option<&Landmark> {
        self.landmarks.iter().find(|l| l.start_ms >= time_ms)
    }

    /// Find the nearest landmark before a given time.
    #[must_use]
    pub fn prev_landmark(&self, time_ms: u64) -> Option<&Landmark> {
        self.landmarks.iter().rev().find(|l| l.start_ms < time_ms)
    }

    /// Find all landmarks that contain the given timestamp.
    #[must_use]
    pub fn landmarks_at(&self, time_ms: u64) -> Vec<&Landmark> {
        self.landmarks
            .iter()
            .filter(|l| l.contains_time(time_ms))
            .collect()
    }

    /// Get all landmarks as a slice.
    #[must_use]
    pub fn all(&self) -> &[Landmark] {
        &self.landmarks
    }

    /// Get the total media duration.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.total_duration_ms
    }

    /// Generate a chapter list (landmarks of kind Chapter).
    #[must_use]
    pub fn chapters(&self) -> Vec<&Landmark> {
        self.filter_by_kind(LandmarkKind::Chapter)
    }

    /// Generate a summary: count of landmarks per kind.
    #[must_use]
    pub fn summary(&self) -> Vec<(LandmarkKind, usize)> {
        use std::collections::HashMap;
        let mut counts: HashMap<LandmarkKind, usize> = HashMap::new();
        for lm in &self.landmarks {
            *counts.entry(lm.kind).or_insert(0) += 1;
        }
        let mut result: Vec<(LandmarkKind, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_landmark_creation() {
        let lm = Landmark::point(1, LandmarkKind::Chapter, 5000, "Chapter 1");
        assert_eq!(lm.id, 1);
        assert_eq!(lm.kind, LandmarkKind::Chapter);
        assert_eq!(lm.start_ms, 5000);
        assert!(lm.end_ms.is_none());
        assert_eq!(lm.duration_ms(), 0);
    }

    #[test]
    fn test_range_landmark_creation() {
        let lm = Landmark::range(2, LandmarkKind::SceneChange, 1000, 3000, "Scene 1");
        assert_eq!(lm.duration_ms(), 2000);
        assert!(lm.contains_time(2000));
        assert!(!lm.contains_time(4000));
    }

    #[test]
    fn test_landmark_contains_time_point() {
        let lm = Landmark::point(1, LandmarkKind::Bookmark, 500, "Mark");
        assert!(lm.contains_time(500));
        assert!(!lm.contains_time(501));
    }

    #[test]
    fn test_landmark_with_description() {
        let lm = Landmark::point(1, LandmarkKind::Dialogue, 1000, "Speech")
            .with_description("Main character speaks");
        assert_eq!(lm.description, "Main character speaks");
    }

    #[test]
    fn test_landmark_with_importance() {
        let lm = Landmark::point(1, LandmarkKind::VisualEvent, 1000, "Explosion")
            .with_importance(9);
        assert_eq!(lm.importance, 9);
        // Clamped
        let lm2 = Landmark::point(2, LandmarkKind::MusicCue, 2000, "Cue")
            .with_importance(15);
        assert_eq!(lm2.importance, 10);
    }

    #[test]
    fn test_index_add_and_get() {
        let mut idx = LandmarkIndex::new(60_000);
        let id = idx.add(LandmarkKind::Chapter, 0, "Intro");
        assert_eq!(idx.len(), 1);
        let lm = idx.get(id).expect("lm should be valid");
        assert_eq!(lm.label, "Intro");
    }

    #[test]
    fn test_index_sorted_order() {
        let mut idx = LandmarkIndex::new(60_000);
        idx.add(LandmarkKind::Chapter, 30_000, "Chapter 2");
        idx.add(LandmarkKind::Chapter, 10_000, "Chapter 1");
        idx.add(LandmarkKind::Chapter, 50_000, "Chapter 3");
        let all = idx.all();
        assert_eq!(all[0].start_ms, 10_000);
        assert_eq!(all[1].start_ms, 30_000);
        assert_eq!(all[2].start_ms, 50_000);
    }

    #[test]
    fn test_index_remove() {
        let mut idx = LandmarkIndex::new(60_000);
        let id = idx.add(LandmarkKind::Bookmark, 1000, "Bm");
        assert!(idx.remove(id));
        assert!(idx.is_empty());
        assert!(!idx.remove(999));
    }

    #[test]
    fn test_filter_by_kind() {
        let mut idx = LandmarkIndex::new(60_000);
        idx.add(LandmarkKind::Chapter, 0, "Ch1");
        idx.add(LandmarkKind::SceneChange, 5000, "Sc1");
        idx.add(LandmarkKind::Chapter, 30_000, "Ch2");
        let chapters = idx.filter_by_kind(LandmarkKind::Chapter);
        assert_eq!(chapters.len(), 2);
    }

    #[test]
    fn test_next_and_prev_landmark() {
        let mut idx = LandmarkIndex::new(60_000);
        idx.add(LandmarkKind::Chapter, 10_000, "Ch1");
        idx.add(LandmarkKind::Chapter, 30_000, "Ch2");
        idx.add(LandmarkKind::Chapter, 50_000, "Ch3");

        let next = idx.next_landmark(15_000).expect("next should be valid");
        assert_eq!(next.start_ms, 30_000);

        let prev = idx.prev_landmark(35_000).expect("prev should be valid");
        assert_eq!(prev.start_ms, 30_000);
    }

    #[test]
    fn test_landmarks_at() {
        let mut idx = LandmarkIndex::new(60_000);
        idx.add_range(LandmarkKind::Dialogue, 1000, 5000, "Speech A");
        idx.add_range(LandmarkKind::MusicCue, 3000, 8000, "Music B");
        let at_4000 = idx.landmarks_at(4000);
        assert_eq!(at_4000.len(), 2);
    }

    #[test]
    fn test_chapters() {
        let mut idx = LandmarkIndex::new(120_000);
        idx.add(LandmarkKind::Chapter, 0, "Intro");
        idx.add(LandmarkKind::SceneChange, 5000, "Scene");
        idx.add(LandmarkKind::Chapter, 60_000, "Part 2");
        let chapters = idx.chapters();
        assert_eq!(chapters.len(), 2);
    }

    #[test]
    fn test_summary() {
        let mut idx = LandmarkIndex::new(60_000);
        idx.add(LandmarkKind::Chapter, 0, "Ch1");
        idx.add(LandmarkKind::Chapter, 30_000, "Ch2");
        idx.add(LandmarkKind::SceneChange, 15_000, "Sc1");
        let summary = idx.summary();
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0].1, 2); // Chapter is most frequent
    }

    #[test]
    fn test_landmark_kind_display() {
        assert_eq!(format!("{}", LandmarkKind::Chapter), "Chapter");
        assert_eq!(format!("{}", LandmarkKind::SceneChange), "Scene Change");
        assert_eq!(format!("{}", LandmarkKind::Credits), "Credits");
        assert_eq!(format!("{}", LandmarkKind::AdBreak), "Ad Break");
    }
}
