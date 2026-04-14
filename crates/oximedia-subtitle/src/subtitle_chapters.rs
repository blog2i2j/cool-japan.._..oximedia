//! Chapter-based subtitle organisation for DVD/Blu-ray-style chapter navigation.
//!
//! Media containers such as MKV, MP4 and DVD often carry chapter markers.
//! This module provides:
//!
//! - [`Chapter`] — a named time range with optional metadata.
//! - [`ChapterTrack`] — an ordered list of chapters with O(log n) lookups.
//! - [`ChapterIndex`] — maps subtitle cues to the chapters they belong to, enabling
//!   per-chapter subtitle extraction, statistics, and chapter-aware re-timing.
//!
//! # Example
//!
//! ```
//! use oximedia_subtitle::subtitle_chapters::{Chapter, ChapterTrack, ChapterIndex};
//! use oximedia_subtitle::Subtitle;
//!
//! let chapters = ChapterTrack::from_vec(vec![
//!     Chapter::new(0, 60_000, "Intro"),
//!     Chapter::new(60_000, 120_000, "Act 1"),
//! ]);
//!
//! let subtitles = vec![
//!     Subtitle::new(1_000, 4_000, "Opening narration.".to_string()),
//!     Subtitle::new(61_000, 64_000, "First line of act 1.".to_string()),
//! ];
//!
//! let index = ChapterIndex::build(&chapters, &subtitles);
//! assert_eq!(index.cues_in_chapter(0).len(), 1);
//! assert_eq!(index.cues_in_chapter(1).len(), 1);
//! ```

#![allow(dead_code)]

use crate::Subtitle;
use std::collections::HashMap;

// ============================================================================
// Chapter
// ============================================================================

/// A single named chapter marker.
#[derive(Clone, Debug, PartialEq)]
pub struct Chapter {
    /// Start time in milliseconds (inclusive).
    pub start_ms: i64,
    /// End time in milliseconds (exclusive).
    pub end_ms: i64,
    /// Human-readable chapter title.
    pub title: String,
    /// Optional zero-based chapter number override.
    pub number: Option<u32>,
    /// Optional language tag for the title (e.g. "en", "ja").
    pub language: Option<String>,
}

impl Chapter {
    /// Create a chapter with a title.
    #[must_use]
    pub fn new(start_ms: i64, end_ms: i64, title: impl Into<String>) -> Self {
        Self {
            start_ms,
            end_ms,
            title: title.into(),
            number: None,
            language: None,
        }
    }

    /// Set the chapter number.
    #[must_use]
    pub fn with_number(mut self, n: u32) -> Self {
        self.number = Some(n);
        self
    }

    /// Set the language tag.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Duration of this chapter in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        self.end_ms - self.start_ms
    }

    /// Check whether a given timestamp falls inside this chapter.
    #[must_use]
    pub fn contains(&self, ts_ms: i64) -> bool {
        ts_ms >= self.start_ms && ts_ms < self.end_ms
    }
}

// ============================================================================
// ChapterTrack
// ============================================================================

/// An ordered collection of chapters for a single media asset.
#[derive(Clone, Debug, Default)]
pub struct ChapterTrack {
    chapters: Vec<Chapter>,
}

impl ChapterTrack {
    /// Create an empty track.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a chapter track from a pre-sorted vector of chapters.
    ///
    /// Chapters are sorted by `start_ms` to allow binary search lookups.
    #[must_use]
    pub fn from_vec(mut chapters: Vec<Chapter>) -> Self {
        chapters.sort_by_key(|c| c.start_ms);
        Self { chapters }
    }

    /// Add a chapter, maintaining sorted order.
    pub fn add(&mut self, chapter: Chapter) {
        let pos = self
            .chapters
            .partition_point(|c| c.start_ms <= chapter.start_ms);
        self.chapters.insert(pos, chapter);
    }

    /// Number of chapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    /// `true` if there are no chapters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Get a reference to a chapter by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }

    /// Iterate over all chapters.
    pub fn iter(&self) -> impl Iterator<Item = &Chapter> {
        self.chapters.iter()
    }

    /// Find which chapter (by index) contains the given timestamp using binary search.
    ///
    /// Returns `None` if the timestamp falls before the first chapter or
    /// after the last chapter ends.
    #[must_use]
    pub fn chapter_at(&self, ts_ms: i64) -> Option<usize> {
        if self.chapters.is_empty() {
            return None;
        }

        // Binary search for the last chapter whose start ≤ ts_ms.
        let idx = self
            .chapters
            .partition_point(|c| c.start_ms <= ts_ms);

        // partition_point returns the *first* index where the predicate is false,
        // so the last chapter with start ≤ ts_ms is at idx-1 (if idx > 0).
        let idx = idx.checked_sub(1)?;

        let chapter = &self.chapters[idx];
        if chapter.contains(ts_ms) {
            Some(idx)
        } else {
            None
        }
    }

    /// Return all chapters as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[Chapter] {
        &self.chapters
    }

    /// Total duration covered by all chapters in milliseconds.
    ///
    /// Gaps between chapters are included in the span from first start to last end.
    #[must_use]
    pub fn total_span_ms(&self) -> i64 {
        match (self.chapters.first(), self.chapters.last()) {
            (Some(first), Some(last)) => last.end_ms - first.start_ms,
            _ => 0,
        }
    }

    /// Retimes all chapters by applying a linear scale + offset.
    ///
    /// `new_ts = (old_ts * scale) + offset_ms`
    #[must_use]
    pub fn retime(&self, scale: f64, offset_ms: i64) -> Self {
        let chapters = self
            .chapters
            .iter()
            .map(|c| {
                let new_start = ((c.start_ms as f64 * scale) as i64) + offset_ms;
                let new_end = ((c.end_ms as f64 * scale) as i64) + offset_ms;
                let mut nc = c.clone();
                nc.start_ms = new_start;
                nc.end_ms = new_end;
                nc
            })
            .collect();
        Self { chapters }
    }
}

// ============================================================================
// ChapterIndex
// ============================================================================

/// Maps subtitle cues to the chapters they belong to, and provides per-chapter
/// extraction and statistics.
#[derive(Clone, Debug)]
pub struct ChapterIndex {
    /// Map from chapter index → sorted list of subtitle cue indices.
    chapter_to_cues: HashMap<usize, Vec<usize>>,
    /// Map from subtitle cue index → chapter index (or `None` for uncovered cues).
    cue_to_chapter: Vec<Option<usize>>,
    /// Total subtitle cues that could not be assigned to any chapter.
    unassigned_count: usize,
}

impl ChapterIndex {
    /// Build the index by assigning each subtitle cue to a chapter.
    ///
    /// A cue is assigned to the chapter containing its `start_time`.
    /// Cues that fall outside all chapter ranges are recorded as unassigned.
    #[must_use]
    pub fn build(track: &ChapterTrack, subtitles: &[Subtitle]) -> Self {
        let mut chapter_to_cues: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut cue_to_chapter: Vec<Option<usize>> = Vec::with_capacity(subtitles.len());
        let mut unassigned_count = 0;

        for (cue_idx, sub) in subtitles.iter().enumerate() {
            let chapter_idx = track.chapter_at(sub.start_time);
            cue_to_chapter.push(chapter_idx);
            if let Some(cidx) = chapter_idx {
                chapter_to_cues.entry(cidx).or_default().push(cue_idx);
            } else {
                unassigned_count += 1;
            }
        }

        Self {
            chapter_to_cues,
            cue_to_chapter,
            unassigned_count,
        }
    }

    /// Return the indices of subtitle cues assigned to a given chapter.
    #[must_use]
    pub fn cues_in_chapter(&self, chapter_idx: usize) -> &[usize] {
        self.chapter_to_cues
            .get(&chapter_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Return the chapter index for a given subtitle cue, if assigned.
    #[must_use]
    pub fn chapter_of_cue(&self, cue_idx: usize) -> Option<usize> {
        self.cue_to_chapter.get(cue_idx).copied().flatten()
    }

    /// Number of subtitle cues not assigned to any chapter.
    #[must_use]
    pub fn unassigned_count(&self) -> usize {
        self.unassigned_count
    }

    /// Number of chapters that have at least one subtitle cue.
    #[must_use]
    pub fn non_empty_chapter_count(&self) -> usize {
        self.chapter_to_cues.len()
    }
}

// ============================================================================
// Chapter-aware extraction utilities
// ============================================================================

/// Extract subtitle cues belonging to a specific chapter.
///
/// Returns a new `Vec<Subtitle>` with timecodes relative to the chapter start,
/// so they start at 0 ms within the chapter.
#[must_use]
pub fn extract_chapter_subtitles(
    track: &ChapterTrack,
    subtitles: &[Subtitle],
    chapter_idx: usize,
) -> Vec<Subtitle> {
    let chapter = match track.get(chapter_idx) {
        Some(c) => c,
        None => return Vec::new(),
    };

    let offset = chapter.start_ms;
    subtitles
        .iter()
        .filter(|sub| {
            // Include cues that overlap with the chapter range.
            sub.start_time < chapter.end_ms && sub.end_time > chapter.start_ms
        })
        .map(|sub| {
            let new_start = (sub.start_time - offset).max(0);
            let new_end = (sub.end_time - offset).max(1);
            let mut out = sub.clone();
            out.start_time = new_start;
            out.end_time = new_end;
            out
        })
        .collect()
}

// ============================================================================
// Chapter statistics
// ============================================================================

/// Statistics about subtitle density within a chapter.
#[derive(Clone, Debug)]
pub struct ChapterSubtitleStats {
    /// Chapter index.
    pub chapter_index: usize,
    /// Chapter title.
    pub chapter_title: String,
    /// Number of subtitle cues in this chapter.
    pub cue_count: usize,
    /// Total characters of subtitle text.
    pub total_chars: usize,
    /// Average cue duration in milliseconds.
    pub avg_duration_ms: f64,
    /// Subtitle density: characters per minute of chapter duration.
    pub chars_per_minute: f64,
}

/// Compute per-chapter subtitle statistics.
#[must_use]
pub fn chapter_stats(
    track: &ChapterTrack,
    subtitles: &[Subtitle],
    index: &ChapterIndex,
) -> Vec<ChapterSubtitleStats> {
    track
        .iter()
        .enumerate()
        .map(|(cidx, chapter)| {
            let cue_indices = index.cues_in_chapter(cidx);
            let cue_count = cue_indices.len();

            let total_chars: usize = cue_indices
                .iter()
                .filter_map(|&i| subtitles.get(i))
                .map(|s| s.text.chars().count())
                .sum();

            let avg_duration_ms = if cue_count == 0 {
                0.0
            } else {
                let total_dur: i64 = cue_indices
                    .iter()
                    .filter_map(|&i| subtitles.get(i))
                    .map(|s| s.duration())
                    .sum();
                total_dur as f64 / cue_count as f64
            };

            let chapter_dur_min = chapter.duration_ms() as f64 / 60_000.0;
            let chars_per_minute = if chapter_dur_min > 0.0 {
                total_chars as f64 / chapter_dur_min
            } else {
                0.0
            };

            ChapterSubtitleStats {
                chapter_index: cidx,
                chapter_title: chapter.title.clone(),
                cue_count,
                total_chars,
                avg_duration_ms,
                chars_per_minute,
            }
        })
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Subtitle;

    fn make_track() -> ChapterTrack {
        ChapterTrack::from_vec(vec![
            Chapter::new(0, 60_000, "Intro"),
            Chapter::new(60_000, 180_000, "Act 1"),
            Chapter::new(180_000, 300_000, "Act 2"),
        ])
    }

    fn make_subs() -> Vec<Subtitle> {
        vec![
            Subtitle::new(5_000, 8_000, "Hello, intro!".to_string()),
            Subtitle::new(10_000, 12_000, "Still in intro.".to_string()),
            Subtitle::new(65_000, 68_000, "Act 1 begins.".to_string()),
            Subtitle::new(100_000, 102_000, "Middle of act 1.".to_string()),
            Subtitle::new(250_000, 253_000, "Act 2 scene.".to_string()),
            Subtitle::new(500_000, 503_000, "After all chapters.".to_string()),
        ]
    }

    #[test]
    fn test_chapter_at_binary_search() {
        let track = make_track();
        assert_eq!(track.chapter_at(0), Some(0));
        assert_eq!(track.chapter_at(30_000), Some(0));
        assert_eq!(track.chapter_at(60_000), Some(1));
        assert_eq!(track.chapter_at(179_999), Some(1));
        assert_eq!(track.chapter_at(180_000), Some(2));
        assert_eq!(track.chapter_at(500_000), None);
    }

    #[test]
    fn test_chapter_contains() {
        let ch = Chapter::new(60_000, 120_000, "Chapter");
        assert!(ch.contains(60_000));
        assert!(ch.contains(90_000));
        assert!(!ch.contains(120_000)); // exclusive end
        assert!(!ch.contains(59_999));
    }

    #[test]
    fn test_chapter_duration() {
        let ch = Chapter::new(0, 90_000, "Test");
        assert_eq!(ch.duration_ms(), 90_000);
    }

    #[test]
    fn test_track_total_span() {
        let track = make_track();
        assert_eq!(track.total_span_ms(), 300_000);
    }

    #[test]
    fn test_chapter_index_build() {
        let track = make_track();
        let subs = make_subs();
        let index = ChapterIndex::build(&track, &subs);

        // Two subs in intro
        assert_eq!(index.cues_in_chapter(0).len(), 2);
        // Two subs in act 1
        assert_eq!(index.cues_in_chapter(1).len(), 2);
        // One sub in act 2
        assert_eq!(index.cues_in_chapter(2).len(), 1);
        // One sub falls outside all chapters
        assert_eq!(index.unassigned_count(), 1);
    }

    #[test]
    fn test_chapter_of_cue() {
        let track = make_track();
        let subs = make_subs();
        let index = ChapterIndex::build(&track, &subs);

        assert_eq!(index.chapter_of_cue(0), Some(0)); // intro
        assert_eq!(index.chapter_of_cue(2), Some(1)); // act 1
        assert_eq!(index.chapter_of_cue(4), Some(2)); // act 2
        assert_eq!(index.chapter_of_cue(5), None); // after chapters
    }

    #[test]
    fn test_extract_chapter_subtitles_retime() {
        let track = make_track();
        let subs = make_subs();

        let intro_subs = extract_chapter_subtitles(&track, &subs, 0);
        // Intro = 0..60_000; subs at 5_000 and 10_000 → retimed to 5_000 and 10_000 (offset=0)
        assert_eq!(intro_subs.len(), 2);
        assert_eq!(intro_subs[0].start_time, 5_000);

        let act1_subs = extract_chapter_subtitles(&track, &subs, 1);
        // Act 1 = 60_000..180_000; subs at 65_000 and 100_000 → offset=60_000
        assert_eq!(act1_subs.len(), 2);
        assert_eq!(act1_subs[0].start_time, 5_000); // 65_000 - 60_000
    }

    #[test]
    fn test_chapter_track_add_maintains_order() {
        let mut track = ChapterTrack::new();
        track.add(Chapter::new(60_000, 120_000, "Second"));
        track.add(Chapter::new(0, 60_000, "First"));
        assert_eq!(track.get(0).map(|c| c.title.as_str()), Some("First"));
        assert_eq!(track.get(1).map(|c| c.title.as_str()), Some("Second"));
    }

    #[test]
    fn test_chapter_retime() {
        let track = ChapterTrack::from_vec(vec![Chapter::new(0, 60_000, "Ch1")]);
        let retimed = track.retime(2.0, 1_000);
        let ch = retimed.get(0).expect("chapter");
        assert_eq!(ch.start_ms, 1_000);
        assert_eq!(ch.end_ms, 121_000);
    }

    #[test]
    fn test_chapter_stats() {
        let track = make_track();
        let subs = make_subs();
        let index = ChapterIndex::build(&track, &subs);
        let stats = chapter_stats(&track, &subs, &index);

        assert_eq!(stats.len(), 3);
        let intro_stats = &stats[0];
        assert_eq!(intro_stats.chapter_title, "Intro");
        assert_eq!(intro_stats.cue_count, 2);
        assert!(intro_stats.chars_per_minute > 0.0);
    }

    #[test]
    fn test_empty_track_chapter_at() {
        let track = ChapterTrack::new();
        assert_eq!(track.chapter_at(5_000), None);
    }

    #[test]
    fn test_chapter_with_number_and_language() {
        let ch = Chapter::new(0, 1000, "Opening")
            .with_number(1)
            .with_language("en");
        assert_eq!(ch.number, Some(1));
        assert_eq!(ch.language.as_deref(), Some("en"));
    }

    #[test]
    fn test_non_empty_chapter_count() {
        let track = make_track();
        let subs = make_subs();
        let index = ChapterIndex::build(&track, &subs);
        // All 3 chapters have at least one sub.
        assert_eq!(index.non_empty_chapter_count(), 3);
    }

    #[test]
    fn test_extract_invalid_chapter_returns_empty() {
        let track = make_track();
        let subs = make_subs();
        let result = extract_chapter_subtitles(&track, &subs, 99);
        assert!(result.is_empty());
    }
}
