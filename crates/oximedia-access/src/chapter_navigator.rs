#![allow(dead_code)]
//! Chapter navigation for accessibility.
//!
//! This module provides chapter-aware navigation for media content, enabling
//! screen readers and other assistive technologies to present a structured table
//! of contents, jump between chapters, and receive human-readable summaries of
//! each chapter's content.
//!
//! # Design
//!
//! A [`ChapterList`] is the central store.  Each [`Chapter`] carries:
//! - A unique numeric ID
//! - A start time in milliseconds
//! - An optional end time (inferred from the next chapter if omitted)
//! - A human-readable title used for AT announcement
//! - An optional accessible title for screen readers that differs from the
//!   visible title (e.g. a richer description)
//! - An optional textual summary used to preview content before navigation
//! - An optional content-type hint (see [`ChapterContentType`])
//!
//! The [`ChapterNavigator`] wraps a [`ChapterList`] and tracks a "current"
//! position so callers can implement prev/next navigation without storing
//! state themselves.

use std::fmt;

use crate::error::{AccessError, AccessResult};

// ---------------------------------------------------------------------------
// Content type hint
// ---------------------------------------------------------------------------

/// High-level description of what kind of content a chapter contains.
///
/// This hint lets TTS engines and screen readers adapt their announcement
/// strategy (e.g. warn before graphic scenes, annotate music-only sections).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChapterContentType {
    /// Standard narrative/dialogue content.
    Narrative,
    /// Primarily musical content with little or no dialogue.
    Music,
    /// Credits roll.
    Credits,
    /// Introductory material (opening titles, recap, etc.).
    Introduction,
    /// Content that may not be suitable for all audiences.
    ContentWarning,
    /// Action or high-intensity visual sequence.
    ActionSequence,
    /// Interview, documentary or talking-head segment.
    Interview,
    /// Silent or very sparse audio; important for hard-of-hearing users.
    Silent,
    /// Animated content.
    Animation,
    /// User-defined / catch-all.
    Other,
}

impl fmt::Display for ChapterContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Narrative => "Narrative",
            Self::Music => "Music",
            Self::Credits => "Credits",
            Self::Introduction => "Introduction",
            Self::ContentWarning => "Content Warning",
            Self::ActionSequence => "Action Sequence",
            Self::Interview => "Interview",
            Self::Silent => "Silent",
            Self::Animation => "Animation",
            Self::Other => "Other",
        };
        write!(f, "{label}")
    }
}

// ---------------------------------------------------------------------------
// Chapter
// ---------------------------------------------------------------------------

/// A single chapter within a media item.
#[derive(Debug, Clone)]
pub struct Chapter {
    /// Unique, stable identifier for this chapter.
    pub id: u64,
    /// Chapter start time in milliseconds from the beginning of the media.
    pub start_ms: u64,
    /// Chapter end time in milliseconds, if explicitly known.
    ///
    /// When `None`, the end is considered to be the start of the next chapter
    /// (or the total duration of the media for the final chapter).
    pub end_ms: Option<u64>,
    /// The primary title shown in menus and read by screen readers.
    pub title: String,
    /// An alternative, richer title for screen readers.
    ///
    /// When `None`, `title` is used for AT announcement as well.
    pub accessible_title: Option<String>,
    /// A short paragraph summarising the chapter's content.
    pub summary: Option<String>,
    /// High-level content classification.
    pub content_type: ChapterContentType,
    /// Chapter index within its parent list (0-based, assigned when added).
    pub(crate) index: usize,
}

impl Chapter {
    /// Create a minimal chapter.
    #[must_use]
    pub fn new(id: u64, start_ms: u64, title: impl Into<String>) -> Self {
        Self {
            id,
            start_ms,
            end_ms: None,
            title: title.into(),
            accessible_title: None,
            summary: None,
            content_type: ChapterContentType::Narrative,
            index: 0,
        }
    }

    /// Set an explicit end time.
    #[must_use]
    pub fn with_end(mut self, end_ms: u64) -> Self {
        self.end_ms = Some(end_ms);
        self
    }

    /// Set the accessible title.
    #[must_use]
    pub fn with_accessible_title(mut self, title: impl Into<String>) -> Self {
        self.accessible_title = Some(title.into());
        self
    }

    /// Set the chapter summary.
    #[must_use]
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.summary = Some(summary.into());
        self
    }

    /// Set the content type.
    #[must_use]
    pub fn with_content_type(mut self, ct: ChapterContentType) -> Self {
        self.content_type = ct;
        self
    }

    /// Return the title the assistive technology should announce.
    ///
    /// Falls back to `title` when no accessible title is set.
    #[must_use]
    pub fn announcement_title(&self) -> &str {
        self.accessible_title.as_deref().unwrap_or(&self.title)
    }

    /// Compute the duration in milliseconds given the total media duration.
    ///
    /// When an explicit `end_ms` is stored the duration is `end_ms - start_ms`.
    /// Otherwise `total_duration_ms` is used as the upper bound.
    #[must_use]
    pub fn duration_ms(&self, total_duration_ms: u64) -> u64 {
        let end = self.end_ms.unwrap_or(total_duration_ms);
        end.saturating_sub(self.start_ms)
    }

    /// Check whether the given timestamp falls inside this chapter.
    ///
    /// When `end_ms` is not set the chapter is treated as a point-in-time
    /// boundary so only `start_ms` == `time_ms` matches.
    #[must_use]
    pub fn contains_time(&self, time_ms: u64, total_duration_ms: u64) -> bool {
        let end = self.end_ms.unwrap_or(total_duration_ms);
        time_ms >= self.start_ms && time_ms < end
    }
}

impl fmt::Display for Chapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {} ({}ms)", self.index + 1, self.title, self.start_ms)
    }
}

// ---------------------------------------------------------------------------
// ChapterList
// ---------------------------------------------------------------------------

/// An ordered collection of chapters for a single media item.
///
/// Chapters are always kept sorted by `start_ms`.
#[derive(Debug, Default)]
pub struct ChapterList {
    /// Total media duration in milliseconds.
    total_duration_ms: u64,
    /// Chapters in ascending start-time order.
    chapters: Vec<Chapter>,
    /// Counter used to assign unique IDs.
    next_id: u64,
}

impl ChapterList {
    /// Create a new, empty chapter list for a media item of the given duration.
    #[must_use]
    pub fn new(total_duration_ms: u64) -> Self {
        Self {
            total_duration_ms,
            chapters: Vec::new(),
            next_id: 1,
        }
    }

    /// Total media duration this list was built for.
    #[must_use]
    pub fn total_duration_ms(&self) -> u64 {
        self.total_duration_ms
    }

    /// Add a chapter.  Returns an error when `start_ms` is beyond
    /// `total_duration_ms`.
    ///
    /// The internal list is re-sorted after each insertion and chapter indices
    /// are refreshed.
    pub fn add(&mut self, start_ms: u64, title: impl Into<String>) -> AccessResult<u64> {
        if start_ms > self.total_duration_ms {
            return Err(AccessError::InvalidTiming(format!(
                "chapter start {start_ms}ms exceeds media duration {}ms",
                self.total_duration_ms
            )));
        }
        let id = self.next_id;
        self.next_id += 1;
        let chapter = Chapter::new(id, start_ms, title);
        self.chapters.push(chapter);
        self.reindex();
        Ok(id)
    }

    /// Add a chapter with full builder-style configuration.
    pub fn add_chapter(&mut self, chapter: Chapter) -> AccessResult<()> {
        if chapter.start_ms > self.total_duration_ms {
            return Err(AccessError::InvalidTiming(format!(
                "chapter start {}ms exceeds media duration {}ms",
                chapter.start_ms, self.total_duration_ms
            )));
        }
        self.chapters.push(chapter);
        self.reindex();
        Ok(())
    }

    /// Remove a chapter by ID.  Returns `true` if a chapter was removed.
    pub fn remove(&mut self, id: u64) -> bool {
        let before = self.chapters.len();
        self.chapters.retain(|c| c.id != id);
        if self.chapters.len() < before {
            self.reindex();
            true
        } else {
            false
        }
    }

    /// Get a chapter by ID.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Chapter> {
        self.chapters.iter().find(|c| c.id == id)
    }

    /// Get a chapter by 0-based index.
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<&Chapter> {
        self.chapters.get(index)
    }

    /// Mutable access to a chapter by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Chapter> {
        self.chapters.iter_mut().find(|c| c.id == id)
    }

    /// Number of chapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    /// Returns `true` when no chapters have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Return all chapters in order.
    #[must_use]
    pub fn all(&self) -> &[Chapter] {
        &self.chapters
    }

    /// Find the chapter that contains a given timestamp.
    #[must_use]
    pub fn chapter_at(&self, time_ms: u64) -> Option<&Chapter> {
        // Infer end times: a chapter spans from its start to the next chapter's
        // start (or the total duration for the last chapter).
        for (i, chapter) in self.chapters.iter().enumerate() {
            let end = if let Some(explicit_end) = chapter.end_ms {
                explicit_end
            } else if let Some(next) = self.chapters.get(i + 1) {
                next.start_ms
            } else {
                self.total_duration_ms
            };
            if time_ms >= chapter.start_ms && time_ms < end {
                return Some(chapter);
            }
        }
        None
    }

    /// Build a table-of-contents representation suitable for rendering in a
    /// UI or announcing via a screen reader.
    #[must_use]
    pub fn table_of_contents(&self) -> Vec<TocEntry> {
        self.chapters
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let inferred_end = if let Some(explicit_end) = c.end_ms {
                    explicit_end
                } else if let Some(next) = self.chapters.get(i + 1) {
                    next.start_ms
                } else {
                    self.total_duration_ms
                };
                TocEntry {
                    id: c.id,
                    index: i,
                    title: c.title.clone(),
                    accessible_title: c.accessible_title.clone(),
                    start_ms: c.start_ms,
                    end_ms: inferred_end,
                    duration_ms: inferred_end.saturating_sub(c.start_ms),
                    has_summary: c.summary.is_some(),
                    content_type: c.content_type,
                }
            })
            .collect()
    }

    /// Generate a plain-text summary of all chapters.
    ///
    /// Each line lists: chapter number, title, start time (H:MM:SS), and
    /// optional content type annotation.
    #[must_use]
    pub fn generate_summary(&self) -> String {
        let mut lines: Vec<String> = Vec::with_capacity(self.chapters.len());
        for entry in self.table_of_contents() {
            let time_str = format_ms(entry.start_ms);
            let ct_note = if entry.content_type != ChapterContentType::Narrative {
                format!(" [{}]", entry.content_type)
            } else {
                String::new()
            };
            lines.push(format!(
                "{}. {}{} — {}",
                entry.index + 1,
                entry.accessible_title.as_deref().unwrap_or(&entry.title),
                ct_note,
                time_str,
            ));
        }
        lines.join("\n")
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Re-sort chapters by start time and refresh 0-based indices.
    fn reindex(&mut self) {
        self.chapters.sort_by_key(|c| c.start_ms);
        for (i, c) in self.chapters.iter_mut().enumerate() {
            c.index = i;
        }
    }
}

// ---------------------------------------------------------------------------
// TocEntry
// ---------------------------------------------------------------------------

/// A lightweight, serialization-friendly representation of one chapter in the
/// table of contents.
#[derive(Debug, Clone)]
pub struct TocEntry {
    /// Chapter ID.
    pub id: u64,
    /// 0-based position within the chapter list.
    pub index: usize,
    /// Primary title.
    pub title: String,
    /// Accessible title (when different from `title`).
    pub accessible_title: Option<String>,
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// Effective end time in milliseconds (inferred from the next chapter or
    /// the total duration when `end_ms` is not set on the source chapter).
    pub end_ms: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// `true` when the underlying chapter carries a textual summary.
    pub has_summary: bool,
    /// Content type classification.
    pub content_type: ChapterContentType,
}

impl fmt::Display for TocEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} ({}–{})",
            self.index + 1,
            self.title,
            format_ms(self.start_ms),
            format_ms(self.end_ms),
        )
    }
}

// ---------------------------------------------------------------------------
// ChapterNavigator
// ---------------------------------------------------------------------------

/// Stateful wrapper around a [`ChapterList`] that tracks the currently active
/// chapter and provides prev/next navigation.
///
/// The current position is expressed both as a chapter index and as a
/// millisecond timestamp so callers can synchronise a media player.
#[derive(Debug)]
pub struct ChapterNavigator {
    list: ChapterList,
    /// 0-based index of the currently selected chapter, or `None` when no
    /// chapter is selected.
    current_index: Option<usize>,
}

impl ChapterNavigator {
    /// Create a navigator from an existing chapter list.
    #[must_use]
    pub fn new(list: ChapterList) -> Self {
        Self {
            list,
            current_index: None,
        }
    }

    /// Reference to the underlying chapter list.
    #[must_use]
    pub fn chapter_list(&self) -> &ChapterList {
        &self.list
    }

    /// Currently selected chapter, if any.
    #[must_use]
    pub fn current(&self) -> Option<&Chapter> {
        self.current_index
            .and_then(|i| self.list.get_by_index(i))
    }

    /// Move to the chapter following the current one.
    ///
    /// Returns the chapter now selected, or `None` when already at the last
    /// chapter.
    pub fn next(&mut self) -> Option<&Chapter> {
        let next_index = match self.current_index {
            None => {
                if self.list.is_empty() {
                    return None;
                }
                0
            }
            Some(i) => {
                let n = i + 1;
                if n >= self.list.len() {
                    return None;
                }
                n
            }
        };
        self.current_index = Some(next_index);
        self.list.get_by_index(next_index)
    }

    /// Move to the chapter preceding the current one.
    ///
    /// Returns the chapter now selected, or `None` when already at the first
    /// chapter.
    pub fn previous(&mut self) -> Option<&Chapter> {
        let prev_index = match self.current_index {
            None | Some(0) => return None,
            Some(i) => i - 1,
        };
        self.current_index = Some(prev_index);
        self.list.get_by_index(prev_index)
    }

    /// Jump directly to the chapter at a given 0-based index.
    ///
    /// Returns an error when the index is out of range.
    pub fn go_to_index(&mut self, index: usize) -> AccessResult<&Chapter> {
        if index >= self.list.len() {
            return Err(AccessError::InvalidTiming(format!(
                "chapter index {index} is out of range (0..{})",
                self.list.len()
            )));
        }
        self.current_index = Some(index);
        self.list.get_by_index(index).ok_or_else(|| {
            AccessError::Other(format!("chapter at index {index} unexpectedly missing"))
        })
    }

    /// Seek to the chapter that contains `time_ms` and select it.
    ///
    /// Returns `None` when no chapter covers the timestamp (e.g. the timestamp
    /// is beyond the total duration).
    pub fn seek_to_time(&mut self, time_ms: u64) -> Option<&Chapter> {
        let chapter = self.list.chapter_at(time_ms)?;
        self.current_index = Some(chapter.index);
        self.list.get_by_index(chapter.index)
    }

    /// Jump to the first chapter.
    pub fn go_to_first(&mut self) -> Option<&Chapter> {
        if self.list.is_empty() {
            return None;
        }
        self.current_index = Some(0);
        self.list.get_by_index(0)
    }

    /// Jump to the last chapter.
    pub fn go_to_last(&mut self) -> Option<&Chapter> {
        let last = self.list.len().checked_sub(1)?;
        self.current_index = Some(last);
        self.list.get_by_index(last)
    }

    /// Check whether a next chapter exists beyond the current position.
    #[must_use]
    pub fn has_next(&self) -> bool {
        match self.current_index {
            None => !self.list.is_empty(),
            Some(i) => i + 1 < self.list.len(),
        }
    }

    /// Check whether a previous chapter exists before the current position.
    #[must_use]
    pub fn has_previous(&self) -> bool {
        matches!(self.current_index, Some(i) if i > 0)
    }

    /// Generate the table of contents from the underlying list.
    #[must_use]
    pub fn table_of_contents(&self) -> Vec<TocEntry> {
        self.list.table_of_contents()
    }

    /// Delegate to [`ChapterList::generate_summary`].
    #[must_use]
    pub fn generate_summary(&self) -> String {
        self.list.generate_summary()
    }

    /// Return the start timestamp of the currently selected chapter, suitable
    /// for seeking a media player.
    #[must_use]
    pub fn current_start_ms(&self) -> Option<u64> {
        self.current().map(|c| c.start_ms)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a millisecond timestamp as `H:MM:SS`.
fn format_ms(ms: u64) -> String {
    let total_secs = ms / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    format!("{h}:{m:02}:{s:02}")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_list() -> ChapterList {
        let mut list = ChapterList::new(90_000);
        list.add(0, "Introduction").expect("add failed");
        list.add(15_000, "Act One").expect("add failed");
        list.add(45_000, "Act Two").expect("add failed");
        list.add(75_000, "Credits").expect("add failed");
        list
    }

    #[test]
    fn test_add_and_retrieve() {
        let mut list = ChapterList::new(60_000);
        let id = list.add(0, "Intro").expect("add ok");
        assert_eq!(list.len(), 1);
        let ch = list.get(id).expect("chapter present");
        assert_eq!(ch.title, "Intro");
        assert_eq!(ch.start_ms, 0);
    }

    #[test]
    fn test_chapters_sorted_on_insert() {
        let mut list = ChapterList::new(120_000);
        list.add(60_000, "Middle").expect("ok");
        list.add(0, "Start").expect("ok");
        list.add(90_000, "End").expect("ok");
        let all = list.all();
        assert_eq!(all[0].start_ms, 0);
        assert_eq!(all[1].start_ms, 60_000);
        assert_eq!(all[2].start_ms, 90_000);
    }

    #[test]
    fn test_add_out_of_bounds_error() {
        let mut list = ChapterList::new(30_000);
        let result = list.add(50_000, "Too Late");
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_chapter() {
        let mut list = sample_list();
        assert_eq!(list.len(), 4);
        let id = list.all()[1].id;
        assert!(list.remove(id));
        assert_eq!(list.len(), 3);
        assert!(list.get(id).is_none());
    }

    #[test]
    fn test_chapter_at_time() {
        let list = sample_list();
        // 0–15_000: Introduction
        let ch = list.chapter_at(5_000).expect("inside intro");
        assert_eq!(ch.title, "Introduction");
        // 15_000–45_000: Act One
        let ch = list.chapter_at(30_000).expect("inside act one");
        assert_eq!(ch.title, "Act One");
        // Beyond last chapter: None
        assert!(list.chapter_at(100_000).is_none());
    }

    #[test]
    fn test_table_of_contents_count() {
        let list = sample_list();
        let toc = list.table_of_contents();
        assert_eq!(toc.len(), 4);
        // Duration of first chapter should be 15_000ms
        assert_eq!(toc[0].duration_ms, 15_000);
    }

    #[test]
    fn test_generate_summary() {
        let list = sample_list();
        let summary = list.generate_summary();
        assert!(summary.contains("Introduction"));
        assert!(summary.contains("Act One"));
        assert!(summary.contains("Credits"));
        // Should contain formatted timestamps
        assert!(summary.contains("0:00:00"));
    }

    #[test]
    fn test_chapter_accessible_title_fallback() {
        let ch = Chapter::new(1, 0, "Chapter 1");
        assert_eq!(ch.announcement_title(), "Chapter 1");

        let ch_with_at = Chapter::new(2, 0, "Chapter 1")
            .with_accessible_title("Chapter 1: The Beginning");
        assert_eq!(ch_with_at.announcement_title(), "Chapter 1: The Beginning");
    }

    #[test]
    fn test_navigator_next_prev() {
        let list = sample_list();
        let mut nav = ChapterNavigator::new(list);

        assert!(nav.current().is_none());
        assert!(nav.has_next());
        assert!(!nav.has_previous());

        let ch = nav.next().expect("first chapter");
        assert_eq!(ch.title, "Introduction");

        let ch = nav.next().expect("second");
        assert_eq!(ch.title, "Act One");

        let ch = nav.previous().expect("back to first");
        assert_eq!(ch.title, "Introduction");

        assert!(nav.previous().is_none()); // already at first
    }

    #[test]
    fn test_navigator_seek_to_time() {
        let list = sample_list();
        let mut nav = ChapterNavigator::new(list);

        let ch = nav.seek_to_time(20_000).expect("inside Act One");
        assert_eq!(ch.title, "Act One");
        assert_eq!(nav.current_start_ms(), Some(15_000));
    }

    #[test]
    fn test_navigator_go_to_index_out_of_range() {
        let list = sample_list();
        let mut nav = ChapterNavigator::new(list);
        assert!(nav.go_to_index(99).is_err());
    }

    #[test]
    fn test_navigator_first_last() {
        let list = sample_list();
        let mut nav = ChapterNavigator::new(list);

        let last = nav.go_to_last().expect("last chapter");
        assert_eq!(last.title, "Credits");

        let first = nav.go_to_first().expect("first chapter");
        assert_eq!(first.title, "Introduction");
    }

    #[test]
    fn test_content_type_display() {
        assert_eq!(ChapterContentType::ContentWarning.to_string(), "Content Warning");
        assert_eq!(ChapterContentType::ActionSequence.to_string(), "Action Sequence");
        assert_eq!(ChapterContentType::Music.to_string(), "Music");
    }

    #[test]
    fn test_toc_entry_display() {
        let list = sample_list();
        let toc = list.table_of_contents();
        let display = toc[0].to_string();
        assert!(display.contains("Introduction"));
        assert!(display.contains("0:00:00"));
    }

    #[test]
    fn test_format_ms_helper() {
        assert_eq!(format_ms(0), "0:00:00");
        assert_eq!(format_ms(60_000), "0:01:00");
        assert_eq!(format_ms(3_661_000), "1:01:01");
    }

    #[test]
    fn test_chapter_duration_and_contains() {
        let ch = Chapter::new(1, 10_000, "Test").with_end(20_000);
        assert_eq!(ch.duration_ms(90_000), 10_000);
        assert!(ch.contains_time(15_000, 90_000));
        assert!(!ch.contains_time(5_000, 90_000));
        assert!(!ch.contains_time(20_000, 90_000)); // end is exclusive
    }
}
