//! Matroska chapter support.
//!
//! Provides chapter handling for Matroska/WebM containers.

#![forbid(unsafe_code)]

use oximedia_core::{OxiError, OxiResult};
use std::collections::HashMap;

/// A chapter in a Matroska file.
#[derive(Debug, Clone)]
pub struct MatroskaChapter {
    /// Chapter UID (unique identifier).
    pub uid: u64,
    /// Start time in nanoseconds.
    pub start_time_ns: u64,
    /// End time in nanoseconds (optional).
    pub end_time_ns: Option<u64>,
    /// Chapter display strings (language -> title).
    pub displays: HashMap<String, String>,
    /// Whether this chapter is hidden.
    pub hidden: bool,
    /// Whether this chapter is enabled.
    pub enabled: bool,
    /// Nested chapters.
    pub children: Vec<MatroskaChapter>,
}

impl MatroskaChapter {
    /// Creates a new Matroska chapter.
    #[must_use]
    pub fn new(uid: u64, start_time_ns: u64) -> Self {
        Self {
            uid,
            start_time_ns,
            end_time_ns: None,
            displays: HashMap::new(),
            hidden: false,
            enabled: true,
            children: Vec::new(),
        }
    }

    /// Sets the end time.
    #[must_use]
    pub const fn with_end_time(mut self, end_time_ns: u64) -> Self {
        self.end_time_ns = Some(end_time_ns);
        self
    }

    /// Adds a display title.
    #[must_use]
    pub fn with_display(mut self, language: impl Into<String>, title: impl Into<String>) -> Self {
        self.displays.insert(language.into(), title.into());
        self
    }

    /// Sets the hidden flag.
    #[must_use]
    pub const fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Sets the enabled flag.
    #[must_use]
    pub const fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Adds a child chapter.
    pub fn add_child(&mut self, child: MatroskaChapter) {
        self.children.push(child);
    }

    /// Returns the title for a specific language.
    #[must_use]
    pub fn title(&self, language: &str) -> Option<&str> {
        self.displays.get(language).map(String::as_str)
    }

    /// Returns the default title (first available).
    #[must_use]
    pub fn default_title(&self) -> Option<&str> {
        self.displays.values().next().map(String::as_str)
    }

    /// Returns the duration in nanoseconds.
    #[must_use]
    pub fn duration_ns(&self) -> Option<u64> {
        self.end_time_ns.map(|end| end - self.start_time_ns)
    }

    /// Returns the start time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn start_time_secs(&self) -> f64 {
        self.start_time_ns as f64 / 1_000_000_000.0
    }

    /// Returns the end time in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn end_time_secs(&self) -> Option<f64> {
        self.end_time_ns.map(|end| end as f64 / 1_000_000_000.0)
    }
}

/// Edition (chapter group) in Matroska.
#[derive(Debug, Clone)]
pub struct MatroskaEdition {
    /// Edition UID.
    pub uid: u64,
    /// Whether this is the default edition.
    pub is_default: bool,
    /// Whether this is a hidden edition.
    pub hidden: bool,
    /// Whether this is an ordered edition.
    pub ordered: bool,
    /// Chapters in this edition.
    pub chapters: Vec<MatroskaChapter>,
}

impl MatroskaEdition {
    /// Creates a new Matroska edition.
    #[must_use]
    pub const fn new(uid: u64) -> Self {
        Self {
            uid,
            is_default: false,
            hidden: false,
            ordered: false,
            chapters: Vec::new(),
        }
    }

    /// Sets the default flag.
    #[must_use]
    pub const fn with_default(mut self, is_default: bool) -> Self {
        self.is_default = is_default;
        self
    }

    /// Sets the hidden flag.
    #[must_use]
    pub const fn with_hidden(mut self, hidden: bool) -> Self {
        self.hidden = hidden;
        self
    }

    /// Sets the ordered flag.
    #[must_use]
    pub const fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Adds a chapter to this edition.
    pub fn add_chapter(&mut self, chapter: MatroskaChapter) {
        self.chapters.push(chapter);
    }

    /// Returns the number of chapters.
    #[must_use]
    pub fn chapter_count(&self) -> usize {
        self.chapters.len()
    }

    /// Finds a chapter at a specific time.
    #[must_use]
    pub fn chapter_at_time(&self, time_ns: u64) -> Option<&MatroskaChapter> {
        self.chapters.iter().find(|ch| {
            ch.start_time_ns <= time_ns && ch.end_time_ns.map_or(true, |end| time_ns < end)
        })
    }
}

/// Container for Matroska chapters.
#[derive(Debug, Clone)]
pub struct MatroskaChapters {
    editions: Vec<MatroskaEdition>,
}

impl MatroskaChapters {
    /// Creates a new Matroska chapters container.
    #[must_use]
    pub fn new() -> Self {
        Self {
            editions: Vec::new(),
        }
    }

    /// Adds an edition.
    pub fn add_edition(&mut self, edition: MatroskaEdition) {
        self.editions.push(edition);
    }

    /// Returns all editions.
    #[must_use]
    pub fn editions(&self) -> &[MatroskaEdition] {
        &self.editions
    }

    /// Returns the default edition.
    #[must_use]
    pub fn default_edition(&self) -> Option<&MatroskaEdition> {
        self.editions
            .iter()
            .find(|e| e.is_default)
            .or_else(|| self.editions.first())
    }

    /// Returns the total number of chapters across all editions.
    #[must_use]
    pub fn total_chapter_count(&self) -> usize {
        self.editions
            .iter()
            .map(MatroskaEdition::chapter_count)
            .sum()
    }

    /// Validates the chapter structure.
    ///
    /// # Errors
    ///
    /// Returns `Err` if duplicate edition UIDs or chapter UIDs are found.
    pub fn validate(&self) -> OxiResult<()> {
        // Check for duplicate UIDs
        let mut edition_uids = std::collections::HashSet::new();

        for edition in &self.editions {
            if !edition_uids.insert(edition.uid) {
                return Err(OxiError::InvalidData(format!(
                    "Duplicate edition UID: {}",
                    edition.uid
                )));
            }

            let mut chapter_uids = std::collections::HashSet::new();
            for chapter in &edition.chapters {
                if !chapter_uids.insert(chapter.uid) {
                    return Err(OxiError::InvalidData(format!(
                        "Duplicate chapter UID: {}",
                        chapter.uid
                    )));
                }
            }
        }

        Ok(())
    }
}

impl Default for MatroskaChapters {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for Matroska chapters.
pub struct MatroskaChaptersBuilder {
    chapters: MatroskaChapters,
    next_edition_uid: u64,
    next_chapter_uid: u64,
}

impl MatroskaChaptersBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chapters: MatroskaChapters::new(),
            next_edition_uid: 1,
            next_chapter_uid: 1,
        }
    }

    /// Adds a simple chapter.
    pub fn add_chapter(&mut self, start_ns: u64, title: &str) -> &mut Self {
        self.add_chapter_with_language(start_ns, title, "eng")
    }

    /// Adds a chapter with a specific language.
    pub fn add_chapter_with_language(
        &mut self,
        start_ns: u64,
        title: &str,
        language: &str,
    ) -> &mut Self {
        if self.chapters.editions.is_empty() {
            let edition = MatroskaEdition::new(self.next_edition_uid).with_default(true);
            self.next_edition_uid += 1;
            self.chapters.add_edition(edition);
        }

        let chapter =
            MatroskaChapter::new(self.next_chapter_uid, start_ns).with_display(language, title);
        self.next_chapter_uid += 1;

        if let Some(edition) = self.chapters.editions.last_mut() {
            edition.add_chapter(chapter);
        }

        self
    }

    /// Adds a chapter with start and end times.
    pub fn add_chapter_with_end(&mut self, start_ns: u64, end_ns: u64, title: &str) -> &mut Self {
        self.add_chapter(start_ns, title);

        if let Some(edition) = self.chapters.editions.last_mut() {
            if let Some(chapter) = edition.chapters.last_mut() {
                chapter.end_time_ns = Some(end_ns);
            }
        }

        self
    }

    /// Builds the chapters.
    #[must_use]
    pub fn build(self) -> MatroskaChapters {
        self.chapters
    }
}

impl Default for MatroskaChaptersBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matroska_chapter() {
        let chapter = MatroskaChapter::new(1, 0)
            .with_end_time(5_000_000_000)
            .with_display("eng", "Chapter 1")
            .with_display("jpn", "チャプター1")
            .with_enabled(true);

        assert_eq!(chapter.uid, 1);
        assert_eq!(chapter.start_time_ns, 0);
        assert_eq!(chapter.end_time_ns, Some(5_000_000_000));
        assert_eq!(chapter.title("eng"), Some("Chapter 1"));
        assert_eq!(chapter.duration_ns(), Some(5_000_000_000));
        assert!((chapter.start_time_secs() - 0.0).abs() < 0.001);
        assert!((chapter.end_time_secs().expect("operation should succeed") - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_matroska_edition() {
        let mut edition = MatroskaEdition::new(1)
            .with_default(true)
            .with_ordered(false);

        let ch1 = MatroskaChapter::new(1, 0).with_display("eng", "Intro");
        let ch2 = MatroskaChapter::new(2, 5_000_000_000).with_display("eng", "Main");

        edition.add_chapter(ch1);
        edition.add_chapter(ch2);

        assert_eq!(edition.chapter_count(), 2);
        assert!(edition.is_default);

        let chapter = edition.chapter_at_time(3_000_000_000);
        assert!(chapter.is_some());
        assert_eq!(chapter.expect("operation should succeed").uid, 1);
    }

    #[test]
    fn test_matroska_chapters() {
        let mut chapters = MatroskaChapters::new();

        let mut edition = MatroskaEdition::new(1).with_default(true);
        edition.add_chapter(MatroskaChapter::new(1, 0).with_display("eng", "Chapter 1"));
        chapters.add_edition(edition);

        assert_eq!(chapters.total_chapter_count(), 1);

        let default = chapters.default_edition();
        assert!(default.is_some());
        assert!(default.expect("operation should succeed").is_default);
    }

    #[test]
    fn test_chapters_builder() {
        let mut builder = MatroskaChaptersBuilder::new();
        builder.add_chapter(0, "Intro");
        builder.add_chapter(5_000_000_000, "Main Content");
        builder.add_chapter_with_end(10_000_000_000, 15_000_000_000, "Credits");

        let chapters = builder.build();
        assert_eq!(chapters.total_chapter_count(), 3);

        let edition = chapters
            .default_edition()
            .expect("operation should succeed");
        assert_eq!(edition.chapters[2].duration_ns(), Some(5_000_000_000));
    }

    #[test]
    fn test_validate_chapters() {
        let mut chapters = MatroskaChapters::new();
        let edition = MatroskaEdition::new(1);
        chapters.add_edition(edition);

        assert!(chapters.validate().is_ok());

        // Duplicate edition UID
        chapters.add_edition(MatroskaEdition::new(1));
        assert!(chapters.validate().is_err());
    }
}
