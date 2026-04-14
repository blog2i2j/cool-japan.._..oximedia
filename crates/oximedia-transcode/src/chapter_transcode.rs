//! Chapter-aware transcoding utilities.
//!
//! Provides data structures and algorithms for preserving, adjusting, and
//! serialising chapter metadata through a transcode pipeline.
//!
//! # Overview
//!
//! When a source file contains chapter markers (e.g. from an MKV or MP4
//! container) those chapters need to be adjusted whenever the output has a
//! different start time than the source — for example when a trim or cut
//! operation has been applied.  This module handles:
//!
//! - Representing individual [`ChapterInfo`] entries.
//! - Collecting them in a [`ChapterMap`] with builder-style helpers.
//! - Adjusting timestamps for a trimmed output via
//!   [`ChapterMap::adjust_for_trim`].
//! - Merging two [`ChapterMap`] instances (useful for concatenation).
//! - Exporting/importing a simple line-based text format.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::fmt;

use crate::{Result, TranscodeError};

// ─────────────────────────────────────────────────────────────────────────────
// ChapterInfo
// ─────────────────────────────────────────────────────────────────────────────

/// A single chapter entry with millisecond-precision timestamps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChapterInfo {
    /// Human-readable chapter title (may be empty).
    pub title: String,
    /// Chapter start time in milliseconds from the beginning of the stream.
    pub start_ms: u64,
    /// Chapter end time in milliseconds (exclusive).  `None` means "until the
    /// next chapter or the end of the stream".
    pub end_ms: Option<u64>,
    /// Optional language code (ISO 639-2, e.g. `"eng"`).
    pub language: Option<String>,
    /// Unique identifier for the chapter (e.g. a UUID string).
    pub uid: Option<String>,
}

impl ChapterInfo {
    /// Creates a new chapter with the given title and start time.
    #[must_use]
    pub fn new(title: impl Into<String>, start_ms: u64) -> Self {
        Self {
            title: title.into(),
            start_ms,
            end_ms: None,
            language: None,
            uid: None,
        }
    }

    /// Sets the end timestamp.
    #[must_use]
    pub fn with_end(mut self, end_ms: u64) -> Self {
        self.end_ms = Some(end_ms);
        self
    }

    /// Sets the language code.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Sets the UID string.
    #[must_use]
    pub fn with_uid(mut self, uid: impl Into<String>) -> Self {
        self.uid = Some(uid.into());
        self
    }

    /// Duration in milliseconds, if both `start_ms` and `end_ms` are known.
    #[must_use]
    pub fn duration_ms(&self) -> Option<u64> {
        self.end_ms.map(|e| e.saturating_sub(self.start_ms))
    }

    /// Returns `true` if this chapter starts after `offset_ms`.
    #[must_use]
    pub fn starts_after(&self, offset_ms: u64) -> bool {
        self.start_ms >= offset_ms
    }

    /// Returns `true` if any part of this chapter overlaps `[from, to)`.
    #[must_use]
    pub fn overlaps(&self, from_ms: u64, to_ms: u64) -> bool {
        let end = self.end_ms.unwrap_or(u64::MAX);
        self.start_ms < to_ms && end > from_ms
    }
}

impl fmt::Display for ChapterInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:>10} ms] {}", self.start_ms, self.title)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterParseError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur when parsing a chapter text file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChapterParseError {
    /// A `START=` line contained a non-integer value.
    InvalidTimestamp(String),
    /// A chapter block was missing its `START=` line.
    MissingStartTime,
    /// A chapter block was missing its `TITLE=` line.
    MissingTitle,
}

impl fmt::Display for ChapterParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimestamp(s) => write!(f, "invalid timestamp: {s}"),
            Self::MissingStartTime => write!(f, "chapter block missing START= line"),
            Self::MissingTitle => write!(f, "chapter block missing TITLE= line"),
        }
    }
}

impl std::error::Error for ChapterParseError {}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterMap
// ─────────────────────────────────────────────────────────────────────────────

/// An ordered collection of [`ChapterInfo`] entries associated with one stream.
#[derive(Debug, Clone, Default)]
pub struct ChapterMap {
    chapters: Vec<ChapterInfo>,
}

impl ChapterMap {
    /// Creates an empty map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            chapters: Vec::new(),
        }
    }

    /// Adds a chapter, keeping the list sorted by `start_ms`.
    pub fn add(&mut self, chapter: ChapterInfo) {
        self.chapters.push(chapter);
        self.chapters.sort_by_key(|c| c.start_ms);
    }

    /// Returns all chapters as a slice.
    #[must_use]
    pub fn chapters(&self) -> &[ChapterInfo] {
        &self.chapters
    }

    /// Returns the number of chapters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chapters.len()
    }

    /// Returns `true` if there are no chapters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chapters.is_empty()
    }

    /// Fills in `end_ms` for each chapter from the next chapter's `start_ms`.
    /// The last chapter's `end_ms` is set to `total_duration_ms` if provided.
    pub fn infer_end_times(&mut self, total_duration_ms: Option<u64>) {
        let n = self.chapters.len();
        for i in 0..n {
            if self.chapters[i].end_ms.is_none() {
                let end = if i + 1 < n {
                    Some(self.chapters[i + 1].start_ms)
                } else {
                    total_duration_ms
                };
                self.chapters[i].end_ms = end;
            }
        }
    }

    /// Returns a new [`ChapterMap`] with timestamps adjusted for a trim operation.
    ///
    /// `trim_start_ms` is the source position of the first output frame.
    /// `trim_end_ms` is the source position of the last output frame (exclusive).
    /// Chapters that fall entirely before `trim_start_ms` or entirely after
    /// `trim_end_ms` are dropped.  Timestamps are shifted left by `trim_start_ms`.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if `trim_end_ms <= trim_start_ms`.
    pub fn adjust_for_trim(
        &self,
        trim_start_ms: u64,
        trim_end_ms: u64,
    ) -> Result<ChapterMap> {
        if trim_end_ms <= trim_start_ms {
            return Err(TranscodeError::InvalidInput(format!(
                "trim_end_ms ({trim_end_ms}) must be greater than trim_start_ms ({trim_start_ms})"
            )));
        }

        let mut out = ChapterMap::new();
        for ch in &self.chapters {
            // Keep chapters that overlap the trim window.
            let ch_end = ch.end_ms.unwrap_or(u64::MAX);
            if ch.start_ms >= trim_end_ms || ch_end <= trim_start_ms {
                continue;
            }

            // Clamp start/end to the trim window and shift.
            let new_start = ch.start_ms.saturating_sub(trim_start_ms);
            let new_end = ch
                .end_ms
                .map(|e| e.min(trim_end_ms).saturating_sub(trim_start_ms));

            let mut adjusted = ChapterInfo {
                title: ch.title.clone(),
                start_ms: new_start,
                end_ms: new_end,
                language: ch.language.clone(),
                uid: ch.uid.clone(),
            };

            // If the chapter started before trim_start_ms, rename it to make
            // it clear that the beginning was cut.
            if ch.start_ms < trim_start_ms && adjusted.start_ms == 0 && !ch.title.is_empty() {
                adjusted.title = format!("{} (continued)", ch.title);
            }

            out.add(adjusted);
        }

        Ok(out)
    }

    /// Merges `other` into `self`, offsetting all timestamps in `other` by
    /// `offset_ms`.  This is used when concatenating two streams.
    pub fn merge_with_offset(&mut self, other: &ChapterMap, offset_ms: u64) {
        for ch in &other.chapters {
            let shifted = ChapterInfo {
                title: ch.title.clone(),
                start_ms: ch.start_ms + offset_ms,
                end_ms: ch.end_ms.map(|e| e + offset_ms),
                language: ch.language.clone(),
                uid: ch.uid.clone(),
            };
            self.add(shifted);
        }
    }

    /// Exports the chapter map to a simple line-based text format.
    ///
    /// Format (one chapter per block, blocks separated by blank lines):
    /// ```text
    /// START=<ms>
    /// END=<ms>           (optional)
    /// LANG=<code>        (optional)
    /// UID=<string>       (optional)
    /// TITLE=<text>
    /// ```
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for (i, ch) in self.chapters.iter().enumerate() {
            if i > 0 {
                out.push('\n');
            }
            out.push_str(&format!("START={}\n", ch.start_ms));
            if let Some(end) = ch.end_ms {
                out.push_str(&format!("END={end}\n"));
            }
            if let Some(lang) = &ch.language {
                out.push_str(&format!("LANG={lang}\n"));
            }
            if let Some(uid) = &ch.uid {
                out.push_str(&format!("UID={uid}\n"));
            }
            out.push_str(&format!("TITLE={}\n", ch.title));
        }
        out
    }

    /// Parses a chapter map from the text format produced by [`ChapterMap::to_text`].
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] wrapping a [`ChapterParseError`]
    /// description when a block is malformed.
    pub fn from_text(text: &str) -> Result<ChapterMap> {
        let mut map = ChapterMap::new();

        // Split into blocks separated by blank lines.
        let blocks: Vec<&str> = text
            .split("\n\n")
            .map(str::trim)
            .filter(|b| !b.is_empty())
            .collect();

        for block in blocks {
            let mut start_ms: Option<u64> = None;
            let mut end_ms: Option<u64> = None;
            let mut title: Option<String> = None;
            let mut language: Option<String> = None;
            let mut uid: Option<String> = None;

            for line in block.lines() {
                let line = line.trim();
                if let Some(val) = line.strip_prefix("START=") {
                    let ms = val.parse::<u64>().map_err(|_| {
                        TranscodeError::InvalidInput(
                            ChapterParseError::InvalidTimestamp(val.to_string()).to_string(),
                        )
                    })?;
                    start_ms = Some(ms);
                } else if let Some(val) = line.strip_prefix("END=") {
                    let ms = val.parse::<u64>().map_err(|_| {
                        TranscodeError::InvalidInput(
                            ChapterParseError::InvalidTimestamp(val.to_string()).to_string(),
                        )
                    })?;
                    end_ms = Some(ms);
                } else if let Some(val) = line.strip_prefix("LANG=") {
                    language = Some(val.to_string());
                } else if let Some(val) = line.strip_prefix("UID=") {
                    uid = Some(val.to_string());
                } else if let Some(val) = line.strip_prefix("TITLE=") {
                    title = Some(val.to_string());
                }
            }

            let start = start_ms.ok_or_else(|| {
                TranscodeError::InvalidInput(
                    ChapterParseError::MissingStartTime.to_string(),
                )
            })?;
            let t = title.ok_or_else(|| {
                TranscodeError::InvalidInput(ChapterParseError::MissingTitle.to_string())
            })?;

            let mut ch = ChapterInfo::new(t, start);
            ch.end_ms = end_ms;
            ch.language = language;
            ch.uid = uid;
            map.add(ch);
        }

        Ok(map)
    }

    /// Splits a single long chapter into two at `split_ms` (relative to the
    /// chapter map's timeline).  The chapter containing `split_ms` is replaced
    /// by two chapters; the second half gets `new_title`.
    ///
    /// Returns the number of chapters added (0 if `split_ms` does not fall
    /// inside any chapter).
    pub fn split_chapter_at(
        &mut self,
        split_ms: u64,
        new_title: impl Into<String>,
    ) -> usize {
        let new_title = new_title.into();
        let pos = self
            .chapters
            .iter()
            .position(|c| {
                let end = c.end_ms.unwrap_or(u64::MAX);
                c.start_ms < split_ms && end > split_ms
            });

        let Some(idx) = pos else {
            return 0;
        };

        let original = self.chapters.remove(idx);
        let first = ChapterInfo {
            title: original.title.clone(),
            start_ms: original.start_ms,
            end_ms: Some(split_ms),
            language: original.language.clone(),
            uid: original.uid.clone(),
        };
        let second = ChapterInfo {
            title: new_title,
            start_ms: split_ms,
            end_ms: original.end_ms,
            language: original.language,
            uid: None,
        };
        self.add(first);
        self.add(second);
        1
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ChapterTranscodeSpec
// ─────────────────────────────────────────────────────────────────────────────

/// How chapter metadata should be handled during transcoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChapterHandling {
    /// Drop all chapter metadata from the output.
    Drop,
    /// Copy chapters verbatim (no timestamp adjustment).
    Copy,
    /// Adjust chapter timestamps for a trimmed output.
    AdjustForTrim,
    /// Copy chapters and merge from a second source (for concat operations).
    MergeConcat,
}

/// Full specification for chapter handling in a transcode job.
#[derive(Debug, Clone)]
pub struct ChapterTranscodeSpec {
    /// Source chapter map loaded from the input container.
    pub source_chapters: ChapterMap,
    /// How to handle chapters in the output.
    pub handling: ChapterHandling,
    /// Trim start in milliseconds (used when `handling == AdjustForTrim`).
    pub trim_start_ms: Option<u64>,
    /// Trim end in milliseconds (used when `handling == AdjustForTrim`).
    pub trim_end_ms: Option<u64>,
    /// Optional additional chapters to merge (used when `handling == MergeConcat`).
    pub concat_chapters: Option<(ChapterMap, u64)>,
}

impl ChapterTranscodeSpec {
    /// Creates a spec from a source map, defaulting to `Copy` handling.
    #[must_use]
    pub fn new(source_chapters: ChapterMap) -> Self {
        Self {
            source_chapters,
            handling: ChapterHandling::Copy,
            trim_start_ms: None,
            trim_end_ms: None,
            concat_chapters: None,
        }
    }

    /// Overrides the handling mode.
    #[must_use]
    pub fn with_handling(mut self, handling: ChapterHandling) -> Self {
        self.handling = handling;
        self
    }

    /// Sets trim parameters (also sets handling to `AdjustForTrim`).
    #[must_use]
    pub fn with_trim(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.trim_start_ms = Some(start_ms);
        self.trim_end_ms = Some(end_ms);
        self.handling = ChapterHandling::AdjustForTrim;
        self
    }

    /// Sets concat merge parameters.
    #[must_use]
    pub fn with_concat(mut self, extra: ChapterMap, offset_ms: u64) -> Self {
        self.concat_chapters = Some((extra, offset_ms));
        self.handling = ChapterHandling::MergeConcat;
        self
    }

    /// Resolves the spec into an output [`ChapterMap`].
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] when `AdjustForTrim` is
    /// requested but `trim_start_ms` / `trim_end_ms` are not set, or when the
    /// trim range is invalid.
    pub fn resolve(&self) -> Result<ChapterMap> {
        match self.handling {
            ChapterHandling::Drop => Ok(ChapterMap::new()),

            ChapterHandling::Copy => Ok(self.source_chapters.clone()),

            ChapterHandling::AdjustForTrim => {
                let start = self.trim_start_ms.ok_or_else(|| {
                    TranscodeError::InvalidInput(
                        "AdjustForTrim requires trim_start_ms".to_string(),
                    )
                })?;
                let end = self.trim_end_ms.ok_or_else(|| {
                    TranscodeError::InvalidInput("AdjustForTrim requires trim_end_ms".to_string())
                })?;
                self.source_chapters.adjust_for_trim(start, end)
            }

            ChapterHandling::MergeConcat => {
                let mut merged = self.source_chapters.clone();
                if let Some((extra, offset)) = &self.concat_chapters {
                    merged.merge_with_offset(extra, *offset);
                }
                Ok(merged)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_map() -> ChapterMap {
        let mut m = ChapterMap::new();
        m.add(ChapterInfo::new("Intro", 0).with_end(30_000));
        m.add(ChapterInfo::new("Part 1", 30_000).with_end(90_000));
        m.add(ChapterInfo::new("Part 2", 90_000).with_end(150_000));
        m.add(ChapterInfo::new("Outro", 150_000).with_end(180_000));
        m
    }

    #[test]
    fn test_chapter_info_basics() {
        let ch = ChapterInfo::new("Intro", 0)
            .with_end(30_000)
            .with_language("eng")
            .with_uid("uid-001");

        assert_eq!(ch.title, "Intro");
        assert_eq!(ch.start_ms, 0);
        assert_eq!(ch.end_ms, Some(30_000));
        assert_eq!(ch.duration_ms(), Some(30_000));
        assert_eq!(ch.language.as_deref(), Some("eng"));
        assert_eq!(ch.uid.as_deref(), Some("uid-001"));
    }

    #[test]
    fn test_chapter_map_add_sorted() {
        let mut m = ChapterMap::new();
        m.add(ChapterInfo::new("B", 60_000));
        m.add(ChapterInfo::new("A", 0));
        m.add(ChapterInfo::new("C", 120_000));

        let titles: Vec<&str> = m.chapters().iter().map(|c| c.title.as_str()).collect();
        assert_eq!(titles, ["A", "B", "C"]);
    }

    #[test]
    fn test_infer_end_times() {
        let mut m = ChapterMap::new();
        m.add(ChapterInfo::new("Ch1", 0));
        m.add(ChapterInfo::new("Ch2", 60_000));
        m.infer_end_times(Some(120_000));

        assert_eq!(m.chapters()[0].end_ms, Some(60_000));
        assert_eq!(m.chapters()[1].end_ms, Some(120_000));
    }

    #[test]
    fn test_adjust_for_trim_basic() {
        let m = sample_map();
        // Trim from 30s to 120s — should keep Part 1 and Part 2, drop Intro and Outro.
        let trimmed = m.adjust_for_trim(30_000, 120_000).unwrap();

        assert_eq!(trimmed.len(), 2);
        assert_eq!(trimmed.chapters()[0].title, "Part 1");
        assert_eq!(trimmed.chapters()[0].start_ms, 0);
        assert_eq!(trimmed.chapters()[0].end_ms, Some(60_000));
        assert_eq!(trimmed.chapters()[1].title, "Part 2");
        assert_eq!(trimmed.chapters()[1].start_ms, 60_000);
        assert_eq!(trimmed.chapters()[1].end_ms, Some(90_000));
    }

    #[test]
    fn test_adjust_for_trim_partial_overlap() {
        let m = sample_map();
        // Trim starting in the middle of "Part 1".
        let trimmed = m.adjust_for_trim(60_000, 180_000).unwrap();

        // "Part 1" started before trim start — should be renamed "(continued)".
        assert!(trimmed.chapters()[0].title.contains("continued"));
        assert_eq!(trimmed.chapters()[0].start_ms, 0);
    }

    #[test]
    fn test_adjust_for_trim_invalid_range() {
        let m = sample_map();
        let err = m.adjust_for_trim(90_000, 30_000);
        assert!(err.is_err());
    }

    #[test]
    fn test_merge_with_offset() {
        let mut m1 = ChapterMap::new();
        m1.add(ChapterInfo::new("A", 0).with_end(60_000));

        let mut m2 = ChapterMap::new();
        m2.add(ChapterInfo::new("B", 0).with_end(60_000));

        m1.merge_with_offset(&m2, 60_000);

        assert_eq!(m1.len(), 2);
        assert_eq!(m1.chapters()[1].title, "B");
        assert_eq!(m1.chapters()[1].start_ms, 60_000);
        assert_eq!(m1.chapters()[1].end_ms, Some(120_000));
    }

    #[test]
    fn test_to_text_and_from_text_roundtrip() {
        let m = sample_map();
        let text = m.to_text();
        let parsed = ChapterMap::from_text(&text).unwrap();

        assert_eq!(parsed.len(), m.len());
        for (a, b) in m.chapters().iter().zip(parsed.chapters().iter()) {
            assert_eq!(a.title, b.title);
            assert_eq!(a.start_ms, b.start_ms);
            assert_eq!(a.end_ms, b.end_ms);
        }
    }

    #[test]
    fn test_from_text_missing_start_error() {
        let text = "TITLE=SomeChapter\n";
        let err = ChapterMap::from_text(text);
        assert!(err.is_err());
    }

    #[test]
    fn test_from_text_invalid_timestamp() {
        let text = "START=notanumber\nTITLE=Ch\n";
        let err = ChapterMap::from_text(text);
        assert!(err.is_err());
    }

    #[test]
    fn test_split_chapter_at() {
        let mut m = ChapterMap::new();
        m.add(ChapterInfo::new("Long Chapter", 0).with_end(120_000));

        let added = m.split_chapter_at(60_000, "Second Half");
        assert_eq!(added, 1);
        assert_eq!(m.len(), 2);
        assert_eq!(m.chapters()[0].end_ms, Some(60_000));
        assert_eq!(m.chapters()[1].title, "Second Half");
        assert_eq!(m.chapters()[1].start_ms, 60_000);
    }

    #[test]
    fn test_split_chapter_at_no_match() {
        let mut m = ChapterMap::new();
        m.add(ChapterInfo::new("Ch", 0).with_end(60_000));

        // split_ms is outside the chapter range
        let added = m.split_chapter_at(90_000, "New");
        assert_eq!(added, 0);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_spec_resolve_drop() {
        let spec = ChapterTranscodeSpec::new(sample_map())
            .with_handling(ChapterHandling::Drop);
        let out = spec.resolve().unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn test_spec_resolve_copy() {
        let m = sample_map();
        let spec = ChapterTranscodeSpec::new(m.clone())
            .with_handling(ChapterHandling::Copy);
        let out = spec.resolve().unwrap();
        assert_eq!(out.len(), m.len());
    }

    #[test]
    fn test_spec_resolve_adjust_for_trim() {
        let spec = ChapterTranscodeSpec::new(sample_map())
            .with_trim(30_000, 120_000);
        let out = spec.resolve().unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_spec_resolve_merge_concat() {
        let mut extra = ChapterMap::new();
        extra.add(ChapterInfo::new("Bonus", 0).with_end(30_000));

        let spec = ChapterTranscodeSpec::new(sample_map())
            .with_concat(extra, 180_000);
        let out = spec.resolve().unwrap();
        // 4 original + 1 extra
        assert_eq!(out.len(), 5);
        assert_eq!(out.chapters().last().map(|c| c.title.as_str()), Some("Bonus"));
    }

    #[test]
    fn test_chapter_info_overlaps() {
        let ch = ChapterInfo::new("Ch", 30_000).with_end(90_000);
        assert!(ch.overlaps(0, 60_000));
        assert!(ch.overlaps(60_000, 120_000));
        assert!(!ch.overlaps(0, 30_000));
        assert!(!ch.overlaps(90_000, 150_000));
    }

    #[test]
    fn test_chapter_display() {
        let ch = ChapterInfo::new("Intro", 1234);
        let s = ch.to_string();
        assert!(s.contains("Intro"));
        assert!(s.contains("1234"));
    }
}
