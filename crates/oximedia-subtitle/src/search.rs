//! Full-text search over subtitle cue collections.
//!
//! Provides case-insensitive substring search across parsed cue text,
//! returning the indices of matching cues so callers can retrieve or
//! highlight them without cloning the entire collection.
//!
//! # Example
//!
//! ```rust
//! use oximedia_subtitle::search::SubtitleSearcher;
//! use oximedia_subtitle::cue_parser::{CueEntry, CueTimestamp};
//!
//! let ts = CueTimestamp::new(0, 0, 0, 0);
//! let cues = vec![
//!     CueEntry::new(None, ts, ts, "Hello world".into()),
//!     CueEntry::new(None, ts, ts, "Goodbye".into()),
//!     CueEntry::new(None, ts, ts, "Hello again".into()),
//! ];
//!
//! let indices = SubtitleSearcher::find_containing(&cues, "hello");
//! assert_eq!(indices, vec![0, 2]);
//! ```

use crate::cue_parser::CueEntry;

/// Full-text searcher for subtitle cue collections.
pub struct SubtitleSearcher;

impl SubtitleSearcher {
    /// Return the indices of all cues whose text contains `query`
    /// (case-insensitive substring match).
    ///
    /// # Parameters
    ///
    /// - `cues`  : slice of parsed subtitle cues.
    /// - `query` : search string (case-insensitive).
    ///
    /// # Returns
    ///
    /// Sorted `Vec<usize>` of cue indices that contain the query.
    /// Returns an empty vector if no cues match or if `query` is empty.
    #[must_use]
    pub fn find_containing(cues: &[CueEntry], query: &str) -> Vec<usize> {
        if query.is_empty() {
            return Vec::new();
        }
        let query_lower = query.to_lowercase();
        cues.iter()
            .enumerate()
            .filter_map(|(i, cue)| {
                if cue.text.to_lowercase().contains(&query_lower) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return a reference to each matching cue alongside its original index.
    ///
    /// Useful when callers need both the position and the data.
    #[must_use]
    pub fn find_with_context<'a>(
        cues: &'a [CueEntry],
        query: &str,
    ) -> Vec<(usize, &'a CueEntry)> {
        if query.is_empty() {
            return Vec::new();
        }
        let query_lower = query.to_lowercase();
        cues.iter()
            .enumerate()
            .filter(|(_, cue)| cue.text.to_lowercase().contains(&query_lower))
            .collect()
    }

    /// Count how many cues contain the query string (case-insensitive).
    #[must_use]
    pub fn count_matches(cues: &[CueEntry], query: &str) -> usize {
        if query.is_empty() {
            return 0;
        }
        let query_lower = query.to_lowercase();
        cues.iter()
            .filter(|cue| cue.text.to_lowercase().contains(&query_lower))
            .count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cue_parser::CueTimestamp;

    fn ts() -> CueTimestamp {
        CueTimestamp::new(0, 0, 0, 0)
    }

    fn make_cue(text: &str) -> CueEntry {
        CueEntry::new(None, ts(), ts(), text.to_string())
    }

    fn sample_cues() -> Vec<CueEntry> {
        vec![
            make_cue("Hello world"),
            make_cue("Goodbye cruel world"),
            make_cue("HELLO again"),
            make_cue("Nothing interesting here"),
        ]
    }

    #[test]
    fn test_find_containing_case_insensitive() {
        let cues = sample_cues();
        let indices = SubtitleSearcher::find_containing(&cues, "hello");
        assert_eq!(indices, vec![0, 2], "should find 'Hello world' and 'HELLO again'");
    }

    #[test]
    fn test_find_containing_single_match() {
        let cues = sample_cues();
        let indices = SubtitleSearcher::find_containing(&cues, "goodbye");
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn test_find_containing_no_match() {
        let cues = sample_cues();
        let indices = SubtitleSearcher::find_containing(&cues, "xyz_not_present");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_find_containing_empty_query() {
        let cues = sample_cues();
        let indices = SubtitleSearcher::find_containing(&cues, "");
        assert!(indices.is_empty(), "empty query should return no results");
    }

    #[test]
    fn test_find_containing_empty_cues() {
        let indices = SubtitleSearcher::find_containing(&[], "hello");
        assert!(indices.is_empty());
    }

    #[test]
    fn test_find_with_context() {
        let cues = sample_cues();
        let results = SubtitleSearcher::find_with_context(&cues, "world");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
        assert_eq!(results[1].0, 1);
    }

    #[test]
    fn test_count_matches() {
        let cues = sample_cues();
        assert_eq!(SubtitleSearcher::count_matches(&cues, "world"), 2);
        assert_eq!(SubtitleSearcher::count_matches(&cues, "HELLO"), 2);
        assert_eq!(SubtitleSearcher::count_matches(&cues, "zzz"), 0);
        assert_eq!(SubtitleSearcher::count_matches(&cues, ""), 0);
    }

    #[test]
    fn test_partial_word_match() {
        let cues = vec![make_cue("International Space Station")];
        let indices = SubtitleSearcher::find_containing(&cues, "nat");
        assert_eq!(indices, vec![0], "should match partial word 'nat' in 'International'");
    }
}
