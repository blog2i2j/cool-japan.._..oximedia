//! Lyrics timing alignment stub.
//!
//! Given a lyrics string and a list of onset times (in milliseconds), assigns
//! each word in the lyrics to the nearest available onset using a greedy
//! left-to-right matching strategy.
//!
//! # Algorithm
//!
//! 1. Split the lyrics into words by whitespace.  Empty tokens are discarded.
//! 2. Iterate over words in order, consuming onsets greedily:
//!    - The first word is assigned to the first onset.
//!    - Each subsequent word is assigned to the onset closest to the previous
//!      word's onset that has not yet been consumed.
//!    - If there are more words than onsets, remaining words share the last
//!      onset time and are each given a 500 ms duration.
//!    - If there are no onsets at all, all words receive start_ms = 0 and
//!      duration = 500 ms.
//! 3. Duration = start_ms of the *next* word − start_ms of *this* word.
//!    The last word always receives a 500 ms duration.
//!
//! This is a **stub implementation**.  Production-quality lyrics alignment
//! requires forced-alignment systems (e.g., ctc-based acoustic models) and is
//! outside the scope of this crate.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// LyricsWord
// ---------------------------------------------------------------------------

/// A single word with its assigned timing information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LyricsWord {
    /// The word text (no surrounding whitespace).
    pub text: String,
    /// Start time in milliseconds.
    pub start_ms: u32,
    /// End time in milliseconds.
    pub end_ms: u32,
}

impl LyricsWord {
    /// Duration of the word in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u32 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

impl std::fmt::Display for LyricsWord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}ms–{}ms]", self.text, self.start_ms, self.end_ms)
    }
}

// ---------------------------------------------------------------------------
// align_lyrics
// ---------------------------------------------------------------------------

/// Align lyrics words to onset times.
///
/// # Arguments
///
/// * `lyrics` — lyrics as a UTF-8 string.  Words are separated by any Unicode
///   whitespace.  Punctuation is kept as part of the word.
/// * `onsets_ms` — onset times in milliseconds, in ascending order.
///
/// # Returns
///
/// One [`LyricsWord`] per word token found in `lyrics`.  An empty lyrics
/// string returns an empty vector.
///
/// # Example
///
/// ```
/// use oximedia_mir::lyrics_align::align_lyrics;
///
/// let lyrics = "Hello world goodbye";
/// let onsets = [0u32, 500, 1000];
/// let words = align_lyrics(lyrics, &onsets);
/// assert_eq!(words.len(), 3);
/// assert_eq!(words[0].text, "Hello");
/// assert_eq!(words[0].start_ms, 0);
/// assert_eq!(words[1].text, "world");
/// assert_eq!(words[1].start_ms, 500);
/// ```
#[must_use]
pub fn align_lyrics(lyrics: &str, onsets_ms: &[u32]) -> Vec<LyricsWord> {
    // Tokenise into words.
    let words: Vec<&str> = lyrics.split_whitespace().collect();
    if words.is_empty() {
        return Vec::new();
    }

    let n_words = words.len();
    let n_onsets = onsets_ms.len();

    // Pre-compute start times for each word.
    let start_times: Vec<u32> = if n_onsets == 0 {
        // No onsets: all words start at 0.
        vec![0u32; n_words]
    } else {
        assign_onsets_greedy(&words, onsets_ms)
    };

    // Build LyricsWord entries.
    let mut result: Vec<LyricsWord> = Vec::with_capacity(n_words);
    for (idx, &word) in words.iter().enumerate() {
        let start = start_times[idx];
        let end = if idx + 1 < n_words {
            // Duration = next word's start − this word's start (minimum 1 ms).
            let next_start = start_times[idx + 1];
            if next_start > start {
                next_start
            } else {
                start + 500
            }
        } else {
            // Last word always gets a 500 ms duration.
            start + 500
        };
        result.push(LyricsWord {
            text: word.to_string(),
            start_ms: start,
            end_ms: end,
        });
    }

    result
}

/// Greedy left-to-right onset assignment.
///
/// Assigns the *n*-th word to the *n*-th onset when the number of onsets ≥
/// number of words.  When there are more words than onsets, excess words are
/// pinned to the last onset.
///
/// When there are more onsets than words, the extra onsets are silently
/// ignored (the surplus gives breathing room for the last word's duration).
fn assign_onsets_greedy(words: &[&str], onsets_ms: &[u32]) -> Vec<u32> {
    let n_words = words.len();
    let n_onsets = onsets_ms.len();

    (0..n_words)
        .map(|word_idx| {
            // Each word gets its own onset if available; otherwise use the last one.
            let onset_idx = word_idx.min(n_onsets - 1);
            onsets_ms[onset_idx]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

/// Split a multi-line lyrics string into individual lines, preserving empty
/// lines as empty strings.
#[must_use]
pub fn split_lines(lyrics: &str) -> Vec<&str> {
    lyrics.lines().collect()
}

/// Return the total duration of all word segments in milliseconds.
#[must_use]
pub fn total_duration_ms(words: &[LyricsWord]) -> u32 {
    words
        .iter()
        .map(|w| w.end_ms)
        .fold(0u32, |acc, end| acc.max(end))
}

/// Filter words that overlap with a given time range `[from_ms, to_ms]`.
#[must_use]
pub fn words_in_range(words: &[LyricsWord], from_ms: u32, to_ms: u32) -> Vec<&LyricsWord> {
    words
        .iter()
        .filter(|w| w.start_ms < to_ms && w.end_ms > from_ms)
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── align_lyrics ──────────────────────────────────────────────────────────

    #[test]
    fn test_align_empty_lyrics() {
        let result = align_lyrics("", &[0, 500, 1000]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_align_empty_onsets() {
        let result = align_lyrics("hello world", &[]);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].start_ms, 0);
        assert_eq!(result[1].start_ms, 0);
        // All words start at 0 and get 500 ms duration.
        assert_eq!(result[0].end_ms, 500);
        assert_eq!(result[1].end_ms, 500);
    }

    #[test]
    fn test_align_three_words_three_onsets() {
        let lyrics = "Hello world goodbye";
        let onsets = [0u32, 500, 1000];
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[0].end_ms, 500); // next word starts at 500

        assert_eq!(words[1].text, "world");
        assert_eq!(words[1].start_ms, 500);
        assert_eq!(words[1].end_ms, 1000); // next word starts at 1000

        assert_eq!(words[2].text, "goodbye");
        assert_eq!(words[2].start_ms, 1000);
        assert_eq!(words[2].end_ms, 1500); // last word gets +500
    }

    #[test]
    fn test_align_more_words_than_onsets() {
        let lyrics = "one two three four five";
        let onsets = [100u32, 200, 300]; // only 3 onsets for 5 words
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 5);
        assert_eq!(words[0].start_ms, 100);
        assert_eq!(words[1].start_ms, 200);
        assert_eq!(words[2].start_ms, 300);
        // Excess words pinned to last onset.
        assert_eq!(words[3].start_ms, 300);
        assert_eq!(words[4].start_ms, 300);
    }

    #[test]
    fn test_align_more_onsets_than_words() {
        let lyrics = "quick brown";
        let onsets = [0u32, 100, 200, 300, 400]; // 5 onsets for 2 words
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].start_ms, 0);
        assert_eq!(words[1].start_ms, 100);
    }

    #[test]
    fn test_align_single_word_single_onset() {
        let words = align_lyrics("only", &[750]);
        assert_eq!(words.len(), 1);
        assert_eq!(words[0].text, "only");
        assert_eq!(words[0].start_ms, 750);
        assert_eq!(words[0].end_ms, 1250); // 750 + 500
    }

    #[test]
    fn test_align_word_duration_ms() {
        let words = align_lyrics("a b", &[0, 300]);
        assert_eq!(words[0].duration_ms(), 300); // 300 - 0
        assert_eq!(words[1].duration_ms(), 500); // last word gets +500
    }

    #[test]
    fn test_align_whitespace_lyrics() {
        // Lyrics that are only whitespace produce no words.
        let result = align_lyrics("   \t  \n  ", &[0, 100]);
        assert!(result.is_empty());
    }

    // ── LyricsWord ────────────────────────────────────────────────────────────

    #[test]
    fn test_lyrics_word_display() {
        let w = LyricsWord {
            text: "hello".to_string(),
            start_ms: 100,
            end_ms: 400,
        };
        let s = format!("{w}");
        assert!(s.contains("hello"));
        assert!(s.contains("100ms"));
        assert!(s.contains("400ms"));
    }

    #[test]
    fn test_lyrics_word_duration_saturating() {
        // end_ms < start_ms → duration clamped to 0.
        let w = LyricsWord {
            text: "x".to_string(),
            start_ms: 500,
            end_ms: 300,
        };
        assert_eq!(w.duration_ms(), 0);
    }

    // ── Helper utilities ──────────────────────────────────────────────────────

    #[test]
    fn test_total_duration_ms() {
        let words = align_lyrics("a b c", &[0, 200, 400]);
        let total = total_duration_ms(&words);
        // Last word ends at 400 + 500 = 900 ms.
        assert_eq!(total, 900);
    }

    #[test]
    fn test_words_in_range() {
        let words = align_lyrics("a b c d", &[0, 100, 200, 300]);
        // Range 50ms – 250ms should include words at 100ms and 200ms.
        let in_range = words_in_range(&words, 50, 250);
        let texts: Vec<&str> = in_range.iter().map(|w| w.text.as_str()).collect();
        assert!(texts.contains(&"b"));
        assert!(texts.contains(&"c"));
    }

    #[test]
    fn test_split_lines_preserves_empty() {
        let lyrics = "line one\n\nline three";
        let lines = split_lines(lyrics);
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1], "");
    }

    #[test]
    fn test_align_multiline_lyrics() {
        // Newlines count as whitespace; split_whitespace ignores them.
        let lyrics = "line one\nline two";
        let onsets = [0u32, 200, 400, 600];
        let words = align_lyrics(lyrics, &onsets);
        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "line");
        assert_eq!(words[1].text, "one");
    }
}
