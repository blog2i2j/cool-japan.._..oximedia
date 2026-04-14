//! Profanity filter for caption text.
//!
//! Replaces occurrences of a configurable word list with asterisk-censored
//! substitutes.  Matching is case-insensitive and whole-word only (bounded
//! by whitespace or punctuation boundaries).
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::profanity::ProfanityFilter;
//!
//! let filter = ProfanityFilter::new(vec!["darn".to_string(), "heck".to_string()]);
//! let cleaned = filter.filter("What the heck is going on here darn it.");
//! assert!(!cleaned.to_lowercase().contains("heck"));
//! assert!(!cleaned.to_lowercase().contains("darn"));
//! ```

// ─── ProfanityFilter ─────────────────────────────────────────────────────────

/// A word-list-based profanity filter.
///
/// Matches words case-insensitively at word boundaries and replaces them with
/// a censored placeholder of the same length.
#[derive(Debug, Clone)]
pub struct ProfanityFilter {
    /// Lower-cased words that should be censored.
    wordlist: Vec<String>,
    /// Character used to fill censored words.  Defaults to `'*'`.
    pub censor_char: char,
}

impl ProfanityFilter {
    /// Create a new filter with the given word list.
    ///
    /// Words are stored in lower-case for case-insensitive matching.
    #[must_use]
    pub fn new(wordlist: Vec<String>) -> Self {
        Self {
            wordlist: wordlist.into_iter().map(|w| w.to_lowercase()).collect(),
            censor_char: '*',
        }
    }

    /// Create a filter with no blocked words (passes text through unchanged).
    #[must_use]
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// Returns the number of blocked words in the filter.
    #[must_use]
    pub fn len(&self) -> usize {
        self.wordlist.len()
    }

    /// Returns `true` if the word list is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.wordlist.is_empty()
    }

    /// Add a word to the blocked list.
    pub fn add_word(&mut self, word: &str) {
        self.wordlist.push(word.to_lowercase());
    }

    /// Filter `text`, replacing blocked words with asterisks.
    ///
    /// Each blocked word is replaced with a string of `censor_char` characters
    /// of the same byte length as the matched word.  Matching is performed
    /// case-insensitively and requires the match to be at a word boundary (the
    /// character before and after the match must be a non-alphabetic byte, or
    /// be at the string edge).
    #[must_use]
    pub fn filter(&self, text: &str) -> String {
        if self.wordlist.is_empty() || text.is_empty() {
            return text.to_owned();
        }

        let mut result = text.to_owned();
        for blocked in &self.wordlist {
            result = self.replace_word(&result, blocked);
        }
        result
    }

    /// Check if `text` contains any blocked word.
    #[must_use]
    pub fn contains_profanity(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        self.wordlist.iter().any(|w| {
            let mut start = 0;
            while let Some(pos) = lower[start..].find(w.as_str()) {
                let abs_pos = start + pos;
                if Self::is_word_boundary(&lower, abs_pos, w.len()) {
                    return true;
                }
                start = abs_pos + 1;
            }
            false
        })
    }

    // ─── Private helpers ──────────────────────────────────────────────────────

    fn replace_word(&self, text: &str, blocked: &str) -> String {
        if blocked.is_empty() {
            return text.to_owned();
        }
        let lower = text.to_lowercase();
        let censored = self.censor_char.to_string().repeat(blocked.len());

        let mut out = String::with_capacity(text.len());
        let mut search_start = 0usize;

        loop {
            match lower[search_start..].find(blocked) {
                None => {
                    out.push_str(&text[search_start..]);
                    break;
                }
                Some(rel_pos) => {
                    let abs_pos = search_start + rel_pos;
                    if Self::is_word_boundary(&lower, abs_pos, blocked.len()) {
                        out.push_str(&text[search_start..abs_pos]);
                        out.push_str(&censored);
                        search_start = abs_pos + blocked.len();
                    } else {
                        // Not a word boundary — copy up to and including this
                        // char and continue searching.
                        let next_char_end = lower[abs_pos..]
                            .char_indices()
                            .nth(1)
                            .map(|(i, _)| abs_pos + i)
                            .unwrap_or(abs_pos + 1);
                        out.push_str(&text[search_start..next_char_end]);
                        search_start = next_char_end;
                    }
                }
            }
        }

        out
    }

    /// Returns `true` if the substring `[pos, pos+len)` is at a word boundary.
    ///
    /// A word boundary means:
    /// - The character *before* the match (if any) is not an ASCII letter or digit.
    /// - The character *after* the match (if any) is not an ASCII letter or digit.
    fn is_word_boundary(lower: &str, pos: usize, len: usize) -> bool {
        let before_ok = if pos == 0 {
            true
        } else {
            lower[..pos]
                .chars()
                .last()
                .map(|c| !c.is_alphanumeric())
                .unwrap_or(true)
        };

        let after_pos = pos + len;
        let after_ok = if after_pos >= lower.len() {
            true
        } else {
            lower[after_pos..]
                .chars()
                .next()
                .map(|c| !c.is_alphanumeric())
                .unwrap_or(true)
        };

        before_ok && after_ok
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_filter() -> ProfanityFilter {
        ProfanityFilter::new(vec!["darn".to_string(), "heck".to_string()])
    }

    #[test]
    fn test_filter_replaces_blocked_word() {
        let f = make_filter();
        let out = f.filter("what the heck");
        assert!(
            !out.to_lowercase().contains("heck"),
            "heck not censored: {out}"
        );
    }

    #[test]
    fn test_filter_case_insensitive() {
        let f = make_filter();
        let out = f.filter("What the HECK is going on");
        assert!(
            !out.to_lowercase().contains("heck"),
            "HECK not censored: {out}"
        );
    }

    #[test]
    fn test_filter_preserves_non_blocked_text() {
        let f = make_filter();
        let out = f.filter("hello world");
        assert_eq!(out, "hello world");
    }

    #[test]
    fn test_filter_empty_wordlist() {
        let f = ProfanityFilter::empty();
        assert_eq!(f.filter("some text"), "some text");
    }

    #[test]
    fn test_filter_empty_text() {
        let f = make_filter();
        assert_eq!(f.filter(""), "");
    }

    #[test]
    fn test_filter_word_at_boundary() {
        let f = ProfanityFilter::new(vec!["bad".to_string()]);
        let out = f.filter("that is bad, indeed");
        assert!(
            !out.to_lowercase().contains("bad"),
            "bad not censored: {out}"
        );
    }

    #[test]
    fn test_filter_no_partial_match() {
        // "heck" should not match inside "heckle"
        let f = ProfanityFilter::new(vec!["heck".to_string()]);
        let out = f.filter("please do not heckle the speaker");
        assert!(out.contains("heckle"), "heckle should not be censored");
    }

    #[test]
    fn test_contains_profanity_true() {
        let f = make_filter();
        assert!(f.contains_profanity("what the heck"));
    }

    #[test]
    fn test_contains_profanity_false() {
        let f = make_filter();
        assert!(!f.contains_profanity("this is fine"));
    }

    #[test]
    fn test_add_word() {
        let mut f = ProfanityFilter::empty();
        f.add_word("oops");
        assert_eq!(f.len(), 1);
        let out = f.filter("oops I did it again");
        assert!(!out.to_lowercase().contains("oops"));
    }

    #[test]
    fn test_censored_length_matches_word_length() {
        let f = ProfanityFilter::new(vec!["darn".to_string()]);
        let out = f.filter("darn it");
        // "darn" (4 chars) should be replaced by 4 asterisks
        assert!(out.starts_with("****"), "expected 4 asterisks, got: {out}");
    }
}
