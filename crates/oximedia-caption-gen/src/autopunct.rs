//! Auto-punctuation for caption text.
//!
//! Applies deterministic rule-based punctuation restoration to caption text:
//!
//! - Capitalises the first character of a sentence (after `. `, `! `, `? `).
//! - Appends a period at the end of the string if no sentence-ending
//!   punctuation is already present.
//! - Capitalises the very first character of the input.
//!
//! This is a lightweight, dependency-free implementation suitable for
//! normalising ASR output before subtitle delivery.
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::autopunct::AutoPunctuator;
//!
//! let result = AutoPunctuator::process("hello world");
//! assert_eq!(result, "Hello world.");
//!
//! let result2 = AutoPunctuator::process("she said hello. he replied");
//! assert_eq!(result2, "She said hello. He replied.");
//! ```

// ─── AutoPunctuator ───────────────────────────────────────────────────────────

/// Rule-based auto-punctuation processor for raw caption text.
#[derive(Debug, Clone, Default)]
pub struct AutoPunctuator;

impl AutoPunctuator {
    /// Create a new instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Apply auto-punctuation rules to `text`.
    ///
    /// Rules applied (in order):
    ///
    /// 1. Capitalise the first character of `text`.
    /// 2. Capitalise the character immediately after each `. `, `! `, or `? `.
    /// 3. If `text` does not end with `.`, `!`, or `?`, append `.`.
    ///
    /// Leading and trailing whitespace is preserved.  An empty string is
    /// returned unchanged.
    #[must_use]
    pub fn process(text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }

        // Work on a mutable byte buffer (ASCII-only operations on the
        // capitalisation markers; non-ASCII characters are passed through).
        let mut out = String::with_capacity(text.len() + 2);

        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();

        // State: whether the *next* letter should be uppercased.
        let mut capitalise_next = true;

        let mut i = 0;
        while i < n {
            let c = chars[i];
            if capitalise_next && c.is_alphabetic() {
                for upper in c.to_uppercase() {
                    out.push(upper);
                }
                capitalise_next = false;
            } else {
                out.push(c);
                // Check if we just wrote a sentence-ending punctuation followed
                // by a space — next alphabetic char should be capitalised.
                if (c == '.' || c == '!' || c == '?') && i + 1 < n && chars[i + 1] == ' ' {
                    capitalise_next = true;
                }
            }
            i += 1;
        }

        // Ensure trailing sentence-ending punctuation.
        let trimmed_end = out.trim_end();
        let last_char = trimmed_end.chars().last();
        match last_char {
            Some('.') | Some('!') | Some('?') => {}
            _ => {
                // Append period at the position just after the last non-space char.
                let trailing_spaces: usize = out.len() - trimmed_end.len();
                let insert_pos = out.len() - trailing_spaces;
                out.insert(insert_pos, '.');
            }
        }

        out
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_returns_empty() {
        assert_eq!(AutoPunctuator::process(""), "");
    }

    #[test]
    fn test_capitalises_first_char() {
        assert!(AutoPunctuator::process("hello").starts_with('H'));
    }

    #[test]
    fn test_adds_period_at_end() {
        let result = AutoPunctuator::process("hello world");
        assert!(result.ends_with('.'), "expected period, got: {result}");
    }

    #[test]
    fn test_no_double_period() {
        let result = AutoPunctuator::process("hello world.");
        assert_eq!(result.chars().filter(|&c| c == '.').count(), 1);
    }

    #[test]
    fn test_capitalises_after_period_space() {
        let result = AutoPunctuator::process("she said hello. he replied");
        assert!(
            result.contains(". He"),
            "expected 'He' after period, got: {result}"
        );
    }

    #[test]
    fn test_capitalises_after_exclamation_space() {
        let result = AutoPunctuator::process("wow! that is great");
        assert!(
            result.contains("! That"),
            "expected 'That' after '!', got: {result}"
        );
    }

    #[test]
    fn test_capitalises_after_question_space() {
        let result = AutoPunctuator::process("are you there? yes i am");
        assert!(
            result.contains("? Yes"),
            "expected 'Yes' after '?', got: {result}"
        );
    }

    #[test]
    fn test_existing_exclamation_no_extra_period() {
        let result = AutoPunctuator::process("hello!");
        assert!(!result.ends_with("!."), "should not add period after '!'");
    }

    #[test]
    fn test_existing_question_mark_no_extra_period() {
        let result = AutoPunctuator::process("are you sure?");
        assert!(result.ends_with('?'));
        assert!(!result.ends_with("?."));
    }

    #[test]
    fn test_single_word() {
        let result = AutoPunctuator::process("yes");
        assert_eq!(result, "Yes.");
    }

    #[test]
    fn test_already_capitalised() {
        let result = AutoPunctuator::process("Hello world");
        assert!(result.starts_with("Hello"));
    }

    #[test]
    fn test_multiple_sentences() {
        let result = AutoPunctuator::process("first sentence. second sentence");
        assert!(result.contains("First"), "first word should be capitalised");
        assert!(
            result.contains(". Second"),
            "after period should be capitalised"
        );
        assert!(result.ends_with('.'), "should end with period");
    }
}
