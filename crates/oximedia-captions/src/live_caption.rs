//! Live captioning / real-time streaming captions.
//!
//! Provides types for ingesting word-level captions as they are recognised
//! by a speech engine, buffering partial results, and committing final text.

#![allow(dead_code)]
#![allow(missing_docs)]

// ── LiveCaptionMode ───────────────────────────────────────────────────────────

/// Indicates the stability level of a recognised word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveCaptionMode {
    /// A word-granularity hypothesis that may still change.
    Word,
    /// An in-progress utterance that is not yet final.
    Partial,
    /// A finalised, committed recognition result.
    Final,
}

impl LiveCaptionMode {
    /// Returns `true` for results that are committed and will not be revised.
    #[must_use]
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Final)
    }
}

// ── LiveWord ─────────────────────────────────────────────────────────────────

/// A single recognised word from a live caption stream.
#[derive(Debug, Clone)]
pub struct LiveWord {
    /// The recognised text of the word.
    pub text: String,
    /// Timestamp when recognition started for this word, in milliseconds.
    pub start_ms: u64,
    /// Confidence score in the range `[0.0, 1.0]`.
    pub confidence: f32,
    /// Stability mode of this word.
    pub mode: LiveCaptionMode,
}

impl LiveWord {
    /// Create a new live word.
    pub fn new(
        text: impl Into<String>,
        start_ms: u64,
        confidence: f32,
        mode: LiveCaptionMode,
    ) -> Self {
        Self {
            text: text.into(),
            start_ms,
            confidence,
            mode,
        }
    }

    /// Returns `true` when the confidence score meets or exceeds `threshold`.
    #[must_use]
    pub fn is_reliable(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

// ── LiveCaptionBuffer ─────────────────────────────────────────────────────────

/// A buffer that accumulates incoming live words and manages committed text.
///
/// Words older than `max_latency_ms` from a reference time are flushed to
/// keep the buffer from growing indefinitely.
#[derive(Debug)]
pub struct LiveCaptionBuffer {
    /// Words currently held in the buffer (partial or word-mode).
    pub words: Vec<LiveWord>,
    /// Text that has been committed (finalised) and will not change.
    pub committed_text: String,
    /// Maximum age (in ms) that a word may remain in the buffer before flush.
    pub max_latency_ms: u32,
}

impl LiveCaptionBuffer {
    /// Create a new, empty buffer.
    #[must_use]
    pub fn new(max_latency_ms: u32) -> Self {
        Self {
            words: Vec::new(),
            committed_text: String::new(),
            max_latency_ms,
        }
    }

    /// Add a word to the buffer.
    pub fn add_word(&mut self, word: LiveWord) {
        self.words.push(word);
    }

    /// Commit all words currently in the buffer to `committed_text` and clear
    /// the word buffer.
    pub fn commit(&mut self) {
        for word in self.words.drain(..) {
            if !self.committed_text.is_empty() {
                self.committed_text.push(' ');
            }
            self.committed_text.push_str(&word.text);
        }
    }

    /// Returns the partial (in-progress) text from unbuffered words only,
    /// without modifying the buffer.
    #[must_use]
    pub fn partial_text(&self) -> String {
        self.words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Number of words that have been committed to `committed_text`.
    ///
    /// Counts space-separated tokens in `committed_text`, returning 0 for an
    /// empty string.
    #[must_use]
    pub fn total_committed_words(&self) -> usize {
        if self.committed_text.is_empty() {
            0
        } else {
            self.committed_text.split_whitespace().count()
        }
    }

    /// Remove words whose `start_ms` is older than `now_ms − max_latency_ms`.
    ///
    /// Removed words are committed to `committed_text` in arrival order before
    /// being dropped, so no recognised content is lost.
    pub fn flush_old(&mut self, now_ms: u64) {
        let cutoff = now_ms.saturating_sub(u64::from(self.max_latency_ms));
        let mut to_commit: Vec<LiveWord> = Vec::new();
        self.words.retain(|w| {
            if w.start_ms < cutoff {
                to_commit.push(w.clone());
                false
            } else {
                true
            }
        });
        // Commit in the order they were retained (stable by insertion order).
        for word in to_commit {
            if !self.committed_text.is_empty() {
                self.committed_text.push(' ');
            }
            self.committed_text.push_str(&word.text);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, start_ms: u64, conf: f32, mode: LiveCaptionMode) -> LiveWord {
        LiveWord::new(text, start_ms, conf, mode)
    }

    // ── LiveCaptionMode ──

    #[test]
    fn test_mode_final_is_committed() {
        assert!(LiveCaptionMode::Final.is_committed());
    }

    #[test]
    fn test_mode_partial_not_committed() {
        assert!(!LiveCaptionMode::Partial.is_committed());
    }

    #[test]
    fn test_mode_word_not_committed() {
        assert!(!LiveCaptionMode::Word.is_committed());
    }

    // ── LiveWord ──

    #[test]
    fn test_word_is_reliable_above_threshold() {
        let w = word("hello", 0, 0.9, LiveCaptionMode::Final);
        assert!(w.is_reliable(0.8));
    }

    #[test]
    fn test_word_is_reliable_below_threshold() {
        let w = word("hmm", 0, 0.4, LiveCaptionMode::Partial);
        assert!(!w.is_reliable(0.7));
    }

    #[test]
    fn test_word_is_reliable_at_exact_threshold() {
        let w = word("ok", 0, 0.5, LiveCaptionMode::Word);
        assert!(w.is_reliable(0.5));
    }

    // ── LiveCaptionBuffer ──

    #[test]
    fn test_buffer_add_and_partial_text() {
        let mut buf = LiveCaptionBuffer::new(1000);
        buf.add_word(word("Hello", 100, 0.9, LiveCaptionMode::Word));
        buf.add_word(word("world", 200, 0.85, LiveCaptionMode::Word));
        assert_eq!(buf.partial_text(), "Hello world");
    }

    #[test]
    fn test_buffer_commit_clears_words() {
        let mut buf = LiveCaptionBuffer::new(1000);
        buf.add_word(word("Hello", 0, 1.0, LiveCaptionMode::Final));
        buf.commit();
        assert!(buf.words.is_empty());
        assert_eq!(buf.committed_text, "Hello");
    }

    #[test]
    fn test_buffer_total_committed_words_empty() {
        let buf = LiveCaptionBuffer::new(500);
        assert_eq!(buf.total_committed_words(), 0);
    }

    #[test]
    fn test_buffer_total_committed_words_after_commit() {
        let mut buf = LiveCaptionBuffer::new(500);
        buf.add_word(word("one", 0, 0.9, LiveCaptionMode::Final));
        buf.add_word(word("two", 100, 0.9, LiveCaptionMode::Final));
        buf.commit();
        assert_eq!(buf.total_committed_words(), 2);
    }

    #[test]
    fn test_buffer_flush_old_commits_stale_words() {
        let mut buf = LiveCaptionBuffer::new(500);
        // word started at 0, now = 1000 → 0 < 1000 - 500 = 500 → flush
        buf.add_word(word("stale", 0, 0.9, LiveCaptionMode::Word));
        // word started at 600, now = 1000 → 600 >= 500 → keep
        buf.add_word(word("fresh", 600, 0.9, LiveCaptionMode::Word));
        buf.flush_old(1000);
        assert_eq!(buf.committed_text, "stale");
        assert_eq!(buf.words.len(), 1);
        assert_eq!(buf.words[0].text, "fresh");
    }

    #[test]
    fn test_buffer_flush_old_keeps_recent_words() {
        let mut buf = LiveCaptionBuffer::new(2000);
        buf.add_word(word("new", 900, 1.0, LiveCaptionMode::Word));
        buf.flush_old(1000); // cutoff = 1000 - 2000 = 0 (saturates to 0) → 900 >= 0 → keep
                             // All words older than (1000 - 2000) = nothing → none flushed
                             // Actually 900 < (1000 saturating_sub 2000) = 0 is false because 900 >= 0
        assert_eq!(buf.words.len(), 1);
        assert!(buf.committed_text.is_empty());
    }

    #[test]
    fn test_buffer_multiple_commits_accumulate() {
        let mut buf = LiveCaptionBuffer::new(1000);
        buf.add_word(word("first", 0, 1.0, LiveCaptionMode::Final));
        buf.commit();
        buf.add_word(word("second", 100, 1.0, LiveCaptionMode::Final));
        buf.commit();
        assert_eq!(buf.committed_text, "first second");
        assert_eq!(buf.total_committed_words(), 2);
    }

    #[test]
    fn test_buffer_partial_text_empty_when_no_words() {
        let buf = LiveCaptionBuffer::new(1000);
        assert_eq!(buf.partial_text(), "");
    }
}
