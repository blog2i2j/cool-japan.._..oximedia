//! Live captioning pipeline.
//!
//! Provides types and utilities for ingesting real-time speech recognition
//! output and assembling caption segments with latency monitoring.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Configuration for a live captioning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveCaptionConfig {
    /// Target maximum end-to-end latency, in milliseconds.
    pub latency_target_ms: u32,
    /// Minimum confidence threshold (0.0–1.0) for accepting words.
    pub confidence_threshold: f32,
    /// Maximum number of words to buffer before flushing.
    pub max_buffer_words: u32,
    /// BCP-47 language tag (e.g. `"en-US"`).
    pub language: String,
}

impl LiveCaptionConfig {
    /// Create a config with sensible defaults for English live captioning.
    #[must_use]
    pub fn default_english() -> Self {
        Self {
            latency_target_ms: 3000,
            confidence_threshold: 0.75,
            max_buffer_words: 20,
            language: "en-US".to_string(),
        }
    }
}

impl Default for LiveCaptionConfig {
    fn default() -> Self {
        Self::default_english()
    }
}

/// A single recognised word from an ASR engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionWord {
    /// The word text.
    pub word: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Start time of the word, in milliseconds since stream start.
    pub start_ms: u64,
    /// End time of the word, in milliseconds since stream start.
    pub end_ms: u64,
    /// Optional speaker identifier for diarised streams.
    pub speaker_id: Option<u32>,
}

impl CaptionWord {
    /// Create a new caption word.
    #[must_use]
    pub fn new(
        word: impl Into<String>,
        confidence: f32,
        start_ms: u64,
        end_ms: u64,
        speaker_id: Option<u32>,
    ) -> Self {
        Self {
            word: word.into(),
            confidence: confidence.clamp(0.0, 1.0),
            start_ms,
            end_ms,
            speaker_id,
        }
    }
}

/// A segment of caption text, formed from one or more words.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionSegment {
    /// The words that make up this segment.
    pub words: Vec<CaptionWord>,
    /// Full text of the segment (words joined with spaces).
    pub text: String,
    /// Start time of the segment, in milliseconds.
    pub start_ms: u64,
    /// End time of the segment, in milliseconds.
    pub end_ms: u64,
    /// Whether the ASR engine has confirmed this segment is final.
    pub is_final: bool,
}

impl CaptionSegment {
    /// Construct a segment from a list of words.
    ///
    /// The `text`, `start_ms`, and `end_ms` fields are derived automatically.
    /// Returns `None` if `words` is empty.
    #[must_use]
    pub fn from_words(words: Vec<CaptionWord>) -> Option<Self> {
        if words.is_empty() {
            return None;
        }
        let start_ms = words.first().map_or(0, |w| w.start_ms);
        let end_ms = words.last().map_or(0, |w| w.end_ms);
        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Some(Self {
            words,
            text,
            start_ms,
            end_ms,
            is_final: false,
        })
    }

    /// Mark this segment as final.
    #[must_use]
    pub fn finalized(mut self) -> Self {
        self.is_final = true;
        self
    }

    /// Duration of this segment in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A rolling buffer that accumulates words and flushes caption segments when a
/// sentence boundary is detected.
#[derive(Debug, Default)]
pub struct LiveCaptionBuffer {
    pending: Vec<CaptionWord>,
}

impl LiveCaptionBuffer {
    /// Create a new empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a word to the buffer.
    pub fn add_word(&mut self, word: CaptionWord) {
        self.pending.push(word);
    }

    /// Attempt to flush a ready segment.
    ///
    /// A segment is considered ready when the last word in the buffer meets
    /// `confidence_min` AND the last word ends a sentence (detected by
    /// `SentenceBoundaryDetector`).
    ///
    /// Returns `Some(CaptionSegment)` and clears the buffer on success.
    pub fn flush_ready(&mut self, confidence_min: f32) -> Option<CaptionSegment> {
        if self.pending.is_empty() {
            return None;
        }

        // Check the last word
        let last = self.pending.last()?;
        let prev = if self.pending.len() >= 2 {
            &self.pending[self.pending.len() - 2].word.clone()
        } else {
            ""
        };

        let at_boundary = SentenceBoundaryDetector::is_boundary(&last.word, prev);
        let high_confidence = last.confidence >= confidence_min;

        if at_boundary && high_confidence {
            let words: Vec<CaptionWord> = self.pending.drain(..).collect();
            CaptionSegment::from_words(words).map(CaptionSegment::finalized)
        } else {
            None
        }
    }

    /// Return the current text of all buffered words joined with spaces.
    #[must_use]
    pub fn current_text(&self) -> String {
        self.pending
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Number of words currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Returns `true` if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Force-flush all pending words as a (possibly non-final) segment,
    /// regardless of whether a sentence boundary has been detected.
    pub fn flush_all(&mut self) -> Option<CaptionSegment> {
        if self.pending.is_empty() {
            return None;
        }
        let words: Vec<CaptionWord> = self.pending.drain(..).collect();
        CaptionSegment::from_words(words)
    }
}

/// Detects sentence boundaries based on punctuation.
pub struct SentenceBoundaryDetector;

impl SentenceBoundaryDetector {
    /// Returns `true` if `word` (after `prev_word`) represents a sentence
    /// boundary.
    ///
    /// A boundary is detected when `word` ends with `.`, `!`, or `?`, or when
    /// `prev_word` ends with those characters.
    #[must_use]
    pub fn is_boundary(word: &str, prev_word: &str) -> bool {
        let ends_sentence = |w: &str| {
            w.ends_with('.') || w.ends_with('!') || w.ends_with('?') || w.ends_with("...")
        };
        ends_sentence(word) || ends_sentence(prev_word)
    }
}

/// Tracks end-to-end captioning latency and computes statistics.
#[derive(Debug, Clone)]
pub struct LatencyMonitor {
    /// Target latency in milliseconds.
    pub target_ms: u32,
    /// Recorded latency measurements.
    pub measurements: Vec<u64>,
}

impl LatencyMonitor {
    /// Create a new monitor with the given target.
    #[must_use]
    pub fn new(target_ms: u32) -> Self {
        Self {
            target_ms,
            measurements: Vec::new(),
        }
    }

    /// Record a latency measurement.
    pub fn add(&mut self, latency_ms: u64) {
        self.measurements.push(latency_ms);
    }

    /// Return the 95th-percentile latency across all recorded measurements.
    ///
    /// Uses the nearest-rank method: returns the value at rank
    /// `ceil(0.95 * N)` (1-based), i.e. index `ceil(0.95 * N) - 1`.
    ///
    /// Returns `0` if no measurements have been recorded.
    #[must_use]
    pub fn p95_ms(&self) -> u64 {
        if self.measurements.is_empty() {
            return 0;
        }
        let mut sorted = self.measurements.clone();
        sorted.sort_unstable();
        let n = sorted.len() as f64;
        // Nearest-rank: rank = ceil(0.95 * N), index = rank - 1
        let rank = (0.95 * n).ceil() as usize;
        let idx = rank.saturating_sub(1).min(sorted.len() - 1);
        sorted[idx]
    }

    /// Returns `true` if the p95 latency is at or below the target.
    #[must_use]
    pub fn is_meeting_target(&self) -> bool {
        self.p95_ms() <= u64::from(self.target_ms)
    }

    /// Returns the number of recorded measurements.
    #[must_use]
    pub fn count(&self) -> usize {
        self.measurements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(text: &str, confidence: f32, start_ms: u64, end_ms: u64) -> CaptionWord {
        CaptionWord::new(text, confidence, start_ms, end_ms, None)
    }

    #[test]
    fn test_caption_config_defaults() {
        let config = LiveCaptionConfig::default_english();
        assert_eq!(config.language, "en-US");
        assert!(config.confidence_threshold > 0.0);
    }

    #[test]
    fn test_caption_word_confidence_clamped() {
        let w = CaptionWord::new("hello", 1.5, 0, 100, None);
        assert_eq!(w.confidence, 1.0);
        let w2 = CaptionWord::new("world", -0.5, 0, 100, None);
        assert_eq!(w2.confidence, 0.0);
    }

    #[test]
    fn test_segment_from_words() {
        let words = vec![
            make_word("Hello", 0.95, 0, 500),
            make_word("world.", 0.90, 500, 1000),
        ];
        let seg =
            CaptionSegment::from_words(words).expect("caption segment creation should succeed");
        assert_eq!(seg.text, "Hello world.");
        assert_eq!(seg.start_ms, 0);
        assert_eq!(seg.end_ms, 1000);
    }

    #[test]
    fn test_segment_from_empty_words() {
        let seg = CaptionSegment::from_words(vec![]);
        assert!(seg.is_none());
    }

    #[test]
    fn test_segment_duration() {
        let words = vec![make_word("Test.", 0.9, 1000, 2500)];
        let seg =
            CaptionSegment::from_words(words).expect("caption segment creation should succeed");
        assert_eq!(seg.duration_ms(), 1500);
    }

    #[test]
    fn test_buffer_add_and_current_text() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.9, 0, 300));
        buf.add_word(make_word("there.", 0.85, 300, 600));
        assert_eq!(buf.current_text(), "Hello there.");
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_buffer_flush_on_sentence_boundary() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.95, 0, 300));
        buf.add_word(make_word("world.", 0.92, 300, 600));
        let seg = buf.flush_ready(0.80);
        assert!(seg.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_no_flush_without_boundary() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Hello", 0.95, 0, 300));
        buf.add_word(make_word("world", 0.92, 300, 600)); // no period
        let seg = buf.flush_ready(0.80);
        assert!(seg.is_none());
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_flush_all() {
        let mut buf = LiveCaptionBuffer::new();
        buf.add_word(make_word("Testing", 0.9, 0, 400));
        let seg = buf.flush_all();
        assert!(seg.is_some());
        assert!(buf.is_empty());
    }

    #[test]
    fn test_sentence_boundary_period() {
        assert!(SentenceBoundaryDetector::is_boundary("world.", "hello"));
        assert!(!SentenceBoundaryDetector::is_boundary("world", "hello"));
    }

    #[test]
    fn test_sentence_boundary_exclamation() {
        assert!(SentenceBoundaryDetector::is_boundary("stop!", "please"));
    }

    #[test]
    fn test_sentence_boundary_question() {
        assert!(SentenceBoundaryDetector::is_boundary("ready?", "are"));
    }

    #[test]
    fn test_latency_monitor_p95() {
        let mut monitor = LatencyMonitor::new(3000);
        for i in 1..=20u64 {
            monitor.add(i * 100);
        }
        // p95 of [100,200,...,2000] → index 19 (95% of 20 = 19) → 1900
        assert_eq!(monitor.p95_ms(), 1900);
    }

    #[test]
    fn test_latency_monitor_empty() {
        let monitor = LatencyMonitor::new(3000);
        assert_eq!(monitor.p95_ms(), 0);
        assert!(monitor.is_meeting_target());
    }

    #[test]
    fn test_latency_monitor_meeting_target() {
        let mut monitor = LatencyMonitor::new(3000);
        for _ in 0..100 {
            monitor.add(1500);
        }
        assert!(monitor.is_meeting_target());
    }

    #[test]
    fn test_latency_monitor_failing_target() {
        let mut monitor = LatencyMonitor::new(1000);
        for _ in 0..100 {
            monitor.add(5000);
        }
        assert!(!monitor.is_meeting_target());
    }
}
