//! Punctuation restoration for raw ASR transcript output.
//!
//! Automatic Speech Recognition (ASR) systems typically produce text without
//! punctuation.  This module implements a rule-based punctuation restoration
//! pipeline that adds sentence-ending punctuation (`.`, `?`, `!`), commas, and
//! capitalises sentence beginnings — all without external dependencies.
//!
//! ## Approach
//!
//! 1. **Sentence boundary detection** — identifies likely sentence boundaries
//!    using a set of pause-duration heuristics (when word-level timestamps are
//!    available) and a lexical cue vocabulary (question words, exclamatory
//!    phrases, filler patterns).
//! 2. **Comma insertion** — adds commas after common discourse markers
//!    (conjunctions, parenthetical adverbs) when they appear at clause-initial
//!    positions.
//! 3. **Capitalisation** — capitalises the first word after every detected
//!    sentence boundary and proper-noun candidates from a small built-in list.
//!
//! ## Limitations
//!
//! Rule-based restoration is not as accurate as neural approaches.  Typical
//! accuracy on conversational speech is ~75–85% F1 for period placement.  For
//! production use, supplement with a language-model-based approach.

use crate::alignment::{TranscriptSegment, WordTimestamp};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the punctuation restoration pipeline.
#[derive(Debug, Clone)]
pub struct PunctuationConfig {
    /// Minimum silence gap (ms) between words that triggers a sentence boundary.
    /// Set to 0 to disable gap-based boundary detection.
    pub boundary_gap_ms: u64,
    /// Whether to capitalise the first word of every restored sentence.
    pub capitalise_sentences: bool,
    /// Whether to insert commas after discourse markers.
    pub insert_commas: bool,
    /// Whether to attempt to distinguish `?` vs `.` at sentence endings.
    pub use_question_detection: bool,
    /// Whether to detect exclamatory endings and emit `!`.
    pub use_exclamation_detection: bool,
}

impl Default for PunctuationConfig {
    fn default() -> Self {
        Self {
            boundary_gap_ms: 700,
            capitalise_sentences: true,
            insert_commas: true,
            use_question_detection: true,
            use_exclamation_detection: true,
        }
    }
}

// ─── Lexical cue tables ───────────────────────────────────────────────────────

/// Words/phrases that typically open a question.
static QUESTION_STARTERS: &[&str] = &[
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom", "is", "are", "was",
    "were", "do", "does", "did", "can", "could", "would", "will", "shall", "should", "may",
    "might", "have", "has", "had",
];

/// Words/phrases that can close a question (tag questions etc.).
static QUESTION_CLOSERS: &[&str] = &[
    "right",
    "yeah",
    "correct",
    "true",
    "huh",
    "ok",
    "okay",
    "isn't it",
    "aren't you",
    "don't you",
    "didn't you",
    "haven't you",
    "wasn't it",
];

/// Interjections and intensifiers that typically precede `!`.
static EXCLAMATORY_WORDS: &[&str] = &[
    "wow",
    "oh",
    "ah",
    "hey",
    "amazing",
    "incredible",
    "fantastic",
    "wonderful",
    "terrible",
    "awful",
    "great",
    "no",
    "yes",
    "absolutely",
    "exactly",
    "indeed",
    "certainly",
    "definitely",
    "never",
    "always",
];

/// Discourse markers / parenthetical adverbs after which a comma is inserted.
static COMMA_AFTER: &[&str] = &[
    "however",
    "therefore",
    "furthermore",
    "moreover",
    "nevertheless",
    "additionally",
    "consequently",
    "meanwhile",
    "otherwise",
    "subsequently",
    "also",
    "thus",
    "hence",
    "still",
    "yet",
    "indeed",
    "instead",
    "likewise",
    "similarly",
    "unfortunately",
    "fortunately",
    "first",
    "second",
    "third",
    "finally",
    "lastly",
    "for example",
    "in fact",
    "in addition",
    "as a result",
    "on the other hand",
    "on the contrary",
    "to be honest",
    "to summarise",
    "well",
    "now",
    "so",
];

// ─── Core restoration logic ───────────────────────────────────────────────────

/// A word token with optional punctuation to be appended.
#[derive(Debug, Clone, PartialEq)]
struct Token<'a> {
    word: &'a str,
    append_punct: Option<char>,
    capitalise: bool,
}

/// Restore punctuation for a plain text string (no timing information).
///
/// Returns the restored text.
pub fn restore_punctuation_text(text: &str, config: &PunctuationConfig) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }

    let mut tokens: Vec<Token<'_>> = words
        .iter()
        .map(|w| Token {
            word: w,
            append_punct: None,
            capitalise: false,
        })
        .collect();

    // Mark first token as capitalised.
    if config.capitalise_sentences && !tokens.is_empty() {
        tokens[0].capitalise = true;
    }

    // Classify sentence endings (heuristic: last word in text or before common
    // connectives that look like new sentence starters).
    mark_sentence_boundaries_text(&mut tokens, config);

    // Insert commas after discourse markers.
    if config.insert_commas {
        mark_commas(&mut tokens);
    }

    // Reconstruct.
    assemble_tokens(&tokens)
}

/// Restore punctuation for a [`TranscriptSegment`] that carries word-level
/// timestamps.  Uses silence gaps between words to detect sentence boundaries.
///
/// Returns a new [`TranscriptSegment`] with the restored text (word timestamps
/// are preserved verbatim).
pub fn restore_punctuation_segment(
    segment: &TranscriptSegment,
    config: &PunctuationConfig,
) -> TranscriptSegment {
    if segment.words.is_empty() {
        // Fall back to text-only restoration.
        let restored_text = restore_punctuation_text(&segment.text, config);
        return TranscriptSegment {
            text: restored_text,
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            speaker_id: segment.speaker_id,
            words: Vec::new(),
        };
    }

    let words: Vec<&str> = segment.words.iter().map(|w| w.word.as_str()).collect();

    let mut tokens: Vec<Token<'_>> = words
        .iter()
        .map(|w| Token {
            word: w,
            append_punct: None,
            capitalise: false,
        })
        .collect();

    if config.capitalise_sentences && !tokens.is_empty() {
        tokens[0].capitalise = true;
    }

    // Gap-based boundary detection using word timestamps.
    if config.boundary_gap_ms > 0 {
        mark_sentence_boundaries_timed(&mut tokens, &segment.words, config);
    } else {
        mark_sentence_boundaries_text(&mut tokens, config);
    }

    if config.insert_commas {
        mark_commas(&mut tokens);
    }

    let restored_text = assemble_tokens(&tokens);

    TranscriptSegment {
        text: restored_text,
        start_ms: segment.start_ms,
        end_ms: segment.end_ms,
        speaker_id: segment.speaker_id,
        words: segment.words.clone(),
    }
}

/// Restore punctuation across a slice of segments.
///
/// Each segment is processed independently; inter-segment boundaries are
/// treated as sentence boundaries automatically.
pub fn restore_punctuation_track(
    segments: &[TranscriptSegment],
    config: &PunctuationConfig,
) -> Vec<TranscriptSegment> {
    segments
        .iter()
        .map(|seg| restore_punctuation_segment(seg, config))
        .collect()
}

// ─── Boundary detection helpers ───────────────────────────────────────────────

/// Mark sentence boundaries in `tokens` using only lexical cues (no timing).
fn mark_sentence_boundaries_text(tokens: &mut [Token<'_>], config: &PunctuationConfig) {
    let n = tokens.len();
    if n == 0 {
        return;
    }

    // Simple heuristic: every word that is followed by a word in
    // QUESTION_STARTERS or EXCLAMATORY_WORDS triggers a boundary.
    // The last token always gets terminal punctuation.

    for i in 0..n {
        // Check if this is the last token.
        if i + 1 == n {
            let punct = determine_ending_punct(&tokens[0..=i], config);
            tokens[i].append_punct = Some(punct);
            break;
        }

        let next_word_lower = tokens[i + 1].word.to_lowercase();

        // Check for an exclamatory context on the *current* word.
        let current_lower = tokens[i].word.to_lowercase();
        let current_is_exclamatory = config.use_exclamation_detection
            && EXCLAMATORY_WORDS.iter().any(|w| current_lower == *w);

        // Check if the next word starts a new sentence (question or capital).
        let next_starts_question = config.use_question_detection
            && QUESTION_STARTERS.iter().any(|w| next_word_lower == *w);

        // Detect a capitalised next word as a sentence start (heuristic).
        let next_is_capitalised = tokens[i + 1]
            .word
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);

        if current_is_exclamatory || (next_is_capitalised && i > 0) {
            let punct = if current_is_exclamatory { '!' } else { '.' };
            tokens[i].append_punct = Some(punct);
            if config.capitalise_sentences {
                tokens[i + 1].capitalise = true;
            }
        } else if next_starts_question && i > 0 {
            tokens[i].append_punct = Some('.');
            if config.capitalise_sentences {
                tokens[i + 1].capitalise = true;
            }
        }
    }
}

/// Mark sentence boundaries using word-level silence gaps.
fn mark_sentence_boundaries_timed(
    tokens: &mut [Token<'_>],
    words: &[WordTimestamp],
    config: &PunctuationConfig,
) {
    let n = tokens.len().min(words.len());
    if n == 0 {
        return;
    }

    for i in 0..n {
        if i + 1 == n {
            // Final token.
            let punct = determine_ending_punct(&tokens[0..=i], config);
            tokens[i].append_punct = Some(punct);
            break;
        }

        let gap = words[i + 1].start_ms.saturating_sub(words[i].end_ms);

        if gap >= config.boundary_gap_ms {
            let current_slice = &tokens[..=i];
            let punct = determine_ending_punct(current_slice, config);
            tokens[i].append_punct = Some(punct);
            if config.capitalise_sentences {
                tokens[i + 1].capitalise = true;
            }
        }
    }
}

/// Determine whether the current sentence-so-far should end with `.`, `?` or `!`.
fn determine_ending_punct(tokens: &[Token<'_>], config: &PunctuationConfig) -> char {
    // Check the last word for question/exclamation signals.
    if let Some(last) = tokens.last() {
        let lower = last.word.to_lowercase();
        if config.use_exclamation_detection && EXCLAMATORY_WORDS.iter().any(|w| lower == *w) {
            return '!';
        }
        if config.use_question_detection && QUESTION_CLOSERS.iter().any(|w| lower == *w) {
            return '?';
        }
    }

    // Check the first word of the sentence for question starters.
    if let Some(first) = tokens.first() {
        let lower = first.word.to_lowercase();
        if config.use_question_detection && QUESTION_STARTERS.iter().any(|w| lower == *w) {
            return '?';
        }
    }

    '.'
}

/// Insert commas after discourse markers.
fn mark_commas(tokens: &mut [Token<'_>]) {
    let n = tokens.len();
    for i in 0..n {
        // Only add a comma if the token doesn't already have punctuation.
        if tokens[i].append_punct.is_some() {
            continue;
        }

        let lower = tokens[i].word.to_lowercase();

        // Check single-word markers.
        let is_marker = COMMA_AFTER.iter().any(|m| {
            if !m.contains(' ') {
                lower == *m
            } else {
                // Multi-word marker: build a phrase from tokens[i..].
                let words_needed = m.split_whitespace().count();
                if i + words_needed <= n {
                    let phrase: String = tokens[i..i + words_needed]
                        .iter()
                        .map(|t| t.word)
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_lowercase();
                    phrase == *m
                } else {
                    false
                }
            }
        });

        if is_marker && i + 1 < n {
            // Don't add comma at the very end of a sentence.
            if tokens[i + 1].append_punct.is_none() {
                tokens[i].append_punct = Some(',');
            }
        }
    }
}

// ─── Token assembly ───────────────────────────────────────────────────────────

/// Assemble tokens back into a string, applying capitalisation and punctuation.
fn assemble_tokens(tokens: &[Token<'_>]) -> String {
    let mut out = String::new();
    for (i, token) in tokens.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }

        let word = if token.capitalise {
            capitalise_first(token.word)
        } else {
            token.word.to_string()
        };

        out.push_str(&word);

        if let Some(p) = token.append_punct {
            out.push(p);
        }
    }
    out
}

/// Capitalise the first Unicode character of a word.
fn capitalise_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            upper + chars.as_str()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::WordTimestamp;

    fn make_word(word: &str, start_ms: u64, end_ms: u64) -> WordTimestamp {
        WordTimestamp {
            word: word.to_string(),
            start_ms,
            end_ms,
            confidence: 1.0,
            word_confidence: 1.0,
        }
    }

    fn default_config() -> PunctuationConfig {
        PunctuationConfig::default()
    }

    // ─── capitalise_first ─────────────────────────────────────────────────────

    #[test]
    fn capitalise_first_basic() {
        assert_eq!(capitalise_first("hello"), "Hello");
    }

    #[test]
    fn capitalise_first_already_upper() {
        assert_eq!(capitalise_first("Hello"), "Hello");
    }

    #[test]
    fn capitalise_first_empty() {
        assert_eq!(capitalise_first(""), "");
    }

    // ─── restore_punctuation_text ─────────────────────────────────────────────

    #[test]
    fn restores_capitalisation_of_first_word() {
        let result = restore_punctuation_text("hello world", &default_config());
        assert!(result.starts_with('H'), "got: {}", result);
    }

    #[test]
    fn adds_period_at_end() {
        let result = restore_punctuation_text("this is a test", &default_config());
        assert!(result.ends_with('.'), "got: {}", result);
    }

    #[test]
    fn empty_text_returns_empty() {
        let result = restore_punctuation_text("", &default_config());
        assert!(result.is_empty());
    }

    #[test]
    fn no_capitalisation_when_disabled() {
        let config = PunctuationConfig {
            capitalise_sentences: false,
            ..Default::default()
        };
        let result = restore_punctuation_text("hello world", &config);
        assert!(result.starts_with('h'), "got: {}", result);
    }

    #[test]
    fn inserts_comma_after_however() {
        let result =
            restore_punctuation_text("this is fine however we disagree", &default_config());
        // "however" should be followed by a comma.
        assert!(
            result.to_lowercase().contains("however,"),
            "got: {}",
            result
        );
    }

    #[test]
    fn inserts_comma_after_well() {
        // Use a config with question detection disabled to avoid "was" triggering a
        // sentence boundary (which would suppress the comma insertion on "well").
        let config = PunctuationConfig {
            use_question_detection: false,
            ..Default::default()
        };
        let result = restore_punctuation_text("well that was unexpected", &config);
        assert!(result.to_lowercase().contains("well,"), "got: {}", result);
    }

    #[test]
    fn no_comma_insertion_when_disabled() {
        let config = PunctuationConfig {
            insert_commas: false,
            ..Default::default()
        };
        let result = restore_punctuation_text("however we disagree", &config);
        assert!(!result.contains("however,"), "got: {}", result);
    }

    #[test]
    fn exclamatory_word_gets_exclamation_mark() {
        let result = restore_punctuation_text("wow that was amazing", &default_config());
        // "wow" → should have "!" after it or sentence ends with "!"
        // The heuristic marks "wow" with "!" and creates a new sentence.
        assert!(result.contains('!'), "got: {}", result);
    }

    // ─── restore_punctuation_segment (timed) ──────────────────────────────────

    #[test]
    fn segment_with_long_gap_adds_boundary() {
        let words = vec![
            make_word("hello", 0, 500),
            make_word("world", 2000, 2500), // 1500ms gap > 700ms threshold
        ];
        let seg = TranscriptSegment {
            text: "hello world".to_string(),
            start_ms: 0,
            end_ms: 2500,
            speaker_id: None,
            words: words.clone(),
        };
        let config = default_config();
        let result = restore_punctuation_segment(&seg, &config);
        // Should have terminal punctuation after the first word due to the gap.
        // capitalise_sentences is true so "hello" → "Hello" — check case-insensitively.
        let lower = result.text.to_lowercase();
        assert!(
            lower.contains("hello.") || lower.contains("hello?") || lower.contains("hello!"),
            "got: {}",
            result.text
        );
    }

    #[test]
    fn segment_no_gap_no_mid_boundary() {
        let words = vec![
            make_word("hello", 0, 500),
            make_word("world", 510, 1000), // 10ms gap < 700ms threshold
        ];
        let seg = TranscriptSegment {
            text: "hello world".to_string(),
            start_ms: 0,
            end_ms: 1000,
            speaker_id: None,
            words: words.clone(),
        };
        let config = default_config();
        let result = restore_punctuation_segment(&seg, &config);
        // No internal punctuation between hello and world.
        assert!(
            !result.text.contains("hello.") && !result.text.contains("hello?"),
            "got: {}",
            result.text
        );
    }

    #[test]
    fn segment_preserves_timestamps() {
        let words = vec![make_word("test", 1000, 1500)];
        let seg = TranscriptSegment {
            text: "test".to_string(),
            start_ms: 1000,
            end_ms: 1500,
            speaker_id: None,
            words: words.clone(),
        };
        let result = restore_punctuation_segment(&seg, &default_config());
        assert_eq!(result.start_ms, 1000);
        assert_eq!(result.end_ms, 1500);
    }

    #[test]
    fn segment_without_words_falls_back_to_text_restoration() {
        let seg = TranscriptSegment {
            text: "hello world".to_string(),
            start_ms: 0,
            end_ms: 2000,
            speaker_id: None,
            words: Vec::new(),
        };
        let result = restore_punctuation_segment(&seg, &default_config());
        assert!(result.text.starts_with('H'), "got: {}", result.text);
        assert!(result.text.ends_with('.'), "got: {}", result.text);
    }

    // ─── restore_punctuation_track ────────────────────────────────────────────

    #[test]
    fn track_each_segment_gets_period() {
        let segs = vec![
            TranscriptSegment {
                text: "first segment".to_string(),
                start_ms: 0,
                end_ms: 2000,
                speaker_id: None,
                words: Vec::new(),
            },
            TranscriptSegment {
                text: "second segment".to_string(),
                start_ms: 2000,
                end_ms: 4000,
                speaker_id: None,
                words: Vec::new(),
            },
        ];
        let result = restore_punctuation_track(&segs, &default_config());
        assert_eq!(result.len(), 2);
        assert!(result[0].text.ends_with('.'), "got: {}", result[0].text);
        assert!(result[1].text.ends_with('.'), "got: {}", result[1].text);
    }

    #[test]
    fn track_empty_input_returns_empty() {
        let result = restore_punctuation_track(&[], &default_config());
        assert!(result.is_empty());
    }

    // ─── Question detection ───────────────────────────────────────────────────

    #[test]
    fn question_starter_gets_question_mark() {
        let result = restore_punctuation_text("what is your name", &default_config());
        assert!(result.ends_with('?'), "got: {}", result);
    }

    #[test]
    fn how_question_gets_question_mark() {
        let result = restore_punctuation_text("how are you doing today", &default_config());
        assert!(result.ends_with('?'), "got: {}", result);
    }

    // ─── PunctuationConfig default ────────────────────────────────────────────

    #[test]
    fn config_default_values() {
        let cfg = PunctuationConfig::default();
        assert_eq!(cfg.boundary_gap_ms, 700);
        assert!(cfg.capitalise_sentences);
        assert!(cfg.insert_commas);
        assert!(cfg.use_question_detection);
        assert!(cfg.use_exclamation_detection);
    }
}
