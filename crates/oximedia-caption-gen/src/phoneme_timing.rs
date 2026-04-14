//! Phoneme-level timing alignment for caption display.
//!
//! This module estimates phoneme-level timing from word-level timestamps using
//! a CMU-style phoneme duration model.  Because we operate without an external
//! forced-aligner, durations are distributed proportionally by phoneme weight —
//! consonant clusters, vowels, and diphthongs are each assigned a canonical
//! *weight* and the word's time span is split accordingly.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_caption_gen::phoneme_timing::{
//!     PhonemeTimingConfig, PhonemeAligner, WordPhonemes,
//! };
//! use oximedia_caption_gen::WordTimestamp;
//!
//! let word = WordTimestamp {
//!     word: "hello".to_string(),
//!     start_ms: 0,
//!     end_ms: 400,
//!     confidence: 1.0,
//!     word_confidence: 1.0,
//! };
//! let config = PhonemeTimingConfig::default();
//! let aligner = PhonemeAligner::new(config);
//! let result = aligner.align_word(&word).unwrap();
//! assert!(!result.phonemes.is_empty());
//! ```

use crate::alignment::WordTimestamp;
use crate::CaptionGenError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// A single phoneme with its estimated timing.
#[derive(Debug, Clone, PartialEq)]
pub struct PhonemeEntry {
    /// The phoneme symbol (ARPABET-style, e.g. "HH", "EH", "L", "OW").
    pub symbol: String,
    /// Estimated start time in milliseconds.
    pub start_ms: u64,
    /// Estimated end time in milliseconds.
    pub end_ms: u64,
    /// Phoneme category.
    pub category: PhonemeCategory,
}

impl PhonemeEntry {
    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Broad phoneme categories used for duration weighting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhonemeCategory {
    /// Short vowel (e.g. IH, EH, AE, AH, AA, AO, UH).
    ShortVowel,
    /// Long vowel or diphthong (e.g. IY, EY, AY, OW, UW, OY, AW).
    LongVowel,
    /// Nasal consonant (M, N, NG).
    Nasal,
    /// Fricative consonant (F, V, TH, DH, S, Z, SH, ZH, HH).
    Fricative,
    /// Stop/plosive consonant (P, B, T, D, K, G).
    Stop,
    /// Affricate (CH, JH).
    Affricate,
    /// Liquid/approximant (L, R, W, Y).
    Liquid,
    /// Syllabic consonant or schwa-like (AX, ER).
    Syllabic,
}

impl PhonemeCategory {
    /// Relative duration weight in [1.0, 3.0].
    ///
    /// Longer categories (long vowels, syllabics) get a higher weight so their
    /// estimated duration within a word is proportionally longer.
    pub fn duration_weight(&self) -> f64 {
        match self {
            PhonemeCategory::ShortVowel => 2.0,
            PhonemeCategory::LongVowel => 3.0,
            PhonemeCategory::Nasal => 1.5,
            PhonemeCategory::Fricative => 1.5,
            PhonemeCategory::Stop => 1.0,
            PhonemeCategory::Affricate => 1.8,
            PhonemeCategory::Liquid => 1.2,
            PhonemeCategory::Syllabic => 2.5,
        }
    }
}

/// A word broken down into estimated phoneme timings.
#[derive(Debug, Clone)]
pub struct WordPhonemes {
    /// The source word.
    pub word: String,
    /// Start time of the word.
    pub start_ms: u64,
    /// End time of the word.
    pub end_ms: u64,
    /// Individual phoneme entries in order.
    pub phonemes: Vec<PhonemeEntry>,
}

/// Configuration for the phoneme aligner.
#[derive(Debug, Clone)]
pub struct PhonemeTimingConfig {
    /// Minimum phoneme duration in milliseconds (prevents zero-duration slices).
    pub min_phoneme_ms: u64,
    /// Whether to apply a short closure gap before stop consonants.
    pub model_stop_closure: bool,
    /// Fraction of a stop's allotted duration assigned to its closure phase [0.0, 0.5].
    pub stop_closure_fraction: f64,
}

impl Default for PhonemeTimingConfig {
    fn default() -> Self {
        Self {
            min_phoneme_ms: 10,
            model_stop_closure: true,
            stop_closure_fraction: 0.3,
        }
    }
}

// ─── CMU phoneme lexicon (subset) ────────────────────────────────────────────

/// A simple English grapheme-to-phoneme mapping for common words and letter
/// clusters.  Returns an ARPABET sequence as a `Vec<(symbol, category)>`.
///
/// For words not in the lexicon, a letter-by-letter grapheme heuristic is used.
fn lookup_phonemes(word: &str) -> Vec<(String, PhonemeCategory)> {
    let lower: String = word
        .chars()
        .filter(|c| c.is_alphabetic())
        .map(|c| c.to_ascii_lowercase())
        .collect();

    // Small static lexicon for common words.
    match lower.as_str() {
        "the" => return ph_seq(&[("DH", FRIC), ("AH", SHORT)]),
        "a" => return ph_seq(&[("AH", SHORT)]),
        "and" => return ph_seq(&[("AE", SHORT), ("N", NASAL), ("D", STOP)]),
        "in" => return ph_seq(&[("IH", SHORT), ("N", NASAL)]),
        "it" => return ph_seq(&[("IH", SHORT), ("T", STOP)]),
        "is" => return ph_seq(&[("IH", SHORT), ("Z", FRIC)]),
        "to" => return ph_seq(&[("T", STOP), ("UW", LONG)]),
        "of" => return ph_seq(&[("AH", SHORT), ("V", FRIC)]),
        "was" => return ph_seq(&[("W", LIQUID), ("AH", SHORT), ("Z", FRIC)]),
        "for" => return ph_seq(&[("F", FRIC), ("AO", SHORT), ("R", LIQUID)]),
        "on" => return ph_seq(&[("AA", SHORT), ("N", NASAL)]),
        "are" => return ph_seq(&[("AA", SHORT), ("R", LIQUID)]),
        "as" => return ph_seq(&[("AE", SHORT), ("Z", FRIC)]),
        "with" => return ph_seq(&[("W", LIQUID), ("IH", SHORT), ("TH", FRIC)]),
        "his" => return ph_seq(&[("HH", FRIC), ("IH", SHORT), ("Z", FRIC)]),
        "that" => {
            return ph_seq(&[("DH", FRIC), ("AE", SHORT), ("T", STOP)]);
        }
        "he" => return ph_seq(&[("HH", FRIC), ("IY", LONG)]),
        "she" => return ph_seq(&[("SH", FRIC), ("IY", LONG)]),
        "they" => return ph_seq(&[("DH", FRIC), ("EY", LONG)]),
        "we" => return ph_seq(&[("W", LIQUID), ("IY", LONG)]),
        "you" => return ph_seq(&[("Y", LIQUID), ("UW", LONG)]),
        "have" => return ph_seq(&[("HH", FRIC), ("AE", SHORT), ("V", FRIC)]),
        "not" => return ph_seq(&[("N", NASAL), ("AA", SHORT), ("T", STOP)]),
        "this" => return ph_seq(&[("DH", FRIC), ("IH", SHORT), ("S", FRIC)]),
        "but" => return ph_seq(&[("B", STOP), ("AH", SHORT), ("T", STOP)]),
        "from" => {
            return ph_seq(&[("F", FRIC), ("R", LIQUID), ("AH", SHORT), ("M", NASAL)]);
        }
        "hello" => {
            return ph_seq(&[("HH", FRIC), ("EH", SHORT), ("L", LIQUID), ("OW", LONG)]);
        }
        "world" => {
            return ph_seq(&[("W", LIQUID), ("ER", SYL), ("L", LIQUID), ("D", STOP)]);
        }
        "caption" => {
            return ph_seq(&[
                ("K", STOP),
                ("AE", SHORT),
                ("P", STOP),
                ("SH", FRIC),
                ("AH", SHORT),
                ("N", NASAL),
            ]);
        }
        "subtitle" => {
            return ph_seq(&[
                ("S", FRIC),
                ("AH", SHORT),
                ("B", STOP),
                ("T", STOP),
                ("AY", LONG),
                ("T", STOP),
                ("AH", SHORT),
                ("L", LIQUID),
            ]);
        }
        _ => {} // fall through to heuristic
    }

    grapheme_heuristic(&lower)
}

// Type aliases for briefer code inside lookup_phonemes.
use PhonemeCategory as Cat;
const SHORT: Cat = Cat::ShortVowel;
const LONG: Cat = Cat::LongVowel;
const NASAL: Cat = Cat::Nasal;
const FRIC: Cat = Cat::Fricative;
const STOP: Cat = Cat::Stop;
const LIQUID: Cat = Cat::Liquid;
const SYL: Cat = Cat::Syllabic;

fn ph_seq(pairs: &[(&str, PhonemeCategory)]) -> Vec<(String, PhonemeCategory)> {
    pairs
        .iter()
        .map(|(s, c)| (s.to_string(), c.clone()))
        .collect()
}

/// Letter-by-letter grapheme heuristic for unknown words.
///
/// Produces a reasonable ARPABET-like sequence by mapping common letter
/// patterns (digraphs first, then single characters) to phoneme categories.
fn grapheme_heuristic(word: &str) -> Vec<(String, PhonemeCategory)> {
    let chars: Vec<char> = word.chars().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        // Try digraphs first.
        if i + 1 < chars.len() {
            let digraph: String = chars[i..=i + 1].iter().collect();
            if let Some(entry) = digraph_phoneme(&digraph) {
                result.push(entry);
                i += 2;
                continue;
            }
        }
        // Single character.
        let ch = chars[i];
        result.push(char_phoneme(ch));
        i += 1;
    }

    result
}

fn digraph_phoneme(dg: &str) -> Option<(String, PhonemeCategory)> {
    match dg {
        "sh" => Some(("SH".to_string(), PhonemeCategory::Fricative)),
        "ch" => Some(("CH".to_string(), PhonemeCategory::Affricate)),
        "th" => Some(("TH".to_string(), PhonemeCategory::Fricative)),
        "ng" => Some(("NG".to_string(), PhonemeCategory::Nasal)),
        "ck" => Some(("K".to_string(), PhonemeCategory::Stop)),
        "ph" => Some(("F".to_string(), PhonemeCategory::Fricative)),
        "gh" => Some(("G".to_string(), PhonemeCategory::Stop)),
        "wh" => Some(("W".to_string(), PhonemeCategory::Liquid)),
        "qu" => Some(("K".to_string(), PhonemeCategory::Stop)),
        "ai" | "ay" => Some(("EY".to_string(), PhonemeCategory::LongVowel)),
        "ee" | "ea" => Some(("IY".to_string(), PhonemeCategory::LongVowel)),
        "oo" => Some(("UW".to_string(), PhonemeCategory::LongVowel)),
        "oa" | "ow" => Some(("OW".to_string(), PhonemeCategory::LongVowel)),
        "oi" | "oy" => Some(("OY".to_string(), PhonemeCategory::LongVowel)),
        "ou" => Some(("AW".to_string(), PhonemeCategory::LongVowel)),
        "ie" => Some(("IY".to_string(), PhonemeCategory::LongVowel)),
        "er" | "ir" | "ur" => Some(("ER".to_string(), PhonemeCategory::Syllabic)),
        _ => None,
    }
}

fn char_phoneme(ch: char) -> (String, PhonemeCategory) {
    match ch {
        'a' => ("AE".to_string(), PhonemeCategory::ShortVowel),
        'e' => ("EH".to_string(), PhonemeCategory::ShortVowel),
        'i' => ("IH".to_string(), PhonemeCategory::ShortVowel),
        'o' => ("AA".to_string(), PhonemeCategory::ShortVowel),
        'u' => ("AH".to_string(), PhonemeCategory::ShortVowel),
        'b' => ("B".to_string(), PhonemeCategory::Stop),
        'c' => ("K".to_string(), PhonemeCategory::Stop),
        'd' => ("D".to_string(), PhonemeCategory::Stop),
        'f' => ("F".to_string(), PhonemeCategory::Fricative),
        'g' => ("G".to_string(), PhonemeCategory::Stop),
        'h' => ("HH".to_string(), PhonemeCategory::Fricative),
        'j' => ("JH".to_string(), PhonemeCategory::Affricate),
        'k' => ("K".to_string(), PhonemeCategory::Stop),
        'l' => ("L".to_string(), PhonemeCategory::Liquid),
        'm' => ("M".to_string(), PhonemeCategory::Nasal),
        'n' => ("N".to_string(), PhonemeCategory::Nasal),
        'p' => ("P".to_string(), PhonemeCategory::Stop),
        'q' => ("K".to_string(), PhonemeCategory::Stop),
        'r' => ("R".to_string(), PhonemeCategory::Liquid),
        's' => ("S".to_string(), PhonemeCategory::Fricative),
        't' => ("T".to_string(), PhonemeCategory::Stop),
        'v' => ("V".to_string(), PhonemeCategory::Fricative),
        'w' => ("W".to_string(), PhonemeCategory::Liquid),
        'x' => ("K".to_string(), PhonemeCategory::Stop),
        'y' => ("Y".to_string(), PhonemeCategory::Liquid),
        'z' => ("Z".to_string(), PhonemeCategory::Fricative),
        _ => ("AH".to_string(), PhonemeCategory::ShortVowel),
    }
}

// ─── PhonemeAligner ───────────────────────────────────────────────────────────

/// Aligns phonemes to word-level timestamps by distributing the word's duration
/// proportionally among the phonemes according to their duration weights.
pub struct PhonemeAligner {
    config: PhonemeTimingConfig,
}

impl PhonemeAligner {
    /// Create a new aligner with the given configuration.
    pub fn new(config: PhonemeTimingConfig) -> Self {
        Self { config }
    }

    /// Estimate phoneme timings for a single word.
    ///
    /// # Errors
    /// Returns [`CaptionGenError::InvalidTimestamp`] if `word.start_ms >= word.end_ms`.
    pub fn align_word(&self, word: &WordTimestamp) -> Result<WordPhonemes, CaptionGenError> {
        if word.start_ms >= word.end_ms {
            return Err(CaptionGenError::InvalidTimestamp);
        }

        let phoneme_specs = lookup_phonemes(&word.word);
        if phoneme_specs.is_empty() {
            return Err(CaptionGenError::InvalidParameter(format!(
                "no phonemes found for word '{}'",
                word.word
            )));
        }

        let total_weight: f64 = phoneme_specs.iter().map(|(_, c)| c.duration_weight()).sum();
        let word_duration = (word.end_ms - word.start_ms) as f64;

        let mut phonemes = Vec::with_capacity(phoneme_specs.len());
        let mut cursor_ms = word.start_ms;

        for (idx, (symbol, category)) in phoneme_specs.iter().enumerate() {
            let is_last = idx + 1 == phoneme_specs.len();

            let raw_duration = if is_last {
                // Last phoneme consumes remaining time to avoid drift.
                word.end_ms.saturating_sub(cursor_ms) as f64
            } else {
                category.duration_weight() / total_weight * word_duration
            };

            let duration_ms = (raw_duration.round() as u64).max(self.config.min_phoneme_ms);

            // For stop consonants: optionally split into closure + burst.
            if self.config.model_stop_closure && *category == PhonemeCategory::Stop && !is_last {
                let closure_ms = ((duration_ms as f64 * self.config.stop_closure_fraction).round()
                    as u64)
                    .max(self.config.min_phoneme_ms);
                let burst_ms = duration_ms.saturating_sub(closure_ms);

                let closure_end = cursor_ms + closure_ms;
                let burst_end = (cursor_ms + duration_ms).min(word.end_ms);

                // closure phase
                phonemes.push(PhonemeEntry {
                    symbol: format!("{}_cl", symbol),
                    start_ms: cursor_ms,
                    end_ms: closure_end,
                    category: category.clone(),
                });
                // burst phase
                let burst_start = closure_end;
                let burst_end_clamped = (burst_start + burst_ms).min(word.end_ms);
                phonemes.push(PhonemeEntry {
                    symbol: symbol.clone(),
                    start_ms: burst_start,
                    end_ms: burst_end_clamped,
                    category: category.clone(),
                });
                cursor_ms = burst_end.min(word.end_ms);
            } else {
                let end_ms = (cursor_ms + duration_ms).min(word.end_ms);
                phonemes.push(PhonemeEntry {
                    symbol: symbol.clone(),
                    start_ms: cursor_ms,
                    end_ms: end_ms,
                    category: category.clone(),
                });
                cursor_ms = end_ms;
            }

            if cursor_ms >= word.end_ms {
                break;
            }
        }

        // Ensure the final phoneme always reaches word.end_ms.
        if let Some(last) = phonemes.last_mut() {
            if last.end_ms < word.end_ms {
                last.end_ms = word.end_ms;
            }
        }

        Ok(WordPhonemes {
            word: word.word.clone(),
            start_ms: word.start_ms,
            end_ms: word.end_ms,
            phonemes,
        })
    }

    /// Align a slice of words, returning one [`WordPhonemes`] per word.
    ///
    /// Words with invalid timestamps (start >= end) are skipped; the error
    /// count is returned alongside the successful results.
    pub fn align_words(&self, words: &[WordTimestamp]) -> (Vec<WordPhonemes>, usize) {
        let mut results = Vec::with_capacity(words.len());
        let mut errors = 0usize;
        for w in words {
            match self.align_word(w) {
                Ok(wp) => results.push(wp),
                Err(_) => errors += 1,
            }
        }
        (results, errors)
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Count the number of vowel phonemes in a [`WordPhonemes`] result, which
/// corresponds roughly to the syllable count.
pub fn count_syllables(wp: &WordPhonemes) -> usize {
    wp.phonemes
        .iter()
        .filter(|p| {
            matches!(
                p.category,
                PhonemeCategory::ShortVowel
                    | PhonemeCategory::LongVowel
                    | PhonemeCategory::Syllabic
            )
        })
        .count()
}

/// Compute the average phoneme duration in milliseconds for a word.
pub fn average_phoneme_duration_ms(wp: &WordPhonemes) -> f64 {
    if wp.phonemes.is_empty() {
        return 0.0;
    }
    let total: u64 = wp.phonemes.iter().map(|p| p.duration_ms()).sum();
    total as f64 / wp.phonemes.len() as f64
}

/// Extract only the vowel phonemes from a [`WordPhonemes`].
pub fn vowel_phonemes(wp: &WordPhonemes) -> Vec<&PhonemeEntry> {
    wp.phonemes
        .iter()
        .filter(|p| {
            matches!(
                p.category,
                PhonemeCategory::ShortVowel
                    | PhonemeCategory::LongVowel
                    | PhonemeCategory::Syllabic
            )
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_word(word: &str, start_ms: u64, end_ms: u64) -> WordTimestamp {
        WordTimestamp {
            word: word.to_string(),
            start_ms,
            end_ms,
            confidence: 1.0,
            word_confidence: 1.0,
        }
    }

    #[test]
    fn align_hello_produces_phonemes() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("hello", 0, 400);
        let result = aligner.align_word(&word).expect("should align");
        assert!(!result.phonemes.is_empty());
        assert_eq!(result.word, "hello");
    }

    #[test]
    fn phonemes_cover_full_word_duration() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("hello", 100, 500);
        let result = aligner.align_word(&word).expect("should align");
        let first_start = result.phonemes.first().map(|p| p.start_ms).unwrap_or(0);
        let last_end = result.phonemes.last().map(|p| p.end_ms).unwrap_or(0);
        assert_eq!(first_start, 100);
        assert_eq!(last_end, 500);
    }

    #[test]
    fn invalid_timestamp_returns_error() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("hello", 500, 100); // start > end
        assert!(aligner.align_word(&word).is_err());
    }

    #[test]
    fn lexicon_word_phonemes_correct_category() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("the", 0, 200);
        let result = aligner.align_word(&word).expect("should align");
        // "the" = DH (fricative) + AH (short vowel)
        assert!(result.phonemes.iter().any(|p| p.symbol == "DH"));
    }

    #[test]
    fn stop_closure_splits_stop_phoneme() {
        let config = PhonemeTimingConfig {
            model_stop_closure: true,
            stop_closure_fraction: 0.4,
            min_phoneme_ms: 5,
        };
        let aligner = PhonemeAligner::new(config);
        let word = make_word("but", 0, 300);
        let result = aligner.align_word(&word).expect("should align");
        // "but" starts with B (stop) — expect a B_cl closure entry.
        let has_closure = result.phonemes.iter().any(|p| p.symbol.ends_with("_cl"));
        assert!(has_closure, "expected closure phase for stop consonant");
    }

    #[test]
    fn no_stop_closure_when_disabled() {
        let config = PhonemeTimingConfig {
            model_stop_closure: false,
            stop_closure_fraction: 0.3,
            min_phoneme_ms: 10,
        };
        let aligner = PhonemeAligner::new(config);
        let word = make_word("but", 0, 300);
        let result = aligner.align_word(&word).expect("should align");
        let has_closure = result.phonemes.iter().any(|p| p.symbol.ends_with("_cl"));
        assert!(!has_closure, "closure should be disabled");
    }

    #[test]
    fn count_syllables_hello() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("hello", 0, 400);
        let wp = aligner.align_word(&word).expect("should align");
        let syllables = count_syllables(&wp);
        // "hello" has 2 vowels: EH + OW
        assert_eq!(syllables, 2);
    }

    #[test]
    fn align_words_batch_skips_invalid() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let words = vec![
            make_word("hello", 0, 400),
            make_word("bad", 500, 200), // invalid
            make_word("world", 600, 1000),
        ];
        let (results, errors) = aligner.align_words(&words);
        assert_eq!(results.len(), 2);
        assert_eq!(errors, 1);
    }

    #[test]
    fn average_phoneme_duration_nonzero() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("world", 0, 500);
        let wp = aligner.align_word(&word).expect("should align");
        let avg = average_phoneme_duration_ms(&wp);
        assert!(avg > 0.0);
    }

    #[test]
    fn grapheme_heuristic_unknown_word() {
        // "zzz" is not in lexicon; heuristic should still produce phonemes.
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("zzz", 0, 300);
        let result = aligner.align_word(&word).expect("should align");
        assert!(!result.phonemes.is_empty());
    }

    #[test]
    fn vowel_phonemes_returns_only_vowels() {
        let aligner = PhonemeAligner::new(PhonemeTimingConfig::default());
        let word = make_word("hello", 0, 400);
        let wp = aligner.align_word(&word).expect("should align");
        let vowels = vowel_phonemes(&wp);
        for v in &vowels {
            assert!(matches!(
                v.category,
                PhonemeCategory::ShortVowel
                    | PhonemeCategory::LongVowel
                    | PhonemeCategory::Syllabic
            ));
        }
    }
}
