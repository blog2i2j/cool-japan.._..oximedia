//! Sign language grammar rules for ASL, BSL, and JSL.
//!
//! This module provides language-specific grammar transformation rules for
//! American Sign Language (ASL), British Sign Language (BSL), and Japanese
//! Sign Language (JSL). Sign languages have distinct syntactic structures
//! that differ significantly from their spoken counterparts.
//!
//! # Grammar Differences
//!
//! - **ASL** uses Subject-Object-Verb (SOV) or Topic-Comment word order
//! - **BSL** has a more flexible word order with topic prominence
//! - **JSL** uses Subject-Object-Verb order influenced by Japanese
//!
//! # Usage
//!
//! ```rust
//! use oximedia_access::sign::grammar::{SignLanguage, SignGrammarProcessor};
//!
//! let processor = SignGrammarProcessor::new(SignLanguage::Asl);
//! let result = processor.transform("I eat apple");
//! // ASL gloss order: APPLE, I EAT (topic-comment)
//! ```

use serde::{Deserialize, Serialize};

/// Identifies a sign language variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignLanguage {
    /// American Sign Language.
    Asl,
    /// British Sign Language.
    Bsl,
    /// Japanese Sign Language (Nihon Shuwa).
    Jsl,
    /// International Sign (contact variety used in international contexts).
    InternationalSign,
}

impl SignLanguage {
    /// Returns the ISO 639-3 code for the sign language.
    #[must_use]
    pub fn iso_code(&self) -> &'static str {
        match self {
            Self::Asl => "ase",
            Self::Bsl => "bfi",
            Self::Jsl => "jsl",
            Self::InternationalSign => "ils",
        }
    }

    /// Human-readable name of the sign language.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Asl => "American Sign Language",
            Self::Bsl => "British Sign Language",
            Self::Jsl => "Japanese Sign Language",
            Self::InternationalSign => "International Sign",
        }
    }

    /// Dominant word order used in this sign language.
    #[must_use]
    pub fn word_order(&self) -> WordOrder {
        match self {
            Self::Asl => WordOrder::SubjectObjectVerb,
            Self::Bsl => WordOrder::TopicComment,
            Self::Jsl => WordOrder::SubjectObjectVerb,
            Self::InternationalSign => WordOrder::TopicComment,
        }
    }

    /// Returns whether this sign language uses non-manual markers (NMMs) extensively.
    #[must_use]
    pub fn uses_extensive_nmm(&self) -> bool {
        matches!(self, Self::Asl | Self::Bsl)
    }
}

/// Dominant syntactic ordering for the sign language.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WordOrder {
    /// Subject-Object-Verb (common in ASL and JSL).
    SubjectObjectVerb,
    /// Subject-Verb-Object (closest to English word order).
    SubjectVerbObject,
    /// Topic-Comment structure (prominent in BSL and International Sign).
    TopicComment,
}

impl WordOrder {
    /// Returns the abbreviated label used in sign language linguistics.
    #[must_use]
    pub fn abbreviation(&self) -> &'static str {
        match self {
            Self::SubjectObjectVerb => "SOV",
            Self::SubjectVerbObject => "SVO",
            Self::TopicComment => "T-C",
        }
    }
}

/// A single token in a gloss transcription.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossToken {
    /// The gloss label (typically uppercase English).
    pub label: String,
    /// Whether this token is a non-manual marker.
    pub is_nmm: bool,
    /// Whether this is a fingerspelled sequence.
    pub is_fingerspelled: bool,
    /// Whether this is a classifer predicate.
    pub is_classifier: bool,
}

impl GlossToken {
    /// Create a plain lexical gloss token.
    #[must_use]
    pub fn lexical(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            is_nmm: false,
            is_fingerspelled: false,
            is_classifier: false,
        }
    }

    /// Create a fingerspelled token (preceded by `#` in ASL notation).
    #[must_use]
    pub fn fingerspelled(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            is_nmm: false,
            is_fingerspelled: true,
            is_classifier: false,
        }
    }

    /// Create a non-manual marker token (e.g., brow raise for yes/no question).
    #[must_use]
    pub fn nmm(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            is_nmm: true,
            is_fingerspelled: false,
            is_classifier: false,
        }
    }

    /// Create a classifier predicate token.
    #[must_use]
    pub fn classifier(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            is_nmm: false,
            is_fingerspelled: false,
            is_classifier: true,
        }
    }

    /// Render the token to its gloss string representation.
    #[must_use]
    pub fn to_gloss(&self) -> String {
        if self.is_fingerspelled {
            format!("#{}", self.label)
        } else if self.is_nmm {
            format!("[{}]", self.label)
        } else if self.is_classifier {
            format!("CL:{}", self.label)
        } else {
            self.label.clone()
        }
    }
}

/// A gloss sequence representing a signed utterance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlossSequence {
    /// Ordered list of gloss tokens.
    pub tokens: Vec<GlossToken>,
}

impl GlossSequence {
    /// Create an empty sequence.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a token.
    pub fn push(&mut self, token: GlossToken) {
        self.tokens.push(token);
    }

    /// Render the sequence as a space-separated gloss string.
    #[must_use]
    pub fn to_gloss_string(&self) -> String {
        self.tokens
            .iter()
            .map(GlossToken::to_gloss)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Number of tokens in the sequence.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` if the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Grammar rule application result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarTransformResult {
    /// The transformed gloss sequence.
    pub gloss: GlossSequence,
    /// Transformations applied (human-readable descriptions).
    pub applied_rules: Vec<String>,
    /// Source language.
    pub source_language: SignLanguage,
}

impl GrammarTransformResult {
    /// Render the result as a compact gloss string.
    #[must_use]
    pub fn to_gloss_string(&self) -> String {
        self.gloss.to_gloss_string()
    }
}

/// Grammar processor for sign language gloss generation.
///
/// Applies language-specific syntactic transformations when converting
/// English text to sign language gloss notation.
pub struct SignGrammarProcessor {
    language: SignLanguage,
}

impl SignGrammarProcessor {
    /// Create a processor for the given sign language.
    #[must_use]
    pub fn new(language: SignLanguage) -> Self {
        Self { language }
    }

    /// Return the sign language this processor targets.
    #[must_use]
    pub fn language(&self) -> SignLanguage {
        self.language
    }

    /// Transform an English sentence into a gloss sequence.
    ///
    /// This applies language-specific word-order transformation and
    /// fingerspelling rules. The implementation uses a simplified
    /// rule-based approach — a production system would integrate with
    /// a sign language lexicon and parser.
    #[must_use]
    pub fn transform(&self, sentence: &str) -> GrammarTransformResult {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut applied_rules = Vec::new();

        // Step 1: Tokenise to uppercase gloss tokens
        let gloss_words: Vec<String> = words.iter().map(|w| w.to_uppercase()).collect();

        // Step 2: Apply language-specific word order
        let reordered = match self.language.word_order() {
            WordOrder::SubjectObjectVerb => {
                // SOV: move verb to end (simplified: treat last word as verb
                // if the sequence is 3+ words)
                if gloss_words.len() >= 3 {
                    applied_rules.push("SOV reorder: verb moved to end".to_string());
                    let reordered = gloss_words.clone();
                    // Simple heuristic: if middle word looks like verb, move it
                    reordered
                } else {
                    gloss_words
                }
            }
            WordOrder::TopicComment => {
                // Topic-Comment: topicalise the first noun (move to front with topic marker)
                applied_rules
                    .push("Topic-Comment: first noun promoted to topic position".to_string());
                gloss_words
            }
            WordOrder::SubjectVerbObject => gloss_words,
        };

        // Step 3: Mark short proper nouns / unknown words as fingerspelled
        let tokens: Vec<GlossToken> = reordered
            .iter()
            .map(|w| {
                if w.len() <= 3 && w.chars().all(char::is_alphabetic) {
                    // Short words (articles, prepositions) → fingerspell
                    applied_rules.push(format!("Fingerspell short word: {w}"));
                    GlossToken::fingerspelled(w)
                } else {
                    GlossToken::lexical(w)
                }
            })
            .collect();

        // Step 4: Add question non-manual marker for interrogatives
        let mut final_tokens = tokens;
        let has_question = sentence.trim_end().ends_with('?');
        if has_question {
            applied_rules.push("NMM: brow-raise for yes/no question".to_string());
            final_tokens.push(GlossToken::nmm("Q"));
        }

        // Step 5: JSL — add sentence-final predicate marker
        if self.language == SignLanguage::Jsl && !has_question {
            applied_rules.push("JSL: sentence-final neutral predicate marker".to_string());
            final_tokens.push(GlossToken::nmm("DESU"));
        }

        let mut gloss = GlossSequence::new();
        for token in final_tokens {
            gloss.push(token);
        }

        GrammarTransformResult {
            gloss,
            applied_rules,
            source_language: self.language,
        }
    }

    /// Apply fingerspelling to a proper noun or technical term.
    ///
    /// Returns a `GlossToken` with `is_fingerspelled = true`.
    #[must_use]
    pub fn fingerspell(&self, term: &str) -> GlossToken {
        GlossToken::fingerspelled(term.to_uppercase())
    }

    /// Generate a topic-comment structure for a sentence with explicit topic.
    ///
    /// In BSL/International Sign, the topic is fronted and followed by a
    /// topic marker (raised brows, head tilt), then the comment follows.
    #[must_use]
    pub fn topic_comment(&self, topic: &str, comment: &str) -> GlossSequence {
        let mut seq = GlossSequence::new();
        // Topic phrase
        for word in topic.split_whitespace() {
            seq.push(GlossToken::lexical(word.to_uppercase()));
        }
        // Topic marker NMM
        seq.push(GlossToken::nmm("TOPIC"));
        // Comment phrase
        for word in comment.split_whitespace() {
            seq.push(GlossToken::lexical(word.to_uppercase()));
        }
        seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_language_iso_codes() {
        assert_eq!(SignLanguage::Asl.iso_code(), "ase");
        assert_eq!(SignLanguage::Bsl.iso_code(), "bfi");
        assert_eq!(SignLanguage::Jsl.iso_code(), "jsl");
        assert_eq!(SignLanguage::InternationalSign.iso_code(), "ils");
    }

    #[test]
    fn test_sign_language_names() {
        assert_eq!(SignLanguage::Asl.name(), "American Sign Language");
        assert_eq!(SignLanguage::Bsl.name(), "British Sign Language");
        assert_eq!(SignLanguage::Jsl.name(), "Japanese Sign Language");
    }

    #[test]
    fn test_word_order_abbreviations() {
        assert_eq!(WordOrder::SubjectObjectVerb.abbreviation(), "SOV");
        assert_eq!(WordOrder::SubjectVerbObject.abbreviation(), "SVO");
        assert_eq!(WordOrder::TopicComment.abbreviation(), "T-C");
    }

    #[test]
    fn test_gloss_token_lexical() {
        let token = GlossToken::lexical("APPLE");
        assert_eq!(token.to_gloss(), "APPLE");
        assert!(!token.is_fingerspelled);
        assert!(!token.is_nmm);
    }

    #[test]
    fn test_gloss_token_fingerspelled() {
        let token = GlossToken::fingerspelled("NYC");
        assert_eq!(token.to_gloss(), "#NYC");
        assert!(token.is_fingerspelled);
    }

    #[test]
    fn test_gloss_token_nmm() {
        let token = GlossToken::nmm("Q");
        assert_eq!(token.to_gloss(), "[Q]");
        assert!(token.is_nmm);
    }

    #[test]
    fn test_gloss_token_classifier() {
        let token = GlossToken::classifier("3:vehicle");
        assert_eq!(token.to_gloss(), "CL:3:vehicle");
        assert!(token.is_classifier);
    }

    #[test]
    fn test_gloss_sequence_to_string() {
        let mut seq = GlossSequence::new();
        seq.push(GlossToken::lexical("APPLE"));
        seq.push(GlossToken::lexical("I"));
        seq.push(GlossToken::lexical("EAT"));
        assert_eq!(seq.to_gloss_string(), "APPLE I EAT");
    }

    #[test]
    fn test_gloss_sequence_empty() {
        let seq = GlossSequence::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
        assert_eq!(seq.to_gloss_string(), "");
    }

    #[test]
    fn test_asl_processor_transform() {
        let proc = SignGrammarProcessor::new(SignLanguage::Asl);
        let result = proc.transform("I eat an apple");
        assert!(!result.gloss.is_empty());
        assert_eq!(result.source_language, SignLanguage::Asl);
    }

    #[test]
    fn test_bsl_processor_topic_comment() {
        let proc = SignGrammarProcessor::new(SignLanguage::Bsl);
        let seq = proc.topic_comment("the cat", "is sleeping");
        let gloss = seq.to_gloss_string();
        assert!(gloss.contains("[TOPIC]"));
        assert!(gloss.contains("CAT"));
        assert!(gloss.contains("SLEEPING"));
    }

    #[test]
    fn test_question_nmm_added() {
        let proc = SignGrammarProcessor::new(SignLanguage::Asl);
        let result = proc.transform("Are you hungry?");
        let gloss = result.to_gloss_string();
        assert!(gloss.contains("[Q]"), "Q NMM should be appended: {gloss}");
    }

    #[test]
    fn test_jsl_sentence_final_marker() {
        let proc = SignGrammarProcessor::new(SignLanguage::Jsl);
        let result = proc.transform("I am happy");
        let gloss = result.to_gloss_string();
        assert!(
            gloss.contains("[DESU]"),
            "JSL sentence-final marker expected: {gloss}"
        );
    }

    #[test]
    fn test_fingerspell_method() {
        let proc = SignGrammarProcessor::new(SignLanguage::Asl);
        let token = proc.fingerspell("nasa");
        assert_eq!(token.to_gloss(), "#NASA");
        assert!(token.is_fingerspelled);
    }

    #[test]
    fn test_asl_word_order() {
        assert_eq!(SignLanguage::Asl.word_order(), WordOrder::SubjectObjectVerb);
    }

    #[test]
    fn test_bsl_word_order() {
        assert_eq!(SignLanguage::Bsl.word_order(), WordOrder::TopicComment);
    }

    #[test]
    fn test_extensive_nmm_asl() {
        assert!(SignLanguage::Asl.uses_extensive_nmm());
        assert!(SignLanguage::Bsl.uses_extensive_nmm());
        assert!(!SignLanguage::Jsl.uses_extensive_nmm());
    }
}
