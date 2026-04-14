//! Forced narrative (FN) and SDH (Subtitles for the Deaf and Hard-of-hearing)
//! subtitle detection and classification.
//!
//! Forced narrative subtitles are captions that are always displayed regardless
//! of user preference — typically for foreign-language dialogue, on-screen text,
//! or sound effects that are essential to understanding the content.
//!
//! SDH subtitles extend captions with speaker identification and non-speech
//! audio descriptions (e.g., `[MUSIC]`, `[DOOR SLAMS]`).
//!
//! ## Classification pipeline
//!
//! 1. [`classify_block`] examines a [`CaptionBlock`] for FN/SDH markers.
//! 2. [`FnSdhClassifier`] runs the full pipeline over a caption track.
//! 3. Results are returned as [`FnSdhAnnotation`] values attached to block IDs.

use crate::alignment::CaptionBlock;

// ─── Annotation types ─────────────────────────────────────────────────────────

/// The kind of forced narrative / SDH annotation assigned to a caption block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FnSdhKind {
    /// Normal caption — no forced-narrative flag.
    Normal,
    /// Forced narrative: must be displayed regardless of user caption preference.
    /// Typical use: foreign-language dialogue, on-screen text, essential plot info.
    ForcedNarrative,
    /// SDH sound-effect descriptor, e.g. `[DOOR SLAMS]`, `(MUSIC PLAYING)`.
    SoundEffect,
    /// SDH speaker identification prefix, e.g. `ALICE:` or `[DR. SMITH]`.
    SpeakerLabel,
    /// On-screen text (titles, signs, credits rendered as captions).
    OnScreenText,
}

/// An annotation assigned to a single caption block.
#[derive(Debug, Clone, PartialEq)]
pub struct FnSdhAnnotation {
    /// The 1-based block ID this annotation refers to.
    pub block_id: u32,
    /// Classification result.
    pub kind: FnSdhKind,
    /// Human-readable explanation of why this classification was assigned.
    pub reason: String,
    /// Confidence of the classification in \[0.0, 1.0\].
    pub confidence: f32,
}

// ─── Heuristic patterns ───────────────────────────────────────────────────────

/// Brackets indicating sound effects or descriptions: `[...]` or `(...)`.
fn is_bracketed_description(text: &str) -> bool {
    let trimmed = text.trim();
    (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
}

/// Check whether every line in the block is wrapped in brackets/parentheses.
fn all_lines_bracketed(block: &CaptionBlock) -> bool {
    !block.lines.is_empty() && block.lines.iter().all(|l| is_bracketed_description(l))
}

/// Check for a speaker-label pattern: a line ending with `:` (optionally in brackets).
fn has_speaker_label_line(block: &CaptionBlock) -> bool {
    block.lines.iter().any(|line| {
        let t = line.trim();
        // "[ALICE]:" or "ALICE:" patterns — all-uppercase word(s) followed by colon.
        if t.ends_with(':') {
            let before_colon = t.trim_end_matches(':').trim();
            // Strip surrounding brackets if present.
            let label = before_colon
                .trim_start_matches('[')
                .trim_end_matches(']')
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim();
            // Accept if label contains only uppercase letters, digits, spaces, dots, hyphens.
            !label.is_empty()
                && label
                    .chars()
                    .all(|c| c.is_uppercase() || c.is_ascii_digit() || " .-_".contains(c))
        } else {
            false
        }
    })
}

/// Detect common sound-effect keywords inside bracket patterns.
static SOUND_EFFECT_KEYWORDS: &[&str] = &[
    "music",
    "applause",
    "laughter",
    "crowd",
    "noise",
    "sound",
    "silence",
    "door",
    "phone",
    "alarm",
    "siren",
    "explosion",
    "thunder",
    "rain",
    "wind",
    "typing",
    "footsteps",
    "knock",
    "crash",
    "beep",
    "buzz",
    "click",
    "bang",
    "slam",
    "whistle",
    "song",
    "singing",
    "humming",
    "crying",
    "screaming",
    "shouting",
    "whispering",
    "laughing",
    "sobbing",
    "gasp",
    "sigh",
    "chime",
    "ringtone",
    "static",
    "thud",
    "splash",
    "creak",
    "growl",
    "roar",
    "bark",
    "meow",
];

fn contains_sound_effect_keyword(text: &str) -> bool {
    let lower = text.to_lowercase();
    SOUND_EFFECT_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Detect on-screen text markers.  Lines that are entirely upper-case and
/// short (≤ 60 chars) with no speech punctuation are likely on-screen text.
fn looks_like_on_screen_text(block: &CaptionBlock) -> bool {
    let Some(first) = block.lines.first() else {
        return false;
    };
    let t = first.trim();
    if t.len() > 60 || t.is_empty() {
        return false;
    }
    // Must be entirely uppercase (letters + digits + common punctuation).
    let alpha_chars: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
    if alpha_chars.is_empty() {
        return false;
    }
    alpha_chars.iter().all(|c| c.is_uppercase())
        && !t.ends_with('.')
        && !t.ends_with('?')
        && !t.ends_with('!')
}

// ─── Block classifier ─────────────────────────────────────────────────────────

/// Classify a single [`CaptionBlock`] and return an [`FnSdhAnnotation`].
///
/// Classification priority (highest to lowest):
/// 1. Bracketed + sound-effect keyword → [`FnSdhKind::SoundEffect`]
/// 2. Bracketed (generic) → [`FnSdhKind::SoundEffect`] (generic description)
/// 3. Speaker-label line → [`FnSdhKind::SpeakerLabel`]
/// 4. On-screen text heuristic → [`FnSdhKind::OnScreenText`]
/// 5. Otherwise → [`FnSdhKind::Normal`]
///
/// Forced-narrative flag is applied on top when the block also carries a
/// `forced` marker (see [`FnSdhClassifier::mark_forced`]).
pub fn classify_block(block: &CaptionBlock) -> FnSdhAnnotation {
    let full_text = block.lines.join(" ");

    // Priority 1 & 2: bracketed sound-effect / description.
    if all_lines_bracketed(block) {
        let (kind, reason, confidence) = if contains_sound_effect_keyword(&full_text) {
            (
                FnSdhKind::SoundEffect,
                "all lines bracketed and contain sound-effect keyword".to_string(),
                0.92,
            )
        } else {
            (
                FnSdhKind::SoundEffect,
                "all lines are bracketed descriptions (SDH)".to_string(),
                0.78,
            )
        };
        return FnSdhAnnotation {
            block_id: block.id,
            kind,
            reason,
            confidence,
        };
    }

    // Priority 3: speaker label.
    if has_speaker_label_line(block) {
        return FnSdhAnnotation {
            block_id: block.id,
            kind: FnSdhKind::SpeakerLabel,
            reason: "line ends with `:` following an all-uppercase label".to_string(),
            confidence: 0.85,
        };
    }

    // Priority 4: on-screen text.
    if looks_like_on_screen_text(block) {
        return FnSdhAnnotation {
            block_id: block.id,
            kind: FnSdhKind::OnScreenText,
            reason: "short all-uppercase text without terminal speech punctuation".to_string(),
            confidence: 0.70,
        };
    }

    // Default.
    FnSdhAnnotation {
        block_id: block.id,
        kind: FnSdhKind::Normal,
        reason: "no FN/SDH markers detected".to_string(),
        confidence: 1.0,
    }
}

// ─── Classifier ───────────────────────────────────────────────────────────────

/// Configuration for [`FnSdhClassifier`].
#[derive(Debug, Clone)]
pub struct FnSdhClassifierConfig {
    /// Minimum confidence threshold for accepting a non-Normal classification.
    /// Blocks with confidence below this value are re-classified as Normal.
    pub min_confidence: f32,
    /// Whether to promote blocks whose `speaker_id` is set but whose text
    /// is otherwise unformatted to [`FnSdhKind::SpeakerLabel`].
    pub promote_speaker_id: bool,
}

impl Default for FnSdhClassifierConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.60,
            promote_speaker_id: false,
        }
    }
}

/// Runs FN/SDH classification over an entire caption track.
pub struct FnSdhClassifier {
    config: FnSdhClassifierConfig,
    /// Block IDs explicitly marked as forced narrative by the caller.
    forced_ids: std::collections::HashSet<u32>,
}

impl FnSdhClassifier {
    /// Create a new classifier with the given configuration.
    pub fn new(config: FnSdhClassifierConfig) -> Self {
        Self {
            config,
            forced_ids: std::collections::HashSet::new(),
        }
    }

    /// Mark a block ID as forced narrative regardless of heuristics.
    pub fn mark_forced(&mut self, block_id: u32) {
        self.forced_ids.insert(block_id);
    }

    /// Classify all blocks in `track` and return a list of annotations.
    pub fn classify_track(&self, track: &[CaptionBlock]) -> Vec<FnSdhAnnotation> {
        track
            .iter()
            .map(|block| {
                // Explicit forced-narrative override takes priority.
                if self.forced_ids.contains(&block.id) {
                    return FnSdhAnnotation {
                        block_id: block.id,
                        kind: FnSdhKind::ForcedNarrative,
                        reason: "explicitly marked as forced narrative".to_string(),
                        confidence: 1.0,
                    };
                }

                let mut ann = classify_block(block);

                // Optionally promote blocks with a speaker_id set.
                if self.config.promote_speaker_id
                    && block.speaker_id.is_some()
                    && ann.kind == FnSdhKind::Normal
                {
                    ann.kind = FnSdhKind::SpeakerLabel;
                    ann.reason = "speaker_id is set and promote_speaker_id is enabled".to_string();
                    ann.confidence = 0.65;
                }

                // Enforce minimum confidence threshold.
                if ann.kind != FnSdhKind::Normal && ann.confidence < self.config.min_confidence {
                    ann.kind = FnSdhKind::Normal;
                    ann.reason = format!(
                        "confidence {:.2} below threshold {:.2}; re-classified as Normal",
                        ann.confidence, self.config.min_confidence
                    );
                    ann.confidence = 1.0;
                }

                ann
            })
            .collect()
    }

    /// Return only the blocks classified as forced narrative or SDH (non-Normal).
    pub fn filter_non_normal<'a>(
        &self,
        track: &'a [CaptionBlock],
    ) -> Vec<(&'a CaptionBlock, FnSdhAnnotation)> {
        self.classify_track(track)
            .into_iter()
            .zip(track.iter())
            .filter_map(|(ann, block)| {
                if ann.kind != FnSdhKind::Normal {
                    Some((block, ann))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for FnSdhClassifier {
    fn default() -> Self {
        Self::new(FnSdhClassifierConfig::default())
    }
}

// ─── Summary statistics ───────────────────────────────────────────────────────

/// Aggregate statistics over an annotated caption track.
#[derive(Debug, Clone, Default)]
pub struct FnSdhTrackStats {
    pub total_blocks: usize,
    pub normal_count: usize,
    pub forced_narrative_count: usize,
    pub sound_effect_count: usize,
    pub speaker_label_count: usize,
    pub on_screen_text_count: usize,
}

impl FnSdhTrackStats {
    /// Compute statistics from a list of annotations.
    pub fn from_annotations(annotations: &[FnSdhAnnotation]) -> Self {
        let mut stats = Self {
            total_blocks: annotations.len(),
            ..Default::default()
        };
        for ann in annotations {
            match ann.kind {
                FnSdhKind::Normal => stats.normal_count += 1,
                FnSdhKind::ForcedNarrative => stats.forced_narrative_count += 1,
                FnSdhKind::SoundEffect => stats.sound_effect_count += 1,
                FnSdhKind::SpeakerLabel => stats.speaker_label_count += 1,
                FnSdhKind::OnScreenText => stats.on_screen_text_count += 1,
            }
        }
        stats
    }

    /// Fraction of blocks that carry any non-Normal annotation.
    pub fn sdh_ratio(&self) -> f32 {
        if self.total_blocks == 0 {
            return 0.0;
        }
        let non_normal = self.total_blocks.saturating_sub(self.normal_count);
        non_normal as f32 / self.total_blocks as f32
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, lines: &[&str]) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms: (id as u64 - 1) * 2000,
            end_ms: id as u64 * 2000,
            lines: lines.iter().map(|s| s.to_string()).collect(),
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    // --- is_bracketed_description ---

    #[test]
    fn bracketed_square_brackets() {
        assert!(is_bracketed_description("[MUSIC PLAYING]"));
    }

    #[test]
    fn bracketed_round_brackets() {
        assert!(is_bracketed_description("(audience laughing)"));
    }

    #[test]
    fn bracketed_normal_text_is_false() {
        assert!(!is_bracketed_description("Hello world"));
    }

    #[test]
    fn bracketed_partial_bracket_is_false() {
        assert!(!is_bracketed_description("[incomplete"));
    }

    // --- all_lines_bracketed ---

    #[test]
    fn all_lines_bracketed_true() {
        let block = make_block(1, &["[DOOR SLAMS]", "[MUSIC STOPS]"]);
        assert!(all_lines_bracketed(&block));
    }

    #[test]
    fn all_lines_bracketed_partial_false() {
        let block = make_block(1, &["[DOOR SLAMS]", "Hello world"]);
        assert!(!all_lines_bracketed(&block));
    }

    // --- has_speaker_label_line ---

    #[test]
    fn speaker_label_detected_simple() {
        let block = make_block(1, &["ALICE:", "How are you?"]);
        assert!(has_speaker_label_line(&block));
    }

    #[test]
    fn speaker_label_detected_bracketed() {
        let block = make_block(1, &["[DR. SMITH]:", "The results are in."]);
        assert!(has_speaker_label_line(&block));
    }

    #[test]
    fn speaker_label_not_detected_for_normal_colon() {
        // "Note: this is important" — "Note" is mixed case
        let block = make_block(1, &["Note: this is important"]);
        assert!(!has_speaker_label_line(&block));
    }

    // --- looks_like_on_screen_text ---

    #[test]
    fn on_screen_text_all_caps_short() {
        let block = make_block(1, &["LONDON, 1940"]);
        assert!(looks_like_on_screen_text(&block));
    }

    #[test]
    fn on_screen_text_lowercase_is_false() {
        let block = make_block(1, &["london, 1940"]);
        assert!(!looks_like_on_screen_text(&block));
    }

    #[test]
    fn on_screen_text_with_period_is_false() {
        let block = make_block(1, &["SOME TEXT."]);
        assert!(!looks_like_on_screen_text(&block));
    }

    // --- classify_block ---

    #[test]
    fn classify_sound_effect_with_keyword() {
        let block = make_block(1, &["[MUSIC PLAYING]"]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::SoundEffect);
        assert!(ann.confidence > 0.85);
    }

    #[test]
    fn classify_sound_effect_generic_brackets() {
        let block = make_block(1, &["[INDISTINCT CHATTER]"]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::SoundEffect);
    }

    #[test]
    fn classify_speaker_label() {
        let block = make_block(1, &["BOB:", "I disagree with that."]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::SpeakerLabel);
    }

    #[test]
    fn classify_on_screen_text() {
        let block = make_block(1, &["THREE YEARS LATER"]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::OnScreenText);
    }

    #[test]
    fn classify_normal_speech() {
        let block = make_block(1, &["Hello, how are you doing today?"]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::Normal);
    }

    // --- FnSdhClassifier ---

    #[test]
    fn classifier_forced_override() {
        let mut clf = FnSdhClassifier::default();
        clf.mark_forced(1);
        let block = make_block(1, &["Hello, how are you?"]);
        let annotations = clf.classify_track(&[block]);
        assert_eq!(annotations[0].kind, FnSdhKind::ForcedNarrative);
        assert!((annotations[0].confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn classifier_confidence_threshold_rejects_low_confidence() {
        let config = FnSdhClassifierConfig {
            min_confidence: 0.95,
            promote_speaker_id: false,
        };
        let clf = FnSdhClassifier::new(config);
        // on-screen text has confidence 0.70 < 0.95 → re-classified as Normal.
        let block = make_block(1, &["THREE YEARS LATER"]);
        let annotations = clf.classify_track(&[block]);
        assert_eq!(annotations[0].kind, FnSdhKind::Normal);
    }

    #[test]
    fn classifier_full_track() {
        let track = vec![
            make_block(1, &["Hello there"]),
            make_block(2, &["[DOOR SLAMS]"]),
            make_block(3, &["ALICE:", "What was that?"]),
            make_block(4, &["NEW YORK CITY"]),
        ];
        let clf = FnSdhClassifier::default();
        let annotations = clf.classify_track(&track);
        assert_eq!(annotations[0].kind, FnSdhKind::Normal);
        assert_eq!(annotations[1].kind, FnSdhKind::SoundEffect);
        assert_eq!(annotations[2].kind, FnSdhKind::SpeakerLabel);
        assert_eq!(annotations[3].kind, FnSdhKind::OnScreenText);
    }

    #[test]
    fn classifier_promote_speaker_id() {
        let config = FnSdhClassifierConfig {
            min_confidence: 0.60,
            promote_speaker_id: true,
        };
        let clf = FnSdhClassifier::new(config);
        let mut block = make_block(1, &["hello world"]);
        block.speaker_id = Some(2);
        let annotations = clf.classify_track(&[block]);
        assert_eq!(annotations[0].kind, FnSdhKind::SpeakerLabel);
    }

    #[test]
    fn classifier_filter_non_normal() {
        let track = vec![
            make_block(1, &["Normal speech here"]),
            make_block(2, &["[APPLAUSE]"]),
        ];
        let clf = FnSdhClassifier::default();
        let non_normal = clf.filter_non_normal(&track);
        assert_eq!(non_normal.len(), 1);
        assert_eq!(non_normal[0].1.kind, FnSdhKind::SoundEffect);
    }

    // --- FnSdhTrackStats ---

    #[test]
    fn track_stats_counts_correctly() {
        let annotations = vec![
            FnSdhAnnotation {
                block_id: 1,
                kind: FnSdhKind::Normal,
                reason: String::new(),
                confidence: 1.0,
            },
            FnSdhAnnotation {
                block_id: 2,
                kind: FnSdhKind::SoundEffect,
                reason: String::new(),
                confidence: 0.9,
            },
            FnSdhAnnotation {
                block_id: 3,
                kind: FnSdhKind::ForcedNarrative,
                reason: String::new(),
                confidence: 1.0,
            },
        ];
        let stats = FnSdhTrackStats::from_annotations(&annotations);
        assert_eq!(stats.total_blocks, 3);
        assert_eq!(stats.normal_count, 1);
        assert_eq!(stats.sound_effect_count, 1);
        assert_eq!(stats.forced_narrative_count, 1);
    }

    #[test]
    fn track_stats_sdh_ratio() {
        let annotations = vec![
            FnSdhAnnotation {
                block_id: 1,
                kind: FnSdhKind::Normal,
                reason: String::new(),
                confidence: 1.0,
            },
            FnSdhAnnotation {
                block_id: 2,
                kind: FnSdhKind::SoundEffect,
                reason: String::new(),
                confidence: 0.9,
            },
        ];
        let stats = FnSdhTrackStats::from_annotations(&annotations);
        assert!((stats.sdh_ratio() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn track_stats_empty_ratio_is_zero() {
        let stats = FnSdhTrackStats::default();
        assert_eq!(stats.sdh_ratio(), 0.0);
    }

    // ─── Additional tests ─────────────────────────────────────────────────────

    #[test]
    fn classify_empty_block_is_normal() {
        let block = make_block(1, &[]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::Normal);
    }

    #[test]
    fn classifier_empty_track_returns_empty() {
        let clf = FnSdhClassifier::default();
        let result = clf.classify_track(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn classify_round_bracket_sound_effect() {
        let block = make_block(1, &["(audience applause)"]);
        let ann = classify_block(&block);
        assert_eq!(ann.kind, FnSdhKind::SoundEffect);
    }

    #[test]
    fn classify_on_screen_text_with_numbers_and_commas() {
        // "NEW YORK, 2025" — uppercase + punctuation, no terminal sentence punct
        let block = make_block(1, &["NEW YORK, 2025"]);
        let ann = classify_block(&block);
        // uppercase alphabetics with no terminal period → OnScreenText
        assert_eq!(ann.kind, FnSdhKind::OnScreenText);
    }

    #[test]
    fn classifier_mark_forced_multiple_blocks() {
        let mut clf = FnSdhClassifier::default();
        clf.mark_forced(2);
        clf.mark_forced(4);
        let track = vec![
            make_block(1, &["Normal speech"]),
            make_block(2, &["Normal speech"]),
            make_block(3, &["Normal speech"]),
            make_block(4, &["Normal speech"]),
        ];
        let anns = clf.classify_track(&track);
        assert_eq!(anns[0].kind, FnSdhKind::Normal);
        assert_eq!(anns[1].kind, FnSdhKind::ForcedNarrative);
        assert_eq!(anns[2].kind, FnSdhKind::Normal);
        assert_eq!(anns[3].kind, FnSdhKind::ForcedNarrative);
    }

    #[test]
    fn classifier_filter_non_normal_empty_track() {
        let clf = FnSdhClassifier::default();
        let non_normal = clf.filter_non_normal(&[]);
        assert!(non_normal.is_empty());
    }

    #[test]
    fn track_stats_all_sound_effects() {
        let annotations: Vec<FnSdhAnnotation> = (1..=5)
            .map(|i| FnSdhAnnotation {
                block_id: i,
                kind: FnSdhKind::SoundEffect,
                reason: String::new(),
                confidence: 0.9,
            })
            .collect();
        let stats = FnSdhTrackStats::from_annotations(&annotations);
        assert_eq!(stats.sound_effect_count, 5);
        assert_eq!(stats.normal_count, 0);
        assert!((stats.sdh_ratio() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn fn_sdh_kind_equality() {
        assert_eq!(FnSdhKind::ForcedNarrative, FnSdhKind::ForcedNarrative);
        assert_ne!(FnSdhKind::ForcedNarrative, FnSdhKind::SoundEffect);
        assert_ne!(FnSdhKind::OnScreenText, FnSdhKind::SpeakerLabel);
    }

    #[test]
    fn annotation_fields_accessible() {
        let ann = FnSdhAnnotation {
            block_id: 42,
            kind: FnSdhKind::ForcedNarrative,
            reason: "test reason".to_string(),
            confidence: 0.88,
        };
        assert_eq!(ann.block_id, 42);
        assert_eq!(ann.kind, FnSdhKind::ForcedNarrative);
        assert_eq!(ann.reason, "test reason");
        assert!((ann.confidence - 0.88).abs() < 1e-5);
    }

    #[test]
    fn classify_mixed_bracket_lines_not_sound_effect() {
        // Only one line is bracketed; the other is not → not all_lines_bracketed
        let block = make_block(1, &["[MUSIC PLAYING]", "Hello there"]);
        let ann = classify_block(&block);
        // Should not be classified as SoundEffect since not all lines bracketed.
        assert_ne!(ann.kind, FnSdhKind::SoundEffect);
    }

    #[test]
    fn fn_sdh_classifier_default_min_confidence_is_0_60() {
        let clf = FnSdhClassifier::default();
        // OnScreenText confidence is 0.70 > 0.60 → should survive threshold.
        let block = make_block(1, &["LONDON"]);
        let anns = clf.classify_track(&[block]);
        assert_eq!(anns[0].kind, FnSdhKind::OnScreenText);
    }
}
