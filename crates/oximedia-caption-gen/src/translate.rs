//! Subtitle translation pipeline.
//!
//! Provides a stub translation pipeline that wraps caption text with a
//! target-language prefix.  In production this would be backed by a
//! machine-translation service; this implementation is an offline placeholder
//! suitable for testing and CI.
//!
//! # Example
//!
//! ```rust
//! use oximedia_caption_gen::translate::SubtitleTranslator;
//!
//! let result = SubtitleTranslator::translate_stub("Hello world", "fr");
//! assert_eq!(result, "[fr]: Hello world");
//! ```

use crate::alignment::CaptionBlock;

// ─── SubtitleTranslator ───────────────────────────────────────────────────────

/// A stateless subtitle translation helper.
///
/// The stub implementation prepends the target-language code to each segment,
/// enabling downstream components to identify the target language without
/// performing actual translation.
#[derive(Debug, Clone, Default)]
pub struct SubtitleTranslator;

impl SubtitleTranslator {
    /// Create a new translator instance.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Translate `text` to `target_lang` (stub implementation).
    ///
    /// Returns `"[{target_lang}]: {text}"`.  The target language code is
    /// normalised to lowercase.
    ///
    /// # Arguments
    ///
    /// * `text`        – Source caption text.
    /// * `target_lang` – BCP-47 language code (e.g. `"fr"`, `"de"`, `"ja"`).
    #[must_use]
    pub fn translate_stub(text: &str, target_lang: &str) -> String {
        format!("[{}]: {}", target_lang.to_lowercase(), text)
    }

    /// Translate all caption blocks in `track`, returning a new `Vec` with
    /// translated `lines`.
    ///
    /// Each line within a block is translated independently via
    /// [`translate_stub`][SubtitleTranslator::translate_stub].
    #[must_use]
    pub fn translate_track(track: &[CaptionBlock], target_lang: &str) -> Vec<CaptionBlock> {
        track
            .iter()
            .map(|block| {
                let translated_lines = block
                    .lines
                    .iter()
                    .map(|line| Self::translate_stub(line, target_lang))
                    .collect();
                CaptionBlock {
                    id: block.id,
                    start_ms: block.start_ms,
                    end_ms: block.end_ms,
                    lines: translated_lines,
                    speaker_id: block.speaker_id,
                    position: block.position.clone(),
                }
            })
            .collect()
    }

    /// Returns a human-readable name for the given BCP-47 language code.
    ///
    /// Falls back to the code itself for unknown codes.
    #[must_use]
    pub fn language_name(code: &str) -> &'static str {
        match code.to_lowercase().as_str() {
            "en" => "English",
            "es" => "Spanish",
            "fr" => "French",
            "de" => "German",
            "it" => "Italian",
            "pt" => "Portuguese",
            "nl" => "Dutch",
            "ja" => "Japanese",
            "zh" => "Chinese",
            "ko" => "Korean",
            "ar" => "Arabic",
            "ru" => "Russian",
            _ => "Unknown",
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_stub_format() {
        let result = SubtitleTranslator::translate_stub("Hello world", "fr");
        assert_eq!(result, "[fr]: Hello world");
    }

    #[test]
    fn test_translate_stub_uppercase_lang_normalised() {
        let result = SubtitleTranslator::translate_stub("Hola", "ES");
        assert_eq!(result, "[es]: Hola");
    }

    #[test]
    fn test_translate_stub_empty_text() {
        let result = SubtitleTranslator::translate_stub("", "de");
        assert_eq!(result, "[de]: ");
    }

    #[test]
    fn test_translate_stub_unknown_lang() {
        let result = SubtitleTranslator::translate_stub("text", "xx");
        assert!(result.starts_with("[xx]:"));
    }

    #[test]
    fn test_language_name_known() {
        assert_eq!(SubtitleTranslator::language_name("en"), "English");
        assert_eq!(SubtitleTranslator::language_name("fr"), "French");
        assert_eq!(SubtitleTranslator::language_name("ja"), "Japanese");
    }

    #[test]
    fn test_language_name_unknown() {
        assert_eq!(SubtitleTranslator::language_name("zz"), "Unknown");
    }

    #[test]
    fn test_translate_track_preserves_timing() {
        use crate::alignment::CaptionPosition;
        let blocks = vec![CaptionBlock {
            id: 1,
            start_ms: 0,
            end_ms: 2000,
            lines: vec!["Hello".to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }];
        let translated = SubtitleTranslator::translate_track(&blocks, "de");
        assert_eq!(translated.len(), 1);
        assert_eq!(translated[0].start_ms, 0);
        assert_eq!(translated[0].end_ms, 2000);
        assert!(translated[0].lines[0].contains("[de]:"));
    }
}
