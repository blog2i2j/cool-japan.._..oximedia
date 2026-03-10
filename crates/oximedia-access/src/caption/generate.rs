//! Caption generation from audio.

use crate::caption::{Caption, CaptionQuality, CaptionType};
use crate::error::{AccessError, AccessResult};
use oximedia_audio::frame::AudioBuffer;
use oximedia_subtitle::Subtitle;
use serde::{Deserialize, Serialize};

/// Configuration for caption generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionConfig {
    /// Language code (e.g., "en", "es", "fr").
    pub language: String,
    /// Caption type.
    pub caption_type: CaptionType,
    /// Quality level.
    pub quality: CaptionQuality,
    /// Maximum characters per line.
    pub max_chars_per_line: usize,
    /// Maximum lines per caption.
    pub max_lines: usize,
    /// Minimum caption duration in milliseconds.
    pub min_duration_ms: i64,
    /// Maximum caption duration in milliseconds.
    pub max_duration_ms: i64,
    /// Enable speaker identification.
    pub identify_speakers: bool,
    /// Enable sound effects descriptions.
    pub include_sound_effects: bool,
    /// Enable music descriptions.
    pub include_music_description: bool,
}

impl Default for CaptionConfig {
    fn default() -> Self {
        Self {
            language: "en".to_string(),
            caption_type: CaptionType::Closed,
            quality: CaptionQuality::Standard,
            max_chars_per_line: 42,
            max_lines: 2,
            min_duration_ms: 1000,
            max_duration_ms: 7000,
            identify_speakers: true,
            include_sound_effects: true,
            include_music_description: true,
        }
    }
}

impl CaptionConfig {
    /// Create a new configuration.
    #[must_use]
    pub fn new(language: String, caption_type: CaptionType) -> Self {
        Self {
            language,
            caption_type,
            ..Default::default()
        }
    }

    /// Set quality level.
    #[must_use]
    pub const fn with_quality(mut self, quality: CaptionQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Set maximum characters per line.
    #[must_use]
    pub const fn with_max_chars_per_line(mut self, max_chars: usize) -> Self {
        self.max_chars_per_line = max_chars;
        self
    }

    /// Enable speaker identification.
    #[must_use]
    pub const fn with_speaker_identification(mut self, enable: bool) -> Self {
        self.identify_speakers = enable;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> AccessResult<()> {
        if self.max_chars_per_line == 0 {
            return Err(AccessError::CaptionFailed(
                "Max characters per line must be positive".to_string(),
            ));
        }

        if self.max_lines == 0 {
            return Err(AccessError::CaptionFailed(
                "Max lines must be positive".to_string(),
            ));
        }

        if self.min_duration_ms <= 0 {
            return Err(AccessError::CaptionFailed(
                "Minimum duration must be positive".to_string(),
            ));
        }

        if self.max_duration_ms < self.min_duration_ms {
            return Err(AccessError::CaptionFailed(
                "Maximum duration must be >= minimum duration".to_string(),
            ));
        }

        Ok(())
    }
}

/// Caption generator.
///
/// Generates captions from audio using speech-to-text.
pub struct CaptionGenerator {
    config: CaptionConfig,
}

impl CaptionGenerator {
    /// Create a new caption generator.
    #[must_use]
    pub fn new(config: CaptionConfig) -> Self {
        Self { config }
    }

    /// Create generator with default configuration.
    #[must_use]
    pub fn default() -> Self {
        Self::new(CaptionConfig::default())
    }

    /// Generate captions from audio.
    ///
    /// This is an integration point for speech-to-text services.
    /// In production, this would call services like:
    /// - AWS Transcribe
    /// - Google Cloud Speech-to-Text
    /// - Microsoft Azure Speech
    /// - `OpenAI` Whisper
    /// - Local STT engines
    pub fn generate_from_audio(&self, _audio: &AudioBuffer) -> AccessResult<Vec<Caption>> {
        self.config.validate()?;

        // Placeholder: In production, call STT service
        // This would:
        // 1. Send audio to STT service
        // 2. Receive transcription with timestamps
        // 3. Split into caption segments
        // 4. Apply formatting rules
        // 5. Add speaker labels if enabled
        // 6. Add sound effects if enabled

        let captions = vec![
            self.create_caption(1000, 3000, "Example caption text.", None),
            self.create_caption(4000, 6000, "Another caption segment.", None),
        ];

        Ok(captions)
    }

    /// Generate from existing transcript.
    pub fn generate_from_transcript(
        &self,
        transcript: &str,
        timestamps: &[(i64, i64)],
    ) -> AccessResult<Vec<Caption>> {
        self.config.validate()?;

        if timestamps.is_empty() {
            return Err(AccessError::CaptionFailed(
                "No timestamps provided".to_string(),
            ));
        }

        let words: Vec<&str> = transcript.split_whitespace().collect();
        let words_per_segment = words.len() / timestamps.len();

        let mut captions = Vec::new();

        for (i, (start, end)) in timestamps.iter().enumerate() {
            let word_start = i * words_per_segment;
            let word_end = ((i + 1) * words_per_segment).min(words.len());

            if word_start < words.len() {
                let text = words[word_start..word_end].join(" ");
                let formatted = self.format_text(&text);
                captions.push(self.create_caption(*start, *end, &formatted, None));
            }
        }

        Ok(captions)
    }

    /// Format text according to caption rules.
    fn format_text(&self, text: &str) -> String {
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.len() + word.len() < self.config.max_chars_per_line {
                if !current_line.is_empty() {
                    current_line.push(' ');
                }
                current_line.push_str(word);
            } else {
                if !current_line.is_empty() {
                    lines.push(current_line.clone());
                    current_line.clear();
                }
                current_line.push_str(word);
            }

            if lines.len() >= self.config.max_lines {
                break;
            }
        }

        if !current_line.is_empty() && lines.len() < self.config.max_lines {
            lines.push(current_line);
        }

        lines.join("\n")
    }

    /// Create a caption with proper formatting.
    fn create_caption(
        &self,
        start_time: i64,
        end_time: i64,
        text: &str,
        speaker: Option<String>,
    ) -> Caption {
        let formatted_text = self.format_text(text);

        let subtitle = Subtitle::new(start_time, end_time, formatted_text);

        let mut caption = Caption::new(subtitle, self.config.caption_type);

        if let Some(speaker_name) = speaker {
            caption = caption.with_speaker(speaker_name);
        }

        caption
    }

    /// Add sound effect description.
    #[allow(dead_code)]
    fn add_sound_effect(&self, time: i64, effect: &str) -> Caption {
        let text = format!("[{effect}]");
        let subtitle = Subtitle::new(time, time + 1000, text);
        Caption::new(subtitle, self.config.caption_type)
    }

    /// Add music description.
    #[allow(dead_code)]
    fn add_music_description(&self, start_time: i64, end_time: i64, description: &str) -> Caption {
        let text = format!("♪ {description} ♪");
        let subtitle = Subtitle::new(start_time, end_time, text);
        Caption::new(subtitle, self.config.caption_type)
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &CaptionConfig {
        &self.config
    }

    /// Validate caption duration.
    pub fn validate_caption(&self, caption: &Caption) -> AccessResult<()> {
        let duration = caption.duration();

        if duration < self.config.min_duration_ms {
            return Err(AccessError::CaptionFailed(format!(
                "Caption duration too short: {}ms < {}ms",
                duration, self.config.min_duration_ms
            )));
        }

        if duration > self.config.max_duration_ms {
            return Err(AccessError::CaptionFailed(format!(
                "Caption duration too long: {}ms > {}ms",
                duration, self.config.max_duration_ms
            )));
        }

        Ok(())
    }

    /// Split long caption into multiple segments.
    #[must_use]
    pub fn split_caption(&self, caption: &Caption) -> Vec<Caption> {
        if caption.duration() <= self.config.max_duration_ms {
            return vec![caption.clone()];
        }

        let words: Vec<&str> = caption.text().split_whitespace().collect();
        let segment_count = (caption.duration() / self.config.max_duration_ms) + 1;
        let words_per_segment = words.len() / segment_count as usize;

        let mut segments = Vec::new();
        let duration_per_segment = caption.duration() / segment_count;

        for i in 0..segment_count as usize {
            let start_word = i * words_per_segment;
            let end_word = ((i + 1) * words_per_segment).min(words.len());

            if start_word < words.len() {
                let text = words[start_word..end_word].join(" ");
                let start_time = caption.start_time() + (i as i64 * duration_per_segment);
                let end_time = start_time + duration_per_segment;

                let subtitle = Subtitle::new(start_time, end_time, text);
                segments.push(Caption::new(subtitle, caption.caption_type));
            }
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = CaptionConfig::default();
        assert_eq!(config.language, "en");
        assert_eq!(config.max_chars_per_line, 42);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = CaptionConfig::default();
        assert!(config.validate().is_ok());

        config.max_chars_per_line = 0;
        assert!(config.validate().is_err());

        config.max_chars_per_line = 42;
        config.min_duration_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_generator_creation() {
        let generator = CaptionGenerator::default();
        assert_eq!(generator.config().language, "en");
    }

    #[test]
    fn test_format_text() {
        let generator = CaptionGenerator::new(CaptionConfig::default().with_max_chars_per_line(20));

        let text = "This is a very long caption that should be split into multiple lines";
        let formatted = generator.format_text(text);

        assert!(formatted.contains('\n'));
    }

    #[test]
    fn test_generate_from_transcript() {
        let generator = CaptionGenerator::default();
        let transcript = "This is a test transcript with some words";
        let timestamps = vec![(1000, 3000), (4000, 6000)];

        let captions = generator
            .generate_from_transcript(transcript, &timestamps)
            .expect("test expectation failed");
        assert_eq!(captions.len(), 2);
    }

    #[test]
    fn test_split_caption() {
        let config = CaptionConfig::default().with_max_chars_per_line(10);
        let generator = CaptionGenerator::new(config);

        let long_text = "Word ".repeat(100);
        let subtitle = Subtitle::new(0, 10000, long_text);
        let caption = Caption::new(subtitle, CaptionType::Closed);

        let segments = generator.split_caption(&caption);
        assert!(segments.len() > 1);
    }
}
