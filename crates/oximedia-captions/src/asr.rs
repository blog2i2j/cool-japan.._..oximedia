//! Automatic Speech Recognition (ASR) integration

use crate::error::{CaptionError, Result};
use crate::types::{Caption, CaptionTrack, Language, Timestamp};
use serde::{Deserialize, Serialize};

/// ASR transcript word with timing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptWord {
    /// Word text
    pub word: String,
    /// Start timestamp
    pub start: Timestamp,
    /// End timestamp
    pub end: Timestamp,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
}

/// ASR transcript with speaker diarization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    /// Language detected
    pub language: Option<Language>,
    /// Words with timing
    pub words: Vec<TranscriptWord>,
    /// Speaker segments (speaker ID, start, end)
    pub speakers: Vec<SpeakerSegment>,
}

/// Speaker segment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerSegment {
    /// Speaker ID
    pub speaker_id: String,
    /// Start timestamp
    pub start: Timestamp,
    /// End timestamp
    pub end: Timestamp,
}

impl Transcript {
    /// Create a new transcript
    #[must_use]
    pub fn new() -> Self {
        Self {
            language: None,
            words: Vec::new(),
            speakers: Vec::new(),
        }
    }

    /// Add a word
    pub fn add_word(&mut self, word: TranscriptWord) {
        self.words.push(word);
    }

    /// Add a speaker segment
    pub fn add_speaker(&mut self, segment: SpeakerSegment) {
        self.speakers.push(segment);
    }

    /// Get words by confidence threshold
    #[must_use]
    pub fn filter_by_confidence(&self, threshold: f64) -> Vec<&TranscriptWord> {
        self.words
            .iter()
            .filter(|w| w.confidence >= threshold)
            .collect()
    }

    /// Get speaker at timestamp
    #[must_use]
    pub fn get_speaker_at(&self, timestamp: Timestamp) -> Option<&SpeakerSegment> {
        self.speakers
            .iter()
            .find(|s| s.start <= timestamp && s.end > timestamp)
    }
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

/// Caption generator from ASR transcript
#[allow(dead_code)]
pub struct CaptionGenerator {
    /// Maximum characters per line
    max_chars_per_line: usize,
    /// Maximum lines per caption
    max_lines: usize,
    /// Target reading speed (WPM)
    target_reading_speed: f64,
    /// Minimum caption duration (milliseconds)
    min_duration_ms: i64,
    /// Include speaker identification
    include_speakers: bool,
}

impl CaptionGenerator {
    /// Create a new caption generator
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_chars_per_line: 42,
            max_lines: 2,
            target_reading_speed: 160.0,
            min_duration_ms: 1000,
            include_speakers: false,
        }
    }

    /// Set maximum characters per line
    #[must_use]
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars_per_line = max_chars;
        self
    }

    /// Set maximum lines
    #[must_use]
    pub fn with_max_lines(mut self, max_lines: usize) -> Self {
        self.max_lines = max_lines;
        self
    }

    /// Enable speaker identification
    #[must_use]
    pub fn with_speakers(mut self, enabled: bool) -> Self {
        self.include_speakers = enabled;
        self
    }

    /// Generate captions from transcript
    pub fn generate(&self, transcript: &Transcript) -> Result<CaptionTrack> {
        let language = transcript
            .language
            .clone()
            .unwrap_or_else(Language::english);
        let mut track = CaptionTrack::new(language);

        let mut current_words = Vec::new();
        let mut current_start: Option<Timestamp> = None;

        for word in &transcript.words {
            if current_start.is_none() {
                current_start = Some(word.start);
            }

            current_words.push(word);

            // Check if we should create a caption
            let should_break = self.should_break_caption(&current_words);

            if should_break {
                if let Some(start) = current_start {
                    let caption = self.create_caption(start, &current_words, transcript)?;
                    track.add_caption(caption)?;
                }

                current_words.clear();
                current_start = None;
            }
        }

        // Add remaining words as final caption
        if !current_words.is_empty() {
            if let Some(start) = current_start {
                let caption = self.create_caption(start, &current_words, transcript)?;
                track.add_caption(caption)?;
            }
        }

        Ok(track)
    }

    fn should_break_caption(&self, words: &[&TranscriptWord]) -> bool {
        if words.is_empty() {
            return false;
        }

        // Check character count
        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if text.len() > self.max_chars_per_line * self.max_lines {
            return true;
        }

        // Check for natural breaks (punctuation)
        if let Some(last_word) = words.last() {
            if last_word.word.ends_with('.')
                || last_word.word.ends_with('!')
                || last_word.word.ends_with('?')
            {
                return true;
            }
        }

        // Check duration
        let start = words[0].start;
        // words is non-empty (checked above), so last() always yields Some
        let end = words[words.len() - 1].end;
        let duration_ms = end.duration_since(start).as_millis();
        if duration_ms > 5000 {
            // Too long
            return true;
        }

        false
    }

    fn create_caption(
        &self,
        start: Timestamp,
        words: &[&TranscriptWord],
        transcript: &Transcript,
    ) -> Result<Caption> {
        if words.is_empty() {
            return Err(CaptionError::Other("No words for caption".to_string()));
        }

        // words is non-empty (checked above), so last() always yields Some
        let end = words[words.len() - 1].end;
        let text = words
            .iter()
            .map(|w| w.word.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        // Apply line breaking
        let lines = self.break_into_lines(&text);
        let formatted_text = lines.join("\n");

        let mut caption = Caption::new(start, end, formatted_text);

        // Add speaker identification if enabled
        if self.include_speakers {
            if let Some(speaker) = transcript.get_speaker_at(start) {
                caption.speaker = Some(speaker.speaker_id.clone());
            }
        }

        Ok(caption)
    }

    fn break_into_lines(&self, text: &str) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in words {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else {
                let test_line = format!("{current_line} {word}");
                if test_line.len() <= self.max_chars_per_line {
                    current_line = test_line;
                } else {
                    lines.push(current_line);
                    current_line = word.to_string();
                    if lines.len() >= self.max_lines {
                        break;
                    }
                }
            }
        }

        if !current_line.is_empty() {
            lines.push(current_line);
        }

        lines
    }

    /// Align existing captions with ASR transcript (forced alignment)
    pub fn align_captions(
        &self,
        track: &mut CaptionTrack,
        transcript: &Transcript,
    ) -> Result<usize> {
        let mut aligned_count = 0;

        for caption in &mut track.captions {
            // Find matching words in transcript
            let words = self.find_matching_words(&caption.text, transcript);

            if !words.is_empty() {
                // Update timing based on word boundaries
                caption.start = words[0].start;
                // words is non-empty (checked above), so last index is valid
                caption.end = words[words.len() - 1].end;
                aligned_count += 1;
            }
        }

        Ok(aligned_count)
    }

    fn find_matching_words<'a>(
        &self,
        text: &str,
        transcript: &'a Transcript,
    ) -> Vec<&'a TranscriptWord> {
        let caption_words: Vec<&str> = text.split_whitespace().collect();
        let mut matches = Vec::new();
        let mut i = 0;

        while i < transcript.words.len() {
            let mut matched = true;
            for (j, &caption_word) in caption_words.iter().enumerate() {
                if i + j >= transcript.words.len() {
                    matched = false;
                    break;
                }

                let transcript_word = &transcript.words[i + j].word;
                if !transcript_word.eq_ignore_ascii_case(caption_word) {
                    matched = false;
                    break;
                }
            }

            if matched {
                matches.extend(&transcript.words[i..i + caption_words.len()]);
                break;
            }

            i += 1;
        }

        matches
    }

    /// Correct drift in caption timing
    pub fn correct_drift(
        &self,
        track: &mut CaptionTrack,
        transcript: &Transcript,
        max_drift_ms: i64,
    ) -> Result<usize> {
        let mut corrected_count = 0;

        for caption in &mut track.captions {
            let words = self.find_matching_words(&caption.text, transcript);

            if !words.is_empty() {
                let transcript_start = words[0].start;
                let drift = (caption.start.as_millis() - transcript_start.as_millis()).abs();

                if drift > max_drift_ms {
                    // Significant drift detected, correct it
                    let offset = transcript_start.as_micros() - caption.start.as_micros();
                    caption.start = Timestamp::from_micros(caption.start.as_micros() + offset);
                    caption.end = Timestamp::from_micros(caption.end.as_micros() + offset);
                    corrected_count += 1;
                }
            }
        }

        Ok(corrected_count)
    }
}

impl Default for CaptionGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Import transcript from common formats
pub mod import {
    use super::{CaptionError, Result, Transcript};

    /// Import from WebVTT-style transcript
    pub fn from_webvtt(data: &str) -> Result<Transcript> {
        let transcript = Transcript::new();

        for line in data.lines() {
            if line.trim().is_empty() || line.starts_with("WEBVTT") {
                continue;
            }

            // Parse timestamp lines (simplified)
            if line.contains("-->") {
                // Would parse actual WebVTT format
            }
        }

        Ok(transcript)
    }

    /// Import from JSON format
    pub fn from_json(data: &str) -> Result<Transcript> {
        serde_json::from_str(data).map_err(|e| CaptionError::Import(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcript_creation() {
        let mut transcript = Transcript::new();
        transcript.add_word(TranscriptWord {
            word: "Hello".to_string(),
            start: Timestamp::from_secs(0),
            end: Timestamp::from_secs(1),
            confidence: 0.95,
        });

        assert_eq!(transcript.words.len(), 1);
    }

    #[test]
    fn test_caption_generation() {
        let mut transcript = Transcript::new();
        transcript.add_word(TranscriptWord {
            word: "Hello".to_string(),
            start: Timestamp::from_secs(0),
            end: Timestamp::from_millis(500),
            confidence: 0.95,
        });
        transcript.add_word(TranscriptWord {
            word: "world.".to_string(),
            start: Timestamp::from_millis(500),
            end: Timestamp::from_secs(1),
            confidence: 0.98,
        });

        let generator = CaptionGenerator::new();
        let track = generator
            .generate(&transcript)
            .expect("generation should succeed");

        assert_eq!(track.captions.len(), 1);
        assert!(track.captions[0].text.contains("Hello world"));
    }

    #[test]
    fn test_confidence_filtering() {
        let mut transcript = Transcript::new();
        transcript.add_word(TranscriptWord {
            word: "High".to_string(),
            start: Timestamp::from_secs(0),
            end: Timestamp::from_secs(1),
            confidence: 0.95,
        });
        transcript.add_word(TranscriptWord {
            word: "Low".to_string(),
            start: Timestamp::from_secs(1),
            end: Timestamp::from_secs(2),
            confidence: 0.45,
        });

        let filtered = transcript.filter_by_confidence(0.80);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].word, "High");
    }

    #[test]
    fn test_speaker_identification() {
        let mut transcript = Transcript::new();
        transcript.add_speaker(SpeakerSegment {
            speaker_id: "Speaker 1".to_string(),
            start: Timestamp::from_secs(0),
            end: Timestamp::from_secs(5),
        });

        let speaker = transcript.get_speaker_at(Timestamp::from_secs(2));
        assert!(speaker.is_some());
        assert_eq!(
            speaker.expect("speaker should be present").speaker_id,
            "Speaker 1"
        );
    }
}
