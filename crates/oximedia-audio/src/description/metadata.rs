//! Audio description metadata and WCAG 2.1 compliance markers.
//!
//! This module provides metadata types for audio description tracks,
//! including language tags, speaker identification, content classification,
//! and WCAG 2.1 accessibility compliance markers.

#![forbid(unsafe_code)]

use crate::{AudioError, AudioResult};
use std::collections::HashMap;

/// Audio description track metadata.
#[derive(Clone, Debug)]
pub struct AudioDescriptionMetadata {
    /// Track identifier.
    pub track_id: String,
    /// Track label (e.g., "English Audio Description").
    pub label: String,
    /// Language tag (RFC 5646/BCP 47).
    pub language: LanguageTag,
    /// Speaker information.
    pub speaker: SpeakerInfo,
    /// Content classification.
    pub classification: ContentClassification,
    /// WCAG compliance level.
    pub wcag_level: WcagLevel,
    /// Whether this is extended audio description (with pausing).
    pub is_extended: bool,
    /// Timing mode.
    pub timing_mode: TimingMode,
    /// Additional metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

impl Default for AudioDescriptionMetadata {
    fn default() -> Self {
        Self {
            track_id: String::new(),
            label: String::from("Audio Description"),
            language: LanguageTag::default(),
            speaker: SpeakerInfo::default(),
            classification: ContentClassification::default(),
            wcag_level: WcagLevel::default(),
            is_extended: false,
            timing_mode: TimingMode::default(),
            metadata: HashMap::new(),
        }
    }
}

impl AudioDescriptionMetadata {
    /// Create new audio description metadata.
    #[must_use]
    pub fn new(track_id: impl Into<String>, language: LanguageTag) -> Self {
        Self {
            track_id: track_id.into(),
            language,
            ..Default::default()
        }
    }

    /// Set track label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Set speaker information.
    #[must_use]
    pub fn with_speaker(mut self, speaker: SpeakerInfo) -> Self {
        self.speaker = speaker;
        self
    }

    /// Set content classification.
    #[must_use]
    pub fn with_classification(mut self, classification: ContentClassification) -> Self {
        self.classification = classification;
        self
    }

    /// Set WCAG compliance level.
    #[must_use]
    pub fn with_wcag_level(mut self, level: WcagLevel) -> Self {
        self.wcag_level = level;
        self
    }

    /// Enable extended audio description mode.
    #[must_use]
    pub fn with_extended(mut self, extended: bool) -> Self {
        self.is_extended = extended;
        self
    }

    /// Set timing mode.
    #[must_use]
    pub fn with_timing_mode(mut self, mode: TimingMode) -> Self {
        self.timing_mode = mode;
        self
    }

    /// Add custom metadata field.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Validate metadata completeness.
    pub fn validate(&self) -> AudioResult<()> {
        if self.track_id.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Track ID cannot be empty".to_string(),
            ));
        }

        if self.label.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Track label cannot be empty".to_string(),
            ));
        }

        self.language.validate()?;

        Ok(())
    }
}

/// Language tag following RFC 5646/BCP 47.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanguageTag {
    /// Primary language subtag (ISO 639).
    pub language: String,
    /// Script subtag (ISO 15924).
    pub script: Option<String>,
    /// Region subtag (ISO 3166-1).
    pub region: Option<String>,
    /// Variant subtags.
    pub variants: Vec<String>,
}

impl Default for LanguageTag {
    fn default() -> Self {
        Self {
            language: String::from("en"),
            script: None,
            region: None,
            variants: Vec::new(),
        }
    }
}

impl LanguageTag {
    /// Create a new language tag.
    #[must_use]
    pub fn new(language: impl Into<String>) -> Self {
        Self {
            language: language.into(),
            ..Default::default()
        }
    }

    /// Set script subtag.
    #[must_use]
    pub fn with_script(mut self, script: impl Into<String>) -> Self {
        self.script = Some(script.into());
        self
    }

    /// Set region subtag.
    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Add variant subtag.
    #[must_use]
    pub fn with_variant(mut self, variant: impl Into<String>) -> Self {
        self.variants.push(variant.into());
        self
    }

    /// Parse from string (e.g., "en-US", "zh-Hans-CN").
    pub fn parse(tag: &str) -> AudioResult<Self> {
        let parts: Vec<&str> = tag.split('-').collect();

        if parts.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Empty language tag".to_string(),
            ));
        }

        let language = parts[0].to_lowercase();
        let mut script = None;
        let mut region = None;
        let mut variants = Vec::new();

        let mut i = 1;
        while i < parts.len() {
            let part = parts[i];

            if part.len() == 4 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                script = Some(part.to_string());
            } else if part.len() == 2 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                region = Some(part.to_uppercase());
            } else if part.len() == 3 && part.chars().all(|c| c.is_ascii_digit()) {
                region = Some(part.to_string());
            } else {
                variants.push(part.to_string());
            }

            i += 1;
        }

        Ok(Self {
            language,
            script,
            region,
            variants,
        })
    }

    /// Format as string.
    #[must_use]
    pub fn to_string(&self) -> String {
        let mut parts = vec![self.language.clone()];

        if let Some(ref script) = self.script {
            parts.push(script.clone());
        }

        if let Some(ref region) = self.region {
            parts.push(region.clone());
        }

        parts.extend(self.variants.clone());

        parts.join("-")
    }

    /// Validate language tag.
    pub fn validate(&self) -> AudioResult<()> {
        if self.language.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Language subtag cannot be empty".to_string(),
            ));
        }

        if self.language.len() < 2 || self.language.len() > 3 {
            return Err(AudioError::InvalidParameter(format!(
                "Invalid language subtag length: {}",
                self.language
            )));
        }

        Ok(())
    }
}

/// Speaker information for audio description.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeakerInfo {
    /// Speaker name or identifier.
    pub name: String,
    /// Speaker gender for TTS voice selection.
    pub gender: SpeakerGender,
    /// Speaker age for TTS voice selection.
    pub age: SpeakerAge,
    /// Voice characteristics.
    pub voice_characteristics: VoiceCharacteristics,
}

impl Default for SpeakerInfo {
    fn default() -> Self {
        Self {
            name: String::from("Narrator"),
            gender: SpeakerGender::Neutral,
            age: SpeakerAge::Adult,
            voice_characteristics: VoiceCharacteristics::default(),
        }
    }
}

impl SpeakerInfo {
    /// Create new speaker info.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set speaker gender.
    #[must_use]
    pub fn with_gender(mut self, gender: SpeakerGender) -> Self {
        self.gender = gender;
        self
    }

    /// Set speaker age.
    #[must_use]
    pub fn with_age(mut self, age: SpeakerAge) -> Self {
        self.age = age;
        self
    }

    /// Set voice characteristics.
    #[must_use]
    pub fn with_voice_characteristics(mut self, characteristics: VoiceCharacteristics) -> Self {
        self.voice_characteristics = characteristics;
        self
    }
}

/// Speaker gender for TTS voice selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpeakerGender {
    /// Male voice.
    Male,
    /// Female voice.
    Female,
    /// Gender-neutral voice.
    #[default]
    Neutral,
}

/// Speaker age for TTS voice selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpeakerAge {
    /// Child voice (under 13).
    Child,
    /// Teenage voice (13-19).
    Teen,
    /// Adult voice (20-59).
    #[default]
    Adult,
    /// Senior voice (60+).
    Senior,
}

/// Voice characteristics for TTS.
#[derive(Clone, Debug, PartialEq)]
pub struct VoiceCharacteristics {
    /// Pitch in Hz (80-300 Hz typical).
    pub pitch_hz: f64,
    /// Speech rate (words per minute, 120-180 typical).
    pub rate_wpm: f64,
    /// Volume level (0.0-1.0).
    pub volume: f64,
}

impl Default for VoiceCharacteristics {
    fn default() -> Self {
        Self {
            pitch_hz: 120.0,
            rate_wpm: 150.0,
            volume: 1.0,
        }
    }
}

/// Content classification for audio description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ContentClassification {
    /// General audience.
    #[default]
    General,
    /// Educational content.
    Educational,
    /// News and current affairs.
    News,
    /// Entertainment/drama.
    Entertainment,
    /// Sports.
    Sports,
    /// Documentary.
    Documentary,
    /// Children's content.
    Children,
}

/// WCAG 2.1 compliance level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WcagLevel {
    /// Level A (minimum).
    A,
    /// Level AA (recommended).
    #[default]
    AA,
    /// Level AAA (enhanced).
    AAA,
}

impl WcagLevel {
    /// Check if this level meets or exceeds another level.
    #[must_use]
    pub fn meets(&self, other: WcagLevel) -> bool {
        match (self, other) {
            (WcagLevel::AAA, _) => true,
            (WcagLevel::AA, WcagLevel::AAA) => false,
            (WcagLevel::AA, _) => true,
            (WcagLevel::A, WcagLevel::A) => true,
            (WcagLevel::A, _) => false,
        }
    }
}

/// Timing mode for audio description.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TimingMode {
    /// Standard audio description (fits in natural pauses).
    #[default]
    Standard,
    /// Extended audio description (pauses main audio).
    Extended,
    /// Continuous audio description (overlays main audio).
    Continuous,
}

/// Audio description cue point.
#[derive(Clone, Debug)]
pub struct DescriptionCue {
    /// Cue identifier.
    pub id: String,
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Description text (if using TTS).
    pub text: Option<String>,
    /// Audio file path (if using pre-recorded audio).
    pub audio_path: Option<String>,
    /// Mixing strategy for this cue.
    pub mixing_strategy: CueMixingStrategy,
    /// Priority level (higher = more important).
    pub priority: u8,
}

impl DescriptionCue {
    /// Create a new description cue.
    #[must_use]
    pub fn new(id: impl Into<String>, start_time: f64, end_time: f64) -> Self {
        Self {
            id: id.into(),
            start_time,
            end_time,
            text: None,
            audio_path: None,
            mixing_strategy: CueMixingStrategy::default(),
            priority: 128,
        }
    }

    /// Set description text.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set audio file path.
    #[must_use]
    pub fn with_audio_path(mut self, path: impl Into<String>) -> Self {
        self.audio_path = Some(path.into());
        self
    }

    /// Set mixing strategy.
    #[must_use]
    pub fn with_mixing_strategy(mut self, strategy: CueMixingStrategy) -> Self {
        self.mixing_strategy = strategy;
        self
    }

    /// Set priority level.
    #[must_use]
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Get duration in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.end_time - self.start_time
    }

    /// Validate cue point.
    pub fn validate(&self) -> AudioResult<()> {
        if self.id.is_empty() {
            return Err(AudioError::InvalidParameter(
                "Cue ID cannot be empty".to_string(),
            ));
        }

        if self.start_time < 0.0 {
            return Err(AudioError::InvalidParameter(format!(
                "Invalid start time: {}",
                self.start_time
            )));
        }

        if self.end_time <= self.start_time {
            return Err(AudioError::InvalidParameter(format!(
                "End time must be after start time: {} <= {}",
                self.end_time, self.start_time
            )));
        }

        if self.text.is_none() && self.audio_path.is_none() {
            return Err(AudioError::InvalidParameter(
                "Cue must have either text or audio path".to_string(),
            ));
        }

        Ok(())
    }
}

/// Mixing strategy for individual cue points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CueMixingStrategy {
    /// Duck main audio by specified amount.
    #[default]
    Duck,
    /// Mix with main audio.
    Mix,
    /// Replace main audio completely.
    Replace,
    /// Pause main audio (extended AD).
    Pause,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_tag_parse() {
        let tag = LanguageTag::parse("en-US").expect("should succeed");
        assert_eq!(tag.language, "en");
        assert_eq!(tag.region, Some("US".to_string()));
        assert_eq!(tag.to_string(), "en-US");
    }

    #[test]
    fn test_language_tag_complex() {
        let tag = LanguageTag::parse("zh-Hans-CN").expect("should succeed");
        assert_eq!(tag.language, "zh");
        assert_eq!(tag.script, Some("Hans".to_string()));
        assert_eq!(tag.region, Some("CN".to_string()));
    }

    #[test]
    fn test_wcag_level_meets() {
        assert!(WcagLevel::AAA.meets(WcagLevel::A));
        assert!(WcagLevel::AAA.meets(WcagLevel::AA));
        assert!(WcagLevel::AAA.meets(WcagLevel::AAA));
        assert!(WcagLevel::AA.meets(WcagLevel::A));
        assert!(!WcagLevel::A.meets(WcagLevel::AA));
    }

    #[test]
    fn test_description_cue_duration() {
        let cue = DescriptionCue::new("cue1", 1.0, 3.5);
        assert!((cue.duration() - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metadata_validation() {
        let mut metadata = AudioDescriptionMetadata::default();
        assert!(metadata.validate().is_err());

        metadata.track_id = "track1".to_string();
        assert!(metadata.validate().is_ok());
    }

    #[test]
    fn test_cue_validation() {
        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test description");
        assert!(cue.validate().is_ok());

        let invalid_cue = DescriptionCue::new("", 1.0, 2.0);
        assert!(invalid_cue.validate().is_err());
    }
}
