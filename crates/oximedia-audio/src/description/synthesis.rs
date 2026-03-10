//! Text-to-speech synthesis for audio description.
//!
//! This module provides text-to-speech integration for generating audio
//! description from text, including voice selection, SSML support, prosody
//! control, and pronunciation dictionaries.

#![forbid(unsafe_code)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]

use super::metadata::{LanguageTag, SpeakerInfo, VoiceCharacteristics};
use crate::AudioResult;
use std::collections::HashMap;

/// Text-to-speech engine trait.
pub trait TtsEngine {
    /// Synthesize text to audio samples.
    fn synthesize(&mut self, text: &str, config: &SynthesisConfig) -> AudioResult<Vec<f64>>;

    /// Get estimated duration for text in seconds.
    fn estimate_duration(&self, text: &str, config: &SynthesisConfig) -> f64;

    /// Check if engine supports a specific language.
    fn supports_language(&self, language: &LanguageTag) -> bool;

    /// Get available voices for a language.
    fn available_voices(&self, language: &LanguageTag) -> Vec<VoiceInfo>;
}

/// Configuration for text-to-speech synthesis.
#[derive(Clone, Debug)]
pub struct SynthesisConfig {
    /// Voice to use.
    pub voice: VoiceInfo,
    /// Speech rate (0.5-2.0, 1.0 = normal).
    pub rate: f64,
    /// Pitch adjustment in semitones (-12 to +12).
    pub pitch_semitones: f64,
    /// Volume (0.0-1.0).
    pub volume: f64,
    /// Sample rate for output.
    pub sample_rate: u32,
    /// Enable SSML processing.
    pub enable_ssml: bool,
    /// Pronunciation dictionary.
    pub pronunciation_dict: PronunciationDictionary,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            voice: VoiceInfo::default(),
            rate: 1.0,
            pitch_semitones: 0.0,
            volume: 1.0,
            sample_rate: 48000,
            enable_ssml: false,
            pronunciation_dict: PronunciationDictionary::new(),
        }
    }
}

impl SynthesisConfig {
    /// Create a new synthesis configuration.
    #[must_use]
    pub fn new(voice: VoiceInfo) -> Self {
        Self {
            voice,
            ..Default::default()
        }
    }

    /// Set speech rate.
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate.clamp(0.5, 2.0);
        self
    }

    /// Set pitch adjustment.
    #[must_use]
    pub fn with_pitch(mut self, semitones: f64) -> Self {
        self.pitch_semitones = semitones.clamp(-12.0, 12.0);
        self
    }

    /// Set volume.
    #[must_use]
    pub fn with_volume(mut self, volume: f64) -> Self {
        self.volume = volume.clamp(0.0, 1.0);
        self
    }

    /// Set sample rate.
    #[must_use]
    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Enable SSML processing.
    #[must_use]
    pub fn with_ssml(mut self, enabled: bool) -> Self {
        self.enable_ssml = enabled;
        self
    }

    /// Set pronunciation dictionary.
    #[must_use]
    pub fn with_pronunciation_dict(mut self, dict: PronunciationDictionary) -> Self {
        self.pronunciation_dict = dict;
        self
    }
}

/// Voice information.
#[derive(Clone, Debug)]
pub struct VoiceInfo {
    /// Voice identifier.
    pub id: String,
    /// Voice name.
    pub name: String,
    /// Language.
    pub language: LanguageTag,
    /// Voice characteristics.
    pub characteristics: VoiceCharacteristics,
    /// Whether this is a neural voice.
    pub is_neural: bool,
}

impl Default for VoiceInfo {
    fn default() -> Self {
        Self {
            id: String::from("default"),
            name: String::from("Default Voice"),
            language: LanguageTag::default(),
            characteristics: VoiceCharacteristics::default(),
            is_neural: false,
        }
    }
}

impl VoiceInfo {
    /// Create a new voice info.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>, language: LanguageTag) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            language,
            ..Default::default()
        }
    }

    /// Set voice characteristics.
    #[must_use]
    pub fn with_characteristics(mut self, characteristics: VoiceCharacteristics) -> Self {
        self.characteristics = characteristics;
        self
    }

    /// Set neural voice flag.
    #[must_use]
    pub fn with_neural(mut self, neural: bool) -> Self {
        self.is_neural = neural;
        self
    }

    /// Create voice from speaker info.
    #[must_use]
    pub fn from_speaker_info(speaker: &SpeakerInfo, language: LanguageTag) -> Self {
        Self {
            id: format!("speaker_{}", speaker.name),
            name: speaker.name.clone(),
            language,
            characteristics: speaker.voice_characteristics.clone(),
            is_neural: true,
        }
    }
}

/// Pronunciation dictionary for custom word pronunciations.
#[derive(Clone, Debug, Default)]
pub struct PronunciationDictionary {
    /// Word to pronunciation mapping.
    entries: HashMap<String, String>,
}

impl PronunciationDictionary {
    /// Create a new pronunciation dictionary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Add pronunciation entry.
    pub fn add(&mut self, word: impl Into<String>, pronunciation: impl Into<String>) {
        self.entries
            .insert(word.into().to_lowercase(), pronunciation.into());
    }

    /// Get pronunciation for word.
    #[must_use]
    pub fn get(&self, word: &str) -> Option<&str> {
        self.entries.get(&word.to_lowercase()).map(String::as_str)
    }

    /// Remove pronunciation entry.
    pub fn remove(&mut self, word: &str) -> bool {
        self.entries.remove(&word.to_lowercase()).is_some()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if dictionary is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Apply pronunciation rules to text.
    #[must_use]
    pub fn apply(&self, text: &str) -> String {
        let mut result = text.to_string();

        for (word, pronunciation) in &self.entries {
            let lower_word = word.to_lowercase();
            result = result
                .replace(&lower_word, pronunciation)
                .replace(&word.to_uppercase(), pronunciation)
                .replace(&Self::capitalize(&lower_word), pronunciation);
        }

        result
    }

    /// Capitalize first letter of word.
    fn capitalize(s: &str) -> String {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => first.to_uppercase().chain(chars).collect(),
        }
    }
}

/// SSML (Speech Synthesis Markup Language) processor.
pub struct SsmlProcessor {
    /// Base synthesis config.
    base_config: SynthesisConfig,
}

impl SsmlProcessor {
    /// Create a new SSML processor.
    #[must_use]
    pub fn new(base_config: SynthesisConfig) -> Self {
        Self { base_config }
    }

    /// Parse SSML and extract text with prosody information.
    pub fn parse(&self, ssml: &str) -> AudioResult<Vec<SsmlSegment>> {
        let mut segments = Vec::new();

        if ssml.trim().is_empty() {
            return Ok(segments);
        }

        let text = self.strip_tags(ssml);
        segments.push(SsmlSegment {
            text,
            rate: self.base_config.rate,
            pitch_semitones: self.base_config.pitch_semitones,
            volume: self.base_config.volume,
            pause_ms: 0.0,
        });

        Ok(segments)
    }

    /// Strip SSML tags and extract plain text.
    fn strip_tags(&self, ssml: &str) -> String {
        let mut result = String::new();
        let mut in_tag = false;

        for ch in ssml.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }

        result.trim().to_string()
    }

    /// Generate SSML from plain text with prosody.
    #[must_use]
    pub fn generate(&self, text: &str, rate: f64, pitch_semitones: f64, volume: f64) -> String {
        let rate_percent = (rate * 100.0) as i32;
        let pitch_str = if pitch_semitones >= 0.0 {
            format!("+{}st", pitch_semitones as i32)
        } else {
            format!("{}st", pitch_semitones as i32)
        };

        format!(
            "<speak><prosody rate=\"{}%\" pitch=\"{}\" volume=\"{}\">{}</prosody></speak>",
            rate_percent,
            pitch_str,
            (volume * 100.0) as i32,
            text
        )
    }
}

/// SSML segment with prosody information.
#[derive(Clone, Debug)]
pub struct SsmlSegment {
    /// Text content.
    pub text: String,
    /// Speech rate.
    pub rate: f64,
    /// Pitch adjustment in semitones.
    pub pitch_semitones: f64,
    /// Volume.
    pub volume: f64,
    /// Pause duration in milliseconds.
    pub pause_ms: f64,
}

/// Prosody control for speech synthesis.
#[derive(Clone, Debug)]
pub struct ProsodyControl {
    /// Base rate.
    base_rate: f64,
    /// Base pitch.
    base_pitch: f64,
    /// Base volume.
    base_volume: f64,
}

impl Default for ProsodyControl {
    fn default() -> Self {
        Self {
            base_rate: 1.0,
            base_pitch: 0.0,
            base_volume: 1.0,
        }
    }
}

impl ProsodyControl {
    /// Create new prosody control.
    #[must_use]
    pub fn new(rate: f64, pitch: f64, volume: f64) -> Self {
        Self {
            base_rate: rate.clamp(0.5, 2.0),
            base_pitch: pitch.clamp(-12.0, 12.0),
            base_volume: volume.clamp(0.0, 1.0),
        }
    }

    /// Apply emphasis to prosody.
    #[must_use]
    pub fn with_emphasis(&self, level: EmphasisLevel) -> Self {
        let (rate_mult, pitch_add) = match level {
            EmphasisLevel::None => (1.0, 0.0),
            EmphasisLevel::Reduced => (1.1, -1.0),
            EmphasisLevel::Moderate => (0.95, 2.0),
            EmphasisLevel::Strong => (0.9, 4.0),
        };

        Self {
            base_rate: (self.base_rate * rate_mult).clamp(0.5, 2.0),
            base_pitch: (self.base_pitch + pitch_add).clamp(-12.0, 12.0),
            base_volume: self.base_volume,
        }
    }

    /// Get rate.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.base_rate
    }

    /// Get pitch.
    #[must_use]
    pub fn pitch(&self) -> f64 {
        self.base_pitch
    }

    /// Get volume.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.base_volume
    }
}

/// Emphasis level for speech.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmphasisLevel {
    /// No emphasis.
    None,
    /// Reduced emphasis.
    Reduced,
    /// Moderate emphasis.
    Moderate,
    /// Strong emphasis.
    Strong,
}

/// Mock TTS engine for testing and fallback.
pub struct MockTtsEngine {
    /// Sample rate.
    sample_rate: u32,
}

impl MockTtsEngine {
    /// Create a new mock TTS engine.
    #[must_use]
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Estimate samples needed for text.
    fn estimate_samples(&self, text: &str, rate: f64) -> usize {
        let word_count = text.split_whitespace().count();
        let words_per_second = 2.5 * rate;
        let duration_seconds = word_count as f64 / words_per_second;
        (duration_seconds * self.sample_rate as f64) as usize
    }

    /// Generate simple tone as placeholder.
    fn generate_tone(&self, duration_samples: usize, frequency: f64) -> Vec<f64> {
        let mut samples = Vec::with_capacity(duration_samples);
        let phase_increment = 2.0 * std::f64::consts::PI * frequency / self.sample_rate as f64;

        for i in 0..duration_samples {
            let phase = i as f64 * phase_increment;
            let sample = (phase.sin() * 0.1).clamp(-0.1, 0.1);
            samples.push(sample);
        }

        samples
    }
}

impl TtsEngine for MockTtsEngine {
    fn synthesize(&mut self, text: &str, config: &SynthesisConfig) -> AudioResult<Vec<f64>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let sample_count = self.estimate_samples(text, config.rate);
        let mut samples = self.generate_tone(sample_count, 440.0);

        for sample in &mut samples {
            *sample *= config.volume;
        }

        Ok(samples)
    }

    fn estimate_duration(&self, text: &str, config: &SynthesisConfig) -> f64 {
        let word_count = text.split_whitespace().count();
        let words_per_second = 2.5 * config.rate;
        word_count as f64 / words_per_second
    }

    fn supports_language(&self, _language: &LanguageTag) -> bool {
        true
    }

    fn available_voices(&self, language: &LanguageTag) -> Vec<VoiceInfo> {
        vec![
            VoiceInfo::new("mock_male", "Mock Male", language.clone()),
            VoiceInfo::new("mock_female", "Mock Female", language.clone()),
        ]
    }
}

/// TTS synthesizer with caching and queueing.
pub struct TtsSynthesizer<E: TtsEngine> {
    /// TTS engine.
    engine: E,
    /// Synthesis configuration.
    config: SynthesisConfig,
    /// Cache of synthesized audio.
    cache: HashMap<String, Vec<f64>>,
    /// Maximum cache size.
    max_cache_size: usize,
}

impl<E: TtsEngine> TtsSynthesizer<E> {
    /// Create a new TTS synthesizer.
    #[must_use]
    pub fn new(engine: E, config: SynthesisConfig) -> Self {
        Self {
            engine,
            config,
            cache: HashMap::new(),
            max_cache_size: 100,
        }
    }

    /// Set synthesis configuration.
    pub fn set_config(&mut self, config: SynthesisConfig) {
        self.config = config;
    }

    /// Get current configuration.
    #[must_use]
    pub fn config(&self) -> &SynthesisConfig {
        &self.config
    }

    /// Synthesize text to audio.
    pub fn synthesize(&mut self, text: &str) -> AudioResult<Vec<f64>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let cache_key = self.make_cache_key(text);

        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        let processed_text = if self.config.enable_ssml {
            text.to_string()
        } else {
            self.config.pronunciation_dict.apply(text)
        };

        let samples = self.engine.synthesize(&processed_text, &self.config)?;

        if self.cache.len() >= self.max_cache_size {
            if let Some(key) = self.cache.keys().next().cloned() {
                self.cache.remove(&key);
            }
        }

        self.cache.insert(cache_key, samples.clone());

        Ok(samples)
    }

    /// Estimate duration for text.
    #[must_use]
    pub fn estimate_duration(&self, text: &str) -> f64 {
        self.engine.estimate_duration(text, &self.config)
    }

    /// Clear synthesis cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get cache size.
    #[must_use]
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Make cache key from text and config.
    fn make_cache_key(&self, text: &str) -> String {
        format!(
            "{}:{}:{}:{}",
            text, self.config.voice.id, self.config.rate, self.config.pitch_semitones
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_config() {
        let voice = VoiceInfo::default();
        let config = SynthesisConfig::new(voice).with_rate(1.5);
        assert!((config.rate - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_pronunciation_dict() {
        let mut dict = PronunciationDictionary::new();
        dict.add("NASA", "N-A-S-A");
        assert_eq!(dict.get("nasa"), Some("N-A-S-A"));
        assert_eq!(dict.len(), 1);
    }

    #[test]
    fn test_pronunciation_apply() {
        let mut dict = PronunciationDictionary::new();
        dict.add("NASA", "N-A-S-A");
        let result = dict.apply("NASA launched a rocket");
        assert!(result.contains("N-A-S-A"));
    }

    #[test]
    fn test_ssml_processor() {
        let config = SynthesisConfig::default();
        let processor = SsmlProcessor::new(config);

        let ssml = "<speak>Hello world</speak>";
        let segments = processor.parse(ssml).expect("parse should succeed");
        assert!(!segments.is_empty());
        assert_eq!(segments[0].text, "Hello world");
    }

    #[test]
    fn test_prosody_control() {
        let prosody = ProsodyControl::new(1.0, 0.0, 1.0);
        let emphasized = prosody.with_emphasis(EmphasisLevel::Strong);
        assert!(emphasized.pitch() > prosody.pitch());
    }

    #[test]
    fn test_mock_tts_engine() {
        let mut engine = MockTtsEngine::new(48000);
        let config = SynthesisConfig::default();

        let samples = engine
            .synthesize("Hello world", &config)
            .expect("should succeed");
        assert!(!samples.is_empty());

        let duration = engine.estimate_duration("Hello world", &config);
        assert!(duration > 0.0);
    }

    #[test]
    fn test_tts_synthesizer_cache() {
        let engine = MockTtsEngine::new(48000);
        let config = SynthesisConfig::default();
        let mut synthesizer = TtsSynthesizer::new(engine, config);

        let samples1 = synthesizer.synthesize("Hello").expect("should succeed");
        assert!(!samples1.is_empty());
        assert_eq!(synthesizer.cache_size(), 1);

        let samples2 = synthesizer.synthesize("Hello").expect("should succeed");
        assert_eq!(samples1.len(), samples2.len());
    }
}
