//! Audio description track mixing for accessibility.
//!
//! This module provides comprehensive audio description (AD) functionality for making
//! video content accessible to people with visual impairments. It implements:
//!
//! - **Multiple mixing strategies**: Replace, Mix, Duck, and Pause modes
//! - **WCAG 2.1 compliance**: Support for accessibility standards
//! - **DVS compatibility**: Descriptive Video Service format support
//! - **Frame-accurate synchronization**: Precise timing control
//! - **Automatic ducking**: Intelligent main audio reduction
//! - **Text-to-speech integration**: Generate AD from text
//! - **Broadcast quality**: Professional audio processing
//!
//! # Example
//!
//! ```
//! use oximedia_audio::description::{
//!     AudioDescriptionMixer, MixingStrategy, AudioDescriptionMetadata, LanguageTag,
//! };
//!
//! # fn example() -> oximedia_audio::AudioResult<()> {
//! // Create metadata for AD track
//! let metadata = AudioDescriptionMetadata::new("ad_track_1", LanguageTag::new("en"))
//!     .with_label("English Audio Description");
//!
//! // Create mixer with Duck strategy
//! let mut mixer = AudioDescriptionMixer::new(
//!     MixingStrategy::Duck,
//!     metadata,
//!     48000.0,
//!     2,
//! );
//!
//! // Mix main audio with AD
//! let main_audio = vec![0.5; 1000];
//! let ad_audio = vec![0.3; 1000];
//! let mut output = vec![0.0; 1000];
//!
//! mixer.process(&main_audio, &ad_audio, &mut output, true)?;
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod ducking;
pub mod metadata;
pub mod mixing;
pub mod synthesis;
pub mod timing;

use crate::{AudioError, AudioResult};
pub use ducking::{AutomaticDucker, DuckingConfig, VoiceActivityDetector};
pub use metadata::{
    AudioDescriptionMetadata, ContentClassification, CueMixingStrategy, DescriptionCue,
    LanguageTag, SpeakerAge, SpeakerGender, SpeakerInfo, TimingMode, VoiceCharacteristics,
    WcagLevel,
};
pub use mixing::{AudioMixer, MixingConfig, MixingMode};
pub use synthesis::{
    EmphasisLevel, MockTtsEngine, PronunciationDictionary, ProsodyControl, SsmlProcessor,
    SynthesisConfig, TtsEngine, TtsSynthesizer, VoiceInfo,
};
pub use timing::{
    CueScheduler, SubtitleSync, TimeGap, Timeline, TimingConfig, TimingIssue, TimingPrecision,
    TimingValidation,
};

/// Audio description mixing strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MixingStrategy {
    /// Replace main audio completely with AD.
    Replace,
    /// Mix AD with main audio at specified levels.
    Mix,
    /// Duck (lower) main audio when AD is active.
    #[default]
    Duck,
    /// Pause main audio during AD (extended AD).
    Pause,
}

impl From<MixingStrategy> for MixingMode {
    fn from(strategy: MixingStrategy) -> Self {
        match strategy {
            MixingStrategy::Replace => MixingMode::Replace,
            MixingStrategy::Mix => MixingMode::Mix,
            MixingStrategy::Duck => MixingMode::Duck,
            MixingStrategy::Pause => MixingMode::Pause,
        }
    }
}

/// Audio description mixer - main entry point for AD functionality.
///
/// This is the primary interface for integrating audio description into your
/// application. It manages mixing, timing, and synthesis.
pub struct AudioDescriptionMixer {
    /// Metadata for the AD track.
    metadata: AudioDescriptionMetadata,
    /// Audio mixer.
    mixer: AudioMixer,
    /// Timeline for cue management.
    timeline: Timeline,
    /// TTS synthesizer (optional).
    tts: Option<TtsSynthesizer<MockTtsEngine>>,
    /// Sample rate.
    sample_rate: f64,
    /// Number of channels.
    channels: usize,
}

impl AudioDescriptionMixer {
    /// Create a new audio description mixer.
    ///
    /// # Arguments
    ///
    /// * `strategy` - Mixing strategy to use
    /// * `metadata` - AD track metadata
    /// * `sample_rate` - Sample rate in Hz
    /// * `channels` - Number of audio channels
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_audio::description::{
    ///     AudioDescriptionMixer, MixingStrategy, AudioDescriptionMetadata, LanguageTag,
    /// };
    ///
    /// let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
    /// let mixer = AudioDescriptionMixer::new(
    ///     MixingStrategy::Duck,
    ///     metadata,
    ///     48000.0,
    ///     2,
    /// );
    /// ```
    #[must_use]
    pub fn new(
        strategy: MixingStrategy,
        metadata: AudioDescriptionMetadata,
        sample_rate: f64,
        channels: usize,
    ) -> Self {
        let mixing_config = match strategy {
            MixingStrategy::Replace => MixingConfig::new(MixingMode::Replace),
            MixingStrategy::Mix => MixingConfig::gentle(),
            MixingStrategy::Duck => MixingConfig::broadcast(),
            MixingStrategy::Pause => MixingConfig::new(MixingMode::Pause),
        };

        let mixer = AudioMixer::new(mixing_config, sample_rate, channels);

        let timing_config = if metadata.timing_mode == TimingMode::Extended {
            TimingConfig::flexible()
        } else {
            TimingConfig::broadcast()
        };

        let timeline = Timeline::new(timing_config, sample_rate);

        Self {
            metadata,
            mixer,
            timeline,
            tts: None,
            sample_rate,
            channels,
        }
    }

    /// Get metadata reference.
    #[must_use]
    pub fn metadata(&self) -> &AudioDescriptionMetadata {
        &self.metadata
    }

    /// Get mutable metadata reference.
    pub fn metadata_mut(&mut self) -> &mut AudioDescriptionMetadata {
        &mut self.metadata
    }

    /// Get timeline reference.
    #[must_use]
    pub fn timeline(&self) -> &Timeline {
        &self.timeline
    }

    /// Get mutable timeline reference.
    pub fn timeline_mut(&mut self) -> &mut Timeline {
        &mut self.timeline
    }

    /// Set mixing configuration.
    pub fn set_mixing_config(&mut self, config: MixingConfig) {
        self.mixer.set_config(config);
    }

    /// Get mixing configuration.
    #[must_use]
    pub fn mixing_config(&self) -> &MixingConfig {
        self.mixer.config()
    }

    /// Enable text-to-speech synthesis.
    pub fn enable_tts(&mut self, config: SynthesisConfig) {
        let engine = MockTtsEngine::new(self.sample_rate as u32);
        self.tts = Some(TtsSynthesizer::new(engine, config));
    }

    /// Disable text-to-speech synthesis.
    pub fn disable_tts(&mut self) {
        self.tts = None;
    }

    /// Add a cue point to the timeline.
    pub fn add_cue(&mut self, cue: DescriptionCue) -> AudioResult<()> {
        cue.validate()?;
        self.timeline.add_cue(cue)
    }

    /// Remove a cue point from the timeline.
    pub fn remove_cue(&mut self, cue_id: &str) -> bool {
        self.timeline.remove_cue(cue_id)
    }

    /// Get active cue at specific time.
    #[must_use]
    pub fn get_cue_at(&self, time: f64) -> Option<&DescriptionCue> {
        self.timeline.get_cue_at(time)
    }

    /// Validate timeline and metadata.
    pub fn validate(&self) -> AudioResult<()> {
        self.metadata.validate()?;

        let timing_validation = self.timeline.validate();
        if !timing_validation.is_valid {
            return Err(AudioError::InvalidParameter(format!(
                "Timeline validation failed: {} issues",
                timing_validation.issues.len()
            )));
        }

        Ok(())
    }

    /// Process audio with AD mixing (interleaved samples).
    ///
    /// # Arguments
    ///
    /// * `main_audio` - Main audio buffer
    /// * `ad_audio` - Audio description buffer
    /// * `output` - Output buffer (must be same size as main_audio)
    /// * `ad_active` - Whether AD is currently active
    pub fn process(
        &mut self,
        main_audio: &[f64],
        ad_audio: &[f64],
        output: &mut [f64],
        ad_active: bool,
    ) -> AudioResult<()> {
        let num_samples = main_audio.len() / self.channels;
        self.mixer
            .mix_interleaved(main_audio, ad_audio, output, num_samples, ad_active)
    }

    /// Process audio with AD mixing (planar samples).
    ///
    /// # Arguments
    ///
    /// * `main_channels` - Main audio channels
    /// * `ad_channels` - Audio description channels
    /// * `output_channels` - Output channels
    /// * `ad_active` - Whether AD is currently active
    pub fn process_planar(
        &mut self,
        main_channels: &[Vec<f64>],
        ad_channels: &[Vec<f64>],
        output_channels: &mut [Vec<f64>],
        ad_active: bool,
    ) -> AudioResult<()> {
        self.mixer
            .mix_planar(main_channels, ad_channels, output_channels, ad_active)
    }

    /// Synthesize cue text to audio using TTS.
    pub fn synthesize_cue(&mut self, cue_id: &str) -> AudioResult<Vec<f64>> {
        let tts = self
            .tts
            .as_mut()
            .ok_or_else(|| AudioError::InvalidParameter("TTS not enabled".to_string()))?;

        let cues = self.timeline.get_all_cues();
        let cue = cues
            .iter()
            .find(|c| c.id == cue_id)
            .ok_or_else(|| AudioError::InvalidParameter(format!("Cue not found: {}", cue_id)))?;

        let text = cue
            .text
            .as_deref()
            .ok_or_else(|| AudioError::InvalidParameter("Cue has no text".to_string()))?;

        tts.synthesize(text)
    }

    /// Get peak output level (for metering).
    #[must_use]
    pub fn peak_level(&self) -> f64 {
        self.mixer.peak_level()
    }

    /// Reset all processing state.
    pub fn reset(&mut self) {
        self.mixer.reset();
        if let Some(ref mut tts) = self.tts {
            tts.clear_cache();
        }
    }

    /// Get sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get channel count.
    #[must_use]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

/// Audio description mixer builder for convenient construction.
pub struct AudioDescriptionMixerBuilder {
    strategy: MixingStrategy,
    metadata: AudioDescriptionMetadata,
    sample_rate: f64,
    channels: usize,
    mixing_config: Option<MixingConfig>,
    timing_config: Option<TimingConfig>,
    tts_config: Option<SynthesisConfig>,
}

impl AudioDescriptionMixerBuilder {
    /// Create a new builder.
    #[must_use]
    pub fn new(metadata: AudioDescriptionMetadata, sample_rate: f64, channels: usize) -> Self {
        Self {
            strategy: MixingStrategy::Duck,
            metadata,
            sample_rate,
            channels,
            mixing_config: None,
            timing_config: None,
            tts_config: None,
        }
    }

    /// Set mixing strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: MixingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set mixing configuration.
    #[must_use]
    pub fn with_mixing_config(mut self, config: MixingConfig) -> Self {
        self.mixing_config = Some(config);
        self
    }

    /// Set timing configuration.
    #[must_use]
    pub fn with_timing_config(mut self, config: TimingConfig) -> Self {
        self.timing_config = Some(config);
        self
    }

    /// Enable TTS with configuration.
    #[must_use]
    pub fn with_tts(mut self, config: SynthesisConfig) -> Self {
        self.tts_config = Some(config);
        self
    }

    /// Build the audio description mixer.
    #[must_use]
    pub fn build(self) -> AudioDescriptionMixer {
        let mut mixer = AudioDescriptionMixer::new(
            self.strategy,
            self.metadata,
            self.sample_rate,
            self.channels,
        );

        if let Some(config) = self.mixing_config {
            mixer.set_mixing_config(config);
        }

        if let Some(config) = self.timing_config {
            mixer.timeline = Timeline::new(config, self.sample_rate);
        }

        if let Some(config) = self.tts_config {
            mixer.enable_tts(config);
        }

        mixer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_description_mixer_creation() {
        let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        let mixer = AudioDescriptionMixer::new(MixingStrategy::Duck, metadata, 48000.0, 2);

        assert_eq!(mixer.sample_rate(), 48000.0);
        assert_eq!(mixer.channels(), 2);
    }

    #[test]
    fn test_mixer_add_cue() {
        let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        let mut mixer = AudioDescriptionMixer::new(MixingStrategy::Duck, metadata, 48000.0, 2);

        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test description");
        mixer.add_cue(cue).expect("should succeed");

        assert_eq!(mixer.timeline().cue_count(), 1);
    }

    #[test]
    fn test_mixer_process() {
        let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        let mut mixer = AudioDescriptionMixer::new(MixingStrategy::Mix, metadata, 48000.0, 2);

        let main_audio = vec![0.5; 200];
        let ad_audio = vec![0.3; 200];
        let mut output = vec![0.0; 200];

        mixer
            .process(&main_audio, &ad_audio, &mut output, false)
            .expect("should succeed");

        assert!(output.iter().any(|&s| s > 0.0));
    }

    #[test]
    fn test_mixer_builder() {
        let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        let mixer = AudioDescriptionMixerBuilder::new(metadata, 48000.0, 2)
            .with_strategy(MixingStrategy::Duck)
            .build();

        assert_eq!(mixer.sample_rate(), 48000.0);
    }

    #[test]
    fn test_mixer_validation() {
        let mut metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        metadata.track_id = String::new();

        let mixer = AudioDescriptionMixer::new(MixingStrategy::Duck, metadata, 48000.0, 2);
        assert!(mixer.validate().is_err());
    }

    #[test]
    fn test_mixer_with_tts() {
        let metadata = AudioDescriptionMetadata::new("track1", LanguageTag::new("en"));
        let mut mixer = AudioDescriptionMixer::new(MixingStrategy::Duck, metadata, 48000.0, 2);

        let tts_config = SynthesisConfig::default();
        mixer.enable_tts(tts_config);

        let cue = DescriptionCue::new("cue1", 1.0, 2.0).with_text("Test");
        mixer.add_cue(cue).expect("should succeed");

        let samples = mixer.synthesize_cue("cue1").expect("should succeed");
        assert!(!samples.is_empty());
    }
}
