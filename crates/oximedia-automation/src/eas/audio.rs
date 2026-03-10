//! EAS audio alert insertion.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, info};

/// EAS attention tone configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionToneConfig {
    /// Frequency in Hz (standard: 853 Hz and 960 Hz)
    pub frequencies: Vec<f32>,
    /// Duration in seconds
    pub duration: f32,
    /// Volume (0.0 - 1.0)
    pub volume: f32,
}

impl Default for AttentionToneConfig {
    fn default() -> Self {
        Self {
            frequencies: vec![853.0, 960.0],
            duration: 8.0,
            volume: 0.8,
        }
    }
}

/// EAS audio configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EasAudioConfig {
    /// Attention tone configuration
    pub attention_tone: AttentionToneConfig,
    /// Path to TTS audio files
    pub tts_audio_path: Option<PathBuf>,
    /// Enable end-of-message tones
    pub enable_eom_tones: bool,
}

impl Default for EasAudioConfig {
    fn default() -> Self {
        Self {
            attention_tone: AttentionToneConfig::default(),
            tts_audio_path: None,
            enable_eom_tones: true,
        }
    }
}

/// EAS audio insertion handler.
pub struct EasAudioInsertion {
    config: EasAudioConfig,
}

impl EasAudioInsertion {
    /// Create a new EAS audio insertion handler.
    pub fn new(config: EasAudioConfig) -> Self {
        info!("Creating EAS audio insertion handler");

        Self { config }
    }

    /// Generate attention tone.
    pub fn generate_attention_tone(&self) -> Result<Vec<f32>> {
        debug!("Generating EAS attention tone");

        let sample_rate = 48000.0;
        let duration = self.config.attention_tone.duration;
        let num_samples = (sample_rate * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let mut sample = 0.0;

            // Generate dual-tone
            for &freq in &self.config.attention_tone.frequencies {
                sample += (2.0 * std::f32::consts::PI * freq * t).sin();
            }

            // Normalize and apply volume
            sample *= self.config.attention_tone.volume
                / self.config.attention_tone.frequencies.len() as f32;

            samples.push(sample);
        }

        Ok(samples)
    }

    /// Generate end-of-message tone.
    pub fn generate_eom_tone(&self) -> Result<Vec<f32>> {
        debug!("Generating EAS end-of-message tone");

        let sample_rate = 48000.0;
        let duration = 3.0; // 3 seconds for EOM
        let num_samples = (sample_rate * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);

        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            // EOM is typically 853 Hz
            let sample =
                (2.0 * std::f32::consts::PI * 853.0 * t).sin() * self.config.attention_tone.volume;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Load TTS audio for message.
    pub fn load_tts_audio(&self, message: &str) -> Result<Vec<f32>> {
        debug!("Loading TTS audio for message: {}", message);

        // In a real implementation, this would:
        // 1. Generate or load pre-recorded TTS audio
        // 2. Return the audio samples

        // For now, return empty samples
        Ok(Vec::new())
    }

    /// Compose complete EAS audio message.
    pub fn compose_message(&self, message: &str) -> Result<Vec<f32>> {
        info!("Composing complete EAS audio message");

        let mut audio = Vec::new();

        // Add attention tone
        audio.extend(self.generate_attention_tone()?);

        // Add silence (1 second)
        audio.extend(vec![0.0; 48000]);

        // Add TTS message
        if let Some(ref _tts_path) = self.config.tts_audio_path {
            audio.extend(self.load_tts_audio(message)?);
        }

        // Add silence (1 second)
        audio.extend(vec![0.0; 48000]);

        // Add end-of-message tone
        if self.config.enable_eom_tones {
            audio.extend(self.generate_eom_tone()?);
        }

        Ok(audio)
    }

    /// Set attention tone volume.
    pub fn set_volume(&mut self, volume: f32) {
        self.config.attention_tone.volume = volume.clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_insertion_creation() {
        let config = EasAudioConfig::default();
        let insertion = EasAudioInsertion::new(config);
        assert_eq!(insertion.config.attention_tone.frequencies.len(), 2);
    }

    #[test]
    fn test_generate_attention_tone() {
        let config = EasAudioConfig::default();
        let insertion = EasAudioInsertion::new(config);

        let tone = insertion
            .generate_attention_tone()
            .expect("generate_attention_tone should succeed");
        assert!(!tone.is_empty());
        assert_eq!(tone.len(), 48000 * 8); // 8 seconds at 48kHz
    }

    #[test]
    fn test_generate_eom_tone() {
        let config = EasAudioConfig::default();
        let insertion = EasAudioInsertion::new(config);

        let tone = insertion
            .generate_eom_tone()
            .expect("generate_eom_tone should succeed");
        assert!(!tone.is_empty());
        assert_eq!(tone.len(), 48000 * 3); // 3 seconds at 48kHz
    }

    #[test]
    fn test_compose_message() {
        let config = EasAudioConfig::default();
        let insertion = EasAudioInsertion::new(config);

        let audio = insertion
            .compose_message("Test message")
            .expect("compose_message should succeed");
        assert!(!audio.is_empty());
    }
}
