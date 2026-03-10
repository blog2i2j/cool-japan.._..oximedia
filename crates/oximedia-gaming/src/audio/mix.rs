//! Multi-source audio mixing.

use crate::{GamingError, GamingResult};

/// Audio mixer for combining multiple audio sources.
#[allow(dead_code)]
pub struct AudioMixer {
    /// Mixer configuration.
    pub config: MixerConfig,
    sources: Vec<AudioSource>,
}

/// Mixer configuration.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MixerConfig {
    /// Output sample rate
    pub sample_rate: u32,
    /// Output channels
    pub channels: u32,
    /// Master volume
    pub master_volume: f32,
}

/// Audio source.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct AudioSource {
    /// Source name
    pub name: String,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Muted
    pub muted: bool,
}

impl AudioMixer {
    /// Create a new audio mixer.
    pub fn new(config: MixerConfig) -> GamingResult<Self> {
        if config.channels == 0 {
            return Err(GamingError::InvalidConfig(
                "Channels must be non-zero".to_string(),
            ));
        }

        Ok(Self {
            config,
            sources: Vec::new(),
        })
    }

    /// Add an audio source.
    pub fn add_source(&mut self, source: AudioSource) {
        self.sources.push(source);
    }

    /// Remove an audio source.
    pub fn remove_source(&mut self, name: &str) {
        self.sources.retain(|s| s.name != name);
    }

    /// Set source volume.
    pub fn set_source_volume(&mut self, name: &str, volume: f32) -> GamingResult<()> {
        let source = self
            .sources
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| GamingError::AudioMixingError(format!("Source not found: {name}")))?;

        source.volume = volume.clamp(0.0, 1.0);
        Ok(())
    }

    /// Mute/unmute a source.
    pub fn set_source_mute(&mut self, name: &str, muted: bool) -> GamingResult<()> {
        let source = self
            .sources
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| GamingError::AudioMixingError(format!("Source not found: {name}")))?;

        source.muted = muted;
        Ok(())
    }

    /// Get number of sources.
    #[must_use]
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }
}

impl Default for MixerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            master_volume: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_creation() {
        let config = MixerConfig::default();
        let mixer = AudioMixer::new(config).expect("valid audio mixer");
        assert_eq!(mixer.source_count(), 0);
    }

    #[test]
    fn test_add_remove_source() {
        let mut mixer = AudioMixer::new(MixerConfig::default()).expect("valid audio mixer");

        let source = AudioSource {
            name: "Game".to_string(),
            volume: 1.0,
            muted: false,
        };

        mixer.add_source(source);
        assert_eq!(mixer.source_count(), 1);

        mixer.remove_source("Game");
        assert_eq!(mixer.source_count(), 0);
    }

    #[test]
    fn test_set_volume() {
        let mut mixer = AudioMixer::new(MixerConfig::default()).expect("valid audio mixer");

        mixer.add_source(AudioSource {
            name: "Game".to_string(),
            volume: 1.0,
            muted: false,
        });

        mixer
            .set_source_volume("Game", 0.5)
            .expect("set volume should succeed");
    }
}
