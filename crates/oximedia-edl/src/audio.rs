//! Audio channel handling for EDL operations.
//!
//! This module provides audio channel mapping and routing functionality
//! for EDL files, supporting multi-channel audio configurations.

use crate::error::{EdlError, EdlResult};
use std::fmt;
use std::str::FromStr;

/// Audio channel identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum AudioChannel {
    /// Audio channel 1 (left in stereo).
    A1,
    /// Audio channel 2 (right in stereo).
    A2,
    /// Audio channel 3.
    A3,
    /// Audio channel 4.
    A4,
    /// Audio channel 5.
    A5,
    /// Audio channel 6.
    A6,
    /// Audio channel 7.
    A7,
    /// Audio channel 8.
    A8,
    /// Custom audio channel (9+).
    Custom(u8),
}

impl AudioChannel {
    /// Get the channel number (1-indexed).
    #[must_use]
    pub const fn number(&self) -> u8 {
        match self {
            Self::A1 => 1,
            Self::A2 => 2,
            Self::A3 => 3,
            Self::A4 => 4,
            Self::A5 => 5,
            Self::A6 => 6,
            Self::A7 => 7,
            Self::A8 => 8,
            Self::Custom(n) => *n,
        }
    }

    /// Create from channel number (1-indexed).
    #[must_use]
    pub const fn from_number(n: u8) -> Self {
        match n {
            1 => Self::A1,
            2 => Self::A2,
            3 => Self::A3,
            4 => Self::A4,
            5 => Self::A5,
            6 => Self::A6,
            7 => Self::A7,
            8 => Self::A8,
            _ => Self::Custom(n),
        }
    }

    /// Get the EDL track identifier string.
    #[must_use]
    pub fn track_id(&self) -> String {
        match self {
            Self::A1 => "A".to_string(),
            Self::A2 => "A2".to_string(),
            Self::A3 => "A3".to_string(),
            Self::A4 => "A4".to_string(),
            Self::A5 => "A5".to_string(),
            Self::A6 => "A6".to_string(),
            Self::A7 => "A7".to_string(),
            Self::A8 => "A8".to_string(),
            Self::Custom(n) => format!("A{n}"),
        }
    }
}

impl FromStr for AudioChannel {
    type Err = EdlError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim().to_uppercase();
        match trimmed.as_str() {
            "A" | "A1" => Ok(Self::A1),
            "A2" => Ok(Self::A2),
            "A3" => Ok(Self::A3),
            "A4" => Ok(Self::A4),
            "A5" => Ok(Self::A5),
            "A6" => Ok(Self::A6),
            "A7" => Ok(Self::A7),
            "A8" => Ok(Self::A8),
            _ => {
                // Try to parse A{n} format
                if let Some(rest) = trimmed.strip_prefix('A') {
                    if let Ok(n) = rest.parse::<u8>() {
                        if n > 0 {
                            return Ok(Self::from_number(n));
                        }
                    }
                }
                Err(EdlError::InvalidAudioChannel(s.to_string()))
            }
        }
    }
}

impl fmt::Display for AudioChannel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.track_id())
    }
}

/// Audio channel mapping for routing audio between tracks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioMapping {
    /// Source channel.
    pub source: AudioChannel,
    /// Destination channel.
    pub destination: AudioChannel,
    /// Gain adjustment in dB (0 = unity gain).
    pub gain_db: i16,
    /// Pan position (-100 = left, 0 = center, 100 = right).
    pub pan: i8,
}

impl AudioMapping {
    /// Create a new audio mapping with unity gain and center pan.
    #[must_use]
    pub const fn new(source: AudioChannel, destination: AudioChannel) -> Self {
        Self {
            source,
            destination,
            gain_db: 0,
            pan: 0,
        }
    }

    /// Create a new audio mapping with specified gain.
    #[must_use]
    pub const fn with_gain(source: AudioChannel, destination: AudioChannel, gain_db: i16) -> Self {
        Self {
            source,
            destination,
            gain_db,
            pan: 0,
        }
    }

    /// Create a new audio mapping with specified pan.
    #[must_use]
    pub const fn with_pan(source: AudioChannel, destination: AudioChannel, pan: i8) -> Self {
        Self {
            source,
            destination,
            gain_db: 0,
            pan,
        }
    }

    /// Set the gain in dB.
    pub fn set_gain(&mut self, gain_db: i16) {
        self.gain_db = gain_db;
    }

    /// Set the pan position.
    pub fn set_pan(&mut self, pan: i8) {
        self.pan = pan.clamp(-100, 100);
    }
}

/// Audio channel configuration for an EDL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioConfig {
    /// Number of audio channels.
    pub channel_count: u8,
    /// Audio sample rate in Hz.
    pub sample_rate: u32,
    /// Channel mappings.
    pub mappings: Vec<AudioMapping>,
}

impl AudioConfig {
    /// Create a new audio configuration.
    #[must_use]
    pub const fn new(channel_count: u8, sample_rate: u32) -> Self {
        Self {
            channel_count,
            sample_rate,
            mappings: Vec::new(),
        }
    }

    /// Create a stereo audio configuration (2 channels, 48kHz).
    #[must_use]
    pub const fn stereo() -> Self {
        Self::new(2, 48000)
    }

    /// Create a mono audio configuration (1 channel, 48kHz).
    #[must_use]
    pub const fn mono() -> Self {
        Self::new(1, 48000)
    }

    /// Create a 5.1 surround audio configuration (6 channels, 48kHz).
    #[must_use]
    pub const fn surround_5_1() -> Self {
        Self::new(6, 48000)
    }

    /// Add an audio mapping.
    pub fn add_mapping(&mut self, mapping: AudioMapping) {
        self.mappings.push(mapping);
    }

    /// Get all mappings for a specific source channel.
    #[must_use]
    pub fn get_mappings_for_source(&self, source: AudioChannel) -> Vec<&AudioMapping> {
        self.mappings
            .iter()
            .filter(|m| m.source == source)
            .collect()
    }

    /// Get all mappings for a specific destination channel.
    #[must_use]
    pub fn get_mappings_for_destination(&self, destination: AudioChannel) -> Vec<&AudioMapping> {
        self.mappings
            .iter()
            .filter(|m| m.destination == destination)
            .collect()
    }

    /// Validate the audio configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> EdlResult<()> {
        if self.channel_count == 0 {
            return Err(EdlError::ValidationError(
                "Channel count must be greater than 0".to_string(),
            ));
        }

        if self.sample_rate == 0 {
            return Err(EdlError::ValidationError(
                "Sample rate must be greater than 0".to_string(),
            ));
        }

        // Validate that all mapping channels are within range
        for mapping in &self.mappings {
            if mapping.source.number() > self.channel_count {
                return Err(EdlError::ValidationError(format!(
                    "Source channel {} exceeds channel count {}",
                    mapping.source.number(),
                    self.channel_count
                )));
            }
            if mapping.destination.number() > self.channel_count {
                return Err(EdlError::ValidationError(format!(
                    "Destination channel {} exceeds channel count {}",
                    mapping.destination.number(),
                    self.channel_count
                )));
            }
        }

        Ok(())
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::stereo()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_channel_parsing() {
        assert_eq!(
            "A".parse::<AudioChannel>()
                .expect("operation should succeed"),
            AudioChannel::A1
        );
        assert_eq!(
            "A1".parse::<AudioChannel>()
                .expect("operation should succeed"),
            AudioChannel::A1
        );
        assert_eq!(
            "A2".parse::<AudioChannel>()
                .expect("operation should succeed"),
            AudioChannel::A2
        );
        assert_eq!(
            "A8".parse::<AudioChannel>()
                .expect("operation should succeed"),
            AudioChannel::A8
        );
    }

    #[test]
    fn test_audio_channel_number() {
        assert_eq!(AudioChannel::A1.number(), 1);
        assert_eq!(AudioChannel::A2.number(), 2);
        assert_eq!(AudioChannel::A8.number(), 8);
        assert_eq!(AudioChannel::Custom(16).number(), 16);
    }

    #[test]
    fn test_audio_channel_from_number() {
        assert_eq!(AudioChannel::from_number(1), AudioChannel::A1);
        assert_eq!(AudioChannel::from_number(2), AudioChannel::A2);
        assert_eq!(AudioChannel::from_number(16), AudioChannel::Custom(16));
    }

    #[test]
    fn test_audio_channel_track_id() {
        assert_eq!(AudioChannel::A1.track_id(), "A");
        assert_eq!(AudioChannel::A2.track_id(), "A2");
        assert_eq!(AudioChannel::Custom(16).track_id(), "A16");
    }

    #[test]
    fn test_audio_mapping() {
        let mapping = AudioMapping::new(AudioChannel::A1, AudioChannel::A2);
        assert_eq!(mapping.source, AudioChannel::A1);
        assert_eq!(mapping.destination, AudioChannel::A2);
        assert_eq!(mapping.gain_db, 0);
        assert_eq!(mapping.pan, 0);
    }

    #[test]
    fn test_audio_config_stereo() {
        let config = AudioConfig::stereo();
        assert_eq!(config.channel_count, 2);
        assert_eq!(config.sample_rate, 48000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_audio_config_mappings() {
        let mut config = AudioConfig::stereo();
        config.add_mapping(AudioMapping::new(AudioChannel::A1, AudioChannel::A2));

        let mappings = config.get_mappings_for_source(AudioChannel::A1);
        assert_eq!(mappings.len(), 1);
        assert_eq!(mappings[0].destination, AudioChannel::A2);
    }

    #[test]
    fn test_audio_config_validation() {
        let mut config = AudioConfig::new(2, 48000);
        config.add_mapping(AudioMapping::new(
            AudioChannel::Custom(10),
            AudioChannel::A1,
        ));

        assert!(config.validate().is_err());
    }
}
