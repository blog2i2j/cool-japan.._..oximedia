//! Delivery and export functionality for audio post-production.

use crate::error::{AudioPostError, AudioPostResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Audio format for export
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    /// WAV (uncompressed)
    Wav,
    /// FLAC (lossless)
    Flac,
    /// MP3 (lossy)
    Mp3,
    /// AAC (lossy)
    Aac,
    /// Opus (lossy)
    Opus,
    /// Broadcast Wave Format
    Bwf,
}

impl AudioFormat {
    /// Get file extension
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Wav | Self::Bwf => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Aac => "m4a",
            Self::Opus => "opus",
        }
    }

    /// Check if format is lossless
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Wav | Self::Flac | Self::Bwf)
    }
}

/// Sample rate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleRate {
    /// 44.1 kHz
    Hz44100,
    /// 48 kHz
    Hz48000,
    /// 88.2 kHz
    Hz88200,
    /// 96 kHz
    Hz96000,
    /// 176.4 kHz
    Hz176400,
    /// 192 kHz
    Hz192000,
}

impl SampleRate {
    /// Get sample rate value
    #[must_use]
    pub fn value(&self) -> u32 {
        match self {
            Self::Hz44100 => 44100,
            Self::Hz48000 => 48000,
            Self::Hz88200 => 88200,
            Self::Hz96000 => 96000,
            Self::Hz176400 => 176400,
            Self::Hz192000 => 192000,
        }
    }

    /// Create from value
    #[must_use]
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            44100 => Some(Self::Hz44100),
            48000 => Some(Self::Hz48000),
            88200 => Some(Self::Hz88200),
            96000 => Some(Self::Hz96000),
            176400 => Some(Self::Hz176400),
            192000 => Some(Self::Hz192000),
            _ => None,
        }
    }
}

/// Bit depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDepth {
    /// 16-bit integer
    Bit16,
    /// 24-bit integer
    Bit24,
    /// 32-bit float
    Float32,
}

impl BitDepth {
    /// Get bit depth value
    #[must_use]
    pub fn value(&self) -> u16 {
        match self {
            Self::Bit16 => 16,
            Self::Bit24 => 24,
            Self::Float32 => 32,
        }
    }

    /// Check if floating point
    #[must_use]
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float32)
    }
}

/// Channel configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChannelConfig {
    /// Mono
    Mono,
    /// Stereo
    Stereo,
    /// 5.1 surround
    Surround51,
    /// 7.1 surround
    Surround71,
    /// 7.1.4 Atmos
    Atmos714,
}

impl ChannelConfig {
    /// Get channel count
    #[must_use]
    pub fn channel_count(&self) -> usize {
        match self {
            Self::Mono => 1,
            Self::Stereo => 2,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
            Self::Atmos714 => 12,
        }
    }
}

/// Dithering algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DitheringAlgorithm {
    /// No dithering
    None,
    /// Triangular PDF dithering
    Triangular,
    /// Rectangular PDF dithering
    Rectangular,
    /// Noise shaping
    NoiseShaping,
}

/// Normalization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationType {
    /// No normalization
    None,
    /// Peak normalization
    Peak,
    /// RMS normalization
    Rms,
    /// LUFS normalization
    Lufs,
}

/// Export settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSettings {
    /// Output file path
    pub output_path: PathBuf,
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Channel configuration
    pub channels: ChannelConfig,
    /// Dithering algorithm
    pub dithering: DitheringAlgorithm,
    /// Normalization type
    pub normalization: NormalizationType,
    /// Normalization target
    pub normalization_target: f32,
    /// Embed metadata
    pub embed_metadata: bool,
}

impl ExportSettings {
    /// Create new export settings
    #[must_use]
    pub fn new(output_path: PathBuf) -> Self {
        Self {
            output_path,
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            dithering: DitheringAlgorithm::None,
            normalization: NormalizationType::None,
            normalization_target: -1.0,
            embed_metadata: true,
        }
    }

    /// Validate settings
    ///
    /// # Errors
    ///
    /// Returns an error if settings are invalid
    pub fn validate(&self) -> AudioPostResult<()> {
        // Check format compatibility
        if !self.format.is_lossless() && self.bit_depth == BitDepth::Float32 {
            return Err(AudioPostError::Generic(
                "Lossy formats do not support float32".to_string(),
            ));
        }

        // Check channel compatibility
        if matches!(self.format, AudioFormat::Mp3)
            && !matches!(self.channels, ChannelConfig::Mono | ChannelConfig::Stereo)
        {
            return Err(AudioPostError::Generic(
                "MP3 only supports mono/stereo".to_string(),
            ));
        }

        Ok(())
    }
}

/// Broadcast Wave Format metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BwfMetadata {
    /// Description
    pub description: String,
    /// Originator
    pub originator: String,
    /// Originator reference
    pub originator_reference: String,
    /// Origination date (YYYY-MM-DD)
    pub origination_date: String,
    /// Origination time (HH:MM:SS)
    pub origination_time: String,
    /// Time reference (samples since midnight)
    pub time_reference: u64,
    /// Coding history
    pub coding_history: String,
}

impl BwfMetadata {
    /// Create new BWF metadata
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set current date and time
    pub fn set_current_datetime(&mut self) {
        let now = chrono::Utc::now();
        self.origination_date = now.format("%Y-%m-%d").to_string();
        self.origination_time = now.format("%H:%M:%S").to_string();
    }
}

/// iXML metadata for production
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IxmlMetadata {
    /// Project name
    pub project: String,
    /// Scene name
    pub scene: String,
    /// Take number
    pub take: String,
    /// Tape name
    pub tape: String,
    /// File UID
    pub file_uid: String,
    /// Track list
    pub tracks: Vec<TrackMetadata>,
}

/// Track metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackMetadata {
    /// Track number
    pub number: u32,
    /// Track name
    pub name: String,
    /// Channel index
    pub channel_index: u32,
}

impl TrackMetadata {
    /// Create new track metadata
    #[must_use]
    pub fn new(number: u32, name: &str, channel_index: u32) -> Self {
        Self {
            number,
            name: name.to_string(),
            channel_index,
        }
    }
}

/// Deliverable specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliverableSpec {
    /// Specification name
    pub name: String,
    /// Audio format
    pub format: AudioFormat,
    /// Sample rate
    pub sample_rate: SampleRate,
    /// Bit depth
    pub bit_depth: BitDepth,
    /// Channel configuration
    pub channels: ChannelConfig,
    /// Loudness target (LUFS)
    pub loudness_target: Option<f32>,
    /// Maximum true peak (dBTP)
    pub max_true_peak: Option<f32>,
    /// Required metadata
    pub require_metadata: bool,
}

impl DeliverableSpec {
    /// Netflix specification
    #[must_use]
    pub fn netflix() -> Self {
        Self {
            name: "Netflix".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Surround51,
            loudness_target: Some(-27.0),
            max_true_peak: Some(-2.0),
            require_metadata: true,
        }
    }

    /// Theatrical DCP specification
    #[must_use]
    pub fn theatrical_dcp() -> Self {
        Self {
            name: "Theatrical DCP".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Surround51,
            loudness_target: Some(-27.0),
            max_true_peak: Some(-10.0),
            require_metadata: false,
        }
    }

    /// Broadcast specification
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            name: "Broadcast".to_string(),
            format: AudioFormat::Bwf,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            loudness_target: Some(-23.0),
            max_true_peak: Some(-1.0),
            require_metadata: true,
        }
    }

    /// Streaming specification
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            name: "Streaming".to_string(),
            format: AudioFormat::Wav,
            sample_rate: SampleRate::Hz48000,
            bit_depth: BitDepth::Bit24,
            channels: ChannelConfig::Stereo,
            loudness_target: Some(-14.0),
            max_true_peak: Some(-1.0),
            require_metadata: false,
        }
    }
}

/// Exporter for creating deliverables
#[derive(Debug)]
pub struct Exporter {
    settings: ExportSettings,
}

impl Exporter {
    /// Create a new exporter
    #[must_use]
    pub fn new(settings: ExportSettings) -> Self {
        Self { settings }
    }

    /// Validate export settings
    ///
    /// # Errors
    ///
    /// Returns an error if settings are invalid
    pub fn validate(&self) -> AudioPostResult<()> {
        self.settings.validate()
    }

    /// Export audio
    ///
    /// # Errors
    ///
    /// Returns an error if export fails
    pub fn export(&self, _audio: &[f32]) -> AudioPostResult<()> {
        self.validate()?;
        // Placeholder - real implementation would write audio file
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_format_extension() {
        assert_eq!(AudioFormat::Wav.extension(), "wav");
        assert_eq!(AudioFormat::Flac.extension(), "flac");
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
    }

    #[test]
    fn test_audio_format_lossless() {
        assert!(AudioFormat::Wav.is_lossless());
        assert!(AudioFormat::Flac.is_lossless());
        assert!(!AudioFormat::Mp3.is_lossless());
    }

    #[test]
    fn test_sample_rate_value() {
        assert_eq!(SampleRate::Hz48000.value(), 48000);
        assert_eq!(SampleRate::Hz96000.value(), 96000);
    }

    #[test]
    fn test_sample_rate_from_value() {
        assert_eq!(SampleRate::from_value(48000), Some(SampleRate::Hz48000));
        assert_eq!(SampleRate::from_value(12345), None);
    }

    #[test]
    fn test_bit_depth_value() {
        assert_eq!(BitDepth::Bit16.value(), 16);
        assert_eq!(BitDepth::Bit24.value(), 24);
        assert_eq!(BitDepth::Float32.value(), 32);
    }

    #[test]
    fn test_bit_depth_is_float() {
        assert!(!BitDepth::Bit16.is_float());
        assert!(BitDepth::Float32.is_float());
    }

    #[test]
    fn test_channel_config_count() {
        assert_eq!(ChannelConfig::Mono.channel_count(), 1);
        assert_eq!(ChannelConfig::Stereo.channel_count(), 2);
        assert_eq!(ChannelConfig::Surround51.channel_count(), 6);
    }

    #[test]
    fn test_export_settings_new() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        assert_eq!(settings.format, AudioFormat::Wav);
        assert_eq!(settings.sample_rate, SampleRate::Hz48000);
    }

    #[test]
    fn test_export_settings_validate() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        assert!(settings.validate().is_ok());
    }

    #[test]
    fn test_invalid_lossy_float() {
        let mut settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.mp3"));
        settings.format = AudioFormat::Mp3;
        settings.bit_depth = BitDepth::Float32;
        assert!(settings.validate().is_err());
    }

    #[test]
    fn test_bwf_metadata() {
        let mut metadata = BwfMetadata::new();
        metadata.description = "Test audio".to_string();
        metadata.set_current_datetime();
        assert!(!metadata.origination_date.is_empty());
    }

    #[test]
    fn test_ixml_metadata() {
        let mut metadata = IxmlMetadata::default();
        metadata.project = "Test Project".to_string();
        metadata.tracks.push(TrackMetadata::new(1, "Boom", 0));
        assert_eq!(metadata.tracks.len(), 1);
    }

    #[test]
    fn test_deliverable_spec_netflix() {
        let spec = DeliverableSpec::netflix();
        assert_eq!(spec.name, "Netflix");
        assert_eq!(spec.loudness_target, Some(-27.0));
    }

    #[test]
    fn test_deliverable_spec_broadcast() {
        let spec = DeliverableSpec::broadcast();
        assert_eq!(spec.format, AudioFormat::Bwf);
        assert_eq!(spec.loudness_target, Some(-23.0));
    }

    #[test]
    fn test_exporter_creation() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        let exporter = Exporter::new(settings);
        assert!(exporter.validate().is_ok());
    }

    #[test]
    fn test_exporter_export() {
        let settings =
            ExportSettings::new(std::env::temp_dir().join("oximedia-audiopost-delivery-out.wav"));
        let exporter = Exporter::new(settings);
        let audio = vec![0.0_f32; 1000];
        assert!(exporter.export(&audio).is_ok());
    }

    #[test]
    fn test_track_metadata() {
        let track = TrackMetadata::new(1, "Dialogue", 0);
        assert_eq!(track.number, 1);
        assert_eq!(track.name, "Dialogue");
    }
}
