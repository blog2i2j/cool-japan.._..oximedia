//! WAV muxer implementation.

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use async_trait::async_trait;
use oximedia_core::{CodecId, OxiError, OxiResult};
use oximedia_io::MediaSource;
use std::io::SeekFrom;

use crate::mux::traits::{Muxer, MuxerConfig};
use crate::{Packet, StreamInfo};

// ============================================================================
// Constants
// ============================================================================

/// RIFF chunk ID.
const RIFF_ID: &[u8; 4] = b"RIFF";

/// WAVE format ID.
const WAVE_ID: &[u8; 4] = b"WAVE";

/// Format chunk ID.
const FMT_ID: &[u8; 4] = b"fmt ";

/// Data chunk ID.
const DATA_ID: &[u8; 4] = b"data";

/// FACT chunk ID (required for non-PCM formats).
#[allow(dead_code)]
const FACT_ID: &[u8; 4] = b"fact";

/// PCM format code.
const FORMAT_PCM: u16 = 1;

/// IEEE Float format code.
const FORMAT_IEEE_FLOAT: u16 = 3;

/// Extensible format code.
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// Maximum data size for 32-bit WAV.
const MAX_DATA_SIZE: u64 = u32::MAX as u64;

// ============================================================================
// WAV Format Configuration
// ============================================================================

/// WAV format type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WavFormat {
    /// Integer PCM.
    #[default]
    Pcm,
    /// IEEE floating point.
    Float,
    /// Extensible format (for >2 channels or >16 bits).
    Extensible,
}

/// WAV format configuration.
#[derive(Clone, Debug)]
pub struct WavFormatConfig {
    /// Sample format.
    pub format: WavFormat,

    /// Bits per sample (8, 16, 24, 32, 64).
    pub bits_per_sample: u16,

    /// Number of channels.
    pub channels: u16,

    /// Sample rate in Hz.
    pub sample_rate: u32,

    /// Channel mask for extensible format.
    pub channel_mask: Option<u32>,
}

impl WavFormatConfig {
    /// Creates a PCM format configuration.
    #[must_use]
    pub const fn pcm(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Self {
        Self {
            format: WavFormat::Pcm,
            bits_per_sample,
            channels,
            sample_rate,
            channel_mask: None,
        }
    }

    /// Creates a float format configuration.
    #[must_use]
    pub const fn float(sample_rate: u32, channels: u16, bits_per_sample: u16) -> Self {
        Self {
            format: WavFormat::Float,
            bits_per_sample,
            channels,
            sample_rate,
            channel_mask: None,
        }
    }

    /// Sets the channel mask for extensible format.
    #[must_use]
    pub const fn with_channel_mask(mut self, mask: u32) -> Self {
        self.channel_mask = Some(mask);
        self
    }

    /// Returns the format code for the WAV header.
    #[must_use]
    pub const fn format_code(&self) -> u16 {
        match self.format {
            WavFormat::Pcm => FORMAT_PCM,
            WavFormat::Float => FORMAT_IEEE_FLOAT,
            WavFormat::Extensible => FORMAT_EXTENSIBLE,
        }
    }

    /// Returns the block alignment (bytes per sample frame).
    #[must_use]
    pub const fn block_align(&self) -> u16 {
        self.channels * self.bits_per_sample.div_ceil(8)
    }

    /// Returns the byte rate (bytes per second).
    #[must_use]
    pub const fn byte_rate(&self) -> u32 {
        self.sample_rate * self.block_align() as u32
    }
}

impl Default for WavFormatConfig {
    fn default() -> Self {
        Self::pcm(44100, 2, 16)
    }
}

// ============================================================================
// WAV Muxer
// ============================================================================

/// WAV/RIFF container muxer.
///
/// Creates WAV files from PCM audio packets.
pub struct WavMuxer<W> {
    /// The underlying writer.
    sink: W,

    /// Muxer configuration.
    config: MuxerConfig,

    /// Registered streams (WAV only supports one stream).
    streams: Vec<StreamInfo>,

    /// WAV format configuration.
    wav_config: Option<WavFormatConfig>,

    /// Whether header has been written.
    header_written: bool,

    /// Current write position.
    position: u64,

    /// Position of RIFF size field (for fixup).
    riff_size_position: u64,

    /// Position of data size field (for fixup).
    data_size_position: u64,

    /// Total data bytes written.
    data_size: u64,
}

impl<W> WavMuxer<W> {
    /// Creates a new WAV muxer.
    ///
    /// # Arguments
    ///
    /// * `sink` - The writer to output to
    /// * `config` - Muxer configuration
    #[must_use]
    pub fn new(sink: W, config: MuxerConfig) -> Self {
        Self {
            sink,
            config,
            streams: Vec::new(),
            wav_config: None,
            header_written: false,
            position: 0,
            riff_size_position: 4,
            data_size_position: 0,
            data_size: 0,
        }
    }

    /// Creates a WAV muxer with a specific format configuration.
    #[must_use]
    pub fn with_format(sink: W, config: MuxerConfig, wav_config: WavFormatConfig) -> Self {
        Self {
            sink,
            config,
            streams: Vec::new(),
            wav_config: Some(wav_config),
            header_written: false,
            position: 0,
            riff_size_position: 4,
            data_size_position: 0,
            data_size: 0,
        }
    }

    /// Returns a reference to the underlying sink.
    #[must_use]
    pub const fn sink(&self) -> &W {
        &self.sink
    }

    /// Returns a mutable reference to the underlying sink.
    pub fn sink_mut(&mut self) -> &mut W {
        &mut self.sink
    }

    /// Consumes the muxer and returns the underlying sink.
    #[must_use]
    #[allow(dead_code)]
    pub fn into_sink(self) -> W {
        self.sink
    }

    /// Returns the total data size written.
    #[must_use]
    pub const fn data_size(&self) -> u64 {
        self.data_size
    }
}

impl<W: MediaSource> WavMuxer<W> {
    /// Writes bytes to the sink.
    async fn write_bytes(&mut self, data: &[u8]) -> OxiResult<()> {
        self.sink.write_all(data).await?;
        self.position += data.len() as u64;
        Ok(())
    }

    /// Builds the WAV format configuration from stream info.
    fn build_format_config(stream: &StreamInfo) -> WavFormatConfig {
        let sample_rate = stream.codec_params.sample_rate.unwrap_or(44100);
        let channels = u16::from(stream.codec_params.channels.unwrap_or(2));

        // Default to 16-bit PCM
        WavFormatConfig::pcm(sample_rate, channels, 16)
    }

    /// Writes the RIFF header.
    async fn write_riff_header(&mut self) -> OxiResult<()> {
        // RIFF chunk ID
        self.write_bytes(RIFF_ID).await?;

        // RIFF chunk size (placeholder - will be fixed up later)
        self.riff_size_position = self.position;
        self.write_bytes(&0u32.to_le_bytes()).await?;

        // WAVE format ID
        self.write_bytes(WAVE_ID).await?;

        Ok(())
    }

    /// Writes the format chunk.
    async fn write_format_chunk(&mut self) -> OxiResult<()> {
        let wav_config = self
            .wav_config
            .clone()
            .ok_or_else(|| OxiError::InvalidData("WAV format not configured".into()))?;

        // Determine if we need extensible format
        let use_extensible = wav_config.channels > 2
            || wav_config.bits_per_sample > 16
            || wav_config.channel_mask.is_some();

        // fmt chunk ID
        self.write_bytes(FMT_ID).await?;

        if use_extensible {
            // Extensible format chunk size: 40 bytes
            self.write_bytes(&40u32.to_le_bytes()).await?;

            // Format code: WAVE_FORMAT_EXTENSIBLE
            self.write_bytes(&FORMAT_EXTENSIBLE.to_le_bytes()).await?;
        } else {
            // Standard format chunk size: 16 bytes
            self.write_bytes(&16u32.to_le_bytes()).await?;

            // Format code
            self.write_bytes(&wav_config.format_code().to_le_bytes())
                .await?;
        }

        // Number of channels
        self.write_bytes(&wav_config.channels.to_le_bytes()).await?;

        // Sample rate
        self.write_bytes(&wav_config.sample_rate.to_le_bytes())
            .await?;

        // Byte rate
        self.write_bytes(&wav_config.byte_rate().to_le_bytes())
            .await?;

        // Block alignment
        self.write_bytes(&wav_config.block_align().to_le_bytes())
            .await?;

        // Bits per sample
        self.write_bytes(&wav_config.bits_per_sample.to_le_bytes())
            .await?;

        if use_extensible {
            // Extension size: 22 bytes
            self.write_bytes(&22u16.to_le_bytes()).await?;

            // Valid bits per sample
            self.write_bytes(&wav_config.bits_per_sample.to_le_bytes())
                .await?;

            // Channel mask
            let mask = wav_config
                .channel_mask
                .unwrap_or_else(|| default_channel_mask(wav_config.channels));
            self.write_bytes(&mask.to_le_bytes()).await?;

            // Sub-format GUID (PCM or IEEE Float)
            let subformat = if wav_config.format == WavFormat::Float {
                // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
                [
                    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00,
                    0x38, 0x9B, 0x71,
                ]
            } else {
                // KSDATAFORMAT_SUBTYPE_PCM
                [
                    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00,
                    0x38, 0x9B, 0x71,
                ]
            };
            self.write_bytes(&subformat).await?;
        }

        Ok(())
    }

    /// Writes the data chunk header.
    async fn write_data_chunk_header(&mut self) -> OxiResult<()> {
        // data chunk ID
        self.write_bytes(DATA_ID).await?;

        // data chunk size (placeholder - will be fixed up later)
        self.data_size_position = self.position;
        self.write_bytes(&0u32.to_le_bytes()).await?;

        Ok(())
    }

    /// Fixes up the RIFF and data chunk sizes.
    async fn fixup_sizes(&mut self) -> OxiResult<()> {
        // Calculate sizes
        let data_size = self.data_size.min(MAX_DATA_SIZE) as u32;
        let riff_size = (self.position - 8) as u32;

        // Save current position
        let current_pos = self.position;

        // Fix up RIFF size
        self.sink
            .seek(SeekFrom::Start(self.riff_size_position))
            .await?;
        self.sink.write_all(&riff_size.to_le_bytes()).await?;

        // Fix up data size
        self.sink
            .seek(SeekFrom::Start(self.data_size_position))
            .await?;
        self.sink.write_all(&data_size.to_le_bytes()).await?;

        // Seek back to end
        self.sink.seek(SeekFrom::Start(current_pos)).await?;

        Ok(())
    }
}

#[async_trait]
impl<W: MediaSource> Muxer for WavMuxer<W> {
    fn add_stream(&mut self, info: StreamInfo) -> OxiResult<usize> {
        if self.header_written {
            return Err(OxiError::InvalidData(
                "Cannot add stream after header is written".into(),
            ));
        }

        // WAV only supports one stream
        if !self.streams.is_empty() {
            return Err(OxiError::unsupported(
                "WAV format only supports a single audio stream",
            ));
        }

        // Validate codec is PCM
        if info.codec != CodecId::Pcm {
            return Err(OxiError::unsupported(format!(
                "WAV muxer only supports PCM codec, got {:?}",
                info.codec
            )));
        }

        // Build format config from stream info if not already set
        if self.wav_config.is_none() {
            self.wav_config = Some(Self::build_format_config(&info));
        }

        let index = self.streams.len();
        self.streams.push(info);
        Ok(index)
    }

    async fn write_header(&mut self) -> OxiResult<()> {
        if self.header_written {
            return Err(OxiError::InvalidData("Header already written".into()));
        }

        if self.streams.is_empty() {
            return Err(OxiError::InvalidData("No streams configured".into()));
        }

        self.write_riff_header().await?;
        self.write_format_chunk().await?;
        self.write_data_chunk_header().await?;

        self.header_written = true;
        Ok(())
    }

    async fn write_packet(&mut self, packet: &Packet) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        if packet.stream_index >= self.streams.len() {
            return Err(OxiError::InvalidData(format!(
                "Invalid stream index: {}",
                packet.stream_index
            )));
        }

        // Write raw PCM data
        self.write_bytes(&packet.data).await?;
        self.data_size += packet.data.len() as u64;

        Ok(())
    }

    async fn write_trailer(&mut self) -> OxiResult<()> {
        if !self.header_written {
            return Err(OxiError::InvalidData("Header not written".into()));
        }

        // Add padding byte if data size is odd
        if self.data_size % 2 != 0 {
            self.write_bytes(&[0]).await?;
        }

        // Fix up sizes
        self.fixup_sizes().await?;

        Ok(())
    }

    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }

    fn config(&self) -> &MuxerConfig {
        &self.config
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Returns the default channel mask for a given number of channels.
#[must_use]
fn default_channel_mask(channels: u16) -> u32 {
    match channels {
        1 => 0x0004, // Front Center
        2 => 0x0003, // Front Left + Front Right
        3 => 0x0007, // FL + FR + FC
        4 => 0x0033, // FL + FR + BL + BR
        5 => 0x0037, // FL + FR + FC + BL + BR
        6 => 0x003F, // FL + FR + FC + LFE + BL + BR
        7 => 0x013F, // FL + FR + FC + LFE + BL + BR + BC
        8 => 0x063F, // FL + FR + FC + LFE + BL + BR + SL + SR
        _ => 0,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use oximedia_core::{Rational, Timestamp};
    use oximedia_io::MemorySource;

    fn create_pcm_stream() -> StreamInfo {
        let mut stream = StreamInfo::new(0, CodecId::Pcm, Rational::new(1, 44100));
        stream.codec_params.sample_rate = Some(44100);
        stream.codec_params.channels = Some(2);
        stream
    }

    #[test]
    fn test_wav_format_config_pcm() {
        let config = WavFormatConfig::pcm(44100, 2, 16);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bits_per_sample, 16);
        assert_eq!(config.format_code(), FORMAT_PCM);
        assert_eq!(config.block_align(), 4);
        assert_eq!(config.byte_rate(), 176400);
    }

    #[test]
    fn test_wav_format_config_float() {
        let config = WavFormatConfig::float(48000, 2, 32);
        assert_eq!(config.format_code(), FORMAT_IEEE_FLOAT);
        assert_eq!(config.block_align(), 8);
        assert_eq!(config.byte_rate(), 384000);
    }

    #[test]
    fn test_default_channel_mask() {
        assert_eq!(default_channel_mask(1), 0x0004);
        assert_eq!(default_channel_mask(2), 0x0003);
        assert_eq!(default_channel_mask(6), 0x003F);
    }

    #[tokio::test]
    async fn test_muxer_new() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let muxer = WavMuxer::new(sink, config);

        assert!(!muxer.header_written);
        assert!(muxer.streams.is_empty());
    }

    #[tokio::test]
    async fn test_muxer_add_stream() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let pcm = create_pcm_stream();
        let idx = muxer.add_stream(pcm).expect("operation should succeed");

        assert_eq!(idx, 0);
        assert_eq!(muxer.streams.len(), 1);
    }

    #[tokio::test]
    async fn test_muxer_add_multiple_streams_fails() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let pcm1 = create_pcm_stream();
        let pcm2 = create_pcm_stream();

        muxer.add_stream(pcm1).expect("operation should succeed");
        let result = muxer.add_stream(pcm2);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_muxer_add_non_pcm_fails() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let opus = StreamInfo::new(0, CodecId::Opus, Rational::new(1, 48000));
        let result = muxer.add_stream(opus);

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_muxer_write_header() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let pcm = create_pcm_stream();
        muxer.add_stream(pcm).expect("operation should succeed");

        let result = muxer.write_header().await;
        assert!(result.is_ok());
        assert!(muxer.header_written);
    }

    #[tokio::test]
    async fn test_muxer_write_packet() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let pcm = create_pcm_stream();
        muxer.add_stream(pcm).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let packet = Packet::new(
            0,
            Bytes::from(vec![0u8; 1024]),
            Timestamp::new(0, Rational::new(1, 44100)),
            crate::PacketFlags::KEYFRAME,
        );

        let result = muxer.write_packet(&packet).await;
        assert!(result.is_ok());
        assert_eq!(muxer.data_size(), 1024);
    }

    #[tokio::test]
    async fn test_muxer_write_trailer() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let mut muxer = WavMuxer::new(sink, config);

        let pcm = create_pcm_stream();
        muxer.add_stream(pcm).expect("operation should succeed");
        muxer
            .write_header()
            .await
            .expect("operation should succeed");

        let packet = Packet::new(
            0,
            Bytes::from(vec![0u8; 1024]),
            Timestamp::new(0, Rational::new(1, 44100)),
            crate::PacketFlags::KEYFRAME,
        );
        muxer
            .write_packet(&packet)
            .await
            .expect("operation should succeed");

        let result = muxer.write_trailer().await;
        assert!(result.is_ok());
    }

    /// Regression test for GitHub issue #7:
    /// WavFormatConfig must be publicly constructable so that callers of
    /// `WavMuxer::with_format` can actually pass a configured value.
    #[test]
    fn test_issue_7_wav_format_config_public() {
        // Float constructor: used in the reporter's minimal repro
        let float_cfg = WavFormatConfig::float(16_000, 1, 32);
        assert_eq!(float_cfg.sample_rate, 16_000);
        assert_eq!(float_cfg.channels, 1);
        assert_eq!(float_cfg.bits_per_sample, 32);
        assert_eq!(float_cfg.format, WavFormat::Float);
        assert_eq!(float_cfg.format_code(), FORMAT_IEEE_FLOAT);
        assert_eq!(float_cfg.block_align(), 4); // 1 ch * 4 bytes
        assert_eq!(float_cfg.byte_rate(), 64_000); // 16000 * 4

        // PCM constructor: the only variant accessible before the fix
        let pcm_cfg = WavFormatConfig::pcm(16_000, 1, 16);
        assert_eq!(pcm_cfg.sample_rate, 16_000);
        assert_eq!(pcm_cfg.channels, 1);
        assert_eq!(pcm_cfg.bits_per_sample, 16);
        assert_eq!(pcm_cfg.format, WavFormat::Pcm);
        assert_eq!(pcm_cfg.format_code(), FORMAT_PCM);
        assert_eq!(pcm_cfg.block_align(), 2); // 1 ch * 2 bytes
        assert_eq!(pcm_cfg.byte_rate(), 32_000); // 16000 * 2

        // Channel-mask builder is also public
        let with_mask = WavFormatConfig::float(48_000, 2, 32).with_channel_mask(0x0003);
        assert_eq!(with_mask.channel_mask, Some(0x0003));
    }

    #[tokio::test]
    async fn test_muxer_with_format() {
        let sink = MemorySource::new_writable(4096);
        let config = MuxerConfig::new();
        let wav_config = WavFormatConfig::float(96000, 2, 32);
        let muxer = WavMuxer::with_format(sink, config, wav_config);

        assert!(muxer.wav_config.is_some());
        let wav_cfg = muxer.wav_config.as_ref().expect("operation should succeed");
        assert_eq!(wav_cfg.sample_rate, 96000);
        assert_eq!(wav_cfg.format, WavFormat::Float);
    }
}
