//! FLAC codec implementation.
//!
//! FLAC (Free Lossless Audio Codec) provides lossless audio compression.
//! It typically achieves 50-70% compression of the original size.
//!
//! # Modules
//!
//! - [`frame`] - Frame parsing and encoding
//! - [`subframe`] - Subframe decoding and encoding
//! - [`rice`] - Rice coding for residuals
//! - [`encoder`] - FLAC encoder implementation
//! - [`crc`] - CRC calculation
//! - [`bitwriter`] - Bit-level writing

#![forbid(unsafe_code)]

pub mod bitwriter;
pub mod crc;
pub mod encoder;
pub mod frame;
pub mod rice;
pub mod subframe;

use crate::{AudioDecoder, AudioDecoderConfig, AudioError, AudioFrame, AudioResult, ChannelLayout};
use oximedia_core::{CodecId, SampleFormat};

// Re-export submodule types
pub use bitwriter::BitWriter;
pub use crc::{crc16, crc8, Crc16, Crc8};
pub use encoder::{CompressionLevel, FlacEncoder};
pub use frame::{BlockingStrategy, ChannelAssignment, FlacFrame, FrameHeader};
pub use rice::{RiceDecoder, RicePartition};
pub use subframe::{Subframe, SubframeHeader, SubframeType, WarmupSamples};

/// FLAC stream info metadata.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct StreamInfo {
    /// Minimum block size in samples.
    pub min_block_size: u16,
    /// Maximum block size in samples.
    pub max_block_size: u16,
    /// Minimum frame size in bytes (0 = unknown).
    pub min_frame_size: u32,
    /// Maximum frame size in bytes (0 = unknown).
    pub max_frame_size: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Total samples (0 = unknown).
    pub total_samples: u64,
    /// MD5 signature of uncompressed audio.
    pub md5_signature: [u8; 16],
}

impl StreamInfo {
    /// Parse STREAMINFO from metadata block data.
    ///
    /// # Errors
    ///
    /// Returns error if data is invalid.
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        if data.len() < 34 {
            return Err(AudioError::InvalidData("STREAMINFO too short".into()));
        }

        let min_block_size = u16::from_be_bytes([data[0], data[1]]);
        let max_block_size = u16::from_be_bytes([data[2], data[3]]);
        let min_frame_size = u32::from_be_bytes([0, data[4], data[5], data[6]]);
        let max_frame_size = u32::from_be_bytes([0, data[7], data[8], data[9]]);

        // Sample rate (20 bits) + channels (3 bits) + bps (5 bits) + total samples (36 bits)
        let sample_rate =
            (u32::from(data[10]) << 12) | (u32::from(data[11]) << 4) | (u32::from(data[12]) >> 4);

        let channels = ((data[12] >> 1) & 0x07) + 1;
        let bits_per_sample = (((data[12] & 0x01) << 4) | (data[13] >> 4)) + 1;

        let total_samples = (u64::from(data[13] & 0x0F) << 32)
            | (u64::from(data[14]) << 24)
            | (u64::from(data[15]) << 16)
            | (u64::from(data[16]) << 8)
            | u64::from(data[17]);

        let mut md5_signature = [0u8; 16];
        md5_signature.copy_from_slice(&data[18..34]);

        Ok(Self {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5_signature,
        })
    }

    /// Get duration in seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_secs(&self) -> Option<f64> {
        if self.sample_rate == 0 || self.total_samples == 0 {
            None
        } else {
            Some(self.total_samples as f64 / f64::from(self.sample_rate))
        }
    }
}

/// FLAC decoder.
pub struct FlacDecoder {
    #[allow(dead_code)]
    config: AudioDecoderConfig,
    sample_rate: u32,
    channels: u8,
    flushing: bool,
    #[allow(dead_code)]
    stream_info: Option<StreamInfo>,
    #[allow(dead_code)]
    current_frame: Option<FlacFrame>,
}

impl FlacDecoder {
    /// Create new FLAC decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Flac {
            return Err(AudioError::InvalidParameter("Expected FLAC codec".into()));
        }
        Ok(Self {
            config: config.clone(),
            sample_rate: config.sample_rate,
            channels: config.channels,
            flushing: false,
            stream_info: None,
            current_frame: None,
        })
    }

    /// Parse STREAMINFO metadata.
    ///
    /// # Errors
    ///
    /// Returns error if data is invalid.
    pub fn parse_stream_info(data: &[u8]) -> AudioResult<StreamInfo> {
        StreamInfo::parse(data)
    }

    /// Parse a frame header.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse_frame_header(&self, data: &[u8]) -> AudioResult<(FrameHeader, usize)> {
        let bps = self.stream_info.as_ref().map_or(16, |s| s.bits_per_sample);
        FrameHeader::parse(data, bps)
    }
}

impl AudioDecoder for FlacDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Flac
    }

    fn send_packet(&mut self, _data: &[u8], _pts: i64) -> AudioResult<()> {
        Ok(())
    }

    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>> {
        Ok(None)
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.flushing = false;
        self.current_frame = None;
    }

    fn output_format(&self) -> Option<SampleFormat> {
        Some(SampleFormat::S16)
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.sample_rate)
    }

    fn channel_layout(&self) -> Option<ChannelLayout> {
        Some(ChannelLayout::from_count(usize::from(self.channels)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flac_decoder() {
        let config = AudioDecoderConfig {
            codec: CodecId::Flac,
            ..Default::default()
        };
        let decoder = FlacDecoder::new(&config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Flac);
    }

    #[test]
    fn test_flac_decoder_wrong_codec() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        assert!(FlacDecoder::new(&config).is_err());
    }

    #[test]
    fn test_flac_decoder_reset() {
        let config = AudioDecoderConfig {
            codec: CodecId::Flac,
            ..Default::default()
        };
        let mut decoder = FlacDecoder::new(&config).expect("should succeed");
        decoder.reset();
        assert!(!decoder.flushing);
    }

    #[test]
    fn test_stream_info_parse() {
        // Minimal valid STREAMINFO
        let mut data = vec![0u8; 34];
        // min_block_size = 4096
        data[0] = 0x10;
        data[1] = 0x00;
        // max_block_size = 4096
        data[2] = 0x10;
        data[3] = 0x00;
        // Sample rate = 44100 (0x00AC44), channels = 2 (1), bps = 16 (15)
        // Byte layout:
        // data[10]: bits [19:12] of sample rate = 0x0A
        // data[11]: bits [11:4] of sample rate = 0xC4
        // data[12]: bits [3:0] of sample rate (upper 4 bits) | channels-1 (bits [3:1]) | bps high bit (bit 0)
        // data[13]: bits [3:0] of bps (upper 4 bits) | bits [35:32] of total samples (lower 4 bits)
        //
        // 44100 = 0x0AC44 (20 bits)
        // bits [3:0] = 4
        // channels = 2, so channels-1 = 1
        // bps = 16, so bps-1 = 15 = 0b01111, high bit = 0, low 4 bits = 0xF
        data[10] = 0x0A; // upper 8 bits of sample rate
        data[11] = 0xC4; // middle 8 bits of sample rate
        data[12] = (4 << 4) | (1 << 1); // low 4 bits of sample rate + channels + bps high bit
        data[13] = 0xF << 4; // bps low 4 bits + total samples high nibble

        let info = StreamInfo::parse(&data).expect("should succeed");
        assert_eq!(info.min_block_size, 4096);
        assert_eq!(info.max_block_size, 4096);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
    }

    #[test]
    fn test_stream_info_duration() {
        let mut info = StreamInfo::default();
        info.sample_rate = 44100;
        info.total_samples = 44100 * 60; // 1 minute

        let duration = info.duration_secs().expect("should succeed");
        assert!((duration - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_stream_info_duration_unknown() {
        let info = StreamInfo::default();
        assert!(info.duration_secs().is_none());
    }
}
