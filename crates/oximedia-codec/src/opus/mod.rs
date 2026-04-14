//! Opus audio codec decoder.
//!
//! Opus is a modern, royalty-free audio codec designed for interactive
//! speech and music transmission over the Internet. It combines SILK
//! (for speech) and CELT (for music) to provide excellent quality
//! across a wide range of bitrates and content types.
//!
//! # Features
//!
//! - **SILK mode**: Optimized for speech (narrowband to wideband)
//! - **CELT mode**: Optimized for music (narrowband to fullband)
//! - **Hybrid mode**: Combines SILK and CELT for mixed content
//! - **Adaptive bandwidth**: 4 kHz (narrowband) to 20 kHz (fullband)
//! - **Low latency**: Frame sizes from 2.5ms to 60ms
//! - **Scalable complexity**: Adjustable CPU usage
//!
//! # Architecture
//!
//! This implementation provides:
//!
//! - Complete packet parsing and frame structure handling
//! - Range decoder for entropy coding
//! - MDCT transforms for CELT mode
//! - Basic CELT decoder for mono audio
//! - Stub implementations for SILK and hybrid modes
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::opus::OpusDecoder;
//!
//! let mut decoder = OpusDecoder::new(48000, 2)?;
//! let output = decoder.decode_packet(&packet_data)?;
//! ```
//!
//! # References
//!
//! - RFC 6716: Definition of the Opus Audio Codec
//! - <https://opus-codec.org/>

pub mod celt;
pub mod encoder;
pub mod hybrid;
pub mod mdct;
pub mod packet;
pub mod range_decoder;
pub mod range_encoder;
pub mod silk;
pub mod vad;

use crate::{AudioFrame, CodecError, CodecResult, SampleFormat};

use celt::CeltDecoder;
use hybrid::HybridDecoder;
use packet::{OpusBandwidth, OpusMode, OpusPacket};
use silk::SilkDecoder;

// Re-export encoder types
pub use encoder::{OpusEncoder, OpusEncoderConfig};

/// Opus decoder configuration.
#[derive(Debug, Clone)]
pub struct OpusConfig {
    /// Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
    pub sample_rate: u32,
    /// Number of channels (1 or 2)
    pub channels: usize,
    /// Output sample format
    pub sample_format: SampleFormat,
}

impl Default for OpusConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channels: 2,
            sample_format: SampleFormat::F32,
        }
    }
}

/// Opus audio decoder.
///
/// Decodes Opus-compressed audio packets to PCM samples.
pub struct OpusDecoder {
    /// Configuration
    config: OpusConfig,
    /// SILK decoder (for speech mode)
    silk: Option<SilkDecoder>,
    /// CELT decoder (for music mode)
    celt: Option<CeltDecoder>,
    /// Hybrid decoder (for mixed mode)
    hybrid: Option<HybridDecoder>,
    /// Current operating mode
    current_mode: Option<OpusMode>,
    /// Frame counter
    frame_count: u64,
}

impl OpusDecoder {
    /// Creates a new Opus decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
    /// * `channels` - Number of channels (1 or 2)
    pub fn new(sample_rate: u32, channels: usize) -> CodecResult<Self> {
        Self::with_config(OpusConfig {
            sample_rate,
            channels,
            sample_format: SampleFormat::F32,
        })
    }

    /// Creates a new Opus decoder with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Decoder configuration
    pub fn with_config(config: OpusConfig) -> CodecResult<Self> {
        // Validate sample rate
        if !matches!(config.sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(CodecError::InvalidData(format!(
                "Invalid sample rate: {}",
                config.sample_rate
            )));
        }

        // Validate channels
        if config.channels == 0 || config.channels > 2 {
            return Err(CodecError::InvalidData(format!(
                "Invalid channel count: {}",
                config.channels
            )));
        }

        Ok(Self {
            config,
            silk: None,
            celt: None,
            hybrid: None,
            current_mode: None,
            frame_count: 0,
        })
    }

    /// Decodes an Opus packet to audio samples.
    ///
    /// # Arguments
    ///
    /// * `data` - Opus packet data
    pub fn decode_packet(&mut self, data: &[u8]) -> CodecResult<AudioFrame> {
        // Parse packet
        let packet = OpusPacket::parse(data)?;

        // Determine frame size
        let frame_size = packet.toc.frame_size as usize;

        // Initialize appropriate decoder if needed
        self.initialize_decoder(&packet.toc.mode, packet.toc.bandwidth, frame_size)?;

        // Allocate output buffer
        let sample_count = frame_size * packet.frame_count();
        let mut samples = vec![0.0f32; sample_count * self.config.channels];

        // Decode each frame
        let mut offset = 0;
        for frame_data in &packet.frames {
            let frame_samples = frame_size * self.config.channels;
            let output_slice = &mut samples[offset..offset + frame_samples];

            self.decode_frame(&packet.toc.mode, frame_data, output_slice, frame_size)?;

            offset += frame_samples;
            self.frame_count += 1;
        }

        // Convert to output format
        let output_samples = self.convert_samples(&samples)?;

        Ok(AudioFrame::new(
            output_samples,
            sample_count,
            self.config.sample_rate,
            self.config.channels,
            self.config.sample_format,
        ))
    }

    /// Initializes the appropriate decoder for the given mode.
    fn initialize_decoder(
        &mut self,
        mode: &OpusMode,
        bandwidth: OpusBandwidth,
        frame_size: usize,
    ) -> CodecResult<()> {
        // Only initialize if mode changed or decoder doesn't exist
        if self.current_mode.as_ref() != Some(mode) {
            match mode {
                OpusMode::Silk => {
                    if self.silk.is_none() {
                        self.silk = Some(SilkDecoder::new(
                            self.config.sample_rate,
                            self.config.channels,
                            bandwidth,
                        ));
                    }
                }
                OpusMode::Celt => {
                    if self.celt.is_none() {
                        self.celt = Some(CeltDecoder::new(
                            self.config.sample_rate,
                            self.config.channels,
                            bandwidth,
                            frame_size,
                        ));
                    }
                }
                OpusMode::Hybrid => {
                    if self.hybrid.is_none() {
                        self.hybrid = Some(HybridDecoder::new(
                            self.config.sample_rate,
                            self.config.channels,
                            bandwidth,
                            frame_size,
                        ));
                    }
                }
            }
            self.current_mode = Some(*mode);
        }

        Ok(())
    }

    /// Decodes a single frame.
    fn decode_frame(
        &mut self,
        mode: &OpusMode,
        data: &[u8],
        output: &mut [f32],
        frame_size: usize,
    ) -> CodecResult<()> {
        match mode {
            OpusMode::Silk => {
                if let Some(silk) = &mut self.silk {
                    silk.decode(data, output, frame_size)?;
                } else {
                    return Err(CodecError::InvalidData(
                        "SILK decoder not initialized".to_string(),
                    ));
                }
            }
            OpusMode::Celt => {
                if let Some(celt) = &mut self.celt {
                    celt.decode(data, output, frame_size)?;
                } else {
                    return Err(CodecError::InvalidData(
                        "CELT decoder not initialized".to_string(),
                    ));
                }
            }
            OpusMode::Hybrid => {
                if let Some(hybrid) = &mut self.hybrid {
                    // For hybrid mode, we need to split the data between SILK and CELT
                    // This is a simplification - real implementation would parse the split
                    let mid = data.len() / 2;
                    hybrid.decode(&data[..mid], &data[mid..], output, frame_size)?;
                } else {
                    return Err(CodecError::InvalidData(
                        "Hybrid decoder not initialized".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Converts f32 samples to the configured output format.
    fn convert_samples(&self, samples: &[f32]) -> CodecResult<Vec<u8>> {
        match self.config.sample_format {
            SampleFormat::F32 => {
                // Convert f32 slice to bytes
                let mut output = Vec::with_capacity(samples.len() * 4);
                for &sample in samples {
                    output.extend_from_slice(&sample.to_le_bytes());
                }
                Ok(output)
            }
            SampleFormat::I16 => {
                // Convert to i16
                let mut output = Vec::with_capacity(samples.len() * 2);
                for &sample in samples {
                    let i16_sample = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                    output.extend_from_slice(&i16_sample.to_le_bytes());
                }
                Ok(output)
            }
            SampleFormat::I32 => {
                // Convert to i32
                let mut output = Vec::with_capacity(samples.len() * 4);
                for &sample in samples {
                    let i32_sample = (sample.clamp(-1.0, 1.0) * 2_147_483_647.0) as i32;
                    output.extend_from_slice(&i32_sample.to_le_bytes());
                }
                Ok(output)
            }
            SampleFormat::U8 => {
                // Convert to u8
                let mut output = Vec::with_capacity(samples.len());
                for &sample in samples {
                    let u8_sample = ((sample.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8;
                    output.push(u8_sample);
                }
                Ok(output)
            }
        }
    }

    /// Resets decoder state.
    pub fn reset(&mut self) {
        if let Some(silk) = &mut self.silk {
            silk.reset();
        }
        if let Some(celt) = &mut self.celt {
            celt.reset();
        }
        if let Some(hybrid) = &mut self.hybrid {
            hybrid.reset();
        }
        self.current_mode = None;
        self.frame_count = 0;
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &OpusConfig {
        &self.config
    }

    /// Returns the number of frames decoded.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Returns the current operating mode.
    #[must_use]
    pub const fn current_mode(&self) -> Option<OpusMode> {
        self.current_mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_opus_decoder_invalid_sample_rate() {
        let decoder = OpusDecoder::new(44100, 2);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_opus_decoder_invalid_channels() {
        let decoder = OpusDecoder::new(48000, 0);
        assert!(decoder.is_err());
    }

    #[test]
    fn test_opus_config_default() {
        let config = OpusConfig::default();
        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.sample_format, SampleFormat::F32);
    }

    #[test]
    fn test_opus_decoder_reset() {
        let mut decoder = OpusDecoder::new(48000, 2).expect("should succeed");
        decoder.reset();
        assert_eq!(decoder.frame_count(), 0);
    }
}
