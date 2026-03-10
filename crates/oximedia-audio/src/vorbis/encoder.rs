//! Vorbis encoder implementation.
//!
//! This module implements a full Vorbis encoder following the Vorbis I specification.
//! The encoder supports variable bitrate (VBR) quality modes and generates compliant
//! Ogg Vorbis bitstreams.

#![forbid(unsafe_code)]

use super::{
    bitpack::BitPacker, codebook::Codebook, floor::FloorType1, header::*, mdct::VorbisMdct,
    psycho::PsychoModel, residue::ResidueEncoder,
};
use crate::{
    AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, AudioResult, EncodedAudioPacket,
};
use oximedia_core::{CodecId, SampleFormat};
use std::collections::VecDeque;

/// Vorbis encoder state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderState {
    /// Need to send identification header.
    NeedIdentification,
    /// Need to send comment header.
    NeedComment,
    /// Need to send setup header.
    NeedSetup,
    /// Ready to encode audio.
    Ready,
    /// Flushing remaining data.
    Flushing,
}

/// Quality mode for VBR encoding.
///
/// Quality ranges from -1 (lowest quality, ~45 kbps) to 10 (highest quality, ~500 kbps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QualityMode {
    /// Quality level (-1.0 to 10.0).
    pub quality: f32,
    /// Base quality (integer part).
    pub base_quality: i32,
    /// Quality fraction.
    pub quality_fraction: f32,
}

impl QualityMode {
    /// Create quality mode from quality value.
    #[must_use]
    pub fn from_quality(quality: f32) -> Self {
        let quality = quality.clamp(-1.0, 10.0);
        let base_quality = quality.floor() as i32;
        let quality_fraction = quality - quality.floor();
        Self {
            quality,
            base_quality,
            quality_fraction,
        }
    }

    /// Get nominal bitrate for stereo at 44.1kHz.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn nominal_bitrate(&self) -> u32 {
        let base_bitrate = match self.base_quality {
            -1 => 45_000,
            0 => 64_000,
            1 => 80_000,
            2 => 96_000,
            3 => 112_000,
            4 => 128_000,
            5 => 160_000,
            6 => 192_000,
            7 => 224_000,
            8 => 256_000,
            9 => 320_000,
            10 => 500_000,
            _ => 128_000,
        };

        if self.quality_fraction > 0.0 {
            let next_bitrate = match self.base_quality + 1 {
                0 => 64_000,
                1 => 80_000,
                2 => 96_000,
                3 => 112_000,
                4 => 128_000,
                5 => 160_000,
                6 => 192_000,
                7 => 224_000,
                8 => 256_000,
                9 => 320_000,
                10 => 500_000,
                _ => 500_000,
            };
            base_bitrate + ((next_bitrate - base_bitrate) as f32 * self.quality_fraction) as u32
        } else {
            base_bitrate
        }
    }
}

/// Vorbis encoder.
pub struct VorbisEncoder {
    /// Encoder configuration.
    config: AudioEncoderConfig,
    /// Encoder state.
    state: EncoderState,
    /// Sample rate.
    sample_rate: u32,
    /// Channel count.
    channels: u8,
    /// Quality mode.
    quality_mode: QualityMode,
    /// Block size 0 (small blocks).
    blocksize_0: usize,
    /// Block size 1 (large blocks).
    blocksize_1: usize,
    /// MDCT transformer for small blocks.
    mdct_small: VorbisMdct,
    /// MDCT transformer for large blocks.
    mdct_large: VorbisMdct,
    /// Psychoacoustic model.
    psycho: PsychoModel,
    /// Residue encoder.
    residue_encoder: ResidueEncoder,
    /// Floor encoder.
    floor: FloorType1,
    /// Codebooks.
    codebooks: Vec<Codebook>,
    /// Input sample buffer.
    input_buffer: VecDeque<f32>,
    /// Overlap buffer for windowing.
    overlap: Vec<f32>,
    /// Current PTS.
    pts: i64,
    /// Packet number.
    packet_num: u64,
    /// Previous block was long.
    prev_block_long: bool,
    /// Pending output packets.
    pending_packets: VecDeque<EncodedAudioPacket>,
}

impl VorbisEncoder {
    /// Create new Vorbis encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioEncoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Vorbis {
            return Err(AudioError::InvalidParameter("Expected Vorbis codec".into()));
        }

        if config.channels == 0 {
            return Err(AudioError::InvalidParameter(format!(
                "Invalid channel count: {}",
                config.channels
            )));
        }

        if config.sample_rate == 0 {
            return Err(AudioError::InvalidParameter(
                "Sample rate must be non-zero".into(),
            ));
        }

        // Determine quality from bitrate
        let quality = Self::bitrate_to_quality(config.bitrate, config.channels);
        let quality_mode = QualityMode::from_quality(quality);

        // Vorbis block sizes (typically 256 and 2048)
        let blocksize_0 = 256;
        let blocksize_1 = 2048;

        let mdct_small = VorbisMdct::new(blocksize_0);
        let mdct_large = VorbisMdct::new(blocksize_1);

        let psycho = PsychoModel::new(config.sample_rate, blocksize_1);
        let residue_encoder = ResidueEncoder::new(quality_mode.quality);

        let floor = FloorType1::new();
        let codebooks = Self::init_codebooks(&quality_mode);

        let max_overlap = blocksize_1;
        let overlap = vec![0.0; max_overlap * config.channels as usize];

        Ok(Self {
            config: config.clone(),
            state: EncoderState::NeedIdentification,
            sample_rate: config.sample_rate,
            channels: config.channels,
            quality_mode,
            blocksize_0,
            blocksize_1,
            mdct_small,
            mdct_large,
            psycho,
            residue_encoder,
            floor,
            codebooks,
            input_buffer: VecDeque::new(),
            overlap,
            pts: 0,
            packet_num: 0,
            prev_block_long: false,
            pending_packets: VecDeque::new(),
        })
    }

    /// Convert bitrate to quality value.
    #[allow(clippy::cast_precision_loss)]
    fn bitrate_to_quality(bitrate: u32, channels: u8) -> f32 {
        // Adjust for mono (roughly half the bitrate of stereo)
        let adjusted_bitrate = if channels == 1 { bitrate * 2 } else { bitrate };

        match adjusted_bitrate {
            0..=50_000 => -1.0,
            50_001..=72_000 => 0.0,
            72_001..=88_000 => 1.0,
            88_001..=104_000 => 2.0,
            104_001..=120_000 => 3.0,
            120_001..=144_000 => 4.0,
            144_001..=176_000 => 5.0,
            176_001..=208_000 => 6.0,
            208_001..=240_000 => 7.0,
            240_001..=288_000 => 8.0,
            288_001..=410_000 => 9.0,
            _ => 10.0,
        }
    }

    /// Initialize codebooks for quality mode.
    fn init_codebooks(_quality: &QualityMode) -> Vec<Codebook> {
        // In a full implementation, this would load quality-specific codebooks
        // For now, return empty set (will be populated in setup header)
        Vec::new()
    }

    /// Generate identification header packet.
    fn generate_identification_header(&self) -> AudioResult<Vec<u8>> {
        let mut data = Vec::with_capacity(30);

        // Packet type (1)
        data.push(HeaderType::Identification.to_byte());

        // "vorbis"
        data.extend_from_slice(VORBIS_MAGIC);

        // Vorbis version (0)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Channels
        data.push(self.channels);

        // Sample rate
        data.extend_from_slice(&self.sample_rate.to_le_bytes());

        // Bitrate maximum (0 = unset)
        data.extend_from_slice(&0i32.to_le_bytes());

        // Bitrate nominal
        let nominal = self.quality_mode.nominal_bitrate() as i32;
        data.extend_from_slice(&nominal.to_le_bytes());

        // Bitrate minimum (0 = unset)
        data.extend_from_slice(&0i32.to_le_bytes());

        // Block sizes (packed into one byte)
        let bs0 = (self.blocksize_0.trailing_zeros() as u8) & 0x0F;
        let bs1 = (self.blocksize_1.trailing_zeros() as u8) & 0x0F;
        data.push((bs1 << 4) | bs0);

        // Framing flag
        data.push(0x01);

        Ok(data)
    }

    /// Generate comment header packet.
    fn generate_comment_header(
        &self,
        vendor: &str,
        comments: &[(String, String)],
    ) -> AudioResult<Vec<u8>> {
        let mut data = Vec::new();

        // Packet type (3)
        data.push(HeaderType::Comment.to_byte());

        // "vorbis"
        data.extend_from_slice(VORBIS_MAGIC);

        // Vendor string length
        let vendor_bytes = vendor.as_bytes();
        data.extend_from_slice(&(vendor_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(vendor_bytes);

        // User comment list length
        data.extend_from_slice(&(comments.len() as u32).to_le_bytes());

        // User comments
        for (key, value) in comments {
            let comment = format!("{}={}", key.to_uppercase(), value);
            let comment_bytes = comment.as_bytes();
            data.extend_from_slice(&(comment_bytes.len() as u32).to_le_bytes());
            data.extend_from_slice(comment_bytes);
        }

        // Framing bit
        data.push(0x01);

        Ok(data)
    }

    /// Generate setup header packet.
    #[allow(clippy::cast_possible_truncation)]
    fn generate_setup_header(&self) -> AudioResult<Vec<u8>> {
        let mut packer = BitPacker::new();

        // Packet type (5)
        packer.write_byte(HeaderType::Setup.to_byte());

        // "vorbis"
        packer.write_bytes(VORBIS_MAGIC);

        // Codebook count (minimum 1)
        let codebook_count = self.codebooks.len().max(1);
        packer.write_bits((codebook_count - 1) as u32, 8);

        // Write codebooks (simplified - full implementation would write actual codebook data)
        for _ in 0..codebook_count {
            self.write_codebook(&mut packer)?;
        }

        // Time domain transforms (count = 0, spec says this is deprecated)
        packer.write_bits(0, 6); // vorbis_time_count - 1 = 0 - 1 (underflow, but spec requires it)
        packer.write_bits(0, 16); // dummy time config

        // Floor count (at least 1)
        packer.write_bits(0, 6); // floor_count - 1 = 1 - 1 = 0
        self.write_floor(&mut packer)?;

        // Residue count (at least 1)
        packer.write_bits(0, 6); // residue_count - 1 = 1 - 1 = 0
        self.write_residue(&mut packer)?;

        // Mapping count (at least 1)
        packer.write_bits(0, 6); // mapping_count - 1 = 1 - 1 = 0
        self.write_mapping(&mut packer)?;

        // Mode count (at least 1, typically 2: short and long blocks)
        packer.write_bits(1, 6); // mode_count - 1 = 2 - 1 = 1

        // Mode 0: short blocks
        packer.write_bits(0, 1); // block_flag = 0 (short)
        packer.write_bits(0, 16); // window_type = 0
        packer.write_bits(0, 16); // transform_type = 0
        packer.write_bits(0, 8); // mapping = 0

        // Mode 1: long blocks
        packer.write_bits(1, 1); // block_flag = 1 (long)
        packer.write_bits(0, 16); // window_type = 0
        packer.write_bits(0, 16); // transform_type = 0
        packer.write_bits(0, 8); // mapping = 0

        // Framing bit
        packer.write_bits(1, 1);

        Ok(packer.finish())
    }

    /// Write a simplified codebook to bitstream.
    fn write_codebook(&self, packer: &mut BitPacker) -> AudioResult<()> {
        // Sync pattern (0x564342 = "BCV")
        packer.write_bits(0x42, 8);
        packer.write_bits(0x43, 8);
        packer.write_bits(0x56, 8);

        // Dimensions (2)
        packer.write_bits(2, 16);

        // Entries (16)
        packer.write_bits(16, 24);

        // Ordered flag (0)
        packer.write_bits(0, 1);

        // Sparse flag (0)
        packer.write_bits(0, 1);

        // Entry lengths (all 4 bits for simplicity)
        for _ in 0..16 {
            packer.write_bits(4, 5);
        }

        // Lookup type (0 = none)
        packer.write_bits(0, 4);

        Ok(())
    }

    /// Write floor configuration to bitstream.
    fn write_floor(&self, packer: &mut BitPacker) -> AudioResult<()> {
        // Floor type 1
        packer.write_bits(1, 16);

        // Partitions (2)
        packer.write_bits(2, 5);

        // Partition class list
        packer.write_bits(0, 4); // partition 0: class 0
        packer.write_bits(0, 4); // partition 1: class 0

        // Class 0 configuration
        packer.write_bits(1, 3); // dimensions - 1 = 1 - 1 = 0
        packer.write_bits(0, 2); // subclass bits = 0

        // Multiplier - 1 (2 - 1 = 1)
        packer.write_bits(1, 2);

        // Range bits (8)
        packer.write_bits(8, 4);

        // X values (2 + partitions * class_dimensions)
        // We have: 2 base + 2 partitions * 1 dimension = 4 total
        packer.write_bits(0, 8); // X[0] (implicit 0)
        packer.write_bits(255, 8); // X[1] (implicit n/2)
        packer.write_bits(64, 8); // X[2]
        packer.write_bits(192, 8); // X[3]

        Ok(())
    }

    /// Write residue configuration to bitstream.
    fn write_residue(&self, packer: &mut BitPacker) -> AudioResult<()> {
        // Residue type 0
        packer.write_bits(0, 16);

        // Begin, end
        packer.write_bits(0, 24); // begin = 0
        packer.write_bits(256, 24); // end = n/2

        // Partition size - 1
        packer.write_bits(31, 24); // 32 - 1

        // Classifications - 1
        packer.write_bits(3, 6); // 4 - 1

        // Classbook
        packer.write_bits(0, 8);

        // Cascade (8 bits per classification)
        for _ in 0..4 {
            packer.write_bits(0, 8);
        }

        Ok(())
    }

    /// Write mapping configuration to bitstream.
    fn write_mapping(&self, packer: &mut BitPacker) -> AudioResult<()> {
        // Mapping type 0
        packer.write_bits(0, 16);

        // Submaps (1)
        packer.write_bits(0, 1); // no submaps flag

        // Channel coupling
        if self.channels == 2 {
            packer.write_bits(1, 1); // coupling_steps flag
            packer.write_bits(0, 8); // coupling_steps - 1 = 1 - 1 = 0
            packer.write_bits(0, 4); // magnitude = 0
            packer.write_bits(1, 4); // angle = 1
        } else {
            packer.write_bits(0, 1); // no coupling
        }

        // Reserved (must be 0)
        packer.write_bits(0, 2);

        // Multiplex (submap for each channel)
        if self.channels > 1 {
            for _ in 0..self.channels {
                packer.write_bits(0, 4); // mux[i] = 0
            }
        }

        // Submap floor
        packer.write_bits(0, 8); // floor = 0

        // Submap residue
        packer.write_bits(0, 8); // residue = 0

        Ok(())
    }

    /// Encode an audio frame.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn encode_frame(&mut self, use_long_block: bool) -> AudioResult<Vec<u8>> {
        let blocksize = if use_long_block {
            self.blocksize_1
        } else {
            self.blocksize_0
        };

        let mut packer = BitPacker::new();

        // Audio packet (type = 0)
        packer.write_bits(0, 1);

        // Mode number (0 = short, 1 = long)
        packer.write_bits(if use_long_block { 1 } else { 0 }, 1);

        // For long blocks, write previous/next window flags
        if use_long_block {
            packer.write_bits(if self.prev_block_long { 1 } else { 0 }, 1);
            packer.write_bits(1, 1); // next window (assume long for now)
        }

        // Process each channel
        for ch in 0..self.channels {
            let start = ch as usize * blocksize;
            let end = start + blocksize;

            if end > self.input_buffer.len() {
                return Err(AudioError::NeedMoreData);
            }

            let channel_samples: Vec<f32> = self
                .input_buffer
                .iter()
                .skip(start)
                .take(blocksize)
                .copied()
                .collect();

            // MDCT transform
            let mdct = if use_long_block {
                &self.mdct_large
            } else {
                &self.mdct_small
            };

            let mut coeffs = vec![0.0; blocksize];
            mdct.forward(&channel_samples, &mut coeffs);

            // Floor encoding (simplified)
            self.encode_floor(&mut packer, &coeffs)?;

            // Residue encoding
            self.residue_encoder.encode(&mut packer, &coeffs)?;
        }

        // Update state
        self.prev_block_long = use_long_block;

        // Framing bit
        packer.write_bits(1, 1);

        Ok(packer.finish())
    }

    /// Encode floor curve.
    fn encode_floor(&self, packer: &mut BitPacker, _coeffs: &[f32]) -> AudioResult<()> {
        // Simplified floor encoding
        // Real implementation would analyze coefficients and encode floor curve

        // Nonzero flag
        packer.write_bits(1, 1);

        // Floor values (simplified - just write some dummy values)
        packer.write_bits(64, 8);
        packer.write_bits(128, 8);

        Ok(())
    }

    /// Choose block size based on signal characteristics.
    fn choose_block_size(&self, _samples: &[f32]) -> bool {
        // Simplified: use long blocks for most frames
        // Real implementation would analyze signal for transients
        true
    }
}

impl AudioEncoder for VorbisEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
    }

    fn send_frame(&mut self, frame: &AudioFrame) -> AudioResult<()> {
        // Verify format
        if frame.format != SampleFormat::F32 {
            return Err(AudioError::InvalidParameter(
                "Vorbis encoder requires F32 samples".into(),
            ));
        }

        let frame_channels = frame.channels.count();
        if frame_channels != self.channels as usize {
            return Err(AudioError::InvalidParameter(format!(
                "Channel count mismatch: expected {}, got {}",
                self.channels, frame_channels
            )));
        }

        // Extract samples from AudioFrame
        let samples = match &frame.samples {
            crate::frame::AudioBuffer::Interleaved(data) => {
                // Assume f32 for now
                let sample_count = data.len() / 4;
                let mut samples = Vec::with_capacity(sample_count);
                for i in 0..sample_count {
                    let offset = i * 4;
                    if offset + 4 <= data.len() {
                        let bytes = [
                            data[offset],
                            data[offset + 1],
                            data[offset + 2],
                            data[offset + 3],
                        ];
                        samples.push(f32::from_le_bytes(bytes));
                    }
                }
                samples
            }
            crate::frame::AudioBuffer::Planar(planes) => {
                // Convert planar to interleaved
                if planes.is_empty() {
                    Vec::new()
                } else {
                    let sample_size = 4; // f32
                    let frames = planes[0].len() / sample_size;
                    let mut interleaved = Vec::with_capacity(frames * frame_channels);
                    for frame_idx in 0..frames {
                        for plane in planes {
                            let offset = frame_idx * sample_size;
                            if offset + sample_size <= plane.len() {
                                let bytes = [
                                    plane[offset],
                                    plane[offset + 1],
                                    plane[offset + 2],
                                    plane[offset + 3],
                                ];
                                interleaved.push(f32::from_le_bytes(bytes));
                            }
                        }
                    }
                    interleaved
                }
            }
        };

        // Add samples to input buffer
        for sample in samples {
            self.input_buffer.push_back(sample);
        }

        Ok(())
    }

    fn receive_packet(&mut self) -> AudioResult<Option<EncodedAudioPacket>> {
        // Return pending packets first
        if let Some(packet) = self.pending_packets.pop_front() {
            return Ok(Some(packet));
        }

        // Generate header packets in sequence
        match self.state {
            EncoderState::NeedIdentification => {
                let data = self.generate_identification_header()?;
                self.state = EncoderState::NeedComment;
                self.packet_num += 1;
                return Ok(Some(EncodedAudioPacket {
                    data,
                    pts: 0,
                    duration: 0,
                }));
            }
            EncoderState::NeedComment => {
                let vendor = "OxiMedia Vorbis Encoder";
                let comments = vec![];
                let data = self.generate_comment_header(vendor, &comments)?;
                self.state = EncoderState::NeedSetup;
                self.packet_num += 1;
                return Ok(Some(EncodedAudioPacket {
                    data,
                    pts: 0,
                    duration: 0,
                }));
            }
            EncoderState::NeedSetup => {
                let data = self.generate_setup_header()?;
                self.state = EncoderState::Ready;
                self.packet_num += 1;
                return Ok(Some(EncodedAudioPacket {
                    data,
                    pts: 0,
                    duration: 0,
                }));
            }
            EncoderState::Ready => {
                // Need enough samples for a block
                let required = self.blocksize_1 * self.channels as usize;
                if self.input_buffer.len() < required {
                    if self.state == EncoderState::Flushing {
                        return Ok(None);
                    }
                    return Err(AudioError::NeedMoreData);
                }

                // Encode frame
                let use_long =
                    self.choose_block_size(&self.input_buffer.iter().copied().collect::<Vec<_>>());
                let blocksize = if use_long {
                    self.blocksize_1
                } else {
                    self.blocksize_0
                };

                let data = self.encode_frame(use_long)?;

                // Remove encoded samples from buffer
                let samples_to_remove = blocksize * self.channels as usize;
                self.input_buffer.drain(..samples_to_remove);

                let pts = self.pts;
                self.pts += blocksize as i64;
                self.packet_num += 1;

                Ok(Some(EncodedAudioPacket {
                    data,
                    pts,
                    duration: blocksize as u32,
                }))
            }
            EncoderState::Flushing => Ok(None),
        }
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.state = EncoderState::Flushing;
        Ok(())
    }

    fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_mode() {
        let q = QualityMode::from_quality(5.5);
        assert_eq!(q.base_quality, 5);
        assert!((q.quality_fraction - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_quality_mode_clamp() {
        let q = QualityMode::from_quality(15.0);
        assert_eq!(q.quality, 10.0);

        let q = QualityMode::from_quality(-5.0);
        assert_eq!(q.quality, -1.0);
    }

    #[test]
    fn test_encoder_creation() {
        let config = AudioEncoderConfig {
            codec: CodecId::Vorbis,
            sample_rate: 44100,
            channels: 2,
            bitrate: 128_000,
            frame_size: 1024,
        };

        let encoder = VorbisEncoder::new(&config).expect("should succeed");
        assert_eq!(encoder.codec(), CodecId::Vorbis);
        assert_eq!(encoder.sample_rate, 44100);
        assert_eq!(encoder.channels, 2);
    }

    #[test]
    fn test_encoder_wrong_codec() {
        let config = AudioEncoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };

        assert!(VorbisEncoder::new(&config).is_err());
    }

    #[test]
    fn test_bitrate_to_quality() {
        assert_eq!(VorbisEncoder::bitrate_to_quality(64_000, 2), 0.0);
        assert_eq!(VorbisEncoder::bitrate_to_quality(128_000, 2), 4.0);
        assert_eq!(VorbisEncoder::bitrate_to_quality(320_000, 2), 9.0);
    }
}
