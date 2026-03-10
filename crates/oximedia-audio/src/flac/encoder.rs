//! FLAC encoder implementation.
//!
//! Encodes PCM audio to FLAC format with lossless compression.

#![forbid(unsafe_code)]

use super::{
    bitwriter::BitWriter,
    crc::{crc8, Crc16},
    frame::{BlockingStrategy, ChannelAssignment, FrameHeader, SampleSize, SYNC_CODE},
    subframe::fixed_coefficients,
    StreamInfo,
};
use crate::{
    AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame, AudioResult, EncodedAudioPacket,
};
use oximedia_core::CodecId;

/// Compression level (0-8).
/// - 0: Fastest (mostly verbatim)
/// - 5: Default balance
/// - 8: Best compression (slower, uses LPC)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompressionLevel(u8);

impl CompressionLevel {
    /// Fastest compression.
    pub const FASTEST: Self = Self(0);
    /// Default compression.
    pub const DEFAULT: Self = Self(5);
    /// Best compression.
    pub const BEST: Self = Self(8);

    /// Create compression level.
    ///
    /// # Errors
    ///
    /// Returns error if level > 8.
    pub fn new(level: u8) -> Result<Self, AudioError> {
        if level > 8 {
            return Err(AudioError::InvalidParameter(
                "Compression level must be 0-8".into(),
            ));
        }
        Ok(Self(level))
    }

    /// Get level value.
    #[must_use]
    pub const fn value(self) -> u8 {
        self.0
    }

    /// Get maximum LPC order for this level.
    #[must_use]
    pub const fn max_lpc_order(self) -> u8 {
        match self.0 {
            0 => 0, // No LPC
            1 => 0, // Fixed only
            2 => 0, // Fixed only
            3 => 6, // Low order LPC
            4 => 8,
            5 => 12,
            6 => 12,
            7 => 12,
            _ => 12, // Level 8: max LPC
        }
    }

    /// Get maximum fixed predictor order.
    #[must_use]
    pub const fn max_fixed_order(self) -> u8 {
        match self.0 {
            0 => 0, // Verbatim only
            _ => 4, // All levels use up to order 4 fixed
        }
    }

    /// Get Rice partition order.
    #[must_use]
    pub const fn partition_order(self) -> u8 {
        match self.0 {
            0..=2 => 0,
            3..=5 => 2,
            6 => 3,
            _ => 4,
        }
    }
}

impl Default for CompressionLevel {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// FLAC encoder.
pub struct FlacEncoder {
    config: AudioEncoderConfig,
    compression_level: CompressionLevel,
    sample_count: u64,
    frame_number: u32,
    buffered_samples: Vec<Vec<i32>>,
    bits_per_sample: u8,
    pending_packet: Option<EncodedAudioPacket>,
}

impl FlacEncoder {
    /// Create new FLAC encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioEncoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Flac {
            return Err(AudioError::InvalidParameter("Expected FLAC codec".into()));
        }

        if config.channels == 0 || config.channels > 8 {
            return Err(AudioError::InvalidParameter(
                "FLAC supports 1-8 channels".into(),
            ));
        }

        if config.frame_size < 16 || config.frame_size > 65535 {
            return Err(AudioError::InvalidParameter(
                "FLAC block size must be 16-65535".into(),
            ));
        }

        // Default to 16-bit samples
        let bits_per_sample = 16;

        Ok(Self {
            config: config.clone(),
            compression_level: CompressionLevel::DEFAULT,
            sample_count: 0,
            frame_number: 0,
            buffered_samples: vec![Vec::new(); config.channels as usize],
            bits_per_sample,
            pending_packet: None,
        })
    }

    /// Create encoder with compression level.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn with_compression_level(
        config: &AudioEncoderConfig,
        level: CompressionLevel,
    ) -> AudioResult<Self> {
        let mut encoder = Self::new(config)?;
        encoder.compression_level = level;
        Ok(encoder)
    }

    /// Set compression level.
    pub fn set_compression_level(&mut self, level: CompressionLevel) {
        self.compression_level = level;
    }

    /// Generate STREAMINFO metadata block.
    ///
    /// # Errors
    ///
    /// Returns error if parameters are invalid.
    pub fn generate_streaminfo(&self, total_samples: u64) -> AudioResult<Vec<u8>> {
        let stream_info = StreamInfo {
            min_block_size: self.config.frame_size as u16,
            max_block_size: self.config.frame_size as u16,
            min_frame_size: 0, // Unknown
            max_frame_size: 0, // Unknown
            sample_rate: self.config.sample_rate,
            channels: self.config.channels,
            bits_per_sample: self.bits_per_sample,
            total_samples,
            md5_signature: [0u8; 16], // Would need to compute from all samples
        };

        Self::encode_streaminfo(&stream_info)
    }

    /// Encode STREAMINFO metadata to bytes.
    fn encode_streaminfo(info: &StreamInfo) -> AudioResult<Vec<u8>> {
        let mut data = Vec::with_capacity(34);

        // Min/max block size (16 bits each)
        data.extend_from_slice(&info.min_block_size.to_be_bytes());
        data.extend_from_slice(&info.max_block_size.to_be_bytes());

        // Min/max frame size (24 bits each)
        data.push((info.min_frame_size >> 16) as u8);
        data.push((info.min_frame_size >> 8) as u8);
        data.push(info.min_frame_size as u8);
        data.push((info.max_frame_size >> 16) as u8);
        data.push((info.max_frame_size >> 8) as u8);
        data.push(info.max_frame_size as u8);

        // Sample rate (20 bits), channels (3 bits), bps (5 bits), total samples (36 bits)
        data.push((info.sample_rate >> 12) as u8);
        data.push((info.sample_rate >> 4) as u8);

        let byte12 = ((info.sample_rate & 0x0F) << 4) as u8
            | (((info.channels - 1) & 0x07) << 1)
            | (((info.bits_per_sample - 1) >> 4) & 0x01);
        data.push(byte12);

        let byte13 = (((info.bits_per_sample - 1) & 0x0F) << 4) as u8
            | ((info.total_samples >> 32) & 0x0F) as u8;
        data.push(byte13);

        // Remaining 32 bits of total samples
        data.push((info.total_samples >> 24) as u8);
        data.push((info.total_samples >> 16) as u8);
        data.push((info.total_samples >> 8) as u8);
        data.push(info.total_samples as u8);

        // MD5 signature
        data.extend_from_slice(&info.md5_signature);

        Ok(data)
    }

    /// Encode a frame.
    #[allow(clippy::too_many_lines)]
    fn encode_frame(&mut self, samples: &[Vec<i32>]) -> AudioResult<Vec<u8>> {
        let block_size = samples[0].len();
        if block_size == 0 {
            return Err(AudioError::InvalidData("Empty block".into()));
        }

        // Determine channel assignment
        let channel_assignment = if samples.len() == 2 {
            // Try stereo decorrelation
            self.choose_channel_assignment(samples)
        } else {
            ChannelAssignment::Independent(samples.len() as u8)
        };

        // Apply stereo decorrelation if needed
        let encoded_channels = self.apply_stereo_decorrelation(samples, channel_assignment);

        // Create frame header
        let header = FrameHeader {
            blocking_strategy: BlockingStrategy::Fixed,
            block_size: block_size as u32,
            sample_rate: self.config.sample_rate,
            channel_assignment,
            sample_size: SampleSize::FromStreamInfo,
            bits_per_sample: self.bits_per_sample,
            frame_number: Some(self.frame_number),
            sample_number: None,
            crc8: 0, // Will be calculated
        };

        let mut writer = BitWriter::with_capacity(block_size * samples.len() * 2);

        // Write frame header
        self.write_frame_header(&mut writer, &header)?;

        // Calculate CRC-8 for header
        let header_bytes = writer.as_bytes().to_vec();
        let crc8_val = crc8(&header_bytes);
        writer.write_bits(u32::from(crc8_val), 8);

        // Start CRC-16 calculation
        let mut crc16 = Crc16::new();
        crc16.update(writer.as_bytes());

        // Encode subframes
        for (ch_idx, channel_samples) in encoded_channels.iter().enumerate() {
            let bps = if let Some(side_ch) = channel_assignment.side_channel() {
                if ch_idx == side_ch {
                    self.bits_per_sample + 1 // Side channel needs 1 extra bit
                } else {
                    self.bits_per_sample
                }
            } else {
                self.bits_per_sample
            };

            self.encode_subframe(&mut writer, channel_samples, bps)?;
        }

        // Byte align
        writer.byte_align();

        // Update CRC-16 with frame data (everything after header)
        let all_bytes = writer.as_bytes();
        let frame_bytes = &all_bytes[header_bytes.len()..];
        crc16.update(frame_bytes);

        // Write CRC-16
        writer.write_bits(u32::from(crc16.value()), 16);

        self.frame_number += 1;

        Ok(writer.finish())
    }

    /// Choose best channel assignment for stereo.
    fn choose_channel_assignment(&self, samples: &[Vec<i32>]) -> ChannelAssignment {
        if samples.len() != 2 {
            return ChannelAssignment::Independent(samples.len() as u8);
        }

        // For simplicity at low compression, use independent
        if self.compression_level.value() < 3 {
            return ChannelAssignment::Independent(2);
        }

        // Try different stereo modes and pick best
        let independent_score =
            self.estimate_compression(samples, ChannelAssignment::Independent(2));
        let left_side_score = self.estimate_compression(samples, ChannelAssignment::LeftSide);
        let right_side_score = self.estimate_compression(samples, ChannelAssignment::RightSide);
        let mid_side_score = self.estimate_compression(samples, ChannelAssignment::MidSide);

        // Pick mode with lowest score (best compression)
        let min_score = independent_score
            .min(left_side_score)
            .min(right_side_score)
            .min(mid_side_score);

        if min_score == left_side_score {
            ChannelAssignment::LeftSide
        } else if min_score == right_side_score {
            ChannelAssignment::RightSide
        } else if min_score == mid_side_score {
            ChannelAssignment::MidSide
        } else {
            ChannelAssignment::Independent(2)
        }
    }

    /// Estimate compression size for channel assignment.
    fn estimate_compression(&self, samples: &[Vec<i32>], assignment: ChannelAssignment) -> u64 {
        let encoded = self.apply_stereo_decorrelation(samples, assignment);
        let mut score = 0u64;

        for channel in &encoded {
            // Simple estimate: sum of absolute values
            for &sample in channel {
                score += sample.unsigned_abs() as u64;
            }
        }

        score
    }

    /// Apply stereo decorrelation based on channel assignment.
    fn apply_stereo_decorrelation(
        &self,
        samples: &[Vec<i32>],
        assignment: ChannelAssignment,
    ) -> Vec<Vec<i32>> {
        if samples.len() != 2 {
            return samples.to_vec();
        }

        let block_size = samples[0].len();
        let mut result = (0..2)
            .map(|_| Vec::with_capacity(block_size))
            .collect::<Vec<_>>();

        match assignment {
            ChannelAssignment::Independent(_) => {
                result[0].clone_from(&samples[0]);
                result[1].clone_from(&samples[1]);
            }
            ChannelAssignment::LeftSide => {
                // Left stays, side = left - right
                result[0].clone_from(&samples[0]);
                for i in 0..block_size {
                    result[1].push(samples[0][i] - samples[1][i]);
                }
            }
            ChannelAssignment::RightSide => {
                // Side = left - right, right stays
                for i in 0..block_size {
                    result[0].push(samples[0][i] - samples[1][i]);
                }
                result[1].clone_from(&samples[1]);
            }
            ChannelAssignment::MidSide => {
                // Mid = (left + right) / 2, side = left - right
                for i in 0..block_size {
                    let left = samples[0][i];
                    let right = samples[1][i];
                    result[0].push((left + right) >> 1);
                    result[1].push(left - right);
                }
            }
        }

        result
    }

    /// Write frame header.
    fn write_frame_header(&self, writer: &mut BitWriter, header: &FrameHeader) -> AudioResult<()> {
        // Sync code (14 bits)
        writer.write_bits(u32::from(SYNC_CODE), 14);

        // Reserved bit (must be 0)
        writer.write_bit(false);

        // Blocking strategy
        writer.write_bit(header.blocking_strategy == BlockingStrategy::Variable);

        // Block size
        let block_size_code = self.get_block_size_code(header.block_size);
        writer.write_bits(u32::from(block_size_code), 4);

        // Sample rate
        let sample_rate_code = self.get_sample_rate_code(header.sample_rate);
        writer.write_bits(u32::from(sample_rate_code), 4);

        // Channel assignment
        let channel_code = match header.channel_assignment {
            ChannelAssignment::Independent(n) => n - 1,
            ChannelAssignment::LeftSide => 8,
            ChannelAssignment::RightSide => 9,
            ChannelAssignment::MidSide => 10,
        };
        writer.write_bits(u32::from(channel_code), 4);

        // Sample size
        let sample_size_code: u8 = match self.bits_per_sample {
            8 => 1,
            16 => 4,
            24 => 6,
            _ => 0, // From STREAMINFO
        };
        writer.write_bits(u32::from(sample_size_code), 3);

        // Reserved bit (must be 0)
        writer.write_bit(false);

        // Frame/sample number
        if header.blocking_strategy == BlockingStrategy::Fixed {
            writer.write_utf8_u32(header.frame_number.unwrap_or(0))?;
        } else {
            writer.write_utf8_u64(header.sample_number.unwrap_or(0))?;
        }

        // Optional block size
        if block_size_code == 6 {
            writer.write_bits(header.block_size - 1, 8);
        } else if block_size_code == 7 {
            writer.write_bits(header.block_size - 1, 16);
        }

        // Optional sample rate
        if sample_rate_code == 12 {
            writer.write_bits(header.sample_rate / 1000, 8);
        } else if sample_rate_code == 13 {
            writer.write_bits(header.sample_rate, 16);
        } else if sample_rate_code == 14 {
            writer.write_bits(header.sample_rate / 10, 16);
        }

        Ok(())
    }

    /// Get block size code for header.
    fn get_block_size_code(&self, block_size: u32) -> u8 {
        match block_size {
            192 => 1,
            576 => 2,
            1152 => 3,
            2304 => 4,
            4608 => 5,
            256 => 8,
            512 => 9,
            1024 => 10,
            2048 => 11,
            4096 => 12,
            8192 => 13,
            16384 => 14,
            32768 => 15,
            1..=256 => 6, // 8-bit at end
            _ => 7,       // 16-bit at end
        }
    }

    /// Get sample rate code for header.
    fn get_sample_rate_code(&self, sample_rate: u32) -> u8 {
        match sample_rate {
            88_200 => 1,
            176_400 => 2,
            192_000 => 3,
            8000 => 4,
            16_000 => 5,
            22_050 => 6,
            24_000 => 7,
            32_000 => 8,
            44_100 => 9,
            48_000 => 10,
            96_000 => 11,
            _ => {
                if sample_rate % 1000 == 0 && sample_rate / 1000 < 256 {
                    12 // kHz
                } else if sample_rate % 10 == 0 {
                    14 // 10Hz
                } else {
                    13 // Hz
                }
            }
        }
    }

    /// Convert samples from byte buffer to i32 arrays per channel.
    fn convert_samples_to_i32(
        &self,
        buffer: &[u8],
        format: oximedia_core::SampleFormat,
        sample_count: usize,
        channel_count: usize,
    ) -> Vec<Vec<i32>> {
        use oximedia_core::SampleFormat;

        let mut channels = vec![Vec::with_capacity(sample_count); channel_count];

        match format {
            SampleFormat::S16 => {
                // Interleaved i16 samples
                for i in 0..sample_count {
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 2;
                        if idx + 1 < buffer.len() {
                            let sample = i16::from_le_bytes([buffer[idx], buffer[idx + 1]]);
                            channels[ch].push(i32::from(sample));
                        }
                    }
                }
            }
            SampleFormat::S32 => {
                // Interleaved i32 samples
                for i in 0..sample_count {
                    for ch in 0..channel_count {
                        let idx = (i * channel_count + ch) * 4;
                        if idx + 3 < buffer.len() {
                            let sample = i32::from_le_bytes([
                                buffer[idx],
                                buffer[idx + 1],
                                buffer[idx + 2],
                                buffer[idx + 3],
                            ]);
                            channels[ch].push(sample >> 16); // Convert to 16-bit range
                        }
                    }
                }
            }
            _ => {
                // For other formats, just return zeros
                for ch in &mut channels {
                    ch.resize(sample_count, 0);
                }
            }
        }

        channels
    }

    /// Encode a subframe.
    #[allow(clippy::cast_possible_wrap)]
    fn encode_subframe(&self, writer: &mut BitWriter, samples: &[i32], bps: u8) -> AudioResult<()> {
        // Check if constant
        if self.is_constant(samples) {
            return self.encode_constant_subframe(writer, samples, bps);
        }

        // Try different prediction methods and pick best
        let mut best_size = usize::MAX;
        let mut best_data = Vec::new();

        // Try verbatim
        let mut verbatim_writer = BitWriter::new();
        self.encode_verbatim_subframe(&mut verbatim_writer, samples, bps)?;
        let verbatim_data = verbatim_writer.finish();
        if verbatim_data.len() < best_size {
            best_size = verbatim_data.len();
            best_data = verbatim_data;
        }

        // Try fixed predictors
        let max_order = self.compression_level.max_fixed_order();
        for order in 1..=max_order.min(4) {
            let mut fixed_writer = BitWriter::new();
            if self
                .encode_fixed_subframe(&mut fixed_writer, samples, order, bps)
                .is_ok()
            {
                let fixed_data = fixed_writer.finish();
                if fixed_data.len() < best_size {
                    best_size = fixed_data.len();
                    best_data = fixed_data;
                }
            }
        }

        // Try LPC if compression level is high enough
        if self.compression_level.max_lpc_order() > 0 {
            for order in 1..=self.compression_level.max_lpc_order().min(12) {
                let mut lpc_writer = BitWriter::new();
                if self
                    .encode_lpc_subframe(&mut lpc_writer, samples, order, bps)
                    .is_ok()
                {
                    let lpc_data = lpc_writer.finish();
                    if lpc_data.len() < best_size {
                        best_size = lpc_data.len();
                        best_data = lpc_data;
                    }
                }
            }
        }

        // Write best encoding
        for &byte in &best_data {
            writer.write_bits(u32::from(byte), 8);
        }

        Ok(())
    }

    /// Check if all samples are constant.
    fn is_constant(&self, samples: &[i32]) -> bool {
        if samples.is_empty() {
            return true;
        }
        let first = samples[0];
        samples.iter().all(|&s| s == first)
    }

    /// Encode constant subframe.
    fn encode_constant_subframe(
        &self,
        writer: &mut BitWriter,
        samples: &[i32],
        bps: u8,
    ) -> AudioResult<()> {
        // Subframe header: 0 + type (000000) + wasted bits (0)
        writer.write_bits(0, 8);

        // Constant value
        writer.write_signed(samples[0], bps);

        Ok(())
    }

    /// Encode verbatim subframe.
    fn encode_verbatim_subframe(
        &self,
        writer: &mut BitWriter,
        samples: &[i32],
        bps: u8,
    ) -> AudioResult<()> {
        // Subframe header: 0 + type (000001) + wasted bits (0)
        writer.write_bits(0b0000_0010, 8);

        // Uncompressed samples
        for &sample in samples {
            writer.write_signed(sample, bps);
        }

        Ok(())
    }

    /// Encode fixed predictor subframe.
    #[allow(clippy::cast_possible_truncation)]
    fn encode_fixed_subframe(
        &self,
        writer: &mut BitWriter,
        samples: &[i32],
        order: u8,
        bps: u8,
    ) -> AudioResult<()> {
        if order > 4 {
            return Err(AudioError::InvalidParameter(
                "Fixed order must be 0-4".into(),
            ));
        }

        // Subframe header: 0 + type (001000 + order) + wasted bits (0)
        let type_bits = 0b0001_0000 | order;
        writer.write_bits(u32::from(type_bits), 8);

        // Warmup samples
        for i in 0..order as usize {
            writer.write_signed(samples[i], bps);
        }

        // Calculate residuals
        let residuals = self.calculate_fixed_residuals(samples, order);

        // Encode residuals
        self.encode_residuals(writer, &residuals)?;

        Ok(())
    }

    /// Calculate residuals for fixed predictor.
    fn calculate_fixed_residuals(&self, samples: &[i32], order: u8) -> Vec<i32> {
        let coeffs = fixed_coefficients::for_order(order);
        let mut residuals = Vec::with_capacity(samples.len() - order as usize);

        for i in order as usize..samples.len() {
            let mut prediction: i64 = 0;
            for (j, &coeff) in coeffs.iter().enumerate() {
                prediction += i64::from(coeff) * i64::from(samples[i - 1 - j]);
            }
            #[allow(clippy::cast_possible_truncation)]
            let residual = samples[i] - prediction as i32;
            residuals.push(residual);
        }

        residuals
    }

    /// Encode LPC subframe.
    #[allow(clippy::cast_possible_truncation)]
    fn encode_lpc_subframe(
        &self,
        writer: &mut BitWriter,
        samples: &[i32],
        order: u8,
        bps: u8,
    ) -> AudioResult<()> {
        if order == 0 || order > 32 {
            return Err(AudioError::InvalidParameter(
                "LPC order must be 1-32".into(),
            ));
        }

        // Calculate LPC coefficients using autocorrelation
        let (coeffs, shift) = self.calculate_lpc_coefficients(samples, order)?;

        // Subframe header: 0 + type (100000 + order-1) + wasted bits (0)
        let type_bits = 0b0100_0000 | (order - 1);
        writer.write_bits(u32::from(type_bits), 8);

        // Warmup samples
        for i in 0..order as usize {
            writer.write_signed(samples[i], bps);
        }

        // Quantized LP coefficient precision - 1 (4 bits)
        let precision = 12; // Use 12-bit precision
        writer.write_bits(u32::from(precision - 1), 4);

        // Quantized LP coefficient shift (5 bits, signed)
        writer.write_signed(i32::from(shift), 5);

        // Quantized LP coefficients
        for &coeff in &coeffs {
            writer.write_signed(coeff, precision);
        }

        // Calculate residuals
        let residuals = self.calculate_lpc_residuals(samples, &coeffs, shift, order);

        // Encode residuals
        self.encode_residuals(writer, &residuals)?;

        Ok(())
    }

    /// Calculate LPC coefficients using Levinson-Durbin algorithm.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn calculate_lpc_coefficients(
        &self,
        samples: &[i32],
        order: u8,
    ) -> AudioResult<(Vec<i32>, i8)> {
        let n = samples.len();
        let order = order as usize;

        if n < order + 1 {
            return Err(AudioError::InvalidData("Not enough samples for LPC".into()));
        }

        // Calculate autocorrelation
        let mut autocorr = vec![0.0f64; order + 1];
        for i in 0..=order {
            let mut sum = 0.0;
            for j in 0..n - i {
                sum += f64::from(samples[j]) * f64::from(samples[j + i]);
            }
            autocorr[i] = sum;
        }

        if autocorr[0].abs() < 1e-10 {
            // Nearly silent, use zero coefficients
            return Ok((vec![0; order], 0));
        }

        // Levinson-Durbin recursion
        let mut lpc = vec![0.0f64; order];
        let mut error = autocorr[0];

        for i in 0..order {
            let mut lambda = 0.0;
            for j in 0..i {
                lambda += lpc[j] * autocorr[i - j];
            }
            lambda = (autocorr[i + 1] - lambda) / error;

            lpc[i] = lambda;
            for j in 0..i / 2 + 1 {
                let tmp = lpc[j];
                lpc[j] += lambda * lpc[i - 1 - j];
                if j != i - 1 - j {
                    lpc[i - 1 - j] += lambda * tmp;
                }
            }

            error *= 1.0 - lambda * lambda;
        }

        // Quantize coefficients
        let shift = 10i8; // Use shift of 10
        let scale = f64::from(1i32 << shift);
        let quantized: Vec<i32> = lpc.iter().map(|&c| (c * scale).round() as i32).collect();

        Ok((quantized, shift))
    }

    /// Calculate residuals for LPC predictor.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn calculate_lpc_residuals(
        &self,
        samples: &[i32],
        coeffs: &[i32],
        shift: i8,
        order: u8,
    ) -> Vec<i32> {
        let mut residuals = Vec::with_capacity(samples.len() - order as usize);

        for i in order as usize..samples.len() {
            let mut prediction: i64 = 0;
            for (j, &coeff) in coeffs.iter().enumerate() {
                prediction += i64::from(coeff) * i64::from(samples[i - 1 - j]);
            }
            prediction >>= shift as u32;
            let residual = samples[i] - prediction as i32;
            residuals.push(residual);
        }

        residuals
    }

    /// Encode residuals using Rice coding.
    fn encode_residuals(&self, writer: &mut BitWriter, residuals: &[i32]) -> AudioResult<()> {
        // Coding method (2 bits): 00 = Rice with 4-bit param
        writer.write_bits(0, 2);

        // Partition order (4 bits)
        let partition_order = self.compression_level.partition_order();
        writer.write_bits(u32::from(partition_order), 4);

        let partition_count = 1usize << partition_order;
        let samples_per_partition = residuals.len() / partition_count;

        for p in 0..partition_count {
            let start = p * samples_per_partition;
            let end = if p == partition_count - 1 {
                residuals.len()
            } else {
                (p + 1) * samples_per_partition
            };
            let partition = &residuals[start..end];

            // Calculate optimal Rice parameter
            let param = self.calculate_rice_parameter(partition);

            // Write parameter (4 bits)
            writer.write_bits(u32::from(param), 4);

            // Encode partition samples
            for &residual in partition {
                writer.write_rice(residual, param);
            }
        }

        Ok(())
    }

    /// Calculate optimal Rice parameter for partition.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn calculate_rice_parameter(&self, residuals: &[i32]) -> u8 {
        if residuals.is_empty() {
            return 0;
        }

        // Calculate mean absolute value
        let sum: u64 = residuals.iter().map(|&r| r.unsigned_abs() as u64).sum();
        let mean = sum / residuals.len() as u64;

        if mean == 0 {
            return 0;
        }

        // Optimal Rice parameter is approximately log2(mean)
        let param = (63 - mean.leading_zeros()) as u8;
        param.min(14) // Maximum parameter for Rice coding
    }
}

impl AudioEncoder for FlacEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Flac
    }

    fn send_frame(&mut self, frame: &AudioFrame) -> AudioResult<()> {
        if self.pending_packet.is_some() {
            return Err(AudioError::Internal(
                "Packet pending, call receive_packet first".into(),
            ));
        }

        // Convert frame samples to i32
        let channel_count = frame.channels.count();
        if channel_count != self.config.channels as usize {
            return Err(AudioError::InvalidData("Channel count mismatch".into()));
        }

        let sample_count = frame.sample_count();

        // Extract samples from frame based on buffer type
        let samples_i32 = match &frame.samples {
            crate::frame::AudioBuffer::Interleaved(data) => {
                self.convert_samples_to_i32(data, frame.format, sample_count, channel_count)
            }
            crate::frame::AudioBuffer::Planar(planes) => {
                // For planar, each plane is a channel
                let mut result = Vec::with_capacity(channel_count);
                for plane in planes {
                    let channel_samples =
                        self.convert_samples_to_i32(plane, frame.format, sample_count, 1);
                    if !channel_samples.is_empty() {
                        result.push(channel_samples[0].clone());
                    }
                }
                result
            }
        };

        // Add to buffers
        for (ch, samples) in samples_i32.iter().enumerate() {
            if ch < self.buffered_samples.len() {
                self.buffered_samples[ch].extend_from_slice(samples);
            }
        }

        // Check if we have enough samples for a block
        while self.buffered_samples[0].len() >= self.config.frame_size as usize {
            let mut block_samples = Vec::with_capacity(channel_count);
            for ch_buffer in &mut self.buffered_samples {
                let block: Vec<i32> = ch_buffer.drain(..self.config.frame_size as usize).collect();
                block_samples.push(block);
            }

            // Encode the block
            let encoded = self.encode_frame(&block_samples)?;
            self.sample_count += self.config.frame_size as u64;

            // Store as pending packet
            self.pending_packet = Some(EncodedAudioPacket {
                data: encoded,
                pts: ((self.sample_count - self.config.frame_size as u64) * 1000
                    / self.config.sample_rate as u64) as i64,
                duration: self.config.frame_size,
            });

            // Can only return one packet at a time
            break;
        }

        Ok(())
    }

    fn receive_packet(&mut self) -> AudioResult<Option<EncodedAudioPacket>> {
        Ok(self.pending_packet.take())
    }

    fn flush(&mut self) -> AudioResult<()> {
        // Encode remaining samples if any
        if !self.buffered_samples[0].is_empty() {
            let block_size = self.buffered_samples[0].len();
            let mut block_samples = Vec::with_capacity(self.config.channels as usize);

            for ch_buffer in &mut self.buffered_samples {
                let block: Vec<i32> = std::mem::take(ch_buffer);
                block_samples.push(block);
            }

            let encoded = self.encode_frame(&block_samples)?;
            self.sample_count += block_size as u64;

            self.pending_packet = Some(EncodedAudioPacket {
                data: encoded,
                pts: ((self.sample_count - block_size as u64) * 1000
                    / self.config.sample_rate as u64) as i64,
                duration: block_size as u32,
            });
        }

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
    fn test_compression_level() {
        assert_eq!(CompressionLevel::FASTEST.value(), 0);
        assert_eq!(CompressionLevel::DEFAULT.value(), 5);
        assert_eq!(CompressionLevel::BEST.value(), 8);

        assert!(CompressionLevel::new(9).is_err());
        assert!(CompressionLevel::new(5).is_ok());
    }

    #[test]
    fn test_compression_level_params() {
        assert_eq!(CompressionLevel::FASTEST.max_lpc_order(), 0);
        assert_eq!(CompressionLevel::DEFAULT.max_lpc_order(), 12);
        assert_eq!(CompressionLevel::FASTEST.max_fixed_order(), 0);
        assert_eq!(CompressionLevel::DEFAULT.max_fixed_order(), 4);
    }

    #[test]
    fn test_flac_encoder_creation() {
        let config = AudioEncoderConfig {
            codec: CodecId::Flac,
            sample_rate: 44100,
            channels: 2,
            bitrate: 0,
            frame_size: 4096,
        };

        let encoder = FlacEncoder::new(&config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_flac_encoder_wrong_codec() {
        let config = AudioEncoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };

        assert!(FlacEncoder::new(&config).is_err());
    }

    #[test]
    fn test_flac_encoder_invalid_channels() {
        let config = AudioEncoderConfig {
            codec: CodecId::Flac,
            channels: 0,
            ..Default::default()
        };

        assert!(FlacEncoder::new(&config).is_err());
    }

    #[test]
    fn test_streaminfo_generation() {
        let config = AudioEncoderConfig {
            codec: CodecId::Flac,
            sample_rate: 44100,
            channels: 2,
            bitrate: 0,
            frame_size: 4096,
        };

        let encoder = FlacEncoder::new(&config).expect("should succeed");
        let streaminfo = encoder.generate_streaminfo(441000).expect("should succeed");
        assert_eq!(streaminfo.len(), 34);

        // Verify we can parse it back
        let parsed = StreamInfo::parse(&streaminfo).expect("should succeed");
        assert_eq!(parsed.sample_rate, 44100);
        assert_eq!(parsed.channels, 2);
        assert_eq!(parsed.total_samples, 441000);
    }

    #[test]
    fn test_is_constant() {
        let config = AudioEncoderConfig {
            codec: CodecId::Flac,
            sample_rate: 44100,
            channels: 1,
            bitrate: 0,
            frame_size: 4096,
        };

        let encoder = FlacEncoder::new(&config).expect("should succeed");

        assert!(encoder.is_constant(&[100, 100, 100, 100]));
        assert!(!encoder.is_constant(&[100, 101, 100, 100]));
        assert!(encoder.is_constant(&[]));
    }

    #[test]
    fn test_calculate_fixed_residuals() {
        let config = AudioEncoderConfig {
            codec: CodecId::Flac,
            sample_rate: 44100,
            channels: 1,
            bitrate: 0,
            frame_size: 4096,
        };

        let encoder = FlacEncoder::new(&config).expect("should succeed");
        let samples = vec![0, 1, 2, 3, 4, 5];

        // Order 1: first difference
        let residuals = encoder.calculate_fixed_residuals(&samples, 1);
        assert_eq!(residuals, vec![1, 1, 1, 1, 1]);
    }
}
