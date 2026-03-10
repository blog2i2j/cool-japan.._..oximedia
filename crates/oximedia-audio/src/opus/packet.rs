//! Opus packet parsing.
//!
//! This module handles parsing of Opus packets, including:
//! - TOC (Table of Contents) byte parsing
//! - Frame duration detection
//! - CBR/VBR mode detection
//! - Multi-frame packet handling
//!
//! # Opus Packet Structure
//!
//! An Opus packet consists of:
//! 1. TOC byte - configuration, stereo flag, frame count code
//! 2. Optional frame count byte (for code 3 packets)
//! 3. Frame length bytes (for VBR multi-frame packets)
//! 4. Compressed frame data

#![forbid(unsafe_code)]

use crate::AudioError;

/// Opus operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpusMode {
    /// SILK-only mode (narrow to wideband speech).
    SilkOnly,
    /// Hybrid mode (SILK + CELT for super-wideband/fullband speech).
    Hybrid,
    /// CELT-only mode (fullband music/general audio).
    CeltOnly,
}

impl OpusMode {
    /// Determine mode from configuration number (0-31).
    #[must_use]
    pub fn from_config(config: u8) -> Self {
        match config {
            0..=11 => OpusMode::SilkOnly,
            12..=15 => OpusMode::Hybrid,
            // 16..=31 and any invalid values default to CeltOnly
            _ => OpusMode::CeltOnly,
        }
    }

    /// Returns true if this mode uses SILK.
    #[must_use]
    pub fn uses_silk(self) -> bool {
        matches!(self, OpusMode::SilkOnly | OpusMode::Hybrid)
    }

    /// Returns true if this mode uses CELT.
    #[must_use]
    pub fn uses_celt(self) -> bool {
        matches!(self, OpusMode::CeltOnly | OpusMode::Hybrid)
    }
}

/// Opus bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum OpusBandwidth {
    /// Narrowband (4kHz).
    Narrow,
    /// Medium band (6kHz).
    Medium,
    /// Wideband (8kHz).
    Wide,
    /// Super wideband (12kHz).
    SuperWide,
    /// Fullband (20kHz).
    Full,
}

impl OpusBandwidth {
    /// Get bandwidth from configuration number.
    #[must_use]
    pub fn from_config(config: u8) -> Self {
        match config {
            0..=3 => OpusBandwidth::Narrow,
            4..=7 | 16..=19 => OpusBandwidth::Medium,
            8..=11 | 20..=23 => OpusBandwidth::Wide,
            12..=13 | 24..=27 => OpusBandwidth::SuperWide,
            // 14..=15, 28..=31, and any other values default to Full
            _ => OpusBandwidth::Full,
        }
    }

    /// Get audio bandwidth in Hz.
    #[must_use]
    pub fn audio_bandwidth_hz(self) -> u32 {
        match self {
            OpusBandwidth::Narrow => 4000,
            OpusBandwidth::Medium => 6000,
            OpusBandwidth::Wide => 8000,
            OpusBandwidth::SuperWide => 12000,
            OpusBandwidth::Full => 20000,
        }
    }
}

/// Frame duration in samples at 48kHz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDuration {
    /// 2.5ms (120 samples at 48kHz).
    Ms2_5,
    /// 5ms (240 samples at 48kHz).
    Ms5,
    /// 10ms (480 samples at 48kHz).
    Ms10,
    /// 20ms (960 samples at 48kHz).
    Ms20,
    /// 40ms (1920 samples at 48kHz).
    Ms40,
    /// 60ms (2880 samples at 48kHz).
    Ms60,
}

impl FrameDuration {
    /// Get duration from configuration number.
    #[must_use]
    pub fn from_config(config: u8) -> Self {
        match config {
            0 | 4 | 8 | 12 | 14 | 16 | 20 | 24 | 28 => FrameDuration::Ms10,
            2 | 6 | 10 | 18 | 22 | 26 | 30 => FrameDuration::Ms40,
            3 | 7 | 11 | 19 | 23 | 27 | 31 => FrameDuration::Ms60,
            // 1, 5, 9, 13, 15, 17, 21, 25, 29 and other values default to Ms20
            _ => FrameDuration::Ms20,
        }
    }

    /// Get CELT-specific frame duration from config.
    #[must_use]
    pub fn from_celt_config(config: u8) -> Self {
        // CELT configs 16-31 have different duration mapping
        match config & 0x03 {
            0 => FrameDuration::Ms2_5,
            1 => FrameDuration::Ms5,
            3 => FrameDuration::Ms20,
            // 2 and any other values default to Ms10
            _ => FrameDuration::Ms10,
        }
    }

    /// Get duration in samples at 48kHz.
    #[must_use]
    pub fn samples_at_48khz(self) -> u32 {
        match self {
            FrameDuration::Ms2_5 => 120,
            FrameDuration::Ms5 => 240,
            FrameDuration::Ms10 => 480,
            FrameDuration::Ms20 => 960,
            FrameDuration::Ms40 => 1920,
            FrameDuration::Ms60 => 2880,
        }
    }

    /// Get duration in microseconds.
    #[must_use]
    pub fn microseconds(self) -> u32 {
        match self {
            FrameDuration::Ms2_5 => 2500,
            FrameDuration::Ms5 => 5000,
            FrameDuration::Ms10 => 10000,
            FrameDuration::Ms20 => 20000,
            FrameDuration::Ms40 => 40000,
            FrameDuration::Ms60 => 60000,
        }
    }
}

/// Frame count code (2 bits from TOC byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCount {
    /// Code 0: 1 frame in packet.
    One,
    /// Code 1: 2 frames in packet, equal size.
    TwoEqualSize,
    /// Code 2: 2 frames in packet, different sizes.
    TwoDifferentSize,
    /// Code 3: Arbitrary number of frames (see frame count byte).
    Arbitrary,
}

impl FrameCount {
    /// Create from 2-bit code.
    #[must_use]
    pub fn from_code(code: u8) -> Self {
        match code & 0x03 {
            0 => FrameCount::One,
            1 => FrameCount::TwoEqualSize,
            2 => FrameCount::TwoDifferentSize,
            3 => FrameCount::Arbitrary,
            _ => unreachable!(),
        }
    }

    /// Returns true if this is a CBR multi-frame packet.
    #[must_use]
    pub fn is_cbr(self) -> bool {
        matches!(self, FrameCount::One | FrameCount::TwoEqualSize)
    }

    /// Get minimum number of frames for this code.
    #[must_use]
    pub fn min_frames(self) -> u8 {
        match self {
            FrameCount::TwoEqualSize | FrameCount::TwoDifferentSize => 2,
            FrameCount::One | FrameCount::Arbitrary => 1,
        }
    }
}

/// Parsed TOC (Table of Contents) byte.
#[derive(Debug, Clone, Copy)]
pub struct TocByte {
    /// Raw TOC byte value.
    pub raw: u8,
    /// Configuration number (0-31).
    pub config: u8,
    /// Stereo flag.
    pub stereo: bool,
    /// Frame count code.
    pub frame_count: FrameCount,
    /// Operating mode.
    pub mode: OpusMode,
    /// Bandwidth.
    pub bandwidth: OpusBandwidth,
    /// Frame duration.
    pub duration: FrameDuration,
}

impl TocByte {
    /// Parse TOC byte from raw value.
    #[must_use]
    pub fn parse(byte: u8) -> Self {
        let config = (byte >> 3) & 0x1F;
        let stereo = (byte & 0x04) != 0;
        let frame_count = FrameCount::from_code(byte & 0x03);
        let mode = OpusMode::from_config(config);
        let bandwidth = OpusBandwidth::from_config(config);
        let duration = if config >= 16 {
            FrameDuration::from_celt_config(config)
        } else {
            FrameDuration::from_config(config)
        };

        Self {
            raw: byte,
            config,
            stereo,
            frame_count,
            mode,
            bandwidth,
            duration,
        }
    }

    /// Get number of channels.
    #[must_use]
    pub fn channels(self) -> u8 {
        if self.stereo {
            2
        } else {
            1
        }
    }
}

/// Opus packet configuration.
#[derive(Debug, Clone)]
pub struct OpusPacketConfig {
    /// TOC byte information.
    pub toc: TocByte,
    /// Number of frames in this packet.
    pub frame_count: u8,
    /// Sizes of each frame in bytes.
    pub frame_sizes: Vec<u16>,
    /// Whether this is a VBR packet.
    pub is_vbr: bool,
    /// Whether padding is present.
    pub has_padding: bool,
    /// Padding length in bytes.
    pub padding_length: usize,
}

/// Parsed Opus packet.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OpusPacket {
    /// Packet configuration.
    pub config: OpusPacketConfig,
    /// Frame data (compressed).
    pub frames: Vec<Vec<u8>>,
    /// Total packet size.
    pub total_size: usize,
}

impl OpusPacket {
    /// Parse an Opus packet from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns error if packet is invalid or truncated.
    #[allow(clippy::too_many_lines, clippy::cast_possible_truncation)]
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("Empty Opus packet".into()));
        }

        let toc = TocByte::parse(data[0]);
        let mut offset = 1;

        // Parse based on frame count code
        let (frame_count, frame_sizes, is_vbr, has_padding, padding_length) = match toc.frame_count
        {
            FrameCount::One => {
                // Single frame, size is rest of packet
                let frame_size = data.len() - 1;
                (1u8, vec![frame_size as u16], false, false, 0)
            }
            FrameCount::TwoEqualSize => {
                // Two frames, equal size (CBR)
                let remaining = data.len() - 1;
                if remaining % 2 != 0 {
                    return Err(AudioError::InvalidData(
                        "Odd size for two equal frames".into(),
                    ));
                }
                let frame_size = (remaining / 2) as u16;
                (2, vec![frame_size, frame_size], false, false, 0)
            }
            FrameCount::TwoDifferentSize => {
                // Two frames, different sizes (VBR)
                if data.len() < 2 {
                    return Err(AudioError::InvalidData("Packet too short for VBR".into()));
                }
                let (size1, bytes_read) = Self::parse_frame_size(&data[offset..])?;
                offset += bytes_read;
                let size2 = (data.len() - offset - size1 as usize) as u16;
                (2, vec![size1, size2], true, false, 0)
            }
            FrameCount::Arbitrary => {
                // Arbitrary number of frames
                if data.len() < 2 {
                    return Err(AudioError::InvalidData(
                        "Packet too short for frame count byte".into(),
                    ));
                }
                let frame_count_byte = data[offset];
                offset += 1;

                let is_vbr = (frame_count_byte & 0x80) != 0;
                let has_padding = (frame_count_byte & 0x40) != 0;
                let count = frame_count_byte & 0x3F;

                if count == 0 {
                    return Err(AudioError::InvalidData("Zero frame count".into()));
                }

                // Parse padding if present
                let mut padding_length = 0usize;
                if has_padding {
                    loop {
                        if offset >= data.len() {
                            return Err(AudioError::InvalidData("Truncated padding length".into()));
                        }
                        let pad_byte = data[offset] as usize;
                        offset += 1;
                        padding_length += pad_byte;
                        if pad_byte != 255 {
                            break;
                        }
                    }
                }

                // Parse frame sizes
                let mut sizes = Vec::with_capacity(count as usize);
                if is_vbr {
                    // VBR: all but last frame have explicit sizes
                    for _ in 0..(count - 1) {
                        let (size, bytes_read) = Self::parse_frame_size(&data[offset..])?;
                        offset += bytes_read;
                        sizes.push(size);
                    }
                    // Last frame size is remainder
                    let total_frame_sizes: usize = sizes.iter().map(|&s| s as usize).sum();
                    let remaining = data.len() - offset - padding_length;
                    if remaining < total_frame_sizes {
                        return Err(AudioError::InvalidData("Frame sizes exceed packet".into()));
                    }
                    sizes.push((remaining - total_frame_sizes) as u16);
                } else {
                    // CBR: all frames same size
                    let remaining = data.len() - offset - padding_length;
                    if remaining % (count as usize) != 0 {
                        return Err(AudioError::InvalidData(
                            "CBR frame size not divisible".into(),
                        ));
                    }
                    let frame_size = (remaining / count as usize) as u16;
                    sizes.resize(count as usize, frame_size);
                }

                (count, sizes, is_vbr, has_padding, padding_length)
            }
        };

        // Extract frames
        let mut frames = Vec::with_capacity(frame_count as usize);
        for &size in &frame_sizes {
            if offset + (size as usize) > data.len() {
                return Err(AudioError::InvalidData(
                    "Frame extends beyond packet".into(),
                ));
            }
            frames.push(data[offset..offset + size as usize].to_vec());
            offset += size as usize;
        }

        Ok(Self {
            config: OpusPacketConfig {
                toc,
                frame_count,
                frame_sizes,
                is_vbr,
                has_padding,
                padding_length,
            },
            frames,
            total_size: data.len(),
        })
    }

    /// Parse variable-length frame size.
    fn parse_frame_size(data: &[u8]) -> Result<(u16, usize), AudioError> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("No frame size data".into()));
        }

        let first = u16::from(data[0]);
        if first < 252 {
            Ok((first, 1))
        } else if data.len() < 2 {
            Err(AudioError::InvalidData("Truncated frame size".into()))
        } else {
            let size = u16::from(data[1]) * 4 + first;
            Ok((size, 2))
        }
    }

    /// Get total duration in samples at 48kHz.
    #[must_use]
    pub fn total_samples_48khz(&self) -> u32 {
        self.config.toc.duration.samples_at_48khz() * u32::from(self.config.frame_count)
    }

    /// Get total duration in microseconds.
    #[must_use]
    pub fn total_duration_us(&self) -> u32 {
        self.config.toc.duration.microseconds() * u32::from(self.config.frame_count)
    }

    /// Check if this is a SILK-only packet.
    #[must_use]
    pub fn is_silk_only(&self) -> bool {
        self.config.toc.mode == OpusMode::SilkOnly
    }

    /// Check if this is a CELT-only packet.
    #[must_use]
    pub fn is_celt_only(&self) -> bool {
        self.config.toc.mode == OpusMode::CeltOnly
    }

    /// Check if this is a hybrid packet.
    #[must_use]
    pub fn is_hybrid(&self) -> bool {
        self.config.toc.mode == OpusMode::Hybrid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toc_byte_parsing() {
        // Config 0, mono, 1 frame
        let toc = TocByte::parse(0b00000000);
        assert_eq!(toc.config, 0);
        assert!(!toc.stereo);
        assert_eq!(toc.frame_count, FrameCount::One);
        assert_eq!(toc.mode, OpusMode::SilkOnly);
        assert_eq!(toc.bandwidth, OpusBandwidth::Narrow);
    }

    #[test]
    fn test_toc_stereo_flag() {
        let toc = TocByte::parse(0b00000100);
        assert!(toc.stereo);
        assert_eq!(toc.channels(), 2);
    }

    #[test]
    fn test_toc_frame_count_codes() {
        assert_eq!(FrameCount::from_code(0), FrameCount::One);
        assert_eq!(FrameCount::from_code(1), FrameCount::TwoEqualSize);
        assert_eq!(FrameCount::from_code(2), FrameCount::TwoDifferentSize);
        assert_eq!(FrameCount::from_code(3), FrameCount::Arbitrary);
    }

    #[test]
    fn test_opus_mode_from_config() {
        assert_eq!(OpusMode::from_config(0), OpusMode::SilkOnly);
        assert_eq!(OpusMode::from_config(11), OpusMode::SilkOnly);
        assert_eq!(OpusMode::from_config(12), OpusMode::Hybrid);
        assert_eq!(OpusMode::from_config(15), OpusMode::Hybrid);
        assert_eq!(OpusMode::from_config(16), OpusMode::CeltOnly);
        assert_eq!(OpusMode::from_config(31), OpusMode::CeltOnly);
    }

    #[test]
    fn test_frame_duration_samples() {
        assert_eq!(FrameDuration::Ms2_5.samples_at_48khz(), 120);
        assert_eq!(FrameDuration::Ms5.samples_at_48khz(), 240);
        assert_eq!(FrameDuration::Ms10.samples_at_48khz(), 480);
        assert_eq!(FrameDuration::Ms20.samples_at_48khz(), 960);
        assert_eq!(FrameDuration::Ms40.samples_at_48khz(), 1920);
        assert_eq!(FrameDuration::Ms60.samples_at_48khz(), 2880);
    }

    #[test]
    fn test_bandwidth_hz() {
        assert_eq!(OpusBandwidth::Narrow.audio_bandwidth_hz(), 4000);
        assert_eq!(OpusBandwidth::Medium.audio_bandwidth_hz(), 6000);
        assert_eq!(OpusBandwidth::Wide.audio_bandwidth_hz(), 8000);
        assert_eq!(OpusBandwidth::SuperWide.audio_bandwidth_hz(), 12000);
        assert_eq!(OpusBandwidth::Full.audio_bandwidth_hz(), 20000);
    }

    #[test]
    fn test_single_frame_packet() {
        // TOC: config 0, mono, 1 frame + 10 bytes of data
        let data = vec![0b00000000, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let packet = OpusPacket::parse(&data).expect("should succeed");
        assert_eq!(packet.config.frame_count, 1);
        assert_eq!(packet.frames.len(), 1);
        assert_eq!(packet.frames[0].len(), 10);
    }

    #[test]
    fn test_two_equal_frames_packet() {
        // TOC: config 0, mono, 2 equal frames + 10 bytes of data
        let data = vec![0b00000001, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let packet = OpusPacket::parse(&data).expect("should succeed");
        assert_eq!(packet.config.frame_count, 2);
        assert_eq!(packet.frames.len(), 2);
        assert_eq!(packet.frames[0].len(), 5);
        assert_eq!(packet.frames[1].len(), 5);
    }

    #[test]
    fn test_empty_packet_error() {
        let result = OpusPacket::parse(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_opus_mode_uses_silk() {
        assert!(OpusMode::SilkOnly.uses_silk());
        assert!(OpusMode::Hybrid.uses_silk());
        assert!(!OpusMode::CeltOnly.uses_silk());
    }

    #[test]
    fn test_opus_mode_uses_celt() {
        assert!(!OpusMode::SilkOnly.uses_celt());
        assert!(OpusMode::Hybrid.uses_celt());
        assert!(OpusMode::CeltOnly.uses_celt());
    }

    #[test]
    fn test_frame_count_is_cbr() {
        assert!(FrameCount::One.is_cbr());
        assert!(FrameCount::TwoEqualSize.is_cbr());
        assert!(!FrameCount::TwoDifferentSize.is_cbr());
        assert!(!FrameCount::Arbitrary.is_cbr());
    }
}
