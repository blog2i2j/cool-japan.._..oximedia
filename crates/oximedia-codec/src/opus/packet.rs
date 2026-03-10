//! Opus packet parsing and frame structure.
//!
//! This module handles parsing of Opus packets according to RFC 6716.
//! Opus packets contain one or more compressed audio frames.

use crate::{CodecError, CodecResult};

/// Opus operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusMode {
    /// SILK mode - optimized for speech
    Silk,
    /// CELT mode - optimized for music
    Celt,
    /// Hybrid mode - combines SILK and CELT
    Hybrid,
}

impl OpusMode {
    /// Returns a string representation of the mode.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Silk => "SILK",
            Self::Celt => "CELT",
            Self::Hybrid => "Hybrid",
        }
    }
}

/// Opus bandwidth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpusBandwidth {
    /// Narrowband (4 kHz)
    Narrowband,
    /// Mediumband (6 kHz)
    Mediumband,
    /// Wideband (8 kHz)
    Wideband,
    /// Super-wideband (12 kHz)
    SuperWideband,
    /// Fullband (20 kHz)
    Fullband,
}

impl OpusBandwidth {
    /// Returns the maximum frequency in Hz.
    #[must_use]
    pub const fn max_frequency(&self) -> u32 {
        match self {
            Self::Narrowband => 4000,
            Self::Mediumband => 6000,
            Self::Wideband => 8000,
            Self::SuperWideband => 12000,
            Self::Fullband => 20000,
        }
    }

    /// Returns the bandwidth from configuration value.
    #[must_use]
    pub const fn from_config(config: u8) -> Self {
        match config {
            0 => Self::Narrowband,
            1 => Self::Mediumband,
            2 => Self::Wideband,
            3 => Self::SuperWideband,
            _ => Self::Fullband,
        }
    }
}

/// Opus frame configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameConfig {
    /// Single frame
    Single,
    /// Two frames (equal size)
    Double,
    /// Two frames (different sizes)
    DoubleDifferent,
    /// Multiple frames (equal size)
    Multiple {
        /// Number of frames
        count: u8,
    },
    /// Arbitrary frame count
    Arbitrary {
        /// Number of frames
        count: u8,
    },
}

/// Table of Contents (TOC) byte information.
///
/// The TOC byte contains configuration information for the entire packet.
#[derive(Debug, Clone)]
pub struct TocInfo {
    /// Operating mode
    pub mode: OpusMode,
    /// Bandwidth
    pub bandwidth: OpusBandwidth,
    /// Frame size in samples at 48 kHz
    pub frame_size: u16,
    /// Frame configuration
    pub config: FrameConfig,
    /// Stereo flag
    pub is_stereo: bool,
}

impl TocInfo {
    /// Parses TOC byte from packet.
    ///
    /// # Arguments
    ///
    /// * `toc` - TOC byte (first byte of packet)
    pub fn parse(toc: u8) -> CodecResult<Self> {
        // Extract configuration (top 5 bits)
        let config = toc >> 3;

        // Extract frame type (bits 2-3)
        let frame_type = (toc >> 3) & 0x1F;

        // Extract stereo flag (bit 2)
        let is_stereo = (toc & 0x04) != 0;

        // Extract frame count code (bits 0-1)
        let frame_code = toc & 0x03;

        // Determine mode and bandwidth from configuration
        let (mode, bandwidth) = Self::decode_config(config)?;

        // Determine frame size from configuration
        let frame_size = Self::decode_frame_size(config)?;

        // Determine frame configuration
        let config = match frame_code {
            0 => FrameConfig::Single,
            1 => FrameConfig::Double,
            2 => FrameConfig::DoubleDifferent,
            3 => FrameConfig::Arbitrary { count: 0 }, // Count determined later
            _ => {
                return Err(CodecError::InvalidData(format!(
                    "Invalid frame code: {frame_code}"
                )))
            }
        };

        Ok(Self {
            mode,
            bandwidth,
            frame_size,
            config,
            is_stereo,
        })
    }

    /// Decodes mode and bandwidth from configuration value.
    fn decode_config(config: u8) -> CodecResult<(OpusMode, OpusBandwidth)> {
        let mode = if config < 12 {
            OpusMode::Silk
        } else if config < 16 {
            OpusMode::Hybrid
        } else {
            OpusMode::Celt
        };

        let bandwidth = match config {
            0..=11 => {
                // SILK mode
                let bw_config = config >> 2;
                OpusBandwidth::from_config(bw_config)
            }
            12..=15 => {
                // Hybrid mode
                let bw_config = config - 12;
                OpusBandwidth::from_config(bw_config + 3) // SuperWideband or Fullband
            }
            16..=31 => {
                // CELT mode
                let bw_config = (config - 16) >> 2;
                OpusBandwidth::from_config(bw_config)
            }
            _ => {
                return Err(CodecError::InvalidData(format!(
                    "Invalid configuration: {config}"
                )))
            }
        };

        Ok((mode, bandwidth))
    }

    /// Decodes frame size from configuration value.
    fn decode_frame_size(config: u8) -> CodecResult<u16> {
        let size = match config {
            0..=15 => {
                // SILK and Hybrid modes
                let size_code = config & 0x03;
                match size_code {
                    0 => 480,  // 10 ms at 48 kHz
                    1 => 960,  // 20 ms at 48 kHz
                    2 => 1920, // 40 ms at 48 kHz
                    3 => 2880, // 60 ms at 48 kHz
                    _ => unreachable!(),
                }
            }
            16..=31 => {
                // CELT mode
                let size_code = config & 0x03;
                match size_code {
                    0 => 120, // 2.5 ms at 48 kHz
                    1 => 240, // 5 ms at 48 kHz
                    2 => 480, // 10 ms at 48 kHz
                    3 => 960, // 20 ms at 48 kHz
                    _ => unreachable!(),
                }
            }
            _ => {
                return Err(CodecError::InvalidData(format!(
                    "Invalid configuration: {config}"
                )))
            }
        };

        Ok(size)
    }
}

/// Opus packet structure.
#[derive(Debug, Clone)]
pub struct OpusPacket<'a> {
    /// TOC information
    pub toc: TocInfo,
    /// Frame data
    pub frames: Vec<&'a [u8]>,
    /// Padding bytes
    pub padding: usize,
}

impl<'a> OpusPacket<'a> {
    /// Parses an Opus packet.
    ///
    /// # Arguments
    ///
    /// * `data` - Packet data
    pub fn parse(data: &'a [u8]) -> CodecResult<Self> {
        if data.is_empty() {
            return Err(CodecError::InvalidData("Empty packet".to_string()));
        }

        // Parse TOC byte
        let toc = TocInfo::parse(data[0])?;
        let mut pos = 1;

        // Parse frames based on configuration
        let (frames, padding) = match toc.config {
            FrameConfig::Single => {
                // Single frame - rest of packet is frame data
                let frame_data = &data[pos..];
                (vec![frame_data], 0)
            }
            FrameConfig::Double => {
                // Two equal-sized frames
                let remaining = data.len() - pos;
                if remaining % 2 != 0 {
                    return Err(CodecError::InvalidData(
                        "Invalid double frame size".to_string(),
                    ));
                }
                let frame_size = remaining / 2;
                let frame1 = &data[pos..pos + frame_size];
                let frame2 = &data[pos + frame_size..pos + 2 * frame_size];
                (vec![frame1, frame2], 0)
            }
            FrameConfig::DoubleDifferent => {
                // Two frames with different sizes
                if pos >= data.len() {
                    return Err(CodecError::InvalidData("Truncated packet".to_string()));
                }
                let size1 = data[pos] as usize;
                pos += 1;

                if pos + size1 >= data.len() {
                    return Err(CodecError::InvalidData("Invalid frame size".to_string()));
                }

                let frame1 = &data[pos..pos + size1];
                pos += size1;
                let frame2 = &data[pos..];

                (vec![frame1, frame2], 0)
            }
            FrameConfig::Multiple { count } | FrameConfig::Arbitrary { count } => {
                // Multiple frames or arbitrary count
                // For now, treat as single frame
                let frame_data = &data[pos..];
                (vec![frame_data], 0)
            }
        };

        Ok(Self {
            toc,
            frames,
            padding,
        })
    }

    /// Returns the number of frames in this packet.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns the total size of all frames.
    #[must_use]
    pub fn total_size(&self) -> usize {
        self.frames.iter().map(|f| f.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_bandwidth() {
        assert_eq!(OpusBandwidth::Narrowband.max_frequency(), 4000);
        assert_eq!(OpusBandwidth::Wideband.max_frequency(), 8000);
        assert_eq!(OpusBandwidth::Fullband.max_frequency(), 20000);
    }

    #[test]
    fn test_toc_parse() {
        // SILK narrowband, 10ms, mono, single frame
        let toc = TocInfo::parse(0x00).expect("should succeed");
        assert_eq!(toc.mode, OpusMode::Silk);
        assert_eq!(toc.bandwidth, OpusBandwidth::Narrowband);
        assert_eq!(toc.frame_size, 480);
        assert!(!toc.is_stereo);
    }

    #[test]
    fn test_packet_parse_single() {
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let packet = OpusPacket::parse(&data).expect("should succeed");
        assert_eq!(packet.frame_count(), 1);
        assert_eq!(packet.frames[0].len(), 3);
    }

    #[test]
    fn test_packet_parse_double() {
        let data = vec![0x01, 0xAA, 0xBB, 0xCC, 0xDD];
        let packet = OpusPacket::parse(&data).expect("should succeed");
        assert_eq!(packet.frame_count(), 2);
        assert_eq!(packet.frames[0].len(), 2);
        assert_eq!(packet.frames[1].len(), 2);
    }

    #[test]
    fn test_empty_packet() {
        let data: Vec<u8> = vec![];
        let result = OpusPacket::parse(&data);
        assert!(result.is_err());
    }
}
