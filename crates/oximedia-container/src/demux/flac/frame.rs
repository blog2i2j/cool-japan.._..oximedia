//! FLAC frame header parsing.
//!
//! This module handles parsing of FLAC audio frame headers.
//!
//! # Frame Structure
//!
//! Each FLAC frame consists of:
//! 1. Frame header (variable length, 4-16 bytes)
//! 2. Subframes (one per channel)
//! 3. Zero padding to byte boundary
//! 4. CRC-16 footer
//!
//! The frame header contains information needed to decode the frame,
//! some of which may override values from `STREAMINFO`.

use oximedia_core::{OxiError, OxiResult};

/// FLAC frame sync code (14 bits: `0x3FFE`).
///
/// All FLAC frames begin with the sync code `0xFFF8` or `0xFFF9`,
/// where the last bit indicates the blocking strategy.
pub const FRAME_SYNC: u16 = 0x3FFE;

/// Frame header parsed from a FLAC audio frame.
///
/// Contains all information needed to decode the frame's audio data.
///
/// # Example
///
/// ```ignore
/// let (header, consumed) = FrameHeader::parse(&frame_data)?;
/// println!("Block size: {} samples", header.block_size);
/// println!("Channels: {}", header.channels());
/// ```
#[derive(Clone, Debug)]
pub struct FrameHeader {
    /// Blocking strategy.
    ///
    /// - `false`: Fixed block size (frame number in header)
    /// - `true`: Variable block size (sample number in header)
    pub variable_blocksize: bool,

    /// Number of samples in this frame (per channel).
    pub block_size: u32,

    /// Sample rate in Hz, if specified in header.
    ///
    /// `None` means use the value from `STREAMINFO`.
    pub sample_rate: Option<u32>,

    /// Channel assignment.
    pub channel_assignment: ChannelAssignment,

    /// Sample size in bits, if specified in header.
    ///
    /// `None` means use the value from `STREAMINFO`.
    pub sample_size: Option<u8>,

    /// Frame number (fixed blocksize) or sample number (variable blocksize).
    pub number: u64,

    /// CRC-8 of the header (excluding sync code).
    pub crc: u8,
}

/// Channel assignment types for stereo decorrelation.
///
/// FLAC supports both independent channels and stereo decorrelation
/// modes for improved compression of stereo audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelAssignment {
    /// Independent channels (no decorrelation).
    ///
    /// The parameter indicates the number of channels (1-8).
    Independent(u8),

    /// Left/side stereo decorrelation.
    ///
    /// Channel 0: left, Channel 1: side (left - right)
    LeftSide,

    /// Right/side stereo decorrelation.
    ///
    /// Channel 0: side (left - right), Channel 1: right
    RightSide,

    /// Mid/side stereo decorrelation.
    ///
    /// Channel 0: mid ((left + right) / 2), Channel 1: side (left - right)
    MidSide,
}

impl FrameHeader {
    /// Parse a frame header from bytes.
    ///
    /// Returns the parsed header and the number of bytes consumed.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The sync code is not found
    /// - Reserved bits are set
    /// - Block size or sample rate codes are invalid
    /// - The input is too short
    #[allow(clippy::too_many_lines)]
    pub fn parse(data: &[u8]) -> OxiResult<(Self, usize)> {
        if data.len() < 4 {
            return Err(OxiError::UnexpectedEof);
        }

        // Check sync code (first 14 bits must be 0x3FFE)
        let sync = (u16::from(data[0]) << 6) | (u16::from(data[1]) >> 2);
        if sync != FRAME_SYNC {
            return Err(OxiError::Parse {
                offset: 0,
                message: format!(
                    "Invalid frame sync code: expected 0x{FRAME_SYNC:04X}, got 0x{sync:04X}"
                ),
            });
        }

        // Bit 14: reserved (must be 0)
        if data[1] & 0x02 != 0 {
            return Err(OxiError::Parse {
                offset: 1,
                message: "Reserved bit is not zero".into(),
            });
        }

        // Bit 15: blocking strategy
        let variable_blocksize = data[1] & 0x01 != 0;

        // Bits 16-19: block size code
        let block_size_code = (data[2] >> 4) & 0x0F;

        // Bits 20-23: sample rate code
        let sample_rate_code = data[2] & 0x0F;

        // Bits 24-27: channel assignment
        let channel_code = (data[3] >> 4) & 0x0F;

        // Bits 28-30: sample size code
        let sample_size_code = (data[3] >> 1) & 0x07;

        // Bit 31: reserved (must be 0)
        if data[3] & 0x01 != 0 {
            return Err(OxiError::Parse {
                offset: 3,
                message: "Reserved bit is not zero".into(),
            });
        }

        let mut offset = 4;

        // Parse UTF-8 coded frame/sample number
        let (number, consumed) = parse_utf8_number(&data[offset..])?;
        offset += consumed;

        // Parse block size if needed
        let block_size = match block_size_code {
            0 => {
                return Err(OxiError::Parse {
                    offset: 2,
                    message: "Reserved block size code".into(),
                })
            }
            1 => 192,
            2..=5 => 576 << (block_size_code - 2),
            6 => {
                if offset >= data.len() {
                    return Err(OxiError::UnexpectedEof);
                }
                let size = u32::from(data[offset]) + 1;
                offset += 1;
                size
            }
            7 => {
                if offset + 1 >= data.len() {
                    return Err(OxiError::UnexpectedEof);
                }
                let size = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]])) + 1;
                offset += 2;
                size
            }
            _ => 256 << (block_size_code - 8),
        };

        // Parse sample rate if needed
        let sample_rate = match sample_rate_code {
            0 => None, // From STREAMINFO
            1 => Some(88_200),
            2 => Some(176_400),
            3 => Some(192_000),
            4 => Some(8_000),
            5 => Some(16_000),
            6 => Some(22_050),
            7 => Some(24_000),
            8 => Some(32_000),
            9 => Some(44_100),
            10 => Some(48_000),
            11 => Some(96_000),
            12 => {
                if offset >= data.len() {
                    return Err(OxiError::UnexpectedEof);
                }
                let rate = u32::from(data[offset]) * 1000;
                offset += 1;
                Some(rate)
            }
            13 => {
                if offset + 1 >= data.len() {
                    return Err(OxiError::UnexpectedEof);
                }
                let rate = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]]));
                offset += 2;
                Some(rate)
            }
            14 => {
                if offset + 1 >= data.len() {
                    return Err(OxiError::UnexpectedEof);
                }
                let rate = u32::from(u16::from_be_bytes([data[offset], data[offset + 1]])) * 10;
                offset += 2;
                Some(rate)
            }
            15 => {
                return Err(OxiError::Parse {
                    offset: 2,
                    message: "Invalid sample rate code 15".into(),
                })
            }
            _ => unreachable!(),
        };

        // Channel assignment
        let channel_assignment = match channel_code {
            0..=7 => ChannelAssignment::Independent(channel_code + 1),
            8 => ChannelAssignment::LeftSide,
            9 => ChannelAssignment::RightSide,
            10 => ChannelAssignment::MidSide,
            _ => {
                return Err(OxiError::Parse {
                    offset: 3,
                    message: format!("Reserved channel assignment code: {channel_code}"),
                })
            }
        };

        // Sample size
        let sample_size = match sample_size_code {
            0 => None, // From STREAMINFO
            1 => Some(8),
            2 => Some(12),
            3 => {
                return Err(OxiError::Parse {
                    offset: 3,
                    message: "Reserved sample size code 3".into(),
                })
            }
            4 => Some(16),
            5 => Some(20),
            6 => Some(24),
            7 => Some(32),
            _ => unreachable!(),
        };

        // CRC-8
        if offset >= data.len() {
            return Err(OxiError::UnexpectedEof);
        }
        let crc = data[offset];
        offset += 1;

        Ok((
            Self {
                variable_blocksize,
                block_size,
                sample_rate,
                channel_assignment,
                sample_size,
                number,
                crc,
            },
            offset,
        ))
    }

    /// Returns the number of channels.
    #[must_use]
    pub const fn channels(&self) -> u8 {
        match self.channel_assignment {
            ChannelAssignment::Independent(n) => n,
            ChannelAssignment::LeftSide
            | ChannelAssignment::RightSide
            | ChannelAssignment::MidSide => 2,
        }
    }

    /// Returns true if this frame uses stereo decorrelation.
    #[must_use]
    pub const fn uses_stereo_decorrelation(&self) -> bool {
        matches!(
            self.channel_assignment,
            ChannelAssignment::LeftSide | ChannelAssignment::RightSide | ChannelAssignment::MidSide
        )
    }
}

/// Parse a UTF-8 coded variable-length integer.
///
/// FLAC uses a modified UTF-8 encoding for frame/sample numbers:
/// - 0xxxxxxx: 7-bit value (0-127)
/// - 110xxxxx 10xxxxxx: 11-bit value
/// - And so on, up to 36-bit values
///
/// Returns the value and the number of bytes consumed.
fn parse_utf8_number(data: &[u8]) -> OxiResult<(u64, usize)> {
    if data.is_empty() {
        return Err(OxiError::UnexpectedEof);
    }

    let first = data[0];

    // Determine number of bytes based on leading bits
    let (mut value, len) = if first < 0x80 {
        // 0xxxxxxx - single byte
        (u64::from(first), 1)
    } else if first < 0xC0 {
        // 10xxxxxx - invalid start byte
        return Err(OxiError::Parse {
            offset: 0,
            message: "Invalid UTF-8 start byte".into(),
        });
    } else if first < 0xE0 {
        // 110xxxxx - two bytes
        if data.len() < 2 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(first & 0x1F) << 6) | u64::from(data[1] & 0x3F);
        (value, 2)
    } else if first < 0xF0 {
        // 1110xxxx - three bytes
        if data.len() < 3 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(first & 0x0F) << 12)
            | (u64::from(data[1] & 0x3F) << 6)
            | u64::from(data[2] & 0x3F);
        (value, 3)
    } else if first < 0xF8 {
        // 11110xxx - four bytes
        if data.len() < 4 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(first & 0x07) << 18)
            | (u64::from(data[1] & 0x3F) << 12)
            | (u64::from(data[2] & 0x3F) << 6)
            | u64::from(data[3] & 0x3F);
        (value, 4)
    } else if first < 0xFC {
        // 111110xx - five bytes
        if data.len() < 5 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(first & 0x03) << 24)
            | (u64::from(data[1] & 0x3F) << 18)
            | (u64::from(data[2] & 0x3F) << 12)
            | (u64::from(data[3] & 0x3F) << 6)
            | u64::from(data[4] & 0x3F);
        (value, 5)
    } else if first < 0xFE {
        // 1111110x - six bytes
        if data.len() < 6 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(first & 0x01) << 30)
            | (u64::from(data[1] & 0x3F) << 24)
            | (u64::from(data[2] & 0x3F) << 18)
            | (u64::from(data[3] & 0x3F) << 12)
            | (u64::from(data[4] & 0x3F) << 6)
            | u64::from(data[5] & 0x3F);
        (value, 6)
    } else {
        // 11111110 - seven bytes (36-bit value)
        if data.len() < 7 {
            return Err(OxiError::UnexpectedEof);
        }
        let value = (u64::from(data[1] & 0x3F) << 30)
            | (u64::from(data[2] & 0x3F) << 24)
            | (u64::from(data[3] & 0x3F) << 18)
            | (u64::from(data[4] & 0x3F) << 12)
            | (u64::from(data[5] & 0x3F) << 6)
            | u64::from(data[6] & 0x3F);
        (value, 7)
    };

    // Validate continuation bytes
    for byte in data.iter().take(len).skip(1) {
        if byte & 0xC0 != 0x80 {
            return Err(OxiError::Parse {
                offset: 0,
                message: "Invalid UTF-8 continuation byte".into(),
            });
        }
    }

    // Re-parse with validation
    if len > 1 {
        value = u64::from(first) & (0x7F >> len);
        for byte in data.iter().take(len).skip(1) {
            value = (value << 6) | u64::from(byte & 0x3F);
        }
    }

    Ok((value, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_sync() {
        assert_eq!(FRAME_SYNC, 0x3FFE);
    }

    /// Helper to get channel count from assignment for testing.
    fn channel_count(ca: ChannelAssignment) -> u8 {
        match ca {
            ChannelAssignment::Independent(n) => n,
            ChannelAssignment::LeftSide
            | ChannelAssignment::RightSide
            | ChannelAssignment::MidSide => 2,
        }
    }

    #[test]
    fn test_channel_assignment() {
        assert_eq!(channel_count(ChannelAssignment::Independent(1)), 1);
        assert_eq!(channel_count(ChannelAssignment::Independent(2)), 2);
        assert_eq!(channel_count(ChannelAssignment::LeftSide), 2);
        assert_eq!(channel_count(ChannelAssignment::RightSide), 2);
        assert_eq!(channel_count(ChannelAssignment::MidSide), 2);
    }

    #[test]
    fn test_utf8_number_single_byte() {
        // 0 to 127 encoded as single byte
        assert_eq!(
            parse_utf8_number(&[0x00]).expect("operation should succeed"),
            (0, 1)
        );
        assert_eq!(
            parse_utf8_number(&[0x7F]).expect("operation should succeed"),
            (127, 1)
        );
        assert_eq!(
            parse_utf8_number(&[0x42]).expect("operation should succeed"),
            (66, 1)
        );
    }

    #[test]
    fn test_utf8_number_two_bytes() {
        // 128 = 0xC2 0x80
        assert_eq!(
            parse_utf8_number(&[0xC2, 0x80]).expect("operation should succeed"),
            (128, 2)
        );
        // 2047 = 0xDF 0xBF
        assert_eq!(
            parse_utf8_number(&[0xDF, 0xBF]).expect("operation should succeed"),
            (2047, 2)
        );
    }

    #[test]
    fn test_utf8_number_three_bytes() {
        // 2048 = 0xE0 0xA0 0x80
        assert_eq!(
            parse_utf8_number(&[0xE0, 0xA0, 0x80]).expect("operation should succeed"),
            (2048, 3)
        );
    }

    #[test]
    fn test_utf8_number_invalid_start() {
        // 10xxxxxx is invalid as start byte
        assert!(parse_utf8_number(&[0x80]).is_err());
        assert!(parse_utf8_number(&[0xBF]).is_err());
    }

    #[test]
    fn test_utf8_number_too_short() {
        // Two-byte sequence with missing continuation
        assert!(parse_utf8_number(&[0xC2]).is_err());
    }

    #[test]
    fn test_frame_header_parse() {
        // Construct a minimal valid frame header
        // Sync: 0xFFF8 (14 bits 0x3FFE + blocking strategy 0)
        // Block size code: 8 (256 samples)
        // Sample rate code: 9 (44100 Hz)
        // Channel: 1 (2 channels, independent)
        // Sample size: 4 (16 bits)
        // Reserved: 0
        let mut data = Vec::new();
        data.push(0xFF); // Sync high
        data.push(0xF8); // Sync low + reserved + blocking strategy
        data.push(0x89); // Block size 8 | sample rate 9
        data.push(0x18); // Channel 1 | sample size 4 | reserved 0
        data.push(0x00); // Frame number (UTF-8: 0)
        data.push(0x00); // CRC-8 (placeholder)

        let result = FrameHeader::parse(&data);
        assert!(result.is_ok());

        let (header, consumed) = result.expect("operation should succeed");
        assert!(!header.variable_blocksize);
        assert_eq!(header.block_size, 256);
        assert_eq!(header.sample_rate, Some(44_100));
        assert_eq!(header.channel_assignment, ChannelAssignment::Independent(2));
        assert_eq!(header.sample_size, Some(16));
        assert_eq!(header.number, 0);
        assert_eq!(consumed, 6);
    }

    #[test]
    fn test_frame_header_variable_blocksize() {
        // Sync with variable blocksize
        let mut data = Vec::new();
        data.push(0xFF);
        data.push(0xF9); // Variable blocksize
        data.push(0x89);
        data.push(0x18);
        data.push(0x00);
        data.push(0x00);

        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert!(header.variable_blocksize);
    }

    #[test]
    fn test_frame_header_invalid_sync() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_frame_header_reserved_bit() {
        // Reserved bit set in byte 1
        let data = [0xFF, 0xFA, 0x89, 0x18, 0x00, 0x00];
        assert!(FrameHeader::parse(&data).is_err());
    }

    #[test]
    fn test_frame_header_stereo_modes() {
        // Left/side
        let mut data = vec![0xFF, 0xF8, 0x89, 0x88, 0x00, 0x00]; // Channel 8
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channel_assignment, ChannelAssignment::LeftSide);
        assert!(header.uses_stereo_decorrelation());

        // Right/side
        data[3] = 0x98; // Channel 9
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channel_assignment, ChannelAssignment::RightSide);

        // Mid/side
        data[3] = 0xA8; // Channel 10
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channel_assignment, ChannelAssignment::MidSide);
    }

    #[test]
    fn test_frame_header_channels() {
        let mut data = vec![0xFF, 0xF8, 0x89, 0x18, 0x00, 0x00];

        // 1 channel (code 0)
        data[3] = 0x08;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channels(), 1);

        // 8 channels (code 7)
        data[3] = 0x78;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channels(), 8);

        // Left/side = 2 channels
        data[3] = 0x88;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.channels(), 2);
    }

    #[test]
    fn test_frame_header_block_sizes() {
        let mut data = vec![0xFF, 0xF8, 0x00, 0x18, 0x00, 0x00];

        // Code 1: 192 samples
        data[2] = 0x19;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 192);

        // Code 2: 576 samples
        data[2] = 0x29;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 576);

        // Code 3: 1152 samples
        data[2] = 0x39;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 1152);

        // Code 8: 256 samples
        data[2] = 0x89;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 256);

        // Code 9: 512 samples
        data[2] = 0x99;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 512);
    }

    #[test]
    fn test_frame_header_extra_block_size() {
        // Code 6: 8-bit block size - 1
        let data = vec![0xFF, 0xF8, 0x69, 0x18, 0x00, 0xFF, 0x00]; // 255+1 = 256
        let (header, consumed) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 256);
        assert_eq!(consumed, 7);

        // Code 7: 16-bit block size - 1
        let data = vec![0xFF, 0xF8, 0x79, 0x18, 0x00, 0x0F, 0xFF, 0x00]; // 4095+1 = 4096
        let (header, consumed) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.block_size, 4096);
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_frame_header_sample_rates() {
        let mut data = vec![0xFF, 0xF8, 0x80, 0x18, 0x00, 0x00];

        // Code 0: from STREAMINFO
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_rate, None);

        // Code 9: 44100 Hz
        data[2] = 0x89;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_rate, Some(44_100));

        // Code 10: 48000 Hz
        data[2] = 0x8A;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_rate, Some(48_000));
    }

    #[test]
    fn test_frame_header_sample_sizes() {
        let mut data = vec![0xFF, 0xF8, 0x89, 0x00, 0x00, 0x00];

        // Code 0: from STREAMINFO
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_size, None);

        // Code 1: 8 bits
        data[3] = 0x02;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_size, Some(8));

        // Code 4: 16 bits
        data[3] = 0x08;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_size, Some(16));

        // Code 6: 24 bits
        data[3] = 0x0C;
        let (header, _) = FrameHeader::parse(&data).expect("operation should succeed");
        assert_eq!(header.sample_size, Some(24));
    }
}
