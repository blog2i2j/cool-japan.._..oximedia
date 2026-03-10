//! RIFF chunk parsing.
//!
//! This module provides types and parsing for RIFF/WAV chunk structures.

use oximedia_core::{OxiError, OxiResult};

/// RIFF chunk header.
///
/// Every chunk in a RIFF file starts with a 4-byte identifier and a 4-byte size.
/// The size does not include the 8 bytes of the header itself.
#[derive(Clone, Debug)]
pub struct RiffChunk {
    /// Chunk ID (4 bytes ASCII).
    pub id: [u8; 4],
    /// Chunk size (not including header).
    pub size: u32,
}

impl RiffChunk {
    /// Size of a RIFF chunk header in bytes.
    pub const HEADER_SIZE: usize = 8;

    /// Parse chunk header from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if data is too short (less than 8 bytes).
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < Self::HEADER_SIZE {
            return Err(OxiError::UnexpectedEof);
        }

        let mut id = [0u8; 4];
        id.copy_from_slice(&data[0..4]);
        let size = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);

        Ok(Self { id, size })
    }

    /// Get chunk ID as a string slice.
    ///
    /// Returns "????" if the ID contains invalid UTF-8.
    #[must_use]
    pub fn id_str(&self) -> &str {
        std::str::from_utf8(&self.id).unwrap_or("????")
    }

    /// Check if this chunk has a specific ID.
    #[must_use]
    pub fn is(&self, id: &[u8; 4]) -> bool {
        &self.id == id
    }

    /// Returns the total size including header.
    #[must_use]
    pub fn total_size(&self) -> u64 {
        u64::from(self.size) + Self::HEADER_SIZE as u64
    }
}

/// WAV audio format codes.
///
/// These are the standard format codes from the WAV specification.
/// PCM (0x0001) is the most common for uncompressed audio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WavFormat {
    /// PCM (uncompressed integer samples).
    Pcm,
    /// IEEE floating-point samples.
    IeeeFloat,
    /// A-law companded.
    Alaw,
    /// mu-law companded.
    Mulaw,
    /// Extensible format (uses sub-format GUID).
    Extensible,
    /// Unknown or unsupported format.
    Unknown(u16),
}

impl From<u16> for WavFormat {
    fn from(value: u16) -> Self {
        match value {
            0x0001 => Self::Pcm,
            0x0003 => Self::IeeeFloat,
            0x0006 => Self::Alaw,
            0x0007 => Self::Mulaw,
            0xFFFE => Self::Extensible,
            other => Self::Unknown(other),
        }
    }
}

impl WavFormat {
    /// Check if format is PCM-compatible (uncompressed).
    #[must_use]
    pub fn is_pcm(&self) -> bool {
        matches!(self, Self::Pcm | Self::IeeeFloat)
    }

    /// Get the format code as a 16-bit value.
    #[must_use]
    pub fn code(&self) -> u16 {
        match self {
            Self::Pcm => 0x0001,
            Self::IeeeFloat => 0x0003,
            Self::Alaw => 0x0006,
            Self::Mulaw => 0x0007,
            Self::Extensible => 0xFFFE,
            Self::Unknown(code) => *code,
        }
    }
}

/// Format chunk (fmt ) data.
///
/// Contains audio format parameters from the fmt chunk.
#[derive(Clone, Debug)]
pub struct FmtChunk {
    /// Audio format code.
    pub format: WavFormat,
    /// Number of audio channels.
    pub channels: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Byte rate (`sample_rate` * `block_align`).
    pub byte_rate: u32,
    /// Block alignment (channels * `bits_per_sample` / 8).
    pub block_align: u16,
    /// Bits per sample (8, 16, 24, 32).
    pub bits_per_sample: u16,
    /// Extension data for `WAVE_FORMAT_EXTENSIBLE`.
    pub extension: Option<FmtExtension>,
}

/// Format extension for `WAVE_FORMAT_EXTENSIBLE`.
///
/// Used for multi-channel audio and formats with more than 16 bits per sample.
#[derive(Clone, Debug)]
pub struct FmtExtension {
    /// Valid bits per sample (may be less than container bits).
    pub valid_bits: u16,
    /// Channel mask indicating speaker positions.
    pub channel_mask: u32,
    /// Sub-format GUID (first 2 bytes indicate actual format).
    pub sub_format: [u8; 16],
}

impl FmtChunk {
    /// Minimum size of fmt chunk data (basic format).
    pub const MIN_SIZE: usize = 16;

    /// Size of fmt chunk data with extension.
    pub const EXTENDED_SIZE: usize = 40;

    /// Parse fmt chunk data.
    ///
    /// # Errors
    ///
    /// Returns error if data is too short or malformed.
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < Self::MIN_SIZE {
            return Err(OxiError::Parse {
                offset: 0,
                message: "fmt chunk too short".into(),
            });
        }

        let format = WavFormat::from(u16::from_le_bytes([data[0], data[1]]));
        let channels = u16::from_le_bytes([data[2], data[3]]);
        let sample_rate = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let byte_rate = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        let block_align = u16::from_le_bytes([data[12], data[13]]);
        let bits_per_sample = u16::from_le_bytes([data[14], data[15]]);

        let extension = if format == WavFormat::Extensible && data.len() >= Self::EXTENDED_SIZE {
            let cb_size = u16::from_le_bytes([data[16], data[17]]);
            if cb_size >= 22 {
                let valid_bits = u16::from_le_bytes([data[18], data[19]]);
                let channel_mask = u32::from_le_bytes([data[20], data[21], data[22], data[23]]);
                let mut sub_format = [0u8; 16];
                sub_format.copy_from_slice(&data[24..40]);
                Some(FmtExtension {
                    valid_bits,
                    channel_mask,
                    sub_format,
                })
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            extension,
        })
    }

    /// Get actual bits per sample (considering extension).
    ///
    /// For extensible format, returns `valid_bits` from extension if available.
    #[must_use]
    pub fn actual_bits_per_sample(&self) -> u16 {
        self.extension
            .as_ref()
            .map_or(self.bits_per_sample, |ext| ext.valid_bits)
    }

    /// Check if this is float format.
    ///
    /// Returns true for IEEE float format or extensible with float sub-format.
    #[must_use]
    pub fn is_float(&self) -> bool {
        match &self.format {
            WavFormat::IeeeFloat => true,
            WavFormat::Extensible => {
                // Check sub-format GUID for IEEE float
                // KSDATAFORMAT_SUBTYPE_IEEE_FLOAT: 00000003-0000-0010-8000-00aa00389b71
                if let Some(ext) = &self.extension {
                    ext.sub_format[0..2] == [0x03, 0x00]
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if this is integer PCM format.
    #[must_use]
    pub fn is_integer_pcm(&self) -> bool {
        match &self.format {
            WavFormat::Pcm => true,
            WavFormat::Extensible => {
                // Check sub-format GUID for PCM
                // KSDATAFORMAT_SUBTYPE_PCM: 00000001-0000-0010-8000-00aa00389b71
                if let Some(ext) = &self.extension {
                    ext.sub_format[0..2] == [0x01, 0x00]
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

/// Channel mask bits from Windows mmreg.h.
///
/// These constants define speaker positions for multi-channel audio.
#[allow(dead_code)]
pub mod channel_mask {
    /// Front left speaker.
    pub const FRONT_LEFT: u32 = 0x0000_0001;
    /// Front right speaker.
    pub const FRONT_RIGHT: u32 = 0x0000_0002;
    /// Front center speaker.
    pub const FRONT_CENTER: u32 = 0x0000_0004;
    /// Low frequency (subwoofer).
    pub const LOW_FREQUENCY: u32 = 0x0000_0008;
    /// Back left speaker.
    pub const BACK_LEFT: u32 = 0x0000_0010;
    /// Back right speaker.
    pub const BACK_RIGHT: u32 = 0x0000_0020;
    /// Front left of center speaker.
    pub const FRONT_LEFT_OF_CENTER: u32 = 0x0000_0040;
    /// Front right of center speaker.
    pub const FRONT_RIGHT_OF_CENTER: u32 = 0x0000_0080;
    /// Back center speaker.
    pub const BACK_CENTER: u32 = 0x0000_0100;
    /// Side left speaker.
    pub const SIDE_LEFT: u32 = 0x0000_0200;
    /// Side right speaker.
    pub const SIDE_RIGHT: u32 = 0x0000_0400;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riff_chunk_parse() {
        let data = b"RIFF\x10\x00\x00\x00WAVE";
        let chunk = RiffChunk::parse(data).expect("operation should succeed");
        assert_eq!(chunk.id_str(), "RIFF");
        assert_eq!(chunk.size, 16);
        assert!(chunk.is(b"RIFF"));
        assert!(!chunk.is(b"WAVE"));
    }

    #[test]
    fn test_riff_chunk_too_short() {
        let data = b"RIFF";
        assert!(RiffChunk::parse(data).is_err());
    }

    #[test]
    fn test_riff_chunk_total_size() {
        let data = b"fmt \x10\x00\x00\x00";
        let chunk = RiffChunk::parse(data).expect("operation should succeed");
        assert_eq!(chunk.total_size(), 24);
    }

    #[test]
    fn test_fmt_chunk_pcm_16bit_stereo() {
        // 16-bit stereo 44.1kHz PCM
        let data = [
            0x01, 0x00, // format = PCM
            0x02, 0x00, // channels = 2
            0x44, 0xAC, 0x00, 0x00, // sample_rate = 44100
            0x10, 0xB1, 0x02, 0x00, // byte_rate = 176400
            0x04, 0x00, // block_align = 4
            0x10, 0x00, // bits_per_sample = 16
        ];

        let fmt = FmtChunk::parse(&data).expect("operation should succeed");
        assert_eq!(fmt.format, WavFormat::Pcm);
        assert_eq!(fmt.channels, 2);
        assert_eq!(fmt.sample_rate, 44_100);
        assert_eq!(fmt.byte_rate, 176_400);
        assert_eq!(fmt.block_align, 4);
        assert_eq!(fmt.bits_per_sample, 16);
        assert!(fmt.is_integer_pcm());
        assert!(!fmt.is_float());
    }

    #[test]
    fn test_fmt_chunk_float_mono() {
        // 32-bit float mono 48kHz
        let data = [
            0x03, 0x00, // format = IEEE Float
            0x01, 0x00, // channels = 1
            0x80, 0xBB, 0x00, 0x00, // sample_rate = 48000
            0x00, 0xEE, 0x02, 0x00, // byte_rate = 192000
            0x04, 0x00, // block_align = 4
            0x20, 0x00, // bits_per_sample = 32
        ];

        let fmt = FmtChunk::parse(&data).expect("operation should succeed");
        assert_eq!(fmt.format, WavFormat::IeeeFloat);
        assert_eq!(fmt.channels, 1);
        assert_eq!(fmt.sample_rate, 48_000);
        assert!(fmt.is_float());
        assert!(!fmt.is_integer_pcm());
    }

    #[test]
    fn test_fmt_chunk_too_short() {
        let data = [0x01, 0x00, 0x02, 0x00];
        assert!(FmtChunk::parse(&data).is_err());
    }

    #[test]
    fn test_wav_format_conversion() {
        assert_eq!(WavFormat::from(1), WavFormat::Pcm);
        assert_eq!(WavFormat::from(3), WavFormat::IeeeFloat);
        assert_eq!(WavFormat::from(6), WavFormat::Alaw);
        assert_eq!(WavFormat::from(7), WavFormat::Mulaw);
        assert_eq!(WavFormat::from(0xFFFE), WavFormat::Extensible);
        assert!(matches!(WavFormat::from(99), WavFormat::Unknown(99)));
    }

    #[test]
    fn test_wav_format_is_pcm() {
        assert!(WavFormat::Pcm.is_pcm());
        assert!(WavFormat::IeeeFloat.is_pcm());
        assert!(!WavFormat::Alaw.is_pcm());
        assert!(!WavFormat::Mulaw.is_pcm());
    }

    #[test]
    fn test_wav_format_code() {
        assert_eq!(WavFormat::Pcm.code(), 0x0001);
        assert_eq!(WavFormat::IeeeFloat.code(), 0x0003);
        assert_eq!(WavFormat::Unknown(0x55).code(), 0x55);
    }

    #[test]
    fn test_actual_bits_per_sample() {
        // Without extension
        let data = [
            0x01, 0x00, 0x02, 0x00, 0x44, 0xAC, 0x00, 0x00, 0x10, 0xB1, 0x02, 0x00, 0x04, 0x00,
            0x18, 0x00, // 24 bits
        ];
        let fmt = FmtChunk::parse(&data).expect("operation should succeed");
        assert_eq!(fmt.actual_bits_per_sample(), 24);
    }
}
