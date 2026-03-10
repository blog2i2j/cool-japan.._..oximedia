//! FLAC metadata block parsing.
//!
//! This module handles parsing of FLAC metadata blocks including:
//!
//! - [`StreamInfo`] - Required stream parameters
//! - [`VorbisComment`] - Tags in Vorbis comment format
//! - [`MetadataBlock`] - Generic metadata block container
//!
//! # FLAC Metadata Structure
//!
//! Each metadata block consists of:
//! 1. A 1-byte header containing the block type and last-block flag
//! 2. A 3-byte length field
//! 3. Block-specific data
//!
//! The `STREAMINFO` block is always first and is required.

use oximedia_core::{OxiError, OxiResult};

/// Metadata block types as defined in the FLAC specification.
///
/// See <https://xiph.org/flac/format.html#metadata_block_header>
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    /// Stream info (required, must be first block).
    ///
    /// Contains essential stream parameters like sample rate,
    /// channel count, and bits per sample.
    StreamInfo,

    /// Padding block.
    ///
    /// Used to reserve space for future metadata without rewriting the file.
    Padding,

    /// Application-specific data.
    ///
    /// Contains data for use by third-party applications.
    Application,

    /// Seek table for sample-accurate seeking.
    ///
    /// Contains pre-computed seek points for efficient random access.
    SeekTable,

    /// Vorbis comment (tags).
    ///
    /// Contains metadata tags in the Vorbis comment format.
    VorbisComment,

    /// Cue sheet.
    ///
    /// Used for CD ripping applications to store track information.
    CueSheet,

    /// Picture (album art, etc.).
    ///
    /// Contains embedded images like album artwork.
    Picture,

    /// Reserved or invalid block type.
    Reserved,
}

impl From<u8> for BlockType {
    fn from(value: u8) -> Self {
        match value & 0x7F {
            0 => Self::StreamInfo,
            1 => Self::Padding,
            2 => Self::Application,
            3 => Self::SeekTable,
            4 => Self::VorbisComment,
            5 => Self::CueSheet,
            6 => Self::Picture,
            _ => Self::Reserved,
        }
    }
}

impl BlockType {
    /// Returns the numeric value of this block type.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::StreamInfo => 0,
            Self::Padding => 1,
            Self::Application => 2,
            Self::SeekTable => 3,
            Self::VorbisComment => 4,
            Self::CueSheet => 5,
            Self::Picture => 6,
            Self::Reserved => 127,
        }
    }

    /// Returns the name of this block type.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::StreamInfo => "STREAMINFO",
            Self::Padding => "PADDING",
            Self::Application => "APPLICATION",
            Self::SeekTable => "SEEKTABLE",
            Self::VorbisComment => "VORBIS_COMMENT",
            Self::CueSheet => "CUESHEET",
            Self::Picture => "PICTURE",
            Self::Reserved => "RESERVED",
        }
    }
}

/// A parsed metadata block.
///
/// Contains the block header information and raw block data.
#[derive(Clone, Debug)]
pub struct MetadataBlock {
    /// Indicates this is the last metadata block before audio frames.
    pub is_last: bool,

    /// The type of this metadata block.
    pub block_type: BlockType,

    /// Length of the block data in bytes (not including header).
    pub length: u32,

    /// Raw block data.
    pub data: Vec<u8>,
}

impl MetadataBlock {
    /// Parse a metadata block from bytes.
    ///
    /// Returns the parsed block and the total number of bytes consumed
    /// (including the 4-byte header).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input is too short (less than 4 bytes for header)
    /// - The input doesn't contain enough data for the block body
    pub fn parse(input: &[u8]) -> OxiResult<(Self, usize)> {
        if input.len() < 4 {
            return Err(OxiError::UnexpectedEof);
        }

        let is_last = input[0] & 0x80 != 0;
        let block_type = BlockType::from(input[0]);
        let length = u32::from_be_bytes([0, input[1], input[2], input[3]]);

        let header_size = 4;
        let total_size = header_size + length as usize;

        if input.len() < total_size {
            return Err(OxiError::UnexpectedEof);
        }

        let data = input[header_size..total_size].to_vec();

        Ok((
            Self {
                is_last,
                block_type,
                length,
                data,
            },
            total_size,
        ))
    }

    /// Returns true if this is the last metadata block.
    #[must_use]
    pub const fn is_last(&self) -> bool {
        self.is_last
    }

    /// Returns the size of the entire block including header.
    #[must_use]
    pub const fn total_size(&self) -> usize {
        4 + self.length as usize
    }
}

/// STREAMINFO metadata block.
///
/// Contains essential stream parameters required for decoding.
/// This block is mandatory and must be the first metadata block.
///
/// # Fields
///
/// All fields except MD5 are required for decoding:
///
/// - Block sizes define the frame structure
/// - Sample rate, channels, and bits per sample define the audio format
/// - Total samples is used for duration calculation
///
/// # Example
///
/// ```ignore
/// let info = StreamInfo::parse(&block_data)?;
/// println!("Sample rate: {} Hz", info.sample_rate);
/// println!("Channels: {}", info.channels);
/// println!("Bits per sample: {}", info.bits_per_sample);
/// if let Some(duration) = info.duration_seconds() {
///     println!("Duration: {:.2} seconds", duration);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct StreamInfo {
    /// Minimum block size in samples used in the stream.
    ///
    /// Valid range: 16-65535.
    pub min_block_size: u16,

    /// Maximum block size in samples used in the stream.
    ///
    /// Valid range: 16-65535.
    pub max_block_size: u16,

    /// Minimum frame size in bytes (0 = unknown).
    pub min_frame_size: u32,

    /// Maximum frame size in bytes (0 = unknown).
    pub max_frame_size: u32,

    /// Sample rate in Hz.
    ///
    /// Valid range: 1-655350 (though 0 is technically valid for unknown).
    pub sample_rate: u32,

    /// Number of audio channels (1-8).
    pub channels: u8,

    /// Bits per sample (4-32).
    pub bits_per_sample: u8,

    /// Total number of samples in the stream (0 = unknown).
    ///
    /// For stereo audio, this is per-channel sample count.
    pub total_samples: u64,

    /// MD5 signature of the unencoded audio data.
    ///
    /// Can be used to verify the audio data after decoding.
    pub md5: [u8; 16],
}

impl StreamInfo {
    /// Size of the STREAMINFO block data in bytes.
    pub const SIZE: usize = 34;

    /// Parse STREAMINFO from 34-byte block data.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not exactly 34 bytes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let info = StreamInfo::parse(&streaminfo_data)?;
    /// assert!(info.sample_rate > 0);
    /// assert!(info.channels >= 1 && info.channels <= 8);
    /// ```
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() != Self::SIZE {
            return Err(OxiError::Parse {
                offset: 0,
                message: format!(
                    "STREAMINFO must be {} bytes, got {}",
                    Self::SIZE,
                    data.len()
                ),
            });
        }

        let min_block_size = u16::from_be_bytes([data[0], data[1]]);
        let max_block_size = u16::from_be_bytes([data[2], data[3]]);
        let min_frame_size = u32::from_be_bytes([0, data[4], data[5], data[6]]);
        let max_frame_size = u32::from_be_bytes([0, data[7], data[8], data[9]]);

        // Bit layout for bytes 10-17:
        // Bits 0-19: sample rate (20 bits)
        // Bits 20-22: channels - 1 (3 bits)
        // Bits 23-27: bits per sample - 1 (5 bits)
        // Bits 28-63: total samples (36 bits)
        //
        // Byte 10: sample_rate[19:12]
        // Byte 11: sample_rate[11:4]
        // Byte 12: sample_rate[3:0] | (channels-1)[2:0] << 1 | (bps-1)[4]
        // Byte 13: (bps-1)[3:0] << 4 | total_samples[35:32]
        // Bytes 14-17: total_samples[31:0]

        let sample_rate =
            (u32::from(data[10]) << 12) | (u32::from(data[11]) << 4) | (u32::from(data[12]) >> 4);

        let channels = ((data[12] >> 1) & 0x07) + 1;
        let bits_per_sample = (((data[12] & 0x01) << 4) | (data[13] >> 4)) + 1;

        let total_samples = (u64::from(data[13] & 0x0F) << 32)
            | (u64::from(data[14]) << 24)
            | (u64::from(data[15]) << 16)
            | (u64::from(data[16]) << 8)
            | u64::from(data[17]);

        let mut md5 = [0u8; 16];
        md5.copy_from_slice(&data[18..34]);

        Ok(Self {
            min_block_size,
            max_block_size,
            min_frame_size,
            max_frame_size,
            sample_rate,
            channels,
            bits_per_sample,
            total_samples,
            md5,
        })
    }

    /// Get duration in seconds.
    ///
    /// Returns `None` if total samples or sample rate is unknown (zero).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> Option<f64> {
        if self.total_samples == 0 || self.sample_rate == 0 {
            None
        } else {
            Some(self.total_samples as f64 / f64::from(self.sample_rate))
        }
    }

    /// Returns the number of bytes per sample (per channel).
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        self.bits_per_sample.div_ceil(8)
    }

    /// Returns true if the MD5 checksum is present (non-zero).
    #[must_use]
    pub fn has_md5(&self) -> bool {
        self.md5 != [0u8; 16]
    }
}

/// Vorbis comment block (tags).
///
/// Contains metadata tags in the Vorbis comment format, which is also used
/// by Ogg Vorbis and Ogg Opus.
///
/// # Format
///
/// Vorbis comments use UTF-8 encoding and have the format:
/// - `ARTIST=Artist Name`
/// - `TITLE=Song Title`
/// - `ALBUM=Album Name`
///
/// Field names are case-insensitive but conventionally uppercase.
///
/// # Example
///
/// ```ignore
/// let comments = VorbisComment::parse(&block_data)?;
/// if let Some(title) = comments.get("TITLE") {
///     println!("Title: {}", title);
/// }
/// for (key, value) in &comments.comments {
///     println!("{}: {}", key, value);
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct VorbisComment {
    /// Vendor string identifying the encoder.
    pub vendor: String,

    /// List of comment entries as (key, value) pairs.
    ///
    /// Keys are stored in uppercase for case-insensitive lookup.
    pub comments: Vec<(String, String)>,
}

impl VorbisComment {
    /// Parse a Vorbis comment block.
    ///
    /// # Note
    ///
    /// Vorbis comments use little-endian byte order, unlike the rest of FLAC.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The data is too short
    /// - String lengths exceed available data
    pub fn parse(data: &[u8]) -> OxiResult<Self> {
        if data.len() < 8 {
            return Err(OxiError::UnexpectedEof);
        }

        let mut offset = 0;

        // Vendor string length (little-endian!)
        let vendor_len = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + vendor_len > data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        let vendor = String::from_utf8_lossy(&data[offset..offset + vendor_len]).into_owned();
        offset += vendor_len;

        if offset + 4 > data.len() {
            return Err(OxiError::UnexpectedEof);
        }

        // Comment count (little-endian)
        let comment_count = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;

        let mut comments = Vec::with_capacity(comment_count.min(1000));

        for _ in 0..comment_count {
            if offset + 4 > data.len() {
                break;
            }

            let comment_len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;

            if offset + comment_len > data.len() {
                break;
            }

            let comment = String::from_utf8_lossy(&data[offset..offset + comment_len]);
            offset += comment_len;

            if let Some((key, value)) = comment.split_once('=') {
                comments.push((key.to_uppercase(), value.to_string()));
            }
        }

        Ok(Self { vendor, comments })
    }

    /// Get a tag value by key (case-insensitive).
    ///
    /// Returns the first matching value if multiple entries have the same key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        let key_upper = key.to_uppercase();
        self.comments
            .iter()
            .find(|(k, _)| k == &key_upper)
            .map(|(_, v)| v.as_str())
    }

    /// Get all values for a key (case-insensitive).
    ///
    /// Vorbis comments allow multiple entries with the same key.
    #[must_use]
    pub fn get_all(&self, key: &str) -> Vec<&str> {
        let key_upper = key.to_uppercase();
        self.comments
            .iter()
            .filter(|(k, _)| k == &key_upper)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    /// Returns true if there are no comments.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.comments.is_empty()
    }

    /// Returns the number of comment entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.comments.len()
    }

    /// Returns an iterator over all comments.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.comments.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_type_from_u8() {
        assert_eq!(BlockType::from(0), BlockType::StreamInfo);
        assert_eq!(BlockType::from(1), BlockType::Padding);
        assert_eq!(BlockType::from(2), BlockType::Application);
        assert_eq!(BlockType::from(3), BlockType::SeekTable);
        assert_eq!(BlockType::from(4), BlockType::VorbisComment);
        assert_eq!(BlockType::from(5), BlockType::CueSheet);
        assert_eq!(BlockType::from(6), BlockType::Picture);
        assert_eq!(BlockType::from(7), BlockType::Reserved);
        assert_eq!(BlockType::from(127), BlockType::Reserved);
    }

    #[test]
    fn test_block_type_with_last_flag() {
        // When is_last flag is set, high bit is 1
        assert_eq!(BlockType::from(0x80), BlockType::StreamInfo);
        assert_eq!(BlockType::from(0x84), BlockType::VorbisComment);
    }

    #[test]
    fn test_block_type_as_u8() {
        assert_eq!(BlockType::StreamInfo.as_u8(), 0);
        assert_eq!(BlockType::VorbisComment.as_u8(), 4);
        assert_eq!(BlockType::Reserved.as_u8(), 127);
    }

    #[test]
    fn test_block_type_name() {
        assert_eq!(BlockType::StreamInfo.name(), "STREAMINFO");
        assert_eq!(BlockType::VorbisComment.name(), "VORBIS_COMMENT");
    }

    #[test]
    fn test_metadata_block_parse() {
        // Create a simple padding block
        let mut data = vec![0x01]; // type=1 (padding), is_last=false
        data.extend_from_slice(&[0x00, 0x00, 0x08]); // length=8
        data.extend_from_slice(&[0x00; 8]); // padding data

        let (block, consumed) = MetadataBlock::parse(&data).expect("operation should succeed");
        assert!(!block.is_last);
        assert_eq!(block.block_type, BlockType::Padding);
        assert_eq!(block.length, 8);
        assert_eq!(block.data.len(), 8);
        assert_eq!(consumed, 12); // 4 header + 8 data
    }

    #[test]
    fn test_metadata_block_is_last() {
        let data = vec![0x81, 0x00, 0x00, 0x00]; // is_last=true, type=1, length=0

        let (block, _) = MetadataBlock::parse(&data).expect("operation should succeed");
        assert!(block.is_last);
        assert!(block.is_last());
    }

    #[test]
    fn test_stream_info_parse() {
        let mut data = vec![0u8; 34];

        // min_block_size = 4096 (0x1000)
        data[0] = 0x10;
        data[1] = 0x00;

        // max_block_size = 4096 (0x1000)
        data[2] = 0x10;
        data[3] = 0x00;

        // min_frame_size = 0
        // max_frame_size = 0

        // sample_rate = 44100 = 0x0AC44
        // channels = 2 (stored as 1)
        // bits_per_sample = 16 (stored as 15)
        // Byte 10: 0x0A
        // Byte 11: 0xC4
        // Byte 12: 0x42 = 0100 0010 = sample_rate[3:0]=4, channels-1=1, bps-1[4]=0
        // Byte 13: 0xF0 = 1111 0000 = bps-1[3:0]=15, total_samples[35:32]=0
        data[10] = 0x0A;
        data[11] = 0xC4;
        data[12] = 0x42;
        data[13] = 0xF0;

        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert_eq!(info.min_block_size, 4096);
        assert_eq!(info.max_block_size, 4096);
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 16);
        assert_eq!(info.total_samples, 0);
    }

    #[test]
    fn test_stream_info_wrong_size() {
        let data = vec![0u8; 33]; // Too short
        assert!(StreamInfo::parse(&data).is_err());

        let data = vec![0u8; 35]; // Too long
        assert!(StreamInfo::parse(&data).is_err());
    }

    #[test]
    fn test_stream_info_duration() {
        let mut data = vec![0u8; 34];

        // 44100 Hz sample rate
        // 44100 = 0x0AC44 (20 bits)
        // Bytes 10-12 encode: sample_rate (20 bits) | (channels-1) (3 bits) | (bps-1)[4:4] (1 bit)
        // Byte 13 encodes: (bps-1)[3:0] (4 bits) | total_samples[35:32] (4 bits)
        data[10] = 0x0A; // sample_rate[19:12]
        data[11] = 0xC4; // sample_rate[11:4]
        data[12] = 0x42; // sample_rate[3:0]=4, (channels-1)=1, (bps-1)[4]=0
        data[13] = 0xF0; // (bps-1)[3:0]=15 (16 bits), total_samples[35:32]=0

        // total_samples = 441000 (10 seconds at 44100 Hz)
        // 441000 = 0x6BAA8
        // Since this fits in 32 bits, high 4 bits (in data[13]) are 0
        data[14] = 0x00;
        data[15] = 0x06;
        data[16] = 0xBA;
        data[17] = 0xA8;

        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert_eq!(info.sample_rate, 44100);
        assert_eq!(info.total_samples, 441000);
        let duration = info.duration_seconds().expect("operation should succeed");
        assert!((duration - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_stream_info_no_duration() {
        let mut data = vec![0u8; 34];
        data[10] = 0x0A;
        data[11] = 0xC4;
        data[12] = 0x42;
        data[13] = 0xF0;
        // total_samples = 0

        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert!(info.duration_seconds().is_none());
    }

    #[test]
    fn test_stream_info_bytes_per_sample() {
        let mut data = vec![0u8; 34];
        data[10] = 0x0A;
        data[11] = 0xC4;
        data[12] = 0x42;
        data[13] = 0xF0; // 16 bits

        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert_eq!(info.bytes_per_sample(), 2);
    }

    #[test]
    fn test_stream_info_has_md5() {
        let mut data = vec![0u8; 34];
        data[10] = 0x0A;
        data[11] = 0xC4;
        data[12] = 0x42;
        data[13] = 0xF0;

        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert!(!info.has_md5());

        // Set MD5
        data[18] = 0x01;
        let info = StreamInfo::parse(&data).expect("operation should succeed");
        assert!(info.has_md5());
    }

    #[test]
    fn test_vorbis_comment_parse() {
        let mut data = Vec::new();

        // Vendor string: "Test" (4 bytes)
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(b"Test");

        // 2 comments
        data.extend_from_slice(&2u32.to_le_bytes());

        // ARTIST=Test Artist
        let comment1 = b"ARTIST=Test Artist";
        data.extend_from_slice(&(comment1.len() as u32).to_le_bytes());
        data.extend_from_slice(comment1);

        // TITLE=Test Title
        let comment2 = b"TITLE=Test Title";
        data.extend_from_slice(&(comment2.len() as u32).to_le_bytes());
        data.extend_from_slice(comment2);

        let vc = VorbisComment::parse(&data).expect("operation should succeed");
        assert_eq!(vc.vendor, "Test");
        assert_eq!(vc.len(), 2);
        assert_eq!(vc.get("ARTIST"), Some("Test Artist"));
        assert_eq!(vc.get("TITLE"), Some("Test Title"));
        assert_eq!(vc.get("artist"), Some("Test Artist")); // case-insensitive
    }

    #[test]
    fn test_vorbis_comment_empty() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes()); // vendor length = 0
        data.extend_from_slice(&0u32.to_le_bytes()); // comment count = 0

        let vc = VorbisComment::parse(&data).expect("operation should succeed");
        assert!(vc.vendor.is_empty());
        assert!(vc.is_empty());
        assert_eq!(vc.len(), 0);
    }

    #[test]
    fn test_vorbis_comment_get_all() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());

        // Two GENRE entries
        let genre1 = b"GENRE=Rock";
        data.extend_from_slice(&(genre1.len() as u32).to_le_bytes());
        data.extend_from_slice(genre1);

        let genre2 = b"GENRE=Alternative";
        data.extend_from_slice(&(genre2.len() as u32).to_le_bytes());
        data.extend_from_slice(genre2);

        let vc = VorbisComment::parse(&data).expect("operation should succeed");
        let genres = vc.get_all("GENRE");
        assert_eq!(genres.len(), 2);
        assert!(genres.contains(&"Rock"));
        assert!(genres.contains(&"Alternative"));
    }

    #[test]
    fn test_vorbis_comment_iter() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        let comment = b"KEY=value";
        data.extend_from_slice(&(comment.len() as u32).to_le_bytes());
        data.extend_from_slice(comment);

        let vc = VorbisComment::parse(&data).expect("operation should succeed");
        let entries: Vec<_> = vc.iter().collect();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], ("KEY", "value"));
    }

    #[test]
    fn test_vorbis_comment_too_short() {
        let data = vec![0u8; 4]; // Only vendor length, no data
        assert!(VorbisComment::parse(&data).is_err());
    }
}
