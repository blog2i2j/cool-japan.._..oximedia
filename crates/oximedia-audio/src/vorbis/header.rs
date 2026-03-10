//! Vorbis header parsing.
//!
//! Vorbis streams begin with three header packets:
//! 1. **Identification header** - Basic stream parameters
//! 2. **Comment header** - Metadata (title, artist, etc.)
//! 3. **Setup header** - Codebooks, floors, residues, mappings, modes
//!
//! All headers start with a common header indicating packet type.

#![forbid(unsafe_code)]

use crate::AudioError;

/// Vorbis magic bytes.
pub const VORBIS_MAGIC: &[u8; 6] = b"vorbis";

/// Header packet types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderType {
    /// Identification header (type 1).
    Identification,
    /// Comment header (type 3).
    Comment,
    /// Setup header (type 5).
    Setup,
}

impl HeaderType {
    /// Get header type from packet type byte.
    #[must_use]
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(HeaderType::Identification),
            3 => Some(HeaderType::Comment),
            5 => Some(HeaderType::Setup),
            _ => None,
        }
    }

    /// Get packet type byte for this header.
    #[must_use]
    pub fn to_byte(self) -> u8 {
        match self {
            HeaderType::Identification => 1,
            HeaderType::Comment => 3,
            HeaderType::Setup => 5,
        }
    }
}

/// Common Vorbis header.
#[derive(Debug, Clone)]
pub struct VorbisHeader {
    /// Header type.
    pub header_type: HeaderType,
    /// Raw header data (after type and magic).
    #[allow(dead_code)]
    pub data: Vec<u8>,
}

impl VorbisHeader {
    /// Parse common header fields.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        if data.len() < 7 {
            return Err(AudioError::InvalidData("Header too short".into()));
        }

        let packet_type = data[0];
        if (packet_type & 1) == 0 {
            return Err(AudioError::InvalidData(
                "Not a header packet (even type)".into(),
            ));
        }

        if &data[1..7] != VORBIS_MAGIC {
            return Err(AudioError::InvalidData("Invalid Vorbis magic".into()));
        }

        let header_type = HeaderType::from_byte(packet_type).ok_or_else(|| {
            AudioError::InvalidData(format!("Unknown header type: {packet_type}"))
        })?;

        Ok(Self {
            header_type,
            data: data[7..].to_vec(),
        })
    }
}

/// Identification header.
///
/// Contains basic audio stream parameters.
#[derive(Debug, Clone)]
pub struct IdentificationHeader {
    /// Vorbis version (must be 0).
    pub vorbis_version: u32,
    /// Number of audio channels.
    pub audio_channels: u8,
    /// Audio sample rate in Hz.
    pub audio_sample_rate: u32,
    /// Maximum bitrate (0 = unset).
    pub bitrate_maximum: i32,
    /// Nominal bitrate (0 = unset).
    pub bitrate_nominal: i32,
    /// Minimum bitrate (0 = unset).
    pub bitrate_minimum: i32,
    /// Block size 0 (small block, power of 2).
    pub blocksize_0: u8,
    /// Block size 1 (large block, power of 2).
    pub blocksize_1: u8,
    /// Framing flag (must be 1).
    pub framing_flag: bool,
}

impl IdentificationHeader {
    /// Minimum valid block size (64 samples).
    pub const MIN_BLOCKSIZE: u8 = 6;
    /// Maximum valid block size (8192 samples).
    pub const MAX_BLOCKSIZE: u8 = 13;

    /// Parse identification header from packet data.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        // Must start with type 1 and "vorbis"
        if data.len() < 30 {
            return Err(AudioError::InvalidData(
                "Identification header too short".into(),
            ));
        }

        let header = VorbisHeader::parse(data)?;
        if header.header_type != HeaderType::Identification {
            return Err(AudioError::InvalidData(
                "Expected identification header".into(),
            ));
        }

        let d = &header.data;
        if d.len() < 23 {
            return Err(AudioError::InvalidData(
                "Identification header data too short".into(),
            ));
        }

        let vorbis_version = u32::from_le_bytes([d[0], d[1], d[2], d[3]]);
        if vorbis_version != 0 {
            return Err(AudioError::InvalidData(format!(
                "Unsupported Vorbis version: {vorbis_version}"
            )));
        }

        let audio_channels = d[4];
        if audio_channels == 0 {
            return Err(AudioError::InvalidData("Zero audio channels".into()));
        }

        let audio_sample_rate = u32::from_le_bytes([d[5], d[6], d[7], d[8]]);
        if audio_sample_rate == 0 {
            return Err(AudioError::InvalidData("Zero sample rate".into()));
        }

        let bitrate_maximum = i32::from_le_bytes([d[9], d[10], d[11], d[12]]);
        let bitrate_nominal = i32::from_le_bytes([d[13], d[14], d[15], d[16]]);
        let bitrate_minimum = i32::from_le_bytes([d[17], d[18], d[19], d[20]]);

        let blocksizes = d[21];
        let blocksize_0 = blocksizes & 0x0F;
        let blocksize_1 = (blocksizes >> 4) & 0x0F;

        // Validate block sizes
        if !(Self::MIN_BLOCKSIZE..=Self::MAX_BLOCKSIZE).contains(&blocksize_0) {
            return Err(AudioError::InvalidData(format!(
                "Invalid blocksize_0: {blocksize_0}"
            )));
        }
        if !(Self::MIN_BLOCKSIZE..=Self::MAX_BLOCKSIZE).contains(&blocksize_1) {
            return Err(AudioError::InvalidData(format!(
                "Invalid blocksize_1: {blocksize_1}"
            )));
        }
        if blocksize_0 > blocksize_1 {
            return Err(AudioError::InvalidData(
                "blocksize_0 must be <= blocksize_1".into(),
            ));
        }

        let framing_flag = (d[22] & 1) != 0;
        if !framing_flag {
            return Err(AudioError::InvalidData("Framing flag not set".into()));
        }

        Ok(Self {
            vorbis_version,
            audio_channels,
            audio_sample_rate,
            bitrate_maximum,
            bitrate_nominal,
            bitrate_minimum,
            blocksize_0,
            blocksize_1,
            framing_flag,
        })
    }

    /// Get block size in samples for `blocksize_0`.
    #[must_use]
    pub fn block_size_0_samples(&self) -> usize {
        1 << self.blocksize_0
    }

    /// Get block size in samples for `blocksize_1`.
    #[must_use]
    pub fn block_size_1_samples(&self) -> usize {
        1 << self.blocksize_1
    }
}

/// User comment (key-value pair).
#[derive(Debug, Clone)]
pub struct UserComment {
    /// Comment key (tag name, e.g., "ARTIST").
    pub key: String,
    /// Comment value.
    pub value: String,
}

impl UserComment {
    /// Parse a user comment from "KEY=VALUE" format.
    #[must_use]
    pub fn parse(comment: &str) -> Self {
        if let Some(pos) = comment.find('=') {
            let (key, value) = comment.split_at(pos);
            Self {
                key: key.to_uppercase(),
                value: value[1..].to_string(),
            }
        } else {
            Self {
                key: String::new(),
                value: comment.to_string(),
            }
        }
    }
}

/// Comment header.
///
/// Contains metadata about the stream.
#[derive(Debug, Clone, Default)]
pub struct CommentHeader {
    /// Vendor string (encoder identification).
    pub vendor: String,
    /// User comments (metadata tags).
    pub comments: Vec<UserComment>,
}

impl CommentHeader {
    /// Parse comment header from packet data.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        let header = VorbisHeader::parse(data)?;
        if header.header_type != HeaderType::Comment {
            return Err(AudioError::InvalidData("Expected comment header".into()));
        }

        let d = &header.data;
        if d.len() < 4 {
            return Err(AudioError::InvalidData(
                "Comment header data too short".into(),
            ));
        }

        let vendor_length = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as usize;
        if d.len() < 4 + vendor_length + 4 {
            return Err(AudioError::InvalidData("Vendor string truncated".into()));
        }

        let vendor = String::from_utf8_lossy(&d[4..4 + vendor_length]).to_string();

        let mut offset = 4 + vendor_length;
        let comment_count =
            u32::from_le_bytes([d[offset], d[offset + 1], d[offset + 2], d[offset + 3]]) as usize;
        offset += 4;

        let mut comments = Vec::with_capacity(comment_count);
        for _ in 0..comment_count {
            if offset + 4 > d.len() {
                break;
            }
            let comment_length =
                u32::from_le_bytes([d[offset], d[offset + 1], d[offset + 2], d[offset + 3]])
                    as usize;
            offset += 4;

            if offset + comment_length > d.len() {
                break;
            }
            let comment_str = String::from_utf8_lossy(&d[offset..offset + comment_length]);
            comments.push(UserComment::parse(&comment_str));
            offset += comment_length;
        }

        Ok(Self { vendor, comments })
    }

    /// Get a comment by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        let key_upper = key.to_uppercase();
        self.comments
            .iter()
            .find(|c| c.key == key_upper)
            .map(|c| c.value.as_str())
    }

    /// Get title metadata.
    #[must_use]
    pub fn title(&self) -> Option<&str> {
        self.get("TITLE")
    }

    /// Get artist metadata.
    #[must_use]
    pub fn artist(&self) -> Option<&str> {
        self.get("ARTIST")
    }

    /// Get album metadata.
    #[must_use]
    pub fn album(&self) -> Option<&str> {
        self.get("ALBUM")
    }
}

/// Mapping configuration.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Mapping {
    /// Mapping type (0 = standard).
    pub mapping_type: u8,
    /// Submaps.
    pub submaps: Vec<Submap>,
    /// Channel coupling.
    pub coupling_steps: Vec<CouplingStep>,
    /// Mux for each channel.
    pub channel_mux: Vec<u8>,
}

/// Submap configuration.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Submap {
    /// Floor number to use.
    pub floor: u8,
    /// Residue number to use.
    pub residue: u8,
}

/// Coupling step for stereo.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct CouplingStep {
    /// Magnitude channel.
    pub magnitude: u8,
    /// Angle channel.
    pub angle: u8,
}

/// Mode configuration.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Mode {
    /// Block flag (0 = small, 1 = large).
    pub block_flag: bool,
    /// Window type (always 0).
    pub window_type: u8,
    /// Transform type (always 0 = MDCT).
    pub transform_type: u8,
    /// Mapping number.
    pub mapping: u8,
}

impl Mode {
    /// Get block size for this mode.
    #[must_use]
    pub fn block_size(&self, id_header: &IdentificationHeader) -> usize {
        if self.block_flag {
            id_header.block_size_1_samples()
        } else {
            id_header.block_size_0_samples()
        }
    }
}

/// Residue configuration skeleton.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct Residue {
    /// Residue type (0, 1, or 2).
    pub residue_type: u8,
    /// Begin position.
    pub begin: u32,
    /// End position.
    pub end: u32,
    /// Partition size.
    pub partition_size: u32,
    /// Classifications.
    pub classifications: u8,
    /// Classbook number.
    pub classbook: u8,
}

/// Setup header.
///
/// Contains all decoding configuration: codebooks, floors, residues, mappings, modes.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct SetupHeader {
    /// Number of codebooks.
    pub codebook_count: u8,
    /// Time domain transforms (deprecated, must be 0).
    pub time_count: u8,
    /// Number of floors.
    pub floor_count: u8,
    /// Number of residues.
    pub residue_count: u8,
    /// Number of mappings.
    pub mapping_count: u8,
    /// Number of modes.
    pub mode_count: u8,
    /// Mode configurations.
    pub modes: Vec<Mode>,
    /// Mapping configurations.
    pub mappings: Vec<Mapping>,
    /// Residue configurations.
    pub residues: Vec<Residue>,
}

impl SetupHeader {
    /// Parse setup header skeleton from packet data.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse(data: &[u8]) -> Result<Self, AudioError> {
        let header = VorbisHeader::parse(data)?;
        if header.header_type != HeaderType::Setup {
            return Err(AudioError::InvalidData("Expected setup header".into()));
        }

        if header.data.is_empty() {
            return Err(AudioError::InvalidData("Setup header data empty".into()));
        }

        // Parse codebook count (first byte + 1)
        let codebook_count = header.data[0].wrapping_add(1);

        // Skeleton: we don't fully parse the complex setup header
        // A full implementation would parse:
        // - Codebooks (variable length, Huffman trees)
        // - Time domain transforms (deprecated)
        // - Floors (spectral envelope)
        // - Residues (spectral detail)
        // - Mappings (channel coupling)
        // - Modes (block size selection)

        Ok(Self {
            codebook_count,
            time_count: 0,
            floor_count: 0,
            residue_count: 0,
            mapping_count: 0,
            mode_count: 0,
            modes: Vec::new(),
            mappings: Vec::new(),
            residues: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_id_header() -> Vec<u8> {
        let mut data = vec![0x01]; // Type 1
        data.extend_from_slice(b"vorbis");
        data.extend_from_slice(&0u32.to_le_bytes()); // version
        data.push(2); // channels
        data.extend_from_slice(&44100u32.to_le_bytes()); // sample rate
        data.extend_from_slice(&0i32.to_le_bytes()); // max bitrate
        data.extend_from_slice(&128000i32.to_le_bytes()); // nominal bitrate
        data.extend_from_slice(&0i32.to_le_bytes()); // min bitrate
        data.push(0x88); // blocksizes (8 and 8 -> 256 and 256)
        data.push(0x01); // framing flag
        data
    }

    #[test]
    fn test_header_type() {
        assert_eq!(HeaderType::from_byte(1), Some(HeaderType::Identification));
        assert_eq!(HeaderType::from_byte(3), Some(HeaderType::Comment));
        assert_eq!(HeaderType::from_byte(5), Some(HeaderType::Setup));
        assert_eq!(HeaderType::from_byte(0), None);
        assert_eq!(HeaderType::from_byte(2), None);
    }

    #[test]
    fn test_header_type_to_byte() {
        assert_eq!(HeaderType::Identification.to_byte(), 1);
        assert_eq!(HeaderType::Comment.to_byte(), 3);
        assert_eq!(HeaderType::Setup.to_byte(), 5);
    }

    #[test]
    fn test_vorbis_header_parse() {
        let mut data = vec![0x01]; // Type 1
        data.extend_from_slice(b"vorbis");
        data.extend_from_slice(&[0x00, 0x00]);

        let header = VorbisHeader::parse(&data).expect("should succeed");
        assert_eq!(header.header_type, HeaderType::Identification);
    }

    #[test]
    fn test_vorbis_header_invalid_magic() {
        let mut data = vec![0x01]; // Type 1
        data.extend_from_slice(b"foobar");
        data.extend_from_slice(&[0x00, 0x00]);

        let result = VorbisHeader::parse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_identification_header_parse() {
        let data = make_id_header();
        let header = IdentificationHeader::parse(&data).expect("should succeed");

        assert_eq!(header.vorbis_version, 0);
        assert_eq!(header.audio_channels, 2);
        assert_eq!(header.audio_sample_rate, 44100);
        assert_eq!(header.bitrate_nominal, 128000);
        assert_eq!(header.blocksize_0, 8);
        assert_eq!(header.blocksize_1, 8);
        assert!(header.framing_flag);
    }

    #[test]
    fn test_identification_header_block_sizes() {
        let data = make_id_header();
        let header = IdentificationHeader::parse(&data).expect("should succeed");

        assert_eq!(header.block_size_0_samples(), 256);
        assert_eq!(header.block_size_1_samples(), 256);
    }

    #[test]
    fn test_user_comment_parse() {
        let comment = UserComment::parse("ARTIST=Test Artist");
        assert_eq!(comment.key, "ARTIST");
        assert_eq!(comment.value, "Test Artist");
    }

    #[test]
    fn test_user_comment_parse_no_equals() {
        let comment = UserComment::parse("No equals sign");
        assert_eq!(comment.key, "");
        assert_eq!(comment.value, "No equals sign");
    }

    #[test]
    fn test_mode_block_size() {
        let data = make_id_header();
        let id_header = IdentificationHeader::parse(&data).expect("should succeed");

        let mode_small = Mode {
            block_flag: false,
            ..Default::default()
        };
        let mode_large = Mode {
            block_flag: true,
            ..Default::default()
        };

        assert_eq!(mode_small.block_size(&id_header), 256);
        assert_eq!(mode_large.block_size(&id_header), 256);
    }

    #[test]
    fn test_setup_header_parse() {
        let mut data = vec![0x05]; // Type 5
        data.extend_from_slice(b"vorbis");
        data.push(0x00); // codebook_count - 1

        let header = SetupHeader::parse(&data).expect("should succeed");
        assert_eq!(header.codebook_count, 1);
    }
}
