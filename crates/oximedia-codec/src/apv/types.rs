//! APV (Advanced Professional Video) types.
//!
//! Configuration, error types, and bitstream structures for the APV codec
//! (ISO/IEC 23009-13). APV is a royalty-free, intra-frame professional codec
//! using 8x8 DCT with per-band quantization and tile-based parallelism.

use thiserror::Error;

/// APV access unit magic bytes identifying an APV bitstream.
pub const APV_MAGIC: &[u8; 4] = b"APV1";

/// Size of the APV access unit header in bytes.
///
/// Layout (16 bytes total):
/// - Offset 0..4:  Magic `"APV1"` (4 bytes)
/// - Offset 4:     Profile (1 byte)
/// - Offset 5..7:  Width (2 bytes, big-endian)
/// - Offset 7..9:  Height (2 bytes, big-endian)
/// - Offset 9:     Bit depth code (1 byte: 0=8, 1=10, 2=12)
/// - Offset 10:    Chroma format code (1 byte: 0=420, 1=422, 2=444)
/// - Offset 11:    QP (1 byte, 0..63)
/// - Offset 12..14: Tile columns (2 bytes, big-endian)
/// - Offset 14..16: Tile rows (2 bytes, big-endian)
pub const APV_HEADER_SIZE: usize = 16;

/// Maximum supported APV frame dimension (width or height).
pub const APV_MAX_DIMENSION: u32 = 16384;

/// Maximum quantization parameter value.
pub const APV_MAX_QP: u8 = 63;

/// APV profile.
///
/// APV defines three profiles with increasing feature sets:
/// - Simple: baseline intra-frame coding (8x8 DCT, exp-Golomb entropy)
/// - Main: adds advanced quantization and extended bit depth
/// - High: adds 4:4:4 chroma and highest bit depths
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApvProfile {
    /// Simple profile — baseline APV-S.
    Simple = 0,
    /// Main profile — APV-M.
    Main = 1,
    /// High profile — APV-H.
    High = 2,
}

impl ApvProfile {
    /// Decode profile from the single-byte wire representation.
    pub fn from_byte(b: u8) -> Result<Self, ApvError> {
        match b {
            0 => Ok(Self::Simple),
            1 => Ok(Self::Main),
            2 => Ok(Self::High),
            _ => Err(ApvError::UnsupportedProfile),
        }
    }

    /// Encode profile to a single byte for the wire format.
    #[must_use]
    pub fn to_byte(self) -> u8 {
        self as u8
    }

    /// Human-readable name of the profile.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Simple => "APV-S",
            Self::Main => "APV-M",
            Self::High => "APV-H",
        }
    }
}

/// Chroma sub-sampling format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApvChromaFormat {
    /// 4:2:0 — chroma at half horizontal and half vertical resolution.
    Yuv420 = 0,
    /// 4:2:2 — chroma at half horizontal resolution.
    Yuv422 = 1,
    /// 4:4:4 — no chroma sub-sampling.
    Yuv444 = 2,
}

impl ApvChromaFormat {
    /// Decode from wire byte.
    pub fn from_byte(b: u8) -> Result<Self, ApvError> {
        match b {
            0 => Ok(Self::Yuv420),
            1 => Ok(Self::Yuv422),
            2 => Ok(Self::Yuv444),
            _ => Err(ApvError::InvalidBitstream(format!(
                "unknown chroma format code {b}"
            ))),
        }
    }

    /// Encode to wire byte.
    #[must_use]
    pub fn to_byte(self) -> u8 {
        self as u8
    }

    /// Number of chroma planes (always 2 for Cb and Cr).
    #[must_use]
    pub fn chroma_planes(self) -> usize {
        2
    }

    /// Horizontal sub-sampling factor for chroma planes.
    #[must_use]
    pub fn chroma_h_shift(self) -> u32 {
        match self {
            Self::Yuv420 | Self::Yuv422 => 1,
            Self::Yuv444 => 0,
        }
    }

    /// Vertical sub-sampling factor for chroma planes.
    #[must_use]
    pub fn chroma_v_shift(self) -> u32 {
        match self {
            Self::Yuv420 => 1,
            Self::Yuv422 | Self::Yuv444 => 0,
        }
    }
}

/// Bit depth of samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApvBitDepth {
    /// 8 bits per sample.
    Eight = 0,
    /// 10 bits per sample.
    Ten = 1,
    /// 12 bits per sample.
    Twelve = 2,
}

impl ApvBitDepth {
    /// Number of bits per sample.
    #[must_use]
    pub fn bits(self) -> u8 {
        match self {
            Self::Eight => 8,
            Self::Ten => 10,
            Self::Twelve => 12,
        }
    }

    /// Maximum sample value for this bit depth.
    #[must_use]
    pub fn max_value(self) -> u16 {
        (1u16 << self.bits()) - 1
    }

    /// Decode from wire byte.
    pub fn from_byte(b: u8) -> Result<Self, ApvError> {
        match b {
            0 => Ok(Self::Eight),
            1 => Ok(Self::Ten),
            2 => Ok(Self::Twelve),
            _ => Err(ApvError::InvalidBitstream(format!(
                "unknown bit depth code {b}"
            ))),
        }
    }

    /// Encode to wire byte.
    #[must_use]
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// APV encoder / decoder configuration.
#[derive(Clone, Debug)]
pub struct ApvConfig {
    /// Frame width in pixels (must be > 0, ≤ 16384).
    pub width: u32,
    /// Frame height in pixels (must be > 0, ≤ 16384).
    pub height: u32,
    /// APV profile.
    pub profile: ApvProfile,
    /// Sample bit depth.
    pub bit_depth: ApvBitDepth,
    /// Chroma sub-sampling format.
    pub chroma_format: ApvChromaFormat,
    /// Quantization parameter (0–63, lower = higher quality).
    /// Default: 22.
    pub qp: u8,
    /// Number of tile columns (≥ 1). Default: 1.
    pub tile_cols: u16,
    /// Number of tile rows (≥ 1). Default: 1.
    pub tile_rows: u16,
}

impl ApvConfig {
    /// Create a new APV configuration with sensible defaults.
    ///
    /// # Errors
    ///
    /// Returns `ApvError::InvalidDimensions` if width or height is zero
    /// or exceeds `APV_MAX_DIMENSION`.
    pub fn new(width: u32, height: u32) -> Result<Self, ApvError> {
        if width == 0 || height == 0 {
            return Err(ApvError::InvalidDimensions {
                width,
                height,
                reason: "width and height must be non-zero".to_string(),
            });
        }
        if width > APV_MAX_DIMENSION || height > APV_MAX_DIMENSION {
            return Err(ApvError::InvalidDimensions {
                width,
                height,
                reason: format!("exceeds maximum dimension {APV_MAX_DIMENSION}"),
            });
        }
        Ok(Self {
            width,
            height,
            profile: ApvProfile::Simple,
            bit_depth: ApvBitDepth::Eight,
            chroma_format: ApvChromaFormat::Yuv420,
            qp: 22,
            tile_cols: 1,
            tile_rows: 1,
        })
    }

    /// Set the APV profile.
    #[must_use]
    pub fn with_profile(mut self, profile: ApvProfile) -> Self {
        self.profile = profile;
        self
    }

    /// Set the bit depth.
    #[must_use]
    pub fn with_bit_depth(mut self, bit_depth: ApvBitDepth) -> Self {
        self.bit_depth = bit_depth;
        self
    }

    /// Set the chroma format.
    #[must_use]
    pub fn with_chroma_format(mut self, chroma_format: ApvChromaFormat) -> Self {
        self.chroma_format = chroma_format;
        self
    }

    /// Set the quantization parameter (clamped to 0–63).
    #[must_use]
    pub fn with_qp(mut self, qp: u8) -> Self {
        self.qp = qp.min(APV_MAX_QP);
        self
    }

    /// Set tile grid dimensions.
    ///
    /// # Errors
    ///
    /// Returns `ApvError::InvalidDimensions` if cols or rows is zero.
    pub fn with_tiles(mut self, cols: u16, rows: u16) -> Result<Self, ApvError> {
        if cols == 0 || rows == 0 {
            return Err(ApvError::InvalidDimensions {
                width: cols as u32,
                height: rows as u32,
                reason: "tile_cols and tile_rows must be ≥ 1".to_string(),
            });
        }
        self.tile_cols = cols;
        self.tile_rows = rows;
        Ok(self)
    }

    /// Validate the entire configuration.
    ///
    /// # Errors
    ///
    /// Returns an appropriate `ApvError` if any parameter is out of range.
    pub fn validate(&self) -> Result<(), ApvError> {
        if self.width == 0 || self.height == 0 {
            return Err(ApvError::InvalidDimensions {
                width: self.width,
                height: self.height,
                reason: "width and height must be non-zero".to_string(),
            });
        }
        if self.width > APV_MAX_DIMENSION || self.height > APV_MAX_DIMENSION {
            return Err(ApvError::InvalidDimensions {
                width: self.width,
                height: self.height,
                reason: format!("exceeds maximum dimension {APV_MAX_DIMENSION}"),
            });
        }
        if self.qp > APV_MAX_QP {
            return Err(ApvError::InvalidQp(self.qp));
        }
        if self.tile_cols == 0 || self.tile_rows == 0 {
            return Err(ApvError::InvalidDimensions {
                width: self.tile_cols as u32,
                height: self.tile_rows as u32,
                reason: "tile_cols and tile_rows must be ≥ 1".to_string(),
            });
        }
        Ok(())
    }

    /// Compute the width of the tile at column `col_idx`.
    ///
    /// The last column absorbs any remainder pixels.
    #[must_use]
    pub fn tile_width(&self, col_idx: u16) -> u32 {
        let base = self.width / self.tile_cols as u32;
        let remainder = self.width % self.tile_cols as u32;
        if col_idx < self.tile_cols - 1 {
            base
        } else {
            base + remainder
        }
    }

    /// Compute the height of the tile at row `row_idx`.
    ///
    /// The last row absorbs any remainder pixels.
    #[must_use]
    pub fn tile_height(&self, row_idx: u16) -> u32 {
        let base = self.height / self.tile_rows as u32;
        let remainder = self.height % self.tile_rows as u32;
        if row_idx < self.tile_rows - 1 {
            base
        } else {
            base + remainder
        }
    }

    /// X-offset of tile column `col_idx` in pixels.
    #[must_use]
    pub fn tile_x_offset(&self, col_idx: u16) -> u32 {
        let base = self.width / self.tile_cols as u32;
        base * col_idx as u32
    }

    /// Y-offset of tile row `row_idx` in pixels.
    #[must_use]
    pub fn tile_y_offset(&self, row_idx: u16) -> u32 {
        let base = self.height / self.tile_rows as u32;
        base * row_idx as u32
    }
}

impl Default for ApvConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            profile: ApvProfile::Simple,
            bit_depth: ApvBitDepth::Eight,
            chroma_format: ApvChromaFormat::Yuv420,
            qp: 22,
            tile_cols: 1,
            tile_rows: 1,
        }
    }
}

/// APV access unit header as read/written in the bitstream.
#[derive(Clone, Debug)]
pub struct ApvFrameHeader {
    /// APV profile.
    pub profile: ApvProfile,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Sample bit depth.
    pub bit_depth: ApvBitDepth,
    /// Chroma sub-sampling format.
    pub chroma_format: ApvChromaFormat,
    /// Quantization parameter used for this frame.
    pub qp: u8,
    /// Number of tile columns.
    pub tile_cols: u16,
    /// Number of tile rows.
    pub tile_rows: u16,
}

impl ApvFrameHeader {
    /// Serialize the header into a 16-byte buffer.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; APV_HEADER_SIZE] {
        let mut buf = [0u8; APV_HEADER_SIZE];
        buf[0..4].copy_from_slice(APV_MAGIC);
        buf[4] = self.profile.to_byte();
        buf[5..7].copy_from_slice(&(self.width as u16).to_be_bytes());
        buf[7..9].copy_from_slice(&(self.height as u16).to_be_bytes());
        buf[9] = self.bit_depth.to_byte();
        buf[10] = self.chroma_format.to_byte();
        buf[11] = self.qp;
        buf[12..14].copy_from_slice(&self.tile_cols.to_be_bytes());
        buf[14..16].copy_from_slice(&self.tile_rows.to_be_bytes());
        buf
    }

    /// Deserialize from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns `ApvError::InvalidBitstream` if the data is too short or
    /// magic bytes do not match.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ApvError> {
        if data.len() < APV_HEADER_SIZE {
            return Err(ApvError::InvalidBitstream(format!(
                "header too short: {} bytes (need {})",
                data.len(),
                APV_HEADER_SIZE
            )));
        }
        if &data[0..4] != APV_MAGIC.as_slice() {
            return Err(ApvError::InvalidBitstream(
                "missing APV1 magic bytes".to_string(),
            ));
        }

        let profile = ApvProfile::from_byte(data[4])?;
        let width = u16::from_be_bytes([data[5], data[6]]) as u32;
        let height = u16::from_be_bytes([data[7], data[8]]) as u32;
        let bit_depth = ApvBitDepth::from_byte(data[9])?;
        let chroma_format = ApvChromaFormat::from_byte(data[10])?;
        let qp = data[11];
        let tile_cols = u16::from_be_bytes([data[12], data[13]]);
        let tile_rows = u16::from_be_bytes([data[14], data[15]]);

        if width == 0 || height == 0 {
            return Err(ApvError::InvalidBitstream(
                "frame dimensions must be non-zero".to_string(),
            ));
        }
        if qp > APV_MAX_QP {
            return Err(ApvError::InvalidQp(qp));
        }
        if tile_cols == 0 || tile_rows == 0 {
            return Err(ApvError::InvalidBitstream(
                "tile_cols and tile_rows must be ≥ 1".to_string(),
            ));
        }

        Ok(Self {
            profile,
            width,
            height,
            bit_depth,
            chroma_format,
            qp,
            tile_cols,
            tile_rows,
        })
    }

    /// Build a header from the encoder config.
    #[must_use]
    pub fn from_config(config: &ApvConfig) -> Self {
        Self {
            profile: config.profile,
            width: config.width,
            height: config.height,
            bit_depth: config.bit_depth,
            chroma_format: config.chroma_format,
            qp: config.qp,
            tile_cols: config.tile_cols,
            tile_rows: config.tile_rows,
        }
    }
}

/// Metadata about a single tile within an APV access unit.
#[derive(Clone, Debug)]
pub struct ApvTileInfo {
    /// Column index (0-based).
    pub col: u16,
    /// Row index (0-based).
    pub row: u16,
    /// Byte offset of tile data within the access unit (after header).
    pub offset: usize,
    /// Byte length of tile data.
    pub size: usize,
    /// Tile width in pixels.
    pub width: u32,
    /// Tile height in pixels.
    pub height: u32,
}

/// APV-specific errors.
#[derive(Debug, Error)]
pub enum ApvError {
    /// Invalid frame dimensions.
    #[error("APV invalid dimensions {width}x{height}: {reason}")]
    InvalidDimensions {
        /// Width.
        width: u32,
        /// Height.
        height: u32,
        /// Reason.
        reason: String,
    },

    /// Quantization parameter out of range.
    #[error("APV invalid QP {0} (must be 0–63)")]
    InvalidQp(u8),

    /// Encoding failed.
    #[error("APV encoding failed: {0}")]
    EncodingFailed(String),

    /// Decoding failed.
    #[error("APV decoding failed: {0}")]
    DecodingFailed(String),

    /// Invalid bitstream data.
    #[error("APV invalid bitstream: {0}")]
    InvalidBitstream(String),

    /// Profile not supported.
    #[error("APV unsupported profile")]
    UnsupportedProfile,

    /// Output buffer too small.
    #[error("APV buffer too small")]
    BufferTooSmall,
}

impl From<ApvError> for crate::error::CodecError {
    fn from(e: ApvError) -> Self {
        match e {
            ApvError::InvalidDimensions {
                width,
                height,
                reason,
            } => crate::error::CodecError::InvalidParameter(format!(
                "APV dimensions {width}x{height}: {reason}"
            )),
            ApvError::InvalidQp(qp) => {
                crate::error::CodecError::InvalidParameter(format!("APV QP {qp} out of range"))
            }
            ApvError::EncodingFailed(msg) => crate::error::CodecError::Internal(msg),
            ApvError::DecodingFailed(msg) => crate::error::CodecError::DecoderError(msg),
            ApvError::InvalidBitstream(msg) => crate::error::CodecError::InvalidBitstream(msg),
            ApvError::UnsupportedProfile => crate::error::CodecError::UnsupportedFeature(
                "APV profile not supported".to_string(),
            ),
            ApvError::BufferTooSmall => {
                crate::error::CodecError::BufferTooSmall { needed: 0, have: 0 }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_new_valid() {
        let config = ApvConfig::new(1920, 1080);
        assert!(config.is_ok());
        let c = config.expect("valid config");
        assert_eq!(c.width, 1920);
        assert_eq!(c.height, 1080);
        assert_eq!(c.qp, 22);
        assert_eq!(c.profile, ApvProfile::Simple);
        assert_eq!(c.bit_depth, ApvBitDepth::Eight);
        assert_eq!(c.chroma_format, ApvChromaFormat::Yuv420);
        assert_eq!(c.tile_cols, 1);
        assert_eq!(c.tile_rows, 1);
    }

    #[test]
    fn test_config_zero_width() {
        let result = ApvConfig::new(0, 480);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_zero_height() {
        let result = ApvConfig::new(640, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_exceeds_max() {
        let result = ApvConfig::new(20000, 1080);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_qp() {
        let config = ApvConfig::new(320, 240).expect("valid").with_qp(50);
        assert_eq!(config.qp, 50);
    }

    #[test]
    fn test_config_qp_clamped() {
        let config = ApvConfig::new(320, 240).expect("valid").with_qp(100);
        assert_eq!(config.qp, APV_MAX_QP);
    }

    #[test]
    fn test_config_validate_invalid_qp() {
        let mut config = ApvConfig::default();
        config.qp = 64;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_tiles() {
        let config = ApvConfig::new(640, 480)
            .expect("valid")
            .with_tiles(2, 2)
            .expect("valid tiles");
        assert_eq!(config.tile_cols, 2);
        assert_eq!(config.tile_rows, 2);
        assert_eq!(config.tile_width(0), 320);
        assert_eq!(config.tile_width(1), 320);
        assert_eq!(config.tile_height(0), 240);
        assert_eq!(config.tile_height(1), 240);
    }

    #[test]
    fn test_config_tiles_remainder() {
        let config = ApvConfig::new(641, 481)
            .expect("valid")
            .with_tiles(2, 2)
            .expect("valid tiles");
        assert_eq!(config.tile_width(0), 320);
        assert_eq!(config.tile_width(1), 321);
        assert_eq!(config.tile_height(0), 240);
        assert_eq!(config.tile_height(1), 241);
    }

    #[test]
    fn test_config_zero_tiles() {
        let result = ApvConfig::new(640, 480).expect("valid").with_tiles(0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_profile_roundtrip() {
        for profile in [ApvProfile::Simple, ApvProfile::Main, ApvProfile::High] {
            let byte = profile.to_byte();
            let decoded = ApvProfile::from_byte(byte).expect("valid byte");
            assert_eq!(decoded, profile);
        }
    }

    #[test]
    fn test_profile_invalid_byte() {
        assert!(ApvProfile::from_byte(3).is_err());
    }

    #[test]
    fn test_profile_name() {
        assert_eq!(ApvProfile::Simple.name(), "APV-S");
        assert_eq!(ApvProfile::Main.name(), "APV-M");
        assert_eq!(ApvProfile::High.name(), "APV-H");
    }

    #[test]
    fn test_chroma_format_roundtrip() {
        for fmt in [
            ApvChromaFormat::Yuv420,
            ApvChromaFormat::Yuv422,
            ApvChromaFormat::Yuv444,
        ] {
            let byte = fmt.to_byte();
            let decoded = ApvChromaFormat::from_byte(byte).expect("valid");
            assert_eq!(decoded, fmt);
        }
    }

    #[test]
    fn test_chroma_format_shifts() {
        assert_eq!(ApvChromaFormat::Yuv420.chroma_h_shift(), 1);
        assert_eq!(ApvChromaFormat::Yuv420.chroma_v_shift(), 1);
        assert_eq!(ApvChromaFormat::Yuv422.chroma_h_shift(), 1);
        assert_eq!(ApvChromaFormat::Yuv422.chroma_v_shift(), 0);
        assert_eq!(ApvChromaFormat::Yuv444.chroma_h_shift(), 0);
        assert_eq!(ApvChromaFormat::Yuv444.chroma_v_shift(), 0);
    }

    #[test]
    fn test_bit_depth_values() {
        assert_eq!(ApvBitDepth::Eight.bits(), 8);
        assert_eq!(ApvBitDepth::Ten.bits(), 10);
        assert_eq!(ApvBitDepth::Twelve.bits(), 12);
        assert_eq!(ApvBitDepth::Eight.max_value(), 255);
        assert_eq!(ApvBitDepth::Ten.max_value(), 1023);
        assert_eq!(ApvBitDepth::Twelve.max_value(), 4095);
    }

    #[test]
    fn test_bit_depth_roundtrip() {
        for bd in [ApvBitDepth::Eight, ApvBitDepth::Ten, ApvBitDepth::Twelve] {
            let byte = bd.to_byte();
            let decoded = ApvBitDepth::from_byte(byte).expect("valid");
            assert_eq!(decoded, bd);
        }
    }

    #[test]
    fn test_frame_header_serialize_roundtrip() {
        let header = ApvFrameHeader {
            profile: ApvProfile::Simple,
            width: 1920,
            height: 1080,
            bit_depth: ApvBitDepth::Ten,
            chroma_format: ApvChromaFormat::Yuv422,
            qp: 30,
            tile_cols: 4,
            tile_rows: 2,
        };

        let bytes = header.to_bytes();
        assert_eq!(bytes.len(), APV_HEADER_SIZE);
        assert_eq!(&bytes[0..4], b"APV1");

        let restored = ApvFrameHeader::from_bytes(&bytes).expect("valid header");
        assert_eq!(restored.profile, ApvProfile::Simple);
        assert_eq!(restored.width, 1920);
        assert_eq!(restored.height, 1080);
        assert_eq!(restored.bit_depth, ApvBitDepth::Ten);
        assert_eq!(restored.chroma_format, ApvChromaFormat::Yuv422);
        assert_eq!(restored.qp, 30);
        assert_eq!(restored.tile_cols, 4);
        assert_eq!(restored.tile_rows, 2);
    }

    #[test]
    fn test_frame_header_from_config() {
        let config = ApvConfig::new(640, 480)
            .expect("valid")
            .with_qp(35)
            .with_profile(ApvProfile::Main);
        let header = ApvFrameHeader::from_config(&config);
        assert_eq!(header.width, 640);
        assert_eq!(header.height, 480);
        assert_eq!(header.qp, 35);
        assert_eq!(header.profile, ApvProfile::Main);
    }

    #[test]
    fn test_header_bad_magic() {
        let mut bytes = [0u8; APV_HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"NOPE");
        assert!(ApvFrameHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_too_short() {
        assert!(ApvFrameHeader::from_bytes(&[0u8; 10]).is_err());
    }

    #[test]
    fn test_header_zero_dimensions() {
        let header = ApvFrameHeader {
            profile: ApvProfile::Simple,
            width: 0,
            height: 100,
            bit_depth: ApvBitDepth::Eight,
            chroma_format: ApvChromaFormat::Yuv420,
            qp: 22,
            tile_cols: 1,
            tile_rows: 1,
        };
        let bytes = header.to_bytes();
        assert!(ApvFrameHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_header_invalid_qp_in_bytes() {
        let mut header = ApvFrameHeader {
            profile: ApvProfile::Simple,
            width: 320,
            height: 240,
            bit_depth: ApvBitDepth::Eight,
            chroma_format: ApvChromaFormat::Yuv420,
            qp: 22,
            tile_cols: 1,
            tile_rows: 1,
        };
        header.qp = 22;
        let mut bytes = header.to_bytes();
        bytes[11] = 64; // QP out of range
        assert!(ApvFrameHeader::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_tile_info_construction() {
        let info = ApvTileInfo {
            col: 0,
            row: 0,
            offset: 16,
            size: 1024,
            width: 320,
            height: 240,
        };
        assert_eq!(info.col, 0);
        assert_eq!(info.offset, 16);
        assert_eq!(info.size, 1024);
    }

    #[test]
    fn test_tile_offsets() {
        let config = ApvConfig::new(640, 480)
            .expect("valid")
            .with_tiles(2, 2)
            .expect("valid tiles");
        assert_eq!(config.tile_x_offset(0), 0);
        assert_eq!(config.tile_x_offset(1), 320);
        assert_eq!(config.tile_y_offset(0), 0);
        assert_eq!(config.tile_y_offset(1), 240);
    }

    #[test]
    fn test_error_display() {
        let err = ApvError::InvalidDimensions {
            width: 0,
            height: 0,
            reason: "test".to_string(),
        };
        assert!(format!("{err}").contains("test"));

        let err = ApvError::InvalidQp(100);
        assert!(format!("{err}").contains("100"));

        let err = ApvError::EncodingFailed("oops".to_string());
        assert!(format!("{err}").contains("oops"));

        let err = ApvError::DecodingFailed("bad".to_string());
        assert!(format!("{err}").contains("bad"));

        let err = ApvError::InvalidBitstream("corrupt".to_string());
        assert!(format!("{err}").contains("corrupt"));

        let err = ApvError::UnsupportedProfile;
        assert!(format!("{err}").contains("unsupported"));

        let err = ApvError::BufferTooSmall;
        assert!(format!("{err}").contains("small"));
    }

    #[test]
    fn test_error_into_codec_error() {
        let err: crate::error::CodecError = ApvError::InvalidDimensions {
            width: 0,
            height: 0,
            reason: "test".to_string(),
        }
        .into();
        assert!(matches!(err, crate::error::CodecError::InvalidParameter(_)));

        let err: crate::error::CodecError = ApvError::InvalidQp(100).into();
        assert!(matches!(err, crate::error::CodecError::InvalidParameter(_)));

        let err: crate::error::CodecError = ApvError::EncodingFailed("x".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::Internal(_)));

        let err: crate::error::CodecError = ApvError::DecodingFailed("x".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::DecoderError(_)));

        let err: crate::error::CodecError = ApvError::InvalidBitstream("x".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::InvalidBitstream(_)));

        let err: crate::error::CodecError = ApvError::UnsupportedProfile.into();
        assert!(matches!(
            err,
            crate::error::CodecError::UnsupportedFeature(_)
        ));
    }

    #[test]
    fn test_config_default() {
        let config = ApvConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_chroma_format_planes() {
        assert_eq!(ApvChromaFormat::Yuv420.chroma_planes(), 2);
        assert_eq!(ApvChromaFormat::Yuv422.chroma_planes(), 2);
        assert_eq!(ApvChromaFormat::Yuv444.chroma_planes(), 2);
    }

    #[test]
    fn test_chroma_format_invalid_byte() {
        assert!(ApvChromaFormat::from_byte(3).is_err());
    }

    #[test]
    fn test_bit_depth_invalid_byte() {
        assert!(ApvBitDepth::from_byte(3).is_err());
    }

    #[test]
    fn test_config_with_all_builders() {
        let config = ApvConfig::new(640, 480)
            .expect("valid")
            .with_profile(ApvProfile::High)
            .with_bit_depth(ApvBitDepth::Twelve)
            .with_chroma_format(ApvChromaFormat::Yuv444)
            .with_qp(10);
        assert_eq!(config.profile, ApvProfile::High);
        assert_eq!(config.bit_depth, ApvBitDepth::Twelve);
        assert_eq!(config.chroma_format, ApvChromaFormat::Yuv444);
        assert_eq!(config.qp, 10);
    }

    #[test]
    fn test_header_zero_tile_cols() {
        let mut header = ApvFrameHeader {
            profile: ApvProfile::Simple,
            width: 320,
            height: 240,
            bit_depth: ApvBitDepth::Eight,
            chroma_format: ApvChromaFormat::Yuv420,
            qp: 22,
            tile_cols: 1,
            tile_rows: 1,
        };
        header.tile_cols = 1;
        header.tile_rows = 1;
        let mut bytes = header.to_bytes();
        // Overwrite tile_cols to 0
        bytes[12] = 0;
        bytes[13] = 0;
        assert!(ApvFrameHeader::from_bytes(&bytes).is_err());
    }
}
