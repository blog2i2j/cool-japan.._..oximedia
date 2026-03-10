//! FFV1 codec types and configuration.
//!
//! Defines the core types for the FFV1 (FF Video Codec 1) lossless video codec
//! as specified in RFC 9043 / ISO/IEC 24114.

use crate::error::{CodecError, CodecResult};

/// FFV1 codec version.
///
/// FFV1 has evolved through several versions, with Version 3 being the most
/// widely used and the version standardized in RFC 9043.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffv1Version {
    /// Version 0 - original, uses Golomb-Rice coding.
    V0,
    /// Version 1 - adds non-planar colorspace support, uses Golomb-Rice coding.
    V1,
    /// Version 2 - experimental (rarely used).
    V2,
    /// Version 3 - RFC 9043 standard, range coder, slice-level CRC.
    V3,
}

impl Ffv1Version {
    /// Parse version number from integer.
    pub fn from_u8(v: u8) -> CodecResult<Self> {
        match v {
            0 => Ok(Self::V0),
            1 => Ok(Self::V1),
            2 => Ok(Self::V2),
            3 => Ok(Self::V3),
            _ => Err(CodecError::InvalidParameter(format!(
                "unsupported FFV1 version: {v}"
            ))),
        }
    }

    /// Convert to integer representation.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::V0 => 0,
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }

    /// Whether this version uses range coding (v3+) or Golomb-Rice (v0/v1).
    #[must_use]
    pub const fn uses_range_coder(self) -> bool {
        matches!(self, Self::V3 | Self::V2)
    }

    /// Whether this version supports error correction (CRC per slice).
    #[must_use]
    pub const fn supports_ec(self) -> bool {
        matches!(self, Self::V3)
    }
}

/// FFV1 colorspace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffv1Colorspace {
    /// YCbCr colorspace (planar).
    YCbCr,
    /// RGB colorspace (encoded as JPEG2000-RCT transformed planes).
    Rgb,
}

impl Ffv1Colorspace {
    /// Parse from integer value in the bitstream.
    pub fn from_u8(v: u8) -> CodecResult<Self> {
        match v {
            0 => Ok(Self::YCbCr),
            1 => Ok(Self::Rgb),
            _ => Err(CodecError::InvalidParameter(format!(
                "unsupported FFV1 colorspace: {v}"
            ))),
        }
    }

    /// Convert to integer representation.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::YCbCr => 0,
            Self::Rgb => 1,
        }
    }

    /// Number of planes for this colorspace.
    #[must_use]
    pub const fn plane_count(self) -> usize {
        match self {
            Self::YCbCr => 3,
            // RGB is encoded as 3 planes (G, B-G, R-G) + optional alpha
            Self::Rgb => 3,
        }
    }
}

/// FFV1 chroma subsampling type (only for YCbCr).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffv1ChromaType {
    /// 4:2:0 chroma subsampling.
    Chroma420,
    /// 4:2:2 chroma subsampling.
    Chroma422,
    /// 4:4:4 chroma subsampling (no subsampling).
    Chroma444,
}

impl Ffv1ChromaType {
    /// Horizontal subsampling shift (log2 of ratio).
    #[must_use]
    pub const fn h_shift(self) -> u32 {
        match self {
            Self::Chroma420 | Self::Chroma422 => 1,
            Self::Chroma444 => 0,
        }
    }

    /// Vertical subsampling shift (log2 of ratio).
    #[must_use]
    pub const fn v_shift(self) -> u32 {
        match self {
            Self::Chroma420 => 1,
            Self::Chroma422 | Self::Chroma444 => 0,
        }
    }

    /// Parse from horizontal and vertical chroma shifts.
    pub fn from_shifts(h_shift: u32, v_shift: u32) -> CodecResult<Self> {
        match (h_shift, v_shift) {
            (1, 1) => Ok(Self::Chroma420),
            (1, 0) => Ok(Self::Chroma422),
            (0, 0) => Ok(Self::Chroma444),
            _ => Err(CodecError::InvalidParameter(format!(
                "unsupported chroma subsampling: h_shift={h_shift}, v_shift={v_shift}"
            ))),
        }
    }
}

/// FFV1 configuration record.
///
/// Contains all parameters needed to initialize the encoder or decoder.
/// In a container, this is stored as codec extradata (the "configuration record").
#[derive(Clone, Debug)]
pub struct Ffv1Config {
    /// FFV1 version.
    pub version: Ffv1Version,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Colorspace.
    pub colorspace: Ffv1Colorspace,
    /// Chroma subsampling (only meaningful for YCbCr).
    pub chroma_type: Ffv1ChromaType,
    /// Bits per raw sample (8, 10, 12, or 16).
    pub bits_per_raw_sample: u8,
    /// Number of horizontal slices.
    pub num_h_slices: u32,
    /// Number of vertical slices.
    pub num_v_slices: u32,
    /// Error correction enabled (CRC32 per slice, v3+ only).
    pub ec: bool,
    /// Range coder state transition table index (0 = default).
    pub state_transition_delta: Vec<i16>,
    /// Whether to use range coder (true) or Golomb-Rice (false).
    /// For v3 this is always true.
    pub range_coder_mode: bool,
}

impl Default for Ffv1Config {
    fn default() -> Self {
        Self {
            version: Ffv1Version::V3,
            width: 0,
            height: 0,
            colorspace: Ffv1Colorspace::YCbCr,
            chroma_type: Ffv1ChromaType::Chroma420,
            bits_per_raw_sample: 8,
            num_h_slices: 1,
            num_v_slices: 1,
            ec: true,
            state_transition_delta: Vec::new(),
            range_coder_mode: true,
        }
    }
}

impl Ffv1Config {
    /// Total number of slices.
    #[must_use]
    pub fn num_slices(&self) -> u32 {
        self.num_h_slices * self.num_v_slices
    }

    /// Maximum sample value for the configured bit depth.
    #[must_use]
    pub fn max_sample_value(&self) -> i32 {
        (1i32 << self.bits_per_raw_sample) - 1
    }

    /// Number of planes for this configuration.
    #[must_use]
    pub fn plane_count(&self) -> usize {
        self.colorspace.plane_count()
    }

    /// Get the dimensions for a given plane index.
    #[must_use]
    pub fn plane_dimensions(&self, plane_index: usize) -> (u32, u32) {
        if plane_index == 0 || self.colorspace == Ffv1Colorspace::Rgb {
            (self.width, self.height)
        } else {
            let w =
                (self.width + (1 << self.chroma_type.h_shift()) - 1) >> self.chroma_type.h_shift();
            let h =
                (self.height + (1 << self.chroma_type.v_shift()) - 1) >> self.chroma_type.v_shift();
            (w, h)
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> CodecResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(CodecError::InvalidParameter(
                "frame dimensions must be nonzero".to_string(),
            ));
        }
        if !matches!(self.bits_per_raw_sample, 8 | 10 | 12 | 16) {
            return Err(CodecError::InvalidParameter(format!(
                "unsupported bits_per_raw_sample: {}",
                self.bits_per_raw_sample
            )));
        }
        if self.num_h_slices == 0 || self.num_v_slices == 0 {
            return Err(CodecError::InvalidParameter(
                "slice counts must be nonzero".to_string(),
            ));
        }
        Ok(())
    }
}

/// Slice header within a frame.
#[derive(Clone, Debug, Default)]
pub struct SliceHeader {
    /// X offset of this slice in pixels.
    pub slice_x: u32,
    /// Y offset of this slice in pixels.
    pub slice_y: u32,
    /// Width of this slice in pixels.
    pub slice_width: u32,
    /// Height of this slice in pixels.
    pub slice_height: u32,
}

/// Number of context states used per plane line in range coder mode.
/// Each context has an adaptive probability state (0..255).
pub const CONTEXT_COUNT: usize = 32;

/// Initial state value for range coder contexts.
pub const INITIAL_STATE: u8 = 128;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_version_roundtrip() {
        for v in [
            Ffv1Version::V0,
            Ffv1Version::V1,
            Ffv1Version::V2,
            Ffv1Version::V3,
        ] {
            let n = v.as_u8();
            let parsed = Ffv1Version::from_u8(n).expect("valid version");
            assert_eq!(parsed, v);
        }
    }

    #[test]
    #[ignore]
    fn test_version_invalid() {
        assert!(Ffv1Version::from_u8(4).is_err());
    }

    #[test]
    #[ignore]
    fn test_colorspace() {
        assert_eq!(Ffv1Colorspace::YCbCr.plane_count(), 3);
        assert_eq!(Ffv1Colorspace::Rgb.plane_count(), 3);
        assert_eq!(Ffv1Colorspace::YCbCr.as_u8(), 0);
        assert_eq!(Ffv1Colorspace::Rgb.as_u8(), 1);
    }

    #[test]
    #[ignore]
    fn test_chroma_type_shifts() {
        assert_eq!(Ffv1ChromaType::Chroma420.h_shift(), 1);
        assert_eq!(Ffv1ChromaType::Chroma420.v_shift(), 1);
        assert_eq!(Ffv1ChromaType::Chroma422.h_shift(), 1);
        assert_eq!(Ffv1ChromaType::Chroma422.v_shift(), 0);
        assert_eq!(Ffv1ChromaType::Chroma444.h_shift(), 0);
        assert_eq!(Ffv1ChromaType::Chroma444.v_shift(), 0);
    }

    #[test]
    #[ignore]
    fn test_chroma_type_from_shifts() {
        assert_eq!(
            Ffv1ChromaType::from_shifts(1, 1).expect("valid"),
            Ffv1ChromaType::Chroma420
        );
        assert_eq!(
            Ffv1ChromaType::from_shifts(1, 0).expect("valid"),
            Ffv1ChromaType::Chroma422
        );
        assert_eq!(
            Ffv1ChromaType::from_shifts(0, 0).expect("valid"),
            Ffv1ChromaType::Chroma444
        );
        assert!(Ffv1ChromaType::from_shifts(2, 0).is_err());
    }

    #[test]
    #[ignore]
    fn test_config_plane_dimensions() {
        let config = Ffv1Config {
            width: 1920,
            height: 1080,
            chroma_type: Ffv1ChromaType::Chroma420,
            ..Default::default()
        };
        assert_eq!(config.plane_dimensions(0), (1920, 1080));
        assert_eq!(config.plane_dimensions(1), (960, 540));
        assert_eq!(config.plane_dimensions(2), (960, 540));
    }

    #[test]
    #[ignore]
    fn test_config_validation() {
        let mut config = Ffv1Config {
            width: 1920,
            height: 1080,
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        config.width = 0;
        assert!(config.validate().is_err());

        config.width = 1920;
        config.bits_per_raw_sample = 9;
        assert!(config.validate().is_err());
    }

    #[test]
    #[ignore]
    fn test_max_sample_value() {
        let config = Ffv1Config {
            bits_per_raw_sample: 8,
            ..Default::default()
        };
        assert_eq!(config.max_sample_value(), 255);

        let config10 = Ffv1Config {
            bits_per_raw_sample: 10,
            ..Default::default()
        };
        assert_eq!(config10.max_sample_value(), 1023);
    }
}
