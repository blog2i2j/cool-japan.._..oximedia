//! JPEG-XL type definitions.
//!
//! Core types for JPEG-XL image headers, configuration, and color spaces.

use crate::error::{CodecError, CodecResult};

/// JPEG-XL codestream signature: 0xFF 0x0A.
pub const JXL_CODESTREAM_SIGNATURE: [u8; 2] = [0xFF, 0x0A];

/// JPEG-XL container (ISOBMFF) signature.
pub const JXL_CONTAINER_SIGNATURE: [u8; 12] = [
    0x00, 0x00, 0x00, 0x0C, 0x4A, 0x58, 0x4C, 0x20, 0x0D, 0x0A, 0x87, 0x0A,
];

/// JPEG-XL image header.
///
/// Contains all metadata needed to interpret a decoded image.
#[derive(Clone, Debug)]
pub struct JxlHeader {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bits per sample (8, 16, or 32).
    pub bits_per_sample: u8,
    /// Number of color channels (1 for gray, 3 for RGB, 4 for RGBA).
    pub num_channels: u8,
    /// Whether samples are floating point.
    pub is_float: bool,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// Color space of the image data.
    pub color_space: JxlColorSpace,
    /// EXIF orientation (1-8, 1 = normal).
    pub orientation: u8,
}

impl JxlHeader {
    /// Create a header for an 8-bit sRGB image.
    pub fn srgb(width: u32, height: u32, channels: u8) -> CodecResult<Self> {
        if channels == 0 || channels > 4 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid channel count: {channels}, must be 1-4"
            )));
        }
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidParameter(
                "Width and height must be non-zero".into(),
            ));
        }
        let has_alpha = channels == 2 || channels == 4;
        let color_space = if channels <= 2 {
            JxlColorSpace::Gray
        } else {
            JxlColorSpace::Srgb
        };
        Ok(Self {
            width,
            height,
            bits_per_sample: 8,
            num_channels: channels,
            is_float: false,
            has_alpha,
            color_space,
            orientation: 1,
        })
    }

    /// Total number of channels including alpha.
    pub fn total_channels(&self) -> u8 {
        self.num_channels
    }

    /// Number of color channels (excluding alpha).
    pub fn color_channels(&self) -> u8 {
        if self.has_alpha {
            self.num_channels.saturating_sub(1)
        } else {
            self.num_channels
        }
    }

    /// Bytes per sample for this bit depth.
    pub fn bytes_per_sample(&self) -> usize {
        match self.bits_per_sample {
            1..=8 => 1,
            9..=16 => 2,
            _ => 4,
        }
    }

    /// Total expected data size in bytes for interleaved pixel data.
    pub fn data_size(&self) -> usize {
        self.width as usize
            * self.height as usize
            * self.num_channels as usize
            * self.bytes_per_sample()
    }

    /// Validate that the header is internally consistent.
    pub fn validate(&self) -> CodecResult<()> {
        if self.width == 0 || self.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Width and height must be non-zero".into(),
            ));
        }
        if self.width > 1_073_741_823 || self.height > 1_073_741_823 {
            return Err(CodecError::InvalidParameter(
                "Dimensions exceed JPEG-XL maximum (2^30 - 1)".into(),
            ));
        }
        if self.num_channels == 0 || self.num_channels > 4 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid channel count: {}",
                self.num_channels
            )));
        }
        match self.bits_per_sample {
            8 | 16 | 32 => {}
            other => {
                return Err(CodecError::InvalidParameter(format!(
                    "Unsupported bit depth: {other}, must be 8, 16, or 32"
                )));
            }
        }
        Ok(())
    }
}

impl Default for JxlHeader {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            bits_per_sample: 8,
            num_channels: 3,
            is_float: false,
            has_alpha: false,
            color_space: JxlColorSpace::Srgb,
            orientation: 1,
        }
    }
}

/// JPEG-XL color space.
///
/// JPEG-XL natively uses the XYB perceptual color space for lossy encoding,
/// but lossless mode typically operates in the original color space with RCT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlColorSpace {
    /// Standard sRGB (IEC 61966-2-1).
    Srgb,
    /// Linear sRGB (no transfer function).
    LinearSrgb,
    /// Grayscale (single luminance channel).
    Gray,
    /// XYB perceptual color space (JPEG-XL native, used for lossy).
    Xyb,
}

impl Default for JxlColorSpace {
    fn default() -> Self {
        Self::Srgb
    }
}

/// Frame encoding mode.
///
/// JPEG-XL supports two fundamentally different encoding modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlFrameEncoding {
    /// VarDCT mode for lossy compression (DCT-based, similar to JPEG but improved).
    VarDct,
    /// Modular mode for lossless (and progressive) compression.
    Modular,
}

/// Encoder configuration.
///
/// Controls the encoding behavior including quality, effort, and mode.
#[derive(Clone, Debug)]
pub struct JxlConfig {
    /// Quality factor: 0.0 = lossless, 100.0 = worst quality.
    /// Values below ~1.0 are effectively lossless.
    pub quality: f32,
    /// Encoding effort (1-9). Higher values produce smaller files but take longer.
    /// - 1: Fastest, largest files
    /// - 7: Default balance
    /// - 9: Slowest, smallest files
    pub effort: u8,
    /// Force lossless encoding regardless of quality setting.
    pub lossless: bool,
    /// Use container format (ISOBMFF box structure) instead of bare codestream.
    pub use_container: bool,
}

impl JxlConfig {
    /// Create a lossless configuration.
    pub fn new_lossless() -> Self {
        Self {
            quality: 0.0,
            effort: 7,
            lossless: true,
            use_container: false,
        }
    }

    /// Create a lossy configuration with given quality.
    pub fn new_lossy(quality: f32) -> Self {
        Self {
            quality: quality.clamp(0.0, 100.0),
            effort: 7,
            lossless: false,
            use_container: false,
        }
    }

    /// Set effort level.
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.clamp(1, 9);
        self
    }

    /// Determine the frame encoding mode from configuration.
    pub fn frame_encoding(&self) -> JxlFrameEncoding {
        if self.lossless {
            JxlFrameEncoding::Modular
        } else {
            JxlFrameEncoding::VarDct
        }
    }

    /// Validate configuration values.
    pub fn validate(&self) -> CodecResult<()> {
        if self.effort < 1 || self.effort > 9 {
            return Err(CodecError::InvalidParameter(format!(
                "Effort must be 1-9, got {}",
                self.effort
            )));
        }
        if self.quality < 0.0 || self.quality > 100.0 {
            return Err(CodecError::InvalidParameter(format!(
                "Quality must be 0.0-100.0, got {}",
                self.quality
            )));
        }
        Ok(())
    }
}

impl Default for JxlConfig {
    fn default() -> Self {
        Self::new_lossless()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_header_srgb() {
        let header = JxlHeader::srgb(1920, 1080, 3).expect("valid header");
        assert_eq!(header.width, 1920);
        assert_eq!(header.height, 1080);
        assert_eq!(header.num_channels, 3);
        assert!(!header.has_alpha);
        assert_eq!(header.color_space, JxlColorSpace::Srgb);
    }

    #[test]
    #[ignore]
    fn test_header_srgb_rgba() {
        let header = JxlHeader::srgb(100, 100, 4).expect("valid header");
        assert!(header.has_alpha);
        assert_eq!(header.color_channels(), 3);
        assert_eq!(header.total_channels(), 4);
    }

    #[test]
    #[ignore]
    fn test_header_gray() {
        let header = JxlHeader::srgb(64, 64, 1).expect("valid header");
        assert_eq!(header.color_space, JxlColorSpace::Gray);
        assert!(!header.has_alpha);
    }

    #[test]
    #[ignore]
    fn test_header_invalid_channels() {
        assert!(JxlHeader::srgb(100, 100, 0).is_err());
        assert!(JxlHeader::srgb(100, 100, 5).is_err());
    }

    #[test]
    #[ignore]
    fn test_header_zero_dimensions() {
        assert!(JxlHeader::srgb(0, 100, 3).is_err());
        assert!(JxlHeader::srgb(100, 0, 3).is_err());
    }

    #[test]
    #[ignore]
    fn test_header_data_size() {
        let header = JxlHeader::srgb(10, 10, 3).expect("valid");
        assert_eq!(header.data_size(), 10 * 10 * 3);
    }

    #[test]
    #[ignore]
    fn test_config_lossless() {
        let config = JxlConfig::new_lossless();
        assert!(config.lossless);
        assert_eq!(config.frame_encoding(), JxlFrameEncoding::Modular);
    }

    #[test]
    #[ignore]
    fn test_config_lossy() {
        let config = JxlConfig::new_lossy(50.0);
        assert!(!config.lossless);
        assert_eq!(config.frame_encoding(), JxlFrameEncoding::VarDct);
    }

    #[test]
    #[ignore]
    fn test_config_effort() {
        let config = JxlConfig::new_lossless().with_effort(3);
        assert_eq!(config.effort, 3);
    }

    #[test]
    #[ignore]
    fn test_config_validate() {
        assert!(JxlConfig::new_lossless().validate().is_ok());
        let mut bad = JxlConfig::new_lossless();
        bad.effort = 0;
        assert!(bad.validate().is_err());
    }

    #[test]
    #[ignore]
    fn test_codestream_signature() {
        assert_eq!(JXL_CODESTREAM_SIGNATURE, [0xFF, 0x0A]);
    }

    #[test]
    #[ignore]
    fn test_container_signature() {
        assert_eq!(JXL_CONTAINER_SIGNATURE.len(), 12);
        // First 4 bytes are box size (12), next 4 are "JXL " type
        assert_eq!(&JXL_CONTAINER_SIGNATURE[4..8], b"JXL ");
    }
}
