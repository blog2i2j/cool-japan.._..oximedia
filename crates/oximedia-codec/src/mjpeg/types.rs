//! MJPEG-specific types.
//!
//! Configuration and error types for the Motion JPEG codec.

use oximedia_core::PixelFormat;
use thiserror::Error;

/// MJPEG encoder/decoder configuration.
#[derive(Clone, Debug)]
pub struct MjpegConfig {
    /// JPEG quality factor (1-100, where 100 is highest quality).
    /// Default: 85.
    pub quality: u8,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Input pixel format.
    pub pixel_format: PixelFormat,
}

impl MjpegConfig {
    /// Create a new MJPEG configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Frame width in pixels (must be > 0)
    /// * `height` - Frame height in pixels (must be > 0)
    ///
    /// # Errors
    ///
    /// Returns `MjpegError::InvalidConfig` if width or height is zero.
    pub fn new(width: u32, height: u32) -> Result<Self, MjpegError> {
        if width == 0 || height == 0 {
            return Err(MjpegError::InvalidConfig(
                "width and height must be non-zero".to_string(),
            ));
        }
        Ok(Self {
            quality: 85,
            width,
            height,
            pixel_format: PixelFormat::Yuv420p,
        })
    }

    /// Set the JPEG quality factor (clamped to 1-100).
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality.clamp(1, 100);
        self
    }

    /// Set the pixel format.
    #[must_use]
    pub fn with_pixel_format(mut self, format: PixelFormat) -> Self {
        self.pixel_format = format;
        self
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns `MjpegError::InvalidConfig` if any parameter is out of range.
    pub fn validate(&self) -> Result<(), MjpegError> {
        if self.width == 0 || self.height == 0 {
            return Err(MjpegError::InvalidConfig(
                "width and height must be non-zero".to_string(),
            ));
        }
        if self.quality == 0 || self.quality > 100 {
            return Err(MjpegError::InvalidConfig(format!(
                "quality must be 1-100, got {}",
                self.quality
            )));
        }
        // Check for reasonable resolution limits (16384 x 16384 max for baseline JPEG)
        if self.width > 16384 || self.height > 16384 {
            return Err(MjpegError::InvalidConfig(format!(
                "dimensions {}x{} exceed baseline JPEG maximum of 16384x16384",
                self.width, self.height
            )));
        }
        Ok(())
    }
}

impl Default for MjpegConfig {
    fn default() -> Self {
        Self {
            quality: 85,
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Yuv420p,
        }
    }
}

/// Errors specific to the MJPEG codec.
#[derive(Debug, Error)]
pub enum MjpegError {
    /// Invalid configuration parameter.
    #[error("MJPEG config error: {0}")]
    InvalidConfig(String),

    /// Encoding failure.
    #[error("MJPEG encode error: {0}")]
    EncodeError(String),

    /// Decoding failure.
    #[error("MJPEG decode error: {0}")]
    DecodeError(String),

    /// Pixel format conversion failure.
    #[error("MJPEG pixel format error: {0}")]
    PixelFormatError(String),

    /// Frame dimensions mismatch.
    #[error(
        "MJPEG dimension mismatch: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
    )]
    DimensionMismatch {
        /// Expected width.
        expected_w: u32,
        /// Expected height.
        expected_h: u32,
        /// Actual width.
        actual_w: u32,
        /// Actual height.
        actual_h: u32,
    },
}

impl From<MjpegError> for crate::error::CodecError {
    fn from(e: MjpegError) -> Self {
        match e {
            MjpegError::InvalidConfig(msg) => crate::error::CodecError::InvalidParameter(msg),
            MjpegError::EncodeError(msg) => crate::error::CodecError::Internal(msg),
            MjpegError::DecodeError(msg) => crate::error::CodecError::DecoderError(msg),
            MjpegError::PixelFormatError(msg) => crate::error::CodecError::InvalidParameter(msg),
            MjpegError::DimensionMismatch {
                expected_w,
                expected_h,
                actual_w,
                actual_h,
            } => crate::error::CodecError::InvalidParameter(format!(
                "dimension mismatch: expected {expected_w}x{expected_h}, got {actual_w}x{actual_h}"
            )),
        }
    }
}

/// AVI1 marker type for MJPEG identification.
///
/// Distinguishes MJPEG frames from regular JPEG files.
/// The AVI1 marker is placed in an APP0 segment (0xFFE0)
/// immediately after the JFIF APP0 marker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MjpegMarkerType {
    /// Standard MJPEG (AVI1 marker present).
    Avi1,
    /// Plain JPEG frame (no AVI1 marker, still valid MJPEG).
    Plain,
}

/// MJPEG frame metadata extracted during decoding.
#[derive(Clone, Debug)]
pub struct MjpegFrameInfo {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Marker type found in the frame.
    pub marker_type: MjpegMarkerType,
    /// Compressed frame size in bytes.
    pub compressed_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = MjpegConfig::default();
        assert_eq!(config.quality, 85);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.pixel_format, PixelFormat::Yuv420p);
    }

    #[test]
    fn test_config_new_valid() {
        let config = MjpegConfig::new(640, 480);
        assert!(config.is_ok());
        let config = config.expect("valid config");
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
    }

    #[test]
    fn test_config_new_zero_width() {
        let result = MjpegConfig::new(0, 480);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_new_zero_height() {
        let result = MjpegConfig::new(640, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_with_quality() {
        let config = MjpegConfig::default().with_quality(50);
        assert_eq!(config.quality, 50);
    }

    #[test]
    fn test_config_quality_clamped_high() {
        let config = MjpegConfig::default().with_quality(200);
        assert_eq!(config.quality, 100);
    }

    #[test]
    fn test_config_quality_clamped_low() {
        let config = MjpegConfig::default().with_quality(0);
        assert_eq!(config.quality, 1);
    }

    #[test]
    fn test_config_with_pixel_format() {
        let config = MjpegConfig::default().with_pixel_format(PixelFormat::Rgb24);
        assert_eq!(config.pixel_format, PixelFormat::Rgb24);
    }

    #[test]
    fn test_config_validate_valid() {
        let config = MjpegConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_oversized() {
        let config = MjpegConfig {
            width: 20000,
            height: 20000,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_error_display() {
        let err = MjpegError::InvalidConfig("test".to_string());
        assert!(format!("{err}").contains("test"));

        let err = MjpegError::EncodeError("enc fail".to_string());
        assert!(format!("{err}").contains("enc fail"));

        let err = MjpegError::DecodeError("dec fail".to_string());
        assert!(format!("{err}").contains("dec fail"));
    }

    #[test]
    fn test_error_into_codec_error() {
        let err: crate::error::CodecError =
            MjpegError::InvalidConfig("bad config".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::InvalidParameter(_)));

        let err: crate::error::CodecError =
            MjpegError::EncodeError("encode fail".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::Internal(_)));

        let err: crate::error::CodecError =
            MjpegError::DecodeError("decode fail".to_string()).into();
        assert!(matches!(err, crate::error::CodecError::DecoderError(_)));
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = MjpegError::DimensionMismatch {
            expected_w: 640,
            expected_h: 480,
            actual_w: 320,
            actual_h: 240,
        };
        let display = format!("{err}");
        assert!(display.contains("640x480"));
        assert!(display.contains("320x240"));
    }

    #[test]
    fn test_mjpeg_frame_info() {
        let info = MjpegFrameInfo {
            width: 1920,
            height: 1080,
            marker_type: MjpegMarkerType::Avi1,
            compressed_size: 50000,
        };
        assert_eq!(info.width, 1920);
        assert_eq!(info.marker_type, MjpegMarkerType::Avi1);
    }

    #[test]
    fn test_config_quality_boundary_1() {
        let config = MjpegConfig::default().with_quality(1);
        assert_eq!(config.quality, 1);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_quality_boundary_100() {
        let config = MjpegConfig::default().with_quality(100);
        assert_eq!(config.quality, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_small_resolution() {
        let config = MjpegConfig::new(8, 8);
        assert!(config.is_ok());
        let config = config.expect("valid config");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_max_resolution() {
        let config = MjpegConfig::new(16384, 16384);
        assert!(config.is_ok());
        let config = config.expect("valid config");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_marker_type_equality() {
        assert_eq!(MjpegMarkerType::Avi1, MjpegMarkerType::Avi1);
        assert_eq!(MjpegMarkerType::Plain, MjpegMarkerType::Plain);
        assert_ne!(MjpegMarkerType::Avi1, MjpegMarkerType::Plain);
    }
}
