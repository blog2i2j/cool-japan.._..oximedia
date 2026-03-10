//! DNG type definitions.

use crate::error::{ImageError, ImageResult};
use std::collections::HashMap;

// ==========================================
// Types
// ==========================================

/// Bayer Color Filter Array patterns.
///
/// The CFA pattern describes the arrangement of color filters on the image
/// sensor. Each 2x2 block of pixels has one of these arrangements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CfaPattern {
    /// Red-Green / Green-Blue (most common: Canon, Nikon, Sony).
    Rggb,
    /// Blue-Green / Green-Red.
    Bggr,
    /// Green-Red / Blue-Green.
    Grbg,
    /// Green-Blue / Red-Green.
    Gbrg,
}

impl CfaPattern {
    /// Returns the 2x2 color indices for this pattern.
    /// 0 = Red, 1 = Green (on red row), 2 = Green (on blue row), 3 = Blue.
    /// Returned as [top-left, top-right, bottom-left, bottom-right].
    #[must_use]
    pub const fn color_indices(&self) -> [u8; 4] {
        match self {
            Self::Rggb => [0, 1, 1, 2],
            Self::Bggr => [2, 1, 1, 0],
            Self::Grbg => [1, 0, 2, 1],
            Self::Gbrg => [1, 2, 0, 1],
        }
    }

    /// Returns the byte representation for TIFF CFA pattern tag.
    #[must_use]
    pub const fn as_bytes(&self) -> [u8; 4] {
        match self {
            Self::Rggb => [0, 1, 1, 2],
            Self::Bggr => [2, 1, 1, 0],
            Self::Grbg => [1, 0, 2, 1],
            Self::Gbrg => [1, 2, 0, 1],
        }
    }
}

/// DNG compression types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DngCompression {
    /// Uncompressed (TIFF compression = 1).
    Uncompressed,
    /// JPEG lossless Huffman (TIFF compression = 7).
    LosslessJpeg,
    /// ZIP/deflate (TIFF compression = 8).
    Deflate,
    /// Lossy DNG (TIFF compression = 34892).
    LossyDng,
}

impl DngCompression {
    pub(crate) fn from_u16(value: u16) -> ImageResult<Self> {
        match value {
            1 => Ok(Self::Uncompressed),
            7 => Ok(Self::LosslessJpeg),
            8 => Ok(Self::Deflate),
            34892 => Ok(Self::LossyDng),
            _ => Err(ImageError::unsupported(format!(
                "DNG compression type: {value}"
            ))),
        }
    }

    pub(crate) const fn to_u16(self) -> u16 {
        match self {
            Self::Uncompressed => 1,
            Self::LosslessJpeg => 7,
            Self::Deflate => 8,
            Self::LossyDng => 34892,
        }
    }
}

/// White balance information from the camera's as-shot settings.
#[derive(Debug, Clone)]
pub struct WhiteBalance {
    /// As-shot neutral white balance (R, G, B multipliers).
    /// These are the inverse of the white balance gains.
    pub as_shot_neutral: [f64; 3],
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self {
            as_shot_neutral: [1.0, 1.0, 1.0],
        }
    }
}

/// Color calibration matrices for converting camera color space to standard.
#[derive(Debug, Clone)]
pub struct ColorCalibration {
    /// Color matrix 1 (3x3, maps camera RGB to CIE XYZ under illuminant 1).
    pub color_matrix_1: [[f64; 3]; 3],
    /// Color matrix 2 (optional, for a second illuminant).
    pub color_matrix_2: Option<[[f64; 3]; 3]>,
    /// Forward matrix 1 (optional, maps white-balanced camera to XYZ).
    pub forward_matrix_1: Option<[[f64; 3]; 3]>,
    /// Calibration illuminant 1 (EXIF LightSource value, 21 = D65, 17 = Standard A).
    pub illuminant_1: u16,
    /// Calibration illuminant 2 (optional).
    pub illuminant_2: Option<u16>,
}

impl Default for ColorCalibration {
    fn default() -> Self {
        // Identity matrix as default (no color transformation)
        Self {
            color_matrix_1: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            color_matrix_2: None,
            forward_matrix_1: None,
            illuminant_1: 21, // D65
            illuminant_2: None,
        }
    }
}

/// DNG metadata describing the capture conditions and camera properties.
#[derive(Debug, Clone)]
pub struct DngMetadata {
    /// DNG specification version (e.g., [1, 4, 0, 0] for DNG 1.4).
    pub dng_version: [u8; 4],
    /// Camera model name.
    pub camera_model: String,
    /// CFA pattern of the sensor.
    pub cfa_pattern: CfaPattern,
    /// White balance settings.
    pub white_balance: WhiteBalance,
    /// Color calibration matrices.
    pub color_calibration: ColorCalibration,
    /// Per-channel black levels (sensor floor values).
    pub black_level: Vec<f64>,
    /// Per-channel white levels (sensor saturation values).
    pub white_level: Vec<u32>,
    /// Active area of the sensor [top, left, bottom, right].
    pub active_area: Option<[u32; 4]>,
    /// Additional EXIF metadata as key-value pairs.
    pub exif: HashMap<String, String>,
}

impl Default for DngMetadata {
    fn default() -> Self {
        Self {
            dng_version: [1, 4, 0, 0],
            camera_model: String::new(),
            cfa_pattern: CfaPattern::Rggb,
            white_balance: WhiteBalance::default(),
            color_calibration: ColorCalibration::default(),
            black_level: vec![0.0],
            white_level: vec![65535],
            active_area: None,
            exif: HashMap::new(),
        }
    }
}

/// A decoded DNG image containing raw sensor data and metadata.
#[derive(Debug)]
pub struct DngImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bit depth of the sensor data (8, 10, 12, 14, or 16).
    pub bit_depth: u8,
    /// Number of channels (1 for raw CFA, 3 for demosaiced RGB).
    pub channels: u8,
    /// Raw sensor values stored as u16 (even for lower bit depths).
    pub raw_data: Vec<u16>,
    /// DNG metadata.
    pub metadata: DngMetadata,
    /// Whether the data has been demosaiced to RGB.
    pub is_demosaiced: bool,
}
