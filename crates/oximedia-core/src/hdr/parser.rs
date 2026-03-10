//! HDR metadata parsers for various container and codec formats.
//!
//! This module provides utilities for parsing HDR metadata from:
//! - HEVC/H.265 SEI messages (structure only, as HEVC is not supported)
//! - VP9 color configuration
//! - AV1 color configuration
//! - Matroska/WebM color elements
#![allow(clippy::match_same_arms)]

use super::metadata::{ContentLightLevel, MasteringDisplayColorVolume};
use super::primaries::{ColorPrimaries, Primaries, WhitePoint};
use super::transfer::TransferCharacteristic;

/// VP9 color configuration.
///
/// Parsed from VP9 bitstream color config syntax.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::Vp9ColorConfig;
///
/// let config = Vp9ColorConfig {
///     bit_depth: 10,
///     color_space: 9, // BT.2020
///     color_range: false,
///     subsampling_x: true,
///     subsampling_y: true,
/// };
///
/// assert_eq!(config.bit_depth, 10);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vp9ColorConfig {
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,

    /// Color space identifier.
    ///
    /// Values:
    /// - 0: Unknown
    /// - 1: BT.601
    /// - 2: BT.709
    /// - 3: SMPTE 170M
    /// - 4: SMPTE 240M
    /// - 5: BT.2020
    /// - 6: Reserved
    /// - 7: sRGB
    /// - 8-9: Reserved
    pub color_space: u8,

    /// Color range (false = limited, true = full).
    pub color_range: bool,

    /// Horizontal chroma subsampling (true = half resolution).
    pub subsampling_x: bool,

    /// Vertical chroma subsampling (true = half resolution).
    pub subsampling_y: bool,
}

impl Vp9ColorConfig {
    /// Creates a new VP9 color configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Vp9ColorConfig;
    ///
    /// let config = Vp9ColorConfig::new(10, 9, false, true, true);
    /// assert_eq!(config.bit_depth, 10);
    /// ```
    #[must_use]
    pub const fn new(
        bit_depth: u8,
        color_space: u8,
        color_range: bool,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Self {
        Self {
            bit_depth,
            color_space,
            color_range,
            subsampling_x,
            subsampling_y,
        }
    }

    /// Converts VP9 color space ID to color primaries.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{Vp9ColorConfig, ColorPrimaries};
    ///
    /// let config = Vp9ColorConfig::new(10, 9, false, true, true);
    /// assert_eq!(config.to_color_primaries(), ColorPrimaries::BT2020);
    /// ```
    #[must_use]
    pub const fn to_color_primaries(&self) -> ColorPrimaries {
        match self.color_space {
            1 | 3 => ColorPrimaries::Smpte170M, // BT.601 / SMPTE 170M
            2 => ColorPrimaries::BT709,
            4 => ColorPrimaries::Smpte240M,
            5 | 9 => ColorPrimaries::BT2020,
            7 => ColorPrimaries::BT709, // sRGB uses BT.709 primaries
            _ => ColorPrimaries::BT709, // Default to BT.709
        }
    }

    /// Returns true if this is 4:2:0 chroma subsampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Vp9ColorConfig;
    ///
    /// let config = Vp9ColorConfig::new(10, 9, false, true, true);
    /// assert!(config.is_420());
    /// ```
    #[must_use]
    pub const fn is_420(&self) -> bool {
        self.subsampling_x && self.subsampling_y
    }

    /// Returns true if this is 4:2:2 chroma subsampling.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Vp9ColorConfig;
    ///
    /// let config = Vp9ColorConfig::new(10, 9, false, true, false);
    /// assert!(config.is_422());
    /// ```
    #[must_use]
    pub const fn is_422(&self) -> bool {
        self.subsampling_x && !self.subsampling_y
    }

    /// Returns true if this is 4:4:4 (no chroma subsampling).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Vp9ColorConfig;
    ///
    /// let config = Vp9ColorConfig::new(10, 9, false, false, false);
    /// assert!(config.is_444());
    /// ```
    #[must_use]
    pub const fn is_444(&self) -> bool {
        !self.subsampling_x && !self.subsampling_y
    }
}

/// AV1 color configuration.
///
/// Parsed from AV1 sequence header color config syntax.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::Av1ColorConfig;
///
/// let config = Av1ColorConfig {
///     bit_depth: 10,
///     mono_chrome: false,
///     color_primaries: 9, // BT.2020
///     transfer_characteristics: 16, // ST.2084 (PQ)
///     matrix_coefficients: 9, // BT.2020 NCL
///     color_range: false,
///     subsampling_x: true,
///     subsampling_y: true,
///     chroma_sample_position: 0,
///     separate_uv_delta_q: false,
/// };
///
/// assert!(config.is_hdr());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools)]
pub struct Av1ColorConfig {
    /// Bit depth (8, 10, or 12).
    pub bit_depth: u8,

    /// Monochrome (grayscale) flag.
    pub mono_chrome: bool,

    /// Color primaries identifier (ISO/IEC 23091-2).
    pub color_primaries: u8,

    /// Transfer characteristics identifier (ISO/IEC 23091-2).
    pub transfer_characteristics: u8,

    /// Matrix coefficients identifier (ISO/IEC 23091-2).
    pub matrix_coefficients: u8,

    /// Color range (false = limited, true = full).
    pub color_range: bool,

    /// Horizontal chroma subsampling.
    pub subsampling_x: bool,

    /// Vertical chroma subsampling.
    pub subsampling_y: bool,

    /// Chroma sample position (0 = unknown, 1 = vertical, 2 = colocated).
    pub chroma_sample_position: u8,

    /// Separate U/V delta quantization flag.
    pub separate_uv_delta_q: bool,
}

impl Av1ColorConfig {
    /// Creates a new AV1 color configuration.
    #[must_use]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub const fn new(
        bit_depth: u8,
        mono_chrome: bool,
        color_primaries: u8,
        transfer_characteristics: u8,
        matrix_coefficients: u8,
        color_range: bool,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Self {
        Self {
            bit_depth,
            mono_chrome,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            subsampling_x,
            subsampling_y,
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
        }
    }

    /// Converts to color primaries enum.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{Av1ColorConfig, ColorPrimaries};
    ///
    /// let config = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
    /// assert_eq!(config.to_color_primaries(), ColorPrimaries::BT2020);
    /// ```
    #[must_use]
    pub const fn to_color_primaries(&self) -> ColorPrimaries {
        match self.color_primaries {
            1 => ColorPrimaries::BT709,
            4 => ColorPrimaries::Bt470M,
            5 => ColorPrimaries::Bt470Bg,
            6 => ColorPrimaries::Smpte170M,
            7 => ColorPrimaries::Smpte240M,
            8 => ColorPrimaries::Film,
            9 => ColorPrimaries::BT2020,
            10 => ColorPrimaries::BT709, // XYZ, map to BT.709
            11 => ColorPrimaries::DciP3,
            12 => ColorPrimaries::DisplayP3,
            _ => ColorPrimaries::BT709, // Unspecified/reserved
        }
    }

    /// Converts to transfer characteristic enum.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{Av1ColorConfig, TransferCharacteristic};
    ///
    /// let config = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
    /// assert_eq!(config.to_transfer_characteristic(), TransferCharacteristic::Pq);
    /// ```
    #[must_use]
    pub const fn to_transfer_characteristic(&self) -> TransferCharacteristic {
        match self.transfer_characteristics {
            1 | 6 | 14 | 15 => TransferCharacteristic::Bt709, // BT.709, BT.601, BT.2020 10-bit, BT.2020 12-bit
            8 => TransferCharacteristic::Linear,
            13 => TransferCharacteristic::Srgb,
            16 => TransferCharacteristic::Pq,   // ST.2084
            18 => TransferCharacteristic::Hlg,  // ARIB STD-B67
            _ => TransferCharacteristic::Bt709, // Unspecified/reserved
        }
    }

    /// Returns true if this represents HDR content.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::Av1ColorConfig;
    ///
    /// let hdr = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
    /// assert!(hdr.is_hdr());
    ///
    /// let sdr = Av1ColorConfig::new(8, false, 1, 1, 1, false, true, true);
    /// assert!(!sdr.is_hdr());
    /// ```
    #[must_use]
    pub const fn is_hdr(&self) -> bool {
        matches!(self.transfer_characteristics, 16 | 18) // PQ or HLG
    }

    /// Returns true if this is 4:2:0 chroma subsampling.
    #[must_use]
    pub const fn is_420(&self) -> bool {
        self.subsampling_x && self.subsampling_y
    }

    /// Returns true if this is 4:2:2 chroma subsampling.
    #[must_use]
    pub const fn is_422(&self) -> bool {
        self.subsampling_x && !self.subsampling_y
    }

    /// Returns true if this is 4:4:4 (no chroma subsampling).
    #[must_use]
    pub const fn is_444(&self) -> bool {
        !self.subsampling_x && !self.subsampling_y
    }
}

/// Matroska/WebM color elements.
///
/// Represents color metadata stored in Matroska/WebM containers.
///
/// # Examples
///
/// ```
/// use oximedia_core::hdr::MatroskaColorElements;
///
/// let color = MatroskaColorElements {
///     matrix_coefficients: Some(9),
///     bits_per_channel: Some(10),
///     chroma_subsampling_horz: Some(1),
///     chroma_subsampling_vert: Some(1),
///     cb_subsampling_horz: Some(1),
///     cb_subsampling_vert: Some(1),
///     chroma_siting_horz: Some(0),
///     chroma_siting_vert: Some(0),
///     range: Some(0),
///     transfer_characteristics: Some(16),
///     primaries: Some(9),
///     max_cll: Some(1000),
///     max_fall: Some(400),
///     primary_r_chromaticity_x: Some(0.708),
///     primary_r_chromaticity_y: Some(0.292),
///     primary_g_chromaticity_x: Some(0.170),
///     primary_g_chromaticity_y: Some(0.797),
///     primary_b_chromaticity_x: Some(0.131),
///     primary_b_chromaticity_y: Some(0.046),
///     white_point_chromaticity_x: Some(0.3127),
///     white_point_chromaticity_y: Some(0.3290),
///     luminance_max: Some(1000.0),
///     luminance_min: Some(0.005),
/// };
///
/// assert!(color.is_hdr());
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatroskaColorElements {
    /// Matrix coefficients.
    pub matrix_coefficients: Option<u8>,
    /// Bits per color channel.
    pub bits_per_channel: Option<u8>,
    /// Horizontal chroma subsampling.
    pub chroma_subsampling_horz: Option<u8>,
    /// Vertical chroma subsampling.
    pub chroma_subsampling_vert: Option<u8>,
    /// Horizontal Cb subsampling.
    pub cb_subsampling_horz: Option<u8>,
    /// Vertical Cb subsampling.
    pub cb_subsampling_vert: Option<u8>,
    /// Horizontal chroma siting.
    pub chroma_siting_horz: Option<u8>,
    /// Vertical chroma siting.
    pub chroma_siting_vert: Option<u8>,
    /// Color range (0 = limited, 1 = full).
    pub range: Option<u8>,
    /// Transfer characteristics.
    pub transfer_characteristics: Option<u8>,
    /// Color primaries.
    pub primaries: Option<u8>,
    /// Maximum content light level.
    pub max_cll: Option<u16>,
    /// Maximum frame-average light level.
    pub max_fall: Option<u16>,
    /// Red primary x chromaticity.
    pub primary_r_chromaticity_x: Option<f64>,
    /// Red primary y chromaticity.
    pub primary_r_chromaticity_y: Option<f64>,
    /// Green primary x chromaticity.
    pub primary_g_chromaticity_x: Option<f64>,
    /// Green primary y chromaticity.
    pub primary_g_chromaticity_y: Option<f64>,
    /// Blue primary x chromaticity.
    pub primary_b_chromaticity_x: Option<f64>,
    /// Blue primary y chromaticity.
    pub primary_b_chromaticity_y: Option<f64>,
    /// White point x chromaticity.
    pub white_point_chromaticity_x: Option<f64>,
    /// White point y chromaticity.
    pub white_point_chromaticity_y: Option<f64>,
    /// Maximum luminance.
    pub luminance_max: Option<f64>,
    /// Minimum luminance.
    pub luminance_min: Option<f64>,
}

impl MatroskaColorElements {
    /// Creates a new Matroska color elements container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            matrix_coefficients: None,
            bits_per_channel: None,
            chroma_subsampling_horz: None,
            chroma_subsampling_vert: None,
            cb_subsampling_horz: None,
            cb_subsampling_vert: None,
            chroma_siting_horz: None,
            chroma_siting_vert: None,
            range: None,
            transfer_characteristics: None,
            primaries: None,
            max_cll: None,
            max_fall: None,
            primary_r_chromaticity_x: None,
            primary_r_chromaticity_y: None,
            primary_g_chromaticity_x: None,
            primary_g_chromaticity_y: None,
            primary_b_chromaticity_x: None,
            primary_b_chromaticity_y: None,
            white_point_chromaticity_x: None,
            white_point_chromaticity_y: None,
            luminance_max: None,
            luminance_min: None,
        }
    }

    /// Extracts color primaries from the metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::hdr::{MatroskaColorElements, ColorPrimaries};
    ///
    /// let mut color = MatroskaColorElements::new();
    /// color.primaries = Some(9); // BT.2020
    /// assert_eq!(color.to_color_primaries(), ColorPrimaries::BT2020);
    /// ```
    #[must_use]
    pub fn to_color_primaries(&self) -> ColorPrimaries {
        if let Some(primaries) = self.primaries {
            match primaries {
                1 => ColorPrimaries::BT709,
                4 => ColorPrimaries::Bt470M,
                5 => ColorPrimaries::Bt470Bg,
                6 => ColorPrimaries::Smpte170M,
                7 => ColorPrimaries::Smpte240M,
                8 => ColorPrimaries::Film,
                9 => ColorPrimaries::BT2020,
                10 => ColorPrimaries::BT709,
                11 => ColorPrimaries::DciP3,
                12 => ColorPrimaries::DisplayP3,
                _ => ColorPrimaries::BT709,
            }
        } else {
            ColorPrimaries::BT709
        }
    }

    /// Extracts transfer characteristic from the metadata.
    #[must_use]
    pub fn to_transfer_characteristic(&self) -> TransferCharacteristic {
        if let Some(transfer) = self.transfer_characteristics {
            match transfer {
                1 | 6 | 14 | 15 => TransferCharacteristic::Bt709,
                8 => TransferCharacteristic::Linear,
                13 => TransferCharacteristic::Srgb,
                16 => TransferCharacteristic::Pq,
                18 => TransferCharacteristic::Hlg,
                _ => TransferCharacteristic::Bt709,
            }
        } else {
            TransferCharacteristic::Bt709
        }
    }

    /// Extracts mastering display color volume if available.
    #[must_use]
    pub fn to_mdcv(&self) -> Option<MasteringDisplayColorVolume> {
        let primaries = Primaries {
            red: (
                self.primary_r_chromaticity_x?,
                self.primary_r_chromaticity_y?,
            ),
            green: (
                self.primary_g_chromaticity_x?,
                self.primary_g_chromaticity_y?,
            ),
            blue: (
                self.primary_b_chromaticity_x?,
                self.primary_b_chromaticity_y?,
            ),
        };

        let white_point = WhitePoint::Custom(
            self.white_point_chromaticity_x?,
            self.white_point_chromaticity_y?,
        );

        Some(MasteringDisplayColorVolume {
            display_primaries: primaries,
            white_point,
            max_luminance: self.luminance_max?,
            min_luminance: self.luminance_min?,
        })
    }

    /// Extracts content light level if available.
    #[must_use]
    pub fn to_cll(&self) -> Option<ContentLightLevel> {
        Some(ContentLightLevel {
            max_cll: self.max_cll?,
            max_fall: self.max_fall?,
        })
    }

    /// Returns true if this represents HDR content.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer_characteristics, Some(16 | 18))
    }
}

/// HEVC SEI parser (structure only).
///
/// Note: This is a placeholder structure since `OxiMedia` does not support
/// HEVC/H.265 (patent-encumbered codec). This is provided for reference
/// and documentation purposes only.
#[derive(Clone, Debug, PartialEq)]
pub struct HevcSeiParser {
    /// Private field to prevent construction.
    _private: (),
}

impl HevcSeiParser {
    /// HEVC is not supported (patent-encumbered).
    ///
    /// This method always returns an error.
    ///
    /// # Errors
    ///
    /// Always returns an error indicating HEVC is not supported.
    pub fn parse_sei_message(_data: &[u8]) -> Result<(), &'static str> {
        Err("HEVC/H.265 is not supported (patent-encumbered codec)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_color_config() {
        let config = Vp9ColorConfig::new(10, 9, false, true, true);
        assert_eq!(config.bit_depth, 10);
        assert_eq!(config.color_space, 9);
        assert!(!config.color_range);
        assert!(config.is_420());
        assert!(!config.is_422());
        assert!(!config.is_444());
    }

    #[test]
    fn test_vp9_to_color_primaries() {
        let config = Vp9ColorConfig::new(10, 9, false, true, true);
        assert_eq!(config.to_color_primaries(), ColorPrimaries::BT2020);

        let config_709 = Vp9ColorConfig::new(8, 2, false, true, true);
        assert_eq!(config_709.to_color_primaries(), ColorPrimaries::BT709);
    }

    #[test]
    fn test_av1_color_config() {
        let config = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
        assert_eq!(config.bit_depth, 10);
        assert!(!config.mono_chrome);
        assert!(config.is_hdr());
        assert!(config.is_420());
    }

    #[test]
    fn test_av1_to_color_primaries() {
        let config = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
        assert_eq!(config.to_color_primaries(), ColorPrimaries::BT2020);
    }

    #[test]
    fn test_av1_to_transfer_characteristic() {
        let config = Av1ColorConfig::new(10, false, 9, 16, 9, false, true, true);
        assert_eq!(
            config.to_transfer_characteristic(),
            TransferCharacteristic::Pq
        );

        let hlg_config = Av1ColorConfig::new(10, false, 9, 18, 9, false, true, true);
        assert_eq!(
            hlg_config.to_transfer_characteristic(),
            TransferCharacteristic::Hlg
        );
    }

    #[test]
    fn test_matroska_color_elements() {
        let mut color = MatroskaColorElements::new();
        color.primaries = Some(9);
        color.transfer_characteristics = Some(16);
        color.max_cll = Some(1000);
        color.max_fall = Some(400);

        assert_eq!(color.to_color_primaries(), ColorPrimaries::BT2020);
        assert_eq!(
            color.to_transfer_characteristic(),
            TransferCharacteristic::Pq
        );
        assert!(color.is_hdr());

        let cll = color.to_cll().expect("CLL extraction should succeed");
        assert_eq!(cll.max_cll, 1000);
        assert_eq!(cll.max_fall, 400);
    }

    #[test]
    fn test_matroska_mdcv_extraction() {
        let mut color = MatroskaColorElements::new();
        color.primary_r_chromaticity_x = Some(0.708);
        color.primary_r_chromaticity_y = Some(0.292);
        color.primary_g_chromaticity_x = Some(0.170);
        color.primary_g_chromaticity_y = Some(0.797);
        color.primary_b_chromaticity_x = Some(0.131);
        color.primary_b_chromaticity_y = Some(0.046);
        color.white_point_chromaticity_x = Some(0.3127);
        color.white_point_chromaticity_y = Some(0.3290);
        color.luminance_max = Some(1000.0);
        color.luminance_min = Some(0.005);

        let mdcv = color.to_mdcv().expect("MDCV extraction should succeed");
        assert_eq!(mdcv.display_primaries.red, (0.708, 0.292));
        assert_eq!(mdcv.max_luminance, 1000.0);
        assert_eq!(mdcv.min_luminance, 0.005);
    }

    #[test]
    fn test_hevc_sei_parser_unsupported() {
        let result = HevcSeiParser::parse_sei_message(&[]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "HEVC/H.265 is not supported (patent-encumbered codec)"
        );
    }
}
