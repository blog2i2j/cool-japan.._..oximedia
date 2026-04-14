//! Color primaries and matrix coefficients metadata for pixel formats.
//!
//! This module provides [`ColorPrimariesId`], [`MatrixCoeffId`],
//! [`TransferCharacteristicId`], and [`PixelFormatColorDescriptor`] —
//! a fully-typed colour-space descriptor that can be attached to any
//! pixel-format pipeline stage.
//!
//! # Design
//!
//! The numeric codes follow the MPEG-4 / ITU-T H.273 / ISO 23001-8
//! specification (the same table used by H.264, HEVC, VP9, and AV1).
//! This module does **not** implement any colour-conversion maths; for that
//! see [`crate::hdr`] and [`crate::convert`].  It only carries typed
//! descriptors so that pipeline stages can reason about colour-space
//! semantics without raw integer comparisons.
//!
//! # Examples
//!
//! ```
//! use oximedia_core::pixel_format_color::{
//!     ColorPrimariesId, MatrixCoeffId, TransferCharacteristicId,
//!     PixelFormatColorDescriptor,
//! };
//!
//! let desc = PixelFormatColorDescriptor::bt709();
//! assert_eq!(desc.primaries, ColorPrimariesId::Bt709);
//! assert_eq!(desc.matrix, MatrixCoeffId::Bt709);
//! assert!(desc.is_hd());
//! assert!(!desc.is_hdr());
//! ```

#![allow(dead_code)]

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// ColorPrimariesId
// ─────────────────────────────────────────────────────────────────────────────

/// ISO 23001-8 / H.273 colour primaries identifier.
///
/// Values match the `colour_primaries` syntax element in HEVC / AV1 / VP9
/// bitstreams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ColorPrimariesId {
    /// ITU-R BT.709 — used for HD and sRGB content. Code 1.
    Bt709,
    /// Unspecified. Code 2.
    Unspecified,
    /// ITU-R BT.470M (NTSC analogue). Code 4.
    Bt470M,
    /// ITU-R BT.470BG (PAL/SECAM analogue). Code 5.
    Bt470Bg,
    /// ITU-R BT.601 (SMPTE 170M) — standard definition. Code 6.
    Bt601,
    /// SMPTE ST 240 (early HDTV). Code 7.
    Smpte240,
    /// Generic film / equal energy illuminant. Code 8.
    GenericFilm,
    /// ITU-R BT.2020 — used for UHD / HDR content. Code 9.
    Bt2020,
    /// SMPTE ST 428 (DCI P3 D65 white point). Code 10.
    Smpte428,
    /// DCI P3 (SMPTE RP 431-2). Code 11.
    P3Dci,
    /// Display P3 (P3 with D65 white point; Apple). Code 12.
    P3D65,
    /// Custom / unknown primaries.
    Custom(u8),
}

impl ColorPrimariesId {
    /// Returns the numeric ISO 23001-8 code for this primaries identifier.
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::Unspecified => 2,
            Self::Bt470M => 4,
            Self::Bt470Bg => 5,
            Self::Bt601 => 6,
            Self::Smpte240 => 7,
            Self::GenericFilm => 8,
            Self::Bt2020 => 9,
            Self::Smpte428 => 10,
            Self::P3Dci => 11,
            Self::P3D65 => 12,
            Self::Custom(c) => c,
        }
    }

    /// Constructs a [`ColorPrimariesId`] from its numeric code.
    ///
    /// Unknown codes are mapped to [`ColorPrimariesId::Custom`].
    #[must_use]
    pub fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Bt709,
            2 => Self::Unspecified,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::GenericFilm,
            9 => Self::Bt2020,
            10 => Self::Smpte428,
            11 => Self::P3Dci,
            12 => Self::P3D65,
            other => Self::Custom(other),
        }
    }

    /// Returns `true` if these primaries are suitable for HDR/WCG content.
    ///
    /// Currently only BT.2020 is considered HDR-capable; all P3 variants are
    /// wide colour gamut but not HDR in isolation.
    #[must_use]
    pub fn is_wide_color_gamut(self) -> bool {
        matches!(self, Self::Bt2020 | Self::P3Dci | Self::P3D65)
    }

    /// Returns a short human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bt709 => "BT.709",
            Self::Unspecified => "Unspecified",
            Self::Bt470M => "BT.470M",
            Self::Bt470Bg => "BT.470BG",
            Self::Bt601 => "BT.601",
            Self::Smpte240 => "SMPTE 240",
            Self::GenericFilm => "Generic Film",
            Self::Bt2020 => "BT.2020",
            Self::Smpte428 => "SMPTE ST 428",
            Self::P3Dci => "DCI P3",
            Self::P3D65 => "Display P3",
            Self::Custom(_) => "Custom",
        }
    }
}

impl fmt::Display for ColorPrimariesId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label(), self.code())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransferCharacteristicId
// ─────────────────────────────────────────────────────────────────────────────

/// ISO 23001-8 / H.273 transfer characteristics (opto-electronic transfer
/// function / EOTF) identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TransferCharacteristicId {
    /// ITU-R BT.709 transfer function. Code 1.
    Bt709,
    /// Unspecified. Code 2.
    Unspecified,
    /// ITU-R BT.470M (gamma 2.2). Code 4.
    Bt470M,
    /// ITU-R BT.470BG (gamma 2.8). Code 5.
    Bt470Bg,
    /// ITU-R BT.601 / SMPTE ST 170M. Code 6.
    Bt601,
    /// SMPTE ST 240 (1987 HDTV). Code 7.
    Smpte240,
    /// Linear gamma (1:1). Code 8.
    Linear,
    /// Logarithmic (100:1 range). Code 9.
    Log100,
    /// Logarithmic (316.22:1 range). Code 10.
    Log316,
    /// IEC 61966-2-4 (xvYCC). Code 11.
    Xvycc,
    /// ITU-R BT.1361 extended colour gamut. Code 12.
    Bt1361,
    /// IEC 61966-2-1 (sRGB / sYCC). Code 13.
    Srgb,
    /// ITU-R BT.2020 10-bit. Code 14.
    Bt2020Ten,
    /// ITU-R BT.2020 12-bit. Code 15.
    Bt2020Twelve,
    /// SMPTE ST 2084 (PQ / HDR10). Code 16.
    Pq,
    /// SMPTE ST 428 (DCI). Code 17.
    Smpte428,
    /// ARIB STD-B67 (HLG — Hybrid Log-Gamma). Code 18.
    Hlg,
    /// Custom / unknown.
    Custom(u8),
}

impl TransferCharacteristicId {
    /// Returns the numeric ISO 23001-8 code.
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Self::Bt709 => 1,
            Self::Unspecified => 2,
            Self::Bt470M => 4,
            Self::Bt470Bg => 5,
            Self::Bt601 => 6,
            Self::Smpte240 => 7,
            Self::Linear => 8,
            Self::Log100 => 9,
            Self::Log316 => 10,
            Self::Xvycc => 11,
            Self::Bt1361 => 12,
            Self::Srgb => 13,
            Self::Bt2020Ten => 14,
            Self::Bt2020Twelve => 15,
            Self::Pq => 16,
            Self::Smpte428 => 17,
            Self::Hlg => 18,
            Self::Custom(c) => c,
        }
    }

    /// Constructs a [`TransferCharacteristicId`] from its numeric code.
    #[must_use]
    pub fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Bt709,
            2 => Self::Unspecified,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::Linear,
            9 => Self::Log100,
            10 => Self::Log316,
            11 => Self::Xvycc,
            12 => Self::Bt1361,
            13 => Self::Srgb,
            14 => Self::Bt2020Ten,
            15 => Self::Bt2020Twelve,
            16 => Self::Pq,
            17 => Self::Smpte428,
            18 => Self::Hlg,
            other => Self::Custom(other),
        }
    }

    /// Returns `true` if this transfer function is HDR (PQ or HLG).
    #[must_use]
    pub fn is_hdr(self) -> bool {
        matches!(self, Self::Pq | Self::Hlg)
    }

    /// Returns a short label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Bt709 => "BT.709",
            Self::Unspecified => "Unspecified",
            Self::Bt470M => "BT.470M",
            Self::Bt470Bg => "BT.470BG",
            Self::Bt601 => "BT.601",
            Self::Smpte240 => "SMPTE 240",
            Self::Linear => "Linear",
            Self::Log100 => "Log100",
            Self::Log316 => "Log316",
            Self::Xvycc => "xvYCC",
            Self::Bt1361 => "BT.1361",
            Self::Srgb => "sRGB",
            Self::Bt2020Ten => "BT.2020 10-bit",
            Self::Bt2020Twelve => "BT.2020 12-bit",
            Self::Pq => "PQ (HDR10)",
            Self::Smpte428 => "SMPTE ST 428",
            Self::Hlg => "HLG",
            Self::Custom(_) => "Custom",
        }
    }
}

impl fmt::Display for TransferCharacteristicId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label(), self.code())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MatrixCoeffId
// ─────────────────────────────────────────────────────────────────────────────

/// ISO 23001-8 / H.273 matrix coefficients identifier.
///
/// Describes the luma/chroma weighting used when deriving Y'CbCr from R'G'B'.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MatrixCoeffId {
    /// Identity (GBR / ICtCp). Code 0.
    Identity,
    /// ITU-R BT.709. Code 1.
    Bt709,
    /// Unspecified. Code 2.
    Unspecified,
    /// ITU-R BT.470M (NTSC legacy). Code 4.
    Bt470M,
    /// ITU-R BT.470BG / BT.601 625-line. Code 5.
    Bt470Bg,
    /// ITU-R BT.601 525-line (SMPTE ST 170M). Code 6.
    Bt601,
    /// SMPTE ST 240. Code 7.
    Smpte240,
    /// YCgCo. Code 8.
    Ycgco,
    /// ITU-R BT.2020 non-constant luminance. Code 9.
    Bt2020Ncl,
    /// ITU-R BT.2020 constant luminance. Code 10.
    Bt2020Cl,
    /// SMPTE ST 2085 (Y'D'zD'x). Code 11.
    Smpte2085,
    /// Chromaticity-derived non-constant luminance. Code 12.
    ChromaDerivedNcl,
    /// Chromaticity-derived constant luminance. Code 13.
    ChromaDerivedCl,
    /// ICtCp (BT.2100). Code 14.
    Ictcp,
    /// Custom / unknown.
    Custom(u8),
}

impl MatrixCoeffId {
    /// Returns the numeric ISO 23001-8 code.
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Self::Identity => 0,
            Self::Bt709 => 1,
            Self::Unspecified => 2,
            Self::Bt470M => 4,
            Self::Bt470Bg => 5,
            Self::Bt601 => 6,
            Self::Smpte240 => 7,
            Self::Ycgco => 8,
            Self::Bt2020Ncl => 9,
            Self::Bt2020Cl => 10,
            Self::Smpte2085 => 11,
            Self::ChromaDerivedNcl => 12,
            Self::ChromaDerivedCl => 13,
            Self::Ictcp => 14,
            Self::Custom(c) => c,
        }
    }

    /// Constructs a [`MatrixCoeffId`] from its numeric code.
    #[must_use]
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::Identity,
            1 => Self::Bt709,
            2 => Self::Unspecified,
            4 => Self::Bt470M,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240,
            8 => Self::Ycgco,
            9 => Self::Bt2020Ncl,
            10 => Self::Bt2020Cl,
            11 => Self::Smpte2085,
            12 => Self::ChromaDerivedNcl,
            13 => Self::ChromaDerivedCl,
            14 => Self::Ictcp,
            other => Self::Custom(other),
        }
    }

    /// Returns a short label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Identity => "Identity (GBR)",
            Self::Bt709 => "BT.709",
            Self::Unspecified => "Unspecified",
            Self::Bt470M => "BT.470M",
            Self::Bt470Bg => "BT.470BG",
            Self::Bt601 => "BT.601",
            Self::Smpte240 => "SMPTE 240",
            Self::Ycgco => "YCgCo",
            Self::Bt2020Ncl => "BT.2020 NCL",
            Self::Bt2020Cl => "BT.2020 CL",
            Self::Smpte2085 => "SMPTE ST 2085",
            Self::ChromaDerivedNcl => "Chroma-derived NCL",
            Self::ChromaDerivedCl => "Chroma-derived CL",
            Self::Ictcp => "ICtCp",
            Self::Custom(_) => "Custom",
        }
    }
}

impl fmt::Display for MatrixCoeffId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.label(), self.code())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VideoRange
// ─────────────────────────────────────────────────────────────────────────────

/// Indicates whether luma/chroma values use the full `[0, 255]` range or the
/// "studio swing" (limited) range `[16, 235]` / `[16, 240]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum VideoRange {
    /// Limited / broadcast range — luma in `[16, 235]`, chroma in `[16, 240]`.
    #[default]
    Limited,
    /// Full range — luma and chroma both in `[0, 255]`.
    Full,
}

impl VideoRange {
    /// Returns `true` for full-range content.
    #[must_use]
    pub fn is_full(self) -> bool {
        matches!(self, Self::Full)
    }

    /// Returns the luma black level for this range.
    #[must_use]
    pub fn luma_black(self) -> u8 {
        match self {
            Self::Limited => 16,
            Self::Full => 0,
        }
    }

    /// Returns the luma white level for 8-bit content.
    #[must_use]
    pub fn luma_white(self) -> u8 {
        match self {
            Self::Limited => 235,
            Self::Full => 255,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PixelFormatColorDescriptor
// ─────────────────────────────────────────────────────────────────────────────

/// A colour-space descriptor that can be associated with any pixel-format
/// pipeline stage.
///
/// Combines [`ColorPrimariesId`], [`TransferCharacteristicId`],
/// [`MatrixCoeffId`], and [`VideoRange`] into one struct following the
/// ISO 23001-8 / H.273 parameterisation used by all modern video codecs.
///
/// # Examples
///
/// ```
/// use oximedia_core::pixel_format_color::{
///     ColorPrimariesId, MatrixCoeffId, TransferCharacteristicId,
///     PixelFormatColorDescriptor, VideoRange,
/// };
///
/// let hdr10 = PixelFormatColorDescriptor::hdr10();
/// assert!(hdr10.transfer.is_hdr());
/// assert!(hdr10.primaries.is_wide_color_gamut());
/// assert!(!hdr10.is_sd());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PixelFormatColorDescriptor {
    /// Colour primaries (chromaticity of the primaries).
    pub primaries: ColorPrimariesId,
    /// Transfer characteristics (opto-electronic transfer function).
    pub transfer: TransferCharacteristicId,
    /// Matrix coefficients (luma/chroma derivation from R'G'B').
    pub matrix: MatrixCoeffId,
    /// Video range (limited vs. full swing).
    pub range: VideoRange,
}

impl PixelFormatColorDescriptor {
    /// Creates a descriptor with the given components.
    #[must_use]
    pub const fn new(
        primaries: ColorPrimariesId,
        transfer: TransferCharacteristicId,
        matrix: MatrixCoeffId,
        range: VideoRange,
    ) -> Self {
        Self {
            primaries,
            transfer,
            matrix,
            range,
        }
    }

    /// Creates a descriptor from raw ISO 23001-8 numeric codes.
    ///
    /// Unknown codes are mapped to the `Custom` variants.
    #[must_use]
    pub fn from_codes(
        primaries_code: u8,
        transfer_code: u8,
        matrix_code: u8,
        full_range: bool,
    ) -> Self {
        Self {
            primaries: ColorPrimariesId::from_code(primaries_code),
            transfer: TransferCharacteristicId::from_code(transfer_code),
            matrix: MatrixCoeffId::from_code(matrix_code),
            range: if full_range {
                VideoRange::Full
            } else {
                VideoRange::Limited
            },
        }
    }

    // ── Well-known preset constructors ────────────────────────────────────────

    /// Standard HD colour space: BT.709 primaries, BT.709 transfer, BT.709
    /// matrix, limited range.
    #[must_use]
    pub const fn bt709() -> Self {
        Self::new(
            ColorPrimariesId::Bt709,
            TransferCharacteristicId::Bt709,
            MatrixCoeffId::Bt709,
            VideoRange::Limited,
        )
    }

    /// Standard-definition NTSC colour space: BT.601 primaries and matrix,
    /// BT.601 transfer, limited range.
    #[must_use]
    pub const fn bt601_ntsc() -> Self {
        Self::new(
            ColorPrimariesId::Bt601,
            TransferCharacteristicId::Bt601,
            MatrixCoeffId::Bt601,
            VideoRange::Limited,
        )
    }

    /// Standard-definition PAL/SECAM colour space: BT.470BG primaries and
    /// matrix, BT.601 transfer, limited range.
    #[must_use]
    pub const fn bt601_pal() -> Self {
        Self::new(
            ColorPrimariesId::Bt470Bg,
            TransferCharacteristicId::Bt601,
            MatrixCoeffId::Bt470Bg,
            VideoRange::Limited,
        )
    }

    /// HDR10 colour space: BT.2020 primaries, PQ transfer, BT.2020 NCL
    /// matrix, limited range.
    #[must_use]
    pub const fn hdr10() -> Self {
        Self::new(
            ColorPrimariesId::Bt2020,
            TransferCharacteristicId::Pq,
            MatrixCoeffId::Bt2020Ncl,
            VideoRange::Limited,
        )
    }

    /// HLG (Hybrid Log-Gamma) colour space: BT.2020 primaries, HLG transfer,
    /// BT.2020 NCL matrix, limited range.
    #[must_use]
    pub const fn hlg() -> Self {
        Self::new(
            ColorPrimariesId::Bt2020,
            TransferCharacteristicId::Hlg,
            MatrixCoeffId::Bt2020Ncl,
            VideoRange::Limited,
        )
    }

    /// sRGB colour space: BT.709 primaries, sRGB transfer, Identity matrix,
    /// full range.
    #[must_use]
    pub const fn srgb() -> Self {
        Self::new(
            ColorPrimariesId::Bt709,
            TransferCharacteristicId::Srgb,
            MatrixCoeffId::Identity,
            VideoRange::Full,
        )
    }

    // ── Query helpers ─────────────────────────────────────────────────────────

    /// Returns `true` if this descriptor represents HDR content (PQ or HLG
    /// transfer function).
    #[must_use]
    pub fn is_hdr(self) -> bool {
        self.transfer.is_hdr()
    }

    /// Returns `true` if the primaries are HD (BT.709) or better.
    ///
    /// Covers BT.709, BT.2020, P3 variants, and SMPTE 428/240.
    #[must_use]
    pub fn is_hd(self) -> bool {
        matches!(
            self.primaries,
            ColorPrimariesId::Bt709
                | ColorPrimariesId::Bt2020
                | ColorPrimariesId::P3Dci
                | ColorPrimariesId::P3D65
                | ColorPrimariesId::Smpte240
                | ColorPrimariesId::Smpte428
        )
    }

    /// Returns `true` if this is a standard-definition colour space
    /// (BT.601 or BT.470 primaries).
    #[must_use]
    pub fn is_sd(self) -> bool {
        matches!(
            self.primaries,
            ColorPrimariesId::Bt601 | ColorPrimariesId::Bt470M | ColorPrimariesId::Bt470Bg
        )
    }

    /// Returns `true` if the video range is full (not limited).
    #[must_use]
    pub fn is_full_range(self) -> bool {
        self.range.is_full()
    }

    /// Returns `true` if the primaries cover a wide colour gamut (BT.2020,
    /// DCI P3, or Display P3).
    #[must_use]
    pub fn is_wide_color_gamut(self) -> bool {
        self.primaries.is_wide_color_gamut()
    }

    /// Converts all components to their raw ISO 23001-8 codes.
    ///
    /// Returns `(primaries_code, transfer_code, matrix_code, full_range)`.
    #[must_use]
    pub fn to_codes(self) -> (u8, u8, u8, bool) {
        (
            self.primaries.code(),
            self.transfer.code(),
            self.matrix.code(),
            self.range.is_full(),
        )
    }
}

impl Default for PixelFormatColorDescriptor {
    /// Defaults to the BT.709 HD colour space.
    fn default() -> Self {
        Self::bt709()
    }
}

impl fmt::Display for PixelFormatColorDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "primaries={} transfer={} matrix={} range={:?}",
            self.primaries.label(),
            self.transfer.label(),
            self.matrix.label(),
            self.range,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bt709_preset_codes_are_correct() {
        let d = PixelFormatColorDescriptor::bt709();
        assert_eq!(d.primaries.code(), 1);
        assert_eq!(d.transfer.code(), 1);
        assert_eq!(d.matrix.code(), 1);
        assert_eq!(d.range, VideoRange::Limited);
    }

    #[test]
    fn bt709_is_hd_not_hdr_not_sd() {
        let d = PixelFormatColorDescriptor::bt709();
        assert!(d.is_hd());
        assert!(!d.is_hdr());
        assert!(!d.is_sd());
        assert!(!d.is_full_range());
    }

    #[test]
    fn hdr10_preset_is_hdr_and_wcg() {
        let d = PixelFormatColorDescriptor::hdr10();
        assert!(d.is_hdr());
        assert!(d.is_wide_color_gamut());
        assert!(!d.is_sd());
        assert_eq!(d.primaries, ColorPrimariesId::Bt2020);
        assert_eq!(d.transfer, TransferCharacteristicId::Pq);
        assert_eq!(d.matrix, MatrixCoeffId::Bt2020Ncl);
    }

    #[test]
    fn hlg_preset_is_hdr() {
        let d = PixelFormatColorDescriptor::hlg();
        assert!(d.is_hdr());
        assert_eq!(d.transfer, TransferCharacteristicId::Hlg);
        assert_eq!(d.transfer.code(), 18);
    }

    #[test]
    fn srgb_preset_is_full_range_identity_matrix() {
        let d = PixelFormatColorDescriptor::srgb();
        assert!(d.is_full_range());
        assert_eq!(d.matrix, MatrixCoeffId::Identity);
        assert_eq!(d.matrix.code(), 0);
        assert!(!d.is_hdr());
    }

    #[test]
    fn bt601_ntsc_is_sd() {
        let d = PixelFormatColorDescriptor::bt601_ntsc();
        assert!(d.is_sd());
        assert!(!d.is_hd());
        assert_eq!(d.primaries, ColorPrimariesId::Bt601);
    }

    #[test]
    fn bt601_pal_is_sd() {
        let d = PixelFormatColorDescriptor::bt601_pal();
        assert!(d.is_sd());
        assert_eq!(d.primaries, ColorPrimariesId::Bt470Bg);
        assert_eq!(d.matrix, MatrixCoeffId::Bt470Bg);
    }

    #[test]
    fn from_codes_roundtrip() {
        let d = PixelFormatColorDescriptor::hdr10();
        let (pc, tc, mc, fr) = d.to_codes();
        let d2 = PixelFormatColorDescriptor::from_codes(pc, tc, mc, fr);
        assert_eq!(d, d2);
    }

    #[test]
    fn from_codes_unknown_maps_to_custom() {
        let d = PixelFormatColorDescriptor::from_codes(200, 201, 202, false);
        assert_eq!(d.primaries, ColorPrimariesId::Custom(200));
        assert_eq!(d.transfer, TransferCharacteristicId::Custom(201));
        assert_eq!(d.matrix, MatrixCoeffId::Custom(202));
    }

    #[test]
    fn color_primaries_from_code_all_known() {
        let known = [1u8, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        for &code in &known {
            let p = ColorPrimariesId::from_code(code);
            assert_eq!(p.code(), code, "roundtrip failed for code {code}");
        }
    }

    #[test]
    fn transfer_from_code_all_known() {
        let known = [1u8, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18];
        for &code in &known {
            let t = TransferCharacteristicId::from_code(code);
            assert_eq!(t.code(), code, "roundtrip failed for code {code}");
        }
    }

    #[test]
    fn matrix_from_code_all_known() {
        let known = [0u8, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        for &code in &known {
            let m = MatrixCoeffId::from_code(code);
            assert_eq!(m.code(), code, "roundtrip failed for code {code}");
        }
    }

    #[test]
    fn video_range_levels() {
        assert_eq!(VideoRange::Limited.luma_black(), 16);
        assert_eq!(VideoRange::Limited.luma_white(), 235);
        assert_eq!(VideoRange::Full.luma_black(), 0);
        assert_eq!(VideoRange::Full.luma_white(), 255);
        assert!(VideoRange::Full.is_full());
        assert!(!VideoRange::Limited.is_full());
    }

    #[test]
    fn display_impl_contains_labels() {
        let d = PixelFormatColorDescriptor::bt709();
        let s = d.to_string();
        assert!(s.contains("BT.709"));
        assert!(s.contains("Limited"));
    }

    #[test]
    fn default_descriptor_is_bt709() {
        let d = PixelFormatColorDescriptor::default();
        assert_eq!(d, PixelFormatColorDescriptor::bt709());
    }

    #[test]
    fn pq_and_hlg_are_hdr_transfers() {
        assert!(TransferCharacteristicId::Pq.is_hdr());
        assert!(TransferCharacteristicId::Hlg.is_hdr());
        assert!(!TransferCharacteristicId::Bt709.is_hdr());
        assert!(!TransferCharacteristicId::Srgb.is_hdr());
    }

    #[test]
    fn bt2020_p3_are_wcg_primaries() {
        assert!(ColorPrimariesId::Bt2020.is_wide_color_gamut());
        assert!(ColorPrimariesId::P3Dci.is_wide_color_gamut());
        assert!(ColorPrimariesId::P3D65.is_wide_color_gamut());
        assert!(!ColorPrimariesId::Bt709.is_wide_color_gamut());
        assert!(!ColorPrimariesId::Bt601.is_wide_color_gamut());
    }

    #[test]
    fn matrix_label_nonempty() {
        // Ensure every named variant returns a non-empty label.
        let variants = [
            MatrixCoeffId::Identity,
            MatrixCoeffId::Bt709,
            MatrixCoeffId::Unspecified,
            MatrixCoeffId::Bt470M,
            MatrixCoeffId::Bt470Bg,
            MatrixCoeffId::Bt601,
            MatrixCoeffId::Smpte240,
            MatrixCoeffId::Ycgco,
            MatrixCoeffId::Bt2020Ncl,
            MatrixCoeffId::Bt2020Cl,
            MatrixCoeffId::Smpte2085,
            MatrixCoeffId::ChromaDerivedNcl,
            MatrixCoeffId::ChromaDerivedCl,
            MatrixCoeffId::Ictcp,
        ];
        for v in &variants {
            assert!(!v.label().is_empty(), "label empty for {v:?}");
        }
    }

    // ── Additional tests ──────────────────────────────────────────────

    #[test]
    fn transfer_label_nonempty_all_variants() {
        let variants = [
            TransferCharacteristicId::Bt709,
            TransferCharacteristicId::Unspecified,
            TransferCharacteristicId::Bt470M,
            TransferCharacteristicId::Bt470Bg,
            TransferCharacteristicId::Bt601,
            TransferCharacteristicId::Smpte240,
            TransferCharacteristicId::Linear,
            TransferCharacteristicId::Log100,
            TransferCharacteristicId::Log316,
            TransferCharacteristicId::Xvycc,
            TransferCharacteristicId::Bt1361,
            TransferCharacteristicId::Srgb,
            TransferCharacteristicId::Bt2020Ten,
            TransferCharacteristicId::Bt2020Twelve,
            TransferCharacteristicId::Pq,
            TransferCharacteristicId::Smpte428,
            TransferCharacteristicId::Hlg,
        ];
        for v in &variants {
            assert!(!v.label().is_empty(), "label empty for {v:?}");
        }
    }

    #[test]
    fn color_primaries_label_nonempty_all_variants() {
        let variants = [
            ColorPrimariesId::Bt709,
            ColorPrimariesId::Unspecified,
            ColorPrimariesId::Bt470M,
            ColorPrimariesId::Bt470Bg,
            ColorPrimariesId::Bt601,
            ColorPrimariesId::Smpte240,
            ColorPrimariesId::GenericFilm,
            ColorPrimariesId::Bt2020,
            ColorPrimariesId::Smpte428,
            ColorPrimariesId::P3Dci,
            ColorPrimariesId::P3D65,
        ];
        for v in &variants {
            assert!(!v.label().is_empty(), "label empty for {v:?}");
        }
    }

    #[test]
    fn custom_primaries_preserves_code() {
        let p = ColorPrimariesId::Custom(99);
        assert_eq!(p.code(), 99);
        assert_eq!(p.label(), "Custom");
        assert!(!p.is_wide_color_gamut());
    }

    #[test]
    fn custom_transfer_preserves_code() {
        let t = TransferCharacteristicId::Custom(77);
        assert_eq!(t.code(), 77);
        assert!(!t.is_hdr());
        assert_eq!(t.label(), "Custom");
    }

    #[test]
    fn custom_matrix_preserves_code() {
        let m = MatrixCoeffId::Custom(55);
        assert_eq!(m.code(), 55);
        assert_eq!(m.label(), "Custom");
    }

    #[test]
    fn descriptor_to_codes_roundtrip_bt601_pal() {
        let d = PixelFormatColorDescriptor::bt601_pal();
        let (pc, tc, mc, fr) = d.to_codes();
        let d2 = PixelFormatColorDescriptor::from_codes(pc, tc, mc, fr);
        assert_eq!(d, d2);
    }

    #[test]
    fn hlg_is_bt2020_wcg() {
        let d = PixelFormatColorDescriptor::hlg();
        assert!(d.is_hdr());
        assert!(d.is_wide_color_gamut());
        assert!(d.is_hd());
        assert!(!d.is_sd());
        assert!(!d.is_full_range());
    }

    #[test]
    fn srgb_is_not_hdr_not_sd_is_full_range() {
        let d = PixelFormatColorDescriptor::srgb();
        assert!(!d.is_hdr());
        assert!(!d.is_sd());
        assert!(d.is_full_range());
        assert!(d.is_hd());
        assert_eq!(d.transfer, TransferCharacteristicId::Srgb);
    }

    #[test]
    fn video_range_default_is_limited() {
        assert_eq!(VideoRange::default(), VideoRange::Limited);
    }

    #[test]
    fn display_hdr10_contains_pq_and_bt2020() {
        let d = PixelFormatColorDescriptor::hdr10();
        let s = d.to_string();
        assert!(s.contains("PQ") || s.contains("HDR10"));
        assert!(s.contains("BT.2020") || s.contains("2020"));
    }
}
