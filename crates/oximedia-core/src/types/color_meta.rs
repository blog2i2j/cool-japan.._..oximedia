//! Compact color metadata enums used by core types.

/// Color primaries as defined in ISO/IEC 23001-8 (CICP) table 2.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
#[repr(u8)]
pub enum ColorPrimaries {
    /// BT.709 color primaries (HDTV standard).
    Bt709 = 1,
    /// Unspecified / unknown color primaries.
    #[default]
    Unspecified = 2,
    /// SMPTE 170M color primaries (NTSC standard definition).
    Smpte170M = 6,
    /// SMPTE 240M color primaries (early HDTV).
    Smpte240M = 7,
    /// Generic film color primaries.
    GenericFilm = 8,
    /// BT.2020 color primaries (ultra-high definition television).
    Bt2020 = 9,
    /// DCI-P3 color primaries (digital cinema).
    DciP3 = 11,
    /// P3-D65 color primaries (DCI-P3 with D65 white point).
    P3D65 = 12,
    /// EBU Tech 3213-E color primaries (PAL standard definition).
    Ebu3213 = 22,
}

impl ColorPrimaries {
    /// Returns the ISO/IEC 23001-8 numeric code for this color primaries value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Constructs a `ColorPrimaries` from its ISO/IEC 23001-8 numeric code.
    ///
    /// Unknown codes map to [`ColorPrimaries::Unspecified`].
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Bt709,
            6 => Self::Smpte170M,
            7 => Self::Smpte240M,
            8 => Self::GenericFilm,
            9 => Self::Bt2020,
            11 => Self::DciP3,
            12 => Self::P3D65,
            22 => Self::Ebu3213,
            _ => Self::Unspecified,
        }
    }
}

/// Transfer characteristics (opto-electronic/electro-optical transfer functions)
/// as defined in ISO/IEC 23001-8 (CICP) table 3.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
#[repr(u8)]
pub enum TransferCharacteristics {
    /// BT.709 transfer characteristics.
    Bt709 = 1,
    /// Unspecified / unknown transfer characteristics.
    #[default]
    Unspecified = 2,
    /// Assumed display gamma 2.2 (BT.470-M System M).
    Gamma22 = 4,
    /// Assumed display gamma 2.8 (BT.470-M System B, G).
    Gamma28 = 5,
    /// Linear light (no transfer function applied).
    Linear = 8,
    /// Logarithmic transfer (100:1 range).
    Log100 = 9,
    /// Logarithmic transfer (100×√10 : 1 range).
    Log100Sqrt10 = 10,
    /// IEC 61966-2-4 extended color gamut transfer characteristics (xvYCC).
    Iec61966_2_4 = 11,
    /// BT.1361 extended color gamut transfer characteristics.
    Bt1361 = 12,
    /// IEC 61966-2-1 sRGB / sYCC transfer characteristics.
    Srgb = 13,
    /// BT.2020 10-bit transfer characteristics.
    Bt2020_10 = 14,
    /// BT.2020 12-bit transfer characteristics.
    Bt2020_12 = 15,
    /// SMPTE ST 2084 (PQ) perceptual quantizer — HDR10.
    Smpte2084 = 16,
    /// SMPTE ST 428-1 (DCDM) transfer characteristics.
    Smpte428 = 17,
    /// BT.2100 Hybrid Log-Gamma (HLG) transfer characteristics.
    Hlg = 18,
}

impl TransferCharacteristics {
    /// Returns the ISO/IEC 23001-8 numeric code for this transfer characteristics value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Constructs a `TransferCharacteristics` from its ISO/IEC 23001-8 numeric code.
    ///
    /// Unknown codes map to [`TransferCharacteristics::Unspecified`].
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Bt709,
            4 => Self::Gamma22,
            5 => Self::Gamma28,
            8 => Self::Linear,
            9 => Self::Log100,
            10 => Self::Log100Sqrt10,
            11 => Self::Iec61966_2_4,
            12 => Self::Bt1361,
            13 => Self::Srgb,
            14 => Self::Bt2020_10,
            15 => Self::Bt2020_12,
            16 => Self::Smpte2084,
            17 => Self::Smpte428,
            18 => Self::Hlg,
            _ => Self::Unspecified,
        }
    }

    /// Returns `true` if this transfer function is an HDR standard (PQ, HLG, or BT.2020).
    #[must_use]
    pub const fn is_hdr(self) -> bool {
        matches!(
            self,
            Self::Smpte2084 | Self::Hlg | Self::Bt2020_10 | Self::Bt2020_12
        )
    }
}

/// Matrix coefficients used to derive luma and chroma signals from RGB primaries,
/// as defined in ISO/IEC 23001-8 (CICP) table 4.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[non_exhaustive]
#[repr(u8)]
pub enum MatrixCoefficients {
    /// Identity matrix — RGB passthrough (no conversion).
    Identity = 0,
    /// BT.709 matrix coefficients (HDTV).
    Bt709 = 1,
    /// Unspecified / unknown matrix coefficients.
    #[default]
    Unspecified = 2,
    /// FCC Title 47 Code of Federal Regulations matrix coefficients.
    FccTitle47 = 4,
    /// BT.470-BG matrix coefficients (PAL/SECAM).
    Bt470Bg = 5,
    /// BT.601 matrix coefficients (standard definition television).
    Bt601 = 6,
    /// SMPTE 240M matrix coefficients (early HDTV).
    Smpte240M = 7,
    /// YCgCo matrix coefficients (lossless integer arithmetic).
    YCgCo = 8,
    /// BT.2020 non-constant luminance matrix coefficients.
    Bt2020Ncl = 9,
    /// BT.2020 constant luminance matrix coefficients.
    Bt2020Cl = 10,
    /// SMPTE ST 2085 matrix coefficients (Y′D′BD′R).
    Smpte2085 = 11,
    /// Chromaticity-derived non-constant luminance matrix coefficients.
    ChromaDerivedNcl = 12,
    /// Chromaticity-derived constant luminance matrix coefficients.
    ChromaDerivedCl = 13,
    /// ICtCp matrix coefficients (BT.2100 HDR).
    ICtCp = 14,
}

impl MatrixCoefficients {
    /// Returns the ISO/IEC 23001-8 numeric code for this matrix coefficients value.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Constructs a `MatrixCoefficients` from its ISO/IEC 23001-8 numeric code.
    ///
    /// Unknown codes map to [`MatrixCoefficients::Unspecified`].
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Identity,
            1 => Self::Bt709,
            4 => Self::FccTitle47,
            5 => Self::Bt470Bg,
            6 => Self::Bt601,
            7 => Self::Smpte240M,
            8 => Self::YCgCo,
            9 => Self::Bt2020Ncl,
            10 => Self::Bt2020Cl,
            11 => Self::Smpte2085,
            12 => Self::ChromaDerivedNcl,
            13 => Self::ChromaDerivedCl,
            14 => Self::ICtCp,
            _ => Self::Unspecified,
        }
    }
}

/// Video color range, indicating whether the luma/chroma values use the full
/// numeric range or the limited (studio swing) range.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum ColorRange {
    /// Limited (studio swing) range: luma 16–235, chroma 16–240 for 8-bit.
    #[default]
    Limited,
    /// Full range: 0–255 for 8-bit content.
    Full,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_primaries_round_trip() {
        let value = ColorPrimaries::Bt2020;
        assert_eq!(ColorPrimaries::from_u8(value.to_u8()), value);
    }

    #[test]
    fn test_transfer_characteristics_round_trip() {
        let value = TransferCharacteristics::Smpte2084;
        assert_eq!(TransferCharacteristics::from_u8(value.to_u8()), value);
    }

    #[test]
    fn test_matrix_coefficients_round_trip() {
        let value = MatrixCoefficients::Bt2020Ncl;
        assert_eq!(MatrixCoefficients::from_u8(value.to_u8()), value);
    }

    #[test]
    fn test_transfer_characteristics_is_hdr() {
        assert!(TransferCharacteristics::Smpte2084.is_hdr());
        assert!(!TransferCharacteristics::Bt709.is_hdr());
    }

    #[test]
    fn test_color_range_default() {
        assert_eq!(ColorRange::default(), ColorRange::Limited);
    }
}
