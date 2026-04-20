//! Color metadata helpers built on the compact core type enums.

#![allow(dead_code)]

pub use crate::types::{ColorPrimaries, ColorRange, MatrixCoefficients, TransferCharacteristics};

// ─────────────────────────────────────────────────────────────────────────────
// Colour space descriptor
// ─────────────────────────────────────────────────────────────────────────────

/// Full colour space descriptor combining primaries and matrix coefficients.
///
/// Attach this to a frame or stream to fully characterise its colour encoding.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct ColorSpace {
    /// Chromaticity of the display primaries.
    #[serde(default)]
    pub primaries: ColorPrimaries,
    /// Matrix coefficients for YCbCr ↔ RGB conversion.
    #[serde(default)]
    pub matrix: MatrixCoefficients,
    /// Whether the signal uses full range (0–255) or limited/studio range (16–235/240).
    #[serde(default)]
    pub full_range: bool,
    /// Transfer characteristics for the signal.
    #[serde(default)]
    pub transfer: TransferCharacteristics,
    /// Video range expressed as an enum for newer call sites.
    #[serde(default)]
    pub range: ColorRange,
}

impl ColorSpace {
    /// Constructs a new `ColorSpace` descriptor.
    #[must_use]
    pub const fn new(
        primaries: ColorPrimaries,
        transfer: TransferCharacteristics,
        matrix: MatrixCoefficients,
        range: ColorRange,
    ) -> Self {
        Self {
            primaries,
            matrix,
            full_range: matches!(range, ColorRange::Full),
            transfer,
            range,
        }
    }

    /// Returns the canonical BT.709 (sRGB / HDTV) colour space with limited range.
    ///
    /// This is the default for most HD content.
    #[must_use]
    pub const fn bt709() -> Self {
        Self::new(
            ColorPrimaries::Bt709,
            TransferCharacteristics::Bt709,
            MatrixCoefficients::Bt709,
            ColorRange::Limited,
        )
    }

    /// Returns the BT.2020 (UHDTV / HDR) colour space with limited range.
    #[must_use]
    pub const fn bt2020() -> Self {
        Self::new(
            ColorPrimaries::Bt2020,
            TransferCharacteristics::Smpte2084,
            MatrixCoefficients::Bt2020Ncl,
            ColorRange::Limited,
        )
    }

    /// Returns the BT.601 (SD PAL) colour space with limited range.
    #[must_use]
    pub const fn bt601_625() -> Self {
        Self::new(
            ColorPrimaries::Smpte170M,
            TransferCharacteristics::Bt709,
            MatrixCoefficients::Bt470Bg,
            ColorRange::Limited,
        )
    }

    /// Returns the BT.601 (SD NTSC) colour space with limited range.
    #[must_use]
    pub const fn bt601_525() -> Self {
        Self::new(
            ColorPrimaries::Smpte170M,
            TransferCharacteristics::Bt709,
            MatrixCoefficients::Bt601,
            ColorRange::Limited,
        )
    }

    /// Returns the sRGB colour space (BT.709 primaries, identity matrix, full range).
    ///
    /// Suitable for web / desktop image output.
    #[must_use]
    pub const fn srgb() -> Self {
        Self::new(
            ColorPrimaries::Bt709,
            TransferCharacteristics::Srgb,
            MatrixCoefficients::Identity,
            ColorRange::Full,
        )
    }

    /// Returns `true` if this colour space is HDR-capable.
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.transfer.is_hdr()
    }
}

impl std::fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "primaries={} matrix={} range={}",
            color_primaries_name(self.primaries),
            matrix_coefficients_name(self.matrix),
            if self.range == ColorRange::Full {
                "full"
            } else {
                "limited"
            }
        )
    }
}

const fn color_primaries_name(value: ColorPrimaries) -> &'static str {
    match value {
        ColorPrimaries::Bt709 => "bt709",
        ColorPrimaries::Unspecified => "unspecified",
        ColorPrimaries::Smpte170M => "smpte170m",
        ColorPrimaries::Smpte240M => "smpte240m",
        ColorPrimaries::GenericFilm => "generic-film",
        ColorPrimaries::Bt2020 => "bt2020",
        ColorPrimaries::DciP3 => "dci-p3",
        ColorPrimaries::P3D65 => "p3-d65",
        ColorPrimaries::Ebu3213 => "ebu3213",
    }
}

const fn matrix_coefficients_name(value: MatrixCoefficients) -> &'static str {
    match value {
        MatrixCoefficients::Identity => "identity",
        MatrixCoefficients::Bt709 => "bt709",
        MatrixCoefficients::Unspecified => "unspecified",
        MatrixCoefficients::FccTitle47 => "fcc-title-47",
        MatrixCoefficients::Bt470Bg => "bt470bg",
        MatrixCoefficients::Bt601 => "bt601",
        MatrixCoefficients::Smpte240M => "smpte240m",
        MatrixCoefficients::YCgCo => "ycgco",
        MatrixCoefficients::Bt2020Ncl => "bt2020-ncl",
        MatrixCoefficients::Bt2020Cl => "bt2020-cl",
        MatrixCoefficients::Smpte2085 => "smpte2085",
        MatrixCoefficients::ChromaDerivedNcl => "chroma-derived-ncl",
        MatrixCoefficients::ChromaDerivedCl => "chroma-derived-cl",
        MatrixCoefficients::ICtCp => "ictcp",
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_primaries_h273_round_trip() {
        let cases = [
            ColorPrimaries::Bt709,
            ColorPrimaries::Bt2020,
            ColorPrimaries::DciP3,
        ];
        for p in cases {
            let code = p.to_u8();
            let decoded = ColorPrimaries::from_u8(code);
            assert_eq!(decoded, p, "round-trip failed for {p:?}");
        }
    }

    #[test]
    fn test_color_primaries_unknown_code() {
        assert_eq!(ColorPrimaries::from_u8(255), ColorPrimaries::Unspecified);
        assert_eq!(ColorPrimaries::from_u8(3), ColorPrimaries::Unspecified);
    }

    #[test]
    fn test_matrix_coefficients_h273_round_trip() {
        let cases = [
            MatrixCoefficients::Identity,
            MatrixCoefficients::Bt709,
            MatrixCoefficients::Unspecified,
            MatrixCoefficients::Bt2020Ncl,
            MatrixCoefficients::Bt2020Cl,
            MatrixCoefficients::ICtCp,
        ];
        for m in cases {
            let code = m.to_u8();
            let decoded = MatrixCoefficients::from_u8(code);
            assert_eq!(decoded, m, "round-trip failed for {m:?}");
        }
    }

    #[test]
    fn test_matrix_unknown_code() {
        assert_eq!(
            MatrixCoefficients::from_u8(200),
            MatrixCoefficients::Unspecified
        );
    }

    #[test]
    fn test_color_space_presets() {
        let bt709 = ColorSpace::bt709();
        assert_eq!(bt709.primaries, ColorPrimaries::Bt709);
        assert_eq!(bt709.transfer, TransferCharacteristics::Bt709);
        assert_eq!(bt709.matrix, MatrixCoefficients::Bt709);
        assert!(!bt709.full_range);
        assert!(!bt709.is_hdr());

        let bt2020 = ColorSpace::bt2020();
        assert!(bt2020.is_hdr());
        assert_eq!(bt2020.transfer, TransferCharacteristics::Smpte2084);

        let srgb = ColorSpace::srgb();
        assert!(srgb.full_range);
        assert_eq!(srgb.range, ColorRange::Full);
        assert_eq!(srgb.matrix, MatrixCoefficients::Identity);
    }

    #[test]
    fn test_color_space_display() {
        let cs = ColorSpace::bt709();
        let s = format!("{cs}");
        assert!(s.contains("bt709"));
        assert!(s.contains("limited"));
    }

    #[test]
    fn test_color_space_default() {
        let cs = ColorSpace::default();
        assert_eq!(cs.primaries, ColorPrimaries::Unspecified);
        assert_eq!(cs.matrix, MatrixCoefficients::Unspecified);
        assert!(!cs.full_range);
        assert_eq!(cs.transfer, TransferCharacteristics::Unspecified);
        assert_eq!(cs.range, ColorRange::Limited);
    }

    #[test]
    fn test_bt601_presets() {
        let pal = ColorSpace::bt601_625();
        assert_eq!(pal.primaries, ColorPrimaries::Smpte170M);

        let ntsc = ColorSpace::bt601_525();
        assert_eq!(ntsc.primaries, ColorPrimaries::Smpte170M);
    }
}
