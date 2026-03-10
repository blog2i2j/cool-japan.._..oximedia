//! High-level color space and transfer function enums plus conversion utilities.
//!
//! Provides a simplified `ColorSpaceId` enum and `TransferFunctionId` enum for
//! identifying standard color spaces and transfer functions, plus a
//! `ColorTransformUtil` for batch pixel conversion.

#![allow(dead_code)]

use crate::colorspaces::ColorSpace;
use crate::error::Result;
use crate::transforms::rgb_to_rgb;

/// Identifier for standard color spaces.
///
/// This is a lightweight enum; use [`ColorSpace`] for the full definition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ColorSpaceId {
    /// ITU-R BT.709 (HD video, same primaries as sRGB).
    Rec709,
    /// ITU-R BT.2020 (UHD / HDR video).
    Rec2020,
    /// DCI-P3 (digital cinema, DCI white point).
    P3DCI,
    /// Display P3 (DCI-P3 primaries, D65 white point).
    P3D65,
    /// ACES AP0 / ACES2065-1 (scene-referred, full gamut).
    AcesAP0,
    /// ACES AP1 / ACEScg (working space).
    AcesAP1,
    /// sRGB (consumer displays and web).
    SRGB,
    /// Scene-linear (no gamma / transfer function).
    Linear,
}

impl ColorSpaceId {
    /// Convert this identifier to a fully-defined [`ColorSpace`].
    ///
    /// # Errors
    ///
    /// Returns an error if matrix computation fails (unlikely for standard spaces).
    pub fn to_color_space(self) -> Result<ColorSpace> {
        match self {
            Self::Rec709 => ColorSpace::rec709(),
            Self::Rec2020 => ColorSpace::rec2020(),
            Self::P3DCI => ColorSpace::dci_p3(),
            Self::P3D65 => ColorSpace::display_p3(),
            Self::AcesAP0 | Self::AcesAP1 => ColorSpace::linear_rec709(), // Approximation
            Self::SRGB => ColorSpace::srgb(),
            Self::Linear => ColorSpace::linear_rec709(),
        }
    }

    /// Return a human-readable name for this color space.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rec709 => "Rec.709",
            Self::Rec2020 => "Rec.2020",
            Self::P3DCI => "DCI-P3",
            Self::P3D65 => "Display P3 (D65)",
            Self::AcesAP0 => "ACES AP0 (ACES2065-1)",
            Self::AcesAP1 => "ACES AP1 (ACEScg)",
            Self::SRGB => "sRGB",
            Self::Linear => "Linear",
        }
    }
}

/// Identifier for standard transfer functions (EOTF / OETF).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransferFunctionId {
    /// sRGB piecewise gamma (IEC 61966-2-1).
    SRGB,
    /// Pure gamma 2.2 curve.
    Gamma22,
    /// ITU-R BT.2100 Perceptual Quantizer (PQ / SMPTE ST 2084).
    PQ,
    /// ITU-R BT.2100 Hybrid Log-Gamma (HLG / ARIB STD-B67).
    HLG,
    /// Scene-linear (no encoding).
    Linear,
    /// ARRI LogC (EI 800).
    LogC,
    /// Sony S-Log3.
    SLog3,
}

impl TransferFunctionId {
    /// Return a human-readable name for this transfer function.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::SRGB => "sRGB",
            Self::Gamma22 => "Gamma 2.2",
            Self::PQ => "PQ (SMPTE ST 2084)",
            Self::HLG => "HLG (ARIB STD-B67)",
            Self::Linear => "Linear",
            Self::LogC => "ARRI LogC",
            Self::SLog3 => "Sony S-Log3",
        }
    }

    /// Apply the OETF (scene-linear → encoded) for this transfer function.
    ///
    /// `linear` should be in the range 0.0–1.0 (or higher for HDR).
    #[must_use]
    pub fn encode(self, linear: f64) -> f64 {
        match self {
            Self::Linear => linear,
            Self::Gamma22 => linear.powf(1.0 / 2.2),
            Self::SRGB => srgb_oetf(linear),
            Self::PQ => pq_oetf(linear),
            Self::HLG => hlg_oetf(linear),
            Self::LogC => logc_oetf(linear),
            Self::SLog3 => slog3_oetf(linear),
        }
    }

    /// Apply the EOTF (encoded → scene-linear) for this transfer function.
    #[must_use]
    pub fn decode(self, encoded: f64) -> f64 {
        match self {
            Self::Linear => encoded,
            Self::Gamma22 => encoded.powf(2.2),
            Self::SRGB => srgb_eotf(encoded),
            Self::PQ => pq_eotf(encoded),
            Self::HLG => hlg_eotf(encoded),
            Self::LogC => logc_eotf(encoded),
            Self::SLog3 => slog3_eotf(encoded),
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer function helpers
// ---------------------------------------------------------------------------

fn srgb_eotf(v: f64) -> f64 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_oetf(v: f64) -> f64 {
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn pq_oetf(linear: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    let y = linear.powf(M1);
    ((C1 + C2 * y) / (1.0 + C3 * y)).powf(M2)
}

fn pq_eotf(encoded: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    let v = encoded.powf(1.0 / M2);
    ((v - C1).max(0.0) / (C2 - C3 * v)).powf(1.0 / M1)
}

fn hlg_oetf(linear: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    if linear <= 1.0 / 12.0 {
        (3.0 * linear).sqrt()
    } else {
        A * (12.0 * linear - B).ln() + C
    }
}

fn hlg_eotf(encoded: f64) -> f64 {
    const A: f64 = 0.178_832_77;
    const B: f64 = 0.284_668_92;
    const C: f64 = 0.559_910_73;
    if encoded <= 0.5 {
        (encoded * encoded) / 3.0
    } else {
        (((encoded - C) / A).exp() + B) / 12.0
    }
}

/// ARRI LogC EI 800 OETF (scene-linear → LogC).
fn logc_oetf(linear: f64) -> f64 {
    const CUT: f64 = 0.010_591;
    const A: f64 = 5.555_556;
    const B: f64 = 0.052_272;
    const C: f64 = 0.247_190;
    const D: f64 = 0.385_537;
    const E: f64 = 5.367_655;
    const F: f64 = 0.092_809;

    if linear > CUT {
        C * (A * linear + B).log10() + D
    } else {
        E * linear + F
    }
}

/// ARRI LogC EI 800 EOTF (LogC → scene-linear).
fn logc_eotf(encoded: f64) -> f64 {
    const CUT: f64 = 0.010_591;
    const A: f64 = 5.555_556;
    const B: f64 = 0.052_272;
    const C: f64 = 0.247_190;
    const D: f64 = 0.385_537;
    const E: f64 = 5.367_655;
    const F: f64 = 0.092_809;
    const LIN_CUT: f64 = 0.005_526;

    if encoded > E * LIN_CUT + F {
        (10_f64.powf((encoded - D) / C) - B) / A
    } else {
        (encoded - F) / E
    }
}

/// Sony S-Log3 OETF (scene-linear → S-Log3).
///
/// Reference: Sony S-Log3 specification.
fn slog3_oetf(linear: f64) -> f64 {
    // IRE cut point: 0.01125 scene-linear
    if linear >= 0.01125 {
        // Logarithmic portion
        (420.0 + 261.5 * (linear + 0.018_573_2).log10()) / 1023.0
    } else {
        // Linear portion
        (linear * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    }
}

/// Sony S-Log3 EOTF (S-Log3 → scene-linear).
fn slog3_eotf(encoded: f64) -> f64 {
    let code = encoded * 1023.0;
    // Linear cut encoded value: 171.2102946929
    if code >= 171.2102946929 {
        10_f64.powf((code - 420.0) / 261.5) - 0.018_573_2
    } else {
        (code - 95.0) * 0.01125 / (171.2102946929 - 95.0)
    }
}

// ---------------------------------------------------------------------------
// ColorTransformUtil
// ---------------------------------------------------------------------------

/// Utility for batch color space conversion.
pub struct ColorTransformUtil;

impl ColorTransformUtil {
    /// Convert a slice of RGB pixels from one color space to another.
    ///
    /// Each element in `pixels` is `[r, g, b]` in the range 0.0–1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if the color space definitions cannot be created.
    pub fn convert(
        pixels: &[[f64; 3]],
        from: ColorSpaceId,
        to: ColorSpaceId,
    ) -> Result<Vec<[f64; 3]>> {
        if from == to {
            return Ok(pixels.to_vec());
        }
        let src = from.to_color_space()?;
        let dst = to.to_color_space()?;
        Ok(pixels.iter().map(|px| rgb_to_rgb(px, &src, &dst)).collect())
    }

    /// Convert a mutable slice of RGB pixels in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if the color space definitions cannot be created.
    pub fn convert_in_place(
        pixels: &mut [[f64; 3]],
        from: ColorSpaceId,
        to: ColorSpaceId,
    ) -> Result<()> {
        if from == to {
            return Ok(());
        }
        let src = from.to_color_space()?;
        let dst = to.to_color_space()?;
        for px in pixels.iter_mut() {
            *px = rgb_to_rgb(px, &src, &dst);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_id_name() {
        assert_eq!(ColorSpaceId::Rec709.name(), "Rec.709");
        assert_eq!(ColorSpaceId::Rec2020.name(), "Rec.2020");
        assert_eq!(ColorSpaceId::P3DCI.name(), "DCI-P3");
        assert_eq!(ColorSpaceId::P3D65.name(), "Display P3 (D65)");
        assert_eq!(ColorSpaceId::AcesAP0.name(), "ACES AP0 (ACES2065-1)");
        assert_eq!(ColorSpaceId::AcesAP1.name(), "ACES AP1 (ACEScg)");
        assert_eq!(ColorSpaceId::SRGB.name(), "sRGB");
        assert_eq!(ColorSpaceId::Linear.name(), "Linear");
    }

    #[test]
    fn test_transfer_function_id_name() {
        assert_eq!(TransferFunctionId::SRGB.name(), "sRGB");
        assert_eq!(TransferFunctionId::Gamma22.name(), "Gamma 2.2");
        assert_eq!(TransferFunctionId::PQ.name(), "PQ (SMPTE ST 2084)");
        assert_eq!(TransferFunctionId::HLG.name(), "HLG (ARIB STD-B67)");
        assert_eq!(TransferFunctionId::Linear.name(), "Linear");
        assert_eq!(TransferFunctionId::LogC.name(), "ARRI LogC");
        assert_eq!(TransferFunctionId::SLog3.name(), "Sony S-Log3");
    }

    #[test]
    fn test_srgb_round_trip() {
        let tf = TransferFunctionId::SRGB;
        let linear = 0.5;
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!((decoded - linear).abs() < 1e-10);
    }

    #[test]
    fn test_logc_round_trip() {
        let tf = TransferFunctionId::LogC;
        let linear = 0.18; // middle grey
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!(
            (decoded - linear).abs() < 1e-8,
            "expected {linear}, got {decoded}"
        );
    }

    #[test]
    fn test_slog3_round_trip() {
        let tf = TransferFunctionId::SLog3;
        let linear = 0.18;
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!(
            (decoded - linear).abs() < 1e-8,
            "expected {linear}, got {decoded}"
        );
    }

    #[test]
    fn test_pq_round_trip() {
        let tf = TransferFunctionId::PQ;
        let linear = 0.5;
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!((decoded - linear).abs() < 1e-10);
    }

    #[test]
    fn test_hlg_round_trip() {
        let tf = TransferFunctionId::HLG;
        let linear = 0.5;
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!((decoded - linear).abs() < 0.001);
    }

    #[test]
    fn test_color_transform_util_same_space() {
        let pixels = vec![[0.5, 0.3, 0.7]];
        let result = ColorTransformUtil::convert(&pixels, ColorSpaceId::SRGB, ColorSpaceId::SRGB)
            .expect("color conversion should succeed");
        assert_eq!(result.len(), 1);
        assert!((result[0][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_color_transform_util_srgb_to_rec2020() {
        let pixels = vec![[0.5_f64, 0.5, 0.5]];
        let result =
            ColorTransformUtil::convert(&pixels, ColorSpaceId::SRGB, ColorSpaceId::Rec2020)
                .expect("operation should succeed in test");
        assert_eq!(result.len(), 1);
        // Values should be in a valid range
        for &v in &result[0] {
            assert!(v >= 0.0 && v <= 1.5, "Unexpected value: {v}");
        }
    }

    #[test]
    fn test_color_transform_util_in_place_identity() {
        let mut pixels = vec![[0.5_f64, 0.3, 0.7]];
        ColorTransformUtil::convert_in_place(&mut pixels, ColorSpaceId::SRGB, ColorSpaceId::SRGB)
            .expect("operation should succeed in test");
        assert!((pixels[0][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_color_space_id_to_color_space() {
        // All IDs should resolve without error
        for id in [
            ColorSpaceId::Rec709,
            ColorSpaceId::Rec2020,
            ColorSpaceId::P3DCI,
            ColorSpaceId::P3D65,
            ColorSpaceId::AcesAP0,
            ColorSpaceId::AcesAP1,
            ColorSpaceId::SRGB,
            ColorSpaceId::Linear,
        ] {
            assert!(id.to_color_space().is_ok(), "Failed for {id:?}");
        }
    }

    #[test]
    fn test_logc_middle_grey() {
        // ARRI LogC EI 800: middle grey (0.18) encodes to ~0.391
        let tf = TransferFunctionId::LogC;
        let encoded = tf.encode(0.18);
        assert!(
            encoded > 0.38 && encoded < 0.40,
            "Expected ~0.391, got {encoded}"
        );
    }

    #[test]
    fn test_gamma22_round_trip() {
        let tf = TransferFunctionId::Gamma22;
        let linear = 0.5;
        let encoded = tf.encode(linear);
        let decoded = tf.decode(encoded);
        assert!((decoded - linear).abs() < 1e-10);
    }

    #[test]
    fn test_linear_passthrough() {
        let tf = TransferFunctionId::Linear;
        assert_eq!(tf.encode(0.5), 0.5);
        assert_eq!(tf.decode(0.5), 0.5);
    }
}
