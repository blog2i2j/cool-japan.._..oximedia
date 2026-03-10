//! Display profile management for color-managed rendering.
//!
//! Provides structures and utilities for describing display device
//! colour characteristics, tone response curves, and calibration state.

#![allow(dead_code)]

use std::fmt;

/// Display colour gamut primaries and white point in CIE xy chromaticity.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayGamut {
    /// Red primary chromaticity (x, y).
    pub red: [f64; 2],
    /// Green primary chromaticity (x, y).
    pub green: [f64; 2],
    /// Blue primary chromaticity (x, y).
    pub blue: [f64; 2],
    /// White point chromaticity (x, y).
    pub white: [f64; 2],
}

impl DisplayGamut {
    /// sRGB / Rec.709 gamut.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            red: [0.6400, 0.3300],
            green: [0.3000, 0.6000],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// DCI-P3 gamut (D65 white point variant).
    #[must_use]
    pub fn dci_p3_d65() -> Self {
        Self {
            red: [0.6800, 0.3200],
            green: [0.2650, 0.6900],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Rec.2020 gamut.
    #[must_use]
    pub fn rec2020() -> Self {
        Self {
            red: [0.7080, 0.2920],
            green: [0.1700, 0.7970],
            blue: [0.1310, 0.0460],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Adobe RGB (1998) gamut.
    #[must_use]
    pub fn adobe_rgb() -> Self {
        Self {
            red: [0.6400, 0.3300],
            green: [0.2100, 0.7100],
            blue: [0.1500, 0.0600],
            white: [0.3127, 0.3290], // D65
        }
    }

    /// Compute the gamut area in xy chromaticity using the shoelace formula.
    ///
    /// Larger values indicate wider gamuts.
    #[must_use]
    pub fn chromaticity_area(&self) -> f64 {
        let r = self.red;
        let g = self.green;
        let b = self.blue;
        // Shoelace formula for triangle area
        0.5 * ((g[0] - r[0]) * (b[1] - r[1]) - (b[0] - r[0]) * (g[1] - r[1])).abs()
    }
}

/// Tone response curve type for a display device.
#[derive(Debug, Clone, PartialEq)]
pub enum ToneResponseCurve {
    /// Pure power-law gamma (e.g., 2.2 for sRGB nominal).
    Gamma(f64),
    /// sRGB piecewise transfer function.
    Srgb,
    /// BT.1886 EOTF (reference display gamma).
    Bt1886 {
        /// Black level in cd/m² (typically 0.01).
        black_level: f64,
        /// White level in cd/m² (typically 100.0).
        white_level: f64,
    },
    /// Perceptual Quantizer (ST 2084 / PQ).
    Pq,
    /// Hybrid Log-Gamma (HLG).
    Hlg,
    /// Linear (no gamma encoding).
    Linear,
    /// Custom LUT points (evenly spaced, input 0-1).
    CustomLut(Vec<f64>),
}

impl ToneResponseCurve {
    /// Apply the electro-optical transfer function (encoded -> display linear).
    ///
    /// Input `v` is expected in [0, 1].  Output is in linear light [0, 1]
    /// (or cd/m² units for absolute TRCs like PQ).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn apply_eotf(&self, v: f64) -> f64 {
        match self {
            Self::Gamma(g) => v.abs().powf(*g).copysign(v),
            Self::Linear => v,
            Self::Srgb => {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            }
            Self::Bt1886 {
                black_level,
                white_level,
            } => {
                // Simplified BT.1886: (v)^2.4 mapped to [black, white]
                let linear = v.powf(2.4);
                black_level + (white_level - black_level) * linear
            }
            Self::Pq => {
                // ST 2084 EOTF
                const M1: f64 = 0.1593017578125;
                const M2: f64 = 78.84375;
                const C1: f64 = 0.8359375;
                const C2: f64 = 18.8515625;
                const C3: f64 = 18.6875;
                let vm2 = v.abs().powf(1.0 / M2);
                let num = (vm2 - C1).max(0.0);
                let den = C2 - C3 * vm2;
                (num / den).powf(1.0 / M1) * 10000.0 * if v < 0.0 { -1.0 } else { 1.0 }
            }
            Self::Hlg => {
                // ARIB STD-B67 OETF inverse (simplified)
                const A: f64 = 0.17883277;
                const B: f64 = 0.28466892;
                const C: f64 = 0.55991073;
                if v <= 0.5 {
                    (v * v) / 3.0
                } else {
                    ((v - C).exp() / A + B) / 12.0
                }
            }
            Self::CustomLut(lut) => {
                if lut.is_empty() {
                    return v;
                }
                let n = lut.len() - 1;
                let pos = v.clamp(0.0, 1.0) * n as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(n);
                let t = pos - lo as f64;
                lut[lo] * (1.0 - t) + lut[hi] * t
            }
        }
    }

    /// Apply the opto-electronic transfer function (linear -> encoded).
    ///
    /// This is the inverse of `apply_eotf` for simple gamma/sRGB cases.
    #[must_use]
    pub fn apply_oetf(&self, v: f64) -> f64 {
        match self {
            Self::Gamma(g) => v.abs().powf(1.0 / g).copysign(v),
            Self::Linear => v,
            Self::Srgb => {
                if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }
            // For other TRCs a full inverse is complex; return v as fallback
            _ => v,
        }
    }
}

/// Calibration state of a display device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalibrationState {
    /// Display has never been calibrated.
    Uncalibrated,
    /// Display has been calibrated, profile may be valid.
    Calibrated,
    /// Calibration date is known (ISO 8601 date string).
    CalibratedAt(String),
    /// Calibration has expired.
    Expired,
}

impl fmt::Display for CalibrationState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uncalibrated => write!(f, "Uncalibrated"),
            Self::Calibrated => write!(f, "Calibrated"),
            Self::CalibratedAt(d) => write!(f, "Calibrated at {d}"),
            Self::Expired => write!(f, "Expired"),
        }
    }
}

/// A complete display profile description.
#[derive(Debug, Clone)]
pub struct DisplayProfile {
    /// Human-readable display name.
    pub name: String,
    /// Colour gamut of the display.
    pub gamut: DisplayGamut,
    /// Tone response curve.
    pub trc: ToneResponseCurve,
    /// Peak luminance in cd/m².
    pub peak_luminance: f64,
    /// Black level in cd/m².
    pub black_level: f64,
    /// Calibration state.
    pub calibration: CalibrationState,
    /// Whether the display supports HDR.
    pub hdr_capable: bool,
}

impl DisplayProfile {
    /// Create a standard sRGB display profile.
    #[must_use]
    pub fn srgb() -> Self {
        Self {
            name: "sRGB IEC61966-2.1".to_string(),
            gamut: DisplayGamut::srgb(),
            trc: ToneResponseCurve::Srgb,
            peak_luminance: 80.0,
            black_level: 0.0,
            calibration: CalibrationState::Calibrated,
            hdr_capable: false,
        }
    }

    /// Create a Rec.2020 HDR10 display profile.
    #[must_use]
    pub fn hdr10() -> Self {
        Self {
            name: "HDR10 (Rec.2020/PQ)".to_string(),
            gamut: DisplayGamut::rec2020(),
            trc: ToneResponseCurve::Pq,
            peak_luminance: 1000.0,
            black_level: 0.0001,
            calibration: CalibrationState::Calibrated,
            hdr_capable: true,
        }
    }

    /// Dynamic contrast ratio (peak / black level).
    ///
    /// Returns `None` if black level is zero.
    #[must_use]
    pub fn dynamic_contrast(&self) -> Option<f64> {
        if self.black_level <= 0.0 {
            None
        } else {
            Some(self.peak_luminance / self.black_level)
        }
    }

    /// Whether this profile covers at least `coverage` fraction of the sRGB gamut.
    #[must_use]
    pub fn covers_srgb(&self, coverage: f64) -> bool {
        let srgb_area = DisplayGamut::srgb().chromaticity_area();
        let this_area = self.gamut.chromaticity_area();
        this_area >= srgb_area * coverage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_display_gamut_srgb_white() {
        let g = DisplayGamut::srgb();
        assert!(approx_eq(g.white[0], 0.3127, 1e-4));
        assert!(approx_eq(g.white[1], 0.3290, 1e-4));
    }

    #[test]
    fn test_gamut_area_rec2020_wider_than_srgb() {
        let srgb = DisplayGamut::srgb().chromaticity_area();
        let r2020 = DisplayGamut::rec2020().chromaticity_area();
        assert!(r2020 > srgb);
    }

    #[test]
    fn test_gamut_area_p3_wider_than_srgb() {
        let srgb = DisplayGamut::srgb().chromaticity_area();
        let p3 = DisplayGamut::dci_p3_d65().chromaticity_area();
        assert!(p3 > srgb);
    }

    #[test]
    fn test_tone_response_curve_linear() {
        let trc = ToneResponseCurve::Linear;
        assert!(approx_eq(trc.apply_eotf(0.5), 0.5, 1e-10));
        assert!(approx_eq(trc.apply_oetf(0.5), 0.5, 1e-10));
    }

    #[test]
    fn test_tone_response_curve_gamma_round_trip() {
        let trc = ToneResponseCurve::Gamma(2.2);
        let v = 0.6;
        let encoded = trc.apply_oetf(v);
        let decoded = trc.apply_eotf(encoded);
        assert!(approx_eq(decoded, v, 1e-8));
    }

    #[test]
    fn test_srgb_trc_round_trip() {
        let trc = ToneResponseCurve::Srgb;
        for &v in &[0.0, 0.01, 0.18, 0.5, 0.9, 1.0] {
            let enc = trc.apply_oetf(v);
            let dec = trc.apply_eotf(enc);
            assert!(approx_eq(dec, v, 1e-8), "v={v} enc={enc} dec={dec}");
        }
    }

    #[test]
    fn test_srgb_eotf_black() {
        let trc = ToneResponseCurve::Srgb;
        assert!(approx_eq(trc.apply_eotf(0.0), 0.0, 1e-10));
    }

    #[test]
    fn test_srgb_eotf_white() {
        let trc = ToneResponseCurve::Srgb;
        assert!(approx_eq(trc.apply_eotf(1.0), 1.0, 1e-8));
    }

    #[test]
    fn test_calibration_state_display() {
        assert_eq!(CalibrationState::Uncalibrated.to_string(), "Uncalibrated");
        let at = CalibrationState::CalibratedAt("2024-01-01".to_string());
        assert!(at.to_string().contains("2024-01-01"));
    }

    #[test]
    fn test_display_profile_srgb_not_hdr() {
        let p = DisplayProfile::srgb();
        assert!(!p.hdr_capable);
        assert!(approx_eq(p.peak_luminance, 80.0, 1e-10));
    }

    #[test]
    fn test_display_profile_hdr10() {
        let p = DisplayProfile::hdr10();
        assert!(p.hdr_capable);
        assert!(p.peak_luminance >= 1000.0);
    }

    #[test]
    fn test_dynamic_contrast_none_for_zero_black() {
        let p = DisplayProfile::srgb();
        // sRGB black_level = 0.0 -> None
        assert!(p.dynamic_contrast().is_none());
    }

    #[test]
    fn test_dynamic_contrast_hdr10() {
        let p = DisplayProfile::hdr10();
        let cr = p
            .dynamic_contrast()
            .expect("dynamic contrast should be available");
        assert!(cr > 1_000_000.0);
    }

    #[test]
    fn test_covers_srgb() {
        let p_srgb = DisplayProfile::srgb();
        let p_hdr10 = DisplayProfile::hdr10();
        assert!(p_srgb.covers_srgb(1.0));
        assert!(p_hdr10.covers_srgb(1.0));
    }

    #[test]
    fn test_custom_lut_trc_interpolates() {
        let lut = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let trc = ToneResponseCurve::CustomLut(lut);
        // Midpoint should interpolate to ~0.5
        assert!(approx_eq(trc.apply_eotf(0.5), 0.5, 1e-6));
    }
}
