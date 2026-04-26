//! ACES (Academy Color Encoding System) color transforms.
//!
//! This module provides a comprehensive implementation of ACES color management,
//! including transforms between ACES color spaces and Output Device Transforms (ODTs).
//!
//! # ACES Color Spaces
//!
//! - **ACES2065-1 (AP0)**: Scene-referred linear space with wide gamut primaries
//! - **`ACEScg` (AP1)**: Working space for CGI and compositing
//! - **`ACEScct`**: Logarithmic working space for color grading
//! - **`ACESproxy`**: 10-bit/12-bit logarithmic encoding for on-set monitoring
//!
//! # Output Transforms
//!
//! - Rec.709 (100 nits)
//! - Rec.2020 (100 nits, 1000 nits, 2000 nits, 4000 nits)
//! - DCI-P3 (48 nits, D60 white point)
//! - sRGB (D65 white point)
//!
//! # Reference
//!
//! Based on ACES 1.3 specification from the Academy of Motion Picture Arts and Sciences.

use crate::error::LutResult;
use crate::matrix::{self, apply_matrix3x3, Matrix3x3};
use crate::Rgb;

/// ACES color space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcesSpace {
    /// ACES2065-1 (AP0, scene-referred linear).
    Aces2065,
    /// `ACEScg` (AP1, working space).
    AcesCg,
    /// `ACEScct` (logarithmic working space).
    AcesCct,
    /// ACES Proxy 10-bit.
    AcesProxy10,
    /// ACES Proxy 12-bit.
    AcesProxy12,
}

/// ACES Output Device Transform (ODT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcesOdt {
    /// Rec.709 (100 nits, D65).
    Rec709,
    /// Rec.2020 (100 nits, D65).
    Rec2020_100,
    /// Rec.2020 (1000 nits, D65, ST2084/PQ).
    Rec2020_1000,
    /// Rec.2020 (2000 nits, D65, ST2084/PQ).
    Rec2020_2000,
    /// Rec.2020 (4000 nits, D65, ST2084/PQ).
    Rec2020_4000,
    /// DCI-P3 (48 nits, D60).
    DciP3,
    /// sRGB (D65).
    Srgb,
}

// ============================================================================
// ACES Color Space Matrices
// ============================================================================

/// AP0 (ACES2065-1) to XYZ matrix.
pub const AP0_TO_XYZ: Matrix3x3 = [
    [0.952_552_395_9, 0.000_000_000_0, 0.000_093_678_6],
    [0.343_966_449_8, 0.728_166_096_6, -0.072_132_546_4],
    [0.000_000_000_0, 0.000_000_000_0, 1.008_825_184_4],
];

/// XYZ to AP0 (ACES2065-1) matrix.
pub const XYZ_TO_AP0: Matrix3x3 = [
    [1.049_811_017_5, 0.000_000_000_0, -0.000_097_484_5],
    [-0.495_903_023_1, 1.373_313_045_8, 0.098_240_036_1],
    [0.000_000_000_0, 0.000_000_000_0, 0.991_252_018_2],
];

/// AP1 (`ACEScg`) to XYZ matrix.
pub const AP1_TO_XYZ: Matrix3x3 = [
    [0.662_454_181_1, 0.134_004_206_5, 0.156_187_687_0],
    [0.272_228_716_8, 0.674_081_765_8, 0.053_689_517_4],
    [-0.005_574_649_5, 0.004_060_733_5, 1.010_339_100_3],
];

/// XYZ to AP1 (`ACEScg`) matrix.
pub const XYZ_TO_AP1: Matrix3x3 = [
    [1.641_023_379_7, -0.324_803_294_2, -0.236_424_695_2],
    [-0.663_662_858_7, 1.615_331_591_7, 0.016_756_347_7],
    [0.011_721_894_3, -0.008_284_442_0, 0.988_394_858_5],
];

/// AP0 to AP1 conversion matrix.
pub const AP0_TO_AP1: Matrix3x3 = [
    [1.451_439_316_1, -0.236_510_746_9, -0.214_928_569_3],
    [-0.076_553_773_4, 1.176_229_699_8, -0.099_675_926_4],
    [0.008_316_148_4, -0.006_032_449_8, 0.997_716_301_4],
];

/// AP1 to AP0 conversion matrix.
pub const AP1_TO_AP0: Matrix3x3 = [
    [0.695_452_241_4, 0.140_678_696_5, 0.163_869_062_2],
    [0.044_794_563_4, 0.859_671_118_5, 0.095_534_318_2],
    [-0.005_525_882_6, 0.004_025_210_3, 1.001_500_672_3],
];

// ============================================================================
// ACES Color Space Conversions
// ============================================================================

impl AcesSpace {
    /// Convert from this ACES space to linear AP0 (ACES2065-1).
    #[must_use]
    pub fn to_aces2065(&self, rgb: &Rgb) -> Rgb {
        match self {
            Self::Aces2065 => *rgb,
            Self::AcesCg => acescg_to_aces2065(rgb),
            Self::AcesCct => acescct_to_aces2065(rgb),
            Self::AcesProxy10 => acesproxy_to_aces2065(rgb, 10),
            Self::AcesProxy12 => acesproxy_to_aces2065(rgb, 12),
        }
    }

    /// Convert from linear AP0 (ACES2065-1) to this ACES space.
    #[must_use]
    pub fn from_aces2065(&self, rgb: &Rgb) -> Rgb {
        match self {
            Self::Aces2065 => *rgb,
            Self::AcesCg => aces2065_to_acescg(rgb),
            Self::AcesCct => aces2065_to_acescct(rgb),
            Self::AcesProxy10 => aces2065_to_acesproxy(rgb, 10),
            Self::AcesProxy12 => aces2065_to_acesproxy(rgb, 12),
        }
    }
}

/// Convert `ACEScg` (AP1 linear) to ACES2065-1 (AP0 linear).
#[must_use]
pub fn acescg_to_aces2065(rgb: &Rgb) -> Rgb {
    apply_matrix3x3(&AP1_TO_AP0, rgb)
}

/// Convert ACES2065-1 (AP0 linear) to `ACEScg` (AP1 linear).
#[must_use]
pub fn aces2065_to_acescg(rgb: &Rgb) -> Rgb {
    apply_matrix3x3(&AP0_TO_AP1, rgb)
}

/// Convert `ACEScct` (AP1 logarithmic) to ACES2065-1 (AP0 linear).
#[must_use]
pub fn acescct_to_aces2065(rgb: &Rgb) -> Rgb {
    // First decode ACEScct to linear ACEScg
    let linear_ap1 = [
        acescct_to_linear(rgb[0]),
        acescct_to_linear(rgb[1]),
        acescct_to_linear(rgb[2]),
    ];
    // Then convert to AP0
    acescg_to_aces2065(&linear_ap1)
}

/// Convert ACES2065-1 (AP0 linear) to `ACEScct` (AP1 logarithmic).
#[must_use]
pub fn aces2065_to_acescct(rgb: &Rgb) -> Rgb {
    // First convert to ACEScg (AP1 linear)
    let ap1 = aces2065_to_acescg(rgb);
    // Then encode to ACEScct
    [
        linear_to_acescct(ap1[0]),
        linear_to_acescct(ap1[1]),
        linear_to_acescct(ap1[2]),
    ]
}

/// `ACEScct` to linear transfer function.
#[must_use]
fn acescct_to_linear(x: f64) -> f64 {
    const X_BRK: f64 = 0.155_251_141_552_511;

    if x > X_BRK {
        (10.0_f64.powf(x * 17.52 - 9.72) - 0.000_089_999_999_999_999_99) / 0.18
    } else if x < 0.073_292_48 {
        (x - 0.071_776_470_588_235_29) / 10.540_237_741_654_5
    } else {
        10.0_f64.powf(x * 17.52 - 9.72) / 0.18
    }
}

/// Linear to `ACEScct` transfer function.
#[must_use]
fn linear_to_acescct(x: f64) -> f64 {
    const LIN_CUT: f64 = 0.007_812_5;

    let x = x * 0.18;

    if x <= LIN_CUT {
        10.540_237_741_654_5 * x + 0.071_776_470_588_235_29
    } else {
        (x.ln() / 10.0_f64.ln() + 9.72) / 17.52
    }
}

/// Convert `ACESproxy` to ACES2065-1.
#[must_use]
pub fn acesproxy_to_aces2065(rgb: &Rgb, bits: u8) -> Rgb {
    let linear_ap1 = match bits {
        10 => [
            acesproxy10_to_linear(rgb[0]),
            acesproxy10_to_linear(rgb[1]),
            acesproxy10_to_linear(rgb[2]),
        ],
        12 => [
            acesproxy12_to_linear(rgb[0]),
            acesproxy12_to_linear(rgb[1]),
            acesproxy12_to_linear(rgb[2]),
        ],
        _ => *rgb,
    };
    acescg_to_aces2065(&linear_ap1)
}

/// Convert ACES2065-1 to `ACESproxy`.
#[must_use]
pub fn aces2065_to_acesproxy(rgb: &Rgb, bits: u8) -> Rgb {
    let ap1 = aces2065_to_acescg(rgb);
    match bits {
        10 => [
            linear_to_acesproxy10(ap1[0]),
            linear_to_acesproxy10(ap1[1]),
            linear_to_acesproxy10(ap1[2]),
        ],
        12 => [
            linear_to_acesproxy12(ap1[0]),
            linear_to_acesproxy12(ap1[1]),
            linear_to_acesproxy12(ap1[2]),
        ],
        _ => ap1,
    }
}

/// `ACESproxy` 10-bit to linear.
#[must_use]
fn acesproxy10_to_linear(x: f64) -> f64 {
    (10.0_f64.powf((x - 64.0) / 50.0) - 0.000_062_514_891_117_416_44) / 0.18
}

/// Linear to `ACESproxy` 10-bit.
#[must_use]
fn linear_to_acesproxy10(x: f64) -> f64 {
    (x * 0.18).max(0.000_062_514_891_117_416_44).ln() / 10.0_f64.ln() * 50.0 + 64.0
}

/// `ACESproxy` 12-bit to linear.
#[must_use]
fn acesproxy12_to_linear(x: f64) -> f64 {
    (10.0_f64.powf((x - 256.0) / 200.0) - 0.000_062_514_891_117_416_44) / 0.18
}

/// Linear to `ACESproxy` 12-bit.
#[must_use]
fn linear_to_acesproxy12(x: f64) -> f64 {
    (x * 0.18).max(0.000_062_514_891_117_416_44).ln() / 10.0_f64.ln() * 200.0 + 256.0
}

// ============================================================================
// ACES Output Device Transforms (ODTs)
// ============================================================================

impl AcesOdt {
    /// Apply the ODT to convert from ACES2065-1 to display-referred RGB.
    ///
    /// # Errors
    ///
    /// This function currently always returns `Ok`.  The `Result` wrapper is
    /// retained for forward compatibility should a future variant require
    /// fallible processing.
    pub fn apply(&self, aces_rgb: &Rgb) -> LutResult<Rgb> {
        match self {
            Self::Rec709 => Ok(aces_odt_rec709(aces_rgb)),
            Self::Rec2020_100 => Ok(aces_odt_rec2020_100(aces_rgb)),
            Self::Rec2020_1000 => Ok(aces_odt_rec2020_hdr(aces_rgb, 1000.0)),
            Self::Rec2020_2000 => Ok(aces_odt_rec2020_hdr(aces_rgb, 2000.0)),
            Self::Rec2020_4000 => Ok(aces_odt_rec2020_hdr(aces_rgb, 4000.0)),
            Self::DciP3 => Ok(aces_odt_dcip3(aces_rgb)),
            Self::Srgb => Ok(aces_odt_srgb(aces_rgb)),
        }
    }
}

/// ACES ODT for Rec.709 (100 nits, D65).
///
/// Applies RRT tone curve, AP1→XYZ→Rec.709 matrix, then Rec.709 OETF.
/// Output is clamped to [0, 1].
#[must_use]
pub fn aces_odt_rec709(aces: &Rgb) -> Rgb {
    // Convert AP0 to AP1 (ACEScg working space)
    let acescg = aces2065_to_acescg(aces);
    // Apply RRT (Reference Rendering Transform) tone curve
    let rrt = aces_rrt(&acescg);
    // Convert AP1 → XYZ → Rec.709 linear
    let linear = apply_matrix3x3(
        &matrix::XYZ_TO_RGB_REC709,
        &apply_matrix3x3(&AP1_TO_XYZ, &rrt),
    );
    // Apply Rec.709 OETF and clamp to [0, 1]
    [
        rec709_oetf(linear[0]).clamp(0.0, 1.0),
        rec709_oetf(linear[1]).clamp(0.0, 1.0),
        rec709_oetf(linear[2]).clamp(0.0, 1.0),
    ]
}

/// ACES ODT for Rec.2020 (100 nits SDR, D65).
///
/// Applies RRT tone curve, AP1→XYZ→Rec.2020 matrix, then the BT.2087
/// Rec.2020 SDR OETF.  Output is clamped to [0, 1].
#[must_use]
pub fn aces_odt_rec2020_100(aces: &Rgb) -> Rgb {
    let acescg = aces2065_to_acescg(aces);
    let rrt = aces_rrt(&acescg);
    let linear = apply_matrix3x3(
        &matrix::XYZ_TO_RGB_REC2020,
        &apply_matrix3x3(&AP1_TO_XYZ, &rrt),
    );
    [
        rec2020_oetf(linear[0]).clamp(0.0, 1.0),
        rec2020_oetf(linear[1]).clamp(0.0, 1.0),
        rec2020_oetf(linear[2]).clamp(0.0, 1.0),
    ]
}

/// ACES ODT for Rec.2020 HDR with SMPTE ST.2084 (PQ) encoding.
///
/// `peak_nits` sets the display peak luminance in cd/m² (e.g. 1000, 2000, 4000).
/// The normalised linear value from the RRT is scaled to absolute nits before
/// the PQ OETF is applied.  Output is clamped to [0, 1].
#[must_use]
fn aces_odt_rec2020_hdr(aces: &Rgb, peak_nits: f64) -> Rgb {
    let acescg = aces2065_to_acescg(aces);
    let rrt = aces_rrt(&acescg);
    // AP1 → XYZ → Rec.2020 linear (normalised [0,1] at peak)
    let linear = apply_matrix3x3(
        &matrix::XYZ_TO_RGB_REC2020,
        &apply_matrix3x3(&AP1_TO_XYZ, &rrt),
    );
    // Scale to absolute nits and apply PQ OETF
    [
        pq_oetf(linear[0] * peak_nits).clamp(0.0, 1.0),
        pq_oetf(linear[1] * peak_nits).clamp(0.0, 1.0),
        pq_oetf(linear[2] * peak_nits).clamp(0.0, 1.0),
    ]
}

/// ACES ODT for DCI-P3 (48 nits, D60 white point).
///
/// Applies RRT tone curve, AP1→XYZ→DCI-P3 matrix, then the standard DCI
/// 2.6 power-law gamma OETF.  Output is clamped to [0, 1].
#[must_use]
pub fn aces_odt_dcip3(aces: &Rgb) -> Rgb {
    let acescg = aces2065_to_acescg(aces);
    let rrt = aces_rrt(&acescg);
    let linear = apply_matrix3x3(
        &matrix::XYZ_TO_RGB_DCIP3,
        &apply_matrix3x3(&AP1_TO_XYZ, &rrt),
    );
    [
        gamma_2_6_oetf(linear[0]).clamp(0.0, 1.0),
        gamma_2_6_oetf(linear[1]).clamp(0.0, 1.0),
        gamma_2_6_oetf(linear[2]).clamp(0.0, 1.0),
    ]
}

/// ACES ODT for sRGB (D65 white point).
///
/// sRGB primaries are identical to Rec.709, so the same gamut matrix is used.
/// The IEC 61966-2-1 sRGB piecewise OETF is applied (distinct from Rec.709:
/// threshold 0.0031308, power 1/2.4).  Output is clamped to [0, 1].
#[must_use]
pub fn aces_odt_srgb(aces: &Rgb) -> Rgb {
    let acescg = aces2065_to_acescg(aces);
    let rrt = aces_rrt(&acescg);
    // sRGB primaries == Rec.709 → same XYZ→RGB matrix
    let linear = apply_matrix3x3(
        &matrix::XYZ_TO_RGB_REC709,
        &apply_matrix3x3(&AP1_TO_XYZ, &rrt),
    );
    [
        srgb_oetf(linear[0]).clamp(0.0, 1.0),
        srgb_oetf(linear[1]).clamp(0.0, 1.0),
        srgb_oetf(linear[2]).clamp(0.0, 1.0),
    ]
}

// ============================================================================
// ACES Reference Rendering Transform (RRT)
// ============================================================================

/// ACES Reference Rendering Transform (RRT).
///
/// Simplified per-channel tone curve that approximates the full Academy RRT.
#[must_use]
fn aces_rrt(rgb: &Rgb) -> Rgb {
    [
        aces_tone_curve(rgb[0]),
        aces_tone_curve(rgb[1]),
        aces_tone_curve(rgb[2]),
    ]
}

/// ACES filmic tone curve (simplified).
///
/// Coefficients from the well-known Jim Hejl / ACES approximation:
/// `f(x) = (x(Ax+B)) / (x(Cx+D)+E)`.
#[must_use]
fn aces_tone_curve(x: f64) -> f64 {
    const A: f64 = 2.51;
    const B: f64 = 0.03;
    const C: f64 = 2.43;
    const D: f64 = 0.59;
    const E: f64 = 0.14;

    if x <= 0.0 {
        0.0
    } else {
        ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Electro-Optical Transfer Functions (OETF)
// ============================================================================

/// ITU-R BT.709 OETF.
///
/// Piecewise: linear below 0.018, `1.099·L^0.45 − 0.099` above.
/// Note: negative inputs clamp to 0 via `max`.
#[must_use]
fn rec709_oetf(linear: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else if linear < 0.018 {
        linear * 4.5
    } else {
        1.099 * linear.powf(0.45) - 0.099
    }
}

/// ITU-R BT.2020 / BT.2087 SDR OETF.
///
/// Same piecewise shape as Rec.709 but with more precise constants:
/// `alpha = 1.09929682680944`, `beta = 0.018053968510807`.
#[must_use]
fn rec2020_oetf(linear: f64) -> f64 {
    const ALPHA: f64 = 1.099_296_826_809_44;
    const BETA: f64 = 0.018_053_968_510_807;

    if linear <= 0.0 {
        0.0
    } else if linear < BETA {
        linear * 4.5
    } else {
        ALPHA * linear.powf(0.45) - (ALPHA - 1.0)
    }
}

/// IEC 61966-2-1 sRGB OETF.
///
/// Distinct from Rec.709: linear threshold is 0.0031308 and power is 1/2.4.
/// `V' = 12.92·L` for `L ≤ 0.0031308`, else `1.055·L^(1/2.4) − 0.055`.
#[must_use]
fn srgb_oetf(linear: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// DCI / P3 2.6 power-law OETF.
///
/// `V' = L^(1/2.6)` where L is normalised linear light in [0, 1].
#[must_use]
fn gamma_2_6_oetf(linear: f64) -> f64 {
    if linear <= 0.0 {
        0.0
    } else {
        linear.powf(1.0 / 2.6)
    }
}

/// SMPTE ST.2084 (PQ) OETF.
///
/// Converts absolute luminance in cd/m² to a normalised PQ signal in [0, 1].
/// Reference peak is 10 000 cd/m².
#[must_use]
fn pq_oetf(nits: f64) -> f64 {
    const M1: f64 = 2610.0 / 16384.0;
    const M2: f64 = 2523.0 / 4096.0 * 128.0;
    const C1: f64 = 3424.0 / 4096.0;
    const C2: f64 = 2413.0 / 4096.0 * 32.0;
    const C3: f64 = 2392.0 / 4096.0 * 32.0;
    const PQ_MAX_NITS: f64 = 10_000.0;

    if nits <= 0.0 {
        return 0.0;
    }
    let y = (nits / PQ_MAX_NITS).clamp(0.0, 1.0);
    let ym1 = y.powf(M1);
    ((C1 + C2 * ym1) / (1.0 + C3 * ym1)).powf(M2)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Existing round-trip tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_ap0_ap1_round_trip() {
        let rgb = [0.5, 0.3, 0.7];
        let ap1 = aces2065_to_acescg(&rgb);
        let back = acescg_to_aces2065(&ap1);
        assert!((rgb[0] - back[0]).abs() < 1e-6);
        assert!((rgb[1] - back[1]).abs() < 1e-6);
        assert!((rgb[2] - back[2]).abs() < 1e-6);
    }

    #[test]
    fn test_acescct_round_trip() {
        let rgb = [0.5, 0.3, 0.7];
        let cct = aces2065_to_acescct(&rgb);
        let back = acescct_to_aces2065(&cct);
        assert!((rgb[0] - back[0]).abs() < 0.01);
        assert!((rgb[1] - back[1]).abs() < 0.01);
        assert!((rgb[2] - back[2]).abs() < 0.01);
    }

    #[test]
    fn test_aces_odt_rec709() {
        let aces = [0.5, 0.3, 0.7];
        let rgb = aces_odt_rec709(&aces);
        assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
        assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
        assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
    }

    #[test]
    fn test_aces_spaces() {
        let rgb = [0.5, 0.3, 0.7];

        // Test each space
        for space in &[AcesSpace::Aces2065, AcesSpace::AcesCg, AcesSpace::AcesCct] {
            let converted = space.from_aces2065(&rgb);
            let back = space.to_aces2065(&converted);
            assert!((rgb[0] - back[0]).abs() < 0.01);
            assert!((rgb[1] - back[1]).abs() < 0.01);
            assert!((rgb[2] - back[2]).abs() < 0.01);
        }
    }

    // -------------------------------------------------------------------------
    // Helper: assert output channels are in [0, 1]
    // -------------------------------------------------------------------------

    fn assert_in_range(label: &str, rgb: &Rgb) {
        for (i, &v) in rgb.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "{label} channel {i} = {v} is out of [0,1]"
            );
        }
    }

    // -------------------------------------------------------------------------
    // AcesOdt::apply — all variants must succeed and produce values in [0,1]
    // -------------------------------------------------------------------------

    #[test]
    fn test_aces_odt_apply_all_variants() {
        let aces: Rgb = [0.5, 0.3, 0.7];

        let variants = [
            AcesOdt::Rec709,
            AcesOdt::Rec2020_100,
            AcesOdt::Rec2020_1000,
            AcesOdt::Rec2020_2000,
            AcesOdt::Rec2020_4000,
            AcesOdt::DciP3,
            AcesOdt::Srgb,
        ];

        for odt in &variants {
            let result = odt.apply(&aces);
            assert!(result.is_ok(), "ODT {odt:?} returned an error");
            let rgb = result.expect("checked above");
            assert_in_range(&format!("{odt:?}"), &rgb);
        }
    }

    #[test]
    fn test_aces_odt_apply_black() {
        let black: Rgb = [0.0, 0.0, 0.0];

        let variants = [
            AcesOdt::Rec709,
            AcesOdt::Rec2020_100,
            AcesOdt::Rec2020_1000,
            AcesOdt::Rec2020_2000,
            AcesOdt::Rec2020_4000,
            AcesOdt::DciP3,
            AcesOdt::Srgb,
        ];

        for odt in &variants {
            let rgb = odt.apply(&black).expect("apply should not error for black");
            assert_in_range(&format!("{odt:?} black"), &rgb);
        }
    }

    // -------------------------------------------------------------------------
    // Rec.2020 HDR variants — PQ ordering by peak
    // -------------------------------------------------------------------------

    #[test]
    fn test_rec2020_hdr_variants_ordering() {
        // Same linear input → higher peak nits → more of the 10000-nit range used
        // → higher PQ code value.
        let aces: Rgb = [0.5, 0.5, 0.5];
        let r1000 = aces_odt_rec2020_hdr(&aces, 1000.0);
        let r2000 = aces_odt_rec2020_hdr(&aces, 2000.0);
        let r4000 = aces_odt_rec2020_hdr(&aces, 4000.0);
        for ch in 0..3 {
            assert!(
                r4000[ch] >= r2000[ch] && r2000[ch] >= r1000[ch],
                "PQ codes should increase with peak nits (ch {ch}): \
                 1000={} 2000={} 4000={}",
                r1000[ch],
                r2000[ch],
                r4000[ch],
            );
        }
    }

    #[test]
    fn test_rec2020_1000_via_enum_matches_direct() {
        let aces: Rgb = [0.3, 0.4, 0.5];
        let via_enum = AcesOdt::Rec2020_1000
            .apply(&aces)
            .expect("Rec2020_1000 must succeed");
        let direct = aces_odt_rec2020_hdr(&aces, 1000.0);
        for i in 0..3 {
            assert!(
                (via_enum[i] - direct[i]).abs() < 1e-12,
                "enum vs direct mismatch at channel {i}: {} vs {}",
                via_enum[i],
                direct[i]
            );
        }
    }

    // -------------------------------------------------------------------------
    // Rec.2020 SDR OETF properties
    // -------------------------------------------------------------------------

    #[test]
    fn test_rec2020_oetf_black_and_white() {
        assert!(
            (rec2020_oetf(0.0) - 0.0).abs() < 1e-9,
            "black should map to 0"
        );
        let white = rec2020_oetf(1.0);
        // At linear 1.0 the OETF evaluates to ALPHA*1^0.45 - (ALPHA-1) = 1.0 exactly.
        assert!(
            (white - 1.0).abs() < 1e-6,
            "Rec.2020 OETF(1.0) = {white}, expected 1.0"
        );
    }

    #[test]
    fn test_rec2020_oetf_monotonic() {
        let mut prev = rec2020_oetf(0.0);
        for i in 1..=20 {
            let linear = i as f64 / 20.0;
            let encoded = rec2020_oetf(linear);
            assert!(
                encoded >= prev,
                "Rec.2020 OETF not monotonic at {linear}: {encoded} < {prev}"
            );
            prev = encoded;
        }
    }

    // -------------------------------------------------------------------------
    // sRGB OETF properties (must differ from Rec.709)
    // -------------------------------------------------------------------------

    #[test]
    fn test_srgb_oetf_differs_from_rec709() {
        // The OETFs diverge because their thresholds and powers differ.
        let s = srgb_oetf(0.5);
        let r = rec709_oetf(0.5);
        assert!(
            (s - r).abs() > 1e-4,
            "sRGB and Rec.709 OETFs should differ at 0.5, got sRGB={s} Rec709={r}"
        );
    }

    #[test]
    fn test_srgb_oetf_known_value() {
        // IEC 61966-2-1: 0.5 > 0.0031308, so: 1.055 * 0.5^(1/2.4) − 0.055
        let expected = 1.055_f64 * 0.5_f64.powf(1.0 / 2.4) - 0.055;
        let got = srgb_oetf(0.5);
        assert!(
            (got - expected).abs() < 1e-9,
            "sRGB OETF mismatch at 0.5: {got} vs {expected}"
        );
    }

    #[test]
    fn test_srgb_oetf_linear_segment() {
        // Below 0.0031308 the function is 12.92*x
        let x = 0.001;
        let expected = 12.92 * x;
        let got = srgb_oetf(x);
        assert!(
            (got - expected).abs() < 1e-9,
            "sRGB linear segment: {got} vs {expected}"
        );
    }

    // -------------------------------------------------------------------------
    // DCI-P3 2.6 gamma OETF
    // -------------------------------------------------------------------------

    #[test]
    fn test_gamma_2_6_oetf_known_value() {
        let linear = 0.5_f64;
        let expected = linear.powf(1.0 / 2.6);
        let got = gamma_2_6_oetf(linear);
        assert!(
            (got - expected).abs() < 1e-9,
            "2.6 OETF at 0.5: {got} vs {expected}"
        );
    }

    #[test]
    fn test_gamma_2_6_oetf_boundary() {
        assert_eq!(gamma_2_6_oetf(0.0), 0.0, "black in -> 0");
        assert_eq!(gamma_2_6_oetf(-1.0), 0.0, "negative in -> 0");
        assert!((gamma_2_6_oetf(1.0) - 1.0).abs() < 1e-12, "white in -> 1");
    }

    // -------------------------------------------------------------------------
    // PQ OETF properties
    // -------------------------------------------------------------------------

    #[test]
    fn test_pq_oetf_100_nits() {
        // 100 nits is a standard reference level; PQ code is approximately 0.508.
        let code = pq_oetf(100.0);
        assert!(
            (code - 0.508).abs() < 0.003,
            "PQ(100 nits) = {code}, expected ~0.508"
        );
    }

    #[test]
    fn test_pq_oetf_10000_nits_is_one() {
        let code = pq_oetf(10_000.0);
        assert!(
            (code - 1.0).abs() < 1e-6,
            "PQ(10 000 nits) = {code}, expected 1.0"
        );
    }

    #[test]
    fn test_pq_oetf_zero_and_negative() {
        assert_eq!(pq_oetf(0.0), 0.0);
        assert_eq!(pq_oetf(-100.0), 0.0);
    }

    #[test]
    fn test_pq_oetf_monotonic() {
        let nit_levels = [0.0_f64, 1.0, 10.0, 100.0, 400.0, 1000.0, 4000.0, 10_000.0];
        let mut prev = pq_oetf(nit_levels[0]);
        for &nits in &nit_levels[1..] {
            let code = pq_oetf(nits);
            assert!(
                code >= prev,
                "PQ OETF not monotonic: pq({nits}) = {code} < {prev}"
            );
            prev = code;
        }
    }

    // -------------------------------------------------------------------------
    // Gray-axis monotonicity: brighter AP0 neutral → brighter output
    //
    // Note: equal R=G=B in ACES2065-1 does NOT produce equal R=G=B out because
    // the ACES2065-1 primaries differ from all display primaries.  What we can
    // verify is that a brighter ACES2065-1 neutral produces a brighter average
    // display output (luminance is monotonically preserved).
    // -------------------------------------------------------------------------

    #[test]
    fn test_gray_axis_monotone_rec709() {
        let dim: Rgb = [0.1, 0.1, 0.1];
        let bright: Rgb = [0.5, 0.5, 0.5];
        let out_dim = aces_odt_rec709(&dim);
        let out_bright = aces_odt_rec709(&bright);
        let lum_dim = (out_dim[0] + out_dim[1] + out_dim[2]) / 3.0;
        let lum_bright = (out_bright[0] + out_bright[1] + out_bright[2]) / 3.0;
        assert!(
            lum_bright > lum_dim,
            "Rec.709: brighter AP0 input should produce higher average output: \
             dim={lum_dim} bright={lum_bright}"
        );
    }

    #[test]
    fn test_gray_axis_monotone_srgb() {
        let dim: Rgb = [0.1, 0.1, 0.1];
        let bright: Rgb = [0.5, 0.5, 0.5];
        let out_dim = aces_odt_srgb(&dim);
        let out_bright = aces_odt_srgb(&bright);
        let lum_dim = (out_dim[0] + out_dim[1] + out_dim[2]) / 3.0;
        let lum_bright = (out_bright[0] + out_bright[1] + out_bright[2]) / 3.0;
        assert!(
            lum_bright > lum_dim,
            "sRGB: brighter AP0 input should produce higher average output: \
             dim={lum_dim} bright={lum_bright}"
        );
    }

    #[test]
    fn test_gray_axis_monotone_dcip3() {
        let dim: Rgb = [0.1, 0.1, 0.1];
        let bright: Rgb = [0.5, 0.5, 0.5];
        let out_dim = aces_odt_dcip3(&dim);
        let out_bright = aces_odt_dcip3(&bright);
        let lum_dim = (out_dim[0] + out_dim[1] + out_dim[2]) / 3.0;
        let lum_bright = (out_bright[0] + out_bright[1] + out_bright[2]) / 3.0;
        assert!(
            lum_bright > lum_dim,
            "DCI-P3: brighter AP0 input should produce higher average output: \
             dim={lum_dim} bright={lum_bright}"
        );
    }

    #[test]
    fn test_gray_axis_monotone_rec2020_hdr() {
        let dim: Rgb = [0.1, 0.1, 0.1];
        let bright: Rgb = [0.5, 0.5, 0.5];
        let out_dim = aces_odt_rec2020_hdr(&dim, 1000.0);
        let out_bright = aces_odt_rec2020_hdr(&bright, 1000.0);
        let lum_dim = (out_dim[0] + out_dim[1] + out_dim[2]) / 3.0;
        let lum_bright = (out_bright[0] + out_bright[1] + out_bright[2]) / 3.0;
        assert!(
            lum_bright > lum_dim,
            "Rec.2020/1000: brighter AP0 input should produce higher average output: \
             dim={lum_dim} bright={lum_bright}"
        );
    }

    // -------------------------------------------------------------------------
    // Clamping: extreme bright inputs must stay in [0,1]
    // -------------------------------------------------------------------------

    #[test]
    fn test_odt_clamping_extreme_bright() {
        let bright: Rgb = [100.0, 100.0, 100.0];
        let variants = [
            AcesOdt::Rec709,
            AcesOdt::Rec2020_100,
            AcesOdt::Rec2020_1000,
            AcesOdt::Rec2020_2000,
            AcesOdt::Rec2020_4000,
            AcesOdt::DciP3,
            AcesOdt::Srgb,
        ];
        for odt in &variants {
            let rgb = odt.apply(&bright).expect("should not error");
            assert_in_range(&format!("{odt:?} bright"), &rgb);
        }
    }
}
