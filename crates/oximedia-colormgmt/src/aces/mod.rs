//! ACES (Academy Color Encoding System) support.
//!
//! This module implements ACES color spaces and transforms including:
//! - ACES2065-1 (linear, AP0 primaries)
//! - `ACEScg` (linear, AP1 primaries)
//! - `ACEScc` (logarithmic, for color grading)
//! - `ACEScct` (logarithmic with toe, for color grading)
//! - IDT, RRT, ODT transforms

use crate::error::Result;
use crate::math::matrix::{multiply_matrix_vector, Matrix3x3};
use oximedia_core::hdr::Primaries;

/// ACES color space variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AcesColorSpace {
    /// ACES2065-1 - AP0 primaries, linear, scene-referred
    ACES2065_1,
    /// `ACEScg` - AP1 primaries, linear, working space for CG
    ACEScg,
    /// `ACEScc` - AP1 primaries, logarithmic encoding for color grading
    ACEScc,
    /// `ACEScct` - AP1 primaries, logarithmic with toe, for color grading
    ACEScct,
}

impl AcesColorSpace {
    /// Returns the primaries for this ACES color space.
    #[must_use]
    pub const fn primaries(&self) -> Primaries {
        match self {
            Self::ACES2065_1 => AP0_PRIMARIES,
            Self::ACEScg | Self::ACEScc | Self::ACEScct => AP1_PRIMARIES,
        }
    }

    /// Returns the name of this color space.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::ACES2065_1 => "ACES2065-1",
            Self::ACEScg => "ACEScg",
            Self::ACEScc => "ACEScc",
            Self::ACEScct => "ACEScct",
        }
    }
}

/// AP0 primaries (ACES2065-1).
const AP0_PRIMARIES: Primaries = Primaries {
    red: (0.7347, 0.2653),
    green: (0.0, 1.0),
    blue: (0.0001, -0.077),
};

/// AP1 primaries (`ACEScg`, `ACEScc`, `ACEScct`).
const AP1_PRIMARIES: Primaries = Primaries {
    red: (0.713, 0.293),
    green: (0.165, 0.830),
    blue: (0.128, 0.044),
};

/// AP0 to AP1 transformation matrix.
const AP0_TO_AP1: Matrix3x3 = [
    [1.451_439_316_2, -0.236_510_746_9, -0.214_928_569_3],
    [-0.076_553_773_2, 1.176_229_699_9, -0.099_675_926_7],
    [0.008_316_148_3, -0.006_032_449_8, 0.997_716_301_5],
];

/// AP1 to AP0 transformation matrix.
const AP1_TO_AP0: Matrix3x3 = [
    [0.695_452_241_4, 0.140_678_696_5, 0.163_869_062_1],
    [0.044_794_563_7, 0.859_671_118_5, 0.095_534_317_8],
    [-0.005_525_882_6, 0.004_025_210_3, 1.001_500_672_3],
];

/// ACES transform for converting between ACES color spaces.
pub struct AcesTransform {
    src: AcesColorSpace,
    dst: AcesColorSpace,
}

impl AcesTransform {
    /// Creates a new ACES transform.
    #[must_use]
    pub const fn new(src: AcesColorSpace, dst: AcesColorSpace) -> Self {
        Self { src, dst }
    }

    /// Applies the transform to RGB values.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform is invalid.
    pub fn apply(&self, rgb: [f64; 3]) -> Result<[f64; 3]> {
        // First, convert source to linear AP0 or AP1
        let linear = match self.src {
            AcesColorSpace::ACES2065_1 => rgb, // Already linear AP0
            AcesColorSpace::ACEScg => rgb,     // Already linear AP1
            AcesColorSpace::ACEScc => acescc_to_linear(rgb),
            AcesColorSpace::ACEScct => acescct_to_linear(rgb),
        };

        // Convert between AP0 and AP1 if needed
        let linear = match (self.src, self.dst) {
            (AcesColorSpace::ACES2065_1, AcesColorSpace::ACES2065_1) => linear,
            (AcesColorSpace::ACES2065_1, _) => {
                // AP0 to AP1
                multiply_matrix_vector(&AP0_TO_AP1, linear)
            }
            (_, AcesColorSpace::ACES2065_1) => {
                // AP1 to AP0
                multiply_matrix_vector(&AP1_TO_AP0, linear)
            }
            _ => linear, // Both are AP1
        };

        // Convert to destination encoding
        Ok(match self.dst {
            AcesColorSpace::ACES2065_1 | AcesColorSpace::ACEScg => linear,
            AcesColorSpace::ACEScc => linear_to_acescc(linear),
            AcesColorSpace::ACEScct => linear_to_acescct(linear),
        })
    }
}

/// Converts linear AP1 to `ACEScc` (logarithmic encoding).
///
/// `ACEScc` is designed for color grading with a logarithmic response.
#[must_use]
fn linear_to_acescc(linear: [f64; 3]) -> [f64; 3] {
    let convert = |x: f64| -> f64 {
        if x <= 0.0 {
            (9.72 - 15.0) / 17.52 // Min value
        } else if x < 2.0_f64.powf(-15.0) {
            (x.ln() / std::f64::consts::LN_2 + 9.72) / 17.52
        } else {
            (x.ln() / std::f64::consts::LN_2 + 9.72) / 17.52
        }
    };

    [convert(linear[0]), convert(linear[1]), convert(linear[2])]
}

/// Converts `ACEScc` to linear AP1.
#[must_use]
fn acescc_to_linear(acescc: [f64; 3]) -> [f64; 3] {
    let convert = |x: f64| -> f64 {
        if x < (9.72 - 15.0) / 17.52 {
            0.0
        } else if x < (9.72 + 9.72) / 17.52 {
            2.0_f64.powf(x * 17.52 - 9.72)
        } else {
            2.0_f64.powf(x * 17.52 - 9.72)
        }
    };

    [convert(acescc[0]), convert(acescc[1]), convert(acescc[2])]
}

/// Converts linear AP1 to `ACEScct` (logarithmic encoding with toe).
///
/// `ACEScct` is similar to `ACEScc` but with a linear toe for better behavior near black.
#[must_use]
fn linear_to_acescct(linear: [f64; 3]) -> [f64; 3] {
    const X_BRK: f64 = 0.0078125; // 1/128

    let convert = |x: f64| -> f64 {
        if x <= X_BRK {
            10.5402377416545 * x + 0.0729055341958355
        } else {
            (x.ln() / std::f64::consts::LN_2 + 9.72) / 17.52
        }
    };

    [convert(linear[0]), convert(linear[1]), convert(linear[2])]
}

/// Converts `ACEScct` to linear AP1.
#[must_use]
fn acescct_to_linear(acescct: [f64; 3]) -> [f64; 3] {
    let convert = |x: f64| -> f64 {
        // Y_BRK = 0.155251141552511
        if x <= 0.155_251_141_552_511 {
            (x - 0.0729055341958355) / 10.5402377416545
        } else {
            2.0_f64.powf(x * 17.52 - 9.72)
        }
    };

    [
        convert(acescct[0]),
        convert(acescct[1]),
        convert(acescct[2]),
    ]
}

/// Reference Rendering Transform (RRT) placeholder.
///
/// The full RRT is complex and typically implemented as a LUT.
/// This is a simplified version for basic tone mapping.
#[must_use]
#[allow(dead_code)]
pub fn apply_rrt(aces: [f64; 3]) -> [f64; 3] {
    // Simplified RRT - just a basic tone curve
    let tone_map = |x: f64| {
        let a = 2.51;
        let b = 0.03;
        let c = 2.43;
        let d = 0.59;
        let e = 0.14;

        ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
    };

    [tone_map(aces[0]), tone_map(aces[1]), tone_map(aces[2])]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aces_color_space_names() {
        assert_eq!(AcesColorSpace::ACES2065_1.name(), "ACES2065-1");
        assert_eq!(AcesColorSpace::ACEScg.name(), "ACEScg");
        assert_eq!(AcesColorSpace::ACEScc.name(), "ACEScc");
        assert_eq!(AcesColorSpace::ACEScct.name(), "ACEScct");
    }

    #[test]
    fn test_aces_transform_identity() {
        let transform = AcesTransform::new(AcesColorSpace::ACEScg, AcesColorSpace::ACEScg);
        let rgb = [0.5, 0.3, 0.7];
        let result = transform
            .apply(rgb)
            .expect("transform application should succeed");

        assert!((result[0] - rgb[0]).abs() < 1e-6);
        assert!((result[1] - rgb[1]).abs() < 1e-6);
        assert!((result[2] - rgb[2]).abs() < 1e-6);
    }

    #[test]
    fn test_acescc_roundtrip() {
        let linear = [0.5, 0.3, 0.7];
        let acescc = linear_to_acescc(linear);
        let linear2 = acescc_to_linear(acescc);

        assert!((linear2[0] - linear[0]).abs() < 1e-6);
        assert!((linear2[1] - linear[1]).abs() < 1e-6);
        assert!((linear2[2] - linear[2]).abs() < 1e-6);
    }

    #[test]
    fn test_acescct_roundtrip() {
        let linear = [0.5, 0.3, 0.7];
        let acescct = linear_to_acescct(linear);
        let linear2 = acescct_to_linear(acescct);

        assert!((linear2[0] - linear[0]).abs() < 1e-6);
        assert!((linear2[1] - linear[1]).abs() < 1e-6);
        assert!((linear2[2] - linear[2]).abs() < 1e-6);
    }

    #[test]
    fn test_ap0_ap1_conversion() {
        let transform1 = AcesTransform::new(AcesColorSpace::ACES2065_1, AcesColorSpace::ACEScg);
        let transform2 = AcesTransform::new(AcesColorSpace::ACEScg, AcesColorSpace::ACES2065_1);

        let rgb = [0.5, 0.3, 0.7];
        let ap1 = transform1
            .apply(rgb)
            .expect("transform application should succeed");
        let ap0 = transform2
            .apply(ap1)
            .expect("transform application should succeed");

        assert!((ap0[0] - rgb[0]).abs() < 1e-6);
        assert!((ap0[1] - rgb[1]).abs() < 1e-6);
        assert!((ap0[2] - rgb[2]).abs() < 1e-6);
    }
}
