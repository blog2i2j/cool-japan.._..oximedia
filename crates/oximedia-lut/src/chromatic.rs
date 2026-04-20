//! Chromatic adaptation transforms.
//!
//! This module provides chromatic adaptation methods for converting colors
//! between different white points (e.g., D65 to D50).
//!
//! # Methods
//!
//! - **Bradford**: Industry standard, most accurate
//! - **Von Kries**: Classic method
//! - **XYZ Scaling**: Simple but less accurate

use crate::matrix::{apply_matrix3x3, invert_matrix3x3, multiply_matrix3x3};
use crate::{Matrix3x3, Rgb, Xyz};

/// Chromatic adaptation method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaticAdaptation {
    /// Bradford transform (recommended).
    Bradford,
    /// Von Kries transform.
    VonKries,
    /// Simple XYZ scaling.
    XyzScaling,
}

/// Standard illuminants (CIE 1931 2° observer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Illuminant {
    /// XYZ tristimulus values (normalized so Y = 1.0).
    pub xyz: Xyz,
}

impl Illuminant {
    /// D65 illuminant (daylight, 6500K).
    pub const D65: Self = Self {
        xyz: [0.95047, 1.0, 1.08883],
    };

    /// D50 illuminant (horizon light, 5000K).
    pub const D50: Self = Self {
        xyz: [0.96422, 1.0, 0.82521],
    };

    /// D60 illuminant (6000K, used in DCI-P3).
    pub const D60: Self = Self {
        xyz: [0.95265, 1.0, 1.00882],
    };

    /// D55 illuminant (5500K).
    pub const D55: Self = Self {
        xyz: [0.95682, 1.0, 0.92149],
    };

    /// Illuminant A (tungsten, 2856K).
    pub const A: Self = Self {
        xyz: [1.09850, 1.0, 0.35585],
    };

    /// Illuminant C (average daylight, 6774K).
    pub const C: Self = Self {
        xyz: [0.98074, 1.0, 1.18232],
    };

    /// E (equal energy).
    pub const E: Self = Self {
        xyz: [1.0, 1.0, 1.0],
    };
}

/// Bradford chromatic adaptation matrix.
const BRADFORD: Matrix3x3 = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Von Kries chromatic adaptation matrix.
const VON_KRIES: Matrix3x3 = [
    [0.40024, 0.70760, -0.08081],
    [-0.22630, 1.16532, 0.04570],
    [0.00000, 0.00000, 0.91822],
];

/// Compute chromatic adaptation matrix from source to destination illuminant.
#[must_use]
pub fn adaptation_matrix(
    source: &Illuminant,
    dest: &Illuminant,
    method: ChromaticAdaptation,
) -> Matrix3x3 {
    match method {
        ChromaticAdaptation::Bradford => bradford_adaptation(&source.xyz, &dest.xyz),
        ChromaticAdaptation::VonKries => von_kries_adaptation(&source.xyz, &dest.xyz),
        ChromaticAdaptation::XyzScaling => xyz_scaling_adaptation(&source.xyz, &dest.xyz),
    }
}

/// Bradford chromatic adaptation.
///
/// # Panics
///
/// Panics if matrix inversion fails (degenerate matrix).
#[must_use]
pub fn bradford_adaptation(source_wp: &Xyz, dest_wp: &Xyz) -> Matrix3x3 {
    // Convert white points to cone response domain
    let source_rgb = apply_matrix3x3(&BRADFORD, source_wp);
    let dest_rgb = apply_matrix3x3(&BRADFORD, dest_wp);

    // Compute scaling matrix
    let scale = [
        [dest_rgb[0] / source_rgb[0], 0.0, 0.0],
        [0.0, dest_rgb[1] / source_rgb[1], 0.0],
        [0.0, 0.0, dest_rgb[2] / source_rgb[2]],
    ];

    // Compute final adaptation matrix: BRADFORD^-1 * scale * BRADFORD
    // BRADFORD is a known invertible constant — fall back to identity only on degenerate inputs
    let bradford_inv =
        invert_matrix3x3(&BRADFORD).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let temp = multiply_matrix3x3(&scale, &BRADFORD);
    multiply_matrix3x3(&bradford_inv, &temp)
}

/// Von Kries chromatic adaptation.
///
/// # Panics
///
/// Panics if matrix inversion fails (degenerate matrix).
#[must_use]
pub fn von_kries_adaptation(source_wp: &Xyz, dest_wp: &Xyz) -> Matrix3x3 {
    let source_rgb = apply_matrix3x3(&VON_KRIES, source_wp);
    let dest_rgb = apply_matrix3x3(&VON_KRIES, dest_wp);

    let scale = [
        [dest_rgb[0] / source_rgb[0], 0.0, 0.0],
        [0.0, dest_rgb[1] / source_rgb[1], 0.0],
        [0.0, 0.0, dest_rgb[2] / source_rgb[2]],
    ];

    // VON_KRIES is a known invertible constant — fall back to identity only on degenerate inputs
    let von_kries_inv =
        invert_matrix3x3(&VON_KRIES).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let temp = multiply_matrix3x3(&scale, &VON_KRIES);
    multiply_matrix3x3(&von_kries_inv, &temp)
}

/// Simple XYZ scaling adaptation.
#[must_use]
pub fn xyz_scaling_adaptation(source_wp: &Xyz, dest_wp: &Xyz) -> Matrix3x3 {
    [
        [dest_wp[0] / source_wp[0], 0.0, 0.0],
        [0.0, dest_wp[1] / source_wp[1], 0.0],
        [0.0, 0.0, dest_wp[2] / source_wp[2]],
    ]
}

/// Apply chromatic adaptation to an XYZ color.
#[must_use]
pub fn adapt_xyz(
    xyz: &Xyz,
    source: &Illuminant,
    dest: &Illuminant,
    method: ChromaticAdaptation,
) -> Xyz {
    let matrix = adaptation_matrix(source, dest, method);
    apply_matrix3x3(&matrix, xyz)
}

/// Apply chromatic adaptation to an RGB color.
///
/// This converts RGB to XYZ, applies adaptation, and converts back.
#[must_use]
pub fn adapt_rgb(
    rgb: &Rgb,
    source: &Illuminant,
    dest: &Illuminant,
    rgb_to_xyz: &Matrix3x3,
    xyz_to_rgb: &Matrix3x3,
    method: ChromaticAdaptation,
) -> Rgb {
    // Convert to XYZ
    let xyz = apply_matrix3x3(rgb_to_xyz, rgb);

    // Apply adaptation
    let adapted = adapt_xyz(&xyz, source, dest, method);

    // Convert back to RGB
    apply_matrix3x3(xyz_to_rgb, &adapted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_white_point() {
        let xyz = [0.5, 0.5, 0.5];
        let adapted = adapt_xyz(
            &xyz,
            &Illuminant::D65,
            &Illuminant::D65,
            ChromaticAdaptation::Bradford,
        );
        assert!((xyz[0] - adapted[0]).abs() < 1e-6);
        assert!((xyz[1] - adapted[1]).abs() < 1e-6);
        assert!((xyz[2] - adapted[2]).abs() < 1e-6);
    }

    #[test]
    fn test_bradford_adaptation() {
        // Test D65 to D50 adaptation
        let xyz = [0.5, 0.5, 0.5];
        let adapted = adapt_xyz(
            &xyz,
            &Illuminant::D65,
            &Illuminant::D50,
            ChromaticAdaptation::Bradford,
        );
        // The adapted color should be different (though may be subtle for neutral gray)
        let changed = (xyz[0] - adapted[0]).abs() > 0.001
            || (xyz[1] - adapted[1]).abs() > 0.001
            || (xyz[2] - adapted[2]).abs() > 0.001;
        assert!(changed, "Adaptation should change the color");
    }

    #[test]
    fn test_von_kries_adaptation() {
        let xyz = [0.5, 0.5, 0.5];
        let adapted = adapt_xyz(
            &xyz,
            &Illuminant::D65,
            &Illuminant::D50,
            ChromaticAdaptation::VonKries,
        );
        let changed = (xyz[0] - adapted[0]).abs() > 0.001
            || (xyz[1] - adapted[1]).abs() > 0.001
            || (xyz[2] - adapted[2]).abs() > 0.001;
        assert!(changed, "Adaptation should change the color");
    }

    #[test]
    fn test_xyz_scaling_adaptation() {
        let xyz = [0.5, 0.5, 0.5];
        let adapted = adapt_xyz(
            &xyz,
            &Illuminant::D65,
            &Illuminant::D50,
            ChromaticAdaptation::XyzScaling,
        );
        let changed = (xyz[0] - adapted[0]).abs() > 0.001
            || (xyz[1] - adapted[1]).abs() > 0.001
            || (xyz[2] - adapted[2]).abs() > 0.001;
        assert!(changed, "Adaptation should change the color");
    }

    #[test]
    fn test_adaptation_reversible() {
        let xyz = [0.5, 0.5, 0.5];
        let adapted = adapt_xyz(
            &xyz,
            &Illuminant::D65,
            &Illuminant::D50,
            ChromaticAdaptation::Bradford,
        );
        let back = adapt_xyz(
            &adapted,
            &Illuminant::D50,
            &Illuminant::D65,
            ChromaticAdaptation::Bradford,
        );
        assert!((xyz[0] - back[0]).abs() < 1e-6);
        assert!((xyz[1] - back[1]).abs() < 1e-6);
        assert!((xyz[2] - back[2]).abs() < 1e-6);
    }
}
