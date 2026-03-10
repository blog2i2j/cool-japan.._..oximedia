//! Color transformation functions and operations.

use crate::colorspaces::ColorSpace;
use crate::error::Result;
use crate::math::chromatic_adaptation::{chromatic_adaptation_transform, ChromaticAdaptation};
use crate::math::matrix::{multiply_matrices, multiply_matrix_vector, Matrix3x3};

pub mod lut;
pub mod lut_formats;
pub mod parametric;

/// Converts RGB from one color space to another.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `src` - Source color space
/// * `dst` - Destination color space
///
/// # Returns
///
/// Converted RGB values [0, 1] (may be out of gamut, use gamut mapping if needed)
#[must_use]
pub fn rgb_to_rgb(rgb: &[f64; 3], src: &ColorSpace, dst: &ColorSpace) -> [f64; 3] {
    // Convert to XYZ
    let xyz = src.rgb_to_xyz(*rgb);

    // Apply chromatic adaptation if white points differ
    let xyz_adapted = if src.white_point == dst.white_point {
        xyz
    } else {
        let src_white = src.white_point_xyz();
        let dst_white = dst.white_point_xyz();

        if let Ok(adaptation_matrix) = chromatic_adaptation_transform(
            src_white.as_array(),
            dst_white.as_array(),
            &ChromaticAdaptation::Bradford.matrix(),
        ) {
            let adapted = multiply_matrix_vector(&adaptation_matrix, xyz.as_array());
            crate::xyz::Xyz::from_array(adapted)
        } else {
            xyz
        }
    };

    // Convert to destination RGB
    dst.xyz_to_rgb(&xyz_adapted)
}

/// Creates a combined transformation matrix for converting RGB between color spaces.
///
/// # Arguments
///
/// * `src` - Source color space
/// * `dst` - Destination color space
///
/// # Returns
///
/// 3x3 transformation matrix that converts linearized RGB from src to linearized RGB in dst.
///
/// Note: This doesn't include transfer functions - apply EOTF before and OETF after.
///
/// # Errors
///
/// Returns an error if chromatic adaptation fails.
pub fn create_rgb_to_rgb_matrix(src: &ColorSpace, dst: &ColorSpace) -> Result<Matrix3x3> {
    // Start with src RGB to XYZ
    let mut transform = src.rgb_to_xyz;

    // Apply chromatic adaptation if needed
    if src.white_point != dst.white_point {
        let src_white = src.white_point_xyz();
        let dst_white = dst.white_point_xyz();

        let adaptation = chromatic_adaptation_transform(
            src_white.as_array(),
            dst_white.as_array(),
            &ChromaticAdaptation::Bradford.matrix(),
        )?;

        transform = multiply_matrices(&adaptation, &transform);
    }

    // Apply XYZ to dst RGB
    Ok(multiply_matrices(&dst.xyz_to_rgb, &transform))
}

/// Applies a 3x3 matrix transformation to RGB values.
#[must_use]
pub fn apply_matrix(rgb: [f64; 3], matrix: &Matrix3x3) -> [f64; 3] {
    multiply_matrix_vector(matrix, rgb)
}

/// Applies a 1D LUT to each channel independently.
///
/// # Arguments
///
/// * `rgb` - Input RGB values [0, 1]
/// * `lut_r` - Red channel LUT
/// * `lut_g` - Green channel LUT
/// * `lut_b` - Blue channel LUT
#[must_use]
pub fn apply_1d_lut(rgb: [f64; 3], lut_r: &[f32], lut_g: &[f32], lut_b: &[f32]) -> [f64; 3] {
    [
        interpolate_1d_lut(rgb[0], lut_r),
        interpolate_1d_lut(rgb[1], lut_g),
        interpolate_1d_lut(rgb[2], lut_b),
    ]
}

/// Interpolates a value in a 1D LUT.
fn interpolate_1d_lut(value: f64, lut: &[f32]) -> f64 {
    if lut.is_empty() {
        return value;
    }

    let size = lut.len();
    let pos = value * (size - 1) as f64;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;

    if idx >= size - 1 {
        return f64::from(lut[size - 1]);
    }

    let v0 = f64::from(lut[idx]);
    let v1 = f64::from(lut[idx + 1]);

    v0 + (v1 - v0) * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_rgb_same_space() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rgb = [0.5, 0.3, 0.7];

        let result = rgb_to_rgb(&rgb, &srgb, &srgb);

        assert!((result[0] - rgb[0]).abs() < 1e-6);
        assert!((result[1] - rgb[1]).abs() < 1e-6);
        assert!((result[2] - rgb[2]).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_to_rgb_different_spaces() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rec2020 = ColorSpace::rec2020().expect("Rec.2020 color space creation should succeed");

        let rgb = [0.5, 0.3, 0.7];
        let result = rgb_to_rgb(&rgb, &srgb, &rec2020);

        // Result should be different but valid
        assert!(result[0] >= 0.0 && result[0] <= 1.5); // May be out of gamut
        assert!(result[1] >= 0.0 && result[1] <= 1.5);
        assert!(result[2] >= 0.0 && result[2] <= 1.5);
    }

    #[test]
    fn test_create_rgb_to_rgb_matrix() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rec709 = ColorSpace::rec709().expect("Rec.709 color space creation should succeed");

        let matrix = create_rgb_to_rgb_matrix(&srgb, &rec709)
            .expect("RGB to RGB matrix creation should succeed");

        // Matrix should be close to identity (same primaries)
        assert!((matrix[0][0] - 1.0).abs() < 0.01);
        assert!((matrix[1][1] - 1.0).abs() < 0.01);
        assert!((matrix[2][2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_apply_matrix_identity() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rgb = [0.5, 0.3, 0.7];

        let result = apply_matrix(rgb, &identity);

        assert_eq!(result, rgb);
    }

    #[test]
    fn test_apply_1d_lut_identity() {
        let lut: Vec<f32> = (0..256).map(|i| i as f32 / 255.0).collect();
        let rgb = [0.5, 0.3, 0.7];

        let result = apply_1d_lut(rgb, &lut, &lut, &lut);

        assert!((result[0] - rgb[0]).abs() < 0.01);
        assert!((result[1] - rgb[1]).abs() < 0.01);
        assert!((result[2] - rgb[2]).abs() < 0.01);
    }
}
