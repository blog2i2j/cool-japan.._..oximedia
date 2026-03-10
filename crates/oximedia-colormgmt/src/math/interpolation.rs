//! Interpolation functions for LUT operations.

use crate::error::{ColorError, Result};

/// Linear interpolation between two values.
///
/// # Arguments
///
/// * `a` - Start value
/// * `b` - End value
/// * `t` - Interpolation factor [0, 1]
#[must_use]
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Trilinear interpolation in a 3D LUT.
///
/// # Arguments
///
/// * `lut` - 3D LUT data (R, G, B values interleaved)
/// * `size` - Size of each dimension of the LUT
/// * `rgb` - Input RGB values [0, 1]
///
/// # Errors
///
/// Returns an error if LUT size is invalid or indices are out of bounds.
pub fn trilinear_interpolate(lut: &[f32], size: usize, rgb: [f64; 3]) -> Result<[f64; 3]> {
    if size < 2 {
        return Err(ColorError::Lut("LUT size must be at least 2".to_string()));
    }

    // Scale input to LUT coordinates
    let r = rgb[0] * (size - 1) as f64;
    let g = rgb[1] * (size - 1) as f64;
    let b = rgb[2] * (size - 1) as f64;

    // Get integer parts
    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;

    // Get fractional parts
    let rf = r - r0 as f64;
    let gf = g - g0 as f64;
    let bf = b - b0 as f64;

    // Get upper indices (clamped)
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    // Helper function to get LUT value
    let get_lut = |ri: usize, gi: usize, bi: usize| -> Result<[f64; 3]> {
        let idx = (ri * size * size + gi * size + bi) * 3;
        if idx + 2 >= lut.len() {
            return Err(ColorError::Lut(format!(
                "LUT index out of bounds: {} >= {}",
                idx + 2,
                lut.len()
            )));
        }
        Ok([
            f64::from(lut[idx]),
            f64::from(lut[idx + 1]),
            f64::from(lut[idx + 2]),
        ])
    };

    // Get all 8 corner values
    let c000 = get_lut(r0, g0, b0)?;
    let c001 = get_lut(r0, g0, b1)?;
    let c010 = get_lut(r0, g1, b0)?;
    let c011 = get_lut(r0, g1, b1)?;
    let c100 = get_lut(r1, g0, b0)?;
    let c101 = get_lut(r1, g0, b1)?;
    let c110 = get_lut(r1, g1, b0)?;
    let c111 = get_lut(r1, g1, b1)?;

    // Interpolate along B axis
    let c00 = [
        lerp(c000[0], c001[0], bf),
        lerp(c000[1], c001[1], bf),
        lerp(c000[2], c001[2], bf),
    ];
    let c01 = [
        lerp(c010[0], c011[0], bf),
        lerp(c010[1], c011[1], bf),
        lerp(c010[2], c011[2], bf),
    ];
    let c10 = [
        lerp(c100[0], c101[0], bf),
        lerp(c100[1], c101[1], bf),
        lerp(c100[2], c101[2], bf),
    ];
    let c11 = [
        lerp(c110[0], c111[0], bf),
        lerp(c110[1], c111[1], bf),
        lerp(c110[2], c111[2], bf),
    ];

    // Interpolate along G axis
    let c0 = [
        lerp(c00[0], c01[0], gf),
        lerp(c00[1], c01[1], gf),
        lerp(c00[2], c01[2], gf),
    ];
    let c1 = [
        lerp(c10[0], c11[0], gf),
        lerp(c10[1], c11[1], gf),
        lerp(c10[2], c11[2], gf),
    ];

    // Interpolate along R axis
    Ok([
        lerp(c0[0], c1[0], rf),
        lerp(c0[1], c1[1], rf),
        lerp(c0[2], c1[2], rf),
    ])
}

/// Tetrahedral interpolation in a 3D LUT (more accurate than trilinear).
///
/// Tetrahedral interpolation divides each cube into 6 tetrahedra and
/// interpolates within the appropriate tetrahedron.
///
/// # Arguments
///
/// * `lut` - 3D LUT data (R, G, B values interleaved)
/// * `size` - Size of each dimension of the LUT
/// * `rgb` - Input RGB values [0, 1]
///
/// # Errors
///
/// Returns an error if LUT size is invalid or indices are out of bounds.
#[allow(clippy::too_many_lines)]
pub fn tetrahedral_interpolate(lut: &[f32], size: usize, rgb: [f64; 3]) -> Result<[f64; 3]> {
    if size < 2 {
        return Err(ColorError::Lut("LUT size must be at least 2".to_string()));
    }

    // Scale input to LUT coordinates
    let r = rgb[0] * (size - 1) as f64;
    let g = rgb[1] * (size - 1) as f64;
    let b = rgb[2] * (size - 1) as f64;

    // Get integer parts
    let r0 = r.floor() as usize;
    let g0 = g.floor() as usize;
    let b0 = b.floor() as usize;

    // Get fractional parts
    let rf = r - r0 as f64;
    let gf = g - g0 as f64;
    let bf = b - b0 as f64;

    // Get upper indices (clamped)
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    // Helper function to get LUT value
    let get_lut = |ri: usize, gi: usize, bi: usize| -> Result<[f64; 3]> {
        let idx = (ri * size * size + gi * size + bi) * 3;
        if idx + 2 >= lut.len() {
            return Err(ColorError::Lut(format!(
                "LUT index out of bounds: {} >= {}",
                idx + 2,
                lut.len()
            )));
        }
        Ok([
            f64::from(lut[idx]),
            f64::from(lut[idx + 1]),
            f64::from(lut[idx + 2]),
        ])
    };

    let c000 = get_lut(r0, g0, b0)?;
    let c111 = get_lut(r1, g1, b1)?;

    let result = if rf > gf {
        if gf > bf {
            // rf > gf > bf
            let c100 = get_lut(r1, g0, b0)?;
            let c110 = get_lut(r1, g1, b0)?;
            [
                c000[0]
                    + rf * (c100[0] - c000[0])
                    + gf * (c110[0] - c100[0])
                    + bf * (c111[0] - c110[0]),
                c000[1]
                    + rf * (c100[1] - c000[1])
                    + gf * (c110[1] - c100[1])
                    + bf * (c111[1] - c110[1]),
                c000[2]
                    + rf * (c100[2] - c000[2])
                    + gf * (c110[2] - c100[2])
                    + bf * (c111[2] - c110[2]),
            ]
        } else if rf > bf {
            // rf > bf >= gf
            let c100 = get_lut(r1, g0, b0)?;
            let c101 = get_lut(r1, g0, b1)?;
            [
                c000[0]
                    + rf * (c100[0] - c000[0])
                    + bf * (c101[0] - c100[0])
                    + gf * (c111[0] - c101[0]),
                c000[1]
                    + rf * (c100[1] - c000[1])
                    + bf * (c101[1] - c100[1])
                    + gf * (c111[1] - c101[1]),
                c000[2]
                    + rf * (c100[2] - c000[2])
                    + bf * (c101[2] - c100[2])
                    + gf * (c111[2] - c101[2]),
            ]
        } else {
            // bf >= rf > gf
            let c001 = get_lut(r0, g0, b1)?;
            let c101 = get_lut(r1, g0, b1)?;
            [
                c000[0]
                    + bf * (c001[0] - c000[0])
                    + rf * (c101[0] - c001[0])
                    + gf * (c111[0] - c101[0]),
                c000[1]
                    + bf * (c001[1] - c000[1])
                    + rf * (c101[1] - c001[1])
                    + gf * (c111[1] - c101[1]),
                c000[2]
                    + bf * (c001[2] - c000[2])
                    + rf * (c101[2] - c001[2])
                    + gf * (c111[2] - c101[2]),
            ]
        }
    } else if bf > gf {
        // bf > gf >= rf
        let c001 = get_lut(r0, g0, b1)?;
        let c011 = get_lut(r0, g1, b1)?;
        [
            c000[0]
                + bf * (c001[0] - c000[0])
                + gf * (c011[0] - c001[0])
                + rf * (c111[0] - c011[0]),
            c000[1]
                + bf * (c001[1] - c000[1])
                + gf * (c011[1] - c001[1])
                + rf * (c111[1] - c011[1]),
            c000[2]
                + bf * (c001[2] - c000[2])
                + gf * (c011[2] - c001[2])
                + rf * (c111[2] - c011[2]),
        ]
    } else if gf > bf {
        // gf > bf >= rf
        let c010 = get_lut(r0, g1, b0)?;
        let c011 = get_lut(r0, g1, b1)?;
        [
            c000[0]
                + gf * (c010[0] - c000[0])
                + bf * (c011[0] - c010[0])
                + rf * (c111[0] - c011[0]),
            c000[1]
                + gf * (c010[1] - c000[1])
                + bf * (c011[1] - c010[1])
                + rf * (c111[1] - c011[1]),
            c000[2]
                + gf * (c010[2] - c000[2])
                + bf * (c011[2] - c010[2])
                + rf * (c111[2] - c011[2]),
        ]
    } else {
        // gf == bf >= rf
        let c010 = get_lut(r0, g1, b0)?;
        let c011 = get_lut(r0, g1, b1)?;
        [
            c000[0]
                + gf * (c010[0] - c000[0])
                + bf * (c011[0] - c010[0])
                + rf * (c111[0] - c011[0]),
            c000[1]
                + gf * (c010[1] - c000[1])
                + bf * (c011[1] - c010[1])
                + rf * (c111[1] - c011[1]),
            c000[2]
                + gf * (c010[2] - c000[2])
                + bf * (c011[2] - c010[2])
                + rf * (c111[2] - c011[2]),
        ]
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((lerp(0.0, 1.0, 0.0) - 0.0).abs() < 1e-10);
        assert!((lerp(0.0, 1.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_trilinear_interpolate_identity() {
        // Create a 2x2x2 identity LUT
        #[allow(clippy::excessive_precision)]
        let lut: Vec<f32> = vec![
            0.0, 0.0, 0.0, // (0,0,0)
            0.0, 0.0, 1.0, // (0,0,1)
            0.0, 1.0, 0.0, // (0,1,0)
            0.0, 1.0, 1.0, // (0,1,1)
            1.0, 0.0, 0.0, // (1,0,0)
            1.0, 0.0, 1.0, // (1,0,1)
            1.0, 1.0, 0.0, // (1,1,0)
            1.0, 1.0, 1.0, // (1,1,1)
        ];

        let result = trilinear_interpolate(&lut, 2, [0.5, 0.5, 0.5])
            .expect("trilinear interpolation should succeed");
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
        assert!((result[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_tetrahedral_interpolate_identity() {
        // Create a 2x2x2 identity LUT
        #[allow(clippy::excessive_precision)]
        let lut: Vec<f32> = vec![
            0.0, 0.0, 0.0, // (0,0,0)
            0.0, 0.0, 1.0, // (0,0,1)
            0.0, 1.0, 0.0, // (0,1,0)
            0.0, 1.0, 1.0, // (0,1,1)
            1.0, 0.0, 0.0, // (1,0,0)
            1.0, 0.0, 1.0, // (1,0,1)
            1.0, 1.0, 0.0, // (1,1,0)
            1.0, 1.0, 1.0, // (1,1,1)
        ];

        let result = tetrahedral_interpolate(&lut, 2, [0.5, 0.5, 0.5])
            .expect("tetrahedral interpolation should succeed");
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!((result[1] - 0.5).abs() < 0.01);
        assert!((result[2] - 0.5).abs() < 0.01);
    }
}
