//! LUT-based color transforms.

use crate::error::{ColorError, Result};
use crate::math::interpolation::{tetrahedral_interpolate, trilinear_interpolate};

/// 3D LUT for color transformations.
#[derive(Clone, Debug)]
pub struct Lut3D {
    /// LUT data (RGB triplets)
    pub data: Vec<f32>,
    /// Size of each dimension
    pub size: usize,
    /// Interpolation method
    pub interpolation: InterpolationMethod,
}

/// Interpolation method for 3D LUTs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Trilinear interpolation (faster)
    Trilinear,
    /// Tetrahedral interpolation (more accurate)
    Tetrahedral,
}

impl Lut3D {
    /// Creates a new 3D LUT.
    ///
    /// # Arguments
    ///
    /// * `data` - LUT data (R, G, B values interleaved)
    /// * `size` - Size of each dimension (e.g., 17, 33, 65)
    ///
    /// # Errors
    ///
    /// Returns an error if the data size doesn't match size³ × 3.
    pub fn new(data: Vec<f32>, size: usize) -> Result<Self> {
        let expected_len = size * size * size * 3;
        if data.len() != expected_len {
            return Err(ColorError::Lut(format!(
                "Invalid LUT data size: expected {}, got {}",
                expected_len,
                data.len()
            )));
        }

        Ok(Self {
            data,
            size,
            interpolation: InterpolationMethod::Tetrahedral,
        })
    }

    /// Creates an identity 3D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let mut data = Vec::with_capacity(size * size * size * 3);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let rf = r as f32 / (size - 1) as f32;
                    let gf = g as f32 / (size - 1) as f32;
                    let bf = b as f32 / (size - 1) as f32;
                    data.push(rf);
                    data.push(gf);
                    data.push(bf);
                }
            }
        }

        Self {
            data,
            size,
            interpolation: InterpolationMethod::Tetrahedral,
        }
    }

    /// Applies the LUT to an RGB value.
    ///
    /// # Arguments
    ///
    /// * `rgb` - Input RGB values [0, 1]
    ///
    /// # Errors
    ///
    /// Returns an error if interpolation fails.
    pub fn apply(&self, rgb: [f64; 3]) -> Result<[f64; 3]> {
        match self.interpolation {
            InterpolationMethod::Trilinear => trilinear_interpolate(&self.data, self.size, rgb),
            InterpolationMethod::Tetrahedral => tetrahedral_interpolate(&self.data, self.size, rgb),
        }
    }

    /// Sets the interpolation method.
    pub fn set_interpolation(&mut self, method: InterpolationMethod) {
        self.interpolation = method;
    }
}

/// 1D LUT for per-channel curves.
#[derive(Clone, Debug)]
pub struct Lut1D {
    /// Red channel LUT
    pub r: Vec<f32>,
    /// Green channel LUT
    pub g: Vec<f32>,
    /// Blue channel LUT
    pub b: Vec<f32>,
}

impl Lut1D {
    /// Creates a new 1D LUT.
    ///
    /// # Errors
    ///
    /// Returns an error if the LUTs have different sizes or are empty.
    pub fn new(r: Vec<f32>, g: Vec<f32>, b: Vec<f32>) -> Result<Self> {
        if r.is_empty() || g.is_empty() || b.is_empty() {
            return Err(ColorError::Lut("1D LUT cannot be empty".to_string()));
        }

        if r.len() != g.len() || g.len() != b.len() {
            return Err(ColorError::Lut(
                "1D LUT channels must have the same size".to_string(),
            ));
        }

        Ok(Self { r, g, b })
    }

    /// Creates an identity 1D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let lut: Vec<f32> = (0..size).map(|i| i as f32 / (size - 1) as f32).collect();
        Self {
            r: lut.clone(),
            g: lut.clone(),
            b: lut,
        }
    }

    /// Applies the 1D LUT to an RGB value.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        crate::transforms::apply_1d_lut(rgb, &self.r, &self.g, &self.b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut3d_identity() {
        let lut = Lut3D::identity(17);
        let rgb = [0.5, 0.3, 0.7];
        let result = lut
            .apply(rgb)
            .expect("transform application should succeed");

        assert!((result[0] - rgb[0]).abs() < 0.01);
        assert!((result[1] - rgb[1]).abs() < 0.01);
        assert!((result[2] - rgb[2]).abs() < 0.01);
    }

    #[test]
    fn test_lut3d_size_validation() {
        let data = vec![0.0; 10];
        let result = Lut3D::new(data, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_lut3d_interpolation_methods() {
        let mut lut = Lut3D::identity(17);
        let rgb = [0.5, 0.3, 0.7];

        lut.set_interpolation(InterpolationMethod::Trilinear);
        let result1 = lut
            .apply(rgb)
            .expect("transform application should succeed");

        lut.set_interpolation(InterpolationMethod::Tetrahedral);
        let result2 = lut
            .apply(rgb)
            .expect("transform application should succeed");

        // Results should be similar but not identical
        assert!((result1[0] - result2[0]).abs() < 0.1);
        assert!((result1[1] - result2[1]).abs() < 0.1);
        assert!((result1[2] - result2[2]).abs() < 0.1);
    }

    #[test]
    fn test_lut1d_identity() {
        let lut = Lut1D::identity(256);
        let rgb = [0.5, 0.3, 0.7];
        let result = lut.apply(rgb);

        assert!((result[0] - rgb[0]).abs() < 0.01);
        assert!((result[1] - rgb[1]).abs() < 0.01);
        assert!((result[2] - rgb[2]).abs() < 0.01);
    }

    #[test]
    fn test_lut1d_validation() {
        let result = Lut1D::new(vec![], vec![1.0], vec![1.0]);
        assert!(result.is_err());

        let result = Lut1D::new(vec![1.0], vec![1.0, 2.0], vec![1.0]);
        assert!(result.is_err());
    }
}
