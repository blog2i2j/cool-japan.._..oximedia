//! Interpolation methods for frame warping.

use crate::warp::boundary::BoundaryMode;
use scirs2_core::ndarray::Array2;

/// Interpolation method for pixel sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Nearest neighbor
    Nearest,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

impl InterpolationMethod {
    /// Interpolate pixel value at non-integer coordinates.
    #[must_use]
    pub fn interpolate(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        match self {
            Self::Nearest => self.nearest(data, x, y, boundary),
            Self::Bilinear => self.bilinear(data, x, y, boundary),
            Self::Bicubic => self.bicubic(data, x, y, boundary),
        }
    }

    /// Nearest neighbor interpolation.
    fn nearest(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        boundary.get_pixel(data, x.round(), y.round())
    }

    /// Bilinear interpolation.
    fn bilinear(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        let x0 = x.floor();
        let y0 = y.floor();
        let dx = x - x0;
        let dy = y - y0;

        let p00 = f64::from(boundary.get_pixel(data, x0, y0));
        let p10 = f64::from(boundary.get_pixel(data, x0 + 1.0, y0));
        let p01 = f64::from(boundary.get_pixel(data, x0, y0 + 1.0));
        let p11 = f64::from(boundary.get_pixel(data, x0 + 1.0, y0 + 1.0));

        let val = (1.0 - dx) * (1.0 - dy) * p00
            + dx * (1.0 - dy) * p10
            + (1.0 - dx) * dy * p01
            + dx * dy * p11;

        val.clamp(0.0, 255.0) as u8
    }

    /// Bicubic interpolation.
    fn bicubic(self, data: &Array2<u8>, x: f64, y: f64, boundary: BoundaryMode) -> u8 {
        let x0 = x.floor();
        let y0 = y.floor();
        let dx = x - x0;
        let dy = y - y0;

        let mut val = 0.0;

        for j in -1..=2 {
            for i in -1..=2 {
                let p = f64::from(boundary.get_pixel(data, x0 + f64::from(i), y0 + f64::from(j)));
                let wx = Self::cubic_weight(dx - f64::from(i));
                let wy = Self::cubic_weight(dy - f64::from(j));
                val += p * wx * wy;
            }
        }

        val.clamp(0.0, 255.0) as u8
    }

    /// Cubic interpolation weight function.
    fn cubic_weight(t: f64) -> f64 {
        let t = t.abs();
        if t < 1.0 {
            1.5 * t.powi(3) - 2.5 * t.powi(2) + 1.0
        } else if t < 2.0 {
            -0.5 * t.powi(3) + 2.5 * t.powi(2) - 4.0 * t + 2.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest() {
        let data = Array2::from_elem((10, 10), 128);
        let val = InterpolationMethod::Nearest.interpolate(&data, 5.5, 5.5, BoundaryMode::Constant);
        assert_eq!(val, 128);
    }

    #[test]
    fn test_bilinear() {
        let data = Array2::from_elem((10, 10), 128);
        let val =
            InterpolationMethod::Bilinear.interpolate(&data, 5.5, 5.5, BoundaryMode::Constant);
        assert_eq!(val, 128);
    }
}
