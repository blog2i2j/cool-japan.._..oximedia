//! 1D LUT for per-channel color correction and shaper LUTs.

use super::types::RgbColor;

/// 1D LUT for per-channel color correction or shaper LUTs.
#[derive(Clone, Debug)]
pub struct Lut1d {
    /// Red channel LUT.
    pub r_lut: Vec<f64>,
    /// Green channel LUT.
    pub g_lut: Vec<f64>,
    /// Blue channel LUT.
    pub b_lut: Vec<f64>,
}

impl Lut1d {
    /// Create a new 1D LUT with the given size.
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            r_lut: vec![0.0; size],
            g_lut: vec![0.0; size],
            b_lut: vec![0.0; size],
        }
    }

    /// Create an identity 1D LUT.
    #[must_use]
    pub fn identity(size: usize) -> Self {
        let mut lut = Self::new(size);
        for i in 0..size {
            let val = i as f64 / (size - 1) as f64;
            lut.r_lut[i] = val;
            lut.g_lut[i] = val;
            lut.b_lut[i] = val;
        }
        lut
    }

    /// Create a gamma curve LUT.
    #[must_use]
    pub fn gamma(size: usize, gamma: f64) -> Self {
        let mut lut = Self::new(size);
        for i in 0..size {
            let val = (i as f64 / (size - 1) as f64).powf(1.0 / gamma);
            lut.r_lut[i] = val;
            lut.g_lut[i] = val;
            lut.b_lut[i] = val;
        }
        lut
    }

    /// Get the size of the LUT.
    #[must_use]
    pub fn size(&self) -> usize {
        self.r_lut.len()
    }

    /// Apply the 1D LUT to a color using linear interpolation.
    #[must_use]
    pub fn apply(&self, color: RgbColor) -> RgbColor {
        let size = self.size();
        let max_idx = (size - 1) as f64;

        // Red channel
        let r_pos = (color.r.clamp(0.0, 1.0) * max_idx).clamp(0.0, max_idx);
        let r_idx = r_pos.floor() as usize;
        let r_frac = r_pos - r_idx as f64;
        let r = if r_idx + 1 < size {
            self.r_lut[r_idx] + (self.r_lut[r_idx + 1] - self.r_lut[r_idx]) * r_frac
        } else {
            self.r_lut[r_idx]
        };

        // Green channel
        let g_pos = (color.g.clamp(0.0, 1.0) * max_idx).clamp(0.0, max_idx);
        let g_idx = g_pos.floor() as usize;
        let g_frac = g_pos - g_idx as f64;
        let g = if g_idx + 1 < size {
            self.g_lut[g_idx] + (self.g_lut[g_idx + 1] - self.g_lut[g_idx]) * g_frac
        } else {
            self.g_lut[g_idx]
        };

        // Blue channel
        let b_pos = (color.b.clamp(0.0, 1.0) * max_idx).clamp(0.0, max_idx);
        let b_idx = b_pos.floor() as usize;
        let b_frac = b_pos - b_idx as f64;
        let b = if b_idx + 1 < size {
            self.b_lut[b_idx] + (self.b_lut[b_idx + 1] - self.b_lut[b_idx]) * b_frac
        } else {
            self.b_lut[b_idx]
        };

        RgbColor::new(r, g, b)
    }
}
