//! Energy map computation for content-aware scaling.
//!
//! This module provides various energy functions used to determine
//! the importance of pixels in an image. Lower energy pixels are
//! considered less important and can be removed first during seam carving.

use crate::error::{CvError, CvResult};
use crate::image::{EdgeDetector, SobelEdge};

/// Energy function type for content-aware scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EnergyFunction {
    /// Gradient magnitude (Sobel).
    #[default]
    Gradient,
    /// Forward energy (considers energy after removal).
    Forward,
    /// Backward energy (considers current pixel gradient).
    Backward,
    /// Entropy-based energy.
    Entropy,
    /// Hybrid combining multiple energy functions.
    Hybrid,
}

impl EnergyFunction {
    /// Compute energy map for an image.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Energy map with same dimensions as input.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn compute(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
        match self {
            Self::Gradient => compute_gradient_energy(image, width, height),
            Self::Forward => compute_forward_energy(image, width, height),
            Self::Backward => compute_backward_energy(image, width, height),
            Self::Entropy => compute_entropy_energy(image, width, height),
            Self::Hybrid => compute_hybrid_energy(image, width, height),
        }
    }
}

/// Energy map for an image.
///
/// Stores pixel-level energy values used for seam carving.
#[derive(Debug, Clone)]
pub struct EnergyMap {
    /// Energy values (row-major order).
    pub data: Vec<f64>,
    /// Map width.
    pub width: u32,
    /// Map height.
    pub height: u32,
}

impl EnergyMap {
    /// Create a new energy map.
    ///
    /// # Arguments
    ///
    /// * `width` - Map width
    /// * `height` - Map height
    ///
    /// # Returns
    ///
    /// Zero-initialized energy map.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            data: vec![0.0; size],
            width,
            height,
        }
    }

    /// Create from raw energy data.
    ///
    /// # Arguments
    ///
    /// * `data` - Energy values
    /// * `width` - Map width
    /// * `height` - Map height
    ///
    /// # Errors
    ///
    /// Returns an error if data size doesn't match dimensions.
    pub fn from_data(data: Vec<f64>, width: u32, height: u32) -> CvResult<Self> {
        let expected = width as usize * height as usize;
        if data.len() != expected {
            return Err(CvError::insufficient_data(expected, data.len()));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Get energy at position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> f64 {
        if x >= self.width || y >= self.height {
            return f64::INFINITY;
        }
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Set energy at position.
    pub fn set(&mut self, x: u32, y: u32, energy: f64) {
        if x < self.width && y < self.height {
            self.data[y as usize * self.width as usize + x as usize] = energy;
        }
    }

    /// Add to energy at position.
    pub fn add(&mut self, x: u32, y: u32, delta: f64) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.width as usize + x as usize;
            self.data[idx] += delta;
        }
    }

    /// Get minimum energy in a row.
    #[must_use]
    pub fn min_in_row(&self, y: u32) -> f64 {
        if y >= self.height {
            return f64::INFINITY;
        }
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        self.data[start..end]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum energy in a row.
    #[must_use]
    pub fn max_in_row(&self, y: u32) -> f64 {
        if y >= self.height {
            return f64::NEG_INFINITY;
        }
        let start = y as usize * self.width as usize;
        let end = start + self.width as usize;
        self.data[start..end]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Normalize energy values to 0-1 range.
    pub fn normalize(&mut self) {
        let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() > f64::EPSILON {
            for value in &mut self.data {
                *value = (*value - min) / (max - min);
            }
        }
    }

    /// Apply protection mask (increase energy in protected regions).
    ///
    /// # Arguments
    ///
    /// * `mask` - Protection mask (0 = unprotected, 255 = protected)
    /// * `scale` - Energy scaling factor for protected regions
    pub fn apply_protection_mask(&mut self, mask: &[u8], scale: f64) {
        let size = self.width as usize * self.height as usize;
        for i in 0..size.min(mask.len()) {
            if mask[i] > 0 {
                let protection = mask[i] as f64 / 255.0;
                self.data[i] += protection * scale;
            }
        }
    }

    /// Duplicate this energy map.
    #[must_use]
    pub fn duplicate(&self) -> Self {
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
        }
    }
}

/// Compute gradient-based energy map using Sobel operators.
///
/// This is the classic seam carving energy function based on
/// image gradients.
fn compute_gradient_energy(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected = width as usize * height as usize;
    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    let sobel = SobelEdge::new();
    let (magnitude, _direction) = sobel.gradient_with_direction(image, width, height)?;

    Ok(magnitude)
}

/// Compute backward energy (classic gradient magnitude).
fn compute_backward_energy(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    compute_gradient_energy(image, width, height)
}

/// Compute forward energy that considers cost of seam removal.
///
/// Forward energy looks at the energy that would be created by
/// removing a seam, rather than just the current pixel energy.
/// This produces better visual results in many cases.
fn compute_forward_energy(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected = width as usize * height as usize;
    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    let w = width as usize;
    let h = height as usize;
    let mut energy = vec![0.0; w * h];

    // Helper function to safely get pixel value
    let get_pixel = |x: i32, y: i32| -> u8 {
        if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
            0
        } else {
            image[y as usize * w + x as usize]
        }
    };

    for y in 0..h {
        for x in 0..w {
            let xi = x as i32;
            let yi = y as i32;

            // Compute cost based on neighbors
            let left = get_pixel(xi - 1, yi);
            let right = get_pixel(xi + 1, yi);
            let up = get_pixel(xi, yi - 1);
            let down = get_pixel(xi, yi + 1);

            // Horizontal and vertical gradients
            let grad_x = (right as f64 - left as f64).abs();
            let grad_y = (down as f64 - up as f64).abs();

            energy[y * w + x] = grad_x + grad_y;
        }
    }

    Ok(energy)
}

/// Compute entropy-based energy.
///
/// Uses local entropy as a measure of information content.
/// High entropy regions contain more detail and should be preserved.
fn compute_entropy_energy(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected = width as usize * height as usize;
    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    let w = width as usize;
    let h = height as usize;
    let mut energy = vec![0.0; w * h];

    // Window size for entropy computation
    const WINDOW: i32 = 3;
    const HALF_WINDOW: i32 = WINDOW / 2;

    for y in 0..h {
        for x in 0..w {
            // Compute histogram in local window
            let mut histogram = [0u32; 256];
            let mut count = 0u32;

            for dy in -HALF_WINDOW..=HALF_WINDOW {
                for dx in -HALF_WINDOW..=HALF_WINDOW {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let pixel = image[ny as usize * w + nx as usize];
                        histogram[pixel as usize] += 1;
                        count += 1;
                    }
                }
            }

            // Compute entropy
            let mut entropy = 0.0;
            if count > 0 {
                for &freq in &histogram {
                    if freq > 0 {
                        let p = freq as f64 / count as f64;
                        entropy -= p * p.log2();
                    }
                }
            }

            energy[y * w + x] = entropy;
        }
    }

    Ok(energy)
}

/// Compute hybrid energy combining multiple energy functions.
///
/// Combines gradient and entropy energies with equal weights.
fn compute_hybrid_energy(image: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    let gradient = compute_gradient_energy(image, width, height)?;
    let entropy = compute_entropy_energy(image, width, height)?;

    // Normalize both to 0-1 range
    let normalize = |values: &mut [f64]| {
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if (max - min).abs() > f64::EPSILON {
            for v in values.iter_mut() {
                *v = (*v - min) / (max - min);
            }
        }
    };

    let mut grad_norm = gradient.clone();
    let mut entr_norm = entropy.clone();
    normalize(&mut grad_norm);
    normalize(&mut entr_norm);

    // Combine with equal weights
    let mut hybrid = vec![0.0; grad_norm.len()];
    for i in 0..hybrid.len() {
        hybrid[i] = 0.5 * grad_norm[i] + 0.5 * entr_norm[i];
    }

    Ok(hybrid)
}

/// Compute cumulative energy map for dynamic programming.
///
/// Used in seam finding to compute the minimum energy path.
///
/// # Arguments
///
/// * `energy` - Input energy map
/// * `vertical` - If true, compute for vertical seams, else horizontal
///
/// # Returns
///
/// Cumulative energy map where each pixel contains the minimum
/// total energy to reach it from the top/left edge.
pub fn compute_cumulative_energy(energy: &EnergyMap, vertical: bool) -> EnergyMap {
    let mut cumulative = EnergyMap::new(energy.width, energy.height);

    if vertical {
        compute_cumulative_vertical(energy, &mut cumulative);
    } else {
        compute_cumulative_horizontal(energy, &mut cumulative);
    }

    cumulative
}

/// Compute cumulative energy for vertical seams.
fn compute_cumulative_vertical(energy: &EnergyMap, cumulative: &mut EnergyMap) {
    let w = energy.width as usize;
    let h = energy.height as usize;

    // First row is same as energy
    for x in 0..w {
        cumulative.data[x] = energy.data[x];
    }

    // Dynamic programming: each pixel's cumulative energy is its energy
    // plus the minimum of the three pixels above it
    for y in 1..h {
        for x in 0..w {
            let e = energy.data[y * w + x];

            let mut min_prev = cumulative.data[(y - 1) * w + x];

            if x > 0 {
                min_prev = min_prev.min(cumulative.data[(y - 1) * w + x - 1]);
            }

            if x < w - 1 {
                min_prev = min_prev.min(cumulative.data[(y - 1) * w + x + 1]);
            }

            cumulative.data[y * w + x] = e + min_prev;
        }
    }
}

/// Compute cumulative energy for horizontal seams.
fn compute_cumulative_horizontal(energy: &EnergyMap, cumulative: &mut EnergyMap) {
    let w = energy.width as usize;
    let h = energy.height as usize;

    // First column is same as energy
    for y in 0..h {
        cumulative.data[y * w] = energy.data[y * w];
    }

    // Dynamic programming: each pixel's cumulative energy is its energy
    // plus the minimum of the three pixels to its left
    for x in 1..w {
        for y in 0..h {
            let e = energy.data[y * w + x];

            let mut min_prev = cumulative.data[y * w + x - 1];

            if y > 0 {
                min_prev = min_prev.min(cumulative.data[(y - 1) * w + x - 1]);
            }

            if y < h - 1 {
                min_prev = min_prev.min(cumulative.data[(y + 1) * w + x - 1]);
            }

            cumulative.data[y * w + x] = e + min_prev;
        }
    }
}

/// Compute energy for a multi-channel image (RGB).
///
/// Converts to grayscale first, then computes energy.
pub fn compute_rgb_energy(
    image: &[u8],
    width: u32,
    height: u32,
    energy_fn: EnergyFunction,
) -> CvResult<Vec<f64>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected = width as usize * height as usize * 3;
    if image.len() < expected {
        return Err(CvError::insufficient_data(expected, image.len()));
    }

    // Convert to grayscale
    let size = width as usize * height as usize;
    let mut grayscale = vec![0u8; size];

    for i in 0..size {
        let r = image[i * 3] as f64;
        let g = image[i * 3 + 1] as f64;
        let b = image[i * 3 + 2] as f64;
        // Rec. 601 luma
        grayscale[i] = (0.299 * r + 0.587 * g + 0.114 * b).round() as u8;
    }

    energy_fn.compute(&grayscale, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_map_new() {
        let map = EnergyMap::new(10, 10);
        assert_eq!(map.width, 10);
        assert_eq!(map.height, 10);
        assert_eq!(map.data.len(), 100);
    }

    #[test]
    fn test_energy_map_get_set() {
        let mut map = EnergyMap::new(10, 10);
        map.set(5, 5, 42.0);
        assert_eq!(map.get(5, 5), 42.0);
    }

    #[test]
    fn test_energy_map_bounds() {
        let map = EnergyMap::new(10, 10);
        // Out of bounds should return infinity
        assert_eq!(map.get(100, 100), f64::INFINITY);
    }

    #[test]
    fn test_gradient_energy() {
        let image = vec![128u8; 100];
        let energy = compute_gradient_energy(&image, 10, 10)
            .expect("compute_gradient_energy should succeed");
        assert_eq!(energy.len(), 100);
        // Uniform image should have low energy
        for &e in &energy {
            assert!(e < 10.0);
        }
    }

    #[test]
    fn test_forward_energy() {
        let image = vec![128u8; 100];
        let energy =
            compute_forward_energy(&image, 10, 10).expect("compute_forward_energy should succeed");
        assert_eq!(energy.len(), 100);
    }

    #[test]
    fn test_entropy_energy() {
        let image = vec![128u8; 100];
        let energy =
            compute_entropy_energy(&image, 10, 10).expect("compute_entropy_energy should succeed");
        assert_eq!(energy.len(), 100);
    }

    #[test]
    fn test_hybrid_energy() {
        let image = vec![128u8; 100];
        let energy =
            compute_hybrid_energy(&image, 10, 10).expect("compute_hybrid_energy should succeed");
        assert_eq!(energy.len(), 100);
    }

    #[test]
    fn test_cumulative_energy_vertical() {
        let mut energy = EnergyMap::new(10, 10);
        for i in 0..100 {
            energy.data[i] = 1.0;
        }

        let cumulative = compute_cumulative_energy(&energy, true);
        // Bottom row should have cumulative energy equal to height
        assert!((cumulative.get(5, 9) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_energy_map_normalize() {
        let mut map = EnergyMap::new(10, 10);
        for i in 0..100 {
            map.data[i] = i as f64;
        }
        map.normalize();

        // Check that values are in 0-1 range
        let min = map.data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = map.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((min - 0.0).abs() < 1e-6);
        assert!((max - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_energy() {
        let image = vec![128u8; 300]; // 10x10 RGB
        let energy = compute_rgb_energy(&image, 10, 10, EnergyFunction::Gradient)
            .expect("compute_rgb_energy should succeed");
        assert_eq!(energy.len(), 100);
    }
}
