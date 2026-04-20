//! Saliency detection for content-aware scaling.
//!
//! This module provides various saliency detection algorithms to identify
//! important regions in images that should be preserved during scaling.

use crate::error::{CvError, CvResult};
use crate::image::{EdgeDetector, SobelEdge};
use std::f64::consts::PI;

/// Saliency detection method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SaliencyMethod {
    /// Spectral residual method.
    #[default]
    SpectralResidual,
    /// Frequency-tuned salient region detection.
    FrequencyTuned,
    /// Edge-based saliency.
    EdgeBased,
    /// Color-based saliency.
    ColorBased,
    /// Combined method using multiple approaches.
    Combined,
}

impl SaliencyMethod {
    /// Compute saliency map.
    ///
    /// # Arguments
    ///
    /// * `image` - Grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Saliency map (0-255, higher values = more salient).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn compute(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        match self {
            Self::SpectralResidual => compute_spectral_residual(image, width, height),
            Self::FrequencyTuned => compute_frequency_tuned(image, width, height),
            Self::EdgeBased => compute_edge_saliency(image, width, height),
            Self::ColorBased => compute_color_saliency(image, width, height),
            Self::Combined => compute_combined_saliency(image, width, height),
        }
    }
}

/// Saliency map.
#[derive(Debug, Clone)]
pub struct SaliencyMap {
    /// Saliency values (0-255).
    pub data: Vec<u8>,
    /// Map width.
    pub width: u32,
    /// Map height.
    pub height: u32,
}

impl SaliencyMap {
    /// Create a new saliency map.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let size = width as usize * height as usize;
        Self {
            data: vec![0u8; size],
            width,
            height,
        }
    }

    /// Create from raw data.
    pub fn from_data(data: Vec<u8>, width: u32, height: u32) -> CvResult<Self> {
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

    /// Get saliency at position.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Threshold to create binary mask.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Threshold value (0-255)
    ///
    /// # Returns
    ///
    /// Binary mask (0 or 255).
    #[must_use]
    pub fn threshold(&self, threshold: u8) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| if v >= threshold { 255 } else { 0 })
            .collect()
    }

    /// Apply Gaussian blur to smooth saliency map.
    pub fn blur(&mut self, sigma: f64) {
        let kernel_size = (sigma * 3.0).ceil() as usize * 2 + 1;
        let kernel = create_gaussian_kernel(sigma, kernel_size);

        let blurred = separable_blur(
            &self.data,
            self.width as usize,
            self.height as usize,
            &kernel,
        );
        self.data = blurred;
    }

    /// Find salient regions above threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum saliency value
    ///
    /// # Returns
    ///
    /// Vector of bounding boxes for salient regions.
    #[must_use]
    pub fn find_regions(&self, threshold: u8) -> Vec<SalientRegion> {
        let binary = self.threshold(threshold);
        find_connected_components(&binary, self.width, self.height)
    }
}

/// Salient region in an image.
#[derive(Debug, Clone)]
pub struct SalientRegion {
    /// Bounding box: (x, y, width, height).
    pub bbox: (u32, u32, u32, u32),
    /// Average saliency in region.
    pub avg_saliency: f64,
    /// Area in pixels.
    pub area: u32,
}

impl SalientRegion {
    /// Create a new salient region.
    #[must_use]
    pub const fn new(bbox: (u32, u32, u32, u32), avg_saliency: f64, area: u32) -> Self {
        Self {
            bbox,
            avg_saliency,
            area,
        }
    }

    /// Check if this region overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        let (x1, y1, w1, h1) = self.bbox;
        let (x2, y2, w2, h2) = other.bbox;

        !(x1 + w1 < x2 || x2 + w2 < x1 || y1 + h1 < y2 || y2 + h2 < y1)
    }

    /// Get center point.
    #[must_use]
    pub const fn center(&self) -> (u32, u32) {
        let (x, y, w, h) = self.bbox;
        (x + w / 2, y + h / 2)
    }
}

/// Compute spectral residual saliency.
///
/// Based on "Saliency Detection: A Spectral Residual Approach" (Hou & Zhang, 2007).
fn compute_spectral_residual(image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let size = width as usize * height as usize;
    if image.len() < size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    // Convert to float
    let mut float_image: Vec<f64> = image.iter().map(|&x| x as f64).collect();

    // Apply FFT (simplified - using DCT-like approach)
    let spectrum = compute_amplitude_spectrum(&float_image, width as usize, height as usize);

    // Compute log spectrum
    let mut log_spectrum: Vec<f64> = spectrum.iter().map(|&x| (x + 1.0).ln()).collect();

    // Apply average filter to get log spectrum average
    let avg_kernel = [1.0 / 9.0; 9]; // 3x3 average
    let log_avg = separable_blur_f64(
        &log_spectrum,
        width as usize,
        height as usize,
        &[1.0 / 3.0; 3],
    );

    // Compute spectral residual
    let mut residual = vec![0.0; size];
    for i in 0..size {
        residual[i] = log_spectrum[i] - log_avg[i];
    }

    // Inverse transform (simplified)
    let saliency_map = compute_inverse_spectrum(&residual, width as usize, height as usize);

    // Normalize to 0-255
    let mut max_val = 0.0f64;
    for &val in &saliency_map {
        max_val = max_val.max(val);
    }

    let result = if max_val > f64::EPSILON {
        saliency_map
            .iter()
            .map(|&x| ((x / max_val) * 255.0).round() as u8)
            .collect()
    } else {
        vec![0u8; size]
    };

    Ok(result)
}

/// Compute frequency-tuned salient region detection.
///
/// Based on "Frequency-tuned Salient Region Detection" (Achanta et al., 2009).
fn compute_frequency_tuned(image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let size = width as usize * height as usize;
    if image.len() < size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    // Compute mean pixel value
    let mean: f64 = image.iter().map(|&x| x as f64).sum::<f64>() / size as f64;

    // Blur image
    let kernel = create_gaussian_kernel(2.0, 5);
    let blurred = separable_blur(image, width as usize, height as usize, &kernel);

    // Compute saliency as difference from mean
    let mut saliency = vec![0.0; size];
    for i in 0..size {
        saliency[i] = (blurred[i] as f64 - mean).abs();
    }

    // Normalize
    let max_sal = saliency.iter().copied().fold(0.0f64, f64::max);
    let result = if max_sal > f64::EPSILON {
        saliency
            .iter()
            .map(|&x| ((x / max_sal) * 255.0).round() as u8)
            .collect()
    } else {
        vec![128u8; size]
    };

    Ok(result)
}

/// Compute edge-based saliency.
fn compute_edge_saliency(image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    let sobel = SobelEdge::new();
    sobel.detect(image, width, height)
}

/// Compute color-based saliency for grayscale (uses texture).
fn compute_color_saliency(image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let size = width as usize * height as usize;
    if image.len() < size {
        return Err(CvError::insufficient_data(size, image.len()));
    }

    // For grayscale, use local variance as proxy for "color" variation
    let w = width as usize;
    let h = height as usize;
    let mut saliency = vec![0.0; size];

    const WINDOW: i32 = 5;
    const HALF: i32 = WINDOW / 2;

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            let mut count = 0;

            for dy in -HALF..=HALF {
                for dx in -HALF..=HALF {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;

                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let val = image[ny as usize * w + nx as usize] as f64;
                        sum += val;
                        sum_sq += val * val;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let mean = sum / count as f64;
                let variance = (sum_sq / count as f64) - (mean * mean);
                saliency[y * w + x] = variance.sqrt();
            }
        }
    }

    // Normalize
    let max_sal = saliency.iter().copied().fold(0.0f64, f64::max);
    let result = if max_sal > f64::EPSILON {
        saliency
            .iter()
            .map(|&x| ((x / max_sal) * 255.0).round() as u8)
            .collect()
    } else {
        vec![128u8; size]
    };

    Ok(result)
}

/// Compute combined saliency using multiple methods.
fn compute_combined_saliency(image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    let edge = compute_edge_saliency(image, width, height)?;
    let freq = compute_frequency_tuned(image, width, height)?;

    // Combine with equal weights
    let size = width as usize * height as usize;
    let mut combined = vec![0u8; size];

    for i in 0..size {
        let val = (edge[i] as u16 + freq[i] as u16) / 2;
        combined[i] = val as u8;
    }

    Ok(combined)
}

/// Compute amplitude spectrum (simplified FFT approximation).
fn compute_amplitude_spectrum(image: &[f64], width: usize, height: usize) -> Vec<f64> {
    let size = width * height;
    let mut spectrum = vec![0.0; size];

    // Simple approximation using DCT-like transform
    for v in 0..height {
        for u in 0..width {
            let mut sum = 0.0;
            for y in 0..height {
                for x in 0..width {
                    let val = image[y * width + x];
                    let angle = PI * ((2 * x + 1) * u) as f64 / (2.0 * width as f64)
                        + PI * ((2 * y + 1) * v) as f64 / (2.0 * height as f64);
                    sum += val * angle.cos();
                }
            }
            spectrum[v * width + u] = sum.abs();
        }
    }

    spectrum
}

/// Compute inverse spectrum (simplified).
fn compute_inverse_spectrum(spectrum: &[f64], width: usize, height: usize) -> Vec<f64> {
    let size = width * height;
    let mut image = vec![0.0; size];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for v in 0..height {
                for u in 0..width {
                    let val = spectrum[v * width + u].exp();
                    let angle = PI * ((2 * x + 1) * u) as f64 / (2.0 * width as f64)
                        + PI * ((2 * y + 1) * v) as f64 / (2.0 * height as f64);
                    sum += val * angle.cos();
                }
            }
            image[y * width + x] = sum * sum; // Square for intensity
        }
    }

    image
}

/// Create a 1D Gaussian kernel.
fn create_gaussian_kernel(sigma: f64, size: usize) -> Vec<f64> {
    let half = size / 2;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    let two_sigma_sq = 2.0 * sigma * sigma;

    for i in 0..size {
        let x = i as f64 - half as f64;
        let value = (-x * x / two_sigma_sq).exp();
        kernel.push(value);
        sum += value;
    }

    // Normalize
    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

/// Apply separable Gaussian blur.
fn separable_blur(image: &[u8], width: usize, height: usize, kernel: &[f64]) -> Vec<u8> {
    let temp = blur_horizontal(image, width, height, kernel);
    blur_vertical(&temp, width, height, kernel)
}

/// Apply separable Gaussian blur on f64 data.
fn separable_blur_f64(image: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let temp = blur_horizontal_f64(image, width, height, kernel);
    blur_vertical_f64(&temp, width, height, kernel)
}

/// Horizontal blur pass.
fn blur_horizontal(image: &[u8], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as i32 + i as i32 - half as i32).clamp(0, width as i32 - 1) as usize;
                sum += image[y * width + xi] as f64 * k;
            }
            result[y * width + x] = sum;
        }
    }

    result
}

/// Vertical blur pass.
fn blur_vertical(image: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<u8> {
    let half = kernel.len() / 2;
    let mut result = vec![0u8; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as i32 + i as i32 - half as i32).clamp(0, height as i32 - 1) as usize;
                sum += image[yi * width + x] * k;
            }
            result[y * width + x] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }

    result
}

/// Horizontal blur pass for f64 data.
fn blur_horizontal_f64(image: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let xi = (x as i32 + i as i32 - half as i32).clamp(0, width as i32 - 1) as usize;
                sum += image[y * width + xi] * k;
            }
            result[y * width + x] = sum;
        }
    }

    result
}

/// Vertical blur pass for f64 data.
fn blur_vertical_f64(image: &[f64], width: usize, height: usize, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let mut result = vec![0.0; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            for (i, &k) in kernel.iter().enumerate() {
                let yi = (y as i32 + i as i32 - half as i32).clamp(0, height as i32 - 1) as usize;
                sum += image[yi * width + x] * k;
            }
            result[y * width + x] = sum;
        }
    }

    result
}

/// Find connected components in binary image.
fn find_connected_components(binary: &[u8], width: u32, height: u32) -> Vec<SalientRegion> {
    let w = width as usize;
    let h = height as usize;
    let mut labels = vec![0i32; w * h];
    let mut next_label = 1i32;

    // Two-pass connected components
    // First pass: assign initial labels
    for y in 0..h {
        for x in 0..w {
            if binary[y * w + x] > 0 {
                let mut neighbors = Vec::new();

                if x > 0 && labels[y * w + x - 1] > 0 {
                    neighbors.push(labels[y * w + x - 1]);
                }
                if y > 0 && labels[(y - 1) * w + x] > 0 {
                    neighbors.push(labels[(y - 1) * w + x]);
                }

                if neighbors.is_empty() {
                    labels[y * w + x] = next_label;
                    next_label += 1;
                } else {
                    labels[y * w + x] = neighbors.iter().copied().min().unwrap_or(next_label);
                }
            }
        }
    }

    // Extract regions
    let mut regions = Vec::new();
    for label in 1..next_label {
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0;
        let mut max_y = 0;
        let mut area = 0;

        for y in 0..h {
            for x in 0..w {
                if labels[y * w + x] == label {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                    area += 1;
                }
            }
        }

        if area > 0 {
            let bbox = (
                min_x as u32,
                min_y as u32,
                (max_x - min_x + 1) as u32,
                (max_y - min_y + 1) as u32,
            );
            regions.push(SalientRegion::new(bbox, 255.0, area));
        }
    }

    regions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saliency_map_new() {
        let map = SaliencyMap::new(10, 10);
        assert_eq!(map.width, 10);
        assert_eq!(map.height, 10);
        assert_eq!(map.data.len(), 100);
    }

    #[test]
    fn test_saliency_threshold() {
        let mut map = SaliencyMap::new(10, 10);
        map.data.fill(200);
        let binary = map.threshold(128);
        assert!(binary.iter().all(|&v| v == 255));
    }

    #[test]
    fn test_edge_saliency() {
        let image = vec![128u8; 100];
        let result =
            compute_edge_saliency(&image, 10, 10).expect("compute_edge_saliency should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_frequency_tuned() {
        let image = vec![128u8; 100];
        let result = compute_frequency_tuned(&image, 10, 10)
            .expect("compute_frequency_tuned should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_combined_saliency() {
        let image = vec![128u8; 100];
        let result = compute_combined_saliency(&image, 10, 10)
            .expect("compute_combined_saliency should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_salient_region() {
        let region = SalientRegion::new((10, 20, 30, 40), 0.8, 100);
        assert_eq!(region.center(), (25, 40));
        assert_eq!(region.area, 100);
    }

    #[test]
    fn test_region_overlap() {
        let r1 = SalientRegion::new((0, 0, 10, 10), 0.8, 100);
        let r2 = SalientRegion::new((5, 5, 10, 10), 0.8, 100);
        let r3 = SalientRegion::new((20, 20, 10, 10), 0.8, 100);

        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }
}
