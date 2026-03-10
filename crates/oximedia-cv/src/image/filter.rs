//! Image filtering operations.
//!
//! This module provides various image filters including:
//! - Gaussian blur (separable)
//! - Box blur
//! - Median filter
//! - Bilateral filter (edge-preserving)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::{GaussianBlur, ImageFilter};
//!
//! let blur = GaussianBlur::new(1.0, 5);
//! assert_eq!(blur.kernel_size(), 5);
//! ```

use crate::error::{CvError, CvResult};

/// Trait for image filters.
pub trait ImageFilter {
    /// Apply the filter to a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `src` - Source grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Filtered image data.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>>;

    /// Get the filter kernel size.
    fn kernel_size(&self) -> usize;
}

/// Gaussian blur filter with separable implementation.
#[derive(Debug, Clone)]
pub struct GaussianBlur {
    /// Standard deviation of the Gaussian.
    sigma: f64,
    /// Kernel size (must be odd).
    size: usize,
    /// Pre-computed 1D kernel.
    kernel: Vec<f64>,
}

impl GaussianBlur {
    /// Create a new Gaussian blur filter.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Standard deviation of the Gaussian
    /// * `size` - Kernel size (must be odd, will be adjusted if even)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::{GaussianBlur, ImageFilter};
    ///
    /// let blur = GaussianBlur::new(1.5, 7);
    /// assert_eq!(blur.sigma(), 1.5);
    /// assert_eq!(blur.kernel_size(), 7);
    /// ```
    #[must_use]
    pub fn new(sigma: f64, size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        let size = size.max(3);
        let kernel = create_gaussian_kernel_1d(sigma, size);

        Self {
            sigma,
            size,
            kernel,
        }
    }

    /// Get the sigma value.
    #[must_use]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Get the 1D kernel.
    #[must_use]
    pub fn kernel(&self) -> &[f64] {
        &self.kernel
    }
}

impl ImageFilter for GaussianBlur {
    fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        // Separable convolution: horizontal then vertical
        let temp = convolve_horizontal(src, width, height, &self.kernel);
        let result = convolve_vertical(&temp, width, height, &self.kernel);

        Ok(result)
    }

    fn kernel_size(&self) -> usize {
        self.size
    }
}

/// Box blur filter (uniform average).
#[derive(Debug, Clone, Copy)]
pub struct BoxBlur {
    /// Kernel size (must be odd).
    size: usize,
}

impl BoxBlur {
    /// Create a new box blur filter.
    ///
    /// # Arguments
    ///
    /// * `size` - Kernel size (must be odd, will be adjusted if even)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::{BoxBlur, ImageFilter};
    ///
    /// let blur = BoxBlur::new(5);
    /// assert_eq!(blur.kernel_size(), 5);
    /// ```
    #[must_use]
    pub fn new(size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        Self { size: size.max(3) }
    }
}

impl ImageFilter for BoxBlur {
    fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        // Use integral image for O(1) box filter per pixel
        let integral = compute_integral_image(src, width, height);
        let result = box_filter_integral(&integral, width, height, self.size);

        Ok(result)
    }

    fn kernel_size(&self) -> usize {
        self.size
    }
}

/// Median filter for noise removal.
#[derive(Debug, Clone, Copy)]
pub struct MedianFilter {
    /// Kernel size (must be odd).
    size: usize,
}

impl MedianFilter {
    /// Create a new median filter.
    ///
    /// # Arguments
    ///
    /// * `size` - Kernel size (must be odd, will be adjusted if even)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::{MedianFilter, ImageFilter};
    ///
    /// let filter = MedianFilter::new(3);
    /// assert_eq!(filter.kernel_size(), 3);
    /// ```
    #[must_use]
    pub fn new(size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        Self { size: size.max(3) }
    }
}

impl ImageFilter for MedianFilter {
    fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let half = self.size / 2;
        let expected_size = width as usize * height as usize;
        let mut dst = vec![0u8; expected_size];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let mut values: Vec<u8> = Vec::with_capacity(self.size * self.size);

                for ky in 0..self.size {
                    let sy =
                        (y as i32 + ky as i32 - half as i32).clamp(0, height as i32 - 1) as usize;

                    for kx in 0..self.size {
                        let sx = (x as i32 + kx as i32 - half as i32).clamp(0, width as i32 - 1)
                            as usize;

                        values.push(src[sy * width as usize + sx]);
                    }
                }

                // Find median
                values.sort_unstable();
                dst[y * width as usize + x] = values[values.len() / 2];
            }
        }

        Ok(dst)
    }

    fn kernel_size(&self) -> usize {
        self.size
    }
}

/// Bilateral filter for edge-preserving smoothing.
#[derive(Debug, Clone)]
pub struct BilateralFilter {
    /// Spatial sigma (affects spatial weighting).
    sigma_space: f64,
    /// Range sigma (affects intensity weighting).
    sigma_color: f64,
    /// Kernel size.
    size: usize,
    /// Pre-computed spatial weights.
    spatial_weights: Vec<f64>,
}

impl BilateralFilter {
    /// Create a new bilateral filter.
    ///
    /// # Arguments
    ///
    /// * `sigma_space` - Standard deviation for spatial weighting
    /// * `sigma_color` - Standard deviation for intensity weighting
    /// * `size` - Kernel size (must be odd, will be adjusted if even)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::BilateralFilter;
    ///
    /// let filter = BilateralFilter::new(10.0, 30.0, 9);
    /// assert_eq!(filter.sigma_space(), 10.0);
    /// ```
    #[must_use]
    pub fn new(sigma_space: f64, sigma_color: f64, size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        let size = size.max(3);

        let half = size / 2;
        let mut spatial_weights = Vec::with_capacity(size * size);

        let two_sigma_sq = 2.0 * sigma_space * sigma_space;

        for y in 0..size {
            for x in 0..size {
                let dx = x as f64 - half as f64;
                let dy = y as f64 - half as f64;
                let dist_sq = dx * dx + dy * dy;
                spatial_weights.push((-dist_sq / two_sigma_sq).exp());
            }
        }

        Self {
            sigma_space,
            sigma_color,
            size,
            spatial_weights,
        }
    }

    /// Get the spatial sigma value.
    #[must_use]
    pub const fn sigma_space(&self) -> f64 {
        self.sigma_space
    }

    /// Get the color sigma value.
    #[must_use]
    pub const fn sigma_color(&self) -> f64 {
        self.sigma_color
    }
}

impl ImageFilter for BilateralFilter {
    fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let half = self.size / 2;
        let expected_size = width as usize * height as usize;
        let mut dst = vec![0u8; expected_size];

        let two_sigma_color_sq = 2.0 * self.sigma_color * self.sigma_color;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let center_val = src[y * width as usize + x] as f64;
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for ky in 0..self.size {
                    let sy =
                        (y as i32 + ky as i32 - half as i32).clamp(0, height as i32 - 1) as usize;

                    for kx in 0..self.size {
                        let sx = (x as i32 + kx as i32 - half as i32).clamp(0, width as i32 - 1)
                            as usize;

                        let neighbor_val = src[sy * width as usize + sx] as f64;
                        let color_diff = neighbor_val - center_val;

                        // Range weight based on intensity difference
                        let range_weight = (-color_diff * color_diff / two_sigma_color_sq).exp();

                        // Combined weight
                        let spatial_idx = ky * self.size + kx;
                        let weight = self.spatial_weights[spatial_idx] * range_weight;

                        weighted_sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                if weight_sum.abs() > f64::EPSILON {
                    dst[y * width as usize + x] =
                        (weighted_sum / weight_sum).round().clamp(0.0, 255.0) as u8;
                } else {
                    dst[y * width as usize + x] = center_val as u8;
                }
            }
        }

        Ok(dst)
    }

    fn kernel_size(&self) -> usize {
        self.size
    }
}

/// Generic 2D convolution kernel.
#[derive(Debug, Clone)]
pub struct ConvolutionKernel {
    /// Kernel coefficients (row-major).
    data: Vec<f64>,
    /// Kernel width.
    width: usize,
    /// Kernel height.
    height: usize,
}

impl ConvolutionKernel {
    /// Create a new convolution kernel.
    ///
    /// # Arguments
    ///
    /// * `data` - Kernel coefficients (row-major)
    /// * `width` - Kernel width
    /// * `height` - Kernel height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::filter::ConvolutionKernel;
    ///
    /// let sharpen = ConvolutionKernel::new(
    ///     vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
    ///     3, 3
    /// );
    /// assert_eq!(sharpen.width(), 3);
    /// ```
    #[must_use]
    pub fn new(data: Vec<f64>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
        }
    }

    /// Get kernel width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Get kernel height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Get kernel coefficients.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Create a sharpening kernel.
    #[must_use]
    pub fn sharpen() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], 3, 3)
    }

    /// Create an emboss kernel.
    #[must_use]
    pub fn emboss() -> Self {
        Self::new(vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 3)
    }

    /// Apply the kernel to an image.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let half_w = self.width / 2;
        let half_h = self.height / 2;
        let expected_size = width as usize * height as usize;
        let mut dst = vec![0u8; expected_size];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let mut sum = 0.0;

                for ky in 0..self.height {
                    let sy =
                        (y as i32 + ky as i32 - half_h as i32).clamp(0, height as i32 - 1) as usize;

                    for kx in 0..self.width {
                        let sx = (x as i32 + kx as i32 - half_w as i32).clamp(0, width as i32 - 1)
                            as usize;

                        let kernel_val = self.data[ky * self.width + kx];
                        sum += src[sy * width as usize + sx] as f64 * kernel_val;
                    }
                }

                dst[y * width as usize + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(dst)
    }
}

/// Create a 1D Gaussian kernel.
fn create_gaussian_kernel_1d(sigma: f64, size: usize) -> Vec<f64> {
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

/// Validate input image dimensions.
fn validate_input(data: &[u8], width: u32, height: u32) -> CvResult<()> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = width as usize * height as usize;
    if data.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, data.len()));
    }

    Ok(())
}

/// Horizontal convolution pass.
fn convolve_horizontal(src: &[u8], width: u32, height: u32, kernel: &[f64]) -> Vec<f64> {
    let half = kernel.len() / 2;
    let expected_size = width as usize * height as usize;
    let mut dst = vec![0.0; expected_size];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut sum = 0.0;

            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - half as i32).clamp(0, width as i32 - 1) as usize;
                sum += src[y * width as usize + sx] as f64 * kv;
            }

            dst[y * width as usize + x] = sum;
        }
    }

    dst
}

/// Vertical convolution pass.
fn convolve_vertical(src: &[f64], width: u32, height: u32, kernel: &[f64]) -> Vec<u8> {
    let half = kernel.len() / 2;
    let expected_size = width as usize * height as usize;
    let mut dst = vec![0u8; expected_size];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut sum = 0.0;

            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - half as i32).clamp(0, height as i32 - 1) as usize;
                sum += src[sy * width as usize + x] * kv;
            }

            dst[y * width as usize + x] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }

    dst
}

/// Compute integral image for fast box filtering.
fn compute_integral_image(src: &[u8], width: u32, height: u32) -> Vec<u64> {
    let w = width as usize;
    let h = height as usize;
    let mut integral = vec![0u64; (w + 1) * (h + 1)];

    for y in 0..h {
        for x in 0..w {
            let idx = (y + 1) * (w + 1) + (x + 1);
            integral[idx] = src[y * w + x] as u64
                + integral[y * (w + 1) + (x + 1)]
                + integral[(y + 1) * (w + 1) + x]
                - integral[y * (w + 1) + x];
        }
    }

    integral
}

/// Apply box filter using integral image.
fn box_filter_integral(integral: &[u64], width: u32, height: u32, size: usize) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let half = size / 2;
    let mut dst = vec![0u8; w * h];

    let iw = w + 1;

    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(half);
            let y0 = y.saturating_sub(half);
            let x1 = (x + half + 1).min(w);
            let y1 = (y + half + 1).min(h);

            let sum = integral[y1 * iw + x1] + integral[y0 * iw + x0]
                - integral[y0 * iw + x1]
                - integral[y1 * iw + x0];

            let count = (x1 - x0) * (y1 - y0);
            dst[y * w + x] = (sum / count as u64) as u8;
        }
    }

    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_creation() {
        let blur = GaussianBlur::new(1.0, 5);
        assert_eq!(blur.kernel_size(), 5);
        assert!((blur.sigma() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gaussian_blur_even_size() {
        let blur = GaussianBlur::new(1.0, 4);
        assert_eq!(blur.kernel_size(), 5); // Should be adjusted to odd
    }

    #[test]
    fn test_gaussian_blur_apply() {
        let src = vec![100u8; 25];
        let blur = GaussianBlur::new(1.0, 3);
        let result = blur.apply(&src, 5, 5).expect("apply should succeed");
        assert_eq!(result.len(), 25);

        // Uniform image should remain close to original
        for &v in &result {
            assert!((v as i32 - 100).abs() < 5);
        }
    }

    #[test]
    fn test_box_blur() {
        let src = vec![100u8; 25];
        let blur = BoxBlur::new(3);
        let result = blur.apply(&src, 5, 5).expect("apply should succeed");
        assert_eq!(result.len(), 25);

        // Uniform image should remain exactly the same
        for &v in &result {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn test_median_filter() {
        // Image with salt and pepper noise
        let mut src = vec![100u8; 25];
        src[12] = 255; // Salt noise

        let filter = MedianFilter::new(3);
        let result = filter.apply(&src, 5, 5).expect("apply should succeed");

        // Median filter should remove the noise
        assert!(result[12] < 200);
    }

    #[test]
    fn test_bilateral_filter() {
        let src = vec![100u8; 25];
        let filter = BilateralFilter::new(10.0, 30.0, 5);
        let result = filter.apply(&src, 5, 5).expect("apply should succeed");
        assert_eq!(result.len(), 25);

        // Uniform image should remain close to original
        for &v in &result {
            assert!((v as i32 - 100).abs() < 5);
        }
    }

    #[test]
    fn test_convolution_kernel_sharpen() {
        let kernel = ConvolutionKernel::sharpen();
        assert_eq!(kernel.width(), 3);
        assert_eq!(kernel.height(), 3);
    }

    #[test]
    fn test_convolution_kernel_apply() {
        let src = vec![100u8; 25];
        let kernel = ConvolutionKernel::sharpen();
        let result = kernel.apply(&src, 5, 5).expect("apply should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_integral_image() {
        let src = vec![1u8; 9];
        let integral = compute_integral_image(&src, 3, 3);

        // Sum at (3,3) should be 9 (all pixels)
        assert_eq!(integral[4 * 4 - 1], 9);
    }

    #[test]
    fn test_invalid_dimensions() {
        let src = vec![0u8; 25];
        let blur = GaussianBlur::new(1.0, 3);
        assert!(blur.apply(&src, 0, 5).is_err());
        assert!(blur.apply(&src, 5, 0).is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let src = vec![0u8; 10];
        let blur = GaussianBlur::new(1.0, 3);
        assert!(blur.apply(&src, 5, 5).is_err()); // Needs 25 bytes
    }
}
