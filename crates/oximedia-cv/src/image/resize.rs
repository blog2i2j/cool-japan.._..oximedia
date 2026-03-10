//! Image resizing operations.
//!
//! This module provides image resizing functionality with multiple
//! interpolation methods including nearest-neighbor, bilinear, bicubic,
//! Lanczos, and area averaging.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::ResizeMethod;
//!
//! let method = ResizeMethod::Bilinear;
//! assert!(method.is_interpolating());
//! ```

use crate::error::{CvError, CvResult};

/// Interpolation method for image resizing.
///
/// Different methods trade off quality for speed:
/// - `Nearest`: Fastest, but produces blocky results
/// - `Bilinear`: Good balance of speed and quality
/// - `Bicubic`: Higher quality, slower than bilinear
/// - `Lanczos`: Highest quality, slowest
/// - `Area`: Best for downscaling, averages pixel areas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeMethod {
    /// Nearest neighbor interpolation.
    /// Fastest method, but produces blocky/aliased results.
    Nearest,

    /// Bilinear interpolation.
    /// Linear interpolation in both dimensions.
    #[default]
    Bilinear,

    /// Bicubic interpolation.
    /// Uses cubic polynomials, produces smoother results than bilinear.
    Bicubic,

    /// Lanczos interpolation (a=3).
    /// High-quality resampling using sinc function approximation.
    Lanczos,

    /// Area-based averaging.
    /// Best for downscaling, considers all source pixels that contribute
    /// to each destination pixel.
    Area,
}

impl ResizeMethod {
    /// Returns true if this method uses interpolation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::ResizeMethod;
    ///
    /// assert!(!ResizeMethod::Nearest.is_interpolating());
    /// assert!(ResizeMethod::Bilinear.is_interpolating());
    /// ```
    #[must_use]
    pub const fn is_interpolating(&self) -> bool {
        !matches!(self, Self::Nearest)
    }

    /// Returns the kernel size for this interpolation method.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::ResizeMethod;
    ///
    /// assert_eq!(ResizeMethod::Nearest.kernel_size(), 1);
    /// assert_eq!(ResizeMethod::Bilinear.kernel_size(), 2);
    /// assert_eq!(ResizeMethod::Bicubic.kernel_size(), 4);
    /// ```
    #[must_use]
    pub const fn kernel_size(&self) -> usize {
        match self {
            Self::Nearest => 1,
            Self::Bilinear => 2,
            Self::Bicubic | Self::Area => 4,
            Self::Lanczos => 6,
        }
    }
}

/// Resize configuration.
#[derive(Debug, Clone)]
pub struct ResizeConfig {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Interpolation method.
    pub method: ResizeMethod,
    /// Number of channels per pixel.
    pub channels: usize,
}

impl ResizeConfig {
    /// Create a new resize configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::resize::{ResizeConfig, ResizeMethod};
    ///
    /// let config = ResizeConfig::new(640, 480, ResizeMethod::Bilinear, 3);
    /// assert_eq!(config.width, 640);
    /// assert_eq!(config.height, 480);
    /// ```
    #[must_use]
    pub const fn new(width: u32, height: u32, method: ResizeMethod, channels: usize) -> Self {
        Self {
            width,
            height,
            method,
            channels,
        }
    }
}

/// Resizes an image using the specified method.
///
/// # Arguments
///
/// * `src` - Source image data (row-major, interleaved channels)
/// * `src_width` - Source image width
/// * `src_height` - Source image height
/// * `config` - Resize configuration
///
/// # Returns
///
/// Resized image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::resize::{resize_image, ResizeConfig, ResizeMethod};
///
/// // Create a simple 2x2 grayscale image
/// let src = vec![0u8, 64, 128, 255];
/// let config = ResizeConfig::new(4, 4, ResizeMethod::Bilinear, 1);
/// let result = resize_image(&src, 2, 2, &config)?;
/// assert_eq!(result.len(), 16);
/// ```
pub fn resize_image(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    config: &ResizeConfig,
) -> CvResult<Vec<u8>> {
    // Validate input
    if src_width == 0 || src_height == 0 {
        return Err(CvError::invalid_dimensions(src_width, src_height));
    }
    if config.width == 0 || config.height == 0 {
        return Err(CvError::invalid_dimensions(config.width, config.height));
    }

    let expected_size = src_width as usize * src_height as usize * config.channels;
    if src.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, src.len()));
    }

    let dst_size = config.width as usize * config.height as usize * config.channels;
    let mut dst = vec![0u8; dst_size];

    match config.method {
        ResizeMethod::Nearest => {
            resize_nearest(src, src_width, src_height, &mut dst, config);
        }
        ResizeMethod::Bilinear => {
            resize_bilinear(src, src_width, src_height, &mut dst, config);
        }
        ResizeMethod::Bicubic => {
            resize_bicubic(src, src_width, src_height, &mut dst, config);
        }
        ResizeMethod::Lanczos => {
            resize_lanczos(src, src_width, src_height, &mut dst, config);
        }
        ResizeMethod::Area => {
            resize_area(src, src_width, src_height, &mut dst, config);
        }
    }

    Ok(dst)
}

/// Nearest neighbor interpolation.
fn resize_nearest(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    config: &ResizeConfig,
) {
    let x_ratio = src_width as f64 / config.width as f64;
    let y_ratio = src_height as f64 / config.height as f64;
    let channels = config.channels;

    for y in 0..config.height {
        let src_y = ((y as f64 + 0.5) * y_ratio - 0.5).round() as u32;
        let src_y = src_y.min(src_height - 1);

        for x in 0..config.width {
            let src_x = ((x as f64 + 0.5) * x_ratio - 0.5).round() as u32;
            let src_x = src_x.min(src_width - 1);

            let src_idx = (src_y as usize * src_width as usize + src_x as usize) * channels;
            let dst_idx = (y as usize * config.width as usize + x as usize) * channels;

            for c in 0..channels {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

/// Bilinear interpolation.
fn resize_bilinear(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    config: &ResizeConfig,
) {
    let x_ratio = (src_width as f64 - 1.0) / (config.width as f64 - 1.0).max(1.0);
    let y_ratio = (src_height as f64 - 1.0) / (config.height as f64 - 1.0).max(1.0);
    let channels = config.channels;
    let src_stride = src_width as usize * channels;

    for y in 0..config.height {
        let src_y = y as f64 * y_ratio;
        let y0 = src_y.floor() as u32;
        let y1 = (y0 + 1).min(src_height - 1);
        let y_frac = src_y - y0 as f64;

        for x in 0..config.width {
            let src_x = x as f64 * x_ratio;
            let x0 = src_x.floor() as u32;
            let x1 = (x0 + 1).min(src_width - 1);
            let x_frac = src_x - x0 as f64;

            let dst_idx = (y as usize * config.width as usize + x as usize) * channels;

            for c in 0..channels {
                let p00 = src[y0 as usize * src_stride + x0 as usize * channels + c] as f64;
                let p10 = src[y0 as usize * src_stride + x1 as usize * channels + c] as f64;
                let p01 = src[y1 as usize * src_stride + x0 as usize * channels + c] as f64;
                let p11 = src[y1 as usize * src_stride + x1 as usize * channels + c] as f64;

                // Bilinear interpolation
                let top = p00 * (1.0 - x_frac) + p10 * x_frac;
                let bottom = p01 * (1.0 - x_frac) + p11 * x_frac;
                let value = top * (1.0 - y_frac) + bottom * y_frac;

                dst[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Bicubic interpolation using cubic convolution.
fn resize_bicubic(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    config: &ResizeConfig,
) {
    let x_ratio = src_width as f64 / config.width as f64;
    let y_ratio = src_height as f64 / config.height as f64;
    let channels = config.channels;
    let src_stride = src_width as usize * channels;

    for y in 0..config.height {
        let src_y = (y as f64 + 0.5) * y_ratio - 0.5;
        let y_int = src_y.floor() as i32;
        let y_frac = src_y - y_int as f64;

        for x in 0..config.width {
            let src_x = (x as f64 + 0.5) * x_ratio - 0.5;
            let x_int = src_x.floor() as i32;
            let x_frac = src_x - x_int as f64;

            let dst_idx = (y as usize * config.width as usize + x as usize) * channels;

            for c in 0..channels {
                let mut value = 0.0;

                for ky in -1..=2 {
                    let py = (y_int + ky).clamp(0, src_height as i32 - 1) as usize;
                    let wy = cubic_weight(ky as f64 - y_frac);

                    for kx in -1..=2 {
                        let px = (x_int + kx).clamp(0, src_width as i32 - 1) as usize;
                        let wx = cubic_weight(kx as f64 - x_frac);

                        let pixel = src[py * src_stride + px * channels + c] as f64;
                        value += pixel * wx * wy;
                    }
                }

                dst[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Cubic interpolation weight function (Catmull-Rom spline).
#[inline]
fn cubic_weight(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        (1.5 * x - 2.5) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Lanczos interpolation (a=3).
fn resize_lanczos(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    dst: &mut [u8],
    config: &ResizeConfig,
) {
    let x_ratio = src_width as f64 / config.width as f64;
    let y_ratio = src_height as f64 / config.height as f64;
    let channels = config.channels;
    let src_stride = src_width as usize * channels;
    let a = 3; // Lanczos parameter

    for y in 0..config.height {
        let src_y = (y as f64 + 0.5) * y_ratio - 0.5;
        let y_int = src_y.floor() as i32;
        let y_frac = src_y - y_int as f64;

        for x in 0..config.width {
            let src_x = (x as f64 + 0.5) * x_ratio - 0.5;
            let x_int = src_x.floor() as i32;
            let x_frac = src_x - x_int as f64;

            let dst_idx = (y as usize * config.width as usize + x as usize) * channels;

            for c in 0..channels {
                let mut value = 0.0;
                let mut weight_sum = 0.0;

                for ky in (1 - a)..=a {
                    let py = (y_int + ky).clamp(0, src_height as i32 - 1) as usize;
                    let wy = lanczos_weight(ky as f64 - y_frac, a);

                    for kx in (1 - a)..=a {
                        let px = (x_int + kx).clamp(0, src_width as i32 - 1) as usize;
                        let wx = lanczos_weight(kx as f64 - x_frac, a);

                        let w = wx * wy;
                        let pixel = src[py * src_stride + px * channels + c] as f64;
                        value += pixel * w;
                        weight_sum += w;
                    }
                }

                if weight_sum.abs() > f64::EPSILON {
                    value /= weight_sum;
                }

                dst[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Lanczos kernel weight function.
#[inline]
fn lanczos_weight(x: f64, a: i32) -> f64 {
    let x = x.abs();
    if x < f64::EPSILON {
        1.0
    } else if x < a as f64 {
        let pi_x = std::f64::consts::PI * x;
        let pi_x_a = pi_x / a as f64;
        (pi_x.sin() / pi_x) * (pi_x_a.sin() / pi_x_a)
    } else {
        0.0
    }
}

/// Area-based averaging for downscaling.
fn resize_area(src: &[u8], src_width: u32, src_height: u32, dst: &mut [u8], config: &ResizeConfig) {
    let x_ratio = src_width as f64 / config.width as f64;
    let y_ratio = src_height as f64 / config.height as f64;
    let channels = config.channels;
    let src_stride = src_width as usize * channels;

    // If upscaling, fall back to bilinear
    if x_ratio <= 1.0 && y_ratio <= 1.0 {
        resize_bilinear(src, src_width, src_height, dst, config);
        return;
    }

    for y in 0..config.height {
        let y_start = y as f64 * y_ratio;
        let y_end = ((y + 1) as f64 * y_ratio).min(src_height as f64);

        for x in 0..config.width {
            let x_start = x as f64 * x_ratio;
            let x_end = ((x + 1) as f64 * x_ratio).min(src_width as f64);

            let dst_idx = (y as usize * config.width as usize + x as usize) * channels;

            for c in 0..channels {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                let y0 = y_start.floor() as u32;
                let y1 = y_end.ceil() as u32;
                let x0 = x_start.floor() as u32;
                let x1 = x_end.ceil() as u32;

                for sy in y0..y1.min(src_height) {
                    let wy = calculate_overlap(sy as f64, (sy + 1) as f64, y_start, y_end);

                    for sx in x0..x1.min(src_width) {
                        let wx = calculate_overlap(sx as f64, (sx + 1) as f64, x_start, x_end);

                        let w = wx * wy;
                        let pixel =
                            src[sy as usize * src_stride + sx as usize * channels + c] as f64;
                        sum += pixel * w;
                        weight_sum += w;
                    }
                }

                if weight_sum.abs() > f64::EPSILON {
                    sum /= weight_sum;
                }

                dst[dst_idx + c] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Calculate overlap between two intervals.
#[inline]
fn calculate_overlap(a_start: f64, a_end: f64, b_start: f64, b_end: f64) -> f64 {
    let start = a_start.max(b_start);
    let end = a_end.min(b_end);
    (end - start).max(0.0)
}

/// Separable convolution for SIMD-ready implementation.
///
/// Performs 1D convolution along rows and columns separately,
/// which is more efficient for separable kernels (like Gaussian).
pub struct SeparableConvolution {
    /// Horizontal kernel coefficients.
    pub h_kernel: Vec<f64>,
    /// Vertical kernel coefficients.
    pub v_kernel: Vec<f64>,
}

impl SeparableConvolution {
    /// Create a new separable convolution with the given kernels.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::resize::SeparableConvolution;
    ///
    /// let conv = SeparableConvolution::new(
    ///     vec![0.25, 0.5, 0.25],
    ///     vec![0.25, 0.5, 0.25],
    /// );
    /// assert_eq!(conv.h_kernel.len(), 3);
    /// ```
    #[must_use]
    pub fn new(h_kernel: Vec<f64>, v_kernel: Vec<f64>) -> Self {
        Self { h_kernel, v_kernel }
    }

    /// Create a Gaussian separable kernel.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Standard deviation of the Gaussian
    /// * `size` - Kernel size (must be odd)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::resize::SeparableConvolution;
    ///
    /// let conv = SeparableConvolution::gaussian(1.0, 5);
    /// assert_eq!(conv.h_kernel.len(), 5);
    /// ```
    #[must_use]
    pub fn gaussian(sigma: f64, size: usize) -> Self {
        let kernel = create_gaussian_kernel(sigma, size);
        Self::new(kernel.clone(), kernel)
    }

    /// Apply separable convolution to a single-channel image.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Convolved image data.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn apply(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = width as usize * height as usize;
        if src.len() < expected_size {
            return Err(CvError::insufficient_data(expected_size, src.len()));
        }

        // Horizontal pass
        let temp = self.convolve_horizontal(src, width, height);

        // Vertical pass
        let result = self.convolve_vertical(&temp, width, height);

        Ok(result)
    }

    /// Horizontal convolution pass.
    fn convolve_horizontal(&self, src: &[u8], width: u32, height: u32) -> Vec<f64> {
        let half = self.h_kernel.len() / 2;
        let mut dst = vec![0.0; width as usize * height as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let mut sum = 0.0;

                for (ki, &kv) in self.h_kernel.iter().enumerate() {
                    let sx =
                        (x as i32 + ki as i32 - half as i32).clamp(0, width as i32 - 1) as usize;
                    sum += src[y * width as usize + sx] as f64 * kv;
                }

                dst[y * width as usize + x] = sum;
            }
        }

        dst
    }

    /// Vertical convolution pass.
    fn convolve_vertical(&self, src: &[f64], width: u32, height: u32) -> Vec<u8> {
        let half = self.v_kernel.len() / 2;
        let mut dst = vec![0u8; width as usize * height as usize];

        for y in 0..height as usize {
            for x in 0..width as usize {
                let mut sum = 0.0;

                for (ki, &kv) in self.v_kernel.iter().enumerate() {
                    let sy =
                        (y as i32 + ki as i32 - half as i32).clamp(0, height as i32 - 1) as usize;
                    sum += src[sy * width as usize + x] * kv;
                }

                dst[y * width as usize + x] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }

        dst
    }
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

/// Image pyramid for multi-scale processing.
#[derive(Debug, Clone)]
pub struct ImagePyramid {
    /// Pyramid levels (index 0 is original, higher indices are smaller).
    pub levels: Vec<PyramidLevel>,
}

/// Single level in an image pyramid.
#[derive(Debug, Clone)]
pub struct PyramidLevel {
    /// Image data.
    pub data: Vec<u8>,
    /// Level width.
    pub width: u32,
    /// Level height.
    pub height: u32,
}

impl ImagePyramid {
    /// Build a Gaussian pyramid from an image.
    ///
    /// # Arguments
    ///
    /// * `src` - Source image data
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of channels
    /// * `num_levels` - Number of pyramid levels
    ///
    /// # Returns
    ///
    /// Image pyramid with the requested number of levels.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn gaussian(
        src: &[u8],
        width: u32,
        height: u32,
        channels: usize,
        num_levels: usize,
    ) -> CvResult<Self> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let mut levels = Vec::with_capacity(num_levels);

        // Level 0 is the original image
        levels.push(PyramidLevel {
            data: src.to_vec(),
            width,
            height,
        });

        let mut current_width = width;
        let mut current_height = height;
        let mut current_data = src.to_vec();

        for _ in 1..num_levels {
            // Downsample by factor of 2
            let new_width = current_width.div_ceil(2);
            let new_height = current_height.div_ceil(2);

            if new_width == 0 || new_height == 0 {
                break;
            }

            let config = ResizeConfig::new(new_width, new_height, ResizeMethod::Area, channels);
            let new_data = resize_image(&current_data, current_width, current_height, &config)?;

            levels.push(PyramidLevel {
                data: new_data.clone(),
                width: new_width,
                height: new_height,
            });

            current_width = new_width;
            current_height = new_height;
            current_data = new_data;
        }

        Ok(Self { levels })
    }

    /// Get the number of levels in the pyramid.
    #[must_use]
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get a specific level.
    #[must_use]
    pub fn level(&self, index: usize) -> Option<&PyramidLevel> {
        self.levels.get(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_method_properties() {
        assert!(!ResizeMethod::Nearest.is_interpolating());
        assert!(ResizeMethod::Bilinear.is_interpolating());
        assert!(ResizeMethod::Bicubic.is_interpolating());
        assert!(ResizeMethod::Lanczos.is_interpolating());
        assert!(ResizeMethod::Area.is_interpolating());

        assert_eq!(ResizeMethod::Nearest.kernel_size(), 1);
        assert_eq!(ResizeMethod::Bilinear.kernel_size(), 2);
        assert_eq!(ResizeMethod::Bicubic.kernel_size(), 4);
        assert_eq!(ResizeMethod::Lanczos.kernel_size(), 6);
    }

    #[test]
    fn test_resize_config() {
        let config = ResizeConfig::new(640, 480, ResizeMethod::Bilinear, 3);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.channels, 3);
    }

    #[test]
    fn test_resize_nearest() {
        let src = vec![0u8, 64, 128, 255];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Nearest, 1);
        let result = resize_image(&src, 2, 2, &config).expect("resize_image should succeed");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_resize_bilinear() {
        let src = vec![0u8, 255, 255, 0];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Bilinear, 1);
        let result = resize_image(&src, 2, 2, &config).expect("resize_image should succeed");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_resize_bicubic() {
        let src = vec![0u8; 16];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Bicubic, 1);
        let result = resize_image(&src, 4, 4, &config).expect("resize_image should succeed");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_resize_lanczos() {
        let src = vec![128u8; 16];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Lanczos, 1);
        let result = resize_image(&src, 4, 4, &config).expect("resize_image should succeed");
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_resize_area() {
        let src = vec![100u8; 64];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Area, 1);
        let result = resize_image(&src, 8, 8, &config).expect("resize_image should succeed");
        assert_eq!(result.len(), 16);
        // Area averaging should produce similar values
        for &v in &result {
            assert!((95..=105).contains(&v));
        }
    }

    #[test]
    fn test_resize_invalid_dimensions() {
        let src = vec![0u8; 16];
        let config = ResizeConfig::new(4, 4, ResizeMethod::Nearest, 1);

        // Zero source dimensions
        assert!(resize_image(&src, 0, 4, &config).is_err());
        assert!(resize_image(&src, 4, 0, &config).is_err());

        // Zero destination dimensions
        let zero_config = ResizeConfig::new(0, 4, ResizeMethod::Nearest, 1);
        assert!(resize_image(&src, 4, 4, &zero_config).is_err());
    }

    #[test]
    fn test_separable_convolution() {
        let src = vec![128u8; 25];
        let conv = SeparableConvolution::gaussian(1.0, 3);
        let result = conv.apply(&src, 5, 5).expect("apply should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = create_gaussian_kernel(1.0, 5);
        assert_eq!(kernel.len(), 5);

        // Kernel should sum to approximately 1
        let sum: f64 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Center should be the largest value
        assert!(kernel[2] > kernel[1]);
        assert!(kernel[2] > kernel[0]);
    }

    #[test]
    fn test_image_pyramid() {
        let src = vec![100u8; 64];
        let pyramid = ImagePyramid::gaussian(&src, 8, 8, 1, 3).expect("gaussian should succeed");

        assert_eq!(pyramid.num_levels(), 3);
        assert_eq!(pyramid.level(0).expect("level should succeed").width, 8);
        assert_eq!(pyramid.level(1).expect("level should succeed").width, 4);
        assert_eq!(pyramid.level(2).expect("level should succeed").width, 2);
    }

    #[test]
    fn test_cubic_weight() {
        // At center, weight should be 1
        assert!((cubic_weight(0.0) - 1.0).abs() < 1e-10);

        // Weights should decrease away from center
        assert!(cubic_weight(0.5) < cubic_weight(0.0));
        assert!(cubic_weight(1.0) < cubic_weight(0.5));

        // Beyond range 2, weight should be 0
        assert!((cubic_weight(2.5) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_lanczos_weight() {
        // At center, weight should be 1
        assert!((lanczos_weight(0.0, 3) - 1.0).abs() < 1e-10);

        // Beyond range a, weight should be 0
        assert!((lanczos_weight(3.5, 3) - 0.0).abs() < 1e-10);
    }
}
