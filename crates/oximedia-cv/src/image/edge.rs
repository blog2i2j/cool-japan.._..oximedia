//! Edge detection operations.
//!
//! This module provides various edge detection algorithms including:
//! - Sobel edge detection (horizontal, vertical, magnitude)
//! - Canny edge detection (with non-maximum suppression and hysteresis)
//! - Laplacian edge detection
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::{SobelEdge, EdgeDetector};
//!
//! let sobel = SobelEdge::new();
//! ```

use crate::error::{CvError, CvResult};

/// Trait for edge detectors.
pub trait EdgeDetector {
    /// Detect edges in a grayscale image.
    ///
    /// # Arguments
    ///
    /// * `src` - Source grayscale image data
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    ///
    /// Edge magnitude image.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    fn detect(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>>;
}

/// Sobel edge detector.
///
/// Computes gradient magnitude using Sobel operators.
#[derive(Debug, Clone, Copy, Default)]
pub struct SobelEdge {
    /// Whether to normalize the output to 0-255.
    normalize: bool,
}

impl SobelEdge {
    /// Create a new Sobel edge detector.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::SobelEdge;
    ///
    /// let sobel = SobelEdge::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self { normalize: true }
    }

    /// Set whether to normalize output.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Compute horizontal gradient (Sobel X).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn gradient_x(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<i16>> {
        validate_input(src, width, height)?;
        Ok(sobel_x(src, width, height))
    }

    /// Compute vertical gradient (Sobel Y).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn gradient_y(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<i16>> {
        validate_input(src, width, height)?;
        Ok(sobel_y(src, width, height))
    }

    /// Compute both gradients and gradient direction.
    ///
    /// # Returns
    ///
    /// Tuple of (magnitude, direction in radians).
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are invalid.
    pub fn gradient_with_direction(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
    ) -> CvResult<(Vec<f64>, Vec<f64>)> {
        validate_input(src, width, height)?;

        let gx = sobel_x(src, width, height);
        let gy = sobel_y(src, width, height);

        let size = width as usize * height as usize;
        let mut magnitude = vec![0.0; size];
        let mut direction = vec![0.0; size];

        for i in 0..size {
            let x = gx[i] as f64;
            let y = gy[i] as f64;
            magnitude[i] = (x * x + y * y).sqrt();
            direction[i] = y.atan2(x);
        }

        Ok((magnitude, direction))
    }
}

impl EdgeDetector for SobelEdge {
    fn detect(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let gx = sobel_x(src, width, height);
        let gy = sobel_y(src, width, height);

        let size = width as usize * height as usize;
        let mut magnitude: Vec<f64> = vec![0.0; size];
        let mut max_mag = 0.0f64;

        for i in 0..size {
            let x = gx[i] as f64;
            let y = gy[i] as f64;
            let mag = (x * x + y * y).sqrt();
            magnitude[i] = mag;
            max_mag = max_mag.max(mag);
        }

        let mut result = vec![0u8; size];

        if self.normalize && max_mag > f64::EPSILON {
            for i in 0..size {
                result[i] = (magnitude[i] * 255.0 / max_mag).round().clamp(0.0, 255.0) as u8;
            }
        } else {
            for i in 0..size {
                result[i] = magnitude[i].round().clamp(0.0, 255.0) as u8;
            }
        }

        Ok(result)
    }
}

/// Canny edge detector.
///
/// Implements the Canny algorithm with Gaussian smoothing,
/// gradient computation, non-maximum suppression, and hysteresis thresholding.
#[derive(Debug, Clone)]
pub struct CannyEdge {
    /// Low threshold for hysteresis.
    low_threshold: f64,
    /// High threshold for hysteresis.
    high_threshold: f64,
    /// Gaussian sigma for smoothing.
    sigma: f64,
}

impl CannyEdge {
    /// Create a new Canny edge detector.
    ///
    /// # Arguments
    ///
    /// * `low_threshold` - Low threshold for hysteresis (0-255)
    /// * `high_threshold` - High threshold for hysteresis (0-255)
    /// * `sigma` - Gaussian sigma for pre-smoothing
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::CannyEdge;
    ///
    /// let canny = CannyEdge::new(50.0, 150.0, 1.4);
    /// ```
    #[must_use]
    pub fn new(low_threshold: f64, high_threshold: f64, sigma: f64) -> Self {
        Self {
            low_threshold,
            high_threshold,
            sigma,
        }
    }

    /// Get the low threshold.
    #[must_use]
    pub const fn low_threshold(&self) -> f64 {
        self.low_threshold
    }

    /// Get the high threshold.
    #[must_use]
    pub const fn high_threshold(&self) -> f64 {
        self.high_threshold
    }
}

impl Default for CannyEdge {
    fn default() -> Self {
        Self::new(50.0, 150.0, 1.4)
    }
}

impl EdgeDetector for CannyEdge {
    fn detect(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let w = width as usize;
        let h = height as usize;
        let size = w * h;

        // Step 1: Gaussian smoothing
        let smoothed = gaussian_smooth(src, width, height, self.sigma);

        // Step 2: Compute gradients
        let gx = sobel_x(&smoothed, width, height);
        let gy = sobel_y(&smoothed, width, height);

        let mut magnitude = vec![0.0; size];
        let mut direction = vec![0.0; size];

        for i in 0..size {
            let x = gx[i] as f64;
            let y = gy[i] as f64;
            magnitude[i] = (x * x + y * y).sqrt();
            direction[i] = y.atan2(x);
        }

        // Step 3: Non-maximum suppression
        let suppressed = non_maximum_suppression(&magnitude, &direction, w, h);

        // Step 4: Double thresholding and hysteresis
        let result =
            hysteresis_thresholding(&suppressed, w, h, self.low_threshold, self.high_threshold);

        Ok(result)
    }
}

/// Laplacian edge detector.
///
/// Uses the Laplacian operator to detect edges.
#[derive(Debug, Clone, Copy, Default)]
pub struct LaplacianEdge {
    /// Kernel type (4-connected or 8-connected).
    eight_connected: bool,
}

impl LaplacianEdge {
    /// Create a new Laplacian edge detector.
    ///
    /// Uses 4-connected Laplacian by default.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::LaplacianEdge;
    ///
    /// let laplacian = LaplacianEdge::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            eight_connected: false,
        }
    }

    /// Use 8-connected Laplacian kernel.
    #[must_use]
    pub const fn eight_connected(mut self) -> Self {
        self.eight_connected = true;
        self
    }
}

impl EdgeDetector for LaplacianEdge {
    fn detect(&self, src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        validate_input(src, width, height)?;

        let w = width as usize;
        let h = height as usize;
        let size = w * h;

        // Laplacian kernels
        let kernel: &[i32] = if self.eight_connected {
            &[1, 1, 1, 1, -8, 1, 1, 1, 1]
        } else {
            &[0, 1, 0, 1, -4, 1, 0, 1, 0]
        };

        let mut result = vec![0u8; size];
        let mut max_val = 0i32;
        let mut values = vec![0i32; size];

        for y in 0..h {
            for x in 0..w {
                let mut sum = 0i32;

                for ky in 0..3 {
                    let sy = (y as i32 + ky as i32 - 1).clamp(0, h as i32 - 1) as usize;

                    for kx in 0..3 {
                        let sx = (x as i32 + kx as i32 - 1).clamp(0, w as i32 - 1) as usize;
                        sum += src[sy * w + sx] as i32 * kernel[ky * 3 + kx];
                    }
                }

                let abs_sum = sum.abs();
                values[y * w + x] = abs_sum;
                max_val = max_val.max(abs_sum);
            }
        }

        // Normalize
        if max_val > 0 {
            for i in 0..size {
                result[i] = (values[i] * 255 / max_val) as u8;
            }
        }

        Ok(result)
    }
}

/// Compute Sobel X gradient.
fn sobel_x(src: &[u8], width: u32, height: u32) -> Vec<i16> {
    let w = width as usize;
    let h = height as usize;
    let mut result = vec![0i16; w * h];

    // Sobel X kernel: [-1, 0, 1]
    //                 [-2, 0, 2]
    //                 [-1, 0, 1]

    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);

            let p00 = src[y0 * w + x0] as i16;
            let p01 = src[y * w + x0] as i16;
            let p02 = src[y2 * w + x0] as i16;
            let p20 = src[y0 * w + x2] as i16;
            let p21 = src[y * w + x2] as i16;
            let p22 = src[y2 * w + x2] as i16;

            result[y * w + x] = -p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22;
        }
    }

    result
}

/// Compute Sobel Y gradient.
fn sobel_y(src: &[u8], width: u32, height: u32) -> Vec<i16> {
    let w = width as usize;
    let h = height as usize;
    let mut result = vec![0i16; w * h];

    // Sobel Y kernel: [-1, -2, -1]
    //                 [ 0,  0,  0]
    //                 [ 1,  2,  1]

    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);
            let y0 = y.saturating_sub(1);
            let y2 = (y + 1).min(h - 1);

            let p00 = src[y0 * w + x0] as i16;
            let p10 = src[y0 * w + x] as i16;
            let p20 = src[y0 * w + x2] as i16;
            let p02 = src[y2 * w + x0] as i16;
            let p12 = src[y2 * w + x] as i16;
            let p22 = src[y2 * w + x2] as i16;

            result[y * w + x] = -p00 - 2 * p10 - p20 + p02 + 2 * p12 + p22;
        }
    }

    result
}

/// Simple Gaussian smoothing.
fn gaussian_smooth(src: &[u8], width: u32, height: u32, sigma: f64) -> Vec<u8> {
    let kernel_size = ((sigma * 6.0).ceil() as usize) | 1;
    let kernel_size = kernel_size.max(3);

    let kernel = create_gaussian_kernel(sigma, kernel_size);

    // Separable convolution
    let temp = convolve_h_f64(src, width, height, &kernel);
    convolve_v_f64(&temp, width, height, &kernel)
}

/// Create Gaussian kernel.
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

    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

/// Horizontal convolution.
fn convolve_h_f64(src: &[u8], width: u32, height: u32, kernel: &[f64]) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let half = kernel.len() / 2;
    let mut result = vec![0.0; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = (x as i32 + ki as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                sum += src[y * w + sx] as f64 * kv;
            }
            result[y * w + x] = sum;
        }
    }

    result
}

/// Vertical convolution.
fn convolve_v_f64(src: &[f64], width: u32, height: u32, kernel: &[f64]) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let half = kernel.len() / 2;
    let mut result = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = (y as i32 + ki as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                sum += src[sy * w + x] * kv;
            }
            result[y * w + x] = sum.round().clamp(0.0, 255.0) as u8;
        }
    }

    result
}

/// Non-maximum suppression for Canny edge detection.
fn non_maximum_suppression(
    magnitude: &[f64],
    direction: &[f64],
    width: usize,
    height: usize,
) -> Vec<f64> {
    let mut result = vec![0.0; width * height];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mag = magnitude[idx];
            let dir = direction[idx];

            // Quantize direction to 4 angles (0, 45, 90, 135 degrees)
            let angle = ((dir * 180.0 / std::f64::consts::PI) + 180.0) % 180.0;

            let (n1_idx, n2_idx) = if !(22.5..157.5).contains(&angle) {
                // Horizontal edge (gradient is vertical)
                (idx - 1, idx + 1)
            } else if angle < 67.5 {
                // Diagonal (45 degrees)
                (idx - width - 1, idx + width + 1)
            } else if angle < 112.5 {
                // Vertical edge (gradient is horizontal)
                (idx - width, idx + width)
            } else {
                // Diagonal (135 degrees)
                (idx - width + 1, idx + width - 1)
            };

            // Keep only local maxima
            if mag >= magnitude[n1_idx] && mag >= magnitude[n2_idx] {
                result[idx] = mag;
            }
        }
    }

    result
}

/// Hysteresis thresholding for Canny edge detection.
fn hysteresis_thresholding(
    edges: &[f64],
    width: usize,
    height: usize,
    low_threshold: f64,
    high_threshold: f64,
) -> Vec<u8> {
    let size = width * height;
    let mut result = vec![0u8; size];

    // Mark strong edges
    let mut strong_edges: Vec<(usize, usize)> = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let val = edges[idx];

            if val >= high_threshold {
                result[idx] = 255;
                strong_edges.push((x, y));
            } else if val >= low_threshold {
                result[idx] = 128; // Weak edge (candidate)
            }
        }
    }

    // Trace weak edges connected to strong edges
    while let Some((x, y)) = strong_edges.pop() {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                    let nidx = ny as usize * width + nx as usize;
                    if result[nidx] == 128 {
                        result[nidx] = 255;
                        strong_edges.push((nx as usize, ny as usize));
                    }
                }
            }
        }
    }

    // Remove remaining weak edges
    for pixel in &mut result {
        if *pixel == 128 {
            *pixel = 0;
        }
    }

    result
}

/// Validate input dimensions.
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

/// Compute gradient direction in degrees (0-360).
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn gradient_direction(src: &[u8], width: u32, height: u32) -> CvResult<Vec<f64>> {
    validate_input(src, width, height)?;

    let gx = sobel_x(src, width, height);
    let gy = sobel_y(src, width, height);

    let size = width as usize * height as usize;
    let mut direction = vec![0.0; size];

    for i in 0..size {
        let angle = (gy[i] as f64).atan2(gx[i] as f64);
        // Convert to degrees and make positive
        let degrees = angle * 180.0 / std::f64::consts::PI;
        direction[i] = if degrees < 0.0 {
            degrees + 360.0
        } else {
            degrees
        };
    }

    Ok(direction)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sobel_edge_new() {
        let sobel = SobelEdge::new();
        assert!(sobel.normalize);
    }

    #[test]
    fn test_sobel_edge_detect() {
        // Create a simple edge image
        let mut src = vec![0u8; 25];
        for y in 0..5 {
            for x in 0..5 {
                if x >= 2 {
                    src[y * 5 + x] = 255;
                }
            }
        }

        let sobel = SobelEdge::new();
        let result = sobel.detect(&src, 5, 5).expect("detect should succeed");
        assert_eq!(result.len(), 25);

        // Edge should be detected around x=2
        assert!(result[2 * 5 + 2] > 0);
    }

    #[test]
    fn test_sobel_gradient() {
        let src = vec![100u8; 25];
        let sobel = SobelEdge::new();

        let gx = sobel
            .gradient_x(&src, 5, 5)
            .expect("gradient_x should succeed");
        let gy = sobel
            .gradient_y(&src, 5, 5)
            .expect("gradient_y should succeed");

        assert_eq!(gx.len(), 25);
        assert_eq!(gy.len(), 25);

        // Uniform image should have zero gradient
        for &v in &gx {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_canny_edge() {
        let src = vec![100u8; 100];
        let canny = CannyEdge::new(50.0, 150.0, 1.4);
        let result = canny.detect(&src, 10, 10).expect("detect should succeed");
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_canny_default() {
        let canny = CannyEdge::default();
        assert!((canny.low_threshold() - 50.0).abs() < f64::EPSILON);
        assert!((canny.high_threshold() - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_laplacian_edge() {
        let src = vec![100u8; 25];
        let laplacian = LaplacianEdge::new();
        let result = laplacian.detect(&src, 5, 5).expect("detect should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_laplacian_eight_connected() {
        let src = vec![100u8; 25];
        let laplacian = LaplacianEdge::new().eight_connected();
        let result = laplacian.detect(&src, 5, 5).expect("detect should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_gradient_direction() {
        let src = vec![100u8; 25];
        let dir = gradient_direction(&src, 5, 5).expect("gradient_direction should succeed");
        assert_eq!(dir.len(), 25);
    }

    #[test]
    fn test_invalid_dimensions() {
        let src = vec![0u8; 25];
        let sobel = SobelEdge::new();
        assert!(sobel.detect(&src, 0, 5).is_err());
        assert!(sobel.detect(&src, 5, 0).is_err());
    }
}
