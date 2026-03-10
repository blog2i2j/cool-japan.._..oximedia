//! CPU-based image upscaling and enhancement algorithms.
//!
//! This module provides pure-CPU implementations of:
//! - Bilinear interpolation upscaling
//! - Bicubic interpolation upscaling (2x/4x)
//! - PSNR (Peak Signal-to-Noise Ratio) quality metric
//! - Unsharp masking for edge-preserving enhancement
//!
//! These work without any ONNX/GPU dependency and serve as fallbacks
//! or standalone enhancements when neural-network models are not available.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::enhance::{SuperResolutionEnhancer, UpscaleMode};
//!
//! let input = vec![128u8; 64 * 64 * 3];
//! let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
//! let output = enhancer.upscale(&input, 64, 64)?;
//! assert_eq!(output.len(), 128 * 128 * 3);
//! ```

#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::many_single_char_names)]
#![allow(dead_code)]

use crate::error::{CvError, CvResult};

/// Upscaling mode for CPU super-resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleMode {
    /// Nearest-neighbor interpolation (fastest, lowest quality).
    Nearest,
    /// Bilinear interpolation (fast, decent quality).
    Bilinear2x,
    /// Bicubic interpolation at 2x scale (good quality).
    Bicubic2x,
    /// Bicubic interpolation at 4x scale (good quality, slower).
    Bicubic4x,
    /// Bicubic with unsharp masking for edge enhancement.
    BicubicSharp2x,
    /// Bicubic with unsharp masking at 4x.
    BicubicSharp4x,
}

impl UpscaleMode {
    /// Get the integer scale factor for this mode.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::UpscaleMode;
    ///
    /// assert_eq!(UpscaleMode::Bicubic2x.scale_factor(), 2);
    /// assert_eq!(UpscaleMode::Bicubic4x.scale_factor(), 4);
    /// ```
    #[must_use]
    pub const fn scale_factor(&self) -> u32 {
        match self {
            Self::Nearest | Self::Bilinear2x | Self::Bicubic2x | Self::BicubicSharp2x => 2,
            Self::Bicubic4x | Self::BicubicSharp4x => 4,
        }
    }

    /// Check if this mode applies unsharp masking post-processing.
    #[must_use]
    pub const fn uses_sharpening(&self) -> bool {
        matches!(self, Self::BicubicSharp2x | Self::BicubicSharp4x)
    }
}

/// CPU-based super-resolution enhancer.
///
/// Provides image upscaling using classical interpolation algorithms without
/// requiring any GPU or ONNX runtime dependencies.
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::{SuperResolutionEnhancer, UpscaleMode};
///
/// let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
/// assert_eq!(enhancer.mode(), UpscaleMode::Bicubic2x);
/// ```
pub struct SuperResolutionEnhancer {
    mode: UpscaleMode,
    /// Sharpening strength for unsharp masking (0.0..2.0).
    sharpness: f32,
    /// Gaussian sigma for unsharp masking blur.
    unsharp_sigma: f32,
}

impl SuperResolutionEnhancer {
    /// Create a new CPU super-resolution enhancer.
    ///
    /// # Arguments
    ///
    /// * `mode` - Upscaling algorithm to use
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::{SuperResolutionEnhancer, UpscaleMode};
    ///
    /// let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bilinear2x);
    /// ```
    #[must_use]
    pub fn new(mode: UpscaleMode) -> Self {
        Self {
            mode,
            sharpness: 0.5,
            unsharp_sigma: 1.0,
        }
    }

    /// Create with custom sharpness settings.
    ///
    /// # Arguments
    ///
    /// * `mode` - Upscaling mode
    /// * `sharpness` - Unsharp mask strength (0.0 to 2.0)
    /// * `sigma` - Gaussian blur sigma for unsharp mask
    #[must_use]
    pub fn with_sharpness(mut self, sharpness: f32, sigma: f32) -> Self {
        self.sharpness = sharpness.clamp(0.0, 2.0);
        self.unsharp_sigma = sigma.max(0.1);
        self
    }

    /// Get the upscaling mode.
    #[must_use]
    pub const fn mode(&self) -> UpscaleMode {
        self.mode
    }

    /// Get the scale factor.
    #[must_use]
    pub fn scale_factor(&self) -> u32 {
        self.mode.scale_factor()
    }

    /// Upscale an RGB image.
    ///
    /// # Arguments
    ///
    /// * `image` - RGB image data (row-major, packed RGB)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    ///
    /// Upscaled RGB image with dimensions `(width * scale, height * scale, 3)`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input dimensions are zero
    /// - Input buffer size doesn't match dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::enhance::{SuperResolutionEnhancer, UpscaleMode};
    ///
    /// let input = vec![128u8; 32 * 32 * 3];
    /// let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
    /// let output = enhancer.upscale(&input, 32, 32)?;
    /// assert_eq!(output.len(), 64 * 64 * 3);
    /// ```
    pub fn upscale(&self, image: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CvError::invalid_dimensions(width, height));
        }

        let expected_size = (width as usize) * (height as usize) * 3;
        if image.len() != expected_size {
            return Err(CvError::insufficient_data(expected_size, image.len()));
        }

        let scale = self.mode.scale_factor();
        let out_width = width * scale;
        let out_height = height * scale;

        let mut output = match self.mode {
            UpscaleMode::Nearest => upscale_nearest(image, width, height, scale),
            UpscaleMode::Bilinear2x => {
                upscale_bilinear_rgb(image, width, height, out_width, out_height)
            }
            UpscaleMode::Bicubic2x
            | UpscaleMode::Bicubic4x
            | UpscaleMode::BicubicSharp2x
            | UpscaleMode::BicubicSharp4x => {
                upscale_bicubic(image, width, height, out_width, out_height)
            }
        };

        if self.mode.uses_sharpening() {
            output = apply_unsharp_mask(
                &output,
                out_width,
                out_height,
                self.unsharp_sigma,
                self.sharpness,
            );
        }

        Ok(output)
    }
}

impl Default for SuperResolutionEnhancer {
    fn default() -> Self {
        Self::new(UpscaleMode::Bicubic2x)
    }
}

/// Nearest-neighbor upscaling.
///
/// # Arguments
///
/// * `image` - RGB image data
/// * `width` - Source width
/// * `height` - Source height
/// * `scale` - Scale factor
pub fn upscale_nearest(image: &[u8], width: u32, height: u32, scale: u32) -> Vec<u8> {
    let out_w = (width * scale) as usize;
    let out_h = (height * scale) as usize;
    let src_w = width as usize;
    let mut output = vec![0u8; out_w * out_h * 3];

    for y in 0..out_h {
        for x in 0..out_w {
            let src_x = x / scale as usize;
            let src_y = y / scale as usize;
            let src_idx = (src_y * src_w + src_x) * 3;
            let dst_idx = (y * out_w + x) * 3;
            output[dst_idx] = image[src_idx];
            output[dst_idx + 1] = image[src_idx + 1];
            output[dst_idx + 2] = image[src_idx + 2];
        }
    }

    output
}

/// Bilinear upscaling for RGB images.
///
/// # Arguments
///
/// * `image` - RGB image data (row-major)
/// * `src_width` - Source image width
/// * `src_height` - Source image height
/// * `dst_width` - Target width
/// * `dst_height` - Target height
pub fn upscale_bilinear_rgb(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let sw = src_width as usize;
    let sh = src_height as usize;
    let dw = dst_width as usize;
    let dh = dst_height as usize;

    let mut output = vec![0u8; dw * dh * 3];

    let x_ratio = (sw as f32 - 1.0) / (dw as f32 - 1.0).max(1.0);
    let y_ratio = (sh as f32 - 1.0) / (dh as f32 - 1.0).max(1.0);

    for y in 0..dh {
        for x in 0..dw {
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            let x0 = (src_x.floor() as usize).min(sw - 1);
            let y0 = (src_y.floor() as usize).min(sh - 1);
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let dst_idx = (y * dw + x) * 3;

            for c in 0..3 {
                let v00 = image[(y0 * sw + x0) * 3 + c] as f32;
                let v01 = image[(y0 * sw + x1) * 3 + c] as f32;
                let v10 = image[(y1 * sw + x0) * 3 + c] as f32;
                let v11 = image[(y1 * sw + x1) * 3 + c] as f32;

                let top = v00 * (1.0 - fx) + v01 * fx;
                let bot = v10 * (1.0 - fx) + v11 * fx;
                let val = top * (1.0 - fy) + bot * fy;

                output[dst_idx + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    output
}

/// Cubic interpolation weight function (Catmull-Rom / Keys kernel).
///
/// # Arguments
///
/// * `t` - Distance from sample point
#[must_use]
pub fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    let a = -0.5_f32; // Catmull-Rom

    if t <= 1.0 {
        (a + 2.0) * t * t * t - (a + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        a * t * t * t - 5.0 * a * t * t + 8.0 * a * t - 4.0 * a
    } else {
        0.0
    }
}

/// Bicubic upscaling for RGB images.
///
/// Uses Catmull-Rom bicubic interpolation for high quality upscaling.
/// Supports arbitrary scale factors.
///
/// # Arguments
///
/// * `image` - RGB image data (row-major, packed RGB)
/// * `src_width` - Source image width
/// * `src_height` - Source image height
/// * `dst_width` - Destination width
/// * `dst_height` - Destination height
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::upscale_bicubic;
///
/// let input = vec![100u8; 16 * 16 * 3];
/// let output = upscale_bicubic(&input, 16, 16, 32, 32);
/// assert_eq!(output.len(), 32 * 32 * 3);
/// ```
pub fn upscale_bicubic(
    image: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Vec<u8> {
    let sw = src_width as usize;
    let sh = src_height as usize;
    let dw = dst_width as usize;
    let dh = dst_height as usize;

    let mut output = vec![0u8; dw * dh * 3];

    let x_ratio = sw as f32 / dw as f32;
    let y_ratio = sh as f32 / dh as f32;

    for y in 0..dh {
        for x in 0..dw {
            // Map destination pixel to source coordinates
            let src_x = (x as f32 + 0.5) * x_ratio - 0.5;
            let src_y = (y as f32 + 0.5) * y_ratio - 0.5;

            let x0 = src_x.floor() as i32;
            let y0 = src_y.floor() as i32;

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            let dst_idx = (y * dw + x) * 3;

            for c in 0..3 {
                let mut value = 0.0_f32;

                for ky in -1i32..=2 {
                    let wy = cubic_weight(fy - ky as f32);
                    let sy = (y0 + ky).clamp(0, sh as i32 - 1) as usize;

                    for kx in -1i32..=2 {
                        let wx = cubic_weight(fx - kx as f32);
                        let sx = (x0 + kx).clamp(0, sw as i32 - 1) as usize;
                        let pixel = image[(sy * sw + sx) * 3 + c] as f32;
                        value += pixel * wx * wy;
                    }
                }

                output[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    output
}

/// Apply unsharp masking for edge enhancement.
///
/// Unsharp masking enhances edges by adding a scaled version of the
/// high-frequency detail back to the image:
/// `output = image + amount * (image - gaussian_blur(image))`
///
/// # Arguments
///
/// * `image` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `sigma` - Gaussian blur sigma (controls blur radius)
/// * `amount` - Enhancement strength (0.0 = no change, 1.0 = full enhancement)
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::apply_unsharp_mask;
///
/// let input = vec![128u8; 32 * 32 * 3];
/// let output = apply_unsharp_mask(&input, 32, 32, 1.0, 0.5);
/// assert_eq!(output.len(), input.len());
/// ```
pub fn apply_unsharp_mask(
    image: &[u8],
    width: u32,
    height: u32,
    sigma: f32,
    amount: f32,
) -> Vec<u8> {
    let blurred = gaussian_blur_rgb(image, width, height, sigma);
    let w = width as usize;
    let h = height as usize;
    let mut output = vec![0u8; w * h * 3];

    for i in 0..output.len() {
        let orig = image[i] as f32;
        let blur = blurred[i] as f32;
        let sharpened = orig + amount * (orig - blur);
        output[i] = sharpened.clamp(0.0, 255.0).round() as u8;
    }

    output
}

/// Gaussian blur for RGB images using separable convolution.
///
/// # Arguments
///
/// * `image` - RGB image data
/// * `width` - Image width
/// * `height` - Image height
/// * `sigma` - Gaussian standard deviation
fn gaussian_blur_rgb(image: &[u8], width: u32, height: u32, sigma: f32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;

    // Build Gaussian kernel
    let radius = ((3.0 * sigma).ceil() as usize).max(1);
    let kernel_size = 2 * radius + 1;
    let mut kernel: Vec<f32> = (0..kernel_size)
        .map(|i| {
            let x = i as f32 - radius as f32;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let kernel_sum: f32 = kernel.iter().sum();
    for k in &mut kernel {
        *k /= kernel_sum;
    }

    // Horizontal pass
    let mut temp = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut val = 0.0_f32;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let sx = (x as i32 + ki as i32 - radius as i32).clamp(0, w as i32 - 1) as usize;
                    val += image[(y * w + sx) * 3 + c] as f32 * kw;
                }
                temp[(y * w + x) * 3 + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Vertical pass
    let mut output = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut val = 0.0_f32;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let sy = (y as i32 + ki as i32 - radius as i32).clamp(0, h as i32 - 1) as usize;
                    val += temp[(sy * w + x) * 3 + c] as f32 * kw;
                }
                output[(y * w + x) * 3 + c] = val.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    output
}

/// Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
///
/// PSNR measures image quality by comparing pixel-level differences.
/// Higher values indicate better quality:
/// - > 40 dB: Excellent (near-lossless)
/// - 30-40 dB: Good quality
/// - 20-30 dB: Acceptable
/// - < 20 dB: Poor quality
///
/// # Arguments
///
/// * `original` - Reference (ground truth) image
/// * `upscaled` - Test image to evaluate
/// * `max_value` - Maximum pixel value (255 for u8 images)
///
/// # Returns
///
/// PSNR value in decibels, or `f64::INFINITY` if images are identical.
///
/// # Errors
///
/// Returns an error if image sizes differ.
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::calculate_psnr;
///
/// let original = vec![100u8; 64 * 64 * 3];
/// let identical = original.clone();
/// let psnr = calculate_psnr(&original, &identical, 255.0)?;
/// assert!(psnr.is_infinite());
/// ```
pub fn calculate_psnr(original: &[u8], upscaled: &[u8], max_value: f64) -> CvResult<f64> {
    if original.len() != upscaled.len() {
        return Err(CvError::insufficient_data(original.len(), upscaled.len()));
    }

    if original.is_empty() {
        return Err(CvError::invalid_dimensions(0, 0));
    }

    let mse: f64 = original
        .iter()
        .zip(upscaled.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b as f64;
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;

    if mse < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    Ok(20.0 * max_value.log10() - 10.0 * mse.log10())
}

/// Calculate MSE (Mean Squared Error) between two images.
///
/// # Arguments
///
/// * `a` - First image
/// * `b` - Second image
///
/// # Returns
///
/// MSE value (lower is better, 0 means identical)
///
/// # Errors
///
/// Returns an error if image sizes differ.
///
/// # Examples
///
/// ```
/// use oximedia_cv::enhance::calculate_mse;
///
/// let a = vec![0u8; 100];
/// let b = vec![0u8; 100];
/// assert_eq!(calculate_mse(&a, &b)?, 0.0);
/// ```
pub fn calculate_mse(a: &[u8], b: &[u8]) -> CvResult<f64> {
    if a.len() != b.len() {
        return Err(CvError::insufficient_data(a.len(), b.len()));
    }

    if a.is_empty() {
        return Ok(0.0);
    }

    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;

    Ok(mse)
}

/// Calculate SSIM (Structural Similarity Index) between two grayscale images.
///
/// SSIM measures perceptual quality taking into account luminance, contrast,
/// and structure. Values range from -1 to 1, where 1 means identical.
///
/// # Arguments
///
/// * `a` - First image (grayscale)
/// * `b` - Second image (grayscale)
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// SSIM value in range [-1, 1]
///
/// # Errors
///
/// Returns an error if image sizes are invalid.
pub fn calculate_ssim(a: &[u8], b: &[u8], width: u32, height: u32) -> CvResult<f64> {
    let size = (width as usize) * (height as usize);
    if a.len() != size || b.len() != size {
        return Err(CvError::insufficient_data(size, a.len().min(b.len())));
    }

    if size == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    // Constants for SSIM (standard values)
    let k1 = 0.01_f64;
    let k2 = 0.03_f64;
    let l = 255.0_f64;
    let c1 = (k1 * l) * (k1 * l);
    let c2 = (k2 * l) * (k2 * l);

    let n = size as f64;

    // Compute means
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Compute variances and covariance
    let var_a: f64 = a
        .iter()
        .map(|&x| {
            let d = x as f64 - mean_a;
            d * d
        })
        .sum::<f64>()
        / n;

    let var_b: f64 = b
        .iter()
        .map(|&x| {
            let d = x as f64 - mean_b;
            d * d
        })
        .sum::<f64>()
        / n;

    let cov_ab: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - mean_a) * (y as f64 - mean_b))
        .sum::<f64>()
        / n;

    // SSIM formula
    let numerator = (2.0 * mean_a * mean_b + c1) * (2.0 * cov_ab + c2);
    let denominator = (mean_a * mean_a + mean_b * mean_b + c1) * (var_a + var_b + c2);

    if denominator.abs() < f64::EPSILON {
        return Ok(1.0);
    }

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_mode_scale_factor() {
        assert_eq!(UpscaleMode::Nearest.scale_factor(), 2);
        assert_eq!(UpscaleMode::Bilinear2x.scale_factor(), 2);
        assert_eq!(UpscaleMode::Bicubic2x.scale_factor(), 2);
        assert_eq!(UpscaleMode::Bicubic4x.scale_factor(), 4);
        assert_eq!(UpscaleMode::BicubicSharp2x.scale_factor(), 2);
        assert_eq!(UpscaleMode::BicubicSharp4x.scale_factor(), 4);
    }

    #[test]
    fn test_upscale_mode_uses_sharpening() {
        assert!(!UpscaleMode::Nearest.uses_sharpening());
        assert!(!UpscaleMode::Bilinear2x.uses_sharpening());
        assert!(!UpscaleMode::Bicubic2x.uses_sharpening());
        assert!(!UpscaleMode::Bicubic4x.uses_sharpening());
        assert!(UpscaleMode::BicubicSharp2x.uses_sharpening());
        assert!(UpscaleMode::BicubicSharp4x.uses_sharpening());
    }

    #[test]
    fn test_super_resolution_enhancer_new() {
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
        assert_eq!(enhancer.mode(), UpscaleMode::Bicubic2x);
        assert_eq!(enhancer.scale_factor(), 2);
    }

    #[test]
    fn test_super_resolution_enhancer_default() {
        let enhancer = SuperResolutionEnhancer::default();
        assert_eq!(enhancer.mode(), UpscaleMode::Bicubic2x);
    }

    #[test]
    fn test_upscale_nearest_2x() {
        // 2x2 image: pixel(0,0)=(R=100,G=200,B=50), pixel(1,0)=(R=75,G=150,B=25),
        //            pixel(0,1)=(R=10,G=20,B=30),   pixel(1,1)=(R=40,G=50,B=60)
        let input = vec![
            100u8, 200, 50, // (0,0)
            75, 150, 25, // (1,0)
            10, 20, 30, // (0,1)
            40, 50, 60, // (1,1)
        ];
        let output = upscale_nearest(&input, 2, 2, 2);
        // Output is 4x4 pixels (each pixel duplicated both ways)
        assert_eq!(output.len(), 4 * 4 * 3, "Output should be 4x4x3");
        // Row 0, x=0: maps to src (0,0) = (100,200,50)
        assert_eq!(output[0], 100, "row0 x=0 R");
        assert_eq!(output[1], 200, "row0 x=0 G");
        assert_eq!(output[2], 50, "row0 x=0 B");
        // Row 0, x=1: also maps to src (0,0) = (100,200,50) [duplicate]
        assert_eq!(output[3], 100, "row0 x=1 R (dup)");
        // Row 0, x=2: maps to src (1,0) = (75,150,25)
        assert_eq!(output[6], 75, "row0 x=2 R");
        assert_eq!(output[7], 150, "row0 x=2 G");
        // Row 0, x=3: also maps to src (1,0) = (75,150,25) [duplicate]
        assert_eq!(output[9], 75, "row0 x=3 R (dup)");
    }

    #[test]
    fn test_upscale_bilinear_2x() {
        let input = vec![0u8; 8 * 8 * 3];
        let output = upscale_bilinear_rgb(&input, 8, 8, 16, 16);
        assert_eq!(output.len(), 16 * 16 * 3);
        // All zeros should stay zeros
        assert!(output.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_upscale_bilinear_uniform_color() {
        // A uniform color image should remain uniform after upscaling
        let input = vec![128u8; 16 * 16 * 3];
        let output = upscale_bilinear_rgb(&input, 16, 16, 32, 32);
        assert_eq!(output.len(), 32 * 32 * 3);
        for &val in &output {
            assert_eq!(
                val, 128,
                "Uniform image should stay uniform after bilinear upscale"
            );
        }
    }

    #[test]
    fn test_cubic_weight_at_zero() {
        // At t=0, weight should be 1.0
        let w = cubic_weight(0.0);
        assert!(
            (w - 1.0).abs() < 1e-6,
            "cubic_weight(0) should be 1.0, got {w}"
        );
    }

    #[test]
    fn test_cubic_weight_at_one() {
        // At t=1.0 exactly, weight should be 0 (boundary)
        let w = cubic_weight(1.0);
        assert!(w >= 0.0 && w <= 0.1, "cubic_weight(1.0) should be near 0");
    }

    #[test]
    fn test_cubic_weight_at_two() {
        // At t >= 2.0, weight should be 0
        let w = cubic_weight(2.0);
        assert_eq!(w, 0.0, "cubic_weight(2.0) should be 0");
        let w2 = cubic_weight(3.0);
        assert_eq!(w2, 0.0, "cubic_weight(3.0) should be 0");
    }

    #[test]
    fn test_upscale_bicubic_2x() {
        let input = vec![128u8; 16 * 16 * 3];
        let output = upscale_bicubic(&input, 16, 16, 32, 32);
        assert_eq!(output.len(), 32 * 32 * 3);
        // Uniform input should produce uniform (or near-uniform) output
        for &val in &output {
            assert!((val as i32 - 128).abs() <= 1, "Expected ~128, got {val}");
        }
    }

    #[test]
    fn test_upscale_bicubic_4x() {
        let input = vec![200u8; 8 * 8 * 3];
        let output = upscale_bicubic(&input, 8, 8, 32, 32);
        assert_eq!(output.len(), 32 * 32 * 3);
        for &val in &output {
            assert!((val as i32 - 200).abs() <= 1, "Expected ~200, got {val}");
        }
    }

    #[test]
    fn test_upscale_via_enhancer_bilinear() {
        let input = vec![100u8; 32 * 32 * 3];
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bilinear2x);
        let output = enhancer
            .upscale(&input, 32, 32)
            .expect("upscale should succeed");
        assert_eq!(output.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_upscale_via_enhancer_bicubic() {
        let input = vec![100u8; 32 * 32 * 3];
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
        let output = enhancer
            .upscale(&input, 32, 32)
            .expect("upscale should succeed");
        assert_eq!(output.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_upscale_via_enhancer_bicubic_sharp() {
        let input = vec![100u8; 16 * 16 * 3];
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::BicubicSharp2x);
        let output = enhancer
            .upscale(&input, 16, 16)
            .expect("upscale should succeed");
        assert_eq!(output.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_upscale_via_enhancer_4x() {
        let input = vec![50u8; 16 * 16 * 3];
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic4x);
        let output = enhancer
            .upscale(&input, 16, 16)
            .expect("upscale should succeed");
        assert_eq!(output.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_upscale_invalid_dimensions() {
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
        let result = enhancer.upscale(&[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_upscale_mismatched_buffer() {
        let enhancer = SuperResolutionEnhancer::new(UpscaleMode::Bicubic2x);
        let small_buffer = vec![0u8; 10]; // Too small for 32x32
        let result = enhancer.upscale(&small_buffer, 32, 32);
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_unsharp_mask_no_effect_on_uniform() {
        // Uniform image should not change much with unsharp masking
        let input = vec![128u8; 32 * 32 * 3];
        let output = apply_unsharp_mask(&input, 32, 32, 1.0, 1.0);
        assert_eq!(output.len(), input.len());
        // Uniform image has no edges, so unsharp mask should not change it significantly
        for (&orig, &sharpened) in input.iter().zip(output.iter()) {
            assert!((orig as i32 - sharpened as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_apply_unsharp_mask_size_preserved() {
        let input = vec![100u8; 64 * 64 * 3];
        let output = apply_unsharp_mask(&input, 64, 64, 1.0, 0.5);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_calculate_psnr_identical() {
        let image = vec![128u8; 64 * 64 * 3];
        let psnr = calculate_psnr(&image, &image, 255.0).expect("calculate_psnr should succeed");
        assert!(
            psnr.is_infinite(),
            "PSNR of identical images should be infinite"
        );
    }

    #[test]
    fn test_calculate_psnr_different() {
        let original = vec![0u8; 100];
        let noisy = vec![10u8; 100];
        let psnr = calculate_psnr(&original, &noisy, 255.0).expect("calculate_psnr should succeed");
        assert!(
            psnr > 0.0 && psnr.is_finite(),
            "PSNR should be finite and positive"
        );
        // MSE = 100.0, PSNR = 20*log10(255) - 10*log10(100) ≈ 48.13 - 20 = 28.13 dB
        assert!(
            psnr > 20.0 && psnr < 40.0,
            "PSNR should be roughly 28 dB, got {psnr}"
        );
    }

    #[test]
    fn test_calculate_psnr_size_mismatch() {
        let a = vec![0u8; 100];
        let b = vec![0u8; 200];
        let result = calculate_psnr(&a, &b, 255.0);
        assert!(result.is_err(), "Should fail on size mismatch");
    }

    #[test]
    fn test_calculate_psnr_empty() {
        let result = calculate_psnr(&[], &[], 255.0);
        assert!(result.is_err(), "Should fail on empty inputs");
    }

    #[test]
    fn test_calculate_mse_identical() {
        let image = vec![128u8; 100];
        let mse = calculate_mse(&image, &image).expect("calculate_mse should succeed");
        assert_eq!(mse, 0.0, "MSE of identical images should be 0");
    }

    #[test]
    fn test_calculate_mse_known() {
        // [0, 0, 0, 0] vs [2, 2, 2, 2]: MSE = 4.0
        let a = vec![0u8; 4];
        let b = vec![2u8; 4];
        let mse = calculate_mse(&a, &b).expect("calculate_mse should succeed");
        assert!((mse - 4.0).abs() < 1e-10, "Expected MSE=4.0, got {mse}");
    }

    #[test]
    fn test_calculate_mse_size_mismatch() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 20];
        assert!(calculate_mse(&a, &b).is_err());
    }

    #[test]
    fn test_calculate_ssim_identical() {
        let image = vec![128u8; 32 * 32];
        let ssim = calculate_ssim(&image, &image, 32, 32).expect("calculate_ssim should succeed");
        assert!(
            (ssim - 1.0).abs() < 1e-6,
            "SSIM of identical images should be ~1.0, got {ssim}"
        );
    }

    #[test]
    fn test_calculate_ssim_different() {
        let a = vec![0u8; 32 * 32];
        let b = vec![255u8; 32 * 32];
        let ssim = calculate_ssim(&a, &b, 32, 32).expect("calculate_ssim should succeed");
        assert!(
            ssim < 0.5,
            "SSIM of opposite images should be low, got {ssim}"
        );
    }

    #[test]
    fn test_calculate_ssim_invalid() {
        let result = calculate_ssim(&[], &[], 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_upscale_with_sharpness_setting() {
        let enhancer =
            SuperResolutionEnhancer::new(UpscaleMode::BicubicSharp2x).with_sharpness(1.0, 2.0);
        let input = vec![128u8; 16 * 16 * 3];
        let output = enhancer
            .upscale(&input, 16, 16)
            .expect("upscale should succeed");
        assert_eq!(output.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_gaussian_blur_uniform() {
        // Gaussian blur of uniform image should remain uniform
        let input = vec![200u8; 32 * 32 * 3];
        let blurred = gaussian_blur_rgb(&input, 32, 32, 1.5);
        assert_eq!(blurred.len(), input.len());
        // Interior pixels should be close to 200
        for &val in blurred[3..blurred.len() - 3].iter() {
            assert!((val as i32 - 200).abs() <= 2, "Expected ~200, got {val}");
        }
    }
}
