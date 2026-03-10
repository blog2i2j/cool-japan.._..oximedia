//! Color space conversion operations.
//!
//! This module provides functions for converting between various color spaces
//! including RGB, BGR, YUV (BT.601 and BT.709), HSV, HSL, LAB, and Grayscale.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::ColorSpace;
//!
//! let space = ColorSpace::Rgb;
//! assert_eq!(space.channels(), 3);
//! ```

use crate::error::{CvError, CvResult};

/// Color space enumeration.
///
/// Represents different color spaces supported for conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorSpace {
    /// RGB color space (Red, Green, Blue).
    #[default]
    Rgb,
    /// BGR color space (Blue, Green, Red) - common in `OpenCV`.
    Bgr,
    /// YUV color space with BT.601 coefficients (SD video).
    YuvBt601,
    /// YUV color space with BT.709 coefficients (HD video).
    YuvBt709,
    /// HSV color space (Hue, Saturation, Value).
    Hsv,
    /// HSL color space (Hue, Saturation, Lightness).
    Hsl,
    /// CIELAB color space.
    Lab,
    /// Grayscale (single channel).
    Grayscale,
}

impl ColorSpace {
    /// Returns the number of channels for this color space.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::ColorSpace;
    ///
    /// assert_eq!(ColorSpace::Rgb.channels(), 3);
    /// assert_eq!(ColorSpace::Grayscale.channels(), 1);
    /// ```
    #[must_use]
    pub const fn channels(&self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::Rgb
            | Self::Bgr
            | Self::YuvBt601
            | Self::YuvBt709
            | Self::Hsv
            | Self::Hsl
            | Self::Lab => 3,
        }
    }

    /// Returns true if this is a luminance-based color space.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::ColorSpace;
    ///
    /// assert!(ColorSpace::YuvBt601.is_luminance_based());
    /// assert!(ColorSpace::Lab.is_luminance_based());
    /// assert!(!ColorSpace::Rgb.is_luminance_based());
    /// ```
    #[must_use]
    pub const fn is_luminance_based(&self) -> bool {
        matches!(
            self,
            Self::YuvBt601 | Self::YuvBt709 | Self::Lab | Self::Grayscale
        )
    }
}

/// YUV matrix coefficients for color conversion.
#[derive(Debug, Clone, Copy)]
pub struct YuvCoefficients {
    /// Red coefficient for Y calculation.
    pub kr: f64,
    /// Green coefficient for Y calculation (calculated as 1 - kr - kb).
    pub kg: f64,
    /// Blue coefficient for Y calculation.
    pub kb: f64,
}

impl YuvCoefficients {
    /// BT.601 coefficients (SD video).
    pub const BT601: Self = Self {
        kr: 0.299,
        kg: 0.587,
        kb: 0.114,
    };

    /// BT.709 coefficients (HD video).
    pub const BT709: Self = Self {
        kr: 0.2126,
        kg: 0.7152,
        kb: 0.0722,
    };
}

/// Convert RGB image to BGR.
///
/// # Arguments
///
/// * `src` - Source RGB image data (interleaved)
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// BGR image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_bgr(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    for i in 0..pixel_count {
        let idx = i * 3;
        dst[idx] = src[idx + 2]; // B <- R
        dst[idx + 1] = src[idx + 1]; // G <- G
        dst[idx + 2] = src[idx]; // R <- B
    }

    Ok(dst)
}

/// Convert BGR image to RGB.
///
/// This is the same operation as `rgb_to_bgr` (channel swap is symmetric).
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn bgr_to_rgb(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    rgb_to_bgr(src, width, height)
}

/// Convert RGB image to YUV using specified coefficients.
///
/// # Arguments
///
/// * `src` - Source RGB image data (interleaved)
/// * `width` - Image width
/// * `height` - Image height
/// * `coefficients` - YUV conversion coefficients
///
/// # Returns
///
/// YUV image data (interleaved Y, U, V per pixel).
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_yuv(
    src: &[u8],
    width: u32,
    height: u32,
    coefficients: YuvCoefficients,
) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    let kr = coefficients.kr;
    let kg = coefficients.kg;
    let kb = coefficients.kb;

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = src[idx] as f64;
        let g = src[idx + 1] as f64;
        let b = src[idx + 2] as f64;

        // Y = Kr*R + Kg*G + Kb*B
        let y = kr * r + kg * g + kb * b;

        // U = (B - Y) / (2 * (1 - Kb)) + 128
        // V = (R - Y) / (2 * (1 - Kr)) + 128
        let u = (b - y) / (2.0 * (1.0 - kb)) + 128.0;
        let v = (r - y) / (2.0 * (1.0 - kr)) + 128.0;

        dst[idx] = y.round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = u.round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = v.round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert YUV image to RGB using specified coefficients.
///
/// # Arguments
///
/// * `src` - Source YUV image data (interleaved)
/// * `width` - Image width
/// * `height` - Image height
/// * `coefficients` - YUV conversion coefficients
///
/// # Returns
///
/// RGB image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn yuv_to_rgb(
    src: &[u8],
    width: u32,
    height: u32,
    coefficients: YuvCoefficients,
) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    let kr = coefficients.kr;
    let kg = coefficients.kg;
    let kb = coefficients.kb;

    for i in 0..pixel_count {
        let idx = i * 3;
        let y = src[idx] as f64;
        let u = src[idx + 1] as f64 - 128.0;
        let v = src[idx + 2] as f64 - 128.0;

        // R = Y + V * 2 * (1 - Kr)
        // B = Y + U * 2 * (1 - Kb)
        // G = (Y - Kr * R - Kb * B) / Kg
        let r = y + v * 2.0 * (1.0 - kr);
        let b = y + u * 2.0 * (1.0 - kb);
        let g = (y - kr * r - kb * b) / kg;

        dst[idx] = r.round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = g.round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = b.round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert RGB image to HSV.
///
/// Output ranges:
/// - H: 0-180 (degrees / 2 to fit in u8)
/// - S: 0-255
/// - V: 0-255
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_hsv(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = src[idx] as f64 / 255.0;
        let g = src[idx + 1] as f64 / 255.0;
        let b = src[idx + 2] as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        // Value
        let v = max;

        // Saturation
        let s = if max > f64::EPSILON { delta / max } else { 0.0 };

        // Hue
        let h = if delta.abs() < f64::EPSILON {
            0.0
        } else if (max - r).abs() < f64::EPSILON {
            60.0 * (((g - b) / delta) % 6.0)
        } else if (max - g).abs() < f64::EPSILON {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        let h = if h < 0.0 { h + 360.0 } else { h };

        // Scale to u8 range (H: 0-180, S: 0-255, V: 0-255)
        dst[idx] = (h / 2.0).round().clamp(0.0, 180.0) as u8;
        dst[idx + 1] = (s * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = (v * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert HSV image to RGB.
///
/// Input ranges:
/// - H: 0-180 (degrees / 2)
/// - S: 0-255
/// - V: 0-255
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn hsv_to_rgb(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    for i in 0..pixel_count {
        let idx = i * 3;
        let h = src[idx] as f64 * 2.0; // Convert back to 0-360
        let s = src[idx + 1] as f64 / 255.0;
        let v = src[idx + 2] as f64 / 255.0;

        let c = v * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        dst[idx] = ((r + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = ((g + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = ((b + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert RGB image to HSL.
///
/// Output ranges:
/// - H: 0-180 (degrees / 2 to fit in u8)
/// - S: 0-255
/// - L: 0-255
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_hsl(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    for i in 0..pixel_count {
        let idx = i * 3;
        let r = src[idx] as f64 / 255.0;
        let g = src[idx + 1] as f64 / 255.0;
        let b = src[idx + 2] as f64 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        // Lightness
        let l = (max + min) / 2.0;

        // Saturation
        let s = if delta.abs() < f64::EPSILON {
            0.0
        } else {
            delta / (1.0 - (2.0 * l - 1.0).abs())
        };

        // Hue (same as HSV)
        let h = if delta.abs() < f64::EPSILON {
            0.0
        } else if (max - r).abs() < f64::EPSILON {
            60.0 * (((g - b) / delta) % 6.0)
        } else if (max - g).abs() < f64::EPSILON {
            60.0 * ((b - r) / delta + 2.0)
        } else {
            60.0 * ((r - g) / delta + 4.0)
        };

        let h = if h < 0.0 { h + 360.0 } else { h };

        dst[idx] = (h / 2.0).round().clamp(0.0, 180.0) as u8;
        dst[idx + 1] = (s * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = (l * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert HSL image to RGB.
///
/// Input ranges:
/// - H: 0-180 (degrees / 2)
/// - S: 0-255
/// - L: 0-255
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn hsl_to_rgb(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    for i in 0..pixel_count {
        let idx = i * 3;
        let h = src[idx] as f64 * 2.0; // Convert back to 0-360
        let s = src[idx + 1] as f64 / 255.0;
        let l = src[idx + 2] as f64 / 255.0;

        let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
        let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
        let m = l - c / 2.0;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        dst[idx] = ((r + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = ((g + m) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = ((b + m) * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert RGB image to CIELAB color space.
///
/// Uses D65 illuminant as reference white.
///
/// Output ranges:
/// - L: 0-255 (scaled from 0-100)
/// - a: 0-255 (scaled from -128 to 127, with 128 as zero)
/// - b: 0-255 (scaled from -128 to 127, with 128 as zero)
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_lab(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    // D65 reference white
    const XN: f64 = 0.950_456;
    const YN: f64 = 1.0;
    const ZN: f64 = 1.088_754;

    for i in 0..pixel_count {
        let idx = i * 3;

        // Convert sRGB to linear RGB
        let r = srgb_to_linear(src[idx] as f64 / 255.0);
        let g = srgb_to_linear(src[idx + 1] as f64 / 255.0);
        let b = srgb_to_linear(src[idx + 2] as f64 / 255.0);

        // Convert linear RGB to XYZ
        let x = 0.412_453 * r + 0.357_580 * g + 0.180_423 * b;
        let y = 0.212_671 * r + 0.715_160 * g + 0.072_169 * b;
        let z = 0.019_334 * r + 0.119_193 * g + 0.950_227 * b;

        // Convert XYZ to Lab
        let fx = lab_f(x / XN);
        let fy = lab_f(y / YN);
        let fz = lab_f(z / ZN);

        let l_star = 116.0 * fy - 16.0;
        let a_star = 500.0 * (fx - fy);
        let b_star = 200.0 * (fy - fz);

        // Scale to u8 range
        dst[idx] = (l_star * 255.0 / 100.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = (a_star + 128.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = (b_star + 128.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert CIELAB image to RGB.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn lab_to_rgb(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let mut dst = vec![0u8; src.len()];
    let pixel_count = (width * height) as usize;

    // D65 reference white
    const XN: f64 = 0.950_456;
    const YN: f64 = 1.0;
    const ZN: f64 = 1.088_754;

    for i in 0..pixel_count {
        let idx = i * 3;

        // Unscale from u8 range
        let l_star = src[idx] as f64 * 100.0 / 255.0;
        let a_star = src[idx + 1] as f64 - 128.0;
        let b_star = src[idx + 2] as f64 - 128.0;

        // Convert Lab to XYZ
        let fy = (l_star + 16.0) / 116.0;
        let fx = a_star / 500.0 + fy;
        let fz = fy - b_star / 200.0;

        let x = XN * lab_f_inv(fx);
        let y = YN * lab_f_inv(fy);
        let z = ZN * lab_f_inv(fz);

        // Convert XYZ to linear RGB
        let r = 3.240_479 * x - 1.537_150 * y - 0.498_535 * z;
        let g = -0.969_256 * x + 1.875_992 * y + 0.041_556 * z;
        let b = 0.055_648 * x - 0.204_043 * y + 1.057_311 * z;

        // Convert linear RGB to sRGB
        dst[idx] = (linear_to_srgb(r) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 1] = (linear_to_srgb(g) * 255.0).round().clamp(0.0, 255.0) as u8;
        dst[idx + 2] = (linear_to_srgb(b) * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert sRGB to linear RGB.
#[inline]
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB to sRGB.
#[inline]
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Lab f function.
#[inline]
fn lab_f(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;
    const DELTA_CUBE: f64 = DELTA * DELTA * DELTA;

    if t > DELTA_CUBE {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// Lab inverse f function.
#[inline]
fn lab_f_inv(t: f64) -> f64 {
    const DELTA: f64 = 6.0 / 29.0;

    if t > DELTA {
        t * t * t
    } else {
        3.0 * DELTA * DELTA * (t - 4.0 / 29.0)
    }
}

/// Convert RGB image to grayscale.
///
/// Uses BT.709 coefficients for luminance calculation.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn rgb_to_grayscale(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    validate_dimensions(src, width, height, 3)?;

    let pixel_count = (width * height) as usize;
    let mut dst = vec![0u8; pixel_count];

    let coeffs = YuvCoefficients::BT709;

    for i in 0..pixel_count {
        let src_idx = i * 3;
        let r = src[src_idx] as f64;
        let g = src[src_idx + 1] as f64;
        let b = src[src_idx + 2] as f64;

        let gray = coeffs.kr * r + coeffs.kg * g + coeffs.kb * b;
        dst[i] = gray.round().clamp(0.0, 255.0) as u8;
    }

    Ok(dst)
}

/// Convert grayscale image to RGB.
///
/// Duplicates the grayscale value across all three channels.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn grayscale_to_rgb(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let pixel_count = (width * height) as usize;
    if src.len() < pixel_count {
        return Err(CvError::insufficient_data(pixel_count, src.len()));
    }

    let mut dst = vec![0u8; pixel_count * 3];

    for i in 0..pixel_count {
        let gray = src[i];
        let dst_idx = i * 3;
        dst[dst_idx] = gray;
        dst[dst_idx + 1] = gray;
        dst[dst_idx + 2] = gray;
    }

    Ok(dst)
}

/// Generic color space converter.
///
/// Converts between any two supported color spaces.
///
/// # Arguments
///
/// * `src` - Source image data
/// * `width` - Image width
/// * `height` - Image height
/// * `from` - Source color space
/// * `to` - Destination color space
///
/// # Returns
///
/// Converted image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or conversion is not supported.
pub fn convert_color_space(
    src: &[u8],
    width: u32,
    height: u32,
    from: ColorSpace,
    to: ColorSpace,
) -> CvResult<Vec<u8>> {
    if from == to {
        return Ok(src.to_vec());
    }

    // First convert to RGB as intermediate
    let rgb = match from {
        ColorSpace::Rgb => src.to_vec(),
        ColorSpace::Bgr => bgr_to_rgb(src, width, height)?,
        ColorSpace::YuvBt601 => yuv_to_rgb(src, width, height, YuvCoefficients::BT601)?,
        ColorSpace::YuvBt709 => yuv_to_rgb(src, width, height, YuvCoefficients::BT709)?,
        ColorSpace::Hsv => hsv_to_rgb(src, width, height)?,
        ColorSpace::Hsl => hsl_to_rgb(src, width, height)?,
        ColorSpace::Lab => lab_to_rgb(src, width, height)?,
        ColorSpace::Grayscale => grayscale_to_rgb(src, width, height)?,
    };

    // Then convert RGB to target
    match to {
        ColorSpace::Rgb => Ok(rgb),
        ColorSpace::Bgr => rgb_to_bgr(&rgb, width, height),
        ColorSpace::YuvBt601 => rgb_to_yuv(&rgb, width, height, YuvCoefficients::BT601),
        ColorSpace::YuvBt709 => rgb_to_yuv(&rgb, width, height, YuvCoefficients::BT709),
        ColorSpace::Hsv => rgb_to_hsv(&rgb, width, height),
        ColorSpace::Hsl => rgb_to_hsl(&rgb, width, height),
        ColorSpace::Lab => rgb_to_lab(&rgb, width, height),
        ColorSpace::Grayscale => rgb_to_grayscale(&rgb, width, height),
    }
}

/// Convert RGB24 (packed, R first) to grayscale using BT.601 luma coefficients.
///
/// Output is a single-channel image of size `w * h` bytes.
/// BT.601: Y = 0.299*R + 0.587*G + 0.114*B
///
/// # Arguments
///
/// * `src` - Source RGB image data (interleaved, 3 bytes per pixel)
/// * `w` - Image width in pixels
/// * `h` - Image height in pixels
///
/// # Returns
///
/// Grayscale image data (`w * h` bytes).
///
/// # Errors
///
/// Returns an error if the source buffer is too small.
pub fn rgb_to_grayscale_bt601(src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
    let pixel_count = w * h;
    let expected = pixel_count * 3;
    if src.len() < expected {
        return Err(CvError::insufficient_data(expected, src.len()));
    }
    let mut out = vec![0u8; pixel_count];
    for i in 0..pixel_count {
        let r = src[i * 3] as f32;
        let g = src[i * 3 + 1] as f32;
        let b = src[i * 3 + 2] as f32;
        out[i] = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
    }
    Ok(out)
}

/// Convert BGR24 to YUV420P (planar: Y plane full-res, then U plane half-res, then V plane half-res).
///
/// The output buffer layout is:
/// - bytes `[0 .. w*h)`: Y plane
/// - bytes `[w*h .. w*h + (w/2)*(h/2))`: U (Cb) plane
/// - bytes `[w*h + (w/2)*(h/2) .. w*h + 2*(w/2)*(h/2))`: V (Cr) plane
///
/// Coefficients follow the studio-swing BT.601 convention (Y: 16-235, UV: 16-240).
///
/// # Arguments
///
/// * `src` - Source BGR image data (interleaved, 3 bytes per pixel, B first)
/// * `w` - Image width in pixels (should be even)
/// * `h` - Image height in pixels (should be even)
///
/// # Returns
///
/// YUV420P planar image data.
///
/// # Errors
///
/// Returns an error if the source buffer is too small or dimensions are zero.
pub fn bgr_to_yuv420p(src: &[u8], w: usize, h: usize) -> CvResult<Vec<u8>> {
    if w == 0 || h == 0 {
        return Err(CvError::invalid_dimensions(w as u32, h as u32));
    }
    let expected = w * h * 3;
    if src.len() < expected {
        return Err(CvError::insufficient_data(expected, src.len()));
    }

    let y_size = w * h;
    let uv_w = w / 2;
    let uv_h = h / 2;
    let uv_size = uv_w * uv_h;
    let mut out = vec![0u8; y_size + 2 * uv_size];

    // Y plane (full resolution)
    for row in 0..h {
        for col in 0..w {
            let off = (row * w + col) * 3;
            let b = src[off] as f32;
            let g = src[off + 1] as f32;
            let r = src[off + 2] as f32;
            out[row * w + col] =
                (0.257 * r + 0.504 * g + 0.098 * b + 16.0).clamp(16.0, 235.0) as u8;
        }
    }

    // U (Cb) and V (Cr) planes — 4:2:0 subsampling: one sample per 2×2 block.
    // We sample the top-left pixel of each 2×2 block (simple subsampling).
    let u_base = y_size;
    let v_base = y_size + uv_size;
    for row in 0..uv_h {
        for col in 0..uv_w {
            let src_row = row * 2;
            let src_col = col * 2;
            let off = (src_row * w + src_col) * 3;
            let b = src[off] as f32;
            let g = src[off + 1] as f32;
            let r = src[off + 2] as f32;
            out[u_base + row * uv_w + col] =
                (-0.148 * r - 0.291 * g + 0.439 * b + 128.0).clamp(16.0, 240.0) as u8;
            out[v_base + row * uv_w + col] =
                (0.439 * r - 0.368 * g - 0.071 * b + 128.0).clamp(16.0, 240.0) as u8;
        }
    }

    Ok(out)
}

/// Validate image dimensions and data size.
fn validate_dimensions(data: &[u8], width: u32, height: u32, channels: usize) -> CvResult<()> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = width as usize * height as usize * channels;
    if data.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, data.len()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_channels() {
        assert_eq!(ColorSpace::Rgb.channels(), 3);
        assert_eq!(ColorSpace::Bgr.channels(), 3);
        assert_eq!(ColorSpace::Grayscale.channels(), 1);
        assert_eq!(ColorSpace::YuvBt601.channels(), 3);
    }

    #[test]
    fn test_color_space_luminance() {
        assert!(ColorSpace::YuvBt601.is_luminance_based());
        assert!(ColorSpace::YuvBt709.is_luminance_based());
        assert!(ColorSpace::Lab.is_luminance_based());
        assert!(ColorSpace::Grayscale.is_luminance_based());
        assert!(!ColorSpace::Rgb.is_luminance_based());
        assert!(!ColorSpace::Hsv.is_luminance_based());
    }

    #[test]
    fn test_rgb_bgr_roundtrip() {
        let src = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // R, G, B pixels
        let bgr = rgb_to_bgr(&src, 3, 1).expect("rgb_to_bgr should succeed");
        let rgb = bgr_to_rgb(&bgr, 3, 1).expect("bgr_to_rgb should succeed");
        assert_eq!(src, rgb);
    }

    #[test]
    fn test_rgb_yuv_bt601_roundtrip() {
        let src = vec![128, 128, 128]; // Gray pixel
        let yuv =
            rgb_to_yuv(&src, 1, 1, YuvCoefficients::BT601).expect("rgb_to_yuv should succeed");
        let rgb =
            yuv_to_rgb(&yuv, 1, 1, YuvCoefficients::BT601).expect("yuv_to_rgb should succeed");

        // Allow small rounding errors
        for i in 0..3 {
            assert!((src[i] as i32 - rgb[i] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let src = vec![200, 100, 50]; // Orange-ish color
        let hsv = rgb_to_hsv(&src, 1, 1).expect("rgb_to_hsv should succeed");
        let rgb = hsv_to_rgb(&hsv, 1, 1).expect("hsv_to_rgb should succeed");

        // Allow small rounding errors
        for i in 0..3 {
            assert!((src[i] as i32 - rgb[i] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_rgb_hsl_roundtrip() {
        let src = vec![100, 150, 200]; // Light blue
        let hsl = rgb_to_hsl(&src, 1, 1).expect("rgb_to_hsl should succeed");
        let rgb = hsl_to_rgb(&hsl, 1, 1).expect("hsl_to_rgb should succeed");

        // Allow small rounding errors
        for i in 0..3 {
            assert!((src[i] as i32 - rgb[i] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_rgb_grayscale() {
        let src = vec![100, 150, 200]; // Light blue
        let gray = rgb_to_grayscale(&src, 1, 1).expect("rgb_to_grayscale should succeed");
        assert_eq!(gray.len(), 1);

        // BT.709: 0.2126 * 100 + 0.7152 * 150 + 0.0722 * 200 = 142.88
        assert!((gray[0] as i32 - 143).abs() <= 1);
    }

    #[test]
    fn test_grayscale_to_rgb() {
        let src = vec![128];
        let rgb = grayscale_to_rgb(&src, 1, 1).expect("grayscale_to_rgb should succeed");
        assert_eq!(rgb, vec![128, 128, 128]);
    }

    #[test]
    fn test_convert_color_space_same() {
        let src = vec![100, 150, 200];
        let result = convert_color_space(&src, 1, 1, ColorSpace::Rgb, ColorSpace::Rgb)
            .expect("convert_color_space should succeed");
        assert_eq!(src, result);
    }

    #[test]
    fn test_invalid_dimensions() {
        let src = vec![0u8; 12];
        assert!(rgb_to_bgr(&src, 0, 4,).is_err());
        assert!(rgb_to_bgr(&src, 4, 0).is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let src = vec![0u8; 6]; // Only 2 pixels worth of data
        assert!(rgb_to_bgr(&src, 3, 1).is_err()); // Needs 9 bytes
    }
}
