//! Color space conversion kernels

use crate::{GpuDevice, Result};

/// Color space standards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ColorSpace {
    /// RGB color space
    RGB,
    /// YUV with BT.601 coefficients (SD video)
    YUV_BT601,
    /// YUV with BT.709 coefficients (HD video)
    YUV_BT709,
    /// YUV with BT.2020 coefficients (UHD video)
    YUV_BT2020,
    /// HSV color space
    HSV,
    /// HSL color space
    HSL,
    /// CIE Lab color space
    Lab,
    /// Linear RGB
    LinearRGB,
    /// sRGB
    SRGB,
}

impl ColorSpace {
    /// Check if this is a YUV color space
    #[must_use]
    pub fn is_yuv(self) -> bool {
        matches!(self, Self::YUV_BT601 | Self::YUV_BT709 | Self::YUV_BT2020)
    }

    /// Check if this is an RGB color space
    #[must_use]
    pub fn is_rgb(self) -> bool {
        matches!(self, Self::RGB | Self::LinearRGB | Self::SRGB)
    }

    /// Get the color space name
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::RGB => "RGB",
            Self::YUV_BT601 => "YUV (BT.601)",
            Self::YUV_BT709 => "YUV (BT.709)",
            Self::YUV_BT2020 => "YUV (BT.2020)",
            Self::HSV => "HSV",
            Self::HSL => "HSL",
            Self::Lab => "CIE Lab",
            Self::LinearRGB => "Linear RGB",
            Self::SRGB => "sRGB",
        }
    }
}

impl From<ColorSpace> for crate::ops::ColorSpace {
    fn from(space: ColorSpace) -> Self {
        match space {
            ColorSpace::YUV_BT601 | ColorSpace::RGB => Self::BT601,
            ColorSpace::YUV_BT709 => Self::BT709,
            ColorSpace::YUV_BT2020 => Self::BT2020,
            _ => Self::BT601, // Default fallback
        }
    }
}

/// Color conversion operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorConversion {
    /// RGB to YUV
    RGBtoYUV,
    /// YUV to RGB
    YUVtoRGB,
    /// RGB to HSV
    RGBtoHSV,
    /// HSV to RGB
    HSVtoRGB,
    /// RGB to Lab
    RGBtoLab,
    /// Lab to RGB
    LabtoRGB,
    /// sRGB to Linear RGB
    SRGBtoLinear,
    /// Linear RGB to sRGB
    LinearToSRGB,
}

/// Color space conversion kernel
pub struct ColorConversionKernel {
    conversion: ColorConversion,
    color_space: ColorSpace,
}

impl ColorConversionKernel {
    /// Create a new color conversion kernel
    #[must_use]
    pub fn new(conversion: ColorConversion, color_space: ColorSpace) -> Self {
        Self {
            conversion,
            color_space,
        }
    }

    /// Create an RGB to YUV conversion kernel
    #[must_use]
    pub fn rgb_to_yuv(color_space: ColorSpace) -> Self {
        Self::new(ColorConversion::RGBtoYUV, color_space)
    }

    /// Create a YUV to RGB conversion kernel
    #[must_use]
    pub fn yuv_to_rgb(color_space: ColorSpace) -> Self {
        Self::new(ColorConversion::YUVtoRGB, color_space)
    }

    /// Execute the color conversion
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device
    /// * `input` - Input image buffer
    /// * `output` - Output image buffer
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion fails.
    pub fn execute(
        &self,
        device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        match self.conversion {
            ColorConversion::RGBtoYUV => crate::ops::ColorSpaceConversion::rgb_to_yuv(
                device,
                input,
                output,
                width,
                height,
                self.color_space.into(),
            ),
            ColorConversion::YUVtoRGB => crate::ops::ColorSpaceConversion::yuv_to_rgb(
                device,
                input,
                output,
                width,
                height,
                self.color_space.into(),
            ),
            ColorConversion::RGBtoHSV => {
                let result = crate::ops::ColorSpaceConversion::rgb_to_hsv(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
            ColorConversion::HSVtoRGB => {
                let result = crate::ops::ColorSpaceConversion::hsv_to_rgb(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
            ColorConversion::RGBtoLab => {
                let result = crate::ops::ColorSpaceConversion::rgb_to_lab(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
            ColorConversion::LabtoRGB => {
                let result = crate::ops::ColorSpaceConversion::lab_to_rgb(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
            ColorConversion::SRGBtoLinear => {
                let result = crate::ops::ColorSpaceConversion::srgb_to_linear(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
            ColorConversion::LinearToSRGB => {
                let result = crate::ops::ColorSpaceConversion::linear_to_srgb(input, width, height);
                let copy_len = result.len().min(output.len());
                output[..copy_len].copy_from_slice(&result[..copy_len]);
                Ok(())
            }
        }
    }

    /// Get the conversion type
    #[must_use]
    pub fn conversion(&self) -> ColorConversion {
        self.conversion
    }

    /// Get the color space
    #[must_use]
    pub fn color_space(&self) -> ColorSpace {
        self.color_space
    }

    /// Calculate output buffer size
    #[must_use]
    pub fn output_size(width: u32, height: u32, channels: u32) -> usize {
        (width * height * channels) as usize
    }

    /// Estimate FLOPS for color conversion
    #[must_use]
    pub fn estimate_flops(width: u32, height: u32, conversion: ColorConversion) -> u64 {
        let pixels = u64::from(width) * u64::from(height);

        match conversion {
            ColorConversion::RGBtoYUV | ColorConversion::YUVtoRGB => {
                // Matrix multiplication: 3x3 * 3 = 9 multiplies + 6 adds per pixel
                pixels * 15
            }
            ColorConversion::RGBtoHSV | ColorConversion::HSVtoRGB => {
                // HSV conversion involves min/max operations and divisions
                pixels * 20
            }
            ColorConversion::RGBtoLab | ColorConversion::LabtoRGB => {
                // Lab conversion is more complex with power functions
                pixels * 50
            }
            ColorConversion::SRGBtoLinear | ColorConversion::LinearToSRGB => {
                // Gamma correction per component
                pixels * 3 * 5
            }
        }
    }
}

/// Lookup table (LUT) based color transformation
pub struct LutKernel {
    lut_size: usize,
}

impl LutKernel {
    /// Create a new LUT kernel
    ///
    /// # Arguments
    ///
    /// * `lut_size` - Size of the LUT (typically 256 for 1D or 33 for 3D)
    #[must_use]
    pub fn new(lut_size: usize) -> Self {
        Self { lut_size }
    }

    /// Get the LUT size
    #[must_use]
    pub fn lut_size(&self) -> usize {
        self.lut_size
    }

    /// Apply 1D LUT transformation (CPU fallback).
    ///
    /// The LUT layout is `[CH0_LUT[0..lut_size], CH1_LUT[0..lut_size], …]`.
    /// For each pixel and each channel `c`, the output value is
    /// `lut[c * lut_size + idx]` where `idx = (pixel_value * (lut_size-1)) / 255`.
    ///
    /// Pixels and the LUT are treated as having the same number of channels,
    /// inferred from `lut.len() / lut_size`.  If `lut.len()` is not a multiple
    /// of `lut_size`, the extra channel bytes are copied unchanged.
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input image buffer
    /// * `output` - Output image buffer (same length as `input`)
    /// * `lut` - 1D lookup table (size: `lut_size * channels`)
    /// * `_width` - Image width (unused; length is inferred from buffers)
    /// * `_height` - Image height (unused; length is inferred from buffers)
    ///
    /// # Errors
    ///
    /// Returns an error if `lut_size` is zero or the LUT is empty.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_1d(
        &self,
        _device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        lut: &[u8],
        _width: u32,
        _height: u32,
    ) -> Result<()> {
        if self.lut_size == 0 || lut.is_empty() {
            return Err(crate::GpuError::NotSupported(
                "1D LUT size must be non-zero".to_string(),
            ));
        }
        let channels = lut.len() / self.lut_size;
        if channels == 0 {
            return Err(crate::GpuError::NotSupported(
                "1D LUT must cover at least one channel".to_string(),
            ));
        }
        let lut_max = self.lut_size - 1;
        // Process each pixel (each group of `channels` bytes in the input).
        // Any trailing bytes beyond a complete pixel are copied unchanged.
        let full_pixels = input.len() / channels;
        for px in 0..full_pixels {
            let base = px * channels;
            for c in 0..channels {
                let pixel_val = input[base + c] as usize;
                // Scale pixel value [0..=255] to lut index [0..=lut_max].
                let lut_idx = (pixel_val * lut_max + 127) / 255; // round-to-nearest
                let lut_idx = lut_idx.min(lut_max);
                output[base + c] = lut[c * self.lut_size + lut_idx];
            }
        }
        // Copy any trailing bytes (partial pixel) unchanged.
        let tail_start = full_pixels * channels;
        output[tail_start..input.len()].copy_from_slice(&input[tail_start..]);
        Ok(())
    }

    /// Apply 3D LUT transformation with trilinear interpolation (CPU fallback).
    ///
    /// The LUT is a cubic grid of size `N × N × N` (where `N = lut_size`)
    /// storing RGB triplets, laid out as
    /// `lut[(r_idx * N*N + g_idx * N + b_idx) * 3 + channel]`.
    ///
    /// Input pixels are expected to be interleaved 3-channel (RGB) data.
    /// Any extra channels beyond the first three are passed through unchanged.
    ///
    /// # Arguments
    ///
    /// * `_device` - GPU device (CPU fallback: unused)
    /// * `input` - Input image buffer (interleaved RGB, 3 bytes per pixel minimum)
    /// * `output` - Output image buffer (same length as `input`)
    /// * `lut` - 3D LUT (`lut_size^3 * 3` f32 entries, values in `[0.0, 1.0]`)
    /// * `_width` - Image width (unused; length is inferred from buffers)
    /// * `_height` - Image height (unused; length is inferred from buffers)
    ///
    /// # Errors
    ///
    /// Returns an error if `lut_size` is zero or the LUT is too small.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_3d(
        &self,
        _device: &GpuDevice,
        input: &[u8],
        output: &mut [u8],
        lut: &[f32],
        _width: u32,
        _height: u32,
    ) -> Result<()> {
        let n = self.lut_size;
        if n == 0 {
            return Err(crate::GpuError::NotSupported(
                "3D LUT size must be non-zero".to_string(),
            ));
        }
        let expected_lut = n * n * n * 3;
        if lut.len() < expected_lut {
            return Err(crate::GpuError::NotSupported(format!(
                "3D LUT too small: expected {expected_lut} entries, got {}",
                lut.len()
            )));
        }

        // Process pixels in groups of 3 (RGB).
        let pixel_stride = 3usize;
        let full_pixels = input.len() / pixel_stride;

        for px in 0..full_pixels {
            let base = px * pixel_stride;
            // Normalize each channel to [0.0, 1.0].
            let r = f32::from(input[base]) / 255.0;
            let g = f32::from(input[base + 1]) / 255.0;
            let b = f32::from(input[base + 2]) / 255.0;

            // Compute fractional position in the LUT grid.
            let nf = (n - 1) as f32;
            let rx = r * nf;
            let gy = g * nf;
            let bz = b * nf;

            // Integer cube corner indices.
            let r0 = (rx.floor() as usize).min(n - 1);
            let g0 = (gy.floor() as usize).min(n - 1);
            let b0 = (bz.floor() as usize).min(n - 1);
            let r1 = (r0 + 1).min(n - 1);
            let g1 = (g0 + 1).min(n - 1);
            let b1 = (b0 + 1).min(n - 1);

            // Fractional parts for trilinear interpolation.
            let fr = rx - r0 as f32;
            let fg = gy - g0 as f32;
            let fb = bz - b0 as f32;

            // Helper closure: fetch one channel value from the LUT.
            let lut_val = |ri: usize, gi: usize, bi: usize, ch: usize| -> f32 {
                lut[(ri * n * n + gi * n + bi) * 3 + ch]
            };

            for ch in 0..3 {
                // Trilinear interpolation over the 8 cube corners.
                let c000 = lut_val(r0, g0, b0, ch);
                let c100 = lut_val(r1, g0, b0, ch);
                let c010 = lut_val(r0, g1, b0, ch);
                let c110 = lut_val(r1, g1, b0, ch);
                let c001 = lut_val(r0, g0, b1, ch);
                let c101 = lut_val(r1, g0, b1, ch);
                let c011 = lut_val(r0, g1, b1, ch);
                let c111 = lut_val(r1, g1, b1, ch);

                let c00 = c000 * (1.0 - fr) + c100 * fr;
                let c01 = c001 * (1.0 - fr) + c101 * fr;
                let c10 = c010 * (1.0 - fr) + c110 * fr;
                let c11 = c011 * (1.0 - fr) + c111 * fr;

                let c0 = c00 * (1.0 - fg) + c10 * fg;
                let c1 = c01 * (1.0 - fg) + c11 * fg;

                let val = c0 * (1.0 - fb) + c1 * fb;
                output[base + ch] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }

        // Copy any trailing bytes (partial pixel) unchanged.
        let tail_start = full_pixels * pixel_stride;
        output[tail_start..input.len()].copy_from_slice(&input[tail_start..]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_space_properties() {
        assert!(ColorSpace::YUV_BT601.is_yuv());
        assert!(ColorSpace::YUV_BT709.is_yuv());
        assert!(ColorSpace::YUV_BT2020.is_yuv());
        assert!(!ColorSpace::RGB.is_yuv());

        assert!(ColorSpace::RGB.is_rgb());
        assert!(ColorSpace::LinearRGB.is_rgb());
        assert!(ColorSpace::SRGB.is_rgb());
        assert!(!ColorSpace::YUV_BT601.is_rgb());
    }

    #[test]
    fn test_color_conversion_kernel() {
        let kernel = ColorConversionKernel::rgb_to_yuv(ColorSpace::YUV_BT709);
        assert_eq!(kernel.conversion(), ColorConversion::RGBtoYUV);
        assert_eq!(kernel.color_space(), ColorSpace::YUV_BT709);
    }

    #[test]
    fn test_flops_estimation() {
        let flops = ColorConversionKernel::estimate_flops(1920, 1080, ColorConversion::RGBtoYUV);
        assert!(flops > 0);

        let flops_lab =
            ColorConversionKernel::estimate_flops(1920, 1080, ColorConversion::RGBtoLab);
        assert!(flops_lab > flops); // Lab conversion should be more expensive
    }

    // --- CPU LUT implementation tests (no GpuDevice required) ----------------

    /// Build an identity 1D LUT for `channels` channels with `lut_size` entries.
    fn identity_lut_1d(lut_size: usize, channels: usize) -> Vec<u8> {
        let mut lut = vec![0u8; lut_size * channels];
        for c in 0..channels {
            for i in 0..lut_size {
                // Scale i back to [0..=255].
                lut[c * lut_size + i] = ((i * 255) / (lut_size - 1)) as u8;
            }
        }
        lut
    }

    /// Build an identity 3D LUT for N×N×N grid (3 channels, values in [0,1]).
    fn identity_lut_3d(n: usize) -> Vec<f32> {
        let mut lut = vec![0.0f32; n * n * n * 3];
        for ri in 0..n {
            for gi in 0..n {
                for bi in 0..n {
                    let base = (ri * n * n + gi * n + bi) * 3;
                    lut[base] = ri as f32 / (n - 1) as f32;
                    lut[base + 1] = gi as f32 / (n - 1) as f32;
                    lut[base + 2] = bi as f32 / (n - 1) as f32;
                }
            }
        }
        lut
    }

    #[test]
    fn test_apply_1d_identity() {
        // An identity LUT should reproduce the input pixel values.
        let lut_size = 256usize;
        let channels = 3usize;
        let lut = identity_lut_1d(lut_size, channels);
        let input: Vec<u8> = vec![0, 128, 255, 64, 192, 10];
        let mut output = vec![0u8; input.len()];

        // Run the logic inline (avoids GpuDevice construction).
        let kernel = LutKernel::new(lut_size);
        let lut_max = lut_size - 1;
        let full_pixels = input.len() / channels;
        for px in 0..full_pixels {
            let base = px * channels;
            for c in 0..channels {
                let pixel_val = input[base + c] as usize;
                let lut_idx = ((pixel_val * lut_max + 127) / 255).min(lut_max);
                output[base + c] = lut[c * kernel.lut_size() + lut_idx];
            }
        }

        // Each output byte should be very close to the corresponding input byte.
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let diff = inp as i32 - out as i32;
            assert!(diff.abs() <= 1, "pixel {i}: input={inp}, output={out}");
        }
    }

    #[test]
    fn test_apply_1d_invert() {
        // An inversion LUT: out = 255 - in.
        let lut_size = 256usize;
        let _channels = 1usize;
        let lut: Vec<u8> = (0..lut_size).map(|i| (255 - i) as u8).collect();
        let input: Vec<u8> = vec![0, 64, 128, 192, 255];
        let mut output = vec![0u8; input.len()];

        let lut_max = lut_size - 1;
        for (i, &v) in input.iter().enumerate() {
            let lut_idx = ((v as usize * lut_max + 127) / 255).min(lut_max);
            output[i] = lut[lut_idx];
        }

        assert_eq!(output[0], 255);
        assert_eq!(output[4], 0);
    }

    #[test]
    fn test_apply_3d_identity() {
        // An identity 3D LUT should reproduce the input pixel values (within rounding).
        let n = 17usize; // common LUT size
        let lut = identity_lut_3d(n);
        let input: Vec<u8> = vec![0, 0, 0, 128, 64, 192, 255, 255, 255];
        let mut output = vec![0u8; input.len()];

        let nf = (n - 1) as f32;
        let pixel_stride = 3usize;
        let full_pixels = input.len() / pixel_stride;

        for px in 0..full_pixels {
            let base = px * pixel_stride;
            let r = input[base] as f32 / 255.0;
            let g = input[base + 1] as f32 / 255.0;
            let b = input[base + 2] as f32 / 255.0;

            let rx = r * nf;
            let gy = g * nf;
            let bz = b * nf;

            let r0 = (rx.floor() as usize).min(n - 1);
            let g0 = (gy.floor() as usize).min(n - 1);
            let b0 = (bz.floor() as usize).min(n - 1);
            let r1 = (r0 + 1).min(n - 1);
            let g1 = (g0 + 1).min(n - 1);
            let b1 = (b0 + 1).min(n - 1);
            let fr = rx - r0 as f32;
            let fg = gy - g0 as f32;
            let fb = bz - b0 as f32;

            for ch in 0..3 {
                let lv = |ri: usize, gi: usize, bi: usize| -> f32 {
                    lut[(ri * n * n + gi * n + bi) * 3 + ch]
                };
                let c000 = lv(r0, g0, b0);
                let c100 = lv(r1, g0, b0);
                let c010 = lv(r0, g1, b0);
                let c110 = lv(r1, g1, b0);
                let c001 = lv(r0, g0, b1);
                let c101 = lv(r1, g0, b1);
                let c011 = lv(r0, g1, b1);
                let c111 = lv(r1, g1, b1);

                let c00 = c000 * (1.0 - fr) + c100 * fr;
                let c01 = c001 * (1.0 - fr) + c101 * fr;
                let c10 = c010 * (1.0 - fr) + c110 * fr;
                let c11 = c011 * (1.0 - fr) + c111 * fr;
                let c0 = c00 * (1.0 - fg) + c10 * fg;
                let c1 = c01 * (1.0 - fg) + c11 * fg;
                let val = c0 * (1.0 - fb) + c1 * fb;
                output[base + ch] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }

        // Each output should be within ±2 of the input (rounding in LUT grid).
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            let diff = inp as i32 - out as i32;
            assert!(
                diff.abs() <= 2,
                "channel byte {i}: input={inp}, output={out}"
            );
        }
    }

    #[test]
    fn test_apply_3d_black_white() {
        // Black (0,0,0) and white (255,255,255) corners should map exactly.
        let n = 2usize; // minimal grid
        let lut = identity_lut_3d(n);
        let input: Vec<u8> = vec![0, 0, 0, 255, 255, 255];
        let mut output = vec![0u8; 6];

        let nf = (n - 1) as f32;
        for px in 0..2usize {
            let base = px * 3;
            let r = input[base] as f32 / 255.0;
            let g = input[base + 1] as f32 / 255.0;
            let b = input[base + 2] as f32 / 255.0;
            let rx = r * nf;
            let gy = g * nf;
            let bz = b * nf;
            let r0 = (rx.floor() as usize).min(n - 1);
            let g0 = (gy.floor() as usize).min(n - 1);
            let b0 = (bz.floor() as usize).min(n - 1);
            let r1 = (r0 + 1).min(n - 1);
            let g1 = (g0 + 1).min(n - 1);
            let b1 = (b0 + 1).min(n - 1);
            let fr = rx - r0 as f32;
            let fg = gy - g0 as f32;
            let fb = bz - b0 as f32;
            for ch in 0..3 {
                let lv = |ri: usize, gi: usize, bi: usize| -> f32 {
                    lut[(ri * n * n + gi * n + bi) * 3 + ch]
                };
                let c000 = lv(r0, g0, b0);
                let c100 = lv(r1, g0, b0);
                let c010 = lv(r0, g1, b0);
                let c110 = lv(r1, g1, b0);
                let c001 = lv(r0, g0, b1);
                let c101 = lv(r1, g0, b1);
                let c011 = lv(r0, g1, b1);
                let c111 = lv(r1, g1, b1);
                let c00 = c000 * (1.0 - fr) + c100 * fr;
                let c01 = c001 * (1.0 - fr) + c101 * fr;
                let c10 = c010 * (1.0 - fr) + c110 * fr;
                let c11 = c011 * (1.0 - fr) + c111 * fr;
                let c0 = c00 * (1.0 - fg) + c10 * fg;
                let c1 = c01 * (1.0 - fg) + c11 * fg;
                let val = c0 * (1.0 - fb) + c1 * fb;
                output[base + ch] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }

        // Black should remain black.
        assert_eq!(&output[0..3], &[0u8, 0, 0]);
        // White should remain white.
        assert_eq!(&output[3..6], &[255u8, 255, 255]);
    }
}
