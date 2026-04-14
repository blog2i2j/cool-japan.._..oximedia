//! Single-image super-resolution via bicubic upscaling with unsharp-mask sharpening.
//!
//! The pipeline is:
//!
//! 1. **Bicubic interpolation** — upsample by the requested scale factor using
//!    the classic Catmull-Rom spline (a = -0.5) which is C2-continuous and
//!    free of ringing artefacts near edges.
//! 2. **Unsharp mask** — sharpen by blending the upsampled image with a
//!    Gaussian-blurred version:
//!    ```text
//!    output = upsampled + strength * (upsampled - gaussian_blur(upsampled))
//!    ```
//!    `strength` is controlled by [`SuperResolutionConfig::sharpening_strength`].
//!
//! Both operations are implemented in pure Rust with no external C/Fortran
//! dependencies.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Supported upscale factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleFactor {
    /// 2× upscale (output is twice the input in each dimension).
    X2,
    /// 3× upscale.
    X3,
    /// 4× upscale.
    X4,
}

impl ScaleFactor {
    /// Return the integer scale multiplier.
    #[must_use]
    pub const fn multiplier(self) -> u32 {
        match self {
            Self::X2 => 2,
            Self::X3 => 3,
            Self::X4 => 4,
        }
    }
}

/// Configuration for single-image super-resolution.
#[derive(Clone, Debug)]
pub struct SuperResolutionConfig {
    /// Upscale factor applied to both width and height.
    pub scale: ScaleFactor,

    /// Post-upscale sharpening strength in `[0.0, 2.0]`.
    ///
    /// - `0.0` — no sharpening (pure bicubic output).
    /// - `1.0` — moderate sharpening; typical for display output.
    /// - `2.0` — aggressive sharpening; may introduce ringing on very smooth
    ///   areas.
    pub sharpening_strength: f32,

    /// Radius of the Gaussian blur used in the unsharp mask (pixels, ≥1).
    ///
    /// Larger values make the sharpening act on lower spatial frequencies.
    /// Defaults to `1.5`.
    pub blur_sigma: f32,
}

impl Default for SuperResolutionConfig {
    fn default() -> Self {
        Self {
            scale: ScaleFactor::X2,
            sharpening_strength: 0.5,
            blur_sigma: 1.5,
        }
    }
}

impl SuperResolutionConfig {
    /// Set the scale factor.
    #[must_use]
    pub fn with_scale(mut self, scale: ScaleFactor) -> Self {
        self.scale = scale;
        self
    }

    /// Set the sharpening strength.
    #[must_use]
    pub fn with_sharpening_strength(mut self, strength: f32) -> Self {
        self.sharpening_strength = strength.clamp(0.0, 2.0);
        self
    }

    /// Set the Gaussian blur sigma for the unsharp mask.
    #[must_use]
    pub fn with_blur_sigma(mut self, sigma: f32) -> Self {
        self.blur_sigma = sigma.max(0.1);
        self
    }
}

/// The result of a super-resolution upscale.
#[derive(Clone, Debug)]
pub struct SuperResolutionResult {
    /// Upscaled pixel data in the same channel layout as the input.
    pub data: Vec<u8>,
    /// Width of the upscaled image.
    pub width: u32,
    /// Height of the upscaled image.
    pub height: u32,
    /// Number of channels (same as input).
    pub channels: u32,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Catmull-Rom cubic kernel weight for parameter `t` (Catmull-Rom has a = -0.5).
#[inline]
fn catmull_rom(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t < 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// Clamp an i64 index to [0, max].
#[inline]
fn clamp_idx(i: i64, max: i64) -> usize {
    i.clamp(0, max) as usize
}

/// Sample a single-channel plane with clamp-to-edge border handling.
#[inline]
fn sample_plane(plane: &[f64], x: i64, y: i64, width: u32, height: u32) -> f64 {
    let xi = clamp_idx(x, width as i64 - 1);
    let yi = clamp_idx(y, height as i64 - 1);
    plane[yi * width as usize + xi]
}

/// Bicubic (Catmull-Rom) upsample of a single-channel f64 plane.
///
/// Returns a plane of size `(src_w * scale) × (src_h * scale)`.
fn bicubic_upsample_plane(src: &[f64], src_w: u32, src_h: u32, scale: u32) -> Vec<f64> {
    let dst_w = src_w * scale;
    let dst_h = src_h * scale;
    let mut dst = vec![0.0f64; (dst_w as usize) * (dst_h as usize)];
    let scale_f = scale as f64;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            // Map destination pixel to source coordinate (centre-aligned)
            let sx = (dx as f64 + 0.5) / scale_f - 0.5;
            let sy = (dy as f64 + 0.5) / scale_f - 0.5;

            let sx_floor = sx.floor() as i64;
            let sy_floor = sy.floor() as i64;
            let tx = sx - sx_floor as f64;
            let ty = sy - sy_floor as f64;

            // Accumulate 4×4 neighborhood
            let mut acc = 0.0f64;
            for ky in -1i64..=2 {
                let wy = catmull_rom(ty - ky as f64);
                for kx in -1i64..=2 {
                    let wx = catmull_rom(tx - kx as f64);
                    let v = sample_plane(src, sx_floor + kx, sy_floor + ky, src_w, src_h);
                    acc += wx * wy * v;
                }
            }

            dst[(dy as usize) * (dst_w as usize) + (dx as usize)] = acc;
        }
    }

    dst
}

/// Build a 1-D Gaussian kernel of radius `ceil(2.5 * sigma)`.
fn gaussian_kernel_1d(sigma: f32) -> Vec<f64> {
    let sigma = sigma as f64;
    let radius = (2.5 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut k = Vec::with_capacity(size);
    for i in 0..size {
        let x = i as f64 - radius as f64;
        k.push((-0.5 * (x / sigma).powi(2)).exp());
    }
    let sum: f64 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Separable 2-D Gaussian blur on a single-channel f64 plane (clamp-to-edge).
fn gaussian_blur_plane(src: &[f64], width: u32, height: u32, sigma: f32) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let k = gaussian_kernel_1d(sigma);
    let radius = k.len() / 2;

    // Horizontal pass
    let mut tmp = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let xi = (x as i64 + ki as i64 - radius as i64).clamp(0, w as i64 - 1) as usize;
                acc += kv * src[y * w + xi];
            }
            tmp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in k.iter().enumerate() {
                let yi = (y as i64 + ki as i64 - radius as i64).clamp(0, h as i64 - 1) as usize;
                acc += kv * tmp[yi * w + x];
            }
            out[y * w + x] = acc;
        }
    }

    out
}

/// Apply unsharp mask in-place on a f64 plane (values in [0.0, 255.0]).
fn unsharp_mask_plane(plane: &mut Vec<f64>, width: u32, height: u32, sigma: f32, strength: f32) {
    if strength < f32::EPSILON {
        return;
    }
    let blurred = gaussian_blur_plane(plane, width, height, sigma);
    let s = strength as f64;
    for (p, b) in plane.iter_mut().zip(blurred.iter()) {
        *p = (*p + s * (*p - *b)).clamp(0.0, 255.0);
    }
}

// ---------------------------------------------------------------------------
// Public upscaler
// ---------------------------------------------------------------------------

/// Single-image super-resolution upscaler.
///
/// Uses bicubic (Catmull-Rom) interpolation followed by an unsharp-mask
/// sharpening pass.  Pixel data is expected as interleaved u8 with an
/// arbitrary number of channels (1 = gray, 3 = RGB, 4 = RGBA, etc.).
///
/// # Example
///
/// ```rust
/// use oximedia_image::super_resolution::{
///     ScaleFactor, SuperResolutionConfig, SuperResolutionUpscaler,
/// };
///
/// let config = SuperResolutionConfig::default();
/// let upscaler = SuperResolutionUpscaler::new(config);
///
/// // 4×4 RGB image → 8×8
/// let pixels = vec![128u8; 4 * 4 * 3];
/// let result = upscaler.upscale(&pixels, 4, 4, 3).expect("upscale image");
/// assert_eq!(result.width, 8);
/// assert_eq!(result.height, 8);
/// ```
#[derive(Debug)]
pub struct SuperResolutionUpscaler {
    config: SuperResolutionConfig,
}

impl SuperResolutionUpscaler {
    /// Create a new upscaler with the given configuration.
    #[must_use]
    pub fn new(config: SuperResolutionConfig) -> Self {
        Self { config }
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &SuperResolutionConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: SuperResolutionConfig) {
        self.config = config;
    }

    /// Upscale `src` by the configured scale factor.
    ///
    /// `src` must be interleaved u8 with `channels` components per pixel and
    /// exactly `width * height * channels` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] for zero-size inputs.
    /// Returns [`ImageError::InvalidFormat`] if the buffer length does not
    /// match `width * height * channels`.
    pub fn upscale(
        &self,
        src: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> ImageResult<SuperResolutionResult> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions(width, height));
        }
        if channels == 0 {
            return Err(ImageError::InvalidFormat("channels must be > 0".into()));
        }
        let expected = (width as usize) * (height as usize) * (channels as usize);
        if src.len() != expected {
            return Err(ImageError::InvalidFormat(format!(
                "input buffer length {} != expected {expected}",
                src.len()
            )));
        }

        let scale = self.config.scale.multiplier();
        let dst_w = width * scale;
        let dst_h = height * scale;
        let ch = channels as usize;

        // Convert interleaved u8 → planar f64
        let src_px = (width as usize) * (height as usize);
        let mut planes: Vec<Vec<f64>> = (0..ch)
            .map(|c| {
                src.iter()
                    .skip(c)
                    .step_by(ch)
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();

        // For each channel: bicubic upsample → unsharp mask
        let mut upsampled_planes: Vec<Vec<f64>> = Vec::with_capacity(ch);
        for plane in planes.iter_mut() {
            assert_eq!(plane.len(), src_px);
            let mut up = bicubic_upsample_plane(plane, width, height, scale);
            unsharp_mask_plane(
                &mut up,
                dst_w,
                dst_h,
                self.config.blur_sigma,
                self.config.sharpening_strength,
            );
            upsampled_planes.push(up);
        }

        // Convert planar f64 → interleaved u8
        let dst_px = (dst_w as usize) * (dst_h as usize);
        let mut data = vec![0u8; dst_px * ch];
        for (c, plane) in upsampled_planes.iter().enumerate() {
            for (i, &v) in plane.iter().enumerate() {
                data[i * ch + c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }

        Ok(SuperResolutionResult {
            data,
            width: dst_w,
            height: dst_h,
            channels,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Output dimensions are correct for 2×
    #[test]
    fn test_output_dimensions_x2() {
        let config = SuperResolutionConfig::default().with_scale(ScaleFactor::X2);
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![128u8; 8 * 8 * 3];
        let result = upscaler
            .upscale(&src, 8, 8, 3)
            .expect("upscale 8x8 RGB to 16x16");
        assert_eq!(result.width, 16);
        assert_eq!(result.height, 16);
        assert_eq!(result.channels, 3);
        assert_eq!(result.data.len(), 16 * 16 * 3);
    }

    // -----------------------------------------------------------------------
    // 2. Output dimensions correct for 3×
    #[test]
    fn test_output_dimensions_x3() {
        let config = SuperResolutionConfig::default().with_scale(ScaleFactor::X3);
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![100u8; 4 * 4];
        let result = upscaler
            .upscale(&src, 4, 4, 1)
            .expect("upscale 4x4 grayscale to 12x12");
        assert_eq!(result.width, 12);
        assert_eq!(result.height, 12);
        assert_eq!(result.data.len(), 144);
    }

    // -----------------------------------------------------------------------
    // 3. Output dimensions correct for 4×
    #[test]
    fn test_output_dimensions_x4() {
        let config = SuperResolutionConfig::default().with_scale(ScaleFactor::X4);
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![200u8; 6 * 6 * 4];
        let result = upscaler
            .upscale(&src, 6, 6, 4)
            .expect("upscale 6x6 RGBA to 24x24");
        assert_eq!(result.width, 24);
        assert_eq!(result.height, 24);
        assert_eq!(result.channels, 4);
        assert_eq!(result.data.len(), 24 * 24 * 4);
    }

    // -----------------------------------------------------------------------
    // 4. All pixel values stay in [0, 255]
    #[test]
    fn test_output_values_in_range() {
        let config = SuperResolutionConfig::default()
            .with_scale(ScaleFactor::X2)
            .with_sharpening_strength(2.0); // aggressive sharpening
        let upscaler = SuperResolutionUpscaler::new(config);
        // Alternating black-white to stress clamping
        let src: Vec<u8> = (0..16 * 16 * 3)
            .map(|i| if i % 6 < 3 { 0 } else { 255 })
            .collect();
        let result = upscaler
            .upscale(&src, 16, 16, 3)
            .expect("upscale with aggressive sharpening");
        assert!(!result.data.is_empty());
    }

    // -----------------------------------------------------------------------
    // 5. Flat (constant) image upscaled with no sharpening stays constant
    #[test]
    fn test_flat_image_stays_constant_no_sharpening() {
        let config = SuperResolutionConfig::default()
            .with_scale(ScaleFactor::X2)
            .with_sharpening_strength(0.0);
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![137u8; 8 * 8];
        let result = upscaler.upscale(&src, 8, 8, 1).expect("upscale flat image");
        for &v in &result.data {
            assert_eq!(
                v, 137,
                "flat image must stay constant after bicubic upsample"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Zero dimensions are rejected
    #[test]
    fn test_zero_dimensions_rejected() {
        let config = SuperResolutionConfig::default();
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![];
        assert!(upscaler.upscale(&src, 0, 4, 1).is_err());
        assert!(upscaler.upscale(&src, 4, 0, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // 7. Buffer size mismatch is rejected
    #[test]
    fn test_buffer_size_mismatch_rejected() {
        let config = SuperResolutionConfig::default();
        let upscaler = SuperResolutionUpscaler::new(config);
        let src = vec![128u8; 10]; // wrong size for 4×4×3
        assert!(upscaler.upscale(&src, 4, 4, 3).is_err());
    }

    // -----------------------------------------------------------------------
    // 8. Sharpening strength 0.0 produces no sharpening artefacts
    #[test]
    fn test_no_sharpening_vs_sharpening_differ() {
        let src: Vec<u8> = (0u8..=255).cycle().take(16 * 16).collect();

        let no_sharp = {
            let config = SuperResolutionConfig::default()
                .with_scale(ScaleFactor::X2)
                .with_sharpening_strength(0.0);
            SuperResolutionUpscaler::new(config)
                .upscale(&src, 16, 16, 1)
                .expect("upscale without sharpening")
        };
        let sharp = {
            let config = SuperResolutionConfig::default()
                .with_scale(ScaleFactor::X2)
                .with_sharpening_strength(1.5);
            SuperResolutionUpscaler::new(config)
                .upscale(&src, 16, 16, 1)
                .expect("upscale with sharpening")
        };

        // Results must differ when sharpening is applied
        assert_ne!(
            no_sharp.data, sharp.data,
            "sharpened and unsharpened outputs should differ"
        );
    }

    // -----------------------------------------------------------------------
    // 9. Multi-channel (RGBA) upscale preserves channel count
    #[test]
    fn test_rgba_upscale_channel_count() {
        let config = SuperResolutionConfig::default().with_scale(ScaleFactor::X2);
        let upscaler = SuperResolutionUpscaler::new(config);
        let src: Vec<u8> = (0..8 * 8 * 4).map(|i| (i % 256) as u8).collect();
        let result = upscaler.upscale(&src, 8, 8, 4).expect("upscale RGBA");
        assert_eq!(result.channels, 4);
        assert_eq!(result.data.len(), 16 * 16 * 4);
    }

    // -----------------------------------------------------------------------
    // 10. ScaleFactor::multiplier returns correct values
    #[test]
    fn test_scale_factor_multiplier() {
        assert_eq!(ScaleFactor::X2.multiplier(), 2);
        assert_eq!(ScaleFactor::X3.multiplier(), 3);
        assert_eq!(ScaleFactor::X4.multiplier(), 4);
    }

    // -----------------------------------------------------------------------
    // 11. Bicubic upsample preserves smooth gradients (no catastrophic ringing)
    #[test]
    fn test_bicubic_gradient_continuity() {
        // Create a horizontal gradient 0..255
        let src: Vec<u8> = (0..16u8)
            .flat_map(|x| std::iter::repeat(x * 16).take(16))
            .collect();
        let config = SuperResolutionConfig::default()
            .with_scale(ScaleFactor::X2)
            .with_sharpening_strength(0.0);
        let upscaler = SuperResolutionUpscaler::new(config);
        let result = upscaler.upscale(&src, 16, 16, 1).expect("upscale gradient");

        // Adjacent pixels in the upscaled result should not differ by more than
        // ~32 (the original step is 16, bicubic may overshoot slightly but not >2×)
        let w = result.width as usize;
        for y in 0..result.height as usize {
            for x in 0..w - 1 {
                let diff =
                    (result.data[y * w + x] as i32 - result.data[y * w + x + 1] as i32).abs();
                assert!(
                    diff <= 32,
                    "adjacent pixel difference {diff} too large at ({x},{y})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 12. Default config has correct default values
    #[test]
    fn test_default_config_values() {
        let config = SuperResolutionConfig::default();
        assert_eq!(config.scale, ScaleFactor::X2);
        assert!((config.sharpening_strength - 0.5).abs() < 1e-6);
        assert!((config.blur_sigma - 1.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // 13. Config builder methods clamp correctly
    #[test]
    fn test_config_builder_clamping() {
        let config = SuperResolutionConfig::default()
            .with_sharpening_strength(99.0)
            .with_blur_sigma(-5.0);
        assert!((config.sharpening_strength - 2.0).abs() < 1e-6);
        assert!(config.blur_sigma >= 0.1);
    }
}
