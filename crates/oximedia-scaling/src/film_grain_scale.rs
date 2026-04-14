//! Film grain removal before scaling and re-synthesis at target resolution.
//!
//! Scaling film grain directly produces undesirable artefacts: the grain
//! either blurs out (downscaling) or magnifies into blocky noise (upscaling).
//! This module implements a three-stage approach:
//!
//! 1. **Grain estimation** — measure the noise standard deviation per
//!    luminance band using a high-frequency residual method.
//! 2. **Grain removal** — low-pass filter the input to obtain a clean
//!    (de-grained) base image.
//! 3. **Grain re-synthesis** — generate new grain at the target resolution
//!    matching the measured spatial frequency and amplitude profile.
//!
//! The synthesised grain uses a deterministic LCG so that repeated calls
//! with the same parameters produce the same grain pattern.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::film_grain_scale::{FilmGrainScaler, GrainProfile};
//!
//! let profile = GrainProfile::auto_detect(&vec![128u8; 64 * 64], 64, 64);
//! let scaler = FilmGrainScaler::new(profile);
//! let output = scaler.scale(&vec![128u8; 64 * 64], 64, 64, 32, 32)
//!     .expect("scale ok");
//! assert_eq!(output.len(), 32 * 32);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::ScalingError;

// ---------------------------------------------------------------------------
// GrainProfile
// ---------------------------------------------------------------------------

/// Measured grain profile for a source image.
///
/// Stores the estimated grain amplitude (standard deviation) and spatial
/// frequency parameter for three luminance zones (shadows, midtones, highlights).
#[derive(Debug, Clone)]
pub struct GrainProfile {
    /// Grain sigma for the shadow region (luma 0–85).
    pub shadow_sigma: f32,
    /// Grain sigma for the midtone region (luma 86–170).
    pub midtone_sigma: f32,
    /// Grain sigma for the highlight region (luma 171–255).
    pub highlight_sigma: f32,
    /// Spatial correlation parameter in [0, 1]. Higher = coarser grain.
    pub spatial_correlation: f32,
}

impl Default for GrainProfile {
    fn default() -> Self {
        Self {
            shadow_sigma: 0.04,
            midtone_sigma: 0.03,
            highlight_sigma: 0.02,
            spatial_correlation: 0.3,
        }
    }
}

impl GrainProfile {
    /// Estimate the grain profile from a grayscale (single-channel u8) image.
    ///
    /// Uses a simple high-frequency residual (image − box-blurred image) to
    /// measure per-zone noise standard deviation.
    #[must_use]
    pub fn auto_detect(pixels: &[u8], width: usize, height: usize) -> Self {
        if width < 3 || height < 3 || pixels.is_empty() {
            return Self::default();
        }

        // 3×3 box blur
        let blurred = box_blur_3x3(pixels, width, height);

        let mut shadow_sq = 0.0f64;
        let mut shadow_count = 0usize;
        let mut mid_sq = 0.0f64;
        let mut mid_count = 0usize;
        let mut hi_sq = 0.0f64;
        let mut hi_count = 0usize;

        for (i, (&orig, &blur)) in pixels.iter().zip(blurred.iter()).enumerate() {
            let luma = pixels[i];
            let diff = orig as f64 - blur as f64;
            let sq = diff * diff;
            match luma {
                0..=85 => {
                    shadow_sq += sq;
                    shadow_count += 1;
                }
                86..=170 => {
                    mid_sq += sq;
                    mid_count += 1;
                }
                _ => {
                    hi_sq += sq;
                    hi_count += 1;
                }
            }
        }

        let sigma = |sq: f64, count: usize| -> f32 {
            if count == 0 {
                0.02
            } else {
                (sq / count as f64).sqrt() as f32 / 255.0
            }
        };

        Self {
            shadow_sigma: sigma(shadow_sq, shadow_count),
            midtone_sigma: sigma(mid_sq, mid_count),
            highlight_sigma: sigma(hi_sq, hi_count),
            spatial_correlation: 0.3,
        }
    }

    /// Return the appropriate sigma for a given luma value.
    #[must_use]
    pub fn sigma_for_luma(&self, luma: u8) -> f32 {
        match luma {
            0..=85 => self.shadow_sigma,
            86..=170 => self.midtone_sigma,
            _ => self.highlight_sigma,
        }
    }
}

// ---------------------------------------------------------------------------
// Box blur helper
// ---------------------------------------------------------------------------

fn box_blur_3x3(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixels.len()];
    for row in 0..height {
        for col in 0..width {
            let mut sum = 0u32;
            let mut count = 0u32;
            for dr in 0usize..3 {
                let r = row.wrapping_add(dr).wrapping_sub(1);
                if r >= height {
                    continue;
                }
                for dc in 0usize..3 {
                    let c = col.wrapping_add(dc).wrapping_sub(1);
                    if c >= width {
                        continue;
                    }
                    sum += pixels[r * width + c] as u32;
                    count += 1;
                }
            }
            out[row * width + col] = (sum / count.max(1)) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Grain synthesis
// ---------------------------------------------------------------------------

/// Generate deterministic grain for a single pixel position.
///
/// Uses a combined multiplicative LCG + XOR-shift to produce white noise,
/// then applies a simple spatial correlation approximation by mixing in the
/// previous sample.
struct GrainGenerator {
    state: u64,
    prev: f32,
    correlation: f32,
}

impl GrainGenerator {
    fn new(seed: u64, correlation: f32) -> Self {
        Self {
            state: seed,
            prev: 0.0,
            correlation,
        }
    }

    fn next(&mut self, sigma: f32) -> f32 {
        // LCG step
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Box-Muller approximation (cheap uniform → approx gaussian via sum)
        let u1 = (self.state >> 33) as f32 / u32::MAX as f32;
        self.state = self.state.rotate_right(17).wrapping_add(0xDEAD_BEEF);
        let u2 = (self.state >> 33) as f32 / u32::MAX as f32;
        let gaussian = (u1 + u2 - 1.0) * 1.7321; // ≈ std 1 uniform sum

        // Spatial correlation
        let correlated = self.correlation * self.prev + (1.0 - self.correlation) * gaussian;
        self.prev = correlated;
        correlated * sigma
    }
}

// ---------------------------------------------------------------------------
// Bilinear resize
// ---------------------------------------------------------------------------

fn bilinear_resize_gray(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return vec![0u8; dw * dh];
    }
    let mut out = vec![0u8; dw * dh];
    let x_ratio = sw as f64 / dw as f64;
    let y_ratio = sh as f64 / dh as f64;

    for row in 0..dh {
        let src_y = (row as f64 + 0.5) * y_ratio - 0.5;
        let y0 = (src_y.floor() as isize).max(0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (src_y - src_y.floor()) as f32;

        for col in 0..dw {
            let src_x = (col as f64 + 0.5) * x_ratio - 0.5;
            let x0 = (src_x.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let fx = (src_x - src_x.floor()) as f32;

            let p00 = src[y0 * sw + x0] as f32;
            let p10 = src[y0 * sw + x1] as f32;
            let p01 = src[y1 * sw + x0] as f32;
            let p11 = src[y1 * sw + x1] as f32;

            let v = p00 * (1.0 - fx) * (1.0 - fy)
                + p10 * fx * (1.0 - fy)
                + p01 * (1.0 - fx) * fy
                + p11 * fx * fy;

            out[row * dw + col] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// FilmGrainScaler
// ---------------------------------------------------------------------------

/// Film grain-aware scaler: removes grain, resizes, then re-synthesises grain.
pub struct FilmGrainScaler {
    profile: GrainProfile,
    /// Fixed seed for grain re-synthesis determinism.
    grain_seed: u64,
}

impl FilmGrainScaler {
    /// Create a new scaler with the given grain profile.
    #[must_use]
    pub fn new(profile: GrainProfile) -> Self {
        Self {
            profile,
            grain_seed: 0xC0FFEE_1234_5678,
        }
    }

    /// Create a scaler with a custom grain seed for reproducible output.
    #[must_use]
    pub fn with_seed(profile: GrainProfile, seed: u64) -> Self {
        Self {
            profile,
            grain_seed: seed,
        }
    }

    /// Scale a grayscale (single-channel u8) image.
    ///
    /// 1. Remove grain via 3×3 low-pass filter.
    /// 2. Bilinear resize to target dimensions.
    /// 3. Synthesise new grain matching the profile at the target size.
    ///
    /// # Errors
    ///
    /// Returns [`ScalingError`] for zero dimensions or mismatched buffer size.
    pub fn scale(
        &self,
        pixels: &[u8],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> Result<Vec<u8>, ScalingError> {
        if src_width == 0 || src_height == 0 || dst_width == 0 || dst_height == 0 {
            return Err(ScalingError::InvalidDimensions(format!(
                "film_grain_scale: zero dimension: src={src_width}×{src_height}, dst={dst_width}×{dst_height}"
            )));
        }
        let expected = src_width * src_height;
        if pixels.len() != expected {
            return Err(ScalingError::InsufficientBuffer {
                expected,
                actual: pixels.len(),
            });
        }

        // Step 1: remove grain
        let clean = box_blur_3x3(pixels, src_width, src_height);

        // Step 2: resize clean base image
        let resized = bilinear_resize_gray(&clean, src_width, src_height, dst_width, dst_height);

        // Step 3: re-synthesise grain
        let mut gen = GrainGenerator::new(self.grain_seed, self.profile.spatial_correlation);
        let mut output = resized;

        for px in output.iter_mut() {
            let sigma = self.profile.sigma_for_luma(*px);
            let grain = gen.next(sigma);
            let new_val = *px as f32 + grain * 255.0;
            *px = new_val.clamp(0.0, 255.0).round() as u8;
        }

        Ok(output)
    }

    /// Return the grain profile in use.
    #[must_use]
    pub fn profile(&self) -> &GrainProfile {
        &self.profile
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grain_profile_default() {
        let p = GrainProfile::default();
        assert!(p.shadow_sigma > 0.0);
        assert!(p.midtone_sigma > 0.0);
        assert!(p.highlight_sigma > 0.0);
    }

    #[test]
    fn test_grain_profile_auto_detect_uniform() {
        // Uniform image → very low grain sigma
        let pixels = vec![128u8; 64 * 64];
        let p = GrainProfile::auto_detect(&pixels, 64, 64);
        assert!(
            p.midtone_sigma < 0.01,
            "uniform image should have near-zero sigma"
        );
    }

    #[test]
    fn test_grain_profile_sigma_for_luma() {
        let p = GrainProfile {
            shadow_sigma: 0.1,
            midtone_sigma: 0.05,
            highlight_sigma: 0.02,
            spatial_correlation: 0.0,
        };
        assert!((p.sigma_for_luma(0) - 0.1).abs() < f32::EPSILON);
        assert!((p.sigma_for_luma(128) - 0.05).abs() < f32::EPSILON);
        assert!((p.sigma_for_luma(255) - 0.02).abs() < f32::EPSILON);
    }

    #[test]
    fn test_film_grain_scaler_output_size() {
        let profile = GrainProfile::default();
        let scaler = FilmGrainScaler::new(profile);
        let input = vec![128u8; 64 * 64];
        let output = scaler.scale(&input, 64, 64, 32, 32).expect("scale ok");
        assert_eq!(output.len(), 32 * 32);
    }

    #[test]
    fn test_film_grain_scaler_upscale() {
        let profile = GrainProfile::default();
        let scaler = FilmGrainScaler::new(profile);
        let input = vec![200u8; 16 * 16];
        let output = scaler.scale(&input, 16, 16, 32, 32).expect("upscale ok");
        assert_eq!(output.len(), 32 * 32);
    }

    #[test]
    fn test_film_grain_scaler_zero_dims_err() {
        let scaler = FilmGrainScaler::new(GrainProfile::default());
        assert!(scaler.scale(&[], 0, 64, 32, 32).is_err());
        assert!(scaler.scale(&[], 64, 64, 0, 32).is_err());
    }

    #[test]
    fn test_film_grain_scaler_deterministic() {
        let profile = GrainProfile::default();
        let s1 = FilmGrainScaler::with_seed(profile.clone(), 42);
        let s2 = FilmGrainScaler::with_seed(profile, 42);
        let input = vec![100u8; 16 * 16];
        let o1 = s1.scale(&input, 16, 16, 8, 8).expect("ok");
        let o2 = s2.scale(&input, 16, 16, 8, 8).expect("ok");
        assert_eq!(o1, o2, "same seed should produce identical output");
    }

    #[test]
    fn test_box_blur_3x3_size() {
        let input = vec![100u8; 10 * 10];
        let out = box_blur_3x3(&input, 10, 10);
        assert_eq!(out.len(), 100);
    }

    #[test]
    fn test_bilinear_resize_2x() {
        let src = vec![128u8; 8 * 8];
        let dst = bilinear_resize_gray(&src, 8, 8, 16, 16);
        assert_eq!(dst.len(), 16 * 16);
    }
}
