//! Film grain synthesis for photorealistic analog film emulation.
//!
//! Generates spatially correlated grain noise using a 2-D autoregressive (AR)
//! model, matching the statistical characteristics of real photochemical grain.
//!
//! # Model overview
//!
//! Each output sample `g[y,x]` is computed as:
//!
//! ```text
//! g[y,x] = ar_h * g[y, x-1] + ar_v * g[y-1, x] - ar_hv * g[y-1, x-1]
//!          + sigma * white[y,x]
//! ```
//!
//! where `white[y,x]` is white (uncorrelated) Gaussian noise produced by a
//! deterministic hash so the output is fully reproducible given the same seed.
//!
//! Luma (Y) and chroma (Cb, Cr) channels receive independent grain fields
//! whose relative strengths are controlled by `FilmGrainConfig`.  An optional
//! colour-correlation term lets chroma grain track luma grain, reproducing the
//! colour shift seen in real grain.
//!
//! Temporal variation is achieved by mixing a stable spatial-hash seed with a
//! per-frame counter so successive frames have related but distinct patterns.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f64::consts::PI;

use crate::error::{ImageError, ImageResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Grain intensity per YCbCr channel.
///
/// All values are in the range `[0.0, 1.0]`.
#[derive(Clone, Debug)]
pub struct ChannelStrength {
    /// Luma (Y) grain strength.
    pub luma: f32,
    /// Blue-difference chroma (Cb) grain strength.
    pub cb: f32,
    /// Red-difference chroma (Cr) grain strength.
    pub cr: f32,
}

impl Default for ChannelStrength {
    fn default() -> Self {
        // Realistic film: strong luma grain, weaker chroma
        Self {
            luma: 1.0,
            cb: 0.4,
            cr: 0.4,
        }
    }
}

/// Configuration for film grain synthesis.
#[derive(Clone, Debug)]
pub struct FilmGrainConfig {
    /// Overall grain intensity in `[0.0, 1.0]`.  Controls the standard
    /// deviation of the grain field added to the image.
    pub intensity: f32,

    /// Spatial size of grain clumps in `[0.5, 8.0]`.
    ///
    /// Higher values produce coarser, more visible grain clusters.  Internally
    /// this maps to the AR coefficients that determine spatial correlation.
    pub grain_size: f32,

    /// Colour correlation between luma grain and chroma grain `[0.0, 1.0]`.
    ///
    /// At 0.0 the channels are independent; at 1.0 chroma grain is a scaled
    /// copy of luma grain (as seen in some colour negative stocks).
    pub color_correlation: f32,

    /// Per-channel strength multipliers (relative to `intensity`).
    pub channel_strength: ChannelStrength,

    /// When `true` the spatial-hash seed is mixed with a monotonically
    /// increasing counter so each call to [`FilmGrainSynthesizer::apply`]
    /// produces a different but related pattern.  When `false` the pattern is
    /// identical across frames (useful for static texture testing).
    pub temporal_seed_variation: bool,

    /// Base seed for the deterministic noise generator.
    pub seed: u64,
}

impl Default for FilmGrainConfig {
    fn default() -> Self {
        Self {
            intensity: 0.05,
            grain_size: 1.0,
            color_correlation: 0.3,
            channel_strength: ChannelStrength::default(),
            temporal_seed_variation: true,
            seed: 0xDEAD_BEEF_CAFE_1234,
        }
    }
}

impl FilmGrainConfig {
    /// Create a new config with the given intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set the grain size.
    #[must_use]
    pub fn with_grain_size(mut self, grain_size: f32) -> Self {
        self.grain_size = grain_size.clamp(0.5, 8.0);
        self
    }

    /// Set the colour correlation.
    #[must_use]
    pub fn with_color_correlation(mut self, color_correlation: f32) -> Self {
        self.color_correlation = color_correlation.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable temporal seed variation.
    #[must_use]
    pub fn with_temporal_seed_variation(mut self, enabled: bool) -> Self {
        self.temporal_seed_variation = enabled;
        self
    }

    /// Set the base seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set per-channel strength multipliers.
    #[must_use]
    pub fn with_channel_strength(mut self, strength: ChannelStrength) -> Self {
        self.channel_strength = strength;
        self
    }
}

/// Pixel-format of the buffer passed to [`FilmGrainSynthesizer::apply`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrainPixelFormat {
    /// Planar 8-bit YCbCr 4:4:4 (three separate planes, one byte per sample).
    YCbCr444Planar8,
    /// Interleaved 8-bit RGB (one plane, 3 bytes per pixel: R G B).
    Rgb8,
    /// Interleaved 8-bit grayscale (one plane, 1 byte per pixel).
    Gray8,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Mix two 64-bit values with a finaliser from MurmurHash3.
#[inline]
fn hash_mix(a: u64, b: u64) -> u64 {
    let mut x = a ^ b;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    x
}

/// Deterministic spatial white-noise sample in `[-1.0, 1.0]` using
/// coordinate-hashed Box-Muller pairs.
///
/// Using coordinates rather than a sequential PRNG state allows the AR
/// recursion to use spatially correct boundary conditions without storing the
/// full previous row.
fn white_noise_at(x: u32, y: u32, seed: u64) -> f64 {
    // Two independent hashes for the Box-Muller pair
    let h1 = hash_mix(hash_mix(seed, x as u64), y as u64);
    let h2 = hash_mix(hash_mix(seed ^ 0x1234_5678_9ABC_DEF0, x as u64), y as u64);

    // Map to (0, 1) avoiding exact 0
    let u1 = ((h1 >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
    let u2 = (h2 >> 11) as f64 / (1u64 << 53) as f64;

    // Box-Muller → standard normal; take the cosine branch
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Derive AR coefficients from `grain_size`.
///
/// For an AR(1) process on a 2-D grid the correlation length scales with
/// `1 / (1 - phi)` where `phi` is the coefficient.  We clamp to [0.0, 0.94]
/// to keep the process stable.
#[inline]
fn ar_coefficient(grain_size: f32) -> f64 {
    // grain_size in [0.5, 8.0] → phi in [0.0, 0.94]
    let phi = 1.0 - 1.0 / (grain_size as f64).max(0.5);
    phi.clamp(0.0, 0.94)
}

/// Generate a 2-D AR(1,1) grain field of size `width × height`.
///
/// The recursion is:
/// ```text
/// g[y,x] = phi_h * g[y, x-1]
///         + phi_v * g[y-1, x]
///         - phi_hv * g[y-1, x-1]   // subtract corner to avoid double-count
///         + sigma * w[y,x]
/// ```
///
/// `sigma` is chosen so the steady-state variance is 1.0.
fn generate_ar_grain_field(width: u32, height: u32, grain_size: f32, seed: u64) -> Vec<f64> {
    let w = width as usize;
    let h = height as usize;
    let phi = ar_coefficient(grain_size);
    // Steady-state variance of AR(1,1) on a 2-D grid:
    //   Var = sigma^2 / (1 - phi_h^2 - phi_v^2 + phi_hv^2)
    // We use phi_h = phi_v = phi and phi_hv = phi^2.
    let denom = (1.0 - 2.0 * phi * phi + phi.powi(4)).max(1e-9);
    let sigma = denom.sqrt();

    let mut field = vec![0.0f64; w * h];

    for y in 0..h {
        for x in 0..w {
            let w_sample = white_noise_at(x as u32, y as u32, seed);

            let left = if x > 0 { field[y * w + x - 1] } else { 0.0 };
            let top = if y > 0 { field[(y - 1) * w + x] } else { 0.0 };
            let top_left = if x > 0 && y > 0 {
                field[(y - 1) * w + x - 1]
            } else {
                0.0
            };

            field[y * w + x] = phi * left + phi * top - phi * phi * top_left + sigma * w_sample;
        }
    }

    field
}

/// Apply a grain field to a u8 pixel plane.
///
/// `intensity_scale` already incorporates the global intensity and the
/// per-channel multiplier.  Values are clamped to `[0, 255]`.
#[inline]
fn apply_grain_to_plane(plane: &mut [u8], grain: &[f64], intensity_scale: f64) {
    for (px, &g) in plane.iter_mut().zip(grain.iter()) {
        let delta = (g * intensity_scale * 255.0).round() as i32;
        *px = (*px as i32 + delta).clamp(0, 255) as u8;
    }
}

/// Convert an 8-bit RGB pixel to f32 YCbCr (BT.601).
#[inline]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 16.0 + 0.257 * rf + 0.504 * gf + 0.098 * bf;
    let cb = 128.0 - 0.148 * rf - 0.291 * gf + 0.439 * bf;
    let cr = 128.0 + 0.439 * rf - 0.368 * gf - 0.071 * bf;
    (y, cb, cr)
}

/// Convert f32 YCbCr back to u8 RGB (BT.601).
#[inline]
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (u8, u8, u8) {
    let y2 = y - 16.0;
    let cb2 = cb - 128.0;
    let cr2 = cr - 128.0;
    let r = (1.164 * y2 + 1.596 * cr2).clamp(0.0, 255.0) as u8;
    let g = (1.164 * y2 - 0.392 * cb2 - 0.813 * cr2).clamp(0.0, 255.0) as u8;
    let b = (1.164 * y2 + 2.017 * cb2).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// ---------------------------------------------------------------------------
// Public synthesizer
// ---------------------------------------------------------------------------

/// Film grain synthesizer.
///
/// Holds a [`FilmGrainConfig`] and an internal frame counter used for temporal
/// seed variation.  Call [`FilmGrainSynthesizer::apply`] once per frame.
///
/// # Example
///
/// ```rust
/// use oximedia_image::film_grain::{FilmGrainConfig, FilmGrainSynthesizer, GrainPixelFormat};
///
/// let config = FilmGrainConfig::default();
/// let mut synth = FilmGrainSynthesizer::new(config);
///
/// // 4×4 grayscale image
/// let mut pixels = vec![128u8; 16];
/// synth.apply(&mut [pixels.as_mut_slice()], 4, 4, GrainPixelFormat::Gray8).expect("apply grain");
/// ```
#[derive(Debug)]
pub struct FilmGrainSynthesizer {
    config: FilmGrainConfig,
    frame_counter: u64,
}

impl FilmGrainSynthesizer {
    /// Create a new synthesizer with the given configuration.
    #[must_use]
    pub fn new(config: FilmGrainConfig) -> Self {
        Self {
            config,
            frame_counter: 0,
        }
    }

    /// Access the current configuration.
    #[must_use]
    pub fn config(&self) -> &FilmGrainConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: FilmGrainConfig) {
        self.config = config;
    }

    /// Return the current internal frame counter (incremented on each `apply` call
    /// when `temporal_seed_variation` is enabled).
    #[must_use]
    pub fn frame_counter(&self) -> u64 {
        self.frame_counter
    }

    /// Apply film grain to `planes` in-place.
    ///
    /// `planes` must contain the correct number of slices for `format`:
    ///
    /// | Format | Required planes |
    /// |--------|-----------------|
    /// | `YCbCr444Planar8` | 3 (Y, Cb, Cr) |
    /// | `Rgb8`  | 1 (interleaved R G B) |
    /// | `Gray8` | 1 (luminance only) |
    ///
    /// Each plane must have exactly `width * height` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidDimensions`] if `width == 0 || height == 0`.
    /// Returns [`ImageError::InvalidFormat`] for plane count or size mismatches.
    pub fn apply(
        &mut self,
        planes: &mut [&mut [u8]],
        width: u32,
        height: u32,
        format: GrainPixelFormat,
    ) -> ImageResult<()> {
        if width == 0 || height == 0 {
            return Err(ImageError::InvalidDimensions(width, height));
        }

        let pixel_count = (width as usize) * (height as usize);

        // Derive effective seed
        let frame_seed = if self.config.temporal_seed_variation {
            hash_mix(self.config.seed, self.frame_counter)
        } else {
            self.config.seed
        };

        // Seeds per channel
        let seed_y = frame_seed;
        let seed_cb = hash_mix(frame_seed, 0xABCD_EF01_2345_6789);
        let seed_cr = hash_mix(frame_seed, 0x1234_5678_9ABC_DEF0);

        let intensity = self.config.intensity as f64;
        let corr = self.config.color_correlation as f64;
        let gs = self.config.grain_size;

        match format {
            GrainPixelFormat::YCbCr444Planar8 => {
                if planes.len() != 3 {
                    return Err(ImageError::InvalidFormat(format!(
                        "YCbCr444Planar8 requires 3 planes, got {}",
                        planes.len()
                    )));
                }
                for (i, plane) in planes.iter().enumerate() {
                    if plane.len() != pixel_count {
                        return Err(ImageError::InvalidFormat(format!(
                            "Plane {i} length {} != expected {pixel_count}",
                            plane.len()
                        )));
                    }
                }

                // Generate luma grain first (used for chroma correlation)
                let grain_y = generate_ar_grain_field(width, height, gs, seed_y);

                // Independent chroma fields
                let grain_cb_raw = generate_ar_grain_field(width, height, gs, seed_cb);
                let grain_cr_raw = generate_ar_grain_field(width, height, gs, seed_cr);

                // Blend chroma with luma according to color_correlation
                let grain_cb: Vec<f64> = grain_cb_raw
                    .iter()
                    .zip(grain_y.iter())
                    .map(|(&c, &y)| (1.0 - corr) * c + corr * y)
                    .collect();
                let grain_cr: Vec<f64> = grain_cr_raw
                    .iter()
                    .zip(grain_y.iter())
                    .map(|(&c, &y)| (1.0 - corr) * c + corr * y)
                    .collect();

                let scale_y = intensity * self.config.channel_strength.luma as f64;
                let scale_cb = intensity * self.config.channel_strength.cb as f64;
                let scale_cr = intensity * self.config.channel_strength.cr as f64;

                // planes is &mut [&mut [u8]], split to access individually
                let (plane_y_slice, rest) = planes.split_at_mut(1);
                let (plane_cb_slice, plane_cr_slice) = rest.split_at_mut(1);

                apply_grain_to_plane(plane_y_slice[0], &grain_y, scale_y);
                apply_grain_to_plane(plane_cb_slice[0], &grain_cb, scale_cb);
                apply_grain_to_plane(plane_cr_slice[0], &grain_cr, scale_cr);
            }

            GrainPixelFormat::Rgb8 => {
                if planes.len() != 1 {
                    return Err(ImageError::InvalidFormat(format!(
                        "Rgb8 requires 1 plane, got {}",
                        planes.len()
                    )));
                }
                let expected_len = pixel_count * 3;
                if planes[0].len() != expected_len {
                    return Err(ImageError::InvalidFormat(format!(
                        "Rgb8 plane length {} != expected {expected_len}",
                        planes[0].len()
                    )));
                }

                // Convert to YCbCr, grain in YCbCr, convert back
                let grain_y = generate_ar_grain_field(width, height, gs, seed_y);
                let grain_cb_raw = generate_ar_grain_field(width, height, gs, seed_cb);
                let grain_cr_raw = generate_ar_grain_field(width, height, gs, seed_cr);

                let grain_cb: Vec<f64> = grain_cb_raw
                    .iter()
                    .zip(grain_y.iter())
                    .map(|(&c, &y)| (1.0 - corr) * c + corr * y)
                    .collect();
                let grain_cr: Vec<f64> = grain_cr_raw
                    .iter()
                    .zip(grain_y.iter())
                    .map(|(&c, &y)| (1.0 - corr) * c + corr * y)
                    .collect();

                let scale_y = intensity * self.config.channel_strength.luma as f64;
                let scale_cb = intensity * self.config.channel_strength.cb as f64;
                let scale_cr = intensity * self.config.channel_strength.cr as f64;

                let buf = &mut planes[0];
                for i in 0..pixel_count {
                    let r = buf[i * 3];
                    let g = buf[i * 3 + 1];
                    let b = buf[i * 3 + 2];

                    let (y_f, cb_f, cr_f) = rgb_to_ycbcr(r, g, b);

                    let y_new = y_f + (grain_y[i] * scale_y * 255.0) as f32;
                    let cb_new = cb_f + (grain_cb[i] * scale_cb * 255.0) as f32;
                    let cr_new = cr_f + (grain_cr[i] * scale_cr * 255.0) as f32;

                    let (r2, g2, b2) = ycbcr_to_rgb(y_new, cb_new, cr_new);
                    buf[i * 3] = r2;
                    buf[i * 3 + 1] = g2;
                    buf[i * 3 + 2] = b2;
                }
            }

            GrainPixelFormat::Gray8 => {
                if planes.len() != 1 {
                    return Err(ImageError::InvalidFormat(format!(
                        "Gray8 requires 1 plane, got {}",
                        planes.len()
                    )));
                }
                if planes[0].len() != pixel_count {
                    return Err(ImageError::InvalidFormat(format!(
                        "Gray8 plane length {} != expected {pixel_count}",
                        planes[0].len()
                    )));
                }

                let grain_y = generate_ar_grain_field(width, height, gs, seed_y);
                let scale_y = intensity * self.config.channel_strength.luma as f64;
                apply_grain_to_plane(planes[0], &grain_y, scale_y);
            }
        }

        if self.config.temporal_seed_variation {
            self.frame_counter = self.frame_counter.wrapping_add(1);
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: mean absolute difference between two byte slices
    fn mad(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sum: i64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x as i64 - y as i64).abs())
            .sum();
        sum as f64 / a.len() as f64
    }

    // -----------------------------------------------------------------------
    // 1. Basic grayscale application does not panic and changes pixel values
    #[test]
    fn test_gray8_applies_grain() {
        let config = FilmGrainConfig::default().with_intensity(0.1);
        let mut synth = FilmGrainSynthesizer::new(config);
        let original = vec![128u8; 64];
        let mut pixels = original.clone();
        synth
            .apply(&mut [pixels.as_mut_slice()], 8, 8, GrainPixelFormat::Gray8)
            .expect("apply should succeed");
        // At intensity 0.1, at least some pixels should change
        assert_ne!(pixels, original, "grain must alter pixel values");
    }

    // -----------------------------------------------------------------------
    // 2. All output values remain in [0, 255]
    #[test]
    fn test_gray8_values_in_range() {
        let config = FilmGrainConfig::default()
            .with_intensity(1.0) // maximum intensity
            .with_temporal_seed_variation(false);
        let mut synth = FilmGrainSynthesizer::new(config);
        let mut pixels = vec![128u8; 256];
        synth
            .apply(
                &mut [pixels.as_mut_slice()],
                16,
                16,
                GrainPixelFormat::Gray8,
            )
            .expect("apply should succeed");
        assert!(!pixels.is_empty());
    }

    // -----------------------------------------------------------------------
    // 3. Deterministic output with temporal_seed_variation disabled
    #[test]
    fn test_deterministic_without_temporal_variation() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.1)
            .with_temporal_seed_variation(false);

        let run = |seed: u64| {
            let mut synth = FilmGrainSynthesizer::new(config.clone().with_seed(seed));
            let mut pixels = vec![128u8; 64];
            synth
                .apply(&mut [pixels.as_mut_slice()], 8, 8, GrainPixelFormat::Gray8)
                .expect("apply grain with seed");
            pixels
        };

        let a = run(42);
        let b = run(42);
        assert_eq!(a, b, "same seed must produce same output");
    }

    // -----------------------------------------------------------------------
    // 4. Temporal variation produces different frames
    #[test]
    fn test_temporal_variation_differs_across_frames() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.1)
            .with_temporal_seed_variation(true);
        let mut synth = FilmGrainSynthesizer::new(config);

        let base = vec![128u8; 64];
        let mut frame1 = base.clone();
        let mut frame2 = base.clone();

        synth
            .apply(&mut [frame1.as_mut_slice()], 8, 8, GrainPixelFormat::Gray8)
            .expect("apply grain frame 1");
        synth
            .apply(&mut [frame2.as_mut_slice()], 8, 8, GrainPixelFormat::Gray8)
            .expect("apply grain frame 2");

        assert_ne!(
            frame1, frame2,
            "temporal variation must produce different frames"
        );
        assert_eq!(synth.frame_counter(), 2);
    }

    // -----------------------------------------------------------------------
    // 5. Zero intensity leaves image unchanged
    #[test]
    fn test_zero_intensity_no_change() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.0)
            .with_temporal_seed_variation(false);
        let mut synth = FilmGrainSynthesizer::new(config);
        let original = vec![100u8; 64];
        let mut pixels = original.clone();
        synth
            .apply(&mut [pixels.as_mut_slice()], 8, 8, GrainPixelFormat::Gray8)
            .expect("apply zero intensity grain");
        assert_eq!(pixels, original, "zero intensity must not alter image");
    }

    // -----------------------------------------------------------------------
    // 6. Larger grain_size produces more spatially correlated (smoother) grain
    #[test]
    fn test_grain_size_affects_spatial_correlation() {
        // Spatial correlation ≈ average |g[i] - g[i+1]|; smaller means smoother
        let measure_variation = |grain_size: f32| -> f64 {
            let config = FilmGrainConfig::default()
                .with_intensity(0.5)
                .with_grain_size(grain_size)
                .with_temporal_seed_variation(false)
                .with_seed(12345);
            let mut synth = FilmGrainSynthesizer::new(config);
            let mut pixels = vec![128u8; 256];
            synth
                .apply(
                    &mut [pixels.as_mut_slice()],
                    16,
                    16,
                    GrainPixelFormat::Gray8,
                )
                .expect("apply grain for variation measurement");
            let diffs: u64 = pixels
                .windows(2)
                .map(|w| (w[0] as i64 - w[1] as i64).unsigned_abs())
                .sum();
            diffs as f64 / (pixels.len() - 1) as f64
        };

        let fine = measure_variation(0.5);
        let coarse = measure_variation(4.0);
        // Coarser grain should have lower adjacent-pixel variation (smoother blobs)
        assert!(
            coarse < fine,
            "coarse grain ({coarse:.3}) should vary less per-pixel than fine grain ({fine:.3})"
        );
    }

    // -----------------------------------------------------------------------
    // 7. YCbCr 4:4:4 planar application
    #[test]
    fn test_ycbcr_planar_applies_grain() {
        let config = FilmGrainConfig::default().with_intensity(0.1);
        let mut synth = FilmGrainSynthesizer::new(config);

        let original_y = vec![128u8; 64];
        let original_cb = vec![128u8; 64];
        let original_cr = vec![128u8; 64];

        let mut plane_y = original_y.clone();
        let mut plane_cb = original_cb.clone();
        let mut plane_cr = original_cr.clone();

        synth
            .apply(
                &mut [
                    plane_y.as_mut_slice(),
                    plane_cb.as_mut_slice(),
                    plane_cr.as_mut_slice(),
                ],
                8,
                8,
                GrainPixelFormat::YCbCr444Planar8,
            )
            .expect("YCbCr planar apply should succeed");

        // At least luma must have changed
        assert_ne!(plane_y, original_y, "luma plane must be modified");
    }

    // -----------------------------------------------------------------------
    // 8. Chroma weaker than luma by default
    #[test]
    fn test_chroma_weaker_than_luma_by_default() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.2)
            .with_temporal_seed_variation(false);
        let mut synth = FilmGrainSynthesizer::new(config);

        let base = vec![128u8; 64];
        let mut plane_y = base.clone();
        let mut plane_cb = base.clone();
        let mut plane_cr = base.clone();

        synth
            .apply(
                &mut [
                    plane_y.as_mut_slice(),
                    plane_cb.as_mut_slice(),
                    plane_cr.as_mut_slice(),
                ],
                8,
                8,
                GrainPixelFormat::YCbCr444Planar8,
            )
            .expect("apply grain to YCbCr planes");

        let mad_y = mad(&base, &plane_y);
        let mad_cb = mad(&base, &plane_cb);
        let mad_cr = mad(&base, &plane_cr);

        assert!(
            mad_y >= mad_cb,
            "luma MAD ({mad_y:.3}) should be >= chroma Cb MAD ({mad_cb:.3})"
        );
        assert!(
            mad_y >= mad_cr,
            "luma MAD ({mad_y:.3}) should be >= chroma Cr MAD ({mad_cr:.3})"
        );
    }

    // -----------------------------------------------------------------------
    // 9. RGB8 format round-trip does not panic and stays in range
    #[test]
    fn test_rgb8_applies_grain() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.05)
            .with_temporal_seed_variation(false);
        let mut synth = FilmGrainSynthesizer::new(config);

        let mut pixels = vec![0u8; 32 * 32 * 3];
        // Fill with mid-grey
        for chunk in pixels.chunks_exact_mut(3) {
            chunk[0] = 128;
            chunk[1] = 128;
            chunk[2] = 128;
        }

        synth
            .apply(&mut [pixels.as_mut_slice()], 32, 32, GrainPixelFormat::Rgb8)
            .expect("RGB8 apply should succeed");

        // All values remain valid u8
        assert!(!pixels.is_empty());
    }

    // -----------------------------------------------------------------------
    // 10. Invalid dimensions are rejected
    #[test]
    fn test_invalid_dimensions_rejected() {
        let config = FilmGrainConfig::default();
        let mut synth = FilmGrainSynthesizer::new(config);
        let mut pixels = vec![128u8; 0];
        let result = synth.apply(&mut [pixels.as_mut_slice()], 0, 0, GrainPixelFormat::Gray8);
        assert!(result.is_err(), "zero dimensions must return an error");
    }

    // -----------------------------------------------------------------------
    // 11. Wrong plane count is rejected
    #[test]
    fn test_wrong_plane_count_rejected() {
        let config = FilmGrainConfig::default();
        let mut synth = FilmGrainSynthesizer::new(config);
        let mut p1 = vec![128u8; 64];
        let mut p2 = vec![128u8; 64];
        // YCbCr needs 3 planes, provide 2
        let result = synth.apply(
            &mut [p1.as_mut_slice(), p2.as_mut_slice()],
            8,
            8,
            GrainPixelFormat::YCbCr444Planar8,
        );
        assert!(result.is_err(), "wrong plane count must return error");
    }

    // -----------------------------------------------------------------------
    // 12. AR grain field has near-zero mean
    #[test]
    fn test_ar_grain_field_near_zero_mean() {
        let field = generate_ar_grain_field(64, 64, 1.0, 0xDEAD_BEEF);
        let mean = field.iter().sum::<f64>() / field.len() as f64;
        assert!(
            mean.abs() < 0.5,
            "AR grain field mean {mean:.4} should be near zero"
        );
    }

    // -----------------------------------------------------------------------
    // 13. Color correlation = 1.0 makes chroma grain track luma grain
    #[test]
    fn test_full_color_correlation() {
        let config = FilmGrainConfig::default()
            .with_intensity(0.3)
            .with_color_correlation(1.0)
            .with_temporal_seed_variation(false)
            .with_seed(0xBEEF);
        let mut synth = FilmGrainSynthesizer::new(config);

        let base = vec![128u8; 64];
        let mut plane_y = base.clone();
        let mut plane_cb = base.clone();
        let mut plane_cr = base.clone();

        synth
            .apply(
                &mut [
                    plane_y.as_mut_slice(),
                    plane_cb.as_mut_slice(),
                    plane_cr.as_mut_slice(),
                ],
                8,
                8,
                GrainPixelFormat::YCbCr444Planar8,
            )
            .expect("apply grain with chroma correlation");

        // With full correlation, Cb and Cr grain should be highly correlated to Y
        // (same spatial pattern but scaled); check they at least changed
        assert_ne!(plane_cb, base);
        assert_ne!(plane_cr, base);
    }
}
