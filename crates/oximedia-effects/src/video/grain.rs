//! Film grain simulation effect.
//!
//! Simulates the photochemical grain found in analog film. The grain is generated
//! using a deterministic pseudo-random noise function seeded per-frame, with
//! optional spatial correlation to mimic the clumped appearance of real silver-halide
//! grains. Grain intensity scales with local luminance (shadows have more visible grain
//! than highlights – the Hurter-Driffield curve effect).

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// Configuration for the grain effect.
#[derive(Debug, Clone)]
pub struct GrainConfig {
    /// Grain intensity (standard deviation of Gaussian noise, in pixel units 0–128).
    pub intensity: f32,
    /// Grain size: 1 = pixel-sized grain, 2+ = larger clumps.
    pub size: u32,
    /// Whether grain is monochromatic (same value for R/G/B) or colored.
    pub monochromatic: bool,
    /// Shadow boost: extra grain added in darker areas (0.0 = none, 1.0 = double grain in shadows).
    pub shadow_boost: f32,
    /// Seed for the deterministic PRNG (change per frame for animation).
    pub seed: u64,
}

impl Default for GrainConfig {
    fn default() -> Self {
        Self {
            intensity: 12.0,
            size: 1,
            monochromatic: true,
            shadow_boost: 0.3,
            seed: 42,
        }
    }
}

impl GrainConfig {
    /// Fine-grain 35mm look.
    #[must_use]
    pub fn film_35mm() -> Self {
        Self {
            intensity: 8.0,
            size: 1,
            monochromatic: true,
            shadow_boost: 0.25,
            seed: 1,
        }
    }

    /// Heavy 16mm/8mm look.
    #[must_use]
    pub fn film_16mm() -> Self {
        Self {
            intensity: 25.0,
            size: 2,
            monochromatic: false,
            shadow_boost: 0.6,
            seed: 2,
        }
    }

    /// Digital noise (sensor noise).
    #[must_use]
    pub fn digital_noise() -> Self {
        Self {
            intensity: 6.0,
            size: 1,
            monochromatic: false,
            shadow_boost: 0.1,
            seed: 3,
        }
    }
}

// --------------------------------------------------------------------------
// Lightweight deterministic pseudo-random noise (xoshiro128** variant).
// We don't use `rand` here to keep the hot loop allocation-free.
// --------------------------------------------------------------------------

/// Fast 64-bit hash from a coordinate, returning a value in [-1, 1].
#[inline]
fn noise1(x: u64) -> f32 {
    // Splitmix64 single step
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    // Map to [-1, 1]
    #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
    let result = (z as i64 as f32) / (i64::MAX as f32);
    result
}

/// Grain effect processor.
pub struct GrainEffect {
    config: GrainConfig,
}

impl GrainEffect {
    /// Create a new grain effect.
    #[must_use]
    pub fn new(config: GrainConfig) -> Self {
        Self { config }
    }

    /// Apply film grain to a pixel buffer in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer size is incorrect.
    pub fn apply(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> VideoResult<()> {
        validate_buffer(data, width, height, format)?;

        let bpp = format.bytes_per_pixel();
        let cfg = &self.config;
        let grain_size = cfg.size.max(1) as usize;

        for py in 0..height {
            for px in 0..width {
                // Map to grain grid cell (for size > 1, multiple pixels share same noise)
                let gx = (px / grain_size) as u64;
                let gy = (py / grain_size) as u64;

                let idx = (py * width + px) * bpp;

                // Local luminance (average of R, G, B) for shadow boost calculation
                let lum =
                    (f32::from(data[idx]) + f32::from(data[idx + 1]) + f32::from(data[idx + 2]))
                        / (3.0 * 255.0);
                // Shadow boost: more grain in darker regions
                let shadow_mul = 1.0 + cfg.shadow_boost * (1.0 - lum);
                let effective_intensity = cfg.intensity * shadow_mul;

                if cfg.monochromatic {
                    // Same grain value for all channels
                    let g = noise1(
                        gx.wrapping_mul(0x9E37_79B9) ^ gy.wrapping_mul(0x6C62_272E) ^ cfg.seed,
                    );
                    let grain_val = g * effective_intensity;
                    data[idx] = clamp_u8(f32::from(data[idx]) + grain_val);
                    data[idx + 1] = clamp_u8(f32::from(data[idx + 1]) + grain_val);
                    data[idx + 2] = clamp_u8(f32::from(data[idx + 2]) + grain_val);
                } else {
                    // Independent grain per channel (color noise)
                    for c in 0..3 {
                        let g = noise1(
                            gx.wrapping_mul(0x9E37_79B9)
                                ^ gy.wrapping_mul(0x6C62_272E)
                                ^ cfg.seed
                                ^ (c as u64 * 0x517C_C1B7),
                        );
                        let grain_val = g * effective_intensity;
                        data[idx + c] = clamp_u8(f32::from(data[idx + c]) + grain_val);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mid_gray(w: usize, h: usize) -> Vec<u8> {
        vec![128u8; w * h * 3]
    }

    #[test]
    fn test_grain_default_applies() {
        let mut buf = mid_gray(64, 64);
        let g = GrainEffect::new(GrainConfig::default());
        assert!(g.apply(&mut buf, 64, 64, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_grain_changes_image() {
        let orig = mid_gray(32, 32);
        let mut buf = orig.clone();
        let g = GrainEffect::new(GrainConfig::default());
        g.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_ne!(buf, orig, "Grain should change pixel values");
    }

    #[test]
    fn test_grain_zero_intensity_unchanged() {
        let orig = mid_gray(32, 32);
        let mut buf = orig.clone();
        let cfg = GrainConfig {
            intensity: 0.0,
            shadow_boost: 0.0,
            ..Default::default()
        };
        let g = GrainEffect::new(cfg);
        g.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "Zero intensity should not change image");
    }

    #[test]
    fn test_grain_different_seeds_differ() {
        let orig = mid_gray(32, 32);

        let mut buf1 = orig.clone();
        GrainEffect::new(GrainConfig {
            seed: 1,
            ..Default::default()
        })
        .apply(&mut buf1, 32, 32, PixelFormat::Rgb)
        .expect("test expectation failed");

        let mut buf2 = orig.clone();
        GrainEffect::new(GrainConfig {
            seed: 2,
            ..Default::default()
        })
        .apply(&mut buf2, 32, 32, PixelFormat::Rgb)
        .expect("test expectation failed");

        assert_ne!(buf1, buf2, "Different seeds should produce different grain");
    }

    #[test]
    fn test_grain_color_noise() {
        let mut buf = mid_gray(32, 32);
        let cfg = GrainConfig {
            monochromatic: false,
            intensity: 20.0,
            ..Default::default()
        };
        let g = GrainEffect::new(cfg);
        g.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        // Color noise: R, G, B should differ for some pixels
        let has_color_diff = buf
            .chunks_exact(3)
            .any(|px| px[0] != px[1] || px[1] != px[2]);
        assert!(
            has_color_diff,
            "Color noise should introduce channel differences"
        );
    }

    #[test]
    fn test_grain_film_35mm() {
        let mut buf = mid_gray(32, 32);
        let g = GrainEffect::new(GrainConfig::film_35mm());
        assert!(g.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_grain_film_16mm() {
        let mut buf = mid_gray(32, 32);
        let g = GrainEffect::new(GrainConfig::film_16mm());
        assert!(g.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_grain_large_size_clumps() {
        let orig = mid_gray(32, 32);
        let mut buf = orig.clone();
        let cfg = GrainConfig {
            size: 4,
            intensity: 30.0,
            ..Default::default()
        };
        let g = GrainEffect::new(cfg);
        g.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        // With size 4, 4x4 blocks of pixels share the same grain value
        let bl = [buf[0], buf[1], buf[2]];
        let adjacent = [buf[3], buf[4], buf[5]];
        assert_eq!(bl, adjacent, "Pixels within same grain cell should match");
    }

    #[test]
    fn test_grain_rgba() {
        let mut buf = vec![128u8; 32 * 32 * 4];
        let g = GrainEffect::new(GrainConfig::default());
        assert!(g.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_grain_wrong_size_err() {
        let mut buf = vec![0u8; 5];
        let g = GrainEffect::new(GrainConfig::default());
        assert!(g.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }
}
