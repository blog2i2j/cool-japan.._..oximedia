//! Vignette effect.
//!
//! Darkens the image towards the edges with a configurable smooth falloff.
//! Two falloff models are provided: a cosine-based smooth curve and a power-law
//! model (matching real lens vignetting physics). The effect is applied to all
//! channels uniformly (luminance-preserving darkening).

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// Falloff curve for the vignette.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VignetteFalloff {
    /// Smooth cosine curve (soft, cinematic look).
    Cosine,
    /// Power-law (more physically accurate, sharper edge).
    Power(f32),
    /// Linear (hard, graphic look).
    Linear,
}

/// Configuration for the vignette effect.
#[derive(Debug, Clone)]
pub struct VignetteConfig {
    /// Maximum darkening amount at image edges (0.0 = no effect, 1.0 = fully black).
    pub strength: f32,
    /// Radius at which darkening begins, as fraction of half the shorter dimension.
    /// Values < 1.0 push the vignette inward; > 1.0 shifts it outward.
    pub inner_radius: f32,
    /// Outer radius for the falloff extent (normalized same as `inner_radius`).
    pub outer_radius: f32,
    /// Falloff curve model.
    pub falloff: VignetteFalloff,
    /// Center X (0.0–1.0).
    pub center_x: f32,
    /// Center Y (0.0–1.0).
    pub center_y: f32,
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            strength: 0.6,
            inner_radius: 0.5,
            outer_radius: 1.4,
            falloff: VignetteFalloff::Cosine,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl VignetteConfig {
    /// Subtle cinematic vignette.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            strength: 0.3,
            inner_radius: 0.7,
            outer_radius: 1.6,
            ..Default::default()
        }
    }

    /// Heavy tunnel-vision vignette.
    #[must_use]
    pub fn heavy() -> Self {
        Self {
            strength: 0.9,
            inner_radius: 0.3,
            outer_radius: 1.1,
            falloff: VignetteFalloff::Power(3.0),
            ..Default::default()
        }
    }
}

/// Vignette effect processor.
pub struct Vignette {
    config: VignetteConfig,
}

impl Vignette {
    /// Create a new vignette effect.
    #[must_use]
    pub fn new(config: VignetteConfig) -> Self {
        Self { config }
    }

    /// Apply the vignette effect in-place.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer size is incorrect.
    #[allow(clippy::cast_precision_loss)]
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

        let cx = cfg.center_x * width as f32;
        let cy = cfg.center_y * height as f32;

        // Normalize by half of the shorter dimension so the radii are
        // expressed in "radius fractions" irrespective of aspect ratio.
        let ref_radius = (width.min(height) as f32) * 0.5;

        let inner = cfg.inner_radius * ref_radius;
        let outer = cfg.outer_radius * ref_radius;
        let range = (outer - inner).max(1.0);

        for py in 0..height {
            for px in 0..width {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // t = 0 inside inner radius, 1 at outer radius and beyond
                let t = ((dist - inner) / range).clamp(0.0, 1.0);

                // Compute darkening factor in [0, 1] where 0 = no dark, 1 = max dark
                let dark = match cfg.falloff {
                    VignetteFalloff::Cosine => {
                        // Smooth step using cosine: goes from 0 to 1 smoothly
                        (1.0 - (std::f32::consts::PI * t).cos()) * 0.5
                    }
                    VignetteFalloff::Power(p) => t.powf(p),
                    VignetteFalloff::Linear => t,
                };

                // Multiply factor: 1 at center, (1 - strength*dark) at edges
                let mul = 1.0 - cfg.strength * dark;

                let idx = (py * width + px) * bpp;
                data[idx] = clamp_u8(f32::from(data[idx]) * mul);
                data[idx + 1] = clamp_u8(f32::from(data[idx + 1]) * mul);
                data[idx + 2] = clamp_u8(f32::from(data[idx + 2]) * mul);
                // Alpha unchanged
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white_buf(w: usize, h: usize) -> Vec<u8> {
        vec![255u8; w * h * 3]
    }

    #[test]
    fn test_vignette_default_applies() {
        let mut buf = white_buf(64, 64);
        let v = Vignette::new(VignetteConfig::default());
        assert!(v.apply(&mut buf, 64, 64, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_vignette_center_brighter_than_corner() {
        let mut buf = white_buf(128, 128);
        let v = Vignette::new(VignetteConfig::default());
        v.apply(&mut buf, 128, 128, PixelFormat::Rgb)
            .expect("apply should succeed");

        let center_idx = (64 * 128 + 64) * 3;
        let corner_idx = 0;
        assert!(
            buf[center_idx] >= buf[corner_idx],
            "Center should be at least as bright as corner"
        );
    }

    #[test]
    fn test_vignette_zero_strength_unchanged() {
        let orig = white_buf(32, 32);
        let mut buf = orig.clone();
        let cfg = VignetteConfig {
            strength: 0.0,
            ..Default::default()
        };
        let v = Vignette::new(cfg);
        v.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "Zero strength should not change image");
    }

    #[test]
    fn test_vignette_subtle_vs_heavy() {
        let orig = white_buf(64, 64);
        let mut subtle = orig.clone();
        Vignette::new(VignetteConfig::subtle())
            .apply(&mut subtle, 64, 64, PixelFormat::Rgb)
            .expect("test expectation failed");

        let mut heavy = orig.clone();
        Vignette::new(VignetteConfig::heavy())
            .apply(&mut heavy, 64, 64, PixelFormat::Rgb)
            .expect("test expectation failed");

        // Heavy should darken the corner more
        let corner = 0usize;
        assert!(
            heavy[corner] <= subtle[corner],
            "Heavy should be darker at corner"
        );
    }

    #[test]
    fn test_vignette_power_falloff() {
        let mut buf = white_buf(32, 32);
        let cfg = VignetteConfig {
            falloff: VignetteFalloff::Power(2.0),
            ..Default::default()
        };
        let v = Vignette::new(cfg);
        assert!(v.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_vignette_linear_falloff() {
        let mut buf = white_buf(32, 32);
        let cfg = VignetteConfig {
            falloff: VignetteFalloff::Linear,
            ..Default::default()
        };
        let v = Vignette::new(cfg);
        assert!(v.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_vignette_rgba() {
        let mut buf = vec![200u8; 32 * 32 * 4];
        let v = Vignette::new(VignetteConfig::default());
        assert!(v.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_vignette_wrong_size_err() {
        let mut buf = vec![0u8; 5];
        let v = Vignette::new(VignetteConfig::default());
        assert!(v.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }
}
