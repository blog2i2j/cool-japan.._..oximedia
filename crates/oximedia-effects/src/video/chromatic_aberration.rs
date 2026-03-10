//! Chromatic aberration effect.
//!
//! Simulates optical chromatic aberration by separating the RGB channels and
//! applying different radial displacements to each. The red channel is pushed
//! outward, the blue channel inward (or vice versa), and green stays near center.
//! This matches the optical behavior of refractive lens systems.

use super::{clamp_u8, sample_bilinear, validate_buffer, PixelFormat, VideoResult};

/// Configuration for chromatic aberration.
#[derive(Debug, Clone)]
pub struct ChromaticAberrationConfig {
    /// Red channel radial offset (positive = expand outward).
    /// Typical range: 0.0–0.02 (fraction of image diagonal).
    pub red_offset: f32,
    /// Green channel radial offset (usually near 0).
    pub green_offset: f32,
    /// Blue channel radial offset (negative = contract inward when red expands).
    /// Typical range: -0.02–0.0.
    pub blue_offset: f32,
    /// Center X for aberration origin (0.0–1.0).
    pub center_x: f32,
    /// Center Y for aberration origin (0.0–1.0).
    pub center_y: f32,
}

impl Default for ChromaticAberrationConfig {
    fn default() -> Self {
        Self {
            red_offset: 0.005,
            green_offset: 0.0,
            blue_offset: -0.005,
            center_x: 0.5,
            center_y: 0.5,
        }
    }
}

impl ChromaticAberrationConfig {
    /// Subtle aberration preset.
    #[must_use]
    pub fn subtle() -> Self {
        Self {
            red_offset: 0.002,
            blue_offset: -0.002,
            ..Default::default()
        }
    }

    /// Strong aberration preset (damaged lens / lo-fi effect).
    #[must_use]
    pub fn strong() -> Self {
        Self {
            red_offset: 0.015,
            blue_offset: -0.015,
            ..Default::default()
        }
    }
}

/// Chromatic aberration effect processor.
pub struct ChromaticAberration {
    config: ChromaticAberrationConfig,
}

impl ChromaticAberration {
    /// Create a new chromatic aberration effect.
    #[must_use]
    pub fn new(config: ChromaticAberrationConfig) -> Self {
        Self { config }
    }

    /// Apply chromatic aberration to a pixel buffer in-place.
    ///
    /// The algorithm reads from a copy of the original buffer and writes the
    /// channel-separated result back. Each channel is sampled from a radially
    /// displaced position relative to the configured center point.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer dimensions are inconsistent.
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
        let diag = ((width * width + height * height) as f32).sqrt();

        let source = data.to_vec(); // read-only copy

        for py in 0..height {
            for px in 0..width {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let dist = (dx * dx + dy * dy).sqrt();

                // Unit direction vector (avoid divide by zero at center)
                let (ux, uy) = if dist < 0.5 {
                    (0.0, 0.0)
                } else {
                    (dx / dist, dy / dist)
                };

                let idx = (py * width + px) * bpp;

                // Red channel – pushed outward
                let roffset_px = cfg.red_offset * diag;
                let rx = px as f32 + ux * roffset_px;
                let ry = py as f32 + uy * roffset_px;
                let r_sample = sample_bilinear(&source, width, height, bpp, rx, ry);

                // Green channel – no or minimal shift
                let goffset_px = cfg.green_offset * diag;
                let gx = px as f32 + ux * goffset_px;
                let gy = py as f32 + uy * goffset_px;
                let g_sample = sample_bilinear(&source, width, height, bpp, gx, gy);

                // Blue channel – pulled inward
                let boffset_px = cfg.blue_offset * diag;
                let bx = px as f32 + ux * boffset_px;
                let by = py as f32 + uy * boffset_px;
                let b_sample = sample_bilinear(&source, width, height, bpp, bx, by);

                data[idx] = clamp_u8(r_sample[0]);
                data[idx + 1] = clamp_u8(g_sample[1]);
                data[idx + 2] = clamp_u8(b_sample[2]);

                if bpp == 4 {
                    // Use original alpha unchanged
                    // (alpha stays as written in source)
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_buf(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) * 3;
                buf[idx] = (px * 255 / w) as u8;
                buf[idx + 1] = (py * 255 / h) as u8;
                buf[idx + 2] = 128;
            }
        }
        buf
    }

    #[test]
    fn test_chromatic_default_applies() {
        let mut buf = gradient_buf(64, 64);
        let ca = ChromaticAberration::new(ChromaticAberrationConfig::default());
        assert!(ca.apply(&mut buf, 64, 64, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_chromatic_zero_offset_unchanged() {
        let orig = gradient_buf(32, 32);
        let mut buf = orig.clone();
        let cfg = ChromaticAberrationConfig {
            red_offset: 0.0,
            green_offset: 0.0,
            blue_offset: 0.0,
            ..Default::default()
        };
        let ca = ChromaticAberration::new(cfg);
        ca.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "Zero offsets should leave image unchanged");
    }

    #[test]
    fn test_chromatic_strong_differs_from_subtle() {
        let orig = gradient_buf(64, 64);

        let mut buf_subtle = orig.clone();
        ChromaticAberration::new(ChromaticAberrationConfig::subtle())
            .apply(&mut buf_subtle, 64, 64, PixelFormat::Rgb)
            .expect("test expectation failed");

        let mut buf_strong = orig.clone();
        ChromaticAberration::new(ChromaticAberrationConfig::strong())
            .apply(&mut buf_strong, 64, 64, PixelFormat::Rgb)
            .expect("test expectation failed");

        let diff: u32 = buf_subtle
            .iter()
            .zip(buf_strong.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();
        assert!(diff > 0, "Strong should differ from subtle");
    }

    #[test]
    fn test_chromatic_rgba() {
        let mut buf = vec![128u8; 32 * 32 * 4];
        let ca = ChromaticAberration::new(ChromaticAberrationConfig::default());
        assert!(ca.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_chromatic_wrong_size_err() {
        let mut buf = vec![0u8; 10];
        let ca = ChromaticAberration::new(ChromaticAberrationConfig::default());
        assert!(ca.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }
}
