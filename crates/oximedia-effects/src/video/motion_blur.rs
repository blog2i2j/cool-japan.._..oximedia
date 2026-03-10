//! Motion blur effect via directional convolution.
//!
//! Implements directional (linear) motion blur by building a 1D convolution kernel
//! oriented at the specified angle and applying it to the image. The kernel length
//! controls the blur distance. A box kernel gives a film-exposure-style blur.

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// Configuration for motion blur.
#[derive(Debug, Clone)]
pub struct MotionBlurConfig {
    /// Blur direction in degrees (0 = horizontal right, 90 = downward).
    pub angle_degrees: f32,
    /// Number of samples / kernel length in pixels.
    /// More samples = more blur.
    pub samples: usize,
}

impl Default for MotionBlurConfig {
    fn default() -> Self {
        Self {
            angle_degrees: 0.0,
            samples: 15,
        }
    }
}

impl MotionBlurConfig {
    /// Horizontal blur (panning right).
    #[must_use]
    pub const fn horizontal(samples: usize) -> Self {
        Self {
            angle_degrees: 0.0,
            samples,
        }
    }

    /// Vertical blur (vertical pan or drop).
    #[must_use]
    pub const fn vertical(samples: usize) -> Self {
        Self {
            angle_degrees: 90.0,
            samples,
        }
    }

    /// Diagonal blur.
    #[must_use]
    pub const fn diagonal(samples: usize) -> Self {
        Self {
            angle_degrees: 45.0,
            samples,
        }
    }
}

/// Motion blur effect using directional box-kernel convolution.
pub struct MotionBlur {
    config: MotionBlurConfig,
}

impl MotionBlur {
    /// Create a new motion blur effect.
    #[must_use]
    pub fn new(config: MotionBlurConfig) -> Self {
        Self { config }
    }

    /// Apply motion blur to a pixel buffer in-place.
    ///
    /// The algorithm builds a set of `samples` offset vectors along the blur direction
    /// and averages the pixel values at those positions (box filter). Bilinear sampling
    /// is used for sub-pixel precision.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer size is incorrect.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    pub fn apply(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> VideoResult<()> {
        validate_buffer(data, width, height, format)?;
        if self.config.samples <= 1 {
            return Ok(()); // Nothing to do
        }

        let bpp = format.bytes_per_pixel();
        let angle_rad = self.config.angle_degrees.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        let n = self.config.samples as f32;

        let source = data.to_vec();

        for py in 0..height {
            for px in 0..width {
                let mut acc_r = 0.0f32;
                let mut acc_g = 0.0f32;
                let mut acc_b = 0.0f32;
                let mut acc_a = 0.0f32;

                for s in 0..self.config.samples {
                    // Distribute samples symmetrically around the current pixel
                    let t = s as f32 - (n - 1.0) * 0.5;
                    let sx = px as f32 + t * cos_a;
                    let sy = py as f32 + t * sin_a;

                    let pixel = super::sample_bilinear(&source, width, height, bpp, sx, sy);
                    acc_r += pixel[0];
                    acc_g += pixel[1];
                    acc_b += pixel[2];
                    acc_a += pixel[3];
                }

                let inv_n = 1.0 / n;
                let idx = (py * width + px) * bpp;
                data[idx] = clamp_u8(acc_r * inv_n);
                data[idx + 1] = clamp_u8(acc_g * inv_n);
                data[idx + 2] = clamp_u8(acc_b * inv_n);
                if bpp == 4 {
                    data[idx + 3] = clamp_u8(acc_a * inv_n);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let val = if (px + py) % 2 == 0 { 255u8 } else { 0u8 };
                let idx = (py * w + px) * 3;
                buf[idx] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
            }
        }
        buf
    }

    #[test]
    fn test_motion_blur_horizontal() {
        let orig = checkerboard(64, 64);
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig::horizontal(9));
        mb.apply(&mut buf, 64, 64, PixelFormat::Rgb)
            .expect("apply should succeed");
        // Blur should reduce sharp transitions; pixel values should be more uniform
        let diff: u32 = buf
            .iter()
            .zip(orig.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();
        assert!(diff > 0, "Blur should change the image");
    }

    #[test]
    fn test_motion_blur_samples_1_no_change() {
        let orig = checkerboard(32, 32);
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig {
            samples: 1,
            ..Default::default()
        });
        mb.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "1 sample should not change image");
    }

    #[test]
    fn test_motion_blur_vertical() {
        let mut buf = checkerboard(32, 32);
        let mb = MotionBlur::new(MotionBlurConfig::vertical(7));
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_motion_blur_diagonal() {
        let mut buf = checkerboard(32, 32);
        let mb = MotionBlur::new(MotionBlurConfig::diagonal(7));
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_motion_blur_rgba() {
        let mut buf = vec![200u8; 32 * 32 * 4];
        let mb = MotionBlur::new(MotionBlurConfig::default());
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_motion_blur_wrong_size_err() {
        let mut buf = vec![0u8; 5];
        let mb = MotionBlur::new(MotionBlurConfig::default());
        assert!(mb.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_motion_blur_constant_image_unchanged() {
        // Blurring a constant-color image gives the same color
        let orig = vec![128u8; 32 * 32 * 3];
        let mut buf = orig.clone();
        let mb = MotionBlur::new(MotionBlurConfig::horizontal(11));
        mb.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert_eq!(buf, orig, "Constant image should survive blur");
    }
}
