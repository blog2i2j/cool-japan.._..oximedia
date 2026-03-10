//! 3-way color grading (lift/gamma/gain).
//!
//! Implements a professional-grade color correction workflow using the
//! lift/gamma/gain model as used in `DaVinci` Resolve, Baselight, and other
//! professional color grading tools.
//!
//! The mathematical formulation per-channel is:
//!
//! ```text
//! normalized = pixel / 255.0
//! lifted    = normalized * (1 - lift) + lift        // offset black point
//! gained    = lifted * gain                         // scale white point
//! corrected = gained ^ (1 / gamma)                 // gamma curve
//! output    = clamp(corrected * 255, 0, 255)
//! ```
//!
//! Lift, gamma, and gain are 3-vectors `[r, g, b]` allowing independent
//! control of shadows (lift), midtones (gamma), and highlights (gain) per channel.

use super::{clamp_u8, validate_buffer, PixelFormat, VideoResult};

/// A 3-channel (R, G, B) color vector for lift/gamma/gain control.
#[derive(Debug, Clone, Copy)]
pub struct ColorVec {
    /// Red component.
    pub r: f32,
    /// Green component.
    pub g: f32,
    /// Blue component.
    pub b: f32,
}

impl ColorVec {
    /// Create a new color vector.
    #[must_use]
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b }
    }

    /// Neutral/identity value for lift (0.0), gamma (1.0), gain (1.0).
    #[must_use]
    pub const fn neutral_lift() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// Neutral gamma.
    #[must_use]
    pub const fn neutral_gamma() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    /// Neutral gain.
    #[must_use]
    pub const fn neutral_gain() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl Default for ColorVec {
    fn default() -> Self {
        Self::neutral_gamma()
    }
}

/// Lift/Gamma/Gain color grading parameters.
#[derive(Debug, Clone, Copy)]
pub struct LiftGammaGain {
    /// Lift (shadow offset). 0.0 = neutral, positive = lift shadows.
    pub lift: ColorVec,
    /// Gamma (midtone power curve). 1.0 = neutral, <1 = brighten, >1 = darken.
    pub gamma: ColorVec,
    /// Gain (highlight multiplier). 1.0 = neutral, >1 = brighter.
    pub gain: ColorVec,
}

impl Default for LiftGammaGain {
    fn default() -> Self {
        Self {
            lift: ColorVec::neutral_lift(),
            gamma: ColorVec::neutral_gamma(),
            gain: ColorVec::neutral_gain(),
        }
    }
}

impl LiftGammaGain {
    /// Apply lift/gamma/gain to a single normalized value.
    #[inline]
    #[must_use]
    pub fn apply_channel(val: f32, lift: f32, gamma: f32, gain: f32) -> f32 {
        // Step 1: Lift (shadow offset)
        let lifted = val * (1.0 - lift) + lift;
        // Step 2: Gain (highlight scale)
        let gained = lifted * gain;
        // Step 3: Gamma (power curve – use safe version that handles <= 0)
        let gamma_clamped = gamma.max(0.001);
        gained.max(0.0).powf(1.0 / gamma_clamped)
    }
}

/// Configuration for the `ColorGrade` effect.
#[derive(Debug, Clone)]
pub struct ColorGradeConfig {
    /// Lift/gamma/gain control.
    pub lgg: LiftGammaGain,
    /// Saturation (1.0 = neutral, 0.0 = desaturate, >1.0 = boost).
    pub saturation: f32,
    /// Overall contrast (1.0 = neutral). Applied as S-curve pivot at 0.5.
    pub contrast: f32,
    /// Overall brightness offset (0.0 = neutral, in normalized \[0,1\] space).
    pub brightness: f32,
}

impl Default for ColorGradeConfig {
    fn default() -> Self {
        Self {
            lgg: LiftGammaGain::default(),
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
        }
    }
}

impl ColorGradeConfig {
    /// Warm cinematic grade (lifted shadows, warm highlights).
    #[must_use]
    pub fn warm_cinematic() -> Self {
        Self {
            lgg: LiftGammaGain {
                lift: ColorVec::new(0.05, 0.03, 0.0),
                gamma: ColorVec::new(1.05, 1.0, 0.95),
                gain: ColorVec::new(1.1, 1.05, 0.9),
            },
            saturation: 1.1,
            contrast: 1.05,
            brightness: 0.0,
        }
    }

    /// Cool teal-and-orange (blockbuster look).
    #[must_use]
    pub fn teal_orange() -> Self {
        Self {
            lgg: LiftGammaGain {
                lift: ColorVec::new(0.0, 0.05, 0.08),
                gamma: ColorVec::new(1.05, 1.0, 0.95),
                gain: ColorVec::new(1.15, 1.0, 0.85),
            },
            saturation: 1.2,
            contrast: 1.1,
            brightness: 0.0,
        }
    }

    /// Bleach bypass (desaturated, high contrast).
    #[must_use]
    pub fn bleach_bypass() -> Self {
        Self {
            lgg: LiftGammaGain::default(),
            saturation: 0.4,
            contrast: 1.4,
            brightness: 0.0,
        }
    }
}

/// 3-way color grading effect processor.
pub struct ColorGrade {
    config: ColorGradeConfig,
}

impl ColorGrade {
    /// Create a new color grade effect.
    #[must_use]
    pub fn new(config: ColorGradeConfig) -> Self {
        Self { config }
    }

    /// Apply color grading to a pixel buffer in-place.
    ///
    /// Processing order:
    /// 1. Lift/gamma/gain per channel
    /// 2. Brightness offset
    /// 3. Contrast (S-curve around 0.5)
    /// 4. Saturation
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
        let lgg = &cfg.lgg;

        for py in 0..height {
            for px in 0..width {
                let idx = (py * width + px) * bpp;

                let r_in = f32::from(data[idx]) / 255.0;
                let g_in = f32::from(data[idx + 1]) / 255.0;
                let b_in = f32::from(data[idx + 2]) / 255.0;

                // 1. Lift / Gamma / Gain
                let r = LiftGammaGain::apply_channel(r_in, lgg.lift.r, lgg.gamma.r, lgg.gain.r);
                let g = LiftGammaGain::apply_channel(g_in, lgg.lift.g, lgg.gamma.g, lgg.gain.g);
                let b = LiftGammaGain::apply_channel(b_in, lgg.lift.b, lgg.gamma.b, lgg.gain.b);

                // 2. Brightness
                let r = (r + cfg.brightness).clamp(0.0, 1.0);
                let g = (g + cfg.brightness).clamp(0.0, 1.0);
                let b = (b + cfg.brightness).clamp(0.0, 1.0);

                // 3. Contrast – scale around midpoint 0.5
                let c = cfg.contrast;
                let r = ((r - 0.5) * c + 0.5).clamp(0.0, 1.0);
                let g = ((g - 0.5) * c + 0.5).clamp(0.0, 1.0);
                let b = ((b - 0.5) * c + 0.5).clamp(0.0, 1.0);

                // 4. Saturation – convert to luma, blend
                // Using Rec. 709 luma weights
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                let r = (luma + (r - luma) * cfg.saturation).clamp(0.0, 1.0);
                let g = (luma + (g - luma) * cfg.saturation).clamp(0.0, 1.0);
                let b = (luma + (b - luma) * cfg.saturation).clamp(0.0, 1.0);

                data[idx] = clamp_u8(r * 255.0);
                data[idx + 1] = clamp_u8(g * 255.0);
                data[idx + 2] = clamp_u8(b * 255.0);
                // Alpha unchanged
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_ramp(w: usize, h: usize) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 3];
        for py in 0..h {
            for px in 0..w {
                let val = (px * 255 / w) as u8;
                let idx = (py * w + px) * 3;
                buf[idx] = val;
                buf[idx + 1] = val;
                buf[idx + 2] = val;
            }
        }
        buf
    }

    #[test]
    fn test_color_grade_identity() {
        let orig = gray_ramp(32, 32);
        let mut buf = orig.clone();
        let cg = ColorGrade::new(ColorGradeConfig::default());
        cg.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        // Identity should leave image unchanged (within 1 LSB rounding)
        for (a, b) in buf.iter().zip(orig.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 1,
                "Identity grade should not change image"
            );
        }
    }

    #[test]
    fn test_color_grade_warm_cinematic() {
        let mut buf = gray_ramp(32, 32);
        let cg = ColorGrade::new(ColorGradeConfig::warm_cinematic());
        assert!(cg.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_color_grade_teal_orange() {
        let mut buf = gray_ramp(32, 32);
        let cg = ColorGrade::new(ColorGradeConfig::teal_orange());
        assert!(cg.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_ok());
    }

    #[test]
    fn test_color_grade_bleach_bypass() {
        let mut buf = gray_ramp(32, 32);
        let cg = ColorGrade::new(ColorGradeConfig::bleach_bypass());
        cg.apply(&mut buf, 32, 32, PixelFormat::Rgb)
            .expect("apply should succeed");
        // After bleach bypass, R==G==B for a gray input (saturation only)
        for px in buf.chunks_exact(3) {
            // Some small diff is ok due to floating point
            assert!((px[0] as i32 - px[1] as i32).abs() <= 2);
        }
    }

    #[test]
    fn test_color_grade_lift_brightens_shadows() {
        let black_px = vec![0u8; 1 * 1 * 3];
        let mut buf = black_px.clone();
        let cfg = ColorGradeConfig {
            lgg: LiftGammaGain {
                lift: ColorVec::new(0.2, 0.2, 0.2),
                ..Default::default()
            },
            ..Default::default()
        };
        let cg = ColorGrade::new(cfg);
        cg.apply(&mut buf, 1, 1, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert!(buf[0] > 0, "Lift should raise black level");
    }

    #[test]
    fn test_color_grade_gain_brightens_highlights() {
        let white_px = vec![200u8; 1 * 1 * 3];
        let mut buf = white_px.clone();
        let cfg = ColorGradeConfig {
            lgg: LiftGammaGain {
                gain: ColorVec::new(1.5, 1.5, 1.5),
                ..Default::default()
            },
            ..Default::default()
        };
        let cg = ColorGrade::new(cfg);
        cg.apply(&mut buf, 1, 1, PixelFormat::Rgb)
            .expect("apply should succeed");
        assert!(buf[0] >= 200, "Gain should not reduce highlights");
    }

    #[test]
    fn test_color_grade_desaturation() {
        // Start with colored pixel
        let mut buf = vec![200u8, 100u8, 50u8];
        let cfg = ColorGradeConfig {
            saturation: 0.0,
            ..Default::default()
        };
        let cg = ColorGrade::new(cfg);
        cg.apply(&mut buf, 1, 1, PixelFormat::Rgb)
            .expect("apply should succeed");
        // All channels should be equal (grayscale)
        assert!((buf[0] as i32 - buf[1] as i32).abs() <= 1);
        assert!((buf[1] as i32 - buf[2] as i32).abs() <= 1);
    }

    #[test]
    fn test_color_grade_rgba() {
        let mut buf = vec![128u8; 32 * 32 * 4];
        let cg = ColorGrade::new(ColorGradeConfig::default());
        assert!(cg.apply(&mut buf, 32, 32, PixelFormat::Rgba).is_ok());
    }

    #[test]
    fn test_color_grade_wrong_size_err() {
        let mut buf = vec![0u8; 5];
        let cg = ColorGrade::new(ColorGradeConfig::default());
        assert!(cg.apply(&mut buf, 32, 32, PixelFormat::Rgb).is_err());
    }

    #[test]
    fn test_lift_gamma_gain_apply_channel() {
        // Identity: lift=0, gamma=1, gain=1
        let out = LiftGammaGain::apply_channel(0.5, 0.0, 1.0, 1.0);
        assert!((out - 0.5).abs() < 0.001);

        // Lift: input 0.0 should be lifted to 0.2
        let out = LiftGammaGain::apply_channel(0.0, 0.2, 1.0, 1.0);
        assert!((out - 0.2).abs() < 0.001);

        // Gamma > 1 uses exponent 1/gamma < 1, which brightens midtones (sqrt-like).
        let out_bright = LiftGammaGain::apply_channel(0.5, 0.0, 2.0, 1.0);
        assert!(
            out_bright > 0.5,
            "gamma=2 should brighten midtones (sqrt curve)"
        );
        // Gamma < 1 uses exponent 1/gamma > 1, which darkens midtones (square-like).
        let out_dark = LiftGammaGain::apply_channel(0.5, 0.0, 0.5, 1.0);
        assert!(
            out_dark < 0.5,
            "gamma=0.5 should darken midtones (square curve)"
        );
    }
}
