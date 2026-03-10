//! Color repair: desaturation fix, color cast removal, and tone restoration.

#![allow(dead_code)]

/// An RGB pixel with f32 components in the range 0.0–1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
}

impl Rgb {
    /// Create a new RGB pixel, clamping components to [0.0, 1.0].
    pub fn new(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Luma (perceived brightness) using BT.709 coefficients.
    pub fn luma(&self) -> f32 {
        0.2126 * self.r + 0.7152 * self.g + 0.0722 * self.b
    }

    /// Saturation in the HSL sense (simplified: range of components / brightness).
    pub fn saturation_estimate(&self) -> f32 {
        let max = self.r.max(self.g).max(self.b);
        let min = self.r.min(self.g).min(self.b);
        if max < f32::EPSILON {
            0.0
        } else {
            (max - min) / max
        }
    }

    /// Convert to a grey pixel (equal R, G, B at luma level).
    pub fn to_grey(&self) -> Self {
        let l = self.luma();
        Self::new(l, l, l)
    }
}

/// Detect whether a pixel is desaturated (saturation below threshold).
pub fn is_desaturated(pixel: &Rgb, threshold: f32) -> bool {
    pixel.saturation_estimate() < threshold
}

/// Restore saturation of a desaturated pixel by scaling colour channels
/// towards the luma value.
///
/// `boost` is a multiplier for saturation restoration (> 1.0 increases
/// colour separation from the luma grey point).
pub fn fix_desaturation(pixel: &Rgb, boost: f32) -> Rgb {
    let luma = pixel.luma();
    let boost = boost.max(0.0);
    Rgb::new(
        luma + (pixel.r - luma) * boost,
        luma + (pixel.g - luma) * boost,
        luma + (pixel.b - luma) * boost,
    )
}

/// Colour cast bias measured per channel.
#[derive(Debug, Clone, Copy)]
pub struct ColorCast {
    /// Bias on the red channel.
    pub r_bias: f32,
    /// Bias on the green channel.
    pub g_bias: f32,
    /// Bias on the blue channel.
    pub b_bias: f32,
}

impl ColorCast {
    /// Estimate colour cast from a sample of pixels by comparing
    /// each channel mean to the overall luma mean.
    pub fn estimate(pixels: &[Rgb]) -> Option<Self> {
        if pixels.is_empty() {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let n = pixels.len() as f32;
        let mean_r: f32 = pixels.iter().map(|p| p.r).sum::<f32>() / n;
        let mean_g: f32 = pixels.iter().map(|p| p.g).sum::<f32>() / n;
        let mean_b: f32 = pixels.iter().map(|p| p.b).sum::<f32>() / n;
        let mean_luma = 0.2126 * mean_r + 0.7152 * mean_g + 0.0722 * mean_b;
        Some(Self {
            r_bias: mean_r - mean_luma,
            g_bias: mean_g - mean_luma,
            b_bias: mean_b - mean_luma,
        })
    }

    /// Dominant cast channel as a label.
    pub fn dominant_channel(&self) -> &'static str {
        let r = self.r_bias.abs();
        let g = self.g_bias.abs();
        let b = self.b_bias.abs();
        if r >= g && r >= b {
            "red"
        } else if g >= r && g >= b {
            "green"
        } else {
            "blue"
        }
    }
}

/// Remove a colour cast from a pixel by subtracting the bias.
pub fn remove_color_cast(pixel: &Rgb, cast: &ColorCast) -> Rgb {
    Rgb::new(
        pixel.r - cast.r_bias,
        pixel.g - cast.g_bias,
        pixel.b - cast.b_bias,
    )
}

/// Tone curve type for tone restoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneCurve {
    /// Lift shadows to compensate for underexposure.
    LiftShadows,
    /// Compress highlights to recover blown-out areas.
    CompressHighlights,
    /// Apply gamma correction.
    GammaCorrect,
    /// Linear pass-through (identity).
    Identity,
}

/// Apply a tone restoration curve to a single normalised value (0.0–1.0).
pub fn apply_tone_curve(value: f32, curve: ToneCurve, strength: f32) -> f32 {
    let v = value.clamp(0.0, 1.0);
    let s = strength.clamp(0.0, 1.0);
    match curve {
        ToneCurve::Identity => v,
        ToneCurve::LiftShadows => {
            // Raise dark values; shadows lifted by up to `s * 0.3`.
            let lift = s * 0.3 * (1.0 - v);
            (v + lift).clamp(0.0, 1.0)
        }
        ToneCurve::CompressHighlights => {
            // Roll off bright values; highlights pulled down by `s * 0.2`.
            let compress = s * 0.2 * v;
            (v - compress).clamp(0.0, 1.0)
        }
        ToneCurve::GammaCorrect => {
            // Blend between v and v^(1/2.2) by strength.
            let gamma = v.powf(1.0 / 2.2);
            (v * (1.0 - s) + gamma * s).clamp(0.0, 1.0)
        }
    }
}

/// Apply tone restoration to all pixels in a buffer (in-place).
pub fn restore_tones(pixels: &mut [Rgb], curve: ToneCurve, strength: f32) {
    for p in pixels.iter_mut() {
        p.r = apply_tone_curve(p.r, curve, strength);
        p.g = apply_tone_curve(p.g, curve, strength);
        p.b = apply_tone_curve(p.b, curve, strength);
    }
}

/// White-balance correction using a reference "grey" pixel.
///
/// Scales each channel so that the reference becomes neutral grey.
pub fn white_balance(pixels: &mut [Rgb], reference: &Rgb) {
    let target_luma = reference.luma();
    let scale_r = if reference.r > f32::EPSILON {
        target_luma / reference.r
    } else {
        1.0
    };
    let scale_g = if reference.g > f32::EPSILON {
        target_luma / reference.g
    } else {
        1.0
    };
    let scale_b = if reference.b > f32::EPSILON {
        target_luma / reference.b
    } else {
        1.0
    };
    for p in pixels.iter_mut() {
        p.r = (p.r * scale_r).clamp(0.0, 1.0);
        p.g = (p.g * scale_g).clamp(0.0, 1.0);
        p.b = (p.b * scale_b).clamp(0.0, 1.0);
    }
}

// ---------------------------------------------------------------------------
// New types: u8-based ColorCast, FadedFilm, WhiteBalanceCorrection
// ---------------------------------------------------------------------------

/// Colour cast expressed as channel shifts (positive = channel is boosted).
///
/// This variant operates on raw `u8` pixel buffers (interleaved RGB: R,G,B,R,G,B,...).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ColorCastU8 {
    /// Red channel shift (mean deviation from overall mean).
    pub red_shift: f32,
    /// Green channel shift.
    pub green_shift: f32,
    /// Blue channel shift.
    pub blue_shift: f32,
}

impl ColorCastU8 {
    /// Overall magnitude of the colour cast.
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.red_shift.powi(2) + self.green_shift.powi(2) + self.blue_shift.powi(2)).sqrt()
    }

    /// Returns the name of the dominant (largest absolute shift) channel.
    #[must_use]
    pub fn dominant_channel(&self) -> &'static str {
        let r = self.red_shift.abs();
        let g = self.green_shift.abs();
        let b = self.blue_shift.abs();
        if r >= g && r >= b {
            "red"
        } else if g >= r && g >= b {
            "green"
        } else {
            "blue"
        }
    }
}

/// Estimate the colour cast from an interleaved RGB `u8` pixel slice.
///
/// Calculates the mean of each channel and computes each channel's deviation
/// from the overall mean.  Returns `None` if `pixels` is empty or not a
/// multiple of 3.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_color_cast(pixels: &[u8]) -> Option<ColorCastU8> {
    if pixels.is_empty() || pixels.len() % 3 != 0 {
        return None;
    }
    let n = (pixels.len() / 3) as f32;
    let mut sum_r = 0u64;
    let mut sum_g = 0u64;
    let mut sum_b = 0u64;
    for chunk in pixels.chunks_exact(3) {
        sum_r += chunk[0] as u64;
        sum_g += chunk[1] as u64;
        sum_b += chunk[2] as u64;
    }
    let mean_r = sum_r as f32 / n;
    let mean_g = sum_g as f32 / n;
    let mean_b = sum_b as f32 / n;
    let overall_mean = (mean_r + mean_g + mean_b) / 3.0;
    Some(ColorCastU8 {
        red_shift: mean_r - overall_mean,
        green_shift: mean_g - overall_mean,
        blue_shift: mean_b - overall_mean,
    })
}

/// Remove a colour cast from an interleaved RGB `u8` pixel buffer in-place.
///
/// Each channel value has the corresponding shift subtracted, then clamped to
/// `[0, 255]`.
pub fn remove_color_cast_u8(pixels: &mut [u8], cast: &ColorCastU8) {
    for chunk in pixels.chunks_exact_mut(3) {
        let r = (chunk[0] as f32 - cast.red_shift).clamp(0.0, 255.0) as u8;
        let g = (chunk[1] as f32 - cast.green_shift).clamp(0.0, 255.0) as u8;
        let b = (chunk[2] as f32 - cast.blue_shift).clamp(0.0, 255.0) as u8;
        chunk[0] = r;
        chunk[1] = g;
        chunk[2] = b;
    }
}

/// Parameters describing the degradation of faded film.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct FadedFilm {
    /// Additive brightness lift in the [0, 255] range (emulates base fog).
    pub luminance_shift: f32,
    /// Fraction of contrast lost in [0.0, 1.0] (0 = no loss, 1 = flat).
    pub contrast_loss: f32,
    /// Colour bleed amount in [0.0, 1.0] (cross-channel contamination).
    pub color_bleed: f32,
}

impl FadedFilm {
    /// Overall severity score in [0.0, 1.0].
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn severity(&self) -> f32 {
        let lum_norm = (self.luminance_shift / 255.0).clamp(0.0, 1.0);
        ((lum_norm + self.contrast_loss.clamp(0.0, 1.0) + self.color_bleed.clamp(0.0, 1.0)) / 3.0)
            .clamp(0.0, 1.0)
    }
}

/// Restore a faded-film appearance by expanding contrast, reducing
/// luminance lift, and suppressing colour bleed.
///
/// Operates in-place on an interleaved RGB `u8` pixel buffer.
#[allow(clippy::cast_precision_loss)]
pub fn restore_faded_film(pixels: &mut [u8], fade: &FadedFilm) {
    let contrast_gain = if fade.contrast_loss >= 1.0 {
        1.0f32
    } else {
        1.0 / (1.0 - fade.contrast_loss.clamp(0.0, 0.99))
    };
    let bleed_inv = 1.0 - fade.color_bleed.clamp(0.0, 1.0);

    for chunk in pixels.chunks_exact_mut(3) {
        let r = chunk[0] as f32;
        let g = chunk[1] as f32;
        let b = chunk[2] as f32;

        // Remove luminance lift.
        let r2 = (r - fade.luminance_shift).max(0.0);
        let g2 = (g - fade.luminance_shift).max(0.0);
        let b2 = (b - fade.luminance_shift).max(0.0);

        // Expand contrast (scale around 128).
        let mid = 128.0f32;
        let r3 = ((r2 - mid) * contrast_gain + mid).clamp(0.0, 255.0);
        let g3 = ((g2 - mid) * contrast_gain + mid).clamp(0.0, 255.0);
        let b3 = ((b2 - mid) * contrast_gain + mid).clamp(0.0, 255.0);

        // Reduce colour bleed: pull each channel toward its own value (bleed_inv).
        let luma = 0.2126 * r3 + 0.7152 * g3 + 0.0722 * b3;
        let r4 = (luma + (r3 - luma) * bleed_inv).clamp(0.0, 255.0) as u8;
        let g4 = (luma + (g3 - luma) * bleed_inv).clamp(0.0, 255.0) as u8;
        let b4 = (luma + (b3 - luma) * bleed_inv).clamp(0.0, 255.0) as u8;

        chunk[0] = r4;
        chunk[1] = g4;
        chunk[2] = b4;
    }
}

/// White-balance utilities.
pub struct WhiteBalanceCorrection;

impl WhiteBalanceCorrection {
    /// Compute per-channel gain factors using the grey-world assumption.
    ///
    /// The grey-world algorithm assumes the average colour of a scene is grey,
    /// so it scales each channel so that all three channel means are equal.
    ///
    /// Returns `(gain_r, gain_g, gain_b)`.  Returns `(1.0, 1.0, 1.0)` if
    /// `pixels` is empty or not a multiple of 3.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn gray_world(pixels: &[u8]) -> (f32, f32, f32) {
        if pixels.is_empty() || pixels.len() % 3 != 0 {
            return (1.0, 1.0, 1.0);
        }
        let n = (pixels.len() / 3) as f32;
        let mut sum_r = 0u64;
        let mut sum_g = 0u64;
        let mut sum_b = 0u64;
        for chunk in pixels.chunks_exact(3) {
            sum_r += chunk[0] as u64;
            sum_g += chunk[1] as u64;
            sum_b += chunk[2] as u64;
        }
        let mean_r = sum_r as f32 / n;
        let mean_g = sum_g as f32 / n;
        let mean_b = sum_b as f32 / n;
        let overall_mean = (mean_r + mean_g + mean_b) / 3.0;
        let gain_r = if mean_r > f32::EPSILON {
            overall_mean / mean_r
        } else {
            1.0
        };
        let gain_g = if mean_g > f32::EPSILON {
            overall_mean / mean_g
        } else {
            1.0
        };
        let gain_b = if mean_b > f32::EPSILON {
            overall_mean / mean_b
        } else {
            1.0
        };
        (gain_r, gain_g, gain_b)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_new_clamps() {
        let p = Rgb::new(1.5, -0.1, 0.5);
        assert_eq!(p.r, 1.0);
        assert_eq!(p.g, 0.0);
        assert!((p.b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rgb_luma_grey() {
        let grey = Rgb::new(0.5, 0.5, 0.5);
        assert!((grey.luma() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_saturation_estimate_grey() {
        let grey = Rgb::new(0.5, 0.5, 0.5);
        assert!(grey.saturation_estimate() < 1e-5);
    }

    #[test]
    fn test_rgb_saturation_estimate_red() {
        let red = Rgb::new(1.0, 0.0, 0.0);
        assert!((red.saturation_estimate() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rgb_to_grey() {
        let p = Rgb::new(0.8, 0.2, 0.4);
        let g = p.to_grey();
        assert!((g.r - g.g).abs() < 1e-6);
        assert!((g.g - g.b).abs() < 1e-6);
    }

    #[test]
    fn test_is_desaturated_grey_pixel() {
        let grey = Rgb::new(0.5, 0.5, 0.5);
        assert!(is_desaturated(&grey, 0.1));
    }

    #[test]
    fn test_is_desaturated_coloured_pixel() {
        let red = Rgb::new(1.0, 0.0, 0.0);
        assert!(!is_desaturated(&red, 0.1));
    }

    #[test]
    fn test_fix_desaturation_boost_one_is_identity() {
        let p = Rgb::new(0.6, 0.3, 0.1);
        let fixed = fix_desaturation(&p, 1.0);
        assert!((fixed.r - p.r).abs() < 1e-5);
        assert!((fixed.g - p.g).abs() < 1e-5);
        assert!((fixed.b - p.b).abs() < 1e-5);
    }

    #[test]
    fn test_color_cast_estimate_neutral() {
        // Neutral grey — cast should be near zero for all channels.
        let pixels = vec![Rgb::new(0.5, 0.5, 0.5); 4];
        let cast = ColorCast::estimate(&pixels).expect("color cast estimation should succeed");
        assert!(cast.r_bias.abs() < 0.01);
        assert!(cast.g_bias.abs() < 0.01);
        assert!(cast.b_bias.abs() < 0.01);
    }

    #[test]
    fn test_color_cast_estimate_none_when_empty() {
        let cast = ColorCast::estimate(&[]);
        assert!(cast.is_none());
    }

    #[test]
    fn test_color_cast_dominant_channel_red() {
        let cast = ColorCast {
            r_bias: 0.3,
            g_bias: -0.05,
            b_bias: -0.02,
        };
        assert_eq!(cast.dominant_channel(), "red");
    }

    #[test]
    fn test_remove_color_cast_neutralises() {
        let pixel = Rgb::new(0.7, 0.5, 0.5);
        let cast = ColorCast {
            r_bias: 0.2,
            g_bias: 0.0,
            b_bias: 0.0,
        };
        let fixed = remove_color_cast(&pixel, &cast);
        assert!((fixed.r - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_apply_tone_curve_identity() {
        assert!((apply_tone_curve(0.4, ToneCurve::Identity, 1.0) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_apply_tone_curve_lift_shadows_raises_dark() {
        let lifted = apply_tone_curve(0.1, ToneCurve::LiftShadows, 1.0);
        assert!(lifted > 0.1);
    }

    #[test]
    fn test_apply_tone_curve_compress_highlights_lowers_bright() {
        let compressed = apply_tone_curve(0.9, ToneCurve::CompressHighlights, 1.0);
        assert!(compressed < 0.9);
    }

    #[test]
    fn test_restore_tones_mutates_buffer() {
        let mut pixels = vec![Rgb::new(0.1, 0.1, 0.1); 5];
        restore_tones(&mut pixels, ToneCurve::LiftShadows, 1.0);
        for p in &pixels {
            assert!(p.r > 0.1);
        }
    }

    #[test]
    fn test_white_balance_neutral_ref() {
        let reference = Rgb::new(0.5, 0.5, 0.5);
        let mut pixels = vec![Rgb::new(0.4, 0.6, 0.3)];
        // With neutral reference, luma/r = luma/g = luma/b = 1.0 → no change.
        white_balance(&mut pixels, &reference);
        // Values should be scaled but within [0, 1].
        assert!(pixels[0].r >= 0.0 && pixels[0].r <= 1.0);
    }

    // --- ColorCastU8 tests ---

    #[test]
    fn test_estimate_color_cast_empty_returns_none() {
        assert!(estimate_color_cast(&[]).is_none());
    }

    #[test]
    fn test_estimate_color_cast_not_multiple_of_3_returns_none() {
        assert!(estimate_color_cast(&[100u8, 100, 100, 100]).is_none());
    }

    #[test]
    fn test_estimate_color_cast_neutral_grey() {
        // All pixels equal → shifts should all be zero.
        let pixels = vec![128u8, 128, 128, 128, 128, 128];
        let cast = estimate_color_cast(&pixels).expect("color cast estimation should succeed");
        assert!(cast.red_shift.abs() < 0.01);
        assert!(cast.green_shift.abs() < 0.01);
        assert!(cast.blue_shift.abs() < 0.01);
    }

    #[test]
    fn test_estimate_color_cast_red_dominant() {
        // Red channel much higher than green and blue.
        let pixels = vec![200u8, 50, 50]; // single pixel with strong red
        let cast = estimate_color_cast(&pixels).expect("color cast estimation should succeed");
        assert!(cast.red_shift > 0.0);
        assert_eq!(cast.dominant_channel(), "red");
    }

    #[test]
    fn test_color_cast_u8_magnitude_zero_when_neutral() {
        let cast = ColorCastU8 {
            red_shift: 0.0,
            green_shift: 0.0,
            blue_shift: 0.0,
        };
        assert!(cast.magnitude() < f32::EPSILON);
    }

    #[test]
    fn test_color_cast_u8_dominant_channel_blue() {
        let cast = ColorCastU8 {
            red_shift: 0.0,
            green_shift: 1.0,
            blue_shift: 5.0,
        };
        assert_eq!(cast.dominant_channel(), "blue");
    }

    #[test]
    fn test_remove_color_cast_u8_shifts_channels() {
        let mut pixels = vec![200u8, 100, 100];
        let cast = ColorCastU8 {
            red_shift: 50.0,
            green_shift: 0.0,
            blue_shift: 0.0,
        };
        remove_color_cast_u8(&mut pixels, &cast);
        assert_eq!(pixels[0], 150); // 200 - 50
        assert_eq!(pixels[1], 100);
        assert_eq!(pixels[2], 100);
    }

    #[test]
    fn test_remove_color_cast_u8_clamps_at_zero() {
        let mut pixels = vec![10u8, 100, 100];
        let cast = ColorCastU8 {
            red_shift: 50.0,
            green_shift: 0.0,
            blue_shift: 0.0,
        };
        remove_color_cast_u8(&mut pixels, &cast);
        assert_eq!(pixels[0], 0); // clamped
    }

    // --- FadedFilm tests ---

    #[test]
    fn test_faded_film_severity_zero_when_no_fade() {
        let fade = FadedFilm {
            luminance_shift: 0.0,
            contrast_loss: 0.0,
            color_bleed: 0.0,
        };
        assert!(fade.severity() < f32::EPSILON);
    }

    #[test]
    fn test_faded_film_severity_max_when_full_fade() {
        let fade = FadedFilm {
            luminance_shift: 255.0,
            contrast_loss: 1.0,
            color_bleed: 1.0,
        };
        assert!((fade.severity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_restore_faded_film_reduces_luminance_shift() {
        let mut pixels = vec![200u8, 200, 200, 200, 200, 200];
        let fade = FadedFilm {
            luminance_shift: 50.0,
            contrast_loss: 0.0,
            color_bleed: 0.0,
        };
        restore_faded_film(&mut pixels, &fade);
        // After removing luminance lift, values should be lower.
        assert!(pixels[0] < 200);
    }

    // --- WhiteBalanceCorrection::gray_world tests ---

    #[test]
    fn test_gray_world_empty_returns_unit_gains() {
        let (gr, gg, gb) = WhiteBalanceCorrection::gray_world(&[]);
        assert!((gr - 1.0).abs() < f32::EPSILON);
        assert!((gg - 1.0).abs() < f32::EPSILON);
        assert!((gb - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gray_world_neutral_returns_unit_gains() {
        // All channels equal → gains should be 1.0.
        let pixels = vec![100u8, 100, 100, 100, 100, 100];
        let (gr, gg, gb) = WhiteBalanceCorrection::gray_world(&pixels);
        assert!((gr - 1.0).abs() < 0.01);
        assert!((gg - 1.0).abs() < 0.01);
        assert!((gb - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gray_world_red_dominant_reduces_red_gain() {
        // Red is much higher → gain_r should be < 1.0.
        let pixels = vec![200u8, 100, 100];
        let (gr, gg, gb) = WhiteBalanceCorrection::gray_world(&pixels);
        assert!(gr < 1.0);
        assert!(gg >= 1.0);
        assert!(gb >= 1.0);
    }
}
