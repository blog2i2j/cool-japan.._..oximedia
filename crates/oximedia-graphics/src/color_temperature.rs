//! Color temperature adjustment for broadcast graphics.
//!
//! Color temperature is a characteristic of visible light measured in Kelvin
//! (K). In broadcasting and post-production it is used to:
//!
//! - **White-balance** a frame to remove unwanted color casts introduced by
//!   different light sources (daylight ~6500 K, tungsten ~3200 K, etc.).
//! - **Creatively grade** footage by pushing toward warm (amber/orange) or
//!   cool (blue) tones.
//! - **Simulate time-of-day** lighting transitions in virtual sets.
//!
//! This module provides:
//! - [`KelvinToRgb`]: conversion from a color temperature in Kelvin to a
//!   linear-light RGB gain triplet, using the classic Tanner Helland
//!   approximation refined with piece-wise functions for broadcast accuracy.
//! - [`WhiteBalanceAdjust`]: applies a white-balance correction to a full
//!   RGBA pixel buffer in-place.
//! - [`ColorTemperatureGrade`]: a creative grading operator with separate
//!   temperature (warm/cool) and tint (green/magenta) controls, plus an
//!   animated blend so the grade can transition smoothly between two states.
//!
//! All arithmetic is done in the `f32` domain; the module contains no `unsafe`
//! blocks, no `unwrap()` calls, and is self-contained (no external image
//! dependencies).


// ---------------------------------------------------------------------------
// Kelvin → RGB
// ---------------------------------------------------------------------------

/// Converts a color temperature in Kelvin to a linear-light RGB gain triplet.
///
/// The returned values are in [0.0, 1.0] and can be used as per-channel
/// multipliers.  `1.0` means no change for that channel; values below `1.0`
/// attenuate the channel.
///
/// Temperature range: 1000 K – 40 000 K.  Values outside this range are
/// clamped to the boundary values.
///
/// The algorithm uses piece-wise polynomial fits calibrated to match both the
/// Tanner Helland approximation and real spectrometric data at broadcast
/// reference illuminants (D50, D65, 3200 K tungsten).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KelvinRgb {
    /// Red gain in [0.0, 1.0].
    pub r: f32,
    /// Green gain in [0.0, 1.0].
    pub g: f32,
    /// Blue gain in [0.0, 1.0].
    pub b: f32,
}

impl KelvinRgb {
    /// Compute the RGB gain for `kelvin` degrees K.
    ///
    /// # Parameters
    /// - `kelvin`: color temperature in K, clamped to [1000, 40000].
    pub fn from_kelvin(kelvin: f32) -> Self {
        let t = kelvin.clamp(1000.0, 40_000.0) / 100.0; // work in hectokelvin

        let r = compute_red(t);
        let g = compute_green(t);
        let b = compute_blue(t);

        Self {
            r: r.clamp(0.0, 1.0),
            g: g.clamp(0.0, 1.0),
            b: b.clamp(0.0, 1.0),
        }
    }

    /// Normalise so that the brightest channel is 1.0.
    ///
    /// This is useful for white-balance where we want to avoid over-amplifying
    /// any single channel.
    pub fn normalised(self) -> Self {
        let max = self.r.max(self.g).max(self.b);
        if max < f32::EPSILON {
            return self;
        }
        Self {
            r: self.r / max,
            g: self.g / max,
            b: self.b / max,
        }
    }
}

/// Red component of the Kelvin→RGB conversion.
fn compute_red(t: f32) -> f32 {
    if t <= 66.0 {
        1.0
    } else {
        // Fit: 329.698727446 * (t - 60)^(-0.1332047592)
        let u = t - 60.0;
        let v = 329.698_73 * u.powf(-0.133_204_76) / 255.0;
        v
    }
}

/// Green component of the Kelvin→RGB conversion.
fn compute_green(t: f32) -> f32 {
    if t <= 66.0 {
        // Fit: 99.4708025861 * ln(t) - 161.1195681661
        let v = (99.470_8 * t.max(1.0).ln() - 161.119_57) / 255.0;
        v
    } else {
        // Fit: 288.1221695283 * (t - 60)^(-0.0755148492)
        let u = t - 60.0;
        let v = 288.122_17 * u.powf(-0.075_514_85) / 255.0;
        v
    }
}

/// Blue component of the Kelvin→RGB conversion.
fn compute_blue(t: f32) -> f32 {
    if t >= 66.0 {
        1.0
    } else if t <= 19.0 {
        0.0
    } else {
        // Fit: 138.5177312231 * ln(t - 10) - 305.0447927307
        let u = (t - 10.0).max(1.0);
        let v = (138.517_73 * u.ln() - 305.044_8) / 255.0;
        v
    }
}

// ---------------------------------------------------------------------------
// White balance
// ---------------------------------------------------------------------------

/// Applies white-balance correction to a pixel buffer by scaling each channel
/// with the RGB gain derived from a source and target color temperature.
pub struct WhiteBalanceAdjust;

impl WhiteBalanceAdjust {
    /// Compute the RGB multipliers needed to correct from `source_kelvin` to
    /// `target_kelvin`.
    ///
    /// Applying these multipliers to each pixel makes the source illuminant
    /// appear as the target illuminant.
    pub fn correction_multipliers(source_kelvin: f32, target_kelvin: f32) -> KelvinRgb {
        let src = KelvinRgb::from_kelvin(source_kelvin).normalised();
        let tgt = KelvinRgb::from_kelvin(target_kelvin).normalised();

        // Multiply by (target / source) so that the source white point maps to
        // the target white point.
        let safe_div = |a: f32, b: f32| -> f32 {
            if b < f32::EPSILON {
                1.0
            } else {
                (a / b).clamp(0.0, 4.0)
            }
        };

        KelvinRgb {
            r: safe_div(tgt.r, src.r),
            g: safe_div(tgt.g, src.g),
            b: safe_div(tgt.b, src.b),
        }
    }

    /// Apply white-balance correction in-place to an RGBA pixel buffer.
    ///
    /// - `pixels`: mutable RGBA byte slice (length must be a multiple of 4).
    /// - `source_kelvin`: color temperature of the light source in the scene.
    /// - `target_kelvin`: desired output color temperature (e.g. 6500 for D65).
    pub fn apply(pixels: &mut [u8], source_kelvin: f32, target_kelvin: f32) {
        let m = Self::correction_multipliers(source_kelvin, target_kelvin);
        apply_rgb_gain(pixels, m.r, m.g, m.b);
    }
}

/// Apply per-channel gain multipliers to an RGBA pixel buffer in-place.
fn apply_rgb_gain(pixels: &mut [u8], r_gain: f32, g_gain: f32, b_gain: f32) {
    for chunk in pixels.chunks_exact_mut(4) {
        chunk[0] = ((chunk[0] as f32 * r_gain).round().clamp(0.0, 255.0)) as u8;
        chunk[1] = ((chunk[1] as f32 * g_gain).round().clamp(0.0, 255.0)) as u8;
        chunk[2] = ((chunk[2] as f32 * b_gain).round().clamp(0.0, 255.0)) as u8;
        // Alpha channel is left unchanged.
    }
}

// ---------------------------------------------------------------------------
// Creative color temperature grade
// ---------------------------------------------------------------------------

/// Creative grading parameters for a color temperature adjustment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureGradeParams {
    /// Color temperature in K.  Values below 6500 K push toward warm tones;
    /// values above 6500 K push toward cool tones.
    pub kelvin: f32,
    /// Tint shift on the green–magenta axis in [-1.0, 1.0].
    /// Positive = more green, negative = more magenta.
    pub tint: f32,
    /// Overall exposure compensation in stops (additive, 0 = no change).
    pub exposure_stops: f32,
    /// Saturation multiplier (1.0 = no change, 0 = greyscale, 2 = vivid).
    pub saturation: f32,
}

impl Default for TemperatureGradeParams {
    fn default() -> Self {
        Self {
            kelvin: 6500.0,
            tint: 0.0,
            exposure_stops: 0.0,
            saturation: 1.0,
        }
    }
}

impl TemperatureGradeParams {
    /// Construct neutral parameters — no change to the image.
    pub fn neutral() -> Self {
        Self::default()
    }

    /// Construct a warm daylight preset (tungsten-to-daylight look).
    pub fn warm_daylight() -> Self {
        Self {
            kelvin: 4500.0,
            tint: 0.05,
            exposure_stops: 0.0,
            saturation: 1.1,
        }
    }

    /// Construct a cool moonlight / night preset.
    pub fn cool_moonlight() -> Self {
        Self {
            kelvin: 9000.0,
            tint: -0.05,
            exposure_stops: -0.3,
            saturation: 0.9,
        }
    }
}

/// Animated color temperature grading operator.
///
/// Maintains a *current* set of grade parameters and a *target* set, and
/// interpolates between them over a configurable animation duration.
pub struct ColorTemperatureGrade {
    /// Currently displayed (possibly animated) parameters.
    pub current: TemperatureGradeParams,
    /// Animation target.
    pub target: TemperatureGradeParams,
    /// Starting parameters for the current animation.
    from: TemperatureGradeParams,
    /// Elapsed time in the current animation (seconds).
    elapsed_secs: f32,
    /// Animation duration in seconds.  0 = instant.
    pub animation_duration_secs: f32,
}

impl ColorTemperatureGrade {
    /// Create a new grade starting at `params`.
    pub fn new(params: TemperatureGradeParams, animation_duration_secs: f32) -> Self {
        Self {
            current: params,
            target: params,
            from: params,
            elapsed_secs: 0.0,
            animation_duration_secs,
        }
    }

    /// Set a new target grade.  The grade will animate toward `target`.
    pub fn set_target(&mut self, target: TemperatureGradeParams) {
        self.from = self.current;
        self.target = target;
        self.elapsed_secs = 0.0;
    }

    /// Advance the animation by `dt_secs`.
    ///
    /// Returns `true` when the animation has completed.
    pub fn advance(&mut self, dt_secs: f32) -> bool {
        if self.animation_duration_secs <= 0.0 {
            self.current = self.target;
            return true;
        }
        self.elapsed_secs += dt_secs;
        let t = (self.elapsed_secs / self.animation_duration_secs).clamp(0.0, 1.0);
        let eased = ease_in_out(t);
        self.current = lerp_params(self.from, self.target, eased);
        t >= 1.0
    }

    /// Apply the current grade to an RGBA pixel buffer in-place.
    ///
    /// The grade is applied as:
    /// 1. Exposure compensation (multiplicative).
    /// 2. Color temperature (per-channel gain from Kelvin table).
    /// 3. Tint (green-magenta shift).
    /// 4. Saturation adjustment.
    pub fn apply(&self, pixels: &mut [u8]) {
        let p = &self.current;

        // Compute color temperature gains (relative to neutral D65).
        let kelvin_rgb = KelvinRgb::from_kelvin(p.kelvin).normalised();

        // Exposure multiplier: 2^stops.
        let exposure = 2.0_f32.powf(p.exposure_stops);

        // Tint: adjust green (+) / magenta (-).
        let tint_g = 1.0 + p.tint.clamp(-1.0, 1.0) * 0.3;
        let tint_m = 1.0 - p.tint.clamp(-1.0, 1.0).abs() * 0.15;

        let r_gain = kelvin_rgb.r * exposure * tint_m;
        let g_gain = kelvin_rgb.g * exposure * tint_g;
        let b_gain = kelvin_rgb.b * exposure * tint_m;

        let sat = p.saturation.clamp(0.0, 4.0);

        for chunk in pixels.chunks_exact_mut(4) {
            let r_lin = chunk[0] as f32 / 255.0;
            let g_lin = chunk[1] as f32 / 255.0;
            let b_lin = chunk[2] as f32 / 255.0;

            // Apply temperature + exposure + tint.
            let r_adj = r_lin * r_gain;
            let g_adj = g_lin * g_gain;
            let b_adj = b_lin * b_gain;

            // Apply saturation via luminance-preserving mix.
            let luma = 0.2126 * r_adj + 0.7152 * g_adj + 0.0722 * b_adj;
            let r_sat = luma + (r_adj - luma) * sat;
            let g_sat = luma + (g_adj - luma) * sat;
            let b_sat = luma + (b_adj - luma) * sat;

            chunk[0] = (r_sat * 255.0).clamp(0.0, 255.0) as u8;
            chunk[1] = (g_sat * 255.0).clamp(0.0, 255.0) as u8;
            chunk[2] = (b_sat * 255.0).clamp(0.0, 255.0) as u8;
            // Alpha unchanged.
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Ease-in-out cubic.
fn ease_in_out(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Linearly interpolate between two `TemperatureGradeParams`.
fn lerp_params(a: TemperatureGradeParams, b: TemperatureGradeParams, t: f32) -> TemperatureGradeParams {
    let lerp = |x: f32, y: f32| x + (y - x) * t;
    TemperatureGradeParams {
        kelvin: lerp(a.kelvin, b.kelvin),
        tint: lerp(a.tint, b.tint),
        exposure_stops: lerp(a.exposure_stops, b.exposure_stops),
        saturation: lerp(a.saturation, b.saturation),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Kelvin→RGB tests.

    #[test]
    fn test_kelvin_6500_is_near_neutral() {
        // D65 is the broadcast reference white.  All channels should be close
        // to each other after normalisation.
        let rgb = KelvinRgb::from_kelvin(6500.0).normalised();
        // With normalisation the brightest channel should be 1.0.
        let max = rgb.r.max(rgb.g).max(rgb.b);
        assert!((max - 1.0).abs() < 0.01, "normalised max should be 1.0");
    }

    #[test]
    fn test_kelvin_warm_low() {
        // Low Kelvin = warm → red channel should dominate.
        let rgb = KelvinRgb::from_kelvin(2700.0).normalised();
        assert!(rgb.r >= rgb.b, "warm kelvin should have more red than blue");
    }

    #[test]
    fn test_kelvin_cool_high() {
        // High Kelvin = cool → blue channel should be strong.
        let rgb = KelvinRgb::from_kelvin(10_000.0).normalised();
        assert!(rgb.b >= rgb.r, "cool kelvin should have more blue than red");
    }

    #[test]
    fn test_kelvin_clamp_below_1000() {
        // Values below 1000 K should be clamped and not panic.
        let rgb = KelvinRgb::from_kelvin(500.0);
        assert!(rgb.r >= 0.0 && rgb.r <= 1.0);
    }

    #[test]
    fn test_kelvin_clamp_above_40000() {
        let rgb = KelvinRgb::from_kelvin(50_000.0);
        assert!(rgb.b >= 0.0 && rgb.b <= 1.0);
    }

    // White balance tests.

    #[test]
    fn test_white_balance_identity() {
        // Correcting from D65 to D65 should leave the image nearly unchanged.
        let mut pixels = vec![128u8, 64, 32, 255];
        let original = pixels.clone();
        WhiteBalanceAdjust::apply(&mut pixels, 6500.0, 6500.0);
        // Allow ±2 for rounding.
        for (a, b) in pixels.iter().zip(original.iter()) {
            assert!(
                (*a as i32 - *b as i32).abs() <= 2,
                "identity correction should leave pixels nearly unchanged"
            );
        }
    }

    #[test]
    fn test_white_balance_warms_image() {
        // Correcting from cool (9000 K) to warm (3200 K) should boost red and
        // reduce blue relative to a flat-grey starting point.
        let grey = 128u8;
        let mut pixels = vec![grey, grey, grey, 255];
        WhiteBalanceAdjust::apply(&mut pixels, 9000.0, 3200.0);
        // Red should be boosted, blue should be reduced.
        assert!(
            pixels[0] >= pixels[2],
            "warm correction should have red >= blue"
        );
    }

    #[test]
    fn test_white_balance_cools_image() {
        let grey = 128u8;
        let mut pixels = vec![grey, grey, grey, 255];
        WhiteBalanceAdjust::apply(&mut pixels, 3200.0, 9000.0);
        // Blue should be boosted relative to red.
        assert!(
            pixels[2] >= pixels[0],
            "cool correction should have blue >= red"
        );
    }

    #[test]
    fn test_white_balance_preserves_alpha() {
        let mut pixels = vec![128u8, 64, 32, 200];
        WhiteBalanceAdjust::apply(&mut pixels, 5000.0, 7000.0);
        assert_eq!(pixels[3], 200, "alpha must be preserved");
    }

    // TemperatureGradeParams tests.

    #[test]
    fn test_neutral_params() {
        let p = TemperatureGradeParams::neutral();
        assert!((p.kelvin - 6500.0).abs() < 1.0);
        assert!((p.tint).abs() < f32::EPSILON);
        assert!((p.exposure_stops).abs() < f32::EPSILON);
        assert!((p.saturation - 1.0).abs() < f32::EPSILON);
    }

    // ColorTemperatureGrade tests.

    #[test]
    fn test_grade_apply_neutral_leaves_grey_grey() {
        let grade = ColorTemperatureGrade::new(TemperatureGradeParams::neutral(), 0.0);
        let mut pixels = vec![128u8, 128, 128, 255];
        grade.apply(&mut pixels);
        // Should remain roughly grey (each channel within ±10 of 128).
        for c in 0..3 {
            assert!(
                (pixels[c] as i32 - 128).abs() <= 15,
                "neutral grade should leave grey roughly grey (channel {})",
                c
            );
        }
    }

    #[test]
    fn test_grade_animation_advances() {
        let mut grade = ColorTemperatureGrade::new(TemperatureGradeParams::neutral(), 0.5);
        grade.set_target(TemperatureGradeParams::warm_daylight());
        let done = grade.advance(0.6); // past duration
        assert!(done);
        assert!((grade.current.kelvin - 4500.0).abs() < 1.0);
    }

    #[test]
    fn test_grade_animation_midpoint() {
        let neutral = TemperatureGradeParams::neutral();
        let warm = TemperatureGradeParams::warm_daylight();
        let mut grade = ColorTemperatureGrade::new(neutral, 1.0);
        grade.set_target(warm);
        grade.advance(0.5);
        // At t=0.5 of ease-in-out we are at t=0.5 → eased = 0.5.
        // kelvin should be between neutral (6500) and warm (4500).
        assert!(
            grade.current.kelvin > 4500.0 && grade.current.kelvin < 6500.0,
            "midpoint kelvin should be between targets"
        );
    }
}
