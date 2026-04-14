//! Creative LUT generators for cinematic and artistic looks.
//!
//! This module provides ready-to-use LUT generators for common cinematic,
//! broadcast, and creative color grading looks.
//!
//! # Available Looks
//!
//! - **Log to Rec.709**: Convert log-encoded footage to standard display
//! - **Cross Process**: Chemical cross-processing simulation
//! - **Teal and Orange**: Popular complementary color grading
//! - **Vintage**: Warm, faded film look
//! - **Bleach Bypass**: Desaturated, high-contrast silver-retention look
//! - **Day for Night**: Simulate night from day footage
//! - **Blockbuster**: High-contrast, saturated summer blockbuster look
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::creative;
//! use oximedia_lut::{LutInterpolation, LutSize};
//!
//! // Generate a cinematic teal and orange LUT
//! let lut = creative::generate_teal_orange(0.6, LutSize::Size33);
//!
//! // Apply to a pixel
//! let input = [0.5, 0.3, 0.7];
//! let output = lut.apply(&input, LutInterpolation::Tetrahedral);
//! assert_eq!(output.len(), 3);
//! ```

use crate::{Lut3d, LutSize, Rgb};

// ============================================================================
// Log to Rec.709 Conversion
// ============================================================================

/// Generate a logarithmic to Rec.709 display LUT.
///
/// Converts log-encoded footage (using a simplified cinematic log curve
/// where 0.0 = black, 0.4 = scene mid-gray, 1.0 = scene white) to
/// Rec.709 display space. The output is a 3D LUT that applies:
/// 1. Log decode (practical log-to-linear curve)
/// 2. Scene-linear to display mapping with highlight compression
/// 3. Gamma encode for Rec.709 display
///
/// # Arguments
///
/// * `size` - LUT grid size (higher = more accurate, slower to generate)
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_log_to_rec709;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_log_to_rec709(LutSize::Size33);
/// let input = [0.4, 0.4, 0.4]; // Mid-gray in log space
/// let output = lut.apply(&input, LutInterpolation::Tetrahedral);
/// assert!(output[0] >= 0.0 && output[0] <= 1.0);
/// // Mid-gray log should produce a visible mid-gray display value
/// assert!(output[0] > 0.3 && output[0] < 0.7);
/// ```
#[must_use]
pub fn generate_log_to_rec709(size: LutSize) -> Lut3d {
    Lut3d::from_fn(size, |rgb| {
        // Decode from practical cinematic log to scene-linear
        // Log 0.0 → linear 0.0, log 0.4 → linear 0.18, log 1.0 → linear ~1.0
        let linear = [
            log_c_decode(rgb[0]),
            log_c_decode(rgb[1]),
            log_c_decode(rgb[2]),
        ];

        // Apply scene linear to display (simple knee compression)
        let display = [
            apply_display_curve(linear[0]),
            apply_display_curve(linear[1]),
            apply_display_curve(linear[2]),
        ];

        // Encode to Rec.709 gamma
        [
            rec709_oetf(display[0]),
            rec709_oetf(display[1]),
            rec709_oetf(display[2]),
        ]
    })
}

/// Practical logarithmic decode.
///
/// Models a generic cinematic log curve where:
/// - 0.0 → 0.0 (black)
/// - 0.4 → 0.18 (scene mid-gray)
/// - 1.0 → 1.0 (scene white, clamped)
///
/// Uses a power function with offset to match log encoding characteristics.
#[must_use]
#[inline]
fn log_c_decode(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    // Practical curve: maps log [0,1] to linear [0,1]
    // Calibrated: log 0.4 → linear 0.18
    // Using: linear = ((x - cut_point) * scale).powf(exp)
    // where cut_point = 0.0, scale ≈ 1.5, exp ≈ 3.0 gives:
    //   f(0.4) = (0.4 * 1.5)^3 = 0.6^3 = 0.216 ≈ 0.18  ✓
    //   f(1.0) = (1.0 * 1.0)^... need to normalise
    // Simpler: linear = cut * (x / pivot)^gamma where pivot=0.4, gamma=3.0
    if x < 1e-6 {
        0.0
    } else {
        let pivot = 0.4_f64;
        let mid_gray = 0.18_f64;
        // linear = mid_gray * (x / pivot)^gamma
        // calibrate gamma so that x=1 gives linear=1.0:
        //   1.0 = mid_gray * (1/pivot)^gamma
        //   (1/pivot)^gamma = 1/mid_gray
        //   gamma * ln(1/pivot) = ln(1/mid_gray)
        //   gamma = ln(1/mid_gray) / ln(1/pivot)
        let gamma = (1.0_f64 / mid_gray).ln() / (1.0_f64 / pivot).ln();
        let linear = mid_gray * (x / pivot).powf(gamma);
        linear.clamp(0.0, 2.0) // Allow slight over-exposure headroom
    }
}

/// Simple display mapping with knee to compress highlights.
#[must_use]
#[inline]
fn apply_display_curve(linear: f64) -> f64 {
    let linear = linear.max(0.0);
    // Soft knee at 1.0 to handle specular highlights
    if linear <= 1.0 {
        linear
    } else {
        1.0 + (linear - 1.0) / (1.0 + (linear - 1.0) * 2.0)
    }
}

/// Rec.709 OETF (linear to gamma-encoded).
#[must_use]
#[inline]
fn rec709_oetf(linear: f64) -> f64 {
    let linear = linear.clamp(0.0, 1.0);
    if linear < 0.018 {
        linear * 4.5
    } else {
        1.099 * linear.powf(0.45) - 0.099
    }
}

// ============================================================================
// Cross Process
// ============================================================================

/// Generate a cross-process look LUT.
///
/// Simulates the chemical cross-processing technique where film intended for
/// one process is developed using the chemistry of another. Produces
/// high-contrast, saturated colors with channel-specific tone shifts.
///
/// # Arguments
///
/// * `strength` - Effect strength from 0.0 (identity) to 1.0 (full effect)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_cross_process;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_cross_process(0.8, LutSize::Size17);
/// let input = [0.5, 0.5, 0.5];
/// let output = lut.apply(&input, LutInterpolation::Tetrahedral);
/// assert!(output[0] >= 0.0 && output[0] <= 1.0);
/// ```
#[must_use]
pub fn generate_cross_process(strength: f64, size: LutSize) -> Lut3d {
    let strength = strength.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        // Each channel gets a different S-curve
        let r_out = cross_process_red(rgb[0]);
        let g_out = cross_process_green(rgb[1]);
        let b_out = cross_process_blue(rgb[2]);

        // Blend with identity based on strength
        [
            lerp(rgb[0], r_out, strength),
            lerp(rgb[1], g_out, strength),
            lerp(rgb[2], b_out, strength),
        ]
    })
}

/// Red channel cross-process curve: boosted shadows, compressed highlights.
#[must_use]
#[inline]
fn cross_process_red(x: f64) -> f64 {
    // Lifted shadows + boosted midtones + compressed highlights
    let lifted = x * 0.95 + 0.05;
    apply_s_curve(lifted, 0.3, 1.8)
}

/// Green channel cross-process curve: slight push with added contrast.
#[must_use]
#[inline]
fn cross_process_green(x: f64) -> f64 {
    apply_s_curve(x, 0.5, 1.4)
}

/// Blue channel cross-process curve: strong lift in shadows, crushed highlights.
#[must_use]
#[inline]
fn cross_process_blue(x: f64) -> f64 {
    // Strong lift (E6 in C41 chemistry effect)
    let lifted = x * 0.80 + 0.08;
    apply_s_curve(lifted, 0.4, 2.2)
}

/// Apply a parametric S-curve around a pivot.
///
/// * `pivot` - midpoint of the S-curve
/// * `contrast` - steepness of the S-curve
#[must_use]
#[inline]
fn apply_s_curve(x: f64, pivot: f64, contrast: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    // Contrast around pivot
    let result = pivot + (x - pivot) * contrast;
    // Soft-clip to [0, 1]
    smooth_clamp(result)
}

/// Smooth clamp that softly clips values to [0, 1].
#[must_use]
#[inline]
fn smooth_clamp(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

// ============================================================================
// Teal and Orange
// ============================================================================

/// Generate a teal-and-orange look LUT.
///
/// Creates the classic complementary color split used extensively in
/// Hollywood cinematography: warm skin tones / highlights (orange) and
/// cool shadows / backgrounds (teal). Operates in HSL-like space to
/// selectively push hues.
///
/// # Arguments
///
/// * `strength` - Effect strength from 0.0 (identity) to 1.0 (full effect)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_teal_orange;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_teal_orange(0.7, LutSize::Size17);
/// let neutral = [0.5, 0.5, 0.5];
/// let output = lut.apply(&neutral, LutInterpolation::Tetrahedral);
/// // Neutral grey stays close to grey
/// assert!((output[0] - output[1]).abs() < 0.15);
/// ```
#[must_use]
pub fn generate_teal_orange(strength: f64, size: LutSize) -> Lut3d {
    let strength = strength.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        let luma = luminance_rec709(&rgb);

        // Determine shadow/highlight weight (highlights = skin, shadows = background)
        let highlight_weight = smooth_step(luma, 0.2, 0.8);
        let shadow_weight = 1.0 - highlight_weight;

        // Orange push: warm highlights (increase R, decrease B slightly)
        let orange_r = rgb[0] + 0.12 * highlight_weight * strength;
        let orange_g = rgb[1] + 0.03 * highlight_weight * strength;
        let orange_b = rgb[2] - 0.08 * highlight_weight * strength;

        // Teal push: cool shadows (decrease R, increase G+B)
        let teal_r = orange_r - 0.10 * shadow_weight * strength;
        let teal_g = orange_g + 0.05 * shadow_weight * strength;
        let teal_b = orange_b + 0.10 * shadow_weight * strength;

        // Desaturate slightly to give a modern filmic feel
        let desat_factor = 0.08 * strength;
        let luma_out = luminance_rec709(&[teal_r, teal_g, teal_b]);
        let r = lerp(teal_r, luma_out, desat_factor).clamp(0.0, 1.0);
        let g = lerp(teal_g, luma_out, desat_factor).clamp(0.0, 1.0);
        let b = lerp(teal_b, luma_out, desat_factor).clamp(0.0, 1.0);

        [r, g, b]
    })
}

// ============================================================================
// Vintage / Warm Film
// ============================================================================

/// Generate a warm vintage film look LUT.
///
/// Produces a faded, warm look reminiscent of older film stocks. Characteristics:
/// - Lifted blacks (faded shadows)
/// - Warm color cast (yellowed highlights)
/// - Slightly desaturated with reduced contrast
/// - Subtle green push in midtones
///
/// # Arguments
///
/// * `warmth` - Warmth from 0.0 (cool) to 1.0 (very warm/vintage)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_vintage;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_vintage(0.7, LutSize::Size17);
/// // Black should be lifted (not pure 0)
/// let black = [0.0, 0.0, 0.0];
/// let lifted_black = lut.apply(&black, LutInterpolation::Tetrahedral);
/// assert!(lifted_black[0] > 0.0 || lifted_black[1] > 0.0 || lifted_black[2] > 0.0);
/// ```
#[must_use]
pub fn generate_vintage(warmth: f64, size: LutSize) -> Lut3d {
    let warmth = warmth.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        // Lift blacks (faded look)
        let black_lift = 0.04 * warmth;
        let r = rgb[0] * (1.0 - black_lift) + black_lift;
        let g = rgb[1] * (1.0 - black_lift) + black_lift;
        let b = rgb[2] * (1.0 - black_lift) + black_lift;

        // Warm color cast (reduce blue, add red/green)
        let r = r + 0.06 * warmth;
        let g = g + 0.02 * warmth;
        let b = b - 0.05 * warmth;

        // Slight yellow-green cast in midtones (Kodak Portra feel)
        let luma = luminance_rec709(&[r, g, b]);
        let midtone_mask = midtone_weight(luma);
        let r = r - 0.01 * warmth * midtone_mask;
        let g = g + 0.03 * warmth * midtone_mask;
        let b = b - 0.02 * warmth * midtone_mask;

        // Gentle desaturation
        let desat = 0.12 * warmth;
        let luma2 = luminance_rec709(&[r, g, b]);
        let r = lerp(r, luma2, desat).clamp(0.0, 1.0);
        let g = lerp(g, luma2, desat).clamp(0.0, 1.0);
        let b = lerp(b, luma2, desat).clamp(0.0, 1.0);

        // Slight contrast reduction
        let contrast = 0.90 + (1.0 - warmth) * 0.10;
        let pivot = 0.5;
        let r = (pivot + (r - pivot) * contrast).clamp(0.0, 1.0);
        let g = (pivot + (g - pivot) * contrast).clamp(0.0, 1.0);
        let b = (pivot + (b - pivot) * contrast).clamp(0.0, 1.0);

        [r, g, b]
    })
}

// ============================================================================
// Bleach Bypass
// ============================================================================

/// Generate a bleach bypass look LUT.
///
/// Simulates silver-retention processing, which skips the bleach step in
/// color film development. The result retains silver, creating:
/// - Desaturated, high-contrast image
/// - Retained grain texture (simulated via value shifts)
/// - Lifted blacks with crushed colors
///
/// # Arguments
///
/// * `strength` - Effect strength from 0.0 (identity) to 1.0 (full bypass)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::{generate_bleach_bypass, saturation_rgb};
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_bleach_bypass(0.8, LutSize::Size17);
/// let colorful = [0.2, 0.7, 0.9];
/// let output = lut.apply(&colorful, LutInterpolation::Tetrahedral);
/// // Bleach bypass desaturates
/// let in_sat = saturation_rgb(&colorful);
/// let out_sat = saturation_rgb(&output);
/// assert!(out_sat <= in_sat + 0.01);
/// ```
#[must_use]
pub fn generate_bleach_bypass(strength: f64, size: LutSize) -> Lut3d {
    let strength = strength.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        // Calculate luminance
        let luma = luminance_rec709(&rgb);

        // Blend color with luminance (desaturation)
        let desat = 0.5 * strength;
        let r = lerp(rgb[0], luma, desat);
        let g = lerp(rgb[1], luma, desat);
        let b = lerp(rgb[2], luma, desat);

        // Apply high contrast S-curve
        let contrast = 1.0 + 0.5 * strength;
        let pivot = 0.42;
        let r = apply_contrast_curve(r, contrast, pivot);
        let g = apply_contrast_curve(g, contrast, pivot);
        let b = apply_contrast_curve(b, contrast, pivot);

        // Lift shadows slightly (silver base fog)
        let lift = 0.02 * strength;
        let r = (r + lift).clamp(0.0, 1.0);
        let g = (g + lift).clamp(0.0, 1.0);
        let b = (b + lift).clamp(0.0, 1.0);

        [r, g, b]
    })
}

/// Compute approximate saturation of an RGB value.
#[must_use]
pub fn saturation_rgb(rgb: &Rgb) -> f64 {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    if max < 1e-10 {
        0.0
    } else {
        (max - min) / max
    }
}

/// Apply a contrast S-curve around a pivot.
#[must_use]
#[inline]
fn apply_contrast_curve(x: f64, contrast: f64, pivot: f64) -> f64 {
    let adjusted = pivot + (x - pivot) * contrast;
    adjusted.clamp(0.0, 1.0)
}

// ============================================================================
// Day for Night
// ============================================================================

/// Generate a day-for-night look LUT.
///
/// Simulates footage shot at night using day footage. Uses selective
/// desaturation, blue push in shadows, and overall exposure reduction.
///
/// # Arguments
///
/// * `strength` - Effect strength from 0.0 (identity) to 1.0 (full night)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_day_for_night;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_day_for_night(0.8, LutSize::Size17);
/// let daylight = [0.7, 0.6, 0.5];
/// let night = lut.apply(&daylight, LutInterpolation::Tetrahedral);
/// // Night should be darker than day
/// let in_lum: f64 = daylight.iter().sum::<f64>() / 3.0;
/// let out_lum: f64 = night.iter().sum::<f64>() / 3.0;
/// assert!(out_lum < in_lum);
/// ```
#[must_use]
pub fn generate_day_for_night(strength: f64, size: LutSize) -> Lut3d {
    let strength = strength.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        // Reduce overall exposure
        let exposure_scale = 1.0 - 0.5 * strength;
        let r = rgb[0] * exposure_scale;
        let g = rgb[1] * exposure_scale;
        let b = rgb[2] * exposure_scale;

        // Moonlight color cast: blue/teal shift
        let r = r - 0.05 * strength;
        let g = g + 0.01 * strength;
        let b = b + 0.10 * strength;

        // Desaturate (scotopic vision is less color-sensitive)
        let luma = luminance_rec709(&[r, g, b]);
        let desat = 0.4 * strength;
        let r = lerp(r, luma, desat).clamp(0.0, 1.0);
        let g = lerp(g, luma, desat).clamp(0.0, 1.0);
        let b = lerp(b, luma, desat).clamp(0.0, 1.0);

        // Crush shadows (moonlit scenes have very dark shadows)
        let r = apply_shadow_crush(r, 0.3 * strength);
        let g = apply_shadow_crush(g, 0.3 * strength);
        let b = apply_shadow_crush(b, 0.3 * strength);

        [r, g, b]
    })
}

/// Crush shadows by applying a toe curve.
#[must_use]
#[inline]
fn apply_shadow_crush(x: f64, amount: f64) -> f64 {
    let power = 1.0 + amount;
    x.powf(power).clamp(0.0, 1.0)
}

// ============================================================================
// Blockbuster
// ============================================================================

/// Generate a blockbuster cinematic look LUT.
///
/// Creates a high-contrast, punchy look commonly used in action and
/// adventure films. Features:
/// - Increased contrast and saturation
/// - Deep, crushed blacks
/// - Slightly cool shadows with warm highlights
///
/// # Arguments
///
/// * `intensity` - Effect intensity from 0.0 (identity) to 1.0 (maximum)
/// * `size` - LUT grid size
///
/// # Example
///
/// ```rust
/// use oximedia_lut::creative::generate_blockbuster;
/// use oximedia_lut::{LutInterpolation, LutSize};
///
/// let lut = generate_blockbuster(0.7, LutSize::Size17);
/// let input = [0.5, 0.4, 0.3];
/// let output = lut.apply(&input, LutInterpolation::Tetrahedral);
/// assert!(output[0] >= 0.0 && output[0] <= 1.0);
/// ```
#[must_use]
pub fn generate_blockbuster(intensity: f64, size: LutSize) -> Lut3d {
    let intensity = intensity.clamp(0.0, 1.0);

    Lut3d::from_fn(size, |rgb| {
        // High contrast S-curve
        let contrast = 1.0 + 0.6 * intensity;
        let pivot = 0.45;
        let r = apply_contrast_curve(rgb[0], contrast, pivot);
        let g = apply_contrast_curve(rgb[1], contrast, pivot);
        let b = apply_contrast_curve(rgb[2], contrast, pivot);

        // Saturation boost
        let luma = luminance_rec709(&[r, g, b]);
        let sat_boost = 1.0 + 0.3 * intensity;
        let r = lerp(luma, r, sat_boost).clamp(0.0, 1.0);
        let g = lerp(luma, g, sat_boost).clamp(0.0, 1.0);
        let b = lerp(luma, b, sat_boost).clamp(0.0, 1.0);

        // Crushed blacks via toe
        let r = apply_shadow_crush(r, 0.2 * intensity);
        let g = apply_shadow_crush(g, 0.2 * intensity);
        let b = apply_shadow_crush(b, 0.2 * intensity);

        // Cool shadows / warm highlights
        let luma2 = luminance_rec709(&[r, g, b]);
        let shadow_mask = 1.0 - smooth_step(luma2, 0.1, 0.5);
        let highlight_mask = smooth_step(luma2, 0.5, 0.9);

        let r = (r + 0.04 * highlight_mask * intensity - 0.02 * shadow_mask * intensity)
            .clamp(0.0, 1.0);
        let g = (g - 0.01 * highlight_mask * intensity).clamp(0.0, 1.0);
        let b = (b - 0.03 * highlight_mask * intensity + 0.04 * shadow_mask * intensity)
            .clamp(0.0, 1.0);

        [r, g, b]
    })
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Linear interpolation between two values.
#[must_use]
#[inline]
pub(crate) fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Calculate luminance using Rec.709 coefficients.
#[must_use]
#[inline]
pub(crate) fn luminance_rec709(rgb: &Rgb) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Smooth step between `edge0` and `edge1`.
#[must_use]
#[inline]
fn smooth_step(x: f64, edge0: f64, edge1: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Weight emphasizing midtones (peaks around 0.5 luminance).
#[must_use]
#[inline]
fn midtone_weight(luma: f64) -> f64 {
    // Triangle-like function peaking at 0.5
    let centered = (luma - 0.5).abs();
    (1.0 - centered * 2.0).clamp(0.0, 1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LutInterpolation;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_log_to_rec709_output_range() {
        let lut = generate_log_to_rec709(LutSize::Size17);
        // All outputs must be in [0, 1]
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(
                        val[0] >= 0.0 && val[0] <= 1.0,
                        "R out of range at ({r},{g},{b}): {}",
                        val[0]
                    );
                    assert!(
                        val[1] >= 0.0 && val[1] <= 1.0,
                        "G out of range at ({r},{g},{b}): {}",
                        val[1]
                    );
                    assert!(
                        val[2] >= 0.0 && val[2] <= 1.0,
                        "B out of range at ({r},{g},{b}): {}",
                        val[2]
                    );
                }
            }
        }
    }

    #[test]
    fn test_log_to_rec709_mid_gray_visible() {
        let lut = generate_log_to_rec709(LutSize::Size33);
        // Log 0.4 = scene mid-gray (18% reflectance).
        // After decode and Rec.709 encode this should be a visible mid-gray.
        let mid = [0.4, 0.4, 0.4];
        let out = lut.apply(&mid, LutInterpolation::Tetrahedral);
        // The output should be somewhere in the mid-range [0.3, 0.7]
        assert!(
            out[0] >= 0.0 && out[0] <= 1.0,
            "Output should be in [0,1]: {}",
            out[0]
        );
        assert!(
            out[0] > 0.3 && out[0] < 0.7,
            "Mid-gray log should map to visible mid-gray display: {}",
            out[0]
        );
    }

    #[test]
    fn test_cross_process_zero_strength_is_identity() {
        let lut = generate_cross_process(0.0, LutSize::Size17);
        let input = [0.3, 0.5, 0.7];
        let out = lut.apply(&input, LutInterpolation::Trilinear);
        assert!(approx_eq(out[0], input[0], 0.02));
        assert!(approx_eq(out[1], input[1], 0.02));
        assert!(approx_eq(out[2], input[2], 0.02));
    }

    #[test]
    fn test_cross_process_output_in_range() {
        let lut = generate_cross_process(1.0, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_teal_orange_preserves_neutral_approximately() {
        let lut = generate_teal_orange(0.5, LutSize::Size33);
        let neutral = [0.5, 0.5, 0.5];
        let out = lut.apply(&neutral, LutInterpolation::Tetrahedral);
        // Grey should stay roughly neutral (within 0.15 of each other channel)
        assert!((out[0] - out[1]).abs() < 0.15);
        assert!((out[1] - out[2]).abs() < 0.15);
    }

    #[test]
    fn test_teal_orange_output_in_range() {
        let lut = generate_teal_orange(0.7, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0, "R={}", val[0]);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0, "G={}", val[1]);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0, "B={}", val[2]);
                }
            }
        }
    }

    #[test]
    fn test_vintage_lifts_black() {
        let lut = generate_vintage(1.0, LutSize::Size17);
        let black = [0.0, 0.0, 0.0];
        let out = lut.apply(&black, LutInterpolation::Tetrahedral);
        // Blacks should be lifted
        assert!(out[0] > 0.01 || out[1] > 0.01 || out[2] > 0.01);
    }

    #[test]
    fn test_vintage_zero_warmth_near_identity() {
        let lut = generate_vintage(0.0, LutSize::Size17);
        let input = [0.5, 0.5, 0.5];
        let out = lut.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], 0.5, 0.02));
        assert!(approx_eq(out[1], 0.5, 0.02));
        assert!(approx_eq(out[2], 0.5, 0.02));
    }

    #[test]
    fn test_vintage_output_in_range() {
        let lut = generate_vintage(0.8, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0, "R={}", val[0]);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0, "G={}", val[1]);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0, "B={}", val[2]);
                }
            }
        }
    }

    #[test]
    fn test_bleach_bypass_desaturates() {
        let lut = generate_bleach_bypass(1.0, LutSize::Size17);
        let colorful = [0.1, 0.7, 0.9];
        let out = lut.apply(&colorful, LutInterpolation::Tetrahedral);
        let in_sat = saturation_rgb(&colorful);
        let out_sat = saturation_rgb(&out);
        assert!(
            out_sat <= in_sat + 0.05,
            "in_sat={in_sat} out_sat={out_sat}"
        );
    }

    #[test]
    fn test_bleach_bypass_output_in_range() {
        let lut = generate_bleach_bypass(0.8, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_day_for_night_darkens() {
        let lut = generate_day_for_night(0.8, LutSize::Size17);
        let daylight = [0.7, 0.6, 0.5];
        let night = lut.apply(&daylight, LutInterpolation::Tetrahedral);
        let in_lum: f64 = daylight.iter().sum::<f64>() / 3.0;
        let out_lum: f64 = night.iter().sum::<f64>() / 3.0;
        assert!(
            out_lum < in_lum,
            "Expected darkening: in={in_lum} out={out_lum}"
        );
    }

    #[test]
    fn test_day_for_night_output_in_range() {
        let lut = generate_day_for_night(0.8, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_blockbuster_boosts_contrast() {
        let lut = generate_blockbuster(0.8, LutSize::Size17);
        // A dark pixel should become darker, a bright pixel brighter (increased contrast)
        let dark = [0.1, 0.1, 0.1];
        let bright = [0.9, 0.9, 0.9];
        let dark_out = lut.apply(&dark, LutInterpolation::Tetrahedral);
        let bright_out = lut.apply(&bright, LutInterpolation::Tetrahedral);
        // Range should be stretched
        assert!(bright_out[0] >= dark_out[0]);
    }

    #[test]
    fn test_blockbuster_output_in_range() {
        let lut = generate_blockbuster(0.7, LutSize::Size17);
        for r in 0..17 {
            for g in 0..17 {
                for b in 0..17 {
                    let val = lut.get(r, g, b);
                    assert!(val[0] >= 0.0 && val[0] <= 1.0);
                    assert!(val[1] >= 0.0 && val[1] <= 1.0);
                    assert!(val[2] >= 0.0 && val[2] <= 1.0);
                }
            }
        }
    }

    #[test]
    fn test_luminance_rec709() {
        let white = [1.0, 1.0, 1.0];
        assert!(approx_eq(luminance_rec709(&white), 1.0, 1e-10));
        let black = [0.0, 0.0, 0.0];
        assert!(approx_eq(luminance_rec709(&black), 0.0, 1e-10));
    }

    #[test]
    fn test_smooth_step() {
        assert!(approx_eq(smooth_step(0.0, 0.0, 1.0), 0.0, 1e-10));
        assert!(approx_eq(smooth_step(1.0, 0.0, 1.0), 1.0, 1e-10));
        assert!(approx_eq(smooth_step(0.5, 0.0, 1.0), 0.5, 1e-10));
    }

    #[test]
    fn test_log_c_decode_midgray() {
        // Log 0.4 should map to approximately 0.18 linear (18% scene gray)
        let decoded = log_c_decode(0.4);
        assert!(
            (decoded - 0.18).abs() < 0.01,
            "log_c_decode(0.4) = {decoded}, expected ~0.18"
        );
        // Black maps to black
        assert_eq!(log_c_decode(0.0), 0.0);
        // Log 1.0 maps to approximately 1.0 linear
        let white = log_c_decode(1.0);
        assert!(
            (white - 1.0).abs() < 0.01,
            "log_c_decode(1.0) = {white}, expected ~1.0"
        );
    }

    #[test]
    fn test_all_looks_size_33() {
        // Smoke test: all generators work at Size33
        let _ = generate_log_to_rec709(LutSize::Size33);
        let _ = generate_cross_process(0.8, LutSize::Size33);
        let _ = generate_teal_orange(0.6, LutSize::Size33);
        let _ = generate_vintage(0.7, LutSize::Size33);
        let _ = generate_bleach_bypass(0.5, LutSize::Size33);
        let _ = generate_day_for_night(0.6, LutSize::Size33);
        let _ = generate_blockbuster(0.7, LutSize::Size33);
    }
}
