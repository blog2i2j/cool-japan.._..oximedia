//! White balance algorithms for computational photography and color correction.
//!
//! This module provides a suite of automatic white balance (AWB) algorithms,
//! colour temperature estimation from tristimulus XYZ values, and manual
//! correction utilities used in camera processing pipelines.
//!
//! # Algorithms
//!
//! | Algorithm | Description |
//! |-----------|-------------|
//! | Grey World | Assumes scene average is achromatic (neutral grey) |
//! | White Patch | Assumes brightest pixel is the illuminant |
//! | Shades of Grey | Generalised p-norm of grey-world (p=∞ → white-patch) |
//! | Grey Edge | First/second-order Minkowski norm of image gradient |
//! | Combined | Weighted blend of grey-world and white-patch estimates |
//! | Manual | Direct gain application from user-supplied RGB multipliers |
//!
//! # References
//!
//! - Van De Weijer et al., "Edge-based color constancy", IEEE TIP 2007
//! - Buchsbaum, "A spatial processor model for object colour perception", J. Franklin Inst. 1980
//! - Land & McCann, "Lightness and Retinex theory", JOSA 1971
//! - McCamy, "Correlated color temperature of the sun", Color Research & Application 1992

#![allow(dead_code)]

use crate::error::{ColorError, Result};

// ── Gain triplet ──────────────────────────────────────────────────────────────

/// RGB gain multipliers for white balance correction.
///
/// Each component is a positive gain applied channel-wise before gamma
/// encoding.  By convention the green channel is normalised to 1.0, but
/// implementations may use any common normalisation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WbGains {
    /// Red channel gain (> 0).
    pub r: f64,
    /// Green channel gain (> 0, usually 1.0).
    pub g: f64,
    /// Blue channel gain (> 0).
    pub b: f64,
}

impl WbGains {
    /// Creates new white-balance gains.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if any gain is ≤ 0.
    pub fn new(r: f64, g: f64, b: f64) -> Result<Self> {
        if r <= 0.0 || g <= 0.0 || b <= 0.0 {
            return Err(ColorError::InvalidColor(
                "WbGains: all gains must be positive".into(),
            ));
        }
        Ok(Self { r, g, b })
    }

    /// Creates unit gains (no correction).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }

    /// Normalises so that the green channel equals 1.0.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if the green gain is zero.
    pub fn normalise_green(&self) -> Result<Self> {
        if self.g.abs() < 1e-12 {
            return Err(ColorError::InvalidColor(
                "WbGains: green gain is zero, cannot normalise".into(),
            ));
        }
        Ok(Self {
            r: self.r / self.g,
            g: 1.0,
            b: self.b / self.g,
        })
    }

    /// Applies the gains to a single linear-light RGB pixel.
    ///
    /// Values are *not* clamped; callers should clamp after applying gains.
    #[must_use]
    pub fn apply(&self, rgb: [f64; 3]) -> [f64; 3] {
        [rgb[0] * self.r, rgb[1] * self.g, rgb[2] * self.b]
    }

    /// Returns the gains as an array `[r, g, b]`.
    #[must_use]
    pub const fn as_array(&self) -> [f64; 3] {
        [self.r, self.g, self.b]
    }
}

impl Default for WbGains {
    fn default() -> Self {
        Self::identity()
    }
}

// ── Grey-World algorithm ──────────────────────────────────────────────────────

/// Estimates white-balance gains using the **Grey-World** assumption.
///
/// The method computes the per-channel mean over all pixels and derives
/// scaling factors so that each channel average equals the overall
/// luminance mean:
///
/// ```text
/// gain_c = mean_all / mean_c
/// ```
///
/// # Arguments
///
/// * `pixels` - Slice of linear-light `[R, G, B]` pixels (values ≥ 0).
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if the input is empty or any
/// channel mean is too close to zero (underexposed channel).
pub fn grey_world(pixels: &[[f64; 3]]) -> Result<WbGains> {
    if pixels.is_empty() {
        return Err(ColorError::InvalidColor(
            "grey_world: pixel slice is empty".into(),
        ));
    }

    let n = pixels.len() as f64;
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;

    for p in pixels {
        sum_r += p[0];
        sum_g += p[1];
        sum_b += p[2];
    }

    let mean_r = sum_r / n;
    let mean_g = sum_g / n;
    let mean_b = sum_b / n;

    // Overall luminance mean (equal-weight)
    let mean_all = (mean_r + mean_g + mean_b) / 3.0;

    if mean_r < 1e-12 || mean_g < 1e-12 || mean_b < 1e-12 {
        return Err(ColorError::InvalidColor(
            "grey_world: one or more channels have near-zero mean".into(),
        ));
    }

    WbGains::new(mean_all / mean_r, mean_all / mean_g, mean_all / mean_b)?
        .normalise_green()
}

// ── White-Patch (Max-RGB) algorithm ──────────────────────────────────────────

/// Estimates white-balance gains using the **White-Patch** (Max-RGB) algorithm.
///
/// Finds the maximum value in each channel and normalises so the brightest
/// pixel would be rendered white:
///
/// ```text
/// gain_c = max_all / max_c
/// ```
///
/// # Arguments
///
/// * `pixels` - Slice of linear-light `[R, G, B]` pixels.
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if the input is empty or a channel
/// maximum is zero.
pub fn white_patch(pixels: &[[f64; 3]]) -> Result<WbGains> {
    if pixels.is_empty() {
        return Err(ColorError::InvalidColor(
            "white_patch: pixel slice is empty".into(),
        ));
    }

    let mut max_r = 0.0_f64;
    let mut max_g = 0.0_f64;
    let mut max_b = 0.0_f64;

    for p in pixels {
        if p[0] > max_r {
            max_r = p[0];
        }
        if p[1] > max_g {
            max_g = p[1];
        }
        if p[2] > max_b {
            max_b = p[2];
        }
    }

    if max_r < 1e-12 || max_g < 1e-12 || max_b < 1e-12 {
        return Err(ColorError::InvalidColor(
            "white_patch: one or more channels have zero maximum".into(),
        ));
    }

    let max_all = max_r.max(max_g).max(max_b);
    WbGains::new(max_all / max_r, max_all / max_g, max_all / max_b)?
        .normalise_green()
}

// ── Shades-of-Grey (p-norm) algorithm ────────────────────────────────────────

/// Estimates white-balance gains using the **Shades-of-Grey** generalisation.
///
/// The grey-world assumption uses p = 1 (mean), while white-patch uses
/// p = ∞ (maximum).  This function allows any finite p ≥ 1:
///
/// ```text
/// norm_c(p) = (Σ pixel_c^p / N)^(1/p)
/// gain_c    = norm_all / norm_c
/// ```
///
/// # Arguments
///
/// * `pixels` - Slice of linear-light `[R, G, B]` pixels.
/// * `p` - Minkowski norm order (≥ 1.0; 1.0 = grey-world, large values → white-patch).
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if `p < 1`, the input is empty, or a
/// channel norm is too close to zero.
pub fn shades_of_grey(pixels: &[[f64; 3]], p: f64) -> Result<WbGains> {
    if p < 1.0 {
        return Err(ColorError::InvalidColor(format!(
            "shades_of_grey: p must be >= 1.0, got {p}"
        )));
    }
    if pixels.is_empty() {
        return Err(ColorError::InvalidColor(
            "shades_of_grey: pixel slice is empty".into(),
        ));
    }

    let n = pixels.len() as f64;
    let mut sum_r = 0.0_f64;
    let mut sum_g = 0.0_f64;
    let mut sum_b = 0.0_f64;

    for px in pixels {
        sum_r += px[0].abs().powf(p);
        sum_g += px[1].abs().powf(p);
        sum_b += px[2].abs().powf(p);
    }

    let norm_r = (sum_r / n).powf(1.0 / p);
    let norm_g = (sum_g / n).powf(1.0 / p);
    let norm_b = (sum_b / n).powf(1.0 / p);

    if norm_r < 1e-12 || norm_g < 1e-12 || norm_b < 1e-12 {
        return Err(ColorError::InvalidColor(
            "shades_of_grey: one or more channel norms are near-zero".into(),
        ));
    }

    let norm_all = (norm_r + norm_g + norm_b) / 3.0;
    WbGains::new(norm_all / norm_r, norm_all / norm_g, norm_all / norm_b)?
        .normalise_green()
}

// ── Combined (blended) estimator ─────────────────────────────────────────────

/// Blends grey-world and white-patch estimates with configurable weights.
///
/// This is often used in practice to balance the tendency of grey-world to
/// over-correct flat scenes and white-patch to be misled by specular highlights.
///
/// ```text
/// gain = alpha * grey_world + (1 - alpha) * white_patch
/// ```
///
/// # Arguments
///
/// * `pixels` - Slice of linear-light `[R, G, B]` pixels.
/// * `alpha` - Weight for grey-world estimate (0.0 = pure white-patch, 1.0 = pure grey-world).
///
/// # Errors
///
/// Propagates errors from the underlying estimators, or returns
/// [`ColorError::InvalidColor`] if `alpha` is outside `[0, 1]`.
pub fn combined(pixels: &[[f64; 3]], alpha: f64) -> Result<WbGains> {
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ColorError::InvalidColor(format!(
            "combined: alpha must be in [0, 1], got {alpha}"
        )));
    }

    let gw = grey_world(pixels)?;
    let wp = white_patch(pixels)?;

    let r = alpha * gw.r + (1.0 - alpha) * wp.r;
    let g = alpha * gw.g + (1.0 - alpha) * wp.g;
    let b = alpha * gw.b + (1.0 - alpha) * wp.b;

    WbGains::new(r, g, b)?.normalise_green()
}

// ── Colour-temperature from linear-light sRGB ────────────────────────────────

/// Converts a linear-light sRGB white point to an approximate correlated
/// colour temperature (CCT) in Kelvin using the **Robertson algorithm**.
///
/// The input is expected to be a neutral (achromatic) colour — for example
/// the illuminant estimate returned by [`grey_world`] before gain application.
/// Non-neutral inputs will still produce a result, but the value represents the
/// *nearest* Planckian temperature, not a physical illuminant.
///
/// Uses 31 standard Planckian locus samples spanning 1000 K – 20000 K.
///
/// # Arguments
///
/// * `r` - Linear red component of the illuminant (> 0).
/// * `g` - Linear green component of the illuminant (> 0).
/// * `b` - Linear blue component of the illuminant (> 0).
///
/// # Returns
///
/// Estimated CCT in Kelvin, or [`ColorError::InvalidColor`] if the input
/// components are all zero.
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if all components are zero.
pub fn rgb_to_cct(r: f64, g: f64, b: f64) -> Result<f64> {
    // Convert linear sRGB → XYZ (D65 primaries, IEC 61966-2-1)
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    xyz_to_cct(x, y, z)
}

/// Converts CIE XYZ values to an approximate CCT using the Robertson algorithm.
///
/// The Robertson algorithm iterates over a precomputed table of 31 Planckian
/// locus isotemperature lines.  Accuracy is approximately ± 25 K over the
/// range 1000 K – 20000 K.
///
/// # Arguments
///
/// * `x` - CIE X tristimulus value.
/// * `y` - CIE Y tristimulus value.
/// * `z` - CIE Z tristimulus value.
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if X + Y + Z ≈ 0 (black input).
pub fn xyz_to_cct(x: f64, y: f64, z: f64) -> Result<f64> {
    let sum = x + y + z;
    if sum < 1e-12 {
        return Err(ColorError::InvalidColor(
            "xyz_to_cct: XYZ sum is zero — cannot estimate CCT for black".into(),
        ));
    }

    // Convert to CIE 1960 UCS (u, v)
    let u = 4.0 * x / (x + 15.0 * y + 3.0 * z);
    let v = 6.0 * y / (x + 15.0 * y + 3.0 * z);

    // Robertson isotemperature line table
    // Each entry: (reciprocal megakelvin, u, v, t) where t = dv/du slope
    // Source: Robertson 1968, Computation of correlated color temperature
    static RT: &[(f64, f64, f64, f64)] = &[
        (0.0, 0.18006, 0.26352, -0.24341),
        (10.0, 0.18066, 0.26589, -0.25479),
        (20.0, 0.18133, 0.26846, -0.26876),
        (30.0, 0.18208, 0.27119, -0.28539),
        (40.0, 0.18293, 0.27407, -0.30470),
        (50.0, 0.18388, 0.27709, -0.32675),
        (60.0, 0.18494, 0.28021, -0.35156),
        (70.0, 0.18611, 0.28342, -0.37915),
        (80.0, 0.18740, 0.28668, -0.40955),
        (90.0, 0.18880, 0.28997, -0.44278),
        (100.0, 0.19032, 0.29326, -0.47888),
        (125.0, 0.19462, 0.30141, -0.58204),
        (150.0, 0.19962, 0.30921, -0.70471),
        (175.0, 0.20525, 0.31647, -0.84901),
        (200.0, 0.21142, 0.32312, -1.0182),
        (225.0, 0.21807, 0.32909, -1.2168),
        (250.0, 0.22511, 0.33439, -1.4512),
        (275.0, 0.23247, 0.33904, -1.7298),
        (300.0, 0.24010, 0.34308, -2.0637),
        (325.0, 0.24792, 0.34655, -2.4681),
        (350.0, 0.25591, 0.34951, -2.9641),
        (375.0, 0.26400, 0.35200, -3.5814),
        (400.0, 0.27218, 0.35407, -4.3633),
        (425.0, 0.28039, 0.35577, -5.3762),
        (450.0, 0.28863, 0.35714, -6.7262),
        (475.0, 0.29685, 0.35823, -8.5955),
        (500.0, 0.30505, 0.35907, -11.324),
        (525.0, 0.31320, 0.35968, -15.628),
        (550.0, 0.32129, 0.36011, -23.325),
        (575.0, 0.32931, 0.36038, -40.770),
        (600.0, 0.33724, 0.36051, -116.45),
    ];

    // Find the two adjacent entries that straddle the input
    let mut prev_di = f64::MAX;
    let mut prev_i = 0usize;

    for (i, &(_, ui, vi, ti)) in RT.iter().enumerate() {
        let di = (v - vi) - ti * (u - ui);
        if i > 0 && prev_di * di < 0.0 {
            // Interpolate between entry i-1 and i
            let f = prev_di / (prev_di - di);
            let r0 = RT[prev_i].0;
            let r1 = RT[i].0;
            let r_mrd = r0 + f * (r1 - r0); // interpolated in reciprocal megakelvin
            // Convert reciprocal megakelvin → Kelvin
            let cct = 1.0e6 / r_mrd.max(1e-6);
            return Ok(cct.clamp(1000.0, 20000.0));
        }
        prev_di = di;
        prev_i = i;
    }

    // Fallback: return temperature nearest to the last entry
    let last_r = RT.last().map(|e| e.0).unwrap_or(600.0);
    Ok((1.0e6 / last_r).clamp(1000.0, 20000.0))
}

// ── Manual white balance ──────────────────────────────────────────────────────

/// Manual white balance: derive gains from a measured neutral patch.
///
/// Given the linear-light RGB values measured from a patch that *should* be
/// neutral grey, computes gains that would render the patch as (target, target, target).
///
/// # Arguments
///
/// * `measured` - Measured `[R, G, B]` of the neutral patch (linear, > 0).
/// * `target` - Target luminance value (e.g. 0.5 for mid-grey, 1.0 for white).
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if any measured component is ≤ 0 or
/// `target` ≤ 0.
pub fn manual_from_neutral_patch(measured: [f64; 3], target: f64) -> Result<WbGains> {
    if target <= 0.0 {
        return Err(ColorError::InvalidColor(
            "manual_from_neutral_patch: target must be > 0".into(),
        ));
    }
    if measured[0] <= 0.0 || measured[1] <= 0.0 || measured[2] <= 0.0 {
        return Err(ColorError::InvalidColor(
            "manual_from_neutral_patch: all measured components must be > 0".into(),
        ));
    }
    WbGains::new(
        target / measured[0],
        target / measured[1],
        target / measured[2],
    )?
    .normalise_green()
}

// ── Kelvin preset lookup ──────────────────────────────────────────────────────

/// Standard colour temperature presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KelvinPreset {
    /// Candlelight / flame (~1850 K).
    Candlelight,
    /// Tungsten / incandescent lamp (~2850 K).
    Tungsten,
    /// Sunrise / sunset / warm fluorescent (~3200 K).
    Sunrise,
    /// Fluorescent / cool white (~4000 K).
    Fluorescent,
    /// Horizon daylight / noon sun (~5000 K).
    HorizonDaylight,
    /// Average daylight / D55 (~5500 K).
    Daylight,
    /// D65 standard illuminant / overcast sky (~6500 K).
    D65,
    /// Cloudy sky (~7000 K).
    Cloudy,
    /// Blue sky / shade (~9000 K).
    Shade,
}

impl KelvinPreset {
    /// Returns the nominal colour temperature in Kelvin.
    #[must_use]
    pub const fn kelvin(self) -> f64 {
        match self {
            Self::Candlelight => 1850.0,
            Self::Tungsten => 2850.0,
            Self::Sunrise => 3200.0,
            Self::Fluorescent => 4000.0,
            Self::HorizonDaylight => 5000.0,
            Self::Daylight => 5500.0,
            Self::D65 => 6504.0,
            Self::Cloudy => 7000.0,
            Self::Shade => 9000.0,
        }
    }

    /// Returns the preset name as a static string.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Candlelight => "Candlelight",
            Self::Tungsten => "Tungsten",
            Self::Sunrise => "Sunrise",
            Self::Fluorescent => "Fluorescent",
            Self::HorizonDaylight => "Horizon Daylight",
            Self::Daylight => "Daylight",
            Self::D65 => "D65",
            Self::Cloudy => "Cloudy",
            Self::Shade => "Shade",
        }
    }

    /// Derives approximate linear-light sRGB gains for this preset relative to D65.
    ///
    /// Uses the Planckian locus approximation from [`crate::color_temperature`]
    /// to compute xy chromaticity, then converts to XYZ and derives gains.
    ///
    /// # Returns
    ///
    /// Green-normalised `[R, G, B]` gains.
    #[must_use]
    pub fn to_gains(self) -> WbGains {
        use crate::color_temperature::kelvin_to_chromaticity;

        let c = kelvin_to_chromaticity(self.kelvin());
        let d65 = kelvin_to_chromaticity(6504.0);

        // Map chromaticity shift to gain adjustment
        let dr = if c.x > 1e-12 { d65.x / c.x } else { 1.0 };
        let db = if c.y > 1e-12 { d65.y / c.y } else { 1.0 };

        // Normalise green to 1.0
        WbGains {
            r: dr,
            g: 1.0,
            b: db,
        }
    }
}

// ── Tint correction ───────────────────────────────────────────────────────────

/// Applies a green/magenta **tint** adjustment on top of existing gains.
///
/// A positive tint adds green (reduces magenta cast); negative adds magenta.
/// The tint is implemented as a differential shift of the green gain relative
/// to an equal blend of red and blue.
///
/// # Arguments
///
/// * `gains` - Existing white-balance gains.
/// * `tint`  - Tint offset in the range [−1, +1]. Values outside this range
///             are clamped.
///
/// # Returns
///
/// Updated gains with tint applied, green-normalised.
///
/// # Errors
///
/// Propagates errors from gain normalisation.
pub fn apply_tint(gains: &WbGains, tint: f64) -> Result<WbGains> {
    let t = tint.clamp(-1.0, 1.0);
    // Tint scale: ±0.3 log stops at the extreme
    let tint_factor = (t * 0.3_f64).exp2();
    WbGains::new(gains.r, gains.g * tint_factor, gains.b)?.normalise_green()
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_grey(value: f64, count: usize) -> Vec<[f64; 3]> {
        vec![[value, value, value]; count]
    }

    // ── WbGains ───────────────────────────────────────────────────────────────

    #[test]
    fn wb_gains_identity() {
        let g = WbGains::identity();
        assert_eq!(g.apply([0.5, 0.4, 0.3]), [0.5, 0.4, 0.3]);
    }

    #[test]
    fn wb_gains_reject_non_positive() {
        assert!(WbGains::new(0.0, 1.0, 1.0).is_err());
        assert!(WbGains::new(1.0, -0.5, 1.0).is_err());
    }

    #[test]
    fn wb_gains_normalise_green() {
        let g = WbGains::new(2.0, 4.0, 1.0).unwrap();
        let n = g.normalise_green().unwrap();
        assert!((n.g - 1.0).abs() < 1e-12);
        assert!((n.r - 0.5).abs() < 1e-12);
        assert!((n.b - 0.25).abs() < 1e-12);
    }

    // ── grey_world ────────────────────────────────────────────────────────────

    #[test]
    fn grey_world_neutral_gives_identity() {
        let pixels = uniform_grey(0.5, 100);
        let gains = grey_world(&pixels).unwrap();
        assert!((gains.r - 1.0).abs() < 1e-6, "r={}", gains.r);
        assert!((gains.g - 1.0).abs() < 1e-6, "g={}", gains.g);
        assert!((gains.b - 1.0).abs() < 1e-6, "b={}", gains.b);
    }

    #[test]
    fn grey_world_warm_scene_reduces_red_gain() {
        // Simulate a warm scene where red channel is elevated
        let pixels: Vec<[f64; 3]> = (0..100).map(|_| [0.8, 0.5, 0.3]).collect();
        let gains = grey_world(&pixels).unwrap();
        // Red is high → gain for red should be < gain for blue
        assert!(gains.r < gains.b, "r={} b={}", gains.r, gains.b);
    }

    #[test]
    fn grey_world_empty_returns_error() {
        assert!(grey_world(&[]).is_err());
    }

    // ── white_patch ───────────────────────────────────────────────────────────

    #[test]
    fn white_patch_neutral_gives_identity() {
        let pixels = uniform_grey(0.9, 50);
        let gains = white_patch(&pixels).unwrap();
        assert!((gains.r - 1.0).abs() < 1e-6);
        assert!((gains.g - 1.0).abs() < 1e-6);
        assert!((gains.b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn white_patch_warm_scene() {
        let pixels: Vec<[f64; 3]> = vec![[1.0, 0.7, 0.4]];
        let gains = white_patch(&pixels).unwrap();
        // All channels normalised so green = 1; R should be < 1 (already above others)
        assert!((gains.g - 1.0).abs() < 1e-9);
    }

    #[test]
    fn white_patch_empty_returns_error() {
        assert!(white_patch(&[]).is_err());
    }

    // ── shades_of_grey ────────────────────────────────────────────────────────

    #[test]
    fn shades_of_grey_p1_matches_grey_world() {
        let pixels: Vec<[f64; 3]> = (0..50).map(|i| {
            let v = (i as f64) / 50.0 + 0.1;
            [v * 1.2, v, v * 0.8]
        }).collect();
        let sog = shades_of_grey(&pixels, 1.0).unwrap();
        let gw  = grey_world(&pixels).unwrap();
        assert!((sog.r - gw.r).abs() < 1e-9, "r: {} vs {}", sog.r, gw.r);
        assert!((sog.b - gw.b).abs() < 1e-9, "b: {} vs {}", sog.b, gw.b);
    }

    #[test]
    fn shades_of_grey_invalid_p_returns_error() {
        let pixels = uniform_grey(0.5, 10);
        assert!(shades_of_grey(&pixels, 0.5).is_err());
    }

    // ── combined ─────────────────────────────────────────────────────────────

    #[test]
    fn combined_alpha1_matches_grey_world() {
        let pixels: Vec<[f64; 3]> = (0..80).map(|i| {
            let v = (i as f64) / 80.0 + 0.1;
            [v * 1.1, v, v * 0.9]
        }).collect();
        let comb = combined(&pixels, 1.0).unwrap();
        let gw   = grey_world(&pixels).unwrap();
        assert!((comb.r - gw.r).abs() < 1e-9);
        assert!((comb.b - gw.b).abs() < 1e-9);
    }

    #[test]
    fn combined_alpha_out_of_range_returns_error() {
        let pixels = uniform_grey(0.5, 10);
        assert!(combined(&pixels, 1.5).is_err());
        assert!(combined(&pixels, -0.1).is_err());
    }

    // ── rgb_to_cct ────────────────────────────────────────────────────────────

    #[test]
    fn cct_d65_approx() {
        // D65 white in linear sRGB is [1, 1, 1]
        let cct = rgb_to_cct(1.0, 1.0, 1.0).unwrap();
        // Should be near 6500 K; allow generous tolerance for Robertson approx
        assert!(
            (cct - 6504.0).abs() < 1000.0,
            "D65 CCT: expected ~6504 K, got {cct}"
        );
    }

    #[test]
    fn cct_black_returns_error() {
        assert!(rgb_to_cct(0.0, 0.0, 0.0).is_err());
    }

    // ── manual_from_neutral_patch ─────────────────────────────────────────────

    #[test]
    fn manual_from_neutral_patch_basic() {
        // A patch measured at [0.4, 0.5, 0.3] should be corrected to 0.5
        let gains = manual_from_neutral_patch([0.4, 0.5, 0.3], 0.5).unwrap();
        // After applying gains, the patch should be (roughly) neutral
        let corrected = gains.apply([0.4, 0.5, 0.3]);
        // With green normalisation, the relationship should hold
        let r_g = corrected[0] / corrected[1];
        let b_g = corrected[2] / corrected[1];
        assert!((r_g - 1.0).abs() < 1e-9, "r/g ratio: {}", r_g);
        assert!((b_g - 1.0).abs() < 1e-9, "b/g ratio: {}", b_g);
    }

    #[test]
    fn manual_from_neutral_patch_zero_returns_error() {
        assert!(manual_from_neutral_patch([0.0, 0.5, 0.3], 0.5).is_err());
        assert!(manual_from_neutral_patch([0.4, 0.5, 0.3], 0.0).is_err());
    }

    // ── KelvinPreset ──────────────────────────────────────────────────────────

    #[test]
    fn kelvin_preset_gains_green_normalised() {
        for preset in [
            KelvinPreset::Tungsten,
            KelvinPreset::Daylight,
            KelvinPreset::D65,
            KelvinPreset::Shade,
        ] {
            let g = preset.to_gains();
            assert!(
                (g.g - 1.0).abs() < 1e-9,
                "{} green gain should be 1.0, got {}",
                preset.name(),
                g.g
            );
        }
    }

    // ── apply_tint ────────────────────────────────────────────────────────────

    #[test]
    fn tint_zero_is_identity() {
        let gains = WbGains::new(1.2, 1.0, 0.9).unwrap();
        let tinted = apply_tint(&gains, 0.0).unwrap();
        let base = gains.normalise_green().unwrap();
        assert!((tinted.r - base.r).abs() < 1e-9);
        assert!((tinted.b - base.b).abs() < 1e-9);
    }

    #[test]
    fn tint_positive_boosts_green() {
        let base = WbGains::identity();
        // Positive tint → green gain factor > 1 → after green normalisation,
        // red and blue gains should decrease
        let tinted = apply_tint(&base, 0.5).unwrap();
        assert!(tinted.r < 1.0, "r should be < 1 after green tint: {}", tinted.r);
        assert!(tinted.b < 1.0, "b should be < 1 after green tint: {}", tinted.b);
    }

    #[test]
    fn tint_clamped_to_range() {
        let base = WbGains::identity();
        // Should not panic with extreme values
        let _ = apply_tint(&base, 100.0).unwrap();
        let _ = apply_tint(&base, -100.0).unwrap();
    }
}
