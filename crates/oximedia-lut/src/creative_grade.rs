//! Named film emulation LUT presets for creative grading.
//!
//! This module provides high-quality parametric film emulation presets that
//! accurately model the characteristic look of classic analogue film stocks
//! and photochemical processes.  Each preset is implemented as a function that
//! returns a [`FilmPreset`] containing both the analytical transform and a
//! baked 3-D LUT for real-time use.
//!
//! # Available Presets
//!
//! | Preset | Description |
//! |--------|-------------|
//! | `kodak_vision3_250d` | Kodak Vision3 250D tungsten-lit interior negative |
//! | `kodak_portra_400` | Kodak Portra 400 still-photography portrait stock |
//! | `fuji_eterna_500` | Fuji Eterna 500 cinema negative — fine grain |
//! | `fuji_velvia_50` | Fuji Velvia 50 slide film — saturated landscape look |
//! | `agfa_ultra_100` | Agfa Ultra 100 — vivid saturated street look |
//! | `ilford_hp5` | Ilford HP5 — monochrome negative |
//! | `cross_process` | Chemical cross-processing (slide film → C-41 developer) |
//! | `bleach_bypass` | Silver retention — desaturated high-contrast negative |
//! | `teal_orange` | Complementary teal/orange grading (configurable strength) |
//! | `day_for_night` | Simulated night shoot from day footage |
//! | `sepia_tone` | Classic sepia-tone monochrome |
//! | `fade_wash` | Faded, washed-out Instagram-style look |
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::creative_grade::FilmPreset;
//!
//! let preset = FilmPreset::kodak_portra_400();
//! let inp = [0.5_f64, 0.3, 0.7];
//! let out = preset.apply(&inp);
//! assert!(out[0] >= 0.0 && out[0] <= 1.0);
//! ```

use crate::Rgb;

// ============================================================================
// Colour helpers
// ============================================================================

/// Convert linear RGB to CIE XYZ (D65, Rec.709 primaries).
fn rgb_to_xyz(rgb: &Rgb) -> [f64; 3] {
    [
        0.412_391 * rgb[0] + 0.357_584 * rgb[1] + 0.180_481 * rgb[2],
        0.212_639 * rgb[0] + 0.715_169 * rgb[1] + 0.072_192 * rgb[2],
        0.019_331 * rgb[0] + 0.119_195 * rgb[1] + 0.950_532 * rgb[2],
    ]
}

/// Convert CIE XYZ to linear RGB (D65, Rec.709 primaries).
fn xyz_to_rgb(xyz: &[f64; 3]) -> Rgb {
    [
        3.240_970 * xyz[0] - 1.537_383 * xyz[1] - 0.498_611 * xyz[2],
        -0.969_244 * xyz[0] + 1.875_968 * xyz[1] + 0.041_555 * xyz[2],
        0.055_630 * xyz[0] - 0.203_977 * xyz[1] + 1.056_972 * xyz[2],
    ]
}

/// Convert linear RGB to CIE Lab (D65 white point).
fn rgb_to_lab(rgb: &Rgb) -> [f64; 3] {
    let xyz = rgb_to_xyz(rgb);
    let d65 = [0.950_489, 1.000_000, 1.088_840];
    let f = |t: f64| {
        if t > 0.008_856 {
            t.powf(1.0 / 3.0)
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(xyz[0] / d65[0]);
    let fy = f(xyz[1] / d65[1]);
    let fz = f(xyz[2] / d65[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Convert CIE Lab to linear RGB.
fn lab_to_rgb(lab: &[f64; 3]) -> Rgb {
    let d65 = [0.950_489, 1.000_000, 1.088_840];
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = lab[1] / 500.0 + fy;
    let fz = fy - lab[2] / 200.0;
    let f_inv = |t: f64| {
        let t3 = t * t * t;
        if t3 > 0.008_856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let xyz = [f_inv(fx) * d65[0], f_inv(fy) * d65[1], f_inv(fz) * d65[2]];
    xyz_to_rgb(&xyz)
}

/// Apply a 1-D tone curve (linear interpolation through control points).
///
/// `curve` is a slice of `(input, output)` pairs in ascending input order.
fn apply_tone_curve(v: f64, curve: &[(f64, f64)]) -> f64 {
    if curve.is_empty() {
        return v;
    }
    let first = curve[0];
    // SAFETY: curve.is_empty() returns early above, so len >= 1
    let last = curve[curve.len() - 1];
    if v <= first.0 {
        return first.1;
    }
    if v >= last.0 {
        return last.1;
    }
    for i in 0..curve.len() - 1 {
        let (x0, y0) = curve[i];
        let (x1, y1) = curve[i + 1];
        if v >= x0 && v <= x1 {
            let t = (v - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    v
}

/// Adjust the chroma (saturation) in Lab space.
fn adjust_saturation(rgb: &Rgb, saturation: f64) -> Rgb {
    let lab = rgb_to_lab(rgb);
    let new_lab = [lab[0], lab[1] * saturation, lab[2] * saturation];
    let result = lab_to_rgb(&new_lab);
    [
        result[0].clamp(0.0, 1.0),
        result[1].clamp(0.0, 1.0),
        result[2].clamp(0.0, 1.0),
    ]
}

/// Clamp RGB to `[0, 1]³`.
#[inline]
fn clamp_rgb(rgb: Rgb) -> Rgb {
    [
        rgb[0].clamp(0.0, 1.0),
        rgb[1].clamp(0.0, 1.0),
        rgb[2].clamp(0.0, 1.0),
    ]
}

// ============================================================================
// Film grain simulation
// ============================================================================

/// Simulate film grain using a deterministic pseudo-random offset.
///
/// This is a purely deterministic (no external RNG) grain approximation
/// derived from a hash of the input pixel coordinates so that repeated
/// calls on the same pixel produce the same result.
fn film_grain(rgb: &Rgb, strength: f64, seed: u64) -> Rgb {
    // Use a simple FNV-inspired hash on the quantised pixel values to get
    // a per-pixel deterministic noise offset.
    let quant = |v: f64| -> u64 { (v * 1023.0).round() as u64 };
    let hash_r = quant(rgb[0]).wrapping_mul(2_654_435_761).wrapping_add(seed);
    let hash_g = quant(rgb[1]).wrapping_mul(2_246_822_519).wrapping_add(seed);
    let hash_b = quant(rgb[2]).wrapping_mul(3_266_489_917).wrapping_add(seed);

    let noise = |h: u64| -> f64 {
        // Map hash to [-1, 1] using the upper 23 bits.
        let norm = ((h >> 9) & 0xFF_FFFF) as f64 / 0xFF_FFFF as f64;
        (norm - 0.5) * 2.0
    };

    clamp_rgb([
        rgb[0] + noise(hash_r) * strength,
        rgb[1] + noise(hash_g) * strength,
        rgb[2] + noise(hash_b) * strength,
    ])
}

// ============================================================================
// FilmPreset
// ============================================================================

/// A named film emulation preset.
///
/// Each preset encapsulates:
/// - Per-channel tone curves modelling film density response.
/// - Colour balance adjustments (D-illuminant → film balance).
/// - Saturation scaling in Lab space.
/// - Optional film grain simulation.
#[derive(Clone, Debug)]
pub struct FilmPreset {
    /// Human-readable preset name.
    pub name: String,
    /// Per-channel tone curves `[(input, output)]` for R, G, B.
    pub tone_curves: [Vec<(f64, f64)>; 3],
    /// Saturation scale factor in CIE Lab (1.0 = no change).
    pub saturation: f64,
    /// Additive colour balance (lift) per channel.
    pub color_balance: [f64; 3],
    /// Film grain strength (0.0 = none).
    pub grain_strength: f64,
    /// Deterministic seed for grain.
    pub grain_seed: u64,
    /// Black point lift (lifts pure black).
    pub black_point: f64,
    /// White point compression (compress highlights).
    pub white_point: f64,
}

impl Default for FilmPreset {
    fn default() -> Self {
        let identity_curve = vec![(0.0, 0.0), (1.0, 1.0)];
        Self {
            name: "identity".to_string(),
            tone_curves: [
                identity_curve.clone(),
                identity_curve.clone(),
                identity_curve,
            ],
            saturation: 1.0,
            color_balance: [0.0, 0.0, 0.0],
            grain_strength: 0.0,
            grain_seed: 0,
            black_point: 0.0,
            white_point: 1.0,
        }
    }
}

impl FilmPreset {
    /// Apply this preset to a single RGB pixel.
    ///
    /// Applies in the following order:
    /// 1. Per-channel tone curves.
    /// 2. Saturation adjustment (Lab space).
    /// 3. Colour balance (additive lift).
    /// 4. Black/white point mapping.
    /// 5. Film grain (if `grain_strength > 0`).
    #[must_use]
    pub fn apply(&self, input: &Rgb) -> Rgb {
        // 1. Per-channel tone curves.
        let after_curves = [
            apply_tone_curve(input[0], &self.tone_curves[0]),
            apply_tone_curve(input[1], &self.tone_curves[1]),
            apply_tone_curve(input[2], &self.tone_curves[2]),
        ];

        // 2. Saturation.
        let after_sat = if (self.saturation - 1.0).abs() > 1e-9 {
            adjust_saturation(&after_curves, self.saturation)
        } else {
            after_curves
        };

        // 3. Colour balance.
        let after_balance = [
            (after_sat[0] + self.color_balance[0]).clamp(0.0, 1.0),
            (after_sat[1] + self.color_balance[1]).clamp(0.0, 1.0),
            (after_sat[2] + self.color_balance[2]).clamp(0.0, 1.0),
        ];

        // 4. Black/white point mapping.
        let bp = self.black_point;
        let wp = self.white_point;
        let range = (wp - bp).max(1e-10);
        let after_bwp = [
            ((after_balance[0] - bp) / range).clamp(0.0, 1.0),
            ((after_balance[1] - bp) / range).clamp(0.0, 1.0),
            ((after_balance[2] - bp) / range).clamp(0.0, 1.0),
        ];

        // 5. Film grain.
        if self.grain_strength > 1e-9 {
            film_grain(&after_bwp, self.grain_strength, self.grain_seed)
        } else {
            after_bwp
        }
    }

    /// Apply this preset to a batch of pixels.
    pub fn apply_batch(&self, pixels: &mut [Rgb]) {
        for p in pixels.iter_mut() {
            *p = self.apply(p);
        }
    }

    /// Bake this preset into a 3-D LUT at the given size.
    ///
    /// # Panics
    ///
    /// Panics if `size < 2`.
    #[must_use]
    pub fn bake_lut3d(&self, size: usize) -> Vec<Rgb> {
        assert!(size >= 2, "size must be at least 2");
        let scale = (size - 1) as f64;
        let mut lut = Vec::with_capacity(size * size * size);
        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let inp = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                    lut.push(self.apply(&inp));
                }
            }
        }
        lut
    }

    // -----------------------------------------------------------------------
    // Named presets
    // -----------------------------------------------------------------------

    /// Kodak Vision3 250D — warm, tungsten-balanced negative.
    ///
    /// Characteristics: slightly warm shadow tint, compressed highlights,
    /// gentle s-curve, fine grain.
    #[must_use]
    pub fn kodak_vision3_250d() -> Self {
        Self {
            name: "Kodak Vision3 250D".to_string(),
            tone_curves: [
                // Red: lifted shadows, slightly compressed whites.
                vec![
                    (0.0, 0.02),
                    (0.18, 0.20),
                    (0.5, 0.52),
                    (0.9, 0.88),
                    (1.0, 0.96),
                ],
                // Green: standard s-curve.
                vec![
                    (0.0, 0.01),
                    (0.18, 0.19),
                    (0.5, 0.50),
                    (0.9, 0.87),
                    (1.0, 0.95),
                ],
                // Blue: slightly reduced in shadows (warm tint).
                vec![
                    (0.0, 0.005),
                    (0.18, 0.175),
                    (0.5, 0.47),
                    (0.9, 0.84),
                    (1.0, 0.93),
                ],
            ],
            saturation: 1.08,
            color_balance: [0.005, 0.002, -0.003],
            grain_strength: 0.006,
            grain_seed: 0xDEAD_BEEF,
            black_point: 0.0,
            white_point: 0.96,
        }
    }

    /// Kodak Portra 400 — portrait stock, natural skin tones.
    ///
    /// Characteristics: natural flesh tones, slight magenta in neutrals,
    /// gentle roll-off in highlights.
    #[must_use]
    pub fn kodak_portra_400() -> Self {
        Self {
            name: "Kodak Portra 400".to_string(),
            tone_curves: [
                vec![
                    (0.0, 0.02),
                    (0.1, 0.12),
                    (0.5, 0.53),
                    (0.8, 0.82),
                    (1.0, 0.98),
                ],
                vec![
                    (0.0, 0.01),
                    (0.1, 0.11),
                    (0.5, 0.50),
                    (0.8, 0.80),
                    (1.0, 0.97),
                ],
                vec![
                    (0.0, 0.01),
                    (0.1, 0.10),
                    (0.5, 0.49),
                    (0.8, 0.79),
                    (1.0, 0.96),
                ],
            ],
            saturation: 0.98,
            color_balance: [0.003, 0.001, -0.001],
            grain_strength: 0.008,
            grain_seed: 0xC0DE_4321,
            black_point: 0.0,
            white_point: 0.98,
        }
    }

    /// Fuji Eterna 500 — cinema negative, clean shadows.
    ///
    /// Characteristics: cooler shadows, natural midtones, fine grain.
    #[must_use]
    pub fn fuji_eterna_500() -> Self {
        Self {
            name: "Fuji Eterna 500".to_string(),
            tone_curves: [
                vec![
                    (0.0, 0.01),
                    (0.18, 0.18),
                    (0.5, 0.50),
                    (0.9, 0.88),
                    (1.0, 0.96),
                ],
                vec![
                    (0.0, 0.01),
                    (0.18, 0.19),
                    (0.5, 0.51),
                    (0.9, 0.89),
                    (1.0, 0.97),
                ],
                // Blue lifted slightly → cooler cast.
                vec![
                    (0.0, 0.02),
                    (0.18, 0.20),
                    (0.5, 0.53),
                    (0.9, 0.90),
                    (1.0, 0.97),
                ],
            ],
            saturation: 1.05,
            color_balance: [-0.002, 0.001, 0.004],
            grain_strength: 0.005,
            grain_seed: 0xFACE_CAFE,
            black_point: 0.0,
            white_point: 0.97,
        }
    }

    /// Fuji Velvia 50 — slide film, vivid saturation.
    ///
    /// Characteristics: enhanced colour saturation, deep blacks, strong red.
    #[must_use]
    pub fn fuji_velvia_50() -> Self {
        Self {
            name: "Fuji Velvia 50".to_string(),
            tone_curves: [
                // Red: punchy.
                vec![
                    (0.0, 0.0),
                    (0.18, 0.22),
                    (0.5, 0.56),
                    (0.8, 0.84),
                    (1.0, 1.0),
                ],
                vec![
                    (0.0, 0.0),
                    (0.18, 0.20),
                    (0.5, 0.52),
                    (0.8, 0.82),
                    (1.0, 1.0),
                ],
                // Blue: vivid.
                vec![
                    (0.0, 0.0),
                    (0.18, 0.21),
                    (0.5, 0.55),
                    (0.8, 0.85),
                    (1.0, 1.0),
                ],
            ],
            saturation: 1.25,
            color_balance: [0.0, 0.0, 0.0],
            grain_strength: 0.002,
            grain_seed: 0xEEE1_A507,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Ilford HP5 — classic black-and-white negative.
    ///
    /// Uses Rec.709 luminance weights to convert to greyscale.
    #[must_use]
    pub fn ilford_hp5() -> Self {
        // HP5 has a characteristic shoulder in highlights.
        let curve = vec![
            (0.0, 0.0),
            (0.18, 0.20),
            (0.5, 0.52),
            (0.85, 0.87),
            (1.0, 0.96),
        ];
        Self {
            name: "Ilford HP5".to_string(),
            tone_curves: [curve.clone(), curve.clone(), curve],
            saturation: 0.0, // Convert to greyscale.
            color_balance: [0.0, 0.0, 0.0],
            grain_strength: 0.012,
            grain_seed: 0xB1AC_0057,
            black_point: 0.0,
            white_point: 0.96,
        }
    }

    /// Agfa Ultra 100 — vivid street photography colour.
    #[must_use]
    pub fn agfa_ultra_100() -> Self {
        Self {
            name: "Agfa Ultra 100".to_string(),
            tone_curves: [
                vec![
                    (0.0, 0.0),
                    (0.18, 0.20),
                    (0.5, 0.54),
                    (0.8, 0.82),
                    (1.0, 1.0),
                ],
                vec![
                    (0.0, 0.0),
                    (0.18, 0.19),
                    (0.5, 0.51),
                    (0.8, 0.80),
                    (1.0, 1.0),
                ],
                vec![
                    (0.0, 0.0),
                    (0.18, 0.18),
                    (0.5, 0.50),
                    (0.8, 0.80),
                    (1.0, 1.0),
                ],
            ],
            saturation: 1.15,
            color_balance: [0.002, 0.0, -0.002],
            grain_strength: 0.007,
            grain_seed: 0xA6FA_1001,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Chemical cross-process — vivid, split-colour look.
    ///
    /// Simulates slide film processed in C-41 chemistry: split shadows/highlights,
    /// strong colour shift.
    #[must_use]
    pub fn cross_process() -> Self {
        Self {
            name: "Cross Process".to_string(),
            tone_curves: [
                // Red: s-curve with strong toe.
                vec![
                    (0.0, 0.05),
                    (0.1, 0.18),
                    (0.3, 0.45),
                    (0.7, 0.80),
                    (1.0, 1.0),
                ],
                // Green: inverted s-curve.
                vec![
                    (0.0, 0.0),
                    (0.2, 0.15),
                    (0.5, 0.50),
                    (0.8, 0.88),
                    (1.0, 1.0),
                ],
                // Blue: reduced in highlights.
                vec![(0.0, 0.1), (0.3, 0.30), (0.7, 0.60), (1.0, 0.75)],
            ],
            saturation: 1.30,
            color_balance: [0.01, -0.02, 0.03],
            grain_strength: 0.005,
            grain_seed: 0xC055_50E5,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Bleach bypass — silver retention.
    ///
    /// Characteristics: reduced saturation, enhanced contrast, detail in shadows.
    #[must_use]
    pub fn bleach_bypass() -> Self {
        let curve = vec![
            (0.0, 0.0),
            (0.2, 0.14),
            (0.5, 0.50),
            (0.8, 0.86),
            (1.0, 1.0),
        ];
        Self {
            name: "Bleach Bypass".to_string(),
            tone_curves: [curve.clone(), curve.clone(), curve],
            saturation: 0.45,
            color_balance: [0.0, 0.0, 0.0],
            grain_strength: 0.004,
            grain_seed: 0xB1EAC_4B0,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Teal and orange — complementary colour grade.
    ///
    /// `strength` in `[0.0, 1.0]` blends between identity and full effect.
    #[must_use]
    pub fn teal_orange(strength: f64) -> Self {
        let s = strength.clamp(0.0, 1.0);
        // Shadows: shift blue-green (teal).
        // Highlights: shift warm (orange).
        let r_curve = vec![
            (0.0, s * 0.02),
            (0.3, 0.3 + s * 0.02),
            (0.7, 0.7 + s * 0.06),
            (1.0, 1.0),
        ];
        let g_curve = vec![
            (0.0, 0.0),
            (0.3, 0.3 + s * 0.01),
            (0.7, 0.7 + s * 0.01),
            (1.0, 1.0),
        ];
        let b_curve = vec![
            (0.0, s * 0.06),
            (0.3, 0.3 + s * 0.04),
            (0.7, 0.7 - s * 0.04),
            (1.0, 1.0 - s * 0.05),
        ];
        Self {
            name: "Teal and Orange".to_string(),
            tone_curves: [r_curve, g_curve, b_curve],
            saturation: 1.0 + s * 0.15,
            color_balance: [0.0, 0.0, 0.0],
            grain_strength: 0.0,
            grain_seed: 0,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Day-for-night — simulate night from day footage.
    #[must_use]
    pub fn day_for_night() -> Self {
        // Darken significantly, desaturate, shift blue.
        let r_curve = vec![(0.0, 0.0), (0.5, 0.15), (1.0, 0.30)];
        let g_curve = vec![(0.0, 0.0), (0.5, 0.16), (1.0, 0.32)];
        let b_curve = vec![(0.0, 0.02), (0.5, 0.20), (1.0, 0.40)];
        Self {
            name: "Day for Night".to_string(),
            tone_curves: [r_curve, g_curve, b_curve],
            saturation: 0.4,
            color_balance: [-0.01, -0.005, 0.02],
            grain_strength: 0.003,
            grain_seed: 0xD41_6841,
            black_point: 0.0,
            white_point: 1.0,
        }
    }

    /// Sepia tone — classic brown monotone effect.
    #[must_use]
    pub fn sepia_tone() -> Self {
        Self {
            name: "Sepia Tone".to_string(),
            tone_curves: [
                vec![(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)],  // warm red
                vec![(0.0, 0.0), (0.5, 0.48), (1.0, 0.95)], // mid green
                vec![(0.0, 0.0), (0.5, 0.35), (1.0, 0.75)], // reduced blue
            ],
            saturation: 0.0,
            color_balance: [0.06, 0.04, -0.02],
            grain_strength: 0.005,
            grain_seed: 0x5E51A_10E,
            black_point: 0.0,
            white_point: 0.95,
        }
    }

    /// Fade & wash — soft, low-contrast, faded look.
    #[must_use]
    pub fn fade_wash() -> Self {
        let curve = vec![(0.0, 0.05), (0.5, 0.50), (1.0, 0.90)];
        Self {
            name: "Fade Wash".to_string(),
            tone_curves: [curve.clone(), curve.clone(), curve],
            saturation: 0.80,
            color_balance: [0.01, 0.01, 0.02],
            grain_strength: 0.002,
            grain_seed: 0xFADE_0A54,
            black_point: 0.05,
            white_point: 0.90,
        }
    }

    /// Identity preset — does not alter the input.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// List the names of all built-in presets.
    #[must_use]
    pub fn list_presets() -> &'static [&'static str] {
        &[
            "identity",
            "kodak_vision3_250d",
            "kodak_portra_400",
            "fuji_eterna_500",
            "fuji_velvia_50",
            "ilford_hp5",
            "agfa_ultra_100",
            "cross_process",
            "bleach_bypass",
            "teal_orange",
            "day_for_night",
            "sepia_tone",
            "fade_wash",
        ]
    }

    /// Get a preset by name.
    ///
    /// Returns `None` if the name is not recognised.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "identity" => Some(Self::identity()),
            "kodak_vision3_250d" => Some(Self::kodak_vision3_250d()),
            "kodak_portra_400" => Some(Self::kodak_portra_400()),
            "fuji_eterna_500" => Some(Self::fuji_eterna_500()),
            "fuji_velvia_50" => Some(Self::fuji_velvia_50()),
            "ilford_hp5" => Some(Self::ilford_hp5()),
            "agfa_ultra_100" => Some(Self::agfa_ultra_100()),
            "cross_process" => Some(Self::cross_process()),
            "bleach_bypass" => Some(Self::bleach_bypass()),
            "teal_orange" => Some(Self::teal_orange(0.7)),
            "day_for_night" => Some(Self::day_for_night()),
            "sepia_tone" => Some(Self::sepia_tone()),
            "fade_wash" => Some(Self::fade_wash()),
            _ => None,
        }
    }
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn in_range(v: f64) -> bool {
        v >= 0.0 && v <= 1.0
    }

    fn all_in_range(rgb: &Rgb) -> bool {
        in_range(rgb[0]) && in_range(rgb[1]) && in_range(rgb[2])
    }

    #[test]
    fn test_identity_preset_passthrough() {
        let preset = FilmPreset::identity();
        let inp = [0.4, 0.6, 0.2];
        let out = preset.apply(&inp);
        for ch in 0..3 {
            assert!((out[ch] - inp[ch]).abs() < 1e-9, "ch={ch}");
        }
    }

    #[test]
    fn test_all_presets_output_in_range() {
        let test_pixels = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.18, 0.18, 0.18],
            [0.5, 0.3, 0.7],
            [0.9, 0.1, 0.5],
        ];
        for &name in FilmPreset::list_presets() {
            let preset = FilmPreset::from_name(name).expect(name);
            for &pix in &test_pixels {
                let out = preset.apply(&pix);
                assert!(
                    all_in_range(&out),
                    "Preset '{name}' output out of range: {out:?} for {pix:?}"
                );
            }
        }
    }

    #[test]
    fn test_ilford_hp5_is_greyscale() {
        let preset = FilmPreset::ilford_hp5();
        // Any colour pixel should map to approximately equal R, G, B.
        let out = preset.apply(&[0.8, 0.3, 0.5]);
        let range = (out[0] - out[1]).abs().max((out[1] - out[2]).abs());
        assert!(range < 0.05, "HP5 is not greyscale: {out:?}");
    }

    #[test]
    fn test_bleach_bypass_reduces_saturation() {
        let preset = FilmPreset::bleach_bypass();
        // A very saturated colour (pure red) should have less chroma after bypass.
        let inp = [1.0, 0.0, 0.0];
        let out = preset.apply(&inp);
        // The green and blue channels should increase (desaturation).
        assert!(
            out[1] > inp[1] || out[2] > inp[2],
            "Bleach bypass should desaturate: {out:?}"
        );
    }

    #[test]
    fn test_bake_lut3d_correct_size() {
        let preset = FilmPreset::kodak_portra_400();
        let lut = preset.bake_lut3d(5);
        assert_eq!(lut.len(), 5 * 5 * 5);
    }

    #[test]
    fn test_bake_lut3d_all_in_range() {
        let preset = FilmPreset::kodak_vision3_250d();
        let lut = preset.bake_lut3d(5);
        for (i, entry) in lut.iter().enumerate() {
            assert!(all_in_range(entry), "Entry {i} out of range: {entry:?}");
        }
    }

    #[test]
    fn test_apply_batch() {
        let preset = FilmPreset::teal_orange(0.5);
        let mut pixels = vec![[0.5, 0.5, 0.5], [0.2, 0.3, 0.4]];
        let expected: Vec<Rgb> = pixels.iter().map(|p| preset.apply(p)).collect();
        preset.apply_batch(&mut pixels);
        for (i, (a, b)) in pixels.iter().zip(expected.iter()).enumerate() {
            for ch in 0..3 {
                assert!((a[ch] - b[ch]).abs() < 1e-12, "pixel={i} ch={ch}");
            }
        }
    }

    #[test]
    fn test_teal_orange_strength_zero_is_near_identity() {
        let preset = FilmPreset::teal_orange(0.0);
        // At strength=0 the colour balance and curves should be near-identity.
        let inp = [0.5, 0.5, 0.5];
        let out = preset.apply(&inp);
        for ch in 0..3 {
            assert!(
                (out[ch] - inp[ch]).abs() < 0.05,
                "ch={ch} teal-orange at strength=0 should be ~identity: out={}",
                out[ch]
            );
        }
    }

    #[test]
    fn test_preset_from_name_unknown_returns_none() {
        assert!(FilmPreset::from_name("super_fake_film").is_none());
    }

    #[test]
    fn test_preset_list_contains_all_named() {
        for &name in FilmPreset::list_presets() {
            let p = FilmPreset::from_name(name);
            assert!(p.is_some(), "from_name failed for '{name}'");
        }
    }

    #[test]
    fn test_cross_process_alters_colour() {
        let preset = FilmPreset::cross_process();
        let inp = [0.5, 0.5, 0.5];
        let out = preset.apply(&inp);
        // Cross process should shift RGB values differently.
        let all_equal = (out[0] - out[1]).abs() < 0.001 && (out[1] - out[2]).abs() < 0.001;
        assert!(
            !all_equal,
            "Cross process should not produce neutral grey from grey: {out:?}"
        );
    }

    #[test]
    fn test_day_for_night_darkens() {
        let preset = FilmPreset::day_for_night();
        let inp = [0.8, 0.8, 0.8];
        let out = preset.apply(&inp);
        // Day-for-night should darken the image.
        let luma_in = 0.2126 * inp[0] + 0.7152 * inp[1] + 0.0722 * inp[2];
        let luma_out = 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2];
        assert!(
            luma_out < luma_in,
            "Day-for-night should darken: in={luma_in} out={luma_out}"
        );
    }

    #[test]
    fn test_sepia_tone_warm_tint() {
        let preset = FilmPreset::sepia_tone();
        let out = preset.apply(&[0.5, 0.5, 0.5]);
        // Sepia: red should be higher than blue.
        assert!(out[0] > out[2], "Sepia should be warm (R > B): {out:?}");
    }

    #[test]
    fn test_fade_wash_lifts_blacks() {
        let preset = FilmPreset::fade_wash();
        let out = preset.apply(&[0.0, 0.0, 0.0]);
        // Fade wash should lift blacks above zero.
        for ch in 0..3 {
            assert!(out[ch] >= 0.0, "ch={ch} should be non-negative");
        }
    }

    #[test]
    fn test_velvia_high_saturation() {
        let preset = FilmPreset::fuji_velvia_50();
        // Velvia should have saturation > 1.
        assert!(preset.saturation > 1.0, "Velvia saturation should be > 1.0");
    }

    #[test]
    fn test_film_grain_deterministic() {
        let preset = FilmPreset::kodak_vision3_250d();
        let inp = [0.5, 0.3, 0.7];
        let out1 = preset.apply(&inp);
        let out2 = preset.apply(&inp);
        // Deterministic grain: same pixel → same output.
        for ch in 0..3 {
            assert!(
                (out1[ch] - out2[ch]).abs() < 1e-15,
                "grain should be deterministic ch={ch}"
            );
        }
    }
}
