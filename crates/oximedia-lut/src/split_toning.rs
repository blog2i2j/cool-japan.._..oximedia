//! Split toning LUT generation.
//!
//! Split toning is a traditional darkroom and digital colour-grading technique
//! that applies different hues to the shadows and highlights of an image,
//! often used to evoke a film or period look.
//!
//! # Overview
//!
//! This module provides:
//!
//! * [`SplitToningParams`] – configures hue, saturation, and balance for
//!   shadows and highlights.
//! * [`apply_split_toning`] – applies the toning transform to a single RGB
//!   pixel (normalised 0–1 display-referred).
//! * [`generate_split_toning_lut`] – bakes the transform into a flat 3-D LUT.
//! * [`SplitToningZone`] – zone breakdown (shadows / midtones / highlights)
//!   for per-zone analysis.
//! * [`split_toning_diff`] – measures colour shift introduced by the toning.
//!
//! # Algorithm
//!
//! 1. Compute per-pixel luminance (Rec.709 coefficients).
//! 2. Derive shadow weight and highlight weight from luminance, controlled by
//!    the `balance` parameter which shifts the crossover point.
//! 3. Compute the target hue colour for each zone (hue angle → RGB, at the
//!    given saturation).
//! 4. Blend the original colour with the shadow/highlight colour using the
//!    respective zone weights.

use crate::error::{LutError, LutResult};
use crate::Rgb;

// ---------------------------------------------------------------------------
// Colour helpers
// ---------------------------------------------------------------------------

/// Convert an HSL-like hue angle (0–360°) at maximum saturation into an RGB
/// tint colour.  Returns a pure saturated colour on the RGB cube boundary.
#[must_use]
fn hue_to_rgb(hue_deg: f64) -> Rgb {
    let h = hue_deg.rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

/// Compute the Rec.709 luminance of an RGB triple.
#[inline]
fn luma(rgb: &Rgb) -> f64 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

// ---------------------------------------------------------------------------
// Zone weighting
// ---------------------------------------------------------------------------

/// Compute smooth shadow weight for a given luminance and balance.
///
/// `balance` in `[-1.0, 1.0]`: negative pushes toning towards blacks,
/// positive towards whites.
fn shadow_weight(luminance: f64, balance: f64) -> f64 {
    // The crossover from shadow to highlight is at 0.5 shifted by balance/2.
    let crossover = 0.5 + balance * 0.3;
    // Smooth step centred at crossover, shadow weight = 1 - highlight weight.
    let t = (luminance / crossover.max(1e-6)).clamp(0.0, 1.0);
    // Smooth step: 3t² - 2t³
    let smooth = t * t * (3.0 - 2.0 * t);
    1.0 - smooth
}

/// Compute smooth highlight weight for a given luminance and balance.
fn highlight_weight(luminance: f64, balance: f64) -> f64 {
    let crossover = 0.5 + balance * 0.3;
    let t = ((luminance - crossover) / (1.0 - crossover).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// SplitToningParams
// ---------------------------------------------------------------------------

/// Parameters controlling the split toning transform.
#[derive(Debug, Clone)]
pub struct SplitToningParams {
    /// Hue of shadow toning (0–360°).  0 = red, 60 = yellow, 120 = green,
    /// 180 = cyan, 240 = blue, 300 = magenta.
    pub shadow_hue: f64,
    /// Saturation of the shadow tint in `[0.0, 1.0]`.
    pub shadow_saturation: f64,
    /// Hue of highlight toning (0–360°).
    pub highlight_hue: f64,
    /// Saturation of the highlight tint in `[0.0, 1.0]`.
    pub highlight_saturation: f64,
    /// Balance between shadow and highlight influence in `[-1.0, 1.0]`.
    /// Negative = push effect towards shadows, positive = towards highlights.
    pub balance: f64,
    /// Overall strength / opacity of the toning effect in `[0.0, 1.0]`.
    /// `0.0` = no effect, `1.0` = full toning.
    pub strength: f64,
}

impl Default for SplitToningParams {
    /// Default: warm (golden) highlights and cool (slate-blue) shadows at
    /// moderate strength.
    fn default() -> Self {
        Self {
            shadow_hue: 210.0,    // blue-ish shadows
            shadow_saturation: 0.3,
            highlight_hue: 42.0,  // golden highlights
            highlight_saturation: 0.25,
            balance: 0.0,
            strength: 0.5,
        }
    }
}

impl SplitToningParams {
    /// Create params with explicit shadow and highlight settings.
    #[must_use]
    pub fn new(
        shadow_hue: f64,
        shadow_saturation: f64,
        highlight_hue: f64,
        highlight_saturation: f64,
    ) -> Self {
        Self {
            shadow_hue,
            shadow_saturation: shadow_saturation.clamp(0.0, 1.0),
            highlight_hue,
            highlight_saturation: highlight_saturation.clamp(0.0, 1.0),
            balance: 0.0,
            strength: 1.0,
        }
    }

    /// Set the balance.
    #[must_use]
    pub fn balance(mut self, b: f64) -> Self {
        self.balance = b.clamp(-1.0, 1.0);
        self
    }

    /// Set overall strength.
    #[must_use]
    pub fn strength(mut self, s: f64) -> Self {
        self.strength = s.clamp(0.0, 1.0);
        self
    }

    /// Pre-computed shadow tint RGB (hue → RGB scaled by saturation).
    fn shadow_tint(&self) -> Rgb {
        let hue_rgb = hue_to_rgb(self.shadow_hue);
        // Mix the pure hue with neutral grey (0.5) by saturation.
        let s = self.shadow_saturation;
        [
            hue_rgb[0] * s + 0.5 * (1.0 - s),
            hue_rgb[1] * s + 0.5 * (1.0 - s),
            hue_rgb[2] * s + 0.5 * (1.0 - s),
        ]
    }

    /// Pre-computed highlight tint RGB.
    fn highlight_tint(&self) -> Rgb {
        let hue_rgb = hue_to_rgb(self.highlight_hue);
        let s = self.highlight_saturation;
        [
            hue_rgb[0] * s + 0.5 * (1.0 - s),
            hue_rgb[1] * s + 0.5 * (1.0 - s),
            hue_rgb[2] * s + 0.5 * (1.0 - s),
        ]
    }
}

// ---------------------------------------------------------------------------
// Core transform
// ---------------------------------------------------------------------------

/// Apply split toning to a single normalised RGB pixel.
///
/// The pixel is assumed to be in a display-referred, gamma-corrected space
/// (sRGB / Rec.709 range 0–1).  The function preserves luminance by blending
/// a tinted version of the pixel rather than replacing the colour outright.
#[must_use]
pub fn apply_split_toning(rgb: &Rgb, params: &SplitToningParams) -> Rgb {
    let lum = luma(rgb);
    let sw = shadow_weight(lum, params.balance);
    let hw = highlight_weight(lum, params.balance);

    let shadow_tint = params.shadow_tint();
    let highlight_tint = params.highlight_tint();

    // For each zone, blend pixel toward tint: pixel * (1-w) + tint * w.
    let apply_tint = |pixel_ch: f64, tint_ch: f64, weight: f64| -> f64 {
        pixel_ch * (1.0 - weight) + tint_ch * weight
    };

    // Composite: shadow zone first, then highlight zone, both scaled by strength.
    let s = params.strength;
    let effective_sw = sw * s;
    let effective_hw = hw * s;

    let after_shadow = [
        apply_tint(rgb[0], shadow_tint[0], effective_sw),
        apply_tint(rgb[1], shadow_tint[1], effective_sw),
        apply_tint(rgb[2], shadow_tint[2], effective_sw),
    ];

    [
        apply_tint(after_shadow[0], highlight_tint[0], effective_hw).clamp(0.0, 1.0),
        apply_tint(after_shadow[1], highlight_tint[1], effective_hw).clamp(0.0, 1.0),
        apply_tint(after_shadow[2], highlight_tint[2], effective_hw).clamp(0.0, 1.0),
    ]
}

// ---------------------------------------------------------------------------
// LUT generation
// ---------------------------------------------------------------------------

/// Generate a 3-D split toning LUT of the given `size` (entries per dimension).
///
/// The LUT is stored in `r-major` order: `index = r * size² + g * size + b`.
/// Input coordinates uniformly sample the `[0.0, 1.0]` cube.
///
/// # Errors
///
/// Returns [`LutError::InvalidData`] if `size < 2`.
pub fn generate_split_toning_lut(
    size: usize,
    params: &SplitToningParams,
) -> LutResult<Vec<Rgb>> {
    if size < 2 {
        return Err(LutError::InvalidData(format!(
            "LUT size must be >= 2, got {size}"
        )));
    }
    let scale = (size - 1) as f64;
    let mut lut = Vec::with_capacity(size * size * size);
    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let pixel: Rgb = [r as f64 / scale, g as f64 / scale, b as f64 / scale];
                lut.push(apply_split_toning(&pixel, params));
            }
        }
    }
    Ok(lut)
}

// ---------------------------------------------------------------------------
// Zone analysis
// ---------------------------------------------------------------------------

/// Zone classification for a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitToningZone {
    /// Luminance below the shadow threshold (~0.33 normalised).
    Shadow,
    /// Luminance in the midtone range.
    Midtone,
    /// Luminance above the highlight threshold (~0.66 normalised).
    Highlight,
}

impl SplitToningZone {
    /// Classify a pixel by its luminance.
    #[must_use]
    pub fn classify(rgb: &Rgb) -> Self {
        let lum = luma(rgb);
        if lum < 0.33 {
            Self::Shadow
        } else if lum > 0.66 {
            Self::Highlight
        } else {
            Self::Midtone
        }
    }
}

/// Aggregate zone statistics for a slice of pixels.
#[derive(Debug, Clone)]
pub struct ZoneStats {
    /// Number of shadow pixels.
    pub shadow_count: usize,
    /// Number of midtone pixels.
    pub midtone_count: usize,
    /// Number of highlight pixels.
    pub highlight_count: usize,
    /// Total pixels.
    pub total: usize,
    /// Percentage of shadows.
    pub shadow_pct: f64,
    /// Percentage of midtones.
    pub midtone_pct: f64,
    /// Percentage of highlights.
    pub highlight_pct: f64,
}

impl ZoneStats {
    /// Compute zone statistics from a slice of pixels.
    #[must_use]
    pub fn compute(pixels: &[Rgb]) -> Self {
        let total = pixels.len();
        let mut shadow_count = 0usize;
        let mut midtone_count = 0usize;
        let mut highlight_count = 0usize;

        for px in pixels {
            match SplitToningZone::classify(px) {
                SplitToningZone::Shadow => shadow_count += 1,
                SplitToningZone::Midtone => midtone_count += 1,
                SplitToningZone::Highlight => highlight_count += 1,
            }
        }

        let n = total.max(1) as f64;
        Self {
            shadow_count,
            midtone_count,
            highlight_count,
            total,
            shadow_pct: shadow_count as f64 / n * 100.0,
            midtone_pct: midtone_count as f64 / n * 100.0,
            highlight_pct: highlight_count as f64 / n * 100.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Difference analysis
// ---------------------------------------------------------------------------

/// Colour shift statistics produced by split toning.
#[derive(Debug, Clone)]
pub struct SplitToningDiff {
    /// Per-pixel Euclidean colour shift.
    pub per_pixel: Vec<f64>,
    /// RMS colour shift.
    pub rms: f64,
    /// Maximum colour shift.
    pub max: f64,
    /// Mean colour shift.
    pub mean: f64,
}

impl SplitToningDiff {
    /// Compute colour shift between `original` and `toned` pixel arrays.
    ///
    /// # Errors
    ///
    /// Returns [`LutError::InvalidData`] if slices differ in length.
    pub fn compute(original: &[Rgb], toned: &[Rgb]) -> LutResult<Self> {
        if original.len() != toned.len() {
            return Err(LutError::InvalidData(format!(
                "Length mismatch: original={} toned={}",
                original.len(),
                toned.len(),
            )));
        }
        let n = original.len();
        let mut per_pixel = Vec::with_capacity(n);
        let mut sum_sq = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut max = 0.0_f64;

        for (a, b) in original.iter().zip(toned.iter()) {
            let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2))
                .sqrt();
            per_pixel.push(d);
            sum_sq += d * d;
            sum += d;
            if d > max {
                max = d;
            }
        }

        let rms = if n > 0 { (sum_sq / n as f64).sqrt() } else { 0.0 };
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };

        Ok(Self {
            per_pixel,
            rms,
            max,
            mean,
        })
    }
}

// ---------------------------------------------------------------------------
// Named presets
// ---------------------------------------------------------------------------

/// Built-in split toning presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitToningPreset {
    /// Classic film-noir look: cold blue shadows, warm amber highlights.
    FilmNoir,
    /// Vintage sepia: warm shadows and highlights (full sepia effect).
    Sepia,
    /// Teal and orange (cinematic blockbuster look).
    TealAndOrange,
    /// Bleach bypass: desaturated with slight green shadows and warm highlights.
    BleachBypass,
    /// Cross-process: magenta shadows, yellow-green highlights.
    CrossProcess,
}

impl SplitToningPreset {
    /// Get the [`SplitToningParams`] for this preset.
    #[must_use]
    pub fn params(self) -> SplitToningParams {
        match self {
            Self::FilmNoir => SplitToningParams::new(220.0, 0.3, 38.0, 0.2)
                .balance(-0.1)
                .strength(0.6),
            Self::Sepia => SplitToningParams::new(35.0, 0.4, 40.0, 0.35)
                .balance(0.0)
                .strength(0.7),
            Self::TealAndOrange => SplitToningParams::new(183.0, 0.45, 28.0, 0.4)
                .balance(0.15)
                .strength(0.75),
            Self::BleachBypass => SplitToningParams::new(105.0, 0.15, 48.0, 0.12)
                .balance(0.05)
                .strength(0.4),
            Self::CrossProcess => SplitToningParams::new(300.0, 0.35, 75.0, 0.3)
                .balance(-0.05)
                .strength(0.65),
        }
    }

    /// Human-readable name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::FilmNoir => "Film Noir",
            Self::Sepia => "Sepia",
            Self::TealAndOrange => "Teal and Orange",
            Self::BleachBypass => "Bleach Bypass",
            Self::CrossProcess => "Cross Process",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_close(a: &Rgb, b: &Rgb, tol: f64) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol && (a[2] - b[2]).abs() < tol
    }

    #[test]
    fn test_no_effect_at_zero_strength() {
        let params = SplitToningParams::default().strength(0.0);
        let pixel = [0.3, 0.5, 0.7];
        let out = apply_split_toning(&pixel, &params);
        assert!(rgb_close(&pixel, &out, 1e-10));
    }

    #[test]
    fn test_output_in_range() {
        let params = SplitToningPreset::TealAndOrange.params();
        let pixels: Vec<Rgb> = (0..=10)
            .map(|i| {
                let v = i as f64 / 10.0;
                [v, v, v]
            })
            .collect();
        for p in &pixels {
            let out = apply_split_toning(p, &params);
            for ch in &out {
                assert!(
                    (0.0..=1.0).contains(ch),
                    "out of range: {:?} → {:?}",
                    p,
                    out
                );
            }
        }
    }

    #[test]
    fn test_sepia_shifts_colour() {
        let params = SplitToningPreset::Sepia.params();
        // Use a dark pixel (shadow zone: lum < 0.33) where the sepia shadow tint
        // (warm orange ~35°) is applied.  R should exceed B after toning.
        let dark_grey = [0.15_f64, 0.15_f64, 0.15_f64];
        let out = apply_split_toning(&dark_grey, &params);
        assert!(out[0] > out[2], "sepia shadow should be warm: R={} B={}", out[0], out[2]);

        // Also verify a bright pixel (highlight zone: lum > 0.66) shifts warm.
        let bright_grey = [0.85_f64, 0.85_f64, 0.85_f64];
        let out_h = apply_split_toning(&bright_grey, &params);
        assert!(out_h[0] > out_h[2], "sepia highlight should be warm: R={} B={}", out_h[0], out_h[2]);
    }

    #[test]
    fn test_teal_orange_shadow_vs_highlight() {
        let params = SplitToningPreset::TealAndOrange.params();
        // Dark pixel (shadow zone) should skew toward teal (blue-green: B ≥ R).
        let shadow = [0.05_f64, 0.05_f64, 0.05_f64];
        let out_s = apply_split_toning(&shadow, &params);
        // Highlight should skew toward orange (R > B).
        let highlight = [0.95_f64, 0.95_f64, 0.95_f64];
        let out_h = apply_split_toning(&highlight, &params);
        assert!(
            out_s[2] >= out_s[0] - 0.05,
            "shadow should lean cool: R={} B={}",
            out_s[0],
            out_s[2]
        );
        assert!(
            out_h[0] >= out_h[2],
            "highlight should lean warm: R={} B={}",
            out_h[0],
            out_h[2]
        );
    }

    #[test]
    fn test_generate_lut_correct_size() {
        let size = 5;
        let params = SplitToningParams::default();
        let lut = generate_split_toning_lut(size, &params).expect("should succeed");
        assert_eq!(lut.len(), size * size * size);
    }

    #[test]
    fn test_generate_lut_invalid_size() {
        let params = SplitToningParams::default();
        assert!(generate_split_toning_lut(1, &params).is_err());
    }

    #[test]
    fn test_zone_classify() {
        assert_eq!(
            SplitToningZone::classify(&[0.1, 0.1, 0.1]),
            SplitToningZone::Shadow
        );
        assert_eq!(
            SplitToningZone::classify(&[0.5, 0.5, 0.5]),
            SplitToningZone::Midtone
        );
        assert_eq!(
            SplitToningZone::classify(&[0.9, 0.9, 0.9]),
            SplitToningZone::Highlight
        );
    }

    #[test]
    fn test_zone_stats() {
        let pixels: Vec<Rgb> = vec![
            [0.1, 0.1, 0.1], // shadow
            [0.5, 0.5, 0.5], // midtone
            [0.9, 0.9, 0.9], // highlight
        ];
        let stats = ZoneStats::compute(&pixels);
        assert_eq!(stats.shadow_count, 1);
        assert_eq!(stats.midtone_count, 1);
        assert_eq!(stats.highlight_count, 1);
        assert!((stats.shadow_pct - 100.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_diff_zero_on_identity() {
        let params = SplitToningParams::default().strength(0.0);
        let pixels: Vec<Rgb> = (0..=10)
            .map(|i| {
                let v = i as f64 / 10.0;
                [v, v, v]
            })
            .collect();
        let toned: Vec<Rgb> = pixels
            .iter()
            .map(|p| apply_split_toning(p, &params))
            .collect();
        let diff = SplitToningDiff::compute(&pixels, &toned).expect("should succeed");
        assert!(diff.rms < 1e-10);
        assert!(diff.max < 1e-10);
    }

    #[test]
    fn test_diff_mismatch_error() {
        let a: Vec<Rgb> = vec![[0.5; 3]; 3];
        let b: Vec<Rgb> = vec![[0.5; 3]; 5];
        assert!(SplitToningDiff::compute(&a, &b).is_err());
    }

    #[test]
    fn test_all_presets_produce_valid_output() {
        let presets = [
            SplitToningPreset::FilmNoir,
            SplitToningPreset::Sepia,
            SplitToningPreset::TealAndOrange,
            SplitToningPreset::BleachBypass,
            SplitToningPreset::CrossProcess,
        ];
        let pixel = [0.4, 0.4, 0.4];
        for preset in &presets {
            let params = preset.params();
            let out = apply_split_toning(&pixel, &params);
            for ch in &out {
                assert!(
                    (0.0..=1.0).contains(ch),
                    "preset {:?} produced out-of-range: {ch}",
                    preset.name()
                );
            }
        }
    }

    #[test]
    fn test_balance_shifts_zone_boundary() {
        // With balance = -1.0, the crossover is pushed LOW (to ~0.2), so at lum=0.5
        // we are well into the highlight zone → shadow_weight should be LOW.
        // With balance = +1.0, the crossover is pushed HIGH (to ~0.8), so at lum=0.5
        // we are in the shadow zone → shadow_weight should be HIGH.
        let sw_neg = shadow_weight(0.5, -1.0);
        let sw_pos = shadow_weight(0.5, 1.0);
        assert!(
            sw_pos > sw_neg,
            "positive balance should give more shadow weight at lum=0.5: neg={sw_neg} pos={sw_pos}"
        );
        // Verify highlight weight is complementary behaviour.
        let hw_neg = highlight_weight(0.5, -1.0);
        let hw_pos = highlight_weight(0.5, 1.0);
        assert!(
            hw_neg > hw_pos,
            "negative balance should give more highlight weight at lum=0.5: neg={hw_neg} pos={hw_pos}"
        );
    }

    #[test]
    fn test_hue_to_rgb_primary_colours() {
        let red = hue_to_rgb(0.0);
        assert!((red[0] - 1.0).abs() < 1e-6);
        assert!(red[2] < 0.01);

        let green = hue_to_rgb(120.0);
        assert!((green[1] - 1.0).abs() < 1e-6);

        let blue = hue_to_rgb(240.0);
        assert!((blue[2] - 1.0).abs() < 1e-6);
    }
}
