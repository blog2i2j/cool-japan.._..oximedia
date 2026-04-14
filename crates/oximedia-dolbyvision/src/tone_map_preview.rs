//! Tone-mapping preview from Dolby Vision HDR to SDR/HDR10/HLG.
//!
//! Implements Reinhard-based tone mapping with configurable targets, saturation
//! preservation, and shadow roll-off. Designed for real-time preview rendering
//! and offline reference conversion.
//!
//! # Reinhard Formula
//!
//! For a given input luminance `L_in` and white-point luminance `L_white`:
//!
//! ```text
//! L_out = L_in / (1 + L_in / L_white)
//! ```
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::tone_map_preview::{
//!     ToneMapper, ToneMapConfig, ToneMapTarget,
//! };
//!
//! let config = ToneMapConfig {
//!     target: ToneMapTarget::Sdr(100.0),
//!     preserve_saturation: true,
//!     shadow_rolloff: 0.02,
//! };
//!
//! let (r, g, b) = ToneMapper::map_pixel(0.8, 0.5, 0.2, 1000.0, &config);
//! assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
//! ```

// ── Target ────────────────────────────────────────────────────────────────────

/// Tone mapping output target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapTarget {
    /// Standard Dynamic Range display with the given peak luminance in nits.
    Sdr(f32),
    /// HDR10 display with the given peak luminance in nits.
    Hdr10(f32),
    /// Hybrid Log-Gamma display (assumes 1000-nit reference peak).
    Hlg,
}

impl ToneMapTarget {
    /// Return the peak nit level associated with this target.
    ///
    /// For HLG the reference diffuse-white peak is 1000 nits.
    #[must_use]
    pub fn peak_nits(self) -> f32 {
        match self {
            Self::Sdr(nits) => nits,
            Self::Hdr10(nits) => nits,
            Self::Hlg => 1000.0,
        }
    }

    /// Return `true` if this is a standard-dynamic-range target.
    #[must_use]
    pub const fn is_sdr(self) -> bool {
        matches!(self, Self::Sdr(_))
    }
}

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for the tone mapping preview operation.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneMapConfig {
    /// Output display target.
    pub target: ToneMapTarget,
    /// When `true`, apply a hue-preserving saturation correction after mapping.
    pub preserve_saturation: bool,
    /// Shadow lift in source luminance space [0.0, 1.0]; values below this
    /// threshold receive a gentle roll-off instead of a direct map.
    pub shadow_rolloff: f32,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            target: ToneMapTarget::Sdr(100.0),
            preserve_saturation: true,
            shadow_rolloff: 0.01,
        }
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Per-frame tone mapping statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct ToneMapStats {
    /// Estimated peak input luminance in nits (assuming PQ encoding, 10 000 nit max).
    pub input_peak_nits: f32,
    /// Actual peak output luminance in nits after tone mapping.
    pub output_peak_nits: f32,
    /// Fraction of pixels whose output was clamped to [0.0, 1.0] range.
    /// Range: [0.0, 1.0].
    pub clipped_fraction: f32,
}

// ── Mapper ────────────────────────────────────────────────────────────────────

/// Tone mapper for converting Dolby Vision pixels to SDR/HDR10/HLG.
///
/// All input and output colour values are in linear light, normalised so that
/// `1.0` equals the maximum source luminance (`metadata_l1_max` × 10 000 nits).
/// After mapping the output is normalised so `1.0` equals the target peak.
pub struct ToneMapper;

impl ToneMapper {
    /// Map a single RGB pixel using Reinhard tone mapping.
    ///
    /// # Parameters
    ///
    /// - `r`, `g`, `b` — linear-light input values normalised to `[0.0, 1.0]`
    ///   relative to the source peak (10 000 nits).
    /// - `metadata_l1_max` — L1 max PQ normalised to `[0.0, 1.0]`; used to
    ///   derive the absolute source luminance.
    /// - `config` — tone mapping configuration.
    ///
    /// # Returns
    ///
    /// Tone-mapped RGB values normalised to `[0.0, 1.0]` relative to the
    /// target display peak.
    #[must_use]
    pub fn map_pixel(
        r: f32,
        g: f32,
        b: f32,
        metadata_l1_max: f32,
        config: &ToneMapConfig,
    ) -> (f32, f32, f32) {
        const SOURCE_PEAK_NITS: f32 = 10_000.0;

        let target_nits = config.target.peak_nits();
        // White point in the normalised source domain (relative to 10 000 nits)
        let l_white = target_nits / SOURCE_PEAK_NITS;

        // Scale inputs into absolute source-domain luminance
        let scale = metadata_l1_max.max(f32::EPSILON);
        let rs = r * scale;
        let gs = g * scale;
        let bs = b * scale;

        // Compute scene luminance (BT.2020 coefficients for DV)
        let luma_in = luminance_bt2020(rs, gs, bs);

        // Apply shadow roll-off: gently lift very dark regions
        let luma_lifted = apply_shadow_rolloff(luma_in, config.shadow_rolloff);

        // Reinhard tone mapping on luminance
        let luma_out = reinhard(luma_lifted, l_white);

        // Scale RGB to preserve hue (ratio of output to input luma)
        let luma_ratio = if luma_in > f32::EPSILON {
            luma_out / luma_in
        } else {
            1.0
        };

        let (rm, gm, bm) = if config.preserve_saturation {
            // Preserve saturation: scale each channel independently relative
            // to source luma, then apply the global luma ratio
            let ri = rs * luma_ratio;
            let gi = gs * luma_ratio;
            let bi = bs * luma_ratio;
            (ri, gi, bi)
        } else {
            // Desaturate toward neutral grey proportionally
            let grey = luma_out;
            let saturation = 0.85_f32; // mild desaturation for SDR feel
            (
                lerp(grey, rs * luma_ratio, saturation),
                lerp(grey, gs * luma_ratio, saturation),
                lerp(grey, bs * luma_ratio, saturation),
            )
        };

        // Clamp to target display range [0, 1]
        (rm.clamp(0.0, 1.0), gm.clamp(0.0, 1.0), bm.clamp(0.0, 1.0))
    }

    /// Apply tone mapping in-place to a slice of `(r, g, b)` pixel tuples.
    ///
    /// Each element is processed via [`Self::map_pixel`].  The buffer is
    /// modified in place.
    pub fn map_frame(pixels: &mut [(f32, f32, f32)], l1_max: f32, config: &ToneMapConfig) {
        for pixel in pixels.iter_mut() {
            let (r, g, b) = *pixel;
            *pixel = Self::map_pixel(r, g, b, l1_max, config);
        }
    }

    /// Compute tone mapping statistics for a slice of input pixels.
    ///
    /// Pixels are treated as if they have already been linearly normalised
    /// to `[0.0, ∞)` relative to a 10 000-nit source.
    ///
    /// The analysis performs a dry-run tone mapping pass to count clipped
    /// pixels without modifying the input.
    #[must_use]
    pub fn analyze(pixels: &[(f32, f32, f32)]) -> ToneMapStats {
        if pixels.is_empty() {
            return ToneMapStats {
                input_peak_nits: 0.0,
                output_peak_nits: 0.0,
                clipped_fraction: 0.0,
            };
        }

        const SOURCE_PEAK_NITS: f32 = 10_000.0;

        let mut max_luma_in: f32 = 0.0;
        let mut max_luma_out: f32 = 0.0;
        let mut clipped_count: u64 = 0;

        // Use a default 100-nit SDR config for the dry-run analysis
        let config = ToneMapConfig::default();
        let l_white = config.target.peak_nits() / SOURCE_PEAK_NITS;

        for &(r, g, b) in pixels {
            let luma = luminance_bt2020(r, g, b);
            if luma > max_luma_in {
                max_luma_in = luma;
            }

            let luma_out = reinhard(luma, l_white);
            if luma_out > max_luma_out {
                max_luma_out = luma_out;
            }

            // A pixel is "clipped" if any channel exceeded 1.0 before clamping
            if r > 1.0 || g > 1.0 || b > 1.0 {
                clipped_count += 1;
            }
        }

        let input_peak_nits = max_luma_in * SOURCE_PEAK_NITS;
        let output_peak_nits = max_luma_out * SOURCE_PEAK_NITS;
        let clipped_fraction = clipped_count as f32 / pixels.len() as f32;

        ToneMapStats {
            input_peak_nits,
            output_peak_nits,
            clipped_fraction,
        }
    }
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/// Reinhard luminance tone map: `L_out = L_in / (1 + L_in / L_white)`.
#[inline]
fn reinhard(l_in: f32, l_white: f32) -> f32 {
    l_in / (1.0 + l_in / l_white.max(f32::EPSILON))
}

/// BT.2020 luminance from linear RGB.
#[inline]
fn luminance_bt2020(r: f32, g: f32, b: f32) -> f32 {
    0.2627 * r + 0.6780 * g + 0.0593 * b
}

/// Shadow rolloff: applies a gentle toe below the rolloff threshold.
#[inline]
fn apply_shadow_rolloff(luma: f32, threshold: f32) -> f32 {
    if threshold <= 0.0 || luma >= threshold {
        return luma;
    }
    // Quadratic knee below the threshold
    let ratio = luma / threshold;
    threshold * ratio * ratio
}

/// Linear interpolation between `a` and `b` at factor `t`.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sdr_config() -> ToneMapConfig {
        ToneMapConfig {
            target: ToneMapTarget::Sdr(100.0),
            preserve_saturation: true,
            shadow_rolloff: 0.0,
        }
    }

    // -- Basic correctness -------------------------------------------------------

    #[test]
    fn test_black_stays_black() {
        let config = sdr_config();
        let (r, g, b) = ToneMapper::map_pixel(0.0, 0.0, 0.0, 1.0, &config);
        assert!(r.abs() < 1e-6, "r={r}");
        assert!(g.abs() < 1e-6, "g={g}");
        assert!(b.abs() < 1e-6, "b={b}");
    }

    #[test]
    fn test_bright_pixel_clamped_to_one() {
        let config = sdr_config();
        let (r, g, b) = ToneMapper::map_pixel(1.0, 1.0, 1.0, 1.0, &config);
        assert!(r <= 1.0, "r={r} exceeds 1.0");
        assert!(g <= 1.0, "g={g} exceeds 1.0");
        assert!(b <= 1.0, "b={b} exceeds 1.0");
    }

    #[test]
    fn test_output_never_exceeds_one() {
        let config = sdr_config();
        for magnitude in [0.1, 0.5, 0.9, 1.0, 2.0, 5.0] {
            let (r, g, b) = ToneMapper::map_pixel(magnitude, magnitude, magnitude, 1.0, &config);
            assert!(r <= 1.0, "r={r} for magnitude={magnitude}");
            assert!(g <= 1.0, "g={g} for magnitude={magnitude}");
            assert!(b <= 1.0, "b={b} for magnitude={magnitude}");
        }
    }

    #[test]
    fn test_output_non_negative() {
        let config = sdr_config();
        let (r, g, b) = ToneMapper::map_pixel(0.5, 0.3, 0.1, 0.8, &config);
        assert!(r >= 0.0, "r={r}");
        assert!(g >= 0.0, "g={g}");
        assert!(b >= 0.0, "b={b}");
    }

    // -- Reinhard formula verification ------------------------------------------

    #[test]
    fn test_reinhard_formula_midgrey() {
        // For a pure-grey pixel (equal R,G,B) the Reinhard formula should match.
        // Input: 0.5 (normalised, relative to l1_max=1.0, so 5000 nit equivalent)
        // l_white = 100.0 / 10_000.0 = 0.01
        // luma_in = 0.5 (BT2020 neutral grey)
        // Reinhard: luma_out = 0.5 / (1 + 0.5 / 0.01) = 0.5 / 51 ≈ 0.009804
        let config = ToneMapConfig {
            target: ToneMapTarget::Sdr(100.0),
            preserve_saturation: true,
            shadow_rolloff: 0.0,
        };
        let (r, g, b) = ToneMapper::map_pixel(0.5, 0.5, 0.5, 1.0, &config);
        // Output luma should be near 0.009804 (very dark for 5000-nit input on SDR)
        let luma_out = luminance_bt2020(r, g, b);
        let expected = 0.5 / (1.0 + 0.5 / 0.01_f32);
        assert!(
            (luma_out - expected).abs() < 1e-4,
            "luma_out={luma_out}, expected={expected}"
        );
    }

    #[test]
    fn test_sdr_reduces_bright_pixels() {
        let config = sdr_config();
        // A very bright pixel (close to peak) should be reduced significantly
        let (r_orig, g_orig, b_orig) = (0.9, 0.8, 0.7);
        let (r, g, b) = ToneMapper::map_pixel(r_orig, g_orig, b_orig, 1.0, &config);
        let luma_in = luminance_bt2020(r_orig, g_orig, b_orig);
        let luma_out = luminance_bt2020(r, g, b);
        assert!(
            luma_out < luma_in,
            "tone mapping should reduce bright luma: in={luma_in}, out={luma_out}"
        );
    }

    // -- Frame processing -------------------------------------------------------

    #[test]
    fn test_map_frame_modifies_pixels() {
        let config = sdr_config();
        let mut pixels = vec![(0.8, 0.6, 0.4), (0.3, 0.2, 0.1)];
        let original = pixels.clone();
        ToneMapper::map_frame(&mut pixels, 1.0, &config);
        // At least the bright pixel should have changed
        let changed = pixels
            .iter()
            .zip(original.iter())
            .any(|(&p, &o)| (p.0 - o.0).abs() > 1e-6);
        assert!(changed, "map_frame should have modified at least one pixel");
    }

    #[test]
    fn test_map_frame_empty_slice() {
        let config = sdr_config();
        let mut pixels: Vec<(f32, f32, f32)> = Vec::new();
        ToneMapper::map_frame(&mut pixels, 1.0, &config); // should not panic
    }

    // -- Statistics analysis ----------------------------------------------------

    #[test]
    fn test_analyze_clipping_fraction_all_above_one() {
        // All pixels above 1.0 → clipped_fraction == 1.0
        let pixels = vec![(2.0, 2.0, 2.0), (1.5, 1.8, 3.0)];
        let stats = ToneMapper::analyze(&pixels);
        assert!(
            (stats.clipped_fraction - 1.0).abs() < 1e-6,
            "fraction={}",
            stats.clipped_fraction
        );
    }

    #[test]
    fn test_analyze_clipping_fraction_none_above_one() {
        // All pixels in [0, 1] → clipped_fraction == 0.0
        let pixels = vec![(0.5, 0.3, 0.1), (0.9, 0.8, 0.7)];
        let stats = ToneMapper::analyze(&pixels);
        assert!(
            stats.clipped_fraction < 1e-6,
            "fraction={}",
            stats.clipped_fraction
        );
    }

    #[test]
    fn test_analyze_empty_slice() {
        let stats = ToneMapper::analyze(&[]);
        assert!((stats.input_peak_nits).abs() < f32::EPSILON);
        assert!((stats.output_peak_nits).abs() < f32::EPSILON);
        assert!((stats.clipped_fraction).abs() < f32::EPSILON);
    }

    #[test]
    fn test_analyze_input_peak_nits_scaling() {
        // A pixel with luma ≈ 0.5 should map to 0.5 * 10_000 = 5000 nit input peak
        let pixels = vec![(0.5, 0.5, 0.5)]; // BT.2020 luma = 0.5
        let stats = ToneMapper::analyze(&pixels);
        // luma ≈ 0.5 * (0.2627 + 0.6780 + 0.0593) = 0.5
        assert!(
            (stats.input_peak_nits - 5000.0).abs() < 100.0,
            "input_peak_nits={}",
            stats.input_peak_nits
        );
    }

    // -- Target variants --------------------------------------------------------

    #[test]
    fn test_hdr10_target_peak_nits() {
        assert!((ToneMapTarget::Hdr10(1000.0).peak_nits() - 1000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hlg_target_peak_nits() {
        assert!((ToneMapTarget::Hlg.peak_nits() - 1000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sdr_is_sdr() {
        assert!(ToneMapTarget::Sdr(100.0).is_sdr());
        assert!(!ToneMapTarget::Hdr10(1000.0).is_sdr());
        assert!(!ToneMapTarget::Hlg.is_sdr());
    }

    #[test]
    fn test_shadow_rolloff_toe() {
        let config = ToneMapConfig {
            target: ToneMapTarget::Sdr(100.0),
            preserve_saturation: true,
            shadow_rolloff: 0.1,
        };
        // A very dark pixel should still be non-negative
        let (r, g, b) = ToneMapper::map_pixel(0.001, 0.001, 0.001, 1.0, &config);
        assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
    }

    #[test]
    fn test_mixed_clipping_fraction() {
        // Half the pixels are above 1.0
        let pixels = vec![(1.5, 1.5, 1.5), (0.4, 0.3, 0.2)];
        let stats = ToneMapper::analyze(&pixels);
        assert!(
            (stats.clipped_fraction - 0.5).abs() < 1e-6,
            "fraction={}",
            stats.clipped_fraction
        );
    }
}
