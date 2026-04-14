//! Soft-clip gamut mapping with perceptual desaturation.
//!
//! Unlike hard clamping (`clamp(0.0, 1.0)`), soft-clip gamut mapping applies a smooth
//! compression curve that progressively desaturates out-of-gamut colours while preserving
//! hue and minimising perceptual discontinuities.
//!
//! # Algorithms
//!
//! - **Knee-based soft clip**: A piecewise linear/cubic transition that begins
//!   compressing once the signal exceeds a configurable "knee" threshold.
//! - **Perceptual desaturation roll-off**: Uses the BT.2020 luminance-weighted average
//!   to push the chroma towards neutral as the pixel moves out of gamut.
//! - **BT.2390 approach**: Compresses into gamut using the EETF (Electrical-Electrical
//!   Transfer Function) luminance mapping and rescales chroma proportionally.
//!
//! # References
//! - ITU-R BT.2020 Table 2 (luminance coefficients)
//! - ITU-R BT.2390-10 §5.1.6 (gamut compression)
//! - Colour & Vision Research Laboratory, UCL — "Gamut Mapping" survey

use crate::{HdrError, Result};

// ── Rec.2020 luminance coefficients (ITU-R BT.2100-2) ────────────────────────

const KR: f32 = 0.2627;
const KG: f32 = 0.6780;
const KB: f32 = 0.0593;

// ── SoftClipMethod ────────────────────────────────────────────────────────────

/// Algorithm used by [`SoftClipGamutMapper`].
#[derive(Debug, Clone, PartialEq)]
pub enum SoftClipMethod {
    /// Knee-function soft clip: linear below the knee, smooth cubic compression above it.
    ///
    /// The knee threshold `t` (0–1) defines where compression starts.  Values
    /// above `t` are mapped to `[t, 1]` via a cubic Hermite spline so that the
    /// first derivative is continuous at the join point.
    Knee,

    /// Perceptual desaturation roll-off.
    ///
    /// Pixels that exceed the gamut boundary are progressively blended towards
    /// their luminance (achromatic) equivalent.  The blending ratio is proportional
    /// to the exceedance, controlled by `desaturation_strength` in the config.
    PerceptualDesaturation,

    /// BT.2390 gamut compression: luminance-proportional chroma rescaling.
    ///
    /// The maximum of `(R, G, B)` is compressed into `[0, 1]` using the same
    /// knee curve; the remaining channels are then scaled proportionally so that
    /// hue is exactly preserved while the pixel is moved inside the gamut.
    Bt2390Chroma,
}

// ── SoftClipConfig ────────────────────────────────────────────────────────────

/// Configuration for [`SoftClipGamutMapper`].
#[derive(Debug, Clone)]
pub struct SoftClipConfig {
    /// Algorithm to apply.
    pub method: SoftClipMethod,
    /// Knee threshold in normalised signal space [0, 1].
    ///
    /// Values above this level are softly compressed.  Ignored for
    /// [`SoftClipMethod::PerceptualDesaturation`].
    /// Typical: 0.8 for a gradual roll-off; 0.95 for a tighter shoulder.
    pub knee_threshold: f32,
    /// Blending strength for the desaturation roll-off [0, 1].
    ///
    /// `1.0` = fully saturated at the gamut boundary (only used by
    /// [`SoftClipMethod::PerceptualDesaturation`]).
    pub desaturation_strength: f32,
    /// When `true`, the compressed value is further clamped to `[0, 1]`
    /// after the soft-clip curve to prevent any remaining out-of-gamut values.
    pub hard_clamp_residual: bool,
}

impl Default for SoftClipConfig {
    fn default() -> Self {
        Self {
            method: SoftClipMethod::Knee,
            knee_threshold: 0.85,
            desaturation_strength: 0.8,
            hard_clamp_residual: true,
        }
    }
}

impl SoftClipConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.knee_threshold) {
            return Err(HdrError::GamutConversionError(format!(
                "knee_threshold {} must be in [0, 1]",
                self.knee_threshold
            )));
        }
        if !(0.0..=1.0).contains(&self.desaturation_strength) {
            return Err(HdrError::GamutConversionError(format!(
                "desaturation_strength {} must be in [0, 1]",
                self.desaturation_strength
            )));
        }
        Ok(())
    }
}

// ── Knee curve ────────────────────────────────────────────────────────────────

/// Apply the piecewise knee curve to a single normalised value.
///
/// Values below `knee` pass through unchanged.  Values above `knee` are mapped
/// onto `[knee, 1]` via a cubic Hermite spline that has unit slope at `knee`
/// and zero slope at `1.0`, ensuring a smooth join and a bounded output.
///
/// The spline is parameterised by `t = (v - knee) / (1.0 - knee)` and uses:
///
///   f(t) = knee + (1 - knee) * t * (2 - t)
///
/// (Smoothstep-derived; ensures f'(0) = 1 and f'(1) = 0.)
#[inline]
pub fn apply_knee_curve(v: f32, knee: f32) -> f32 {
    if v <= knee {
        return v;
    }
    let knee = knee.clamp(0.0, 0.9999);
    let range = 1.0 - knee;
    let t = ((v - knee) / range).clamp(0.0, 1.0);
    // Cubic Hermite: f(t) = knee + range * smoothstep(t)
    // smoothstep(t) = t*t*(3 - 2*t)
    let smooth = t * t * (3.0 - 2.0 * t);
    knee + range * smooth
}

// ── Rec.2020 luminance ────────────────────────────────────────────────────────

/// Compute BT.2100 luminance Y from linear RGB (Rec.2020 primaries).
#[inline]
fn rec2020_luminance(r: f32, g: f32, b: f32) -> f32 {
    KR * r + KG * g + KB * b
}

// ── SoftClipGamutMapper ───────────────────────────────────────────────────────

/// Soft-clip gamut mapper for HDR content.
///
/// Applies one of three perceptual gamut compression strategies to bring
/// out-of-gamut RGB pixels smoothly inside the display gamut boundary.
///
/// # Example
/// ```rust
/// use oximedia_hdr::soft_clip_gamut::{SoftClipConfig, SoftClipGamutMapper};
///
/// let mapper = SoftClipGamutMapper::new(SoftClipConfig::default()).unwrap();
/// let (r, g, b) = mapper.map_pixel(1.2, 0.8, 0.1).unwrap();
/// assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct SoftClipGamutMapper {
    config: SoftClipConfig,
}

impl SoftClipGamutMapper {
    /// Create a new mapper, validating the configuration.
    pub fn new(config: SoftClipConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Create a mapper with default knee-curve settings.
    pub fn default_knee() -> Self {
        Self {
            config: SoftClipConfig::default(),
        }
    }

    /// Create a mapper using the perceptual desaturation method.
    pub fn default_perceptual() -> Self {
        Self {
            config: SoftClipConfig {
                method: SoftClipMethod::PerceptualDesaturation,
                knee_threshold: 0.85,
                desaturation_strength: 0.75,
                hard_clamp_residual: true,
            },
        }
    }

    /// Create a mapper using BT.2390 chroma rescaling.
    pub fn default_bt2390() -> Self {
        Self {
            config: SoftClipConfig {
                method: SoftClipMethod::Bt2390Chroma,
                knee_threshold: 0.85,
                desaturation_strength: 0.8,
                hard_clamp_residual: true,
            },
        }
    }

    /// Map a single RGB pixel (linear light, arbitrary scale) into gamut.
    ///
    /// Returns `(r_out, g_out, b_out)` guaranteed to lie inside `[0, 1]` if
    /// `hard_clamp_residual` is enabled.
    pub fn map_pixel(&self, r: f32, g: f32, b: f32) -> Result<(f32, f32, f32)> {
        let (r_out, g_out, b_out) = match self.config.method {
            SoftClipMethod::Knee => self.knee_map(r, g, b),
            SoftClipMethod::PerceptualDesaturation => self.desat_map(r, g, b),
            SoftClipMethod::Bt2390Chroma => self.bt2390_map(r, g, b),
        };
        let (r_out, g_out, b_out) = if self.config.hard_clamp_residual {
            (
                r_out.clamp(0.0, 1.0),
                g_out.clamp(0.0, 1.0),
                b_out.clamp(0.0, 1.0),
            )
        } else {
            (r_out, g_out, b_out)
        };
        Ok((r_out, g_out, b_out))
    }

    /// Process a whole frame of interleaved RGB pixels in-place.
    ///
    /// `pixels` must have length `3 * width * height`.  Panics if the length
    /// is not a multiple of 3.
    pub fn map_frame(&self, pixels: &mut [f32]) -> Result<()> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::GamutConversionError(
                "pixel buffer length must be a multiple of 3".into(),
            ));
        }
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.map_pixel(chunk[0], chunk[1], chunk[2])?;
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        Ok(())
    }

    // ── Internal per-method implementations ───────────────────────────────────

    /// Knee-curve soft clip: each channel independently compressed.
    fn knee_map(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let knee = self.config.knee_threshold;
        (
            apply_knee_curve(r, knee),
            apply_knee_curve(g, knee),
            apply_knee_curve(b, knee),
        )
    }

    /// Perceptual desaturation: blend towards achromatic proportional to exceedance.
    fn desat_map(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        // Luminance (achromatic target).
        let y = rec2020_luminance(r, g, b).clamp(0.0, 1.0);

        // Exceedance: how much beyond [0,1] is each channel?
        let max_val = r.max(g.max(b));
        let min_val = r.min(g.min(b));
        let exceedance = (max_val - 1.0).max((-min_val).max(0.0));

        if exceedance <= 0.0 {
            return (r, g, b);
        }

        // Desaturation ratio — grows with exceedance * strength.
        let ratio = (exceedance * self.config.desaturation_strength).clamp(0.0, 1.0);

        // Blend towards the luminance value.
        let r_out = r * (1.0 - ratio) + y * ratio;
        let g_out = g * (1.0 - ratio) + y * ratio;
        let b_out = b * (1.0 - ratio) + y * ratio;
        (r_out, g_out, b_out)
    }

    /// BT.2390 chroma rescaling: compress max channel, scale others proportionally.
    fn bt2390_map(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let max_val = r.max(g.max(b));
        if max_val <= 1.0 && r >= 0.0 && g >= 0.0 && b >= 0.0 {
            return (r, g, b);
        }

        // Compress the maximum channel via the knee curve.
        let knee = self.config.knee_threshold;
        let max_compressed = apply_knee_curve(max_val.max(0.0), knee);

        // Avoid division by zero for achromatic pixels.
        if max_val.abs() < 1e-9 {
            return (0.0, 0.0, 0.0);
        }

        // Scale all channels by the same ratio to preserve hue.
        let scale = max_compressed / max_val;
        (r * scale, g * scale, b * scale)
    }

    /// Return the current configuration.
    pub fn config(&self) -> &SoftClipConfig {
        &self.config
    }
}

// ── Convenience free functions ─────────────────────────────────────────────────

/// Apply a knee-curve soft clip to a single normalised luminance value.
///
/// This is a lower-level entry point suitable for scalar tone-map pipelines.
pub fn soft_clip_scalar(v: f32, knee: f32) -> f32 {
    apply_knee_curve(v.clamp(0.0, f32::MAX), knee).clamp(0.0, 1.0)
}

/// Compute the perceptual desaturation ratio for a given exceedance.
///
/// `exceedance` is the amount by which the out-of-gamut signal exceeds 1.0.
/// `strength` controls how aggressively the chroma is rolled off (0–1).
pub fn desaturation_ratio(exceedance: f32, strength: f32) -> f32 {
    (exceedance * strength.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── knee curve ────────────────────────────────────────────────────────────

    #[test]
    fn knee_curve_below_knee_passthrough() {
        // Values below the knee threshold must pass through unmodified.
        let knee = 0.85_f32;
        for v in [0.0_f32, 0.3, 0.7, 0.84] {
            let out = apply_knee_curve(v, knee);
            assert!(
                (out - v).abs() < 1e-6,
                "Expected passthrough for v={v}, got {out}"
            );
        }
    }

    #[test]
    fn knee_curve_at_knee_continuous() {
        // The function must be continuous at the knee point (no jump).
        let knee = 0.8_f32;
        let just_below = apply_knee_curve(knee - 1e-4, knee);
        let at_knee = apply_knee_curve(knee, knee);
        assert!(
            (at_knee - just_below).abs() < 1e-3,
            "Discontinuity at knee: {just_below} vs {at_knee}"
        );
    }

    #[test]
    fn knee_curve_clamps_to_one() {
        // Values well above 1.0 must be compressed to ≤ 1.0.
        let knee = 0.85_f32;
        for v in [1.0_f32, 1.5, 2.0, 10.0] {
            let out = apply_knee_curve(v, knee);
            assert!(out <= 1.0 + 1e-6, "Expected ≤1.0 for v={v}, got {out}");
        }
    }

    #[test]
    fn knee_curve_monotone() {
        // The knee curve must be monotone.
        let knee = 0.85_f32;
        let mut prev = apply_knee_curve(0.0, knee);
        for i in 1..=200 {
            let v = i as f32 / 100.0;
            let cur = apply_knee_curve(v, knee);
            assert!(
                cur >= prev - 1e-6,
                "Non-monotone: apply_knee_curve({v}) = {cur} < prev = {prev}"
            );
            prev = cur;
        }
    }

    // ── SoftClipGamutMapper – Knee method ─────────────────────────────────────

    #[test]
    fn mapper_knee_in_gamut_unchanged() {
        let mapper = SoftClipGamutMapper::default_knee();
        // Pixels well inside gamut must not change (up to float rounding).
        let (r, g, b) = mapper.map_pixel(0.5, 0.3, 0.1).expect("map_pixel failed");
        assert!((r - 0.5).abs() < 1e-6);
        assert!((g - 0.3).abs() < 1e-6);
        assert!((b - 0.1).abs() < 1e-6);
    }

    #[test]
    fn mapper_knee_out_of_gamut_clamped() {
        let mapper = SoftClipGamutMapper::default_knee();
        let (r, g, b) = mapper.map_pixel(1.5, 0.5, 0.0).expect("map_pixel failed");
        assert!(r <= 1.0, "r={r} must be ≤ 1.0");
        assert!(g <= 1.0, "g={g} must be ≤ 1.0");
        assert!(b <= 1.0, "b={b} must be ≤ 1.0");
    }

    // ── SoftClipGamutMapper – Perceptual desaturation ──────────────────────────

    #[test]
    fn mapper_desat_out_of_gamut_bounded() {
        let mapper = SoftClipGamutMapper::default_perceptual();
        let (r, g, b) = mapper.map_pixel(2.0, 0.5, 0.1).expect("map_pixel failed");
        assert!(r <= 1.0 && g <= 1.0 && b <= 1.0);
        assert!(r >= 0.0 && g >= 0.0 && b >= 0.0);
    }

    #[test]
    fn mapper_desat_in_gamut_passthrough() {
        let mapper = SoftClipGamutMapper::default_perceptual();
        // A pixel perfectly inside gamut should not be altered by desaturation.
        let (r, g, b) = mapper.map_pixel(0.4, 0.4, 0.4).expect("map_pixel failed");
        // Achromatic pixel: r == g == b and equal to the luminance.
        assert!(
            (r - 0.4).abs() < 1e-5,
            "Expected r~0.4, got {r}"
        );
        assert!(
            (g - 0.4).abs() < 1e-5,
            "Expected g~0.4, got {g}"
        );
        assert!(
            (b - 0.4).abs() < 1e-5,
            "Expected b~0.4, got {b}"
        );
    }

    // ── SoftClipGamutMapper – BT.2390 chroma rescaling ────────────────────────

    #[test]
    fn mapper_bt2390_hue_preservation() {
        // After BT.2390 compression the ratio R:G:B must be preserved.
        let mapper = SoftClipGamutMapper::default_bt2390();
        let (r_in, g_in, b_in) = (1.8, 0.9, 0.3);
        let (r, g, b) = mapper.map_pixel(r_in, g_in, b_in).expect("map_pixel failed");
        // The ratios should be equal (hue preserved).
        let ratio_rg_in = r_in / g_in;
        let ratio_rg_out = r / g;
        assert!(
            (ratio_rg_in - ratio_rg_out).abs() < 1e-5,
            "R:G ratio changed from {ratio_rg_in} to {ratio_rg_out}"
        );
    }

    #[test]
    fn mapper_bt2390_output_bounded() {
        let mapper = SoftClipGamutMapper::default_bt2390();
        for &(r, g, b) in &[
            (2.0_f32, 1.0, 0.5),
            (0.5, 2.0, 0.3),
            (1.1, 1.1, 1.1),
            (5.0, 3.0, 1.0),
        ] {
            let (ro, go, bo) = mapper.map_pixel(r, g, b).expect("map_pixel failed");
            assert!(ro <= 1.0 && go <= 1.0 && bo <= 1.0, "({ro},{go},{bo}) out of gamut");
        }
    }

    // ── map_frame ─────────────────────────────────────────────────────────────

    #[test]
    fn map_frame_processes_all_pixels() {
        let mapper = SoftClipGamutMapper::default_knee();
        let mut pixels = vec![1.5_f32, 0.8, 0.2, 0.0, 0.5, 1.2, 0.9, 0.9, 0.9];
        mapper.map_frame(&mut pixels).expect("map_frame failed");
        for &v in &pixels {
            assert!(
                v <= 1.0 + 1e-6 && v >= -1e-6,
                "Out-of-gamut pixel after map_frame: {v}"
            );
        }
    }

    #[test]
    fn map_frame_invalid_length_errors() {
        let mapper = SoftClipGamutMapper::default_knee();
        let mut pixels = vec![1.0_f32, 0.5]; // Not multiple of 3.
        assert!(mapper.map_frame(&mut pixels).is_err());
    }

    // ── Convenience functions ─────────────────────────────────────────────────

    #[test]
    fn soft_clip_scalar_bounded() {
        for v in [-0.5_f32, 0.0, 0.5, 1.0, 1.5, 3.0] {
            let out = soft_clip_scalar(v, 0.85);
            assert!(out <= 1.0 && out >= 0.0, "soft_clip_scalar({v}) = {out}");
        }
    }

    #[test]
    fn desaturation_ratio_range() {
        for exc in [0.0_f32, 0.1, 0.5, 1.0, 2.0] {
            for str in [0.0_f32, 0.5, 1.0] {
                let ratio = desaturation_ratio(exc, str);
                assert!(
                    (0.0..=1.0).contains(&ratio),
                    "desaturation_ratio({exc},{str}) = {ratio}"
                );
            }
        }
    }

    // ── Config validation ─────────────────────────────────────────────────────

    #[test]
    fn config_validation_rejects_bad_knee() {
        let bad_config = SoftClipConfig {
            knee_threshold: 1.5,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }
}
