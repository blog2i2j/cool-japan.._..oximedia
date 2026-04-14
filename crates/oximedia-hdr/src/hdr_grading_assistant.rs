//! HDR grading assistance — creative look presets for HDR content.
//!
//! Provides saturation and contrast operations in PQ space, a library of
//! creative intent presets, and serialisable creative intent metadata that can
//! accompany a graded HDR stream.
//!
//! All computations keep values in the normalised PQ signal domain (0.0–1.0)
//! where 1.0 represents 10 000 cd/m² (nits).  Call site is responsible for
//! applying the appropriate EOTF/OETF before and after grading.

use crate::{HdrError, Result};

// ─── Creative look presets ────────────────────────────────────────────────────

/// Named creative look for an HDR deliverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CreativeLook {
    /// No modification; pass-through identity grade.
    Neutral,
    /// Cinema-style warm shadows, slightly desaturated midtones, lifted blacks.
    CinemaWarm,
    /// Cool, clean look with increased contrast and blue-toned shadows.
    CoolContrast,
    /// Vivid, highly saturated colours with bright highlights.
    VividVibrant,
    /// Flat log-like appearance preserving maximum detail in highlights.
    FlatLog,
    /// Vintage film emulation with compressed highlights and warm cast.
    VintageFilm,
    /// Dramatic high-contrast black-and-white conversion.
    BwDramatic,
    /// Pastel, low-contrast desaturated look suitable for documentary.
    Documentary,
}

impl CreativeLook {
    /// Return the `HdrGradeParams` that realise this look.
    pub fn params(self) -> HdrGradeParams {
        match self {
            CreativeLook::Neutral => HdrGradeParams::identity(),
            CreativeLook::CinemaWarm => HdrGradeParams {
                pq_exposure_shift: 0.01,
                contrast_log_centre: 0.10,
                contrast_factor: 1.05,
                saturation: 0.90,
                shadow_tint: [0.05, 0.02, -0.04],
                highlight_tint: [0.03, 0.01, -0.02],
                highlight_rolloff: 0.90,
            },
            CreativeLook::CoolContrast => HdrGradeParams {
                pq_exposure_shift: 0.0,
                contrast_log_centre: 0.12,
                contrast_factor: 1.15,
                saturation: 1.05,
                shadow_tint: [-0.02, 0.0, 0.06],
                highlight_tint: [-0.01, 0.0, 0.04],
                highlight_rolloff: 0.85,
            },
            CreativeLook::VividVibrant => HdrGradeParams {
                pq_exposure_shift: 0.02,
                contrast_log_centre: 0.12,
                contrast_factor: 1.05,
                saturation: 1.35,
                shadow_tint: [0.0, 0.0, 0.0],
                highlight_tint: [0.0, 0.0, 0.0],
                highlight_rolloff: 0.95,
            },
            CreativeLook::FlatLog => HdrGradeParams {
                pq_exposure_shift: 0.0,
                contrast_log_centre: 0.18,
                contrast_factor: 0.70,
                saturation: 0.75,
                shadow_tint: [0.0, 0.0, 0.0],
                highlight_tint: [0.0, 0.0, 0.0],
                highlight_rolloff: 1.0,
            },
            CreativeLook::VintageFilm => HdrGradeParams {
                pq_exposure_shift: -0.02,
                contrast_log_centre: 0.10,
                contrast_factor: 0.90,
                saturation: 0.80,
                shadow_tint: [0.04, 0.02, -0.06],
                highlight_tint: [0.05, 0.03, -0.03],
                highlight_rolloff: 0.80,
            },
            CreativeLook::BwDramatic => HdrGradeParams {
                pq_exposure_shift: 0.0,
                contrast_log_centre: 0.12,
                contrast_factor: 1.25,
                saturation: 0.0,
                shadow_tint: [0.0, 0.0, 0.0],
                highlight_tint: [0.0, 0.0, 0.0],
                highlight_rolloff: 0.88,
            },
            CreativeLook::Documentary => HdrGradeParams {
                pq_exposure_shift: -0.01,
                contrast_log_centre: 0.14,
                contrast_factor: 0.88,
                saturation: 0.72,
                shadow_tint: [0.01, 0.01, 0.02],
                highlight_tint: [0.0, 0.01, 0.02],
                highlight_rolloff: 0.92,
            },
        }
    }
}

// ─── Grade parameters ─────────────────────────────────────────────────────────

/// Parameters for a single HDR creative grade applied in PQ signal space.
///
/// All fields operate on normalised PQ values (0.0–1.0).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HdrGradeParams {
    /// Additive offset applied to the PQ signal before all other operations.
    /// Positive values increase overall exposure.  Clamped to \[−1, 1\].
    pub pq_exposure_shift: f32,

    /// The log-domain midpoint around which the contrast S-curve pivots.
    /// Typical values: 0.10–0.20 (normalised PQ).
    pub contrast_log_centre: f32,

    /// Contrast expansion factor.  1.0 = neutral; > 1.0 increases contrast;
    /// < 1.0 compresses contrast.
    pub contrast_factor: f32,

    /// Global saturation multiplier.  1.0 = neutral; 0.0 = greyscale.
    pub saturation: f32,

    /// Additive RGB tint applied to the shadow region (PQ < 0.20).
    /// Values are in PQ signal units; typical range ±0.10.
    pub shadow_tint: [f32; 3],

    /// Additive RGB tint applied to the highlight region (PQ > 0.70).
    /// Values are in PQ signal units; typical range ±0.10.
    pub highlight_tint: [f32; 3],

    /// Soft rolloff threshold for highlight tint blending (0.0–1.0).
    /// Below this PQ value the highlight tint is smoothly blended out.
    pub highlight_rolloff: f32,
}

impl HdrGradeParams {
    /// Identity (no-op) grade.
    pub fn identity() -> Self {
        Self {
            pq_exposure_shift: 0.0,
            contrast_log_centre: 0.18,
            contrast_factor: 1.0,
            saturation: 1.0,
            shadow_tint: [0.0, 0.0, 0.0],
            highlight_tint: [0.0, 0.0, 0.0],
            highlight_rolloff: 1.0,
        }
    }

    /// Validate that all parameters are within meaningful operating ranges.
    ///
    /// Returns `Err` if any field is numerically out of range.
    pub fn validate(&self) -> Result<()> {
        if !(-1.0_f32..=1.0_f32).contains(&self.pq_exposure_shift) {
            return Err(HdrError::GamutConversionError(format!(
                "pq_exposure_shift out of range: {}",
                self.pq_exposure_shift
            )));
        }
        if self.contrast_factor < 0.0 {
            return Err(HdrError::ToneMappingError(format!(
                "contrast_factor must be non-negative: {}",
                self.contrast_factor
            )));
        }
        if self.saturation < 0.0 {
            return Err(HdrError::GamutConversionError(format!(
                "saturation must be non-negative: {}",
                self.saturation
            )));
        }
        if !(0.0_f32..=1.0_f32).contains(&self.highlight_rolloff) {
            return Err(HdrError::GamutConversionError(format!(
                "highlight_rolloff out of range: {}",
                self.highlight_rolloff
            )));
        }
        Ok(())
    }

    /// Apply this grade to a normalised PQ RGB pixel.
    ///
    /// Input channels are in PQ signal space (0.0–1.0).  The function applies:
    /// 1. Exposure shift (additive)
    /// 2. Log-domain contrast S-curve
    /// 3. Saturation (Rec. 709 luma weighting)
    /// 4. Shadow and highlight tints
    ///
    /// The output is clamped to \[0.0, 1.0\].
    pub fn apply_pq(&self, rgb: [f32; 3]) -> [f32; 3] {
        // Step 1: exposure shift
        let shifted = [
            rgb[0] + self.pq_exposure_shift,
            rgb[1] + self.pq_exposure_shift,
            rgb[2] + self.pq_exposure_shift,
        ];

        // Step 2: log-domain contrast around pivot
        let contrasted = apply_log_contrast(
            shifted,
            self.contrast_log_centre,
            self.contrast_factor,
        );

        // Step 3: saturation
        let luma = rec709_luma(contrasted);
        let sat = self.saturation;
        let saturated = [
            luma + sat * (contrasted[0] - luma),
            luma + sat * (contrasted[1] - luma),
            luma + sat * (contrasted[2] - luma),
        ];

        // Step 4: tints
        let pq_luma = luma.clamp(0.0, 1.0);
        let shadow_weight = shadow_blend_weight(pq_luma);
        let highlight_weight = highlight_blend_weight(pq_luma, self.highlight_rolloff);

        let tinted = [
            saturated[0]
                + self.shadow_tint[0] * shadow_weight
                + self.highlight_tint[0] * highlight_weight,
            saturated[1]
                + self.shadow_tint[1] * shadow_weight
                + self.highlight_tint[1] * highlight_weight,
            saturated[2]
                + self.shadow_tint[2] * shadow_weight
                + self.highlight_tint[2] * highlight_weight,
        ];

        // Clamp output to valid PQ range
        [
            tinted[0].clamp(0.0, 1.0),
            tinted[1].clamp(0.0, 1.0),
            tinted[2].clamp(0.0, 1.0),
        ]
    }
}

// ─── Creative intent metadata ─────────────────────────────────────────────────

/// Serialisable creative intent metadata for a graded HDR deliverable.
///
/// This can be embedded in a container (e.g. as a user-data box) so downstream
/// tools can reproduce or adapt the intended look.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CreativeIntentMetadata {
    /// Name or label for the creative intent (e.g. "Cinema Theatrical").
    pub intent_label: String,

    /// The named preset that was applied (if any).
    pub preset: Option<CreativeLook>,

    /// The concrete grade parameters (may override or extend the preset).
    pub grade: HdrGradeParams,

    /// Reference display peak luminance (nits) for which this grade was authored.
    pub reference_peak_nits: f32,

    /// Intended viewing environment luminance in lux (e.g. 5.0 for dark cinema).
    pub intended_viewing_lux: f32,

    /// Free-form notes from the colourist.
    pub notes: String,
}

impl CreativeIntentMetadata {
    /// Create a new metadata block from a named preset.
    pub fn from_preset(
        preset: CreativeLook,
        reference_peak_nits: f32,
        intended_viewing_lux: f32,
    ) -> Self {
        Self {
            intent_label: format!("{preset:?}"),
            preset: Some(preset),
            grade: preset.params(),
            reference_peak_nits,
            intended_viewing_lux,
            notes: String::new(),
        }
    }

    /// Create a neutral (identity) metadata block.
    pub fn neutral(reference_peak_nits: f32) -> Self {
        Self {
            intent_label: "Neutral".to_string(),
            preset: Some(CreativeLook::Neutral),
            grade: HdrGradeParams::identity(),
            reference_peak_nits,
            intended_viewing_lux: 100.0,
            notes: String::new(),
        }
    }

    /// Validate the contained grade parameters.
    pub fn validate(&self) -> Result<()> {
        if self.reference_peak_nits <= 0.0 {
            return Err(HdrError::InvalidLuminance(self.reference_peak_nits));
        }
        if self.intended_viewing_lux < 0.0 {
            return Err(HdrError::InvalidLuminance(self.intended_viewing_lux));
        }
        self.grade.validate()
    }
}

// ─── Batch processing helper ──────────────────────────────────────────────────

/// Apply `params` to every pixel in a flat `[R, G, B, R, G, B, …]` PQ frame
/// buffer in-place.
///
/// `pixels` length must be a multiple of 3.  Returns an error if it is not.
pub fn apply_grade_to_frame(pixels: &mut [f32], params: &HdrGradeParams) -> Result<()> {
    if !pixels.len().is_multiple_of(3) {
        return Err(HdrError::GamutConversionError(format!(
            "pixel buffer length {} is not a multiple of 3",
            pixels.len()
        )));
    }
    for chunk in pixels.chunks_exact_mut(3) {
        let graded = params.apply_pq([chunk[0], chunk[1], chunk[2]]);
        chunk[0] = graded[0];
        chunk[1] = graded[1];
        chunk[2] = graded[2];
    }
    Ok(())
}

/// Apply `params` to every pixel in a flat `[R, G, B, R, G, B, …]` PQ frame
/// buffer, returning a new `Vec<f32>`.
///
/// `pixels` length must be a multiple of 3.
pub fn grade_frame(pixels: &[f32], params: &HdrGradeParams) -> Result<Vec<f32>> {
    if !pixels.len().is_multiple_of(3) {
        return Err(HdrError::GamutConversionError(format!(
            "pixel buffer length {} is not a multiple of 3",
            pixels.len()
        )));
    }
    let mut out = Vec::with_capacity(pixels.len());
    for chunk in pixels.chunks_exact(3) {
        let graded = params.apply_pq([chunk[0], chunk[1], chunk[2]]);
        out.extend_from_slice(&graded);
    }
    Ok(out)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Rec. 709 luma coefficient weighting.
#[inline]
fn rec709_luma(rgb: [f32; 3]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// Log-domain contrast: expand or compress around a pivot point.
///
/// Maps `x` → `pivot * (x / pivot) ^ factor` in linear PQ, emulating a
/// logarithmic pivot operation without actually going to log space.
#[inline]
fn apply_log_contrast(rgb: [f32; 3], pivot: f32, factor: f32) -> [f32; 3] {
    let safe_pivot = pivot.max(1e-6);
    let mut out = [0.0f32; 3];
    for (i, &ch) in rgb.iter().enumerate() {
        let norm = (ch / safe_pivot).max(0.0);
        out[i] = safe_pivot * norm.powf(factor);
    }
    out
}

/// Shadow blend weight: 1.0 at PQ = 0.0, smoothly falls to 0.0 at 0.30.
#[inline]
fn shadow_blend_weight(pq: f32) -> f32 {
    let t = (pq / 0.30_f32).clamp(0.0, 1.0);
    1.0 - smooth_step(t)
}

/// Highlight blend weight: 0.0 below `rolloff`, smoothly rises to 1.0 at 1.0.
#[inline]
fn highlight_blend_weight(pq: f32, rolloff: f32) -> f32 {
    let safe_rolloff = rolloff.clamp(0.0, 1.0 - 1e-6);
    let t = ((pq - safe_rolloff) / (1.0 - safe_rolloff)).clamp(0.0, 1.0);
    smooth_step(t)
}

/// Classic Hermite smooth-step t³(6t² − 15t + 10) (smoothstep variant).
#[inline]
fn smooth_step(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HdrGradeParams identity ───────────────────────────────────────────────

    #[test]
    fn test_identity_grade_is_passthrough() {
        let params = HdrGradeParams::identity();
        let pixel = [0.3f32, 0.5, 0.7];
        let out = params.apply_pq(pixel);
        for i in 0..3 {
            assert!(
                (out[i] - pixel[i]).abs() < 1e-4,
                "ch{i}: expected {}, got {}",
                pixel[i],
                out[i]
            );
        }
    }

    #[test]
    fn test_identity_validate_ok() {
        let params = HdrGradeParams::identity();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_zero_saturation_makes_grey() {
        let mut params = HdrGradeParams::identity();
        params.saturation = 0.0;
        let out = params.apply_pq([0.8, 0.1, 0.3]);
        let luma = 0.2126 * 0.8 + 0.7152 * 0.1 + 0.0722 * 0.3;
        for i in 0..3 {
            assert!(
                (out[i] - luma).abs() < 1e-4,
                "ch{i}: expected luma {luma}, got {}",
                out[i]
            );
        }
    }

    #[test]
    fn test_output_clamped_to_unit_range() {
        let params = HdrGradeParams {
            pq_exposure_shift: 1.0,
            contrast_log_centre: 0.18,
            contrast_factor: 3.0,
            saturation: 2.0,
            shadow_tint: [0.5, 0.5, 0.5],
            highlight_tint: [0.5, 0.5, 0.5],
            highlight_rolloff: 0.5,
        };
        let out = params.apply_pq([1.0, 1.0, 1.0]);
        for &ch in &out {
            assert!(ch <= 1.0 && ch >= 0.0, "out-of-range: {ch}");
        }
    }

    #[test]
    fn test_validate_rejects_negative_saturation() {
        let mut params = HdrGradeParams::identity();
        params.saturation = -0.1;
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_exposure_shift() {
        let mut params = HdrGradeParams::identity();
        params.pq_exposure_shift = 2.0;
        assert!(params.validate().is_err());
    }

    // ── CreativeLook ──────────────────────────────────────────────────────────

    #[test]
    fn test_all_presets_have_valid_params() {
        let looks = [
            CreativeLook::Neutral,
            CreativeLook::CinemaWarm,
            CreativeLook::CoolContrast,
            CreativeLook::VividVibrant,
            CreativeLook::FlatLog,
            CreativeLook::VintageFilm,
            CreativeLook::BwDramatic,
            CreativeLook::Documentary,
        ];
        for look in &looks {
            let params = look.params();
            assert!(
                params.validate().is_ok(),
                "preset {look:?} params failed validation"
            );
        }
    }

    #[test]
    fn test_bw_dramatic_desaturates() {
        let params = CreativeLook::BwDramatic.params();
        assert_eq!(params.saturation, 0.0);
    }

    // ── CreativeIntentMetadata ────────────────────────────────────────────────

    #[test]
    fn test_from_preset_stores_correct_preset() {
        let meta = CreativeIntentMetadata::from_preset(CreativeLook::CinemaWarm, 1000.0, 5.0);
        assert_eq!(meta.preset, Some(CreativeLook::CinemaWarm));
    }

    #[test]
    fn test_metadata_validate_rejects_zero_peak() {
        let mut meta = CreativeIntentMetadata::neutral(1000.0);
        meta.reference_peak_nits = 0.0;
        assert!(meta.validate().is_err());
    }

    // ── Batch processing ──────────────────────────────────────────────────────

    #[test]
    fn test_apply_grade_to_frame_identity() {
        let original = vec![0.5f32, 0.3, 0.7, 0.1, 0.9, 0.4];
        let mut pixels = original.clone();
        let params = HdrGradeParams::identity();
        apply_grade_to_frame(&mut pixels, &params).expect("frame grade");
        for (orig, graded) in original.iter().zip(pixels.iter()) {
            assert!((orig - graded).abs() < 1e-4, "got {graded} expected {orig}");
        }
    }

    #[test]
    fn test_apply_grade_to_frame_bad_length() {
        let mut pixels = vec![0.5f32, 0.3]; // not a multiple of 3
        let params = HdrGradeParams::identity();
        assert!(apply_grade_to_frame(&mut pixels, &params).is_err());
    }

    #[test]
    fn test_grade_frame_returns_same_length() {
        let pixels = vec![0.5f32, 0.3, 0.7, 0.2, 0.8, 0.4];
        let params = CreativeLook::VividVibrant.params();
        let out = grade_frame(&pixels, &params).expect("grade frame");
        assert_eq!(out.len(), pixels.len());
    }

    #[test]
    fn test_smooth_step_boundary_values() {
        assert!((smooth_step(0.0) - 0.0).abs() < 1e-6);
        assert!((smooth_step(1.0) - 1.0).abs() < 1e-6);
        // Mid-point should be 0.5 for symmetric smooth-step
        assert!((smooth_step(0.5) - 0.5).abs() < 1e-6);
    }
}
