//! HLG reference display conditions per ITU-R BT.2390.
//!
//! Implements the HLG reference display OOTF (Optical-Optical Transfer
//! Function), display-referred HLG signal mapping to absolute luminance for
//! several reference display peak values (100 / 300 / 1000 nits), ambient
//! light adaptation (viewing environment correction), and nominal peak
//! luminance selection.
//!
//! # Background
//!
//! Unlike PQ (which is absolute), HLG is a *relative* signal: the same encoded
//! value represents different luminance levels on different displays.  The OOTF
//! maps scene-referred linear light (`E′` after the OETF inverse) to
//! display-referred light using a power-law whose exponent depends on the
//! display peak `Lw` and the ambient viewing level.
//!
//! BT.2390 Section 4.6 gives:
//!
//! ```text
//! Ys = α · E^(γ - 1)  (scene luminance → display luma)
//! where α = Lw / 12^γ  and  γ = 1.2 + 0.42 · log10(Lw / 1000)
//! ```
//!
//! # Quick start
//! ```rust,ignore
//! use oximedia_hdr::hlg_reference_display::{HlgReferenceDisplay, ReferenceDisplayPeak};
//!
//! let display = HlgReferenceDisplay::new(ReferenceDisplayPeak::Nits1000);
//! // scene_linear: normalised HLG scene-linear signal [0, 1].
//! let display_nits = display.scene_to_display(0.5)?;
//! ```

use crate::{HdrError, Result};

// ── Reference display peak presets ───────────────────────────────────────────

/// Nominal peak luminance presets for HLG reference displays.
///
/// The ITU-R BT.2390 OOTF exponent is a function of the display peak; these
/// presets correspond to the three most common reference monitors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReferenceDisplayPeak {
    /// 100 cd/m² — SDR-grade reference display.
    Nits100,
    /// 300 cd/m² — HDR-capable reference monitor.
    Nits300,
    /// 1000 cd/m² — High-end HDR reference display.
    Nits1000,
    /// Custom peak luminance in nits.  Must be in `(0, 10_000]`.
    Custom(f32),
}

impl ReferenceDisplayPeak {
    /// Return the peak luminance value in nits.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` for a `Custom` value ≤ 0 or NaN.
    pub fn peak_nits(self) -> Result<f32> {
        match self {
            Self::Nits100 => Ok(100.0),
            Self::Nits300 => Ok(300.0),
            Self::Nits1000 => Ok(1000.0),
            Self::Custom(v) => {
                if v <= 0.0 || !v.is_finite() {
                    Err(HdrError::InvalidLuminance(v))
                } else {
                    Ok(v)
                }
            }
        }
    }
}

// ── Ambient viewing environment ───────────────────────────────────────────────

/// Ambient luminance class for viewing-environment adaptation.
///
/// BT.2390 Table 3 gives representative surround luminance values used to
/// adjust the system gamma and hence the OOTF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AmbientViewingEnvironment {
    /// Dark surround (cinema / darkened room).  Effectively 0 lux ambient.
    Dark,
    /// Dim surround (typical home-viewing).  Approximately 5 cd/m² surround.
    Dim,
    /// Average surround (studio / office).  Approximately 20 cd/m² surround.
    Average,
    /// Bright surround (daylight viewing).  Approximately 200 cd/m² surround.
    Bright,
}

impl AmbientViewingEnvironment {
    /// Return the BT.2390 ambient correction additive term `Δγ` applied to the
    /// base system gamma.
    ///
    /// The correction shifts the gamma to account for the Stevens effect (the
    /// perceived contrast change with adaptation level).
    pub fn gamma_delta(self) -> f32 {
        match self {
            Self::Dark => 0.0,
            Self::Dim => -0.05,
            Self::Average => -0.10,
            Self::Bright => -0.20,
        }
    }
}

// ── HLG OOTF parameters ───────────────────────────────────────────────────────

/// Precomputed parameters for the HLG display-referred OOTF.
///
/// Constructed once per `(peak_nits, ambient)` pair; apply to individual pixel
/// values with [`apply`].
///
/// [`apply`]: HlgOotfParams::apply
#[derive(Debug, Clone)]
pub struct HlgOotfParams {
    /// BT.2390 system gamma `γ` (including ambient correction).
    pub gamma: f32,
    /// Scale factor `α = Lw / 12^γ`.
    pub alpha: f32,
    /// Display peak luminance in nits.
    pub peak_nits: f32,
}

impl HlgOotfParams {
    /// Compute OOTF parameters for the given display peak and ambient
    /// environment.
    ///
    /// Gamma formula (BT.2390 § 4.6):
    /// `γ = 1.2 + 0.42 · log10(Lw / 1000) + Δγ`
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `peak` resolves to a
    /// non-positive or non-finite value.
    pub fn new(peak: ReferenceDisplayPeak, ambient: AmbientViewingEnvironment) -> Result<Self> {
        let lw = peak.peak_nits()?;
        let gamma_base = 1.2_f32 + 0.42_f32 * (lw / 1000.0_f32).log10();
        let gamma = (gamma_base + ambient.gamma_delta()).max(0.01);
        // α = Lw / 12^γ
        let alpha = lw / (12.0_f32).powf(gamma);
        Ok(Self { gamma, alpha, peak_nits: lw })
    }

    /// Apply the OOTF to a scene-linear normalised luminance value `e` ∈ [0, 1].
    ///
    /// Returns display luminance in nits.
    ///
    /// The OOTF is: `Fd = α · E^γ`  where `E` is the scene-linear signal.
    /// For `E = 0` the result is 0 (no light).
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `e` is negative or NaN.
    pub fn apply(&self, e: f32) -> Result<f32> {
        if e.is_nan() || e < 0.0 {
            return Err(HdrError::InvalidLuminance(e));
        }
        if e == 0.0 {
            return Ok(0.0);
        }
        Ok(self.alpha * e.powf(self.gamma))
    }

    /// Apply the OOTF to a full RGB triplet, returning display-referred
    /// `(R_d, G_d, B_d)` in nits.
    ///
    /// BT.2390 uses the luma-dependent formulation:
    /// `Ys = 0.2627 R + 0.6780 G + 0.0593 B`
    /// `Fd(C) = C · (α · Ys^(γ-1))`
    ///
    /// This preserves hue while applying the OOTF.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if any component is negative or NaN.
    pub fn apply_rgb(&self, r: f32, g: f32, b: f32) -> Result<(f32, f32, f32)> {
        for v in [r, g, b] {
            if v.is_nan() || v < 0.0 {
                return Err(HdrError::InvalidLuminance(v));
            }
        }
        // BT.2100 luma coefficients for Rec.2020.
        const KR: f32 = 0.2627;
        const KG: f32 = 0.6780;
        const KB: f32 = 0.0593;
        let ys = KR * r + KG * g + KB * b;
        if ys == 0.0 {
            return Ok((0.0, 0.0, 0.0));
        }
        let scale = self.alpha * ys.powf(self.gamma - 1.0);
        Ok((r * scale, g * scale, b * scale))
    }
}

// ── HlgReferenceDisplay ───────────────────────────────────────────────────────

/// Complete HLG reference display model including OETF inverse, OOTF, and
/// optional ambient adaptation.
///
/// This combines the HLG OETF inverse (scene-linear signal recovery) with the
/// display-referred OOTF to give an end-to-end scene-code-to-display-nits
/// mapping.
#[derive(Debug, Clone)]
pub struct HlgReferenceDisplay {
    ootf: HlgOotfParams,
}

impl HlgReferenceDisplay {
    /// Create a reference display model for the given peak luminance with the
    /// default dim-surround ambient environment (BT.2390 reference viewing).
    ///
    /// # Errors
    /// Propagates errors from [`HlgOotfParams::new`].
    pub fn new(peak: ReferenceDisplayPeak) -> Result<Self> {
        let ootf = HlgOotfParams::new(peak, AmbientViewingEnvironment::Dim)?;
        Ok(Self { ootf })
    }

    /// Create a reference display model with an explicit ambient environment.
    ///
    /// # Errors
    /// Propagates errors from [`HlgOotfParams::new`].
    pub fn with_ambient(
        peak: ReferenceDisplayPeak,
        ambient: AmbientViewingEnvironment,
    ) -> Result<Self> {
        let ootf = HlgOotfParams::new(peak, ambient)?;
        Ok(Self { ootf })
    }

    /// Map an HLG-encoded signal value `hlg` ∈ [0, 1] to display luminance in
    /// nits via the full HLG OETF inverse followed by the OOTF.
    ///
    /// HLG OETF inverse (BT.2100 Table 5):
    /// - For `hlg ∈ [0, 0.5]`:  `E = (hlg / a)^2 / 3`   where `a = 0.17883277`
    /// - For `hlg ∈ (0.5, 1]`:  `E = (exp((hlg - c) / b) + d) / 12`
    ///
    /// where `a = 0.17883277`, `b = 0.28466892`, `c = 0.55991073`, `d = 0.02372241`.
    ///
    /// # Errors
    /// Returns `HdrError::InvalidLuminance` if `hlg` is outside [0, 1] or NaN.
    pub fn scene_to_display(&self, hlg: f32) -> Result<f32> {
        let e = hlg_oetf_inverse(hlg)?;
        self.ootf.apply(e)
    }

    /// Map an HLG-encoded RGB triplet to display-referred nits.
    ///
    /// # Errors
    /// Returns an error if any component is outside [0, 1] or NaN.
    pub fn scene_to_display_rgb(&self, r: f32, g: f32, b: f32) -> Result<(f32, f32, f32)> {
        let er = hlg_oetf_inverse(r)?;
        let eg = hlg_oetf_inverse(g)?;
        let eb = hlg_oetf_inverse(b)?;
        self.ootf.apply_rgb(er, eg, eb)
    }

    /// Return a reference to the underlying OOTF parameters.
    pub fn ootf_params(&self) -> &HlgOotfParams {
        &self.ootf
    }

    /// Select the recommended reference display peak for a given content
    /// MaxCLL value (nits) per BT.2390 Table 2 guidance.
    ///
    /// | MaxCLL         | Recommended display peak |
    /// |----------------|--------------------------|
    /// | ≤ 150 nits     | 100 nits                 |
    /// | 150 – 400 nits | 300 nits                 |
    /// | > 400 nits     | 1000 nits                |
    pub fn recommend_peak(max_cll_nits: f32) -> ReferenceDisplayPeak {
        if max_cll_nits <= 150.0 {
            ReferenceDisplayPeak::Nits100
        } else if max_cll_nits <= 400.0 {
            ReferenceDisplayPeak::Nits300
        } else {
            ReferenceDisplayPeak::Nits1000
        }
    }
}

// ── HLG OETF inverse (EOTF) ──────────────────────────────────────────────────

/// HLG OETF inverse: convert an HLG-encoded signal `hlg` ∈ [0, 1] to
/// scene-linear light `E` ∈ [0, 1].
///
/// Constants per ITU-R BT.2100 Table 5.
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `hlg` is outside [0, 1] or NaN.
pub fn hlg_oetf_inverse(hlg: f32) -> Result<f32> {
    if hlg.is_nan() || !(0.0..=1.0).contains(&hlg) {
        return Err(HdrError::InvalidLuminance(hlg));
    }
    const A: f32 = 0.17883277;
    const B: f32 = 0.28466892;
    const C: f32 = 0.559_910_7;
    const D: f32 = 0.02372241;

    let e = if hlg <= 0.5 {
        (hlg / A).powi(2) / 3.0
    } else {
        (((hlg - C) / B).exp() + D) / 12.0
    };
    Ok(e.max(0.0))
}

/// Compute the system gamma `γ` for an HLG display with peak luminance `lw`
/// (nits) and the given ambient environment.
///
/// `γ = 1.2 + 0.42 · log10(Lw / 1000) + Δγ`
///
/// Returns a value clamped to a minimum of 0.01.
///
/// # Errors
/// Returns `HdrError::InvalidLuminance` if `lw` is not finite or ≤ 0.
pub fn hlg_system_gamma_bt2390(lw: f32, ambient: AmbientViewingEnvironment) -> Result<f32> {
    if !lw.is_finite() || lw <= 0.0 {
        return Err(HdrError::InvalidLuminance(lw));
    }
    let gamma = 1.2_f32 + 0.42_f32 * (lw / 1000.0).log10() + ambient.gamma_delta();
    Ok(gamma.max(0.01))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons.
    const EPS: f32 = 1e-4;

    #[test]
    fn test_reference_display_peak_nits() {
        assert_eq!(ReferenceDisplayPeak::Nits100.peak_nits().unwrap(), 100.0);
        assert_eq!(ReferenceDisplayPeak::Nits300.peak_nits().unwrap(), 300.0);
        assert_eq!(ReferenceDisplayPeak::Nits1000.peak_nits().unwrap(), 1000.0);
    }

    #[test]
    fn test_custom_peak_invalid() {
        assert!(ReferenceDisplayPeak::Custom(-1.0).peak_nits().is_err());
        assert!(ReferenceDisplayPeak::Custom(0.0).peak_nits().is_err());
        assert!(ReferenceDisplayPeak::Custom(f32::NAN).peak_nits().is_err());
    }

    #[test]
    fn test_custom_peak_valid() {
        let v = ReferenceDisplayPeak::Custom(500.0).peak_nits().unwrap();
        assert!((v - 500.0).abs() < EPS);
    }

    #[test]
    fn test_ambient_gamma_delta() {
        assert_eq!(AmbientViewingEnvironment::Dark.gamma_delta(), 0.0);
        assert!(AmbientViewingEnvironment::Dim.gamma_delta() < 0.0);
        assert!(AmbientViewingEnvironment::Bright.gamma_delta() < AmbientViewingEnvironment::Dim.gamma_delta());
    }

    #[test]
    fn test_hlg_oetf_inverse_zero() {
        let e = hlg_oetf_inverse(0.0).unwrap();
        assert!(e.abs() < EPS, "e={e}");
    }

    #[test]
    fn test_hlg_oetf_inverse_one() {
        // At hlg=1.0 we should get close to the maximum scene-linear value.
        let e = hlg_oetf_inverse(1.0).unwrap();
        assert!(e > 0.0 && e <= 1.0, "e={e}");
    }

    #[test]
    fn test_hlg_oetf_inverse_out_of_range() {
        assert!(hlg_oetf_inverse(-0.01).is_err());
        assert!(hlg_oetf_inverse(1.01).is_err());
        assert!(hlg_oetf_inverse(f32::NAN).is_err());
    }

    #[test]
    fn test_ootf_params_1000nits_dark() {
        let params = HlgOotfParams::new(
            ReferenceDisplayPeak::Nits1000,
            AmbientViewingEnvironment::Dark,
        )
        .unwrap();
        // For Lw=1000, log10(1000/1000)=0, so γ_base = 1.2; dark Δγ=0 → γ=1.2.
        assert!((params.gamma - 1.2).abs() < EPS, "gamma={}", params.gamma);
        // α = 1000 / 12^1.2
        let expected_alpha = 1000.0_f32 / (12.0_f32).powf(1.2);
        assert!((params.alpha - expected_alpha).abs() < 0.1, "alpha={}", params.alpha);
    }

    #[test]
    fn test_ootf_apply_zero() {
        let params = HlgOotfParams::new(
            ReferenceDisplayPeak::Nits1000,
            AmbientViewingEnvironment::Dark,
        )
        .unwrap();
        assert_eq!(params.apply(0.0).unwrap(), 0.0);
    }

    #[test]
    fn test_ootf_apply_negative_error() {
        let params = HlgOotfParams::new(
            ReferenceDisplayPeak::Nits100,
            AmbientViewingEnvironment::Dim,
        )
        .unwrap();
        assert!(params.apply(-0.1).is_err());
    }

    #[test]
    fn test_reference_display_scene_to_display_midtone() {
        // For a 1000-nit display, hlg=0.5 should map to some positive nit value < 1000.
        let display = HlgReferenceDisplay::new(ReferenceDisplayPeak::Nits1000).unwrap();
        let nits = display.scene_to_display(0.5).unwrap();
        assert!(nits > 0.0 && nits < 1000.0, "nits={nits}");
    }

    #[test]
    fn test_recommend_peak() {
        assert_eq!(HlgReferenceDisplay::recommend_peak(100.0), ReferenceDisplayPeak::Nits100);
        assert_eq!(HlgReferenceDisplay::recommend_peak(200.0), ReferenceDisplayPeak::Nits300);
        assert_eq!(HlgReferenceDisplay::recommend_peak(600.0), ReferenceDisplayPeak::Nits1000);
    }

    #[test]
    fn test_system_gamma_increases_with_peak() {
        let g100 = hlg_system_gamma_bt2390(100.0, AmbientViewingEnvironment::Dark).unwrap();
        let g1000 = hlg_system_gamma_bt2390(1000.0, AmbientViewingEnvironment::Dark).unwrap();
        // Higher peak → higher (less negative log term) → higher gamma.
        assert!(g1000 > g100, "g100={g100} g1000={g1000}");
    }

    #[test]
    fn test_system_gamma_invalid_input() {
        assert!(hlg_system_gamma_bt2390(-100.0, AmbientViewingEnvironment::Dark).is_err());
        assert!(hlg_system_gamma_bt2390(0.0, AmbientViewingEnvironment::Dark).is_err());
        assert!(hlg_system_gamma_bt2390(f32::INFINITY, AmbientViewingEnvironment::Dark).is_err());
    }

    #[test]
    fn test_ootf_apply_rgb_preserves_ratio() {
        // A grey pixel (r=g=b) should output the same r:g:b ratio.
        let params = HlgOotfParams::new(
            ReferenceDisplayPeak::Nits1000,
            AmbientViewingEnvironment::Dark,
        )
        .unwrap();
        let (rd, gd, bd) = params.apply_rgb(0.5, 0.5, 0.5).unwrap();
        // All channels should be equal.
        assert!((rd - gd).abs() < EPS, "rd={rd} gd={gd}");
        assert!((gd - bd).abs() < EPS, "gd={gd} bd={bd}");
    }
}
