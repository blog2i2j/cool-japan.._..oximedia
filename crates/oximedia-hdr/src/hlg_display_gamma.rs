//! HLG system gamma adjustment based on target display peak luminance.
//!
//! The ITU-R BT.2100-2 HLG system includes a display-adaptive system gamma
//! formula that adjusts the OOTF (Optical-Optical Transfer Function) exponent
//! based on the nominal peak luminance of the target display.  This module
//! provides a complete, configurable implementation of that formula along with
//! derived utilities:
//!
//! - [`hlg_system_gamma_for_display`] — reference BT.2100-2 formula
//! - [`HlgDisplayGammaConfig`] — per-display configuration
//! - [`HlgDisplayAdapter`] — frame-level OOTF application
//! - [`HlgGammaTable`] — precomputed look-up table for the OOTF power law
//!
//! # Formula (BT.2100-2 §3.5)
//!
//! ```text
//! γ = 1.2 + 0.42 × log₁₀(L_W / 1000)
//! ```
//!
//! where `L_W` is the nominal peak luminance of the target display in cd/m².
//! At `L_W = 1000 nits`, `γ = 1.2` (the BT.2100-2 reference value).
//!
//! # References
//! - ITU-R BT.2100-2 (2018), §3.5 "System Gamma"
//! - ITU-R BT.2390-10 (2023) §5.1.5 "HLG OOTF Adaptation"
//! - BBC R&D White Paper WHP 309 "HLG for HDR Television"

use crate::{HdrError, Result};

// ── Reference constants ───────────────────────────────────────────────────────

/// Reference display peak luminance (nits) used to anchor the BT.2100-2 formula.
pub const REFERENCE_PEAK_NITS: f32 = 1000.0;

/// BT.2100-2 reference system gamma at 1000 nits.
pub const REFERENCE_SYSTEM_GAMMA: f32 = 1.2;

/// BT.2100-2 slope coefficient for the log-luminance correction.
pub const GAMMA_SLOPE: f32 = 0.42;

/// Minimum allowed peak luminance for the display model (nits).
const MIN_PEAK_NITS: f32 = 100.0;

/// Maximum allowed peak luminance for the display model (nits).
const MAX_PEAK_NITS: f32 = 10_000.0;

/// HLG reference white luminance per BT.2100-2 (nits).
pub const HLG_REFERENCE_WHITE_NITS: f32 = 203.0;

// ── Core formula ──────────────────────────────────────────────────────────────

/// Compute the HLG system gamma for a given display peak luminance.
///
/// Implements the BT.2100-2 adaptive formula:
///
/// ```text
/// γ(L_W) = 1.2 + 0.42 × log₁₀(L_W / 1000)
/// ```
///
/// # Arguments
/// * `peak_nits` – Nominal peak luminance of the target display in cd/m² (nits).
///   Must be in `[100, 10_000]`.
///
/// # Errors
/// Returns [`HdrError::InvalidLuminance`] if `peak_nits` is outside the valid range.
///
/// # Examples
/// ```rust
/// use oximedia_hdr::hlg_display_gamma::hlg_system_gamma_for_display;
///
/// let gamma = hlg_system_gamma_for_display(1000.0).unwrap();
/// assert!((gamma - 1.2).abs() < 1e-5, "Reference gamma at 1000 nits must be 1.2");
///
/// let gamma_400 = hlg_system_gamma_for_display(400.0).unwrap();
/// assert!(gamma_400 < 1.2, "Gamma should be lower for dimmer displays");
/// ```
pub fn hlg_system_gamma_for_display(peak_nits: f32) -> Result<f32> {
    if !peak_nits.is_finite() || !(MIN_PEAK_NITS..=MAX_PEAK_NITS).contains(&peak_nits) {
        return Err(HdrError::InvalidLuminance(peak_nits));
    }
    let log_ratio = (peak_nits / REFERENCE_PEAK_NITS).log10();
    Ok(REFERENCE_SYSTEM_GAMMA + GAMMA_SLOPE * log_ratio)
}

/// Unchecked variant of [`hlg_system_gamma_for_display`] that clamps the input.
///
/// This is intended for inner loops where the input has already been validated.
#[inline]
pub fn hlg_system_gamma_clamped(peak_nits: f32) -> f32 {
    let clamped = peak_nits.clamp(MIN_PEAK_NITS, MAX_PEAK_NITS).max(f32::EPSILON);
    let log_ratio = (clamped / REFERENCE_PEAK_NITS).log10();
    REFERENCE_SYSTEM_GAMMA + GAMMA_SLOPE * log_ratio
}

// ── HlgDisplayGammaConfig ─────────────────────────────────────────────────────

/// Configuration for an HLG display adaptation pipeline.
#[derive(Debug, Clone)]
pub struct HlgDisplayGammaConfig {
    /// Nominal peak luminance of the target display in nits.
    pub peak_nits: f32,
    /// Reference white luminance in nits (default 203 per BT.2100-2).
    pub reference_white_nits: f32,
    /// Black-level lift / minimum display luminance in nits (default 0.005).
    ///
    /// Corresponds to L_B in BT.2100 legalisation.  Must be less than `peak_nits`.
    pub black_level_nits: f32,
    /// Whether to normalise the OOTF output to `[0, 1]` relative to `peak_nits`.
    /// If `false`, the output is in absolute nits.
    pub normalise_output: bool,
}

impl Default for HlgDisplayGammaConfig {
    /// BT.2100-2 reference system: 1000-nit display, 203-nit reference white.
    fn default() -> Self {
        Self {
            peak_nits: 1000.0,
            reference_white_nits: HLG_REFERENCE_WHITE_NITS,
            black_level_nits: 0.005,
            normalise_output: true,
        }
    }
}

impl HlgDisplayGammaConfig {
    /// Create a configuration for a specific display peak luminance.
    pub fn for_display(peak_nits: f32) -> Result<Self> {
        if !peak_nits.is_finite() || !(MIN_PEAK_NITS..=MAX_PEAK_NITS).contains(&peak_nits) {
            return Err(HdrError::InvalidLuminance(peak_nits));
        }
        Ok(Self {
            peak_nits,
            ..Default::default()
        })
    }

    /// Compute the adapted system gamma for this display.
    pub fn system_gamma(&self) -> Result<f32> {
        hlg_system_gamma_for_display(self.peak_nits)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if !(MIN_PEAK_NITS..=MAX_PEAK_NITS).contains(&self.peak_nits) {
            return Err(HdrError::InvalidLuminance(self.peak_nits));
        }
        if self.black_level_nits >= self.peak_nits {
            return Err(HdrError::ToneMappingError(format!(
                "black_level_nits ({}) must be less than peak_nits ({})",
                self.black_level_nits, self.peak_nits
            )));
        }
        if self.reference_white_nits <= 0.0 || self.reference_white_nits > self.peak_nits {
            return Err(HdrError::ToneMappingError(format!(
                "reference_white_nits ({}) must be in (0, peak_nits={}]",
                self.reference_white_nits, self.peak_nits
            )));
        }
        Ok(())
    }
}

// ── HlgDisplayAdapter ─────────────────────────────────────────────────────────

/// Applies the HLG OOTF with adaptive system gamma to a frame.
///
/// The OOTF maps scene-linear light (HLG OETF output decoded back to scene) to
/// display-linear light.  The key parameter is the system gamma `γ`, which is
/// a function of the target display peak luminance.
///
/// Per BT.2100-2 the OOTF is:
///
/// ```text
/// F_D = α × E_s^(γ-1) × [E_R, E_G, E_B]
/// ```
///
/// where `E_s = 0.2627·E_R + 0.6780·E_G + 0.0593·E_B` (Rec.2020 luminance),
/// and `α = L_W` (the display peak nit value).
#[derive(Debug, Clone)]
pub struct HlgDisplayAdapter {
    config: HlgDisplayGammaConfig,
    /// Pre-computed system gamma for the configured display.
    system_gamma: f32,
}

impl HlgDisplayAdapter {
    /// Create an adapter from a validated [`HlgDisplayGammaConfig`].
    pub fn new(config: HlgDisplayGammaConfig) -> Result<Self> {
        config.validate()?;
        let system_gamma = config.system_gamma()?;
        Ok(Self {
            config,
            system_gamma,
        })
    }

    /// Create an adapter targeting a specific display peak luminance.
    pub fn for_display_nits(peak_nits: f32) -> Result<Self> {
        let config = HlgDisplayGammaConfig::for_display(peak_nits)?;
        Self::new(config)
    }

    /// Return the computed system gamma value.
    pub fn system_gamma(&self) -> f32 {
        self.system_gamma
    }

    /// Apply the HLG OOTF to a single scene-linear RGB triplet.
    ///
    /// `(r, g, b)` must be normalised scene-linear light in `[0, 1]`
    /// (i.e. the signal after HLG EOTF decoding, before display rendering).
    ///
    /// Returns display-linear RGB in nits (absolute) if `normalise_output = false`,
    /// or in `[0, 1]` relative to `peak_nits` if `normalise_output = true`.
    pub fn apply_ootf(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let r = r.max(0.0);
        let g = g.max(0.0);
        let b = b.max(0.0);

        // Rec.2020 scene luminance.
        let e_s = 0.2627 * r + 0.6780 * g + 0.0593 * b;

        let alpha = self.config.peak_nits;
        // OOTF exponent = γ - 1 (applied to luminance only; channels are scaled
        // uniformly to preserve chromaticity).
        let gamma_minus_one = self.system_gamma - 1.0;

        // Avoid 0^negative: protect the power computation.
        let ootf_factor = if e_s < f32::EPSILON {
            0.0
        } else {
            alpha * e_s.powf(gamma_minus_one)
        };

        let r_out = ootf_factor * r;
        let g_out = ootf_factor * g;
        let b_out = ootf_factor * b;

        if self.config.normalise_output {
            let inv_peak = 1.0 / self.config.peak_nits;
            (
                (r_out * inv_peak).clamp(0.0, 1.0),
                (g_out * inv_peak).clamp(0.0, 1.0),
                (b_out * inv_peak).clamp(0.0, 1.0),
            )
        } else {
            (r_out, g_out, b_out)
        }
    }

    /// Apply the OOTF to a whole frame of interleaved scene-linear RGB pixels.
    ///
    /// `pixels` must be a flat slice of `3 × width × height` `f32` values.
    pub fn apply_ootf_frame(&self, pixels: &mut [f32]) -> Result<()> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(
                "pixel buffer length must be a multiple of 3".into(),
            ));
        }
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.apply_ootf(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        Ok(())
    }

    /// Return a reference to the underlying configuration.
    pub fn config(&self) -> &HlgDisplayGammaConfig {
        &self.config
    }
}

// ── HlgGammaTable ─────────────────────────────────────────────────────────────

/// Precomputed OOTF power-law look-up table for fast HLG display adaptation.
///
/// Stores `N` entries of `e_s^(γ-1)` pre-evaluated at uniformly-spaced input
/// luminance values in `[0, 1]`.  Intermediate values are obtained via linear
/// interpolation between adjacent table entries.
///
/// This provides ~10× speedup over per-pixel `powf` at the cost of small
/// interpolation errors (< 0.1% at N = 1024).
#[derive(Debug, Clone)]
pub struct HlgGammaTable {
    /// Look-up table entries: `e_s^(γ-1)` at evenly-spaced `e_s` values.
    table: Vec<f32>,
    /// Number of table entries (resolution).
    n: usize,
    /// Peak luminance scaling factor (= `alpha = L_W`).
    alpha: f32,
    /// `γ - 1` exponent stored for reference.
    gamma_minus_one: f32,
    /// Whether the output is normalised to `[0, 1]`.
    normalise_output: bool,
    /// Reciprocal of peak nits, precomputed.
    inv_peak: f32,
}

impl HlgGammaTable {
    /// Build a look-up table for the given display adapter.
    ///
    /// `n` is the number of table entries; 1024 is sufficient for most uses.
    pub fn build(adapter: &HlgDisplayAdapter, n: usize) -> Result<Self> {
        if n < 2 {
            return Err(HdrError::ToneMappingError(
                "HlgGammaTable requires at least 2 entries".into(),
            ));
        }
        let gamma_minus_one = adapter.system_gamma - 1.0;
        let alpha = adapter.config.peak_nits;
        let inv_peak = 1.0 / alpha;

        let table: Vec<f32> = (0..n)
            .map(|i| {
                let e_s = i as f32 / (n - 1) as f32;
                if e_s < f32::EPSILON {
                    0.0
                } else {
                    alpha * e_s.powf(gamma_minus_one)
                }
            })
            .collect();

        Ok(Self {
            table,
            n,
            alpha,
            gamma_minus_one,
            normalise_output: adapter.config.normalise_output,
            inv_peak,
        })
    }

    /// Look up the OOTF scale factor for a given scene luminance `e_s` in `[0, 1]`.
    #[inline]
    fn lookup(&self, e_s: f32) -> f32 {
        if e_s <= 0.0 {
            return 0.0;
        }
        let idx_f = (e_s * (self.n - 1) as f32).clamp(0.0, (self.n - 1) as f32);
        let lo = idx_f.floor() as usize;
        let hi = (lo + 1).min(self.n - 1);
        let frac = idx_f - lo as f32;
        self.table[lo] * (1.0 - frac) + self.table[hi] * frac
    }

    /// Apply the precomputed OOTF to a single RGB pixel.
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let r = r.max(0.0);
        let g = g.max(0.0);
        let b = b.max(0.0);
        let e_s = 0.2627 * r + 0.6780 * g + 0.0593 * b;
        let scale = self.lookup(e_s);
        let r_out = scale * r;
        let g_out = scale * g;
        let b_out = scale * b;
        if self.normalise_output {
            (
                (r_out * self.inv_peak).clamp(0.0, 1.0),
                (g_out * self.inv_peak).clamp(0.0, 1.0),
                (b_out * self.inv_peak).clamp(0.0, 1.0),
            )
        } else {
            (r_out, g_out, b_out)
        }
    }

    /// Apply the precomputed OOTF to a flat interleaved RGB pixel buffer.
    pub fn apply_frame(&self, pixels: &mut [f32]) -> Result<()> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(
                "pixel buffer length must be a multiple of 3".into(),
            ));
        }
        for chunk in pixels.chunks_exact_mut(3) {
            let (r, g, b) = self.apply(chunk[0], chunk[1], chunk[2]);
            chunk[0] = r;
            chunk[1] = g;
            chunk[2] = b;
        }
        Ok(())
    }

    /// Number of table entries.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The `γ - 1` exponent used to build this table.
    pub fn gamma_minus_one(&self) -> f32 {
        self.gamma_minus_one
    }

    /// The peak luminance scaling factor `α`.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

// ── Display-gamma comparison utilities ────────────────────────────────────────

/// Compare the system gamma for two display targets and return the difference.
///
/// Useful for deciding whether a re-encode or OOTF recalculation is needed
/// when the viewing environment changes.
pub fn gamma_delta(peak_a: f32, peak_b: f32) -> Result<f32> {
    let ga = hlg_system_gamma_for_display(peak_a)?;
    let gb = hlg_system_gamma_for_display(peak_b)?;
    Ok((ga - gb).abs())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── hlg_system_gamma_for_display ──────────────────────────────────────────

    #[test]
    fn reference_gamma_at_1000_nits() {
        let gamma = hlg_system_gamma_for_display(1000.0).expect("1000 nits should be valid");
        assert!(
            (gamma - 1.2).abs() < 1e-5,
            "Expected 1.2 at 1000 nits, got {gamma}"
        );
    }

    #[test]
    fn gamma_increases_with_peak_luminance() {
        let g400 = hlg_system_gamma_for_display(400.0).expect("valid");
        let g1000 = hlg_system_gamma_for_display(1000.0).expect("valid");
        let g4000 = hlg_system_gamma_for_display(4000.0).expect("valid");
        assert!(g400 < g1000, "gamma(400) should be < gamma(1000)");
        assert!(g1000 < g4000, "gamma(1000) should be < gamma(4000)");
    }

    #[test]
    fn gamma_rejects_out_of_range() {
        assert!(hlg_system_gamma_for_display(50.0).is_err(), "50 nits too low");
        assert!(
            hlg_system_gamma_for_display(15_000.0).is_err(),
            "15000 nits too high"
        );
        assert!(
            hlg_system_gamma_for_display(f32::NAN).is_err(),
            "NaN should error"
        );
    }

    #[test]
    fn gamma_clamped_matches_formula_at_boundary() {
        let clamped = hlg_system_gamma_clamped(1000.0);
        let exact = hlg_system_gamma_for_display(1000.0).expect("valid");
        assert!((clamped - exact).abs() < 1e-6);
    }

    // ── HlgDisplayGammaConfig ─────────────────────────────────────────────────

    #[test]
    fn config_validation_ok_for_valid_params() {
        let cfg = HlgDisplayGammaConfig::for_display(2000.0).expect("valid");
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn config_validation_rejects_black_above_peak() {
        let cfg = HlgDisplayGammaConfig {
            peak_nits: 500.0,
            black_level_nits: 600.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_system_gamma_matches_formula() {
        let cfg = HlgDisplayGammaConfig::for_display(600.0).expect("valid");
        let gamma_cfg = cfg.system_gamma().expect("valid");
        let gamma_formula = hlg_system_gamma_for_display(600.0).expect("valid");
        assert!((gamma_cfg - gamma_formula).abs() < 1e-6);
    }

    // ── HlgDisplayAdapter ─────────────────────────────────────────────────────

    #[test]
    fn adapter_ootf_achromatic_normalised() {
        // For an achromatic (r==g==b) pixel, the OOTF is well-defined and
        // the output should remain achromatic.
        let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("valid");
        let (r, g, b) = adapter.apply_ootf(0.5, 0.5, 0.5);
        let diff_rg = (r - g).abs();
        let diff_rb = (r - b).abs();
        assert!(diff_rg < 1e-5, "r and g differ by {diff_rg}");
        assert!(diff_rb < 1e-5, "r and b differ by {diff_rb}");
    }

    #[test]
    fn adapter_ootf_zero_input_gives_zero() {
        let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("valid");
        let (r, g, b) = adapter.apply_ootf(0.0, 0.0, 0.0);
        assert_eq!(r, 0.0);
        assert_eq!(g, 0.0);
        assert_eq!(b, 0.0);
    }

    #[test]
    fn adapter_ootf_output_bounded_when_normalised() {
        let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("valid");
        for &v in &[0.1_f32, 0.5, 0.9, 1.0] {
            let (r, g, b) = adapter.apply_ootf(v, v * 0.5, v * 0.2);
            assert!(r <= 1.0 && g <= 1.0 && b <= 1.0, "Output out of range at v={v}");
            assert!(r >= 0.0 && g >= 0.0 && b >= 0.0, "Negative output at v={v}");
        }
    }

    #[test]
    fn adapter_frame_applies_per_pixel() {
        let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("valid");
        // Two identical pixels — output should be identical.
        let mut pixels = vec![0.5_f32, 0.4, 0.3, 0.5, 0.4, 0.3];
        adapter.apply_ootf_frame(&mut pixels).expect("frame ok");
        let diff = (pixels[0] - pixels[3]).abs()
            + (pixels[1] - pixels[4]).abs()
            + (pixels[2] - pixels[5]).abs();
        assert!(diff < 1e-6, "Identical input pixels should produce identical output");
    }

    // ── HlgGammaTable ────────────────────────────────────────────────────────

    #[test]
    fn gamma_table_lookup_matches_exact_at_nodes() {
        let adapter = HlgDisplayAdapter::for_display_nits(1000.0).expect("valid");
        let table = HlgGammaTable::build(&adapter, 512).expect("build ok");

        // At the exact grid points the LUT value should match the direct power computation.
        for i in [0_usize, 128, 255, 511] {
            let e_s = i as f32 / 511.0;
            let (r, g, b) = adapter.apply_ootf(e_s, e_s, e_s);
            let (rt, gt, bt) = table.apply(e_s, e_s, e_s);
            let err = (r - rt).abs() + (g - gt).abs() + (b - bt).abs();
            assert!(err < 0.002, "LUT error at node {i}: {err}");
        }
    }

    #[test]
    fn gamma_table_invalid_size_errors() {
        let adapter = HlgDisplayAdapter::for_display_nits(500.0).expect("valid");
        assert!(HlgGammaTable::build(&adapter, 1).is_err());
    }

    #[test]
    fn gamma_delta_is_zero_for_same_display() {
        let delta = gamma_delta(1000.0, 1000.0).expect("valid");
        assert!(delta.abs() < 1e-6, "Same display should have zero gamma delta");
    }

    #[test]
    fn gamma_delta_positive_for_different_displays() {
        let delta = gamma_delta(400.0, 4000.0).expect("valid");
        assert!(delta > 0.0, "Different displays must have non-zero gamma delta");
    }
}
