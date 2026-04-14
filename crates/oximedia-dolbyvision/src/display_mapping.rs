//! Dolby Vision display mapping — target display luminance to L2 trim parameter
//! calculation, display capability matching, and display profile selection.
//!
//! This module implements the forward display mapping path that maps a source
//! content master (defined by its L6 mastering metadata) to a specific target
//! display defined by its peak luminance and black level.  The computed trim
//! parameters correspond to the Dolby Vision Level-2 metadata fields.
//!
//! # Algorithm Overview
//!
//! Given source mastering display peak luminance `M` and a target display peak
//! luminance `T`, the trim slope `s`, offset `o`, and power `p` are derived
//! through a perceptually motivated mapping that respects the PQ transfer
//! function.  The implementation follows the principles described in the Dolby
//! Vision metadata specification (public documentation only).
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::display_mapping::{
//!     DisplayProfile, DisplayMappingParams, DisplayMapper,
//! };
//!
//! let master = DisplayProfile::custom(4000.0, 0.005);
//! let target = DisplayProfile::custom(1000.0, 0.005);
//!
//! let mapper = DisplayMapper::new(master);
//! let params = mapper.compute_trim(&target);
//!
//! // Slope should be < 1 since target peak is lower than master peak
//! assert!(params.trim_slope < 1.0);
//! ```

// ── Display capability profile ────────────────────────────────────────────────

/// Standard display profiles used in common Dolby Vision deployments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DisplayTier {
    /// 100-nit SDR display (reference monitor / consumer SDR TV).
    Sdr100,
    /// 600-nit consumer HDR display.
    Hdr600,
    /// 1000-nit professional reference HDR display.
    Hdr1000,
    /// 4000-nit professional mastering reference display.
    Hdr4000,
    /// Custom display with user-defined peak and black level.
    Custom,
}

/// Display luminance capability profile used as mapping source or target.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayProfile {
    /// Peak luminance in nits (cd/m²).
    pub peak_nits: f32,
    /// Minimum black luminance in nits.
    pub black_nits: f32,
    /// Display tier / classification.
    pub tier: DisplayTier,
}

impl DisplayProfile {
    /// Standard 100-nit SDR display.
    pub const SDR_100: Self = Self {
        peak_nits: 100.0,
        black_nits: 0.1,
        tier: DisplayTier::Sdr100,
    };

    /// 600-nit HDR consumer display.
    pub const HDR_600: Self = Self {
        peak_nits: 600.0,
        black_nits: 0.005,
        tier: DisplayTier::Hdr600,
    };

    /// 1000-nit HDR professional reference display.
    pub const HDR_1000: Self = Self {
        peak_nits: 1000.0,
        black_nits: 0.005,
        tier: DisplayTier::Hdr1000,
    };

    /// 4000-nit mastering reference display.
    pub const HDR_4000: Self = Self {
        peak_nits: 4000.0,
        black_nits: 0.001,
        tier: DisplayTier::Hdr4000,
    };

    /// Create a custom display profile.
    #[must_use]
    pub fn custom(peak_nits: f32, black_nits: f32) -> Self {
        Self {
            peak_nits,
            black_nits,
            tier: DisplayTier::Custom,
        }
    }

    /// Return true if this display supports HDR (peak ≥ 400 nits).
    #[must_use]
    pub fn is_hdr(&self) -> bool {
        self.peak_nits >= 400.0
    }

    /// Return the display contrast ratio (peak / black).
    ///
    /// Returns `f32::INFINITY` if `black_nits` is zero.
    #[must_use]
    pub fn contrast_ratio(&self) -> f32 {
        if self.black_nits <= 0.0 {
            return f32::INFINITY;
        }
        self.peak_nits / self.black_nits
    }

    /// Convert peak luminance to a normalised PQ code value in [0, 1].
    #[must_use]
    pub fn peak_pq(&self) -> f32 {
        nits_to_pq(self.peak_nits)
    }

    /// Convert minimum luminance to a normalised PQ code value in [0, 1].
    #[must_use]
    pub fn black_pq(&self) -> f32 {
        nits_to_pq(self.black_nits.max(0.0))
    }
}

// ── Computed L2 trim parameters ───────────────────────────────────────────────

/// Computed Level-2 trim parameters for a specific target display.
///
/// These values correspond directly to the `trim_slope`, `trim_offset`, and
/// `trim_power` fields of a Dolby Vision L2 metadata entry.  All values are
/// expressed as floating-point for internal computation; the caller should
/// quantise them to the integer ranges used in the RPU bitstream.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayMappingParams {
    /// Slope (gain) applied to the PQ signal: typical range [0.0, 2.0].
    pub trim_slope: f32,
    /// Offset (lift/pedestal) applied after slope: typical range [-1.0, 1.0].
    pub trim_offset: f32,
    /// Power (gamma) applied to the PQ signal: typical range [0.0, 2.0].
    pub trim_power: f32,
    /// Chroma weight applied to the colour vector: typical range [-1.0, 1.0].
    pub chroma_weight: f32,
    /// Saturation gain multiplier: typical range [-1.0, 1.0].
    pub saturation_gain: f32,
    /// PQ code value (0–4095) for the target display peak luminance.
    pub target_max_pq_code: u16,
}

impl DisplayMappingParams {
    /// Create an identity (no-op) trim parameter set for the given target peak.
    #[must_use]
    pub fn identity(target_max_pq_code: u16) -> Self {
        Self {
            trim_slope: 1.0,
            trim_offset: 0.0,
            trim_power: 1.0,
            chroma_weight: 0.0,
            saturation_gain: 0.0,
            target_max_pq_code,
        }
    }

    /// Apply this trim pass to a single normalised PQ value, returning the
    /// mapped PQ output.  The output is clamped to [0, 1].
    ///
    /// The evaluation order follows the Dolby Vision display management pipeline:
    ///   1. Scale by slope
    ///   2. Add offset
    ///   3. Apply power (only for positive values)
    #[must_use]
    pub fn apply(&self, pq_in: f32) -> f32 {
        let after_slope = pq_in * self.trim_slope;
        let after_offset = after_slope + self.trim_offset;
        let clamped = after_offset.clamp(0.0, 1.0);
        // Power operation; guard against NaN from negative base
        let after_power = if self.trim_power.abs() < 1e-6 {
            // power ≈ 0 → output is 1.0 for any non-zero input
            if clamped > 0.0 { 1.0 } else { 0.0 }
        } else {
            clamped.powf(self.trim_power)
        };
        after_power.clamp(0.0, 1.0)
    }
}

// ── Display mapper ────────────────────────────────────────────────────────────

/// Computes Dolby Vision Level-2 trim parameters to map content mastered on a
/// `source` display to a `target` display.
///
/// The mapper uses a linear gain/offset model in PQ space, calibrated so that:
/// - The source peak maps to the target peak.
/// - The source black level maps to the target black level.
/// - A perceptual gamma adjustment compensates for the differing dynamic ranges.
#[derive(Debug, Clone)]
pub struct DisplayMapper {
    /// The display on which the content was mastered.
    pub source: DisplayProfile,
}

impl DisplayMapper {
    /// Create a new mapper for content mastered on `source`.
    #[must_use]
    pub fn new(source: DisplayProfile) -> Self {
        Self { source }
    }

    /// Compute the L2 trim parameters to adapt to `target`.
    ///
    /// The calculation is performed in normalised PQ space [0, 1].
    #[must_use]
    pub fn compute_trim(&self, target: &DisplayProfile) -> DisplayMappingParams {
        let src_peak_pq = self.source.peak_pq();
        let src_black_pq = self.source.black_pq();
        let tgt_peak_pq = target.peak_pq();
        let tgt_black_pq = target.black_pq();

        // Guard against degenerate source range
        let src_range = (src_peak_pq - src_black_pq).max(1e-6);
        let tgt_range = (tgt_peak_pq - tgt_black_pq).max(1e-6);

        // Linear slope: ratio of target range to source range
        let trim_slope = (tgt_range / src_range).clamp(0.0, 2.0);

        // Offset: after applying slope, black level should map correctly
        let trim_offset = (tgt_black_pq - src_black_pq * trim_slope).clamp(-1.0, 1.0);

        // Perceptual power correction: higher ratio → brighter target → gamma < 1.
        // We use the log ratio of peak luminances as a proxy.
        let trim_power = compute_perceptual_power(self.source.peak_nits, target.peak_nits);

        // Saturation is reduced slightly for lower-range targets to preserve
        // colour fidelity when peak luminance is compressed.
        let sat_gain = compute_saturation_gain(trim_slope);

        let target_max_pq_code = pq_to_code(tgt_peak_pq);

        DisplayMappingParams {
            trim_slope,
            trim_offset,
            trim_power,
            chroma_weight: 0.0,
            saturation_gain: sat_gain,
            target_max_pq_code,
        }
    }

    /// Compute trim parameters for a list of target displays.
    #[must_use]
    pub fn compute_trims<'a>(
        &self,
        targets: impl IntoIterator<Item = &'a DisplayProfile>,
    ) -> Vec<DisplayMappingParams> {
        targets.into_iter().map(|t| self.compute_trim(t)).collect()
    }
}

// ── Display capability matcher ────────────────────────────────────────────────

/// Selects the best matching display profile from a list of candidates for a
/// given target display.
///
/// Matching is done by minimising the peak-luminance distance in nits.
#[derive(Debug, Default)]
pub struct DisplayCapabilityMatcher;

impl DisplayCapabilityMatcher {
    /// Find the display profile in `candidates` whose peak luminance is closest
    /// to `target_peak_nits`.
    ///
    /// Returns `None` if `candidates` is empty.
    #[must_use]
    pub fn find_closest<'a>(
        &self,
        candidates: &'a [DisplayProfile],
        target_peak_nits: f32,
    ) -> Option<&'a DisplayProfile> {
        candidates.iter().min_by(|a, b| {
            let da = (a.peak_nits - target_peak_nits).abs();
            let db = (b.peak_nits - target_peak_nits).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return only those candidates whose peak luminance falls within the
    /// closed interval `[min_nits, max_nits]`.
    #[must_use]
    pub fn filter_by_range<'a>(
        &self,
        candidates: &'a [DisplayProfile],
        min_nits: f32,
        max_nits: f32,
    ) -> Vec<&'a DisplayProfile> {
        candidates
            .iter()
            .filter(|d| d.peak_nits >= min_nits && d.peak_nits <= max_nits)
            .collect()
    }
}

// ── PQ transfer function utilities ───────────────────────────────────────────

/// PQ EOTF constant: system gamma.
const PQ_M1: f64 = 0.159_301_758_125;
/// PQ EOTF constant.
const PQ_M2: f64 = 78.843_75;
/// PQ EOTF constant c1.
const PQ_C1: f64 = 0.835_937_5;
/// PQ EOTF constant c2.
const PQ_C2: f64 = 18.851_563;
/// PQ EOTF constant c3.
const PQ_C3: f64 = 18.6875;
/// Reference white level (10 000 nits) per SMPTE ST 2084.
const PQ_L_REF: f64 = 10_000.0;

/// Convert absolute luminance in nits to a normalised PQ code value [0, 1].
///
/// Uses the ST 2084 EOTF inverse.
#[must_use]
pub fn nits_to_pq(nits: f32) -> f32 {
    if nits <= 0.0 {
        return 0.0;
    }
    let y = f64::from(nits) / PQ_L_REF;
    let ym1 = y.powf(PQ_M1);
    let num = PQ_C1 + PQ_C2 * ym1;
    let den = 1.0 + PQ_C3 * ym1;
    (num / den).powf(PQ_M2) as f32
}

/// Convert a normalised PQ code value [0, 1] to absolute luminance in nits.
#[must_use]
pub fn pq_to_nits(pq: f32) -> f32 {
    if pq <= 0.0 {
        return 0.0;
    }
    let e = f64::from(pq.clamp(0.0, 1.0));
    let ep = e.powf(1.0 / PQ_M2);
    let num = (ep - PQ_C1).max(0.0);
    let den = (PQ_C2 - PQ_C3 * ep).max(1e-10);
    ((num / den).powf(1.0 / PQ_M1) * PQ_L_REF) as f32
}

/// Convert a normalised PQ value to a 12-bit PQ code value (0–4095).
#[must_use]
pub fn pq_to_code(pq: f32) -> u16 {
    (pq.clamp(0.0, 1.0) * 4095.0).round() as u16
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute a perceptual power correction factor.
///
/// When mapping from a high-peak source to a lower-peak target, a gamma value
/// slightly above 1.0 compensates for the perceptual contrast compression.
/// When mapping upwards (rare), gamma is slightly below 1.0.
fn compute_perceptual_power(source_peak: f32, target_peak: f32) -> f32 {
    if source_peak <= 0.0 || target_peak <= 0.0 {
        return 1.0;
    }
    let ratio = f64::from(target_peak) / f64::from(source_peak);
    if ratio >= 1.0 {
        // Upward mapping: mild gamma expansion
        let gamma = 1.0 - 0.05 * (ratio - 1.0).min(1.0);
        (gamma as f32).clamp(0.5, 1.0)
    } else {
        // Downward mapping: perceptual gamma compensation
        let log_ratio = ratio.ln().abs();
        let gamma = 1.0 + 0.2 * log_ratio;
        (gamma as f32).clamp(1.0, 2.0)
    }
}

/// Compute a mild saturation gain correction.
///
/// When slope is significantly below 1 (content is compressed), reducing
/// saturation slightly avoids chromatic artifacts on over-saturated displays.
fn compute_saturation_gain(trim_slope: f32) -> f32 {
    if trim_slope >= 1.0 {
        return 0.0;
    }
    // Negative gain → slight desaturation for compressed content
    let reduction = (1.0 - trim_slope).clamp(0.0, 1.0) * -0.1;
    reduction.clamp(-1.0, 0.0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_params_apply_is_noop() {
        let params = DisplayMappingParams::identity(2081);
        for pq in [0.0f32, 0.1, 0.25, 0.5, 0.75, 1.0] {
            let out = params.apply(pq);
            assert!((out - pq).abs() < 1e-5, "pq={pq} out={out}");
        }
    }

    #[test]
    fn test_nits_to_pq_round_trip() {
        for &nits in &[0.005f32, 100.0, 600.0, 1000.0, 4000.0, 10000.0] {
            let pq = nits_to_pq(nits);
            let back = pq_to_nits(pq);
            let rel_err = (back - nits).abs() / nits;
            assert!(rel_err < 0.001, "nits={nits} pq={pq} back={back} err={rel_err}");
        }
    }

    #[test]
    fn test_display_profile_is_hdr() {
        assert!(!DisplayProfile::SDR_100.is_hdr());
        assert!(DisplayProfile::HDR_600.is_hdr());
        assert!(DisplayProfile::HDR_1000.is_hdr());
        assert!(DisplayProfile::HDR_4000.is_hdr());
    }

    #[test]
    fn test_downward_mapping_slope_less_than_one() {
        let src = DisplayProfile::HDR_4000;
        let tgt = DisplayProfile::HDR_1000;
        let mapper = DisplayMapper::new(src);
        let params = mapper.compute_trim(&tgt);
        assert!(params.trim_slope < 1.0, "slope={}", params.trim_slope);
    }

    #[test]
    fn test_identity_mapping_same_display() {
        let display = DisplayProfile::HDR_1000;
        let mapper = DisplayMapper::new(display);
        let params = mapper.compute_trim(&display);
        // Slope should be approximately 1.0 for same-display mapping
        assert!((params.trim_slope - 1.0).abs() < 1e-4, "slope={}", params.trim_slope);
    }

    #[test]
    fn test_target_max_pq_code_in_range() {
        let src = DisplayProfile::HDR_4000;
        let tgt = DisplayProfile::HDR_1000;
        let mapper = DisplayMapper::new(src);
        let params = mapper.compute_trim(&tgt);
        // Code should be in [0, 4095]
        assert!(params.target_max_pq_code <= 4095);
    }

    #[test]
    fn test_upward_mapping_slope_greater_than_one() {
        // Mapping from SDR to HDR: target is brighter
        let src = DisplayProfile::SDR_100;
        let tgt = DisplayProfile::HDR_1000;
        let mapper = DisplayMapper::new(src);
        let params = mapper.compute_trim(&tgt);
        assert!(params.trim_slope > 1.0, "slope={}", params.trim_slope);
    }

    #[test]
    fn test_capability_matcher_closest() {
        let candidates = vec![
            DisplayProfile::SDR_100,
            DisplayProfile::HDR_600,
            DisplayProfile::HDR_1000,
            DisplayProfile::HDR_4000,
        ];
        let matcher = DisplayCapabilityMatcher;
        let best = matcher.find_closest(&candidates, 800.0).expect("should find closest");
        assert_eq!(best.peak_nits, 600.0); // 600 is closer than 1000 to 800
    }

    #[test]
    fn test_capability_matcher_filter_by_range() {
        let candidates = vec![
            DisplayProfile::SDR_100,
            DisplayProfile::HDR_600,
            DisplayProfile::HDR_1000,
            DisplayProfile::HDR_4000,
        ];
        let matcher = DisplayCapabilityMatcher;
        let filtered = matcher.filter_by_range(&candidates, 500.0, 1500.0);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_compute_trims_multiple_targets() {
        let src = DisplayProfile::HDR_4000;
        let mapper = DisplayMapper::new(src);
        let targets = [DisplayProfile::HDR_600, DisplayProfile::HDR_1000];
        let trims = mapper.compute_trims(targets.iter());
        assert_eq!(trims.len(), 2);
        // 600-nit trim slope should be smaller than 1000-nit trim slope
        assert!(trims[0].trim_slope < trims[1].trim_slope, "{:?}", trims);
    }

    #[test]
    fn test_contrast_ratio() {
        let display = DisplayProfile::custom(1000.0, 0.01);
        let cr = display.contrast_ratio();
        assert!((cr - 100_000.0).abs() < 1.0, "cr={cr}");
    }
}
