//! HLG broadcast constraints validator.
//!
//! Validates HLG-encoded content against the signal range, system gamma, and
//! peak level requirements specified in:
//!
//! - **ARIB STD-B67:2015** — Hybrid Log-Gamma OETF definition and signal range
//! - **ITU-R BT.2100-2 (2018)** — HLG image parameter values and system gamma
//! - **ITU-R BT.2408-5 (2023)** — Operational practices for HLG content
//! - **EBU R 103 (2017)** — HLG peak level recommendations for broadcast
//!
//! # Validation model
//!
//! The validator checks each constraint independently and collects
//! [`BroadcastViolation`]s.  A [`BroadcastReport`] is *compliant* only when no
//! *error*-severity violations are present.  Warnings are advisory.
//!
//! # Example
//!
//! ```rust
//! use oximedia_hdr::hlg_broadcast_constraints::{
//!     HlgBroadcastValidator, HlgSignalParams, Severity,
//! };
//!
//! let params = HlgSignalParams {
//!     peak_signal: 0.75,
//!     black_signal: 0.0,
//!     nominal_peak_nits: 1000.0,
//!     system_gamma: 1.2,
//!     extended_range: false,
//! };
//! let report = HlgBroadcastValidator::validate(&params);
//! assert!(report.is_compliant());
//! ```

use crate::{HdrError, Result};

/// Severity of a broadcast constraint violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Advisory: compliant but not optimal.
    Warning,
    /// Non-compliant: must be corrected for broadcast delivery.
    Error,
}

/// A single constraint violation found during validation.
#[derive(Debug, Clone)]
pub struct BroadcastViolation {
    /// Severity of the violation.
    pub severity: Severity,
    /// Short identifier of the violated constraint (e.g. `"SIGNAL_RANGE"`).
    pub constraint: &'static str,
    /// Human-readable description.
    pub message: String,
    /// The measured value that caused the violation.
    pub measured: f64,
    /// The limit that was violated.
    pub limit: f64,
}

/// Validation result for a set of HLG signal parameters.
#[derive(Debug, Clone, Default)]
pub struct BroadcastReport {
    /// All violations found.  Empty ↔ fully compliant.
    pub violations: Vec<BroadcastViolation>,
}

impl BroadcastReport {
    /// Return `true` if no *error*-severity violations are present.
    pub fn is_compliant(&self) -> bool {
        self.violations
            .iter()
            .all(|v| v.severity < Severity::Error)
    }

    /// Return the number of error-severity violations.
    pub fn error_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count()
    }

    /// Return the number of warning-severity violations.
    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Warning)
            .count()
    }
}

/// Parameters describing the HLG signal to validate.
#[derive(Debug, Clone)]
pub struct HlgSignalParams {
    /// Measured peak HLG signal value in [0, 1].
    /// Per ARIB STD-B67, valid range is [0, 1].
    pub peak_signal: f64,

    /// Measured minimum (black) HLG signal value.
    /// Should be ≥ 0.0 for standard range, or ≥ −0.07 for extended range.
    pub black_signal: f64,

    /// Declared nominal peak luminance of the display in cd/m² (nits).
    /// BT.2408 recommends 1000 nits for broadcast HLG mastering.
    pub nominal_peak_nits: f64,

    /// System gamma applied to the OOTF.
    /// BT.2100 specifies γ = 1.2 as the nominal value; valid range [1.0, 1.4].
    pub system_gamma: f64,

    /// Whether extended signal range (values slightly below 0) is permitted.
    /// Some professional acquisition paths use a narrow extended range.
    pub extended_range: bool,
}

/// PLUGE (Picture Line-Up Generation Equipment) signal constraints.
///
/// Used to validate that the black level and super-black levels are within
/// the range expected by broadcast receivers.
#[derive(Debug, Clone)]
pub struct PlugeLevels {
    /// Minimum permitted black signal (0.0 for standard range).
    pub black_min: f64,
    /// Maximum permitted black signal (should not exceed nominal black).
    pub black_max: f64,
    /// Maximum "super-black" signal level (extended pedestal).
    pub super_black_max: f64,
}

impl Default for PlugeLevels {
    fn default() -> Self {
        Self {
            black_min: 0.0,
            black_max: 0.02,  // 2 % of full scale
            super_black_max: -0.07,
        }
    }
}

/// HLG broadcast constraints validator.
///
/// Checks signal parameters against ARIB STD-B67, BT.2100, BT.2408, and EBU R 103.
pub struct HlgBroadcastValidator;

impl HlgBroadcastValidator {
    /// Validate an [`HlgSignalParams`] set and return a [`BroadcastReport`].
    pub fn validate(params: &HlgSignalParams) -> BroadcastReport {
        let mut report = BroadcastReport::default();

        Self::check_peak_signal(params, &mut report);
        Self::check_black_signal(params, &mut report);
        Self::check_system_gamma(params, &mut report);
        Self::check_nominal_peak_nits(params, &mut report);
        Self::check_pluge_compatibility(params, &mut report);

        report
    }

    /// Validate and return `Err` if any error-severity violation is found.
    ///
    /// # Errors
    /// Returns [`HdrError::MetadataParseError`] with a summary of all errors.
    pub fn validate_strict(params: &HlgSignalParams) -> Result<BroadcastReport> {
        let report = Self::validate(params);
        if !report.is_compliant() {
            let msgs: Vec<String> = report
                .violations
                .iter()
                .filter(|v| v.severity == Severity::Error)
                .map(|v| format!("[{}] {}", v.constraint, v.message))
                .collect();
            return Err(HdrError::MetadataParseError(msgs.join("; ")));
        }
        Ok(report)
    }

    fn check_peak_signal(params: &HlgSignalParams, report: &mut BroadcastReport) {
        // ARIB STD-B67: signal range [0, 1]
        if params.peak_signal > 1.0 {
            report.violations.push(BroadcastViolation {
                severity: Severity::Error,
                constraint: "PEAK_SIGNAL_RANGE",
                message: format!(
                    "HLG peak signal {:.4} exceeds 1.0 (ARIB STD-B67 limit)",
                    params.peak_signal
                ),
                measured: params.peak_signal,
                limit: 1.0,
            });
        } else if params.peak_signal > 0.92 {
            // EBU R 103 recommends ≤ 0.92 for headroom
            report.violations.push(BroadcastViolation {
                severity: Severity::Warning,
                constraint: "PEAK_SIGNAL_HEADROOM",
                message: format!(
                    "HLG peak signal {:.4} exceeds EBU R 103 recommended headroom limit 0.92",
                    params.peak_signal
                ),
                measured: params.peak_signal,
                limit: 0.92,
            });
        }
    }

    fn check_black_signal(params: &HlgSignalParams, report: &mut BroadcastReport) {
        let min_allowed = if params.extended_range { -0.07 } else { 0.0 };
        if params.black_signal < min_allowed {
            report.violations.push(BroadcastViolation {
                severity: Severity::Error,
                constraint: "BLACK_SIGNAL_RANGE",
                message: format!(
                    "HLG black signal {:.4} is below minimum {} (extended_range={})",
                    params.black_signal, min_allowed, params.extended_range
                ),
                measured: params.black_signal,
                limit: min_allowed,
            });
        }
        if params.black_signal < 0.0 && !params.extended_range {
            // Should have been caught above; guard for logic clarity
            report.violations.push(BroadcastViolation {
                severity: Severity::Error,
                constraint: "BLACK_BELOW_ZERO_NOT_EXTENDED",
                message: format!(
                    "HLG black signal {:.4} is negative but extended_range is false",
                    params.black_signal
                ),
                measured: params.black_signal,
                limit: 0.0,
            });
        }
    }

    fn check_system_gamma(params: &HlgSignalParams, report: &mut BroadcastReport) {
        // BT.2100 nominal: 1.2; valid production range: [1.0, 1.4]
        const GAMMA_MIN: f64 = 1.0;
        const GAMMA_MAX: f64 = 1.4;
        const GAMMA_NOMINAL: f64 = 1.2;

        if params.system_gamma < GAMMA_MIN || params.system_gamma > GAMMA_MAX {
            report.violations.push(BroadcastViolation {
                severity: Severity::Error,
                constraint: "SYSTEM_GAMMA_RANGE",
                message: format!(
                    "system gamma {:.4} outside BT.2100 valid range [{}, {}]",
                    params.system_gamma, GAMMA_MIN, GAMMA_MAX
                ),
                measured: params.system_gamma,
                limit: if params.system_gamma < GAMMA_MIN {
                    GAMMA_MIN
                } else {
                    GAMMA_MAX
                },
            });
        } else if (params.system_gamma - GAMMA_NOMINAL).abs() > 0.05 {
            report.violations.push(BroadcastViolation {
                severity: Severity::Warning,
                constraint: "SYSTEM_GAMMA_NOMINAL",
                message: format!(
                    "system gamma {:.4} deviates from BT.2100 nominal {:.1} by more than 0.05",
                    params.system_gamma, GAMMA_NOMINAL
                ),
                measured: params.system_gamma,
                limit: GAMMA_NOMINAL,
            });
        }
    }

    fn check_nominal_peak_nits(params: &HlgSignalParams, report: &mut BroadcastReport) {
        // BT.2408: HLG broadcast mastering reference is 1000 nits.
        // Acceptable range for broadcast: [400, 4000] nits.
        const NITS_MIN_BROADCAST: f64 = 400.0;
        const NITS_MAX_BROADCAST: f64 = 4000.0;
        const NITS_RECOMMENDED: f64 = 1000.0;

        if params.nominal_peak_nits < 1.0 {
            report.violations.push(BroadcastViolation {
                severity: Severity::Error,
                constraint: "NOMINAL_NITS_ZERO",
                message: format!(
                    "nominal peak {:.1} nits is not a valid luminance",
                    params.nominal_peak_nits
                ),
                measured: params.nominal_peak_nits,
                limit: 1.0,
            });
        } else if params.nominal_peak_nits < NITS_MIN_BROADCAST
            || params.nominal_peak_nits > NITS_MAX_BROADCAST
        {
            report.violations.push(BroadcastViolation {
                severity: Severity::Warning,
                constraint: "NOMINAL_NITS_BROADCAST_RANGE",
                message: format!(
                    "nominal peak {:.1} nits is outside BT.2408 broadcast range [{}, {}]",
                    params.nominal_peak_nits, NITS_MIN_BROADCAST, NITS_MAX_BROADCAST
                ),
                measured: params.nominal_peak_nits,
                limit: NITS_RECOMMENDED,
            });
        }
    }

    fn check_pluge_compatibility(params: &HlgSignalParams, report: &mut BroadcastReport) {
        let pluge = PlugeLevels::default();
        // Black must not be above 2 % of full scale (cannot have lifted blacks)
        if params.black_signal > pluge.black_max {
            report.violations.push(BroadcastViolation {
                severity: Severity::Warning,
                constraint: "PLUGE_BLACK_PEDESTAL",
                message: format!(
                    "black signal {:.4} exceeds PLUGE maximum pedestal {:.4}",
                    params.black_signal, pluge.black_max
                ),
                measured: params.black_signal,
                limit: pluge.black_max,
            });
        }
    }
}

/// Compute the expected BT.2100 nominal system gamma for a given display peak.
///
/// Formula: γ = 1.2 × (1.111)^log₂(Lw / 1000)
/// where Lw is the display peak luminance in nits.
pub fn bt2100_nominal_gamma(display_peak_nits: f64) -> f64 {
    if display_peak_nits <= 0.0 {
        return 1.2;
    }
    1.2 * (1.111f64).powf((display_peak_nits / 1000.0).log2())
}

/// Clamp a system gamma value to the BT.2100 valid range [1.0, 1.4].
pub fn clamp_system_gamma(gamma: f64) -> f64 {
    gamma.clamp(1.0, 1.4)
}

/// Compute the HLG OOTF output luminance for a given scene luminance and
/// system gamma.
///
/// Per BT.2100: `Lout = α * Ys^(γ - 1) * Yin`
/// where `α = Lw / Ls^γ` (Lw = display peak, Ls = scene reference).
///
/// For normalised inputs (Lw = 1, Ls = 1) this simplifies to `Yin^γ`.
///
/// # Errors
/// Returns [`HdrError::InvalidLuminance`] if `scene_luminance < 0.0`.
pub fn hlg_ootf_luminance(scene_luminance: f64, system_gamma: f64) -> Result<f64> {
    if scene_luminance < 0.0 {
        return Err(HdrError::InvalidLuminance(scene_luminance as f32));
    }
    Ok(scene_luminance.powf(system_gamma))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_params() -> HlgSignalParams {
        HlgSignalParams {
            peak_signal: 0.75,
            black_signal: 0.0,
            nominal_peak_nits: 1000.0,
            system_gamma: 1.2,
            extended_range: false,
        }
    }

    #[test]
    fn test_nominal_params_compliant() {
        let report = HlgBroadcastValidator::validate(&nominal_params());
        assert!(report.is_compliant(), "{:?}", report.violations);
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn test_peak_signal_above_1_is_error() {
        let mut p = nominal_params();
        p.peak_signal = 1.05;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(!report.is_compliant());
        assert!(report.violations.iter().any(|v| v.constraint == "PEAK_SIGNAL_RANGE"));
    }

    #[test]
    fn test_peak_signal_headroom_warning() {
        let mut p = nominal_params();
        p.peak_signal = 0.95; // above 0.92 headroom limit
        let report = HlgBroadcastValidator::validate(&p);
        assert!(report.is_compliant()); // only a warning
        assert!(report.violations.iter().any(|v| v.constraint == "PEAK_SIGNAL_HEADROOM"));
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn test_black_signal_negative_no_extended_range_error() {
        let mut p = nominal_params();
        p.black_signal = -0.01;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_black_signal_extended_range_allowed() {
        let mut p = nominal_params();
        p.black_signal = -0.05;
        p.extended_range = true;
        let report = HlgBroadcastValidator::validate(&p);
        // -0.05 is within extended range [-0.07, 1.0]
        assert!(report.is_compliant(), "{:?}", report.violations);
    }

    #[test]
    fn test_system_gamma_out_of_range_error() {
        let mut p = nominal_params();
        p.system_gamma = 1.6;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(!report.is_compliant());
        assert!(report.violations.iter().any(|v| v.constraint == "SYSTEM_GAMMA_RANGE"));
    }

    #[test]
    fn test_system_gamma_below_min_error() {
        let mut p = nominal_params();
        p.system_gamma = 0.8;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_system_gamma_deviated_warning() {
        let mut p = nominal_params();
        p.system_gamma = 1.3; // within [1.0, 1.4] but >0.05 from 1.2
        let report = HlgBroadcastValidator::validate(&p);
        assert!(report.is_compliant());
        assert!(report.violations.iter().any(|v| v.constraint == "SYSTEM_GAMMA_NOMINAL"));
    }

    #[test]
    fn test_nominal_nits_below_broadcast_range_warning() {
        let mut p = nominal_params();
        p.nominal_peak_nits = 200.0;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(report.is_compliant());
        assert!(report.violations.iter().any(|v| v.constraint == "NOMINAL_NITS_BROADCAST_RANGE"));
    }

    #[test]
    fn test_validate_strict_returns_err_on_error() {
        let mut p = nominal_params();
        p.peak_signal = 1.1;
        let result = HlgBroadcastValidator::validate_strict(&p);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_strict_ok_on_compliant() {
        let result = HlgBroadcastValidator::validate_strict(&nominal_params());
        assert!(result.is_ok());
    }

    #[test]
    fn test_bt2100_nominal_gamma_1000_nits() {
        let g = bt2100_nominal_gamma(1000.0);
        assert!((g - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_bt2100_nominal_gamma_2000_nits() {
        // log2(2) = 1 → 1.2 × 1.111^1 ≈ 1.333
        let g = bt2100_nominal_gamma(2000.0);
        assert!((g - 1.2 * 1.111).abs() < 1e-9);
    }

    #[test]
    fn test_clamp_system_gamma_min() {
        assert!((clamp_system_gamma(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_system_gamma_max() {
        assert!((clamp_system_gamma(2.0) - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_system_gamma_in_range() {
        let g = clamp_system_gamma(1.2);
        assert!((g - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_hlg_ootf_luminance_zero() {
        let out = hlg_ootf_luminance(0.0, 1.2).unwrap();
        assert!((out - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hlg_ootf_luminance_one() {
        let out = hlg_ootf_luminance(1.0, 1.2).unwrap();
        assert!((out - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hlg_ootf_luminance_negative_error() {
        assert!(hlg_ootf_luminance(-0.5, 1.2).is_err());
    }

    #[test]
    fn test_black_signal_lifted_blacks_warning() {
        let mut p = nominal_params();
        p.black_signal = 0.05; // above 0.02 PLUGE limit
        let report = HlgBroadcastValidator::validate(&p);
        assert!(report.violations.iter().any(|v| v.constraint == "PLUGE_BLACK_PEDESTAL"));
    }

    #[test]
    fn test_nominal_nits_zero_is_error() {
        let mut p = nominal_params();
        p.nominal_peak_nits = 0.0;
        let report = HlgBroadcastValidator::validate(&p);
        assert!(!report.is_compliant());
        assert!(report.violations.iter().any(|v| v.constraint == "NOMINAL_NITS_ZERO"));
    }
}
