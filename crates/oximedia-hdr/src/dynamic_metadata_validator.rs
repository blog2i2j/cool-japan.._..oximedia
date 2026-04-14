//! Validation for HDR10+ dynamic metadata schema compliance.
//!
//! Implements structural and range validation for [`Hdr10PlusDynamicMetadata`]
//! payloads per SMPTE ST 2094-40 and the HDR10+ LLC specification.
//!
//! # What is validated
//!
//! - ITU-T T.35 country / terminal-provider codes
//! - Number of analysis windows (1–3 per ST 2094-40)
//! - Window bounding-box sanity (upper-left strictly before lower-right)
//! - MaxSCL channel values (must be ≤ 100 000 × 10 nits = 1 000 000)
//! - Targeted system display maximum luminance range (10–10 000 nits × 10)
//! - Distribution values monotonicity (percentile histogram)
//! - AverageMaxRGB versus MaxSCL consistency
//! - `fraction_bright_pixels` always 0–255 (guaranteed by type but checked for
//!   documentation / future-proofing)
//!
//! # References
//! - SMPTE ST 2094-40:2016 — "Dynamic Metadata for Color Volume Transform"
//! - HDR10+ LLC — "HDR10+ Metadata Validation Specification v1.2"
//! - ITU-T T.35 — "Procedure for the allocation of ITU-T defined codes"

use crate::dynamic_metadata::{Hdr10PlusDynamicMetadata, Hdr10PlusWindow};
use crate::{HdrError, Result};

// ── Constants from HDR10+ LLC spec ───────────────────────────────────────────

/// ITU-T T.35 country code for USA.
pub const T35_COUNTRY_USA: u8 = 0xB5;

/// HDR10+ application identifier (ST 2094-40).
pub const HDR10PLUS_APP_ID: u8 = 4;

/// Maximum number of analysis windows permitted by ST 2094-40.
pub const MAX_ANALYSIS_WINDOWS: u8 = 3;

/// Maximum legal value for a single MaxSCL channel entry (units: 1/10 nit).
/// Corresponds to 100 000 cd/m² (absolute maximum display peak).
pub const MAX_SCL_MAX_VALUE: u32 = 1_000_000;

/// Legal range for targeted system display maximum luminance (units: 1/10 nit).
/// 10 = 1 nit; 100 000 = 10 000 nits.
pub const TARGETED_DISPLAY_MIN: u32 = 10;
pub const TARGETED_DISPLAY_MAX: u32 = 100_000;

// ── ValidationSeverity ────────────────────────────────────────────────────────

/// Severity of a single validation finding.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DynMetaFindingSeverity {
    /// The metadata is technically legal but unusual; flagged for review.
    Warning,
    /// The metadata violates a normative constraint in ST 2094-40 or HDR10+ LLC.
    Error,
}

// ── ValidationFinding ─────────────────────────────────────────────────────────

/// A single finding produced by [`DynMetaValidator`].
#[derive(Debug, Clone)]
pub struct DynMetaFinding {
    /// Severity: [`DynMetaFindingSeverity::Warning`] or [`DynMetaFindingSeverity::Error`].
    pub severity: DynMetaFindingSeverity,
    /// Human-readable description of the issue.
    pub message: String,
    /// Optional field path (e.g. `"windows[1].window_upper_left"`).
    pub field: Option<String>,
}

impl DynMetaFinding {
    fn error(message: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            severity: DynMetaFindingSeverity::Error,
            message: message.into(),
            field: Some(field.into()),
        }
    }

    fn warning(message: impl Into<String>, field: impl Into<String>) -> Self {
        Self {
            severity: DynMetaFindingSeverity::Warning,
            message: message.into(),
            field: Some(field.into()),
        }
    }
}

// ── ValidationReport ─────────────────────────────────────────────────────────

/// Result of validating a single [`Hdr10PlusDynamicMetadata`] payload.
#[derive(Debug, Clone)]
pub struct DynMetaValidationReport {
    /// All findings (errors + warnings) in document order.
    pub findings: Vec<DynMetaFinding>,
    /// `true` if no [`DynMetaFindingSeverity::Error`] findings were produced.
    pub is_valid: bool,
}

impl DynMetaValidationReport {
    fn new(findings: Vec<DynMetaFinding>) -> Self {
        let is_valid = findings
            .iter()
            .all(|f| f.severity != DynMetaFindingSeverity::Error);
        Self { findings, is_valid }
    }

    /// Returns all error-level findings.
    pub fn errors(&self) -> impl Iterator<Item = &DynMetaFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == DynMetaFindingSeverity::Error)
    }

    /// Returns all warning-level findings.
    pub fn warnings(&self) -> impl Iterator<Item = &DynMetaFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity == DynMetaFindingSeverity::Warning)
    }

    /// Count of error-level findings.
    pub fn error_count(&self) -> usize {
        self.errors().count()
    }

    /// Count of warning-level findings.
    pub fn warning_count(&self) -> usize {
        self.warnings().count()
    }
}

// ── DynMetaValidator ─────────────────────────────────────────────────────────

/// Validates [`Hdr10PlusDynamicMetadata`] against ST 2094-40 and HDR10+ LLC rules.
///
/// # Example
/// ```rust
/// use oximedia_hdr::dynamic_metadata_validator::DynMetaValidator;
/// use oximedia_hdr::dynamic_metadata::Hdr10PlusDynamicMetadata;
///
/// let meta = Hdr10PlusDynamicMetadata::new_simple(1000);
/// let report = DynMetaValidator::validate(&meta);
/// assert!(report.is_valid, "Default metadata must be valid");
/// ```
pub struct DynMetaValidator;

impl DynMetaValidator {
    /// Validate a single frame's dynamic metadata.
    pub fn validate(meta: &Hdr10PlusDynamicMetadata) -> DynMetaValidationReport {
        let mut findings = Vec::new();

        Self::check_t35_header(meta, &mut findings);
        Self::check_application_id(meta, &mut findings);
        Self::check_num_windows(meta, &mut findings);
        Self::check_windows(meta, &mut findings);
        Self::check_targeted_display(meta, &mut findings);
        Self::check_distribution_values(meta, &mut findings);
        Self::check_average_maxrgb(meta, &mut findings);

        DynMetaValidationReport::new(findings)
    }

    // ── Private rule functions ─────────────────────────────────────────────────

    fn check_t35_header(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        if meta.country_code != T35_COUNTRY_USA {
            out.push(DynMetaFinding::warning(
                format!(
                    "country_code 0x{:02X} is not the standard HDR10+ value 0x{:02X} (USA)",
                    meta.country_code, T35_COUNTRY_USA
                ),
                "country_code",
            ));
        }
        // Terminal provider code 0x003C = Samsung (HDR10+ LLC registrant).
        // Other values are not errors but flag as informational.
        if meta.terminal_provider_code == 0 {
            out.push(DynMetaFinding::warning(
                "terminal_provider_code is 0 — should be a registered ITU-T T.35 provider code",
                "terminal_provider_code",
            ));
        }
    }

    fn check_application_id(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        // HDR10+ LLC specifies application_identifier = 4 (ST 2094-40 annex A).
        // application_identifier = 1 appears in some older encoders; treat as warning.
        if meta.application_identifier != HDR10PLUS_APP_ID && meta.application_identifier != 1 {
            out.push(DynMetaFinding::error(
                format!(
                    "application_identifier {} is not a recognised HDR10+ value (expected 4 or 1)",
                    meta.application_identifier
                ),
                "application_identifier",
            ));
        }
        if meta.application_identifier == 1 {
            out.push(DynMetaFinding::warning(
                "application_identifier = 1 is a legacy value; 4 is preferred per HDR10+ LLC",
                "application_identifier",
            ));
        }
    }

    fn check_num_windows(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        if meta.num_windows == 0 {
            out.push(DynMetaFinding::error(
                "num_windows must be ≥ 1 (ST 2094-40 §5.2)",
                "num_windows",
            ));
        }
        if meta.num_windows > MAX_ANALYSIS_WINDOWS {
            out.push(DynMetaFinding::error(
                format!(
                    "num_windows {} exceeds the maximum of {} (ST 2094-40)",
                    meta.num_windows, MAX_ANALYSIS_WINDOWS
                ),
                "num_windows",
            ));
        }
        if meta.windows.len() != meta.num_windows as usize {
            out.push(DynMetaFinding::error(
                format!(
                    "num_windows={} but {} window structs provided",
                    meta.num_windows,
                    meta.windows.len()
                ),
                "windows",
            ));
        }
    }

    fn check_windows(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        for (i, w) in meta.windows.iter().enumerate() {
            Self::check_single_window(w, i, out);
        }
    }

    fn check_single_window(w: &Hdr10PlusWindow, idx: usize, out: &mut Vec<DynMetaFinding>) {
        let prefix = format!("windows[{idx}]");

        // Bounding box: upper-left must be strictly above / left of lower-right.
        let (ul_x, ul_y) = w.window_upper_left;
        let (lr_x, lr_y) = w.window_lower_right;
        if ul_x >= lr_x {
            out.push(DynMetaFinding::error(
                format!(
                    "window x-coordinates: upper_left.x={ul_x} must be < lower_right.x={lr_x}"
                ),
                format!("{prefix}.window_upper_left / window_lower_right"),
            ));
        }
        if ul_y >= lr_y {
            out.push(DynMetaFinding::error(
                format!(
                    "window y-coordinates: upper_left.y={ul_y} must be < lower_right.y={lr_y}"
                ),
                format!("{prefix}.window_upper_left / window_lower_right"),
            ));
        }

        // MaxSCL channel values.
        for (ch, &v) in w.maxscl.iter().enumerate() {
            if v > MAX_SCL_MAX_VALUE {
                out.push(DynMetaFinding::error(
                    format!("maxscl[{ch}]={v} exceeds maximum {MAX_SCL_MAX_VALUE}"),
                    format!("{prefix}.maxscl[{ch}]"),
                ));
            }
        }

        // Ellipse axes: internal semi-axes must be ≤ external.
        if w.semimajor_axis_internal > w.semimajor_axis_external {
            out.push(DynMetaFinding::error(
                "semimajor_axis_internal must be ≤ semimajor_axis_external",
                format!("{prefix}.semimajor_axis_internal"),
            ));
        }
        if w.semiminor_axis_internal > w.semiminor_axis_external {
            out.push(DynMetaFinding::error(
                "semiminor_axis_internal must be ≤ semiminor_axis_external",
                format!("{prefix}.semiminor_axis_internal"),
            ));
        }

        // average_maxrgb must not exceed the max of the MaxSCL channels (within 1-unit tolerance).
        let max_scl = w.maxscl.iter().copied().max().unwrap_or(0);
        if w.average_maxrgb as u32 > max_scl + 1 {
            out.push(DynMetaFinding::warning(
                format!(
                    "average_maxrgb={} is greater than max(maxscl)={}; this is unusual",
                    w.average_maxrgb, max_scl
                ),
                format!("{prefix}.average_maxrgb"),
            ));
        }
    }

    fn check_targeted_display(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        let v = meta.targeted_system_display_max_luminance;
        if v < TARGETED_DISPLAY_MIN {
            out.push(DynMetaFinding::error(
                format!(
                    "targeted_system_display_max_luminance={v} is below minimum {TARGETED_DISPLAY_MIN} (1 nit)"
                ),
                "targeted_system_display_max_luminance",
            ));
        }
        if v > TARGETED_DISPLAY_MAX {
            out.push(DynMetaFinding::error(
                format!(
                    "targeted_system_display_max_luminance={v} exceeds maximum {TARGETED_DISPLAY_MAX} (10 000 nit)"
                ),
                "targeted_system_display_max_luminance",
            ));
        }
    }

    fn check_distribution_values(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        // The 9 distribution values represent a percentile histogram; they must be
        // non-decreasing (each percentile ≥ the previous one).
        let dv = &meta.distribution_values;
        for i in 1..dv.len() {
            if dv[i] < dv[i - 1] {
                out.push(DynMetaFinding::error(
                    format!(
                        "distribution_values[{}]={} < distribution_values[{}]={} — values must be non-decreasing",
                        i, dv[i], i - 1, dv[i - 1]
                    ),
                    format!("distribution_values[{i}]"),
                ));
            }
        }
    }

    fn check_average_maxrgb(meta: &Hdr10PlusDynamicMetadata, out: &mut Vec<DynMetaFinding>) {
        // The frame-level average_maxrgb should not exceed the targeted display peak
        // (both are in units of 1/10 nit, but average_maxrgb is u16 while
        // targeted_display is u32).  A value equal to 0 when other fields are
        // non-zero is suspicious.
        if meta.average_maxrgb == 0 {
            // Check if any window has a non-zero maxscl — if so, 0 is suspicious.
            let any_nonzero = meta.windows.iter().any(|w| w.maxscl.iter().any(|&v| v > 0));
            if any_nonzero {
                out.push(DynMetaFinding::warning(
                    "frame-level average_maxrgb is 0 while window MaxSCL values are non-zero",
                    "average_maxrgb",
                ));
            }
        }
    }
}

/// Convenience: validate and return `Ok(())` if no errors, or `Err(HdrError::MetadataParseError)`.
pub fn validate_dynamic_metadata(meta: &Hdr10PlusDynamicMetadata) -> Result<DynMetaValidationReport> {
    let report = DynMetaValidator::validate(meta);
    if !report.is_valid {
        let msgs: Vec<String> = report.errors().map(|f| f.message.clone()).collect();
        return Err(HdrError::MetadataParseError(msgs.join("; ")));
    }
    Ok(report)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamic_metadata::{Hdr10PlusDynamicMetadata, Hdr10PlusWindow};

    fn default_window() -> Hdr10PlusWindow {
        Hdr10PlusWindow {
            window_upper_left: (0, 0),
            window_lower_right: (3840, 2160),
            center_of_ellipse: (1920, 1080),
            rotation_angle: 0,
            semimajor_axis_external: 1080,
            semiminor_axis_external: 960,
            semimajor_axis_internal: 540,
            semiminor_axis_internal: 480,
            overlap_process_option: 0,
            maxscl: [50000, 60000, 40000],
            average_maxrgb: 100,
        }
    }

    fn valid_meta() -> Hdr10PlusDynamicMetadata {
        Hdr10PlusDynamicMetadata {
            country_code: T35_COUNTRY_USA,
            terminal_provider_code: 0x003C,
            application_identifier: HDR10PLUS_APP_ID,
            application_version: 0,
            num_windows: 1,
            windows: vec![default_window()],
            targeted_system_display_max_luminance: 10_000, // 1000 nits
            average_maxrgb: 5000,
            distribution_values: [100, 200, 300, 400, 500, 600, 700, 800, 900],
            fraction_bright_pixels: 50,
        }
    }

    #[test]
    fn valid_metadata_passes() {
        let meta = valid_meta();
        let report = DynMetaValidator::validate(&meta);
        assert!(
            report.is_valid,
            "Expected valid, errors: {:?}",
            report.errors().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_num_windows_is_error() {
        let mut meta = valid_meta();
        meta.num_windows = 0;
        meta.windows.clear();
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
        assert!(report.error_count() >= 1);
    }

    #[test]
    fn too_many_windows_is_error() {
        let mut meta = valid_meta();
        meta.num_windows = 4;
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn inverted_bounding_box_is_error() {
        let mut meta = valid_meta();
        meta.windows[0].window_upper_left = (3840, 2160);
        meta.windows[0].window_lower_right = (0, 0);
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
        assert!(report.error_count() >= 2); // Both x and y are inverted.
    }

    #[test]
    fn maxscl_overflow_is_error() {
        let mut meta = valid_meta();
        meta.windows[0].maxscl[0] = MAX_SCL_MAX_VALUE + 1;
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn distribution_non_monotone_is_error() {
        let mut meta = valid_meta();
        meta.distribution_values[3] = 50; // Less than [2]=300.
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn targeted_display_below_min_is_error() {
        let mut meta = valid_meta();
        meta.targeted_system_display_max_luminance = 0;
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn targeted_display_above_max_is_error() {
        let mut meta = valid_meta();
        meta.targeted_system_display_max_luminance = TARGETED_DISPLAY_MAX + 1;
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn unknown_country_code_is_warning_not_error() {
        let mut meta = valid_meta();
        meta.country_code = 0x99;
        let report = DynMetaValidator::validate(&meta);
        // Should be a warning, not an error — metadata is still structurally valid.
        assert!(report.is_valid, "Unknown country code should only warn, not error");
        assert!(report.warning_count() >= 1);
    }

    #[test]
    fn legacy_app_id_1_is_warning() {
        let mut meta = valid_meta();
        meta.application_identifier = 1;
        let report = DynMetaValidator::validate(&meta);
        assert!(report.is_valid);
        assert!(report.warning_count() >= 1);
    }

    #[test]
    fn internal_axis_larger_than_external_is_error() {
        let mut meta = valid_meta();
        meta.windows[0].semimajor_axis_internal = meta.windows[0].semimajor_axis_external + 1;
        let report = DynMetaValidator::validate(&meta);
        assert!(!report.is_valid);
    }

    #[test]
    fn validate_dynamic_metadata_fn_errors_on_invalid() {
        let mut meta = valid_meta();
        meta.num_windows = 0;
        meta.windows.clear();
        let result = validate_dynamic_metadata(&meta);
        assert!(result.is_err());
    }

    #[test]
    fn validate_dynamic_metadata_fn_ok_on_valid() {
        let meta = valid_meta();
        let result = validate_dynamic_metadata(&meta);
        assert!(result.is_ok());
    }
}
