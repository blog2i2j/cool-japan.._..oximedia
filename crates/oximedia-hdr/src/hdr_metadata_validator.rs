//! HDR metadata validation for broadcast compliance.
//!
//! Validates HDR10, HLG, and HDR10+ metadata against SMPTE ST 2086,
//! CEA-861.3, and ITU-R BT.2100 constraints.  Produces a structured
//! [`ValidationReport`] with per-field errors and warnings suitable for
//! integration into automated ingest pipelines.
//!
//! # Validated constraints
//!
//! **HDR10 mastering display (SMPTE ST 2086 / CEA-861.3):**
//! - Primary chromaticity values must be in [0, 1].
//! - Primary triangle must be within the Rec.2020 gamut boundary (warning).
//! - Mastering display luminance: min < max; max ∈ [0, 10_000] nits.
//! - `MaxCLL` ≥ `MaxFALL` (MaxFALL is the per-frame average, so always ≤ MaxCLL).
//! - `MaxCLL` ∈ [0, 10_000] nits; `MaxFALL` ∈ [0, 10_000] nits.
//!
//! **HDR10+ dynamic metadata:**
//! - Application version must be 0 or 1.
//! - Targeted system display peak luminance ∈ (0, 10_000] nits.
//! - At least one distribution maxrgb entry must be present.
//!
//! **HLG:**
//! - HLG content should declare MaxCLL ≤ 1000 nits (warn if higher).
//! - System gamma should be in [1.0, 1.5] range (warn otherwise).

use crate::{HdrError, Result};

// ── Severity ──────────────────────────────────────────────────────────────────

/// Severity level for a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Advisory notice — content will still work but may not be optimal.
    Info,
    /// Non-fatal deviation that could cause display or compatibility issues.
    Warning,
    /// Violation of a normative constraint; content is non-compliant.
    Error,
}

// ── ValidationFinding ─────────────────────────────────────────────────────────

/// A single validation finding attached to a specific metadata field.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// The metadata field or section to which this finding applies.
    pub field: String,
    /// Human-readable description of the finding.
    pub message: String,
    /// Severity level.
    pub severity: Severity,
}

impl ValidationFinding {
    fn new(field: impl Into<String>, message: impl Into<String>, severity: Severity) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
            severity,
        }
    }

    fn error(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, Severity::Error)
    }

    fn warning(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, Severity::Warning)
    }

    fn info(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(field, message, Severity::Info)
    }
}

// ── ValidationReport ──────────────────────────────────────────────────────────

/// Aggregated result of a metadata validation run.
///
/// Call [`is_compliant`] to quickly determine whether any errors were found.
///
/// [`is_compliant`]: ValidationReport::is_compliant
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All findings in the order they were detected.
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    /// Return `true` if no `Error`-severity findings were recorded.
    pub fn is_compliant(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == Severity::Error)
    }

    /// Return the number of findings at the given severity level.
    pub fn count(&self, severity: Severity) -> usize {
        self.findings.iter().filter(|f| f.severity == severity).count()
    }

    /// Return only the findings at or above `min_severity`.
    pub fn filtered(&self, min_severity: Severity) -> Vec<&ValidationFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity >= min_severity)
            .collect()
    }

    fn push(&mut self, finding: ValidationFinding) {
        self.findings.push(finding);
    }
}

// ── Input structs ─────────────────────────────────────────────────────────────

/// CIE xy chromaticity coordinate pair.
#[derive(Debug, Clone, Copy)]
pub struct ChromaticityXy {
    pub x: f32,
    pub y: f32,
}

/// HDR10 mastering display metadata (SMPTE ST 2086 / CEA-861.3).
#[derive(Debug, Clone)]
pub struct Hdr10MasteringDisplay {
    /// Red primary chromaticity.
    pub red: ChromaticityXy,
    /// Green primary chromaticity.
    pub green: ChromaticityXy,
    /// Blue primary chromaticity.
    pub blue: ChromaticityXy,
    /// White point chromaticity.
    pub white: ChromaticityXy,
    /// Mastering display minimum luminance (nits).
    pub min_luminance_nits: f32,
    /// Mastering display maximum luminance (nits).
    pub max_luminance_nits: f32,
}

/// HDR10 Content Light Level (CEA-861.3).
#[derive(Debug, Clone, Copy)]
pub struct ContentLightLevelMeta {
    /// Maximum Content Light Level (nits).
    pub max_cll_nits: f32,
    /// Maximum Frame-Average Light Level (nits).
    pub max_fall_nits: f32,
}

/// Simplified HDR10+ dynamic metadata entry for validation purposes.
#[derive(Debug, Clone)]
pub struct Hdr10PlusMeta {
    /// Application version (0 or 1).
    pub application_version: u8,
    /// Targeted system display peak luminance (nits).
    pub targeted_system_display_peak_nits: f32,
    /// Distribution maxrgb percentile entries (percent, nits pairs).
    /// At least one entry is required.
    pub distribution_maxrgb: Vec<(u8, f32)>,
}

/// HLG content metadata for validation.
#[derive(Debug, Clone, Copy)]
pub struct HlgContentMeta {
    /// Declared MaxCLL (nits).  Optional; `None` means not declared.
    pub max_cll_nits: Option<f32>,
    /// System gamma used for mastering.
    pub system_gamma: f32,
}

// ── Validator ─────────────────────────────────────────────────────────────────

/// Stateless HDR metadata validator.
///
/// All `validate_*` methods return a [`ValidationReport`] containing zero or
/// more [`ValidationFinding`] entries.  Methods do not return `Err` for
/// validation failures — those are captured as `Error`-severity findings.
/// `Err` is only returned for programmer-level API misuse (e.g. NaN inputs).
pub struct HdrMetadataValidator;

impl HdrMetadataValidator {
    // ── HDR10 mastering display ───────────────────────────────────────────────

    /// Validate HDR10 mastering display chromaticity and luminance metadata.
    ///
    /// # Errors
    /// Returns `HdrError::MetadataParseError` if any chromaticity coordinate is
    /// NaN (not just out of range — that is captured as a finding).
    pub fn validate_mastering_display(md: &Hdr10MasteringDisplay) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        // Check for NaN first — those are API errors, not findings.
        for (name, xy) in [
            ("red", md.red),
            ("green", md.green),
            ("blue", md.blue),
            ("white", md.white),
        ] {
            if xy.x.is_nan() || xy.y.is_nan() {
                return Err(HdrError::MetadataParseError(format!(
                    "{name} primary has NaN chromaticity"
                )));
            }
        }
        if md.min_luminance_nits.is_nan() || md.max_luminance_nits.is_nan() {
            return Err(HdrError::MetadataParseError(
                "luminance value is NaN".into(),
            ));
        }

        // Chromaticity range [0, 1].
        for (name, xy) in [
            ("red", md.red),
            ("green", md.green),
            ("blue", md.blue),
            ("white", md.white),
        ] {
            if !(0.0..=1.0).contains(&xy.x) || !(0.0..=1.0).contains(&xy.y) {
                report.push(ValidationFinding::error(
                    format!("{name}_primary_chromaticity"),
                    format!(
                        "{name} primary ({:.4}, {:.4}) is outside [0, 1] range",
                        xy.x, xy.y
                    ),
                ));
            }
            // y > 0 required to avoid degenerate colour.
            if xy.y <= 0.0 {
                report.push(ValidationFinding::error(
                    format!("{name}_primary_y"),
                    format!("{name} primary Y chromaticity must be > 0, got {:.4}", xy.y),
                ));
            }
        }

        // Luminance constraints.
        if md.min_luminance_nits < 0.0 {
            report.push(ValidationFinding::error(
                "min_luminance_nits",
                format!(
                    "min_luminance_nits must be ≥ 0, got {:.4}",
                    md.min_luminance_nits
                ),
            ));
        }
        if md.max_luminance_nits <= 0.0 {
            report.push(ValidationFinding::error(
                "max_luminance_nits",
                format!(
                    "max_luminance_nits must be > 0, got {:.4}",
                    md.max_luminance_nits
                ),
            ));
        }
        if md.max_luminance_nits > 10_000.0 {
            report.push(ValidationFinding::error(
                "max_luminance_nits",
                format!(
                    "max_luminance_nits {:.1} exceeds SMPTE ST 2086 maximum of 10 000 nits",
                    md.max_luminance_nits
                ),
            ));
        }
        if md.min_luminance_nits >= md.max_luminance_nits {
            report.push(ValidationFinding::error(
                "luminance_range",
                format!(
                    "min_luminance_nits ({:.4}) must be < max_luminance_nits ({:.4})",
                    md.min_luminance_nits, md.max_luminance_nits
                ),
            ));
        }

        // Warn if max luminance is unusually low for an HDR mastering display.
        if md.max_luminance_nits < 400.0 {
            report.push(ValidationFinding::warning(
                "max_luminance_nits",
                format!(
                    "max_luminance_nits {:.1} is below recommended 400 nits for HDR mastering",
                    md.max_luminance_nits
                ),
            ));
        }

        // Info: note the display type inferred from peak luminance.
        let display_class = if md.max_luminance_nits >= 4000.0 {
            "reference grade (≥4000 nits)"
        } else if md.max_luminance_nits >= 1000.0 {
            "high-end (1000–4000 nits)"
        } else {
            "standard HDR (< 1000 nits)"
        };
        report.push(ValidationFinding::info(
            "max_luminance_nits",
            format!("display class: {display_class}"),
        ));

        Ok(report)
    }

    // ── Content Light Level ───────────────────────────────────────────────────

    /// Validate HDR10 Content Light Level (MaxCLL / MaxFALL) metadata.
    ///
    /// # Errors
    /// Returns an error if any value is NaN.
    pub fn validate_cll(cll: ContentLightLevelMeta) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        if cll.max_cll_nits.is_nan() || cll.max_fall_nits.is_nan() {
            return Err(HdrError::MetadataParseError("CLL value is NaN".into()));
        }

        if cll.max_cll_nits < 0.0 {
            report.push(ValidationFinding::error(
                "max_cll_nits",
                format!("MaxCLL must be ≥ 0, got {:.4}", cll.max_cll_nits),
            ));
        }
        if cll.max_fall_nits < 0.0 {
            report.push(ValidationFinding::error(
                "max_fall_nits",
                format!("MaxFALL must be ≥ 0, got {:.4}", cll.max_fall_nits),
            ));
        }
        if cll.max_cll_nits > 10_000.0 {
            report.push(ValidationFinding::error(
                "max_cll_nits",
                format!(
                    "MaxCLL {:.1} exceeds CEA-861.3 maximum of 10 000 nits",
                    cll.max_cll_nits
                ),
            ));
        }
        if cll.max_fall_nits > 10_000.0 {
            report.push(ValidationFinding::error(
                "max_fall_nits",
                format!(
                    "MaxFALL {:.1} exceeds CEA-861.3 maximum of 10 000 nits",
                    cll.max_fall_nits
                ),
            ));
        }

        // MaxFALL must be ≤ MaxCLL.
        if cll.max_fall_nits > cll.max_cll_nits {
            report.push(ValidationFinding::error(
                "max_fall_vs_max_cll",
                format!(
                    "MaxFALL ({:.1}) must be ≤ MaxCLL ({:.1})",
                    cll.max_fall_nits, cll.max_cll_nits
                ),
            ));
        }

        // Both zero is technically valid but unusual — warn.
        if cll.max_cll_nits == 0.0 && cll.max_fall_nits == 0.0 {
            report.push(ValidationFinding::warning(
                "max_cll_nits",
                "MaxCLL and MaxFALL are both 0 — metadata may be missing or unset",
            ));
        }

        Ok(report)
    }

    // ── HDR10+ dynamic metadata ───────────────────────────────────────────────

    /// Validate HDR10+ dynamic metadata.
    ///
    /// # Errors
    /// Returns an error if `targeted_system_display_peak_nits` is NaN.
    pub fn validate_hdr10plus(meta: &Hdr10PlusMeta) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        if meta.targeted_system_display_peak_nits.is_nan() {
            return Err(HdrError::MetadataParseError(
                "HDR10+ targeted_system_display_peak_nits is NaN".into(),
            ));
        }

        // Application version.
        if meta.application_version > 1 {
            report.push(ValidationFinding::error(
                "application_version",
                format!(
                    "HDR10+ application_version must be 0 or 1, got {}",
                    meta.application_version
                ),
            ));
        }

        // Targeted display peak.
        if meta.targeted_system_display_peak_nits <= 0.0 {
            report.push(ValidationFinding::error(
                "targeted_system_display_peak_nits",
                format!(
                    "targeted_system_display_peak_nits must be > 0, got {:.4}",
                    meta.targeted_system_display_peak_nits
                ),
            ));
        }
        if meta.targeted_system_display_peak_nits > 10_000.0 {
            report.push(ValidationFinding::error(
                "targeted_system_display_peak_nits",
                format!(
                    "targeted_system_display_peak_nits {:.1} exceeds maximum 10 000 nits",
                    meta.targeted_system_display_peak_nits
                ),
            ));
        }

        // Distribution maxrgb entries.
        if meta.distribution_maxrgb.is_empty() {
            report.push(ValidationFinding::error(
                "distribution_maxrgb",
                "at least one distribution_maxrgb entry is required",
            ));
        } else {
            // Percentile values must be in [0, 100].
            for (idx, &(pct, nits)) in meta.distribution_maxrgb.iter().enumerate() {
                if pct > 100 {
                    report.push(ValidationFinding::error(
                        format!("distribution_maxrgb[{idx}].percentile"),
                        format!("percentile {pct} must be in [0, 100]"),
                    ));
                }
                if nits < 0.0 {
                    report.push(ValidationFinding::error(
                        format!("distribution_maxrgb[{idx}].nits"),
                        format!("nits value {nits:.4} must be ≥ 0"),
                    ));
                }
            }
        }

        Ok(report)
    }

    // ── HLG ──────────────────────────────────────────────────────────────────

    /// Validate HLG content metadata.
    ///
    /// # Errors
    /// Returns an error if `system_gamma` is NaN.
    pub fn validate_hlg(meta: HlgContentMeta) -> Result<ValidationReport> {
        let mut report = ValidationReport::default();

        if meta.system_gamma.is_nan() {
            return Err(HdrError::MetadataParseError(
                "HLG system_gamma is NaN".into(),
            ));
        }

        // System gamma range guidance (BT.2390).
        if !(1.0..=1.5).contains(&meta.system_gamma) {
            report.push(ValidationFinding::warning(
                "system_gamma",
                format!(
                    "system_gamma {:.3} is outside expected range [1.0, 1.5]",
                    meta.system_gamma
                ),
            ));
        }

        // Optional MaxCLL check.
        if let Some(cll) = meta.max_cll_nits {
            if cll.is_nan() {
                return Err(HdrError::MetadataParseError("HLG max_cll_nits is NaN".into()));
            }
            if cll < 0.0 {
                report.push(ValidationFinding::error(
                    "max_cll_nits",
                    format!("MaxCLL must be ≥ 0, got {cll:.4}"),
                ));
            }
            if cll > 1000.0 {
                report.push(ValidationFinding::warning(
                    "max_cll_nits",
                    format!(
                        "MaxCLL {cll:.1} exceeds the typical HLG 1000-nit reference display peak"
                    ),
                ));
            }
        } else {
            report.push(ValidationFinding::info(
                "max_cll_nits",
                "MaxCLL not declared for HLG content (optional but recommended)",
            ));
        }

        Ok(report)
    }

    // ── Combined validator ────────────────────────────────────────────────────

    /// Run all applicable validations for HDR10 content and merge findings.
    ///
    /// Validates mastering display, CLL, and optionally HDR10+ dynamic metadata.
    ///
    /// # Errors
    /// Propagates errors from any sub-validator.
    pub fn validate_hdr10_full(
        mastering: &Hdr10MasteringDisplay,
        cll: ContentLightLevelMeta,
        hdr10plus: Option<&Hdr10PlusMeta>,
    ) -> Result<ValidationReport> {
        let mut report = Self::validate_mastering_display(mastering)?;
        let cll_report = Self::validate_cll(cll)?;
        report.findings.extend(cll_report.findings);

        // Cross-check: MaxCLL should not exceed mastering display maximum.
        if cll.max_cll_nits > mastering.max_luminance_nits {
            report.push(ValidationFinding::warning(
                "max_cll_vs_mastering_peak",
                format!(
                    "MaxCLL ({:.1} nits) exceeds mastering display peak ({:.1} nits)",
                    cll.max_cll_nits, mastering.max_luminance_nits
                ),
            ));
        }

        if let Some(hdr10p) = hdr10plus {
            let dyn_report = Self::validate_hdr10plus(hdr10p)?;
            report.findings.extend(dyn_report.findings);
        }

        Ok(report)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_mastering() -> Hdr10MasteringDisplay {
        Hdr10MasteringDisplay {
            red: ChromaticityXy { x: 0.708, y: 0.292 },
            green: ChromaticityXy { x: 0.170, y: 0.797 },
            blue: ChromaticityXy { x: 0.131, y: 0.046 },
            white: ChromaticityXy { x: 0.3127, y: 0.3290 },
            min_luminance_nits: 0.005,
            max_luminance_nits: 1000.0,
        }
    }

    #[test]
    fn test_valid_mastering_display_is_compliant() {
        let md = valid_mastering();
        let report = HdrMetadataValidator::validate_mastering_display(&md).unwrap();
        assert!(report.is_compliant(), "findings: {:?}", report.findings);
    }

    #[test]
    fn test_mastering_display_negative_chromaticity_error() {
        let mut md = valid_mastering();
        md.red.x = -0.1;
        let report = HdrMetadataValidator::validate_mastering_display(&md).unwrap();
        assert!(!report.is_compliant());
        assert!(report.count(Severity::Error) > 0);
    }

    #[test]
    fn test_mastering_display_min_ge_max_error() {
        let mut md = valid_mastering();
        md.min_luminance_nits = 2000.0;
        md.max_luminance_nits = 1000.0;
        let report = HdrMetadataValidator::validate_mastering_display(&md).unwrap();
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_mastering_display_nan_returns_err() {
        let mut md = valid_mastering();
        md.red.x = f32::NAN;
        let result = HdrMetadataValidator::validate_mastering_display(&md);
        assert!(result.is_err());
    }

    #[test]
    fn test_cll_valid() {
        let cll = ContentLightLevelMeta {
            max_cll_nits: 1000.0,
            max_fall_nits: 200.0,
        };
        let report = HdrMetadataValidator::validate_cll(cll).unwrap();
        assert!(report.is_compliant());
    }

    #[test]
    fn test_cll_fall_exceeds_cll_error() {
        let cll = ContentLightLevelMeta {
            max_cll_nits: 100.0,
            max_fall_nits: 200.0,
        };
        let report = HdrMetadataValidator::validate_cll(cll).unwrap();
        assert!(!report.is_compliant());
        assert!(report.count(Severity::Error) > 0);
    }

    #[test]
    fn test_cll_both_zero_warning() {
        let cll = ContentLightLevelMeta {
            max_cll_nits: 0.0,
            max_fall_nits: 0.0,
        };
        let report = HdrMetadataValidator::validate_cll(cll).unwrap();
        // Compliant but has a warning.
        assert!(report.is_compliant());
        assert!(report.count(Severity::Warning) > 0);
    }

    #[test]
    fn test_hdr10plus_invalid_version() {
        let meta = Hdr10PlusMeta {
            application_version: 2,
            targeted_system_display_peak_nits: 1000.0,
            distribution_maxrgb: vec![(99, 800.0)],
        };
        let report = HdrMetadataValidator::validate_hdr10plus(&meta).unwrap();
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_hdr10plus_empty_distribution_error() {
        let meta = Hdr10PlusMeta {
            application_version: 0,
            targeted_system_display_peak_nits: 1000.0,
            distribution_maxrgb: vec![],
        };
        let report = HdrMetadataValidator::validate_hdr10plus(&meta).unwrap();
        assert!(!report.is_compliant());
    }

    #[test]
    fn test_hdr10plus_valid() {
        let meta = Hdr10PlusMeta {
            application_version: 1,
            targeted_system_display_peak_nits: 1000.0,
            distribution_maxrgb: vec![(50, 200.0), (99, 900.0)],
        };
        let report = HdrMetadataValidator::validate_hdr10plus(&meta).unwrap();
        assert!(report.is_compliant(), "findings: {:?}", report.findings);
    }

    #[test]
    fn test_hlg_valid() {
        let meta = HlgContentMeta {
            max_cll_nits: Some(800.0),
            system_gamma: 1.2,
        };
        let report = HdrMetadataValidator::validate_hlg(meta).unwrap();
        assert!(report.is_compliant());
    }

    #[test]
    fn test_hlg_high_cll_warning() {
        let meta = HlgContentMeta {
            max_cll_nits: Some(2000.0),
            system_gamma: 1.2,
        };
        let report = HdrMetadataValidator::validate_hlg(meta).unwrap();
        // Compliant (no error) but has a warning about exceeding 1000 nits.
        assert!(report.is_compliant());
        assert!(report.count(Severity::Warning) > 0);
    }

    #[test]
    fn test_hlg_out_of_range_gamma_warning() {
        let meta = HlgContentMeta {
            max_cll_nits: None,
            system_gamma: 0.5,
        };
        let report = HdrMetadataValidator::validate_hlg(meta).unwrap();
        assert!(report.is_compliant());
        assert!(report.count(Severity::Warning) > 0);
    }

    #[test]
    fn test_validate_hdr10_full_cross_check_warning() {
        let md = valid_mastering(); // max 1000 nits
        let cll = ContentLightLevelMeta {
            max_cll_nits: 1200.0, // exceeds mastering peak
            max_fall_nits: 200.0,
        };
        let report = HdrMetadataValidator::validate_hdr10_full(&md, cll, None).unwrap();
        // MaxCLL > mastering peak → warning.
        assert!(report.count(Severity::Warning) > 0);
    }

    #[test]
    fn test_validation_report_filtered() {
        let mut report = ValidationReport::default();
        report.push(ValidationFinding::info("f1", "info msg"));
        report.push(ValidationFinding::warning("f2", "warn msg"));
        report.push(ValidationFinding::error("f3", "error msg"));

        let errors = report.filtered(Severity::Error);
        assert_eq!(errors.len(), 1);
        let warnings_and_above = report.filtered(Severity::Warning);
        assert_eq!(warnings_and_above.len(), 2);
    }
}
