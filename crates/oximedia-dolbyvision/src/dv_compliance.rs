//! Dolby Vision compliance checker — profile constraints validation, L1/L2/L6
//! metadata consistency checks, and conformance reporting.
//!
//! This module verifies that a set of Dolby Vision metadata satisfies the
//! constraints imposed by a given profile and delivery specification.  It is
//! intended for quality-control workflows and pre-delivery validation.
//!
//! # Validation Layers
//!
//! | Layer | Description |
//! |-------|-------------|
//! | L1 consistency | min ≤ mid ≤ max, values in [0, 1] |
//! | L2 count | at least one L2 entry per shot when required |
//! | L6 presence | mastering metadata must be present for broadcast deliveries |
//! | Profile constraints | profile-specific peak luminance limits |
//! | Cross-field consistency | L1 max must not exceed L6 mastering peak |
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::dv_compliance::{
//!     ComplianceChecker, ComplianceSpec, ComplianceReport, ComplianceViolation,
//!     MasteringDisplayInfo,
//! };
//! use oximedia_dolbyvision::dv_xml_export::{DvShotEntry, DvL2Entry};
//!
//! let checker = ComplianceChecker::new(ComplianceSpec::Broadcast);
//!
//! let shots = vec![DvShotEntry {
//!     frame_start: 0,
//!     frame_end: 23,
//!     l1_min: 0.0,
//!     l1_mid: 0.1,
//!     l1_max: 0.45,
//!     l2_entries: vec![DvL2Entry::identity(2081)],
//! }];
//!
//! let mastering = MasteringDisplayInfo::standard_4000nit();
//! let report = checker.check_shots(&shots, Some(&mastering));
//! assert!(report.is_compliant(), "should be compliant: {:?}", report.violations);
//! ```

use crate::dv_xml_export::DvShotEntry;

// ── Compliance specification ──────────────────────────────────────────────────

/// Specifies the ruleset used for compliance checking.
///
/// Higher specs apply all rules from lower specs plus additional constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComplianceSpec {
    /// Permissive: structural validity only (L1 range and ordering).
    Permissive,
    /// Streaming: adds L2 entry requirement, trims range checks.
    Streaming,
    /// Broadcast: adds L6 mastering metadata requirement and
    /// cross-field consistency checks.
    Broadcast,
}

impl std::fmt::Display for ComplianceSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Permissive => write!(f, "Permissive"),
            Self::Streaming => write!(f, "Streaming"),
            Self::Broadcast => write!(f, "Broadcast"),
        }
    }
}

// ── Violation severity ────────────────────────────────────────────────────────

/// Severity level of a compliance violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ViolationSeverity {
    /// Advisory — does not block delivery but should be reviewed.
    Advisory,
    /// Warning — may cause display artefacts on some displays.
    Warning,
    /// Error — must be fixed before delivery.
    Error,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Advisory => write!(f, "ADVISORY"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ── Violation categories ──────────────────────────────────────────────────────

/// Category of a compliance violation.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationKind {
    /// L1 metadata value is out of the valid normalised PQ range [0, 1].
    L1OutOfRange {
        /// Which field: "min", "mid", or "max".
        field: &'static str,
        /// The offending value.
        value: f32,
    },
    /// L1 ordering constraint violated: mid > max or min > mid.
    L1OrderingViolation {
        /// Description of the ordering constraint that was violated.
        description: String,
    },
    /// Shot has no L2 trim entries when at least one is required.
    MissingL2Entries,
    /// An L2 trim parameter is outside its valid range.
    L2TrimOutOfRange {
        /// Which trim parameter: "slope", "offset", or "power".
        param: &'static str,
        /// The offending value.
        value: f32,
        /// The allowed minimum.
        min: f32,
        /// The allowed maximum.
        max: f32,
    },
    /// L6 mastering metadata is absent but required for this delivery spec.
    MissingL6Metadata,
    /// L1 max exceeds the mastering display peak defined in L6 metadata.
    L1ExceedsL6MasteringPeak {
        /// L1 max value (normalised PQ).
        l1_max: f32,
        /// L6 mastering peak (normalised PQ).
        l6_peak: f32,
    },
    /// L1 max exceeds the profile-specific peak brightness limit.
    ExceedsProfilePeakLimit {
        /// L1 max value (normalised PQ).
        l1_max: f32,
        /// Maximum allowed value (normalised PQ).
        limit: f32,
    },
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::L1OutOfRange { field, value } => {
                write!(f, "L1 {field} value {value:.6} outside [0.0, 1.0]")
            }
            Self::L1OrderingViolation { description } => {
                write!(f, "L1 ordering violation: {description}")
            }
            Self::MissingL2Entries => write!(f, "Shot has no L2 trim entries"),
            Self::L2TrimOutOfRange {
                param,
                value,
                min,
                max,
            } => write!(
                f,
                "L2 {param} value {value:.4} outside [{min:.4}, {max:.4}]"
            ),
            Self::MissingL6Metadata => {
                write!(f, "L6 mastering metadata is absent (required)")
            }
            Self::L1ExceedsL6MasteringPeak { l1_max, l6_peak } => write!(
                f,
                "L1 max {l1_max:.6} exceeds L6 mastering peak {l6_peak:.6}"
            ),
            Self::ExceedsProfilePeakLimit { l1_max, limit } => write!(
                f,
                "L1 max {l1_max:.6} exceeds profile peak limit {limit:.6}"
            ),
        }
    }
}

// ── Compliance violation ──────────────────────────────────────────────────────

/// A single compliance violation attached to a frame or sequence location.
#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    /// Absolute frame index where the violation was detected.
    pub frame: u64,
    /// Shot index (0-based) in the sequence.
    pub shot_index: usize,
    /// Violation category.
    pub kind: ViolationKind,
    /// Severity of the violation.
    pub severity: ViolationSeverity,
}

impl ComplianceViolation {
    fn new(
        frame: u64,
        shot_index: usize,
        kind: ViolationKind,
        severity: ViolationSeverity,
    ) -> Self {
        Self {
            frame,
            shot_index,
            kind,
            severity,
        }
    }
}

impl std::fmt::Display for ComplianceViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] frame={} shot={}: {}",
            self.severity, self.frame, self.shot_index, self.kind
        )
    }
}

// ── Mastering display metadata (L6 surrogate) ─────────────────────────────────

/// Mastering display metadata corresponding to DV Level 6.
///
/// Provided optionally to enable cross-field consistency checks.
#[derive(Debug, Clone, Copy)]
pub struct MasteringDisplayInfo {
    /// Mastering display maximum luminance in nits.
    pub max_luminance_nits: f32,
    /// Mastering display minimum luminance in nits.
    pub min_luminance_nits: f32,
    /// Maximum content light level (MaxCLL) in nits.
    pub max_cll_nits: f32,
    /// Maximum frame-average light level (MaxFALL) in nits.
    pub max_fall_nits: f32,
}

impl MasteringDisplayInfo {
    /// Create a standard 4000-nit mastering display profile.
    #[must_use]
    pub fn standard_4000nit() -> Self {
        Self {
            max_luminance_nits: 4000.0,
            min_luminance_nits: 0.005,
            max_cll_nits: 4000.0,
            max_fall_nits: 400.0,
        }
    }

    /// Create a 1000-nit mastering display profile.
    #[must_use]
    pub fn standard_1000nit() -> Self {
        Self {
            max_luminance_nits: 1000.0,
            min_luminance_nits: 0.005,
            max_cll_nits: 1000.0,
            max_fall_nits: 200.0,
        }
    }

    /// Convert mastering peak to a normalised PQ value.
    #[must_use]
    pub fn peak_pq(&self) -> f32 {
        crate::display_mapping::nits_to_pq(self.max_luminance_nits)
    }
}

// ── Compliance report ─────────────────────────────────────────────────────────

/// Aggregate compliance report for a shot sequence.
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    /// Compliance specification used.
    pub spec: ComplianceSpec,
    /// All violations found.
    pub violations: Vec<ComplianceViolation>,
    /// Total number of shots checked.
    pub shots_checked: usize,
    /// Total number of frames covered.
    pub frames_checked: u64,
}

impl ComplianceReport {
    /// Return `true` if there are no `Error`-severity violations.
    #[must_use]
    pub fn is_compliant(&self) -> bool {
        !self
            .violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Error)
    }

    /// Return all violations at or above the given severity.
    #[must_use]
    pub fn violations_at_or_above(&self, severity: ViolationSeverity) -> Vec<&ComplianceViolation> {
        self.violations
            .iter()
            .filter(|v| v.severity >= severity)
            .collect()
    }

    /// Count violations by severity level.
    #[must_use]
    pub fn count_by_severity(&self, severity: ViolationSeverity) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == severity)
            .count()
    }

    /// Return a human-readable summary of the compliance report.
    #[must_use]
    pub fn summary(&self) -> String {
        let errors = self.count_by_severity(ViolationSeverity::Error);
        let warnings = self.count_by_severity(ViolationSeverity::Warning);
        let advisories = self.count_by_severity(ViolationSeverity::Advisory);
        let status = if self.is_compliant() { "PASS" } else { "FAIL" };
        format!(
            "{status} [{spec}] shots={shots} frames={frames} errors={errors} warnings={warnings} advisories={advisories}",
            spec = self.spec,
            shots = self.shots_checked,
            frames = self.frames_checked,
        )
    }
}

// ── Compliance checker ────────────────────────────────────────────────────────

/// Checks a sequence of DV shots for compliance with a given specification.
pub struct ComplianceChecker {
    /// The compliance specification to apply.
    pub spec: ComplianceSpec,
    /// Optional profile-specific peak luminance limit (normalised PQ).
    ///
    /// If `None`, a default limit of 1.0 (10 000 nits) is used.
    pub profile_peak_limit: Option<f32>,
}

impl ComplianceChecker {
    /// Create a compliance checker using the specified [`ComplianceSpec`].
    #[must_use]
    pub fn new(spec: ComplianceSpec) -> Self {
        Self {
            spec,
            profile_peak_limit: None,
        }
    }

    /// Create a checker with a custom profile peak limit.
    #[must_use]
    pub fn with_peak_limit(spec: ComplianceSpec, limit_pq: f32) -> Self {
        Self {
            spec,
            profile_peak_limit: Some(limit_pq),
        }
    }

    /// Check a sequence of shots against the compliance spec.
    ///
    /// `mastering_info` is optional mastering display metadata (L6 surrogate).
    /// When provided under [`ComplianceSpec::Broadcast`], its absence would
    /// itself be flagged, but when it is provided, cross-field consistency is
    /// also verified.
    #[must_use]
    pub fn check_shots(
        &self,
        shots: &[DvShotEntry],
        mastering_info: Option<&MasteringDisplayInfo>,
    ) -> ComplianceReport {
        let mut violations: Vec<ComplianceViolation> = Vec::new();
        let mut total_frames: u64 = 0;

        // Broadcast spec: L6 mastering metadata is required
        if self.spec >= ComplianceSpec::Broadcast && mastering_info.is_none() {
            violations.push(ComplianceViolation::new(
                0,
                0,
                ViolationKind::MissingL6Metadata,
                ViolationSeverity::Error,
            ));
        }

        for (shot_idx, shot) in shots.iter().enumerate() {
            let frame = shot.frame_start;
            let duration = shot.duration();
            total_frames += duration;

            // L1 range checks (all specs)
            self.check_l1_range(shot, frame, shot_idx, &mut violations);

            // L1 ordering checks (all specs)
            self.check_l1_ordering(shot, frame, shot_idx, &mut violations);

            // Profile peak limit (all specs if set)
            self.check_profile_peak(shot, frame, shot_idx, &mut violations);

            // Streaming and Broadcast: L2 and trim range checks
            if self.spec >= ComplianceSpec::Streaming {
                self.check_l2_presence(shot, frame, shot_idx, &mut violations);
                self.check_l2_trim_ranges(shot, frame, shot_idx, &mut violations);
            }

            // Broadcast: cross-field L1 vs L6 consistency
            if self.spec >= ComplianceSpec::Broadcast {
                if let Some(md) = mastering_info {
                    self.check_l1_vs_l6(shot, frame, shot_idx, md, &mut violations);
                }
            }
        }

        ComplianceReport {
            spec: self.spec,
            violations,
            shots_checked: shots.len(),
            frames_checked: total_frames,
        }
    }

    // ── Private checkers ──────────────────────────────────────────────────────

    fn check_l1_range(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        let fields = [
            ("min", shot.l1_min),
            ("mid", shot.l1_mid),
            ("max", shot.l1_max),
        ];
        for (field, value) in fields {
            if !(0.0..=1.0).contains(&value) {
                violations.push(ComplianceViolation::new(
                    frame,
                    shot_idx,
                    ViolationKind::L1OutOfRange { field, value },
                    ViolationSeverity::Error,
                ));
            }
        }
    }

    fn check_l1_ordering(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if shot.l1_max < shot.l1_mid {
            violations.push(ComplianceViolation::new(
                frame,
                shot_idx,
                ViolationKind::L1OrderingViolation {
                    description: format!(
                        "l1_max ({:.6}) < l1_mid ({:.6})",
                        shot.l1_max, shot.l1_mid
                    ),
                },
                ViolationSeverity::Error,
            ));
        }
        if shot.l1_mid < shot.l1_min {
            violations.push(ComplianceViolation::new(
                frame,
                shot_idx,
                ViolationKind::L1OrderingViolation {
                    description: format!(
                        "l1_mid ({:.6}) < l1_min ({:.6})",
                        shot.l1_mid, shot.l1_min
                    ),
                },
                ViolationSeverity::Error,
            ));
        }
    }

    fn check_profile_peak(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if let Some(limit) = self.profile_peak_limit {
            if shot.l1_max > limit {
                violations.push(ComplianceViolation::new(
                    frame,
                    shot_idx,
                    ViolationKind::ExceedsProfilePeakLimit {
                        l1_max: shot.l1_max,
                        limit,
                    },
                    ViolationSeverity::Warning,
                ));
            }
        }
    }

    fn check_l2_presence(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        if shot.l2_entries.is_empty() {
            violations.push(ComplianceViolation::new(
                frame,
                shot_idx,
                ViolationKind::MissingL2Entries,
                ViolationSeverity::Warning,
            ));
        }
    }

    fn check_l2_trim_ranges(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        for l2 in &shot.l2_entries {
            let checks: &[(&'static str, f32, f32, f32)] = &[
                ("slope", l2.trim_slope, 0.0, 2.0),
                ("offset", l2.trim_offset, -1.0, 1.0),
                ("power", l2.trim_power, 0.0, 2.0),
            ];
            for &(param, value, min, max) in checks {
                if value < min || value > max {
                    violations.push(ComplianceViolation::new(
                        frame,
                        shot_idx,
                        ViolationKind::L2TrimOutOfRange {
                            param,
                            value,
                            min,
                            max,
                        },
                        ViolationSeverity::Error,
                    ));
                }
            }
        }
    }

    fn check_l1_vs_l6(
        &self,
        shot: &DvShotEntry,
        frame: u64,
        shot_idx: usize,
        mastering: &MasteringDisplayInfo,
        violations: &mut Vec<ComplianceViolation>,
    ) {
        let l6_peak = mastering.peak_pq();
        if shot.l1_max > l6_peak + 1e-5 {
            violations.push(ComplianceViolation::new(
                frame,
                shot_idx,
                ViolationKind::L1ExceedsL6MasteringPeak {
                    l1_max: shot.l1_max,
                    l6_peak,
                },
                ViolationSeverity::Error,
            ));
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dv_xml_export::{DvL2Entry, DvShotEntry};

    fn valid_shot(start: u64, end: u64, l1_max: f32) -> DvShotEntry {
        DvShotEntry {
            frame_start: start,
            frame_end: end,
            l1_min: 0.001,
            l1_mid: l1_max * 0.25,
            l1_max,
            l2_entries: vec![DvL2Entry::identity(2081)],
        }
    }

    #[test]
    fn test_valid_sequence_broadcast_compliant() {
        let checker = ComplianceChecker::new(ComplianceSpec::Broadcast);
        let shots = vec![valid_shot(0, 23, 0.40), valid_shot(24, 47, 0.35)];
        let md = MasteringDisplayInfo::standard_4000nit();
        let report = checker.check_shots(&shots, Some(&md));
        assert!(
            report.is_compliant(),
            "expected compliant: {:?}",
            report.violations
        );
    }

    #[test]
    fn test_missing_l6_broadcast_violation() {
        let checker = ComplianceChecker::new(ComplianceSpec::Broadcast);
        let shots = vec![valid_shot(0, 23, 0.40)];
        let report = checker.check_shots(&shots, None);
        assert!(
            !report.is_compliant(),
            "should be non-compliant without L6"
        );
        let l6_missing = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::MissingL6Metadata));
        assert!(l6_missing, "expected MissingL6Metadata violation");
    }

    #[test]
    fn test_l1_out_of_range_error() {
        let checker = ComplianceChecker::new(ComplianceSpec::Permissive);
        let mut shot = valid_shot(0, 23, 1.5); // l1_max > 1.0
        shot.l1_mid = 0.1;
        let report = checker.check_shots(&[shot], None);
        assert!(!report.is_compliant());
        let out_of_range = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::L1OutOfRange { .. }));
        assert!(out_of_range, "expected L1OutOfRange violation");
    }

    #[test]
    fn test_l1_ordering_violation() {
        let checker = ComplianceChecker::new(ComplianceSpec::Permissive);
        let shot = DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.0,
            l1_mid: 0.5,  // mid > max: invalid
            l1_max: 0.3,
            l2_entries: vec![],
        };
        let report = checker.check_shots(&[shot], None);
        assert!(!report.is_compliant());
        let ordering = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::L1OrderingViolation { .. }));
        assert!(ordering, "expected L1OrderingViolation");
    }

    #[test]
    fn test_missing_l2_streaming_warning() {
        let checker = ComplianceChecker::new(ComplianceSpec::Streaming);
        let shot = DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.001,
            l1_mid: 0.1,
            l1_max: 0.4,
            l2_entries: vec![],
        };
        let report = checker.check_shots(&[shot], None);
        // MissingL2Entries is a Warning, not an Error → still compliant
        assert!(report.is_compliant(), "warnings should not block compliance");
        let missing_l2 = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::MissingL2Entries));
        assert!(missing_l2, "expected MissingL2Entries warning");
    }

    #[test]
    fn test_l2_trim_out_of_range_error() {
        let checker = ComplianceChecker::new(ComplianceSpec::Streaming);
        let shot = DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.001,
            l1_mid: 0.1,
            l1_max: 0.4,
            l2_entries: vec![crate::dv_xml_export::DvL2Entry {
                target_max_pq: 2081,
                trim_slope: 3.0, // > 2.0: invalid
                trim_offset: 0.0,
                trim_power: 1.0,
            }],
        };
        let report = checker.check_shots(&[shot], None);
        assert!(!report.is_compliant());
        let trim_err = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::L2TrimOutOfRange { param: "slope", .. }));
        assert!(trim_err, "expected L2TrimOutOfRange(slope) violation");
    }

    #[test]
    fn test_l1_exceeds_l6_mastering_peak() {
        let checker = ComplianceChecker::new(ComplianceSpec::Broadcast);
        // 1000-nit mastering, but shot has L1 max equivalent to 4000 nits
        let md = MasteringDisplayInfo::standard_1000nit();
        let l6_peak = md.peak_pq();
        let shot = DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.001,
            l1_mid: 0.1,
            l1_max: (l6_peak + 0.1).clamp(0.0, 1.0), // exceeds L6 peak
            l2_entries: vec![DvL2Entry::identity(2081)],
        };
        let report = checker.check_shots(&[shot], Some(&md));
        let exceeds = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::L1ExceedsL6MasteringPeak { .. }));
        assert!(exceeds, "expected L1ExceedsL6MasteringPeak violation");
    }

    #[test]
    fn test_profile_peak_limit_warning() {
        let checker = ComplianceChecker::with_peak_limit(ComplianceSpec::Permissive, 0.5);
        let shot = valid_shot(0, 23, 0.8); // l1_max > 0.5 limit
        let report = checker.check_shots(&[shot], None);
        // ExceedsProfilePeakLimit is Warning level → still compliant
        assert!(report.is_compliant());
        let peak_warn = report
            .violations
            .iter()
            .any(|v| matches!(v.kind, ViolationKind::ExceedsProfilePeakLimit { .. }));
        assert!(peak_warn, "expected ExceedsProfilePeakLimit warning");
    }

    #[test]
    fn test_summary_format_pass() {
        let checker = ComplianceChecker::new(ComplianceSpec::Streaming);
        let shots = vec![valid_shot(0, 23, 0.35)];
        let report = checker.check_shots(&shots, None);
        let summary = report.summary();
        assert!(summary.starts_with("PASS"), "summary={summary}");
    }

    #[test]
    fn test_count_by_severity() {
        let checker = ComplianceChecker::new(ComplianceSpec::Streaming);
        let shot = DvShotEntry {
            frame_start: 0,
            frame_end: 23,
            l1_min: 0.001,
            l1_mid: 0.1,
            l1_max: 0.4,
            l2_entries: vec![],  // triggers MissingL2Entries (Warning)
        };
        let report = checker.check_shots(&[shot], None);
        assert_eq!(report.count_by_severity(ViolationSeverity::Warning), 1);
        assert_eq!(report.count_by_severity(ViolationSeverity::Error), 0);
    }
}
