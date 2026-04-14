//! RPU (Reference Processing Unit) metadata sequence validator.
//!
//! Validates structural and semantic integrity of Dolby Vision RPU metadata
//! sequences at three strictness levels: Basic, Strict, and Broadcast.
//!
//! # Validation Rules
//!
//! | Check | Basic | Strict | Broadcast |
//! |-------|-------|--------|-----------|
//! | L1 PQ in [0, 1] | ✓ | ✓ | ✓ |
//! | L1 ordering (max >= mid >= min) | ✓ | ✓ | ✓ |
//! | L2 trim ranges | — | ✓ | ✓ |
//! | Missing L2 entries | — | ✓ | ✓ |
//! | Discontinuous L1 max across shots | — | — | ✓ |
//! | Exceeded peak brightness (4000 nit equiv.) | — | ✓ | ✓ |
//!
//! # Examples
//!
//! ```rust
//! use oximedia_dolbyvision::rpu_validator::{
//!     RpuValidator, RpuValidationLevel,
//! };
//! use oximedia_dolbyvision::dv_xml_export::{DvShotEntry, DvL2Entry};
//!
//! let shots = vec![DvShotEntry {
//!     frame_start: 0,
//!     frame_end: 23,
//!     l1_min: 0.0,
//!     l1_mid: 0.1,
//!     l1_max: 0.58,
//!     l2_entries: vec![DvL2Entry::identity(2081)],
//! }];
//!
//! let validator = RpuValidator::new(RpuValidationLevel::Strict);
//! let result = validator.validate_sequence(&shots);
//! assert_eq!(result.frames_analyzed, 24);
//! assert!(result.pass_rate >= 1.0);
//! ```

use crate::dv_xml_export::{DvL2Entry, DvShotEntry};

// ── Severity ──────────────────────────────────────────────────────────────────

/// Severity level for a single [`RpuIssue`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IssueSeverity {
    /// Informational note — no action required.
    Info,
    /// Warning — metadata is technically valid but may cause display artefacts.
    Warning,
    /// Error — metadata is invalid; broadcast delivery will be rejected.
    Error,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ── Issue type ────────────────────────────────────────────────────────────────

/// Specific category of RPU validation problem.
#[derive(Debug, Clone, PartialEq)]
pub enum RpuIssueType {
    /// An L1 value (min, mid, or max) falls outside the normalised PQ range
    /// [0.0, 1.0].
    InvalidL1Range,
    /// No L2 trim entry was found in a shot that requires one.
    MissingL2,
    /// The L1 max value changed too abruptly between consecutive shots.
    /// The payload is `|delta|` in normalised PQ units.
    DiscontinuousL1Max(f32),
    /// The L1 max corresponds to a luminance above the target peak brightness
    /// allowed by the validation profile.
    ExceededPeakBrightness,
    /// A metadata field encodes an invalid Dolby Vision profile number.
    InvalidProfile,
}

impl std::fmt::Display for RpuIssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidL1Range => write!(f, "InvalidL1Range"),
            Self::MissingL2 => write!(f, "MissingL2"),
            Self::DiscontinuousL1Max(delta) => write!(f, "DiscontinuousL1Max({delta:.4})"),
            Self::ExceededPeakBrightness => write!(f, "ExceededPeakBrightness"),
            Self::InvalidProfile => write!(f, "InvalidProfile"),
        }
    }
}

// ── Issue ─────────────────────────────────────────────────────────────────────

/// A single validation issue attached to a frame offset.
#[derive(Debug, Clone, PartialEq)]
pub struct RpuIssue {
    /// Frame index (0-based, within the full sequence) where the issue occurs.
    pub frame: u64,
    /// Category of the issue.
    pub issue_type: RpuIssueType,
    /// Severity of the issue.
    pub severity: IssueSeverity,
    /// Human-readable description.
    pub description: String,
}

impl RpuIssue {
    fn new(
        frame: u64,
        issue_type: RpuIssueType,
        severity: IssueSeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            frame,
            issue_type,
            severity,
            description: description.into(),
        }
    }
}

// ── Validation level ──────────────────────────────────────────────────────────

/// Controls how strictly the validator applies its rule set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RpuValidationLevel {
    /// Check only structural invariants (L1 range and ordering).
    Basic,
    /// Additionally verify L2 trim ranges and peak brightness limits.
    Strict,
    /// Full broadcast-delivery validation, including scene-to-scene continuity.
    Broadcast,
}

// ── Validation result ─────────────────────────────────────────────────────────

/// Aggregate result of validating an RPU sequence.
#[derive(Debug, Clone)]
pub struct RpuValidationResult {
    /// All issues found across the validated sequence.
    pub issues: Vec<RpuIssue>,
    /// Total number of frames analysed.
    pub frames_analyzed: u64,
    /// Fraction of frames with no errors (warnings do not reduce pass rate).
    /// Range: [0.0, 1.0].
    pub pass_rate: f32,
}

impl RpuValidationResult {
    /// Return `true` if no [`IssueSeverity::Error`] issues were found.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == IssueSeverity::Error)
    }

    /// Return all issues with at least the specified severity.
    #[must_use]
    pub fn issues_at_or_above(&self, severity: IssueSeverity) -> Vec<&RpuIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= severity)
            .collect()
    }
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum allowed L1 max in Strict/Broadcast mode (4000 nit / 10 000 nit = 0.4
/// on a normalised PQ scale).
const MAX_L1_MAX_STRICT: f32 = 0.4;

/// Maximum allowed absolute L1 max delta between consecutive shots in Broadcast
/// mode.  A change larger than this is flagged as a discontinuity.
const BROADCAST_DISCONTINUITY_THRESHOLD: f32 = 0.3;

/// Valid range for L2 trim slope: [0.0, 2.0].
const L2_SLOPE_MIN: f32 = 0.0;
const L2_SLOPE_MAX: f32 = 2.0;

/// Valid range for L2 trim offset: [-1.0, 1.0].
const L2_OFFSET_MIN: f32 = -1.0;
const L2_OFFSET_MAX: f32 = 1.0;

/// Valid range for L2 trim power: [0.0, 2.0].
const L2_POWER_MIN: f32 = 0.0;
const L2_POWER_MAX: f32 = 2.0;

// ── Validator ─────────────────────────────────────────────────────────────────

/// Validates Dolby Vision RPU metadata sequences.
pub struct RpuValidator {
    /// Strictness level applied during validation.
    pub level: RpuValidationLevel,
}

impl RpuValidator {
    /// Create a new validator with the given strictness level.
    #[must_use]
    pub fn new(level: RpuValidationLevel) -> Self {
        Self { level }
    }

    /// Validate a single shot and return any issues found.
    ///
    /// `frame_offset` is the absolute frame number of the shot's first frame
    /// within the full sequence.
    #[must_use]
    pub fn validate_shot(&self, shot: &DvShotEntry, frame_offset: u64) -> Vec<RpuIssue> {
        let mut issues = Vec::new();
        let first_frame = frame_offset + shot.frame_start;

        // ── L1 range checks (all levels) ──────────────────────────────────────
        self.check_l1_range(shot, first_frame, &mut issues);

        // ── L1 ordering: max >= mid >= min (all levels) ───────────────────────
        self.check_l1_ordering(shot, first_frame, &mut issues);

        // ── Strict and Broadcast only ─────────────────────────────────────────
        if self.level >= RpuValidationLevel::Strict {
            self.check_peak_brightness(shot, first_frame, &mut issues);
            self.check_l2_presence(shot, first_frame, &mut issues);
            self.check_l2_ranges(shot, first_frame, &mut issues);
        }

        issues
    }

    /// Validate an entire sequence of shots and return aggregate results.
    ///
    /// Shots are assumed to be in presentation order.  Broadcast-level checks
    /// also compare consecutive shots for continuity.
    #[must_use]
    pub fn validate_sequence(&self, shots: &[DvShotEntry]) -> RpuValidationResult {
        let mut all_issues: Vec<RpuIssue> = Vec::new();
        let mut total_frames: u64 = 0;
        let mut frames_with_errors: u64 = 0;

        for shot in shots {
            let frame_offset = 0; // frame_start already encodes absolute position
            let issues = self.validate_shot(shot, frame_offset);

            let duration = shot.duration();
            total_frames += duration;

            // Count frames with errors
            let has_error = issues.iter().any(|i| i.severity == IssueSeverity::Error);
            if has_error {
                frames_with_errors += duration;
            }

            all_issues.extend(issues);
        }

        // ── Broadcast: cross-shot continuity ──────────────────────────────────
        if self.level >= RpuValidationLevel::Broadcast {
            let mut continuity_issues =
                self.check_sequence_continuity(shots, &mut frames_with_errors, total_frames);
            all_issues.append(&mut continuity_issues);
        }

        let pass_rate = if total_frames == 0 {
            1.0
        } else {
            let passing = total_frames.saturating_sub(frames_with_errors);
            passing as f32 / total_frames as f32
        };

        RpuValidationResult {
            issues: all_issues,
            frames_analyzed: total_frames,
            pass_rate,
        }
    }

    // ── Private checkers ──────────────────────────────────────────────────────

    fn check_l1_range(&self, shot: &DvShotEntry, frame: u64, issues: &mut Vec<RpuIssue>) {
        let check = |val: f32, name: &str| -> Option<RpuIssue> {
            if val < 0.0 || val > 1.0 {
                Some(RpuIssue::new(
                    frame,
                    RpuIssueType::InvalidL1Range,
                    IssueSeverity::Error,
                    format!(
                        "L1 {name} value {val:.6} is outside normalised PQ range [0.0, 1.0]"
                    ),
                ))
            } else {
                None
            }
        };

        if let Some(issue) = check(shot.l1_min, "min") {
            issues.push(issue);
        }
        if let Some(issue) = check(shot.l1_mid, "mid") {
            issues.push(issue);
        }
        if let Some(issue) = check(shot.l1_max, "max") {
            issues.push(issue);
        }
    }

    fn check_l1_ordering(&self, shot: &DvShotEntry, frame: u64, issues: &mut Vec<RpuIssue>) {
        if shot.l1_max < shot.l1_mid {
            issues.push(RpuIssue::new(
                frame,
                RpuIssueType::InvalidL1Range,
                IssueSeverity::Error,
                format!(
                    "L1 max ({:.6}) must be >= L1 mid ({:.6})",
                    shot.l1_max, shot.l1_mid
                ),
            ));
        }
        if shot.l1_mid < shot.l1_min {
            issues.push(RpuIssue::new(
                frame,
                RpuIssueType::InvalidL1Range,
                IssueSeverity::Error,
                format!(
                    "L1 mid ({:.6}) must be >= L1 min ({:.6})",
                    shot.l1_mid, shot.l1_min
                ),
            ));
        }
    }

    fn check_peak_brightness(&self, shot: &DvShotEntry, frame: u64, issues: &mut Vec<RpuIssue>) {
        if shot.l1_max > MAX_L1_MAX_STRICT {
            issues.push(RpuIssue::new(
                frame,
                RpuIssueType::ExceededPeakBrightness,
                IssueSeverity::Warning,
                format!(
                    "L1 max ({:.4}) exceeds 4000-nit strict limit ({MAX_L1_MAX_STRICT:.4})",
                    shot.l1_max
                ),
            ));
        }
    }

    fn check_l2_presence(&self, shot: &DvShotEntry, frame: u64, issues: &mut Vec<RpuIssue>) {
        if shot.l2_entries.is_empty() {
            issues.push(RpuIssue::new(
                frame,
                RpuIssueType::MissingL2,
                IssueSeverity::Warning,
                "shot has no L2 trim entries; at least one target display is expected".to_string(),
            ));
        }
    }

    fn check_l2_ranges(&self, shot: &DvShotEntry, frame: u64, issues: &mut Vec<RpuIssue>) {
        for l2 in &shot.l2_entries {
            check_l2_entry_ranges(l2, frame, issues);
        }
    }

    fn check_sequence_continuity(
        &self,
        shots: &[DvShotEntry],
        frames_with_errors: &mut u64,
        _total_frames: u64,
    ) -> Vec<RpuIssue> {
        let mut issues = Vec::new();

        for window in shots.windows(2) {
            let prev = &window[0];
            let curr = &window[1];
            let delta = (curr.l1_max - prev.l1_max).abs();

            if delta > BROADCAST_DISCONTINUITY_THRESHOLD {
                let frame = curr.frame_start;
                issues.push(RpuIssue::new(
                    frame,
                    RpuIssueType::DiscontinuousL1Max(delta),
                    IssueSeverity::Warning,
                    format!(
                        "L1 max changed abruptly by {delta:.4} between shots ending at frame {} \
                         and starting at frame {} (threshold {BROADCAST_DISCONTINUITY_THRESHOLD:.4})",
                        prev.frame_end, curr.frame_start
                    ),
                ));
                // Count the current shot's frames as warned-about (not errors)
                // — we do not increment frames_with_errors for warnings.
                let _ = frames_with_errors; // suppress unused warning
            }
        }

        issues
    }
}

// ── L2 range helper ───────────────────────────────────────────────────────────

fn check_l2_entry_ranges(l2: &DvL2Entry, frame: u64, issues: &mut Vec<RpuIssue>) {
    if !(L2_SLOPE_MIN..=L2_SLOPE_MAX).contains(&l2.trim_slope) {
        issues.push(RpuIssue::new(
            frame,
            RpuIssueType::InvalidL1Range, // re-use type for trim ranges
            IssueSeverity::Error,
            format!(
                "L2 trim_slope {:.4} outside [{L2_SLOPE_MIN}, {L2_SLOPE_MAX}] for target_max_pq={}",
                l2.trim_slope, l2.target_max_pq
            ),
        ));
    }
    if !(L2_OFFSET_MIN..=L2_OFFSET_MAX).contains(&l2.trim_offset) {
        issues.push(RpuIssue::new(
            frame,
            RpuIssueType::InvalidL1Range,
            IssueSeverity::Error,
            format!(
                "L2 trim_offset {:.4} outside [{L2_OFFSET_MIN}, {L2_OFFSET_MAX}] for target_max_pq={}",
                l2.trim_offset, l2.target_max_pq
            ),
        ));
    }
    if !(L2_POWER_MIN..=L2_POWER_MAX).contains(&l2.trim_power) {
        issues.push(RpuIssue::new(
            frame,
            RpuIssueType::InvalidL1Range,
            IssueSeverity::Error,
            format!(
                "L2 trim_power {:.4} outside [{L2_POWER_MIN}, {L2_POWER_MAX}] for target_max_pq={}",
                l2.trim_power, l2.target_max_pq
            ),
        ));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dv_xml_export::{DvL2Entry, DvShotEntry};

    fn valid_shot(start: u64, end: u64) -> DvShotEntry {
        DvShotEntry {
            frame_start: start,
            frame_end: end,
            l1_min: 0.001,
            l1_mid: 0.10,
            l1_max: 0.35,
            l2_entries: vec![DvL2Entry::identity(2081)],
        }
    }

    // -- Valid shot passes ------------------------------------------------------

    #[test]
    fn test_valid_shot_no_issues_basic() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let shot = valid_shot(0, 23);
        let issues = validator.validate_shot(&shot, 0);
        assert!(issues.is_empty(), "expected no issues, got: {issues:?}");
    }

    #[test]
    fn test_valid_shot_no_errors_strict() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        let shot = valid_shot(0, 23);
        let issues = validator.validate_shot(&shot, 0);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
    }

    // -- L1 range violations ----------------------------------------------------

    #[test]
    fn test_l1_max_above_one() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let mut shot = valid_shot(0, 23);
        shot.l1_max = 1.5;
        let issues = validator.validate_shot(&shot, 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == RpuIssueType::InvalidL1Range),
            "expected InvalidL1Range error"
        );
    }

    #[test]
    fn test_l1_min_below_zero() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let mut shot = valid_shot(0, 23);
        shot.l1_min = -0.01;
        let issues = validator.validate_shot(&shot, 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == RpuIssueType::InvalidL1Range),
            "expected InvalidL1Range error"
        );
    }

    #[test]
    fn test_l1_ordering_max_less_than_mid() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let mut shot = valid_shot(0, 23);
        shot.l1_max = 0.05;
        shot.l1_mid = 0.10; // mid > max: invalid
        let issues = validator.validate_shot(&shot, 0);
        let ordering_errors: Vec<_> = issues
            .iter()
            .filter(|i| {
                i.severity == IssueSeverity::Error
                    && i.description.contains("max")
                    && i.description.contains("mid")
            })
            .collect();
        assert!(
            !ordering_errors.is_empty(),
            "expected ordering error: {issues:?}"
        );
    }

    // -- L2 checks (Strict) -----------------------------------------------------

    #[test]
    fn test_missing_l2_strict() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        let mut shot = valid_shot(0, 23);
        shot.l2_entries.clear();
        let issues = validator.validate_shot(&shot, 0);
        assert!(
            issues
                .iter()
                .any(|i| i.issue_type == RpuIssueType::MissingL2),
            "expected MissingL2 issue"
        );
    }

    #[test]
    fn test_l2_trim_slope_out_of_range() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        let mut shot = valid_shot(0, 23);
        shot.l2_entries[0].trim_slope = 3.0;
        let issues = validator.validate_shot(&shot, 0);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error && i.description.contains("trim_slope"))
            .collect();
        assert!(!errors.is_empty(), "expected trim_slope error: {issues:?}");
    }

    #[test]
    fn test_l2_trim_offset_out_of_range() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        let mut shot = valid_shot(0, 23);
        shot.l2_entries[0].trim_offset = -2.0;
        let issues = validator.validate_shot(&shot, 0);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| {
                i.severity == IssueSeverity::Error && i.description.contains("trim_offset")
            })
            .collect();
        assert!(
            !errors.is_empty(),
            "expected trim_offset error: {issues:?}"
        );
    }

    // -- Sequence validation / pass rate ----------------------------------------

    #[test]
    fn test_pass_rate_all_valid() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        let shots = vec![valid_shot(0, 23), valid_shot(24, 47)];
        let result = validator.validate_sequence(&shots);
        assert!(
            (result.pass_rate - 1.0).abs() < 1e-6,
            "pass_rate={}",
            result.pass_rate
        );
    }

    #[test]
    fn test_pass_rate_one_bad_shot() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let mut bad = valid_shot(0, 23);
        bad.l1_max = 1.5; // Error
        let shots = vec![bad, valid_shot(24, 47)];
        let result = validator.validate_sequence(&shots);
        // First shot (24 frames) has errors, second (24 frames) is fine
        // pass_rate should be 24/48 = 0.5
        assert!(
            (result.pass_rate - 0.5).abs() < 1e-5,
            "pass_rate={}",
            result.pass_rate
        );
    }

    #[test]
    fn test_frames_analyzed_count() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let shots = vec![valid_shot(0, 23), valid_shot(24, 71)];
        let result = validator.validate_sequence(&shots);
        // First shot: 24 frames (0..=23), second: 48 frames (24..=71)
        assert_eq!(result.frames_analyzed, 72);
    }

    // -- Broadcast: discontinuity detection -------------------------------------

    #[test]
    fn test_discontinuity_detected() {
        let validator = RpuValidator::new(RpuValidationLevel::Broadcast);
        let mut shot1 = valid_shot(0, 23);
        let mut shot2 = valid_shot(24, 47);
        shot1.l1_max = 0.05;
        shot2.l1_max = 0.90; // delta = 0.85 >> threshold 0.3
        let shots = vec![shot1, shot2];
        let result = validator.validate_sequence(&shots);
        let disc: Vec<_> = result
            .issues
            .iter()
            .filter(|i| matches!(i.issue_type, RpuIssueType::DiscontinuousL1Max(_)))
            .collect();
        assert!(!disc.is_empty(), "expected discontinuity issue");
    }

    #[test]
    fn test_no_discontinuity_within_threshold() {
        let validator = RpuValidator::new(RpuValidationLevel::Broadcast);
        let mut shot1 = valid_shot(0, 23);
        let mut shot2 = valid_shot(24, 47);
        shot1.l1_max = 0.30;
        shot2.l1_max = 0.35; // delta = 0.05 << threshold
        let shots = vec![shot1, shot2];
        let result = validator.validate_sequence(&shots);
        let disc: Vec<_> = result
            .issues
            .iter()
            .filter(|i| matches!(i.issue_type, RpuIssueType::DiscontinuousL1Max(_)))
            .collect();
        assert!(disc.is_empty(), "unexpected discontinuity issues: {disc:?}");
    }

    #[test]
    fn test_is_valid_with_only_warnings() {
        let validator = RpuValidator::new(RpuValidationLevel::Strict);
        // Missing L2 → Warning, no Errors
        let mut shot = valid_shot(0, 23);
        shot.l2_entries.clear();
        let shots = vec![shot];
        let result = validator.validate_sequence(&shots);
        assert!(result.is_valid(), "warnings-only result should be valid");
    }

    #[test]
    fn test_is_invalid_with_errors() {
        let validator = RpuValidator::new(RpuValidationLevel::Basic);
        let mut shot = valid_shot(0, 23);
        shot.l1_max = 2.0;
        let shots = vec![shot];
        let result = validator.validate_sequence(&shots);
        assert!(!result.is_valid(), "should be invalid due to L1 range error");
    }
}
