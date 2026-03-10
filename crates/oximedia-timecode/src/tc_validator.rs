//! Timecode validation rules and violation reporting.
//!
//! This module provides structured validation of SMPTE timecodes against a
//! configurable set of rules, producing typed violation reports rather than
//! bare errors so callers can decide how to handle each issue.

#![allow(dead_code)]

use crate::{FrameRateInfo, Timecode};

// ── Validation rules ──────────────────────────────────────────────────────────

/// A single validation rule that can be applied to a timecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationRule {
    /// Hours must be in 0–23.
    HoursInRange,
    /// Minutes must be in 0–59.
    MinutesInRange,
    /// Seconds must be in 0–59.
    SecondsInRange,
    /// Frame count must be in 0–(fps-1).
    FramesInRange,
    /// Frames 0 and 1 are illegal at the start of non-tenth minutes (DF only).
    DropFramePositions,
    /// Timecode must lie within an explicit allowed range \[start, end\].
    WithinRange,
}

impl std::fmt::Display for ValidationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HoursInRange => write!(f, "hours-in-range"),
            Self::MinutesInRange => write!(f, "minutes-in-range"),
            Self::SecondsInRange => write!(f, "seconds-in-range"),
            Self::FramesInRange => write!(f, "frames-in-range"),
            Self::DropFramePositions => write!(f, "drop-frame-positions"),
            Self::WithinRange => write!(f, "within-range"),
        }
    }
}

// ── Violations ────────────────────────────────────────────────────────────────

/// A validation violation: which rule failed and a human-readable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TcViolation {
    /// The rule that was violated.
    pub rule: ValidationRule,
    /// A description of the problem.
    pub message: String,
}

impl TcViolation {
    /// Create a new violation.
    pub fn new(rule: ValidationRule, message: impl Into<String>) -> Self {
        Self {
            rule,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TcViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.rule, self.message)
    }
}

// ── Validator ─────────────────────────────────────────────────────────────────

/// Configuration for `TimecodeValidator`.
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// Rules to check. Defaults to all rules except `WithinRange`.
    pub rules: Vec<ValidationRule>,
    /// Optional allowed range `[start_frames, end_frames]` (inclusive).
    /// Only checked when `ValidationRule::WithinRange` is enabled.
    pub allowed_range: Option<(u64, u64)>,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            rules: vec![
                ValidationRule::HoursInRange,
                ValidationRule::MinutesInRange,
                ValidationRule::SecondsInRange,
                ValidationRule::FramesInRange,
                ValidationRule::DropFramePositions,
            ],
            allowed_range: None,
        }
    }
}

/// Validates timecodes against a configurable set of rules.
///
/// # Example
/// ```
/// use oximedia_timecode::{Timecode, FrameRate};
/// use oximedia_timecode::tc_validator::{TimecodeValidator, ValidatorConfig};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let validator = TimecodeValidator::new(ValidatorConfig::default());
/// let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25)?;
/// assert!(validator.validate(&tc).is_empty());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TimecodeValidator {
    config: ValidatorConfig,
}

impl TimecodeValidator {
    /// Create a new validator with the given configuration.
    pub fn new(config: ValidatorConfig) -> Self {
        Self { config }
    }

    /// Create a validator with default rules.
    pub fn default_validator() -> Self {
        Self::new(ValidatorConfig::default())
    }

    /// Validate a `Timecode` and return all violations found.
    /// An empty `Vec` means the timecode is valid under the configured rules.
    pub fn validate(&self, tc: &Timecode) -> Vec<TcViolation> {
        let mut violations = Vec::new();
        for &rule in &self.config.rules {
            match rule {
                ValidationRule::HoursInRange => {
                    if tc.hours > 23 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("hours {} exceeds maximum of 23", tc.hours),
                        ));
                    }
                }
                ValidationRule::MinutesInRange => {
                    if tc.minutes > 59 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("minutes {} exceeds maximum of 59", tc.minutes),
                        ));
                    }
                }
                ValidationRule::SecondsInRange => {
                    if tc.seconds > 59 {
                        violations.push(TcViolation::new(
                            rule,
                            format!("seconds {} exceeds maximum of 59", tc.seconds),
                        ));
                    }
                }
                ValidationRule::FramesInRange => {
                    if tc.frames >= tc.frame_rate.fps {
                        violations.push(TcViolation::new(
                            rule,
                            format!("frames {} >= fps {}", tc.frames, tc.frame_rate.fps),
                        ));
                    }
                }
                ValidationRule::DropFramePositions => {
                    if tc.frame_rate.drop_frame
                        && tc.seconds == 0
                        && tc.frames < 2
                        && !tc.minutes.is_multiple_of(10)
                    {
                        violations.push(TcViolation::new(
                            rule,
                            format!(
                                "frames {f} at {m}:00 is an illegal drop-frame position",
                                f = tc.frames,
                                m = tc.minutes,
                            ),
                        ));
                    }
                }
                ValidationRule::WithinRange => {
                    if let Some((start, end)) = self.config.allowed_range {
                        let pos = tc.to_frames();
                        if pos < start || pos > end {
                            violations.push(TcViolation::new(
                                rule,
                                format!(
                                    "frame position {pos} is outside allowed range [{start}, {end}]"
                                ),
                            ));
                        }
                    }
                }
            }
        }
        violations
    }

    /// Validate a range of consecutive timecodes for continuity.
    /// Returns violations for any timecode in the slice that fails validation.
    pub fn validate_range(&self, timecodes: &[Timecode]) -> Vec<(usize, TcViolation)> {
        let mut out = Vec::new();
        for (i, tc) in timecodes.iter().enumerate() {
            for v in self.validate(tc) {
                out.push((i, v));
            }
        }
        out
    }

    /// Return `true` if the timecode passes all configured rules.
    pub fn is_valid(&self, tc: &Timecode) -> bool {
        self.validate(tc).is_empty()
    }
}

// ── Helper: build a raw Timecode bypassing constructor checks ─────────────────

/// Build a `Timecode` directly without the safe constructor (for tests).
fn raw_timecode(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u8, drop: bool) -> Timecode {
    Timecode {
        hours,
        minutes,
        seconds,
        frames,
        frame_rate: FrameRateInfo {
            fps,
            drop_frame: drop,
        },
        user_bits: 0,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn valid_25fps() -> Timecode {
        Timecode::new(1, 30, 0, 12, FrameRate::Fps25).expect("valid timecode")
    }

    #[test]
    fn test_valid_timecode_no_violations() {
        let v = TimecodeValidator::default_validator();
        assert!(v.validate(&valid_25fps()).is_empty());
    }

    #[test]
    fn test_is_valid_returns_true_for_good_tc() {
        let v = TimecodeValidator::default_validator();
        assert!(v.is_valid(&valid_25fps()));
    }

    #[test]
    fn test_hours_out_of_range() {
        let tc = raw_timecode(24, 0, 0, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::HoursInRange));
    }

    #[test]
    fn test_minutes_out_of_range() {
        let tc = raw_timecode(0, 60, 0, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::MinutesInRange));
    }

    #[test]
    fn test_seconds_out_of_range() {
        let tc = raw_timecode(0, 0, 60, 0, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::SecondsInRange));
    }

    #[test]
    fn test_frames_out_of_range() {
        let tc = raw_timecode(0, 0, 0, 25, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::FramesInRange));
    }

    #[test]
    fn test_drop_frame_illegal_position_detected() {
        // Frames 0 at minute 1, second 0 — illegal in 29.97 DF
        let tc = raw_timecode(0, 1, 0, 0, 30, true);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(vios
            .iter()
            .any(|x| x.rule == ValidationRule::DropFramePositions));
    }

    #[test]
    fn test_drop_frame_tenth_minute_is_ok() {
        // Minute 10 is a "keep" minute for DF — frames 0 is legal
        let tc = raw_timecode(0, 10, 0, 0, 30, true);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        assert!(!vios
            .iter()
            .any(|x| x.rule == ValidationRule::DropFramePositions));
    }

    #[test]
    fn test_within_range_pass() {
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode"); // 25 frames
        let cfg = ValidatorConfig {
            rules: vec![ValidationRule::WithinRange],
            allowed_range: Some((0, 100)),
        };
        let v = TimecodeValidator::new(cfg);
        assert!(v.validate(&tc).is_empty());
    }

    #[test]
    fn test_within_range_fail() {
        let tc = Timecode::new(0, 0, 10, 0, FrameRate::Fps25).expect("valid timecode"); // 250 frames
        let cfg = ValidatorConfig {
            rules: vec![ValidationRule::WithinRange],
            allowed_range: Some((0, 100)),
        };
        let v = TimecodeValidator::new(cfg);
        let vios = v.validate(&tc);
        assert!(vios.iter().any(|x| x.rule == ValidationRule::WithinRange));
    }

    #[test]
    fn test_validate_range_empty_slice() {
        let v = TimecodeValidator::default_validator();
        assert!(v.validate_range(&[]).is_empty());
    }

    #[test]
    fn test_validate_range_all_valid() {
        let tcs: Vec<Timecode> = (0u8..5)
            .map(|f| Timecode::new(0, 0, 0, f, FrameRate::Fps25).expect("valid timecode"))
            .collect();
        let v = TimecodeValidator::default_validator();
        assert!(v.validate_range(&tcs).is_empty());
    }

    #[test]
    fn test_validate_range_with_violation() {
        let good = Timecode::new(0, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let bad = raw_timecode(0, 0, 0, 25, 25, false); // frames == fps
        let v = TimecodeValidator::default_validator();
        let results = v.validate_range(&[good, bad]);
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 1); // index of bad timecode
    }

    #[test]
    fn test_rule_display() {
        assert_eq!(ValidationRule::HoursInRange.to_string(), "hours-in-range");
        assert_eq!(
            ValidationRule::DropFramePositions.to_string(),
            "drop-frame-positions"
        );
        assert_eq!(ValidationRule::WithinRange.to_string(), "within-range");
    }

    #[test]
    fn test_violation_display() {
        let v = TcViolation::new(ValidationRule::FramesInRange, "frames 30 >= fps 30");
        let s = v.to_string();
        assert!(s.contains("frames-in-range"));
        assert!(s.contains("frames 30"));
    }

    #[test]
    fn test_no_rules_produces_no_violations() {
        let tc = raw_timecode(99, 99, 99, 99, 25, false); // everything out of range
        let cfg = ValidatorConfig {
            rules: vec![],
            allowed_range: None,
        };
        let v = TimecodeValidator::new(cfg);
        assert!(v.validate(&tc).is_empty());
    }

    #[test]
    fn test_multiple_violations_accumulate() {
        let tc = raw_timecode(24, 60, 60, 25, 25, false);
        let v = TimecodeValidator::default_validator();
        let vios = v.validate(&tc);
        // Should find at least hours, minutes, seconds, frames violations
        assert!(vios.len() >= 4);
    }
}
