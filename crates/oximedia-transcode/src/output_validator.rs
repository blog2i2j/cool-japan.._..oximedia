//! Transcode output validation — spec-comparison based.
//!
//! This module provides a structured way to validate that a completed transcode
//! operation produced output that matches a defined specification.  It differs
//! from [`crate::output_verify`] which applies constraint predicates: here we
//! compare a set of *expected* properties (codec, resolution, bitrate, duration,
//! frame rate, container format) against *actual* measured properties, with
//! configurable per-field tolerances.
//!
//! # Example
//!
//! ```rust
//! use oximedia_transcode::output_validator::{
//!     OutputSpec, ActualOutputProperties, OutputValidator, ValidationProfile,
//! };
//!
//! let spec = OutputSpec::builder()
//!     .video_codec("vp9")
//!     .resolution(1920, 1080)
//!     .video_bitrate_bps(5_000_000)
//!     .build();
//!
//! let actual = ActualOutputProperties {
//!     video_codec: Some("vp9".to_string()),
//!     audio_codec: None,
//!     width: Some(1920),
//!     height: Some(1080),
//!     video_bitrate_bps: Some(4_950_000),
//!     audio_bitrate_bps: None,
//!     duration_secs: None,
//!     frame_rate_num: None,
//!     frame_rate_den: None,
//!     container_format: None,
//! };
//!
//! let validator = OutputValidator::new(ValidationProfile::streaming());
//! let report = validator.validate(&spec, &actual);
//! assert!(report.passed());
//! ```

#![allow(clippy::cast_precision_loss)]

use std::fmt;

use crate::{Result, TranscodeError};

// ─────────────────────────────────────────────────────────────────────────────
// Tolerances
// ─────────────────────────────────────────────────────────────────────────────

/// Per-field tolerance configuration used during validation.
///
/// A field passes if the actual value is within the specified tolerance of the
/// expected value.
#[derive(Debug, Clone)]
pub struct ValidationTolerances {
    /// Allowed relative deviation for bitrate checks (e.g. 0.10 = ±10 %).
    pub bitrate_relative: f64,
    /// Allowed absolute deviation for duration checks in seconds.
    pub duration_secs: f64,
    /// Allowed absolute deviation for frame rate in frames per second.
    pub frame_rate_fps: f64,
}

impl ValidationTolerances {
    /// Strict tolerances for archival/broadcast deliverables.
    #[must_use]
    pub fn strict() -> Self {
        Self {
            bitrate_relative: 0.02,
            duration_secs: 0.1,
            frame_rate_fps: 0.001,
        }
    }

    /// Moderate tolerances for streaming platforms.
    #[must_use]
    pub fn moderate() -> Self {
        Self {
            bitrate_relative: 0.10,
            duration_secs: 0.5,
            frame_rate_fps: 0.01,
        }
    }

    /// Relaxed tolerances for preview / proxy workflows.
    #[must_use]
    pub fn relaxed() -> Self {
        Self {
            bitrate_relative: 0.20,
            duration_secs: 2.0,
            frame_rate_fps: 0.1,
        }
    }
}

impl Default for ValidationTolerances {
    fn default() -> Self {
        Self::moderate()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationProfile
// ─────────────────────────────────────────────────────────────────────────────

/// A named validation profile combining tolerances and which fields are
/// considered *required* (vs. optional) for a passing result.
#[derive(Debug, Clone)]
pub struct ValidationProfile {
    /// Human-readable name of this profile.
    pub name: String,
    /// Tolerance values for numeric fields.
    pub tolerances: ValidationTolerances,
    /// Whether codec mismatches automatically fail the validation.
    pub require_codec_match: bool,
    /// Whether resolution must match exactly (no tolerance).
    pub require_exact_resolution: bool,
    /// Whether bitrate fields are checked at all.
    pub check_bitrate: bool,
    /// Whether duration is checked.
    pub check_duration: bool,
    /// Whether frame rate is checked.
    pub check_frame_rate: bool,
    /// Whether container format is checked.
    pub check_container: bool,
}

impl ValidationProfile {
    /// Profile for streaming platform delivery (e.g. YouTube, Vimeo).
    #[must_use]
    pub fn streaming() -> Self {
        Self {
            name: "streaming".to_string(),
            tolerances: ValidationTolerances::moderate(),
            require_codec_match: true,
            require_exact_resolution: true,
            check_bitrate: true,
            check_duration: true,
            check_frame_rate: true,
            check_container: false,
        }
    }

    /// Profile for broadcast deliverables (strict).
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            name: "broadcast".to_string(),
            tolerances: ValidationTolerances::strict(),
            require_codec_match: true,
            require_exact_resolution: true,
            check_bitrate: true,
            check_duration: true,
            check_frame_rate: true,
            check_container: true,
        }
    }

    /// Profile for preview / proxy workflows (relaxed).
    #[must_use]
    pub fn preview() -> Self {
        Self {
            name: "preview".to_string(),
            tolerances: ValidationTolerances::relaxed(),
            require_codec_match: false,
            require_exact_resolution: false,
            check_bitrate: false,
            check_duration: true,
            check_frame_rate: false,
            check_container: false,
        }
    }
}

impl Default for ValidationProfile {
    fn default() -> Self {
        Self::streaming()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OutputSpec
// ─────────────────────────────────────────────────────────────────────────────

/// The *expected* specification for a transcode output.
///
/// All fields are optional; only fields that are `Some` are validated.
#[derive(Debug, Clone, Default)]
pub struct OutputSpec {
    /// Expected video codec identifier (e.g. `"vp9"`, `"av1"`).
    pub video_codec: Option<String>,
    /// Expected audio codec identifier (e.g. `"opus"`, `"flac"`).
    pub audio_codec: Option<String>,
    /// Expected video width in pixels.
    pub width: Option<u32>,
    /// Expected video height in pixels.
    pub height: Option<u32>,
    /// Expected video bitrate in bits per second.
    pub video_bitrate_bps: Option<u64>,
    /// Expected audio bitrate in bits per second.
    pub audio_bitrate_bps: Option<u64>,
    /// Expected duration in seconds.
    pub duration_secs: Option<f64>,
    /// Expected frame rate numerator.
    pub frame_rate_num: Option<u32>,
    /// Expected frame rate denominator.
    pub frame_rate_den: Option<u32>,
    /// Expected container format identifier (e.g. `"mp4"`, `"webm"`, `"mkv"`).
    pub container_format: Option<String>,
}

impl OutputSpec {
    /// Creates a new builder for constructing an `OutputSpec`.
    #[must_use]
    pub fn builder() -> OutputSpecBuilder {
        OutputSpecBuilder::default()
    }
}

/// Builder for [`OutputSpec`].
#[derive(Debug, Default)]
pub struct OutputSpecBuilder {
    spec: OutputSpec,
}

impl OutputSpecBuilder {
    /// Sets the expected video codec.
    #[must_use]
    pub fn video_codec(mut self, codec: impl Into<String>) -> Self {
        self.spec.video_codec = Some(codec.into());
        self
    }

    /// Sets the expected audio codec.
    #[must_use]
    pub fn audio_codec(mut self, codec: impl Into<String>) -> Self {
        self.spec.audio_codec = Some(codec.into());
        self
    }

    /// Sets the expected output resolution.
    #[must_use]
    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.spec.width = Some(width);
        self.spec.height = Some(height);
        self
    }

    /// Sets the expected video bitrate.
    #[must_use]
    pub fn video_bitrate_bps(mut self, bps: u64) -> Self {
        self.spec.video_bitrate_bps = Some(bps);
        self
    }

    /// Sets the expected audio bitrate.
    #[must_use]
    pub fn audio_bitrate_bps(mut self, bps: u64) -> Self {
        self.spec.audio_bitrate_bps = Some(bps);
        self
    }

    /// Sets the expected duration.
    #[must_use]
    pub fn duration_secs(mut self, secs: f64) -> Self {
        self.spec.duration_secs = Some(secs);
        self
    }

    /// Sets the expected frame rate as a rational number.
    #[must_use]
    pub fn frame_rate(mut self, num: u32, den: u32) -> Self {
        self.spec.frame_rate_num = Some(num);
        self.spec.frame_rate_den = Some(den);
        self
    }

    /// Sets the expected container format.
    #[must_use]
    pub fn container(mut self, fmt: impl Into<String>) -> Self {
        self.spec.container_format = Some(fmt.into());
        self
    }

    /// Builds the [`OutputSpec`].
    #[must_use]
    pub fn build(self) -> OutputSpec {
        self.spec
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ActualOutputProperties
// ─────────────────────────────────────────────────────────────────────────────

/// Measured properties of an actual transcode output.
///
/// All fields are optional; `None` means "not measured / not available".
#[derive(Debug, Clone, Default)]
pub struct ActualOutputProperties {
    /// Detected video codec.
    pub video_codec: Option<String>,
    /// Detected audio codec.
    pub audio_codec: Option<String>,
    /// Detected video width in pixels.
    pub width: Option<u32>,
    /// Detected video height in pixels.
    pub height: Option<u32>,
    /// Measured video bitrate in bits per second.
    pub video_bitrate_bps: Option<u64>,
    /// Measured audio bitrate in bits per second.
    pub audio_bitrate_bps: Option<u64>,
    /// Measured duration in seconds.
    pub duration_secs: Option<f64>,
    /// Detected frame rate numerator.
    pub frame_rate_num: Option<u32>,
    /// Detected frame rate denominator.
    pub frame_rate_den: Option<u32>,
    /// Detected container format.
    pub container_format: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// FieldResult
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome for a single validated field.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldResult {
    /// Field was not checked (not present in spec or disabled by profile).
    Skipped,
    /// Field passed validation.
    Pass,
    /// Field failed — description includes expected vs. actual values.
    Fail(String),
    /// Field was expected but the actual value was not available.
    Missing(String),
}

impl FieldResult {
    /// Returns `true` if the result is not a failure.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Pass | Self::Skipped)
    }
}

impl fmt::Display for FieldResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Skipped => write!(f, "SKIPPED"),
            Self::Pass => write!(f, "PASS"),
            Self::Fail(msg) => write!(f, "FAIL: {msg}"),
            Self::Missing(field) => write!(f, "MISSING: {field} not measured"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ValidationReport
// ─────────────────────────────────────────────────────────────────────────────

/// Per-field validation results for a single output.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Result for the video codec field.
    pub video_codec: FieldResult,
    /// Result for the audio codec field.
    pub audio_codec: FieldResult,
    /// Result for the video width field.
    pub width: FieldResult,
    /// Result for the video height field.
    pub height: FieldResult,
    /// Result for the video bitrate field.
    pub video_bitrate: FieldResult,
    /// Result for the audio bitrate field.
    pub audio_bitrate: FieldResult,
    /// Result for the duration field.
    pub duration: FieldResult,
    /// Result for the frame rate field.
    pub frame_rate: FieldResult,
    /// Result for the container format field.
    pub container: FieldResult,
}

impl ValidationReport {
    /// Returns `true` if every checked field passed.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.video_codec.is_ok()
            && self.audio_codec.is_ok()
            && self.width.is_ok()
            && self.height.is_ok()
            && self.video_bitrate.is_ok()
            && self.audio_bitrate.is_ok()
            && self.duration.is_ok()
            && self.frame_rate.is_ok()
            && self.container.is_ok()
    }

    /// Collects all failing field results into a `Vec`.
    #[must_use]
    pub fn failures(&self) -> Vec<(&'static str, &FieldResult)> {
        let fields: [(&'static str, &FieldResult); 9] = [
            ("video_codec", &self.video_codec),
            ("audio_codec", &self.audio_codec),
            ("width", &self.width),
            ("height", &self.height),
            ("video_bitrate", &self.video_bitrate),
            ("audio_bitrate", &self.audio_bitrate),
            ("duration", &self.duration),
            ("frame_rate", &self.frame_rate),
            ("container", &self.container),
        ];
        fields
            .into_iter()
            .filter(|(_, r)| !r.is_ok())
            .collect()
    }

    /// Returns the number of failing fields.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures().len()
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ValidationReport {{")?;
        writeln!(f, "  video_codec:  {}", self.video_codec)?;
        writeln!(f, "  audio_codec:  {}", self.audio_codec)?;
        writeln!(f, "  width:        {}", self.width)?;
        writeln!(f, "  height:       {}", self.height)?;
        writeln!(f, "  video_bitrate:{}", self.video_bitrate)?;
        writeln!(f, "  audio_bitrate:{}", self.audio_bitrate)?;
        writeln!(f, "  duration:     {}", self.duration)?;
        writeln!(f, "  frame_rate:   {}", self.frame_rate)?;
        writeln!(f, "  container:    {}", self.container)?;
        write!(
            f,
            "  OVERALL: {}",
            if self.passed() { "PASS" } else { "FAIL" }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OutputValidator
// ─────────────────────────────────────────────────────────────────────────────

/// Validates transcode output properties against a specification.
///
/// Instantiate with a [`ValidationProfile`] and call [`validate`] for each
/// output to obtain a [`ValidationReport`].
///
/// [`validate`]: OutputValidator::validate
#[derive(Debug, Clone)]
pub struct OutputValidator {
    profile: ValidationProfile,
}

impl OutputValidator {
    /// Creates a new validator with the given profile.
    #[must_use]
    pub fn new(profile: ValidationProfile) -> Self {
        Self { profile }
    }

    /// Validates `actual` against `spec` and returns a detailed report.
    #[must_use]
    pub fn validate(
        &self,
        spec: &OutputSpec,
        actual: &ActualOutputProperties,
    ) -> ValidationReport {
        let tol = &self.profile.tolerances;

        ValidationReport {
            video_codec: self.check_codec(
                "video_codec",
                spec.video_codec.as_deref(),
                actual.video_codec.as_deref(),
                self.profile.require_codec_match,
            ),
            audio_codec: self.check_codec(
                "audio_codec",
                spec.audio_codec.as_deref(),
                actual.audio_codec.as_deref(),
                self.profile.require_codec_match,
            ),
            width: self.check_exact_u32(
                "width",
                spec.width,
                actual.width,
                self.profile.require_exact_resolution,
            ),
            height: self.check_exact_u32(
                "height",
                spec.height,
                actual.height,
                self.profile.require_exact_resolution,
            ),
            video_bitrate: Self::check_bitrate(
                "video_bitrate",
                spec.video_bitrate_bps,
                actual.video_bitrate_bps,
                tol.bitrate_relative,
                self.profile.check_bitrate,
            ),
            audio_bitrate: Self::check_bitrate(
                "audio_bitrate",
                spec.audio_bitrate_bps,
                actual.audio_bitrate_bps,
                tol.bitrate_relative,
                self.profile.check_bitrate,
            ),
            duration: Self::check_duration(
                spec.duration_secs,
                actual.duration_secs,
                tol.duration_secs,
                self.profile.check_duration,
            ),
            frame_rate: Self::check_frame_rate(
                spec.frame_rate_num,
                spec.frame_rate_den,
                actual.frame_rate_num,
                actual.frame_rate_den,
                tol.frame_rate_fps,
                self.profile.check_frame_rate,
            ),
            container: self.check_codec(
                "container",
                spec.container_format.as_deref(),
                actual.container_format.as_deref(),
                self.profile.check_container,
            ),
        }
    }

    /// Validates and returns an error if the report failed.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidOutput`] listing all failed fields.
    pub fn validate_strict(
        &self,
        spec: &OutputSpec,
        actual: &ActualOutputProperties,
    ) -> Result<ValidationReport> {
        let report = self.validate(spec, actual);
        if !report.passed() {
            let msgs: Vec<String> = report
                .failures()
                .into_iter()
                .map(|(name, r)| format!("{name}: {r}"))
                .collect();
            Err(TranscodeError::InvalidOutput(msgs.join("; ")))
        } else {
            Ok(report)
        }
    }

    // ── private helpers ─────────────────────────────────────────────────────

    fn check_codec(
        &self,
        field: &str,
        expected: Option<&str>,
        actual: Option<&str>,
        required: bool,
    ) -> FieldResult {
        let Some(exp) = expected else {
            return FieldResult::Skipped;
        };
        if !required {
            return FieldResult::Skipped;
        }
        match actual {
            None => FieldResult::Missing(field.to_string()),
            Some(act) if act.eq_ignore_ascii_case(exp) => FieldResult::Pass,
            Some(act) => FieldResult::Fail(format!("expected {exp:?}, got {act:?}")),
        }
    }

    fn check_exact_u32(
        &self,
        field: &str,
        expected: Option<u32>,
        actual: Option<u32>,
        required: bool,
    ) -> FieldResult {
        let Some(exp) = expected else {
            return FieldResult::Skipped;
        };
        if !required {
            return FieldResult::Skipped;
        }
        match actual {
            None => FieldResult::Missing(field.to_string()),
            Some(act) if act == exp => FieldResult::Pass,
            Some(act) => FieldResult::Fail(format!("expected {exp}, got {act}")),
        }
    }

    fn check_bitrate(
        field: &str,
        expected: Option<u64>,
        actual: Option<u64>,
        relative_tol: f64,
        enabled: bool,
    ) -> FieldResult {
        if !enabled {
            return FieldResult::Skipped;
        }
        let Some(exp) = expected else {
            return FieldResult::Skipped;
        };
        let Some(act) = actual else {
            return FieldResult::Missing(field.to_string());
        };
        let exp_f = exp as f64;
        let act_f = act as f64;
        let rel_diff = (act_f - exp_f).abs() / exp_f.max(1.0);
        if rel_diff <= relative_tol {
            FieldResult::Pass
        } else {
            FieldResult::Fail(format!(
                "expected {exp} bps, got {act} bps (diff {:.1}% > tolerance {:.1}%)",
                rel_diff * 100.0,
                relative_tol * 100.0
            ))
        }
    }

    fn check_duration(
        expected: Option<f64>,
        actual: Option<f64>,
        tol_secs: f64,
        enabled: bool,
    ) -> FieldResult {
        if !enabled {
            return FieldResult::Skipped;
        }
        let Some(exp) = expected else {
            return FieldResult::Skipped;
        };
        let Some(act) = actual else {
            return FieldResult::Missing("duration".to_string());
        };
        let diff = (act - exp).abs();
        if diff <= tol_secs {
            FieldResult::Pass
        } else {
            FieldResult::Fail(format!(
                "expected {exp:.3}s, got {act:.3}s (diff {diff:.3}s > tolerance {tol_secs:.3}s)"
            ))
        }
    }

    fn check_frame_rate(
        exp_num: Option<u32>,
        exp_den: Option<u32>,
        act_num: Option<u32>,
        act_den: Option<u32>,
        tol_fps: f64,
        enabled: bool,
    ) -> FieldResult {
        if !enabled {
            return FieldResult::Skipped;
        }
        let (Some(en), Some(ed)) = (exp_num, exp_den) else {
            return FieldResult::Skipped;
        };
        if ed == 0 {
            return FieldResult::Fail("expected frame rate has zero denominator".to_string());
        }
        let (Some(an), Some(ad)) = (act_num, act_den) else {
            return FieldResult::Missing("frame_rate".to_string());
        };
        if ad == 0 {
            return FieldResult::Fail("actual frame rate has zero denominator".to_string());
        }
        let exp_fps = en as f64 / ed as f64;
        let act_fps = an as f64 / ad as f64;
        let diff = (act_fps - exp_fps).abs();
        if diff <= tol_fps {
            FieldResult::Pass
        } else {
            FieldResult::Fail(format!(
                "expected {exp_fps:.4} fps, got {act_fps:.4} fps (diff {diff:.4} > {tol_fps:.4})"
            ))
        }
    }
}

impl Default for OutputValidator {
    fn default() -> Self {
        Self::new(ValidationProfile::default())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_actual() -> ActualOutputProperties {
        ActualOutputProperties {
            video_codec: Some("vp9".to_string()),
            audio_codec: Some("opus".to_string()),
            width: Some(1920),
            height: Some(1080),
            video_bitrate_bps: Some(5_000_000),
            audio_bitrate_bps: Some(128_000),
            duration_secs: Some(60.0),
            frame_rate_num: Some(30),
            frame_rate_den: Some(1),
            container_format: Some("webm".to_string()),
        }
    }

    fn full_spec() -> OutputSpec {
        OutputSpec::builder()
            .video_codec("vp9")
            .audio_codec("opus")
            .resolution(1920, 1080)
            .video_bitrate_bps(5_000_000)
            .audio_bitrate_bps(128_000)
            .duration_secs(60.0)
            .frame_rate(30, 1)
            .container("webm")
            .build()
    }

    #[test]
    fn test_perfect_match_passes() {
        let validator = OutputValidator::new(ValidationProfile::broadcast());
        let report = validator.validate(&full_spec(), &perfect_actual());
        assert!(report.passed(), "report: {report}");
    }

    #[test]
    fn test_codec_mismatch_fails() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::builder().video_codec("vp9").build();
        let actual = ActualOutputProperties {
            video_codec: Some("h264".to_string()),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(!report.passed());
        assert!(matches!(report.video_codec, FieldResult::Fail(_)));
    }

    #[test]
    fn test_bitrate_within_tolerance_passes() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        // 10 % tolerance, bitrate is 5 % off.
        let spec = OutputSpec::builder().video_bitrate_bps(5_000_000).build();
        let actual = ActualOutputProperties {
            video_bitrate_bps: Some(5_250_000),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(report.video_bitrate.is_ok());
    }

    #[test]
    fn test_bitrate_outside_tolerance_fails() {
        let validator = OutputValidator::new(ValidationProfile::broadcast()); // 2 % tol
        let spec = OutputSpec::builder().video_bitrate_bps(5_000_000).build();
        let actual = ActualOutputProperties {
            video_bitrate_bps: Some(4_500_000), // 10 % off
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(matches!(report.video_bitrate, FieldResult::Fail(_)));
    }

    #[test]
    fn test_resolution_mismatch_fails() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::builder().resolution(1920, 1080).build();
        let actual = ActualOutputProperties {
            width: Some(1280),
            height: Some(720),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(matches!(report.width, FieldResult::Fail(_)));
        assert!(matches!(report.height, FieldResult::Fail(_)));
    }

    #[test]
    fn test_duration_within_tolerance() {
        let validator = OutputValidator::new(ValidationProfile::streaming()); // 0.5 s tol
        let spec = OutputSpec::builder().duration_secs(60.0).build();
        let actual = ActualOutputProperties {
            duration_secs: Some(60.3),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(report.duration.is_ok());
    }

    #[test]
    fn test_duration_outside_tolerance() {
        let validator = OutputValidator::new(ValidationProfile::broadcast()); // 0.1 s tol
        let spec = OutputSpec::builder().duration_secs(60.0).build();
        let actual = ActualOutputProperties {
            duration_secs: Some(61.0),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(matches!(report.duration, FieldResult::Fail(_)));
    }

    #[test]
    fn test_frame_rate_pass() {
        let validator = OutputValidator::new(ValidationProfile::broadcast());
        let spec = OutputSpec::builder().frame_rate(30, 1).build();
        let actual = ActualOutputProperties {
            frame_rate_num: Some(30),
            frame_rate_den: Some(1),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(report.frame_rate.is_ok());
    }

    #[test]
    fn test_frame_rate_mismatch_fails() {
        let validator = OutputValidator::new(ValidationProfile::broadcast());
        let spec = OutputSpec::builder().frame_rate(30, 1).build();
        let actual = ActualOutputProperties {
            frame_rate_num: Some(25),
            frame_rate_den: Some(1),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(matches!(report.frame_rate, FieldResult::Fail(_)));
    }

    #[test]
    fn test_missing_actual_value() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::builder().video_codec("vp9").build();
        let actual = ActualOutputProperties::default(); // no codec measured
        let report = validator.validate(&spec, &actual);
        assert!(matches!(report.video_codec, FieldResult::Missing(_)));
    }

    #[test]
    fn test_skipped_when_spec_field_absent() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::default(); // no fields set
        let actual = perfect_actual();
        let report = validator.validate(&spec, &actual);
        // Everything should be skipped.
        assert!(report.passed());
    }

    #[test]
    fn test_validate_strict_returns_error_on_failure() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::builder().video_codec("vp9").build();
        let actual = ActualOutputProperties {
            video_codec: Some("av1".to_string()),
            ..Default::default()
        };
        let result = validator.validate_strict(&spec, &actual);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_strict_ok_on_pass() {
        let validator = OutputValidator::new(ValidationProfile::preview());
        let spec = OutputSpec::builder().duration_secs(10.0).build();
        let actual = ActualOutputProperties {
            duration_secs: Some(10.1),
            ..Default::default()
        };
        let result = validator.validate_strict(&spec, &actual);
        assert!(result.is_ok());
    }

    #[test]
    fn test_failure_count() {
        let validator = OutputValidator::new(ValidationProfile::broadcast());
        let spec = full_spec();
        let actual = ActualOutputProperties {
            video_codec: Some("h264".to_string()),
            audio_codec: Some("aac".to_string()),
            width: Some(1280),
            height: Some(720),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        // codec x2, resolution x2 fail; bitrate/duration/framerate are Missing.
        assert!(report.failure_count() > 0);
    }

    #[test]
    fn test_codec_case_insensitive() {
        let validator = OutputValidator::new(ValidationProfile::streaming());
        let spec = OutputSpec::builder().video_codec("VP9").build();
        let actual = ActualOutputProperties {
            video_codec: Some("vp9".to_string()),
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        assert!(report.video_codec.is_ok());
    }

    #[test]
    fn test_preview_profile_skips_bitrate() {
        let validator = OutputValidator::new(ValidationProfile::preview());
        let spec = OutputSpec::builder().video_bitrate_bps(5_000_000).build();
        let actual = ActualOutputProperties {
            video_bitrate_bps: Some(1_000_000), // way off
            ..Default::default()
        };
        let report = validator.validate(&spec, &actual);
        // preview profile does not check bitrate — should be skipped.
        assert_eq!(report.video_bitrate, FieldResult::Skipped);
    }
}
