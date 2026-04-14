//! Quality control (QC) checks for ingested media assets.
//!
//! Provides a composable framework for running technical and editorial QC
//! passes on media assets:
//!
//! * `QcCheckSpec` – describes a named QC check and its severity.
//! * `QcIssue` – a single finding raised by a check.
//! * `QcReport` – the full result of running all checks against an asset.
//! * `QcStatus` – aggregated pass/fail/warning status.
//! * `QcProfile` – a named bundle of `QcCheckSpec`s (e.g. "Broadcast", "Web").
//! * `QcEngine` – runs a set of specs against a `QcTarget` and returns a report.
//!
//! The actual check logic is defined via the `QcChecker` trait so that
//! callers can plug in domain-specific validators without modifying this
//! module.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// QcSeverity
// ---------------------------------------------------------------------------

/// How severe a QC issue is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum QcSeverity {
    /// Informational: no action required.
    Info,
    /// Advisory: should be reviewed but does not block delivery.
    Warning,
    /// Critical: must be resolved before delivery.
    Error,
    /// Fatal: the asset is unusable.
    Fatal,
}

impl QcSeverity {
    /// Human-readable label.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Fatal => "fatal",
        }
    }

    /// Returns `true` if the severity blocks delivery.
    #[must_use]
    pub fn blocks_delivery(&self) -> bool {
        matches!(self, Self::Error | Self::Fatal)
    }
}

// ---------------------------------------------------------------------------
// QcCheckSpec
// ---------------------------------------------------------------------------

/// Specification for a single QC check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcCheckSpec {
    /// Unique identifier (e.g. `"audio.loudness.ebur128"`).
    pub id: String,
    /// Short human-readable title.
    pub title: String,
    /// Verbose description of what is checked.
    pub description: String,
    /// Severity if this check raises an issue.
    pub severity: QcSeverity,
    /// Whether this check can be auto-remediated.
    pub auto_remediate: bool,
    /// Optional JSON-encoded configuration for the checker.
    pub config: Option<serde_json::Value>,
}

impl QcCheckSpec {
    /// Create a new spec with the minimum required fields.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        severity: QcSeverity,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            severity,
            auto_remediate: false,
            config: None,
        }
    }

    /// Builder: set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Builder: enable auto-remediation.
    #[must_use]
    pub fn with_auto_remediate(mut self) -> Self {
        self.auto_remediate = true;
        self
    }

    /// Builder: attach JSON configuration.
    #[must_use]
    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = Some(config);
        self
    }
}

// ---------------------------------------------------------------------------
// QcIssue
// ---------------------------------------------------------------------------

/// A single finding raised by a QC check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcIssue {
    /// The check that raised this issue.
    pub check_id: String,
    /// Severity of this finding.
    pub severity: QcSeverity,
    /// Short message describing the issue.
    pub message: String,
    /// Optional timecode location in milliseconds.
    pub timecode_ms: Option<u64>,
    /// Optional end timecode for range issues.
    pub timecode_end_ms: Option<u64>,
    /// Optional machine-readable detail payload.
    pub detail: Option<serde_json::Value>,
    /// Whether this issue was auto-remediated.
    pub remediated: bool,
}

impl QcIssue {
    /// Create a new issue.
    #[must_use]
    pub fn new(
        check_id: impl Into<String>,
        severity: QcSeverity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            check_id: check_id.into(),
            severity,
            message: message.into(),
            timecode_ms: None,
            timecode_end_ms: None,
            detail: None,
            remediated: false,
        }
    }

    /// Builder: set a timecode location.
    #[must_use]
    pub fn at_timecode(mut self, ms: u64) -> Self {
        self.timecode_ms = Some(ms);
        self
    }

    /// Builder: set a timecode range.
    #[must_use]
    pub fn at_range(mut self, start_ms: u64, end_ms: u64) -> Self {
        self.timecode_ms = Some(start_ms);
        self.timecode_end_ms = Some(end_ms);
        self
    }

    /// Builder: attach a JSON detail payload.
    #[must_use]
    pub fn with_detail(mut self, detail: serde_json::Value) -> Self {
        self.detail = Some(detail);
        self
    }
}

// ---------------------------------------------------------------------------
// QcStatus
// ---------------------------------------------------------------------------

/// Aggregated pass/fail/warning status of a QC report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QcStatus {
    /// No issues found.
    Pass,
    /// Only informational or warning issues found.
    PassWithWarnings,
    /// At least one blocking (Error/Fatal) issue found.
    Fail,
}

impl QcStatus {
    /// Returns `true` if the asset can proceed to delivery.
    #[must_use]
    pub fn is_deliverable(&self) -> bool {
        matches!(self, Self::Pass | Self::PassWithWarnings)
    }
}

// ---------------------------------------------------------------------------
// QcReport
// ---------------------------------------------------------------------------

/// The full result of running QC checks against an asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcReport {
    /// Unique report id.
    pub id: Uuid,
    /// The asset that was checked.
    pub asset_id: Uuid,
    /// Name of the QC profile that was run.
    pub profile_name: String,
    /// Aggregated pass/fail status.
    pub status: QcStatus,
    /// All issues found across all checks.
    pub issues: Vec<QcIssue>,
    /// Checks that passed with no issues.
    pub passed_checks: Vec<String>,
    /// Checks that were skipped (e.g. required metadata absent).
    pub skipped_checks: Vec<String>,
    /// When the QC run completed.
    pub completed_at: DateTime<Utc>,
    /// Duration of the QC run in milliseconds.
    pub duration_ms: u64,
    /// Optional free-form operator notes.
    pub notes: Option<String>,
}

impl QcReport {
    /// Build a report from a list of issues and the set of checks that ran.
    #[must_use]
    pub fn build(
        asset_id: Uuid,
        profile_name: impl Into<String>,
        issues: Vec<QcIssue>,
        passed_checks: Vec<String>,
        skipped_checks: Vec<String>,
        duration_ms: u64,
    ) -> Self {
        let status = if issues.iter().any(|i| i.severity.blocks_delivery()) {
            QcStatus::Fail
        } else if issues.is_empty() {
            QcStatus::Pass
        } else {
            QcStatus::PassWithWarnings
        };

        Self {
            id: Uuid::new_v4(),
            asset_id,
            profile_name: profile_name.into(),
            status,
            issues,
            passed_checks,
            skipped_checks,
            completed_at: Utc::now(),
            duration_ms,
            notes: None,
        }
    }

    /// Return only issues at or above the given severity.
    #[must_use]
    pub fn issues_at_or_above(&self, min: QcSeverity) -> Vec<&QcIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity >= min)
            .collect()
    }

    /// Count issues by severity.
    #[must_use]
    pub fn issue_counts(&self) -> HashMap<QcSeverity, usize> {
        let mut counts = HashMap::new();
        for issue in &self.issues {
            *counts.entry(issue.severity).or_insert(0) += 1;
        }
        counts
    }

    /// Returns `true` if there are no blocking issues after remediation is
    /// applied (i.e. all Error/Fatal issues are marked `remediated`).
    #[must_use]
    pub fn is_clear_after_remediation(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity.blocks_delivery() && !i.remediated)
    }
}

// ---------------------------------------------------------------------------
// QcTarget
// ---------------------------------------------------------------------------

/// Metadata about an asset submitted to the QC engine.
///
/// Real implementations would include decoded frame/sample data; this
/// struct carries the properties that pure-logic checks can inspect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcTarget {
    /// Asset id.
    pub asset_id: Uuid,
    /// File name (for format checks).
    pub filename: String,
    /// File size in bytes.
    pub file_size_bytes: u64,
    /// Duration in milliseconds.
    pub duration_ms: Option<u64>,
    /// Video width in pixels (if applicable).
    pub width: Option<u32>,
    /// Video height in pixels.
    pub height: Option<u32>,
    /// Frame rate (frames per second).
    pub frame_rate: Option<f64>,
    /// Number of audio channels.
    pub audio_channels: Option<u32>,
    /// Audio sample rate (Hz).
    pub audio_sample_rate: Option<u32>,
    /// Integrated loudness (LUFS) from prior measurement.
    pub integrated_loudness_lufs: Option<f64>,
    /// True-peak level (dBTP).
    pub true_peak_dbtp: Option<f64>,
    /// Detected codec string (e.g. `"h264"`, `"aac"`).
    pub video_codec: Option<String>,
    /// Detected audio codec string.
    pub audio_codec: Option<String>,
    /// Whether the file is interlaced.
    pub interlaced: Option<bool>,
    /// Extra key-value metadata for extensibility.
    pub extra: HashMap<String, String>,
}

impl QcTarget {
    /// Create a minimal target for an asset.
    #[must_use]
    pub fn new(asset_id: Uuid, filename: impl Into<String>, file_size_bytes: u64) -> Self {
        Self {
            asset_id,
            filename: filename.into(),
            file_size_bytes,
            duration_ms: None,
            width: None,
            height: None,
            frame_rate: None,
            audio_channels: None,
            audio_sample_rate: None,
            integrated_loudness_lufs: None,
            true_peak_dbtp: None,
            video_codec: None,
            audio_codec: None,
            interlaced: None,
            extra: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// QcChecker trait
// ---------------------------------------------------------------------------

/// Trait for implementing a single QC check.
///
/// Implementors inspect the `QcTarget` and return zero or more `QcIssue`s.
/// Returning an empty `Vec` means the check passed.
pub trait QcChecker: Send + Sync {
    /// The unique identifier of this checker (must match a `QcCheckSpec.id`).
    fn check_id(&self) -> &str;

    /// Run the check against the given target.
    ///
    /// # Errors
    ///
    /// Returns a `String` describing the failure if the check itself
    /// encountered an internal error (distinct from a found issue).
    fn run(&self, target: &QcTarget) -> Result<Vec<QcIssue>, String>;
}

// ---------------------------------------------------------------------------
// Built-in checkers
// ---------------------------------------------------------------------------

/// Checks that the file is above a minimum size.
pub struct MinFileSizeChecker {
    /// Minimum acceptable file size in bytes.
    pub min_bytes: u64,
}

impl QcChecker for MinFileSizeChecker {
    fn check_id(&self) -> &str {
        "file.min_size"
    }

    fn run(&self, target: &QcTarget) -> Result<Vec<QcIssue>, String> {
        if target.file_size_bytes < self.min_bytes {
            Ok(vec![QcIssue::new(
                self.check_id(),
                QcSeverity::Error,
                format!(
                    "File size {} B is below minimum {} B",
                    target.file_size_bytes, self.min_bytes
                ),
            )])
        } else {
            Ok(vec![])
        }
    }
}

/// Checks that the video resolution meets a minimum requirement.
pub struct MinResolutionChecker {
    /// Minimum width in pixels.
    pub min_width: u32,
    /// Minimum height in pixels.
    pub min_height: u32,
}

impl QcChecker for MinResolutionChecker {
    fn check_id(&self) -> &str {
        "video.min_resolution"
    }

    fn run(&self, target: &QcTarget) -> Result<Vec<QcIssue>, String> {
        let mut issues = Vec::new();
        if let (Some(w), Some(h)) = (target.width, target.height) {
            if w < self.min_width || h < self.min_height {
                issues.push(
                    QcIssue::new(
                        self.check_id(),
                        QcSeverity::Error,
                        format!(
                            "Resolution {w}x{h} is below minimum {}x{}",
                            self.min_width, self.min_height
                        ),
                    )
                    .with_detail(serde_json::json!({
                        "actual_width": w,
                        "actual_height": h,
                        "min_width": self.min_width,
                        "min_height": self.min_height
                    })),
                );
            }
        }
        Ok(issues)
    }
}

/// Checks that integrated loudness is within EBU R128 limits.
pub struct LoudnessChecker {
    /// Target integrated loudness in LUFS (typically -23.0 for broadcast).
    pub target_lufs: f64,
    /// Tolerance in LU (typically ±1.0).
    pub tolerance_lu: f64,
    /// Maximum true peak in dBTP (typically -1.0).
    pub max_true_peak_dbtp: f64,
}

impl LoudnessChecker {
    /// Create a checker with EBU R128 broadcast defaults.
    #[must_use]
    pub fn ebu_r128_broadcast() -> Self {
        Self {
            target_lufs: -23.0,
            tolerance_lu: 1.0,
            max_true_peak_dbtp: -1.0,
        }
    }
}

impl QcChecker for LoudnessChecker {
    fn check_id(&self) -> &str {
        "audio.loudness"
    }

    fn run(&self, target: &QcTarget) -> Result<Vec<QcIssue>, String> {
        let mut issues = Vec::new();

        if let Some(lufs) = target.integrated_loudness_lufs {
            let diff = (lufs - self.target_lufs).abs();
            if diff > self.tolerance_lu {
                issues.push(
                    QcIssue::new(
                        self.check_id(),
                        QcSeverity::Error,
                        format!(
                            "Integrated loudness {lufs:.1} LUFS deviates by {diff:.1} LU from target {:.1} LUFS",
                            self.target_lufs
                        ),
                    )
                    .with_detail(serde_json::json!({
                        "measured_lufs": lufs,
                        "target_lufs": self.target_lufs,
                        "tolerance_lu": self.tolerance_lu,
                    })),
                );
            }
        }

        if let Some(tp) = target.true_peak_dbtp {
            if tp > self.max_true_peak_dbtp {
                issues.push(QcIssue::new(
                    self.check_id(),
                    QcSeverity::Error,
                    format!(
                        "True peak {tp:.1} dBTP exceeds maximum {:.1} dBTP",
                        self.max_true_peak_dbtp
                    ),
                ));
            }
        }

        Ok(issues)
    }
}

/// Checks that the frame rate matches one of the expected values.
pub struct FrameRateChecker {
    /// Acceptable frame rates.
    pub allowed_fps: Vec<f64>,
    /// Tolerance for floating-point comparison.
    pub tolerance: f64,
}

impl FrameRateChecker {
    /// Common broadcast frame rates (23.976, 24, 25, 29.97, 30, 50, 59.94, 60).
    #[must_use]
    pub fn broadcast() -> Self {
        Self {
            allowed_fps: vec![23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0],
            tolerance: 0.01,
        }
    }
}

impl QcChecker for FrameRateChecker {
    fn check_id(&self) -> &str {
        "video.frame_rate"
    }

    fn run(&self, target: &QcTarget) -> Result<Vec<QcIssue>, String> {
        if let Some(fps) = target.frame_rate {
            let matched = self
                .allowed_fps
                .iter()
                .any(|&allowed| (fps - allowed).abs() <= self.tolerance);
            if !matched {
                return Ok(vec![QcIssue::new(
                    self.check_id(),
                    QcSeverity::Warning,
                    format!("Frame rate {fps} fps is not in the allowed set"),
                )
                .with_detail(serde_json::json!({
                    "fps": fps,
                    "allowed": self.allowed_fps,
                }))]);
            }
        }
        Ok(vec![])
    }
}

// ---------------------------------------------------------------------------
// QcProfile
// ---------------------------------------------------------------------------

/// A named bundle of QC check specifications.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QcProfile {
    /// Profile identifier (e.g. `"broadcast_hd"`).
    pub id: String,
    /// Display name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Check specs included in this profile.
    pub checks: Vec<QcCheckSpec>,
}

impl QcProfile {
    /// Create an empty profile.
    #[must_use]
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: None,
            checks: Vec::new(),
        }
    }

    /// Add a check spec to the profile.
    pub fn add_check(&mut self, spec: QcCheckSpec) {
        self.checks.push(spec);
    }

    /// Number of checks in this profile.
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Return a preset broadcast HD profile with common check specs.
    #[must_use]
    pub fn broadcast_hd() -> Self {
        let mut profile = Self::new("broadcast_hd", "Broadcast HD");
        profile.description = Some(
            "Technical QC for HD broadcast delivery (EBU R128, 1920×1080, 25/29.97 fps)."
                .to_string(),
        );
        profile.add_check(
            QcCheckSpec::new("file.min_size", "Minimum File Size", QcSeverity::Error)
                .with_description("File must be at least 1 MB"),
        );
        profile.add_check(
            QcCheckSpec::new(
                "video.min_resolution",
                "Minimum HD Resolution",
                QcSeverity::Error,
            )
            .with_description("Video must be at least 1920×1080"),
        );
        profile.add_check(
            QcCheckSpec::new("audio.loudness", "EBU R128 Loudness", QcSeverity::Error)
                .with_description("Integrated loudness must be -23 LUFS ±1 LU"),
        );
        profile.add_check(
            QcCheckSpec::new("video.frame_rate", "Broadcast Frame Rate", QcSeverity::Warning)
                .with_description("Frame rate must be a standard broadcast value"),
        );
        profile
    }

    /// Return a lightweight web-streaming profile.
    #[must_use]
    pub fn web_streaming() -> Self {
        let mut profile = Self::new("web_streaming", "Web Streaming");
        profile.description =
            Some("Basic QC for web/OTT streaming (360p+, -16 LUFS).".to_string());
        profile.add_check(QcCheckSpec::new(
            "file.min_size",
            "Minimum File Size",
            QcSeverity::Warning,
        ));
        profile.add_check(
            QcCheckSpec::new("audio.loudness", "Streaming Loudness", QcSeverity::Warning)
                .with_config(serde_json::json!({ "target_lufs": -16.0, "tolerance_lu": 2.0 })),
        );
        profile
    }
}

// ---------------------------------------------------------------------------
// QcEngine
// ---------------------------------------------------------------------------

/// Runs registered `QcChecker`s against a `QcTarget` and produces a `QcReport`.
#[derive(Default)]
pub struct QcEngine {
    /// Registered checkers keyed by their check id.
    checkers: HashMap<String, Box<dyn QcChecker>>,
}

impl QcEngine {
    /// Create an empty engine.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a checker.  If a checker with the same id already exists, it
    /// is replaced.
    pub fn register<C: QcChecker + 'static>(&mut self, checker: C) {
        self.checkers
            .insert(checker.check_id().to_string(), Box::new(checker));
    }

    /// Create an engine pre-populated with the built-in broadcast checkers.
    #[must_use]
    pub fn broadcast_defaults() -> Self {
        let mut engine = Self::new();
        engine.register(MinFileSizeChecker { min_bytes: 1_048_576 });
        engine.register(MinResolutionChecker {
            min_width: 1920,
            min_height: 1080,
        });
        engine.register(LoudnessChecker::ebu_r128_broadcast());
        engine.register(FrameRateChecker::broadcast());
        engine
    }

    /// Run all checks in the given profile against the target.
    ///
    /// Checks not registered in the engine are counted as skipped.
    #[must_use]
    pub fn run(&self, target: &QcTarget, profile: &QcProfile) -> QcReport {
        let start = std::time::Instant::now();
        let mut all_issues: Vec<QcIssue> = Vec::new();
        let mut passed_checks: Vec<String> = Vec::new();
        let mut skipped_checks: Vec<String> = Vec::new();

        for spec in &profile.checks {
            match self.checkers.get(&spec.id) {
                None => {
                    skipped_checks.push(spec.id.clone());
                }
                Some(checker) => match checker.run(target) {
                    Err(_msg) => {
                        skipped_checks.push(spec.id.clone());
                    }
                    Ok(mut issues) => {
                        if issues.is_empty() {
                            passed_checks.push(spec.id.clone());
                        } else {
                            // Override severity from profile spec if higher
                            for issue in &mut issues {
                                if spec.severity > issue.severity {
                                    issue.severity = spec.severity;
                                }
                            }
                            all_issues.extend(issues);
                        }
                    }
                },
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        QcReport::build(
            target.asset_id,
            &profile.name,
            all_issues,
            passed_checks,
            skipped_checks,
            duration_ms,
        )
    }

    /// Return how many checkers are registered.
    #[must_use]
    pub fn checker_count(&self) -> usize {
        self.checkers.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_target() -> QcTarget {
        let mut t = QcTarget::new(Uuid::new_v4(), "promo.mp4", 10_000_000);
        t.width = Some(1920);
        t.height = Some(1080);
        t.frame_rate = Some(25.0);
        t.audio_channels = Some(2);
        t.audio_sample_rate = Some(48_000);
        t.integrated_loudness_lufs = Some(-23.0);
        t.true_peak_dbtp = Some(-2.0);
        t.duration_ms = Some(120_000);
        t
    }

    #[test]
    fn test_qc_severity_ordering() {
        assert!(QcSeverity::Fatal > QcSeverity::Error);
        assert!(QcSeverity::Error > QcSeverity::Warning);
        assert!(QcSeverity::Warning > QcSeverity::Info);
    }

    #[test]
    fn test_qc_severity_blocks_delivery() {
        assert!(QcSeverity::Error.blocks_delivery());
        assert!(QcSeverity::Fatal.blocks_delivery());
        assert!(!QcSeverity::Warning.blocks_delivery());
        assert!(!QcSeverity::Info.blocks_delivery());
    }

    #[test]
    fn test_min_file_size_pass() {
        let checker = MinFileSizeChecker { min_bytes: 1_000 };
        let target = QcTarget::new(Uuid::new_v4(), "test.mp4", 5_000);
        let issues = checker.run(&target).expect("checker should not error");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_min_file_size_fail() {
        let checker = MinFileSizeChecker { min_bytes: 100_000 };
        let target = QcTarget::new(Uuid::new_v4(), "tiny.mp4", 500);
        let issues = checker.run(&target).expect("checker should not error");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, QcSeverity::Error);
    }

    #[test]
    fn test_resolution_checker_pass() {
        let checker = MinResolutionChecker {
            min_width: 1280,
            min_height: 720,
        };
        let mut target = QcTarget::new(Uuid::new_v4(), "hd.mp4", 1_000_000);
        target.width = Some(1920);
        target.height = Some(1080);
        let issues = checker.run(&target).expect("no error");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_resolution_checker_fail() {
        let checker = MinResolutionChecker {
            min_width: 1920,
            min_height: 1080,
        };
        let mut target = QcTarget::new(Uuid::new_v4(), "sd.mp4", 1_000_000);
        target.width = Some(640);
        target.height = Some(360);
        let issues = checker.run(&target).expect("no error");
        assert_eq!(issues.len(), 1);
    }

    #[test]
    fn test_loudness_checker_pass() {
        let checker = LoudnessChecker::ebu_r128_broadcast();
        let target = make_target();
        let issues = checker.run(&target).expect("no error");
        assert!(issues.is_empty(), "loudness should pass for -23 LUFS");
    }

    #[test]
    fn test_loudness_checker_fail_lufs() {
        let checker = LoudnessChecker::ebu_r128_broadcast();
        let mut target = make_target();
        target.integrated_loudness_lufs = Some(-18.0); // too loud
        let issues = checker.run(&target).expect("no error");
        assert!(!issues.is_empty());
        assert_eq!(issues[0].severity, QcSeverity::Error);
    }

    #[test]
    fn test_loudness_checker_fail_true_peak() {
        let checker = LoudnessChecker::ebu_r128_broadcast();
        let mut target = make_target();
        target.true_peak_dbtp = Some(0.5); // above -1 dBTP
        let issues = checker.run(&target).expect("no error");
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_frame_rate_checker_pass() {
        let checker = FrameRateChecker::broadcast();
        let target = make_target(); // 25 fps
        let issues = checker.run(&target).expect("no error");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_frame_rate_checker_warning_nonstandard() {
        let checker = FrameRateChecker::broadcast();
        let mut target = make_target();
        target.frame_rate = Some(15.0);
        let issues = checker.run(&target).expect("no error");
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].severity, QcSeverity::Warning);
    }

    #[test]
    fn test_qc_report_status_pass() {
        let report = QcReport::build(
            Uuid::new_v4(),
            "test",
            vec![],
            vec!["file.min_size".to_string()],
            vec![],
            100,
        );
        assert_eq!(report.status, QcStatus::Pass);
        assert!(report.status.is_deliverable());
    }

    #[test]
    fn test_qc_report_status_fail() {
        let issue = QcIssue::new("video.min_resolution", QcSeverity::Error, "Too small");
        let report = QcReport::build(Uuid::new_v4(), "test", vec![issue], vec![], vec![], 50);
        assert_eq!(report.status, QcStatus::Fail);
        assert!(!report.status.is_deliverable());
    }

    #[test]
    fn test_qc_report_issue_counts() {
        let issues = vec![
            QcIssue::new("a", QcSeverity::Warning, "warn1"),
            QcIssue::new("b", QcSeverity::Warning, "warn2"),
            QcIssue::new("c", QcSeverity::Error, "err1"),
        ];
        let report = QcReport::build(Uuid::new_v4(), "test", issues, vec![], vec![], 10);
        let counts = report.issue_counts();
        assert_eq!(counts.get(&QcSeverity::Warning), Some(&2));
        assert_eq!(counts.get(&QcSeverity::Error), Some(&1));
    }

    #[test]
    fn test_qc_engine_broadcast_defaults_pass() {
        let engine = QcEngine::broadcast_defaults();
        let profile = QcProfile::broadcast_hd();
        let target = make_target();
        let report = engine.run(&target, &profile);
        assert_eq!(
            report.status,
            QcStatus::Pass,
            "broadcast target should pass: {:?}",
            report.issues
        );
    }

    #[test]
    fn test_qc_engine_fail_on_bad_target() {
        let engine = QcEngine::broadcast_defaults();
        let profile = QcProfile::broadcast_hd();
        let target = QcTarget::new(Uuid::new_v4(), "bad.mp4", 100); // tiny file
        let report = engine.run(&target, &profile);
        assert_eq!(report.status, QcStatus::Fail);
    }

    #[test]
    fn test_qc_profile_broadcast_hd_has_four_checks() {
        let profile = QcProfile::broadcast_hd();
        assert_eq!(profile.check_count(), 4);
    }

    #[test]
    fn test_qc_profile_web_streaming_has_two_checks() {
        let profile = QcProfile::web_streaming();
        assert_eq!(profile.check_count(), 2);
    }

    #[test]
    fn test_qc_issue_timecode_range() {
        let issue = QcIssue::new("audio.clip", QcSeverity::Warning, "Clipping detected")
            .at_range(5_000, 10_000);
        assert_eq!(issue.timecode_ms, Some(5_000));
        assert_eq!(issue.timecode_end_ms, Some(10_000));
    }

    #[test]
    fn test_qc_report_is_clear_after_remediation() {
        let mut issue = QcIssue::new("audio.loudness", QcSeverity::Error, "Too loud");
        issue.remediated = true;
        let report = QcReport::build(Uuid::new_v4(), "test", vec![issue], vec![], vec![], 10);
        assert!(report.is_clear_after_remediation());
    }
}
