//! Python bindings for `oximedia-qc` quality control and validation.
//!
//! Provides `PyQcChecker`, `PyQcReport`, `PyQcRule`, `PyQcIssue`,
//! and standalone convenience functions for QC from Python.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyQcIssue
// ---------------------------------------------------------------------------

/// A single QC check issue/result.
#[pyclass]
#[derive(Clone)]
pub struct PyQcIssue {
    /// Name of the rule that produced this issue.
    #[pyo3(get)]
    pub rule_name: String,
    /// Whether the check passed.
    #[pyo3(get)]
    pub passed: bool,
    /// Severity level (info, warning, error, critical).
    #[pyo3(get)]
    pub severity: String,
    /// Detailed message.
    #[pyo3(get)]
    pub message: String,
    /// Optional recommendation for fixing.
    #[pyo3(get)]
    pub recommendation: Option<String>,
}

#[pymethods]
impl PyQcIssue {
    fn __repr__(&self) -> String {
        format!(
            "PyQcIssue(rule='{}', passed={}, severity='{}', message='{}')",
            self.rule_name, self.passed, self.severity, self.message,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "rule_name".to_string(),
            self.rule_name
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "passed".to_string(),
            self.passed
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind(),
        );
        m.insert(
            "severity".to_string(),
            self.severity.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "message".to_string(),
            self.message.clone().into_pyobject(py)?.into_any().unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyQcRule
// ---------------------------------------------------------------------------

/// Description of a QC rule.
#[pyclass]
#[derive(Clone)]
pub struct PyQcRule {
    /// Rule name identifier.
    #[pyo3(get)]
    pub name: String,
    /// Category (video, audio, container, compliance).
    #[pyo3(get)]
    pub category: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
}

#[pymethods]
impl PyQcRule {
    fn __repr__(&self) -> String {
        format!(
            "PyQcRule(name='{}', category='{}')",
            self.name, self.category,
        )
    }
}

// ---------------------------------------------------------------------------
// PyQcReport
// ---------------------------------------------------------------------------

/// Comprehensive QC report result.
#[pyclass]
#[derive(Clone)]
pub struct PyQcReport {
    /// File path that was checked.
    #[pyo3(get)]
    pub file_path: String,
    /// Whether all checks passed overall.
    #[pyo3(get)]
    pub overall_passed: bool,
    /// Total number of checks.
    #[pyo3(get)]
    pub total_checks: usize,
    /// Number of passed checks.
    #[pyo3(get)]
    pub passed_checks: usize,
    /// Number of failed checks.
    #[pyo3(get)]
    pub failed_checks: usize,
    /// Validation duration in seconds.
    #[pyo3(get)]
    pub duration: Option<f64>,
    issues: Vec<PyQcIssue>,
}

#[pymethods]
impl PyQcReport {
    /// Get all issues (passed and failed).
    fn issues(&self) -> Vec<PyQcIssue> {
        self.issues.clone()
    }

    /// Get only failed issues.
    fn failures(&self) -> Vec<PyQcIssue> {
        self.issues.iter().filter(|i| !i.passed).cloned().collect()
    }

    /// Get issues filtered by severity.
    fn by_severity(&self, severity: &str) -> Vec<PyQcIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity.to_lowercase() == severity.to_lowercase())
            .cloned()
            .collect()
    }

    /// Get a summary string.
    fn summary(&self) -> String {
        format!(
            "QC Report for '{}': {} ({}/{} passed)",
            self.file_path,
            if self.overall_passed { "PASS" } else { "FAIL" },
            self.passed_checks,
            self.total_checks,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PyQcReport(passed={}, checks={}, failures={})",
            self.overall_passed, self.total_checks, self.failed_checks,
        )
    }
}

// ---------------------------------------------------------------------------
// PyQcChecker
// ---------------------------------------------------------------------------

/// Quality control checker for media files.
#[pyclass]
pub struct PyQcChecker {
    preset_name: String,
}

#[pymethods]
impl PyQcChecker {
    /// Create a new QC checker with a preset.
    ///
    /// Args:
    ///     preset: Preset name (basic, streaming, broadcast, comprehensive, youtube, vimeo).
    #[new]
    #[pyo3(signature = (preset="comprehensive"))]
    fn new(preset: &str) -> Self {
        Self {
            preset_name: preset.to_string(),
        }
    }

    /// Create a broadcast preset checker.
    #[classmethod]
    fn broadcast(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            preset_name: "broadcast".to_string(),
        }
    }

    /// Create a streaming preset checker.
    #[classmethod]
    fn streaming(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            preset_name: "streaming".to_string(),
        }
    }

    /// Run QC checks on a media file.
    ///
    /// Args:
    ///     path: File path to check.
    ///
    /// Returns:
    ///     QC report with all check results.
    fn check(&self, path: &str) -> PyResult<PyQcReport> {
        let preset = resolve_preset(&self.preset_name);
        let qc = oximedia_qc::QualityControl::with_preset(preset);
        let report = qc
            .validate(path)
            .map_err(|e| PyRuntimeError::new_err(format!("QC check failed: {e}")))?;
        Ok(report_to_py(&report, path))
    }

    /// Validate a file against a specific spec.
    ///
    /// Args:
    ///     path: File path to validate.
    ///     spec: Spec name (broadcast, streaming).
    fn validate(&self, path: &str, spec: &str) -> PyResult<PyQcReport> {
        let preset = resolve_preset(spec);
        let qc = oximedia_qc::QualityControl::with_preset(preset);
        let report = match spec.to_lowercase().as_str() {
            "broadcast" => qc.validate_broadcast(path).map_err(|e| {
                PyRuntimeError::new_err(format!("Broadcast validation failed: {e}"))
            })?,
            "streaming" | "web" => qc.validate_streaming(path).map_err(|e| {
                PyRuntimeError::new_err(format!("Streaming validation failed: {e}"))
            })?,
            _ => qc
                .validate(path)
                .map_err(|e| PyRuntimeError::new_err(format!("Validation failed: {e}")))?,
        };
        Ok(report_to_py(&report, path))
    }

    /// Auto-fix common QC issues.
    ///
    /// Args:
    ///     path: File path to analyze for fixable issues.
    ///
    /// Returns:
    ///     List of fixable issues with recommendations.
    fn fix(&self, path: &str) -> PyResult<Vec<PyQcIssue>> {
        let preset = resolve_preset(&self.preset_name);
        let qc = oximedia_qc::QualityControl::with_preset(preset);
        let report = qc
            .validate(path)
            .map_err(|e| PyRuntimeError::new_err(format!("QC analysis failed: {e}")))?;

        let fixable: Vec<PyQcIssue> = report
            .results
            .iter()
            .filter(|r| !r.passed && r.recommendation.is_some())
            .map(check_result_to_py)
            .collect();
        Ok(fixable)
    }

    /// List available QC rules.
    fn rules(&self) -> Vec<PyQcRule> {
        list_all_rules()
    }

    fn __repr__(&self) -> String {
        format!("PyQcChecker(preset='{}')", self.preset_name)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Run a QC check on a media file.
///
/// Args:
///     path: File path.
///     rules: Optional comma-separated rule names (unused currently, reserved).
///
/// Returns:
///     QC report.
#[pyfunction]
#[pyo3(signature = (path, rules=None))]
pub fn run_qc_check(path: &str, rules: Option<&str>) -> PyResult<PyQcReport> {
    let preset_name = rules.unwrap_or("comprehensive");
    let preset = resolve_preset(preset_name);
    let qc = oximedia_qc::QualityControl::with_preset(preset);
    let report = qc
        .validate(path)
        .map_err(|e| PyRuntimeError::new_err(format!("QC check failed: {e}")))?;
    Ok(report_to_py(&report, path))
}

/// List all available QC rules.
#[pyfunction]
pub fn list_qc_rules() -> Vec<PyQcRule> {
    list_all_rules()
}

/// Generate a comprehensive QC report for a file.
///
/// Args:
///     path: File path.
///
/// Returns:
///     QC report.
#[pyfunction]
pub fn qc_report(path: &str) -> PyResult<PyQcReport> {
    let qc = oximedia_qc::QualityControl::with_preset(oximedia_qc::QcPreset::Comprehensive);
    let report = qc
        .validate(path)
        .map_err(|e| PyRuntimeError::new_err(format!("QC report failed: {e}")))?;
    Ok(report_to_py(&report, path))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_preset(name: &str) -> oximedia_qc::QcPreset {
    match name.to_lowercase().as_str() {
        "basic" => oximedia_qc::QcPreset::Basic,
        "streaming" => oximedia_qc::QcPreset::Streaming,
        "broadcast" => oximedia_qc::QcPreset::Broadcast,
        "youtube" => oximedia_qc::QcPreset::YouTube,
        "vimeo" => oximedia_qc::QcPreset::Vimeo,
        _ => oximedia_qc::QcPreset::Comprehensive,
    }
}

fn check_result_to_py(r: &oximedia_qc::rules::CheckResult) -> PyQcIssue {
    PyQcIssue {
        rule_name: r.rule_name.clone(),
        passed: r.passed,
        severity: format!("{}", r.severity),
        message: r.message.clone(),
        recommendation: r.recommendation.clone(),
    }
}

fn report_to_py(report: &oximedia_qc::report::QcReport, path: &str) -> PyQcReport {
    let issues: Vec<PyQcIssue> = report.results.iter().map(check_result_to_py).collect();
    PyQcReport {
        file_path: path.to_string(),
        overall_passed: report.overall_passed,
        total_checks: report.total_checks,
        passed_checks: report.passed_checks,
        failed_checks: report.failed_checks,
        duration: report.validation_duration,
        issues,
    }
}

fn list_all_rules() -> Vec<PyQcRule> {
    let rule_defs = [
        (
            "video_codec_validation",
            "video",
            "Validates video codec is patent-free",
        ),
        ("resolution_check", "video", "Checks resolution constraints"),
        ("framerate_check", "video", "Validates frame rate"),
        ("bitrate_check", "video", "Checks video bitrate range"),
        (
            "interlacing_detection",
            "video",
            "Detects interlaced content",
        ),
        ("black_frame_detection", "video", "Detects black frames"),
        ("freeze_frame_detection", "video", "Detects frozen frames"),
        ("audio_codec_validation", "audio", "Validates audio codec"),
        ("sample_rate_check", "audio", "Checks audio sample rate"),
        (
            "loudness_compliance",
            "audio",
            "EBU R128/ATSC A/85 loudness check",
        ),
        ("clipping_detection", "audio", "Detects audio clipping"),
        ("silence_detection", "audio", "Detects extended silence"),
        ("phase_check", "audio", "Checks phase correlation"),
        ("dc_offset_detection", "audio", "Detects DC offset"),
        (
            "format_validation",
            "container",
            "Validates container format",
        ),
        ("stream_sync", "container", "Checks stream synchronization"),
        (
            "timestamp_continuity",
            "container",
            "Validates timestamp continuity",
        ),
        ("keyframe_interval", "container", "Checks keyframe interval"),
        (
            "broadcast_spec",
            "compliance",
            "Broadcast delivery spec check",
        ),
        (
            "streaming_spec",
            "compliance",
            "Streaming platform spec check",
        ),
        (
            "patent_free_codec",
            "compliance",
            "Patent-free codec enforcement",
        ),
    ];

    rule_defs
        .iter()
        .map(|(name, cat, desc)| PyQcRule {
            name: name.to_string(),
            category: cat.to_string(),
            description: desc.to_string(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all QC bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQcChecker>()?;
    m.add_class::<PyQcReport>()?;
    m.add_class::<PyQcRule>()?;
    m.add_class::<PyQcIssue>()?;
    m.add_function(wrap_pyfunction!(run_qc_check, m)?)?;
    m.add_function(wrap_pyfunction!(list_qc_rules, m)?)?;
    m.add_function(wrap_pyfunction!(qc_report, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_preset_basic() {
        let preset = resolve_preset("basic");
        assert_eq!(preset, oximedia_qc::QcPreset::Basic);
    }

    #[test]
    fn test_resolve_preset_unknown() {
        let preset = resolve_preset("nonexistent");
        assert_eq!(preset, oximedia_qc::QcPreset::Comprehensive);
    }

    #[test]
    fn test_list_all_rules_not_empty() {
        let rules = list_all_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().any(|r| r.category == "video"));
        assert!(rules.iter().any(|r| r.category == "audio"));
    }

    #[test]
    fn test_qc_issue_repr() {
        let issue = PyQcIssue {
            rule_name: "test_rule".to_string(),
            passed: false,
            severity: "error".to_string(),
            message: "Something failed".to_string(),
            recommendation: Some("Fix it".to_string()),
        };
        let repr = issue.__repr__();
        assert!(repr.contains("test_rule"));
        assert!(repr.contains("false"));
    }

    #[test]
    fn test_qc_report_summary() {
        let report = PyQcReport {
            file_path: "test.mkv".to_string(),
            overall_passed: true,
            total_checks: 5,
            passed_checks: 5,
            failed_checks: 0,
            duration: Some(0.1),
            issues: Vec::new(),
        };
        let summary = report.summary();
        assert!(summary.contains("PASS"));
        assert!(summary.contains("5/5"));
    }
}
