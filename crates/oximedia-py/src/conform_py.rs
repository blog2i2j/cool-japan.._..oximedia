//! Python bindings for `oximedia-conform` delivery specification checking.
//!
//! Provides `PyDeliverySpec`, `PyConformCheck`, `PyConformReport`, and
//! `PyConformChecker` for validating media properties against delivery specs.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyDeliverySpec
// ---------------------------------------------------------------------------

/// A delivery specification defining acceptable media properties.
#[pyclass]
#[derive(Clone)]
pub struct PyDeliverySpec {
    /// Spec name.
    #[pyo3(get)]
    pub name: String,
    /// Human-readable description.
    #[pyo3(get)]
    pub description: String,
    /// Required video codec (if any).
    #[pyo3(get)]
    pub video_codec: Option<String>,
    /// Required audio codec (if any).
    #[pyo3(get)]
    pub audio_codec: Option<String>,
    /// Required container format (if any).
    #[pyo3(get)]
    pub container: Option<String>,
    /// Minimum acceptable width.
    #[pyo3(get)]
    pub min_width: Option<u32>,
    /// Maximum acceptable width.
    #[pyo3(get)]
    pub max_width: Option<u32>,
    /// Minimum acceptable height.
    #[pyo3(get)]
    pub min_height: Option<u32>,
    /// Maximum acceptable height.
    #[pyo3(get)]
    pub max_height: Option<u32>,
    /// Allowed frame rates.
    #[pyo3(get)]
    pub frame_rates: Vec<f64>,
    /// Allowed sample rates.
    #[pyo3(get)]
    pub sample_rates: Vec<u32>,
    /// Maximum integrated loudness (LUFS).
    #[pyo3(get)]
    pub max_loudness_lufs: Option<f64>,
    /// Maximum true peak (dBTP).
    #[pyo3(get)]
    pub max_true_peak_dbtp: Option<f64>,
    /// Maximum video bitrate (kbps).
    #[pyo3(get)]
    pub max_bitrate_kbps: Option<u32>,
    /// Maximum file size (MB).
    #[pyo3(get)]
    pub max_file_size_mb: Option<u64>,
}

#[pymethods]
impl PyDeliverySpec {
    /// Broadcast HD delivery spec.
    #[classmethod]
    fn broadcast_hd(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            name: "broadcast_hd".to_string(),
            description: "Broadcast HD delivery (EBU R128)".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(1920),
            max_width: Some(1920),
            min_height: Some(1080),
            max_height: Some(1080),
            frame_rates: vec![25.0, 29.97, 30.0, 50.0, 59.94],
            sample_rates: vec![48000],
            max_loudness_lufs: Some(-23.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(50_000),
            max_file_size_mb: None,
        }
    }

    /// Netflix delivery spec.
    #[classmethod]
    fn netflix(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            name: "netflix".to_string(),
            description: "Netflix streaming delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(1920),
            max_width: Some(3840),
            min_height: Some(1080),
            max_height: Some(2160),
            frame_rates: vec![23.976, 24.0, 25.0, 29.97],
            sample_rates: vec![48000],
            max_loudness_lufs: Some(-27.0),
            max_true_peak_dbtp: Some(-2.0),
            max_bitrate_kbps: Some(80_000),
            max_file_size_mb: None,
        }
    }

    /// YouTube delivery spec.
    #[classmethod]
    fn youtube(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            name: "youtube".to_string(),
            description: "YouTube streaming delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: Some("mp4".to_string()),
            min_width: Some(426),
            max_width: Some(3840),
            min_height: Some(240),
            max_height: Some(2160),
            frame_rates: vec![24.0, 25.0, 30.0, 48.0, 50.0, 60.0],
            sample_rates: vec![44100, 48000],
            max_loudness_lufs: Some(-14.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(85_000),
            max_file_size_mb: Some(256_000),
        }
    }

    /// Theatrical DCP delivery spec.
    #[classmethod]
    fn theatrical_dcp(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            name: "theatrical_dcp".to_string(),
            description: "Digital Cinema Package delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(2048),
            max_width: Some(4096),
            min_height: Some(858),
            max_height: Some(2160),
            frame_rates: vec![24.0, 25.0, 30.0, 48.0],
            sample_rates: vec![48000, 96000],
            max_loudness_lufs: Some(-20.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(250_000),
            max_file_size_mb: None,
        }
    }

    /// Podcast audio delivery spec.
    #[classmethod]
    fn podcast(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            name: "podcast".to_string(),
            description: "Podcast audio delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            frame_rates: Vec::new(),
            sample_rates: vec![44100, 48000],
            max_loudness_lufs: Some(-16.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(320),
            max_file_size_mb: Some(500),
        }
    }

    /// Custom delivery spec with the given name.
    #[classmethod]
    fn custom(_cls: &Bound<'_, PyType>, name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Custom spec: {name}"),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            frame_rates: Vec::new(),
            sample_rates: Vec::new(),
            max_loudness_lufs: None,
            max_true_peak_dbtp: None,
            max_bitrate_kbps: None,
            max_file_size_mb: None,
        }
    }

    /// Set the acceptable resolution range.
    fn with_resolution_range(
        &mut self,
        min_w: u32,
        min_h: u32,
        max_w: u32,
        max_h: u32,
    ) -> PyResult<()> {
        if min_w > max_w || min_h > max_h {
            return Err(PyValueError::new_err(
                "min values must not exceed max values",
            ));
        }
        self.min_width = Some(min_w);
        self.min_height = Some(min_h);
        self.max_width = Some(max_w);
        self.max_height = Some(max_h);
        Ok(())
    }

    /// Set the allowed frame rates.
    fn with_frame_rates(&mut self, rates: Vec<f64>) -> PyResult<()> {
        for &r in &rates {
            if r <= 0.0 {
                return Err(PyValueError::new_err(format!(
                    "Frame rate must be > 0, got {r}"
                )));
            }
        }
        self.frame_rates = rates;
        Ok(())
    }

    /// Set loudness constraints.
    fn with_loudness(&mut self, max_lufs: f64, max_tp: f64) {
        self.max_loudness_lufs = Some(max_lufs);
        self.max_true_peak_dbtp = Some(max_tp);
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDeliverySpec(name='{}', res={}x{}-{}x{}, fps={:?})",
            self.name,
            self.min_width.unwrap_or(0),
            self.min_height.unwrap_or(0),
            self.max_width.unwrap_or(0),
            self.max_height.unwrap_or(0),
            self.frame_rates,
        )
    }
}

// ---------------------------------------------------------------------------
// PyConformCheck
// ---------------------------------------------------------------------------

/// A single conformance check result.
#[pyclass]
#[derive(Clone)]
pub struct PyConformCheck {
    /// Name of the check.
    #[pyo3(get)]
    pub check_name: String,
    /// Whether the check passed.
    #[pyo3(get)]
    pub passed: bool,
    /// Actual value found.
    #[pyo3(get)]
    pub actual_value: String,
    /// Expected value or range.
    #[pyo3(get)]
    pub expected_value: String,
    /// Severity: "error", "warning", or "info".
    #[pyo3(get)]
    pub severity: String,
}

#[pymethods]
impl PyConformCheck {
    fn __repr__(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        format!(
            "PyConformCheck({}: {} [{}] actual={}, expected={})",
            self.check_name, status, self.severity, self.actual_value, self.expected_value,
        )
    }
}

// ---------------------------------------------------------------------------
// PyConformReport
// ---------------------------------------------------------------------------

/// A complete conformance report with multiple checks.
#[pyclass]
#[derive(Clone)]
pub struct PyConformReport {
    /// Spec name that was checked against.
    #[pyo3(get)]
    pub spec_name: String,
    /// Whether the overall check passed.
    #[pyo3(get)]
    pub overall_pass: bool,
    /// Number of errors.
    #[pyo3(get)]
    pub error_count: u32,
    /// Number of warnings.
    #[pyo3(get)]
    pub warning_count: u32,
    /// All checks performed.
    checks: Vec<PyConformCheck>,
}

#[pymethods]
impl PyConformReport {
    /// Get all checks.
    fn checks(&self) -> Vec<PyConformCheck> {
        self.checks.clone()
    }

    /// Get only failed checks.
    fn failed_checks(&self) -> Vec<PyConformCheck> {
        self.checks.iter().filter(|c| !c.passed).cloned().collect()
    }

    /// Get only warning checks.
    fn warnings(&self) -> Vec<PyConformCheck> {
        self.checks
            .iter()
            .filter(|c| c.severity == "warning")
            .cloned()
            .collect()
    }

    /// Get a human-readable summary.
    fn summary(&self) -> String {
        let status = if self.overall_pass {
            "PASSED"
        } else {
            "FAILED"
        };
        format!(
            "Conform check '{}': {} ({} checks, {} errors, {} warnings)",
            self.spec_name,
            status,
            self.checks.len(),
            self.error_count,
            self.warning_count,
        )
    }

    fn __repr__(&self) -> String {
        self.summary()
    }
}

// ---------------------------------------------------------------------------
// PyConformChecker
// ---------------------------------------------------------------------------

/// Checker that validates media properties against a delivery spec.
#[pyclass]
pub struct PyConformChecker {
    spec: PyDeliverySpec,
}

#[pymethods]
impl PyConformChecker {
    /// Create a new conformance checker with the given spec.
    #[new]
    fn new(spec: PyDeliverySpec) -> Self {
        Self { spec }
    }

    /// Check a file path for conformance.
    ///
    /// This performs basic file-level checks (existence, size) and
    /// returns a report. For detailed media property checks, use
    /// `check_properties` with a dict of properties.
    fn check_file(&self, path: &str) -> PyResult<PyConformReport> {
        let metadata = std::fs::metadata(path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot access file: {e}")))?;

        let mut checks = Vec::new();
        let file_size_mb = metadata.len() / (1024 * 1024);

        // File size check
        if let Some(max_mb) = self.spec.max_file_size_mb {
            let passed = file_size_mb <= max_mb;
            checks.push(PyConformCheck {
                check_name: "file_size".to_string(),
                passed,
                actual_value: format!("{file_size_mb} MB"),
                expected_value: format!("<= {max_mb} MB"),
                severity: if passed {
                    "info".to_string()
                } else {
                    "error".to_string()
                },
            });
        }

        // File exists check (always passes since we got metadata)
        checks.push(PyConformCheck {
            check_name: "file_exists".to_string(),
            passed: true,
            actual_value: "exists".to_string(),
            expected_value: "exists".to_string(),
            severity: "info".to_string(),
        });

        build_report(&self.spec.name, &checks)
    }

    /// Check a dictionary of media properties against the spec.
    ///
    /// Supported keys: "width", "height", "frame_rate", "sample_rate",
    /// "video_codec", "audio_codec", "container", "bitrate_kbps",
    /// "loudness_lufs", "true_peak_dbtp", "file_size_mb".
    fn check_properties(&self, properties: HashMap<String, String>) -> PyResult<PyConformReport> {
        let checks = run_property_checks(&self.spec, &properties);
        build_report(&self.spec.name, &checks)
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Check a file path against a named delivery spec.
#[pyfunction]
pub fn check_conform(path: &str, spec_name: &str) -> PyResult<PyConformReport> {
    let spec = get_spec_by_name(spec_name)?;
    let checker = PyConformChecker { spec };
    checker.check_file(path)
}

/// List all built-in delivery specs.
#[pyfunction]
pub fn list_delivery_specs() -> Vec<PyDeliverySpec> {
    builtin_specs()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn get_spec_by_name(name: &str) -> PyResult<PyDeliverySpec> {
    builtin_specs()
        .into_iter()
        .find(|s| s.name == name)
        .ok_or_else(|| {
            PyValueError::new_err(format!(
                "Spec '{}' not found. Available: {}",
                name,
                builtin_specs()
                    .iter()
                    .map(|s| s.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

fn builtin_specs() -> Vec<PyDeliverySpec> {
    // We build them manually to avoid needing Bound<'_, PyType>
    vec![
        PyDeliverySpec {
            name: "broadcast_hd".to_string(),
            description: "Broadcast HD delivery (EBU R128)".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(1920),
            max_width: Some(1920),
            min_height: Some(1080),
            max_height: Some(1080),
            frame_rates: vec![25.0, 29.97, 30.0, 50.0, 59.94],
            sample_rates: vec![48000],
            max_loudness_lufs: Some(-23.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(50_000),
            max_file_size_mb: None,
        },
        PyDeliverySpec {
            name: "netflix".to_string(),
            description: "Netflix streaming delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(1920),
            max_width: Some(3840),
            min_height: Some(1080),
            max_height: Some(2160),
            frame_rates: vec![23.976, 24.0, 25.0, 29.97],
            sample_rates: vec![48000],
            max_loudness_lufs: Some(-27.0),
            max_true_peak_dbtp: Some(-2.0),
            max_bitrate_kbps: Some(80_000),
            max_file_size_mb: None,
        },
        PyDeliverySpec {
            name: "youtube".to_string(),
            description: "YouTube streaming delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: Some("mp4".to_string()),
            min_width: Some(426),
            max_width: Some(3840),
            min_height: Some(240),
            max_height: Some(2160),
            frame_rates: vec![24.0, 25.0, 30.0, 48.0, 50.0, 60.0],
            sample_rates: vec![44100, 48000],
            max_loudness_lufs: Some(-14.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(85_000),
            max_file_size_mb: Some(256_000),
        },
        PyDeliverySpec {
            name: "theatrical_dcp".to_string(),
            description: "Digital Cinema Package delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: Some(2048),
            max_width: Some(4096),
            min_height: Some(858),
            max_height: Some(2160),
            frame_rates: vec![24.0, 25.0, 30.0, 48.0],
            sample_rates: vec![48000, 96000],
            max_loudness_lufs: Some(-20.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(250_000),
            max_file_size_mb: None,
        },
        PyDeliverySpec {
            name: "podcast".to_string(),
            description: "Podcast audio delivery".to_string(),
            video_codec: None,
            audio_codec: None,
            container: None,
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            frame_rates: Vec::new(),
            sample_rates: vec![44100, 48000],
            max_loudness_lufs: Some(-16.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(320),
            max_file_size_mb: Some(500),
        },
    ]
}

fn run_property_checks(
    spec: &PyDeliverySpec,
    props: &HashMap<String, String>,
) -> Vec<PyConformCheck> {
    let mut checks = Vec::new();

    // Width check
    if let Some(w_str) = props.get("width") {
        if let Ok(w) = w_str.parse::<u32>() {
            if let Some(min_w) = spec.min_width {
                let passed = w >= min_w;
                checks.push(PyConformCheck {
                    check_name: "min_width".to_string(),
                    passed,
                    actual_value: w.to_string(),
                    expected_value: format!(">= {min_w}"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
            if let Some(max_w) = spec.max_width {
                let passed = w <= max_w;
                checks.push(PyConformCheck {
                    check_name: "max_width".to_string(),
                    passed,
                    actual_value: w.to_string(),
                    expected_value: format!("<= {max_w}"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // Height check
    if let Some(h_str) = props.get("height") {
        if let Ok(h) = h_str.parse::<u32>() {
            if let Some(min_h) = spec.min_height {
                let passed = h >= min_h;
                checks.push(PyConformCheck {
                    check_name: "min_height".to_string(),
                    passed,
                    actual_value: h.to_string(),
                    expected_value: format!(">= {min_h}"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
            if let Some(max_h) = spec.max_height {
                let passed = h <= max_h;
                checks.push(PyConformCheck {
                    check_name: "max_height".to_string(),
                    passed,
                    actual_value: h.to_string(),
                    expected_value: format!("<= {max_h}"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // Frame rate check
    if let Some(fps_str) = props.get("frame_rate") {
        if let Ok(fps) = fps_str.parse::<f64>() {
            if !spec.frame_rates.is_empty() {
                let passed = spec
                    .frame_rates
                    .iter()
                    .any(|&allowed| (allowed - fps).abs() < 0.01);
                checks.push(PyConformCheck {
                    check_name: "frame_rate".to_string(),
                    passed,
                    actual_value: format!("{fps:.3}"),
                    expected_value: format!("{:?}", spec.frame_rates),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // Sample rate check
    if let Some(sr_str) = props.get("sample_rate") {
        if let Ok(sr) = sr_str.parse::<u32>() {
            if !spec.sample_rates.is_empty() {
                let passed = spec.sample_rates.contains(&sr);
                checks.push(PyConformCheck {
                    check_name: "sample_rate".to_string(),
                    passed,
                    actual_value: sr.to_string(),
                    expected_value: format!("{:?}", spec.sample_rates),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // Container check
    if let Some(ct) = props.get("container") {
        if let Some(required_ct) = &spec.container {
            let passed = ct == required_ct;
            checks.push(PyConformCheck {
                check_name: "container".to_string(),
                passed,
                actual_value: ct.clone(),
                expected_value: required_ct.clone(),
                severity: if passed {
                    "info".to_string()
                } else {
                    "error".to_string()
                },
            });
        }
    }

    // Bitrate check
    if let Some(br_str) = props.get("bitrate_kbps") {
        if let Ok(br) = br_str.parse::<u32>() {
            if let Some(max_br) = spec.max_bitrate_kbps {
                let passed = br <= max_br;
                checks.push(PyConformCheck {
                    check_name: "bitrate".to_string(),
                    passed,
                    actual_value: format!("{br} kbps"),
                    expected_value: format!("<= {max_br} kbps"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // Loudness check
    if let Some(lufs_str) = props.get("loudness_lufs") {
        if let Ok(lufs) = lufs_str.parse::<f64>() {
            if let Some(max_lufs) = spec.max_loudness_lufs {
                // Loudness should be at or below the target (more negative = quieter)
                let passed = lufs <= max_lufs;
                checks.push(PyConformCheck {
                    check_name: "loudness".to_string(),
                    passed,
                    actual_value: format!("{lufs:.1} LUFS"),
                    expected_value: format!("<= {max_lufs:.1} LUFS"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    // True peak check
    if let Some(tp_str) = props.get("true_peak_dbtp") {
        if let Ok(tp) = tp_str.parse::<f64>() {
            if let Some(max_tp) = spec.max_true_peak_dbtp {
                let passed = tp <= max_tp;
                checks.push(PyConformCheck {
                    check_name: "true_peak".to_string(),
                    passed,
                    actual_value: format!("{tp:.1} dBTP"),
                    expected_value: format!("<= {max_tp:.1} dBTP"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "warning".to_string()
                    },
                });
            }
        }
    }

    // File size check
    if let Some(fs_str) = props.get("file_size_mb") {
        if let Ok(fs) = fs_str.parse::<u64>() {
            if let Some(max_fs) = spec.max_file_size_mb {
                let passed = fs <= max_fs;
                checks.push(PyConformCheck {
                    check_name: "file_size".to_string(),
                    passed,
                    actual_value: format!("{fs} MB"),
                    expected_value: format!("<= {max_fs} MB"),
                    severity: if passed {
                        "info".to_string()
                    } else {
                        "error".to_string()
                    },
                });
            }
        }
    }

    checks
}

fn build_report(spec_name: &str, checks: &[PyConformCheck]) -> PyResult<PyConformReport> {
    let error_count = checks
        .iter()
        .filter(|c| !c.passed && c.severity == "error")
        .count() as u32;
    let warning_count = checks
        .iter()
        .filter(|c| c.severity == "warning" && !c.passed)
        .count() as u32;
    let overall_pass = error_count == 0;

    Ok(PyConformReport {
        spec_name: spec_name.to_string(),
        overall_pass,
        error_count,
        warning_count,
        checks: checks.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all conform bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDeliverySpec>()?;
    m.add_class::<PyConformCheck>()?;
    m.add_class::<PyConformReport>()?;
    m.add_class::<PyConformChecker>()?;
    m.add_function(wrap_pyfunction!(check_conform, m)?)?;
    m.add_function(wrap_pyfunction!(list_delivery_specs, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_specs_count() {
        let specs = builtin_specs();
        assert_eq!(specs.len(), 5);
    }

    #[test]
    fn test_property_checks_pass() {
        let spec = builtin_specs()
            .into_iter()
            .find(|s| s.name == "youtube")
            .expect("should find youtube spec");

        let mut props = HashMap::new();
        props.insert("width".to_string(), "1920".to_string());
        props.insert("height".to_string(), "1080".to_string());
        props.insert("frame_rate".to_string(), "30.0".to_string());
        props.insert("sample_rate".to_string(), "48000".to_string());

        let checks = run_property_checks(&spec, &props);
        assert!(checks.iter().all(|c| c.passed));
    }

    #[test]
    fn test_property_checks_fail_resolution() {
        let spec = builtin_specs()
            .into_iter()
            .find(|s| s.name == "broadcast_hd")
            .expect("should find broadcast_hd spec");

        let mut props = HashMap::new();
        props.insert("width".to_string(), "1280".to_string());
        props.insert("height".to_string(), "720".to_string());

        let checks = run_property_checks(&spec, &props);
        let failed = checks.iter().filter(|c| !c.passed).count();
        assert!(failed > 0);
    }

    #[test]
    fn test_build_report() {
        let checks = vec![
            PyConformCheck {
                check_name: "test_pass".to_string(),
                passed: true,
                actual_value: "ok".to_string(),
                expected_value: "ok".to_string(),
                severity: "info".to_string(),
            },
            PyConformCheck {
                check_name: "test_fail".to_string(),
                passed: false,
                actual_value: "bad".to_string(),
                expected_value: "good".to_string(),
                severity: "error".to_string(),
            },
        ];
        let report = build_report("test", &checks);
        assert!(report.is_ok());
        let r = report.expect("should build report");
        assert!(!r.overall_pass);
        assert_eq!(r.error_count, 1);
    }

    #[test]
    fn test_conform_report_methods() {
        let checks = vec![
            PyConformCheck {
                check_name: "a".to_string(),
                passed: true,
                actual_value: "1".to_string(),
                expected_value: "1".to_string(),
                severity: "info".to_string(),
            },
            PyConformCheck {
                check_name: "b".to_string(),
                passed: false,
                actual_value: "2".to_string(),
                expected_value: "3".to_string(),
                severity: "warning".to_string(),
            },
        ];
        let report = build_report("test", &checks).expect("should build");
        assert!(report.overall_pass); // only warnings, no errors
        assert_eq!(report.warning_count, 1);
        assert_eq!(report.failed_checks().len(), 1);
        assert_eq!(report.warnings().len(), 1);
        assert!(report.summary().contains("PASSED"));
    }
}
