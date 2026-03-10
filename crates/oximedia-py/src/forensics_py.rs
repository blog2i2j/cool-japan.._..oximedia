//! Python bindings for `oximedia-forensics` image/video forensic analysis.
//!
//! Provides `PyForensicConfig`, `PyForensicTest`, `PyForensicResult`,
//! `PyForensicAnalyzer`, and standalone convenience functions for tamper
//! detection from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use oximedia_forensics::{
    ConfidenceLevel, ForensicTest, ForensicsAnalyzer, ForensicsConfig, TamperingReport,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn confidence_level_str(level: ConfidenceLevel) -> String {
    match level {
        ConfidenceLevel::VeryLow => "very_low".to_string(),
        ConfidenceLevel::Low => "low".to_string(),
        ConfidenceLevel::Medium => "medium".to_string(),
        ConfidenceLevel::High => "high".to_string(),
        ConfidenceLevel::VeryHigh => "very_high".to_string(),
    }
}

fn forensic_test_to_py(test: &ForensicTest) -> PyForensicTest {
    PyForensicTest {
        name: test.name.clone(),
        tampering_detected: test.tampering_detected,
        confidence: test.confidence,
        confidence_level: confidence_level_str(test.confidence_level()),
        findings: test.findings.clone(),
    }
}

fn report_to_py(report: &TamperingReport) -> PyForensicResult {
    let tests: Vec<PyForensicTest> = report.tests.values().map(forensic_test_to_py).collect();

    PyForensicResult {
        tampering_detected: report.tampering_detected,
        overall_confidence: report.overall_confidence,
        summary: report.summary.clone(),
        recommendations: report.recommendations.clone(),
        tests,
    }
}

fn build_internal_config(py_cfg: &PyForensicConfig) -> ForensicsConfig {
    ForensicsConfig {
        enable_compression_analysis: py_cfg.enable_compression,
        enable_ela: py_cfg.enable_ela,
        enable_noise_analysis: py_cfg.enable_noise,
        enable_metadata_analysis: py_cfg.enable_metadata,
        enable_geometric_analysis: py_cfg.enable_geometric,
        enable_lighting_analysis: py_cfg.enable_lighting,
        min_confidence_threshold: py_cfg.min_confidence,
    }
}

// ---------------------------------------------------------------------------
// PyForensicConfig
// ---------------------------------------------------------------------------

/// Configuration for forensic analysis.
#[pyclass]
#[derive(Clone)]
pub struct PyForensicConfig {
    /// Enable compression artifact analysis.
    #[pyo3(get)]
    pub enable_compression: bool,
    /// Enable Error Level Analysis (ELA).
    #[pyo3(get)]
    pub enable_ela: bool,
    /// Enable noise pattern analysis.
    #[pyo3(get)]
    pub enable_noise: bool,
    /// Enable metadata analysis.
    #[pyo3(get)]
    pub enable_metadata: bool,
    /// Enable geometric (copy-move) analysis.
    #[pyo3(get)]
    pub enable_geometric: bool,
    /// Enable lighting inconsistency analysis.
    #[pyo3(get)]
    pub enable_lighting: bool,
    /// Minimum confidence threshold for reporting (0.0-1.0).
    #[pyo3(get)]
    pub min_confidence: f64,
}

#[pymethods]
impl PyForensicConfig {
    /// Create a new forensic configuration with all analyses enabled.
    #[new]
    fn new() -> Self {
        Self {
            enable_compression: true,
            enable_ela: true,
            enable_noise: true,
            enable_metadata: true,
            enable_geometric: true,
            enable_lighting: true,
            min_confidence: 0.3,
        }
    }

    /// Quick analysis config: only compression + ELA.
    #[classmethod]
    fn quick(_cls: &Bound<'_, PyType>) -> Self {
        Self {
            enable_compression: true,
            enable_ela: true,
            enable_noise: false,
            enable_metadata: false,
            enable_geometric: false,
            enable_lighting: false,
            min_confidence: 0.3,
        }
    }

    /// Full analysis config: all tests enabled.
    #[classmethod]
    fn full(_cls: &Bound<'_, PyType>) -> Self {
        Self::new()
    }

    /// Set the minimum confidence threshold.
    fn with_confidence_threshold(&mut self, threshold: f64) -> PyResult<()> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(PyValueError::new_err(format!(
                "threshold must be between 0.0 and 1.0, got {threshold}"
            )));
        }
        self.min_confidence = threshold;
        Ok(())
    }

    fn __repr__(&self) -> String {
        let enabled: Vec<&str> = [
            if self.enable_compression {
                Some("compression")
            } else {
                None
            },
            if self.enable_ela { Some("ela") } else { None },
            if self.enable_noise {
                Some("noise")
            } else {
                None
            },
            if self.enable_metadata {
                Some("metadata")
            } else {
                None
            },
            if self.enable_geometric {
                Some("geometric")
            } else {
                None
            },
            if self.enable_lighting {
                Some("lighting")
            } else {
                None
            },
        ]
        .iter()
        .filter_map(|x| *x)
        .collect();

        format!(
            "PyForensicConfig(tests=[{}], min_confidence={:.2})",
            enabled.join(", "),
            self.min_confidence,
        )
    }
}

// ---------------------------------------------------------------------------
// PyForensicTest
// ---------------------------------------------------------------------------

/// Result of a single forensic test.
#[pyclass]
#[derive(Clone)]
pub struct PyForensicTest {
    /// Name of the forensic test.
    #[pyo3(get)]
    pub name: String,
    /// Whether tampering was detected by this test.
    #[pyo3(get)]
    pub tampering_detected: bool,
    /// Confidence score (0.0-1.0).
    #[pyo3(get)]
    pub confidence: f64,
    /// Confidence level category (very_low/low/medium/high/very_high).
    #[pyo3(get)]
    pub confidence_level: String,
    /// Detailed findings from the test.
    #[pyo3(get)]
    pub findings: Vec<String>,
}

#[pymethods]
impl PyForensicTest {
    fn __repr__(&self) -> String {
        format!(
            "PyForensicTest(name='{}', tampering={}, confidence={:.3}, level='{}')",
            self.name, self.tampering_detected, self.confidence, self.confidence_level,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "name".to_string(),
            self.name.clone().into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "tampering_detected".to_string(),
            self.tampering_detected
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind(),
        );
        m.insert(
            "confidence".to_string(),
            self.confidence.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "confidence_level".to_string(),
            self.confidence_level
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        );
        m.insert(
            "findings".to_string(),
            self.findings.clone().into_pyobject(py)?.into_any().unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyForensicResult
// ---------------------------------------------------------------------------

/// Comprehensive forensic analysis result.
#[pyclass]
#[derive(Clone)]
pub struct PyForensicResult {
    /// Whether any tampering was detected overall.
    #[pyo3(get)]
    pub tampering_detected: bool,
    /// Overall confidence score (0.0-1.0).
    #[pyo3(get)]
    pub overall_confidence: f64,
    /// Human-readable summary of findings.
    #[pyo3(get)]
    pub summary: String,
    /// Recommended follow-up actions.
    #[pyo3(get)]
    pub recommendations: Vec<String>,
    tests: Vec<PyForensicTest>,
}

#[pymethods]
impl PyForensicResult {
    /// Get all individual test results.
    fn tests(&self) -> Vec<PyForensicTest> {
        self.tests.clone()
    }

    /// Get a test result by name (returns None if not found).
    fn test_by_name(&self, name: &str) -> Option<PyForensicTest> {
        self.tests.iter().find(|t| t.name == name).cloned()
    }

    /// Get the names of all tests that were run.
    fn test_names(&self) -> Vec<String> {
        self.tests.iter().map(|t| t.name.clone()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyForensicResult(tampering={}, confidence={:.3}, tests={})",
            self.tampering_detected,
            self.overall_confidence,
            self.tests.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyForensicAnalyzer
// ---------------------------------------------------------------------------

/// Forensic analyzer for detecting image/video tampering.
#[pyclass]
pub struct PyForensicAnalyzer {
    analyzer: ForensicsAnalyzer,
}

#[pymethods]
impl PyForensicAnalyzer {
    /// Create a new forensic analyzer.
    ///
    /// Args:
    ///     config: Optional configuration. If None, uses default (all tests enabled).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyForensicConfig>) -> PyResult<Self> {
        let analyzer = match config {
            Some(cfg) => ForensicsAnalyzer::with_config(build_internal_config(cfg)),
            None => ForensicsAnalyzer::new(),
        };
        Ok(Self { analyzer })
    }

    /// Perform Error Level Analysis on image data.
    ///
    /// Args:
    ///     image_data: Image file bytes (JPEG, PNG, etc. -- as loaded from disk).
    ///     width: Image width (informational, actual dimensions come from image data).
    ///     height: Image height (informational).
    ///
    /// Returns:
    ///     Forensic test result for ELA.
    #[pyo3(signature = (image_data, width=0, height=0))]
    fn ela(
        &self,
        image_data: Vec<u8>,
        #[allow(unused_variables)] width: u32,
        #[allow(unused_variables)] height: u32,
    ) -> PyResult<PyForensicTest> {
        // Run a full analysis with only ELA enabled
        let config = ForensicsConfig {
            enable_compression_analysis: false,
            enable_ela: true,
            enable_noise_analysis: false,
            enable_metadata_analysis: false,
            enable_geometric_analysis: false,
            enable_lighting_analysis: false,
            min_confidence_threshold: 0.0,
        };
        let analyzer = ForensicsAnalyzer::with_config(config);
        let report = analyzer
            .analyze(&image_data)
            .map_err(|e| PyRuntimeError::new_err(format!("ELA analysis failed: {e}")))?;

        report
            .tests
            .values()
            .next()
            .map(forensic_test_to_py)
            .ok_or_else(|| PyRuntimeError::new_err("ELA test produced no results"))
    }

    /// Analyze noise patterns in an image.
    ///
    /// Args:
    ///     image_data: Image file bytes.
    ///     width: Image width (informational).
    ///     height: Image height (informational).
    #[pyo3(signature = (image_data, width=0, height=0))]
    fn noise_analysis(
        &self,
        image_data: Vec<u8>,
        #[allow(unused_variables)] width: u32,
        #[allow(unused_variables)] height: u32,
    ) -> PyResult<PyForensicTest> {
        let config = ForensicsConfig {
            enable_compression_analysis: false,
            enable_ela: false,
            enable_noise_analysis: true,
            enable_metadata_analysis: false,
            enable_geometric_analysis: false,
            enable_lighting_analysis: false,
            min_confidence_threshold: 0.0,
        };
        let analyzer = ForensicsAnalyzer::with_config(config);
        let report = analyzer
            .analyze(&image_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Noise analysis failed: {e}")))?;

        report
            .tests
            .values()
            .next()
            .map(forensic_test_to_py)
            .ok_or_else(|| PyRuntimeError::new_err("Noise test produced no results"))
    }

    /// Analyze compression artifacts in an image.
    ///
    /// Args:
    ///     image_data: Image file bytes.
    ///     width: Image width (informational).
    ///     height: Image height (informational).
    #[pyo3(signature = (image_data, width=0, height=0))]
    fn compression_analysis(
        &self,
        image_data: Vec<u8>,
        #[allow(unused_variables)] width: u32,
        #[allow(unused_variables)] height: u32,
    ) -> PyResult<PyForensicTest> {
        let config = ForensicsConfig {
            enable_compression_analysis: true,
            enable_ela: false,
            enable_noise_analysis: false,
            enable_metadata_analysis: false,
            enable_geometric_analysis: false,
            enable_lighting_analysis: false,
            min_confidence_threshold: 0.0,
        };
        let analyzer = ForensicsAnalyzer::with_config(config);
        let report = analyzer
            .analyze(&image_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Compression analysis failed: {e}")))?;

        report
            .tests
            .values()
            .next()
            .map(forensic_test_to_py)
            .ok_or_else(|| PyRuntimeError::new_err("Compression test produced no results"))
    }

    /// Detect splicing (copy-move) in an image.
    ///
    /// Args:
    ///     image_data: Image file bytes.
    ///     width: Image width (informational).
    ///     height: Image height (informational).
    #[pyo3(signature = (image_data, width=0, height=0))]
    fn splicing_detection(
        &self,
        image_data: Vec<u8>,
        #[allow(unused_variables)] width: u32,
        #[allow(unused_variables)] height: u32,
    ) -> PyResult<PyForensicTest> {
        let config = ForensicsConfig {
            enable_compression_analysis: false,
            enable_ela: false,
            enable_noise_analysis: false,
            enable_metadata_analysis: false,
            enable_geometric_analysis: true,
            enable_lighting_analysis: false,
            min_confidence_threshold: 0.0,
        };
        let analyzer = ForensicsAnalyzer::with_config(config);
        let report = analyzer
            .analyze(&image_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Splicing detection failed: {e}")))?;

        report
            .tests
            .values()
            .next()
            .map(forensic_test_to_py)
            .ok_or_else(|| PyRuntimeError::new_err("Splicing test produced no results"))
    }

    /// Run a comprehensive forensic report on an image.
    ///
    /// All analyses configured on this analyzer will be executed.
    ///
    /// Args:
    ///     image_data: Image file bytes.
    ///     width: Image width (informational).
    ///     height: Image height (informational).
    #[pyo3(signature = (image_data, width=0, height=0))]
    fn full_report(
        &self,
        image_data: Vec<u8>,
        #[allow(unused_variables)] width: u32,
        #[allow(unused_variables)] height: u32,
    ) -> PyResult<PyForensicResult> {
        let report = self
            .analyzer
            .analyze(&image_data)
            .map_err(|e| PyRuntimeError::new_err(format!("Forensic analysis failed: {e}")))?;

        Ok(report_to_py(&report))
    }

    fn __repr__(&self) -> String {
        let cfg = self.analyzer.config();
        let mut enabled = Vec::new();
        if cfg.enable_compression_analysis {
            enabled.push("compression");
        }
        if cfg.enable_ela {
            enabled.push("ela");
        }
        if cfg.enable_noise_analysis {
            enabled.push("noise");
        }
        if cfg.enable_metadata_analysis {
            enabled.push("metadata");
        }
        if cfg.enable_geometric_analysis {
            enabled.push("geometric");
        }
        if cfg.enable_lighting_analysis {
            enabled.push("lighting");
        }
        format!("PyForensicAnalyzer(tests=[{}])", enabled.join(", "))
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Run a full forensic analysis on image data.
///
/// Args:
///     image_data: Image file bytes (JPEG, PNG, etc.).
///     width: Image width (informational).
///     height: Image height (informational).
///     config: Optional configuration.
///
/// Returns:
///     Comprehensive forensic result.
#[pyfunction]
#[pyo3(signature = (image_data, width=0, height=0, config=None))]
pub fn forensic_report(
    image_data: Vec<u8>,
    #[allow(unused_variables)] width: u32,
    #[allow(unused_variables)] height: u32,
    config: Option<&PyForensicConfig>,
) -> PyResult<PyForensicResult> {
    let analyzer = match config {
        Some(cfg) => ForensicsAnalyzer::with_config(build_internal_config(cfg)),
        None => ForensicsAnalyzer::new(),
    };
    let report = analyzer
        .analyze(&image_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Forensic analysis failed: {e}")))?;

    Ok(report_to_py(&report))
}

/// Quick integrity check on image data.
///
/// Returns True if no tampering is detected with high confidence.
///
/// Args:
///     image_data: Image file bytes.
///     width: Image width (informational).
///     height: Image height (informational).
#[pyfunction]
#[pyo3(signature = (image_data, width=0, height=0))]
pub fn check_integrity(
    image_data: Vec<u8>,
    #[allow(unused_variables)] width: u32,
    #[allow(unused_variables)] height: u32,
) -> PyResult<bool> {
    let analyzer = ForensicsAnalyzer::new();
    let report = analyzer
        .analyze(&image_data)
        .map_err(|e| PyRuntimeError::new_err(format!("Integrity check failed: {e}")))?;

    Ok(!report.tampering_detected)
}

/// Perform Error Level Analysis and return the ELA difference image bytes.
///
/// The returned bytes are a raw grayscale image (width * height bytes)
/// representing the error levels. Higher values indicate more difference
/// from the recompressed version.
///
/// Args:
///     image_data: Image file bytes.
///     width: Image width (informational).
///     height: Image height (informational).
///     quality: JPEG recompression quality (1-100, default: 90).
#[pyfunction]
#[pyo3(signature = (image_data, width=0, height=0, quality=None))]
pub fn ela_analysis(
    image_data: Vec<u8>,
    #[allow(unused_variables)] width: u32,
    #[allow(unused_variables)] height: u32,
    #[allow(unused_variables)] quality: Option<u32>,
) -> PyResult<Vec<u8>> {
    // Run ELA and extract the anomaly map as grayscale bytes
    let config = ForensicsConfig {
        enable_compression_analysis: false,
        enable_ela: true,
        enable_noise_analysis: false,
        enable_metadata_analysis: false,
        enable_geometric_analysis: false,
        enable_lighting_analysis: false,
        min_confidence_threshold: 0.0,
    };
    let analyzer = ForensicsAnalyzer::with_config(config);
    let report = analyzer
        .analyze(&image_data)
        .map_err(|e| PyRuntimeError::new_err(format!("ELA analysis failed: {e}")))?;

    // Look for the ELA test result and its anomaly map
    if let Some(test) = report.tests.values().next() {
        if let Some(ref map) = test.anomaly_map {
            // Convert the f64 anomaly map to u8 bytes (normalized to 0-255)
            let max_val = map.iter().cloned().fold(0.0_f64, f64::max);
            let scale = if max_val > 0.0 { 255.0 / max_val } else { 0.0 };
            let bytes: Vec<u8> = map
                .iter()
                .map(|&v| (v * scale).round().min(255.0).max(0.0) as u8)
                .collect();
            return Ok(bytes);
        }
    }

    // If no anomaly map was produced, return an empty result
    // (image may have been too small or parse failed gracefully)
    Err(PyRuntimeError::new_err(
        "ELA analysis did not produce an anomaly map",
    ))
}

// ---------------------------------------------------------------------------
// Registration helper
// ---------------------------------------------------------------------------

/// Register all forensics bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForensicConfig>()?;
    m.add_class::<PyForensicTest>()?;
    m.add_class::<PyForensicResult>()?;
    m.add_class::<PyForensicAnalyzer>()?;
    m.add_function(wrap_pyfunction!(forensic_report, m)?)?;
    m.add_function(wrap_pyfunction!(check_integrity, m)?)?;
    m.add_function(wrap_pyfunction!(ela_analysis, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = PyForensicConfig::new();
        assert!(cfg.enable_compression);
        assert!(cfg.enable_ela);
        assert!(cfg.enable_noise);
        assert!(cfg.enable_metadata);
        assert!(cfg.enable_geometric);
        assert!(cfg.enable_lighting);
        assert!((cfg.min_confidence - 0.3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_threshold_validation() {
        let mut cfg = PyForensicConfig::new();
        assert!(cfg.with_confidence_threshold(0.5).is_ok());
        assert!(cfg.with_confidence_threshold(-0.1).is_err());
        assert!(cfg.with_confidence_threshold(1.5).is_err());
    }

    #[test]
    fn test_forensic_test_repr() {
        let t = PyForensicTest {
            name: "ELA".to_string(),
            tampering_detected: false,
            confidence: 0.25,
            confidence_level: "low".to_string(),
            findings: vec!["No issues".to_string()],
        };
        let repr = t.__repr__();
        assert!(repr.contains("ELA"));
        assert!(repr.contains("0.250"));
    }

    #[test]
    fn test_forensic_result_test_names() {
        let result = PyForensicResult {
            tampering_detected: false,
            overall_confidence: 0.1,
            summary: "Clean".to_string(),
            recommendations: Vec::new(),
            tests: vec![
                PyForensicTest {
                    name: "Compression".to_string(),
                    tampering_detected: false,
                    confidence: 0.1,
                    confidence_level: "very_low".to_string(),
                    findings: Vec::new(),
                },
                PyForensicTest {
                    name: "ELA".to_string(),
                    tampering_detected: false,
                    confidence: 0.15,
                    confidence_level: "very_low".to_string(),
                    findings: Vec::new(),
                },
            ],
        };
        let names = result.test_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"Compression".to_string()));
        assert!(names.contains(&"ELA".to_string()));
    }

    #[test]
    fn test_forensic_result_test_by_name() {
        let result = PyForensicResult {
            tampering_detected: false,
            overall_confidence: 0.1,
            summary: "Clean".to_string(),
            recommendations: Vec::new(),
            tests: vec![PyForensicTest {
                name: "Noise".to_string(),
                tampering_detected: true,
                confidence: 0.7,
                confidence_level: "high".to_string(),
                findings: vec!["Pattern detected".to_string()],
            }],
        };
        let noise = result.test_by_name("Noise");
        assert!(noise.is_some());
        let n = noise.expect("should find noise test");
        assert!(n.tampering_detected);
        assert!(result.test_by_name("Missing").is_none());
    }

    #[test]
    fn test_build_internal_config() {
        let py_cfg = PyForensicConfig {
            enable_compression: true,
            enable_ela: false,
            enable_noise: true,
            enable_metadata: false,
            enable_geometric: false,
            enable_lighting: true,
            min_confidence: 0.5,
        };
        let ic = build_internal_config(&py_cfg);
        assert!(ic.enable_compression_analysis);
        assert!(!ic.enable_ela);
        assert!(ic.enable_noise_analysis);
        assert!(!ic.enable_metadata_analysis);
        assert!(!ic.enable_geometric_analysis);
        assert!(ic.enable_lighting_analysis);
        assert!((ic.min_confidence_threshold - 0.5).abs() < f64::EPSILON);
    }
}
