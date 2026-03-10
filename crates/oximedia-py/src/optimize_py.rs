//! Python bindings for `oximedia-optimize` codec optimization.
//!
//! Provides `PyComplexityReport`, `PyCrfResult`, `PyOptimizer`, and
//! standalone convenience functions for media optimization from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// PyComplexityReport
// ---------------------------------------------------------------------------

/// Report of content complexity analysis.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyComplexityReport {
    /// Overall complexity score (0.0-1.0).
    #[pyo3(get)]
    pub overall_complexity: f64,
    /// Spatial complexity score (0.0-1.0).
    #[pyo3(get)]
    pub spatial_complexity: f64,
    /// Temporal complexity score (0.0-1.0).
    #[pyo3(get)]
    pub temporal_complexity: f64,
    /// Estimated number of scenes.
    #[pyo3(get)]
    pub scene_count: u32,
    /// Estimated average bitrate needed (kbps).
    #[pyo3(get)]
    pub avg_bitrate_needed: u32,
    /// Recommended CRF value.
    #[pyo3(get)]
    pub recommended_crf: u32,
    /// Recommended ladder strategy.
    #[pyo3(get)]
    pub recommended_ladder: String,
}

#[pymethods]
impl PyComplexityReport {
    fn __repr__(&self) -> String {
        format!(
            "PyComplexityReport(overall={:.2}, spatial={:.2}, temporal={:.2}, \
             scenes={}, crf={}, bitrate={}kbps)",
            self.overall_complexity,
            self.spatial_complexity,
            self.temporal_complexity,
            self.scene_count,
            self.recommended_crf,
            self.avg_bitrate_needed,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "overall_complexity".to_string(),
                self.overall_complexity
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "spatial_complexity".to_string(),
                self.spatial_complexity
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "temporal_complexity".to_string(),
                self.temporal_complexity
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "scene_count".to_string(),
                self.scene_count
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "avg_bitrate_needed".to_string(),
                self.avg_bitrate_needed
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "recommended_crf".to_string(),
                self.recommended_crf
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "recommended_ladder".to_string(),
                self.recommended_ladder
                    .clone()
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m
        })
    }
}

// ---------------------------------------------------------------------------
// PyCrfResult
// ---------------------------------------------------------------------------

/// Result of a single CRF probe encoding.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyCrfResult {
    /// CRF value used.
    #[pyo3(get)]
    pub crf: u32,
    /// Estimated file size in bytes.
    #[pyo3(get)]
    pub file_size: u64,
    /// Estimated bitrate in kbps.
    #[pyo3(get)]
    pub bitrate_kbps: u64,
    /// Estimated PSNR (dB).
    #[pyo3(get)]
    pub psnr: f64,
    /// Estimated SSIM (0-1).
    #[pyo3(get)]
    pub ssim: f64,
    /// Estimated VMAF (0-100).
    #[pyo3(get)]
    pub vmaf: f64,
}

#[pymethods]
impl PyCrfResult {
    fn __repr__(&self) -> String {
        format!(
            "PyCrfResult(crf={}, size={}B, bitrate={}kbps, psnr={:.2}, ssim={:.4}, vmaf={:.1})",
            self.crf, self.file_size, self.bitrate_kbps, self.psnr, self.ssim, self.vmaf,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self) -> HashMap<String, Py<PyAny>> {
        Python::attach(|py| {
            let mut m = HashMap::new();
            m.insert(
                "crf".to_string(),
                self.crf
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "file_size".to_string(),
                self.file_size
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "bitrate_kbps".to_string(),
                self.bitrate_kbps
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "psnr".to_string(),
                self.psnr
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "ssim".to_string(),
                self.ssim
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m.insert(
                "vmaf".to_string(),
                self.vmaf
                    .into_pyobject(py)
                    .map(|o| o.into_any().unbind())
                    .unwrap_or_else(|_| py.None()),
            );
            m
        })
    }
}

// ---------------------------------------------------------------------------
// PyOptimizer
// ---------------------------------------------------------------------------

/// Codec optimizer for analyzing content and recommending encoding parameters.
#[pyclass]
pub struct PyOptimizer {
    codec: String,
    preset: String,
}

#[pymethods]
impl PyOptimizer {
    /// Create a new optimizer for the specified codec.
    ///
    /// Args:
    ///     codec: Codec name (av1, vp9, vp8). Default: "av1".
    ///     preset: Encoder preset (fast, medium, slow). Default: "medium".
    #[new]
    #[pyo3(signature = (codec=None, preset=None))]
    fn new(codec: Option<&str>, preset: Option<&str>) -> PyResult<Self> {
        let c = codec.unwrap_or("av1");
        let p = preset.unwrap_or("medium");

        match c {
            "av1" | "vp9" | "vp8" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported codec '{}'. Supported: av1, vp9, vp8",
                    other
                )));
            }
        }

        match p {
            "fast" | "medium" | "slow" | "placebo" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown preset '{}'. Expected: fast, medium, slow, placebo",
                    other
                )));
            }
        }

        Ok(Self {
            codec: c.to_string(),
            preset: p.to_string(),
        })
    }

    /// Analyze the complexity of a media file.
    ///
    /// Args:
    ///     input: Path to the input media file.
    ///
    /// Returns:
    ///     PyComplexityReport with analysis results.
    fn analyze_complexity(&self, input: &str) -> PyResult<PyComplexityReport> {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Input file not found: {input}"
            )));
        }

        let file_size = std::fs::metadata(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
            .len();

        let report = compute_complexity_estimate(file_size, &self.codec);
        Ok(report)
    }

    /// Sweep CRF values to find the optimal quality/size tradeoff.
    ///
    /// Args:
    ///     input: Path to the input media file.
    ///     min_crf: Minimum CRF value (default: 18).
    ///     max_crf: Maximum CRF value (default: 40).
    ///     step: CRF step size (default: 2).
    ///
    /// Returns:
    ///     List of PyCrfResult for each CRF value tested.
    #[pyo3(signature = (input, min_crf=None, max_crf=None, step=None))]
    fn crf_sweep(
        &self,
        input: &str,
        min_crf: Option<u32>,
        max_crf: Option<u32>,
        step: Option<u32>,
    ) -> PyResult<Vec<PyCrfResult>> {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Input file not found: {input}"
            )));
        }

        let min = min_crf.unwrap_or(18);
        let max = max_crf.unwrap_or(40);
        let s = step.unwrap_or(2);

        if min >= max {
            return Err(PyValueError::new_err(format!(
                "min_crf ({min}) must be less than max_crf ({max})"
            )));
        }
        if s == 0 {
            return Err(PyValueError::new_err("step must be greater than 0"));
        }

        let file_size = std::fs::metadata(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
            .len();

        let mut results = Vec::new();
        let mut crf = min;
        while crf <= max {
            let result = simulate_crf_encode(crf, file_size, &self.codec);
            results.push(result);
            crf += s;
        }

        Ok(results)
    }

    /// Recommend a CRF value for a target quality level.
    ///
    /// Args:
    ///     input: Path to the input media file.
    ///     target_quality: Target quality (0.0-1.0, where 1.0 is best).
    ///
    /// Returns:
    ///     Recommended CRF value.
    #[pyo3(signature = (input, target_quality=None))]
    fn recommend_crf(&self, input: &str, target_quality: Option<f64>) -> PyResult<u32> {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Input file not found: {input}"
            )));
        }

        let target = target_quality.unwrap_or(0.8);
        if !(0.0..=1.0).contains(&target) {
            return Err(PyValueError::new_err(format!(
                "target_quality must be between 0.0 and 1.0, got {target}"
            )));
        }

        // Map target quality to CRF range
        // Higher quality -> lower CRF
        let max_crf: f64 = match self.codec.as_str() {
            "av1" => 63.0,
            "vp9" => 63.0,
            "vp8" => 63.0,
            _ => 51.0,
        };

        let crf = ((1.0 - target) * max_crf).round() as u32;
        let crf = crf.max(10).min(max_crf as u32 - 5);

        Ok(crf)
    }

    /// Generate a bitrate ladder for adaptive streaming.
    ///
    /// Args:
    ///     input: Path to the input media file.
    ///     strategy: Ladder strategy ("auto", "fixed", "per_title").
    ///
    /// Returns:
    ///     List of dicts, each representing a ladder rung.
    #[pyo3(signature = (input, strategy=None))]
    fn generate_ladder(
        &self,
        input: &str,
        strategy: Option<&str>,
    ) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Input file not found: {input}"
            )));
        }

        let strat = strategy.unwrap_or("auto");
        match strat {
            "auto" | "fixed" | "per_title" => {}
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown strategy '{}'. Expected: auto, fixed, per_title",
                    other
                )));
            }
        }

        let file_size = std::fs::metadata(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
            .len();

        let complexity = compute_complexity_estimate(file_size, &self.codec);

        // Generate ladder based on complexity
        let rungs_data = generate_ladder_rungs(&complexity, strat);

        Python::attach(|py| {
            let result: Vec<HashMap<String, Py<PyAny>>> = rungs_data
                .iter()
                .map(|rung| {
                    let mut m = HashMap::new();
                    m.insert(
                        "label".to_string(),
                        rung.label
                            .clone()
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "width".to_string(),
                        rung.width
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "height".to_string(),
                        rung.height
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "bitrate_kbps".to_string(),
                        rung.bitrate_kbps
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "codec".to_string(),
                        self.codec
                            .clone()
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "frame_rate".to_string(),
                        30.0_f64
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m
                })
                .collect();
            Ok(result)
        })
    }

    /// Benchmark multiple codecs on a given input.
    ///
    /// Args:
    ///     input: Path to the input media file.
    ///     codecs: List of codec names to benchmark.
    ///     duration: Optional duration in seconds to limit the test.
    ///
    /// Returns:
    ///     List of dicts with benchmark results per codec.
    #[pyo3(signature = (input, codecs=None, duration=None))]
    fn benchmark_codecs(
        &self,
        input: &str,
        codecs: Option<Vec<String>>,
        duration: Option<f64>,
    ) -> PyResult<Vec<HashMap<String, Py<PyAny>>>> {
        let path = PathBuf::from(input);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Input file not found: {input}"
            )));
        }

        let codec_list = codecs.unwrap_or_else(|| vec!["av1".to_string(), "vp9".to_string()]);

        for c in &codec_list {
            match c.as_str() {
                "av1" | "vp9" | "vp8" => {}
                other => {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported codec '{}'. Supported: av1, vp9, vp8",
                        other
                    )));
                }
            }
        }

        let file_size = std::fs::metadata(&path)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
            .len();

        let dur = duration.unwrap_or(60.0);

        Python::attach(|py| {
            let results: Vec<HashMap<String, Py<PyAny>>> = codec_list
                .iter()
                .map(|codec| {
                    let (speed_factor, quality_factor) = match codec.as_str() {
                        "av1" => (0.3, 1.0),
                        "vp9" => (0.7, 0.9),
                        "vp8" => (1.0, 0.75),
                        _ => (0.5, 0.8),
                    };

                    let est_fps = 30.0 * speed_factor;
                    let est_size = (file_size as f64 * quality_factor * 0.4) as u64;
                    let est_encode_time = dur / est_fps;

                    let mut m = HashMap::new();
                    m.insert(
                        "codec".to_string(),
                        codec
                            .clone()
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "estimated_fps".to_string(),
                        est_fps
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "estimated_encode_time_secs".to_string(),
                        est_encode_time
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "estimated_output_size_bytes".to_string(),
                        est_size
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m.insert(
                        "quality_score".to_string(),
                        (quality_factor * 100.0)
                            .into_pyobject(py)
                            .map(|o| o.into_any().unbind())
                            .unwrap_or_else(|_| py.None()),
                    );
                    m
                })
                .collect();
            Ok(results)
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyOptimizer(codec='{}', preset='{}')",
            self.codec, self.preset,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Analyze content complexity of a media file.
///
/// Args:
///     path: Path to the input media file.
///
/// Returns:
///     PyComplexityReport with analysis results.
#[pyfunction]
pub fn analyze_complexity(path: &str) -> PyResult<PyComplexityReport> {
    let p = PathBuf::from(path);
    if !p.exists() {
        return Err(PyValueError::new_err(format!(
            "Input file not found: {path}"
        )));
    }

    let file_size = std::fs::metadata(&p)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
        .len();

    Ok(compute_complexity_estimate(file_size, "av1"))
}

/// Recommend encoding settings for a media file.
///
/// Args:
///     path: Path to the input media file.
///     target: Target quality level ("low", "medium", "high", "maximum").
///
/// Returns:
///     Dict with recommended settings.
#[pyfunction]
#[pyo3(signature = (path, target=None))]
pub fn recommend_settings(path: &str, target: Option<&str>) -> PyResult<HashMap<String, String>> {
    let p = PathBuf::from(path);
    if !p.exists() {
        return Err(PyValueError::new_err(format!(
            "Input file not found: {path}"
        )));
    }

    let file_size = std::fs::metadata(&p)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read metadata: {e}")))?
        .len();

    let target_level = target.unwrap_or("high");
    let complexity = compute_complexity_estimate(file_size, "av1");

    let (crf, preset, bitrate): (u32, &str, &str) = match target_level {
        "low" => (38, "fast", "1000"),
        "medium" => (30, "medium", "3000"),
        "high" => (24, "slow", "5000"),
        "maximum" => (18, "placebo", "8000"),
        _ => {
            return Err(PyValueError::new_err(format!(
                "Unknown target '{}'. Expected: low, medium, high, maximum",
                target_level
            )));
        }
    };

    // Adjust based on complexity
    let adjusted_crf = if complexity.overall_complexity > 0.7 {
        crf.saturating_sub(2)
    } else if complexity.overall_complexity < 0.3 {
        crf + 4
    } else {
        crf
    };

    let mut settings = HashMap::new();
    settings.insert("codec".to_string(), "av1".to_string());
    settings.insert("crf".to_string(), adjusted_crf.to_string());
    settings.insert("preset".to_string(), preset.to_string());
    settings.insert("target_bitrate_kbps".to_string(), bitrate.to_string());
    settings.insert(
        "complexity".to_string(),
        format!("{:.2}", complexity.overall_complexity),
    );
    settings.insert(
        "ladder_strategy".to_string(),
        complexity.recommended_ladder.clone(),
    );

    Ok(settings)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Internal ladder rung data.
struct LadderRung {
    label: String,
    width: u32,
    height: u32,
    bitrate_kbps: u32,
}

/// Compute a complexity estimate from file size and codec.
fn compute_complexity_estimate(file_size: u64, codec: &str) -> PyComplexityReport {
    let size_mb = file_size as f64 / (1024.0 * 1024.0);

    let raw = (size_mb / 100.0).min(1.0);
    let spatial = (raw * 0.7 + 0.15).min(1.0);
    let temporal = (raw * 0.5 + 0.1).min(1.0);
    let overall = spatial * 0.6 + temporal * 0.4;

    let scene_count = (size_mb / 5.0).max(1.0) as u32;

    let codec_factor = match codec {
        "av1" => 1.0,
        "vp9" => 1.15,
        "vp8" => 1.4,
        _ => 1.2,
    };

    let base_bitrate = if overall > 0.7 {
        5000.0
    } else if overall > 0.4 {
        3000.0
    } else {
        1500.0
    };

    let recommended_crf = if overall > 0.7 {
        24
    } else if overall > 0.4 {
        28
    } else {
        32
    };

    let ladder = if overall > 0.6 { "per_title" } else { "fixed" };

    PyComplexityReport {
        overall_complexity: overall,
        spatial_complexity: spatial,
        temporal_complexity: temporal,
        scene_count,
        avg_bitrate_needed: (base_bitrate * codec_factor) as u32,
        recommended_crf,
        recommended_ladder: ladder.to_string(),
    }
}

/// Simulate a CRF encode result for estimation purposes.
fn simulate_crf_encode(crf: u32, file_size: u64, codec: &str) -> PyCrfResult {
    let max_crf = 63.0_f64;
    let quality_factor = 1.0 - (crf as f64 / max_crf);
    let size_factor = quality_factor * quality_factor;

    let codec_efficiency = match codec {
        "av1" => 0.35,
        "vp9" => 0.45,
        "vp8" => 0.55,
        _ => 0.50,
    };

    let est_size = (file_size as f64 * size_factor * codec_efficiency) as u64;
    let est_bitrate = (est_size as f64 * 8.0 / 60.0 / 1000.0) as u64;
    let est_psnr = 25.0 + quality_factor * 25.0;
    let est_ssim = 0.85 + quality_factor * 0.14;
    let est_vmaf = 40.0 + quality_factor * 60.0;

    PyCrfResult {
        crf,
        file_size: est_size,
        bitrate_kbps: est_bitrate,
        psnr: est_psnr,
        ssim: est_ssim,
        vmaf: est_vmaf,
    }
}

/// Generate ladder rungs based on complexity and strategy.
fn generate_ladder_rungs(report: &PyComplexityReport, strategy: &str) -> Vec<LadderRung> {
    let bitrate_multiplier = match strategy {
        "per_title" => {
            if report.overall_complexity > 0.7 {
                1.3
            } else if report.overall_complexity < 0.3 {
                0.7
            } else {
                1.0
            }
        }
        _ => 1.0,
    };

    let base_rungs = [
        ("1080p", 1920, 1080, 5000),
        ("720p", 1280, 720, 2500),
        ("480p", 854, 480, 1200),
        ("360p", 640, 360, 700),
        ("240p", 426, 240, 400),
    ];

    base_rungs
        .iter()
        .map(|(label, w, h, bitrate)| LadderRung {
            label: label.to_string(),
            width: *w,
            height: *h,
            bitrate_kbps: (*bitrate as f64 * bitrate_multiplier) as u32,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all optimize bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyComplexityReport>()?;
    m.add_class::<PyCrfResult>()?;
    m.add_class::<PyOptimizer>()?;
    m.add_function(wrap_pyfunction!(analyze_complexity, m)?)?;
    m.add_function(wrap_pyfunction!(recommend_settings, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_estimate_default() {
        let report = compute_complexity_estimate(50_000_000, "av1");
        assert!(report.overall_complexity > 0.0);
        assert!(report.overall_complexity <= 1.0);
        assert!(report.recommended_crf > 0);
    }

    #[test]
    fn test_simulate_crf_encode() {
        let result = simulate_crf_encode(28, 100_000_000, "av1");
        assert_eq!(result.crf, 28);
        assert!(result.file_size > 0);
        assert!(result.psnr > 0.0);
        assert!(result.ssim > 0.0 && result.ssim <= 1.0);
        assert!(result.vmaf > 0.0 && result.vmaf <= 100.0);
    }

    #[test]
    fn test_optimizer_creation_valid() {
        let opt = PyOptimizer::new(Some("av1"), Some("medium"));
        assert!(opt.is_ok());
    }

    #[test]
    fn test_optimizer_creation_invalid_codec() {
        let opt = PyOptimizer::new(Some("h264"), None);
        assert!(opt.is_err());
    }

    #[test]
    fn test_generate_ladder_rungs() {
        let report = compute_complexity_estimate(100_000_000, "av1");
        let rungs = generate_ladder_rungs(&report, "fixed");
        assert_eq!(rungs.len(), 5);
        assert_eq!(rungs[0].label, "1080p");
        assert_eq!(rungs[0].width, 1920);
    }
}
