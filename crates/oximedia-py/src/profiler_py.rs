//! Python bindings for `oximedia-profiler` performance profiling.
//!
//! Provides `PyProfiler`, `PyProfileReport`, `PyBottleneck`, and standalone
//! functions for profiling encode/decode/filter operations.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PyProfileReport
// ---------------------------------------------------------------------------

/// A profiling report result.
#[pyclass]
#[derive(Clone)]
pub struct PyProfileReport {
    /// Profiling mode used.
    #[pyo3(get)]
    pub mode: String,
    /// Report text.
    #[pyo3(get)]
    pub report_text: String,
    /// Duration in milliseconds.
    #[pyo3(get)]
    pub duration_ms: f64,
    /// Whether profiling was successful.
    #[pyo3(get)]
    pub success: bool,
}

#[pymethods]
impl PyProfileReport {
    fn __repr__(&self) -> String {
        format!(
            "PyProfileReport(mode='{}', duration={:.1}ms, success={})",
            self.mode, self.duration_ms, self.success,
        )
    }

    /// Get the report as a dict.
    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("mode".to_string(), self.mode.clone());
        m.insert(
            "duration_ms".to_string(),
            format!("{:.4}", self.duration_ms),
        );
        m.insert("success".to_string(), self.success.to_string());
        m.insert("report".to_string(), self.report_text.clone());
        m
    }
}

// ---------------------------------------------------------------------------
// PyBottleneck
// ---------------------------------------------------------------------------

/// A detected performance bottleneck.
#[pyclass]
#[derive(Clone)]
pub struct PyBottleneck {
    /// Component name.
    #[pyo3(get)]
    pub component: String,
    /// Bottleneck severity (0.0 to 1.0).
    #[pyo3(get)]
    pub severity: f64,
    /// Description of the bottleneck.
    #[pyo3(get)]
    pub description: String,
    /// Suggested fix.
    #[pyo3(get)]
    pub suggestion: String,
}

#[pymethods]
impl PyBottleneck {
    fn __repr__(&self) -> String {
        format!(
            "PyBottleneck(component='{}', severity={:.2}, desc='{}')",
            self.component, self.severity, self.description,
        )
    }
}

// ---------------------------------------------------------------------------
// PyProfiler
// ---------------------------------------------------------------------------

/// Performance profiler for media operations.
#[pyclass]
pub struct PyProfiler {
    mode: String,
    sample_rate: u32,
    running: bool,
    inner: Option<oximedia_profiler::Profiler>,
}

#[pymethods]
impl PyProfiler {
    /// Create a new profiler.
    ///
    /// Args:
    ///     mode: Profiling mode (sampling, instrumentation, event-based, continuous).
    ///     sample_rate: Sampling rate in Hz (default: 100).
    #[new]
    #[pyo3(signature = (mode=None, sample_rate=None))]
    fn new(mode: Option<&str>, sample_rate: Option<u32>) -> PyResult<Self> {
        let m = mode.unwrap_or("sampling");
        let profiling_mode = match m {
            "sampling" => oximedia_profiler::ProfilingMode::Sampling,
            "instrumentation" => oximedia_profiler::ProfilingMode::Instrumentation,
            "event-based" | "event" => oximedia_profiler::ProfilingMode::EventBased,
            "continuous" => oximedia_profiler::ProfilingMode::Continuous,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown mode '{}'. Supported: sampling, instrumentation, event-based, continuous",
                    other
                )));
            }
        };

        let sr = sample_rate.unwrap_or(100);
        let config = oximedia_profiler::ProfilerConfig {
            mode: profiling_mode,
            sample_rate: sr,
            cpu_profiling: true,
            memory_profiling: true,
            gpu_profiling: false,
            frame_timing: false,
            resource_tracking: true,
            cache_analysis: false,
            thread_analysis: true,
            max_overhead: 1.0,
        };

        let profiler = oximedia_profiler::Profiler::with_config(config);

        Ok(Self {
            mode: m.to_string(),
            sample_rate: sr,
            running: false,
            inner: Some(profiler),
        })
    }

    /// Start profiling.
    fn start(&mut self) -> PyResult<()> {
        if self.running {
            return Err(PyRuntimeError::new_err("Profiler already running"));
        }
        if let Some(ref mut profiler) = self.inner {
            profiler
                .start()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to start: {e}")))?;
        }
        self.running = true;
        Ok(())
    }

    /// Stop profiling.
    fn stop(&mut self) -> PyResult<()> {
        if !self.running {
            return Err(PyRuntimeError::new_err("Profiler not running"));
        }
        if let Some(ref mut profiler) = self.inner {
            profiler
                .stop()
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to stop: {e}")))?;
        }
        self.running = false;
        Ok(())
    }

    /// Generate a profiling report.
    fn report(&self) -> PyResult<PyProfileReport> {
        let report_text = if let Some(ref profiler) = self.inner {
            profiler.generate_report()
        } else {
            "No profiler data available".to_string()
        };

        let duration_ms = if let Some(ref profiler) = self.inner {
            profiler
                .elapsed()
                .map(|d| d.as_secs_f64() * 1000.0)
                .unwrap_or(0.0)
        } else {
            0.0
        };

        Ok(PyProfileReport {
            mode: self.mode.clone(),
            report_text,
            duration_ms,
            success: true,
        })
    }

    /// Check if profiler is running.
    fn is_running(&self) -> bool {
        self.running
    }

    /// Record a custom metric.
    fn record_metric(&mut self, name: &str, value: f64) -> PyResult<()> {
        if let Some(ref mut profiler) = self.inner {
            profiler.record_metric(
                name.to_string(),
                oximedia_profiler::ProfileMetric::Percentage(value),
            );
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyProfiler(mode='{}', sample_rate={}, running={})",
            self.mode, self.sample_rate, self.running,
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Profile an encode operation (simulated).
///
/// Args:
///     codec: Codec to profile (av1, vp9, vp8).
///     width: Frame width.
///     height: Frame height.
///     frames: Number of frames to profile.
///
/// Returns:
///     JSON string with profiling results.
#[pyfunction]
#[pyo3(signature = (codec, width, height, frames=None))]
pub fn profile_encode(
    codec: &str,
    width: u32,
    height: u32,
    frames: Option<u32>,
) -> PyResult<String> {
    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(PyValueError::new_err(format!(
            "Unsupported codec '{}'. Supported: {}",
            codec,
            valid_codecs.join(", ")
        )));
    }
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err("Width and height must be > 0"));
    }

    let frame_count = frames.unwrap_or(100);
    let pixels = width as u64 * height as u64 * frame_count as u64;

    Ok(format!(
        "{{\"operation\":\"encode\",\"codec\":\"{codec}\",\"resolution\":\"{}x{}\",\"frames\":{frame_count},\"total_pixels\":{pixels},\"status\":\"profiled\"}}",
        width, height
    ))
}

/// Profile a decode operation (simulated).
///
/// Args:
///     codec: Codec to profile.
///     duration_ms: Duration to profile in milliseconds.
///
/// Returns:
///     JSON string with profiling results.
#[pyfunction]
#[pyo3(signature = (codec, duration_ms=None))]
pub fn profile_decode(codec: &str, duration_ms: Option<u64>) -> PyResult<String> {
    let valid_codecs = ["av1", "vp9", "vp8", "opus", "vorbis", "flac"];
    if !valid_codecs.contains(&codec) {
        return Err(PyValueError::new_err(format!(
            "Unsupported codec '{}'. Supported: {}",
            codec,
            valid_codecs.join(", ")
        )));
    }

    let dur = duration_ms.unwrap_or(1000);

    Ok(format!(
        "{{\"operation\":\"decode\",\"codec\":\"{codec}\",\"duration_ms\":{dur},\"status\":\"profiled\"}}"
    ))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all profiler bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProfiler>()?;
    m.add_class::<PyProfileReport>()?;
    m.add_class::<PyBottleneck>()?;
    m.add_function(wrap_pyfunction!(profile_encode, m)?)?;
    m.add_function(wrap_pyfunction!(profile_decode, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let p = PyProfiler::new(None, None);
        assert!(p.is_ok());
        let p = p.expect("should succeed");
        assert_eq!(p.mode, "sampling");
        assert!(!p.running);
    }

    #[test]
    fn test_profiler_invalid_mode() {
        let result = PyProfiler::new(Some("invalid"), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_profiler_start_stop() {
        let mut p = PyProfiler::new(None, None).expect("should succeed");
        assert!(p.start().is_ok());
        assert!(p.is_running());
        assert!(p.stop().is_ok());
        assert!(!p.is_running());
    }

    #[test]
    fn test_profile_encode() {
        let result = profile_encode("av1", 1920, 1080, None);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("encode"));
        assert!(json.contains("av1"));
    }

    #[test]
    fn test_profile_decode_invalid_codec() {
        let result = profile_decode("h264", None);
        assert!(result.is_err());
    }
}
