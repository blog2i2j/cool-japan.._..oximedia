//! Python bindings for `oximedia-restore` audio restoration.
//!
//! Provides `PyRestorer`, `PyRestoreConfig`, `PyDegradationReport`,
//! and standalone functions for audio restoration from Python.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;

use oximedia_restore::presets::{BroadcastCleanup, TapeRestoration, VinylRestoration};
use oximedia_restore::{dc::DcRemover, RestorationStep, RestoreChain};

// ---------------------------------------------------------------------------
// PyRestoreConfig
// ---------------------------------------------------------------------------

/// Configuration for audio restoration.
#[pyclass]
#[derive(Clone)]
pub struct PyRestoreConfig {
    /// Restoration preset: vinyl, tape, broadcast, archival, custom.
    #[pyo3(get, set)]
    pub preset: String,
    /// Sample rate in Hz.
    #[pyo3(get, set)]
    pub sample_rate: u32,
    /// Enable click removal.
    #[pyo3(get, set)]
    pub click_removal: bool,
    /// Enable crackle removal.
    #[pyo3(get, set)]
    pub crackle_removal: bool,
    /// Enable hum removal.
    #[pyo3(get, set)]
    pub hum_removal: bool,
    /// Enable hiss removal.
    #[pyo3(get, set)]
    pub hiss_removal: bool,
    /// Enable declipping.
    #[pyo3(get, set)]
    pub declip: bool,
    /// Enable DC offset removal.
    #[pyo3(get, set)]
    pub dc_removal: bool,
}

#[pymethods]
impl PyRestoreConfig {
    /// Create a new default configuration.
    #[new]
    #[pyo3(signature = (preset="vinyl", sample_rate=44100))]
    fn new(preset: &str, sample_rate: u32) -> Self {
        Self {
            preset: preset.to_string(),
            sample_rate,
            click_removal: true,
            crackle_removal: true,
            hum_removal: true,
            hiss_removal: true,
            declip: false,
            dc_removal: true,
        }
    }

    /// Create a vinyl restoration config.
    #[classmethod]
    fn vinyl(_cls: &Bound<'_, PyType>, sample_rate: u32) -> Self {
        Self::new("vinyl", sample_rate)
    }

    /// Create a tape restoration config.
    #[classmethod]
    fn tape(_cls: &Bound<'_, PyType>, sample_rate: u32) -> Self {
        Self {
            preset: "tape".to_string(),
            sample_rate,
            click_removal: false,
            crackle_removal: false,
            hum_removal: false,
            hiss_removal: true,
            declip: false,
            dc_removal: true,
        }
    }

    /// Create a broadcast cleanup config.
    #[classmethod]
    fn broadcast(_cls: &Bound<'_, PyType>, sample_rate: u32) -> Self {
        Self {
            preset: "broadcast".to_string(),
            sample_rate,
            click_removal: false,
            crackle_removal: false,
            hum_removal: false,
            hiss_removal: false,
            declip: true,
            dc_removal: true,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PyRestoreConfig(preset='{}', sr={}, click={}, crackle={}, hum={}, hiss={}, declip={}, dc={})",
            self.preset, self.sample_rate, self.click_removal,
            self.crackle_removal, self.hum_removal, self.hiss_removal,
            self.declip, self.dc_removal,
        )
    }
}

// ---------------------------------------------------------------------------
// PyDegradationReport
// ---------------------------------------------------------------------------

/// Report of degradation analysis.
#[pyclass]
#[derive(Clone)]
pub struct PyDegradationReport {
    /// Peak sample level (0.0-1.0).
    #[pyo3(get)]
    pub peak_level: f64,
    /// Number of clipped samples.
    #[pyo3(get)]
    pub clipped_samples: usize,
    /// Clipping percentage.
    #[pyo3(get)]
    pub clipping_percent: f64,
    /// DC offset value.
    #[pyo3(get)]
    pub dc_offset: f64,
    /// RMS level.
    #[pyo3(get)]
    pub rms_level: f64,
    /// Crest factor (peak/RMS).
    #[pyo3(get)]
    pub crest_factor: f64,
    /// Total number of samples analyzed.
    #[pyo3(get)]
    pub sample_count: usize,
}

#[pymethods]
impl PyDegradationReport {
    fn __repr__(&self) -> String {
        format!(
            "PyDegradationReport(peak={:.4}, clip={:.2}%, dc={:.6}, rms={:.4})",
            self.peak_level, self.clipping_percent, self.dc_offset, self.rms_level,
        )
    }

    /// Convert to a Python dict.
    fn to_dict(&self, py: Python<'_>) -> PyResult<HashMap<String, Py<PyAny>>> {
        let mut m = HashMap::new();
        m.insert(
            "peak_level".to_string(),
            self.peak_level.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "clipped_samples".to_string(),
            self.clipped_samples.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "clipping_percent".to_string(),
            self.clipping_percent.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "dc_offset".to_string(),
            self.dc_offset.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "rms_level".to_string(),
            self.rms_level.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "crest_factor".to_string(),
            self.crest_factor.into_pyobject(py)?.into_any().unbind(),
        );
        m.insert(
            "sample_count".to_string(),
            self.sample_count.into_pyobject(py)?.into_any().unbind(),
        );
        Ok(m)
    }
}

// ---------------------------------------------------------------------------
// PyRestorer
// ---------------------------------------------------------------------------

/// Audio restorer with configurable processing chain.
#[pyclass]
pub struct PyRestorer {
    config: PyRestoreConfig,
}

#[pymethods]
impl PyRestorer {
    /// Create a new restorer with the given configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&PyRestoreConfig>) -> Self {
        let cfg = config
            .cloned()
            .unwrap_or_else(|| PyRestoreConfig::new("vinyl", 44100));
        Self { config: cfg }
    }

    /// Restore mono audio samples.
    ///
    /// Args:
    ///     samples: Audio samples as a list of floats.
    ///     sample_rate: Sample rate in Hz (overrides config if provided).
    ///
    /// Returns:
    ///     Restored audio samples as a list of floats.
    #[pyo3(signature = (samples, sample_rate=None))]
    fn restore(&self, samples: Vec<f32>, sample_rate: Option<u32>) -> PyResult<Vec<f32>> {
        let sr = sample_rate.unwrap_or(self.config.sample_rate);
        if samples.is_empty() {
            return Err(PyValueError::new_err("Empty sample buffer"));
        }

        let mut chain = build_chain(&self.config, sr);
        chain
            .process(&samples, sr)
            .map_err(|e| PyRuntimeError::new_err(format!("Restoration failed: {e}")))
    }

    /// Restore stereo audio samples.
    ///
    /// Args:
    ///     left: Left channel samples.
    ///     right: Right channel samples.
    ///     sample_rate: Sample rate in Hz.
    ///
    /// Returns:
    ///     Tuple of (left_restored, right_restored).
    #[pyo3(signature = (left, right, sample_rate=None))]
    fn restore_stereo(
        &self,
        left: Vec<f32>,
        right: Vec<f32>,
        sample_rate: Option<u32>,
    ) -> PyResult<(Vec<f32>, Vec<f32>)> {
        let sr = sample_rate.unwrap_or(self.config.sample_rate);
        if left.is_empty() || right.is_empty() {
            return Err(PyValueError::new_err("Empty sample buffer(s)"));
        }

        let mut chain = build_chain(&self.config, sr);
        chain
            .process_stereo(&left, &right, sr)
            .map_err(|e| PyRuntimeError::new_err(format!("Stereo restoration failed: {e}")))
    }

    /// Get the current configuration.
    fn get_config(&self) -> PyRestoreConfig {
        self.config.clone()
    }

    fn __repr__(&self) -> String {
        format!("PyRestorer(preset='{}')", self.config.preset)
    }
}

/// Build a `RestoreChain` from a `PyRestoreConfig`.
fn build_chain(config: &PyRestoreConfig, sample_rate: u32) -> RestoreChain {
    let mut chain = RestoreChain::new();

    match config.preset.to_lowercase().as_str() {
        "vinyl" => {
            let mut preset = VinylRestoration::new(sample_rate);
            preset.click_removal = config.click_removal;
            preset.crackle_removal = config.crackle_removal;
            preset.hum_removal = config.hum_removal;
            chain.add_preset(preset);
        }
        "tape" => {
            let mut preset = TapeRestoration::new(sample_rate);
            preset.hiss_removal = config.hiss_removal;
            chain.add_preset(preset);
        }
        "broadcast" => {
            let preset = BroadcastCleanup::new(sample_rate);
            chain.add_preset(preset);
        }
        "archival" => {
            chain.add_preset(VinylRestoration::new(sample_rate));
            chain.add_preset(TapeRestoration::new(sample_rate));
        }
        _ => {
            // Custom: add individual steps
            if config.dc_removal {
                chain.add_step(RestorationStep::DcRemoval(DcRemover::new(
                    10.0,
                    sample_rate,
                )));
            }
        }
    }

    chain
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Restore mono audio samples with a given mode.
///
/// Args:
///     samples: Audio samples as a list of floats.
///     sample_rate: Sample rate in Hz.
///     mode: Restoration mode (vinyl, tape, broadcast, archival).
///
/// Returns:
///     Restored audio samples.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate=44100, mode="vinyl"))]
pub fn restore_audio(samples: Vec<f32>, sample_rate: u32, mode: &str) -> PyResult<Vec<f32>> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("Empty sample buffer"));
    }
    let config = PyRestoreConfig::new(mode, sample_rate);
    let mut chain = build_chain(&config, sample_rate);
    chain
        .process(&samples, sample_rate)
        .map_err(|e| PyRuntimeError::new_err(format!("Restoration failed: {e}")))
}

/// Analyze audio degradation.
///
/// Args:
///     samples: Audio samples as a list of floats.
///
/// Returns:
///     PyDegradationReport with analysis results.
#[pyfunction]
pub fn analyze_degradation(samples: Vec<f32>) -> PyResult<PyDegradationReport> {
    if samples.is_empty() {
        return Err(PyValueError::new_err("Empty sample buffer"));
    }

    let peak = samples.iter().fold(0.0_f32, |max, &s| max.max(s.abs()));
    let clip_count = samples.iter().filter(|&&s| s.abs() >= 0.999).count();
    let clip_pct = (clip_count as f64 / samples.len() as f64) * 100.0;
    let dc: f64 = samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len() as f64;
    let rms: f64 = (samples
        .iter()
        .map(|&s| (s as f64) * (s as f64))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    let crest = if rms > 0.0 { peak as f64 / rms } else { 0.0 };

    Ok(PyDegradationReport {
        peak_level: peak as f64,
        clipped_samples: clip_count,
        clipping_percent: clip_pct,
        dc_offset: dc,
        rms_level: rms,
        crest_factor: crest,
        sample_count: samples.len(),
    })
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register all restore bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRestoreConfig>()?;
    m.add_class::<PyDegradationReport>()?;
    m.add_class::<PyRestorer>()?;
    m.add_function(wrap_pyfunction!(restore_audio, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_degradation, m)?)?;
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
        let cfg = PyRestoreConfig::new("vinyl", 44100);
        assert_eq!(cfg.preset, "vinyl");
        assert_eq!(cfg.sample_rate, 44100);
        assert!(cfg.click_removal);
        assert!(cfg.dc_removal);
    }

    #[test]
    fn test_config_repr() {
        let cfg = PyRestoreConfig::new("tape", 48000);
        let repr = cfg.__repr__();
        assert!(repr.contains("tape"));
        assert!(repr.contains("48000"));
    }

    #[test]
    fn test_analyze_degradation_clean() {
        let samples = vec![0.5_f32; 1000];
        let report = analyze_degradation(samples);
        assert!(report.is_ok());
        let r = report.expect("analysis should succeed");
        assert_eq!(r.clipped_samples, 0);
        assert!((r.dc_offset - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_analyze_degradation_clipped() {
        let samples = vec![1.0_f32; 100];
        let report = analyze_degradation(samples);
        assert!(report.is_ok());
        let r = report.expect("analysis should succeed");
        assert_eq!(r.clipped_samples, 100);
        assert!(r.clipping_percent > 99.0);
    }

    #[test]
    fn test_analyze_degradation_empty() {
        let result = analyze_degradation(vec![]);
        assert!(result.is_err());
    }
}
