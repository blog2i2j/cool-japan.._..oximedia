//! Python bindings for Music Information Retrieval (MIR) via `oximedia-mir`.
//!
//! Exposes tempo detection, key detection and structural segmentation as
//! free functions on the Python `oximedia` module.

use pyo3::prelude::*;

use oximedia_mir::{MirAnalyzer, MirConfig};

// ---------------------------------------------------------------------------
// Python-visible result types
// ---------------------------------------------------------------------------

/// Tempo detection result returned by `detect_tempo`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyTempoResult {
    /// Detected BPM (beats per minute).
    #[pyo3(get)]
    pub bpm: f32,
    /// Detection confidence in [0.0, 1.0].
    #[pyo3(get)]
    pub confidence: f32,
    /// Tempo stability measure in [0.0, 1.0].
    #[pyo3(get)]
    pub stability: f32,
}

#[pymethods]
impl PyTempoResult {
    fn __repr__(&self) -> String {
        format!(
            "PyTempoResult(bpm={:.1}, confidence={:.3}, stability={:.3})",
            self.bpm, self.confidence, self.stability
        )
    }
}

/// Key detection result returned by `detect_key`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyKeyResult {
    /// Detected key string, e.g. ``"C major"`` or ``"A minor"``.
    #[pyo3(get)]
    pub key: String,
    /// Root note index: C=0, C#=1, … B=11.
    #[pyo3(get)]
    pub root: u8,
    /// ``True`` for major, ``False`` for minor.
    #[pyo3(get)]
    pub is_major: bool,
    /// Detection confidence in [0.0, 1.0].
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl PyKeyResult {
    fn __repr__(&self) -> String {
        format!(
            "PyKeyResult(key='{}', confidence={:.3})",
            self.key, self.confidence
        )
    }
}

/// A structural segment returned inside `segment_music`.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PyMusicSegment {
    /// Start time in seconds.
    #[pyo3(get)]
    pub start: f32,
    /// End time in seconds.
    #[pyo3(get)]
    pub end: f32,
    /// Segment label, e.g. ``"intro"``, ``"verse"``, ``"chorus"``.
    #[pyo3(get)]
    pub label: String,
    /// Confidence score in [0.0, 1.0].
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl PyMusicSegment {
    fn __repr__(&self) -> String {
        format!(
            "PyMusicSegment(label='{}', start={:.2}, end={:.2})",
            self.label, self.start, self.end
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_err(msg: String) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(msg)
}

fn validate_samples(samples: &[f32]) -> PyResult<()> {
    if samples.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "samples_f32 must not be empty",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

/// Detect the tempo (BPM) of an audio signal.
///
/// Parameters
/// ----------
/// samples_f32 : list[float]
///     Mono audio samples in the range [-1.0, 1.0].
/// sample_rate : float
///     Sample rate in Hz (e.g. 44100.0).
///
/// Returns
/// -------
/// PyTempoResult
#[pyfunction]
#[pyo3(signature = (samples_f32, sample_rate))]
pub fn detect_tempo(samples_f32: Vec<f32>, sample_rate: f32) -> PyResult<PyTempoResult> {
    validate_samples(&samples_f32)?;

    // Only enable beat tracking / tempo detection to keep analysis fast
    let config = MirConfig {
        enable_beat_tracking: true,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples_f32, sample_rate)
        .map_err(|e| make_err(e.to_string()))?;

    match result.tempo {
        Some(t) => Ok(PyTempoResult {
            bpm: t.bpm,
            confidence: t.confidence,
            stability: t.stability,
        }),
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Tempo detection returned no result",
        )),
    }
}

/// Detect the musical key of an audio signal.
///
/// Parameters
/// ----------
/// samples_f32 : list[float]
///     Mono audio samples in the range [-1.0, 1.0].
/// sample_rate : float
///     Sample rate in Hz.
///
/// Returns
/// -------
/// PyKeyResult
#[pyfunction]
#[pyo3(signature = (samples_f32, sample_rate))]
pub fn detect_key(samples_f32: Vec<f32>, sample_rate: f32) -> PyResult<PyKeyResult> {
    validate_samples(&samples_f32)?;

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: true,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: false,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples_f32, sample_rate)
        .map_err(|e| make_err(e.to_string()))?;

    match result.key {
        Some(k) => Ok(PyKeyResult {
            key: k.key,
            root: k.root,
            is_major: k.is_major,
            confidence: k.confidence,
        }),
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Key detection returned no result",
        )),
    }
}

/// Segment an audio signal into structural sections (intro, verse, chorus, etc.).
///
/// Parameters
/// ----------
/// samples_f32 : list[float]
///     Mono audio samples in the range [-1.0, 1.0].
/// sample_rate : float
///     Sample rate in Hz.
///
/// Returns
/// -------
/// list[PyMusicSegment]
///     Structural segments in chronological order.
#[pyfunction]
#[pyo3(signature = (samples_f32, sample_rate))]
pub fn segment_music(samples_f32: Vec<f32>, sample_rate: f32) -> PyResult<Vec<PyMusicSegment>> {
    validate_samples(&samples_f32)?;

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: false,
        enable_melody_extraction: false,
        enable_structure_analysis: true,
        enable_genre_classification: false,
        enable_mood_detection: false,
        enable_spectral_features: false,
        enable_rhythm_features: false,
        enable_harmonic_analysis: false,
        ..MirConfig::default()
    };

    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples_f32, sample_rate)
        .map_err(|e| make_err(e.to_string()))?;

    match result.structure {
        Some(s) => {
            let segments = s
                .segments
                .into_iter()
                .map(|seg| PyMusicSegment {
                    start: seg.start,
                    end: seg.end,
                    label: seg.label,
                    confidence: seg.confidence,
                })
                .collect();
            Ok(segments)
        }
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Structure analysis returned no result",
        )),
    }
}

// ---------------------------------------------------------------------------
// Module registration helper
// ---------------------------------------------------------------------------

/// Register all MIR classes and free functions into the given Python module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTempoResult>()?;
    m.add_class::<PyKeyResult>()?;
    m.add_class::<PyMusicSegment>()?;
    m.add_function(wrap_pyfunction!(detect_tempo, m)?)?;
    m.add_function(wrap_pyfunction!(detect_key, m)?)?;
    m.add_function(wrap_pyfunction!(segment_music, m)?)?;
    Ok(())
}
