//! Python bindings for audio analysis and metering.
//!
//! Wraps `oximedia-audio-analysis` for spectral features, beat detection,
//! and silence detection, plus `oximedia-metering` for EBU R128 and other
//! broadcast loudness standards.

use pyo3::prelude::*;

use oximedia_audio_analysis::silence_detect::{SilenceDetectConfig, SilenceDetector};
use oximedia_audio_analysis::{AnalysisConfig, AudioAnalyzer};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of loudness measurement (EBU R128 / ATSC A/85 / streaming targets).
#[pyclass]
#[derive(Clone)]
pub struct PyLoudnessResult {
    /// Integrated loudness in LUFS.
    #[pyo3(get)]
    pub integrated_lufs: f64,
    /// Momentary loudness in LUFS (400 ms window).
    #[pyo3(get)]
    pub momentary_lufs: f64,
    /// Short-term loudness in LUFS (3 s window).
    #[pyo3(get)]
    pub short_term_lufs: f64,
    /// Loudness range in LU.
    #[pyo3(get)]
    pub loudness_range: f64,
    /// True peak in dBTP (maximum across all channels).
    #[pyo3(get)]
    pub true_peak_dbtp: f64,
}

#[pymethods]
impl PyLoudnessResult {
    fn __repr__(&self) -> String {
        format!(
            "PyLoudnessResult(integrated={:.1} LUFS, peak={:.1} dBTP)",
            self.integrated_lufs, self.true_peak_dbtp
        )
    }
}

/// Summary of spectral features for an audio buffer.
#[pyclass]
#[derive(Clone)]
pub struct PySpectralFeatures {
    /// Spectral centroid in Hz (center of mass of spectrum).
    #[pyo3(get)]
    pub centroid: f32,
    /// Spectral flatness (0-1, higher = more noise-like).
    #[pyo3(get)]
    pub flatness: f32,
    /// Spectral bandwidth in Hz.
    #[pyo3(get)]
    pub bandwidth: f32,
    /// RMS level (linear amplitude).
    #[pyo3(get)]
    pub rms: f32,
}

#[pymethods]
impl PySpectralFeatures {
    fn __repr__(&self) -> String {
        format!(
            "PySpectralFeatures(centroid={:.1} Hz, flatness={:.4}, bw={:.1} Hz, rms={:.4})",
            self.centroid, self.flatness, self.bandwidth, self.rms
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_standard(s: &str) -> PyResult<Standard> {
    match s {
        "ebu-r128" | "ebu_r128" | "EBU R128" => Ok(Standard::EbuR128),
        "atsc-a85" | "atsc_a85" | "ATSC A/85" => Ok(Standard::AtscA85),
        "spotify" | "Spotify" => Ok(Standard::Spotify),
        "youtube" | "YouTube" => Ok(Standard::YouTube),
        "apple-music" | "apple_music" | "Apple Music" => Ok(Standard::AppleMusic),
        "netflix" | "Netflix" => Ok(Standard::Netflix),
        "amazon-prime" | "amazon_prime" | "Amazon Prime Video" => Ok(Standard::AmazonPrime),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown loudness standard: '{other}'. Supported: ebu-r128, atsc-a85, \
             spotify, youtube, apple-music, netflix, amazon-prime"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Module functions
// ---------------------------------------------------------------------------

/// Measure loudness of audio samples according to a broadcast standard.
///
/// Parameters
/// ----------
/// samples : list[float]
///     Interleaved audio samples (float32).
/// sample_rate : float
///     Sample rate in Hz.
/// channels : int, default 2
///     Number of audio channels.
/// standard : str, default "ebu-r128"
///     Loudness standard to use.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, channels=2, standard="ebu-r128"))]
pub fn measure_loudness(
    samples: Vec<f32>,
    sample_rate: f64,
    channels: usize,
    standard: &str,
) -> PyResult<PyLoudnessResult> {
    let std = parse_standard(standard)?;
    let config = MeterConfig::new(std, sample_rate, channels);
    let mut meter = LoudnessMeter::new(config)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
    meter.process_f32(&samples);
    let metrics = meter.metrics();
    Ok(PyLoudnessResult {
        integrated_lufs: metrics.integrated_lufs,
        momentary_lufs: metrics.momentary_lufs,
        short_term_lufs: metrics.short_term_lufs,
        loudness_range: metrics.loudness_range,
        true_peak_dbtp: metrics.true_peak_dbtp,
    })
}

/// Detect beats in mono audio and return their timestamps in seconds.
///
/// Uses onset detection + tempo estimation from `oximedia-audio-analysis`.
#[pyfunction]
pub fn detect_beats(samples: Vec<f32>, sample_rate: f32) -> PyResult<Vec<f64>> {
    use oximedia_audio_analysis::beat::{estimate_tempo_from_onsets, BeatTracker};
    use oximedia_audio_analysis::onset::{OnsetDetector, OnsetMethod};

    if samples.is_empty() {
        return Ok(Vec::new());
    }

    // Detect onsets using energy-based method
    let detector = OnsetDetector::new(OnsetMethod::EnergyBased, 0.5, sample_rate as u32);
    let onsets = detector.detect(&samples);

    // Onset events already have time_ms
    let onset_ms: Vec<u64> = onsets.iter().map(|o| o.time_ms).collect();

    if onset_ms.is_empty() {
        return Ok(Vec::new());
    }

    // Estimate tempo, then track beats
    let tempo = estimate_tempo_from_onsets(&onset_ms);
    let tracker = BeatTracker {
        tempo_estimate: tempo,
    };
    let beats = tracker.track(&onset_ms);

    // Convert beat events to seconds
    Ok(beats.iter().map(|b| b.time_ms as f64 / 1000.0).collect())
}

/// Compute spectral features of a mono audio buffer.
#[pyfunction]
pub fn spectral_features(samples: Vec<f32>, sample_rate: f32) -> PyResult<PySpectralFeatures> {
    let config = AnalysisConfig::default();
    let analyzer = AudioAnalyzer::new(config);
    let result = analyzer
        .analyze(&samples, sample_rate)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    Ok(PySpectralFeatures {
        centroid: result.spectral.centroid,
        flatness: result.spectral.flatness,
        bandwidth: result.spectral.bandwidth,
        rms: result.dynamics.rms,
    })
}

/// Detect silence regions in mono audio.
///
/// Returns a list of `(start_s, end_s)` tuples for each silent region.
///
/// Parameters
/// ----------
/// samples : list[float]
///     Mono audio samples.
/// sample_rate : float
///     Sample rate in Hz.
/// threshold_db : float, default -40.0
///     Threshold in dBFS below which audio is considered silent.
/// min_duration_ms : float, default 100.0
///     Minimum silence duration in milliseconds.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, threshold_db=-40.0, min_duration_ms=100.0))]
pub fn detect_silence(
    samples: Vec<f32>,
    sample_rate: f32,
    threshold_db: f32,
    min_duration_ms: f32,
) -> PyResult<Vec<(f64, f64)>> {
    let config = SilenceDetectConfig {
        threshold_dbfs: f64::from(threshold_db),
        min_silence_duration_s: f64::from(min_duration_ms) / 1000.0,
        ..SilenceDetectConfig::default()
    };
    let detector = SilenceDetector::new(config);
    let result = detector.detect(&samples, f64::from(sample_rate));

    Ok(result
        .regions
        .iter()
        .filter(|r| r.is_silent)
        .map(|r| (r.start_s, r.end_s))
        .collect())
}
