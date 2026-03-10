//! WASM audio analysis functions.
//!
//! This module provides WebAssembly bindings for audio analysis operations
//! including loudness measurement (EBU R128), beat detection, and spectral
//! feature extraction. All functions operate on PCM float samples and return
//! JSON strings with structured analysis results.
//!
//! # Available Functions
//!
//! - [`wasm_analyze_loudness`] - EBU R128 loudness measurement
//! - [`wasm_detect_beats`] - Beat detection and BPM estimation
//! - [`wasm_spectral_features`] - Spectral centroid, flatness, bandwidth, etc.
//!
//! # JavaScript Example
//!
//! ```javascript
//! // Analyze loudness of stereo audio
//! const result = JSON.parse(oximedia.wasm_analyze_loudness(samples, 48000, 2));
//! console.log('Integrated:', result.integrated_lufs, 'LUFS');
//!
//! // Detect beats in mono audio
//! const beats = JSON.parse(oximedia.wasm_detect_beats(samples, 44100));
//! console.log('BPM:', beats.bpm);
//!
//! // Get spectral features
//! const spectral = JSON.parse(oximedia.wasm_spectral_features(samples, 44100));
//! console.log('Centroid:', spectral.centroid, 'Hz');
//! ```

use wasm_bindgen::prelude::*;

use oximedia_audio_analysis::beat::{estimate_tempo_from_onsets, BeatTracker};
use oximedia_audio_analysis::onset::{OnsetDetector, OnsetMethod};
use oximedia_audio_analysis::{AnalysisConfig, AudioAnalyzer};
use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};

/// Analyze loudness of PCM audio data using the EBU R128 standard.
///
/// Takes interleaved f32 PCM samples and returns a JSON string with loudness
/// metrics including integrated loudness (LUFS), momentary loudness,
/// short-term loudness, loudness range (LRA), and true peak level (dBTP).
///
/// # Arguments
///
/// * `samples` - Interleaved f32 PCM samples normalized to -1.0..1.0
/// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000)
/// * `channels` - Number of audio channels (1 for mono, 2 for stereo, etc.)
///
/// # Returns
///
/// JSON string with fields:
/// - `integrated_lufs`: Gated integrated loudness
/// - `momentary_lufs`: Momentary loudness (400ms window)
/// - `short_term_lufs`: Short-term loudness (3s window)
/// - `loudness_range`: Loudness range in LU
/// - `true_peak_dbtp`: Maximum true peak in dBTP
/// - `max_momentary`: Maximum momentary loudness seen
/// - `max_short_term`: Maximum short-term loudness seen
///
/// # Errors
///
/// Returns an error if the sample rate or channel count is invalid.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const metrics = JSON.parse(oximedia.wasm_analyze_loudness(pcm, 48000, 2));
/// if (metrics.integrated_lufs > -23.0) {
///     console.log('Audio may be too loud for EBU R128 broadcast');
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_analyze_loudness(
    samples: &[f32],
    sample_rate: u32,
    channels: u32,
) -> Result<String, JsValue> {
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }
    if channels == 0 {
        return Err(crate::utils::js_err("Channel count must be > 0"));
    }
    if samples.is_empty() {
        return Err(crate::utils::js_err("No audio samples provided"));
    }

    let config = MeterConfig::new(Standard::EbuR128, f64::from(sample_rate), channels as usize);
    let mut meter = LoudnessMeter::new(config)
        .map_err(|e| crate::utils::js_err(&format!("Meter creation error: {e}")))?;

    meter.process_f32(samples);
    let metrics = meter.metrics();

    let json = serde_json::json!({
        "integrated_lufs": metrics.integrated_lufs,
        "momentary_lufs": metrics.momentary_lufs,
        "short_term_lufs": metrics.short_term_lufs,
        "loudness_range": metrics.loudness_range,
        "true_peak_dbtp": metrics.true_peak_dbtp,
        "max_momentary": metrics.max_momentary,
        "max_short_term": metrics.max_short_term,
        "channel_peaks_dbtp": metrics.channel_peaks_dbtp,
    });

    serde_json::to_string(&json)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

/// Detect beats in PCM audio and estimate tempo.
///
/// Performs onset detection on the audio signal, then estimates tempo (BPM)
/// and projects beat positions. The input should be mono audio; if stereo
/// data is provided, it will be analyzed as-is (interleaved).
///
/// # Arguments
///
/// * `samples` - f32 PCM samples (mono preferred)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// JSON string with fields:
/// - `bpm`: Estimated beats per minute
/// - `confidence`: Confidence of tempo estimate (0.0-1.0)
/// - `period_ms`: Beat period in milliseconds
/// - `beat_count`: Number of detected beats
/// - `beats`: Array of beat timestamps in milliseconds
///
/// # Errors
///
/// Returns an error if the sample rate is zero or samples are empty.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const result = JSON.parse(oximedia.wasm_detect_beats(monoSamples, 44100));
/// console.log(`Tempo: ${result.bpm} BPM (confidence: ${result.confidence})`);
/// result.beats.forEach(t => console.log(`Beat at ${t} ms`));
/// ```
#[wasm_bindgen]
pub fn wasm_detect_beats(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }
    if samples.is_empty() {
        return Err(crate::utils::js_err("No audio samples provided"));
    }

    // Detect onsets using energy-based method
    let onset_detector = OnsetDetector::new(OnsetMethod::EnergyBased, 0.3, sample_rate);
    let onsets = onset_detector.detect(samples);

    // Extract onset times for tempo estimation
    let onset_times_ms: Vec<u64> = onsets.iter().map(|o| o.time_ms).collect();

    // Estimate tempo from onset intervals
    let tempo = estimate_tempo_from_onsets(&onset_times_ms);

    // Track beats using the estimated tempo
    let tracker = BeatTracker::new(tempo.clone());
    let beats = tracker.track(&onset_times_ms);

    let beat_times_ms: Vec<u64> = beats.iter().map(|b| b.time_ms).collect();

    let json = serde_json::json!({
        "bpm": tempo.bpm,
        "confidence": tempo.confidence,
        "period_ms": tempo.period_ms,
        "beat_count": beats.len(),
        "beats": beat_times_ms,
    });

    serde_json::to_string(&json)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

/// Compute spectral features of PCM audio.
///
/// Performs frequency-domain analysis to extract spectral characteristics
/// including centroid (center of mass), flatness (noise-likeness), bandwidth,
/// rolloff frequency, and crest factor.
///
/// # Arguments
///
/// * `samples` - f32 PCM samples (mono preferred)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
///
/// JSON string with fields:
/// - `centroid`: Spectral centroid in Hz (center of mass of spectrum)
/// - `flatness`: Spectral flatness (0-1, higher = more noise-like)
/// - `crest`: Spectral crest factor (peak-to-average ratio)
/// - `bandwidth`: Spectral bandwidth in Hz
/// - `rolloff`: Spectral rolloff frequency in Hz (85% energy threshold)
/// - `flux`: Spectral flux (frame-to-frame change)
/// - `dynamics`: Object with peak, rms, crest, dynamic_range_db
///
/// # Errors
///
/// Returns an error if the sample rate is invalid or insufficient samples
/// are provided (minimum 2048 samples for default FFT size).
///
/// # Example (JavaScript)
///
/// ```javascript
/// const features = JSON.parse(oximedia.wasm_spectral_features(mono, 44100));
/// if (features.flatness > 0.8) {
///     console.log('Audio is noise-like');
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_spectral_features(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    if sample_rate == 0 {
        return Err(crate::utils::js_err("Sample rate must be > 0"));
    }
    if samples.is_empty() {
        return Err(crate::utils::js_err("No audio samples provided"));
    }

    let config = AnalysisConfig::default();
    let analyzer = AudioAnalyzer::new(config);

    let result = analyzer
        .analyze(samples, sample_rate as f32)
        .map_err(|e| crate::utils::js_err(&format!("Spectral analysis error: {e}")))?;

    let json = serde_json::json!({
        "centroid": result.spectral.centroid,
        "flatness": result.spectral.flatness,
        "crest": result.spectral.crest,
        "bandwidth": result.spectral.bandwidth,
        "rolloff": result.spectral.rolloff,
        "flux": result.spectral.flux,
        "dynamics": {
            "peak": result.dynamics.peak,
            "rms": result.dynamics.rms,
            "crest": result.dynamics.crest,
            "dynamic_range_db": result.dynamics.dynamic_range_db,
        },
        "pitch": {
            "mean_f0": result.pitch.mean_f0,
            "voicing_rate": result.pitch.voicing_rate,
        },
    });

    serde_json::to_string(&json)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization error: {e}")))
}

/// Pure-Rust validation guard for loudness analysis (testable without WASM runtime).
///
/// Returns `Ok(())` when inputs are acceptable, or an `Err(String)` describing
/// the validation failure.  The actual processing is left to `wasm_analyze_loudness`.
#[cfg(test)]
pub(crate) fn validate_loudness_inputs(
    samples: &[f32],
    sample_rate: u32,
    channels: u32,
) -> Result<(), String> {
    if sample_rate == 0 {
        return Err("Sample rate must be > 0".to_string());
    }
    if channels == 0 {
        return Err("Channel count must be > 0".to_string());
    }
    if samples.is_empty() {
        return Err("No audio samples provided".to_string());
    }
    Ok(())
}

/// Pure-Rust validation guard for beat detection.
#[cfg(test)]
pub(crate) fn validate_beat_inputs(samples: &[f32], sample_rate: u32) -> Result<(), String> {
    if sample_rate == 0 {
        return Err("Sample rate must be > 0".to_string());
    }
    if samples.is_empty() {
        return Err("No audio samples provided".to_string());
    }
    Ok(())
}

/// Pure-Rust validation guard for spectral analysis.
#[cfg(test)]
pub(crate) fn validate_spectral_inputs(samples: &[f32], sample_rate: u32) -> Result<(), String> {
    if sample_rate == 0 {
        return Err("Sample rate must be > 0".to_string());
    }
    if samples.is_empty() {
        return Err("No audio samples provided".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_loudness_inputs ────────────────────────────────────────────

    #[test]
    fn test_validate_loudness_zero_sample_rate() {
        assert!(validate_loudness_inputs(&[0.0_f32; 100], 0, 1).is_err());
    }

    #[test]
    fn test_validate_loudness_zero_channels() {
        assert!(validate_loudness_inputs(&[0.0_f32; 100], 44100, 0).is_err());
    }

    #[test]
    fn test_validate_loudness_empty_samples() {
        assert!(validate_loudness_inputs(&[], 44100, 1).is_err());
    }

    #[test]
    fn test_validate_loudness_valid_inputs() {
        assert!(validate_loudness_inputs(&vec![0.0_f32; 48000], 48000, 1).is_ok());
    }

    #[test]
    fn test_validate_loudness_stereo_valid() {
        assert!(validate_loudness_inputs(&vec![0.0_f32; 96000], 48000, 2).is_ok());
    }

    // ── validate_beat_inputs ───────────────────────────────────────────────

    #[test]
    fn test_validate_beats_zero_sample_rate() {
        assert!(validate_beat_inputs(&[0.0_f32; 100], 0).is_err());
    }

    #[test]
    fn test_validate_beats_empty_samples() {
        assert!(validate_beat_inputs(&[], 44100).is_err());
    }

    #[test]
    fn test_validate_beats_valid() {
        assert!(validate_beat_inputs(&[0.0_f32; 1024], 44100).is_ok());
    }

    // ── validate_spectral_inputs ───────────────────────────────────────────

    #[test]
    fn test_validate_spectral_zero_sample_rate() {
        assert!(validate_spectral_inputs(&[0.0_f32; 2048], 0).is_err());
    }

    #[test]
    fn test_validate_spectral_empty_samples() {
        assert!(validate_spectral_inputs(&[], 44100).is_err());
    }

    #[test]
    fn test_validate_spectral_valid() {
        assert!(validate_spectral_inputs(&[0.5_f32; 4096], 44100).is_ok());
    }

    // ── beat detection core ────────────────────────────────────────────────

    #[test]
    fn test_beat_onset_detection_empty_gives_no_beats() {
        use oximedia_audio_analysis::beat::{estimate_tempo_from_onsets, BeatTracker};
        // Passing an empty slice to estimate_tempo_from_onsets should not panic
        // and should return a tempo struct (bpm may be 0.0 or some default).
        let tempo = estimate_tempo_from_onsets(&[]);
        let tracker = BeatTracker::new(tempo.clone());
        let beats = tracker.track(&[]);
        assert_eq!(beats.len(), 0, "no onsets → no beats");
    }

    #[test]
    fn test_loudness_meter_config_valid() {
        use oximedia_metering::{LoudnessMeter, MeterConfig, Standard};
        // Constructing a meter with valid parameters must not fail.
        let config = MeterConfig::new(Standard::EbuR128, 48000.0, 2);
        assert!(LoudnessMeter::new(config).is_ok());
    }
}
