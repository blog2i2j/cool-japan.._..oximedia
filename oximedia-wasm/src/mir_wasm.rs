//! WebAssembly bindings for Music Information Retrieval (MIR).
//!
//! Provides tempo detection, key detection, structural segmentation,
//! chord analysis, and full MIR analysis from the browser.

use wasm_bindgen::prelude::*;

use oximedia_mir::{MirAnalyzer, MirConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_samples(samples: &[f32]) -> Result<(), JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("samples must not be empty"));
    }
    Ok(())
}

fn validate_sample_rate(sample_rate: u32) -> Result<f32, JsValue> {
    if sample_rate == 0 {
        return Err(crate::utils::js_err("sample_rate must be > 0"));
    }
    Ok(sample_rate as f32)
}

// ---------------------------------------------------------------------------
// Tempo detection
// ---------------------------------------------------------------------------

/// Detect the tempo (BPM) of an audio signal.
///
/// Parameters:
/// - `samples`: Mono audio samples (f32) in [-1.0, 1.0].
/// - `sample_rate`: Sample rate in Hz (e.g. 44100).
///
/// Returns a JSON string:
/// ```json
/// {
///   "bpm": 120.0,
///   "confidence": 0.85,
///   "stability": 0.92,
///   "beats": [0.5, 1.0, 1.5, ...]
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_detect_tempo(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    validate_samples(samples)?;
    let sr = validate_sample_rate(sample_rate)?;

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
        .analyze(samples, sr)
        .map_err(|e| crate::utils::js_err(&format!("Tempo analysis failed: {e}")))?;

    let tempo = result
        .tempo
        .ok_or_else(|| crate::utils::js_err("Tempo detection returned no result"))?;

    let beat_times = result
        .beat
        .as_ref()
        .map(|b| b.beat_times.clone())
        .unwrap_or_default();

    let value = serde_json::json!({
        "bpm": tempo.bpm,
        "confidence": tempo.confidence,
        "stability": tempo.stability,
        "beats": beat_times,
    });

    serde_json::to_string(&value)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Key detection
// ---------------------------------------------------------------------------

/// Detect the musical key of an audio signal.
///
/// Parameters:
/// - `samples`: Mono audio samples (f32) in [-1.0, 1.0].
/// - `sample_rate`: Sample rate in Hz.
///
/// Returns a JSON string:
/// ```json
/// {
///   "key": "C major",
///   "root": 0,
///   "mode": "major",
///   "confidence": 0.78
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_detect_key(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    validate_samples(samples)?;
    let sr = validate_sample_rate(sample_rate)?;

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
        .analyze(samples, sr)
        .map_err(|e| crate::utils::js_err(&format!("Key detection failed: {e}")))?;

    let key = result
        .key
        .ok_or_else(|| crate::utils::js_err("Key detection returned no result"))?;

    let value = serde_json::json!({
        "key": key.key,
        "root": key.root,
        "mode": if key.is_major { "major" } else { "minor" },
        "confidence": key.confidence,
    });

    serde_json::to_string(&value)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Audio segmentation
// ---------------------------------------------------------------------------

/// Segment audio into structural sections (intro, verse, chorus, etc.).
///
/// Parameters:
/// - `samples`: Mono audio samples (f32).
/// - `sample_rate`: Sample rate in Hz.
/// - `min_duration`: Minimum segment duration in seconds (e.g. 2.0).
///
/// Returns a JSON array of segments:
/// ```json
/// [
///   {"start": 0.0, "end": 15.2, "label": "intro", "confidence": 0.9},
///   {"start": 15.2, "end": 45.0, "label": "verse", "confidence": 0.85},
///   ...
/// ]
/// ```
#[wasm_bindgen]
pub fn wasm_segment_audio(
    samples: &[f32],
    sample_rate: u32,
    min_duration: f64,
) -> Result<String, JsValue> {
    validate_samples(samples)?;
    let sr = validate_sample_rate(sample_rate)?;

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
        .analyze(samples, sr)
        .map_err(|e| crate::utils::js_err(&format!("Segmentation failed: {e}")))?;

    let structure = result
        .structure
        .ok_or_else(|| crate::utils::js_err("Structure analysis returned no result"))?;

    let min_dur = min_duration as f32;
    let segments: Vec<serde_json::Value> = structure
        .segments
        .iter()
        .filter(|s| (s.end - s.start) >= min_dur)
        .map(|s| {
            serde_json::json!({
                "start": s.start,
                "end": s.end,
                "label": s.label,
                "confidence": s.confidence,
                "duration": s.end - s.start,
            })
        })
        .collect();

    serde_json::to_string(&segments)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Chord detection
// ---------------------------------------------------------------------------

/// Detect chord progression in audio.
///
/// Parameters:
/// - `samples`: Mono audio samples (f32).
/// - `sample_rate`: Sample rate in Hz.
///
/// Returns a JSON array of chord events:
/// ```json
/// [
///   {"start": 0.0, "end": 2.5, "label": "C", "confidence": 0.8},
///   {"start": 2.5, "end": 5.0, "label": "Am", "confidence": 0.75},
///   ...
/// ]
/// ```
#[wasm_bindgen]
pub fn wasm_detect_chords(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    validate_samples(samples)?;
    let sr = validate_sample_rate(sample_rate)?;

    let config = MirConfig {
        enable_beat_tracking: false,
        enable_key_detection: false,
        enable_chord_recognition: true,
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
        .analyze(samples, sr)
        .map_err(|e| crate::utils::js_err(&format!("Chord detection failed: {e}")))?;

    let chord_result = result
        .chord
        .ok_or_else(|| crate::utils::js_err("Chord recognition returned no result"))?;

    let chords: Vec<serde_json::Value> = chord_result
        .chords
        .iter()
        .map(|c| {
            serde_json::json!({
                "start": c.start,
                "end": c.end,
                "label": c.label,
                "confidence": c.confidence,
            })
        })
        .collect();

    serde_json::to_string(&chords)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Full MIR analysis
// ---------------------------------------------------------------------------

/// Perform a full MIR analysis on audio.
///
/// Parameters:
/// - `samples`: Mono audio samples (f32).
/// - `sample_rate`: Sample rate in Hz.
///
/// Returns a comprehensive JSON object with all analysis results:
/// ```json
/// {
///   "duration": 180.5,
///   "sample_rate": 44100,
///   "tempo": {"bpm": 120, "confidence": 0.9, "stability": 0.85},
///   "key": {"key": "C major", "root": 0, "mode": "major", "confidence": 0.8},
///   "structure": {"segment_count": 5, "complexity": 0.7, "segments": [...]},
///   "chords": {"chord_count": 45, "complexity": 0.6},
///   "genre": {"top_genre": "rock", "confidence": 0.7},
///   "mood": {"valence": 0.6, "arousal": 0.8}
/// }
/// ```
#[wasm_bindgen]
pub fn wasm_mir_analyze(samples: &[f32], sample_rate: u32) -> Result<String, JsValue> {
    validate_samples(samples)?;
    let sr = validate_sample_rate(sample_rate)?;

    let config = MirConfig::default();
    let analyzer = MirAnalyzer::new(config);
    let result = analyzer
        .analyze(samples, sr)
        .map_err(|e| crate::utils::js_err(&format!("MIR analysis failed: {e}")))?;

    let value = serde_json::json!({
        "duration": result.duration,
        "sample_rate": result.sample_rate,
        "tempo": result.tempo.as_ref().map(|t| serde_json::json!({
            "bpm": t.bpm,
            "confidence": t.confidence,
            "stability": t.stability,
        })),
        "key": result.key.as_ref().map(|k| serde_json::json!({
            "key": k.key,
            "root": k.root,
            "mode": if k.is_major { "major" } else { "minor" },
            "confidence": k.confidence,
        })),
        "structure": result.structure.as_ref().map(|s| serde_json::json!({
            "segment_count": s.segments.len(),
            "complexity": s.complexity,
            "segments": s.segments.iter().map(|seg| serde_json::json!({
                "label": seg.label,
                "start": seg.start,
                "end": seg.end,
                "confidence": seg.confidence,
            })).collect::<Vec<_>>(),
        })),
        "chords": result.chord.as_ref().map(|c| serde_json::json!({
            "chord_count": c.chords.len(),
            "complexity": c.complexity,
            "progressions": c.progressions,
        })),
        "genre": result.genre.as_ref().map(|g| serde_json::json!({
            "top_genre": g.top_genre_name,
            "confidence": g.top_genre_confidence,
        })),
        "mood": result.mood.as_ref().map(|m| serde_json::json!({
            "valence": m.valence,
            "arousal": m.arousal,
        })),
    });

    serde_json::to_string(&value)
        .map_err(|e| crate::utils::js_err(&format!("JSON serialization failed: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize) -> Vec<f32> {
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / 44100.0;
            samples.push((2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5);
        }
        samples
    }

    #[test]
    fn test_detect_tempo_empty() {
        let result = wasm_detect_tempo(&[], 44100);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_tempo_valid() {
        let samples = make_samples(44100);
        let result = wasm_detect_tempo(&samples, 44100);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("bpm"));
        assert!(json.contains("confidence"));
    }

    #[test]
    fn test_detect_key_valid() {
        let samples = make_samples(44100);
        let result = wasm_detect_key(&samples, 44100);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("key"));
        assert!(json.contains("mode"));
    }

    #[test]
    fn test_detect_chords_valid() {
        let samples = make_samples(44100);
        let result = wasm_detect_chords(&samples, 44100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_segment_audio_valid() {
        let samples = make_samples(44100);
        let result = wasm_segment_audio(&samples, 44100, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mir_analyze_full() {
        let samples = make_samples(44100);
        let result = wasm_mir_analyze(&samples, 44100);
        assert!(result.is_ok());
        let json = result.expect("should succeed");
        assert!(json.contains("duration"));
        assert!(json.contains("sample_rate"));
    }

    #[test]
    fn test_zero_sample_rate() {
        let samples = make_samples(100);
        let result = wasm_detect_tempo(&samples, 0);
        assert!(result.is_err());
    }
}
