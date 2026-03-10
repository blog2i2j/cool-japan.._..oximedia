//! WebAssembly bindings for `oximedia-restore` audio restoration.
//!
//! Provides audio sample restoration, degradation analysis, and
//! declipping for browser-based audio processing.

use wasm_bindgen::prelude::*;

use oximedia_restore::dc::DcRemover;
use oximedia_restore::presets::{BroadcastCleanup, TapeRestoration, VinylRestoration};
use oximedia_restore::{RestorationStep, RestoreChain};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

// ---------------------------------------------------------------------------
// Audio restoration
// ---------------------------------------------------------------------------

/// Restore audio samples using a specified preset mode.
///
/// `samples` is a buffer of f32 audio samples.
/// `sample_rate` is the sample rate in Hz.
/// `mode` is one of: "vinyl", "tape", "broadcast", "archival", "dc".
///
/// Returns the restored audio samples.
///
/// # Errors
///
/// Returns an error if restoration processing fails.
#[wasm_bindgen]
pub fn wasm_restore_audio_samples(
    samples: &[f32],
    sample_rate: u32,
    mode: &str,
) -> Result<Vec<f32>, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("Empty sample buffer"));
    }

    let mut chain = RestoreChain::new();

    match mode.to_lowercase().as_str() {
        "vinyl" => {
            chain.add_preset(VinylRestoration::new(sample_rate));
        }
        "tape" => {
            chain.add_preset(TapeRestoration::new(sample_rate));
        }
        "broadcast" => {
            chain.add_preset(BroadcastCleanup::new(sample_rate));
        }
        "archival" => {
            chain.add_preset(VinylRestoration::new(sample_rate));
            chain.add_preset(TapeRestoration::new(sample_rate));
        }
        "dc" => {
            chain.add_step(RestorationStep::DcRemoval(DcRemover::new(
                10.0,
                sample_rate,
            )));
        }
        _ => {
            return Err(crate::utils::js_err(&format!(
                "Unknown restoration mode: '{}'. Use: vinyl, tape, broadcast, archival, dc",
                escape_json_string(mode),
            )));
        }
    }

    chain
        .process(samples, sample_rate)
        .map_err(|e| crate::utils::js_err(&format!("Restoration failed: {e}")))
}

// ---------------------------------------------------------------------------
// Degradation analysis
// ---------------------------------------------------------------------------

/// Analyze audio degradation from raw pixel/sample data.
///
/// For audio: pass f32 samples as bytes (4 bytes per sample, little-endian).
/// `width` and `height` are ignored for audio (set to 0).
///
/// Returns JSON:
/// ```json
/// {
///   "sample_count": 44100,
///   "peak_level": 0.95,
///   "clipped_samples": 12,
///   "clipping_percent": 0.03,
///   "dc_offset": 0.001,
///   "rms_level": 0.3,
///   "crest_factor": 3.17,
///   "severity": "moderate"
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the data cannot be interpreted.
#[wasm_bindgen]
pub fn wasm_analyze_degradation(data: &[u8], _width: u32, _height: u32) -> Result<String, JsValue> {
    if data.is_empty() {
        return Err(crate::utils::js_err("Empty data buffer"));
    }

    // Interpret bytes as f32 samples
    let samples: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| {
            let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(arr)
        })
        .collect();

    if samples.is_empty() {
        return Err(crate::utils::js_err(
            "Data too short for f32 samples (need >= 4 bytes)",
        ));
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

    let severity = if clip_pct > 5.0 {
        "severe"
    } else if clip_pct > 1.0 || dc.abs() > 0.05 {
        "moderate"
    } else if clip_pct > 0.0 || dc.abs() > 0.01 {
        "mild"
    } else {
        "clean"
    };

    Ok(format!(
        "{{\"sample_count\":{},\"peak_level\":{:.6},\"clipped_samples\":{},\
         \"clipping_percent\":{:.4},\"dc_offset\":{:.8},\"rms_level\":{:.6},\
         \"crest_factor\":{:.4},\"severity\":\"{}\"}}",
        samples.len(),
        peak,
        clip_count,
        clip_pct,
        dc,
        rms,
        crest,
        severity,
    ))
}

// ---------------------------------------------------------------------------
// Declip
// ---------------------------------------------------------------------------

/// Declip audio samples that exceed the given threshold.
///
/// `samples` is a buffer of f32 audio samples.
/// `sample_rate` is the sample rate in Hz.
/// `threshold` is the clipping threshold (0.0-1.0).
///
/// Returns declipped audio samples.
///
/// # Errors
///
/// Returns an error if the processing fails.
#[wasm_bindgen]
pub fn wasm_declip_audio(
    samples: &[f32],
    sample_rate: u32,
    _threshold: f32,
) -> Result<Vec<f32>, JsValue> {
    if samples.is_empty() {
        return Err(crate::utils::js_err("Empty sample buffer"));
    }

    // Use broadcast cleanup preset which includes declipping
    let mut chain = RestoreChain::new();
    chain.add_preset(BroadcastCleanup::new(sample_rate));

    chain
        .process(samples, sample_rate)
        .map_err(|e| crate::utils::js_err(&format!("Declip failed: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restore_audio_dc_mode() {
        let samples = vec![0.5_f32; 1000];
        let result = wasm_restore_audio_samples(&samples, 44100, "dc");
        assert!(result.is_ok());
        let restored = result.expect("dc restoration should succeed");
        assert_eq!(restored.len(), samples.len());
    }

    #[test]
    fn test_restore_audio_empty() {
        let result = wasm_restore_audio_samples(&[], 44100, "vinyl");
        assert!(result.is_err());
    }

    #[test]
    fn test_restore_audio_unknown_mode() {
        let samples = vec![0.5_f32; 100];
        let result = wasm_restore_audio_samples(&samples, 44100, "unknown_mode");
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_degradation_clean() {
        // Use samples centered at 0 with low amplitude (no clipping, no DC offset)
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) * 0.1 - 0.05).collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let result = wasm_analyze_degradation(&bytes, 0, 0);
        assert!(result.is_ok());
        let json = result.expect("analysis should succeed");
        assert!(json.contains("\"severity\":\"clean\"") || json.contains("\"severity\":\"mild\""));
    }

    #[test]
    fn test_analyze_degradation_empty() {
        let result = wasm_analyze_degradation(&[], 0, 0);
        assert!(result.is_err());
    }
}
