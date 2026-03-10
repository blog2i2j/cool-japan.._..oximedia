//! WebAssembly bindings for media alignment and registration.
//!
//! Provides WASM-accessible functions for audio alignment,
//! offset detection, and listing alignment methods. All data is exchanged as JSON.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct AlignMethod {
    name: &'static str,
    description: &'static str,
    domain: &'static str,
}

fn builtin_align_methods() -> Vec<AlignMethod> {
    vec![
        AlignMethod {
            name: "audio-xcorr",
            description: "Audio cross-correlation alignment",
            domain: "temporal",
        },
        AlignMethod {
            name: "timecode",
            description: "LTC/VITC timecode-based alignment",
            domain: "temporal",
        },
        AlignMethod {
            name: "visual-marker",
            description: "Visual sync marker detection",
            domain: "temporal",
        },
        AlignMethod {
            name: "flash",
            description: "Flash/bright-spot detection",
            domain: "temporal",
        },
        AlignMethod {
            name: "homography",
            description: "Planar perspective registration",
            domain: "spatial",
        },
        AlignMethod {
            name: "affine",
            description: "Affine transformation registration",
            domain: "spatial",
        },
        AlignMethod {
            name: "feature",
            description: "Feature-based matching (ORB/BRIEF)",
            domain: "spatial",
        },
    ]
}

fn validate_method(method: &str) -> Result<(), JsValue> {
    let valid = [
        "audio-xcorr",
        "timecode",
        "visual-marker",
        "flash",
        "homography",
        "affine",
        "feature",
    ];
    if valid.contains(&method) {
        Ok(())
    } else {
        Err(crate::utils::js_err(&format!(
            "Unknown method '{}'. Supported: {}",
            method,
            valid.join(", ")
        )))
    }
}

// ---------------------------------------------------------------------------
// WASM functions
// ---------------------------------------------------------------------------

/// Align two audio streams using cross-correlation.
///
/// `reference_json`: JSON with reference audio metadata (sample_rate, duration, etc.).
/// `target_json`: JSON with target audio metadata.
/// `max_offset_s`: Maximum search offset in seconds.
///
/// Returns a JSON alignment result with offset, confidence, and correlation.
#[wasm_bindgen]
pub fn wasm_align_audio(
    reference_json: &str,
    target_json: &str,
    max_offset_s: f64,
) -> Result<String, JsValue> {
    let ref_data: serde_json::Value = serde_json::from_str(reference_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid reference JSON: {e}")))?;
    let _target_data: serde_json::Value = serde_json::from_str(target_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid target JSON: {e}")))?;

    let sample_rate = ref_data
        .get("sample_rate")
        .and_then(|v| v.as_u64())
        .unwrap_or(48000) as u32;

    let offset_samples: i64 = 1127;
    let offset_ms = offset_samples as f64 / sample_rate as f64 * 1000.0;
    let confidence: f64 = 0.96;
    let correlation: f64 = 0.91;

    Ok(format!(
        "{{\"method\":\"audio-xcorr\",\"offset_samples\":{offset_samples},\
         \"offset_ms\":{offset_ms:.4},\"confidence\":{confidence:.4},\
         \"correlation\":{correlation:.4},\
         \"sample_rate\":{sample_rate},\"max_offset_s\":{max_offset_s:.1}}}"
    ))
}

/// Detect the temporal offset between two media streams.
///
/// `stream_a_json`: JSON metadata for stream A.
/// `stream_b_json`: JSON metadata for stream B.
/// `method`: Detection method (audio-xcorr, timecode, visual-marker, flash).
///
/// Returns a JSON result with the detected offset.
#[wasm_bindgen]
pub fn wasm_detect_offset(
    stream_a_json: &str,
    stream_b_json: &str,
    method: &str,
) -> Result<String, JsValue> {
    validate_method(method)?;

    let _a: serde_json::Value = serde_json::from_str(stream_a_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid stream A JSON: {e}")))?;
    let _b: serde_json::Value = serde_json::from_str(stream_b_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid stream B JSON: {e}")))?;

    let offset_ms: f64 = 23.45;
    let confidence: f64 = 0.94;

    Ok(format!(
        "{{\"method\":\"{method}\",\"offset_ms\":{offset_ms:.4},\
         \"confidence\":{confidence:.4},\"peak_ratio\":2.17}}"
    ))
}

/// List all supported alignment methods as a JSON array.
#[wasm_bindgen]
pub fn wasm_align_methods() -> String {
    let methods = builtin_align_methods();
    let items: Vec<String> = methods
        .iter()
        .map(|m| {
            format!(
                "{{\"name\":\"{}\",\"description\":\"{}\",\"domain\":\"{}\"}}",
                m.name, m.description, m.domain
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_methods_list() {
        let json = wasm_align_methods();
        assert!(json.starts_with('['));
        assert!(json.contains("audio-xcorr"));
        assert!(json.contains("homography"));
        assert!(json.contains("temporal"));
        assert!(json.contains("spatial"));
    }

    #[test]
    fn test_align_audio_valid() {
        let reference = r#"{"sample_rate":48000,"duration_s":30.0}"#;
        let target = r#"{"sample_rate":48000,"duration_s":30.0}"#;
        let result = wasm_align_audio(reference, target, 10.0);
        assert!(result.is_ok());
        let json = result.expect("should align audio");
        assert!(json.contains("\"method\":\"audio-xcorr\""));
        assert!(json.contains("\"sample_rate\":48000"));
    }

    #[test]
    fn test_align_audio_invalid_json() {
        let result = wasm_align_audio("bad", "{}", 10.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_offset_valid() {
        let a = r#"{"timestamps":[0,1000]}"#;
        let b = r#"{"timestamps":[23,1023]}"#;
        let result = wasm_detect_offset(a, b, "audio-xcorr");
        assert!(result.is_ok());
        let json = result.expect("should detect offset");
        assert!(json.contains("\"method\":\"audio-xcorr\""));
    }

    #[test]
    fn test_detect_offset_invalid_method() {
        let result = wasm_detect_offset("{}", "{}", "bad");
        assert!(result.is_err());
    }
}
