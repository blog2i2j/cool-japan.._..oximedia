//! WebAssembly bindings for `oximedia-forensics` image forensic analysis.
//!
//! All functions accept image file bytes (JPEG, PNG, etc.) and return JSON
//! strings with analysis results. Designed for synchronous browser usage.

use wasm_bindgen::prelude::*;

use oximedia_forensics::{ConfidenceLevel, ForensicsAnalyzer, ForensicsConfig};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn confidence_level_str(level: ConfidenceLevel) -> &'static str {
    match level {
        ConfidenceLevel::VeryLow => "very_low",
        ConfidenceLevel::Low => "low",
        ConfidenceLevel::Medium => "medium",
        ConfidenceLevel::High => "high",
        ConfidenceLevel::VeryHigh => "very_high",
    }
}

fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

/// Build a JSON array from a Vec<String>.
fn json_string_array(items: &[String]) -> String {
    let entries: Vec<String> = items
        .iter()
        .map(|s| format!("\"{}\"", escape_json_string(s)))
        .collect();
    format!("[{}]", entries.join(","))
}

// ---------------------------------------------------------------------------
// ELA Analysis
// ---------------------------------------------------------------------------

/// Perform Error Level Analysis on image data.
///
/// `image_data` should contain image file bytes (JPEG, PNG, etc.).
/// `width` and `height` are informational (actual dimensions from image data).
/// `quality` is the JPEG recompression quality (1-100).
///
/// Returns the ELA difference image as raw grayscale bytes (one byte per pixel,
/// normalized to 0-255). Higher values indicate more error.
///
/// # Errors
///
/// Returns an error if the image cannot be parsed or analysis fails.
#[wasm_bindgen]
pub fn wasm_ela_analysis(
    image_data: &[u8],
    _width: u32,
    _height: u32,
    _quality: u32,
) -> Result<Vec<u8>, JsValue> {
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
        .analyze(image_data)
        .map_err(|e| crate::utils::js_err(&format!("ELA analysis failed: {e}")))?;

    // Extract anomaly map from the ELA test result
    for test in report.tests.values() {
        if let Some(ref map) = test.anomaly_map {
            let max_val = map.iter().cloned().fold(0.0_f64, f64::max);
            let scale = if max_val > 0.0 { 255.0 / max_val } else { 0.0 };
            let bytes: Vec<u8> = map
                .iter()
                .map(|&v| (v * scale).round().min(255.0).max(0.0) as u8)
                .collect();
            return Ok(bytes);
        }
    }

    Err(crate::utils::js_err(
        "ELA analysis did not produce an anomaly map",
    ))
}

// ---------------------------------------------------------------------------
// Noise Analysis
// ---------------------------------------------------------------------------

/// Analyze noise patterns in an image for tampering detection.
///
/// Returns JSON:
/// ```json
/// {
///   "name": "Noise Pattern Analysis",
///   "tampering_detected": false,
///   "confidence": 0.15,
///   "confidence_level": "very_low",
///   "findings": ["PRNU pattern extracted...", ...]
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the image cannot be parsed.
#[wasm_bindgen]
pub fn wasm_noise_analysis(
    image_data: &[u8],
    _width: u32,
    _height: u32,
) -> Result<String, JsValue> {
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
        .analyze(image_data)
        .map_err(|e| crate::utils::js_err(&format!("Noise analysis failed: {e}")))?;

    if let Some(test) = report.tests.values().next() {
        let findings = json_string_array(&test.findings);
        return Ok(format!(
            "{{\"name\":\"{}\",\"tampering_detected\":{},\"confidence\":{},\
             \"confidence_level\":\"{}\",\"findings\":{}}}",
            escape_json_string(&test.name),
            test.tampering_detected,
            test.confidence,
            confidence_level_str(test.confidence_level()),
            findings,
        ));
    }

    Err(crate::utils::js_err("Noise analysis produced no results"))
}

// ---------------------------------------------------------------------------
// Compression Analysis
// ---------------------------------------------------------------------------

/// Analyze compression artifacts in an image.
///
/// Returns JSON with the same structure as `wasm_noise_analysis`.
///
/// # Errors
///
/// Returns an error if the image cannot be parsed.
#[wasm_bindgen]
pub fn wasm_compression_analysis(
    image_data: &[u8],
    _width: u32,
    _height: u32,
) -> Result<String, JsValue> {
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
        .analyze(image_data)
        .map_err(|e| crate::utils::js_err(&format!("Compression analysis failed: {e}")))?;

    if let Some(test) = report.tests.values().next() {
        let findings = json_string_array(&test.findings);
        return Ok(format!(
            "{{\"name\":\"{}\",\"tampering_detected\":{},\"confidence\":{},\
             \"confidence_level\":\"{}\",\"findings\":{}}}",
            escape_json_string(&test.name),
            test.tampering_detected,
            test.confidence,
            confidence_level_str(test.confidence_level()),
            findings,
        ));
    }

    Err(crate::utils::js_err(
        "Compression analysis produced no results",
    ))
}

// ---------------------------------------------------------------------------
// Full Forensic Report
// ---------------------------------------------------------------------------

/// Run a comprehensive forensic report on an image.
///
/// Returns JSON:
/// ```json
/// {
///   "tampering_detected": false,
///   "overall_confidence": 0.2,
///   "summary": "No significant tampering detected...",
///   "recommendations": [],
///   "tests": [
///     {"name": "...", "tampering_detected": false, "confidence": 0.1, ...},
///     ...
///   ]
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the image cannot be parsed or analysis fails.
#[wasm_bindgen]
pub fn wasm_forensic_report(
    image_data: &[u8],
    _width: u32,
    _height: u32,
) -> Result<String, JsValue> {
    let analyzer = ForensicsAnalyzer::new();
    let report = analyzer
        .analyze(image_data)
        .map_err(|e| crate::utils::js_err(&format!("Forensic report failed: {e}")))?;

    // Build tests JSON array
    let tests_json: Vec<String> = report
        .tests
        .values()
        .map(|test| {
            let findings = json_string_array(&test.findings);
            format!(
                "{{\"name\":\"{}\",\"tampering_detected\":{},\"confidence\":{},\
                 \"confidence_level\":\"{}\",\"findings\":{}}}",
                escape_json_string(&test.name),
                test.tampering_detected,
                test.confidence,
                confidence_level_str(test.confidence_level()),
                findings,
            )
        })
        .collect();

    let recommendations = json_string_array(&report.recommendations);

    Ok(format!(
        "{{\"tampering_detected\":{},\"overall_confidence\":{},\
         \"summary\":\"{}\",\"recommendations\":{},\"tests\":[{}]}}",
        report.tampering_detected,
        report.overall_confidence,
        escape_json_string(&report.summary),
        recommendations,
        tests_json.join(","),
    ))
}

// ---------------------------------------------------------------------------
// Quick integrity check
// ---------------------------------------------------------------------------

/// Quick integrity check on image data.
///
/// Returns `true` if the image appears authentic (no tampering detected).
///
/// # Errors
///
/// Returns an error if the image cannot be parsed.
#[wasm_bindgen]
pub fn wasm_check_integrity(image_data: &[u8], _width: u32, _height: u32) -> Result<bool, JsValue> {
    let analyzer = ForensicsAnalyzer::new();
    let report = analyzer
        .analyze(image_data)
        .map_err(|e| crate::utils::js_err(&format!("Integrity check failed: {e}")))?;

    Ok(!report.tampering_detected)
}

// ---------------------------------------------------------------------------
// List available tests
// ---------------------------------------------------------------------------

/// List available forensic tests as a JSON array.
///
/// Returns:
/// ```json
/// ["compression","ela","noise","metadata","geometric","lighting"]
/// ```
#[wasm_bindgen]
pub fn wasm_forensic_tests() -> Result<String, JsValue> {
    Ok("[\"compression\",\"ela\",\"noise\",\"metadata\",\"geometric\",\"lighting\"]".to_string())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forensic_tests_list() {
        let result = wasm_forensic_tests();
        assert!(result.is_ok());
        let json = result.expect("should produce tests list");
        assert!(json.contains("compression"));
        assert!(json.contains("ela"));
        assert!(json.contains("noise"));
        assert!(json.contains("metadata"));
        assert!(json.contains("geometric"));
        assert!(json.contains("lighting"));
    }

    #[test]
    fn test_escape_json_string() {
        let escaped = escape_json_string("hello \"world\"\nnewline");
        assert_eq!(escaped, "hello \\\"world\\\"\\nnewline");
    }

    #[test]
    fn test_json_string_array() {
        let items = vec!["a".to_string(), "b".to_string()];
        let json = json_string_array(&items);
        assert_eq!(json, "[\"a\",\"b\"]");
    }

    #[test]
    fn test_json_string_array_empty() {
        let items: Vec<String> = Vec::new();
        let json = json_string_array(&items);
        assert_eq!(json, "[]");
    }

    #[test]
    fn test_confidence_level_str_mapping() {
        assert_eq!(confidence_level_str(ConfidenceLevel::VeryLow), "very_low");
        assert_eq!(confidence_level_str(ConfidenceLevel::Low), "low");
        assert_eq!(confidence_level_str(ConfidenceLevel::Medium), "medium");
        assert_eq!(confidence_level_str(ConfidenceLevel::High), "high");
        assert_eq!(confidence_level_str(ConfidenceLevel::VeryHigh), "very_high");
    }
}
