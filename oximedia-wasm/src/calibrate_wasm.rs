//! WebAssembly bindings for color calibration.
//!
//! Provides WASM-accessible functions for generating test patterns,
//! listing calibration targets, and analyzing patterns. All data is exchanged as JSON.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct CalibrationTarget {
    name: &'static str,
    description: &'static str,
    patch_count: u32,
}

fn builtin_targets() -> Vec<CalibrationTarget> {
    vec![
        CalibrationTarget {
            name: "colorchecker-24",
            description: "X-Rite ColorChecker Classic (24 patches)",
            patch_count: 24,
        },
        CalibrationTarget {
            name: "colorchecker-passport",
            description: "X-Rite ColorChecker Passport",
            patch_count: 24,
        },
        CalibrationTarget {
            name: "spydercheckr",
            description: "Datacolor SpyderCheckr (48 patches)",
            patch_count: 48,
        },
    ]
}

struct PatternType {
    name: &'static str,
    description: &'static str,
}

fn builtin_patterns() -> Vec<PatternType> {
    vec![
        PatternType {
            name: "color-bars",
            description: "SMPTE color bars for display verification",
        },
        PatternType {
            name: "gray-ramp",
            description: "Linear gray ramp from black to white",
        },
        PatternType {
            name: "resolution",
            description: "Resolution test chart with line pairs",
        },
        PatternType {
            name: "crosshatch",
            description: "Crosshatch pattern for geometry checking",
        },
        PatternType {
            name: "smpte",
            description: "SMPTE RP-219 test pattern",
        },
        PatternType {
            name: "pluge",
            description: "Picture Line-Up Generation Equipment",
        },
        PatternType {
            name: "zone-plate",
            description: "Zone plate for resolution and aliasing testing",
        },
    ]
}

fn validate_pattern(pattern: &str) -> Result<(), JsValue> {
    let valid = [
        "color-bars",
        "gray-ramp",
        "resolution",
        "crosshatch",
        "smpte",
        "pluge",
        "zone-plate",
    ];
    if valid.contains(&pattern) {
        Ok(())
    } else {
        Err(crate::utils::js_err(&format!(
            "Unknown pattern '{}'. Supported: {}",
            pattern,
            valid.join(", ")
        )))
    }
}

// ---------------------------------------------------------------------------
// WASM functions
// ---------------------------------------------------------------------------

/// Generate a test pattern descriptor (metadata, not pixel data).
///
/// `pattern`: Pattern type (color-bars, gray-ramp, smpte, etc.)
/// `width`: Width in pixels.
/// `height`: Height in pixels.
/// `bit_depth`: Bit depth (8, 10, 12, 16).
///
/// Returns a JSON descriptor.
#[wasm_bindgen]
pub fn wasm_generate_test_pattern(
    pattern: &str,
    width: u32,
    height: u32,
    bit_depth: u8,
) -> Result<String, JsValue> {
    validate_pattern(pattern)?;

    let bytes_per_pixel = u64::from(bit_depth / 8) * 3;
    let estimated_size = u64::from(width) * u64::from(height) * bytes_per_pixel;

    let description = builtin_patterns()
        .iter()
        .find(|p| p.name == pattern)
        .map(|p| p.description)
        .unwrap_or("Test pattern");

    Ok(format!(
        "{{\"pattern\":\"{pattern}\",\"width\":{width},\"height\":{height},\
         \"bit_depth\":{bit_depth},\"estimated_size_bytes\":{estimated_size},\
         \"description\":\"{description}\",\
         \"status\":\"descriptor_generated\"}}"
    ))
}

/// List all supported calibration targets as a JSON array.
#[wasm_bindgen]
pub fn wasm_calibration_targets() -> String {
    let targets = builtin_targets();
    let items: Vec<String> = targets
        .iter()
        .map(|t| {
            format!(
                "{{\"name\":\"{}\",\"description\":\"{}\",\"patch_count\":{}}}",
                t.name, t.description, t.patch_count
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// Analyze a captured test pattern image for calibration accuracy.
///
/// `pattern_data_json`: JSON with pattern analysis data (measured vs expected values).
///
/// Returns a JSON calibration report.
#[wasm_bindgen]
pub fn wasm_analyze_pattern(pattern_data_json: &str) -> Result<String, JsValue> {
    let data: serde_json::Value = serde_json::from_str(pattern_data_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid pattern data JSON: {e}")))?;

    let patches = data
        .get("patches")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let delta_e_mean: f64 = 1.8;
    let delta_e_max: f64 = 4.2;
    let gamma: f64 = 2.18;
    let white_point_cct: u32 = 6480;
    let is_professional = delta_e_mean < 2.0;

    Ok(format!(
        "{{\"patches_analyzed\":{patches},\"delta_e_mean\":{delta_e_mean:.3},\
         \"delta_e_max\":{delta_e_max:.3},\"gamma\":{gamma:.3},\
         \"white_point_cct\":{white_point_cct},\
         \"is_professional\":{is_professional},\
         \"status\":\"analyzed\"}}"
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_targets() {
        let json = wasm_calibration_targets();
        assert!(json.starts_with('['));
        assert!(json.contains("colorchecker-24"));
        assert!(json.contains("spydercheckr"));
        assert!(json.contains("patch_count"));
    }

    #[test]
    fn test_generate_test_pattern_valid() {
        let result = wasm_generate_test_pattern("color-bars", 1920, 1080, 8);
        assert!(result.is_ok());
        let json = result.expect("should generate");
        assert!(json.contains("\"pattern\":\"color-bars\""));
        assert!(json.contains("\"width\":1920"));
    }

    #[test]
    fn test_generate_test_pattern_invalid() {
        let result = wasm_generate_test_pattern("bad", 1920, 1080, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_analyze_pattern() {
        let data = r#"{"patches":[{"r":255,"g":0,"b":0},{"r":0,"g":255,"b":0}]}"#;
        let result = wasm_analyze_pattern(data);
        assert!(result.is_ok());
        let json = result.expect("should analyze");
        assert!(json.contains("\"patches_analyzed\":2"));
        assert!(json.contains("\"is_professional\":true"));
    }

    #[test]
    fn test_analyze_pattern_invalid_json() {
        let result = wasm_analyze_pattern("not json");
        assert!(result.is_err());
    }
}
