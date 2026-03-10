//! WebAssembly bindings for `oximedia-conform` delivery specification checking.
//!
//! Provides WASM-accessible functions for checking media properties against
//! broadcast/streaming delivery specs. All data is exchanged as JSON strings.

use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal types and helpers
// ---------------------------------------------------------------------------

struct DeliverySpec {
    name: &'static str,
    description: &'static str,
    min_width: Option<u32>,
    max_width: Option<u32>,
    min_height: Option<u32>,
    max_height: Option<u32>,
    frame_rates: &'static [f64],
    sample_rates: &'static [u32],
    max_loudness_lufs: Option<f64>,
    max_true_peak_dbtp: Option<f64>,
    max_bitrate_kbps: Option<u32>,
    max_file_size_mb: Option<u64>,
    container: Option<&'static str>,
}

fn builtin_specs() -> Vec<DeliverySpec> {
    vec![
        DeliverySpec {
            name: "broadcast_hd",
            description: "Broadcast HD delivery (EBU R128)",
            min_width: Some(1920),
            max_width: Some(1920),
            min_height: Some(1080),
            max_height: Some(1080),
            frame_rates: &[25.0, 29.97, 30.0, 50.0, 59.94],
            sample_rates: &[48000],
            max_loudness_lufs: Some(-23.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(50_000),
            max_file_size_mb: None,
            container: None,
        },
        DeliverySpec {
            name: "netflix",
            description: "Netflix streaming delivery",
            min_width: Some(1920),
            max_width: Some(3840),
            min_height: Some(1080),
            max_height: Some(2160),
            frame_rates: &[23.976, 24.0, 25.0, 29.97],
            sample_rates: &[48000],
            max_loudness_lufs: Some(-27.0),
            max_true_peak_dbtp: Some(-2.0),
            max_bitrate_kbps: Some(80_000),
            max_file_size_mb: None,
            container: None,
        },
        DeliverySpec {
            name: "youtube",
            description: "YouTube streaming delivery",
            min_width: Some(426),
            max_width: Some(3840),
            min_height: Some(240),
            max_height: Some(2160),
            frame_rates: &[24.0, 25.0, 30.0, 48.0, 50.0, 60.0],
            sample_rates: &[44100, 48000],
            max_loudness_lufs: Some(-14.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(85_000),
            max_file_size_mb: Some(256_000),
            container: Some("mp4"),
        },
        DeliverySpec {
            name: "theatrical_dcp",
            description: "Digital Cinema Package delivery",
            min_width: Some(2048),
            max_width: Some(4096),
            min_height: Some(858),
            max_height: Some(2160),
            frame_rates: &[24.0, 25.0, 30.0, 48.0],
            sample_rates: &[48000, 96000],
            max_loudness_lufs: Some(-20.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(250_000),
            max_file_size_mb: None,
            container: None,
        },
        DeliverySpec {
            name: "podcast",
            description: "Podcast audio delivery",
            min_width: None,
            max_width: None,
            min_height: None,
            max_height: None,
            frame_rates: &[],
            sample_rates: &[44100, 48000],
            max_loudness_lufs: Some(-16.0),
            max_true_peak_dbtp: Some(-1.0),
            max_bitrate_kbps: Some(320),
            max_file_size_mb: Some(500),
            container: None,
        },
    ]
}

fn spec_to_json(spec: &DeliverySpec) -> String {
    let fr: Vec<String> = spec.frame_rates.iter().map(|f| format!("{f:.3}")).collect();
    let sr: Vec<String> = spec.sample_rates.iter().map(|s| s.to_string()).collect();
    let ct = spec.container.unwrap_or("");

    format!(
        "{{\"name\":\"{}\",\"description\":\"{}\",\
         \"min_width\":{},\"max_width\":{},\
         \"min_height\":{},\"max_height\":{},\
         \"frame_rates\":[{}],\"sample_rates\":[{}],\
         \"max_loudness_lufs\":{},\"max_true_peak_dbtp\":{},\
         \"max_bitrate_kbps\":{},\"max_file_size_mb\":{},\
         \"container\":\"{}\"}}",
        spec.name,
        spec.description,
        opt_u32_json(spec.min_width),
        opt_u32_json(spec.max_width),
        opt_u32_json(spec.min_height),
        opt_u32_json(spec.max_height),
        fr.join(","),
        sr.join(","),
        opt_f64_json(spec.max_loudness_lufs),
        opt_f64_json(spec.max_true_peak_dbtp),
        opt_u32_json(spec.max_bitrate_kbps),
        opt_u64_json(spec.max_file_size_mb),
        ct,
    )
}

fn opt_u32_json(v: Option<u32>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

fn opt_u64_json(v: Option<u64>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "null".to_string(),
    }
}

fn opt_f64_json(v: Option<f64>) -> String {
    match v {
        Some(n) => format!("{n:.1}"),
        None => "null".to_string(),
    }
}

struct CheckResult {
    name: String,
    passed: bool,
    actual: String,
    expected: String,
    severity: String,
}

fn check_range_from_map(
    spec_map: &HashMap<String, serde_json::Value>,
    props: &HashMap<String, serde_json::Value>,
    prop_key: &str,
    min_key: &str,
    max_key: &str,
    checks: &mut Vec<CheckResult>,
) {
    if let Some(serde_json::Value::Number(val_n)) = props.get(prop_key) {
        if let Some(val) = val_n.as_u64().map(|v| v as u32) {
            if let Some(min_v) = spec_map
                .get(min_key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
            {
                let passed = val >= min_v;
                checks.push(CheckResult {
                    name: format!("min_{prop_key}"),
                    passed,
                    actual: val.to_string(),
                    expected: format!(">= {min_v}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
            if let Some(max_v) = spec_map
                .get(max_key)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
            {
                let passed = val <= max_v;
                checks.push(CheckResult {
                    name: format!("max_{prop_key}"),
                    passed,
                    actual: val.to_string(),
                    expected: format!("<= {max_v}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }
}

fn run_checks(spec: &DeliverySpec, props: &HashMap<String, serde_json::Value>) -> Vec<CheckResult> {
    let mut checks = Vec::new();

    // Width checks
    if let Some(serde_json::Value::Number(w)) = props.get("width") {
        if let Some(wv) = w.as_u64().map(|v| v as u32) {
            if let Some(min_w) = spec.min_width {
                let passed = wv >= min_w;
                checks.push(CheckResult {
                    name: "min_width".to_string(),
                    passed,
                    actual: wv.to_string(),
                    expected: format!(">= {min_w}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
            if let Some(max_w) = spec.max_width {
                let passed = wv <= max_w;
                checks.push(CheckResult {
                    name: "max_width".to_string(),
                    passed,
                    actual: wv.to_string(),
                    expected: format!("<= {max_w}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // Height checks
    if let Some(serde_json::Value::Number(h)) = props.get("height") {
        if let Some(hv) = h.as_u64().map(|v| v as u32) {
            if let Some(min_h) = spec.min_height {
                let passed = hv >= min_h;
                checks.push(CheckResult {
                    name: "min_height".to_string(),
                    passed,
                    actual: hv.to_string(),
                    expected: format!(">= {min_h}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
            if let Some(max_h) = spec.max_height {
                let passed = hv <= max_h;
                checks.push(CheckResult {
                    name: "max_height".to_string(),
                    passed,
                    actual: hv.to_string(),
                    expected: format!("<= {max_h}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // Frame rate check
    if let Some(serde_json::Value::Number(fps_n)) = props.get("frame_rate") {
        if let Some(fps) = fps_n.as_f64() {
            if !spec.frame_rates.is_empty() {
                let passed = spec
                    .frame_rates
                    .iter()
                    .any(|&allowed| (allowed - fps).abs() < 0.01);
                checks.push(CheckResult {
                    name: "frame_rate".to_string(),
                    passed,
                    actual: format!("{fps:.3}"),
                    expected: format!("{:?}", spec.frame_rates),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // Sample rate check
    if let Some(serde_json::Value::Number(sr_n)) = props.get("sample_rate") {
        if let Some(sr) = sr_n.as_u64().map(|v| v as u32) {
            if !spec.sample_rates.is_empty() {
                let passed = spec.sample_rates.contains(&sr);
                checks.push(CheckResult {
                    name: "sample_rate".to_string(),
                    passed,
                    actual: sr.to_string(),
                    expected: format!("{:?}", spec.sample_rates),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // Loudness check
    if let Some(serde_json::Value::Number(lufs_n)) = props.get("loudness_lufs") {
        if let Some(lufs) = lufs_n.as_f64() {
            if let Some(max_lufs) = spec.max_loudness_lufs {
                let passed = lufs <= max_lufs;
                checks.push(CheckResult {
                    name: "loudness".to_string(),
                    passed,
                    actual: format!("{lufs:.1}"),
                    expected: format!("<= {max_lufs:.1}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // True peak check
    if let Some(serde_json::Value::Number(tp_n)) = props.get("true_peak_dbtp") {
        if let Some(tp) = tp_n.as_f64() {
            if let Some(max_tp) = spec.max_true_peak_dbtp {
                let passed = tp <= max_tp;
                checks.push(CheckResult {
                    name: "true_peak".to_string(),
                    passed,
                    actual: format!("{tp:.1}"),
                    expected: format!("<= {max_tp:.1}"),
                    severity: if passed { "info" } else { "warning" }.to_string(),
                });
            }
        }
    }

    // Bitrate check
    if let Some(serde_json::Value::Number(br_n)) = props.get("bitrate_kbps") {
        if let Some(br) = br_n.as_u64().map(|v| v as u32) {
            if let Some(max_br) = spec.max_bitrate_kbps {
                let passed = br <= max_br;
                checks.push(CheckResult {
                    name: "bitrate".to_string(),
                    passed,
                    actual: format!("{br}"),
                    expected: format!("<= {max_br}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    checks
}

fn checks_to_json(spec_name: &str, checks: &[CheckResult]) -> String {
    let error_count = checks
        .iter()
        .filter(|c| !c.passed && c.severity == "error")
        .count();
    let warning_count = checks
        .iter()
        .filter(|c| !c.passed && c.severity == "warning")
        .count();
    let overall_pass = error_count == 0;

    let items: Vec<String> = checks
        .iter()
        .map(|c| {
            format!(
                "{{\"check_name\":\"{}\",\"passed\":{},\"actual_value\":\"{}\",\
                 \"expected_value\":\"{}\",\"severity\":\"{}\"}}",
                c.name, c.passed, c.actual, c.expected, c.severity
            )
        })
        .collect();

    format!(
        "{{\"spec_name\":\"{spec_name}\",\"overall_pass\":{overall_pass},\
         \"error_count\":{error_count},\"warning_count\":{warning_count},\
         \"checks\":[{}]}}",
        items.join(",")
    )
}

// ---------------------------------------------------------------------------
// WASM functions
// ---------------------------------------------------------------------------

/// Check media properties (JSON) against a named delivery spec.
///
/// `properties_json`: JSON object with keys like "width", "height", "frame_rate", etc.
/// `spec`: Name of a built-in spec (e.g. "netflix", "youtube", "broadcast_hd").
///
/// Returns a JSON conformance report.
#[wasm_bindgen]
pub fn wasm_check_conform(properties_json: &str, spec: &str) -> Result<String, JsValue> {
    let props: HashMap<String, serde_json::Value> = serde_json::from_str(properties_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid properties JSON: {e}")))?;

    let specs = builtin_specs();
    let found = specs.iter().find(|s| s.name == spec).ok_or_else(|| {
        crate::utils::js_err(&format!(
            "Spec '{}' not found. Available: {}",
            spec,
            specs.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
        ))
    })?;

    let checks = run_checks(found, &props);
    Ok(checks_to_json(spec, &checks))
}

/// List all built-in delivery specs as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_delivery_specs() -> String {
    let specs = builtin_specs();
    let items: Vec<String> = specs.iter().map(|s| spec_to_json(s)).collect();
    format!("[{}]", items.join(","))
}

/// Get a specific delivery spec by name as a JSON object.
#[wasm_bindgen]
pub fn wasm_get_delivery_spec(name: &str) -> Result<String, JsValue> {
    let specs = builtin_specs();
    for s in &specs {
        if s.name == name {
            return Ok(spec_to_json(s));
        }
    }
    Err(crate::utils::js_err(&format!(
        "Spec '{}' not found. Available: {}",
        name,
        specs.iter().map(|s| s.name).collect::<Vec<_>>().join(", ")
    )))
}

/// Validate properties against a custom spec (both passed as JSON).
///
/// `properties_json`: JSON object with media properties.
/// `spec_json`: JSON object describing the spec (same schema as built-in specs).
#[wasm_bindgen]
pub fn wasm_validate_against_spec(
    properties_json: &str,
    spec_json: &str,
) -> Result<String, JsValue> {
    let props: HashMap<String, serde_json::Value> = serde_json::from_str(properties_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid properties JSON: {e}")))?;

    let spec_map: HashMap<String, serde_json::Value> = serde_json::from_str(spec_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid spec JSON: {e}")))?;

    // For custom specs, we manually run checks since DeliverySpec has static lifetimes
    // and we cannot borrow from the local spec_map.
    let mut checks: Vec<CheckResult> = Vec::new();

    // Width/height/loudness/true_peak/bitrate checks from spec_map
    check_range_from_map(
        &spec_map,
        &props,
        "width",
        "min_width",
        "max_width",
        &mut checks,
    );
    check_range_from_map(
        &spec_map,
        &props,
        "height",
        "min_height",
        "max_height",
        &mut checks,
    );

    if let Some(max_lufs) = spec_map.get("max_loudness_lufs").and_then(|v| v.as_f64()) {
        if let Some(serde_json::Value::Number(n)) = props.get("loudness_lufs") {
            if let Some(lufs) = n.as_f64() {
                let passed = lufs <= max_lufs;
                checks.push(CheckResult {
                    name: "loudness".to_string(),
                    passed,
                    actual: format!("{lufs:.1}"),
                    expected: format!("<= {max_lufs:.1}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    if let Some(max_tp) = spec_map.get("max_true_peak_dbtp").and_then(|v| v.as_f64()) {
        if let Some(serde_json::Value::Number(n)) = props.get("true_peak_dbtp") {
            if let Some(tp) = n.as_f64() {
                let passed = tp <= max_tp;
                checks.push(CheckResult {
                    name: "true_peak".to_string(),
                    passed,
                    actual: format!("{tp:.1}"),
                    expected: format!("<= {max_tp:.1}"),
                    severity: if passed { "info" } else { "warning" }.to_string(),
                });
            }
        }
    }

    if let Some(max_br) = spec_map
        .get("max_bitrate_kbps")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
    {
        if let Some(serde_json::Value::Number(n)) = props.get("bitrate_kbps") {
            if let Some(br) = n.as_u64().map(|v| v as u32) {
                let passed = br <= max_br;
                checks.push(CheckResult {
                    name: "bitrate".to_string(),
                    passed,
                    actual: format!("{br}"),
                    expected: format!("<= {max_br}"),
                    severity: if passed { "info" } else { "error" }.to_string(),
                });
            }
        }
    }

    // Handle frame_rates from the custom spec JSON
    if let Some(serde_json::Value::Array(fr_arr)) = spec_map.get("frame_rates") {
        let allowed_frs: Vec<f64> = fr_arr.iter().filter_map(|v| v.as_f64()).collect();
        if !allowed_frs.is_empty() {
            if let Some(serde_json::Value::Number(fps_n)) = props.get("frame_rate") {
                if let Some(fps) = fps_n.as_f64() {
                    let passed = allowed_frs.iter().any(|&a| (a - fps).abs() < 0.01);
                    checks.push(CheckResult {
                        name: "frame_rate".to_string(),
                        passed,
                        actual: format!("{fps:.3}"),
                        expected: format!("{allowed_frs:?}"),
                        severity: if passed { "info" } else { "error" }.to_string(),
                    });
                }
            }
        }
    }

    // Handle sample_rates from the custom spec JSON
    if let Some(serde_json::Value::Array(sr_arr)) = spec_map.get("sample_rates") {
        let allowed_srs: Vec<u32> = sr_arr
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as u32))
            .collect();
        if !allowed_srs.is_empty() {
            if let Some(serde_json::Value::Number(sr_n)) = props.get("sample_rate") {
                if let Some(sr) = sr_n.as_u64().map(|v| v as u32) {
                    let passed = allowed_srs.contains(&sr);
                    checks.push(CheckResult {
                        name: "sample_rate".to_string(),
                        passed,
                        actual: sr.to_string(),
                        expected: format!("{allowed_srs:?}"),
                        severity: if passed { "info" } else { "error" }.to_string(),
                    });
                }
            }
        }
    }

    let spec_name = spec_map
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("custom");
    Ok(checks_to_json(spec_name, &checks))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_delivery_specs() {
        let json = wasm_list_delivery_specs();
        assert!(json.starts_with('['));
        assert!(json.contains("broadcast_hd"));
        assert!(json.contains("netflix"));
        assert!(json.contains("youtube"));
    }

    #[test]
    fn test_get_delivery_spec_found() {
        let result = wasm_get_delivery_spec("netflix");
        assert!(result.is_ok());
        let json = result.expect("should find spec");
        assert!(json.contains("Netflix"));
    }

    #[test]
    fn test_get_delivery_spec_not_found() {
        let result = wasm_get_delivery_spec("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_check_conform_pass() {
        let props = r#"{"width":1920,"height":1080,"frame_rate":25.0,"sample_rate":48000}"#;
        let result = wasm_check_conform(props, "broadcast_hd");
        assert!(result.is_ok());
        let json = result.expect("should check");
        assert!(json.contains("\"overall_pass\":true"));
    }

    #[test]
    fn test_check_conform_fail() {
        let props = r#"{"width":1280,"height":720}"#;
        let result = wasm_check_conform(props, "broadcast_hd");
        assert!(result.is_ok());
        let json = result.expect("should check");
        assert!(json.contains("\"overall_pass\":false"));
    }

    #[test]
    fn test_validate_against_custom_spec() {
        let props = r#"{"width":1920,"height":1080,"frame_rate":30.0}"#;
        let spec = r#"{"name":"custom_test","min_width":1920,"max_width":1920,"min_height":1080,"max_height":1080,"frame_rates":[30.0]}"#;
        let result = wasm_validate_against_spec(props, spec);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"overall_pass\":true"));
    }
}
