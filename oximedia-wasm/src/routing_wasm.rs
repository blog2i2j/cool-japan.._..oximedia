//! WebAssembly bindings for audio/video routing.
//!
//! Provides WASM-accessible functions for validating routes,
//! querying route info, and listing node types. All data is exchanged as JSON.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct NodeTypeInfo {
    name: &'static str,
    description: &'static str,
    max_channels: u32,
}

fn builtin_node_types() -> Vec<NodeTypeInfo> {
    vec![
        NodeTypeInfo {
            name: "input",
            description: "Audio/video input source",
            max_channels: 64,
        },
        NodeTypeInfo {
            name: "output",
            description: "Audio/video output destination",
            max_channels: 64,
        },
        NodeTypeInfo {
            name: "mixer",
            description: "Multi-input summing mixer",
            max_channels: 128,
        },
        NodeTypeInfo {
            name: "splitter",
            description: "Single input to multiple outputs",
            max_channels: 64,
        },
        NodeTypeInfo {
            name: "processor",
            description: "Signal processor (EQ, dynamics, etc.)",
            max_channels: 32,
        },
        NodeTypeInfo {
            name: "monitor",
            description: "Monitoring output (AFL/PFL/Solo)",
            max_channels: 8,
        },
        NodeTypeInfo {
            name: "bus",
            description: "Mix bus for grouping signals",
            max_channels: 64,
        },
    ]
}

// ---------------------------------------------------------------------------
// WASM functions
// ---------------------------------------------------------------------------

/// Validate a routing configuration.
///
/// `config_json`: JSON object describing a route (source, destination, gain, etc.)
///
/// Returns a JSON validation report.
#[wasm_bindgen]
pub fn wasm_validate_route(config_json: &str) -> Result<String, JsValue> {
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid config JSON: {e}")))?;

    let has_source = config.get("source").is_some();
    let has_destination = config.get("destination").is_some();
    let is_valid = has_source && has_destination;

    let mut warnings: Vec<String> = Vec::new();
    if !has_source {
        warnings.push("missing 'source' field".to_string());
    }
    if !has_destination {
        warnings.push("missing 'destination' field".to_string());
    }

    // Check gain range
    if let Some(gain) = config.get("gain_db").and_then(|v| v.as_f64()) {
        if gain > 20.0 {
            warnings.push(format!("gain {gain:.1} dB exceeds +20 dB headroom"));
        }
        if gain < -80.0 {
            warnings.push(format!("gain {gain:.1} dB is extremely low"));
        }
    }

    let warnings_json: Vec<String> = warnings
        .iter()
        .map(|w| format!("\"{}\"", w.replace('"', "\\\"")))
        .collect();

    Ok(format!(
        "{{\"is_valid\":{is_valid},\"has_source\":{has_source},\
         \"has_destination\":{has_destination},\
         \"warnings\":[{}],\"warning_count\":{}}}",
        warnings_json.join(","),
        warnings.len()
    ))
}

/// Get information about a routing configuration.
///
/// `config_json`: JSON object with matrix configuration.
///
/// Returns a JSON info report.
#[wasm_bindgen]
pub fn wasm_route_info(config_json: &str) -> Result<String, JsValue> {
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid config JSON: {e}")))?;

    let name = config
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed");
    let matrix_type = config
        .get("matrix_type")
        .and_then(|v| v.as_str())
        .unwrap_or("audio");
    let inputs = config.get("inputs").and_then(|v| v.as_u64()).unwrap_or(0);
    let outputs = config.get("outputs").and_then(|v| v.as_u64()).unwrap_or(0);

    let connections = config
        .get("connections")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    let crosspoints = inputs * outputs;

    Ok(format!(
        "{{\"name\":\"{name}\",\"matrix_type\":\"{matrix_type}\",\
         \"inputs\":{inputs},\"outputs\":{outputs},\
         \"crosspoints\":{crosspoints},\"active_connections\":{connections}}}"
    ))
}

/// List all supported routing node types as a JSON array.
#[wasm_bindgen]
pub fn wasm_list_node_types() -> String {
    let types = builtin_node_types();
    let items: Vec<String> = types
        .iter()
        .map(|t| {
            format!(
                "{{\"name\":\"{}\",\"description\":\"{}\",\"max_channels\":{}}}",
                t.name, t.description, t.max_channels
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
    fn test_list_node_types() {
        let json = wasm_list_node_types();
        assert!(json.starts_with('['));
        assert!(json.contains("input"));
        assert!(json.contains("mixer"));
        assert!(json.contains("monitor"));
    }

    #[test]
    fn test_validate_route_valid() {
        let config = r#"{"source":"mic1","destination":"monitor","gain_db":-6.0}"#;
        let result = wasm_validate_route(config);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"is_valid\":true"));
    }

    #[test]
    fn test_validate_route_missing_source() {
        let config = r#"{"destination":"monitor"}"#;
        let result = wasm_validate_route(config);
        assert!(result.is_ok());
        let json = result.expect("should validate");
        assert!(json.contains("\"is_valid\":false"));
        assert!(json.contains("missing 'source' field"));
    }

    #[test]
    fn test_route_info() {
        let config = r#"{"name":"main","matrix_type":"audio","inputs":16,"outputs":8,"connections":[{},{}]}"#;
        let result = wasm_route_info(config);
        assert!(result.is_ok());
        let json = result.expect("should get info");
        assert!(json.contains("\"name\":\"main\""));
        assert!(json.contains("\"inputs\":16"));
        assert!(json.contains("\"active_connections\":2"));
    }

    #[test]
    fn test_validate_route_invalid_json() {
        let result = wasm_validate_route("not json");
        assert!(result.is_err());
    }
}
