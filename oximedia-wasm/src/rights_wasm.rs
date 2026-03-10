//! WebAssembly bindings for digital rights checking and license information.
//!
//! Provides WASM-accessible functions for checking rights status,
//! listing license types, and generating rights reports.

use wasm_bindgen::prelude::*;

/// Check rights status for an asset.
///
/// `request_json`: JSON object with keys:
///   - `asset` (string): asset identifier
///   - `intended_use` (string, optional): broadcast, streaming, theatrical, download, physical
///   - `territory` (string, optional): ISO 3166 code
///
/// Returns a JSON rights check result.
#[wasm_bindgen]
pub fn wasm_check_rights(request_json: &str) -> Result<String, JsValue> {
    let request: serde_json::Value = serde_json::from_str(request_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid request JSON: {e}")))?;

    let asset = request
        .get("asset")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let intended_use = request.get("intended_use").and_then(|v| v.as_str());
    let territory = request.get("territory").and_then(|v| v.as_str());

    // Validate intended use if provided
    if let Some(use_type) = intended_use {
        let valid = [
            "broadcast",
            "streaming",
            "theatrical",
            "download",
            "physical",
        ];
        if !valid.contains(&use_type) {
            return Err(crate::utils::js_err(&format!(
                "Invalid intended_use '{}'. Expected: {}",
                use_type,
                valid.join(", ")
            )));
        }
    }

    let result = serde_json::json!({
        "asset": asset,
        "intended_use": intended_use.unwrap_or("unspecified"),
        "territory": territory.unwrap_or("worldwide"),
        "cleared": false,
        "status": "unknown",
        "rights_count": 0,
        "active_licenses": 0,
        "message": "Client-side check only. Full rights verification requires server-side database.",
        "rights_types_available": [
            "master", "sync", "mechanical", "performance", "reproduction", "distribution"
        ],
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
}

/// List available license types with descriptions.
///
/// Returns a JSON array of license type descriptors.
#[wasm_bindgen]
pub fn wasm_license_types() -> String {
    let types = serde_json::json!([
        {
            "type": "royalty-free",
            "description": "One-time fee, unlimited usage within terms",
            "typical_duration": "perpetual",
            "requires_attribution": false,
            "transferable": false,
        },
        {
            "type": "rights-managed",
            "description": "Per-use licensing with specific terms",
            "typical_duration": "1-5 years",
            "requires_attribution": true,
            "transferable": false,
        },
        {
            "type": "editorial",
            "description": "Restricted to editorial/news use only",
            "typical_duration": "perpetual",
            "requires_attribution": true,
            "transferable": false,
        },
        {
            "type": "creative-commons",
            "description": "Open license with varying restriction levels",
            "typical_duration": "perpetual",
            "requires_attribution": true,
            "transferable": true,
            "variants": [
                "CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA", "CC-BY-ND", "CC-BY-NC-ND"
            ],
        },
        {
            "type": "public-domain",
            "description": "No copyright restrictions",
            "typical_duration": "perpetual",
            "requires_attribution": false,
            "transferable": true,
        },
    ]);
    serde_json::to_string(&types).unwrap_or_else(|_| "[]".to_string())
}

/// Generate a rights report summary.
///
/// `params_json`: JSON object with optional keys:
///   - `report_type` (string): summary, expiring, royalties, territory, compliance
///   - `holder` (string, optional): filter by rights holder
///   - `expiry_days` (u32, optional): days threshold for expiring report
///
/// Returns a JSON report object.
#[wasm_bindgen]
pub fn wasm_rights_report(params_json: &str) -> Result<String, JsValue> {
    let params: serde_json::Value = serde_json::from_str(params_json)
        .map_err(|e| crate::utils::js_err(&format!("Invalid params JSON: {e}")))?;

    let report_type = params
        .get("report_type")
        .and_then(|v| v.as_str())
        .unwrap_or("summary");
    let holder = params.get("holder").and_then(|v| v.as_str());
    let expiry_days = params
        .get("expiry_days")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);

    let valid_types = [
        "summary",
        "expiring",
        "royalties",
        "territory",
        "compliance",
    ];
    if !valid_types.contains(&report_type) {
        return Err(crate::utils::js_err(&format!(
            "Invalid report_type '{}'. Expected: {}",
            report_type,
            valid_types.join(", ")
        )));
    }

    let result = serde_json::json!({
        "report_type": report_type,
        "holder_filter": holder,
        "expiry_days": expiry_days,
        "total_rights": 0,
        "active_licenses": 0,
        "expiring_soon": 0,
        "total_royalty_obligations": 0.0,
        "territory_coverage": [],
        "compliance_issues": [],
        "message": "Client-side report scaffold. Populate with server data.",
    });

    serde_json::to_string(&result)
        .map_err(|e| crate::utils::js_err(&format!("Serialization error: {e}")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_rights_basic() {
        let request = r#"{"asset":"video-001"}"#;
        let result = wasm_check_rights(request);
        assert!(result.is_ok());
        let json = result.expect("should check");
        assert!(json.contains("\"cleared\":false"));
        assert!(json.contains("video-001"));
    }

    #[test]
    fn test_check_rights_with_use() {
        let request = r#"{"asset":"v1","intended_use":"streaming","territory":"US"}"#;
        let result = wasm_check_rights(request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_rights_invalid_use() {
        let request = r#"{"asset":"v1","intended_use":"illegal"}"#;
        let result = wasm_check_rights(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_license_types() {
        let types = wasm_license_types();
        assert!(types.contains("royalty-free"));
        assert!(types.contains("rights-managed"));
        assert!(types.contains("creative-commons"));
        assert!(types.contains("public-domain"));
    }

    #[test]
    fn test_rights_report() {
        let params = r#"{"report_type":"summary"}"#;
        let result = wasm_rights_report(params);
        assert!(result.is_ok());
        let json = result.expect("should generate");
        assert!(json.contains("\"report_type\":\"summary\""));
    }

    #[test]
    fn test_rights_report_invalid_type() {
        let params = r#"{"report_type":"invalid"}"#;
        let result = wasm_rights_report(params);
        assert!(result.is_err());
    }
}
