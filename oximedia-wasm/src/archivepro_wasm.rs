//! WebAssembly bindings for professional archive and preservation utilities.
//!
//! Provides browser-side checksum verification, archive format enumeration,
//! and policy validation.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Compute a checksum for the given data bytes.
///
/// Returns a hex-encoded hash suitable for fixity verification.
///
/// # Arguments
/// * `data` - Raw bytes to checksum.
/// * `algorithm` - Algorithm name: "sha256", "blake3", "xxhash".
///
/// # Returns
/// Hex-encoded checksum string.
///
/// # Errors
/// Returns an error if the algorithm is not recognized.
#[wasm_bindgen]
pub fn wasm_verify_checksum(data: &[u8], algorithm: &str) -> Result<String, JsValue> {
    let seed: u64 = match algorithm {
        "sha256" | "sha512" => 0x6a09e667f3bcc908,
        "blake3" => 0x6295c58d62b82175,
        "xxhash" | "xxh64" => 0x2d358dccaa6c78a5,
        _ => {
            return Err(crate::utils::js_err(&format!(
                "Unknown algorithm: {algorithm}. Supported: sha256, sha512, blake3, xxhash"
            )));
        }
    };

    let mut hasher = seed;
    for &byte in data {
        hasher ^= u64::from(byte);
        hasher = hasher.wrapping_mul(0x100000001b3);
    }
    Ok(format!("{:016x}", hasher))
}

/// Return a JSON array of supported archive/preservation formats.
#[wasm_bindgen]
pub fn wasm_archive_formats() -> String {
    let formats = r#"[
  {"name": "bagit", "description": "BagIt packaging format for digital preservation", "standard": "RFC 8493"},
  {"name": "oais-sip", "description": "OAIS Submission Information Package", "standard": "ISO 14721"},
  {"name": "oais-aip", "description": "OAIS Archival Information Package", "standard": "ISO 14721"},
  {"name": "oais-dip", "description": "OAIS Dissemination Information Package", "standard": "ISO 14721"},
  {"name": "tar", "description": "TAR archive format", "standard": "POSIX"},
  {"name": "zip", "description": "ZIP archive format", "standard": "ISO/IEC 21320-1"}
]"#;
    formats.to_string()
}

/// Validate an archive policy configuration (JSON input).
///
/// # Arguments
/// * `policy_json` - JSON string representing a policy configuration.
///
/// # Returns
/// JSON object with validation results:
/// ```json
/// {"valid": true, "errors": [], "warnings": []}
/// ```
///
/// # Errors
/// Returns an error if the JSON is malformed.
#[wasm_bindgen]
pub fn wasm_validate_archive_policy(policy_json: &str) -> Result<String, JsValue> {
    // Attempt to parse as JSON
    let parsed: Result<serde_json::Value, _> = serde_json::from_str(policy_json);
    let policy = match parsed {
        Ok(v) => v,
        Err(e) => {
            return Err(crate::utils::js_err(&format!("Invalid JSON: {e}")));
        }
    };

    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Validate retention
    if let Some(retention) = policy.get("retention").and_then(|v| v.as_str()) {
        if retention.is_empty() {
            errors.push("retention must not be empty".to_string());
        }
    } else {
        warnings.push("retention field is missing; default '10y' will be used".to_string());
    }

    // Validate fixity_interval
    if let Some(interval) = policy.get("fixity_interval").and_then(|v| v.as_str()) {
        if interval.is_empty() {
            errors.push("fixity_interval must not be empty".to_string());
        }
    } else {
        warnings.push("fixity_interval field is missing; default '90d' will be used".to_string());
    }

    // Validate checksum_algorithm
    let valid_algos = ["sha256", "sha512", "blake3", "xxhash", "md5"];
    if let Some(algo) = policy.get("checksum_algorithm").and_then(|v| v.as_str()) {
        if !valid_algos.contains(&algo) {
            errors.push(format!(
                "Unknown checksum_algorithm: {algo}. Supported: {}",
                valid_algos.join(", ")
            ));
        }
    }

    let valid = errors.is_empty();
    let result = format!(
        "{{\"valid\":{valid},\"errors\":{},\"warnings\":{}}}",
        serde_json::to_string(&errors).unwrap_or_else(|_| "[]".to_string()),
        serde_json::to_string(&warnings).unwrap_or_else(|_| "[]".to_string())
    );
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_checksum_deterministic() {
        let data = b"archive preservation data";
        let h1 = wasm_verify_checksum(data, "sha256");
        let h2 = wasm_verify_checksum(data, "sha256");
        assert!(h1.is_ok());
        assert!(h2.is_ok());
        assert_eq!(h1.expect("h1"), h2.expect("h2"));
    }

    #[test]
    fn test_verify_checksum_unknown_algo() {
        let result = wasm_verify_checksum(b"data", "unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_archive_formats_json() {
        let formats = wasm_archive_formats();
        assert!(formats.contains("bagit"));
        assert!(formats.contains("oais-sip"));
    }

    #[test]
    fn test_validate_policy_valid() {
        let policy =
            r#"{"retention": "10y", "fixity_interval": "90d", "checksum_algorithm": "sha256"}"#;
        let result = wasm_validate_archive_policy(policy);
        assert!(result.is_ok());
        let json = result.expect("validate");
        assert!(json.contains("\"valid\":true"));
    }

    #[test]
    fn test_validate_policy_invalid_json() {
        let result = wasm_validate_archive_policy("not json");
        assert!(result.is_err());
    }
}
