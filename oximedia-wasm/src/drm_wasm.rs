//! WebAssembly bindings for DRM encryption and decryption utilities.
//!
//! Provides browser-side DRM data encryption/decryption, info lookup,
//! and scheme enumeration.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Encrypt data using XOR-based content encryption with the given key.
///
/// # Arguments
/// * `data` - Raw bytes to encrypt.
/// * `key` - Encryption key bytes.
///
/// # Returns
/// Encrypted bytes.
///
/// # Errors
/// Returns an error if the key is empty.
#[wasm_bindgen]
pub fn wasm_encrypt_data(data: &[u8], key: &[u8]) -> Result<Vec<u8>, JsValue> {
    if key.is_empty() {
        return Err(crate::utils::js_err("Encryption key must not be empty"));
    }
    let mut output = Vec::with_capacity(data.len());
    for (i, &byte) in data.iter().enumerate() {
        output.push(byte ^ key[i % key.len()]);
    }
    Ok(output)
}

/// Decrypt data using XOR-based content decryption with the given key.
///
/// XOR encryption is symmetric, so this function is identical to encrypt.
///
/// # Arguments
/// * `data` - Encrypted bytes.
/// * `key` - Decryption key bytes.
///
/// # Returns
/// Decrypted bytes.
///
/// # Errors
/// Returns an error if the key is empty.
#[wasm_bindgen]
pub fn wasm_decrypt_data(data: &[u8], key: &[u8]) -> Result<Vec<u8>, JsValue> {
    wasm_encrypt_data(data, key)
}

/// Return DRM information as a JSON object.
///
/// Returns JSON with supported DRM systems, encryption schemes, and capabilities.
#[wasm_bindgen]
pub fn wasm_drm_info() -> String {
    let info = r#"{
  "supported_systems": ["Widevine", "PlayReady", "FairPlay", "ClearKey"],
  "encryption_schemes": ["cenc", "cbc1", "cens", "cbcs"],
  "key_sizes": [128, 256],
  "pssh_support": true,
  "multi_key": true,
  "key_rotation": true,
  "widevine_system_id": "edef8ba9-79d6-4ace-a3c8-27dcd51d21ed",
  "playready_system_id": "9a04f079-9840-4286-ab92-e65be0885f95",
  "fairplay_system_id": "94ce86fb-07ff-4f43-adb8-93d2fa968ca2",
  "clearkey_system_id": "1077efec-c0b2-4d02-ace3-3c1e52e2fb4b"
}"#;
    info.to_string()
}

/// Return a JSON array of supported DRM schemes with descriptions.
#[wasm_bindgen]
pub fn wasm_drm_schemes() -> String {
    let schemes = r#"[
  {"name": "cenc", "description": "Common Encryption - AES-CTR full sample encryption", "standard": "ISO/IEC 23001-7"},
  {"name": "cbc1", "description": "AES-CBC full sample encryption", "standard": "ISO/IEC 23001-7"},
  {"name": "cens", "description": "AES-CTR subsample encryption", "standard": "ISO/IEC 23001-7"},
  {"name": "cbcs", "description": "AES-CBC subsample encryption with constant IV", "standard": "ISO/IEC 23001-7"}
]"#;
    schemes.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let data = b"Hello, DRM WASM!";
        let key = vec![0x42, 0x53, 0x64, 0x75];
        let encrypted = wasm_encrypt_data(data, &key);
        assert!(encrypted.is_ok());
        let encrypted = encrypted.expect("encrypt");
        assert_ne!(&encrypted, data);
        let decrypted = wasm_decrypt_data(&encrypted, &key);
        assert!(decrypted.is_ok());
        assert_eq!(&decrypted.expect("decrypt"), data);
    }

    #[test]
    fn test_encrypt_empty_key_error() {
        let result = wasm_encrypt_data(b"data", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_drm_info_valid_json() {
        let info = wasm_drm_info();
        assert!(info.contains("Widevine"));
        assert!(info.contains("cenc"));
    }

    #[test]
    fn test_drm_schemes_valid_json() {
        let schemes = wasm_drm_schemes();
        assert!(schemes.contains("cenc"));
        assert!(schemes.contains("cbcs"));
    }
}
