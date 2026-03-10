//! Python bindings for DRM (Digital Rights Management).
//!
//! Provides `PyDrmManager`, `PyDrmKey`, `PyEncryptionConfig`,
//! and standalone functions for encrypting and decrypting media.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> PyResult<Vec<u8>> {
    let hex = hex.trim_start_matches("0x");
    if hex.len() % 2 != 0 {
        return Err(PyValueError::new_err("Hex string must have even length"));
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|e| PyValueError::new_err(format!("Invalid hex at position {i}: {e}")))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

fn generate_key(bits: u32) -> Vec<u8> {
    let len = (bits / 8) as usize;
    let mut key = Vec::with_capacity(len);
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let mut state = seed as u64;
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        key.push((state >> 33) as u8);
    }
    key
}

// ---------------------------------------------------------------------------
// PyDrmKey
// ---------------------------------------------------------------------------

/// A DRM content encryption key.
#[pyclass]
#[derive(Clone)]
pub struct PyDrmKey {
    /// Key ID (hex string).
    #[pyo3(get)]
    pub key_id: String,
    /// Key value (hex string).
    #[pyo3(get)]
    pub key: String,
    /// Key length in bits.
    #[pyo3(get)]
    pub bits: u32,
}

#[pymethods]
impl PyDrmKey {
    /// Create a new DRM key from hex strings.
    #[new]
    fn new(key_id: &str, key: &str) -> PyResult<Self> {
        let _ = hex_decode(key_id)?;
        let key_bytes = hex_decode(key)?;
        Ok(Self {
            key_id: key_id.to_string(),
            key: key.to_string(),
            bits: (key_bytes.len() * 8) as u32,
        })
    }

    /// Generate a random DRM key pair.
    #[staticmethod]
    #[pyo3(signature = (bits=128))]
    fn generate(bits: u32) -> PyResult<Self> {
        if bits != 128 && bits != 256 {
            return Err(PyValueError::new_err("Key bits must be 128 or 256"));
        }
        let key = generate_key(bits);
        let key_id = generate_key(128);
        Ok(Self {
            key_id: hex_encode(&key_id),
            key: hex_encode(&key),
            bits,
        })
    }

    fn __repr__(&self) -> String {
        format!("PyDrmKey(key_id='{}', bits={})", self.key_id, self.bits)
    }

    fn to_dict(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("key_id".to_string(), self.key_id.clone());
        m.insert("key".to_string(), self.key.clone());
        m.insert("bits".to_string(), self.bits.to_string());
        m
    }
}

// ---------------------------------------------------------------------------
// PyEncryptionConfig
// ---------------------------------------------------------------------------

/// Encryption configuration for DRM operations.
#[pyclass]
#[derive(Clone)]
pub struct PyEncryptionConfig {
    /// DRM system name.
    #[pyo3(get)]
    pub system: String,
    /// Encryption scheme: cenc, cbc1, cens, cbcs.
    #[pyo3(get)]
    pub scheme: String,
    /// License server URL (optional).
    #[pyo3(get)]
    pub license_url: Option<String>,
}

#[pymethods]
impl PyEncryptionConfig {
    #[new]
    #[pyo3(signature = (system="clearkey", scheme="cenc", license_url=None))]
    fn new(system: &str, scheme: &str, license_url: Option<String>) -> PyResult<Self> {
        let valid_systems = ["widevine", "playready", "fairplay", "clearkey"];
        if !valid_systems.contains(&system.to_lowercase().as_str()) {
            return Err(PyValueError::new_err(format!(
                "Unknown DRM system: {system}. Supported: {}",
                valid_systems.join(", ")
            )));
        }
        Ok(Self {
            system: system.to_string(),
            scheme: scheme.to_string(),
            license_url,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PyEncryptionConfig(system='{}', scheme='{}')",
            self.system, self.scheme
        )
    }
}

// ---------------------------------------------------------------------------
// PyDrmManager
// ---------------------------------------------------------------------------

/// DRM manager for encrypting and decrypting media content.
#[pyclass]
pub struct PyDrmManager {
    keys: HashMap<String, PyDrmKey>,
    config: PyEncryptionConfig,
}

#[pymethods]
impl PyDrmManager {
    /// Create a new DRM manager with the given configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<PyEncryptionConfig>) -> Self {
        Self {
            keys: HashMap::new(),
            config: config.unwrap_or_else(|| PyEncryptionConfig {
                system: "clearkey".to_string(),
                scheme: "cenc".to_string(),
                license_url: None,
            }),
        }
    }

    /// Add a key to the manager.
    fn add_key(&mut self, key: PyDrmKey) {
        self.keys.insert(key.key_id.clone(), key);
    }

    /// Remove a key by key ID.
    fn remove_key(&mut self, key_id: &str) -> bool {
        self.keys.remove(key_id).is_some()
    }

    /// Get a key by key ID.
    fn get_key(&self, key_id: &str) -> Option<PyDrmKey> {
        self.keys.get(key_id).cloned()
    }

    /// List all key IDs.
    fn list_key_ids(&self) -> Vec<String> {
        self.keys.keys().cloned().collect()
    }

    /// Get the number of keys.
    fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Encrypt data using the specified key.
    fn encrypt(&self, data: &[u8], key_id: &str) -> PyResult<Vec<u8>> {
        let key = self
            .keys
            .get(key_id)
            .ok_or_else(|| PyValueError::new_err(format!("Key not found: {key_id}")))?;
        let key_bytes = hex_decode(&key.key)?;
        if key_bytes.is_empty() {
            return Err(PyRuntimeError::new_err("Key is empty"));
        }
        let mut output = Vec::with_capacity(data.len());
        for (i, &byte) in data.iter().enumerate() {
            output.push(byte ^ key_bytes[i % key_bytes.len()]);
        }
        Ok(output)
    }

    /// Decrypt data using the specified key.
    fn decrypt(&self, data: &[u8], key_id: &str) -> PyResult<Vec<u8>> {
        // XOR encryption is symmetric
        self.encrypt(data, key_id)
    }

    /// Get the current configuration.
    fn get_config(&self) -> PyEncryptionConfig {
        self.config.clone()
    }

    /// Get supported DRM systems.
    #[staticmethod]
    fn supported_systems() -> Vec<String> {
        vec![
            "Widevine".to_string(),
            "PlayReady".to_string(),
            "FairPlay".to_string(),
            "ClearKey".to_string(),
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "PyDrmManager(system='{}', keys={})",
            self.config.system,
            self.keys.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Standalone functions
// ---------------------------------------------------------------------------

/// Encrypt media data with a given key.
#[pyfunction]
pub fn encrypt_media(data: &[u8], key_hex: &str) -> PyResult<Vec<u8>> {
    let key = hex_decode(key_hex)?;
    if key.is_empty() {
        return Err(PyValueError::new_err("Key must not be empty"));
    }
    let mut output = Vec::with_capacity(data.len());
    for (i, &byte) in data.iter().enumerate() {
        output.push(byte ^ key[i % key.len()]);
    }
    Ok(output)
}

/// Decrypt media data with a given key.
#[pyfunction]
pub fn decrypt_media(data: &[u8], key_hex: &str) -> PyResult<Vec<u8>> {
    encrypt_media(data, key_hex)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Register DRM bindings on a PyModule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDrmKey>()?;
    m.add_class::<PyEncryptionConfig>()?;
    m.add_class::<PyDrmManager>()?;
    m.add_function(wrap_pyfunction!(encrypt_media, m)?)?;
    m.add_function(wrap_pyfunction!(decrypt_media, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_roundtrip() {
        let original = vec![0xab, 0xcd, 0xef, 0x01];
        let encoded = hex_encode(&original);
        assert_eq!(encoded, "abcdef01");
        let decoded = hex_decode(&encoded);
        assert!(decoded.is_ok());
        assert_eq!(decoded.expect("decode"), original);
    }

    #[test]
    fn test_hex_decode_invalid() {
        assert!(hex_decode("xyz").is_err());
        assert!(hex_decode("abc").is_err());
    }

    #[test]
    fn test_generate_key_length() {
        let k128 = generate_key(128);
        assert_eq!(k128.len(), 16);
        let k256 = generate_key(256);
        assert_eq!(k256.len(), 32);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let data = b"test encryption data";
        let key = hex_encode(&[0x42, 0x53, 0x64, 0x75]);
        let encrypted = encrypt_media(data, &key);
        assert!(encrypted.is_ok());
        let encrypted = encrypted.expect("encrypt");
        let decrypted = decrypt_media(&encrypted, &key);
        assert!(decrypted.is_ok());
        assert_eq!(&decrypted.expect("decrypt"), data);
    }

    #[test]
    fn test_drm_manager_key_ops() {
        let mut mgr = PyDrmManager::new(None);
        assert_eq!(mgr.key_count(), 0);
        let key = PyDrmKey {
            key_id: "aabb".to_string(),
            key: "ccdd".to_string(),
            bits: 16,
        };
        mgr.add_key(key);
        assert_eq!(mgr.key_count(), 1);
        assert!(mgr.get_key("aabb").is_some());
        assert!(mgr.remove_key("aabb"));
        assert_eq!(mgr.key_count(), 0);
    }
}
