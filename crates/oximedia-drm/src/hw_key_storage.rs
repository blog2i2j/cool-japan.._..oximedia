//! Hardware key storage abstraction layer.
//!
//! Provides a simplified, unified interface for storing content-encryption keys
//! in either software (in-process HashMap) or hardware-backed stores (TPM 2.0,
//! Apple Secure Enclave, Android Keystore).  All hardware backends are stubs in
//! this implementation; they log an informational message and fall back to the
//! software store when `fallback_to_software` is enabled.
//!
//! # Design
//!
//! The XOR-based obfuscation used by the `Software` backend is intentionally
//! simple — it is NOT cryptographic protection.  Production deployments would
//! replace the backend with real OS/HSM key-wrapping APIs.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Backend enum
// ---------------------------------------------------------------------------

/// Identifies the hardware security module to use for key storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwStorageBackend {
    /// Pure-software in-memory store (XOR-obfuscated with key ID as mask).
    Software,
    /// TPM 2.0 (Trusted Platform Module) — stub implementation.
    Tpm,
    /// Apple Secure Enclave — stub implementation.
    SecureEnclave,
    /// Android Keystore — stub implementation.
    AndroidKeystore,
}

impl HwStorageBackend {
    /// Returns `true` for backends that would be hardware-accelerated in
    /// a production deployment.
    #[must_use]
    pub fn is_hardware(&self) -> bool {
        matches!(
            self,
            HwStorageBackend::Tpm
                | HwStorageBackend::SecureEnclave
                | HwStorageBackend::AndroidKeystore
        )
    }

    /// Human-readable name of the backend.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            HwStorageBackend::Software => "Software",
            HwStorageBackend::Tpm => "TPM",
            HwStorageBackend::SecureEnclave => "SecureEnclave",
            HwStorageBackend::AndroidKeystore => "AndroidKeystore",
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`HwKeyStorage`].
#[derive(Debug, Clone)]
pub struct HwKeyStorageConfig {
    /// The preferred hardware (or software) backend.
    pub backend: HwStorageBackend,
    /// When `true`, hardware backends that are unavailable (all stubs) will
    /// silently fall back to the software store.
    pub fallback_to_software: bool,
}

impl HwKeyStorageConfig {
    /// Create a software-only configuration.
    #[must_use]
    pub fn software() -> Self {
        Self {
            backend: HwStorageBackend::Software,
            fallback_to_software: false,
        }
    }

    /// Create a configuration for the given backend with an explicit fallback setting.
    #[must_use]
    pub fn with_backend(backend: HwStorageBackend, fallback_to_software: bool) -> Self {
        Self {
            backend,
            fallback_to_software,
        }
    }
}

// ---------------------------------------------------------------------------
// XOR obfuscation helpers (Software backend)
// ---------------------------------------------------------------------------

/// XOR-obfuscate `data` using the bytes of `key_id` as a repeating mask.
///
/// This is a symmetric operation — calling it twice recovers the original.
fn xor_with_id(data: &[u8], key_id: &str) -> Vec<u8> {
    let mask = key_id.as_bytes();
    if mask.is_empty() {
        return data.to_vec();
    }
    data.iter()
        .enumerate()
        .map(|(i, &b)| b ^ mask[i % mask.len()])
        .collect()
}

// ---------------------------------------------------------------------------
// HwKeyStorage
// ---------------------------------------------------------------------------

/// A unified key-storage interface that dispatches to the configured backend.
///
/// For the `Software` backend keys are XOR-masked with the key ID before
/// being stored in a `HashMap`.  For all other (hardware) backends the current
/// implementation logs a notice, and — if `fallback_to_software` is `true` —
/// falls through to the software store; otherwise the operation returns an error.
pub struct HwKeyStorage {
    config: HwKeyStorageConfig,
    /// Software-fallback in-memory store: key_id → obfuscated bytes.
    store: HashMap<String, Vec<u8>>,
    /// The backend that is actually active (may differ from `config.backend`
    /// when a hardware backend fell back to software).
    active_backend: HwStorageBackend,
}

impl HwKeyStorage {
    /// Create a new `HwKeyStorage` with the given configuration.
    ///
    /// If the requested backend is a hardware stub and `fallback_to_software`
    /// is `true`, the `active_backend` will be `Software`.
    #[must_use]
    pub fn new(config: HwKeyStorageConfig) -> Self {
        let active_backend = if config.backend.is_hardware() && config.fallback_to_software {
            // In a real implementation we would probe for hardware availability.
            // Here all hardware backends are stubs, so we always fall back.
            HwStorageBackend::Software
        } else {
            config.backend
        };

        Self {
            config,
            store: HashMap::new(),
            active_backend,
        }
    }

    /// Store `key_data` under `key_id`.
    ///
    /// # Errors
    /// Returns `Err(String)` when the active backend is a hardware stub and
    /// `fallback_to_software` is `false`.
    pub fn store_key(&mut self, key_id: &str, key_data: &[u8]) -> Result<(), String> {
        match self.active_backend {
            HwStorageBackend::Software => {
                let obfuscated = xor_with_id(key_data, key_id);
                self.store.insert(key_id.to_string(), obfuscated);
                Ok(())
            }
            other => {
                // Hardware stubs — log and optionally fall back.
                let msg = format!(
                    "HwKeyStorage: backend '{}' not available (stub implementation)",
                    other.name()
                );
                if self.config.fallback_to_software {
                    // Silently store in software map.
                    let obfuscated = xor_with_id(key_data, key_id);
                    self.store.insert(key_id.to_string(), obfuscated);
                    Ok(())
                } else {
                    Err(msg)
                }
            }
        }
    }

    /// Retrieve the key stored under `key_id`, or `None` if not present.
    pub fn retrieve_key(&self, key_id: &str) -> Option<Vec<u8>> {
        let obfuscated = self.store.get(key_id)?;
        Some(xor_with_id(obfuscated, key_id))
    }

    /// Delete the key stored under `key_id`.
    ///
    /// Returns `true` if the key existed and was removed, `false` otherwise.
    pub fn delete_key(&mut self, key_id: &str) -> bool {
        self.store.remove(key_id).is_some()
    }

    /// Returns `true` only when the *active* backend is a non-Software variant.
    ///
    /// Because all hardware backends are stubs that fall back to software when
    /// `fallback_to_software` is `true`, this method always returns `false` in
    /// the current implementation.
    #[must_use]
    pub fn is_hw_backed(&self) -> bool {
        // True hardware backing is not available in the current stub implementation.
        false
    }

    /// Return the active backend variant.
    #[must_use]
    pub fn active_backend(&self) -> HwStorageBackend {
        self.active_backend
    }

    /// Return the number of keys currently in the store.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.store.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn software_store() -> HwKeyStorage {
        HwKeyStorage::new(HwKeyStorageConfig::software())
    }

    // --- Basic Software backend operations ---

    #[test]
    fn test_store_and_retrieve_roundtrip() {
        let mut storage = software_store();
        let key_data = vec![0xAB_u8; 16];
        storage.store_key("key-1", &key_data).expect("store should succeed");
        let retrieved = storage.retrieve_key("key-1").expect("retrieve should succeed");
        assert_eq!(retrieved, key_data, "retrieved key must equal original");
    }

    #[test]
    fn test_retrieve_missing_returns_none() {
        let storage = software_store();
        assert!(storage.retrieve_key("nonexistent").is_none());
    }

    #[test]
    fn test_delete_key_returns_true_when_found() {
        let mut storage = software_store();
        let key_data = vec![0x01_u8; 8];
        storage.store_key("del-key", &key_data).expect("store should succeed");
        assert!(storage.delete_key("del-key"), "delete should return true");
        assert!(storage.retrieve_key("del-key").is_none(), "key must be gone after delete");
    }

    #[test]
    fn test_delete_key_returns_false_when_not_found() {
        let mut storage = software_store();
        assert!(!storage.delete_key("ghost"), "delete of missing key must return false");
    }

    #[test]
    fn test_software_backend_is_not_hw_backed() {
        let storage = software_store();
        assert!(!storage.is_hw_backed(), "software backend must not be hw-backed");
    }

    #[test]
    fn test_xor_obfuscation_differs_from_plaintext() {
        let mut storage = software_store();
        let key_data = vec![0xFF_u8; 16];
        storage.store_key("obf-key", &key_data).expect("store should succeed");
        // Peek at the raw stored bytes (obfuscated) — they must differ from plaintext.
        let raw = storage.store.get("obf-key").cloned().expect("raw entry");
        assert_ne!(raw, key_data, "stored bytes must be obfuscated");
    }

    #[test]
    fn test_key_count() {
        let mut storage = software_store();
        assert_eq!(storage.key_count(), 0);
        storage.store_key("k1", &[1; 16]).expect("store k1");
        storage.store_key("k2", &[2; 16]).expect("store k2");
        assert_eq!(storage.key_count(), 2);
        storage.delete_key("k1");
        assert_eq!(storage.key_count(), 1);
    }

    // --- Hardware backends with fallback ---

    #[test]
    fn test_tpm_with_fallback_stores_in_software() {
        let config = HwKeyStorageConfig::with_backend(HwStorageBackend::Tpm, true);
        let mut storage = HwKeyStorage::new(config);
        let key_data = vec![0x42_u8; 16];
        storage.store_key("tpm-key", &key_data).expect("should fallback to software");
        let retrieved = storage.retrieve_key("tpm-key").expect("retrieve after fallback");
        assert_eq!(retrieved, key_data);
    }

    #[test]
    fn test_hardware_backend_no_fallback_returns_error() {
        // Hardware backend is a stub; without fallback it must error.
        let config = HwKeyStorageConfig::with_backend(HwStorageBackend::SecureEnclave, false);
        let mut storage = HwKeyStorage::new(config);
        let result = storage.store_key("se-key", &[0xDE; 16]);
        assert!(result.is_err(), "hw stub without fallback must return Err");
    }

    #[test]
    fn test_active_backend_reflects_fallback() {
        let config = HwKeyStorageConfig::with_backend(HwStorageBackend::AndroidKeystore, true);
        let storage = HwKeyStorage::new(config);
        assert_eq!(
            storage.active_backend(),
            HwStorageBackend::Software,
            "active backend should be Software when hw stub falls back"
        );
    }

    #[test]
    fn test_is_hw_backed_always_false_for_stubs() {
        let config = HwKeyStorageConfig::with_backend(HwStorageBackend::Tpm, false);
        let storage = HwKeyStorage::new(config);
        // Current implementation: hardware stubs always return false
        assert!(!storage.is_hw_backed());
    }

    #[test]
    fn test_multiple_keys_independent() {
        let mut storage = software_store();
        let key_a = vec![0x11_u8; 16];
        let key_b = vec![0x22_u8; 16];
        storage.store_key("key-a", &key_a).expect("store a");
        storage.store_key("key-b", &key_b).expect("store b");
        assert_eq!(storage.retrieve_key("key-a").expect("key-a"), key_a);
        assert_eq!(storage.retrieve_key("key-b").expect("key-b"), key_b);
    }

    #[test]
    fn test_overwrite_key() {
        let mut storage = software_store();
        storage.store_key("ow-key", &[0xAA; 8]).expect("first store");
        storage.store_key("ow-key", &[0xBB; 8]).expect("second store (overwrite)");
        let retrieved = storage.retrieve_key("ow-key").expect("retrieve");
        assert_eq!(retrieved, vec![0xBB_u8; 8], "overwritten value must be returned");
    }
}
