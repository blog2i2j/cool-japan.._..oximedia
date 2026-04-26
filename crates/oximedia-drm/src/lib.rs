//! OxiMedia DRM and Encryption Support
//!
//! This crate provides DRM (Digital Rights Management) and encryption support
//! for OxiMedia streaming, including:
//! - CENC (Common Encryption) implementation
//! - Widevine DRM
//! - PlayReady DRM
//! - FairPlay Streaming
//! - W3C Clear Key

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;
use uuid::Uuid;

// Hardcoded DRM system UUIDs as compile-time constants (no parsing, no panics).
const WIDEVINE_UUID: Uuid = Uuid::from_bytes([
    0xed, 0xef, 0x8b, 0xa9, 0x79, 0xd6, 0x4a, 0xce, 0xa3, 0xc8, 0x27, 0xdc, 0xd5, 0x1d, 0x21, 0xed,
]);
const PLAYREADY_UUID: Uuid = Uuid::from_bytes([
    0x9a, 0x04, 0xf0, 0x79, 0x98, 0x40, 0x42, 0x86, 0xab, 0x92, 0xe6, 0x5b, 0xe0, 0x88, 0x5f, 0x95,
]);
const FAIRPLAY_UUID: Uuid = Uuid::from_bytes([
    0x94, 0xce, 0x86, 0xfb, 0x07, 0xff, 0x4f, 0x43, 0xad, 0xb8, 0x93, 0xd2, 0xfa, 0x96, 0x8c, 0xa2,
]);
const CLEARKEY_UUID: Uuid = Uuid::from_bytes([
    0x10, 0x77, 0xef, 0xec, 0xc0, 0xb2, 0x4d, 0x02, 0xac, 0xe3, 0x3c, 0x1e, 0x52, 0xe2, 0xfb, 0x4b,
]);

pub mod access_grant;
pub mod aes_cbc;
pub mod aes_ctr;
pub mod analytics;
pub mod audit_trail;
pub mod buf_pool;
pub mod cenc;
pub mod cmaf_encrypt;
pub mod compliance;
pub mod content_key;
pub mod cpix;
pub mod device_auth;
pub mod device_registry;
pub mod entitlement;
pub mod geo_fence;
pub mod hw_key_store;
pub mod key_lifecycle;
pub mod key_management;
pub mod key_rotation;
pub mod key_rotation_schedule;
pub mod key_wrap;
pub mod license_chain;
pub mod license_server;
pub mod license_validator;
pub mod managed_license;
pub mod multi_drm;
pub mod multi_key;
pub mod offline;
pub mod output_control;
pub mod playback_policy;
pub mod playback_rules;
pub mod policy;
pub mod policy_engine;
pub mod pssh;
pub mod rate_limit;
pub mod session_token;
pub mod token;
pub mod watermark_detect;
pub mod watermark_embed;

#[cfg(feature = "clearkey")]
pub mod clearkey;

#[cfg(feature = "widevine")]
pub mod widevine;

#[cfg(feature = "playready")]
pub mod playready;

#[cfg(feature = "fairplay")]
pub mod fairplay;

/// DRM-related errors
#[derive(Error, Debug)]
pub enum DrmError {
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Decryption error: {0}")]
    DecryptionError(String),

    #[error("Invalid key: {0}")]
    InvalidKey(String),

    #[error("Invalid initialization vector: {0}")]
    InvalidIv(String),

    #[error("License error: {0}")]
    LicenseError(String),

    #[error("PSSH parsing error: {0}")]
    PsshError(String),

    #[error("Unsupported DRM system: {0}")]
    UnsupportedDrmSystem(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("XML error: {0}")]
    XmlError(String),
}

pub type Result<T> = std::result::Result<T, DrmError>;

/// DRM system identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DrmSystem {
    /// Widevine DRM (Google)
    Widevine,
    /// PlayReady DRM (Microsoft)
    PlayReady,
    /// FairPlay Streaming (Apple)
    FairPlay,
    /// W3C Clear Key (for testing)
    ClearKey,
}

impl DrmSystem {
    /// Get the system ID UUID for this DRM system
    pub fn system_id(&self) -> Uuid {
        match self {
            DrmSystem::Widevine => WIDEVINE_UUID,
            DrmSystem::PlayReady => PLAYREADY_UUID,
            DrmSystem::FairPlay => FAIRPLAY_UUID,
            DrmSystem::ClearKey => CLEARKEY_UUID,
        }
    }

    /// Get DRM system from UUID
    pub fn from_uuid(uuid: &Uuid) -> Option<Self> {
        if *uuid == WIDEVINE_UUID {
            Some(DrmSystem::Widevine)
        } else if *uuid == PLAYREADY_UUID {
            Some(DrmSystem::PlayReady)
        } else if *uuid == FAIRPLAY_UUID {
            Some(DrmSystem::FairPlay)
        } else if *uuid == CLEARKEY_UUID {
            Some(DrmSystem::ClearKey)
        } else {
            None
        }
    }
}

impl fmt::Display for DrmSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DrmSystem::Widevine => write!(f, "Widevine"),
            DrmSystem::PlayReady => write!(f, "PlayReady"),
            DrmSystem::FairPlay => write!(f, "FairPlay"),
            DrmSystem::ClearKey => write!(f, "ClearKey"),
        }
    }
}

/// DRM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrmConfig {
    /// DRM system to use
    pub system: DrmSystem,
    /// Content key ID
    pub key_id: Vec<u8>,
    /// Content key
    pub key: Vec<u8>,
    /// License server URL
    pub license_url: Option<String>,
    /// Additional system-specific data
    pub system_data: Option<Vec<u8>>,
}

impl DrmConfig {
    /// Create a new DRM configuration
    pub fn new(system: DrmSystem, key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            system,
            key_id,
            key,
            license_url: None,
            system_data: None,
        }
    }

    /// Set license server URL
    pub fn with_license_url(mut self, url: String) -> Self {
        self.license_url = Some(url);
        self
    }

    /// Set system-specific data
    pub fn with_system_data(mut self, data: Vec<u8>) -> Self {
        self.system_data = Some(data);
        self
    }
}

/// Key provider trait for obtaining content keys
pub trait KeyProvider: Send + Sync {
    /// Get a content key by key ID
    fn get_key(&self, key_id: &[u8]) -> Result<Vec<u8>>;

    /// Get multiple keys at once
    fn get_keys(&self, key_ids: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        key_ids.iter().map(|id| self.get_key(id)).collect()
    }

    /// Check if a key exists
    fn has_key(&self, key_id: &[u8]) -> bool {
        self.get_key(key_id).is_ok()
    }
}

/// Simple in-memory key provider
#[derive(Debug, Clone)]
pub struct MemoryKeyProvider {
    keys: std::collections::HashMap<Vec<u8>, Vec<u8>>,
}

impl MemoryKeyProvider {
    /// Create a new memory key provider
    pub fn new() -> Self {
        Self {
            keys: std::collections::HashMap::new(),
        }
    }

    /// Add a key to the provider
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Add multiple keys at once
    pub fn add_keys(&mut self, keys: Vec<(Vec<u8>, Vec<u8>)>) {
        for (key_id, key) in keys {
            self.keys.insert(key_id, key);
        }
    }

    /// Remove a key
    pub fn remove_key(&mut self, key_id: &[u8]) {
        self.keys.remove(key_id);
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Check if provider is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Default for MemoryKeyProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyProvider for MemoryKeyProvider {
    fn get_key(&self, key_id: &[u8]) -> Result<Vec<u8>> {
        self.keys
            .get(key_id)
            .cloned()
            .ok_or_else(|| DrmError::InvalidKey("Key not found".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drm_system_uuid() {
        let widevine_uuid =
            Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed").expect("UUID should parse");
        assert_eq!(DrmSystem::Widevine.system_id(), widevine_uuid);

        let playready_uuid =
            Uuid::parse_str("9a04f079-9840-4286-ab92-e65be0885f95").expect("UUID should parse");
        assert_eq!(DrmSystem::PlayReady.system_id(), playready_uuid);

        let fairplay_uuid =
            Uuid::parse_str("94ce86fb-07ff-4f43-adb8-93d2fa968ca2").expect("UUID should parse");
        assert_eq!(DrmSystem::FairPlay.system_id(), fairplay_uuid);

        let clearkey_uuid =
            Uuid::parse_str("1077efec-c0b2-4d02-ace3-3c1e52e2fb4b").expect("UUID should parse");
        assert_eq!(DrmSystem::ClearKey.system_id(), clearkey_uuid);
    }

    #[test]
    fn test_drm_system_from_uuid() {
        let widevine_uuid =
            Uuid::parse_str("edef8ba9-79d6-4ace-a3c8-27dcd51d21ed").expect("UUID should parse");
        assert_eq!(
            DrmSystem::from_uuid(&widevine_uuid),
            Some(DrmSystem::Widevine)
        );

        let unknown_uuid =
            Uuid::parse_str("00000000-0000-0000-0000-000000000000").expect("UUID should parse");
        assert_eq!(DrmSystem::from_uuid(&unknown_uuid), None);
    }

    #[test]
    fn test_memory_key_provider() {
        let mut provider = MemoryKeyProvider::new();
        let key_id = vec![1, 2, 3, 4];
        let key = vec![5, 6, 7, 8];

        provider.add_key(key_id.clone(), key.clone());
        assert!(provider.has_key(&key_id));
        assert_eq!(
            provider.get_key(&key_id).expect("get_key should succeed"),
            key
        );

        provider.remove_key(&key_id);
        assert!(!provider.has_key(&key_id));
    }
}
