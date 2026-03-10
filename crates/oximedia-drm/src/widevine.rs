//! Widevine DRM implementation
//!
//! Supports Widevine L3 (software-based) DRM integration.
//! Note: This is a partial implementation for educational purposes.
//! Full Widevine integration requires licensed CDM libraries.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Widevine PSSH data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevinePsshData {
    /// Algorithm (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<String>,
    /// Key IDs
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub key_ids: Vec<Vec<u8>>,
    /// Provider (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Content ID (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_id: Option<Vec<u8>>,
    /// Track type (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_type: Option<String>,
    /// Policy (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub policy: Option<String>,
    /// Crypto period index (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub crypto_period_index: Option<u32>,
    /// Protection scheme (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protection_scheme: Option<String>,
}

impl WidevinePsshData {
    /// Create new Widevine PSSH data
    pub fn new() -> Self {
        Self {
            algorithm: None,
            key_ids: Vec::new(),
            provider: None,
            content_id: None,
            track_type: None,
            policy: None,
            crypto_period_index: None,
            protection_scheme: Some("cenc".to_string()),
        }
    }

    /// Add a key ID
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.key_ids.push(key_id);
    }

    /// Set provider
    pub fn with_provider(mut self, provider: String) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set content ID
    pub fn with_content_id(mut self, content_id: Vec<u8>) -> Self {
        self.content_id = Some(content_id);
        self
    }

    /// Set track type
    pub fn with_track_type(mut self, track_type: String) -> Self {
        self.track_type = Some(track_type);
        self
    }

    /// Set protection scheme
    pub fn with_protection_scheme(mut self, scheme: String) -> Self {
        self.protection_scheme = Some(scheme);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from bytes (simplified - would use protobuf in production)
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

impl Default for WidevinePsshData {
    fn default() -> Self {
        Self::new()
    }
}

/// Widevine license request type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseType {
    /// Streaming license (temporary)
    Streaming,
    /// Offline license (persistent)
    Offline,
    /// License renewal
    Renewal,
    /// License release
    Release,
}

impl LicenseType {
    /// Get license type as string
    pub fn as_str(&self) -> &'static str {
        match self {
            LicenseType::Streaming => "STREAMING",
            LicenseType::Offline => "OFFLINE",
            LicenseType::Renewal => "RENEWAL",
            LicenseType::Release => "RELEASE",
        }
    }
}

/// Widevine license request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineLicenseRequest {
    /// License type
    pub license_type: LicenseType,
    /// Content ID
    pub content_id: Vec<u8>,
    /// Key IDs being requested
    pub key_ids: Vec<Vec<u8>>,
    /// Client ID (optional)
    pub client_id: Option<Vec<u8>>,
    /// Request ID (optional)
    pub request_id: Option<Vec<u8>>,
    /// Session ID (optional)
    pub session_id: Option<Vec<u8>>,
}

impl WidevineLicenseRequest {
    /// Create a new Widevine license request
    pub fn new(license_type: LicenseType, content_id: Vec<u8>, key_ids: Vec<Vec<u8>>) -> Self {
        Self {
            license_type,
            content_id,
            key_ids,
            client_id: None,
            request_id: None,
            session_id: None,
        }
    }

    /// Set client ID
    pub fn with_client_id(mut self, client_id: Vec<u8>) -> Self {
        self.client_id = Some(client_id);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Serialize to base64
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(STANDARD.encode(&bytes))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

/// Widevine key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineKey {
    /// Key ID
    pub key_id: Vec<u8>,
    /// Key value (encrypted)
    pub key: Vec<u8>,
    /// Key type (optional)
    pub key_type: Option<String>,
}

impl WidevineKey {
    /// Create a new Widevine key
    pub fn new(key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            key_id,
            key,
            key_type: Some("CONTENT".to_string()),
        }
    }

    /// Set key type
    pub fn with_key_type(mut self, key_type: String) -> Self {
        self.key_type = Some(key_type);
        self
    }
}

/// Widevine license response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidevineLicenseResponse {
    /// Status (e.g., "OK")
    pub status: String,
    /// License duration in seconds (optional)
    pub license_duration: Option<u64>,
    /// Playback duration in seconds (optional)
    pub playback_duration: Option<u64>,
    /// Renewal server URL (optional)
    pub renewal_server_url: Option<String>,
    /// Keys
    pub keys: Vec<WidevineKey>,
}

impl WidevineLicenseResponse {
    /// Create a new Widevine license response
    pub fn new(keys: Vec<WidevineKey>) -> Self {
        Self {
            status: "OK".to_string(),
            license_duration: None,
            playback_duration: None,
            renewal_server_url: None,
            keys,
        }
    }

    /// Set license duration
    pub fn with_license_duration(mut self, duration: u64) -> Self {
        self.license_duration = Some(duration);
        self
    }

    /// Set playback duration
    pub fn with_playback_duration(mut self, duration: u64) -> Self {
        self.playback_duration = Some(duration);
        self
    }

    /// Serialize to bytes (simplified - would use protobuf in production)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Serialize to base64
    pub fn to_base64(&self) -> Result<String> {
        let bytes = self.to_bytes()?;
        Ok(STANDARD.encode(&bytes))
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }

    /// Get keys as a map
    pub fn get_keys_map(&self) -> HashMap<Vec<u8>, Vec<u8>> {
        self.keys
            .iter()
            .map(|k| (k.key_id.clone(), k.key.clone()))
            .collect()
    }
}

/// Widevine key hierarchy levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLevel {
    /// Root key
    Root,
    /// Intermediate key
    Intermediate,
    /// Content key
    Content,
}

/// Widevine CDM (Content Decryption Module) client (simplified)
pub struct WidevineCdm {
    /// Client ID
    client_id: Vec<u8>,
    /// Session keys
    sessions: HashMap<Vec<u8>, HashMap<Vec<u8>, Vec<u8>>>,
}

impl WidevineCdm {
    /// Create a new Widevine CDM
    pub fn new(client_id: Vec<u8>) -> Self {
        Self {
            client_id,
            sessions: HashMap::new(),
        }
    }

    /// Generate a license request
    pub fn generate_request(
        &self,
        license_type: LicenseType,
        content_id: Vec<u8>,
        key_ids: Vec<Vec<u8>>,
    ) -> Result<WidevineLicenseRequest> {
        let request = WidevineLicenseRequest::new(license_type, content_id, key_ids)
            .with_client_id(self.client_id.clone());

        Ok(request)
    }

    /// Process a license response
    pub fn process_response(
        &mut self,
        session_id: Vec<u8>,
        response: &WidevineLicenseResponse,
    ) -> Result<()> {
        if response.status != "OK" {
            return Err(DrmError::LicenseError(format!(
                "License error: {}",
                response.status
            )));
        }

        // Store keys for this session
        let session_keys: HashMap<Vec<u8>, Vec<u8>> = response.get_keys_map();
        self.sessions.insert(session_id, session_keys);

        Ok(())
    }

    /// Get a content key for a session
    pub fn get_key(&self, session_id: &[u8], key_id: &[u8]) -> Option<Vec<u8>> {
        self.sessions
            .get(session_id)
            .and_then(|keys| keys.get(key_id).cloned())
    }

    /// Close a session and remove its keys
    pub fn close_session(&mut self, session_id: &[u8]) {
        self.sessions.remove(session_id);
    }

    /// Get number of active sessions
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

/// Widevine license server (for testing/mocking)
pub struct WidevineLicenseServer {
    keys: HashMap<Vec<u8>, HashMap<Vec<u8>, Vec<u8>>>, // content_id -> (key_id -> key)
    license_duration: u64,
    playback_duration: u64,
}

impl WidevineLicenseServer {
    /// Create a new Widevine license server
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
            license_duration: 86400, // 24 hours
            playback_duration: 7200, // 2 hours
        }
    }

    /// Add keys for a content ID
    pub fn add_content_keys(&mut self, content_id: Vec<u8>, keys: HashMap<Vec<u8>, Vec<u8>>) {
        self.keys.insert(content_id, keys);
    }

    /// Add a single key for a content ID
    pub fn add_key(&mut self, content_id: Vec<u8>, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys
            .entry(content_id)
            .or_insert_with(HashMap::new)
            .insert(key_id, key);
    }

    /// Set license duration
    pub fn set_license_duration(&mut self, duration: u64) {
        self.license_duration = duration;
    }

    /// Set playback duration
    pub fn set_playback_duration(&mut self, duration: u64) {
        self.playback_duration = duration;
    }

    /// Process a license request
    pub fn process_request(
        &self,
        request: &WidevineLicenseRequest,
    ) -> Result<WidevineLicenseResponse> {
        // Get keys for this content
        let content_keys = self.keys.get(&request.content_id).ok_or_else(|| {
            DrmError::LicenseError(format!(
                "Content not found: {}",
                hex::encode(&request.content_id)
            ))
        })?;

        // Collect requested keys
        let mut response_keys = Vec::new();
        for key_id in &request.key_ids {
            let key = content_keys.get(key_id).ok_or_else(|| {
                DrmError::LicenseError(format!("Key not found: {}", hex::encode(key_id)))
            })?;

            response_keys.push(WidevineKey::new(key_id.clone(), key.clone()));
        }

        Ok(WidevineLicenseResponse::new(response_keys)
            .with_license_duration(self.license_duration)
            .with_playback_duration(self.playback_duration))
    }

    /// Get number of content entries
    pub fn content_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for WidevineLicenseServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a Widevine PSSH box
pub fn create_widevine_pssh(key_ids: Vec<Vec<u8>>, content_id: Vec<u8>) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let mut pssh_data = WidevinePsshData::new();
    for key_id in &key_ids {
        pssh_data.add_key_id(key_id.clone());
    }
    pssh_data = pssh_data.with_content_id(content_id);

    let data = pssh_data.to_bytes()?;
    let pssh = PsshBox::new_v1(DrmSystem::Widevine.system_id(), key_ids, data);
    pssh.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widevine_pssh_data() {
        let mut pssh_data = WidevinePsshData::new();
        pssh_data.add_key_id(vec![1, 2, 3, 4]);
        pssh_data.add_key_id(vec![5, 6, 7, 8]);

        let bytes = pssh_data.to_bytes().expect("operation should succeed");
        let parsed = WidevinePsshData::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(parsed.key_ids.len(), 2);
        assert_eq!(parsed.key_ids[0], vec![1, 2, 3, 4]);
        assert_eq!(parsed.key_ids[1], vec![5, 6, 7, 8]);
    }

    #[test]
    fn test_license_type() {
        assert_eq!(LicenseType::Streaming.as_str(), "STREAMING");
        assert_eq!(LicenseType::Offline.as_str(), "OFFLINE");
        assert_eq!(LicenseType::Renewal.as_str(), "RENEWAL");
        assert_eq!(LicenseType::Release.as_str(), "RELEASE");
    }

    #[test]
    fn test_license_request() {
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(
            LicenseType::Streaming,
            content_id.clone(),
            vec![key_id.clone()],
        );

        assert_eq!(request.license_type, LicenseType::Streaming);
        assert_eq!(request.content_id, content_id);
        assert_eq!(request.key_ids[0], key_id);
    }

    #[test]
    fn test_license_request_serialization() {
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id]);

        let bytes = request.to_bytes().expect("operation should succeed");
        let parsed = WidevineLicenseRequest::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(parsed.license_type, request.license_type);
        assert_eq!(parsed.content_id, request.content_id);
    }

    #[test]
    fn test_license_response() {
        let key_id = vec![1, 2, 3, 4];
        let key = vec![5, 6, 7, 8];

        let widevine_key = WidevineKey::new(key_id.clone(), key.clone());
        let response = WidevineLicenseResponse::new(vec![widevine_key])
            .with_license_duration(3600)
            .with_playback_duration(1800);

        assert_eq!(response.status, "OK");
        assert_eq!(response.license_duration, Some(3600));
        assert_eq!(response.playback_duration, Some(1800));
        assert_eq!(response.keys.len(), 1);

        let keys_map = response.get_keys_map();
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_widevine_cdm() {
        let client_id = vec![1, 2, 3, 4];
        let mut cdm = WidevineCdm::new(client_id.clone());

        let content_id = vec![5, 6, 7, 8];
        let key_id = vec![9, 10, 11, 12];
        let key = vec![13, 14, 15, 16];

        let request = cdm
            .generate_request(LicenseType::Streaming, content_id, vec![key_id.clone()])
            .expect("operation should succeed");

        assert_eq!(request.client_id, Some(client_id));

        let widevine_key = WidevineKey::new(key_id.clone(), key.clone());
        let response = WidevineLicenseResponse::new(vec![widevine_key]);

        let session_id = vec![20, 21, 22, 23];
        cdm.process_response(session_id.clone(), &response)
            .expect("operation should succeed");

        assert_eq!(cdm.get_key(&session_id, &key_id), Some(key));
        assert_eq!(cdm.session_count(), 1);

        cdm.close_session(&session_id);
        assert_eq!(cdm.session_count(), 0);
    }

    #[test]
    fn test_license_server() {
        let mut server = WidevineLicenseServer::new();
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];
        let key = vec![9, 10, 11, 12];

        server.add_key(content_id.clone(), key_id.clone(), key.clone());

        let request =
            WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id.clone()]);

        let response = server
            .process_request(&request)
            .expect("operation should succeed");

        assert_eq!(response.status, "OK");
        assert_eq!(response.keys.len(), 1);
        assert_eq!(response.keys[0].key_id, key_id);
        assert_eq!(response.keys[0].key, key);
    }

    #[test]
    fn test_license_server_missing_content() {
        let server = WidevineLicenseServer::new();
        let content_id = vec![1, 2, 3, 4];
        let key_id = vec![5, 6, 7, 8];

        let request = WidevineLicenseRequest::new(LicenseType::Streaming, content_id, vec![key_id]);

        let result = server.process_request(&request);
        assert!(result.is_err());
    }
}
