//! FairPlay Streaming (FPS) implementation
//!
//! Supports Apple FairPlay Streaming DRM for HLS.
//! Note: This is a partial implementation for educational purposes.
//! Full FairPlay integration requires Apple developer credentials and FPS SDK.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// FairPlay content key context format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkcFormat {
    /// Binary format
    Binary,
    /// Base64 format
    Base64,
}

/// FairPlay streaming key request
#[derive(Debug, Clone)]
pub struct FairPlayKeyRequest {
    /// Asset ID
    pub asset_id: String,
    /// SPC (Server Playback Context) data
    pub spc_data: Vec<u8>,
    /// Application certificate
    pub certificate: Option<Vec<u8>>,
}

impl FairPlayKeyRequest {
    /// Create a new FairPlay key request
    pub fn new(asset_id: String, spc_data: Vec<u8>) -> Self {
        Self {
            asset_id,
            spc_data,
            certificate: None,
        }
    }

    /// Set application certificate
    pub fn with_certificate(mut self, certificate: Vec<u8>) -> Self {
        self.certificate = Some(certificate);
        self
    }

    /// Get SPC data as base64
    pub fn spc_base64(&self) -> String {
        STANDARD.encode(&self.spc_data)
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        #[derive(Serialize)]
        struct JsonRequest {
            asset_id: String,
            spc: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            certificate: Option<String>,
        }

        let json_req = JsonRequest {
            asset_id: self.asset_id.clone(),
            spc: self.spc_base64(),
            certificate: self.certificate.as_ref().map(|cert| STANDARD.encode(cert)),
        };

        serde_json::to_string(&json_req).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        #[derive(Deserialize)]
        struct JsonRequest {
            asset_id: String,
            spc: String,
            #[serde(default)]
            certificate: Option<String>,
        }

        let json_req: JsonRequest = serde_json::from_str(json).map_err(DrmError::JsonError)?;

        let spc_data = STANDARD
            .decode(&json_req.spc)
            .map_err(DrmError::Base64Error)?;

        let certificate = if let Some(cert_b64) = json_req.certificate {
            Some(STANDARD.decode(&cert_b64).map_err(DrmError::Base64Error)?)
        } else {
            None
        };

        let mut request = Self::new(json_req.asset_id, spc_data);
        if let Some(cert) = certificate {
            request = request.with_certificate(cert);
        }

        Ok(request)
    }
}

/// FairPlay streaming key response (CKC - Content Key Context)
#[derive(Debug, Clone)]
pub struct FairPlayKeyResponse {
    /// CKC (Content Key Context) data
    pub ckc_data: Vec<u8>,
    /// Format
    pub format: CkcFormat,
}

impl FairPlayKeyResponse {
    /// Create a new FairPlay key response
    pub fn new(ckc_data: Vec<u8>, format: CkcFormat) -> Self {
        Self { ckc_data, format }
    }

    /// Get CKC data as base64
    pub fn ckc_base64(&self) -> String {
        STANDARD.encode(&self.ckc_data)
    }

    /// Get formatted CKC data based on format
    pub fn formatted_ckc(&self) -> Vec<u8> {
        match self.format {
            CkcFormat::Binary => self.ckc_data.clone(),
            CkcFormat::Base64 => self.ckc_base64().into_bytes(),
        }
    }
}

/// FairPlay content key
#[derive(Debug, Clone)]
pub struct FairPlayContentKey {
    /// Asset ID
    pub asset_id: String,
    /// Content key
    pub key: Vec<u8>,
    /// Key ID (optional)
    pub key_id: Option<Vec<u8>>,
    /// Initialization vector (optional)
    pub iv: Option<Vec<u8>>,
}

impl FairPlayContentKey {
    /// Create a new FairPlay content key
    pub fn new(asset_id: String, key: Vec<u8>) -> Self {
        Self {
            asset_id,
            key,
            key_id: None,
            iv: None,
        }
    }

    /// Set key ID
    pub fn with_key_id(mut self, key_id: Vec<u8>) -> Self {
        self.key_id = Some(key_id);
        self
    }

    /// Set initialization vector
    pub fn with_iv(mut self, iv: Vec<u8>) -> Self {
        self.iv = Some(iv);
        self
    }
}

/// FairPlay key server (KSM - Key Security Module)
pub struct FairPlayKeyServer {
    /// Content keys indexed by asset ID
    keys: HashMap<String, FairPlayContentKey>,
    /// Application secret key
    #[allow(dead_code)]
    app_secret_key: Vec<u8>,
    /// Random number generator
    #[allow(dead_code)]
    rng: SystemRandom,
}

impl FairPlayKeyServer {
    /// Create a new FairPlay key server
    pub fn new(app_secret_key: Vec<u8>) -> Self {
        Self {
            keys: HashMap::new(),
            app_secret_key,
            rng: SystemRandom::new(),
        }
    }

    /// Add a content key
    pub fn add_key(&mut self, content_key: FairPlayContentKey) {
        self.keys.insert(content_key.asset_id.clone(), content_key);
    }

    /// Process an SPC and generate CKC
    pub fn process_spc(&self, request: &FairPlayKeyRequest) -> Result<FairPlayKeyResponse> {
        // Get the content key for this asset
        let content_key = self.keys.get(&request.asset_id).ok_or_else(|| {
            DrmError::LicenseError(format!("Asset not found: {}", request.asset_id))
        })?;

        // In a real implementation, this would:
        // 1. Verify the SPC signature
        // 2. Decrypt the SPC using the app secret key
        // 3. Extract the session key
        // 4. Encrypt the content key with the session key
        // 5. Create and sign the CKC

        // Simplified version: create a CKC containing the content key
        let ckc_data = self.create_ckc(content_key)?;

        Ok(FairPlayKeyResponse::new(ckc_data, CkcFormat::Binary))
    }

    /// Create CKC data (simplified)
    fn create_ckc(&self, content_key: &FairPlayContentKey) -> Result<Vec<u8>> {
        // In a real implementation, this would create a properly formatted CKC
        // For now, we create a simple structure
        let mut ckc = Vec::new();

        // Magic number (simplified)
        ckc.extend_from_slice(b"CKC\x00");

        // Version
        ckc.push(1);

        // Key length
        ckc.push(content_key.key.len() as u8);

        // Key data
        ckc.extend_from_slice(&content_key.key);

        // IV (if present)
        if let Some(ref iv) = content_key.iv {
            ckc.push(iv.len() as u8);
            ckc.extend_from_slice(iv);
        } else {
            ckc.push(0);
        }

        Ok(ckc)
    }

    /// Get number of keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Check if asset exists
    pub fn has_asset(&self, asset_id: &str) -> bool {
        self.keys.contains_key(asset_id)
    }
}

/// FairPlay SPC generator (simplified)
pub struct FairPlaySpcGenerator {
    rng: SystemRandom,
}

impl FairPlaySpcGenerator {
    /// Create a new SPC generator
    pub fn new() -> Self {
        Self {
            rng: SystemRandom::new(),
        }
    }

    /// Generate an SPC (Server Playback Context)
    pub fn generate_spc(&self, asset_id: &str, certificate: &[u8]) -> Result<Vec<u8>> {
        // In a real implementation, this would:
        // 1. Generate a session key
        // 2. Encrypt the asset ID and other metadata
        // 3. Sign the SPC
        // 4. Return the complete SPC blob

        // Simplified version: create a basic structure
        let mut spc = Vec::new();

        // Magic number
        spc.extend_from_slice(b"SPC\x00");

        // Version
        spc.push(1);

        // Random session ID
        let mut session_id = vec![0u8; 16];
        self.rng.fill(&mut session_id).map_err(|e| {
            DrmError::EncryptionError(format!("Failed to generate session ID: {:?}", e))
        })?;
        spc.extend_from_slice(&session_id);

        // Asset ID length and data
        let asset_id_bytes = asset_id.as_bytes();
        spc.push(asset_id_bytes.len() as u8);
        spc.extend_from_slice(asset_id_bytes);

        // Certificate hash (simplified - just first 16 bytes)
        let cert_hash = &certificate[..16.min(certificate.len())];
        spc.push(cert_hash.len() as u8);
        spc.extend_from_slice(cert_hash);

        Ok(spc)
    }
}

impl Default for FairPlaySpcGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// FairPlay client
pub struct FairPlayClient {
    /// Application certificate
    certificate: Vec<u8>,
    /// SPC generator
    spc_generator: FairPlaySpcGenerator,
    /// CKC cache
    ckc_cache: HashMap<String, Vec<u8>>,
}

impl FairPlayClient {
    /// Create a new FairPlay client
    pub fn new(certificate: Vec<u8>) -> Self {
        Self {
            certificate,
            spc_generator: FairPlaySpcGenerator::new(),
            ckc_cache: HashMap::new(),
        }
    }

    /// Request a key for an asset
    pub fn request_key(&mut self, asset_id: String) -> Result<FairPlayKeyRequest> {
        // Generate SPC
        let spc_data = self
            .spc_generator
            .generate_spc(&asset_id, &self.certificate)?;

        Ok(FairPlayKeyRequest::new(asset_id, spc_data).with_certificate(self.certificate.clone()))
    }

    /// Process CKC response
    pub fn process_ckc(&mut self, asset_id: String, response: FairPlayKeyResponse) -> Result<()> {
        // In a real implementation, this would:
        // 1. Verify the CKC signature
        // 2. Extract the content key
        // 3. Store it for decryption

        // Simplified: just cache the CKC
        self.ckc_cache.insert(asset_id, response.ckc_data);

        Ok(())
    }

    /// Get cached CKC for an asset
    pub fn get_ckc(&self, asset_id: &str) -> Option<&Vec<u8>> {
        self.ckc_cache.get(asset_id)
    }

    /// Clear CKC cache
    pub fn clear_cache(&mut self) {
        self.ckc_cache.clear();
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.ckc_cache.len()
    }
}

/// FairPlay PSSH data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairPlayPsshData {
    /// Asset ID
    pub asset_id: String,
    /// Key IDs
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub key_ids: Vec<Vec<u8>>,
    /// License server URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license_url: Option<String>,
}

impl FairPlayPsshData {
    /// Create new FairPlay PSSH data
    pub fn new(asset_id: String) -> Self {
        Self {
            asset_id,
            key_ids: Vec::new(),
            license_url: None,
        }
    }

    /// Add a key ID
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.key_ids.push(key_id);
    }

    /// Set license server URL
    pub fn with_license_url(mut self, url: String) -> Self {
        self.license_url = Some(url);
        self
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

/// Create a FairPlay PSSH box
pub fn create_fairplay_pssh(asset_id: String, key_ids: Vec<Vec<u8>>) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let mut pssh_data = FairPlayPsshData::new(asset_id);
    for key_id in &key_ids {
        pssh_data.add_key_id(key_id.clone());
    }

    let data = pssh_data.to_bytes()?;
    let pssh = PsshBox::new_v1(DrmSystem::FairPlay.system_id(), key_ids, data);
    pssh.to_bytes()
}

/// HLS integration helpers
pub mod hls {
    /// Generate HLS key URI for FairPlay
    pub fn generate_key_uri(license_url: &str, asset_id: &str) -> String {
        format!("skd://{}?asset={}", license_url, asset_id)
    }

    /// Parse asset ID from HLS key URI
    pub fn parse_asset_id_from_uri(uri: &str) -> Option<String> {
        if !uri.starts_with("skd://") {
            return None;
        }

        // Extract asset ID from query parameter
        if let Some(query_start) = uri.find('?') {
            let query = &uri[query_start + 1..];
            for param in query.split('&') {
                if let Some((key, value)) = param.split_once('=') {
                    if key == "asset" {
                        return Some(value.to_string());
                    }
                }
            }
        }

        None
    }

    /// Generate HLS #EXT-X-KEY tag for FairPlay
    pub fn generate_key_tag(license_url: &str, asset_id: &str, key_format: &str) -> String {
        let uri = generate_key_uri(license_url, asset_id);
        format!(
            "#EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"{}\",KEYFORMAT=\"{}\",KEYFORMATVERSIONS=\"1\"",
            uri, key_format
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fairplay_key_request() {
        let asset_id = "test_asset".to_string();
        let spc_data = vec![1, 2, 3, 4, 5];

        let request = FairPlayKeyRequest::new(asset_id.clone(), spc_data.clone());

        assert_eq!(request.asset_id, asset_id);
        assert_eq!(request.spc_data, spc_data);
        assert!(request.certificate.is_none());
    }

    #[test]
    fn test_fairplay_key_request_json() {
        let asset_id = "test_asset".to_string();
        let spc_data = vec![1, 2, 3, 4, 5];

        let request = FairPlayKeyRequest::new(asset_id, spc_data);
        let json = request.to_json().expect("operation should succeed");

        let parsed = FairPlayKeyRequest::from_json(&json).expect("operation should succeed");
        assert_eq!(parsed.asset_id, request.asset_id);
        assert_eq!(parsed.spc_data, request.spc_data);
    }

    #[test]
    fn test_fairplay_content_key() {
        let asset_id = "test_asset".to_string();
        let key = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key_id = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let content_key =
            FairPlayContentKey::new(asset_id.clone(), key.clone()).with_key_id(key_id.clone());

        assert_eq!(content_key.asset_id, asset_id);
        assert_eq!(content_key.key, key);
        assert_eq!(content_key.key_id, Some(key_id));
    }

    #[test]
    fn test_fairplay_key_server() {
        let app_secret = vec![0u8; 32];
        let mut server = FairPlayKeyServer::new(app_secret);

        let asset_id = "test_asset".to_string();
        let key = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let content_key = FairPlayContentKey::new(asset_id.clone(), key);

        server.add_key(content_key);
        assert!(server.has_asset(&asset_id));
        assert_eq!(server.key_count(), 1);
    }

    #[test]
    fn test_spc_generator() {
        let generator = FairPlaySpcGenerator::new();
        let asset_id = "test_asset";
        let certificate = vec![0u8; 32];

        let spc = generator
            .generate_spc(asset_id, &certificate)
            .expect("operation should succeed");
        assert!(!spc.is_empty());
        assert!(spc.starts_with(b"SPC\x00"));
    }

    #[test]
    fn test_fairplay_client() {
        let certificate = vec![0u8; 32];
        let mut client = FairPlayClient::new(certificate);

        let asset_id = "test_asset".to_string();
        let request = client
            .request_key(asset_id.clone())
            .expect("operation should succeed");

        assert_eq!(request.asset_id, asset_id);
        assert!(!request.spc_data.is_empty());
    }

    #[test]
    fn test_fairplay_pssh_data() {
        let asset_id = "test_asset".to_string();
        let mut pssh_data = FairPlayPsshData::new(asset_id.clone());
        pssh_data.add_key_id(vec![1, 2, 3, 4]);

        let bytes = pssh_data.to_bytes().expect("operation should succeed");
        let parsed = FairPlayPsshData::from_bytes(&bytes).expect("operation should succeed");

        assert_eq!(parsed.asset_id, asset_id);
        assert_eq!(parsed.key_ids.len(), 1);
    }

    #[test]
    fn test_hls_key_uri() {
        let license_url = "license.example.com";
        let asset_id = "test_asset";

        let uri = hls::generate_key_uri(license_url, asset_id);
        assert_eq!(uri, "skd://license.example.com?asset=test_asset");

        let parsed_asset = hls::parse_asset_id_from_uri(&uri).expect("operation should succeed");
        assert_eq!(parsed_asset, asset_id);
    }

    #[test]
    fn test_hls_key_tag() {
        let license_url = "license.example.com";
        let asset_id = "test_asset";
        let key_format = "com.apple.streamingkeydelivery";

        let tag = hls::generate_key_tag(license_url, asset_id, key_format);
        assert!(tag.contains("#EXT-X-KEY"));
        assert!(tag.contains("METHOD=SAMPLE-AES"));
        assert!(tag.contains("skd://"));
        assert!(tag.contains(key_format));
    }
}
