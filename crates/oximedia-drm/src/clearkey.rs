//! W3C Clear Key DRM implementation
//!
//! Clear Key is a simple DRM system defined by W3C for testing and development.
//! It uses unencrypted key exchange and is not suitable for production use.

use crate::{DrmError, DrmSystem, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Clear Key license request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyRequest {
    /// List of key IDs being requested
    pub kids: Vec<String>,
    /// Request type (typically "temporary")
    #[serde(rename = "type")]
    pub request_type: Option<String>,
}

impl ClearKeyRequest {
    /// Create a new Clear Key request
    pub fn new(key_ids: Vec<Vec<u8>>) -> Self {
        let kids = key_ids
            .into_iter()
            .map(|id| URL_SAFE_NO_PAD.encode(&id))
            .collect();

        Self {
            kids,
            request_type: Some("temporary".to_string()),
        }
    }

    /// Add a key ID to the request
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.kids.push(URL_SAFE_NO_PAD.encode(&key_id));
    }

    /// Get key IDs as bytes
    pub fn get_key_ids(&self) -> Result<Vec<Vec<u8>>> {
        self.kids
            .iter()
            .map(|kid| URL_SAFE_NO_PAD.decode(kid).map_err(DrmError::Base64Error))
            .collect()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(DrmError::JsonError)
    }
}

/// JSON Web Key (JWK) for Clear Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonWebKey {
    /// Key type (always "oct" for symmetric keys)
    pub kty: String,
    /// Key ID (base64url encoded)
    pub kid: String,
    /// Key value (base64url encoded)
    pub k: String,
}

impl JsonWebKey {
    /// Create a new JSON Web Key
    pub fn new(key_id: Vec<u8>, key: Vec<u8>) -> Self {
        Self {
            kty: "oct".to_string(),
            kid: URL_SAFE_NO_PAD.encode(&key_id),
            k: URL_SAFE_NO_PAD.encode(&key),
        }
    }

    /// Get key ID as bytes
    pub fn get_key_id(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.kid)
            .map_err(DrmError::Base64Error)
    }

    /// Get key as bytes
    pub fn get_key(&self) -> Result<Vec<u8>> {
        URL_SAFE_NO_PAD
            .decode(&self.k)
            .map_err(DrmError::Base64Error)
    }
}

/// Clear Key license response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyResponse {
    /// List of keys in JWK format
    pub keys: Vec<JsonWebKey>,
    /// Response type (typically "temporary")
    #[serde(rename = "type")]
    pub response_type: Option<String>,
}

impl ClearKeyResponse {
    /// Create a new Clear Key response
    pub fn new(keys: Vec<JsonWebKey>) -> Self {
        Self {
            keys,
            response_type: Some("temporary".to_string()),
        }
    }

    /// Add a key to the response
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.push(JsonWebKey::new(key_id, key));
    }

    /// Get all keys as a map
    pub fn get_keys_map(&self) -> Result<HashMap<Vec<u8>, Vec<u8>>> {
        let mut map = HashMap::new();
        for jwk in &self.keys {
            let key_id = jwk.get_key_id()?;
            let key = jwk.get_key()?;
            map.insert(key_id, key);
        }
        Ok(map)
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(DrmError::JsonError)
    }
}

/// Clear Key license server (for testing)
pub struct ClearKeyServer {
    keys: HashMap<Vec<u8>, Vec<u8>>,
}

impl ClearKeyServer {
    /// Create a new Clear Key server
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Add a key to the server
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Add multiple keys
    pub fn add_keys(&mut self, keys: Vec<(Vec<u8>, Vec<u8>)>) {
        for (key_id, key) in keys {
            self.keys.insert(key_id, key);
        }
    }

    /// Process a license request and generate a response
    pub fn process_request(&self, request: &ClearKeyRequest) -> Result<ClearKeyResponse> {
        let mut response = ClearKeyResponse::new(Vec::new());

        let key_ids = request.get_key_ids()?;
        for key_id in key_ids {
            if let Some(key) = self.keys.get(&key_id) {
                response.add_key(key_id, key.clone());
            } else {
                return Err(DrmError::LicenseError(format!(
                    "Key not found: {}",
                    hex::encode(&key_id)
                )));
            }
        }

        Ok(response)
    }

    /// Process a request from JSON and return response as JSON
    pub fn process_request_json(&self, request_json: &str) -> Result<String> {
        let request = ClearKeyRequest::from_json(request_json)?;
        let response = self.process_request(&request)?;
        response.to_json()
    }

    /// Get number of keys in server
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }

    /// Check if server has a key
    pub fn has_key(&self, key_id: &[u8]) -> bool {
        self.keys.contains_key(key_id)
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }
}

impl Default for ClearKeyServer {
    fn default() -> Self {
        Self::new()
    }
}

/// Clear Key PSSH data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClearKeyPsshData {
    /// List of key IDs
    pub key_ids: Vec<String>,
}

impl ClearKeyPsshData {
    /// Create new Clear Key PSSH data
    pub fn new(key_ids: Vec<Vec<u8>>) -> Self {
        let key_ids = key_ids
            .into_iter()
            .map(|id| URL_SAFE_NO_PAD.encode(&id))
            .collect();

        Self { key_ids }
    }

    /// Add a key ID
    pub fn add_key_id(&mut self, key_id: Vec<u8>) {
        self.key_ids.push(URL_SAFE_NO_PAD.encode(&key_id));
    }

    /// Get key IDs as bytes
    pub fn get_key_ids(&self) -> Result<Vec<Vec<u8>>> {
        self.key_ids
            .iter()
            .map(|kid| URL_SAFE_NO_PAD.decode(kid).map_err(DrmError::Base64Error))
            .collect()
    }

    /// Serialize to JSON bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(DrmError::JsonError)
    }

    /// Deserialize from JSON bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data).map_err(DrmError::JsonError)
    }
}

/// Clear Key client
pub struct ClearKeyClient {
    keys: HashMap<Vec<u8>, Vec<u8>>,
}

impl ClearKeyClient {
    /// Create a new Clear Key client
    pub fn new() -> Self {
        Self {
            keys: HashMap::new(),
        }
    }

    /// Request keys from a Clear Key server
    pub fn request_keys(&mut self, key_ids: Vec<Vec<u8>>, server: &ClearKeyServer) -> Result<()> {
        let request = ClearKeyRequest::new(key_ids);
        let response = server.process_request(&request)?;

        for jwk in response.keys {
            let key_id = jwk.get_key_id()?;
            let key = jwk.get_key()?;
            self.keys.insert(key_id, key);
        }

        Ok(())
    }

    /// Add a key directly
    pub fn add_key(&mut self, key_id: Vec<u8>, key: Vec<u8>) {
        self.keys.insert(key_id, key);
    }

    /// Get a key by ID
    pub fn get_key(&self, key_id: &[u8]) -> Option<&Vec<u8>> {
        self.keys.get(key_id)
    }

    /// Get all keys
    pub fn get_all_keys(&self) -> &HashMap<Vec<u8>, Vec<u8>> {
        &self.keys
    }

    /// Clear all keys
    pub fn clear(&mut self) {
        self.keys.clear();
    }

    /// Get number of keys
    pub fn key_count(&self) -> usize {
        self.keys.len()
    }
}

impl Default for ClearKeyClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a Clear Key PSSH box
pub fn create_clearkey_pssh(key_ids: Vec<Vec<u8>>) -> Result<Vec<u8>> {
    use crate::cenc::PsshBox;

    let pssh_data = ClearKeyPsshData::new(key_ids.clone());
    let data = pssh_data.to_bytes()?;

    let pssh = PsshBox::new_v1(DrmSystem::ClearKey.system_id(), key_ids, data);
    pssh.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clearkey_request() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let request = ClearKeyRequest::new(vec![key_id.clone()]);

        assert_eq!(request.kids.len(), 1);
        let decoded = request.get_key_ids().expect("get_key_ids should succeed");
        assert_eq!(decoded[0], key_id);
    }

    #[test]
    fn test_clearkey_request_json() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let request = ClearKeyRequest::new(vec![key_id]);

        let json = request.to_json().expect("to_json should succeed");
        let parsed = ClearKeyRequest::from_json(&json).expect("from_json should parse");

        assert_eq!(parsed.kids, request.kids);
    }

    #[test]
    fn test_json_web_key() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let jwk = JsonWebKey::new(key_id.clone(), key.clone());

        assert_eq!(jwk.kty, "oct");
        assert_eq!(jwk.get_key_id().expect("get_key_id should decode"), key_id);
        assert_eq!(jwk.get_key().expect("get_key should decode"), key);
    }

    #[test]
    fn test_clearkey_response() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut response = ClearKeyResponse::new(Vec::new());
        response.add_key(key_id.clone(), key.clone());

        assert_eq!(response.keys.len(), 1);

        let keys_map = response
            .get_keys_map()
            .expect("get_keys_map should succeed");
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_response_json() {
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let mut response = ClearKeyResponse::new(Vec::new());
        response.add_key(key_id, key);

        let json = response.to_json().expect("to_json should succeed");
        let parsed = ClearKeyResponse::from_json(&json).expect("from_json should parse");

        assert_eq!(parsed.keys.len(), response.keys.len());
    }

    #[test]
    fn test_clearkey_server() {
        let mut server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        server.add_key(key_id.clone(), key.clone());
        assert!(server.has_key(&key_id));

        let request = ClearKeyRequest::new(vec![key_id.clone()]);
        let response = server
            .process_request(&request)
            .expect("process_request should succeed");

        assert_eq!(response.keys.len(), 1);
        let keys_map = response
            .get_keys_map()
            .expect("get_keys_map should succeed");
        assert_eq!(keys_map.get(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_server_missing_key() {
        let server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

        let request = ClearKeyRequest::new(vec![key_id]);
        let result = server.process_request(&request);

        assert!(result.is_err());
    }

    #[test]
    fn test_clearkey_client() {
        let mut server = ClearKeyServer::new();
        let key_id = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        server.add_key(key_id.clone(), key.clone());

        let mut client = ClearKeyClient::new();
        client
            .request_keys(vec![key_id.clone()], &server)
            .expect("request_keys should succeed");

        assert_eq!(client.key_count(), 1);
        assert_eq!(client.get_key(&key_id), Some(&key));
    }

    #[test]
    fn test_clearkey_pssh_data() {
        let key_id1 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let key_id2 = vec![16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

        let pssh_data = ClearKeyPsshData::new(vec![key_id1.clone(), key_id2.clone()]);
        let bytes = pssh_data.to_bytes().expect("to_bytes should succeed");

        let parsed = ClearKeyPsshData::from_bytes(&bytes).expect("from_bytes should parse");
        let parsed_ids = parsed.get_key_ids().expect("get_key_ids should decode");

        assert_eq!(parsed_ids.len(), 2);
        assert_eq!(parsed_ids[0], key_id1);
        assert_eq!(parsed_ids[1], key_id2);
    }
}
