//! License server simulation for DRM systems.
//!
//! Provides a simulated license server that can issue, cache, and validate
//! DRM licenses for Widevine, PlayReady, FairPlay, and ClearKey systems.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// DRM system identifier for license requests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum DrmSystem {
    Widevine,
    PlayReady,
    FairPlay,
    ClearKey,
}

/// Client information included in license requests
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ClientInfo {
    pub device_id: String,
    pub platform: String,
    pub app_id: String,
    pub sdk_version: String,
}

impl ClientInfo {
    /// Create a new ClientInfo
    pub fn new(
        device_id: impl Into<String>,
        platform: impl Into<String>,
        app_id: impl Into<String>,
        sdk_version: impl Into<String>,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            platform: platform.into(),
            app_id: app_id.into(),
            sdk_version: sdk_version.into(),
        }
    }
}

/// A license request sent by a client to the license server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LicenseRequest {
    pub key_id: Vec<u8>,
    pub drm_system: DrmSystem,
    pub client_info: ClientInfo,
    pub token: Option<String>,
}

impl LicenseRequest {
    /// Create a new LicenseRequest
    pub fn new(key_id: Vec<u8>, drm_system: DrmSystem, client_info: ClientInfo) -> Self {
        Self {
            key_id,
            drm_system,
            client_info,
            token: None,
        }
    }

    /// Attach a bearer token
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

/// Track types that a license may grant access to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum TrackType {
    Video,
    Audio,
    Text,
    Thumbnail,
}

/// A license response issued by the server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LicenseResponse {
    pub key_id: Vec<u8>,
    pub content_key: Vec<u8>,
    pub expiry_secs: u64,
    pub allowed_tracks: Vec<TrackType>,
}

/// Output protection level required for playback
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[allow(dead_code)]
pub enum OutputProtectionLevel {
    None,
    Hdcp2_2,
    Hdcp2_3,
}

/// Policy that governs what a issued license permits
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct LicensePolicy {
    pub max_concurrent_streams: u32,
    pub allow_download: bool,
    pub output_protection: OutputProtectionLevel,
    pub analytics_required: bool,
}

impl Default for LicensePolicy {
    fn default() -> Self {
        Self {
            max_concurrent_streams: 3,
            allow_download: false,
            output_protection: OutputProtectionLevel::Hdcp2_2,
            analytics_required: false,
        }
    }
}

/// A cached license entry: (key_id, response, issued_at_secs)
type CacheEntry = (Vec<u8>, LicenseResponse, u64);

/// Simple LRU-style license cache capped at 1000 entries
pub struct LicenseCache {
    entries: Vec<CacheEntry>,
    max_entries: usize,
}

impl LicenseCache {
    /// Create a new cache with the given maximum size
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Insert a new license; evicts the oldest entry when at capacity
    pub fn insert(&mut self, key_id: Vec<u8>, response: LicenseResponse, issued_at: u64) {
        // Remove existing entry for the same key_id if present
        self.entries.retain(|(k, _, _)| k != &key_id);

        if self.entries.len() >= self.max_entries {
            self.entries.remove(0); // evict oldest
        }
        self.entries.push((key_id, response, issued_at));
    }

    /// Look up a cached response by key_id
    pub fn get(&self, key_id: &[u8]) -> Option<&LicenseResponse> {
        self.entries
            .iter()
            .find(|(k, _, _)| k.as_slice() == key_id)
            .map(|(_, r, _)| r)
    }

    /// Return current number of cached entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when no entries are cached
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// License server that issues content licenses based on a configurable policy
pub struct LicenseServer {
    cache: LicenseCache,
}

impl LicenseServer {
    /// Create a new license server with an empty 1000-entry cache
    pub fn new() -> Self {
        Self {
            cache: LicenseCache::new(1000),
        }
    }

    /// Issue a license for the given request and policy.
    ///
    /// The server derives a deterministic content key from the key_id and
    /// returns a response valid for `expiry_secs` seconds.
    pub fn issue_license(
        &mut self,
        request: &LicenseRequest,
        policy: &LicensePolicy,
    ) -> Result<LicenseResponse, String> {
        // Validate request
        if request.key_id.is_empty() {
            return Err("key_id must not be empty".to_string());
        }

        // Return cached license if available
        if let Some(cached) = self.cache.get(&request.key_id) {
            return Ok(cached.clone());
        }

        // Derive a deterministic 16-byte content key from the key_id
        let content_key = derive_content_key(&request.key_id);

        // Choose allowed tracks based on policy
        let mut allowed_tracks = vec![TrackType::Video, TrackType::Audio];
        if policy.allow_download {
            allowed_tracks.push(TrackType::Text);
            allowed_tracks.push(TrackType::Thumbnail);
        }

        let expiry_secs = match request.drm_system {
            DrmSystem::Widevine => 86_400,  // 24 hours
            DrmSystem::PlayReady => 43_200, // 12 hours
            DrmSystem::FairPlay => 86_400,  // 24 hours
            DrmSystem::ClearKey => 3_600,   // 1 hour (less secure)
        };

        let response = LicenseResponse {
            key_id: request.key_id.clone(),
            content_key,
            expiry_secs,
            allowed_tracks,
        };

        let now = current_time_secs();
        self.cache
            .insert(request.key_id.clone(), response.clone(), now);

        Ok(response)
    }

    /// Return a reference to the internal cache (for testing / inspection)
    pub fn cache(&self) -> &LicenseCache {
        &self.cache
    }
}

impl Default for LicenseServer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Derive a 16-byte content key from key_id using a simple XOR-hash approach.
fn derive_content_key(key_id: &[u8]) -> Vec<u8> {
    let mut key = [0u8; 16];
    for (i, &b) in key_id.iter().enumerate() {
        key[i % 16] ^= b.wrapping_add((i as u8).wrapping_mul(0x37));
    }
    // Ensure no zero bytes for realism
    for byte in key.iter_mut() {
        if *byte == 0 {
            *byte = 0xAA;
        }
    }
    key.to_vec()
}

/// Return the current Unix time in seconds, defaulting to 0 on error
fn current_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client() -> ClientInfo {
        ClientInfo::new("device-001", "android", "com.example.player", "1.0.0")
    }

    fn make_request(key_id: Vec<u8>) -> LicenseRequest {
        LicenseRequest::new(key_id, DrmSystem::Widevine, make_client())
    }

    #[test]
    fn test_issue_license_basic() {
        let mut server = LicenseServer::new();
        let req = make_request(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let policy = LicensePolicy::default();
        let resp = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert_eq!(resp.key_id, req.key_id);
        assert_eq!(resp.content_key.len(), 16);
        assert!(resp.expiry_secs > 0);
    }

    #[test]
    fn test_issue_license_clearkey_shorter_expiry() {
        let mut server = LicenseServer::new();
        let mut req = make_request(vec![1; 16]);
        req.drm_system = DrmSystem::ClearKey;
        let policy = LicensePolicy::default();
        let resp = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert_eq!(resp.expiry_secs, 3_600);
    }

    #[test]
    fn test_issue_license_playready_expiry() {
        let mut server = LicenseServer::new();
        let mut req = make_request(vec![2; 16]);
        req.drm_system = DrmSystem::PlayReady;
        let policy = LicensePolicy::default();
        let resp = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert_eq!(resp.expiry_secs, 43_200);
    }

    #[test]
    fn test_cache_hit_returns_same_response() {
        let mut server = LicenseServer::new();
        let req = make_request(vec![3; 16]);
        let policy = LicensePolicy::default();
        let r1 = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        let r2 = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert_eq!(r1.content_key, r2.content_key);
    }

    #[test]
    fn test_different_key_ids_different_keys() {
        let mut server = LicenseServer::new();
        let policy = LicensePolicy::default();
        let r1 = server
            .issue_license(&make_request(vec![1; 16]), &policy)
            .expect("license operation should succeed");
        let r2 = server
            .issue_license(&make_request(vec![2; 16]), &policy)
            .expect("license operation should succeed");
        assert_ne!(r1.content_key, r2.content_key);
    }

    #[test]
    fn test_empty_key_id_returns_error() {
        let mut server = LicenseServer::new();
        let req = make_request(vec![]);
        let policy = LicensePolicy::default();
        assert!(server.issue_license(&req, &policy).is_err());
    }

    #[test]
    fn test_allow_download_adds_tracks() {
        let mut server = LicenseServer::new();
        let req = make_request(vec![5; 16]);
        let policy = LicensePolicy {
            allow_download: true,
            ..LicensePolicy::default()
        };
        let resp = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert!(resp.allowed_tracks.contains(&TrackType::Text));
        assert!(resp.allowed_tracks.contains(&TrackType::Thumbnail));
    }

    #[test]
    fn test_no_download_restricts_tracks() {
        let mut server = LicenseServer::new();
        let req = make_request(vec![6; 16]);
        let policy = LicensePolicy {
            allow_download: false,
            ..LicensePolicy::default()
        };
        let resp = server
            .issue_license(&req, &policy)
            .expect("license operation should succeed");
        assert!(!resp.allowed_tracks.contains(&TrackType::Thumbnail));
    }

    #[test]
    fn test_license_cache_eviction() {
        let mut cache = LicenseCache::new(3);
        let dummy_resp = LicenseResponse {
            key_id: vec![0],
            content_key: vec![0u8; 16],
            expiry_secs: 100,
            allowed_tracks: vec![],
        };
        for i in 0u8..4 {
            cache.insert(vec![i], dummy_resp.clone(), 1000 + i as u64);
        }
        // The first entry (key_id=[0]) should have been evicted
        assert_eq!(cache.len(), 3);
        assert!(cache.get(&[0]).is_none());
        assert!(cache.get(&[3]).is_some());
    }

    #[test]
    fn test_license_cache_update_existing() {
        let mut cache = LicenseCache::new(10);
        let r1 = LicenseResponse {
            key_id: vec![1],
            content_key: vec![1u8; 16],
            expiry_secs: 100,
            allowed_tracks: vec![],
        };
        let r2 = LicenseResponse {
            key_id: vec![1],
            content_key: vec![2u8; 16],
            expiry_secs: 200,
            allowed_tracks: vec![],
        };
        cache.insert(vec![1], r1, 1000);
        cache.insert(vec![1], r2, 2000);
        assert_eq!(cache.len(), 1);
        assert_eq!(
            cache
                .get(&[1])
                .expect("cache entry should exist")
                .expiry_secs,
            200
        );
    }

    #[test]
    fn test_with_token() {
        let req = make_request(vec![7; 16]).with_token("bearer-abc");
        assert_eq!(req.token.as_deref(), Some("bearer-abc"));
    }

    #[test]
    fn test_output_protection_ordering() {
        assert!(OutputProtectionLevel::None < OutputProtectionLevel::Hdcp2_2);
        assert!(OutputProtectionLevel::Hdcp2_2 < OutputProtectionLevel::Hdcp2_3);
    }
}
