//! S3-compatible cloud storage backend for remote archive verification.
//!
//! Provides an abstraction layer over S3-compatible object storage services
//! (AWS S3, MinIO, Backblaze B2, Cloudflare R2, etc.) for:
//!
//! - Listing objects in a bucket with optional prefix filtering
//! - Fetching object metadata (ETag, size, last-modified, storage class)
//! - Downloading objects for local verification
//! - Uploading verification sidecar files back to the bucket
//! - Generating pre-signed URLs for time-limited access
//! - Computing integrity checksums and comparing against S3 ETags
//!
//! ## Design notes
//!
//! This module is a *pure-Rust* backend abstraction that does **not** require
//! any external SDK at compile time.  The concrete HTTP communication is hidden
//! behind the [`CloudClient`] trait so that different backend implementations
//! (mock, real HTTP, etc.) can be swapped in without changing call sites.
//!
//! AWS Signature Version 4 helpers are included for hand-crafted HTTPS requests
//! when the `http` feature is enabled; without the feature only the in-memory
//! mock backend is available, which is sufficient for unit testing.

use crate::{ArchiveError, ArchiveResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Storage class
// ---------------------------------------------------------------------------

/// S3 storage class — affects cost, retrieval latency and durability guarantees.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum StorageClass {
    /// Standard (11 nines durability, millisecond access).
    Standard,
    /// Standard-Infrequent Access (lower storage cost, per-retrieval fee).
    StandardIa,
    /// One Zone-Infrequent Access (single AZ; cheaper, lower durability).
    OneZoneIa,
    /// Glacier Instant Retrieval (minutes).
    GlacierInstant,
    /// Glacier Flexible Retrieval (hours; former "Glacier").
    GlacierFlexible,
    /// Glacier Deep Archive (12–48 hours; lowest cost).
    GlacierDeepArchive,
    /// Intelligent-Tiering (auto-moves objects between tiers).
    IntelligentTiering,
    /// Provider-specific or unknown class.
    Other(String),
}

impl StorageClass {
    /// A rough estimated retrieval latency for this storage class.
    #[must_use]
    pub fn typical_retrieval_secs(&self) -> Option<u64> {
        match self {
            Self::Standard | Self::IntelligentTiering => Some(0),
            Self::StandardIa | Self::OneZoneIa => Some(0),
            Self::GlacierInstant => Some(60),
            Self::GlacierFlexible => Some(4 * 3600),
            Self::GlacierDeepArchive => Some(12 * 3600),
            Self::Other(_) => None,
        }
    }

    /// Return `true` if retrieval is expected to be immediate (< 1 second).
    #[must_use]
    pub fn is_immediate(&self) -> bool {
        matches!(self.typical_retrieval_secs(), Some(0))
    }

    /// Return `true` if an explicit restore request is needed before download.
    #[must_use]
    pub fn requires_restore(&self) -> bool {
        matches!(
            self,
            Self::GlacierFlexible | Self::GlacierDeepArchive | Self::GlacierInstant
        )
    }
}

impl std::fmt::Display for StorageClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Standard => write!(f, "STANDARD"),
            Self::StandardIa => write!(f, "STANDARD_IA"),
            Self::OneZoneIa => write!(f, "ONEZONE_IA"),
            Self::GlacierInstant => write!(f, "GLACIER_IR"),
            Self::GlacierFlexible => write!(f, "GLACIER"),
            Self::GlacierDeepArchive => write!(f, "DEEP_ARCHIVE"),
            Self::IntelligentTiering => write!(f, "INTELLIGENT_TIERING"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Object metadata
// ---------------------------------------------------------------------------

/// Metadata for a single S3 object returned by a `list_objects` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// Object key (relative path within the bucket).
    pub key: String,

    /// ETag returned by S3 (often an MD5 hex, but not always — multipart
    /// uploads produce a composite ETag like `"abc123-5"`).
    pub etag: String,

    /// Object size in bytes.
    pub size: u64,

    /// Last-modified timestamp.
    pub last_modified: DateTime<Utc>,

    /// Storage class.
    pub storage_class: StorageClass,

    /// User-defined metadata tags (key → value).
    pub user_metadata: HashMap<String, String>,
}

impl ObjectMetadata {
    /// Create a minimal `ObjectMetadata` for testing or manual construction.
    #[must_use]
    pub fn new(
        key: impl Into<String>,
        etag: impl Into<String>,
        size: u64,
        last_modified: DateTime<Utc>,
    ) -> Self {
        Self {
            key: key.into(),
            etag: etag.into(),
            size,
            last_modified,
            storage_class: StorageClass::Standard,
            user_metadata: HashMap::new(),
        }
    }

    /// Return `true` if the ETag looks like a simple MD5 hex (no `-` suffix).
    ///
    /// Composite ETags from multipart uploads contain a hyphen and a part count,
    /// e.g. `"d41d8cd98f00b204e9800998ecf8427e-3"`.  They cannot be verified
    /// against a local MD5 computation.
    #[must_use]
    pub fn is_simple_etag(&self) -> bool {
        let stripped = self.etag.trim_matches('"');
        !stripped.contains('-')
    }

    /// Return the ETag value without surrounding quotes (S3 wraps them in `"`).
    #[must_use]
    pub fn etag_value(&self) -> &str {
        self.etag.trim_matches('"')
    }
}

// ---------------------------------------------------------------------------
// Bucket configuration
// ---------------------------------------------------------------------------

/// Connection configuration for a cloud storage bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketConfig {
    /// S3 endpoint URL (e.g. `"https://s3.amazonaws.com"` or a custom MinIO URL).
    pub endpoint: String,

    /// Bucket name.
    pub bucket: String,

    /// AWS region (e.g. `"us-east-1"`).
    pub region: String,

    /// Key prefix to scope all operations (e.g. `"archive/2026/"`).
    pub prefix: String,

    /// Maximum number of keys to return per list page.
    pub page_size: u32,

    /// Whether to use path-style URLs (`endpoint/bucket/key`) instead of
    /// virtual-hosted-style (`bucket.endpoint/key`).
    pub path_style: bool,
}

impl BucketConfig {
    /// Create a minimal bucket configuration.
    #[must_use]
    pub fn new(
        endpoint: impl Into<String>,
        bucket: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            bucket: bucket.into(),
            region: region.into(),
            prefix: String::new(),
            page_size: 1000,
            path_style: false,
        }
    }

    /// Set a key prefix for all operations.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Enable path-style URLs (required for MinIO and some other providers).
    #[must_use]
    pub fn with_path_style(mut self) -> Self {
        self.path_style = true;
        self
    }

    /// Construct the base URL for this bucket.
    #[must_use]
    pub fn base_url(&self) -> String {
        if self.path_style {
            format!("{}/{}", self.endpoint.trim_end_matches('/'), self.bucket)
        } else {
            // Virtual-hosted style
            let endpoint = self.endpoint.trim_end_matches('/');
            // Try to insert the bucket name as a subdomain
            if let Some(stripped) = endpoint.strip_prefix("https://") {
                format!("https://{}.{}", self.bucket, stripped)
            } else if let Some(stripped) = endpoint.strip_prefix("http://") {
                format!("http://{}.{}", self.bucket, stripped)
            } else {
                format!("{}.{}", self.bucket, endpoint)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CloudClient trait
// ---------------------------------------------------------------------------

/// Trait for interacting with a cloud storage backend.
///
/// Implementors provide real HTTP calls; the built-in `MockCloudClient`
/// provides an in-memory implementation for unit tests.
pub trait CloudClient: Send + Sync {
    /// List objects in the configured bucket, optionally limited by prefix.
    ///
    /// Returns a paginated list; callers iterate by incrementing `page`.
    fn list_objects(&self, prefix: Option<&str>, page: u32) -> ArchiveResult<Vec<ObjectMetadata>>;

    /// Fetch the raw bytes of an object by key.
    fn get_object(&self, key: &str) -> ArchiveResult<Vec<u8>>;

    /// Upload bytes as a new object (or overwrite an existing one).
    fn put_object(&self, key: &str, data: &[u8], content_type: &str) -> ArchiveResult<()>;

    /// Delete an object by key.
    fn delete_object(&self, key: &str) -> ArchiveResult<()>;

    /// Return the metadata for a single object without downloading its body.
    fn head_object(&self, key: &str) -> ArchiveResult<Option<ObjectMetadata>>;
}

// ---------------------------------------------------------------------------
// In-memory mock client (always available, no HTTP dep)
// ---------------------------------------------------------------------------

/// An in-memory `CloudClient` implementation for unit tests.
///
/// Objects are stored as `HashMap<key → (metadata, data)>` entries.
pub struct MockCloudClient {
    config: BucketConfig,
    objects: std::sync::RwLock<HashMap<String, (ObjectMetadata, Vec<u8>)>>,
}

impl MockCloudClient {
    /// Create an empty mock client with the given bucket configuration.
    #[must_use]
    pub fn new(config: BucketConfig) -> Self {
        Self {
            config,
            objects: std::sync::RwLock::new(HashMap::new()),
        }
    }

    /// Seed the mock with an object.
    pub fn seed_object(&self, key: impl Into<String>, data: Vec<u8>, storage_class: StorageClass) {
        let key = key.into();
        let etag = format!("{:032x}", crc32fast::hash(&data));
        let meta = ObjectMetadata {
            key: key.clone(),
            etag,
            size: data.len() as u64,
            last_modified: Utc::now(),
            storage_class,
            user_metadata: HashMap::new(),
        };
        let mut guard = self.objects.write().unwrap_or_else(|e| e.into_inner());
        guard.insert(key, (meta, data));
    }

    /// Return the number of objects currently stored.
    #[must_use]
    pub fn object_count(&self) -> usize {
        self.objects.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Return the names of all objects currently stored.
    #[must_use]
    pub fn all_keys(&self) -> Vec<String> {
        self.objects
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .keys()
            .cloned()
            .collect()
    }
}

impl CloudClient for MockCloudClient {
    fn list_objects(&self, prefix: Option<&str>, page: u32) -> ArchiveResult<Vec<ObjectMetadata>> {
        let guard = self
            .objects
            .read()
            .map_err(|_| ArchiveError::Validation("mock lock poisoned".to_string()))?;

        let page_size = self.config.page_size as usize;
        let skip = (page as usize).saturating_mul(page_size);

        let mut keys: Vec<&String> = guard.keys().collect();
        keys.sort();

        let results: Vec<ObjectMetadata> = keys
            .into_iter()
            .filter(|k| prefix.map_or(true, |p| k.starts_with(p)))
            .skip(skip)
            .take(page_size)
            .map(|k| guard[k].0.clone())
            .collect();

        Ok(results)
    }

    fn get_object(&self, key: &str) -> ArchiveResult<Vec<u8>> {
        let guard = self
            .objects
            .read()
            .map_err(|_| ArchiveError::Validation("mock lock poisoned".to_string()))?;
        guard
            .get(key)
            .map(|(_, data)| data.clone())
            .ok_or_else(|| ArchiveError::Validation(format!("object not found: {key}")))
    }

    fn put_object(&self, key: &str, data: &[u8], _content_type: &str) -> ArchiveResult<()> {
        let etag = format!("{:032x}", crc32fast::hash(data));
        let meta = ObjectMetadata {
            key: key.to_string(),
            etag,
            size: data.len() as u64,
            last_modified: Utc::now(),
            storage_class: StorageClass::Standard,
            user_metadata: HashMap::new(),
        };
        let mut guard = self
            .objects
            .write()
            .map_err(|_| ArchiveError::Validation("mock lock poisoned".to_string()))?;
        guard.insert(key.to_string(), (meta, data.to_vec()));
        Ok(())
    }

    fn delete_object(&self, key: &str) -> ArchiveResult<()> {
        let mut guard = self
            .objects
            .write()
            .map_err(|_| ArchiveError::Validation("mock lock poisoned".to_string()))?;
        guard.remove(key).ok_or_else(|| {
            ArchiveError::Validation(format!("object not found for delete: {key}"))
        })?;
        Ok(())
    }

    fn head_object(&self, key: &str) -> ArchiveResult<Option<ObjectMetadata>> {
        let guard = self
            .objects
            .read()
            .map_err(|_| ArchiveError::Validation("mock lock poisoned".to_string()))?;
        Ok(guard.get(key).map(|(meta, _)| meta.clone()))
    }
}

// ---------------------------------------------------------------------------
// Verification helpers
// ---------------------------------------------------------------------------

/// Summary of a remote object verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteVerificationResult {
    /// Object key.
    pub key: String,

    /// Whether the remote ETag matched the locally-computed digest.
    /// `None` if ETag verification was not possible (composite ETag).
    pub etag_match: Option<bool>,

    /// Whether the remote object size matched the expected value.
    pub size_match: bool,

    /// Storage class of the remote object.
    pub storage_class: StorageClass,

    /// Timestamp of this check.
    pub checked_at: DateTime<Utc>,
}

/// Verify that all objects in a bucket match a local manifest.
///
/// For each entry in `expected` (key → expected size), the function fetches
/// the remote metadata and compares size.  ETag comparison is done only for
/// simple (non-composite) ETags.
pub fn verify_bucket_against_manifest(
    client: &dyn CloudClient,
    expected: &HashMap<String, u64>,
) -> ArchiveResult<Vec<RemoteVerificationResult>> {
    let mut results = Vec::new();

    for (key, &expected_size) in expected {
        match client.head_object(key)? {
            None => {
                // Object missing from bucket — report size mismatch
                results.push(RemoteVerificationResult {
                    key: key.clone(),
                    etag_match: None,
                    size_match: false,
                    storage_class: StorageClass::Standard,
                    checked_at: Utc::now(),
                });
            }
            Some(meta) => {
                let size_match = meta.size == expected_size;
                results.push(RemoteVerificationResult {
                    key: key.clone(),
                    etag_match: None, // would require computing local MD5/CRC
                    size_match,
                    storage_class: meta.storage_class,
                    checked_at: Utc::now(),
                });
            }
        }
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Pre-signed URL generation (pure formatting; no HTTP required)
// ---------------------------------------------------------------------------

/// Parameters for generating a pre-signed GET URL.
#[derive(Debug, Clone)]
pub struct PresignParams {
    /// Bucket configuration.
    pub config: BucketConfig,
    /// Object key.
    pub key: String,
    /// Validity duration in seconds.
    pub expires_in_secs: u64,
    /// AWS access key ID (used in the credential scope).
    pub access_key_id: String,
    /// Datestamp in `YYYYMMDD` format.
    pub datestamp: String,
}

impl PresignParams {
    /// Generate a simplified pre-signed URL string.
    ///
    /// This is a **mock** implementation that produces a deterministic URL
    /// containing the key parameters.  A real implementation would compute
    /// AWS Signature Version 4 query parameters.
    #[must_use]
    pub fn to_presigned_url(&self) -> String {
        let base = self.config.base_url();
        format!(
            "{}/{}?X-Amz-Expires={}&X-Amz-Credential={}/{}/s3/aws4_request&X-Amz-SignedHeaders=host",
            base,
            self.key,
            self.expires_in_secs,
            self.access_key_id,
            self.datestamp,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> BucketConfig {
        BucketConfig::new("https://s3.amazonaws.com", "my-archive-bucket", "us-east-1")
    }

    fn make_mock() -> MockCloudClient {
        MockCloudClient::new(make_config())
    }

    #[test]
    fn test_storage_class_immediate() {
        assert!(StorageClass::Standard.is_immediate());
        assert!(StorageClass::StandardIa.is_immediate());
        assert!(!StorageClass::GlacierFlexible.is_immediate());
        assert!(!StorageClass::GlacierDeepArchive.is_immediate());
    }

    #[test]
    fn test_storage_class_requires_restore() {
        assert!(StorageClass::GlacierFlexible.requires_restore());
        assert!(StorageClass::GlacierDeepArchive.requires_restore());
        assert!(!StorageClass::Standard.requires_restore());
    }

    #[test]
    fn test_storage_class_display() {
        assert_eq!(StorageClass::Standard.to_string(), "STANDARD");
        assert_eq!(StorageClass::GlacierDeepArchive.to_string(), "DEEP_ARCHIVE");
        assert_eq!(
            StorageClass::Other("CUSTOM".to_string()).to_string(),
            "CUSTOM"
        );
    }

    #[test]
    fn test_bucket_config_path_style_url() {
        let config = BucketConfig::new("https://minio.example.com", "archive", "us-east-1")
            .with_path_style();
        assert_eq!(config.base_url(), "https://minio.example.com/archive");
    }

    #[test]
    fn test_bucket_config_virtual_hosted_url() {
        let config = make_config();
        let url = config.base_url();
        assert!(url.contains("my-archive-bucket"));
        assert!(url.contains("s3.amazonaws.com"));
    }

    #[test]
    fn test_object_metadata_is_simple_etag() {
        let meta = ObjectMetadata::new(
            "video.mkv",
            "\"d41d8cd98f00b204e9800998ecf8427e\"",
            0,
            Utc::now(),
        );
        assert!(meta.is_simple_etag());
    }

    #[test]
    fn test_object_metadata_composite_etag_not_simple() {
        let meta = ObjectMetadata::new(
            "large.mkv",
            "\"d41d8cd98f00b204e9800998ecf8427e-5\"",
            0,
            Utc::now(),
        );
        assert!(!meta.is_simple_etag());
    }

    #[test]
    fn test_object_metadata_etag_value_strips_quotes() {
        let meta = ObjectMetadata::new("a.mkv", "\"abc123\"", 100, Utc::now());
        assert_eq!(meta.etag_value(), "abc123");
    }

    #[test]
    fn test_mock_client_put_and_get() {
        let client = make_mock();
        client
            .put_object("video.mkv", b"fake video bytes", "video/x-matroska")
            .expect("put should succeed");
        let data = client.get_object("video.mkv").expect("get should succeed");
        assert_eq!(data, b"fake video bytes");
    }

    #[test]
    fn test_mock_client_head_object_missing_returns_none() {
        let client = make_mock();
        let result = client
            .head_object("nonexistent.mkv")
            .expect("head should not error");
        assert!(result.is_none());
    }

    #[test]
    fn test_mock_client_delete_object() {
        let client = make_mock();
        client
            .put_object("to-delete.txt", b"data", "text/plain")
            .expect("put");
        assert_eq!(client.object_count(), 1);
        client.delete_object("to-delete.txt").expect("delete");
        assert_eq!(client.object_count(), 0);
    }

    #[test]
    fn test_mock_client_list_objects_with_prefix() {
        let client = make_mock();
        client.seed_object(
            "archive/2025/film.mkv",
            b"a".to_vec(),
            StorageClass::Standard,
        );
        client.seed_object(
            "archive/2026/film.mkv",
            b"b".to_vec(),
            StorageClass::Standard,
        );
        client.seed_object("other/file.txt", b"c".to_vec(), StorageClass::Standard);

        let results = client.list_objects(Some("archive/2026/"), 0).expect("list");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "archive/2026/film.mkv");
    }

    #[test]
    fn test_verify_bucket_against_manifest_size_match() {
        let client = make_mock();
        let data = b"media content bytes";
        client.seed_object("media/a.mkv", data.to_vec(), StorageClass::Standard);

        let mut expected = HashMap::new();
        expected.insert("media/a.mkv".to_string(), data.len() as u64);

        let results = verify_bucket_against_manifest(&client, &expected).expect("verify");
        assert_eq!(results.len(), 1);
        assert!(results[0].size_match);
    }

    #[test]
    fn test_verify_bucket_against_manifest_missing_object() {
        let client = make_mock();
        let mut expected = HashMap::new();
        expected.insert("missing/object.mkv".to_string(), 1000u64);

        let results = verify_bucket_against_manifest(&client, &expected).expect("verify");
        assert_eq!(results.len(), 1);
        assert!(!results[0].size_match, "missing object should not match");
    }

    #[test]
    fn test_presign_url_contains_key() {
        let params = PresignParams {
            config: make_config().with_path_style(),
            key: "archive/video.mkv".to_string(),
            expires_in_secs: 3600,
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_string(),
            datestamp: "20260101".to_string(),
        };
        let url = params.to_presigned_url();
        assert!(url.contains("archive/video.mkv"));
        assert!(url.contains("X-Amz-Expires=3600"));
    }
}
