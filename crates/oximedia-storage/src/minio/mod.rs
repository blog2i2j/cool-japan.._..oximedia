#![allow(dead_code)]
//! MinIO backend — S3-compatible self-hosted object storage.
//!
//! MinIO is wire-compatible with the AWS S3 API; this module provides a
//! thin wrapper that routes calls through the same AWS SDK client but
//! pointed at a custom endpoint (your MinIO server).
//!
//! When the `minio` feature is enabled (which implies `s3`), a real
//! AWS SDK client configured with a custom endpoint is used.  When the
//! feature is disabled the module falls back to a pure in-memory stub
//! that satisfies the `CloudStorage` trait without any network I/O.
//!
//! # Feature gates
//!
//! | Feature | Implementation |
//! |---------|---------------|
//! | *(none)* | In-memory stub |
//! | `minio` | Real MinIO via aws-sdk-s3 + custom endpoint |

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ListOptions, ListResult, ObjectMetadata, Result,
    StorageError, UploadOptions,
};
use async_trait::async_trait;
use std::path::Path;

// These imports are only needed by the in-memory stub
#[cfg(not(feature = "minio"))]
use bytes::Bytes;
#[cfg(not(feature = "minio"))]
use chrono::{DateTime, Utc};
#[cfg(not(feature = "minio"))]
use sha2::Digest as _;
#[cfg(not(feature = "minio"))]
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MinioConfig — common to both implementations
// ---------------------------------------------------------------------------

/// Configuration for the MinIO storage backend.
#[derive(Debug, Clone)]
pub struct MinioConfig {
    /// MinIO server endpoint, e.g. `"http://localhost:9000"`.
    pub endpoint: String,
    /// Access key (MinIO username).
    pub access_key: String,
    /// Secret key (MinIO password).
    pub secret_key: String,
    /// Bucket name to operate on.
    pub bucket: String,
    /// Whether to use TLS (`https://`) when connecting.
    pub use_ssl: bool,
    /// S3 region string (MinIO accepts any non-empty region, e.g. `"us-east-1"`).
    pub region: String,
    /// Enable path-style addressing (`endpoint/bucket/key` instead of `bucket.endpoint/key`).
    /// MinIO typically requires path-style.
    pub path_style: bool,
}

impl MinioConfig {
    /// Create a configuration for a local MinIO instance with default settings.
    pub fn local(
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: "http://localhost:9000".to_string(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            bucket: bucket.into(),
            use_ssl: false,
            region: "us-east-1".to_string(),
            path_style: true,
        }
    }

    /// Create a configuration for a remote MinIO instance with TLS.
    pub fn remote(
        endpoint: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            bucket: bucket.into(),
            use_ssl: true,
            region: "us-east-1".to_string(),
            path_style: true,
        }
    }

    /// Validate the configuration fields.
    pub fn validate(&self) -> Result<()> {
        if self.endpoint.is_empty() {
            return Err(StorageError::InvalidConfig(
                "MinIO endpoint must not be empty".into(),
            ));
        }
        if self.access_key.is_empty() {
            return Err(StorageError::InvalidConfig(
                "MinIO access_key must not be empty".into(),
            ));
        }
        if self.secret_key.is_empty() {
            return Err(StorageError::InvalidConfig(
                "MinIO secret_key must not be empty".into(),
            ));
        }
        if self.bucket.is_empty() {
            return Err(StorageError::InvalidConfig(
                "MinIO bucket must not be empty".into(),
            ));
        }
        Ok(())
    }

    /// Return the effective base URL for S3-API requests.
    ///
    /// If `use_ssl` is true and the endpoint starts with `http://`, the scheme
    /// is upgraded to `https://`.
    pub fn effective_endpoint(&self) -> String {
        if self.use_ssl && self.endpoint.starts_with("http://") {
            self.endpoint.replacen("http://", "https://", 1)
        } else {
            self.endpoint.clone()
        }
    }

    /// Convert to a [`crate::UnifiedConfig`] suitable for [`crate::s3::S3Storage`].
    #[cfg(feature = "minio")]
    pub(crate) fn to_unified_config(&self) -> crate::UnifiedConfig {
        crate::UnifiedConfig {
            provider: crate::StorageProvider::S3,
            bucket: self.bucket.clone(),
            region: Some(self.region.clone()),
            endpoint: Some(self.effective_endpoint()),
            access_key: Some(self.access_key.clone()),
            secret_key: Some(self.secret_key.clone()),
            project_id: None,
            credentials_file: None,
            transfer_acceleration: false,
            path_style: self.path_style,
            max_connections: 10,
            timeout_seconds: 300,
            enable_cache: false,
            cache_dir: None,
            max_cache_size: 10 * 1024 * 1024 * 1024,
            retry: crate::RetryConfig::default(),
            pool_config: crate::ConnectionPoolConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Real MinIO implementation (feature = "minio")
// ---------------------------------------------------------------------------

/// MinIO storage backend backed by a real S3-compatible client.
///
/// Internally wraps [`crate::s3::S3Storage`] with the MinIO endpoint injected
/// into the unified config so that all S3 API calls are routed to MinIO.
#[cfg(feature = "minio")]
pub struct MinioStorage {
    inner: crate::s3::S3Storage,
    config: MinioConfig,
}

#[cfg(feature = "minio")]
impl MinioStorage {
    /// Create a new `MinioStorage` from config.
    ///
    /// This builds an AWS SDK client pointed at the MinIO endpoint.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::InvalidConfig` if the configuration is invalid,
    /// or a provider error if the AWS SDK cannot be initialised.
    pub async fn new(config: MinioConfig) -> Result<Self> {
        config.validate()?;
        let unified = config.to_unified_config();
        let inner = crate::s3::S3Storage::new(unified).await?;
        Ok(Self { inner, config })
    }

    /// Return the bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Return the effective endpoint URL.
    pub fn endpoint(&self) -> String {
        self.config.effective_endpoint()
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &MinioConfig {
        &self.config
    }
}

#[cfg(feature = "minio")]
#[async_trait]
impl CloudStorage for MinioStorage {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        self.inner.upload_stream(key, stream, size, options).await
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        self.inner.upload_file(key, file_path, options).await
    }

    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream> {
        self.inner.download_stream(key, options).await
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
    ) -> Result<()> {
        self.inner.download_file(key, file_path, options).await
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        self.inner.get_metadata(key).await
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        self.inner.delete_object(key).await
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        self.inner.delete_objects(keys).await
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        self.inner.list_objects(options).await
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        self.inner.object_exists(key).await
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        self.inner.copy_object(source_key, dest_key).await
    }

    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String> {
        self.inner
            .generate_presigned_url(key, expiration_secs)
            .await
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        self.inner
            .generate_presigned_upload_url(key, expiration_secs)
            .await
    }
}

// ---------------------------------------------------------------------------
// In-memory stub (no features / when minio feature is not enabled)
// ---------------------------------------------------------------------------

/// In-memory object store entry.
#[cfg(not(feature = "minio"))]
#[derive(Debug, Clone)]
struct StoreEntry {
    data: Vec<u8>,
    content_type: Option<String>,
    metadata: HashMap<String, String>,
    last_modified: DateTime<Utc>,
}

/// MinIO storage backend.
///
/// When the `minio` feature is **not** enabled, this is an in-memory stub
/// useful for testing without a real MinIO server.  When `minio` is enabled,
/// the real implementation delegates to `S3Storage` (requires the `s3` feature).
#[cfg(not(feature = "minio"))]
pub struct MinioStorage {
    config: MinioConfig,
    /// In-memory object store for the stub implementation.
    objects: std::sync::Mutex<HashMap<String, StoreEntry>>,
}

#[cfg(not(feature = "minio"))]
impl MinioStorage {
    /// Create a new `MinioStorage` from config.
    ///
    /// # Errors
    ///
    /// Returns `StorageError::InvalidConfig` if the configuration is invalid.
    pub fn new(config: MinioConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            objects: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Return the bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Return the effective endpoint URL.
    pub fn endpoint(&self) -> String {
        self.config.effective_endpoint()
    }

    /// Return a reference to the configuration.
    pub fn config(&self) -> &MinioConfig {
        &self.config
    }

    fn lock_objects(&self) -> std::sync::MutexGuard<'_, HashMap<String, StoreEntry>> {
        self.objects.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[cfg(not(feature = "minio"))]
#[async_trait]
impl CloudStorage for MinioStorage {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        _size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        use futures::StreamExt;
        let mut stream = stream;
        let mut data = Vec::new();
        while let Some(chunk) = stream.next().await {
            data.extend_from_slice(&chunk?);
        }
        let etag = hex::encode(sha2::Sha256::digest(&data));
        let entry = StoreEntry {
            data,
            content_type: options.content_type,
            metadata: options.metadata,
            last_modified: Utc::now(),
        };
        self.lock_objects().insert(key.to_string(), entry);
        Ok(etag)
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        let data = tokio::fs::read(file_path).await?;
        let etag = hex::encode(sha2::Sha256::digest(&data));
        let entry = StoreEntry {
            data,
            content_type: options.content_type,
            metadata: options.metadata,
            last_modified: Utc::now(),
        };
        self.lock_objects().insert(key.to_string(), entry);
        Ok(etag)
    }

    async fn download_stream(&self, key: &str, _options: DownloadOptions) -> Result<ByteStream> {
        use futures::stream;
        let data = {
            let guard = self.lock_objects();
            guard
                .get(key)
                .map(|e| e.data.clone())
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };
        Ok(Box::pin(stream::once(async move { Ok(Bytes::from(data)) })))
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        _options: DownloadOptions,
    ) -> Result<()> {
        let data = {
            let guard = self.lock_objects();
            guard
                .get(key)
                .map(|e| e.data.clone())
                .ok_or_else(|| StorageError::NotFound(key.to_string()))?
        };
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(file_path, &data).await?;
        Ok(())
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        let guard = self.lock_objects();
        let entry = guard
            .get(key)
            .ok_or_else(|| StorageError::NotFound(key.to_string()))?;
        Ok(ObjectMetadata {
            key: key.to_string(),
            size: entry.data.len() as u64,
            content_type: entry.content_type.clone(),
            last_modified: entry.last_modified,
            etag: Some(hex::encode(sha2::Sha256::digest(&entry.data))),
            metadata: entry.metadata.clone(),
            storage_class: None,
        })
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        self.lock_objects().remove(key);
        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        let mut results = Vec::new();
        for key in keys {
            self.lock_objects().remove(key.as_str());
            results.push(Ok(()));
        }
        Ok(results)
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        let guard = self.lock_objects();
        let prefix = options.prefix.as_deref().unwrap_or("");
        let max = options.max_results.unwrap_or(usize::MAX);

        let mut objects: Vec<ObjectMetadata> = guard
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .take(max)
            .map(|(k, e)| ObjectMetadata {
                key: k.clone(),
                size: e.data.len() as u64,
                content_type: e.content_type.clone(),
                last_modified: e.last_modified,
                etag: None,
                metadata: e.metadata.clone(),
                storage_class: None,
            })
            .collect();

        // Sort for determinism
        objects.sort_by(|a, b| a.key.cmp(&b.key));

        let total = guard.keys().filter(|k| k.starts_with(prefix)).count();
        let has_more = total > max;

        Ok(ListResult {
            objects,
            prefixes: Vec::new(),
            next_token: None,
            has_more,
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        Ok(self.lock_objects().contains_key(key))
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        let entry = {
            let guard = self.lock_objects();
            guard
                .get(source_key)
                .cloned()
                .ok_or_else(|| StorageError::NotFound(source_key.to_string()))?
        };
        self.lock_objects().insert(dest_key.to_string(), entry);
        Ok(())
    }

    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String> {
        Ok(format!(
            "{}/presigned/{}/{}?expiry={}",
            self.config.effective_endpoint(),
            self.config.bucket,
            key,
            expiration_secs
        ))
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        Ok(format!(
            "{}/presigned-upload/{}/{}?expiry={}",
            self.config.effective_endpoint(),
            self.config.bucket,
            key,
            expiration_secs
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Config tests (always run) ──────────────────────────────────────────────

    #[test]
    fn test_config_local_defaults() {
        let cfg = MinioConfig::local("key", "secret", "bucket");
        assert_eq!(cfg.endpoint, "http://localhost:9000");
        assert!(!cfg.use_ssl);
        assert!(cfg.path_style);
    }

    #[test]
    fn test_config_remote_uses_ssl() {
        let cfg = MinioConfig::remote("http://minio.example.com", "k", "s", "bkt");
        assert!(cfg.use_ssl);
        assert_eq!(cfg.effective_endpoint(), "https://minio.example.com");
    }

    #[test]
    fn test_config_ssl_upgrade() {
        let cfg = MinioConfig {
            endpoint: "http://minio.internal:9000".to_string(),
            use_ssl: true,
            ..MinioConfig::local("k", "s", "b")
        };
        assert!(cfg.effective_endpoint().starts_with("https://"));
    }

    #[test]
    fn test_config_validate_empty_bucket_fails() {
        let cfg = MinioConfig {
            bucket: String::new(),
            ..MinioConfig::local("k", "s", "b")
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_empty_access_key_fails() {
        let cfg = MinioConfig::local("", "s", "b");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_empty_secret_fails() {
        let cfg = MinioConfig::local("k", "", "b");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_empty_endpoint_fails() {
        let cfg = MinioConfig {
            endpoint: String::new(),
            ..MinioConfig::local("k", "s", "b")
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_no_ssl_upgrade_when_already_https() {
        let cfg = MinioConfig {
            endpoint: "https://secure.minio.example.com".to_string(),
            use_ssl: true,
            ..MinioConfig::local("k", "s", "b")
        };
        assert_eq!(cfg.effective_endpoint(), "https://secure.minio.example.com");
    }

    // ── Tests for the real minio feature ─────────────────────────────────────

    /// Test that `MinioConfig::to_unified_config()` maps fields correctly.
    #[cfg(feature = "minio")]
    #[test]
    fn test_to_unified_config_maps_fields() {
        let cfg = MinioConfig::local("admin", "password", "test-bucket");
        let unified = cfg.to_unified_config();
        assert_eq!(unified.bucket, "test-bucket");
        assert_eq!(unified.access_key.as_deref(), Some("admin"));
        assert_eq!(unified.secret_key.as_deref(), Some("password"));
        assert_eq!(unified.endpoint.as_deref(), Some("http://localhost:9000"));
        assert!(unified.path_style);
    }

    /// Test that `MinioStorage::new` validates config before attempting AWS SDK init.
    #[cfg(feature = "minio")]
    #[tokio::test]
    async fn test_minio_storage_new_invalid_config_fails() {
        let cfg = MinioConfig {
            bucket: String::new(),
            ..MinioConfig::local("k", "s", "b")
        };
        let result = MinioStorage::new(cfg).await;
        assert!(result.is_err());
    }

    // ── Tests for the in-memory stub (non-minio feature path) ────────────────

    #[cfg(not(feature = "minio"))]
    fn make_config() -> MinioConfig {
        MinioConfig::local("minioadmin", "minioadmin", "test-bucket")
    }

    #[cfg(not(feature = "minio"))]
    fn make_storage() -> MinioStorage {
        MinioStorage::new(make_config()).expect("valid minio storage")
    }

    #[cfg(not(feature = "minio"))]
    fn bytes_stream(data: Vec<u8>) -> ByteStream {
        use futures::stream;
        let b = Bytes::from(data);
        Box::pin(stream::once(async move { Ok::<Bytes, StorageError>(b) }))
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_upload_stream_and_exists() {
        let storage = make_storage();
        let stream = bytes_stream(b"hello minio".to_vec());
        storage
            .upload_stream(
                "objects/test.bin",
                stream,
                Some(11),
                UploadOptions::default(),
            )
            .await
            .expect("upload should succeed");
        assert!(storage
            .object_exists("objects/test.bin")
            .await
            .expect("exists check"));
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_download_stream_returns_data() {
        let storage = make_storage();
        let data = b"round-trip data".to_vec();
        let stream = bytes_stream(data.clone());
        storage
            .upload_stream("rt.bin", stream, None, UploadOptions::default())
            .await
            .expect("upload");

        use futures::StreamExt;
        let mut dl = storage
            .download_stream("rt.bin", DownloadOptions::default())
            .await
            .expect("download stream");
        let mut out = Vec::new();
        while let Some(chunk) = dl.next().await {
            out.extend_from_slice(&chunk.expect("chunk"));
        }
        assert_eq!(out, data);
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_get_metadata_returns_correct_size() {
        let storage = make_storage();
        let stream = bytes_stream(vec![0u8; 512]);
        storage
            .upload_stream("meta.bin", stream, Some(512), UploadOptions::default())
            .await
            .expect("upload");
        let meta = storage.get_metadata("meta.bin").await.expect("metadata");
        assert_eq!(meta.size, 512);
        assert!(meta.etag.is_some());
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_delete_object_removes_key() {
        let storage = make_storage();
        let stream = bytes_stream(b"delete me".to_vec());
        storage
            .upload_stream("del.bin", stream, None, UploadOptions::default())
            .await
            .expect("upload");
        storage.delete_object("del.bin").await.expect("delete");
        assert!(!storage
            .object_exists("del.bin")
            .await
            .expect("exists check"));
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_list_objects_with_prefix() {
        let storage = make_storage();
        for i in 0..3u8 {
            let stream = bytes_stream(vec![i; 10]);
            storage
                .upload_stream(
                    &format!("prefix/{i}.bin"),
                    stream,
                    None,
                    UploadOptions::default(),
                )
                .await
                .expect("upload");
        }
        // Add one outside the prefix
        let stream = bytes_stream(b"other".to_vec());
        storage
            .upload_stream("other.bin", stream, None, UploadOptions::default())
            .await
            .expect("upload");

        let result = storage
            .list_objects(ListOptions {
                prefix: Some("prefix/".to_string()),
                ..Default::default()
            })
            .await
            .expect("list");
        assert_eq!(result.objects.len(), 3);
    }

    #[cfg(not(feature = "minio"))]
    #[tokio::test]
    async fn test_presigned_url_contains_key() {
        let storage = make_storage();
        let url = storage
            .generate_presigned_url("media/clip.mp4", 3600)
            .await
            .expect("presigned url");
        assert!(url.contains("media/clip.mp4"));
        assert!(url.contains("3600"));
    }
}
