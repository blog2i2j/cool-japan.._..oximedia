//! Server-side copy optimisation for same-provider transfers.
//!
//! When copying objects between locations within the same storage provider,
//! it is far more efficient to ask the provider to perform the copy server-side
//! (no data traverses the network to the client) rather than downloading and
//! re-uploading.  This module provides a `ServerSideCopy` helper that
//! encapsulates the copy request parameters and a provider-agnostic description
//! of the operation.
//!
//! Storage backends that support native server-side copy (S3 `CopyObject`, Azure
//! `Copy Blob`, GCS `rewriteObject`) should inspect `ServerSideCopy` and
//! delegate to the appropriate API call.  Backends that do not support it can
//! fall back to a download-and-upload path.

#![allow(dead_code)]

use crate::StorageProvider;

// ─── CopyOptions ─────────────────────────────────────────────────────────────

/// Additional options for a server-side copy operation.
#[derive(Debug, Clone, Default)]
pub struct CopyOptions {
    /// Override the destination content type.  If `None`, the source content
    /// type is preserved.
    pub content_type: Option<String>,
    /// Additional metadata to attach to the destination object.  If `None`,
    /// metadata is copied from the source.
    pub metadata: Option<std::collections::HashMap<String, String>>,
    /// Target storage class for the destination object (provider-specific
    /// string, e.g. `"STANDARD"`, `"INFREQUENT_ACCESS"`, `"GLACIER"`).
    pub storage_class: Option<String>,
    /// Source object version ID to copy (useful when versioning is enabled).
    pub source_version_id: Option<String>,
    /// Whether to replace the destination's metadata with `metadata` above
    /// (`true`) or to copy source metadata (`false`, default).
    pub replace_metadata: bool,
}

// ─── ServerSideCopy ───────────────────────────────────────────────────────────

/// Describes a server-side copy request.
///
/// Use [`ServerSideCopy::new`] to build a request and pass it to a storage
/// backend that implements server-side copy support.
#[derive(Debug, Clone)]
pub struct ServerSideCopy {
    /// Storage provider (must be the same for source and destination).
    pub provider: StorageProvider,
    /// Source bucket / container.
    pub source_bucket: String,
    /// Source object key.
    pub source_key: String,
    /// Destination bucket / container.
    pub dest_bucket: String,
    /// Destination object key.
    pub dest_key: String,
    /// Optional copy options.
    pub options: CopyOptions,
}

impl ServerSideCopy {
    /// Create a simple same-bucket copy request.
    pub fn new(
        provider: StorageProvider,
        bucket: impl Into<String>,
        source_key: impl Into<String>,
        dest_key: impl Into<String>,
    ) -> Self {
        let bucket = bucket.into();
        Self {
            provider,
            source_bucket: bucket.clone(),
            source_key: source_key.into(),
            dest_bucket: bucket,
            dest_key: dest_key.into(),
            options: CopyOptions::default(),
        }
    }

    /// Create a cross-bucket copy request (same provider).
    pub fn cross_bucket(
        provider: StorageProvider,
        source_bucket: impl Into<String>,
        source_key: impl Into<String>,
        dest_bucket: impl Into<String>,
        dest_key: impl Into<String>,
    ) -> Self {
        Self {
            provider,
            source_bucket: source_bucket.into(),
            source_key: source_key.into(),
            dest_bucket: dest_bucket.into(),
            dest_key: dest_key.into(),
            options: CopyOptions::default(),
        }
    }

    /// Apply copy options.
    pub fn with_options(mut self, options: CopyOptions) -> Self {
        self.options = options;
        self
    }

    /// Override destination content type.
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.options.content_type = Some(content_type.into());
        self
    }

    /// Override destination storage class.
    pub fn with_storage_class(mut self, storage_class: impl Into<String>) -> Self {
        self.options.storage_class = Some(storage_class.into());
        self
    }

    /// Returns `true` if source and destination are in the same bucket.
    pub fn is_intra_bucket(&self) -> bool {
        self.source_bucket == self.dest_bucket
    }

    /// Build an S3-style copy source string (`"{bucket}/{key}"`).
    ///
    /// This is suitable for use as the `x-amz-copy-source` HTTP header value.
    pub fn s3_copy_source(&self) -> String {
        format!("{}/{}", self.source_bucket, self.source_key)
    }

    /// Build an S3-style copy source with optional version ID.
    pub fn s3_copy_source_versioned(&self) -> String {
        match &self.options.source_version_id {
            Some(vid) => format!("{}/{}?versionId={vid}", self.source_bucket, self.source_key),
            None => self.s3_copy_source(),
        }
    }
}

// ─── CopyResult ───────────────────────────────────────────────────────────────

/// Result returned after a successful server-side copy.
#[derive(Debug, Clone)]
pub struct CopyResult {
    /// ETag of the newly created destination object.
    pub etag: String,
    /// Size of the copied object in bytes.
    pub size_bytes: u64,
    /// Whether the copy was performed server-side (true) or via download/upload.
    pub server_side: bool,
}

impl CopyResult {
    /// Create a result representing a server-side copy.
    pub fn server_side(etag: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            etag: etag.into(),
            size_bytes,
            server_side: true,
        }
    }

    /// Create a result representing a client-side (download/upload) fallback.
    pub fn client_side(etag: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            etag: etag.into(),
            size_bytes,
            server_side: false,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_bucket_copy() {
        let req = ServerSideCopy::new(StorageProvider::S3, "my-bucket", "src/a.mp4", "dst/a.mp4");
        assert_eq!(req.source_bucket, "my-bucket");
        assert_eq!(req.dest_bucket, "my-bucket");
        assert!(req.is_intra_bucket());
    }

    #[test]
    fn test_cross_bucket_copy() {
        let req = ServerSideCopy::cross_bucket(
            StorageProvider::S3,
            "bucket-a",
            "key/a",
            "bucket-b",
            "key/b",
        );
        assert!(!req.is_intra_bucket());
    }

    #[test]
    fn test_s3_copy_source() {
        let req = ServerSideCopy::new(StorageProvider::S3, "bucket", "videos/a.ts", "archive/a.ts");
        assert_eq!(req.s3_copy_source(), "bucket/videos/a.ts");
    }

    #[test]
    fn test_s3_copy_source_with_version() {
        let mut req = ServerSideCopy::new(StorageProvider::S3, "bucket", "obj", "dest");
        req.options.source_version_id = Some("v123".to_string());
        assert_eq!(req.s3_copy_source_versioned(), "bucket/obj?versionId=v123");
    }

    #[test]
    fn test_with_content_type() {
        let req = ServerSideCopy::new(StorageProvider::S3, "b", "src", "dst")
            .with_content_type("video/mp4");
        assert_eq!(req.options.content_type.as_deref(), Some("video/mp4"));
    }

    #[test]
    fn test_copy_result_server_side_flag() {
        let r = CopyResult::server_side("etag-abc", 1024);
        assert!(r.server_side);
        let r2 = CopyResult::client_side("etag-xyz", 2048);
        assert!(!r2.server_side);
    }
}
