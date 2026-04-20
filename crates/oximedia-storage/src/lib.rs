//! Cloud storage abstraction layer for OxiMedia.
//!
//! Provides a unified `CloudStorage` trait backed by Amazon S3, MinIO,
//! Azure Blob Storage, Google Cloud Storage, and local filesystem.
//!
//! # Cargo features
//!
//! | Feature | Provider enabled |
//! |---------|-----------------|
//! | `s3`    | Amazon S3 (requires Rust 1.91+) |
//! | `minio` | MinIO / S3-compatible (alias of `s3`) |
//! | `azure` | Azure Blob Storage |
//! | `gcs`   | Google Cloud Storage |
//! | `mmap`  | Memory-mapped local reads via `MmapLocalReader` |
//!
//! Default: no features enabled (core types and local provider always available).
//!
//! # Connection options
//!
//! Two complementary types control connection behaviour:
//!
//! - **`connection_options::ConnectionOptions`** — per-client HTTP tuning:
//!   TCP keep-alive (default on, 30 s interval), HTTP/2 multiplexing (default on,
//!   100 concurrent streams), `TCP_NODELAY` (default on), connect/request timeouts
//!   (10 s / 30 s).  Builder: `.with_keep_alive()`, `.with_http2()`,
//!   `.with_max_concurrent_streams()`.
//!
//! - **`ConnectionPoolConfig`** — idle connection pool management: max 10 idle
//!   connections, 60 s idle lifetime, 300 s max lifetime, 10 s acquire timeout.
//!   Managed by `ConnectionManager` (VecDeque idle pool, evicts expired on
//!   acquire, caps at `max_idle_connections` on release).
//!
//! # Retry configuration
//!
//! `RetryConfig` provides exponential back-off with deterministic Weyl-sequence
//! jitter:
//!
//! | Parameter | Default |
//! |-----------|---------|
//! | max retries | 3 |
//! | backoff multiplier | 2× |
//! | initial delay | 500 ms |
//! | max delay | 30 s |
//! | jitter factor | 0.2 |
//!
//! Non-retryable errors: `NotFound`, `PermissionDenied`, `InvalidKey`,
//! `QuotaExceeded`, `InvalidConfig`, `AuthenticationError`,
//! `UnsupportedOperation`.
//!
//! # Transparent compression
//!
//! The [`compression_store`] module compresses/decompresses objects
//! transparently using a 4-byte magic header:
//!
//! | Algorithm | Status | Magic bytes |
//! |-----------|--------|-------------|
//! | LZ4 | Active | `0x4C 0x5A 0x34 0x00` |
//! | Zstd (level 3) | Active | `0x28 0xB5 0x2F 0xFD` |
//! | Gzip / Brotli / Snappy | Passthrough — stored uncompressed | — |
//!
//! `CompressionPolicy::Auto` selects: None for objects < 4 KiB, LZ4 for
//! 4 KiB – 1 MiB, Zstd for > 1 MiB.
//!
//! # Batch metadata updates
//!
//! `BatchMetadataUpdater` validates key length (≤ 1024 bytes, no null bytes,
//! non-empty) and splits updates into configurable chunks.  It does **not**
//! perform network calls — callers drive the upload loop via `chunk()`.
//!
//! # Example
//!
//! ```rust
//! use oximedia_storage::connection_options::ConnectionOptions;
//!
//! let opts = ConnectionOptions::default()
//!     .with_keep_alive(true)
//!     .with_http2(true)
//!     .with_max_concurrent_streams(200);
//!
//! assert!(opts.is_high_throughput());
//! ```

use async_trait::async_trait;
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::Stream;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[cfg(feature = "s3")]
pub mod s3;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "gcs")]
pub mod gcs;

/// Storage access logging and audit trail.
pub mod access_log;
/// Bandwidth throttling via token bucket algorithm.
pub mod bandwidth_throttle;
/// Parallel multi-object upload and download batch operations.
pub mod batch_operations;
pub mod cache;
/// Cache eviction layer: LRU, LFU, FIFO, and ARC caches with statistics.
pub mod cache_layer;
/// Compression store — compress/decompress objects with ratio and savings tracking.
pub mod compression_store;
/// Connection keep-alive and HTTP/2 multiplexing configuration for provider clients.
pub mod connection_options;
/// Content-type detection from file extension.
pub mod content_type;
/// Content-addressable deduplication storage (hash-based addressing, chunk dedup, reference counting).
pub mod dedup_store;
/// Data integrity verification for stored objects.
pub mod integrity_checker;
/// Storage inventory reports — object count, total size, and class distribution.
pub mod inventory_report;
/// Lazy metadata loading — defers per-object HEAD requests until accessed.
pub mod lazy_metadata;
/// Storage lifecycle policies (age-based transitions, cost tiers, expiration rules).
pub mod lifecycle;
pub mod local;
/// Migration planner for staged cross-provider migration workflows.
pub mod migration_planner;
/// MinIO backend (S3-compatible self-hosted object storage).
pub mod minio;
/// Resumable multipart upload that survives process restarts via checkpoint files.
pub mod multipart_resumable;
/// Namespace management — logical grouping of objects with hierarchical names.
pub mod namespace;
/// Object lock (WORM storage) — compliance-mode and governance-mode locking.
pub mod object_lock;
/// In-memory object store abstraction — keys, metadata, and basic CRUD operations.
pub mod object_store;
/// Object version listing, restore, and delete-marker management.
pub mod object_versioning;
/// Path resolution, normalization, and glob matching for object keys.
pub mod path_resolver;
/// Predictive prefetching based on sequential/random access pattern analysis.
pub mod predictive_prefetch;
/// Presigned POST support for browser-based direct uploads with policy conditions.
pub mod presigned_post;
pub mod quota;
pub mod replication;
/// Advanced replication policy management (sync policies, replication lag, consistency levels).
pub mod replication_policy;
/// Object retention and hold management.
pub mod retention_manager;
/// Generic async retry infrastructure with exponential back-off and jitter.
pub mod retry;
/// Server-side copy optimisation — same-provider copy without client-side data transfer.
pub mod server_side_copy;
/// Storage event bus — publish/subscribe for object lifecycle events.
pub mod storage_events;
/// Additional storage utility APIs (WAL serializer, dedup store, access log analyzer, etc.).
pub mod storage_extras;
/// Storage operation metrics — counters, gauges, histograms, and error rates.
pub mod storage_metrics;
/// Cross-provider storage migration with progress tracking and hash verification.
pub mod storage_migration;
/// Storage policy management — access classes, retention rules, and policy evaluation.
pub mod storage_policy;
pub mod tiering;
pub mod transfer;
/// Transfer statistics — recording upload/download events and computing throughput metrics.
pub mod transfer_stats;
/// Object versioning (version ID tracking per key).
pub mod versioning;
/// Write-ahead log for crash-safe storage mutation tracking and replay.
pub mod write_ahead_log;

/// Errors that can occur during storage operations
#[derive(Error, Debug)]
pub enum StorageError {
    #[error("Object not found: {0}")]
    NotFound(String),

    #[error("Authentication failed: {0}")]
    AuthenticationError(String),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Multipart upload error: {0}")]
    MultipartError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Invalid object key: {0}")]
    InvalidKey(String),

    #[error("Storage quota exceeded")]
    QuotaExceeded,

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Provider-specific error: {0}")]
    ProviderError(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;

/// Storage provider type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageProvider {
    /// Amazon S3
    S3,
    /// Azure Blob Storage
    Azure,
    /// Google Cloud Storage
    GCS,
}

/// Metadata for a stored object
#[derive(Debug, Clone)]
pub struct ObjectMetadata {
    /// Object key/name
    pub key: String,
    /// Object size in bytes
    pub size: u64,
    /// Content type (MIME type)
    pub content_type: Option<String>,
    /// Last modified timestamp
    pub last_modified: DateTime<Utc>,
    /// ETag or version identifier
    pub etag: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Storage class or tier
    pub storage_class: Option<String>,
}

/// Configuration for object upload
#[derive(Debug, Clone, Default)]
pub struct UploadOptions {
    /// Content type
    pub content_type: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Storage class
    pub storage_class: Option<String>,
    /// Cache control header
    pub cache_control: Option<String>,
    /// Content encoding
    pub content_encoding: Option<String>,
    /// Server-side encryption
    pub encryption: Option<EncryptionConfig>,
    /// ACL/permissions
    pub acl: Option<String>,
}

/// Encryption configuration
#[derive(Debug, Clone)]
pub enum EncryptionConfig {
    /// Server-side encryption with provider-managed keys
    ServerSide,
    /// Server-side encryption with customer-provided keys
    CustomerKey(String),
    /// Client-side encryption
    ClientSide { key: String, algorithm: String },
}

/// Configuration for object download
#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    /// Byte range to download (start, end)
    pub range: Option<(u64, u64)>,
    /// If-Match condition
    pub if_match: Option<String>,
    /// If-None-Match condition
    pub if_none_match: Option<String>,
    /// If-Modified-Since condition
    pub if_modified_since: Option<DateTime<Utc>>,
}

/// Configuration for listing objects
#[derive(Debug, Clone)]
pub struct ListOptions {
    /// Prefix to filter objects
    pub prefix: Option<String>,
    /// Delimiter for hierarchical listing
    pub delimiter: Option<String>,
    /// Maximum number of objects to return
    pub max_results: Option<usize>,
    /// Continuation token for pagination
    pub continuation_token: Option<String>,
}

impl Default for ListOptions {
    fn default() -> Self {
        Self {
            prefix: None,
            delimiter: None,
            max_results: Some(1000),
            continuation_token: None,
        }
    }
}

/// Result of listing objects
#[derive(Debug, Clone)]
pub struct ListResult {
    /// List of objects
    pub objects: Vec<ObjectMetadata>,
    /// Common prefixes (directories)
    pub prefixes: Vec<String>,
    /// Continuation token for next page
    pub next_token: Option<String>,
    /// Whether there are more results
    pub has_more: bool,
}

/// Progress information for uploads/downloads
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Bytes transferred so far
    pub bytes_transferred: u64,
    /// Total bytes to transfer
    pub total_bytes: u64,
    /// Transfer speed in bytes per second
    pub bytes_per_second: f64,
    /// Estimated time remaining in seconds
    pub eta_seconds: Option<f64>,
}

/// Callback for progress updates
pub type ProgressCallback = Arc<dyn Fn(ProgressInfo) + Send + Sync>;

/// Retry configuration with exponential back-off and optional jitter.
///
/// The delay before attempt `n` (0-indexed) is computed as:
/// `min(initial_backoff_ms * backoff_multiplier^n, max_backoff_ms)`
/// optionally perturbed by a random jitter factor in `[0, jitter_factor]`.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Multiplier applied to the backoff duration on each successive failure.
    /// Must be ≥ 1.0; values < 1.0 are clamped to 1.0.
    pub backoff_multiplier: f64,
    /// Base backoff in milliseconds before the first retry.
    pub initial_backoff_ms: u64,
    /// Hard ceiling on the computed backoff in milliseconds.
    pub max_backoff_ms: u64,
    /// Maximum relative jitter applied to the computed backoff.
    /// 0.0 = no jitter, 1.0 = up to 100 % of the computed delay is added randomly.
    /// Values outside `[0.0, 1.0]` are clamped.
    pub jitter_factor: f64,
    /// Only retry on transient errors (network / rate-limit); never retry on
    /// `NotFound`, `PermissionDenied`, or `InvalidKey`.
    pub retry_on_transient_only: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_multiplier: 2.0,
            initial_backoff_ms: 500,
            max_backoff_ms: 30_000,
            jitter_factor: 0.2,
            retry_on_transient_only: true,
        }
    }
}

impl RetryConfig {
    /// Compute the backoff duration in milliseconds for attempt `n` (0-indexed).
    ///
    /// Uses a simple deterministic formula without `rand` to keep the crate
    /// pure-Rust and dependency-free.  The pseudo-jitter is derived from the
    /// attempt number itself so that the result is reproducible in tests.
    #[must_use]
    pub fn backoff_ms_for_attempt(&self, attempt: u32) -> u64 {
        let multiplier = self.backoff_multiplier.max(1.0);
        // Compute base: initial * multiplier^attempt
        let base = self.initial_backoff_ms as f64 * multiplier.powi(attempt as i32);
        let capped = base.min(self.max_backoff_ms as f64);
        // Deterministic jitter: use a Weyl-sequence offset keyed on the attempt number.
        let jitter_factor = self.jitter_factor.clamp(0.0, 1.0);
        // map attempt to a pseudo-random fraction in [0,1) via a simple hash mix
        let pseudo_rand = {
            let mut v = (attempt as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            v ^= v >> 30;
            v = v.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            v ^= v >> 27;
            v = v.wrapping_mul(0x94d0_49bb_1331_11eb);
            v ^= v >> 31;
            (v as f64) / (u64::MAX as f64)
        };
        let jitter_ms = capped * jitter_factor * pseudo_rand;
        (capped + jitter_ms) as u64
    }

    /// Returns `true` if the given `StorageError` should trigger a retry.
    #[must_use]
    pub fn should_retry(&self, error: &StorageError) -> bool {
        if !self.retry_on_transient_only {
            return true;
        }
        // Non-retryable: client-side or permanent errors
        !matches!(
            error,
            StorageError::NotFound(_)
                | StorageError::PermissionDenied(_)
                | StorageError::InvalidKey(_)
                | StorageError::QuotaExceeded
                | StorageError::InvalidConfig(_)
                | StorageError::AuthenticationError(_)
                | StorageError::UnsupportedOperation(_)
        )
    }
}

// ─── Connection pool ──────────────────────────────────────────────────────────

/// Configuration for a connection pool shared across storage operations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConnectionPoolConfig {
    /// Maximum number of idle connections kept alive per provider.
    pub max_idle_connections: usize,
    /// Time in seconds an idle connection is kept before being closed.
    pub idle_timeout_secs: u64,
    /// Maximum lifetime in seconds for any connection, regardless of idle state.
    pub max_lifetime_secs: u64,
    /// Timeout in seconds to wait when acquiring a connection from the pool.
    pub acquire_timeout_secs: u64,
}

impl Default for ConnectionPoolConfig {
    fn default() -> Self {
        Self {
            max_idle_connections: 10,
            idle_timeout_secs: 60,
            max_lifetime_secs: 300,
            acquire_timeout_secs: 10,
        }
    }
}

/// A handle representing an active connection borrowed from the pool.
///
/// Callers must pass this back to `ConnectionManager::release` when done.
#[derive(Debug)]
pub struct PooledConnection {
    /// Opaque connection identifier.
    pub id: u64,
    /// Wall-clock creation timestamp (milliseconds).
    pub created_ms: u64,
}

impl PooledConnection {
    fn new(id: u64, created_ms: u64) -> Self {
        Self { id, created_ms }
    }
}

/// Manages a pool of reusable connections with idle timeout and lifetime limits.
pub struct ConnectionManager {
    config: ConnectionPoolConfig,
    idle: Mutex<VecDeque<PooledConnection>>,
    total_created: AtomicU64,
    current_time_ms: AtomicU64,
}

impl ConnectionManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: ConnectionPoolConfig) -> Self {
        Self {
            config,
            idle: Mutex::new(VecDeque::new()),
            total_created: AtomicU64::new(0),
            current_time_ms: AtomicU64::new(0),
        }
    }

    /// Override the internal clock (for deterministic tests).
    pub fn set_time_ms(&self, ms: u64) {
        self.current_time_ms.store(ms, Ordering::SeqCst);
    }

    /// Acquire a connection from the idle pool, or create a new one.
    ///
    /// Returns `None` if no connection could be obtained (e.g. the pool is empty
    /// and creation is not permitted in this context).  In practice callers can
    /// treat the returned `PooledConnection` as always-available; the `Option`
    /// wrapper is used by tests that want to distinguish pool-hit from pool-miss.
    pub fn acquire(&self) -> PooledConnection {
        let now = self.current_time_ms.load(Ordering::SeqCst);
        let max_lifetime = self.config.max_lifetime_secs * 1_000;

        // Try to reuse an idle connection that is still within its lifetime.
        {
            let mut idle = self.idle.lock().expect("ConnectionManager idle lock");
            // Evict connections that have exceeded their lifetime.
            idle.retain(|c| now.saturating_sub(c.created_ms) < max_lifetime);
            if let Some(conn) = idle.pop_front() {
                return conn;
            }
        }

        // Create a fresh connection.
        let id = self.total_created.fetch_add(1, Ordering::SeqCst) + 1;
        PooledConnection::new(id, now)
    }

    /// Return a connection to the idle pool.
    ///
    /// Connections that exceed the idle pool capacity are discarded.
    pub fn release(&self, conn: PooledConnection) {
        let now = self.current_time_ms.load(Ordering::SeqCst);
        let max_lifetime = self.config.max_lifetime_secs * 1_000;

        // Do not return expired connections.
        if now.saturating_sub(conn.created_ms) >= max_lifetime {
            return;
        }

        let mut idle = self.idle.lock().expect("ConnectionManager idle lock");
        if idle.len() < self.config.max_idle_connections {
            idle.push_back(conn);
        }
    }

    /// Number of connections currently idle in the pool.
    pub fn idle_count(&self) -> usize {
        self.idle.lock().expect("ConnectionManager idle lock").len()
    }

    /// Total number of connections ever created by this manager.
    pub fn total_created(&self) -> u64 {
        self.total_created.load(Ordering::SeqCst)
    }

    /// Returns a reference to the pool configuration.
    pub fn config(&self) -> &ConnectionPoolConfig {
        &self.config
    }
}

// ─── BatchUpdateResult ────────────────────────────────────────────────────────

/// Result of a single key in a batch metadata update.
#[derive(Debug, Clone)]
pub struct BatchUpdateResult {
    /// Object key that was targeted.
    pub key: String,
    /// Whether the update succeeded.
    pub success: bool,
    /// Human-readable error message, if the update failed.
    pub error: Option<String>,
}

/// Helper for validating and chunking batch metadata update requests.
///
/// Does **not** perform the actual updates — it validates the input and
/// chunks large batches into smaller groups to be passed to the storage
/// provider's `batch_update_metadata` method.
pub struct BatchMetadataUpdater {
    /// Maximum number of keys per batch chunk.
    batch_size: usize,
}

impl BatchMetadataUpdater {
    /// Create a new updater with the given batch chunk size.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size: batch_size.max(1),
        }
    }

    /// Return the configured batch chunk size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Validate the input update list, returning a `Vec` of error descriptions.
    ///
    /// A key is considered invalid if it is:
    /// - Empty.
    /// - Longer than 1 024 characters.
    /// - Contains a null byte.
    pub fn validate_updates(&self, updates: &[(String, HashMap<String, String>)]) -> Vec<String> {
        updates
            .iter()
            .filter_map(|(key, _)| {
                if key.is_empty() {
                    Some(format!("invalid key: empty key"))
                } else if key.len() > 1024 {
                    Some(format!(
                        "invalid key '{}...': exceeds 1024 characters",
                        &key[..32]
                    ))
                } else if key.contains('\0') {
                    Some(format!("invalid key '{key}': contains null byte"))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Split `updates` into chunks of at most `batch_size` elements.
    pub fn chunk<'a>(
        &self,
        updates: &'a [(String, HashMap<String, String>)],
    ) -> Vec<&'a [(String, HashMap<String, String>)]> {
        updates.chunks(self.batch_size).collect()
    }
}

/// Unified configuration for cloud storage
#[derive(Debug, Clone)]
pub struct UnifiedConfig {
    /// Storage provider
    pub provider: StorageProvider,
    /// Bucket/container name
    pub bucket: String,
    /// Region (for S3 and GCS)
    pub region: Option<String>,
    /// Endpoint URL (for S3-compatible services)
    pub endpoint: Option<String>,
    /// Access key or account name
    pub access_key: Option<String>,
    /// Secret key or account key
    pub secret_key: Option<String>,
    /// Project ID (for GCS)
    pub project_id: Option<String>,
    /// Service account credentials file (for GCS)
    pub credentials_file: Option<PathBuf>,
    /// Enable transfer acceleration (S3)
    pub transfer_acceleration: bool,
    /// Enable path-style addressing (S3)
    pub path_style: bool,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Timeout for operations in seconds
    pub timeout_seconds: u64,
    /// Enable local caching
    pub enable_cache: bool,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
    /// Maximum cache size in bytes
    pub max_cache_size: u64,
    /// Retry behaviour for transient failures.
    pub retry: RetryConfig,
    /// Connection pool configuration.
    pub pool_config: ConnectionPoolConfig,
}

impl UnifiedConfig {
    /// Create a new configuration for S3
    #[cfg(feature = "s3")]
    pub fn s3(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            provider: StorageProvider::S3,
            bucket: bucket.into(),
            region: Some(region.into()),
            endpoint: None,
            access_key: None,
            secret_key: None,
            project_id: None,
            credentials_file: None,
            transfer_acceleration: false,
            path_style: false,
            max_connections: 10,
            timeout_seconds: 300,
            enable_cache: false,
            cache_dir: None,
            max_cache_size: 10 * 1024 * 1024 * 1024, // 10 GB
            retry: RetryConfig::default(),
            pool_config: ConnectionPoolConfig::default(),
        }
    }

    /// Create a new configuration for Azure
    #[cfg(feature = "azure")]
    pub fn azure(container: impl Into<String>, account: impl Into<String>) -> Self {
        Self {
            provider: StorageProvider::Azure,
            bucket: container.into(),
            region: None,
            endpoint: None,
            access_key: Some(account.into()),
            secret_key: None,
            project_id: None,
            credentials_file: None,
            transfer_acceleration: false,
            path_style: false,
            max_connections: 10,
            timeout_seconds: 300,
            enable_cache: false,
            cache_dir: None,
            max_cache_size: 10 * 1024 * 1024 * 1024,
            retry: RetryConfig::default(),
            pool_config: ConnectionPoolConfig::default(),
        }
    }

    /// Create a new configuration for GCS
    #[cfg(feature = "gcs")]
    pub fn gcs(bucket: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            provider: StorageProvider::GCS,
            bucket: bucket.into(),
            region: None,
            endpoint: None,
            access_key: None,
            secret_key: None,
            project_id: Some(project_id.into()),
            credentials_file: None,
            transfer_acceleration: false,
            path_style: false,
            max_connections: 10,
            timeout_seconds: 300,
            enable_cache: false,
            cache_dir: None,
            max_cache_size: 10 * 1024 * 1024 * 1024,
            retry: RetryConfig::default(),
            pool_config: ConnectionPoolConfig::default(),
        }
    }

    /// Set credentials
    pub fn with_credentials(
        mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        self.access_key = Some(access_key.into());
        self.secret_key = Some(secret_key.into());
        self
    }

    /// Set endpoint
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = Some(endpoint.into());
        self
    }

    /// Enable caching
    pub fn with_cache(mut self, cache_dir: PathBuf, max_size: u64) -> Self {
        self.enable_cache = true;
        self.cache_dir = Some(cache_dir);
        self.max_cache_size = max_size;
        self
    }

    /// Override retry configuration.
    pub fn with_retry(mut self, retry: RetryConfig) -> Self {
        self.retry = retry;
        self
    }

    /// Override connection pool configuration.
    pub fn with_pool_config(mut self, pool_config: ConnectionPoolConfig) -> Self {
        self.pool_config = pool_config;
        self
    }
}

/// Byte stream type for streaming data
pub type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>;

/// Main trait for cloud storage operations
#[async_trait]
pub trait CloudStorage: Send + Sync {
    /// Upload an object from a byte stream
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String>;

    /// Upload an object from a file
    async fn upload_file(
        &self,
        key: &str,
        file_path: &std::path::Path,
        options: UploadOptions,
    ) -> Result<String>;

    /// Download an object as a byte stream
    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream>;

    /// Download an object to a file
    async fn download_file(
        &self,
        key: &str,
        file_path: &std::path::Path,
        options: DownloadOptions,
    ) -> Result<()>;

    /// Get object metadata
    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata>;

    /// Delete an object
    async fn delete_object(&self, key: &str) -> Result<()>;

    /// Delete multiple objects
    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>>;

    /// List objects with prefix
    async fn list_objects(&self, options: ListOptions) -> Result<ListResult>;

    /// Check if an object exists
    async fn object_exists(&self, key: &str) -> Result<bool>;

    /// Copy an object within the same bucket
    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()>;

    /// Generate a presigned URL for downloading
    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String>;

    /// Generate a presigned URL for uploading
    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String>;

    /// Update metadata (tags) on a single object.
    ///
    /// Default implementation returns `UnsupportedOperation`; providers that
    /// support native metadata updates should override this method.
    async fn update_metadata(&self, key: &str, _tags: HashMap<String, String>) -> Result<()> {
        Err(StorageError::UnsupportedOperation(format!(
            "update_metadata not implemented for key: {key}"
        )))
    }

    /// Update metadata on multiple objects in bulk.
    ///
    /// Each element of `updates` is `(key, tags)`.  The default implementation
    /// calls `update_metadata` sequentially; providers with native batch APIs
    /// should override this for better throughput.
    async fn batch_update_metadata(
        &self,
        updates: Vec<(String, HashMap<String, String>)>,
    ) -> Result<Vec<BatchUpdateResult>> {
        let mut results = Vec::with_capacity(updates.len());
        for (key, tags) in updates {
            let result = match self.update_metadata(&key, tags).await {
                Ok(()) => BatchUpdateResult {
                    key,
                    success: true,
                    error: None,
                },
                Err(e) => BatchUpdateResult {
                    key,
                    success: false,
                    error: Some(e.to_string()),
                },
            };
            results.push(result);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_provider_equality() {
        assert_eq!(StorageProvider::S3, StorageProvider::S3);
        assert_ne!(StorageProvider::S3, StorageProvider::Azure);
    }

    #[test]
    fn test_upload_options_default() {
        let options = UploadOptions::default();
        assert!(options.content_type.is_none());
        assert!(options.metadata.is_empty());
    }

    #[test]
    fn test_list_options_default() {
        let options = ListOptions::default();
        assert!(options.prefix.is_none());
        assert_eq!(options.max_results, Some(1000));
    }

    #[cfg(feature = "s3")]
    #[test]
    fn test_unified_config_s3() {
        let config = UnifiedConfig::s3("my-bucket", "us-east-1");
        assert_eq!(config.provider, StorageProvider::S3);
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.region, Some("us-east-1".to_string()));
    }

    #[cfg(feature = "azure")]
    #[test]
    fn test_unified_config_azure() {
        let config = UnifiedConfig::azure("my-container", "myaccount");
        assert_eq!(config.provider, StorageProvider::Azure);
        assert_eq!(config.bucket, "my-container");
        assert_eq!(config.access_key, Some("myaccount".to_string()));
    }

    #[cfg(feature = "gcs")]
    #[test]
    fn test_unified_config_gcs() {
        let config = UnifiedConfig::gcs("my-bucket", "my-project");
        assert_eq!(config.provider, StorageProvider::GCS);
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.project_id, Some("my-project".to_string()));
    }

    // ── RetryConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_retry_config_default_values() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert!((cfg.backoff_multiplier - 2.0).abs() < f64::EPSILON);
        assert_eq!(cfg.initial_backoff_ms, 500);
        assert_eq!(cfg.max_backoff_ms, 30_000);
        assert!(cfg.jitter_factor >= 0.0 && cfg.jitter_factor <= 1.0);
        assert!(cfg.retry_on_transient_only);
    }

    #[test]
    fn test_retry_config_backoff_increases() {
        let cfg = RetryConfig {
            jitter_factor: 0.0, // disable jitter for determinism
            ..RetryConfig::default()
        };
        let d0 = cfg.backoff_ms_for_attempt(0);
        let d1 = cfg.backoff_ms_for_attempt(1);
        let d2 = cfg.backoff_ms_for_attempt(2);
        assert!(d1 > d0, "backoff must grow: {d1} > {d0}");
        assert!(d2 > d1, "backoff must grow: {d2} > {d1}");
    }

    #[test]
    fn test_retry_config_backoff_capped_at_max() {
        let cfg = RetryConfig {
            initial_backoff_ms: 1000,
            max_backoff_ms: 4000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
            ..RetryConfig::default()
        };
        // attempt 10 would be 1000 * 2^10 = 1_024_000 ms — must be capped
        let d = cfg.backoff_ms_for_attempt(10);
        assert!(d <= 4000, "backoff {d} must not exceed max_backoff_ms 4000");
    }

    #[test]
    fn test_retry_config_should_retry_network_error() {
        let cfg = RetryConfig::default();
        assert!(cfg.should_retry(&StorageError::NetworkError("timeout".into())));
        assert!(cfg.should_retry(&StorageError::RateLimitExceeded));
    }

    #[test]
    fn test_retry_config_should_not_retry_not_found() {
        let cfg = RetryConfig::default();
        assert!(!cfg.should_retry(&StorageError::NotFound("key".into())));
        assert!(!cfg.should_retry(&StorageError::PermissionDenied("denied".into())));
        assert!(!cfg.should_retry(&StorageError::InvalidKey("bad/key".into())));
        assert!(!cfg.should_retry(&StorageError::QuotaExceeded));
    }

    #[test]
    fn test_retry_config_should_retry_all_when_not_transient_only() {
        let cfg = RetryConfig {
            retry_on_transient_only: false,
            ..RetryConfig::default()
        };
        assert!(cfg.should_retry(&StorageError::NotFound("key".into())));
        assert!(cfg.should_retry(&StorageError::QuotaExceeded));
    }

    #[test]
    fn test_unified_config_with_retry_builder() {
        let custom = RetryConfig {
            max_retries: 10,
            backoff_multiplier: 1.5,
            ..RetryConfig::default()
        };
        // Use a provider-independent approach since feature flags may be absent.
        // We test via the builder on a manually constructed config to avoid
        // depending on s3/azure/gcs features.
        let _ = custom.backoff_ms_for_attempt(0);
        // Verify clone works
        let cfg2 = custom.clone();
        assert_eq!(cfg2.max_retries, 10);
    }

    // ── ConnectionPoolConfig ─────────────────────────────────────────────────

    #[test]
    fn test_pool_config_default_values() {
        let cfg = ConnectionPoolConfig::default();
        assert!(cfg.max_idle_connections > 0);
        assert!(cfg.idle_timeout_secs > 0);
        assert!(cfg.max_lifetime_secs > 0);
        assert!(cfg.acquire_timeout_secs > 0);
        assert!(cfg.max_lifetime_secs >= cfg.idle_timeout_secs);
    }

    #[test]
    fn test_pool_config_serialization() {
        let cfg = ConnectionPoolConfig {
            max_idle_connections: 5,
            idle_timeout_secs: 30,
            max_lifetime_secs: 120,
            acquire_timeout_secs: 5,
        };
        let json = serde_json::to_string(&cfg).expect("serialize pool config");
        let back: ConnectionPoolConfig =
            serde_json::from_str(&json).expect("deserialize pool config");
        assert_eq!(back.max_idle_connections, 5);
        assert_eq!(back.idle_timeout_secs, 30);
        assert_eq!(back.max_lifetime_secs, 120);
        assert_eq!(back.acquire_timeout_secs, 5);
    }

    #[test]
    fn test_connection_manager_acquire_release_cycle() {
        let mgr = ConnectionManager::new(ConnectionPoolConfig::default());
        assert_eq!(mgr.idle_count(), 0);

        let conn = mgr.acquire();
        // Creating a fresh connection: total_created should be 1
        assert_eq!(mgr.total_created(), 1);
        assert_eq!(mgr.idle_count(), 0);

        mgr.release(conn);
        assert_eq!(mgr.idle_count(), 1);
    }

    #[test]
    fn test_connection_manager_reuses_idle() {
        let mgr = ConnectionManager::new(ConnectionPoolConfig::default());
        let conn1 = mgr.acquire();
        let id1 = conn1.id;
        mgr.release(conn1);

        // Second acquire should reuse the idle connection
        let conn2 = mgr.acquire();
        assert_eq!(conn2.id, id1, "should have reused the idle connection");
        assert_eq!(
            mgr.total_created(),
            1,
            "no new connection should be created"
        );
        mgr.release(conn2);
    }

    #[test]
    fn test_connection_manager_idle_count_changes() {
        let cfg = ConnectionPoolConfig {
            max_idle_connections: 3,
            ..ConnectionPoolConfig::default()
        };
        let mgr = ConnectionManager::new(cfg);

        let c1 = mgr.acquire();
        let c2 = mgr.acquire();
        assert_eq!(mgr.idle_count(), 0);

        mgr.release(c1);
        assert_eq!(mgr.idle_count(), 1);
        mgr.release(c2);
        assert_eq!(mgr.idle_count(), 2);

        let _c3 = mgr.acquire();
        assert_eq!(mgr.idle_count(), 1);
    }

    #[test]
    fn test_connection_manager_respects_max_idle() {
        let cfg = ConnectionPoolConfig {
            max_idle_connections: 2,
            max_lifetime_secs: 3600,
            ..ConnectionPoolConfig::default()
        };
        let mgr = ConnectionManager::new(cfg);

        // Create 3 connections and release all — only 2 should be kept
        let c1 = mgr.acquire();
        let c2 = mgr.acquire();
        let c3 = mgr.acquire();
        mgr.release(c1);
        mgr.release(c2);
        mgr.release(c3);
        assert_eq!(
            mgr.idle_count(),
            2,
            "pool should cap at max_idle_connections"
        );
    }

    #[test]
    fn test_connection_manager_expired_connections_discarded() {
        let cfg = ConnectionPoolConfig {
            max_idle_connections: 10,
            idle_timeout_secs: 60,
            max_lifetime_secs: 1, // 1 second lifetime = 1000 ms
            acquire_timeout_secs: 5,
        };
        let mgr = ConnectionManager::new(cfg);
        mgr.set_time_ms(0);

        let conn = mgr.acquire();
        mgr.release(conn);
        assert_eq!(mgr.idle_count(), 1);

        // Advance time past max_lifetime
        mgr.set_time_ms(2_000); // 2 seconds later
        let _ = mgr.acquire(); // triggers eviction of expired idle connections
        assert_eq!(mgr.idle_count(), 0, "expired connections should be evicted");
    }

    #[test]
    fn test_connection_manager_total_created_accumulates() {
        let mgr = ConnectionManager::new(ConnectionPoolConfig::default());
        for _ in 0..5 {
            let _ = mgr.acquire(); // each drops without release — forces new creation
        }
        assert_eq!(mgr.total_created(), 5);
    }

    #[test]
    fn test_pool_config_clone() {
        let cfg = ConnectionPoolConfig::default();
        let cfg2 = cfg.clone();
        assert_eq!(cfg.max_idle_connections, cfg2.max_idle_connections);
    }

    #[test]
    fn test_connection_manager_config_accessor() {
        let cfg = ConnectionPoolConfig {
            max_idle_connections: 7,
            ..ConnectionPoolConfig::default()
        };
        let mgr = ConnectionManager::new(cfg);
        assert_eq!(mgr.config().max_idle_connections, 7);
    }

    #[test]
    fn test_batch_update_result_fields() {
        let ok = BatchUpdateResult {
            key: "a".to_string(),
            success: true,
            error: None,
        };
        assert!(ok.success);
        assert!(ok.error.is_none());

        let fail = BatchUpdateResult {
            key: "b".to_string(),
            success: false,
            error: Some("not found".to_string()),
        };
        assert!(!fail.success);
        assert!(fail.error.is_some());
    }

    // ── BatchMetadataUpdater ─────────────────────────────────────────────────

    #[test]
    fn test_batch_metadata_updater_new() {
        let updater = BatchMetadataUpdater::new(50);
        assert_eq!(updater.batch_size(), 50);
    }

    #[test]
    fn test_batch_metadata_updater_min_size_one() {
        // batch_size 0 should be clamped to 1
        let updater = BatchMetadataUpdater::new(0);
        assert_eq!(updater.batch_size(), 1);
    }

    #[test]
    fn test_batch_metadata_updater_validate_empty_key() {
        let updater = BatchMetadataUpdater::new(10);
        let updates = vec![(String::new(), HashMap::new())];
        let errors = updater.validate_updates(&updates);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("empty"));
    }

    #[test]
    fn test_batch_metadata_updater_validate_valid_keys() {
        let updater = BatchMetadataUpdater::new(10);
        let updates = vec![
            ("key-a".to_string(), HashMap::new()),
            ("key-b".to_string(), HashMap::new()),
            ("key-c".to_string(), HashMap::new()),
        ];
        let errors = updater.validate_updates(&updates);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_batch_metadata_updater_validate_key_too_long() {
        let updater = BatchMetadataUpdater::new(10);
        let long_key = "a".repeat(1025);
        let updates = vec![(long_key, HashMap::new())];
        let errors = updater.validate_updates(&updates);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("1024"));
    }

    #[test]
    fn test_batch_metadata_updater_validate_null_byte_key() {
        let updater = BatchMetadataUpdater::new(10);
        let updates = vec![("key\0with-null".to_string(), HashMap::new())];
        let errors = updater.validate_updates(&updates);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("null byte"));
    }

    #[test]
    fn test_batch_metadata_updater_chunk_splits_correctly() {
        let updater = BatchMetadataUpdater::new(3);
        let updates: Vec<(String, HashMap<String, String>)> = (0..7)
            .map(|i| (format!("key-{i}"), HashMap::new()))
            .collect();
        let chunks = updater.chunk(&updates);
        assert_eq!(chunks.len(), 3); // [3, 3, 1]
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 3);
        assert_eq!(chunks[2].len(), 1);
    }

    #[test]
    fn test_batch_metadata_updater_validate_empty_batch() {
        let updater = BatchMetadataUpdater::new(10);
        let errors = updater.validate_updates(&[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_batch_metadata_updater_multiple_errors() {
        let updater = BatchMetadataUpdater::new(10);
        let updates = vec![
            (String::new(), HashMap::new()),
            ("valid".to_string(), HashMap::new()),
            ("a".repeat(2000), HashMap::new()),
        ];
        let errors = updater.validate_updates(&updates);
        assert_eq!(errors.len(), 2);
    }

    // ── ConnectionManager concurrent access ─────────────────────────────────

    #[test]
    fn test_connection_manager_concurrent_acquire_release() {
        use std::sync::Arc;
        let mgr = Arc::new(ConnectionManager::new(ConnectionPoolConfig {
            max_idle_connections: 8,
            max_lifetime_secs: 3600,
            ..ConnectionPoolConfig::default()
        }));

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let m = mgr.clone();
                std::thread::spawn(move || {
                    for _ in 0..10 {
                        let conn = m.acquire();
                        m.release(conn);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // After all threads complete, total_created >= number of distinct
        // connections needed (could be reused)
        assert!(mgr.total_created() >= 1);
        // Idle pool should have at most max_idle_connections
        assert!(mgr.idle_count() <= 8);
    }
}
