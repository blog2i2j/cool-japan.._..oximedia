# oximedia-storage

![Status: Stable](https://img.shields.io/badge/status-stable-green)

Cloud storage abstraction layer for OxiMedia providing unified access to S3, Azure Blob Storage, and Google Cloud Storage.

Part of the [oximedia](https://github.com/cool-japan/oximedia) workspace — a comprehensive pure-Rust media processing framework.

Version: 0.1.3 — 2026-04-15

## Features

- **Unified API** - Single interface across all cloud providers
- **Streaming** - Efficient streaming uploads/downloads without buffering entire files
- **Multipart Upload** - Automatic handling of large files
- **Progress Tracking** - Real-time progress callbacks
- **Retry Logic** - Exponential backoff for failed operations
- **Parallel Transfers** - Concurrent chunk downloads for large files
- **Local Caching** - Optional LRU cache with write-through/write-back policies
- **Rate Limiting** - Control bandwidth usage
- **Async/Await** - Full async support with tokio
- **Access Logging** - Structured storage access log and audit trail
- **Deduplication** - Content-addressable deduplication storage
- **Integrity Checking** - Data integrity verification for stored objects
- **Lifecycle Policies** - Age-based transitions, cost tiers, expiration rules
- **Namespace Management** - Logical grouping of objects with hierarchical names
- **Quota Management** - Storage quota enforcement and reporting
- **Replication** - Multi-site replication with policy management
- **Retention Management** - Object retention holds and policies
- **Storage Events** - Publish/subscribe for object lifecycle events
- **Storage Metrics** - Operation counters, gauges, histograms, error rates
- **Tiering** - Automatic storage class tiering
- **Transfer Statistics** - Throughput metrics for uploads/downloads
- **Write-ahead Log** - Crash-safe storage mutation tracking and replay
- **Compression Store** - Transparent compression with ratio tracking

## Supported Providers

### Amazon S3
- Standard and multipart uploads
- Presigned URLs
- Object versioning
- Storage classes (Standard, IA, Glacier, etc.)
- Server-side encryption

**Note**: Requires Rust 1.91+ due to AWS SDK requirements. Currently disabled by default.

### Azure Blob Storage
- Block blob operations
- Container management
- Access tiers (Hot/Cool/Archive)
- SAS token support

### Google Cloud Storage
- Standard uploads
- Bucket operations
- Object composition
- Signed URLs

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
oximedia-storage = { version = "0.1.3", features = ["azure", "gcs"] }

# Enable S3 (requires Rust 1.91+)
# oximedia-storage = { version = "0.1.3", features = ["s3"] }
```

```rust
use oximedia_storage::{UnifiedConfig, CloudStorage, UploadOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = UnifiedConfig::azure("my-container", "myaccount")
        .with_credentials("myaccount", "account_key");

    let storage = oximedia_storage::azure::AzureStorage::new(config).await?;

    let options = UploadOptions::default();
    let etag = storage.upload_file(
        "media/video.mp4",
        std::path::Path::new("/local/path/video.mp4"),
        options
    ).await?;

    println!("Uploaded with ETag: {}", etag);
    Ok(())
}
```

## Implementation Status

### Core Library
- CloudStorage trait with unified interface
- UnifiedConfig with provider-specific constructors
- Error types: NotFound, AuthenticationError, NetworkError, QuotaExceeded

### Providers
- Azure Blob Storage — partial (core operations)
- Google Cloud Storage — partial (core operations)
- Amazon S3 — blocked by Rust 1.91+ requirement

### Supporting Modules
- Transfer management with retry and parallel downloads
- LRU caching layer with write-through/write-back
- Access logging, integrity checking, deduplication
- Lifecycle policies, tiering, replication

## API Overview

- `CloudStorage` — Main async trait: upload_stream, upload_file, download_stream, download_file, list_objects, delete_object, copy_object, generate_presigned_url
- `UnifiedConfig` — Provider configuration builder: s3(), azure(), gcs(), with_credentials(), with_cache()
- `StorageProvider` — S3, Azure, GCS
- `ObjectMetadata` — Key, size, content_type, etag, last_modified, custom metadata
- `UploadOptions` — Content type, metadata, storage class, encryption, ACL
- `DownloadOptions` — Byte range, conditional headers
- `ListOptions` / `ListResult` — Pagination and prefix filtering
- `StorageError` / `Result` — Comprehensive error handling
- `ProgressInfo` / `ProgressCallback` — Transfer progress reporting
- Modules: `access_log`, `cache`, `compression_store`, `dedup_store`, `integrity_checker`, `lifecycle`, `local`, `namespace`, `object_store`, `path_resolver`, `quota`, `replication`, `replication_policy`, `retention_manager`, `storage_events`, `storage_metrics`, `storage_policy`, `tiering`, `transfer`, `transfer_stats`, `write_ahead_log`

## License

Apache-2.0 — Copyright 2024-2026 COOLJAPAN OU (Team Kitasan)
