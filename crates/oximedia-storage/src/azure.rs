//! Azure Blob Storage implementation

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ListOptions, ListResult, ObjectMetadata, Result,
    StorageError, UnifiedConfig, UploadOptions,
};
use async_trait::async_trait;
use azure_storage::StorageCredentials;
use azure_storage_blobs::{
    blob::{BlobBlockType, BlockList},
    prelude::{AccessTier as AzureAccessTier, BlobClient, BlobServiceClient, ContainerClient},
};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::StreamExt;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info};

/// Minimum block size (256 KB)
const MIN_BLOCK_SIZE: usize = 256 * 1024;

/// Maximum block size (100 MB)
const MAX_BLOCK_SIZE: usize = 100 * 1024 * 1024;

/// Default block size (4 MB)
const DEFAULT_BLOCK_SIZE: usize = 4 * 1024 * 1024;

/// Maximum number of blocks
const MAX_BLOCKS: usize = 50000;

/// Threshold for using block blob upload
const BLOCK_UPLOAD_THRESHOLD: u64 = 10 * 1024 * 1024;

/// Azure Blob Storage access tier
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessTier {
    Hot,
    Cool,
    Archive,
}

impl AccessTier {
    fn as_str(&self) -> &'static str {
        match self {
            AccessTier::Hot => "Hot",
            AccessTier::Cool => "Cool",
            AccessTier::Archive => "Archive",
        }
    }

    #[allow(dead_code)]
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hot" => Some(AccessTier::Hot),
            "cool" => Some(AccessTier::Cool),
            "archive" => Some(AccessTier::Archive),
            _ => None,
        }
    }
}

/// Azure Blob Storage client
pub struct AzureStorage {
    container_client: Arc<ContainerClient>,
    container: String,
    _config: UnifiedConfig,
}

impl AzureStorage {
    /// Create a new Azure storage client from configuration
    pub async fn new(config: UnifiedConfig) -> Result<Self> {
        let account = config
            .access_key
            .as_ref()
            .ok_or_else(|| StorageError::InvalidConfig("Account name required for Azure".into()))?;

        let credentials = if let Some(account_key) = &config.secret_key {
            StorageCredentials::access_key(account.clone(), account_key.clone())
        } else {
            return Err(StorageError::InvalidConfig(
                "Account key required for Azure".into(),
            ));
        };

        let service_client = BlobServiceClient::new(account, credentials);
        let container_client = service_client.container_client(&config.bucket);

        Ok(Self {
            container_client: Arc::new(container_client),
            container: config.bucket.clone(),
            _config: config,
        })
    }

    /// Get blob client
    fn blob_client(&self, blob_name: &str) -> BlobClient {
        self.container_client.blob_client(blob_name)
    }

    /// Upload using block blob (simplified version)
    async fn upload_blocks(
        &self,
        blob_name: &str,
        stream: ByteStream,
        total_size: u64,
        _options: &UploadOptions,
    ) -> Result<String> {
        info!("Starting block blob upload for: {}", blob_name);

        let blob_client = self.blob_client(blob_name);
        let block_size = calculate_block_size(total_size);

        let mut block_list = Vec::new();
        let mut block_id = 0u64;
        let mut stream = stream;
        let mut buffer = Vec::new();

        while let Some(result) = stream.next().await {
            let chunk = result?;
            buffer.extend_from_slice(&chunk);

            while buffer.len() >= block_size {
                let block_data = buffer.drain(..block_size).collect::<Vec<_>>();
                let block_id_str = format!("{block_id:016x}");

                debug!("Uploading block {} for blob: {}", block_id_str, blob_name);

                blob_client
                    .put_block(block_id_str.clone(), block_data)
                    .await
                    .map_err(|e| {
                        StorageError::ProviderError(format!("Failed to upload block: {e}"))
                    })?;

                block_list.push(BlobBlockType::new_uncommitted(block_id_str));
                block_id += 1;

                if block_list.len() > MAX_BLOCKS {
                    return Err(StorageError::ProviderError(
                        "Exceeded maximum number of blocks".into(),
                    ));
                }
            }
        }

        // Upload remaining data
        if !buffer.is_empty() {
            let block_id_str = format!("{block_id:016x}");
            blob_client
                .put_block(block_id_str.clone(), buffer)
                .await
                .map_err(|e| {
                    StorageError::ProviderError(format!("Failed to upload final block: {e}"))
                })?;

            block_list.push(BlobBlockType::new_uncommitted(block_id_str));
        }

        // Commit block list
        debug!(
            "Committing {} blocks for blob: {}",
            block_list.len(),
            blob_name
        );

        let response = blob_client
            .put_block_list(BlockList { blocks: block_list })
            .await
            .map_err(|e| {
                StorageError::ProviderError(format!("Failed to commit block list: {e}"))
            })?;

        let etag = response.etag.clone();

        info!("Block blob upload completed for: {}", blob_name);
        Ok(etag)
    }

    /// Create container
    pub async fn create_container(&self) -> Result<()> {
        info!("Creating container: {}", self.container);

        self.container_client
            .create()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to create container: {e}")))?;

        info!("Container created: {}", self.container);
        Ok(())
    }

    /// Delete container
    pub async fn delete_container(&self) -> Result<()> {
        info!("Deleting container: {}", self.container);

        self.container_client
            .delete()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to delete container: {e}")))?;

        info!("Container deleted: {}", self.container);
        Ok(())
    }

    /// Check if container exists
    pub async fn container_exists(&self) -> Result<bool> {
        match self.container_client.get_properties().await {
            Ok(_) => Ok(true),
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("ContainerNotFound") {
                    Ok(false)
                } else {
                    Err(StorageError::ProviderError(format!(
                        "Failed to check container existence: {e}"
                    )))
                }
            }
        }
    }

    /// Set blob access tier
    pub async fn set_blob_tier(&self, blob_name: &str, tier: AccessTier) -> Result<()> {
        info!("Setting blob tier to {} for: {}", tier.as_str(), blob_name);

        let blob_client = self.blob_client(blob_name);

        let azure_tier = match tier {
            AccessTier::Hot => AzureAccessTier::Hot,
            AccessTier::Cool => AzureAccessTier::Cool,
            AccessTier::Archive => AzureAccessTier::Archive,
        };

        blob_client
            .set_blob_tier(azure_tier)
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to set blob tier: {e}")))?;

        info!("Blob tier set to {} for: {}", tier.as_str(), blob_name);
        Ok(())
    }
}

#[async_trait]
impl CloudStorage for AzureStorage {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading stream to blob: {}", key);

        // Use block blob upload for large files or unknown sizes
        if size.map_or(true, |s| s > BLOCK_UPLOAD_THRESHOLD) {
            return self
                .upload_blocks(key, stream, size.unwrap_or(0), &options)
                .await;
        }

        // Simple upload for small files
        let mut chunks = Vec::new();
        let mut stream = stream;

        while let Some(result) = stream.next().await {
            let chunk = result?;
            chunks.extend_from_slice(&chunk);
        }

        let blob_client = self.blob_client(key);

        let response = blob_client
            .put_block_blob(chunks)
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to upload blob: {e}")))?;

        let etag = response.etag.clone();

        info!("Uploaded blob: {}", key);
        Ok(etag)
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        debug!("Uploading file {:?} to blob: {}", file_path, key);

        let mut file = File::open(file_path).await?;
        let metadata = file.metadata().await?;
        let file_size = metadata.len();

        if file_size > BLOCK_UPLOAD_THRESHOLD {
            // Use block upload
            let stream: ByteStream =
                Box::pin(futures::stream::try_unfold(file, |mut file| async move {
                    let mut buffer = vec![0u8; DEFAULT_BLOCK_SIZE];
                    let n = file.read(&mut buffer).await?;
                    if n == 0 {
                        Ok(None)
                    } else {
                        buffer.truncate(n);
                        Ok(Some((Bytes::from(buffer), file)))
                    }
                }));

            self.upload_blocks(key, stream, file_size, &options).await
        } else {
            // Simple upload
            let mut contents = Vec::new();
            file.read_to_end(&mut contents).await?;

            let blob_client = self.blob_client(key);

            let response = blob_client
                .put_block_blob(contents)
                .await
                .map_err(|e| StorageError::ProviderError(format!("Failed to upload file: {e}")))?;

            let etag = response.etag.clone();

            info!("Uploaded file to blob: {}", key);
            Ok(etag)
        }
    }

    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream> {
        debug!("Downloading stream from blob: {}", key);

        let blob_client = self.blob_client(key);

        let mut req = blob_client.get();

        if let Some((start, end)) = options.range {
            req = req.range(start..end);
        }

        let key_owned = key.to_string();
        let response = req
            .into_stream()
            .next()
            .await
            .ok_or_else(|| StorageError::ProviderError("No response from blob".to_string()))?
            .map_err(|e| {
                let err_str = e.to_string();
                if err_str.contains("404") || err_str.contains("BlobNotFound") {
                    StorageError::NotFound(key_owned)
                } else {
                    StorageError::ProviderError(format!("Failed to download blob: {e}"))
                }
            })?;

        let stream = response.data.map(|result| {
            result.map_err(|e| StorageError::NetworkError(format!("Stream error: {e}")))
        });

        Ok(Box::pin(stream))
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
    ) -> Result<()> {
        debug!("Downloading file from blob: {} to {:?}", key, file_path);

        let mut stream = self.download_stream(key, options).await?;
        let mut file = File::create(file_path).await?;

        while let Some(result) = stream.next().await {
            let chunk = result?;
            file.write_all(&chunk).await?;
        }

        file.flush().await?;
        info!("Downloaded file from blob: {} to {:?}", key, file_path);
        Ok(())
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        debug!("Getting metadata for blob: {}", key);

        let blob_client = self.blob_client(key);

        let properties = blob_client.get_properties().await.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("404") || err_str.contains("BlobNotFound") {
                StorageError::NotFound(key.to_string())
            } else {
                StorageError::ProviderError(format!("Failed to get blob properties: {e}"))
            }
        })?;

        // Convert time::OffsetDateTime to chrono::DateTime<Utc>
        let last_modified = {
            let offset_dt = properties.blob.properties.last_modified;
            let unix_timestamp = offset_dt.unix_timestamp();
            DateTime::from_timestamp(unix_timestamp, 0).unwrap_or_else(Utc::now)
        };

        let metadata: HashMap<String, String> = properties.blob.metadata.unwrap_or_default();

        Ok(ObjectMetadata {
            key: key.to_string(),
            size: properties.blob.properties.content_length,
            content_type: Some(properties.blob.properties.content_type),
            last_modified,
            etag: Some(properties.blob.properties.etag.to_string()),
            metadata,
            storage_class: None,
        })
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        debug!("Deleting blob: {}", key);

        let blob_client = self.blob_client(key);

        blob_client
            .delete()
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to delete blob: {e}")))?;

        info!("Deleted blob: {}", key);
        Ok(())
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        debug!("Deleting {} blobs", keys.len());

        let mut results = Vec::new();

        for key in keys {
            let result = self.delete_object(key).await;
            results.push(result);
        }

        info!("Deleted {} blobs", keys.len());
        Ok(results)
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        debug!("Listing blobs with prefix: {:?}", options.prefix);

        let mut req = self.container_client.list_blobs();

        if let Some(prefix) = options.prefix.clone() {
            req = req.prefix(prefix);
        }

        let response = req
            .into_stream()
            .next()
            .await
            .ok_or_else(|| StorageError::ProviderError("No response from list operation".into()))?
            .map_err(|e| StorageError::ProviderError(format!("Failed to list blobs: {e}")))?;

        let objects = response
            .blobs
            .blobs()
            .map(|blob| {
                let last_modified = {
                    let offset_dt = blob.properties.last_modified;
                    let unix_timestamp = offset_dt.unix_timestamp();
                    DateTime::from_timestamp(unix_timestamp, 0).unwrap_or_else(Utc::now)
                };

                ObjectMetadata {
                    key: blob.name.clone(),
                    size: blob.properties.content_length,
                    content_type: Some(blob.properties.content_type.clone()),
                    last_modified,
                    etag: Some(blob.properties.etag.to_string()),
                    metadata: HashMap::new(),
                    storage_class: None,
                }
            })
            .collect();

        Ok(ListResult {
            objects,
            prefixes: Vec::new(),
            next_token: response
                .next_marker
                .as_ref()
                .map(|m| m.as_str().to_string()),
            has_more: response.next_marker.is_some(),
        })
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        match self.get_metadata(key).await {
            Ok(_) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        debug!("Copying blob from {} to {}", source_key, dest_key);

        let source_blob_client = self.blob_client(source_key);
        let dest_blob_client = self.blob_client(dest_key);

        let source_url = source_blob_client
            .url()
            .map_err(|e| StorageError::ProviderError(format!("Failed to get source URL: {e}")))?;

        dest_blob_client
            .copy(source_url)
            .await
            .map_err(|e| StorageError::ProviderError(format!("Failed to copy blob: {e}")))?;

        info!("Copied blob from {} to {}", source_key, dest_key);
        Ok(())
    }

    async fn generate_presigned_url(&self, key: &str, _expiration_secs: u64) -> Result<String> {
        debug!("Generating presigned URL for blob: {}", key);

        let blob_client = self.blob_client(key);
        let base_url = blob_client
            .url()
            .map_err(|e| StorageError::ProviderError(format!("Failed to get blob URL: {e}")))?;

        // Note: Full SAS token generation requires additional Azure SDK features
        // This returns the base URL - in production, you'd add SAS token parameters
        Ok(base_url.to_string())
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        _expiration_secs: u64,
    ) -> Result<String> {
        debug!("Generating presigned upload URL for blob: {}", key);

        let blob_client = self.blob_client(key);
        let base_url = blob_client
            .url()
            .map_err(|e| StorageError::ProviderError(format!("Failed to get blob URL: {e}")))?;

        // Note: Full SAS token generation requires additional Azure SDK features
        // This returns the base URL - in production, you'd add SAS token parameters
        Ok(base_url.to_string())
    }
}

/// Calculate optimal block size
fn calculate_block_size(total_size: u64) -> usize {
    if total_size == 0 {
        return DEFAULT_BLOCK_SIZE;
    }

    let mut block_size = DEFAULT_BLOCK_SIZE;
    while total_size / block_size as u64 > MAX_BLOCKS as u64 && block_size < MAX_BLOCK_SIZE {
        block_size *= 2;
    }

    block_size.min(MAX_BLOCK_SIZE).max(MIN_BLOCK_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tier_conversion() {
        assert_eq!(AccessTier::Hot.as_str(), "Hot");
        assert_eq!(AccessTier::Cool.as_str(), "Cool");
        assert_eq!(AccessTier::Archive.as_str(), "Archive");

        assert_eq!(AccessTier::from_str("hot"), Some(AccessTier::Hot));
        assert_eq!(AccessTier::from_str("cool"), Some(AccessTier::Cool));
        assert_eq!(AccessTier::from_str("archive"), Some(AccessTier::Archive));
        assert_eq!(AccessTier::from_str("invalid"), None);
    }

    #[test]
    fn test_calculate_block_size() {
        assert_eq!(calculate_block_size(0), DEFAULT_BLOCK_SIZE);
        assert_eq!(calculate_block_size(100 * 1024 * 1024), DEFAULT_BLOCK_SIZE);

        let large_size = 1024u64 * 1024 * 1024 * 1024; // 1 TB
        let block_size = calculate_block_size(large_size);
        assert!(block_size <= MAX_BLOCK_SIZE);
        assert!(block_size >= MIN_BLOCK_SIZE);
    }
}
