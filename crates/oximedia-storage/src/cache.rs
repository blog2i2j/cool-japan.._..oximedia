//! Local caching layer for cloud storage

use crate::{
    ByteStream, CloudStorage, DownloadOptions, ListOptions, ListResult, ObjectMetadata, Result,
    StorageError, UploadOptions,
};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream;
use lru::LruCache;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Default cache size (1 GB)
const DEFAULT_CACHE_SIZE: u64 = 1024 * 1024 * 1024;

/// Default cache entry limit
const DEFAULT_CACHE_ENTRIES: usize = 10000;

/// Cache write policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePolicy {
    /// Write to cache and storage simultaneously
    WriteThrough,
    /// Write to cache first, then to storage asynchronously
    WriteBack,
    /// Write only to storage, update cache on read
    WriteAround,
}

/// Cache eviction policy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes
    pub max_size: u64,
    /// Maximum number of cache entries
    pub max_entries: usize,
    /// Write policy
    pub write_policy: WritePolicy,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Enable prefetching
    pub prefetch_enabled: bool,
    /// Prefetch size (number of objects)
    pub prefetch_size: usize,
    /// TTL for cache entries (seconds)
    pub ttl_seconds: Option<u64>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_dir: std::env::temp_dir().join("oximedia-cache"),
            max_size: DEFAULT_CACHE_SIZE,
            max_entries: DEFAULT_CACHE_ENTRIES,
            write_policy: WritePolicy::WriteThrough,
            eviction_policy: EvictionPolicy::LRU,
            prefetch_enabled: false,
            prefetch_size: 5,
            ttl_seconds: None,
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    key: String,
    file_path: PathBuf,
    size: u64,
    access_count: u64,
    created_at: std::time::SystemTime,
    last_accessed: std::time::SystemTime,
    etag: Option<String>,
}

/// Cached storage implementation
pub struct CachedStorage<S: CloudStorage + 'static> {
    storage: Arc<S>,
    config: CacheConfig,
    cache_index: Arc<RwLock<LruCache<String, CacheEntry>>>,
    current_size: Arc<RwLock<u64>>,
    dirty_entries: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl<S: CloudStorage> CachedStorage<S> {
    /// Create a new cached storage
    pub async fn new(storage: S, config: CacheConfig) -> Result<Self> {
        // Create cache directory if it doesn't exist
        fs::create_dir_all(&config.cache_dir).await?;

        let cache_capacity = NonZeroUsize::new(config.max_entries)
            .ok_or_else(|| StorageError::InvalidConfig("Cache entries must be > 0".into()))?;

        let cache_index = Arc::new(RwLock::new(LruCache::new(cache_capacity)));

        let cached_storage = Self {
            storage: Arc::new(storage),
            config,
            cache_index,
            current_size: Arc::new(RwLock::new(0)),
            dirty_entries: Arc::new(RwLock::new(HashMap::new())),
        };

        // Load existing cache index
        cached_storage.load_index().await?;

        Ok(cached_storage)
    }

    /// Load cache index from disk
    async fn load_index(&self) -> Result<()> {
        let index_file = self.config.cache_dir.join("index.json");

        if !index_file.exists() {
            return Ok(());
        }

        let content = fs::read_to_string(&index_file).await?;
        let entries: Vec<CacheEntry> = serde_json::from_str(&content).map_err(|e| {
            StorageError::SerializationError(format!("Failed to load cache index: {e}"))
        })?;

        let mut cache = self.cache_index.write().await;
        let mut size = self.current_size.write().await;

        for entry in entries {
            // Check if cached file still exists
            if entry.file_path.exists() {
                *size += entry.size;
                cache.put(entry.key.clone(), entry);
            }
        }

        info!(
            "Loaded cache index with {} entries ({} bytes)",
            cache.len(),
            *size
        );
        Ok(())
    }

    /// Save cache index to disk
    #[allow(dead_code)]
    async fn save_index(&self) -> Result<()> {
        let cache = self.cache_index.read().await;
        let entries: Vec<CacheEntry> = cache.iter().map(|(_, v)| v.clone()).collect();

        let json = serde_json::to_string_pretty(&entries).map_err(|e| {
            StorageError::SerializationError(format!("Failed to serialize cache index: {e}"))
        })?;

        let index_file = self.config.cache_dir.join("index.json");
        fs::write(&index_file, json).await?;

        Ok(())
    }

    /// Get cache file path for a key
    fn get_cache_path(&self, key: &str) -> PathBuf {
        let hash = Sha256::digest(key.as_bytes());
        let hash_str = hex::encode(hash);
        self.config.cache_dir.join(&hash_str[..2]).join(&hash_str)
    }

    /// Check if entry is in cache and valid
    async fn is_cached(&self, key: &str) -> Option<CacheEntry> {
        let mut cache = self.cache_index.write().await;
        let entry = cache.get(key)?;

        // Check TTL if enabled
        if let Some(ttl_secs) = self.config.ttl_seconds {
            if let Ok(elapsed) = entry.created_at.elapsed() {
                if elapsed.as_secs() > ttl_secs {
                    // Entry expired
                    let removed = cache.pop(key);
                    if let Some(entry) = removed {
                        // Delete cached file
                        if entry.file_path.exists() {
                            let _ = fs::remove_file(&entry.file_path).await;
                        }
                        let mut size = self.current_size.write().await;
                        *size = size.saturating_sub(entry.size);
                    }
                    return None;
                }
            }
        }

        // Update access time
        let mut entry = entry.clone();
        entry.last_accessed = std::time::SystemTime::now();
        entry.access_count += 1;

        Some(entry)
    }

    /// Add entry to cache
    async fn add_to_cache(&self, key: &str, data: &[u8], etag: Option<String>) -> Result<()> {
        let cache_path = self.get_cache_path(key);

        // Create parent directory
        if let Some(parent) = cache_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Write data to cache file
        fs::write(&cache_path, data).await?;

        let size = data.len() as u64;

        // Make room if necessary
        self.make_room(size).await?;

        // Add to index
        let entry = CacheEntry {
            key: key.to_string(),
            file_path: cache_path,
            size,
            access_count: 0,
            created_at: std::time::SystemTime::now(),
            last_accessed: std::time::SystemTime::now(),
            etag,
        };

        {
            let mut cache = self.cache_index.write().await;
            cache.put(key.to_string(), entry);
        }

        {
            let mut current_size = self.current_size.write().await;
            *current_size += size;
        }

        debug!("Added {} to cache ({} bytes)", key, size);
        Ok(())
    }

    /// Make room in cache for new entry
    async fn make_room(&self, required_size: u64) -> Result<()> {
        let mut current_size = self.current_size.write().await;

        while *current_size + required_size > self.config.max_size {
            let evicted = {
                let mut cache = self.cache_index.write().await;
                cache.pop_lru()
            };

            if let Some((key, entry)) = evicted {
                // Delete cached file
                if entry.file_path.exists() {
                    fs::remove_file(&entry.file_path).await?;
                }

                *current_size = current_size.saturating_sub(entry.size);
                debug!("Evicted {} from cache ({} bytes)", key, entry.size);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Invalidate cache entry
    pub async fn invalidate(&self, key: &str) -> Result<()> {
        let mut cache = self.cache_index.write().await;

        if let Some(entry) = cache.pop(key) {
            // Delete cached file
            if entry.file_path.exists() {
                fs::remove_file(&entry.file_path).await?;
            }

            let mut size = self.current_size.write().await;
            *size = size.saturating_sub(entry.size);

            info!("Invalidated cache entry: {}", key);
        }

        Ok(())
    }

    /// Clear entire cache
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing cache");

        let mut cache = self.cache_index.write().await;

        // Delete all cached files
        while let Some((_, entry)) = cache.pop_lru() {
            if entry.file_path.exists() {
                let _ = fs::remove_file(&entry.file_path).await;
            }
        }

        {
            let mut size = self.current_size.write().await;
            *size = 0;
        }

        info!("Cache cleared");
        Ok(())
    }

    /// Flush dirty entries (for write-back policy)
    pub async fn flush(&self) -> Result<()> {
        if self.config.write_policy != WritePolicy::WriteBack {
            return Ok(());
        }

        info!("Flushing dirty cache entries");

        let dirty = {
            let mut entries = self.dirty_entries.write().await;
            std::mem::take(&mut *entries)
        };

        let dirty_count = dirty.len();

        for (key, data) in dirty {
            let stream: ByteStream = Box::pin(stream::once(async move { Ok(Bytes::from(data)) }));

            let options = UploadOptions::default();

            self.storage
                .upload_stream(&key, stream, None, options)
                .await?;

            debug!("Flushed dirty entry: {}", key);
        }

        info!("Flushed {} dirty entries", dirty_count);
        Ok(())
    }

    /// Get cache statistics
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache_index.read().await;
        let size = self.current_size.read().await;
        let dirty = self.dirty_entries.read().await;

        CacheStats {
            entries: cache.len(),
            size_bytes: *size,
            max_size_bytes: self.config.max_size,
            dirty_entries: dirty.len(),
            hit_rate: 0.0, // Would need to track hits/misses
        }
    }

    /// Prefetch related objects
    async fn prefetch<'a>(&self, key: &'a str) -> Result<()>
    where
        S: 'a,
    {
        if !self.config.prefetch_enabled {
            return Ok(());
        }

        debug!("Prefetching related objects for: {}", key);

        // Get prefix from key
        let prefix = if let Some(idx) = key.rfind('/') {
            &key[..=idx]
        } else {
            ""
        };

        // List objects with same prefix
        let list_options = ListOptions {
            prefix: Some(prefix.to_string()),
            max_results: Some(self.config.prefetch_size + 1),
            ..Default::default()
        };

        let result = self.storage.list_objects(list_options).await?;

        // Prefetch first few objects
        for obj in result.objects.iter().take(self.config.prefetch_size) {
            if obj.key != key {
                // Download and cache in background
                let storage = self.storage.clone();
                let obj_key = obj.key.clone();
                let cache_path = self.get_cache_path(&obj.key);
                let _etag = obj.etag.clone();

                tokio::spawn(async move {
                    if let Ok(mut stream) = storage
                        .download_stream(&obj_key, DownloadOptions::default())
                        .await
                    {
                        let mut data = Vec::new();
                        while let Some(Ok(chunk)) = futures::StreamExt::next(&mut stream).await {
                            data.extend_from_slice(&chunk);
                        }

                        if let Some(parent) = cache_path.parent() {
                            let _ = fs::create_dir_all(parent).await;
                        }
                        let _ = fs::write(&cache_path, &data).await;
                    }
                });
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<S: CloudStorage + Send + Sync> CloudStorage for CachedStorage<S> {
    async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        size: Option<u64>,
        options: UploadOptions,
    ) -> Result<String> {
        match self.config.write_policy {
            WritePolicy::WriteThrough => {
                // Upload to storage and cache simultaneously
                let mut chunks = Vec::new();
                let mut stream = stream;

                while let Some(result) = futures::StreamExt::next(&mut stream).await {
                    let chunk = result?;
                    chunks.extend_from_slice(&chunk);
                }

                let data = chunks.clone();

                let upload_stream: ByteStream =
                    Box::pin(stream::once(async move { Ok(Bytes::from(chunks)) }));

                let etag = self
                    .storage
                    .upload_stream(key, upload_stream, size, options)
                    .await?;

                // Add to cache
                self.add_to_cache(key, &data, Some(etag.clone())).await?;

                Ok(etag)
            }
            WritePolicy::WriteBack => {
                // Write to cache first
                let mut chunks = Vec::new();
                let mut stream = stream;

                while let Some(result) = futures::StreamExt::next(&mut stream).await {
                    let chunk = result?;
                    chunks.extend_from_slice(&chunk);
                }

                // Add to dirty entries
                {
                    let mut dirty = self.dirty_entries.write().await;
                    dirty.insert(key.to_string(), chunks.clone());
                }

                // Add to cache
                self.add_to_cache(key, &chunks, None).await?;

                Ok("cached".to_string())
            }
            WritePolicy::WriteAround => {
                // Write only to storage
                self.storage.upload_stream(key, stream, size, options).await
            }
        }
    }

    async fn upload_file(
        &self,
        key: &str,
        file_path: &Path,
        options: UploadOptions,
    ) -> Result<String> {
        let etag = self.storage.upload_file(key, file_path, options).await?;

        // Add to cache if write-through
        if self.config.write_policy == WritePolicy::WriteThrough {
            let data = fs::read(file_path).await?;
            self.add_to_cache(key, &data, Some(etag.clone())).await?;
        }

        Ok(etag)
    }

    async fn download_stream(&self, key: &str, options: DownloadOptions) -> Result<ByteStream> {
        // Check cache first
        if let Some(entry) = self.is_cached(key).await {
            debug!("Cache hit: {}", key);

            let data = fs::read(&entry.file_path).await?;
            let stream = stream::once(async move { Ok(Bytes::from(data)) });
            return Ok(Box::pin(stream));
        }

        debug!("Cache miss: {}", key);

        // Download from storage
        let mut stream = self.storage.download_stream(key, options).await?;
        let mut data = Vec::new();

        while let Some(result) = futures::StreamExt::next(&mut stream).await {
            let chunk = result?;
            data.extend_from_slice(&chunk);
        }

        // Add to cache
        self.add_to_cache(key, &data, None).await?;

        // Prefetch if enabled
        let _ = self.prefetch(key).await;

        let result_stream = stream::once(async move { Ok(Bytes::from(data)) });
        Ok(Box::pin(result_stream))
    }

    async fn download_file(
        &self,
        key: &str,
        file_path: &Path,
        options: DownloadOptions,
    ) -> Result<()> {
        // Check cache first
        if let Some(entry) = self.is_cached(key).await {
            debug!("Cache hit: {}", key);
            fs::copy(&entry.file_path, file_path).await?;
            return Ok(());
        }

        debug!("Cache miss: {}", key);

        // Download from storage
        self.storage.download_file(key, file_path, options).await?;

        // Add to cache
        let data = fs::read(file_path).await?;
        self.add_to_cache(key, &data, None).await?;

        Ok(())
    }

    async fn get_metadata(&self, key: &str) -> Result<ObjectMetadata> {
        self.storage.get_metadata(key).await
    }

    async fn delete_object(&self, key: &str) -> Result<()> {
        // Invalidate cache
        self.invalidate(key).await?;

        // Delete from storage
        self.storage.delete_object(key).await
    }

    async fn delete_objects(&self, keys: &[String]) -> Result<Vec<Result<()>>> {
        // Invalidate cache for all keys
        for key in keys {
            self.invalidate(key).await?;
        }

        // Delete from storage
        self.storage.delete_objects(keys).await
    }

    async fn list_objects(&self, options: ListOptions) -> Result<ListResult> {
        self.storage.list_objects(options).await
    }

    async fn object_exists(&self, key: &str) -> Result<bool> {
        self.storage.object_exists(key).await
    }

    async fn copy_object(&self, source_key: &str, dest_key: &str) -> Result<()> {
        self.storage.copy_object(source_key, dest_key).await
    }

    async fn generate_presigned_url(&self, key: &str, expiration_secs: u64) -> Result<String> {
        self.storage
            .generate_presigned_url(key, expiration_secs)
            .await
    }

    async fn generate_presigned_upload_url(
        &self,
        key: &str,
        expiration_secs: u64,
    ) -> Result<String> {
        self.storage
            .generate_presigned_upload_url(key, expiration_secs)
            .await
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub size_bytes: u64,
    pub max_size_bytes: u64,
    pub dirty_entries: usize,
    pub hit_rate: f64,
}

// Implement Serialize/Deserialize for CacheEntry
impl serde::Serialize for CacheEntry {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("CacheEntry", 7)?;
        state.serialize_field("key", &self.key)?;
        state.serialize_field("file_path", &self.file_path)?;
        state.serialize_field("size", &self.size)?;
        state.serialize_field("access_count", &self.access_count)?;
        state.serialize_field(
            "created_at",
            &self
                .created_at
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        )?;
        state.serialize_field(
            "last_accessed",
            &self
                .last_accessed
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        )?;
        state.serialize_field("etag", &self.etag)?;
        state.end()
    }
}

impl<'de> serde::Deserialize<'de> for CacheEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            key: String,
            file_path: PathBuf,
            size: u64,
            access_count: u64,
            created_at: u64,
            last_accessed: u64,
            etag: Option<String>,
        }

        let helper = Helper::deserialize(deserializer)?;

        Ok(CacheEntry {
            key: helper.key,
            file_path: helper.file_path,
            size: helper.size,
            access_count: helper.access_count,
            created_at: std::time::UNIX_EPOCH + std::time::Duration::from_secs(helper.created_at),
            last_accessed: std::time::UNIX_EPOCH
                + std::time::Duration::from_secs(helper.last_accessed),
            etag: helper.etag,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.max_size, DEFAULT_CACHE_SIZE);
        assert_eq!(config.max_entries, DEFAULT_CACHE_ENTRIES);
        assert_eq!(config.write_policy, WritePolicy::WriteThrough);
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
    }

    #[test]
    fn test_write_policy() {
        assert_eq!(WritePolicy::WriteThrough, WritePolicy::WriteThrough);
        assert_ne!(WritePolicy::WriteThrough, WritePolicy::WriteBack);
    }

    #[test]
    fn test_eviction_policy() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
    }
}
