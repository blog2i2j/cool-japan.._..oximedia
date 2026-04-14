// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Distributed storage management.

use crate::error::{Error, Result};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncReadExt;

/// Storage backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Local file system
    Local,
    /// Network file system (NFS/SMB)
    NetworkFS,
    /// Object storage (S3-compatible)
    ObjectStorage,
    /// Distributed cache
    DistributedCache,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Backend type
    pub backend: StorageBackend,
    /// Root path
    pub root_path: PathBuf,
    /// Enable compression
    pub enable_compression: bool,
    /// Enable deduplication
    pub enable_dedup: bool,
    /// Cache size limit (bytes)
    pub cache_size_limit: u64,
    /// Endpoint (for object storage)
    pub endpoint: Option<String>,
    /// Access key (for object storage)
    pub access_key: Option<String>,
    /// Secret key (for object storage)
    pub secret_key: Option<String>,
    /// Bucket name (for object storage)
    pub bucket: Option<String>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Local,
            root_path: PathBuf::from("/var/renderfarm/storage"),
            enable_compression: true,
            enable_dedup: true,
            cache_size_limit: 100 * 1024 * 1024 * 1024, // 100 GB
            endpoint: None,
            access_key: None,
            secret_key: None,
            bucket: None,
        }
    }
}

/// File metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    /// File path
    pub path: PathBuf,
    /// File size
    pub size: u64,
    /// Content hash (BLAKE3)
    pub hash: String,
    /// MIME type
    pub mime_type: String,
    /// Compression codec
    pub compression: Option<String>,
    /// Chunk references (for dedup)
    pub chunks: Vec<String>,
}

/// Storage manager
pub struct StorageManager {
    config: StorageConfig,
    file_index: HashMap<PathBuf, FileMetadata>,
    chunk_store: HashMap<String, Vec<u8>>,
}

impl StorageManager {
    /// Create a new storage manager
    #[must_use]
    pub fn new(config: StorageConfig) -> Self {
        Self {
            config,
            file_index: HashMap::new(),
            chunk_store: HashMap::new(),
        }
    }

    /// Initialize storage
    pub async fn initialize(&mut self) -> Result<()> {
        // Create root directory if it doesn't exist
        if !self.config.root_path.exists() {
            fs::create_dir_all(&self.config.root_path).await?;
        }

        // Load file index
        self.load_index().await?;

        Ok(())
    }

    /// Store a file
    pub async fn store_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        data: Vec<u8>,
    ) -> Result<FileMetadata> {
        let path = path.as_ref().to_path_buf();

        // Calculate hash
        let mut hasher = Hasher::new();
        hasher.update(&data);
        let hash = hasher.finalize().to_hex().to_string();

        // Compress if enabled
        let data = if self.config.enable_compression {
            self.compress_data(&data)?
        } else {
            data
        };

        // Deduplicate if enabled
        let chunks = if self.config.enable_dedup {
            self.deduplicate_data(&data)?
        } else {
            // Store as single chunk
            let chunk_hash = format!("chunk_{hash}");
            self.chunk_store.insert(chunk_hash.clone(), data);
            vec![chunk_hash]
        };

        let metadata = FileMetadata {
            path: path.clone(),
            size: chunks
                .iter()
                .map(|id| self.chunk_store.get(id).map_or(0, Vec::len) as u64)
                .sum(),
            hash,
            mime_type: self.detect_mime_type(&path),
            compression: if self.config.enable_compression {
                Some("lz4".to_string())
            } else {
                None
            },
            chunks,
        };

        self.file_index.insert(path, metadata.clone());

        Ok(metadata)
    }

    /// Retrieve a file
    pub async fn retrieve_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<u8>> {
        let path = path.as_ref();

        let metadata = self
            .file_index
            .get(path)
            .ok_or_else(|| Error::AssetNotFound(path.display().to_string()))?;

        // Reconstruct data from chunks
        let mut data = Vec::new();
        for chunk_id in &metadata.chunks {
            if let Some(chunk_data) = self.chunk_store.get(chunk_id) {
                data.extend_from_slice(chunk_data);
            } else {
                return Err(Error::Storage(format!("Chunk not found: {chunk_id}")));
            }
        }

        // Decompress if needed
        let data = if metadata.compression.is_some() {
            self.decompress_data(&data)?
        } else {
            data
        };

        Ok(data)
    }

    /// Delete a file
    pub async fn delete_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        if let Some(metadata) = self.file_index.remove(path) {
            // Remove chunks (if not referenced elsewhere)
            for chunk_id in &metadata.chunks {
                // Check if any other file uses this chunk
                let in_use = self
                    .file_index
                    .values()
                    .any(|m| m.chunks.contains(chunk_id));

                if !in_use {
                    self.chunk_store.remove(chunk_id);
                }
            }
        }

        Ok(())
    }

    /// Check if file exists
    #[must_use]
    pub fn file_exists<P: AsRef<Path>>(&self, path: P) -> bool {
        self.file_index.contains_key(path.as_ref())
    }

    /// Get file metadata
    #[must_use]
    pub fn get_metadata<P: AsRef<Path>>(&self, path: P) -> Option<&FileMetadata> {
        self.file_index.get(path.as_ref())
    }

    /// List files
    #[must_use]
    pub fn list_files(&self) -> Vec<PathBuf> {
        self.file_index.keys().cloned().collect()
    }

    /// Get total storage size
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.chunk_store.values().map(|v| v.len() as u64).sum()
    }

    /// Compress data using lz4_flex (pure Rust replacement for zstd)
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    /// Decompress data using lz4_flex
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        let decompressed = lz4_flex::decompress_size_prepended(data)
            .map_err(|e| Error::Storage(format!("Decompression failed: {e}")))?;
        Ok(decompressed)
    }

    /// Deduplicate data by chunking
    fn deduplicate_data(&mut self, data: &[u8]) -> Result<Vec<String>> {
        const CHUNK_SIZE: usize = 64 * 1024; // 64 KB chunks

        let mut chunk_ids = Vec::new();

        for chunk in data.chunks(CHUNK_SIZE) {
            // Calculate chunk hash
            let mut hasher = Hasher::new();
            hasher.update(chunk);
            let chunk_hash = hasher.finalize().to_hex().to_string();

            // Store chunk if not already present
            if !self.chunk_store.contains_key(&chunk_hash) {
                self.chunk_store.insert(chunk_hash.clone(), chunk.to_vec());
            }

            chunk_ids.push(chunk_hash);
        }

        Ok(chunk_ids)
    }

    /// Detect MIME type
    fn detect_mime_type(&self, path: &Path) -> String {
        if let Some(ext) = path.extension() {
            let ext = ext.to_string_lossy().to_lowercase();
            match ext.as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "exr" => "image/x-exr",
                "mp4" => "video/mp4",
                "mov" => "video/quicktime",
                "wav" => "audio/wav",
                "mp3" => "audio/mpeg",
                _ => "application/octet-stream",
            }
            .to_string()
        } else {
            "application/octet-stream".to_string()
        }
    }

    /// Load file index
    async fn load_index(&mut self) -> Result<()> {
        let index_path = self.config.root_path.join("index.json");

        if index_path.exists() {
            let mut file = fs::File::open(&index_path).await?;
            let mut contents = String::new();
            file.read_to_string(&mut contents).await?;

            self.file_index = serde_json::from_str(&contents)?;
        }

        Ok(())
    }

    /// Save file index
    pub async fn save_index(&self) -> Result<()> {
        let index_path = self.config.root_path.join("index.json");
        let contents = serde_json::to_string_pretty(&self.file_index)?;

        fs::write(&index_path, contents).await?;

        Ok(())
    }

    /// Get storage statistics
    #[must_use]
    pub fn get_stats(&self) -> StorageStats {
        let total_files = self.file_index.len();
        let total_chunks = self.chunk_store.len();
        let total_size = self.total_size();
        let total_logical_size: u64 = self.file_index.values().map(|m| m.size).sum();

        let dedup_ratio = if total_logical_size > 0 {
            total_size as f64 / total_logical_size as f64
        } else {
            1.0
        };

        StorageStats {
            total_files,
            total_chunks,
            total_size,
            total_logical_size,
            dedup_ratio,
            cache_utilization: (total_size as f64 / self.config.cache_size_limit as f64)
                .clamp(0.0, 1.0),
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total files
    pub total_files: usize,
    /// Total chunks
    pub total_chunks: usize,
    /// Total physical size
    pub total_size: u64,
    /// Total logical size (before dedup)
    pub total_logical_size: u64,
    /// Deduplication ratio
    pub dedup_ratio: f64,
    /// Cache utilization (0.0 to 1.0)
    pub cache_utilization: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_manager_creation() {
        let config = StorageConfig::default();
        let manager = StorageManager::new(config);
        assert_eq!(manager.file_index.len(), 0);
    }

    #[tokio::test]
    async fn test_store_and_retrieve_file() -> Result<()> {
        let config = StorageConfig {
            enable_compression: false,
            enable_dedup: false,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        let data = b"Hello, World!".to_vec();
        let path = PathBuf::from("test.txt");

        // Store file
        let metadata = manager.store_file(&path, data.clone()).await?;
        assert_eq!(metadata.path, path);

        // Retrieve file
        let retrieved = manager.retrieve_file(&path).await?;
        assert_eq!(retrieved, data);

        Ok(())
    }

    #[tokio::test]
    async fn test_file_exists() -> Result<()> {
        let config = StorageConfig {
            enable_compression: false,
            enable_dedup: false,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        let path = PathBuf::from("test.txt");
        assert!(!manager.file_exists(&path));

        manager.store_file(&path, b"test".to_vec()).await?;
        assert!(manager.file_exists(&path));

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_file() -> Result<()> {
        let config = StorageConfig {
            enable_compression: false,
            enable_dedup: false,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        let path = PathBuf::from("test.txt");
        manager.store_file(&path, b"test".to_vec()).await?;

        manager.delete_file(&path).await?;
        assert!(!manager.file_exists(&path));

        Ok(())
    }

    #[tokio::test]
    async fn test_compression() -> Result<()> {
        let config = StorageConfig {
            enable_compression: true,
            enable_dedup: false,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        let data = vec![0u8; 10000]; // Highly compressible
        let path = PathBuf::from("test.bin");

        manager.store_file(&path, data.clone()).await?;
        let retrieved = manager.retrieve_file(&path).await?;

        assert_eq!(retrieved, data);
        assert!(manager.total_size() < data.len() as u64);

        Ok(())
    }

    #[tokio::test]
    async fn test_deduplication() -> Result<()> {
        let config = StorageConfig {
            enable_compression: false,
            enable_dedup: true,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        let data = vec![0u8; 100000]; // Same data
        let path1 = PathBuf::from("test1.bin");
        let path2 = PathBuf::from("test2.bin");

        manager.store_file(&path1, data.clone()).await?;
        manager.store_file(&path2, data.clone()).await?;

        // Should use same chunks due to dedup
        let metadata1 = manager
            .get_metadata(&path1)
            .expect("should succeed in test");
        let metadata2 = manager
            .get_metadata(&path2)
            .expect("should succeed in test");

        // Chunks should be identical
        assert_eq!(metadata1.chunks, metadata2.chunks);

        Ok(())
    }

    #[tokio::test]
    async fn test_storage_stats() -> Result<()> {
        let config = StorageConfig {
            enable_compression: false,
            enable_dedup: false,
            ..Default::default()
        };
        let mut manager = StorageManager::new(config);

        manager.store_file("file1.txt", b"test".to_vec()).await?;
        manager.store_file("file2.txt", b"test2".to_vec()).await?;

        let stats = manager.get_stats();
        assert_eq!(stats.total_files, 2);

        Ok(())
    }

    #[test]
    fn test_mime_type_detection() {
        let config = StorageConfig::default();
        let manager = StorageManager::new(config);

        assert_eq!(
            manager.detect_mime_type(Path::new("test.jpg")),
            "image/jpeg"
        );
        assert_eq!(manager.detect_mime_type(Path::new("test.png")), "image/png");
        assert_eq!(manager.detect_mime_type(Path::new("test.mp4")), "video/mp4");
    }
}
