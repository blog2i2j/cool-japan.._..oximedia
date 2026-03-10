//! Pipeline cache management for faster startup and reduced compilation overhead
//!
//! This module provides caching mechanisms for compiled pipelines and shaders,
//! allowing them to be reused across application runs.

use crate::{GpuError, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// Cache entry metadata
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cache key (usually a hash of the shader source)
    key: String,
    /// Timestamp when the entry was created
    timestamp: SystemTime,
    /// Size in bytes
    size: usize,
}

/// Pipeline cache for storing compiled pipelines
pub struct PipelineCache {
    cache_dir: PathBuf,
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_cache_size: usize,
    enabled: bool,
}

impl PipelineCache {
    /// Create a new pipeline cache
    ///
    /// # Arguments
    ///
    /// * `cache_dir` - Directory to store cached pipelines
    /// * `max_cache_size` - Maximum cache size in bytes (0 = unlimited)
    pub fn new(cache_dir: impl AsRef<Path>, max_cache_size: usize) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        // Create cache directory if it doesn't exist
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).map_err(|e| {
                GpuError::Internal(format!("Failed to create cache directory: {e}"))
            })?;
        }

        let cache = Self {
            cache_dir,
            entries: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size,
            enabled: true,
        };

        // Load existing cache entries
        cache.load_cache_index()?;

        Ok(cache)
    }

    /// Create a default pipeline cache in the system cache directory
    pub fn default_cache() -> Result<Self> {
        let cache_dir = Self::default_cache_dir()?;
        Self::new(cache_dir, 100 * 1024 * 1024) // 100 MB default
    }

    /// Get the default cache directory for the current platform
    fn default_cache_dir() -> Result<PathBuf> {
        let cache_dir = if let Some(cache_dir) = dirs::cache_dir() {
            cache_dir.join("oximedia").join("gpu_cache")
        } else {
            PathBuf::from(".oximedia_cache")
        };

        Ok(cache_dir)
    }

    /// Load the cache index from disk
    fn load_cache_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");

        if !index_path.exists() {
            return Ok(());
        }

        let mut file = File::open(&index_path)
            .map_err(|e| GpuError::Internal(format!("Failed to open cache index: {e}")))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| GpuError::Internal(format!("Failed to read cache index: {e}")))?;

        // Parse JSON index (simplified - in production, use serde_json)
        // For now, just create empty index
        Ok(())
    }

    /// Save the cache index to disk
    #[allow(dead_code)]
    fn save_cache_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");

        let mut file = File::create(&index_path)
            .map_err(|e| GpuError::Internal(format!("Failed to create cache index: {e}")))?;

        // Write JSON index (simplified)
        file.write_all(b"{}")
            .map_err(|e| GpuError::Internal(format!("Failed to write cache index: {e}")))?;

        Ok(())
    }

    /// Get a cached pipeline by key
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key (usually shader source hash)
    ///
    /// # Returns
    ///
    /// Cached pipeline data if found, None otherwise
    #[must_use]
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        if !self.enabled {
            return None;
        }

        let entries = self.entries.read();
        if !entries.contains_key(key) {
            return None;
        }

        let cache_path = self.cache_dir.join(format!("{key}.bin"));
        if !cache_path.exists() {
            return None;
        }

        let mut file = File::open(&cache_path).ok()?;
        let mut data = Vec::new();
        file.read_to_end(&mut data).ok()?;

        Some(data)
    }

    /// Store a pipeline in the cache
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key
    /// * `data` - Pipeline data to cache
    pub fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        // Check cache size limit
        if self.max_cache_size > 0 {
            let current_size = self.total_cache_size();
            if current_size + data.len() > self.max_cache_size {
                self.evict_oldest()?;
            }
        }

        let cache_path = self.cache_dir.join(format!("{key}.bin"));
        let mut file = File::create(&cache_path)
            .map_err(|e| GpuError::Internal(format!("Failed to create cache file: {e}")))?;

        file.write_all(data)
            .map_err(|e| GpuError::Internal(format!("Failed to write cache file: {e}")))?;

        // Update cache index
        let mut entries = self.entries.write();
        entries.insert(
            key.to_string(),
            CacheEntry {
                key: key.to_string(),
                timestamp: SystemTime::now(),
                size: data.len(),
            },
        );

        Ok(())
    }

    /// Remove a cached pipeline
    pub fn remove(&self, key: &str) -> Result<()> {
        let cache_path = self.cache_dir.join(format!("{key}.bin"));

        if cache_path.exists() {
            fs::remove_file(&cache_path)
                .map_err(|e| GpuError::Internal(format!("Failed to remove cache file: {e}")))?;
        }

        let mut entries = self.entries.write();
        entries.remove(key);

        Ok(())
    }

    /// Clear the entire cache
    pub fn clear(&self) -> Result<()> {
        let entries: Vec<String> = {
            let entries = self.entries.read();
            entries.keys().cloned().collect()
        };

        for key in entries {
            self.remove(&key)?;
        }

        Ok(())
    }

    /// Get the total cache size in bytes
    #[must_use]
    pub fn total_cache_size(&self) -> usize {
        let entries = self.entries.read();
        entries.values().map(|e| e.size).sum()
    }

    /// Get the number of cached entries
    #[must_use]
    pub fn entry_count(&self) -> usize {
        let entries = self.entries.read();
        entries.len()
    }

    /// Evict the oldest cache entry
    fn evict_oldest(&self) -> Result<()> {
        let oldest_key = {
            let entries = self.entries.read();
            entries
                .values()
                .min_by_key(|e| e.timestamp)
                .map(|e| e.key.clone())
        };

        if let Some(key) = oldest_key {
            self.remove(&key)?;
        }

        Ok(())
    }

    /// Enable or disable the cache
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if the cache is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entry_count: self.entry_count(),
            total_size: self.total_cache_size(),
            max_size: self.max_cache_size,
            enabled: self.enabled,
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cached entries
    pub entry_count: usize,
    /// Total cache size in bytes
    pub total_size: usize,
    /// Maximum cache size in bytes
    pub max_size: usize,
    /// Whether the cache is enabled
    pub enabled: bool,
}

impl CacheStats {
    /// Get cache utilization as a percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_size == 0 {
            0.0
        } else {
            (self.total_size as f64 / self.max_size as f64) * 100.0
        }
    }

    /// Get total size in megabytes
    #[must_use]
    pub fn size_mb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0)
    }
}

/// Shader cache for storing compiled shader modules
pub struct ShaderCache {
    cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl ShaderCache {
    /// Create a new shader cache
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a cached shader
    #[must_use]
    pub fn get(&self, source_hash: &str) -> Option<Vec<u8>> {
        let cache = self.cache.read();
        cache.get(source_hash).cloned()
    }

    /// Store a shader in the cache
    pub fn put(&self, source_hash: String, compiled_shader: Vec<u8>) {
        let mut cache = self.cache.write();
        cache.insert(source_hash, compiled_shader);
    }

    /// Clear the shader cache
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.clear();
    }

    /// Get the number of cached shaders
    #[must_use]
    pub fn size(&self) -> usize {
        let cache = self.cache.read();
        cache.len()
    }
}

impl Default for ShaderCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_cache_creation() {
        let temp_dir = std::env::temp_dir().join("oximedia_cache_test");
        let cache = PipelineCache::new(&temp_dir, 1024 * 1024)
            .expect("pipeline cache creation should succeed");

        assert!(cache.is_enabled());
        assert_eq!(cache.entry_count(), 0);
        assert_eq!(cache.total_cache_size(), 0);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_pipeline_cache_put_get() {
        let temp_dir = std::env::temp_dir().join("oximedia_cache_test_2");
        let cache = PipelineCache::new(&temp_dir, 1024 * 1024)
            .expect("pipeline cache creation should succeed");

        let key = "test_shader";
        let data = vec![1, 2, 3, 4, 5];

        cache.put(key, &data).expect("cache put should succeed");
        let retrieved = cache.get(key).expect("cache get should return stored data");

        assert_eq!(data, retrieved);
        assert_eq!(cache.entry_count(), 1);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_shader_cache() {
        let cache = ShaderCache::new();
        assert_eq!(cache.size(), 0);

        cache.put("shader1".to_string(), vec![1, 2, 3]);
        assert_eq!(cache.size(), 1);

        let shader = cache.get("shader1");
        assert_eq!(shader, Some(vec![1, 2, 3]));

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = std::env::temp_dir().join("oximedia_cache_test_3");
        let cache =
            PipelineCache::new(&temp_dir, 1024).expect("pipeline cache creation should succeed");

        cache
            .put("key1", &[0u8; 100])
            .expect("cache put should succeed");
        cache
            .put("key2", &[0u8; 200])
            .expect("cache put should succeed");

        let stats = cache.stats();
        assert_eq!(stats.entry_count, 2);
        assert_eq!(stats.total_size, 300);

        // Clean up
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
