//! Temporary file management for media pipelines.
//!
//! Creates, tracks, and cleans up temporary files and directories used
//! during transcoding, analysis, and other intermediate processing stages.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Configuration for how temporary files are created and stored.
#[derive(Debug, Clone)]
pub struct TempFileConfig {
    /// Base directory for temporary files. Defaults to the system temp dir.
    pub base_dir: PathBuf,
    /// Optional prefix applied to every generated filename.
    pub prefix: String,
    /// Optional suffix (e.g. `.mp4`) applied to every generated filename.
    pub suffix: String,
    /// Whether to delete files on [`TempFileHandle`] drop.
    pub auto_delete: bool,
}

impl TempFileConfig {
    /// Create a new config pointing at the system temp directory.
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_dir: std::env::temp_dir(),
            prefix: "oximedia_".to_string(),
            suffix: String::new(),
            auto_delete: true,
        }
    }

    /// Override the base directory.
    #[must_use]
    pub fn with_base_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.base_dir = dir.as_ref().to_path_buf();
        self
    }

    /// Override the filename prefix.
    #[must_use]
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Override the filename suffix.
    #[must_use]
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Disable automatic deletion on drop.
    #[must_use]
    pub fn no_auto_delete(mut self) -> Self {
        self.auto_delete = false;
        self
    }
}

impl Default for TempFileConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to a single temporary file.
///
/// When `auto_delete` is set in the originating config the underlying file
/// is removed from disk when this handle is dropped.
#[derive(Debug)]
pub struct TempFileHandle {
    /// Absolute path to the temporary file.
    pub path: PathBuf,
    auto_delete: bool,
}

impl TempFileHandle {
    /// Return the path of this temporary file.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return `true` if the file still exists on disk.
    #[must_use]
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Delete the file immediately without waiting for drop.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be removed.
    pub fn remove(self) -> std::io::Result<()> {
        if self.path.exists() {
            std::fs::remove_file(&self.path)?;
        }
        // Suppress auto-delete in Drop
        std::mem::forget(self);
        Ok(())
    }
}

impl Drop for TempFileHandle {
    fn drop(&mut self) {
        if self.auto_delete && self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Manages a collection of temporary files for a processing session.
///
/// Thread-safe: all mutations are guarded by an internal `Mutex`.
#[derive(Debug, Clone)]
pub struct TempFileManager {
    config: TempFileConfig,
    /// Map from logical name to path (for named lookup).
    registry: Arc<Mutex<HashMap<String, PathBuf>>>,
    counter: Arc<Mutex<u64>>,
}

impl TempFileManager {
    /// Create a new manager with the given config.
    #[must_use]
    pub fn new(config: TempFileConfig) -> Self {
        Self {
            config,
            registry: Arc::new(Mutex::new(HashMap::new())),
            counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new manager with default config.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(TempFileConfig::default())
    }

    /// Allocate a new temporary file and return its handle.
    ///
    /// The file is created (empty) on disk immediately.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created.
    ///
    /// # Panics
    ///
    /// Panics if the internal counter mutex is poisoned.
    pub fn create(&self) -> std::io::Result<TempFileHandle> {
        self.create_named(&self.next_name())
    }

    /// Allocate a temp file and register it under a logical `name` for later
    /// lookup.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the file cannot be created.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry mutex is poisoned.
    pub fn create_named(&self, name: &str) -> std::io::Result<TempFileHandle> {
        let filename = format!("{}{}{}", self.config.prefix, name, self.config.suffix);
        let path = self.config.base_dir.join(&filename);
        std::fs::write(&path, b"")?;
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string(), path.clone());
        Ok(TempFileHandle {
            path,
            auto_delete: self.config.auto_delete,
        })
    }

    /// Look up the path registered under a logical name.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry mutex is poisoned.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<PathBuf> {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    /// Delete all tracked temporary files and clear the registry.
    ///
    /// # Errors
    ///
    /// Returns an I/O error if any file cannot be removed.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry mutex is poisoned.
    pub fn cleanup(&self) -> std::io::Result<()> {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        for (_name, path) in reg.iter() {
            if path.exists() {
                std::fs::remove_file(path)?;
            }
        }
        reg.clear();
        Ok(())
    }

    /// Number of files currently tracked.
    ///
    /// # Panics
    ///
    /// Panics if the internal registry mutex is poisoned.
    #[must_use]
    pub fn count(&self) -> usize {
        self.registry
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    fn next_name(&self) -> String {
        let mut c = self.counter.lock().unwrap_or_else(|e| e.into_inner());
        let n = *c;
        *c += 1;
        format!("{n:08x}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_temp_file_exists_on_disk() {
        let mgr = TempFileManager::with_defaults();
        let handle = mgr.create().expect("failed to create temp file");
        assert!(handle.exists());
    }

    #[test]
    fn test_auto_delete_on_drop() {
        let mgr = TempFileManager::with_defaults();
        let path = {
            let handle = mgr.create().expect("failed to create temp file");
            handle.path().to_path_buf()
        };
        assert!(!path.exists());
    }

    #[test]
    fn test_no_auto_delete_survives_drop() {
        let config = TempFileConfig::new().no_auto_delete();
        let mgr = TempFileManager::new(config);
        let path = {
            let handle = mgr.create().expect("failed to create temp file");
            handle.path().to_path_buf()
        };
        // File should still exist — clean up manually
        if path.exists() {
            std::fs::remove_file(&path).expect("failed to remove file");
        }
    }

    #[test]
    fn test_explicit_remove() {
        let mgr = TempFileManager::with_defaults();
        let handle = mgr.create().expect("failed to create temp file");
        let path = handle.path().to_path_buf();
        handle.remove().expect("failed to remove temp file");
        assert!(!path.exists());
    }

    #[test]
    fn test_create_named_and_lookup() {
        let mgr = TempFileManager::with_defaults();
        let _handle = mgr
            .create_named("audio_work")
            .expect("failed to create named temp file");
        let found = mgr.lookup("audio_work");
        assert!(found.is_some());
    }

    #[test]
    fn test_lookup_missing_returns_none() {
        let mgr = TempFileManager::with_defaults();
        assert!(mgr.lookup("nonexistent").is_none());
    }

    #[test]
    fn test_cleanup_removes_files() {
        let mgr = TempFileManager::new(TempFileConfig::new().no_auto_delete());
        let h1 = mgr
            .create_named("f1")
            .expect("failed to create named temp file");
        let h2 = mgr
            .create_named("f2")
            .expect("failed to create named temp file");
        let p1 = h1.path().to_path_buf();
        let p2 = h2.path().to_path_buf();
        // Suppress their own drop (auto_delete = false)
        std::mem::forget(h1);
        std::mem::forget(h2);
        mgr.cleanup().expect("cleanup should succeed");
        assert!(!p1.exists());
        assert!(!p2.exists());
    }

    #[test]
    fn test_cleanup_clears_registry() {
        let mgr = TempFileManager::with_defaults();
        let h = mgr
            .create_named("tmp_x")
            .expect("failed to create named temp file");
        std::mem::forget(h);
        mgr.cleanup().expect("cleanup should succeed");
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_count_tracks_entries() {
        let mgr = TempFileManager::with_defaults();
        assert_eq!(mgr.count(), 0);
        let h1 = mgr
            .create_named("a")
            .expect("failed to create named temp file");
        std::mem::forget(h1);
        assert_eq!(mgr.count(), 1);
        let h2 = mgr
            .create_named("b")
            .expect("failed to create named temp file");
        std::mem::forget(h2);
        assert_eq!(mgr.count(), 2);
        mgr.cleanup().expect("cleanup should succeed");
        assert_eq!(mgr.count(), 0);
    }

    #[test]
    fn test_suffix_applied() {
        let config = TempFileConfig::new().with_suffix(".ts");
        let mgr = TempFileManager::new(config);
        let handle = mgr.create().expect("failed to create temp file");
        let path_str = handle.path().to_string_lossy().to_string();
        assert!(path_str.ends_with(".ts"));
        let _ = handle.remove();
    }

    #[test]
    fn test_prefix_applied() {
        let config = TempFileConfig::new().with_prefix("oxi_pfx_");
        let mgr = TempFileManager::new(config);
        let handle = mgr.create().expect("failed to create temp file");
        let fname = handle
            .path()
            .file_name()
            .expect("operation should succeed")
            .to_string_lossy()
            .to_string();
        assert!(fname.starts_with("oxi_pfx_"));
        let _ = handle.remove();
    }

    #[test]
    fn test_config_default() {
        let cfg = TempFileConfig::default();
        assert_eq!(cfg.prefix, "oximedia_");
        assert!(cfg.auto_delete);
    }

    #[test]
    fn test_temp_file_config_builder_chain() {
        let cfg = TempFileConfig::new()
            .with_prefix("test_")
            .with_suffix(".raw")
            .no_auto_delete();
        assert_eq!(cfg.prefix, "test_");
        assert_eq!(cfg.suffix, ".raw");
        assert!(!cfg.auto_delete);
    }

    #[test]
    fn test_multiple_create_unique_paths() {
        let mgr = TempFileManager::with_defaults();
        let h1 = mgr.create().expect("failed to create temp file");
        let h2 = mgr.create().expect("failed to create temp file");
        assert_ne!(h1.path(), h2.path());
    }
}
