//! Output management for packaged content.

use crate::config::OutputConfig;
use crate::error::{PackagerError, PackagerResult};
use camino::{Utf8Path, Utf8PathBuf};
use std::collections::HashMap;
#[cfg(not(feature = "s3"))]
use tracing::warn;
use tracing::{debug, info};

/// Output directory structure.
pub struct OutputStructure {
    /// Root output directory.
    pub root: Utf8PathBuf,
    /// Variant-specific directories.
    pub variants: HashMap<String, Utf8PathBuf>,
    /// Manifest directory.
    pub manifests: Utf8PathBuf,
}

impl OutputStructure {
    /// Create a new output structure.
    #[must_use]
    pub fn new(root: Utf8PathBuf) -> Self {
        let manifests = root.join("manifests");

        Self {
            root,
            variants: HashMap::new(),
            manifests,
        }
    }

    /// Add a variant directory.
    pub fn add_variant(&mut self, name: &str, bitrate: u32) -> Utf8PathBuf {
        let variant_dir = self.root.join(format!("{name}_{bitrate}"));
        self.variants.insert(name.to_string(), variant_dir.clone());
        variant_dir
    }

    /// Get variant directory.
    #[must_use]
    pub fn get_variant(&self, name: &str) -> Option<&Utf8PathBuf> {
        self.variants.get(name)
    }

    /// Create all directories.
    pub async fn create_directories(&self) -> PackagerResult<()> {
        tokio::fs::create_dir_all(&self.root).await?;
        tokio::fs::create_dir_all(&self.manifests).await?;

        for variant_dir in self.variants.values() {
            tokio::fs::create_dir_all(variant_dir).await?;
        }

        info!("Created output directory structure at {}", self.root);

        Ok(())
    }
}

/// Output manager for handling packaged content output.
pub struct OutputManager {
    config: OutputConfig,
    structure: OutputStructure,
}

impl OutputManager {
    /// Create a new output manager.
    pub fn new(config: OutputConfig) -> PackagerResult<Self> {
        let root = Utf8PathBuf::try_from(config.directory.clone())
            .map_err(|_| PackagerError::invalid_config("Invalid output directory path"))?;

        let structure = OutputStructure::new(root);

        Ok(Self { config, structure })
    }

    /// Get the output structure.
    #[must_use]
    pub fn structure(&self) -> &OutputStructure {
        &self.structure
    }

    /// Get mutable output structure.
    pub fn structure_mut(&mut self) -> &mut OutputStructure {
        &mut self.structure
    }

    /// Initialize output directories.
    pub async fn initialize(&self) -> PackagerResult<()> {
        self.structure.create_directories().await?;
        Ok(())
    }

    /// Write file to output.
    pub async fn write_file(&self, relative_path: &Utf8Path, content: &[u8]) -> PackagerResult<()> {
        let full_path = self.structure.root.join(relative_path);

        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        tokio::fs::write(&full_path, content).await?;

        debug!("Wrote file: {} ({} bytes)", full_path, content.len());

        Ok(())
    }

    /// Read file from output.
    pub async fn read_file(&self, relative_path: &Utf8Path) -> PackagerResult<Vec<u8>> {
        let full_path = self.structure.root.join(relative_path);
        let content = tokio::fs::read(&full_path).await?;

        Ok(content)
    }

    /// Delete file from output.
    pub async fn delete_file(&self, relative_path: &Utf8Path) -> PackagerResult<()> {
        let full_path = self.structure.root.join(relative_path);

        if full_path.exists() {
            tokio::fs::remove_file(&full_path).await?;
            debug!("Deleted file: {}", full_path);
        }

        Ok(())
    }

    /// List files in directory.
    pub async fn list_files(&self, relative_path: &Utf8Path) -> PackagerResult<Vec<Utf8PathBuf>> {
        let full_path = self.structure.root.join(relative_path);
        let mut files = Vec::new();

        let mut entries = tokio::fs::read_dir(&full_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            if let Ok(path) = Utf8PathBuf::try_from(entry.path()) {
                files.push(path);
            }
        }

        Ok(files)
    }

    /// Clean up old segments.
    pub async fn cleanup_old_segments(
        &self,
        variant: &str,
        max_segments: usize,
    ) -> PackagerResult<()> {
        let variant_dir = self
            .structure
            .get_variant(variant)
            .ok_or_else(|| PackagerError::invalid_config("Variant not found"))?;

        let mut segment_files = Vec::new();
        let mut entries = tokio::fs::read_dir(variant_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "ts" || ext == "m4s" {
                    if let Ok(metadata) = entry.metadata().await {
                        segment_files.push((path.clone(), metadata.modified()?));
                    }
                }
            }
        }

        // Sort by modification time
        segment_files.sort_by_key(|(_, time)| *time);

        // Delete old segments
        if segment_files.len() > max_segments {
            let to_delete = segment_files.len() - max_segments;
            for (path, _) in segment_files.iter().take(to_delete) {
                tokio::fs::remove_file(path).await?;
                debug!("Deleted old segment: {}", path.display());
            }

            info!(
                "Cleaned up {} old segments for variant {}",
                to_delete, variant
            );
        }

        Ok(())
    }

    /// Upload to S3 (if enabled).
    #[cfg(feature = "s3")]
    pub async fn upload_to_s3(&self, relative_path: &Utf8Path) -> PackagerResult<()> {
        use aws_sdk_s3::Client;

        if !self.config.s3_upload {
            return Ok(());
        }

        let bucket =
            self.config.s3_bucket.as_ref().ok_or_else(|| {
                PackagerError::UploadFailed("S3 bucket not configured".to_string())
            })?;

        let key = if let Some(prefix) = &self.config.s3_prefix {
            format!("{prefix}/{relative_path}")
        } else {
            relative_path.to_string()
        };

        let full_path = self.structure.root.join(relative_path);
        let content = tokio::fs::read(&full_path).await?;

        let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .load()
            .await;
        let client = Client::new(&config);

        client
            .put_object()
            .bucket(bucket)
            .key(&key)
            .body(content.into())
            .send()
            .await
            .map_err(|e| PackagerError::UploadFailed(format!("S3 upload failed: {e}")))?;

        info!("Uploaded {} to S3: s3://{}/{}", relative_path, bucket, key);

        // Delete local file if not keeping
        if !self.config.keep_local {
            tokio::fs::remove_file(&full_path).await?;
            debug!("Deleted local file after upload: {}", full_path);
        }

        Ok(())
    }

    /// Upload to S3 (stub when feature is disabled).
    #[cfg(not(feature = "s3"))]
    pub async fn upload_to_s3(&self, _relative_path: &Utf8Path) -> PackagerResult<()> {
        warn!("S3 upload requested but s3 feature is not enabled");
        Ok(())
    }

    /// Get full path for relative path.
    #[must_use]
    pub fn get_full_path(&self, relative_path: &Utf8Path) -> Utf8PathBuf {
        self.structure.root.join(relative_path)
    }

    /// Get base URL for manifests.
    #[must_use]
    pub fn base_url(&self) -> Option<&str> {
        self.config.base_url.as_deref()
    }
}

/// Segment cleanup policy.
#[derive(Debug, Clone, Copy)]
pub enum CleanupPolicy {
    /// Keep all segments.
    KeepAll,
    /// Keep a maximum number of segments.
    MaxSegments(usize),
    /// Keep segments within a time window.
    TimeWindow(std::time::Duration),
}

/// Segment cleanup manager.
pub struct CleanupManager {
    policy: CleanupPolicy,
    output_manager: OutputManager,
}

impl CleanupManager {
    /// Create a new cleanup manager.
    #[must_use]
    pub fn new(policy: CleanupPolicy, output_manager: OutputManager) -> Self {
        Self {
            policy,
            output_manager,
        }
    }

    /// Run cleanup based on policy.
    pub async fn run_cleanup(&self, variant: &str) -> PackagerResult<()> {
        match self.policy {
            CleanupPolicy::KeepAll => Ok(()),
            CleanupPolicy::MaxSegments(max) => {
                self.output_manager.cleanup_old_segments(variant, max).await
            }
            CleanupPolicy::TimeWindow(duration) => {
                self.cleanup_by_time_window(variant, duration).await
            }
        }
    }

    /// Cleanup segments outside time window.
    async fn cleanup_by_time_window(
        &self,
        variant: &str,
        window: std::time::Duration,
    ) -> PackagerResult<()> {
        let variant_dir = self
            .output_manager
            .structure()
            .get_variant(variant)
            .ok_or_else(|| PackagerError::invalid_config("Variant not found"))?;

        let now = std::time::SystemTime::now();
        let cutoff = now - window;

        let mut entries = tokio::fs::read_dir(variant_dir).await?;
        let mut deleted_count = 0;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if let Ok(metadata) = entry.metadata().await {
                if let Ok(modified) = metadata.modified() {
                    if modified < cutoff {
                        tokio::fs::remove_file(&path).await?;
                        deleted_count += 1;
                        debug!("Deleted old segment: {}", path.display());
                    }
                }
            }
        }

        if deleted_count > 0 {
            info!(
                "Cleaned up {} segments outside time window for variant {}",
                deleted_count, variant
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_structure_creation() {
        let root =
            Utf8PathBuf::from_path_buf(std::env::temp_dir().join("oximedia-packager-output-root"))
                .expect("temp path should be valid UTF-8");
        let mut structure = OutputStructure::new(root.clone());

        let variant_dir = structure.add_variant("video", 1000000);
        assert_eq!(variant_dir, root.join("video_1000000"));

        assert_eq!(structure.get_variant("video"), Some(&variant_dir));
    }

    #[test]
    fn test_cleanup_policy() {
        let policy = CleanupPolicy::MaxSegments(10);
        match policy {
            CleanupPolicy::MaxSegments(max) => assert_eq!(max, 10),
            _ => panic!("Wrong policy type"),
        }
    }
}
