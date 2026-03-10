//! Metadata embedding in files

use crate::Result;
use std::collections::HashMap;
use std::path::Path;

/// Metadata embedder
pub struct MetadataEmbedder {
    metadata: HashMap<String, String>,
}

impl Default for MetadataEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataEmbedder {
    /// Create a new metadata embedder
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
        }
    }

    /// Add metadata field
    #[must_use]
    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Embed metadata in a file
    ///
    /// # Errors
    ///
    /// Returns an error if embedding fails
    ///
    /// Note: Actual implementation would use format-specific libraries
    /// (e.g., `FFmpeg` for video, `ExifTool` for images)
    pub fn embed(&self, _path: &Path) -> Result<()> {
        // This is a placeholder - actual implementation would use:
        // - FFmpeg metadata for video/audio
        // - EXIF for images
        // - XMP for cross-format metadata
        Ok(())
    }

    /// Embed as sidecar file
    ///
    /// # Errors
    ///
    /// Returns an error if the sidecar cannot be written
    pub fn embed_sidecar(&self, path: &Path) -> Result<()> {
        let sidecar_path = path.with_extension("json");
        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        std::fs::write(sidecar_path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_metadata_embedder() {
        let embedder = MetadataEmbedder::new()
            .with_field("title", "Test Video")
            .with_field("creator", "Test User");

        assert_eq!(embedder.metadata.len(), 2);
    }

    #[test]
    fn test_embed_sidecar() {
        let file = NamedTempFile::new().expect("operation should succeed");
        let embedder = MetadataEmbedder::new().with_field("title", "Test");

        let result = embedder.embed_sidecar(file.path());
        assert!(result.is_ok());

        let sidecar_path = file.path().with_extension("json");
        assert!(sidecar_path.exists());
    }
}
