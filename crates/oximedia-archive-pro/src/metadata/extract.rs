//! Metadata extraction from files

use crate::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Metadata extractor
pub struct MetadataExtractor;

impl Default for MetadataExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl MetadataExtractor {
    /// Create a new metadata extractor
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Extract metadata from a file
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails
    ///
    /// Note: Actual implementation would use format-specific libraries
    pub fn extract(&self, path: &Path) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();

        // Basic file metadata
        if let Ok(file_metadata) = fs::metadata(path) {
            metadata.insert("size".to_string(), file_metadata.len().to_string());

            if let Ok(modified) = file_metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    metadata.insert("modified".to_string(), duration.as_secs().to_string());
                }
            }
        }

        // File name and extension
        if let Some(name) = path.file_name() {
            metadata.insert("filename".to_string(), name.to_string_lossy().to_string());
        }

        if let Some(ext) = path.extension() {
            metadata.insert("extension".to_string(), ext.to_string_lossy().to_string());
        }

        Ok(metadata)
    }

    /// Extract metadata from sidecar file
    ///
    /// # Errors
    ///
    /// Returns an error if the sidecar cannot be read
    pub fn extract_sidecar(&self, path: &Path) -> Result<HashMap<String, String>> {
        let sidecar_path = path.with_extension("json");

        if !sidecar_path.exists() {
            return Ok(HashMap::new());
        }

        let content = fs::read_to_string(sidecar_path)?;
        let metadata: HashMap<String, String> = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Metadata(format!("JSON parse failed: {e}")))?;

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_extract_basic_metadata() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let extractor = MetadataExtractor::new();
        let metadata = extractor
            .extract(file.path())
            .expect("operation should succeed");

        assert!(metadata.contains_key("size"));
        assert_eq!(
            metadata.get("size").expect("operation should succeed"),
            "12"
        );
    }

    #[test]
    fn test_extract_sidecar() {
        let file = NamedTempFile::new().expect("operation should succeed");
        let sidecar_path = file.path().with_extension("json");

        fs::write(&sidecar_path, r#"{"title":"Test","creator":"User"}"#)
            .expect("operation should succeed");

        let extractor = MetadataExtractor::new();
        let metadata = extractor
            .extract_sidecar(file.path())
            .expect("operation should succeed");

        assert_eq!(
            metadata.get("title").expect("operation should succeed"),
            "Test"
        );
        assert_eq!(
            metadata.get("creator").expect("operation should succeed"),
            "User"
        );
    }
}
