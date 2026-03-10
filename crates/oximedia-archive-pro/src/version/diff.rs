//! Diff generation between versions

use crate::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// File difference information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDiff {
    /// Old version number
    pub old_version: u32,
    /// New version number
    pub new_version: u32,
    /// Size change in bytes
    pub size_delta: i64,
    /// Checksum changed
    pub checksum_changed: bool,
    /// Timestamp of diff
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Diff generator
pub struct DiffGenerator;

impl Default for DiffGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffGenerator {
    /// Create a new diff generator
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Generate diff between two file versions
    ///
    /// # Errors
    ///
    /// Returns an error if diff generation fails
    pub fn generate(
        &self,
        old: &Path,
        new: &Path,
        old_version: u32,
        new_version: u32,
    ) -> Result<FileDiff> {
        let old_size = std::fs::metadata(old)?.len() as i64;
        let new_size = std::fs::metadata(new)?.len() as i64;
        let size_delta = new_size - old_size;

        // Simple comparison - in real implementation would compare checksums
        let checksum_changed = size_delta != 0;

        Ok(FileDiff {
            old_version,
            new_version,
            size_delta,
            checksum_changed,
            timestamp: chrono::Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_diff() {
        let mut file1 = NamedTempFile::new().expect("operation should succeed");
        let mut file2 = NamedTempFile::new().expect("operation should succeed");

        file1
            .write_all(b"Version 1")
            .expect("operation should succeed");
        file2
            .write_all(b"Version 2 is longer")
            .expect("operation should succeed");
        file1.flush().expect("operation should succeed");
        file2.flush().expect("operation should succeed");

        let generator = DiffGenerator::new();
        let diff = generator
            .generate(file1.path(), file2.path(), 1, 2)
            .expect("operation should succeed");

        assert_eq!(diff.old_version, 1);
        assert_eq!(diff.new_version, 2);
        assert!(diff.size_delta > 0);
        assert!(diff.checksum_changed);
    }
}
