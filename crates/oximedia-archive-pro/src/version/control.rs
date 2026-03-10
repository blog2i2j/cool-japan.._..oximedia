//! Version control system for archive files

use crate::{checksum::ChecksumGenerator, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    /// Version number
    pub version: u32,
    /// File path
    pub path: PathBuf,
    /// Checksum
    pub checksum: String,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Comment
    pub comment: Option<String>,
    /// File size
    pub size: u64,
}

/// Version control system
pub struct VersionControl {
    repository: PathBuf,
    generator: ChecksumGenerator,
}

impl VersionControl {
    /// Create a new version control system
    ///
    /// # Errors
    ///
    /// Returns an error if the repository cannot be created
    pub fn new(repository: PathBuf) -> Result<Self> {
        fs::create_dir_all(&repository)?;
        fs::create_dir_all(repository.join("versions"))?;
        fs::create_dir_all(repository.join("metadata"))?;

        Ok(Self {
            repository,
            generator: ChecksumGenerator::new(),
        })
    }

    /// Add a new version of a file
    ///
    /// # Errors
    ///
    /// Returns an error if the version cannot be added
    pub fn add_version(&self, file: &Path, comment: Option<String>) -> Result<VersionInfo> {
        let metadata = fs::metadata(file)?;
        let size = metadata.len();

        // Generate checksum
        let checksum_result = self.generator.generate_file(file)?;
        let checksum = checksum_result
            .checksums
            .values()
            .next()
            .ok_or_else(|| crate::Error::Metadata("No checksum generated".to_string()))?
            .clone();

        // Determine version number
        let version = self.get_next_version(file)?;

        // Copy file to version storage
        let version_name = format!(
            "{}.v{}",
            file.file_name().unwrap_or_default().to_string_lossy(),
            version
        );
        let version_path = self.repository.join("versions").join(&version_name);
        fs::copy(file, &version_path)?;

        let version_info = VersionInfo {
            version,
            path: version_path,
            checksum,
            timestamp: chrono::Utc::now(),
            comment,
            size,
        };

        // Save version metadata
        self.save_version_metadata(file, &version_info)?;

        Ok(version_info)
    }

    fn get_next_version(&self, file: &Path) -> Result<u32> {
        let versions = self.list_versions(file)?;
        Ok(versions.iter().map(|v| v.version).max().unwrap_or(0) + 1)
    }

    fn save_version_metadata(&self, file: &Path, info: &VersionInfo) -> Result<()> {
        let metadata_file = self.get_metadata_path(file);

        let mut versions = if metadata_file.exists() {
            let content = fs::read_to_string(&metadata_file)?;
            serde_json::from_str::<Vec<VersionInfo>>(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        versions.push(info.clone());

        let json = serde_json::to_string_pretty(&versions)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        fs::write(metadata_file, json)?;

        Ok(())
    }

    fn get_metadata_path(&self, file: &Path) -> PathBuf {
        let filename = file.file_name().unwrap_or_default().to_string_lossy();
        self.repository
            .join("metadata")
            .join(format!("{filename}.json"))
    }

    /// List all versions of a file
    ///
    /// # Errors
    ///
    /// Returns an error if versions cannot be listed
    pub fn list_versions(&self, file: &Path) -> Result<Vec<VersionInfo>> {
        let metadata_file = self.get_metadata_path(file);

        if !metadata_file.exists() {
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(metadata_file)?;
        let versions = serde_json::from_str(&content)
            .map_err(|e| crate::Error::Metadata(format!("JSON parse failed: {e}")))?;

        Ok(versions)
    }

    /// Get a specific version
    ///
    /// # Errors
    ///
    /// Returns an error if the version cannot be found
    pub fn get_version(&self, file: &Path, version: u32) -> Result<VersionInfo> {
        let versions = self.list_versions(file)?;
        versions
            .into_iter()
            .find(|v| v.version == version)
            .ok_or_else(|| crate::Error::Metadata(format!("Version {version} not found")))
    }

    /// Restore a specific version
    ///
    /// # Errors
    ///
    /// Returns an error if restoration fails
    pub fn restore_version(&self, file: &Path, version: u32) -> Result<()> {
        let version_info = self.get_version(file, version)?;
        fs::copy(&version_info.path, file)?;
        Ok(())
    }

    /// Delete old versions (keep only N most recent)
    ///
    /// # Errors
    ///
    /// Returns an error if deletion fails
    pub fn prune_versions(&self, file: &Path, keep: usize) -> Result<()> {
        let mut versions = self.list_versions(file)?;
        versions.sort_by(|a, b| b.version.cmp(&a.version));

        if versions.len() <= keep {
            return Ok(());
        }

        let to_delete = &versions[keep..];
        for version in to_delete {
            if version.path.exists() {
                fs::remove_file(&version.path)?;
            }
        }

        versions.truncate(keep);
        let metadata_file = self.get_metadata_path(file);
        let json = serde_json::to_string_pretty(&versions)
            .map_err(|e| crate::Error::Metadata(format!("JSON serialization failed: {e}")))?;
        fs::write(metadata_file, json)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_add_version() {
        let repo_dir = TempDir::new().expect("operation should succeed");
        let vc =
            VersionControl::new(repo_dir.path().to_path_buf()).expect("operation should succeed");

        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Version 1")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let version = vc
            .add_version(file.path(), Some("Initial version".to_string()))
            .expect("operation should succeed");
        assert_eq!(version.version, 1);
        assert_eq!(version.comment, Some("Initial version".to_string()));
    }

    #[test]
    fn test_multiple_versions() {
        let repo_dir = TempDir::new().expect("operation should succeed");
        let vc =
            VersionControl::new(repo_dir.path().to_path_buf()).expect("operation should succeed");

        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Version 1")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        vc.add_version(file.path(), None)
            .expect("operation should succeed");

        file.write_all(b" - Updated")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let v2 = vc
            .add_version(file.path(), Some("Update".to_string()))
            .expect("operation should succeed");
        assert_eq!(v2.version, 2);

        let versions = vc
            .list_versions(file.path())
            .expect("operation should succeed");
        assert_eq!(versions.len(), 2);
    }

    #[test]
    fn test_prune_versions() {
        let repo_dir = TempDir::new().expect("operation should succeed");
        let vc =
            VersionControl::new(repo_dir.path().to_path_buf()).expect("operation should succeed");

        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test").expect("operation should succeed");
        file.flush().expect("operation should succeed");

        for i in 1..=5 {
            file.write_all(format!(" v{}", i).as_bytes())
                .expect("operation should succeed");
            file.flush().expect("operation should succeed");
            vc.add_version(file.path(), None)
                .expect("operation should succeed");
        }

        vc.prune_versions(file.path(), 2)
            .expect("operation should succeed");
        let versions = vc
            .list_versions(file.path())
            .expect("operation should succeed");
        assert_eq!(versions.len(), 2);
    }
}
