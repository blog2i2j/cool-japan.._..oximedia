//! OAIS (Open Archival Information System) package format
//!
//! OAIS defines three package types:
//! - SIP (Submission Information Package): For submission to archive
//! - AIP (Archival Information Package): For long-term preservation
//! - DIP (Dissemination Information Package): For access/distribution

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

/// OAIS package types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OaisPackageType {
    /// Submission Information Package
    Sip,
    /// Archival Information Package
    Aip,
    /// Dissemination Information Package
    Dip,
}

impl OaisPackageType {
    /// Returns the package type name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Sip => "SIP",
            Self::Aip => "AIP",
            Self::Dip => "DIP",
        }
    }

    /// Returns a description of the package type
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Sip => "Submission Information Package - for archive submission",
            Self::Aip => "Archival Information Package - for long-term preservation",
            Self::Dip => "Dissemination Information Package - for access and distribution",
        }
    }
}

/// OAIS package structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaisPackage {
    /// Package root directory
    pub root: PathBuf,
    /// Package type
    pub package_type: OaisPackageType,
    /// Package identifier
    pub id: String,
    /// Creation timestamp
    pub created: chrono::DateTime<chrono::Utc>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

impl OaisPackage {
    /// Load an existing OAIS package
    ///
    /// # Errors
    ///
    /// Returns an error if the package is invalid
    pub fn load(root: &Path) -> Result<Self> {
        let manifest_path = root.join("OAIS-MANIFEST.json");
        if !manifest_path.exists() {
            return Err(Error::InvalidOais("Missing OAIS-MANIFEST.json".to_string()));
        }

        let file = File::open(manifest_path)?;
        let package: Self = serde_json::from_reader(file)
            .map_err(|e| Error::InvalidOais(format!("Invalid manifest: {e}")))?;

        Ok(package)
    }

    /// Get the content directory
    #[must_use]
    pub fn content_dir(&self) -> PathBuf {
        self.root.join("content")
    }

    /// Get the metadata directory
    #[must_use]
    pub fn metadata_dir(&self) -> PathBuf {
        self.root.join("metadata")
    }

    /// Get the submission documentation directory (SIP only)
    #[must_use]
    pub fn submission_dir(&self) -> PathBuf {
        self.root.join("submission")
    }

    /// Get the preservation metadata directory (AIP only)
    #[must_use]
    pub fn preservation_dir(&self) -> PathBuf {
        self.root.join("preservation")
    }
}

/// OAIS package builder
pub struct OaisBuilder {
    root: PathBuf,
    package_type: OaisPackageType,
    id: String,
    algorithm: ChecksumAlgorithm,
    metadata: HashMap<String, String>,
    content_files: Vec<(PathBuf, PathBuf)>,
    metadata_files: Vec<(PathBuf, PathBuf)>,
}

impl OaisBuilder {
    /// Create a new OAIS builder
    #[must_use]
    pub fn new(root: PathBuf, package_type: OaisPackageType, id: String) -> Self {
        Self {
            root,
            package_type,
            id,
            algorithm: ChecksumAlgorithm::Sha256,
            metadata: HashMap::new(),
            content_files: Vec::new(),
            metadata_files: Vec::new(),
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add package metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a content file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_content_file(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }
        self.content_files
            .push((source.to_path_buf(), dest.to_path_buf()));
        Ok(self)
    }

    /// Add a metadata file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_metadata_file(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }
        self.metadata_files
            .push((source.to_path_buf(), dest.to_path_buf()));
        Ok(self)
    }

    /// Build the OAIS package
    ///
    /// # Errors
    ///
    /// Returns an error if the package cannot be created
    pub fn build(self) -> Result<OaisPackage> {
        // Create directory structure
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(self.root.join("content"))?;
        fs::create_dir_all(self.root.join("metadata"))?;

        match self.package_type {
            OaisPackageType::Sip => {
                fs::create_dir_all(self.root.join("submission"))?;
            }
            OaisPackageType::Aip => {
                fs::create_dir_all(self.root.join("preservation"))?;
            }
            OaisPackageType::Dip => {
                // DIP may have additional access-specific directories
            }
        }

        // Copy content files
        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);
        let mut content_checksums = HashMap::new();

        for (source, dest) in &self.content_files {
            let target = self.root.join("content").join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;

            let checksum = generator.generate_file(&target)?;
            if let Some(hash) = checksum.checksums.get(&self.algorithm) {
                content_checksums.insert(dest.display().to_string(), hash.clone());
            }
        }

        // Copy metadata files
        for (source, dest) in &self.metadata_files {
            let target = self.root.join("metadata").join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;
        }

        // Write checksums file
        let checksums_path = self.root.join("CHECKSUMS.txt");
        let mut checksums_file = File::create(checksums_path)?;
        for (path, hash) in &content_checksums {
            writeln!(checksums_file, "{hash}  content/{path}")?;
        }

        // Create package manifest
        let package = OaisPackage {
            root: self.root.clone(),
            package_type: self.package_type,
            id: self.id,
            created: chrono::Utc::now(),
            metadata: self.metadata,
        };

        let manifest_path = self.root.join("OAIS-MANIFEST.json");
        let manifest_file = File::create(manifest_path)?;
        serde_json::to_writer_pretty(manifest_file, &package)
            .map_err(|e| Error::InvalidOais(format!("Failed to write manifest: {e}")))?;

        Ok(package)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_oais_package_types() {
        assert_eq!(OaisPackageType::Sip.name(), "SIP");
        assert_eq!(OaisPackageType::Aip.name(), "AIP");
        assert_eq!(OaisPackageType::Dip.name(), "DIP");
    }

    #[test]
    fn test_create_sip_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-sip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"SIP content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let package =
            OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Sip, "SIP-001".to_string())
                .with_metadata("Creator", "Test System")
                .add_content_file(test_file.path(), Path::new("video.mkv"))
                .expect("operation should succeed")
                .build()
                .expect("operation should succeed");

        assert_eq!(package.package_type, OaisPackageType::Sip);
        assert_eq!(package.id, "SIP-001");
        assert!(package.root.join("content").exists());
        assert!(package.root.join("metadata").exists());
        assert!(package.root.join("submission").exists());
        assert!(package.root.join("OAIS-MANIFEST.json").exists());
    }

    #[test]
    fn test_create_aip_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-aip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"AIP content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let package =
            OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Aip, "AIP-001".to_string())
                .add_content_file(test_file.path(), Path::new("preservation.mkv"))
                .expect("operation should succeed")
                .build()
                .expect("operation should succeed");

        assert_eq!(package.package_type, OaisPackageType::Aip);
        assert!(package.root.join("preservation").exists());
    }

    #[test]
    fn test_load_oais_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let pkg_dir = temp_dir.path().join("test-load");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Load test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        OaisBuilder::new(pkg_dir.clone(), OaisPackageType::Dip, "DIP-001".to_string())
            .add_content_file(test_file.path(), Path::new("access.mp4"))
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let loaded = OaisPackage::load(&pkg_dir).expect("operation should succeed");
        assert_eq!(loaded.package_type, OaisPackageType::Dip);
        assert_eq!(loaded.id, "DIP-001");
    }
}
