//! Fixity verification implementation

use crate::{
    checksum::{ChecksumAlgorithm, ChecksumGenerator},
    Result,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Fixity check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixityResult {
    /// File path
    pub path: PathBuf,
    /// Check passed
    pub passed: bool,
    /// Expected checksum
    pub expected_checksum: Option<String>,
    /// Actual checksum
    pub actual_checksum: Option<String>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Error message
    pub error: Option<String>,
}

/// Fixity checker
pub struct FixityChecker {
    algorithm: ChecksumAlgorithm,
    stored_checksums: std::collections::HashMap<PathBuf, String>,
}

impl Default for FixityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl FixityChecker {
    /// Create a new fixity checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: ChecksumAlgorithm::Sha256,
            stored_checksums: std::collections::HashMap::new(),
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Store a checksum for later verification
    pub fn store_checksum(&mut self, path: PathBuf, checksum: String) {
        self.stored_checksums.insert(path, checksum);
    }

    /// Verify file integrity
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails
    pub fn verify(&self, path: &Path) -> Result<FixityResult> {
        let expected = match self.stored_checksums.get(path) {
            Some(cs) => cs.clone(),
            None => {
                return Ok(FixityResult {
                    path: path.to_path_buf(),
                    passed: false,
                    expected_checksum: None,
                    actual_checksum: None,
                    timestamp: chrono::Utc::now(),
                    error: Some("No stored checksum found".to_string()),
                });
            }
        };

        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);
        let current = generator.generate_file(path)?;
        let actual = current.checksums.get(&self.algorithm).cloned();

        let passed = actual.as_ref() == Some(&expected);

        Ok(FixityResult {
            path: path.to_path_buf(),
            passed,
            expected_checksum: Some(expected),
            actual_checksum: actual,
            timestamp: chrono::Utc::now(),
            error: None,
        })
    }

    /// Verify multiple files
    ///
    /// # Errors
    ///
    /// Returns an error if batch verification fails
    pub fn verify_batch(&self, paths: &[PathBuf]) -> Result<Vec<FixityResult>> {
        paths.iter().map(|p| self.verify(p)).collect()
    }

    /// Load checksums from a manifest file
    ///
    /// # Errors
    ///
    /// Returns an error if manifest cannot be loaded
    pub fn load_manifest(&mut self, manifest_path: &Path) -> Result<()> {
        let content = std::fs::read_to_string(manifest_path)?;

        for line in content.lines() {
            if let Some((checksum, path)) = line.split_once(char::is_whitespace) {
                self.stored_checksums
                    .insert(PathBuf::from(path.trim()), checksum.trim().to_string());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_store_and_verify() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");
        let hash = checksum
            .checksums
            .get(&ChecksumAlgorithm::Sha256)
            .expect("operation should succeed");

        let mut checker = FixityChecker::new();
        checker.store_checksum(file.path().to_path_buf(), hash.clone());

        let result = checker
            .verify(file.path())
            .expect("operation should succeed");
        assert!(result.passed);
    }

    #[test]
    fn test_verify_without_stored_checksum() {
        let file = NamedTempFile::new().expect("operation should succeed");
        let checker = FixityChecker::new();
        let result = checker
            .verify(file.path())
            .expect("operation should succeed");

        assert!(!result.passed);
        assert!(result.error.is_some());
    }
}
