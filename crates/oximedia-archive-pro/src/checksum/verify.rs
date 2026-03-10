//! Checksum verification for integrity checking

use super::{ChecksumAlgorithm, ChecksumGenerator, FileChecksum};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Result of a checksum verification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Checksum matched
    Success,
    /// Checksum mismatch
    Failed {
        /// Expected checksum
        expected: String,
        /// Actual checksum
        actual: String,
    },
    /// File not found
    Missing,
    /// File size mismatch
    SizeMismatch {
        /// Expected size
        expected: u64,
        /// Actual size
        actual: u64,
    },
}

impl VerificationResult {
    /// Returns `true` if the verification succeeded
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns `true` if the verification failed
    #[must_use]
    pub const fn is_failed(&self) -> bool {
        !self.is_success()
    }
}

/// Verification report for a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVerificationReport {
    /// File path
    pub path: PathBuf,
    /// Verification results by algorithm
    pub results: HashMap<ChecksumAlgorithm, VerificationResult>,
    /// Timestamp of verification
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FileVerificationReport {
    /// Returns `true` if all verifications succeeded
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.results.values().all(VerificationResult::is_success)
    }

    /// Returns the number of failed verifications
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.results.values().filter(|r| r.is_failed()).count()
    }
}

/// Verification report for multiple files
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Per-file reports
    pub files: Vec<FileVerificationReport>,
    /// Timestamp of verification
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl VerificationReport {
    /// Returns `true` if all files passed verification
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.files.iter().all(FileVerificationReport::is_success)
    }

    /// Returns the number of files that passed verification
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.files.iter().filter(|f| f.is_success()).count()
    }

    /// Returns the number of files that failed verification
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.files.len() - self.success_count()
    }

    /// Returns a summary string
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Verification: {} succeeded, {} failed out of {} total",
            self.success_count(),
            self.failed_count(),
            self.files.len()
        )
    }
}

/// Checksum verifier
pub struct ChecksumVerifier {
    generator: ChecksumGenerator,
}

impl Default for ChecksumVerifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumVerifier {
    /// Create a new checksum verifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            generator: ChecksumGenerator::new(),
        }
    }

    /// Create a verifier with specific algorithms
    #[must_use]
    pub fn with_algorithms(algorithms: Vec<ChecksumAlgorithm>) -> Self {
        Self {
            generator: ChecksumGenerator::new().with_algorithms(algorithms),
        }
    }

    /// Verify a single file against expected checksums
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn verify_file(&self, expected: &FileChecksum) -> Result<FileVerificationReport> {
        let path = &expected.path;
        let mut results = HashMap::new();

        // Check if file exists
        if !path.exists() {
            for algo in expected.checksums.keys() {
                results.insert(*algo, VerificationResult::Missing);
            }
            return Ok(FileVerificationReport {
                path: path.clone(),
                results,
                timestamp: chrono::Utc::now(),
            });
        }

        // Check file size
        let metadata = std::fs::metadata(path)?;
        let actual_size = metadata.len();
        if actual_size != expected.size {
            for algo in expected.checksums.keys() {
                results.insert(
                    *algo,
                    VerificationResult::SizeMismatch {
                        expected: expected.size,
                        actual: actual_size,
                    },
                );
            }
            return Ok(FileVerificationReport {
                path: path.clone(),
                results,
                timestamp: chrono::Utc::now(),
            });
        }

        // Generate current checksums
        let algorithms: Vec<ChecksumAlgorithm> = expected.checksums.keys().copied().collect();
        let current = self
            .generator
            .clone()
            .with_algorithms(algorithms)
            .generate_file(path)?;

        // Compare checksums
        for (algo, expected_hash) in &expected.checksums {
            let result = if let Some(actual_hash) = current.checksums.get(algo) {
                if actual_hash == expected_hash {
                    VerificationResult::Success
                } else {
                    VerificationResult::Failed {
                        expected: expected_hash.clone(),
                        actual: actual_hash.clone(),
                    }
                }
            } else {
                VerificationResult::Missing
            };
            results.insert(*algo, result);
        }

        Ok(FileVerificationReport {
            path: path.clone(),
            results,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify multiple files
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read
    pub fn verify_batch(&self, expected: &[FileChecksum]) -> Result<VerificationReport> {
        let files = expected
            .iter()
            .map(|e| self.verify_file(e))
            .collect::<Result<Vec<_>>>()?;

        Ok(VerificationReport {
            files,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Verify a file against a single expected checksum
    ///
    /// # Errors
    ///
    /// Returns an error if verification fails or file cannot be read
    pub fn verify_simple(path: &Path, algorithm: ChecksumAlgorithm, expected: &str) -> Result<()> {
        let generator = ChecksumGenerator::new().with_algorithms(vec![algorithm]);
        let checksum = generator.generate_file(path)?;

        let actual = checksum
            .checksums
            .get(&algorithm)
            .ok_or_else(|| Error::Metadata(format!("Checksum not generated for {algorithm:?}")))?;

        if actual != expected {
            return Err(Error::ChecksumMismatch {
                expected: expected.to_string(),
                actual: actual.clone(),
            });
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
    fn test_verify_success() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let expected = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_file(&expected)
            .expect("operation should succeed");

        assert!(report.is_success());
        assert_eq!(report.failed_count(), 0);
    }

    #[test]
    fn test_verify_mismatch() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Original content")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let mut expected = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        // Modify the expected checksum
        expected
            .checksums
            .insert(ChecksumAlgorithm::Sha256, "deadbeef".to_string());

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_file(&expected)
            .expect("operation should succeed");

        assert!(!report.is_success());
        assert_eq!(report.failed_count(), 1);
    }

    #[test]
    fn test_verify_simple() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Simple test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");
        let expected = checksum
            .checksums
            .get(&ChecksumAlgorithm::Sha256)
            .expect("operation should succeed");

        let result =
            ChecksumVerifier::verify_simple(file.path(), ChecksumAlgorithm::Sha256, expected);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_batch() {
        let mut file1 = NamedTempFile::new().expect("operation should succeed");
        let mut file2 = NamedTempFile::new().expect("operation should succeed");
        file1
            .write_all(b"File 1")
            .expect("operation should succeed");
        file2
            .write_all(b"File 2")
            .expect("operation should succeed");
        file1.flush().expect("operation should succeed");
        file2.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let expected = vec![
            generator
                .generate_file(file1.path())
                .expect("operation should succeed"),
            generator
                .generate_file(file2.path())
                .expect("operation should succeed"),
        ];

        let verifier = ChecksumVerifier::new();
        let report = verifier
            .verify_batch(&expected)
            .expect("operation should succeed");

        assert!(report.is_success());
        assert_eq!(report.success_count(), 2);
        assert_eq!(report.failed_count(), 0);
    }
}
