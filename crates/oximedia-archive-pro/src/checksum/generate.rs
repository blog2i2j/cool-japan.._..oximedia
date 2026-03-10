//! Checksum generation for files and data

use super::ChecksumAlgorithm;
use crate::{Error, Result};
use blake3::Hasher as Blake3Hasher;
use md5::{Digest, Md5};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Sha512};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use xxhash_rust::xxh3::Xxh3;

/// File checksum with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileChecksum {
    /// File path
    pub path: PathBuf,
    /// File size in bytes
    pub size: u64,
    /// Checksums by algorithm
    pub checksums: HashMap<ChecksumAlgorithm, String>,
    /// Timestamp of checksum generation
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Checksum generator for files and directories
#[derive(Clone)]
pub struct ChecksumGenerator {
    algorithms: Vec<ChecksumAlgorithm>,
    buffer_size: usize,
}

impl Default for ChecksumGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ChecksumGenerator {
    /// Create a new checksum generator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithms: vec![ChecksumAlgorithm::Sha256],
            buffer_size: 8192,
        }
    }

    /// Set the checksum algorithms to use
    #[must_use]
    pub fn with_algorithms(mut self, algorithms: Vec<ChecksumAlgorithm>) -> Self {
        self.algorithms = algorithms;
        self
    }

    /// Set the buffer size for reading files
    #[must_use]
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Generate checksums for a single file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read
    pub fn generate_file(&self, path: &Path) -> Result<FileChecksum> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let size = metadata.len();

        let mut reader = BufReader::with_capacity(self.buffer_size, file);
        let mut checksums = HashMap::new();

        // Initialize hashers
        let mut md5 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Md5)
            .then(Md5::new);
        let mut sha256 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Sha256)
            .then(Sha256::new);
        let mut sha512 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Sha512)
            .then(Sha512::new);
        let mut xxhash = self
            .algorithms
            .contains(&ChecksumAlgorithm::XxHash64)
            .then(Xxh3::new);
        let mut blake3 = self
            .algorithms
            .contains(&ChecksumAlgorithm::Blake3)
            .then(Blake3Hasher::new);

        // Read file and update all hashers
        let mut buffer = vec![0u8; self.buffer_size];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }

            if let Some(ref mut h) = md5 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = sha256 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = sha512 {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = xxhash {
                h.update(&buffer[..n]);
            }
            if let Some(ref mut h) = blake3 {
                h.update(&buffer[..n]);
            }
        }

        // Finalize and store checksums
        if let Some(h) = md5 {
            checksums.insert(ChecksumAlgorithm::Md5, format!("{:x}", h.finalize()));
        }
        if let Some(h) = sha256 {
            checksums.insert(ChecksumAlgorithm::Sha256, format!("{:x}", h.finalize()));
        }
        if let Some(h) = sha512 {
            checksums.insert(ChecksumAlgorithm::Sha512, format!("{:x}", h.finalize()));
        }
        if let Some(h) = xxhash {
            checksums.insert(ChecksumAlgorithm::XxHash64, format!("{:x}", h.digest()));
        }
        if let Some(h) = blake3 {
            checksums.insert(
                ChecksumAlgorithm::Blake3,
                format!("{}", h.finalize().to_hex()),
            );
        }

        Ok(FileChecksum {
            path: path.to_path_buf(),
            size,
            checksums,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Generate checksums for all files in a directory (recursively)
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or any file fails
    pub fn generate_directory(&self, dir: &Path) -> Result<Vec<FileChecksum>> {
        let entries: Vec<PathBuf> = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(walkdir::DirEntry::into_path)
            .collect();

        entries
            .par_iter()
            .map(|path| self.generate_file(path))
            .collect()
    }

    /// Generate checksums for a list of files in parallel
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be read
    pub fn generate_batch(&self, paths: &[PathBuf]) -> Result<Vec<FileChecksum>> {
        paths
            .par_iter()
            .map(|path| self.generate_file(path))
            .collect()
    }
}

/// Generate a quick checksum using BLAKE3 (fastest)
///
/// # Errors
///
/// Returns an error if the file cannot be read
pub fn quick_checksum(path: &Path) -> Result<String> {
    let generator = ChecksumGenerator::new().with_algorithms(vec![ChecksumAlgorithm::Blake3]);
    let checksum = generator.generate_file(path)?;
    checksum
        .checksums
        .get(&ChecksumAlgorithm::Blake3)
        .cloned()
        .ok_or_else(|| Error::Metadata("BLAKE3 checksum not found".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_generate_file_checksum() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Hello, World!")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new();
        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");

        assert_eq!(checksum.size, 13);
        assert_eq!(checksum.checksums.len(), 1);
        assert!(checksum.checksums.contains_key(&ChecksumAlgorithm::Sha256));
    }

    #[test]
    fn test_multiple_algorithms() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Test data")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let generator = ChecksumGenerator::new().with_algorithms(vec![
            ChecksumAlgorithm::Md5,
            ChecksumAlgorithm::Sha256,
            ChecksumAlgorithm::Blake3,
        ]);

        let checksum = generator
            .generate_file(file.path())
            .expect("operation should succeed");
        assert_eq!(checksum.checksums.len(), 3);
    }

    #[test]
    fn test_quick_checksum() {
        let mut file = NamedTempFile::new().expect("operation should succeed");
        file.write_all(b"Quick test")
            .expect("operation should succeed");
        file.flush().expect("operation should succeed");

        let hash = quick_checksum(file.path()).expect("operation should succeed");
        assert!(!hash.is_empty());
    }
}
