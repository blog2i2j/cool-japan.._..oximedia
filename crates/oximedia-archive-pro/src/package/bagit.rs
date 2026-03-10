//! `BagIt` package format implementation
//!
//! `BagIt` is a hierarchical file system packaging format for storage and transfer
//! of digital content. See: <https://tools.ietf.org/html/rfc8493>

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator, ChecksumVerifier, FileChecksum};
use crate::{Error, Result};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// `BagIt` version
const BAGIT_VERSION: &str = "1.0";
const BAGIT_ENCODING: &str = "UTF-8";

/// `BagIt` package
#[derive(Debug, Clone)]
pub struct BagItPackage {
    /// Root directory of the bag
    pub root: PathBuf,
    /// Bag metadata
    pub metadata: HashMap<String, String>,
    /// Checksum algorithm used
    pub algorithm: ChecksumAlgorithm,
}

impl BagItPackage {
    /// Load an existing `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if the bag is invalid or cannot be read
    pub fn load(root: &Path) -> Result<Self> {
        let bagit_txt = root.join("bagit.txt");
        if !bagit_txt.exists() {
            return Err(Error::InvalidBag("Missing bagit.txt".to_string()));
        }

        let metadata = Self::read_metadata(root)?;
        let algorithm = Self::detect_algorithm(root)?;

        Ok(Self {
            root: root.to_path_buf(),
            metadata,
            algorithm,
        })
    }

    /// Read bag metadata from bag-info.txt
    fn read_metadata(root: &Path) -> Result<HashMap<String, String>> {
        let bag_info = root.join("bag-info.txt");
        let mut metadata = HashMap::new();

        if bag_info.exists() {
            let file = File::open(bag_info)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                if let Some((key, value)) = line.split_once(':') {
                    metadata.insert(key.trim().to_string(), value.trim().to_string());
                }
            }
        }

        Ok(metadata)
    }

    /// Detect the checksum algorithm used
    fn detect_algorithm(root: &Path) -> Result<ChecksumAlgorithm> {
        // Try different manifest filename formats
        for (algo, names) in &[
            (
                ChecksumAlgorithm::Sha256,
                vec!["manifest-sha-256.txt", "manifest-sha256.txt"],
            ),
            (
                ChecksumAlgorithm::Sha512,
                vec!["manifest-sha-512.txt", "manifest-sha512.txt"],
            ),
            (ChecksumAlgorithm::Md5, vec!["manifest-md5.txt"]),
        ] {
            for name in names {
                if root.join(name).exists() {
                    return Ok(*algo);
                }
            }
        }
        Err(Error::InvalidBag("No manifest file found".to_string()))
    }

    /// Get the data directory
    #[must_use]
    pub fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }

    /// List all files in the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest cannot be read
    pub fn list_files(&self) -> Result<Vec<PathBuf>> {
        let manifest = self.read_manifest()?;
        Ok(manifest.into_keys().collect())
    }

    /// Read the manifest file
    fn read_manifest(&self) -> Result<HashMap<PathBuf, String>> {
        let manifest_name = format!("manifest-{}.txt", self.algorithm.name().to_lowercase());
        let manifest_path = self.root.join(manifest_name);

        let mut manifest = HashMap::new();
        let file = File::open(manifest_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if let Some((hash, path)) = line.split_once(char::is_whitespace) {
                manifest.insert(PathBuf::from(path.trim()), hash.trim().to_string());
            }
        }

        Ok(manifest)
    }
}

/// `BagIt` package builder
pub struct BagItBuilder {
    root: PathBuf,
    algorithm: ChecksumAlgorithm,
    metadata: HashMap<String, String>,
    files: Vec<(PathBuf, PathBuf)>, // (source, destination in bag)
}

impl BagItBuilder {
    /// Create a new `BagIt` builder
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self {
            root,
            algorithm: ChecksumAlgorithm::Sha256,
            metadata: HashMap::new(),
            files: Vec::new(),
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add bag metadata
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a file to the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_file(mut self, source: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }

        let filename = source
            .file_name()
            .ok_or_else(|| Error::InvalidBag("Invalid filename".to_string()))?;
        let dest = PathBuf::from("data").join(filename);
        self.files.push((source.to_path_buf(), dest));
        Ok(self)
    }

    /// Add a file with a custom destination path in the bag
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be accessed
    pub fn add_file_as(mut self, source: &Path, dest: &Path) -> Result<Self> {
        if !source.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File not found: {}", source.display()),
            )));
        }

        let dest_in_data = PathBuf::from("data").join(dest);
        self.files.push((source.to_path_buf(), dest_in_data));
        Ok(self)
    }

    /// Build the `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if the bag cannot be created
    pub fn build(self) -> Result<BagItPackage> {
        // Create bag directory structure
        fs::create_dir_all(&self.root)?;
        fs::create_dir_all(self.root.join("data"))?;

        // Write bagit.txt
        let bagit_txt = self.root.join("bagit.txt");
        let mut file = File::create(bagit_txt)?;
        writeln!(file, "BagIt-Version: {BAGIT_VERSION}")?;
        writeln!(file, "Tag-File-Character-Encoding: {BAGIT_ENCODING}")?;

        // Copy files and generate checksums
        let mut checksums = Vec::new();
        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);

        for (source, dest) in &self.files {
            let target = self.root.join(dest);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::copy(source, &target)?;

            // Generate checksum
            let checksum = generator.generate_file(&target)?;
            checksums.push((dest.clone(), checksum));
        }

        // Write manifest
        let manifest_name = format!("manifest-{}.txt", self.algorithm.name().to_lowercase());
        let manifest_path = self.root.join(manifest_name);
        let mut manifest_file = File::create(manifest_path)?;

        for (path, checksum) in &checksums {
            if let Some(hash) = checksum.checksums.get(&self.algorithm) {
                writeln!(manifest_file, "{}  {}", hash, path.display())?;
            }
        }

        // Write bag-info.txt
        let mut metadata = self.metadata.clone();
        metadata
            .entry("Bagging-Date".to_string())
            .or_insert_with(|| chrono::Utc::now().format("%Y-%m-%d").to_string());
        metadata
            .entry("Bag-Software-Agent".to_string())
            .or_insert_with(|| "oximedia-archive-pro".to_string());
        metadata
            .entry("Payload-Oxum".to_string())
            .or_insert_with(|| {
                let total_size: u64 = checksums.iter().map(|(_, cs)| cs.size).sum();
                format!("{}.{}", total_size, checksums.len())
            });

        let bag_info = self.root.join("bag-info.txt");
        let mut info_file = File::create(bag_info)?;
        for (key, value) in &metadata {
            writeln!(info_file, "{key}: {value}")?;
        }

        Ok(BagItPackage {
            root: self.root,
            metadata,
            algorithm: self.algorithm,
        })
    }
}

/// `BagIt` package validator
pub struct BagItValidator;

impl BagItValidator {
    /// Validate a `BagIt` package
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails
    pub fn validate(bag: &BagItPackage) -> Result<()> {
        // Check for required files
        if !bag.root.join("bagit.txt").exists() {
            return Err(Error::InvalidBag("Missing bagit.txt".to_string()));
        }

        // Read and verify manifest
        let manifest = bag.read_manifest()?;
        let verifier = ChecksumVerifier::with_algorithms(vec![bag.algorithm]);

        for (path, expected_hash) in manifest {
            let full_path = bag.root.join(&path);
            if !full_path.exists() {
                return Err(Error::InvalidBag(format!(
                    "Missing file in manifest: {}",
                    path.display()
                )));
            }

            let checksum = FileChecksum {
                path: full_path.clone(),
                size: fs::metadata(&full_path)?.len(),
                checksums: [(bag.algorithm, expected_hash)].into_iter().collect(),
                timestamp: chrono::Utc::now(),
            };

            let report = verifier.verify_file(&checksum)?;
            if !report.is_success() {
                return Err(Error::InvalidBag(format!(
                    "Checksum mismatch for {}",
                    path.display()
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_create_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Test content")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let bag = BagItBuilder::new(bag_dir.clone())
            .with_metadata("Contact-Name", "Test User")
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        assert!(bag.root.join("bagit.txt").exists());
        assert!(bag.root.join("bag-info.txt").exists());
        assert!(bag.root.join("data").exists());
    }

    #[test]
    fn test_validate_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Validation test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let bag = BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let result = BagItValidator::validate(&bag);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_bagit_package() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let bag_dir = temp_dir.path().join("test-bag");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Load test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        BagItBuilder::new(bag_dir.clone())
            .add_file(test_file.path())
            .expect("operation should succeed")
            .build()
            .expect("operation should succeed");

        let loaded = BagItPackage::load(&bag_dir).expect("operation should succeed");
        assert_eq!(loaded.algorithm, ChecksumAlgorithm::Sha256);
    }
}
