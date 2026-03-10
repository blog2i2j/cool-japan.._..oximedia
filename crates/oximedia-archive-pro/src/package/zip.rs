//! ZIP archive creation with checksums

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator};
use crate::{Error, Result};
use oxiarc_archive::zip::{ZipCompressionLevel, ZipReader, ZipWriter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// ZIP archiver with checksum support
pub struct ZipArchiver {
    algorithm: ChecksumAlgorithm,
    compression: ZipCompressionLevel,
}

impl Default for ZipArchiver {
    fn default() -> Self {
        Self::new()
    }
}

impl ZipArchiver {
    /// Create a new ZIP archiver
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: ChecksumAlgorithm::Sha256,
            compression: ZipCompressionLevel::Normal,
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set compression level
    #[must_use]
    pub fn with_compression(mut self, compression: ZipCompressionLevel) -> Self {
        self.compression = compression;
        self
    }

    /// Create a ZIP archive from files
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_archive(
        &self,
        output: &Path,
        files: &[(PathBuf, PathBuf)], // (source, path_in_archive)
    ) -> Result<String> {
        let file = File::create(output)?;
        let mut zip = ZipWriter::new(file);
        zip.set_compression(self.compression);

        for (source, dest) in files {
            let mut source_file = File::open(source)?;
            let mut buffer = Vec::new();
            source_file.read_to_end(&mut buffer)?;

            zip.add_file(&dest.to_string_lossy(), &buffer)
                .map_err(|e| Error::Archive(format!("Failed to add file: {e}")))?;
        }

        zip.finish()
            .map_err(|e| Error::Archive(format!("Failed to finalize archive: {e}")))?;

        // Generate checksum for the archive
        let generator = ChecksumGenerator::new().with_algorithms(vec![self.algorithm]);
        let checksum = generator.generate_file(output)?;

        checksum
            .checksums
            .get(&self.algorithm)
            .cloned()
            .ok_or_else(|| Error::Metadata("Checksum generation failed".to_string()))
    }

    /// Create archive from a directory
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_from_directory(&self, output: &Path, dir: &Path) -> Result<String> {
        let files: Vec<(PathBuf, PathBuf)> = walkdir::WalkDir::new(dir)
            .follow_links(false)
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter(|e| e.file_type().is_file())
            .filter_map(|e| {
                let path = e.path().to_path_buf();
                let relative = path.strip_prefix(dir).ok()?.to_path_buf();
                Some((path, relative))
            })
            .collect();

        self.create_archive(output, &files)
    }

    /// Extract a ZIP archive
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails
    pub fn extract_archive(archive_path: &Path, output_dir: &Path) -> Result<()> {
        let file = File::open(archive_path)?;
        let mut reader = ZipReader::new(file)
            .map_err(|e| Error::Archive(format!("Failed to open archive: {e}")))?;

        std::fs::create_dir_all(output_dir)?;

        for entry in reader.entries().to_vec() {
            let data = reader
                .extract(&entry)
                .map_err(|e| Error::Archive(format!("Failed to extract entry: {e}")))?;

            let entry_path = output_dir.join(&entry.name);
            if let Some(parent) = entry_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            if !entry.name.ends_with('/') {
                let mut out_file = File::create(&entry_path)?;
                out_file.write_all(&data)?;
            }
        }

        Ok(())
    }

    /// List contents of a ZIP archive
    ///
    /// # Errors
    ///
    /// Returns an error if the archive cannot be read
    pub fn list_contents(archive_path: &Path) -> Result<Vec<PathBuf>> {
        let file = File::open(archive_path)?;
        let reader = ZipReader::new(file)
            .map_err(|e| Error::Archive(format!("Failed to open archive: {e}")))?;

        let contents = reader
            .entries()
            .iter()
            .filter(|e| !e.name.ends_with('/'))
            .map(|e| PathBuf::from(&e.name))
            .collect();

        Ok(contents)
    }

    /// Create archive with checksum file
    ///
    /// # Errors
    ///
    /// Returns an error if archive creation fails
    pub fn create_with_checksum_file(
        &self,
        output: &Path,
        files: &[(PathBuf, PathBuf)],
    ) -> Result<()> {
        let checksum = self.create_archive(output, files)?;

        let checksum_path = output.with_extension("zip.sha256");
        let mut checksum_file = File::create(checksum_path)?;
        writeln!(
            checksum_file,
            "{}  {}",
            checksum,
            output.file_name().unwrap_or_default().to_string_lossy()
        )?;

        Ok(())
    }

    /// Create uncompressed (stored) archive for preservation
    #[must_use]
    pub fn uncompressed() -> Self {
        Self::new().with_compression(ZipCompressionLevel::Store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_create_zip_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.zip");

        let mut file1 = NamedTempFile::new().expect("operation should succeed");
        let mut file2 = NamedTempFile::new().expect("operation should succeed");
        file1
            .write_all(b"File 1 content")
            .expect("operation should succeed");
        file2
            .write_all(b"File 2 content")
            .expect("operation should succeed");
        file1.flush().expect("operation should succeed");
        file2.flush().expect("operation should succeed");

        let files = vec![
            (file1.path().to_path_buf(), PathBuf::from("file1.txt")),
            (file2.path().to_path_buf(), PathBuf::from("file2.txt")),
        ];

        let archiver = ZipArchiver::new();
        let checksum = archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }

    #[test]
    fn test_extract_zip_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.zip");
        let extract_dir = temp_dir.path().join("extracted");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Extract test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("test.txt"))];

        let archiver = ZipArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        ZipArchiver::extract_archive(&archive_path, &extract_dir)
            .expect("operation should succeed");

        assert!(extract_dir.join("test.txt").exists());
    }

    #[test]
    fn test_list_zip_contents() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.zip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"List test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("listed.txt"))];

        let archiver = ZipArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        let contents = ZipArchiver::list_contents(&archive_path).expect("operation should succeed");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0], PathBuf::from("listed.txt"));
    }

    #[test]
    fn test_uncompressed_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("uncompressed.zip");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Uncompressed test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("test.txt"))];

        let archiver = ZipArchiver::uncompressed();
        let checksum = archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }
}
