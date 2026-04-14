//! TAR archive creation with checksums

use crate::checksum::{ChecksumAlgorithm, ChecksumGenerator};
use crate::{Error, Result};
use oxiarc_archive::{TarReader, TarWriter};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// TAR archiver with checksum support
pub struct TarArchiver {
    algorithm: ChecksumAlgorithm,
}

impl Default for TarArchiver {
    fn default() -> Self {
        Self::new()
    }
}

impl TarArchiver {
    /// Create a new TAR archiver
    #[must_use]
    pub fn new() -> Self {
        Self {
            algorithm: ChecksumAlgorithm::Sha256,
        }
    }

    /// Set the checksum algorithm
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: ChecksumAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Create a TAR archive from files
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
        let mut tar_writer = TarWriter::new(file);

        for (source, dest) in files {
            let dest_str = dest
                .to_str()
                .ok_or_else(|| Error::Archive("Non-UTF8 path in archive".to_string()))?;
            let file_data = std::fs::read(source)
                .map_err(|e| Error::Archive(format!("Failed to read file: {e}")))?;
            tar_writer
                .add_file(dest_str, &file_data)
                .map_err(|e| Error::Archive(format!("Failed to add file: {e}")))?;
        }

        tar_writer
            .finish()
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

    /// Extract a TAR archive
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails
    pub fn extract_archive(archive_path: &Path, output_dir: &Path) -> Result<()> {
        let file = File::open(archive_path)?;
        // TarReader requires Read+Seek, so read file into memory and use Cursor
        let mut file_data = Vec::new();
        {
            let mut f = file;
            f.read_to_end(&mut file_data)?;
        }
        let cursor = std::io::Cursor::new(file_data);
        let mut tar_reader = TarReader::new(cursor)
            .map_err(|e| Error::Archive(format!("Failed to read archive: {e}")))?;

        let entries = tar_reader.entries().to_vec();
        for entry in &entries {
            let target_path = output_dir.join(&entry.name);
            if entry.is_dir() {
                std::fs::create_dir_all(&target_path)?;
            } else if entry.is_file() {
                if let Some(parent) = target_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let data = tar_reader
                    .extract_to_vec(entry)
                    .map_err(|e| Error::Archive(format!("Failed to extract entry: {e}")))?;
                std::fs::write(&target_path, &data)?;
            }
        }

        Ok(())
    }

    /// List contents of a TAR archive
    ///
    /// # Errors
    ///
    /// Returns an error if the archive cannot be read
    pub fn list_contents(archive_path: &Path) -> Result<Vec<PathBuf>> {
        let file = File::open(archive_path)?;
        let mut file_data = Vec::new();
        {
            let mut f = file;
            f.read_to_end(&mut file_data)?;
        }
        let cursor = std::io::Cursor::new(file_data);
        let tar_reader = TarReader::new(cursor)
            .map_err(|e| Error::Archive(format!("Failed to read archive: {e}")))?;

        let contents = tar_reader
            .entries()
            .iter()
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

        let checksum_path = output.with_extension("tar.sha256");
        let mut checksum_file = File::create(checksum_path)?;
        writeln!(
            checksum_file,
            "{}  {}",
            checksum,
            output.file_name().unwrap_or_default().to_string_lossy()
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_create_tar_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");

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

        let archiver = TarArchiver::new();
        let checksum = archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }

    #[test]
    fn test_extract_tar_archive() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");
        let extract_dir = temp_dir.path().join("extracted");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"Extract test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("test.txt"))];

        let archiver = TarArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        TarArchiver::extract_archive(&archive_path, &extract_dir)
            .expect("operation should succeed");

        assert!(extract_dir.join("test.txt").exists());
    }

    #[test]
    fn test_list_tar_contents() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let archive_path = temp_dir.path().join("test.tar");

        let mut test_file = NamedTempFile::new().expect("operation should succeed");
        test_file
            .write_all(b"List test")
            .expect("operation should succeed");
        test_file.flush().expect("operation should succeed");

        let files = vec![(test_file.path().to_path_buf(), PathBuf::from("listed.txt"))];

        let archiver = TarArchiver::new();
        archiver
            .create_archive(&archive_path, &files)
            .expect("operation should succeed");

        let contents = TarArchiver::list_contents(&archive_path).expect("operation should succeed");
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0], PathBuf::from("listed.txt"));
    }

    #[test]
    fn test_create_from_directory() {
        let temp_dir = TempDir::new().expect("operation should succeed");
        let source_dir = temp_dir.path().join("source");
        fs::create_dir_all(&source_dir).expect("operation should succeed");

        let test_file = source_dir.join("test.txt");
        fs::write(&test_file, b"Directory archive test").expect("operation should succeed");

        let archive_path = temp_dir.path().join("dir.tar");
        let archiver = TarArchiver::new();
        let checksum = archiver
            .create_from_directory(&archive_path, &source_dir)
            .expect("operation should succeed");

        assert!(!checksum.is_empty());
        assert!(archive_path.exists());
    }
}
