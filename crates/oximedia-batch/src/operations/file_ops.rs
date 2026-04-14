//! File operation implementations

use crate::error::{BatchError, Result};
use crate::job::BatchJob;
use crate::operations::OperationExecutor;
use async_trait::async_trait;
use oxiarc_archive::zip::{ZipCompressionLevel, ZipWriter};
use oxiarc_archive::TarWriter;
use oxiarc_deflate::GzipStreamEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// File operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileOperation {
    /// Copy files
    Copy {
        /// Overwrite existing files
        overwrite: bool,
    },
    /// Move files
    Move {
        /// Overwrite existing files
        overwrite: bool,
    },
    /// Rename files
    Rename {
        /// Name template
        template: String,
    },
    /// Delete files
    Delete {
        /// Confirm deletion
        confirm: bool,
    },
    /// Create archive
    Archive {
        /// Archive format
        format: ArchiveFormat,
        /// Compression level (0-9)
        compression: u32,
    },
    /// Extract archive
    Extract {
        /// Destination directory
        destination: PathBuf,
    },
    /// Verify file integrity
    Verify {
        /// Verification method
        method: VerifyMethod,
    },
    /// Calculate checksum
    Checksum {
        /// Hash algorithm
        algorithm: HashAlgorithm,
    },
}

/// Archive formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchiveFormat {
    /// ZIP format
    Zip,
    /// TAR format
    Tar,
    /// TAR.GZ format
    TarGz,
}

/// Verification methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerifyMethod {
    /// Checksum verification
    Checksum {
        /// Expected checksum
        expected: String,
        /// Hash algorithm
        algorithm: HashAlgorithm,
    },
    /// File size verification
    Size {
        /// Expected size in bytes
        expected: u64,
    },
}

/// Hash algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashAlgorithm {
    /// SHA-256
    Sha256,
    /// MD5
    Md5,
}

/// File operation executor
pub struct FileOperationExecutor;

impl FileOperationExecutor {
    /// Create a new file operation executor
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn copy_file(src: &Path, dst: &Path, overwrite: bool) -> Result<()> {
        if dst.exists() && !overwrite {
            return Err(BatchError::FileOperationError(format!(
                "Destination file already exists: {}",
                dst.display()
            )));
        }

        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::copy(src, dst)?;
        Ok(())
    }

    fn move_file(src: &Path, dst: &Path, overwrite: bool) -> Result<()> {
        if dst.exists() && !overwrite {
            return Err(BatchError::FileOperationError(format!(
                "Destination file already exists: {}",
                dst.display()
            )));
        }

        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::rename(src, dst)?;
        Ok(())
    }

    fn delete_file(path: &Path) -> Result<()> {
        if path.is_file() {
            fs::remove_file(path)?;
        } else if path.is_dir() {
            fs::remove_dir_all(path)?;
        }
        Ok(())
    }

    fn create_zip_archive(files: &[PathBuf], output: &Path, compression: u32) -> Result<()> {
        let file = fs::File::create(output)?;
        let mut zip = ZipWriter::new(file);

        let level = match compression {
            0 => ZipCompressionLevel::Store,
            1..=3 => ZipCompressionLevel::Fast,
            4..=7 => ZipCompressionLevel::Normal,
            _ => ZipCompressionLevel::Best,
        };
        zip.set_compression(level);

        for input_path in files {
            if input_path.is_file() {
                let file_name = input_path
                    .file_name()
                    .ok_or_else(|| BatchError::FileOperationError("Invalid file name".to_string()))?
                    .to_str()
                    .ok_or_else(|| {
                        BatchError::FileOperationError("Non-UTF8 file name".to_string())
                    })?;

                let mut f = fs::File::open(input_path)?;
                let mut buffer = Vec::new();
                f.read_to_end(&mut buffer)?;
                zip.add_file(file_name, &buffer)
                    .map_err(|e| BatchError::FileOperationError(format!("Zip add error: {e}")))?;
            } else if input_path.is_dir() {
                for entry in WalkDir::new(input_path) {
                    let entry = entry
                        .map_err(|e| BatchError::FileOperationError(format!("Walk error: {e}")))?;
                    let path = entry.path();
                    if path.is_file() {
                        let name = path
                            .strip_prefix(input_path)
                            .map_err(|e| {
                                BatchError::FileOperationError(format!("Path error: {e}"))
                            })?
                            .to_str()
                            .ok_or_else(|| {
                                BatchError::FileOperationError("Non-UTF8 path".to_string())
                            })?;
                        let mut f = fs::File::open(path)?;
                        let mut buffer = Vec::new();
                        f.read_to_end(&mut buffer)?;
                        zip.add_file(name, &buffer).map_err(|e| {
                            BatchError::FileOperationError(format!("Zip add error: {e}"))
                        })?;
                    }
                }
            }
        }

        zip.finish()
            .map_err(|e| BatchError::FileOperationError(format!("Zip finish error: {e}")))?;
        Ok(())
    }

    fn create_tar_archive(files: &[PathBuf], output: &Path, gzip: bool) -> Result<()> {
        let file = fs::File::create(output)?;

        if gzip {
            let enc = GzipStreamEncoder::new(file, 6);
            let mut tar_writer = TarWriter::new(enc);
            Self::add_paths_to_tar(&mut tar_writer, files)?;
            let gzip_encoder = tar_writer.into_inner().map_err(|e| {
                BatchError::FileOperationError(format!("Failed to finish tar: {e}"))
            })?;
            gzip_encoder.finish().map_err(|e| {
                BatchError::FileOperationError(format!("Failed to finish gzip: {e}"))
            })?;
        } else {
            let mut tar_writer = TarWriter::new(file);
            Self::add_paths_to_tar(&mut tar_writer, files)?;
            tar_writer.finish().map_err(|e| {
                BatchError::FileOperationError(format!("Failed to finish tar: {e}"))
            })?;
        }

        Ok(())
    }

    fn add_paths_to_tar<W: std::io::Write>(
        tar_writer: &mut TarWriter<W>,
        files: &[PathBuf],
    ) -> Result<()> {
        for input_path in files {
            if input_path.is_file() {
                let name = input_path
                    .file_name()
                    .ok_or_else(|| BatchError::FileOperationError("Invalid file name".to_string()))?
                    .to_str()
                    .ok_or_else(|| {
                        BatchError::FileOperationError("Non-UTF8 file name".to_string())
                    })?;
                let mut f = fs::File::open(input_path)?;
                let mut buffer = Vec::new();
                f.read_to_end(&mut buffer)?;
                tar_writer
                    .add_file(name, &buffer)
                    .map_err(|e| BatchError::FileOperationError(format!("Tar add error: {e}")))?;
            } else if input_path.is_dir() {
                let dir_name = input_path
                    .file_name()
                    .ok_or_else(|| BatchError::FileOperationError("Invalid dir name".to_string()))?
                    .to_str()
                    .ok_or_else(|| {
                        BatchError::FileOperationError("Non-UTF8 dir name".to_string())
                    })?;
                for entry in WalkDir::new(input_path) {
                    let entry = entry
                        .map_err(|e| BatchError::FileOperationError(format!("Walk error: {e}")))?;
                    let path = entry.path();
                    if path.is_file() {
                        let rel = path.strip_prefix(input_path).map_err(|e| {
                            BatchError::FileOperationError(format!("Path error: {e}"))
                        })?;
                        let archive_name = format!(
                            "{}/{}",
                            dir_name,
                            rel.to_str().ok_or_else(|| {
                                BatchError::FileOperationError("Non-UTF8 path".to_string())
                            })?
                        );
                        let mut f = fs::File::open(path)?;
                        let mut buffer = Vec::new();
                        f.read_to_end(&mut buffer)?;
                        tar_writer.add_file(&archive_name, &buffer).map_err(|e| {
                            BatchError::FileOperationError(format!("Tar add error: {e}"))
                        })?;
                    } else if path.is_dir() && path != input_path {
                        let rel = path.strip_prefix(input_path).map_err(|e| {
                            BatchError::FileOperationError(format!("Path error: {e}"))
                        })?;
                        let archive_name = format!(
                            "{}/{}",
                            dir_name,
                            rel.to_str().ok_or_else(|| {
                                BatchError::FileOperationError("Non-UTF8 path".to_string())
                            })?
                        );
                        tar_writer.add_directory(&archive_name).map_err(|e| {
                            BatchError::FileOperationError(format!("Tar add dir error: {e}"))
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    fn calculate_sha256(path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0; 8192];

        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        Ok(hex::encode(hasher.finalize()))
    }
}

impl Default for FileOperationExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OperationExecutor for FileOperationExecutor {
    async fn execute(&self, job: &BatchJob, input_files: &[PathBuf]) -> Result<Vec<PathBuf>> {
        let start = std::time::Instant::now();
        let mut output_files = Vec::new();

        match &job.operation {
            crate::job::BatchOperation::FileOp { operation } => match operation {
                FileOperation::Copy { overwrite } => {
                    for input_file in input_files {
                        for output_spec in &job.outputs {
                            let output_path = PathBuf::from(&output_spec.template);
                            Self::copy_file(input_file, &output_path, *overwrite)?;
                            output_files.push(output_path);
                        }
                    }
                }
                FileOperation::Move { overwrite } => {
                    for input_file in input_files {
                        for output_spec in &job.outputs {
                            let output_path = PathBuf::from(&output_spec.template);
                            Self::move_file(input_file, &output_path, *overwrite)?;
                            output_files.push(output_path);
                        }
                    }
                }
                FileOperation::Delete { confirm: _ } => {
                    for input_file in input_files {
                        Self::delete_file(input_file)?;
                    }
                }
                FileOperation::Archive {
                    format,
                    compression,
                } => {
                    if let Some(output_spec) = job.outputs.first() {
                        let output_path = PathBuf::from(&output_spec.template);
                        match format {
                            ArchiveFormat::Zip => {
                                Self::create_zip_archive(input_files, &output_path, *compression)?;
                            }
                            ArchiveFormat::Tar => {
                                Self::create_tar_archive(input_files, &output_path, false)?;
                            }
                            ArchiveFormat::TarGz => {
                                Self::create_tar_archive(input_files, &output_path, true)?;
                            }
                        }
                        output_files.push(output_path);
                    }
                }
                FileOperation::Checksum { algorithm: _ } => {
                    for input_file in input_files {
                        let checksum = Self::calculate_sha256(input_file)?;
                        let checksum_file = input_file.with_extension("sha256");
                        fs::write(&checksum_file, checksum)?;
                        output_files.push(checksum_file);
                    }
                }
                _ => {
                    return Err(BatchError::FileOperationError(
                        "Unsupported operation".to_string(),
                    ));
                }
            },
            _ => {
                return Err(BatchError::FileOperationError(
                    "Not a file operation".to_string(),
                ));
            }
        }

        tracing::info!("File operation completed in {:?}", start.elapsed());

        Ok(output_files)
    }

    fn validate(&self, job: &BatchJob) -> Result<()> {
        if !matches!(job.operation, crate::job::BatchOperation::FileOp { .. }) {
            return Err(BatchError::ValidationError(
                "Not a file operation".to_string(),
            ));
        }
        Ok(())
    }

    fn estimate_duration(&self, _job: &BatchJob, _input_files: &[PathBuf]) -> Option<u64> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_copy_file() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let src = temp_dir.path().join("source.txt");
        let dst = temp_dir.path().join("dest.txt");

        fs::write(&src, b"test content").expect("operation should succeed");

        let result = FileOperationExecutor::copy_file(&src, &dst, false);
        assert!(result.is_ok());
        assert!(dst.exists());

        let content = fs::read_to_string(&dst).expect("formatting should succeed");
        assert_eq!(content, "test content");
    }

    #[test]
    fn test_copy_file_no_overwrite() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let src = temp_dir.path().join("source.txt");
        let dst = temp_dir.path().join("dest.txt");

        fs::write(&src, b"test content").expect("operation should succeed");
        fs::write(&dst, b"existing content").expect("operation should succeed");

        let result = FileOperationExecutor::copy_file(&src, &dst, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_move_file() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let src = temp_dir.path().join("source.txt");
        let dst = temp_dir.path().join("dest.txt");

        fs::write(&src, b"test content").expect("operation should succeed");

        let result = FileOperationExecutor::move_file(&src, &dst, false);
        assert!(result.is_ok());
        assert!(!src.exists());
        assert!(dst.exists());
    }

    #[test]
    fn test_delete_file() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let file = temp_dir.path().join("test.txt");

        fs::write(&file, b"test content").expect("operation should succeed");

        let result = FileOperationExecutor::delete_file(&file);
        assert!(result.is_ok());
        assert!(!file.exists());
    }

    #[test]
    fn test_calculate_sha256() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let file = temp_dir.path().join("test.txt");

        fs::write(&file, b"test content").expect("operation should succeed");

        let result = FileOperationExecutor::calculate_sha256(&file);
        assert!(result.is_ok());

        let hash = result.expect("result should be valid");
        assert_eq!(hash.len(), 64); // SHA-256 is 64 hex characters
    }

    #[test]
    fn test_create_zip_archive() {
        let temp_dir = TempDir::new().expect("failed to create temp dir");
        let file1 = temp_dir.path().join("file1.txt");
        let file2 = temp_dir.path().join("file2.txt");
        let archive = temp_dir.path().join("archive.zip");

        fs::write(&file1, b"content 1").expect("operation should succeed");
        fs::write(&file2, b"content 2").expect("operation should succeed");

        let result = FileOperationExecutor::create_zip_archive(&[file1, file2], &archive, 6);
        assert!(result.is_ok());
        assert!(archive.exists());
    }
}
