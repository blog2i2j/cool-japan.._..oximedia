//! File metadata probing for media I/O.
//!
//! Provides lightweight inspection of file attributes (size, kind, timestamps)
//! without opening the full media container.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// High-level classification of a file by its role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    /// A regular data file.
    Regular,
    /// A symbolic link.
    Symlink,
    /// A directory entry.
    Directory,
    /// A special device file (block or character).
    Device,
    /// Something the prober could not classify.
    Unknown,
}

/// Metadata snapshot of a file on disk.
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Byte size of the file.
    pub size_bytes: u64,
    /// High-level kind classification.
    pub kind: FileKind,
    /// Last-modification time, if available.
    pub modified: Option<SystemTime>,
    /// Creation time, if available.
    pub created: Option<SystemTime>,
    /// Whether the process has read permission.
    pub readable: bool,
    /// Whether the process has write permission.
    pub writable: bool,
}

impl FileMetadata {
    /// Return the file size in mebibytes (MiB).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn size_mib(&self) -> f64 {
        self.size_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Return `true` if the file is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }

    /// Return `true` if this entry is a regular file.
    #[must_use]
    pub fn is_regular(&self) -> bool {
        self.kind == FileKind::Regular
    }
}

/// Reads [`FileMetadata`] from the filesystem.
pub struct FileMetadataReader;

impl FileMetadataReader {
    /// Probe a path and return its [`FileMetadata`].
    ///
    /// # Errors
    /// Returns an `std::io::Error` if the path does not exist or is
    /// otherwise inaccessible.
    pub fn probe(path: impl AsRef<Path>) -> std::io::Result<FileMetadata> {
        let path = path.as_ref();
        let meta = std::fs::metadata(path)?;

        let kind = if meta.is_file() {
            FileKind::Regular
        } else if meta.is_dir() {
            FileKind::Directory
        } else if meta.file_type().is_symlink() {
            FileKind::Symlink
        } else {
            FileKind::Unknown
        };

        let readable = std::fs::File::open(path).is_ok();
        let writable = !meta.permissions().readonly();

        Ok(FileMetadata {
            path: path.to_path_buf(),
            size_bytes: meta.len(),
            kind,
            modified: meta.modified().ok(),
            created: meta.created().ok(),
            readable,
            writable,
        })
    }

    /// Probe a path without following symlinks (uses `symlink_metadata`).
    ///
    /// # Errors
    ///
    /// Returns an I/O error if the path cannot be accessed.
    pub fn probe_no_follow(path: impl AsRef<Path>) -> std::io::Result<FileMetadata> {
        let path = path.as_ref();
        let meta = std::fs::symlink_metadata(path)?;

        let kind = if meta.is_file() {
            FileKind::Regular
        } else if meta.is_dir() {
            FileKind::Directory
        } else if meta.file_type().is_symlink() {
            FileKind::Symlink
        } else {
            FileKind::Unknown
        };

        let readable = std::fs::File::open(path).is_ok();
        let writable = !meta.permissions().readonly();

        Ok(FileMetadata {
            path: path.to_path_buf(),
            size_bytes: meta.len(),
            kind,
            modified: meta.modified().ok(),
            created: meta.created().ok(),
            readable,
            writable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(content: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        f.write_all(content).expect("failed to write");
        let p = f.path().to_path_buf();
        (f, p)
    }

    #[test]
    fn test_probe_regular_file() {
        let (_f, path) = write_temp(b"hello");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert_eq!(meta.kind, FileKind::Regular);
    }

    #[test]
    fn test_probe_size() {
        let data = b"0123456789";
        let (_f, path) = write_temp(data);
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert_eq!(meta.size_bytes, data.len() as u64);
    }

    #[test]
    fn test_probe_is_not_empty() {
        let (_f, path) = write_temp(b"x");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_probe_empty_file() {
        let (_f, path) = write_temp(b"");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert!(meta.is_empty());
    }

    #[test]
    fn test_probe_is_regular() {
        let (_f, path) = write_temp(b"data");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert!(meta.is_regular());
    }

    #[test]
    fn test_probe_directory() {
        let dir = tempfile::tempdir().expect("failed to create temp file");
        let meta = FileMetadataReader::probe(dir.path()).expect("operation should succeed");
        assert_eq!(meta.kind, FileKind::Directory);
    }

    #[test]
    fn test_probe_readable() {
        let (_f, path) = write_temp(b"read me");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert!(meta.readable);
    }

    #[test]
    fn test_probe_path_stored() {
        let (_f, path) = write_temp(b"path check");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert_eq!(meta.path, path);
    }

    #[test]
    fn test_size_mib() {
        let (_f, path) = write_temp(&vec![0u8; 1024 * 1024]);
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        let mib = meta.size_mib();
        assert!((mib - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_size_mib_small() {
        let (_f, path) = write_temp(b"small");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        assert!(meta.size_mib() < 1.0);
    }

    #[test]
    fn test_probe_nonexistent_returns_error() {
        let result = FileMetadataReader::probe("/nonexistent/path/file.mp4");
        assert!(result.is_err());
    }

    #[test]
    fn test_file_kind_variants() {
        assert_ne!(FileKind::Regular, FileKind::Directory);
        assert_ne!(FileKind::Symlink, FileKind::Unknown);
        assert_ne!(FileKind::Device, FileKind::Regular);
    }

    #[test]
    fn test_probe_no_follow_regular_file() {
        let (_f, path) = write_temp(b"no follow");
        let meta = FileMetadataReader::probe_no_follow(&path).expect("operation should succeed");
        assert_eq!(meta.kind, FileKind::Regular);
    }

    #[test]
    fn test_modified_timestamp_present() {
        let (_f, path) = write_temp(b"ts check");
        let meta = FileMetadataReader::probe(&path).expect("operation should succeed");
        // Most filesystems support mtime
        assert!(meta.modified.is_some());
    }

    #[test]
    fn test_probe_directory_size_zero_or_nonzero() {
        let dir = tempfile::tempdir().expect("failed to create temp file");
        let meta = FileMetadataReader::probe(dir.path()).expect("operation should succeed");
        // Just verify we get back a valid FileMetadata for a dir
        assert_eq!(meta.kind, FileKind::Directory);
    }
}
