//! High-throughput file copy engine.
//!
//! Provides configurable copy strategies (buffered, direct, sparse) and
//! reports measured throughput after a copy job completes.

#![allow(dead_code)]

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Strategy used by the [`CopyEngine`] when copying data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CopyMode {
    /// Standard buffered copy using the OS page cache.
    #[default]
    Buffered,
    /// Attempt to preserve sparse regions (holes) in the source file.
    Sparse,
    /// Copy in fixed-size chunks, useful for progress reporting.
    Chunked,
}

impl std::fmt::Display for CopyMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CopyMode::Buffered => write!(f, "buffered"),
            CopyMode::Sparse => write!(f, "sparse"),
            CopyMode::Chunked => write!(f, "chunked"),
        }
    }
}

/// Describes a single file copy operation.
#[derive(Debug, Clone)]
pub struct CopyJob {
    /// Source path.
    pub src: PathBuf,
    /// Destination path.
    pub dst: PathBuf,
    /// Copy strategy to use.
    pub mode: CopyMode,
    /// Chunk size in bytes (used only for [`CopyMode::Chunked`]).
    pub chunk_size: usize,
    /// Whether to overwrite an existing destination file.
    pub overwrite: bool,
}

impl CopyJob {
    /// Create a new [`CopyJob`] with default settings.
    pub fn new(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Self {
        Self {
            src: src.as_ref().to_path_buf(),
            dst: dst.as_ref().to_path_buf(),
            mode: CopyMode::default(),
            chunk_size: 64 * 1024,
            overwrite: true,
        }
    }

    /// Set the copy mode.
    #[must_use]
    pub fn with_mode(mut self, mode: CopyMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the chunk size (bytes).
    #[must_use]
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set whether to overwrite the destination.
    #[must_use]
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.overwrite = overwrite;
        self
    }
}

/// Result of a completed copy operation.
#[derive(Debug, Clone)]
pub struct CopyResult {
    /// Total bytes copied.
    pub bytes_copied: u64,
    /// Wall-clock duration in seconds.
    pub elapsed_secs: f64,
}

impl CopyResult {
    /// Throughput in mebibytes per second (MiB/s).
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        if self.elapsed_secs == 0.0 {
            return 0.0;
        }
        (self.bytes_copied as f64) / (self.elapsed_secs * 1024.0 * 1024.0)
    }
}

/// Executes [`CopyJob`]s and measures throughput.
pub struct CopyEngine;

impl CopyEngine {
    /// Create a new [`CopyEngine`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Execute a single [`CopyJob`] and return a [`CopyResult`].
    ///
    /// # Errors
    /// Returns `io::Error` if the source cannot be read, the destination
    /// cannot be written, or the job specifies no-overwrite and the
    /// destination already exists.
    pub fn run(&self, job: &CopyJob) -> io::Result<CopyResult> {
        if !job.overwrite && job.dst.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "destination already exists and overwrite is disabled",
            ));
        }

        let start = Instant::now();
        let bytes_copied = match job.mode {
            CopyMode::Buffered | CopyMode::Sparse => std::fs::copy(&job.src, &job.dst)?,
            CopyMode::Chunked => Self::copy_chunked(&job.src, &job.dst, job.chunk_size)?,
        };
        let elapsed_secs = start.elapsed().as_secs_f64();

        Ok(CopyResult {
            bytes_copied,
            elapsed_secs,
        })
    }

    /// Copy in fixed chunks, returning total bytes written.
    fn copy_chunked(src: &Path, dst: &Path, chunk_size: usize) -> io::Result<u64> {
        let mut reader = std::fs::File::open(src)?;
        let mut writer = std::fs::File::create(dst)?;
        let mut buf = vec![0u8; chunk_size];
        let mut total: u64 = 0;

        loop {
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            writer.write_all(&buf[..n])?;
            total += n as u64;
        }
        Ok(total)
    }
}

impl Default for CopyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_src(content: &[u8]) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        f.write_all(content).expect("failed to write");
        let p = f.path().to_path_buf();
        (f, p)
    }

    fn dst_path() -> (tempfile::NamedTempFile, PathBuf) {
        let f = tempfile::NamedTempFile::new().expect("failed to create temp file");
        let p = f.path().to_path_buf();
        (f, p)
    }

    #[test]
    fn test_copy_buffered_basic() {
        let (_sf, src) = make_src(b"hello world");
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let job = CopyJob::new(&src, &dst);
        let result = engine.run(&job).expect("copy should succeed");
        assert_eq!(result.bytes_copied, 11);
        assert_eq!(
            std::fs::read(&dst).expect("failed to read file"),
            b"hello world"
        );
    }

    #[test]
    fn test_copy_chunked() {
        let data: Vec<u8> = (0u8..=255).collect();
        let (_sf, src) = make_src(&data);
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let job = CopyJob::new(&src, &dst)
            .with_mode(CopyMode::Chunked)
            .with_chunk_size(32);
        let result = engine.run(&job).expect("copy should succeed");
        assert_eq!(result.bytes_copied, 256);
        assert_eq!(std::fs::read(&dst).expect("failed to read file"), data);
    }

    #[test]
    fn test_copy_sparse_mode() {
        let (_sf, src) = make_src(b"sparse data");
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let job = CopyJob::new(&src, &dst).with_mode(CopyMode::Sparse);
        let result = engine.run(&job).expect("copy should succeed");
        assert_eq!(result.bytes_copied, 11);
    }

    #[test]
    fn test_copy_no_overwrite_existing_fails() {
        let (_sf, src) = make_src(b"original");
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        // First copy succeeds
        engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        // Second copy with overwrite=false should fail
        let job = CopyJob::new(&src, &dst).with_overwrite(false);
        let err = engine.run(&job).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn test_copy_overwrite_existing_succeeds() {
        let (_sf, src) = make_src(b"new content");
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        // Overwrite enabled (default)
        let result = engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        assert_eq!(result.bytes_copied, 11);
    }

    #[test]
    fn test_throughput_mbps_nonzero() {
        let data = vec![42u8; 1024 * 1024];
        let (_sf, src) = make_src(&data);
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let result = engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        assert!(result.throughput_mbps() > 0.0);
    }

    #[test]
    fn test_throughput_zero_elapsed() {
        let r = CopyResult {
            bytes_copied: 1000,
            elapsed_secs: 0.0,
        };
        assert_eq!(r.throughput_mbps(), 0.0);
    }

    #[test]
    fn test_copy_empty_file() {
        let (_sf, src) = make_src(b"");
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let result = engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        assert_eq!(result.bytes_copied, 0);
    }

    #[test]
    fn test_copy_mode_default() {
        assert_eq!(CopyMode::default(), CopyMode::Buffered);
    }

    #[test]
    fn test_copy_mode_display() {
        assert_eq!(CopyMode::Buffered.to_string(), "buffered");
        assert_eq!(CopyMode::Sparse.to_string(), "sparse");
        assert_eq!(CopyMode::Chunked.to_string(), "chunked");
    }

    #[test]
    fn test_copy_job_builder_chain() {
        let job = CopyJob::new("/src", "/dst")
            .with_mode(CopyMode::Chunked)
            .with_chunk_size(8192)
            .with_overwrite(false);
        assert_eq!(job.mode, CopyMode::Chunked);
        assert_eq!(job.chunk_size, 8192);
        assert!(!job.overwrite);
    }

    #[test]
    fn test_copy_nonexistent_src_fails() {
        let engine = CopyEngine::new();
        let job = CopyJob::new("/nonexistent/file.bin", "/tmp/dst_oxi_test.bin");
        assert!(engine.run(&job).is_err());
    }

    #[test]
    fn test_elapsed_secs_positive() {
        let data = vec![0u8; 512];
        let (_sf, src) = make_src(&data);
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let result = engine
            .run(&CopyJob::new(&src, &dst))
            .expect("copy should succeed");
        assert!(result.elapsed_secs >= 0.0);
    }

    #[test]
    fn test_copy_engine_default() {
        let _engine = CopyEngine::default();
    }

    #[test]
    fn test_copy_large_chunk_larger_than_file() {
        let data = b"small";
        let (_sf, src) = make_src(data);
        let (_df, dst) = dst_path();
        let engine = CopyEngine::new();
        let job = CopyJob::new(&src, &dst)
            .with_mode(CopyMode::Chunked)
            .with_chunk_size(1024 * 1024);
        let result = engine.run(&job).expect("copy should succeed");
        assert_eq!(result.bytes_copied, data.len() as u64);
    }
}
