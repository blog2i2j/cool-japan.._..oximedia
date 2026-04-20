//! Buffered asynchronous (sync stub) file reader for OxiMedia I/O.
//!
//! [`BufferedAsyncReader`] wraps a standard file handle with an in-memory
//! read-ahead buffer. On WASM targets (no filesystem) the reader is stubbed
//! out so that the crate still compiles.
//!
//! # Design
//!
//! - The reader is **synchronous under the hood** (uses `std::fs::File`) but
//!   presents a simple chunk-oriented API that is easy to wrap in an async
//!   executor if needed.
//! - Chunks are at most `buf_size` bytes; the final chunk may be shorter.
//! - [`BufferedAsyncReader::read_chunk`] returns `Some(Vec<u8>)` while data
//!   remains and `None` at EOF.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_io::async_reader::BufferedAsyncReader;
//!
//! let mut reader = BufferedAsyncReader::new("/tmp/video.mp4", 65536)
//!     .expect("should open");
//!
//! while let Some(chunk) = reader.read_chunk() {
//!     println!("got {} bytes", chunk.len());
//! }
//! ```

#![allow(dead_code)]

use std::io::{self, Read};

/// Result type for [`BufferedAsyncReader`] operations.
pub type AsyncReaderResult<T> = Result<T, io::Error>;

/// A synchronous-file–backed chunked reader that mimics an async interface.
///
/// Internally holds an open `std::fs::File` and reads up to `buf_size` bytes
/// per [`read_chunk`](Self::read_chunk) call.
pub struct BufferedAsyncReader {
    /// The underlying file handle.
    #[cfg(not(target_arch = "wasm32"))]
    file: std::fs::File,
    /// Maximum bytes to return per chunk.
    buf_size: usize,
    /// Set to `true` once EOF has been reached.
    exhausted: bool,
    /// Total bytes consumed so far.
    bytes_read: u64,
}

impl BufferedAsyncReader {
    /// Open the file at `path` and create a reader with the given `buf_size`.
    ///
    /// # Errors
    ///
    /// Returns an [`io::Error`] if the file cannot be opened.
    ///
    /// # Panics
    ///
    /// Will panic on WASM (filesystem not available); use the `wasm32` feature
    /// gate to avoid calling this on those targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new(path: &str, buf_size: usize) -> AsyncReaderResult<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            file,
            buf_size: buf_size.max(1),
            exhausted: false,
            bytes_read: 0,
        })
    }

    /// WASM stub — always returns an unsupported error.
    #[cfg(target_arch = "wasm32")]
    pub fn new(_path: &str, buf_size: usize) -> AsyncReaderResult<Self> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "BufferedAsyncReader: filesystem not available on wasm32",
        ))
    }

    /// Read the next chunk from the file.
    ///
    /// Returns `Some(Vec<u8>)` containing up to `buf_size` bytes, or `None`
    /// at end-of-file. On an I/O error the method returns `None` and sets the
    /// internal `exhausted` flag so that subsequent calls also return `None`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_chunk(&mut self) -> Option<Vec<u8>> {
        if self.exhausted {
            return None;
        }
        let mut buf = vec![0u8; self.buf_size];
        match self.file.read(&mut buf) {
            Ok(0) => {
                self.exhausted = true;
                None
            }
            Ok(n) => {
                buf.truncate(n);
                self.bytes_read += n as u64;
                Some(buf)
            }
            Err(_) => {
                self.exhausted = true;
                None
            }
        }
    }

    /// WASM stub — always returns `None`.
    #[cfg(target_arch = "wasm32")]
    pub fn read_chunk(&mut self) -> Option<Vec<u8>> {
        None
    }

    /// Total bytes consumed so far.
    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Returns `true` if the end of the file has been reached.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.exhausted
    }

    /// The configured chunk size in bytes.
    #[must_use]
    pub fn buf_size(&self) -> usize {
        self.buf_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_chunks_from_temp_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_io_async_reader_test.bin");
        let data: Vec<u8> = (0u8..=255).collect(); // 256 bytes
        std::fs::write(&path, &data).expect("write");

        let path_str = path.to_string_lossy().to_string();
        let mut reader = BufferedAsyncReader::new(&path_str, 64).expect("open");

        let mut collected: Vec<u8> = Vec::new();
        while let Some(chunk) = reader.read_chunk() {
            collected.extend_from_slice(&chunk);
        }

        assert_eq!(collected, data);
        assert!(reader.is_exhausted());
        assert_eq!(reader.bytes_read(), 256);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_read_chunk_returns_none_after_eof() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_io_async_reader_eof_test.bin");
        std::fs::write(&path, b"abc").expect("write");

        let path_str = path.to_string_lossy().to_string();
        let mut reader = BufferedAsyncReader::new(&path_str, 1024).expect("open");

        let first = reader.read_chunk();
        assert!(first.is_some());
        assert_eq!(first.unwrap(), b"abc");

        let second = reader.read_chunk();
        assert!(second.is_none());
        assert!(reader.is_exhausted());

        // Repeated calls after exhaustion also return None
        assert!(reader.read_chunk().is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_nonexistent_file_returns_error() {
        let missing = std::env::temp_dir().join("oximedia-io-async-nonexistent_xyz.bin");
        let result = BufferedAsyncReader::new(&missing, 1024);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_io_async_reader_empty.bin");
        std::fs::write(&path, b"").expect("write");

        let path_str = path.to_string_lossy().to_string();
        let mut reader = BufferedAsyncReader::new(&path_str, 64).expect("open");

        assert!(reader.read_chunk().is_none());
        assert!(reader.is_exhausted());
        assert_eq!(reader.bytes_read(), 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_buf_size_accessor() {
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_io_buf_size_test.bin");
        std::fs::write(&path, b"data").expect("write");

        let path_str = path.to_string_lossy().to_string();
        let reader = BufferedAsyncReader::new(&path_str, 4096).expect("open");
        assert_eq!(reader.buf_size(), 4096);

        let _ = std::fs::remove_file(&path);
    }
}
