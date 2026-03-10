//! WASM-compatible I/O implementations.
//!
//! This module provides I/O abstractions that work in the browser environment.
//! Since WASM doesn't have file system access by default, we use in-memory
//! buffers backed by JavaScript `Uint8Array` or `ArrayBuffer`.

use bytes::Bytes;
use oximedia_core::{OxiError, OxiResult};
use std::io::{Cursor, Read, Seek, SeekFrom};

/// A WASM-compatible byte source backed by in-memory data.
///
/// This type wraps a `Cursor<Bytes>` to provide synchronous read and seek
/// operations on data passed from JavaScript.
///
/// Unlike the async `MediaSource` trait used in the main library, this
/// provides synchronous operations suitable for the WASM single-threaded
/// environment.
#[allow(dead_code)]
pub struct ByteSource {
    cursor: Cursor<Bytes>,
    size: u64,
}

#[allow(dead_code)]
impl ByteSource {
    /// Creates a new `ByteSource` from bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - The byte data to wrap
    #[must_use]
    pub fn new(data: Bytes) -> Self {
        let size = data.len() as u64;
        Self {
            cursor: Cursor::new(data),
            size,
        }
    }

    /// Creates a new `ByteSource` from a vector.
    ///
    /// # Arguments
    ///
    /// * `data` - The byte data to wrap
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::new(Bytes::from(data))
    }

    /// Returns the total size of the source in bytes.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// Returns the current position in the source.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.cursor.position()
    }

    /// Reads data into the provided buffer.
    ///
    /// Returns the number of bytes read.
    ///
    /// # Errors
    ///
    /// Returns an error if the read operation fails.
    pub fn read(&mut self, buf: &mut [u8]) -> OxiResult<usize> {
        self.cursor.read(buf).map_err(OxiError::Io)
    }

    /// Reads exact number of bytes into the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if EOF is reached before filling the buffer.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> OxiResult<()> {
        self.cursor.read_exact(buf).map_err(OxiError::Io)
    }

    /// Seeks to a position in the source.
    ///
    /// # Errors
    ///
    /// Returns an error if seeking to an invalid position.
    pub fn seek(&mut self, pos: SeekFrom) -> OxiResult<u64> {
        self.cursor.seek(pos).map_err(OxiError::Io)
    }

    /// Returns a reference to the underlying bytes.
    #[must_use]
    pub fn get_ref(&self) -> &Bytes {
        self.cursor.get_ref()
    }

    /// Checks if we're at the end of the source.
    #[must_use]
    pub fn is_eof(&self) -> bool {
        self.cursor.position() >= self.size
    }

    /// Reads a slice of bytes without advancing the position.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset to read from
    /// * `len` - The number of bytes to read
    ///
    /// # Errors
    ///
    /// Returns an error if the range is out of bounds.
    pub fn peek(&self, offset: u64, len: usize) -> OxiResult<Bytes> {
        let start = offset as usize;
        let end = start
            .checked_add(len)
            .ok_or_else(|| OxiError::InvalidData("Overflow in peek".to_string()))?;

        if end > self.size as usize {
            return Err(OxiError::InvalidData(
                "Peek range out of bounds".to_string(),
            ));
        }

        let bytes = self.cursor.get_ref();
        Ok(bytes.slice(start..end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_source_new() {
        let data = Bytes::from_static(b"Hello, World!");
        let source = ByteSource::new(data);
        assert_eq!(source.size(), 13);
        assert_eq!(source.position(), 0);
    }

    #[test]
    fn test_byte_source_read() {
        let data = Bytes::from_static(b"Hello, World!");
        let mut source = ByteSource::new(data);

        let mut buf = [0u8; 5];
        let n = source.read(&mut buf).expect("read should succeed");
        assert_eq!(n, 5);
        assert_eq!(&buf, b"Hello");
        assert_eq!(source.position(), 5);
    }

    #[test]
    fn test_byte_source_read_exact() {
        let data = Bytes::from_static(b"Hello, World!");
        let mut source = ByteSource::new(data);

        let mut buf = [0u8; 13];
        source.read_exact(&mut buf).expect("read should succeed");
        assert_eq!(&buf, b"Hello, World!");
    }

    #[test]
    fn test_byte_source_seek() {
        let data = Bytes::from_static(b"Hello, World!");
        let mut source = ByteSource::new(data);

        source
            .seek(SeekFrom::Start(7))
            .expect("seek should succeed");
        assert_eq!(source.position(), 7);

        let mut buf = [0u8; 5];
        source.read(&mut buf).expect("read should succeed");
        assert_eq!(&buf, b"World");
    }

    #[test]
    fn test_byte_source_peek() {
        let data = Bytes::from_static(b"Hello, World!");
        let source = ByteSource::new(data);

        let peeked = source.peek(7, 5).expect("peek should succeed");
        assert_eq!(&peeked[..], b"World");
        assert_eq!(source.position(), 0); // Position unchanged
    }

    #[test]
    fn test_byte_source_is_eof() {
        let data = Bytes::from_static(b"Hi");
        let mut source = ByteSource::new(data);

        assert!(!source.is_eof());
        source.seek(SeekFrom::End(0)).expect("seek should succeed");
        assert!(source.is_eof());
    }
}
