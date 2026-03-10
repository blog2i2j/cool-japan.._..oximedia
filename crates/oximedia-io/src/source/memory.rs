//! In-memory media source implementation.

use super::MediaSource;
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use oximedia_core::{OxiError, OxiResult};
use std::io::SeekFrom;

/// A media source backed by an in-memory buffer.
///
/// Useful for testing, processing already-loaded data, or working with
/// data that fits entirely in memory. Supports both reading and writing.
///
/// # Example
///
/// ```
/// use oximedia_io::source::{MemorySource, MediaSource};
/// use std::io::SeekFrom;
///
/// #[tokio::main]
/// async fn main() -> oximedia_core::OxiResult<()> {
///     let data = vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9];
///     let mut source = MemorySource::from_vec(data);
///
///     let mut buffer = [0u8; 5];
///     let n = source.read(&mut buffer).await?;
///     assert_eq!(n, 5);
///     assert_eq!(&buffer, &[0, 1, 2, 3, 4]);
///
///     Ok(())
/// }
/// ```
#[derive(Clone, Debug)]
pub struct MemorySource {
    /// Read-only data (when not in writable mode).
    data: Bytes,
    /// Writable buffer (when in writable mode).
    writable_data: BytesMut,
    position: u64,
    writable: bool,
}

impl MemorySource {
    /// Creates a new `MemorySource` from a `Bytes` buffer.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::source::MemorySource;
    /// use bytes::Bytes;
    ///
    /// let bytes = Bytes::from_static(b"Hello, World!");
    /// let source = MemorySource::new(bytes);
    /// ```
    #[must_use]
    pub fn new(data: Bytes) -> Self {
        Self {
            data,
            writable_data: BytesMut::new(),
            position: 0,
            writable: false,
        }
    }

    /// Creates a new writable `MemorySource` with the given initial capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::source::MemorySource;
    ///
    /// let source = MemorySource::new_writable(1024);
    /// ```
    #[must_use]
    pub fn new_writable(capacity: usize) -> Self {
        Self {
            data: Bytes::new(),
            writable_data: BytesMut::with_capacity(capacity),
            position: 0,
            writable: true,
        }
    }

    /// Creates a new `MemorySource` from a `Vec<u8>`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::source::MemorySource;
    ///
    /// let data = vec![1, 2, 3, 4, 5];
    /// let source = MemorySource::from_vec(data);
    /// ```
    #[must_use]
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self::new(Bytes::from(data))
    }

    /// Creates a new `MemorySource` from a byte slice.
    ///
    /// The slice is copied into owned memory.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_io::source::MemorySource;
    ///
    /// let data: &[u8] = &[1, 2, 3, 4, 5];
    /// let source = MemorySource::from_slice(data);
    /// ```
    #[must_use]
    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(Bytes::copy_from_slice(data))
    }

    /// Returns a reference to the underlying data.
    #[must_use]
    pub fn data(&self) -> &Bytes {
        if self.writable {
            // This creates a temporary Bytes view; use written_data() for writable sources
            &self.data
        } else {
            &self.data
        }
    }

    /// Returns the written data (for writable sources).
    #[must_use]
    pub fn written_data(&self) -> &[u8] {
        if self.writable {
            &self.writable_data
        } else {
            &self.data
        }
    }

    /// Returns the buffer length.
    fn buffer_len(&self) -> usize {
        if self.writable {
            self.writable_data.len()
        } else {
            self.data.len()
        }
    }
}

#[async_trait]
impl MediaSource for MemorySource {
    #[allow(clippy::cast_possible_truncation)]
    async fn read(&mut self, buf: &mut [u8]) -> OxiResult<usize> {
        let pos = self.position as usize;
        let data_len = self.buffer_len();

        if pos >= data_len {
            return Ok(0); // EOF
        }

        let remaining = data_len - pos;
        let to_read = buf.len().min(remaining);

        if self.writable {
            buf[..to_read].copy_from_slice(&self.writable_data[pos..pos + to_read]);
        } else {
            buf[..to_read].copy_from_slice(&self.data[pos..pos + to_read]);
        }
        self.position += to_read as u64;

        Ok(to_read)
    }

    #[allow(clippy::cast_possible_truncation)]
    async fn write_all(&mut self, buf: &[u8]) -> OxiResult<()> {
        if !self.writable {
            return Err(OxiError::unsupported("MemorySource is not writable"));
        }

        let pos = self.position as usize;
        let end_pos = pos + buf.len();

        // Extend buffer if needed
        if end_pos > self.writable_data.len() {
            self.writable_data.resize(end_pos, 0);
        }

        // Write data
        self.writable_data[pos..end_pos].copy_from_slice(buf);
        self.position = end_pos as u64;

        Ok(())
    }

    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    async fn seek(&mut self, pos: SeekFrom) -> OxiResult<u64> {
        let data_len = self.buffer_len() as i64;
        let current = self.position as i64;

        let new_pos = match pos {
            SeekFrom::Start(n) => n as i64,
            SeekFrom::End(n) => data_len + n,
            SeekFrom::Current(n) => current + n,
        };

        if new_pos < 0 {
            return Err(OxiError::InvalidData(
                "Seek position cannot be negative".to_string(),
            ));
        }

        self.position = new_pos as u64;
        Ok(self.position)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn len(&self) -> Option<u64> {
        Some(self.buffer_len() as u64)
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn position(&self) -> u64 {
        self.position
    }

    fn is_writable(&self) -> bool {
        self.writable
    }
}

impl Default for MemorySource {
    fn default() -> Self {
        Self::new(Bytes::new())
    }
}

impl From<Vec<u8>> for MemorySource {
    fn from(data: Vec<u8>) -> Self {
        Self::from_vec(data)
    }
}

impl From<Bytes> for MemorySource {
    fn from(data: Bytes) -> Self {
        Self::new(data)
    }
}

impl From<&[u8]> for MemorySource {
    fn from(data: &[u8]) -> Self {
        Self::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_source_new() {
        let data = Bytes::from_static(b"Hello, World!");
        let source = MemorySource::new(data);
        assert_eq!(source.len(), Some(13));
        assert_eq!(source.position(), 0);
        assert!(source.is_seekable());
    }

    #[tokio::test]
    async fn test_memory_source_read() {
        let mut source = MemorySource::from_vec(vec![1, 2, 3, 4, 5]);

        let mut buffer = [0u8; 3];
        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 3);
        assert_eq!(&buffer, &[1, 2, 3]);
        assert_eq!(source.position(), 3);

        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 2);
        assert_eq!(&buffer[..2], &[4, 5]);
        assert_eq!(source.position(), 5);

        // EOF
        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 0);
    }

    #[tokio::test]
    async fn test_memory_source_seek() {
        let mut source = MemorySource::from_slice(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // Seek from start
        let pos = source
            .seek(SeekFrom::Start(5))
            .await
            .expect("seek should succeed");
        assert_eq!(pos, 5);
        assert_eq!(source.position(), 5);

        // Read after seek
        let mut buffer = [0u8; 3];
        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 3);
        assert_eq!(&buffer, &[5, 6, 7]);

        // Seek from current
        let pos = source
            .seek(SeekFrom::Current(-3))
            .await
            .expect("seek should succeed");
        assert_eq!(pos, 5);

        // Seek from end
        let pos = source
            .seek(SeekFrom::End(-2))
            .await
            .expect("seek should succeed");
        assert_eq!(pos, 8);

        let n = source.read(&mut buffer).await.expect("failed to read");
        assert_eq!(n, 2);
        assert_eq!(&buffer[..2], &[8, 9]);
    }

    #[tokio::test]
    async fn test_memory_source_seek_negative() {
        let mut source = MemorySource::from_vec(vec![1, 2, 3]);
        let result = source.seek(SeekFrom::Start(0)).await;
        assert!(result.is_ok());

        let result = source.seek(SeekFrom::Current(-10)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_memory_source_empty() {
        let source = MemorySource::default();
        assert!(source.is_empty());
        assert_eq!(source.len(), Some(0));
    }
}
